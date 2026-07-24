"""SI 좌표계에서 사용하는 강체 변환 유틸리티."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


class FrameTransformError(ValueError):
    """좌표계 transform 또는 초기 pose 설정이 일관되지 않을 때의 오류."""


@dataclass(frozen=True)
class WokFrameContext:
    """시뮬레이션 시작 시점의 base/wok/pan 좌표계 관계."""

    T_base_wok: np.ndarray
    T_base_pan0: np.ndarray
    T_wok_pan0: np.ndarray
    source: str


_MISSING = object()


def _lookup(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _vector(value: Any, length: int, *, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise FrameTransformError(f"{name}은(는) 유한한 길이 {length} 벡터여야 합니다.") from exc
    if result.shape != (length,) or not np.isfinite(result).all():
        raise FrameTransformError(f"{name}은(는) 유한한 길이 {length} 벡터여야 합니다.")
    return result


def validate_homogeneous_transform(
    value: Any,
    name: str = "transform",
    atol: float = 1.0e-8,
) -> np.ndarray:
    """값을 유효한 SE(3) homogeneous transform으로 검증해 복사본을 반환한다."""

    try:
        tolerance = float(atol)
    except (TypeError, ValueError) as exc:
        raise FrameTransformError("atol은 유한한 음이 아닌 실수여야 합니다.") from exc
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise FrameTransformError("atol은 유한한 음이 아닌 실수여야 합니다.")

    try:
        transform = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise FrameTransformError(f"{name}은(는) 유한한 4x4 행렬이어야 합니다.") from exc
    if transform.shape != (4, 4):
        raise FrameTransformError(
            f"{name} shape은 (4, 4)여야 합니다. 현재 shape={transform.shape}."
        )
    if not np.isfinite(transform).all():
        raise FrameTransformError(f"{name}에 NaN 또는 inf가 포함되어 있습니다.")
    if not np.allclose(
        transform[3],
        np.array([0.0, 0.0, 0.0, 1.0]),
        atol=tolerance,
        rtol=0.0,
    ):
        raise FrameTransformError(f"{name}의 마지막 행은 [0, 0, 0, 1]이어야 합니다.")

    rotation = transform[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=tolerance,
        rtol=0.0,
    ):
        raise FrameTransformError(f"{name}의 회전 블록이 직교행렬이 아닙니다.")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=tolerance, rtol=0.0):
        raise FrameTransformError(
            f"{name}의 회전 블록 determinant는 +1이어야 합니다. 현재 값={determinant:.12g}."
        )
    return transform


def rpy_to_matrix(rpy_rad: np.ndarray | list[float]) -> np.ndarray:
    """고정축 xyz roll-pitch-yaw를 3x3 회전행렬로 변환한다."""

    rpy = np.asarray(rpy_rad, dtype=float)
    if rpy.shape != (3,) or not np.isfinite(rpy).all():
        raise ValueError("rpy_rad는 유한한 길이 3 벡터여야 합니다.")
    return Rotation.from_euler("xyz", rpy).as_matrix()


def matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    """3x3 회전행렬을 xyz roll-pitch-yaw로 변환한다."""

    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation은 유한한 3x3 행렬이어야 합니다.")
    return Rotation.from_matrix(matrix).as_euler("xyz")


def quaternion_to_matrix(quaternion_wxyz: np.ndarray | list[float]) -> np.ndarray:
    """MuJoCo 순서(w, x, y, z)의 quaternion을 회전행렬로 바꾼다."""

    quat = np.asarray(quaternion_wxyz, dtype=float)
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError("quaternion은 유한한 길이 4 벡터여야 합니다.")
    norm = np.linalg.norm(quat)
    if norm <= np.finfo(float).eps:
        raise ValueError("영 quaternion은 회전을 정의하지 못합니다.")
    w, x, y, z = quat / norm
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """회전행렬을 MuJoCo 순서(w, x, y, z)의 quaternion으로 바꾼다."""

    x, y, z, w = Rotation.from_matrix(np.asarray(rotation, dtype=float)).as_quat()
    quaternion = np.array([w, x, y, z], dtype=float)
    # q와 -q가 같은 회전이므로 export의 연속성을 위해 w >= 0을 택한다.
    if quaternion[0] < 0:
        quaternion *= -1.0
    return quaternion


def compose_transform(
    translation_m: np.ndarray | list[float],
    rotation: np.ndarray | None = None,
    *,
    rpy_rad: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    """translation과 rotation으로 4x4 homogeneous transform을 만든다."""

    translation = np.asarray(translation_m, dtype=float)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("translation_m은 유한한 길이 3 벡터여야 합니다.")
    if rotation is not None and rpy_rad is not None:
        raise ValueError("rotation과 rpy_rad 중 하나만 지정하세요.")
    if rpy_rad is not None:
        rotation_matrix = rpy_to_matrix(rpy_rad)
    elif rotation is None:
        rotation_matrix = np.eye(3)
    else:
        rotation_matrix = np.asarray(rotation, dtype=float)
        if rotation_matrix.shape != (3, 3) or not np.isfinite(rotation_matrix).all():
            raise ValueError("rotation은 유한한 3x3 행렬이어야 합니다.")
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """강체 4x4 transform의 역행렬을 안정적으로 계산한다."""

    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("transform은 유한한 4x4 행렬이어야 합니다.")
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse = np.eye(4, dtype=float)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """N개의 3D point에 homogeneous transform을 적용한다."""

    matrix = np.asarray(transform, dtype=float)
    xyz = np.asarray(points, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError("transform은 4x4 행렬이어야 합니다.")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("points는 (N, 3) 배열이어야 합니다.")
    return xyz @ matrix[:3, :3].T + matrix[:3, 3]


def transform_to_pose(
    transform: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SE(3) transform을 position, xyz RPY, wxyz quaternion으로 분해한다."""

    matrix = validate_homogeneous_transform(transform)
    rotation = matrix[:3, :3]
    return (
        matrix[:3, 3].copy(),
        matrix_to_rpy(rotation),
        matrix_to_quaternion(rotation),
    )


def _pose_to_transform(
    pose: Any,
    *,
    name: str,
    default_position_m: np.ndarray | list[float],
) -> np.ndarray:
    if pose is None:
        raise FrameTransformError(f"{name}이(가) 없습니다.")
    if not isinstance(pose, Mapping) and not any(
        hasattr(pose, key)
        for key in (
            "position_m",
            "position",
            "quaternion_wxyz",
            "quaternion",
            "rpy_rad",
            "orientation_rpy_rad",
        )
    ):
        raise FrameTransformError(f"{name}은(는) pose mapping이어야 합니다.")

    position_raw = _lookup(
        pose,
        "position_m",
        _lookup(pose, "position", default_position_m),
    )
    position = _vector(position_raw, 3, name=f"{name}.position_m")

    quaternion_raw = _lookup(
        pose,
        "quaternion_wxyz",
        _lookup(pose, "quaternion", _MISSING),
    )
    rpy_raw = _lookup(
        pose,
        "rpy_rad",
        _lookup(pose, "orientation_rpy_rad", _MISSING),
    )
    if quaternion_raw is None:
        quaternion_raw = _MISSING
    if rpy_raw is None:
        rpy_raw = _MISSING
    if quaternion_raw is not _MISSING:
        quaternion = _vector(
            quaternion_raw,
            4,
            name=f"{name}.quaternion_wxyz",
        )
        try:
            rotation = quaternion_to_matrix(quaternion)
        except ValueError as exc:
            raise FrameTransformError(f"{name}.quaternion_wxyz가 유효하지 않습니다.") from exc
        if rpy_raw is not _MISSING:
            try:
                rpy_rotation = rpy_to_matrix(_vector(rpy_raw, 3, name=f"{name}.rpy_rad"))
            except ValueError as exc:
                raise FrameTransformError(f"{name}.rpy_rad가 유효하지 않습니다.") from exc
            orientation_error = Rotation.from_matrix(rotation.T @ rpy_rotation).magnitude()
            if orientation_error > 1.0e-8:
                raise FrameTransformError(
                    f"{name}의 quaternion과 rpy가 서로 다른 회전을 나타냅니다."
                )
    elif rpy_raw is not _MISSING:
        try:
            rotation = rpy_to_matrix(_vector(rpy_raw, 3, name=f"{name}.rpy_rad"))
        except ValueError as exc:
            raise FrameTransformError(f"{name}.rpy_rad가 유효하지 않습니다.") from exc
    else:
        rotation = np.eye(3)

    return validate_homogeneous_transform(
        compose_transform(position, rotation),
        name=name,
    )


def _trajectory_start_transform(trajectory: Any) -> np.ndarray | None:
    position_raw = _lookup(trajectory, "start_position_m", _MISSING)
    if position_raw is _MISSING or position_raw is None:
        return None

    pose: dict[str, Any] = {"position_m": position_raw}
    rpy_raw = _lookup(trajectory, "start_rpy_rad", _MISSING)
    quaternion_raw = _lookup(
        trajectory,
        "start_quaternion_wxyz",
        _lookup(trajectory, "start_quaternion", _MISSING),
    )
    if rpy_raw is not _MISSING and rpy_raw is not None:
        pose["rpy_rad"] = rpy_raw
    if quaternion_raw is not _MISSING and quaternion_raw is not None:
        pose["quaternion_wxyz"] = quaternion_raw
    return _pose_to_transform(
        pose,
        name="trajectory.start",
        default_position_m=np.zeros(3),
    )


def _configured_tolerance(
    simulation: Any,
    *,
    key: str,
    aliases: tuple[str, ...],
) -> float:
    raw = _lookup(simulation, key, _MISSING)
    if raw is _MISSING:
        for alias in aliases:
            raw = _lookup(simulation, alias, _MISSING)
            if raw is not _MISSING:
                break
    if raw is _MISSING or raw is None:
        return 1.0e-8
    try:
        result = float(raw)
    except (TypeError, ValueError) as exc:
        raise FrameTransformError(
            f"simulation.{key}은(는) 유한한 음이 아닌 실수여야 합니다."
        ) from exc
    if not np.isfinite(result) or result < 0.0:
        raise FrameTransformError(f"simulation.{key}은(는) 유한한 음이 아닌 실수여야 합니다.")
    return result


def _assert_pose_close(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    actual_name: str,
    expected_name: str,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
) -> None:
    position_error = float(np.linalg.norm(actual[:3, 3] - expected[:3, 3]))
    orientation_error = float(Rotation.from_matrix(expected[:3, :3].T @ actual[:3, :3]).magnitude())
    if position_error > position_tolerance_m or orientation_error > orientation_tolerance_rad:
        raise FrameTransformError(
            f"{actual_name}과(와) {expected_name}이(가) 일치하지 않습니다: "
            f"position error={position_error:.12g} m "
            f"(tolerance={position_tolerance_m:.12g}), "
            f"orientation error={orientation_error:.12g} rad "
            f"(tolerance={orientation_tolerance_rad:.12g})."
        )


def resolve_wok_frame_context(config: Any) -> WokFrameContext:
    """설정에서 authoritative 초기 pan pose와 base/wok 관계를 결정한다.

    ``pan.initial_pose``는 base/world 기준, ``trajectory.start_*``는 wok 기준이다.
    teaching transform 쌍이 있으면 그 곱을 authoritative 초기 pan pose로 사용한다.
    """

    robot = _lookup(config, "robot", {}) or {}
    pan = _lookup(config, "pan", {}) or {}
    trajectory = _lookup(config, "trajectory", {}) or {}
    simulation = _lookup(config, "simulation", {}) or {}

    base_wok_raw = _lookup(robot, "T_base_wok", None)
    T_base_wok = (
        np.eye(4)
        if base_wok_raw is None
        else validate_homogeneous_transform(base_wok_raw, name="robot.T_base_wok")
    )

    base_tcp_raw = _lookup(robot, "T_base_tcp_teach", None)
    tcp_pan_raw = _lookup(robot, "T_tcp_pan", None)
    if base_tcp_raw is not None and tcp_pan_raw is None:
        raise FrameTransformError("robot.T_base_tcp_teach가 있으면 robot.T_tcp_pan도 필요합니다.")

    T_base_tcp_teach = (
        None
        if base_tcp_raw is None
        else validate_homogeneous_transform(
            base_tcp_raw,
            name="robot.T_base_tcp_teach",
        )
    )
    T_tcp_pan = (
        None
        if tcp_pan_raw is None
        else validate_homogeneous_transform(tcp_pan_raw, name="robot.T_tcp_pan")
    )

    pan_pose_raw = _lookup(pan, "initial_pose", _MISSING)
    has_pan_pose = pan_pose_raw is not _MISSING and pan_pose_raw is not None
    T_base_pan_config = (
        _pose_to_transform(
            pan_pose_raw,
            name="pan.initial_pose",
            default_position_m=[0.0, 0.0, 0.12],
        )
        if has_pan_pose
        else None
    )
    T_wok_pan_start = _trajectory_start_transform(trajectory)

    position_tolerance_m = _configured_tolerance(
        simulation,
        key="frame_position_tolerance_m",
        aliases=("frame_translation_tolerance_m", "frame_position_tolerance"),
    )
    orientation_tolerance_rad = _configured_tolerance(
        simulation,
        key="frame_orientation_tolerance_rad",
        aliases=("frame_rotation_tolerance_rad", "frame_orientation_tolerance"),
    )

    if T_base_tcp_teach is not None and T_tcp_pan is not None:
        T_base_pan0 = validate_homogeneous_transform(
            T_base_tcp_teach @ T_tcp_pan,
            name="robot.T_base_tcp_teach @ robot.T_tcp_pan",
        )
        source = "robot.teaching"
        if T_base_pan_config is not None:
            _assert_pose_close(
                T_base_pan_config,
                T_base_pan0,
                actual_name="pan.initial_pose",
                expected_name="teaching pan pose",
                position_tolerance_m=position_tolerance_m,
                orientation_tolerance_rad=orientation_tolerance_rad,
            )
    elif T_base_pan_config is not None:
        T_base_pan0 = T_base_pan_config
        source = "pan.initial_pose"
    elif T_wok_pan_start is not None:
        T_base_pan0 = validate_homogeneous_transform(
            T_base_wok @ T_wok_pan_start,
            name="robot.T_base_wok @ trajectory.start",
        )
        source = "trajectory.start_position_m"
    else:
        T_base_pan0 = compose_transform([0.0, 0.0, 0.12])
        source = "default"

    T_wok_pan0 = validate_homogeneous_transform(
        invert_transform(T_base_wok) @ T_base_pan0,
        name="T_wok_pan0",
    )
    if T_wok_pan_start is not None:
        _assert_pose_close(
            T_wok_pan_start,
            T_wok_pan0,
            actual_name="trajectory.start",
            expected_name="derived T_wok_pan0",
            position_tolerance_m=position_tolerance_m,
            orientation_tolerance_rad=orientation_tolerance_rad,
        )

    return WokFrameContext(
        T_base_wok=T_base_wok.copy(),
        T_base_pan0=T_base_pan0.copy(),
        T_wok_pan0=T_wok_pan0.copy(),
        source=source,
    )


__all__ = [
    "FrameTransformError",
    "WokFrameContext",
    "compose_transform",
    "invert_transform",
    "matrix_to_quaternion",
    "matrix_to_rpy",
    "quaternion_to_matrix",
    "resolve_wok_frame_context",
    "rpy_to_matrix",
    "transform_points",
    "transform_to_pose",
    "validate_homogeneous_transform",
]
