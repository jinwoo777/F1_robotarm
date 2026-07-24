"""Cartesian/joint trajectory를 CSV, JSON, NPZ로 저장한다."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wok_sim.geometry.transforms import (
    compose_transform,
    invert_transform,
    matrix_to_quaternion,
    validate_homogeneous_transform,
)

from .base import RobotValidationResult


@dataclass
class TrajectoryExportData:
    """export에 필요한 동일 길이의 SI trajectory 배열."""

    timestamp_s: np.ndarray
    pan_position_m: np.ndarray
    pan_rpy_rad: np.ndarray
    linear_velocity_m_s: np.ndarray
    angular_velocity_rad_s: np.ndarray
    linear_acceleration_m_s2: np.ndarray
    angular_acceleration_rad_s2: np.ndarray
    linear_jerk_m_s3: np.ndarray
    angular_jerk_rad_s3: np.ndarray

    @classmethod
    def from_trajectory(cls, trajectory: Any) -> TrajectoryExportData:
        """wok_sim Trajectory 또는 동일 attribute 계약에서 생성한다."""

        required = (
            "time_s",
            "position_m",
            "orientation_rpy_rad",
            "linear_velocity_m_s",
            "angular_velocity_rad_s",
            "linear_acceleration_m_s2",
            "angular_acceleration_rad_s2",
            "linear_jerk_m_s3",
            "angular_jerk_rad_s3",
        )
        missing = [name for name in required if not hasattr(trajectory, name)]
        if missing:
            raise ValueError(f"trajectory 필수 attribute가 없습니다: {', '.join(missing)}")
        return cls(
            timestamp_s=np.asarray(trajectory.time_s, dtype=float),
            pan_position_m=np.asarray(trajectory.position_m, dtype=float),
            pan_rpy_rad=np.asarray(trajectory.orientation_rpy_rad, dtype=float),
            linear_velocity_m_s=np.asarray(trajectory.linear_velocity_m_s, dtype=float),
            angular_velocity_rad_s=np.asarray(trajectory.angular_velocity_rad_s, dtype=float),
            linear_acceleration_m_s2=np.asarray(trajectory.linear_acceleration_m_s2, dtype=float),
            angular_acceleration_rad_s2=np.asarray(
                trajectory.angular_acceleration_rad_s2, dtype=float
            ),
            linear_jerk_m_s3=np.asarray(trajectory.linear_jerk_m_s3, dtype=float),
            angular_jerk_rad_s3=np.asarray(trajectory.angular_jerk_rad_s3, dtype=float),
        )

    def validate(self) -> None:
        """shape, finite, monotonic time을 검사한다."""

        time = np.asarray(self.timestamp_s, dtype=float)
        if (
            time.ndim != 1
            or len(time) < 2
            or not np.isfinite(time).all()
            or np.any(np.diff(time) <= 0)
        ):
            raise ValueError(
                "timestamp_s는 2개 이상의 유한하고 엄격히 증가하는 1D 배열이어야 합니다."
            )
        for name in (
            "pan_position_m",
            "pan_rpy_rad",
            "linear_velocity_m_s",
            "angular_velocity_rad_s",
            "linear_acceleration_m_s2",
            "angular_acceleration_rad_s2",
            "linear_jerk_m_s3",
            "angular_jerk_rad_s3",
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (len(time), 3):
                raise ValueError(f"{name} shape은 {(len(time), 3)}이어야 합니다: {value.shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name}에 NaN 또는 inf가 있습니다.")


def _pose_transforms(data: TrajectoryExportData) -> np.ndarray:
    return np.stack(
        [
            compose_transform(position, rpy_rad=rpy)
            for position, rpy in zip(data.pan_position_m, data.pan_rpy_rad, strict=True)
        ]
    )


@dataclass(frozen=True)
class _FrameDerivatives:
    """한 rigid frame 원점의 world-frame 선형·각 미분."""

    linear_velocity_m_s: np.ndarray
    angular_velocity_rad_s: np.ndarray
    linear_acceleration_m_s2: np.ndarray
    angular_acceleration_rad_s2: np.ndarray
    linear_jerk_m_s3: np.ndarray
    angular_jerk_rad_s3: np.ndarray


def _pan_derivatives(data: TrajectoryExportData) -> _FrameDerivatives:
    """입력 trajectory의 pan-frame 원점 미분을 명시적 구조로 묶는다."""

    return _FrameDerivatives(
        linear_velocity_m_s=np.asarray(data.linear_velocity_m_s, dtype=float),
        angular_velocity_rad_s=np.asarray(data.angular_velocity_rad_s, dtype=float),
        linear_acceleration_m_s2=np.asarray(data.linear_acceleration_m_s2, dtype=float),
        angular_acceleration_rad_s2=np.asarray(data.angular_acceleration_rad_s2, dtype=float),
        linear_jerk_m_s3=np.asarray(data.linear_jerk_m_s3, dtype=float),
        angular_jerk_rad_s3=np.asarray(data.angular_jerk_rad_s3, dtype=float),
    )


def _tcp_derivatives(
    data: TrajectoryExportData,
    pan_transforms: np.ndarray,
    tcp_transforms: np.ndarray,
) -> _FrameDerivatives:
    """고정 rigid offset으로 TCP 원점의 world-frame 미분을 계산한다.

    ``r``을 pan 원점에서 TCP 원점으로 향하는 world-frame 벡터라 하면
    ``r_dot = omega × r``이고, 이를 두 번 더 미분해 선형
    velocity/acceleration/jerk의 offset 항을 얻는다. TCP와 pan은 고정
    transform으로 연결되므로 angular velocity/acceleration/jerk는 같다.
    """

    pan = _pan_derivatives(data)
    offset_world_m = np.asarray(tcp_transforms[:, :3, 3], dtype=float) - np.asarray(
        pan_transforms[:, :3, 3], dtype=float
    )
    offset_velocity_m_s = np.cross(pan.angular_velocity_rad_s, offset_world_m)
    offset_acceleration_m_s2 = np.cross(pan.angular_acceleration_rad_s2, offset_world_m) + np.cross(
        pan.angular_velocity_rad_s, offset_velocity_m_s
    )
    offset_jerk_m_s3 = (
        np.cross(pan.angular_jerk_rad_s3, offset_world_m)
        + 2.0 * np.cross(pan.angular_acceleration_rad_s2, offset_velocity_m_s)
        + np.cross(pan.angular_velocity_rad_s, offset_acceleration_m_s2)
    )
    return _FrameDerivatives(
        linear_velocity_m_s=pan.linear_velocity_m_s + offset_velocity_m_s,
        angular_velocity_rad_s=pan.angular_velocity_rad_s.copy(),
        linear_acceleration_m_s2=(pan.linear_acceleration_m_s2 + offset_acceleration_m_s2),
        angular_acceleration_rad_s2=pan.angular_acceleration_rad_s2.copy(),
        linear_jerk_m_s3=pan.linear_jerk_m_s3 + offset_jerk_m_s3,
        angular_jerk_rad_s3=pan.angular_jerk_rad_s3.copy(),
    )


def _add_vector_columns(
    columns: dict[str, np.ndarray],
    prefix: str,
    values: np.ndarray,
) -> None:
    """(N, 3) vector를 x/y/z scalar 열로 펼친다."""

    array = np.asarray(values, dtype=float)
    for index, axis in enumerate("xyz"):
        columns[f"{prefix}_{axis}"] = array[:, index]


def _dataframe(
    data: TrajectoryExportData,
    T_tcp_pan: np.ndarray | None,
    robot_result: RobotValidationResult | None,
) -> tuple[pd.DataFrame, str]:
    data.validate()
    pan_transforms = _pose_transforms(data)
    if T_tcp_pan is None:
        # 시뮬레이션 export를 완결하기 위한 명시적 identity 가정이며 실제 TCP가 아니다.
        tcp_transforms = pan_transforms.copy()
        tcp_assumption = "simulation_identity_not_robot_validated"
    else:
        tcp_pan = validate_homogeneous_transform(
            T_tcp_pan,
            name="T_tcp_pan",
        )
        inverse = invert_transform(tcp_pan)
        tcp_transforms = np.einsum("tij,jk->tik", pan_transforms, inverse)
        tcp_assumption = "configured_T_tcp_pan"
    pan_quat = np.stack([matrix_to_quaternion(item[:3, :3]) for item in pan_transforms])
    tcp_quat = np.stack([matrix_to_quaternion(item[:3, :3]) for item in tcp_transforms])
    pan_derivatives = _pan_derivatives(data)
    tcp_derivatives = _tcp_derivatives(data, pan_transforms, tcp_transforms)

    columns: dict[str, np.ndarray] = {"timestamp_s": np.asarray(data.timestamp_s)}
    _add_vector_columns(columns, "pan_position_m", data.pan_position_m)
    for prefix, array in (
        ("pan_quaternion_wxyz", pan_quat),
        ("tcp_quaternion_wxyz", tcp_quat),
    ):
        for index, axis in enumerate("wxyz"):
            columns[f"{prefix}_{axis}"] = np.asarray(array)[:, index]
    _add_vector_columns(columns, "tcp_position_m", tcp_transforms[:, :3, 3])

    # 기존 generic 열은 pan-frame 원점 미분으로 유지해 이전 consumer를 깨지 않는다.
    for prefix, array in (
        ("linear_velocity_m_s", pan_derivatives.linear_velocity_m_s),
        ("angular_velocity_rad_s", pan_derivatives.angular_velocity_rad_s),
        ("linear_acceleration_m_s2", pan_derivatives.linear_acceleration_m_s2),
        ("angular_acceleration_rad_s2", pan_derivatives.angular_acceleration_rad_s2),
        ("linear_jerk_m_s3", pan_derivatives.linear_jerk_m_s3),
        ("angular_jerk_rad_s3", pan_derivatives.angular_jerk_rad_s3),
        ("pan_linear_velocity_m_s", pan_derivatives.linear_velocity_m_s),
        ("pan_angular_velocity_rad_s", pan_derivatives.angular_velocity_rad_s),
        ("pan_linear_acceleration_m_s2", pan_derivatives.linear_acceleration_m_s2),
        (
            "pan_angular_acceleration_rad_s2",
            pan_derivatives.angular_acceleration_rad_s2,
        ),
        ("pan_linear_jerk_m_s3", pan_derivatives.linear_jerk_m_s3),
        ("pan_angular_jerk_rad_s3", pan_derivatives.angular_jerk_rad_s3),
        ("tcp_linear_velocity_m_s", tcp_derivatives.linear_velocity_m_s),
        ("tcp_angular_velocity_rad_s", tcp_derivatives.angular_velocity_rad_s),
        ("tcp_linear_acceleration_m_s2", tcp_derivatives.linear_acceleration_m_s2),
        (
            "tcp_angular_acceleration_rad_s2",
            tcp_derivatives.angular_acceleration_rad_s2,
        ),
        ("tcp_linear_jerk_m_s3", tcp_derivatives.linear_jerk_m_s3),
        ("tcp_angular_jerk_rad_s3", tcp_derivatives.angular_jerk_rad_s3),
    ):
        _add_vector_columns(columns, prefix, array)

    if robot_result is not None:
        for prefix, array in (
            ("joint_position_rad", robot_result.q),
            ("joint_velocity_rad_s", robot_result.qd),
            ("joint_acceleration_rad_s2", robot_result.qdd),
        ):
            if array is None:
                continue
            values = np.asarray(array, dtype=float)
            if values.shape[0] != len(data.timestamp_s):
                raise ValueError(f"{prefix}의 sample 수가 Cartesian trajectory와 다릅니다.")
            for joint in range(values.shape[1]):
                columns[f"{prefix}_j{joint + 1}"] = values[:, joint]
    frame = pd.DataFrame(columns)
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("export dataframe에 NaN 또는 inf가 있습니다.")
    return frame, tcp_assumption


def export_trajectory(
    data: TrajectoryExportData,
    output_path: str | Path,
    *,
    T_tcp_pan: np.ndarray | None = None,
    robot_result: RobotValidationResult | None = None,
) -> Path:
    """확장자에 따라 CSV, JSON 또는 NPZ를 원자적으로 저장한다."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame, tcp_assumption = _dataframe(data, T_tcp_pan, robot_result)
    suffix = path.suffix.lower()
    temporary = path.with_name(f".{path.name}.tmp")
    if suffix == ".csv":
        frame.to_csv(temporary, index=False)
    elif suffix == ".json":
        payload = {
            "metadata": {
                "tcp_transform_source": tcp_assumption,
                "robot_validation": (
                    robot_result.summary()
                    if robot_result is not None
                    else RobotValidationResult.not_evaluated(
                        "robot validator가 실행되지 않았습니다."
                    ).summary()
                ),
            },
            "trajectory": frame.to_dict(orient="records"),
        }
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    elif suffix == ".npz":
        # np.savez는 문자열 경로에 .npz를 덧붙일 수 있어 열린 file handle을 사용한다.
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                **{column: frame[column].to_numpy() for column in frame.columns},
                tcp_transform_source=np.asarray(tcp_assumption),
                robot_validation_status=np.asarray(
                    robot_result.status if robot_result is not None else "not_evaluated"
                ),
            )
    else:
        raise ValueError("output 확장자는 .csv, .json, .npz 중 하나여야 합니다.")
    temporary.replace(path)
    return path
