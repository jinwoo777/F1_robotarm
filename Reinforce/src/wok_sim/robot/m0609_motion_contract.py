"""M0609용 비실행 Cartesian motion contract와 오프라인 export.

이 모듈은 Doosan/ROS 모듈을 import하지 않고 어떤 로봇 명령도 실행하지
않는다. 입력 trajectory의 TCP Cartesian 미분을 provisional limit와
비교하고, 실제 controller integration 전에 사람이 검토할 parameter
data만 만든다.
"""

from __future__ import annotations

import json
import math
import pprint
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from wok_sim.geometry.transforms import (
    compose_transform,
    invert_transform,
    matrix_to_rpy,
    validate_homogeneous_transform,
)

from .exporter import TrajectoryExportData

SAFETY_STATUS = "unverified_without_urdf_teaching"
SCHEMA_VERSION = "wok_sim.doosan_offline_motion_plan/v1"
DOOSAN_MOVESX_MAX_WAYPOINTS = 100

# 사용자가 제공한 M0609 nominal reference다. 아래 값만으로 trajectory의
# reachability, singularity, joint limit, collision 또는 controller tracking을
# 보증할 수 없다.
M0609_REFERENCE_SPEC: dict[str, Any] = {
    "model": "M0609",
    "nominal_payload_kg": 6.0,
    "nominal_reach_m": 0.9,
    "approximate_max_tcp_linear_speed_m_s": 1.0,
    "reference_scope": "nominal_model_reference_not_runtime_verified",
}


class M0609MotionContractError(ValueError):
    """오프라인 motion contract 입력 또는 한계 검사가 실패했을 때 발생한다."""


@dataclass(frozen=True, slots=True)
class M0609CartesianCaps:
    """M0609 후보 동작에 적용할 보수적인 provisional Cartesian caps.

    linear speed 기본값은 사용자가 제공한 약 1 m/s nominal maximum의
    25%다. angular/acceleration/jerk 기본값은 제조사 한계가 아니라 초기
    시뮬레이션용 engineering caps이며 실제 장비 parameter로 교체해야 한다.
    """

    linear_velocity_m_s: float = 0.25
    angular_velocity_rad_s: float = math.radians(30.0)
    linear_acceleration_m_s2: float = 0.50
    angular_acceleration_rad_s2: float = math.radians(60.0)
    linear_jerk_m_s3: float = 2.0
    angular_jerk_rad_s3: float = math.radians(240.0)

    def __post_init__(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if not math.isfinite(value) or value <= 0.0:
                raise M0609MotionContractError(f"{name}은 유한한 양수여야 합니다.")
        nominal_speed = float(M0609_REFERENCE_SPEC["approximate_max_tcp_linear_speed_m_s"])
        if self.linear_velocity_m_s > nominal_speed:
            raise M0609MotionContractError(
                "linear_velocity_m_s provisional cap이 제공된 M0609 nominal "
                f"TCP speed reference({nominal_speed:g} m/s)를 초과합니다."
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> M0609CartesianCaps:
        """config mapping에서 명시적인 cap set을 만든다."""

        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise M0609MotionContractError(f"알 수 없는 Cartesian cap key: {', '.join(unknown)}")
        try:
            return cls(**{key: float(value) for key, value in values.items()})
        except (TypeError, ValueError) as exc:
            raise M0609MotionContractError("Cartesian caps는 숫자여야 합니다.") from exc

    def summary(self) -> dict[str, float | str]:
        """logging/export용 SI cap mapping을 반환한다."""

        return {
            **asdict(self),
            "source": "provisional_engineering_caps_not_manufacturer_limits",
        }


@dataclass(frozen=True, slots=True)
class M0609MotionViolation:
    """하나의 Cartesian cap 위반."""

    quantity: str
    observed: float
    limit: float
    timestamp_s: float
    sample_index: int
    units: str

    def summary(self) -> dict[str, str | float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class M0609MotionReport:
    """strict Cartesian cap validation 결과."""

    status: str
    safety_status: str
    within_caps: bool
    sample_count: int
    duration_s: float
    tcp_transform_source: str
    peaks: dict[str, float]
    required_uniform_time_scale: float | None
    caps: M0609CartesianCaps
    violations: tuple[M0609MotionViolation, ...]
    payload_kg: float | None
    payload_within_nominal_reference: bool | None

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "safety_status": self.safety_status,
            "within_caps": self.within_caps,
            "sample_count": self.sample_count,
            "duration_s": self.duration_s,
            "tcp_transform_source": self.tcp_transform_source,
            "peaks": dict(self.peaks),
            "required_uniform_time_scale": self.required_uniform_time_scale,
            "caps": self.caps.summary(),
            "violations": [item.summary() for item in self.violations],
            "payload_kg": self.payload_kg,
            "payload_within_nominal_reference": self.payload_within_nominal_reference,
        }

    def require_within_caps(self) -> None:
        """위반 trajectory의 offline controller export를 차단한다."""

        if self.within_caps:
            return
        details = ", ".join(
            f"{item.quantity}={item.observed:.6g}>{item.limit:.6g} {item.units}"
            for item in self.violations
        )
        raise M0609MotionContractError(
            "M0609 provisional Cartesian motion contract 위반: " + details
        )


@dataclass(frozen=True, slots=True)
class _TCPMotionSamples:
    """base/world 표현 TCP pose와 미분."""

    time_s: np.ndarray
    position_m: np.ndarray
    rpy_rad: np.ndarray
    linear_velocity_m_s: np.ndarray
    angular_velocity_rad_s: np.ndarray
    linear_acceleration_m_s2: np.ndarray
    angular_acceleration_rad_s2: np.ndarray
    linear_jerk_m_s3: np.ndarray
    angular_jerk_rad_s3: np.ndarray
    transform_source: str


def _as_export_data(trajectory: Any) -> TrajectoryExportData:
    if isinstance(trajectory, TrajectoryExportData):
        data = trajectory
    else:
        data = TrajectoryExportData.from_trajectory(trajectory)
    data.validate()
    return data


def _tcp_motion_samples(
    trajectory: Any,
    T_tcp_pan: np.ndarray | None,
) -> _TCPMotionSamples:
    data = _as_export_data(trajectory)
    pan_transforms = np.stack(
        [
            compose_transform(position, rpy_rad=rpy)
            for position, rpy in zip(data.pan_position_m, data.pan_rpy_rad, strict=True)
        ]
    )
    if T_tcp_pan is None:
        tcp_transforms = pan_transforms.copy()
        transform_source = "pan_pose_identity_placeholder"
    else:
        tcp_pan = validate_homogeneous_transform(T_tcp_pan, name="T_tcp_pan")
        tcp_transforms = np.einsum(
            "tij,jk->tik",
            pan_transforms,
            invert_transform(tcp_pan),
        )
        transform_source = "configured_T_tcp_pan"

    rpy = np.vstack([matrix_to_rpy(transform[:3, :3]) for transform in tcp_transforms])
    omega = np.asarray(data.angular_velocity_rad_s, dtype=float)
    alpha = np.asarray(data.angular_acceleration_rad_s2, dtype=float)
    angular_jerk = np.asarray(data.angular_jerk_rad_s3, dtype=float)
    offset = tcp_transforms[:, :3, 3] - pan_transforms[:, :3, 3]
    offset_velocity = np.cross(omega, offset)
    offset_acceleration = np.cross(alpha, offset) + np.cross(omega, offset_velocity)
    offset_jerk = (
        np.cross(angular_jerk, offset)
        + 2.0 * np.cross(alpha, offset_velocity)
        + np.cross(omega, offset_acceleration)
    )
    return _TCPMotionSamples(
        time_s=np.asarray(data.timestamp_s, dtype=float),
        position_m=tcp_transforms[:, :3, 3],
        rpy_rad=rpy,
        linear_velocity_m_s=np.asarray(data.linear_velocity_m_s, dtype=float) + offset_velocity,
        angular_velocity_rad_s=omega,
        linear_acceleration_m_s2=(
            np.asarray(data.linear_acceleration_m_s2, dtype=float) + offset_acceleration
        ),
        angular_acceleration_rad_s2=alpha,
        linear_jerk_m_s3=np.asarray(data.linear_jerk_m_s3, dtype=float) + offset_jerk,
        angular_jerk_rad_s3=angular_jerk,
        transform_source=transform_source,
    )


def _validated_payload(payload_kg: float | None) -> tuple[float | None, bool | None]:
    if payload_kg is None:
        return None, None
    try:
        payload = float(payload_kg)
    except (TypeError, ValueError) as exc:
        raise M0609MotionContractError("payload_kg는 유한한 음이 아닌 값이어야 합니다.") from exc
    if not math.isfinite(payload) or payload < 0.0:
        raise M0609MotionContractError("payload_kg는 유한한 음이 아닌 값이어야 합니다.")
    return payload, payload <= float(M0609_REFERENCE_SPEC["nominal_payload_kg"])


def validate_m0609_cartesian_motion(
    trajectory: Any,
    *,
    caps: M0609CartesianCaps | Mapping[str, Any] | None = None,
    T_tcp_pan: np.ndarray | None = None,
    payload_kg: float | None = None,
) -> M0609MotionReport:
    """TCP speed/acceleration/jerk를 모든 sample에서 strict cap과 비교한다.

    이 검사는 Cartesian 숫자 한계만 평가한다. 통과해도 M0609 joint-space
    reachability 또는 실제 운전 안전성을 뜻하지 않는다.
    """

    if caps is None:
        resolved_caps = M0609CartesianCaps()
    elif isinstance(caps, M0609CartesianCaps):
        resolved_caps = caps
    elif isinstance(caps, Mapping):
        resolved_caps = M0609CartesianCaps.from_mapping(caps)
    else:
        raise M0609MotionContractError("caps는 M0609CartesianCaps 또는 mapping이어야 합니다.")

    samples = _tcp_motion_samples(trajectory, T_tcp_pan)
    payload, payload_ok = _validated_payload(payload_kg)
    quantities = (
        (
            "tcp_linear_velocity",
            samples.linear_velocity_m_s,
            resolved_caps.linear_velocity_m_s,
            "m/s",
            1,
        ),
        (
            "tcp_angular_velocity",
            samples.angular_velocity_rad_s,
            resolved_caps.angular_velocity_rad_s,
            "rad/s",
            1,
        ),
        (
            "tcp_linear_acceleration",
            samples.linear_acceleration_m_s2,
            resolved_caps.linear_acceleration_m_s2,
            "m/s^2",
            2,
        ),
        (
            "tcp_angular_acceleration",
            samples.angular_acceleration_rad_s2,
            resolved_caps.angular_acceleration_rad_s2,
            "rad/s^2",
            2,
        ),
        (
            "tcp_linear_jerk",
            samples.linear_jerk_m_s3,
            resolved_caps.linear_jerk_m_s3,
            "m/s^3",
            3,
        ),
        (
            "tcp_angular_jerk",
            samples.angular_jerk_rad_s3,
            resolved_caps.angular_jerk_rad_s3,
            "rad/s^3",
            3,
        ),
    )
    peaks: dict[str, float] = {}
    violations: list[M0609MotionViolation] = []
    time_scale_candidates = [1.0]
    for name, vectors, limit, units, derivative_order in quantities:
        norms = np.linalg.norm(np.asarray(vectors, dtype=float), axis=1)
        sample_index = int(np.argmax(norms))
        peak = float(norms[sample_index])
        peaks[name] = peak
        time_scale_candidates.append((peak / limit) ** (1.0 / derivative_order))
        # 의도적으로 tolerance를 더하지 않는다. cap은 hard pre-export gate다.
        if peak > limit:
            violations.append(
                M0609MotionViolation(
                    quantity=name,
                    observed=peak,
                    limit=limit,
                    timestamp_s=float(samples.time_s[sample_index]),
                    sample_index=sample_index,
                    units=units,
                )
            )
    if payload_ok is False:
        violations.append(
            M0609MotionViolation(
                quantity="payload",
                observed=float(payload),
                limit=float(M0609_REFERENCE_SPEC["nominal_payload_kg"]),
                timestamp_s=float(samples.time_s[0]),
                sample_index=0,
                units="kg",
            )
        )

    within_caps = not violations
    return M0609MotionReport(
        status=(
            "within_provisional_cartesian_caps"
            if within_caps
            else "invalid_cartesian_motion_contract"
        ),
        safety_status=SAFETY_STATUS,
        within_caps=within_caps,
        sample_count=len(samples.time_s),
        duration_s=float(samples.time_s[-1] - samples.time_s[0]),
        tcp_transform_source=samples.transform_source,
        peaks=peaks,
        required_uniform_time_scale=(
            float(max(time_scale_candidates)) if payload_ok is not False else None
        ),
        caps=resolved_caps,
        violations=tuple(violations),
        payload_kg=payload,
        payload_within_nominal_reference=payload_ok,
    )


def _waypoint_indices(
    sample_count: int,
    stride: int | None,
    *,
    maximum_count: int | None,
) -> tuple[np.ndarray, int, str]:
    if stride is None:
        resolved_stride = (
            1
            if maximum_count is None
            else max(1, math.ceil((sample_count - 1) / (maximum_count - 1)))
        )
        stride_source = "automatic_controller_waypoint_limit"
    else:
        if isinstance(stride, bool) or not isinstance(stride, (int, np.integer)):
            raise M0609MotionContractError("waypoint_stride는 양의 정수 또는 None이어야 합니다.")
        resolved_stride = int(stride)
        stride_source = "explicit"
    try:
        valid_stride = resolved_stride > 0
    except TypeError as exc:
        raise M0609MotionContractError(
            "waypoint_stride는 양의 정수 또는 None이어야 합니다."
        ) from exc
    if not valid_stride:
        raise M0609MotionContractError("waypoint_stride는 양의 정수 또는 None이어야 합니다.")
    indices = np.arange(0, sample_count, resolved_stride, dtype=int)
    if indices[-1] != sample_count - 1:
        indices = np.append(indices, sample_count - 1)
    if maximum_count is not None and len(indices) > maximum_count:
        raise M0609MotionContractError(
            f"선택된 waypoint {len(indices)}개가 movesx 단일 명령 한계 "
            f"{maximum_count}개를 초과합니다. waypoint_stride를 늘리세요."
        )
    return indices, resolved_stride, stride_source


def build_doosan_offline_plan(
    trajectory: Any,
    *,
    caps: M0609CartesianCaps | Mapping[str, Any] | None = None,
    T_tcp_pan: np.ndarray | None = None,
    payload_kg: float | None = None,
    function: Literal["movesx", "movel"] = "movesx",
    waypoint_stride: int | None = None,
) -> dict[str, Any]:
    """strict validation을 통과한 비실행 Doosan parameter data를 만든다."""

    if function not in {"movesx", "movel"}:
        raise M0609MotionContractError("function은 'movesx' 또는 'movel'이어야 합니다.")
    samples = _tcp_motion_samples(trajectory, T_tcp_pan)
    report = validate_m0609_cartesian_motion(
        trajectory,
        caps=caps,
        T_tcp_pan=T_tcp_pan,
        payload_kg=payload_kg,
    )
    report.require_within_caps()
    indices, resolved_stride, stride_source = _waypoint_indices(
        len(samples.time_s),
        waypoint_stride,
        maximum_count=DOOSAN_MOVESX_MAX_WAYPOINTS if function == "movesx" else None,
    )
    poses_mm_deg = np.column_stack(
        (
            samples.position_m * 1000.0,
            np.rad2deg(samples.rpy_rad),
        )
    )[indices]
    timestamps_s = samples.time_s[indices]
    waypoints = [
        {
            "index": int(index),
            "timestamp_s": float(timestamp),
            "posx_mm_deg": [float(value) for value in pose],
        }
        for index, timestamp, pose in zip(indices, timestamps_s, poses_mm_deg, strict=True)
    ]
    resolved_caps = report.caps
    common_parameters: dict[str, Any] = {
        "vel": [
            resolved_caps.linear_velocity_m_s * 1000.0,
            math.degrees(resolved_caps.angular_velocity_rad_s),
        ],
        "acc": [
            resolved_caps.linear_acceleration_m_s2 * 1000.0,
            math.degrees(resolved_caps.angular_acceleration_rad_s2),
        ],
        "ref": "DR_BASE",
        "mod": "DR_MV_MOD_ABS",
    }
    if function == "movesx":
        function_parameters: dict[str, Any] = {
            "pos_list": [item["posx_mm_deg"] for item in waypoints],
            **common_parameters,
            "time": None,
            "vel_opt": "DR_MVS_VEL_NONE",
        }
    else:
        function_parameters = {
            "segments": [
                {
                    "reference_segment_duration_s": float(
                        timestamps_s[index] - timestamps_s[index - 1]
                    ),
                    "parameters": {
                        "pos": item["posx_mm_deg"],
                        **common_parameters,
                        # 공식 controller semantics상 time을 지정하면 vel/acc가
                        # 무시되므로 cap-driven template에서는 None으로 둔다.
                        "time": None,
                        "radius": 0.0,
                        "ra": "DR_MV_RA_DUPLICATE",
                        "app_type": "DR_MV_APP_NONE",
                    },
                }
                for index, item in enumerate(waypoints)
                if index > 0
            ]
        }

    missing_transform_reason = (
        ["T_tcp_pan_missing_pan_pose_used_as_identity_placeholder"] if T_tcp_pan is None else []
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "plan_type": "non_executable_offline_parameter_template",
        "executable": False,
        "function_name": function,
        "robot_model_reference": dict(M0609_REFERENCE_SPEC),
        "safety": {
            "status": SAFETY_STATUS,
            "controller_revalidation_required": True,
            "command_ready": False,
            "reasons": [
                "URDF_and_teaching_not_evaluated",
                "joint_limits_singularity_and_collision_not_evaluated",
                "controller_interpolation_can_change_sampled_kinematics",
                "jerk_is_validated_offline_but_not_a_Doosan_function_parameter",
                "pos_list_requires_posx_construction_with_DR_FIX_XYZ",
                (
                    "movesx_orientation_change_is_not_recommended_by_controller_manual"
                    if function == "movesx"
                    else "movel_segment_timing_is_reference_only"
                ),
                *missing_transform_reason,
            ],
        },
        "coordinate_convention": {
            "pose": "[x_mm, y_mm, z_mm, A_deg, B_deg, C_deg]",
            "pose_frame": "base/world candidate; verify against controller DR_BASE",
            "orientation_type": "DR_FIX_XYZ",
            "tcp_transform_source": samples.transform_source,
        },
        "position_constructor_template": {
            "name": "posx",
            "ori_type": "DR_FIX_XYZ",
            "instruction": "construct each pos_list item as posx(values, ori_type=DR_FIX_XYZ)",
        },
        "motion_contract": report.summary(),
        "reference_timing": {
            "timestamps_s": [float(value) for value in timestamps_s],
            "controller_timing_reproduction_guaranteed": False,
        },
        "waypoint_selection": {
            "source_sample_count": len(samples.time_s),
            "exported_waypoint_count": len(indices),
            "stride": resolved_stride,
            "stride_source": stride_source,
            "single_movesx_waypoint_limit": (
                DOOSAN_MOVESX_MAX_WAYPOINTS if function == "movesx" else None
            ),
        },
        "function_parameters": function_parameters,
        "waypoints": waypoints,
    }


def _csv_frame(plan: Mapping[str, Any]) -> pd.DataFrame:
    contract = plan["motion_contract"]
    caps = contract["caps"]
    safety = plan["safety"]
    records: list[dict[str, Any]] = []
    for waypoint in plan["waypoints"]:
        pose = waypoint["posx_mm_deg"]
        records.append(
            {
                "schema_version": plan["schema_version"],
                "executable": plan["executable"],
                "function_name": plan["function_name"],
                "safety_status": safety["status"],
                "controller_revalidation_required": safety["controller_revalidation_required"],
                "tcp_transform_source": plan["coordinate_convention"]["tcp_transform_source"],
                "waypoint_index": waypoint["index"],
                "timestamp_s": waypoint["timestamp_s"],
                "x_mm": pose[0],
                "y_mm": pose[1],
                "z_mm": pose[2],
                "a_deg": pose[3],
                "b_deg": pose[4],
                "c_deg": pose[5],
                "orientation_type": plan["coordinate_convention"]["orientation_type"],
                "linear_velocity_cap_mm_s": caps["linear_velocity_m_s"] * 1000.0,
                "angular_velocity_cap_deg_s": math.degrees(caps["angular_velocity_rad_s"]),
                "linear_acceleration_cap_mm_s2": caps["linear_acceleration_m_s2"] * 1000.0,
                "angular_acceleration_cap_deg_s2": math.degrees(
                    caps["angular_acceleration_rad_s2"]
                ),
                "linear_jerk_cap_mm_s3": caps["linear_jerk_m_s3"] * 1000.0,
                "angular_jerk_cap_deg_s3": math.degrees(caps["angular_jerk_rad_s3"]),
            }
        )
    return pd.DataFrame.from_records(records)


def export_doosan_offline_plan(
    trajectory: Any,
    output_path: str | Path,
    **plan_kwargs: Any,
) -> Path:
    """JSON/CSV/Python data template로 원자적으로 저장한다.

    Python 형식도 dictionary assignment만 포함하며 API import나 함수 호출은
    만들지 않는다.
    """

    plan = build_doosan_offline_plan(trajectory, **plan_kwargs)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    suffix = path.suffix.lower()
    if suffix == ".json":
        temporary.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    elif suffix == ".csv":
        _csv_frame(plan).to_csv(temporary, index=False)
    elif suffix == ".py":
        source = (
            "# Generated offline parameter data. It does not execute a robot command.\n"
            f"# safety_status: {SAFETY_STATUS}\n"
            "DOOSAN_OFFLINE_PLAN = " + pprint.pformat(plan, sort_dicts=False, width=100) + "\n"
        )
        temporary.write_text(source, encoding="utf-8")
    else:
        raise M0609MotionContractError("output 확장자는 .json, .csv, .py 중 하나여야 합니다.")
    temporary.replace(path)
    return path
