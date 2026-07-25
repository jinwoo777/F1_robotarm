"""M0609 nominal joint/TCP speed envelope for offline wok trajectory retiming.

The kinematic chain and nominal joint velocities are pinned to the official
Doosan ROS 2 description at commit ``ec9242546ec6202835900dbcd8498e2daabfa6a6``.
This remains an offline speed check: an assumed teaching pose cannot replace
the measured robot pose, controller acceleration settings, payload dynamics,
collision checking, or a real controller dry-run.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from wok_sim.geometry.transforms import compose_transform, invert_transform

M0609_NOMINAL_JOINT_VELOCITY_LIMITS_RAD_S = np.deg2rad(
    np.asarray([150.0, 150.0, 180.0, 225.0, 225.0, 225.0], dtype=float)
)
M0609_DEFAULT_JOINT_POSITION_LOWER_RAD = np.deg2rad(
    np.asarray([-360.0, -95.0, -135.0, -360.0, -135.0, -360.0], dtype=float)
)
M0609_DEFAULT_JOINT_POSITION_UPPER_RAD = -M0609_DEFAULT_JOINT_POSITION_LOWER_RAD
M0609_NOMINAL_TCP_LINEAR_SPEED_M_S = 1.0
M0609_CONSERVATIVE_TCP_ANGULAR_SPEED_RAD_S = float(
    M0609_NOMINAL_JOINT_VELOCITY_LIMITS_RAD_S.min()
)


class M0609JointSpeedError(ValueError):
    """Raised when the nominal M0609 speed-only kinematic check cannot run."""


def _six_vector(value: Any, *, name: str, positive: bool = False) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise M0609JointSpeedError(f"{name}은 유한한 길이 6 벡터여야 합니다.") from exc
    if result.shape != (6,) or not np.isfinite(result).all():
        raise M0609JointSpeedError(f"{name}은 유한한 길이 6 벡터여야 합니다.")
    if positive and np.any(result <= 0.0):
        raise M0609JointSpeedError(f"{name}은 모두 양수여야 합니다.")
    return result


def _three_vector(value: Any, *, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise M0609JointSpeedError(f"{name}은 유한한 길이 3 벡터여야 합니다.") from exc
    if result.shape != (3,) or not np.isfinite(result).all():
        raise M0609JointSpeedError(f"{name}은 유한한 길이 3 벡터여야 합니다.")
    return result


@dataclass(frozen=True, slots=True)
class M0609JointSpeedSettings:
    """Settings for a nominal speed-only M0609 sequential-IK check."""

    target_fraction: float
    tcp_offset_m: float
    tcp_offset_axis: tuple[float, float, float]
    q_teach_rad: tuple[float, float, float, float, float, float]
    joint_velocity_limits_rad_s: tuple[float, float, float, float, float, float]
    joint_position_lower_rad: tuple[float, float, float, float, float, float]
    joint_position_upper_rad: tuple[float, float, float, float, float, float]
    tcp_linear_velocity_limit_m_s: float
    tcp_angular_velocity_limit_rad_s: float
    minimum_time_scale: float
    maximum_time_scale: float
    ik_tolerance: float = 2.0e-7
    ik_damping: float = 1.0e-7
    ik_step_size: float = 0.75
    ik_max_iterations: int = 30

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> M0609JointSpeedSettings:
        """Validate a config mapping and construct immutable settings."""

        try:
            target_fraction = float(values.get("target_fraction", 0.90))
            tcp_offset_m = float(values.get("tcp_offset_m", 0.40))
            minimum_time_scale = float(values.get("minimum_time_scale", 0.05))
            maximum_time_scale = float(values.get("maximum_time_scale", 3.0))
            tcp_linear_limit = float(
                values.get(
                    "tcp_linear_velocity_limit_m_s",
                    M0609_NOMINAL_TCP_LINEAR_SPEED_M_S,
                )
            )
            tcp_angular_limit = float(
                values.get(
                    "tcp_angular_velocity_limit_rad_s",
                    M0609_CONSERVATIVE_TCP_ANGULAR_SPEED_RAD_S,
                )
            )
            ik_tolerance = float(values.get("ik_tolerance", 2.0e-7))
            ik_damping = float(values.get("ik_damping", 1.0e-7))
            ik_step_size = float(values.get("ik_step_size", 0.75))
            ik_max_iterations = int(values.get("ik_max_iterations", 30))
        except (TypeError, ValueError) as exc:
            raise M0609JointSpeedError("M0609 joint speed 설정값이 숫자가 아닙니다.") from exc
        finite_positive = (
            tcp_offset_m,
            minimum_time_scale,
            maximum_time_scale,
            tcp_linear_limit,
            tcp_angular_limit,
            ik_tolerance,
            ik_damping,
            ik_step_size,
        )
        if (
            not 0.0 < target_fraction <= 1.0
            or not np.isfinite(finite_positive).all()
            or any(item <= 0.0 for item in finite_positive)
            or minimum_time_scale > 1.0
            or maximum_time_scale < 1.0
            or not 0.0 < ik_step_size <= 1.0
            or ik_max_iterations <= 0
        ):
            raise M0609JointSpeedError("M0609 joint speed retiming 설정 범위가 유효하지 않습니다.")

        axis = _three_vector(values.get("tcp_offset_axis", [0.0, 0.0, 1.0]), name="tcp_offset_axis")
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= np.finfo(float).eps:
            raise M0609JointSpeedError("tcp_offset_axis는 영벡터일 수 없습니다.")
        axis /= axis_norm
        q_teach = _six_vector(values.get("q_teach_rad"), name="q_teach_rad")
        velocity_limits = _six_vector(
            values.get(
                "joint_velocity_limits_rad_s",
                M0609_NOMINAL_JOINT_VELOCITY_LIMITS_RAD_S,
            ),
            name="joint_velocity_limits_rad_s",
            positive=True,
        )
        lower = _six_vector(
            values.get(
                "joint_position_lower_rad",
                M0609_DEFAULT_JOINT_POSITION_LOWER_RAD,
            ),
            name="joint_position_lower_rad",
        )
        upper = _six_vector(
            values.get(
                "joint_position_upper_rad",
                M0609_DEFAULT_JOINT_POSITION_UPPER_RAD,
            ),
            name="joint_position_upper_rad",
        )
        if np.any(lower >= upper):
            raise M0609JointSpeedError("joint position lower는 upper보다 작아야 합니다.")
        if np.any(q_teach < lower) or np.any(q_teach > upper):
            raise M0609JointSpeedError("q_teach_rad가 joint position 범위를 벗어납니다.")
        return cls(
            target_fraction=target_fraction,
            tcp_offset_m=tcp_offset_m,
            tcp_offset_axis=tuple(float(item) for item in axis),
            q_teach_rad=tuple(float(item) for item in q_teach),
            joint_velocity_limits_rad_s=tuple(float(item) for item in velocity_limits),
            joint_position_lower_rad=tuple(float(item) for item in lower),
            joint_position_upper_rad=tuple(float(item) for item in upper),
            tcp_linear_velocity_limit_m_s=tcp_linear_limit,
            tcp_angular_velocity_limit_rad_s=tcp_angular_limit,
            minimum_time_scale=minimum_time_scale,
            maximum_time_scale=maximum_time_scale,
            ik_tolerance=ik_tolerance,
            ik_damping=ik_damping,
            ik_step_size=ik_step_size,
            ik_max_iterations=ik_max_iterations,
        )


@dataclass(frozen=True, slots=True)
class M0609JointSpeedReport:
    """Serializable nominal joint/TCP speed utilization report."""

    status: str
    target_fraction: float
    required_uniform_time_scale: float
    tcp_offset_m: float
    tcp_offset_axis: tuple[float, float, float]
    q_teach_rad: tuple[float, float, float, float, float, float]
    joint_velocity_limits_rad_s: tuple[float, float, float, float, float, float]
    peak_joint_velocity_rad_s: tuple[float, float, float, float, float, float]
    joint_velocity_utilization: tuple[float, float, float, float, float, float]
    peak_joint_acceleration_rad_s2: tuple[float, float, float, float, float, float]
    peak_tcp_linear_velocity_m_s: float
    tcp_linear_velocity_limit_m_s: float
    tcp_linear_velocity_utilization: float
    peak_tcp_angular_velocity_rad_s: float
    tcp_angular_velocity_limit_rad_s: float
    tcp_angular_velocity_utilization: float
    limiting_quantity: str
    limiting_utilization: float
    minimum_joint_position_margin_rad: float
    maximum_jacobian_condition_number: float
    maximum_ik_position_error_m: float
    maximum_ik_orientation_error_rad: float
    ik_sample_count: int
    safety_status: str = "offline_speed_only_assumed_q_teach"

    def summary(self) -> dict[str, Any]:
        result = asdict(self)
        result["joint_velocity_limits_deg_s"] = np.rad2deg(
            self.joint_velocity_limits_rad_s
        ).tolist()
        result["peak_joint_velocity_deg_s"] = np.rad2deg(
            self.peak_joint_velocity_rad_s
        ).tolist()
        result["peak_joint_acceleration_deg_s2"] = np.rad2deg(
            self.peak_joint_acceleration_rad_s2
        ).tolist()
        result["q_teach_deg"] = np.rad2deg(self.q_teach_rad).tolist()
        result["minimum_joint_position_margin_deg"] = float(
            np.rad2deg(self.minimum_joint_position_margin_rad)
        )
        result["source"] = {
            "joint_speed_spec": "Doosan M0609 V3 manual",
            "kinematic_chain": (
                "DoosanRobotics/doosan-robot2 "
                "ec9242546ec6202835900dbcd8498e2daabfa6a6"
            ),
            "acceleration_limits": "not_manufacturer_verified",
        }
        return result


def _kinematic_xml(offset_m: float, axis: tuple[float, float, float]) -> str:
    offset = np.asarray(axis, dtype=float) * float(offset_m)
    inertial = '<inertial pos="0 0 0" mass="1" diaginertia=".001 .001 .001"/>'
    # Link origins/quaternions match the official generated M0609 MJCF.
    # The tool0 rotation is the generated equivalent of URDF rpy=(pi,-pi/2,0).
    return f"""
<mujoco model="m0609_nominal_speed_kinematics">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="link_1" pos="0 0 0.1345">{inertial}
      <joint name="joint_1" axis="0 0 1"/>
      <body name="link_2" pos="0 0.0062 0" quat=".499898 -.500102 -.5 -.5">{inertial}
        <joint name="joint_2" axis="0 0 1"/>
        <body name="link_3" pos=".411 0 0" quat=".707035 0 0 .707179">{inertial}
          <joint name="joint_3" axis="0 0 1"/>
          <body name="link_4" pos="0 -.368 0" quat=".707035 .707179 0 0">{inertial}
            <joint name="joint_4" axis="0 0 1"/>
            <body name="link_5" quat=".707035 -.707179 0 0">{inertial}
              <joint name="joint_5" axis="0 0 1"/>
              <body name="link_6" pos="0 -.121 0" quat=".707035 .707179 0 0">{inertial}
                <joint name="joint_6" axis="0 0 1"/>
                <body name="tool0" quat="0 .7071067811865476 0 .7071067811865476">
                  <site name="pan_tcp" pos="{offset[0]:.16g} {offset[1]:.16g} {offset[2]:.16g}"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@lru_cache(maxsize=8)
def _cached_model(
    offset_m: float,
    axis: tuple[float, float, float],
) -> tuple[mujoco.MjModel, int]:
    model = mujoco.MjModel.from_xml_string(_kinematic_xml(offset_m, axis))
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pan_tcp")
    if site_id < 0:
        raise M0609JointSpeedError("M0609 kinematic model에 pan_tcp site가 없습니다.")
    return model, site_id


def _site_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = data.site_xmat[site_id].reshape(3, 3)
    transform[:3, 3] = data.site_xpos[site_id]
    jacobian_position = np.zeros((3, model.nv), dtype=float)
    jacobian_rotation = np.zeros((3, model.nv), dtype=float)
    mujoco.mj_jacSite(
        model,
        data,
        jacobian_position,
        jacobian_rotation,
        site_id,
    )
    return transform, np.vstack((jacobian_position, jacobian_rotation))


def _target_transforms(trajectory: Any) -> np.ndarray:
    positions = np.asarray(trajectory.position_m, dtype=float)
    orientations = np.asarray(trajectory.orientation_rpy_rad, dtype=float)
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or orientations.shape != positions.shape
        or len(positions) < 2
        or not np.isfinite(positions).all()
        or not np.isfinite(orientations).all()
    ):
        raise M0609JointSpeedError("trajectory pose 배열은 유효한 (T,3)이어야 합니다.")
    return np.stack(
        [
            compose_transform(position, rpy_rad=orientation)
            for position, orientation in zip(positions, orientations, strict=True)
        ]
    )


def evaluate_m0609_joint_speeds(
    trajectory: Any,
    settings: M0609JointSpeedSettings | Mapping[str, Any],
) -> M0609JointSpeedReport:
    """Run sequential IK and measure all six joint speeds plus TCP speed."""

    resolved = (
        settings
        if isinstance(settings, M0609JointSpeedSettings)
        else M0609JointSpeedSettings.from_mapping(settings)
    )
    model, site_id = _cached_model(resolved.tcp_offset_m, resolved.tcp_offset_axis)
    if model.nq != 6 or model.nv != 6:
        raise M0609JointSpeedError("M0609 nominal model은 정확히 6개 관절이어야 합니다.")
    data = mujoco.MjData(model)
    q = np.asarray(resolved.q_teach_rad, dtype=float).copy()
    lower = np.asarray(resolved.joint_position_lower_rad, dtype=float)
    upper = np.asarray(resolved.joint_position_upper_rad, dtype=float)

    pan_transforms = _target_transforms(trajectory)
    relative = np.einsum(
        "ij,tjk->tik",
        invert_transform(pan_transforms[0]),
        pan_transforms,
    )
    base_tcp_start, _ = _site_state(model, data, site_id, q)
    target_transforms = np.einsum("ij,tjk->tik", base_tcp_start, relative)
    rotation_map = base_tcp_start[:3, :3] @ pan_transforms[0, :3, :3].T

    linear_velocity = np.asarray(trajectory.linear_velocity_m_s, dtype=float)
    angular_velocity = np.asarray(trajectory.angular_velocity_rad_s, dtype=float)
    timestamps = np.asarray(trajectory.time_s, dtype=float)
    if (
        linear_velocity.shape != (len(timestamps), 3)
        or angular_velocity.shape != (len(timestamps), 3)
        or np.any(np.diff(timestamps) <= 0.0)
    ):
        raise M0609JointSpeedError("trajectory time/velocity 배열이 유효하지 않습니다.")

    q_sequence = np.empty((len(timestamps), 6), dtype=float)
    qd_sequence = np.empty_like(q_sequence)
    condition_numbers = np.empty(len(timestamps), dtype=float)
    position_errors = np.empty(len(timestamps), dtype=float)
    orientation_errors = np.empty(len(timestamps), dtype=float)

    for sample_index, target in enumerate(target_transforms):
        converged = False
        for _ in range(resolved.ik_max_iterations):
            current, jacobian = _site_state(model, data, site_id, q)
            position_error = target[:3, 3] - current[:3, 3]
            orientation_error = Rotation.from_matrix(
                target[:3, :3] @ current[:3, :3].T
            ).as_rotvec()
            if (
                float(np.linalg.norm(position_error)) <= resolved.ik_tolerance
                and float(np.linalg.norm(orientation_error)) <= resolved.ik_tolerance
            ):
                converged = True
                break
            error = np.concatenate((position_error, orientation_error))
            lhs = jacobian @ jacobian.T + resolved.ik_damping * np.eye(6)
            delta = jacobian.T @ np.linalg.solve(lhs, error)
            q = np.clip(q + resolved.ik_step_size * delta, lower, upper)
        if not converged:
            raise M0609JointSpeedError(
                f"0.4 m TCP sequential IK가 sample {sample_index}에서 수렴하지 않았습니다."
            )
        current, jacobian = _site_state(model, data, site_id, q)
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        minimum_singular = max(float(singular_values[-1]), np.finfo(float).eps)
        condition_numbers[sample_index] = float(singular_values[0] / minimum_singular)
        position_errors[sample_index] = float(
            np.linalg.norm(target[:3, 3] - current[:3, 3])
        )
        orientation_errors[sample_index] = float(
            np.linalg.norm(
                Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec()
            )
        )
        target_twist = np.concatenate(
            (
                rotation_map @ linear_velocity[sample_index],
                rotation_map @ angular_velocity[sample_index],
            )
        )
        qd = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + 1.0e-9 * np.eye(6),
            target_twist,
        )
        q_sequence[sample_index] = q
        qd_sequence[sample_index] = qd

    if not (np.isfinite(q_sequence).all() and np.isfinite(qd_sequence).all()):
        raise M0609JointSpeedError("M0609 joint trajectory에 NaN 또는 inf가 있습니다.")
    qdd_sequence = np.gradient(qd_sequence, timestamps, axis=0, edge_order=2)
    peak_joint_velocity = np.max(np.abs(qd_sequence), axis=0)
    peak_joint_acceleration = np.max(np.abs(qdd_sequence), axis=0)
    joint_limits = np.asarray(resolved.joint_velocity_limits_rad_s, dtype=float)
    joint_utilization = peak_joint_velocity / joint_limits

    tcp_linear_speed = np.linalg.norm(linear_velocity, axis=1)
    tcp_angular_speed = np.linalg.norm(angular_velocity, axis=1)
    peak_tcp_linear = float(np.max(tcp_linear_speed))
    peak_tcp_angular = float(np.max(tcp_angular_speed))
    tcp_linear_utilization = peak_tcp_linear / resolved.tcp_linear_velocity_limit_m_s
    tcp_angular_utilization = peak_tcp_angular / resolved.tcp_angular_velocity_limit_rad_s
    candidates = {
        **{
            f"joint_{index + 1}_velocity": float(utilization)
            for index, utilization in enumerate(joint_utilization)
        },
        "tcp_linear_velocity": tcp_linear_utilization,
        "tcp_angular_velocity": tcp_angular_utilization,
    }
    limiting_quantity, limiting_utilization = max(
        candidates.items(),
        key=lambda item: item[1],
    )
    required_scale = limiting_utilization / resolved.target_fraction
    position_margin = float(
        np.min(np.minimum(q_sequence - lower, upper - q_sequence))
    )
    status = (
        "within_nominal_90_percent_speed_envelope"
        if limiting_utilization <= resolved.target_fraction * 1.002
        else "requires_uniform_retiming"
    )
    return M0609JointSpeedReport(
        status=status,
        target_fraction=resolved.target_fraction,
        required_uniform_time_scale=float(required_scale),
        tcp_offset_m=resolved.tcp_offset_m,
        tcp_offset_axis=resolved.tcp_offset_axis,
        q_teach_rad=resolved.q_teach_rad,
        joint_velocity_limits_rad_s=resolved.joint_velocity_limits_rad_s,
        peak_joint_velocity_rad_s=tuple(float(item) for item in peak_joint_velocity),
        joint_velocity_utilization=tuple(float(item) for item in joint_utilization),
        peak_joint_acceleration_rad_s2=tuple(
            float(item) for item in peak_joint_acceleration
        ),
        peak_tcp_linear_velocity_m_s=peak_tcp_linear,
        tcp_linear_velocity_limit_m_s=resolved.tcp_linear_velocity_limit_m_s,
        tcp_linear_velocity_utilization=float(tcp_linear_utilization),
        peak_tcp_angular_velocity_rad_s=peak_tcp_angular,
        tcp_angular_velocity_limit_rad_s=resolved.tcp_angular_velocity_limit_rad_s,
        tcp_angular_velocity_utilization=float(tcp_angular_utilization),
        limiting_quantity=limiting_quantity,
        limiting_utilization=float(limiting_utilization),
        minimum_joint_position_margin_rad=position_margin,
        maximum_jacobian_condition_number=float(np.max(condition_numbers)),
        maximum_ik_position_error_m=float(np.max(position_errors)),
        maximum_ik_orientation_error_rad=float(np.max(orientation_errors)),
        ik_sample_count=len(timestamps),
    )
