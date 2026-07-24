"""전체 웍질을 한 번에 보간하는 global degree-5 B-spline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline

from wok_sim.geometry.transforms import (
    WokFrameContext,
    matrix_to_rpy,
    resolve_wok_frame_context,
    rpy_to_matrix,
    transform_to_pose,
)

from .parameters import TrajectoryParameters, map_action, trajectory_section
from .waypoints import PanPose, WaypointSequence, build_repeated_waypoints

if TYPE_CHECKING:
    from .validator import ValidationResult


class SplineGenerationError(ValueError):
    """global spline 생성 또는 평가가 실패했을 때 발생한다."""


def _lookup(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


@dataclass(slots=True)
class Trajectory:
    """공통 time axis의 base/world pose와 wok-local 원본 trajectory."""

    time_s: np.ndarray
    position_m: np.ndarray
    orientation_rpy_rad: np.ndarray
    linear_velocity_m_s: np.ndarray
    angular_velocity_rad_s: np.ndarray
    linear_acceleration_m_s2: np.ndarray
    angular_acceleration_rad_s2: np.ndarray
    linear_jerk_m_s3: np.ndarray
    angular_jerk_rad_s3: np.ndarray
    parameters: TrajectoryParameters
    waypoints: WaypointSequence
    spline: Any
    frame_context: WokFrameContext
    position_wok_m: np.ndarray
    orientation_wok_rpy_rad: np.ndarray
    linear_velocity_wok_m_s: np.ndarray
    angular_velocity_wok_rad_s: np.ndarray
    linear_acceleration_wok_m_s2: np.ndarray
    angular_acceleration_wok_rad_s2: np.ndarray
    linear_jerk_wok_m_s3: np.ndarray
    angular_jerk_wok_rad_s3: np.ndarray
    validation: ValidationResult | None = None

    def __post_init__(self) -> None:
        count = np.asarray(self.time_s).size
        if np.asarray(self.time_s).shape != (count,):
            raise SplineGenerationError("time_s는 1차원이어야 합니다.")
        arrays = (
            self.position_m,
            self.orientation_rpy_rad,
            self.linear_velocity_m_s,
            self.angular_velocity_rad_s,
            self.linear_acceleration_m_s2,
            self.angular_acceleration_rad_s2,
            self.linear_jerk_m_s3,
            self.angular_jerk_rad_s3,
            self.position_wok_m,
            self.orientation_wok_rpy_rad,
            self.linear_velocity_wok_m_s,
            self.angular_velocity_wok_rad_s,
            self.linear_acceleration_wok_m_s2,
            self.angular_acceleration_wok_rad_s2,
            self.linear_jerk_wok_m_s3,
            self.angular_jerk_wok_rad_s3,
        )
        if any(np.asarray(array).shape != (count, 3) for array in arrays):
            raise SplineGenerationError("모든 pose/미분 배열 shape은 (N, 3)이어야 합니다.")
        if np.any(np.diff(self.time_s) <= 0.0):
            raise SplineGenerationError("trajectory time_s는 엄격히 증가해야 합니다.")
        if not all(np.isfinite(array).all() for array in (self.time_s, *arrays)):
            raise SplineGenerationError("trajectory에 NaN 또는 inf가 포함되어 있습니다.")

    @property
    def times(self) -> np.ndarray:
        """time_s 호환 alias."""

        return self.time_s

    @property
    def timestamps(self) -> np.ndarray:
        """export용 timestamp alias."""

        return self.time_s

    @property
    def positions(self) -> np.ndarray:
        """position_m 호환 alias."""

        return self.position_m

    @property
    def pose(self) -> np.ndarray:
        """[x,y,z,roll,pitch,yaw] 샘플."""

        return np.column_stack((self.position_m, self.orientation_rpy_rad))

    @property
    def velocity(self) -> np.ndarray:
        """linear/angular velocity를 합친 (N, 6) 배열."""

        return np.column_stack((self.linear_velocity_m_s, self.angular_velocity_rad_s))

    @property
    def acceleration(self) -> np.ndarray:
        """linear/angular acceleration을 합친 (N, 6) 배열."""

        return np.column_stack((self.linear_acceleration_m_s2, self.angular_acceleration_rad_s2))

    @property
    def jerk(self) -> np.ndarray:
        """linear/angular jerk를 합친 (N, 6) 배열."""

        return np.column_stack((self.linear_jerk_m_s3, self.angular_jerk_rad_s3))

    @property
    def duration_s(self) -> float:
        """전체 trajectory 지속시간(s)."""

        return float(self.time_s[-1] - self.time_s[0])

    @property
    def cycle_count(self) -> int:
        """trajectory에 포함된 웍질 횟수."""

        return self.waypoints.cycle_count

    @property
    def valid(self) -> bool | None:
        """validation을 생략했으면 None, 아니면 validation 결과."""

        return None if self.validation is None else self.validation.valid

    def evaluate(self, time_s: float | np.ndarray, derivative: int = 0) -> np.ndarray:
        """analytic wok spline을 평가해 base/world pose 또는 twist로 반환한다."""

        local = [self.spline.evaluate(time_s, derivative=index) for index in range(derivative + 1)]
        return _transform_spline_evaluation(self.frame_context, local, derivative)

    def evaluate_wok(self, time_s: float | np.ndarray, derivative: int = 0) -> np.ndarray:
        """wok-local pose 또는 실제 angular twist를 포함한 미분을 반환한다."""

        local = [self.spline.evaluate(time_s, derivative=index) for index in range(derivative + 1)]
        transformed = _local_kinematics(local)
        return transformed[derivative]

    @property
    def local_position_m(self) -> np.ndarray:
        """``position_wok_m`` 호환 alias."""

        return self.position_wok_m

    @property
    def local_orientation_rpy_rad(self) -> np.ndarray:
        """``orientation_wok_rpy_rad`` 호환 alias."""

        return self.orientation_wok_rpy_rad


def _at_least_2d(values: np.ndarray) -> tuple[np.ndarray, bool]:
    array = np.asarray(values, dtype=float)
    scalar = array.ndim == 1
    return (array[None, :] if scalar else array), scalar


def _restore_shape(values: np.ndarray, scalar: bool) -> np.ndarray:
    return values[0] if scalar else values


def _actual_angular_kinematics(
    rpy: np.ndarray,
    rpy_velocity: np.ndarray,
    rpy_acceleration: np.ndarray | None = None,
    rpy_jerk: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """ZYX RPY 미분을 world/wok 축 actual angular twist로 변환한다."""

    values, scalar = _at_least_2d(rpy)
    velocity, _ = _at_least_2d(rpy_velocity)
    acceleration = (
        np.zeros_like(velocity) if rpy_acceleration is None else _at_least_2d(rpy_acceleration)[0]
    )
    jerk = np.zeros_like(velocity) if rpy_jerk is None else _at_least_2d(rpy_jerk)[0]
    roll_rate, pitch_rate, yaw_rate = velocity.T
    roll_acc, pitch_acc, yaw_acc = acceleration.T
    roll_jerk, pitch_jerk, yaw_jerk = jerk.T
    pitch = values[:, 1]
    yaw = values[:, 2]
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    a = np.column_stack((cy * cp, sy * cp, -sp))
    b = np.column_stack((-sy, cy, np.zeros_like(yaw)))
    c = np.broadcast_to(np.array([0.0, 0.0, 1.0]), a.shape)
    a_p = np.column_stack((-cy * sp, -sy * sp, -cp))
    a_y = np.column_stack((-sy * cp, cy * cp, np.zeros_like(yaw)))
    b_y = np.column_stack((-cy, -sy, np.zeros_like(yaw)))
    a_dot = a_p * pitch_rate[:, None] + a_y * yaw_rate[:, None]
    b_dot = b_y * yaw_rate[:, None]

    omega = a * roll_rate[:, None] + b * pitch_rate[:, None] + c * yaw_rate[:, None]
    alpha = (
        a * roll_acc[:, None]
        + b * pitch_acc[:, None]
        + c * yaw_acc[:, None]
        + a_dot * roll_rate[:, None]
        + b_dot * pitch_rate[:, None]
    )

    a_pp = np.column_stack((-cy * cp, -sy * cp, sp))
    a_py = np.column_stack((sy * sp, -cy * sp, np.zeros_like(yaw)))
    a_yy = np.column_stack((-cy * cp, -sy * cp, np.zeros_like(yaw)))
    b_yy = np.column_stack((sy, -cy, np.zeros_like(yaw)))
    a_ddot = (
        a_pp * (pitch_rate**2)[:, None]
        + 2.0 * a_py * (pitch_rate * yaw_rate)[:, None]
        + a_yy * (yaw_rate**2)[:, None]
        + a_p * pitch_acc[:, None]
        + a_y * yaw_acc[:, None]
    )
    b_ddot = b_yy * (yaw_rate**2)[:, None] + b_y * yaw_acc[:, None]
    angular_jerk = (
        a * roll_jerk[:, None]
        + b * pitch_jerk[:, None]
        + c * yaw_jerk[:, None]
        + 2.0 * a_dot * roll_acc[:, None]
        + 2.0 * b_dot * pitch_acc[:, None]
        + a_ddot * roll_rate[:, None]
        + b_ddot * pitch_rate[:, None]
    )
    return (
        _restore_shape(omega, scalar),
        None if rpy_acceleration is None else _restore_shape(alpha, scalar),
        None if rpy_jerk is None else _restore_shape(angular_jerk, scalar),
    )


def _local_kinematics(local_derivatives: list[np.ndarray]) -> list[np.ndarray]:
    """spline의 [xyz, RPY] 미분을 local pose/[linear, angular]로 변환한다."""

    pose = np.asarray(local_derivatives[0], dtype=float)
    results = [pose]
    if len(local_derivatives) == 1:
        return results
    first = np.asarray(local_derivatives[1], dtype=float)
    omega, _, _ = _actual_angular_kinematics(pose[..., 3:], first[..., 3:])
    results.append(np.concatenate((first[..., :3], omega), axis=-1))
    if len(local_derivatives) == 2:
        return results
    second = np.asarray(local_derivatives[2], dtype=float)
    _, alpha, _ = _actual_angular_kinematics(pose[..., 3:], first[..., 3:], second[..., 3:])
    assert alpha is not None
    results.append(np.concatenate((second[..., :3], alpha), axis=-1))
    if len(local_derivatives) == 3:
        return results
    third = np.asarray(local_derivatives[3], dtype=float)
    _, _, angular_jerk = _actual_angular_kinematics(
        pose[..., 3:],
        first[..., 3:],
        second[..., 3:],
        third[..., 3:],
    )
    assert angular_jerk is not None
    results.append(np.concatenate((third[..., :3], angular_jerk), axis=-1))
    return results


def _transform_spline_evaluation(
    frame_context: WokFrameContext,
    local_derivatives: list[np.ndarray],
    derivative: int,
) -> np.ndarray:
    """wok-local spline 평가값을 base/world pose 및 twist로 변환한다."""

    local = _local_kinematics(local_derivatives)
    rotation = frame_context.T_base_wok[:3, :3]
    translation = frame_context.T_base_wok[:3, 3]
    selected = np.asarray(local[derivative], dtype=float)
    values, scalar = _at_least_2d(selected)
    if derivative == 0:
        position = values[:, :3] @ rotation.T + translation
        base_rpy = np.vstack(
            [matrix_to_rpy(rotation @ rpy_to_matrix(rpy)) for rpy in values[:, 3:]]
        )
        result = np.column_stack((position, base_rpy))
    else:
        result = np.column_stack((values[:, :3] @ rotation.T, values[:, 3:] @ rotation.T))
    return _restore_shape(result, scalar)


def _sample_spline(
    spline: Any,
    parameters: Any,
    sample_rate_hz: float,
    frame_context: WokFrameContext,
) -> Trajectory:
    """공통 spline interface를 공통 time axis의 :class:`Trajectory`로 샘플한다."""

    try:
        rate = float(sample_rate_hz)
    except (TypeError, ValueError) as exc:
        raise SplineGenerationError("sample_rate_hz는 숫자여야 합니다.") from exc
    if not np.isfinite(rate) or rate <= 0.0:
        raise SplineGenerationError("sample_rate_hz는 양의 유한값이어야 합니다.")
    duration = spline.end_time_s - spline.start_time_s
    intervals = max(1, int(np.ceil(duration * rate)))
    uniform_times = np.linspace(spline.start_time_s, spline.end_time_s, intervals + 1)
    # 논리적 경계에서 정확히 미분을 검사할 수 있도록 knot도 포함한다.
    times = np.unique(np.concatenate((uniform_times, spline.waypoints.times)))
    local_derivatives = [spline.evaluate(times, derivative=index) for index in range(4)]
    local = _local_kinematics(local_derivatives)
    transformed = [
        _transform_spline_evaluation(
            frame_context,
            local_derivatives[: derivative + 1],
            derivative,
        )
        for derivative in range(4)
    ]
    pose, velocity, acceleration, jerk = transformed
    return Trajectory(
        time_s=times,
        position_m=pose[:, :3],
        orientation_rpy_rad=pose[:, 3:],
        linear_velocity_m_s=velocity[:, :3],
        angular_velocity_rad_s=velocity[:, 3:],
        linear_acceleration_m_s2=acceleration[:, :3],
        angular_acceleration_rad_s2=acceleration[:, 3:],
        linear_jerk_m_s3=jerk[:, :3],
        angular_jerk_rad_s3=jerk[:, 3:],
        parameters=parameters,
        waypoints=spline.waypoints,
        spline=spline,
        frame_context=frame_context,
        position_wok_m=local[0][:, :3],
        orientation_wok_rpy_rad=local[0][:, 3:],
        linear_velocity_wok_m_s=local[1][:, :3],
        angular_velocity_wok_rad_s=local[1][:, 3:],
        linear_acceleration_wok_m_s2=local[2][:, :3],
        angular_acceleration_wok_rad_s2=local[2][:, 3:],
        linear_jerk_wok_m_s3=local[3][:, :3],
        angular_jerk_wok_rad_s3=local[3][:, 3:],
    )


@dataclass(frozen=True, slots=True)
class GlobalQuinticSpline:
    """P0~P5 반복 전체를 보간하는 C4(global degree-5) B-spline."""

    waypoints: WaypointSequence
    _spline: BSpline

    @classmethod
    def from_waypoints(cls, waypoints: WaypointSequence) -> GlobalQuinticSpline:
        """waypoint를 통과하고 양 끝 velocity/acceleration이 0인 spline을 생성한다."""

        times = waypoints.times
        poses = waypoints.poses
        if len(times) < 6:
            raise SplineGenerationError("degree-5 spline에는 최소 6개 waypoint가 필요합니다.")
        zero = np.zeros(poses.shape[1], dtype=float)
        boundary_conditions = (
            [(1, zero.copy()), (2, zero.copy())],
            [(1, zero.copy()), (2, zero.copy())],
        )
        try:
            spline = make_interp_spline(
                times,
                poses,
                k=5,
                bc_type=boundary_conditions,
                axis=0,
                check_finite=True,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise SplineGenerationError(f"global quintic spline 생성 실패: {exc}") from exc
        return cls(waypoints, spline)

    @property
    def start_time_s(self) -> float:
        """spline 시작 시각(s)."""

        return float(self.waypoints.times[0])

    @property
    def end_time_s(self) -> float:
        """spline 종료 시각(s)."""

        return float(self.waypoints.times[-1])

    def evaluate(self, time_s: float | np.ndarray, derivative: int = 0) -> np.ndarray:
        """0=pose, 1=velocity, 2=acceleration, 3=jerk를 analytic 계산한다."""

        if derivative not in (0, 1, 2, 3):
            raise ValueError("derivative는 0, 1, 2, 3 중 하나여야 합니다.")
        query = np.asarray(time_s, dtype=float)
        if not np.isfinite(query).all():
            raise SplineGenerationError("평가 시각에 NaN 또는 inf가 포함되어 있습니다.")
        tolerance = 1.0e-12
        if np.any(query < self.start_time_s - tolerance) or np.any(
            query > self.end_time_s + tolerance
        ):
            raise SplineGenerationError("spline 정의 구간 밖의 시각을 평가할 수 없습니다.")
        result = np.asarray(self._spline(query, nu=derivative, extrapolate=False), dtype=float)
        if not np.isfinite(result).all():
            raise SplineGenerationError("spline 평가 결과에 NaN 또는 inf가 있습니다.")
        return result

    def sample(
        self,
        parameters: TrajectoryParameters,
        sample_rate_hz: float,
        frame_context: WokFrameContext,
    ) -> Trajectory:
        """local spline과 base/world 변환 결과를 공통 time axis에 샘플한다."""

        return _sample_spline(self, parameters, sample_rate_hz, frame_context)


@dataclass(frozen=True, slots=True)
class PhasewiseMinimumJerkSpline:
    """각 teaching phase를 monotone minimum-jerk quintic으로 연결한 C2 spline.

    ``s(u)=10u³-15u⁴+6u⁵``를 각 waypoint 구간에 적용한다. 따라서 position과
    orientation은 두 endpoint 사이를 벗어나지 않고 phase 경계의 속도와
    가속도는 정확히 0이다. jerk는 phase 경계에서 유한하게 바뀔 수 있다.
    """

    waypoints: WaypointSequence

    @classmethod
    def from_waypoints(cls, waypoints: WaypointSequence) -> PhasewiseMinimumJerkSpline:
        if len(waypoints) < 2:
            raise SplineGenerationError("phasewise spline에는 최소 2개 waypoint가 필요합니다.")
        return cls(waypoints)

    @property
    def start_time_s(self) -> float:
        return float(self.waypoints.times[0])

    @property
    def end_time_s(self) -> float:
        return float(self.waypoints.times[-1])

    def evaluate(self, time_s: float | np.ndarray, derivative: int = 0) -> np.ndarray:
        """0=pose, 1=velocity, 2=acceleration, 3=jerk를 analytic 계산한다."""

        if derivative not in (0, 1, 2, 3):
            raise ValueError("derivative는 0, 1, 2, 3 중 하나여야 합니다.")
        query = np.asarray(time_s, dtype=float)
        if not np.isfinite(query).all():
            raise SplineGenerationError("평가 시각에 NaN 또는 inf가 포함되어 있습니다.")
        tolerance = 1.0e-12
        if np.any(query < self.start_time_s - tolerance) or np.any(
            query > self.end_time_s + tolerance
        ):
            raise SplineGenerationError("spline 정의 구간 밖의 시각을 평가할 수 없습니다.")

        times = self.waypoints.times
        poses = self.waypoints.poses
        flat_query = np.clip(query.reshape(-1), self.start_time_s, self.end_time_s)
        indices = np.searchsorted(times, flat_query, side="right") - 1
        indices = np.clip(indices, 0, len(times) - 2)
        durations = times[indices + 1] - times[indices]
        u = (flat_query - times[indices]) / durations
        delta = poses[indices + 1] - poses[indices]

        if derivative == 0:
            scale = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
            result = poses[indices] + delta * scale[:, None]
        elif derivative == 1:
            scale = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / durations
            result = delta * scale[:, None]
        elif derivative == 2:
            scale = (60.0 * u - 180.0 * u**2 + 120.0 * u**3) / durations**2
            result = delta * scale[:, None]
        else:
            scale = (60.0 - 360.0 * u + 360.0 * u**2) / durations**3
            result = delta * scale[:, None]

        shaped = result.reshape(query.shape + (poses.shape[1],))
        if not np.isfinite(shaped).all():
            raise SplineGenerationError("phasewise spline 평가 결과가 유한하지 않습니다.")
        return shaped

    def sample(
        self,
        parameters: Any,
        sample_rate_hz: float,
        frame_context: WokFrameContext,
    ) -> Trajectory:
        return _sample_spline(self, parameters, sample_rate_hz, frame_context)


def generate_trajectory(
    action_or_parameters: (Sequence[float] | np.ndarray | Mapping[str, Any] | TrajectoryParameters),
    config: Any,
    *,
    start_pose: PanPose | Sequence[float] | Mapping[str, Any] | None = None,
    cycles: int | None = None,
    sample_rate_hz: float | None = None,
    validate: bool = True,
    strict_action: bool = False,
) -> Trajectory:
    """정규화 action에서 5회 global trajectory와 선택적 validation을 생성한다."""

    if isinstance(action_or_parameters, TrajectoryParameters):
        parameters = action_or_parameters
    elif isinstance(action_or_parameters, Mapping) or any(
        hasattr(action_or_parameters, name)
        for name in (
            "insertion_distance",
            "insertion_distance_m",
            "lift_height",
            "lift_height_m",
        )
    ):
        parameters = TrajectoryParameters.from_mapping(action_or_parameters)
    else:
        parameters = map_action(action_or_parameters, config, clip=not strict_action)

    try:
        frame_context = resolve_wok_frame_context(config)
    except ValueError as exc:
        raise SplineGenerationError(f"trajectory frame 설정 오류: {exc}") from exc
    wok_position, wok_rpy, _ = transform_to_pose(frame_context.T_wok_pan0)
    authoritative_start = PanPose.from_value({"position_m": wok_position, "rpy_rad": wok_rpy})
    if start_pose is not None:
        requested_start = PanPose.from_value(start_pose)
        if not np.allclose(
            requested_start.as_array(),
            authoritative_start.as_array(),
            rtol=0.0,
            atol=1e-8,
        ):
            raise SplineGenerationError(
                "start_pose가 teaching/pan에서 해석한 authoritative wok P0와 일치하지 않습니다."
            )
    waypoints = build_repeated_waypoints(
        parameters,
        config,
        start_pose=authoritative_start,
        cycles=cycles,
    )
    spline = GlobalQuinticSpline.from_waypoints(waypoints)
    section = trajectory_section(config)
    rate = _lookup(section, "sample_rate_hz", None) if sample_rate_hz is None else sample_rate_hz
    if rate is None:
        raise SplineGenerationError("trajectory.sample_rate_hz 설정이 필요합니다.")
    trajectory = spline.sample(parameters, float(rate), frame_context)
    if validate:
        from .validator import validate_trajectory

        trajectory.validation = validate_trajectory(trajectory, config)
    return trajectory


def create_trajectory(
    action_or_parameters: (Sequence[float] | np.ndarray | Mapping[str, Any] | TrajectoryParameters),
    config: Any,
    **kwargs: Any,
) -> Trajectory:
    """``generate_trajectory``의 호환 alias."""

    return generate_trajectory(action_or_parameters, config, **kwargs)
