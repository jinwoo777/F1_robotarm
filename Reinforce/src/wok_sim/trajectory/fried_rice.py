"""볶음밥용 3단 teaching 궤적과 전용 action mapping."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from wok_sim.geometry.transforms import resolve_wok_frame_context, transform_to_pose

from .parameters import trajectory_section
from .spline import PhasewiseMinimumJerkSpline, SplineGenerationError, Trajectory
from .waypoints import PanPose, Waypoint, WaypointError, WaypointSequence

FRIED_RICE_ACTION_NAMES: tuple[str, ...] = (
    "insertion_distance",
    "tilt_angle",
    "linear_speed",
    "angular_speed",
)
"""볶음밥 profile의 normalized action 순서."""


_DEFAULT_RANGES: dict[str, tuple[float, float]] = {
    "insertion_distance_range_m": (0.20, 0.30),
    "tilt_angle_range_rad": (np.deg2rad(5.0), np.deg2rad(15.0)),
    "linear_speed_range_m_s": (0.20, 0.30),
    "angular_speed_range_rad_s": (0.15, 0.30),
}

_ACTION_RANGE_KEYS: tuple[str, ...] = tuple(_DEFAULT_RANGES)
_MINIMUM_JERK_PEAK_VELOCITY = 1.875
_MINIMUM_JERK_PEAK_ACCELERATION = 5.773502691896258
_MINIMUM_JERK_PEAK_JERK = 60.0


class FriedRiceTrajectoryError(ValueError):
    """볶음밥 profile action 또는 waypoint 설정이 유효하지 않을 때 발생한다."""


def _lookup(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def fried_rice_section(config: Any) -> Any:
    """``trajectory.fried_rice`` section을 반환하며 없으면 빈 mapping을 반환한다."""

    section = trajectory_section(config)
    profile = _lookup(section, "fried_rice", None)
    return {} if profile is None else profile


def _range_from_profile(config: Any, key: str) -> tuple[float, float]:
    profile = fried_rice_section(config)
    raw = _lookup(profile, key, _DEFAULT_RANGES[key])
    try:
        low, high = float(raw[0]), float(raw[1])
    except (IndexError, TypeError, ValueError) as exc:
        raise FriedRiceTrajectoryError(
            f"trajectory.fried_rice.{key}는 [min, max] 형식이어야 합니다."
        ) from exc
    if not np.isfinite([low, high]).all() or low > high:
        raise FriedRiceTrajectoryError(
            f"trajectory.fried_rice.{key} 범위가 유효하지 않습니다: {raw!r}"
        )
    return low, high


@dataclass(frozen=True, slots=True)
class FriedRiceParameters:
    """한 에피소드의 3단 teaching을 결정하는 SI 단위 파라미터."""

    insertion_distance: float
    tilt_angle: float
    linear_speed: float
    angular_speed: float
    linear_acceleration_limit: float = 0.80
    angular_acceleration_limit: float = 1.10
    linear_jerk_limit: float = 8.0
    angular_jerk_limit: float = 10.0
    minimum_phase_duration: float = 0.50

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                *self.as_array(),
                self.linear_acceleration_limit,
                self.angular_acceleration_limit,
                self.linear_jerk_limit,
                self.angular_jerk_limit,
                self.minimum_phase_duration,
            ),
            dtype=float,
        )
        if not np.isfinite(values).all():
            raise FriedRiceTrajectoryError("볶음밥 궤적 파라미터에 NaN 또는 inf가 있습니다.")
        if self.insertion_distance <= 0.0:
            raise FriedRiceTrajectoryError("insertion_distance는 양수여야 합니다.")
        if self.tilt_angle <= 0.0:
            raise FriedRiceTrajectoryError("tilt_angle은 양수여야 합니다.")
        if np.any(values[2:] <= 0.0):
            raise FriedRiceTrajectoryError("속도, 가속도, jerk와 phase duration은 양수여야 합니다.")

    @property
    def insertion_distance_m(self) -> float:
        return self.insertion_distance

    @property
    def tilt_angle_rad(self) -> float:
        return self.tilt_angle

    @property
    def linear_speed_m_s(self) -> float:
        return self.linear_speed

    @property
    def angular_speed_rad_s(self) -> float:
        return self.angular_speed

    @staticmethod
    def _minimum_jerk_duration(
        displacement: float,
        velocity_limit: float,
        acceleration_limit: float,
        jerk_limit: float,
        minimum_duration: float,
    ) -> float:
        return float(
            max(
                minimum_duration,
                _MINIMUM_JERK_PEAK_VELOCITY * displacement / velocity_limit,
                np.sqrt(_MINIMUM_JERK_PEAK_ACCELERATION * displacement / acceleration_limit),
                np.cbrt(_MINIMUM_JERK_PEAK_JERK * displacement / jerk_limit),
            )
        )

    @property
    def phase_durations_s(self) -> tuple[float, float, float]:
        """삽입, tilt, 동시 복귀 phase 시간을 속도/미분 gate에서 계산한다."""

        insertion = self._minimum_jerk_duration(
            self.insertion_distance,
            self.linear_speed,
            self.linear_acceleration_limit,
            self.linear_jerk_limit,
            self.minimum_phase_duration,
        )
        tilt = self._minimum_jerk_duration(
            self.tilt_angle,
            self.angular_speed,
            self.angular_acceleration_limit,
            self.angular_jerk_limit,
            self.minimum_phase_duration,
        )
        return insertion, tilt, max(insertion, tilt)

    @property
    def cycle_time(self) -> float:
        return float(sum(self.phase_durations_s))

    @property
    def cycle_time_s(self) -> float:
        return self.cycle_time

    @property
    def insert_phase_ratio(self) -> float:
        return self.phase_durations_s[0] / self.cycle_time

    @property
    def tilt_phase_ratio(self) -> float:
        return self.phase_durations_s[1] / self.cycle_time

    def as_array(self) -> np.ndarray:
        """``FRIED_RICE_ACTION_NAMES`` 순서의 물리값을 반환한다."""

        return np.asarray(
            (
                self.insertion_distance,
                self.tilt_angle,
                self.linear_speed,
                self.angular_speed,
            ),
            dtype=float,
        )

    def as_dict(self) -> dict[str, float]:
        """로깅 가능한 물리 파라미터 mapping."""

        insertion_duration, tilt_duration, return_duration = self.phase_durations_s
        return {
            **dict(zip(FRIED_RICE_ACTION_NAMES, self.as_array(), strict=True)),
            "cycle_time": self.cycle_time,
            "insert_phase_duration": insertion_duration,
            "tilt_phase_duration": tilt_duration,
            "return_phase_duration": return_duration,
            "insert_phase_ratio": self.insert_phase_ratio,
            "tilt_phase_ratio": self.tilt_phase_ratio,
            "linear_acceleration_limit": self.linear_acceleration_limit,
            "angular_acceleration_limit": self.angular_acceleration_limit,
            "linear_jerk_limit": self.linear_jerk_limit,
            "angular_jerk_limit": self.angular_jerk_limit,
        }

    @classmethod
    def teaching_default(cls) -> FriedRiceParameters:
        """20~30 cm/5~15도/speed 범위의 중앙 teaching."""

        return cls(
            insertion_distance=0.25,
            tilt_angle=np.deg2rad(10.0),
            linear_speed=0.25,
            angular_speed=0.225,
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | Any) -> FriedRiceParameters:
        """SI 단위 mapping 또는 동일 attribute를 가진 객체에서 생성한다."""

        aliases = {
            "insertion_distance": ("insertion_distance", "insertion_distance_m"),
            "tilt_angle": ("tilt_angle", "tilt_angle_rad"),
            "linear_speed": ("linear_speed", "linear_speed_m_s"),
            "angular_speed": ("angular_speed", "angular_speed_rad_s"),
        }
        resolved: dict[str, float] = {}
        for name, candidates in aliases.items():
            raw = None
            for candidate in candidates:
                raw = _lookup(values, candidate, None)
                if raw is not None:
                    break
            if raw is None:
                raise FriedRiceTrajectoryError(f"볶음밥 파라미터 '{name}' 값이 없습니다.")
            try:
                resolved[name] = float(raw)
            except (TypeError, ValueError) as exc:
                raise FriedRiceTrajectoryError(
                    f"볶음밥 파라미터 '{name}' 값이 숫자가 아닙니다."
                ) from exc
        optional = {
            "linear_acceleration_limit": 0.80,
            "angular_acceleration_limit": 1.10,
            "linear_jerk_limit": 8.0,
            "angular_jerk_limit": 10.0,
            "minimum_phase_duration": 0.50,
        }
        for name, default in optional.items():
            resolved[name] = float(_lookup(values, name, default))
        return cls(**resolved)


def map_fried_rice_action(
    action: Sequence[float] | np.ndarray,
    config: Any,
    *,
    clip: bool = True,
) -> FriedRiceParameters:
    """4D normalized action을 거리/각도/목표 peak 속도로 변환한다."""

    normalized = np.asarray(action, dtype=float)
    expected = len(FRIED_RICE_ACTION_NAMES)
    if normalized.shape != (expected,):
        raise FriedRiceTrajectoryError(
            f"볶음밥 action shape은 ({expected},)여야 합니다: {normalized.shape}"
        )
    if not np.isfinite(normalized).all():
        raise FriedRiceTrajectoryError("볶음밥 action에 NaN 또는 inf가 있습니다.")
    if not clip and np.any(np.abs(normalized) > 1.0):
        raise FriedRiceTrajectoryError("normalized 볶음밥 action은 [-1, 1] 범위여야 합니다.")
    normalized = np.clip(normalized, -1.0, 1.0)

    physical = []
    for value, key in zip(normalized, _ACTION_RANGE_KEYS, strict=True):
        low, high = _range_from_profile(config, key)
        physical.append(low + 0.5 * (value + 1.0) * (high - low))
    return FriedRiceParameters(
        *physical,
        linear_acceleration_limit=_profile_number(
            config,
            "linear_acceleration_limit_m_s2",
            0.80,
        ),
        angular_acceleration_limit=_profile_number(
            config,
            "angular_acceleration_limit_rad_s2",
            1.10,
        ),
        linear_jerk_limit=_profile_number(config, "linear_jerk_limit_m_s3", 8.0),
        angular_jerk_limit=_profile_number(config, "angular_jerk_limit_rad_s3", 10.0),
        minimum_phase_duration=_profile_number(
            config,
            "minimum_phase_duration_s",
            0.50,
        ),
    )


def fried_rice_phase_times(parameters: FriedRiceParameters) -> np.ndarray:
    """P0=시작, P1=삽입 완료, P2=tilt 완료, P3=복귀의 상대 시각."""

    insertion_duration, tilt_duration, _ = parameters.phase_durations_s
    times = np.asarray(
        (
            0.0,
            insertion_duration,
            insertion_duration + tilt_duration,
            parameters.cycle_time,
        ),
        dtype=float,
    )
    if np.any(np.diff(times) <= 0.0):
        raise FriedRiceTrajectoryError("볶음밥 teaching의 모든 phase duration은 양수여야 합니다.")
    return times


def _profile_number(config: Any, key: str, default: float) -> float:
    raw = _lookup(fried_rice_section(config), key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise FriedRiceTrajectoryError(f"trajectory.fried_rice.{key}는 숫자여야 합니다.") from exc
    if not np.isfinite(value):
        raise FriedRiceTrajectoryError(f"trajectory.fried_rice.{key}는 유한한 숫자여야 합니다.")
    return value


def _configured_start(config: Any) -> PanPose:
    section = trajectory_section(config)
    position = _lookup(section, "start_position_m", None)
    if position is not None:
        return PanPose.from_value(
            {
                "position_m": position,
                "rpy_rad": _lookup(section, "start_rpy_rad", np.zeros(3)),
            }
        )
    pan = _lookup(config, "pan", None)
    initial = _lookup(pan, "initial_pose", None) if pan is not None else None
    return PanPose(0.0, 0.0, 0.0) if initial is None else PanPose.from_value(initial)


def build_fried_rice_cycle_waypoints(
    parameters: FriedRiceParameters,
    config: Any,
    *,
    start_pose: PanPose | Sequence[float] | Mapping[str, Any] | None = None,
    cycle_index: int = 0,
    start_time_s: float = 0.0,
) -> tuple[Waypoint, ...]:
    """한 번의 3단 teaching waypoint를 wok xz 평면에 구성한다."""

    if cycle_index < 0:
        raise FriedRiceTrajectoryError("cycle_index는 0 이상이어야 합니다.")
    if not np.isfinite(start_time_s):
        raise FriedRiceTrajectoryError("start_time_s는 유한해야 합니다.")
    p0 = _configured_start(config) if start_pose is None else PanPose.from_value(start_pose)

    trajectory = trajectory_section(config)
    insertion_angle_deg = _profile_number(
        config,
        "insertion_angle_deg",
        float(_lookup(trajectory, "insertion_angle_deg", 45.0)),
    )
    if not np.isclose(insertion_angle_deg, 45.0, atol=1.0e-12):
        raise FriedRiceTrajectoryError(
            "볶음밥 teaching의 insertion_angle_deg는 현재 45도로 고정됩니다."
        )
    tilt_direction = _profile_number(config, "tilt_direction", -1.0)
    if tilt_direction not in (-1.0, 1.0):
        raise FriedRiceTrajectoryError("tilt_direction은 -1 또는 1이어야 합니다.")

    insertion_angle = np.deg2rad(insertion_angle_deg)
    distance = parameters.insertion_distance
    p1 = PanPose(
        p0.x + distance * np.cos(insertion_angle),
        p0.y,
        p0.z - distance * np.sin(insertion_angle),
        p0.roll,
        p0.pitch,
        p0.yaw,
    )
    # 2단계: 삽입 위치를 유지한 채 기본 -pitch 방향으로 15도 기울인다.
    p2 = PanPose(
        p1.x,
        p1.y,
        p1.z,
        p0.roll,
        p0.pitch + tilt_direction * parameters.tilt_angle,
        p0.yaw,
    )
    # 3단계: P0로 후퇴하면서 pitch도 동시에 원래 teaching 자세로 복원한다.
    p3 = p0
    times = fried_rice_phase_times(parameters) + float(start_time_s)
    return tuple(
        Waypoint(f"P{index}", cycle_index, float(time_s), pose)
        for index, (time_s, pose) in enumerate(zip(times, (p0, p1, p2, p3), strict=True))
    )


def build_repeated_fried_rice_waypoints(
    parameters: FriedRiceParameters,
    config: Any,
    *,
    start_pose: PanPose | Sequence[float] | Mapping[str, Any] | None = None,
    cycles: int | None = None,
) -> WaypointSequence:
    """3단 teaching의 논리적 P0~P3를 phasewise waypoint 열로 연결한다."""

    trajectory = trajectory_section(config)
    profile = fried_rice_section(config)
    raw_cycles = (
        _lookup(profile, "cycles", _lookup(trajectory, "cycles", 5)) if cycles is None else cycles
    )
    try:
        cycle_count = int(raw_cycles)
    except (TypeError, ValueError) as exc:
        raise FriedRiceTrajectoryError("볶음밥 cycles는 정수여야 합니다.") from exc
    if cycle_count < 2 or cycle_count != raw_cycles:
        raise FriedRiceTrajectoryError(
            "global quintic 볶음밥 trajectory에는 2 이상의 정수 cycles가 필요합니다."
        )

    resolved_start = (
        _configured_start(config) if start_pose is None else PanPose.from_value(start_pose)
    )
    joined: list[Waypoint] = []
    for cycle_index in range(cycle_count):
        logical_points = build_fried_rice_cycle_waypoints(
            parameters,
            config,
            start_pose=resolved_start,
            cycle_index=cycle_index,
            start_time_s=cycle_index * parameters.cycle_time,
        )
        joined.extend(logical_points if cycle_index == 0 else logical_points[1:])
    boundaries = np.arange(cycle_count + 1, dtype=float) * parameters.cycle_time
    try:
        return WaypointSequence(tuple(joined), boundaries, cycle_count)
    except WaypointError as exc:
        raise FriedRiceTrajectoryError(str(exc)) from exc


def generate_fried_rice_trajectory(
    action_or_parameters: (Sequence[float] | np.ndarray | Mapping[str, Any] | FriedRiceParameters),
    config: Any,
    *,
    start_pose: PanPose | Sequence[float] | Mapping[str, Any] | None = None,
    cycles: int | None = None,
    sample_rate_hz: float | None = None,
    validate: bool = True,
    strict_action: bool = False,
) -> Trajectory:
    """볶음밥 teaching을 overshoot 없는 phasewise minimum-jerk 궤적으로 만든다."""

    if isinstance(action_or_parameters, FriedRiceParameters):
        parameters = action_or_parameters
    elif isinstance(action_or_parameters, Mapping) or any(
        hasattr(action_or_parameters, name)
        for name in ("insertion_distance", "insertion_distance_m", "tilt_angle")
    ):
        parameters = FriedRiceParameters.from_mapping(action_or_parameters)
    else:
        parameters = map_fried_rice_action(
            action_or_parameters,
            config,
            clip=not strict_action,
        )

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
            atol=1.0e-8,
        ):
            raise SplineGenerationError(
                "start_pose가 teaching/pan에서 해석한 authoritative wok P0와 일치하지 않습니다."
            )

    waypoints = build_repeated_fried_rice_waypoints(
        parameters,
        config,
        start_pose=authoritative_start,
        cycles=cycles,
    )
    spline = PhasewiseMinimumJerkSpline.from_waypoints(waypoints)
    trajectory = trajectory_section(config)
    profile = fried_rice_section(config)
    rate = (
        _lookup(profile, "sample_rate_hz", _lookup(trajectory, "sample_rate_hz", None))
        if sample_rate_hz is None
        else sample_rate_hz
    )
    if rate is None:
        raise SplineGenerationError(
            "trajectory.fried_rice.sample_rate_hz 또는 trajectory.sample_rate_hz가 필요합니다."
        )
    sampled = spline.sample(parameters, float(rate), frame_context)
    if validate:
        from .validator import validate_trajectory

        sampled.validation = validate_trajectory(sampled, config)
    return sampled
