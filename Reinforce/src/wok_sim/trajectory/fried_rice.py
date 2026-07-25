"""볶음밥용 4단 teaching 궤적과 전용 action mapping."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from wok_sim.geometry.transforms import resolve_wok_frame_context, transform_to_pose

from .parameters import trajectory_section
from .spline import (
    GlobalQuinticSpline,
    PhasewiseMinimumJerkSpline,
    PitchHoldGlobalSpline,
    SplineGenerationError,
    Trajectory,
)
from .waypoints import PanPose, Waypoint, WaypointError, WaypointSequence

FRIED_RICE_ACTION_NAMES: tuple[str, ...] = (
    "descent_angle",
    "pan_tilt_angle",
    "descent_speed",
    "lift_angle",
)
"""볶음밥 profile의 normalized action 순서."""


_DEFAULT_RANGES: dict[str, tuple[float, float]] = {
    "descent_angle_range_rad": (np.deg2rad(35.0), np.deg2rad(55.0)),
    "pan_tilt_angle_range_rad": (np.deg2rad(20.0), np.deg2rad(40.0)),
    "descent_speed_range_m_s": (0.38, 0.48),
    "lift_angle_range_rad": (0.0, np.deg2rad(30.0)),
}

_ACTION_RANGE_KEYS: tuple[str, ...] = tuple(_DEFAULT_RANGES)
_MINIMUM_JERK_PEAK_VELOCITY = 1.875
_MINIMUM_JERK_PEAK_ACCELERATION = 5.773502691896258
_MINIMUM_JERK_PEAK_JERK = 60.0
_PHASEWISE_MOTION_PROFILE = "phasewise_minimum_jerk"
_CONTINUOUS_MOTION_PROFILE = "continuous_blended_global_quintic"


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
    """한 에피소드의 4단 teaching을 결정하는 SI 단위 파라미터."""

    descent_angle: float
    pan_tilt_angle: float
    insertion_distance: float = 0.25
    tilt_recovery_angle: float = np.deg2rad(15.0)
    linear_speed: float = 0.43
    angular_speed: float = 1.20
    linear_acceleration_limit: float = 1.40
    angular_acceleration_limit: float = 2.50
    linear_jerk_limit: float = 15.0
    angular_jerk_limit: float = 20.0
    minimum_phase_duration: float = 0.50
    time_scale: float = 1.0
    lift_return_time_factor: float = 1.0
    adaptive_retiming_scale: float = 1.0

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                self.descent_angle,
                self.pan_tilt_angle,
                self.insertion_distance,
                self.tilt_recovery_angle,
                self.linear_speed,
                self.angular_speed,
                self.linear_acceleration_limit,
                self.angular_acceleration_limit,
                self.linear_jerk_limit,
                self.angular_jerk_limit,
                self.minimum_phase_duration,
                self.time_scale,
                self.lift_return_time_factor,
                self.adaptive_retiming_scale,
            ),
            dtype=float,
        )
        if not np.isfinite(values).all():
            raise FriedRiceTrajectoryError("볶음밥 궤적 파라미터에 NaN 또는 inf가 있습니다.")
        if not 0.0 < self.descent_angle < np.pi / 2.0:
            raise FriedRiceTrajectoryError("descent_angle은 0도 초과 90도 미만이어야 합니다.")
        if self.pan_tilt_angle <= 0.0:
            raise FriedRiceTrajectoryError("pan_tilt_angle은 양수여야 합니다.")
        if not 0.0 <= self.tilt_recovery_angle <= np.pi / 2.0:
            raise FriedRiceTrajectoryError("tilt_recovery_angle은 0도 이상 90도 이하여야 합니다.")
        positive_values = np.asarray(
            (
                self.insertion_distance,
                self.linear_speed,
                self.angular_speed,
                self.linear_acceleration_limit,
                self.angular_acceleration_limit,
                self.linear_jerk_limit,
                self.angular_jerk_limit,
                self.minimum_phase_duration,
                self.time_scale,
                self.lift_return_time_factor,
                self.adaptive_retiming_scale,
            ),
            dtype=float,
        )
        if np.any(positive_values <= 0.0):
            raise FriedRiceTrajectoryError(
                "거리, 속도, 가속도, jerk와 phase duration은 양수여야 합니다."
            )
        if self.lift_return_time_factor > 1.0:
            raise FriedRiceTrajectoryError(
                "lift_return_time_factor는 0 초과 1 이하여야 합니다."
            )

    @property
    def insertion_angle(self) -> float:
        """기존 이름과 호환되는 하강 경로 각도(rad)."""

        return self.descent_angle

    @property
    def tilt_angle(self) -> float:
        """기존 이름과 호환되는 최대 팬 pitch(rad)."""

        return self.pan_tilt_angle

    @property
    def insertion_distance_m(self) -> float:
        return self.insertion_distance

    @property
    def descent_angle_rad(self) -> float:
        return self.descent_angle

    @property
    def pan_tilt_angle_rad(self) -> float:
        return self.pan_tilt_angle

    @property
    def tilt_angle_rad(self) -> float:
        return self.pan_tilt_angle

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
    def phase_durations_s(self) -> tuple[float, float, float, float]:
        """선행 tilt, 하강, 학습 lift 회전, 위치·각도 원복 시간을 계산한다."""

        tilt_out = self._minimum_jerk_duration(
            self.pan_tilt_angle,
            self.angular_speed,
            self.angular_acceleration_limit,
            self.angular_jerk_limit,
            self.minimum_phase_duration,
        )
        descent = self._minimum_jerk_duration(
            self.insertion_distance,
            self.linear_speed,
            self.linear_acceleration_limit,
            self.linear_jerk_limit,
            self.minimum_phase_duration,
        )
        partial_recovery = self._minimum_jerk_duration(
            self.tilt_recovery_angle,
            self.angular_speed,
            self.angular_acceleration_limit,
            self.angular_jerk_limit,
            self.minimum_phase_duration,
        )
        remaining_tilt = abs(self.pan_tilt_angle - self.tilt_recovery_angle)
        final_recovery = self._minimum_jerk_duration(
            remaining_tilt,
            self.angular_speed,
            self.angular_acceleration_limit,
            self.angular_jerk_limit,
            self.minimum_phase_duration,
        )
        phase_durations = (
            tilt_out,
            descent,
            partial_recovery * self.lift_return_time_factor,
            max(descent, final_recovery) * self.lift_return_time_factor,
        )
        effective_scale = self.time_scale * self.adaptive_retiming_scale
        return tuple(duration * effective_scale for duration in phase_durations)

    @property
    def cycle_time(self) -> float:
        return float(sum(self.phase_durations_s))

    @property
    def cycle_time_s(self) -> float:
        return self.cycle_time

    @property
    def insert_phase_ratio(self) -> float:
        return self.phase_durations_s[1] / self.cycle_time

    @property
    def tilt_phase_ratio(self) -> float:
        return self.phase_durations_s[0] / self.cycle_time

    def as_array(self) -> np.ndarray:
        """``FRIED_RICE_ACTION_NAMES`` 순서의 학습 대상 물리값을 반환한다."""

        return np.asarray(
            (
                self.descent_angle,
                self.pan_tilt_angle,
                self.linear_speed,
                self.tilt_recovery_angle,
            ),
            dtype=float,
        )

    def as_dict(self) -> dict[str, float]:
        """학습 대상 각도와 고정 teaching 값을 함께 반환한다."""

        tilt_out_duration, descent_duration, recovery_duration, return_duration = (
            self.phase_durations_s
        )
        return {
            **dict(zip(FRIED_RICE_ACTION_NAMES, self.as_array(), strict=True)),
            "insertion_distance": self.insertion_distance,
            "tilt_recovery_angle": self.tilt_recovery_angle,
            "linear_speed": self.linear_speed,
            "angular_speed": self.angular_speed,
            "cycle_time": self.cycle_time,
            "tilt_out_phase_duration": tilt_out_duration,
            "descent_phase_duration": descent_duration,
            "partial_recovery_phase_duration": recovery_duration,
            "return_phase_duration": return_duration,
            "insert_phase_duration": descent_duration,
            "tilt_phase_duration": tilt_out_duration,
            "insert_phase_ratio": self.insert_phase_ratio,
            "tilt_phase_ratio": self.tilt_phase_ratio,
            "linear_acceleration_limit": self.linear_acceleration_limit,
            "angular_acceleration_limit": self.angular_acceleration_limit,
            "linear_jerk_limit": self.linear_jerk_limit,
            "angular_jerk_limit": self.angular_jerk_limit,
            "time_scale": self.time_scale,
            "lift_return_time_factor": self.lift_return_time_factor,
            "adaptive_retiming_scale": self.adaptive_retiming_scale,
            "effective_time_scale": self.time_scale * self.adaptive_retiming_scale,
        }

    @classmethod
    def teaching_default(cls) -> FriedRiceParameters:
        """45도 하강·30도 팬 tilt·15도 lift 회전의 중앙 teaching."""

        return cls(
            descent_angle=np.deg2rad(45.0),
            pan_tilt_angle=np.deg2rad(30.0),
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | Any) -> FriedRiceParameters:
        """SI 단위 mapping 또는 동일 attribute를 가진 객체에서 생성한다."""

        aliases = {
            "descent_angle": ("descent_angle", "descent_angle_rad", "insertion_angle"),
            "pan_tilt_angle": ("pan_tilt_angle", "pan_tilt_angle_rad", "tilt_angle"),
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
            "insertion_distance": (("insertion_distance", "insertion_distance_m"), 0.25),
            "tilt_recovery_angle": (
                (
                    "lift_angle",
                    "lift_angle_rad",
                    "tilt_recovery_angle",
                    "tilt_recovery_angle_rad",
                ),
                np.deg2rad(15.0),
            ),
            "linear_speed": (
                ("linear_speed", "linear_speed_m_s", "descent_speed", "descent_speed_m_s"),
                0.43,
            ),
            "angular_speed": (("angular_speed", "angular_speed_rad_s"), 1.20),
            "linear_acceleration_limit": (("linear_acceleration_limit",), 1.40),
            "angular_acceleration_limit": (("angular_acceleration_limit",), 2.50),
            "linear_jerk_limit": (("linear_jerk_limit",), 15.0),
            "angular_jerk_limit": (("angular_jerk_limit",), 20.0),
            "minimum_phase_duration": (("minimum_phase_duration",), 0.50),
            "time_scale": (("time_scale", "continuous_time_scale"), 1.0),
            "lift_return_time_factor": (("lift_return_time_factor",), 1.0),
            "adaptive_retiming_scale": (("adaptive_retiming_scale",), 1.0),
        }
        for name, (candidates, default) in optional.items():
            raw = default
            for candidate in candidates:
                candidate_value = _lookup(values, candidate, None)
                if candidate_value is not None:
                    raw = candidate_value
                    break
            resolved[name] = float(raw)
        return cls(**resolved)


def map_fried_rice_action(
    action: Sequence[float] | np.ndarray,
    config: Any,
    *,
    clip: bool = True,
) -> FriedRiceParameters:
    """4D action을 하강 각도, 최대 팬 tilt, 하강 속도, lift 회전량으로 변환한다."""

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
        descent_angle=physical[0],
        pan_tilt_angle=physical[1],
        linear_speed=physical[2],
        tilt_recovery_angle=physical[3],
        insertion_distance=_profile_number(config, "insertion_distance_m", 0.25),
        angular_speed=_profile_number(config, "angular_speed_rad_s", 1.20),
        linear_acceleration_limit=_profile_number(
            config,
            "linear_acceleration_limit_m_s2",
            1.40,
        ),
        angular_acceleration_limit=_profile_number(
            config,
            "angular_acceleration_limit_rad_s2",
            2.50,
        ),
        linear_jerk_limit=_profile_number(config, "linear_jerk_limit_m_s3", 15.0),
        angular_jerk_limit=_profile_number(config, "angular_jerk_limit_rad_s3", 20.0),
        minimum_phase_duration=_profile_number(
            config,
            "minimum_phase_duration_s",
            0.50,
        ),
        time_scale=_profile_number(config, "continuous_time_scale", 1.0),
        lift_return_time_factor=_profile_number(
            config,
            "lift_return_time_factor",
            1.0,
        ),
    )


def _adaptive_retiming_settings(
    config: Any,
) -> tuple[bool, float, float, bool, float, float, tuple[float, ...]]:
    """Offline Cartesian cap 기반 전역 retiming 설정을 반환한다."""

    profile = fried_rice_section(config)
    enabled = bool(_lookup(profile, "adaptive_robot_retiming", False))
    cap_fraction = _profile_number(config, "adaptive_robot_cap_fraction", 1.0)
    jerk_cap_fraction = _profile_number(
        config,
        "adaptive_robot_jerk_cap_fraction",
        cap_fraction,
    )
    allow_speedup = bool(_lookup(profile, "adaptive_allow_speedup", False))
    minimum_scale = _profile_number(config, "adaptive_min_time_scale", 0.25)
    maximum_scale = _profile_number(config, "adaptive_max_time_scale", 3.0)
    robot = _lookup(config, "robot", {})
    caps = _lookup(robot, "cartesian_caps", {})
    keys = (
        "linear_velocity_m_s",
        "angular_velocity_rad_s",
        "linear_acceleration_m_s2",
        "angular_acceleration_rad_s2",
        "linear_jerk_m_s3",
        "angular_jerk_rad_s3",
    )
    try:
        values = tuple(float(_lookup(caps, key, np.nan)) for key in keys)
    except (TypeError, ValueError) as exc:
        raise FriedRiceTrajectoryError("robot.cartesian_caps 값이 숫자가 아닙니다.") from exc
    if enabled and (
        not 0.0 < cap_fraction <= 1.0
        or not 0.0 < jerk_cap_fraction <= 1.0
        or not 0.0 < minimum_scale <= 1.0
        or maximum_scale < 1.0
        or not np.isfinite(values).all()
        or any(value <= 0.0 for value in values)
    ):
        raise FriedRiceTrajectoryError(
            "adaptive retiming에는 유효한 cap fraction, max scale 및 "
            "robot.cartesian_caps 6개가 필요합니다."
        )
    return (
        enabled,
        cap_fraction,
        jerk_cap_fraction,
        allow_speedup,
        minimum_scale,
        maximum_scale,
        values,
    )


def _required_adaptive_scale(
    trajectory: Trajectory,
    *,
    cap_fraction: float,
    jerk_cap_fraction: float,
    caps: tuple[float, ...],
) -> float:
    """속도/가속도/jerk cap을 만족시키는 추가 uniform time scale."""

    arrays = (
        trajectory.linear_velocity_m_s,
        trajectory.angular_velocity_rad_s,
        trajectory.linear_acceleration_m_s2,
        trajectory.angular_acceleration_rad_s2,
        trajectory.linear_jerk_m_s3,
        trajectory.angular_jerk_rad_s3,
    )
    peaks = tuple(float(np.max(np.linalg.norm(array, axis=1))) for array in arrays)
    orders = (1.0, 1.0, 2.0, 2.0, 3.0, 3.0)
    fractions = (
        cap_fraction,
        cap_fraction,
        cap_fraction,
        cap_fraction,
        jerk_cap_fraction,
        jerk_cap_fraction,
    )
    requirements = [
        (peak / (cap * fraction)) ** (1.0 / order)
        for peak, cap, order, fraction in zip(
            peaks,
            caps,
            orders,
            fractions,
            strict=True,
        )
    ]
    return max(requirements)


def _joint_speed_retiming_settings(config: Any) -> Mapping[str, Any] | None:
    """Return the optional nominal M0609 joint/TCP speed retiming mapping."""

    robot = _lookup(config, "robot", {})
    raw = _lookup(robot, "nominal_joint_speed_retiming", None)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise FriedRiceTrajectoryError(
            "robot.nominal_joint_speed_retiming은 mapping이어야 합니다."
        )
    return raw if bool(raw.get("enabled", False)) else None


def fried_rice_phase_times(parameters: FriedRiceParameters) -> np.ndarray:
    """P0 시작부터 선행 tilt, 하강, 학습 lift 회전, 원복까지의 상대 시각."""

    tilt_out, descent, partial_recovery, _ = parameters.phase_durations_s
    times = np.asarray(
        (
            0.0,
            tilt_out,
            tilt_out + descent,
            tilt_out + descent + partial_recovery,
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


def _motion_profile(config: Any) -> str:
    raw = _lookup(
        fried_rice_section(config),
        "motion_profile",
        _PHASEWISE_MOTION_PROFILE,
    )
    profile = str(raw).strip().lower()
    if profile not in {
        _PHASEWISE_MOTION_PROFILE,
        _CONTINUOUS_MOTION_PROFILE,
    }:
        raise FriedRiceTrajectoryError(
            "trajectory.fried_rice.motion_profile은 "
            f"'{_PHASEWISE_MOTION_PROFILE}' 또는 '{_CONTINUOUS_MOTION_PROFILE}'이어야 합니다."
        )
    return profile


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
    restore_pitch_at_cycle_end: bool = True,
) -> tuple[Waypoint, ...]:
    """선행 tilt를 포함한 4단 teaching waypoint를 wok xz 평면에 구성한다."""

    if cycle_index < 0:
        raise FriedRiceTrajectoryError("cycle_index는 0 이상이어야 합니다.")
    if not np.isfinite(start_time_s):
        raise FriedRiceTrajectoryError("start_time_s는 유한해야 합니다.")
    if not isinstance(restore_pitch_at_cycle_end, bool):
        raise FriedRiceTrajectoryError("restore_pitch_at_cycle_end는 bool이어야 합니다.")
    p0 = _configured_start(config) if start_pose is None else PanPose.from_value(start_pose)

    tilt_direction = _profile_number(config, "tilt_direction", 1.0)
    if tilt_direction not in (-1.0, 1.0):
        raise FriedRiceTrajectoryError("tilt_direction은 -1 또는 1이어야 합니다.")

    descent_angle = parameters.descent_angle
    distance = parameters.insertion_distance
    maximum_pitch = tilt_direction * parameters.pan_tilt_angle
    residual_pitch = tilt_direction * (parameters.pan_tilt_angle - parameters.tilt_recovery_angle)
    descent_x = distance * np.cos(descent_angle)
    descent_z = -distance * np.sin(descent_angle)
    if _motion_profile(config) == _CONTINUOUS_MOTION_PROFILE:
        # 실제 웍질처럼 translation과 rotation을 겹친다. P1에서 이미
        # 하강 중이고 전역 quintic이 이 점들을 C4로 연결해 phase 경계의
        # 정지를 없앤다.
        pretilt_progress = _profile_number(config, "pretilt_translation_fraction", 0.20)
        return_progress = _profile_number(config, "lift_return_translation_fraction", 0.65)
        return_arc_height = _profile_number(config, "lift_return_arc_height_m", 0.025)
        hold_insertion_pitch = bool(
            _lookup(
                fried_rice_section(config),
                "hold_insertion_pitch_during_lift_retreat",
                False,
            )
        )
        if not 0.0 < pretilt_progress < return_progress < 1.0:
            raise FriedRiceTrajectoryError(
                "continuous motion의 translation fraction은 "
                "0 < pretilt < lift_return < 1이어야 합니다."
            )
        if return_arc_height <= 0.0:
            raise FriedRiceTrajectoryError(
                "trajectory.fried_rice.lift_return_arc_height_m은 양수여야 합니다."
            )
        p1 = PanPose(
            p0.x + pretilt_progress * descent_x,
            p0.y,
            p0.z + pretilt_progress * descent_z,
            p0.roll,
            p0.pitch + maximum_pitch,
            p0.yaw,
        )
        p2 = PanPose(
            p0.x + descent_x,
            p0.y,
            p0.z + descent_z,
            p0.roll,
            p0.pitch + maximum_pitch,
            p0.yaw,
        )
        if hold_insertion_pitch:
            # 학습 lift angle은 팬 pitch 복원량이 아니라 P2→P3의 직선 복귀
            # 경로에 더하는 상승각이다. 0°면 직선 복귀, 30°면 같은 수평
            # 후퇴 거리에서 더 높은 arc를 만들며 팬 pitch는 그대로 유지한다.
            retreat_distance_x = abs((1.0 - return_progress) * descent_x)
            p3_z = (
                p0.z
                + return_progress * descent_z
                + retreat_distance_x * np.tan(parameters.tilt_recovery_angle)
            )
            p3_pitch = p0.pitch + maximum_pitch
        else:
            p3_z = p0.z + return_progress * descent_z + return_arc_height
            p3_pitch = p0.pitch + residual_pitch
        p3 = PanPose(
            p0.x + return_progress * descent_x,
            p0.y,
            p3_z,
            p0.roll,
            p3_pitch,
            p0.yaw,
        )
    else:
        # 호환 profile: 원래 위치에서 tilt한 뒤 위치 고정 lift를 수행한다.
        p1 = PanPose(
            p0.x,
            p0.y,
            p0.z,
            p0.roll,
            p0.pitch + maximum_pitch,
            p0.yaw,
        )
        p2 = PanPose(
            p0.x + descent_x,
            p0.y,
            p0.z + descent_z,
            p0.roll,
            p0.pitch + maximum_pitch,
            p0.yaw,
        )
        p3 = PanPose(
            p2.x,
            p2.y,
            p2.z,
            p0.roll,
            p0.pitch + residual_pitch,
            p0.yaw,
        )
    # 중간 cycle의 후퇴에서는 삽입 pitch를 유지해 다음 삽입과 연속으로
    # 연결한다. 전체 반복의 마지막 cycle에서만 teaching pitch로 복원한다.
    p4 = (
        p0
        if restore_pitch_at_cycle_end
        else PanPose(
            p0.x,
            p0.y,
            p0.z,
            p0.roll,
            p0.pitch + maximum_pitch,
            p0.yaw,
        )
    )
    times = fried_rice_phase_times(parameters) + float(start_time_s)
    poses = (p0, p1, p2, p3, p4)
    return tuple(
        Waypoint(f"P{index}", cycle_index, float(time_s), pose)
        for index, (time_s, pose) in enumerate(zip(times, poses, strict=True))
    )


def build_repeated_fried_rice_waypoints(
    parameters: FriedRiceParameters,
    config: Any,
    *,
    start_pose: PanPose | Sequence[float] | Mapping[str, Any] | None = None,
    cycles: int | None = None,
) -> WaypointSequence:
    """4단 teaching의 논리적 P0~P4를 phasewise waypoint 열로 연결한다."""

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
    hold_insertion_pitch_between_cycles = bool(
        _lookup(profile, "hold_insertion_pitch_between_cycles", False)
    )
    joined: list[Waypoint] = []
    for cycle_index in range(cycle_count):
        logical_points = build_fried_rice_cycle_waypoints(
            parameters,
            config,
            start_pose=resolved_start,
            cycle_index=cycle_index,
            start_time_s=cycle_index * parameters.cycle_time,
            restore_pitch_at_cycle_end=(
                not hold_insertion_pitch_between_cycles
                or cycle_index == cycle_count - 1
            ),
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
    """볶음밥 teaching을 설정된 phasewise 또는 연속 blended 궤적으로 만든다."""

    if isinstance(action_or_parameters, FriedRiceParameters):
        parameters = action_or_parameters
    elif isinstance(action_or_parameters, Mapping) or any(
        hasattr(action_or_parameters, name)
        for name in ("descent_angle", "descent_angle_rad", "pan_tilt_angle")
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

    motion_profile = _motion_profile(config)
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

    def sample(resolved_parameters: FriedRiceParameters) -> Trajectory:
        waypoints = build_repeated_fried_rice_waypoints(
            resolved_parameters,
            config,
            start_pose=authoritative_start,
            cycles=cycles,
        )
        if motion_profile == _CONTINUOUS_MOTION_PROFILE:
            hold_insertion_pitch = bool(
                _lookup(
                    profile,
                    "hold_insertion_pitch_during_lift_retreat",
                    False,
                )
            )
            spline = (
                PitchHoldGlobalSpline.from_waypoints(waypoints)
                if hold_insertion_pitch
                else GlobalQuinticSpline.from_waypoints(waypoints)
            )
        else:
            spline = PhasewiseMinimumJerkSpline.from_waypoints(waypoints)
        return spline.sample(resolved_parameters, float(rate), frame_context)

    sampled = sample(parameters)
    (
        retiming_enabled,
        cap_fraction,
        jerk_cap_fraction,
        allow_speedup,
        minimum_scale,
        maximum_scale,
        caps,
    ) = _adaptive_retiming_settings(config)
    if retiming_enabled:
        # Geometry/action 범위는 그대로 두고 시간만 늘린다. 0.875 lift-return
        # target을 먼저 시도한 뒤, 해당 action이 offline M0609 Cartesian gate를
        # 넘는 경우에만 전체 cycle을 필요한 만큼 느리게 한다.
        for _ in range(3):
            required = _required_adaptive_scale(
                sampled,
                cap_fraction=cap_fraction,
                jerk_cap_fraction=jerk_cap_fraction,
                caps=caps,
            )
            if (required < 1.0 and not allow_speedup) or abs(required - 1.0) <= 0.002:
                break
            next_scale = max(
                minimum_scale,
                parameters.adaptive_retiming_scale * required * 1.001,
            )
            if next_scale > maximum_scale:
                raise SplineGenerationError(
                    "adaptive retiming에 필요한 추가 time scale "
                    f"{next_scale:.6g}가 adaptive_max_time_scale="
                    f"{maximum_scale:.6g}를 초과합니다."
                )
            if np.isclose(
                next_scale,
                parameters.adaptive_retiming_scale,
                rtol=0.0,
                atol=1.0e-9,
            ):
                break
            parameters = replace(
                parameters,
                adaptive_retiming_scale=next_scale,
            )
            sampled = sample(parameters)
        else:
            raise SplineGenerationError("adaptive retiming이 3회 안에 수렴하지 않았습니다.")
    joint_speed_settings = _joint_speed_retiming_settings(config)
    if joint_speed_settings is not None:
        from wok_sim.robot.m0609_joint_speed import (
            M0609JointSpeedError,
            M0609JointSpeedSettings,
            evaluate_m0609_joint_speeds,
        )

        try:
            resolved_joint_settings = M0609JointSpeedSettings.from_mapping(
                joint_speed_settings
            )
            joint_report = None
            for _ in range(4):
                joint_report = evaluate_m0609_joint_speeds(
                    sampled,
                    resolved_joint_settings,
                )
                required = joint_report.required_uniform_time_scale
                if abs(required - 1.0) <= 0.002:
                    break
                next_scale = parameters.adaptive_retiming_scale * required * 1.001
                if next_scale < resolved_joint_settings.minimum_time_scale:
                    raise SplineGenerationError(
                        "M0609 joint speed retiming에 필요한 추가 time scale "
                        f"{next_scale:.6g}가 minimum_time_scale="
                        f"{resolved_joint_settings.minimum_time_scale:.6g}보다 작습니다."
                    )
                if next_scale > resolved_joint_settings.maximum_time_scale:
                    raise SplineGenerationError(
                        "M0609 joint speed retiming에 필요한 추가 time scale "
                        f"{next_scale:.6g}가 maximum_time_scale="
                        f"{resolved_joint_settings.maximum_time_scale:.6g}보다 큽니다."
                    )
                parameters = replace(
                    parameters,
                    adaptive_retiming_scale=next_scale,
                )
                sampled = sample(parameters)
            else:
                raise SplineGenerationError(
                    "M0609 nominal joint speed retiming이 4회 안에 수렴하지 않았습니다."
                )
            if joint_report is None:
                raise SplineGenerationError("M0609 joint speed report를 만들지 못했습니다.")
            sampled.joint_speed_report = joint_report.summary()
        except M0609JointSpeedError as exc:
            raise SplineGenerationError(f"M0609 joint speed retiming 실패: {exc}") from exc
    if validate:
        from .validator import validate_trajectory

        sampled.validation = validate_trajectory(sampled, config)
    return sampled
