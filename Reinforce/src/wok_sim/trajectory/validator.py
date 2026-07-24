"""샘플 기반 Cartesian/평면/workspace 궤적 제한 검사."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .parameters import trajectory_section


class TrajectoryValidationError(ValueError):
    """검사할 trajectory 자체 또는 limit 설정이 잘못된 경우."""


def _lookup(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _positive_limit(section: Any, key: str) -> float | None:
    raw = _lookup(section, key, None)
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        raw = _lookup(raw, "linear", _lookup(raw, "translation", None))
        if raw is None:
            return None
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise TrajectoryValidationError(f"trajectory.{key}는 숫자여야 합니다.") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise TrajectoryValidationError(f"trajectory.{key}는 양의 유한값이어야 합니다.")
    return value


def _angular_limit(section: Any, derivative_key: str) -> float | None:
    """별도 angular limit 또는 Cartesian limit mapping의 angular 값을 읽는다."""

    explicit = _lookup(section, f"max_angular_{derivative_key}", None)
    if explicit is not None:
        proxy = {f"max_angular_{derivative_key}": explicit}
        return _positive_limit(proxy, f"max_angular_{derivative_key}")
    cartesian = _lookup(section, f"max_cartesian_{derivative_key}", None)
    if not isinstance(cartesian, Mapping):
        return None
    raw = _lookup(cartesian, "angular", _lookup(cartesian, "rotation", None))
    if raw is None:
        return None
    proxy = {f"max_angular_{derivative_key}": raw}
    return _positive_limit(proxy, f"max_angular_{derivative_key}")


def _nonnegative(section: Any, key: str, default: float) -> float:
    raw = _lookup(section, key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise TrajectoryValidationError(f"trajectory.{key}는 숫자여야 합니다.") from exc
    if not np.isfinite(value) or value < 0.0:
        raise TrajectoryValidationError(f"trajectory.{key}는 0 이상의 유한값이어야 합니다.")
    return value


def _range(value: Any, name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    try:
        low, high = float(value[0]), float(value[1])
    except (IndexError, TypeError, ValueError) as exc:
        raise TrajectoryValidationError(f"{name}은 [min, max] 형식이어야 합니다.") from exc
    if not np.isfinite([low, high]).all() or low > high:
        raise TrajectoryValidationError(f"{name} 범위가 유효하지 않습니다: {value!r}")
    return low, high


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """궤적 제한 검사 결과."""

    valid: bool
    violations: tuple[str, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)
    messages: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        """``valid``의 설명적인 alias."""

        return self.valid

    def __bool__(self) -> bool:
        return self.valid

    def __getitem__(self, key: str) -> Any:
        """dict 스타일 consumer를 위한 최소 호환 접근."""

        if key == "valid":
            return self.valid
        if key == "violations":
            return self.violations
        if key == "metrics":
            return self.metrics
        if key == "messages":
            return self.messages
        raise KeyError(key)

    def as_dict(self) -> dict[str, Any]:
        """JSON/logging에 적합한 plain dict를 반환한다."""

        return {
            "valid": self.valid,
            "violations": list(self.violations),
            "messages": list(self.messages),
            "metrics": dict(self.metrics),
        }


def _trajectory_arrays(trajectory: Any) -> tuple[np.ndarray, ...]:
    # Cartesian workspace와 xz/roll/yaw 평면 제약은 wok-local frame에서
    # 정의된다. 변환된 trajectory가 local 배열을 제공하면 반드시 그것을
    # 사용하고, 이전 adapter에는 base/public 배열로 fallback한다.
    name_options = (
        ("time_s",),
        ("position_wok_m", "position_m"),
        ("orientation_wok_rpy_rad", "orientation_rpy_rad"),
        ("linear_velocity_wok_m_s", "linear_velocity_m_s"),
        ("angular_velocity_wok_rad_s", "angular_velocity_rad_s"),
        ("linear_acceleration_wok_m_s2", "linear_acceleration_m_s2"),
        ("angular_acceleration_wok_rad_s2", "angular_acceleration_rad_s2"),
        ("linear_jerk_wok_m_s3", "linear_jerk_m_s3"),
        ("angular_jerk_wok_rad_s3", "angular_jerk_rad_s3"),
    )
    try:
        arrays = tuple(
            np.asarray(
                next(getattr(trajectory, name) for name in options if hasattr(trajectory, name)),
                dtype=float,
            )
            for options in name_options
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise TrajectoryValidationError("trajectory 필수 배열을 읽을 수 없습니다.") from exc
    time = arrays[0]
    count = time.size
    if time.shape != (count,) or count < 2:
        raise TrajectoryValidationError(
            "trajectory.time_s는 길이 2 이상의 1차원 배열이어야 합니다."
        )
    if any(array.shape != (count, 3) for array in arrays[1:]):
        raise TrajectoryValidationError(
            "trajectory pose/미분 배열 shape은 모두 (N, 3)이어야 합니다."
        )
    return arrays


def validate_trajectory(trajectory: Any, config: Any) -> ValidationResult:
    """샘플된 전체 궤적의 overshoot, 미분 limit, 평면 제약을 검사한다."""

    section = trajectory_section(config)
    arrays = _trajectory_arrays(trajectory)
    (
        time,
        position,
        orientation,
        linear_velocity,
        angular_velocity,
        linear_acceleration,
        angular_acceleration,
        linear_jerk,
        angular_jerk,
    ) = arrays

    violations: list[str] = []
    messages: list[str] = []

    def add(code: str, message: str) -> None:
        if code not in violations:
            violations.append(code)
            messages.append(message)

    if not all(np.isfinite(array).all() for array in arrays):
        add("non_finite", "trajectory에 NaN 또는 inf가 포함되어 있습니다.")
        return ValidationResult(False, tuple(violations), {}, tuple(messages))
    if np.any(np.diff(time) <= 0.0):
        add("non_monotonic_time", "timestamp가 엄격히 증가하지 않습니다.")

    speed = np.linalg.norm(linear_velocity, axis=1)
    angular_speed = np.linalg.norm(angular_velocity, axis=1)
    acceleration = np.linalg.norm(linear_acceleration, axis=1)
    angular_acceleration_norm = np.linalg.norm(angular_acceleration, axis=1)
    jerk = np.linalg.norm(linear_jerk, axis=1)
    angular_jerk_norm = np.linalg.norm(angular_jerk, axis=1)
    metrics = {
        "duration_s": float(time[-1] - time[0]),
        "x_min_m": float(np.min(position[:, 0])),
        "x_max_m": float(np.max(position[:, 0])),
        "z_min_m": float(np.min(position[:, 2])),
        "z_max_m": float(np.max(position[:, 2])),
        "pitch_min_rad": float(np.min(orientation[:, 1])),
        "pitch_max_rad": float(np.max(orientation[:, 1])),
        "max_cartesian_velocity": float(np.max(speed)),
        "max_angular_velocity": float(np.max(angular_speed)),
        "max_cartesian_acceleration": float(np.max(acceleration)),
        "max_angular_acceleration": float(np.max(angular_acceleration_norm)),
        "max_cartesian_jerk": float(np.max(jerk)),
        "max_angular_jerk": float(np.max(angular_jerk_norm)),
        "max_y_error": float(np.max(np.abs(position[:, 1] - position[0, 1]))),
        "max_roll_error": float(np.max(np.abs(orientation[:, 0] - orientation[0, 0]))),
        "max_yaw_error": float(np.max(np.abs(orientation[:, 2] - orientation[0, 2]))),
    }

    velocity_limit = _positive_limit(section, "max_cartesian_velocity")
    acceleration_limit = _positive_limit(section, "max_cartesian_acceleration")
    jerk_limit = _positive_limit(section, "max_cartesian_jerk")
    angular_velocity_limit = _angular_limit(section, "velocity")
    angular_acceleration_limit = _angular_limit(section, "acceleration")
    angular_jerk_limit = _angular_limit(section, "jerk")
    limit_specs = (
        ("max_cartesian_velocity", metrics["max_cartesian_velocity"], velocity_limit),
        (
            "max_cartesian_acceleration",
            metrics["max_cartesian_acceleration"],
            acceleration_limit,
        ),
        ("max_cartesian_jerk", metrics["max_cartesian_jerk"], jerk_limit),
        (
            "max_angular_velocity",
            metrics["max_angular_velocity"],
            angular_velocity_limit,
        ),
        (
            "max_angular_acceleration",
            metrics["max_angular_acceleration"],
            angular_acceleration_limit,
        ),
        ("max_angular_jerk", metrics["max_angular_jerk"], angular_jerk_limit),
    )
    for code, measured, limit in limit_specs:
        if limit is not None and measured > limit:
            add(code, f"{code}={measured:.9g}, limit={limit:.9g}")

    y_tolerance = _nonnegative(section, "y_tolerance", 1.0e-9)
    roll_tolerance = _nonnegative(section, "roll_tolerance", 1.0e-9)
    yaw_tolerance = _nonnegative(section, "yaw_tolerance", 1.0e-9)
    for code, measured, tolerance in (
        ("y_planar_constraint", metrics["max_y_error"], y_tolerance),
        ("roll_planar_constraint", metrics["max_roll_error"], roll_tolerance),
        ("yaw_planar_constraint", metrics["max_yaw_error"], yaw_tolerance),
    ):
        if measured > tolerance:
            add(code, f"{code} error={measured:.9g}, tolerance={tolerance:.9g}")

    workspace = _lookup(section, "workspace", None)
    workspace = {} if workspace is None else workspace
    ranges = {
        "workspace_x": (
            position[:, 0],
            _range(
                _lookup(workspace, "x_m", _lookup(section, "x_range_m", None)),
                "trajectory.workspace.x_m",
            ),
        ),
        "workspace_y": (
            position[:, 1],
            _range(
                _lookup(workspace, "y_m", _lookup(section, "y_range_m", None)),
                "trajectory.workspace.y_m",
            ),
        ),
        "workspace_z": (
            position[:, 2],
            _range(
                _lookup(workspace, "z_m", _lookup(section, "z_range_m", None)),
                "trajectory.workspace.z_m",
            ),
        ),
        "workspace_pitch": (
            orientation[:, 1],
            _range(
                _lookup(
                    workspace,
                    "pitch_rad",
                    _lookup(section, "pitch_range_rad", None),
                ),
                "trajectory.workspace.pitch_rad",
            ),
        ),
    }
    for code, (values, bounds) in ranges.items():
        if bounds is None:
            continue
        observed_min = float(np.min(values))
        observed_max = float(np.max(values))
        if observed_min < bounds[0] or observed_max > bounds[1]:
            add(
                code,
                f"{code} observed=[{observed_min:.9g}, {observed_max:.9g}], "
                f"allowed=[{bounds[0]:.9g}, {bounds[1]:.9g}]",
            )

    endpoint_velocity_tolerance = _nonnegative(section, "endpoint_velocity_tolerance", 1.0e-8)
    endpoint_acceleration_tolerance = _nonnegative(
        section, "endpoint_acceleration_tolerance", 1.0e-7
    )
    endpoint_speed = np.linalg.norm(
        np.column_stack((linear_velocity[[0, -1]], angular_velocity[[0, -1]])),
        axis=1,
    )
    endpoint_acceleration = np.linalg.norm(
        np.column_stack((linear_acceleration[[0, -1]], angular_acceleration[[0, -1]])),
        axis=1,
    )
    metrics["max_endpoint_velocity"] = float(np.max(endpoint_speed))
    metrics["max_endpoint_acceleration"] = float(np.max(endpoint_acceleration))
    if metrics["max_endpoint_velocity"] > endpoint_velocity_tolerance:
        add("endpoint_velocity", "trajectory 시작/끝 velocity가 0이 아닙니다.")
    if metrics["max_endpoint_acceleration"] > endpoint_acceleration_tolerance:
        add("endpoint_acceleration", "trajectory 시작/끝 acceleration이 0이 아닙니다.")

    waypoints = getattr(trajectory, "waypoints", None)
    boundaries = getattr(waypoints, "cycle_boundaries_s", None)
    if boundaries is not None and len(boundaries) > 2:
        intermediate = np.asarray(boundaries[1:-1], dtype=float)
        try:
            boundary_velocity = np.asarray(trajectory.evaluate(intermediate, derivative=1))
        except (AttributeError, TypeError, ValueError):
            indices = np.asarray([np.argmin(np.abs(time - value)) for value in intermediate])
            boundary_velocity = np.column_stack(
                (linear_velocity[indices], angular_velocity[indices])
            )
        boundary_speed = np.linalg.norm(boundary_velocity, axis=1)
        metrics["min_cycle_boundary_velocity"] = float(np.min(boundary_speed))
        dwell_tolerance = _nonnegative(section, "cycle_dwell_velocity_tolerance", 1.0e-8)
        allow_cycle_boundary_stop = bool(section.get("allow_cycle_boundary_stop", False))
        if not allow_cycle_boundary_stop and np.any(boundary_speed <= dwell_tolerance):
            add(
                "cycle_boundary_dwell",
                "중간 cycle 경계에서 불필요한 정지가 발생했습니다.",
            )

    return ValidationResult(
        not violations,
        tuple(violations),
        metrics,
        tuple(messages),
    )


def trajectory_is_valid(trajectory: Any, config: Any) -> bool:
    """간단한 boolean validation helper."""

    return validate_trajectory(trajectory, config).valid
