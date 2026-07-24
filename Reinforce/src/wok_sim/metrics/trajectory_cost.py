"""궤적 acceleration 및 jerk 적분 cost."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _time_vector(timestamps_s: ArrayLike) -> NDArray[np.float64]:
    times = np.asarray(timestamps_s, dtype=np.float64)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("timestamps_s must be a one-dimensional array with >= 2 values")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("timestamps_s must be finite and strictly increasing")
    return times


def _samples(name: str, values: ArrayLike, sample_count: int) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] != sample_count:
        raise ValueError(f"{name} must have shape (T,) or (T, D)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _normalization(name: str, value: float) -> float:
    normalization = float(value)
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError(f"{name} must be a positive finite value")
    return normalization


def integrated_squared_norm(
    timestamps_s: ArrayLike,
    values: ArrayLike,
) -> float:
    """``integral ||values(t)||² dt``를 trapezoidal rule로 계산한다."""

    times = _time_vector(timestamps_s)
    samples = _samples("values", values, times.size)
    squared_norm = np.sum(samples * samples, axis=1)
    dt = np.diff(times)
    integral = np.sum(
        0.5 * (squared_norm[:-1] + squared_norm[1:]) * dt,
        dtype=np.float64,
    )
    return float(integral)


def acceleration_cost(
    acceleration: ArrayLike,
    timestamps_s: ArrayLike,
    *,
    normalization: float = 1.0,
) -> float:
    """정규화한 acceleration 제곱 적분 cost."""

    scale = _normalization("normalization", normalization)
    return integrated_squared_norm(timestamps_s, acceleration) / scale


def jerk_cost(
    jerk: ArrayLike,
    timestamps_s: ArrayLike,
    *,
    normalization: float = 1.0,
) -> float:
    """정규화한 jerk 제곱 적분 cost."""

    scale = _normalization("normalization", normalization)
    return integrated_squared_norm(timestamps_s, jerk) / scale


@dataclass(frozen=True)
class TrajectoryCosts:
    """reward와 logging에 사용하는 궤적 smoothness cost."""

    acceleration_integral: float
    jerk_integral: float
    acceleration_cost: float
    jerk_cost: float
    duration_s: float

    @property
    def total_acceleration_cost(self) -> float:
        return self.acceleration_cost

    @property
    def total_jerk_cost(self) -> float:
        return self.jerk_cost

    def as_dict(self) -> dict[str, float]:
        """필드명과 호환 alias를 포함한 dict를 반환한다."""

        return {
            "acceleration_integral": self.acceleration_integral,
            "jerk_integral": self.jerk_integral,
            "acceleration_cost": self.acceleration_cost,
            "jerk_cost": self.jerk_cost,
            "total_acceleration_cost": self.acceleration_cost,
            "total_jerk_cost": self.jerk_cost,
            "duration_s": self.duration_s,
        }

    def __getitem__(self, key: str) -> float:
        return self.as_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.as_dict())


def compute_trajectory_costs(
    timestamps_s: ArrayLike,
    acceleration: ArrayLike,
    jerk: ArrayLike,
    *,
    acceleration_normalization: float = 1.0,
    jerk_normalization: float = 1.0,
    angular_acceleration: ArrayLike | None = None,
    angular_jerk: ArrayLike | None = None,
    angular_acceleration_weight: float = 1.0,
    angular_jerk_weight: float = 1.0,
) -> TrajectoryCosts:
    """Cartesian acceleration/jerk cost를 한 번에 계산한다.

    각도 성분을 제공하면 선형 성분의 제곱 norm에 명시된 weight를 곱해
    더한다. meter 계열과 radian 계열의 단위 크기를 암묵적으로 섞지 않기
    위해 weight는 호출자가 명시적으로 조정할 수 있다.
    """

    times = _time_vector(timestamps_s)
    linear_acceleration = _samples("acceleration", acceleration, times.size)
    linear_jerk = _samples("jerk", jerk, times.size)
    acceleration_weight = float(angular_acceleration_weight)
    jerk_weight = float(angular_jerk_weight)
    if (
        not np.isfinite(acceleration_weight)
        or acceleration_weight < 0.0
        or not np.isfinite(jerk_weight)
        or jerk_weight < 0.0
    ):
        raise ValueError("angular weights must be non-negative finite values")

    acceleration_squared_norm = np.sum(linear_acceleration * linear_acceleration, axis=1)
    jerk_squared_norm = np.sum(linear_jerk * linear_jerk, axis=1)
    if angular_acceleration is not None:
        angular_acceleration_samples = _samples(
            "angular_acceleration", angular_acceleration, times.size
        )
        acceleration_squared_norm += acceleration_weight * np.sum(
            angular_acceleration_samples * angular_acceleration_samples, axis=1
        )
    if angular_jerk is not None:
        angular_jerk_samples = _samples("angular_jerk", angular_jerk, times.size)
        jerk_squared_norm += jerk_weight * np.sum(
            angular_jerk_samples * angular_jerk_samples, axis=1
        )

    dt = np.diff(times)
    acceleration_integral = float(
        np.sum(
            0.5 * (acceleration_squared_norm[:-1] + acceleration_squared_norm[1:]) * dt,
            dtype=np.float64,
        )
    )
    jerk_integral = float(
        np.sum(
            0.5 * (jerk_squared_norm[:-1] + jerk_squared_norm[1:]) * dt,
            dtype=np.float64,
        )
    )
    acceleration_scale = _normalization("acceleration_normalization", acceleration_normalization)
    jerk_scale = _normalization("jerk_normalization", jerk_normalization)
    return TrajectoryCosts(
        acceleration_integral=acceleration_integral,
        jerk_integral=jerk_integral,
        acceleration_cost=acceleration_integral / acceleration_scale,
        jerk_cost=jerk_integral / jerk_scale,
        duration_s=float(times[-1] - times[0]),
    )


__all__ = [
    "TrajectoryCosts",
    "acceleration_cost",
    "compute_trajectory_costs",
    "integrated_squared_norm",
    "jerk_cost",
]
