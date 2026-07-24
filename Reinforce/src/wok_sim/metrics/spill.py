"""pan-local 최종 상태와 경계 이력을 이용한 유실 판정."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _positive_finite(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite value")
    return value


def _nonnegative_finite(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a non-negative finite value")
    return value


def _positions_3d(positions: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(positions, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] not in (2, 3):
        raise ValueError("positions must have shape (N, 2) or (N, 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError("positions must contain only finite values")
    if array.shape[1] == 2:
        array = np.column_stack((array, np.zeros(array.shape[0], dtype=np.float64)))
    return np.asarray(array, dtype=np.float64)


def _boolean_vector(name: str, values: ArrayLike, count: int) -> NDArray[np.bool_]:
    raw = np.asarray(values)
    if raw.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},)")
    return np.asarray(raw, dtype=np.bool_)


@dataclass(frozen=True)
class SpillMetrics:
    """입자 개수 및 질량 기준 유실 결과."""

    spilled_mask: NDArray[np.bool_]
    spill_count: int
    spill_count_ratio: float
    spill_mass_kg: float
    spill_mass_ratio: float
    initial_total_mass_kg: float

    def __post_init__(self) -> None:
        mask = np.asarray(self.spilled_mask, dtype=np.bool_)
        if mask.ndim != 1 or mask.size == 0:
            raise ValueError("spilled_mask must be a non-empty vector")
        mask = mask.copy()
        mask.setflags(write=False)
        object.__setattr__(self, "spilled_mask", mask)

    @property
    def spill_mass(self) -> float:
        """``spill_mass_kg``의 호환 alias."""

        return self.spill_mass_kg

    def as_dict(self) -> dict[str, object]:
        """로깅 가능한 dict로 변환한다."""

        return {
            "spilled_mask": self.spilled_mask.copy(),
            "spill_count": self.spill_count,
            "spill_count_ratio": self.spill_count_ratio,
            "spill_mass_kg": self.spill_mass_kg,
            "spill_mass": self.spill_mass_kg,
            "spill_mass_ratio": self.spill_mass_ratio,
            "initial_total_mass_kg": self.initial_total_mass_kg,
        }


def classify_spilled_particles(
    final_pan_local_positions: ArrayLike,
    *,
    rim_radius_m: float,
    spill_boundary_radius_m: float | None = None,
    spill_z_m: float = -np.inf,
    boundary_crossed: ArrayLike | None = None,
    final_contacts: ArrayLike | None = None,
    no_contact_durations_s: ArrayLike | None = None,
    minimum_no_contact_time_s: float = 0.0,
) -> NDArray[np.bool_]:
    """최종 pan-local 상태로 유실 입자를 판정한다.

    판정은 다음 두 경로만 사용한다.

    1. 시뮬레이션 중 명시적 spill boundary를 통과한 입자
    2. 종료 시 팬으로 돌아오지 않은 입자(림 바깥 또는 하부 경계 아래)

    따라서 팬 위에서 잠시 비행했지만 종료 전에 림 내부로 복귀한 입자는
    접촉 이력과 관계없이 유실이 아니다. ``minimum_no_contact_time_s``를
    양수로 설정하고 duration을 제공하면 림 바깥 판정에는 해당 지속 시간도
    요구한다. 하부 또는 명시적 경계 통과는 즉시 확정된다.
    """

    positions = _positions_3d(final_pan_local_positions)
    count = positions.shape[0]
    rim_radius = _positive_finite("rim_radius_m", rim_radius_m)
    boundary_radius = (
        None
        if spill_boundary_radius_m is None
        else _positive_finite("spill_boundary_radius_m", spill_boundary_radius_m)
    )
    if boundary_radius is not None and boundary_radius < rim_radius:
        raise ValueError("spill_boundary_radius_m must be >= rim_radius_m")
    spill_z = float(spill_z_m)
    if np.isnan(spill_z) or spill_z == np.inf:
        raise ValueError("spill_z_m must be finite or -inf")
    minimum_no_contact = _nonnegative_finite("minimum_no_contact_time_s", minimum_no_contact_time_s)

    radial_distance = np.linalg.norm(positions[:, :2], axis=1)
    outside_rim_final = radial_distance > rim_radius
    below_spill_final = positions[:, 2] < spill_z

    crossed = np.zeros(count, dtype=np.bool_)
    if boundary_radius is not None:
        crossed |= radial_distance > boundary_radius
    if boundary_crossed is not None:
        crossed |= _boolean_vector("boundary_crossed", boundary_crossed, count)

    if final_contacts is not None:
        # 입력 검증 및 API 대칭성을 위해 받는다. 림 내부 복귀가 우선이며,
        # 림 바깥 접촉(예: 외부 바닥 접촉)은 팬 복귀로 간주하지 않는다.
        _boolean_vector("final_contacts", final_contacts, count)

    if no_contact_durations_s is None:
        long_enough_outside = outside_rim_final
    else:
        durations = np.asarray(no_contact_durations_s, dtype=np.float64)
        if (
            durations.shape != (count,)
            or not np.all(np.isfinite(durations))
            or np.any(durations < 0.0)
        ):
            raise ValueError(
                "no_contact_durations_s must contain one non-negative finite duration per particle"
            )
        long_enough_outside = outside_rim_final & (durations >= minimum_no_contact)

    return np.asarray(crossed | below_spill_final | long_enough_outside, dtype=np.bool_)


def compute_spill_metrics(
    spilled_mask: ArrayLike,
    masses_kg: ArrayLike,
) -> SpillMetrics:
    """유실 mask에서 개수 및 질량 기준 metric을 계산한다."""

    masses = np.asarray(masses_kg, dtype=np.float64)
    if masses.ndim != 1 or masses.size == 0:
        raise ValueError("masses_kg must be a non-empty vector")
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("masses_kg must contain only positive finite values")
    mask = _boolean_vector("spilled_mask", spilled_mask, masses.size)
    total_mass = float(np.sum(masses, dtype=np.float64))
    spill_mass = float(np.sum(masses[mask], dtype=np.float64))
    spill_count = int(np.count_nonzero(mask))
    return SpillMetrics(
        spilled_mask=mask,
        spill_count=spill_count,
        spill_count_ratio=float(spill_count / masses.size),
        spill_mass_kg=spill_mass,
        spill_mass_ratio=float(np.clip(spill_mass / total_mass, 0.0, 1.0)),
        initial_total_mass_kg=total_mass,
    )


def evaluate_spill(
    final_pan_local_positions: ArrayLike,
    masses_kg: ArrayLike,
    *,
    rim_radius_m: float,
    spill_boundary_radius_m: float | None = None,
    spill_z_m: float = -np.inf,
    boundary_crossed: ArrayLike | None = None,
    final_contacts: ArrayLike | None = None,
    no_contact_durations_s: ArrayLike | None = None,
    minimum_no_contact_time_s: float = 0.0,
) -> SpillMetrics:
    """최종 유실 판정과 집계를 한 번에 수행한다."""

    mask = classify_spilled_particles(
        final_pan_local_positions,
        rim_radius_m=rim_radius_m,
        spill_boundary_radius_m=spill_boundary_radius_m,
        spill_z_m=spill_z_m,
        boundary_crossed=boundary_crossed,
        final_contacts=final_contacts,
        no_contact_durations_s=no_contact_durations_s,
        minimum_no_contact_time_s=minimum_no_contact_time_s,
    )
    return compute_spill_metrics(mask, masses_kg)


calculate_spill_metrics = evaluate_spill


class SpillTracker:
    """시간에 따른 비접촉 duration과 spill boundary 통과를 추적한다."""

    def __init__(
        self,
        particle_count: int,
        *,
        rim_radius_m: float,
        spill_boundary_radius_m: float,
        spill_z_m: float = -np.inf,
        minimum_no_contact_time_s: float = 0.0,
    ) -> None:
        if (
            isinstance(particle_count, bool)
            or int(particle_count) != particle_count
            or int(particle_count) <= 0
        ):
            raise ValueError("particle_count must be a positive integer")
        self.particle_count = int(particle_count)
        self.rim_radius_m = _positive_finite("rim_radius_m", rim_radius_m)
        self.spill_boundary_radius_m = _positive_finite(
            "spill_boundary_radius_m", spill_boundary_radius_m
        )
        if self.spill_boundary_radius_m < self.rim_radius_m:
            raise ValueError("spill_boundary_radius_m must be >= rim_radius_m")
        self.spill_z_m = float(spill_z_m)
        if np.isnan(self.spill_z_m) or self.spill_z_m == np.inf:
            raise ValueError("spill_z_m must be finite or -inf")
        self.minimum_no_contact_time_s = _nonnegative_finite(
            "minimum_no_contact_time_s", minimum_no_contact_time_s
        )
        self.boundary_crossed = np.zeros(self.particle_count, dtype=np.bool_)
        self.no_contact_durations_s = np.zeros(self.particle_count, dtype=np.float64)
        self._last_time_s: float | None = None
        self._last_positions: NDArray[np.float64] | None = None
        self._last_contacts: NDArray[np.bool_] | None = None

    def reset(self) -> None:
        """모든 episode 이력을 초기화한다."""

        self.boundary_crossed.fill(False)
        self.no_contact_durations_s.fill(0.0)
        self._last_time_s = None
        self._last_positions = None
        self._last_contacts = None

    def update(
        self,
        time_s: float,
        pan_local_positions: ArrayLike,
        pan_contacts: ArrayLike,
    ) -> None:
        """한 timestep의 위치와 팬 접촉 상태를 기록한다."""

        time_value = float(time_s)
        if not np.isfinite(time_value) or time_value < 0.0:
            raise ValueError("time_s must be a non-negative finite value")
        if self._last_time_s is not None and time_value <= self._last_time_s:
            raise ValueError("time_s must increase strictly between updates")
        positions = _positions_3d(pan_local_positions)
        if positions.shape[0] != self.particle_count:
            raise ValueError("pan_local_positions particle count changed")
        contacts = _boolean_vector("pan_contacts", pan_contacts, self.particle_count)

        radial_distance = np.linalg.norm(positions[:, :2], axis=1)
        self.boundary_crossed |= (radial_distance > self.spill_boundary_radius_m) | (
            positions[:, 2] < self.spill_z_m
        )
        if self._last_time_s is not None:
            dt = time_value - self._last_time_s
            self.no_contact_durations_s = np.where(
                contacts,
                0.0,
                self.no_contact_durations_s + dt,
            )
        else:
            self.no_contact_durations_s[contacts] = 0.0

        self._last_time_s = time_value
        self._last_positions = positions.copy()
        self._last_contacts = contacts.copy()

    def finalize(self, masses_kg: ArrayLike) -> SpillMetrics:
        """마지막 기록 상태에서 최종 유실 결과를 계산한다."""

        if self._last_positions is None or self._last_contacts is None:
            raise RuntimeError("SpillTracker.finalize() requires at least one update")
        return evaluate_spill(
            self._last_positions,
            masses_kg,
            rim_radius_m=self.rim_radius_m,
            spill_boundary_radius_m=self.spill_boundary_radius_m,
            spill_z_m=self.spill_z_m,
            boundary_crossed=self.boundary_crossed,
            final_contacts=self._last_contacts,
            no_contact_durations_s=self.no_contact_durations_s,
            minimum_no_contact_time_s=self.minimum_no_contact_time_s,
        )


__all__ = [
    "SpillMetrics",
    "SpillTracker",
    "calculate_spill_metrics",
    "classify_spilled_particles",
    "compute_spill_metrics",
    "evaluate_spill",
]
