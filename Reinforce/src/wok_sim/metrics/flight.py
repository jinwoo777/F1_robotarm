"""팬 접촉 전이 기반 입자 비행 event 추적."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .mixing import size_group_labels


def _particle_vectors(
    name: str,
    values: ArrayLike,
    particle_count: int,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (particle_count, 3):
        raise ValueError(f"{name} must have shape ({particle_count}, 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _contacts(values: ArrayLike, particle_count: int) -> NDArray[np.bool_]:
    array = np.asarray(values)
    if array.shape != (particle_count,):
        raise ValueError(f"pan_contacts must have shape ({particle_count},)")
    return np.asarray(array, dtype=np.bool_)


def _surface_velocity(
    values: ArrayLike,
    particle_count: int,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape == (3,):
        array = np.broadcast_to(array, (particle_count, 3))
    if array.shape != (particle_count, 3) or not np.all(np.isfinite(array)):
        raise ValueError(
            "pan_surface_velocity_world must have shape (3,) or (N, 3) and contain finite values"
        )
    return np.asarray(array, dtype=np.float64)


@dataclass
class FlightEvent:
    """한 입자의 takeoff부터 landing까지의 비행."""

    particle_index: int
    takeoff_time_s: float
    takeoff_world_z_m: float
    takeoff_relative_z_m: float
    launch_velocity_relative_m_s: NDArray[np.float64]
    launch_angle_xz_rad: float
    launch_elevation_rad: float
    apex_world_z_m: float
    apex_relative_z_m: float
    landing_time_s: float | None = None

    @property
    def max_flight_height_world(self) -> float:
        """takeoff 위치 대비 world-z 최대 상승 높이."""

        return max(0.0, self.apex_world_z_m - self.takeoff_world_z_m)

    @property
    def max_flight_height_relative(self) -> float:
        """takeoff 위치 대비 pan-local-z 최대 상승 높이."""

        return max(0.0, self.apex_relative_z_m - self.takeoff_relative_z_m)

    @property
    def flight_duration_s(self) -> float | None:
        """착지한 event의 비행 시간. 진행 중이면 ``None``."""

        if self.landing_time_s is None:
            return None
        return max(0.0, self.landing_time_s - self.takeoff_time_s)

    def duration_at(self, current_time_s: float | None = None) -> float:
        """진행 중 event까지 포함한 현재 비행 시간을 반환한다."""

        if self.landing_time_s is not None:
            return max(0.0, self.landing_time_s - self.takeoff_time_s)
        if current_time_s is None:
            return 0.0
        return max(0.0, float(current_time_s) - self.takeoff_time_s)

    def as_dict(self, *, current_time_s: float | None = None) -> dict[str, Any]:
        """요구된 비행 필드를 직렬화 가능한 dict로 반환한다."""

        return {
            "particle_index": self.particle_index,
            "max_flight_height_world": self.max_flight_height_world,
            "max_flight_height_relative": self.max_flight_height_relative,
            "takeoff_time": self.takeoff_time_s,
            "landing_time": self.landing_time_s,
            "flight_duration": self.duration_at(current_time_s),
            "launch_angle_xz": self.launch_angle_xz_rad,
            "launch_elevation_angle": self.launch_elevation_rad,
            "takeoff_world_z_m": self.takeoff_world_z_m,
            "takeoff_relative_z_m": self.takeoff_relative_z_m,
            "apex_world_z_m": self.apex_world_z_m,
            "apex_relative_z_m": self.apex_relative_z_m,
            "launch_velocity_relative_m_s": self.launch_velocity_relative_m_s.copy(),
        }


class FlightTracker:
    """직전 접촉 → 현재 비접촉 전이를 이용해 비행을 검출한다.

    takeoff에는 양의 pan-relative 수직 속도가 추가로 필요하다. 비행각은
    입자 속도에서 해당 위치의 팬 표면 속도를 뺀 상대 속도로 계산한다.
    """

    def __init__(
        self,
        particle_count: int,
        *,
        minimum_takeoff_vertical_velocity_m_s: float = 0.0,
    ) -> None:
        if (
            isinstance(particle_count, bool)
            or int(particle_count) != particle_count
            or int(particle_count) <= 0
        ):
            raise ValueError("particle_count must be a positive integer")
        threshold = float(minimum_takeoff_vertical_velocity_m_s)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("minimum_takeoff_vertical_velocity_m_s must be non-negative")
        self.particle_count = int(particle_count)
        self.minimum_takeoff_vertical_velocity_m_s = threshold
        self.events: list[list[FlightEvent]] = [[] for _ in range(self.particle_count)]
        self._active: list[FlightEvent | None] = [None for _ in range(self.particle_count)]
        self._previous_contacts: NDArray[np.bool_] | None = None
        self._last_time_s: float | None = None

    def reset(self) -> None:
        """episode 추적 상태를 초기화한다."""

        self.events = [[] for _ in range(self.particle_count)]
        self._active = [None for _ in range(self.particle_count)]
        self._previous_contacts = None
        self._last_time_s = None

    def update(
        self,
        time_s: float,
        particle_positions_world: ArrayLike,
        particle_velocities_world: ArrayLike,
        pan_contacts: ArrayLike,
        *,
        pan_surface_velocity_world: ArrayLike = (0.0, 0.0, 0.0),
        particle_positions_local: ArrayLike | None = None,
        pan_origin_world: ArrayLike | None = None,
        pan_rotation_world_from_local: ArrayLike | None = None,
    ) -> None:
        """한 timestep의 입자와 팬 상태를 반영한다.

        정확한 pan-local 좌표가 있으면 ``particle_positions_local``을
        전달한다. 없으면 ``world_z - pan_origin_world[2]``를 상대 높이의
        근사로 사용하며, pan origin도 없으면 world z 자체를 사용한다.
        팬이 회전한다면 ``pan_rotation_world_from_local``에 3x3 회전행렬을
        전달해야 takeoff 수직 속도와 launch angle이 pan-local xz 기준으로
        계산된다.
        """

        current_time = float(time_s)
        if not np.isfinite(current_time) or current_time < 0.0:
            raise ValueError("time_s must be a non-negative finite value")
        if self._last_time_s is not None and current_time <= self._last_time_s:
            raise ValueError("time_s must increase strictly between updates")
        positions_world = _particle_vectors(
            "particle_positions_world",
            particle_positions_world,
            self.particle_count,
        )
        velocities_world = _particle_vectors(
            "particle_velocities_world",
            particle_velocities_world,
            self.particle_count,
        )
        contacts = _contacts(pan_contacts, self.particle_count)
        surface_velocity = _surface_velocity(pan_surface_velocity_world, self.particle_count)
        relative_velocity_world = velocities_world - surface_velocity
        if pan_rotation_world_from_local is None:
            relative_velocity = relative_velocity_world
        else:
            rotation = np.asarray(pan_rotation_world_from_local, dtype=np.float64)
            if (
                rotation.shape != (3, 3)
                or not np.all(np.isfinite(rotation))
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7)
            ):
                raise ValueError(
                    "pan_rotation_world_from_local must be a finite 3x3 proper rotation matrix"
                )
            # row-vector convention: v_local = v_world @ R_world_from_local
            relative_velocity = relative_velocity_world @ rotation

        if particle_positions_local is not None:
            local_positions = _particle_vectors(
                "particle_positions_local",
                particle_positions_local,
                self.particle_count,
            )
            relative_z = local_positions[:, 2]
        elif pan_origin_world is not None:
            origin = np.asarray(pan_origin_world, dtype=np.float64)
            if origin.shape != (3,) or not np.all(np.isfinite(origin)):
                raise ValueError("pan_origin_world must contain three finite values")
            relative_z = positions_world[:, 2] - origin[2]
        else:
            relative_z = positions_world[:, 2]

        if self._previous_contacts is None:
            # 첫 sample은 전이를 알 수 없으므로 baseline으로만 저장한다.
            self._previous_contacts = contacts.copy()
            self._last_time_s = current_time
            return

        took_off = (
            self._previous_contacts
            & ~contacts
            & (relative_velocity[:, 2] > self.minimum_takeoff_vertical_velocity_m_s)
        )
        for particle_index in np.flatnonzero(took_off):
            velocity = relative_velocity[particle_index].copy()
            horizontal_speed = float(np.hypot(velocity[0], velocity[1]))
            event = FlightEvent(
                particle_index=int(particle_index),
                takeoff_time_s=current_time,
                takeoff_world_z_m=float(positions_world[particle_index, 2]),
                takeoff_relative_z_m=float(relative_z[particle_index]),
                launch_velocity_relative_m_s=velocity,
                launch_angle_xz_rad=float(np.arctan2(velocity[2], velocity[0])),
                launch_elevation_rad=float(np.arctan2(velocity[2], horizontal_speed)),
                apex_world_z_m=float(positions_world[particle_index, 2]),
                apex_relative_z_m=float(relative_z[particle_index]),
            )
            self.events[particle_index].append(event)
            self._active[particle_index] = event

        for particle_index, event in enumerate(self._active):
            if event is None or contacts[particle_index]:
                continue
            event.apex_world_z_m = max(
                event.apex_world_z_m,
                float(positions_world[particle_index, 2]),
            )
            event.apex_relative_z_m = max(
                event.apex_relative_z_m,
                float(relative_z[particle_index]),
            )

        landed = ~self._previous_contacts & contacts
        for particle_index in np.flatnonzero(landed):
            event = self._active[particle_index]
            if event is not None:
                event.landing_time_s = current_time
                self._active[particle_index] = None

        self._previous_contacts = contacts.copy()
        self._last_time_s = current_time

    @property
    def active_flight_mask(self) -> NDArray[np.bool_]:
        """현재 비행 event가 진행 중인 입자 mask."""

        return np.asarray([event is not None for event in self._active], dtype=np.bool_)

    def particle_records(self) -> list[dict[str, Any]]:
        """입자별 최대 상대 높이 event를 한 행으로 요약한다."""

        records: list[dict[str, Any]] = []
        for particle_index, particle_events in enumerate(self.events):
            if not particle_events:
                records.append(
                    {
                        "particle_index": particle_index,
                        "flight_count": 0,
                        "max_flight_height_world": 0.0,
                        "max_flight_height_relative": 0.0,
                        "takeoff_time": None,
                        "landing_time": None,
                        "flight_duration": 0.0,
                        "launch_angle_xz": None,
                        "launch_elevation_angle": None,
                    }
                )
                continue
            representative = max(
                particle_events,
                key=lambda event: (
                    event.max_flight_height_relative,
                    event.max_flight_height_world,
                ),
            )
            record = representative.as_dict(current_time_s=self._last_time_s)
            record["flight_count"] = len(particle_events)
            records.append(record)
        return records

    def statistics(
        self,
        *,
        radii_m: ArrayLike | None = None,
        spilled_mask: ArrayLike | None = None,
    ) -> dict[str, Any]:
        """입자별 대표 비행에 대한 episode 통계를 반환한다."""

        records = self.particle_records()
        has_flight = np.asarray([record["flight_count"] > 0 for record in records], dtype=np.bool_)
        relative_heights = np.asarray(
            [record["max_flight_height_relative"] for record in records],
            dtype=np.float64,
        )
        world_heights = np.asarray(
            [record["max_flight_height_world"] for record in records],
            dtype=np.float64,
        )
        angles = np.asarray(
            [
                0.0 if record["launch_angle_xz"] is None else record["launch_angle_xz"]
                for record in records
            ],
            dtype=np.float64,
        )

        def mean_or_zero(values: NDArray[np.float64], mask: NDArray[np.bool_]) -> float:
            return float(np.mean(values[mask])) if np.any(mask) else 0.0

        def std_or_zero(values: NDArray[np.float64], mask: NDArray[np.bool_]) -> float:
            return float(np.std(values[mask])) if np.any(mask) else 0.0

        result: dict[str, Any] = {
            "particles_with_flight": int(np.count_nonzero(has_flight)),
            "flight_event_count": int(sum(len(events) for events in self.events)),
            "mean_flight_height": mean_or_zero(relative_heights, has_flight),
            "mean_flight_height_relative": mean_or_zero(relative_heights, has_flight),
            "mean_flight_height_world": mean_or_zero(world_heights, has_flight),
            "max_flight_height": (
                float(np.max(relative_heights[has_flight])) if np.any(has_flight) else 0.0
            ),
            "max_flight_height_relative": (
                float(np.max(relative_heights[has_flight])) if np.any(has_flight) else 0.0
            ),
            "max_flight_height_world": (
                float(np.max(world_heights[has_flight])) if np.any(has_flight) else 0.0
            ),
            "flight_height_std": std_or_zero(relative_heights, has_flight),
            "mean_launch_angle": mean_or_zero(angles, has_flight),
            "launch_angle_std": std_or_zero(angles, has_flight),
            "particle_flights": records,
        }

        if radii_m is not None:
            radii = np.asarray(radii_m, dtype=np.float64)
            if (
                radii.shape != (self.particle_count,)
                or not np.all(np.isfinite(radii))
                or np.any(radii <= 0.0)
            ):
                raise ValueError("radii_m must contain one positive finite radius per particle")
            group_ids = size_group_labels(radii)
            names = ("small", "medium", "large")
            group_stats: dict[str, dict[str, float | int]] = {}
            for group_id, name in enumerate(names):
                group_mask = has_flight & (group_ids == group_id)
                group_stats[name] = {
                    "particle_count": int(np.count_nonzero(group_ids == group_id)),
                    "particles_with_flight": int(np.count_nonzero(group_mask)),
                    "mean_flight_height": mean_or_zero(relative_heights, group_mask),
                    "mean_launch_angle": mean_or_zero(angles, group_mask),
                }
            result["size_group_flight_statistics"] = group_stats
        else:
            result["size_group_flight_statistics"] = None

        if spilled_mask is not None:
            spilled = np.asarray(spilled_mask)
            if spilled.shape != (self.particle_count,):
                raise ValueError(f"spilled_mask must have shape ({self.particle_count},)")
            spilled = np.asarray(spilled, dtype=np.bool_)
            spilled_with_flight = spilled & has_flight
            result.update(
                {
                    "spilled_mean_flight_height": mean_or_zero(
                        relative_heights, spilled_with_flight
                    ),
                    "spilled_mean_launch_angle": mean_or_zero(angles, spilled_with_flight),
                }
            )
        else:
            result.update(
                {
                    "spilled_mean_flight_height": 0.0,
                    "spilled_mean_launch_angle": 0.0,
                }
            )
        return result

    summary = statistics


def flight_statistics_from_summary(
    particle_flight_summary: Mapping[str, ArrayLike],
    *,
    radii_m: ArrayLike | None = None,
    spilled_mask: ArrayLike | None = None,
) -> dict[str, Any]:
    """시뮬레이터의 입자별 비행 배열을 episode 통계로 집계한다.

    :class:`FlightTracker` 외에 simulation contact tracker가 이미 생성한
    summary도 동일한 통계 계약으로 사용할 수 있게 하는 adapter다.
    """

    relative_raw = particle_flight_summary.get(
        "max_flight_height_relative_m",
        particle_flight_summary.get("max_flight_height_relative"),
    )
    world_raw = particle_flight_summary.get(
        "max_flight_height_world_m",
        particle_flight_summary.get("max_flight_height_world"),
    )
    angle_raw = particle_flight_summary.get(
        "launch_angle_xz_rad",
        particle_flight_summary.get("launch_angle_xz"),
    )
    if relative_raw is None or world_raw is None or angle_raw is None:
        raise ValueError(
            "particle_flight_summary requires relative/world height and launch-angle arrays"
        )
    relative = np.asarray(relative_raw, dtype=np.float64)
    world = np.asarray(world_raw, dtype=np.float64)
    angles = np.asarray(angle_raw, dtype=np.float64)
    if (
        relative.ndim != 1
        or relative.size == 0
        or world.shape != relative.shape
        or angles.shape != relative.shape
        or not np.all(np.isfinite(relative))
        or not np.all(np.isfinite(world))
        or np.any(relative < 0.0)
        or np.any(world < 0.0)
    ):
        raise ValueError(
            "flight height arrays must be non-empty, equally shaped, non-negative and finite"
        )
    flight_count_raw = particle_flight_summary.get("flight_count")
    if flight_count_raw is None:
        has_flight = np.isfinite(angles)
        flight_count = has_flight.astype(np.int64)
    else:
        flight_count = np.asarray(flight_count_raw, dtype=np.int64)
        if flight_count.shape != relative.shape or np.any(flight_count < 0):
            raise ValueError("flight_count must have one non-negative value per particle")
        has_flight = flight_count > 0
    if np.any(has_flight & ~np.isfinite(angles)):
        raise ValueError("particles with flight must have finite launch angles")

    def mean_or_zero(values: NDArray[np.float64], mask: NDArray[np.bool_]) -> float:
        return float(np.mean(values[mask])) if np.any(mask) else 0.0

    def std_or_zero(values: NDArray[np.float64], mask: NDArray[np.bool_]) -> float:
        return float(np.std(values[mask])) if np.any(mask) else 0.0

    result: dict[str, Any] = {
        "particles_with_flight": int(np.count_nonzero(has_flight)),
        "flight_event_count": int(np.sum(flight_count)),
        "mean_flight_height": mean_or_zero(relative, has_flight),
        "mean_flight_height_relative": mean_or_zero(relative, has_flight),
        "mean_flight_height_world": mean_or_zero(world, has_flight),
        "max_flight_height": (float(np.max(relative[has_flight])) if np.any(has_flight) else 0.0),
        "max_flight_height_relative": (
            float(np.max(relative[has_flight])) if np.any(has_flight) else 0.0
        ),
        "max_flight_height_world": (
            float(np.max(world[has_flight])) if np.any(has_flight) else 0.0
        ),
        "flight_height_std": std_or_zero(relative, has_flight),
        "mean_launch_angle": mean_or_zero(angles, has_flight),
        "launch_angle_std": std_or_zero(angles, has_flight),
    }

    if radii_m is not None:
        radii = np.asarray(radii_m, dtype=np.float64)
        if radii.shape != relative.shape:
            raise ValueError("radii_m must contain one radius per particle")
        groups = size_group_labels(radii)
        group_statistics: dict[str, dict[str, float | int]] = {}
        for group_id, name in enumerate(("small", "medium", "large")):
            in_group = groups == group_id
            group_flights = in_group & has_flight
            group_statistics[name] = {
                "particle_count": int(np.count_nonzero(in_group)),
                "particles_with_flight": int(np.count_nonzero(group_flights)),
                "mean_flight_height": mean_or_zero(relative, group_flights),
                "mean_launch_angle": mean_or_zero(angles, group_flights),
            }
        result["size_group_flight_statistics"] = group_statistics
    else:
        result["size_group_flight_statistics"] = None

    if spilled_mask is not None:
        spilled = np.asarray(spilled_mask, dtype=np.bool_)
        if spilled.shape != relative.shape:
            raise ValueError("spilled_mask must contain one value per particle")
        spilled_flights = spilled & has_flight
        result["spilled_mean_flight_height"] = mean_or_zero(relative, spilled_flights)
        result["spilled_mean_launch_angle"] = mean_or_zero(angles, spilled_flights)
    else:
        result["spilled_mean_flight_height"] = 0.0
        result["spilled_mean_launch_angle"] = 0.0
    return result


def track_flights(
    times_s: ArrayLike,
    particle_positions_world: ArrayLike,
    particle_velocities_world: ArrayLike,
    pan_contacts: ArrayLike,
    *,
    pan_surface_velocities_world: ArrayLike | None = None,
    particle_positions_local: ArrayLike | None = None,
    pan_rotations_world_from_local: ArrayLike | None = None,
    minimum_takeoff_vertical_velocity_m_s: float = 0.0,
) -> FlightTracker:
    """배열 trajectory 전체를 :class:`FlightTracker`로 처리하는 편의 함수."""

    times = np.asarray(times_s, dtype=np.float64)
    positions = np.asarray(particle_positions_world, dtype=np.float64)
    velocities = np.asarray(particle_velocities_world, dtype=np.float64)
    contacts = np.asarray(pan_contacts)
    if (
        times.ndim != 1
        or times.size < 1
        or positions.ndim != 3
        or positions.shape[0] != times.size
        or positions.shape[2] != 3
        or velocities.shape != positions.shape
        or contacts.shape != positions.shape[:2]
    ):
        raise ValueError(
            "trajectory shapes must be times=(T,), positions/velocities=(T,N,3), contacts=(T,N)"
        )
    if (
        not np.all(np.isfinite(times))
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(velocities))
        or (times.size > 1 and np.any(np.diff(times) <= 0.0))
    ):
        raise ValueError("trajectory values must be finite with strictly increasing time")
    particle_count = positions.shape[1]
    if pan_surface_velocities_world is None:
        surface = np.zeros((times.size, 3), dtype=np.float64)
    else:
        surface = np.asarray(pan_surface_velocities_world, dtype=np.float64)
        if surface.shape not in ((times.size, 3), positions.shape):
            raise ValueError("pan_surface_velocities_world must have shape (T,3) or (T,N,3)")
    if particle_positions_local is None:
        local = None
    else:
        local = np.asarray(particle_positions_local, dtype=np.float64)
        if local.shape != positions.shape:
            raise ValueError("particle_positions_local must have shape (T,N,3)")
    if pan_rotations_world_from_local is None:
        rotations = None
    else:
        rotations = np.asarray(pan_rotations_world_from_local, dtype=np.float64)
        if rotations.shape != (times.size, 3, 3):
            raise ValueError("pan_rotations_world_from_local must have shape (T,3,3)")

    tracker = FlightTracker(
        particle_count,
        minimum_takeoff_vertical_velocity_m_s=(minimum_takeoff_vertical_velocity_m_s),
    )
    for index, time_value in enumerate(times):
        tracker.update(
            float(time_value),
            positions[index],
            velocities[index],
            contacts[index],
            pan_surface_velocity_world=surface[index],
            particle_positions_local=None if local is None else local[index],
            pan_rotation_world_from_local=(None if rotations is None else rotations[index]),
        )
    return tracker


__all__ = [
    "FlightEvent",
    "FlightTracker",
    "flight_statistics_from_summary",
    "track_flights",
]
