"""팬-입자 contact와 takeoff/landing event 추적."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


def _normalize_quaternion(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=float)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("pan quaternion은 유한한 wxyz 길이 4 배열이어야 합니다.")
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        raise ValueError("pan quaternion norm이 0입니다.")
    return quaternion / norm


def quaternion_to_matrix(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    """MuJoCo 순서(wxyz) quaternion을 3x3 회전행렬로 바꾼다."""

    w, x, y, z = _normalize_quaternion(quaternion_wxyz)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


@dataclass(frozen=True)
class ContactSnapshot:
    """한 physics step의 입자별 팬 contact 정보."""

    touching_pan: np.ndarray
    normal_force_n: np.ndarray
    contact_count: np.ndarray


def collect_particle_pan_contacts(
    model: Any,
    data: Any,
    particle_geom_ids: Sequence[int],
    pan_geom_ids: Sequence[int],
    *,
    include_force: bool = True,
) -> ContactSnapshot:
    """MuJoCo contact list를 입자별 bool/normal force로 정리한다."""

    particle_ids = np.asarray(particle_geom_ids, dtype=int)
    particle_lookup = {int(geom_id): index for index, geom_id in enumerate(particle_ids)}
    pan_set = {int(geom_id) for geom_id in pan_geom_ids}
    touching = np.zeros(len(particle_ids), dtype=bool)
    force = np.zeros(len(particle_ids), dtype=float)
    count = np.zeros(len(particle_ids), dtype=np.int32)

    mujoco_module = None
    if include_force:
        try:
            import mujoco as mujoco_module
        except ImportError:
            include_force = False

    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        particle_index: int | None = None
        if geom1 in particle_lookup and geom2 in pan_set:
            particle_index = particle_lookup[geom1]
        elif geom2 in particle_lookup and geom1 in pan_set:
            particle_index = particle_lookup[geom2]
        if particle_index is None:
            continue
        touching[particle_index] = True
        count[particle_index] += 1
        if include_force and mujoco_module is not None:
            contact_force = np.zeros(6, dtype=float)
            mujoco_module.mj_contactForce(model, data, contact_index, contact_force)
            force[particle_index] += abs(float(contact_force[0]))
    return ContactSnapshot(touching, force, count)


@dataclass
class FlightEvent:
    """한 입자의 한 번의 비행."""

    particle_index: int
    takeoff_time_s: float
    landing_time_s: float | None
    flight_duration_s: float | None
    takeoff_position_world_m: tuple[float, float, float]
    takeoff_position_pan_m: tuple[float, float, float]
    takeoff_relative_velocity_world_m_s: tuple[float, float, float]
    takeoff_relative_velocity_pan_m_s: tuple[float, float, float]
    launch_angle_xz_rad: float
    elevation_angle_rad: float
    max_height_world_m: float
    max_height_relative_m: float
    max_world_z_m: float
    max_pan_z_m: float

    def as_dict(self) -> dict[str, Any]:
        """JSON/CSV 직렬화에 적합한 dict를 반환한다."""

        return asdict(self)


class ContactEventTracker:
    """contact edge와 팬 기준 상대 속도로 takeoff/landing을 검출한다."""

    def __init__(
        self,
        particle_count: int,
        *,
        takeoff_vertical_velocity_threshold_m_s: float = 1e-4,
        contact_gap_tolerance_s: float = 0.0,
    ) -> None:
        if particle_count <= 0:
            raise ValueError("particle_count는 양수여야 합니다.")
        self.particle_count = int(particle_count)
        self.threshold = float(takeoff_vertical_velocity_threshold_m_s)
        if not np.isfinite(self.threshold) or self.threshold < 0.0:
            raise ValueError("takeoff velocity threshold는 음수가 아닌 유한값이어야 합니다.")
        self.contact_gap_tolerance_s = float(contact_gap_tolerance_s)
        if not np.isfinite(self.contact_gap_tolerance_s) or self.contact_gap_tolerance_s < 0.0:
            raise ValueError("contact_gap_tolerance_s는 음수가 아닌 유한값이어야 합니다.")
        self.previous_contacts: np.ndarray | None = None
        self.events: list[FlightEvent] = []
        self._active: list[FlightEvent | None] = [None] * self.particle_count
        self._candidates: list[FlightEvent | None] = [None] * self.particle_count
        self._last_time_s: float | None = None

    @property
    def in_flight(self) -> np.ndarray:
        """현재 tracker상 비행 중인 입자 mask."""

        return np.asarray([event is not None for event in self._active], dtype=bool)

    def reset(self, initial_contacts: Sequence[bool] | None = None) -> None:
        """모든 event를 지우고 선택적으로 초기 contact 상태를 지정한다."""

        if initial_contacts is None:
            self.previous_contacts = None
        else:
            contacts = np.asarray(initial_contacts, dtype=bool)
            if contacts.shape != (self.particle_count,):
                raise ValueError("initial_contacts shape이 particle_count와 다릅니다.")
            self.previous_contacts = contacts.copy()
        self.events.clear()
        self._active = [None] * self.particle_count
        self._candidates = [None] * self.particle_count
        self._last_time_s = None

    def update(
        self,
        time_s: float,
        contacts: Sequence[bool],
        positions_world_m: np.ndarray,
        velocities_world_m_s: np.ndarray,
        *,
        pan_position_world_m: Sequence[float],
        pan_quaternion_wxyz: Sequence[float],
        pan_linear_velocity_world_m_s: Sequence[float] = (0.0, 0.0, 0.0),
        pan_angular_velocity_world_rad_s: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> list[FlightEvent]:
        """새 takeoff/landing을 처리하고 이 호출에서 완료된 event를 반환한다."""

        time_value = float(time_s)
        if not np.isfinite(time_value):
            raise ValueError("time_s는 유한해야 합니다.")
        if self._last_time_s is not None and time_value < self._last_time_s:
            raise ValueError("ContactEventTracker time_s는 단조 증가해야 합니다.")
        self._last_time_s = time_value

        contact_array = np.asarray(contacts, dtype=bool)
        positions = np.asarray(positions_world_m, dtype=float)
        velocities = np.asarray(velocities_world_m_s, dtype=float)
        if contact_array.shape != (self.particle_count,):
            raise ValueError("contacts shape이 particle_count와 다릅니다.")
        if positions.shape != (self.particle_count, 3):
            raise ValueError("positions_world_m shape은 (particle_count, 3)이어야 합니다.")
        if velocities.shape != positions.shape:
            raise ValueError("velocities_world_m_s shape이 positions와 다릅니다.")
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise ValueError("particle state에 NaN 또는 inf가 있습니다.")

        pan_position = np.asarray(pan_position_world_m, dtype=float)
        pan_linear = np.asarray(pan_linear_velocity_world_m_s, dtype=float)
        pan_angular = np.asarray(pan_angular_velocity_world_rad_s, dtype=float)
        if pan_position.shape != (3,) or pan_linear.shape != (3,) or pan_angular.shape != (3,):
            raise ValueError("pan position/velocity는 길이 3 배열이어야 합니다.")
        rotation = quaternion_to_matrix(pan_quaternion_wxyz)
        positions_relative_world = positions - pan_position
        positions_pan = positions_relative_world @ rotation
        pan_surface_velocity = pan_linear + np.cross(pan_angular, positions_relative_world)
        relative_velocity_world = velocities - pan_surface_velocity
        relative_velocity_pan = relative_velocity_world @ rotation

        if self.previous_contacts is None:
            self.previous_contacts = contact_array.copy()
            return []

        completed: list[FlightEvent] = []
        for index in range(self.particle_count):
            active = self._active[index]
            if active is not None:
                world_z = float(positions[index, 2])
                pan_z = float(positions_pan[index, 2])
                active.max_world_z_m = max(active.max_world_z_m, world_z)
                active.max_pan_z_m = max(active.max_pan_z_m, pan_z)
                active.max_height_world_m = max(
                    active.max_height_world_m,
                    world_z - active.takeoff_position_world_m[2],
                )
                active.max_height_relative_m = max(
                    active.max_height_relative_m,
                    pan_z - active.takeoff_position_pan_m[2],
                )

            candidate = self._candidates[index]
            if candidate is not None and not contact_array[index]:
                world_z = float(positions[index, 2])
                pan_z = float(positions_pan[index, 2])
                candidate.max_world_z_m = max(candidate.max_world_z_m, world_z)
                candidate.max_pan_z_m = max(candidate.max_pan_z_m, pan_z)
                candidate.max_height_world_m = max(
                    candidate.max_height_world_m,
                    world_z - candidate.takeoff_position_world_m[2],
                )
                candidate.max_height_relative_m = max(
                    candidate.max_height_relative_m,
                    pan_z - candidate.takeoff_position_pan_m[2],
                )

            lost_contact = bool(self.previous_contacts[index] and not contact_array[index])
            upward = relative_velocity_pan[index, 2] > self.threshold
            if lost_contact and upward and active is None and self._candidates[index] is None:
                velocity_pan = relative_velocity_pan[index]
                velocity_world = relative_velocity_world[index]
                horizontal_speed = math.hypot(float(velocity_pan[0]), float(velocity_pan[1]))
                event = FlightEvent(
                    particle_index=index,
                    takeoff_time_s=time_value,
                    landing_time_s=None,
                    flight_duration_s=None,
                    takeoff_position_world_m=tuple(float(v) for v in positions[index]),
                    takeoff_position_pan_m=tuple(float(v) for v in positions_pan[index]),
                    takeoff_relative_velocity_world_m_s=tuple(float(v) for v in velocity_world),
                    takeoff_relative_velocity_pan_m_s=tuple(float(v) for v in velocity_pan),
                    launch_angle_xz_rad=float(
                        math.atan2(float(velocity_pan[2]), float(velocity_pan[0]))
                    ),
                    elevation_angle_rad=float(math.atan2(float(velocity_pan[2]), horizontal_speed)),
                    max_height_world_m=0.0,
                    max_height_relative_m=0.0,
                    max_world_z_m=float(positions[index, 2]),
                    max_pan_z_m=float(positions_pan[index, 2]),
                )
                self._candidates[index] = event
                candidate = event

            candidate = self._candidates[index]
            if candidate is not None:
                if contact_array[index]:
                    # 짧은 solver contact chatter는 비행으로 세지 않는다.
                    self._candidates[index] = None
                    candidate = None
                elif time_value - candidate.takeoff_time_s >= self.contact_gap_tolerance_s:
                    self._active[index] = candidate
                    self.events.append(candidate)
                    self._candidates[index] = None
                    active = candidate

            if active is not None and contact_array[index]:
                active.landing_time_s = time_value
                active.flight_duration_s = max(0.0, time_value - active.takeoff_time_s)
                self._active[index] = None
                completed.append(active)

        self.previous_contacts = contact_array.copy()
        return completed

    def finalize(self, final_time_s: float | None = None) -> None:
        """rollout 종료 시 미착륙 event의 duration을 현재까지로 기록한다."""

        if final_time_s is None:
            final_time_s = self._last_time_s
        if final_time_s is None:
            return
        final_time = float(final_time_s)
        for event in self._active:
            if event is not None:
                event.flight_duration_s = max(0.0, final_time - event.takeoff_time_s)

    def particle_summary(self) -> dict[str, np.ndarray]:
        """입자별 최대 높이와 대표(max-relative) 비행 배열을 반환한다.

        world/relative 최대 높이는 각 기준에서 독립적으로 올바른 최댓값을
        사용한다. 시간과 각도는 최대 relative height event 한 건에서 함께
        가져와 서로 다른 비행의 first/last/cumulative 값이 섞이지 않게 한다.
        """

        max_world = np.zeros(self.particle_count, dtype=float)
        max_relative = np.zeros(self.particle_count, dtype=float)
        takeoff_time = np.full(self.particle_count, np.nan, dtype=float)
        landing_time = np.full(self.particle_count, np.nan, dtype=float)
        duration = np.zeros(self.particle_count, dtype=float)
        launch_angle = np.full(self.particle_count, np.nan, dtype=float)
        elevation_angle = np.full(self.particle_count, np.nan, dtype=float)
        flight_count = np.zeros(self.particle_count, dtype=np.int32)
        total_duration = np.zeros(self.particle_count, dtype=float)
        for index in range(self.particle_count):
            particle_events = [event for event in self.events if event.particle_index == index]
            flight_count[index] = len(particle_events)
            if not particle_events:
                continue
            representative = max(
                particle_events,
                key=lambda event: (
                    event.max_height_relative_m,
                    event.max_height_world_m,
                ),
            )
            max_world[index] = max(event.max_height_world_m for event in particle_events)
            max_relative[index] = representative.max_height_relative_m
            takeoff_time[index] = representative.takeoff_time_s
            if representative.landing_time_s is not None:
                landing_time[index] = representative.landing_time_s
            duration[index] = float(representative.flight_duration_s or 0.0)
            total_duration[index] = sum(
                float(event.flight_duration_s or 0.0) for event in particle_events
            )
            launch_angle[index] = representative.launch_angle_xz_rad
            elevation_angle[index] = representative.elevation_angle_rad
        return {
            "max_flight_height_world_m": max_world,
            "max_flight_height_relative_m": max_relative,
            "takeoff_time_s": takeoff_time,
            "landing_time_s": landing_time,
            "flight_duration_s": duration,
            "total_flight_duration_s": total_duration,
            "launch_angle_xz_rad": launch_angle,
            "elevation_angle_rad": elevation_angle,
            "flight_count": flight_count,
        }


__all__ = [
    "ContactEventTracker",
    "ContactSnapshot",
    "FlightEvent",
    "collect_particle_pan_contacts",
    "quaternion_to_matrix",
]
