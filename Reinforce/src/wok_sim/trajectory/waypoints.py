"""P0~P5 웍질 waypoint 구성과 5회 반복 연결."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .parameters import TrajectoryParameters, trajectory_section


class WaypointError(ValueError):
    """waypoint 형상 또는 시간 순서가 유효하지 않을 때 발생한다."""


def _lookup(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _as_vector(value: Any, length: int, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise WaypointError(f"{name}은 길이 {length}의 숫자 배열이어야 합니다.") from exc
    if array.shape != (length,) or not np.isfinite(array).all():
        raise WaypointError(f"{name}은 유한한 길이 {length} 배열이어야 합니다: {value!r}")
    return array


@dataclass(frozen=True, slots=True)
class PanPose:
    """wok frame 기준 팬 pose(x, y, z, roll, pitch, yaw)."""

    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def as_array(self) -> np.ndarray:
        """6D pose를 float ndarray로 반환한다."""

        result = np.asarray((self.x, self.y, self.z, self.roll, self.pitch, self.yaw), dtype=float)
        if not np.isfinite(result).all():
            raise WaypointError("팬 pose에 NaN 또는 inf가 포함되어 있습니다.")
        return result

    @property
    def position_m(self) -> np.ndarray:
        """위치 [x, y, z](m)."""

        return self.as_array()[:3]

    @property
    def rpy_rad(self) -> np.ndarray:
        """자세 [roll, pitch, yaw](rad)."""

        return self.as_array()[3:]

    @classmethod
    def from_value(cls, value: Any) -> PanPose:
        """PanPose, 6-vector 또는 일반 pose mapping을 변환한다."""

        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping) or any(
            hasattr(value, key) for key in ("position_m", "position", "x", "y", "z")
        ):
            position = _lookup(value, "position_m", _lookup(value, "position", None))
            rpy = _lookup(value, "rpy_rad", _lookup(value, "orientation_rpy_rad", None))
            if position is not None:
                xyz = _as_vector(position, 3, "position_m")
                angles = np.zeros(3, dtype=float) if rpy is None else _as_vector(rpy, 3, "rpy_rad")
                return cls(*xyz, *angles)
            keys = ("x", "y", "z", "roll", "pitch", "yaw")
            values = [_lookup(value, key, None) for key in keys]
            if all(item is not None for item in values):
                return cls(*_as_vector(values, 6, "pose"))
        return cls(*_as_vector(value, 6, "pose"))


@dataclass(frozen=True, slots=True)
class Waypoint:
    """하나의 논리적 P0~P5 waypoint."""

    name: str
    cycle_index: int
    time_s: float
    pose: PanPose

    @property
    def position_m(self) -> np.ndarray:
        """waypoint 위치(m)."""

        return self.pose.position_m

    @property
    def orientation_rpy_rad(self) -> np.ndarray:
        """waypoint roll/pitch/yaw(rad)."""

        return self.pose.rpy_rad

    @property
    def x(self) -> float:
        """waypoint x(m)."""

        return self.pose.x

    @property
    def y(self) -> float:
        """waypoint y(m)."""

        return self.pose.y

    @property
    def z(self) -> float:
        """waypoint z(m)."""

        return self.pose.z

    @property
    def pitch(self) -> float:
        """waypoint pitch(rad)."""

        return self.pose.pitch


@dataclass(frozen=True, slots=True)
class WaypointSequence(Sequence[Waypoint]):
    """global spline 입력용으로 연결된 waypoint와 cycle 경계."""

    points: tuple[Waypoint, ...]
    cycle_boundaries_s: np.ndarray
    cycle_count: int

    def __post_init__(self) -> None:
        if self.cycle_count <= 0:
            raise WaypointError("cycle_count는 양수여야 합니다.")
        if len(self.points) < 6:
            raise WaypointError("quintic spline에는 최소 6개 waypoint가 필요합니다.")
        times = self.times
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0.0):
            raise WaypointError("waypoint 시간은 유한하고 엄격히 증가해야 합니다.")

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, index: int | slice) -> Waypoint | tuple[Waypoint, ...]:
        return self.points[index]

    def __iter__(self) -> Iterator[Waypoint]:
        return iter(self.points)

    @property
    def times(self) -> np.ndarray:
        """waypoint time axis(s)."""

        return np.asarray([point.time_s for point in self.points], dtype=float)

    @property
    def poses(self) -> np.ndarray:
        """waypoint 6D pose 행렬."""

        return np.vstack([point.pose.as_array() for point in self.points])

    @property
    def positions_m(self) -> np.ndarray:
        """모든 waypoint의 위치 행렬(m)."""

        return self.poses[:, :3]

    @property
    def orientation_rpy_rad(self) -> np.ndarray:
        """모든 waypoint의 RPY 행렬(rad)."""

        return self.poses[:, 3:]


def _start_pose_from_config(config: Any) -> PanPose:
    section = trajectory_section(config)
    position = _lookup(section, "start_position_m", None)
    rpy = _lookup(section, "start_rpy_rad", None)
    if position is not None:
        return PanPose.from_value(
            {
                "position_m": position,
                "rpy_rad": np.zeros(3) if rpy is None else rpy,
            }
        )

    pan = _lookup(config, "pan", None)
    initial = _lookup(pan, "initial_pose", None) if pan is not None else None
    if initial is not None:
        return PanPose.from_value(initial)
    return PanPose(0.0, 0.0, 0.0)


def _catch_offset(config: Any) -> np.ndarray:
    section = trajectory_section(config)
    raw = _lookup(section, "catch_offset_m", _lookup(section, "catch_offset", None))
    if raw is None:
        return np.zeros(3, dtype=float)
    if isinstance(raw, Mapping) or any(
        hasattr(raw, key) for key in ("x_m", "x", "y_m", "y", "z_m", "z")
    ):
        raw = (
            _lookup(raw, "x_m", _lookup(raw, "x", 0.0)),
            _lookup(raw, "y_m", _lookup(raw, "y", 0.0)),
            _lookup(raw, "z_m", _lookup(raw, "z", 0.0)),
        )
    result = _as_vector(raw, 3, "trajectory.catch_offset_m")
    if not np.isclose(result[1], 0.0, atol=1.0e-12):
        raise WaypointError("xz 평면 제약 때문에 catch_offset의 y 성분은 0이어야 합니다.")
    return result


def phase_times(parameters: TrajectoryParameters) -> np.ndarray:
    """P0~P5의 한 사이클 상대 시각을 계산한다.

    P1은 삽입 phase 끝, P4는 catch/복귀 phase 시작이다. P2와 P3은
    두 시각 사이를 같은 양의 시간 간격으로 나눠 급격한 방향 전환과
    시간 순서 역전을 방지한다.
    """

    insert_end = parameters.cycle_time * parameters.insert_phase_ratio
    catch_start = parameters.cycle_time * (1.0 - parameters.catch_phase_ratio)
    middle = np.linspace(insert_end, catch_start, 4)
    result = np.asarray(
        (0.0, insert_end, middle[1], middle[2], catch_start, parameters.cycle_time),
        dtype=float,
    )
    if np.any(np.diff(result) <= 0.0):
        raise WaypointError("모든 waypoint phase duration은 양수여야 합니다.")
    return result


def build_cycle_waypoints(
    parameters: TrajectoryParameters,
    config: Any,
    *,
    start_pose: PanPose | Sequence[float] | Mapping[str, Any] | None = None,
    cycle_index: int = 0,
    start_time_s: float = 0.0,
) -> tuple[Waypoint, ...]:
    """P0~P5 한 사이클을 요구된 xz 기하로 생성한다."""

    if cycle_index < 0:
        raise WaypointError("cycle_index는 0 이상이어야 합니다.")
    if not np.isfinite(start_time_s):
        raise WaypointError("start_time_s는 유한해야 합니다.")
    p0 = _start_pose_from_config(config) if start_pose is None else PanPose.from_value(start_pose)
    section = trajectory_section(config)
    try:
        angle_rad = np.deg2rad(float(_lookup(section, "insertion_angle_deg", 45.0)))
        transition_forward = float(_lookup(section, "transition_forward_offset_m", 0.0))
        transition_lift = float(_lookup(section, "transition_lift_offset_m", 0.0))
    except (TypeError, ValueError) as exc:
        raise WaypointError("trajectory waypoint offset 설정은 숫자여야 합니다.") from exc
    if not np.isfinite([angle_rad, transition_forward, transition_lift]).all():
        raise WaypointError("trajectory waypoint offset 설정은 유한해야 합니다.")

    distance = parameters.insertion_distance
    p1 = PanPose(
        p0.x + distance * np.cos(angle_rad),
        p0.y,
        p0.z - distance * np.sin(angle_rad),
        p0.roll,
        p0.pitch,
        p0.yaw,
    )
    p2 = PanPose(
        p1.x + transition_forward,
        p0.y,
        p1.z + transition_lift,
        p0.roll,
        p0.pitch + 0.5 * parameters.pitch_amplitude,
        p0.yaw,
    )
    # P3의 후퇴와 상승은 같은 P2->P3 시간구간에서 함께 일어난다.
    p3 = PanPose(
        p1.x - parameters.backward_distance,
        p0.y,
        p1.z + parameters.lift_height,
        p0.roll,
        p0.pitch + parameters.pitch_amplitude,
        p0.yaw,
    )
    if not p3.x < p2.x or not p3.z > p2.z:
        raise WaypointError(
            "launch P2->P3에서는 x 후퇴와 z 상승이 동시에 양수여야 합니다. "
            "transition offset과 action 범위를 확인하세요."
        )

    catch = _catch_offset(config)
    p4 = PanPose(
        p0.x + catch[0],
        p0.y,
        p0.z + catch[2],
        p0.roll,
        p0.pitch + 0.5 * parameters.pitch_amplitude,
        p0.yaw,
    )
    p5 = p0
    times = phase_times(parameters) + float(start_time_s)
    poses = (p0, p1, p2, p3, p4, p5)
    return tuple(
        Waypoint(f"P{index}", cycle_index, float(time_s), pose)
        for index, (time_s, pose) in enumerate(zip(times, poses, strict=True))
    )


def build_repeated_waypoints(
    parameters: TrajectoryParameters,
    config: Any,
    *,
    start_pose: PanPose | Sequence[float] | Mapping[str, Any] | None = None,
    cycles: int | None = None,
) -> WaypointSequence:
    """동일 action의 사이클들을 중복 P0 없이 하나의 waypoint 열로 연결한다."""

    section = trajectory_section(config)
    raw_cycles = _lookup(section, "cycles", 5) if cycles is None else cycles
    try:
        cycle_count = int(raw_cycles)
    except (TypeError, ValueError) as exc:
        raise WaypointError("trajectory.cycles는 정수여야 합니다.") from exc
    if cycle_count <= 0 or cycle_count != raw_cycles:
        raise WaypointError("trajectory.cycles는 양의 정수여야 합니다.")

    resolved_start = (
        _start_pose_from_config(config) if start_pose is None else PanPose.from_value(start_pose)
    )
    joined: list[Waypoint] = []
    for cycle_index in range(cycle_count):
        cycle = build_cycle_waypoints(
            parameters,
            config,
            start_pose=resolved_start,
            cycle_index=cycle_index,
            start_time_s=cycle_index * parameters.cycle_time,
        )
        # 이전 P5와 현재 P0는 같은 시각/pose이므로 한 점으로 공유한다.
        joined.extend(cycle if cycle_index == 0 else cycle[1:])
    boundaries = np.arange(cycle_count + 1, dtype=float) * parameters.cycle_time
    return WaypointSequence(tuple(joined), boundaries, cycle_count)


def build_waypoints(
    parameters: TrajectoryParameters,
    config: Any,
    *,
    start_pose: PanPose | Sequence[float] | Mapping[str, Any] | None = None,
    cycles: int | None = None,
) -> WaypointSequence:
    """``build_repeated_waypoints``의 짧은 public alias."""

    return build_repeated_waypoints(parameters, config, start_pose=start_pose, cycles=cycles)
