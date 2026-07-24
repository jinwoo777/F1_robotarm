"""Kinematic pan을 따라 sphere particle을 실행하는 MuJoCo simulator."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .contact_events import (
    ContactEventTracker,
    ContactSnapshot,
    FlightEvent,
    collect_particle_pan_contacts,
    quaternion_to_matrix,
)
from .model_builder import BuiltModel, ModelBuilder, ParticleModelSpec


class SimulationError(RuntimeError):
    """물리 rollout을 완료할 수 없을 때 발생한다."""


class InvalidTrajectoryError(SimulationError):
    """validation 실패 trajectory가 simulator에 전달되었을 때 발생한다."""

    def __init__(self, violations: Sequence[str] | None = None) -> None:
        self.violations = tuple(str(item) for item in (violations or ()))
        message = "invalid trajectory이므로 입자 simulation을 실행하지 않습니다."
        if self.violations:
            message += " 위반: " + "; ".join(self.violations)
        super().__init__(message)


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"mapping 설정이 필요합니다. 받은 타입: {type(value).__name__}")


def _get(value: Any, *names: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    else:
        for name in names:
            if hasattr(value, name):
                return getattr(value, name)
    return default


def rpy_to_quaternion_wxyz(rpy_rad: Sequence[float]) -> np.ndarray:
    """roll-pitch-yaw(Rz*Ry*Rx)를 MuJoCo wxyz quaternion으로 변환한다."""

    roll, pitch, yaw = np.asarray(rpy_rad, dtype=float)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    quaternion = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=float,
    )
    return quaternion / np.linalg.norm(quaternion)


def quaternion_wxyz_to_rpy(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    """wxyz quaternion을 roll-pitch-yaw로 변환한다."""

    w, x, y, z = np.asarray(quaternion_wxyz, dtype=float)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not np.isfinite(norm) or norm < 1e-12:
        raise SimulationError("trajectory quaternion이 유효하지 않습니다.")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_argument = float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    pitch = math.asin(pitch_argument)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=float)


@dataclass(frozen=True)
class TrajectorySample:
    """한 시각의 pan pose 및 속도."""

    position_m: np.ndarray
    rpy_rad: np.ndarray
    quaternion_wxyz: np.ndarray
    linear_velocity_m_s: np.ndarray
    angular_velocity_rad_s: np.ndarray


class TrajectoryView:
    """trajectory dataclass/mapping을 simulator용 공통 interface로 정규화한다."""

    def __init__(self, trajectory: Any) -> None:
        self.source = trajectory
        time_value = _get(trajectory, "time_s", "times", "timestamps", "t", default=None)
        if time_value is None:
            raise SimulationError("trajectory에 time_s/times 배열이 없습니다.")
        self.time_s = np.asarray(time_value, dtype=float)
        if (
            self.time_s.ndim != 1
            or len(self.time_s) < 2
            or not np.isfinite(self.time_s).all()
            or np.any(np.diff(self.time_s) <= 0.0)
        ):
            raise SimulationError(
                "trajectory time은 2개 이상의 유한하고 엄격히 증가하는 1차원 배열이어야 합니다."
            )

        pose = _get(trajectory, "pose", default=None)
        position = _get(
            trajectory, "position_m", "positions", "positions_m", "position", default=None
        )
        orientation = _get(
            trajectory,
            "orientation_rpy_rad",
            "rpy_rad",
            "orientations_rpy_rad",
            default=None,
        )
        quaternion = _get(
            trajectory,
            "quaternion_wxyz",
            "quaternions_wxyz",
            "quaternions",
            default=None,
        )
        if pose is not None:
            pose_array = np.asarray(pose, dtype=float)
            if pose_array.shape == (len(self.time_s), 6):
                if position is None:
                    position = pose_array[:, :3]
                if orientation is None:
                    orientation = pose_array[:, 3:]
        self.position_m = np.asarray(position, dtype=float)
        if self.position_m.shape != (len(self.time_s), 3):
            raise SimulationError("trajectory position shape은 (N, 3)이어야 합니다.")

        if orientation is not None:
            self.rpy_rad = np.asarray(orientation, dtype=float)
        elif quaternion is not None:
            quaternion_array = np.asarray(quaternion, dtype=float)
            if quaternion_array.shape != (len(self.time_s), 4):
                raise SimulationError("trajectory quaternion shape은 (N, 4)이어야 합니다.")
            self.rpy_rad = np.vstack([quaternion_wxyz_to_rpy(item) for item in quaternion_array])
        else:
            pitch = _get(trajectory, "pitch_rad", "pitch", default=None)
            if pitch is None:
                self.rpy_rad = np.zeros((len(self.time_s), 3), dtype=float)
            else:
                pitch_array = np.asarray(pitch, dtype=float)
                if pitch_array.shape != (len(self.time_s),):
                    raise SimulationError("trajectory pitch shape은 (N,)이어야 합니다.")
                self.rpy_rad = np.zeros((len(self.time_s), 3), dtype=float)
                self.rpy_rad[:, 1] = pitch_array
        if self.rpy_rad.shape != (len(self.time_s), 3):
            raise SimulationError("trajectory orientation_rpy_rad shape은 (N, 3)이어야 합니다.")

        linear_velocity = _get(
            trajectory,
            "linear_velocity_m_s",
            "linear_velocity",
            default=None,
        )
        angular_velocity = _get(
            trajectory,
            "angular_velocity_rad_s",
            "angular_velocity",
            default=None,
        )
        velocity = _get(trajectory, "velocity", default=None)
        if velocity is not None:
            velocity_array = np.asarray(velocity, dtype=float)
            if velocity_array.shape == (len(self.time_s), 6):
                if linear_velocity is None:
                    linear_velocity = velocity_array[:, :3]
                if angular_velocity is None:
                    angular_velocity = velocity_array[:, 3:]
        edge_order = 2 if len(self.time_s) >= 3 else 1
        self.linear_velocity_m_s = (
            np.gradient(self.position_m, self.time_s, axis=0, edge_order=edge_order)
            if linear_velocity is None
            else np.asarray(linear_velocity, dtype=float)
        )
        self.angular_velocity_rad_s = (
            np.gradient(self.rpy_rad, self.time_s, axis=0, edge_order=edge_order)
            if angular_velocity is None
            else np.asarray(angular_velocity, dtype=float)
        )
        expected = (len(self.time_s), 3)
        if self.linear_velocity_m_s.shape != expected:
            raise SimulationError("trajectory linear velocity shape은 (N, 3)이어야 합니다.")
        if self.angular_velocity_rad_s.shape != expected:
            raise SimulationError("trajectory angular velocity shape은 (N, 3)이어야 합니다.")
        arrays = (
            self.position_m,
            self.rpy_rad,
            self.linear_velocity_m_s,
            self.angular_velocity_rad_s,
        )
        if not all(np.isfinite(value).all() for value in arrays):
            raise SimulationError("trajectory에 NaN 또는 inf가 포함되어 있습니다.")

        validation = _get(trajectory, "validation", default=None)
        raw_valid = _get(validation, "valid", default=None)
        if raw_valid is None:
            raw_valid = _get(trajectory, "valid", default=None)
        self.valid = None if raw_valid is None else bool(raw_valid)
        violations = _get(validation, "violations", "reasons", default=())
        self.violations = tuple(str(item) for item in (violations or ()))
        self.cycle_count = _get(trajectory, "cycle_count", default=None)
        self._analytic_evaluate = _get(trajectory, "evaluate", default=None)

    @property
    def start_time_s(self) -> float:
        return float(self.time_s[0])

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0])

    def sample(self, relative_time_s: float) -> TrajectorySample:
        """시작을 0으로 본 rollout 시각에서 pose/velocity를 평가한다."""

        relative_time = float(np.clip(relative_time_s, 0.0, self.duration_s))
        query = self.start_time_s + relative_time
        if callable(self._analytic_evaluate):
            pose = np.asarray(self._analytic_evaluate(query, derivative=0), dtype=float)
            derivative = np.asarray(self._analytic_evaluate(query, derivative=1), dtype=float)
            if pose.shape == (6,) and derivative.shape == (6,):
                position = pose[:3]
                rpy = pose[3:]
                linear_velocity = derivative[:3]
                angular_velocity = derivative[3:]
            else:
                position, rpy, linear_velocity, angular_velocity = self._interpolate(query)
        else:
            position, rpy, linear_velocity, angular_velocity = self._interpolate(query)
        return TrajectorySample(
            position_m=np.asarray(position, dtype=float),
            rpy_rad=np.asarray(rpy, dtype=float),
            quaternion_wxyz=rpy_to_quaternion_wxyz(rpy),
            linear_velocity_m_s=np.asarray(linear_velocity, dtype=float),
            angular_velocity_rad_s=np.asarray(angular_velocity, dtype=float),
        )

    def _interpolate(self, query: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        def interpolate(array: np.ndarray) -> np.ndarray:
            return np.asarray([np.interp(query, self.time_s, array[:, axis]) for axis in range(3)])

        return (
            interpolate(self.position_m),
            interpolate(self.rpy_rad),
            interpolate(self.linear_velocity_m_s),
            interpolate(self.angular_velocity_rad_s),
        )


@dataclass
class SimulationResult:
    """전체 rollout에서 기록한 입자/pan/contact state."""

    time_s: np.ndarray
    particle_positions_world_m: np.ndarray
    particle_velocities_world_m_s: np.ndarray
    particle_positions_pan_m: np.ndarray
    pan_position_world_m: np.ndarray
    pan_quaternion_wxyz: np.ndarray
    contact_with_pan: np.ndarray
    contact_normal_force_n: np.ndarray
    final_no_contact_duration_s: np.ndarray
    crossed_spill_boundary: np.ndarray
    flight_events: tuple[FlightEvent, ...]
    particle_flight_summary: dict[str, np.ndarray]
    rgb_frames: tuple[np.ndarray, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def positions(self) -> np.ndarray:
        """particle_positions_world_m 호환 alias."""

        return self.particle_positions_world_m

    @property
    def velocities(self) -> np.ndarray:
        """particle_velocities_world_m_s 호환 alias."""

        return self.particle_velocities_world_m_s

    @property
    def contacts(self) -> np.ndarray:
        """contact_with_pan 호환 alias."""

        return self.contact_with_pan

    @property
    def final_positions_world_m(self) -> np.ndarray:
        return self.particle_positions_world_m[-1]

    @property
    def final_positions_pan_m(self) -> np.ndarray:
        return self.particle_positions_pan_m[-1]

    @property
    def initial_positions_pan_m(self) -> np.ndarray:
        return self.particle_positions_pan_m[0]


class WokSimulator:
    """MuJoCo kinematic pan simulator.

    ``rollout``은 전달된 trajectory 전체를 단 한 번 실행하며, 입자 상태를
    trajectory 생성/수정에 되먹임하지 않는다.
    """

    metadata = {
        "render_modes": [None, "human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        config: Mapping[str, Any] | Any,
        particles: ParticleModelSpec | Any,
        *,
        render_mode: str | None = None,
        positions_are_pan_local: bool | None = None,
    ) -> None:
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode는 None, 'human', 'rgb_array' 중 하나여야 합니다.")
        self.config = _mapping(config)
        self.render_mode = render_mode
        self.builder = ModelBuilder(config)
        self.built: BuiltModel = self.builder.build(
            particles, positions_are_pan_local=positions_are_pan_local
        )
        self.model = self.built.model
        self.data = self.built.data
        try:
            import mujoco
        except ImportError as exc:  # ModelBuilder가 이미 확인하지만 type checker용 방어
            raise SimulationError("MuJoCo를 import할 수 없습니다.") from exc
        self._mujoco = mujoco
        self._renderer: Any = None
        self._viewer: Any = None
        self._render_failure: Exception | None = None
        self._closed = False
        self._settled = False
        self.rollout_count = 0
        self.physics_step_count = 0
        self._initial_qpos = self.data.qpos.copy()
        self._initial_qvel = self.data.qvel.copy()
        particle_config = _mapping(self.config.get("particles", {}))
        default_linear_damping = float(particle_config.get("linear_damping", 0.0))
        default_angular_damping = float(particle_config.get("angular_damping", 0.0))
        self._linear_damping_per_s = (
            np.full(self.particle_count, default_linear_damping, dtype=float)
            if self.built.particles.linear_damping_per_s is None
            else np.asarray(
                self.built.particles.linear_damping_per_s,
                dtype=float,
            ).copy()
        )
        self._angular_damping_per_s = (
            np.full(self.particle_count, default_angular_damping, dtype=float)
            if self.built.particles.angular_damping_per_s is None
            else np.asarray(
                self.built.particles.angular_damping_per_s,
                dtype=float,
            ).copy()
        )
        if (
            self._linear_damping_per_s.shape != (self.particle_count,)
            or self._angular_damping_per_s.shape != (self.particle_count,)
            or not np.isfinite(self._linear_damping_per_s).all()
            or np.any(self._linear_damping_per_s < 0.0)
            or not np.isfinite(self._angular_damping_per_s).all()
            or np.any(self._angular_damping_per_s < 0.0)
        ):
            raise ValueError(
                "particle별 damping은 shape (N,)의 음수가 아닌 유한한 1/s 값이어야 합니다."
            )
        self._set_pan_pose(*self.built.pan.initial_pose)
        self._mujoco.mj_forward(self.model, self.data)

    @property
    def particle_count(self) -> int:
        return len(self.built.particle_body_ids)

    @property
    def settled(self) -> bool:
        return self._settled

    def reset(self) -> None:
        """초기 particle/pan 상태로 되돌린다."""

        self._assert_open()
        self._mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._initial_qpos
        self.data.qvel[:] = self._initial_qvel
        self._set_pan_pose(*self.built.pan.initial_pose)
        self._mujoco.mj_forward(self.model, self.data)
        self._settled = False
        self.rollout_count = 0
        self.physics_step_count = 0

    def particle_state(self) -> tuple[np.ndarray, np.ndarray]:
        """현재 입자 world position/linear velocity의 복사본을 반환한다."""

        self._assert_open()
        return self._particle_state()

    def particle_positions_pan(self) -> np.ndarray:
        """현재 입자 중심을 현재 pan-local frame으로 변환한다."""

        self._assert_open()
        positions, _ = self._particle_state()
        pan_position = self.data.mocap_pos[self.built.pan_mocap_id]
        pan_quaternion = self.data.mocap_quat[self.built.pan_mocap_id]
        return self._world_to_pan(positions, pan_position, pan_quaternion)

    def contact_snapshot(self) -> ContactSnapshot:
        """현재 particle-pan contact snapshot을 반환한다."""

        self._assert_open()
        return self._contacts()

    def settle(
        self,
        *,
        max_time_s: float | None = None,
        velocity_threshold_m_s: float | None = None,
    ) -> dict[str, float | int | bool]:
        """팬을 정지시킨 채 입자 최대 속도가 threshold 이하가 될 때까지 적분한다."""

        self._assert_open()
        particle_config = _mapping(self.config.get("particles", {}))
        if max_time_s is None:
            max_time_s = float(
                particle_config.get("settle_time_s", particle_config.get("settle_max_time_s", 0.8))
            )
        if velocity_threshold_m_s is None:
            velocity_threshold_m_s = float(particle_config.get("settle_velocity_threshold", 0.03))
        max_time = float(max_time_s)
        threshold = float(velocity_threshold_m_s)
        if not np.isfinite(max_time) or max_time < 0.0:
            raise ValueError("settle max_time_s는 음수가 아닌 유한값이어야 합니다.")
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("settle velocity threshold는 음수가 아닌 유한값이어야 합니다.")
        if max_time == 0.0:
            self._settled = True
            return {
                "settled": True,
                "elapsed_s": 0.0,
                "steps": 0,
                "max_speed_m_s": float(np.max(np.linalg.norm(self._particle_state()[1], axis=1))),
            }

        simulation = _mapping(self.config.get("simulation", {}))
        minimum_time = min(
            max_time,
            float(particle_config.get("settle_min_time_s", min(0.05, max_time))),
        )
        stable_time_required = float(
            particle_config.get("settle_stable_time_s", min(0.04, max_time))
        )
        stable_elapsed = 0.0
        elapsed = 0.0
        steps = 0
        max_speed = float("inf")
        dt = float(self.model.opt.timestep)
        pan_position, pan_quaternion = self.built.pan.initial_pose
        self._set_pan_pose(pan_position, pan_quaternion)
        while elapsed + 0.5 * dt < max_time:
            self._mujoco.mj_step(self.model, self.data)
            self._apply_particle_damping(dt)
            self.physics_step_count += 1
            steps += 1
            elapsed += dt
            _, velocities = self._particle_state()
            max_speed = float(np.max(np.linalg.norm(velocities, axis=1)))
            if elapsed >= minimum_time and max_speed <= threshold:
                stable_elapsed += dt
                if stable_elapsed >= stable_time_required:
                    break
            else:
                stable_elapsed = 0.0
            if (
                self.render_mode == "human"
                and steps % int(simulation.get("human_render_stride", 5)) == 0
            ):
                self.render()
        self._settled = True
        return {
            "settled": bool(max_speed <= threshold),
            "elapsed_s": elapsed,
            "steps": steps,
            "max_speed_m_s": max_speed,
        }

    def rollout(
        self,
        trajectory: Any,
        *,
        record_rgb: bool | None = None,
    ) -> SimulationResult:
        """validation을 통과한 전체 trajectory를 실행하고 state/contact를 기록한다."""

        self._assert_open()
        view = trajectory if isinstance(trajectory, TrajectoryView) else TrajectoryView(trajectory)
        if view.valid is False:
            # 이 검사 전에는 mj_step이나 mocap 변경을 하지 않는다.
            raise InvalidTrajectoryError(view.violations)
        if self.rollout_count:
            raise SimulationError(
                "같은 WokSimulator에서 rollout은 한 번만 허용됩니다. "
                "새 episode에는 reset 후 settle하거나 새 simulator를 만드세요."
            )
        if not self._settled:
            self.settle()

        simulation = _mapping(self.config.get("simulation", {}))
        logging_config = _mapping(self.config.get("logging", {}))
        spill_config = _mapping(
            self.config.get("spill", self.config.get("metrics", {}).get("spill", {}))
        )
        record_stride = max(1, int(simulation.get("record_every_n_steps", 1)))
        human_stride = max(1, int(simulation.get("human_render_stride", 5)))
        rgb_stride = max(1, int(simulation.get("rgb_record_every_n_steps", record_stride)))
        if record_rgb is None:
            record_rgb = bool(logging_config.get("save_video", False))

        spill_radius = float(
            spill_config.get(
                "spill_boundary_radius_m",
                simulation.get(
                    "spill_boundary_radius_m",
                    spill_config.get(
                        "boundary_radius_m", self.built.pan.proxy.inner_radius_m * 1.35
                    ),
                ),
            )
        )
        below_rim = float(simulation.get("spill_boundary_below_rim_m", 0.08))
        spill_below_z = float(
            spill_config.get(
                "spill_below_pan_z_m",
                self.built.pan.proxy.rim_z_m
                - float(
                    spill_config.get(
                        "below_rim_threshold_m",
                        below_rim,
                    )
                ),
            )
        )
        if spill_radius <= 0.0:
            raise SimulationError("spill boundary radius는 양수여야 합니다.")

        initial_sample = view.sample(0.0)
        settled_position, settled_quaternion = self.built.pan.initial_pose
        position_error = float(np.linalg.norm(initial_sample.position_m - settled_position))
        quaternion_dot = float(
            np.clip(
                abs(
                    np.dot(
                        initial_sample.quaternion_wxyz,
                        settled_quaternion,
                    )
                ),
                0.0,
                1.0,
            )
        )
        orientation_error = 2.0 * math.acos(quaternion_dot)
        position_tolerance = float(simulation.get("frame_position_tolerance_m", 1e-8))
        orientation_tolerance = float(simulation.get("frame_orientation_tolerance_rad", 1e-8))
        if position_error > position_tolerance or orientation_error > orientation_tolerance:
            raise InvalidTrajectoryError(
                (
                    "trajectory_start_pose_mismatch: "
                    f"position_error={position_error:.9g}m "
                    f"(tol={position_tolerance:.9g}), "
                    f"orientation_error={orientation_error:.9g}rad "
                    f"(tol={orientation_tolerance:.9g})",
                )
            )
        self._set_pan_pose(initial_sample.position_m, initial_sample.quaternion_wxyz)
        self._mujoco.mj_forward(self.model, self.data)
        initial_contacts = self._contacts()
        tracker = ContactEventTracker(
            self.particle_count,
            takeoff_vertical_velocity_threshold_m_s=float(
                simulation.get("takeoff_velocity_threshold_m_s", 1e-4)
            ),
            contact_gap_tolerance_s=float(simulation.get("contact_gap_tolerance_s", 0.0)),
        )
        tracker.reset(initial_contacts.touching_pan)

        times: list[float] = []
        positions_world_history: list[np.ndarray] = []
        velocities_history: list[np.ndarray] = []
        positions_pan_history: list[np.ndarray] = []
        pan_position_history: list[np.ndarray] = []
        pan_quaternion_history: list[np.ndarray] = []
        contact_history: list[np.ndarray] = []
        force_history: list[np.ndarray] = []
        rgb_frames: list[np.ndarray] = []
        crossed_spill = np.zeros(self.particle_count, dtype=bool)
        no_contact_duration_s = np.zeros(self.particle_count, dtype=float)

        def record(
            relative_time: float,
            sample: TrajectorySample,
            positions_world: np.ndarray,
            velocities_world: np.ndarray,
            contacts: ContactSnapshot,
        ) -> None:
            positions_pan = self._world_to_pan(
                positions_world, sample.position_m, sample.quaternion_wxyz
            )
            times.append(float(relative_time))
            positions_world_history.append(positions_world.copy())
            velocities_history.append(velocities_world.copy())
            positions_pan_history.append(positions_pan)
            pan_position_history.append(sample.position_m.copy())
            pan_quaternion_history.append(sample.quaternion_wxyz.copy())
            contact_history.append(contacts.touching_pan.copy())
            force_history.append(contacts.normal_force_n.copy())

        initial_positions, initial_velocities = self._particle_state()
        record(
            0.0,
            initial_sample,
            initial_positions,
            initial_velocities,
            initial_contacts,
        )
        if record_rgb and self.render_mode == "rgb_array":
            frame = self.render()
            if frame is not None:
                rgb_frames.append(frame.copy())

        dt = float(self.model.opt.timestep)
        total_steps = max(1, int(math.ceil(view.duration_s / dt - 1.0e-12)))
        post_rollout_settle_s = max(0.0, float(simulation.get("post_rollout_settle_s", 0.0)))
        post_steps = (
            int(math.ceil(post_rollout_settle_s / dt - 1.0e-12))
            if post_rollout_settle_s > 0.0
            else 0
        )
        maximum_steps = int(simulation.get("max_steps", total_steps + post_steps))
        if total_steps + post_steps > maximum_steps:
            raise SimulationError(
                "trajectory physics step 수가 simulation.max_steps를 초과합니다: "
                f"{total_steps + post_steps} > {maximum_steps}"
            )
        last_time = 0.0
        for step_index in range(1, total_steps + 1):
            relative_time = min(step_index * dt, view.duration_s)
            step_duration = relative_time - last_time
            if step_duration <= 0.0:
                raise SimulationError("trajectory physics step duration이 양수가 아닙니다.")
            sample = view.sample(relative_time)
            self._set_pan_pose(sample.position_m, sample.quaternion_wxyz)
            self.model.opt.timestep = step_duration
            try:
                self._mujoco.mj_step(self.model, self.data)
            finally:
                self.model.opt.timestep = dt
            self._apply_particle_damping(step_duration)
            self.physics_step_count += 1
            positions_world, velocities_world = self._particle_state()
            contacts = self._contacts()
            no_contact_duration_s = np.where(
                contacts.touching_pan,
                0.0,
                no_contact_duration_s + step_duration,
            )
            positions_pan = self._world_to_pan(
                positions_world, sample.position_m, sample.quaternion_wxyz
            )
            crossed_spill |= (np.linalg.norm(positions_pan[:, :2], axis=1) > spill_radius) | (
                positions_pan[:, 2] < spill_below_z
            )
            tracker.update(
                relative_time,
                contacts.touching_pan,
                positions_world,
                velocities_world,
                pan_position_world_m=sample.position_m,
                pan_quaternion_wxyz=sample.quaternion_wxyz,
                pan_linear_velocity_world_m_s=sample.linear_velocity_m_s,
                pan_angular_velocity_world_rad_s=sample.angular_velocity_rad_s,
            )
            should_record = step_index % record_stride == 0 or step_index == total_steps
            if should_record:
                record(
                    relative_time,
                    sample,
                    positions_world,
                    velocities_world,
                    contacts,
                )
            if self.render_mode == "human" and step_index % human_stride == 0:
                self.render()
            if (
                record_rgb
                and self.render_mode == "rgb_array"
                and (step_index % rgb_stride == 0 or step_index == total_steps)
            ):
                frame = self.render()
                if frame is not None:
                    rgb_frames.append(frame.copy())
            last_time = relative_time

        # P5에서 팬은 정지한 채 재료가 다시 팬으로 돌아올 시간을 준다.
        final_sample = view.sample(view.duration_s)
        resting_sample = TrajectorySample(
            position_m=final_sample.position_m,
            rpy_rad=final_sample.rpy_rad,
            quaternion_wxyz=final_sample.quaternion_wxyz,
            linear_velocity_m_s=np.zeros(3, dtype=float),
            angular_velocity_rad_s=np.zeros(3, dtype=float),
        )
        for post_index in range(1, post_steps + 1):
            relative_time = view.duration_s + min(post_index * dt, post_rollout_settle_s)
            step_duration = relative_time - last_time
            if step_duration <= 0.0:
                raise SimulationError("post-rollout physics step duration이 양수가 아닙니다.")
            self._set_pan_pose(resting_sample.position_m, resting_sample.quaternion_wxyz)
            self.model.opt.timestep = step_duration
            try:
                self._mujoco.mj_step(self.model, self.data)
            finally:
                self.model.opt.timestep = dt
            self._apply_particle_damping(step_duration)
            self.physics_step_count += 1
            positions_world, velocities_world = self._particle_state()
            contacts = self._contacts()
            no_contact_duration_s = np.where(
                contacts.touching_pan,
                0.0,
                no_contact_duration_s + step_duration,
            )
            positions_pan = self._world_to_pan(
                positions_world,
                resting_sample.position_m,
                resting_sample.quaternion_wxyz,
            )
            crossed_spill |= (np.linalg.norm(positions_pan[:, :2], axis=1) > spill_radius) | (
                positions_pan[:, 2] < spill_below_z
            )
            tracker.update(
                relative_time,
                contacts.touching_pan,
                positions_world,
                velocities_world,
                pan_position_world_m=resting_sample.position_m,
                pan_quaternion_wxyz=resting_sample.quaternion_wxyz,
            )
            absolute_index = total_steps + post_index
            should_record = absolute_index % record_stride == 0 or post_index == post_steps
            if should_record:
                record(
                    relative_time,
                    resting_sample,
                    positions_world,
                    velocities_world,
                    contacts,
                )
            if self.render_mode == "human" and absolute_index % human_stride == 0:
                self.render()
            if (
                record_rgb
                and self.render_mode == "rgb_array"
                and (absolute_index % rgb_stride == 0 or post_index == post_steps)
            ):
                frame = self.render()
                if frame is not None:
                    rgb_frames.append(frame.copy())
            last_time = relative_time

        tracker.finalize(last_time)
        self.rollout_count += 1
        result = SimulationResult(
            time_s=np.asarray(times, dtype=float),
            particle_positions_world_m=np.asarray(positions_world_history, dtype=float),
            particle_velocities_world_m_s=np.asarray(velocities_history, dtype=float),
            particle_positions_pan_m=np.asarray(positions_pan_history, dtype=float),
            pan_position_world_m=np.asarray(pan_position_history, dtype=float),
            pan_quaternion_wxyz=np.asarray(pan_quaternion_history, dtype=float),
            contact_with_pan=np.asarray(contact_history, dtype=bool),
            contact_normal_force_n=np.asarray(force_history, dtype=float),
            final_no_contact_duration_s=no_contact_duration_s.copy(),
            crossed_spill_boundary=crossed_spill,
            flight_events=tuple(tracker.events),
            particle_flight_summary=tracker.particle_summary(),
            rgb_frames=tuple(rgb_frames),
            metadata={
                **self.built.metadata,
                "trajectory_duration_s": view.duration_s,
                "trajectory_cycle_count": view.cycle_count,
                "physics_steps": total_steps + post_steps,
                "physics_timestep_s": dt,
                "fractional_final_step_enabled": True,
                "physics_elapsed_s": last_time,
                "state_record_stride": record_stride,
                "post_rollout_settle_s": post_rollout_settle_s,
                "open_loop_external_control": True,
            },
        )
        self._validate_result(result)
        return result

    def _set_pan_pose(
        self,
        position_m: Sequence[float],
        quaternion_wxyz: Sequence[float],
    ) -> None:
        position = np.asarray(position_m, dtype=float)
        quaternion = np.asarray(quaternion_wxyz, dtype=float)
        if position.shape != (3,) or quaternion.shape != (4,):
            raise SimulationError("pan pose shape은 position(3), quaternion(4)이어야 합니다.")
        if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
            raise SimulationError("pan pose에 NaN 또는 inf가 있습니다.")
        norm = float(np.linalg.norm(quaternion))
        if norm < 1e-12:
            raise SimulationError("pan quaternion norm이 0입니다.")
        self.data.mocap_pos[self.built.pan_mocap_id] = position
        self.data.mocap_quat[self.built.pan_mocap_id] = quaternion / norm

    def _particle_state(self) -> tuple[np.ndarray, np.ndarray]:
        positions = np.asarray(self.data.xpos[self.built.particle_body_ids], dtype=float).copy()
        velocities = np.vstack(
            [
                np.asarray(self.data.qvel[address : address + 3], dtype=float)
                for address in self.built.particle_dof_addresses
            ]
        )
        return positions, velocities

    def _apply_particle_damping(self, timestep_s: float) -> None:
        """config/batch의 입자별 damping(1/s)을 free-body qvel에 적용한다."""

        linear_factor = np.exp(-self._linear_damping_per_s * timestep_s)
        angular_factor = np.exp(-self._angular_damping_per_s * timestep_s)
        for index, address in enumerate(self.built.particle_dof_addresses):
            self.data.qvel[address : address + 3] *= linear_factor[index]
            self.data.qvel[address + 3 : address + 6] *= angular_factor[index]

    def _contacts(self) -> ContactSnapshot:
        return collect_particle_pan_contacts(
            self.model,
            self.data,
            self.built.particle_geom_ids,
            self.built.pan_collision_geom_ids,
        )

    @staticmethod
    def _world_to_pan(
        positions_world_m: np.ndarray,
        pan_position_world_m: Sequence[float],
        pan_quaternion_wxyz: Sequence[float],
    ) -> np.ndarray:
        rotation = quaternion_to_matrix(pan_quaternion_wxyz)
        return (
            np.asarray(positions_world_m, dtype=float)
            - np.asarray(pan_position_world_m, dtype=float)
        ) @ rotation

    @staticmethod
    def _validate_result(result: SimulationResult) -> None:
        numeric_arrays = (
            result.time_s,
            result.particle_positions_world_m,
            result.particle_velocities_world_m_s,
            result.particle_positions_pan_m,
            result.pan_position_world_m,
            result.pan_quaternion_wxyz,
            result.contact_normal_force_n,
            result.final_no_contact_duration_s,
        )
        if not all(np.isfinite(array).all() for array in numeric_arrays):
            raise SimulationError("simulation 결과에 NaN 또는 inf가 발생했습니다.")
        if np.any(np.diff(result.time_s) <= 0.0):
            raise SimulationError("recorded state time이 엄격히 증가하지 않습니다.")
        if np.any(result.final_no_contact_duration_s < 0.0):
            raise SimulationError("final no-contact duration이 음수입니다.")

    def render(self) -> np.ndarray | None:
        """human viewer를 sync하거나 rgb_array frame을 반환한다."""

        self._assert_open()
        if self.render_mode is None:
            return None
        if self.render_mode == "human":
            if self._viewer is None:
                try:
                    from mujoco import viewer

                    self._viewer = viewer.launch_passive(self.model, self.data)
                except Exception as exc:
                    raise SimulationError(
                        f"interactive MuJoCo viewer를 열 수 없습니다: {exc}"
                    ) from exc
            self._viewer.sync()
            return None

        if self._render_failure is not None:
            raise SimulationError(
                f"MuJoCo offscreen rendering을 사용할 수 없습니다: {self._render_failure}"
            ) from self._render_failure
        if self._renderer is None:
            render_config = _mapping(self.config.get("render", {}))
            try:
                self._renderer = self._mujoco.Renderer(
                    self.model,
                    height=int(render_config.get("height", 480)),
                    width=int(render_config.get("width", 640)),
                )
            except Exception as exc:
                self._render_failure = exc
                raise SimulationError(
                    "MuJoCo offscreen renderer 생성에 실패했습니다. headless EGL/OSMesa "
                    f"설정을 확인하세요: {exc}"
                ) from exc
        try:
            self._renderer.update_scene(self.data, camera="overview")
            return np.asarray(self._renderer.render()).copy()
        except Exception as exc:
            self._render_failure = exc
            raise SimulationError(f"MuJoCo rgb_array render 실패: {exc}") from exc

    def close(self) -> None:
        """renderer/viewer resource를 안전하게 닫는다."""

        if self._closed:
            return
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None
        self._closed = True

    def _assert_open(self) -> None:
        if self._closed:
            raise SimulationError("이미 close된 WokSimulator입니다.")

    def __enter__(self) -> WokSimulator:
        self._assert_open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# 외부 코드에서 간결하게 사용할 수 있는 호환 alias.
Simulator = WokSimulator


__all__ = [
    "InvalidTrajectoryError",
    "SimulationError",
    "SimulationResult",
    "Simulator",
    "TrajectorySample",
    "TrajectoryView",
    "WokSimulator",
    "quaternion_wxyz_to_rpy",
    "rpy_to_quaternion_wxyz",
]
