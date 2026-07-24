"""MuJoCo model과 one-step Gym 환경 통합 테스트."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import pytest

from wok_sim.config import load_config
from wok_sim.envs import EpisodeAlreadyDoneError, WokMixingEnv
from wok_sim.simulation import (
    InvalidTrajectoryError,
    ModelBuilder,
    ParticleGenerator,
    WokSimulator,
)
from wok_sim.trajectory import generate_trajectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def test_config() -> dict[str, Any]:
    return load_config(PROJECT_ROOT / "configs" / "test.yaml")


def test_actual_stl_is_visual_only_and_collision_is_compound() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    generator = ParticleGenerator(
        count=2,
        density_kg_m3=900.0,
        nominal_radius_m=0.008,
        radius_jitter_fraction=0.05,
        spawn_radius_m=0.04,
        spawn_height_m=0.01,
    )
    particles = generator.generate(0.008, seed=3)
    built = ModelBuilder(config).build(particles)

    visual_id = mujoco.mj_name2id(built.model, mujoco.mjtObj.mjOBJ_GEOM, "pan_visual")
    assert visual_id >= 0
    assert built.model.geom_contype[visual_id] == 0
    assert built.model.geom_conaffinity[visual_id] == 0
    assert len(built.pan_collision_geom_ids) == 1 + 2 * built.pan.proxy.wall_segments
    assert built.metadata["pan_collision_mode"] == "primitive_compound"
    assert built.metadata["pan_asset"]["triangle_count"] > 0
    np.testing.assert_allclose(
        built.metadata["pan_asset"]["extents_m"],
        [0.239946289, 0.239973145, 0.060],
        rtol=0.0,
        atol=2e-6,
    )


def test_gym_reset_and_one_step_rollout(test_config: dict[str, Any]) -> None:
    environment = WokMixingEnv(test_config)
    try:
        observation, reset_info = environment.reset(seed=11, options={"target_mass_kg": 0.020})
        assert isinstance(environment.observation_space, gym.spaces.Box)
        assert isinstance(environment.action_space, gym.spaces.Box)
        assert environment.action_space.shape == (7,)
        assert observation.shape == (1,)
        assert observation.dtype == np.float32
        assert environment.observation_space.contains(observation)
        assert reset_info["action_count"] == 0
        assert reset_info["actual_total_mass_kg"] == pytest.approx(0.020, abs=1e-10)

        next_observation, reward, terminated, truncated, info = environment.step(
            np.zeros(7, dtype=np.float32)
        )
        assert environment.observation_space.contains(next_observation)
        assert np.isfinite(reward)
        assert terminated is True
        assert truncated is False
        assert info["action_count"] == 1
        assert info["trajectory_valid"] is True
        assert info["invalid_trajectory"] is False
        assert info["simulation_metadata"]["trajectory_cycle_count"] == 5
        assert info["simulation_metadata"]["physics_elapsed_s"] == pytest.approx(
            info["simulation_metadata"]["trajectory_duration_s"]
            + info["simulation_metadata"]["post_rollout_settle_s"],
            abs=1.0e-12,
        )
        assert info["simulation_result"]["particle_positions_world_m"].ndim == 3
        assert np.isfinite(info["simulation_result"]["particle_positions_world_m"]).all()
        assert environment.simulator.rollout_count == 1
        assert not any(warning.number for warning in environment.simulator.data.warning)

        with pytest.raises(EpisodeAlreadyDoneError, match="reset"):
            environment.step(np.zeros(7, dtype=np.float32))
    finally:
        environment.close()


def test_reset_seed_reproduces_particle_context(test_config: dict[str, Any]) -> None:
    environment = WokMixingEnv(test_config)
    try:
        observation_a, _ = environment.reset(seed=29, options={"target_total_mass_kg": 0.021})
        radii_a = environment.particles.radii_m.copy()
        positions_a = environment.particles.positions_m.copy()
        observation_b, _ = environment.reset(seed=29, options={"target_total_mass_kg": 0.021})
        np.testing.assert_array_equal(observation_a, observation_b)
        np.testing.assert_array_equal(radii_a, environment.particles.radii_m)
        np.testing.assert_array_equal(positions_a, environment.particles.positions_m)
    finally:
        environment.close()


class _FakeSimulator:
    def __init__(self, _config: Any, particles: Any, **_kwargs: Any) -> None:
        self.positions = np.asarray(particles.positions_m).copy()
        self.rollout_count = 0
        self.closed = False

    def settle(self) -> dict[str, Any]:
        return {"settled": True, "elapsed_s": 0.0, "steps": 0}

    def particle_positions_pan(self) -> np.ndarray:
        return self.positions.copy()

    def rollout(self, _trajectory: Any) -> None:
        self.rollout_count += 1
        raise AssertionError("invalid trajectory에서 rollout이 호출되면 안 됩니다.")

    def render(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def test_invalid_trajectory_skips_physics(test_config: dict[str, Any]) -> None:
    invalid = SimpleNamespace(
        valid=False,
        validation=SimpleNamespace(valid=False, violations=("forced invalid",)),
        parameters={},
    )

    def trajectory_factory(*_args: Any, **_kwargs: Any) -> Any:
        return invalid

    environment = WokMixingEnv(
        test_config,
        simulator_factory=_FakeSimulator,
        trajectory_factory=trajectory_factory,
    )
    try:
        environment.reset(seed=4, options={"target_mass_kg": 0.020})
        _, reward, terminated, truncated, info = environment.step(np.zeros(7, dtype=np.float32))
        assert terminated is True
        assert truncated is False
        assert reward == -test_config["reward"]["w_invalid"]
        assert info["trajectory_valid"] is False
        assert info["simulation_skipped"] is True
        assert "forced invalid" in " ".join(info["invalid_reasons"])
        assert environment.simulator.rollout_count == 0
    finally:
        environment.close()


def _small_particle_batch(config: dict[str, Any], seed: int = 5) -> Any:
    particle = config["particles"]
    generator = ParticleGenerator(
        count=particle["count"],
        density_kg_m3=particle["density_kg_m3"],
        nominal_radius_m=particle["nominal_radius_m"],
        radius_jitter_fraction=particle["radius_jitter_fraction"],
        spawn_radius_m=particle["spawn_radius_m"],
        spawn_height_m=particle["spawn_height_m"],
        mass_tolerance_kg=particle["mass_tolerance_kg"],
        minimum_clearance_m=particle["spawn_clearance_m"],
        max_spawn_attempts=particle["max_spawn_attempts"],
    )
    return generator.generate(0.020, seed=seed)


def test_start_pose_mismatch_is_rejected_without_pan_teleport(
    test_config: dict[str, Any],
) -> None:
    simulator = WokSimulator(test_config, _small_particle_batch(test_config))
    try:
        simulator.settle()
        mocap_before = simulator.data.mocap_pos[simulator.built.pan_mocap_id].copy()
        start = np.asarray(test_config["pan"]["initial_pose"]["position_m"], dtype=float)
        mismatched = {
            "time_s": np.array([0.0, 0.1]),
            "position_m": np.vstack((start + [0.01, 0.0, 0.0], start)),
            "orientation_rpy_rad": np.zeros((2, 3)),
            "validation": {"valid": True},
        }
        with pytest.raises(InvalidTrajectoryError, match="start_pose_mismatch"):
            simulator.rollout(mismatched)
        np.testing.assert_array_equal(
            simulator.data.mocap_pos[simulator.built.pan_mocap_id],
            mocap_before,
        )
        assert simulator.rollout_count == 0
    finally:
        simulator.close()


def test_no_contact_duration_is_independent_of_record_stride(
    test_config: dict[str, Any],
) -> None:
    trajectory = generate_trajectory(np.zeros(7), test_config)
    durations: list[np.ndarray] = []
    history_lengths: list[int] = []
    for stride in (1, 9):
        config = deepcopy(test_config)
        config["simulation"]["record_every_n_steps"] = stride
        simulator = WokSimulator(config, _small_particle_batch(config, seed=17))
        try:
            simulator.settle()
            result = simulator.rollout(trajectory)
            durations.append(result.final_no_contact_duration_s)
            history_lengths.append(len(result.time_s))
        finally:
            simulator.close()
    assert history_lengths[1] < history_lengths[0]
    np.testing.assert_allclose(durations[0], durations[1], rtol=0.0, atol=1e-12)
