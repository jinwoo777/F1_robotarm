"""볶음밥 profile의 환경 dispatch와 SAC random-walk 연결 테스트."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wok_sim.config import load_config
from wok_sim.envs import WokMixingEnv
from wok_sim.envs.wok_mixing_env import _lift_approach_curriculum
from wok_sim.training import PersistentEpisodeRandomWalkNoise
from wok_sim.trajectory import generate_configured_trajectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_lift_approach_curriculum_uses_60_120_270_episode_split() -> None:
    training = load_config(PROJECT_ROOT / "configs" / "fried_rice.yaml")["training"]

    assert _lift_approach_curriculum(training, 0) == ("approach_100", 1.0)
    assert _lift_approach_curriculum(training, 59) == ("approach_100", 1.0)
    assert _lift_approach_curriculum(training, 60) == ("approach_anneal", 0.5)
    stage, midpoint = _lift_approach_curriculum(training, 120)
    assert stage == "approach_anneal"
    assert 0.24 < midpoint < 0.26
    assert _lift_approach_curriculum(training, 179) == ("approach_anneal", 0.0)
    assert _lift_approach_curriculum(training, 180) == ("final_objective", 0.0)
    assert _lift_approach_curriculum(training, 449) == ("final_objective", 0.0)
    # 일반 preview/evaluation은 curriculum shaping 없이 실제 목적함수만 쓴다.
    assert _lift_approach_curriculum(training, None) == ("final_objective", 0.0)


class _ResetOnlySimulator:
    def __init__(self, _config: Any, particles: Any, **_kwargs: Any) -> None:
        self._positions = np.asarray(particles.positions_m, dtype=float).copy()

    def settle(self) -> dict[str, Any]:
        return {"settled": True, "elapsed_s": 0.0, "steps": 0}

    def particle_positions_pan(self) -> np.ndarray:
        return self._positions.copy()

    def close(self) -> None:
        return None


def test_fried_rice_env_uses_weight_conditioned_context_and_variable_equal_mix() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "fried_rice.yaml")
    environment = WokMixingEnv(config, simulator_factory=_ResetOnlySimulator)
    try:
        observation, info = environment.reset(seed=19)
        assert environment.action_space.shape == (4,)
        assert observation.shape == (2,)
        mass_low, mass_high = config["particles"]["target_mass_range_kg"]
        expected_context = (
            2.0 * (info["actual_total_mass_kg"] - mass_low) / (mass_high - mass_low) - 1.0
        )
        assert observation[0] == pytest.approx(expected_context)
        assert environment.observation_space.contains(observation)
        np.testing.assert_array_equal(environment.observation_space.low, [-1.0, -1.0])
        np.testing.assert_array_equal(environment.observation_space.high, [1.0, 1.0])
        assert 0.056 <= info["actual_total_mass_kg"] <= 0.128
        assert info["target_total_mass_kg"] == info["actual_total_mass_kg"]
        assert 60 <= info["particle_count"] <= 120
        species_counts = info["particle_species_counts"]
        assert set(species_counts) == {"ellipsoid", "large_sphere", "small_sphere"}
        assert len(set(species_counts.values())) == 1
        count_per_type = next(iter(species_counts.values()))
        assert 20 <= count_per_type <= 40
        assert info["count_per_type"] == count_per_type
        assert observation[1] == pytest.approx(2.0 * (count_per_type - 20) / 20.0 - 1.0)
        assert info["particle_amount_fraction"] == pytest.approx(info["particle_count"] / 60)
        ellipsoid = environment.particles.species == "ellipsoid"
        np.testing.assert_array_equal(environment.particles.geom_types[ellipsoid], "ellipsoid")

        repeated_observation, repeated_info = environment.reset(seed=19)
        np.testing.assert_array_equal(repeated_observation, observation)
        assert repeated_info["particle_species_counts"] == species_counts
        assert repeated_info["actual_total_mass_kg"] == info["actual_total_mass_kg"]
    finally:
        environment.close()


def test_m0609_speed_target_is_observed_logged_and_applied_to_episode_config() -> None:
    config = load_config(
        PROJECT_ROOT / "configs" / "fried_rice_m0609_speed_sweep_150.yaml"
    )
    environment = WokMixingEnv(config, simulator_factory=_ResetOnlySimulator)
    try:
        observation, info = environment.reset(
            seed=29,
            options={
                "count_per_type": 30,
                "nominal_joint_speed_target_fraction": 0.84,
            },
        )
        assert observation.shape == (3,)
        assert environment.observation_space.contains(observation)
        assert observation[1] == pytest.approx(0.0)
        assert observation[2] == pytest.approx(-0.2)
        assert info["nominal_joint_speed_target_fraction"] == pytest.approx(0.84)
        episode_config = environment._trajectory_config_for_episode()
        target = episode_config["robot"]["nominal_joint_speed_retiming"]["target_fraction"]
        assert target == pytest.approx(0.84)
        assert config["robot"]["nominal_joint_speed_retiming"]["target_fraction"] == pytest.approx(
            0.90
        )
    finally:
        environment.close()


@pytest.mark.parametrize("count_per_type", (20, 25, 30, 35, 40))
def test_fried_rice_env_allows_reproducible_weight_strata(
    count_per_type: int,
) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "fried_rice.yaml")
    environment = WokMixingEnv(config, simulator_factory=_ResetOnlySimulator)
    try:
        observation, info = environment.reset(
            seed=23,
            options={"count_per_type": count_per_type},
        )
        assert environment.observation_space.contains(observation)
        assert info["particle_count"] == 3 * count_per_type
        assert set(info["particle_species_counts"].values()) == {count_per_type}
        assert info["count_per_type"] == count_per_type
        assert observation[1] == pytest.approx(2.0 * (count_per_type - 20) / 20.0 - 1.0)
        assert info["particle_amount_fraction"] == pytest.approx(count_per_type / 20)
    finally:
        environment.close()


def test_fried_rice_env_rejects_target_mass_rescaling() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "fried_rice.yaml")
    with pytest.raises(ValueError, match="재스케일"):
        WokMixingEnv(
            config,
            target_mass_kg=0.060,
            simulator_factory=_ResetOnlySimulator,
        )


@pytest.mark.parametrize(
    "action", tuple(np.asarray(item) for item in product((-1.0, 1.0), repeat=4))
)
def test_production_fried_rice_motion_stays_inside_provisional_caps(
    action: np.ndarray,
) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "fried_rice.yaml")
    trajectory = generate_configured_trajectory(action, config)

    assert trajectory.validation is not None
    assert trajectory.validation.valid
    assert trajectory.validation.violations == ()


def test_production_fried_rice_center_motion_maps_four_actions_and_four_phases() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "fried_rice.yaml")
    trajectory = generate_configured_trajectory(np.zeros(4), config)

    assert np.rad2deg(trajectory.parameters.descent_angle) == pytest.approx(45.0)
    assert np.rad2deg(trajectory.parameters.pan_tilt_angle) == pytest.approx(30.0)
    assert trajectory.parameters.linear_speed == pytest.approx(0.43)
    assert trajectory.parameters.insertion_distance == pytest.approx(0.25)
    assert np.rad2deg(trajectory.parameters.tilt_recovery_angle) == pytest.approx(15.0)
    assert trajectory.parameters.angular_speed == pytest.approx(1.20)
    assert trajectory.parameters.time_scale == pytest.approx(1.50)
    assert trajectory.parameters.lift_return_time_factor == pytest.approx(0.875)
    assert trajectory.parameters.phase_durations_s == pytest.approx(
        (2.4290644853, 2.2779205742, 1.6869560397, 1.9931805024)
    )
    assert trajectory.parameters.cycle_time == pytest.approx(8.3871216016)
    assert trajectory.duration_s == pytest.approx(41.9356080078)
    pitch_deg = np.rad2deg(trajectory.orientation_wok_rpy_rad[:, 1])
    assert np.min(pitch_deg) == pytest.approx(0.0)
    assert np.max(pitch_deg) == pytest.approx(30.0)

    robot_caps = config["robot"]["cartesian_caps"]
    metrics = trajectory.validation.metrics
    assert metrics["max_cartesian_velocity"] <= robot_caps["linear_velocity_m_s"]
    assert metrics["max_angular_velocity"] <= robot_caps["angular_velocity_rad_s"]
    assert metrics["max_cartesian_acceleration"] <= robot_caps["linear_acceleration_m_s2"]
    assert metrics["max_angular_acceleration"] <= robot_caps["angular_acceleration_rad_s2"]
    assert metrics["max_cartesian_jerk"] <= robot_caps["linear_jerk_m_s3"]
    assert metrics["max_angular_jerk"] <= robot_caps["angular_jerk_rad_s3"]


def test_sac_random_walk_noise_is_reproducible_bounded_and_reset_persistent() -> None:
    first = PersistentEpisodeRandomWalkNoise(
        4,
        step_std=[0.05, 0.05, 0.03, 0.05],
        bound=[0.25, 0.25, 0.20, 0.25],
        seed=34,
    )
    second = PersistentEpisodeRandomWalkNoise(
        4,
        step_std=[0.05, 0.05, 0.03, 0.05],
        bound=[0.25, 0.25, 0.20, 0.25],
        seed=34,
    )
    for _ in range(25):
        actual = first()
        expected = second()
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == np.float32
        assert np.all(np.abs(actual) <= np.asarray([0.25, 0.25, 0.20, 0.25]))
        before_reset = first.current_noise
        first.reset()
        np.testing.assert_array_equal(first.current_noise, before_reset)
