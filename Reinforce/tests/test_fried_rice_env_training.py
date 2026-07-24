"""볶음밥 profile의 환경 dispatch와 SAC random-walk 연결 테스트."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wok_sim.config import load_config
from wok_sim.envs import WokMixingEnv
from wok_sim.training import PersistentEpisodeRandomWalkNoise
from wok_sim.trajectory import generate_configured_trajectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def test_production_fried_rice_center_motion_is_speed_driven_and_caps_are_synchronized() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "fried_rice.yaml")
    trajectory = generate_configured_trajectory(np.zeros(4), config)

    assert trajectory.parameters.insertion_distance == pytest.approx(0.25)
    assert np.rad2deg(trajectory.parameters.tilt_angle) == pytest.approx(10.0)
    assert trajectory.parameters.linear_speed == pytest.approx(0.25)
    assert trajectory.parameters.angular_speed == pytest.approx(0.225)
    assert trajectory.parameters.cycle_time == pytest.approx(5.2044410433)
    assert trajectory.duration_s == pytest.approx(26.0222052166)
    pitch_deg = np.rad2deg(trajectory.orientation_wok_rpy_rad[:, 1])
    assert np.min(pitch_deg) == pytest.approx(-10.0)
    assert np.max(pitch_deg) == pytest.approx(0.0)
    assert trajectory.validation.metrics["max_cartesian_velocity"] <= 0.25
    assert trajectory.validation.metrics["max_angular_velocity"] <= 0.225

    trajectory_caps = config["trajectory"]
    robot_caps = config["robot"]["cartesian_caps"]
    assert robot_caps == {
        "linear_velocity_m_s": trajectory_caps["max_cartesian_velocity"],
        "angular_velocity_rad_s": trajectory_caps["max_angular_velocity"],
        "linear_acceleration_m_s2": trajectory_caps["max_cartesian_acceleration"],
        "angular_acceleration_rad_s2": trajectory_caps["max_angular_acceleration"],
        "linear_jerk_m_s3": trajectory_caps["max_cartesian_jerk"],
        "angular_jerk_rad_s3": trajectory_caps["max_angular_jerk"],
    }


def test_sac_random_walk_noise_is_reproducible_bounded_and_reset_persistent() -> None:
    first = PersistentEpisodeRandomWalkNoise(
        4,
        step_std=[0.04, 0.04, 0.05, 0.05],
        bound=0.25,
        seed=34,
    )
    second = PersistentEpisodeRandomWalkNoise(
        4,
        step_std=[0.04, 0.04, 0.05, 0.05],
        bound=0.25,
        seed=34,
    )
    for _ in range(25):
        actual = first()
        expected = second()
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == np.float32
        assert np.all(np.abs(actual) <= 0.25)
        before_reset = first.current_noise
        first.reset()
        np.testing.assert_array_equal(first.current_noise, before_reset)
