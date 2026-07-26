"""6D pitch-release action, reward curriculum and wall-clock gate tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.search_pitch_release import build_candidates, select_candidates
from wok_sim.config import load_config
from wok_sim.envs.wok_mixing_env import _toss_curriculum
from wok_sim.metrics.reward import compute_reward_terms
from wok_sim.training.train_sac import _WallClockStopCallback
from wok_sim.trajectory import (
    PITCH_RELEASE_ACTION_NAMES,
    CoupledPitchLiftImpulseSpline,
    action_names_for_config,
    generate_fried_rice_trajectory,
    map_fried_rice_action,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "fried_rice_pitch_release_6d_8h.yaml"


def _config() -> dict[str, object]:
    return load_config(CONFIG_PATH)


def test_pitch_release_layout_maps_six_independent_dimensions() -> None:
    config = _config()
    low = map_fried_rice_action(np.full(6, -1.0), config)
    middle = map_fried_rice_action(np.zeros(6), config)
    high = map_fried_rice_action(np.full(6, 1.0), config)

    assert action_names_for_config(config) == PITCH_RELEASE_ACTION_NAMES
    assert middle.tilt_recovery_angle == pytest.approx(0.0)
    assert low.pitch_release_angle == pytest.approx(0.0)
    assert high.pitch_release_angle == pytest.approx(np.deg2rad(12.0))
    assert middle.pitch_release_angle == pytest.approx(np.deg2rad(6.0))
    assert low.pitch_release_phase_fraction == pytest.approx(0.35)
    assert high.pitch_release_phase_fraction == pytest.approx(0.65)
    assert low.pitch_release_angular_acceleration == pytest.approx(0.45)
    assert high.pitch_release_angular_acceleration == pytest.approx(0.95)


def test_pitch_release_pulse_returns_to_held_pitch_with_continuous_derivatives() -> None:
    config = _config()
    action = np.asarray([-0.7, 1.0, -1.0 / 3.0, -0.5, 0.0, 1.0])
    trajectory = generate_fried_rice_trajectory(action, config)
    parameters = trajectory.parameters
    recovery_start = sum(parameters.phase_durations_s[:2])
    recovery_duration = sum(parameters.phase_durations_s[2:4])
    peak = recovery_start + parameters.pitch_release_phase_fraction * recovery_duration
    half = parameters.pitch_release_half_duration_s
    pulse_times = np.asarray([peak - half, peak, peak + half])

    pose = trajectory.spline.evaluate(pulse_times)
    velocity = trajectory.spline.evaluate(pulse_times, derivative=1)
    acceleration = trajectory.spline.evaluate(pulse_times, derivative=2)
    held_pitch = parameters.pan_tilt_angle
    np.testing.assert_allclose(
        pose[:, 4],
        [
            held_pitch,
            held_pitch - parameters.pitch_release_angle,
            held_pitch,
        ],
        atol=1.0e-9,
    )
    np.testing.assert_allclose(velocity[:, 4], 0.0, atol=1.0e-9)
    np.testing.assert_allclose(acceleration[:, 4], 0.0, atol=1.0e-8)

    dense = np.linspace(peak - half, peak + half, 2001)
    peak_acceleration = float(
        np.max(np.abs(trajectory.spline.evaluate(dense, derivative=2)[:, 4]))
    )
    assert peak_acceleration == pytest.approx(
        parameters.pitch_release_effective_angular_acceleration,
        rel=2.0e-5,
    )
    assert trajectory.validation is not None
    assert trajectory.validation.valid


def test_coupled_pitch_lift_pulse_shares_peak_and_continuous_boundaries() -> None:
    config = load_config(
        CONFIG_PATH,
        overrides={
            "trajectory": {
                "fried_rice": {
                    "pitch_release_lift_impulse_height_m": 0.003,
                }
            }
        },
    )
    action = np.asarray([-0.7, 1.0, -1.0 / 3.0, 0.0, -1.0, 1.0])
    trajectory = generate_fried_rice_trajectory(action, config)
    assert isinstance(trajectory.spline, CoupledPitchLiftImpulseSpline)

    parameters = trajectory.parameters
    recovery_start = sum(parameters.phase_durations_s[:2])
    recovery_duration = sum(parameters.phase_durations_s[2:4])
    peak = recovery_start + parameters.pitch_release_phase_fraction * recovery_duration
    half = parameters.pitch_release_half_duration_s
    pulse_times = np.asarray([peak - half, peak, peak + half])
    coupled = trajectory.spline.evaluate(pulse_times)
    pitch_only = trajectory.spline._pitch_spline.evaluate(pulse_times)
    coupled_velocity = trajectory.spline.evaluate(pulse_times, derivative=1)
    pitch_velocity = trajectory.spline._pitch_spline.evaluate(
        pulse_times,
        derivative=1,
    )
    coupled_acceleration = trajectory.spline.evaluate(pulse_times, derivative=2)
    pitch_acceleration = trajectory.spline._pitch_spline.evaluate(
        pulse_times,
        derivative=2,
    )

    np.testing.assert_allclose(
        coupled[:, 2] - pitch_only[:, 2],
        [0.0, 0.003, 0.0],
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        coupled_velocity[:, 2] - pitch_velocity[:, 2],
        0.0,
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        coupled_acceleration[:, 2] - pitch_acceleration[:, 2],
        0.0,
        atol=1.0e-8,
    )


def test_pitch_release_rejects_pulse_that_cannot_fit_recovery_return() -> None:
    config = _config()
    # 12 deg, phase 0.35, 0.45 rad/s² is intentionally too wide for P2→P4.
    action = np.asarray([-0.7, 1.0, -1.0 / 3.0, 1.0, -1.0, -1.0])
    with pytest.raises(ValueError, match="recovery/return"):
        generate_fried_rice_trajectory(action, config)


def test_peak_toss_reward_requires_peak_and_spill_contract() -> None:
    config = {
        "w_mix": 1.0,
        "w_peak_toss": 20.0,
        "peak_toss_goal_ratio": 0.20,
        "toss_success_spill_ratio": 0.05,
        "toss_success_bonus": 5.0,
        "w_spill": 12.0,
        "w_spill_quadratic": 40.0,
        "spill_severity_mode": "count",
    }
    success, signals = compute_reward_terms(
        mixing_improvement=0.5,
        lifted_particle_count=99,
        peak_lifted_particle_ratio=0.20,
        spill_mass_ratio=0.0,
        spill_count_ratio=0.05,
        jerk_cost=0.0,
        acceleration_cost=0.0,
        height_penalty=0.0,
        reward_config=config,
    )
    high_spill, high_spill_signals = compute_reward_terms(
        mixing_improvement=0.5,
        lifted_particle_count=99,
        peak_lifted_particle_ratio=0.20,
        spill_mass_ratio=0.0,
        spill_count_ratio=0.051,
        jerk_cost=0.0,
        acceleration_cost=0.0,
        height_penalty=0.0,
        reward_config=config,
    )

    assert success["peak_toss"] == pytest.approx(4.0)
    assert success["toss_success"] == pytest.approx(5.0)
    assert success["lift"] == 0.0
    assert signals["toss_success"] == pytest.approx(1.0)
    assert high_spill["toss_success"] == 0.0
    assert high_spill_signals["toss_success"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("episode", "stage", "goal", "spill", "bonus"),
    [
        (119, "toss_warmup", 0.10, 0.10, 2.0),
        (120, "toss_anneal", 0.10, 0.10, 2.0),
        (419, "toss_anneal", 0.20, 0.05, 5.0),
        (420, "final_objective", 0.20, 0.05, 5.0),
    ],
)
def test_toss_curriculum_boundaries(
    episode: int,
    stage: str,
    goal: float,
    spill: float,
    bonus: float,
) -> None:
    config = _config()
    actual = _toss_curriculum(
        config["training"],
        config["reward"],
        episode,
    )
    assert actual[0] == stage
    assert actual[1:] == pytest.approx((goal, spill, bonus))


def test_wall_clock_callback_stops_at_first_completed_step_after_limit() -> None:
    times = iter((100.0, 102.0, 106.1, 107.0))
    callback = _WallClockStopCallback(6.0, clock=lambda: next(times))
    callback._on_training_start()

    assert callback._on_step()
    assert not callback._on_step()
    assert callback.limit_reached
    callback._on_training_end()
    assert callback.elapsed_time_s == pytest.approx(7.0)


def test_pitch_release_screen_builds_72_candidates_and_balances_selection() -> None:
    summary = {
        "best_any": {
            "descent_angle_rad": 0.66,
            "pan_tilt_angle_rad": 0.69,
            "descent_speed_m_s": 0.41,
        },
        "best_safe": {
            "descent_angle_rad": 0.73,
            "pan_tilt_angle_rad": 0.66,
            "descent_speed_m_s": 0.48,
        },
    }
    candidates = build_candidates(_config(), summary)
    assert len(candidates) == 72
    rows = [
        {
            "candidate_id": candidate.candidate_id,
            "base_source": candidate.base_source,
            "trajectory_valid": True,
            "simulation_budget_pass": True,
            "peak_release_angular_velocity_rad_s": candidate.physical_action[3],
            "effective_angular_acceleration_rad_s2": candidate.physical_action[5],
            "duration_s": 50.0,
        }
        for candidate in candidates
    ]
    selected = select_candidates(candidates, rows)
    assert len(selected) == 12
    assert sum(item.base_source == "best_any" for item in selected) == 6
    assert sum(item.base_source == "best_safe" for item in selected) == 6
