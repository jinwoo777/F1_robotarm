"""overshoot 및 Cartesian limits validation 테스트."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from wok_sim.trajectory import generate_trajectory, validate_trajectory


def _config() -> dict[str, object]:
    return {
        "trajectory": {
            "cycles": 5,
            "insertion_angle_deg": 45.0,
            "insertion_distance_range_m": [0.02, 0.04],
            "lift_height_range_m": [0.04, 0.06],
            "backward_distance_range_m": [0.03, 0.05],
            "pitch_amplitude_range_rad": [0.1, 0.2],
            "cycle_time_range_s": [0.9, 1.1],
            "insert_phase_ratio_range": [0.18, 0.22],
            "catch_phase_ratio_range": [0.18, 0.22],
            "transition_forward_offset_m": 0.005,
            "transition_lift_offset_m": 0.006,
            "catch_offset_m": [-0.008, 0.0, 0.012],
            "sample_rate_hz": 150.0,
            "max_cartesian_velocity": 10.0,
            "max_cartesian_acceleration": 100.0,
            "max_cartesian_jerk": 10000.0,
            "workspace": {
                "x_m": [-0.5, 0.5],
                "y_m": [-0.1, 0.1],
                "z_m": [-0.5, 0.5],
                "pitch_rad": [-1.0, 1.0],
            },
            "y_tolerance": 1.0e-9,
            "roll_tolerance": 1.0e-9,
            "yaw_tolerance": 1.0e-9,
            "start_position_m": [0.0, 0.0, 0.2],
            "start_rpy_rad": [0.0, 0.0, 0.0],
        }
    }


def test_valid_trajectory_reports_sampled_extrema_and_endpoint_rest() -> None:
    config = _config()
    trajectory = generate_trajectory(np.zeros(7), config)
    result = trajectory.validation

    assert result is not None
    assert result.valid
    assert not result.violations
    assert result.metrics["max_cartesian_velocity"] > 0.0
    assert result.metrics["max_cartesian_acceleration"] > 0.0
    assert result.metrics["max_cartesian_jerk"] > 0.0
    assert result.metrics["max_endpoint_velocity"] < 1.0e-10
    assert result.metrics["max_endpoint_acceleration"] < 1.0e-9
    assert result.metrics["min_cycle_boundary_velocity"] > 0.0


def test_velocity_acceleration_and_jerk_limits_are_hard_invalidations() -> None:
    base = _config()
    trajectory = generate_trajectory(np.zeros(7), base, validate=False)
    limited = deepcopy(base)
    limited["trajectory"]["max_cartesian_velocity"] = 1.0e-6
    limited["trajectory"]["max_cartesian_acceleration"] = 1.0e-6
    limited["trajectory"]["max_cartesian_jerk"] = 1.0e-6

    result = validate_trajectory(trajectory, limited)

    assert not result.valid
    assert "max_cartesian_velocity" in result.violations
    assert "max_cartesian_acceleration" in result.violations
    assert "max_cartesian_jerk" in result.violations


def test_workspace_overshoot_is_invalid() -> None:
    base = _config()
    trajectory = generate_trajectory(np.zeros(7), base, validate=False)
    limited = deepcopy(base)
    limited["trajectory"]["workspace"]["x_m"] = [0.0, 0.001]
    limited["trajectory"]["workspace"]["z_m"] = [0.199, 0.201]
    limited["trajectory"]["workspace"]["pitch_rad"] = [0.0, 0.01]

    result = validate_trajectory(trajectory, limited)

    assert not result.valid
    assert "workspace_x" in result.violations
    assert "workspace_z" in result.violations
    assert "workspace_pitch" in result.violations


def test_optional_angular_limits_validate_pitch_derivatives() -> None:
    base = _config()
    trajectory = generate_trajectory(np.zeros(7), base, validate=False)
    limited = deepcopy(base)
    limited["trajectory"]["max_angular_velocity"] = 1.0e-6
    limited["trajectory"]["max_angular_acceleration"] = 1.0e-6
    limited["trajectory"]["max_angular_jerk"] = 1.0e-6

    result = validate_trajectory(trajectory, limited)

    assert "max_angular_velocity" in result.violations
    assert "max_angular_acceleration" in result.violations
    assert "max_angular_jerk" in result.violations
