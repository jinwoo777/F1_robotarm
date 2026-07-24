"""5회 global quintic spline의 연속성과 공통 time axis 테스트."""

from __future__ import annotations

import numpy as np

from wok_sim.trajectory import generate_trajectory


def _config() -> dict[str, object]:
    return {
        "trajectory": {
            "cycles": 5,
            "insertion_angle_deg": 45.0,
            "insertion_distance_range_m": [0.03, 0.05],
            "lift_height_range_m": [0.05, 0.07],
            "backward_distance_range_m": [0.04, 0.06],
            "pitch_amplitude_range_rad": [0.15, 0.25],
            "cycle_time_range_s": [0.9, 1.1],
            "insert_phase_ratio_range": [0.18, 0.22],
            "catch_phase_ratio_range": [0.18, 0.22],
            "transition_forward_offset_m": 0.006,
            "transition_lift_offset_m": 0.008,
            "catch_offset_m": [-0.01, 0.0, 0.015],
            "sample_rate_hz": 120.0,
            "max_cartesian_velocity": 100.0,
            "max_cartesian_acceleration": 1000.0,
            "max_cartesian_jerk": 100000.0,
            "workspace": {
                "x_m": [-1.0, 1.0],
                "y_m": [-1.0, 1.0],
                "z_m": [-1.0, 1.0],
                "pitch_rad": [-2.0, 2.0],
            },
            "y_tolerance": 1.0e-10,
            "roll_tolerance": 1.0e-10,
            "yaw_tolerance": 1.0e-10,
            "start_position_m": [0.1, -0.2, 0.3],
            "start_rpy_rad": [0.4, -0.1, 0.2],
        }
    }


def test_global_spline_hits_every_waypoint_and_only_rests_at_ends() -> None:
    trajectory = generate_trajectory(np.zeros(7), _config())
    knots = trajectory.waypoints.times

    np.testing.assert_allclose(
        trajectory.evaluate(knots, derivative=0),
        trajectory.waypoints.poses,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        trajectory.evaluate([knots[0], knots[-1]], derivative=1),
        0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        trajectory.evaluate([knots[0], knots[-1]], derivative=2),
        0.0,
        atol=1.0e-11,
    )

    cycle_boundaries = trajectory.waypoints.cycle_boundaries_s[1:-1]
    boundary_speed = np.linalg.norm(trajectory.evaluate(cycle_boundaries, derivative=1), axis=1)
    assert np.all(boundary_speed > 1.0e-4)
    assert trajectory.validation is not None
    assert "cycle_boundary_dwell" not in trajectory.validation.violations


def test_position_velocity_acceleration_are_continuous_at_internal_knots() -> None:
    trajectory = generate_trajectory(np.zeros(7), _config())
    internal = trajectory.waypoints.times[1:-1]
    epsilon = 1.0e-7

    for derivative, tolerance in ((0, 1.0e-5), (1, 1.0e-4), (2, 2.0e-2)):
        left = trajectory.evaluate(internal - epsilon, derivative=derivative)
        right = trajectory.evaluate(internal + epsilon, derivative=derivative)
        assert float(np.max(np.abs(left - right))) < tolerance


def test_x_z_pitch_and_all_derivatives_share_one_time_axis() -> None:
    trajectory = generate_trajectory(np.zeros(7), _config())
    count = len(trajectory.time_s)

    assert trajectory.pose.shape == (count, 6)
    assert trajectory.velocity.shape == (count, 6)
    assert trajectory.acceleration.shape == (count, 6)
    assert trajectory.jerk.shape == (count, 6)
    assert np.all(np.diff(trajectory.time_s) > 0.0)
    assert np.isfinite(trajectory.pose).all()
    assert np.isfinite(trajectory.velocity).all()
    assert np.isfinite(trajectory.acceleration).all()
    assert np.isfinite(trajectory.jerk).all()


def test_y_roll_yaw_remain_constant_and_twist_follows_pitch_axis() -> None:
    trajectory = generate_trajectory(np.zeros(7), _config())

    np.testing.assert_allclose(trajectory.position_m[:, 1], -0.2, atol=2.0e-14)
    np.testing.assert_allclose(trajectory.orientation_rpy_rad[:, 0], 0.4, atol=2.0e-14)
    np.testing.assert_allclose(trajectory.orientation_rpy_rad[:, 2], 0.2, atol=2.0e-14)
    np.testing.assert_allclose(trajectory.linear_velocity_m_s[:, 1], 0.0, atol=2.0e-14)

    rpy_rate = trajectory.spline.evaluate(trajectory.time_s, derivative=1)[:, 3:]
    np.testing.assert_allclose(rpy_rate[:, [0, 2]], 0.0, atol=2.0e-14)
    pitch_rate = rpy_rate[:, 1]
    expected_angular_velocity = pitch_rate[:, None] * np.array([-np.sin(0.2), np.cos(0.2), 0.0])
    np.testing.assert_allclose(
        trajectory.angular_velocity_rad_s,
        expected_angular_velocity,
        atol=2.0e-14,
    )
