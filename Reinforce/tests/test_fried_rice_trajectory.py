"""볶음밥 3단 teaching과 episode random walk 테스트."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from wok_sim.exploration import (
    BoundedEpisodeRandomWalk,
    RandomWalkConfigurationError,
)
from wok_sim.trajectory import (
    FRIED_RICE_ACTION_NAMES,
    FriedRiceParameters,
    FriedRiceTrajectoryError,
    build_fried_rice_cycle_waypoints,
    build_repeated_fried_rice_waypoints,
    generate_fried_rice_trajectory,
    map_fried_rice_action,
)


def _config() -> dict[str, object]:
    return {
        "pan": {
            "initial_pose": {
                "position_m": [0.0, 0.0, 0.20],
                "rpy_rad": [0.0, 0.0, 0.0],
            }
        },
        "trajectory": {
            "cycles": 5,
            "sample_rate_hz": 120.0,
            "max_cartesian_velocity": 0.50,
            "max_angular_velocity": 1.50,
            "max_cartesian_acceleration": 3.0,
            "max_angular_acceleration": 6.0,
            "max_cartesian_jerk": 30.0,
            "max_angular_jerk": 50.0,
            "allow_cycle_boundary_stop": True,
            "workspace": {
                "x_m": [-0.02, 0.24],
                "y_m": [-0.001, 0.001],
                "z_m": [-0.04, 0.22],
                "pitch_rad": [-0.30, 0.10],
            },
            "y_tolerance": 1.0e-10,
            "roll_tolerance": 1.0e-10,
            "yaw_tolerance": 1.0e-10,
            "start_position_m": [0.0, 0.0, 0.20],
            "start_rpy_rad": [0.0, 0.0, 0.0],
            "fried_rice": {
                "cycles": 5,
                "insertion_angle_deg": 45.0,
                "tilt_direction": -1,
                "insertion_distance_range_m": [0.20, 0.30],
                "tilt_angle_range_rad": [
                    float(np.deg2rad(5.0)),
                    float(np.deg2rad(15.0)),
                ],
                "linear_speed_range_m_s": [0.20, 0.30],
                "angular_speed_range_rad_s": [0.15, 0.30],
                "linear_acceleration_limit_m_s2": 0.80,
                "angular_acceleration_limit_rad_s2": 1.10,
                "linear_jerk_limit_m_s3": 8.0,
                "angular_jerk_limit_rad_s3": 10.0,
                "minimum_phase_duration_s": 0.50,
            },
        },
        "simulation": {},
        "robot": {},
    }


def test_fried_rice_action_has_four_independent_episode_parameters() -> None:
    config = _config()
    low = map_fried_rice_action(np.full(4, -1.0), config)
    middle = map_fried_rice_action(np.zeros(4), config)
    high = map_fried_rice_action(np.full(4, 1.0), config)

    assert FRIED_RICE_ACTION_NAMES == (
        "insertion_distance",
        "tilt_angle",
        "linear_speed",
        "angular_speed",
    )
    np.testing.assert_allclose(
        middle.as_array(),
        [0.25, np.deg2rad(10.0), 0.25, 0.225],
    )
    np.testing.assert_allclose(
        middle.as_array(),
        0.5 * (low.as_array() + high.as_array()),
    )


def test_fried_rice_action_validation_is_separate_from_legacy_seven_dimensions() -> None:
    with pytest.raises(FriedRiceTrajectoryError, match="shape"):
        map_fried_rice_action(np.zeros(7), _config())
    with pytest.raises(FriedRiceTrajectoryError, match=r"\[-1, 1\]"):
        map_fried_rice_action(np.full(4, 1.01), _config(), clip=False)
    with pytest.raises(FriedRiceTrajectoryError, match="NaN"):
        map_fried_rice_action([0.0, 0.0, np.nan, 0.0], _config())


def test_three_teaching_stages_match_requested_geometry() -> None:
    parameters = FriedRiceParameters.teaching_default()
    points = build_fried_rice_cycle_waypoints(parameters, _config())
    p0, p1, p2, p3 = points

    assert [point.name for point in points] == ["P0", "P1", "P2", "P3"]
    assert p1.x - p0.x == pytest.approx(p0.z - p1.z, abs=1.0e-12)
    assert np.linalg.norm(p1.position_m - p0.position_m) == pytest.approx(0.25)
    np.testing.assert_allclose(p2.position_m, p1.position_m, atol=0.0)
    assert p2.pitch - p0.pitch == pytest.approx(-np.deg2rad(10.0))
    np.testing.assert_allclose(p3.pose.as_array(), p0.pose.as_array(), atol=0.0)
    insertion_duration, tilt_duration, _ = parameters.phase_durations_s
    assert p1.time_s == pytest.approx(insertion_duration)
    assert p2.time_s == pytest.approx(insertion_duration + tilt_duration)
    assert p3.time_s == pytest.approx(parameters.cycle_time)


def test_five_cycles_are_one_phasewise_waypoint_sequence_without_duplicate_knots() -> None:
    parameters = FriedRiceParameters.teaching_default()
    sequence = build_repeated_fried_rice_waypoints(parameters, _config())

    assert sequence.cycle_count == 5
    assert len(sequence) == 1 + 3 * 5
    assert np.all(np.diff(sequence.times) > 0.0)
    np.testing.assert_allclose(
        sequence.cycle_boundaries_s,
        np.arange(6) * parameters.cycle_time,
    )
    assert sum(point.name == "P1" for point in sequence) == 5
    assert sum(point.name == "P2" for point in sequence) == 5
    assert sum(point.name == "P3" for point in sequence) == 5
    assert not any(point.name.startswith("S") for point in sequence)


def test_phasewise_minimum_jerk_has_no_geometry_or_pitch_overshoot() -> None:
    trajectory = generate_fried_rice_trajectory(np.zeros(4), _config())
    pitch_deg = np.rad2deg(trajectory.orientation_wok_rpy_rad[:, 1])

    assert np.min(pitch_deg) == pytest.approx(-10.0)
    assert np.max(pitch_deg) == pytest.approx(0.0)
    assert np.min(trajectory.position_wok_m[:, 0]) == pytest.approx(0.0)
    assert np.max(trajectory.position_wok_m[:, 0]) == pytest.approx(0.25 / np.sqrt(2.0))


def test_phasewise_fried_rice_spline_is_c2_and_stops_at_direction_changes() -> None:
    trajectory = generate_fried_rice_trajectory(np.zeros(4), _config())
    knots = trajectory.waypoints.times

    np.testing.assert_allclose(
        trajectory.evaluate_wok(knots, derivative=0),
        trajectory.waypoints.poses,
        atol=3.0e-13,
    )
    for derivative, tolerance in ((1, 1.0e-11), (2, 1.0e-10)):
        np.testing.assert_allclose(
            trajectory.evaluate_wok([knots[0], knots[-1]], derivative=derivative),
            0.0,
            atol=tolerance,
        )

    epsilon = 1.0e-7
    for derivative, tolerance in ((0, 1.0e-5), (1, 1.0e-4), (2, 1.0e-2)):
        left = trajectory.evaluate_wok(knots[1:-1] - epsilon, derivative=derivative)
        right = trajectory.evaluate_wok(knots[1:-1] + epsilon, derivative=derivative)
        assert float(np.max(np.abs(left - right))) < tolerance

    all_internal_speeds = np.linalg.norm(
        trajectory.evaluate_wok(knots[1:-1], derivative=1),
        axis=1,
    )
    np.testing.assert_allclose(all_internal_speeds, 0.0, atol=1.0e-10)
    assert trajectory.validation is not None
    assert trajectory.validation.valid
    assert "cycle_boundary_dwell" not in trajectory.validation.violations


def test_cartesian_limits_remain_hard_validation_contract() -> None:
    base = _config()
    trajectory = generate_fried_rice_trajectory(
        FriedRiceParameters.teaching_default(),
        base,
    )
    assert trajectory.validation is not None
    assert trajectory.validation.valid
    assert trajectory.validation.metrics["max_cartesian_velocity"] < 0.50
    assert trajectory.validation.metrics["max_cartesian_acceleration"] < 3.0
    assert trajectory.validation.metrics["max_angular_velocity"] < 1.50
    assert trajectory.validation.metrics["max_angular_acceleration"] < 6.0

    impossible = deepcopy(base)
    impossible["trajectory"]["max_cartesian_velocity"] = 1.0e-6
    invalid = generate_fried_rice_trajectory(
        FriedRiceParameters.teaching_default(),
        impossible,
    )
    assert invalid.validation is not None
    assert not invalid.validation.valid
    assert "max_cartesian_velocity" in invalid.validation.violations


def test_random_walk_changes_only_when_advancing_episode_and_is_reproducible() -> None:
    first = BoundedEpisodeRandomWalk(4, step_std=0.15, seed=73)
    second = BoundedEpisodeRandomWalk(4, step_std=0.15, seed=73)

    initial = first.current_proposal()
    again = first.current_proposal()
    assert initial.episode_index == 0
    np.testing.assert_array_equal(initial.action, again.action)
    assert not initial.action.flags.writeable

    proposals = [first.advance_episode() for _ in range(20)]
    reference = [second.advance_episode() for _ in range(20)]
    for expected_index, (actual, expected) in enumerate(
        zip(proposals, reference, strict=True),
        start=1,
    ):
        assert actual.episode_index == expected_index
        np.testing.assert_allclose(actual.action, expected.action)
        assert np.all(actual.action >= -1.0)
        assert np.all(actual.action <= 1.0)


def test_random_walk_reflects_large_steps_and_protects_internal_state() -> None:
    walk = BoundedEpisodeRandomWalk(
        4,
        step_std=[20.0, 15.0, 10.0, 5.0],
        initial_action=np.full(4, 0.9),
        seed=5,
    )
    external = walk.current_action
    external[:] = -99.0
    np.testing.assert_allclose(walk.current_action, 0.9)

    for _ in range(100):
        action = walk.advance_episode().action
        assert np.all((-1.0 <= action) & (action <= 1.0))

    with pytest.raises(RandomWalkConfigurationError, match="low < high"):
        BoundedEpisodeRandomWalk(4, low=1.0, high=1.0)
