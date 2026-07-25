"""볶음밥 4단 teaching과 episode random walk 테스트."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from wok_sim.config import load_config
from wok_sim.exploration import (
    BoundedEpisodeRandomWalk,
    RandomWalkConfigurationError,
)
from wok_sim.robot import M0609CartesianCaps, validate_m0609_cartesian_motion
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
            "max_angular_velocity": 1.30,
            "max_cartesian_acceleration": 1.50,
            "max_angular_acceleration": 2.70,
            "max_cartesian_jerk": 16.0,
            "max_angular_jerk": 22.0,
            "allow_cycle_boundary_stop": True,
            "workspace": {
                "x_m": [-0.02, 0.24],
                "y_m": [-0.001, 0.001],
                "z_m": [-0.04, 0.22],
                "pitch_rad": [-0.20, 0.75],
            },
            "y_tolerance": 1.0e-10,
            "roll_tolerance": 1.0e-10,
            "yaw_tolerance": 1.0e-10,
            "start_position_m": [0.0, 0.0, 0.20],
            "start_rpy_rad": [0.0, 0.0, 0.0],
            "fried_rice": {
                "cycles": 5,
                "descent_angle_range_rad": [
                    float(np.deg2rad(35.0)),
                    float(np.deg2rad(55.0)),
                ],
                "pan_tilt_angle_range_rad": [
                    float(np.deg2rad(20.0)),
                    float(np.deg2rad(40.0)),
                ],
                "descent_speed_range_m_s": [0.38, 0.48],
                "lift_angle_range_rad": [0.0, float(np.deg2rad(30.0))],
                "insertion_distance_m": 0.25,
                "angular_speed_rad_s": 1.20,
                "tilt_direction": 1,
                "linear_acceleration_limit_m_s2": 1.40,
                "angular_acceleration_limit_rad_s2": 2.50,
                "linear_jerk_limit_m_s3": 15.0,
                "angular_jerk_limit_rad_s3": 20.0,
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
        "descent_angle",
        "pan_tilt_angle",
        "descent_speed",
        "lift_angle",
    )
    np.testing.assert_allclose(
        middle.as_array(),
        [np.deg2rad(45.0), np.deg2rad(30.0), 0.43, np.deg2rad(15.0)],
    )
    np.testing.assert_allclose(
        low.as_array(),
        [np.deg2rad(35.0), np.deg2rad(20.0), 0.38, 0.0],
    )
    np.testing.assert_allclose(
        high.as_array(),
        [np.deg2rad(55.0), np.deg2rad(40.0), 0.48, np.deg2rad(30.0)],
    )
    np.testing.assert_allclose(
        middle.as_array(),
        0.5 * (low.as_array() + high.as_array()),
    )
    assert middle.insertion_distance == pytest.approx(0.25)
    assert middle.tilt_recovery_angle == pytest.approx(np.deg2rad(15.0))
    assert middle.angular_speed == pytest.approx(1.20)


def test_fried_rice_action_validation_is_separate_from_legacy_seven_dimensions() -> None:
    with pytest.raises(FriedRiceTrajectoryError, match="shape"):
        map_fried_rice_action(np.zeros(7), _config())
    with pytest.raises(FriedRiceTrajectoryError, match=r"\[-1, 1\]"):
        map_fried_rice_action(np.full(4, 1.01), _config(), clip=False)
    with pytest.raises(FriedRiceTrajectoryError, match="NaN"):
        map_fried_rice_action([0.0, np.nan, 0.0, 0.0], _config())


def test_four_teaching_stages_match_requested_geometry_and_timing() -> None:
    parameters = FriedRiceParameters.teaching_default()
    points = build_fried_rice_cycle_waypoints(parameters, _config())
    p0, p1, p2, p3, p4 = points

    assert [point.name for point in points] == ["P0", "P1", "P2", "P3", "P4"]
    np.testing.assert_allclose(p1.position_m, p0.position_m, atol=0.0)
    assert p1.pitch - p0.pitch == pytest.approx(np.deg2rad(30.0))
    assert p2.x - p1.x == pytest.approx(0.25 / np.sqrt(2.0))
    assert p1.z - p2.z == pytest.approx(0.25 / np.sqrt(2.0))
    assert np.linalg.norm(p2.position_m - p1.position_m) == pytest.approx(0.25)
    assert p2.pitch == pytest.approx(p1.pitch)
    np.testing.assert_allclose(p3.position_m, p2.position_m, atol=0.0)
    assert p3.pitch - p0.pitch == pytest.approx(np.deg2rad(15.0))
    np.testing.assert_allclose(p4.pose.as_array(), p0.pose.as_array(), atol=0.0)

    tilt_duration, descent_duration, recovery_duration, return_duration = (
        parameters.phase_durations_s
    )
    assert parameters.phase_durations_s == pytest.approx(
        (1.1624473515, 1.0901162791, 0.9226350743, 1.0901162791)
    )
    assert p1.time_s == pytest.approx(tilt_duration)
    assert p2.time_s == pytest.approx(tilt_duration + descent_duration)
    assert p3.time_s == pytest.approx(tilt_duration + descent_duration + recovery_duration)
    assert p4.time_s == pytest.approx(
        tilt_duration + descent_duration + recovery_duration + return_duration
    )
    assert p4.time_s == pytest.approx(parameters.cycle_time)


def test_five_cycles_are_one_phasewise_waypoint_sequence_without_duplicate_knots() -> None:
    parameters = FriedRiceParameters.teaching_default()
    sequence = build_repeated_fried_rice_waypoints(parameters, _config())

    assert sequence.cycle_count == 5
    assert len(sequence) == 1 + 4 * 5
    assert np.all(np.diff(sequence.times) > 0.0)
    np.testing.assert_allclose(
        sequence.cycle_boundaries_s,
        np.arange(6) * parameters.cycle_time,
    )
    assert sum(point.name == "P1" for point in sequence) == 5
    assert sum(point.name == "P2" for point in sequence) == 5
    assert sum(point.name == "P3" for point in sequence) == 5
    assert sum(point.name == "P4" for point in sequence) == 5
    assert not any(point.name.startswith("S") for point in sequence)


def test_phasewise_minimum_jerk_has_no_geometry_or_pitch_overshoot() -> None:
    trajectory = generate_fried_rice_trajectory(np.zeros(4), _config())
    pitch_deg = np.rad2deg(trajectory.orientation_wok_rpy_rad[:, 1])

    assert np.min(pitch_deg) == pytest.approx(0.0)
    assert np.max(pitch_deg) == pytest.approx(30.0)
    assert np.min(trajectory.position_wok_m[:, 0]) == pytest.approx(0.0)
    assert np.max(trajectory.position_wok_m[:, 0]) == pytest.approx(0.25 / np.sqrt(2.0))
    assert np.min(trajectory.position_wok_m[:, 2]) == pytest.approx(0.20 - 0.25 / np.sqrt(2.0))


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
    assert trajectory.validation.metrics["max_cartesian_acceleration"] < 1.50
    assert trajectory.validation.metrics["max_cartesian_jerk"] < 16.0
    assert trajectory.validation.metrics["max_angular_velocity"] < 1.30
    assert trajectory.validation.metrics["max_angular_acceleration"] < 2.70
    assert trajectory.validation.metrics["max_angular_jerk"] <= 22.0

    impossible = deepcopy(base)
    impossible["trajectory"]["max_cartesian_velocity"] = 1.0e-6
    invalid = generate_fried_rice_trajectory(
        FriedRiceParameters.teaching_default(),
        impossible,
    )
    assert invalid.validation is not None
    assert not invalid.validation.valid
    assert "max_cartesian_velocity" in invalid.validation.violations


def test_continuous_low_lift_preview_has_no_internal_dwell_and_passes_m0609_gate() -> None:
    config = deepcopy(_config())
    config["trajectory"].update(
        {
            "allow_cycle_boundary_stop": False,
            "workspace": {
                "x_m": [-0.03, 0.24],
                "y_m": [-0.001, 0.001],
                "z_m": [-0.04, 0.23],
                "pitch_rad": [-0.23, 0.90],
            },
        }
    )
    config["trajectory"]["fried_rice"].update(
        {
            "motion_profile": "continuous_blended_global_quintic",
            "pretilt_translation_fraction": 0.20,
            "lift_return_translation_fraction": 0.65,
            "lift_return_arc_height_m": 0.025,
            "continuous_time_scale": 1.50,
        }
    )
    action = np.asarray([0.0, 0.0, 0.0, -2.0 / 3.0])
    parameters = map_fried_rice_action(action, config)
    points = build_fried_rice_cycle_waypoints(parameters, config)
    p0, p1, p2, p3, p4 = points
    descent = p2.position_m - p0.position_m

    assert parameters.tilt_recovery_angle == pytest.approx(np.deg2rad(5.0))
    assert parameters.time_scale == pytest.approx(1.50)
    np.testing.assert_allclose(p1.position_m - p0.position_m, 0.20 * descent)
    np.testing.assert_allclose(
        p3.position_m,
        p0.position_m + 0.65 * descent + np.asarray([0.0, 0.0, 0.025]),
    )
    np.testing.assert_allclose(p4.pose.as_array(), p0.pose.as_array())

    trajectory = generate_fried_rice_trajectory(action, config)
    knots = trajectory.waypoints.times
    internal_velocity = trajectory.evaluate_wok(knots[1:-1], derivative=1)
    internal_speed = np.linalg.norm(internal_velocity, axis=1)

    assert float(np.min(internal_speed)) > 0.01
    assert trajectory.validation is not None
    assert trajectory.validation.valid
    assert trajectory.validation.metrics["min_cycle_boundary_velocity"] > 0.01

    caps = M0609CartesianCaps()
    report = validate_m0609_cartesian_motion(trajectory, caps=caps, payload_kg=0.76)
    assert report.within_caps
    assert report.required_uniform_time_scale == pytest.approx(1.0)
    assert report.peaks["tcp_linear_acceleration"] < caps.linear_acceleration_m_s2
    assert report.peaks["tcp_angular_acceleration"] < caps.angular_acceleration_rad_s2
    assert report.peaks["tcp_linear_jerk"] < caps.linear_jerk_m_s3
    assert report.peaks["tcp_angular_jerk"] < caps.angular_jerk_rad_s3


def test_continuous_lift_retreat_holds_insertion_pitch_and_uses_lift_path_angle() -> None:
    config = deepcopy(_config())
    config["trajectory"]["fried_rice"].update(
        {
            "motion_profile": "continuous_blended_global_quintic",
            "pretilt_translation_fraction": 0.20,
            "lift_return_translation_fraction": 0.65,
            "lift_return_arc_height_m": 0.025,
            "hold_insertion_pitch_during_lift_retreat": True,
            "continuous_time_scale": 1.50,
        }
    )
    low_lift = map_fried_rice_action(np.asarray([0.0, 0.0, 0.0, -1.0]), config)
    high_lift = map_fried_rice_action(np.asarray([0.0, 0.0, 0.0, 1.0]), config)
    low_points = build_fried_rice_cycle_waypoints(low_lift, config)
    high_points = build_fried_rice_cycle_waypoints(high_lift, config)
    _, _, low_p2, low_p3, low_p4 = low_points
    _, _, high_p2, high_p3, _ = high_points

    assert low_lift.tilt_recovery_angle == pytest.approx(0.0)
    assert high_lift.tilt_recovery_angle == pytest.approx(np.deg2rad(30.0))
    assert low_p3.pitch == pytest.approx(low_p2.pitch)
    assert high_p3.pitch == pytest.approx(high_p2.pitch)
    retreat_distance_x = abs(high_p2.x - high_p3.x)
    assert high_p3.z - low_p3.z == pytest.approx(
        retreat_distance_x * np.tan(np.deg2rad(30.0))
    )
    np.testing.assert_allclose(low_p4.pose.as_array(), low_points[0].pose.as_array())

    trajectory = generate_fried_rice_trajectory(np.zeros(4), config)
    p2_time, p3_time = trajectory.waypoints.times[2:4]
    retreat_times = np.linspace(p2_time, p3_time, 51)
    retreat_pitch = trajectory.evaluate_wok(retreat_times)[:, 4]
    np.testing.assert_allclose(retreat_pitch, low_p2.pitch, atol=1.0e-12)
    retreat_linear_speed = np.linalg.norm(
        trajectory.evaluate_wok(retreat_times[1:-1], derivative=1)[:, :3],
        axis=1,
    )
    assert float(np.max(retreat_linear_speed)) > 0.01


def test_repeated_continuous_cycles_hold_pitch_through_retreat_to_next_insertion() -> None:
    config = deepcopy(_config())
    config["trajectory"]["fried_rice"].update(
        {
            "motion_profile": "continuous_blended_global_quintic",
            "pretilt_translation_fraction": 0.20,
            "lift_return_translation_fraction": 0.65,
            "lift_return_arc_height_m": 0.025,
            "hold_insertion_pitch_during_lift_retreat": True,
            "hold_insertion_pitch_between_cycles": True,
            "continuous_time_scale": 1.50,
        }
    )
    parameters = map_fried_rice_action(np.zeros(4), config)
    sequence = build_repeated_fried_rice_waypoints(parameters, config)
    maximum_pitch = parameters.pan_tilt_angle

    intermediate_p4 = [
        point for point in sequence if point.name == "P4" and point.cycle_index < 4
    ]
    assert len(intermediate_p4) == 4
    assert all(point.pitch == pytest.approx(maximum_pitch) for point in intermediate_p4)
    final_p4 = sequence[-1]
    assert final_p4.name == "P4"
    assert final_p4.pitch == pytest.approx(0.0)

    trajectory = generate_fried_rice_trajectory(np.zeros(4), config)
    # 첫 cycle P3부터 다음 cycle P1까지는 후퇴와 재삽입이 계속 진행되지만
    # pitch는 삽입 각도에서 한 번도 0도로 복원되지 않는다.
    first_p3_time = sequence[3].time_s
    second_p1_time = sequence[5].time_s
    transition_times = np.linspace(first_p3_time, second_p1_time, 101)
    transition_pose = trajectory.evaluate_wok(transition_times)
    np.testing.assert_allclose(transition_pose[:, 4], maximum_pitch, atol=1.0e-12)
    transition_speed = np.linalg.norm(
        trajectory.evaluate_wok(transition_times, derivative=1)[:, :3],
        axis=1,
    )
    assert float(np.max(transition_speed)) > 0.01


def test_adaptive_retiming_preserves_zero_to_30_lift_range_and_slows_unsafe_corner() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "fried_rice.yaml")
    # 35° descent, 40° pan tilt, 0.48 m/s, 0° lift는 고정 0.875 timing에서
    # 각속도/각 jerk가 큰 corner다.
    action = np.asarray([-1.0, 1.0, 1.0, -1.0])

    trajectory = generate_fried_rice_trajectory(action, config)
    parameters = trajectory.parameters
    caps = M0609CartesianCaps.from_mapping(config["robot"]["cartesian_caps"])
    report = validate_m0609_cartesian_motion(trajectory, caps=caps, payload_kg=0.76)

    assert parameters.tilt_recovery_angle == pytest.approx(0.0)
    assert parameters.lift_return_time_factor == pytest.approx(0.875)
    assert parameters.adaptive_retiming_scale > 1.0
    assert report.within_caps
    limits = {
        "tcp_linear_velocity": caps.linear_velocity_m_s,
        "tcp_angular_velocity": caps.angular_velocity_rad_s,
        "tcp_linear_acceleration": caps.linear_acceleration_m_s2,
        "tcp_angular_acceleration": caps.angular_acceleration_rad_s2,
        "tcp_linear_jerk": caps.linear_jerk_m_s3,
        "tcp_angular_jerk": caps.angular_jerk_rad_s3,
    }
    for name, limit in limits.items():
        assert report.peaks[name] <= 0.952 * limit


def test_fast_150_adaptive_retiming_keeps_pitch_hold_motion_inside_target() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "fried_rice_fast_150.yaml")
    trajectory = generate_fried_rice_trajectory(
        np.asarray([-1.0, -1.0, 0.0, 0.0]),
        config,
    )
    parameters = trajectory.parameters
    caps = M0609CartesianCaps.from_mapping(config["robot"]["cartesian_caps"])
    report = validate_m0609_cartesian_motion(trajectory, caps=caps, payload_kg=0.76)

    # pitch-hold profile은 마지막 복원 구간의 angular jerk가 지배하므로
    # 기존 geometry보다 빨라지는 대신 target까지 자동으로 느려진다.
    assert parameters.adaptive_retiming_scale > 1.0
    assert report.within_caps
    speed_acceleration_utilization = (
        report.peaks["tcp_linear_velocity"] / caps.linear_velocity_m_s,
        report.peaks["tcp_angular_velocity"] / caps.angular_velocity_rad_s,
        report.peaks["tcp_linear_acceleration"] / caps.linear_acceleration_m_s2,
        report.peaks["tcp_angular_acceleration"] / caps.angular_acceleration_rad_s2,
    )
    jerk_utilization = (
        report.peaks["tcp_linear_jerk"] / caps.linear_jerk_m_s3,
        report.peaks["tcp_angular_jerk"] / caps.angular_jerk_rad_s3,
    )
    assert max(speed_acceleration_utilization) <= 0.902
    assert max(jerk_utilization) <= 1.002
    assert max((*speed_acceleration_utilization, *jerk_utilization)) >= 0.895


def test_random_walk_changes_only_when_advancing_episode_and_is_reproducible() -> None:
    first = BoundedEpisodeRandomWalk(3, step_std=0.15, seed=73)
    second = BoundedEpisodeRandomWalk(3, step_std=0.15, seed=73)

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
        3,
        step_std=[20.0, 15.0, 10.0],
        initial_action=np.full(3, 0.9),
        seed=5,
    )
    external = walk.current_action
    external[:] = -99.0
    np.testing.assert_allclose(walk.current_action, 0.9)

    for _ in range(100):
        action = walk.advance_episode().action
        assert np.all((-1.0 <= action) & (action <= 1.0))

    with pytest.raises(RandomWalkConfigurationError, match="low < high"):
        BoundedEpisodeRandomWalk(3, low=1.0, high=1.0)
