"""정규화 action 및 P0~P5 기하 테스트."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from wok_sim.trajectory import (
    ActionMappingError,
    PanPose,
    build_cycle_waypoints,
    build_repeated_waypoints,
    map_action,
)


def _trajectory_config() -> dict[str, object]:
    return {
        "cycles": 5,
        "insertion_angle_deg": 45.0,
        "insertion_distance_range_m": [0.02, 0.06],
        "lift_height_range_m": [0.04, 0.08],
        "backward_distance_range_m": [0.03, 0.07],
        "pitch_amplitude_range_rad": [0.1, 0.3],
        "cycle_time_range_s": [0.8, 1.2],
        "insert_phase_ratio_range": [0.15, 0.25],
        "catch_phase_ratio_range": [0.15, 0.25],
        "transition_forward_offset_m": 0.005,
        "transition_lift_offset_m": 0.005,
        "catch_offset_m": [-0.01, 0.0, 0.015],
        "sample_rate_hz": 100.0,
    }


@dataclass(slots=True)
class _ConfigObject:
    trajectory: dict[str, object]


def test_normalized_action_maps_all_seven_physical_parameters() -> None:
    config = _ConfigObject(_trajectory_config())
    low = map_action(np.full(7, -1.0), config)
    middle = map_action(np.zeros(7), config)
    high = map_action(np.full(7, 1.0), config)

    np.testing.assert_allclose(low.as_array(), [0.02, 0.04, 0.03, 0.1, 0.8, 0.15, 0.15])
    np.testing.assert_allclose(high.as_array(), [0.06, 0.08, 0.07, 0.3, 1.2, 0.25, 0.25])
    np.testing.assert_allclose(middle.as_array(), 0.5 * (low.as_array() + high.as_array()))


def test_action_shape_finite_and_strict_range_validation() -> None:
    config = _trajectory_config()
    with pytest.raises(ActionMappingError, match="shape"):
        map_action(np.zeros(6), config)
    with pytest.raises(ActionMappingError, match="NaN"):
        map_action([0.0, 0.0, 0.0, np.nan, 0.0, 0.0, 0.0], config)
    with pytest.raises(ActionMappingError, match=r"\[-1, 1\]"):
        map_action(np.full(7, 1.1), config, clip=False)

    clipped = map_action(np.full(7, 1.1), config)
    np.testing.assert_allclose(clipped.as_array(), [0.06, 0.08, 0.07, 0.3, 1.2, 0.25, 0.25])


def test_p1_is_45_degree_forward_down_translation() -> None:
    config = _trajectory_config()
    parameters = map_action(np.zeros(7), config)
    start = PanPose(0.1, -0.2, 0.3, 0.4, -0.1, 0.2)
    points = build_cycle_waypoints(parameters, config, start_pose=start)
    p0, p1 = points[:2]

    assert p1.x - p0.x == pytest.approx(p0.z - p1.z, abs=1.0e-12)
    assert p1.y == pytest.approx(p0.y, abs=0.0)
    # "45도 삽입"은 pitch 45도가 아니라 translation 방향이다.
    assert p1.pitch == pytest.approx(p0.pitch)


def test_launch_retracts_x_and_lifts_z_in_same_interval() -> None:
    config = _trajectory_config()
    parameters = map_action(np.zeros(7), config)
    points = build_cycle_waypoints(parameters, config)
    p2, p3 = points[2], points[3]

    assert p3.time_s > p2.time_s
    assert p3.x < p2.x
    assert p3.z > p2.z
    assert p3.x == pytest.approx(points[1].x - parameters.backward_distance)
    assert p3.z == pytest.approx(points[1].z + parameters.lift_height)


def test_phase_ratios_create_positive_ordered_durations() -> None:
    config = _trajectory_config()
    parameters = map_action(np.zeros(7), config)
    points = build_cycle_waypoints(parameters, config)
    times = np.asarray([point.time_s for point in points])

    assert np.all(np.diff(times) > 0.0)
    assert times[1] == pytest.approx(parameters.cycle_time * parameters.insert_phase_ratio)
    assert times[4] == pytest.approx(parameters.cycle_time * (1.0 - parameters.catch_phase_ratio))


def test_exactly_five_cycles_are_connected_without_duplicate_boundary_knots() -> None:
    config = _trajectory_config()
    parameters = map_action(np.zeros(7), config)
    sequence = build_repeated_waypoints(parameters, config)

    assert sequence.cycle_count == 5
    assert len(sequence) == 1 + 5 * 5
    assert sum(point.name == "P1" for point in sequence) == 5
    assert sum(point.name == "P5" for point in sequence) == 5
    assert np.all(np.diff(sequence.times) > 0.0)
    np.testing.assert_allclose(
        sequence.cycle_boundaries_s,
        np.arange(6) * parameters.cycle_time,
    )
    for boundary_index in range(1, 6):
        boundary_time = boundary_index * parameters.cycle_time
        point = next(point for point in sequence if point.time_s == pytest.approx(boundary_time))
        assert point.name == "P5"
