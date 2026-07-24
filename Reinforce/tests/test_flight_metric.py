"""접촉 전이, 상대 비행 높이, launch angle 검증."""

from __future__ import annotations

import numpy as np
import pytest

from wok_sim.metrics.flight import FlightTracker, flight_statistics_from_summary
from wok_sim.simulation.contact_events import (
    ContactEventTracker,
)
from wok_sim.simulation.contact_events import (
    FlightEvent as SimulationFlightEvent,
)


def test_contact_transition_tracks_takeoff_apex_and_landing() -> None:
    tracker = FlightTracker(2, minimum_takeoff_vertical_velocity_m_s=0.05)
    tracker.update(
        0.0,
        [[0.0, 0.0, 0.10], [0.0, 0.0, 0.10]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [True, True],
        particle_positions_local=[[0.0, 0.0, 0.01], [0.0, 0.0, 0.01]],
    )
    tracker.update(
        0.1,
        [[0.0, 0.0, 0.20], [0.0, 0.0, 0.10]],
        [[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        [False, True],
        pan_surface_velocity_world=[0.2, 0.0, 0.1],
        particle_positions_local=[[0.0, 0.0, 0.05], [0.0, 0.0, 0.01]],
    )
    tracker.update(
        0.2,
        [[0.1, 0.0, 0.50], [0.0, 0.0, 0.10]],
        [[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [False, True],
        particle_positions_local=[[0.1, 0.0, 0.35], [0.0, 0.0, 0.01]],
    )
    tracker.update(
        0.4,
        [[0.0, 0.0, 0.10], [0.0, 0.0, 0.10]],
        [[0.0, 0.0, -0.2], [0.0, 0.0, 0.0]],
        [True, True],
        particle_positions_local=[[0.0, 0.0, 0.01], [0.0, 0.0, 0.01]],
    )

    records = tracker.particle_records()
    first = records[0]
    assert first["flight_count"] == 1
    assert first["max_flight_height_world"] == pytest.approx(0.30)
    assert first["max_flight_height_relative"] == pytest.approx(0.30)
    assert first["takeoff_time"] == pytest.approx(0.1)
    assert first["landing_time"] == pytest.approx(0.4)
    assert first["flight_duration"] == pytest.approx(0.3)
    assert first["launch_angle_xz"] == pytest.approx(np.arctan2(0.9, 0.8))
    assert records[1]["flight_count"] == 0

    statistics = tracker.statistics(radii_m=[0.006, 0.010], spilled_mask=[True, False])
    assert statistics["particles_with_flight"] == 1
    assert statistics["mean_flight_height"] == pytest.approx(0.30)
    assert statistics["max_flight_height"] == pytest.approx(0.30)
    assert statistics["spilled_mean_launch_angle"] == pytest.approx(np.arctan2(0.9, 0.8))


def test_nonpositive_relative_vertical_velocity_is_not_takeoff() -> None:
    tracker = FlightTracker(1)
    tracker.update(0.0, [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [True])
    tracker.update(
        0.1,
        [[0.0, 0.0, 0.01]],
        [[0.3, 0.0, 0.1]],
        [False],
        pan_surface_velocity_world=[0.0, 0.0, 0.2],
    )
    assert tracker.statistics()["flight_event_count"] == 0


def test_rotated_pan_uses_pan_local_vertical_velocity() -> None:
    tracker = FlightTracker(1)
    rotation_world_from_local = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    tracker.update(
        0.0,
        [[0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0]],
        [True],
        pan_rotation_world_from_local=rotation_world_from_local,
    )
    # world +x는 이 회전에서 pan-local +z다.
    tracker.update(
        0.1,
        [[0.1, 0.0, 0.0]],
        [[1.0, 0.0, 0.0]],
        [False],
        particle_positions_local=[[0.0, 0.0, 0.1]],
        pan_rotation_world_from_local=rotation_world_from_local,
    )

    record = tracker.particle_records()[0]
    assert record["flight_count"] == 1
    assert record["launch_angle_xz"] == pytest.approx(np.pi / 2.0)


def test_simulator_particle_summary_adapter() -> None:
    statistics = flight_statistics_from_summary(
        {
            "max_flight_height_relative_m": [0.2, 0.0, 0.4],
            "max_flight_height_world_m": [0.1, 0.0, 0.3],
            "launch_angle_xz_rad": [0.5, np.nan, 1.0],
            "flight_count": [1, 0, 2],
        },
        radii_m=[0.006, 0.008, 0.010],
        spilled_mask=[False, False, True],
    )
    assert statistics["flight_event_count"] == 3
    assert statistics["mean_flight_height"] == pytest.approx(0.3)
    assert statistics["max_flight_height"] == pytest.approx(0.4)
    assert statistics["mean_launch_angle"] == pytest.approx(0.75)
    assert statistics["spilled_mean_launch_angle"] == pytest.approx(1.0)


def test_multi_flight_summary_uses_coherent_representative_event() -> None:
    tracker = ContactEventTracker(1)
    common = {
        "particle_index": 0,
        "takeoff_position_world_m": (0.0, 0.0, 0.1),
        "takeoff_position_pan_m": (0.0, 0.0, 0.01),
        "takeoff_relative_velocity_world_m_s": (1.0, 0.0, 1.0),
        "takeoff_relative_velocity_pan_m_s": (1.0, 0.0, 1.0),
        "elevation_angle_rad": 0.5,
        "max_world_z_m": 1.0,
        "max_pan_z_m": 1.0,
    }
    tracker.events = [
        SimulationFlightEvent(
            **common,
            takeoff_time_s=1.0,
            landing_time_s=1.3,
            flight_duration_s=0.3,
            launch_angle_xz_rad=0.7,
            max_height_world_m=0.2,
            max_height_relative_m=0.5,
        ),
        SimulationFlightEvent(
            **common,
            takeoff_time_s=2.0,
            landing_time_s=2.6,
            flight_duration_s=0.6,
            launch_angle_xz_rad=1.2,
            max_height_world_m=0.8,
            max_height_relative_m=0.4,
        ),
    ]

    summary = tracker.particle_summary()

    assert summary["flight_count"][0] == 2
    assert summary["max_flight_height_world_m"][0] == pytest.approx(0.8)
    assert summary["max_flight_height_relative_m"][0] == pytest.approx(0.5)
    assert summary["takeoff_time_s"][0] == pytest.approx(1.0)
    assert summary["landing_time_s"][0] == pytest.approx(1.3)
    assert summary["flight_duration_s"][0] == pytest.approx(0.3)
    assert summary["total_flight_duration_s"][0] == pytest.approx(0.9)
    assert summary["launch_angle_xz_rad"][0] == pytest.approx(0.7)
