"""최종 복귀와 영구 boundary를 구분하는 유실 판정 검증."""

from __future__ import annotations

import numpy as np
import pytest

from wok_sim.metrics.spill import (
    SpillTracker,
    classify_spilled_particles,
    compute_spill_metrics,
)


def test_inside_particle_is_not_spilled_and_boundary_crossing_is_spilled() -> None:
    positions = np.array(
        [
            [0.02, 0.01, 0.01],
            [0.16, 0.00, 0.04],
            [0.00, 0.00, -0.06],
        ]
    )
    mask = classify_spilled_particles(
        positions,
        rim_radius_m=0.10,
        spill_boundary_radius_m=0.15,
        spill_z_m=-0.05,
    )
    np.testing.assert_array_equal(mask, [False, True, True])

    metrics = compute_spill_metrics(mask, [1.0, 2.0, 3.0])
    assert metrics.spill_count == 2
    assert metrics.spill_count_ratio == pytest.approx(2.0 / 3.0)
    assert metrics.spill_mass_kg == pytest.approx(5.0)
    assert metrics.spill_mass_ratio == pytest.approx(5.0 / 6.0)


def test_transient_flight_that_returns_to_pan_is_not_spill() -> None:
    tracker = SpillTracker(
        1,
        rim_radius_m=0.10,
        spill_boundary_radius_m=0.15,
        spill_z_m=-0.05,
        minimum_no_contact_time_s=0.08,
    )
    tracker.update(0.00, [[0.01, 0.00, 0.01]], [True])
    tracker.update(0.04, [[0.04, 0.00, 0.08]], [False])
    tracker.update(0.09, [[0.03, 0.00, 0.03]], [False])
    tracker.update(0.12, [[0.02, 0.00, 0.01]], [True])

    metrics = tracker.finalize([0.01])
    assert metrics.spill_count == 0
    assert metrics.spill_mass_ratio == 0.0


def test_outside_rim_requires_grace_time_but_boundary_is_irreversible() -> None:
    tracker = SpillTracker(
        1,
        rim_radius_m=0.10,
        spill_boundary_radius_m=0.15,
        minimum_no_contact_time_s=0.10,
    )
    tracker.update(0.00, [[0.02, 0.00, 0.01]], [True])
    tracker.update(0.05, [[0.11, 0.00, 0.05]], [False])
    assert tracker.finalize([0.01]).spill_count == 0
    tracker.update(0.16, [[0.12, 0.00, 0.02]], [False])
    assert tracker.finalize([0.01]).spill_count == 1

    crossed = SpillTracker(
        1,
        rim_radius_m=0.10,
        spill_boundary_radius_m=0.15,
    )
    crossed.update(0.00, [[0.02, 0.00, 0.01]], [True])
    crossed.update(0.05, [[0.16, 0.00, 0.02]], [False])
    crossed.update(0.10, [[0.02, 0.00, 0.01]], [True])
    assert crossed.finalize([0.01]).spill_count == 1
