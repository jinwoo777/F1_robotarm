"""조건부 엔트로피 혼합도와 보조 metric 검증."""

from __future__ import annotations

import numpy as np
import pytest

from wok_sim.metrics.mixing import (
    assign_initial_labels,
    compute_mixing_metrics,
    mixing_score,
    occupancy_ratio,
    size_group_labels,
)


def test_separated_distribution_scores_low_and_uniform_mixture_high() -> None:
    cell_centers = np.array(
        [
            [-0.5, -0.5],
            [0.5, -0.5],
            [-0.5, 0.5],
            [0.5, 0.5],
        ]
    )
    labels = np.repeat(np.arange(4), 4)
    separated = np.repeat(cell_centers, 4, axis=0)

    mixed_points: list[np.ndarray] = []
    mixed_labels: list[int] = []
    offsets = np.array(
        [
            [-0.03, -0.03],
            [0.03, -0.03],
            [-0.03, 0.03],
            [0.03, 0.03],
        ]
    )
    for center in cell_centers:
        for label, offset in enumerate(offsets):
            mixed_points.append(center + offset)
            mixed_labels.append(label)
    mixed = np.asarray(mixed_points)

    options = {
        "grid_rows": 2,
        "grid_cols": 2,
        "bounds": (-1.0, 1.0, -1.0, 1.0),
    }
    separated_score = mixing_score(separated, labels, **options)
    mixed_score = mixing_score(mixed, np.asarray(mixed_labels), **options)

    assert separated_score == pytest.approx(0.0, abs=1e-12)
    assert mixed_score == pytest.approx(1.0, abs=1e-12)
    assert 0.0 <= separated_score <= 1.0
    assert 0.0 <= mixed_score <= 1.0


def test_initial_and_final_use_common_pan_local_grid() -> None:
    initial = np.array(
        [
            [-0.75, -0.75],
            [-0.70, -0.70],
            [0.70, -0.75],
            [0.75, -0.70],
            [-0.75, 0.70],
            [-0.70, 0.75],
            [0.70, 0.70],
            [0.75, 0.75],
        ]
    )
    labels = np.repeat(np.arange(4), 2)
    final = np.array(
        [
            [-0.75, -0.75],
            [0.75, 0.75],
            [-0.75, -0.75],
            [0.75, 0.75],
            [-0.75, -0.75],
            [0.75, 0.75],
            [-0.75, -0.75],
            [0.75, 0.75],
        ]
    )
    radii = np.linspace(0.006, 0.010, initial.shape[0])
    metrics = compute_mixing_metrics(
        initial,
        final,
        initial_region_labels=labels,
        radii_m=radii,
        grid_rows=2,
        grid_cols=2,
        bounds=(-1.0, 1.0, -1.0, 1.0),
    )

    assert metrics["initial_mixing_score"] == pytest.approx(0.0)
    assert metrics["final_mixing_score"] == pytest.approx(1.0)
    assert metrics["mixing_improvement"] == pytest.approx(1.0)
    assert 0.0 <= metrics["final_size_mixing_score"] <= 1.0
    assert 0.0 <= metrics["initial_occupancy_ratio"] <= 1.0
    assert 0.0 <= metrics["final_occupancy_ratio"] <= 1.0
    assert size_group_labels(radii).shape == radii.shape


def test_empty_distribution_is_rejected_and_empty_cells_are_safe() -> None:
    with pytest.raises(ValueError, match="at least one"):
        mixing_score(np.empty((0, 2)), np.empty(0))

    score = mixing_score(
        np.array([[-0.5, -0.5], [0.5, 0.5]]),
        np.array([0, 1]),
        grid_rows=8,
        grid_cols=8,
        bounds=(-1.0, 1.0, -1.0, 1.0),
    )
    assert np.isfinite(score)
    assert occupancy_ratio(
        np.array([[-0.5, -0.5], [0.5, 0.5]]),
        grid_rows=8,
        grid_cols=8,
        bounds=(-1.0, 1.0, -1.0, 1.0),
    ) == pytest.approx(2.0 / 64.0)


def test_outside_particles_reduce_score_without_crashing() -> None:
    points = np.array(
        [
            [-0.5, -0.5],
            [-0.5, -0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        ]
    )
    labels = np.array([0, 1, 0, 1])
    options = {
        "grid_rows": 2,
        "grid_cols": 2,
        "bounds": (-1.0, 1.0, -1.0, 1.0),
    }
    assert mixing_score(points, labels, **options) == pytest.approx(1.0)

    partly_outside = points.copy()
    partly_outside[2:] = 2.0
    assert mixing_score(partly_outside, labels, **options) == pytest.approx(0.5)

    all_outside = np.full_like(points, 2.0)
    assert mixing_score(all_outside, labels, **options) == pytest.approx(0.0)
    metrics = compute_mixing_metrics(
        points,
        all_outside,
        initial_region_labels=labels,
        **options,
    )
    assert metrics["final_mixing_score"] == pytest.approx(0.0)
    assert metrics["final_occupancy_ratio"] == pytest.approx(0.0)


def test_left_right_labels_follow_wok_frame_y_axis() -> None:
    labels = assign_initial_labels(
        np.array(
            [
                [10.0, -1.0],
                [-10.0, 1.0],
            ]
        ),
        mode="left_right",
    )
    assert labels.tolist() == [0, 1]
