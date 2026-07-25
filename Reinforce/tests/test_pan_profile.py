from __future__ import annotations

import numpy as np
import pytest

from wok_sim.visualization.pan_profile import pan_side_profile_local


def _exact_quarter_circle_config() -> dict[str, object]:
    return {
        "pan": {
            "collision_proxy": {
                "bottom_radius_m": 0.060,
                "inner_radius_m": 0.115,
                "rim_radius_m": 0.120,
                "bottom_z_m": 0.0,
                "rim_z_m": 0.055,
                "bottom_thickness_m": 0.005,
                "wall_thickness_m": 0.005,
            }
        }
    }


def test_closed_profile_has_exact_flat_bottom_and_concentric_quarter_circle_sides() -> None:
    arc_points = 9
    profile = pan_side_profile_local(
        _exact_quarter_circle_config(),
        arc_point_count=arc_points,
    )

    assert profile.shape == (4 * arc_points + 1, 3)
    np.testing.assert_allclose(profile[0], [-0.115, 0.0, 0.055], atol=1.0e-15)
    np.testing.assert_allclose(profile[-1], profile[0], atol=0.0)
    np.testing.assert_allclose(profile[:, 1], 0.0)
    assert np.min(profile[:, 0]) == pytest.approx(-0.120)
    assert np.max(profile[:, 0]) == pytest.approx(+0.120)
    assert np.min(profile[:, 2]) == pytest.approx(-0.005)
    assert np.max(profile[:, 2]) == pytest.approx(+0.055)

    # Sequence layout: left inner arc, inner flat, right inner arc, rim,
    # right outer arc, outer flat, left outer arc, close.
    inner_right = profile[arc_points + 1 : 2 * arc_points]
    inner_right = np.vstack(([0.060, 0.0, 0.0], inner_right))
    outer_right_descending = profile[2 * arc_points + 1 : 3 * arc_points]
    outer_right = np.vstack(
        (
            outer_right_descending[::-1],
            [0.120, 0.0, 0.055],
        )
    )
    np.testing.assert_allclose(
        (inner_right[:, 0] - 0.060) ** 2 + (inner_right[:, 2] - 0.055) ** 2,
        0.055**2,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        (outer_right[:, 0] - 0.060) ** 2 + (outer_right[:, 2] - 0.055) ** 2,
        0.060**2,
        atol=1.0e-14,
    )

    inner_flat_start = profile[arc_points - 1]
    inner_flat_end = profile[arc_points]
    outer_flat_start = profile[3 * arc_points - 1]
    outer_flat_end = profile[3 * arc_points]
    np.testing.assert_allclose(inner_flat_start[[0, 2]], [-0.060, 0.0])
    np.testing.assert_allclose(inner_flat_end[[0, 2]], [+0.060, 0.0])
    np.testing.assert_allclose(outer_flat_start[[0, 2]], [+0.060, -0.005])
    np.testing.assert_allclose(outer_flat_end[[0, 2]], [-0.060, -0.005])


def test_profile_rejects_too_few_arc_points() -> None:
    with pytest.raises(ValueError, match="arc_point_count"):
        pan_side_profile_local(_exact_quarter_circle_config(), arc_point_count=2)
