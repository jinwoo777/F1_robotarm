"""Pan-local X-Z side-profile geometry used by plots and GIFs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from wok_sim.geometry import quaternion_to_matrix
from wok_sim.simulation.pan_model import CollisionProxyConfig


def _circular_arc_with_flat_bottom_tangent(
    *,
    bottom_radius_m: float,
    bottom_z_m: float,
    rim_radius_m: float,
    rim_z_m: float,
    point_count: int,
) -> np.ndarray:
    """Return the right-side arc from the flat bottom to the rim.

    The circle passes through both configured endpoints and has a horizontal
    tangent where it meets the flat bottom. This is an exact circular arc, not
    a fitted Bezier or ellipse.
    """

    radial_change = float(rim_radius_m) - float(bottom_radius_m)
    vertical_change = float(rim_z_m) - float(bottom_z_m)
    if radial_change <= 0.0 or vertical_change <= 0.0:
        raise ValueError("pan side arc에는 rim이 bottom보다 바깥쪽이고 높아야 합니다.")
    circle_radius = (
        radial_change * radial_change + vertical_change * vertical_change
    ) / (2.0 * vertical_change)
    center_x = float(bottom_radius_m)
    center_z = float(bottom_z_m) + circle_radius
    start_angle = -0.5 * np.pi
    end_angle = float(
        np.arctan2(float(rim_z_m) - center_z, float(rim_radius_m) - center_x)
    )
    angles = np.linspace(start_angle, end_angle, point_count, dtype=float)
    points = np.column_stack(
        (
            center_x + circle_radius * np.cos(angles),
            np.zeros(point_count, dtype=float),
            center_z + circle_radius * np.sin(angles),
        )
    )
    # Avoid tiny trigonometric endpoint drift in saved artifacts and tests.
    points[0] = (bottom_radius_m, 0.0, bottom_z_m)
    points[-1] = (rim_radius_m, 0.0, rim_z_m)
    return points


def pan_side_profile_local(
    config: Mapping[str, Any],
    *,
    arc_point_count: int = 33,
) -> np.ndarray:
    """Build a closed flat-bottom, double-arc wok section in pan-local X-Z.

    The inner surface uses ``bottom_radius_m/bottom_z_m`` and
    ``inner_radius_m/rim_z_m``. The outer surface independently uses the
    configured bottom underside and outer rim. For the 60/115/120 mm,
    55 mm-high, 5 mm-thick profile both sides are exact concentric quarter
    circles of radii 55 mm and 60 mm.

    This is display geometry only. It does not alter MuJoCo collision shapes.
    """

    if (
        isinstance(arc_point_count, bool)
        or int(arc_point_count) != arc_point_count
        or int(arc_point_count) < 3
    ):
        raise ValueError("arc_point_count는 3 이상의 정수여야 합니다.")
    arc_point_count = int(arc_point_count)
    pan = config.get("pan", {})
    proxy_mapping = pan.get("collision_proxy", {}) if isinstance(pan, Mapping) else {}
    proxy = CollisionProxyConfig.from_mapping(proxy_mapping)

    inner_right = _circular_arc_with_flat_bottom_tangent(
        bottom_radius_m=proxy.bottom_radius_m,
        bottom_z_m=proxy.bottom_z_m,
        rim_radius_m=proxy.inner_radius_m,
        rim_z_m=proxy.rim_z_m,
        point_count=arc_point_count,
    )
    outer_bottom_z = proxy.bottom_z_m - proxy.bottom_thickness_m
    outer_right = _circular_arc_with_flat_bottom_tangent(
        bottom_radius_m=proxy.bottom_radius_m,
        bottom_z_m=outer_bottom_z,
        rim_radius_m=proxy.rim_radius_m,
        rim_z_m=proxy.rim_z_m,
        point_count=arc_point_count,
    )

    left_inner_descending = inner_right[::-1].copy()
    left_inner_descending[:, 0] *= -1.0
    right_outer_descending = outer_right[::-1]
    left_outer_ascending = outer_right.copy()
    left_outer_ascending[:, 0] *= -1.0
    first_point = left_inner_descending[0].copy()

    return np.vstack(
        (
            left_inner_descending,
            np.asarray([[proxy.bottom_radius_m, 0.0, proxy.bottom_z_m]]),
            inner_right[1:],
            np.asarray([[proxy.rim_radius_m, 0.0, proxy.rim_z_m]]),
            right_outer_descending[1:],
            np.asarray([[-proxy.bottom_radius_m, 0.0, outer_bottom_z]]),
            left_outer_ascending[1:],
            first_point[None, :],
        )
    )


def transform_pan_local_history(
    pan_position_world_m: np.ndarray,
    pan_quaternion_wxyz: np.ndarray,
    local_points_m: np.ndarray,
) -> np.ndarray:
    """Transform fixed pan-local points at every recorded pan pose."""

    pan = np.asarray(pan_position_world_m, dtype=float)
    quaternions = np.asarray(pan_quaternion_wxyz, dtype=float)
    local = np.asarray(local_points_m, dtype=float)
    if pan.ndim != 2 or pan.shape[1:] != (3,):
        raise ValueError("pan_position_world_m shape은 (T,3)이어야 합니다.")
    if quaternions.shape != (len(pan), 4):
        raise ValueError("pan_quaternion_wxyz shape은 (T,4)여야 합니다.")
    if local.ndim != 2 or local.shape[1:] != (3,) or not len(local):
        raise ValueError("local_points_m shape은 (P,3)이어야 합니다.")
    if not all(np.isfinite(item).all() for item in (pan, quaternions, local)):
        raise ValueError("pan pose와 local profile은 모두 유한해야 합니다.")
    rotations = np.stack([quaternion_to_matrix(item) for item in quaternions])
    return pan[:, None, :] + np.einsum("tij,pj->tpi", rotations, local)


def resample_point_history(
    time_s: np.ndarray,
    values: np.ndarray,
    sample_time_s: np.ndarray,
) -> np.ndarray:
    """Linearly resample a ``(T,...,3)`` recorded point history."""

    source_time = np.asarray(time_s, dtype=float)
    source = np.asarray(values, dtype=float)
    sample_time = np.asarray(sample_time_s, dtype=float)
    if (
        source_time.ndim != 1
        or len(source_time) < 2
        or np.any(np.diff(source_time) <= 0.0)
    ):
        raise ValueError("time_s는 엄격히 증가하는 2개 이상의 1차원 배열이어야 합니다.")
    if source.ndim < 2 or source.shape[0] != len(source_time) or source.shape[-1] != 3:
        raise ValueError("values shape은 (T,...,3)이어야 합니다.")
    if sample_time.ndim != 1 or np.any(np.diff(sample_time) <= 0.0):
        raise ValueError("sample_time_s는 엄격히 증가하는 1차원 배열이어야 합니다.")
    if (
        len(sample_time) == 0
        or sample_time[0] < source_time[0] - 1.0e-12
        or sample_time[-1] > source_time[-1] + 1.0e-12
    ):
        raise ValueError("sample_time_s는 source time 범위 안이어야 합니다.")
    if not all(np.isfinite(item).all() for item in (source_time, source, sample_time)):
        raise ValueError("resample 입력은 모두 유한해야 합니다.")
    flattened = source.reshape(len(source_time), -1)
    sampled = np.column_stack(
        [
            np.interp(sample_time, source_time, flattened[:, index])
            for index in range(flattened.shape[1])
        ]
    )
    return sampled.reshape((len(sample_time), *source.shape[1:]))


__all__ = [
    "pan_side_profile_local",
    "resample_point_history",
    "transform_pan_local_history",
]
