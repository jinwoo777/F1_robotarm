"""Circular-arc pan collision profile configuration and model-build tests."""

from __future__ import annotations

from copy import deepcopy

import mujoco
import numpy as np
import pytest

from wok_sim.config import load_config
from wok_sim.geometry import quaternion_to_matrix
from wok_sim.simulation import CollisionProxyConfig, ModelBuilder, PanAssetError, ParticleBatch


def _curved_proxy_mapping() -> dict[str, object]:
    return {
        "inner_radius_m": 0.115,
        "bottom_radius_m": 0.060,
        "bottom_z_m": 0.0,
        "wall_height_m": 0.055,
        "bottom_thickness_m": 0.005,
        "wall_thickness_m": 0.005,
        "wall_segments": 24,
        "wall_profile": "circular_arc",
        "radial_wall_segments": 8,
        "rim_radius_m": 0.120,
    }


def _single_particle(position_m: np.ndarray | None = None) -> ParticleBatch:
    position = np.asarray(
        [0.0, 0.0, 0.010] if position_m is None else position_m,
        dtype=float,
    )
    return ParticleBatch(
        radii_m=np.asarray([0.003]),
        masses_kg=np.asarray([0.001]),
        positions_m=position[None, :],
        density_kg_m3=1_000.0,
        target_total_mass_kg=0.001,
        seed=1,
    )


def _config_with_proxy(proxy: dict[str, object]) -> dict[str, object]:
    config = deepcopy(load_config("configs/test.yaml"))
    config["pan"]["collision_proxy"] = proxy
    return config


def test_circular_arc_profile_matches_exact_wok_inner_surface() -> None:
    proxy = CollisionProxyConfig.from_mapping(_curved_proxy_mapping())

    assert proxy.wall_profile == "circular_arc"
    assert proxy.radial_wall_segments == 8
    points = proxy.inner_wall_profile_points()
    assert points.shape == (9, 2)
    np.testing.assert_allclose(points[0], [0.060, 0.0], atol=1.0e-15)
    np.testing.assert_allclose(points[-1], [0.115, 0.055], atol=1.0e-15)

    circle_center = np.asarray([0.060, 0.055])
    np.testing.assert_allclose(
        np.linalg.norm(points - circle_center, axis=1),
        0.055,
        atol=2.0e-15,
    )
    assert np.all(np.diff(points[:, 0]) > 0.0)
    assert np.all(np.diff(points[:, 1]) > 0.0)


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"wall_profile": "unknown"}, "wall_profile"),
        (
            {"wall_profile": "circular_arc", "radial_wall_segments": 1},
            "2 이상",
        ),
        ({"wall_profile": "straight", "radial_wall_segments": 8}, "straight"),
        ({"wall_profile": "circular_arc", "radial_wall_segments": 2.5}, "양의 정수"),
    ),
)
def test_wall_profile_config_rejects_invalid_combinations(
    override: dict[str, object],
    message: str,
) -> None:
    mapping = {**_curved_proxy_mapping(), **override}

    with pytest.raises(PanAssetError, match=message):
        CollisionProxyConfig.from_mapping(mapping)


def test_legacy_straight_profile_keeps_one_wall_box_per_azimuth() -> None:
    config = load_config("configs/test.yaml")
    built = ModelBuilder(config).build(_single_particle())
    proxy = built.pan.proxy

    assert proxy.wall_profile == "straight"
    assert proxy.radial_wall_segments == 1
    assert len(built.pan_collision_geom_ids) == 1 + 2 * proxy.wall_segments
    assert built.metadata["pan_collision_wall_profile"] == "straight"
    assert built.metadata["pan_collision_radial_wall_segments"] == 1
    assert (
        mujoco.mj_name2id(
            built.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "pan_collision_wall_000",
        )
        >= 0
    )


def test_curved_profile_builds_radial_wall_bands_with_inner_faces_on_chords() -> None:
    config = _config_with_proxy(_curved_proxy_mapping())
    built = ModelBuilder(config).build(_single_particle())
    proxy = built.pan.proxy
    profile = proxy.inner_wall_profile_points()

    expected_geom_count = 1 + proxy.wall_segments * (proxy.radial_wall_segments + 1)
    assert len(built.pan_collision_geom_ids) == expected_geom_count
    assert built.metadata["pan_collision_wall_profile"] == "circular_arc"
    assert built.metadata["pan_collision_radial_wall_segments"] == 8
    assert (
        mujoco.mj_name2id(
            built.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "pan_collision_wall_000",
        )
        == -1
    )

    for radial_index, (start_rz, end_rz) in enumerate(zip(profile[:-1], profile[1:], strict=True)):
        geom_id = mujoco.mj_name2id(
            built.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"pan_collision_wall_000_{radial_index:02d}",
        )
        assert geom_id >= 0
        assert int(built.model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_BOX)

        rotation = quaternion_to_matrix(built.model.geom_quat[geom_id])
        normal = rotation[:, 1]
        wall_axis = rotation[:, 2]
        half_thickness = float(built.model.geom_size[geom_id, 1])
        inner_face_midpoint = built.model.geom_pos[geom_id] + normal * half_thickness
        expected_midpoint = np.asarray(
            [
                0.5 * (start_rz[0] + end_rz[0]),
                0.0,
                0.5 * (start_rz[1] + end_rz[1]),
            ]
        )
        expected_delta = np.asarray(
            [
                end_rz[0] - start_rz[0],
                0.0,
                end_rz[1] - start_rz[1],
            ]
        )
        expected_axis = expected_delta / np.linalg.norm(expected_delta)

        assert half_thickness == pytest.approx(0.0025)
        np.testing.assert_allclose(inner_face_midpoint, expected_midpoint, atol=2.0e-12)
        np.testing.assert_allclose(wall_axis, expected_axis, atol=2.0e-12)


def test_curved_wall_inner_face_is_an_active_particle_contact_surface() -> None:
    proxy_mapping = _curved_proxy_mapping()
    proxy = CollisionProxyConfig.from_mapping(proxy_mapping)
    profile = proxy.inner_wall_profile_points()
    radial_index = 4
    start_rz, end_rz = profile[radial_index : radial_index + 2]
    delta = end_rz - start_rz
    inward_normal = np.asarray([-delta[1], 0.0, delta[0]]) / np.linalg.norm(delta)
    chord_midpoint = np.asarray(
        [
            0.5 * (start_rz[0] + end_rz[0]),
            0.0,
            0.5 * (start_rz[1] + end_rz[1]),
        ]
    )
    penetration_m = 0.0002
    particle_center = chord_midpoint + inward_normal * (0.003 - penetration_m)

    built = ModelBuilder(_config_with_proxy(proxy_mapping)).build(_single_particle(particle_center))
    mujoco.mj_forward(built.model, built.data)

    contacts = []
    for index in range(built.data.ncon):
        contact = built.data.contact[index]
        names = {
            mujoco.mj_id2name(
                built.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                int(contact.geom1),
            ),
            mujoco.mj_id2name(
                built.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                int(contact.geom2),
            ),
        }
        contacts.append((names, float(contact.dist)))

    expected_wall = f"pan_collision_wall_000_{radial_index:02d}"
    matching = [distance for names, distance in contacts if expected_wall in names]
    assert matching
    assert min(matching) == pytest.approx(-penetration_m, abs=2.0e-12)
