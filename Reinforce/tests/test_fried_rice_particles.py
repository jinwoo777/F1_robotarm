"""볶음밥 heterogeneous 입자와 낙하 반발 calibration 테스트."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from wok_sim.config import load_config
from wok_sim.simulation import (
    DEFAULT_FRIED_RICE_SPECIES,
    FriedRiceParticleGenerator,
    ModelBuilder,
    WokSimulator,
    restitution_from_heights,
    simulate_drop_rebound_height,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _generator() -> FriedRiceParticleGenerator:
    return FriedRiceParticleGenerator(
        spawn_radius_m=0.065,
        spawn_height_m=0.012,
        minimum_clearance_m=0.0002,
    )


def test_default_fried_rice_batch_has_three_prescribed_species() -> None:
    generator = _generator()
    batch = generator.generate(0.123, seed=17)
    repeated = generator.generate(seed=17)

    assert batch.count == 60
    np.testing.assert_array_equal(batch.masses_kg, repeated.masses_kg)
    np.testing.assert_array_equal(batch.positions_m, repeated.positions_m)
    np.testing.assert_array_equal(batch.quaternions_wxyz, repeated.quaternions_wxyz)
    assert batch.target_total_mass_kg == batch.actual_total_mass_kg

    large = batch.species == "large_sphere"
    small = batch.species == "small_sphere"
    ellipsoid = batch.species == "ellipsoid"
    assert np.count_nonzero(large) == np.count_nonzero(small) == np.count_nonzero(ellipsoid) == 20
    np.testing.assert_array_equal(batch.geom_types[large], "sphere")
    np.testing.assert_array_equal(batch.geom_types[small], "sphere")
    np.testing.assert_array_equal(batch.geom_types[ellipsoid], "ellipsoid")
    np.testing.assert_allclose(
        batch.sizes_m[large],
        np.tile([0.005, 0.005, 0.005], (20, 1)),
    )
    np.testing.assert_allclose(
        batch.sizes_m[small],
        np.tile([0.003, 0.003, 0.003], (20, 1)),
    )
    np.testing.assert_allclose(
        batch.sizes_m[ellipsoid],
        np.tile([0.005, 0.002, 0.002], (20, 1)),
    )

    assert np.ptp(batch.masses_kg[large]) > 0.0
    assert np.all((0.0018 <= batch.masses_kg[large]) & (batch.masses_kg[large] <= 0.0022))
    np.testing.assert_allclose(batch.masses_kg[small], 0.0008, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(batch.masses_kg[ellipsoid], 0.0002, rtol=0.0, atol=0.0)

    expected_restitution = {
        "large_sphere": restitution_from_heights(0.20, 0.030),
        "small_sphere": restitution_from_heights(0.20, 0.010),
        "ellipsoid": restitution_from_heights(0.20, 0.015),
    }
    for species, expected in expected_restitution.items():
        mask = batch.species == species
        np.testing.assert_allclose(batch.restitution_coefficients[mask], expected)
        np.testing.assert_allclose(batch.reference_drop_heights_m[mask], 0.20)

    differences = batch.positions_m[:, None, :] - batch.positions_m[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    required = batch.radii_m[:, None] + batch.radii_m[None, :] + 0.0002
    off_diagonal = ~np.eye(batch.count, dtype=bool)
    assert np.all(distances[off_diagonal] >= required[off_diagonal])


def test_from_config_accepts_profile_overrides_without_mass_scaling() -> None:
    generator = FriedRiceParticleGenerator.from_config(
        {
            "spawn_radius_m": 0.07,
            "spawn_height_m": 0.01,
            "friction": 0.4,
            "linear_damping": 0.12,
            "angular_damping": 0.08,
            "fried_rice": {
                "count_per_type": 20,
                "large_sphere": {"mass_jitter_fraction": 0.05},
            },
        }
    )
    batch = generator.generate(999.0, seed=3)

    assert batch.count == 60
    assert 0.058 <= batch.actual_total_mass_kg <= 0.062
    assert batch.actual_total_mass_kg != 999.0
    np.testing.assert_allclose(batch.frictions[:, 0], 0.4)
    np.testing.assert_allclose(batch.linear_damping_per_s, 0.12)
    np.testing.assert_allclose(batch.angular_damping_per_s, 0.08)


def test_variable_amount_uses_one_reproducible_shared_species_count() -> None:
    generator = FriedRiceParticleGenerator(
        spawn_radius_m=0.065,
        spawn_height_m=0.012,
        minimum_clearance_m=0.0002,
        count_per_type_range=(20, 40),
    )

    observed_counts: set[int] = set()
    for seed in range(1, 6):
        batch = generator.generate(seed=seed)
        repeated = generator.generate(seed=seed)
        names, counts = np.unique(batch.species, return_counts=True)

        assert set(names) == {"ellipsoid", "large_sphere", "small_sphere"}
        assert len(set(counts)) == 1
        count_per_type = int(counts[0])
        assert 20 <= count_per_type <= 40
        assert batch.count == 3 * count_per_type
        assert 0.056 <= batch.actual_total_mass_kg <= 0.128
        np.testing.assert_array_equal(batch.masses_kg, repeated.masses_kg)
        np.testing.assert_array_equal(batch.positions_m, repeated.positions_m)
        observed_counts.add(count_per_type)

    assert len(observed_counts) > 1
    assert generator.count == 60
    assert generator.count_range == (60, 120)


@pytest.mark.parametrize("count_per_type", (20, 25, 30, 35, 40))
def test_variable_amount_can_be_fixed_for_weight_stratified_evaluation(
    count_per_type: int,
) -> None:
    generator = FriedRiceParticleGenerator(
        spawn_radius_m=0.065,
        spawn_height_m=0.012,
        count_per_type_range=(20, 40),
    )

    batch = generator.generate(seed=7, count_per_type=count_per_type)

    assert batch.count == 3 * count_per_type
    assert set(np.unique(batch.species, return_counts=True)[1]) == {count_per_type}


@pytest.mark.parametrize("invalid_count", (19, 41, 20.5, True))
def test_variable_amount_rejects_count_outside_configured_integer_range(
    invalid_count: object,
) -> None:
    generator = FriedRiceParticleGenerator(
        spawn_radius_m=0.065,
        spawn_height_m=0.012,
        count_per_type_range=(20, 40),
    )

    with pytest.raises(ValueError, match="count_per_type must be an integer"):
        generator.generate(seed=7, count_per_type=invalid_count)


def test_model_builder_compiles_per_particle_geometry_contact_and_metadata() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "test.yaml")
    batch = _generator().generate(seed=9)
    built = ModelBuilder(config).build(batch)

    large_id = int(built.particle_geom_ids[0])
    small_id = int(built.particle_geom_ids[20])
    ellipsoid_id = int(built.particle_geom_ids[40])
    assert int(built.model.geom_type[large_id]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
    assert int(built.model.geom_type[small_id]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
    assert int(built.model.geom_type[ellipsoid_id]) == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)
    np.testing.assert_allclose(built.model.geom_size[large_id], [0.005, 0.0, 0.0])
    np.testing.assert_allclose(built.model.geom_size[small_id], [0.003, 0.0, 0.0])
    np.testing.assert_allclose(
        built.model.geom_size[ellipsoid_id],
        [0.005, 0.002, 0.002],
    )
    assert built.model.geom_priority[large_id] == 1
    assert built.model.geom_priority[small_id] == 1
    assert built.model.geom_priority[ellipsoid_id] == 1
    np.testing.assert_allclose(
        built.model.geom_solref[[large_id, small_id, ellipsoid_id], 1],
        [item.contact_damping_ratio for item in DEFAULT_FRIED_RICE_SPECIES],
    )
    assert built.metadata["particle_species_counts"] == {
        "ellipsoid": 20,
        "large_sphere": 20,
        "small_sphere": 20,
    }
    assert built.metadata["particle_geom_type_counts"] == {
        "ellipsoid": 20,
        "sphere": 40,
    }
    assert built.metadata["particle_rebound_targets"]["large_sphere"][
        "target_rebound_height_m"
    ] == pytest.approx(0.03)


def test_simulator_applies_batch_particle_damping_arrays() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "test.yaml")
    batch = _generator().generate(seed=12)
    simulator = WokSimulator(config, batch)
    try:
        for address in simulator.built.particle_dof_addresses:
            simulator.data.qvel[address : address + 6] = 1.0
        simulator._apply_particle_damping(0.5)
        for index, address in enumerate(simulator.built.particle_dof_addresses):
            np.testing.assert_allclose(
                simulator.data.qvel[address : address + 3],
                np.exp(-batch.linear_damping_per_s[index] * 0.5),
            )
            np.testing.assert_allclose(
                simulator.data.qvel[address + 3 : address + 6],
                np.exp(-batch.angular_damping_per_s[index] * 0.5),
            )
    finally:
        simulator.close()


@pytest.mark.parametrize("specification", DEFAULT_FRIED_RICE_SPECIES)
def test_mujoco_drop_reproduces_target_first_rebound(specification: object) -> None:
    measured_height = simulate_drop_rebound_height(
        geom_type=specification.geom_type,
        sizes_m=specification.semi_axes_m,
        mass_kg=specification.nominal_mass_kg,
        drop_height_m=0.20,
        contact_damping_ratio=specification.contact_damping_ratio,
        contact_time_constant_s=0.006,
        timestep_s=0.002,
    )
    assert measured_height == pytest.approx(
        specification.target_rebound_height_m,
        abs=2.0e-4,
    )
