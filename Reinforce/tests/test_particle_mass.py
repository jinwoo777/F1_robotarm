"""입자 질량 scaling, seed 재현성, 비겹침 검증."""

from __future__ import annotations

import numpy as np

from wok_sim.simulation.particle_generator import ParticleGenerator


def _generator() -> ParticleGenerator:
    return ParticleGenerator(
        count=18,
        density_kg_m3=920.0,
        nominal_radius_m=0.007,
        radius_jitter_fraction=0.15,
        spawn_radius_m=0.080,
        spawn_height_m=0.004,
        mass_tolerance_kg=1e-12,
        minimum_clearance_m=0.0002,
    )


def test_target_total_mass_is_matched_by_global_radius_scaling() -> None:
    generator = _generator()
    light = generator.generate(0.050, seed=19)
    heavy = generator.generate(0.090, seed=19)

    assert light.count == heavy.count == generator.count
    assert np.isclose(light.actual_total_mass_kg, 0.050, rtol=0.0, atol=1e-12)
    assert np.isclose(heavy.actual_total_mass_kg, 0.090, rtol=0.0, atol=1e-12)

    expected_scale_ratio = np.cbrt(0.090 / 0.050)
    np.testing.assert_allclose(
        heavy.radii_m / light.radii_m,
        expected_scale_ratio,
        rtol=2e-15,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        heavy.masses_kg,
        generator.density_kg_m3 * (4.0 * np.pi / 3.0) * heavy.radii_m**3,
        rtol=2e-15,
        atol=0.0,
    )


def test_seed_reproduces_radii_and_non_overlapping_positions() -> None:
    generator = _generator()
    first = generator.generate(0.060, seed=7)
    repeated = generator.generate(0.060, seed=7)
    different = generator.generate(0.060, seed=8)

    np.testing.assert_array_equal(first.radii_m, repeated.radii_m)
    np.testing.assert_array_equal(first.positions_m, repeated.positions_m)
    assert not np.array_equal(first.radii_m, different.radii_m)
    assert not np.array_equal(first.positions_m, different.positions_m)

    difference = first.positions_m[:, None, :] - first.positions_m[None, :, :]
    distance = np.linalg.norm(difference, axis=2)
    required = first.radii_m[:, None] + first.radii_m[None, :] + generator.minimum_clearance_m
    off_diagonal = ~np.eye(first.count, dtype=bool)
    assert np.all(distance[off_diagonal] >= required[off_diagonal])

    center_radius = np.linalg.norm(first.positions_m[:, :2], axis=1)
    assert np.all(
        center_radius + first.radii_m + 0.5 * generator.minimum_clearance_m
        <= generator.spawn_radius_m + 1e-15
    )
    assert np.all(first.positions_m[:, 2] >= generator.spawn_height_m + first.radii_m)


def test_dense_target_uses_vertical_layers_without_overlap() -> None:
    """기본 demo의 최대 질량처럼 한 평면에 못 들어가는 경우도 생성한다."""

    generator = ParticleGenerator(
        count=32,
        density_kg_m3=900.0,
        nominal_radius_m=0.010,
        radius_jitter_fraction=0.12,
        spawn_radius_m=0.065,
        spawn_height_m=0.012,
        mass_tolerance_kg=1e-9,
        minimum_clearance_m=0.0005,
        max_spawn_attempts=10_000,
    )
    particles = generator.generate(0.200, seed=7)
    difference = particles.positions_m[:, None, :] - particles.positions_m[None, :, :]
    distance = np.linalg.norm(difference, axis=2)
    required = (
        particles.radii_m[:, None] + particles.radii_m[None, :] + generator.minimum_clearance_m
    )
    off_diagonal = ~np.eye(particles.count, dtype=bool)

    assert np.isclose(particles.actual_total_mass_kg, 0.200, atol=1e-9, rtol=0.0)
    assert np.all(distance[off_diagonal] >= required[off_diagonal])
    assert np.ptp(particles.positions_m[:, 2]) > 2.0 * np.max(particles.radii_m)
