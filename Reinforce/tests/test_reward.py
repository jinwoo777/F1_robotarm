from __future__ import annotations

import numpy as np
import pytest

from wok_sim.metrics.reward import (
    compute_recovery_return_lift_metrics,
    compute_reward_terms,
    recovery_return_sample_mask,
)


def test_recovery_return_mask_selects_post_descent_part_of_every_cycle() -> None:
    time_s = np.arange(0.0, 8.1, 1.0)

    mask = recovery_return_sample_mask(
        time_s,
        cycle_time_s=4.0,
        recovery_start_s=2.0,
        motion_end_s=8.0,
    )

    np.testing.assert_allclose(time_s[mask], [2.0, 3.0, 4.0, 6.0, 7.0, 8.0])


def test_lift_requires_grain_top_to_strictly_exceed_margin() -> None:
    time_s = np.arange(5.0)
    positions = np.zeros((5, 1, 3), dtype=float)
    contacts = np.ones((5, 1), dtype=bool)
    radii = np.full(1, 0.005)
    rim_z_m = 0.05

    # top == rim + 1 mm은 strict threshold를 넘지 않으므로 보상하지 않는다.
    positions[:, 0, 2] = 0.03
    positions[2:, 0, 2] = rim_z_m + 0.001 - radii[0]
    contacts[2:, 0] = False

    metrics = compute_recovery_return_lift_metrics(
        time_s,
        positions,
        contacts,
        radii,
        np.asarray([False]),
        rim_z_m=rim_z_m,
        cycle_time_s=4.0,
        recovery_start_s=2.0,
        motion_end_s=4.0,
        top_margin_m=0.001,
    )

    assert metrics["lifted_particle_count"] == 0

    positions[3, 0, 2] += 1.0e-6
    metrics = compute_recovery_return_lift_metrics(
        time_s,
        positions,
        contacts,
        radii,
        np.asarray([False]),
        rim_z_m=rim_z_m,
        cycle_time_s=4.0,
        recovery_start_s=2.0,
        motion_end_s=4.0,
        top_margin_m=0.001,
    )

    assert metrics["lifted_particle_count"] == 1
    assert metrics["lifted_particle_ratio"] == pytest.approx(1.0)
    assert metrics["maximum_grain_top_clearance_m"] == pytest.approx(0.001001)


def test_lift_counts_each_retained_airborne_grain_once_in_recovery_return() -> None:
    time_s = np.arange(9.0)
    positions = np.zeros((9, 4, 3), dtype=float)
    contacts = np.ones((9, 4), dtype=bool)
    radii = np.full(4, 0.005)
    rim_z_m = 0.05
    positions[:, :, 2] = 0.03

    # 0: 여러 sample/cycle에서 통과하지만 한 번만 센다.
    positions[[2, 3, 6], 0, 2] = [0.052, 0.053, 0.052]
    contacts[[2, 3, 6], 0] = False
    # 1: 통과 후 최종 유출, 2: 높지만 pan 접촉 중, 3: recovery 이전만 통과.
    positions[3, 1, 2] = 0.055
    contacts[3, 1] = False
    positions[3, 2, 2] = 0.055
    positions[1, 3, 2] = 0.055
    contacts[1, 3] = False

    metrics = compute_recovery_return_lift_metrics(
        time_s,
        positions,
        contacts,
        radii,
        np.asarray([False, True, False, False]),
        rim_z_m=rim_z_m,
        cycle_time_s=4.0,
        recovery_start_s=2.0,
        motion_end_s=8.0,
        top_margin_m=0.001,
    )

    assert metrics["lifted_particle_count"] == 1
    assert metrics["lifted_particle_ratio"] == pytest.approx(0.25)
    assert metrics["lift_score"] == pytest.approx(0.25)
    assert metrics["peak_lifted_particle_ratio"] == pytest.approx(0.25)


def test_lift_approach_scores_retained_grains_fractionally_below_rim() -> None:
    time_s = np.arange(5.0)
    positions = np.zeros((5, 3, 3), dtype=float)
    positions[:, :, 2] = 0.005
    contacts = np.ones((5, 3), dtype=bool)
    radii = np.full(3, 0.005)
    rim_z_m = 0.055
    # recovery 중 grain top이 rim 아래 20 mm, rim, rim까지 가지만
    # 마지막 grain은 spill 처리되어 접근 보상을 받지 못한다.
    positions[3, 0, 2] = rim_z_m - 0.020 - radii[0]
    positions[3, 1:, 2] = rim_z_m - radii[1]

    metrics = compute_recovery_return_lift_metrics(
        time_s,
        positions,
        contacts,
        radii,
        np.asarray([False, False, True]),
        rim_z_m=rim_z_m,
        cycle_time_s=4.0,
        recovery_start_s=2.0,
        motion_end_s=4.0,
        top_margin_m=0.001,
        approach_band_m=0.040,
    )

    assert metrics["lifted_particle_count"] == 0
    assert metrics["lift_approach_particle_equivalents"] == pytest.approx(1.5)
    assert metrics["mean_lift_approach_fraction"] == pytest.approx(0.5)


def test_reward_uses_mixing_delta_lift_bonus_and_convex_spill_penalty() -> None:
    config = {
        "mixing_reward_mode": "improvement",
        "w_mix": 3.0,
        "lift_reward_per_particle": 0.3,
        "w_spill": 12.0,
        "w_spill_quadratic": 40.0,
        "spill_severity_mode": "max_count_mass",
    }

    no_spill, signals = compute_reward_terms(
        mixing_improvement=0.2,
        lifted_particle_count=2,
        spill_mass_ratio=0.0,
        spill_count_ratio=0.0,
        jerk_cost=0.0,
        acceleration_cost=0.0,
        height_penalty=0.0,
        reward_config=config,
    )
    ten_percent, _ = compute_reward_terms(
        mixing_improvement=0.2,
        lifted_particle_count=2,
        spill_mass_ratio=0.05,
        spill_count_ratio=0.1,
        jerk_cost=0.0,
        acceleration_cost=0.0,
        height_penalty=0.0,
        reward_config=config,
    )
    twenty_percent, _ = compute_reward_terms(
        mixing_improvement=0.2,
        lifted_particle_count=2,
        spill_mass_ratio=0.2,
        spill_count_ratio=0.1,
        jerk_cost=0.0,
        acceleration_cost=0.0,
        height_penalty=0.0,
        reward_config=config,
    )

    assert no_spill["mix"] == pytest.approx(0.6)
    assert no_spill["lift"] == pytest.approx(0.6)
    assert no_spill["spill"] == 0.0
    assert signals["mixing_delta"] == pytest.approx(0.2)
    assert signals["lifted_particle_count"] == pytest.approx(2.0)
    assert signals["lift_reward_per_particle"] == pytest.approx(0.3)
    assert ten_percent["spill"] == pytest.approx(-1.6)
    assert twenty_percent["spill"] == pytest.approx(-4.0)
    assert abs(twenty_percent["spill"] - ten_percent["spill"]) > abs(ten_percent["spill"])


def test_reward_applies_curriculum_multiplier_only_to_approach_term() -> None:
    terms, signals = compute_reward_terms(
        mixing_improvement=0.2,
        lifted_particle_count=2,
        spill_mass_ratio=0.0,
        spill_count_ratio=0.0,
        jerk_cost=0.0,
        acceleration_cost=0.0,
        height_penalty=0.0,
        reward_config={
            "w_mix": 3.0,
            "lift_reward_per_particle": 0.1,
            "lift_approach_reward_per_particle": 0.1,
        },
        lift_approach_particle_equivalents=3.5,
        lift_approach_multiplier=0.5,
    )

    assert terms["mix"] == pytest.approx(0.6)
    assert terms["lift"] == pytest.approx(0.2)
    assert terms["lift_approach"] == pytest.approx(0.175)
    assert signals["lift_approach_particle_equivalents"] == pytest.approx(3.5)
    assert signals["lift_approach_multiplier"] == pytest.approx(0.5)
