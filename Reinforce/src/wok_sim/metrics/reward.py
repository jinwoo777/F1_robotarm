"""볶음밥 episode의 lift 신호와 reward 항을 계산한다."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def recovery_return_sample_mask(
    time_s: np.ndarray,
    *,
    cycle_time_s: float,
    recovery_start_s: float,
    motion_end_s: float,
) -> np.ndarray:
    """각 cycle에서 하강 완료 뒤 복원·복귀 구간만 선택한다."""

    time = np.asarray(time_s, dtype=float)
    cycle_time = float(cycle_time_s)
    recovery_start = float(recovery_start_s)
    motion_end = float(motion_end_s)
    if time.ndim != 1 or len(time) == 0 or not np.isfinite(time).all():
        raise ValueError("time_s는 비어 있지 않은 유한한 1차원 배열이어야 합니다.")
    if np.any(np.diff(time) < 0.0):
        raise ValueError("time_s는 단조 증가해야 합니다.")
    if (
        not np.isfinite([cycle_time, recovery_start, motion_end]).all()
        or cycle_time <= 0.0
        or recovery_start < 0.0
        or recovery_start >= cycle_time
        or motion_end <= 0.0
    ):
        raise ValueError("cycle/recovery/motion 시간이 유효하지 않습니다.")

    tolerance = max(1.0e-9, cycle_time * 1.0e-9)
    clipped = np.minimum(time, motion_end)
    phase_time = np.remainder(clipped, cycle_time)
    at_cycle_end = (clipped > 0.0) & np.isclose(
        phase_time,
        0.0,
        rtol=0.0,
        atol=tolerance,
    )
    phase_time = np.where(at_cycle_end, cycle_time, phase_time)
    return (time <= motion_end + tolerance) & (phase_time >= recovery_start - tolerance)


def compute_recovery_return_lift_metrics(
    time_s: np.ndarray,
    particle_positions_pan_m: np.ndarray,
    contact_with_pan: np.ndarray,
    radii_m: np.ndarray,
    spilled_mask: np.ndarray,
    *,
    rim_z_m: float,
    cycle_time_s: float,
    recovery_start_s: float,
    motion_end_s: float,
    top_margin_m: float,
    approach_band_m: float = 0.0,
) -> dict[str, Any]:
    """복원·복귀 중 윗부분이 pan rim을 넘은 뒤 보존된 grain을 센다.

    grain 윗부분(center z + radius)이 ``top_margin_m``보다 더 높고 pan과
    비접촉인 sample을 인정한다. 같은 grain은 episode당 한 번만 세며, 유출된
    grain은 제외해 밖으로 던져 bonus를 얻는 것을 막는다.
    """

    time = np.asarray(time_s, dtype=float)
    positions = np.asarray(particle_positions_pan_m, dtype=float)
    contacts = np.asarray(contact_with_pan, dtype=bool)
    radii = np.asarray(radii_m, dtype=float)
    spilled = np.asarray(spilled_mask, dtype=bool)
    rim_z = float(rim_z_m)
    top_margin = float(top_margin_m)
    approach_band = float(approach_band_m)
    if positions.ndim != 3 or positions.shape[0] != len(time) or positions.shape[2] != 3:
        raise ValueError("particle_positions_pan_m shape은 (T, N, 3)이어야 합니다.")
    if contacts.shape != positions.shape[:2]:
        raise ValueError("contact_with_pan shape은 (T, N)이어야 합니다.")
    if radii.shape != (positions.shape[1],):
        raise ValueError("radii_m shape은 particle 개수와 같아야 합니다.")
    if spilled.shape != (positions.shape[1],):
        raise ValueError("spilled_mask shape은 particle 개수와 같아야 합니다.")
    if (
        not np.isfinite(positions).all()
        or not np.isfinite(radii).all()
        or np.any(radii <= 0.0)
        or not np.isfinite(rim_z)
    ):
        raise ValueError("particle 위치·반경과 rim_z_m이 유효하지 않습니다.")
    if not np.isfinite(top_margin) or top_margin < 0.0:
        raise ValueError("top_margin_m은 유한한 0 이상 값이어야 합니다.")
    if not np.isfinite(approach_band) or approach_band < 0.0:
        raise ValueError("approach_band_m은 유한한 0 이상 값이어야 합니다.")

    evaluation_mask = recovery_return_sample_mask(
        time,
        cycle_time_s=cycle_time_s,
        recovery_start_s=recovery_start_s,
        motion_end_s=motion_end_s,
    )
    if not np.any(evaluation_mask):
        raise ValueError("복원·복귀 구간에 해당하는 simulation sample이 없습니다.")

    top_clearance = positions[evaluation_mask, :, 2] + radii[np.newaxis, :] - rim_z
    airborne = ~contacts[evaluation_mask]
    retained = ~spilled
    maximum_any_clearance = np.max(top_clearance, axis=0)
    if approach_band > 0.0:
        approach_fraction = np.clip(
            (maximum_any_clearance + approach_band) / approach_band,
            0.0,
            1.0,
        )
        approach_fraction = np.where(retained, approach_fraction, 0.0)
    else:
        approach_fraction = np.zeros_like(maximum_any_clearance)
    eligible_clearance = np.where(airborne, top_clearance, -np.inf)
    maximum_clearance = np.max(eligible_clearance, axis=0)
    comparison_threshold = top_margin + max(1.0e-12, abs(rim_z) * 1.0e-12)
    lifted = (maximum_clearance > comparison_threshold) & retained
    lifted_by_sample = airborne & (top_clearance > comparison_threshold) & retained[np.newaxis, :]
    lifted_ratios = np.mean(lifted_by_sample, axis=1)
    peak_index = int(np.argmax(lifted_ratios))
    evaluation_times = time[evaluation_mask]

    return {
        "evaluation_phase": "post_descent_recovery_and_return",
        "evaluation_sample_count": int(np.count_nonzero(evaluation_mask)),
        "rim_z_m": rim_z,
        "top_margin_m": top_margin,
        "approach_band_m": approach_band,
        "retained_particle_count": int(np.count_nonzero(retained)),
        "lift_approach_particle_equivalents": float(np.sum(approach_fraction)),
        "mean_lift_approach_fraction": float(np.mean(approach_fraction)),
        "lifted_particle_count": int(np.count_nonzero(lifted)),
        "lifted_particle_ratio": float(np.mean(lifted)),
        "peak_lifted_particle_ratio": float(lifted_ratios[peak_index]),
        "peak_lifted_time_s": float(evaluation_times[peak_index]),
        "mean_lifted_top_clearance_m": (
            float(np.mean(maximum_clearance[lifted])) if np.any(lifted) else 0.0
        ),
        "maximum_grain_top_clearance_m": (
            float(np.max(maximum_clearance[np.isfinite(maximum_clearance)]))
            if np.any(np.isfinite(maximum_clearance))
            else 0.0
        ),
        # 기존 분석 코드와의 호환을 위해 binary unique-particle ratio를 남긴다.
        "lift_score": float(np.mean(lifted)),
    }


def compute_reward_terms(
    *,
    mixing_improvement: float,
    lifted_particle_count: int,
    spill_mass_ratio: float,
    spill_count_ratio: float,
    jerk_cost: float,
    acceleration_cost: float,
    height_penalty: float,
    reward_config: Mapping[str, Any],
    lift_approach_particle_equivalents: float = 0.0,
    lift_approach_multiplier: float = 0.0,
) -> tuple[dict[str, float], dict[str, float | str]]:
    """초기 대비 mixing, retained lift, 비선형 spill 감점을 합성한다."""

    mix_delta = float(mixing_improvement)
    lift_count = float(lifted_particle_count)
    mass_ratio = float(spill_mass_ratio)
    count_ratio = float(spill_count_ratio)
    approach_equivalents = float(lift_approach_particle_equivalents)
    approach_multiplier = float(lift_approach_multiplier)
    if not np.isfinite(
        [
            mix_delta,
            lift_count,
            mass_ratio,
            count_ratio,
            jerk_cost,
            acceleration_cost,
            height_penalty,
            approach_equivalents,
            approach_multiplier,
        ]
    ).all():
        raise ValueError("reward 입력은 모두 유한해야 합니다.")
    if lift_count < 0.0 or not lift_count.is_integer():
        raise ValueError("lifted_particle_count는 0 이상의 정수여야 합니다.")
    if approach_equivalents < 0.0 or approach_multiplier < 0.0:
        raise ValueError("lift 접근 점수와 curriculum multiplier는 0 이상이어야 합니다.")
    if not 0.0 <= mass_ratio <= 1.0 or not 0.0 <= count_ratio <= 1.0:
        raise ValueError("spill ratio는 [0, 1] 범위여야 합니다.")

    mixing_mode = str(reward_config.get("mixing_reward_mode", "improvement")).strip().lower()
    if mixing_mode != "improvement":
        raise ValueError("mixing_reward_mode은 'improvement'만 지원합니다.")

    severity_mode = str(reward_config.get("spill_severity_mode", "mass")).strip().lower()
    if severity_mode == "mass":
        spill_severity = mass_ratio
    elif severity_mode == "count":
        spill_severity = count_ratio
    elif severity_mode == "mean_count_mass":
        spill_severity = 0.5 * (mass_ratio + count_ratio)
    elif severity_mode == "max_count_mass":
        spill_severity = max(mass_ratio, count_ratio)
    else:
        raise ValueError(f"지원하지 않는 spill_severity_mode입니다: {severity_mode!r}")

    spill_linear = float(reward_config.get("w_spill", 1.0)) * spill_severity
    spill_quadratic = float(reward_config.get("w_spill_quadratic", 0.0)) * spill_severity**2
    lift_reward_per_particle = float(reward_config.get("lift_reward_per_particle", 0.0))
    approach_reward_per_particle = float(
        reward_config.get("lift_approach_reward_per_particle", 0.0)
    )
    if not np.isfinite(lift_reward_per_particle) or lift_reward_per_particle < 0.0:
        raise ValueError("lift_reward_per_particle은 유한한 0 이상 값이어야 합니다.")
    if not np.isfinite(approach_reward_per_particle) or approach_reward_per_particle < 0.0:
        raise ValueError(
            "lift_approach_reward_per_particle은 유한한 0 이상 값이어야 합니다."
        )
    terms = {
        "mix": float(reward_config.get("w_mix", 1.0)) * mix_delta,
        "lift_approach": (
            approach_reward_per_particle * approach_multiplier * approach_equivalents
        ),
        "lift": lift_reward_per_particle * lift_count,
        "spill": -(spill_linear + spill_quadratic),
        "jerk": -float(reward_config.get("w_jerk", 0.0)) * float(jerk_cost),
        "acceleration": -float(reward_config.get("w_acc", 0.0)) * float(acceleration_cost),
        "height": -float(height_penalty),
        "invalid": 0.0,
    }
    signals: dict[str, float | str] = {
        "mixing_reward_mode": mixing_mode,
        "mixing_delta": mix_delta,
        "lifted_particle_count": lift_count,
        "lift_reward_per_particle": lift_reward_per_particle,
        "lift_approach_particle_equivalents": approach_equivalents,
        "lift_approach_multiplier": approach_multiplier,
        "lift_approach_reward_per_particle": approach_reward_per_particle,
        "spill_mass_ratio": mass_ratio,
        "spill_count_ratio": count_ratio,
        "spill_severity": spill_severity,
        "spill_severity_mode": severity_mode,
        "spill_linear_penalty": spill_linear,
        "spill_quadratic_penalty": spill_quadratic,
    }
    return terms, signals


__all__ = [
    "compute_recovery_return_lift_metrics",
    "compute_reward_terms",
    "recovery_return_sample_mask",
]
