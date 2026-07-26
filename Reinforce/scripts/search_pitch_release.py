"""Screen 6D pitch-release trajectories and confirm the physical training gate."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from wok_sim.config import load_config
from wok_sim.envs import WokMixingEnv
from wok_sim.trajectory import generate_fried_rice_trajectory

SCREEN_PEAK_COUNT = 15
SCREEN_SPILL_COUNT = 18
CONFIRMATION_SEEDS = (17_011, 29_009, 43_003)


@dataclass(frozen=True, slots=True)
class PitchReleaseCandidate:
    candidate_id: int
    base_source: str
    physical_action: tuple[float, float, float, float, float, float]
    normalized_action: tuple[float, float, float, float, float, float]


def _profile(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config["trajectory"]["fried_rice"]


def _normalize(
    physical_action: Sequence[float],
    config: Mapping[str, Any],
) -> tuple[float, float, float, float, float, float]:
    keys = (
        "descent_angle_range_rad",
        "pan_tilt_angle_range_rad",
        "descent_speed_range_m_s",
        "pitch_release_angle_range_rad",
        "pitch_release_phase_fraction_range",
        "pitch_release_angular_acceleration_range_rad_s2",
    )
    values: list[float] = []
    for value, key in zip(physical_action, keys, strict=True):
        low, high = (float(item) for item in _profile(config)[key])
        if high == low:
            values.append(0.0)
        else:
            values.append(2.0 * (float(value) - low) / (high - low) - 1.0)
    normalized = np.asarray(values, dtype=float)
    if np.any(normalized < -1.0 - 1.0e-9) or np.any(normalized > 1.0 + 1.0e-9):
        raise ValueError(f"physical action이 설정 범위를 벗어납니다: {physical_action!r}")
    return tuple(float(item) for item in np.clip(normalized, -1.0, 1.0))


def build_candidates(
    config: Mapping[str, Any],
    four_d_summary: Mapping[str, Any],
) -> list[PitchReleaseCandidate]:
    """2 bases x 4 angles x 3 timings x 3 accelerations = 72."""

    bases = (
        ("best_any", four_d_summary["best_any"]),
        ("best_safe", four_d_summary["best_safe"]),
    )
    candidates: list[PitchReleaseCandidate] = []
    for base_source, row in bases:
        base = (
            float(row["descent_angle_rad"]),
            float(row["pan_tilt_angle_rad"]),
            float(row["descent_speed_m_s"]),
        )
        for angle_deg in (3.0, 6.0, 9.0, 12.0):
            for phase_fraction in (0.35, 0.50, 0.65):
                for acceleration in (0.45, 0.70, 0.95):
                    physical = (
                        *base,
                        float(np.deg2rad(angle_deg)),
                        phase_fraction,
                        acceleration,
                    )
                    candidates.append(
                        PitchReleaseCandidate(
                            candidate_id=len(candidates),
                            base_source=base_source,
                            physical_action=physical,
                            normalized_action=_normalize(physical, config),
                        )
                    )
    if len(candidates) != 72:
        raise RuntimeError("pitch release screen은 정확히 72개 후보여야 합니다.")
    return candidates


def evaluate_offline_candidate(
    config_path: str | Path,
    candidate: PitchReleaseCandidate,
) -> dict[str, Any]:
    config = load_config(config_path)
    base = {
        "candidate_id": candidate.candidate_id,
        "base_source": candidate.base_source,
        "release_angle_deg": float(np.rad2deg(candidate.physical_action[3])),
        "release_phase_fraction": candidate.physical_action[4],
        "requested_angular_acceleration_rad_s2": candidate.physical_action[5],
    }
    try:
        trajectory = generate_fried_rice_trajectory(
            np.asarray(candidate.normalized_action, dtype=np.float32),
            config,
            validate=True,
        )
    except Exception as exc:
        return {
            **base,
            "trajectory_valid": False,
            "simulation_budget_pass": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    validation = trajectory.validation
    assert validation is not None
    simulation = config["simulation"]
    required_steps = int(
        np.ceil(
            (
                trajectory.duration_s
                + float(simulation.get("post_rollout_settle_s", 0.0))
            )
            / float(simulation["timestep_s"])
        )
    )
    parameters = trajectory.parameters
    release_velocity = (
        1.875
        * float(parameters.pitch_release_angle)
        / float(parameters.pitch_release_half_duration_s)
    )
    return {
        **base,
        "trajectory_valid": bool(validation.valid),
        "simulation_budget_pass": required_steps <= int(simulation["max_steps"]),
        "error": "",
        "violations": " | ".join(validation.violations),
        "duration_s": float(trajectory.duration_s),
        "required_physics_steps": required_steps,
        "adaptive_retiming_scale": float(parameters.adaptive_retiming_scale),
        "effective_angular_acceleration_rad_s2": float(
            parameters.pitch_release_effective_angular_acceleration
        ),
        "peak_release_angular_velocity_rad_s": release_velocity,
        "maximum_angular_velocity_rad_s": float(
            validation.metrics["max_angular_velocity"]
        ),
        "maximum_angular_acceleration_rad_s2": float(
            validation.metrics["max_angular_acceleration"]
        ),
        "maximum_angular_jerk_rad_s3": float(
            validation.metrics["max_angular_jerk"]
        ),
    }


def select_candidates(
    candidates: Sequence[PitchReleaseCandidate],
    offline_rows: Sequence[Mapping[str, Any]],
    *,
    per_base: int = 6,
) -> list[PitchReleaseCandidate]:
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    selected: list[PitchReleaseCandidate] = []
    for base_source in ("best_any", "best_safe"):
        valid = [
            row
            for row in offline_rows
            if row["base_source"] == base_source
            and bool(row.get("trajectory_valid"))
            and bool(row.get("simulation_budget_pass"))
        ]
        ranked = sorted(
            valid,
            key=lambda row: (
                float(row["peak_release_angular_velocity_rad_s"]),
                float(row["effective_angular_acceleration_rad_s2"]),
                -float(row["duration_s"]),
            ),
            reverse=True,
        )
        selected.extend(
            by_id[int(row["candidate_id"])] for row in ranked[:per_base]
        )
    return selected


def evaluate_physics_candidate(
    config_path: str | Path,
    candidate: PitchReleaseCandidate,
    *,
    seed: int,
    count_per_type: int,
) -> dict[str, Any]:
    config = load_config(config_path)
    environment = WokMixingEnv(config)
    try:
        environment.reset(
            seed=int(seed),
            options={
                "count_per_type": int(count_per_type),
                "curriculum_episode": 1499,
                "nominal_joint_speed_target_fraction": 0.90,
            },
        )
        _, reward, terminated, truncated, info = environment.step(
            np.asarray(candidate.normalized_action, dtype=np.float32)
        )
        if not terminated or truncated:
            raise RuntimeError("one-step physics episode가 정상 종료되지 않았습니다.")
    finally:
        environment.close()
    particle_count = int(info["particle_count"])
    peak_ratio = float(info.get("peak_lifted_particle_ratio", 0.0))
    spill_count = int(info.get("spill_count", particle_count))
    peak_count = int(round(peak_ratio * particle_count))
    return {
        "candidate_id": candidate.candidate_id,
        "base_source": candidate.base_source,
        "random_seed": int(seed),
        "count_per_type": int(count_per_type),
        "particle_count": particle_count,
        "normalized_action": list(candidate.normalized_action),
        "physical_action": list(candidate.physical_action),
        "trajectory_valid": bool(info.get("trajectory_valid", False)),
        "invalid_reasons": " | ".join(info.get("invalid_reasons", ())),
        "final_reward": float(reward),
        "mixing_improvement": float(info.get("mixing_improvement", 0.0)),
        "peak_lifted_particle_count": peak_count,
        "peak_lifted_particle_ratio": peak_ratio,
        "lifted_particle_count": int(info.get("lifted_particle_count", 0)),
        "spill_count": spill_count,
        "spill_count_ratio": float(info.get("spill_count_ratio", 1.0)),
        "screen_pass": (
            bool(info.get("trajectory_valid", False))
            and peak_count >= SCREEN_PEAK_COUNT
            and spill_count <= SCREEN_SPILL_COUNT
        ),
        "action_parameters": dict(info.get("action_parameters", {})),
    }


def _parallel_evaluate(
    config_path: Path,
    jobs: Sequence[tuple[PitchReleaseCandidate, int]],
    *,
    count_per_type: int,
    workers: int,
) -> list[dict[str, Any]]:
    if not jobs:
        return []
    rows: list[dict[str, Any]] = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(workers, len(jobs)),
        mp_context=context,
    ) as executor:
        futures = {
            executor.submit(
                evaluate_physics_candidate,
                str(config_path),
                candidate,
                seed=seed,
                count_per_type=count_per_type,
            ): (candidate, seed)
            for candidate, seed in jobs
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                json.dumps(
                    {
                        "completed": len(rows),
                        "candidate_id": row["candidate_id"],
                        "seed": row["random_seed"],
                        "peak": row["peak_lifted_particle_count"],
                        "spill": row["spill_count"],
                        "pass": row["screen_pass"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return rows


def _parallel_offline(
    config_path: Path,
    candidates: Sequence[PitchReleaseCandidate],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(workers, len(candidates)),
        mp_context=context,
    ) as executor:
        futures = {
            executor.submit(
                evaluate_offline_candidate,
                str(config_path),
                candidate,
            ): candidate
            for candidate in candidates
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            if len(rows) % 6 == 0 or len(rows) == len(candidates):
                print(
                    json.dumps(
                        {
                            "offline_completed": len(rows),
                            "offline_total": len(candidates),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    rows.sort(key=lambda row: int(row["candidate_id"]))
    return rows


def _rank_physics(row: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(bool(row["screen_pass"])),
        float(row["peak_lifted_particle_count"]),
        -float(row["spill_count"]),
        float(row["mixing_improvement"]),
        float(row["final_reward"]),
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    flattened = []
    for row in rows:
        flattened.append(
            {
                key: (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
                for key, value in row.items()
            }
        )
    fieldnames = list(
        dict.fromkeys(key for row in flattened for key in row)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)


def _read_offline_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as stream:
        for raw in csv.DictReader(stream):
            row: dict[str, Any] = dict(raw)
            row["candidate_id"] = int(raw["candidate_id"])
            for key in ("trajectory_valid", "simulation_budget_pass"):
                row[key] = raw[key].strip().lower() == "true"
            for key in (
                "peak_release_angular_velocity_rad_s",
                "effective_angular_acceleration_rad_s2",
                "duration_s",
            ):
                row[key] = float(raw[key]) if raw.get(key) else np.nan
            rows.append(row)
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--four-d-summary", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--selection-seed", type=int, default=64_860_847)
    parser.add_argument("--count-per-type", type=int, default=40)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--reuse-offline-csv",
        type=Path,
        help="이미 계산한 72-candidate offline CSV를 재사용",
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.workers <= 0:
        raise ValueError("workers는 양수여야 합니다.")
    config = load_config(arguments.config)
    four_d_summary = json.loads(
        arguments.four_d_summary.read_text(encoding="utf-8")
    )
    candidates = build_candidates(config, four_d_summary)
    offline_rows = (
        _read_offline_csv(arguments.reuse_offline_csv)
        if arguments.reuse_offline_csv is not None
        else _parallel_offline(
            arguments.config,
            candidates,
            workers=arguments.workers,
        )
    )
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(arguments.output_directory / "offline_candidates.csv", offline_rows)
    selected = select_candidates(candidates, offline_rows)
    selection_rows = _parallel_evaluate(
        arguments.config,
        [(candidate, arguments.selection_seed) for candidate in selected],
        count_per_type=arguments.count_per_type,
        workers=arguments.workers,
    )
    selection_rows.sort(key=lambda row: int(row["candidate_id"]))
    _write_csv(arguments.output_directory / "selection.csv", selection_rows)

    screened = sorted(
        (row for row in selection_rows if bool(row["screen_pass"])),
        key=_rank_physics,
        reverse=True,
    )
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    confirmation_candidates = [
        by_id[int(row["candidate_id"])] for row in screened[:2]
    ]
    confirmation_rows = _parallel_evaluate(
        arguments.config,
        [
            (candidate, seed)
            for candidate in confirmation_candidates
            for seed in CONFIRMATION_SEEDS
        ],
        count_per_type=arguments.count_per_type,
        workers=arguments.workers,
    ) if confirmation_candidates else []
    confirmation_rows.sort(
        key=lambda row: (int(row["candidate_id"]), int(row["random_seed"]))
    )
    _write_csv(arguments.output_directory / "confirmation.csv", confirmation_rows)
    confirmation_pass_counts = {
        candidate.candidate_id: sum(
            bool(row["screen_pass"])
            for row in confirmation_rows
            if int(row["candidate_id"]) == candidate.candidate_id
        )
        for candidate in confirmation_candidates
    }
    promoted_ids = [
        candidate_id
        for candidate_id, pass_count in confirmation_pass_counts.items()
        if pass_count >= 2
    ]
    summary = {
        "config": str(arguments.config.resolve()),
        "four_d_summary": str(arguments.four_d_summary.resolve()),
        "selection_seed": arguments.selection_seed,
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "count_per_type": arguments.count_per_type,
        "offline_candidate_count": len(candidates),
        "offline_trajectory_valid_count": sum(
            bool(row.get("trajectory_valid")) for row in offline_rows
        ),
        "offline_simulation_budget_valid_count": sum(
            bool(row.get("trajectory_valid"))
            and bool(row.get("simulation_budget_pass"))
            for row in offline_rows
        ),
        "selected_candidate_ids": [
            candidate.candidate_id for candidate in selected
        ],
        "selected_candidate_count": len(selected),
        "selection_screen_pass_count": len(screened),
        "confirmation_pass_counts": confirmation_pass_counts,
        "promoted_candidate_ids": promoted_ids,
        "training_gate_pass": bool(promoted_ids),
        "screen_gate": {
            "minimum_peak_lifted_particle_count": SCREEN_PEAK_COUNT,
            "maximum_spill_count": SCREEN_SPILL_COUNT,
            "confirmation_required_passes": 2,
            "confirmation_episode_count": 3,
        },
        "best_selection": (
            max(selection_rows, key=_rank_physics) if selection_rows else None
        ),
        "candidates": [asdict(candidate) for candidate in candidates],
    }
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary["training_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
