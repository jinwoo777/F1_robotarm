"""Evaluate final and log-selected pitch-release SAC checkpoints on fixed seeds."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from wok_sim.config import load_config
from wok_sim.envs import WokMixingEnv

HELD_OUT_SEEDS = (71_011, 79_009, 83_003)
COUNTS_PER_TYPE = (20, 30, 40)


def _checkpoint_step(path: Path) -> int | None:
    match = re.search(r"_(\d+)_steps$", path.stem)
    return None if match is None else int(match.group(1))


def select_log_checkpoint(
    episodes_csv: str | Path,
    checkpoint_directory: str | Path,
) -> Path | None:
    """episode 600 이후 60-episode block이 가장 좋은 실제 checkpoint를 고른다."""

    frame = pd.read_csv(episodes_csv)
    required = {
        "episode_id",
        "peak_lifted_particle_ratio",
        "spill_count_ratio",
        "final_reward",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"episodes.csv 필수 열이 없습니다: {sorted(missing)}")
    candidates: list[tuple[tuple[float, ...], Path]] = []
    for path in Path(checkpoint_directory).glob("*_steps.zip"):
        step = _checkpoint_step(path)
        if step is None or step < 600:
            continue
        block = frame[
            (frame["episode_id"] >= step - 60)
            & (frame["episode_id"] < step)
        ]
        if len(block) < 30:
            continue
        success = (
            (block["peak_lifted_particle_ratio"] >= 0.20)
            & (block["spill_count_ratio"] <= 0.05)
            & block["trajectory_valid"].astype(bool)
        )
        score = (
            float(success.mean()),
            float(block["peak_lifted_particle_ratio"].median()),
            -float(block["spill_count_ratio"].quantile(0.90)),
            float(block["final_reward"].mean()),
        )
        candidates.append((score, path))
    return None if not candidates else max(candidates, key=lambda item: item[0])[1]


def evaluate_checkpoint_episode(
    config_path: str | Path,
    checkpoint: str | Path,
    *,
    count_per_type: int,
    seed: int,
) -> dict[str, Any]:
    config = load_config(config_path)
    model = SAC.load(checkpoint)
    environment = WokMixingEnv(config)
    try:
        observation, _ = environment.reset(
            seed=int(seed),
            options={
                "count_per_type": int(count_per_type),
                "curriculum_episode": 1499,
                "nominal_joint_speed_target_fraction": 0.90,
            },
        )
        action, _ = model.predict(observation, deterministic=True)
        _, reward, terminated, truncated, info = environment.step(action)
        if not terminated or truncated:
            raise RuntimeError("deterministic 평가 episode가 정상 종료되지 않았습니다.")
    finally:
        environment.close()
    particle_count = int(info["particle_count"])
    peak_ratio = float(info.get("peak_lifted_particle_ratio", 0.0))
    peak_count = int(round(peak_ratio * particle_count))
    spill_count = int(info.get("spill_count", particle_count))
    required_peak = int(np.ceil(0.20 * particle_count))
    maximum_spill = int(np.floor(0.05 * particle_count))
    severe_spill = int(np.floor(0.10 * particle_count))
    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "count_per_type": int(count_per_type),
        "particle_count": particle_count,
        "random_seed": int(seed),
        "trajectory_valid": bool(info.get("trajectory_valid", False)),
        "peak_lifted_particle_count": peak_count,
        "peak_lifted_particle_ratio": peak_ratio,
        "lifted_particle_count": int(info.get("lifted_particle_count", 0)),
        "spill_count": spill_count,
        "spill_count_ratio": float(info.get("spill_count_ratio", 1.0)),
        "mixing_improvement": float(info.get("mixing_improvement", 0.0)),
        "final_reward": float(reward),
        "required_peak_count": required_peak,
        "maximum_spill_count": maximum_spill,
        "success": (
            bool(info.get("trajectory_valid", False))
            and peak_count >= required_peak
            and spill_count <= maximum_spill
        ),
        "severe_spill": spill_count > severe_spill,
        "normalized_action": np.asarray(info["normalized_action"]).tolist(),
        "action_parameters": dict(info.get("action_parameters", {})),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    flattened = [
        {
            key: (
                json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list, tuple))
                else value
            )
            for key, value in row.items()
        }
        for row in rows
    ]
    fieldnames = list(
        dict.fromkeys(key for row in flattened for key in row)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)


def _checkpoint_summary(
    checkpoint: Path,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected = [
        row for row in rows if Path(str(row["checkpoint"])) == checkpoint.resolve()
    ]
    successes = sum(bool(row["success"]) for row in selected)
    successes_120 = sum(
        bool(row["success"]) for row in selected if int(row["particle_count"]) == 120
    )
    promoted = (
        len(selected) == 9
        and successes >= 6
        and successes_120 >= 2
        and not any(bool(row["severe_spill"]) for row in selected)
        and all(bool(row["trajectory_valid"]) for row in selected)
    )
    return {
        "checkpoint": str(checkpoint.resolve()),
        "evaluation_episode_count": len(selected),
        "success_count": successes,
        "success_count_120": successes_120,
        "median_peak_lifted_particle_ratio": float(
            np.median([float(row["peak_lifted_particle_ratio"]) for row in selected])
        ),
        "spill_count_ratio_p90": float(
            np.quantile(
                [float(row["spill_count_ratio"]) for row in selected],
                0.90,
            )
        ),
        "severe_spill_count": sum(bool(row["severe_spill"]) for row in selected),
        "all_trajectories_valid": all(
            bool(row["trajectory_valid"]) for row in selected
        ),
        "promoted": promoted,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.workers <= 0:
        raise ValueError("workers는 양수여야 합니다.")
    best_log_checkpoint = select_log_checkpoint(
        arguments.run_directory / "episodes.csv",
        arguments.checkpoint_directory,
    )
    checkpoints = [arguments.final_checkpoint]
    if (
        best_log_checkpoint is not None
        and best_log_checkpoint.resolve() != arguments.final_checkpoint.resolve()
    ):
        checkpoints.append(best_log_checkpoint)
    jobs = [
        (checkpoint, count_per_type, seed)
        for checkpoint in checkpoints
        for count_per_type in COUNTS_PER_TYPE
        for seed in HELD_OUT_SEEDS
    ]
    rows: list[dict[str, Any]] = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(arguments.workers, len(jobs)),
        mp_context=context,
    ) as executor:
        futures = {
            executor.submit(
                evaluate_checkpoint_episode,
                arguments.config,
                checkpoint,
                count_per_type=count_per_type,
                seed=seed,
            ): (checkpoint, count_per_type, seed)
            for checkpoint, count_per_type, seed in jobs
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                json.dumps(
                    {
                        "completed": len(rows),
                        "checkpoint": Path(row["checkpoint"]).name,
                        "particle_count": row["particle_count"],
                        "seed": row["random_seed"],
                        "peak": row["peak_lifted_particle_count"],
                        "spill": row["spill_count"],
                        "success": row["success"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    rows.sort(
        key=lambda row: (
            str(row["checkpoint"]),
            int(row["particle_count"]),
            int(row["random_seed"]),
        )
    )
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(arguments.output_directory / "episodes.csv", rows)
    checkpoint_summaries = [
        _checkpoint_summary(checkpoint, rows) for checkpoint in checkpoints
    ]
    promoted = [item for item in checkpoint_summaries if bool(item["promoted"])]
    best = max(
        checkpoint_summaries,
        key=lambda item: (
            float(bool(item["promoted"])),
            float(item["success_count"]),
            float(item["success_count_120"]),
            float(item["median_peak_lifted_particle_ratio"]),
            -float(item["spill_count_ratio_p90"]),
        ),
    )
    summary = {
        "config": str(arguments.config.resolve()),
        "run_directory": str(arguments.run_directory.resolve()),
        "held_out_seeds": list(HELD_OUT_SEEDS),
        "counts_per_type": list(COUNTS_PER_TYPE),
        "checkpoints": checkpoint_summaries,
        "promoted_checkpoint": (
            str(best["checkpoint"]) if promoted else None
        ),
        "best_experimental_checkpoint": (
            None if promoted else str(best["checkpoint"])
        ),
        "promotion_gate_pass": bool(promoted),
    }
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if promoted else 2


if __name__ == "__main__":
    raise SystemExit(main())
