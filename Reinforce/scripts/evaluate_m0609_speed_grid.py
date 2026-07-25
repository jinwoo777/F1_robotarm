"""Deterministically evaluate one policy on the 3 x 6 M0609 speed grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv

from wok_sim.config import load_config
from wok_sim.training.train_sac import _make_monitored_environment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=901)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    config = load_config(arguments.config)
    counts = (20, 30, 40)
    speeds = (0.80, 0.82, 0.84, 0.86, 0.88, 0.90)
    conditions = tuple((count, speed) for count in counts for speed in speeds)
    count_schedule = tuple(item[0] for item in conditions)
    speed_schedule = tuple(item[1] for item in conditions)
    workers = 6
    environment = SubprocVecEnv(
        [
            lambda rank=rank: _make_monitored_environment(
                config,
                count_schedule=count_schedule,
                nominal_joint_speed_target_schedule=speed_schedule,
                rank=rank,
                stride=workers,
            )
            for rank in range(workers)
        ],
        start_method="spawn",
    )
    rows: list[dict[str, Any]] = []
    try:
        environment.seed(arguments.seed)
        observation = environment.reset()
        policy = SAC.load(arguments.checkpoint, env=environment, device="cpu")
        for batch_index in range(len(conditions) // workers):
            action, _state = policy.predict(observation, deterministic=True)
            observation, rewards, dones, infos = environment.step(action)
            if not np.all(dones):
                raise RuntimeError("one-step evaluation environment가 종료되지 않았습니다.")
            for rank, (reward, info) in enumerate(zip(rewards, infos, strict=True)):
                episode_id = batch_index * workers + rank
                expected_count, expected_speed = conditions[episode_id]
                action_parameters = dict(info.get("action_parameters", {}))
                joint_report = dict(info.get("joint_speed_report", {}))
                rows.append(
                    {
                        "episode_id": episode_id,
                        "count_per_type": int(info["count_per_type"]),
                        "nominal_joint_speed_target_fraction": float(
                            info["nominal_joint_speed_target_fraction"]
                        ),
                        "expected_count_per_type": expected_count,
                        "expected_speed_target_fraction": expected_speed,
                        "final_reward": float(reward),
                        "trajectory_valid": bool(info.get("trajectory_valid", False)),
                        "mixing_improvement": float(info.get("mixing_improvement", 0.0)),
                        "spill_count": int(info.get("spill_count", 0)),
                        "spill_count_ratio": float(info.get("spill_count_ratio", 0.0)),
                        "lifted_particle_count": int(info.get("lifted_particle_count", 0)),
                        "descent_angle_rad": float(action_parameters["descent_angle"]),
                        "pan_tilt_angle_rad": float(action_parameters["pan_tilt_angle"]),
                        "descent_speed_m_s": float(action_parameters["descent_speed"]),
                        "lift_angle_rad": float(action_parameters["lift_angle"]),
                        "motion_duration_s": float(
                            info["trajectory_validation"]["metrics"]["duration_s"]
                        ),
                        "limiting_quantity": joint_report.get("limiting_quantity"),
                        "limiting_utilization": joint_report.get("limiting_utilization"),
                    }
                )
    finally:
        environment.close()

    frame = pd.DataFrame(rows)
    if not (
        np.array_equal(frame["count_per_type"], frame["expected_count_per_type"])
        and np.allclose(
            frame["nominal_joint_speed_target_fraction"],
            frame["expected_speed_target_fraction"],
        )
    ):
        raise RuntimeError("평가 condition schedule과 실제 reset condition이 다릅니다.")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(arguments.output, index=False)
    summary = {
        "episodes": len(frame),
        "valid_episodes": int(frame["trajectory_valid"].sum()),
        "mean_reward": float(frame["final_reward"].mean()),
        "reward_std": float(frame["final_reward"].std(ddof=1)),
        "mean_spill_count_ratio": float(frame["spill_count_ratio"].mean()),
        "mean_lifted_particle_count": float(frame["lifted_particle_count"].mean()),
        "by_speed": {
            f"{speed:.2f}": {
                "mean_reward": float(group["final_reward"].mean()),
                "mean_spill_count_ratio": float(group["spill_count_ratio"].mean()),
                "mean_lifted_particle_count": float(
                    group["lifted_particle_count"].mean()
                ),
            }
            for speed, group in frame.groupby("nominal_joint_speed_target_fraction")
        },
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
