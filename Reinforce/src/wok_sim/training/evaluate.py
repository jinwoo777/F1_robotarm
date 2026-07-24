"""학습 checkpoint deterministic 평가."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import SAC

from wok_sim.envs import WokMixingEnv

from .baselines import compact_episode_info


def evaluate_policy(
    config: Mapping[str, Any],
    checkpoint_path: str | Path,
    *,
    episodes: int = 10,
    seed: int = 0,
    masses_kg: Sequence[float] | None = None,
    counts_per_type: Sequence[int] | None = None,
    keep_physics_history: bool = False,
) -> list[dict[str, Any]]:
    """checkpoint를 deterministic action으로 평가한다."""

    if episodes <= 0:
        raise ValueError("episodes는 1 이상이어야 합니다.")
    checkpoint = Path(checkpoint_path).expanduser()
    candidate = (
        checkpoint
        if checkpoint.is_file() or checkpoint.suffix == ".zip"
        else Path(f"{checkpoint}.zip")
    )
    if not candidate.is_file():
        raise FileNotFoundError(f"SAC checkpoint를 찾을 수 없습니다: {checkpoint}")
    if masses_kg is not None and counts_per_type is not None:
        raise ValueError("masses_kg와 counts_per_type schedule은 함께 사용할 수 없습니다.")
    mass_schedule: np.ndarray | None = None
    if masses_kg is not None:
        try:
            mass_schedule = np.asarray(masses_kg, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("masses_kg는 유한한 양수의 1차원 배열이어야 합니다.") from exc
        if mass_schedule.ndim != 1:
            raise ValueError("masses_kg는 유한한 양수의 1차원 배열이어야 합니다.")
        if mass_schedule.size == 0:
            mass_schedule = None
        elif not np.isfinite(mass_schedule).all() or np.any(mass_schedule <= 0.0):
            raise ValueError("masses_kg는 유한한 양수만 포함해야 합니다.")
    count_schedule: np.ndarray | None = None
    if counts_per_type is not None:
        try:
            raw_counts = np.asarray(counts_per_type)
            count_schedule = raw_counts.astype(np.int64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("counts_per_type은 양의 정수 1차원 배열이어야 합니다.") from exc
        if count_schedule.ndim != 1:
            raise ValueError("counts_per_type은 양의 정수 1차원 배열이어야 합니다.")
        if count_schedule.size == 0:
            count_schedule = None
        elif (
            np.any(count_schedule <= 0)
            or raw_counts.dtype == np.bool_
            or not np.equal(raw_counts, count_schedule).all()
        ):
            raise ValueError("counts_per_type은 양의 정수만 포함해야 합니다.")
    model = SAC.load(candidate)
    environment = WokMixingEnv(config)
    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []
    try:
        for episode_id in range(episodes):
            options: dict[str, Any] = {}
            if mass_schedule is not None:
                options["target_mass_kg"] = float(mass_schedule[episode_id % len(mass_schedule)])
            if count_schedule is not None:
                options["count_per_type"] = int(count_schedule[episode_id % len(count_schedule)])
            episode_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            observation, reset_info = environment.reset(seed=episode_seed, options=options)
            action, _ = model.predict(observation, deterministic=True)
            _, reward, terminated, truncated, info = environment.step(action)
            if not terminated or truncated:
                raise RuntimeError("WokMixingEnv는 one-step terminated episode여야 합니다.")
            record = {
                "episode_id": episode_id,
                **compact_episode_info(reset_info),
                **compact_episode_info(info),
                "final_reward": float(reward),
            }
            if keep_physics_history and "simulation_result" in info:
                record["simulation_result"] = info["simulation_result"]
            results.append(record)
    finally:
        environment.close()
    return results
