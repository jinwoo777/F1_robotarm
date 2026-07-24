"""Random/fixed open-loop action baseline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from wok_sim.envs import WokMixingEnv
from wok_sim.exploration import BoundedEpisodeRandomWalk


def compact_episode_info(info: Mapping[str, Any]) -> dict[str, Any]:
    """큰 physics history를 제외하고 episode summary만 복사한다."""

    return {
        key: value
        for key, value in info.items()
        if key not in {"simulation_result", "trajectory", "particle_batch"}
    }


def run_baseline(
    config: Mapping[str, Any],
    *,
    episodes: int,
    seed: int = 0,
    strategy: str = "random",
    target_mass_kg: float | None = None,
    keep_physics_history: bool = False,
) -> list[dict[str, Any]]:
    """random, random-walk 또는 중앙 action baseline을 실행한다."""

    if episodes <= 0:
        raise ValueError("episodes는 1 이상이어야 합니다.")
    normalized_strategy = strategy.strip().lower().replace("-", "_")
    if normalized_strategy not in {"random", "random_walk", "center"}:
        raise ValueError("strategy는 'random', 'random_walk' 또는 'center'여야 합니다.")
    rng = np.random.default_rng(seed)
    environment = WokMixingEnv(config, target_mass_kg=target_mass_kg)
    walk: BoundedEpisodeRandomWalk | None = None
    if normalized_strategy == "random_walk":
        random_walk_config = config.get("training", {}).get("random_walk", {})
        walk = BoundedEpisodeRandomWalk(
            dimension=environment.action_space.shape[0],
            step_std=random_walk_config.get("step_std", 0.08),
            seed=int(random_walk_config.get("seed", seed)),
        )
    results: list[dict[str, Any]] = []
    try:
        for episode_id in range(episodes):
            episode_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            _, reset_info = environment.reset(seed=episode_seed)
            if normalized_strategy == "random":
                action = rng.uniform(-1.0, 1.0, size=environment.action_space.shape)
            elif walk is not None:
                action = walk.current_action
            else:
                action = np.zeros(environment.action_space.shape, dtype=float)
            _, reward, terminated, truncated, info = environment.step(action.astype(np.float32))
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
            if walk is not None:
                walk.advance_episode()
    finally:
        environment.close()
    return results
