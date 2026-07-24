"""실행별 metadata, episode summary, 선택적 particle trajectory logger."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wok_sim.config import dump_effective_config


def create_run_directory(root: str | Path) -> Path:
    """충돌 가능성이 낮은 timestamp 기반 결과 디렉터리를 생성한다."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory = Path(root).expanduser() / timestamp
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__}은 JSON으로 직렬화할 수 없습니다.")


def _flatten(
    source: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in source.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten(value, prefix=name))
        elif isinstance(value, (list, tuple, np.ndarray)):
            flattened[name] = json.dumps(value, default=_json_default, ensure_ascii=False)
        elif isinstance(value, np.generic):
            flattened[name] = value.item()
        else:
            flattened[name] = value
    return flattened


class EpisodeLogger:
    """하나의 run directory에 episode 결과를 append한다."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        run_directory: str | Path | None = None,
    ):
        logging_config = config.get("logging", {})
        root = logging_config.get("output_directory", "results")
        self.run_directory = (
            Path(run_directory).expanduser()
            if run_directory is not None
            else create_run_directory(root)
        )
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.episodes_path = self.run_directory / "episodes.csv"
        self._save_csv = bool(logging_config.get("save_csv", True))
        self._save_particles = bool(
            logging_config.get("save_particle_trajectories", False)
            and logging_config.get("save_npz", True)
        )
        dump_effective_config(config, self.run_directory / "effective_config.yaml")

    def log_episode(
        self,
        episode: Mapping[str, Any],
        *,
        particle_trajectory: Mapping[str, np.ndarray] | None = None,
    ) -> Path:
        """summary 한 행과 선택적 particle NPZ를 저장한다."""

        row = _flatten(episode)
        if not row:
            raise ValueError("episode summary가 비어 있습니다.")
        if self._save_csv:
            frame = pd.DataFrame([row])
            frame.to_csv(
                self.episodes_path,
                mode="a",
                header=not self.episodes_path.exists(),
                index=False,
            )
        if self._save_particles and particle_trajectory is not None:
            episode_id = row.get("episode_id", "unknown")
            output = self.run_directory / f"particles_{episode_id}.npz"
            arrays = {key: np.asarray(value) for key, value in particle_trajectory.items()}
            if any(not np.isfinite(value).all() for value in arrays.values()):
                raise ValueError("particle trajectory에 NaN 또는 inf가 있습니다.")
            np.savez_compressed(output, **arrays)
        return self.episodes_path

    def write_metadata(self, metadata: Mapping[str, Any]) -> Path:
        """run-level 검증 범위와 주의사항을 JSON으로 저장한다."""

        output = self.run_directory / "metadata.json"
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        temporary.replace(output)
        return output
