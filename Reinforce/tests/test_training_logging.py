from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from typer.testing import CliRunner

from wok_sim.cli import app
from wok_sim.logging import EpisodeLogger
from wok_sim.training import evaluate_policy, run_baseline, train_sac


class _OneStepTrainingEnv(gym.Env[np.ndarray, np.ndarray]):
    """SAC callback/evaluation 배선만 빠르게 검사하는 one-step 환경."""

    observation_space = gym.spaces.Box(
        low=np.array([0.0], dtype=np.float32),
        high=np.array([1.0], dtype=np.float32),
        dtype=np.float32,
    )
    action_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(7,),
        dtype=np.float32,
    )
    instances: list[_OneStepTrainingEnv] = []

    def __init__(self, _config: Any) -> None:
        super().__init__()
        self.instance_id = len(self.instances)
        self.reset_seeds: list[int | None] = []
        self.instances.append(self)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        self.reset_seeds.append(seed)
        return np.array([0.5], dtype=np.float32), {"instance_id": self.instance_id}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert action.shape == (7,)
        return (
            np.array([0.5], dtype=np.float32),
            1.0,
            True,
            False,
            {"instance_id": self.instance_id, "final_reward": 1.0},
        )


def test_train_sac_logs_episodes_and_runs_seeded_periodic_evaluation(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    train_module = importlib.import_module("wok_sim.training.train_sac")
    _OneStepTrainingEnv.instances.clear()
    monkeypatch.setattr(train_module, "WokMixingEnv", _OneStepTrainingEnv)
    consumed: list[tuple[int, int]] = []
    checkpoint = tmp_path / "agent.custom"
    evaluation_directory = tmp_path / "evaluation"
    config = {
        "training": {
            "algorithm": "SAC",
            "seed": 23,
            "learning_rate": 3e-4,
            "buffer_size": 16,
            "learning_starts": 100,
            "batch_size": 2,
            "gamma": 0.0,
            "train_freq": 1,
            "gradient_steps": 4,
            "ent_coef": "auto",
            "checkpoint_interval": 1,
            "checkpoint_save_replay_buffer": True,
            "evaluation_interval": 1,
            "evaluation_episodes": 2,
            "verbose": 0,
            "device": "cpu",
            "policy_kwargs": {"net_arch": [8]},
        }
    }

    result = train_sac(
        config,
        checkpoint_path=checkpoint,
        total_timesteps=2,
        episode_consumer=lambda episode_id, info: consumed.append(
            (episode_id, int(info["instance_id"]))
        ),
        evaluation_directory=evaluation_directory,
    )

    assert result.checkpoint_path == checkpoint
    assert result.checkpoint_path.is_file()
    assert result.total_timesteps == 2
    assert (tmp_path / "agent_intermediate_1_steps.zip").is_file()
    assert (tmp_path / "agent_intermediate_replay_buffer_1_steps.pkl").is_file()
    assert (tmp_path / "agent_intermediate_2_steps.zip").is_file()
    assert (tmp_path / "agent_intermediate_replay_buffer_2_steps.pkl").is_file()
    restored = SAC.load(checkpoint)
    assert restored.gamma == 0.0
    assert restored.gradient_steps == 4
    assert consumed == [(0, 0), (1, 0)]
    assert len(_OneStepTrainingEnv.instances) == 2
    assert _OneStepTrainingEnv.instances[0].reset_seeds[0] == 23
    assert _OneStepTrainingEnv.instances[1].reset_seeds[0] == 24
    assert (evaluation_directory / "best_model.zip").is_file()
    with np.load(evaluation_directory / "evaluations.npz") as evaluations:
        np.testing.assert_array_equal(evaluations["timesteps"], [1, 2])
        assert evaluations["results"].shape == (2, 2)


class _HistoryEnv:
    action_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(7,),
        dtype=np.float32,
    )
    reset_options: list[dict[str, Any]] = []

    def __init__(self, _config: Any, **_kwargs: Any) -> None:
        pass

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        selected_options = {} if options is None else dict(options)
        self.reset_options.append(selected_options)
        return np.array([0.1], dtype=np.float32), {"random_seed": seed}

    def step(
        self,
        _action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        result = {
            "time_s": np.array([0.0, 0.1]),
            "particle_positions_world_m": np.zeros((2, 1, 3)),
            "particle_velocities_world_m_s": np.zeros((2, 1, 3)),
            "particle_positions_pan_m": np.zeros((2, 1, 3)),
            "contact_with_pan": np.ones((2, 1), dtype=bool),
        }
        return (
            np.array([0.1], dtype=np.float32),
            0.75,
            True,
            False,
            {"simulation_result": result, "final_reward": 0.75},
        )

    def close(self) -> None:
        pass


class _PredictableModel:
    loaded_path: Path | None = None

    @classmethod
    def load(cls, path: Path) -> _PredictableModel:
        cls.loaded_path = Path(path)
        return cls()

    def predict(
        self,
        _observation: np.ndarray,
        *,
        deterministic: bool,
    ) -> tuple[np.ndarray, None]:
        assert deterministic
        return np.zeros(7, dtype=np.float32), None


def test_baseline_and_evaluation_optionally_preserve_particle_history(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    baseline_module = importlib.import_module("wok_sim.training.baselines")
    evaluate_module = importlib.import_module("wok_sim.training.evaluate")
    monkeypatch.setattr(baseline_module, "WokMixingEnv", _HistoryEnv)
    monkeypatch.setattr(evaluate_module, "WokMixingEnv", _HistoryEnv)
    monkeypatch.setattr(evaluate_module, "SAC", _PredictableModel)

    compact = run_baseline({}, episodes=1, strategy="center")
    retained = run_baseline(
        {},
        episodes=1,
        strategy="center",
        keep_physics_history=True,
    )
    assert "simulation_result" not in compact[0]
    assert retained[0]["simulation_result"]["time_s"].shape == (2,)

    requested_checkpoint = tmp_path / "policy.snapshot"
    actual_checkpoint = Path(f"{requested_checkpoint}.zip")
    actual_checkpoint.touch()
    _HistoryEnv.reset_options.clear()
    evaluated = evaluate_policy(
        {},
        requested_checkpoint,
        episodes=3,
        masses_kg=np.array([0.1, 0.2]),
        keep_physics_history=True,
    )
    assert _PredictableModel.loaded_path == actual_checkpoint
    assert [item["target_mass_kg"] for item in _HistoryEnv.reset_options] == [
        0.1,
        0.2,
        0.1,
    ]
    assert all("simulation_result" in item for item in evaluated)

    _HistoryEnv.reset_options.clear()
    evaluate_policy(
        {},
        requested_checkpoint,
        episodes=5,
        counts_per_type=[20, 25, 30],
    )
    assert [item["count_per_type"] for item in _HistoryEnv.reset_options] == [
        20,
        25,
        30,
        20,
        25,
    ]


def test_episode_logger_honors_csv_and_npz_flags(tmp_path: Path) -> None:
    particle_history = {
        "time_s": np.array([0.0, 0.1]),
        "position_world_m": np.zeros((2, 1, 3)),
        "contact_with_pan": np.ones((2, 1), dtype=np.uint8),
    }
    without_csv = EpisodeLogger(
        {
            "logging": {
                "output_directory": str(tmp_path),
                "save_csv": False,
                "save_npz": True,
                "save_particle_trajectories": True,
            }
        },
        run_directory=tmp_path / "without_csv",
    )
    without_csv.log_episode(
        {"episode_id": 4, "final_reward": 1.0},
        particle_trajectory=particle_history,
    )
    assert not without_csv.episodes_path.exists()
    assert (without_csv.run_directory / "particles_4.npz").is_file()

    without_npz = EpisodeLogger(
        {
            "logging": {
                "output_directory": str(tmp_path),
                "save_csv": True,
                "save_npz": False,
                "save_particle_trajectories": True,
            }
        },
        run_directory=tmp_path / "without_npz",
    )
    without_npz.log_episode(
        {"episode_id": 5, "final_reward": 2.0},
        particle_trajectory=particle_history,
    )
    assert without_npz.episodes_path.is_file()
    assert not (without_npz.run_directory / "particles_5.npz").exists()


def test_episode_record_serializes_no_flight_nan_as_json_null() -> None:
    cli_module = importlib.import_module("wok_sim.cli")

    record = cli_module._episode_record(
        {"particles": {}},
        {
            "flight": {
                "particle_flight_summary": {
                    "takeoff_time_s": np.array([np.nan, 0.5]),
                }
            }
        },
        episode_id=0,
    )

    encoded = record["particle_flight_summary_json"]
    assert "NaN" not in encoded
    assert json.loads(encoded)["takeoff_time_s"] == [None, 0.5]


def test_baseline_cli_creates_timestamp_run_and_metadata(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    cli_module = importlib.import_module("wok_sim.cli")
    result_root = tmp_path / "results"
    loaded = {
        "particles": {"friction": 0.5, "restitution": 0.1},
        "logging": {
            "output_directory": str(result_root),
            "save_csv": True,
            "save_npz": True,
            "save_particle_trajectories": False,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(cli_module, "_load_config", lambda _path: loaded)
    monkeypatch.setattr(
        cli_module,
        "run_baseline",
        lambda *_args, episodes, **_kwargs: [
            {
                "episode_id": episode_id,
                "random_seed": episode_id,
                "final_reward": float(episode_id),
            }
            for episode_id in range(episodes)
        ],
    )

    outcome = CliRunner().invoke(
        app,
        [
            "baseline",
            "--config",
            str(config_path),
            "--episodes",
            "2",
            "--strategy",
            "center",
        ],
    )

    assert outcome.exit_code == 0, outcome.output
    run_directories = list(result_root.iterdir())
    assert len(run_directories) == 1
    run_directory = run_directories[0]
    assert re.fullmatch(r"\d{8}_\d{6}_\d{6}", run_directory.name)
    assert len(pd.read_csv(run_directory / "episodes.csv")) == 2
    metadata = json.loads((run_directory / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["command"] == "baseline"
    assert metadata["episodes"] == 2
    assert (run_directory / "effective_config.yaml").is_file()
