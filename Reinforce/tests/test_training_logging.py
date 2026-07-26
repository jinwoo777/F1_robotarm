from __future__ import annotations

import importlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
from stable_baselines3 import SAC
from stable_baselines3.common.noise import VectorizedActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from typer.testing import CliRunner

from wok_sim.cli import app
from wok_sim.logging import EpisodeLogger
from wok_sim.training import (
    PersistentEpisodeRandomWalkNoise,
    evaluate_policy,
    run_baseline,
    train_sac,
)


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
        shape=(3,),
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
        assert action.shape == (3,)
        return (
            np.array([0.5], dtype=np.float32),
            1.0,
            True,
            False,
            {"instance_id": self.instance_id, "final_reward": 1.0},
        )


class _CountScheduleProbeEnv(gym.Env[np.ndarray, np.ndarray]):
    observation_space = gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        selected = {} if options is None else dict(options)
        info: dict[str, Any] = {
            "count_per_type": int(selected["count_per_type"]),
            "curriculum_episode": int(selected["curriculum_episode"]),
        }
        if "nominal_joint_speed_target_fraction" in selected:
            info["nominal_joint_speed_target_fraction"] = float(
                selected["nominal_joint_speed_target_fraction"]
            )
        return np.zeros(1, dtype=np.float32), info

    def step(
        self,
        _action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return np.zeros(1, dtype=np.float32), 0.0, True, False, {}


def test_six_worker_count_schedule_runs_each_particle_stratum_exactly_50_times() -> None:
    train_module = importlib.import_module("wok_sim.training.train_sac")
    schedule = train_module._training_count_schedule(
        {
            "count_per_type_schedule": [20, 30, 40],
            "episodes_per_count": 50,
        },
        steps=150,
        parallel_environments=6,
    )
    workers = [
        train_module._ScheduledCountWrapper(
            _CountScheduleProbeEnv(),
            schedule,
            rank=rank,
            stride=6,
        )
        for rank in range(6)
    ]

    try:
        counts = [
            int(workers[rank].reset()[1]["count_per_type"])
            for local_episode in range(25)
            for rank in range(6)
        ]
    finally:
        for worker in workers:
            worker.close()

    assert counts == [20, 30, 40] * 50
    assert Counter(counts) == {20: 50, 30: 50, 40: 50}


def test_speed_and_particle_schedules_are_exact_and_cross_balanced() -> None:
    train_module = importlib.import_module("wok_sim.training.train_sac")
    counts, speeds = train_module._balanced_training_condition_schedules(
        {
            "count_per_type_schedule": [20, 30, 40],
            "episodes_per_count": 50,
            "nominal_joint_speed_target_schedule": [0.80, 0.82, 0.84, 0.86, 0.88, 0.90],
            "episodes_per_speed_target": 25,
            "condition_schedule_seed": 151,
        },
        steps=150,
        parallel_environments=6,
        seed=51,
    )

    assert Counter(counts) == {20: 50, 30: 50, 40: 50}
    assert Counter(speeds) == {
        0.80: 25,
        0.82: 25,
        0.84: 25,
        0.86: 25,
        0.88: 25,
        0.90: 25,
    }
    cross_counts = Counter(zip(counts, speeds, strict=True))
    assert set(cross_counts.values()) == {8, 9}

    workers = [
        train_module._ScheduledCountWrapper(
            _CountScheduleProbeEnv(),
            counts,
            nominal_joint_speed_target_schedule=speeds,
            rank=rank,
            stride=6,
        )
        for rank in range(6)
    ]
    try:
        observed = [
            workers[rank].reset()[1]
            for _local_episode in range(25)
            for rank in range(6)
        ]
    finally:
        for worker in workers:
            worker.close()
    assert Counter(item["count_per_type"] for item in observed) == {20: 50, 30: 50, 40: 50}
    assert Counter(item["nominal_joint_speed_target_fraction"] for item in observed) == {
        0.80: 25,
        0.82: 25,
        0.84: 25,
        0.86: 25,
        0.88: 25,
        0.90: 25,
    }


def test_scheduled_conditions_continue_from_episode_offset() -> None:
    train_module = importlib.import_module("wok_sim.training.train_sac")
    workers = [
        train_module._ScheduledCountWrapper(
            _CountScheduleProbeEnv(),
            [20, 30, 40],
            nominal_joint_speed_target_schedule=[0.90, 0.90, 0.90],
            rank=rank,
            stride=2,
            episode_offset=300,
        )
        for rank in range(2)
    ]
    try:
        observed = [
            workers[rank].reset()[1]
            for _local_episode in range(2)
            for rank in range(2)
        ]
    finally:
        for worker in workers:
            worker.close()

    assert [item["curriculum_episode"] for item in observed] == [300, 301, 302, 303]
    assert [item["count_per_type"] for item in observed] == [20, 30, 40, 20]
    assert all(
        item["nominal_joint_speed_target_fraction"] == pytest.approx(0.90)
        for item in observed
    )


def test_parallel_training_builds_independent_reproducible_worker_random_walks(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    train_module = importlib.import_module("wok_sim.training.train_sac")
    captured: dict[str, Any] = {}

    class _CapturingSAC:
        def __init__(self, _policy: str, _environment: Any, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.num_timesteps = 0

        def learn(
            self,
            *,
            total_timesteps: int,
            progress_bar: bool,
            callback: Any,
        ) -> _CapturingSAC:
            del progress_bar, callback
            self.num_timesteps = total_timesteps
            return self

        def save(self, _path: Path) -> None:
            return None

    def _dummy_subproc(env_fns: list[Any], *, start_method: str) -> DummyVecEnv:
        assert start_method == "spawn"
        return DummyVecEnv(env_fns)

    monkeypatch.setattr(train_module, "WokMixingEnv", _OneStepTrainingEnv)
    monkeypatch.setattr(train_module, "SubprocVecEnv", _dummy_subproc)
    monkeypatch.setattr(train_module, "SAC", _CapturingSAC)
    train_sac(
        {
            "training": {
                "algorithm": "SAC",
                "total_timesteps": 150,
                "parallel_environments": 5,
                "count_per_type_schedule": [20, 30, 40],
                "episodes_per_count": 50,
                "checkpoint_interval": 0,
                "evaluation_interval": 0,
                "gradient_steps": 1,
                "device": "cpu",
                "random_walk": {
                    "enabled": True,
                    "step_std": [0.05, 0.05, 0.03],
                    "bound": [0.25, 0.25, 0.20],
                    "seed": 34,
                },
            }
        },
        checkpoint_path=tmp_path / "noise_policy",
    )

    noise = captured["action_noise"]
    assert isinstance(noise, VectorizedActionNoise)
    assert len(noise.noises) == 5
    actual = noise()
    expected = np.stack(
        [
            PersistentEpisodeRandomWalkNoise(
                3,
                step_std=[0.05, 0.05, 0.03],
                bound=[0.25, 0.25, 0.20],
                seed=34 + rank,
            )()
            for rank in range(5)
        ]
    )
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (5, 3)
    assert len({row.tobytes() for row in actual}) == 5
    assert np.all(np.abs(actual) <= np.asarray([0.25, 0.25, 0.20]))
    assert captured["gradient_steps"] == 5


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


def test_train_sac_resumes_checkpoint_and_replay_buffer_as_additional_timesteps(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    train_module = importlib.import_module("wok_sim.training.train_sac")
    _OneStepTrainingEnv.instances.clear()
    monkeypatch.setattr(train_module, "WokMixingEnv", _OneStepTrainingEnv)
    base_training = {
        "algorithm": "SAC",
        "seed": 23,
        "learning_rate": 3e-4,
        "buffer_size": 16,
        "learning_starts": 100,
        "batch_size": 2,
        "gamma": 0.0,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto",
        "evaluation_interval": 0,
        "verbose": 0,
        "device": "cpu",
        "policy_kwargs": {"net_arch": [8]},
    }
    initial_checkpoint = tmp_path / "initial_agent"
    train_sac(
        {
            "training": {
                **base_training,
                "checkpoint_interval": 2,
                "checkpoint_save_replay_buffer": True,
            }
        },
        checkpoint_path=initial_checkpoint,
        total_timesteps=2,
    )
    resume_checkpoint = tmp_path / "initial_agent_intermediate_2_steps.zip"
    resume_replay_buffer = (
        tmp_path / "initial_agent_intermediate_replay_buffer_2_steps.pkl"
    )
    consumed_ids: list[int] = []

    result = train_sac(
        {
            "training": {
                **base_training,
                "checkpoint_interval": 0,
                "episode_offset": 2,
            }
        },
        checkpoint_path=tmp_path / "resumed_agent",
        resume_from=resume_checkpoint,
        resume_replay_buffer_from=resume_replay_buffer,
        total_timesteps=2,
        episode_consumer=lambda episode_id, _info: consumed_ids.append(episode_id),
    )

    assert result.initial_timesteps == 2
    assert result.additional_timesteps == 2
    assert result.total_timesteps == 4
    assert result.resume_from == resume_checkpoint
    assert result.resume_replay_buffer_from == resume_replay_buffer
    resumed = SAC.load(result.checkpoint_path)
    assert resumed.num_timesteps == 4
    assert consumed_ids == [2, 3]


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


def test_episode_record_preserves_per_particle_lift_reward_fields() -> None:
    cli_module = importlib.import_module("wok_sim.cli")

    record = cli_module._episode_record(
        {"particles": {}},
        {
            "lift": {
                "top_margin_m": 0.001,
                "lift_score": 0.1,
                "lifted_particle_count": 6,
                "lifted_particle_ratio": 0.1,
                "peak_lifted_particle_ratio": 0.05,
                "maximum_grain_top_clearance_m": 0.012,
            },
            "reward_signals": {"lift_reward_per_particle": 0.05},
            "reward_terms": {"lift": 0.3},
        },
        episode_id=3,
    )

    assert record["lift_top_margin_m"] == pytest.approx(0.001)
    assert record["lifted_particle_count"] == 6
    assert record["lift_reward_per_particle"] == pytest.approx(0.05)
    assert record["maximum_grain_top_clearance_m"] == pytest.approx(0.012)
    assert record["lift_reward"] == pytest.approx(0.3)


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
