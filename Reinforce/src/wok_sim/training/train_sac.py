"""Stable-Baselines3 SAC 학습 entry point."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from wok_sim.envs import WokMixingEnv

from .random_walk_noise import PersistentEpisodeRandomWalkNoise


@dataclass(frozen=True)
class TrainingResult:
    """학습 checkpoint와 실제 step 수."""

    checkpoint_path: Path
    total_timesteps: int
    seed: int
    evaluation_directory: Path | None = None


class _EpisodeInfoCallback(BaseCallback):
    """완료된 one-step episode의 info를 외부 logger로 전달한다."""

    def __init__(
        self,
        consumer: Callable[[int, Mapping[str, Any]], None],
    ) -> None:
        super().__init__(verbose=0)
        self._consumer = consumer
        self._episode_id = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", ())
        infos = self.locals.get("infos", ())
        for done, info in zip(dones, infos, strict=True):
            if bool(done):
                self._consumer(self._episode_id, info)
                self._episode_id += 1
        return True


def train_sac(
    config: Mapping[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
    total_timesteps: int | None = None,
    progress_bar: bool = False,
    episode_consumer: Callable[[int, Mapping[str, Any]], None] | None = None,
    evaluation_directory: str | Path | None = None,
) -> TrainingResult:
    """질량 context→궤적 action SAC를 학습하고 checkpoint를 저장한다."""

    training = config.get("training", {})
    algorithm = str(training.get("algorithm", "SAC")).upper()
    if algorithm != "SAC":
        raise ValueError(f"이 구현이 지원하는 training.algorithm은 SAC입니다: {algorithm}")
    steps = int(
        training.get("total_timesteps", 10_000) if total_timesteps is None else total_timesteps
    )
    if steps <= 0:
        raise ValueError("total_timesteps는 1 이상이어야 합니다.")
    seed = int(training.get("seed", 0))
    if checkpoint_path is None:
        checkpoint_path = Path(str(training.get("checkpoint_directory", "checkpoints"))) / "sac_wok"
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    environment = Monitor(WokMixingEnv(config))
    evaluation_environment: VecEnv | None = None
    resolved_evaluation_directory: Path | None = None
    actual_timesteps = 0
    try:
        callbacks: list[BaseCallback] = []
        if episode_consumer is not None:
            callbacks.append(_EpisodeInfoCallback(episode_consumer))

        checkpoint_interval = int(training.get("checkpoint_interval", 0))
        if checkpoint_interval < 0:
            raise ValueError("training.checkpoint_interval은 0 이상이어야 합니다.")
        if checkpoint_interval > 0:
            callbacks.append(
                CheckpointCallback(
                    save_freq=checkpoint_interval,
                    save_path=str(checkpoint.parent),
                    name_prefix=f"{checkpoint.stem}_intermediate",
                    save_replay_buffer=bool(training.get("checkpoint_save_replay_buffer", False)),
                    save_vecnormalize=False,
                    verbose=0,
                )
            )

        evaluation_interval = int(training.get("evaluation_interval", 0))
        if evaluation_interval < 0:
            raise ValueError("training.evaluation_interval은 0 이상이어야 합니다.")
        if evaluation_interval > 0:
            evaluation_episodes = int(training.get("evaluation_episodes", 5))
            if evaluation_episodes <= 0:
                raise ValueError("training.evaluation_episodes는 1 이상이어야 합니다.")
            resolved_evaluation_directory = (
                Path(evaluation_directory)
                if evaluation_directory is not None
                else checkpoint.parent / "evaluation"
            )
            resolved_evaluation_directory.mkdir(parents=True, exist_ok=True)
            evaluation_environment = DummyVecEnv([lambda: Monitor(WokMixingEnv(config))])
            # EvalCallback은 별도 env를 자동으로 seed하지 않는다. 첫 reset에
            # training seed에서 결정되는 독립 seed를 예약해 평가 질량과 입자
            # 초기화까지 실행 간 재현되게 한다.
            evaluation_environment.seed(seed + 1)
            callbacks.append(
                EvalCallback(
                    evaluation_environment,
                    best_model_save_path=str(resolved_evaluation_directory),
                    log_path=str(resolved_evaluation_directory),
                    eval_freq=evaluation_interval,
                    n_eval_episodes=evaluation_episodes,
                    deterministic=True,
                    render=False,
                )
            )

        random_walk_config = training.get("random_walk", {})
        action_noise = None
        if bool(random_walk_config.get("enabled", False)):
            action_shape = environment.action_space.shape
            if action_shape is None or len(action_shape) != 1:
                raise ValueError("random-walk 탐색에는 1차원 연속 action space가 필요합니다.")
            action_noise = PersistentEpisodeRandomWalkNoise(
                action_shape[0],
                step_std=random_walk_config.get("step_std", 0.06),
                bound=random_walk_config.get("bound", 0.30),
                seed=int(random_walk_config.get("seed", seed + 17)),
            )

        gamma = float(training.get("gamma", 0.99))
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("training.gamma는 0과 1 사이여야 합니다.")
        train_frequency = int(training.get("train_freq", 1))
        gradient_steps = int(training.get("gradient_steps", 1))
        if train_frequency <= 0:
            raise ValueError("training.train_freq는 1 이상이어야 합니다.")
        if gradient_steps <= 0:
            raise ValueError("training.gradient_steps는 1 이상이어야 합니다.")
        entropy_coefficient: str | float = training.get("ent_coef", "auto")
        if not isinstance(entropy_coefficient, str):
            entropy_coefficient = float(entropy_coefficient)

        model = SAC(
            "MlpPolicy",
            environment,
            learning_rate=float(training.get("learning_rate", 3e-4)),
            buffer_size=int(training.get("buffer_size", 50_000)),
            learning_starts=int(training.get("learning_starts", 100)),
            batch_size=int(training.get("batch_size", 128)),
            gamma=gamma,
            train_freq=train_frequency,
            gradient_steps=gradient_steps,
            ent_coef=entropy_coefficient,
            seed=seed,
            verbose=int(training.get("verbose", 1)),
            device=str(training.get("device", "auto")),
            policy_kwargs=dict(training.get("policy_kwargs", {})),
            action_noise=action_noise,
        )
        callback = CallbackList(callbacks) if callbacks else None
        model.learn(
            total_timesteps=steps,
            progress_bar=progress_bar,
            callback=callback,
        )
        actual_timesteps = int(model.num_timesteps)
        model.save(checkpoint)
    finally:
        environment.close()
        if evaluation_environment is not None:
            evaluation_environment.close()
    # SB3는 suffix가 전혀 없을 때만 ".zip"을 덧붙인다. 예를 들어
    # "agent.custom"은 ZIP 컨테이너이지만 파일명은 그대로 보존된다.
    actual_path = checkpoint if checkpoint.suffix else Path(f"{checkpoint}.zip")
    return TrainingResult(
        actual_path,
        actual_timesteps,
        seed,
        resolved_evaluation_directory,
    )
