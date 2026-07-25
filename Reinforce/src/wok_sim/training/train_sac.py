"""Stable-Baselines3 SAC 학습 entry point."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import VectorizedActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from wok_sim.envs import WokMixingEnv

from .random_walk_noise import PersistentEpisodeRandomWalkNoise


@dataclass(frozen=True)
class TrainingResult:
    """학습 checkpoint와 실제 step 수."""

    checkpoint_path: Path
    total_timesteps: int
    seed: int
    evaluation_directory: Path | None = None
    initial_timesteps: int = 0
    additional_timesteps: int = 0
    resume_from: Path | None = None
    resume_replay_buffer_from: Path | None = None


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


class _ScheduledCountWrapper(gym.Wrapper):
    """worker rank를 포함한 global episode 순서로 학습 조건을 주입한다."""

    def __init__(
        self,
        environment: gym.Env,
        schedule: Sequence[int] = (),
        *,
        nominal_joint_speed_target_schedule: Sequence[float] = (),
        rank: int,
        stride: int,
    ) -> None:
        super().__init__(environment)
        self._schedule = tuple(int(item) for item in schedule)
        self._nominal_joint_speed_target_schedule = tuple(
            float(item) for item in nominal_joint_speed_target_schedule
        )
        if not self._schedule and not self._nominal_joint_speed_target_schedule:
            raise ValueError("count 또는 nominal joint speed target schedule이 필요합니다.")
        self._rank = int(rank)
        self._stride = int(stride)
        self._local_episode = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        resolved_options = {} if options is None else dict(options)
        global_episode = self._local_episode * self._stride + self._rank
        if self._schedule and "count_per_type" not in resolved_options:
            resolved_options["count_per_type"] = self._schedule[
                global_episode % len(self._schedule)
            ]
        if (
            self._nominal_joint_speed_target_schedule
            and "nominal_joint_speed_target_fraction" not in resolved_options
        ):
            resolved_options["nominal_joint_speed_target_fraction"] = (
                self._nominal_joint_speed_target_schedule[
                    global_episode % len(self._nominal_joint_speed_target_schedule)
                ]
            )
        resolved_options.setdefault("curriculum_episode", global_episode)
        self._local_episode += 1
        return self.env.reset(seed=seed, options=resolved_options)


def _training_count_schedule(
    training: Mapping[str, Any],
    *,
    steps: int,
    parallel_environments: int,
) -> tuple[int, ...]:
    raw = training.get("count_per_type_schedule")
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes)):
        raise ValueError("training.count_per_type_schedule은 양의 정수 배열이어야 합니다.")
    try:
        original = tuple(raw)
        schedule = tuple(int(item) for item in original)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("training.count_per_type_schedule은 양의 정수 배열이어야 합니다.") from exc
    if not schedule or any(
        isinstance(item, bool) or parsed <= 0 or parsed != item
        for item, parsed in zip(original, schedule, strict=True)
    ):
        raise ValueError("training.count_per_type_schedule은 양의 정수 배열이어야 합니다.")
    episodes_per_count = training.get("episodes_per_count")
    if episodes_per_count is not None:
        requested = int(episodes_per_count)
        if requested <= 0 or requested != episodes_per_count:
            raise ValueError("training.episodes_per_count는 양의 정수여야 합니다.")
        if steps != requested * len(schedule):
            raise ValueError(
                "total_timesteps는 episodes_per_count × count_per_type_schedule 길이여야 합니다."
            )
    # synchronous VecEnv의 완료 episode 순서는 local_episode * stride + rank로
    # 전역 0..steps-1을 정확히 한 번씩 덮는다. 따라서 worker별 episode 수가
    # schedule 길이의 배수일 필요는 없고, 위의 전체 episode 계약만으로 각
    # stratum 횟수가 보장된다(예: 150 episode / 6 workers / 3 strata).
    return schedule


def _training_nominal_joint_speed_target_schedule(
    training: Mapping[str, Any],
    *,
    steps: int,
) -> tuple[float, ...]:
    """M0609 nominal joint speed target strata를 검증한다."""

    raw = training.get("nominal_joint_speed_target_schedule")
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes)):
        raise ValueError(
            "training.nominal_joint_speed_target_schedule은 (0,1] 실수 배열이어야 합니다."
        )
    try:
        schedule = tuple(float(item) for item in raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "training.nominal_joint_speed_target_schedule은 (0,1] 실수 배열이어야 합니다."
        ) from exc
    if (
        not schedule
        or not all(np.isfinite(item) and 0.0 < item <= 1.0 for item in schedule)
        or len(set(schedule)) != len(schedule)
    ):
        raise ValueError(
            "training.nominal_joint_speed_target_schedule은 중복 없는 (0,1] 실수 배열이어야 합니다."
        )
    episodes_per_target = training.get("episodes_per_speed_target")
    if episodes_per_target is None:
        raise ValueError(
            "nominal joint speed schedule에는 training.episodes_per_speed_target이 필요합니다."
        )
    requested = int(episodes_per_target)
    if requested <= 0 or requested != episodes_per_target:
        raise ValueError("training.episodes_per_speed_target은 양의 정수여야 합니다.")
    if steps != requested * len(schedule):
        raise ValueError(
            "total_timesteps는 episodes_per_speed_target × "
            "nominal_joint_speed_target_schedule 길이여야 합니다."
        )
    return schedule


def _balanced_training_condition_schedules(
    training: Mapping[str, Any],
    *,
    steps: int,
    parallel_environments: int,
    seed: int,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """입자 수와 속도 target의 주변 횟수를 보존하며 조합을 균등 배분한다."""

    count_targets = _training_count_schedule(
        training,
        steps=steps,
        parallel_environments=parallel_environments,
    )
    speed_targets = _training_nominal_joint_speed_target_schedule(training, steps=steps)
    if not speed_targets:
        return count_targets, ()
    if not count_targets:
        repeated_speeds = np.repeat(
            np.asarray(speed_targets, dtype=np.float64),
            int(training["episodes_per_speed_target"]),
        )
        rng = np.random.default_rng(int(training.get("condition_schedule_seed", seed + 101)))
        rng.shuffle(repeated_speeds)
        return (), tuple(float(item) for item in repeated_speeds)

    episodes_per_count = training.get("episodes_per_count")
    if episodes_per_count is None:
        raise ValueError(
            "속도/입자 조합 균등 배분에는 training.episodes_per_count가 필요합니다."
        )
    row_target = int(episodes_per_count)
    column_target = int(training["episodes_per_speed_target"])
    count_size = len(count_targets)
    speed_size = len(speed_targets)
    matrix = np.full(
        (count_size, speed_size),
        steps // (count_size * speed_size),
        dtype=np.int64,
    )
    row_remaining = np.full(count_size, row_target, dtype=np.int64) - matrix.sum(axis=1)
    column_remaining = (
        np.full(speed_size, column_target, dtype=np.int64) - matrix.sum(axis=0)
    )
    for speed_index in range(speed_size):
        for offset in range(int(column_remaining[speed_index])):
            candidates = np.flatnonzero(row_remaining > 0)
            if len(candidates) == 0:
                raise ValueError("입자 수/속도 target의 균등 schedule을 구성할 수 없습니다.")
            start = (speed_index + offset) % count_size
            count_index = min(
                candidates,
                key=lambda item: (
                    -int(row_remaining[item]),
                    (int(item) - start) % count_size,
                ),
            )
            matrix[count_index, speed_index] += 1
            row_remaining[count_index] -= 1
    if np.any(row_remaining != 0) or np.any(matrix.sum(axis=0) != column_target):
        raise ValueError("입자 수/속도 target 횟수 계약을 만족하는 schedule이 없습니다.")

    pairs = [
        (int(count_targets[count_index]), float(speed_targets[speed_index]))
        for count_index in range(count_size)
        for speed_index in range(speed_size)
        for _ in range(int(matrix[count_index, speed_index]))
    ]
    rng = np.random.default_rng(int(training.get("condition_schedule_seed", seed + 101)))
    rng.shuffle(pairs)
    return (
        tuple(item[0] for item in pairs),
        tuple(item[1] for item in pairs),
    )


def _make_monitored_environment(
    config: Mapping[str, Any],
    *,
    count_schedule: Sequence[int] = (),
    nominal_joint_speed_target_schedule: Sequence[float] = (),
    rank: int = 0,
    stride: int = 1,
) -> Monitor:
    """SubprocVecEnv의 Windows spawn에서도 직렬화 가능한 환경 factory."""

    environment: gym.Env = WokMixingEnv(config)
    if count_schedule or nominal_joint_speed_target_schedule:
        environment = _ScheduledCountWrapper(
            environment,
            count_schedule,
            nominal_joint_speed_target_schedule=nominal_joint_speed_target_schedule,
            rank=rank,
            stride=stride,
        )
    return Monitor(environment)


def train_sac(
    config: Mapping[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
    resume_from: str | Path | None = None,
    resume_replay_buffer_from: str | Path | None = None,
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
    resolved_resume: Path | None = None
    if resume_from is not None:
        resolved_resume = Path(resume_from).expanduser()
        if not resolved_resume.is_file() and not resolved_resume.suffix:
            resolved_resume = resolved_resume.with_suffix(".zip")
        if not resolved_resume.is_file():
            raise FileNotFoundError(f"재개할 SAC checkpoint를 찾을 수 없습니다: {resume_from}")
    resolved_replay_buffer: Path | None = None
    if resume_replay_buffer_from is not None:
        if resolved_resume is None:
            raise ValueError("resume_replay_buffer_from에는 resume_from도 필요합니다.")
        resolved_replay_buffer = Path(resume_replay_buffer_from).expanduser()
        if not resolved_replay_buffer.is_file():
            raise FileNotFoundError(
                f"재개할 replay buffer를 찾을 수 없습니다: {resume_replay_buffer_from}"
            )

    parallel_environments = int(training.get("parallel_environments", 1))
    if parallel_environments <= 0:
        raise ValueError("training.parallel_environments는 1 이상이어야 합니다.")
    if steps % parallel_environments != 0:
        raise ValueError(
            "total_timesteps는 training.parallel_environments의 배수여야 "
            "정확한 episode 수를 실행할 수 있습니다."
        )
    count_schedule, nominal_joint_speed_target_schedule = (
        _balanced_training_condition_schedules(
        training,
        steps=steps,
        parallel_environments=parallel_environments,
        seed=seed,
        )
    )
    if parallel_environments == 1:
        environment: VecEnv | Monitor = _make_monitored_environment(
            config,
            count_schedule=count_schedule,
            nominal_joint_speed_target_schedule=nominal_joint_speed_target_schedule,
        )
    else:
        environment = SubprocVecEnv(
            [
                lambda rank=rank, config=config, count_schedule=count_schedule,
                speed_schedule=nominal_joint_speed_target_schedule: (
                    _make_monitored_environment(
                        config,
                        count_schedule=count_schedule,
                        nominal_joint_speed_target_schedule=speed_schedule,
                        rank=rank,
                        stride=parallel_environments,
                    )
                )
                for rank in range(parallel_environments)
            ],
            start_method="spawn",
        )
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
            if checkpoint_interval % parallel_environments != 0:
                raise ValueError(
                    "training.checkpoint_interval은 parallel_environments의 배수여야 합니다."
                )
            callbacks.append(
                CheckpointCallback(
                    save_freq=checkpoint_interval // parallel_environments,
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
            if evaluation_interval % parallel_environments != 0:
                raise ValueError(
                    "training.evaluation_interval은 parallel_environments의 배수여야 합니다."
                )
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
                    eval_freq=evaluation_interval // parallel_environments,
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
            noise_seed = int(random_walk_config.get("seed", seed + 17))
            noise_options = {
                "step_std": random_walk_config.get("step_std", 0.06),
                "bound": random_walk_config.get("bound", 0.30),
            }
            if parallel_environments == 1:
                action_noise = PersistentEpisodeRandomWalkNoise(
                    action_shape[0],
                    seed=noise_seed,
                    **noise_options,
                )
            else:
                base_noise = PersistentEpisodeRandomWalkNoise(
                    action_shape[0],
                    seed=noise_seed,
                    **noise_options,
                )
                vectorized_noise = VectorizedActionNoise(base_noise, parallel_environments)
                vectorized_noise.noises = [
                    PersistentEpisodeRandomWalkNoise(
                        action_shape[0],
                        seed=noise_seed + rank,
                        **noise_options,
                    )
                    for rank in range(parallel_environments)
                ]
                action_noise = vectorized_noise
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

        if resolved_resume is None:
            model = SAC(
                "MlpPolicy",
                environment,
                learning_rate=float(training.get("learning_rate", 3e-4)),
                buffer_size=int(training.get("buffer_size", 50_000)),
                learning_starts=int(training.get("learning_starts", 100)),
                batch_size=int(training.get("batch_size", 128)),
                gamma=gamma,
                train_freq=train_frequency,
                gradient_steps=gradient_steps * parallel_environments,
                ent_coef=entropy_coefficient,
                seed=seed,
                verbose=int(training.get("verbose", 1)),
                device=str(training.get("device", "auto")),
                policy_kwargs=dict(training.get("policy_kwargs", {})),
                action_noise=action_noise,
            )
            initial_timesteps = 0
        else:
            model = SAC.load(
                resolved_resume,
                env=environment,
                device=str(training.get("device", "auto")),
                action_noise=action_noise,
            )
            initial_timesteps = int(model.num_timesteps)
            if resolved_replay_buffer is not None:
                model.load_replay_buffer(resolved_replay_buffer)
        callback = CallbackList(callbacks) if callbacks else None
        learn_options: dict[str, Any] = {
            "total_timesteps": steps,
            "progress_bar": progress_bar,
            "callback": callback,
        }
        if resolved_resume is not None:
            learn_options["reset_num_timesteps"] = False
        model.learn(**learn_options)
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
        initial_timesteps,
        steps,
        resolved_resume,
        resolved_replay_buffer,
    )
