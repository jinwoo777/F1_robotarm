"""Gymnasium one-step contextual RL environment."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from wok_sim.geometry.transforms import compose_transform
from wok_sim.metrics.mixing import assign_initial_labels, compute_mixing_metrics
from wok_sim.metrics.reward import (
    compute_recovery_return_lift_metrics,
    compute_reward_terms,
)
from wok_sim.metrics.spill import evaluate_spill
from wok_sim.metrics.trajectory_cost import compute_trajectory_costs
from wok_sim.simulation.pan_model import PanModel
from wok_sim.simulation.particle_generator import ParticleBatch, ParticleGenerator
from wok_sim.simulation.simulator import (
    InvalidTrajectoryError,
    SimulationResult,
    WokSimulator,
)
from wok_sim.trajectory import action_size_for_config, generate_configured_trajectory


class EpisodeAlreadyDoneError(RuntimeError):
    """one-step episode에서 reset 없이 두 번째 action을 보냈을 때 발생한다."""


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"config는 mapping이어야 합니다. 받은 타입: {type(value).__name__}")


def _serializable_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "as_dict"):
        return dict(value.as_dict())
    if hasattr(value, "summary"):
        return dict(value.summary())
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return {key: item for key, item in vars(value).items() if not key.startswith("_")}
    return {"value": value}


def _as_finite_action(
    action: Sequence[float] | np.ndarray,
    *,
    expected_size: int,
) -> np.ndarray:
    try:
        array = np.asarray(action, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"action은 길이 {expected_size}의 숫자 배열이어야 합니다.") from exc
    if array.shape != (expected_size,):
        raise ValueError(f"action shape은 ({expected_size},)이어야 합니다: {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("action에 NaN 또는 inf가 포함되어 있습니다.")
    return np.clip(array, -1.0, 1.0)


def _lift_approach_curriculum(
    training_config: Mapping[str, Any],
    episode_index: int | None,
) -> tuple[str, float]:
    """450-episode curriculum의 단계명과 접근 보상 multiplier를 계산한다."""

    raw = training_config.get("lift_approach_curriculum", {})
    curriculum = raw if isinstance(raw, Mapping) else {}
    final_multiplier = float(curriculum.get("final_multiplier", 0.0))
    if not bool(curriculum.get("enabled", False)) or episode_index is None:
        return "final_objective", final_multiplier
    initial_episodes = int(curriculum.get("initial_episodes", 0))
    transition_episodes = int(curriculum.get("transition_episodes", 0))
    if episode_index < initial_episodes:
        return "approach_100", float(curriculum.get("initial_multiplier", 1.0))
    transition_index = episode_index - initial_episodes
    if transition_index < transition_episodes:
        start = float(curriculum.get("transition_start_multiplier", 0.5))
        progress = transition_index / float(max(1, transition_episodes - 1))
        return "approach_anneal", (1.0 - progress) * start + progress * final_multiplier
    return "final_objective", final_multiplier


def _toss_curriculum(
    training_config: Mapping[str, Any],
    reward_config: Mapping[str, Any],
    episode_index: int | None,
) -> tuple[str, float, float, float]:
    """동시 toss 목표, 허용 유출과 성공 bonus curriculum을 계산한다."""

    raw = training_config.get("toss_curriculum", {})
    curriculum = raw if isinstance(raw, Mapping) else {}
    final_goal = float(
        curriculum.get(
            "final_goal_ratio",
            reward_config.get("peak_toss_goal_ratio", 0.20),
        )
    )
    final_spill = float(
        curriculum.get(
            "final_spill_ratio",
            reward_config.get("toss_success_spill_ratio", 0.05),
        )
    )
    final_bonus = float(
        curriculum.get(
            "final_bonus",
            reward_config.get("toss_success_bonus", 0.0),
        )
    )
    if not bool(curriculum.get("enabled", False)) or episode_index is None:
        return "final_objective", final_goal, final_spill, final_bonus
    initial_episodes = int(curriculum.get("initial_episodes", 0))
    transition_episodes = int(curriculum.get("transition_episodes", 0))
    initial_goal = float(curriculum.get("initial_goal_ratio", final_goal))
    initial_spill = float(curriculum.get("initial_spill_ratio", final_spill))
    initial_bonus = float(curriculum.get("initial_bonus", final_bonus))
    if episode_index < initial_episodes:
        return "toss_warmup", initial_goal, initial_spill, initial_bonus
    transition_index = episode_index - initial_episodes
    if transition_index < transition_episodes:
        progress = transition_index / float(max(1, transition_episodes - 1))
        return (
            "toss_anneal",
            (1.0 - progress) * initial_goal + progress * final_goal,
            (1.0 - progress) * initial_spill + progress * final_spill,
            (1.0 - progress) * initial_bonus + progress * final_bonus,
        )
    return "final_objective", final_goal, final_spill, final_bonus


class WokMixingEnv(gym.Env[np.ndarray, np.ndarray]):
    """질량 context에 대해 trajectory parameter를 한 번 선택하는 Gym 환경.

    ``step(action)`` 한 번이 동일 action으로 만든 5-cycle open-loop
    trajectory 전체를 실행한다. episode 중간 particle state는 reward와
    metric에만 사용하며 observation으로 반환하지 않는다.
    """

    metadata = {
        "render_modes": [None, "human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        config: Mapping[str, Any] | Any,
        *,
        render_mode: str | None = None,
        target_mass_kg: float | None = None,
        simulator_factory: Callable[..., Any] | None = None,
        trajectory_factory: Callable[..., Any] | None = None,
        particle_generator: Any | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode는 None, 'human', 'rgb_array' 중 하나여야 합니다.")
        self.config = _mapping(config)
        self.render_mode = render_mode
        self.fixed_target_mass_kg = None if target_mass_kg is None else float(target_mass_kg)
        self._simulator_factory = simulator_factory or WokSimulator
        self._trajectory_factory = trajectory_factory or generate_configured_trajectory
        self._action_size = action_size_for_config(self.config)
        self._pan = PanModel.from_config(self.config)

        particle_config = _mapping(self.config.get("particles", {}))
        particle_profile = (
            str(particle_config.get("profile", particle_config.get("mode", "scaled_spheres")))
            .strip()
            .lower()
        )
        self._fried_rice_particles = particle_profile in {
            "fried_rice",
            "fried-rice",
            "fried_rice_mixture",
        }
        try:
            mass_range = np.asarray(particle_config["target_mass_range_kg"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("particles.target_mass_range_kg=[min,max] 설정이 필요합니다.") from exc
        if (
            mass_range.shape != (2,)
            or not np.isfinite(mass_range).all()
            or mass_range[0] <= 0.0
            or mass_range[0] > mass_range[1]
        ):
            raise ValueError("target_mass_range_kg는 양수이며 증가하는 [min,max]여야 합니다.")
        self._mass_range_kg = mass_range
        if self._fried_rice_particles and self.fixed_target_mass_kg is not None:
            raise ValueError(
                "fried_rice 입자는 지정된 크기와 개별 질량을 유지하므로 "
                "target_mass_kg로 질량을 재스케일할 수 없습니다. "
                "총량은 count_per_type 범위로 조절하세요."
            )
        if self.fixed_target_mass_kg is not None:
            self._validate_target_mass(self.fixed_target_mass_kg)

        training_config = _mapping(self.config.get("training", {}))
        self._include_radius_stats = bool(
            training_config.get(
                "observation_radius_stats",
                training_config.get("include_radius_stats", False),
            )
        )
        self._include_count_per_type = bool(
            training_config.get("observation_count_per_type", False)
        )
        self._include_nominal_joint_speed_target = bool(
            training_config.get("observation_nominal_joint_speed_target", False)
        )
        self._nominal_joint_speed_target_range: tuple[float, float] | None = None
        if self._include_nominal_joint_speed_target:
            raw_speed_targets = training_config.get("nominal_joint_speed_target_schedule")
            try:
                speed_targets = np.asarray(raw_speed_targets, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "speed target 관측에는 "
                    "training.nominal_joint_speed_target_schedule이 필요합니다."
                ) from exc
            if (
                speed_targets.ndim != 1
                or len(speed_targets) < 2
                or not np.isfinite(speed_targets).all()
                or np.any(speed_targets <= 0.0)
                or np.any(speed_targets > 1.0)
                or float(np.min(speed_targets)) == float(np.max(speed_targets))
            ):
                raise ValueError(
                    "speed target 관측 schedule은 서로 다른 (0,1] 값 2개 이상이어야 합니다."
                )
            self._nominal_joint_speed_target_range = (
                float(np.min(speed_targets)),
                float(np.max(speed_targets)),
            )
        self._count_per_type_range: tuple[int, int] | None = None
        if self._include_count_per_type:
            if not self._fried_rice_particles:
                raise ValueError(
                    "training.observation_count_per_type은 fried_rice 입자 profile에서만 "
                    "사용할 수 있습니다."
                )
            fried_rice_config = _mapping(particle_config.get("fried_rice", {}))
            try:
                count_low_raw, count_high_raw = fried_rice_config["count_per_type_range"]
                count_low, count_high = int(count_low_raw), int(count_high_raw)
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "count_per_type 관측에는 "
                    "particles.fried_rice.count_per_type_range=[min,max]가 필요합니다."
                ) from exc
            if (
                isinstance(count_low_raw, bool)
                or isinstance(count_high_raw, bool)
                or count_low != count_low_raw
                or count_high != count_high_raw
                or count_low <= 0
                or count_low >= count_high
            ):
                raise ValueError(
                    "count_per_type 관측 범위는 양수이며 서로 다른 [min,max]여야 합니다."
                )
            self._count_per_type_range = (count_low, count_high)
        mass_normalization = (
            str(training_config.get("observation_mass_normalization", "none")).strip().lower()
        )
        if mass_normalization in {"none", "raw", "raw_kg"}:
            self._mass_observation_normalization = "none"
        elif mass_normalization in {"symmetric_range", "symmetric", "minus_one_to_one"}:
            if self._mass_range_kg[0] == self._mass_range_kg[1]:
                raise ValueError(
                    "symmetric_range 질량 관측에는 서로 다른 target mass 최솟값/최댓값이 "
                    "필요합니다."
                )
            self._mass_observation_normalization = "symmetric_range"
        else:
            raise ValueError(
                "training.observation_mass_normalization은 "
                "'none' 또는 'symmetric_range'여야 합니다."
            )
        low_values = [
            -1.0
            if self._mass_observation_normalization == "symmetric_range"
            else float(self._mass_range_kg[0])
        ]
        high_values = [
            1.0
            if self._mass_observation_normalization == "symmetric_range"
            else float(self._mass_range_kg[1])
        ]
        if self._include_count_per_type:
            low_values.append(-1.0)
            high_values.append(1.0)
        if self._include_nominal_joint_speed_target:
            low_values.append(-1.0)
            high_values.append(1.0)
        if self._include_radius_stats:
            low_values.extend([0.0, 0.0])
            high_values.extend([np.inf, np.inf])
        low = np.asarray(low_values, dtype=np.float32)
        high = np.asarray(high_values, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._action_size,),
            dtype=np.float32,
        )

        if particle_generator is not None:
            self._particle_generator = particle_generator
        elif self._fried_rice_particles:
            from wok_sim.simulation.particle_generator import FriedRiceParticleGenerator

            self._particle_generator = FriedRiceParticleGenerator.from_config(particle_config)
        else:
            self._particle_generator = ParticleGenerator(
                count=int(particle_config["count"]),
                density_kg_m3=float(particle_config["density_kg_m3"]),
                nominal_radius_m=float(particle_config["nominal_radius_m"]),
                radius_jitter_fraction=float(particle_config.get("radius_jitter_fraction", 0.0)),
                spawn_radius_m=float(particle_config["spawn_radius_m"]),
                spawn_height_m=float(particle_config.get("spawn_height_m", 0.0)),
                mass_tolerance_kg=float(particle_config.get("mass_tolerance_kg", 1e-12)),
                minimum_clearance_m=float(
                    particle_config.get(
                        "minimum_clearance_m",
                        particle_config.get("spawn_clearance_m", 0.0),
                    )
                ),
                max_spawn_attempts=int(particle_config.get("max_spawn_attempts", 20_000)),
            )

        self._simulator: Any = None
        self._particles: ParticleBatch | Any | None = None
        self._initial_positions_pan_m: np.ndarray | None = None
        self._initial_labels: np.ndarray | None = None
        self._target_mass_kg: float | None = None
        self._episode_seed: int | None = None
        self._curriculum_episode: int | None = None
        self._nominal_joint_speed_target_fraction: float | None = None
        self._episode_active = False
        self._action_used = False
        self._last_result: SimulationResult | Any | None = None
        self._last_info: dict[str, Any] = {}

    @property
    def particles(self) -> ParticleBatch | Any | None:
        """현재 episode의 입자 batch."""

        return self._particles

    @property
    def simulator(self) -> Any:
        """현재 episode simulator. reset 전에는 None이다."""

        return self._simulator

    @property
    def last_result(self) -> SimulationResult | Any | None:
        """가장 최근 valid rollout의 simulation result."""

        return self._last_result

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """질량/입자를 생성하고 정지 팬에서 settling한 새 episode를 연다."""

        super().reset(seed=seed)
        options = {} if options is None else dict(options)
        raw_curriculum_episode = options.get("curriculum_episode")
        if raw_curriculum_episode is None:
            self._curriculum_episode = None
        else:
            curriculum_episode = int(raw_curriculum_episode)
            if (
                isinstance(raw_curriculum_episode, bool)
                or curriculum_episode != raw_curriculum_episode
                or curriculum_episode < 0
            ):
                raise ValueError("curriculum_episode은 0 이상의 정수여야 합니다.")
            self._curriculum_episode = curriculum_episode
        speed_target = options.get("nominal_joint_speed_target_fraction")
        if speed_target is None:
            robot_config = _mapping(self.config.get("robot", {}))
            retiming_config = _mapping(robot_config.get("nominal_joint_speed_retiming", {}))
            speed_target = retiming_config.get("target_fraction")
        if speed_target is None:
            self._nominal_joint_speed_target_fraction = None
        else:
            resolved_speed_target = float(speed_target)
            if not np.isfinite(resolved_speed_target) or not 0.0 < resolved_speed_target <= 1.0:
                raise ValueError("nominal_joint_speed_target_fraction은 (0,1]이어야 합니다.")
            if self._nominal_joint_speed_target_range is not None:
                speed_low, speed_high = self._nominal_joint_speed_target_range
                if not speed_low <= resolved_speed_target <= speed_high:
                    raise ValueError(
                        "nominal_joint_speed_target_fraction이 observation schedule 범위 "
                        f"[{speed_low}, {speed_high}] 밖입니다."
                    )
            self._nominal_joint_speed_target_fraction = resolved_speed_target
        if self._fried_rice_particles:
            if any(key in options for key in ("target_mass_kg", "target_total_mass_kg", "mass_kg")):
                raise ValueError(
                    "fried_rice 입자는 개별 질량이 정해져 있어 reset에서 "
                    "target mass를 지정할 수 없습니다. count_per_type을 지정하세요."
                )
            target_mass: float | None = None
            count_per_type = options.get("count_per_type")
        else:
            target_mass = self._target_mass_from_options(options)
            self._validate_target_mass(target_mass)
            count_per_type = None
        if seed is not None:
            episode_seed = int(seed)
        elif "particle_seed" in options:
            episode_seed = int(options["particle_seed"])
        else:
            episode_seed = int(self.np_random.integers(0, np.iinfo(np.int32).max, dtype=np.int64))

        if self._simulator is not None:
            self._simulator.close()
        self._simulator = None
        self._last_result = None
        self._last_info = {}

        # particle generator의 center_m z는 pan-local bottom surface다.
        generator_options: dict[str, Any] = {
            "seed": episode_seed,
            "center_m": (0.0, 0.0, self._pan.proxy.bottom_z_m),
        }
        if self._fried_rice_particles and count_per_type is not None:
            generator_options["count_per_type"] = count_per_type
        self._particles = self._particle_generator.generate(target_mass, **generator_options)
        actual_mass = float(self._particles.actual_total_mass_kg)
        if self._fried_rice_particles:
            self._validate_target_mass(actual_mass)
            target_mass = actual_mass
        self._simulator = self._make_simulator(self._particles)
        settle_info = (
            self._simulator.settle()
            if hasattr(self._simulator, "settle")
            else {"settled": None, "elapsed_s": 0.0, "steps": 0}
        )
        if hasattr(self._simulator, "particle_positions_pan"):
            initial_positions = np.asarray(
                self._simulator.particle_positions_pan(), dtype=np.float64
            )
        else:
            initial_positions = np.asarray(self._particles.positions_m, dtype=np.float64)
        if initial_positions.shape != (len(self._particles.radii_m), 3):
            raise RuntimeError("simulator 초기 pan-local position shape이 올바르지 않습니다.")
        self._initial_positions_pan_m = initial_positions.copy()
        mixing_config = _mapping(self.config.get("mixing", {}))
        self._initial_labels = assign_initial_labels(
            initial_positions,
            mode=str(mixing_config.get("label_mode", "quadrant")),
        )

        assert target_mass is not None
        self._target_mass_kg = target_mass
        self._episode_seed = episode_seed
        self._episode_active = True
        self._action_used = False
        observation = self._observation()
        info = {
            **self._episode_metadata(),
            "settling": settle_info,
            "action_count": 0,
        }
        self._last_info = info
        return observation, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """action 한 번으로 전체 trajectory를 실행하고 episode를 종료한다."""

        if not self._episode_active or self._particles is None or self._simulator is None:
            raise EpisodeAlreadyDoneError(
                "활성 episode가 없습니다. step(action) 전에 reset()을 호출하세요."
            )
        if self._action_used:
            raise EpisodeAlreadyDoneError(
                "이 one-step episode에서는 action을 이미 한 번 사용했습니다. "
                "두 번째 step 전에 reset()을 호출하세요."
            )
        normalized_action = _as_finite_action(action, expected_size=self._action_size)
        self._action_used = True

        trajectory: Any | None = None
        trajectory_error: Exception | None = None
        try:
            trajectory_config = self._trajectory_config_for_episode()
            trajectory = self._trajectory_factory(
                normalized_action,
                trajectory_config,
                validate=True,
            )
        except Exception as exc:
            trajectory_error = exc

        trajectory_valid = trajectory_error is None and self._trajectory_valid(trajectory)
        robot_validation: dict[str, Any] = {
            "status": "not_evaluated",
            "reason": "trajectory 생성 전",
        }
        if trajectory_valid and trajectory is not None:
            robot_config = _mapping(self.config.get("robot", {}))
            if not bool(robot_config.get("enabled", False)):
                robot_validation = {
                    "status": "not_evaluated",
                    "reason": (
                        "robot.enabled=false인 pan-only 실행; M0609를 import하거나 연결하지 않음"
                    ),
                }
            else:
                try:
                    robot_result = self._validate_robot(trajectory)
                    robot_validation = robot_result.summary()
                    if robot_result.status == "invalid":
                        trajectory_valid = False
                        trajectory_error = RuntimeError(robot_result.reason)
                except Exception as exc:
                    trajectory_valid = False
                    trajectory_error = exc

        if not trajectory_valid or trajectory is None:
            reward_config = _mapping(self.config.get("reward", {}))
            reward = -float(reward_config.get("w_invalid", 20.0))
            info = self._invalid_info(
                normalized_action,
                trajectory,
                trajectory_error,
                robot_validation,
                reward,
            )
            self._episode_active = False
            self._last_info = info
            return self._observation(), reward, True, False, info

        try:
            result = self._simulator.rollout(trajectory)
        except InvalidTrajectoryError as exc:
            reward_config = _mapping(self.config.get("reward", {}))
            reward = -float(reward_config.get("w_invalid", 20.0))
            info = self._invalid_info(
                normalized_action,
                trajectory,
                exc,
                robot_validation,
                reward,
            )
            self._episode_active = False
            self._last_info = info
            return self._observation(), reward, True, False, info

        self._last_result = result
        info, reward = self._evaluate_result(
            normalized_action,
            trajectory,
            result,
            robot_validation,
        )
        self._episode_active = False
        self._last_info = info
        return self._observation(), float(reward), True, False, info

    def _target_mass_from_options(self, options: Mapping[str, Any]) -> float:
        for key in ("target_mass_kg", "target_total_mass_kg", "mass_kg"):
            if key in options:
                return float(options[key])
        if self.fixed_target_mass_kg is not None:
            return self.fixed_target_mass_kg
        sampling = str(
            _mapping(self.config.get("training", {})).get("target_mass_sampling", "uniform")
        ).lower()
        if sampling in {"uniform", "random"}:
            return float(self.np_random.uniform(*self._mass_range_kg))
        if sampling in {"midpoint", "fixed_midpoint"}:
            return float(np.mean(self._mass_range_kg))
        raise ValueError(f"지원하지 않는 training.target_mass_sampling: {sampling}")

    def _validate_target_mass(self, value: float) -> None:
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("target mass는 유한한 양수여야 합니다.")
        if value < self._mass_range_kg[0] or value > self._mass_range_kg[1]:
            raise ValueError(
                f"target mass {value} kg이 설정 범위 "
                f"[{self._mass_range_kg[0]}, {self._mass_range_kg[1]}] 밖입니다."
            )

    def _make_simulator(self, particles: Any) -> Any:
        try:
            return self._simulator_factory(
                self.config,
                particles,
                render_mode=self.render_mode,
                positions_are_pan_local=True,
            )
        except TypeError as first_error:
            # 테스트 double 또는 이전 adapter의 간결한 signature 지원.
            try:
                return self._simulator_factory(self.config, particles, render_mode=self.render_mode)
            except TypeError as second_error:
                raise first_error from second_error

    @staticmethod
    def _trajectory_valid(trajectory: Any) -> bool:
        validation = getattr(trajectory, "validation", None)
        if isinstance(trajectory, Mapping):
            validation = trajectory.get("validation", validation)
        if validation is not None:
            valid = (
                validation.get("valid")
                if isinstance(validation, Mapping)
                else getattr(validation, "valid", None)
            )
            if valid is not None:
                return bool(valid)
        valid = (
            trajectory.get("valid")
            if isinstance(trajectory, Mapping)
            else getattr(trajectory, "valid", None)
        )
        return True if valid is None else bool(valid)

    def _validate_robot(self, trajectory: Any) -> Any:
        # pan-only profile에서는 이 branch 자체가 실행되지 않으며, Pinocchio,
        # Doosan SDK 또는 ROS를 import하지 않는다.
        from wok_sim.robot.m0609_validator import M0609Validator

        positions = np.asarray(
            getattr(trajectory, "position_m", trajectory.positions),
            dtype=float,
        )
        rpy = np.asarray(trajectory.orientation_rpy_rad, dtype=float)
        transforms = np.stack(
            [
                compose_transform(position, rpy_rad=angles)
                for position, angles in zip(positions, rpy, strict=True)
            ]
        )
        validator = M0609Validator(_mapping(self.config.get("robot", {})))
        return validator.validate(np.asarray(trajectory.time_s, dtype=float), transforms)

    def _evaluate_result(
        self,
        normalized_action: np.ndarray,
        trajectory: Any,
        result: SimulationResult | Any,
        robot_validation: dict[str, Any],
    ) -> tuple[dict[str, Any], float]:
        assert self._particles is not None
        assert self._initial_positions_pan_m is not None
        assert self._initial_labels is not None
        mixing_config = _mapping(self.config.get("mixing", {}))
        mixing_metrics = compute_mixing_metrics(
            self._initial_positions_pan_m,
            np.asarray(result.final_positions_pan_m),
            initial_region_labels=self._initial_labels,
            label_mode=str(mixing_config.get("label_mode", "quadrant")),
            radii_m=self._particles.radii_m,
            grid_rows=int(mixing_config.get("grid_rows", 4)),
            grid_cols=int(mixing_config.get("grid_cols", 4)),
            pan_radius_m=float(mixing_config.get("pan_radius_m", self._pan.proxy.inner_radius_m)),
            minimum_particles_per_cell=int(mixing_config.get("minimum_particles_per_cell", 1)),
            radial_bins=int(mixing_config.get("radial_bins", 5)),
        )

        simulation_config = _mapping(self.config.get("simulation", {}))
        spill_boundary = float(
            simulation_config.get("spill_boundary_radius_m", self._pan.proxy.inner_radius_m * 1.35)
        )
        below_rim = float(simulation_config.get("spill_boundary_below_rim_m", 0.08))
        if hasattr(result, "final_no_contact_duration_s"):
            no_contact_duration = np.asarray(
                result.final_no_contact_duration_s,
                dtype=float,
            )
        else:
            no_contact_duration = self._final_no_contact_duration(
                np.asarray(result.time_s), np.asarray(result.contact_with_pan)
            )
        spill_metrics_object = evaluate_spill(
            np.asarray(result.final_positions_pan_m),
            self._particles.masses_kg,
            rim_radius_m=self._pan.proxy.inner_radius_m,
            spill_boundary_radius_m=spill_boundary,
            spill_z_m=self._pan.proxy.rim_z_m - below_rim,
            boundary_crossed=np.asarray(result.crossed_spill_boundary),
            final_contacts=np.asarray(result.contact_with_pan[-1]),
            no_contact_durations_s=no_contact_duration,
            minimum_no_contact_time_s=float(simulation_config.get("spill_grace_time_s", 0.0)),
        )
        spill_metrics = spill_metrics_object.as_dict()
        flight_metrics = self._flight_metrics(
            result.particle_flight_summary,
            self._particles.radii_m,
            spill_metrics_object.spilled_mask,
        )
        reward_config = _mapping(self.config.get("reward", {}))
        lift_metrics: dict[str, Any] = {
            "evaluation_phase": "unavailable",
            "evaluation_sample_count": 0,
            "rim_z_m": float(self._pan.proxy.rim_z_m),
            "top_margin_m": float(reward_config.get("lift_top_margin_m", 0.0)),
            "approach_band_m": float(reward_config.get("lift_approach_band_m", 0.0)),
            "retained_particle_count": int(np.count_nonzero(~spill_metrics_object.spilled_mask)),
            "lift_approach_particle_equivalents": 0.0,
            "mean_lift_approach_fraction": 0.0,
            "lifted_particle_count": 0,
            "lifted_particle_ratio": 0.0,
            "peak_lifted_particle_ratio": 0.0,
            "peak_lifted_time_s": 0.0,
            "mean_lifted_top_clearance_m": 0.0,
            "maximum_grain_top_clearance_m": 0.0,
            "lift_score": 0.0,
        }
        parameters = getattr(trajectory, "parameters", None)
        phase_durations = getattr(parameters, "phase_durations_s", None)
        if phase_durations is not None and len(phase_durations) >= 4:
            recovery_start_s = float(phase_durations[0] + phase_durations[1])
            lift_metrics = compute_recovery_return_lift_metrics(
                np.asarray(result.time_s),
                np.asarray(result.particle_positions_pan_m),
                np.asarray(result.contact_with_pan),
                self._particles.radii_m,
                spill_metrics_object.spilled_mask,
                rim_z_m=self._pan.proxy.rim_z_m,
                cycle_time_s=float(parameters.cycle_time),
                recovery_start_s=recovery_start_s,
                motion_end_s=float(np.asarray(trajectory.time_s)[-1]),
                top_margin_m=float(reward_config.get("lift_top_margin_m", 0.0)),
                approach_band_m=float(reward_config.get("lift_approach_band_m", 0.0)),
            )
        elif float(reward_config.get("lift_reward_per_particle", 0.0)) > 0.0:
            raise RuntimeError("lift reward에는 phase duration이 있는 trajectory가 필요합니다.")
        trajectory_costs_object = compute_trajectory_costs(
            np.asarray(trajectory.time_s),
            np.asarray(trajectory.linear_acceleration_m_s2),
            np.asarray(trajectory.linear_jerk_m_s3),
            acceleration_normalization=float(reward_config.get("acceleration_normalization", 1.0)),
            jerk_normalization=float(reward_config.get("jerk_normalization", 1.0)),
            angular_acceleration=np.asarray(trajectory.angular_acceleration_rad_s2),
            angular_jerk=np.asarray(trajectory.angular_jerk_rad_s3),
        )
        costs = trajectory_costs_object.as_dict()
        height_penalty = 0.0
        if bool(reward_config.get("height_penalty_enabled", False)):
            excess = max(
                0.0,
                float(flight_metrics["max_flight_height"])
                - float(reward_config.get("maximum_height_m", 0.30)),
            )
            height_penalty = float(reward_config.get("w_height", 0.0)) * excess
        approach_stage, approach_multiplier = _lift_approach_curriculum(
            _mapping(self.config.get("training", {})),
            self._curriculum_episode,
        )
        curriculum_stage, toss_goal, toss_spill_limit, toss_bonus = _toss_curriculum(
            _mapping(self.config.get("training", {})),
            reward_config,
            self._curriculum_episode,
        )
        if curriculum_stage == "final_objective":
            curriculum_stage = approach_stage
        terms, reward_signals = compute_reward_terms(
            mixing_improvement=float(mixing_metrics["mixing_improvement"]),
            lifted_particle_count=int(lift_metrics["lifted_particle_count"]),
            peak_lifted_particle_ratio=float(
                lift_metrics["peak_lifted_particle_ratio"]
            ),
            spill_mass_ratio=float(spill_metrics["spill_mass_ratio"]),
            spill_count_ratio=float(spill_metrics["spill_count_ratio"]),
            jerk_cost=float(costs["jerk_cost"]),
            acceleration_cost=float(costs["acceleration_cost"]),
            height_penalty=height_penalty,
            reward_config=reward_config,
            lift_approach_particle_equivalents=float(
                lift_metrics["lift_approach_particle_equivalents"]
            ),
            lift_approach_multiplier=approach_multiplier,
            peak_toss_goal_ratio=toss_goal,
            toss_success_spill_ratio=toss_spill_limit,
            toss_success_bonus=toss_bonus,
        )
        reward = float(sum(terms.values()))
        info = {
            **self._episode_metadata(),
            "action_count": 1,
            "normalized_action": normalized_action.copy(),
            "action_parameters": _serializable_mapping(getattr(trajectory, "parameters", None)),
            "trajectory_valid": True,
            "invalid_trajectory": False,
            "trajectory_validation": _serializable_mapping(getattr(trajectory, "validation", None)),
            "joint_speed_report": _serializable_mapping(
                getattr(trajectory, "joint_speed_report", None)
            ),
            "robot_validation": robot_validation,
            "mixing": mixing_metrics,
            "spill": spill_metrics,
            "flight": flight_metrics,
            "lift": lift_metrics,
            "trajectory_costs": costs,
            "reward_terms": terms,
            "reward_signals": reward_signals,
            "curriculum_episode": self._curriculum_episode,
            "curriculum_stage": curriculum_stage,
            "final_reward": reward,
            "simulation_metadata": dict(getattr(result, "metadata", {})),
            # Gym checker의 deterministic info 비교가 가능하도록 object 대신
            # ndarray/mapping view를 둔다. 원본은 env.last_result/env.particles에
            # 그대로 보존된다.
            "simulation_result": self._simulation_result_view(result),
            "trajectory": self._trajectory_view(trajectory),
            "particle_batch": self._particle_batch_view(),
        }
        info.update(
            {
                "initial_mixing_score": float(mixing_metrics["initial_mixing_score"]),
                "final_mixing_score": float(mixing_metrics["final_mixing_score"]),
                "mixing_improvement": float(mixing_metrics["mixing_improvement"]),
                "spill_count": int(spill_metrics["spill_count"]),
                "spill_count_ratio": float(spill_metrics["spill_count_ratio"]),
                "spill_mass_kg": float(spill_metrics["spill_mass_kg"]),
                "spill_mass_ratio": float(spill_metrics["spill_mass_ratio"]),
                "lift_score": float(lift_metrics["lift_score"]),
                "lifted_particle_count": int(lift_metrics["lifted_particle_count"]),
                "lifted_particle_ratio": float(lift_metrics["lifted_particle_ratio"]),
                "peak_lifted_particle_ratio": float(lift_metrics["peak_lifted_particle_ratio"]),
                "lift_top_margin_m": float(lift_metrics["top_margin_m"]),
                "lift_reward_per_particle": float(reward_signals["lift_reward_per_particle"]),
                "peak_toss_goal_ratio": float(reward_signals["peak_toss_goal_ratio"]),
                "toss_success_spill_ratio": float(
                    reward_signals["toss_success_spill_ratio"]
                ),
                "toss_success_bonus": float(reward_signals["toss_success_bonus"]),
                "toss_success": bool(reward_signals["toss_success"]),
                "lift_approach_particle_equivalents": float(
                    reward_signals["lift_approach_particle_equivalents"]
                ),
                "lift_approach_multiplier": float(
                    reward_signals["lift_approach_multiplier"]
                ),
                "maximum_grain_top_clearance_m": float(
                    lift_metrics["maximum_grain_top_clearance_m"]
                ),
                "spill_severity": float(reward_signals["spill_severity"]),
                "mean_flight_height": float(flight_metrics["mean_flight_height"]),
                "max_flight_height": float(flight_metrics["max_flight_height"]),
                "flight_height_std": float(flight_metrics["flight_height_std"]),
                "mean_launch_angle": float(flight_metrics["mean_launch_angle"]),
                "launch_angle_std": float(flight_metrics["launch_angle_std"]),
                "acceleration_cost": float(costs["acceleration_cost"]),
                "jerk_cost": float(costs["jerk_cost"]),
            }
        )
        info["episode_summary"] = self._episode_summary(
            info,
            trajectory=trajectory,
        )
        return info, reward

    def _invalid_info(
        self,
        normalized_action: np.ndarray,
        trajectory: Any | None,
        error: Exception | None,
        robot_validation: dict[str, Any],
        reward: float,
    ) -> dict[str, Any]:
        validation = None if trajectory is None else getattr(trajectory, "validation", None)
        approach_stage, approach_multiplier = _lift_approach_curriculum(
            _mapping(self.config.get("training", {})),
            self._curriculum_episode,
        )
        reward_config = _mapping(self.config.get("reward", {}))
        curriculum_stage, toss_goal, toss_spill_limit, toss_bonus = _toss_curriculum(
            _mapping(self.config.get("training", {})),
            reward_config,
            self._curriculum_episode,
        )
        if curriculum_stage == "final_objective":
            curriculum_stage = approach_stage
        reasons: list[str] = []
        if validation is not None:
            violations = getattr(validation, "violations", ())
            reasons.extend(str(item) for item in (violations or ()))
        if error is not None:
            reasons.append(str(error))
        info = {
            **self._episode_metadata(),
            "action_count": 1,
            "normalized_action": normalized_action.copy(),
            "action_parameters": (
                {}
                if trajectory is None
                else _serializable_mapping(getattr(trajectory, "parameters", None))
            ),
            "trajectory_valid": False,
            "invalid_trajectory": True,
            "invalid_reasons": reasons or ["trajectory validation failed"],
            "trajectory_validation": _serializable_mapping(validation),
            "robot_validation": robot_validation,
            "simulation_skipped": True,
            "curriculum_episode": self._curriculum_episode,
            "curriculum_stage": curriculum_stage,
            "reward_terms": {
                "mix": 0.0,
                "lift_approach": 0.0,
                "lift": 0.0,
                "peak_toss": 0.0,
                "toss_success": 0.0,
                "spill": 0.0,
                "jerk": 0.0,
                "acceleration": 0.0,
                "height": 0.0,
                "invalid": reward,
            },
            "reward_signals": {
                "mixing_delta": 0.0,
                "lifted_particle_count": 0.0,
                "lift_approach_particle_equivalents": 0.0,
                "lift_approach_multiplier": approach_multiplier,
                "lift_approach_reward_per_particle": float(
                    _mapping(self.config.get("reward", {})).get(
                        "lift_approach_reward_per_particle",
                        0.0,
                    )
                ),
                "lift_reward_per_particle": float(
                    _mapping(self.config.get("reward", {})).get("lift_reward_per_particle", 0.0)
                ),
                "peak_lifted_particle_ratio": 0.0,
                "peak_toss_goal_ratio": toss_goal,
                "toss_success_spill_ratio": toss_spill_limit,
                "toss_success_bonus": toss_bonus,
                "toss_success": 0.0,
                "spill_mass_ratio": 0.0,
                "spill_count_ratio": 0.0,
                "spill_severity": 0.0,
            },
            "final_reward": reward,
            "trajectory": (None if trajectory is None else self._trajectory_view(trajectory)),
            "particle_batch": self._particle_batch_view(),
        }
        info["episode_summary"] = self._episode_summary(
            info,
            trajectory=trajectory,
        )
        return info

    def _particle_batch_view(self) -> dict[str, Any]:
        assert self._particles is not None
        view: dict[str, Any] = {
            "radii_m": np.asarray(self._particles.radii_m).copy(),
            "masses_kg": np.asarray(self._particles.masses_kg).copy(),
            "positions_m": np.asarray(self._particles.positions_m).copy(),
            "density_kg_m3": float(self._particles.density_kg_m3),
            "target_total_mass_kg": float(self._particles.target_total_mass_kg),
            "actual_total_mass_kg": float(self._particles.actual_total_mass_kg),
            "seed": self._particles.seed,
        }
        for name in (
            "geom_types",
            "sizes_m",
            "species",
            "quaternions_wxyz",
            "restitution_coefficients",
            "contact_time_constants_s",
            "contact_damping_ratios",
            "frictions",
            "linear_damping_per_s",
            "angular_damping_per_s",
            "reference_drop_heights_m",
            "target_rebound_heights_m",
        ):
            value = getattr(self._particles, name, None)
            if value is not None:
                view[name] = np.asarray(value).copy()
        return view

    @staticmethod
    def _trajectory_view(trajectory: Any) -> dict[str, Any]:
        fields = (
            "time_s",
            "position_m",
            "orientation_rpy_rad",
            "linear_velocity_m_s",
            "angular_velocity_rad_s",
            "linear_acceleration_m_s2",
            "angular_acceleration_rad_s2",
            "linear_jerk_m_s3",
            "angular_jerk_rad_s3",
        )
        view: dict[str, Any] = {}
        for name in fields:
            value = (
                trajectory.get(name)
                if isinstance(trajectory, Mapping)
                else getattr(trajectory, name, None)
            )
            if value is not None:
                view[name] = np.asarray(value).copy()
        view["parameters"] = _serializable_mapping(
            trajectory.get("parameters")
            if isinstance(trajectory, Mapping)
            else getattr(trajectory, "parameters", None)
        )
        view["validation"] = _serializable_mapping(
            trajectory.get("validation")
            if isinstance(trajectory, Mapping)
            else getattr(trajectory, "validation", None)
        )
        view["cycle_count"] = (
            trajectory.get("cycle_count")
            if isinstance(trajectory, Mapping)
            else getattr(trajectory, "cycle_count", None)
        )
        return view

    @staticmethod
    def _simulation_result_view(result: Any) -> dict[str, Any]:
        fields = (
            "time_s",
            "particle_positions_world_m",
            "particle_velocities_world_m_s",
            "particle_positions_pan_m",
            "pan_position_world_m",
            "pan_quaternion_wxyz",
            "contact_with_pan",
            "contact_normal_force_n",
            "final_no_contact_duration_s",
            "crossed_spill_boundary",
        )
        view = {name: np.asarray(getattr(result, name)).copy() for name in fields}
        view["particle_flight_summary"] = {
            key: np.asarray(value).copy() for key, value in result.particle_flight_summary.items()
        }
        view["flight_events"] = [
            (event.as_dict() if hasattr(event, "as_dict") else _serializable_mapping(event))
            for event in result.flight_events
        ]
        view["metadata"] = dict(getattr(result, "metadata", {}))
        return view

    def _episode_summary(
        self,
        info: Mapping[str, Any],
        *,
        trajectory: Any | None,
    ) -> dict[str, Any]:
        """큰 physics 배열을 제외한 logger용 episode record를 만든다."""

        trajectory_validation = info.get("trajectory_validation", {})
        validation_metrics = (
            trajectory_validation.get("metrics", {})
            if isinstance(trajectory_validation, Mapping)
            else {}
        )
        return {
            "condition": self._episode_metadata(),
            "action": {
                "normalized": np.asarray(info["normalized_action"]).copy(),
                **dict(info.get("action_parameters", {})),
            },
            "trajectory": {
                "valid": bool(info.get("trajectory_valid", False)),
                "invalid_reasons": list(info.get("invalid_reasons", ())),
                **dict(validation_metrics),
                **dict(info.get("trajectory_costs", {})),
            },
            "robot": dict(info.get("robot_validation", {})),
            "metrics": {
                "mixing": dict(info.get("mixing", {})),
                "spill": dict(info.get("spill", {})),
                "flight": dict(info.get("flight", {})),
                "lift": dict(info.get("lift", {})),
            },
            "reward_signals": dict(info.get("reward_signals", {})),
            "reward_terms": dict(info.get("reward_terms", {})),
            "final_reward": float(info["final_reward"]),
            "simulation_skipped": bool(info.get("simulation_skipped", False)),
            "trajectory_cycle_count": (
                None if trajectory is None else getattr(trajectory, "cycle_count", None)
            ),
        }

    def _episode_metadata(self) -> dict[str, Any]:
        assert self._particles is not None
        metadata: dict[str, Any] = {
            "random_seed": self._episode_seed,
            "target_total_mass_kg": self._target_mass_kg,
            "actual_total_mass_kg": float(self._particles.actual_total_mass_kg),
            "particle_count": int(len(self._particles.radii_m)),
            "particle_density_kg_m3": float(self._particles.density_kg_m3),
            "mean_radius_m": float(np.mean(self._particles.radii_m)),
            "radius_std_m": float(np.std(self._particles.radii_m)),
            "nominal_joint_speed_target_fraction": (
                self._nominal_joint_speed_target_fraction
            ),
        }
        species = getattr(self._particles, "species", None)
        if species is not None:
            names, counts = np.unique(np.asarray(species, dtype=str), return_counts=True)
            metadata["particle_species_counts"] = {
                str(name): int(count) for name, count in zip(names, counts, strict=True)
            }
            if self._fried_rice_particles and len(set(counts)) == 1:
                metadata["count_per_type"] = int(counts[0])
        if self._fried_rice_particles:
            baseline_count = int(getattr(self._particle_generator, "count", 0))
            if baseline_count > 0:
                metadata["particle_amount_fraction"] = metadata["particle_count"] / baseline_count
        return metadata

    def _observation(self) -> np.ndarray:
        if self._particles is None:
            raise RuntimeError("observation을 만들기 전에 reset()이 필요합니다.")
        mass_kg = float(self._particles.actual_total_mass_kg)
        if self._mass_observation_normalization == "symmetric_range":
            mass_low, mass_high = self._mass_range_kg
            mass_value = 2.0 * (mass_kg - mass_low) / (mass_high - mass_low) - 1.0
            mass_value = float(np.clip(mass_value, -1.0, 1.0))
        else:
            mass_value = mass_kg
        values = [mass_value]
        if self._include_count_per_type:
            assert self._count_per_type_range is not None
            species = np.asarray(getattr(self._particles, "species", ()), dtype=str)
            _, counts = np.unique(species, return_counts=True)
            if len(counts) == 0 or len(set(counts)) != 1:
                raise RuntimeError(
                    "count_per_type 관측에는 모든 fried_rice 종의 동일한 입자 수가 필요합니다."
                )
            count_low, count_high = self._count_per_type_range
            count_value = 2.0 * (float(counts[0]) - count_low) / (count_high - count_low) - 1.0
            values.append(float(np.clip(count_value, -1.0, 1.0)))
        if self._include_nominal_joint_speed_target:
            assert self._nominal_joint_speed_target_range is not None
            assert self._nominal_joint_speed_target_fraction is not None
            speed_low, speed_high = self._nominal_joint_speed_target_range
            speed_value = (
                2.0
                * (self._nominal_joint_speed_target_fraction - speed_low)
                / (speed_high - speed_low)
                - 1.0
            )
            values.append(float(np.clip(speed_value, -1.0, 1.0)))
        if self._include_radius_stats:
            values.extend(
                [
                    float(np.mean(self._particles.radii_m)),
                    float(np.std(self._particles.radii_m)),
                ]
            )
        return np.asarray(values, dtype=np.float32)

    def _trajectory_config_for_episode(self) -> Mapping[str, Any]:
        """현재 episode의 nominal speed target을 trajectory 설정에 반영한다."""

        if self._nominal_joint_speed_target_fraction is None:
            return self.config
        trajectory_config = deepcopy(dict(self.config))
        robot_config = dict(_mapping(trajectory_config.get("robot", {})))
        retiming_config = dict(
            _mapping(robot_config.get("nominal_joint_speed_retiming", {}))
        )
        retiming_config["target_fraction"] = self._nominal_joint_speed_target_fraction
        robot_config["nominal_joint_speed_retiming"] = retiming_config
        trajectory_config["robot"] = robot_config
        return trajectory_config

    @staticmethod
    def _final_no_contact_duration(times_s: np.ndarray, contacts: np.ndarray) -> np.ndarray:
        durations = np.zeros(contacts.shape[1], dtype=float)
        for index in range(contacts.shape[1]):
            contact_indices = np.flatnonzero(contacts[:, index])
            if len(contact_indices) == 0:
                durations[index] = float(times_s[-1] - times_s[0])
            elif contact_indices[-1] == len(times_s) - 1:
                durations[index] = 0.0
            else:
                durations[index] = float(times_s[-1] - times_s[contact_indices[-1]])
        return durations

    @staticmethod
    def _flight_metrics(
        summary: Mapping[str, np.ndarray],
        radii_m: np.ndarray,
        spilled_mask: np.ndarray,
    ) -> dict[str, Any]:
        heights = np.asarray(summary["max_flight_height_relative_m"], dtype=float)
        world_heights = np.asarray(summary["max_flight_height_world_m"], dtype=float)
        angles = np.asarray(summary["launch_angle_xz_rad"], dtype=float)
        counts = np.asarray(summary["flight_count"], dtype=int)
        has_flight = counts > 0

        def mean(values: np.ndarray, mask: np.ndarray) -> float:
            selected = values[mask & np.isfinite(values)]
            return float(np.mean(selected)) if len(selected) else 0.0

        def std(values: np.ndarray, mask: np.ndarray) -> float:
            selected = values[mask & np.isfinite(values)]
            return float(np.std(selected)) if len(selected) else 0.0

        order = np.argsort(np.asarray(radii_m), kind="stable")
        group_ids = np.empty(len(order), dtype=int)
        group_ids[order] = np.minimum(2, (3 * np.arange(len(order), dtype=int)) // len(order))
        group_names = ("small", "medium", "large")
        group_statistics = {}
        for group_id, name in enumerate(group_names):
            mask = has_flight & (group_ids == group_id)
            group_statistics[name] = {
                "particle_count": int(np.count_nonzero(group_ids == group_id)),
                "particles_with_flight": int(np.count_nonzero(mask)),
                "mean_flight_height": mean(heights, mask),
                "mean_launch_angle": mean(angles, mask),
            }
        return {
            "particles_with_flight": int(np.count_nonzero(has_flight)),
            "flight_event_count": int(np.sum(counts)),
            "mean_flight_height": mean(heights, has_flight),
            "mean_flight_height_relative": mean(heights, has_flight),
            "mean_flight_height_world": mean(world_heights, has_flight),
            "max_flight_height": (
                float(np.max(heights[has_flight])) if np.any(has_flight) else 0.0
            ),
            "max_flight_height_relative": (
                float(np.max(heights[has_flight])) if np.any(has_flight) else 0.0
            ),
            "max_flight_height_world": (
                float(np.max(world_heights[has_flight])) if np.any(has_flight) else 0.0
            ),
            "flight_height_std": std(heights, has_flight),
            "mean_launch_angle": mean(angles, has_flight),
            "launch_angle_std": std(angles, has_flight),
            "size_group_flight_statistics": group_statistics,
            "spilled_mean_flight_height": mean(
                heights, has_flight & np.asarray(spilled_mask, dtype=bool)
            ),
            "spilled_mean_launch_angle": mean(
                angles, has_flight & np.asarray(spilled_mask, dtype=bool)
            ),
            "particle_flight_summary": {
                key: np.asarray(value).copy() for key, value in summary.items()
            },
        }

    def render(self) -> np.ndarray | None:
        """현재 simulator render 결과를 반환한다."""

        if self._simulator is None:
            return None
        return self._simulator.render()

    def close(self) -> None:
        """MuJoCo resource를 해제한다."""

        if self._simulator is not None:
            self._simulator.close()
            self._simulator = None


__all__ = ["EpisodeAlreadyDoneError", "WokMixingEnv"]
