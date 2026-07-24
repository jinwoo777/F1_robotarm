"""Wok simulation 명령행 인터페이스."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import typer

from wok_sim.config import ConfigError, load_config
from wok_sim.envs import WokMixingEnv
from wok_sim.geometry.pan_asset import PanAssetError, inspect_pan_asset
from wok_sim.geometry.transforms import compose_transform
from wok_sim.logging import EpisodeLogger
from wok_sim.robot.base import RobotValidationResult
from wok_sim.robot.exporter import TrajectoryExportData, export_trajectory
from wok_sim.training import evaluate_policy, run_baseline, train_sac
from wok_sim.training.baselines import compact_episode_info
from wok_sim.trajectory import generate_configured_trajectory

app = typer.Typer(
    name="wok-sim",
    no_args_is_help=True,
    help="합산 질량에 따른 5-cycle open-loop 웍질 시뮬레이션",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if hasattr(value, "summary"):
        return value.summary()
    return str(value)


def _print_json(value: Any) -> None:
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default))


def _json_safe(value: Any) -> Any:
    """NumPy 값과 비유한 float를 표준 JSON 값으로 재귀 변환한다."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_config(path: Path) -> dict[str, Any]:
    try:
        return load_config(path)
    except (ConfigError, OSError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--config") from exc


def _mass_kg(mass_g: float) -> float:
    if not np.isfinite(mass_g) or mass_g <= 0:
        raise typer.BadParameter("--mass-g는 유한한 양수여야 합니다.")
    return float(mass_g) / 1000.0


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if hasattr(value, "summary"):
        return value.summary()
    return {}


def _episode_record(
    config: Mapping[str, Any],
    info: Mapping[str, Any],
    *,
    episode_id: int,
) -> dict[str, Any]:
    """요구된 episode CSV 열을 안정적인 flat schema로 만든다."""

    particles = config.get("particles", {})
    action = _mapping(info.get("action_parameters"))
    validation = _mapping(info.get("trajectory_validation"))
    trajectory_metrics = _mapping(validation.get("metrics"))
    costs = _mapping(info.get("trajectory_costs"))
    robot = _mapping(info.get("robot_validation"))
    mixing = _mapping(info.get("mixing"))
    spill = _mapping(info.get("spill"))
    flight = _mapping(info.get("flight"))
    particle_profile = (
        str(particles.get("profile", particles.get("mode", "scaled_spheres"))).strip().lower()
    )
    heterogeneous_particles = particle_profile in {
        "fried_rice",
        "fried-rice",
        "fried_rice_mixture",
    }
    return {
        "episode_id": episode_id,
        "random_seed": info.get("random_seed"),
        "target_total_mass_kg": info.get("target_total_mass_kg"),
        "actual_total_mass_kg": info.get("actual_total_mass_kg"),
        "particle_count": info.get("particle_count"),
        "count_per_type": info.get("count_per_type"),
        "particle_amount_fraction": info.get("particle_amount_fraction"),
        "particle_density_kg_m3": info.get("particle_density_kg_m3"),
        "mean_radius_m": info.get("mean_radius_m"),
        "radius_std_m": info.get("radius_std_m"),
        "particle_species_counts_json": json.dumps(
            _json_safe(info.get("particle_species_counts", {})),
            ensure_ascii=False,
            allow_nan=False,
        ),
        "friction": particles.get("friction"),
        "restitution": (
            "per_species_drop_calibrated"
            if heterogeneous_particles
            else particles.get("restitution")
        ),
        "insertion_distance_m": action.get("insertion_distance"),
        "lift_height_m": action.get("lift_height"),
        "backward_distance_m": action.get("backward_distance"),
        "pitch_amplitude_rad": action.get("pitch_amplitude"),
        "tilt_angle_rad": action.get("tilt_angle"),
        "target_linear_speed_m_s": action.get("linear_speed"),
        "target_angular_speed_rad_s": action.get("angular_speed"),
        "cycle_time_s": action.get("cycle_time"),
        "insert_phase_ratio": action.get("insert_phase_ratio"),
        "catch_phase_ratio": action.get("catch_phase_ratio"),
        "maximum_cartesian_velocity_m_s": trajectory_metrics.get("max_cartesian_velocity"),
        "maximum_angular_velocity_rad_s": trajectory_metrics.get("max_angular_velocity"),
        "maximum_cartesian_acceleration_m_s2": trajectory_metrics.get("max_cartesian_acceleration"),
        "maximum_angular_acceleration_rad_s2": trajectory_metrics.get("max_angular_acceleration"),
        "maximum_cartesian_jerk_m_s3": trajectory_metrics.get("max_cartesian_jerk"),
        "maximum_angular_jerk_rad_s3": trajectory_metrics.get("max_angular_jerk"),
        "total_jerk_cost": costs.get("jerk_cost"),
        "total_acceleration_cost": costs.get("acceleration_cost"),
        "robot_validation_status": robot.get("status", "not_evaluated"),
        "ik_success": robot.get("ik_success"),
        "minimum_joint_limit_margin_rad": robot.get("minimum_joint_limit_margin_rad"),
        "maximum_joint_velocity_rad_s": robot.get("maximum_joint_velocity_rad_s"),
        "maximum_joint_acceleration_rad_s2": robot.get("maximum_joint_acceleration_rad_s2"),
        "minimum_singularity_margin": robot.get("minimum_singular_value"),
        "collision_status": robot.get("collision_status", "not_evaluated"),
        "initial_mixing_score": mixing.get("initial_mixing_score"),
        "final_mixing_score": mixing.get("final_mixing_score"),
        "mixing_improvement": mixing.get("mixing_improvement"),
        "spill_count": spill.get("spill_count"),
        "spill_count_ratio": spill.get("spill_count_ratio"),
        "spill_mass_kg": spill.get("spill_mass_kg"),
        "spill_mass_ratio": spill.get("spill_mass_ratio"),
        "mean_flight_height_m": flight.get("mean_flight_height"),
        "maximum_flight_height_m": flight.get("max_flight_height"),
        "flight_height_std_m": flight.get("flight_height_std"),
        "mean_launch_angle_rad": flight.get("mean_launch_angle"),
        "launch_angle_std_rad": flight.get("launch_angle_std"),
        "particle_flight_summary_json": json.dumps(
            _json_safe(flight.get("particle_flight_summary", {})),
            ensure_ascii=False,
            allow_nan=False,
        ),
        "trajectory_valid": info.get("trajectory_valid"),
        "invalid_reasons": info.get("invalid_reasons"),
        "final_reward": info.get("final_reward"),
    }


def _printable_episode_info(info: Mapping[str, Any]) -> dict[str, Any]:
    """CLI에는 중복·입자별 배열을 제외한 검토 가능한 summary만 표시한다."""

    summary = compact_episode_info(info)
    summary.pop("episode_summary", None)
    for section_name, excluded in (
        (
            "mixing",
            {
                "initial_radial_distribution",
                "final_radial_distribution",
                "size_group_labels",
            },
        ),
        ("spill", {"spilled_mask"}),
        ("flight", {"particle_flight_summary"}),
    ):
        section = summary.get(section_name)
        if isinstance(section, Mapping):
            summary[section_name] = {
                key: value for key, value in section.items() if key not in excluded
            }
    return summary


def _particle_history(info: Mapping[str, Any]) -> dict[str, np.ndarray] | None:
    result = info.get("simulation_result")
    if result is None:
        return None
    if isinstance(result, Mapping):
        return {
            "time_s": np.asarray(result["time_s"]),
            "position_world_m": np.asarray(result["particle_positions_world_m"]),
            "velocity_world_m_s": np.asarray(result["particle_velocities_world_m_s"]),
            "position_pan_m": np.asarray(result["particle_positions_pan_m"]),
            "contact_with_pan": np.asarray(result["contact_with_pan"], dtype=np.uint8),
        }
    return {
        "time_s": np.asarray(result.time_s),
        "position_world_m": np.asarray(result.particle_positions_world_m),
        "velocity_world_m_s": np.asarray(result.particle_velocities_world_m_s),
        "position_pan_m": np.asarray(result.particle_positions_pan_m),
        "contact_with_pan": np.asarray(result.contact_with_pan, dtype=np.uint8),
    }


def _particle_logging_enabled(config: Mapping[str, Any]) -> bool:
    logging_config = _mapping(config.get("logging", {}))
    return bool(
        logging_config.get("save_particle_trajectories", False)
        and logging_config.get("save_npz", True)
    )


def _policy_action(
    checkpoint: Path | None,
    observation: np.ndarray,
    action_shape: tuple[int, ...],
) -> np.ndarray:
    if checkpoint is None:
        return np.zeros(action_shape, dtype=np.float32)
    candidate = (
        checkpoint
        if checkpoint.is_file() or checkpoint.suffix == ".zip"
        else Path(f"{checkpoint}.zip")
    )
    if not candidate.is_file():
        raise typer.BadParameter(
            f"SAC checkpoint를 찾을 수 없습니다: {checkpoint}",
            param_hint="--checkpoint",
        )
    from stable_baselines3 import SAC

    model = SAC.load(candidate)
    action, _ = model.predict(observation, deterministic=True)
    return np.asarray(action, dtype=np.float32)


@app.command("check-assets")
def check_assets(
    config: Path = typer.Option(..., exists=True, readable=True, help="YAML 설정 파일"),
) -> None:
    """STL 로딩, bounding box, scale 및 watertight 여부를 검사한다."""

    loaded = _load_config(config)
    try:
        report = inspect_pan_asset(loaded["pan"])
    except PanAssetError as exc:
        raise typer.BadParameter(str(exc), param_hint="--config") from exc
    payload = report.to_dict()
    payload["notice"] = (
        "procedural demo pan: 사용자 STL 검증이 아닙니다."
        if report.procedural_demo
        else "STL은 visual 전용이며 collision은 compound proxy입니다."
    )
    logger = EpisodeLogger(loaded)
    logger.write_metadata(
        {
            "command": "check-assets",
            "asset_report": payload,
            "simulation_core": "not_run",
            "m0609_validation": "not_evaluated",
            "real_robot_execution": "not_implemented",
        }
    )
    payload["result_directory"] = str(logger.run_directory)
    _print_json(payload)


@app.command()
def rollout(
    config: Path = typer.Option(..., exists=True, readable=True),
    mass_g: float | None = typer.Option(
        None,
        help="목표 합산 질량(g). fried_rice에서는 count-per-type을 사용",
    ),
    count_per_type: int | None = typer.Option(
        None,
        min=1,
        help="fried_rice 각 종류의 입자 개수",
    ),
    seed: int = typer.Option(1),
    render: bool = typer.Option(False, "--render", help="interactive viewer"),
    headless: bool = typer.Option(False, "--headless", help="명시적 headless 실행"),
    random_action: bool = typer.Option(False, help="중앙 action 대신 random action"),
    checkpoint: Path | None = typer.Option(None, help="선택적 SAC checkpoint"),
) -> None:
    """선택한 질량/입자 수에서 전체 5-cycle episode를 한 번 실행한다."""

    if render and headless:
        raise typer.BadParameter("--render와 --headless를 동시에 사용할 수 없습니다.")
    if random_action and checkpoint is not None:
        raise typer.BadParameter("--random-action과 --checkpoint는 함께 쓸 수 없습니다.")
    loaded = _load_config(config)
    target_mass = None if mass_g is None else _mass_kg(mass_g)
    environment = WokMixingEnv(
        loaded,
        render_mode="human" if render else None,
        target_mass_kg=target_mass,
    )
    try:
        reset_options = {} if count_per_type is None else {"count_per_type": count_per_type}
        observation, _ = environment.reset(seed=seed, options=reset_options)
        if random_action:
            environment.action_space.seed(seed)
            action = environment.action_space.sample()
        else:
            action = _policy_action(checkpoint, observation, environment.action_space.shape)
        _, reward, terminated, truncated, info = environment.step(action)
        if not terminated or truncated:
            raise RuntimeError("예상한 one-step episode 종료 상태가 아닙니다.")
        logger = EpisodeLogger(loaded)
        logger.log_episode(
            _episode_record(loaded, info, episode_id=0),
            particle_trajectory=_particle_history(info),
        )
        logger.write_metadata(
            {
                "simulation_core": "verified_for_this_rollout",
                "pan_asset": (
                    "procedural_demo_not_user_stl"
                    if loaded["pan"].get("use_procedural_demo", False)
                    else loaded["pan"].get("stl_path")
                ),
                "m0609_validation": info.get("robot_validation"),
                "real_robot_execution": "not_implemented",
            }
        )
        summary = _printable_episode_info(info)
        summary["final_reward"] = float(reward)
        summary["result_directory"] = str(logger.run_directory)
        _print_json(summary)
    finally:
        environment.close()


@app.command()
def baseline(
    config: Path = typer.Option(..., exists=True, readable=True),
    episodes: int = typer.Option(100, min=1),
    seed: int = typer.Option(1),
    strategy: str = typer.Option("random", help="random, random_walk 또는 center"),
) -> None:
    """random, episode random-walk 또는 중앙 action baseline을 실행한다."""

    loaded = _load_config(config)
    logger = EpisodeLogger(loaded)
    results = run_baseline(
        loaded,
        episodes=episodes,
        seed=seed,
        strategy=strategy,
        keep_physics_history=_particle_logging_enabled(loaded),
    )
    for episode_id, info in enumerate(results):
        logger.log_episode(
            _episode_record(loaded, info, episode_id=episode_id),
            particle_trajectory=_particle_history(info),
        )
    logger.write_metadata(
        {
            "command": "baseline",
            "strategy": strategy,
            "episodes": episodes,
            "simulation_core": "verified_for_completed_episodes",
            "m0609_validation": "per_episode_status_in_episodes_csv",
            "real_robot_execution": "not_implemented",
        }
    )
    rewards = np.asarray([item["final_reward"] for item in results], dtype=float)
    _print_json(
        {
            "episodes": episodes,
            "strategy": strategy,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "result_directory": str(logger.run_directory),
        }
    )


@app.command()
def train(
    config: Path = typer.Option(..., exists=True, readable=True),
    checkpoint: Path | None = typer.Option(None),
    timesteps: int | None = typer.Option(None, min=1),
) -> None:
    """Stable-Baselines3 SAC를 학습한다."""

    loaded = _load_config(config)
    logger = EpisodeLogger(loaded)
    save_particle_history = _particle_logging_enabled(loaded)

    def log_training_episode(episode_id: int, info: Mapping[str, Any]) -> None:
        logger.log_episode(
            _episode_record(loaded, info, episode_id=episode_id),
            particle_trajectory=(_particle_history(info) if save_particle_history else None),
        )

    result = train_sac(
        loaded,
        checkpoint_path=checkpoint,
        total_timesteps=timesteps,
        episode_consumer=log_training_episode,
        evaluation_directory=logger.run_directory / "evaluation",
    )
    logger.write_metadata(
        {
            "command": "train",
            "algorithm": "SAC",
            "checkpoint": result.checkpoint_path,
            "total_timesteps": result.total_timesteps,
            "periodic_evaluation": result.evaluation_directory,
            "m0609_validation": "per_episode_status_in_episodes_csv",
            "real_robot_execution": "not_implemented",
        }
    )
    payload = asdict(result)
    payload["result_directory"] = logger.run_directory
    _print_json(payload)


@app.command()
def evaluate(
    config: Path = typer.Option(..., exists=True, readable=True),
    checkpoint: Path = typer.Option(...),
    episodes: int = typer.Option(10, min=1),
    seed: int = typer.Option(1),
    count_per_type: list[int] | None = typer.Option(
        None,
        "--count-per-type",
        min=1,
        help="볶음밥 종별 개수 schedule; 여러 번 지정 가능",
    ),
) -> None:
    """학습 checkpoint를 deterministic action으로 평가한다."""

    loaded = _load_config(config)
    logger = EpisodeLogger(loaded)
    results = evaluate_policy(
        loaded,
        checkpoint,
        episodes=episodes,
        seed=seed,
        counts_per_type=count_per_type,
        keep_physics_history=_particle_logging_enabled(loaded),
    )
    for episode_id, info in enumerate(results):
        logger.log_episode(
            _episode_record(loaded, info, episode_id=episode_id),
            particle_trajectory=_particle_history(info),
        )
    logger.write_metadata(
        {
            "command": "evaluate",
            "checkpoint": checkpoint,
            "episodes": episodes,
            "count_per_type_schedule": count_per_type,
            "simulation_core": "verified_for_completed_episodes",
            "m0609_validation": "per_episode_status_in_episodes_csv",
            "real_robot_execution": "not_implemented",
        }
    )
    rewards = np.asarray([item["final_reward"] for item in results], dtype=float)
    _print_json(
        {
            "episodes": episodes,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "result_directory": str(logger.run_directory),
        }
    )


@app.command("export-trajectory")
def export_trajectory_command(
    config: Path = typer.Option(..., exists=True, readable=True),
    mass_g: float | None = typer.Option(
        None,
        help="목표 합산 질량(g). fried_rice 고정 조성에서는 생략",
    ),
    output: Path = typer.Option(...),
    checkpoint: Path | None = typer.Option(None),
    seed: int = typer.Option(1),
) -> None:
    """정책 action의 Cartesian 및 선택적 joint trajectory를 내보낸다."""

    loaded = _load_config(config)
    target_mass = None if mass_g is None else _mass_kg(mass_g)
    environment = WokMixingEnv(loaded, target_mass_kg=target_mass)
    try:
        observation, _ = environment.reset(seed=seed)
        action = _policy_action(checkpoint, observation, environment.action_space.shape)
    finally:
        environment.close()
    trajectory = generate_configured_trajectory(action, loaded)
    pan_transforms = np.stack(
        [
            compose_transform(position, rpy_rad=rpy)
            for position, rpy in zip(
                trajectory.position_m,
                trajectory.orientation_rpy_rad,
                strict=True,
            )
        ]
    )
    if bool(loaded["robot"].get("enabled", False)):
        from wok_sim.robot.m0609_validator import M0609Validator

        robot_result = M0609Validator(loaded["robot"]).validate(
            trajectory.time_s,
            pan_transforms,
        )
    else:
        robot_result = RobotValidationResult.not_evaluated(
            "robot.enabled=false인 pan-only export; M0609를 import하거나 연결하지 않음"
        )
    T_tcp_pan_raw = loaded["robot"].get("T_tcp_pan")
    path = export_trajectory(
        TrajectoryExportData.from_trajectory(trajectory),
        output,
        T_tcp_pan=(None if T_tcp_pan_raw is None else np.asarray(T_tcp_pan_raw, dtype=float)),
        robot_result=robot_result,
    )
    logger = EpisodeLogger(loaded)
    logger.write_metadata(
        {
            "command": "export-trajectory",
            "trajectory_output": path,
            "target_mass_kg": target_mass,
            "checkpoint": checkpoint,
            "trajectory_valid": trajectory.validation.valid,
            "m0609_validation": robot_result.summary(),
            "real_robot_execution": "not_implemented",
        }
    )
    _print_json(
        {
            "output": str(path),
            "samples": len(trajectory.time_s),
            "duration_s": trajectory.duration_s,
            "trajectory_valid": trajectory.validation.valid,
            "robot_validation": robot_result.summary(),
            "result_directory": str(logger.run_directory),
        }
    )


@app.command("check-motion-contract")
def check_motion_contract(
    config: Path = typer.Option(..., exists=True, readable=True),
    checkpoint: Path | None = typer.Option(None),
    seed: int = typer.Option(1),
) -> None:
    """로봇 연결 없이 M0609 후보 Cartesian cap만 strict 검사한다."""

    from wok_sim.robot.m0609_motion_contract import (
        M0609CartesianCaps,
        validate_m0609_cartesian_motion,
    )

    loaded = _load_config(config)
    environment = WokMixingEnv(loaded)
    try:
        observation, reset_info = environment.reset(seed=seed)
        action = _policy_action(checkpoint, observation, environment.action_space.shape)
    finally:
        environment.close()
    trajectory = generate_configured_trajectory(action, loaded)
    caps = M0609CartesianCaps.from_mapping(
        _mapping(loaded.get("robot", {}).get("cartesian_caps", {}))
    )
    T_tcp_pan_raw = loaded.get("robot", {}).get("T_tcp_pan")
    payload_kg = float(loaded["pan"]["mass_kg"]) + float(reset_info["actual_total_mass_kg"])
    report = validate_m0609_cartesian_motion(
        trajectory,
        caps=caps,
        T_tcp_pan=(None if T_tcp_pan_raw is None else np.asarray(T_tcp_pan_raw, dtype=float)),
        payload_kg=payload_kg,
    )
    _print_json(
        {
            "action": action,
            "action_parameters": trajectory.parameters,
            "trajectory_validation": trajectory.validation,
            "motion_contract": report.summary(),
            "notice": ("오프라인 Cartesian 수치 검사일 뿐이며 URDF/TEACHING/실기 검증이 아닙니다."),
        }
    )


@app.command("export-doosan-plan")
def export_doosan_plan_command(
    config: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(..., help=".json, .csv 또는 .py 비실행 parameter data"),
    checkpoint: Path | None = typer.Option(None),
    function: str = typer.Option("movel", help="movel 또는 movesx"),
    waypoint_stride: int | None = typer.Option(None, min=1),
    seed: int = typer.Option(1),
) -> None:
    """strict cap 통과 trajectory를 비실행 Doosan parameter data로 저장한다."""

    from wok_sim.robot.m0609_motion_contract import (
        M0609CartesianCaps,
        M0609MotionContractError,
        export_doosan_offline_plan,
    )

    loaded = _load_config(config)
    environment = WokMixingEnv(loaded)
    try:
        observation, reset_info = environment.reset(seed=seed)
        action = _policy_action(checkpoint, observation, environment.action_space.shape)
    finally:
        environment.close()
    trajectory = generate_configured_trajectory(action, loaded)
    caps = M0609CartesianCaps.from_mapping(
        _mapping(loaded.get("robot", {}).get("cartesian_caps", {}))
    )
    T_tcp_pan_raw = loaded.get("robot", {}).get("T_tcp_pan")
    payload_kg = float(loaded["pan"]["mass_kg"]) + float(reset_info["actual_total_mass_kg"])
    try:
        path = export_doosan_offline_plan(
            trajectory,
            output,
            caps=caps,
            T_tcp_pan=(None if T_tcp_pan_raw is None else np.asarray(T_tcp_pan_raw, dtype=float)),
            payload_kg=payload_kg,
            function=function,
            waypoint_stride=waypoint_stride,
        )
    except M0609MotionContractError as exc:
        raise typer.BadParameter(str(exc), param_hint="--config/--output") from exc
    _print_json(
        {
            "output": path,
            "function": function,
            "executable": False,
            "safety_status": "unverified_without_urdf_teaching",
            "real_robot_command_sent": False,
        }
    )


@app.command("smoke-test")
def smoke_test(
    config: Path = typer.Option(Path("configs/test.yaml"), exists=True, readable=True),
    seed: int = typer.Option(1),
) -> None:
    """중앙 action headless rollout과 CSV export를 한 번 검증한다."""

    loaded = _load_config(config)
    mass_range = loaded["particles"]["target_mass_range_kg"]
    particle_profile = (
        str(
            loaded["particles"].get(
                "profile",
                loaded["particles"].get("mode", "scaled_spheres"),
            )
        )
        .strip()
        .lower()
    )
    target_mass = (
        None
        if particle_profile in {"fried_rice", "fried-rice", "fried_rice_mixture"}
        else float(np.mean(mass_range))
    )
    environment = WokMixingEnv(loaded, target_mass_kg=target_mass)
    try:
        observation, _ = environment.reset(seed=seed)
        action = np.zeros(environment.action_space.shape, dtype=np.float32)
        _, reward, terminated, truncated, info = environment.step(action)
    finally:
        environment.close()
    if not terminated or truncated or not info.get("trajectory_valid", False):
        raise RuntimeError(f"smoke rollout 실패: {info.get('invalid_reasons')}")
    trajectory = generate_configured_trajectory(action, loaded)
    logger = EpisodeLogger(loaded)
    output = export_trajectory(
        TrajectoryExportData.from_trajectory(trajectory),
        logger.run_directory / "smoke_trajectory.csv",
    )
    logger.log_episode(
        _episode_record(loaded, info, episode_id=0),
        particle_trajectory=_particle_history(info),
    )
    logger.write_metadata(
        {
            "command": "smoke-test",
            "simulation_core": "verified",
            "pan_asset": (
                "procedural_demo_not_user_stl"
                if loaded["pan"].get("use_procedural_demo", False)
                else loaded["pan"].get("stl_path")
            ),
            "m0609_validation": info["robot_validation"],
            "real_robot_execution": "not_implemented",
        }
    )
    _print_json(
        {
            "simulation_core": "verified",
            "procedural_pan_rollout": bool(loaded["pan"].get("use_procedural_demo", False)),
            "user_stl": (
                "not_used"
                if loaded["pan"].get("use_procedural_demo", False)
                else loaded["pan"].get("stl_path")
            ),
            "m0609_urdf_validation": info["robot_validation"]["status"],
            "reward": float(reward),
            "trajectory_csv": str(output),
            "result_directory": str(logger.run_directory),
        }
    )


if __name__ == "__main__":
    app()
