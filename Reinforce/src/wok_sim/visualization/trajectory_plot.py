"""최종 SAC 정책의 60/90/120입자 3D·X-Z 궤적과 mixing을 시각화한다."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402

from wok_sim.config import load_config
from wok_sim.envs import WokMixingEnv
from wok_sim.geometry import quaternion_to_matrix
from wok_sim.simulation.pan_model import CollisionProxyConfig

from .pan_profile import pan_side_profile_local, transform_pan_local_history

DEFAULT_COUNTS_PER_TYPE = (20, 30, 40)
DEFAULT_TOTAL_PARTICLE_COUNTS = tuple(3 * count for count in DEFAULT_COUNTS_PER_TYPE)
PAN_CENTER_COLOR = "#D62728"
PAN_RIM_COLOR = "#1F77B4"
PAN_CENTER_LABEL = "pan center / center trajectory"
PAN_RIM_LABEL = "pan rim / outline"
PAN_RIM_ENDPOINT_LABEL = "pan rim endpoints / 0.1 s points"
PAN_RIM_ENDPOINT_ORDER = ("rear / local -x", "front / local +x")


def _interpolate_history(
    time_s: np.ndarray, values: np.ndarray, sample_time_s: np.ndarray
) -> np.ndarray:
    source = np.asarray(values, dtype=float)
    flattened = source.reshape(len(time_s), -1)
    sampled = np.column_stack(
        [
            np.interp(sample_time_s, time_s, flattened[:, index])
            for index in range(flattened.shape[1])
        ]
    )
    return sampled.reshape((len(sample_time_s), *source.shape[1:]))


def pan_rim_geometry_from_config(
    config: Mapping[str, Any] | Any,
) -> tuple[np.ndarray, float, float]:
    """collision proxy와 같은 pan-local rim 끝점 두 개를 반환한다.

    반환 배열의 순서는 rear/local -x, front/local +x다. ``rim_radius_m``은
    바깥 반지름이므로 실제 capsule 중심선은 inner/outer 반지름의 평균을 쓴다.
    """

    pan_config = config.get("pan", {}) if isinstance(config, Mapping) else {}
    proxy_config = pan_config.get("collision_proxy", {}) if isinstance(pan_config, Mapping) else {}
    proxy = CollisionProxyConfig.from_mapping(proxy_config)
    centerline_radius = 0.5 * (proxy.inner_radius_m + proxy.rim_radius_m)
    rim_z = float(proxy.rim_z_m)
    local_endpoints = np.asarray(
        (
            (-centerline_radius, 0.0, rim_z),
            (+centerline_radius, 0.0, rim_z),
        ),
        dtype=float,
    )
    return local_endpoints, float(centerline_radius), rim_z


def pan_rim_endpoint_history(
    pan_position_world_m: np.ndarray,
    pan_quaternion_wxyz: np.ndarray,
    local_endpoints_m: np.ndarray,
) -> np.ndarray:
    """각 raw pan pose에서 두 rim 끝점을 world frame으로 변환한다."""

    pan = np.asarray(pan_position_world_m, dtype=float)
    quaternions = np.asarray(pan_quaternion_wxyz, dtype=float)
    local_endpoints = np.asarray(local_endpoints_m, dtype=float)
    if pan.ndim != 2 or pan.shape[1:] != (3,):
        raise ValueError("pan_position_world_m shape은 (T, 3)이어야 합니다.")
    if quaternions.shape != (len(pan), 4):
        raise ValueError("pan_quaternion_wxyz shape은 (T, 4)이어야 합니다.")
    if local_endpoints.shape != (2, 3):
        raise ValueError("local_endpoints_m shape은 (2, 3)이어야 합니다.")
    if not (
        np.isfinite(pan).all()
        and np.isfinite(quaternions).all()
        and np.isfinite(local_endpoints).all()
    ):
        raise ValueError("pan pose와 local rim 끝점은 모두 유한해야 합니다.")
    rotations = np.stack([quaternion_to_matrix(item) for item in quaternions])
    rotated = np.einsum("tij,pj->tpi", rotations, local_endpoints)
    return pan[:, None, :] + rotated


def resample_pan_rim_endpoints(
    time_s: np.ndarray,
    endpoint_history_world_m: np.ndarray,
    sample_time_s: np.ndarray,
) -> np.ndarray:
    """raw rim 끝점 궤적을 center/particle과 같은 시각으로 보간한다."""

    time = np.asarray(time_s, dtype=float)
    endpoints = np.asarray(endpoint_history_world_m, dtype=float)
    sample_time = np.asarray(sample_time_s, dtype=float)
    if (
        time.ndim != 1
        or len(time) < 2
        or not np.isfinite(time).all()
        or np.any(np.diff(time) <= 0.0)
    ):
        raise ValueError("time_s는 유한하고 엄격히 증가하는 1차원 배열이어야 합니다.")
    if endpoints.shape != (len(time), 2, 3) or not np.isfinite(endpoints).all():
        raise ValueError("endpoint_history_world_m shape은 유한한 (T, 2, 3)이어야 합니다.")
    if sample_time.ndim != 1 or not np.isfinite(sample_time).all():
        raise ValueError("sample_time_s는 유한한 1차원 배열이어야 합니다.")
    if len(sample_time) and (
        sample_time[0] < time[0] - 1.0e-12 or sample_time[-1] > time[-1] + 1.0e-12
    ):
        raise ValueError("sample_time_s는 raw time_s 범위 안에 있어야 합니다.")
    return _interpolate_history(time, endpoints, sample_time)


def resample_trajectory_history(
    time_s: np.ndarray,
    pan_position_world_m: np.ndarray,
    particle_positions_world_m: np.ndarray,
    *,
    interval_s: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """pan/particle world trajectory를 고정 시간 간격으로 선형 보간한다."""

    time = np.asarray(time_s, dtype=float)
    pan = np.asarray(pan_position_world_m, dtype=float)
    particles = np.asarray(particle_positions_world_m, dtype=float)
    if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("time_s는 유한하고 엄격히 증가하는 1차원 배열이어야 합니다.")
    if pan.shape != (len(time), 3):
        raise ValueError("pan_position_world_m shape은 (T, 3)이어야 합니다.")
    if particles.ndim != 3 or particles.shape[0] != len(time) or particles.shape[2] != 3:
        raise ValueError("particle_positions_world_m shape은 (T, N, 3)이어야 합니다.")
    if not np.isfinite(interval_s) or interval_s <= 0:
        raise ValueError("interval_s는 유한한 양수여야 합니다.")
    sample_time = np.arange(time[0], time[-1] + interval_s * 1e-9, interval_s)
    return (
        sample_time,
        _interpolate_history(time, pan, sample_time),
        _interpolate_history(time, particles, sample_time),
    )


def _set_equal_limits(axes: list[Any], points: list[np.ndarray]) -> None:
    combined = np.concatenate([item.reshape(-1, 3) for item in points], axis=0)
    minimum = combined.min(axis=0)
    maximum = combined.max(axis=0)
    center = (minimum + maximum) * 0.5
    radius = max(float(np.max(maximum - minimum)) * 0.525, 0.01)
    for axis in axes:
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1, 1, 1))


def _final_pan_rim(case: dict[str, Any], *, point_count: int = 73) -> np.ndarray:
    """복원된 최종 raw pan pose에서 collision rim outline을 만든다."""

    center = np.asarray(case["pan_final_position_world_m"], dtype=float)
    rotation = quaternion_to_matrix(case["pan_final_quaternion_wxyz"])
    radius = float(case["pan_rim_center_radius_m"])
    rim_z = float(case["pan_rim_z_m"])
    theta = np.linspace(0.0, 2.0 * np.pi, point_count)
    local_rim = np.column_stack(
        (
            radius * np.cos(theta),
            radius * np.sin(theta),
            np.full_like(theta, rim_z),
        )
    )
    return center + local_rim @ rotation.T


def _draw_rim_endpoint_paths_3d(axis: Any, endpoints: np.ndarray) -> None:
    for index in range(2):
        label = PAN_RIM_ENDPOINT_LABEL if index == 0 else "_nolegend_"
        axis.plot(
            endpoints[:, index, 0],
            endpoints[:, index, 1],
            endpoints[:, index, 2],
            color=PAN_RIM_COLOR,
            linewidth=1.7,
            alpha=0.9,
            label=label,
        )
        axis.scatter(
            endpoints[:, index, 0],
            endpoints[:, index, 1],
            endpoints[:, index, 2],
            color=PAN_RIM_COLOR,
            s=7,
            alpha=0.75,
            linewidths=0,
        )


def _draw_rim_endpoint_paths_xz(axis: Any, endpoints: np.ndarray) -> None:
    for index in range(2):
        label = PAN_RIM_ENDPOINT_LABEL if index == 0 else "_nolegend_"
        axis.plot(
            endpoints[:, index, 0],
            endpoints[:, index, 2],
            color=PAN_RIM_COLOR,
            linewidth=1.7,
            alpha=0.9,
            label=label,
        )
        axis.scatter(
            endpoints[:, index, 0],
            endpoints[:, index, 2],
            color=PAN_RIM_COLOR,
            s=8,
            alpha=0.75,
            linewidths=0,
        )


def _draw_case(axis: Any, case: dict[str, Any]) -> None:
    particles = case["particles"]
    pan = case["pan"]
    rim_endpoints = case["pan_rim_endpoints"]
    time = case["time"]
    colors = np.repeat(time, particles.shape[1])
    axis.scatter(
        particles[:, :, 0].reshape(-1),
        particles[:, :, 1].reshape(-1),
        particles[:, :, 2].reshape(-1),
        c=colors,
        cmap="viridis",
        s=0.35,
        alpha=0.16,
        linewidths=0,
        rasterized=True,
    )
    axis.plot(
        pan[:, 0],
        pan[:, 1],
        pan[:, 2],
        color=PAN_CENTER_COLOR,
        linewidth=2.2,
        label=PAN_CENTER_LABEL,
    )
    axis.scatter(pan[:, 0], pan[:, 1], pan[:, 2], color=PAN_CENTER_COLOR, s=7, alpha=0.8)
    _draw_rim_endpoint_paths_3d(axis, rim_endpoints)
    rim = _final_pan_rim(case)
    axis.plot(
        rim[:, 0],
        rim[:, 1],
        rim[:, 2],
        color=PAN_RIM_COLOR,
        linewidth=2.0,
        label=PAN_RIM_LABEL,
    )
    side_profile = case["pan_side_profile_final_world_m"]
    axis.plot(
        side_profile[:, 0],
        side_profile[:, 1],
        side_profile[:, 2],
        color=PAN_RIM_COLOR,
        linewidth=2.0,
        alpha=0.9,
        label="pan curved side profile",
    )
    axis.set_title(
        f"{case['total_particle_count']} particles | actual "
        f"{case['actual_mass_g']:.2f} g | mix {case['final_mixing_score']:.3f} | "
        f"spill {case['spill_count']}"
    )
    axis.set_xlabel("world x (m)")
    axis.set_ylabel("world y (m)")
    axis.set_zlabel("world z (m)")
    axis.legend(loc="upper right", fontsize=7)


def _draw_xz_case(axis: Any, case: dict[str, Any]) -> None:
    particles = case["particles"]
    pan = case["pan"]
    rim_endpoints = case["pan_rim_endpoints"]
    time = case["time"]
    colors = np.repeat(time, particles.shape[1])
    axis.scatter(
        particles[:, :, 0].reshape(-1),
        particles[:, :, 2].reshape(-1),
        c=colors,
        cmap="viridis",
        s=0.45,
        alpha=0.18,
        linewidths=0,
        rasterized=True,
    )
    axis.plot(
        pan[:, 0],
        pan[:, 2],
        color=PAN_CENTER_COLOR,
        linewidth=2.2,
        label=PAN_CENTER_LABEL,
    )
    axis.scatter(pan[:, 0], pan[:, 2], color=PAN_CENTER_COLOR, s=8, alpha=0.8)
    _draw_rim_endpoint_paths_xz(axis, rim_endpoints)
    rim = _final_pan_rim(case)
    axis.plot(
        rim[:, 0],
        rim[:, 2],
        color=PAN_RIM_COLOR,
        linewidth=2.0,
        label=PAN_RIM_LABEL,
    )
    side_profile = case["pan_side_profile_final_world_m"]
    axis.plot(
        side_profile[:, 0],
        side_profile[:, 2],
        color=PAN_RIM_COLOR,
        linewidth=2.2,
        alpha=0.9,
        label="pan curved side profile",
    )
    axis.set_title(
        f"{case['total_particle_count']} particles | {case['actual_mass_g']:.2f} g\n"
        f"mix {case['final_mixing_score']:.3f} "
        f"(Δ {case['mixing_improvement']:+.3f}) | spill {case['spill_count']}"
    )
    axis.set_xlabel("world x (m)")
    axis.set_ylabel("world z (m)")
    axis.legend(loc="upper right", fontsize=7)


def _set_equal_xz_limits(axes: list[Any], points: list[np.ndarray]) -> None:
    combined = np.concatenate([item.reshape(-1, 3)[:, (0, 2)] for item in points], axis=0)
    minimum = combined.min(axis=0)
    maximum = combined.max(axis=0)
    center = (minimum + maximum) * 0.5
    radius = max(float(np.max(maximum - minimum)) * 0.525, 0.01)
    for axis in axes:
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_aspect("equal", adjustable="box")


def save_trajectory_plot_artifacts(
    cases: list[dict[str, Any]],
    output_directory: Path,
    *,
    interval_s: float,
) -> dict[str, Path]:
    """60/90/120입자 case의 3D, X-Z 및 mixing 비교 그래프를 저장한다."""

    actual_counts = tuple(int(case["total_particle_count"]) for case in cases)
    if actual_counts != DEFAULT_TOTAL_PARTICLE_COUNTS:
        raise ValueError(
            f"trajectory plot case는 총입자 60/90/120 순서여야 합니다: {actual_counts!r}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    all_points = (
        [case["particles"] for case in cases]
        + [case["pan"] for case in cases]
        + [case["pan_rim_endpoints"] for case in cases]
        + [_final_pan_rim(case) for case in cases]
        + [case["pan_side_profile_final_world_m"] for case in cases]
    )

    figure = plt.figure(figsize=(18, 6), constrained_layout=True)
    axes_3d = [figure.add_subplot(1, 3, index + 1, projection="3d") for index in range(3)]
    for axis, case in zip(axes_3d, cases, strict=True):
        _draw_case(axis, case)
    _set_equal_limits(axes_3d, all_points)
    figure.suptitle(
        f"Final SAC policy: 3D pan/particle trajectories ({interval_s:g} s points)",
        fontsize=15,
    )
    overview_3d = output_directory / "trajectory_3d_comparison_60_90_120_particles.png"
    figure.savefig(overview_3d, dpi=180)
    plt.close(figure)

    figure, axes_xz_array = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    axes_xz = list(np.asarray(axes_xz_array).reshape(-1))
    for axis, case in zip(axes_xz, cases, strict=True):
        _draw_xz_case(axis, case)
    _set_equal_xz_limits(axes_xz, all_points)
    figure.suptitle(
        f"Final SAC policy: X-Z pan/particle trajectories ({interval_s:g} s points)",
        fontsize=15,
    )
    overview_xz = output_directory / "trajectory_xz_comparison_60_90_120_particles.png"
    figure.savefig(overview_xz, dpi=180)
    plt.close(figure)

    counts = np.asarray(actual_counts, dtype=int)
    initial_scores = np.asarray([case["initial_mixing_score"] for case in cases], dtype=float)
    final_scores = np.asarray([case["final_mixing_score"] for case in cases], dtype=float)
    improvements = np.asarray([case["mixing_improvement"] for case in cases], dtype=float)
    figure, (score_axis, improvement_axis) = plt.subplots(
        2,
        1,
        figsize=(9, 8),
        sharex=True,
        constrained_layout=True,
    )
    score_axis.plot(counts, initial_scores, "o--", label="initial mixing score")
    score_axis.plot(counts, final_scores, "o-", label="final mixing score")
    score_axis.set_ylabel("normalized mixing score")
    score_axis.set_ylim(-0.05, 1.05)
    score_axis.grid(alpha=0.25)
    score_axis.legend()
    improvement_axis.bar(counts, improvements, width=12.0, color="seagreen", alpha=0.8)
    improvement_axis.axhline(0.0, color="black", linewidth=0.8)
    improvement_axis.set_xlabel("total particle count")
    improvement_axis.set_ylabel("mixing improvement")
    improvement_axis.set_xticks(counts)
    improvement_axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Mixing score by particle count", fontsize=15)
    mixing_plot = output_directory / "mixing_score_by_particle_count.png"
    figure.savefig(mixing_plot, dpi=180)
    plt.close(figure)

    for case in cases:
        figure = plt.figure(figsize=(10, 8), constrained_layout=True)
        axis = figure.add_subplot(111, projection="3d")
        _draw_case(axis, case)
        _set_equal_limits(
            [axis],
            [
                case["particles"],
                case["pan"],
                case["pan_rim_endpoints"],
                _final_pan_rim(case),
            ],
        )
        figure.savefig(
            output_directory / f"trajectory_3d_particles_{case['total_particle_count']:03d}.png",
            dpi=180,
        )
        plt.close(figure)

    return {
        "trajectory_3d": overview_3d,
        "trajectory_xz": overview_xz,
        "mixing_score": mixing_plot,
    }


def generate_policy_trajectory_plots(
    config_path: Path,
    checkpoint_path: Path,
    output_directory: Path,
    *,
    seed: int = 1,
    interval_s: float = 0.1,
) -> Path:
    config = load_config(config_path)
    checkpoint = (
        checkpoint_path if checkpoint_path.is_file() else checkpoint_path.with_suffix(".zip")
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAC checkpoint를 찾을 수 없습니다: {checkpoint_path}")
    output_directory.mkdir(parents=True, exist_ok=True)
    model = SAC.load(checkpoint, device="cpu")
    environment = WokMixingEnv(config)
    local_rim_endpoints, rim_center_radius, rim_z = pan_rim_geometry_from_config(config)
    local_side_profile = pan_side_profile_local(config)
    rng = np.random.default_rng(seed)
    cases: list[dict[str, Any]] = []
    try:
        for count, expected_total in zip(
            DEFAULT_COUNTS_PER_TYPE,
            DEFAULT_TOTAL_PARTICLE_COUNTS,
            strict=True,
        ):
            episode_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            observation, reset_info = environment.reset(
                seed=episode_seed,
                options={"count_per_type": count},
            )
            action, _ = model.predict(observation, deterministic=True)
            _, reward, terminated, truncated, info = environment.step(action)
            if not terminated or truncated:
                raise RuntimeError("WokMixingEnv는 one-step terminated episode여야 합니다.")
            if not bool(info.get("trajectory_valid", False)):
                raise RuntimeError(f"trajectory가 유효하지 않습니다: {info.get('invalid_reasons')}")
            actual_total = int(info.get("particle_count", reset_info["particle_count"]))
            if actual_total != expected_total:
                raise RuntimeError(
                    f"종별 {count}개 조건의 총입자가 {expected_total}개가 아닙니다: {actual_total}"
                )
            result = info["simulation_result"]
            raw_time = np.asarray(result["time_s"], dtype=float)
            raw_pan = np.asarray(result["pan_position_world_m"], dtype=float)
            raw_pan_quaternion = np.asarray(result["pan_quaternion_wxyz"], dtype=float)
            sampled_time, sampled_pan, sampled_particles = resample_trajectory_history(
                raw_time,
                raw_pan,
                result["particle_positions_world_m"],
                interval_s=interval_s,
            )
            raw_rim_endpoints = pan_rim_endpoint_history(
                raw_pan,
                raw_pan_quaternion,
                local_rim_endpoints,
            )
            sampled_rim_endpoints = resample_pan_rim_endpoints(
                raw_time,
                raw_rim_endpoints,
                sampled_time,
            )
            final_side_profile = transform_pan_local_history(
                raw_pan[-1:],
                raw_pan_quaternion[-1:],
                local_side_profile,
            )[0]
            actual_mass_g = (
                float(info.get("actual_total_mass_kg", reset_info["actual_total_mass_kg"])) * 1000.0
            )
            mixing = info["mixing"]
            spill = info["spill"]
            case = {
                "actual_mass_g": actual_mass_g,
                "count_per_type": count,
                "total_particle_count": actual_total,
                "seed": episode_seed,
                "reward": float(reward),
                "initial_mixing_score": float(mixing["initial_mixing_score"]),
                "final_mixing_score": float(mixing["final_mixing_score"]),
                "mixing_improvement": float(mixing["mixing_improvement"]),
                "spill_count": int(spill["spill_count"]),
                "spill_count_ratio": float(spill["spill_count_ratio"]),
                "spill_mass_g": float(spill["spill_mass_kg"]) * 1000.0,
                "spill_mass_ratio": float(spill["spill_mass_ratio"]),
                "time": sampled_time,
                "pan": sampled_pan,
                "pan_radius_m": float(config.get("mixing", {}).get("pan_radius_m", 0.112)),
                "pan_rim_endpoints": sampled_rim_endpoints,
                "pan_rim_center_radius_m": rim_center_radius,
                "pan_rim_z_m": rim_z,
                "pan_final_position_world_m": raw_pan[-1].copy(),
                "pan_final_quaternion_wxyz": raw_pan_quaternion[-1].copy(),
                "pan_side_profile_final_world_m": final_side_profile,
                "particles": sampled_particles,
                "normalized_action": np.asarray(action, dtype=float),
            }
            cases.append(case)
            np.savez_compressed(
                output_directory / f"trajectory_particles_{actual_total:03d}.npz",
                time_s=sampled_time,
                pan_position_world_m=sampled_pan,
                pan_rim_endpoints_world_m=sampled_rim_endpoints,
                pan_rim_endpoint_order=np.asarray(PAN_RIM_ENDPOINT_ORDER),
                pan_rim_center_radius_m=rim_center_radius,
                pan_rim_z_m=rim_z,
                pan_side_profile_final_world_m=final_side_profile,
                particle_positions_world_m=sampled_particles,
                normalized_action=case["normalized_action"],
                actual_mass_g=actual_mass_g,
                count_per_type=count,
                total_particle_count=actual_total,
                seed=episode_seed,
                reward=float(reward),
                initial_mixing_score=case["initial_mixing_score"],
                final_mixing_score=case["final_mixing_score"],
                mixing_improvement=case["mixing_improvement"],
                spill_count=case["spill_count"],
                spill_count_ratio=case["spill_count_ratio"],
                spill_mass_g=case["spill_mass_g"],
                spill_mass_ratio=case["spill_mass_ratio"],
                sample_interval_s=interval_s,
            )
    finally:
        environment.close()

    artifacts = save_trajectory_plot_artifacts(
        cases,
        output_directory,
        interval_s=interval_s,
    )
    summary = {
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "sample_interval_s": interval_s,
        "pan_rim_endpoint_order": list(PAN_RIM_ENDPOINT_ORDER),
        "pan_rim_center_radius_m": rim_center_radius,
        "pan_rim_z_m": rim_z,
        "pan_side_profile": "closed_flat_bottom_inner_outer_circular_arcs",
        "pan_side_profile_point_count": len(local_side_profile),
        "strata": [
            {
                "actual_mass_g": case["actual_mass_g"],
                "count_per_type": case["count_per_type"],
                "total_particle_count": case["total_particle_count"],
                "seed": case["seed"],
                "reward": case["reward"],
                "mixing": {
                    "initial_score": case["initial_mixing_score"],
                    "final_score": case["final_mixing_score"],
                    "improvement": case["mixing_improvement"],
                },
                "spill": {
                    "count": case["spill_count"],
                    "count_ratio": case["spill_count_ratio"],
                    "mass_g": case["spill_mass_g"],
                    "mass_ratio": case["spill_mass_ratio"],
                },
                "samples": len(case["time"]),
                "duration_s": float(case["time"][-1]),
                "npz": str(
                    output_directory
                    / f"trajectory_particles_{case['total_particle_count']:03d}.npz"
                ),
            }
            for case in cases
        ],
        "plots": {name: str(path) for name, path in artifacts.items()},
    }
    summary_path = output_directory / "trajectory_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return artifacts["trajectory_xz"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/fried_rice.yaml"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/fried_rice_sac_150_clustered_lift4d_bonus010/policy.zip"),
    )
    parser.add_argument("--output", type=Path, default=Path("results/fried_rice/trajectory_plots"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--interval-s", type=float, default=0.1)
    arguments = parser.parse_args()
    output = generate_policy_trajectory_plots(
        arguments.config,
        arguments.checkpoint,
        arguments.output,
        seed=arguments.seed,
        interval_s=arguments.interval_s,
    )
    print(
        json.dumps(
            {
                "trajectory_xz": str(output),
                "summary": str(arguments.output / "trajectory_summary.json"),
                "output_directory": str(arguments.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
