"""변경된 볶음밥 teaching을 단 한 episode 실행해 검토용 그래프로 저장한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402

from wok_sim.config import load_config
from wok_sim.envs import WokMixingEnv

from .pan_profile import (
    pan_side_profile_local,
    resample_point_history,
    transform_pan_local_history,
)
from .trajectory_plot import (
    PAN_CENTER_COLOR,
    PAN_CENTER_LABEL,
    PAN_RIM_COLOR,
    PAN_RIM_ENDPOINT_LABEL,
    PAN_RIM_ENDPOINT_ORDER,
    PAN_RIM_LABEL,
    pan_rim_endpoint_history,
    pan_rim_geometry_from_config,
    resample_pan_rim_endpoints,
    resample_trajectory_history,
)

SPECIES_COLORS = {
    "large_sphere": "#E76F51",
    "small_sphere": "#2A9D8F",
    "ellipsoid": "#457B9D",
}


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _species_color(species: str) -> str:
    return SPECIES_COLORS.get(str(species), "#6C757D")


def _draw_world_xz(
    axis: Any,
    time_s: np.ndarray,
    pan: np.ndarray,
    particles: np.ndarray,
    species: np.ndarray,
    pan_radius_m: float,
    pan_rim_endpoints: np.ndarray | None = None,
    pan_side_profile_history: np.ndarray | None = None,
) -> None:
    for index, name in enumerate(species):
        color = _species_color(str(name))
        axis.plot(
            particles[:, index, 0],
            particles[:, index, 2],
            color=color,
            linewidth=0.35,
            alpha=0.16,
        )
        axis.scatter(
            particles[:, index, 0],
            particles[:, index, 2],
            color=color,
            s=1.5,
            alpha=0.20,
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
    axis.scatter(
        pan[:, 0],
        pan[:, 2],
        color=PAN_CENTER_COLOR,
        s=9,
        linewidths=0,
        zorder=4,
        label="pan center: 0.1 s points",
    )
    if pan_rim_endpoints is None:
        final_rim = np.asarray(
            [
                [pan[-1, 0] - pan_radius_m, pan[-1, 2]],
                [pan[-1, 0] + pan_radius_m, pan[-1, 2]],
            ]
        )
    else:
        endpoints = np.asarray(pan_rim_endpoints, dtype=float)
        if endpoints.shape != (len(time_s), 2, 3):
            raise ValueError("pan_rim_endpoints shape은 (T, 2, 3)이어야 합니다.")
        for endpoint_index in range(2):
            axis.plot(
                endpoints[:, endpoint_index, 0],
                endpoints[:, endpoint_index, 2],
                color=PAN_RIM_COLOR,
                linewidth=1.7,
                alpha=0.9,
                label=(PAN_RIM_ENDPOINT_LABEL if endpoint_index == 0 else "_nolegend_"),
            )
            axis.scatter(
                endpoints[:, endpoint_index, 0],
                endpoints[:, endpoint_index, 2],
                color=PAN_RIM_COLOR,
                s=9,
                alpha=0.78,
                linewidths=0,
                zorder=3,
            )
        final_rim = endpoints[-1][:, (0, 2)]
    final_outline = final_rim
    if pan_side_profile_history is not None:
        profiles = np.asarray(pan_side_profile_history, dtype=float)
        if profiles.ndim != 3 or profiles.shape[0] != len(time_s) or profiles.shape[2] != 3:
            raise ValueError("pan_side_profile_history shape은 (T,P,3)이어야 합니다.")
        final_outline = profiles[-1][:, (0, 2)]
    axis.plot(
        final_outline[:, 0],
        final_outline[:, 1],
        color=PAN_RIM_COLOR,
        linewidth=2.0,
        label=PAN_RIM_LABEL,
    )
    handles = [
        plt.Line2D([0], [0], color=color, linewidth=2, label=name.replace("_", " "))
        for name, color in SPECIES_COLORS.items()
    ]
    handles.extend(
        [
            plt.Line2D([0], [0], color=PAN_CENTER_COLOR, linewidth=2, label=PAN_CENTER_LABEL),
            plt.Line2D(
                [0],
                [0],
                color=PAN_RIM_COLOR,
                marker="o",
                markersize=3,
                linewidth=2,
                label=PAN_RIM_ENDPOINT_LABEL,
            ),
        ]
    )
    axis.legend(handles=handles, loc="best", fontsize=8, frameon=False)
    axis.set_title("A. Pan + particle paths in world X-Z")
    axis.set_xlabel("world x (m) → front")
    axis.set_ylabel("world z (m)")
    axis.grid(alpha=0.18)


def _draw_teaching_geometry(
    axis: Any,
    trajectory: dict[str, Any],
    pan_radius_m: float,
    *,
    sampled_time_s: np.ndarray | None = None,
    sampled_pan: np.ndarray | None = None,
    sampled_rim_endpoints: np.ndarray | None = None,
    sampled_pan_side_profiles: np.ndarray | None = None,
) -> None:
    trajectory_time = np.asarray(trajectory["time_s"], dtype=float)
    trajectory_position = np.asarray(trajectory["position_m"], dtype=float)
    parameters = trajectory["parameters"]
    cycle_time = float(parameters["cycle_time"])
    tilt_out = float(parameters["tilt_out_phase_duration"])
    descent = float(parameters["descent_phase_duration"])
    partial_recovery = float(parameters["partial_recovery_phase_duration"])
    phase_times = np.asarray(
        [
            0.0,
            tilt_out,
            tilt_out + descent,
            tilt_out + descent + partial_recovery,
            cycle_time,
        ]
    )
    time_s = trajectory_time if sampled_time_s is None else np.asarray(sampled_time_s, dtype=float)
    position = trajectory_position if sampled_pan is None else np.asarray(sampled_pan, dtype=float)
    if position.shape != (len(time_s), 3):
        raise ValueError("sampled_pan shape은 (T, 3)이어야 합니다.")
    mask = time_s <= cycle_time + 1.0e-9
    if not np.any(mask):
        raise ValueError("첫 teaching cycle에 해당하는 sampled pan point가 없습니다.")
    axis.plot(
        position[mask, 0],
        position[mask, 2],
        color=PAN_CENTER_COLOR,
        linewidth=2.2,
        label=PAN_CENTER_LABEL,
    )
    axis.scatter(
        position[mask, 0],
        position[mask, 2],
        color=PAN_CENTER_COLOR,
        s=11,
        linewidths=0,
    )
    if sampled_rim_endpoints is None:
        final_rim = np.asarray(
            [
                [position[mask][-1, 0] - pan_radius_m, position[mask][-1, 2]],
                [position[mask][-1, 0] + pan_radius_m, position[mask][-1, 2]],
            ]
        )
    else:
        endpoints = np.asarray(sampled_rim_endpoints, dtype=float)
        if endpoints.shape != (len(time_s), 2, 3):
            raise ValueError("sampled_rim_endpoints shape은 (T, 2, 3)이어야 합니다.")
        first_cycle_endpoints = endpoints[mask]
        for endpoint_index in range(2):
            axis.plot(
                first_cycle_endpoints[:, endpoint_index, 0],
                first_cycle_endpoints[:, endpoint_index, 2],
                color=PAN_RIM_COLOR,
                linewidth=1.7,
                alpha=0.9,
                label=(PAN_RIM_ENDPOINT_LABEL if endpoint_index == 0 else "_nolegend_"),
            )
            axis.scatter(
                first_cycle_endpoints[:, endpoint_index, 0],
                first_cycle_endpoints[:, endpoint_index, 2],
                color=PAN_RIM_COLOR,
                s=10,
                alpha=0.78,
                linewidths=0,
            )
        final_rim = first_cycle_endpoints[-1][:, (0, 2)]
    waypoint_x = np.interp(phase_times, time_s, position[:, 0])
    waypoint_z = np.interp(phase_times, time_s, position[:, 2])
    axis.scatter(waypoint_x, waypoint_z, s=48, color=PAN_CENTER_COLOR, zorder=5)
    final_outline = final_rim
    if sampled_pan_side_profiles is not None:
        profiles = np.asarray(sampled_pan_side_profiles, dtype=float)
        if profiles.ndim != 3 or profiles.shape[0] != len(time_s) or profiles.shape[2] != 3:
            raise ValueError("sampled_pan_side_profiles shape은 (T,P,3)이어야 합니다.")
        final_outline = profiles[mask][-1][:, (0, 2)]
    axis.plot(
        final_outline[:, 0],
        final_outline[:, 1],
        color=PAN_RIM_COLOR,
        linewidth=2.0,
        label=PAN_RIM_LABEL,
    )
    lift_angle = float(parameters.get("lift_angle", parameters.get("tilt_recovery_angle", 0.0)))
    tilt_angle = float(parameters["pan_tilt_angle"])
    residual_pitch = tilt_angle - lift_angle
    descent_angle = float(parameters["descent_angle"])
    trajectory_orientation = np.asarray(
        trajectory.get("orientation_rpy_rad", np.zeros((len(trajectory_time), 3))),
        dtype=float,
    )
    if trajectory_orientation.shape == (len(trajectory_time), 3):
        phase_pitch_deg = np.rad2deg(
            np.interp(phase_times, trajectory_time, trajectory_orientation[:, 1])
        )
    else:
        phase_pitch_deg = np.rad2deg([0.0, tilt_angle, tilt_angle, residual_pitch, 0.0])
    axis.annotate(
        f"P0 / P4\n({phase_pitch_deg[0]:.1f}° / {phase_pitch_deg[4]:.1f}°)",
        (waypoint_x[0], waypoint_z[0]),
        xytext=(7, 5),
        textcoords="offset points",
        fontsize=9,
        weight="bold",
    )
    axis.annotate(
        f"P1 / P2 tilt {phase_pitch_deg[1]:.1f}°\nP3 residual {phase_pitch_deg[3]:.1f}°",
        (waypoint_x[2], waypoint_z[2]),
        xytext=(7, -23),
        textcoords="offset points",
        fontsize=9,
        weight="bold",
    )
    axis.annotate(
        f"pre-tilt to {np.rad2deg(tilt_angle):.1f}° before insert",
        xy=(waypoint_x[1], waypoint_z[1]),
        xytext=(0.10, 0.68),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#374151"},
        fontsize=9,
    )
    axis.annotate(
        f"{np.rad2deg(descent_angle):.1f}° insert, hold pitch {np.rad2deg(tilt_angle):.1f}°",
        xy=(waypoint_x[2], waypoint_z[2]),
        xytext=(0.39, 0.18),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#374151"},
        fontsize=9,
    )
    axis.annotate(
        f"lift {np.rad2deg(lift_angle):.1f}° → residual {np.rad2deg(residual_pitch):.1f}°",
        xy=(waypoint_x[3], waypoint_z[3]),
        xytext=(0.42, 0.72),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#374151"},
        fontsize=9,
    )
    axis.set_title("B. One teaching cycle (P0 → P1 → P2 → P3 → P4)")
    axis.set_xlabel("pan x (m) → front")
    axis.set_ylabel("pan z (m)")
    axis.grid(alpha=0.18)
    axis.legend(loc="best", fontsize=8, frameon=False)


def _draw_final_top_view(
    axis: Any,
    initial: np.ndarray,
    final: np.ndarray,
    species: np.ndarray,
    pan_radius_m: float,
) -> None:
    for name in np.unique(species):
        selected = species == name
        color = _species_color(str(name))
        axis.scatter(
            initial[selected, 0],
            initial[selected, 1],
            facecolors="none",
            edgecolors=color,
            s=24,
            alpha=0.45,
            linewidths=0.8,
        )
        axis.scatter(
            final[selected, 0],
            final[selected, 1],
            color=color,
            s=22,
            alpha=0.88,
            linewidths=0,
            label=str(name).replace("_", " "),
        )
    circle = plt.Circle(
        (0.0, 0.0),
        pan_radius_m,
        fill=False,
        color=PAN_RIM_COLOR,
        linewidth=1.8,
        label=PAN_RIM_LABEL,
    )
    axis.add_patch(circle)
    axis.scatter(
        [0.0],
        [0.0],
        color=PAN_CENTER_COLOR,
        s=30,
        zorder=5,
        label="pan center",
    )
    axis.axvline(0.0, color="#6B7280", linestyle="--", linewidth=0.9)
    axis.annotate(
        "+x front",
        xy=(pan_radius_m, 0.0),
        xytext=(pan_radius_m * 0.25, pan_radius_m * 0.82),
        arrowprops={"arrowstyle": "->", "color": "#111827"},
        fontsize=9,
    )
    axis.set_aspect("equal", adjustable="box")
    limit = pan_radius_m * 1.45
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_title("C. Pan-local distribution (open=initial, filled=final)")
    axis.set_xlabel("pan x (m) → front")
    axis.set_ylabel("pan y (m)")
    axis.legend(loc="upper left", fontsize=8, frameon=False)
    axis.grid(alpha=0.16)


def _draw_metrics(axis: Any, info: dict[str, Any]) -> None:
    mixing = info["mixing"]
    lift = info["lift"]
    spill = info["spill"]
    terms = info["reward_terms"]
    signals = info["reward_signals"]
    labels = ["mix delta", "lift bonus", "spill penalty (nonlinear)"]
    values = np.asarray([terms["mix"], terms["lift"], terms["spill"]], dtype=float)
    colors = ["#2A9D8F", "#4C78A8", "#E76F51"]
    positions = np.arange(len(labels))
    bars = axis.barh(positions, values, color=colors, alpha=0.88)
    axis.set_yticks(positions, labels)
    axis.set_ylim(4.6, -0.7)
    axis.axvline(0.0, color="#374151", linewidth=1.0)
    magnitude = max(float(np.max(np.abs(values))), 0.1)
    label_offset = magnitude * 0.025
    for bar, value in zip(bars, values, strict=True):
        positive = value >= 0.0
        axis.text(
            float(value) + (label_offset if positive else -label_offset),
            bar.get_y() + bar.get_height() * 0.5,
            f"{value:+.3f}",
            ha="left" if positive else "right",
            va="center",
            fontsize=9,
            weight="bold",
        )
    axis.set_xlim(
        min(float(np.min(values)) - magnitude * 0.18, -magnitude * 0.20),
        max(float(np.max(values)) + magnitude * 0.18, magnitude * 0.20),
    )
    axis.set_xlabel("reward contribution")
    axis.set_title(f"D. Reward contributions (total {float(info['final_reward']):+.3f})")
    axis.grid(axis="x", alpha=0.18)
    motion_cost = sum(float(terms[name]) for name in ("jerk", "acceleration", "height"))
    summary = (
        f"mix {float(mixing['initial_mixing_score']):.3f} → "
        f"{float(mixing['final_mixing_score']):.3f} "
        f"(Δ {float(mixing['mixing_improvement']):+.3f})\n"
        f"lifted {int(lift['lifted_particle_count'])}/{int(info['particle_count'])} "
        f"({float(lift['lifted_particle_ratio']):.1%}); "
        f"{float(signals['lift_reward_per_particle']):.2f}/grain; "
        f"max grain-top clearance "
        f"{float(lift['maximum_grain_top_clearance_m']) * 1000.0:.1f} mm\n"
        f"spill {int(spill['spill_count'])}/{int(info['particle_count'])}; mass "
        f"{float(spill['spill_mass_ratio']):.1%}; severity "
        f"{float(signals['spill_severity']):.3f} "
        f"(linear {float(signals['spill_linear_penalty']):.3f} + "
        f"quadratic {float(signals['spill_quadratic_penalty']):.3f})\n"
        f"motion cost {motion_cost:+.3f}; total reward "
        f"{float(info['final_reward']):+.3f}"
    )
    axis.text(
        0.02,
        0.03,
        summary,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "alpha": 0.88,
            "edgecolor": "#D1D5DB",
        },
    )


def _reward_npz_scalars(info: dict[str, Any]) -> dict[str, float | int]:
    """대표 rollout의 reward 입력과 기여도를 pickle 없는 scalar로 만든다."""

    mixing = info["mixing"]
    lift = info["lift"]
    spill = info["spill"]
    terms = info["reward_terms"]
    signals = info["reward_signals"]
    return {
        "initial_mixing_score": float(mixing["initial_mixing_score"]),
        "final_mixing_score": float(mixing["final_mixing_score"]),
        "mixing_improvement": float(mixing["mixing_improvement"]),
        "lift_score": float(lift["lift_score"]),
        "lifted_particle_count": int(lift["lifted_particle_count"]),
        "lifted_particle_ratio": float(lift["lifted_particle_ratio"]),
        "peak_lifted_particle_ratio": float(lift["peak_lifted_particle_ratio"]),
        "lift_top_margin_m": float(lift["top_margin_m"]),
        "lift_reward_per_particle": float(signals["lift_reward_per_particle"]),
        "maximum_grain_top_clearance_m": float(lift["maximum_grain_top_clearance_m"]),
        "spill_count": int(spill["spill_count"]),
        "spill_count_ratio": float(spill["spill_count_ratio"]),
        "spill_mass_kg": float(spill["spill_mass_kg"]),
        "spill_mass_ratio": float(spill["spill_mass_ratio"]),
        "spill_severity": float(signals["spill_severity"]),
        "spill_linear_penalty": float(signals["spill_linear_penalty"]),
        "spill_quadratic_penalty": float(signals["spill_quadratic_penalty"]),
        "reward_term_mix": float(terms["mix"]),
        "reward_term_lift": float(terms["lift"]),
        "reward_term_spill": float(terms["spill"]),
        "reward_term_jerk": float(terms["jerk"]),
        "reward_term_acceleration": float(terms["acceleration"]),
        "reward_term_height": float(terms["height"]),
        "reward_term_invalid": float(terms["invalid"]),
        "reward_total": float(info["final_reward"]),
    }


def _preview_policy_action(
    environment: WokMixingEnv | Any,
    observation: np.ndarray,
    checkpoint_path: Path | None,
) -> tuple[np.ndarray, str, Path | None]:
    """center 4D action 또는 CPU deterministic SAC action을 선택한다."""

    expected_shape = environment.action_space.shape
    if expected_shape is None or len(expected_shape) != 1:
        raise ValueError("teaching preview에는 1차원 연속 action space가 필요합니다.")
    if tuple(expected_shape) != (4,):
        raise ValueError(
            "fried-rice teaching preview에는 4D action space가 필요합니다: "
            f"environment={tuple(expected_shape)}"
        )
    if checkpoint_path is None:
        return (
            np.zeros(expected_shape, dtype=np.float32),
            "center_zero_action",
            None,
        )

    candidate = Path(checkpoint_path)
    checkpoint = candidate if candidate.is_file() else candidate.with_suffix(".zip")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAC checkpoint를 찾을 수 없습니다: {checkpoint_path}")
    model = SAC.load(checkpoint, device="cpu")
    model_shape = getattr(getattr(model, "action_space", None), "shape", None)
    if model_shape is not None and tuple(model_shape) != tuple(expected_shape):
        raise ValueError(
            "SAC checkpoint action shape이 현재 환경과 다릅니다: "
            f"checkpoint={tuple(model_shape)}, environment={tuple(expected_shape)}"
        )
    predicted, _ = model.predict(observation, deterministic=True)
    action = np.asarray(predicted, dtype=np.float32)
    if action.shape != expected_shape:
        raise ValueError(
            "SAC checkpoint가 잘못된 action shape을 반환했습니다: "
            f"predicted={action.shape}, environment={expected_shape}"
        )
    if not np.isfinite(action).all():
        raise ValueError("SAC checkpoint action에 NaN 또는 inf가 있습니다.")
    return action, "sac_checkpoint", checkpoint.resolve()


def run_teaching_preview(
    config_path: Path,
    output_directory: Path,
    *,
    seed: int = 1,
    count_per_type: int = 20,
    interval_s: float = 0.1,
    checkpoint_path: Path | None = None,
) -> Path:
    """중앙 action 또는 final SAC policy를 실제 MuJoCo에서 정확히 한 번 실행한다."""

    config = load_config(config_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    environment = WokMixingEnv(config)
    try:
        observation, reset_info = environment.reset(
            seed=seed,
            options={"count_per_type": count_per_type},
        )
        normalized_action, policy_source, resolved_checkpoint = _preview_policy_action(
            environment,
            observation,
            checkpoint_path,
        )
        _, reward, terminated, truncated, info = environment.step(normalized_action)
    finally:
        environment.close()
    if not terminated or truncated:
        raise RuntimeError("검토 rollout은 one-step terminated episode여야 합니다.")
    if not bool(info.get("trajectory_valid", False)):
        raise RuntimeError(f"검토 trajectory가 유효하지 않습니다: {info.get('invalid_reasons')}")

    result = info["simulation_result"]
    trajectory = info["trajectory"]
    raw_time = np.asarray(result["time_s"], dtype=float)
    raw_pan = np.asarray(result["pan_position_world_m"], dtype=float)
    raw_pan_quaternion = np.asarray(result["pan_quaternion_wxyz"], dtype=float)
    sampled_time, sampled_pan, sampled_particles = resample_trajectory_history(
        raw_time,
        raw_pan,
        np.asarray(result["particle_positions_world_m"]),
        interval_s=interval_s,
    )
    local_rim_endpoints, rim_center_radius, rim_z = pan_rim_geometry_from_config(config)
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
    local_side_profile = pan_side_profile_local(config)
    raw_side_profiles = transform_pan_local_history(
        raw_pan,
        raw_pan_quaternion,
        local_side_profile,
    )
    sampled_side_profiles = resample_point_history(
        raw_time,
        raw_side_profiles,
        sampled_time,
    )
    pan_local = np.asarray(result["particle_positions_pan_m"], dtype=float)
    initial = pan_local[0]
    final = pan_local[-1]
    species = np.asarray(info["particle_batch"]["species"], dtype=str)
    initial_front_fraction = float(np.mean(initial[:, 0] >= 0.0))
    final_front_fraction = float(np.mean(final[:, 0] >= 0.0))
    front_metrics = {
        "initial_front_fraction": initial_front_fraction,
        "final_front_fraction": final_front_fraction,
        "centroid_x_shift_m": float(np.mean(final[:, 0]) - np.mean(initial[:, 0])),
    }

    figure, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    pan_radius = float(config["mixing"]["pan_radius_m"])
    _draw_world_xz(
        axes[0, 0],
        sampled_time,
        sampled_pan,
        sampled_particles,
        species,
        pan_radius,
        sampled_rim_endpoints,
        sampled_side_profiles,
    )
    _draw_teaching_geometry(
        axes[0, 1],
        trajectory,
        pan_radius,
        sampled_time_s=sampled_time,
        sampled_pan=sampled_pan,
        sampled_rim_endpoints=sampled_rim_endpoints,
        sampled_pan_side_profiles=sampled_side_profiles,
    )
    _draw_final_top_view(axes[1, 0], initial, final, species, pan_radius)
    _draw_metrics(axes[1, 1], info)
    action = info["action_parameters"]
    lift_angle = float(action.get("lift_angle", action.get("tilt_recovery_angle", 0.0)))
    figure.suptitle(
        "Teaching preview — ONE rollout only | "
        f"{int(info['particle_count'])} particles | "
        f"descent {np.rad2deg(action['descent_angle']):.1f}° | "
        f"pitch {np.rad2deg(action['pan_tilt_angle']):.1f}° | "
        f"lift {np.rad2deg(lift_angle):.1f}° | "
        f"speed {float(action['descent_speed']):.3f} m/s\n"
        f"source {policy_source} | trajectory dots sampled every {interval_s:.1f} s",
        fontsize=15,
        weight="bold",
    )
    image_path = output_directory / "teaching_preview_60_particles.png"
    figure.savefig(image_path, dpi=180)
    plt.close(figure)

    data_path = output_directory / "teaching_preview_60_particles.npz"
    np.savez_compressed(
        data_path,
        time_s=sampled_time,
        pan_position_world_m=sampled_pan,
        pan_rim_endpoints_world_m=sampled_rim_endpoints,
        pan_rim_endpoint_order=np.asarray(PAN_RIM_ENDPOINT_ORDER),
        pan_rim_center_radius_m=rim_center_radius,
        pan_rim_z_m=rim_z,
        pan_side_profile_world_m=sampled_side_profiles,
        particle_positions_world_m=sampled_particles,
        particle_positions_pan_initial_m=initial,
        particle_positions_pan_final_m=final,
        species=species,
        normalized_action=normalized_action,
        sample_interval_s=interval_s,
        seed=seed,
        count_per_type=count_per_type,
        policy_source=np.asarray(policy_source),
        policy_checkpoint=np.asarray(
            "" if resolved_checkpoint is None else str(resolved_checkpoint)
        ),
        **_reward_npz_scalars(info),
    )
    summary = {
        "execution": "one_rollout_only",
        "training_started": False,
        "policy_source": policy_source,
        "checkpoint": (None if resolved_checkpoint is None else str(resolved_checkpoint)),
        "config": str(config_path.resolve()),
        "image": str(image_path.resolve()),
        "data": str(data_path.resolve()),
        "seed": seed,
        "count_per_type": count_per_type,
        "total_particle_count": int(info["particle_count"]),
        "actual_total_mass_g": float(reset_info["actual_total_mass_kg"]) * 1000.0,
        "sample_interval_s": interval_s,
        "sample_count": len(sampled_time),
        "duration_s": float(sampled_time[-1]),
        "pan_rim_endpoint_order": list(PAN_RIM_ENDPOINT_ORDER),
        "pan_rim_center_radius_m": rim_center_radius,
        "pan_rim_z_m": rim_z,
        "pan_side_profile": "closed_flat_bottom_inner_outer_circular_arcs",
        "pan_side_profile_point_count": len(local_side_profile),
        "normalized_action": normalized_action,
        "action_parameters": action,
        "mixing": info["mixing"],
        "spill": info["spill"],
        "lift": info["lift"],
        "reward_terms": info["reward_terms"],
        "reward_signals": info["reward_signals"],
        "reward_total": float(info["final_reward"]),
        "front_accumulation": front_metrics,
        "final_reward": float(reward),
    }
    summary_path = output_directory / "teaching_preview_summary.json"
    summary_path.write_text(
        json.dumps(_json_value(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return image_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/fried_rice.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fried_rice/teaching_preview"),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--count-per-type", type=int, default=20)
    parser.add_argument("--interval-s", type=float, default=0.1)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="생략하면 4D center zero action, 지정하면 CPU deterministic SAC policy",
    )
    arguments = parser.parse_args()
    output = run_teaching_preview(
        arguments.config,
        arguments.output,
        seed=arguments.seed,
        count_per_type=arguments.count_per_type,
        interval_s=arguments.interval_s,
        checkpoint_path=arguments.checkpoint,
    )
    print(
        json.dumps(
            {"preview": str(output), "output_directory": str(arguments.output)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
