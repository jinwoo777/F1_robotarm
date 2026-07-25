"""Training 또는 명시 physical action을 actual-time X-Z GIF로 저장한다."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from wok_sim.config import load_config
from wok_sim.envs import WokMixingEnv

from .pan_profile import (
    pan_side_profile_local,
    resample_point_history,
    transform_pan_local_history,
)
from .trajectory_plot import (
    PAN_CENTER_COLOR,
    PAN_RIM_COLOR,
    PAN_RIM_ENDPOINT_ORDER,
    pan_rim_endpoint_history,
    pan_rim_geometry_from_config,
    resample_pan_rim_endpoints,
    resample_trajectory_history,
)

ACTION_COLUMNS = (
    "descent_angle_rad",
    "pan_tilt_angle_rad",
    "descent_speed_m_s",
    "lift_angle_rad",
)
ACTION_RANGE_KEYS = (
    "descent_angle_range_rad",
    "pan_tilt_angle_range_rad",
    "descent_speed_range_m_s",
    "lift_angle_range_rad",
)
SPECIES_COLORS = {
    "large_sphere": "#E76F51",
    "small_sphere": "#2A9D8F",
    "ellipsoid": "#457B9D",
}


@dataclass(frozen=True, slots=True)
class EpisodeReplaySelection:
    """CSV 한 행에서 복원한 episode 조건과 정확한 물리 action."""

    episode_id: int
    random_seed: int
    count_per_type: int
    particle_count: int
    final_reward: float
    physical_action: tuple[float, float, float, float]
    actual_total_mass_kg: float | None
    nominal_joint_speed_target_fraction: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "random_seed": self.random_seed,
            "count_per_type": self.count_per_type,
            "particle_count": self.particle_count,
            "final_reward": self.final_reward,
            "actual_total_mass_kg": self.actual_total_mass_kg,
            "nominal_joint_speed_target_fraction": (
                self.nominal_joint_speed_target_fraction
            ),
            "physical_action": dict(zip(ACTION_COLUMNS, self.physical_action, strict=True)),
        }


def _required_float(row: Mapping[str, str], name: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"episode CSV의 {name} 값이 숫자가 아닙니다.") from exc
    if not np.isfinite(value):
        raise ValueError(f"episode CSV의 {name} 값은 유한해야 합니다.")
    return value


def _required_int(row: Mapping[str, str], name: str) -> int:
    value = _required_float(row, name)
    parsed = int(value)
    if parsed != value:
        raise ValueError(f"episode CSV의 {name} 값은 정수여야 합니다.")
    return parsed


def _trajectory_valid(row: Mapping[str, str]) -> bool:
    raw = str(row.get("trajectory_valid", "true")).strip().lower()
    return raw in {"1", "true", "yes"}


def _selection_from_row(row: Mapping[str, str]) -> EpisodeReplaySelection:
    if not _trajectory_valid(row):
        raise ValueError("trajectory_valid가 아닌 episode는 GIF로 재실행할 수 없습니다.")
    count_per_type = _required_int(row, "count_per_type")
    particle_count = _required_int(row, "particle_count")
    if particle_count != 3 * count_per_type:
        raise ValueError(
            "fried-rice particle_count는 count_per_type의 3배여야 합니다: "
            f"{particle_count} != 3 * {count_per_type}"
        )
    mass_raw = str(row.get("actual_total_mass_kg", "")).strip()
    mass = None if not mass_raw else _required_float(row, "actual_total_mass_kg")
    speed_target_raw = str(
        row.get("nominal_joint_speed_target_fraction", "")
    ).strip()
    speed_target = (
        None
        if not speed_target_raw
        else _required_float(row, "nominal_joint_speed_target_fraction")
    )
    if speed_target is not None and not 0.0 < speed_target <= 1.0:
        raise ValueError("nominal_joint_speed_target_fraction은 (0,1]이어야 합니다.")
    return EpisodeReplaySelection(
        episode_id=_required_int(row, "episode_id"),
        random_seed=_required_int(row, "random_seed"),
        count_per_type=count_per_type,
        particle_count=particle_count,
        final_reward=_required_float(row, "final_reward"),
        physical_action=tuple(_required_float(row, name) for name in ACTION_COLUMNS),
        actual_total_mass_kg=mass,
        nominal_joint_speed_target_fraction=speed_target,
    )


def select_episode_from_csv(
    episodes_csv: Path,
    *,
    episode_id: int | None = None,
    episode_start: int = 144,
    episode_end: int = 149,
) -> EpisodeReplaySelection:
    """명시 episode 또는 범위 안에서 final_reward가 가장 높은 완료 행을 고른다."""

    path = Path(episodes_csv)
    if not path.is_file():
        raise FileNotFoundError(f"episodes.csv를 찾을 수 없습니다: {path}")
    if episode_id is None and episode_start > episode_end:
        raise ValueError("episode_start는 episode_end 이하여야 합니다.")
    candidates: list[EpisodeReplaySelection] = []
    malformed_ids: list[int] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("episode CSV header가 없습니다.")
        required = {
            "episode_id",
            "random_seed",
            "count_per_type",
            "particle_count",
            "final_reward",
            "trajectory_valid",
            *ACTION_COLUMNS,
        }
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"episode CSV 필수 열이 없습니다: {', '.join(missing)}")
        for row in reader:
            try:
                row_episode_id = _required_int(row, "episode_id")
            except ValueError:
                continue
            selected = (
                row_episode_id == episode_id
                if episode_id is not None
                else episode_start <= row_episode_id <= episode_end
            )
            if not selected:
                continue
            try:
                candidates.append(_selection_from_row(row))
            except ValueError:
                malformed_ids.append(row_episode_id)
    if not candidates:
        target = (
            f"episode {episode_id}"
            if episode_id is not None
            else f"episode 범위 [{episode_start}, {episode_end}]"
        )
        malformed = f"; 불완전/invalid 행={malformed_ids}" if malformed_ids else ""
        raise ValueError(f"{target}에서 재실행 가능한 완료 행을 찾지 못했습니다{malformed}")
    if episode_id is not None:
        if len(candidates) != 1:
            raise ValueError(f"episode_id={episode_id} 행이 중복되어 있습니다.")
        return candidates[0]
    return max(candidates, key=lambda item: (item.final_reward, item.episode_id))


def normalized_action_from_selection(
    selection: EpisodeReplaySelection,
    config: Mapping[str, Any],
) -> np.ndarray:
    """CSV의 네 물리 action을 현재 config의 normalized 4D action으로 역변환한다."""

    return normalized_action_from_physical_action(selection.physical_action, config)


def normalized_action_from_physical_action(
    physical_action: Sequence[float],
    config: Mapping[str, Any],
) -> np.ndarray:
    """네 물리 action을 현재 config의 normalized 4D action으로 역변환한다."""

    try:
        values = tuple(float(value) for value in physical_action)
    except (TypeError, ValueError) as exc:
        raise ValueError("physical_action은 네 개의 숫자여야 합니다.") from exc
    if len(values) != len(ACTION_COLUMNS) or not np.isfinite(values).all():
        raise ValueError("physical_action은 유한한 네 개의 숫자여야 합니다.")
    trajectory = config.get("trajectory", {})
    profile = trajectory.get("fried_rice", {}) if isinstance(trajectory, Mapping) else {}
    normalized: list[float] = []
    for physical, range_key in zip(
        values,
        ACTION_RANGE_KEYS,
        strict=True,
    ):
        try:
            low, high = (float(item) for item in profile[range_key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"trajectory.fried_rice.{range_key} [min,max]가 필요합니다.") from exc
        if not np.isfinite([low, high]).all() or high <= low:
            raise ValueError(f"trajectory.fried_rice.{range_key} 범위가 유효하지 않습니다.")
        value = 2.0 * (physical - low) / (high - low) - 1.0
        if value < -1.0 - 1.0e-6 or value > 1.0 + 1.0e-6:
            raise ValueError(
                f"physical action {range_key}={physical}이 config 범위 [{low}, {high}] 밖입니다."
            )
        normalized.append(float(np.clip(value, -1.0, 1.0)))
    return np.asarray(normalized, dtype=np.float32)


def _pan_outline_local(config: Mapping[str, Any]) -> np.ndarray:
    """Compatibility wrapper for the closed curved display profile."""

    return pan_side_profile_local(config)


def trajectory_dot_times(
    time_s: np.ndarray,
    *,
    interval_s: float = 0.1,
) -> np.ndarray:
    """Return an exact fixed-time grid, appending the final time when needed."""

    time = np.asarray(time_s, dtype=float)
    if time.ndim != 1 or len(time) < 2 or np.any(np.diff(time) <= 0.0):
        raise ValueError("time_s는 엄격히 증가하는 2개 이상의 1차원 배열이어야 합니다.")
    if not np.isfinite(time).all():
        raise ValueError("time_s는 모두 유한해야 합니다.")
    if not np.isfinite(interval_s) or interval_s <= 0.0:
        raise ValueError("interval_s는 유한한 양수여야 합니다.")
    duration = float(time[-1] - time[0])
    step_count = int(np.floor(duration / interval_s + 1.0e-12))
    samples = time[0] + np.arange(step_count + 1, dtype=float) * interval_s
    if samples[-1] < time[-1] - 1.0e-12:
        samples = np.append(samples, time[-1])
    else:
        samples[-1] = time[-1]
    return samples


def _fixed_xz_limits(*arrays: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    points = np.concatenate([np.asarray(item).reshape(-1, 3) for item in arrays], axis=0)
    if not np.isfinite(points).all():
        raise ValueError("GIF history에 NaN 또는 inf가 있습니다.")
    x_min, z_min = np.min(points[:, (0, 2)], axis=0)
    x_max, z_max = np.max(points[:, (0, 2)], axis=0)
    span = max(float(x_max - x_min), float(z_max - z_min), 0.1)
    margin = max(0.025, span * 0.07)
    return (float(x_min - margin), float(x_max + margin)), (
        float(z_min - margin),
        float(z_max + margin),
    )


def render_xz_gif(
    output_path: Path,
    *,
    time_s: np.ndarray,
    pan_position_world_m: np.ndarray,
    pan_rim_endpoints_world_m: np.ndarray,
    pan_outline_world_m: np.ndarray,
    particle_positions_world_m: np.ndarray,
    species: Sequence[str] | np.ndarray,
    fps: int = 20,
    trajectory_dot_interval_s: float = 0.1,
    title: str = "Wok episode replay",
    focus_on_pan_motion: bool = False,
) -> dict[str, float | int]:
    """표본화된 X-Z state와 누적 궤적 점을 actual-time GIF로 저장한다."""

    time = np.asarray(time_s, dtype=float)
    pan = np.asarray(pan_position_world_m, dtype=float)
    endpoints = np.asarray(pan_rim_endpoints_world_m, dtype=float)
    outline = np.asarray(pan_outline_world_m, dtype=float)
    particles = np.asarray(particle_positions_world_m, dtype=float)
    species_array = np.asarray(species, dtype=str)
    if isinstance(fps, bool) or int(fps) != fps or not 1 <= int(fps) <= 60:
        raise ValueError("fps는 1~60 범위의 정수여야 합니다.")
    fps = int(fps)
    if time.ndim != 1 or len(time) < 2 or np.any(np.diff(time) <= 0.0):
        raise ValueError("time_s는 엄격히 증가하는 2개 이상의 1차원 배열이어야 합니다.")
    if pan.shape != (len(time), 3):
        raise ValueError("pan_position_world_m shape은 (T,3)이어야 합니다.")
    if endpoints.shape != (len(time), 2, 3):
        raise ValueError("pan_rim_endpoints_world_m shape은 (T,2,3)이어야 합니다.")
    if outline.ndim != 3 or outline.shape[0] != len(time) or outline.shape[2] != 3:
        raise ValueError("pan_outline_world_m shape은 (T,P,3)이어야 합니다.")
    if particles.ndim != 3 or particles.shape[0] != len(time) or particles.shape[2] != 3:
        raise ValueError("particle_positions_world_m shape은 (T,N,3)이어야 합니다.")
    if species_array.shape != (particles.shape[1],):
        raise ValueError("species shape은 particle 수와 같은 (N,)이어야 합니다.")
    if not all(np.isfinite(item).all() for item in (time, pan, endpoints, outline, particles)):
        raise ValueError("GIF 입력 history는 모두 유한해야 합니다.")
    dot_time = trajectory_dot_times(time, interval_s=trajectory_dot_interval_s)
    dot_pan = resample_point_history(time, pan, dot_time)
    dot_endpoints = resample_point_history(time, endpoints, dot_time)

    output = Path(output_path)
    if output.suffix.lower() != ".gif":
        raise ValueError("output_path 확장자는 .gif여야 합니다.")
    output.parent.mkdir(parents=True, exist_ok=True)
    limit_arrays = (
        (pan, endpoints, outline, particles[:1])
        if focus_on_pan_motion
        else (particles, pan, endpoints, outline)
    )
    x_limits, z_limits = _fixed_xz_limits(*limit_arrays)
    particle_colors = [SPECIES_COLORS.get(name, "#6C757D") for name in species_array]

    figure, axis = plt.subplots(figsize=(7.2, 5.4), dpi=90, constrained_layout=True)
    particle_artist = axis.scatter(
        particles[0, :, 0],
        particles[0, :, 2],
        c=particle_colors,
        s=10,
        alpha=0.82,
        linewidths=0,
        rasterized=True,
    )
    (center_trail_line,) = axis.plot(
        dot_pan[:1, 0],
        dot_pan[:1, 2],
        color=PAN_CENTER_COLOR,
        linewidth=1.2,
        alpha=0.62,
        zorder=2,
    )
    center_trail_dots = axis.scatter(
        dot_pan[:1, 0],
        dot_pan[:1, 2],
        color=PAN_CENTER_COLOR,
        s=9,
        alpha=0.72,
        linewidths=0,
        zorder=3,
    )
    rim_trail_lines: list[Any] = []
    rim_trail_dots: list[Any] = []
    for endpoint_index in range(2):
        (trail_line,) = axis.plot(
            dot_endpoints[:1, endpoint_index, 0],
            dot_endpoints[:1, endpoint_index, 2],
            color=PAN_RIM_COLOR,
            linewidth=1.0,
            alpha=0.55,
            zorder=2,
        )
        trail_dots = axis.scatter(
            dot_endpoints[:1, endpoint_index, 0],
            dot_endpoints[:1, endpoint_index, 2],
            color=PAN_RIM_COLOR,
            s=8,
            alpha=0.62,
            linewidths=0,
            zorder=3,
        )
        rim_trail_lines.append(trail_line)
        rim_trail_dots.append(trail_dots)
    (outline_artist,) = axis.plot(
        outline[0, :, 0],
        outline[0, :, 2],
        color=PAN_RIM_COLOR,
        linewidth=2.8,
        solid_capstyle="round",
    )
    rim_artist = axis.scatter(
        endpoints[0, :, 0],
        endpoints[0, :, 2],
        color=PAN_RIM_COLOR,
        s=42,
        edgecolors="white",
        linewidths=0.7,
        zorder=5,
    )
    center_artist = axis.scatter(
        [pan[0, 0]],
        [pan[0, 2]],
        color=PAN_CENTER_COLOR,
        s=48,
        edgecolors="white",
        linewidths=0.7,
        zorder=6,
    )
    clock_artist = axis.text(
        0.02,
        0.97,
        "",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.82},
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=PAN_CENTER_COLOR,
            label=f"pan center + {trajectory_dot_interval_s:.1f} s trail",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=PAN_RIM_COLOR,
            markerfacecolor=PAN_RIM_COLOR,
            label=f"pan outline + rear/front {trajectory_dot_interval_s:.1f} s trails",
        ),
    ]
    for name in sorted(set(species_array)):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=SPECIES_COLORS.get(name, "#6C757D"),
                label=name.replace("_", " "),
            )
        )
    axis.legend(handles=legend_handles, loc="lower left", fontsize=8, frameon=False)
    axis.set_xlim(*x_limits)
    axis.set_ylim(*z_limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("world x (m) → front")
    axis.set_ylabel("world z (m)")
    axis.set_title(title)
    axis.grid(alpha=0.18)

    def update(frame_index: int) -> tuple[Any, ...]:
        visible_dot_count = int(
            np.searchsorted(dot_time, time[frame_index] + 1.0e-12, side="right")
        )
        particle_artist.set_offsets(particles[frame_index][:, (0, 2)])
        center_trail_line.set_data(
            dot_pan[:visible_dot_count, 0],
            dot_pan[:visible_dot_count, 2],
        )
        center_trail_dots.set_offsets(dot_pan[:visible_dot_count][:, (0, 2)])
        for endpoint_index, (trail_line, trail_dots) in enumerate(
            zip(rim_trail_lines, rim_trail_dots, strict=True)
        ):
            trail_line.set_data(
                dot_endpoints[:visible_dot_count, endpoint_index, 0],
                dot_endpoints[:visible_dot_count, endpoint_index, 2],
            )
            trail_dots.set_offsets(
                dot_endpoints[:visible_dot_count, endpoint_index][:, (0, 2)]
            )
        outline_artist.set_data(
            outline[frame_index, :, 0],
            outline[frame_index, :, 2],
        )
        rim_artist.set_offsets(endpoints[frame_index][:, (0, 2)])
        center_artist.set_offsets(np.asarray([[pan[frame_index, 0], pan[frame_index, 2]]]))
        clock_artist.set_text(f"t = {time[frame_index]:5.2f} / {time[-1]:5.2f} s")
        return (
            particle_artist,
            center_trail_line,
            center_trail_dots,
            *rim_trail_lines,
            *rim_trail_dots,
            outline_artist,
            rim_artist,
            center_artist,
            clock_artist,
        )

    animation = FuncAnimation(
        figure,
        update,
        frames=len(time),
        interval=1000.0 / fps,
        blit=True,
        repeat=False,
    )
    try:
        animation.save(
            output,
            writer=PillowWriter(
                fps=fps,
                metadata={"title": title, "artist": "wok-sim"},
            ),
            dpi=90,
        )
    finally:
        plt.close(figure)
    gif_duration = len(time) / fps
    motion_playback_duration = (len(time) - 1) / fps
    simulation_duration = float(time[-1] - time[0])
    return {
        "fps": fps,
        "frame_count": len(time),
        "trajectory_dot_interval_s": float(trajectory_dot_interval_s),
        "trajectory_dot_count": len(dot_time),
        "simulation_duration_s": simulation_duration,
        "motion_playback_duration_s": motion_playback_duration,
        "gif_playback_duration_s": gif_duration,
        "final_frame_hold_s": 1.0 / fps,
        "playback_speed_ratio": (
            simulation_duration / motion_playback_duration
            if motion_playback_duration > 0.0
            else 1.0
        ),
        "focus_on_pan_motion": bool(focus_on_pan_motion),
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class _ReplayRun:
    normalized_action: np.ndarray
    requested_physical_action: tuple[float, float, float, float]
    reset_info: Mapping[str, Any]
    info: Mapping[str, Any]
    replay_reward: float
    frame_time_s: np.ndarray
    frame_pan_world_m: np.ndarray
    frame_rim_endpoints_world_m: np.ndarray
    frame_pan_outline_world_m: np.ndarray
    frame_particles_world_m: np.ndarray
    species: np.ndarray
    frame_interval_s: float
    rim_center_radius_m: float
    rim_z_m: float
    profile_point_count: int
    trajectory_motion_profile: str


def _validated_fps(fps: int) -> int:
    if isinstance(fps, bool) or int(fps) != fps or not 1 <= int(fps) <= 60:
        raise ValueError("fps는 1~60 범위의 정수여야 합니다.")
    return int(fps)


def _run_action_replay(
    config: Mapping[str, Any],
    physical_action: Sequence[float],
    *,
    seed: int,
    count_per_type: int,
    expected_particle_count: int,
    fps: int,
    curriculum_episode: int | None = None,
    nominal_joint_speed_target_fraction: float | None = None,
) -> _ReplayRun:
    fps = _validated_fps(fps)
    if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
        raise ValueError("seed는 0 이상의 정수여야 합니다.")
    if (
        isinstance(count_per_type, bool)
        or int(count_per_type) != count_per_type
        or int(count_per_type) <= 0
    ):
        raise ValueError("count_per_type은 양의 정수여야 합니다.")
    requested_action = tuple(float(value) for value in physical_action)
    normalized_action = normalized_action_from_physical_action(requested_action, config)
    environment = WokMixingEnv(config)
    try:
        reset_options = {"count_per_type": int(count_per_type)}
        if curriculum_episode is not None:
            if (
                isinstance(curriculum_episode, bool)
                or int(curriculum_episode) != curriculum_episode
                or int(curriculum_episode) < 0
            ):
                raise ValueError("curriculum_episode은 0 이상의 정수여야 합니다.")
            reset_options["curriculum_episode"] = int(curriculum_episode)
        if nominal_joint_speed_target_fraction is not None:
            speed_target = float(nominal_joint_speed_target_fraction)
            if not np.isfinite(speed_target) or not 0.0 < speed_target <= 1.0:
                raise ValueError("nominal_joint_speed_target_fraction은 (0,1]이어야 합니다.")
            reset_options["nominal_joint_speed_target_fraction"] = speed_target
        _, reset_info = environment.reset(
            seed=int(seed),
            options=reset_options,
        )
        _, replay_reward, terminated, truncated, info = environment.step(normalized_action)
    finally:
        environment.close()
    if not terminated or truncated:
        raise RuntimeError("GIF replay는 one-step terminated episode여야 합니다.")
    if not bool(info.get("trajectory_valid", False)):
        raise RuntimeError(f"replay trajectory가 유효하지 않습니다: {info.get('invalid_reasons')}")
    if int(info["particle_count"]) != int(expected_particle_count):
        raise RuntimeError(
            "replay particle_count가 요청 조건과 다릅니다: "
            f"{info['particle_count']} != {expected_particle_count}"
        )

    result = info["simulation_result"]
    raw_time = np.asarray(result["time_s"], dtype=float)
    raw_pan = np.asarray(result["pan_position_world_m"], dtype=float)
    raw_quaternion = np.asarray(result["pan_quaternion_wxyz"], dtype=float)
    frame_interval = 1.0 / fps
    frame_time, frame_pan, frame_particles = resample_trajectory_history(
        raw_time,
        raw_pan,
        np.asarray(result["particle_positions_world_m"], dtype=float),
        interval_s=frame_interval,
    )
    local_endpoints, rim_center_radius, rim_z = pan_rim_geometry_from_config(config)
    raw_endpoints = pan_rim_endpoint_history(raw_pan, raw_quaternion, local_endpoints)
    frame_endpoints = resample_pan_rim_endpoints(raw_time, raw_endpoints, frame_time)
    local_outline = _pan_outline_local(config)
    raw_outline = transform_pan_local_history(
        raw_pan,
        raw_quaternion,
        local_outline,
    )
    frame_outline = resample_point_history(raw_time, raw_outline, frame_time)
    species = np.asarray(info["particle_batch"]["species"], dtype=str)
    trajectory_section = config.get("trajectory", {})
    fried_rice_section = (
        trajectory_section.get("fried_rice", {})
        if isinstance(trajectory_section, Mapping)
        else {}
    )
    motion_profile = str(
        fried_rice_section.get("motion_profile", "phasewise_minimum_jerk")
        if isinstance(fried_rice_section, Mapping)
        else "phasewise_minimum_jerk"
    )
    return _ReplayRun(
        normalized_action=normalized_action,
        requested_physical_action=requested_action,
        reset_info=reset_info,
        info=info,
        replay_reward=float(replay_reward),
        frame_time_s=frame_time,
        frame_pan_world_m=frame_pan,
        frame_rim_endpoints_world_m=frame_endpoints,
        frame_pan_outline_world_m=frame_outline,
        frame_particles_world_m=frame_particles,
        species=species,
        frame_interval_s=frame_interval,
        rim_center_radius_m=rim_center_radius,
        rim_z_m=rim_z,
        profile_point_count=len(local_outline),
        trajectory_motion_profile=motion_profile,
    )


def _replayed_physical_action(run: _ReplayRun) -> dict[str, float]:
    action = run.info["action_parameters"]
    return {
        "descent_angle_rad": float(action["descent_angle"]),
        "pan_tilt_angle_rad": float(action["pan_tilt_angle"]),
        "descent_speed_m_s": float(action["descent_speed"]),
        "lift_angle_rad": float(action["lift_angle"]),
    }


def _write_replay_gif(
    config_path: Path,
    output_path: Path,
    run: _ReplayRun,
    *,
    fps: int,
    title: str,
    metadata_fields: Mapping[str, Any],
    replay_metadata_fields: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    fps = _validated_fps(fps)
    gif_path = Path(output_path)
    timing_metadata = render_xz_gif(
        gif_path,
        time_s=run.frame_time_s,
        pan_position_world_m=run.frame_pan_world_m,
        pan_rim_endpoints_world_m=run.frame_rim_endpoints_world_m,
        pan_outline_world_m=run.frame_pan_outline_world_m,
        particle_positions_world_m=run.frame_particles_world_m,
        species=run.species,
        fps=fps,
        trajectory_dot_interval_s=0.1,
        title=title,
        focus_on_pan_motion=run.info.get("joint_speed_report") is not None,
    )
    replay_mass = float(run.reset_info["actual_total_mass_kg"])
    replay_metadata = {
        "random_seed": int(run.info["random_seed"]),
        "particle_count": int(run.info["particle_count"]),
        "count_per_type": int(run.info["count_per_type"]),
        "actual_total_mass_kg": replay_mass,
        "nominal_joint_speed_target_fraction": run.info.get(
            "nominal_joint_speed_target_fraction"
        ),
        "final_reward": run.replay_reward,
        "trajectory_valid": bool(run.info["trajectory_valid"]),
    }
    if replay_metadata_fields is not None:
        replay_metadata.update(replay_metadata_fields)
    metadata = {
        "gif": str(gif_path.resolve()),
        "config": str(Path(config_path).resolve()),
        **dict(metadata_fields),
        "normalized_action": run.normalized_action,
        "replayed_physical_action": _replayed_physical_action(run),
        "replay": replay_metadata,
        "trajectory": {
            "motion_profile": run.trajectory_motion_profile,
            "valid": bool(run.info["trajectory_valid"]),
            "validation": run.info.get("trajectory_validation", {}),
        },
        "m0609_nominal_joint_speed": run.info.get("joint_speed_report"),
        "timing": timing_metadata,
        "frame_sample_interval_s": run.frame_interval_s,
        "trajectory_dot_interval_s": 0.1,
        "pan_rim_endpoint_order": list(PAN_RIM_ENDPOINT_ORDER),
        "pan_rim_center_radius_m": run.rim_center_radius_m,
        "pan_rim_z_m": run.rim_z_m,
        "render": {
            "plane": "world_xz",
            "pan_center_color": PAN_CENTER_COLOR,
            "pan_rim_color": PAN_RIM_COLOR,
            "actual_time_scale": "1 simulated second = 1 GIF second",
            "pan_profile": "closed_flat_bottom_inner_outer_circular_arcs",
            "pan_profile_point_count": run.profile_point_count,
            "pan_profile_is_display_only": True,
        },
    }
    metadata_path = gif_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(_json_value(metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return gif_path, metadata_path


def generate_episode_gif(
    config_path: Path,
    episodes_csv: Path,
    output_path: Path,
    *,
    episode_id: int | None = None,
    episode_start: int = 144,
    episode_end: int = 149,
    fps: int = 20,
) -> tuple[Path, Path]:
    """CSV에서 episode를 선택하고 동일 조건으로 재실행한 actual-time GIF를 만든다."""

    fps = _validated_fps(fps)
    config = load_config(config_path)
    selection = select_episode_from_csv(
        episodes_csv,
        episode_id=episode_id,
        episode_start=episode_start,
        episode_end=episode_end,
    )
    run = _run_action_replay(
        config,
        selection.physical_action,
        seed=selection.random_seed,
        count_per_type=selection.count_per_type,
        expected_particle_count=selection.particle_count,
        fps=fps,
        curriculum_episode=selection.episode_id,
        nominal_joint_speed_target_fraction=(
            selection.nominal_joint_speed_target_fraction
        ),
    )
    action = run.info["action_parameters"]
    title = (
        f"Episode {selection.episode_id} replay | {selection.particle_count} particles | "
        f"reward {selection.final_reward:+.3f}\n"
        f"descent {np.rad2deg(float(action['descent_angle'])):.1f}° | "
        f"tilt {np.rad2deg(float(action['pan_tilt_angle'])):.1f}° | "
        f"lift {np.rad2deg(float(action['lift_angle'])):.1f}° | "
        f"speed {float(action['descent_speed']):.3f} m/s"
    )
    if selection.nominal_joint_speed_target_fraction is not None:
        title += (
            f" | M0609 target "
            f"{100.0 * selection.nominal_joint_speed_target_fraction:.0f}%"
        )
    replay_physical_action = _replayed_physical_action(run)
    logged_action = dict(zip(ACTION_COLUMNS, selection.physical_action, strict=True))
    action_deltas = {
        name: replay_physical_action[name] - logged_action[name] for name in ACTION_COLUMNS
    }
    replay_mass = float(run.reset_info["actual_total_mass_kg"])
    return _write_replay_gif(
        config_path,
        output_path,
        run,
        fps=fps,
        title=title,
        metadata_fields={
            "source_csv": str(Path(episodes_csv).resolve()),
            "selection_mode": (
                "explicit_episode_id"
                if episode_id is not None
                else "highest_final_reward_in_range"
            ),
            "selection_range": (
                None if episode_id is not None else [episode_start, episode_end]
            ),
            "selected_episode": selection.as_dict(),
            "physical_action_delta_from_csv": action_deltas,
        },
        replay_metadata_fields={
            "reward_delta_from_csv": run.replay_reward - selection.final_reward,
            "mass_delta_from_csv_kg": (
                None
                if selection.actual_total_mass_kg is None
                else replay_mass - selection.actual_total_mass_kg
            ),
        },
    )


def generate_physical_action_gif(
    config_path: Path,
    output_path: Path,
    *,
    seed: int,
    count_per_type: int,
    descent_angle_deg: float,
    pan_tilt_angle_deg: float,
    descent_speed_m_s: float,
    lift_angle_deg: float,
    fps: int = 20,
) -> tuple[Path, Path]:
    """CSV 없이 명시한 physical action으로 actual-time GIF를 만든다."""

    fps = _validated_fps(fps)
    config = load_config(config_path)
    degree_values = np.asarray(
        [descent_angle_deg, pan_tilt_angle_deg, lift_angle_deg],
        dtype=float,
    )
    if not np.isfinite(degree_values).all() or not np.isfinite(descent_speed_m_s):
        raise ValueError("명시 physical action은 모두 유한해야 합니다.")
    physical_action = (
        float(np.deg2rad(descent_angle_deg)),
        float(np.deg2rad(pan_tilt_angle_deg)),
        float(descent_speed_m_s),
        float(np.deg2rad(lift_angle_deg)),
    )
    run = _run_action_replay(
        config,
        physical_action,
        seed=seed,
        count_per_type=count_per_type,
        expected_particle_count=3 * int(count_per_type),
        fps=fps,
    )
    action = run.info["action_parameters"]
    title = (
        f"Explicit physical action | {int(run.info['particle_count'])} particles | "
        f"reward {run.replay_reward:+.3f}\n"
        f"descent {np.rad2deg(float(action['descent_angle'])):.1f}° | "
        f"tilt {np.rad2deg(float(action['pan_tilt_angle'])):.1f}° | "
        f"lift {np.rad2deg(float(action['lift_angle'])):.1f}° | "
        f"speed {float(action['descent_speed']):.3f} m/s"
    )
    joint_speed_report = run.info.get("joint_speed_report")
    if isinstance(joint_speed_report, Mapping):
        title += (
            "\nM0609 nominal 0.4 m TCP | "
            f"J peak {100.0 * float(joint_speed_report['limiting_utilization']):.1f}% | "
            f"TCP {float(joint_speed_report['peak_tcp_linear_velocity_m_s']):.3f} m/s"
        )
    requested_action = dict(zip(ACTION_COLUMNS, physical_action, strict=True))
    replayed_action = _replayed_physical_action(run)
    action_deltas = {
        name: replayed_action[name] - requested_action[name] for name in ACTION_COLUMNS
    }
    return _write_replay_gif(
        config_path,
        output_path,
        run,
        fps=fps,
        title=title,
        metadata_fields={
            "selection_mode": "explicit_physical_action",
            "requested_physical_action": requested_action,
            "requested_physical_action_degrees": {
                "descent_angle_deg": float(descent_angle_deg),
                "pan_tilt_angle_deg": float(pan_tilt_angle_deg),
                "descent_speed_m_s": float(descent_speed_m_s),
                "lift_angle_deg": float(lift_angle_deg),
            },
            "physical_action_delta_from_requested": action_deltas,
        },
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for CSV replay and explicit physical-action modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--episodes-csv",
        type=Path,
        help="training run의 episodes.csv",
    )
    mode.add_argument("--seed", type=int, help="CSV 없는 명시 action rollout seed")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "CSV mode에서 생략하면 episodes.csv 옆 effective_config.yaml, "
            "명시 action mode에서는 configs/fried_rice.yaml 사용"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-id", type=int, default=None)
    parser.add_argument("--episode-start", type=int, default=144)
    parser.add_argument("--episode-end", type=int, default=149)
    parser.add_argument("--count-per-type", type=int, default=None)
    parser.add_argument("--descent-angle-deg", type=float, default=None)
    parser.add_argument("--pan-tilt-angle-deg", type=float, default=None)
    parser.add_argument("--descent-speed-m-s", type=float, default=None)
    parser.add_argument("--lift-angle-deg", type=float, default=None)
    parser.add_argument("--fps", type=int, default=20)
    return parser


def resolve_cli_mode(
    parser: argparse.ArgumentParser,
    arguments: argparse.Namespace,
) -> tuple[str, Path]:
    """Validate conditional CLI fields and resolve the effective config path."""

    explicit_field_names = (
        "count_per_type",
        "descent_angle_deg",
        "pan_tilt_angle_deg",
        "descent_speed_m_s",
        "lift_angle_deg",
    )
    if arguments.episodes_csv is not None:
        supplied = [name for name in explicit_field_names if getattr(arguments, name) is not None]
        if supplied:
            parser.error(
                "CSV mode에는 explicit action 옵션을 함께 사용할 수 없습니다: "
                + ", ".join(supplied)
            )
        config_path = (
            arguments.episodes_csv.parent / "effective_config.yaml"
            if arguments.config is None
            else arguments.config
        )
        return "csv", config_path

    missing = [name for name in explicit_field_names if getattr(arguments, name) is None]
    if missing:
        parser.error(
            "--seed mode에는 다음 옵션이 모두 필요합니다: "
            + ", ".join(name.replace("_", "-") for name in missing)
        )
    if arguments.episode_id is not None:
        parser.error("--episode-id는 --episodes-csv mode에서만 사용할 수 있습니다.")
    config_path = Path("configs/fried_rice.yaml") if arguments.config is None else arguments.config
    return "explicit", config_path


def main() -> None:
    parser = build_argument_parser()
    arguments = parser.parse_args()
    mode, config_path = resolve_cli_mode(parser, arguments)
    if not config_path.is_file():
        raise FileNotFoundError(f"config를 찾을 수 없습니다: {config_path}")
    if mode == "csv":
        gif_path, metadata_path = generate_episode_gif(
            config_path,
            arguments.episodes_csv,
            arguments.output,
            episode_id=arguments.episode_id,
            episode_start=arguments.episode_start,
            episode_end=arguments.episode_end,
            fps=arguments.fps,
        )
    else:
        gif_path, metadata_path = generate_physical_action_gif(
            config_path,
            arguments.output,
            seed=arguments.seed,
            count_per_type=arguments.count_per_type,
            descent_angle_deg=arguments.descent_angle_deg,
            pan_tilt_angle_deg=arguments.pan_tilt_angle_deg,
            descent_speed_m_s=arguments.descent_speed_m_s,
            lift_angle_deg=arguments.lift_angle_deg,
            fps=arguments.fps,
        )
    print(
        json.dumps(
            {
                "gif": str(gif_path),
                "metadata": str(metadata_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
