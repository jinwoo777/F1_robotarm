"""Export every condition-specific best-GIF replay as 20 FPS trajectory data."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from wok_sim.config import load_config
from wok_sim.geometry import matrix_to_rpy, quaternion_to_matrix
from wok_sim.visualization.episode_gif import (
    _run_action_replay,
    select_episode_from_csv,
)

CONDITION_METADATA_PATTERN = re.compile(
    r"^best_(?P<condition_mass_g>\d+)g_.*\.metadata\.json$"
)
PAN_RIM_ENDPOINT_ORDER = ("rear_local_minus_x", "front_local_plus_x")


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


def _interpolate_history(
    source_time_s: np.ndarray,
    values: np.ndarray,
    sample_time_s: np.ndarray,
) -> np.ndarray:
    source_time = np.asarray(source_time_s, dtype=float)
    source = np.asarray(values, dtype=float)
    sample_time = np.asarray(sample_time_s, dtype=float)
    if source.shape[0] != len(source_time):
        raise ValueError("보간할 배열의 첫 축이 source time 길이와 다릅니다.")
    flattened = source.reshape(len(source_time), -1)
    sampled = np.column_stack(
        [
            np.interp(sample_time, source_time, flattened[:, index])
            for index in range(flattened.shape[1])
        ]
    )
    return sampled.reshape((len(sample_time), *source.shape[1:]))


def _resample_quaternions_wxyz(
    source_time_s: np.ndarray,
    quaternions_wxyz: np.ndarray,
    sample_time_s: np.ndarray,
) -> np.ndarray:
    source_time = np.asarray(source_time_s, dtype=float)
    quaternions = np.asarray(quaternions_wxyz, dtype=float)
    sample_time = np.asarray(sample_time_s, dtype=float)
    if quaternions.shape != (len(source_time), 4):
        raise ValueError("quaternions_wxyz shape은 (T,4)이어야 합니다.")
    norms = np.linalg.norm(quaternions, axis=1)
    if not np.isfinite(quaternions).all() or np.any(norms <= np.finfo(float).eps):
        raise ValueError("quaternion은 유한한 non-zero 값이어야 합니다.")
    normalized = quaternions / norms[:, None]
    rotations = Rotation.from_quat(normalized[:, [1, 2, 3, 0]])
    sampled_xyzw = Slerp(source_time, rotations)(sample_time).as_quat()
    sampled_wxyz = sampled_xyzw[:, [3, 0, 1, 2]]
    sampled_wxyz[sampled_wxyz[:, 0] < 0.0] *= -1.0
    return sampled_wxyz


def _nearest_source_indices(
    source_time_s: np.ndarray,
    sample_time_s: np.ndarray,
) -> np.ndarray:
    source_time = np.asarray(source_time_s, dtype=float)
    sample_time = np.asarray(sample_time_s, dtype=float)
    right = np.searchsorted(source_time, sample_time, side="left")
    right = np.clip(right, 0, len(source_time) - 1)
    left = np.clip(right - 1, 0, len(source_time) - 1)
    choose_left = np.abs(sample_time - source_time[left]) <= np.abs(
        source_time[right] - sample_time
    )
    return np.where(choose_left, left, right)


def _rpy_from_quaternions(quaternions_wxyz: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            matrix_to_rpy(quaternion_to_matrix(quaternion))
            for quaternion in np.asarray(quaternions_wxyz, dtype=float)
        ]
    )


def _sample_replay_arrays(run: Any) -> dict[str, np.ndarray]:
    result = run.info["simulation_result"]
    raw_time = np.asarray(result["time_s"], dtype=float)
    sample_time = np.asarray(run.frame_time_s, dtype=float)
    pan = np.asarray(run.frame_pan_world_m, dtype=float)
    particles_world = np.asarray(run.frame_particles_world_m, dtype=float)
    quaternion = _resample_quaternions_wxyz(
        raw_time,
        np.asarray(result["pan_quaternion_wxyz"], dtype=float),
        sample_time,
    )
    rotations = np.stack([quaternion_to_matrix(item) for item in quaternion])
    particles_pan = np.einsum(
        "tni,tij->tnj",
        particles_world - pan[:, None, :],
        rotations,
    )
    nearest_indices = _nearest_source_indices(raw_time, sample_time)
    return {
        "time_s": sample_time,
        "pan_position_world_m": pan,
        "pan_quaternion_wxyz": quaternion,
        "pan_rpy_rad": _rpy_from_quaternions(quaternion),
        "pan_rim_endpoints_world_m": np.asarray(
            run.frame_rim_endpoints_world_m,
            dtype=float,
        ),
        "particle_positions_world_m": particles_world,
        "particle_positions_pan_m": particles_pan,
        "particle_velocities_world_m_s": _interpolate_history(
            raw_time,
            np.asarray(result["particle_velocities_world_m_s"], dtype=float),
            sample_time,
        ),
        "contact_with_pan": np.asarray(result["contact_with_pan"], dtype=bool)[
            nearest_indices
        ],
        "contact_normal_force_n": _interpolate_history(
            raw_time,
            np.asarray(result["contact_normal_force_n"], dtype=float),
            sample_time,
        ),
        "crossed_spill_boundary": np.asarray(
            result["crossed_spill_boundary"],
            dtype=bool,
        ),
        "species": np.asarray(run.info["particle_batch"]["species"], dtype=str),
        "particle_radii_m": np.asarray(
            run.info["particle_batch"]["radii_m"],
            dtype=float,
        ),
        "particle_masses_kg": np.asarray(
            run.info["particle_batch"]["masses_kg"],
            dtype=float,
        ),
    }


def _pan_dataframe(
    arrays: Mapping[str, np.ndarray],
    *,
    condition_mass_g: int,
    actual_mass_g: float,
    episode_id: int,
) -> pd.DataFrame:
    time = arrays["time_s"]
    position = arrays["pan_position_world_m"]
    quaternion = arrays["pan_quaternion_wxyz"]
    rpy = arrays["pan_rpy_rad"]
    endpoints = arrays["pan_rim_endpoints_world_m"]
    return pd.DataFrame(
        {
            "condition_mass_g": condition_mass_g,
            "actual_mass_g": actual_mass_g,
            "episode_id": episode_id,
            "frame_index": np.arange(len(time)),
            "time_s": time,
            "pan_position_world_m_x": position[:, 0],
            "pan_position_world_m_y": position[:, 1],
            "pan_position_world_m_z": position[:, 2],
            "pan_quaternion_w": quaternion[:, 0],
            "pan_quaternion_x": quaternion[:, 1],
            "pan_quaternion_y": quaternion[:, 2],
            "pan_quaternion_z": quaternion[:, 3],
            "pan_roll_rad": rpy[:, 0],
            "pan_pitch_rad": rpy[:, 1],
            "pan_yaw_rad": rpy[:, 2],
            "rim_rear_world_m_x": endpoints[:, 0, 0],
            "rim_rear_world_m_y": endpoints[:, 0, 1],
            "rim_rear_world_m_z": endpoints[:, 0, 2],
            "rim_front_world_m_x": endpoints[:, 1, 0],
            "rim_front_world_m_y": endpoints[:, 1, 1],
            "rim_front_world_m_z": endpoints[:, 1, 2],
        }
    )


def _particle_dataframe(
    arrays: Mapping[str, np.ndarray],
    *,
    condition_mass_g: int,
    actual_mass_g: float,
    episode_id: int,
) -> pd.DataFrame:
    time = arrays["time_s"]
    world = arrays["particle_positions_world_m"]
    pan = arrays["particle_positions_pan_m"]
    velocity = arrays["particle_velocities_world_m_s"]
    contact = arrays["contact_with_pan"]
    force = arrays["contact_normal_force_n"]
    frame_count, particle_count, _ = world.shape
    rows = frame_count * particle_count
    frame_index = np.repeat(np.arange(frame_count), particle_count)
    particle_index = np.tile(np.arange(particle_count), frame_count)
    world_flat = world.reshape(rows, 3)
    pan_flat = pan.reshape(rows, 3)
    velocity_flat = velocity.reshape(rows, 3)
    return pd.DataFrame(
        {
            "condition_mass_g": condition_mass_g,
            "actual_mass_g": actual_mass_g,
            "episode_id": episode_id,
            "frame_index": frame_index,
            "time_s": np.repeat(time, particle_count),
            "particle_index": particle_index,
            "species": np.tile(arrays["species"], frame_count),
            "radius_m": np.tile(arrays["particle_radii_m"], frame_count),
            "mass_kg": np.tile(arrays["particle_masses_kg"], frame_count),
            "position_world_m_x": world_flat[:, 0],
            "position_world_m_y": world_flat[:, 1],
            "position_world_m_z": world_flat[:, 2],
            "position_pan_m_x": pan_flat[:, 0],
            "position_pan_m_y": pan_flat[:, 1],
            "position_pan_m_z": pan_flat[:, 2],
            "velocity_world_m_s_x": velocity_flat[:, 0],
            "velocity_world_m_s_y": velocity_flat[:, 1],
            "velocity_world_m_s_z": velocity_flat[:, 2],
            "contact_with_pan": contact.reshape(rows),
            "contact_normal_force_n": force.reshape(rows),
            "crossed_spill_boundary": np.tile(
                arrays["crossed_spill_boundary"],
                frame_count,
            ),
        }
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)


def _write_npz(
    arrays: Mapping[str, np.ndarray],
    path: Path,
    *,
    metadata: Mapping[str, Any],
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **arrays,
            condition_mass_g=np.asarray(metadata["condition_mass_g"]),
            actual_mass_g=np.asarray(metadata["actual_mass_g"]),
            episode_id=np.asarray(metadata["episode_id"]),
            random_seed=np.asarray(metadata["random_seed"]),
            final_reward=np.asarray(metadata["final_reward"]),
            normalized_action=np.asarray(metadata["normalized_action"], dtype=float),
            physical_action=np.asarray(metadata["physical_action"], dtype=float),
            fps=np.asarray(metadata["fps"]),
            sample_interval_s=np.asarray(metadata["sample_interval_s"]),
            pan_rim_endpoint_order=np.asarray(PAN_RIM_ENDPOINT_ORDER),
        )
    temporary.replace(path)


def _write_comparison_plot(cases: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    figure, axes_array = plt.subplots(
        1,
        len(cases),
        figsize=(6 * len(cases), 6),
        constrained_layout=True,
        squeeze=False,
    )
    axes = list(axes_array.reshape(-1))
    all_points: list[np.ndarray] = []
    for case in cases:
        arrays = case["arrays"]
        all_points.extend(
            [
                arrays["pan_position_world_m"],
                arrays["pan_rim_endpoints_world_m"].reshape(-1, 3),
                arrays["particle_positions_world_m"].reshape(-1, 3),
            ]
        )
    merged = np.concatenate(all_points, axis=0)
    x_margin = max(0.02, 0.05 * float(np.ptp(merged[:, 0])))
    z_margin = max(0.02, 0.05 * float(np.ptp(merged[:, 2])))
    x_limits = (float(merged[:, 0].min() - x_margin), float(merged[:, 0].max() + x_margin))
    z_limits = (float(merged[:, 2].min() - z_margin), float(merged[:, 2].max() + z_margin))

    species_colors = {
        "rice": "#E9C46A",
        "carrot": "#F77F00",
        "pea": "#2A9D8F",
    }
    for axis, case in zip(axes, cases, strict=True):
        arrays = case["arrays"]
        particles = arrays["particle_positions_world_m"]
        species = arrays["species"]
        for particle_index, particle_species in enumerate(species):
            axis.plot(
                particles[:, particle_index, 0],
                particles[:, particle_index, 2],
                color=species_colors.get(str(particle_species), "#808080"),
                linewidth=0.45,
                alpha=0.18,
            )
        centroid = particles.mean(axis=1)
        pan = arrays["pan_position_world_m"]
        endpoints = arrays["pan_rim_endpoints_world_m"]
        axis.plot(pan[:, 0], pan[:, 2], color="#D62728", linewidth=2.0, label="pan center")
        axis.plot(
            endpoints[:, 0, 0],
            endpoints[:, 0, 2],
            color="#1F77B4",
            linewidth=1.2,
            label="rear/front rim",
        )
        axis.plot(
            endpoints[:, 1, 0],
            endpoints[:, 1, 2],
            color="#1F77B4",
            linewidth=1.2,
        )
        axis.plot(
            centroid[:, 0],
            centroid[:, 2],
            color="black",
            linewidth=1.4,
            linestyle="--",
            label="particle centroid",
        )
        axis.set_title(
            f"{case['condition_mass_g']}g | ep {case['episode_id']} | "
            f"{case['particle_count']} particles\n"
            f"actual {case['actual_mass_g']:.3f}g | {case['frame_count']} frames"
        )
        axis.set_xlim(*x_limits)
        axis.set_ylim(*z_limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("world x (m)")
        axis.set_ylabel("world z (m)")
        axis.grid(alpha=0.2)
        axis.legend(loc="best", fontsize=8, frameon=False)
    figure.suptitle("Best learned trajectories by mass condition (20 FPS / 0.05 s)")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _condition_metadata_files(best_gifs_directory: Path) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for path in best_gifs_directory.glob("best_*g_*.metadata.json"):
        match = CONDITION_METADATA_PATTERN.match(path.name)
        if match is not None:
            found.append((int(match.group("condition_mass_g")), path))
    found.sort(key=lambda item: item[0])
    if not found:
        raise FileNotFoundError(
            f"조건별 best GIF metadata를 찾지 못했습니다: {best_gifs_directory}"
        )
    labels = [label for label, _ in found]
    if len(labels) != len(set(labels)):
        raise ValueError(f"조건 질량 label이 중복되었습니다: {labels}")
    return found


def export_best_gif_trajectories(
    run_directory: Path,
    *,
    output_directory: Path | None = None,
    fps: int = 20,
) -> Path:
    run_directory = Path(run_directory).resolve()
    best_gifs_directory = run_directory / "best_gifs"
    if not best_gifs_directory.is_dir():
        raise FileNotFoundError(f"best_gifs 폴더를 찾지 못했습니다: {best_gifs_directory}")
    if isinstance(fps, bool) or int(fps) != fps or not 1 <= int(fps) <= 60:
        raise ValueError("fps는 1~60 범위의 정수여야 합니다.")
    fps = int(fps)
    interval_s = 1.0 / fps
    output = (
        Path(output_directory).resolve()
        if output_directory is not None
        else best_gifs_directory / f"trajectories_{fps}fps"
    )
    output.mkdir(parents=True, exist_ok=True)

    manifest_cases: list[dict[str, Any]] = []
    plot_cases: list[dict[str, Any]] = []
    for condition_mass_g, metadata_path in _condition_metadata_files(best_gifs_directory):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        selected = metadata["selected_episode"]
        episode_id = int(selected["episode_id"])
        config_path = Path(metadata.get("config", run_directory / "effective_config.yaml"))
        source_csv = Path(metadata.get("source_csv", run_directory / "episodes.csv"))
        selection = select_episode_from_csv(source_csv, episode_id=episode_id)
        config = load_config(config_path)
        run = _run_action_replay(
            config,
            selection.physical_action,
            seed=selection.random_seed,
            count_per_type=selection.count_per_type,
            expected_particle_count=selection.particle_count,
            fps=fps,
            curriculum_episode=selection.episode_id,
            nominal_joint_speed_target_fraction=selection.nominal_joint_speed_target_fraction,
        )
        arrays = _sample_replay_arrays(run)
        actual_mass_g = float(run.reset_info["actual_total_mass_kg"]) * 1000.0
        expected_timing = metadata.get("timing", {})
        if fps == int(expected_timing.get("fps", fps)):
            expected_frames = int(expected_timing.get("frame_count", len(arrays["time_s"])))
            if expected_frames != len(arrays["time_s"]):
                raise RuntimeError(
                    f"{condition_mass_g}g replay frame 수가 기존 GIF와 다릅니다: "
                    f"{len(arrays['time_s'])} != {expected_frames}"
                )
        reward_delta = float(run.replay_reward) - float(selection.final_reward)
        mass_delta_kg = (
            None
            if selection.actual_total_mass_kg is None
            else float(run.reset_info["actual_total_mass_kg"])
            - float(selection.actual_total_mass_kg)
        )
        if abs(reward_delta) > 1.0e-9:
            raise RuntimeError(
                f"{condition_mass_g}g replay reward가 학습 로그와 다릅니다: {reward_delta}"
            )
        if mass_delta_kg is not None and abs(mass_delta_kg) > 1.0e-12:
            raise RuntimeError(
                f"{condition_mass_g}g replay mass가 학습 로그와 다릅니다: {mass_delta_kg}"
            )

        stem = f"{condition_mass_g:03d}g_ep{episode_id:03d}"
        pan_path = output / f"{stem}_pan_trajectory_{fps}fps.csv"
        particles_path = output / f"{stem}_particle_trajectories_{fps}fps.csv"
        npz_path = output / f"{stem}_trajectory_{fps}fps.npz"
        _write_csv(
            _pan_dataframe(
                arrays,
                condition_mass_g=condition_mass_g,
                actual_mass_g=actual_mass_g,
                episode_id=episode_id,
            ),
            pan_path,
        )
        particle_frame = _particle_dataframe(
            arrays,
            condition_mass_g=condition_mass_g,
            actual_mass_g=actual_mass_g,
            episode_id=episode_id,
        )
        _write_csv(particle_frame, particles_path)
        normalized_action = np.asarray(run.normalized_action, dtype=float)
        condition_metadata = {
            "condition_mass_g": condition_mass_g,
            "actual_mass_g": actual_mass_g,
            "episode_id": episode_id,
            "random_seed": selection.random_seed,
            "particle_count": selection.particle_count,
            "count_per_type": selection.count_per_type,
            "final_reward": float(run.replay_reward),
            "normalized_action": normalized_action,
            "physical_action": np.asarray(selection.physical_action, dtype=float),
            "fps": fps,
            "sample_interval_s": interval_s,
            "frame_count": len(arrays["time_s"]),
            "duration_s": float(arrays["time_s"][-1] - arrays["time_s"][0]),
            "reward_delta_from_csv": reward_delta,
            "mass_delta_from_csv_kg": mass_delta_kg,
        }
        _write_npz(arrays, npz_path, metadata=condition_metadata)
        manifest_case = {
            **condition_metadata,
            "source_gif_metadata": str(metadata_path.resolve()),
            "pan_csv": str(pan_path.resolve()),
            "particle_csv": str(particles_path.resolve()),
            "npz": str(npz_path.resolve()),
            "pan_csv_rows": len(arrays["time_s"]),
            "particle_csv_rows": len(particle_frame),
        }
        manifest_cases.append(manifest_case)
        plot_cases.append({**manifest_case, "arrays": arrays})

    comparison_plot = output / f"trajectory_xz_comparison_{fps}fps.png"
    _write_comparison_plot(plot_cases, comparison_plot)
    summary_frame = pd.DataFrame(
        [
            {
                key: value
                for key, value in case.items()
                if key
                not in {
                    "normalized_action",
                    "physical_action",
                }
            }
            for case in manifest_cases
        ]
    )
    summary_csv = output / f"trajectory_conditions_{fps}fps.csv"
    _write_csv(summary_frame, summary_csv)
    manifest = {
        "source_run_directory": str(run_directory),
        "source_best_gifs_directory": str(best_gifs_directory),
        "fps": fps,
        "sample_interval_s": interval_s,
        "coordinate_frames": {
            "world": "MuJoCo world frame; SI units",
            "pan": "position relative to the moving pan pose; SI units",
            "quaternion_order": "wxyz",
            "rpy_order": "fixed-axis xyz roll-pitch-yaw in radians",
            "pan_rim_endpoint_order": list(PAN_RIM_ENDPOINT_ORDER),
        },
        "particle_csv_layout": "long format; one row per frame and particle",
        "contact_sampling": "nearest raw simulation sample",
        "continuous_value_sampling": "linear interpolation; quaternion uses SLERP",
        "comparison_plot": str(comparison_plot.resolve()),
        "summary_csv": str(summary_csv.resolve()),
        "conditions": manifest_cases,
    }
    manifest_path = output / f"trajectory_manifest_{fps}fps.json"
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary_manifest.write_text(
        json.dumps(_json_value(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    return manifest_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="effective_config.yaml, episodes.csv, best_gifs가 있는 학습 실행 폴더",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="기본값: <run-dir>/best_gifs/trajectories_<fps>fps",
    )
    parser.add_argument("--fps", type=int, default=20)
    return parser


def main() -> int:
    arguments = build_argument_parser().parse_args()
    manifest = export_best_gif_trajectories(
        arguments.run_dir,
        output_directory=arguments.output_dir,
        fps=arguments.fps,
    )
    print(json.dumps({"manifest": str(manifest.resolve())}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
