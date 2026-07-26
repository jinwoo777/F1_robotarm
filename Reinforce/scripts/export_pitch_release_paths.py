"""Export selected 6D pitch-release replays as 20 FPS pan/grain paths."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

if __package__:
    from scripts.export_best_gif_trajectories import (  # noqa: E402
        _pan_dataframe,
        _particle_dataframe,
        _sample_replay_arrays,
        _write_csv,
    )
else:
    from export_best_gif_trajectories import (  # type: ignore[import-not-found] # noqa: E402
        _pan_dataframe,
        _particle_dataframe,
        _sample_replay_arrays,
        _write_csv,
    )
from wok_sim.config import load_config  # noqa: E402
from wok_sim.envs import WokMixingEnv  # noqa: E402
from wok_sim.visualization.trajectory_plot import (  # noqa: E402
    pan_rim_endpoint_history,
    pan_rim_geometry_from_config,
    resample_pan_rim_endpoints,
    resample_trajectory_history,
)


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


def _selection_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "candidate_id",
        "random_seed",
        "count_per_type",
        "particle_count",
        "normalized_action",
    }
    if not rows or required.difference(rows[0]):
        available = rows[0] if rows else {}
        raise ValueError(
            f"selection CSV 필수 열이 없습니다: "
            f"{sorted(required.difference(available))}"
        )
    rows.sort(key=lambda row: int(row["particle_count"]))
    return rows


def _replay_row(
    config: Mapping[str, Any],
    row: Mapping[str, str],
    *,
    fps: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    action = np.asarray(json.loads(row["normalized_action"]), dtype=np.float32)
    seed = int(row["random_seed"])
    count_per_type = int(row["count_per_type"])
    expected_particles = int(row["particle_count"])
    environment = WokMixingEnv(config)
    try:
        reset_observation, reset_info = environment.reset(
            seed=seed,
            options={
                "count_per_type": count_per_type,
                "curriculum_episode": 1499,
                "nominal_joint_speed_target_fraction": 0.90,
            },
        )
        _, reward, terminated, truncated, info = environment.step(action)
    finally:
        environment.close()
    if not terminated or truncated:
        raise RuntimeError("pitch-release 경로 replay가 정상 종료되지 않았습니다.")
    if not bool(info.get("trajectory_valid", False)):
        raise RuntimeError(f"pitch-release 경로가 invalid입니다: {info.get('invalid_reasons')}")
    if int(info["particle_count"]) != expected_particles:
        raise RuntimeError(
            f"입자 수가 selection과 다릅니다: {info['particle_count']} != {expected_particles}"
        )

    result = info["simulation_result"]
    raw_time = np.asarray(result["time_s"], dtype=float)
    frame_time, frame_pan, frame_particles = resample_trajectory_history(
        raw_time,
        np.asarray(result["pan_position_world_m"], dtype=float),
        np.asarray(result["particle_positions_world_m"], dtype=float),
        interval_s=1.0 / fps,
    )
    local_endpoints, _, _ = pan_rim_geometry_from_config(config)
    raw_endpoints = pan_rim_endpoint_history(
        np.asarray(result["pan_position_world_m"], dtype=float),
        np.asarray(result["pan_quaternion_wxyz"], dtype=float),
        local_endpoints,
    )
    frame_endpoints = resample_pan_rim_endpoints(raw_time, raw_endpoints, frame_time)
    run = SimpleNamespace(
        info=info,
        reset_info=reset_info,
        reset_observation=reset_observation,
        frame_time_s=frame_time,
        frame_pan_world_m=frame_pan,
        frame_particles_world_m=frame_particles,
        frame_rim_endpoints_world_m=frame_endpoints,
        normalized_action=action,
    )
    arrays = _sample_replay_arrays(run)
    particle_count = int(info["particle_count"])
    peak_ratio = float(info["peak_lifted_particle_ratio"])
    metadata = {
        "episode_id": int(row["candidate_id"]),
        "random_seed": seed,
        "grain_count": particle_count,
        "count_per_type": count_per_type,
        "actual_mass_g": float(reset_info["actual_total_mass_kg"]) * 1000.0,
        "normalized_action": action,
        "fps": fps,
        "sample_interval_s": 1.0 / fps,
        "frame_count": len(frame_time),
        "duration_s": float(frame_time[-1] - frame_time[0]),
        "peak_lifted_particle_count": int(round(peak_ratio * particle_count)),
        "peak_lifted_particle_ratio": peak_ratio,
        "lifted_particle_count": int(info["lifted_particle_count"]),
        "spill_count": int(info["spill_count"]),
        "spill_count_ratio": float(info["spill_count_ratio"]),
        "replay_reward_final_objective": float(reward),
        "nominal_joint_speed_target_fraction": float(
            info["nominal_joint_speed_target_fraction"]
        ),
        "joint_speed_report": dict(info["joint_speed_report"]),
    }
    return arrays, metadata


def _write_npz(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            **arrays,
            grain_count=np.asarray(metadata["grain_count"]),
            episode_id=np.asarray(metadata["episode_id"]),
            random_seed=np.asarray(metadata["random_seed"]),
            fps=np.asarray(metadata["fps"]),
            sample_interval_s=np.asarray(metadata["sample_interval_s"]),
            peak_lifted_particle_count=np.asarray(
                metadata["peak_lifted_particle_count"]
            ),
            lifted_particle_count=np.asarray(metadata["lifted_particle_count"]),
            spill_count=np.asarray(metadata["spill_count"]),
            normalized_action=np.asarray(metadata["normalized_action"], dtype=float),
        )
    temporary.replace(path)


def _write_comparison_plot(
    cases: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        1,
        len(cases),
        figsize=(6 * len(cases), 6),
        constrained_layout=True,
        squeeze=False,
    )
    flattened_axes = list(axes.reshape(-1))
    all_points = np.concatenate(
        [
            np.concatenate(
                [
                    case["arrays"]["pan_position_world_m"],
                    case["arrays"]["particle_positions_world_m"].reshape(-1, 3),
                ],
                axis=0,
            )
            for case in cases
        ],
        axis=0,
    )
    x_margin = max(0.02, 0.05 * float(np.ptp(all_points[:, 0])))
    z_margin = max(0.02, 0.05 * float(np.ptp(all_points[:, 2])))
    x_limits = (
        float(all_points[:, 0].min() - x_margin),
        float(all_points[:, 0].max() + x_margin),
    )
    z_limits = (
        float(all_points[:, 2].min() - z_margin),
        float(all_points[:, 2].max() + z_margin),
    )
    for axis, case in zip(flattened_axes, cases, strict=True):
        arrays = case["arrays"]
        metadata = case["metadata"]
        particles = arrays["particle_positions_world_m"]
        for particle_index in range(particles.shape[1]):
            axis.plot(
                particles[:, particle_index, 0],
                particles[:, particle_index, 2],
                color="#E07A3F",
                linewidth=0.4,
                alpha=0.14,
            )
        pan = arrays["pan_position_world_m"]
        rims = arrays["pan_rim_endpoints_world_m"]
        axis.plot(pan[:, 0], pan[:, 2], color="#D62728", linewidth=2.0, label="pan")
        axis.plot(rims[:, 0, 0], rims[:, 0, 2], color="#1F77B4", linewidth=1.1)
        axis.plot(
            rims[:, 1, 0],
            rims[:, 1, 2],
            color="#1F77B4",
            linewidth=1.1,
            label="rim",
        )
        axis.set_title(
            f"{metadata['grain_count']} grain | ep {metadata['episode_id']}\n"
            f"peak {metadata['peak_lifted_particle_count']} | "
            f"unique {metadata['lifted_particle_count']} | "
            f"spill {metadata['spill_count']}"
        )
        axis.set_xlim(*x_limits)
        axis.set_ylim(*z_limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("world x (m)")
        axis.set_ylabel("world z (m)")
        axis.grid(alpha=0.2)
        axis.legend(loc="best", frameon=False)
    figure.suptitle("Best safe pan and grain trajectories at 20 FPS")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def export_paths(
    config_path: Path,
    selection_csv: Path,
    output_directory: Path,
    *,
    fps: int = 20,
) -> Path:
    if isinstance(fps, bool) or int(fps) != fps or not 1 <= int(fps) <= 60:
        raise ValueError("fps는 1~60 범위의 정수여야 합니다.")
    fps = int(fps)
    config = load_config(config_path)
    rows = _selection_rows(selection_csv)
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_cases: list[dict[str, Any]] = []
    plot_cases: list[dict[str, Any]] = []
    for row in rows:
        arrays, metadata = _replay_row(config, row, fps=fps)
        grain_count = int(metadata["grain_count"])
        episode_id = int(metadata["episode_id"])
        stem = f"{grain_count:03d}grain_ep{episode_id:03d}"
        pan_path = output_directory / f"{stem}_pan_trajectory_{fps}fps.csv"
        particle_path = (
            output_directory / f"{stem}_grain_trajectories_{fps}fps.csv"
        )
        npz_path = output_directory / f"{stem}_trajectory_{fps}fps.npz"
        pan_frame = _pan_dataframe(
            arrays,
            condition_mass_g=grain_count,
            actual_mass_g=float(metadata["actual_mass_g"]),
            episode_id=episode_id,
        ).rename(columns={"condition_mass_g": "grain_count"})
        grain_frame = _particle_dataframe(
            arrays,
            condition_mass_g=grain_count,
            actual_mass_g=float(metadata["actual_mass_g"]),
            episode_id=episode_id,
        ).rename(columns={"condition_mass_g": "grain_count"})
        _write_csv(pan_frame, pan_path)
        _write_csv(grain_frame, particle_path)
        _write_npz(npz_path, arrays, metadata)
        manifest_case = {
            **metadata,
            "pan_csv": str(pan_path.resolve()),
            "grain_csv": str(particle_path.resolve()),
            "npz": str(npz_path.resolve()),
            "pan_csv_rows": len(pan_frame),
            "grain_csv_rows": len(grain_frame),
        }
        manifest_cases.append(manifest_case)
        plot_cases.append({"arrays": arrays, "metadata": metadata})

    comparison_plot = output_directory / f"trajectory_xz_comparison_{fps}fps.png"
    _write_comparison_plot(plot_cases, comparison_plot)
    summary_path = output_directory / f"trajectory_conditions_{fps}fps.csv"
    summary_rows = [
        {
            key: (
                json.dumps(_json_value(value), ensure_ascii=False)
                if isinstance(value, (dict, list, tuple, np.ndarray))
                else value
            )
            for key, value in case.items()
            if key not in {"normalized_action"}
        }
        for case in manifest_cases
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    manifest = {
        "config": str(config_path.resolve()),
        "selection_csv": str(selection_csv.resolve()),
        "fps": fps,
        "sample_interval_s": 1.0 / fps,
        "coordinate_frames": {
            "world": "MuJoCo world frame; SI units",
            "pan": "position relative to the moving pan pose; SI units",
            "quaternion_order": "wxyz",
            "rpy_order": "fixed-axis xyz roll-pitch-yaw in radians",
        },
        "particle_csv_layout": "long format; one row per frame and grain",
        "comparison_plot": str(comparison_plot.resolve()),
        "summary_csv": str(summary_path.resolve()),
        "conditions": manifest_cases,
    }
    manifest_path = output_directory / f"trajectory_manifest_{fps}fps.json"
    manifest_path.write_text(
        json.dumps(_json_value(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection-csv", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=20)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    manifest = export_paths(
        arguments.config,
        arguments.selection_csv,
        arguments.output_directory,
        fps=arguments.fps,
    )
    print(json.dumps({"manifest": str(manifest.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
