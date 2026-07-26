"""Replay one screened 6D pitch-release candidate as an actual-time X-Z GIF."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from wok_sim.config import load_config
from wok_sim.envs import WokMixingEnv
from wok_sim.visualization.episode_gif import render_xz_gif
from wok_sim.visualization.pan_profile import (
    pan_side_profile_local,
    resample_point_history,
    transform_pan_local_history,
)
from wok_sim.visualization.trajectory_plot import (
    pan_rim_endpoint_history,
    pan_rim_geometry_from_config,
    resample_pan_rim_endpoints,
    resample_trajectory_history,
)


def _candidate_row(selection_csv: Path, candidate_id: int) -> dict[str, str]:
    with selection_csv.open(encoding="utf-8", newline="") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if int(row["candidate_id"]) == candidate_id
        ]
    if len(rows) != 1:
        raise ValueError(
            f"selection CSV에서 candidate_id={candidate_id} 행을 하나 찾을 수 없습니다."
        )
    return rows[0]


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


def render_candidate(
    config_path: Path,
    selection_csv: Path,
    output_path: Path,
    *,
    candidate_id: int,
    fps: int = 10,
) -> tuple[Path, Path]:
    row = _candidate_row(selection_csv, candidate_id)
    config = load_config(config_path)
    normalized_action = np.asarray(
        json.loads(row["normalized_action"]),
        dtype=np.float32,
    )
    seed = int(row["random_seed"])
    count_per_type = int(row["count_per_type"])
    environment = WokMixingEnv(config)
    try:
        environment.reset(
            seed=seed,
            options={
                "count_per_type": count_per_type,
                "curriculum_episode": 1499,
                "nominal_joint_speed_target_fraction": 0.90,
            },
        )
        _, reward, terminated, truncated, info = environment.step(normalized_action)
    finally:
        environment.close()
    if not terminated or truncated:
        raise RuntimeError("GIF replay episode가 정상 종료되지 않았습니다.")
    if not bool(info.get("trajectory_valid", False)):
        raise RuntimeError(f"GIF replay trajectory가 invalid입니다: {info.get('invalid_reasons')}")

    result = info["simulation_result"]
    raw_time = np.asarray(result["time_s"], dtype=float)
    raw_pan = np.asarray(result["pan_position_world_m"], dtype=float)
    raw_quaternion = np.asarray(result["pan_quaternion_wxyz"], dtype=float)
    frame_time, frame_pan, frame_particles = resample_trajectory_history(
        raw_time,
        raw_pan,
        np.asarray(result["particle_positions_world_m"], dtype=float),
        interval_s=1.0 / int(fps),
    )
    local_endpoints, rim_center_radius, rim_z = pan_rim_geometry_from_config(config)
    raw_endpoints = pan_rim_endpoint_history(raw_pan, raw_quaternion, local_endpoints)
    frame_endpoints = resample_pan_rim_endpoints(
        raw_time,
        raw_endpoints,
        frame_time,
    )
    local_outline = pan_side_profile_local(config)
    raw_outline = transform_pan_local_history(
        raw_pan,
        raw_quaternion,
        local_outline,
    )
    frame_outline = resample_point_history(raw_time, raw_outline, frame_time)
    species = np.asarray(info["particle_batch"]["species"], dtype=str)
    action = dict(info["action_parameters"])
    peak_ratio = float(info["peak_lifted_particle_ratio"])
    particle_count = int(info["particle_count"])
    peak_count = int(round(peak_ratio * particle_count))
    spill_count = int(info["spill_count"])
    fried_rice_profile = config["trajectory"]["fried_rice"]
    coupled_lift_height_m = float(
        fried_rice_profile.get("pitch_release_lift_impulse_height_m", 0.0)
    )
    joint_speed_report = dict(info.get("joint_speed_report", {}))
    joint_utilization = [
        float(item)
        for item in joint_speed_report.get("joint_velocity_utilization", [])
    ]
    joint_text = ""
    if len(joint_utilization) == 6:
        percentages = "/".join(f"{100.0 * item:.0f}" for item in joint_utilization)
        joint_text = (
            f"\nJ1-6 peak utilization {percentages}% | "
            f"limit {100.0 * float(joint_speed_report['limiting_utilization']):.1f}%"
        )
    title = (
        f"Pitch release candidate {candidate_id} | peak {peak_count}/{particle_count} | "
        f"spill {spill_count}\n"
        f"release {np.rad2deg(float(action['pitch_release_angle'])):.1f}° | "
        f"coupled lift {1000.0 * coupled_lift_height_m:.1f} mm | "
        f"phase {float(action['pitch_release_phase_fraction']):.2f} | "
        f"effective α {float(action['pitch_release_effective_angular_acceleration']):.3f} "
        f"rad/s²{joint_text}"
    )
    timing = render_xz_gif(
        output_path,
        time_s=frame_time,
        pan_position_world_m=frame_pan,
        pan_rim_endpoints_world_m=frame_endpoints,
        pan_outline_world_m=frame_outline,
        particle_positions_world_m=frame_particles,
        species=species,
        fps=fps,
        trajectory_dot_interval_s=0.1,
        title=title,
        focus_on_pan_motion=True,
    )
    metadata = {
        "gif": str(output_path.resolve()),
        "config": str(config_path.resolve()),
        "source_selection_csv": str(selection_csv.resolve()),
        "candidate_id": candidate_id,
        "selection_metrics": {
            "peak_lifted_particle_count": int(row["peak_lifted_particle_count"]),
            "peak_lifted_particle_ratio": float(row["peak_lifted_particle_ratio"]),
            "lifted_particle_count": int(row["lifted_particle_count"]),
            "spill_count": int(row["spill_count"]),
            "spill_count_ratio": float(row["spill_count_ratio"]),
            "final_reward": float(row["final_reward"]),
        },
        "replay": {
            "random_seed": seed,
            "count_per_type": count_per_type,
            "particle_count": particle_count,
            "normalized_action": normalized_action,
            "action_parameters": action,
            "peak_lifted_particle_count": peak_count,
            "peak_lifted_particle_ratio": peak_ratio,
            "lifted_particle_count": int(info["lifted_particle_count"]),
            "spill_count": spill_count,
            "spill_count_ratio": float(info["spill_count_ratio"]),
            "final_reward": float(reward),
        },
        "coupled_pitch_lift": {
            "lift_impulse_height_m": coupled_lift_height_m,
            "synchronized_with_pitch_release": coupled_lift_height_m > 0.0,
        },
        "m0609_nominal_joint_speed": joint_speed_report,
        "pan_rim_center_radius_m": rim_center_radius,
        "pan_rim_z_m": rim_z,
        "timing": timing,
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(_json_value(metadata), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path, metadata_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection-csv", type=Path, required=True)
    parser.add_argument("--candidate-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=10)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    gif_path, metadata_path = render_candidate(
        arguments.config,
        arguments.selection_csv,
        arguments.output,
        candidate_id=arguments.candidate_id,
        fps=arguments.fps,
    )
    print(
        json.dumps(
            {"gif": str(gif_path.resolve()), "metadata": str(metadata_path.resolve())},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
