from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from wok_sim.visualization import episode_gif as gif_module
from wok_sim.visualization.episode_gif import (
    ACTION_COLUMNS,
    EpisodeReplaySelection,
    _run_action_replay,
    build_argument_parser,
    generate_physical_action_gif,
    normalized_action_from_physical_action,
    normalized_action_from_selection,
    render_xz_gif,
    resolve_cli_mode,
    select_episode_from_csv,
    trajectory_dot_times,
    transform_pan_local_history,
)


def _episode_row(
    episode_id: int,
    reward: float,
    *,
    valid: bool = True,
) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "random_seed": 10_000 + episode_id,
        "particle_count": 60 + 30 * (episode_id % 3),
        "count_per_type": 20 + 10 * (episode_id % 3),
        "actual_total_mass_kg": 0.06,
        "nominal_joint_speed_target_fraction": 0.84,
        "descent_angle_rad": np.deg2rad(45.0),
        "pan_tilt_angle_rad": np.deg2rad(30.0),
        "descent_speed_m_s": 0.43,
        "lift_angle_rad": np.deg2rad(15.0),
        "trajectory_valid": valid,
        "final_reward": reward,
    }


def _write_episode_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_selects_highest_completed_reward_from_default_final_batch(tmp_path: Path) -> None:
    path = tmp_path / "episodes.csv"
    rows = [
        _episode_row(143, 99.0),
        _episode_row(144, 1.0),
        _episode_row(145, 4.0),
        _episode_row(146, 8.0, valid=False),
        _episode_row(147, 3.0),
        _episode_row(149, 2.0),
        _episode_row(150, 100.0),
    ]
    _write_episode_csv(path, rows)

    selected = select_episode_from_csv(path)

    assert selected.episode_id == 145
    assert selected.random_seed == 10_145
    assert selected.final_reward == pytest.approx(4.0)
    assert selected.nominal_joint_speed_target_fraction == pytest.approx(0.84)
    assert selected.physical_action == pytest.approx(
        (
            np.deg2rad(45.0),
            np.deg2rad(30.0),
            0.43,
            np.deg2rad(15.0),
        )
    )
    assert select_episode_from_csv(path, episode_id=147).episode_id == 147


def test_inverse_maps_logged_physical_action_to_exact_center_4d() -> None:
    selection = EpisodeReplaySelection(
        episode_id=149,
        random_seed=7,
        count_per_type=40,
        particle_count=120,
        final_reward=2.0,
        physical_action=(
            np.deg2rad(45.0),
            np.deg2rad(30.0),
            0.43,
            np.deg2rad(15.0),
        ),
        actual_total_mass_kg=0.12,
    )
    config = {
        "trajectory": {
            "fried_rice": {
                "descent_angle_range_rad": np.deg2rad([35.0, 55.0]),
                "pan_tilt_angle_range_rad": np.deg2rad([20.0, 40.0]),
                "descent_speed_range_m_s": [0.38, 0.48],
                "lift_angle_range_rad": np.deg2rad([0.0, 30.0]),
            }
        }
    }

    action = normalized_action_from_selection(selection, config)

    assert action.dtype == np.float32
    np.testing.assert_allclose(action, np.zeros(4), atol=1.0e-6)
    assert len(ACTION_COLUMNS) == 4


def test_inverse_maps_explicit_physical_action_bounds_and_rejects_outside() -> None:
    config = {
        "trajectory": {
            "fried_rice": {
                "descent_angle_range_rad": [1.0, 2.0],
                "pan_tilt_angle_range_rad": [3.0, 4.0],
                "descent_speed_range_m_s": [5.0, 6.0],
                "lift_angle_range_rad": [7.0, 8.0],
            }
        }
    }

    np.testing.assert_array_equal(
        normalized_action_from_physical_action([1.0, 3.0, 5.0, 7.0], config),
        -np.ones(4, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        normalized_action_from_physical_action([2.0, 4.0, 6.0, 8.0], config),
        np.ones(4, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="config 범위"):
        normalized_action_from_physical_action([0.9, 3.0, 5.0, 7.0], config)


def test_trajectory_dot_times_uses_exact_tenths_and_keeps_final_time() -> None:
    np.testing.assert_allclose(
        trajectory_dot_times(np.asarray([1.0, 1.17, 1.35])),
        [1.0, 1.1, 1.2, 1.3, 1.35],
    )


def test_pan_local_outline_follows_positive_pitch_quaternion() -> None:
    center = np.asarray([[0.0, 0.0, 0.2], [0.0, 0.0, 0.2]])
    half_pitch = np.deg2rad(30.0) * 0.5
    quaternion = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [np.cos(half_pitch), 0.0, np.sin(half_pitch), 0.0],
        ]
    )
    local = np.asarray([[-0.11, 0.0, 0.06], [0.11, 0.0, 0.06]])

    transformed = transform_pan_local_history(center, quaternion, local)

    np.testing.assert_allclose(transformed[0], center[0] + local)
    assert transformed[1, 1, 2] < transformed[1, 0, 2]


def test_render_xz_gif_writes_real_time_frames(tmp_path: Path) -> None:
    time = np.asarray([0.0, 0.2, 0.4])
    pan = np.column_stack((0.1 * time, np.zeros_like(time), 0.2 - 0.05 * time))
    endpoints = np.repeat(pan[:, None, :], 2, axis=1)
    endpoints[:, 0, 0] -= 0.11
    endpoints[:, 1, 0] += 0.11
    endpoints[:, :, 2] += 0.06
    outline = np.repeat(pan[:, None, :], 4, axis=1)
    outline[:, :, 0] += [-0.11, -0.04, 0.04, 0.11]
    outline[:, :, 2] += [0.06, 0.0, 0.0, 0.06]
    particles = np.repeat(pan[:, None, :], 2, axis=1)
    particles[:, 0, 0] -= 0.02
    particles[:, 1, 0] += 0.02
    particles[:, :, 2] += 0.02
    output = tmp_path / "episode.gif"

    timing = render_xz_gif(
        output,
        time_s=time,
        pan_position_world_m=pan,
        pan_rim_endpoints_world_m=endpoints,
        pan_outline_world_m=outline,
        particle_positions_world_m=particles,
        species=["small_sphere", "ellipsoid"],
        fps=5,
        title="tiny replay",
    )

    assert output.is_file()
    with Image.open(output) as image:
        assert image.n_frames == 3
        assert image.info["duration"] == 200
    assert timing["fps"] == 5
    assert timing["frame_count"] == 3
    assert timing["trajectory_dot_interval_s"] == pytest.approx(0.1)
    assert timing["trajectory_dot_count"] == 5
    assert timing["simulation_duration_s"] == pytest.approx(0.4)
    assert timing["motion_playback_duration_s"] == pytest.approx(0.4)
    assert timing["playback_speed_ratio"] == pytest.approx(1.0)


def _explicit_config() -> dict[str, Any]:
    return {
        "trajectory": {
            "fried_rice": {
                "descent_angle_range_rad": np.deg2rad([35.0, 55.0]),
                "pan_tilt_angle_range_rad": np.deg2rad([20.0, 40.0]),
                "descent_speed_range_m_s": [0.38, 0.48],
                "lift_angle_range_rad": np.deg2rad([0.0, 30.0]),
            }
        },
        "pan": {
            "collision_proxy": {
                "bottom_radius_m": 0.060,
                "inner_radius_m": 0.115,
                "rim_radius_m": 0.120,
                "bottom_z_m": 0.0,
                "rim_z_m": 0.055,
                "bottom_thickness_m": 0.005,
                "wall_thickness_m": 0.005,
            }
        },
    }


class _ExplicitFakeEnvironment:
    received_seed: int | None = None
    received_count: int | None = None
    received_action: np.ndarray | None = None
    received_options: dict[str, int] | None = None

    def __init__(self, _config: dict[str, Any]) -> None:
        return None

    def reset(
        self,
        *,
        seed: int,
        options: dict[str, int],
    ) -> tuple[np.ndarray, dict[str, float]]:
        type(self).received_seed = seed
        type(self).received_count = options["count_per_type"]
        type(self).received_options = dict(options)
        return np.zeros(2), {"actual_total_mass_kg": 0.006}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        type(self).received_action = action.copy()
        count_per_type = int(type(self).received_count or 0)
        total = 3 * count_per_type
        time = np.asarray([0.0, 0.1, 0.2])
        pan = np.column_stack((0.1 * time, np.zeros_like(time), 0.2 - 0.1 * time))
        particles = np.repeat(pan[:, None, :], total, axis=1)
        return (
            np.zeros(2),
            1.25,
            True,
            False,
            {
                "trajectory_valid": True,
                "random_seed": int(type(self).received_seed or 0),
                "count_per_type": count_per_type,
                "particle_count": total,
                "action_parameters": {
                    "descent_angle": np.deg2rad(45.0),
                    "pan_tilt_angle": np.deg2rad(30.0),
                    "descent_speed": 0.43,
                    "lift_angle": np.deg2rad(15.0),
                },
                "particle_batch": {"species": ["small_sphere"] * total},
                "simulation_result": {
                    "time_s": time,
                    "pan_position_world_m": pan,
                    "pan_quaternion_wxyz": np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
                    "particle_positions_world_m": particles,
                },
            },
        )

    def close(self) -> None:
        return None


def test_explicit_physical_action_mode_runs_without_csv_and_writes_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    output = tmp_path / "explicit.gif"
    monkeypatch.setattr(gif_module, "load_config", lambda _path: _explicit_config())
    monkeypatch.setattr(gif_module, "WokMixingEnv", _ExplicitFakeEnvironment)

    def _fake_render(path: Path, **kwargs: Any) -> dict[str, float | int]:
        path.touch()
        assert kwargs["trajectory_dot_interval_s"] == pytest.approx(0.1)
        return {"fps": kwargs["fps"], "trajectory_dot_interval_s": 0.1}

    monkeypatch.setattr(gif_module, "render_xz_gif", _fake_render)

    gif_path, metadata_path = generate_physical_action_gif(
        config_path,
        output,
        seed=91,
        count_per_type=2,
        descent_angle_deg=45.0,
        pan_tilt_angle_deg=30.0,
        descent_speed_m_s=0.43,
        lift_angle_deg=15.0,
        fps=10,
    )

    assert gif_path == output
    assert metadata_path.is_file()
    assert _ExplicitFakeEnvironment.received_seed == 91
    assert _ExplicitFakeEnvironment.received_count == 2
    np.testing.assert_allclose(_ExplicitFakeEnvironment.received_action, np.zeros(4), atol=1e-6)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["selection_mode"] == "explicit_physical_action"
    assert "source_csv" not in metadata
    assert "reward_delta_from_csv" not in metadata["replay"]
    assert metadata["render"]["pan_profile"] == (
        "closed_flat_bottom_inner_outer_circular_arcs"
    )


def test_csv_replay_can_restore_curriculum_episode_for_exact_reward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gif_module, "WokMixingEnv", _ExplicitFakeEnvironment)

    _run_action_replay(
        _explicit_config(),
        (np.deg2rad(45.0), np.deg2rad(30.0), 0.43, np.deg2rad(15.0)),
        seed=26,
        count_per_type=2,
        expected_particle_count=6,
        fps=10,
        curriculum_episode=26,
        nominal_joint_speed_target_fraction=0.84,
    )

    assert _ExplicitFakeEnvironment.received_options == {
        "count_per_type": 2,
        "curriculum_episode": 26,
        "nominal_joint_speed_target_fraction": 0.84,
    }


def test_cli_validates_csv_and_explicit_modes(tmp_path: Path) -> None:
    parser = build_argument_parser()
    csv_path = tmp_path / "episodes.csv"
    csv_arguments = parser.parse_args(["--episodes-csv", str(csv_path), "--output", "out.gif"])
    mode, config_path = resolve_cli_mode(parser, csv_arguments)
    assert mode == "csv"
    assert config_path == tmp_path / "effective_config.yaml"

    partial = parser.parse_args(["--seed", "7", "--output", "out.gif"])
    with pytest.raises(SystemExit):
        resolve_cli_mode(parser, partial)
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--episodes-csv",
                str(csv_path),
                "--seed",
                "7",
                "--output",
                "out.gif",
            ]
        )
