from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wok_sim.visualization import trajectory_plot as plot_module
from wok_sim.visualization.trajectory_plot import (
    generate_policy_trajectory_plots,
    pan_rim_endpoint_history,
    pan_rim_geometry_from_config,
    resample_trajectory_history,
)


def test_resample_trajectory_history_uses_requested_interval() -> None:
    time = np.array([0.0, 0.15, 0.3])
    pan = np.column_stack((time, 2.0 * time, 3.0 * time))
    particles = np.stack((pan, pan + 1.0), axis=1)

    sampled_time, sampled_pan, sampled_particles = resample_trajectory_history(
        time, pan, particles, interval_s=0.1
    )

    np.testing.assert_allclose(sampled_time, [0.0, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(sampled_pan[:, 0], sampled_time)
    np.testing.assert_allclose(sampled_particles[:, 1, 0], sampled_time + 1.0)


def test_resample_trajectory_history_rejects_invalid_interval() -> None:
    time = np.array([0.0, 0.1])
    pan = np.zeros((2, 3))
    particles = np.zeros((2, 1, 3))
    with pytest.raises(ValueError, match="interval_s"):
        resample_trajectory_history(time, pan, particles, interval_s=0.0)


def test_pan_rim_endpoints_follow_identity_and_positive_30_degree_pitch() -> None:
    local_endpoints, centerline_radius, rim_z = pan_rim_geometry_from_config(
        {
            "pan": {
                "collision_proxy": {
                    "inner_radius_m": 0.10,
                    "rim_radius_m": 0.12,
                    "rim_z_m": 0.06,
                }
            }
        }
    )
    assert centerline_radius == pytest.approx(0.11)
    assert rim_z == pytest.approx(0.06)
    np.testing.assert_allclose(
        local_endpoints,
        [[-0.11, 0.0, 0.06], [0.11, 0.0, 0.06]],
    )

    center = np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    half_pitch = np.deg2rad(30.0) * 0.5
    quaternions = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [np.cos(half_pitch), 0.0, np.sin(half_pitch), 0.0],
        ]
    )
    actual = pan_rim_endpoint_history(center, quaternions, local_endpoints)
    pitch = np.deg2rad(30.0)
    rotation = np.asarray(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch)],
        ]
    )
    expected = np.stack(
        (
            center[0] + local_endpoints,
            center[1] + local_endpoints @ rotation.T,
        )
    )
    np.testing.assert_allclose(actual, expected, atol=1.0e-12)
    assert actual[1, 1, 2] < actual[1, 0, 2]


class _FakePolicy:
    def predict(
        self,
        _observation: np.ndarray,
        *,
        deterministic: bool,
    ) -> tuple[np.ndarray, None]:
        assert deterministic
        return np.zeros(4, dtype=np.float32), None


class _FakeSAC:
    @staticmethod
    def load(_checkpoint: Path, *, device: str) -> _FakePolicy:
        assert device == "cpu"
        return _FakePolicy()


class _FakeEnvironment:
    reset_counts: list[int] = []

    def __init__(self, _config: dict[str, Any]) -> None:
        self.count_per_type = 0

    def reset(
        self,
        *,
        seed: int,
        options: dict[str, int],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        assert seed >= 0
        self.count_per_type = options["count_per_type"]
        self.reset_counts.append(self.count_per_type)
        total = 3 * self.count_per_type
        return np.zeros(2), {
            "actual_total_mass_kg": total * 0.001,
            "particle_count": total,
        }

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        np.testing.assert_array_equal(action, np.zeros(4))
        total = 3 * self.count_per_type
        time = np.asarray([0.0, 0.1, 0.2])
        pan = np.column_stack((time, np.zeros_like(time), 0.2 - 0.5 * time))
        pitch = np.deg2rad([0.0, 15.0, 30.0])
        pan_quaternion = np.column_stack(
            (
                np.cos(pitch * 0.5),
                np.zeros_like(pitch),
                np.sin(pitch * 0.5),
                np.zeros_like(pitch),
            )
        )
        particles = np.repeat(pan[:, None, :], total, axis=1)
        particles[:, :, 0] += np.linspace(-0.02, 0.02, total)
        final_score = 0.5 + total / 600.0
        spill_count = self.count_per_type // 20 - 1
        reward = final_score - 0.1 * spill_count
        return (
            np.zeros(2),
            reward,
            True,
            False,
            {
                "trajectory_valid": True,
                "particle_count": total,
                "actual_total_mass_kg": total * 0.001,
                "mixing": {
                    "initial_mixing_score": 0.25,
                    "final_mixing_score": final_score,
                    "mixing_improvement": final_score - 0.25,
                },
                "spill": {
                    "spill_count": spill_count,
                    "spill_count_ratio": spill_count / total,
                    "spill_mass_kg": spill_count * 0.001,
                    "spill_mass_ratio": spill_count / total,
                },
                "simulation_result": {
                    "time_s": time,
                    "pan_position_world_m": pan,
                    "pan_quaternion_wxyz": pan_quaternion,
                    "particle_positions_world_m": particles,
                },
            },
        )

    def close(self) -> None:
        return None


def test_policy_plots_use_only_latest_particle_strata_and_write_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "policy.zip"
    checkpoint.touch()
    output = tmp_path / "plots"
    _FakeEnvironment.reset_counts = []
    monkeypatch.setattr(plot_module, "SAC", _FakeSAC)
    monkeypatch.setattr(plot_module, "WokMixingEnv", _FakeEnvironment)
    monkeypatch.setattr(
        plot_module,
        "load_config",
        lambda _path: {
            "pan": {
                "collision_proxy": {
                    "inner_radius_m": 0.10,
                    "rim_radius_m": 0.12,
                    "rim_z_m": 0.06,
                }
            }
        },
    )

    overview = generate_policy_trajectory_plots(
        tmp_path / "config.yaml",
        checkpoint,
        output,
        seed=7,
        interval_s=0.1,
    )

    assert _FakeEnvironment.reset_counts == [20, 30, 40]
    assert overview == output / "trajectory_xz_comparison_60_90_120_particles.png"
    expected_plots = {
        "trajectory_3d_comparison_60_90_120_particles.png",
        "trajectory_xz_comparison_60_90_120_particles.png",
        "mixing_score_by_particle_count.png",
    }
    assert expected_plots <= {path.name for path in output.glob("*.png")}
    summary = json.loads((output / "trajectory_summary.json").read_text(encoding="utf-8"))
    assert [item["total_particle_count"] for item in summary["strata"]] == [60, 90, 120]
    assert [item["count_per_type"] for item in summary["strata"]] == [20, 30, 40]
    assert all("actual_mass_g" in item for item in summary["strata"])
    assert all("mixing" in item and "final_score" in item["mixing"] for item in summary["strata"])
    assert all("spill" in item and "count" in item["spill"] for item in summary["strata"])
    assert all("reward" in item for item in summary["strata"])
    for total in (60, 90, 120):
        with np.load(output / f"trajectory_particles_{total:03d}.npz") as archive:
            assert int(archive["total_particle_count"]) == total
            assert float(archive["sample_interval_s"]) == pytest.approx(0.1)
            assert archive["pan_rim_endpoints_world_m"].shape == (3, 2, 3)
            assert archive["pan_side_profile_final_world_m"].shape[1] == 3
            np.testing.assert_allclose(
                archive["pan_side_profile_final_world_m"][0],
                archive["pan_side_profile_final_world_m"][-1],
            )
            assert float(archive["pan_rim_center_radius_m"]) == pytest.approx(0.11)
            assert float(archive["pan_rim_z_m"]) == pytest.approx(0.06)
            np.testing.assert_array_equal(
                archive["pan_rim_endpoint_order"],
                ["rear / local -x", "front / local +x"],
            )
