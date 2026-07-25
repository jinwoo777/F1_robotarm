from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from matplotlib import pyplot as plt

from wok_sim.visualization import teaching_preview as preview_module
from wok_sim.visualization.teaching_preview import (
    _draw_metrics,
    _draw_teaching_geometry,
    _draw_world_xz,
    _preview_policy_action,
    _reward_npz_scalars,
)
from wok_sim.visualization.trajectory_plot import PAN_RIM_COLOR


def _reward_info() -> dict[str, Any]:
    terms = {
        "mix": 0.6,
        "lift": 0.9,
        "spill": -1.6,
        "jerk": -0.02,
        "acceleration": -0.03,
        "height": 0.0,
        "invalid": 0.0,
    }
    return {
        "particle_count": 60,
        "mixing": {
            "initial_mixing_score": 0.2,
            "final_mixing_score": 0.4,
            "mixing_improvement": 0.2,
        },
        "lift": {
            "lift_score": 0.4,
            "lifted_particle_count": 18,
            "lifted_particle_ratio": 0.3,
            "peak_lifted_particle_ratio": 0.25,
            "top_margin_m": 0.001,
            "maximum_grain_top_clearance_m": 0.034,
        },
        "spill": {
            "spill_count": 6,
            "spill_count_ratio": 0.1,
            "spill_mass_kg": 0.004,
            "spill_mass_ratio": 0.05,
        },
        "reward_terms": terms,
        "reward_signals": {
            "lifted_particle_count": 18.0,
            "lift_reward_per_particle": 0.05,
            "spill_severity": 0.1,
            "spill_linear_penalty": 1.2,
            "spill_quadratic_penalty": 0.4,
        },
        "final_reward": sum(terms.values()),
    }


def test_reward_panel_draws_signed_main_contributions_and_raw_signals() -> None:
    figure, axis = plt.subplots()
    try:
        _draw_metrics(axis, _reward_info())

        np.testing.assert_allclose(
            [patch.get_width() for patch in axis.patches],
            [0.6, 0.9, -1.6],
        )
        assert [tick.get_text() for tick in axis.get_yticklabels()] == [
            "mix delta",
            "lift bonus",
            "spill penalty (nonlinear)",
        ]
        assert any(np.allclose(line.get_xdata(), [0.0, 0.0]) for line in axis.lines)
        text = "\n".join(item.get_text() for item in axis.texts)
        assert "mix 0.200 → 0.400 (Δ +0.200)" in text
        assert "lifted 18/60 (30.0%)" in text
        assert "0.05/grain" in text
        assert "quadratic 0.400" in text
        assert "total reward -0.150" in text
    finally:
        plt.close(figure)


def test_reward_npz_fields_are_numeric_and_load_without_pickle(tmp_path: Path) -> None:
    fields = _reward_npz_scalars(_reward_info())
    output = tmp_path / "reward_fields.npz"
    np.savez_compressed(output, **fields)

    with np.load(output, allow_pickle=False) as archive:
        assert set(archive.files) == set(fields)
        assert all(archive[name].dtype.kind in "fiu" for name in archive.files)
        assert float(archive["reward_term_spill"]) == -1.6
        assert int(archive["lifted_particle_count"]) == 18
        assert np.isclose(float(archive["reward_total"]), -0.15)


def test_world_and_first_cycle_panels_draw_sampled_rim_endpoint_paths() -> None:
    time = np.arange(0.0, 1.01, 0.1)
    pan = np.column_stack((0.2 * time, np.zeros_like(time), 0.2 - 0.1 * time))
    endpoints = np.repeat(pan[:, None, :], 2, axis=1)
    endpoints[:, 0, 0] -= 0.11
    endpoints[:, 1, 0] += 0.11
    endpoints[:, 0, 2] += 0.04 + 0.03 * time
    endpoints[:, 1, 2] += 0.04 - 0.03 * time
    particles = pan[:, None, :]
    species = np.asarray(["small_sphere"])
    trajectory = {
        "time_s": time,
        "position_m": pan,
        "orientation_rpy_rad": np.column_stack(
            (
                np.zeros_like(time),
                np.deg2rad(
                    np.interp(
                        time,
                        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        [0.0, 30.0, 30.0, 15.0, 0.0, 0.0],
                    )
                ),
                np.zeros_like(time),
            )
        ),
        "parameters": {
            "cycle_time": 0.8,
            "tilt_out_phase_duration": 0.2,
            "descent_phase_duration": 0.2,
            "partial_recovery_phase_duration": 0.2,
            "descent_angle": np.deg2rad(45.0),
            "pan_tilt_angle": np.deg2rad(30.0),
            "lift_angle": np.deg2rad(15.0),
            "tilt_recovery_angle": np.deg2rad(15.0),
        },
    }

    figure, (world_axis, cycle_axis) = plt.subplots(1, 2)
    try:
        _draw_world_xz(
            world_axis,
            time,
            pan,
            particles,
            species,
            0.11,
            endpoints,
        )
        _draw_teaching_geometry(
            cycle_axis,
            trajectory,
            0.11,
            sampled_time_s=time,
            sampled_pan=pan,
            sampled_rim_endpoints=endpoints,
        )

        world_blue = [line for line in world_axis.lines if line.get_color() == PAN_RIM_COLOR]
        cycle_blue = [line for line in cycle_axis.lines if line.get_color() == PAN_RIM_COLOR]
        assert any(np.allclose(line.get_xdata(), endpoints[:, 0, 0]) for line in world_blue)
        first_cycle = time <= 0.8 + 1.0e-9
        assert any(
            np.allclose(line.get_xdata(), endpoints[first_cycle, 1, 0]) for line in cycle_blue
        )
        labels = "\n".join(text.get_text() for text in cycle_axis.texts)
        assert "lift 15.0° → residual 15.0°" in labels
        assert "P3 residual 15.0°" in labels
    finally:
        plt.close(figure)


def test_preview_policy_action_supports_center_and_cpu_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = SimpleNamespace(action_space=SimpleNamespace(shape=(4,)))
    observation = np.asarray([0.25, -0.5], dtype=np.float32)
    center, center_source, center_checkpoint = _preview_policy_action(
        environment,
        observation,
        None,
    )
    np.testing.assert_array_equal(center, np.zeros(4, dtype=np.float32))
    assert center_source == "center_zero_action"
    assert center_checkpoint is None

    checkpoint = tmp_path / "policy.zip"
    checkpoint.touch()

    class _FakePolicy:
        action_space = SimpleNamespace(shape=(4,))

        def predict(
            self,
            actual_observation: np.ndarray,
            *,
            deterministic: bool,
        ) -> tuple[np.ndarray, None]:
            np.testing.assert_array_equal(actual_observation, observation)
            assert deterministic
            return np.asarray([0.1, -0.2, 0.3, -0.4], dtype=np.float32), None

    class _FakeSAC:
        @staticmethod
        def load(actual_checkpoint: Path, *, device: str) -> _FakePolicy:
            assert actual_checkpoint == checkpoint
            assert device == "cpu"
            return _FakePolicy()

    monkeypatch.setattr(preview_module, "SAC", _FakeSAC)
    action, source, resolved = _preview_policy_action(
        environment,
        observation,
        checkpoint,
    )
    np.testing.assert_allclose(action, [0.1, -0.2, 0.3, -0.4])
    assert source == "sac_checkpoint"
    assert resolved == checkpoint.resolve()


def test_preview_policy_action_rejects_checkpoint_action_shape_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "old_3d_policy.zip"
    checkpoint.touch()
    environment = SimpleNamespace(action_space=SimpleNamespace(shape=(4,)))

    class _OldPolicy:
        action_space = SimpleNamespace(shape=(3,))

    class _OldSAC:
        @staticmethod
        def load(_checkpoint: Path, *, device: str) -> _OldPolicy:
            assert device == "cpu"
            return _OldPolicy()

    monkeypatch.setattr(preview_module, "SAC", _OldSAC)
    with pytest.raises(ValueError, match=r"4D action space.*environment=\(3,\)"):
        _preview_policy_action(
            SimpleNamespace(action_space=SimpleNamespace(shape=(3,))),
            np.zeros(2, dtype=np.float32),
            None,
        )
    with pytest.raises(ValueError, match=r"checkpoint=\(3,\).*environment=\(4,\)"):
        _preview_policy_action(
            environment,
            np.zeros(2, dtype=np.float32),
            checkpoint,
        )
