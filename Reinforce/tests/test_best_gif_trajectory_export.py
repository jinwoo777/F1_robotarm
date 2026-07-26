from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "export_best_gif_trajectories.py"
)
SPEC = importlib.util.spec_from_file_location("export_best_gif_trajectories", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)

_nearest_source_indices = EXPORTER._nearest_source_indices
_particle_dataframe = EXPORTER._particle_dataframe
_resample_quaternions_wxyz = EXPORTER._resample_quaternions_wxyz
_sample_replay_arrays = EXPORTER._sample_replay_arrays


def test_resample_quaternion_uses_slerp_and_wxyz_order() -> None:
    source_time = np.asarray([0.0, 1.0])
    source_quaternion = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0],
        ]
    )

    sampled = _resample_quaternions_wxyz(
        source_time,
        source_quaternion,
        np.asarray([0.0, 0.5, 1.0]),
    )

    np.testing.assert_allclose(np.linalg.norm(sampled, axis=1), 1.0)
    np.testing.assert_allclose(
        sampled[1],
        [np.cos(np.pi / 8.0), 0.0, np.sin(np.pi / 8.0), 0.0],
    )


def test_nearest_source_indices_breaks_ties_toward_earlier_sample() -> None:
    indices = _nearest_source_indices(
        np.asarray([0.0, 0.2, 0.4]),
        np.asarray([0.0, 0.1, 0.31, 0.4]),
    )

    np.testing.assert_array_equal(indices, [0, 0, 2, 2])


def test_sampled_replay_and_particle_csv_preserve_frame_particle_mapping() -> None:
    raw_time = np.asarray([0.0, 0.1])
    particles_world = np.asarray(
        [
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[1.1, 0.0, 0.0], [2.1, 0.0, 0.0]],
        ]
    )
    result = {
        "time_s": raw_time,
        "pan_quaternion_wxyz": np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
        "particle_velocities_world_m_s": np.ones((2, 2, 3)),
        "contact_with_pan": np.asarray([[True, False], [False, True]]),
        "contact_normal_force_n": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        "crossed_spill_boundary": np.asarray([False, True]),
    }
    run = SimpleNamespace(
        info={
            "simulation_result": result,
            "particle_batch": {
                "species": np.asarray(["rice", "pea"]),
                "radii_m": np.asarray([0.01, 0.02]),
                "masses_kg": np.asarray([0.001, 0.002]),
            },
        },
        frame_time_s=raw_time,
        frame_pan_world_m=np.zeros((2, 3)),
        frame_particles_world_m=particles_world,
        frame_rim_endpoints_world_m=np.zeros((2, 2, 3)),
    )

    arrays = _sample_replay_arrays(run)
    frame = _particle_dataframe(
        arrays,
        condition_mass_g=60,
        actual_mass_g=60.0,
        episode_id=49,
    )

    assert len(frame) == 4
    assert frame["frame_index"].tolist() == [0, 0, 1, 1]
    assert frame["particle_index"].tolist() == [0, 1, 0, 1]
    assert frame["species"].tolist() == ["rice", "pea", "rice", "pea"]
    np.testing.assert_allclose(
        frame["position_pan_m_x"],
        [1.0, 2.0, 1.1, 2.1],
    )
    assert frame["crossed_spill_boundary"].tolist() == [False, True, False, True]
