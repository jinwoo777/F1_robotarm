from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from wok_sim.config import load_config
from wok_sim.geometry.transforms import (
    FrameTransformError,
    compose_transform,
    invert_transform,
    quaternion_to_matrix,
    resolve_wok_frame_context,
    rpy_to_matrix,
    transform_to_pose,
    validate_homogeneous_transform,
)
from wok_sim.simulation import PanModel
from wok_sim.trajectory import generate_trajectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_nonidentity_wok_frame_and_teaching_pair_define_initial_pan_pose() -> None:
    T_base_wok = compose_transform(
        [0.42, -0.18, 0.31],
        rpy_rad=[0.12, -0.08, 0.35],
    )
    T_wok_pan0 = compose_transform(
        [0.06, -0.02, 0.17],
        rpy_rad=[0.04, 0.21, -0.03],
    )
    expected_T_base_pan0 = T_base_wok @ T_wok_pan0
    T_tcp_pan = compose_transform(
        [0.02, 0.01, -0.09],
        rpy_rad=[0.0, np.pi / 2.0, 0.0],
    )
    T_base_tcp_teach = expected_T_base_pan0 @ invert_transform(T_tcp_pan)

    pan_position, _, pan_quaternion = transform_to_pose(expected_T_base_pan0)
    start_position, start_rpy, _ = transform_to_pose(T_wok_pan0)
    context = resolve_wok_frame_context(
        {
            "robot": {
                "T_base_wok": T_base_wok.tolist(),
                "T_base_tcp_teach": T_base_tcp_teach.tolist(),
                "T_tcp_pan": T_tcp_pan.tolist(),
            },
            "pan": {
                "initial_pose": {
                    "position_m": pan_position.tolist(),
                    "quaternion_wxyz": pan_quaternion.tolist(),
                }
            },
            "trajectory": {
                "start_position_m": start_position.tolist(),
                "start_rpy_rad": start_rpy.tolist(),
            },
        }
    )

    assert context.source == "robot.teaching"
    np.testing.assert_allclose(context.T_base_wok, T_base_wok, atol=1.0e-12)
    np.testing.assert_allclose(context.T_base_pan0, expected_T_base_pan0, atol=1.0e-12)
    np.testing.assert_allclose(context.T_wok_pan0, T_wok_pan0, atol=1.0e-12)


def test_trajectory_is_validated_in_wok_frame_and_exposed_in_base_frame() -> None:
    config = deepcopy(load_config(PROJECT_ROOT / "configs" / "test.yaml"))
    T_base_wok = compose_transform(
        [0.42, -0.18, 0.0],
        rpy_rad=[0.0, 0.0, 0.55],
    )
    T_wok_pan0 = compose_transform(
        [0.0, 0.0, 0.18],
        rpy_rad=[0.0, 0.0, 0.15],
    )
    T_base_pan0 = T_base_wok @ T_wok_pan0
    T_tcp_pan = compose_transform(
        [0.02, 0.01, -0.09],
        rpy_rad=[0.0, 0.30, 0.0],
    )
    T_base_tcp_teach = T_base_pan0 @ invert_transform(T_tcp_pan)
    base_position, _, base_quaternion = transform_to_pose(T_base_pan0)
    wok_position, wok_rpy, _ = transform_to_pose(T_wok_pan0)
    config["robot"].update(
        {
            "T_base_wok": T_base_wok.tolist(),
            "T_base_tcp_teach": T_base_tcp_teach.tolist(),
            "T_tcp_pan": T_tcp_pan.tolist(),
        }
    )
    config["pan"]["initial_pose"] = {
        "position_m": base_position.tolist(),
        "quaternion_wxyz": base_quaternion.tolist(),
    }
    config["trajectory"]["start_position_m"] = wok_position.tolist()
    config["trajectory"]["start_rpy_rad"] = wok_rpy.tolist()

    trajectory = generate_trajectory(np.zeros(7), config)
    rotation = T_base_wok[:3, :3]
    translation = T_base_wok[:3, 3]

    assert trajectory.valid is True
    assert np.min(trajectory.position_m[:, 0]) > config["trajectory"]["workspace"]["x_m"][1]
    np.testing.assert_allclose(
        trajectory.position_m,
        trajectory.position_wok_m @ rotation.T + translation,
        atol=1.0e-12,
    )
    for base_values, wok_values in (
        (trajectory.linear_velocity_m_s, trajectory.linear_velocity_wok_m_s),
        (trajectory.angular_velocity_rad_s, trajectory.angular_velocity_wok_rad_s),
        (trajectory.linear_acceleration_m_s2, trajectory.linear_acceleration_wok_m_s2),
        (trajectory.angular_acceleration_rad_s2, trajectory.angular_acceleration_wok_rad_s2),
        (trajectory.linear_jerk_m_s3, trajectory.linear_jerk_wok_m_s3),
        (trajectory.angular_jerk_rad_s3, trajectory.angular_jerk_wok_rad_s3),
    ):
        np.testing.assert_allclose(base_values, wok_values @ rotation.T, atol=1.0e-11)
    for base_rpy, wok_rpy_sample in zip(
        trajectory.orientation_rpy_rad,
        trajectory.orientation_wok_rpy_rad,
        strict=True,
    ):
        np.testing.assert_allclose(
            rpy_to_matrix(base_rpy),
            rotation @ rpy_to_matrix(wok_rpy_sample),
            atol=1.0e-12,
        )

    pan = PanModel.from_config(config)
    pan_position, pan_quaternion = pan.initial_pose
    np.testing.assert_allclose(pan_position, T_base_pan0[:3, 3], atol=1.0e-12)
    np.testing.assert_allclose(
        quaternion_to_matrix(pan_quaternion),
        T_base_pan0[:3, :3],
        atol=1.0e-12,
    )


def test_validate_homogeneous_transform_rejects_bad_bottom_row() -> None:
    invalid = np.eye(4)
    invalid[3, 0] = 0.1

    with pytest.raises(FrameTransformError, match="마지막 행"):
        validate_homogeneous_transform(invalid, name="bad_transform")


@pytest.mark.parametrize(
    "rotation",
    [
        np.diag([-1.0, 1.0, 1.0]),
        np.array(
            [
                [1.0, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    ],
)
def test_validate_homogeneous_transform_rejects_non_so3_rotation(
    rotation: np.ndarray,
) -> None:
    invalid = np.eye(4)
    invalid[:3, :3] = rotation

    with pytest.raises(FrameTransformError):
        validate_homogeneous_transform(invalid)


def test_teaching_pose_mismatch_with_explicit_pan_pose_is_rejected() -> None:
    with pytest.raises(FrameTransformError, match="pan.initial_pose"):
        resolve_wok_frame_context(
            {
                "robot": {
                    "T_base_tcp_teach": np.eye(4).tolist(),
                    "T_tcp_pan": np.eye(4).tolist(),
                },
                "pan": {
                    "initial_pose": {
                        "position_m": [0.001, 0.0, 0.0],
                        "rpy_rad": [0.0, 0.0, 0.0],
                    }
                },
                "simulation": {
                    "frame_position_tolerance_m": 1.0e-5,
                    "frame_orientation_tolerance_rad": 1.0e-5,
                },
            }
        )


def test_explicit_wok_local_trajectory_start_mismatch_is_rejected() -> None:
    with pytest.raises(FrameTransformError, match="trajectory.start"):
        resolve_wok_frame_context(
            {
                "pan": {
                    "initial_pose": {
                        "position_m": [0.0, 0.0, 0.12],
                        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    }
                },
                "trajectory": {
                    "start_position_m": [0.01, 0.0, 0.12],
                    "start_rpy_rad": [0.0, 0.0, 0.0],
                },
            }
        )


def test_tcp_pan_without_teaching_pose_is_allowed_for_robot_validator() -> None:
    context = resolve_wok_frame_context(
        {
            "robot": {"T_tcp_pan": np.eye(4).tolist()},
            "pan": {
                "initial_pose": {
                    "position_m": [0.0, 0.0, 0.12],
                    "rpy_rad": [0.0, 0.0, 0.0],
                }
            },
        }
    )

    assert context.source == "pan.initial_pose"
    np.testing.assert_allclose(context.T_base_pan0[:3, 3], [0.0, 0.0, 0.12])


def test_teaching_pose_without_tcp_pan_is_rejected() -> None:
    with pytest.raises(FrameTransformError, match="T_tcp_pan"):
        resolve_wok_frame_context({"robot": {"T_base_tcp_teach": np.eye(4).tolist()}})
