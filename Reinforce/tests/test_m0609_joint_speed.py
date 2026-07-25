"""Nominal M0609 joint/TCP speed-only preview checks."""

from __future__ import annotations

import numpy as np
import pytest

from wok_sim.config import load_config
from wok_sim.robot.m0609_joint_speed import (
    M0609JointSpeedError,
    M0609JointSpeedSettings,
)
from wok_sim.trajectory.fried_rice import generate_fried_rice_trajectory
from wok_sim.visualization.episode_gif import normalized_action_from_physical_action


def test_joint90_preview_retimes_episode_74_geometry_to_nominal_90_percent() -> None:
    config = load_config("configs/fried_rice_m0609_joint90_preview.yaml")
    physical_action = (
        0.7084595643407352,
        0.6516548376320495,
        0.4366275715827942,
        0.007254653667349503,
    )
    action = normalized_action_from_physical_action(physical_action, config)

    trajectory = generate_fried_rice_trajectory(action, config)
    report = trajectory.joint_speed_report

    assert trajectory.validation is not None
    assert trajectory.validation.valid
    assert report is not None
    assert report["tcp_offset_m"] == pytest.approx(0.4)
    assert report["joint_velocity_limits_deg_s"] == pytest.approx(
        [150.0, 150.0, 180.0, 225.0, 225.0, 225.0]
    )
    assert max(report["joint_velocity_utilization"]) <= 0.902
    assert report["tcp_linear_velocity_utilization"] <= 0.902
    assert report["tcp_angular_velocity_utilization"] <= 0.902
    assert report["limiting_utilization"] == pytest.approx(0.90, abs=0.003)
    assert report["limiting_quantity"] == "tcp_angular_velocity"
    assert report["maximum_ik_position_error_m"] < 1.0e-6
    assert report["maximum_ik_orientation_error_rad"] < 1.0e-6
    assert trajectory.duration_s < 10.0


def test_joint_speed_settings_reject_zero_tcp_axis() -> None:
    with pytest.raises(M0609JointSpeedError, match="영벡터"):
        M0609JointSpeedSettings.from_mapping(
            {
                "q_teach_rad": np.zeros(6),
                "tcp_offset_axis": np.zeros(3),
            }
        )
