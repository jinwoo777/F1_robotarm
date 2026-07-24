from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wok_sim.config import load_config
from wok_sim.robot import TrajectoryExportData, export_trajectory
from wok_sim.trajectory import generate_trajectory


def test_csv_export_has_required_monotonic_finite_columns(tmp_path) -> None:
    config = load_config("configs/test.yaml")
    trajectory = generate_trajectory(np.zeros(7), config)
    output = tmp_path / "trajectory.csv"

    export_trajectory(TrajectoryExportData.from_trajectory(trajectory), output)

    assert output.is_file()
    frame = pd.read_csv(output)
    required = {
        "timestamp_s",
        "pan_position_m_x",
        "pan_position_m_y",
        "pan_position_m_z",
        "pan_quaternion_wxyz_w",
        "tcp_position_m_x",
        "linear_velocity_m_s_x",
        "angular_velocity_rad_s_y",
        "linear_acceleration_m_s2_z",
        "angular_acceleration_rad_s2_y",
        "linear_jerk_m_s3_x",
        "angular_jerk_rad_s3_y",
        "pan_linear_velocity_m_s_x",
        "pan_angular_acceleration_rad_s2_y",
        "tcp_linear_velocity_m_s_x",
        "tcp_angular_acceleration_rad_s2_y",
        "tcp_linear_jerk_m_s3_x",
    }
    assert required <= set(frame.columns)
    assert np.all(np.diff(frame["timestamp_s"]) > 0.0)
    assert np.isfinite(frame.to_numpy(dtype=float)).all()


def test_npz_and_json_export(tmp_path) -> None:
    config = load_config("configs/test.yaml")
    data = TrajectoryExportData.from_trajectory(generate_trajectory(np.zeros(7), config))

    npz = export_trajectory(data, tmp_path / "trajectory.npz")
    json_path = export_trajectory(data, tmp_path / "trajectory.json")

    with np.load(npz) as payload:
        assert "timestamp_s" in payload.files
        assert "pan_linear_velocity_m_s_x" in payload.files
        assert "tcp_linear_velocity_m_s_x" in payload.files
        assert payload["robot_validation_status"].item() == "not_evaluated"
    assert '"tcp_transform_source"' in json_path.read_text(encoding="utf-8")


def test_tcp_derivatives_include_rigid_body_offset_terms(tmp_path) -> None:
    """회전하는 pan에서 offset TCP의 접선·구심·jerk 항을 내보낸다."""

    time_s = np.array([0.0, 0.25, 0.5])
    yaw = time_s**3
    angular_speed = 3.0 * time_s**2
    angular_acceleration = 6.0 * time_s
    angular_jerk = np.full_like(time_s, 6.0)
    zeros = np.zeros((time_s.size, 3))
    angular_velocity = np.zeros_like(zeros)
    angular_velocity[:, 2] = angular_speed
    angular_acceleration_vectors = np.zeros_like(zeros)
    angular_acceleration_vectors[:, 2] = angular_acceleration
    angular_jerk_vectors = np.zeros_like(zeros)
    angular_jerk_vectors[:, 2] = angular_jerk
    data = TrajectoryExportData(
        timestamp_s=time_s,
        pan_position_m=zeros.copy(),
        pan_rpy_rad=np.column_stack((np.zeros_like(yaw), np.zeros_like(yaw), yaw)),
        linear_velocity_m_s=zeros.copy(),
        angular_velocity_rad_s=angular_velocity,
        linear_acceleration_m_s2=zeros.copy(),
        angular_acceleration_rad_s2=angular_acceleration_vectors,
        linear_jerk_m_s3=zeros.copy(),
        angular_jerk_rad_s3=angular_jerk_vectors,
    )
    # T_pan_tcp의 translation이 [+1, 0, 0]이 되도록 그 역변환을 설정한다.
    T_tcp_pan = np.eye(4)
    T_tcp_pan[0, 3] = -1.0
    output = export_trajectory(
        data,
        tmp_path / "offset.csv",
        T_tcp_pan=T_tcp_pan,
    )
    frame = pd.read_csv(output)

    offset_world = np.column_stack((np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)))
    tangent_world = np.column_stack((-np.sin(yaw), np.cos(yaw), np.zeros_like(yaw)))
    expected_velocity = angular_speed[:, None] * tangent_world
    expected_acceleration = (
        angular_acceleration[:, None] * tangent_world - angular_speed[:, None] ** 2 * offset_world
    )
    expected_jerk = (angular_jerk - angular_speed**3)[:, None] * tangent_world - (
        3.0 * angular_speed * angular_acceleration
    )[:, None] * offset_world

    def xyz(prefix: str) -> np.ndarray:
        return frame[[f"{prefix}_{axis}" for axis in "xyz"]].to_numpy()

    np.testing.assert_allclose(xyz("tcp_position_m"), offset_world, atol=1e-14)
    np.testing.assert_allclose(xyz("tcp_linear_velocity_m_s"), expected_velocity, atol=1e-14)
    np.testing.assert_allclose(
        xyz("tcp_linear_acceleration_m_s2"), expected_acceleration, atol=1e-14
    )
    np.testing.assert_allclose(xyz("tcp_linear_jerk_m_s3"), expected_jerk, atol=1e-14)
    np.testing.assert_allclose(xyz("pan_linear_velocity_m_s"), 0.0, atol=0.0)
    np.testing.assert_allclose(xyz("linear_velocity_m_s"), 0.0, atol=0.0)
    np.testing.assert_allclose(xyz("tcp_angular_velocity_rad_s"), angular_velocity, atol=0.0)
    np.testing.assert_allclose(
        xyz("tcp_angular_acceleration_rad_s2"),
        angular_acceleration_vectors,
        atol=0.0,
    )
    np.testing.assert_allclose(xyz("tcp_angular_jerk_rad_s3"), angular_jerk_vectors, atol=0.0)


def test_export_rejects_non_rigid_tcp_transform(tmp_path) -> None:
    config = load_config("configs/test.yaml")
    data = TrajectoryExportData.from_trajectory(generate_trajectory(np.zeros(7), config))
    non_rigid = np.eye(4)
    non_rigid[0, 0] = 2.0

    with pytest.raises(ValueError, match="T_tcp_pan"):
        export_trajectory(
            data,
            tmp_path / "invalid.csv",
            T_tcp_pan=non_rigid,
        )
