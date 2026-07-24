from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from wok_sim.robot import (
    DOOSAN_MOVESX_MAX_WAYPOINTS,
    SAFETY_STATUS,
    M0609CartesianCaps,
    M0609MotionContractError,
    TrajectoryExportData,
    build_doosan_offline_plan,
    export_doosan_offline_plan,
    validate_m0609_cartesian_motion,
)


def _data(
    *,
    time_s: np.ndarray | None = None,
    position_m: np.ndarray | None = None,
    rpy_rad: np.ndarray | None = None,
    linear_velocity: np.ndarray | None = None,
    angular_velocity: np.ndarray | None = None,
    linear_acceleration: np.ndarray | None = None,
    angular_acceleration: np.ndarray | None = None,
    linear_jerk: np.ndarray | None = None,
    angular_jerk: np.ndarray | None = None,
) -> TrajectoryExportData:
    timestamps = np.array([0.0, 0.5, 1.0]) if time_s is None else np.asarray(time_s, dtype=float)
    zeros = np.zeros((len(timestamps), 3))
    return TrajectoryExportData(
        timestamp_s=timestamps,
        pan_position_m=zeros.copy() if position_m is None else np.asarray(position_m, dtype=float),
        pan_rpy_rad=zeros.copy() if rpy_rad is None else np.asarray(rpy_rad, dtype=float),
        linear_velocity_m_s=(
            zeros.copy() if linear_velocity is None else np.asarray(linear_velocity, dtype=float)
        ),
        angular_velocity_rad_s=(
            zeros.copy() if angular_velocity is None else np.asarray(angular_velocity, dtype=float)
        ),
        linear_acceleration_m_s2=(
            zeros.copy()
            if linear_acceleration is None
            else np.asarray(linear_acceleration, dtype=float)
        ),
        angular_acceleration_rad_s2=(
            zeros.copy()
            if angular_acceleration is None
            else np.asarray(angular_acceleration, dtype=float)
        ),
        linear_jerk_m_s3=(
            zeros.copy() if linear_jerk is None else np.asarray(linear_jerk, dtype=float)
        ),
        angular_jerk_rad_s3=(
            zeros.copy() if angular_jerk is None else np.asarray(angular_jerk, dtype=float)
        ),
    )


def test_default_caps_are_provisional_and_below_nominal_tcp_speed() -> None:
    caps = M0609CartesianCaps()

    assert caps.linear_velocity_m_s == pytest.approx(0.25)
    assert caps.linear_velocity_m_s < 1.0
    assert caps.summary()["source"] == "provisional_engineering_caps_not_manufacturer_limits"


def test_caps_reject_nonpositive_unknown_and_above_nominal_speed() -> None:
    with pytest.raises(M0609MotionContractError, match="양수"):
        M0609CartesianCaps(linear_jerk_m_s3=0.0)
    with pytest.raises(M0609MotionContractError, match="nominal"):
        M0609CartesianCaps(linear_velocity_m_s=1.01)
    with pytest.raises(M0609MotionContractError, match="알 수 없는"):
        M0609CartesianCaps.from_mapping({"not_a_cap": 1.0})


def test_strict_validation_reports_caps_but_never_safety_approval() -> None:
    linear_velocity = np.zeros((3, 3))
    linear_velocity[1] = [0.15, 0.20, 0.0]  # norm is exactly 0.25 m/s

    report = validate_m0609_cartesian_motion(
        _data(linear_velocity=linear_velocity),
        payload_kg=0.8,
    )

    assert report.within_caps
    assert report.status == "within_provisional_cartesian_caps"
    assert report.safety_status == SAFETY_STATUS
    assert report.peaks["tcp_linear_velocity"] == pytest.approx(0.25)
    assert report.required_uniform_time_scale == pytest.approx(1.0)
    assert report.payload_within_nominal_reference is True
    assert "passed" not in report.status


@pytest.mark.parametrize(
    ("field", "quantity"),
    [
        ("linear_velocity", "tcp_linear_velocity"),
        ("angular_velocity", "tcp_angular_velocity"),
        ("linear_acceleration", "tcp_linear_acceleration"),
        ("angular_acceleration", "tcp_angular_acceleration"),
        ("linear_jerk", "tcp_linear_jerk"),
        ("angular_jerk", "tcp_angular_jerk"),
    ],
)
def test_each_cartesian_derivative_is_a_hard_preexport_gate(
    field: str,
    quantity: str,
) -> None:
    arrays = {field: np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 0.0]])}

    report = validate_m0609_cartesian_motion(_data(**arrays))

    assert not report.within_caps
    violation = next(item for item in report.violations if item.quantity == quantity)
    assert violation.observed == pytest.approx(10.0)
    assert violation.sample_index == 1
    assert violation.timestamp_s == pytest.approx(0.5)
    with pytest.raises(M0609MotionContractError, match=quantity):
        report.require_within_caps()


def test_tcp_offset_terms_are_validated_instead_of_only_pan_origin() -> None:
    angular_velocity = np.zeros((3, 3))
    angular_velocity[:, 2] = 1.0
    T_tcp_pan = np.eye(4)
    T_tcp_pan[0, 3] = -1.0  # pan->TCP offset is +1 m along pan x.

    report = validate_m0609_cartesian_motion(
        _data(angular_velocity=angular_velocity),
        caps=M0609CartesianCaps(
            linear_velocity_m_s=0.5,
            angular_velocity_rad_s=2.0,
        ),
        T_tcp_pan=T_tcp_pan,
    )

    assert not report.within_caps
    assert report.peaks["tcp_linear_velocity"] == pytest.approx(1.0)
    assert any(item.quantity == "tcp_linear_velocity" for item in report.violations)


def test_payload_above_nominal_reference_blocks_export() -> None:
    report = validate_m0609_cartesian_motion(_data(), payload_kg=6.01)

    assert not report.within_caps
    assert report.payload_within_nominal_reference is False
    assert report.required_uniform_time_scale is None
    assert report.violations[-1].quantity == "payload"


def test_report_estimates_uniform_time_scaling_for_derivative_orders() -> None:
    linear_velocity = np.zeros((3, 3))
    linear_acceleration = np.zeros((3, 3))
    linear_jerk = np.zeros((3, 3))
    linear_velocity[1, 0] = 0.5  # 2x cap => 2x time.
    linear_acceleration[1, 0] = 2.0  # 4x cap => sqrt(4)=2x time.
    linear_jerk[1, 0] = 16.0  # 8x cap => cbrt(8)=2x time.

    report = validate_m0609_cartesian_motion(
        _data(
            linear_velocity=linear_velocity,
            linear_acceleration=linear_acceleration,
            linear_jerk=linear_jerk,
        )
    )

    assert report.required_uniform_time_scale == pytest.approx(2.0)


def test_movesx_plan_has_doosan_parameter_shape_and_nonexecuting_safety_metadata() -> None:
    time_s = np.array([0.0, 0.25, 0.5])
    position = np.array([[0.0, 0.0, 0.0], [0.01, -0.02, 0.03], [0.02, -0.04, 0.06]])
    rpy = np.deg2rad(np.array([[0.0, 0.0, 0.0], [5.0, -10.0, 15.0], [10.0, -20.0, 30.0]]))

    plan = build_doosan_offline_plan(
        _data(time_s=time_s, position_m=position, rpy_rad=rpy),
        T_tcp_pan=np.eye(4),
        payload_kg=0.8,
        function="movesx",
        waypoint_stride=2,
    )

    assert plan["executable"] is False
    assert plan["plan_type"] == "non_executable_offline_parameter_template"
    assert plan["function_name"] == "movesx"
    assert plan["safety"]["status"] == SAFETY_STATUS
    assert plan["safety"]["command_ready"] is False
    assert plan["safety"]["controller_revalidation_required"] is True
    assert plan["motion_contract"]["within_caps"] is True
    assert plan["coordinate_convention"]["tcp_transform_source"] == "configured_T_tcp_pan"
    assert plan["coordinate_convention"]["orientation_type"] == "DR_FIX_XYZ"
    assert plan["position_constructor_template"]["ori_type"] == "DR_FIX_XYZ"
    parameters = plan["function_parameters"]
    assert set(("pos_list", "vel", "acc", "time", "ref", "mod", "vel_opt")) <= set(parameters)
    assert len(parameters["pos_list"]) == 2
    np.testing.assert_allclose(parameters["pos_list"][-1][:3], [20.0, -40.0, 60.0])
    np.testing.assert_allclose(parameters["pos_list"][-1][3:], [10.0, -20.0, 30.0])
    assert parameters["vel"] == pytest.approx([250.0, 30.0])
    assert parameters["acc"] == pytest.approx([500.0, 60.0])


def test_missing_tcp_transform_is_explicit_placeholder_not_command_ready() -> None:
    plan = build_doosan_offline_plan(_data())

    assert plan["coordinate_convention"]["tcp_transform_source"] == "pan_pose_identity_placeholder"
    assert "T_tcp_pan_missing_pan_pose_used_as_identity_placeholder" in plan["safety"]["reasons"]
    assert plan["safety"]["command_ready"] is False


def test_movel_plan_encodes_reference_segment_timing() -> None:
    plan = build_doosan_offline_plan(
        _data(time_s=np.array([0.0, 0.2, 0.5])),
        function="movel",
    )

    segments = plan["function_parameters"]["segments"]
    assert [segment["reference_segment_duration_s"] for segment in segments] == pytest.approx(
        [0.2, 0.3]
    )
    assert all(segment["parameters"]["time"] is None for segment in segments)
    assert all(segment["parameters"]["radius"] == 0.0 for segment in segments)


def test_movesx_automatically_respects_single_command_waypoint_limit() -> None:
    data = _data(time_s=np.linspace(0.0, 2.0, 205))

    plan = build_doosan_offline_plan(data, function="movesx")

    selection = plan["waypoint_selection"]
    assert selection["source_sample_count"] == 205
    assert selection["stride_source"] == "automatic_controller_waypoint_limit"
    assert selection["exported_waypoint_count"] <= DOOSAN_MOVESX_MAX_WAYPOINTS
    assert selection["single_movesx_waypoint_limit"] == DOOSAN_MOVESX_MAX_WAYPOINTS

    with pytest.raises(M0609MotionContractError, match="100개"):
        build_doosan_offline_plan(data, function="movesx", waypoint_stride=1)
    with pytest.raises(M0609MotionContractError, match="양의 정수"):
        build_doosan_offline_plan(data, function="movesx", waypoint_stride=1.0)


def test_plan_refuses_any_invalid_motion() -> None:
    velocity = np.zeros((3, 3))
    velocity[1, 0] = 0.250000001

    with pytest.raises(M0609MotionContractError, match="tcp_linear_velocity"):
        build_doosan_offline_plan(_data(linear_velocity=velocity))


def test_json_csv_and_python_exports_preserve_offline_safety_contract(tmp_path) -> None:
    data = _data()
    json_path = export_doosan_offline_plan(data, tmp_path / "plan.json")
    csv_path = export_doosan_offline_plan(data, tmp_path / "plan.csv")
    python_path = export_doosan_offline_plan(data, tmp_path / "plan.py")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["executable"] is False
    assert payload["safety"]["status"] == SAFETY_STATUS

    frame = pd.read_csv(csv_path)
    assert set(
        (
            "safety_status",
            "controller_revalidation_required",
            "linear_velocity_cap_mm_s",
            "angular_acceleration_cap_deg_s2",
            "linear_jerk_cap_mm_s3",
        )
    ) <= set(frame.columns)
    assert set(frame["safety_status"]) == {SAFETY_STATUS}
    assert not frame["executable"].any()

    source = python_path.read_text(encoding="utf-8")
    compile(source, str(python_path), "exec")
    assert "DOOSAN_OFFLINE_PLAN =" in source
    assert "movesx(" not in source
    assert "movel(" not in source
    assert "\nimport " not in source


def test_invalid_tcp_transform_and_export_extension_are_rejected(tmp_path) -> None:
    non_rigid = np.eye(4)
    non_rigid[0, 0] = 2.0
    with pytest.raises(ValueError, match="T_tcp_pan"):
        validate_m0609_cartesian_motion(_data(), T_tcp_pan=non_rigid)

    with pytest.raises(M0609MotionContractError, match="확장자"):
        export_doosan_offline_plan(_data(), tmp_path / "plan.txt")


def test_cap_mapping_converts_degrees_only_at_doosan_boundary() -> None:
    caps = M0609CartesianCaps.from_mapping(
        {
            "linear_velocity_m_s": 0.1,
            "angular_velocity_rad_s": math.pi / 4.0,
            "linear_acceleration_m_s2": 0.2,
            "angular_acceleration_rad_s2": math.pi / 2.0,
            "linear_jerk_m_s3": 1.0,
            "angular_jerk_rad_s3": math.pi,
        }
    )

    plan = build_doosan_offline_plan(_data(), caps=caps)

    assert plan["function_parameters"]["vel"] == pytest.approx([100.0, 45.0])
    assert plan["function_parameters"]["acc"] == pytest.approx([200.0, 90.0])
