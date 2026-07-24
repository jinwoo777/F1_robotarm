"""사용자 제공 URDF에만 의존하는 M0609 기구학 validator."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from wok_sim.geometry.transforms import invert_transform

from .base import RobotValidationError, RobotValidationResult


def pan_to_tcp_transforms(
    pan_transforms: np.ndarray,
    T_tcp_pan: np.ndarray,
) -> np.ndarray:
    """T_base_pan * inverse(T_tcp_pan)으로 TCP trajectory를 계산한다."""

    pans = np.asarray(pan_transforms, dtype=float)
    tcp_pan = np.asarray(T_tcp_pan, dtype=float)
    if pans.ndim != 3 or pans.shape[1:] != (4, 4):
        raise ValueError("pan_transforms는 (T, 4, 4) 배열이어야 합니다.")
    if tcp_pan.shape != (4, 4):
        raise ValueError("T_tcp_pan은 4x4 행렬이어야 합니다.")
    inverse = invert_transform(tcp_pan)
    return np.einsum("tij,jk->tik", pans, inverse)


class M0609Validator:
    """Pinocchio 기반 sequential IK와 관절 제한 검사.

    M0609 고유 제한을 코드에 넣지 않는다. 모든 제한은 URDF 또는 명시적
    config에서만 읽는다.
    """

    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)

    def preflight(self) -> RobotValidationResult | None:
        """비활성/누락 상태를 판정하고 검증 가능하면 None을 반환한다."""

        enabled = bool(self.config.get("enabled", False))
        required = bool(self.config.get("required", False))
        urdf_raw = self.config.get("urdf_path")
        if not enabled:
            if required:
                raise RobotValidationError("robot.required=true이지만 robot.enabled=false입니다.")
            return RobotValidationResult.not_evaluated(
                "robot.enabled=false: M0609 기구학을 평가하지 않았습니다."
            )
        if not urdf_raw:
            message = "M0609 URDF 경로가 없어 robot validation을 평가하지 않았습니다."
            if required:
                raise RobotValidationError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            return RobotValidationResult.not_evaluated(message)
        urdf_path = Path(str(urdf_raw)).expanduser().resolve()
        if not urdf_path.is_file():
            message = f"M0609 URDF 파일을 찾을 수 없습니다: {urdf_path}"
            if required:
                raise RobotValidationError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            return RobotValidationResult.not_evaluated(message)
        if self.config.get("q_teach") is None:
            message = "robot.q_teach가 없어 sequential IK seed를 만들 수 없습니다."
            if required:
                raise RobotValidationError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            return RobotValidationResult.not_evaluated(message)
        if self.config.get("tcp_link") is None:
            message = "robot.tcp_link가 없어 TCP frame을 식별할 수 없습니다."
            if required:
                raise RobotValidationError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            return RobotValidationResult.not_evaluated(message)
        return None

    def validate(
        self,
        timestamps_s: np.ndarray,
        pan_transforms: np.ndarray,
    ) -> RobotValidationResult:
        """전체 pan trajectory를 TCP로 변환해 sequential IK를 검사한다."""

        early = self.preflight()
        if early is not None:
            return early
        T_tcp_pan_raw = self.config.get("T_tcp_pan")
        if T_tcp_pan_raw is None:
            message = "robot.T_tcp_pan이 없어 pan pose를 TCP pose로 변환할 수 없습니다."
            if bool(self.config.get("required", False)):
                raise RobotValidationError(message)
            return RobotValidationResult.not_evaluated(message)
        timestamps = np.asarray(timestamps_s, dtype=float)
        if timestamps.ndim != 1 or len(timestamps) != len(pan_transforms):
            raise ValueError("timestamp와 pan transform 길이가 일치해야 합니다.")
        if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
            raise ValueError("timestamp는 2개 이상이며 엄격히 증가해야 합니다.")
        targets = pan_to_tcp_transforms(
            np.asarray(pan_transforms, dtype=float),
            np.asarray(T_tcp_pan_raw, dtype=float),
        )
        return self._validate_with_pinocchio(timestamps, targets)

    def _validate_with_pinocchio(
        self,
        timestamps: np.ndarray,
        target_transforms: np.ndarray,
    ) -> RobotValidationResult:
        try:
            import pinocchio as pin
        except ImportError as exc:
            message = (
                "robot validation에는 optional dependency Pinocchio가 필요합니다. "
                "`pip install -e '.[robot]'`을 실행하세요."
            )
            if bool(self.config.get("required", False)):
                raise RobotValidationError(message) from exc
            return RobotValidationResult.not_evaluated(message)

        urdf_path = str(Path(str(self.config["urdf_path"])).expanduser().resolve())
        try:
            model = pin.buildModelFromUrdf(urdf_path)
        except Exception as exc:
            raise RobotValidationError(f"URDF 모델 생성 실패: {exc}") from exc
        data = model.createData()
        tcp_name = str(self.config["tcp_link"])
        if not model.existFrame(tcp_name):
            raise RobotValidationError(f"URDF에 TCP frame '{tcp_name}'이 없습니다.")
        frame_id = model.getFrameId(tcp_name)
        q = np.asarray(self.config["q_teach"], dtype=float)
        if q.shape != (model.nq,) or not np.isfinite(q).all():
            raise RobotValidationError(f"q_teach 길이는 model.nq={model.nq}여야 합니다: {q.shape}")

        max_iterations = int(self.config.get("ik_max_iterations", 150))
        tolerance = float(self.config.get("ik_tolerance", 1e-5))
        damping = float(self.config.get("ik_damping", 1e-8))
        integration_step = float(self.config.get("ik_step_size", 0.35))
        q_sequence: list[np.ndarray] = []
        condition_numbers: list[float] = []
        minimum_singular_values: list[float] = []

        for sample_index, target_matrix in enumerate(target_transforms):
            target = pin.SE3(target_matrix[:3, :3], target_matrix[:3, 3])
            converged = False
            for _ in range(max_iterations):
                pin.forwardKinematics(model, data, q)
                pin.updateFramePlacements(model, data)
                current = data.oMf[frame_id]
                error = pin.log6(current.actInv(target)).vector
                if float(np.linalg.norm(error)) < tolerance:
                    converged = True
                    break
                jacobian = pin.computeFrameJacobian(
                    model, data, q, frame_id, pin.ReferenceFrame.LOCAL
                )
                lhs = jacobian @ jacobian.T + damping * np.eye(6)
                velocity = jacobian.T @ np.linalg.solve(lhs, error)
                q = pin.integrate(model, q, integration_step * velocity)
            if not converged:
                return RobotValidationResult(
                    status="invalid",
                    reason=f"IK가 sample {sample_index}에서 수렴하지 않았습니다.",
                    ik_success=False,
                    q=np.asarray(q_sequence) if q_sequence else None,
                )
            jacobian = pin.computeFrameJacobian(
                model, data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            singular_values = np.linalg.svd(jacobian, compute_uv=False)
            minimum = float(singular_values[-1]) if singular_values.size else 0.0
            condition = (
                float(singular_values[0] / minimum)
                if minimum > np.finfo(float).eps
                else float("inf")
            )
            minimum_singular_values.append(minimum)
            condition_numbers.append(condition)
            q_sequence.append(q.copy())

        q_array = np.asarray(q_sequence)
        qd = np.gradient(q_array, timestamps, axis=0, edge_order=1)
        qdd = np.gradient(qd, timestamps, axis=0, edge_order=1)
        if not (np.isfinite(q_array).all() and np.isfinite(qd).all() and np.isfinite(qdd).all()):
            return RobotValidationResult(
                status="invalid",
                reason="joint trajectory에 NaN 또는 inf가 발생했습니다.",
                ik_success=True,
                q=q_array,
                qd=qd,
                qdd=qdd,
            )

        lower = np.asarray(model.lowerPositionLimit, dtype=float)
        upper = np.asarray(model.upperPositionLimit, dtype=float)
        finite_limits = np.isfinite(lower) & np.isfinite(upper)
        position_evaluated = bool(finite_limits.any())
        margin = float("inf")
        position_ok = True
        if position_evaluated:
            margins = np.minimum(q_array - lower, upper - q_array)[:, finite_limits]
            margin = float(np.min(margins))
            position_ok = margin >= 0.0

        velocity_limits_raw = self.config.get("joint_velocity_limits")
        if velocity_limits_raw is None:
            velocity_limits = np.asarray(model.velocityLimit, dtype=float)
            velocity_evaluated = bool(
                velocity_limits.shape == (model.nv,)
                and np.isfinite(velocity_limits).all()
                and np.all(velocity_limits > 0)
            )
        else:
            velocity_limits = np.asarray(velocity_limits_raw, dtype=float)
            velocity_evaluated = velocity_limits.shape == (model.nv,)
        max_velocity = float(np.max(np.abs(qd)))
        velocity_ok = not velocity_evaluated or bool(np.all(np.abs(qd) <= velocity_limits + 1e-9))

        acceleration_limits_raw = self.config.get("joint_acceleration_limits")
        acceleration_evaluated = acceleration_limits_raw is not None
        max_acceleration = float(np.max(np.abs(qdd)))
        acceleration_ok = True
        if acceleration_evaluated:
            acceleration_limits = np.asarray(acceleration_limits_raw, dtype=float)
            if acceleration_limits.shape != (model.nv,) or np.any(acceleration_limits <= 0):
                raise RobotValidationError(
                    f"joint_acceleration_limits는 양수인 길이 {model.nv} 벡터여야 합니다."
                )
            acceleration_ok = bool(np.all(np.abs(qdd) <= acceleration_limits + 1e-9))

        branch_limit = self.config.get("branch_jump_threshold_rad")
        branch_ok = True
        if branch_limit is not None and len(q_array) > 1:
            branch_ok = bool(np.max(np.abs(np.diff(q_array, axis=0))) <= float(branch_limit))

        collision_status = self._check_self_collision(pin, model, q_array, urdf_path)
        collision_ok = collision_status not in {"collision_detected", "check_failed"}
        all_ok = position_ok and velocity_ok and acceleration_ok and branch_ok and collision_ok
        reasons: list[str] = []
        if not position_ok:
            reasons.append("joint position limit 초과")
        if not velocity_ok:
            reasons.append("joint velocity limit 초과")
        if not acceleration_ok:
            reasons.append("joint acceleration limit 초과")
        if not branch_ok:
            reasons.append("joint branch jump 초과")
        if not collision_ok:
            reasons.append(f"collision status={collision_status}")
        reason = (
            "사용자 URDF와 제공된 제한에 대한 기구학 검사를 통과했습니다."
            if all_ok
            else ", ".join(reasons)
        )
        return RobotValidationResult(
            status="passed_kinematic_checks" if all_ok else "invalid",
            reason=reason,
            ik_success=True,
            minimum_joint_limit_margin_rad=margin if position_evaluated else None,
            maximum_joint_velocity_rad_s=max_velocity,
            maximum_joint_acceleration_rad_s2=max_acceleration,
            maximum_jacobian_condition_number=float(np.max(condition_numbers)),
            minimum_singular_value=float(np.min(minimum_singular_values)),
            collision_status=collision_status,
            joint_position_limits_evaluated=position_evaluated,
            joint_velocity_limits_evaluated=velocity_evaluated,
            joint_acceleration_limits_evaluated=acceleration_evaluated,
            q=q_array,
            qd=qd,
            qdd=qdd,
        )

    def _check_self_collision(
        self,
        pin: Any,
        model: Any,
        q_sequence: np.ndarray,
        urdf_path: str,
    ) -> str:
        if not bool(self.config.get("collision_check", False)):
            return "not_evaluated"
        try:
            geometry_model = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.COLLISION)
            geometry_model.addAllCollisionPairs()
            geometry_data = pin.GeometryData(geometry_model)
            for q in q_sequence:
                pin.updateGeometryPlacements(
                    model,
                    model.createData(),
                    geometry_model,
                    geometry_data,
                    q,
                )
                if pin.computeCollisions(geometry_model, geometry_data, False):
                    return "collision_detected"
        except Exception:
            return "check_failed"
        return "no_self_collision_detected"
