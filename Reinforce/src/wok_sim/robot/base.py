"""로봇 validator의 simulator-independent 자료형."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


class RobotValidationError(RuntimeError):
    """필수 robot 자산이나 기구학 검사가 실패했을 때 발생한다."""


@dataclass
class RobotValidationResult:
    """로봇 기구학 검증 결과와 선택적인 joint trajectory."""

    status: str
    reason: str
    ik_success: bool | None = None
    minimum_joint_limit_margin_rad: float | None = None
    maximum_joint_velocity_rad_s: float | None = None
    maximum_joint_acceleration_rad_s2: float | None = None
    maximum_jacobian_condition_number: float | None = None
    minimum_singular_value: float | None = None
    collision_status: str = "not_evaluated"
    joint_position_limits_evaluated: bool = False
    joint_velocity_limits_evaluated: bool = False
    joint_acceleration_limits_evaluated: bool = False
    q: np.ndarray | None = field(default=None, repr=False)
    qd: np.ndarray | None = field(default=None, repr=False)
    qdd: np.ndarray | None = field(default=None, repr=False)

    @classmethod
    def not_evaluated(cls, reason: str) -> RobotValidationResult:
        """검증을 통과한 것으로 오해되지 않는 비활성 결과를 만든다."""

        return cls(status="not_evaluated", reason=reason, ik_success=None)

    def summary(self) -> dict[str, Any]:
        """큰 joint 배열을 제외한 logging용 mapping을 반환한다."""

        result = asdict(self)
        result.pop("q", None)
        result.pop("qd", None)
        result.pop("qdd", None)
        return result
