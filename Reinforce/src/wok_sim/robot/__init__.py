"""선택적 M0609 기구학 검증과 궤적 export."""

from typing import TYPE_CHECKING, Any

from .base import RobotValidationError, RobotValidationResult
from .exporter import TrajectoryExportData, export_trajectory
from .m0609_motion_contract import (
    DOOSAN_MOVESX_MAX_WAYPOINTS,
    M0609_REFERENCE_SPEC,
    SAFETY_STATUS,
    M0609CartesianCaps,
    M0609MotionContractError,
    M0609MotionReport,
    M0609MotionViolation,
    build_doosan_offline_plan,
    export_doosan_offline_plan,
    validate_m0609_cartesian_motion,
)

if TYPE_CHECKING:
    from .m0609_validator import M0609Validator


def __getattr__(name: str) -> Any:
    """M0609/Pinocchio validator는 명시적으로 요청할 때만 import한다."""

    if name == "M0609Validator":
        from .m0609_validator import M0609Validator

        return M0609Validator
    raise AttributeError(name)


__all__ = [
    "DOOSAN_MOVESX_MAX_WAYPOINTS",
    "M0609_REFERENCE_SPEC",
    "M0609Validator",
    "M0609CartesianCaps",
    "M0609MotionContractError",
    "M0609MotionReport",
    "M0609MotionViolation",
    "RobotValidationError",
    "RobotValidationResult",
    "SAFETY_STATUS",
    "TrajectoryExportData",
    "build_doosan_offline_plan",
    "export_doosan_offline_plan",
    "export_trajectory",
    "validate_m0609_cartesian_motion",
]
