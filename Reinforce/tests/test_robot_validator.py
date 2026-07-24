from __future__ import annotations

import numpy as np
import pytest

from wok_sim.robot import M0609Validator, RobotValidationError


def test_missing_urdf_is_not_evaluated_when_optional() -> None:
    validator = M0609Validator(
        {
            "enabled": True,
            "required": False,
            "urdf_path": None,
            "q_teach": [0.0] * 6,
            "tcp_link": "tool0",
        }
    )

    with pytest.warns(RuntimeWarning, match="URDF"):
        result = validator.validate(np.array([0.0, 0.1]), np.repeat(np.eye(4)[None], 2, 0))

    assert result.status == "not_evaluated"
    assert result.ik_success is None
    assert "평가하지 않았습니다" in result.reason


def test_missing_urdf_raises_when_required() -> None:
    validator = M0609Validator(
        {
            "enabled": True,
            "required": True,
            "urdf_path": None,
            "q_teach": [0.0] * 6,
            "tcp_link": "tool0",
        }
    )

    with pytest.raises(RobotValidationError, match="URDF"):
        validator.preflight()


def test_disabled_validator_never_fabricates_pass() -> None:
    result = M0609Validator({"enabled": False, "required": False}).preflight()

    assert result is not None
    assert result.status == "not_evaluated"
    assert result.status != "passed_kinematic_checks"
