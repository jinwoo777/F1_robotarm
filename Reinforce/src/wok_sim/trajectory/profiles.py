"""설정으로 legacy/볶음밥 trajectory profile을 선택하는 얇은 adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .fried_rice import FRIED_RICE_ACTION_NAMES, generate_fried_rice_trajectory
from .parameters import ACTION_NAMES
from .spline import Trajectory, generate_trajectory

LEGACY_PROFILE = "legacy_launch_catch"
FRIED_RICE_PROFILE = "fried_rice_teaching"

_LEGACY_ALIASES = {"legacy", "default", "launch_catch", LEGACY_PROFILE}
_FRIED_RICE_ALIASES = {"fried_rice", "fried-rice", "teaching", FRIED_RICE_PROFILE}


def trajectory_profile(config: Mapping[str, Any] | Any) -> str:
    """설정의 profile 이름을 canonical 값으로 정규화한다."""

    section = config.get("trajectory", {}) if isinstance(config, Mapping) else config
    if isinstance(section, Mapping):
        raw = section.get("profile", LEGACY_PROFILE)
    else:
        raw = getattr(section, "profile", LEGACY_PROFILE)
    profile = str(raw).strip().lower()
    if profile in _LEGACY_ALIASES:
        return LEGACY_PROFILE
    if profile in _FRIED_RICE_ALIASES:
        return FRIED_RICE_PROFILE
    supported = ", ".join((LEGACY_PROFILE, FRIED_RICE_PROFILE))
    raise ValueError(f"지원하지 않는 trajectory.profile={raw!r}; 지원값: {supported}")


def action_names_for_config(config: Mapping[str, Any] | Any) -> tuple[str, ...]:
    """선택된 profile의 정규화 action 순서를 반환한다."""

    return (
        FRIED_RICE_ACTION_NAMES
        if trajectory_profile(config) == FRIED_RICE_PROFILE
        else ACTION_NAMES
    )


def action_size_for_config(config: Mapping[str, Any] | Any) -> int:
    """선택된 profile의 action 차원을 반환한다."""

    return len(action_names_for_config(config))


def generate_configured_trajectory(
    action: Sequence[float] | np.ndarray,
    config: Mapping[str, Any] | Any,
    **kwargs: Any,
) -> Trajectory:
    """설정 profile에 맞는 open-loop trajectory를 생성한다."""

    if trajectory_profile(config) == FRIED_RICE_PROFILE:
        return generate_fried_rice_trajectory(action, config, **kwargs)
    return generate_trajectory(action, config, **kwargs)


__all__ = [
    "FRIED_RICE_PROFILE",
    "LEGACY_PROFILE",
    "action_names_for_config",
    "action_size_for_config",
    "generate_configured_trajectory",
    "trajectory_profile",
]
