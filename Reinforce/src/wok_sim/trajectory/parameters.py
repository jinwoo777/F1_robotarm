"""정규화된 정책 action과 SI 단위 궤적 파라미터의 변환."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

ACTION_NAMES: tuple[str, ...] = (
    "insertion_distance",
    "lift_height",
    "backward_distance",
    "pitch_amplitude",
    "cycle_time",
    "insert_phase_ratio",
    "catch_phase_ratio",
)
"""정책 action의 고정 순서."""


_RANGE_KEYS: tuple[str, ...] = (
    "insertion_distance_range_m",
    "lift_height_range_m",
    "backward_distance_range_m",
    "pitch_amplitude_range_rad",
    "cycle_time_range_s",
    "insert_phase_ratio_range",
    "catch_phase_ratio_range",
)


class ActionMappingError(ValueError):
    """action 또는 action 범위가 유효하지 않을 때 발생하는 예외."""


def _lookup(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def trajectory_section(config: Any) -> Any:
    """전체 설정 또는 trajectory 설정 자체에서 trajectory section을 얻는다."""

    section = _lookup(config, "trajectory", None)
    return config if section is None else section


def _range_from_config(config: Any, range_key: str, action_name: str) -> tuple[float, float]:
    section = trajectory_section(config)
    raw = _lookup(section, range_key, None)
    if raw is None:
        nested = _lookup(section, "action_ranges", None)
        raw = _lookup(nested, action_name, None) if nested is not None else None
    if raw is None:
        raise ActionMappingError(f"trajectory.{range_key} 설정이 필요합니다.")
    try:
        low, high = float(raw[0]), float(raw[1])
    except (IndexError, TypeError, ValueError) as exc:
        raise ActionMappingError(
            f"trajectory.{range_key}는 [min, max] 형식이어야 합니다: {raw!r}"
        ) from exc
    if not np.isfinite([low, high]).all() or low > high:
        raise ActionMappingError(f"trajectory.{range_key} 범위가 유효하지 않습니다: {raw!r}")
    return low, high


@dataclass(frozen=True, slots=True)
class TrajectoryParameters:
    """한 에피소드에서 5회 반복할 웍질의 물리 파라미터.

    길이, 시간, 각도는 각각 m, s, rad 단위다. 두 phase ratio는 사이클
    앞쪽의 삽입 구간과 뒤쪽의 catch/복귀 구간이 차지하는 비율이다.
    """

    insertion_distance: float
    lift_height: float
    backward_distance: float
    pitch_amplitude: float
    cycle_time: float
    insert_phase_ratio: float
    catch_phase_ratio: float

    def __post_init__(self) -> None:
        values = np.asarray(self.as_array(), dtype=float)
        if not np.isfinite(values).all():
            raise ActionMappingError("궤적 파라미터에 NaN 또는 inf가 포함되어 있습니다.")
        if (
            min(
                self.insertion_distance,
                self.lift_height,
                self.backward_distance,
                self.cycle_time,
            )
            <= 0.0
        ):
            raise ActionMappingError("거리와 cycle_time은 모두 양수여야 합니다.")
        if not 0.0 < self.insert_phase_ratio < 1.0:
            raise ActionMappingError("insert_phase_ratio는 0과 1 사이여야 합니다.")
        if not 0.0 < self.catch_phase_ratio < 1.0:
            raise ActionMappingError("catch_phase_ratio는 0과 1 사이여야 합니다.")
        if self.insert_phase_ratio + self.catch_phase_ratio >= 1.0:
            raise ActionMappingError(
                "insert_phase_ratio + catch_phase_ratio는 1보다 작아야 합니다."
            )

    @property
    def insertion_distance_m(self) -> float:
        """삽입 거리(m) 호환 alias."""

        return self.insertion_distance

    @property
    def lift_height_m(self) -> float:
        """launch 상승 높이(m) 호환 alias."""

        return self.lift_height

    @property
    def backward_distance_m(self) -> float:
        """launch 후퇴 거리(m) 호환 alias."""

        return self.backward_distance

    @property
    def pitch_amplitude_rad(self) -> float:
        """pitch 진폭(rad) 호환 alias."""

        return self.pitch_amplitude

    @property
    def cycle_time_s(self) -> float:
        """사이클 시간(s) 호환 alias."""

        return self.cycle_time

    def as_array(self) -> np.ndarray:
        """ACTION_NAMES 순서의 물리 파라미터 배열을 반환한다."""

        return np.asarray(
            (
                self.insertion_distance,
                self.lift_height,
                self.backward_distance,
                self.pitch_amplitude,
                self.cycle_time,
                self.insert_phase_ratio,
                self.catch_phase_ratio,
            ),
            dtype=float,
        )

    def as_dict(self) -> dict[str, float]:
        """로깅 가능한 물리 파라미터 mapping을 반환한다."""

        return dict(zip(ACTION_NAMES, self.as_array(), strict=True))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | Any) -> TrajectoryParameters:
        """dict 또는 같은 이름의 attribute를 가진 객체에서 생성한다."""

        aliases = {
            "insertion_distance": ("insertion_distance", "insertion_distance_m"),
            "lift_height": ("lift_height", "lift_height_m"),
            "backward_distance": ("backward_distance", "backward_distance_m"),
            "pitch_amplitude": ("pitch_amplitude", "pitch_amplitude_rad"),
            "cycle_time": ("cycle_time", "cycle_time_s"),
            "insert_phase_ratio": ("insert_phase_ratio",),
            "catch_phase_ratio": ("catch_phase_ratio",),
        }
        resolved: dict[str, float] = {}
        for name, candidates in aliases.items():
            raw = None
            for candidate in candidates:
                raw = _lookup(values, candidate, None)
                if raw is not None:
                    break
            if raw is None:
                raise ActionMappingError(f"물리 파라미터 '{name}' 값이 없습니다.")
            try:
                resolved[name] = float(raw)
            except (TypeError, ValueError) as exc:
                raise ActionMappingError(f"'{name}' 값이 숫자가 아닙니다: {raw!r}") from exc
        return cls(**resolved)


def map_action(
    action: Sequence[float] | np.ndarray,
    config: Any,
    *,
    clip: bool = True,
) -> TrajectoryParameters:
    """정규화 action 7개를 설정된 SI 물리 범위로 선형 변환한다.

    기본적으로 환경 경계에서 생길 수 있는 작은 수치 오차를 포함해
    action을 ``[-1, 1]``로 clip한다. ``clip=False``이면 범위 밖 action을
    명시적으로 거부한다.
    """

    normalized = np.asarray(action, dtype=float)
    if normalized.shape != (len(ACTION_NAMES),):
        raise ActionMappingError(
            f"action shape은 ({len(ACTION_NAMES)},)여야 합니다: {normalized.shape}"
        )
    if not np.isfinite(normalized).all():
        raise ActionMappingError("action에 NaN 또는 inf가 포함되어 있습니다.")
    if not clip and np.any(np.abs(normalized) > 1.0):
        raise ActionMappingError("정규화 action은 [-1, 1] 범위여야 합니다.")
    normalized = np.clip(normalized, -1.0, 1.0)

    physical: list[float] = []
    for value, range_key, action_name in zip(normalized, _RANGE_KEYS, ACTION_NAMES, strict=True):
        low, high = _range_from_config(config, range_key, action_name)
        physical.append(low + 0.5 * (value + 1.0) * (high - low))
    return TrajectoryParameters(*physical)


def normalized_action_to_parameters(
    action: Sequence[float] | np.ndarray,
    config: Any,
    *,
    clip: bool = True,
) -> TrajectoryParameters:
    """``map_action``의 설명적인 호환 함수."""

    return map_action(action, config, clip=clip)
