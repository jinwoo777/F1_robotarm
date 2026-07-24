"""실행 중 feedback을 사용하지 않는 bounded episode random walk."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


class RandomWalkConfigurationError(ValueError):
    """random-walk 차원, 경계 또는 분산 설정이 유효하지 않을 때 발생한다."""


def _vector(
    value: float | Sequence[float] | np.ndarray,
    dimension: int,
    *,
    name: str,
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise RandomWalkConfigurationError(f"{name}은 유한한 숫자여야 합니다.") from exc
    if array.ndim == 0:
        array = np.full(dimension, float(array), dtype=float)
    if array.shape != (dimension,) or not np.isfinite(array).all():
        raise RandomWalkConfigurationError(
            f"{name}은 유한한 scalar 또는 shape ({dimension},) 배열이어야 합니다."
        )
    return array


def _reflect(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """큰 Gaussian step도 편향된 clipping 없이 닫힌 구간 안으로 반사한다."""

    widths = high - low
    phase = np.mod(values - low, 2.0 * widths)
    return low + np.where(phase <= widths, phase, 2.0 * widths - phase)


@dataclass(frozen=True, slots=True)
class RandomWalkProposal:
    """한 에피소드 전체에 고정해서 사용할 normalized action."""

    episode_index: int
    action: np.ndarray

    def __post_init__(self) -> None:
        if self.episode_index < 0:
            raise RandomWalkConfigurationError("episode_index는 0 이상이어야 합니다.")
        action = np.array(self.action, dtype=float, copy=True)
        if action.ndim != 1 or not np.isfinite(action).all():
            raise RandomWalkConfigurationError("proposal action은 유한한 1차원 배열이어야 합니다.")
        action.setflags(write=False)
        object.__setattr__(self, "action", action)


@dataclass(slots=True)
class BoundedEpisodeRandomWalk:
    """normalized action을 에피소드 사이에서만 한 번 갱신한다.

    ``current_proposal``로 받은 action은 해당 에피소드가 끝날 때까지 고정한다.
    다음 탐색점은 반드시 ``advance_episode``를 명시적으로 호출해야 생성된다.
    관측, reward 또는 simulation-step feedback을 받는 메서드를 의도적으로
    제공하지 않아 실행 중 jitter가 궤적에 섞이지 않게 한다.
    """

    dimension: int
    step_std: float | Sequence[float] | np.ndarray = 0.08
    low: float | Sequence[float] | np.ndarray = -1.0
    high: float | Sequence[float] | np.ndarray = 1.0
    initial_action: Sequence[float] | np.ndarray | None = None
    seed: int | None = None
    _step_std: np.ndarray = field(init=False, repr=False)
    _low: np.ndarray = field(init=False, repr=False)
    _high: np.ndarray = field(init=False, repr=False)
    _current: np.ndarray = field(init=False, repr=False)
    _episode_index: int = field(init=False, default=0, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.dimension, int) or isinstance(self.dimension, bool):
            raise RandomWalkConfigurationError("dimension은 양의 정수여야 합니다.")
        if self.dimension <= 0:
            raise RandomWalkConfigurationError("dimension은 양의 정수여야 합니다.")
        self._step_std = _vector(self.step_std, self.dimension, name="step_std")
        self._low = _vector(self.low, self.dimension, name="low")
        self._high = _vector(self.high, self.dimension, name="high")
        if np.any(self._step_std < 0.0):
            raise RandomWalkConfigurationError("step_std는 0 이상이어야 합니다.")
        if np.any(self._low >= self._high):
            raise RandomWalkConfigurationError("모든 random-walk 경계는 low < high여야 합니다.")
        self._rng = np.random.default_rng(self.seed)
        initial = (
            np.zeros(self.dimension, dtype=float)
            if self.initial_action is None
            else _vector(self.initial_action, self.dimension, name="initial_action")
        )
        if np.any(initial < self._low) or np.any(initial > self._high):
            raise RandomWalkConfigurationError("initial_action이 configured bounds 밖에 있습니다.")
        self._current = initial.copy()

    @property
    def episode_index(self) -> int:
        """현재 proposal을 사용할 에피소드 번호."""

        return self._episode_index

    @property
    def current_action(self) -> np.ndarray:
        """현재 에피소드의 action 복사본."""

        return self._current.copy()

    def current_proposal(self) -> RandomWalkProposal:
        """상태를 변경하지 않고 현재 에피소드 proposal을 반환한다."""

        return RandomWalkProposal(self._episode_index, self._current)

    def advance_episode(self) -> RandomWalkProposal:
        """Gaussian step을 한 번 적용해 다음 에피소드 proposal을 만든다."""

        perturbation = self._rng.normal(loc=0.0, scale=self._step_std)
        self._current = _reflect(self._current + perturbation, self._low, self._high)
        self._episode_index += 1
        return self.current_proposal()

    def reset(
        self,
        *,
        initial_action: Sequence[float] | np.ndarray | None = None,
        seed: int | None = None,
    ) -> RandomWalkProposal:
        """새 seed와 시작 action으로 episode 0 상태를 복원한다."""

        initial = (
            np.zeros(self.dimension, dtype=float)
            if initial_action is None
            else _vector(initial_action, self.dimension, name="initial_action")
        )
        if np.any(initial < self._low) or np.any(initial > self._high):
            raise RandomWalkConfigurationError("initial_action이 configured bounds 밖에 있습니다.")
        self._rng = np.random.default_rng(self.seed if seed is None else seed)
        self._current = initial.copy()
        self._episode_index = 0
        return self.current_proposal()
