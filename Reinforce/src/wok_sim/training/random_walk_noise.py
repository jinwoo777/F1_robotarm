"""SAC에 에피소드 간 상관된 bounded random-walk 탐색을 더한다."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from stable_baselines3.common.noise import ActionNoise

from wok_sim.exploration import BoundedEpisodeRandomWalk


class PersistentEpisodeRandomWalkNoise(ActionNoise):
    """one-step episode 사이에서만 갱신되는 SAC additive action noise.

    Stable-Baselines3는 episode 종료 때 ``reset()``을 호출한다. 이 구현은
    한 step이 곧 한 episode인 환경에서 walk 상관성을 보존하기 위해 그
    reset을 의도적으로 no-op으로 둔다. 물리 simulation 중에는 호출되지
    않으므로 실행 중 궤적에 jitter를 추가하지 않는다.
    """

    def __init__(
        self,
        dimension: int,
        *,
        step_std: float | Sequence[float] | np.ndarray = 0.06,
        bound: float | Sequence[float] | np.ndarray = 0.30,
        seed: int | None = None,
    ) -> None:
        bound_array = np.asarray(bound, dtype=float)
        if bound_array.ndim == 0:
            if not np.isfinite(bound_array) or float(bound_array) <= 0.0:
                raise ValueError("random-walk noise bound는 양수여야 합니다.")
            low: float | np.ndarray = -float(bound_array)
            high: float | np.ndarray = float(bound_array)
        else:
            if (
                bound_array.shape != (dimension,)
                or not np.isfinite(bound_array).all()
                or np.any(bound_array <= 0.0)
            ):
                raise ValueError(
                    f"random-walk noise bound는 양수 scalar 또는 shape ({dimension},)이어야 합니다."
                )
            low = -bound_array
            high = bound_array
        self._walk = BoundedEpisodeRandomWalk(
            dimension=dimension,
            step_std=step_std,
            low=low,
            high=high,
            seed=seed,
        )
        super().__init__()

    def __call__(self) -> np.ndarray:
        """다음 episode에 적용할 normalized additive noise를 반환한다."""

        return self._walk.advance_episode().action.astype(np.float32, copy=True)

    def reset(self) -> None:
        """SB3의 매-episode reset에도 walk 상태를 유지한다."""

    @property
    def current_noise(self) -> np.ndarray:
        """검사/로깅용 현재 noise 복사본."""

        return self._walk.current_action


__all__ = ["PersistentEpisodeRandomWalkNoise"]
