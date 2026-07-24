"""제곱 norm 시간 적분 및 normalization 검증."""

from __future__ import annotations

import numpy as np
import pytest

from wok_sim.metrics.trajectory_cost import compute_trajectory_costs


def test_constant_acceleration_and_jerk_have_exact_trapezoid_cost() -> None:
    times = np.array([0.0, 0.5, 1.5, 2.0])
    acceleration = np.tile([2.0, 0.0, 0.0], (times.size, 1))
    jerk = np.tile([0.0, 3.0, 0.0], (times.size, 1))

    costs = compute_trajectory_costs(
        times,
        acceleration,
        jerk,
        acceleration_normalization=4.0,
        jerk_normalization=6.0,
    )
    assert costs.acceleration_integral == pytest.approx(8.0)
    assert costs.jerk_integral == pytest.approx(18.0)
    assert costs.acceleration_cost == pytest.approx(2.0)
    assert costs.jerk_cost == pytest.approx(3.0)
    assert costs.duration_s == pytest.approx(2.0)
