"""낙하/반발 높이로 MuJoCo 입자 접촉 파라미터를 보정한다.

MuJoCo는 restitution 계수를 직접 받지 않고 ``solref``의 time constant와
damping ratio로 soft contact를 정의한다. 따라서 실험에서 쉽게 측정할 수
있는 낙하 높이/첫 반발 높이를 보존하면서 solver 파라미터도 명시적으로
기록한다.

이 모듈의 drop simulation은 팬 geometry와 분리된 수평 plane을 사용한다.
팬 바닥 proxy의 contact 설정과 똑같이 particle geom의 ``priority=1``을
사용하므로 입자별 ``solref``가 선택된다. 실제 식재료/팬의 마찰과 변형까지
식별하는 도구는 아니며, 첫 수직 반발 높이를 맞추는 경량 calibration이다.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReboundTarget:
    """한 입자 종류의 기준 낙하/반발 및 MuJoCo contact target."""

    drop_height_m: float
    rebound_height_m: float
    restitution_coefficient: float
    theoretical_damping_ratio: float
    contact_time_constant_s: float
    contact_damping_ratio: float
    calibration_timestep_s: float


def restitution_from_heights(drop_height_m: float, rebound_height_m: float) -> float:
    """에너지 비에서 이상적 normal restitution ``sqrt(h_rebound/h_drop)``를 구한다."""

    drop = float(drop_height_m)
    rebound = float(rebound_height_m)
    if not np.isfinite(drop) or drop <= 0.0:
        raise ValueError("drop_height_m must be a positive finite value")
    if not np.isfinite(rebound) or rebound < 0.0 or rebound >= drop:
        raise ValueError("rebound_height_m must be finite and in [0, drop_height_m)")
    return math.sqrt(rebound / drop)


def damping_ratio_from_restitution(restitution_coefficient: float) -> float:
    """선형 underdamped 접촉의 restitution에 대응하는 이론 damping ratio.

    MuJoCo의 impedance 곡선과 discrete integration 때문에 이 값 자체가
    정확한 MuJoCo calibration 값은 아니다. 그 값은
    :func:`calibrate_mujoco_contact_damping_ratio`로 구한다.
    """

    restitution = float(restitution_coefficient)
    if not np.isfinite(restitution) or not 0.0 < restitution < 1.0:
        raise ValueError("restitution_coefficient must be finite and in (0, 1)")
    logarithm = math.log(restitution)
    return -logarithm / math.sqrt(math.pi**2 + logarithm**2)


def make_rebound_target(
    *,
    drop_height_m: float,
    rebound_height_m: float,
    contact_damping_ratio: float,
    contact_time_constant_s: float = 0.006,
    calibration_timestep_s: float = 0.002,
) -> ReboundTarget:
    """측정 높이와 calibration된 MuJoCo damping ratio를 한 record로 묶는다."""

    restitution = restitution_from_heights(drop_height_m, rebound_height_m)
    contact_damping = float(contact_damping_ratio)
    contact_time = float(contact_time_constant_s)
    timestep = float(calibration_timestep_s)
    if not np.isfinite(contact_damping) or contact_damping <= 0.0:
        raise ValueError("contact_damping_ratio must be a positive finite value")
    if not np.isfinite(contact_time) or contact_time <= 0.0:
        raise ValueError("contact_time_constant_s must be a positive finite value")
    if not np.isfinite(timestep) or timestep <= 0.0:
        raise ValueError("calibration_timestep_s must be a positive finite value")
    return ReboundTarget(
        drop_height_m=float(drop_height_m),
        rebound_height_m=float(rebound_height_m),
        restitution_coefficient=restitution,
        theoretical_damping_ratio=damping_ratio_from_restitution(restitution),
        contact_time_constant_s=contact_time,
        contact_damping_ratio=contact_damping,
        calibration_timestep_s=timestep,
    )


def _geom_vertical_radius(geom_type: str, sizes_m: np.ndarray) -> float:
    if geom_type == "sphere":
        if not np.allclose(sizes_m, sizes_m[0], rtol=1.0e-12, atol=1.0e-15):
            raise ValueError("sphere sizes_m must have equal axes")
        return float(sizes_m[0])
    if geom_type == "ellipsoid":
        return float(sizes_m[2])
    raise ValueError("geom_type must be 'sphere' or 'ellipsoid'")


def simulate_drop_rebound_height(
    *,
    geom_type: str,
    sizes_m: tuple[float, float, float] | np.ndarray,
    mass_kg: float,
    drop_height_m: float,
    contact_damping_ratio: float,
    contact_time_constant_s: float = 0.006,
    timestep_s: float = 0.002,
    maximum_time_s: float = 1.5,
) -> float:
    """MuJoCo 수평 plane에서 첫 반발의 바닥 기준 높이를 측정한다.

    ``drop_height_m``과 반환값은 geom의 가장 낮은 점과 plane 사이 거리다.
    ellipsoid는 identity orientation으로 두므로 세 번째 semi-axis가 수직이다.
    """

    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - optional diagnostic dependency
        raise RuntimeError("MuJoCo drop calibration requires the `mujoco` package") from exc

    geom = str(geom_type).lower()
    sizes = np.asarray(sizes_m, dtype=np.float64)
    if sizes.shape != (3,) or not np.all(np.isfinite(sizes)) or np.any(sizes <= 0.0):
        raise ValueError("sizes_m must contain three positive finite semi-axes")
    vertical_radius = _geom_vertical_radius(geom, sizes)
    mass = float(mass_kg)
    drop = float(drop_height_m)
    damping = float(contact_damping_ratio)
    time_constant = float(contact_time_constant_s)
    timestep = float(timestep_s)
    maximum_time = float(maximum_time_s)
    positive_values = {
        "mass_kg": mass,
        "drop_height_m": drop,
        "contact_damping_ratio": damping,
        "contact_time_constant_s": time_constant,
        "timestep_s": timestep,
        "maximum_time_s": maximum_time,
    }
    for name, value in positive_values.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite value")

    size_text = (
        f"{float(sizes[0]):.12g}"
        if geom == "sphere"
        else " ".join(f"{value:.12g}" for value in sizes)
    )
    initial_center_z = vertical_radius + drop
    xml = f"""
<mujoco model="particle_drop_calibration">
  <option timestep="{timestep:.12g}" gravity="0 0 -9.81"
          integrator="implicitfast" iterations="80"/>
  <worldbody>
    <geom name="calibration_plane" type="plane" size="0.5 0.5 0.1"
          friction="0 0 0" solref="0.006 1" solimp="0.92 0.98 0.001"/>
    <body name="particle" pos="0 0 {initial_center_z:.12g}">
      <freejoint/>
      <geom name="particle_geom" type="{geom}" size="{size_text}"
            mass="{mass:.12g}" friction="0 0 0" priority="1"
            solref="{time_constant:.12g} {damping:.12g}"
            solimp="0.9 0.95 0.001" condim="3"/>
    </body>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "particle"))

    contacted = False
    separated_after_contact = False
    maximum_rebound = 0.0
    maximum_steps = int(math.ceil(maximum_time / timestep))
    for _ in range(maximum_steps):
        mujoco.mj_step(model, data)
        touching = bool(data.ncon)
        if touching:
            contacted = True
        elif contacted:
            separated_after_contact = True
            clearance = max(0.0, float(data.xpos[body_id, 2]) - vertical_radius)
            maximum_rebound = max(maximum_rebound, clearance)
            vertical_velocity = float(data.qvel[2])
            if vertical_velocity <= 0.0:
                break
    if not contacted:
        raise RuntimeError("drop calibration ended before the first contact")
    if not separated_after_contact:
        return 0.0
    return maximum_rebound


def calibrate_mujoco_contact_damping_ratio(
    *,
    geom_type: str,
    sizes_m: tuple[float, float, float] | np.ndarray,
    mass_kg: float,
    drop_height_m: float,
    target_rebound_height_m: float,
    contact_time_constant_s: float = 0.006,
    timestep_s: float = 0.002,
    iterations: int = 28,
) -> float:
    """binary search로 목표 첫 반발 높이에 맞는 MuJoCo damping ratio를 구한다."""

    target = float(target_rebound_height_m)
    restitution_from_heights(drop_height_m, target)
    if isinstance(iterations, bool) or int(iterations) != iterations or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    lower = 0.01
    upper = 1.25

    def simulate(damping_ratio: float) -> float:
        return simulate_drop_rebound_height(
            geom_type=geom_type,
            sizes_m=sizes_m,
            mass_kg=mass_kg,
            drop_height_m=drop_height_m,
            contact_damping_ratio=damping_ratio,
            contact_time_constant_s=contact_time_constant_s,
            timestep_s=timestep_s,
        )

    lower_height = simulate(lower)
    upper_height = simulate(upper)
    if not upper_height <= target <= lower_height:
        raise RuntimeError(
            "target rebound is outside the calibration bracket: "
            f"target={target:.6g}, range=[{upper_height:.6g}, {lower_height:.6g}]"
        )
    for _ in range(int(iterations)):
        midpoint = 0.5 * (lower + upper)
        if simulate(midpoint) > target:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def _default_calibration() -> list[dict[str, Any]]:
    specifications = (
        ("large_sphere", "sphere", (0.005, 0.005, 0.005), 0.002, 0.030),
        ("small_sphere", "sphere", (0.003, 0.003, 0.003), 0.0008, 0.010),
        ("ellipsoid", "ellipsoid", (0.005, 0.002, 0.002), 0.0002, 0.015),
    )
    records: list[dict[str, Any]] = []
    for name, geom_type, sizes, mass, rebound in specifications:
        damping = calibrate_mujoco_contact_damping_ratio(
            geom_type=geom_type,
            sizes_m=sizes,
            mass_kg=mass,
            drop_height_m=0.2,
            target_rebound_height_m=rebound,
        )
        measured = simulate_drop_rebound_height(
            geom_type=geom_type,
            sizes_m=sizes,
            mass_kg=mass,
            drop_height_m=0.2,
            contact_damping_ratio=damping,
        )
        target = make_rebound_target(
            drop_height_m=0.2,
            rebound_height_m=rebound,
            contact_damping_ratio=damping,
        )
        records.append(
            {
                "species": name,
                **asdict(target),
                "measured_rebound_height_m": measured,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate fried-rice particle rebound targets")
    parser.parse_args()
    print(json.dumps(_default_calibration(), indent=2))


if __name__ == "__main__":  # pragma: no cover - manual calibration tool
    main()


__all__ = [
    "ReboundTarget",
    "calibrate_mujoco_contact_damping_ratio",
    "damping_ratio_from_restitution",
    "make_rebound_target",
    "restitution_from_heights",
    "simulate_drop_rebound_height",
]
