#!/usr/bin/env python3
"""세 볶음밥 입자의 MuJoCo 첫 반발 높이/solref damping을 재보정한다."""

from __future__ import annotations

import json
from dataclasses import asdict

from wok_sim.simulation.particle_generator import DEFAULT_FRIED_RICE_SPECIES
from wok_sim.simulation.rebound import (
    calibrate_mujoco_contact_damping_ratio,
    make_rebound_target,
    simulate_drop_rebound_height,
)


def main() -> None:
    records: list[dict[str, float | str]] = []
    for specification in DEFAULT_FRIED_RICE_SPECIES:
        damping_ratio = calibrate_mujoco_contact_damping_ratio(
            geom_type=specification.geom_type,
            sizes_m=specification.semi_axes_m,
            mass_kg=specification.nominal_mass_kg,
            drop_height_m=0.20,
            target_rebound_height_m=specification.target_rebound_height_m,
            contact_time_constant_s=0.006,
            timestep_s=0.002,
        )
        measured_height = simulate_drop_rebound_height(
            geom_type=specification.geom_type,
            sizes_m=specification.semi_axes_m,
            mass_kg=specification.nominal_mass_kg,
            drop_height_m=0.20,
            contact_damping_ratio=damping_ratio,
            contact_time_constant_s=0.006,
            timestep_s=0.002,
        )
        target = make_rebound_target(
            drop_height_m=0.20,
            rebound_height_m=specification.target_rebound_height_m,
            contact_damping_ratio=damping_ratio,
            contact_time_constant_s=0.006,
            calibration_timestep_s=0.002,
        )
        records.append(
            {
                "species": specification.name,
                **asdict(target),
                "measured_rebound_height_m": measured_height,
            }
        )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
