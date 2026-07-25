from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wok_sim.config import ConfigError, load_config
from wok_sim.geometry.pan_asset import PanAssetError, inspect_pan_asset


def test_default_config_resolves_real_stl() -> None:
    config = load_config("configs/default.yaml")
    report = inspect_pan_asset(config["pan"])

    assert not report.procedural_demo
    assert Path(report.path).name == "Wok.stl"
    assert report.faces == 7954
    assert report.extents_m == pytest.approx([0.2399463, 0.2399731, 0.06])
    assert report.watertight is False


def test_test_config_requires_explicit_procedural_mode() -> None:
    config = load_config("configs/test.yaml")
    report = inspect_pan_asset(config["pan"])

    assert report.procedural_demo
    assert report.path is None


def test_invalid_stl_path_is_not_silently_replaced() -> None:
    with pytest.raises(PanAssetError, match="찾을 수 없습니다"):
        inspect_pan_asset(
            {
                "stl_path": "/definitely/missing/pan.stl",
                "stl_scale": 0.001,
                "use_procedural_demo": False,
            }
        )


def test_invalid_config_has_clear_message(tmp_path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("pan: {}\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="필수 설정 섹션"):
        load_config(path)


def test_base_config_relative_asset_is_resolved_from_base_file(tmp_path) -> None:
    child = tmp_path / "override.yaml"
    child.write_text("training:\n  seed: 99\n", encoding="utf-8")

    config = load_config(child, base_path="configs/default.yaml")

    assert Path(config["pan"]["stl_path"]).samefile("Wok.stl")
    assert config["training"]["seed"] == 99


def test_yaml_declared_base_config_can_be_loaded_directly(tmp_path) -> None:
    child = tmp_path / "override.yaml"
    default_path = Path("configs/default.yaml").resolve()
    child.write_text(
        f"base_config: {default_path}\ntraining:\n  seed: 101\n",
        encoding="utf-8",
    )

    config = load_config(child)

    assert Path(config["pan"]["stl_path"]).samefile("Wok.stl")
    assert config["training"]["seed"] == 101


def test_fried_rice_config_covers_full_variable_count_mass_support() -> None:
    config = load_config("configs/fried_rice.yaml")

    assert config["particles"]["count"] == 60
    assert config["particles"]["spawn_radius_m"] == pytest.approx(0.050)
    assert config["particles"]["fried_rice"]["count_per_type_range"] == [20, 40]
    assert config["particles"]["target_mass_range_kg"] == [0.056, 0.128]
    assert config["training"]["observation_mass_normalization"] == "symmetric_range"
    assert config["training"]["observation_count_per_type"] is True
    assert config["training"]["total_timesteps"] == 450
    assert config["training"]["parallel_environments"] == 6
    assert config["training"]["count_per_type_schedule"] == [20, 30, 40]
    assert config["training"]["episodes_per_count"] == 150
    assert config["training"]["checkpoint_interval"] == 30
    assert config["training"]["random_walk"]["step_std"] == [0.05, 0.05, 0.03, 0.20]
    assert config["training"]["random_walk"]["bound"] == [0.25, 0.25, 0.20, 1.00]
    assert config["training"]["lift_approach_curriculum"] == pytest.approx(
        {
            "enabled": True,
            "initial_episodes": 60,
            "transition_episodes": 120,
            "initial_multiplier": 1.0,
            "transition_start_multiplier": 0.5,
            "final_multiplier": 0.0,
        }
    )
    assert config["reward"]["mixing_reward_mode"] == "improvement"
    assert config["reward"]["lift_top_margin_m"] == pytest.approx(0.001)
    assert config["reward"]["lift_reward_per_particle"] == pytest.approx(0.10)
    assert config["reward"]["lift_approach_band_m"] == pytest.approx(0.040)
    assert config["reward"]["lift_approach_reward_per_particle"] == pytest.approx(0.10)
    assert config["reward"]["spill_severity_mode"] == "max_count_mass"
    assert config["reward"]["w_spill"] == pytest.approx(12.0)
    assert config["reward"]["w_spill_quadratic"] == pytest.approx(40.0)
    trajectory = config["trajectory"]
    profile = trajectory["fried_rice"]
    proxy = config["pan"]["collision_proxy"]
    assert proxy["wall_profile"] == "circular_arc"
    assert proxy["radial_wall_segments"] == 8
    assert proxy["bottom_radius_m"] == pytest.approx(0.060)
    assert proxy["inner_radius_m"] == pytest.approx(0.115)
    assert proxy["rim_radius_m"] == pytest.approx(0.120)
    assert proxy["rim_z_m"] == pytest.approx(0.055)
    assert proxy["wall_thickness_m"] == pytest.approx(0.005)
    assert trajectory["allow_cycle_boundary_stop"] is False
    assert profile["motion_profile"] == "continuous_blended_global_quintic"
    assert profile["pretilt_translation_fraction"] == pytest.approx(0.20)
    assert profile["lift_return_translation_fraction"] == pytest.approx(0.65)
    assert profile["lift_return_arc_height_m"] == pytest.approx(0.025)
    assert profile["continuous_time_scale"] == pytest.approx(1.50)
    assert profile["lift_return_time_factor"] == pytest.approx(0.875)
    assert profile["adaptive_robot_retiming"] is True
    assert profile["adaptive_robot_cap_fraction"] == pytest.approx(0.95)
    assert profile["adaptive_max_time_scale"] == pytest.approx(3.0)
    assert profile["descent_angle_range_rad"] == pytest.approx(
        [0.6108652381980153, 0.9599310885968813]
    )
    assert profile["pan_tilt_angle_range_rad"] == pytest.approx(
        [0.3490658503988659, 0.6981317007977318]
    )
    assert profile["descent_speed_range_m_s"] == pytest.approx([0.38, 0.48])
    assert profile["lift_angle_range_rad"] == pytest.approx([0.0, 0.5235987755982988])
    assert profile["insertion_distance_m"] == pytest.approx(0.25)
    assert profile["angular_speed_rad_s"] == pytest.approx(1.20)
    assert profile["linear_acceleration_limit_m_s2"] == pytest.approx(1.40)
    assert profile["angular_acceleration_limit_rad_s2"] == pytest.approx(2.50)
    assert profile["linear_jerk_limit_m_s3"] == pytest.approx(15.0)
    assert profile["angular_jerk_limit_rad_s3"] == pytest.approx(20.0)
    assert {
        "max_cartesian_velocity": trajectory["max_cartesian_velocity"],
        "max_angular_velocity": trajectory["max_angular_velocity"],
        "max_cartesian_acceleration": trajectory["max_cartesian_acceleration"],
        "max_angular_acceleration": trajectory["max_angular_acceleration"],
        "max_cartesian_jerk": trajectory["max_cartesian_jerk"],
        "max_angular_jerk": trajectory["max_angular_jerk"],
    } == {
        "max_cartesian_velocity": 0.50,
        "max_angular_velocity": 1.30,
        "max_cartesian_acceleration": 1.50,
        "max_angular_acceleration": 2.70,
        "max_cartesian_jerk": 16.0,
        "max_angular_jerk": 22.0,
    }
    assert config["robot"]["cartesian_caps"] == pytest.approx(
        {
            "linear_velocity_m_s": 0.25,
            "angular_velocity_rad_s": 0.5235987755982988,
            "linear_acceleration_m_s2": 0.50,
            "angular_acceleration_rad_s2": 1.0471975511965976,
            "linear_jerk_m_s3": 2.0,
            "angular_jerk_rad_s3": 4.1887902047863905,
        }
    )

    with pytest.raises(ConfigError, match="전체 support"):
        load_config(
            "configs/fried_rice.yaml",
            overrides={"particles": {"target_mass_range_kg": [0.056, 0.064]}},
        )


def test_fast_150_config_inherits_nested_base_and_targets_90_percent_caps() -> None:
    config = load_config("configs/fried_rice_fast_150.yaml")
    profile = config["trajectory"]["fried_rice"]
    training = config["training"]

    assert config["particles"]["profile"] == "fried_rice"
    assert profile["lift_angle_range_rad"] == pytest.approx([0.0, np.deg2rad(30.0)])
    assert profile["continuous_time_scale"] == pytest.approx(1.50)
    assert profile["adaptive_robot_cap_fraction"] == pytest.approx(0.90)
    assert profile["adaptive_robot_jerk_cap_fraction"] == pytest.approx(1.0)
    assert profile["adaptive_allow_speedup"] is True
    assert profile["adaptive_min_time_scale"] == pytest.approx(0.25)
    assert training["total_timesteps"] == 150
    assert training["episodes_per_count"] == 50
    assert training["lift_approach_curriculum"]["initial_episodes"] == 18
    assert training["lift_approach_curriculum"]["transition_episodes"] == 42
    assert training["checkpoint_directory"].endswith(
        "fried_rice_sac_150_target090_bonus010"
    )


def test_nested_base_config_rejects_cycles(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("base_config: second.yaml\n", encoding="utf-8")
    second.write_text("base_config: first.yaml\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="순환 참조"):
        load_config(first)


@pytest.mark.parametrize(
    "invalid_range",
    ([20.5, 40], [0, 40], [40, 20]),
)
def test_fried_rice_config_rejects_invalid_count_per_type_range(
    invalid_range: list[float],
) -> None:
    with pytest.raises(ConfigError, match="count_per_type_range"):
        load_config(
            "configs/fried_rice.yaml",
            overrides={"particles": {"fried_rice": {"count_per_type_range": invalid_range}}},
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"lift_reward_per_particle": -0.1}, "lift_reward_per_particle"),
        (
            {"lift_approach_reward_per_particle": 0.1, "lift_approach_band_m": 0.0},
            "lift_approach_band_m",
        ),
        ({"lift_top_margin_m": -0.001}, "lift_top_margin_m"),
        ({"mixing_reward_mode": "final"}, "mixing_reward_mode"),
        ({"spill_severity_mode": "unknown"}, "spill_severity_mode"),
    ],
)
def test_reward_config_rejects_invalid_lift_and_spill_settings(
    override: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config("configs/fried_rice.yaml", overrides={"reward": override})
