from __future__ import annotations

from pathlib import Path

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
    assert config["particles"]["fried_rice"]["count_per_type_range"] == [20, 40]
    assert config["particles"]["target_mass_range_kg"] == [0.056, 0.128]
    assert config["training"]["observation_mass_normalization"] == "symmetric_range"
    assert config["training"]["observation_count_per_type"] is True

    with pytest.raises(ConfigError, match="전체 support"):
        load_config(
            "configs/fried_rice.yaml",
            overrides={"particles": {"target_mass_range_kg": [0.056, 0.064]}},
        )


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
