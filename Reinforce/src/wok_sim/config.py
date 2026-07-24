"""YAML 설정 로딩과 교차 필드 검증."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class ConfigError(ValueError):
    """사용자가 수정할 수 있는 설정 오류."""


def _deep_merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = deepcopy(value)
    return result


def _pair(section: Mapping[str, Any], key: str, *, positive: bool = False) -> tuple[float, float]:
    try:
        values = section[key]
        low, high = float(values[0]), float(values[1])
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ConfigError(f"'{key}'는 [최솟값, 최댓값] 형식이어야 합니다.") from exc
    if not np.isfinite([low, high]).all() or low > high:
        raise ConfigError(f"'{key}' 범위가 유효하지 않습니다: {values!r}")
    if positive and low <= 0:
        raise ConfigError(f"'{key}' 범위는 양수여야 합니다: {values!r}")
    return low, high


def _integer_pair(
    section: Mapping[str, Any],
    key: str,
    *,
    positive: bool = False,
) -> tuple[int, int]:
    try:
        raw_low, raw_high = section[key]
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"'{key}'는 [최솟값, 최댓값] 형식이어야 합니다.") from exc
    try:
        low, high = int(raw_low), int(raw_high)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigError(f"'{key}' 범위는 정수여야 합니다: {section[key]!r}") from exc
    if (
        isinstance(raw_low, bool)
        or isinstance(raw_high, bool)
        or low != raw_low
        or high != raw_high
    ):
        raise ConfigError(f"'{key}' 범위는 정수여야 합니다: {section[key]!r}")
    if low > high or (positive and low <= 0):
        raise ConfigError(f"'{key}' 범위가 유효하지 않습니다: {section[key]!r}")
    return low, high


def validate_config(config: Mapping[str, Any]) -> None:
    """필수 설정과 위험한 수치 오류를 조기에 검사한다."""

    required = (
        "pan",
        "particles",
        "trajectory",
        "simulation",
        "mixing",
        "reward",
        "robot",
        "training",
        "logging",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ConfigError(f"필수 설정 섹션이 없습니다: {', '.join(missing)}")

    pan = config["pan"]
    particles = config["particles"]
    trajectory = config["trajectory"]
    simulation = config["simulation"]
    mixing = config["mixing"]
    robot = config["robot"]
    training = config["training"]

    if float(pan.get("stl_scale", 0.0)) <= 0:
        raise ConfigError("pan.stl_scale은 양수여야 합니다.")
    if not bool(pan.get("kinematic", False)):
        raise ConfigError(
            "현재 구현은 pan.kinematic=true인 mocap 팬만 지원합니다. "
            "dynamic 팬 관성은 아직 모델링하지 않습니다."
        )
    if str(pan.get("collision_mode", "")) != "compound_proxy":
        raise ConfigError("pan.collision_mode은 현재 'compound_proxy'만 지원합니다.")
    if not bool(pan.get("use_procedural_demo", False)) and not pan.get("stl_path"):
        raise ConfigError(
            "실제 실행에는 pan.stl_path가 필요합니다. 테스트용이면 "
            "pan.use_procedural_demo=true를 명시하세요."
        )
    particle_profile = (
        str(particles.get("profile", particles.get("mode", "scaled_spheres"))).strip().lower()
    )
    fried_rice_particles = particle_profile in {
        "fried_rice",
        "fried-rice",
        "fried_rice_mixture",
    }
    count_per_type_range: tuple[int, int] | None = None
    if fried_rice_particles:
        if float(particles.get("spawn_radius_m", 0.0)) <= 0:
            raise ConfigError("particles.spawn_radius_m는 양수여야 합니다.")
        mass_low, mass_high = _pair(
            particles,
            "target_mass_range_kg",
            positive=True,
        )
        profile = particles.get("fried_rice", {})
        if not isinstance(profile, Mapping):
            raise ConfigError("particles.fried_rice는 mapping이어야 합니다.")
        drop_height = float(profile.get("reference_drop_height_m", 0.20))
        if not np.isfinite(drop_height) or drop_height <= 0.0:
            raise ConfigError("particles.fried_rice.reference_drop_height_m는 양수여야 합니다.")
        common_count = int(profile.get("count_per_type", 20))
        if common_count <= 0:
            raise ConfigError("particles.fried_rice.count_per_type은 1 이상이어야 합니다.")
        count_per_type_range = (
            None
            if "count_per_type_range" not in profile
            else _integer_pair(profile, "count_per_type_range", positive=True)
        )
        defaults = {
            "large_sphere": (0.002, 0.10, 0.030),
            "small_sphere": (0.0008, 0.0, 0.010),
            "ellipsoid": (0.0002, 0.0, 0.015),
        }
        total_count = 0
        support_low = 0.0
        support_high = 0.0
        for name, (default_mass, default_jitter, default_rebound) in defaults.items():
            species = profile.get(name, {})
            if not isinstance(species, Mapping):
                raise ConfigError(f"particles.fried_rice.{name}은 mapping이어야 합니다.")
            count = int(species.get("count", common_count))
            if count_per_type_range is not None and count != common_count:
                raise ConfigError(
                    "particles.fried_rice.count_per_type_range를 사용할 때는 "
                    "종별 count override를 사용할 수 없습니다."
                )
            mass = float(
                species.get(
                    "mass_mean_kg",
                    species.get("mass_kg", default_mass),
                )
            )
            jitter = float(species.get("mass_jitter_fraction", default_jitter))
            rebound = float(
                species.get(
                    "target_rebound_height_m",
                    species.get("rebound_height_m", default_rebound),
                )
            )
            if count <= 0 or not np.isfinite(mass) or mass <= 0.0:
                raise ConfigError(f"particles.fried_rice.{name} count/mass가 유효하지 않습니다.")
            if not np.isfinite(jitter) or not 0.0 <= jitter < 1.0:
                raise ConfigError(
                    f"particles.fried_rice.{name}.mass_jitter_fraction은 [0,1)이어야 합니다."
                )
            if not np.isfinite(rebound) or not 0.0 < rebound < drop_height:
                raise ConfigError(
                    f"particles.fried_rice.{name} 반발 높이는 기준 낙하 높이보다 "
                    "작은 양수여야 합니다."
                )
            if name == "ellipsoid":
                axes = np.asarray(
                    species.get(
                        "semi_axes_m",
                        species.get("axes_m", [0.005, 0.002, 0.002]),
                    ),
                    dtype=float,
                )
                if axes.shape != (3,) or not np.isfinite(axes).all() or np.any(axes <= 0):
                    raise ConfigError(
                        "particles.fried_rice.ellipsoid.semi_axes_m는 양수 3개여야 합니다."
                    )
            else:
                radius = float(
                    species.get(
                        "radius_m",
                        0.005 if name == "large_sphere" else 0.003,
                    )
                )
                if not np.isfinite(radius) or radius <= 0.0:
                    raise ConfigError(f"particles.fried_rice.{name}.radius_m는 양수여야 합니다.")
            total_count += count
            minimum_count = count if count_per_type_range is None else count_per_type_range[0]
            maximum_count = count if count_per_type_range is None else count_per_type_range[1]
            support_low += minimum_count * mass * (1.0 - jitter)
            support_high += maximum_count * mass * (1.0 + jitter)
        configured_count = int(particles.get("count", total_count))
        if configured_count != total_count:
            raise ConfigError(
                f"particles.count={configured_count}이 종별 합계 {total_count}과 다릅니다."
            )
        tolerance = 1.0e-12
        if mass_low > support_low + tolerance or mass_high < support_high - tolerance:
            raise ConfigError(
                "particles.target_mass_range_kg가 fried_rice 질량 jitter의 "
                f"전체 support [{support_low:.9g}, {support_high:.9g}]를 포함해야 합니다."
            )
    else:
        if int(particles.get("count", 0)) <= 0:
            raise ConfigError("particles.count는 1 이상이어야 합니다.")
        for key in ("density_kg_m3", "nominal_radius_m", "spawn_radius_m"):
            if float(particles.get(key, 0.0)) <= 0:
                raise ConfigError(f"particles.{key}는 양수여야 합니다.")
        mass_low, mass_high = _pair(particles, "target_mass_range_kg", positive=True)

    mass_normalization = str(training.get("observation_mass_normalization", "none")).strip().lower()
    if mass_normalization not in {
        "none",
        "raw",
        "raw_kg",
        "symmetric_range",
        "symmetric",
        "minus_one_to_one",
    }:
        raise ConfigError(
            "training.observation_mass_normalization은 'none' 또는 'symmetric_range'여야 합니다."
        )
    if (
        mass_normalization in {"symmetric_range", "symmetric", "minus_one_to_one"}
        and mass_low == mass_high
    ):
        raise ConfigError(
            "symmetric_range 질량 관측에는 서로 다른 target mass 최솟값/최댓값이 필요합니다."
        )
    if bool(training.get("observation_count_per_type", False)):
        if not fried_rice_particles:
            raise ConfigError(
                "training.observation_count_per_type은 fried_rice 입자 profile에서만 "
                "사용할 수 있습니다."
            )
        if count_per_type_range is None or count_per_type_range[0] == count_per_type_range[1]:
            raise ConfigError(
                "count_per_type 관측에는 서로 다른 "
                "particles.fried_rice.count_per_type_range=[min,max]가 필요합니다."
            )

    if int(trajectory.get("cycles", 0)) <= 0:
        raise ConfigError("trajectory.cycles는 1 이상이어야 합니다.")
    trajectory_profile = str(trajectory.get("profile", "legacy_launch_catch")).strip().lower()
    fried_rice_trajectory = trajectory_profile in {
        "fried_rice",
        "fried-rice",
        "teaching",
        "fried_rice_teaching",
    }
    if fried_rice_trajectory:
        profile = trajectory.get("fried_rice", {})
        if not isinstance(profile, Mapping):
            raise ConfigError("trajectory.fried_rice는 mapping이어야 합니다.")
        for key in (
            "insertion_distance_range_m",
            "tilt_angle_range_rad",
            "linear_speed_range_m_s",
            "angular_speed_range_rad_s",
        ):
            _pair(profile, key, positive=True)
        derivative_gates = (
            ("linear_acceleration_limit_m_s2", "max_cartesian_acceleration"),
            ("angular_acceleration_limit_rad_s2", "max_angular_acceleration"),
            ("linear_jerk_limit_m_s3", "max_cartesian_jerk"),
            ("angular_jerk_limit_rad_s3", "max_angular_jerk"),
        )
        for profile_key, trajectory_key in derivative_gates:
            value = float(profile.get(profile_key, 0.0))
            outer_limit = float(trajectory.get(trajectory_key, 0.0))
            if not np.isfinite(value) or value <= 0.0:
                raise ConfigError(f"trajectory.fried_rice.{profile_key}는 양수여야 합니다.")
            if value > outer_limit:
                raise ConfigError(
                    f"trajectory.fried_rice.{profile_key}={value:g}가 "
                    f"trajectory.{trajectory_key}={outer_limit:g}보다 클 수 없습니다."
                )
        if float(profile.get("minimum_phase_duration_s", 0.0)) <= 0.0:
            raise ConfigError("trajectory.fried_rice.minimum_phase_duration_s는 양수여야 합니다.")
        if _pair(profile, "linear_speed_range_m_s", positive=True)[1] > float(
            trajectory.get("max_cartesian_velocity", 0.0)
        ):
            raise ConfigError(
                "fried_rice linear speed 최댓값이 trajectory Cartesian velocity gate를 초과합니다."
            )
        if _pair(profile, "angular_speed_range_rad_s", positive=True)[1] > float(
            trajectory.get("max_angular_velocity", 0.0)
        ):
            raise ConfigError(
                "fried_rice angular speed 최댓값이 trajectory angular velocity gate를 초과합니다."
            )
        if not np.isclose(
            float(profile.get("insertion_angle_deg", 45.0)),
            45.0,
            atol=1.0e-12,
        ):
            raise ConfigError("fried_rice insertion_angle_deg는 현재 45도여야 합니다.")
    else:
        for key in (
            "insertion_distance_range_m",
            "lift_height_range_m",
            "backward_distance_range_m",
            "cycle_time_range_s",
            "insert_phase_ratio_range",
            "catch_phase_ratio_range",
        ):
            _pair(trajectory, key, positive=True)
        _pair(trajectory, "pitch_amplitude_range_rad")
    if float(trajectory.get("sample_rate_hz", 0.0)) <= 0:
        raise ConfigError("trajectory.sample_rate_hz는 양수여야 합니다.")
    if not fried_rice_trajectory:
        insert_high = _pair(trajectory, "insert_phase_ratio_range", positive=True)[1]
        catch_high = _pair(trajectory, "catch_phase_ratio_range", positive=True)[1]
        if insert_high + catch_high >= 0.95:
            raise ConfigError("insert/catch phase ratio 최댓값의 합은 0.95 미만이어야 합니다.")

    if float(simulation.get("timestep_s", 0.0)) <= 0:
        raise ConfigError("simulation.timestep_s는 양수여야 합니다.")
    if int(simulation.get("max_steps", 0)) <= 0:
        raise ConfigError("simulation.max_steps는 1 이상이어야 합니다.")

    if int(mixing.get("grid_rows", 0)) <= 0 or int(mixing.get("grid_cols", 0)) <= 0:
        raise ConfigError("mixing grid 크기는 양수여야 합니다.")
    if bool(robot.get("required", False)) and not bool(robot.get("enabled", False)):
        raise ConfigError("robot.required=true이면 robot.enabled도 true여야 합니다.")


def _resolve_paths(config: dict[str, Any], config_path: Path) -> None:
    for section_name, key in (("pan", "stl_path"), ("robot", "urdf_path")):
        raw = config.get(section_name, {}).get(key)
        if not raw:
            continue
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = (config_path.parent / path).resolve()
        config[section_name][key] = str(path)


def load_config(
    path: str | Path,
    *,
    base_path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """YAML을 읽고 선택적 base/override를 병합한 뒤 경로와 값을 검증한다."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"설정 파일을 찾을 수 없습니다: {config_path}")

    def read_yaml(yaml_path: Path) -> dict[str, Any]:
        try:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"YAML 파싱 실패 ({yaml_path}): {exc}") from exc
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ConfigError(f"YAML 최상위 값은 mapping이어야 합니다: {yaml_path}")
        return data

    current_data = read_yaml(config_path)
    declared_base = current_data.pop("base_config", None)
    if base_path is not None and declared_base is not None:
        raise ConfigError("base_path 인자와 YAML base_config는 동시에 지정할 수 없습니다.")
    effective_base = base_path
    if declared_base is not None:
        declared_path = Path(str(declared_base)).expanduser()
        effective_base = (
            declared_path if declared_path.is_absolute() else config_path.parent / declared_path
        )

    data: dict[str, Any] = {}
    if effective_base is not None:
        base = Path(effective_base).expanduser().resolve()
        if not base.is_file():
            raise ConfigError(f"기본 설정 파일을 찾을 수 없습니다: {base}")
        data = read_yaml(base)
        _resolve_paths(data, base)
    _resolve_paths(current_data, config_path)
    data = _deep_merge(data, current_data)
    if overrides:
        override_data = deepcopy(dict(overrides))
        _resolve_paths(override_data, config_path)
        data = _deep_merge(data, override_data)
    data["_config_path"] = str(config_path)
    validate_config(data)
    return data


def dump_effective_config(config: Mapping[str, Any], path: str | Path) -> Path:
    """실행에 사용된 설정을 재현 가능한 YAML로 저장한다."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serializable = {key: value for key, value in config.items() if not key.startswith("_")}
    destination.write_text(
        yaml.safe_dump(serializable, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return destination
