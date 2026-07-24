"""질량 보존 입자 생성 유틸리티.

이 모듈은 설정 시스템이나 MuJoCo에 의존하지 않는다. 따라서 테스트, 다른
물리 엔진, 또는 모델 XML 생성 코드에서 동일한 입자 표본을 재사용할 수 있다.
모든 입력과 출력은 SI 단위다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

_SPHERE_VOLUME_FACTOR = 4.0 * np.pi / 3.0
_SUPPORTED_GEOM_TYPES = frozenset({"sphere", "ellipsoid"})


def _positive_finite(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite value, got {value!r}")
    return value


def _nonnegative_finite(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a non-negative finite value, got {value!r}")
    return value


@dataclass(frozen=True)
class ParticleBatch:
    """한 에피소드에서 사용하는 입자 반지름, 질량, 초기 위치.

    Attributes:
        radii_m: 각 입자의 반지름, shape ``(N,)``.
        masses_kg: 각 입자의 질량, shape ``(N,)``.
        positions_m: pan-local 초기 중심 위치, shape ``(N, 3)``.
        density_kg_m3: 모든 입자에 공통으로 사용한 밀도.
        target_total_mass_kg: 요청한 합산 질량.
        seed: 생성에 사용한 seed. ``None``이면 비결정적 seed를 사용했다.
        geom_types: MuJoCo geom type. 생략하면 모든 입자가 ``sphere``다.
        sizes_m: MuJoCo geom half-size, shape ``(N, 3)``. 구는 세 축이 같다.
        species: 입자 종류 식별자. 기록/종류별 평가에 사용한다.
        quaternions_wxyz: pan-local 초기 자세. 생략하면 단위 quaternion이다.
        restitution_coefficients: 기준 낙하/반발 높이에서 얻은 이상적 반발계수.
        contact_time_constants_s/contact_damping_ratios: MuJoCo ``solref`` 값.
        frictions: MuJoCo sliding/torsional/rolling friction, shape ``(N, 3)``.
        linear_damping_per_s/angular_damping_per_s: 입자별 안정적 속도 감쇠율.
        reference_drop_heights_m/target_rebound_heights_m: 반발 물성의 실험 목표.

    새 필드는 모두 optional이므로 기존 exact-mass 구형 입자 API와 직렬화
    계약은 그대로 유지된다.
    """

    radii_m: NDArray[np.float64]
    masses_kg: NDArray[np.float64]
    positions_m: NDArray[np.float64]
    density_kg_m3: float
    target_total_mass_kg: float
    seed: int | None
    geom_types: NDArray[np.str_] | None = None
    sizes_m: NDArray[np.float64] | None = None
    species: NDArray[np.str_] | None = None
    quaternions_wxyz: NDArray[np.float64] | None = None
    restitution_coefficients: NDArray[np.float64] | None = None
    contact_time_constants_s: NDArray[np.float64] | None = None
    contact_damping_ratios: NDArray[np.float64] | None = None
    frictions: NDArray[np.float64] | None = None
    linear_damping_per_s: NDArray[np.float64] | None = None
    angular_damping_per_s: NDArray[np.float64] | None = None
    reference_drop_heights_m: NDArray[np.float64] | None = None
    target_rebound_heights_m: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        radii = np.asarray(self.radii_m, dtype=np.float64)
        masses = np.asarray(self.masses_kg, dtype=np.float64)
        positions = np.asarray(self.positions_m, dtype=np.float64)
        if radii.ndim != 1 or radii.size == 0:
            raise ValueError("radii_m must be a non-empty one-dimensional array")
        if masses.shape != radii.shape:
            raise ValueError("masses_kg must have the same shape as radii_m")
        if positions.shape != (radii.size, 3):
            raise ValueError("positions_m must have shape (N, 3)")
        if (
            not np.all(np.isfinite(radii))
            or not np.all(np.isfinite(masses))
            or not np.all(np.isfinite(positions))
            or np.any(radii <= 0.0)
            or np.any(masses <= 0.0)
        ):
            raise ValueError("particle arrays must contain finite, positive sizes and masses")

        # frozen dataclass에서도 입력 view의 외부 변경을 막기 위해 독립 복사본을 둔다.
        radii = radii.copy()
        masses = masses.copy()
        positions = positions.copy()
        radii.setflags(write=False)
        masses.setflags(write=False)
        positions.setflags(write=False)
        object.__setattr__(self, "radii_m", radii)
        object.__setattr__(self, "masses_kg", masses)
        object.__setattr__(self, "positions_m", positions)

        count = radii.size
        geom_types = (
            np.full(count, "sphere", dtype="<U10")
            if self.geom_types is None
            else np.asarray(self.geom_types, dtype="<U10")
        )
        if geom_types.shape != (count,):
            raise ValueError("geom_types must have shape (N,)")
        unsupported = set(np.unique(geom_types)) - _SUPPORTED_GEOM_TYPES
        if unsupported:
            raise ValueError(f"unsupported particle geom types: {sorted(unsupported)}")

        sizes = (
            np.repeat(radii[:, None], 3, axis=1)
            if self.sizes_m is None
            else np.asarray(self.sizes_m, dtype=np.float64)
        )
        if sizes.shape != (count, 3):
            raise ValueError("sizes_m must have shape (N, 3)")
        if not np.all(np.isfinite(sizes)) or np.any(sizes <= 0.0):
            raise ValueError("sizes_m must contain only positive finite values")
        if np.any(np.max(sizes, axis=1) > radii * (1.0 + 1.0e-12)):
            raise ValueError("radii_m must bound every geom semi-axis in sizes_m")
        sphere_mask = geom_types == "sphere"
        if np.any(
            ~np.isclose(
                sizes[sphere_mask],
                radii[sphere_mask, None],
                rtol=1.0e-12,
                atol=1.0e-15,
            )
        ):
            raise ValueError("sphere sizes_m must equal radii_m on all three axes")

        species = (
            geom_types.copy() if self.species is None else np.asarray(self.species, dtype="<U64")
        )
        if species.shape != (count,) or np.any(np.char.str_len(species) == 0):
            raise ValueError("species must contain N non-empty strings")

        quaternions = (
            np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (count, 1))
            if self.quaternions_wxyz is None
            else np.asarray(self.quaternions_wxyz, dtype=np.float64)
        )
        if quaternions.shape != (count, 4) or not np.all(np.isfinite(quaternions)):
            raise ValueError("quaternions_wxyz must be a finite array with shape (N, 4)")
        quaternion_norms = np.linalg.norm(quaternions, axis=1)
        if np.any(quaternion_norms <= 1.0e-12):
            raise ValueError("quaternions_wxyz cannot contain a zero quaternion")
        quaternions = quaternions / quaternion_norms[:, None]

        def optional_vector(
            name: str,
            value: NDArray[np.float64] | None,
            *,
            positive: bool,
            upper_bound: float | None = None,
        ) -> NDArray[np.float64] | None:
            if value is None:
                return None
            array = np.asarray(value, dtype=np.float64)
            if array.shape != (count,) or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be a finite array with shape (N,)")
            if positive and np.any(array <= 0.0):
                raise ValueError(f"{name} must contain only positive values")
            if not positive and np.any(array < 0.0):
                raise ValueError(f"{name} must contain only non-negative values")
            if upper_bound is not None and np.any(array >= upper_bound):
                raise ValueError(f"{name} must be smaller than {upper_bound}")
            return array.copy()

        restitution = optional_vector(
            "restitution_coefficients",
            self.restitution_coefficients,
            positive=False,
            upper_bound=1.0,
        )
        contact_time = optional_vector(
            "contact_time_constants_s",
            self.contact_time_constants_s,
            positive=True,
        )
        contact_damping = optional_vector(
            "contact_damping_ratios",
            self.contact_damping_ratios,
            positive=True,
        )
        linear_damping = optional_vector(
            "linear_damping_per_s",
            self.linear_damping_per_s,
            positive=False,
        )
        angular_damping = optional_vector(
            "angular_damping_per_s",
            self.angular_damping_per_s,
            positive=False,
        )
        drop_heights = optional_vector(
            "reference_drop_heights_m",
            self.reference_drop_heights_m,
            positive=True,
        )
        rebound_heights = optional_vector(
            "target_rebound_heights_m",
            self.target_rebound_heights_m,
            positive=False,
        )
        if drop_heights is not None and rebound_heights is not None:
            if np.any(rebound_heights >= drop_heights):
                raise ValueError("target rebound heights must be lower than drop heights")

        frictions: NDArray[np.float64] | None
        if self.frictions is None:
            frictions = None
        else:
            frictions = np.asarray(self.frictions, dtype=np.float64)
            if frictions.shape != (count, 3):
                raise ValueError("frictions must have shape (N, 3)")
            if not np.all(np.isfinite(frictions)) or np.any(frictions < 0.0):
                raise ValueError("frictions must contain only non-negative finite values")
            frictions = frictions.copy()

        normalized_arrays: tuple[tuple[str, np.ndarray | None], ...] = (
            ("geom_types", geom_types.copy()),
            ("sizes_m", sizes.copy()),
            ("species", species.copy()),
            ("quaternions_wxyz", quaternions.copy()),
            ("restitution_coefficients", restitution),
            ("contact_time_constants_s", contact_time),
            ("contact_damping_ratios", contact_damping),
            ("frictions", frictions),
            ("linear_damping_per_s", linear_damping),
            ("angular_damping_per_s", angular_damping),
            ("reference_drop_heights_m", drop_heights),
            ("target_rebound_heights_m", rebound_heights),
        )
        for name, array in normalized_arrays:
            if array is not None:
                array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def count(self) -> int:
        """입자 개수."""

        return int(self.radii_m.size)

    @property
    def actual_total_mass_kg(self) -> float:
        """부동소수점 계산으로 얻은 실제 합산 질량."""

        return float(np.sum(self.masses_kg, dtype=np.float64))

    @property
    def mean_radius_m(self) -> float:
        """평균 반지름."""

        return float(np.mean(self.radii_m))

    @property
    def radius_std_m(self) -> float:
        """반지름의 모집단 표준편차."""

        return float(np.std(self.radii_m))

    # 간결한 이름은 시뮬레이터 adapter와 대화형 사용을 위한 호환 alias다.
    @property
    def radii(self) -> NDArray[np.float64]:
        return self.radii_m

    @property
    def masses(self) -> NDArray[np.float64]:
        return self.masses_kg

    @property
    def positions(self) -> NDArray[np.float64]:
        return self.positions_m


# 이전/외부 코드가 ParticleSet이라는 이름을 사용해도 같은 계약을 제공한다.
ParticleSet = ParticleBatch


def particle_masses(
    radii_m: ArrayLike,
    density_kg_m3: float,
) -> NDArray[np.float64]:
    """구형 입자의 질량을 계산한다.

    Args:
        radii_m: 양의 반지름 배열.
        density_kg_m3: 양의 공통 밀도.
    """

    density = _positive_finite("density_kg_m3", density_kg_m3)
    radii = np.asarray(radii_m, dtype=np.float64)
    if radii.ndim != 1 or radii.size == 0:
        raise ValueError("radii_m must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("radii_m must contain only positive finite values")
    return density * _SPHERE_VOLUME_FACTOR * np.power(radii, 3)


def generate_radii_for_target_mass(
    *,
    count: int,
    density_kg_m3: float,
    nominal_radius_m: float,
    radius_jitter_fraction: float,
    target_total_mass_kg: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """상대 크기를 표본화하고 목표 합산 질량에 맞게 전역 스케일한다.

    상대 크기 ``u_i``는 ``[1-jitter, 1+jitter]``의 균등 분포에서
    생성한다. 그 뒤 요구사항의 해석식으로 모든 반지름에 동일한 scale을
    적용한다. 그러므로 입자 수와 밀도는 에피소드 간 고정되고, 상대 크기
    순서도 global scaling에 의해 바뀌지 않는다.
    """

    if isinstance(count, bool) or int(count) != count or int(count) <= 0:
        raise ValueError(f"count must be a positive integer, got {count!r}")
    count = int(count)
    density = _positive_finite("density_kg_m3", density_kg_m3)
    nominal_radius = _positive_finite("nominal_radius_m", nominal_radius_m)
    target_mass = _positive_finite("target_total_mass_kg", target_total_mass_kg)
    jitter = _nonnegative_finite("radius_jitter_fraction", radius_jitter_fraction)
    if jitter >= 1.0:
        raise ValueError("radius_jitter_fraction must be smaller than 1")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an instance of numpy.random.Generator")

    relative_sizes = rng.uniform(1.0 - jitter, 1.0 + jitter, size=count)
    unscaled = nominal_radius * relative_sizes
    unscaled_mass = (
        density * _SPHERE_VOLUME_FACTOR * float(np.sum(np.power(unscaled, 3), dtype=np.float64))
    )
    global_scale = np.cbrt(target_mass / unscaled_mass)
    radii = np.asarray(global_scale * unscaled, dtype=np.float64)
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise FloatingPointError("target-mass radius scaling produced invalid radii")
    return radii


def sample_non_overlapping_positions(
    radii_m: ArrayLike,
    *,
    spawn_radius_m: float,
    spawn_height_m: float,
    rng: np.random.Generator,
    center_m: ArrayLike = (0.0, 0.0, 0.0),
    minimum_clearance_m: float = 0.0,
    max_attempts_per_particle: int = 20_000,
) -> NDArray[np.float64]:
    """팬 중앙 원판에 서로 겹치지 않는 입자 중심을 생성한다.

    입자들은 pan-local ``xy`` 원판 안에서 면적 균일하게 표본화된다.
    ``center_m[2] + spawn_height_m``은 팬 바닥에서 띄운 추가 간격이며,
    실제 구 중심 높이는 여기에 각 반지름을 더한다. 한 평면의 실용적
    packing 밀도를 넘는 경우에만 여러 수직 층을 사용한다. 층 간격은
    가장 큰 입자의 지름 이상이므로 다른 층끼리도 겹치지 않는다. 따라서
    바닥을 관통하지 않으며, settling 동안 자연스럽게 팬 바닥으로
    내려온다. 큰 입자부터 배치해 rejection sampling의 실패율을 낮추며,
    반환 순서는 원래 입자 순서를 보존한다.

    요청한 입자들이 원판에 들어갈 수 없으면 겹침을 허용하지 않고 명확한
    ``RuntimeError``를 발생시킨다.
    """

    radii = np.asarray(radii_m, dtype=np.float64)
    if radii.ndim != 1 or radii.size == 0:
        raise ValueError("radii_m must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("radii_m must contain only positive finite values")
    spawn_radius = _positive_finite("spawn_radius_m", spawn_radius_m)
    spawn_height = _nonnegative_finite("spawn_height_m", spawn_height_m)
    clearance = _nonnegative_finite("minimum_clearance_m", minimum_clearance_m)
    if (
        isinstance(max_attempts_per_particle, bool)
        or int(max_attempts_per_particle) != max_attempts_per_particle
        or int(max_attempts_per_particle) <= 0
    ):
        raise ValueError("max_attempts_per_particle must be a positive integer")
    max_attempts_per_particle = int(max_attempts_per_particle)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an instance of numpy.random.Generator")

    center = np.asarray(center_m, dtype=np.float64)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("center_m must contain exactly three finite coordinates")
    if float(np.max(radii)) + 0.5 * clearance > spawn_radius:
        raise ValueError("spawn_radius_m is too small for the largest particle and clearance")

    positions = np.empty((radii.size, 3), dtype=np.float64)
    placed: list[int] = []

    # 순차 무작위 원판 packing은 약 0.5부터 실패율이 급격히 커진다.
    # 0.45를 목표로 두면 마지막 몇 입자의 rejection 폭증을 피하면서도
    # 불필요하게 높은 수직 stack을 만들지 않는다.
    # 필요한 경우에만 층을 추가해 고질량 demo 설정도 겹침 없이 생성한다.
    effective_radii = radii + 0.5 * clearance
    projected_area_ratio = float(
        np.sum(effective_radii**2, dtype=np.float64) / (spawn_radius - 0.5 * clearance) ** 2
    )
    layer_count = max(1, int(np.ceil(projected_area_ratio / 0.45)))
    layer_spacing = 2.0 * float(np.max(radii)) + clearance

    # 동일 반지름일 때 seed에 따라 배치 순서도 달라지도록 tie-breaker를 둔다.
    tie_breaker = rng.random(radii.size)
    order = np.lexsort((tie_breaker, -radii))
    for particle_index in order:
        radius = float(radii[particle_index])
        available_radius = spawn_radius - radius - 0.5 * clearance
        accepted = False
        for _ in range(max_attempts_per_particle):
            radial_distance = available_radius * np.sqrt(rng.random())
            angle = rng.uniform(-np.pi, np.pi)
            layer_index = int(rng.integers(0, layer_count))
            candidate = np.array(
                [
                    center[0] + radial_distance * np.cos(angle),
                    center[1] + radial_distance * np.sin(angle),
                    center[2] + spawn_height + radius + layer_index * layer_spacing,
                ],
                dtype=np.float64,
            )
            if placed:
                placed_indices = np.asarray(placed, dtype=np.int64)
                distances = np.linalg.norm(positions[placed_indices] - candidate[None, :], axis=1)
                minimum_distances = radii[placed_indices] + radius + clearance
                if np.any(distances < minimum_distances):
                    continue
            positions[particle_index] = candidate
            placed.append(int(particle_index))
            accepted = True
            break
        if not accepted:
            raise RuntimeError(
                "could not place all particles without overlap; increase "
                "spawn_radius_m, reduce target mass/count/clearance, or increase "
                "max_attempts_per_particle"
            )

    return positions


class ParticleGenerator:
    """고정 입자 수·밀도를 유지하는 재현 가능한 에피소드 생성기."""

    def __init__(
        self,
        *,
        count: int,
        density_kg_m3: float,
        nominal_radius_m: float,
        radius_jitter_fraction: float,
        spawn_radius_m: float,
        spawn_height_m: float,
        mass_tolerance_kg: float = 1e-12,
        minimum_clearance_m: float = 0.0,
        max_spawn_attempts: int = 20_000,
    ) -> None:
        if isinstance(count, bool) or int(count) != count or int(count) <= 0:
            raise ValueError("count must be a positive integer")
        self.count = int(count)
        self.density_kg_m3 = _positive_finite("density_kg_m3", density_kg_m3)
        self.nominal_radius_m = _positive_finite("nominal_radius_m", nominal_radius_m)
        self.radius_jitter_fraction = _nonnegative_finite(
            "radius_jitter_fraction", radius_jitter_fraction
        )
        if self.radius_jitter_fraction >= 1.0:
            raise ValueError("radius_jitter_fraction must be smaller than 1")
        self.spawn_radius_m = _positive_finite("spawn_radius_m", spawn_radius_m)
        self.spawn_height_m = _nonnegative_finite("spawn_height_m", spawn_height_m)
        self.mass_tolerance_kg = _positive_finite("mass_tolerance_kg", mass_tolerance_kg)
        self.minimum_clearance_m = _nonnegative_finite("minimum_clearance_m", minimum_clearance_m)
        if (
            isinstance(max_spawn_attempts, bool)
            or int(max_spawn_attempts) != max_spawn_attempts
            or int(max_spawn_attempts) <= 0
        ):
            raise ValueError("max_spawn_attempts must be a positive integer")
        self.max_spawn_attempts = int(max_spawn_attempts)

    def generate(
        self,
        target_total_mass_kg: float,
        *,
        seed: int | None = None,
        center_m: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> ParticleBatch:
        """한 에피소드의 입자를 생성한다.

        같은 생성기 인자, 목표 질량, seed는 bitwise 동일한 반지름과
        위치를 만든다. 반지름 표본화와 위치 표본화는 하나의 RNG stream을
        사용하므로 seed 관리도 한 곳에서 이뤄진다.
        """

        target_mass = _positive_finite("target_total_mass_kg", target_total_mass_kg)
        rng = np.random.default_rng(seed)
        radii = generate_radii_for_target_mass(
            count=self.count,
            density_kg_m3=self.density_kg_m3,
            nominal_radius_m=self.nominal_radius_m,
            radius_jitter_fraction=self.radius_jitter_fraction,
            target_total_mass_kg=target_mass,
            rng=rng,
        )
        masses = particle_masses(radii, self.density_kg_m3)
        actual_mass = float(np.sum(masses, dtype=np.float64))
        if not np.isclose(
            actual_mass,
            target_mass,
            rtol=0.0,
            atol=self.mass_tolerance_kg,
        ):
            raise FloatingPointError(
                "generated particle mass does not match target: "
                f"target={target_mass:.17g}, actual={actual_mass:.17g}, "
                f"tolerance={self.mass_tolerance_kg:.3g}"
            )
        positions = sample_non_overlapping_positions(
            radii,
            spawn_radius_m=self.spawn_radius_m,
            spawn_height_m=self.spawn_height_m,
            rng=rng,
            center_m=center_m,
            minimum_clearance_m=self.minimum_clearance_m,
            max_attempts_per_particle=self.max_spawn_attempts,
        )
        return ParticleBatch(
            radii_m=radii,
            masses_kg=masses,
            positions_m=positions,
            density_kg_m3=self.density_kg_m3,
            target_total_mass_kg=target_mass,
            seed=seed,
        )


@dataclass(frozen=True)
class FriedRiceSpeciesSpec:
    """볶음밥 입자 한 종류의 geometry, 질량, 반발 물성."""

    name: str
    geom_type: str
    count: int
    semi_axes_m: tuple[float, float, float]
    nominal_mass_kg: float
    mass_jitter_fraction: float
    target_rebound_height_m: float
    contact_damping_ratio: float
    linear_damping_per_s: float = 0.05
    angular_damping_per_s: float = 0.03
    friction: tuple[float, float, float] = (0.55, 0.01, 0.001)

    def __post_init__(self) -> None:
        if not str(self.name):
            raise ValueError("fried-rice species name cannot be empty")
        geom_type = str(self.geom_type).lower()
        if geom_type not in _SUPPORTED_GEOM_TYPES:
            raise ValueError(f"unsupported fried-rice geom type: {geom_type}")
        if isinstance(self.count, bool) or int(self.count) != self.count or self.count <= 0:
            raise ValueError("fried-rice species count must be a positive integer")
        axes = np.asarray(self.semi_axes_m, dtype=np.float64)
        if axes.shape != (3,) or not np.all(np.isfinite(axes)) or np.any(axes <= 0.0):
            raise ValueError("semi_axes_m must contain three positive finite values")
        if geom_type == "sphere" and not np.allclose(axes, axes[0], rtol=1.0e-12, atol=1.0e-15):
            raise ValueError("sphere semi_axes_m must have equal axes")
        positive_values = {
            "nominal_mass_kg": self.nominal_mass_kg,
            "target_rebound_height_m": self.target_rebound_height_m,
            "contact_damping_ratio": self.contact_damping_ratio,
        }
        for name, raw_value in positive_values.items():
            value = float(raw_value)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite value")
        jitter = float(self.mass_jitter_fraction)
        if not np.isfinite(jitter) or not 0.0 <= jitter < 1.0:
            raise ValueError("mass_jitter_fraction must be finite and in [0, 1)")
        damping_values = (
            float(self.linear_damping_per_s),
            float(self.angular_damping_per_s),
        )
        if not np.isfinite(damping_values).all() or min(damping_values) < 0.0:
            raise ValueError("species damping values must be non-negative and finite")
        friction = np.asarray(self.friction, dtype=np.float64)
        if friction.shape != (3,) or not np.isfinite(friction).all() or np.any(friction < 0.0):
            raise ValueError("species friction must contain three non-negative finite values")
        object.__setattr__(self, "geom_type", geom_type)
        object.__setattr__(self, "count", int(self.count))
        object.__setattr__(self, "semi_axes_m", tuple(float(item) for item in axes))
        object.__setattr__(self, "friction", tuple(float(item) for item in friction))

    @property
    def bounding_radius_m(self) -> float:
        return float(max(self.semi_axes_m))

    @property
    def volume_m3(self) -> float:
        axes = np.asarray(self.semi_axes_m, dtype=np.float64)
        return float(_SPHERE_VOLUME_FACTOR * np.prod(axes))


# MuJoCo 3.10, timestep=0.002 s, contact_time_constant=0.006 s에서
# ``python -m wok_sim.simulation.rebound``로 flat-plane 첫 반발을 보정한 값.
# priority=1인 particle geom이 팬 proxy보다 높은 contact priority를 가져야
# 이 입자별 damping ratio가 실제 contact에 선택된다.
DEFAULT_FRIED_RICE_SPECIES: tuple[FriedRiceSpeciesSpec, ...] = (
    FriedRiceSpeciesSpec(
        name="large_sphere",
        geom_type="sphere",
        count=20,
        semi_axes_m=(0.005, 0.005, 0.005),
        nominal_mass_kg=0.002,
        mass_jitter_fraction=0.10,
        target_rebound_height_m=0.030,
        contact_damping_ratio=0.300627381729,
    ),
    FriedRiceSpeciesSpec(
        name="small_sphere",
        geom_type="sphere",
        count=20,
        semi_axes_m=(0.003, 0.003, 0.003),
        nominal_mass_kg=0.0008,
        mass_jitter_fraction=0.0,
        target_rebound_height_m=0.010,
        contact_damping_ratio=0.516754797652,
    ),
    FriedRiceSpeciesSpec(
        name="ellipsoid",
        geom_type="ellipsoid",
        count=20,
        semi_axes_m=(0.005, 0.002, 0.002),
        nominal_mass_kg=0.0002,
        mass_jitter_fraction=0.0,
        target_rebound_height_m=0.015,
        contact_damping_ratio=0.469157864254,
    ),
)


def _random_unit_quaternions(
    count: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    quaternions = rng.normal(size=(count, 4))
    quaternions /= np.linalg.norm(quaternions, axis=1)[:, None]
    # q와 -q는 같은 회전이므로 w>=0 canonical form을 사용한다.
    quaternions[quaternions[:, 0] < 0.0] *= -1.0
    return np.asarray(quaternions, dtype=np.float64)


def _config_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError("fried-rice particle config must be a mapping")


def _friction_tuple(value: Any) -> tuple[float, float, float]:
    if np.isscalar(value):
        return (float(value), 0.01, 0.001)
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError("friction must be a scalar or length-three array")
    return tuple(float(item) for item in array)


class FriedRiceParticleGenerator:
    """세 종류의 heterogeneous 볶음밥 입자를 생성한다.

    첫 종류의 질량만 ``nominal*(1±jitter)`` 균등분포에서 표본화하며,
    나머지 두 종류는 지정 질량을 그대로 사용한다. 크기/질량을 target mass에
    맞춰 전역 scaling하지 않는다. legacy 환경에 주입할 수 있도록
    :meth:`generate`의 첫 positional target mass는 optional로 받지만 무시하고,
    실제 표본 질량 합을 batch의 target/actual mass로 기록한다.

    ``count_per_type_range``가 설정되면 에피소드마다 범위 안의 정수 하나를
    표본화해 모든 종류에 같은 개수를 적용한다. 따라서 혼합비 1:1:1을
    유지하면서 총 입자 수와 총질량을 함께 domain-randomize할 수 있다.
    """

    def __init__(
        self,
        *,
        spawn_radius_m: float,
        spawn_height_m: float,
        species: Sequence[FriedRiceSpeciesSpec] = DEFAULT_FRIED_RICE_SPECIES,
        count_per_type_range: Sequence[int] | None = None,
        reference_drop_height_m: float = 0.20,
        contact_time_constant_s: float = 0.006,
        calibration_timestep_s: float = 0.002,
        minimum_clearance_m: float = 0.0,
        max_spawn_attempts: int = 20_000,
    ) -> None:
        if not species:
            raise ValueError("at least one fried-rice species is required")
        self.species = tuple(species)
        if not all(isinstance(item, FriedRiceSpeciesSpec) for item in self.species):
            raise TypeError("species must contain FriedRiceSpeciesSpec values")
        if count_per_type_range is None:
            self.count_per_type_range: tuple[int, int] | None = None
        else:
            try:
                raw_low, raw_high = count_per_type_range
            except (TypeError, ValueError) as exc:
                raise ValueError("count_per_type_range must contain exactly two integers") from exc
            try:
                count_low, count_high = int(raw_low), int(raw_high)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "count_per_type_range must be an increasing pair of positive integers"
                ) from exc
            if (
                isinstance(raw_low, bool)
                or isinstance(raw_high, bool)
                or count_low != raw_low
                or count_high != raw_high
                or count_low <= 0
                or count_low > count_high
            ):
                raise ValueError(
                    "count_per_type_range must be an increasing pair of positive integers"
                )
            if len({item.count for item in self.species}) != 1:
                raise ValueError(
                    "variable count_per_type_range requires equal baseline species counts"
                )
            self.count_per_type_range = (count_low, count_high)
        self.spawn_radius_m = _positive_finite("spawn_radius_m", spawn_radius_m)
        self.spawn_height_m = _nonnegative_finite("spawn_height_m", spawn_height_m)
        self.reference_drop_height_m = _positive_finite(
            "reference_drop_height_m", reference_drop_height_m
        )
        self.contact_time_constant_s = _positive_finite(
            "contact_time_constant_s", contact_time_constant_s
        )
        self.calibration_timestep_s = _positive_finite(
            "calibration_timestep_s", calibration_timestep_s
        )
        self.minimum_clearance_m = _nonnegative_finite("minimum_clearance_m", minimum_clearance_m)
        if any(
            item.target_rebound_height_m >= self.reference_drop_height_m for item in self.species
        ):
            raise ValueError("every target rebound height must be below the reference drop height")
        if (
            isinstance(max_spawn_attempts, bool)
            or int(max_spawn_attempts) != max_spawn_attempts
            or int(max_spawn_attempts) <= 0
        ):
            raise ValueError("max_spawn_attempts must be a positive integer")
        self.max_spawn_attempts = int(max_spawn_attempts)

    @classmethod
    def from_config(cls, particle_config: Mapping[str, Any] | Any) -> FriedRiceParticleGenerator:
        """``particles`` section과 optional ``fried_rice`` overrides를 읽는다.

        지원 schema 예시::

            particles:
              profile: fried_rice
              spawn_radius_m: 0.065
              spawn_height_m: 0.012
              friction: [0.55, 0.01, 0.001]
              linear_damping: 0.05
              angular_damping: 0.03
              fried_rice:
                count_per_type: 20
                count_per_type_range: [20, 40]
                reference_drop_height_m: 0.20
                contact_time_constant_s: 0.006
                calibration_timestep_s: 0.002
                large_sphere:
                  radius_m: 0.005
                  mass_mean_kg: 0.002
                  mass_jitter_fraction: 0.10
                  target_rebound_height_m: 0.030
        """

        root = _config_mapping(particle_config)
        profile = _config_mapping(root.get("fried_rice", {}))
        common_count = int(profile.get("count_per_type", 20))
        count_per_type_range = profile.get("count_per_type_range")
        common_friction = _friction_tuple(root.get("friction", (0.55, 0.01, 0.001)))
        common_linear_damping = float(root.get("linear_damping", 0.05))
        common_angular_damping = float(root.get("angular_damping", 0.03))

        defaults_by_name = {item.name: item for item in DEFAULT_FRIED_RICE_SPECIES}
        specs: list[FriedRiceSpeciesSpec] = []
        for name in ("large_sphere", "small_sphere", "ellipsoid"):
            default = defaults_by_name[name]
            override = _config_mapping(profile.get(name, {}))
            if name == "ellipsoid":
                axes_raw = override.get(
                    "semi_axes_m",
                    override.get("axes_m", default.semi_axes_m),
                )
                axes = tuple(float(item) for item in np.asarray(axes_raw, dtype=float))
            else:
                radius = float(override.get("radius_m", default.semi_axes_m[0]))
                axes = (radius, radius, radius)
            nominal_mass = float(
                override.get(
                    "mass_mean_kg",
                    override.get("mass_kg", default.nominal_mass_kg),
                )
            )
            specs.append(
                FriedRiceSpeciesSpec(
                    name=name,
                    geom_type=default.geom_type,
                    count=int(override.get("count", common_count)),
                    semi_axes_m=axes,
                    nominal_mass_kg=nominal_mass,
                    mass_jitter_fraction=float(
                        override.get(
                            "mass_jitter_fraction",
                            default.mass_jitter_fraction,
                        )
                    ),
                    target_rebound_height_m=float(
                        override.get(
                            "target_rebound_height_m",
                            override.get(
                                "rebound_height_m",
                                default.target_rebound_height_m,
                            ),
                        )
                    ),
                    contact_damping_ratio=float(
                        override.get(
                            "contact_damping_ratio",
                            default.contact_damping_ratio,
                        )
                    ),
                    linear_damping_per_s=float(
                        override.get(
                            "linear_damping_per_s",
                            override.get("linear_damping", common_linear_damping),
                        )
                    ),
                    angular_damping_per_s=float(
                        override.get(
                            "angular_damping_per_s",
                            override.get("angular_damping", common_angular_damping),
                        )
                    ),
                    friction=_friction_tuple(override.get("friction", common_friction)),
                )
            )
        return cls(
            spawn_radius_m=float(root.get("spawn_radius_m", 0.065)),
            spawn_height_m=float(root.get("spawn_height_m", 0.012)),
            species=specs,
            count_per_type_range=count_per_type_range,
            reference_drop_height_m=float(profile.get("reference_drop_height_m", 0.20)),
            contact_time_constant_s=float(profile.get("contact_time_constant_s", 0.006)),
            calibration_timestep_s=float(profile.get("calibration_timestep_s", 0.002)),
            minimum_clearance_m=float(
                root.get("minimum_clearance_m", root.get("spawn_clearance_m", 0.0))
            ),
            max_spawn_attempts=int(root.get("max_spawn_attempts", 20_000)),
        )

    @property
    def count(self) -> int:
        """기준(100%) 입자 수."""

        return sum(item.count for item in self.species)

    @property
    def count_range(self) -> tuple[int, int]:
        """생성 가능한 총 입자 수의 inclusive 범위."""

        if self.count_per_type_range is None:
            return (self.count, self.count)
        low, high = self.count_per_type_range
        species_count = len(self.species)
        return (species_count * low, species_count * high)

    def generate(
        self,
        target_total_mass_kg: float | None = None,
        *,
        seed: int | None = None,
        center_m: Sequence[float] = (0.0, 0.0, 0.0),
        count_per_type: int | None = None,
    ) -> ParticleBatch:
        """seed에 대해 개수·질량·배치가 재현 가능한 heterogeneous batch를 만든다.

        ``count_per_type``을 지정하면 설정 범위 안에서 무작위 표본 대신 해당
        개수를 사용한다. 이는 무게 구간별 평가를 위한 제어 입력이며 입자
        질량 자체를 재스케일하지 않는다.
        """

        # Legacy env compatibility only. Prescribed geometry/mass must not be rescaled.
        del target_total_mass_kg
        from .rebound import restitution_from_heights

        rng = np.random.default_rng(seed)
        if self.count_per_type_range is None:
            if count_per_type is not None:
                raise ValueError("count_per_type override requires configured count_per_type_range")
            episode_counts = [item.count for item in self.species]
        else:
            count_low, count_high = self.count_per_type_range
            if count_per_type is None:
                shared_count = int(rng.integers(count_low, count_high + 1))
            else:
                try:
                    parsed_count = int(count_per_type)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        "count_per_type must be an integer in configured range "
                        f"[{count_low}, {count_high}]"
                    ) from exc
                if (
                    isinstance(count_per_type, bool)
                    or parsed_count != count_per_type
                    or not count_low <= parsed_count <= count_high
                ):
                    raise ValueError(
                        "count_per_type must be an integer in configured range "
                        f"[{count_low}, {count_high}]"
                    )
                shared_count = parsed_count
            episode_counts = [shared_count] * len(self.species)

        radii_parts: list[np.ndarray] = []
        size_parts: list[np.ndarray] = []
        mass_parts: list[np.ndarray] = []
        geom_parts: list[np.ndarray] = []
        species_parts: list[np.ndarray] = []
        quaternion_parts: list[np.ndarray] = []
        restitution_parts: list[np.ndarray] = []
        contact_time_parts: list[np.ndarray] = []
        contact_damping_parts: list[np.ndarray] = []
        friction_parts: list[np.ndarray] = []
        linear_damping_parts: list[np.ndarray] = []
        angular_damping_parts: list[np.ndarray] = []
        drop_height_parts: list[np.ndarray] = []
        rebound_height_parts: list[np.ndarray] = []
        volume_parts: list[np.ndarray] = []

        for item, episode_count in zip(self.species, episode_counts, strict=True):
            radii_parts.append(np.full(episode_count, item.bounding_radius_m))
            size_parts.append(np.tile(np.asarray(item.semi_axes_m), (episode_count, 1)))
            if item.mass_jitter_fraction > 0.0:
                factors = rng.uniform(
                    1.0 - item.mass_jitter_fraction,
                    1.0 + item.mass_jitter_fraction,
                    size=episode_count,
                )
            else:
                factors = np.ones(episode_count)
            mass_parts.append(item.nominal_mass_kg * factors)
            geom_parts.append(np.full(episode_count, item.geom_type, dtype="<U10"))
            species_parts.append(np.full(episode_count, item.name, dtype="<U64"))
            quaternion_parts.append(_random_unit_quaternions(episode_count, rng))
            restitution = restitution_from_heights(
                self.reference_drop_height_m,
                item.target_rebound_height_m,
            )
            restitution_parts.append(np.full(episode_count, restitution))
            contact_time_parts.append(np.full(episode_count, self.contact_time_constant_s))
            contact_damping_parts.append(np.full(episode_count, item.contact_damping_ratio))
            friction_parts.append(np.tile(np.asarray(item.friction), (episode_count, 1)))
            linear_damping_parts.append(np.full(episode_count, item.linear_damping_per_s))
            angular_damping_parts.append(np.full(episode_count, item.angular_damping_per_s))
            drop_height_parts.append(np.full(episode_count, self.reference_drop_height_m))
            rebound_height_parts.append(np.full(episode_count, item.target_rebound_height_m))
            volume_parts.append(np.full(episode_count, item.volume_m3))

        radii = np.concatenate(radii_parts).astype(np.float64, copy=False)
        sizes = np.concatenate(size_parts).astype(np.float64, copy=False)
        masses = np.concatenate(mass_parts).astype(np.float64, copy=False)
        positions = sample_non_overlapping_positions(
            radii,
            spawn_radius_m=self.spawn_radius_m,
            spawn_height_m=self.spawn_height_m,
            rng=rng,
            center_m=center_m,
            minimum_clearance_m=self.minimum_clearance_m,
            max_attempts_per_particle=self.max_spawn_attempts,
        )
        total_mass = float(np.sum(masses, dtype=np.float64))
        total_volume = float(np.sum(np.concatenate(volume_parts), dtype=np.float64))
        aggregate_density = total_mass / total_volume
        return ParticleBatch(
            radii_m=radii,
            masses_kg=masses,
            positions_m=positions,
            density_kg_m3=aggregate_density,
            target_total_mass_kg=total_mass,
            seed=seed,
            geom_types=np.concatenate(geom_parts),
            sizes_m=sizes,
            species=np.concatenate(species_parts),
            quaternions_wxyz=np.concatenate(quaternion_parts),
            restitution_coefficients=np.concatenate(restitution_parts),
            contact_time_constants_s=np.concatenate(contact_time_parts),
            contact_damping_ratios=np.concatenate(contact_damping_parts),
            frictions=np.concatenate(friction_parts),
            linear_damping_per_s=np.concatenate(linear_damping_parts),
            angular_damping_per_s=np.concatenate(angular_damping_parts),
            reference_drop_heights_m=np.concatenate(drop_height_parts),
            target_rebound_heights_m=np.concatenate(rebound_height_parts),
        )


def generate_particles(
    *,
    count: int,
    density_kg_m3: float,
    nominal_radius_m: float,
    radius_jitter_fraction: float,
    target_total_mass_kg: float,
    spawn_radius_m: float,
    spawn_height_m: float,
    seed: int | None = None,
    center_m: Sequence[float] = (0.0, 0.0, 0.0),
    mass_tolerance_kg: float = 1e-12,
    minimum_clearance_m: float = 0.0,
    max_spawn_attempts: int = 20_000,
) -> ParticleBatch:
    """일회성 사용을 위한 :class:`ParticleGenerator` 편의 함수."""

    generator = ParticleGenerator(
        count=count,
        density_kg_m3=density_kg_m3,
        nominal_radius_m=nominal_radius_m,
        radius_jitter_fraction=radius_jitter_fraction,
        spawn_radius_m=spawn_radius_m,
        spawn_height_m=spawn_height_m,
        mass_tolerance_kg=mass_tolerance_kg,
        minimum_clearance_m=minimum_clearance_m,
        max_spawn_attempts=max_spawn_attempts,
    )
    return generator.generate(target_total_mass_kg, seed=seed, center_m=center_m)


__all__ = [
    "DEFAULT_FRIED_RICE_SPECIES",
    "FriedRiceParticleGenerator",
    "FriedRiceSpeciesSpec",
    "ParticleBatch",
    "ParticleGenerator",
    "ParticleSet",
    "generate_particles",
    "generate_radii_for_target_mass",
    "particle_masses",
    "sample_non_overlapping_positions",
]
