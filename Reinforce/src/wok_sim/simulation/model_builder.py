"""MuJoCo MJCF model builder.

팬은 mocap body로 생성되어 입자 접촉력에 영향을 받지 않고 주어진 pose를
정확히 따른다. STL은 visual geom이며, collision은 primitive compound만
활성화한다.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .pan_model import PanModel
from .rebound import damping_ratio_from_restitution


class ModelBuildError(RuntimeError):
    """MJCF 구성 또는 MuJoCo compile 실패."""


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"mapping 설정이 필요합니다. 받은 타입: {type(value).__name__}")


def _get(value: Any, *names: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    else:
        for name in names:
            if hasattr(value, name):
                return getattr(value, name)
    return default


def _numbers(values: Sequence[float]) -> str:
    return " ".join(f"{float(item):.12g}" for item in values)


def _normalize_quaternion(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=float)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ModelBuildError("quaternion은 유한한 wxyz 길이 4 배열이어야 합니다.")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ModelBuildError("quaternion norm이 0입니다.")
    return quaternion / norm


def _quaternion_to_matrix(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = _normalize_quaternion(quaternion_wxyz)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """회전행렬을 MuJoCo 순서(wxyz) quaternion으로 변환한다."""

    matrix = np.asarray(rotation, dtype=float)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    return _normalize_quaternion(quaternion)


@dataclass(frozen=True)
class ParticleModelSpec:
    """MuJoCo에 넣을 heterogeneous 입자들의 SI 단위 물성."""

    radii_m: np.ndarray
    masses_kg: np.ndarray
    positions_m: np.ndarray
    geom_types: np.ndarray | None = None
    sizes_m: np.ndarray | None = None
    species: np.ndarray | None = None
    quaternions_wxyz: np.ndarray | None = None
    restitution_coefficients: np.ndarray | None = None
    contact_time_constants_s: np.ndarray | None = None
    contact_damping_ratios: np.ndarray | None = None
    frictions: np.ndarray | None = None
    linear_damping_per_s: np.ndarray | None = None
    angular_damping_per_s: np.ndarray | None = None
    reference_drop_heights_m: np.ndarray | None = None
    target_rebound_heights_m: np.ndarray | None = None

    @classmethod
    def from_value(cls, particles: Any) -> ParticleModelSpec:
        """ParticleBatch, mapping 또는 legacy 구형 배열 객체를 정규화한다."""

        radii = np.asarray(_get(particles, "radii_m", "radii", default=None), dtype=float)
        masses = np.asarray(_get(particles, "masses_kg", "masses", default=None), dtype=float)
        positions = np.asarray(
            _get(
                particles,
                "positions_m",
                "positions",
                "initial_positions_m",
                default=None,
            ),
            dtype=float,
        )
        if radii.ndim != 1 or len(radii) == 0:
            raise ModelBuildError("particle radii는 비어 있지 않은 1차원 배열이어야 합니다.")
        if masses.shape != radii.shape:
            raise ModelBuildError("particle masses shape은 radii와 같아야 합니다.")
        if positions.shape != (len(radii), 3):
            raise ModelBuildError(
                f"particle positions shape은 {(len(radii), 3)}이어야 합니다: {positions.shape}"
            )
        if (
            not np.isfinite(radii).all()
            or not np.isfinite(masses).all()
            or not np.isfinite(positions).all()
        ):
            raise ModelBuildError("particle 배열에 NaN 또는 inf가 있습니다.")
        if np.any(radii <= 0.0) or np.any(masses <= 0.0):
            raise ModelBuildError("particle radius와 mass는 모두 양수여야 합니다.")
        count = len(radii)

        geom_types_value = _get(particles, "geom_types", default=None)
        geom_types = (
            np.full(count, "sphere", dtype="<U10")
            if geom_types_value is None
            else np.asarray(geom_types_value, dtype="<U10")
        )
        if geom_types.shape != (count,):
            raise ModelBuildError("particle geom_types shape은 (N,)이어야 합니다.")
        unsupported = set(np.unique(geom_types)) - {"sphere", "ellipsoid"}
        if unsupported:
            raise ModelBuildError(f"지원하지 않는 particle geom type: {sorted(unsupported)}")

        sizes_value = _get(particles, "sizes_m", "geom_sizes_m", default=None)
        sizes = (
            np.repeat(radii[:, None], 3, axis=1)
            if sizes_value is None
            else np.asarray(sizes_value, dtype=float)
        )
        if sizes.shape != (count, 3):
            raise ModelBuildError("particle sizes_m shape은 (N, 3)이어야 합니다.")
        if not np.isfinite(sizes).all() or np.any(sizes <= 0.0):
            raise ModelBuildError("particle sizes_m는 유한한 양수여야 합니다.")
        if np.any(np.max(sizes, axis=1) > radii * (1.0 + 1.0e-12)):
            raise ModelBuildError("particle radii_m는 모든 geom semi-axis 이상이어야 합니다.")
        sphere_mask = geom_types == "sphere"
        if np.any(
            ~np.isclose(
                sizes[sphere_mask],
                radii[sphere_mask, None],
                rtol=1.0e-12,
                atol=1.0e-15,
            )
        ):
            raise ModelBuildError("sphere sizes_m 세 축은 radii_m와 같아야 합니다.")

        species_value = _get(particles, "species", "particle_species", default=None)
        species = (
            geom_types.copy() if species_value is None else np.asarray(species_value, dtype="<U64")
        )
        if species.shape != (count,) or np.any(np.char.str_len(species) == 0):
            raise ModelBuildError("particle species는 비어 있지 않은 shape (N,) 문자열입니다.")

        quaternion_value = _get(
            particles,
            "quaternions_wxyz",
            "orientations_wxyz",
            default=None,
        )
        quaternions = (
            np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (count, 1))
            if quaternion_value is None
            else np.asarray(quaternion_value, dtype=float)
        )
        if quaternions.shape != (count, 4) or not np.isfinite(quaternions).all():
            raise ModelBuildError("particle quaternions_wxyz shape은 (N, 4)이어야 합니다.")
        quaternion_norms = np.linalg.norm(quaternions, axis=1)
        if np.any(quaternion_norms <= 1.0e-12):
            raise ModelBuildError("particle quaternion norm은 0일 수 없습니다.")
        quaternions = quaternions / quaternion_norms[:, None]

        def optional_vector(
            name: str,
            *aliases: str,
            positive: bool,
            upper_bound: float | None = None,
        ) -> np.ndarray | None:
            value = _get(particles, name, *aliases, default=None)
            if value is None:
                return None
            array = np.asarray(value, dtype=float)
            if array.shape != (count,) or not np.isfinite(array).all():
                raise ModelBuildError(f"particle {name} shape은 유한한 (N,)이어야 합니다.")
            if positive and np.any(array <= 0.0):
                raise ModelBuildError(f"particle {name} 값은 모두 양수여야 합니다.")
            if not positive and np.any(array < 0.0):
                raise ModelBuildError(f"particle {name} 값은 음수가 아니어야 합니다.")
            if upper_bound is not None and np.any(array >= upper_bound):
                raise ModelBuildError(f"particle {name} 값은 {upper_bound} 미만이어야 합니다.")
            return array.copy()

        restitution = optional_vector(
            "restitution_coefficients",
            "restitutions",
            positive=False,
            upper_bound=1.0,
        )
        contact_time = optional_vector(
            "contact_time_constants_s",
            positive=True,
        )
        contact_damping = optional_vector(
            "contact_damping_ratios",
            positive=True,
        )
        linear_damping = optional_vector(
            "linear_damping_per_s",
            positive=False,
        )
        angular_damping = optional_vector(
            "angular_damping_per_s",
            positive=False,
        )
        drop_heights = optional_vector(
            "reference_drop_heights_m",
            positive=True,
        )
        rebound_heights = optional_vector(
            "target_rebound_heights_m",
            positive=False,
        )
        if drop_heights is not None and rebound_heights is not None:
            if np.any(rebound_heights >= drop_heights):
                raise ModelBuildError("particle rebound height는 drop height보다 낮아야 합니다.")

        friction_value = _get(particles, "frictions", "friction_coefficients", default=None)
        frictions: np.ndarray | None
        if friction_value is None:
            frictions = None
        else:
            frictions = np.asarray(friction_value, dtype=float)
            if frictions.shape != (count, 3):
                raise ModelBuildError("particle frictions shape은 (N, 3)이어야 합니다.")
            if not np.isfinite(frictions).all() or np.any(frictions < 0.0):
                raise ModelBuildError("particle frictions는 유한한 음이 아닌 값이어야 합니다.")
            frictions = frictions.copy()

        return cls(
            radii_m=radii.copy(),
            masses_kg=masses.copy(),
            positions_m=positions.copy(),
            geom_types=geom_types.copy(),
            sizes_m=sizes.copy(),
            species=species.copy(),
            quaternions_wxyz=quaternions.copy(),
            restitution_coefficients=restitution,
            contact_time_constants_s=contact_time,
            contact_damping_ratios=contact_damping,
            frictions=frictions,
            linear_damping_per_s=linear_damping,
            angular_damping_per_s=angular_damping,
            reference_drop_heights_m=drop_heights,
            target_rebound_heights_m=rebound_heights,
        )


@dataclass
class BuiltModel:
    """compile된 MuJoCo model/data와 빠른 상태 접근용 ID."""

    model: Any
    data: Any
    xml: str
    pan_body_id: int
    pan_mocap_id: int
    pan_collision_geom_ids: np.ndarray
    particle_body_ids: np.ndarray
    particle_geom_ids: np.ndarray
    particle_joint_ids: np.ndarray
    particle_dof_addresses: np.ndarray
    particle_qpos_addresses: np.ndarray
    pan: PanModel
    particles: ParticleModelSpec
    metadata: dict[str, Any]


class ModelBuilder:
    """mapping config로 kinematic-pan + sphere-particle MuJoCo model을 만든다."""

    def __init__(
        self,
        config: Mapping[str, Any] | Any,
        *,
        base_directory: str | Path | None = None,
    ) -> None:
        self.config = _mapping(config)
        self.pan = PanModel.from_config(config, base_directory=base_directory)

    def build_xml(
        self,
        particles: ParticleModelSpec | Any,
        *,
        positions_are_pan_local: bool | None = None,
    ) -> str:
        """MJCF XML을 생성한다. 이 단계는 MuJoCo import 없이 검사 가능하다."""

        particle_spec = ParticleModelSpec.from_value(particles)
        simulation = _mapping(self.config.get("simulation", {}))
        particles_config = _mapping(self.config.get("particles", {}))
        render_config = _mapping(self.config.get("render", {}))

        if positions_are_pan_local is None:
            frame = str(
                particles_config.get(
                    "initial_position_frame",
                    particles_config.get("positions_frame", "pan"),
                )
            ).lower()
            positions_are_pan_local = frame in {"pan", "pan_local", "wok", "wok_frame"}
        positions_world = particle_spec.positions_m.copy()
        if positions_are_pan_local:
            pan_position, pan_quaternion = self.pan.initial_pose
            positions_world = (
                positions_world @ _quaternion_to_matrix(pan_quaternion).T + pan_position
            )

        timestep = float(simulation.get("timestep_s", simulation.get("timestep", 0.002)))
        if not np.isfinite(timestep) or timestep <= 0.0:
            raise ModelBuildError("simulation.timestep_s는 유한한 양수여야 합니다.")
        gravity = np.asarray(simulation.get("gravity_m_s2", (0.0, 0.0, -9.81)), dtype=float)
        if gravity.shape != (3,) or not np.isfinite(gravity).all():
            raise ModelBuildError("simulation.gravity_m_s2는 유한한 길이 3 배열이어야 합니다.")

        root = ET.Element("mujoco", {"model": "wok_mixing"})
        ET.SubElement(
            root,
            "compiler",
            {
                "angle": "radian",
                "coordinate": "local",
                "autolimits": "true",
                "inertiafromgeom": "true",
            },
        )
        ET.SubElement(
            root,
            "option",
            {
                "timestep": f"{timestep:.12g}",
                "gravity": _numbers(gravity),
                "integrator": str(simulation.get("integrator", "implicitfast")),
                "cone": str(simulation.get("cone", "elliptic")),
                "iterations": str(int(simulation.get("solver_iterations", 80))),
                "noslip_iterations": str(int(simulation.get("noslip_iterations", 2))),
            },
        )
        ET.SubElement(
            root,
            "size",
            {
                "njmax": str(max(500, len(particle_spec.radii_m) * 30)),
                "nconmax": str(max(200, len(particle_spec.radii_m) * 20)),
            },
        )
        visual = ET.SubElement(root, "visual")
        ET.SubElement(
            visual,
            "global",
            {
                "offwidth": str(int(render_config.get("width", 640))),
                "offheight": str(int(render_config.get("height", 480))),
            },
        )
        ET.SubElement(
            visual,
            "quality",
            {"shadowsize": str(int(render_config.get("shadow_size", 2048)))},
        )

        asset = ET.SubElement(root, "asset")
        ET.SubElement(
            asset,
            "material",
            {
                "name": "pan_visual_material",
                "rgba": _numbers(self.pan.visual_rgba),
                "metallic": "0.35",
                "roughness": "0.42",
            },
        )
        pan_asset_info = None
        if self.pan.stl_path is not None:
            # 실제로 읽어 bbox/형식 오류를 XML compile 전에 명확하게 보고한다.
            pan_asset_info = self.pan.inspect_asset()
            ET.SubElement(
                asset,
                "mesh",
                {
                    "name": "pan_visual_mesh",
                    "file": str(self.pan.stl_path),
                    "scale": _numbers(self.pan.stl_scale),
                },
            )

        worldbody = ET.SubElement(root, "worldbody")
        ET.SubElement(
            worldbody,
            "light",
            {
                "name": "key_light",
                "pos": "0 -0.25 0.55",
                "dir": "0 0.35 -1",
                "diffuse": "0.9 0.9 0.9",
                "castshadow": "true",
            },
        )
        ET.SubElement(
            worldbody,
            "camera",
            {
                "name": "overview",
                "mode": "targetbody",
                "target": "pan",
                "pos": _numbers(render_config.get("camera_position_m", (0.0, -0.42, 0.30))),
                "fovy": str(float(render_config.get("fovy_deg", 48.0))),
            },
        )
        if bool(simulation.get("ground_enabled", True)):
            ground_z = float(simulation.get("ground_z_m", -0.35))
            ET.SubElement(
                worldbody,
                "geom",
                {
                    "name": "spill_ground",
                    "type": "plane",
                    "pos": f"0 0 {ground_z:.12g}",
                    "size": "1 1 0.02",
                    "rgba": "0.12 0.13 0.15 1",
                    "friction": "0.8 0.01 0.001",
                    "contype": "2",
                    "conaffinity": "1",
                },
            )

        pan_position, pan_quaternion = self.pan.initial_pose
        pan_body = ET.SubElement(
            worldbody,
            "body",
            {
                "name": "pan",
                "mocap": "true",
                "pos": _numbers(pan_position),
                "quat": _numbers(pan_quaternion),
            },
        )
        if self.pan.stl_path is not None:
            if self.pan.visual_offset_m is not None:
                visual_offset = np.asarray(self.pan.visual_offset_m, dtype=float)
            elif pan_asset_info is not None:
                # 사용자가 scale을 명시한 뒤에는 mesh 좌표 원점을 추측하지
                # 않고, compound proxy의 bottom/중심에 visual만 정렬한다.
                bounds = pan_asset_info.bounds_m
                visual_offset = np.array(
                    [
                        -0.5 * (bounds[0, 0] + bounds[1, 0]),
                        -0.5 * (bounds[0, 1] + bounds[1, 1]),
                        self.pan.proxy.bottom_z_m - bounds[0, 2],
                    ]
                )
            else:
                visual_offset = np.zeros(3, dtype=float)
            ET.SubElement(
                pan_body,
                "geom",
                {
                    "name": "pan_visual",
                    "type": "mesh",
                    "mesh": "pan_visual_mesh",
                    "material": "pan_visual_material",
                    "pos": _numbers(visual_offset),
                    "contype": "0",
                    "conaffinity": "0",
                    "group": "2",
                    "mass": "0",
                },
            )
        self._add_compound_collision(pan_body)

        friction = particles_config.get("friction", (0.65, 0.01, 0.001))
        if np.isscalar(friction):
            friction = (float(friction), 0.01, 0.001)
        friction_array = np.asarray(friction, dtype=float)
        if friction_array.shape != (3,) or np.any(friction_array < 0.0):
            raise ModelBuildError("particles.friction은 음수가 아닌 길이 3 배열이어야 합니다.")
        restitution = float(particles_config.get("restitution", 0.08))
        if not 0.0 <= restitution < 1.0:
            raise ModelBuildError("particles.restitution은 [0, 1) 범위여야 합니다.")
        # MuJoCo는 restitution을 직접 받지 않으므로 solref time constant로
        # 안정적인 저반발 접촉을 설정한다.
        contact_time = max(0.0025, 0.012 * (1.0 - restitution))
        color_a = np.asarray(particles_config.get("color_a_rgba", (0.95, 0.38, 0.10, 1.0)))
        color_b = np.asarray(particles_config.get("color_b_rgba", (0.15, 0.62, 0.94, 1.0)))
        has_per_particle_contact = any(
            value is not None
            for value in (
                particle_spec.restitution_coefficients,
                particle_spec.contact_time_constants_s,
                particle_spec.contact_damping_ratios,
                particle_spec.frictions,
            )
        )
        for index in range(len(particle_spec.radii_m)):
            mass = float(particle_spec.masses_kg[index])
            position = positions_world[index]
            geom_type = str(particle_spec.geom_types[index])
            sizes = particle_spec.sizes_m[index]
            quaternion = particle_spec.quaternions_wxyz[index]
            body = ET.SubElement(
                worldbody,
                "body",
                {
                    "name": f"particle_{index:04d}",
                    "pos": _numbers(position),
                    "quat": _numbers(quaternion),
                },
            )
            ET.SubElement(
                body,
                "joint",
                {
                    "name": f"particle_joint_{index:04d}",
                    "type": "free",
                    # free joint의 회전 관성은 매우 작다. config의 1/s
                    # damping을 joint viscous torque로 직접 넣으면 수치적으로
                    # 폭주하므로 simulator에서 qvel exponential damping으로
                    # 적용한다.
                    "damping": "0",
                },
            )
            fraction = index / max(len(particle_spec.radii_m) - 1, 1)
            rgba = color_a * (1.0 - fraction) + color_b * fraction
            particle_friction = (
                friction_array
                if particle_spec.frictions is None
                else particle_spec.frictions[index]
            )
            particle_contact_time = (
                contact_time
                if particle_spec.contact_time_constants_s is None
                else float(particle_spec.contact_time_constants_s[index])
            )
            if particle_spec.contact_damping_ratios is not None:
                particle_contact_damping = float(particle_spec.contact_damping_ratios[index])
            elif particle_spec.restitution_coefficients is not None:
                particle_restitution = float(particle_spec.restitution_coefficients[index])
                particle_contact_damping = (
                    1.0
                    if particle_restitution <= 0.0
                    else damping_ratio_from_restitution(particle_restitution)
                )
            else:
                particle_contact_damping = 1.0
            size_text = f"{float(sizes[0]):.12g}" if geom_type == "sphere" else _numbers(sizes)
            geom_attributes = {
                "name": f"particle_geom_{index:04d}",
                "type": geom_type,
                "size": size_text,
                "mass": f"{mass:.12g}",
                "rgba": _numbers(rgba),
                "friction": _numbers(particle_friction),
                "solref": (f"{particle_contact_time:.12g} {particle_contact_damping:.12g}"),
                "solimp": "0.9 0.95 0.001",
                "condim": "4",
                "contype": "1",
                "conaffinity": "3",
            }
            if has_per_particle_contact:
                # pan proxy(priority=0)보다 높여 각 입자의 solref/friction이
                # particle-pan contact pair에 선택되게 한다.
                geom_attributes["priority"] = "1"
            ET.SubElement(
                body,
                "geom",
                geom_attributes,
            )

        return ET.tostring(root, encoding="unicode")

    def _add_compound_collision(self, pan_body: ET.Element) -> None:
        proxy = self.pan.proxy
        common = {
            "friction": _numbers(self.pan.friction),
            "solref": "0.006 1",
            "solimp": "0.92 0.98 0.001",
            "condim": "4",
            "contype": "1",
            "conaffinity": "1",
            "group": "3",
            "rgba": _numbers(self.pan.proxy_rgba),
        }
        bottom_position_z = proxy.bottom_z_m - proxy.bottom_thickness_m * 0.5
        ET.SubElement(
            pan_body,
            "geom",
            {
                **common,
                "name": "pan_collision_bottom",
                "type": "cylinder",
                "pos": f"0 0 {bottom_position_z:.12g}",
                "size": _numbers((proxy.bottom_radius_m, proxy.bottom_thickness_m * 0.5)),
            },
        )

        radial_change = proxy.inner_radius_m - proxy.bottom_radius_m
        vertical_change = proxy.rim_z_m - proxy.bottom_z_m
        wall_length = math.hypot(radial_change, vertical_change)
        mid_radius = 0.5 * (proxy.inner_radius_m + proxy.bottom_radius_m)
        mid_z = 0.5 * (proxy.rim_z_m + proxy.bottom_z_m)
        tangent_half_length = (
            proxy.inner_radius_m * math.tan(math.pi / proxy.wall_segments) * proxy.segment_overlap
        )
        for index in range(proxy.wall_segments):
            theta = 2.0 * math.pi * index / proxy.wall_segments
            radial = np.array([math.cos(theta), math.sin(theta), 0.0])
            tangent = np.array([-math.sin(theta), math.cos(theta), 0.0])
            wall_axis = (
                np.array(
                    [
                        radial_change * math.cos(theta),
                        radial_change * math.sin(theta),
                        vertical_change,
                    ]
                )
                / wall_length
            )
            normal = np.cross(wall_axis, tangent)
            rotation = np.column_stack((tangent, normal, wall_axis))
            quaternion = _matrix_to_quaternion(rotation)
            center = radial * mid_radius
            center[2] = mid_z
            ET.SubElement(
                pan_body,
                "geom",
                {
                    **common,
                    "name": f"pan_collision_wall_{index:03d}",
                    "type": "box",
                    "pos": _numbers(center),
                    "quat": _numbers(quaternion),
                    "size": _numbers(
                        (
                            tangent_half_length,
                            proxy.wall_thickness_m * 0.5,
                            wall_length * 0.5 * proxy.segment_overlap,
                        )
                    ),
                },
            )

            theta_next = 2.0 * math.pi * (index + 1) / proxy.wall_segments
            rim_center_radius = 0.5 * (proxy.inner_radius_m + proxy.rim_radius_m)
            start = np.array(
                [
                    rim_center_radius * math.cos(theta),
                    rim_center_radius * math.sin(theta),
                    proxy.rim_z_m,
                ]
            )
            end = np.array(
                [
                    rim_center_radius * math.cos(theta_next),
                    rim_center_radius * math.sin(theta_next),
                    proxy.rim_z_m,
                ]
            )
            ET.SubElement(
                pan_body,
                "geom",
                {
                    **common,
                    "name": f"pan_collision_rim_{index:03d}",
                    "type": "capsule",
                    "fromto": _numbers(np.concatenate((start, end))),
                    "size": f"{proxy.rim_tube_radius_m:.12g}",
                },
            )

    def build(
        self,
        particles: ParticleModelSpec | Any,
        *,
        positions_are_pan_local: bool | None = None,
    ) -> BuiltModel:
        """MJCF를 MuJoCo model/data로 compile한다."""

        particle_spec = ParticleModelSpec.from_value(particles)
        xml = self.build_xml(particle_spec, positions_are_pan_local=positions_are_pan_local)
        try:
            import mujoco
        except ImportError as exc:
            raise ModelBuildError(
                "MuJoCo Python 패키지가 설치되지 않았습니다. `pip install mujoco` 후 "
                "다시 실행하세요."
            ) from exc
        try:
            model = mujoco.MjModel.from_xml_string(xml)
            data = mujoco.MjData(model)
        except Exception as exc:
            raise ModelBuildError(f"MuJoCo MJCF compile에 실패했습니다: {exc}") from exc

        def object_id(object_type: Any, name: str) -> int:
            identifier = int(mujoco.mj_name2id(model, object_type, name))
            if identifier < 0:
                raise ModelBuildError(f"compile된 model에서 이름을 찾지 못했습니다: {name}")
            return identifier

        pan_body_id = object_id(mujoco.mjtObj.mjOBJ_BODY, "pan")
        pan_mocap_id = int(model.body_mocapid[pan_body_id])
        if pan_mocap_id < 0:
            raise ModelBuildError("pan body가 mocap body로 compile되지 않았습니다.")

        particle_body_ids = []
        particle_geom_ids = []
        particle_joint_ids = []
        for index in range(len(particle_spec.radii_m)):
            particle_body_ids.append(object_id(mujoco.mjtObj.mjOBJ_BODY, f"particle_{index:04d}"))
            particle_geom_ids.append(
                object_id(mujoco.mjtObj.mjOBJ_GEOM, f"particle_geom_{index:04d}")
            )
            particle_joint_ids.append(
                object_id(mujoco.mjtObj.mjOBJ_JOINT, f"particle_joint_{index:04d}")
            )

        pan_collision_geom_ids = [object_id(mujoco.mjtObj.mjOBJ_GEOM, "pan_collision_bottom")]
        for index in range(self.pan.proxy.wall_segments):
            pan_collision_geom_ids.append(
                object_id(mujoco.mjtObj.mjOBJ_GEOM, f"pan_collision_wall_{index:03d}")
            )
            pan_collision_geom_ids.append(
                object_id(mujoco.mjtObj.mjOBJ_GEOM, f"pan_collision_rim_{index:03d}")
            )

        particle_joint_ids_array = np.asarray(particle_joint_ids, dtype=int)
        asset_info = self.pan.inspect_asset()
        unique_species, species_counts = np.unique(
            particle_spec.species,
            return_counts=True,
        )
        unique_geom_types, geom_type_counts = np.unique(
            particle_spec.geom_types,
            return_counts=True,
        )
        rebound_targets: dict[str, dict[str, float]] = {}
        if (
            particle_spec.reference_drop_heights_m is not None
            and particle_spec.target_rebound_heights_m is not None
            and particle_spec.restitution_coefficients is not None
        ):
            for species_name in unique_species:
                first_index = int(np.flatnonzero(particle_spec.species == species_name)[0])
                target_record = {
                    "drop_height_m": float(particle_spec.reference_drop_heights_m[first_index]),
                    "target_rebound_height_m": float(
                        particle_spec.target_rebound_heights_m[first_index]
                    ),
                    "restitution_coefficient": float(
                        particle_spec.restitution_coefficients[first_index]
                    ),
                }
                if particle_spec.contact_time_constants_s is not None:
                    target_record["contact_time_constant_s"] = float(
                        particle_spec.contact_time_constants_s[first_index]
                    )
                if particle_spec.contact_damping_ratios is not None:
                    target_record["contact_damping_ratio"] = float(
                        particle_spec.contact_damping_ratios[first_index]
                    )
                rebound_targets[str(species_name)] = target_record
        return BuiltModel(
            model=model,
            data=data,
            xml=xml,
            pan_body_id=pan_body_id,
            pan_mocap_id=pan_mocap_id,
            pan_collision_geom_ids=np.asarray(pan_collision_geom_ids, dtype=int),
            particle_body_ids=np.asarray(particle_body_ids, dtype=int),
            particle_geom_ids=np.asarray(particle_geom_ids, dtype=int),
            particle_joint_ids=particle_joint_ids_array,
            particle_dof_addresses=np.asarray(
                model.jnt_dofadr[particle_joint_ids_array], dtype=int
            ),
            particle_qpos_addresses=np.asarray(
                model.jnt_qposadr[particle_joint_ids_array], dtype=int
            ),
            pan=self.pan,
            particles=particle_spec,
            metadata={
                "pan_mass_kg_metadata_only": self.pan.mass_kg,
                "pan_kinematic": True,
                "pan_collision_mode": "primitive_compound",
                "pan_asset": None if asset_info is None else asset_info.as_dict(),
                "particle_count": len(particle_spec.radii_m),
                "particle_species_counts": {
                    str(name): int(count)
                    for name, count in zip(
                        unique_species,
                        species_counts,
                        strict=True,
                    )
                },
                "particle_geom_type_counts": {
                    str(name): int(count)
                    for name, count in zip(
                        unique_geom_types,
                        geom_type_counts,
                        strict=True,
                    )
                },
                "particle_rebound_targets": rebound_targets,
            },
        )


__all__ = [
    "BuiltModel",
    "ModelBuildError",
    "ModelBuilder",
    "ParticleModelSpec",
]
