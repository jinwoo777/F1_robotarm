"""후라이팬 visual asset과 compound collision proxy 정의.

STL mesh는 시각화에만 사용한다. MuJoCo가 mesh collision에 사용하는 단일
convex hull은 오목한 웍 내부를 표현할 수 없으므로, 충돌 형상은 원형 바닥,
기울어진 벽 segment, rim capsule의 조합으로 별도 생성한다.
"""

from __future__ import annotations

import struct
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from wok_sim.geometry.transforms import (
    WokFrameContext,
    resolve_wok_frame_context,
    transform_to_pose,
)


class PanAssetError(ValueError):
    """팬 STL 또는 collision 설정이 잘못되었을 때 발생한다."""


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"설정은 mapping이어야 합니다. 받은 타입: {type(value).__name__}")


def _triple(value: Any, *, name: str) -> tuple[float, float, float]:
    if np.isscalar(value):
        number = float(value)
        result = (number, number, number)
    else:
        array = np.asarray(value, dtype=float)
        if array.shape != (3,):
            raise PanAssetError(f"{name}은 scalar 또는 길이 3 배열이어야 합니다.")
        result = tuple(float(item) for item in array)
    if not np.isfinite(result).all() or min(result) <= 0.0:
        raise PanAssetError(f"{name}은 유한한 양수여야 합니다: {result}")
    return result


def _resolve_path(path_value: str | Path, base_directory: str | Path | None) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute() and base_directory is not None:
        path = Path(base_directory).expanduser() / path
    return path.resolve()


@dataclass(frozen=True)
class PanAssetInfo:
    """STL 검사 결과. bbox 값에는 ``stl_scale``이 적용되어 있다."""

    path: Path
    scale: tuple[float, float, float]
    bounds_m: np.ndarray
    extents_m: np.ndarray
    is_watertight: bool | None
    triangle_count: int

    def as_dict(self) -> dict[str, Any]:
        """직렬화 가능한 metadata를 반환한다."""

        return {
            "path": str(self.path),
            "scale": list(self.scale),
            "bounds_m": self.bounds_m.tolist(),
            "extents_m": self.extents_m.tolist(),
            "is_watertight": self.is_watertight,
            "triangle_count": self.triangle_count,
        }


@dataclass(frozen=True)
class CollisionProxyConfig:
    """오목 팬을 근사하는 configurable primitive compound."""

    inner_radius_m: float = 0.112
    bottom_radius_m: float = 0.042
    bottom_z_m: float = 0.0
    rim_z_m: float = 0.060
    bottom_thickness_m: float = 0.004
    wall_thickness_m: float = 0.004
    wall_segments: int = 20
    rim_radius_m: float = 0.116
    rim_tube_radius_m: float = 0.003
    segment_overlap: float = 1.08

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> CollisionProxyConfig:
        """중첩 mapping에서 proxy 설정을 읽고 물리적으로 유효한지 검사한다."""

        raw = {} if value is None else dict(_mapping(value))
        aliases = {
            "radius_m": "inner_radius_m",
            "pan_radius_m": "inner_radius_m",
            "bottom_radius": "bottom_radius_m",
            "bottom_z": "bottom_z_m",
            "rim_height_m": "rim_z_m",
            "segments": "wall_segments",
            "thickness_m": "wall_thickness_m",
        }
        for old, new in aliases.items():
            if old in raw and new not in raw:
                raw[new] = raw[old]
        if "wall_height_m" in raw and "rim_z_m" not in raw:
            raw["rim_z_m"] = float(raw.get("bottom_z_m", 0.0)) + float(raw["wall_height_m"])
        # 기존 설정에서 rim_radius_m은 웍 중심부터 rim 바깥쪽까지의
        # 반지름이다. capsule 자체의 tube 반지름과 구분한다.
        if "rim_tube_radius_m" not in raw:
            inner = float(raw.get("inner_radius_m", cls.inner_radius_m))
            outer = float(raw.get("rim_radius_m", inner + cls.rim_tube_radius_m))
            raw["rim_tube_radius_m"] = max(
                float(raw.get("wall_thickness_m", cls.wall_thickness_m)) * 0.5,
                (outer - inner) * 0.5,
                1.0e-4,
            )
        accepted = {field.name for field in cls.__dataclass_fields__.values()}
        config = cls(**{key: raw[key] for key in accepted if key in raw})
        numbers = np.asarray(
            [
                config.inner_radius_m,
                config.bottom_radius_m,
                config.bottom_z_m,
                config.rim_z_m,
                config.bottom_thickness_m,
                config.wall_thickness_m,
                config.rim_radius_m,
                config.rim_tube_radius_m,
                config.segment_overlap,
            ],
            dtype=float,
        )
        if not np.isfinite(numbers).all():
            raise PanAssetError("collision proxy 값은 모두 유한해야 합니다.")
        if config.inner_radius_m <= config.bottom_radius_m:
            raise PanAssetError("inner_radius_m은 bottom_radius_m보다 커야 합니다.")
        if config.bottom_radius_m <= 0.0:
            raise PanAssetError("bottom_radius_m은 양수여야 합니다.")
        if config.rim_z_m <= config.bottom_z_m:
            raise PanAssetError("rim_z_m은 bottom_z_m보다 높아야 합니다.")
        if (
            min(
                config.bottom_thickness_m,
                config.wall_thickness_m,
                config.rim_radius_m,
                config.rim_tube_radius_m,
                config.segment_overlap,
            )
            <= 0.0
        ):
            raise PanAssetError("proxy 두께, rim 반지름, overlap은 양수여야 합니다.")
        if config.wall_segments < 8:
            raise PanAssetError("wall_segments는 안정적인 폐곡면을 위해 8 이상이어야 합니다.")
        if config.rim_radius_m < config.inner_radius_m:
            raise PanAssetError("rim_radius_m은 inner_radius_m 이상이어야 합니다.")
        return config


@dataclass(frozen=True)
class PanModel:
    """visual STL과 kinematic pan collision proxy 설정."""

    stl_path: Path | None
    stl_scale: tuple[float, float, float]
    mass_kg: float
    initial_position_m: tuple[float, float, float]
    initial_quaternion_wxyz: tuple[float, float, float, float]
    proxy: CollisionProxyConfig
    frame_context: WokFrameContext
    use_procedural_demo: bool = False
    visual_offset_m: tuple[float, float, float] | None = None
    friction: tuple[float, float, float] = (0.8, 0.01, 0.001)
    visual_rgba: tuple[float, float, float, float] = (0.55, 0.57, 0.60, 1.0)
    proxy_rgba: tuple[float, float, float, float] = (0.32, 0.34, 0.37, 0.35)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | Any,
        *,
        base_directory: str | Path | None = None,
    ) -> PanModel:
        """전체 config 또는 ``pan`` section에서 모델 설정을 만든다.

        STL이 없을 때 procedural pan으로 조용히 대체하지 않는다.
        ``use_procedural_demo: true``를 사용자가 명시해야만 mesh 없이 동작한다.
        """

        root = _mapping(config)
        pan = _mapping(root.get("pan", root))
        if base_directory is None:
            base_directory = root.get("_config_dir") or root.get("config_directory")

        use_demo = bool(pan.get("use_procedural_demo", pan.get("procedural_demo", False)))
        path_value = pan.get("stl_path")
        stl_path: Path | None = None
        if path_value not in (None, ""):
            stl_path = _resolve_path(path_value, base_directory)
            if not stl_path.is_file():
                if use_demo:
                    warnings.warn(
                        f"STL을 찾지 못해 명시적으로 허용된 procedural demo pan을 사용합니다: "
                        f"{stl_path}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    stl_path = None
                else:
                    raise PanAssetError(
                        f"팬 STL 파일을 찾을 수 없습니다: {stl_path}. "
                        "경로/scale을 확인하거나 테스트에 한해 "
                        "pan.use_procedural_demo=true를 명시하세요."
                    )
        elif not use_demo:
            raise PanAssetError(
                "pan.stl_path가 필요합니다. 테스트용 procedural pan을 쓰려면 "
                "pan.use_procedural_demo=true를 명시하세요."
            )

        scale = _triple(pan.get("stl_scale", 1.0), name="pan.stl_scale")
        mass_kg = float(pan.get("mass_kg", 0.700))
        if not np.isfinite(mass_kg) or mass_kg <= 0.0:
            raise PanAssetError("pan.mass_kg은 유한한 양수여야 합니다.")

        frame_config = root if "pan" in root else {"pan": pan}
        try:
            frame_context = resolve_wok_frame_context(frame_config)
            position_array, _, quaternion_array = transform_to_pose(frame_context.T_base_pan0)
        except ValueError as exc:
            raise PanAssetError(f"팬 초기 frame 설정이 유효하지 않습니다: {exc}") from exc

        collision_mode = str(pan.get("collision_mode", "compound")).lower()
        if collision_mode not in {
            "compound",
            "compound_proxy",
            "primitive_compound",
            "segments",
        }:
            if collision_mode in {"mesh", "convex", "convex_hull"}:
                raise PanAssetError(
                    "오목한 팬에 단일 mesh/convex hull collision은 지원하지 않습니다. "
                    "collision_mode='compound'를 사용하세요."
                )
            raise PanAssetError(f"지원하지 않는 pan.collision_mode: {collision_mode}")
        proxy_raw = pan.get("collision_proxy", pan.get("proxy", {}))
        proxy = CollisionProxyConfig.from_mapping(proxy_raw)

        friction_array = np.asarray(pan.get("friction", (0.8, 0.01, 0.001)), dtype=float)
        if friction_array.shape == (1,):
            friction_array = np.array([friction_array[0], 0.01, 0.001])
        if friction_array.shape != (3,) or np.any(friction_array < 0.0):
            raise PanAssetError("pan.friction은 음수가 아닌 길이 3 배열이어야 합니다.")

        visual_offset_value = pan.get("visual_offset_m", pan.get("mesh_offset_m"))
        visual_offset: tuple[float, float, float] | None = None
        if visual_offset_value is not None:
            visual_offset_array = np.asarray(visual_offset_value, dtype=float)
            if visual_offset_array.shape != (3,) or not np.isfinite(visual_offset_array).all():
                raise PanAssetError("pan.visual_offset_m은 유한한 길이 3 배열이어야 합니다.")
            visual_offset = tuple(float(item) for item in visual_offset_array)

        return cls(
            stl_path=stl_path,
            stl_scale=scale,
            mass_kg=mass_kg,
            initial_position_m=tuple(float(item) for item in position_array),
            initial_quaternion_wxyz=tuple(float(item) for item in quaternion_array),
            proxy=proxy,
            frame_context=frame_context,
            use_procedural_demo=use_demo,
            visual_offset_m=visual_offset,
            friction=tuple(float(item) for item in friction_array),
        )

    def inspect_asset(self) -> PanAssetInfo | None:
        """STL bbox, triangle 수, watertight 여부를 검사한다.

        ``trimesh``를 우선 사용하며, 설치 문제로 import할 수 없는 경우에도
        binary STL의 bbox와 triangle 수는 직접 읽는다. 이 fallback의
        watertight 결과는 ``None``이다.
        """

        if self.stl_path is None:
            return None
        return inspect_stl(self.stl_path, self.stl_scale)

    @property
    def initial_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """초기 position과 quaternion(wxyz)의 복사본을 반환한다."""

        return (
            np.asarray(self.initial_position_m, dtype=float),
            np.asarray(self.initial_quaternion_wxyz, dtype=float),
        )


def inspect_stl(
    path: str | Path,
    scale: float | Sequence[float] = 1.0,
) -> PanAssetInfo:
    """STL 파일을 실제로 로드해 bbox와 watertight metadata를 구한다."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise PanAssetError(f"팬 STL 파일을 찾을 수 없습니다: {resolved}")
    scale_xyz = _triple(scale, name="stl_scale")

    try:
        import trimesh
    except ImportError:
        return _inspect_binary_stl(resolved, scale_xyz)

    try:
        loaded = trimesh.load_mesh(resolved, file_type="stl", process=False)
        if isinstance(loaded, trimesh.Scene):
            if not loaded.geometry:
                raise PanAssetError(f"STL에 geometry가 없습니다: {resolved}")
            loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
        vertices = np.asarray(loaded.vertices, dtype=float)
        faces = np.asarray(loaded.faces)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
            raise PanAssetError(f"유효한 vertex가 없는 STL입니다: {resolved}")
        scaled_vertices = vertices * np.asarray(scale_xyz)
        bounds = np.stack([scaled_vertices.min(axis=0), scaled_vertices.max(axis=0)])
        return PanAssetInfo(
            path=resolved,
            scale=scale_xyz,
            bounds_m=bounds,
            extents_m=np.ptp(scaled_vertices, axis=0),
            is_watertight=bool(loaded.is_watertight),
            triangle_count=int(len(faces)),
        )
    except PanAssetError:
        raise
    except Exception as exc:
        raise PanAssetError(f"STL 로딩에 실패했습니다: {resolved}: {exc}") from exc


def _inspect_binary_stl(
    path: Path,
    scale: tuple[float, float, float],
) -> PanAssetInfo:
    try:
        with path.open("rb") as stream:
            stream.read(80)
            count_bytes = stream.read(4)
            if len(count_bytes) != 4:
                raise PanAssetError(f"잘린 STL header입니다: {path}")
            triangle_count = struct.unpack("<I", count_bytes)[0]
            payload = stream.read()
        expected = triangle_count * 50
        if len(payload) != expected:
            raise PanAssetError(
                "trimesh를 import할 수 없고 파일도 표준 binary STL이 아닙니다. "
                f"예상 payload={expected}, 실제={len(payload)}: {path}"
            )
        dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
        triangles = np.frombuffer(payload, dtype=dtype, count=triangle_count)
        vertices = triangles["vertices"].reshape(-1, 3).astype(float)
        if not np.isfinite(vertices).all():
            raise PanAssetError(f"STL vertex에 NaN/inf가 있습니다: {path}")
        vertices *= np.asarray(scale)
        bounds = np.stack([vertices.min(axis=0), vertices.max(axis=0)])
        return PanAssetInfo(
            path=path,
            scale=scale,
            bounds_m=bounds,
            extents_m=np.ptp(vertices, axis=0),
            is_watertight=None,
            triangle_count=int(triangle_count),
        )
    except PanAssetError:
        raise
    except Exception as exc:
        raise PanAssetError(f"binary STL 로딩에 실패했습니다: {path}: {exc}") from exc


__all__ = [
    "CollisionProxyConfig",
    "PanAssetError",
    "PanAssetInfo",
    "PanModel",
    "inspect_stl",
]
