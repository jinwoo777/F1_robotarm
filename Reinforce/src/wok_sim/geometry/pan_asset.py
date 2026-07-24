"""STL 자산을 물리 모델과 독립적으로 검사한다."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


class PanAssetError(RuntimeError):
    """pan STL을 안전하게 사용할 수 없을 때 발생한다."""


@dataclass(frozen=True)
class PanAssetReport:
    """STL 검사 결과. bounds/extents의 단위는 meter다."""

    path: str | None
    stl_scale: float
    mesh_type: str
    vertices: int
    faces: int
    bounds_m: list[list[float]]
    extents_m: list[float]
    watertight: bool | None
    procedural_demo: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화 가능한 mapping으로 반환한다."""

        return asdict(self)


def inspect_pan_asset(pan_config: Mapping[str, Any]) -> PanAssetReport:
    """STL 경로, scale, finite bounds, watertight 여부를 검사한다.

    procedural mode는 반드시 명시해야 하며 잘못된 실제 경로를 조용히 대체하지
    않는다.
    """

    scale = float(pan_config.get("stl_scale", 0.0))
    if not np.isfinite(scale) or scale <= 0:
        raise PanAssetError(f"유효하지 않은 pan.stl_scale: {scale!r}")
    if bool(pan_config.get("use_procedural_demo", False)):
        proxy = pan_config.get("collision_proxy", {})
        radius = float(proxy.get("rim_radius_m", proxy.get("inner_radius_m", 0.11)))
        height = float(proxy.get("wall_height_m", 0.055))
        return PanAssetReport(
            path=None,
            stl_scale=scale,
            mesh_type="procedural_compound_proxy",
            vertices=0,
            faces=0,
            bounds_m=[[-radius, -radius, -height], [radius, radius, 0.0]],
            extents_m=[2.0 * radius, 2.0 * radius, height],
            watertight=None,
            procedural_demo=True,
        )

    raw_path = pan_config.get("stl_path")
    if not raw_path:
        raise PanAssetError(
            "pan.stl_path가 없습니다. 실제 STL을 지정하거나 테스트에서만 "
            "use_procedural_demo=true를 명시하세요."
        )
    path = Path(str(raw_path)).expanduser().resolve()
    if not path.is_file():
        raise PanAssetError(f"pan STL 파일을 찾을 수 없습니다: {path}")
    if path.suffix.lower() != ".stl":
        raise PanAssetError(f"pan 자산은 STL이어야 합니다: {path}")

    try:
        import trimesh
    except ImportError as exc:
        raise PanAssetError(
            "STL 검사에는 trimesh가 필요합니다. `pip install -e .`로 설치하세요."
        ) from exc

    try:
        loaded = trimesh.load(path, force="mesh", process=False)
    except Exception as exc:
        raise PanAssetError(f"STL 로딩 실패 ({path}): {exc}") from exc
    if not isinstance(loaded, trimesh.Trimesh) or loaded.vertices.size == 0:
        raise PanAssetError(f"유효한 triangle mesh가 아닙니다: {path}")
    vertices = np.asarray(loaded.vertices, dtype=float) * scale
    if not np.isfinite(vertices).all():
        raise PanAssetError(f"STL vertex에 NaN 또는 inf가 있습니다: {path}")
    bounds = np.stack((vertices.min(axis=0), vertices.max(axis=0)))
    extents = bounds[1] - bounds[0]
    if np.any(extents <= 0):
        raise PanAssetError(f"STL bounding box가 퇴화했습니다: {extents.tolist()}")
    return PanAssetReport(
        path=str(path),
        stl_scale=scale,
        mesh_type=type(loaded).__name__,
        vertices=int(len(loaded.vertices)),
        faces=int(len(loaded.faces)),
        bounds_m=bounds.tolist(),
        extents_m=extents.tolist(),
        watertight=bool(loaded.is_watertight),
        procedural_demo=False,
    )
