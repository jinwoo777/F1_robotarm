"""팬 로컬 좌표 기반 혼합도와 공간 분포 metric."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

LabelMode = Literal["quadrant", "left_right"]
GridBounds = tuple[float, float, float, float]


def _positions_2d(positions: ArrayLike, *, allow_empty: bool = False) -> NDArray[np.float64]:
    array = np.asarray(positions, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] not in (2, 3):
        raise ValueError("positions must have shape (N, 2) or (N, 3)")
    if not allow_empty and array.shape[0] == 0:
        raise ValueError("mixing metrics require at least one particle")
    if not np.all(np.isfinite(array)):
        raise ValueError("positions must contain only finite values")
    return np.asarray(array[:, :2], dtype=np.float64)


def _positive_grid_size(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _coerce_bounds(bounds: ArrayLike) -> GridBounds:
    raw = np.asarray(bounds, dtype=np.float64)
    if raw.shape == (2, 2):
        u_min, u_max = raw[0]
        v_min, v_max = raw[1]
    elif raw.shape == (4,):
        u_min, u_max, v_min, v_max = raw
    else:
        raise ValueError(
            "bounds must be (u_min, u_max, v_min, v_max) or ((u_min, u_max), (v_min, v_max))"
        )
    values = np.array([u_min, u_max, v_min, v_max], dtype=np.float64)
    if not np.all(np.isfinite(values)) or u_min >= u_max or v_min >= v_max:
        raise ValueError("bounds must contain finite, strictly increasing limits")
    return float(u_min), float(u_max), float(v_min), float(v_max)


def resolve_grid_bounds(
    positions: ArrayLike,
    *,
    bounds: ArrayLike | None = None,
    pan_radius_m: float | None = None,
) -> GridBounds:
    """명시 경계, 팬 반지름, 또는 데이터 범위 순으로 grid 경계를 결정한다."""

    points = _positions_2d(positions)
    if bounds is not None and pan_radius_m is not None:
        raise ValueError("provide either bounds or pan_radius_m, not both")
    if bounds is not None:
        return _coerce_bounds(bounds)
    if pan_radius_m is not None:
        radius = float(pan_radius_m)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("pan_radius_m must be a positive finite value")
        return -radius, radius, -radius, radius

    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    span = upper - lower
    # 한 축의 좌표가 모두 같아도 유효한 grid를 만든다.
    fallback_span = max(float(np.max(span)), 1.0)
    padding = np.where(span > 0.0, np.maximum(span * 1e-12, 1e-12), fallback_span * 0.5)
    return (
        float(lower[0] - padding[0]),
        float(upper[0] + padding[0]),
        float(lower[1] - padding[1]),
        float(upper[1] + padding[1]),
    )


def assign_initial_labels(
    pan_local_positions: ArrayLike,
    *,
    mode: LabelMode = "quadrant",
) -> NDArray[np.int64]:
    """초기 pan-local 위치에서 물리 영향이 없는 가상 재료 label을 만든다."""

    points = _positions_2d(pan_local_positions)
    if mode == "quadrant":
        # 축 위의 점도 항상 한 영역에 속하도록 >=를 사용한다.
        return (points[:, 0] >= 0.0).astype(np.int64) + 2 * (points[:, 1] >= 0.0).astype(np.int64)
    if mode == "left_right":
        # wok_frame의 +y가 좌측이므로 두 번째 바닥 좌표로 구분한다.
        return (points[:, 1] >= 0.0).astype(np.int64)
    raise ValueError(f"unsupported label mode: {mode!r}")


def size_group_labels(radii_m: ArrayLike) -> NDArray[np.int64]:
    """반지름 percentile rank로 small/medium/large(0/1/2)를 부여한다.

    quantile 경계값이 같은 경우에도 세 그룹이 가능한 한 균등해지도록
    stable rank를 사용한다. 이는 동일하거나 거의 동일한 크기 표본에서도
    빈 그룹 때문에 metric이 불안정해지는 것을 피한다.
    """

    radii = np.asarray(radii_m, dtype=np.float64)
    if radii.ndim != 1 or radii.size == 0:
        raise ValueError("radii_m must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("radii_m must contain only positive finite values")
    order = np.argsort(radii, kind="stable")
    labels = np.empty(radii.size, dtype=np.int64)
    ranks = np.arange(radii.size, dtype=np.int64)
    labels[order] = np.minimum((3 * ranks) // radii.size, 2)
    return labels


def _label_ids(labels: ArrayLike, expected_count: int) -> tuple[NDArray[np.int64], int]:
    raw = np.asarray(labels)
    if raw.ndim != 1 or raw.shape[0] != expected_count:
        raise ValueError("labels must be one-dimensional with one value per particle")
    # np.unique는 숫자/문자 label을 모두 안정적으로 정수 id에 매핑한다.
    try:
        _, inverse = np.unique(raw, return_inverse=True)
    except TypeError as error:
        raise ValueError("labels must contain mutually comparable scalar values") from error
    inverse = np.asarray(inverse, dtype=np.int64)
    return inverse, int(np.max(inverse) + 1)


def _entropy_from_counts(counts: NDArray[np.float64]) -> float:
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0
    probabilities = counts[counts > 0.0] / total
    return float(-np.sum(probabilities * np.log(probabilities)))


def grid_cell_indices(
    pan_local_positions: ArrayLike,
    *,
    grid_rows: int,
    grid_cols: int,
    bounds: ArrayLike | None = None,
    pan_radius_m: float | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.bool_], GridBounds]:
    """각 입자의 평탄화된 grid cell id와 경계 내부 mask를 반환한다."""

    points = _positions_2d(pan_local_positions)
    rows = _positive_grid_size("grid_rows", grid_rows)
    cols = _positive_grid_size("grid_cols", grid_cols)
    resolved = resolve_grid_bounds(points, bounds=bounds, pan_radius_m=pan_radius_m)
    u_min, u_max, v_min, v_max = resolved
    # upper edge는 마지막 cell에 포함한다.
    inside = (
        (points[:, 0] >= u_min)
        & (points[:, 0] <= u_max)
        & (points[:, 1] >= v_min)
        & (points[:, 1] <= v_max)
    )
    u_scaled = (points[:, 0] - u_min) / (u_max - u_min)
    v_scaled = (points[:, 1] - v_min) / (v_max - v_min)
    col_index = np.clip(np.floor(u_scaled * cols).astype(np.int64), 0, cols - 1)
    row_index = np.clip(np.floor(v_scaled * rows).astype(np.int64), 0, rows - 1)
    cell_index = row_index * cols + col_index
    cell_index[~inside] = -1
    return cell_index, inside, resolved


def mixing_score(
    pan_local_positions: ArrayLike,
    labels: ArrayLike,
    *,
    grid_rows: int = 4,
    grid_cols: int = 4,
    bounds: ArrayLike | None = None,
    pan_radius_m: float | None = None,
    minimum_particles_per_cell: int = 1,
) -> float:
    """정규화 조건부 엔트로피 ``H(C|B) / H(C)``를 계산한다.

    빈 cell은 합에 참여하지 않는다. grid 밖 입자와 최소 개수보다 작은
    cell의 가중치는 0으로 남기고, 분모와 가중치는 항상 전체 초기 label
    population을 사용한다. 따라서 유실된 입자만 제외해 점수가 인위적으로
    높아지지 않으며 모든 입자가 grid 밖이면 0.0이다. 단일 label만 존재해
    ``H(C)=0``이면 grid 안에 남은 입자 비율을 반환한다.
    """

    points = _positions_2d(pan_local_positions)
    min_count = _positive_grid_size("minimum_particles_per_cell", minimum_particles_per_cell)
    label_ids, number_of_labels = _label_ids(labels, points.shape[0])
    cell_ids, inside, _ = grid_cell_indices(
        points,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        bounds=bounds,
        pan_radius_m=pan_radius_m,
    )
    total_particles = points.shape[0]
    if not np.any(inside):
        return 0.0
    global_counts = np.bincount(label_ids, minlength=number_of_labels).astype(np.float64)
    marginal_entropy = _entropy_from_counts(global_counts)
    if marginal_entropy <= np.finfo(np.float64).eps:
        return float(np.count_nonzero(inside) / total_particles)

    conditional_entropy = 0.0
    for cell_id in np.unique(cell_ids[inside]):
        in_cell = inside & (cell_ids == cell_id)
        cell_count = int(np.count_nonzero(in_cell))
        if cell_count < min_count:
            continue
        counts = np.bincount(label_ids[in_cell], minlength=number_of_labels).astype(np.float64)
        conditional_entropy += (cell_count / total_particles) * _entropy_from_counts(counts)

    score = conditional_entropy / marginal_entropy
    # 반올림 때문에 1을 수 ulp 넘는 경우를 포함해 유효 범위를 보장한다.
    return float(np.clip(score, 0.0, 1.0))


conditional_entropy_mixing_score = mixing_score
compute_mixing_score = mixing_score


def occupancy_ratio(
    pan_local_positions: ArrayLike,
    *,
    grid_rows: int = 4,
    grid_cols: int = 4,
    bounds: ArrayLike | None = None,
    pan_radius_m: float | None = None,
    minimum_particles_per_cell: int = 1,
) -> float:
    """전체 직사각 grid 중 최소 입자 수를 충족한 cell 비율."""

    points = _positions_2d(pan_local_positions)
    rows = _positive_grid_size("grid_rows", grid_rows)
    cols = _positive_grid_size("grid_cols", grid_cols)
    min_count = _positive_grid_size("minimum_particles_per_cell", minimum_particles_per_cell)
    cell_ids, inside, _ = grid_cell_indices(
        points,
        grid_rows=rows,
        grid_cols=cols,
        bounds=bounds,
        pan_radius_m=pan_radius_m,
    )
    if not np.any(inside):
        return 0.0
    counts = np.bincount(cell_ids[inside], minlength=rows * cols)
    occupied = int(np.count_nonzero(counts >= min_count))
    return float(occupied / (rows * cols))


def spatial_dispersion(
    pan_local_positions: ArrayLike,
    *,
    pan_radius_m: float | None = None,
) -> float:
    """입자 centroid로부터의 RMS 거리.

    ``pan_radius_m``을 주면 반지름으로 나눈 무차원 값을 반환하고, 생략하면
    meter 단위의 원시 RMS 거리를 반환한다.
    """

    points = _positions_2d(pan_local_positions)
    centered = points - np.mean(points, axis=0, keepdims=True)
    dispersion = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    if pan_radius_m is None:
        return dispersion
    radius = float(pan_radius_m)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("pan_radius_m must be a positive finite value")
    return dispersion / radius


def particle_radial_distribution(
    pan_local_positions: ArrayLike,
    *,
    radial_bins: int | Sequence[float] = 5,
    pan_radius_m: float | None = None,
    center_uv_m: ArrayLike = (0.0, 0.0),
) -> dict[str, NDArray[np.float64] | NDArray[np.int64]]:
    """팬 중심으로부터의 반경 histogram을 반환한다."""

    points = _positions_2d(pan_local_positions)
    center = np.asarray(center_uv_m, dtype=np.float64)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("center_uv_m must contain two finite coordinates")
    radii = np.linalg.norm(points - center[None, :], axis=1)
    if isinstance(radial_bins, (int, np.integer)):
        bins = _positive_grid_size("radial_bins", int(radial_bins))
        if pan_radius_m is None:
            upper = max(float(np.max(radii)), np.finfo(np.float64).eps)
        else:
            upper = float(pan_radius_m)
            if not np.isfinite(upper) or upper <= 0.0:
                raise ValueError("pan_radius_m must be a positive finite value")
        edges = np.linspace(0.0, upper, bins + 1)
    else:
        edges = np.asarray(radial_bins, dtype=np.float64)
        if (
            edges.ndim != 1
            or edges.size < 2
            or not np.all(np.isfinite(edges))
            or np.any(np.diff(edges) <= 0.0)
        ):
            raise ValueError("radial_bins edges must be finite and strictly increasing")
    counts, edges = np.histogram(radii, bins=edges)
    fractions = counts.astype(np.float64) / points.shape[0]
    return {
        "bin_edges_m": np.asarray(edges, dtype=np.float64),
        "counts": np.asarray(counts, dtype=np.int64),
        "fractions": fractions,
    }


def compute_mixing_metrics(
    initial_pan_local_positions: ArrayLike,
    final_pan_local_positions: ArrayLike,
    *,
    initial_region_labels: ArrayLike | None = None,
    label_mode: LabelMode = "quadrant",
    radii_m: ArrayLike | None = None,
    grid_rows: int = 4,
    grid_cols: int = 4,
    bounds: ArrayLike | None = None,
    pan_radius_m: float | None = None,
    minimum_particles_per_cell: int = 1,
    radial_bins: int | Sequence[float] = 5,
) -> dict[str, Any]:
    """초기/최종 region 및 size 혼합도와 보조 공간 metric을 계산한다."""

    initial = _positions_2d(initial_pan_local_positions)
    final = _positions_2d(final_pan_local_positions)
    if initial.shape != final.shape:
        raise ValueError("initial and final positions must have the same shape")
    labels = (
        assign_initial_labels(initial, mode=label_mode)
        if initial_region_labels is None
        else np.asarray(initial_region_labels)
    )
    _label_ids(labels, initial.shape[0])

    if bounds is None and pan_radius_m is None:
        common_bounds: ArrayLike = resolve_grid_bounds(np.concatenate((initial, final), axis=0))
        common_pan_radius = None
    else:
        common_bounds = bounds
        common_pan_radius = pan_radius_m

    common_options = {
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "bounds": common_bounds,
        "pan_radius_m": common_pan_radius,
        "minimum_particles_per_cell": minimum_particles_per_cell,
    }
    initial_score = mixing_score(initial, labels, **common_options)
    final_score = mixing_score(final, labels, **common_options)
    result: dict[str, Any] = {
        "initial_mixing_score": initial_score,
        "final_mixing_score": final_score,
        "mixing_improvement": final_score - initial_score,
        "delta_mix": final_score - initial_score,
        "initial_spatial_dispersion": spatial_dispersion(initial, pan_radius_m=pan_radius_m),
        "final_spatial_dispersion": spatial_dispersion(final, pan_radius_m=pan_radius_m),
        "initial_occupancy_ratio": occupancy_ratio(initial, **common_options),
        "final_occupancy_ratio": occupancy_ratio(final, **common_options),
        "initial_radial_distribution": particle_radial_distribution(
            initial,
            radial_bins=radial_bins,
            pan_radius_m=pan_radius_m,
        ),
        "final_radial_distribution": particle_radial_distribution(
            final,
            radial_bins=radial_bins,
            pan_radius_m=pan_radius_m,
        ),
    }

    if radii_m is not None:
        sizes = np.asarray(radii_m, dtype=np.float64)
        if sizes.shape != (initial.shape[0],):
            raise ValueError("radii_m must contain one radius per particle")
        size_labels = size_group_labels(sizes)
        initial_size_score = mixing_score(initial, size_labels, **common_options)
        final_size_score = mixing_score(final, size_labels, **common_options)
        result.update(
            {
                "initial_size_mixing_score": initial_size_score,
                "final_size_mixing_score": final_size_score,
                "size_mixing_improvement": final_size_score - initial_size_score,
                "size_group_labels": size_labels,
            }
        )
    else:
        result.update(
            {
                "initial_size_mixing_score": None,
                "final_size_mixing_score": None,
                "size_mixing_improvement": None,
                "size_group_labels": None,
            }
        )
    return result


__all__ = [
    "assign_initial_labels",
    "compute_mixing_metrics",
    "compute_mixing_score",
    "conditional_entropy_mixing_score",
    "grid_cell_indices",
    "mixing_score",
    "occupancy_ratio",
    "particle_radial_distribution",
    "resolve_grid_bounds",
    "size_group_labels",
    "spatial_dispersion",
]
