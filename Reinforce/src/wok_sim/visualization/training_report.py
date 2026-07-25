"""Generate a compact PNG and machine-readable JSON report from ``episodes.csv``."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

MOVING_MEAN_WINDOW = 12
SUMMARY_WINDOW = 30
REQUIRED_COLUMNS = (
    "episode_id",
    "final_reward",
    "lift_angle_rad",
    "lifted_particle_count",
    "lift_reward",
    "spill_count",
    "spill_reward",
    "particle_count",
)


def _finite_float(row: dict[str, str], column: str, row_number: int) -> float:
    raw = row.get(column)
    if raw is None or not str(raw).strip():
        raise ValueError(f"row {row_number}: {column} is empty")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"row {row_number}: {column} is not numeric: {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"row {row_number}: {column} must be finite")
    return value


def _integer(row: dict[str, str], column: str, row_number: int) -> int:
    value = _finite_float(row, column, row_number)
    if not value.is_integer():
        raise ValueError(f"row {row_number}: {column} must be an integer")
    return int(value)


def _read_episodes(episodes_csv: Path) -> list[dict[str, int | float]]:
    if not episodes_csv.is_file():
        raise FileNotFoundError(f"episodes CSV not found: {episodes_csv}")

    with episodes_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("episodes CSV has no header")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"episodes CSV is missing required columns: {', '.join(missing)}")

        episodes: list[dict[str, int | float]] = []
        for row_number, row in enumerate(reader, start=2):
            if not any(str(value or "").strip() for value in row.values()):
                continue
            episodes.append(
                {
                    "episode_id": _integer(row, "episode_id", row_number),
                    "final_reward": _finite_float(row, "final_reward", row_number),
                    "lift_angle_deg": math.degrees(
                        _finite_float(row, "lift_angle_rad", row_number)
                    ),
                    "lifted_particle_count": _integer(row, "lifted_particle_count", row_number),
                    "lift_reward": _finite_float(row, "lift_reward", row_number),
                    "spill_count": _integer(row, "spill_count", row_number),
                    "spill_reward": _finite_float(row, "spill_reward", row_number),
                    "particle_count": _integer(row, "particle_count", row_number),
                }
            )

    if not episodes:
        raise ValueError("episodes CSV contains no data rows")
    episodes.sort(key=lambda item: int(item["episode_id"]))
    episode_ids = [int(item["episode_id"]) for item in episodes]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("episode_id values must be unique")
    return episodes


def _trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    result = np.empty_like(values, dtype=np.float64)
    for index in range(len(values)):
        start = max(0, index + 1 - window)
        result[index] = (cumulative[index + 1] - cumulative[start]) / (index + 1 - start)
    return result


def _reward_statistics(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "standard_deviation": float(np.std(values, ddof=0)),
    }


def _window_summary(
    episodes: list[dict[str, int | float]],
    indices: np.ndarray,
) -> dict[str, Any]:
    rewards = np.asarray([episodes[index]["final_reward"] for index in indices], dtype=float)
    lifted = np.asarray([episodes[index]["lifted_particle_count"] for index in indices], dtype=int)
    spilled = np.asarray([episodes[index]["spill_count"] for index in indices], dtype=int)
    return {
        "episode_count": int(len(indices)),
        "episode_id_start": int(episodes[int(indices[0])]["episode_id"]),
        "episode_id_end": int(episodes[int(indices[-1])]["episode_id"]),
        "final_reward": _reward_statistics(rewards),
        "lift_occurrence_rate": float(np.mean(lifted > 0)),
        "spill_occurrence_rate": float(np.mean(spilled > 0)),
        "mean_lifted_particle_count": float(np.mean(lifted)),
        "mean_spill_count": float(np.mean(spilled)),
    }


def _linear_trend(x: np.ndarray, y: np.ndarray) -> dict[str, float | int | None]:
    if len(x) < 2 or np.ptp(x) == 0:
        return {
            "sample_count": int(len(x)),
            "slope_per_episode": None,
            "intercept": None,
            "r_squared": None,
        }
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual_sum = float(np.sum((y - predicted) ** 2))
    total_sum = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = None if total_sum == 0.0 else float(1.0 - residual_sum / total_sum)
    return {
        "sample_count": int(len(x)),
        "slope_per_episode": float(slope),
        "intercept": float(intercept),
        "r_squared": r_squared,
    }


def _late_variability(values: np.ndarray) -> dict[str, float | int]:
    median = float(np.median(values))
    differences = np.diff(values)
    return {
        "sample_count": int(len(values)),
        "standard_deviation": float(np.std(values, ddof=0)),
        "median_absolute_deviation": float(np.median(np.abs(values - median))),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "range": float(np.ptp(values)),
        "mean_absolute_episode_change": (
            0.0 if len(differences) == 0 else float(np.mean(np.abs(differences)))
        ),
    }


def _build_summary(
    episodes_csv: Path,
    episodes: list[dict[str, int | float]],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    episode_ids = np.asarray([item["episode_id"] for item in episodes], dtype=int)
    rewards = np.asarray([item["final_reward"] for item in episodes], dtype=float)
    lift_angles = np.asarray([item["lift_angle_deg"] for item in episodes], dtype=float)
    lifted_counts = np.asarray([item["lifted_particle_count"] for item in episodes], dtype=int)
    lift_rewards = np.asarray([item["lift_reward"] for item in episodes], dtype=float)
    spill_counts = np.asarray([item["spill_count"] for item in episodes], dtype=int)
    spill_rewards = np.asarray([item["spill_reward"] for item in episodes], dtype=float)
    particle_counts = np.asarray([item["particle_count"] for item in episodes], dtype=int)
    moving_mean = _trailing_mean(rewards, MOVING_MEAN_WINDOW)
    cumulative_best = np.maximum.accumulate(rewards)

    window_size = min(SUMMARY_WINDOW, len(episodes))
    initial_indices = np.arange(window_size, dtype=int)
    final_indices = np.arange(len(episodes) - window_size, len(episodes), dtype=int)
    best_index = int(np.argmax(rewards))

    strata: dict[str, Any] = {}
    for particle_count in np.unique(particle_counts):
        indices = np.flatnonzero(particle_counts == particle_count)
        strata[str(int(particle_count))] = _window_summary(episodes, indices)

    episode_records = []
    for index, item in enumerate(episodes):
        episode_records.append(
            {
                **item,
                "reward_moving_mean_12": float(moving_mean[index]),
                "cumulative_best_reward": float(cumulative_best[index]),
            }
        )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "source_csv": str(episodes_csv.resolve()),
        "episode_count": len(episodes),
        "moving_mean_window": MOVING_MEAN_WINDOW,
        "summary_window": window_size,
        "overall": _window_summary(episodes, np.arange(len(episodes), dtype=int)),
        "initial_30": _window_summary(episodes, initial_indices),
        "final_30": _window_summary(episodes, final_indices),
        "by_particle_count": strata,
        "best_episode": {
            **episodes[best_index],
            "training_order_index": best_index,
        },
        "cumulative_best": [
            {
                "episode_id": int(episode_ids[index]),
                "final_reward": float(cumulative_best[index]),
            }
            for index in range(len(episodes))
        ],
        "linear_trend": {
            "overall": _linear_trend(episode_ids.astype(float), rewards),
            "final_30": _linear_trend(
                episode_ids[final_indices].astype(float),
                rewards[final_indices],
            ),
        },
        "late_variability": _late_variability(rewards[final_indices]),
        "episodes": episode_records,
    }
    arrays = {
        "episode_ids": episode_ids,
        "rewards": rewards,
        "moving_mean": moving_mean,
        "cumulative_best": cumulative_best,
        "lift_angles": lift_angles,
        "lifted_counts": lifted_counts,
        "lift_rewards": lift_rewards,
        "spill_counts": spill_counts,
        "spill_rewards": spill_rewards,
        "particle_counts": particle_counts,
    }
    return summary, arrays


def _reward_needs_symlog(values: np.ndarray) -> bool:
    lower, upper = np.quantile(values, [0.05, 0.95])
    central_range = float(upper - lower)
    full_range = float(np.ptp(values))
    if full_range == 0.0:
        return False
    return central_range == 0.0 or full_range > 15.0 * central_range


def _apply_reward_scale(axis: Any, values: np.ndarray) -> None:
    if not _reward_needs_symlog(values):
        return
    lower, median, upper = np.quantile(values, [0.25, 0.5, 0.75])
    linthresh = max(1.0, abs(float(median)) + 2.0 * float(upper - lower))
    axis.set_yscale("symlog", linthresh=linthresh)
    axis.text(
        0.01,
        0.02,
        "symlog y-scale; all raw rewards shown",
        transform=axis.transAxes,
        fontsize=8,
        color="#555555",
    )


def _stratum_colors(particle_counts: np.ndarray) -> dict[int, Any]:
    unique = np.unique(particle_counts)
    color_map = plt.get_cmap("tab10")
    return {int(value): color_map(index % 10) for index, value in enumerate(unique)}


def _plot_training_report(
    arrays: dict[str, np.ndarray],
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    x = arrays["episode_ids"]
    rewards = arrays["rewards"]
    colors = _stratum_colors(arrays["particle_counts"])
    point_colors = [colors[int(value)] for value in arrays["particle_counts"]]

    figure, axes = plt.subplots(2, 3, figsize=(17, 11), constrained_layout=True)

    reward_axis = axes[0, 0]
    reward_axis.scatter(
        x,
        rewards,
        c=point_colors,
        s=21,
        alpha=0.70,
        linewidths=0,
        label="raw reward",
    )
    reward_axis.plot(
        x,
        arrays["moving_mean"],
        color="#111111",
        linewidth=2.0,
        label=f"{MOVING_MEAN_WINDOW}-episode moving mean",
    )
    overall_trend = summary["linear_trend"]["overall"]
    if overall_trend["slope_per_episode"] is not None:
        trend = float(overall_trend["slope_per_episode"]) * x + float(overall_trend["intercept"])
        reward_axis.plot(x, trend, color="#9467BD", linestyle="--", label="linear trend")
    _apply_reward_scale(reward_axis, rewards)
    reward_axis.set(title="A. Reward history", xlabel="episode", ylabel="final reward")
    reward_axis.grid(alpha=0.20)
    reward_axis.legend(fontsize=8)

    best_axis = axes[0, 1]
    best_axis.step(
        x,
        arrays["cumulative_best"],
        where="post",
        color="#2CA02C",
        linewidth=2.0,
        label="cumulative best",
    )
    best = summary["best_episode"]
    best_axis.scatter(
        [best["episode_id"]],
        [best["final_reward"]],
        marker="*",
        color="#D62728",
        s=130,
        zorder=3,
        label=f"best episode {best['episode_id']}",
    )
    best_axis.set(title="B. Best reward reached", xlabel="episode", ylabel="reward")
    best_axis.grid(alpha=0.20)
    best_axis.legend(fontsize=8)

    angle_axis = axes[0, 2]
    for particle_count, color in colors.items():
        mask = arrays["particle_counts"] == particle_count
        angle_axis.scatter(
            x[mask],
            arrays["lift_angles"][mask],
            color=color,
            s=24,
            alpha=0.78,
            linewidths=0,
            label=f"{particle_count} particles",
        )
    angle_axis.set(title="C. Learned lift angle", xlabel="episode", ylabel="lift angle (deg)")
    angle_axis.grid(alpha=0.20)
    angle_axis.legend(fontsize=8)

    lift_axis = axes[1, 0]
    lift_axis.scatter(
        x,
        arrays["lifted_counts"],
        c=point_colors,
        s=22,
        alpha=0.75,
        linewidths=0,
        label="lifted particles",
    )
    lift_reward_axis = lift_axis.twinx()
    lift_reward_axis.plot(
        x,
        arrays["lift_rewards"],
        color="#FF7F0E",
        linewidth=1.25,
        alpha=0.85,
        label="lift reward",
    )
    lift_axis.set(
        title="D. Lift events and reward",
        xlabel="episode",
        ylabel="lifted particle count",
    )
    lift_reward_axis.set_ylabel("lift reward", color="#FF7F0E")
    lift_axis.grid(alpha=0.20)
    lift_axis.legend(loc="upper left", fontsize=8)
    lift_reward_axis.legend(loc="upper right", fontsize=8)

    spill_axis = axes[1, 1]
    spill_axis.scatter(
        x,
        arrays["spill_counts"],
        c=point_colors,
        s=22,
        alpha=0.75,
        linewidths=0,
        label="spilled particles",
    )
    spill_reward_axis = spill_axis.twinx()
    spill_reward_axis.plot(
        x,
        arrays["spill_rewards"],
        color="#D62728",
        linewidth=1.25,
        alpha=0.85,
        label="spill reward",
    )
    spill_axis.set(
        title="E. Spill events and penalty",
        xlabel="episode",
        ylabel="spill count",
    )
    spill_reward_axis.set_ylabel("spill reward", color="#D62728")
    spill_axis.grid(alpha=0.20)
    spill_axis.legend(loc="upper left", fontsize=8)
    spill_reward_axis.legend(loc="lower right", fontsize=8)

    strata_axis = axes[1, 2]
    strata_counts = sorted(colors)
    strata_rewards = [
        rewards[arrays["particle_counts"] == particle_count] for particle_count in strata_counts
    ]
    boxplot = strata_axis.boxplot(
        strata_rewards,
        tick_labels=[str(value) for value in strata_counts],
        patch_artist=True,
        showfliers=True,
    )
    for patch, particle_count in zip(boxplot["boxes"], strata_counts, strict=True):
        patch.set_facecolor(colors[particle_count])
        patch.set_alpha(0.45)
    means = [float(np.mean(values)) for values in strata_rewards]
    strata_axis.scatter(
        np.arange(1, len(strata_counts) + 1),
        means,
        marker="D",
        color="#111111",
        s=28,
        label="mean",
        zorder=3,
    )
    _apply_reward_scale(strata_axis, rewards)
    strata_axis.set(
        title="F. Reward by particle-count stratum",
        xlabel="particle count",
        ylabel="final reward",
    )
    strata_axis.grid(axis="y", alpha=0.20)
    strata_axis.legend(fontsize=8)

    figure.suptitle(
        f"Fried-rice SAC training report — {summary['episode_count']} episodes",
        fontsize=16,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def generate_training_report(
    episodes_csv: str | Path,
    output_directory: str | Path,
) -> dict[str, Path]:
    """Create ``training_report.png`` and ``training_report.json`` from episode logs."""

    csv_path = Path(episodes_csv)
    output_path = Path(output_directory)
    episodes = _read_episodes(csv_path)
    summary, arrays = _build_summary(csv_path, episodes)

    output_path.mkdir(parents=True, exist_ok=True)
    png_path = output_path / "training_report.png"
    json_path = output_path / "training_report.json"
    _plot_training_report(arrays, summary, png_path)
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"png": png_path, "json": json_path}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes_csv", type=Path, help="training run episodes.csv")
    parser.add_argument(
        "output_directory",
        type=Path,
        nargs="?",
        help="output directory (default: <episodes directory>/training_report)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_directory = (
        args.output_directory
        if args.output_directory is not None
        else args.episodes_csv.parent / "training_report"
    )
    outputs = generate_training_report(args.episodes_csv, output_directory)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
