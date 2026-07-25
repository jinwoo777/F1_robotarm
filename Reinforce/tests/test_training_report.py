"""Training CSV report generation tests."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from wok_sim.visualization.training_report import generate_training_report, main

FIELDNAMES = (
    "episode_id",
    "final_reward",
    "lift_angle_rad",
    "lifted_particle_count",
    "lift_reward",
    "spill_count",
    "spill_reward",
    "particle_count",
)


def _write_fixture(path: Path, *, count: int = 40) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        for episode_id in range(count):
            lifted_count = int(episode_id % 2 == 0)
            spill_count = 2 if episode_id % 5 == 0 else 0
            writer.writerow(
                {
                    "episode_id": episode_id,
                    "final_reward": -10_000.0 if episode_id == 5 else float(episode_id),
                    "lift_angle_rad": math.radians(episode_id % 31),
                    "lifted_particle_count": lifted_count,
                    "lift_reward": 0.05 * lifted_count,
                    "spill_count": spill_count,
                    "spill_reward": -0.4 * spill_count,
                    "particle_count": (60, 90, 120)[episode_id % 3],
                }
            )


def test_generate_training_report_writes_png_and_complete_json(tmp_path: Path) -> None:
    episodes_csv = tmp_path / "episodes.csv"
    _write_fixture(episodes_csv)

    outputs = generate_training_report(episodes_csv, tmp_path / "report")

    assert set(outputs) == {"png", "json"}
    assert outputs["png"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    summary = json.loads(outputs["json"].read_text(encoding="utf-8"))

    assert summary["episode_count"] == 40
    assert summary["moving_mean_window"] == 12
    assert summary["summary_window"] == 30
    assert summary["overall"]["final_reward"]["minimum"] == -10_000.0
    assert summary["initial_30"]["episode_id_start"] == 0
    assert summary["initial_30"]["episode_id_end"] == 29
    assert summary["final_30"]["episode_id_start"] == 10
    assert summary["final_30"]["episode_id_end"] == 39
    assert summary["final_30"]["final_reward"]["mean"] == pytest.approx(24.5)
    assert summary["final_30"]["final_reward"]["median"] == pytest.approx(24.5)
    assert summary["final_30"]["lift_occurrence_rate"] == pytest.approx(0.5)
    assert summary["final_30"]["spill_occurrence_rate"] == pytest.approx(0.2)
    assert set(summary["by_particle_count"]) == {"60", "90", "120"}

    assert summary["best_episode"]["episode_id"] == 39
    assert summary["best_episode"]["final_reward"] == 39.0
    assert summary["cumulative_best"][-1] == {
        "episode_id": 39,
        "final_reward": 39.0,
    }
    assert summary["linear_trend"]["overall"]["sample_count"] == 40
    assert summary["linear_trend"]["overall"]["slope_per_episode"] is not None
    assert summary["late_variability"]["sample_count"] == 30
    assert summary["late_variability"]["standard_deviation"] == pytest.approx(
        np.std(np.arange(10.0, 40.0), ddof=0)
    )

    assert summary["episodes"][0]["reward_moving_mean_12"] == 0.0
    assert summary["episodes"][11]["reward_moving_mean_12"] == pytest.approx(-828.25)
    assert summary["episodes"][-1]["reward_moving_mean_12"] == pytest.approx(33.5)
    assert summary["episodes"][-1]["lift_angle_deg"] == pytest.approx(8.0)


def test_report_rejects_missing_required_columns(tmp_path: Path) -> None:
    episodes_csv = tmp_path / "episodes.csv"
    episodes_csv.write_text("episode_id,final_reward\n0,1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        generate_training_report(episodes_csv, tmp_path / "report")


def test_training_report_cli_uses_default_output_directory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    episodes_csv = tmp_path / "episodes.csv"
    _write_fixture(episodes_csv, count=3)

    assert main([str(episodes_csv)]) == 0

    output = json.loads(capsys.readouterr().out)
    assert Path(output["png"]) == tmp_path / "training_report" / "training_report.png"
    assert Path(output["json"]) == tmp_path / "training_report" / "training_report.json"
