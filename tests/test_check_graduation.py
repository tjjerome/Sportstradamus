"""Unit tests for the Stage 0 check_graduation CLI (deliverable 0.4)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from sportstradamus.scripts.check_graduation import _classify_lifecycle, main


@pytest.mark.parametrize(
    ("gate1_bss", "n_settled", "book_bss_30d", "expected"),
    [
        (math.nan, 500, 0.10, "not-shipped"),
        (-0.05, 500, 0.10, "not-shipped"),
        (0.10, 50, 0.10, "in-test"),
        (0.10, 250, math.nan, "in-test"),  # live row missing entirely
        (0.10, 250, 0.05, "graduated"),
        (0.10, 250, 0.0, "graduated"),
        (0.10, 250, -0.04, "demoted"),
    ],
)
def test_lifecycle_classification(gate1_bss, n_settled, book_bss_30d, expected):
    assert _classify_lifecycle(gate1_bss, n_settled, book_bss_30d) == expected


def _seed_model_stats(path):
    rows = []
    cells = [
        ("NBA", "PTS", "SkewNormal", 0.12),
        ("NBA", "FG3M", "ZINB", 0.08),
        ("WNBA", "PTS", "SkewNormal", -0.05),
        ("WNBA", "FG3M", "ZINB", 0.04),
        ("NFL", "rushing_yards", "SkewNormal", 0.07),
        ("NFL", "passing_tds", "ZINB", math.nan),
    ]
    for league, market, dist, bss in cells:
        # Required filter keys.
        rows.append(
            {
                "league": league,
                "market": market,
                "distribution": dist,
                "row_kind": "model",
                "metric_row": "calibrated",
                "brier_skill_score": bss,
                "predicted_over_rate": 0.52,
                "empirical_over_rate": 0.50,
                "kelly_shrinkage": max(bss if not math.isnan(bss) else 0.0, 0.0),
            }
        )
        # Book baseline row that the CLI must filter out.
        rows.append(
            {
                "league": league,
                "market": market,
                "distribution": dist,
                "row_kind": "book_baseline",
                "metric_row": None,
                "brier_skill_score": 0.0,
                "predicted_over_rate": 0.50,
                "empirical_over_rate": 0.50,
                "kelly_shrinkage": 0.0,
            }
        )
    pd.DataFrame(rows).to_parquet(path, engine="pyarrow", index=False)


def _seed_live_metrics(path):
    rows = []
    # (league, market, n_settled, book_bss) — supplies the Gate 2 side.
    cells = [
        ("NBA", "PTS", 300, 0.05),       # graduated
        ("NBA", "FG3M", 220, -0.03),     # demoted
        ("WNBA", "PTS", 250, 0.10),      # gate1 fails → still not-shipped
        ("WNBA", "FG3M", 80, 0.02),      # in-test (n < 200)
        ("NFL", "rushing_yards", 350, 0.04),  # graduated
        # NFL passing_tds intentionally absent — outer-merge keeps the model row.
    ]
    for league, market, n, bss in cells:
        for window in (7, 30):
            rows.append(
                {
                    "league": league,
                    "market": market,
                    "computed_at": pd.Timestamp("2026-05-20"),
                    "window_days": np.int16(window),
                    "n_settled": np.int64(n),
                    "book_bss": bss,
                    "empirical_over_rate": 0.51,
                    "predicted_over_rate": 0.53,
                    "top_decile_mae": 2.1,
                    "profit_sim_yield": 0.04,
                }
            )
    pd.DataFrame(rows).to_parquet(path, engine="pyarrow", index=False)


def test_check_graduation_synthetic_parquets(tmp_path):
    model_stats = tmp_path / "model_stats.parquet"
    live_metrics = tmp_path / "live_metrics.parquet"
    _seed_model_stats(model_stats)
    _seed_live_metrics(live_metrics)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--model-stats-path", str(model_stats),
            "--live-metrics-path", str(live_metrics),
        ],
    )
    assert result.exit_code == 0, result.output
    # Every (league, market) cell from model_stats should appear once.
    expected_cells = [
        ("NBA", "PTS"), ("NBA", "FG3M"), ("WNBA", "PTS"), ("WNBA", "FG3M"),
        ("NFL", "rushing_yards"), ("NFL", "passing_tds"),
    ]
    for league, market in expected_cells:
        assert f"{league}" in result.output
        assert market in result.output
    # All four lifecycle states should show up across the synthetic fixture.
    for state in ("graduated", "in-test", "demoted", "not-shipped"):
        assert state in result.output


def test_check_graduation_filters_by_league(tmp_path):
    model_stats = tmp_path / "model_stats.parquet"
    live_metrics = tmp_path / "live_metrics.parquet"
    _seed_model_stats(model_stats)
    _seed_live_metrics(live_metrics)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--league", "NBA",
            "--model-stats-path", str(model_stats),
            "--live-metrics-path", str(live_metrics),
        ],
    )
    assert result.exit_code == 0
    assert "NBA" in result.output
    # The WNBA / NFL cells should not appear when filtered.
    assert "WNBA" not in result.output
    assert "NFL" not in result.output


def test_check_graduation_missing_live_parquet_classifies_all_as_in_test(tmp_path):
    model_stats = tmp_path / "model_stats.parquet"
    _seed_model_stats(model_stats)
    missing_live = tmp_path / "live_does_not_exist.parquet"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--model-stats-path", str(model_stats),
            "--live-metrics-path", str(missing_live),
        ],
    )
    assert result.exit_code == 0
    # Cells with positive Gate 1 BSS but no live row → in-test.
    assert "in-test" in result.output
    # Cells with NaN or negative Gate 1 BSS → not-shipped (even without live data).
    assert "not-shipped" in result.output


def test_check_graduation_missing_model_stats_errors(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--model-stats-path", str(tmp_path / "missing.parquet"),
            "--live-metrics-path", str(tmp_path / "also_missing.parquet"),
        ],
    )
    assert result.exit_code != 0
    assert "model_stats" in result.output.lower()
