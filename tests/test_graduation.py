"""Unit tests for the shared graduation classifier (training/graduation.py)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from sportstradamus.training.graduation import (
    classify_lifecycle,
    graduated_cells,
    read_gate1,
    read_gate2,
)


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
def test_classify_lifecycle(gate1_bss, n_settled, book_bss_30d, expected):
    assert classify_lifecycle(gate1_bss, n_settled, book_bss_30d) == expected


def _seed_model_stats(path):
    rows = []
    for league, market, bss in [
        ("NBA", "PTS", 0.12),
        ("NBA", "FG3M", 0.08),
        ("WNBA", "PTS", -0.05),
    ]:
        rows.append(
            {
                "league": league,
                "market": market,
                "distribution": "SkewNormal",
                "row_kind": "model",
                "metric_row": "calibrated",
                "brier_skill_score": bss,
                "predicted_over_rate": 0.52,
                "empirical_over_rate": 0.50,
                "kelly_shrinkage": 0.1,
            }
        )
        # A book_baseline row the reader must filter out.
        rows.append(
            {
                "league": league,
                "market": market,
                "distribution": "SkewNormal",
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
    for league, market, n, bss in [("NBA", "PTS", 300, 0.05), ("NBA", "FG3M", 220, -0.03)]:
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


def test_read_gate1_filters_and_renames(tmp_path):
    p = tmp_path / "model_stats.parquet"
    _seed_model_stats(p)
    df = read_gate1(p)
    assert "gate1_bss" in df.columns
    assert "brier_skill_score" not in df.columns
    assert len(df) == 3  # one model+calibrated row per cell; book_baseline filtered out
    assert set(zip(df["league"], df["market"], strict=False)) == {
        ("NBA", "PTS"),
        ("NBA", "FG3M"),
        ("WNBA", "PTS"),
    }


def test_read_gate1_league_filter(tmp_path):
    p = tmp_path / "model_stats.parquet"
    _seed_model_stats(p)
    df = read_gate1(p, league="WNBA")
    assert set(df["league"]) == {"WNBA"}


def test_read_gate2_window_filter_and_rename(tmp_path):
    p = tmp_path / "live.parquet"
    _seed_live_metrics(p)
    df = read_gate2(p)
    assert "gate2_book_bss" in df.columns
    assert len(df) == 2  # only the 30d window rows survive
    assert set(df["n_settled"]) == {300, 220}


def test_read_gate2_missing_returns_empty_with_columns(tmp_path):
    df = read_gate2(tmp_path / "nope.parquet")
    assert df.empty
    assert "gate2_book_bss" in df.columns


def test_graduated_cells(tmp_path):
    ms = tmp_path / "model_stats.parquet"
    lm = tmp_path / "live.parquet"
    _seed_model_stats(ms)
    _seed_live_metrics(lm)
    # NBA/PTS: gate1 0.12, n 300, book 0.05 -> graduated.
    # NBA/FG3M: gate1 0.08, n 220, book -0.03 -> demoted.
    # WNBA/PTS: gate1 -0.05 -> not-shipped.
    assert graduated_cells(ms, lm) == {("NBA", "PTS")}


def test_graduated_cells_missing_model_stats_is_empty(tmp_path):
    assert graduated_cells(tmp_path / "nope.parquet", tmp_path / "nolive.parquet") == set()
