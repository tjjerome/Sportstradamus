"""Unit tests for the shared graduation classifier (training/graduation.py)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from sportstradamus.training.graduation import (
    MIN_PRECISION_OVER,
    classify_lifecycle,
    graduated_cells,
    lifecycle_table,
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


@pytest.mark.parametrize(
    ("precision_over", "expected"),
    [
        # Above threshold -> graduated.
        (0.60, "graduated"),
        (0.51, "graduated"),
        # Exactly at threshold is still graduated (strict "<" demote).
        (MIN_PRECISION_OVER, "graduated"),
        # Below threshold -> demoted.
        (MIN_PRECISION_OVER - 1e-6, "demoted"),
        (0.40, "demoted"),
        # NaN (too few Over bets to estimate) skips the check -> graduated
        # since the other gates pass.
        (math.nan, "graduated"),
    ],
)
def test_classify_lifecycle_precision_over_gate(precision_over, expected):
    """precision_over-based gate: demote when Bet=Over recommendations lose more than they win."""
    assert (
        classify_lifecycle(
            gate1_bss=0.10,
            n_settled=500,
            book_bss_30d=0.05,
            precision_over_live=precision_over,
        )
        == expected
    )


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


def _seed_live_metrics_with_precision(path):
    """Seed two cells: one with healthy Over precision, one losing on its Over recs."""
    rows = []
    for league, market, n, bss, prec_over, prec_under in [
        ("NBA", "PTS", 300, 0.05, 0.55, 0.52),  # graduated
        ("NBA", "FG3M", 1300, 0.054, 0.42, 0.45),  # precision_over < 0.50 -> demoted
    ]:
        for window in (7, 30):
            rows.append(
                {
                    "league": league,
                    "market": market,
                    "computed_at": pd.Timestamp("2026-05-24"),
                    "window_days": np.int16(window),
                    "n_settled": np.int64(n),
                    "book_bss": bss,
                    "empirical_over_rate": 0.50,
                    "predicted_over_rate": 0.50,
                    "precision_over_live": prec_over,
                    "precision_under_live": prec_under,
                    "top_decile_mae": 2.1,
                    "profit_sim_yield": 0.04,
                }
            )
    pd.DataFrame(rows).to_parquet(path, engine="pyarrow", index=False)


def test_lifecycle_table_demotes_on_precision_over_below_breakeven(tmp_path):
    """A positive-book_bss cell must still demote if live Bet=Over hit rate is below 50%.

    Regression guard for NBA/FG3M circa 2026-05-24: cell had +0.054 live book_bss
    but its Over recommendations were losing money. The Brier-based gate let it
    through; the precision-over gate must catch it.
    """
    ms = tmp_path / "model_stats.parquet"
    lm = tmp_path / "live.parquet"
    _seed_model_stats(ms)
    _seed_live_metrics_with_precision(lm)
    table = lifecycle_table(ms, lm, league="NBA")
    states = dict(zip(table["market"], table["lifecycle_state"], strict=False))
    assert states["PTS"] == "graduated"
    assert states["FG3M"] == "demoted"


def test_read_gate2_back_fills_precision_columns_for_old_parquets(tmp_path):
    """Live parquets written before the precision columns shipped get NaN-filled."""
    # _seed_live_metrics writes the pre-precision schema (no precision_*_live cols).
    p = tmp_path / "old_live.parquet"
    _seed_live_metrics(p)
    df = read_gate2(p)
    assert "precision_over_live" in df.columns
    assert "precision_under_live" in df.columns
    assert df["precision_over_live"].isna().all()
    assert df["precision_under_live"].isna().all()
