"""Unit tests for the shared graduation classifier (training/graduation.py)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from sportstradamus.realized import RECOMMENDED_EDGE_MIN
from sportstradamus.training.graduation import (
    DESCRIPTIVE_MIN_SELECTED,
    MIN_PRECISION_SIDE,
    classify_lifecycle,
    free_passer_cells,
    graduated_cells,
    lifecycle_table,
    read_gate1,
    read_gate2,
    read_tail,
    tail_state,
)

TAIL_COLS = ["tail_n", "tail_gap", "tail_gap_hi", "live_n", "live_gap", "live_gap_hi", "tail_state"]


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
    ("precision_over", "precision_under", "expected"),
    [
        # Both sides above threshold -> graduated.
        (0.60, 0.60, "graduated"),
        # Exactly at threshold on both sides is still graduated (strict "<" demote).
        (MIN_PRECISION_SIDE, MIN_PRECISION_SIDE, "graduated"),
        # Over below threshold, Under healthy -> demoted.
        (0.45, 0.60, "demoted"),
        # Under below threshold, Over healthy -> demoted.
        (0.60, 0.45, "demoted"),
        # Neither side has enough bets to estimate (NaN) -> precision check
        # skipped entirely -> not demoted on precision.
        (math.nan, math.nan, "graduated"),
    ],
)
def test_classify_lifecycle_precision_side_gate(precision_over, precision_under, expected):
    """Precision gate is symmetric: demote when either recommended side loses more than it wins."""
    assert (
        classify_lifecycle(
            gate1_bss=0.10,
            n_settled=500,
            book_bss_30d=0.05,
            precision_over_live=precision_over,
            precision_under_live=precision_under,
        )
        == expected
    )


def _seed_model_stats(path):
    """Seed the wide one-row-per-cell ``model_stats.parquet`` schema."""
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
                "brier_skill_score": bss,
                "predicted_over_rate": 0.52,
                "empirical_over_rate": 0.50,
                "kelly_shrinkage": 0.1,
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


def test_read_gate1_renames_brier_skill_score(tmp_path):
    """read_gate1 renames ``brier_skill_score`` to ``gate1_bss`` over the wide rows."""
    p = tmp_path / "model_stats.parquet"
    _seed_model_stats(p)
    df = read_gate1(p)
    assert "gate1_bss" in df.columns
    assert "brier_skill_score" not in df.columns
    assert len(df) == 3
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


def _seed_model_stats_with_ship(path):
    """Seed model_stats with an explicit per-cell ``ship`` (True / False / NA)."""
    rows = [
        {"league": "NBA", "market": "PTS", "ship": True},  # passes ship
        {"league": "NBA", "market": "REB", "ship": False},  # fails ship
        {"league": "WNBA", "market": "AST", "ship": True},  # passes ship, already served
        {"league": "NFL", "market": "carries", "ship": None},  # no scored evidence
    ]
    pd.DataFrame(rows).to_parquet(path, engine="pyarrow", index=False)


def test_free_passer_cells(tmp_path):
    ms = tmp_path / "model_stats.parquet"
    _seed_model_stats_with_ship(ms)
    meta = {
        "NBA": {"PTS": {"shipped": "withheld"}, "REB": {"shipped": "withheld"}},
        "WNBA": {"AST": {"shipped": "devel"}},
        "NFL": {"carries": {"shipped": "withheld"}},
    }
    # Only NBA/PTS: ship==True AND still withheld. REB fails ship; WNBA/AST already
    # served; NFL/carries has NA ship (no evidence -> not a pass).
    assert free_passer_cells(meta, ms) == [("NBA", "PTS")]


def test_free_passer_cells_missing_model_stats_is_empty(tmp_path):
    meta = {"NBA": {"PTS": {"shipped": "withheld"}}}
    assert free_passer_cells(meta, tmp_path / "nope.parquet") == []


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


@pytest.mark.parametrize(
    ("n", "gap_hi", "expected"),
    [
        (DESCRIPTIVE_MIN_SELECTED - 1, 0.0, "thin"),
        (math.nan, 0.0, "thin"),
        (DESCRIPTIVE_MIN_SELECTED, RECOMMENDED_EDGE_MIN - 0.001, "honest"),
        (DESCRIPTIVE_MIN_SELECTED, RECOMMENDED_EDGE_MIN, "overstated"),
    ],
)
def test_tail_state_at_the_bar(n, gap_hi, expected):
    """Decision D2: the bar is strict on both sides (n below the floor, gap_hi below the edge)."""
    assert tail_state(n, gap_hi) == expected


def _seed_tail_scorecard(path):
    """The scorecard's cell rows: `market` is the cell name; `market_read` is the read it
    reports under the headline `market` column, carried here to show it is never joined."""
    rows = [
        # (league, market, n, gap, gap_hi, live_n, live_gap, live_gap_hi)
        ("NBA", "PTS", 150, 0.080, 0.110, 60, 0.030, 0.090),  # overstated; graduated by the gates
        ("NBA", "FG3M", 40, 0.120, 0.300, np.nan, np.nan, np.nan),  # thin
        ("WNBA", "PTS", 200, 0.020, 0.040, 20, 0.010, 0.200),  # honest; not-shipped by the gates
    ]
    frame = pd.DataFrame(
        rows,
        columns=["league", "market", "n", "gap", "gap_hi", "live_n", "live_gap", "live_gap_hi"],
    )
    frame["scope"] = "cell"
    frame["key"] = frame["league"] + "/" + frame["market"]
    frame["read"] = 0.60
    frame["hit"] = frame["read"] - frame["gap"]
    frame["market_read"] = 0.58
    frame["descriptive"] = frame["n"] < DESCRIPTIVE_MIN_SELECTED
    frame["model_version"] = "20261010.1"
    frame["computed_at"] = pd.Timestamp("2026-10-17")
    frame.to_parquet(path, engine="pyarrow", index=False)


def _seed_live_gap(path):
    rows = [
        # (league, market, live_n, live_gap, live_gap_lo, live_gap_hi)
        ("NBA", "PTS", 90, 0.050, 0.010, 0.095),  # also in the scorecard: the nightly wins
        ("NFL", "carries", 30, 0.000, -0.050, 0.020),  # no scorecard row
    ]
    frame = pd.DataFrame(
        rows, columns=["league", "market", "live_n", "live_gap", "live_gap_lo", "live_gap_hi"]
    )
    frame["computed_at"] = pd.Timestamp("2026-10-17")
    frame["window_days"] = np.int16(30)
    frame.to_parquet(path, engine="pyarrow", index=False)


def test_read_tail_prefers_the_nightly_live_half(tmp_path):
    tail = tmp_path / "tail_scorecard.parquet"
    live = tmp_path / "live_gap_per_cell.parquet"
    _seed_tail_scorecard(tail)
    _seed_live_gap(live)
    df = read_tail(tail, live)
    assert set(df.columns) == {"league", "market", *TAIL_COLS}
    rows = df.set_index(["league", "market"])
    pts = rows.loc[("NBA", "PTS")]
    assert (pts["tail_n"], pts["tail_gap"], pts["tail_gap_hi"]) == (150, 0.08, 0.11)
    assert (pts["live_n"], pts["live_gap"], pts["live_gap_hi"]) == (90, 0.05, 0.095)
    # Only the scorecard holds WNBA/PTS, so its own live half stands.
    assert rows.loc[("WNBA", "PTS"), "live_gap"] == 0.01
    # Only the nightly holds NFL/carries: live columns present, no offline read.
    assert rows.loc[("NFL", "carries"), "live_n"] == 30
    assert math.isnan(rows.loc[("NFL", "carries"), "tail_n"])
    assert dict(rows["tail_state"]) == {
        ("NBA", "PTS"): "overstated",
        ("NBA", "FG3M"): "thin",
        ("WNBA", "PTS"): "honest",
        ("NFL", "carries"): "thin",
    }


def test_read_tail_missing_sidecars_keep_every_column(tmp_path):
    live = tmp_path / "live_gap_per_cell.parquet"
    _seed_live_gap(live)
    neither = read_tail(tmp_path / "no_tail.parquet", tmp_path / "no_live.parquet")
    assert neither.empty
    assert set(neither.columns) == {"league", "market", *TAIL_COLS}
    nightly_only = read_tail(tmp_path / "no_tail.parquet", live)
    assert len(nightly_only) == 2
    assert nightly_only["tail_gap"].isna().all()
    assert nightly_only["live_gap"].tolist() == [0.05, 0.0]
    assert nightly_only["tail_state"].eq("thin").all()


def test_lifecycle_table_joins_the_tail_read_without_moving_states(tmp_path):
    """The tail read rides the lifecycle frame as monitor columns; it never gates."""
    ms = tmp_path / "model_stats.parquet"
    lm = tmp_path / "live.parquet"
    tail = tmp_path / "tail_scorecard.parquet"
    live = tmp_path / "live_gap_per_cell.parquet"
    _seed_model_stats(ms)
    _seed_live_metrics(lm)
    _seed_tail_scorecard(tail)
    _seed_live_gap(live)
    bare = lifecycle_table(
        ms, lm, tail_path=tmp_path / "no_tail.parquet", live_gap_path=tmp_path / "no_live.parquet"
    )
    joined = lifecycle_table(ms, lm, tail_path=tail, live_gap_path=live)
    assert joined["lifecycle_state"].tolist() == bare["lifecycle_state"].tolist()
    rows = joined.set_index(["league", "market"])
    # An overstated tail does not demote a graduated cell, nor an honest one promote.
    assert rows.loc[("NBA", "PTS"), ["lifecycle_state", "tail_state"]].tolist() == [
        "graduated",
        "overstated",
    ]
    assert rows.loc[("WNBA", "PTS"), ["lifecycle_state", "tail_state"]].tolist() == [
        "not-shipped",
        "honest",
    ]
    assert rows.loc[("NBA", "PTS"), "live_gap"] == 0.05
    assert rows.loc[("NBA", "PTS"), "tail_gap_hi"] == 0.11
    # Without either sidecar every tail column is still there, empty.
    assert set(TAIL_COLS) <= set(bare.columns)
    assert bare[TAIL_COLS].isna().all().all()
    assert len(bare) == len(joined) == 3
