"""Pin for ``prediction.cli._upsert_history`` — the protected-dedup upsert (P8 Task 0.7).

A naive ``pd.concat([...]).drop_duplicates(keep="last")`` on stale + fresh
offer rows would let a freshly-scored, always-unclosed row (``Close Market
Prob`` is NaN at prophecize-time) evict an already-closed row from a prior
``reflect`` run on the same ``(PREDICTION_KEY, Line, Platform)`` key, silently
destroying captured CLV data. ``_upsert_history`` sorts on "has a closed
Close Market Prob" before the stable dedup so a closed row always wins
regardless of concat order, while an unclosed-vs-unclosed collision still
prefers the newer (freshly scored) row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sportstradamus.prediction.cli import _upsert_history


def _row(player, line, platform, close_p, projection=20.0, bet="Over"):
    return {
        "Player": player,
        "League": "NBA",
        "Date": "2026-05-04",
        "Market": "points",
        "Team": "BOS",
        "Projection": projection,
        "Line": line,
        "Boost": 1.0,
        "Platform": platform,
        "Bet": bet,
        "Win Prob": 0.58,
        "Market Prob": 0.52,
        "Close Market Prob": close_p,
        "Market CLV": np.nan if pd.isna(close_p) else 0.05,
        "Model CLV": np.nan if pd.isna(close_p) else 0.03,
        "Actual": np.nan,
    }


def test_new_empty_returns_history_unchanged():
    history = pd.DataFrame([_row("P1", 20.5, "Underdog", np.nan)])
    out = _upsert_history(history, pd.DataFrame())
    assert out is history


def test_history_empty_returns_new_df():
    new_df = pd.DataFrame([_row("P1", 20.5, "Underdog", np.nan)])
    out = _upsert_history(pd.DataFrame(), new_df)
    assert out is new_df


def test_closed_row_survives_against_fresh_unclosed_same_key():
    """Old row already closed by reflect; new prophecize pass rescoring the
    same (key, Line, Platform) must NOT wipe out the captured CLV."""
    history = pd.DataFrame([_row("P1", 20.5, "Underdog", close_p=0.60, projection=20.0)])
    new_df = pd.DataFrame([_row("P1", 20.5, "Underdog", close_p=np.nan, projection=21.0)])

    out = _upsert_history(history, new_df)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["Close Market Prob"] == 0.60
    assert row["Market CLV"] == 0.05
    # Old row wins wholesale (including its stale Projection) — closed status
    # is the only precedence signal; this is a documented limitation, not a bug.
    assert row["Projection"] == 20.0


def test_unclosed_vs_unclosed_prefers_newer_row():
    """Neither side has closing data yet -> ordinary keep='last' freshness wins."""
    history = pd.DataFrame([_row("P1", 20.5, "Underdog", close_p=np.nan, projection=20.0)])
    new_df = pd.DataFrame([_row("P1", 20.5, "Underdog", close_p=np.nan, projection=21.0)])

    out = _upsert_history(history, new_df)

    assert len(out) == 1
    assert out.iloc[0]["Projection"] == 21.0


def test_distinct_keys_all_survive_with_correct_precedence():
    """Two independent (key, Line, Platform) cells in the same upsert: one
    protects a closed row, the other takes the fresh unclosed row. A third,
    untouched-this-run offer for a different Line on P1 must also survive."""
    history = pd.DataFrame(
        [
            _row("P1", 20.5, "Underdog", close_p=0.60, projection=20.0),  # closed, protect
            _row("P1", 22.5, "Sleeper", close_p=np.nan, projection=20.0),  # untouched this run
            _row("P2", 8.5, "Underdog", close_p=np.nan, projection=10.0),  # unclosed, will refresh
        ]
    )
    new_df = pd.DataFrame(
        [
            _row("P1", 20.5, "Underdog", close_p=np.nan, projection=99.0),  # must lose to closed
            _row("P2", 8.5, "Underdog", close_p=np.nan, projection=11.0),  # must win (newer)
        ]
    )

    out = _upsert_history(history, new_df).set_index(["Player", "Line", "Platform"])

    assert len(out) == 3
    assert out.loc[("P1", 20.5, "Underdog"), "Close Market Prob"] == 0.60
    assert out.loc[("P1", 22.5, "Sleeper"), "Projection"] == 20.0
    assert out.loc[("P2", 8.5, "Underdog"), "Projection"] == 11.0


def test_prediction_level_freshness_falls_out_of_keep_last():
    """A touched offer-key picks up the newer prediction-level Team/Projection
    without any extra merge step — keep='last' on the concat handles it."""
    history = pd.DataFrame([_row("P1", 20.5, "Underdog", close_p=np.nan, projection=20.0)])
    new_row = _row("P1", 20.5, "Underdog", close_p=np.nan, projection=20.0)
    new_row["Team"] = "LAL"  # traded since the last prophecize run
    new_df = pd.DataFrame([new_row])

    out = _upsert_history(history, new_df)

    assert out.iloc[0]["Team"] == "LAL"
