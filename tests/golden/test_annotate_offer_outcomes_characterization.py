"""Pin for ``analysis.annotate_offer_outcomes`` (P8 Task 0.7 ``explode_offers`` successor).

The flat history frame is already one row per (prediction x book offer), so
the replacement for ``explode_offers`` only needs to compute each row's
Over/Under/Push ``Result`` from ``Actual`` vs ``Line`` and fold in the
Kelly-relevant columns (``_add_kelly_columns``, unchanged) — no groupby /
merge-offers mechanics. Covers resolved Over, resolved Under, Push, an
unresolved (Actual NaN) row, and the empty-frame passthrough.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sportstradamus.analysis import annotate_offer_outcomes


def _row(actual, line, bet="Over", player="Player A"):
    return {
        "Player": player,
        "League": "NBA",
        "Date": "2026-05-04",
        "Market": "points",
        "Line": line,
        "Boost": 1.0,
        "Platform": "Underdog",
        "Bet": bet,
        "Win Prob": 0.58,
        "Market Prob": 0.52,
        "Actual": actual,
    }


def test_resolved_over_result():
    history = pd.DataFrame([_row(actual=25.0, line=20.5, bet="Over")])
    out = annotate_offer_outcomes(history)
    assert out.iloc[0]["Result"] == "Over"
    assert out.iloc[0]["Hit"] == 1


def test_resolved_under_result():
    history = pd.DataFrame([_row(actual=15.0, line=20.5, bet="Under")])
    out = annotate_offer_outcomes(history)
    assert out.iloc[0]["Result"] == "Under"
    assert out.iloc[0]["Hit"] == 1


def test_bet_wrong_side_is_a_miss():
    history = pd.DataFrame([_row(actual=25.0, line=20.5, bet="Under")])
    out = annotate_offer_outcomes(history)
    assert out.iloc[0]["Result"] == "Over"
    assert out.iloc[0]["Hit"] == 0


def test_push_result_when_actual_equals_line():
    history = pd.DataFrame([_row(actual=20.5, line=20.5, bet="Over")])
    out = annotate_offer_outcomes(history)
    assert out.iloc[0]["Result"] == "Push"


def test_unresolved_row_has_nan_result_and_no_hit():
    history = pd.DataFrame([_row(actual=np.nan, line=20.5, bet="Over")])
    out = annotate_offer_outcomes(history)
    assert pd.isna(out.iloc[0]["Result"])
    assert "Hit" not in out.columns or pd.isna(out.iloc[0].get("Hit"))


def test_matchup_player_resolves_on_actual_plus_line_sign():
    """" vs. " players (moneyline/spread-style) resolve on sign(actual + line),
    not actual vs. line directly."""
    history = pd.DataFrame(
        [_row(actual=3.0, line=-1.0, bet="Over", player="Team A vs. Team B")]
    )
    out = annotate_offer_outcomes(history)
    assert out.iloc[0]["Result"] == "Over"  # 3.0 + (-1.0) = 2.0 > 0


def test_empty_frame_returned_unchanged():
    empty = pd.DataFrame()
    out = annotate_offer_outcomes(empty)
    assert out is empty
