"""Pin for ``dashboard.data.get_prediction_history``'s dedup (P8 Task 0.7).

The flat history frame has one row per (prediction x book offer); CRPS /
coverage analysis needs one row per prediction. This pins that
``get_prediction_history``'s ``drop_duplicates(subset=PREDICTION_KEY,
keep="last")`` agrees with the equivalent ``groupby(PREDICTION_KEY).last()``
projection, and that the league/date-range filters apply before the dedup.
"""

from __future__ import annotations

import pandas as pd

from sportstradamus.dashboard.data import get_prediction_history
from sportstradamus.history_schema import PREDICTION_KEY


def _offer_row(player, league, date, market, line, platform, projection):
    return {
        "Player": player,
        "League": league,
        "Date": date,
        "Market": market,
        "Team": "BOS",
        "Projection": projection,
        "Line": line,
        "Platform": platform,
        "Bet": "Over",
    }


def _multi_offer_history():
    """Two predictions, each with two book offers (four rows, two prediction keys)."""
    return pd.DataFrame(
        [
            _offer_row("Player A", "NBA", "2026-05-04", "points", 20.5, "Underdog", 21.0),
            _offer_row("Player A", "NBA", "2026-05-04", "points", 21.5, "Sleeper", 21.0),
            _offer_row("Player B", "NBA", "2026-05-05", "rebounds", 8.5, "Underdog", 9.0),
            _offer_row("Player B", "NBA", "2026-05-05", "rebounds", 9.5, "Sleeper", 9.0),
        ]
    )


def test_dedup_matches_groupby_last_projection():
    history = _multi_offer_history()
    out = get_prediction_history(history)
    expected = (
        history.groupby(PREDICTION_KEY, as_index=False, sort=False)
        .last()
        .reindex(columns=out.columns)
        .reset_index(drop=True)
    )
    assert len(out) == 2
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True).sort_values("Player").reset_index(drop=True),
        expected.sort_values("Player").reset_index(drop=True),
    )


def test_one_row_per_prediction_key_no_duplicates():
    history = _multi_offer_history()
    out = get_prediction_history(history)
    assert not out.duplicated(subset=PREDICTION_KEY).any()


def test_league_filter_applies_before_dedup():
    history = _multi_offer_history()
    wnba_row = _offer_row("Player C", "WNBA", "2026-05-04", "points", 15.5, "Underdog", 16.0)
    history = pd.concat([history, pd.DataFrame([wnba_row])], ignore_index=True)

    out = get_prediction_history(history, leagues=["NBA"])
    assert set(out["League"]) == {"NBA"}
    assert len(out) == 2


def test_date_range_filter_applies_before_dedup():
    history = _multi_offer_history()
    out = get_prediction_history(
        history, date_range=(pd.Timestamp("2026-05-05").date(), pd.Timestamp("2026-05-05").date())
    )
    assert len(out) == 1
    assert out.iloc[0]["Player"] == "Player B"
