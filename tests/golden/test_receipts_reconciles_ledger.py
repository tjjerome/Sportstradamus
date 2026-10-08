"""The Receipts hero reconciles with the nightly realized-by-side ledger to the leg.

Receipts prices ``cohort_summary`` over the windowed posted frame and its recommended cohort;
nightly persists ``compute_realized_by_side``. Over one hand-built history (both platforms,
an unposted side, a push, an alt line, a payout past the cap, a ``book_fallback`` row, one prop
on both platforms, one whose line moved and one last scored on an unposted side) the two agree
for every ledger window and cohort. ``HISTORY`` and ``NOW`` also feed the Receipts render tests.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from sportstradamus.history_schema import HISTORY_COLS
from sportstradamus.realized import (
    COUNT_ONCE_FROM,
    REALIZED_WINDOWS,
    cohort_summary,
    compute_realized_by_side,
    settled_offers,
    window,
)

NOW = datetime(2026, 10, 10, 12)
# The first game date a moved prop counts once on, and three prophecize runs ahead of its games.
_GAME = f"{COUNT_ONCE_FROM:%Y-%m-%d}"
_RUNS = pd.date_range("2026-10-08 13:50", periods=3, freq="6h")
# Every posted Underdog leg here loses and every Sleeper payout is a binary fraction, so
# each Unit is exact in floating point and a sum of them is the same whatever the order —
# the page sums a cohort in one pass, the ledger per side.
_COLS = [
    "Player",
    "League",
    "Date",
    "Market",
    "Line",
    "Bet",
    "Platform",
    "Boost",
    "Win Prob",
    "Market Prob",
    "Actual",
    "Scored At",
]
_ROWS = [
    # The same prop on both platforms: two offers.
    ("A", "NBA", "2026-09-30", "PTS", 20.5, "Over", "Underdog", 1.0, 0.62, 0.55, 18.0),
    ("A", "NBA", "2026-09-30", "PTS", 20.5, "Over", "Sleeper", 1.75, 0.62, 0.55, 18.0),
    # B's line moved twice before its game, and M's moved onto a side the platform never
    # posted. Their last value is the run that scored the line; no other row was stamped.
    ("B", "NBA", _GAME, "REB", 10.5, "Under", "Sleeper", 1.75, 0.64, 0.52, 6.0, _RUNS[0]),
    ("B", "NBA", _GAME, "REB", 9.5, "Under", "Sleeper", 1.75, 0.64, 0.52, 6.0, _RUNS[1]),
    ("B", "NBA", _GAME, "REB", 8.5, "Under", "Sleeper", 1.75, 0.64, 0.52, 6.0, _RUNS[2]),
    ("M", "NBA", _GAME, "PTS", 20.5, "Over", "Sleeper", 1.75, 0.62, 0.55, 18.0, _RUNS[1]),
    ("M", "NBA", _GAME, "PTS", 21.5, "Over", "Sleeper", 0.0, 0.62, 0.55, 18.0, _RUNS[2]),
    # C rides an alt line and D is book-priced (both stamped below).
    ("C", "NBA", "2026-09-20", "PTS", 30.5, "Under", "Underdog", 0.8, 0.8, 0.7, 33.0),
    ("D", "WNBA", "2026-09-15", "PTS", 18.5, "Over", "Underdog", 1.0, 0.61, 0.6, 15.0),
    # Bettable, never recommended: a payout past the 2.5x cap, then an edge under 5%.
    ("E", "NBA", "2026-09-25", "PTS", 12.5, "Over", "Sleeper", 3.0, 0.45, 0.33, 15.0),
    ("F", "WNBA", "2026-09-22", "REB", 6.5, "Under", "Sleeper", 1.875, 0.55, 0.5, 5.0),
    # Never a bet: a side the platform did not post, then a push.
    ("G", "NBA", "2026-09-29", "PTS", 20.5, "Over", "Underdog", 0.0, 0.7, 0.55, 25.0),
    ("H", "NBA", "2026-09-27", "PTS", 15.0, "Over", "Sleeper", 1.75, 0.62, 0.55, 15.0),
    # 30 to 90 days before NOW.
    ("I", "WNBA", "2026-08-15", "PTS", 14.5, "Over", "Sleeper", 1.5, 0.72, 0.6, 20.0),
    ("J", "WNBA", "2026-08-20", "AST", 4.5, "Under", "Underdog", 1.0, 0.66, 0.55, 6.0),
    # Older than both ledger windows.
    ("K", "WNBA", "2026-06-20", "PTS", 16.5, "Over", "Sleeper", 1.5, 0.72, 0.6, 20.0),
]
HISTORY = (
    pd.DataFrame(_ROWS, columns=_COLS)
    .assign(
        **{
            "Alt Line": lambda h: h["Player"] == "C",
            "Model Version": lambda h: h["Player"].map({"D": "book_fallback"}).fillna("v1"),
            "Market Projection": 20.0,
        }
    )
    .reindex(columns=HISTORY_COLS)
)

# Who each window and cohort holds: A twice (one offer per platform); B once, at the last
# of its three lines; G (unposted), H (push) and M (last scored on an unposted side) never;
# E (payout 3.0) and F (edge 3%) bettable only; K past 90 days.
_MEMBERS = {
    (30, "recommended"): ["A", "A", "B", "C", "D"],
    (30, "bettable"): ["A", "A", "B", "C", "D", "E", "F"],
    (90, "recommended"): ["A", "A", "B", "C", "D", "I", "J"],
    (90, "bettable"): ["A", "A", "B", "C", "D", "E", "F", "I", "J"],
}


@pytest.mark.parametrize("cohort", ["bettable", "recommended"])
@pytest.mark.parametrize("days", REALIZED_WINDOWS)
def test_cohort_summary_equals_the_ledger_side_rows(days, cohort):
    offers = window(settled_offers(HISTORY), days, NOW)
    if cohort == "recommended":
        offers = offers[offers["Recommended"]]
    page = cohort_summary(offers)
    ledger = compute_realized_by_side(HISTORY, now=NOW)
    sides = ledger[
        (ledger["window_days"] == days)
        & (ledger["cohort"] == cohort)
        & (ledger["split"] == "side")
        & (ledger["key"] == "all")
    ]
    n = sides["n"].sum()

    assert sorted(offers["Player"]) == _MEMBERS[(days, cohort)]
    assert page["n"] == n
    assert page["units"] == sides["units"].sum()
    assert page["hit_rate"] == pytest.approx((sides["hit_rate"] * sides["n"]).sum() / n, abs=1e-12)
    # A side's breakeven is its n over its summed payout, so the sides' payouts sum back.
    assert page["breakeven_rate"] == pytest.approx(
        n / (sides["n"] / sides["breakeven_rate"]).sum(), abs=1e-12
    )


def test_the_two_platform_prop_counts_as_two_offers():
    offers = window(settled_offers(HISTORY), 30, NOW)
    prop = offers[offers["Player"] == "A"]
    assert sorted(prop["Platform"]) == ["Sleeper", "Underdog"]
    assert prop["Recommended"].all()
