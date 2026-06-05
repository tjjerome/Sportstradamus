"""Characterization pin for ``analysis.compute_parlay_metrics``.

Captures the four output frames (profit_df, daily_parlays, size_stats,
corr_cal) for a synthetic resolved-parlay slate so the CC-22 decomposition
of ``compute_parlay_metrics`` is byte-for-byte behavior-preserving. The
function has no other behavioral coverage (``test_analysis.py`` only covers
``compute_book_brier_skill_score``); the dashboard reflect path is the sole
production caller.

Dates are anchored to *today* minus a fixed offset so timeframe-bucket
membership (7d / 30d / 3m / 6m / 1y) and the 365-day pre-filter are stable
across CI days. The only date-bearing output column (``daily_parlays.Date``)
is rebuilt from the same offsets in-test — an independent reconstruction, not
a circular call back through the function under test.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from sportstradamus.analysis import compute_parlay_metrics

REF = datetime.today().date()


def _d(days_ago: int) -> str:
    return str(REF - timedelta(days=days_ago))


def _build_parlays() -> pd.DataFrame:
    """Resolved slate: Underdog + Sleeper, two leagues, two date offsets.

    Rows 1 & 4 share (Game, Date) to exercise the profit groupby-mean; the
    WNBA row sits at REF-100 so it lands only in the 6m/1y frames.
    """
    rows = [
        {
            "Date": _d(3),
            "League": "NBA",
            "Platform": "Underdog",
            "Game": "BOS/MIA",
            "Legs": 2,
            "Misses": 0,
            "Boost": 1.0,
            "Rec Bet": 1.0,
            "Model EV": 0.10,
            "Bet Size": 2,
            "P": 0.40,
            "Indep P": 0.42,
        },
        {
            "Date": _d(3),
            "League": "NBA",
            "Platform": "Underdog",
            "Game": "LAL/DEN",
            "Legs": 3,
            "Misses": 1,
            "Boost": 1.0,
            "Rec Bet": 1.5,
            "Model EV": 0.20,
            "Bet Size": 3,
            "P": 0.30,
            "Indep P": 0.28,
        },
        {
            "Date": _d(100),
            "League": "WNBA",
            "Platform": "Underdog",
            "Game": "NY/LV",
            "Legs": 2,
            "Misses": 0,
            "Boost": 2.5,
            "Rec Bet": 1.0,
            "Model EV": 0.15,
            "Bet Size": 2,
            "P": 0.55,
            "Indep P": 0.50,
        },
        {
            "Date": _d(3),
            "League": "NBA",
            "Platform": "Underdog",
            "Game": "BOS/MIA",
            "Legs": 2,
            "Misses": 0,
            "Boost": 1.0,
            "Rec Bet": 1.0,
            "Model EV": 0.05,
            "Bet Size": 2,
            "P": 0.45,
            "Indep P": 0.47,
        },
        {
            "Date": _d(3),
            "League": "NBA",
            "Platform": "Sleeper",
            "Game": "MIL/PHI",
            "Legs": 2,
            "Misses": 0,
            "Boost": 1.0,
            "Rec Bet": 1.0,
            "Model EV": 0.10,
            "Bet Size": 2,
            "P": 0.60,
            "Indep P": 0.58,
        },
        {
            "Date": _d(3),
            "League": "NBA",
            "Platform": "Sleeper",
            "Game": "GSW/SAC",
            "Legs": 3,
            "Misses": 2,
            "Boost": 1.0,
            "Rec Bet": 1.0,
            "Model EV": 0.05,
            "Bet Size": 3,
            "P": 0.20,
            "Indep P": 0.19,
        },
    ]
    return pd.DataFrame(rows)


_EXPECTED_PROFIT = [
    {
        "Platform": "Underdog",
        "League": "All",
        "Period": "7d",
        "Profit": 1.0,
        "Parlays": 3,
        "Hit Rate": 0.6667,
    },
    {
        "Platform": "Underdog",
        "League": "All",
        "Period": "30d",
        "Profit": 1.0,
        "Parlays": 3,
        "Hit Rate": 0.6667,
    },
    {
        "Platform": "Underdog",
        "League": "All",
        "Period": "3m",
        "Profit": 1.0,
        "Parlays": 3,
        "Hit Rate": 0.6667,
    },
    {
        "Platform": "Underdog",
        "League": "All",
        "Period": "6m",
        "Profit": 8.75,
        "Parlays": 4,
        "Hit Rate": 0.75,
    },
    {
        "Platform": "Underdog",
        "League": "All",
        "Period": "1y",
        "Profit": 8.75,
        "Parlays": 4,
        "Hit Rate": 0.75,
    },
    {
        "Platform": "Underdog",
        "League": "NBA",
        "Period": "7d",
        "Profit": 1.0,
        "Parlays": 3,
        "Hit Rate": 0.6667,
    },
    {
        "Platform": "Underdog",
        "League": "NBA",
        "Period": "30d",
        "Profit": 1.0,
        "Parlays": 3,
        "Hit Rate": 0.6667,
    },
    {
        "Platform": "Underdog",
        "League": "NBA",
        "Period": "3m",
        "Profit": 1.0,
        "Parlays": 3,
        "Hit Rate": 0.6667,
    },
    {
        "Platform": "Underdog",
        "League": "NBA",
        "Period": "6m",
        "Profit": 1.0,
        "Parlays": 3,
        "Hit Rate": 0.6667,
    },
    {
        "Platform": "Underdog",
        "League": "NBA",
        "Period": "1y",
        "Profit": 1.0,
        "Parlays": 3,
        "Hit Rate": 0.6667,
    },
    {
        "Platform": "Underdog",
        "League": "WNBA",
        "Period": "6m",
        "Profit": 7.75,
        "Parlays": 1,
        "Hit Rate": 1.0,
    },
    {
        "Platform": "Underdog",
        "League": "WNBA",
        "Period": "1y",
        "Profit": 7.75,
        "Parlays": 1,
        "Hit Rate": 1.0,
    },
]

_EXPECTED_SIZE = [
    {
        "Platform": "Underdog",
        "Period": "7d",
        "Size": 2,
        "Actual Rate": 1.0,
        "Hit All": 2,
        "Missed 1": 0,
        "Missed 2+": 0,
        "Count": 2,
        "Predicted P": 0.425,
        "Independent Rate": 0.445,
    },
    {
        "Platform": "Underdog",
        "Period": "7d",
        "Size": 3,
        "Actual Rate": 0.0,
        "Hit All": 0,
        "Missed 1": 1,
        "Missed 2+": 0,
        "Count": 1,
        "Predicted P": 0.3,
        "Independent Rate": 0.28,
    },
    {
        "Platform": "Underdog",
        "Period": "30d",
        "Size": 2,
        "Actual Rate": 1.0,
        "Hit All": 2,
        "Missed 1": 0,
        "Missed 2+": 0,
        "Count": 2,
        "Predicted P": 0.425,
        "Independent Rate": 0.445,
    },
    {
        "Platform": "Underdog",
        "Period": "30d",
        "Size": 3,
        "Actual Rate": 0.0,
        "Hit All": 0,
        "Missed 1": 1,
        "Missed 2+": 0,
        "Count": 1,
        "Predicted P": 0.3,
        "Independent Rate": 0.28,
    },
    {
        "Platform": "Underdog",
        "Period": "3m",
        "Size": 2,
        "Actual Rate": 1.0,
        "Hit All": 2,
        "Missed 1": 0,
        "Missed 2+": 0,
        "Count": 2,
        "Predicted P": 0.425,
        "Independent Rate": 0.445,
    },
    {
        "Platform": "Underdog",
        "Period": "3m",
        "Size": 3,
        "Actual Rate": 0.0,
        "Hit All": 0,
        "Missed 1": 1,
        "Missed 2+": 0,
        "Count": 1,
        "Predicted P": 0.3,
        "Independent Rate": 0.28,
    },
    {
        "Platform": "Underdog",
        "Period": "6m",
        "Size": 2,
        "Actual Rate": 1.0,
        "Hit All": 3,
        "Missed 1": 0,
        "Missed 2+": 0,
        "Count": 3,
        "Predicted P": 0.4667,
        "Independent Rate": 0.4633,
    },
    {
        "Platform": "Underdog",
        "Period": "6m",
        "Size": 3,
        "Actual Rate": 0.0,
        "Hit All": 0,
        "Missed 1": 1,
        "Missed 2+": 0,
        "Count": 1,
        "Predicted P": 0.3,
        "Independent Rate": 0.28,
    },
    {
        "Platform": "Underdog",
        "Period": "1y",
        "Size": 2,
        "Actual Rate": 1.0,
        "Hit All": 3,
        "Missed 1": 0,
        "Missed 2+": 0,
        "Count": 3,
        "Predicted P": 0.4667,
        "Independent Rate": 0.4633,
    },
    {
        "Platform": "Underdog",
        "Period": "1y",
        "Size": 3,
        "Actual Rate": 0.0,
        "Hit All": 0,
        "Missed 1": 1,
        "Missed 2+": 0,
        "Count": 1,
        "Predicted P": 0.3,
        "Independent Rate": 0.28,
    },
    {
        "Platform": "Sleeper",
        "Period": "7d",
        "Size": 2,
        "Actual Rate": 1.0,
        "Hit All": 1,
        "Missed 1": 0,
        "Missed 2+": 0,
        "Count": 1,
        "Predicted P": 0.6,
        "Independent Rate": 0.58,
    },
    {
        "Platform": "Sleeper",
        "Period": "7d",
        "Size": 3,
        "Actual Rate": 0.0,
        "Hit All": 0,
        "Missed 1": 0,
        "Missed 2+": 1,
        "Count": 1,
        "Predicted P": 0.2,
        "Independent Rate": 0.19,
    },
    {
        "Platform": "Sleeper",
        "Period": "30d",
        "Size": 2,
        "Actual Rate": 1.0,
        "Hit All": 1,
        "Missed 1": 0,
        "Missed 2+": 0,
        "Count": 1,
        "Predicted P": 0.6,
        "Independent Rate": 0.58,
    },
    {
        "Platform": "Sleeper",
        "Period": "30d",
        "Size": 3,
        "Actual Rate": 0.0,
        "Hit All": 0,
        "Missed 1": 0,
        "Missed 2+": 1,
        "Count": 1,
        "Predicted P": 0.2,
        "Independent Rate": 0.19,
    },
    {
        "Platform": "Sleeper",
        "Period": "3m",
        "Size": 2,
        "Actual Rate": 1.0,
        "Hit All": 1,
        "Missed 1": 0,
        "Missed 2+": 0,
        "Count": 1,
        "Predicted P": 0.6,
        "Independent Rate": 0.58,
    },
    {
        "Platform": "Sleeper",
        "Period": "3m",
        "Size": 3,
        "Actual Rate": 0.0,
        "Hit All": 0,
        "Missed 1": 0,
        "Missed 2+": 1,
        "Count": 1,
        "Predicted P": 0.2,
        "Independent Rate": 0.19,
    },
    {
        "Platform": "Sleeper",
        "Period": "6m",
        "Size": 2,
        "Actual Rate": 1.0,
        "Hit All": 1,
        "Missed 1": 0,
        "Missed 2+": 0,
        "Count": 1,
        "Predicted P": 0.6,
        "Independent Rate": 0.58,
    },
    {
        "Platform": "Sleeper",
        "Period": "6m",
        "Size": 3,
        "Actual Rate": 0.0,
        "Hit All": 0,
        "Missed 1": 0,
        "Missed 2+": 1,
        "Count": 1,
        "Predicted P": 0.2,
        "Independent Rate": 0.19,
    },
    {
        "Platform": "Sleeper",
        "Period": "1y",
        "Size": 2,
        "Actual Rate": 1.0,
        "Hit All": 1,
        "Missed 1": 0,
        "Missed 2+": 0,
        "Count": 1,
        "Predicted P": 0.6,
        "Independent Rate": 0.58,
    },
    {
        "Platform": "Sleeper",
        "Period": "1y",
        "Size": 3,
        "Actual Rate": 0.0,
        "Hit All": 0,
        "Missed 1": 0,
        "Missed 2+": 1,
        "Count": 1,
        "Predicted P": 0.2,
        "Independent Rate": 0.19,
    },
]

_EXPECTED_CORR = [
    {"Bin": "(0.0, 0.1]", "Predicted": float("nan"), "Actual": float("nan"), "Count": 0},
    {"Bin": "(0.1, 0.2]", "Predicted": 0.2, "Actual": 0.0, "Count": 1},
    {"Bin": "(0.2, 0.3]", "Predicted": 0.3, "Actual": 0.0, "Count": 1},
    {"Bin": "(0.3, 0.4]", "Predicted": 0.4, "Actual": 1.0, "Count": 1},
    {"Bin": "(0.4, 0.5]", "Predicted": 0.45, "Actual": 1.0, "Count": 1},
    {"Bin": "(0.5, 0.6]", "Predicted": 0.575, "Actual": 1.0, "Count": 2},
    {"Bin": "(0.6, 0.7]", "Predicted": float("nan"), "Actual": float("nan"), "Count": 0},
    {"Bin": "(0.7, 0.8]", "Predicted": float("nan"), "Actual": float("nan"), "Count": 0},
    {"Bin": "(0.8, 0.9]", "Predicted": float("nan"), "Actual": float("nan"), "Count": 0},
    {"Bin": "(0.9, 1.0]", "Predicted": float("nan"), "Actual": float("nan"), "Count": 0},
]


def _expected_daily() -> list[dict]:
    """Rebuild the date-bearing daily table from the same offsets used above."""
    d3, d100 = _d(3), _d(100)
    return [
        {
            "Date": d100,
            "Platform": "Underdog",
            "League": "WNBA",
            "Parlays": 1,
            "Hits": 1,
            "Misses_0": 1,
            "Misses_1": 0,
            "Misses_2_plus": 0,
        },
        {
            "Date": d3,
            "Platform": "Sleeper",
            "League": "NBA",
            "Parlays": 2,
            "Hits": 1,
            "Misses_0": 1,
            "Misses_1": 0,
            "Misses_2_plus": 1,
        },
        {
            "Date": d3,
            "Platform": "Underdog",
            "League": "NBA",
            "Parlays": 3,
            "Hits": 2,
            "Misses_0": 2,
            "Misses_1": 1,
            "Misses_2_plus": 0,
        },
    ]


def _assert(actual: pd.DataFrame, expected: list[dict]) -> None:
    expected_df = pd.DataFrame(expected)
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected_df, check_dtype=False)


def test_compute_parlay_metrics_full_slate():
    profit_df, daily_parlays, size_stats, corr_cal = compute_parlay_metrics(
        _build_parlays(), {}, {}
    )
    _assert(profit_df, _EXPECTED_PROFIT)
    _assert(daily_parlays, _expected_daily())
    _assert(size_stats, _EXPECTED_SIZE)
    _assert(corr_cal, _EXPECTED_CORR)


def _build_legprobs() -> pd.DataFrame:
    """Slate with ``Leg Probs`` and no ``Indep P`` / ``P`` columns."""
    return pd.DataFrame(
        [
            {
                "Date": _d(3),
                "League": "NBA",
                "Platform": "Sleeper",
                "Game": "BOS/MIA",
                "Legs": 2,
                "Misses": 0,
                "Boost": 1.0,
                "Rec Bet": 1.0,
                "Model EV": 0.10,
                "Bet Size": 2,
                "Leg Probs": [0.6, 0.7],
            },
            {
                "Date": _d(3),
                "League": "NBA",
                "Platform": "Sleeper",
                "Game": "LAL/DEN",
                "Legs": 2,
                "Misses": 1,
                "Boost": 1.0,
                "Rec Bet": 1.0,
                "Model EV": 0.10,
                "Bet Size": 2,
                "Leg Probs": [0.5, 0.5],
            },
        ]
    )


_EXPECTED_LEGPROBS_SIZE = [
    {
        "Platform": "Sleeper",
        "Period": p,
        "Size": 2,
        "Actual Rate": 0.5,
        "Hit All": 1,
        "Missed 1": 1,
        "Missed 2+": 0,
        "Count": 2,
        "Independent Rate": 0.335,
    }
    for p in ("7d", "30d", "3m", "6m", "1y")
]


def test_compute_parlay_metrics_leg_probs_branch():
    """``Independent Rate`` falls back to the product of ``Leg Probs`` when
    there is no ``Indep P`` column; absent ``P`` leaves corr_cal/profit empty."""
    profit_df, _daily, size_stats, corr_cal = compute_parlay_metrics(_build_legprobs(), {}, {})
    _assert(size_stats, _EXPECTED_LEGPROBS_SIZE)
    assert corr_cal.empty
    assert profit_df.empty
    assert "Predicted P" not in size_stats.columns


def test_compute_parlay_metrics_empty_when_all_stale():
    """Every parlay older than the 365-day window → four empty frames."""
    stale = pd.DataFrame(
        [
            {
                "Date": _d(400),
                "League": "NBA",
                "Platform": "Underdog",
                "Game": "BOS/MIA",
                "Legs": 2,
                "Misses": 0,
                "Boost": 1.0,
                "Rec Bet": 1.0,
                "Model EV": 0.10,
                "Bet Size": 2,
                "P": 0.5,
            },
        ]
    )
    result = compute_parlay_metrics(stale, {}, {})
    assert all(df.empty for df in result)
