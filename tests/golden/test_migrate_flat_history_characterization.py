"""Characterization pin for ``analysis._migrate_flat_history``.

Converts the legacy flat history schema (one row per offer) to the normalized
one-row-per-prediction schema with an ``Offers`` list of 9-tuples. No behavioral
coverage exists. This pins: the (Player, League, Date, Market) grouping, the
9-tuple build with the NaN-padded closing trio, the Model P / Books P derivation
from the old boosted ``Model`` / ``Books`` ÷ ``Boost``, the (Line, Platform)
dedup keeping the last occurrence, the NaN-``Line`` skip, and prediction-level
columns taken from the group's last row.

Dates are fixed literals — ``_migrate_flat_history`` has no date logic.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from sportstradamus.analysis import _migrate_flat_history


def _build_flat() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Player": "P1",
                "League": "NBA",
                "Date": "2026-05-04",
                "Market": "points",
                "Line": 18.5,
                "Boost": 1.0,
                "Platform": "Underdog",
                "Bet": "Over",
                "Model EV": 0.60,
                "Market EV": 0.55,
                "Team": "BOS",
                "Projection": 0.10,
                "Dist": "Poisson",
            },
            {
                "Player": "P1",
                "League": "NBA",
                "Date": "2026-05-04",
                "Market": "points",
                "Line": 20.5,
                "Boost": 2.0,
                "Platform": "Underdog",
                "Bet": "Under",
                "Model EV": 1.10,
                "Market EV": 1.00,
                "Team": "BOS",
                "Projection": 0.12,
                "Dist": "Poisson",
            },
            {
                "Player": "P1",
                "League": "NBA",
                "Date": "2026-05-04",
                "Market": "points",
                "Line": 18.5,
                "Boost": 1.0,
                "Platform": "Underdog",
                "Bet": "Over",
                "Model EV": 0.62,
                "Market EV": 0.56,
                "Team": "BOS",
                "Projection": 0.15,
                "Dist": "Poisson",
            },
            {
                "Player": "P1",
                "League": "NBA",
                "Date": "2026-05-04",
                "Market": "points",
                "Line": np.nan,
                "Boost": 1.0,
                "Platform": "Sleeper",
                "Bet": "Over",
                "Model EV": 0.50,
                "Market EV": 0.50,
                "Team": "BOS",
                "Projection": 0.0,
                "Dist": "Poisson",
            },
            {
                "Player": "P2",
                "League": "NBA",
                "Date": "2026-05-04",
                "Market": "rebounds",
                "Line": 8.5,
                "Boost": 1.0,
                "Platform": "Sleeper",
                "Bet": "Over",
                "Model EV": 0.58,
                "Market EV": 0.52,
                "Team": "MIA",
                "Projection": 0.05,
                "Dist": "Poisson",
            },
        ]
    )


_EXPECTED_OFFERS = {
    "P1": [
        (18.5, 1.0, "Underdog", "Over", 0.62, 0.56, np.nan, np.nan, np.nan),
        (20.5, 2.0, "Underdog", "Under", 0.55, 0.5, np.nan, np.nan, np.nan),
    ],
    "P2": [
        (8.5, 1.0, "Sleeper", "Over", 0.58, 0.52, np.nan, np.nan, np.nan),
    ],
}


def _offer_eq(a: tuple, b: tuple) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b, strict=True):
        if isinstance(x, float) and math.isnan(x):
            if not (isinstance(y, float) and math.isnan(y)):
                return False
        elif x != y:
            return False
    return True


def _offers_eq(actual: list, expected: list) -> bool:
    return len(actual) == len(expected) and all(
        _offer_eq(a, e) for a, e in zip(actual, expected, strict=True)
    )


def test_migrate_flat_history_normalizes_offers():
    out = _migrate_flat_history(_build_flat())

    assert len(out) == 2
    assert list(out.columns) == [
        "Player",
        "League",
        "Date",
        "Market",
        "Offers",
        "Team",
        "Projection",
        "Market Projection",
        "Dist",
        "CV",
        "Model Param",
        "Gate",
        "Temperature",
        "Disp Cal",
        "Step",
        "Actual",
    ]

    by_player = {r["Player"]: r for _, r in out.iterrows()}
    for player, expected in _EXPECTED_OFFERS.items():
        assert _offers_eq(by_player[player]["Offers"], expected), player

    # Prediction-level cols come from the group's LAST row (P1's last row is the
    # NaN-Line Sleeper row with Model EV 0.0).
    assert by_player["P1"]["Team"] == "BOS"
    assert by_player["P1"]["Projection"] == 0.0
    assert by_player["P1"]["Dist"] == "Poisson"
    assert math.isnan(by_player["P1"]["Actual"])
    assert by_player["P2"]["Team"] == "MIA"
    assert by_player["P2"]["Projection"] == 0.05
