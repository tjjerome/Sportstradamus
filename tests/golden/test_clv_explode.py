"""Snapshot test for ``analysis.explode_offers`` with the 9-tuple Offers shape.

Verifies that the closing trio (Close Books P, Market CLV, Model CLV)
surfaces as columns alongside the open Model P / Books P / Bet etc.
Also confirms backward compatibility: legacy 6-tuple offers explode with
NaN closing fields rather than crashing.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from sportstradamus.analysis import explode_offers


def _build_fixture():
    return pd.DataFrame(
        [
            {
                "Player": "Player A",
                "League": "NBA",
                "Team": "BOS",
                "Date": "2026-05-04",
                "Market": "points",
                "Actual": 22,
                "Offers": [
                    # Resolved 9-tuple with full CLV trio
                    (
                        18.5,
                        1.0,
                        "Underdog",
                        "Over",
                        0.60,
                        0.55,
                        0.62,
                        0.07,
                        0.02,
                    ),
                ],
            },
            {
                "Player": "Player B",
                "League": "NBA",
                "Team": "MIA",
                "Date": "2026-05-04",
                "Market": "rebounds",
                "Actual": 8,
                "Offers": [
                    # Legacy 6-tuple — must pad to NaN closing trio.
                    (5.5, 1.0, "Sleeper", "Over", 0.58, 0.52),
                ],
            },
        ]
    )


def test_explode_offers_surfaces_clv_columns():
    df = explode_offers(_build_fixture())

    expected_cols = {
        "Line",
        "Boost",
        "Platform",
        "Bet",
        "Model P",
        "Books P",
        "Close Books P",
        "Market CLV",
        "Model CLV",
        "Result",
    }
    assert expected_cols.issubset(df.columns)


def test_explode_offers_preserves_resolved_clv():
    df = explode_offers(_build_fixture())
    row = df.loc[df["Player"] == "Player A"].iloc[0]
    assert row["Close Books P"] == 0.62
    assert row["Market CLV"] == 0.07
    assert row["Model CLV"] == 0.02
    assert row["Result"] == "Over"


def test_explode_offers_pads_legacy_six_tuple_with_nan():
    df = explode_offers(_build_fixture())
    row = df.loc[df["Player"] == "Player B"].iloc[0]
    assert math.isnan(row["Close Books P"])
    assert math.isnan(row["Market CLV"])
    assert math.isnan(row["Model CLV"])
    assert row["Bet"] == "Over"
    assert row["Model P"] == 0.58


def test_explode_offers_handles_empty_offers_list():
    df = pd.DataFrame(
        [
            {
                "Player": "Empty",
                "League": "NBA",
                "Date": "2026-05-04",
                "Market": "points",
                "Actual": np.nan,
                "Offers": [],
            }
        ]
    )
    out = explode_offers(df)
    assert out.empty
