"""Exact-pin characterization for ``analysis.explode_offers``.

``test_clv_explode.py`` only checks that the closing-trio columns *surface*
and that legacy 6-tuples pad without crashing; it does not pin the Result
branch logic, the Hit derivation, or the Kelly Model/Books/K columns. This
pins the full exploded frame (every column, every row) so the CC-20
decomposition of ``explode_offers`` stays byte-for-byte behavior-preserving.

Branches covered: standard Over / Under / Push (``actual`` vs ``line``);
matchup ``" vs. "`` Over / Under / Push (sign of ``actual + line``); Underdog
payout (``Boost * 1.78``) vs other-platform (``* 1.0`` → degenerate K = NaN);
legacy 6-tuple padding; a NaN ``Actual`` (Result/Hit NaN but Model/Books/K
still computed); a multi-offer prediction; and an empty ``Offers`` list (folded
in from the former ``test_clv_explode.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sportstradamus.analysis import explode_offers


def _fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Player": "P1",
                "League": "NBA",
                "Date": "2026-05-04",
                "Market": "points",
                "Actual": 22,
                "Offers": [(18.5, 1.0, "Underdog", "Over", 0.60, 0.55, 0.62, 0.07, 0.02)],
            },
            {
                "Player": "P2",
                "League": "NBA",
                "Date": "2026-05-04",
                "Market": "rebounds",
                "Actual": 5,
                "Offers": [(8.5, 1.0, "Sleeper", "Over", 0.58, 0.52, 0.60, 0.05, 0.01)],
            },
            {
                "Player": "P3",
                "League": "NBA",
                "Date": "2026-05-04",
                "Market": "assists",
                "Actual": 10,
                "Offers": [(10.0, 1.0, "Underdog", "Over", 0.50, 0.50, 0.51, 0.0, 0.0)],
            },
            {
                "Player": "A vs. B",
                "League": "NBA",
                "Date": "2026-05-04",
                "Market": "points",
                "Actual": 3,
                "Offers": [(-1.0, 1.0, "Underdog", "Over", 0.55, 0.50, 0.56, 0.06, 0.05)],
            },
            {
                "Player": "C vs. D",
                "League": "NBA",
                "Date": "2026-05-04",
                "Market": "points",
                "Actual": -3,
                "Offers": [(1.0, 1.0, "Underdog", "Under", 0.40, 0.45, 0.41, 0.0, 0.0)],
            },
            {
                "Player": "E vs. F",
                "League": "NBA",
                "Date": "2026-05-04",
                "Market": "points",
                "Actual": 2,
                "Offers": [(-2.0, 1.0, "Underdog", "Over", 0.48, 0.50, 0.49, 0.0, 0.0)],
            },
            {
                "Player": "P7",
                "League": "NBA",
                "Date": "2026-05-04",
                "Market": "points",
                "Actual": 8,
                "Offers": [(5.5, 1.0, "Sleeper", "Over", 0.58, 0.52)],
            },
            {
                "Player": "P8",
                "League": "NBA",
                "Date": "2026-05-04",
                "Market": "points",
                "Actual": np.nan,
                "Offers": [(5.5, 1.0, "Underdog", "Over", 0.58, 0.52, 0.6, 0.0, 0.0)],
            },
            {
                "Player": "P9",
                "League": "NBA",
                "Date": "2026-05-04",
                "Market": "points",
                "Actual": 15,
                "Offers": [
                    (12.5, 1.0, "Underdog", "Over", 0.60, 0.55, 0.61, 0.0, 0.0),
                    (18.5, 1.0, "Underdog", "Under", 0.52, 0.50, 0.53, 0.0, 0.0),
                ],
            },
        ]
    )


_EXPECTED_COLS = [
    "Player",
    "League",
    "Date",
    "Market",
    "Actual",
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
    "Hit",
    "Model",
    "Books",
    "K",
]

_EXPECTED = [
    {
        "Player": "P1",
        "League": "NBA",
        "Date": "2026-05-04",
        "Market": "points",
        "Actual": 22.0,
        "Line": 18.5,
        "Boost": 1.0,
        "Platform": "Underdog",
        "Bet": "Over",
        "Model P": 0.6,
        "Books P": 0.55,
        "Close Books P": 0.62,
        "Market CLV": 0.07,
        "Model CLV": 0.02,
        "Result": "Over",
        "Hit": 1.0,
        "Model": 1.068,
        "Books": 0.9790000000000001,
        "K": 0.08717948717948726,
    },
    {
        "Player": "P2",
        "League": "NBA",
        "Date": "2026-05-04",
        "Market": "rebounds",
        "Actual": 5.0,
        "Line": 8.5,
        "Boost": 1.0,
        "Platform": "Sleeper",
        "Bet": "Over",
        "Model P": 0.58,
        "Books P": 0.52,
        "Close Books P": 0.6,
        "Market CLV": 0.05,
        "Model CLV": 0.01,
        "Result": "Under",
        "Hit": 0.0,
        "Model": 0.58,
        "Books": 0.52,
        "K": float("nan"),
    },
    {
        "Player": "P3",
        "League": "NBA",
        "Date": "2026-05-04",
        "Market": "assists",
        "Actual": 10.0,
        "Line": 10.0,
        "Boost": 1.0,
        "Platform": "Underdog",
        "Bet": "Over",
        "Model P": 0.5,
        "Books P": 0.5,
        "Close Books P": 0.51,
        "Market CLV": 0.0,
        "Model CLV": 0.0,
        "Result": "Push",
        "Hit": 0.0,
        "Model": 0.89,
        "Books": 0.89,
        "K": -0.141025641025641,
    },
    {
        "Player": "A vs. B",
        "League": "NBA",
        "Date": "2026-05-04",
        "Market": "points",
        "Actual": 3.0,
        "Line": -1.0,
        "Boost": 1.0,
        "Platform": "Underdog",
        "Bet": "Over",
        "Model P": 0.55,
        "Books P": 0.5,
        "Close Books P": 0.56,
        "Market CLV": 0.06,
        "Model CLV": 0.05,
        "Result": "Over",
        "Hit": 1.0,
        "Model": 0.9790000000000001,
        "Books": 0.89,
        "K": -0.026923076923076803,
    },
    {
        "Player": "C vs. D",
        "League": "NBA",
        "Date": "2026-05-04",
        "Market": "points",
        "Actual": -3.0,
        "Line": 1.0,
        "Boost": 1.0,
        "Platform": "Underdog",
        "Bet": "Under",
        "Model P": 0.4,
        "Books P": 0.45,
        "Close Books P": 0.41,
        "Market CLV": 0.0,
        "Model CLV": 0.0,
        "Result": "Under",
        "Hit": 1.0,
        "Model": 0.7120000000000001,
        "Books": 0.801,
        "K": -0.36923076923076914,
    },
    {
        "Player": "E vs. F",
        "League": "NBA",
        "Date": "2026-05-04",
        "Market": "points",
        "Actual": 2.0,
        "Line": -2.0,
        "Boost": 1.0,
        "Platform": "Underdog",
        "Bet": "Over",
        "Model P": 0.48,
        "Books P": 0.5,
        "Close Books P": 0.49,
        "Market CLV": 0.0,
        "Model CLV": 0.0,
        "Result": "Push",
        "Hit": 0.0,
        "Model": 0.8543999999999999,
        "Books": 0.89,
        "K": -0.18666666666666673,
    },
    {
        "Player": "P7",
        "League": "NBA",
        "Date": "2026-05-04",
        "Market": "points",
        "Actual": 8.0,
        "Line": 5.5,
        "Boost": 1.0,
        "Platform": "Sleeper",
        "Bet": "Over",
        "Model P": 0.58,
        "Books P": 0.52,
        "Close Books P": float("nan"),
        "Market CLV": float("nan"),
        "Model CLV": float("nan"),
        "Result": "Over",
        "Hit": 1.0,
        "Model": 0.58,
        "Books": 0.52,
        "K": float("nan"),
    },
    {
        "Player": "P8",
        "League": "NBA",
        "Date": "2026-05-04",
        "Market": "points",
        "Actual": float("nan"),
        "Line": 5.5,
        "Boost": 1.0,
        "Platform": "Underdog",
        "Bet": "Over",
        "Model P": 0.58,
        "Books P": 0.52,
        "Close Books P": 0.6,
        "Market CLV": 0.0,
        "Model CLV": 0.0,
        "Result": float("nan"),
        "Hit": float("nan"),
        "Model": 1.0324,
        "Books": 0.9256000000000001,
        "K": 0.04153846153846152,
    },
    {
        "Player": "P9",
        "League": "NBA",
        "Date": "2026-05-04",
        "Market": "points",
        "Actual": 15.0,
        "Line": 12.5,
        "Boost": 1.0,
        "Platform": "Underdog",
        "Bet": "Over",
        "Model P": 0.6,
        "Books P": 0.55,
        "Close Books P": 0.61,
        "Market CLV": 0.0,
        "Model CLV": 0.0,
        "Result": "Over",
        "Hit": 1.0,
        "Model": 1.068,
        "Books": 0.9790000000000001,
        "K": 0.08717948717948726,
    },
    {
        "Player": "P9",
        "League": "NBA",
        "Date": "2026-05-04",
        "Market": "points",
        "Actual": 15.0,
        "Line": 18.5,
        "Boost": 1.0,
        "Platform": "Underdog",
        "Bet": "Under",
        "Model P": 0.52,
        "Books P": 0.5,
        "Close Books P": 0.53,
        "Market CLV": 0.0,
        "Model CLV": 0.0,
        "Result": "Under",
        "Hit": 1.0,
        "Model": 0.9256000000000001,
        "Books": 0.89,
        "K": -0.09538461538461526,
    },
]


def test_explode_offers_full_exact_pin():
    out = explode_offers(_fixture())
    assert list(out.columns) == _EXPECTED_COLS
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True),
        pd.DataFrame(_EXPECTED),
        check_dtype=False,
    )


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
