"""Characterization pin for ``prediction.cli._merge_prediction_row``.

Extracted from ``main``'s history-merge loop in the nesting sweep. Pins the two
paths: an existing pred-key row has its Offers merged (via ``_merge_offers``)
and its prediction-level columns overwritten only where the new value is
non-NaN; a new pred-key row is inserted whole.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sportstradamus.history_schema import PREDICTION_LEVEL_COLS
from sportstradamus.prediction.cli import _merge_prediction_row

_PRED_KEY = ["Player", "League", "Date", "Market"]


def _frame(rows):
    df = pd.DataFrame(rows)
    for col in PREDICTION_LEVEL_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df.set_index(_PRED_KEY)


def test_merge_prediction_row_update_merges_offers_and_overwrites_non_nan():
    history = _frame(
        [
            {
                "Player": "P1", "League": "NBA", "Date": "2026-05-04", "Market": "points",
                "Offers": [(18.5, 1.0, "Underdog", "Over", 0.6, 0.55, np.nan, np.nan, np.nan)],
                "Team": "BOS", "Model EV": 0.1, "Dist": "Poisson",
            }
        ]
    )
    new_df = _frame(
        [
            {
                "Player": "P1", "League": "NBA", "Date": "2026-05-04", "Market": "points",
                "Offers": [(20.5, 1.0, "Sleeper", "Under", 0.5, 0.5, np.nan, np.nan, np.nan)],
                "Team": "BOS", "Model EV": 0.2, "Dist": np.nan,  # NaN -> must NOT overwrite
            }
        ]
    )
    idx = ("P1", "NBA", "2026-05-04", "points")

    _merge_prediction_row(history, new_df, idx)

    merged = history.at[idx, "Offers"]
    assert {(o[0], o[2]) for o in merged} == {(18.5, "Underdog"), (20.5, "Sleeper")}
    assert history.at[idx, "Model EV"] == 0.2     # non-NaN -> overwritten
    assert history.at[idx, "Dist"] == "Poisson"   # new value NaN -> original kept


def test_merge_prediction_row_inserts_new_index():
    history = _frame(
        [
            {
                "Player": "P1", "League": "NBA", "Date": "2026-05-04", "Market": "points",
                "Offers": [(18.5, 1.0, "Underdog", "Over", 0.6, 0.55, np.nan, np.nan, np.nan)],
                "Team": "BOS", "Model EV": 0.1, "Dist": "Poisson",
            }
        ]
    )
    new_df = _frame(
        [
            {
                "Player": "P2", "League": "NBA", "Date": "2026-05-04", "Market": "rebounds",
                "Offers": [(8.5, 1.0, "Underdog", "Over", 0.58, 0.52, np.nan, np.nan, np.nan)],
                "Team": "MIA", "Model EV": 0.05, "Dist": "Poisson",
            }
        ]
    )
    idx = ("P2", "NBA", "2026-05-04", "rebounds")
    assert idx not in history.index

    _merge_prediction_row(history, new_df, idx)

    assert idx in history.index
    assert history.at[idx, "Team"] == "MIA"
    assert history.at[idx, "Model EV"] == 0.05
