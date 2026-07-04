"""Round-trip pin for the flat history parquet store (P8 Task 0.7).

``write_history``/``read_history`` are now plain ``_atomic_write_parquet`` /
``read_parquet_safe`` calls — no tuple<->dict offer codec. This pins that a
flat one-row-per-(prediction x book offer) frame with mixed dtypes (str,
float, NaN) survives a parquet round-trip unchanged, and that a missing file
reads back empty.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sportstradamus.helpers import io


def _patch_path(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "HISTORY_PATH", tmp_path / "history.parquet")


def _flat_history():
    return pd.DataFrame(
        [
            {
                "Player": "Player A",
                "League": "NBA",
                "Team": "BOS",
                "Date": "2026-05-04",
                "Market": "points",
                "Projection": 24.5,
                "Dist": "SkewNormal",
                "Line": 24.5,
                "Boost": 1.0,
                "Platform": "Underdog",
                "Bet": "Over",
                "Win Prob": 0.58,
                "Market Prob": 0.52,
                "Close Market Prob": np.nan,
                "Market CLV": np.nan,
                "Model CLV": np.nan,
                "Alt Line": False,
                "Actual": np.nan,
            },
            {
                "Player": "Player A",
                "League": "NBA",
                "Team": "BOS",
                "Date": "2026-05-04",
                "Market": "points",
                "Projection": 24.5,
                "Dist": "SkewNormal",
                "Line": 26.5,
                "Boost": 1.0,
                "Platform": "Sleeper",
                "Bet": "Under",
                "Win Prob": 0.55,
                "Market Prob": 0.50,
                "Close Market Prob": 0.60,
                "Market CLV": 0.10,
                "Model CLV": 0.05,
                "Alt Line": True,
                "Actual": 22.0,
            },
        ]
    )


def test_read_history_returns_empty_when_absent(tmp_path, monkeypatch):
    _patch_path(tmp_path, monkeypatch)
    assert io.read_history().empty


def test_write_then_read_preserves_flat_shape_and_values(tmp_path, monkeypatch):
    _patch_path(tmp_path, monkeypatch)
    original = _flat_history()
    io.write_history(original)
    back = io.read_history()

    assert len(back) == len(original) == 2
    assert "Offers" not in back.columns
    assert list(back.columns) == list(original.columns)
    assert back.loc[0, "Line"] == 24.5
    assert back.loc[0, "Platform"] == "Underdog"
    assert np.isnan(back.loc[0, "Close Market Prob"])
    assert back.loc[1, "Line"] == 26.5
    assert back.loc[1, "Close Market Prob"] == 0.60
    assert bool(back.loc[1, "Alt Line"]) is True


def test_two_offer_rows_for_same_prediction_key_both_persist(tmp_path, monkeypatch):
    """The whole point of the flat schema: same (Player, League, Date, Market),
    two book offers -> two rows, not one row with a nested Offers list."""
    _patch_path(tmp_path, monkeypatch)
    io.write_history(_flat_history())
    back = io.read_history()
    key_cols = ["Player", "League", "Date", "Market"]
    assert back.drop_duplicates(subset=key_cols).shape[0] == 1
    assert back.shape[0] == 2
