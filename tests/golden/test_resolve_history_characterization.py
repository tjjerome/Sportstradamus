"""Characterization pin for ``analysis.resolve_history``.

``resolve_history`` fills the ``Actual`` column on history rows whose game has
already happened; the dashboard ``reflect`` path is its only caller and the
integration suite stubs that, so it has no behavioral coverage. This pins the
single / ``" + "`` (sum) / ``" vs. "`` (difference) player resolution plus every
skip path (no game that date, NaN market value, future date, missing market
column → KeyError, league absent from ``stats``, already-resolved row).

Dates are today-anchored so the ``date < today`` pending filter is stable
across CI days. The expected Actual values come from the synthetic gamelog the
test defines — an independent reconstruction, not a call back through the
function under test.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from sportstradamus.analysis import resolve_history

REF = datetime.today().date()
_D3 = str(REF - timedelta(days=3))
_TODAY = str(REF)


class _FakeStats:
    def __init__(self, gamelog: pd.DataFrame):
        self.gamelog = gamelog
        self.log_strings = {"player": "playerName", "date": "gameDate"}


def _stats() -> dict:
    gamelog = pd.DataFrame(
        [
            {"playerName": "A", "gameDate": _D3, "points": 15.0},
            {"playerName": "B", "gameDate": _D3, "points": 8.0},
            {"playerName": "C", "gameDate": _D3, "points": np.nan},
        ]
    )
    return {"NBA": _FakeStats(gamelog)}


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"League": "NBA", "Player": "A", "Market": "points", "Date": _D3},
            {"League": "NBA", "Player": "A + B", "Market": "points", "Date": _D3},
            {"League": "NBA", "Player": "A vs. B", "Market": "points", "Date": _D3},
            {"League": "NBA", "Player": "D", "Market": "points", "Date": _D3},
            {"League": "NBA", "Player": "C", "Market": "points", "Date": _D3},
            {"League": "NBA", "Player": "A", "Market": "points", "Date": _TODAY},
            {"League": "NBA", "Player": "A", "Market": "assists", "Date": _D3},
            {"League": "XYZ", "Player": "A", "Market": "points", "Date": _D3},
            {"League": "NBA", "Player": "A", "Market": "points", "Date": _D3, "Actual": 99.0},
        ]
    )


# idx: 0 single→15, 1 combo→23, 2 matchup→7, 3 no-game, 4 NaN-value, 5 today
# (not pending), 6 missing-market KeyError, 7 league-absent, 8 already-resolved.
_EXPECTED_ACTUAL = [15.0, 23.0, 7.0, np.nan, np.nan, np.nan, np.nan, np.nan, 99.0]


def test_resolve_history_fills_actual_and_skips():
    out = resolve_history(_history(), _stats())
    pd.testing.assert_series_equal(
        out["Actual"].reset_index(drop=True),
        pd.Series(_EXPECTED_ACTUAL, name="Actual"),
        check_dtype=False,
    )
