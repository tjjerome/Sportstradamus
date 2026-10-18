"""The live half of the selected-tail read (``nightly.recommended_gap_by_cell``).

Pins the per-cell frame reflect writes to ``LIVE_GAP_PATH``: the recommended settled
offers of the trailing window per (league, market), the mean of (Win Prob - Hit) with its
day-block bootstrap interval, a cell with nothing recommended in the window absent, the
same interval on every rerun, and the empty typed frame when nothing qualifies.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from sportstradamus import nightly
from sportstradamus.history_schema import HISTORY_COLS
from sportstradamus.nightly import (
    _LIVE_GAP_SEED,
    LIVE_GAP_COLUMNS,
    LIVE_METRICS_WINDOWS,
    _precompute_live_gap,
    recommended_gap_by_cell,
)
from sportstradamus.realized import settled_offers

_NOW = datetime(2026, 10, 3, 12)
# The chosen Under at 20.5 hits on 18 and misses on 25.
_HIT, _MISS = 18.0, 25.0
# A settled Underdog Under at the flat payout (Boost 1.0 -> 1.83): 0.6 x 1.83 - 1 = 0.098
# clears the recommended edge floor, 0.55 (edge 0.007) does not.
_OFFER = {
    "Player": "A",
    "League": "NBA",
    "Date": "2026-09-26",
    "Market": "PTS",
    "Line": 20.5,
    "Boost": 1.0,
    "Platform": "Underdog",
    "Bet": "Under",
    "Win Prob": 0.6,
    "Market Prob": 0.55,
    "Alt Line": False,
    "Model Version": "v1",
    "Market Projection": 19.5,
    "Actual": _HIT,
}
# Four game days inside the 30-day window are the bootstrap's clusters; _OLD sits outside it.
_DAYS = ("2026-09-26", "2026-09-27", "2026-09-28", "2026-09-29")
_OLD = "2026-08-20"
_EXPECTED_DTYPES = {
    "league": "object",
    "market": "object",
    "computed_at": "datetime64[ns]",
    "window_days": "int16",
    "live_n": "int64",
    "live_gap": "float64",
    "live_gap_lo": "float64",
    "live_gap_hi": "float64",
}
# PTS: six recommended settled legs over the four days; the 0.55 leg, the unsettled leg
# and the recommended leg dated before the window do not count. REB: both recommended
# legs are dated before the window and the leg inside it is not recommended, so the cell
# is absent at 30 days and reads two legs at 90.
_PTS = [
    _OFFER | {"Player": "A", "Date": _DAYS[0]},
    _OFFER | {"Player": "B", "Date": _DAYS[0], "Actual": _MISS},
    _OFFER | {"Player": "C", "Date": _DAYS[1], "Win Prob": 0.7},
    _OFFER | {"Player": "D", "Date": _DAYS[2]},
    _OFFER | {"Player": "E", "Date": _DAYS[2], "Win Prob": 0.7, "Actual": _MISS},
    _OFFER | {"Player": "F", "Date": _DAYS[3]},
    _OFFER | {"Player": "G", "Date": _DAYS[1], "Win Prob": 0.55},
    _OFFER | {"Player": "H", "Date": _DAYS[2], "Actual": np.nan},
    _OFFER | {"Player": "I", "Date": _OLD},
]
_REB = [
    _OFFER | {"Player": "J", "Market": "REB", "Date": _OLD},
    _OFFER | {"Player": "K", "Market": "REB", "Date": _OLD, "Actual": _MISS},
    _OFFER | {"Player": "L", "Market": "REB", "Date": _DAYS[0], "Win Prob": 0.55},
]
# Win Prob - Hit over the six PTS legs: -0.4, 0.6, -0.3, -0.4, 0.7, -0.4.
_PTS_GAP = -0.2 / 6


def _history(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=HISTORY_COLS)


def _gaps(history, *, window_days=30):
    return recommended_gap_by_cell(
        settled_offers(history),
        window_days=window_days,
        now=_NOW,
        rng=np.random.default_rng(_LIVE_GAP_SEED),
    )


def test_one_row_per_cell_with_recommended_legs_in_the_window():
    out = _gaps(_history(*_PTS, *_REB))
    assert list(out.columns) == list(LIVE_GAP_COLUMNS)
    assert out.dtypes.astype(str).to_dict() == _EXPECTED_DTYPES
    assert out[["league", "market"]].to_numpy().tolist() == [["NBA", "PTS"]]
    row = out.iloc[0]
    assert row["live_n"] == 6
    assert row["live_gap"] == pytest.approx(_PTS_GAP, abs=1e-12)
    assert row["live_gap_lo"] <= row["live_gap"] <= row["live_gap_hi"]
    assert row["live_gap_lo"] < row["live_gap_hi"]
    assert row["window_days"] == 30
    assert row["computed_at"] == pd.Timestamp(_NOW)


def test_the_window_decides_which_cells_appear():
    out = _gaps(_history(*_PTS, *_REB), window_days=90).set_index("market")
    assert out["live_n"].to_dict() == {"PTS": 7, "REB": 2}


def test_the_same_seed_prints_the_same_interval():
    history = _history(*_PTS, *_REB)
    pd.testing.assert_frame_equal(_gaps(history), _gaps(history))


@pytest.mark.parametrize(
    "history",
    [_history(), _history(*_REB)],
    ids=["nothing settled", "nothing recommended in the window"],
)
def test_nothing_in_the_window_gives_the_empty_typed_frame(history):
    out = _gaps(history)
    assert out.empty
    assert list(out.columns) == list(LIVE_GAP_COLUMNS)
    assert out.dtypes.astype(str).to_dict() == _EXPECTED_DTYPES


def test_precompute_writes_the_graduation_window(tmp_path, monkeypatch):
    path = tmp_path / "live_gap_per_cell.parquet"
    monkeypatch.setattr(nightly, "LIVE_GAP_PATH", path)
    # Dated off the wall clock: the writer anchors its window on now.
    today = datetime.now().date().isoformat()
    _precompute_live_gap(_history(*(row | {"Date": today} for row in _PTS)))
    out = pd.read_parquet(path)
    assert out.dtypes.astype(str).to_dict() == _EXPECTED_DTYPES
    assert out["window_days"].tolist() == [max(LIVE_METRICS_WINDOWS)]
    assert out["live_n"].tolist() == [7]
