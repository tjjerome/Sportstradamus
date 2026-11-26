"""AppTest for the Board: it opens on the board's picks, ranks by EV and shows their record.

The offers mix two picks with legs the board did not pick. The pick the model likes more
(Model EV 1.10, the only positive Kelly among the picks) has the lower EV, and the leg with
the highest EV is no pick, so a Board that ranks or filters on the model's claim, or on EV
alone, comes out in a different order.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import streamlit as st
from streamlit.testing.v1 import AppTest

_APP = Path("src/sportstradamus/dashboard/app.py").resolve()

_BASE = {
    "League": "NBA",
    "Date": "2026-07-04",
    "Team": "NYK",
    "Opponent": "SAS",
    "Home": False,
    "Game": "NYK/SAS",
    "Market": "PTS",
    "Platform": "Underdog",
    "Bet": "Over",
    "Line": 20.5,
    "Boost": 1.0,
    "Win Prob": 0.55,
    "Market EV": 1.0,
    "Kelly": 0.0,
    "Star": 1.0,
}


def _offer(player: str, fit: float, ev: float, model_ev: float, *, pick: bool = False) -> dict:
    return {
        **_BASE,
        "Player": player,
        "Fit Prob": fit,
        "Fit Prob Low": fit - 0.04,
        "EV": ev,
        "Model EV": model_ev,
        "Pick": pick,
    }


_OFFERS = [
    _offer("C. Unpriced", np.nan, np.nan, 1.20),
    _offer("B. Pick", 0.50, 0.915, 1.10, pick=True) | {"Kelly": 0.05},
    _offer("D. Unpriced", np.nan, np.nan, 0.90),
    _offer("A. Pick", 0.52, 0.95, 1.02, pick=True),
    _offer("E. Passed", 0.54, 0.99, 0.85),
]

_RECORD_ROW = {
    "dates": 30,
    "first_date": "2026-09-01",
    "last_date": "2026-11-25",
    "hit_rate": 0.561,
    "breakeven_rate": 0.546,
}
_RECORD = [
    {**_RECORD_ROW, "play": "model_edge", "league": "NBA", "kind": "underdog_even", "n": 900}
    | {"roi": -0.081, "roi_lo": -0.12, "roi_hi": -0.04},
    {**_RECORD_ROW, "play": "ev_pick", "league": "NBA", "kind": "underdog_even", "n": 269}
    | {"roi": 0.027, "roi_lo": -0.041, "roi_hi": 0.096},
]


def _board(monkeypatch, tmp_path) -> AppTest:
    offers, record = tmp_path / "current_offers.parquet", tmp_path / "play_record.parquet"
    pd.DataFrame(_OFFERS).to_parquet(offers)
    pd.DataFrame(_RECORD).to_parquet(record)
    data_module = importlib.import_module("sportstradamus.dashboard.data")
    monkeypatch.setattr(data_module, "CURRENT_OFFERS_PATH", offers)
    monkeypatch.setattr(data_module, "PLAY_RECORD_PATH", record)
    st.cache_data.clear()

    at = AppTest.from_file(str(_APP), default_timeout=30)
    at.run()
    at.switch_page("surfaces/board.py")
    at.run()
    assert not at.exception
    return at


def _grid_rows(at: AppTest) -> pd.DataFrame:
    grid = next(c for c in at.get("component_instance") if "AgGrid" in c.proto.component_name)
    rows = next(arg for arg in grid.proto.special_args if arg.key == "data")
    return pa.ipc.open_stream(rows.arrow_dataframe.data.data).read_all().to_pandas()


def test_board_opens_on_the_picks_in_ev_order(monkeypatch, tmp_path):
    at = _board(monkeypatch, tmp_path)
    assert at.segmented_control(key="board_lens").value == "Picks"
    assert _grid_rows(at)["Player"].tolist() == ["A. Pick", "B. Pick"]


def test_legs_with_no_ev_list_after_the_ranked_legs_by_model_ev(monkeypatch, tmp_path):
    at = _board(monkeypatch, tmp_path)
    at.segmented_control(key="board_lens").set_value("All").run()
    rows = _grid_rows(at)
    assert rows["Player"].tolist() == [
        "E. Passed",
        "A. Pick",
        "B. Pick",
        "C. Unpriced",
        "D. Unpriced",
    ]
    assert {"Win %", "EV", "Model win %", "Model EV"} <= set(rows.columns)
    assert "Model Edge" not in rows.columns


def test_board_shows_the_picks_record_for_their_league_and_kind(monkeypatch, tmp_path):
    at = _board(monkeypatch, tmp_path)
    record = next(c for c in at.caption if c.value.startswith("NBA even picks"))
    assert record.value == "NBA even picks: 269 legs, +2.7%"
    assert "-4.1% to +9.6%" in record.proto.help


def test_a_picks_detail_leads_with_its_ev_and_carries_its_record(monkeypatch, tmp_path):
    at = _board(monkeypatch, tmp_path)
    # The Board keeps an open dialog's stack across a rerun only on a correlation hop.
    at.session_state["detail_stack"] = [0]
    at.session_state["corr_nav"] = True
    at.run()
    assert not at.exception
    assert any(m.value.startswith("**EV 0.95** · Win 52%") for m in at.markdown)
    assert [c.value for c in at.caption].count("NBA even picks: 269 legs, +2.7%") == 2
