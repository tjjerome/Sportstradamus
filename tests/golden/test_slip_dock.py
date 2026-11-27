"""AppTest smoke for the mobile slip dock: hidden when empty, bar when filled,
sheet contents + remove-leg behavior when expanded.

Renders the dock in a minimal script (not the full app) so the test exercises
the component alone; legs are seeded as canonical structured legs the way
slip_state stores them (all LEG_FIELDS present). Offers/corr parquets aren't
needed — score_slip prices off the leg snapshots and the corr loader's empty
fallback. The refused-pair case points the pair-modifier loader at a one-row
tmp parquet.
"""

from __future__ import annotations

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from sportstradamus.dashboard.legs import corr_key
from sportstradamus.helpers.io import PAIR_MODIFIER_COLS
from sportstradamus.leg_schema import leg_label

_SCRIPT = """
import streamlit as st
from sportstradamus.dashboard.components.slip_dock import render_slip_dock
from sportstradamus.dashboard.components.slip_state import init_slip_state

init_slip_state()
render_slip_dock()
"""


def _leg(player: str, market: str, team: str) -> dict:
    return {
        "player": player,
        "team": team,
        "market": market,
        "stat": market,
        "bet": "Over",
        "line": 25.5,
        "league": "NBA",
        "game": "NYK/BOS",
        "date": "2026-07-16",
        "platform": "Underdog",
        "win_prob": 0.60,
        "boost": 1.0,
        "push_prob": 0.0,
        "kelly": 0.05,
        "fit_prob": 0.55,
        "ev": 0.95,
        "pick": False,
    }


def _dock_test(legs: list[dict], *, open_sheet: bool = False) -> AppTest:
    at = AppTest.from_string(_SCRIPT, default_timeout=15)
    at.session_state["slip_legs"] = legs
    if open_sheet:
        at.session_state["slip_dock_open"] = True
    at.run()
    assert not at.exception
    return at


def test_empty_slip_renders_nothing():
    at = _dock_test([])
    assert not any("slip_dock" in (b.key or "") for b in at.button)


def test_bar_leads_with_ev_beside_model_ev():
    """The legs' fitted .55 prices the slip at 3.5 x .55² and their claimed .60 at 3.5 x .6²,
    so reading one for the other fails."""
    at = _dock_test([_leg("Jalen Brunson", "PTS", "NYK"), _leg("Jayson Tatum", "PTS", "BOS")])
    assert any(b.key == "slip_dock_toggle" for b in at.button)
    summary = " ".join(m.value for m in at.markdown)
    assert "2 legs · EV 1.06 · Model EV 1.26 · $" in summary


def test_sheet_leads_with_ev_then_the_payout_to_check():
    at = _dock_test(
        [_leg("Jalen Brunson", "PTS", "NYK"), _leg("Jayson Tatum", "PTS", "BOS")],
        open_sheet=True,
    )
    captions = [c.value for c in at.caption]
    assert captions[0].startswith("EV 1.06 · Model EV 1.26 · Kelly stake $")
    assert captions[0].endswith(" · joint 30%")
    assert captions[1:] == [
        "Worth a bet if the app pays at least 3.31x (we assume 3.50x).",
        "Legs here share a game, so this value is unproven until you check the app's payout "
        "against that number.",
    ]


def test_unpriced_slip_names_its_leg_and_still_locks():
    unpriced = _leg("Jayson Tatum", "PTS", "BOS") | {"fit_prob": None, "ev": None}
    at = _dock_test([_leg("Jalen Brunson", "PTS", "NYK"), unpriced], open_sheet=True)
    summary = " ".join(m.value for m in at.markdown)
    assert "2 legs · not priced · Model EV 1.26 · $0" in summary
    assert [c.value for c in at.caption] == [
        "Model EV 1.26 · Kelly stake $0",
        f"Not priced: only one side is posted for {leg_label(unpriced)}.",
    ]
    assert not at.button(key="slip_dock_lock").disabled


def test_sheet_lists_legs_and_remove_works():
    at = _dock_test(
        [_leg("Jalen Brunson", "PTS", "NYK"), _leg("Jayson Tatum", "PTS", "BOS")],
        open_sheet=True,
    )
    body = " ".join(m.value for m in at.markdown)
    assert "Brunson" in body
    at.button(key="slip_dock_rm_0").click().run()
    assert len(at.session_state["slip_legs"]) == 1


def test_refused_pair_blocks_the_lock(monkeypatch, tmp_path):
    """A pair Underdog refuses: the bar says so, the sheet warns, and the lock is off."""
    legs = [_leg("Jalen Brunson", "PTS", "NYK"), _leg("Jayson Tatum", "PTS", "BOS")]
    mods = tmp_path / "current_pair_modifiers.parquet"
    pd.DataFrame(
        [["Underdog", "NBA", "NYK/BOS", *sorted(corr_key(leg) for leg in legs), 0.0]],
        columns=PAIR_MODIFIER_COLS,
    ).to_parquet(mods)
    monkeypatch.setattr("sportstradamus.dashboard.data.CURRENT_PAIR_MODIFIERS_PATH", mods)

    at = _dock_test(legs, open_sheet=True)
    assert "won't pair" in " ".join(m.value for m in at.markdown)
    assert any("won't pair" in w.value for w in at.warning)
    assert at.button(key="slip_dock_lock").disabled


def test_lock_stores_the_fitted_ev_beside_the_model_ev(monkeypatch, tmp_path):
    """A priced slip locks with its ev and its model_ev; an unpriced one with no ev."""
    path = tmp_path / "user_slips.parquet"
    monkeypatch.setattr("sportstradamus.helpers.io.USER_SLIPS_PATH", path)
    priced = [_leg("Jalen Brunson", "PTS", "NYK"), _leg("Jayson Tatum", "PTS", "BOS")]
    unpriced = [priced[0], priced[1] | {"fit_prob": None, "ev": None}]
    for legs in (priced, unpriced):
        at = _dock_test(legs, open_sheet=True)
        at.button(key="slip_dock_lock").click().run()
        assert not at.exception
    slips = pd.read_parquet(path)
    assert slips["ev"].tolist()[0] == pytest.approx(3.5 * 0.55**2)
    assert pd.isna(slips["ev"].tolist()[1])
    assert slips["model_ev"].tolist() == pytest.approx([3.5 * 0.6**2] * 2)
