"""AppTest smoke for the mobile slip dock: hidden when empty, bar when filled,
sheet contents + remove-leg behavior when expanded.

Renders the dock in a minimal script (not the full app) so the test exercises
the component alone; legs are seeded as canonical structured legs the way
slip_state stores them (all LEG_FIELDS present). Offers/corr parquets aren't
needed — score_slip prices off the leg snapshots and the corr loader's empty
fallback.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

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


def test_bar_summarizes_slip():
    at = _dock_test([_leg("Jalen Brunson", "PTS", "NYK"), _leg("Jayson Tatum", "PTS", "BOS")])
    assert any(b.key == "slip_dock_toggle" for b in at.button)
    summary = " ".join(m.value for m in at.markdown)
    assert "2 legs" in summary


def test_sheet_lists_legs_and_remove_works():
    at = _dock_test(
        [_leg("Jalen Brunson", "PTS", "NYK"), _leg("Jayson Tatum", "PTS", "BOS")],
        open_sheet=True,
    )
    body = " ".join(m.value for m in at.markdown)
    assert "Brunson" in body
    at.button(key="slip_dock_rm_0").click().run()
    assert len(at.session_state["slip_legs"]) == 1
