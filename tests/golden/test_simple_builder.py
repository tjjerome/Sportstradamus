"""AppTest pins for the Board's cross-game slip builder: a slip it won't lock says why.

Renders the builder alone in a minimal script over seeded canonical legs; the
pair-modifier map is seeded beside them.
"""

from __future__ import annotations

import pandas as pd
from streamlit.testing.v1 import AppTest

from sportstradamus.dashboard.legs import corr_key
from sportstradamus.dashboard.slip_engine import modifier_map
from sportstradamus.helpers.io import PAIR_MODIFIER_COLS

_SCRIPT = """
import pandas as pd
import streamlit as st
from sportstradamus.dashboard.components.slip_builder import render_simple_builder
from sportstradamus.dashboard.components.slip_state import init_slip_state

init_slip_state()
render_simple_builder(pd.DataFrame(), {}, st.session_state["mods"])
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


def _refused(*legs: dict) -> dict:
    return modifier_map(
        pd.DataFrame(
            [["Underdog", "NBA", "NYK/BOS", *sorted(corr_key(leg) for leg in legs), 0.0]],
            columns=PAIR_MODIFIER_COLS,
        )
    )


def _builder(legs: list[dict], mods: dict) -> AppTest:
    at = AppTest.from_string(_SCRIPT, default_timeout=15)
    at.session_state["slip_legs"] = legs
    at.session_state["slip_builder"] = "simple"
    at.session_state["mods"] = mods
    at.run()
    assert not at.exception
    return at


def test_one_team_slip_locks():
    """The Board relaxes the both-teams rule: two legs off one team price and lock."""
    legs = [_leg("Jalen Brunson", "PTS", "NYK"), _leg("Josh Hart", "REB", "NYK")]
    at = _builder(legs, {})
    assert not at.warning
    assert not at.button(key="sb_lock").disabled


def test_same_player_slip_says_why_it_cannot_lock():
    """prophecize refuses a player paired with himself; the pair note skips that pair,
    so the distinct-player warning is the reason on screen."""
    legs = [_leg("Jalen Brunson", "PTS", "NYK"), _leg("Jalen Brunson", "AST", "NYK")]
    at = _builder(legs, _refused(*legs))
    assert [w.value for w in at.warning] == [
        "Two legs share a player — each leg needs a distinct player."
    ]
    assert at.button(key="sb_lock").disabled


def test_refused_pair_says_why_it_cannot_lock():
    legs = [_leg("Jalen Brunson", "PTS", "NYK"), _leg("Jayson Tatum", "PTS", "BOS")]
    at = _builder(legs, _refused(*legs))
    assert len(at.warning) == 1
    assert "Underdog won't pair" in at.warning[0].value
    assert at.button(key="sb_lock").disabled
