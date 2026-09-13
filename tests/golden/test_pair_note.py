"""AppTest pins for the slip's pair note (``dashboard/components/pair_note.py``).

The note warns about each pair the platform refuses and captions the pairs it
reprices. Renders the component alone in a minimal script; the ``SlipScore`` is built
there from the seeded ``pair_mods``, the only part of it the note reads.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from sportstradamus.leg_schema import leg_label

_SCRIPT = """
from decimal import Decimal

import streamlit as st

from sportstradamus.dashboard.components.pair_note import render_pair_note
from sportstradamus.dashboard.slip_engine import SlipScore

score = SlipScore(
    0.2, 0.2, 3.0, 0.9, 3, "Power", Decimal("0"), False, st.session_state["pair_mods"]
)
render_pair_note(score, st.session_state["legs"], "Underdog")
"""


def _leg(player: str, market: str) -> dict:
    return {"player": player, "market": market, "bet": "Over", "line": 20.5, "league": "NBA"}


_LEGS = [_leg("Jalen Brunson", "PTS"), _leg("Jalen Brunson", "AST"), _leg("Jayson Tatum", "REB")]
_BRUNSON_PTS, _BRUNSON_AST, _TATUM_REB = (leg_label(leg) for leg in _LEGS)


def _note(pair_mods: tuple) -> AppTest:
    at = AppTest.from_string(_SCRIPT, default_timeout=15)
    at.session_state["legs"] = _LEGS
    at.session_state["pair_mods"] = pair_mods
    at.run()
    assert not at.exception
    return at


def test_refused_pair_warning_names_both_legs():
    at = _note(((0, 2, 0.0),))
    assert [w.value for w in at.warning] == [
        f"Underdog won't pair {_BRUNSON_PTS} with {_TATUM_REB}. Remove one to lock it in."
    ]


def test_same_player_refusal_is_left_to_the_distinct_player_warning():
    at = _note(((0, 1, 0.0),))
    assert not at.warning
    assert not at.caption


def test_refused_slip_gets_no_reprice_caption():
    at = _note(((0, 2, 0.0), (1, 2, 0.85)))
    assert at.warning
    assert not at.caption


def test_repriced_pairs_caption_their_modifiers():
    at = _note(((0, 2, 0.85),))
    assert not at.warning
    assert [c.value for c in at.caption] == [
        f"Underdog reprices {_BRUNSON_PTS} with {_TATUM_REB} ×0.85; the payout includes it."
    ]

    at = _note(((0, 2, 0.85), (1, 2, 1.1)))
    assert [c.value for c in at.caption] == [
        f"Underdog reprices {_BRUNSON_PTS} with {_TATUM_REB} ×0.85; "
        f"{_BRUNSON_AST} with {_TATUM_REB} ×1.1; the payout includes them."
    ]


def test_clean_slip_renders_nothing():
    at = _note(())
    assert not at.warning
    assert not at.caption
