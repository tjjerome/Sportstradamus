"""AppTest for the Board's mobile card list: cards render from a scored-offers
frame, paging extends, Add seeds the simple slip, Detail pushes the dialog stack."""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from sportstradamus.dashboard import columns

_SCRIPT = """
import pandas as pd
import streamlit as st
from sportstradamus.dashboard.components.deep_dive import init_detail_state
from sportstradamus.dashboard.components.offer_cards import render_offer_cards
from sportstradamus.dashboard.components.slip_state import init_slip_state

init_slip_state()
init_detail_state()
df = pd.DataFrame(st.session_state["_fixture_rows"])
render_offer_cards(df)
"""


def _rows(n: int) -> list[dict]:
    return [
        {
            "League": "NBA",
            "Match": "NYK vs BOS",
            "Player": f"Player {i}",
            "Market": "PTS",
            "Stat": "PTS",
            "Market Display": "Points",
            "Bet": "Over",
            "Line": 20.5 + i,
            "Boost": 1.0,
            "Win Prob": 0.55,
            "Model Edge": 4.0,
            "Consensus Edge": 2.0,
            "Platform": "Underdog",
            "Game": "NYK/BOS",
            "Team": "NYK",
            "Date": "2026-07-16",
            "Model EV": 1.05,
            "Kelly": 0.03,
            "Push Prob": 0.0,
        }
        for i in range(n)
    ]


def _card_test(rows: list[dict]) -> AppTest:
    at = AppTest.from_string(_SCRIPT, default_timeout=15)
    at.session_state["_fixture_rows"] = rows
    at.run()
    assert not at.exception
    return at


def test_cards_render_and_page():
    at = _card_test(_rows(35))
    assert any(b.key == "offer_cards_more" for b in at.button)
    body = " ".join(m.value for m in at.markdown)
    assert "Player 0" in body and "Player 34" not in body
    at.button(key="offer_cards_more").click().run()
    body = " ".join(m.value for m in at.markdown)
    assert "Player 34" in body


def test_add_seeds_simple_slip():
    at = _card_test(_rows(3))
    at.button(key="offer_card_add_0").click().run()
    assert len(at.session_state["slip_legs"]) == 1
    assert at.session_state["slip_builder"] == "simple"


def test_detail_pushes_stack():
    at = _card_test(_rows(3))
    at.button(key="offer_card_detail_1").click().run()
    assert at.session_state["detail_stack"] == [1]


def test_card_prints_move_text_as_given_with_no_arrow_from_the_fair_move():
    """``Move Text`` carries the posted line's own arrow. ``Move`` is the fair line's delta,
    which can run the other way — here the posted line rose while the fair line fell — so
    the card must not derive an arrow from its sign. A price-only move prints no text.
    """
    rows = _rows(2)
    rows[0] |= {columns.MOVE: -0.3, columns.MOVE_TEXT: "Line `21.5 → 22.5` ▲"}
    rows[1] |= {columns.MOVE: 0.16, columns.MOVE_TEXT: ""}
    moved, price_only = (m.value for m in _card_test(rows).markdown)
    assert moved.endswith("  \nLine `21.5 → 22.5` ▲")
    assert "Line `" not in price_only
