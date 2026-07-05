"""Click-dispatch pins for the constellation builder (P8 Task C6 two-lens click routing).

``_apply_constellation_action`` is the one place a Plotly click's ``{key, action}``
payload gets routed: a key that resolves in ``pool`` (an ordinary star or, when the
"look deeper" lens is on, a dim background star drawn off that same frame) toggles
via ``_toggle_leg``; a key that doesn't falls back to ``_add_wider_leg``, the
add-only path for a "look wider" satellite dot. These call the private dispatch
functions directly against a bare ``st.session_state`` (Streamlit's own documented
"bare mode" — a plain dict outside a running app), rather than standing up a full
``AppTest`` page render, since no component click event needs simulating here.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from sportstradamus.dashboard.components.slip_builder import (
    _add_wider_leg,
    _apply_constellation_action,
    _open_offer_detail,
    _toggle_leg,
)
from sportstradamus.dashboard.components.slip_state import _LEGS


def _offer(player, market, bet, line, game, team, platform, kelly):
    return {
        "Player": player,
        "Market": market,
        "Bet": bet,
        "Line": line,
        "Game": game,
        "Team": team,
        "Platform": platform,
        "Kelly": kelly,
        "League": "NBA",
        "Date": "2026-06-14",
        "Win Prob": 0.6,
        "Boost": 2.0,
    }


def _pool_df(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def setup_function() -> None:
    # A bare (no ScriptRunContext) st.session_state is a genuine process-wide global —
    # xdist reuses one worker process across several tests in this file, so every key
    # a test can write (the slip, the per-key_prefix click nonce, the detail stack)
    # must be reset here or a later test can inherit a nonce and get deduped away.
    st.session_state[_LEGS] = []
    st.session_state.pop("cb_cst_nonce", None)
    st.session_state.pop("detail_stack", None)


def test_toggle_leg_adds_a_pool_candidate():
    pool = _pool_df(_offer("A", "PTS", "Over", 25.5, "NYK/SAS", "NYK", "Underdog", 0.3))
    assert _toggle_leg("A|PTS|Over", pool) is True
    assert [leg["player"] for leg in st.session_state[_LEGS]] == ["A"]


def test_toggle_leg_removes_an_already_slipped_leg():
    pool = _pool_df(_offer("A", "PTS", "Over", 25.5, "NYK/SAS", "NYK", "Underdog", 0.3))
    _toggle_leg("A|PTS|Over", pool)
    assert _toggle_leg("A|PTS|Over", pool) is True
    assert st.session_state[_LEGS] == []


def test_toggle_leg_resolves_a_deep_star_the_same_way_as_an_ordinary_one():
    # A model-passed row (Kelly <= 0) drawn only because "look deeper" put the same
    # unfiltered pool frame in as deep_pool — _toggle_leg doesn't know or care which
    # lens revealed it, it just matches the key against `pool`.
    pool = _pool_df(_offer("D", "PTS", "Under", 10.5, "NYK/SAS", "NYK", "Underdog", -0.2))
    assert _toggle_leg("D|PTS|Under", pool) is True
    assert [leg["player"] for leg in st.session_state[_LEGS]] == ["D"]


def test_toggle_leg_returns_false_for_a_key_outside_the_pool():
    pool = _pool_df(_offer("A", "PTS", "Over", 25.5, "NYK/SAS", "NYK", "Underdog", 0.3))
    assert _toggle_leg("ZZZ|PTS|Over", pool) is False
    assert st.session_state[_LEGS] == []


def test_add_wider_leg_adds_from_the_matching_group():
    groups = [
        ("NYK/SAS", [_offer("A", "PTS", "Over", 25.5, "NYK/SAS", "NYK", "Underdog", 0.3)]),
        ("MIA/ORL", [_offer("B", "AST", "Over", 6.5, "MIA/ORL", "MIA", "Underdog", 0.4)]),
    ]
    assert _add_wider_leg("B|AST|Over", groups) is True
    assert [leg["player"] for leg in st.session_state[_LEGS]] == ["B"]


def test_add_wider_leg_is_add_only_not_a_toggle():
    # satellite_groups already excludes in-slip keys from its own output, so a wider
    # dot's key is never one already in the slip — _add_wider_leg has no remove branch.
    groups = [("NYK/SAS", [_offer("A", "PTS", "Over", 25.5, "NYK/SAS", "NYK", "Underdog", 0.3)])]
    _add_wider_leg("A|PTS|Over", groups)
    assert _add_wider_leg("A|PTS|Over", groups) is True
    assert len(st.session_state[_LEGS]) == 2  # added twice, not toggled off


def test_add_wider_leg_returns_false_when_key_not_in_any_group():
    groups = [("NYK/SAS", [_offer("A", "PTS", "Over", 25.5, "NYK/SAS", "NYK", "Underdog", 0.3)])]
    assert _add_wider_leg("ZZZ|PTS|Over", groups) is False


def test_add_wider_leg_returns_false_when_no_groups():
    assert _add_wider_leg("A|PTS|Over", None) is False


def test_apply_constellation_action_click_resolves_pool_key_via_toggle():
    pool = _pool_df(_offer("A", "PTS", "Over", 25.5, "NYK/SAS", "NYK", "Underdog", 0.3))
    action = {"action": "click", "key": "A|PTS|Over", "nonce": 1}
    changed = _apply_constellation_action(action, pd.DataFrame(), pool, None, "cb")
    assert changed is True
    assert [leg["player"] for leg in st.session_state[_LEGS]] == ["A"]


def test_apply_constellation_action_click_falls_back_to_wider_group_add():
    pool = _pool_df(_offer("A", "PTS", "Over", 25.5, "NYK/SAS", "NYK", "Underdog", 0.3))
    groups = [("MIA/ORL", [_offer("B", "AST", "Over", 6.5, "MIA/ORL", "MIA", "Underdog", 0.4)])]
    action = {"action": "click", "key": "B|AST|Over", "nonce": 1}
    changed = _apply_constellation_action(action, pd.DataFrame(), pool, groups, "cb")
    assert changed is True
    assert [leg["player"] for leg in st.session_state[_LEGS]] == ["B"]


def test_apply_constellation_action_dedupes_by_nonce():
    pool = _pool_df(_offer("A", "PTS", "Over", 25.5, "NYK/SAS", "NYK", "Underdog", 0.3))
    action = {"action": "click", "key": "A|PTS|Over", "nonce": 7}
    assert _apply_constellation_action(action, pd.DataFrame(), pool, None, "cb") is True
    assert _apply_constellation_action(action, pd.DataFrame(), pool, None, "cb") is False
    assert len(st.session_state[_LEGS]) == 1  # the repeat send didn't double-add


def test_open_offer_detail_resolves_a_wider_dot_against_the_full_offers_frame():
    offers = _pool_df(
        _offer("A", "PTS", "Over", 25.5, "NYK/SAS", "NYK", "Underdog", 0.3),
        _offer("B", "AST", "Over", 6.5, "MIA/ORL", "MIA", "Underdog", 0.4),
    )
    pool = offers[offers["Game"] == "NYK/SAS"]
    groups = [("MIA/ORL", [_offer("B", "AST", "Over", 6.5, "MIA/ORL", "MIA", "Underdog", 0.4)])]
    _open_offer_detail("B|AST|Over", offers, pool, groups)
    assert offers.loc[st.session_state.detail_stack[-1], "Player"] == "B"
