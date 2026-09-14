"""Star-intent pins for the constellation builder's map value.

``_apply_constellation_action`` is the one place the map's ``{seq, lit, detail}`` value
lands: a lit star missing from the slip is added from ``pool`` (an ordinary star or, when
the "look deeper" lens is on, a dim background star drawn off that same frame), else from
``wider_groups`` (a "look wider" sky star); an unlit star in the slip is removed; a value
whose ``seq`` is already applied is ignored. These call the private functions directly
against a bare ``st.session_state`` (Streamlit's own documented "bare mode" — a plain dict
outside a running app), rather than standing up a full ``AppTest`` page render, since no
component event needs simulating here.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from sportstradamus.dashboard.components.slip_builder import (
    _active_lenses,
    _apply_constellation_action,
    _open_offer_detail,
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
    # a test can write (the slip, the map's value and applied seq, the detail stack) must
    # be reset here or a later test inherits it.
    st.session_state[_LEGS] = []
    st.session_state.pop("cb_constellation", None)
    st.session_state.pop("cb_constellation_seq", None)
    st.session_state.pop("detail_stack", None)


def test_active_lenses_both_off_resolves_none_and_none():
    offers = _pool_df(_offer("A", "PTS", "Over", 25.5, "NYK/SAS", "NYK", "Underdog", 0.3))
    pool = offers[offers["Game"] == "NYK/SAS"]
    st.session_state["lens_deep"] = False
    st.session_state["lens_wider"] = False
    deep_pool, wider_groups = _active_lenses(
        offers, [], pool, focus_game="NYK/SAS", platform="Underdog"
    )
    assert deep_pool is None
    assert wider_groups is None


def test_active_lenses_deep_only_passes_pool_through_and_skips_wider_query():
    offers = _pool_df(_offer("A", "PTS", "Over", 25.5, "NYK/SAS", "NYK", "Underdog", 0.3))
    pool = offers[offers["Game"] == "NYK/SAS"]
    st.session_state["lens_deep"] = True
    st.session_state["lens_wider"] = False
    deep_pool, wider_groups = _active_lenses(
        offers, [], pool, focus_game="NYK/SAS", platform="Underdog"
    )
    assert deep_pool is pool
    assert wider_groups is None


def test_active_lenses_wider_only_runs_satellite_groups_and_skips_deep():
    offers = _pool_df(
        _offer("A", "PTS", "Over", 25.5, "NYK/SAS", "NYK", "Underdog", 0.3),
        _offer("B", "AST", "Over", 6.5, "MIA/ORL", "MIA", "Underdog", 0.4),
    )
    pool = offers[offers["Game"] == "NYK/SAS"]
    st.session_state["lens_deep"] = False
    st.session_state["lens_wider"] = True
    deep_pool, wider_groups = _active_lenses(
        offers, [], pool, focus_game="NYK/SAS", platform="Underdog"
    )
    assert deep_pool is None
    assert wider_groups == [("MIA/ORL", [offers.iloc[1].to_dict()])]


def _emit(seq: int, lit: dict[str, bool], detail: str | None = None) -> str:
    """Land a map value in its session-state slot; return the slot's key."""
    st.session_state["cb_constellation"] = {"seq": seq, "lit": lit, "detail": detail}
    return "cb_constellation"


def test_apply_adds_then_removes_a_pool_star():
    pool = _pool_df(_offer("A", "PTS", "Over", 25.5, "NYK/SAS", "NYK", "Underdog", 0.3))
    assert _apply_constellation_action(_emit(1, {"A|PTS|Over": True}), pd.DataFrame(), pool, None)
    assert [leg["player"] for leg in st.session_state[_LEGS]] == ["A"]
    assert _apply_constellation_action(_emit(2, {"A|PTS|Over": False}), pd.DataFrame(), pool, None)
    assert st.session_state[_LEGS] == []


def test_apply_resolves_a_deep_star_the_same_way_as_an_ordinary_one():
    # A model-passed row (Kelly <= 0) drawn only because "look deeper" put the same
    # unfiltered pool frame in as deep_pool — the apply doesn't know or care which lens
    # revealed it, it just matches the key against `pool`.
    pool = _pool_df(_offer("D", "PTS", "Under", 10.5, "NYK/SAS", "NYK", "Underdog", -0.2))
    _apply_constellation_action(_emit(1, {"D|PTS|Under": True}), pd.DataFrame(), pool, None)
    assert [leg["player"] for leg in st.session_state[_LEGS]] == ["D"]


def test_apply_adds_a_wider_star_and_skips_a_key_that_resolves_nowhere():
    pool = _pool_df(_offer("A", "PTS", "Over", 25.5, "NYK/SAS", "NYK", "Underdog", 0.3))
    groups = [("MIA/ORL", [_offer("B", "AST", "Over", 6.5, "MIA/ORL", "MIA", "Underdog", 0.4)])]
    value = _emit(1, {"B|AST|Over": True, "ZZZ|PTS|Over": True})
    _apply_constellation_action(value, pd.DataFrame(), pool, groups)
    assert [leg["player"] for leg in st.session_state[_LEGS]] == ["B"]


def test_apply_repeated_lit_intent_adds_no_duplicate():
    # The frontend re-sends an intent until an ack covers it, so the next click's value
    # can repeat one already applied — a wider star's included.
    groups = [("MIA/ORL", [_offer("B", "AST", "Over", 6.5, "MIA/ORL", "MIA", "Underdog", 0.4)])]
    for seq in (1, 2):
        value = _emit(seq, {"B|AST|Over": True})
        _apply_constellation_action(value, pd.DataFrame(), pd.DataFrame(), groups)
    assert [leg["player"] for leg in st.session_state[_LEGS]] == ["B"]


def test_apply_ignores_a_value_at_or_below_the_applied_seq():
    # The map's on_change and the apply right after the map both see each value.
    pool = _pool_df(_offer("A", "PTS", "Over", 25.5, "NYK/SAS", "NYK", "Underdog", 0.3))
    assert _apply_constellation_action("cb_constellation", pd.DataFrame(), pool, None) is False
    _apply_constellation_action(_emit(5, {"A|PTS|Over": True}), pd.DataFrame(), pool, None)
    assert _apply_constellation_action("cb_constellation", pd.DataFrame(), pool, None) is False
    stale = _emit(4, {"A|PTS|Over": False})
    assert _apply_constellation_action(stale, pd.DataFrame(), pool, None) is False
    assert [leg["player"] for leg in st.session_state[_LEGS]] == ["A"]


def test_apply_lands_every_intent_one_value_carries():
    # Streamlit keeps only the newest value when it coalesces reruns, so quick clicks
    # arrive as one value.
    pool = _pool_df(
        _offer("A", "PTS", "Over", 25.5, "NYK/SAS", "NYK", "Underdog", 0.3),
        _offer("C", "REB", "Over", 8.5, "NYK/SAS", "SAS", "Underdog", 0.2),
    )
    _apply_constellation_action(_emit(1, {"A|PTS|Over": True}), pd.DataFrame(), pool, None)
    value = _emit(2, {"A|PTS|Over": False, "C|REB|Over": True})
    _apply_constellation_action(value, pd.DataFrame(), pool, None)
    assert [leg["player"] for leg in st.session_state[_LEGS]] == ["C"]


def test_apply_lands_intents_and_opens_detail_from_one_value():
    offers = _pool_df(
        _offer("A", "PTS", "Over", 25.5, "NYK/SAS", "NYK", "Underdog", 0.3),
        _offer("C", "REB", "Over", 8.5, "NYK/SAS", "SAS", "Underdog", 0.2),
    )
    value = _emit(1, {"A|PTS|Over": True}, detail="C|REB|Over")
    _apply_constellation_action(value, offers, offers, None)
    assert [leg["player"] for leg in st.session_state[_LEGS]] == ["A"]
    assert offers.loc[st.session_state.detail_stack[-1], "Player"] == "C"


def test_apply_detail_alone_opens_detail_without_touching_the_slip():
    pool = _pool_df(_offer("A", "PTS", "Over", 25.5, "NYK/SAS", "NYK", "Underdog", 0.3))
    _apply_constellation_action(_emit(1, {"A|PTS|Over": True}), pool, pool, None)
    assert _apply_constellation_action(_emit(2, {}, detail="A|PTS|Over"), pool, pool, None)
    assert [leg["player"] for leg in st.session_state[_LEGS]] == ["A"]
    assert st.session_state.detail_stack == [0]


def test_open_offer_detail_resolves_a_wider_dot_against_the_full_offers_frame():
    offers = _pool_df(
        _offer("A", "PTS", "Over", 25.5, "NYK/SAS", "NYK", "Underdog", 0.3),
        _offer("B", "AST", "Over", 6.5, "MIA/ORL", "MIA", "Underdog", 0.4),
    )
    pool = offers[offers["Game"] == "NYK/SAS"]
    groups = [("MIA/ORL", [_offer("B", "AST", "Over", 6.5, "MIA/ORL", "MIA", "Underdog", 0.4)])]
    _open_offer_detail("B|AST|Over", offers, pool, groups)
    assert offers.loc[st.session_state.detail_stack[-1], "Player"] == "B"
