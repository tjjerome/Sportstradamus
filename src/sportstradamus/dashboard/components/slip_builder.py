"""The slip builders — main-page interactive editors over a session-state slip.

Two render entry points share the ``slip_state`` API (session-state contract,
seed/add/lock primitives) and one live scorer (``slip_engine.score_slip``):

* :func:`render_constellation_builder` — same-game, correlation-aware, with a
  live deterministic thesis headline. Hosted on the Slips surface; seeded from a
  story (Bankroll Builder / Shoot the Moon), a Game-tab selection, or a sidebar
  edit. The literal star/edge visual is a later theming step; this is the editor.
* :func:`render_simple_builder` — any-game, grade-only (no thesis); hosted on the
  Board over a cross-game selection.

Both end in **Lock it in!**, which upserts the slip to ``user_slips.parquet``
(status ``pending``) for the sidebar shelf and nightly grading. Money is
``Decimal``; legs are snapshotted from ``current_offers`` at seed/add time so
scoring never re-reads the frame.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal

import pandas as pd
import streamlit as st

from sportstradamus.dashboard.components.constellation import constellation_figure
from sportstradamus.dashboard.components.constellation_component import render_constellation
from sportstradamus.dashboard.components.deep_dive import init_detail_state, show_detail
from sportstradamus.dashboard.components.satellite_picker import render_satellites
from sportstradamus.dashboard.components.slip_state import (
    _BANKROLL,
    _BUILDER,
    _LEGS,
    _PLATFORM,
    _leg_from_offer,
    clear_slip,
    lock_in,
    remove_leg,
)
from sportstradamus.dashboard.data import load_model_stats
from sportstradamus.dashboard.legs import corr_key
from sportstradamus.dashboard.slip_engine import SlipScore, score_slip, slip_headline
from sportstradamus.prediction.stories.legs import validate_parlay_legs


def _slip_shrinkage(legs: Sequence[Mapping]) -> float:
    """Resolved slip shrinkage = min per-leg ``kelly_shrinkage`` from model_stats (default 1.0)."""
    stats = load_model_stats()
    if stats.empty or "kelly_shrinkage" not in stats.columns:
        return 1.0
    vals = []
    for leg in legs:
        cell = stats.loc[(stats["league"] == leg["League"]) & (stats["market"] == leg["Market"])]
        if not cell.empty and pd.notna(cell.iloc[0]["kelly_shrinkage"]):
            vals.append(float(cell.iloc[0]["kelly_shrinkage"]))
    return min(vals) if vals else 1.0


def render_constellation_builder(
    offers: pd.DataFrame, corr: pd.DataFrame, ctxs: Mapping, *, key_prefix: str = "cb"
) -> None:
    """Same-game correlation-aware editor with a live deterministic thesis headline.

    The constellation is the editor: a full-color star is a slip leg, a desaturated
    one a candidate, and clicking either toggles it; a star's hover card opens the
    full offer detail without disturbing the slip.
    """
    legs = st.session_state[_LEGS]
    if not legs:
        st.info("Pick a story above to start a slip, or load one from the rail.")
        return
    focus, pool = _focus_pool(legs, offers)
    focus_legs = [leg for leg in legs if leg["Game"] == focus]
    _render_constellation(focus_legs, corr, pool, key_prefix)
    _render_leg_list(key_prefix, focus_game=focus, removable=False)
    act = render_satellites(
        offers,
        focus_game=focus,
        platform=st.session_state[_PLATFORM],
        legs=legs,
        key_prefix=key_prefix,
    )
    if _apply_satellite_action(act, legs):
        st.rerun()
    _draw_detail_dialog(offers)
    valid, reason = validate_parlay_legs(legs)
    if len(legs) < 2:
        st.caption(reason or "Tap a gray star to add a leg.")
        return
    if not valid:
        st.warning(reason)
    shrink = _slip_shrinkage(legs)
    score = _score(legs, corr, shrink)
    headline = slip_headline(focus_legs, offers, ctxs)
    if headline:
        st.markdown(f"#### {headline}")
    _render_metrics(score, correlated=True)
    st.caption("Pairing-block risk arrives with the correlation-block model.")
    _render_lock_in(score, headline, shrink, key_prefix, can_lock=valid)


def render_simple_builder(
    offers: pd.DataFrame, corr: pd.DataFrame, *, key_prefix: str = "sb"
) -> None:
    """Any-game grade-only editor (no thesis); legs come from a Board selection."""
    legs = st.session_state[_LEGS]
    if not legs or st.session_state[_BUILDER] != "simple":
        return
    _render_leg_list(key_prefix)
    if len(legs) < 2:
        st.caption("Select at least two legs to price the slip.")
        return
    shrink = _slip_shrinkage(legs)
    score = _score(legs, corr, shrink)
    _render_metrics(score, correlated=False)
    _render_lock_in(score, "", shrink, key_prefix)


def _score(legs: Sequence[Mapping], corr: pd.DataFrame, shrink: float) -> SlipScore:
    return score_slip(
        legs,
        corr,
        platform=st.session_state[_PLATFORM],
        bankroll=Decimal(str(st.session_state[_BANKROLL])),
        shrinkage=shrink,
    )


def _render_leg_list(
    key_prefix: str, *, focus_game: str | None = None, removable: bool = True
) -> None:
    """List slip legs. ``focus_game`` shows only that game's legs (satellites are listed
    in the picker); ``removable=False`` drops the button column because a leg is removed
    by clicking its star.
    """
    legs = st.session_state[_LEGS]
    for i, leg in enumerate(legs):
        if focus_game is not None and leg["Game"] != focus_game:
            continue
        line = f"{leg['Desc']}  ·  {leg['League']}"
        if not removable:
            st.write(line)
            continue
        text_col, rm_col = st.columns([8, 1])
        text_col.write(line)
        if rm_col.button(":material/close:", key=f"{key_prefix}_rm_{i}", help="Remove leg"):
            remove_leg(i)
            st.rerun()


def _render_constellation(
    legs: list[dict], corr: pd.DataFrame, pool: pd.DataFrame, key_prefix: str
) -> None:
    """Draw the interactive star map and act on the component's click/detail callback.

    A star click toggles its leg (rerun to refresh the map); the hover card's **Full
    detail** seeds the offer dialog, drawn by ``_draw_detail_dialog`` once the map and
    the satellite picker have both rendered (so a satellite's detail opens the same
    way). The component re-sends its last action on every rerun, so each is deduped by
    nonce and fires once.
    """
    action = render_constellation(
        constellation_figure(legs, corr, pool), key=f"{key_prefix}_constellation"
    )
    if _apply_constellation_action(action, legs, pool, key_prefix):
        st.rerun()


def _draw_detail_dialog(offers: pd.DataFrame) -> None:
    """Draw the offer-detail dialog for whichever offer is on the detail stack.

    Shared by the constellation stars and the satellite picks — both push an
    offers-frame index. The dialog's correlation-nav is scoped to that offer's own
    game pool; correlated legs are same-game, so the scope holds across the reruns
    navigation triggers.
    """
    stack = st.session_state.get("detail_stack")
    if not stack or stack[-1] not in offers.index:
        return
    row = offers.loc[stack[-1]]
    game_pool = offers[
        (offers["Game"] == row["Game"]) & (offers["Platform"] == st.session_state[_PLATFORM])
    ]
    show_detail(row, game_pool)


def _apply_constellation_action(
    action: Mapping | None, legs: list[dict], pool: pd.DataFrame, key_prefix: str
) -> bool:
    """Process the component's last action once; return True if the slip changed.

    A ``click`` toggles the star's leg (caller reruns); a ``detail`` seeds the offer
    dialog and leaves the slip alone. The nonce is deduped against the last handled,
    since the component re-sends the same value until the user acts again.
    """
    if not action:
        return False
    nonce_key = f"{key_prefix}_cst_nonce"
    if action.get("nonce") == st.session_state.get(nonce_key):
        return False
    st.session_state[nonce_key] = action.get("nonce")
    key = action.get("key")
    if action.get("action") == "detail":
        _open_offer_detail(key, pool)
        return False
    return _toggle_leg(key, legs, pool)


def _apply_satellite_action(action: Mapping | None, legs: list[dict]) -> bool:
    """Apply the satellite picker's action; return True if the slip changed.

    ``add``/``remove`` mutate the slip (caller reruns); ``detail`` seeds the offer
    dialog and leaves the slip alone (``_draw_detail_dialog`` draws it).
    """
    if not action:
        return False
    if "add" in action:
        legs.append(_leg_from_offer(action["add"]))
        return True
    if "remove" in action:
        remove_leg(action["remove"])
        return True
    init_detail_state()
    st.session_state.detail_stack = [action["detail"]]
    return False


def _toggle_leg(key: str, legs: list[dict], pool: pd.DataFrame) -> bool:
    """Toggle the clicked star: a slip leg → remove, a candidate → add (True if changed)."""
    for i, leg in enumerate(legs):
        if corr_key(leg) == key:
            remove_leg(i)
            return True
    match = _pool_match_for_key(pool, key)
    if match is None:
        return False
    legs.append(_leg_from_offer(match[1]))
    return True


def _open_offer_detail(key: str, pool: pd.DataFrame) -> None:
    """Reuses the Board's ``deep_dive`` dialog and its ``detail_stack`` navigation; the
    slip lives in session state, so opening detail never clears the parlay.
    """
    match = _pool_match_for_key(pool, key)
    if match is None:
        return
    init_detail_state()
    st.session_state.detail_stack = [match[0]]


def _pool_match_for_key(pool: pd.DataFrame, key: str) -> tuple | None:
    """``(index label, row dict)`` for the pool offer matching a star's key, or ``None``."""
    for idx, row in zip(pool.index, pool.to_dict("records"), strict=True):
        if corr_key(row) == key:
            return idx, row
    return None


def _focus_pool(legs: Sequence[Mapping], offers: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """The slip's focus game (the oldest leg's game) and its candidate offers on the platform.

    The constellation anchors on this one game; legs from other games are satellites,
    rendered outside the map. ``legs`` is non-empty (the caller guards).
    """
    focus = legs[0]["Game"]
    platform = st.session_state[_PLATFORM]
    return focus, offers.loc[(offers["Game"] == focus) & (offers["Platform"] == platform)]


def _render_metrics(score: SlipScore, *, correlated: bool) -> None:
    cols = st.columns(5)
    cols[0].metric("Legs", f"{score.bet_size} · {score.play_type}")
    cols[1].metric("Independent", f"{score.indep_p:.1%}")
    if correlated:
        cols[2].metric(
            "With correlation",
            f"{score.joint_p:.1%}",
            f"{score.joint_p - score.indep_p:+.1%}",
        )
    else:
        cols[2].metric("Joint", f"{score.joint_p:.1%}")
    cols[3].metric("Payout", f"{score.payout:.2f}x")
    cols[4].metric("EV", f"{score.model_ev - 1:+.1%}")
    bankroll = float(st.session_state[_BANKROLL])
    st.caption(f"Kelly stake ${score.stake} of ${bankroll:,.0f}")
    if score.payout_approximate:
        st.caption("Sleeper payout approximate — correlation factor pending.")


def _render_lock_in(
    score: SlipScore, headline: str, shrink: float, key_prefix: str, *, can_lock: bool = True
) -> None:
    lock_col, clear_col = st.columns(2)
    if lock_col.button(
        "Lock it in!", key=f"{key_prefix}_lock", type="primary", disabled=not can_lock
    ):
        lock_in(score, headline, shrink)
    if clear_col.button("Clear", key=f"{key_prefix}_clear"):
        clear_slip()
        st.rerun()
