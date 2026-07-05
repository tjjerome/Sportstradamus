"""The slip builders — main-page interactive editors over a session-state slip.

Two render entry points share the ``slip_state`` API (session-state contract,
seed/add/lock primitives) and one live scorer (``slip_engine.score_slip``):

* :func:`render_constellation_builder` — same-game, correlation-aware, with a
  live deterministic thesis headline. Hosted on the Games surface; seeded from a
  story (Bankroll Builder / Shoot the Moon), a game-seed selection, or a sidebar
  edit. Carries the focus game's Total / Spread / Shape context banner.
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

from sportstradamus.dashboard.components.astrolabe_component import render_astrolabe
from sportstradamus.dashboard.components.constellation import constellation_figure
from sportstradamus.dashboard.components.constellation_component import render_constellation
from sportstradamus.dashboard.components.deep_dive import init_detail_state, show_detail
from sportstradamus.dashboard.components.satellite_picker import (
    render_disliked_legs,
    render_satellites,
)
from sportstradamus.dashboard.components.slip_state import (
    _BANKROLL,
    _BUILDER,
    _LEGS,
    _PLATFORM,
    clear_slip,
    lock_in,
    remove_leg,
)
from sportstradamus.dashboard.data import load_model_stats
from sportstradamus.dashboard.legs import corr_key
from sportstradamus.dashboard.slip_engine import (
    SlipScore,
    astrolabe_payload,
    score_slip,
    slip_headline,
)
from sportstradamus.leg_schema import build_leg, is_model_liked, leg_label
from sportstradamus.prediction.stories.legs import validate_parlay_legs


def _slip_shrinkage(legs: Sequence[Mapping]) -> float:
    stats = load_model_stats()
    if stats.empty or "kelly_shrinkage" not in stats.columns:
        return 1.0
    vals = []
    for leg in legs:
        cell = stats.loc[(stats["league"] == leg["league"]) & (stats["market"] == leg["market"])]
        if not cell.empty and pd.notna(cell.iloc[0]["kelly_shrinkage"]):
            vals.append(float(cell.iloc[0]["kelly_shrinkage"]))
    return min(vals) if vals else 1.0


def render_constellation_builder(
    offers: pd.DataFrame,
    corr: pd.DataFrame,
    ctxs: Mapping,
    *,
    focus_game: str,
    key_prefix: str = "cb",
) -> None:
    """Same-game correlation-aware editor with a live deterministic thesis headline.

    Draws ``focus_game``'s star map whether or not a slip is seeded: every candidate
    (model-liked, Kelly > 0) leg is a clickable star, so a slip can be built from
    scratch by clicking or pre-seeded from a story; a star's hover card opens the
    full offer detail without disturbing the slip. Other-game slip legs show as
    satellites. The caller draws the game's context banner above this.
    """
    if not focus_game:
        st.info("Pick a game above to see its constellation.")
        return
    legs = st.session_state[_LEGS]
    platform = st.session_state[_PLATFORM]
    pool = offers.loc[(offers["Game"] == focus_game) & (offers["Platform"] == platform)]
    focus_legs = [leg for leg in legs if leg["game"] == focus_game]
    # Only the model-liked legs are stars; a manually-added model-passed leg (Kelly ≤ 0)
    # rides in the slip like a satellite — listed by the picker, never drawn on the map.
    star_legs = [leg for leg in focus_legs if is_model_liked(leg)]
    _render_constellation(star_legs, corr, pool, key_prefix)
    _render_leg_list(key_prefix, focus_game=focus_game, removable=False)
    _render_supplemental_pickers(
        offers, focus_game=focus_game, platform=platform, legs=legs, key_prefix=key_prefix
    )
    _draw_detail_dialog(offers)
    if len(legs) < 2:
        if legs:
            _, reason = validate_parlay_legs(legs)
            st.caption(reason or "Tap a star to add another leg.")
        else:
            st.caption("Tap a star to start a slip from this game.")
        return
    valid, reason = validate_parlay_legs(legs)
    if not valid:
        st.warning(reason)
    shrink = _slip_shrinkage(legs)
    score = score_slip(
        legs,
        corr,
        platform=platform,
        bankroll=Decimal(str(st.session_state[_BANKROLL])),
        shrinkage=shrink,
    )
    headline = slip_headline(focus_legs, offers, ctxs)
    if headline:
        st.markdown(f"#### {headline}")
    _render_metrics(score, legs, key_prefix=key_prefix)
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
    score = score_slip(
        legs,
        corr,
        platform=st.session_state[_PLATFORM],
        bankroll=Decimal(str(st.session_state[_BANKROLL])),
        shrinkage=shrink,
    )
    _render_metrics(score, legs, key_prefix=key_prefix)
    _render_lock_in(score, "", shrink, key_prefix)


def _render_leg_list(
    key_prefix: str, *, focus_game: str | None = None, removable: bool = True
) -> None:
    """List slip legs. ``focus_game`` shows only that game's **star** legs (Kelly > 0);
    satellites and same-game model-passed legs are listed by the pickers. ``removable=False``
    drops the button column because a star leg is removed by clicking it on the map.
    """
    legs = st.session_state[_LEGS]
    for i, leg in enumerate(legs):
        if focus_game is not None and (leg["game"] != focus_game or not is_model_liked(leg)):
            continue
        line = f"{leg_label(leg)}  ·  {leg['league']}"
        if not removable:
            st.write(line)
            continue
        text_col, rm_col = st.columns([8, 1])
        text_col.write(line)
        if rm_col.button(":material/close:", key=f"{key_prefix}_rm_{i}", help="Remove leg"):
            remove_leg(i)
            st.rerun()


def _render_supplemental_pickers(
    offers: pd.DataFrame,
    *,
    focus_game: str,
    platform: str,
    legs: list[dict],
    key_prefix: str,
) -> None:
    """Non-star leg pickers below the map: other-game satellites (only once the slip has
    a leg to complete) and same-game model-passed legs; rerun when either changes the slip.
    """
    if legs:
        act = render_satellites(
            offers, focus_game=focus_game, platform=platform, legs=legs, key_prefix=key_prefix
        )
        if _apply_satellite_action(act, legs):
            st.rerun()
    dis = render_disliked_legs(
        offers, focus_game=focus_game, platform=platform, legs=legs, key_prefix=key_prefix
    )
    if _apply_satellite_action(dis, legs):
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
    if _apply_constellation_action(action, pool, key_prefix):
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
    action: Mapping | None, pool: pd.DataFrame, key_prefix: str
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
    return _toggle_leg(key, pool)


def _apply_satellite_action(action: Mapping | None, legs: list[dict]) -> bool:
    """Apply the satellite picker's action; return True if the slip changed.

    ``add``/``remove`` mutate the slip (caller reruns); ``detail`` seeds the offer
    dialog and leaves the slip alone (``_draw_detail_dialog`` draws it).
    """
    if not action:
        return False
    if "add" in action:
        legs.append(build_leg(action["add"]))
        return True
    if "remove" in action:
        remove_leg(action["remove"])
        return True
    init_detail_state()
    st.session_state.detail_stack = [action["detail"]]
    return False


def _toggle_leg(key: str, pool: pd.DataFrame) -> bool:
    """Toggle the clicked star against the active slip: a slip leg → remove, a candidate
    → add. Reads/mutates the canonical slip (``st.session_state[_LEGS]``) directly — the
    map is drawn over the focus game's *filtered* leg view, so toggling that throwaway
    copy would drop the add and mis-index the remove once a slip has cross-game legs.
    """
    legs = st.session_state[_LEGS]
    for i, leg in enumerate(legs):
        if corr_key(leg) == key:
            remove_leg(i)
            return True
    match = _pool_match_for_key(pool, key)
    if match is None:
        return False
    legs.append(build_leg(match[1]))
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
    for idx, row in zip(pool.index, pool.to_dict("records"), strict=True):
        if corr_key(row) == key:
            return idx, row
    return None


def _render_metrics(score: SlipScore, legs: Sequence[Mapping], *, key_prefix: str) -> None:
    """Draw the astrolabe readout in place of the old flat metric row.

    The nonce is derived from the current leg-set (not a session-state counter) so the
    component's CSS dials sweep only on a real add/remove — an unrelated rerun recomputes
    the identical ``SlipScore`` from the same legs and hashes to the same nonce.
    """
    nonce = hash(tuple(sorted(corr_key(leg) for leg in legs)))
    payload = astrolabe_payload(score, nonce=nonce)
    render_astrolabe(payload, key=f"{key_prefix}_astrolabe")
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
