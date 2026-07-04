"""Supplemental leg pickers — non-star legs to round out a same-game slip.

The constellation builder is same-game (DESIGN §4a) and only draws model-liked stars
(Kelly ``K`` > 0). Two gaps need a non-star path, both rendered as chip sections below
the map:

* :func:`render_satellites` — a game whose ``K`` > 0 legs sit on only one of its two
  teams can't form a valid parlay alone (``validate_parlay_legs`` needs two distinct
  teams), so this offers ``K`` > 0 legs from *other* games on the platform, grouped by
  game — add one validating leg or a whole second cluster. They ride along as satellites.
* :func:`render_disliked_legs` — the manual override for a same-game leg you believe in
  that the model passes on (``K`` ≤ 0, so it isn't a star).

Pure queries plus thin Streamlit renders that return an add/remove/detail action for the
builder to apply (the builder owns the slip state). No Archive, no parquet — they slice
the in-memory ``current_offers`` the builder already holds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
import streamlit as st

from sportstradamus.dashboard.components.constellation import star_label
from sportstradamus.dashboard.components.deep_dive import render_offer_card
from sportstradamus.dashboard.legs import corr_key, find_offer_idx
from sportstradamus.leg_schema import is_model_liked, leg_label
from sportstradamus.prediction.stories.legs import validate_parlay_legs

# Top legs offered per other game — enough to grab a small second cluster, few enough
# to keep the section uncluttered.
_PER_GAME_CAP = 6
# Same-game model-passed legs offered — a few chips, the least-disliked first.
_DISLIKED_CAP = 9
# Chip columns per satellite game row — matches the visual density of the constellation.
_SATELLITE_COLS = 3


def satellite_groups(
    offers: pd.DataFrame,
    *,
    focus_game: str,
    platform: str,
    exclude_keys: set[str],
) -> list[tuple[str, list[dict]]]:
    """Other-game edge legs to offer, grouped by game, games ranked by best edge.

    Filters ``offers`` to model-liked legs (``K`` > 0) on ``platform`` in games other
    than ``focus_game``, drops anything already in the slip (``exclude_keys`` of
    ``corr_key``), keeps each game's top ``_PER_GAME_CAP`` by ``K``, and orders the
    games by their single strongest leg so the best options lead.
    """
    if offers.empty:
        return []
    pool = offers[
        (offers["Platform"] == platform) & (offers["Game"] != focus_game) & (offers["Kelly"] > 0)
    ]
    groups: list[tuple[str, list[dict]]] = []
    for game, block in pool.groupby("Game", sort=False):
        rows = [
            row
            for row in block.sort_values("Kelly", ascending=False).to_dict("records")
            if corr_key(row) not in exclude_keys
        ]
        if rows:
            groups.append((str(game), rows[:_PER_GAME_CAP]))
    groups.sort(key=lambda g: float(g[1][0]["Kelly"]), reverse=True)
    return groups


def render_satellites(
    offers: pd.DataFrame,
    *,
    focus_game: str,
    platform: str,
    legs: Sequence[Mapping],
    key_prefix: str,
) -> dict | None:
    """Render the 'add a leg from another game' section; return an add/remove action.

    ``{"add": offer_row}`` when a candidate chip is clicked, ``{"remove": leg_index}``
    when an added satellite's remove is clicked, else ``None``. The builder applies the
    action — it owns the slip state. The expander auto-opens while the slip can't yet be
    locked in (single-team / one leg) and stays collapsed once it is valid.
    """
    satellites = [(i, leg) for i, leg in enumerate(legs) if leg["game"] != focus_game]
    exclude = {corr_key(leg) for leg in legs}
    groups = satellite_groups(
        offers, focus_game=focus_game, platform=platform, exclude_keys=exclude
    )
    with st.expander("Add a leg from another game", expanded=not validate_parlay_legs(legs)[0]):
        action = _render_added(satellites, key_prefix, caption="From other games", infix="sat")
        if not groups:
            st.caption(f"No other-game edge legs on {platform}.")
            return action
        for game, rows in groups:
            st.caption(game)
            cols = st.columns(_SATELLITE_COLS)
            for j, row in enumerate(rows):
                with cols[j % _SATELLITE_COLS].popover(star_label(row), width="stretch"):
                    picked = _render_pick_popover(row, offers, platform, key_prefix)
                if picked:
                    action = picked
    return action


def _render_pick_popover(
    row: Mapping, offers: pd.DataFrame, platform: str, key_prefix: str
) -> dict | None:
    """A satellite pick's popover: the condensed card plus Add / Full-detail actions.

    Returns ``{"add": row}`` or ``{"detail": offers_index}`` for the builder to apply,
    else ``None``. The same condensed card the constellation shows on hover, with an
    explicit Add (a popover click isn't itself an add the way a star click is).
    """
    render_offer_card(row)
    add_col, det_col = st.columns(2)
    if add_col.button(
        "Add to slip", key=f"{key_prefix}_sat_add_{corr_key(row)}", type="primary", width="stretch"
    ):
        return {"add": row}
    if det_col.button(
        "Full detail",
        icon=":material/arrow_forward:",
        key=f"{key_prefix}_sat_det_{corr_key(row)}",
        width="stretch",
    ):
        idx = find_offer_idx(row, offers, platform)
        if idx is not None:
            return {"detail": idx}
    return None


def _render_added(
    items: list[tuple[int, Mapping]], key_prefix: str, *, caption: str, infix: str
) -> dict | None:
    """List non-star slip legs with a remove control; return a remove action.

    Used for both other-game satellites and same-game model-passed legs — neither is a
    star on the map, so this list is their only removal path. ``items`` are
    ``(slip_index, leg)`` pairs; ``infix`` namespaces the button keys per section.
    """
    if not items:
        return None
    st.caption(caption)
    action: dict | None = None
    for i, leg in items:
        text_col, rm_col = st.columns([8, 1])
        text_col.write(leg_label(leg))
        if rm_col.button(":material/close:", key=f"{key_prefix}_{infix}_rm_{i}", help="Remove leg"):
            action = {"remove": i}
    return action


def render_disliked_legs(
    offers: pd.DataFrame,
    *,
    focus_game: str,
    platform: str,
    legs: Sequence[Mapping],
    key_prefix: str,
) -> dict | None:
    """Render the 'add a leg the model doesn't like' section; return an add/detail action.

    The constellation only draws the focus game's model-liked stars (``K`` > 0); this is
    the manual override for a same-game leg you believe in that the model passes on
    (``K`` ≤ 0). Such a leg rides in the slip like a satellite — never drawn on the map,
    listed here with a remove control. Chips are the game's other non-star offers,
    least-disliked first, capped to ``_DISLIKED_CAP``; returns the same
    ``{"add": row}`` / ``{"remove": idx}`` / ``{"detail": idx}`` union the satellite
    picker does (already-slipped legs are excluded from the chips).
    """
    # P8: restyle these chips/popovers to match the constellation hover card.
    added = [
        (i, leg)
        for i, leg in enumerate(legs)
        if leg["game"] == focus_game and not is_model_liked(leg)
    ]
    exclude = {corr_key(leg) for leg in legs}
    kelly = pd.to_numeric(offers["Kelly"], errors="coerce")
    pool = offers[(offers["Platform"] == platform) & (offers["Game"] == focus_game) & (kelly <= 0)]
    rows = [
        row
        for row in pool.sort_values("Kelly", ascending=False).to_dict("records")
        if corr_key(row) not in exclude
    ]
    with st.expander("Add a leg the model doesn't like", expanded=False):
        action = _render_added(
            added, key_prefix, caption="In this slip (model passes)", infix="dis"
        )
        if not rows:
            if not added:
                st.caption("No other legs in this game.")
            return action
        cols = st.columns(_SATELLITE_COLS)
        for j, row in enumerate(rows[:_DISLIKED_CAP]):
            with cols[j % _SATELLITE_COLS].popover(star_label(row), width="stretch"):
                picked = _render_pick_popover(row, offers, platform, f"{key_prefix}_dis")
            if picked:
                action = picked
        if len(rows) > _DISLIKED_CAP:
            st.caption(f"Showing {_DISLIKED_CAP} of {len(rows)} — nearest the model's line first.")
        return action
