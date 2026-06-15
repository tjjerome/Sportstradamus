"""Satellite legs — edge legs from other games, to complete a single-team slip.

The constellation builder is same-game (DESIGN §4a): a game whose model-liked legs
(Kelly ``K`` > 0) sit on only one of its two teams can't form a valid parlay on its
own (``validate_parlay_legs`` needs two distinct teams). This picker offers the game's
complement — ``K`` > 0 legs from *other* games on the slip's platform, grouped by
game — so a user can add one validating leg or a whole second cluster. The
constellation stays untouched; these ride along as satellites.

Pure query plus a thin Streamlit render that returns an add/remove action for the
builder to apply (the builder owns the slip state). No Archive, no parquet — it slices
the in-memory ``current_offers`` the builder already holds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
import streamlit as st

from sportstradamus.dashboard.components.constellation import star_label
from sportstradamus.dashboard.components.deep_dive import render_offer_card
from sportstradamus.dashboard.legs import corr_key, find_offer_idx
from sportstradamus.prediction.stories.legs import validate_parlay_legs

# Top legs offered per other game — enough to grab a small second cluster, few enough
# to keep the section uncluttered.
_PER_GAME_CAP = 6
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
    satellites = [(i, leg) for i, leg in enumerate(legs) if leg["Game"] != focus_game]
    exclude = {corr_key(leg) for leg in legs}
    groups = satellite_groups(
        offers, focus_game=focus_game, platform=platform, exclude_keys=exclude
    )
    with st.expander("Add a leg from another game", expanded=not validate_parlay_legs(legs)[0]):
        action = _render_added(satellites, key_prefix)
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


def _render_added(satellites: list[tuple[int, Mapping]], key_prefix: str) -> dict | None:
    """List the slip's current satellites with a remove control; return a remove action."""
    if not satellites:
        return None
    st.caption("From other games")
    action: dict | None = None
    for i, leg in satellites:
        text_col, rm_col = st.columns([8, 1])
        text_col.write(leg["Desc"])
        if rm_col.button(":material/close:", key=f"{key_prefix}_sat_rm_{i}", help="Remove leg"):
            action = {"remove": i}
    return action
