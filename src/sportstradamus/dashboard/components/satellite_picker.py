"""Supplemental leg data — non-star legs that ride along in a same-game slip.

The constellation builder is same-game (DESIGN §4a), so a leg from another game can
never be one of its stars. A game whose priced legs sit on only one of its two teams can't
form a valid parlay alone (``validate_parlay_legs`` needs two distinct teams), so
:func:`satellite_groups` offers the priced legs of *other* games on the platform, grouped
by game — the "look wider" lens (``constellation_wider``) scatters its output
through the open sky around the map, clustered by game; tapping one adds it as a
satellite.

``satellite_groups`` is a pure query (no Streamlit, no Archive) — it slices the
in-memory ``current_offers`` the builder already holds. :func:`render_added_legs` lists
whichever satellites are currently in the slip, since none of them is ever drawn as a
star — it returns a remove action for the builder to apply.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import streamlit as st

from sportstradamus.dashboard.legs import corr_key
from sportstradamus.leg_schema import leg_label

# Top legs offered per other game — enough to grab a small second cluster, few enough
# that one game's sky cluster still reads as a group.
_PER_GAME_CAP = 6


def satellite_groups(
    offers: pd.DataFrame,
    *,
    focus_game: str,
    platform: str,
    exclude_keys: set[str],
) -> list[tuple[str, list[dict]]]:
    """Other games' priced legs to offer, grouped by game, games ranked by their best EV.

    Filters ``offers`` to the legs the fit priced on ``platform`` in games other than
    ``focus_game``, drops anything already in the slip (``exclude_keys`` of
    ``corr_key``), keeps each game's top ``_PER_GAME_CAP`` by ``EV``, and orders the
    games by their single highest-EV leg so the best options lead.
    """
    if offers.empty:
        return []
    pool = offers[
        (offers["Platform"] == platform) & (offers["Game"] != focus_game) & offers["EV"].notna()
    ]
    groups: list[tuple[str, list[dict]]] = []
    for game, block in pool.groupby("Game", sort=False):
        rows = [
            row
            for row in block.sort_values("EV", ascending=False).to_dict("records")
            if corr_key(row) not in exclude_keys
        ]
        if rows:
            groups.append((str(game), rows[:_PER_GAME_CAP]))
    groups.sort(key=lambda g: float(g[1][0]["EV"]), reverse=True)
    return groups


def render_added_legs(
    items: list[tuple[int, Mapping]], key_prefix: str, *, caption: str, infix: str
) -> dict | None:
    """List the slip's other-game satellites with a remove control; return the action.

    A satellite is never a star on the focus game's map, so this list is its only
    removal path — its sky dot vanishes as soon as the *wider* lens toggles off.
    ``items`` are ``(slip_index, leg)`` pairs; ``infix`` namespaces the button keys
    per section.
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
