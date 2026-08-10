"""Supplemental leg data — non-star legs that ride along in a same-game slip.

The constellation builder is same-game (DESIGN §4a) and only draws model-liked stars
(Kelly ``K`` > 0). Two gaps need a non-star path:

* :func:`satellite_groups` — a game whose ``K`` > 0 legs sit on only one of its two
  teams can't form a valid parlay alone (``validate_parlay_legs`` needs two distinct
  teams), so this offers ``K`` > 0 legs from *other* games on the platform, grouped by
  game — the "look wider" lens (``constellation.py``) draws its output on the star
  map's outer ring; tapping one adds it as a satellite.
* A same-game leg you believe in that the model passes on (``K`` ≤ 0) rides the same
  way — the "look deeper" lens draws those directly off the game's own offer pool.

``satellite_groups`` is a pure query (no Streamlit, no Archive) — it slices the
in-memory ``current_offers`` the builder already holds. :func:`render_added_legs` is
the shared non-star leg list (used for whichever of the two kinds of leg above are
currently in the slip, since neither is ever drawn as a star) — it returns a remove
action for the builder to apply.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import streamlit as st

from sportstradamus.dashboard.legs import corr_key
from sportstradamus.leg_schema import leg_label

# Top legs offered per other game — enough to grab a small second cluster, few enough
# to keep the "look wider" ring uncluttered.
_PER_GAME_CAP = 6


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


def render_added_legs(
    items: list[tuple[int, Mapping]], key_prefix: str, *, caption: str, infix: str
) -> dict | None:
    """List non-star slip legs with a remove control; return a remove action.

    Used for both other-game satellites and same-game model-passed legs — neither is a
    star on the map, so this list is their only removal path (their traces vanish
    once the lens that revealed them toggles off). ``items`` are ``(slip_index, leg)``
    pairs; ``infix`` namespaces the button keys per section.
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
