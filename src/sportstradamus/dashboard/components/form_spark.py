"""The constellation hover card's two chart rows: last-five form and line movement.

The card is assembled in the component's JavaScript, so each row has to reach it as
finished markup — ``spark_svg`` keeps every coordinate on this side. One grouped tail over
the league gamelog serves every star's form row: ``deep_dive_charts.build_recent_history``
re-scans the full log per row, which a thirty-star game would pay for thirty times. The
movement row is the Board's own ``Move`` cell, lifted onto the card (:func:`move_sparks`).
"""

from __future__ import annotations

import pandas as pd

from sportstradamus.dashboard import columns
from sportstradamus.dashboard.components.spark_svg import form_svg
from sportstradamus.dashboard.data import GAMELOG_SCHEMA, load_current_line_movement, load_gamelog
from sportstradamus.dashboard.legs import corr_key

# Five games is the recent-form window the offer row already carries as ``Avg 5``, so the
# card's trace and its headline number describe the same stretch of games.
_FORM_GAMES = 5


def form_sparks(pool: pd.DataFrame) -> dict[str, str]:
    """``corr_key`` → the hover card's ``.cst-spark`` inner HTML, per offer in ``pool``.

    Oldest game first, so the trace reads left to right against the line's rule. An offer
    whose player has no gamelog rows, or whose stat names no column of that league's
    gamelog, is absent from the map — the frontend draws its scar for those.

    Groups by league itself rather than taking one, because the "look wider" lens puts
    other games' stars on the map and those games need not share the focus game's league.
    """
    sparks: dict[str, str] = {}
    for league, offers in pool.groupby("League", sort=False):
        gamelog = load_gamelog(str(league))
        if gamelog.empty:
            continue
        pcol, dcol = GAMELOG_SCHEMA[league]["player"], GAMELOG_SCHEMA[league]["date"]
        recent = gamelog[gamelog[pcol].isin(set(offers["Player"]))]
        # NFL's gamelog exposes no date column, but its rows are written in (season, week)
        # order, so the frame's own order is already the chronology there.
        if dcol:
            recent = recent.sort_values(dcol)
        by_player = {
            player: games.tail(_FORM_GAMES) for player, games in recent.groupby(pcol, sort=False)
        }
        for offer in offers.to_dict("records"):
            games = by_player.get(offer["Player"])
            # Same coalesce the History tab reads the gamelog through (deep_dive_tabs).
            stat = offer.get("Stat") or offer["Market"]
            if games is None or stat not in games.columns:
                continue
            values = games[stat].tolist()
            line = float(offer["Line"])
            hits = sum(v >= line for v in values)
            sparks[corr_key(offer)] = (
                f"{form_svg(values, line)}"
                f"<span>last {len(values)} vs line · {hits}/{len(values)} over</span>"
            )
    return sparks


def move_sparks(pool: pd.DataFrame) -> dict[str, str]:
    """``corr_key`` → the hover card's line-movement row, per offer in ``pool``.

    Each row is the Board's ``Move`` trace beside its summary, so the card and the grid tell
    a move in the same words. An offer ``columns.add_line_movement`` leaves blank is absent,
    and the frontend then draws no row at all rather than a scar: an offer the ladder never
    saw has no movement to be missing.

    ``pool`` needs ``Bet`` and the ``LINE_MOVEMENT_KEYS`` columns. Each offer reads its own
    row by them, so a frame that mixes in the "look wider" lens's other games and leagues
    needs nothing extra.
    """
    moved = columns.add_line_movement(pool, load_current_line_movement())
    return {
        corr_key(offer): f"{offer[columns.MOVE_SPARK]}<span>{offer[columns.MOVE_SUMMARY]}</span>"
        for offer in moved.to_dict("records")
        if offer[columns.MOVE_SPARK]
    }
