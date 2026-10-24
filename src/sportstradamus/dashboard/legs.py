"""Lookup helpers for parlay legs: the offer row, the correlation key and the model cell."""

from collections.abc import Mapping

import pandas as pd

from sportstradamus.helpers import archive_market
from sportstradamus.leg_schema import leg_field

__all__ = ["corr_key", "find_offer_idx", "leg_cell_market"]


def corr_key(leg: Mapping) -> str:
    """Canonical correlation-slice key for a leg: ``player|market|bet``.

    Matches ``current_game_corr``'s ``leg_a``/``leg_b`` keys so the slip engine
    and the constellation share one key format (no duplicate clone of it).
    ``leg`` may be a canonical lowercase leg or a raw uppercase
    ``current_offers`` row — ``leg_field`` bridges the two shapes.
    """
    return f"{leg_field(leg, 'player')}|{leg_field(leg, 'market')}|{leg_field(leg, 'bet')}"


def leg_cell_market(leg: Mapping) -> str:
    """The league's own name for a canonical leg's market: the name its model cell goes by.

    ``market`` is the name the platform or the board gives the leg (NHL ``AST``), so the
    cell is read off ``stat``. That holds the league's key (``assists``), or the board's
    name on a leg saved while the board's ``Stat`` copied its ``Market``; folding through
    :func:`~sportstradamus.helpers.archive_market` brings either to the league's key.
    ``market`` stands in for a leg with no ``stat``.
    """
    return archive_market(leg["league"], leg.get("stat") or leg["market"])


def find_offer_idx(leg: Mapping, offers: pd.DataFrame, platform: str | None = None) -> int | None:
    """Find the offers-frame index for a leg, or ``None`` if it moved.

    ``leg`` is a canonical lowercase leg (already carrying the canonical
    market code, via ``build_leg``) or a raw uppercase ``current_offers`` row;
    ``leg_field`` reads player/bet/market/line off either shape. ``platform``
    is the parlay's book; it pins the match to that book — the same (player,
    market, line) trades on both Underdog and Sleeper with different payout
    multipliers, so a platform-blind match would snapshot the wrong book's
    ``Boost`` and misprice the slip.
    """
    if not leg or offers.empty:
        return None
    mask = (
        (offers["Player"] == leg_field(leg, "player"))
        & (offers["Bet"] == leg_field(leg, "bet"))
        & (offers["Market"] == leg_field(leg, "market"))
        & (pd.to_numeric(offers["Line"], errors="coerce") == leg_field(leg, "line"))
    )
    if platform is not None:
        mask &= offers["Platform"] == platform
    matches = offers.index[mask]
    return int(matches[0]) if len(matches) else None
