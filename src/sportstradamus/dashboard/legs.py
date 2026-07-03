"""Offer-lookup helpers for parlay legs.

The leg-string parser ``parse_leg`` lives in ``prediction/stories.py`` and is
re-exported here; no dashboard surface still calls it (every leg producer now
emits structured legs directly) but its own unit pins keep importing it from
this module.
"""

from collections.abc import Mapping

import pandas as pd

from sportstradamus.leg_schema import leg_field
from sportstradamus.prediction.stories import parse_leg

__all__ = ["corr_key", "find_offer_idx", "parse_leg"]


def corr_key(leg: Mapping) -> str:
    """Canonical correlation-slice key for a leg: ``player|market|bet``.

    Matches ``current_game_corr``'s ``leg_a``/``leg_b`` keys so the slip engine
    and the constellation share one key format (no duplicate clone of it).
    ``leg`` may be a canonical lowercase leg or a raw uppercase
    ``current_offers`` row — ``leg_field`` bridges the two shapes.
    """
    return f"{leg_field(leg, 'player')}|{leg_field(leg, 'market')}|{leg_field(leg, 'bet')}"


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
