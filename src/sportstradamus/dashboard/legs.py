"""Offer-lookup helpers for parlay legs.

The leg-string parser ``parse_leg`` lives in ``prediction/stories.py`` — the
prediction layer produces those strings — and is re-exported here for the
dashboard surfaces that import it from this module.
"""

import pandas as pd

from sportstradamus.helpers import stat_map
from sportstradamus.prediction.stories import parse_leg

__all__ = ["find_offer_idx", "parse_leg"]


def _candidate_markets(market: str, platform: str | None) -> set[str]:
    """Canonical-code aliases for a leg's display market under a platform.

    Parlay legs carry the platform's source/display market label (Underdog
    "Pts + Rebs + Asts", Sleeper "pts_reb_ast") while ``current_offers.parquet``
    stores the canonical code ("PRA"). ``stat_map[platform]`` is the codebase's
    display→code table; the space-stripped lookup covers Underdog's spaced
    names ("Pts + Rebs + Asts" vs the table key "Pts+Rebs+Asts").
    """
    out = {market}
    pmap = stat_map.get(platform, {}) if platform else {}
    if market in pmap:
        out.add(pmap[market])
    nospace = market.replace(" ", "")
    if nospace in pmap:
        out.add(pmap[nospace])
    return out


def find_offer_idx(parsed: dict, offers: pd.DataFrame, platform: str | None = None) -> int | None:
    """Find the offers-frame index for a parsed leg, or ``None`` if it moved.

    ``platform`` is the parlay's book; it lets the leg's display market be
    translated to the canonical code stored in the offers snapshot.
    """
    if not parsed or offers.empty:
        return None
    markets = _candidate_markets(parsed["Market"], platform)
    mask = (
        (offers["Player"] == parsed["Player"])
        & (offers["Bet"] == parsed["Bet"])
        & (offers["Market"].isin(markets))
        & (pd.to_numeric(offers["Line"], errors="coerce") == parsed["Line"])
    )
    matches = offers.index[mask]
    return int(matches[0]) if len(matches) else None
