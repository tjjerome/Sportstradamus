"""Parlay-leg parsing and offer-lookup helpers."""

import pandas as pd

from sportstradamus.helpers import stat_map


def parse_leg(leg: str) -> dict | None:
    """Parse a parlay leg string into its offer components.

    Leg format is ``"{Player} {Bet} {Line} {Market} - {Model P}%, {Boost}x"``
    (see ``prediction/correlation.py``).  Splitting on the unambiguous
    ``" Over "`` / ``" Under "`` token yields the player name even when it
    contains spaces.  Returns ``None`` for anything that does not parse.
    """
    if not isinstance(leg, str) or not leg.strip():
        return None
    head = leg.split(" - ", 1)[0].strip()
    for bet in ("Over", "Under"):
        token = f" {bet} "
        if token not in head:
            continue
        player, rest = head.split(token, 1)
        rest = rest.strip().split()
        if not rest:
            return None
        try:
            line = float(rest[0])
        except ValueError:
            return None
        market = " ".join(rest[1:]).strip()
        if not player.strip() or not market:
            return None
        return {"Player": player.strip(), "Bet": bet, "Line": line, "Market": market}
    return None


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
