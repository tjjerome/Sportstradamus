"""Complete, fail-closed NHL training-market position policy."""

from __future__ import annotations

NHL_POSITION_CODES = {"C": 1, "W": 2, "D": 3, "G": 4}

_SKATER_POSITIONS = frozenset(("C", "W", "D"))
_GOALIE_POSITIONS = frozenset(("G",))
_ALL_POSITIONS = frozenset(NHL_POSITION_CODES)

NHL_MARKET_POSITIONS: dict[str, frozenset[str]] = {
    "timeOnIce": _ALL_POSITIONS,
    "shotsAgainst": _GOALIE_POSITIONS,
    "saves": _GOALIE_POSITIONS,
    "shots": _SKATER_POSITIONS,
    "points": _SKATER_POSITIONS,
    "goalsAgainst": _GOALIE_POSITIONS,
    "goalie fantasy points underdog": _GOALIE_POSITIONS,
    "skater fantasy points underdog": _SKATER_POSITIONS,
    "blocked": _SKATER_POSITIONS,
    "powerPlayPoints": _SKATER_POSITIONS,
    "sogBS": _SKATER_POSITIONS,
    "hits": _SKATER_POSITIONS,
    "goals": _SKATER_POSITIONS,
    "assists": _SKATER_POSITIONS,
    "faceOffWins": _SKATER_POSITIONS,
}


def allowed_nhl_positions(market: str) -> frozenset[str]:
    """Return eligible positions, rejecting unclassified NHL markets."""
    try:
        return NHL_MARKET_POSITIONS[market]
    except KeyError:
        raise ValueError(f"NHL market has no position policy: {market!r}") from None


def allowed_nhl_position_codes(market: str) -> frozenset[int]:
    """Return the one-based matrix codes eligible for ``market``."""
    return frozenset(NHL_POSITION_CODES[position] for position in allowed_nhl_positions(market))
