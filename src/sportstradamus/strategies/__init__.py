"""Bet-sizing and contest-construction strategies.

Phase 3 §3.1 introduces fractional-Kelly sizing here; subsequent phases will
add the Underdog ``pickem-build`` orchestrator alongside it.
"""

from sportstradamus.strategies.kelly import (
    DEFAULT_KELLY_FRACTION,
    LIVE_BLEND_FLOOR,
    LIVE_BLEND_FULL,
    MAX_FRACTION_OF_BANKROLL,
    SHRINKAGE_FLOOR,
    KellyCandidate,
    fractional_kelly_stake,
    joint_kelly_portfolio,
    resolve_shrinkage,
)

__all__ = [
    "DEFAULT_KELLY_FRACTION",
    "LIVE_BLEND_FLOOR",
    "LIVE_BLEND_FULL",
    "MAX_FRACTION_OF_BANKROLL",
    "SHRINKAGE_FLOOR",
    "KellyCandidate",
    "fractional_kelly_stake",
    "joint_kelly_portfolio",
    "resolve_shrinkage",
]
