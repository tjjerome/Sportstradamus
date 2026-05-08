"""Bet-sizing and contest-construction strategies.

Phase 3 §3.1 introduces fractional-Kelly sizing here; §3.3 + §3.5 Rivals add
the :mod:`underdog_pickem` orchestrator that drives ``pickem-build``.
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
from sportstradamus.strategies.underdog_pickem import (
    PickemConfig,
    RecommendedEntry,
    construct_entries,
)

__all__ = [
    "DEFAULT_KELLY_FRACTION",
    "LIVE_BLEND_FLOOR",
    "LIVE_BLEND_FULL",
    "MAX_FRACTION_OF_BANKROLL",
    "SHRINKAGE_FLOOR",
    "KellyCandidate",
    "PickemConfig",
    "RecommendedEntry",
    "construct_entries",
    "fractional_kelly_stake",
    "joint_kelly_portfolio",
    "resolve_shrinkage",
]
