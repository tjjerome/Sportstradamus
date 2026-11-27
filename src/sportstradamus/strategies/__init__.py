"""Bet-sizing and contest-construction strategies.

Phase 3 §3.1 introduces fractional-Kelly sizing here; §3.3 adds the
:mod:`underdog_pickem` orchestrator behind ``bet pickem``.
"""

from sportstradamus.strategies.kelly import (
    DEFAULT_KELLY_FRACTION,
    MAX_FRACTION_OF_BANKROLL,
    SHRINKAGE_FLOOR,
    KellyCandidate,
    fractional_kelly_stake,
    joint_kelly_portfolio,
)
from sportstradamus.strategies.underdog_pickem import (
    PickemConfig,
    RecommendedEntry,
    construct_entries,
)

__all__ = [
    "DEFAULT_KELLY_FRACTION",
    "MAX_FRACTION_OF_BANKROLL",
    "SHRINKAGE_FLOOR",
    "KellyCandidate",
    "PickemConfig",
    "RecommendedEntry",
    "construct_entries",
    "fractional_kelly_stake",
    "joint_kelly_portfolio",
]
