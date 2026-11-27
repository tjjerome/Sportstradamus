"""Bet sizing, Pick'em entry construction, and the simulated-bettor ledger.

``kelly`` sizes stakes, ``underdog_pickem`` builds the entries behind ``bet pickem``, and
``ledger`` commits the simulated bettors' entries. ``profit_sim`` is the Kelly-sized backtest.
Underscore-prefixed modules are internal to this package.
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
