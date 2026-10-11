"""Stats package: per-league player statistics and feature engineering.

``sportstradamus.stats`` is the canonical import path for the ``Stats`` classes
and ``load_live_leagues``, re-exported here from the submodules that define them.
"""

from sportstradamus.stats.base import Stats
from sportstradamus.stats.live import load_live_leagues
from sportstradamus.stats.mlb import StatsMLB
from sportstradamus.stats.nba import StatsNBA
from sportstradamus.stats.nfl import StatsNFL
from sportstradamus.stats.nhl import StatsNHL
from sportstradamus.stats.wnba import StatsWNBA

__all__ = [
    "Stats",
    "StatsMLB",
    "StatsNBA",
    "StatsNFL",
    "StatsNHL",
    "StatsWNBA",
    "load_live_leagues",
]
