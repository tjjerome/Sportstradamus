"""Stats package: per-league player statistics and feature engineering.

Public API re-exports for back-compat with any code that imported from
``sportstradamus.stats`` directly.
"""

from sportstradamus.stats.base import Stats
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
]
