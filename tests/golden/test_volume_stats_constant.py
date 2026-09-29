"""``VOLUME_STATS`` is the one list of serve-time volume denominators: the Stats
classes project from it and ``prune_model_pickle`` keeps its pickles."""

import pytest

from sportstradamus.helpers.io import VOLUME_STATS
from sportstradamus.stats import StatsMLB, StatsNBA, StatsNFL, StatsNHL, StatsWNBA
from sportstradamus.training.markets import ALL_MARKETS


@pytest.mark.parametrize("cls", [StatsNBA, StatsWNBA, StatsMLB, StatsNFL, StatsNHL])
def test_volume_stats_read_the_shared_constant(cls):
    stats = cls(load_live_pitchers=False) if cls is StatsMLB else cls()
    assert tuple(stats.volume_stats) == VOLUME_STATS[stats.league]


def test_volume_stats_are_trainable_markets():
    assert set(VOLUME_STATS) == set(ALL_MARKETS)
    for league, markets in VOLUME_STATS.items():
        assert set(markets) <= set(ALL_MARKETS[league]), league
