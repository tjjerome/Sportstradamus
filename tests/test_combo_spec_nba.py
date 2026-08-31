"""Pin ``StatsNBA._fantasy_combo_spec``: pure-marginal DFS specs for the two fantasy markets.

The spec's weight table (``NBA_FANTASY_WEIGHTS``) must stay in lockstep with the settled
gamelog formula in ``StatsNBA._compute_derived_player_stats``; the finite-difference test
extracts each coefficient from that formula directly so the two can't silently diverge.
"""

import pytest

from sportstradamus.stats.base import ComboSpec
from sportstradamus.stats.nba import NBA_FANTASY_WEIGHTS, StatsNBA

FANTASY_MARKETS = ("fantasy points prizepicks", "fantasy points underdog")


def _settled_game(**overrides):
    game = dict.fromkeys(
        ("PTS", "REB", "AST", "BLK", "STL", "TOV", "FTM", "FGA", "FG3A", "BLKA", "OREB", "DREB"),
        0,
    )
    game["MIN"] = 30
    game.update(overrides)
    StatsNBA._compute_derived_player_stats(game)
    return game


@pytest.mark.parametrize("market", FANTASY_MARKETS)
def test_fantasy_markets_get_pure_marginal_spec(market):
    spec = object.__new__(StatsNBA)._fantasy_combo_spec(market)
    assert isinstance(spec, ComboSpec)
    assert spec.marginals == NBA_FANTASY_WEIGHTS
    assert spec.sampled == ()
    assert spec.bernoulli == ()
    assert spec.post_builder is None
    assert spec.analytics == ()


@pytest.mark.parametrize("market", ["PRA", "PTS", "fantasy points parlay"])
def test_other_markets_return_none(market):
    # Simple combos ride combo_props and legacy fantasy/proration paths stay untouched.
    assert object.__new__(StatsNBA)._fantasy_combo_spec(market) is None


@pytest.mark.parametrize("market", FANTASY_MARKETS)
def test_weights_match_settled_formula(market):
    # The settled formula is linear with zero intercept, so a unit bump per component
    # recovers its coefficient exactly; together these pin the whole weight table.
    base_game = _settled_game()
    assert base_game[market] == 0
    for submarket, weight in NBA_FANTASY_WEIGHTS:
        bumped = _settled_game(**{submarket: 1})
        assert bumped[market] - base_game[market] == pytest.approx(weight)
