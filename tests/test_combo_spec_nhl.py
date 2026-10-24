"""Pin ``StatsNHL._fantasy_combo_spec`` and the goalie ``win`` Bernoulli resolver.

The three weight tables must stay in lockstep with the settled gamelog formulas in
``StatsNHL._skater_row``; the settlement golden cross-pins that. Here the specs are
pinned against the module constants and the constants against the platform scoring
rules, plus the goalie parameterization identity and the moneyline clamp.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from sportstradamus.stats import nhl
from sportstradamus.stats.base import ComboSpec
from sportstradamus.stats.nhl import (
    NHL_ASSUMABLE_SKATER_COMPONENTS,
    NHL_GOALIE_UNDERDOG_BERNOULLI,
    NHL_GOALIE_UNDERDOG_WEIGHTS,
    NHL_SKATER_PRIZEPICKS_WEIGHTS,
    NHL_SKATER_UNDERDOG_WEIGHTS,
    StatsNHL,
)


def bare_stats():
    return object.__new__(StatsNHL)


def test_skater_underdog_spec():
    spec = bare_stats()._fantasy_combo_spec("skater fantasy points underdog", "Any Skater")
    assert spec == ComboSpec(
        marginals=NHL_SKATER_UNDERDOG_WEIGHTS,
        assumable=NHL_ASSUMABLE_SKATER_COMPONENTS,
    )
    # No sportsbook prices skater hits, so it is the one component allowed to fall
    # back to the player's trailing rate; the rest must be book-quoted.
    assert set(NHL_ASSUMABLE_SKATER_COMPONENTS) == {"hits"}
    assert NHL_SKATER_UNDERDOG_WEIGHTS == (
        ("goals", 6),
        ("assists", 4),
        ("shots", 1),
        ("blocked", 1),
        ("hits", 0.5),
        ("powerPlayPoints", 0.5),
    )


def test_skater_prizepicks_spec():
    spec = bare_stats()._fantasy_combo_spec("fantasy points prizepicks", "Any Skater")
    assert spec == ComboSpec(marginals=NHL_SKATER_PRIZEPICKS_WEIGHTS)
    assert NHL_SKATER_PRIZEPICKS_WEIGHTS == (
        ("goals", 8),
        ("assists", 5),
        ("shots", 1.5),
        ("blocked", 1.5),
    )


def test_goalie_underdog_spec():
    spec = bare_stats()._fantasy_combo_spec("goalie fantasy points underdog", "Any Goalie")
    assert spec == ComboSpec(
        marginals=NHL_GOALIE_UNDERDOG_WEIGHTS,
        bernoulli=NHL_GOALIE_UNDERDOG_BERNOULLI,
        analytics=("win_ml",),
    )
    assert NHL_GOALIE_UNDERDOG_WEIGHTS == (("saves", 0.6), ("goalsAgainst", -3))
    assert NHL_GOALIE_UNDERDOG_BERNOULLI == (("win", 6),)


def test_goalie_quoted_pair_equals_shots_against_form():
    # saves + goalsAgainst == shotsAgainst on every goalie row, so the quoted-pair
    # spec settles identically to the brief's 0.6*SA - 3.6*GA re-parameterization
    # (kept on saves/goalsAgainst because shotsAgainst has no book quotes).
    saves = np.array([31.0, 24.0, 40.0, 0.0])
    goals_against = np.array([2.0, 0.0, 5.0, 1.0])
    shots_against = saves + goals_against
    np.testing.assert_allclose(
        0.6 * saves - 3 * goals_against, 0.6 * shots_against - 3.6 * goals_against
    )


@pytest.mark.parametrize("market", ["sogBS", "goalie fantasy points parlay", "saves"])
def test_other_markets_return_none(market):
    # sogBS rides combo_props; parlay fantasy and plain props stay off the spec path.
    assert bare_stats()._fantasy_combo_spec(market, "Any Skater") is None


class _StubArchive:
    def __init__(self, moneyline):
        self.moneyline = moneyline
        self.calls = []

    def get_moneyline(self, league, date, team):
        self.calls.append((league, date, team))
        return self.moneyline


@pytest.mark.parametrize(("raw", "clamped"), [(0.999, 0.99), (0.001, 0.01), (0.62, 0.62)])
def test_combo_bernoulli_p_clamps_team_moneyline(monkeypatch, raw, clamped):
    stats = bare_stats()
    stats.log_strings = {"player": "playerName", "team": "team"}
    stats.short_gamelog = pd.DataFrame({"playerName": ["Stub Goalie"], "team": ["COL"]})
    stub = _StubArchive(raw)
    monkeypatch.setattr(nhl, "archive", stub)
    day = datetime.today().date().strftime("%Y-%m-%d")

    assert stats._combo_bernoulli_p("win", "Stub Goalie", day) == pytest.approx(clamped)
    assert stub.calls == [("NHL", day, "COL")]
