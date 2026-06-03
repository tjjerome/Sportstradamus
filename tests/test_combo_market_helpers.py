"""Unit tests for the shared combo-market EV helpers on ``Stats`` (base).

The per-league ``check_combo_markets`` overrides are archive-bound and have no
direct automated coverage; these pin the shared kernel they now delegate to
(``_submarket_ev`` / ``_combo_market_ev`` / ``_fantasy_default_contribution``)
with a fake archive and controlled cv/dist tables, so the NBA/NHL/MLB rewires
can't silently drift from the original inline math.
"""

import numpy as np
import pandas as pd
import pytest

from sportstradamus.stats import base


class _FakeArchive:
    def __init__(self, evs=None, lines=None):
        self._evs = evs or {}
        self._lines = lines or {}

    def get_ev(self, league, submarket, date, player):
        return self._evs.get(submarket, np.nan)

    def get_line(self, league, submarket, date, player):
        return self._lines.get(submarket, 0)


@pytest.fixture
def stats(monkeypatch):
    s = base.Stats.__new__(base.Stats)
    s.league = "TEST"
    # Controlled config so cv=1 and every submarket shares the parent's dist
    # (no distribution conversion path) unless a test overrides it.
    monkeypatch.setattr(base, "stat_cv", {"TEST": {}})
    monkeypatch.setattr(base, "stat_dist", {"TEST": {}})
    monkeypatch.setattr(base, "combo_props", {"COMBO": ["A", "B"]})
    return s


def test_submarket_ev_no_conversion(stats, monkeypatch):
    monkeypatch.setattr(base, "archive", _FakeArchive({"A": 1.5}, {"A": 0.5}))
    v, subline, sub_cv, sub_dist = stats._submarket_ev("A", "2026-01-01", "P", "Gamma", 1)
    assert v == 1.5
    assert subline == 0.5
    assert sub_cv == 1
    assert sub_dist == "Gamma"


def test_submarket_ev_missing_is_nan(stats, monkeypatch):
    monkeypatch.setattr(base, "archive", _FakeArchive())
    v, subline, _, _ = stats._submarket_ev("A", "2026-01-01", "P", "Gamma", 1)
    assert np.isnan(v)
    assert subline == 0


def test_submarket_ev_missing_league_no_keyerror(stats, monkeypatch):
    # cv lookup must fall back to 1 when the league is absent from stat_cv,
    # not raise KeyError (regression: stat_cv[self.league] subscript).
    monkeypatch.setattr(base, "stat_cv", {})
    monkeypatch.setattr(base, "archive", _FakeArchive({"A": 1.5}, {"A": 0.5}))
    v, subline, sub_cv, sub_dist = stats._submarket_ev("A", "2026-01-01", "P", "Gamma", 1)
    assert sub_cv == 1
    assert v == 1.5


def test_combo_market_ev_sums_legs(stats, monkeypatch):
    monkeypatch.setattr(base, "archive", _FakeArchive({"A": 1.5, "B": 2.0}))
    assert stats._combo_market_ev("COMBO", "2026-01-01", "P", "Gamma", 1) == pytest.approx(3.5)


def test_combo_market_ev_zero_when_leg_missing(stats, monkeypatch):
    monkeypatch.setattr(base, "archive", _FakeArchive({"A": 1.5}))  # B -> nan
    assert stats._combo_market_ev("COMBO", "2026-01-01", "P", "Gamma", 1) == 0


def test_combo_market_ev_zero_when_leg_zero(stats, monkeypatch):
    monkeypatch.setattr(base, "archive", _FakeArchive({"A": 1.5, "B": 0}))
    assert stats._combo_market_ev("COMBO", "2026-01-01", "P", "Gamma", 1) == 0


def test_fantasy_default_contribution_book(stats):
    games = pd.DataFrame({"PTS": [10, 20, 30]})
    contribution, from_book = stats._fantasy_default_contribution(
        "PTS", 2.0, 5.0, 4.5, 1, "Gamma", games
    )
    assert contribution == pytest.approx(10.0)
    assert from_book is True


def test_fantasy_default_contribution_empirical(stats):
    games = pd.DataFrame({"PTS": [10, 20, 30]})
    contribution, from_book = stats._fantasy_default_contribution(
        "PTS", 2.0, np.nan, 15.5, 1, "Gamma", games
    )
    assert from_book is False
    assert np.isfinite(contribution)


def test_fantasy_default_contribution_no_line_no_history(stats):
    contribution, from_book = stats._fantasy_default_contribution(
        "PTS", 2.0, np.nan, 0, 1, "Gamma", pd.DataFrame()
    )
    assert contribution == 0
    assert from_book is False


def test_nhl_combo_opponent_uses_upcoming_games_key(monkeypatch):
    # Regression: the upcoming-game branch must read upcoming_games[team]["Opponent"]
    # (capital, as the dict is built), not log_strings["opponent"] ("opponent").
    from datetime import datetime, timedelta

    from sportstradamus.stats import nhl

    s = nhl.StatsNHL.__new__(nhl.StatsNHL)
    s.league = "NHL"
    s.log_strings = {"player": "playerName", "team": "team",
                     "opponent": "opponent", "date": "gameDate"}
    s.short_gamelog = pd.DataFrame({"playerName": ["P"], "team": ["BOS"]})
    s.gamelog = pd.DataFrame({"playerName": [], "gameDate": []})
    s.upcoming_games = {"BOS": {"Opponent": "TOR", "Home": True}}

    captured = {}

    class _Arch:
        def get_total(self, league, date, opponent):
            captured["opponent"] = opponent
            return 5.5

    monkeypatch.setattr(nhl, "archive", _Arch())
    future = datetime.today().date() + timedelta(days=1)
    ev = s.check_combo_markets("goalsAgainst", "P", future)

    assert ev == 5.5
    assert captured["opponent"] == "TOR"
