"""One cell's crash must not blank a whole DFS platform.

``prediction/cli.py`` scores each platform inside a single try/except, so before
the bulkhead in ``_match_league_offers`` one raising ``(league, market)`` cost that
platform its scoring, its parlays and every ``archive.add_dfs`` capture that had not
yet run — 305 blackouts across 1642 prod runs. Captures now land ahead of anything
that can raise, and the two catches hold the blast radius at one league's scoring
and one market's records.
"""

from __future__ import annotations

import logging
from datetime import date

import pytest

from sportstradamus.helpers import stat_map
from sportstradamus.prediction import scoring

_BOOK = "Underdog"
_DATE = "2026-09-12"


@pytest.fixture(autouse=True)
def no_opening_night(monkeypatch):
    """No opening night known unless a test sets one, whatever feed snapshot the box holds."""
    monkeypatch.setattr(scoring.odds_budget, "season_opener", lambda league: None)


class _ArchiveRecorder:
    """Stands in for the module's ``LazyArchive``, recording every ``add_dfs`` batch."""

    def __init__(self):
        self.calls = []

    def add_dfs(self, offers, platform, key):
        self.calls.append((platform, key, list(offers)))

    @property
    def markets(self):
        return sorted({offer["Market"] for _, _, batch in self.calls for offer in batch})


class _Pbar:
    def __init__(self):
        self.count = 0

    def update(self, n):
        self.count += n


class _StatsStub:
    """The ``Stats`` surface league feature prep touches, nothing else."""

    def __init__(self, depth_error=None):
        self.depth_error = depth_error
        self.depth_calls = 0

    def get_depth(self, offers):
        self.depth_calls += 1
        self.depth_offers = offers
        if self.depth_error is not None:
            raise self.depth_error

    def get_volume_stats(self, offers, pitcher=False):
        pass


def _nba_markets():
    return {
        market: [
            {"Player": player, "Market": market, "League": "NBA", "Line": 1.5, "Date": _DATE}
            for player in players
        ]
        for market, players in (("PTS", ["A"]), ("REB", ["B", "C"]))
    }


def test_one_market_crash_costs_only_that_market(monkeypatch, caplog):
    recorder = _ArchiveRecorder()
    monkeypatch.setattr(scoring, "archive", recorder)

    def score(offers, league, market, book, stat_data):
        if market == "PTS":
            raise ValueError("unservable cell")
        return [{"Player": offer["Player"], "Market": market} for offer in offers]

    monkeypatch.setattr(scoring, "_score_market", score)
    pbar = _Pbar()
    stats = {"NBA": _StatsStub()}

    with caplog.at_level(logging.ERROR, logger="log"):
        scored = scoring._match_league_offers("NBA", _nba_markets(), stats, _BOOK, pbar)

    assert [record["Player"] for record in scored] == ["B", "C"]
    assert recorder.markets == ["PTS", "REB"]
    assert [platform for platform, _, _ in recorder.calls] == [_BOOK, _BOOK]
    assert recorder.calls[0][1] is stat_map[_BOOK]
    assert pbar.count == 3
    assert "NBA PTS not scored" in caplog.text


def test_feature_prep_crash_costs_only_that_leagues_scoring(monkeypatch, caplog):
    recorder = _ArchiveRecorder()
    monkeypatch.setattr(scoring, "archive", recorder)
    monkeypatch.setattr(scoring, "_score_market", lambda *args: [{"scored": True}])
    pbar = _Pbar()
    stats = {"NBA": _StatsStub(depth_error=KeyError("proj carries loc"))}

    with caplog.at_level(logging.ERROR, logger="log"):
        scored = scoring._match_league_offers("NBA", _nba_markets(), stats, _BOOK, pbar)

    assert scored == []
    assert recorder.markets == ["PTS", "REB"]
    assert pbar.count == 3
    assert "NBA features unavailable" in caplog.text


def test_inactive_league_is_archived_but_not_scored(monkeypatch):
    recorder = _ArchiveRecorder()
    monkeypatch.setattr(scoring, "archive", recorder)
    monkeypatch.setattr(scoring, "_score_market", lambda *args: [{"scored": True}])
    pbar = _Pbar()

    scored = scoring._match_league_offers("NBA", _nba_markets(), {}, _BOOK, pbar)

    assert scored == []
    assert recorder.markets == ["PTS", "REB"]
    assert pbar.count == 3


def _score_every_offer(offers, league, market, book, stat_data):
    return [{"Player": offer["Player"], "Market": market} for offer in offers]


def test_preseason_offers_are_archived_but_never_scored(monkeypatch, caplog):
    recorder = _ArchiveRecorder()
    monkeypatch.setattr(scoring, "archive", recorder)
    monkeypatch.setattr(scoring, "_score_market", _score_every_offer)
    monkeypatch.setattr(scoring.odds_budget, "season_opener", lambda league: date(2026, 9, 13))
    pbar = _Pbar()
    stats = {"NBA": _StatsStub()}

    with caplog.at_level(logging.INFO, logger="log"):
        scored = scoring._match_league_offers("NBA", _nba_markets(), stats, _BOOK, pbar)

    assert scored == []
    assert recorder.markets == ["PTS", "REB"]
    assert pbar.count == 3
    assert stats["NBA"].depth_calls == 0
    assert "NBA: 3 offers before opening night 2026-09-13 not scored" in caplog.text


def test_opening_night_offers_are_scored(monkeypatch):
    monkeypatch.setattr(scoring, "archive", _ArchiveRecorder())
    monkeypatch.setattr(scoring, "_score_market", _score_every_offer)
    monkeypatch.setattr(scoring.odds_budget, "season_opener", lambda league: date(2026, 9, 12))

    scored = scoring._match_league_offers(
        "NBA", _nba_markets(), {"NBA": _StatsStub()}, _BOOK, _Pbar()
    )

    assert [record["Player"] for record in scored] == ["A", "B", "C"]


def test_a_market_keeps_only_its_in_season_offers(monkeypatch):
    recorder = _ArchiveRecorder()
    monkeypatch.setattr(scoring, "archive", recorder)
    monkeypatch.setattr(scoring, "_score_market", _score_every_offer)
    monkeypatch.setattr(scoring.odds_budget, "season_opener", lambda league: date(2026, 9, 12))
    markets = _nba_markets()
    markets["PTS"][0]["Date"] = "2026-09-11"
    markets["REB"][0]["Date"] = "2026-09-11"

    scored = scoring._match_league_offers("NBA", markets, {"NBA": _StatsStub()}, _BOOK, _Pbar())

    assert scored == [{"Player": "C", "Market": "REB"}]
    assert recorder.markets == ["PTS", "REB"]


def test_an_offer_with_no_date_is_never_dropped_as_preseason(monkeypatch):
    monkeypatch.setattr(scoring, "archive", _ArchiveRecorder())
    monkeypatch.setattr(scoring, "_score_market", _score_every_offer)
    monkeypatch.setattr(scoring.odds_budget, "season_opener", lambda league: date(2026, 9, 13))
    markets = _nba_markets()
    markets["PTS"][0]["Date"] = None

    scored = scoring._match_league_offers("NBA", markets, {"NBA": _StatsStub()}, _BOOK, _Pbar())

    assert scored == [{"Player": "A", "Market": "PTS"}]


def test_feature_prep_gets_each_players_earliest_dated_offer(monkeypatch):
    """Features are one row a player, built for his earlier game, so depth and volume
    have to be prepared for that game too.
    """
    monkeypatch.setattr(scoring, "archive", _ArchiveRecorder())
    monkeypatch.setattr(scoring, "_score_market", _score_every_offer)
    later = "2026-09-13"
    markets = _nba_markets()
    markets["PTS"].append(markets["PTS"][0] | {"Date": later})
    markets["REB"][0]["Date"] = later
    stats = {"NBA": _StatsStub()}

    scoring._match_league_offers("NBA", markets, stats, _BOOK, _Pbar())

    prepared = {offer["Player"]: offer["Date"] for offer in stats["NBA"].depth_offers}
    assert prepared == {"A": _DATE, "B": later, "C": _DATE}
    assert len(stats["NBA"].depth_offers) == 3
