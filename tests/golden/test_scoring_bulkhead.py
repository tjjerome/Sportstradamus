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

from sportstradamus.helpers import stat_map
from sportstradamus.prediction import scoring

_BOOK = "Underdog"
_DATE = "2026-09-12"


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

    def get_depth(self, offers):
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
