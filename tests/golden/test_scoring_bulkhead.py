"""One cell's crash must not blank a whole DFS platform.

``prediction/cli.py`` scores each platform inside a single try/except, so before
the bulkhead in ``_match_league_offers`` one raising ``(league, market)`` cost that
platform its scoring, its parlays and every ``archive.add_dfs`` capture that had not
yet run — 305 blackouts across 1642 prod runs. Captures now land ahead of anything
that can raise, and the two catches hold the blast radius at one game date's pass
and one market's records.

The passes are pinned here too: one a game date, never before today, its feature
prep ahead of every market's scoring, as the training matrix builds each gameday.
"""

from __future__ import annotations

import logging
import pickle
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
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


@pytest.fixture(autouse=True)
def recorder(monkeypatch):
    """Every test archives into a recorder, never the real DuckDB."""
    monkeypatch.setattr(scoring, "archive", _ArchiveRecorder())
    return scoring.archive


class _Pbar:
    def __init__(self):
        self.count = 0

    def update(self, n):
        self.count += n


class _StatsStub:
    """The ``Stats`` surface league feature prep touches, logging each call in ``events``.

    ``depth_error`` raises from ``get_depth`` on ``failing_day``, or on every day
    when that is None.
    """

    def __init__(self, depth_error=None, failing_day=None):
        self.depth_error = depth_error
        self.failing_day = failing_day
        self.events = []

    def get_depth(self, offers, date):
        self.depth_offers = offers
        self.events.append(("depth", date, _players(offers)))
        if self.depth_error is not None and self.failing_day in (None, date):
            raise self.depth_error

    def get_volume_stats(self, offers, date, pitcher=False):
        self.events.append(("volume", date, _players(offers), pitcher))


def _players(offers):
    return sorted(offer["Player"] for offer in offers)


def _dated_markets(*rows):
    """``{market: [offer, ...]}`` from ``(player, market, Date)`` rows, in row order."""
    markets = {}
    for player, market, game_date in rows:
        markets.setdefault(market, []).append(
            {"Player": player, "Market": market, "Line": 1.5, "Date": game_date}
        )
    return markets


def _nba_markets():
    return _dated_markets(("A", "PTS", _DATE), ("B", "REB", _DATE), ("C", "REB", _DATE))


def test_one_market_crash_costs_only_that_market(monkeypatch, caplog, recorder):
    def score(offers, league, market, book, stat_data, date):
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


def test_feature_prep_crash_scores_nothing_but_keeps_every_capture(monkeypatch, caplog, recorder):
    monkeypatch.setattr(scoring, "_score_market", lambda *args: [{"scored": True}])
    pbar = _Pbar()
    stats = {"NBA": _StatsStub(depth_error=KeyError("proj carries loc"))}

    with caplog.at_level(logging.ERROR, logger="log"):
        scored = scoring._match_league_offers("NBA", _nba_markets(), stats, _BOOK, pbar)

    assert scored == []
    assert recorder.markets == ["PTS", "REB"]
    assert pbar.count == 3
    assert "NBA features unavailable" in caplog.text


def test_inactive_league_is_archived_but_not_scored(monkeypatch, recorder):
    monkeypatch.setattr(scoring, "_score_market", lambda *args: [{"scored": True}])
    pbar = _Pbar()

    scored = scoring._match_league_offers("NBA", _nba_markets(), {}, _BOOK, pbar)

    assert scored == []
    assert recorder.markets == ["PTS", "REB"]
    assert pbar.count == 3


def _score_every_offer(offers, league, market, book, stat_data, date):
    stat_data.events.append(("score", market, date, _players(offers)))
    return [{"Player": offer["Player"], "Market": market} for offer in offers]


def test_preseason_offers_are_archived_but_never_scored(monkeypatch, caplog, recorder):
    monkeypatch.setattr(scoring, "_score_market", _score_every_offer)
    monkeypatch.setattr(scoring.odds_budget, "season_opener", lambda league: date(2026, 9, 13))
    pbar = _Pbar()
    stats = {"NBA": _StatsStub()}

    with caplog.at_level(logging.INFO, logger="log"):
        scored = scoring._match_league_offers("NBA", _nba_markets(), stats, _BOOK, pbar)

    assert scored == []
    assert recorder.markets == ["PTS", "REB"]
    assert pbar.count == 3
    assert stats["NBA"].events == []
    assert "NBA: 3 offers before opening night 2026-09-13 not scored" in caplog.text


def test_opening_night_offers_are_scored(monkeypatch):
    monkeypatch.setattr(scoring, "_score_market", _score_every_offer)
    monkeypatch.setattr(scoring.odds_budget, "season_opener", lambda league: date(2026, 9, 12))

    scored = scoring._match_league_offers(
        "NBA", _nba_markets(), {"NBA": _StatsStub()}, _BOOK, _Pbar()
    )

    assert [record["Player"] for record in scored] == ["A", "B", "C"]


def test_a_market_keeps_only_its_in_season_offers(monkeypatch, recorder):
    monkeypatch.setattr(scoring, "_score_market", _score_every_offer)
    monkeypatch.setattr(scoring.odds_budget, "season_opener", lambda league: date(2026, 9, 12))
    markets = _nba_markets()
    markets["PTS"][0]["Date"] = "2026-09-11"
    markets["REB"][0]["Date"] = "2026-09-11"

    scored = scoring._match_league_offers("NBA", markets, {"NBA": _StatsStub()}, _BOOK, _Pbar())

    assert scored == [{"Player": "C", "Market": "REB"}]
    assert recorder.markets == ["PTS", "REB"]


def test_an_offer_with_no_date_is_never_dropped_as_preseason(monkeypatch):
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


def test_each_game_date_is_one_pass_in_date_order(monkeypatch):
    """Depth, volume and every market's scoring run at one date before the next starts.

    ``base_profile`` rebuilds the profile whenever the date moves, so a call at another
    date inside a pass would leave its scoring on wiped depth and volume columns.
    """
    monkeypatch.setattr(scoring, "_score_market", _score_every_offer)
    today = date.today()
    later = today + timedelta(days=3)
    markets = _dated_markets(
        ("B", "PTS", later.isoformat()),
        ("A", "PTS", today.isoformat()),
        ("C", "REB", later.isoformat()),
        ("A", "REB", later.isoformat()),
    )
    stats = _StatsStub()

    scoring._match_league_offers("NBA", markets, {"NBA": stats}, _BOOK, _Pbar())

    assert stats.events == [
        ("depth", today, ["A"]),
        ("volume", today, ["A"], False),
        ("score", "PTS", today, ["A"]),
        ("score", "REB", today, ["A"]),
        ("depth", later, ["B", "C"]),
        ("volume", later, ["B", "C"], False),
        ("score", "PTS", later, ["B"]),
        ("score", "REB", later, ["C"]),
    ]


def test_a_player_is_scored_in_every_market_in_his_earliest_pass(monkeypatch):
    """His one feature row is built for his earliest game, so his later offers ride along."""
    handed = {}

    def score(offers, league, market, book, stat_data, date):
        handed[market, date] = [(offer["Player"], offer["Date"]) for offer in offers]
        return []

    monkeypatch.setattr(scoring, "_score_market", score)
    today = date.today()
    later = today + timedelta(days=3)
    markets = _dated_markets(
        ("X", "PTS", today.isoformat()),
        ("Y", "REB", later.isoformat()),
        ("X", "REB", later.isoformat()),
    )

    scoring._match_league_offers("NBA", markets, {"NBA": _StatsStub()}, _BOOK, _Pbar())

    assert handed == {
        ("PTS", today): [("X", today.isoformat())],
        ("REB", today): [("X", later.isoformat())],
        ("REB", later): [("Y", later.isoformat())],
    }


def test_past_and_undated_offers_are_built_as_of_today(monkeypatch):
    """A past date would build the row from the realized game; no date has no other day."""
    monkeypatch.setattr(scoring, "_score_market", _score_every_offer)
    today = date.today()
    yesterday = today - timedelta(days=1)
    markets = _dated_markets(("A", "PTS", yesterday.isoformat()), ("B", "PTS", None))
    stats = _StatsStub()

    scoring._match_league_offers("NBA", markets, {"NBA": stats}, _BOOK, _Pbar())

    assert stats.events == [
        ("depth", today, ["A", "B"]),
        ("volume", today, ["A", "B"], False),
        ("score", "PTS", today, ["A", "B"]),
    ]


def test_a_feature_prep_crash_costs_only_its_dates_pass(monkeypatch, caplog):
    monkeypatch.setattr(scoring, "_score_market", _score_every_offer)
    today = date.today()
    later = today + timedelta(days=3)
    markets = _dated_markets(("A", "PTS", today.isoformat()), ("B", "PTS", later.isoformat()))
    stats = _StatsStub(depth_error=KeyError("proj carries loc"), failing_day=today)

    with caplog.at_level(logging.ERROR, logger="log"):
        scored = scoring._match_league_offers("NBA", markets, {"NBA": stats}, _BOOK, _Pbar())

    assert scored == [{"Player": "B", "Market": "PTS"}]
    assert "NBA features unavailable" in caplog.text
    assert today.isoformat() in caplog.text


def test_mlb_prepares_pitcher_volume_in_every_pass(monkeypatch):
    monkeypatch.setattr(scoring, "_score_market", _score_every_offer)
    today = date.today()
    later = today + timedelta(days=1)
    markets = _dated_markets(("A", "hits", today.isoformat()), ("B", "hits", later.isoformat()))
    stats = _StatsStub()

    scoring._match_league_offers("MLB", markets, {"MLB": stats}, _BOOK, _Pbar())

    assert stats.events == [
        ("depth", today, ["A"]),
        ("volume", today, ["A"], False),
        ("volume", today, ["A"], True),
        ("score", "hits", today, ["A"]),
        ("depth", later, ["B"]),
        ("volume", later, ["B"], False),
        ("volume", later, ["B"], True),
        ("score", "hits", later, ["B"]),
    ]


def test_match_offers_builds_as_of_its_date_and_zeroes_inf_as_training_does(monkeypatch, tmp_path):
    """The model reads its features as the training matrix ends them: NaN and ±inf as 0."""
    (tmp_path / "models").mkdir()
    with open(tmp_path / "models" / "NBA_PTS.mdl", "wb") as outfile:
        pickle.dump({"expected_columns": ["MeanYr", "Rate", "Unbuilt"]}, outfile)
    monkeypatch.setattr(scoring.pkg_resources, "files", lambda package: tmp_path)
    asked = []

    def get_stats(market, offers, day):
        asked.append((market, day))
        return pd.DataFrame(
            {
                "MeanYr": [np.nan, 20.0, 99.0],
                "Rate": [np.inf, -np.inf, 5.0],
                "Untrained": [1.0, 2.0, 3.0],
            },
            index=["A", "B", "A"],
        )

    stat_data = SimpleNamespace(gamelog=pd.DataFrame(columns=["PTS"]), get_stats=get_stats)
    later = date.today() + timedelta(days=3)

    built = scoring.match_offers([], "NBA", "PTS", _BOOK, stat_data, later)

    assert asked == [("PTS", later)]
    assert list(built.columns) == ["MeanYr", "Rate", "Unbuilt"]
    assert built.to_dict("index") == {
        "A": {"MeanYr": 0.0, "Rate": 0.0, "Unbuilt": 0.0},
        "B": {"MeanYr": 20.0, "Rate": 0.0, "Unbuilt": 0.0},
    }
