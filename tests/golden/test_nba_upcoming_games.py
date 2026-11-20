"""The NBA/WNBA schedule read behind serve-time ``Home``: ``fetch_upcoming_games``.

It reads ``ScheduleLeagueV2``, which lists every game of the season, through
``nba_client.fetch``. The broadcast schedule it replaced listed televised games only, so
most home teams were served as away. The stub answers in the live response's shape,
trimmed to the keys the read uses: ``leagueSchedule.gameDates[].games[]``, each game with
its US Eastern ``gameDateEst`` and each side's ``teamTricode``. Nothing here touches the
network.
"""

import logging
from datetime import date
from types import SimpleNamespace

import pytest
from nba_api.stats.endpoints import scheduleleaguev2

from sportstradamus.stats import nba_client
from sportstradamus.stats.base import fetch_upcoming_games
from sportstradamus.stats.nba_client import NBAStatsError

_TODAY = date(2026, 11, 20)


def _game(day: str, away: str | None, home: str | None) -> dict:
    return {
        "gameDateEst": f"{day}T00:00:00Z",
        "awayTeam": {"teamTricode": away},
        "homeTeam": {"teamTricode": home},
    }


def _schedule(*games: dict) -> dict:
    """The live payload: each day's games grouped under it, days in date order."""
    days: dict[str, list[dict]] = {}
    for game in games:
        days.setdefault(game["gameDateEst"][:10], []).append(game)
    return {"leagueSchedule": {"gameDates": [{"games": games} for games in days.values()]}}


# Yesterday and the day after the window bracket its three days. OKC plays twice inside it,
# New Orleans comes as the schedule's NOP, and an NBA Cup placeholder names no teams yet.
_NBA_SCHEDULE = _schedule(
    _game("2026-11-19", "BOS", "NYK"),
    _game("2026-11-20", "NOP", "OKC"),
    _game("2026-11-21", "LAL", "BOS"),
    _game("2026-11-22", "OKC", "MIA"),
    _game("2026-11-22", None, None),
    _game("2026-11-23", "LAL", "OKC"),
)
# Portland comes as the schedule's PDX.
_WNBA_SCHEDULE = _schedule(_game("2026-11-20", "PDX", "SEA"))


@pytest.fixture
def schedule_calls(monkeypatch) -> list[tuple]:
    """Answer ``nba_client.fetch`` with the asked league's schedule; record each call."""
    calls = []
    payloads = {"00": _NBA_SCHEDULE, "10": _WNBA_SCHEDULE}

    def fetch(endpoint_cls, *, host, **params):
        calls.append((endpoint_cls, host, params))
        return SimpleNamespace(get_dict=lambda: payloads[params["league_id"]])

    monkeypatch.setattr(nba_client, "fetch", fetch)
    return calls


def test_every_game_in_the_window_lists_both_teams(schedule_calls):
    _, by_date = fetch_upcoming_games("NBA", "2026-27", _TODAY)

    assert by_date == {
        ("OKC", "2026-11-20"): {"Opponent": "NO", "Home": True},
        ("NO", "2026-11-20"): {"Opponent": "OKC", "Home": False},
        ("BOS", "2026-11-21"): {"Opponent": "LAL", "Home": True},
        ("LAL", "2026-11-21"): {"Opponent": "BOS", "Home": False},
        ("MIA", "2026-11-22"): {"Opponent": "OKC", "Home": True},
        ("OKC", "2026-11-22"): {"Opponent": "MIA", "Home": False},
    }


def test_a_teams_first_game_in_the_window_is_its_next_game(schedule_calls):
    upcoming, _ = fetch_upcoming_games("NBA", "2026-27", _TODAY)

    # OKC's home game today wins over its road game in two days; BOS and LAL keep their
    # game inside the window, not yesterday's or the one past it.
    assert upcoming == {
        "OKC": {"Opponent": "NO", "Home": True},
        "NO": {"Opponent": "OKC", "Home": False},
        "BOS": {"Opponent": "LAL", "Home": True},
        "LAL": {"Opponent": "BOS", "Home": False},
        "MIA": {"Opponent": "OKC", "Home": True},
    }


@pytest.mark.parametrize(
    ("league", "season", "league_id", "host"),
    [("NBA", "2026-27", "00", nba_client.NBA_HOST), ("WNBA", 2026, "10", nba_client.WNBA_HOST)],
)
def test_each_league_reads_its_own_schedule_on_its_own_host(
    schedule_calls, league, season, league_id, host
):
    fetch_upcoming_games(league, season, _TODAY)

    assert schedule_calls == [
        (scheduleleaguev2.ScheduleLeagueV2, host, {"league_id": league_id, "season": season})
    ]


def test_wnba_teams_carry_the_offers_codes(schedule_calls):
    upcoming, _ = fetch_upcoming_games("WNBA", 2026, _TODAY)

    assert upcoming == {
        "SEA": {"Opponent": "POR", "Home": True},
        "POR": {"Opponent": "SEA", "Home": False},
    }


def test_failed_fetch_leaves_both_maps_empty(monkeypatch, caplog):
    def fail(*_args, **_kwargs):
        raise NBAStatsError("ScheduleLeagueV2 failed after 5 attempts")

    monkeypatch.setattr(nba_client, "fetch", fail)
    with caplog.at_level(logging.WARNING, logger="log"):
        assert fetch_upcoming_games("NBA", "2026-27", _TODAY) == ({}, {})

    assert "NBA schedule unavailable" in caplog.text
