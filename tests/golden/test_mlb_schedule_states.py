"""An MLB game stays on the schedule map until the feed says it is over.

The feed moves a game to ``Warmup`` minutes before first pitch and to
``Delayed Start: ...`` in bad weather. A filter that knew only ``Scheduled`` and
``Pre-Game`` dropped both, so the last board run before a game scored it with no
starter, no home flag and no posted batting order.

A game called early ends as ``Completed Early`` and is parsed as a ``Final`` one is.

The schedule request reaches past today, and a game dated after today is listed by
day only: a later-dated offer reads its own game there, and the next-game map holds
what it did when the request ended today.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from sportstradamus.helpers import text
from sportstradamus.stats import mlb as stats_mlb

_TODAY = date(2026, 10, 11)
_YESTERDAY = "2026-10-10"
_TOMORROW = "2026-10-12"
_TEAMS = {"teams": [{"id": 1, "abbreviation": "NYY"}, {"id": 2, "abbreviation": "BOS"}]}
_BOXSCORE = {
    "playerInfo": {"ID10": {"id": 10, "fullName": "Lead Off"}},
    "away": {"battingOrder": [10]},
    "home": {"battingOrder": []},
}
_NOT_OVER = [
    "Scheduled",
    "Pre-Game",
    "Warmup",
    "Delayed Start: Rain",
    "In Progress",
    "Delayed: Rain",
    "Manager challenge: Tag play",
    "Suspended: Rain",
]
_BOX_SCORE_FINAL = ["Final", "Final: Tied", "Completed Early", "Completed Early: Rain"]
_OVER_WITH_NO_BOX_SCORE = [
    "Game Over",
    "Postponed",
    "Postponed: Rain",
    "Cancelled: Snow",
    "Forfeit: Refusal",
]
_OVER = [*_BOX_SCORE_FINAL, *_OVER_WITH_NO_BOX_SCORE]


def _game(status: str, away_pitcher: str = "Away Arm", **fields) -> dict:
    return {
        "status": status,
        "game_type": "R",
        "game_id": 7,
        "game_num": 1,
        "game_date": _TODAY.isoformat(),
        "away_id": 1,
        "home_id": 2,
        "away_probable_pitcher": away_pitcher,
        "home_probable_pitcher": "Home Arm",
    } | fields


def _builder_before_later_days(mlb_games, mlb_teams):
    """The builder as it was while the schedule request ended today."""
    mlb_upcoming_games = {}
    for game in mlb_games:
        if not text.mlb_game_is_over(game["status"]) and game["game_type"] not in ["A"]:
            awayTeam = stats_mlb._mlb_team_abbr(mlb_teams, game["away_id"])
            homeTeam = stats_mlb._mlb_team_abbr(mlb_teams, game["home_id"])
            game_bs = stats_mlb.mlb.boxscore_data(game["game_id"])
            players = {p["id"]: p["fullName"] for k, p in game_bs["playerInfo"].items()}
            playoff = game["game_type"] in stats_mlb._MLB_POSTSEASON_GAME_TYPES
            away_entry = {
                "Pitcher": text.remove_accents(game["away_probable_pitcher"]),
                "Home": False,
                "Opponent": homeTeam,
                "Opponent Pitcher": text.remove_accents(game["home_probable_pitcher"]),
                "Batting Order": [players[i] for i in game_bs["away"]["battingOrder"]],
                "Playoff": playoff,
                "game type": game["game_type"],
            }
            home_entry = {
                "Pitcher": text.remove_accents(game["home_probable_pitcher"]),
                "Home": True,
                "Opponent": awayTeam,
                "Opponent Pitcher": text.remove_accents(game["away_probable_pitcher"]),
                "Batting Order": [players[i] for i in game_bs["home"]["battingOrder"]],
                "Playoff": playoff,
                "game type": game["game_type"],
            }
            suffix = "" if game["game_num"] == 1 else str(game["game_num"])
            mlb_upcoming_games[awayTeam + suffix] = away_entry
            mlb_upcoming_games[homeTeam + suffix] = home_entry
    return mlb_upcoming_games


@pytest.mark.parametrize("status", _NOT_OVER)
def test_game_not_over_keeps_its_schedule_entry(
    monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    monkeypatch.setattr(stats_mlb.mlb, "boxscore_data", lambda game_id: _BOXSCORE)

    upcoming, _ = stats_mlb._build_mlb_upcoming_games([_game(status)], _TEAMS, _TODAY)

    assert upcoming["BOS"]["Home"]
    assert not upcoming["NYY"]["Home"]
    assert upcoming["NYY"]["Opponent Pitcher"] == "Home Arm"
    assert upcoming["NYY"]["Batting Order"] == ["Lead Off"]


@pytest.mark.parametrize("status", _OVER)
def test_game_over_leaves_the_schedule_map(monkeypatch: pytest.MonkeyPatch, status: str) -> None:
    monkeypatch.setattr(stats_mlb.mlb, "boxscore_data", lambda game_id: _BOXSCORE)

    assert stats_mlb._build_mlb_upcoming_games([_game(status)], _TEAMS, _TODAY) == ({}, {})


def test_a_later_dated_game_is_listed_by_day_only_with_no_box_score_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No lineup is posted a day ahead, so the game costs no box score request."""
    asked = []
    monkeypatch.setattr(stats_mlb.mlb, "boxscore_data", asked.append)

    upcoming, by_date = stats_mlb._build_mlb_upcoming_games(
        [_game("Scheduled", game_date=_TOMORROW, game_type="D")], _TEAMS, _TODAY
    )

    assert upcoming == {}
    assert asked == []
    assert by_date == {
        ("NYY", _TOMORROW): {
            "Pitcher": "Away Arm",
            "Home": False,
            "Opponent": "BOS",
            "Opponent Pitcher": "Home Arm",
            "Batting Order": [],
            "Playoff": True,
            "game type": "D",
        },
        ("BOS", _TOMORROW): {
            "Pitcher": "Home Arm",
            "Home": True,
            "Opponent": "NYY",
            "Opponent Pitcher": "Away Arm",
            "Batting Order": [],
            "Playoff": True,
            "game type": "D",
        },
    }


def _feed_through_today() -> list[dict]:
    return [
        _game("Suspended: Rain", away_pitcher="Suspended Arm", game_date=_YESTERDAY),
        _game("Final", game_date=_YESTERDAY),
        _game("In Progress"),
        _game("Scheduled", away_pitcher="Nightcap Arm", game_num=2),
        _game("Scheduled", away_pitcher="All Star Arm", game_type="A"),
    ]


def test_a_game_through_today_is_also_listed_under_its_own_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under the next-game map's own keys, so the nightcap keeps its suffix."""
    monkeypatch.setattr(stats_mlb.mlb, "boxscore_data", lambda game_id: _BOXSCORE)

    upcoming, by_date = stats_mlb._build_mlb_upcoming_games(_feed_through_today(), _TEAMS, _TODAY)

    today = _TODAY.isoformat()
    assert set(by_date) == {
        ("NYY", _YESTERDAY),
        ("BOS", _YESTERDAY),
        ("NYY", today),
        ("BOS", today),
        ("NYY2", today),
        ("BOS2", today),
    }
    assert by_date["NYY", _YESTERDAY]["Pitcher"] == "Suspended Arm"
    assert by_date["NYY", today] == upcoming["NYY"]
    assert by_date["NYY2", today] == upcoming["NYY2"]
    assert by_date["BOS", today]["Batting Order"] == []
    assert by_date["NYY", today]["Batting Order"] == ["Lead Off"]


def test_the_next_game_map_is_what_it_was_when_the_request_ended_today(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stats_mlb.mlb, "boxscore_data", lambda game_id: _BOXSCORE)
    through_today = _feed_through_today()
    later = [
        _game("Scheduled", away_pitcher="Tomorrow Arm", game_date=_TOMORROW),
        _game("Scheduled", away_pitcher="Third Day Arm", game_date="2026-10-13", game_num=2),
    ]
    before = _builder_before_later_days(through_today, _TEAMS)

    upcoming, _ = stats_mlb._build_mlb_upcoming_games(through_today, _TEAMS, _TODAY)
    upcoming_with_later, by_date = stats_mlb._build_mlb_upcoming_games(
        [*through_today, *later], _TEAMS, _TODAY
    )

    assert before["NYY"]["Pitcher"] == "Away Arm"
    assert before["NYY2"]["Pitcher"] == "Nightcap Arm"
    assert upcoming == before
    assert upcoming_with_later == before
    assert by_date["NYY", _TOMORROW]["Pitcher"] == "Tomorrow Arm"
    assert by_date["NYY2", "2026-10-13"]["Pitcher"] == "Third Day Arm"


@pytest.mark.parametrize(
    ("last_game_days_ago", "request_end_days_from_today"),
    [(1, 2), (90, -30)],
    ids=["up to date", "catching up"],
)
def test_the_schedule_request_reaches_two_days_past_today(
    monkeypatch: pytest.MonkeyPatch, last_game_days_ago: int, request_end_days_from_today: int
) -> None:
    """Both maps are filled from it, and the later day changes nothing else the update does."""
    today = date.today()
    last_game = (today - timedelta(days=last_game_days_ago)).isoformat()
    tomorrow = (today + timedelta(days=1)).isoformat()
    feed = [
        _game("Final", game_id=5, game_date=last_game, game_type="D"),
        _game("Scheduled", game_id=7, game_date=today.isoformat()),
        _game("Scheduled", game_id=8, game_date=tomorrow, away_pitcher="Tomorrow Arm"),
    ]
    requests = []
    people = {"people": []}
    monkeypatch.setattr(stats_mlb.mlb, "schedule", lambda **window: requests.append(window) or feed)
    monkeypatch.setattr(
        stats_mlb.mlb, "get", lambda endpoint, params: _TEAMS if endpoint == "teams" else people
    )
    monkeypatch.setattr(stats_mlb.mlb, "boxscore_data", lambda game_id: _BOXSCORE)
    monkeypatch.setattr(stats_mlb, "write_gamelog", lambda *args: None)
    parsed = []
    stats = stats_mlb.StatsMLB.__new__(stats_mlb.StatsMLB)
    stats.parse_game = parsed.append
    stats.season_start = today - timedelta(days=100)
    stats.players = {}
    stored = {"gameId": [5], "gameDate": [last_game], "team": ["NYY"], "opponent": ["BOS"]}
    stats.gamelog = pd.DataFrame(stored | {"playerId": [10]})
    stats.teamlog = pd.DataFrame(stored)

    stats._update()

    assert requests == [
        {
            "start_date": last_game,
            "end_date": (today + timedelta(days=request_end_days_from_today)).isoformat(),
        }
    ]
    assert stats.upcoming_games["NYY"]["Pitcher"] == "Away Arm"
    assert stats.upcoming_games_by_date["NYY", today.isoformat()] == stats.upcoming_games["NYY"]
    assert stats.upcoming_games_by_date["NYY", tomorrow]["Pitcher"] == "Tomorrow Arm"
    assert parsed == []
    assert stats.gamelog["game type"].tolist() == ["D"]
    assert stats.teamlog["game type"].tolist() == ["D"]


@pytest.mark.parametrize("status", _BOX_SCORE_FINAL)
def test_a_game_whose_box_score_is_final_is_parsed(status: str) -> None:
    """A game called early ends as ``Completed Early`` for good, so nothing later parses it."""
    assert stats_mlb._mlb_final_game_ids([_game(status)], prev_game_ids=[]) == [7]


@pytest.mark.parametrize("status", [*_NOT_OVER, *_OVER_WITH_NO_BOX_SCORE])
def test_a_game_with_no_final_box_score_is_not_parsed(status: str) -> None:
    """``Game Over`` waits: the feed lists that game again as ``Final``."""
    assert stats_mlb._mlb_final_game_ids([_game(status)], prev_game_ids=[]) == []


def test_probable_pitchers_keep_todays_starter_through_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First seen wins per team, so dropping today's game would list tomorrow's starter."""
    schedule = [_game("Warmup"), _game("Scheduled", away_pitcher="Tomorrow Arm")]
    monkeypatch.setattr(text.mlb, "schedule", lambda **_: schedule)
    monkeypatch.setattr(text.mlb, "get", lambda *_: _TEAMS)

    pitchers = text.get_mlb_pitchers.__wrapped__()

    assert pitchers["NYY"] == "Away Arm"
    assert pitchers["BOS"] == "Home Arm"


def test_probable_pitchers_skip_a_finished_game(monkeypatch: pytest.MonkeyPatch) -> None:
    schedule = [_game("Final"), _game("Scheduled", away_pitcher="Tomorrow Arm")]
    monkeypatch.setattr(text.mlb, "schedule", lambda **_: schedule)
    monkeypatch.setattr(text.mlb, "get", lambda *_: _TEAMS)

    assert text.get_mlb_pitchers.__wrapped__()["NYY"] == "Tomorrow Arm"
