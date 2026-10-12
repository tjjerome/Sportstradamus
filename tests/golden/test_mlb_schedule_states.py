"""An MLB game stays on the schedule map until the feed says it is over.

The feed moves a game to ``Warmup`` minutes before first pitch and to
``Delayed Start: ...`` in bad weather. A filter that knew only ``Scheduled`` and
``Pre-Game`` dropped both, so the last board run before a game scored it with no
starter, no home flag and no posted batting order.

A game called early ends as ``Completed Early`` and is parsed as a ``Final`` one is.
"""

from __future__ import annotations

import pytest

from sportstradamus.helpers import text
from sportstradamus.stats import mlb as stats_mlb

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


def _game(status: str, away_pitcher: str = "Away Arm") -> dict:
    return {
        "status": status,
        "game_type": "R",
        "game_id": 7,
        "game_num": 1,
        "away_id": 1,
        "home_id": 2,
        "away_probable_pitcher": away_pitcher,
        "home_probable_pitcher": "Home Arm",
    }


@pytest.mark.parametrize("status", _NOT_OVER)
def test_game_not_over_keeps_its_schedule_entry(
    monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    monkeypatch.setattr(stats_mlb.mlb, "boxscore_data", lambda game_id: _BOXSCORE)

    upcoming = stats_mlb._build_mlb_upcoming_games([_game(status)], _TEAMS)

    assert upcoming["BOS"]["Home"]
    assert not upcoming["NYY"]["Home"]
    assert upcoming["NYY"]["Opponent Pitcher"] == "Home Arm"
    assert upcoming["NYY"]["Batting Order"] == ["Lead Off"]


@pytest.mark.parametrize("status", _OVER)
def test_game_over_leaves_the_schedule_map(monkeypatch: pytest.MonkeyPatch, status: str) -> None:
    monkeypatch.setattr(stats_mlb.mlb, "boxscore_data", lambda game_id: _BOXSCORE)

    assert stats_mlb._build_mlb_upcoming_games([_game(status)], _TEAMS) == {}


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
