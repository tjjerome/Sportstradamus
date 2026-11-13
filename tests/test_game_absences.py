"""Game absences: NBA and WNBA inactives and box-score comments, NHL scratches, resume, failures."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from sportstradamus.collectors import game_absences as ga
from sportstradamus.stats.nba_client import NBA_HOST, WNBA_HOST, NBAStatsError

# The file's columns, spelled out here so a change to the module's copy fails a test.
_DTYPES = {
    "game_id": "string",
    "game_date": "datetime64[ns]",
    "team": "string",
    "player_id": "Int64",
    "player_name": "string",
    "status": "string",
}

# One game's box scores per side: (tricode, BoxScoreSummaryV3 inactives, BoxScoreTraditionalV3
# players). A player is (personId, firstName, familyName); a box-score one adds its comment, ""
# for one who played.
_IND_AT_CHA = {
    "awayTeam": (
        "IND",
        [(204456, "T.J.", "McConnell")],
        [
            (1627783, "Pascal", "Siakam", ""),
            (1629614, "Andrew", "Nembhard", "DNP - Injury/Illness"),
        ],
    ),
    "homeTeam": (
        "CHA",
        [(1628970, "Miles", "Bridges"), (1630208, "Nick", "Richards")],
        [(203995, "Vasilije", "Micić", "DNP - Coach's Decision"), (1630182, "Josh", "Green", "")],
    ),
}
_NOBODY_SAT = {
    "awayTeam": ("IND", [], [(1627783, "Pascal", "Siakam", "")]),
    "homeTeam": ("CHA", [], [(1630182, "Josh", "Green", "")]),
}
# Three NBA game days of 2024-25, each gamelog row as the parquet stores it.
_WEEK = [
    ("0022400184", "2024-11-08T00:00:00", 1627783),
    ("0022400190", "2024-11-09T00:00:00", 1627783),
    ("0022400201", "2024-11-10T00:00:00", 1627783),
]

# The right rail's gameInfo for FLA at BOS, 2025-10-21: names are locale maps.
_FLA_AT_BOS = {
    "awayTeam": {
        "headCoach": {"default": "Paul Maurice"},
        "scratches": [
            {
                "id": 8481600,
                "firstName": {"default": "Tobias"},
                "lastName": {"default": "Bjornfot", "cs": "Björnfot"},
            },
            {"id": 8481655, "firstName": {"default": "Cole"}, "lastName": {"default": "Schwindt"}},
        ],
    },
    "homeTeam": {
        "headCoach": {"default": "Marco Sturm"},
        "scratches": [
            {"id": 8476854, "firstName": {"default": "Hampus"}, "lastName": {"default": "Lindholm"}}
        ],
    },
}
# WPG at TB, 2025-10-23: Tampa Bay's side carries no scratches key.
_WPG_AT_TB = {
    "awayTeam": {
        "headCoach": {"default": "Scott Arniel"},
        "scratches": [
            {"id": 8482459, "firstName": {"default": "Tyrel"}, "lastName": {"default": "Bauer"}}
        ],
    },
    "homeTeam": {"headCoach": {"default": "Jon Cooper"}},
}
# NHL gamelog rows: (gameId, gameDate, team, opponent, home, playerId), in our spelling (TB).
_BOS = (2025020103, "2025-10-21", "BOS", "FLA", True, "8478498")
_FLA = (2025020103, "2025-10-21", "FLA", "BOS", False, "8477493")
_TB = (2025020120, "2025-10-23", "TB", "WPG", True, "8476453")
_WPG = (2025020120, "2025-10-23", "WPG", "TB", False, "8476460")
_OTT_2022 = (2022020244, "2022-11-14", "OTT", "NYI", True, "8478472")
_RIGHT_RAIL = "https://api-web.nhle.com/v1/gamecenter/{}/right-rail"


def _gamelog(monkeypatch, tmp_path, league: str, rows: list[tuple]) -> None:
    """Write ``rows`` as the league's gamelog under ``tmp_path`` and point the module there."""
    if league == "NHL":
        columns = ["gameId", "gameDate", "team", "opponent", "home", "playerId"]
    else:
        columns = ["GAME_ID", "GAME_DATE", "PLAYER_ID"]
    (tmp_path / league.lower()).mkdir(exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_parquet(tmp_path / league.lower() / "gamelog.parquet")
    monkeypatch.setattr(ga, "_LEAGUES_DIR", tmp_path)


def _replies(box_scores: dict) -> dict[str, dict]:
    """One game's two replies by endpoint name, shaped as ``get_dict`` returns them."""
    summary, traditional = {}, {}
    for side, (team, inactives, players) in box_scores.items():
        summary[side] = {
            "teamTricode": team,
            "inactives": [
                {"personId": pid, "firstName": first, "familyName": family, "jerseyNum": "0   "}
                for pid, first, family in inactives
            ],
        }
        traditional[side] = {
            "teamTricode": team,
            "players": [
                {"personId": pid, "firstName": first, "familyName": family, "comment": comment}
                for pid, first, family, comment in players
            ],
        }
    return {
        "BoxScoreSummaryV3": {"boxScoreSummary": summary},
        "BoxScoreTraditionalV3": {"boxScoreTraditional": traditional},
    }


def _stub_fetch(monkeypatch, games: dict[str, dict], raises: dict | None = None) -> list[tuple]:
    """Serve each game's box scores through ``fetch``; return the (endpoint, host, game id) asked.

    ``raises`` maps a game id to the error its first request raises.
    """
    raises = raises or {}
    asked = []

    def fake_fetch(endpoint_cls, *, host, game_id):
        asked.append((endpoint_cls.__name__, host, game_id))
        if game_id in raises:
            raise raises[game_id]
        reply = _replies(games[game_id])[endpoint_cls.__name__]
        return SimpleNamespace(get_dict=lambda: reply)

    monkeypatch.setattr(ga, "fetch", fake_fetch)
    return asked


def _stub_right_rail(monkeypatch, games: dict[str, dict], refused: tuple[str, ...] = ()) -> list:
    """Serve each game's right rail; a ``refused`` game gets a 503. Return the URLs asked."""
    asked = []

    def fake_get(url, timeout):
        asked.append(url)
        game_id = url.split("/")[-2]
        response = requests.Response()
        response.status_code = 503 if game_id in refused else 200
        response._content = json.dumps({"gameInfo": games[game_id]}).encode()
        return response

    monkeypatch.setattr(ga.requests, "get", fake_get)
    return asked


def _games_asked(asked: list[tuple]) -> list[str]:
    return [game_id for endpoint, _, game_id in asked if endpoint == "BoxScoreSummaryV3"]


def _frame(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(_DTYPES)).astype(_DTYPES)


def test_nba_rows_are_the_inactive_list_and_the_box_score_players_with_a_comment(
    monkeypatch, tmp_path
):
    """A player who played (an empty comment) drops; the away team comes first."""
    _gamelog(monkeypatch, tmp_path, "NBA", _WEEK[:1])
    asked = _stub_fetch(monkeypatch, {"0022400184": _IND_AT_CHA})
    path = tmp_path / "absences.parquet"

    assert ga.update_season("NBA", 2024, path) == 5

    assert asked == [
        ("BoxScoreSummaryV3", NBA_HOST, "0022400184"),
        ("BoxScoreTraditionalV3", NBA_HOST, "0022400184"),
    ]
    day = "2024-11-08"
    expected = _frame(
        [
            ("0022400184", day, "IND", 204456, "T.J. McConnell", "Inactive"),
            ("0022400184", day, "IND", 1629614, "Andrew Nembhard", "DNP - Injury/Illness"),
            ("0022400184", day, "CHA", 1628970, "Miles Bridges", "Inactive"),
            ("0022400184", day, "CHA", 1630208, "Nick Richards", "Inactive"),
            ("0022400184", day, "CHA", 203995, "Vasilije Micić", "DNP - Coach's Decision"),
        ]
    )
    pd.testing.assert_frame_equal(pd.read_parquet(path), expected)


def test_wnba_reads_its_own_host_and_keeps_the_box_scores_dnd_and_nwt_comments(
    monkeypatch, tmp_path
):
    """Its inactive lists come back empty; its gamelog dates carry no time."""
    sea_at_phx = {
        "awayTeam": (
            "SEA",
            [],
            [
                (202252, "Alysha", "Clark", ""),
                (1629478, "Katie Lou", "Samuelson", "DND - Injury/Illness"),
            ],
        ),
        "homeTeam": (
            "PHX",
            [],
            [
                (1627674, "Kahleah", "Copper", "DND - Injury/Illness"),
                (1630149, "Satou", "Sabally", ""),
                (1629490, "Kitija", "Laksa", "NWT - Not With Team"),
            ],
        ),
    }
    _gamelog(monkeypatch, tmp_path, "WNBA", [("1022500006", "2025-05-17", 202252)])
    asked = _stub_fetch(monkeypatch, {"1022500006": sea_at_phx})
    path = tmp_path / "absences.parquet"

    assert ga.update_season("WNBA", 2025, path) == 3

    assert [host for _, host, _ in asked] == [WNBA_HOST, WNBA_HOST]
    day = "2025-05-17"
    expected = _frame(
        [
            ("1022500006", day, "SEA", 1629478, "Katie Lou Samuelson", "DND - Injury/Illness"),
            ("1022500006", day, "PHX", 1627674, "Kahleah Copper", "DND - Injury/Illness"),
            ("1022500006", day, "PHX", 1629490, "Kitija Laksa", "NWT - Not With Team"),
        ]
    )
    pd.testing.assert_frame_equal(pd.read_parquet(path), expected)


@pytest.mark.parametrize("home_first", [True, False], ids=["home row first", "away row first"])
def test_nhl_scratches_take_away_and_home_from_the_gamelog(monkeypatch, tmp_path, home_first):
    """Either team's gamelog row names both, in our spelling; a team without a scratches key adds
    no row. A game id's first four digits are its season: 2022020244 is not 2025's."""
    rows = [_BOS, _FLA, _TB, _WPG] if home_first else [_FLA, _BOS, _WPG, _TB]
    _gamelog(monkeypatch, tmp_path, "NHL", [_OTT_2022, *rows])
    asked = _stub_right_rail(monkeypatch, {"2025020103": _FLA_AT_BOS, "2025020120": _WPG_AT_TB})
    path = tmp_path / "absences.parquet"

    assert ga.update_season("NHL", 2025, path) == 4

    assert asked == [_RIGHT_RAIL.format(game_id) for game_id in ("2025020103", "2025020120")]
    expected = _frame(
        [
            ("2025020103", "2025-10-21", "FLA", 8481600, "Tobias Bjornfot", "Scratch"),
            ("2025020103", "2025-10-21", "FLA", 8481655, "Cole Schwindt", "Scratch"),
            ("2025020103", "2025-10-21", "BOS", 8476854, "Hampus Lindholm", "Scratch"),
            ("2025020120", "2025-10-23", "WPG", 8482459, "Tyrel Bauer", "Scratch"),
        ]
    )
    pd.testing.assert_frame_equal(pd.read_parquet(path), expected)


def test_a_season_is_its_own_games_each_asked_once_by_date(monkeypatch, tmp_path):
    """Digits 4-5 of a game id are its season's start year: the 2024 Finals (0042300401) are
    2023's, and 2024 holds its regular season, NBA Cup final, play-in and playoffs. A season the
    gamelog lacks writes no file."""
    rows = [
        ("0022500141", "2025-11-08T00:00:00", 1627783),
        ("0042400101", "2025-04-19T00:00:00", 1627783),
        ("0052400111", "2025-04-15T00:00:00", 1627783),
        ("0062400001", "2024-12-17T00:00:00", 1627783),
        ("0022400184", "2024-11-08T00:00:00", 1627783),
        ("0022400184", "2024-11-08T00:00:00", 1630182),
        ("0042300401", "2024-06-06T00:00:00", 1628369),
        ("0042300401", "2024-06-06T00:00:00", 1629029),
    ]
    _gamelog(monkeypatch, tmp_path, "NBA", rows)
    asked = _stub_fetch(monkeypatch, {game_id: _IND_AT_CHA for game_id, *_ in rows})

    assert ga.update_season("NBA", 2024, tmp_path / "2024.parquet") == 4 * 5
    assert _games_asked(asked) == ["0022400184", "0062400001", "0052400111", "0042400101"]
    asked.clear()
    assert ga.update_season("NBA", 2023, tmp_path / "2023.parquet") == 5
    assert _games_asked(asked) == ["0042300401"]
    asked.clear()
    assert ga.update_season("NBA", 2022, tmp_path / "2022.parquet") == 0
    assert asked == []
    assert not (tmp_path / "2022.parquet").exists()


def test_a_rerun_asks_only_for_games_the_file_lacks_and_keeps_its_rows(monkeypatch, tmp_path):
    """A game where nobody sat left no row, so the rerun asks for it again."""
    asked = _stub_fetch(
        monkeypatch,
        {"0022400184": _IND_AT_CHA, "0022400190": _NOBODY_SAT, "0022400201": _IND_AT_CHA},
    )
    path = tmp_path / "absences.parquet"
    _gamelog(monkeypatch, tmp_path, "NBA", _WEEK[:2])
    assert ga.update_season("NBA", 2024, path) == 5
    stored = pd.read_parquet(path)
    _gamelog(monkeypatch, tmp_path, "NBA", _WEEK)
    asked.clear()

    assert ga.update_season("NBA", 2024, path) == 10

    assert _games_asked(asked) == ["0022400190", "0022400201"]
    rows = pd.read_parquet(path)
    pd.testing.assert_frame_equal(rows.iloc[:5], stored)
    assert set(rows["game_id"].iloc[5:]) == {"0022400201"}


def test_a_game_whose_fetch_gives_up_is_logged_and_left_for_the_next_run(
    monkeypatch, tmp_path, caplog
):
    games = [game_id for game_id, *_ in _WEEK]
    gave_up = NBAStatsError("BoxScoreSummaryV3 failed after 5 attempts")
    _gamelog(monkeypatch, tmp_path, "NBA", _WEEK)
    _stub_fetch(monkeypatch, dict.fromkeys(games, _IND_AT_CHA), raises={"0022400190": gave_up})
    path = tmp_path / "absences.parquet"

    with caplog.at_level(logging.WARNING, logger="log"):
        assert ga.update_season("NBA", 2024, path) == 10

    assert pd.read_parquet(path)["game_id"].unique().tolist() == ["0022400184", "0022400201"]
    assert f"NBA game 0022400190 absences not fetched: {gave_up}" in caplog.text
    assert "NBA 2024: 1 of 3 games not fetched" in caplog.text

    asked = _stub_fetch(monkeypatch, dict.fromkeys(games, _IND_AT_CHA))
    assert ga.update_season("NBA", 2024, path) == 15
    assert _games_asked(asked) == ["0022400190"]


def test_a_game_the_right_rail_refuses_is_left_for_the_next_run(monkeypatch, tmp_path):
    """The rerun asks for the refused game alone: stored NHL ids match the gamelog's as strings."""
    _gamelog(monkeypatch, tmp_path, "NHL", [_FLA, _WPG])
    games = {"2025020103": _FLA_AT_BOS, "2025020120": _WPG_AT_TB}
    _stub_right_rail(monkeypatch, games, refused=("2025020103",))
    path = tmp_path / "absences.parquet"

    assert ga.update_season("NHL", 2025, path) == 1
    assert pd.read_parquet(path)["game_id"].tolist() == ["2025020120"]

    asked = _stub_right_rail(monkeypatch, games)
    assert ga.update_season("NHL", 2025, path) == 4
    assert asked == [_RIGHT_RAIL.format("2025020103")]


@pytest.mark.parametrize("error", [KeyboardInterrupt, KeyError])
def test_an_interrupted_run_stops_with_what_it_fetched_written(monkeypatch, tmp_path, error):
    """Only a failed request is skipped: anything else stops the run, after the write."""
    games = [game_id for game_id, *_ in _WEEK]
    _gamelog(monkeypatch, tmp_path, "NBA", _WEEK)
    asked = _stub_fetch(
        monkeypatch, dict.fromkeys(games, _IND_AT_CHA), raises={"0022400190": error()}
    )
    path = tmp_path / "absences.parquet"

    with pytest.raises(error):
        ga.update_season("NBA", 2024, path)

    assert _games_asked(asked) == ["0022400184", "0022400190"]
    assert pd.read_parquet(path)["game_id"].unique().tolist() == ["0022400184"]
