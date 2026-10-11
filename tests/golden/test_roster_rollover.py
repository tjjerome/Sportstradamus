"""A new season opens with no roster and, in the NBA, no seed file of its own.

Before the season's first game the league's bios come back empty and Cleaning the
Glass has not published the season's player export. A season's roster can then lack
a player until that player's first game, so the roster lookup lets last season's row
stand in for every player the season does not hold yet, in the NBA, the WNBA and the
NHL. The seed read falls on the newest earlier file, and the season's own roster
fills team by team as its first updates find players.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import pytest

from sportstradamus.stats import StatsNBA, StatsNHL, StatsWNBA, nba

_LAST_SEASON = {"BOS": {"A": {"POS": "G"}}, "NYK": {"B": {"POS": "F"}, "C": {"POS": "C"}}}
_THIS_SEASON = {"LAL": {"D": {"POS": "W"}}}
_NHL_LAST_SEASON = {
    8470001: {"playerName": "A", "team": "BOS", "position": "C"},
    8470002: {"playerName": "B", "team": "NYR", "position": "G"},
}


def _pairs(stats, frame):
    """The frame's (team, player) pairs in row order."""
    return list(
        zip(frame[stats.log_strings["team"]], frame[stats.log_strings["player"]], strict=True)
    )


@pytest.mark.parametrize(
    "players",
    [{"2025-26": _LAST_SEASON}, {"2025-26": _LAST_SEASON, "2026-27": {}}],
    ids=["season absent", "season empty"],
)
def test_nba_roster_is_last_seasons_while_this_season_holds_nobody(players):
    stats = StatsNBA()
    stats.players = players

    frame = stats._roster_player_df(2026)

    assert _pairs(stats, frame) == [("BOS", "A"), ("NYK", "B"), ("NYK", "C")]
    assert frame["POS"].tolist() == ["G", "F", "C"]


def test_nba_last_seasons_row_stands_in_for_a_player_this_season_has_not_seen():
    stats = StatsNBA()
    stats.players = {"2025-26": _LAST_SEASON, "2026-27": _THIS_SEASON}

    frame = stats._roster_player_df(2026)

    assert _pairs(stats, frame) == [("LAL", "D"), ("BOS", "A"), ("NYK", "B"), ("NYK", "C")]


def test_a_player_this_season_holds_reads_on_this_seasons_team():
    stats = StatsNBA()
    stats.players = {
        "2025-26": {"BOS": {"Moved": {"POS": "G"}, "B": {"POS": "F"}}},
        "2026-27": {"LAL": {"Moved": {"POS": "W"}}},
    }

    frame = stats._roster_player_df(2026)

    assert _pairs(stats, frame) == [("LAL", "Moved"), ("BOS", "B")]
    assert frame["POS"].tolist() == ["W", "F"]


def test_rows_of_one_season_that_share_a_name_both_stay():
    stats = StatsNBA()
    stats.players = {
        "2025-26": {
            "DEN": {"Traded": {"POS": "G"}, "Dealt": {"POS": "F"}},
            "MIA": {"Dealt": {"POS": "F"}},
        },
        "2026-27": {"BOS": {"Traded": {"POS": "G"}}, "LAL": {"Traded": {"POS": "G"}}},
    }

    frame = stats._roster_player_df(2026)

    assert _pairs(stats, frame) == [
        ("BOS", "Traded"),
        ("LAL", "Traded"),
        ("DEN", "Dealt"),
        ("MIA", "Dealt"),
    ]


def test_wnba_roster_stands_in_by_its_year_keys():
    stats = StatsWNBA()
    stats.players = {2025: _LAST_SEASON, 2026: _THIS_SEASON}

    frame = stats._roster_player_df(2026)

    assert _pairs(stats, frame) == [("LAL", "D"), ("BOS", "A"), ("NYK", "B"), ("NYK", "C")]


@pytest.mark.parametrize(
    "players",
    [{2025: _NHL_LAST_SEASON}, {2025: _NHL_LAST_SEASON, 2026: {}}],
    ids=["season absent", "season empty"],
)
def test_nhl_roster_is_last_seasons_while_this_season_holds_nobody(players):
    stats = StatsNHL()
    stats.players = players

    frame = stats._roster_player_df(2026)

    assert _pairs(stats, frame) == [("BOS", "A"), ("NYR", "B")]
    assert frame["position"].tolist() == ["C", "G"]


def test_nhl_last_seasons_row_stands_in_per_player_once_the_season_is_stored():
    stats = StatsNHL()
    stats.players = {
        2025: _NHL_LAST_SEASON,
        2026: {8470001: {"playerName": "A", "team": "TOR", "position": "C"}},
    }

    frame = stats._roster_player_df(2026)

    assert _pairs(stats, frame) == [("TOR", "A"), ("NYR", "B")]


@pytest.mark.parametrize(
    ("league", "players"),
    [
        (StatsNBA, {"2024-25": _LAST_SEASON}),
        (StatsWNBA, {2024: _LAST_SEASON}),
        (StatsNHL, {2024: _NHL_LAST_SEASON}),
    ],
    ids=["NBA", "WNBA", "NHL"],
)
def test_a_league_with_neither_season_stored_has_an_empty_roster(league, players):
    stats = league()
    stats.players = players

    assert stats._roster_player_df(2026).empty


def test_roster_seed_is_the_seasons_own_file_when_it_exists(monkeypatch, tmp_path):
    for season in ("2024-25", "2025-26"):
        (tmp_path / f"nba_players_{season}.csv").touch()
    monkeypatch.setattr(nba, "_ROSTER_SEED_DIR", tmp_path)

    assert nba._roster_seed_path("2025-26") == tmp_path / "nba_players_2025-26.csv"


def test_roster_seed_falls_back_to_the_newest_earlier_season(monkeypatch, tmp_path, caplog):
    for season in ("2023-24", "2024-25", "2025-26"):
        (tmp_path / f"nba_players_{season}.csv").touch()
    monkeypatch.setattr(nba, "_ROSTER_SEED_DIR", tmp_path)

    with caplog.at_level(logging.WARNING, logger="log"):
        seed = nba._roster_seed_path("2026-27")

    assert seed == tmp_path / "nba_players_2025-26.csv"
    assert "nba_players_2026-27.csv is missing" in caplog.text


def test_roster_merge_adds_a_team_the_season_lacks_and_keeps_what_is_stored():
    stats = StatsNBA()
    stats.players = {
        "2026-27": {
            "BOS": {"A": {"POS": "G", "AGE": 30}, "Traded": {"POS": "F", "AGE": 27}},
            "LAL": {"D": {"POS": "W", "AGE": 24}},
        }
    }

    stats._merge_season_roster(
        "2026-27",
        {"NYK": {"B": {"POS": "F", "AGE": 22}}, "BOS": {"A": {"POS": "G", "AGE": 31}}},
    )

    assert stats.players["2026-27"] == {
        "BOS": {"A": {"POS": "G", "AGE": 31}, "Traded": {"POS": "F", "AGE": 27}},
        "LAL": {"D": {"POS": "W", "AGE": 24}},
        "NYK": {"B": {"POS": "F", "AGE": 22}},
    }
    assert list(stats.players["2026-27"]) == ["BOS", "LAL", "NYK"]


def test_roster_merge_starts_a_season_the_store_does_not_hold():
    stats = StatsWNBA()
    stats.players = {2025: _LAST_SEASON}

    stats._merge_season_roster(2026, _THIS_SEASON)

    assert stats.players == {2025: _LAST_SEASON, 2026: _THIS_SEASON}


def _nba_before_opening_night(players):
    stats = StatsNBA()
    stats._set_season_start(date(2026, 10, 20))
    stats.players = players
    return stats


def test_empty_bios_leave_the_new_season_empty_and_last_season_alone():
    stats = _nba_before_opening_night({"2025-26": _LAST_SEASON})

    stats._build_player_profiles(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {})

    assert stats.players == {"2025-26": _LAST_SEASON, "2026-27": {}}
    assert stats.players["2026-27"] is not stats.players["2025-26"]


def test_empty_bios_never_replace_a_roster_the_season_already_holds():
    stats = _nba_before_opening_night({"2025-26": _LAST_SEASON, "2026-27": _THIS_SEASON})

    stats._build_player_profiles(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {})

    assert stats.players["2026-27"] == _THIS_SEASON
