"""A new NBA or WNBA season opens with no roster and no seed file of its own.

Before the season's first game the league's bios come back empty and Cleaning the
Glass has not published the season's player export. The roster lookup then stands
on last season's rosters, the seed read on the newest earlier file, and the season's
own roster fills team by team as its first updates find players.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import pytest

from sportstradamus.stats import StatsNBA, StatsWNBA, nba

_LAST_SEASON = {"BOS": {"A": {"POS": "G"}}, "NYK": {"B": {"POS": "F"}, "C": {"POS": "C"}}}
_THIS_SEASON = {"LAL": {"D": {"POS": "W"}}}


def _pairs(stats, frame):
    return sorted(
        zip(frame[stats.log_strings["team"]], frame[stats.log_strings["player"]], strict=True)
    )


@pytest.mark.parametrize(
    "players",
    [{"2025-26": _LAST_SEASON}, {"2025-26": _LAST_SEASON, "2026-27": {}}],
    ids=["season absent", "season empty"],
)
def test_nba_roster_stands_on_last_season_until_this_one_has_players(players):
    stats = StatsNBA()
    stats.players = players

    frame = stats._roster_player_df(2026)

    assert _pairs(stats, frame) == [("BOS", "A"), ("NYK", "B"), ("NYK", "C")]
    assert frame["POS"].tolist() == ["G", "F", "C"]


def test_nba_roster_reads_this_season_once_it_has_players():
    stats = StatsNBA()
    stats.players = {"2025-26": _LAST_SEASON, "2026-27": _THIS_SEASON}

    assert _pairs(stats, stats._roster_player_df(2026)) == [("LAL", "D")]


def test_wnba_roster_stands_on_last_season_by_its_year_key():
    stats = StatsWNBA()
    stats.players = {2025: _LAST_SEASON}

    frame = stats._roster_player_df(2026)

    assert _pairs(stats, frame) == [("BOS", "A"), ("NYK", "B"), ("NYK", "C")]


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
