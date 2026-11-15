"""Unit tests for the pure assembly helpers extracted from ``StatsNBA.update``.

``update`` itself is a network-bound fetch orchestrator with no automated
coverage; decomposing it for complexity moved the arithmetic into small pure
helpers, which these tests pin with synthetic inputs. The instance is built via
``__new__`` to skip ``__init__``'s file I/O; each test sets the few attributes its
helper reads.
"""

import pandas as pd
import pytest

from sportstradamus.stats import nba

_LOG_STRINGS = {"team": "TEAM_ABBREVIATION", "opponent": "OPP", "position": "POS"}
_POSITIONS = ["P", "C", "F", "W", "B"]


def _stats():
    s = nba.StatsNBA.__new__(nba.StatsNBA)
    s.log_strings = dict(_LOG_STRINGS)
    s.positions = list(_POSITIONS)
    return s


def test_compute_derived_player_stats():
    game = {
        "PTS": 20,
        "REB": 10,
        "AST": 5,
        "BLK": 2,
        "STL": 1,
        "TOV": 3,
        "FTM": 4,
        "FGA": 15,
        "FG3A": 6,
        "FG3M": 2,
        "BLKA": 1,
        "OREB": 3,
        "DREB": 7,
        "MIN": 30,
    }
    nba.StatsNBA._compute_derived_player_stats(game)

    assert game["PRA"] == 35
    assert game["PR"] == 30
    assert game["PA"] == 25
    assert game["RA"] == 15
    assert game["BLST"] == 3
    assert game["fantasy points prizepicks"] == pytest.approx(45.5)
    assert game["fantasy points underdog"] == pytest.approx(45.5)
    assert game["fantasy points parlay"] == pytest.approx(38)
    assert game["FTR"] == pytest.approx(4 / 15)
    assert game["FG3_RATIO"] == pytest.approx(6 / 15)
    assert game["BLK_PCT"] == pytest.approx(2.0)
    assert game["FGA_48"] == pytest.approx(24)
    assert game["REB_48"] == pytest.approx(16)
    assert game["STL_48"] == pytest.approx(1.6)


def test_compute_derived_player_stats_zero_denominators():
    game = {
        "PTS": 0,
        "REB": 0,
        "AST": 0,
        "BLK": 0,
        "STL": 0,
        "TOV": 0,
        "FTM": 0,
        "FGA": 0,
        "FG3A": 0,
        "FG3M": 0,
        "BLKA": 0,
        "OREB": 0,
        "DREB": 0,
        "MIN": 1,
    }
    nba.StatsNBA._compute_derived_player_stats(game)

    assert game["FTR"] == 0
    assert game["FG3_RATIO"] == 0
    assert game["BLK_PCT"] == 0


def test_canonicalize_team_abbrevs():
    df = pd.DataFrame(
        {
            "TEAM_ABBREVIATION": ["UTAH", "NOP", "GS", "NY", "SA", "BOS"],
            "OPP": ["SA", "NY", "GS", "NOP", "UTAH", "BOS"],
        }
    )
    _stats()._canonicalize_team_abbrevs(df)

    assert list(df["TEAM_ABBREVIATION"]) == ["UTA", "NO", "GSW", "NYK", "SAS", "BOS"]
    assert list(df["OPP"]) == ["SAS", "NYK", "GSW", "NO", "UTA", "BOS"]


def test_merge_team_logs():
    teamlog = [
        {"TEAM_ID": 1, "GAME_ID": "G1", "PTS": 100},
        {"TEAM_ID": 2, "GAME_ID": "G1", "PTS": 98},
    ]
    adv_idx = {(1, "G1"): {"OFF_RATING": 110}}
    sco_idx = {(1, "G1"): {"PCT_PTS_2PT": 0.5}}
    nba.StatsNBA._merge_team_logs(teamlog, adv_idx, sco_idx)

    assert teamlog[0] == {
        "TEAM_ID": 1,
        "GAME_ID": "G1",
        "PTS": 100,
        "OFF_RATING": 110,
        "PCT_PTS_2PT": 0.5,
    }
    # Team 2 has no advanced/scoring entry — left untouched.
    assert teamlog[1] == {"TEAM_ID": 2, "GAME_ID": "G1", "PTS": 98}


def test_pair_team_rows():
    stats = _stats()
    stats.teamlog = pd.DataFrame(columns=["TEAM_ABBREVIATION", "OPP", "FTR", "OPP_PTS"])
    teamlog = [
        {"TEAM_ABBREVIATION": "BOS", "FTM": 10, "FGA": 50, "BLK": 4, "BLKA": 2, "PTS": 100},
        {"TEAM_ABBREVIATION": "LAL", "FTM": 8, "FGA": 40, "BLK": 6, "BLKA": 3, "PTS": 98},
    ]
    rows = stats._pair_team_rows(teamlog)

    assert len(rows) == 2
    bos, lal = rows
    assert bos["OPP"] == "LAL"
    assert lal["OPP"] == "BOS"
    assert bos["FTR"] == pytest.approx(10 / 50)
    assert bos["BLK_RATIO"] == pytest.approx(4 / 2)
    # Only OPP_ columns present in self.teamlog.columns are copied across.
    assert bos["OPP_PTS"] == 98
    assert lal["OPP_PTS"] == 100
    assert "OPP_BLK" not in bos


def test_pair_team_rows_zero_denominators():
    stats = _stats()
    stats.teamlog = pd.DataFrame(columns=["TEAM_ABBREVIATION", "OPP"])
    teamlog = [
        {"TEAM_ABBREVIATION": "BOS", "FTM": 0, "FGA": 0, "BLK": 0, "BLKA": 0},
        {"TEAM_ABBREVIATION": "LAL", "FTM": 0, "FGA": 0, "BLK": 0, "BLKA": 0},
    ]
    rows = stats._pair_team_rows(teamlog)

    assert rows[0]["FTR"] == 0
    assert rows[0]["BLK_RATIO"] == 0


def test_resolve_pos_from_history():
    stats = _stats()
    stats.players = {
        "2022-23": {"BOS": {"Jayson Tatum": {"POS": "F"}}},
        "2023-24": {"LAL": {"LeBron James": {"POS": "W"}}},
    }
    assert stats._resolve_pos_from_history("LeBron James") == "W"
    assert stats._resolve_pos_from_history("Jayson Tatum") == "F"
    assert stats._resolve_pos_from_history("Nobody") is None


def test_resolve_pos_from_history_skips_invalid():
    stats = _stats()
    # Newest season has a corrupt (non-string) POS; older season has a valid one.
    stats.players = {
        "2022-23": {"BOS": {"Jayson Tatum": {"POS": "F"}}},
        "2023-24": {"BOS": {"Jayson Tatum": {"POS": 0}}},
    }
    assert stats._resolve_pos_from_history("Jayson Tatum") == "F"


def test_assembled_row_keeps_the_accent_free_name_the_endpoint_rows_carry_raw():
    """Offers, the roster and profiles key by accent-free names; the endpoints send raw ones."""
    stats = _stats()
    stats.season = "2025-26"
    stats.gamelog = pd.DataFrame(columns=["PLAYER_ID", "GAME_ID"])
    stats.players = {"2025-26": {"DEN": {"Nikola Jokic": {"POS": "C"}}}}
    key = (203999, "0022500001")
    raw = {
        "PLAYER_ID": key[0],
        "GAME_ID": key[1],
        "PLAYER_NAME": "Nikola Jokić",
        "TEAM_ABBREVIATION": "DEN",
    }
    box_score = {"PTS": 30, "REB": 12, "AST": 10, "BLK": 1, "STL": 2, "TOV": 3, "FTM": 6}
    shots = {"FGA": 20, "FG3A": 4, "BLKA": 1, "OREB": 3, "DREB": 9}
    game = {**raw, **box_score, **shots, "MIN": 36, "MATCHUP": "DEN vs. LAL"}

    (row,) = stats._assemble_player_rows(
        [game], {key: {**raw, "OFF_RATING": 125.0}}, {key: {**raw, "USG_PCT": 0.31}}, {}
    )

    assert row["PLAYER_NAME"] == "Nikola Jokic"
    assert (row["POS"], row["OPP"], row["OFF_RATING"], row["USG_PCT"]) == ("C", "LAL", 125.0, 0.31)
