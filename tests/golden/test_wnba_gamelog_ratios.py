"""A WNBA ratio reads 0 on a zero denominator, as the NBA's do.

A player who took no shot from the field or had none blocked, and a team with no shot
blocked, divided by zero: the stored gamelog held 4,948 infinite cells and the team log 156.
"""

from __future__ import annotations

import pandas as pd

from sportstradamus.stats import StatsWNBA

_GAME_ID = "1022600001"
# The other box-score columns the player assembly reads; their values do not matter here.
_OTHER_STATS = dict.fromkeys(["PTS", "REB", "OREB", "DREB", "AST", "STL", "TOV"], 0)


def _player_row(player_id, **shooting):
    return {
        "PLAYER_ID": player_id,
        "GAME_ID": _GAME_ID,
        "GAME_DATE": "2026-06-01T00:00:00",
        "PLAYER_NAME": f"Player {player_id}",
        "TEAM_ABBREVIATION": "LVA",
        "MATCHUP": "LVA vs. NYL",
        "MIN": 20.0,
        **_OTHER_STATS,
        **shooting,
    }


def _team_row(team_id, matchup, **shooting):
    return {
        "TEAM_ID": team_id,
        "GAME_ID": _GAME_ID,
        "GAME_DATE": "2026-06-01T00:00:00",
        "TEAM_ABBREVIATION": matchup[:3],
        "MATCHUP": matchup,
        **shooting,
    }


def test_player_ratios_read_zero_on_a_zero_denominator():
    stats = StatsWNBA()
    shooter = _player_row(1, FTM=5, FGA=10, FG3A=4, BLK=2, BLKA=1)
    no_attempt = _player_row(2, FTM=2, FGA=0, FG3A=0, BLK=1, BLKA=0)
    # The advanced and usage logs, cut to their join keys: no ratio reads a column of theirs.
    keys = [{"PLAYER_ID": 1, "GAME_ID": _GAME_ID}, {"PLAYER_ID": 2, "GAME_ID": _GAME_ID}]
    positions = pd.DataFrame({"PLAYER_ID": [1, 2], "POS": ["G", "F"]})

    rows = stats._assemble_wnba_player_rows([shooter, no_attempt], keys, keys, positions)

    assert rows.set_index("PLAYER_ID")[["FTR", "FG3_RATIO", "BLK_PCT"]].to_dict("index") == {
        1: {"FTR": 0.5, "FG3_RATIO": 0.4, "BLK_PCT": 2.0},
        2: {"FTR": 0.0, "FG3_RATIO": 0.0, "BLK_PCT": 0.0},
    }


def test_team_ratios_read_zero_on_a_zero_denominator():
    stats = StatsWNBA()
    # A team's row carries its opponent's figure for each OPP_ column the stored teamlog names.
    stats.teamlog = pd.DataFrame(columns=["OPP_FTR", "OPP_BLK_RATIO"])
    home = _team_row(1, "LVA vs. NYL", FTM=10, FGA=50, BLK=4, BLKA=2)
    away = _team_row(2, "NYL @ LVA", FTM=8, FGA=0, BLK=3, BLKA=0)
    # The scoring and advanced logs, cut to their join keys.
    keys = [{"TEAM_ID": 1, "GAME_ID": _GAME_ID}, {"TEAM_ID": 2, "GAME_ID": _GAME_ID}]

    rows = stats._assemble_wnba_team_rows([home, away], keys, keys)

    ratios = ["FTR", "BLK_RATIO", "OPP_FTR", "OPP_BLK_RATIO"]
    assert rows.set_index("TEAM_ID")[ratios].to_dict("index") == {
        1: {"FTR": 0.2, "BLK_RATIO": 2.0, "OPP_FTR": 0.0, "OPP_BLK_RATIO": 0.0},
        2: {"FTR": 0.0, "BLK_RATIO": 0.0, "OPP_FTR": 0.2, "OPP_BLK_RATIO": 2.0},
    }
