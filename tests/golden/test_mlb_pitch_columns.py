"""Pitch counts from Savant's game feed (``stats/mlb_pitch.py``), live and for history.

The feed excerpt is the parse_game fixture's: every pitch game 716896 (CHC @ DET, 2023-08-22)
threw to its four batters, as the feed sent it, trimmed to the fields the counts read. The
expected counts were tallied by hand from those records.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sportstradamus.stats import mlb_pitch
from sportstradamus.stats.mlb_pitch import PITCH_COLUMNS, fill_pitch_columns, pitch_columns

GAME = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "mlb_parse_game_716896.json").read_text()
)
SMYLY, HAPP, CARPENTER = 592767, 664023, 681481

# Smyly's 18 pitches to Carpenter (three times through) and McKinstry.
SMYLY_COUNTS = {
    "pitcher times through order": 3,
    "pitcher fastballs": 11,
    "pitcher fastball mph total": 1007.4,
    "pitcher swings": 10,
    "pitcher whiffs": 2,
    "pitcher chases": 5,
    "pitcher zone pitches": 7,
    "pitcher out-of-zone pitches": 11,
    "pitcher zone swings": 5,
    "pitcher zone contacts": 4,
}
# Happ saw 26 pitches from three pitchers and put two in play: 86.7 mph at 29 degrees, 75.9
# at -13. Carpenter saw 21 and was walked intentionally once, which is no pitch.
HAPP_COUNTS = {
    "batter pitches": 26,
    "batter swings": 11,
    "batter whiffs": 3,
    "batter chases": 8,
    "batter out-of-zone pitches": 18,
    "batter batted balls": 2,
    "batter exit velocity total": 162.6,
    "batter hard-hit balls": 0,
    "batter barrels": 0,
    "batter sweet-spot balls": 1,
    "batter ground balls": 1,
}
CARPENTER_COUNTS = {
    "batter pitches": 21,
    "batter swings": 10,
    "batter whiffs": 2,
    "batter chases": 7,
    "batter out-of-zone pitches": 16,
    "batter batted balls": 1,
    "batter exit velocity total": 98.7,
    "batter hard-hit balls": 1,
    "batter ground balls": 1,
}


def _expected(counts: dict) -> pd.Series:
    return pd.Series(counts, dtype=float).reindex(PITCH_COLUMNS, fill_value=0.0)


def test_counts_each_player_from_the_pitches_he_threw_or_saw():
    ids = pd.Series([SMYLY, HAPP, CARPENTER, 1])
    got = pitch_columns(GAME, ids)

    assert list(got.columns) == PITCH_COLUMNS
    assert list(got.index) == ids.tolist()
    for player, counts in (
        (SMYLY, SMYLY_COUNTS),
        (HAPP, HAPP_COUNTS),
        (CARPENTER, CARPENTER_COUNTS),
    ):
        pd.testing.assert_series_equal(got.loc[player], _expected(counts), check_names=False)
    assert (got.loc[1] == 0).all(), "a player the feed shows no pitch for must read 0"


def test_a_ball_the_feed_sends_without_launch_angle_left_the_bat_at_zero_degrees():
    game = copy.deepcopy(GAME)
    batted = next(
        r for r in game["team_home"] if r["batter"] == HAPP and r.get("launch_angle") == "29"
    )
    del batted["launch_angle"], batted["is_barrel"]

    got = pitch_columns(game, pd.Series([HAPP])).loc[HAPP]

    assert got["batter sweet-spot balls"] == 0
    assert got["batter ground balls"] == 2


def test_fill_reads_only_the_games_lacking_the_columns(monkeypatch):
    gamelog = pd.DataFrame(
        {
            "gameId": [716896, 716896, 111, 222],
            "gameDate": ["2023-08-22", "2023-08-22", "2023-08-23", "2024-04-01"],
            "playerId": [SMYLY, HAPP, 5, 6],
        }
    )
    reads = []

    def get(url, max_attempts):
        reads.append(url.rsplit("=", 1)[1])
        return GAME if url.endswith("716896") else {}

    monkeypatch.setattr(mlb_pitch.scraper, "get", get)
    monkeypatch.setattr(mlb_pitch, "sleep", lambda _: None)

    filled = fill_pitch_columns(gamelog, seasons={2023})

    assert reads == ["716896", "111"]
    assert filled.loc[0, "pitcher fastballs"] == 11
    assert filled.loc[1, "batter pitches"] == 26
    assert filled.loc[2:, PITCH_COLUMNS].isna().all().all(), (
        "a game the feed failed, or outside the seasons asked for, must stay blank"
    )

    reads.clear()
    again = fill_pitch_columns(filled, seasons={2023})

    assert reads == ["111"], "a filled game was read again"
    np.testing.assert_array_equal(again.loc[:1, PITCH_COLUMNS], filled.loc[:1, PITCH_COLUMNS])
