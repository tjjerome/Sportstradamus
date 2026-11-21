"""Two NHL players who share a name keep separate histories.

The feeds tell them apart only by id, and offers, the archive and every feature are keyed by
name. A gamelog row is keyed by game and player id, and the lesser-quoted player of a pair is
logged under his name and id (``_DISTINCT_PLAYER_NAMES``) in new rows, the roster pull and,
through ``load``, the rows already stored.
"""

import pandas as pd

from sportstradamus.stats import StatsNHL, base

CENTER, DEFENSEMAN = 8480012, 8483678
NAME = "Elias Pettersson"
DISTINCT = "Elias Pettersson 8483678"
GAME_A, GAME_B = 2025020001, 2025020002


def test_load_gives_the_lesser_quoted_namesake_his_distinct_name(monkeypatch):
    gamelog = pd.DataFrame(
        {"gameId": [GAME_A, GAME_A], "playerId": [str(CENTER), str(DEFENSEMAN)], "playerName": NAME}
    )
    roster = {CENTER: {"playerName": NAME}, DEFENSEMAN: {"playerName": NAME}}
    monkeypatch.setattr(
        base,
        "read_gamelog",
        lambda league: {"gamelog": gamelog, "teamlog": pd.DataFrame(), "players": {2025: roster}},
    )
    stats = StatsNHL()

    stats.load()

    assert stats.gamelog["playerName"].tolist() == [NAME, DISTINCT]
    assert stats.players[2025] == {
        CENTER: {"playerName": NAME},
        DEFENSEMAN: {"playerName": DISTINCT},
    }


def _row(game, player_id, name):
    return {"gameId": game, "gameDate": "2025-10-08", "playerId": player_id, "playerName": name}


def test_merge_keys_new_rows_by_game_and_player_id(monkeypatch):
    monkeypatch.setattr(StatsNHL, "_enrich_team_markets", lambda self, frame, **columns: None)
    stats = StatsNHL()
    stats.gamelog = pd.DataFrame([_row(GAME_A, str(CENTER), NAME)])
    stats.teamlog = pd.DataFrame(columns=["gameId", "gameDate", "team"])

    stats._merge_new_gamelogs(
        [
            _row(GAME_A, CENTER, "Elias Petterson"),
            _row(GAME_B, CENTER, NAME),
            _row(GAME_B, DEFENSEMAN, NAME),
        ],
        [],
    )

    keys = stats.gamelog[["gameId", "playerId"]].astype(int).to_numpy().tolist()
    assert sorted(keys) == [[GAME_A, CENTER], [GAME_B, CENTER], [GAME_B, DEFENSEMAN]]


class _MoneypuckRow(dict):
    """A moneypuck skater row whose every stat reads 1."""

    def __missing__(self, key):
        return 1.0


def test_skater_row_names_a_namesake_by_id_and_reads_his_own_power_play():
    game_df = pd.DataFrame(
        {
            "playerName": ["Jeremy Swayman", "VAN"],
            "position": ["G", "Team Level"],
            "team": ["BOS", "VAN"],
            "OffIce_F_iceTime": [3600.0, 3600.0],
            "OffIce_F_shotsOnGoal": [30.0, 30.0],
        }
    )
    pp_df = pd.DataFrame(
        {"playerId": [CENTER, DEFENSEMAN], "playerName": [NAME, NAME], "I_F_points": [2.0, 0.0]}
    )

    rows = [
        StatsNHL()._skater_row(
            _MoneypuckRow(playerId=player_id, playerName=NAME, position="C"),
            GAME_A,
            "2025-10-07",
            "VAN",
            "BOS",
            True,
            True,
            game_df,
            pp_df,
            {},
        )
        for player_id in (CENTER, DEFENSEMAN)
    ]

    assert [(row["playerName"], row["powerPlayPoints"]) for row in rows] == [
        (NAME, 2.0),
        (DISTINCT, 0.0),
    ]
