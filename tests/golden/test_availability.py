"""Pin ``stats.availability``: the NBA absence readers and the features built from them.

Every fixture sets a trap a naive build falls into: gamelog rows out of date order, New Orleans
spelled NOP on the official lists, a game-time DNP, a player with too few games, a traded player
whose last gamelog team is not the list's, an older ESPN capture with a later clock, and an absent
player with a gamelog row on the game date itself.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from sportstradamus.collectors import game_absences
from sportstradamus.collectors.game_absences import ABSENCE_DTYPES
from sportstradamus.collectors.injury_capture import INJURY_DTYPES
from sportstradamus.stats import StatsNBA, nba
from sportstradamus.stats.availability import (
    AVAILABILITY_COLUMNS,
    availability_features,
    espn_absences,
    official_absences,
    player_weights,
)

# The sources' spellings of the gamelog's team codes; production's map lives in stats/nba.py.
TEAM_CODES = {"UTAH": "UTA", "NOP": "NO", "GS": "GSW", "NY": "NYK", "SA": "SAS", "WSH": "WAS"}
GAME_DATE = pd.Timestamp("2026-01-13")

# The absent players' games, (PLAYER_ID, GAME_DATE, MIN, PTS, AST, REB, FGA), out of date order.
ABSENT_PLAYER_GAMES = [
    (1, "2026-01-13", 48, 40, 10, 12, 30),  # on the game date itself: its weight must skip it
    (3, "2026-01-12", 20, 10, 2, 6, 8),
    (1, "2026-01-11", 30, 20, 5, 4, 15),
    (5, "2026-01-12", 10, 4, 1, 3, 3),
    (4, "2026-01-07", 25, 12, 3, 3, 10),
    (1, "2026-01-12", 32, 22, 6, 5, 16),
    (2, "2026-01-12", 36, 25, 7, 6, 18),
    (3, "2026-01-08", 20, 10, 2, 6, 8),
    (5, "2026-01-10", 10, 4, 1, 3, 3),
    (4, "2026-01-11", 25, 12, 3, 3, 10),
    (1, "2026-01-10", 28, 18, 4, 3, 14),
    (2, "2026-01-11", 36, 25, 7, 6, 18),
    (3, "2026-01-10", 20, 10, 2, 6, 8),
    (5, "2026-01-11", 10, 4, 1, 3, 3),
    (4, "2026-01-09", 25, 12, 3, 3, 10),
]
# (game_date, team, player_id, status). NO hosts LAL on 2026-01-13; NO plays daily from 01-01,
# LAL every other day from 12-30, BOS on 01-08, 01-10 and 01-12.
OFFICIAL_LISTS = [
    ("2026-01-13", "NOP", 1, "Inactive - Injury/Illness"),
    ("2026-01-13", "NOP", 2, "Inactive - G League - Two-Way"),
    ("2026-01-13", "NOP", 5, "DND - Injury/Illness"),
    # Player 3 played his games for BOS: the list's team counts, not his last gamelog team.
    ("2026-01-13", "LAL", 3, "Inactive"),
    ("2026-01-13", "LAL", 4, "DNP - Coach's Decision"),
    ("2026-01-02", "NOP", 10, "Inactive"),
    ("2026-01-03", "NOP", 10, "Inactive"),
    ("2026-01-07", "NOP", 10, "DNP - Coach's Decision"),
    ("2026-01-12", "NOP", 10, "Inactive"),
    ("2026-01-08", "BOS", 50, "Inactive"),
    ("2026-01-11", "NOP", 50, "NWT - Personal"),
    ("2025-12-30", "LAL", 20, "Inactive"),
    ("2026-01-11", "LAL", 20, "Inactive"),
]


def _write_lists(base, season, rows):
    """Store one season's official lists of ``(game_date, team, player_id, status)`` rows."""
    lists = pd.DataFrame(
        [
            {
                "game_id": f"{date}-{team}",
                "game_date": date,
                "team": team,
                "player_id": player_id,
                "player_name": f"Player {player_id}",
                "status": status,
            }
            for date, team, player_id, status in rows
        ]
    ).astype(ABSENCE_DTYPES)
    path = base / "NBA" / "injuries" / f"{season}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    lists.to_parquet(path, index=False)


def _write_capture(capture_dir, day, clock, rows):
    """Store one ESPN capture of ``(team, name, status)`` rows."""
    report = pd.DataFrame(
        [{"team": team, "name": name, "status": status} for team, name, status in rows],
        columns=list(INJURY_DTYPES),
    ).astype(INJURY_DTYPES)
    path = capture_dir / "NBA" / f"date={day}" / f"{clock}.parquet"
    path.parent.mkdir(parents=True)
    report.to_parquet(path, index=False)


def test_player_weights_roll_each_players_last_15_games_in_date_order():
    days = pd.date_range("2026-01-01", periods=16)
    # Player 7 plays 16 days running, his MIN the day's number; player 8 plays twice.
    games = [(7, day, n) for n, day in enumerate(days, start=1)]
    games += [(8, days[0], 30), (8, days[1], 30)]
    gamelog = pd.DataFrame(
        [
            {
                "PLAYER_ID": player,
                "GAME_DATE": day.isoformat(),
                "MIN": float(minutes),
                "PTS": 2 * minutes,
                "AST": 1,
                "REB": 5,
                "FGA": 10,
            }
            for player, day, minutes in reversed(games)
        ]
    )

    weights = player_weights(gamelog)

    # Player 8 never reaches 3 games. Player 7's game n >= 3 means his games 1..n, (n + 1) / 2,
    # until game 16, whose 15-game window drops game 1: (2 + ... + 16) / 15 = 9.
    expected_min = [(n + 1) / 2 for n in range(3, 16)] + [9.0]
    assert list(weights.columns) == ["PLAYER_ID", "GAME_DATE", "MIN", "PTS", "AST", "REB", "FGA"]
    assert weights["PLAYER_ID"].tolist() == [7] * 14
    assert weights["GAME_DATE"].tolist() == list(days[2:])
    assert weights["MIN"].tolist() == expected_min
    assert weights["PTS"].tolist() == [2 * minutes for minutes in expected_min]
    assert weights[["AST", "REB", "FGA"]].drop_duplicates().to_numpy().tolist() == [[1, 5, 10]]


def test_official_absences_keep_pregame_statuses_in_gamelog_codes(tmp_path):
    _write_lists(tmp_path, 2024, [("2025-01-02", "LAL", 3, "Inactive")])
    _write_lists(
        tmp_path,
        2025,
        [
            ("2026-01-13", "NOP", 1, "Inactive"),
            ("2026-01-13", "NOP", 2, "DNP - Coach's Decision"),
            ("2026-01-13", "NOP", 5, "DNP - Rest"),
            ("2026-01-13", "LAL", 4, "DNP - Injury/Illness"),
            ("2026-01-13", "LAL", 6, "DND - Coach's Decision"),
            ("2026-01-13", "LAL", 7, "DND - Injury/Illness"),
            ("2026-01-13", "UTA", 8, "NWT - Personal"),
            ("2026-01-13", "UTA", 9, "Inactive - G League - Two-Way"),
        ],
    )

    absences = official_absences("NBA", TEAM_CODES, base=tmp_path)

    # Both seasons read; the four statuses decided in game drop out; NOP is the gamelog's NO.
    expected = pd.DataFrame(
        {
            "team": ["LAL", "NO", "LAL", "UTA", "UTA"],
            "date": pd.to_datetime(["2025-01-02"] + ["2026-01-13"] * 4),
            "player_id": [3, 1, 7, 8, 9],
        }
    )
    pd.testing.assert_frame_equal(absences, expected)


def test_espn_absences_read_only_the_newest_capture(tmp_path):
    # The older capture has the later clock; only its date partition marks it older.
    _write_capture(tmp_path, "2026-11-13", "235006", [("BOS", "Jayson Tatum", "Out")])
    _write_capture(
        tmp_path,
        "2026-11-14",
        "005006",
        [
            ("GS", "Draymond Green", "Out"),
            ("UTAH", "Jusuf Nurkić", "Suspension"),
            ("NO", "Trey Murphy III", "Out"),
            ("WSH", "Bilal Coulibaly", "Day-To-Day"),
            ("SA", "Rookie Nobody", "Out"),
        ],
    )
    player_ids = {
        "Jayson Tatum": 1628369,
        "Draymond Green": 203110,
        "Jusuf Nurkic": 203994,
        "Trey Murphy": 1630530,
        "Bilal Coulibaly": 1641731,
    }

    absences = espn_absences("NBA", TEAM_CODES, player_ids, capture_dir=tmp_path)

    # Names lose accents and suffixes; Day-To-Day may play; the rookie has no gamelog id.
    expected = pd.DataFrame({"team": ["GSW", "UTA", "NO"], "player_id": [203110, 203994, 1630530]})
    pd.testing.assert_frame_equal(absences, expected)


def test_absence_readers_with_nothing_stored_return_typed_empty_frames(tmp_path):
    official = official_absences("NBA", TEAM_CODES, base=tmp_path)
    espn = espn_absences("NBA", TEAM_CODES, {}, capture_dir=tmp_path)

    assert official.empty
    assert espn.empty
    assert official.dtypes.to_dict() == {
        "team": "object",
        "date": "datetime64[ns]",
        "player_id": "int64",
    }
    assert espn.dtypes.to_dict() == {"team": "object", "player_id": "int64"}


@pytest.mark.parametrize("upcoming", [False, True], ids=["training", "serving"])
def test_availability_features_by_hand(tmp_path, upcoming):
    _write_lists(tmp_path, 2025, OFFICIAL_LISTS)
    gamelog = pd.DataFrame(
        ABSENT_PLAYER_GAMES, columns=["PLAYER_ID", "GAME_DATE", "MIN", "PTS", "AST", "REB", "FGA"]
    )
    schedule = [("NO", day) for day in pd.date_range("2026-01-01", "2026-01-12")]
    schedule += [("LAL", day) for day in pd.date_range("2025-12-30", "2026-01-11", freq="2D")]
    schedule += [("BOS", day) for day in pd.to_datetime(["2026-01-08", "2026-01-10", "2026-01-12"])]
    # At serving the game is not in the gamelog yet: the rows alone place it.
    if not upcoming:
        schedule += [(team, GAME_DATE) for team in ("NO", "LAL", "BOS", "MIA")]
    rows = pd.DataFrame(
        {
            "team": ["NO", "NO", "LAL", "LAL", "BOS"],
            "opponent": ["LAL", "LAL", "NO", "NO", "MIA"],
            "date": GAME_DATE,
            "player_id": [10.0, 50.0, 20.0, float("nan"), 30.0],
        },
        index=["Ten", "Fifty", "Twenty", "Rookie", "Thirty"],
    )

    features = availability_features(
        rows,
        official_absences("NBA", TEAM_CODES, base=tmp_path),
        player_weights(gamelog),
        pd.DataFrame(schedule, columns=["team", "date"]),
    )

    # Weights are means over games strictly before 01-13 (player 1's 01-13 row is left out):
    #   player 1: MIN (28 + 30 + 32) / 3 = 30, PTS (18 + 20 + 22) / 3 = 20, AST 5, REB 4, FGA 15
    #   player 5: MIN 10, PTS 4, AST 1, REB 3, FGA 3
    #   player 2: two games, so 0 throughout
    #   player 3: MIN 20, PTS 10, AST 2, REB 6, FGA 8, from his BOS games
    # NO out (players 1, 2, 5): MIN 30 + 0 + 10 = 40, PTS 20 + 4 = 24, AST 5 + 1 = 6,
    #   REB 4 + 3 = 7, FGA 15 + 3 = 18, top MIN 30.
    # LAL out: player 3 alone, since player 4's DNP was decided in game: 20, 10, 2, 6, 8, top 20.
    no_out = [40.0, 24.0, 6.0, 7.0, 18.0, 30.0]
    lal_out = [20.0, 10.0, 2.0, 6.0, 8.0, 20.0]
    # Own: 01-13 is NO's 13th game, so NO's previous 10 run 01-03 to 01-12.
    #   Ten: 01-03 and 01-12 count; 01-02 is 11 games back, 01-07 a DNP. 2 missed, back.
    #   Fifty: 01-11 counts; his 01-08 absence was BOS's. 1 missed, played 01-12.
    #   Twenty: LAL's previous 7 games include 12-30 (14 days back) and 01-11 (the previous
    #     game, 2 days back). 2 missed, back.
    expected = pd.DataFrame(
        [
            [*no_out, 20.0, 10.0, 20.0, 2.0, 1.0],
            [*no_out, 20.0, 10.0, 20.0, 1.0, 0.0],
            [*lal_out, 40.0, 24.0, 30.0, 2.0, 1.0],
            [*lal_out, 40.0, 24.0, 30.0, 0.0, 0.0],
            [0.0] * len(AVAILABILITY_COLUMNS),
        ],
        index=rows.index,
        columns=list(AVAILABILITY_COLUMNS),
    )
    pd.testing.assert_frame_equal(features, expected)


def test_a_game_seen_only_as_the_opponent_still_orders_its_teams_games():
    # Serving two dates: NO has rows only on 01-14, but LAL's 01-13 row says NO plays then too.
    team_games = pd.DataFrame({"team": ["NO", "LAL"], "date": pd.to_datetime(["2026-01-10"] * 2)})
    rows = pd.DataFrame(
        {
            "team": ["LAL", "NO"],
            "opponent": ["NO", "BOS"],
            "date": pd.to_datetime(["2026-01-13", "2026-01-14"]),
            "player_id": [20.0, 10.0],
        }
    )
    absences = pd.DataFrame(
        {"team": ["NO"], "date": pd.to_datetime(["2026-01-13"]), "player_id": [10]}
    )
    weights = player_weights(
        pd.DataFrame(
            ABSENT_PLAYER_GAMES,
            columns=["PLAYER_ID", "GAME_DATE", "MIN", "PTS", "AST", "REB", "FGA"],
        )
    )

    features = availability_features(rows, absences, weights, team_games)

    # NO's 01-13 game is its previous one on 01-14, and player 10 sat it.
    assert features.loc[1, ["avail own missed10", "avail own back"]].tolist() == [1.0, 1.0]


def test_a_failed_list_refresh_is_logged_and_never_raised(monkeypatch, caplog):
    """NBA's update ends with the refresh, after its gamelog is written: a changed payload must
    leave the lists a run behind, not take NBA off the board."""
    calls = []

    def broken_update_season(league, season, path, budget_s):
        calls.append((league, season, path.name, budget_s))
        raise KeyError("boxScoreSummary")

    monkeypatch.setattr(game_absences, "update_season", broken_update_season)
    stats = StatsNBA()
    stats.season = "2026-27"

    with caplog.at_level(logging.ERROR, logger="log"):
        stats._refresh_absence_lists()

    assert calls == [("NBA", 2026, "2026.parquet", nba._ABSENCE_REFRESH_BUDGET_S)]
    assert "NBA absence lists not refreshed this run" in caplog.text
