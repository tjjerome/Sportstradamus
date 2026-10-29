"""The NHL gamelog carries the per-game sums behind moneypuck's season-summary rates.

``_skater_row`` logs them for every new game and ``backfill_season_sums`` re-parses the
games whose stored rows predate the columns, so a season's comp profile can be rebuilt
from the games already played (I6f: the season summary looked ahead).
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from sportstradamus.stats import StatsNHL, nhl
from sportstradamus.stats.nhl import SEASON_SUM_COLUMNS

_TEAM_MAP = {"SJS": "SJ", "LAK": "LA", "NJD": "NJ", "TBL": "TB", "WSH": "WAS"}
_SUMS = list(SEASON_SUM_COLUMNS)
_GAME_A, _GAME_B = 2025020001, 2025020002


def test_skater_row_logs_every_season_sum_source_as_a_float():
    sources = {src: 100 + i for i, src in enumerate(SEASON_SUM_COLUMNS.values())}
    player = pd.Series(
        {
            "playerId": 8470001,
            "playerName": "Auston Matthews",
            "position": "C",
            "I_F_points": 2,
            "I_F_shotsOnGoal": 5,
            "I_A_blockedShotAttempts": 1,
            "I_F_goals": 1,
            "I_F_primaryAssists": 1,
            "I_F_secondaryAssists": 0,
            "I_F_hits": 2,
            "I_F_faceOffsWon": 9,
            "I_F_iceTime": 1200,
            "OnIce_A_savedShotsOnGoal": 8,
            "OnIce_A_shotsOnGoal": 9,
            "OnIce_A_goals": 1,
            "OnIce_unblockedShotAttempts_For_Percentage": 0.55,
            "OnIce_A_flurryScoreVenueAdjustedxGoals": 0.8,
            "OnIce_A_reboundGoals": 0,
            "OnIce_A_reboundxGoals": 0.1,
        }
        | sources
    )
    game_df = pd.DataFrame(
        {
            "playerName": ["Jeremy Swayman", "TOR"],
            "position": ["G", "Team Level"],
            "team": ["BOS", "TOR"],
            "OffIce_F_iceTime": [3600, 3600],
            "OffIce_F_shotsOnGoal": [30, 30],
        }
    )

    row = StatsNHL()._skater_row(
        player,
        _GAME_A,
        "2025-10-07",
        "TOR",
        "BOS",
        True,
        True,
        game_df,
        pd.DataFrame({"playerName": []}),
        _TEAM_MAP,
    )

    for name, src in SEASON_SUM_COLUMNS.items():
        assert row[name] == float(player[src])
        assert isinstance(row[name], float)
    assert {"shots", "blocked", "Fenwick", "opponent goalie"} <= row.keys()
    assert row["opponent goalie"] == "Jeremy Swayman"


def _row(game, player, shots, sums):
    return {
        "gameId": game,
        "gameDate": f"2025-10-0{game % 10 + 6}",
        "playerName": player,
        "team": "TOR",
        "shots": shots,
    } | dict.fromkeys(_SUMS, sums)


def _gamelog():
    return pd.DataFrame(
        [
            _row(_GAME_B, "B1", 3.0, np.nan),
            _row(_GAME_A, "A1", 1.0, 1.5),
            _row(_GAME_B, "B2", 4.0, np.nan),
            _row(_GAME_A, "A2", 2.0, 2.5),
            _row(_GAME_B, "B3", 5.0, np.nan),
        ]
    )


def _fake_parse_game(calls):
    """Two of each game's players come back, with ``shots`` moved to prove it stays put."""

    def parse_game(self, gameId, gameDate):
        calls.append((gameId, gameDate))
        letter = "A" if gameId == _GAME_A else "B"
        return [_row(gameId, f"{letter}1", 99.0, 7.0), _row(gameId, f"{letter}2", 99.0, 8.0)], []

    return parse_game


def test_backfill_fills_only_the_season_sums_of_the_games_that_lack_them(monkeypatch):
    calls = []
    monkeypatch.setattr(StatsNHL, "parse_game", _fake_parse_game(calls))
    stats = StatsNHL()
    stats.gamelog = _gamelog()
    before = stats.gamelog.drop(columns=_SUMS)

    fetched = stats.backfill_season_sums()

    assert fetched == 1
    assert calls == [(_GAME_B, "2025-10-08")]
    after = stats.gamelog.set_index("playerName")
    assert after.loc["B1", _SUMS].tolist() == [7.0] * len(_SUMS)
    assert after.loc["B2", _SUMS].tolist() == [8.0] * len(_SUMS)
    assert after.loc["B3", _SUMS].isna().all()
    assert after.loc[["A1", "A2"], _SUMS].to_numpy().tolist() == [[1.5] * 21, [2.5] * 21]
    assert after.loc["B1", "shots"] == 3.0
    pd.testing.assert_frame_equal(stats.gamelog.drop(columns=_SUMS), before)

    assert stats.backfill_season_sums(seasons={2024}) == 0
    assert calls == [(_GAME_B, "2025-10-08")]


def test_backfill_adds_the_columns_a_stored_frame_never_had(monkeypatch):
    calls = []
    monkeypatch.setattr(StatsNHL, "parse_game", _fake_parse_game(calls))
    stats = StatsNHL()
    stats.gamelog = _gamelog().drop(columns=_SUMS)

    fetched = stats.backfill_season_sums(seasons={2025})

    assert fetched == 2
    assert calls == [(_GAME_B, "2025-10-08"), (_GAME_A, "2025-10-07")]
    assert list(stats.gamelog.columns[-len(_SUMS) :]) == _SUMS
    filled = stats.gamelog.set_index("playerName").loc[["B1", "B2", "A1", "A2"], _SUMS]
    assert filled.to_numpy().tolist() == [[7.0] * 21, [8.0] * 21, [7.0] * 21, [8.0] * 21]
    assert stats.gamelog.set_index("playerName").loc["B3", _SUMS].isna().all()


class _SilentSites:
    def get(self, url):
        return {}

    def get_csv(self, url):
        return pd.DataFrame()


def test_update_heals_the_running_season(monkeypatch):
    monkeypatch.setattr(nhl, "scraper", _SilentSites())
    monkeypatch.setattr(nhl, "write_gamelog", lambda *bundle: None)
    monkeypatch.setattr(StatsNHL, "dump_goalie_list", lambda self: None)
    healed = []
    monkeypatch.setattr(
        StatsNHL, "backfill_season_sums", lambda self, seasons=None: healed.append(seasons) or 0
    )
    stats = StatsNHL()
    stats.season_start = date.today() + timedelta(days=5)
    stats.gamelog = pd.DataFrame(columns=["gameId", "gameDate", "playerName", "team"])
    stats.teamlog = pd.DataFrame(columns=["gameId", "gameDate", "team"])

    stats._update()

    assert healed == [{stats.season_start.year}]
