"""The NHL comp pool at a date knows nothing played on or after it.

``players.json`` holds one row per player per season: static bios plus the moneypuck
season summary as of the last fetch. Past seasons' rows are finals and pool as stored; the
running season's row is season-to-date as of the fetch, so the pool reads only its bios and
rebuilds its rates from the regular-season games before the date (I6f).
"""

import copy
import importlib.resources as pkg_resources
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from sportstradamus import data
from sportstradamus.stats import StatsNHL
from sportstradamus.stats.nhl import PLAYER_BIO_FIELDS, SEASON_SUM_COLUMNS

D = date(2026, 10, 20)
A, B, N, G1, O = 8470001, 8470002, 8470003, 8470004, 8470005
_SKATER_RATES = [
    "Fenwick",
    "timePerGame",
    "timePerShift",
    "shotAttempts",
    "xGoals",
    "shotsOnGoal",
    "goals",
    "rebounds",
    "freeze",
    "oZoneStarts",
    "flyStarts",
    "hits",
    "takeaways",
    "giveaways",
    "assists",
    "penaltyMinutes",
    "penaltyMinutesDrawn",
    "blockedShots",
]
_GOALIE_RATES = ["timePerGame", "savePct", "freezeAgainst", "reboundsAgainst", "goalsAgainst"]


def _bio(name, position, team, age, birth_date=None):
    bio = {
        "playerName": name,
        "position": position,
        "team": team,
        "weight": 200.0,
        "height": 74,
        "shootsCatches": "L",
        "bmi": 0.0365,
        "age": age,
    }
    return bio if birth_date is None else bio | {"birthDate": birth_date}


_PLAYERS = {
    2025: {
        A: _bio("Auston", "C", "TOR", 28.1) | dict.fromkeys(_SKATER_RATES, 1.5),
        N: _bio("Nick", "W", "BOS", 31.2) | dict.fromkeys(_SKATER_RATES, 2.5),
        G1: _bio("Jeremy", "G", "BOS", 26.9) | dict.fromkeys(_GOALIE_RATES, 3.5),
        O: _bio("Old", "D", "NYR", 38.0) | dict.fromkeys(_SKATER_RATES, 4.5),
    },
    2026: {
        A: _bio("Auston", "C", "TOR", 29.1, "1997-09-17") | dict.fromkeys(_SKATER_RATES, 999),
        B: _bio("Brandon", "D", "TOR", 24.5) | dict.fromkeys(_SKATER_RATES, 999),
        N: _bio("Nick", "W", "BOS", 32.2) | dict.fromkeys(_SKATER_RATES, 999),
        G1: _bio("Jeremy", "G", "BOS", 27.9, "1998-11-24") | dict.fromkeys(_GOALIE_RATES, 999),
    },
}
_BOX = ["timeOnIce", "shots", "goals", "assists", "hits", "blocked", "shotsAgainst", "goalsAgainst"]
_A_GAMES = [
    {
        "timeOnIce": 20,
        "hits": 2,
        "xGoals": 0.5,
        "shotAttempts": 5,
        "shifts": 20,
        "oZoneStarts": 6,
        "dZoneStarts": 4,
        "flyStarts": 10,
        "onIceFenwickFor": 15,
        "onIceFenwickAgainst": 10,
        "offIceFenwickFor": 20,
        "offIceFenwickAgainst": 20,
    },
    {
        "timeOnIce": 16,
        "hits": 4,
        "xGoals": 0.3,
        "shotAttempts": 3,
        "shifts": 16,
        "oZoneStarts": 4,
        "dZoneStarts": 2,
        "flyStarts": 6,
        "onIceFenwickFor": 5,
        "onIceFenwickAgainst": 10,
        "offIceFenwickFor": 20,
        "offIceFenwickAgainst": 10,
    },
]
_G1_GAMES = [
    {
        "timeOnIce": 60,
        "shotsAgainst": 30,
        "goalsAgainst": 2,
        "freezesAgainst": 6,
        "xFreezesAgainst": 4,
    },
    {
        "timeOnIce": 40,
        "shotsAgainst": 20,
        "goalsAgainst": 3,
        "freezesAgainst": 4,
        "xFreezesAgainst": 3,
    },
]


def _row(game, day, pid, position, **values):
    """One gamelog row; every stat the profile reads is 1.0 unless given."""
    return (
        {
            "gameId": game,
            "gameDate": day,
            "playerId": str(pid),
            "playerName": _PLAYERS[2026][pid]["playerName"],
            "position": position,
        }
        | dict.fromkeys([*_BOX, *SEASON_SUM_COLUMNS], 1.0)
        | values
    )


def _gamelog():
    huge = dict.fromkeys([*_BOX, *SEASON_SUM_COLUMNS], 5000.0)
    return pd.DataFrame(
        [
            _row(2026020001, "2026-10-08", A, "C", **_A_GAMES[0]),
            _row(2026020001, "2026-10-08", B, "D"),
            _row(2026020001, "2026-10-08", G1, "G", **_G1_GAMES[0]),
            _row(2026030001, "2026-10-10", A, "C", **huge),
            _row(2026020010, "2026-10-12", A, "C", **_A_GAMES[1]),
            _row(2026020010, "2026-10-12", G1, "G", **_G1_GAMES[1]),
            _row(2026020020, "2026-10-15", B, "D"),
            _row(2026020030, "2026-10-20", A, "C", **huge),
            _row(2026020030, "2026-10-20", N, "W", **huge),
            _row(2026020040, "2026-10-25", A, "C", **huge),
            _row(2026020040, "2026-10-25", B, "D", **huge),
            _row(2026020040, "2026-10-25", N, "W", **huge),
            _row(2026020040, "2026-10-25", G1, "G", **huge),
            _row(2025020500, "2026-03-01", A, "C", **huge),
        ]
    )


def _stats(gamelog=None):
    stats = StatsNHL()
    stats.players = copy.deepcopy(_PLAYERS)
    stats.gamelog = _gamelog() if gamelog is None else gamelog
    return stats


def test_profile_counts_the_seasons_regular_season_games_before_the_date_only():
    gamelog = _gamelog()
    full = _stats(gamelog)._season_to_date_profile(2026, D)
    before_d = gamelog[pd.to_datetime(gamelog["gameDate"]).dt.date < D]
    counted = before_d[before_d["gameId"].isin([2026020001, 2026020010, 2026020020])]

    pd.testing.assert_frame_equal(full, _stats(before_d)._season_to_date_profile(2026, D))
    pd.testing.assert_frame_equal(full, _stats(counted)._season_to_date_profile(2026, D))
    assert set(full.index) == {A, B, G1}
    assert _stats(gamelog)._season_to_date_profile(2025, D).index.tolist() == [A]


def test_profile_rates_are_the_season_summary_formulas_over_the_sums():
    profile = _stats()._season_to_date_profile(2026, D)

    skater = profile.loc[A]
    assert skater["hits"] == pytest.approx(6 / (36 * 60) * 3600)
    assert skater["xGoals"] == pytest.approx(0.8 / 8)
    assert skater["Fenwick"] == pytest.approx(20 / (20 + 20) - 40 / (40 + 30))
    assert skater["timePerGame"] == pytest.approx(36 / 2)
    assert skater["timePerShift"] == pytest.approx(36 * 60 / 36)
    assert skater["oZoneStarts"] == pytest.approx(10 / (10 + 6))
    assert skater["flyStarts"] == pytest.approx(16 / 36)
    assert skater[_GOALIE_RATES[1:]].isna().all()
    goalie = profile.loc[G1]
    assert goalie["freezeAgainst"] == pytest.approx((10 - 7) / (50 - 5))
    assert goalie["timePerGame"] == pytest.approx(100 / 2)
    assert goalie[[c for c in _SKATER_RATES if c != "timePerGame"]].isna().all()


def test_pool_reads_the_running_seasons_row_for_its_bios_only():
    stats = _stats()
    with open(pkg_resources.files(data) / "config" / "playerCompStats.json") as f:
        weights = json.load(f)["NHL"]

    pool = stats._player_seasons_through(D)

    assert 999 not in {v for row in pool.values() for v in row.values()}
    for pid, position in ((A, "C"), (B, "D"), (G1, "G")):
        assert set(weights[position]) <= set(pool[pid]), position
    assert set(pool[A]) == set(PLAYER_BIO_FIELDS) | set(_SKATER_RATES)
    assert set(pool[B]) == set(PLAYER_BIO_FIELDS) - {"birthDate"} | set(_SKATER_RATES)
    assert set(pool[G1]) == set(PLAYER_BIO_FIELDS) | set(_GOALIE_RATES)
    assert pool[A]["hits"] == pytest.approx(10.0)
    assert pool[A]["team"] == "TOR" and pool[G1]["position"] == "G"
    assert pool[N] == _PLAYERS[2025][N]
    assert pool[O] == _PLAYERS[2025][O]
    assert stats.players == _PLAYERS


def test_pool_ages_a_row_with_a_birth_date_as_of_the_date():
    pool = _stats()._player_seasons_through(D)

    assert pool[A]["age"] == pytest.approx((D - date(1997, 9, 17)).days / 365.25)
    assert pool[G1]["age"] == pytest.approx((D - date(1998, 11, 24)).days / 365.25)
    assert pool[B]["age"] == 24.5


def test_pool_leaves_the_gamelog_alone_for_a_season_the_roster_does_not_hold():
    stats = _stats(pd.DataFrame())
    del stats.players[2026]

    assert stats._player_seasons_through(D) == _PLAYERS[2025]


@pytest.mark.parametrize(
    ("day", "season"), [(date(2026, 9, 30), 2026), (date(2026, 7, 31), 2025), (D, 2026)]
)
def test_season_key_rolls_over_in_august(day, season):
    assert StatsNHL()._current_season_key(day) == season


def test_profile_refuses_a_slice_missing_a_season_sum():
    gamelog = _gamelog()
    gamelog.loc[gamelog["gameId"] == 2026030001, "shifts"] = np.nan
    gamelog.loc[gamelog["gameDate"] == "2026-10-25", "xGoals"] = np.nan
    _stats(gamelog)._season_to_date_profile(2026, D)

    gamelog.loc[(gamelog["gameId"] == 2026020001) & (gamelog["playerId"] == str(A)), "shifts"] = (
        np.nan
    )
    with pytest.raises(RuntimeError, match="backfill-nhl-gamelog"):
        _stats(gamelog)._season_to_date_profile(2026, D)
