"""Pin the NFL and MLB weather features: ``stats.weather_features``'s two tables and their hook.

Every fixture sets a trap a naive build falls into: rows out of game, hour and lead order, a game
stored at three leads (only lead 1's hours count) and one stored at lead 0 alone, a dome's lone
row with no lead, an older capture with a later clock, a game only an older capture holds, a dome
the capture still dates from its run, and MLB's statsapi game id, an integer in the gamelog.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from sportstradamus.collectors.weather_capture import CAPTURE_DTYPES
from sportstradamus.collectors.weather_history import _DTYPES as HISTORY_DTYPES
from sportstradamus.helpers.io import write_capture_parquet
from sportstradamus.stats import Stats, StatsMLB, StatsNFL, base, weather_features
from sportstradamus.stats.mlb import _build_mlb_upcoming_games
from sportstradamus.stats.weather_features import (
    WEATHER_COLUMNS,
    forecast_weather,
    history_weather,
)

# A fixture row's fields; every other column is set per file.
FIELDS = [
    "game_id",
    "roof",
    "forecast_hour",
    "wind_speed_10m",
    "wind_gusts_10m",
    "temperature_2m",
    "precipitation",
    "snowfall",
    "relative_humidity_2m",
]
NAN = float("nan")
NO_WEATHER = [NAN] * 6
H1, H2, H3 = "2025-07-04 23:00", "2025-07-05 00:00", "2025-07-05 01:00"


@pytest.fixture
def stores(monkeypatch, tmp_path):
    """Both readers pointed under ``tmp_path``: the history's leagues dir and the capture dir."""
    leagues_dir, capture_dir = tmp_path / "leagues", tmp_path / "weather"
    monkeypatch.setattr(weather_features, "LEAGUES_DIR", leagues_dir)
    monkeypatch.setattr(weather_features, "WEATHER_CAPTURE_DIR", capture_dir)
    return leagues_dir, capture_dir


def _store_history(leagues_dir, league, rows, **columns):
    """Store ``(*FIELDS, lead_days)`` rows as the league's history file, typed as stored."""
    frame = pd.DataFrame(rows, columns=[*FIELDS, "lead_days"]).assign(**columns)
    path = leagues_dir / league.lower() / "weather_history.parquet"
    path.parent.mkdir(parents=True)
    frame.reindex(columns=list(HISTORY_DTYPES)).astype(HISTORY_DTYPES).to_parquet(path, index=False)


def _store_capture(capture_dir, league, polled_at, rows, **columns):
    """Store ``FIELDS`` rows as one run's capture, where and as the run writes it."""
    polled_at = pd.Timestamp(polled_at)
    frame = pd.DataFrame(rows, columns=FIELDS).assign(polled_at=polled_at, **columns)
    typed = frame.reindex(columns=list(CAPTURE_DTYPES)).astype(CAPTURE_DTYPES)
    write_capture_parquet(typed, capture_dir / league, polled_at)


def _expected(league, games):
    """The table of ``{game_id: values in WEATHER_COLUMNS order}``."""
    return pd.DataFrame(
        list(games.values()),
        index=pd.Index(list(games), dtype="string", name="game_id"),
        columns=list(WEATHER_COLUMNS[league]),
        dtype=float,
    )


def test_history_reads_one_forecast_a_game_lead_one_where_stored(stores):
    leagues_dir, _ = stores
    # Lambeau holds leads 0-2 and Dallas (retractable) 0-1, lead 1 never first; Miami (2022)
    # holds lead 0 only, snowfall unarchived; Detroit is a dome: one row, no hour, no lead.
    _store_history(
        leagues_dir,
        "NFL",
        [
            ("2024_16_NO_GB", "open", H2, 30, 40, 0, 5, 1, 99, 2),
            ("2024_16_TB_DAL", "retractable", H1, 2, 3, 50, 1, 0, 90, 0),
            ("2022_15_NYJ_MIA", "open", H2, 5, 8, 77, 0, NAN, 60, 0),
            ("2024_16_NO_GB", "open", H1, 5, 9, 30, 0, 0, 80, 0),
            ("2024_16_CHI_DET", "dome", None, *NO_WEATHER, None),
            ("2024_16_NO_GB", "open", H2, 14, 25, 22, 0.3, 0.5, 70, 1),
            ("2024_16_TB_DAL", "retractable", H2, 8, 14, 72, 0, 0, 50, 1),
            ("2024_16_NO_GB", "open", H1, 30, 40, 0, 5, 1, 99, 2),
            ("2024_16_TB_DAL", "retractable", H2, 2, 3, 50, 1, 0, 90, 0),
            ("2022_15_NYJ_MIA", "open", H1, 3, 6, 75, 0, NAN, 50, 0),
            ("2024_16_NO_GB", "open", H1, 10, 20, 18, 0.1, 0.0, 60, 1),
            ("2024_16_NO_GB", "open", H2, 7, 11, 32, 0.2, 0, 82, 0),
            ("2024_16_TB_DAL", "retractable", H1, 8, 12, 70, 0, 0, 40, 1),
        ],
    )

    # Means of wind, temperature and humidity, the gust's max, precipitation and snow summed:
    #   Lambeau lead 1: wind (10 + 14) / 2, gust 25, temp (18 + 22) / 2, precip 0.1 + 0.3,
    #     snow 0 + 0.5, humidity (60 + 70) / 2
    #   Dallas lead 1: 8, 14, 71, 0, 0, 45, flagged retractable
    #   Miami lead 0: 4, 8, 76, 0, snow NaN, 55
    expected = _expected(
        "NFL",
        {
            "2022_15_NYJ_MIA": [4, 8, 76, 0, NAN, 55, 0, 0, 0],
            "2024_16_CHI_DET": [*NO_WEATHER, NAN, 1, 0],
            "2024_16_NO_GB": [12, 25, 20, 0.4, 0.5, 65, 1, 0, 0],
            "2024_16_TB_DAL": [8, 14, 71, 0, 0, 45, 1, 0, 1],
        },
    )
    pd.testing.assert_frame_equal(history_weather("NFL"), expected)


def test_mlb_wind_out_is_the_wind_toward_centre_field_at_each_park(stores):
    leagues_dir, _ = stores
    # Fenway's field points 45 degrees and Camden Yards' 31. Open-Meteo gives where the wind
    # comes from: from 225 at Fenway blows straight out (+10), from 45 straight in (-6), from 135
    # across (0); from 211 at Camden Yards straight out (+12).
    _store_history(
        leagues_dir,
        "MLB",
        [
            ("746000", "open", H2, 6, 9, 70, 0, 0, 50, 1),
            ("746001", "open", H1, 12, 15, 80, 0, 0, 40, 1),
            ("746000", "open", H3, 8, 9, 70, 0, 0, 50, 1),
            ("746000", "open", H1, 10, 12, 70, 0, 0, 50, 1),
        ],
        venue=["3", "2", "3", "3"],
        wind_direction_10m=[45.0, 211.0, 135.0, 225.0],
    )

    features = history_weather("MLB")

    assert list(features.columns) == list(WEATHER_COLUMNS["MLB"])
    assert features.loc["746000", "weather wind out"] == pytest.approx((10 - 6 + 0) / 3)
    assert features.loc["746001", "weather wind out"] == pytest.approx(12)


def test_serving_reads_only_the_newest_capture(stores):
    _, capture_dir = stores
    # The older capture has the later clock, and holds a game the newest does not.
    _store_capture(
        capture_dir,
        "NFL",
        "2026-11-15 23:50:06",
        [
            ("2026_11_MIA_BUF", "open", H1, 99, 99, 99, 9, 9, 99),
            ("2026_11_PIT_CLE", "open", H1, 7, 9, 40, 0, 0, 70),
        ],
        kickoff=pd.Timestamp("2026-11-17 12:50"),
    )
    _store_capture(
        capture_dir,
        "NFL",
        "2026-11-16 00:05:00",
        [("2026_11_MIA_BUF", "open", H1, 50, 50, 50, 5, 5, 50)],
        kickoff=pd.Timestamp("2026-11-17 12:50"),
    )
    _store_capture(
        capture_dir,
        "NFL",
        "2026-11-16 00:50:00",
        [
            ("2026_11_MIA_BUF", "open", H2, 6, 12, 34, 0, 1.0, 80),
            ("2026_11_TB_DET", "dome", None, *NO_WEATHER),
            ("2026_11_MIA_BUF", "open", H1, 4, 10, 30, 0.2, 0.5, 70),
        ],
        kickoff=pd.to_datetime(["2026-11-17 12:50", "2026-11-22 18:00", "2026-11-17 12:50"]),
    )

    # Buffalo kicks off a day and a half after the run; the dome's game has no forecast, so no
    # lead either.
    expected = _expected(
        "NFL",
        {
            "2026_11_MIA_BUF": [5, 12, 32, 0.2, 1.5, 75, 1.5, 0, 0],
            "2026_11_TB_DET": [*NO_WEATHER, NAN, 1, 0],
        },
    )
    pd.testing.assert_frame_equal(forecast_weather("NFL"), expected)


@pytest.mark.parametrize("league", ["NFL", "MLB"])
@pytest.mark.parametrize("reader", [history_weather, forecast_weather], ids=["history", "capture"])
def test_with_nothing_stored_a_table_is_empty_with_the_leagues_columns(stores, reader, league):
    table = reader(league)

    assert table.empty
    assert list(table.columns) == list(WEATHER_COLUMNS[league])
    assert (table.dtypes == float).all()


def test_the_hook_gives_a_game_the_same_row_from_the_history_and_the_capture(stores):
    leagues_dir, capture_dir = stores
    kickoff = pd.Timestamp("2026-01-04 18:00")
    hours = [
        ("2025_18_DET_GB", "open", H2, 14, 25, 22, 0.3, 0.5, 70),
        ("2025_18_GB_DET", "dome", None, *NO_WEATHER),
        ("2025_18_DET_GB", "open", H1, 10, 20, 18, 0.1, 0.0, 60),
    ]
    # The history holds the capture's hours at lead 1 (the dome's without a lead) and a stale
    # lead-0 hour.
    _store_history(
        leagues_dir,
        "NFL",
        [
            ("2025_18_DET_GB", "open", H1, 30, 40, 0, 5, 1, 99, 0),
            (*hours[0], 1),
            (*hours[1], None),
            (*hours[2], 1),
        ],
    )
    # Polled a day before kickoff, so the capture's lead matches the history's.
    _store_capture(
        capture_dir, "NFL", kickoff - pd.Timedelta(days=1), hours, kickoff=[kickoff] * len(hours)
    )
    stats = Stats()
    stats.league = "NFL"
    game_ids = {
        "Jordan Love": "2025_18_DET_GB",
        "Jared Goff": "2025_18_GB_DET",
        "Unknown Game": "2025_18_XXX_YYY",
        "No Schedule": None,
    }
    trained, served = pd.DataFrame(index=list(game_ids)), pd.DataFrame(index=list(game_ids))

    stats._weather_context(trained, date(2026, 1, 4), game_ids)
    stats._weather_context(served, date(2099, 1, 4), game_ids)

    pd.testing.assert_frame_equal(trained, served)
    assert list(trained.columns) == list(WEATHER_COLUMNS["NFL"])
    assert trained.loc["Jordan Love"].tolist() == [12, 25, 20, 0.4, 0.5, 65, 1, 0, 0]
    assert trained.loc["Jared Goff", "weather indoor"] == 1
    assert trained.loc[["Unknown Game", "No Schedule"]].isna().all().all()


def test_a_league_that_plays_indoors_gets_no_weather_columns():
    stats = Stats()
    stats.league = "NBA"
    frame = pd.DataFrame(index=["Nikola Jokic"])

    stats._weather_context(frame, date(2026, 1, 4), {"Nikola Jokic": "0022500501"})

    assert frame.columns.empty


class _StubArchive:
    """The moneyline and total reads of ``_game_context``'s upcoming branch, off DuckDB."""

    def get_moneyline(self, league, game_date, team, **_):
        return 0.5

    def get_total(self, league, game_date, team, **_):
        return 4.5


def test_mlb_game_context_finds_a_games_weather_by_its_integer_game_id(stores, monkeypatch):
    leagues_dir, capture_dir = stores
    kickoff = pd.Timestamp("2025-07-04 23:10")
    hours = [
        ("746000", "open", H1, 10, 12, 70, 0, 0, 50),
        ("746000", "open", H2, 6, 8, 66, 0, 0, 60),
    ]
    fenway = {"venue": "3", "wind_direction_10m": 225.0}
    _store_history(leagues_dir, "MLB", [(*row, 1) for row in hours], **fenway)
    _store_capture(
        capture_dir, "MLB", kickoff - pd.Timedelta(days=1), hours, kickoff=kickoff, **fenway
    )
    monkeypatch.setattr(base, "archive", _StubArchive())
    stats = StatsMLB()
    stats.gamelog = pd.DataFrame(
        {
            "gameId": 746000,
            "gameDate": "2025-07-04",
            "playerName": ["Rafael Devers", "Aaron Judge"],
            "team": ["BOS", "NYY"],
            "opponent": ["NYY", "BOS"],
            "home": [True, False],
            "moneyline": 0.5,
            "totals": 4.5,
            "opponent pitcher": ["Gerrit Cole", "Brayan Bello"],
        }
    )
    stats.upcoming_games = {
        "BOS": {"Home": True, "Opponent Pitcher": "Gerrit Cole", "Game ID": "746000"},
        "NYY": {"Home": False, "Opponent Pitcher": "Brayan Bello", "Game ID": "746000"},
    }
    teams = {"Rafael Devers": "BOS", "Aaron Judge": "NYY"}
    opponents = {"Rafael Devers": "NYY", "Aaron Judge": "BOS"}
    offers = [{"Player": p, "Team": t, "Opponent": opponents[p]} for p, t in teams.items()]
    columns = list(WEATHER_COLUMNS["MLB"])

    trained, *_ = stats._game_context(
        pd.DataFrame(index=list(teams)), offers, "hits", date(2025, 7, 4), teams, opponents
    )
    served, *_ = stats._game_context(
        pd.DataFrame(index=list(teams)), offers, "hits", date(2099, 7, 4), teams, opponents
    )

    assert trained[columns].notna().all().all()
    pd.testing.assert_frame_equal(trained[columns], served[columns])


def test_each_leagues_schedule_entry_carries_its_game_id_as_text():
    nfl = StatsNFL()
    nfl._compute_upcoming_games(
        pd.DataFrame(
            {
                "game_id": ["2099_11_MIA_BUF"],
                "gameday": ["2099-11-22"],
                "away_team": ["MIA"],
                "home_team": ["BUF"],
                "week": [11],
            }
        )
    )
    game = {
        "game_id": 746000,
        "game_date": "2099-07-04",
        "status": "Scheduled",
        "game_type": "R",
        "game_num": 1,
        "away_id": 147,
        "home_id": 111,
        "away_probable_pitcher": "Gerrit Cole",
        "home_probable_pitcher": "Brayan Bello",
    }
    mlb_teams = {"teams": [{"id": 147, "abbreviation": "NYY"}, {"id": 111, "abbreviation": "BOS"}]}
    _, mlb_by_date = _build_mlb_upcoming_games([game], mlb_teams, date(2099, 7, 3))

    assert {entry["Game ID"] for entry in nfl.upcoming_games.values()} == {"2099_11_MIA_BUF"}
    assert {entry["Game ID"] for entry in mlb_by_date.values()} == {"746000"}
