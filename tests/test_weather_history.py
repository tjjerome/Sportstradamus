"""Weather history backfill: request spans and bill, lead reshaping, budget, resume, failures."""

from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd
import pytest
import requests
from click.testing import CliRunner

from sportstradamus.collectors import weather_history as wh
from sportstradamus.collectors.weather import MLB_SCHEDULE_URL, VENUES

# The hourly variables the archives keep: precipitation_probability is the live forecast's alone.
_ARCHIVED = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "precipitation",
    "relative_humidity_2m",
    "snowfall",
]
_PREVIOUS_RUNS_NAMES = [
    f"{var}_previous_day{lead}" if lead else var for lead in range(8) for var in _ARCHIVED
]
_PREVIOUS_RUNS = "https://previous-runs-api.open-meteo.com/v1/forecast"
_HISTORICAL_FORECAST = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# Three Week 2 2025 games at three open-air venues: three requests, 48 rows each.
_WEEK_2 = [
    (2025, "2025_02_NYJ_BUF", "2025-09-14", "13:00", "outdoors", "Highmark Stadium"),
    (2025, "2025_02_WAS_GB", "2025-09-14", "16:25", "outdoors", "Lambeau Field"),
    (2025, "2025_02_MIN_CHI", "2025-09-14", "13:00", "outdoors", "Soldier Field"),
]


def _nfl_schedule(*games: tuple) -> pd.DataFrame:
    """nflverse schedule rows: (season, game_id, gameday, gametime US Eastern, roof, stadium)."""
    return pd.DataFrame(
        games, columns=["season", "game_id", "gameday", "gametime", "roof", "stadium"]
    )


def _mlb_payload(*games: tuple[int, int, pd.Timestamp]) -> dict:
    """A statsapi season schedule, one date per (gamePk, venue id, UTC first pitch)."""
    return {
        "dates": [
            {
                "games": [
                    {
                        "gamePk": pk,
                        "gameDate": f"{start:%Y-%m-%dT%H:%M:%SZ}",
                        "status": {"detailedState": "Final", "startTimeTBD": False},
                        "venue": {"id": venue},
                    }
                ]
            }
            for pk, venue, start in games
        ]
    }


def _reply(params: dict) -> dict:
    """Open-Meteo's answer: every hour of the asked dates, each value lead * 100 + hour of day."""
    hours = pd.date_range(params["start_date"], f"{params['end_date']} 23:00", freq="h")
    hourly = {"time": list(hours.strftime("%Y-%m-%dT%H:%M"))}
    for name in params["hourly"].split(","):
        lead = int(name.partition("_previous_day")[2] or 0)
        hourly[name] = [lead * 100 + hour.hour for hour in hours]
    location = {"latitude": float(params["latitude"]), "longitude": float(params["longitude"])}
    return {**location, "hourly": hourly}


def _stub(
    monkeypatch,
    tmp_path,
    *,
    nfl: pd.DataFrame | None = None,
    mlb: dict[int, dict] | None = None,
    billed: str | None = "7.5",
    refuse: tuple[int, int] | None = None,
    statsapi_status: int = 200,
) -> dict[str, list]:
    """Serve the schedules and Open-Meteo, write under ``tmp_path``; return what was asked.

    ``billed`` is each reply's X-RateLimit-Cost header (None: no header). ``refuse`` is
    (status, n): the n-th Open-Meteo request and later get that status. Every Open-Meteo request
    also records how many rows the files held when it was made.
    """
    asked: dict[str, list] = {
        "nflverse": [],
        "statsapi": [],
        "open_meteo": [],
        "held": [],
        "slept": [],
    }

    def fake_get(url, params, timeout):
        response = requests.Response()
        response.status_code = 200
        if url == MLB_SCHEDULE_URL:
            asked["statsapi"].append(params["season"])
            response.status_code = statsapi_status
            body = (mlb or {}).get(params["season"], {"dates": []})
        else:
            asked["open_meteo"].append({"url": url, **params})
            files = tmp_path.glob("*/weather_history.parquet")
            asked["held"].append(sum(len(pd.read_parquet(file)) for file in files))
            body = _reply(params)
            if billed is not None:
                response.headers["X-RateLimit-Cost"] = billed
            if refuse is not None and len(asked["open_meteo"]) >= refuse[1]:
                response.status_code = refuse[0]
                body = {"error": True, "reason": "Hourly API request limit exceeded."}
        response._content = json.dumps(body).encode()
        return response

    def fake_import_schedules(years):
        asked["nflverse"].append(list(years))
        return nfl if nfl is not None else _nfl_schedule()

    monkeypatch.setattr(wh, "_LEAGUES_DIR", tmp_path)
    monkeypatch.setattr(wh.requests, "get", fake_get)
    monkeypatch.setattr(wh.nfl, "import_schedules", fake_import_schedules)
    monkeypatch.setattr(wh.time, "sleep", asked["slept"].append)
    return asked


def _run(args: list[str], exit_code: int = 0):
    result = CliRunner().invoke(wh.weather_history, args)
    assert result.exit_code == exit_code, result.output
    return result


def _stored(tmp_path, league: str = "nfl") -> pd.DataFrame:
    return pd.read_parquet(tmp_path / league / "weather_history.parquet")


def test_the_bill_matches_what_open_meteo_charged():
    """X-RateLimit-Cost seen: 56 variables over 3 days 5.6, 7 over 397 days 19.85, 7 over 2 1.0."""
    day = date(2024, 6, 1)

    assert wh._cost(56, day, day + timedelta(days=2)) == pytest.approx(5.6)
    assert wh._cost(7, day, day + timedelta(days=396)) == pytest.approx(19.85)
    assert wh._cost(7, day, day + timedelta(days=1)) == 1.0


def test_spans_join_across_a_gap_only_while_one_request_bills_no_more_than_two():
    """56 variables bill 5.6 for any span up to 14 days, 0.4 a day beyond; 7 bill 1 up to 20."""
    day = [date(2025, 6, 1) + timedelta(days=offset) for offset in range(70)]

    assert wh._spans([day[3], day[1], day[2], day[1]], 56) == [[day[1], day[3]]]
    assert wh._spans([day[0], day[7]], 56) == [[day[0], day[7]]]
    assert wh._spans([day[0], day[35]], 56) == [[day[0], day[0]], [day[35], day[35]]]
    # 28 days joined bill 11.2, the same as two apart: a tie joins; 29 days bill 11.6.
    assert wh._spans([day[0], day[27]], 56) == [[day[0], day[27]]]
    assert wh._spans([day[0], day[28]], 56) == [[day[0], day[0]], [day[28], day[28]]]
    # Joined, two 14-day runs 10 days apart bill 15.2 against 11.2 apart, although the second
    # run's first day alone would join (10.0 against 11.2): runs join whole, not day by day.
    two_runs = [*day[0:14], *day[24:38]]
    assert wh._spans(two_runs, 56) == [[day[0], day[13]], [day[24], day[37]]]
    # One missing day splits runs too: 29 days joined bill 11.6 against 11.2 apart.
    assert wh._spans([*day[0:14], *day[15:29]], 56) == [[day[0], day[13]], [day[15], day[28]]]
    assert wh._spans([day[0], *day[5:8]], 56) == [[day[0], day[7]]]
    assert wh._spans([day[0], day[7], day[14], day[21], day[28]], 7) == [[day[0], day[28]]]
    assert wh._spans([day[0], day[60]], 7) == [[day[0], day[0]], [day[60], day[60]]]


def test_previous_runs_columns_reshape_to_one_row_per_lead():
    """Lead 0 is the unsuffixed column, lead N the variable's ``_previous_dayN`` column."""
    values = {name: [0.0] for name in _PREVIOUS_RUNS_NAMES}
    values["temperature_2m"] = [61.0]
    values["temperature_2m_previous_day3"] = [55.0]
    values["wind_gusts_10m_previous_day7"] = [31.0]
    hour = pd.Timestamp("2025-09-14 17:00")
    hourly = pd.DataFrame({"venue": ["Lambeau Field"], "forecast_hour": [hour], **values})

    long = wh._by_lead(hourly)

    expected = pd.DataFrame(0.0, index=range(8), columns=_ARCHIVED)
    expected.loc[0, "temperature_2m"] = 61.0
    expected.loc[3, "temperature_2m"] = 55.0
    expected.loc[7, "wind_gusts_10m"] = 31.0
    assert long.columns.tolist() == ["venue", "forecast_hour", *_ARCHIVED, "lead_days"]
    assert long["lead_days"].tolist() == list(range(8))
    assert (long["forecast_hour"] == hour).all()
    pd.testing.assert_frame_equal(long[_ARCHIVED], expected)


def test_each_season_reads_its_archive_over_every_utc_date_its_window_touches(
    monkeypatch, tmp_path
):
    """2023 reads the Historical Forecast, 2025 Previous Runs at every lead. Both 2025 windows
    cross UTC midnight: Sunday night's back from its 00:20 kickoff, the 4:25 pm game's forward
    to 00:00. No cost header: the bill is estimated."""
    schedule = _nfl_schedule(
        (2025, "2025_02_WAS_GB", "2025-09-14", "20:20", "outdoors", "Lambeau Field"),
        (2025, "2025_02_MIN_CHI", "2025-09-14", "16:25", "outdoors", "Soldier Field"),
        (2023, "2023_02_CHI_GB", "2023-09-17", "13:00", "outdoors", "Lambeau Field"),
    )
    asked = _stub(monkeypatch, tmp_path, nfl=schedule, billed=None)

    result = _run(["--league", "nfl", "--season", "2023", "--season", "2025"])

    assert asked["statsapi"] == []
    assert [
        (call["url"], call["start_date"], call["end_date"]) for call in asked["open_meteo"]
    ] == [
        (_HISTORICAL_FORECAST, "2023-09-17", "2023-09-17"),
        (_PREVIOUS_RUNS, "2025-09-14", "2025-09-15"),
        (_PREVIOUS_RUNS, "2025-09-14", "2025-09-15"),
    ]
    historical, *previous = (call["hourly"].split(",") for call in asked["open_meteo"])
    assert historical == _ARCHIVED
    assert [sorted(names) for names in previous] == [sorted(_PREVIOUS_RUNS_NAMES)] * 2

    rows = _stored(tmp_path)
    assert rows.dtypes.to_dict() == {
        column: pd.Series(dtype=dtype).dtype for column, dtype in wh._DTYPES.items()
    }
    games = rows.groupby("game_id")
    assert games["forecast_hour"].min().to_dict() == {
        "2023_02_CHI_GB": pd.Timestamp("2023-09-17 16:00"),
        "2025_02_MIN_CHI": pd.Timestamp("2025-09-14 19:00"),
        "2025_02_WAS_GB": pd.Timestamp("2025-09-14 23:00"),
    }
    assert games["source"].first().to_dict() == {
        "2023_02_CHI_GB": "historical_forecast",
        "2025_02_MIN_CHI": "previous_runs",
        "2025_02_WAS_GB": "previous_runs",
    }
    assert rows.groupby(["game_id", "lead_days"]).size().to_dict() == {
        ("2023_02_CHI_GB", 0): 6,
        **{(game, lead): 6 for game in ("2025_02_MIN_CHI", "2025_02_WAS_GB") for lead in range(8)},
    }
    # The fake reply's value is lead * 100 + hour: each row holds its own lead's hour.
    assert (rows["temperature_2m"] == rows["lead_days"] * 100 + rows["forecast_hour"].dt.hour).all()
    assert rows["precipitation_probability"].isna().all()
    assert asked["slept"] == pytest.approx([cost * 3600 / 4500 for cost in (1.0, 5.6, 5.6)])
    assert result.stdout.strip() == "weather-history: 3 requests, cost 12.2, new rows NFL=102"


def test_games_younger_than_two_days_and_dome_games_are_not_requested(monkeypatch, tmp_path):
    """And a league with nothing to fetch (the NFL here, no games) writes no file."""
    now = pd.Timestamp.now("UTC").tz_localize(None).floor("h")
    payload = _mlb_payload(
        (1, 17, now - pd.Timedelta(days=3)),
        (2, 17, now - pd.Timedelta(days=1)),
        (3, 12, now - pd.Timedelta(days=5)),
    )
    asked = _stub(monkeypatch, tmp_path, mlb={now.year: payload})

    _run(["--season", str(now.year)])

    assert asked["statsapi"] == [now.year]
    assert [call["latitude"] for call in asked["open_meteo"]] == [str(VENUES["MLB"]["17"]["lat"])]
    assert set(_stored(tmp_path, "mlb")["game_id"]) == {"1"}
    assert not (tmp_path / "nfl").exists()


@pytest.mark.parametrize("max_cost", ["10", "15"])
def test_max_cost_stops_once_the_run_has_spent_it(monkeypatch, tmp_path, max_cost):
    """Billed 7.5 each, the second request crosses a 10-call budget and reaches a 15-call one."""
    asked = _stub(monkeypatch, tmp_path, nfl=_nfl_schedule(*_WEEK_2), billed="7.5")

    result = _run(["--league", "NFL", "--season", "2025", "--max-cost", max_cost])

    assert len(asked["open_meteo"]) == 2
    assert _stored(tmp_path)["game_id"].nunique() == 2
    assert result.stdout.strip() == (
        "weather-history: 2 requests, cost 15.0, new rows NFL=96; "
        f"stopped at --max-cost {max_cost}, rerun tomorrow"
    )


def test_a_rerun_requests_only_games_without_rows_from_their_source(monkeypatch, tmp_path):
    """Buffalo's game is stored from its source; Chicago's only from the other archive."""
    stored = pd.DataFrame(
        {
            "game_id": ["2025_02_NYJ_BUF", "2025_02_MIN_CHI"],
            "source": ["previous_runs", "historical_forecast"],
        }
    )
    stored = stored.reindex(columns=list(wh._DTYPES)).astype(wh._DTYPES)
    (tmp_path / "nfl").mkdir()
    stored.to_parquet(tmp_path / "nfl" / "weather_history.parquet")
    asked = _stub(monkeypatch, tmp_path, nfl=_nfl_schedule(*_WEEK_2))

    _run(["--league", "NFL", "--season", "2025"])

    table = VENUES["NFL"]
    assert sorted(call["latitude"] for call in asked["open_meteo"]) == sorted(
        str(table[venue]["lat"]) for venue in ("Lambeau Field", "Soldier Field")
    )
    rows = _stored(tmp_path)
    assert len(rows) == 2 + 2 * 48
    pd.testing.assert_frame_equal(rows.iloc[:2], stored)


@pytest.mark.parametrize(("status", "error"), [(429, SystemExit), (500, requests.HTTPError)])
def test_a_refused_request_keeps_the_rows_already_fetched(monkeypatch, tmp_path, status, error):
    """A 429 (a rate limit spent) exits with a message; any other error propagates as is."""
    _stub(monkeypatch, tmp_path, nfl=_nfl_schedule(*_WEEK_2), refuse=(status, 2))

    result = _run(["--league", "NFL", "--season", "2025"], exit_code=1)

    assert isinstance(result.exception, error)
    assert ("rate limit is spent" in result.output) == (status == 429)
    assert len(_stored(tmp_path)) == 48


def test_rows_are_written_every_few_requests_not_only_at_the_end(monkeypatch, tmp_path):
    """Lambeau's Week 2 and Week 12 games are two requests, each storing only its own game."""
    monkeypatch.setattr(wh, "_WRITE_EVERY", 2)
    week_12 = (2025, "2025_12_NYG_GB", "2025-11-23", "13:00", "outdoors", "Lambeau Field")
    asked = _stub(monkeypatch, tmp_path, nfl=_nfl_schedule(*_WEEK_2[:2], week_12))

    _run(["--league", "NFL", "--season", "2025"])

    assert asked["held"] == [0, 0, 96]
    assert _stored(tmp_path).groupby("game_id").size().tolist() == [48, 48, 48]


def test_a_failed_schedule_read_stops_the_run_before_open_meteo(monkeypatch, tmp_path):
    asked = _stub(monkeypatch, tmp_path, statsapi_status=503)

    result = _run(["--league", "MLB", "--season", "2025"], exit_code=1)

    assert isinstance(result.exception, requests.HTTPError)
    assert asked["open_meteo"] == []


def test_dry_run_prints_the_plan_and_requests_nothing_from_open_meteo(monkeypatch, tmp_path):
    """Defaults: both leagues, 2022 through the current year."""
    schedule = _nfl_schedule(
        (2023, "2023_02_CHI_GB", "2023-09-17", "13:00", "outdoors", "Lambeau Field"),
        *_WEEK_2[1:],
    )
    homestand = _mlb_payload(
        *(
            (pk, 17, pd.Timestamp(f"2024-06-{day:02d} 18:20"))
            for pk, day in [(1, 7), (2, 8), (3, 9)]
        )
    )
    asked = _stub(monkeypatch, tmp_path, nfl=schedule, mlb={2024: homestand})

    result = _run(["--dry-run"])

    seasons = list(range(2022, date.today().year + 1))
    assert asked["nflverse"] == [seasons]
    assert asked["statsapi"] == seasons
    assert asked["open_meteo"] == []
    assert not list(tmp_path.rglob("*.parquet"))
    assert result.stdout.splitlines() == [
        "NFL 2023 historical_forecast: 1 games, 1 requests, cost 1.0",
        "NFL 2025 previous_runs: 2 games, 2 requests, cost 11.2",
        "MLB 2024 previous_runs: 3 games, 1 requests, cost 5.6",
        "total: 6 games, 4 requests, cost 17.8",
    ]
