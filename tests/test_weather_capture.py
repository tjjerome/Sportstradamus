"""Per-run weather capture: the forecast around each NFL and MLB kickoff of the coming week."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from sportstradamus.collectors import weather
from sportstradamus.collectors import weather_capture as wc
from sportstradamus.helpers.scraping import CAPTURE_TIMEOUT_S

# A Sunday run at 6:50 PM CDT: the run's Scored At, naive UTC, with the microseconds a real stamp
# carries. Its horizon ends a week on, 2026-10-04 23:50:07 UTC.
_SCORED_AT = pd.Timestamp("2026-09-27 23:50:07.123456")
_NFL_COLUMNS = ["game_id", "gameday", "gametime", "roof", "stadium"]
# nflverse's schedule, gametime US Eastern (EDT): the afternoon game already played, a week of
# games at open, retractable (blank label, former name) and dome venues, then the next Sunday
# night game, past the horizon.
_NFL_WEEK = pd.DataFrame(
    [
        ("2026_03_LV_KC", "2026-09-27", "16:25", "outdoors", "GEHA Field at Arrowhead Stadium"),
        ("2026_03_DAL_PHI", "2026-09-27", "20:20", "outdoors", "Lincoln Financial Field"),
        ("2026_03_TEN_HOU", "2026-09-28", "20:15", None, "Reliant Stadium"),
        ("2026_04_BUF_MIA", "2026-10-01", "20:15", "outdoors", "Hard Rock Stadium"),
        ("2026_04_MIN_DET", "2026-10-04", "13:00", "dome", "Ford Field"),
        ("2026_04_CHI_GB", "2026-10-04", "16:25", "outdoors", "Lambeau Field"),
        ("2026_04_NYG_PHI", "2026-10-04", "20:20", "outdoors", "Lincoln Financial Field"),
    ],
    columns=_NFL_COLUMNS,
)
# 8:40 PM CDT on July 4, already the 5th in UTC; its horizon ends 2026-07-12 01:40 UTC.
_EVENING_RUN = pd.Timestamp("2026-07-05 01:40")


def _mlb_game(pk: int, venue: int, start: str) -> dict:
    """A statsapi schedule game, cut to the fields the capture's query asks for."""
    return {
        "gamePk": pk,
        "gameDate": start,
        "status": {"detailedState": "Scheduled", "startTimeTBD": False},
        "venue": {"id": venue},
    }


# statsapi's days by US official date: the evening's East Coast game already played, a first
# pitch at the run itself and a West Coast game still to come (all filed under the 4th), a
# Wrigley series, and the last evening's games either side of the horizon.
_MLB_DAYS = [
    {
        "date": "2026-07-04",
        "games": [
            _mlb_game(101, 3313, "2026-07-04T23:05:00Z"),
            _mlb_game(102, 1, "2026-07-05T01:40:00Z"),
            _mlb_game(103, 22, "2026-07-05T02:10:00Z"),
        ],
    },
    {"date": "2026-07-05", "games": [_mlb_game(104, 17, "2026-07-05T18:20:00Z")]},
    {"date": "2026-07-06", "games": [_mlb_game(105, 17, "2026-07-06T18:20:00Z")]},
    {
        "date": "2026-07-11",
        "games": [
            _mlb_game(106, 2680, "2026-07-12T01:40:00Z"),
            _mlb_game(107, 2395, "2026-07-12T02:05:00Z"),
        ],
    },
    {"date": "2026-07-12", "games": [_mlb_game(108, 17, "2026-07-12T18:20:00Z")]},
]
# A game at Tropicana Field, a dome, inside _SCORED_AT's week.
_DOME_DAYS = [{"date": "2026-09-29", "games": [_mlb_game(201, 12, "2026-09-29T23:08:00Z")]}]


def _serve(
    monkeypatch,
    tmp_path,
    scored_at: pd.Timestamp,
    days: Sequence[dict] = (),
    refused: str | None = None,
) -> list[tuple[str, dict, float]]:
    """Stub statsapi and Open-Meteo; return each request as ``(url, params, timeout)``.

    statsapi serves the ``days`` whose official ``date`` lies between the asked dates, the way it
    filters, and drops that key, the way the capture's field list does. Open-Meteo answers each
    asked location in order (an array, or a bare object for one) with ``forecast_days`` of hours
    from 00:00 UTC on the run's date, every variable reading the location's latitude so a row
    shows whose forecast it holds. The ``refused`` URL answers 500 over the same body.
    """
    requested = []

    def fake_get(url, params, timeout):
        requested.append((url, params, timeout))
        response = requests.Response()
        response.status_code = 500 if url == refused else 200
        if url == weather.MLB_SCHEDULE_URL:
            served = [
                {"games": day["games"]}
                for day in days
                if params["startDate"] <= day["date"] <= params["endDate"]
            ]
            payload = {"dates": served}
        else:
            hours = pd.date_range(
                scored_at.normalize(), periods=24 * params["forecast_days"], freq="h"
            )
            locations = [
                {
                    "hourly": {
                        "time": list(hours.strftime("%Y-%m-%dT%H:%M")),
                        **dict.fromkeys(params["hourly"].split(","), [float(lat)] * len(hours)),
                    }
                }
                for lat in params["latitude"].split(",")
            ]
            payload = locations if len(locations) > 1 else locations[0]
            response.headers["X-RateLimit-Cost"] = f"{len(locations)}.0"
        response._content = json.dumps(payload).encode()
        return response

    monkeypatch.setattr(wc, "WEATHER_CAPTURE_DIR", tmp_path)
    monkeypatch.setattr(wc.requests, "get", fake_get)
    return requested


def _written(tmp_path) -> list[str]:
    return sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*.parquet"))


def _nfl_week_rows(monkeypatch, tmp_path) -> pd.DataFrame:
    _serve(monkeypatch, tmp_path, _SCORED_AT)
    wc.capture_weather({"NFL": SimpleNamespace(schedule=_NFL_WEEK)}, _SCORED_AT)
    return pd.read_parquet(tmp_path / "NFL/date=2026-09-27/235007.parquet")


def test_the_nfl_week_is_one_forecast_request_and_one_file(monkeypatch, tmp_path, caplog):
    """MLB, not live, is not asked; the run log shows what the request cost."""
    requested = _serve(monkeypatch, tmp_path, _SCORED_AT)

    with caplog.at_level(logging.INFO, logger="log"):
        wc.capture_weather({"NFL": SimpleNamespace(schedule=_NFL_WEEK)}, _SCORED_AT)

    ((url, params, timeout),) = requested
    assert (url, timeout) == ("https://api.open-meteo.com/v1/forecast", CAPTURE_TIMEOUT_S)
    # Lincoln Financial Field, NRG Stadium, Hard Rock Stadium and Lambeau Field, in schedule
    # order: Ford Field is a dome.
    assert params["latitude"] == "39.901,29.685,25.958,44.501"
    assert params["longitude"] == "-75.168,-95.411,-80.239,-88.062"
    assert params["hourly"] == ",".join(weather.HOURLY_VARS)
    assert params["forecast_days"] == 9
    assert [record.getMessage() for record in caplog.records] == [
        "NFL weather: 5 games, 4 venues requested, Open-Meteo cost 4.0"
    ]
    assert _written(tmp_path) == ["NFL/date=2026-09-27/235007.parquet"]


def test_each_game_ahead_gets_its_window_of_its_own_venues_forecast(monkeypatch, tmp_path):
    """Neither the game already played nor the one past the horizon gets a row.

    Lambeau's window ends at 00:00 UTC on day nine of the forecast, and holds weather there too.
    """
    rows = _nfl_week_rows(monkeypatch, tmp_path)

    assert rows.groupby("game_id").size().to_dict() == {
        "2026_03_DAL_PHI": 6,
        "2026_03_TEN_HOU": 6,
        "2026_04_BUF_MIA": 6,
        "2026_04_CHI_GB": 6,
        "2026_04_MIN_DET": 1,
    }
    outside = rows[rows["roof"] != "dome"]
    latitudes = {name: entry["lat"] for name, entry in weather.VENUES["NFL"].items()}
    assert outside["temperature_2m"].tolist() == outside["venue"].map(latitudes).tolist()
    lambeau = outside[outside["venue"] == "Lambeau Field"]
    assert lambeau["forecast_hour"].max() == pd.Timestamp("2026-10-05 00:00")
    assert set(outside.loc[outside["venue"] == "NRG Stadium", "roof"]) == {"retractable"}


def test_rows_carry_the_run_stamp_each_hours_lead_and_the_pinned_dtypes(monkeypatch, tmp_path):
    rows = _nfl_week_rows(monkeypatch, tmp_path)

    assert list(rows.dtypes.astype(str).items()) == list(wc.CAPTURE_DTYPES.items())
    assert (rows["polled_at"] == _SCORED_AT).all()
    # Sunday night's kickoff hour, 00:00 UTC, comes 9 min 52.876544 s after the run.
    kickoff_hour = rows[
        (rows["game_id"] == "2026_03_DAL_PHI")
        & (rows["forecast_hour"] == pd.Timestamp("2026-09-28 00:00"))
    ]
    assert kickoff_hour["lead_hours"].item() == pytest.approx(592.876544 / 3600)


def test_a_slate_all_in_domes_asks_no_forecast_and_writes_a_row_per_game(monkeypatch, tmp_path):
    requested = _serve(monkeypatch, tmp_path, _SCORED_AT)
    domes = pd.DataFrame(
        [
            ("2026_04_MIN_DET", "2026-10-04", "13:00", "dome", "Ford Field"),
            ("2026_04_TB_NO", "2026-10-04", "13:00", "dome", "Mercedes-Benz Superdome"),
        ],
        columns=_NFL_COLUMNS,
    )

    wc.capture_weather({"NFL": SimpleNamespace(schedule=domes)}, _SCORED_AT)

    assert not requested
    rows = pd.read_parquet(tmp_path / "NFL/date=2026-09-27/235007.parquet")
    assert rows[["game_id", "venue", "roof"]].values.tolist() == [
        ["2026_04_MIN_DET", "Ford Field", "dome"],
        ["2026_04_TB_NO", "Caesars Superdome", "dome"],
    ]
    assert rows[["forecast_hour", *weather.HOURLY_VARS, "lead_hours"]].isna().all(axis=None)
    assert list(rows.dtypes.astype(str).items()) == list(wc.CAPTURE_DTYPES.items())


def test_mlb_asks_statsapi_from_the_day_before_the_runs_utc_date(monkeypatch, tmp_path):
    """An evening run is already the next day in UTC, while that night's West Coast game is not.

    A game is kept from just after the run up to a full week after it: neither the first pitch
    at the run nor the games past the horizon get rows, and Wrigley's series is one location.
    """
    requested = _serve(monkeypatch, tmp_path, _EVENING_RUN, days=_MLB_DAYS)

    wc.capture_weather({"MLB": None}, _EVENING_RUN)

    (schedule_url, schedule_params, timeout), (forecast_url, forecast_params, _) = requested
    assert (schedule_url, forecast_url, timeout) == (
        weather.MLB_SCHEDULE_URL,
        wc.FORECAST_URL,
        CAPTURE_TIMEOUT_S,
    )
    assert schedule_params == {
        **weather.MLB_SCHEDULE_PARAMS,
        "startDate": "2026-07-04",
        "endDate": "2026-07-12",
    }
    # Dodger Stadium, Wrigley Field and Petco Park.
    assert forecast_params["latitude"] == "34.0737,41.9482,32.7079"
    rows = pd.read_parquet(tmp_path / "MLB/date=2026-07-05/014000.parquet")
    assert rows["game_id"].unique().tolist() == ["103", "104", "105", "106"]


def test_a_week_without_a_game_asks_no_forecast_and_writes_nothing(monkeypatch, tmp_path, caplog):
    """statsapi's off-season reply lists no day; the NFL, absent from ``stats``, is skipped quietly."""
    requested = _serve(monkeypatch, tmp_path, _SCORED_AT)

    with caplog.at_level(logging.WARNING, logger="log"):
        wc.capture_weather({"MLB": None}, _SCORED_AT)

    assert [url for url, _, _ in requested] == [weather.MLB_SCHEDULE_URL]
    assert not _written(tmp_path)
    assert not caplog.records


@pytest.mark.parametrize(
    ("refused", "failed", "written"),
    [(weather.MLB_SCHEDULE_URL, "MLB", "NFL"), (wc.FORECAST_URL, "NFL", "MLB")],
)
def test_a_league_whose_capture_fails_costs_the_other_nothing(
    monkeypatch, tmp_path, caplog, refused, failed, written
):
    """A refused request is an ERROR in the log, never the run's exception.

    With Open-Meteo refusing, MLB's dome game still writes: it needs no forecast.
    """
    _serve(monkeypatch, tmp_path, _SCORED_AT, days=_DOME_DAYS, refused=refused)
    stats = {"NFL": SimpleNamespace(schedule=_NFL_WEEK), "MLB": None}

    with caplog.at_level(logging.ERROR, logger="log"):
        wc.capture_weather(stats, _SCORED_AT)

    assert [path.split("/")[0] for path in _written(tmp_path)] == [written]
    assert [record.getMessage() for record in caplog.records] == [
        f"{failed} weather capture not written"
    ]
