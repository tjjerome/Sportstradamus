"""Venue table, Open-Meteo query and kickoff windows shared by the NFL and MLB weather collectors.

The per-run capture and the dev-side backfill both read games through :func:`nfl_games` and
:func:`mlb_games`, query Open-Meteo with :func:`request_params`, flatten the reply with
:func:`hourly_frame` and store :func:`window_rows`. Every venue sits in
``data/config/venues.json`` with its coordinates and roof class (MLB adds the field azimuth, home
plate to centre field). NFL games key on nflverse's ``stadium`` name through the table's aliases,
never ``stadium_id`` (JAX00 also labels a London game); MLB games on statsapi's venue id, never
the home team (one 2026 Red Sox home game was at Tropicana). A dome gets one row and no forecast;
a venue the table lacks gets no row and a warning, so a new neutral site shows in the run log.

``MODEL`` is pinned so live and backfill rows come from one forecast source. Probed 2026-11-12 at
Lambeau and Wrigley: ``gfs_global`` has the longest history among models not retiring, the
Previous Runs archive at a 4-day lead from 2024-01-22 (temperature, wind, direction, gusts,
precipitation, humidity) and the Historical Forecast from 2021-06 (all of 2022-23).
``gfs_seamless`` blends HRRR into the first ~48 h of a live US forecast, and HRRR retires
2026-11-30; ICON Global is discontinued 2026-11-30 and its Historical Forecast starts 2022-11-16;
``ecmwf_ifs025`` archives no gusts, precipitation probability or snowfall, and nothing before
2024. ``gfs_global``'s archive never holds ``precipitation_probability`` (live forecast only) and
holds snowfall from 2024-12-05.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import json
from collections.abc import Sequence

import pandas as pd

from sportstradamus import data
from sportstradamus.spiderLogger import logger

MODEL = "gfs_global"
HOURLY_VARS: tuple[str, ...] = (
    "temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "precipitation",
    "precipitation_probability",
    "relative_humidity_2m",
    "snowfall",
)
# Column order and dtype of the rows both collectors store, pinned so every file reads together:
# an all-null column left to pyarrow is typed ``null`` (a dome's weather, or the archive's
# precipitation_probability).
WEATHER_DTYPES: dict[str, str] = {
    "game_id": "string",
    "venue": "string",
    "roof": "string",
    "schedule_label": "string",
    "kickoff": "datetime64[ns]",
    "forecast_hour": "datetime64[ns]",
    **dict.fromkeys(HOURLY_VARS, "float64"),
    "model": "string",
}
# Forecast hours stored per game around its kickoff hour: from the hour before (warm-ups and the
# first pitch or kick) to four after (an NFL game runs about 3 h 10 min and an MLB game about
# 2 h 40 min, so four hours holds overtime, extra innings or a short delay).
HOURS_BEFORE_KICKOFF = 1
HOURS_AFTER_KICKOFF = 4
# statsapi's schedule query for mlb_games: regular season and every postseason round, with the
# game's weather label, cut to the fields mlb_games reads. Callers add the dates or the season.
MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
MLB_SCHEDULE_PARAMS = {
    "sportId": 1,
    "gameTypes": "R,F,D,L,W",
    "hydrate": "weather",
    "fields": "dates,games,gamePk,gameDate,status,detailedState,startTimeTBD,venue,id,weather,"
    "condition",
}
# statsapi states of a game that will not be played at its listed time.
_UNPLAYED_STATES = ("Postponed", "Cancelled")
# nflverse gives gametime in US Eastern.
_NFL_TZ = "America/New_York"
_GAME_COLUMNS = ["game_id", "venue", "roof", "lat", "lon", "kickoff", "schedule_label"]

with (pkg_resources.files(data) / "config" / "venues.json").open() as infile:
    VENUES: dict[str, dict[str, dict]] = json.load(infile)
# nflverse spells some venues by a former or non-sponsor name: each such spelling to its entry.
_NFL_ALIASES = {alias: name for name, entry in VENUES["NFL"].items() for alias in entry["aliases"]}


def request_params(venues: pd.DataFrame, variables: Sequence[str]) -> dict[str, str]:
    """The Open-Meteo query every endpoint shares, one location per ``venues`` row in order.

    Callers add ``forecast_days`` or ``start_date`` / ``end_date``. Open-Meteo answers the
    locations in request order, which :func:`hourly_frame` relies on.
    """
    return {
        "latitude": ",".join(venues["lat"].astype(str)),
        "longitude": ",".join(venues["lon"].astype(str)),
        "hourly": ",".join(variables),
        "models": MODEL,
        "timezone": "UTC",
        "wind_speed_unit": "mph",
        "temperature_unit": "fahrenheit",
    }


def _locate(games: pd.DataFrame, league: str) -> pd.DataFrame:
    """Each game with its venue's coordinates and roof; a venue the table lacks drops its games."""
    table = pd.DataFrame.from_dict(VENUES[league], orient="index")[["lat", "lon", "roof"]]
    known = games["venue"].isin(table.index)
    for venue in games.loc[~known, "venue"].unique():
        logger.warning(f"{league} weather: no venue entry for {venue!r}; its games get no row")
    return games[known].join(table, on="venue")[_GAME_COLUMNS].reset_index(drop=True)


def nfl_games(schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per NFL game with a kickoff time, at its venue.

    ``schedule`` is the nflverse frame ``nfl_data_py.import_schedules`` returns, ``gametime`` in
    US Eastern; a game without one (TBD) gets no row. ``schedule_label`` is nflverse's ``roof``,
    wrong at some neutral sites and blank at a retractable until the game is played, so the
    table's ``roof`` decides.
    """
    # A failed fetch leaves StatsNFL._fetch_schedule's fallback, which has no gametime or stadium:
    # read it as a schedule of no games, so the result carries the dtypes of a full one.
    if schedule.empty:
        schedule = pd.DataFrame(columns=["game_id", "gameday", "gametime", "roof", "stadium"])
    timed = schedule.dropna(subset=["gametime"])
    kickoff = pd.to_datetime(timed["gameday"] + " " + timed["gametime"])
    games = pd.DataFrame(
        {
            "game_id": timed["game_id"],
            "venue": timed["stadium"].replace(_NFL_ALIASES),
            "kickoff": kickoff.dt.tz_localize(_NFL_TZ).dt.tz_convert("UTC").dt.tz_localize(None),
            "schedule_label": timed["roof"],
        }
    )
    return _locate(games, "NFL")


def mlb_games(payload: dict) -> pd.DataFrame:
    """One row per MLB game of a statsapi schedule payload, at its venue.

    A postponed or cancelled game gets no row, nor one whose start time is TBD (a traditional
    doubleheader's second game, whose ``gameDate`` is a placeholder minutes after the first
    game's). A ``gamePk`` listed twice (a suspended game, on the date it began and the date it
    resumes) keeps its latest ``gameDate``. ``schedule_label`` is statsapi's game condition
    ("Roof Closed", "Dome", "Clear", ...), NA until posted.
    """
    games = pd.DataFrame(
        [
            {
                "game_id": str(game["gamePk"]),
                "venue": str(game["venue"]["id"]),
                "kickoff": game["gameDate"],
                "schedule_label": game.get("weather", {}).get("condition"),
            }
            for day in payload["dates"]
            for game in day["games"]
            if game["status"]["detailedState"] not in _UNPLAYED_STATES
            and not game["status"]["startTimeTBD"]
        ],
        columns=["game_id", "venue", "kickoff", "schedule_label"],
    )
    games["kickoff"] = pd.to_datetime(games["kickoff"], utc=True).dt.tz_localize(None)
    games = games.sort_values("kickoff").drop_duplicates("game_id", keep="last")
    return _locate(games, "MLB")


def hourly_frame(reply: dict | list[dict], venues: Sequence[str]) -> pd.DataFrame:
    """One row per (venue, forecast hour) of an Open-Meteo reply, each hourly variable a column.

    Open-Meteo answers several locations with a JSON array in request order and one location
    with a bare object; ``venues`` keys each location in that order. ``hourly.time`` is naive UTC,
    as :func:`request_params` asks. Variable names stay as given: a Previous Runs reply's carry
    their ``_previous_dayN`` suffix.
    """
    locations = reply if isinstance(reply, list) else [reply]
    hourly = pd.concat(
        [
            pd.DataFrame({"venue": venue, **location["hourly"]})
            for venue, location in zip(venues, locations, strict=True)
        ],
        ignore_index=True,
    ).rename(columns={"time": "forecast_hour"})
    hourly["forecast_hour"] = pd.to_datetime(hourly["forecast_hour"])
    return hourly


def window_rows(games: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    """Each game's forecast hours around kickoff, joined to its venue's hourly forecast.

    A game outside a dome gets the hours from ``HOURS_BEFORE_KICKOFF`` before its kickoff hour to
    ``HOURS_AFTER_KICKOFF`` after; an hour past the forecast's horizon keeps its row, weather
    NaN. A dome game gets one row, ``forecast_hour`` NaT and weather NaN: no forecast is
    requested for a dome. Dtypes are left as built; callers pin them with ``WEATHER_DTYPES``.
    """
    dome = games["roof"] == "dome"
    offsets = pd.DataFrame(
        {"offset": pd.to_timedelta(range(-HOURS_BEFORE_KICKOFF, HOURS_AFTER_KICKOFF + 1), unit="h")}
    )
    windows = games[~dome].merge(offsets, how="cross")
    windows["forecast_hour"] = windows["kickoff"].dt.floor("h") + windows.pop("offset")
    rows = pd.concat([windows, games[dome].assign(forecast_hour=pd.NaT)], ignore_index=True)
    rows = rows.merge(hourly, on=["venue", "forecast_hour"], how="left").assign(model=MODEL)
    game = ["game_id", "venue", "roof", "schedule_label", "kickoff"]
    weather = hourly.columns.drop(["venue", "forecast_hour"]).tolist()
    return rows[[*game, "forecast_hour", *weather, "model"]]
