"""Archived pre-game forecasts of NFL and MLB games already played, to train weather features on.

``sportstradamus fetch weather-history``, a dev-side backfill run by hand, stores for each game
that kicked off two days ago or more outside a dome the forecast that existed before it: the
per-run capture's rows (:mod:`weather`) plus ``lead_days`` and ``source``, one file per league at
``data/leagues/<league>/weather_history.parquet``.

Seasons from 2024 read Open-Meteo's Previous Runs archive, each hour as forecast 0 to 7 days
before it (``lead_days``; every lead from 2024-01-25); 2022 and 2023 the Historical Forecast,
lead 0 only. Neither keeps ``precipitation_probability`` (stored NaN); ``snowfall`` is NaN before
2024-12-05. ``schedule_label`` is a game-time fact here (nflverse's roof after the game,
statsapi's game-time condition) and must not feed a decision-time feature.

A run paces itself under Open-Meteo's hourly limit, stops once it has spent ``--max-cost`` calls
(a free IP gets 10,000 a day) and on a 429 writes what it holds and exits. The file is the
record: a rerun requests only the games it lacks, so rerun daily until ``--dry-run`` plans nothing.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import time
from datetime import date, timedelta
from http import HTTPStatus
from pathlib import Path
from typing import NamedTuple

import click
import nfl_data_py as nfl
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from sportstradamus import data
from sportstradamus.collectors import weather
from sportstradamus.helpers.io import _atomic_write_parquet
from sportstradamus.helpers.scraping import REQUEST_TIMEOUT_S

_LEAGUES = ("NFL", "MLB")
# The first season the Historical Forecast archive covers whole (it starts 2021-06); the injury
# history starts there too, so both cover the same games.
_FIRST_SEASON = 2022
# Seasons from this one read the Previous Runs archive; earlier ones the Historical Forecast.
_PREVIOUS_RUNS_FROM = 2024
# The Previous Runs archive keeps each hour as forecast up to seven days before it.
_MAX_LEAD_DAYS = 7
# No archive keeps precipitation_probability: only the live forecast has it.
_ARCHIVED_VARS = [var for var in weather.HOURLY_VARS if var != "precipitation_probability"]
# lead_days -> {Previous Runs column: variable}; lead 0 is the unsuffixed column.
_LEAD_COLUMNS = {
    lead: {f"{var}_previous_day{lead}" if lead else var: var for var in _ARCHIVED_VARS}
    for lead in range(_MAX_LEAD_DAYS + 1)
}
_URLS = {
    "previous_runs": "https://previous-runs-api.open-meteo.com/v1/forecast",
    "historical_forecast": "https://historical-forecast-api.open-meteo.com/v1/forecast",
}
_VARIABLES = {
    "previous_runs": [column for columns in _LEAD_COLUMNS.values() for column in columns],
    "historical_forecast": _ARCHIVED_VARS,
}
_DTYPES = {**weather.WEATHER_DTYPES, "lead_days": "Int64", "source": "string"}
# Open-Meteo's archives fill an hour in within about two days.
_SETTLE = pd.Timedelta(days=2)
# Open-Meteo bills a location max(1, variables / 10 * max(1, days / 14)) calls: any span up to
# 14 days bills as 14 (calculateQueryWeight in its ForecastApiResult.swift; 56 variables over
# 3 days billed 5.6).
_BILLED_VARIABLES = 10
_BILLED_DAYS = 14
# The free tier allows 10,000 calls a day per IP; the production box on the same network spends
# ~650 a day on the per-run capture.
_DEFAULT_MAX_COST = 8_000.0
# Seconds to wait per call billed: 4,500 an hour, under Open-Meteo's 5,000-per-hour limit.
_SECONDS_PER_CALL = 3600 / 4_500
# Requests between file writes: a crash loses at most this many.
_WRITE_EVERY = 25
_LEAGUES_DIR = pkg_resources.files(data) / "leagues"


class _Request(NamedTuple):
    """One Open-Meteo request: a venue's games whose windows fall in [start, end], UTC dates."""

    league: str
    season: int
    source: str
    start: date
    end: date
    games: pd.DataFrame
    cost: float


def _history_path(league: str) -> Path:
    return _LEAGUES_DIR / league.lower() / "weather_history.parquet"


def _cost(variables: int, start: date, end: date) -> float:
    days = (end - start).days + 1
    return max(1, variables / _BILLED_VARIABLES * max(1, days / _BILLED_DAYS))


def _spans(days: list[date], variables: int) -> list[list[date]]:
    """Inclusive date spans covering ``days``, one request each.

    Consecutive days form runs; a run joins the span before it, gap and all, while one request
    for both bills no more than two.
    """
    runs: list[list[date]] = []
    for day in sorted(set(days)):
        if runs and day - runs[-1][1] == timedelta(days=1):
            runs[-1][1] = day
        else:
            runs.append([day, day])
    spans = runs[:1]
    for start, end in runs[1:]:
        joined = _cost(variables, spans[-1][0], end)
        if joined <= _cost(variables, *spans[-1]) + _cost(variables, start, end):
            spans[-1][1] = end
        else:
            spans.append([start, end])
    return spans


def _games(league: str, seasons: list[int]) -> pd.DataFrame:
    """The league's games of ``seasons`` played ``_SETTLE`` ago or more outside a dome.

    Each carries its season, its archive and the first and last UTC dates its window touches.
    """
    if league == "NFL":
        schedule = nfl.import_schedules(seasons)
        by_season = {
            season: weather.nfl_games(schedule[schedule["season"] == season]) for season in seasons
        }
    else:
        by_season = {}
        for season in seasons:
            response = requests.get(
                weather.MLB_SCHEDULE_URL,
                params={**weather.MLB_SCHEDULE_PARAMS, "season": season},
                timeout=REQUEST_TIMEOUT_S,
            )
            response.raise_for_status()
            by_season[season] = weather.mlb_games(response.json())
    games = pd.concat(
        [games.assign(season=season) for season, games in by_season.items()], ignore_index=True
    )
    settled = pd.Timestamp.now("UTC").tz_localize(None) - _SETTLE
    games = games[(games["kickoff"] <= settled) & (games["roof"] != "dome")]
    previous_runs = games["season"] >= _PREVIOUS_RUNS_FROM
    return games.assign(
        source=np.where(previous_runs, "previous_runs", "historical_forecast"),
        first_day=(games["kickoff"] - pd.Timedelta(hours=weather.HOURS_BEFORE_KICKOFF)).dt.date,
        last_day=(games["kickoff"] + pd.Timedelta(hours=weather.HOURS_AFTER_KICKOFF)).dt.date,
    )


def _plan(league: str, seasons: list[int], stored: pd.DataFrame) -> list[_Request]:
    """The requests for every game without stored rows from its source, per season and venue."""
    games = _games(league, seasons)
    done = set(zip(stored["game_id"], stored["source"], strict=True))
    missing = games.loc[
        [key not in done for key in zip(games["game_id"], games["source"], strict=True)]
    ]
    plan = []
    for (season, source, _), venue_games in missing.groupby(["season", "source", "venue"]):
        variables = len(_VARIABLES[source])
        for start, end in _spans([*venue_games["first_day"], *venue_games["last_day"]], variables):
            inside = venue_games[venue_games["first_day"].between(start, end)]
            cost = _cost(variables, start, end)
            plan.append(_Request(league, season, source, start, end, inside, cost))
    return plan


def _by_lead(hourly: pd.DataFrame) -> pd.DataFrame:
    """A Previous Runs reply long: one row per (venue, forecast_hour, lead_days)."""
    leads = [
        hourly[["venue", "forecast_hour", *cols]].rename(columns=cols).assign(lead_days=lead)
        for lead, cols in _LEAD_COLUMNS.items()
    ]
    return pd.concat(leads, ignore_index=True)


def _fetch(request: _Request) -> tuple[pd.DataFrame, float]:
    """The request's rows, typed like the file's, and the calls Open-Meteo billed for it."""
    venue = request.games.iloc[:1]
    response = requests.get(
        _URLS[request.source],
        params={
            **weather.request_params(venue, _VARIABLES[request.source]),
            "start_date": str(request.start),
            "end_date": str(request.end),
        },
        timeout=REQUEST_TIMEOUT_S,
    )
    if response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
        raise click.ClickException(
            f"Open-Meteo's rate limit is spent ({response.text}); rows so far are written, "
            "rerun later"
        )
    response.raise_for_status()
    hourly = weather.hourly_frame(response.json(), venue["venue"].tolist())
    hourly = _by_lead(hourly) if request.source == "previous_runs" else hourly.assign(lead_days=0)
    rows = weather.window_rows(request.games, hourly).assign(
        precipitation_probability=np.nan, source=request.source
    )
    billed = float(response.headers.get("X-RateLimit-Cost", request.cost))
    return rows[list(_DTYPES)].astype(_DTYPES), billed


def _stored(league: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(_history_path(league))
    except FileNotFoundError:
        return pd.DataFrame(columns=list(_DTYPES)).astype(_DTYPES)


def _flush(stored: dict[str, pd.DataFrame], held: dict[str, list[pd.DataFrame]]) -> None:
    """Append each league's held rows to its file and let go of them."""
    for league, frames in held.items():
        if frames:
            stored[league] = pd.concat([stored[league], *frames], ignore_index=True)
            _atomic_write_parquet(stored[league], _history_path(league))
            frames.clear()


def _echo_plan(plan: list[_Request]) -> None:
    summary = pd.DataFrame(
        [(r.league, r.season, r.source, len(r.games), r.cost) for r in plan],
        columns=["league", "season", "source", "games", "cost"],
    )
    buckets = summary.groupby(["league", "season", "source"], sort=False)
    for key, bucket in [*buckets, (("total",), summary)]:
        click.echo(
            f"{' '.join(map(str, key))}: {bucket['games'].sum()} games, {len(bucket)} requests, "
            f"cost {bucket['cost'].sum():.1f}"
        )


@click.command()
@click.option(
    "--league",
    "leagues",
    multiple=True,
    type=click.Choice(_LEAGUES, case_sensitive=False),
    help="League to backfill; repeatable. Defaults to both.",
)
@click.option(
    "--season",
    "seasons",
    multiple=True,
    type=int,
    help=f"Season to backfill; repeatable. Defaults to {_FIRST_SEASON} through the current year.",
)
@click.option(
    "--max-cost",
    default=_DEFAULT_MAX_COST,
    show_default=True,
    help="Stop once the run has spent this many Open-Meteo calls.",
)
@click.option(
    "--dry-run", is_flag=True, help="Print the plan and its cost; request nothing from Open-Meteo."
)
def weather_history(
    leagues: tuple[str, ...], seasons: tuple[int, ...], max_cost: float, dry_run: bool
) -> None:
    """Store the archived pre-game forecast of NFL and MLB games already played."""
    seasons = list(seasons or range(_FIRST_SEASON, date.today().year + 1))
    stored = {league: _stored(league) for league in leagues or _LEAGUES}
    plan = [request for league in stored for request in _plan(league, seasons, stored[league])]
    if dry_run:
        _echo_plan(plan)
        return
    held: dict[str, list[pd.DataFrame]] = {league: [] for league in stored}
    added = dict.fromkeys(stored, 0)
    made, spent, stopped = 0, 0.0, ""
    try:
        for request in tqdm(plan, desc="weather history", unit="request"):
            if spent >= max_cost:
                stopped = f"; stopped at --max-cost {max_cost:g}, rerun tomorrow"
                break
            rows, cost = _fetch(request)
            held[request.league].append(rows)
            added[request.league] += len(rows)
            made, spent = made + 1, spent + cost
            if made % _WRITE_EVERY == 0:
                _flush(stored, held)
            time.sleep(cost * _SECONDS_PER_CALL)
    finally:
        _flush(stored, held)
    counts = " ".join(f"{league}={count}" for league, count in added.items())
    click.echo(f"weather-history: {made} requests, cost {spent:.1f}, new rows {counts}{stopped}")
