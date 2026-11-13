"""Per-run capture of the weather forecast around every NFL and MLB kickoff of the coming week.

Each ``prophecize`` run with the NFL or MLB live stores, for every game kicking off within
``HORIZON``, Open-Meteo's forecast for the hours around kickoff, the roof class from the venue
table, the schedule's own label and each hour's lead from the run, at
``WEATHER_CAPTURE_DIR/<league>/date=YYYY-MM-DD/HHMMSS.parquet``, named by the run's ``Scored At``
(UTC). A retractable roof's state is unknown before kickoff, so it is forecast like an open one;
a dome gets its one row and no forecast. Budget: one Open-Meteo request per league, billed about
one call per venue, and for MLB one statsapi schedule request (the NFL's schedule comes with its
update). Nothing reads these files yet; the model build will. The pinned model and what its
archive lacks: :mod:`sportstradamus.collectors.weather`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import requests

from sportstradamus.collectors.weather import (
    HOURLY_VARS,
    MLB_SCHEDULE_PARAMS,
    MLB_SCHEDULE_URL,
    WEATHER_DTYPES,
    hourly_frame,
    mlb_games,
    nfl_games,
    request_params,
    window_rows,
)
from sportstradamus.helpers.io import WEATHER_CAPTURE_DIR, write_capture_parquet
from sportstradamus.helpers.scraping import CAPTURE_TIMEOUT_S
from sportstradamus.spiderLogger import logger

if TYPE_CHECKING:
    from sportstradamus.stats import Stats

# The outdoor leagues: the NBA, WNBA and NHL play indoors.
WEATHER_LEAGUES = ("NFL", "MLB")
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
# The owner asked for at least four days ahead; seven covers every game of the coming NFL week.
HORIZON = pd.Timedelta(days=7)
# Forecast hours start at 00:00 UTC today, so the window of a game kicking off a full HORIZON
# after a late run ends inside day nine. Nine days of eight variables still bill one call per
# location: Open-Meteo bills locations x max(1, variables / 10 x max(1, days / 14)).
FORECAST_DAYS = 9
# The shared rows between the run's stamp and each hour's lead from it (NaN for a dome's row),
# pinned so a league's files read together.
CAPTURE_DTYPES = {"polled_at": "datetime64[ns]", **WEATHER_DTYPES, "lead_hours": "float64"}


def _mlb_schedule(scored_at: pd.Timestamp) -> pd.DataFrame:
    # statsapi files a game under its US official date, and a US evening game's official date is
    # the UTC date before its kickoff's: from the run's own UTC date, an evening run would miss
    # that night's West Coast games.
    response = requests.get(
        MLB_SCHEDULE_URL,
        params={
            **MLB_SCHEDULE_PARAMS,
            "startDate": (scored_at - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "endDate": (scored_at + HORIZON).strftime("%Y-%m-%d"),
        },
        timeout=CAPTURE_TIMEOUT_S,
    )
    response.raise_for_status()
    return mlb_games(response.json())


def _forecast(venues: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Each venue's hourly forecast from one request, and the calls Open-Meteo billed for it.

    No venue to ask for (no game ahead, or every one in a dome) makes no request and no rows,
    typed like a reply's: with no game either, pandas refuses :func:`window_rows`' merge on an
    untyped ``forecast_hour``.
    """
    if venues.empty:
        columns = ["venue", "forecast_hour", *HOURLY_VARS]
        hourly = pd.DataFrame(columns=columns).astype({col: WEATHER_DTYPES[col] for col in columns})
        return hourly, None
    response = requests.get(
        FORECAST_URL,
        params={**request_params(venues, HOURLY_VARS), "forecast_days": FORECAST_DAYS},
        timeout=CAPTURE_TIMEOUT_S,
    )
    response.raise_for_status()
    hourly = hourly_frame(response.json(), venues["venue"].tolist())
    return hourly, response.headers.get("X-RateLimit-Cost")


def _league_rows(league: str, stats: dict[str, Stats], scored_at: pd.Timestamp) -> pd.DataFrame:
    games = nfl_games(stats["NFL"].schedule) if league == "NFL" else _mlb_schedule(scored_at)
    games = games[games["kickoff"].between(scored_at, scored_at + HORIZON, inclusive="right")]
    venues = games.loc[games["roof"] != "dome", ["venue", "lat", "lon"]].drop_duplicates("venue")
    hourly, cost = _forecast(venues)
    logger.info(
        f"{league} weather: {len(games)} games, {len(venues)} venues requested, "
        f"Open-Meteo cost {cost}"
    )
    rows = window_rows(games, hourly)
    lead_hours = (rows["forecast_hour"] - scored_at) / pd.Timedelta(hours=1)
    rows = rows.assign(polled_at=scored_at, lead_hours=lead_hours)
    return rows[list(CAPTURE_DTYPES)].astype(CAPTURE_DTYPES)


def capture_weather(stats: dict[str, Stats], scored_at: pd.Timestamp) -> None:
    """Write the forecast around each live NFL and MLB game's kickoff, one parquet per league.

    A league absent from ``stats`` (out of season, or its update failed) or with no game within
    ``HORIZON`` writes nothing.
    """
    for league in WEATHER_LEAGUES:
        if league not in stats:
            continue
        # Diagnostics: nothing that goes wrong in a capture may cost the board, or the other
        # league, its run.
        try:
            rows = _league_rows(league, stats, scored_at)
            if not rows.empty:
                write_capture_parquet(rows, WEATHER_CAPTURE_DIR / league, scored_at)
        except Exception:
            logger.exception(f"{league} weather capture not written")
