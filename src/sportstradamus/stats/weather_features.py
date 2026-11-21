"""NFL and MLB weather features: one pre-game forecast of the hours around each game's kickoff.

Training reads the archived forecasts (:mod:`sportstradamus.collectors.weather_history`), the one
made a day before where stored (every game from 2024), else the shortest-lead one (lead 0, all a
game before 2024 has). Serving reads the newest per-run capture
(:mod:`sportstradamus.collectors.weather_capture`), which holds every game of the coming week. A
retractable roof's state is unknown before kickoff, so its game is forecast like an open one and
flagged; the realized state (``schedule_label``) is a game-time fact and feeds nothing. A dome's
game is flagged and has no forecast: its weather and lead are NaN, as is every column of a game
neither source holds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sportstradamus.collectors.weather import VENUES
from sportstradamus.helpers.io import LEAGUES_DIR, WEATHER_CAPTURE_DIR

_SHARED_COLUMNS = (
    "weather wind",
    "weather gust",
    "weather temp",
    "weather precip",
    "weather snow",
    "weather humidity",
    "weather lead",
    "weather indoor",
    "weather retractable",
)
# MLB adds the wind's component toward centre field, from its venue table's field azimuths.
WEATHER_COLUMNS: dict[str, tuple[str, ...]] = {
    "NFL": _SHARED_COLUMNS,
    "MLB": (*_SHARED_COLUMNS, "weather wind out"),
}
# The lead the plan trains on: of the leads stored from 2024, the nearest a pre-game decision.
_TRAINING_LEAD_DAYS = 1
_MLB_AZIMUTH = pd.Series({venue: entry["azimuth"] for venue, entry in VENUES["MLB"].items()})


def history_weather(league: str) -> pd.DataFrame:
    """Each game's :data:`WEATHER_COLUMNS` in the league's stored forecast history.

    ``weather lead`` is the stored lead in days. The history is gitignored, so a fresh clone or
    CI has none.

    Returns:
        Floats indexed by game id (text); no rows when no history is stored.
    """
    try:
        history = pd.read_parquet(LEAGUES_DIR / league.lower() / "weather_history.parquet")
    except FileNotFoundError:
        return pd.DataFrame(columns=list(WEATHER_COLUMNS[league]), dtype=float)
    # The training lead sorts ahead of every other, the rest shortest first; a dome's lone row
    # has no lead and sorts last.
    ranked = history.sort_values(
        "lead_days", key=lambda lead: lead.replace(_TRAINING_LEAD_DAYS, -1)
    )
    picked = ranked.drop_duplicates("game_id")[["game_id", "lead_days"]]
    # A merge matches null keys too, so the dome's row keeps itself.
    return _game_weather(history.merge(picked, on=["game_id", "lead_days"]), league)


def forecast_weather(league: str) -> pd.DataFrame:
    """Each game's :data:`WEATHER_COLUMNS` in the league's newest weather capture.

    ``weather lead`` is the days from the capture's run to kickoff.

    Returns:
        Floats indexed by game id (text); no rows when no capture is stored.
    """
    captures = sorted((WEATHER_CAPTURE_DIR / league).glob("date=*/*.parquet"))
    if not captures:
        return pd.DataFrame(columns=list(WEATHER_COLUMNS[league]), dtype=float)
    capture = pd.read_parquet(captures[-1])
    lead_days = (capture["kickoff"] - capture["polled_at"]) / pd.Timedelta(days=1)
    return _game_weather(capture.assign(lead_days=lead_days), league)


def _game_weather(rows: pd.DataFrame, league: str) -> pd.DataFrame:
    """The league's :data:`WEATHER_COLUMNS` per game of ``rows``, one forecast's hours a game.

    Each row's ``lead_days`` is its game's lead.
    """
    games = rows.groupby("game_id")
    roof = games["roof"].first()
    weather = pd.DataFrame(
        {
            "weather wind": games["wind_speed_10m"].mean(),
            "weather gust": games["wind_gusts_10m"].max(),
            "weather temp": games["temperature_2m"].mean(),
            # min_count keeps a dome's absent forecast NaN rather than a measured 0.
            "weather precip": games["precipitation"].sum(min_count=1),
            "weather snow": games["snowfall"].sum(min_count=1),
            "weather humidity": games["relative_humidity_2m"].mean(),
            # No forecast, no lead: a capture still dates a dome's game from the run.
            "weather lead": games["lead_days"].first().where(roof != "dome"),
            "weather indoor": roof == "dome",
            "weather retractable": roof == "retractable",
        }
    )
    if league == "MLB":
        # Open-Meteo gives the direction the wind blows from; the azimuth runs from home plate
        # to centre field, so a positive component blows out. The neutral sites the venue table
        # gives no azimuth get NaN.
        toward = rows["wind_direction_10m"] + 180
        bearing = np.radians(toward - rows["venue"].map(_MLB_AZIMUTH))
        wind_out = rows["wind_speed_10m"] * np.cos(bearing)
        weather["weather wind out"] = wind_out.groupby(rows["game_id"]).mean()
    return weather.astype(float)
