"""Each game's start time, stored per league, for the training read of its book quotes.

A training row reads the sportsbook quotes as of its game's start minus ``TRAINING_LOOKBACK``
(:func:`read_times`), the last board before the game. The starts live in
``data/leagues/<league>/game_starts.parquet``: ``game_id``, the gamelog's game id as text, and
``start``, naive UTC. A lookup fetches only the games the store lacks, a whole season (NBA,
WNBA, NFL) or the weeks and date span (NHL, MLB) holding them, and stores every game asked
for, NaT where the schedule gives no start, so no game is fetched twice. Sources:

- NBA, WNBA: ``ScheduleLeagueV2`` (:func:`nba_client.schedule_games`), ``gameDateTimeUTC``.
- NHL: api-web's weekly schedule, ``startTimeUTC``.
- MLB: statsapi's schedule, read as the weather capture reads it
  (:func:`collectors.weather.mlb_schedule_games`): no start for a TBD game.
- NFL: nflverse's schedule (:func:`collectors.weather.nfl_kickoffs`).

A failed fetch raises: a row cached on the stand-in read would keep it.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
from collections.abc import Callable
from datetime import date, timedelta

import nfl_data_py as nfl
import pandas as pd
import requests

from sportstradamus import data
from sportstradamus.collectors.weather import (
    MLB_SCHEDULE_PARAMS,
    MLB_SCHEDULE_URL,
    mlb_schedule_games,
    nfl_kickoffs,
)
from sportstradamus.helpers import Scrape
from sportstradamus.helpers.archive import TRAINING_LOOKBACK
from sportstradamus.helpers.io import _atomic_write_parquet
from sportstradamus.helpers.scraping import REQUEST_TIMEOUT_S
from sportstradamus.spiderLogger import logger
from sportstradamus.stats import nba_client

# The read of a row whose game has no stored start: 12:00 UTC on its game date, every training
# row's read before starts were stored, ahead of every regular US start.
UNTIMED_READ = timedelta(hours=12)
NHL_SCHEDULE_URL = "https://api-web.nhle.com/v1/schedule/{day}"
_LEAGUES_DIR = pkg_resources.files(data) / "leagues"

scraper = Scrape()


def read_times(league: str, game_ids: pd.Series, dates: pd.Series) -> pd.Series:
    """Each training row's quote read: its game's start minus ``TRAINING_LOOKBACK``.

    The rule ``Stats.get_training_matrix`` builds rows on and
    ``scripts/inject_backfilled_odds`` re-resolves cached rows on. A row whose game id is NaN,
    or whose game the store holds no start for, reads at ``UNTIMED_READ`` on its game date;
    one warning counts those rows.

    Args:
        league: The league whose store holds the games.
        game_ids: Each row's game id as the league's gamelog spells it.
        dates: Each row's game date, in the same row order as ``game_ids``.

    Returns:
        Naive UTC read times, one per row in order, indexed like ``game_ids``.
    """
    # Positional: a gamelog an update concatenated can repeat index labels.
    rows = pd.DataFrame({"game_id": game_ids.to_numpy(), "date": pd.to_datetime(dates).to_numpy()})
    known = rows.dropna(subset=["game_id"])
    games = pd.DataFrame({"game_id": known["game_id"].astype(str), "date": known["date"].dt.date})
    starts = (
        games["game_id"]
        .map(game_starts(league, games))
        .reindex(rows.index)
        .astype("datetime64[ns]")
    )
    untimed = starts.isna()
    if untimed.any():
        logger.warning(
            f"{league}: {untimed.sum()} of {len(rows)} training rows have no stored game start; "
            "they read at 12:00 UTC on their game date"
        )
    reads = (starts - TRAINING_LOOKBACK).fillna(rows["date"] + UNTIMED_READ)
    return reads.set_axis(game_ids.index)


def game_starts(league: str, games: pd.DataFrame) -> pd.Series:
    """The start of each stored game, after fetching the ones ``games`` names that it lacks.

    Args:
        league: The league whose store to read.
        games: ``game_id`` (text) and ``date`` (its game date, a ``datetime.date``).

    Returns:
        Every stored game's start, naive UTC, indexed by game id; NaT where the schedule
        gives none.
    """
    path = _LEAGUES_DIR / league.lower() / "game_starts.parquet"
    try:
        stored = pd.read_parquet(path)
    except FileNotFoundError:
        stored = pd.DataFrame(
            {"game_id": pd.Series(dtype=str), "start": pd.Series(dtype="datetime64[ns]")}
        )
    missing = games.loc[~games["game_id"].isin(stored["game_id"])].drop_duplicates("game_id")
    if not missing.empty:
        fetched = pd.to_datetime(_SOURCES[league](league, missing), utc=True).dt.tz_localize(None)
        added = pd.DataFrame(
            {"game_id": missing["game_id"], "start": missing["game_id"].map(fetched)}
        )
        stored = pd.concat([stored, added.astype({"start": "datetime64[ns]"})], ignore_index=True)
        _atomic_write_parquet(stored, path)
    return stored.set_index("game_id")["start"]


def _nba_starts(league: str, missing: pd.DataFrame) -> pd.Series:
    """Every game of each season holding a missing game, from ``ScheduleLeagueV2``."""
    # A stats.nba.com game id carries its season's opening year in characters 3-4: 0022400061
    # opened in 2024. The endpoint names an NBA season "2024-25", a WNBA one by its year.
    opening_years = sorted({2000 + int(game_id[3:5]) for game_id in missing["game_id"]})
    seasons = [
        f"{year}-{(year + 1) % 100:02d}" if league == "NBA" else year for year in opening_years
    ]
    games = [game for season in seasons for game in nba_client.schedule_games(league, season)]
    return pd.Series({game["gameId"]: game["gameDateTimeUTC"] for game in games}, dtype=object)


def _nhl_starts(_league: str, missing: pd.DataFrame) -> pd.Series:
    """Every game of each api-web schedule week holding a missing game's date."""
    starts, covered = {}, set()
    for day in sorted(set(missing["date"])):
        if day in covered:
            continue
        week = scraper.get(NHL_SCHEDULE_URL.format(day=f"{day:%Y-%m-%d}")).get("gameWeek")
        # Scrape.get answers {} once its retries run out; a week asked for from a game's own date
        # always lists that game.
        if not week:
            raise RuntimeError(f"NHL schedule for the week of {day} unavailable")
        covered.update(date.fromisoformat(game_day["date"]) for game_day in week)
        starts.update(
            {
                str(game["id"]): game["startTimeUTC"]
                for game_day in week
                for game in game_day["games"]
            }
        )
    return pd.Series(starts, dtype=object)


def _mlb_starts(_league: str, missing: pd.DataFrame) -> pd.Series:
    """Every game statsapi lists from the first missing game's date to the last one's."""
    response = requests.get(
        MLB_SCHEDULE_URL,
        params={
            **MLB_SCHEDULE_PARAMS,
            "startDate": f"{missing['date'].min():%Y-%m-%d}",
            "endDate": f"{missing['date'].max():%Y-%m-%d}",
        },
        timeout=REQUEST_TIMEOUT_S,
    )
    response.raise_for_status()
    return mlb_schedule_games(response.json()).set_index("game_id")["kickoff"]


def _nfl_starts(_league: str, missing: pd.DataFrame) -> pd.Series:
    """Every game of each season holding a missing game, from nflverse's schedule."""
    # An nflverse game id leads with its season: 2026_10_MIA_IND.
    schedule = nfl.import_schedules(sorted({int(game_id[:4]) for game_id in missing["game_id"]}))
    return nfl_kickoffs(schedule).set_axis(schedule["game_id"])


# Each fetch answers the starts of every game it read, indexed by game id: naive UTC or ISO text.
_SOURCES: dict[str, Callable[[str, pd.DataFrame], pd.Series]] = {
    "NBA": _nba_starts,
    "WNBA": _nba_starts,
    "NHL": _nhl_starts,
    "MLB": _mlb_starts,
    "NFL": _nfl_starts,
}
