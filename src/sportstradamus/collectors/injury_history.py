"""Injury history for training: one parquet per (league, season) under ``player_data/``.

A dev-side pull run by hand beside the weekly ``meditate``; the model-build lane derives
injury features from these files. Every run re-pulls the seasons it covers and overwrites
their files, since the current season changes week to week.

NFL is nflverse's weekly injury report with practice statuses, one row per (season, week,
team, gsis_id), written as delivered. The 2022-2024 files carry ``date_modified`` (UTC; a
median 47 h before kickoff) and later ones only the week. nflverse overwrites a week's rows
as the report updates, so only each player's final status for the week survives.

MLB is the statsapi transactions feed cut to Status Change rows that name the injured list
("disabled list" before 2019), each parsed into its action, list length and injury. An
activation from another list (paternity, bereavement, restricted) drops, and so does one
naming no list at all ("New York Yankees activated SS Braden Shewmake."; 517 in 2025),
since nothing says which list it ended.
"""

from __future__ import annotations

import re
from datetime import date

import click
import nfl_data_py as nfl
import nflreadpy as nflr
import pandas as pd
import requests
from tqdm import tqdm

from sportstradamus.collectors.tabular import PLAYER_DATA_BASE
from sportstradamus.helpers.io import _atomic_write_parquet
from sportstradamus.helpers.scraping import REQUEST_TIMEOUT_S

_LEAGUES = ("NFL", "MLB")

# Oldest season pulled by default; the weather backfill starts there too, so both histories
# cover the same games.
_FIRST_SEASON = 2022

# One request per calendar year (~17,000 rows). sportId 1 is the major leagues.
_MLB_TRANSACTIONS_URL = (
    "https://statsapi.mlb.com/api/v1/transactions"
    "?startDate={season}-01-01&endDate={season}-12-31&sportId=1"
)

# A transfer names two lists; the lazy gap settles on the one that ends the sentence, which
# is the list moved to. A placement's "retroactive to <date>" sits between list and injury.
_IL_MOVE = re.compile(
    r"\b(?P<action>placed|transferred|activated)\b.*?"
    r"\b(?P<list_days>\d+)-day (?:injured|disabled) list"
    r"(?: retroactive to [^.]+)?\.\s*(?P<injury>.+)?"
)

# statsapi field -> output column, in output order.
_MLB_COLUMNS = {
    "id": "id",
    "date": "date",
    "effectiveDate": "effective_date",
    "resolutionDate": "resolution_date",
    "person.id": "player_id",
    "person.fullName": "player_name",
    "toTeam.id": "team_id",
    "toTeam.name": "team",
    "description": "description",
}


def _mlb_injuries(season: int) -> pd.DataFrame:
    """One season of injured-list moves; ``effective_date`` already carries a retroactive date."""
    response = requests.get(_MLB_TRANSACTIONS_URL.format(season=season), timeout=REQUEST_TIMEOUT_S)
    response.raise_for_status()
    feed = pd.json_normalize(response.json()["transactions"])
    status = feed.loc[feed["typeCode"] == "SC", list(_MLB_COLUMNS)].rename(columns=_MLB_COLUMNS)
    moves = status.join(status["description"].str.extract(_IL_MOVE)).dropna(subset=["action"])
    # The feed repeats moves: in 2022 one id up to eight times with only one copy naming the
    # player, and a few dozen moves a season under two or three ids. One row per move survives.
    moves = moves.dropna(subset=["player_id"]).drop_duplicates(["date", "description"])
    return moves.astype({"player_id": "Int64", "team_id": "Int64", "list_days": "Int64"})


@click.command()
@click.option(
    "--league",
    "leagues",
    multiple=True,
    type=click.Choice(_LEAGUES, case_sensitive=False),
    help="League to pull; repeatable. Defaults to both.",
)
@click.option(
    "--season",
    "seasons",
    multiple=True,
    type=int,
    help=f"Season to pull; repeatable. Defaults to {_FIRST_SEASON} through the current one.",
)
def injury_history(leagues: tuple[str, ...], seasons: tuple[int, ...]) -> None:
    """Re-pull NFL injury reports and MLB injured-list moves, one parquet per season.

    The NFL default stops at nflreadpy's current season, not the calendar year: nflverse
    has no file for a season until its injury reports start in September.
    """
    current = {"NFL": nflr.get_current_season(), "MLB": date.today().year}
    pairs = [
        (league, season)
        for league in leagues or _LEAGUES
        for season in seasons or range(_FIRST_SEASON, current[league] + 1)
    ]
    written = {}
    for league, season in tqdm(pairs, desc="injury history", unit="season"):
        frame = nfl.import_injuries([season]) if league == "NFL" else _mlb_injuries(season)
        _atomic_write_parquet(frame, PLAYER_DATA_BASE / league / "injuries" / f"{season}.parquet")
        written[f"{league}/{season}"] = len(frame)
    click.echo("injury-history: " + " ".join(f"{name}={rows}" for name, rows in written.items()))
