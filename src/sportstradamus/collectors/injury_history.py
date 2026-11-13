"""Injury history for training: one parquet per (league, season) under ``player_data/``.

A dev-side pull run by hand beside the weekly ``meditate``; the model-build lane derives
injury features from these files. The NFL, MLB and NHL injured-reserve files are re-pulled
and overwritten every run, since the current season changes week to week; a pull that finds
no injured-reserve move writes nothing, so a blank reply never overwrites a stored year.

NFL is nflverse's weekly injury report with practice statuses, one row per (season, week,
team, gsis_id), written as delivered. The 2022-2024 files carry ``date_modified`` (UTC; a
median 47 h before kickoff) and later ones only the week. nflverse overwrites a week's rows
as the report updates, so only each player's final status for the week survives.

MLB is the statsapi transactions feed cut to Status Change rows that name the injured list
("disabled list" before 2019), each parsed into its action, list length and injury. An
activation from another list (paternity, bereavement, restricted) drops, and so does one
naming no list at all ("New York Yankees activated SS Braden Shewmake."; 517 in 2025),
since nothing says which list it ended.

NBA, WNBA and NHL are the per-game lists of who did not play, which ``game_absences``
fetches, writes and resumes; that module describes them.

NHL scratch lists leave out players on injured reserve, who are off the active roster, so NHL
also gets ESPN's dated transactions cut to injured-reserve moves, one row per player moved, at
``player_data/NHL/injured_reserve/<year>.parquet``. They are ESPN's record, not the league's,
and name players without ids, so the model build resolves the names. Each file is a calendar
year, which is what ESPN's ``season`` parameter means: the 2024 file runs from 2024-01-01 to
2024-12-31, across two NHL seasons.
"""

from __future__ import annotations

import itertools
import re
from datetime import date

import click
import nfl_data_py as nfl
import nflreadpy as nflr
import pandas as pd
import requests
from tqdm import tqdm

from sportstradamus.collectors import game_absences
from sportstradamus.collectors.tabular import PLAYER_DATA_BASE
from sportstradamus.helpers.io import _atomic_write_parquet
from sportstradamus.helpers.scraping import REQUEST_TIMEOUT_S

_LEAGUES = ("NFL", "MLB", *game_absences.ABSENCE_LEAGUES)

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

# Keyless. A calendar year holds up to about 2,000 moves: two pages of 1,000.
_NHL_TRANSACTIONS_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/transactions"
    "?season={year}&limit=1000&page={page}"
)

# One clause moving players on or off injured reserve ("Placed D Dylan DeMelo on injured
# reserve", "Moved RW Troy Terry from the non-roster list to long-term injured reserve"). Its
# player list takes a period only after an initial ("J.T. Miller") or "St." ("Jack St. Ivany"),
# so a match never runs into the next sentence. A verb after "and" is lower case ("... and
# placed G Ukko-Pekka Luukkonen on injured reserve"), and an "and" clause with none ("... and D
# Matt Roy on injured reserve") takes the verb of the clause before it.
_IR_MOVE = re.compile(
    r"\b(?:(?P<action>(?i:placed|activated|reinstated|transferred|moved))|and(?=\s[A-Z]))\s+"
    r"(?P<players>(?:[^.]|(?<=\b[A-Z])\.|(?<=\bSt)\.)+?)"
    r"\s+(?:on|from|to|off)\s+(?:the\s+)?(?:(?:non-roster list|active roster)\s+(?:to|from)\s+)?"
    r"(?P<list>(?:long-term |non-roster )?injured reserve)"
)
_PLAYER_SEPARATOR = r",\s+(?:and\s+)?|\s+and\s+"

# ESPN's position codes: C, W, LW, RW, F; D, LD, RD; G. A player can carry two ("C/LW"), and a
# plural ("Fs Owen Tippett and Travis Konecny") covers each name after it in the clause.
_POSITION = r"(?:[LR]?[WD]|[CFG])"
_PLAYER = re.compile(rf"(?:(?P<position>{_POSITION}(?:/{_POSITION})*)s?\s+)?(?P<player_name>.+)")

# Column order and dtype, pinned so the years read together: an all-null column left to
# pyarrow is typed ``null``.
_IR_DTYPES = {
    "date": "datetime64[ns]",
    "team": "string",
    "action": "string",
    "list": "string",
    "position": "string",
    "player_name": "string",
    "description": "string",
}


def _nhl_injured_reserve(year: int) -> pd.DataFrame:
    """One calendar year of ESPN's NHL injured-reserve moves, one row per player moved.

    ``action`` is ESPN's verb (Placed, Activated, Reinstated, Transferred, Moved) and ``list``
    its list: injured reserve, or long-term or non-roster injured reserve.
    """
    transactions = []
    for page in itertools.count(1):
        response = requests.get(
            _NHL_TRANSACTIONS_URL.format(year=year, page=page), timeout=REQUEST_TIMEOUT_S
        )
        response.raise_for_status()
        reply = response.json()
        transactions += reply["transactions"]
        if reply["pageIndex"] >= reply["pageCount"]:
            break
    # A new calendar year's first days can hold no injured-reserve move yet.
    if not any(_IR_MOVE.search(transaction["description"]) for transaction in transactions):
        return pd.DataFrame(columns=list(_IR_DTYPES)).astype(_IR_DTYPES)
    feed = pd.json_normalize(transactions).rename(columns={"team.abbreviation": "team"})
    clauses = feed["description"].str.extractall(_IR_MOVE)
    clauses["action"] = clauses.groupby(level=0)["action"].ffill().str.capitalize()
    players = (
        clauses["players"].str.split(_PLAYER_SEPARATOR, regex=True).explode().str.extract(_PLAYER)
    )
    players["position"] = players.groupby(level=[0, 1])["position"].ffill()
    moves = (
        players.join(clauses[["action", "list"]])
        .droplevel("match")
        .join(feed[["date", "team", "description"]])
    )
    # ESPN stamps a move at Pacific midnight (07:00Z or 08:00Z), so its UTC day is its day.
    moves["date"] = pd.to_datetime(moves["date"]).dt.tz_localize(None).dt.normalize()
    return moves[list(_IR_DTYPES)].astype(_IR_DTYPES)


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
    help="League to pull; repeatable. Defaults to every league.",
)
@click.option(
    "--season",
    "seasons",
    multiple=True,
    type=int,
    help=f"Season to pull; repeatable. Defaults to {_FIRST_SEASON} through the current one.",
)
def injury_history(leagues: tuple[str, ...], seasons: tuple[int, ...]) -> None:
    """Pull every league's injury history, one parquet per season.

    NFL injury reports, MLB injured-list moves and ESPN's NHL injured-reserve moves are
    re-pulled whole; the NBA, WNBA and NHL per-game absence lists fetch only games not yet
    stored. The NHL injured-reserve file for a season holds that calendar year.

    The NFL default stops at nflreadpy's current season, not the calendar year: nflverse
    has no file for a season until its injury reports start in September.
    """
    current = {"NFL": nflr.get_current_season()}
    pairs = [
        (league, season)
        for league in leagues or _LEAGUES
        for season in seasons or range(_FIRST_SEASON, current.get(league, date.today().year) + 1)
    ]
    written = {}
    for league, season in tqdm(pairs, desc="injury history", unit="season"):
        injuries = PLAYER_DATA_BASE / league / "injuries" / f"{season}.parquet"
        if league in game_absences.ABSENCE_LEAGUES:
            # A per-game pull resumes from its own file, so it alone writes that file.
            written[f"{league}/{season}"] = game_absences.update_season(league, season, injuries)
        else:
            frame = nfl.import_injuries([season]) if league == "NFL" else _mlb_injuries(season)
            _atomic_write_parquet(frame, injuries)
            written[f"{league}/{season}"] = len(frame)
        if league == "NHL":
            moves = _nhl_injured_reserve(season)
            # An empty year, or a blank reply, never overwrites a year already stored.
            if not moves.empty:
                _atomic_write_parquet(
                    moves, PLAYER_DATA_BASE / "NHL" / "injured_reserve" / f"{season}.parquet"
                )
            written[f"NHL-IR/{season}"] = len(moves)
    click.echo("injury-history: " + " ".join(f"{name}={rows}" for name, rows in written.items()))
