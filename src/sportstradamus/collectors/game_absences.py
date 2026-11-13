"""Who sat out each NBA, WNBA and NHL game in our gamelogs, from the leagues' own game records.

The gamelogs hold only players who played, so an absence cannot be read off a missing row.
:func:`update_season` asks the league about each of a season's gamelog games and stores one row
per absent player, one parquet per (league, season); ``fetch injury-history`` runs it dev-side,
weekly. The file is the record: a run asks only for games it does not hold. A game where nobody
sat leaves no row, so every run asks for it again.

NBA and WNBA read two endpoints a game from stats.nba.com and stats.wnba.com. BoxScoreSummaryV3's
``inactives`` are the players left off the active list, status ``"Inactive"``; the list gives no
reason, so injury, a G League two-way, personal leave and suspension all land there.
BoxScoreTraditionalV3's players with a ``comment`` were in the box score and did not play; the
status is the comment as delivered ("DNP - Coach's Decision", "DNP - Injury/Illness",
"DND - Injury/Illness", "NWT - Not With Team"). Many WNBA games leave the inactive list empty
(2021: 128 of 209), their box scores listing every rostered player, and some 2026-27 NBA box
scores list the inactive players themselves, with the reason ("Inactive - Injury/Illness",
"Inactive - G League - Two-Way"), and leave the list empty. stats.wnba.com stops answering a
process after about 600 requests (300 games) and answers a new one at once: stop a cold WNBA
backfill when it stalls and rerun it, since a run resumes from its file.

NHL reads api-web.nhle.com's gamecenter right rail: each team's scratches, status ``"Scratch"``,
the active-roster players who did not dress. A player on injured reserve is off the active roster
and never listed (Florida's 2025-10-21 list omits Barkov and Tkachuk, both on long-term IR);
:mod:`injury_history` pulls ESPN's dated injured-reserve moves for them. The scratch lists name no
team, so home and away come from the gamelog, in its spelling (TB, not the NHL's TBL).

These are game-time facts (who sat and, for the NBA and WNBA, partly why), not the pregame status
a decision saw. Ids are the leagues' own: NBA and WNBA ``personId`` is the gamelog's
``PLAYER_ID``, NHL ``id`` its ``playerId`` (a string there).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests
from nba_api.stats.endpoints import BoxScoreSummaryV3, BoxScoreTraditionalV3
from tqdm import tqdm

from sportstradamus.helpers.io import _atomic_write_parquet, _gamelog_paths, read_parquet_safe
from sportstradamus.helpers.scraping import REQUEST_TIMEOUT_S
from sportstradamus.spiderLogger import logger
from sportstradamus.stats.nba_client import NBA_HOST, WNBA_HOST, NBAStatsError, StatsHost, fetch

ABSENCE_LEAGUES = ("NBA", "WNBA", "NHL")
# Column order and dtype, pinned so a league's seasons read together: an all-null column left to
# pyarrow is typed ``null``.
ABSENCE_DTYPES = {
    "game_id": "string",
    "game_date": "datetime64[ns]",
    "team": "string",
    "player_id": "Int64",
    "player_name": "string",
    "status": "string",
}
# Each league reads its own host: WNBA calls spend stats.wnba.com's rate budget, not the NBA's.
_HOSTS = {"NBA": NBA_HOST, "WNBA": WNBA_HOST}
_RIGHT_RAIL_URL = "https://api-web.nhle.com/v1/gamecenter/{game_id}/right-rail"
# One absent player: (team, player_id, player_name, status), the file's columns after the game's.
_Absence = tuple[str, int, str, str]


def _games(league: str, season: int) -> pd.DataFrame:
    """The season's gamelog games by date: ``game_id``, ``game_date``, and NHL ``away``/``home``."""
    path = _gamelog_paths(league)["gamelog"]
    if league == "NHL":
        log = pd.read_parquet(path, columns=["gameId", "gameDate", "team", "opponent", "home"])
        # An NHL game id leads with its season's start year: 2025020103 -> 2025.
        log = log[log["gameId"] // 1_000_000 == season].drop_duplicates("gameId")
        games = pd.DataFrame(
            {
                "game_id": log["gameId"].astype(str),
                "game_date": log["gameDate"],
                "away": log["opponent"].where(log["home"], log["team"]),
                "home": log["team"].where(log["home"], log["opponent"]),
            }
        )
    else:
        log = pd.read_parquet(path, columns=["GAME_ID", "GAME_DATE"]).drop_duplicates("GAME_ID")
        # Digits 4-5 of an NBA or WNBA game id are its season's start year: 0042300401, the
        # 2024 Finals, is 2023's.
        log = log[log["GAME_ID"].str[3:5].astype(int) + 2000 == season]
        games = log.rename(columns={"GAME_ID": "game_id", "GAME_DATE": "game_date"})
    games = games.assign(game_date=pd.to_datetime(games["game_date"]).dt.normalize())
    return games.sort_values(["game_date", "game_id"])


def _nba_absences(host: StatsHost, game_id: str) -> list[_Absence]:
    """The inactive list, then the box-score players with a comment, away team first."""
    summary = fetch(BoxScoreSummaryV3, host=host, game_id=game_id).get_dict()["boxScoreSummary"]
    box = fetch(BoxScoreTraditionalV3, host=host, game_id=game_id).get_dict()["boxScoreTraditional"]
    absences: list[_Absence] = []
    for side in ("awayTeam", "homeTeam"):
        team = summary[side]["teamTricode"]
        sat = [(player, "Inactive") for player in summary[side]["inactives"]]
        sat += [(player, player["comment"]) for player in box[side]["players"] if player["comment"]]
        absences += [
            (team, player["personId"], f"{player['firstName']} {player['familyName']}", status)
            for player, status in sat
        ]
    return absences


def _nhl_scratches(game_id: str, away: str, home: str) -> list[_Absence]:
    response = requests.get(_RIGHT_RAIL_URL.format(game_id=game_id), timeout=REQUEST_TIMEOUT_S)
    response.raise_for_status()
    teams = response.json()["gameInfo"]
    # The NHL API leaves out what it lacks (the probed replies hold no empty list and no null), so
    # a team that scratched no one may come back without the key.
    return [
        (
            team,
            scratch["id"],
            f"{scratch['firstName']['default']} {scratch['lastName']['default']}",
            "Scratch",
        )
        for side, team in (("awayTeam", away), ("homeTeam", home))
        for scratch in teams[side].get("scratches", [])
    ]


def update_season(league: str, season: int, path: Path) -> int:
    """Fetch the season's not-yet-stored games, write ``path``, return the rows it holds.

    A game whose request fails after its retries is logged and left out, so the next run asks for
    it again. Any other error stops the run, with what it fetched already written.
    """
    stored = read_parquet_safe(path).reindex(columns=list(ABSENCE_DTYPES)).astype(ABSENCE_DTYPES)
    games = _games(league, season)
    pending = games[~games["game_id"].isin(stored["game_id"])]
    rows: list[tuple] = []
    failed = 0
    try:
        for game in tqdm(
            pending.itertuples(index=False),
            total=len(pending),
            desc=f"{league} {season}",
            unit="game",
            leave=False,
        ):
            try:
                if league == "NHL":
                    absences = _nhl_scratches(game.game_id, game.away, game.home)
                else:
                    absences = _nba_absences(_HOSTS[league], game.game_id)
            except (NBAStatsError, requests.RequestException) as exc:
                failed += 1
                logger.warning(f"{league} game {game.game_id} absences not fetched: {exc}")
                continue
            rows += [(game.game_id, game.game_date, *absence) for absence in absences]
    finally:
        fetched = pd.DataFrame(rows, columns=list(ABSENCE_DTYPES)).astype(ABSENCE_DTYPES)
        frame = pd.concat([stored, fetched], ignore_index=True)
        if rows:
            _atomic_write_parquet(frame, path)
    if failed:
        logger.warning(
            f"{league} {season}: {failed} of {len(pending)} games not fetched; rerun to retry them"
        )
    return len(frame)
