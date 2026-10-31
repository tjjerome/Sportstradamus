"""``fetch nba-lineups``: stats.nba.com's daily lineups (starters, actives, inactives), poll-stamped.

The JSON's own ``timestamp`` is unreliable (the probes found stale values), so each row carries the
poll time; a decision time joins to the latest poll at or before it. One parquet per poll at
``NBA_LINEUPS_DIR/date=<ET date>/<ET HHMMSS>.parquet``, the clock Eastern like its partition so a
date's files sort past 8 PM. A tick makes no request unless an NBA event in ``upcoming_events.json``
commences within ``DUE_WINDOW``: lineups post about half an hour before tip, so an evening of
quarter-hour ticks costs nothing on an off night. Licence: stats.nba.com terms (the class nba-api
already reads); the cron line is the owner's to install.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import click
import pandas as pd
import requests

from sportstradamus.helpers.io import NBA_LINEUPS_DIR, _atomic_write_parquet, read_upcoming_events
from sportstradamus.helpers.scraping import REQUEST_TIMEOUT_S
from sportstradamus.stats.nba_client import NBA_STATS_HEADERS

LINEUPS_URL = "https://stats.nba.com/js/data/leaders/00_daily_lineups_{day}.json"
# How close an NBA tip must be before a tick fetches: lineups post ~30 min out, earlier polls repeat.
DUE_WINDOW = timedelta(hours=2)
_ET = ZoneInfo("America/New_York")
LINEUP_DTYPES = {
    "polled_at": "datetime64[ns]",
    "game_id": "string",
    "game_status": "string",
    "team": "string",
    "person_id": "Int64",
    "player": "string",
    "lineup_status": "string",
    "position": "string",
    "roster_status": "string",
    "source_timestamp": "string",
}


def due(events: list[dict], now: datetime) -> bool:
    """Whether any NBA event commences within ``DUE_WINDOW`` of ``now``."""
    for event in events:
        if event["league"] != "NBA":
            continue
        commence = datetime.fromisoformat(event["commence_time"])
        if now <= commence <= now + DUE_WINDOW:
            return True
    return False


def lineup_rows(payload: dict, polled_at: datetime) -> pd.DataFrame:
    """One row per listed player on both teams of every game."""
    stamp = pd.Timestamp(polled_at).tz_convert("UTC").tz_localize(None)
    rows = []
    for game in payload.get("games", []):
        for side in ("homeTeam", "awayTeam"):
            team = game[side]
            for player in team.get("players", []):
                rows.append(
                    {
                        "polled_at": stamp,
                        "game_id": game["gameId"],
                        "game_status": game.get("gameStatusText"),
                        "team": team.get("teamAbbreviation"),
                        "person_id": player.get("personId"),
                        "player": player.get("playerName"),
                        "lineup_status": player.get("lineupStatus"),
                        "position": player.get("position"),
                        "roster_status": player.get("rosterStatus"),
                        "source_timestamp": player.get("timestamp"),
                    }
                )
    return pd.DataFrame(rows, columns=list(LINEUP_DTYPES)).astype(LINEUP_DTYPES)


@click.command("nba-lineups")
def nba_lineups() -> None:
    """Snapshot today's NBA daily lineups when a tip is within two hours."""
    now = datetime.now(UTC)
    if not due(read_upcoming_events(), now):
        click.echo("nba-lineups: no NBA tip within the window, no request")
        return
    clock = now.astimezone(_ET)
    response = requests.get(
        LINEUPS_URL.format(day=clock.strftime("%Y%m%d")),
        headers=NBA_STATS_HEADERS,
        timeout=REQUEST_TIMEOUT_S,
    )
    response.raise_for_status()
    rows = lineup_rows(response.json(), now)
    _atomic_write_parquet(
        rows,
        NBA_LINEUPS_DIR
        / f"date={clock.strftime('%Y-%m-%d')}"
        / f"{clock.strftime('%H%M%S')}.parquet",
    )
    click.echo(f"nba-lineups: {len(rows)} rows over {rows['game_id'].nunique()} games")
