"""Per-run capture of the starters each live league names for its next games, and the NBA's news.

``prophecize`` updates every live league before it scores, and the update reads the schedule:
``StatsMLB.upcoming_games`` holds today's probable pitchers and the batting orders posted so far,
``StatsNHL.upcoming_games`` dobbersports' predicted starting goalie. The next update overwrites
both, so the T2a news guard would never see what a decision saw. Each run stores them, one row
per (team, role, player), at ``LINEUP_CAPTURE_DIR/<league>/date=YYYY-MM-DD/HHMMSS.parquet``,
named by the run's ``Scored At`` (UTC). ``team`` is the ``upcoming_games`` key, so an MLB
doubleheader's second game carries a "2" suffix. No request: the update already made them.

The NBA update names no starters, so with the NBA live a run reads stats.nba.com's daily lineups
(starters, actives, inactives) into the same tree, one row per listed player, while a tip is
within ``DUE_WINDOW``. Their own ``timestamp`` is unreliable (the probes found stale values) and
stays only as ``source_timestamp``. The run also stores the newest official injury report
(``collectors.nba_injury_report``). Licence: NBA.com terms (the class nba-api already reads).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from sportstradamus.collectors.nba_injury_report import capture_injury_report
from sportstradamus.helpers.io import (
    LINEUP_CAPTURE_DIR,
    read_upcoming_events,
    write_capture_parquet,
)
from sportstradamus.helpers.scraping import CAPTURE_TIMEOUT_S
from sportstradamus.spiderLogger import logger
from sportstradamus.stats.nba_client import NBA_STATS_HEADERS

if TYPE_CHECKING:
    from sportstradamus.stats import Stats

# The leagues whose starters a run stores: MLB and NHL from their update, the NBA from
# stats.nba.com.
LINEUP_LEAGUES = ("MLB", "NHL", "NBA")
# Column order and dtype of every league's table, pinned so a date's files read together: an
# all-null column left to pyarrow is typed ``null``.
LINEUP_DTYPES = {
    "polled_at": "datetime64[ns]",
    "team": "string",
    "opponent": "string",
    "home": "boolean",
    "role": "string",
    "slot": "Int64",
    "player": "string",
}
LINEUPS_URL = "https://stats.nba.com/js/data/leaders/00_daily_lineups_{day}.json"
# How close an NBA tip must be for a run to fetch: lineups post ~30 min out, earlier polls repeat.
DUE_WINDOW = timedelta(hours=2)
# stats.nba.com names a day's lineups file by its Eastern date.
_ET = ZoneInfo("America/New_York")
NBA_LINEUP_DTYPES = {
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


def _starters(league: str, game: dict) -> list[tuple[str, int | None, str]]:
    """``(role, slot, name)`` per starter: an NHL goalie, or an MLB pitcher and batting order."""
    if league == "NHL":
        return [("goalie", None, game["Goalie"])]
    return [("pitcher", None, game["Pitcher"])] + [
        ("batter", slot, name) for slot, name in enumerate(game["Batting Order"], start=1)
    ]


def lineup_rows(
    league: str, upcoming_games: dict[str, dict], polled_at: pd.Timestamp
) -> pd.DataFrame:
    """One row per starter of each team's next game; one not yet named keeps its row, player NA."""
    rows = [
        {
            "polled_at": polled_at,
            "team": team,
            "opponent": game["Opponent"],
            "home": game["Home"],
            "role": role,
            "slot": slot,
            "player": name or None,
        }
        for team, game in upcoming_games.items()
        for role, slot, name in _starters(league, game)
    ]
    return pd.DataFrame(rows, columns=list(LINEUP_DTYPES)).astype(LINEUP_DTYPES)


def due(events: list[dict], now: datetime) -> bool:
    """Whether any NBA event commences within ``DUE_WINDOW`` of ``now``."""
    return any(
        now <= datetime.fromisoformat(event["commence_time"]) <= now + DUE_WINDOW
        for event in events
        if event["league"] == "NBA"
    )


def nba_lineup_rows(payload: dict, polled_at: pd.Timestamp) -> pd.DataFrame:
    """One row per listed player on both teams of every game."""
    rows = [
        {
            "polled_at": polled_at,
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
        for game in payload.get("games", [])
        for team in (game["homeTeam"], game["awayTeam"])
        for player in team.get("players", [])
    ]
    return pd.DataFrame(rows, columns=list(NBA_LINEUP_DTYPES)).astype(NBA_LINEUP_DTYPES)


def _nba_lineups(scored_at: pd.Timestamp) -> pd.DataFrame:
    """Today's stats.nba.com lineups, or no rows (and no request) with no tip within the window."""
    now = scored_at.tz_localize(UTC)
    if not due(read_upcoming_events(), now):
        return pd.DataFrame()
    response = requests.get(
        LINEUPS_URL.format(day=now.astimezone(_ET).strftime("%Y%m%d")),
        headers=NBA_STATS_HEADERS,
        timeout=CAPTURE_TIMEOUT_S,
    )
    response.raise_for_status()
    return nba_lineup_rows(response.json(), scored_at)


def capture_lineups(stats: dict[str, Stats], scored_at: pd.Timestamp) -> None:
    """Write each live league's starters, one parquet per league, and the NBA injury report.

    A league absent from ``stats`` (out of season, or its update failed) or with no game ahead
    writes nothing.
    """
    for league in LINEUP_LEAGUES:
        if league not in stats:
            continue
        # Diagnostics: nothing that goes wrong in a capture may cost the board, or another
        # capture, its run.
        try:
            rows = (
                _nba_lineups(scored_at)
                if league == "NBA"
                else lineup_rows(league, stats[league].upcoming_games, scored_at)
            )
            if not rows.empty:
                write_capture_parquet(rows, LINEUP_CAPTURE_DIR / league, scored_at)
        except Exception:
            logger.exception(f"{league} lineup capture not written")
    if "NBA" in stats:
        # Apart from the lineups: the report lives on another host, so it fails on its own.
        try:
            capture_injury_report(scored_at)
        except Exception:
            logger.exception("NBA injury report not captured")
