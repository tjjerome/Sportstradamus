"""Per-run capture of the starters the live leagues' ``Stats`` name for each team's next game.

``prophecize`` updates every live league before it scores, and the update reads the schedule:
``StatsMLB.upcoming_games`` holds today's probable pitchers and the batting orders posted so far,
``StatsNHL.upcoming_games`` dobbersports' predicted starting goalie. The next update overwrites
both, so the T2a news guard would never see what a decision saw. Each run stores them, one row
per (team, role, player), at ``LINEUP_CAPTURE_DIR/<league>/date=YYYY-MM-DD/HHMMSS.parquet``,
named by the run's ``Scored At`` (UTC). ``team`` is the ``upcoming_games`` key, so an MLB
doubleheader's second game carries a "2" suffix. No request: the update already made them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from sportstradamus.helpers.io import LINEUP_CAPTURE_DIR, write_capture_parquet
from sportstradamus.spiderLogger import logger

if TYPE_CHECKING:
    from sportstradamus.stats import Stats

# The leagues whose update names the starters of each team's next game.
LINEUP_LEAGUES = ("MLB", "NHL")
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


def capture_lineups(stats: dict[str, Stats], scored_at: pd.Timestamp) -> None:
    """Write each live league's starters, one parquet per league, named by the run's stamp.

    A league absent from ``stats`` (out of season, or its update failed) or with no game ahead
    writes nothing.
    """
    for league in LINEUP_LEAGUES:
        if league not in stats:
            continue
        # Diagnostics: nothing that goes wrong in a capture may cost the board, or another
        # league's capture, its run.
        try:
            rows = lineup_rows(league, stats[league].upcoming_games, scored_at)
            if not rows.empty:
                write_capture_parquet(rows, LINEUP_CAPTURE_DIR / league, scored_at)
        except Exception:
            logger.exception(f"{league} lineup capture not written")
