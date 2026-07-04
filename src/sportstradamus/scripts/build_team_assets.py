#!/usr/bin/env python3
"""Authoring aid: print each league's real team codes for team_assets.json.

Reads ``data/leagues/{lg}/teamlog.parquet`` (falling back to ``gamelog.parquet``
if teamlog is absent) and prints the unique, non-null team codes found there, so
a human authoring or re-authoring ``team_assets.json`` (next season, next
expansion) sees reality directly instead of trusting a stale docstring. MLB has
no teamlog/gamelog in this repo (checked: neither file exists under
``data/leagues/mlb/``) — for that league this prints a note pointing at
``config/abbreviations.json``'s ``"MLB"`` section instead of crashing.

Usage
-----
    poetry run python -m sportstradamus.scripts.build_team_assets
    poetry run python -m sportstradamus.scripts.build_team_assets --league NHL
"""

from __future__ import annotations

from importlib import resources

import click
import pandas as pd

from sportstradamus import data

# Column NBA/WNBA teamlog+gamelog carry the team code under; NFL/NHL use "team".
_ABBR_COLUMN_BY_LEAGUE = {
    "NBA": "TEAM_ABBREVIATION",
    "WNBA": "TEAM_ABBREVIATION",
    "NFL": "team",
    "NHL": "team",
}

_LEAGUES = ("NBA", "NFL", "NHL", "MLB", "WNBA")


def _codes_from_parquet(league: str, filename: str) -> list[str] | None:
    path = resources.files(data) / "leagues" / league.lower() / filename
    if not path.is_file():
        return None
    df = pd.read_parquet(str(path))
    column = _ABBR_COLUMN_BY_LEAGUE[league]
    return sorted(code for code in df[column].unique() if code is not None)


def _print_league(league: str) -> None:
    if league == "MLB":
        click.echo(
            f"{league}: no teamlog.parquet or gamelog.parquet in this repo — "
            'use config/abbreviations.json\'s "MLB" section (name -> code map) instead.'
        )
        return

    teamlog_codes = _codes_from_parquet(league, "teamlog.parquet")
    gamelog_codes = _codes_from_parquet(league, "gamelog.parquet")
    codes = sorted(set(teamlog_codes or []) | set(gamelog_codes or []))

    click.echo(f"{league}: {len(codes)} unique code(s)")
    click.echo(f"  {codes}")
    if (
        teamlog_codes is not None
        and gamelog_codes is not None
        and (set(teamlog_codes) != set(gamelog_codes))
    ):
        click.echo(
            f"  note: teamlog ({len(teamlog_codes)}) and gamelog ({len(gamelog_codes)}) "
            "code sets differ"
        )


@click.command()
@click.option(
    "--league",
    type=click.Choice(_LEAGUES),
    default=None,
    help="Print only one league; default prints all five.",
)
def main(league: str | None) -> None:
    """Print the unique team codes found in each league's teamlog/gamelog parquet."""
    for lg in [league] if league else _LEAGUES:
        _print_league(lg)


if __name__ == "__main__":
    main()
