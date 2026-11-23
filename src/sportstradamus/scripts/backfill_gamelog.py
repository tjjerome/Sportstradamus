"""Fill the gamelog columns that rows stored before them lack, writing the gamelog back.

NHL: the I6f season-to-date comp profile is derived from ``SEASON_SUM_COLUMNS``; the dev
gamelog's older rows were fetched before those columns existed, so this re-fetches their
games. ``--reports`` fills the stats-service ``REPORT_COLUMNS`` instead, which the hourly
update fills only two weeks back; it re-parses no game.

MLB: re-reads Savant's game feed for each game whose rows lack ``PITCH_COLUMNS``, which the
update writes only for the games it logs. The gamelog is written after each season, so a
stopped run resumes in the season it stopped in.
"""

from __future__ import annotations

import click
import pandas as pd

from sportstradamus.helpers.io import write_gamelog
from sportstradamus.stats import StatsMLB, StatsNHL
from sportstradamus.stats.mlb_pitch import PITCH_COLUMNS, fill_pitch_columns
from sportstradamus.stats.nhl import SEASON_SUM_COLUMNS
from sportstradamus.stats.nhl_reports import REPORT_COLUMNS, fill_report_columns


@click.command()
@click.option("--league", type=click.Choice(["NHL", "MLB"]), required=True)
@click.option(
    "--season",
    type=int,
    multiple=True,
    help="Restrict to these seasons, by start year, e.g. 2025.",
)
@click.option(
    "--reports",
    is_flag=True,
    help="NHL: fill the stats-service report columns instead; re-parses no game.",
)
def main(league: str, season: tuple[int, ...], reports: bool) -> None:
    """Fill the gamelog rows that lack NHL season sums or report columns, or MLB pitch counts."""
    seasons = set(season) or None
    if league == "MLB":
        if reports:
            raise click.UsageError("--reports fills NHL columns")
        mlb = StatsMLB()
        mlb.load()
        for year in sorted(seasons or set(pd.to_datetime(mlb.gamelog["gameDate"]).dt.year)):
            mlb.gamelog = fill_pitch_columns(mlb.gamelog, seasons={year})
            write_gamelog("mlb", mlb.gamelog, mlb.teamlog, mlb.players)
        lacking = mlb.gamelog[PITCH_COLUMNS].isna().any(axis=1).sum()
        click.echo(f"rows still lacking a pitch column: {lacking}")
        return

    nhl = StatsNHL()
    nhl.load()
    if reports:
        nhl.gamelog = fill_report_columns(nhl.gamelog, seasons=seasons)
        columns, summary = REPORT_COLUMNS, "rows still lacking a report column"
    else:
        refetched = nhl.backfill_season_sums(seasons=seasons)
        columns = list(SEASON_SUM_COLUMNS)
        summary = f"games re-fetched: {refetched}; rows still lacking a season-sum column"
    write_gamelog("nhl", nhl.gamelog, nhl.teamlog, nhl.players)
    click.echo(f"{summary}: {nhl.gamelog[columns].isna().any(axis=1).sum()}")


if __name__ == "__main__":
    main()
