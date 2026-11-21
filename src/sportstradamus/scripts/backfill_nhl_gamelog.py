"""Fill the NHL gamelog columns that rows stored before them lack, writing the gamelog back.

The I6f season-to-date comp profile is derived from ``SEASON_SUM_COLUMNS``; the dev
gamelog's older rows were fetched before those columns existed, so this re-fetches their
games. ``--reports`` fills the stats-service ``REPORT_COLUMNS`` instead, which the hourly
update fills only two weeks back; it re-parses no game.
"""

from __future__ import annotations

import click

from sportstradamus.helpers.io import write_gamelog
from sportstradamus.stats import StatsNHL
from sportstradamus.stats.nhl import SEASON_SUM_COLUMNS
from sportstradamus.stats.nhl_reports import REPORT_COLUMNS, fill_report_columns


@click.command()
@click.option(
    "--season", type=int, multiple=True, help="Restrict to these season start years, e.g. 2025."
)
@click.option(
    "--reports",
    is_flag=True,
    help="Fill the stats-service report columns instead; re-parses no game.",
)
def main(season: tuple[int, ...], reports: bool) -> None:
    """Fill the NHL gamelog rows that lack the season-sum columns, or the report columns."""
    nhl = StatsNHL()
    nhl.load()
    seasons = set(season) or None
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
