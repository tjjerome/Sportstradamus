"""Fill the moneypuck season-sum columns on NHL gamelog rows that predate them.

The I6f season-to-date comp profile is derived from ``SEASON_SUM_COLUMNS``; the dev
gamelog's older rows were fetched before those columns existed, so this re-fetches their
games and writes the gamelog back in place.
"""

from __future__ import annotations

import click

from sportstradamus.helpers.io import write_gamelog
from sportstradamus.stats import StatsNHL
from sportstradamus.stats.nhl import SEASON_SUM_COLUMNS


@click.command()
@click.option(
    "--season", type=int, multiple=True, help="Restrict to these season start years, e.g. 2025."
)
def main(season: tuple[int, ...]) -> None:
    """Re-fetch the NHL games whose gamelog rows lack the season-sum columns."""
    nhl = StatsNHL()
    nhl.load()
    refetched = nhl.backfill_season_sums(seasons=set(season) or None)
    write_gamelog("nhl", nhl.gamelog, nhl.teamlog, nhl.players)
    missing = nhl.gamelog[list(SEASON_SUM_COLUMNS)].isna().any(axis=1).sum()
    click.echo(f"games re-fetched: {refetched}; rows still lacking a season-sum column: {missing}")


if __name__ == "__main__":
    main()
