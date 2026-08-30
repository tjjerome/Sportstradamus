#!/usr/bin/env python3
"""Refit one league's player-prop book weights from the (repaired) archive.

The fitted ``book_weights.json`` still encodes whatever the archive held when
``meditate`` last ran — after ``quarantine_sleeper_wnba`` nulls the poisoned
Sleeper WNBA quotes, the weights keep trusting Sleeper (0.626 on WNBA PRA)
until they are refit. This driver reruns
:func:`sportstradamus.training.calibration.fit_book_weights` over every player
market in the league — the same per-market fit + whole-file dump ``meditate``
performs — without a full training run. Nulled evs enter the fit as NaN and are
masked out; no fitter changes are involved.

Only the named league's player-market entries are rewritten. Its
Moneyline/Totals entries and every other league's entries are read from the
existing file and written back untouched.

    poetry run python -m sportstradamus.scripts.refit_book_weights --league WNBA
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import json

import click

from sportstradamus import data
from sportstradamus.helpers import LazyArchive
from sportstradamus.stats import StatsMLB, StatsNBA, StatsNFL, StatsNHL, StatsWNBA
from sportstradamus.training.calibration import fit_book_weights
from sportstradamus.training.markets import ALL_MARKETS

_LEAGUE_CLASSES = {
    "NBA": StatsNBA,
    "NFL": StatsNFL,
    "WNBA": StatsWNBA,
    "MLB": StatsMLB,
    "NHL": StatsNHL,
}

_BOOK_WEIGHTS_PATH = pkg_resources.files(data) / "config" / "book_weights.json"


def refit_book_weights(league: str) -> dict:
    """Refit every ``league`` player-market book weight and rewrite the config.

    Returns the league's refit ``{market: {book: weight}}`` entries. A market
    the fit declines (too few graded samples, no book columns) comes back
    ``{}``, exactly as a ``meditate`` pass would leave it.
    """
    stats_cls = _LEAGUE_CLASSES[league]
    # Probable pitchers describe the live slate over the network; the fit only
    # needs the cached gamelog (prophecize/meditate own the league-API refresh).
    stat_data = stats_cls(load_live_pitchers=False) if stats_cls is StatsMLB else stats_cls()
    stat_data.load()
    archive = LazyArchive()

    with open(_BOOK_WEIGHTS_PATH) as infile:
        book_weights = json.load(infile)

    for market in ALL_MARKETS[league]:
        book_weights.setdefault(league, {}).setdefault(market, {})
        book_weights[league][market] = fit_book_weights(
            league, market, stat_data, archive, book_weights
        )
        click.echo(f"[{league}] {market}: {len(book_weights[league][market])} books fitted")

    with open(_BOOK_WEIGHTS_PATH, "w") as outfile:
        json.dump(book_weights, outfile, indent=4)
    return book_weights[league]


@click.command()
@click.option(
    "--league",
    required=True,
    type=click.Choice(list(ALL_MARKETS)),
    help="League whose player-market book weights to refit.",
)
def main(league: str) -> None:
    """Refit --league's player-prop book weights and rewrite book_weights.json.

    Run it after an archive repair (e.g. quarantine_sleeper_wnba --apply) so the
    fitted consensus weights reflect the repaired rows. Other leagues' entries
    and this league's Moneyline/Totals entries are preserved.
    """
    refitted = refit_book_weights(league)
    click.echo(f"wrote {_BOOK_WEIGHTS_PATH} ({league}: {len(refitted)} player markets refit)")


if __name__ == "__main__":
    main()
