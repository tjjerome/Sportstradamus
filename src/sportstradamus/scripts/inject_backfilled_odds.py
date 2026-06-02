"""Refresh the book columns (``Line``/``Odds``/``EV``) of cached training
matrices from the archive, without rebuilding the expensive feature columns.

After a historical backfill (``scripts/backfill_historical_odds.py``) lands real
two-sided prices into ``archive.duckdb``, the cached per-cell matrices in
``data/training_data/{cell}.parquet`` still carry the OLD degenerate book
columns: ``training.pipeline._build_training_matrix`` only fetches game-dates
*after* the cached cutoff each ``meditate`` run, so historical rows' ``Odds`` /
``EV`` are frozen at first build and a backfill never reaches them.

This driver re-derives ``Line``/``Odds``/``EV`` for every cached row from the
current archive using the SAME point-in-time read the training join uses
(``target_at`` = game-date 20:00 UTC commence stand-in minus the training
lookback; see ``stats/base.py`` ``get_training_matrix``). Because the backfill
stamped its rows at ``01:00`` and the legacy seed sits at midnight, that read
returns the backfilled price — so the injected values equal what a full rebuild
would produce, while the (slow) feature columns are left untouched. A normal
``meditate --force`` run then reuses the refreshed cache and retrains the cell
against the real book.

    poetry run python -m sportstradamus.scripts.inject_backfilled_odds \
        --league NFL --markets "passing yards,attempts,completions" --dry-run
"""

import datetime
import importlib.resources as pkg_resources

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from sportstradamus import data
from sportstradamus.helpers import Archive, get_odds, stat_cv, stat_dist
from sportstradamus.helpers.archive import TRAINING_LOOKBACK
from sportstradamus.helpers.io import market_file_slug

# 8 PM UTC commence-time stand-in from stats/base.py get_training_matrix. The
# archive read happens at this minus the training lookback; backfilled rows
# (observed_at 01:00) and the midnight seed both precede it, so get_ev returns
# the latest observation = the backfill.
_COMMENCE_STANDIN = datetime.timedelta(hours=20)


def _target_at(game_date: datetime.date) -> datetime.datetime:
    return datetime.datetime.combine(game_date, datetime.time()) + _COMMENCE_STANDIN - TRAINING_LOOKBACK


def _refresh_one(archive, league: str, market: str, M: pd.DataFrame):
    """Re-derive ``Line``/``Odds``/``EV`` from the archive for every cached row.

    Mirrors the ``Odds = 1 - get_odds(line, ev)`` derivation in ``stats/base.py``
    ``get_training_matrix`` so the injected columns equal a full rebuild's. Rows
    with no archived price keep their cached book columns. Returns
    ``(n_changed, frac_half_before, frac_half_after)`` where the coin-flip
    fraction (``Odds == 0.5``) is the degeneracy fingerprint the backfill clears.
    """
    cv = stat_cv[league].get(market, 1)
    dist = stat_dist.get(league, {}).get(market, "Gamma")
    has_synth = "Odds_synthetic" in M.columns
    lines, evs, odds, synth = [], [], [], []
    for _, row in tqdm(M.iterrows(), total=len(M), desc=f"{league} {market}", leave=False):
        game_date = pd.to_datetime(row["Date"]).date()
        at = _target_at(game_date)
        date = game_date.strftime("%Y-%m-%d")
        ev = archive.get_ev(league, market, date, row["Player"], at=at)
        line = archive.get_line(league, market, date, row["Player"], at=at)
        if np.isnan(ev) or line <= 0:
            lines.append(row["Line"])
            evs.append(row["EV"])
            odds.append(row["Odds"])
            synth.append(row["Odds_synthetic"] if has_synth else False)
            continue
        lines.append(line)
        evs.append(ev)
        odds.append(1 - get_odds(line, ev, dist, cv=cv))
        synth.append(False)

    new_odds = np.asarray(odds, dtype=float)
    old_odds = M["Odds"].to_numpy(dtype=float)
    n_changed = int((~np.isclose(new_odds, old_odds)).sum())
    frac_before = float(np.isclose(old_odds, 0.5).mean())
    frac_after = float(np.isclose(new_odds, 0.5).mean())

    M["Line"] = lines
    M["EV"] = evs
    M["Odds"] = new_odds
    if has_synth:
        M["Odds_synthetic"] = synth
    return n_changed, frac_before, frac_after


@click.command()
@click.option("--league", default="NFL", help="League whose cached matrices to refresh.")
@click.option("--markets", required=True, help="Comma-separated market names (stat_map values).")
@click.option("--dry-run", is_flag=True, help="Report the refresh delta; do not write the parquet.")
def main(league, markets, dry_run):
    """Inject backfilled archive odds into cached training matrices (no feature
    rebuild). Follow with ``meditate --force`` to retrain against the real book."""
    archive = Archive()
    for market in (m.strip() for m in markets.split(",")):
        path = pkg_resources.files(data) / f"training_data/{market_file_slug(league, market)}.parquet"
        if not path.is_file():
            click.echo(f"  {market}: no cached matrix, skip")
            continue
        M = pd.read_parquet(path)
        n_changed, frac_before, frac_after = _refresh_one(archive, league, market, M)
        click.echo(
            f"  {market}: {n_changed}/{len(M)} rows changed; "
            f"Odds==0.5 {frac_before:.1%} -> {frac_after:.1%}"
        )
        if not dry_run:
            M.to_parquet(path, compression="zstd", index=True)
    if dry_run:
        click.echo("DRY RUN — no parquet written.")


if __name__ == "__main__":
    main()
