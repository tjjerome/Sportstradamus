"""Repair the cached training matrices' book columns for count cells whose
per-book ``ev`` was corrupted by the parser's old ``get_ev`` SkewNormal default.

Before the moneylines.py dist fix, the props parser inverted every book price
under get_ev's default ``SkewNormal``. For ZINB/NegBin count markets that
inversion is ill-conditioned — its CDF asymptotes to ``Phi(-1/cv)`` — so a
de-vigged under-prob near the asymptote drove the solved mean toward infinity
and the stored per-book ``ev`` blew up (up to ``line * 1e6``). The cached
matrices' ``Odds``/``EV`` are the weighted average of those corrupt per-book
evs, so training reads a garbage book probability.

This driver fixes only the rows that feed training — the cached matrices in
``data/training_data/`` — and never mutates the archive. For each cached
(player, date) it reads the per-book evs the training join would (point-in-time
``target_at``), recovers each book's de-vigged under exactly from the corrupt ev
(the old inversion used the SkewNormal CDF with ``a=0``, whose analytic inverse
is ``under = Phi((line - ev) / (ev * cv))`` — so a book that was *not* in the
blow-up band recovers to its own quote unchanged), re-projects it under the
market's real distribution, and re-averages through the archive's own book
weighting. Rows with no archived price keep their cached (synthesized) columns —
the matrix's no-book fill is honest and must not be touched, which is why this
works per-book off the archive rather than off the already-averaged matrix EV.

    poetry run python -m sportstradamus.scripts.repair_matrix_ev --dry-run
"""

import datetime
import importlib.resources as pkg_resources
from functools import lru_cache

import click
import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm

from sportstradamus import data
from sportstradamus.helpers import Archive, get_ev, get_odds, stat_cv, stat_dist
from sportstradamus.helpers.archive import TRAINING_LOOKBACK
from sportstradamus.helpers.io import market_file_slug

# Distributions whose get_ev count branch the old SkewNormal default corrupted.
_COUNT_DISTS = {"ZINB", "NegBin", "Poisson"}
_PRODUCTION_LEAGUES = ("NBA", "NFL", "NHL", "WNBA", "MLB")
# 8 PM UTC commence-time stand-in from stats/base.py get_training_matrix; the
# training join reads the archive at this minus the lookback (mirrors
# scripts/inject_backfilled_odds.py so repaired evs match what training reads).
_COMMENCE_STANDIN = datetime.timedelta(hours=20)
# A stored ev this many times its line is a SkewNormal blow-up, not a projection.
_BLOWUP_RATIO = 5.0


def _target_at(game_date: datetime.date) -> datetime.datetime:
    return datetime.datetime.combine(game_date, datetime.time()) + _COMMENCE_STANDIN - TRAINING_LOOKBACK


def _count_cells(league_filter: str | None, market_filter: str | None):
    return [
        (league, market)
        for league in _PRODUCTION_LEAGUES
        for market, dist in stat_dist.get(league, {}).items()
        if dist in _COUNT_DISTS
        and (league_filter is None or league == league_filter)
        and (market_filter is None or market == market_filter)
    ]


@lru_cache(maxsize=None)
def _reproject(line: float, under: float, cv: float, dist: str) -> float:
    """Re-project a recovered under-prob under the market's real distribution.

    Cached on the (line, binned-under, cv, dist) key: a cell's line set is small
    and discrete (½-floored) and cv is constant, so per-book repair collapses to
    a few hundred distinct get_ev solves instead of one per archived row.
    """
    return get_ev(line, under, cv, dist=dist)


def _repaired_book_ev(ev_corrupt: float, line: float, cv: float, dist: str) -> float:
    under = float(np.clip(norm.cdf((line - ev_corrupt) / (ev_corrupt * cv)), 1e-6, 1 - 1e-6))
    return _reproject(line, round(under, 4), cv, dist)


def _repair_one(archive, league: str, market: str, M: pd.DataFrame):
    """Re-derive ``Line``/``Odds``/``EV`` from per-book repaired archive evs.

    Returns ``(n_changed, blown_before, blown_after)`` where the blown fraction
    (ev exceeding :data:`_BLOWUP_RATIO` times its line) is the corruption
    fingerprint the repair clears.
    """
    cv = stat_cv[league].get(market, 1)
    dist = stat_dist[league][market]
    lines, evs, odds = [], [], []
    for _, row in tqdm(M.iterrows(), total=len(M), desc=f"{league} {market}", leave=False):
        game_date = pd.to_datetime(row["Date"]).date()
        at = _target_at(game_date)
        date = game_date.strftime("%Y-%m-%d")
        book_rows = archive._book_rows(league, market, date, row["Player"], at=at)
        line = archive.get_line(league, market, date, row["Player"], at=at)
        if not book_rows or line <= 0:
            lines.append(row["Line"])
            evs.append(row["EV"])
            odds.append(row["Odds"])
            continue
        repaired = [
            (book, _repaired_book_ev(ev, line, cv, dist))
            for book, ev in book_rows
            if ev and ev > 0
        ]
        ev_new = archive._weighted_book_ev(league, market, repaired)
        lines.append(line)
        evs.append(ev_new)
        odds.append(1 - get_odds(line, ev_new, dist, cv=cv))

    new_ev = np.asarray(evs, dtype=float)
    new_line = np.asarray(lines, dtype=float)
    old_ev = M["EV"].to_numpy(dtype=float)
    old_line = M["Line"].to_numpy(dtype=float)
    n_changed = int((~np.isclose(new_ev, old_ev)).sum())
    blown_before = float((old_ev > _BLOWUP_RATIO * np.clip(old_line, 0.5, None)).mean())
    blown_after = float((new_ev > _BLOWUP_RATIO * np.clip(new_line, 0.5, None)).mean())

    M["Line"] = new_line
    M["EV"] = new_ev
    M["Odds"] = odds
    return n_changed, blown_before, blown_after


@click.command()
@click.option("--league", default=None, help="Restrict to one league (default: all production count cells).")
@click.option("--market", default=None, help="Restrict to one market (requires --league).")
@click.option("--dry-run", is_flag=True, help="Report the repair delta; do not write the parquet.")
def main(league, market, dry_run):
    """Repair the cached training matrices for production count cells in place.

    Follow with ``meditate --force`` (the refreshed cache is reused) to retrain
    each cell against the repaired book."""
    archive = Archive()
    for lg, mkt in _count_cells(league, market):
        path = pkg_resources.files(data) / f"training_data/{market_file_slug(lg, mkt)}.parquet"
        if not path.is_file():
            continue
        M = pd.read_parquet(path)
        if "EV" not in M.columns:
            continue
        n_changed, blown_before, blown_after = _repair_one(archive, lg, mkt, M)
        click.echo(
            f"  {lg} {mkt}: {n_changed}/{len(M)} rows changed; "
            f"ev>{_BLOWUP_RATIO:.0f}x line {blown_before:.1%} -> {blown_after:.1%}"
        )
        if not dry_run:
            M.to_parquet(path, compression="zstd", index=True)
    if dry_run:
        click.echo("DRY RUN — no parquet written.")


if __name__ == "__main__":
    main()
