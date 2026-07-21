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

Or refresh every cached matrix at once after a broad backfill (the ``*_corr``
feature matrices carry no book columns and are skipped):

    poetry run python -m sportstradamus.scripts.inject_backfilled_odds --all-cached
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
from sportstradamus.scripts.delete_corrupt_seed import BLOWN_ABS_CEILING, BLOWN_LINE_FACTOR

# 8 PM UTC commence-time stand-in from stats/base.py get_training_matrix. The
# archive read happens at this minus the training lookback; backfilled rows
# (observed_at 01:00) and the midnight seed both precede it, so get_ev returns
# the latest observation = the backfill.
_COMMENCE_STANDIN = datetime.timedelta(hours=20)

# Sentinel Odds value written for synthesized rows so pipeline._step_synthesize_odds
# catches them via its `Odds == 0` mask arm — works even when Odds_synthetic is absent.
_SYNTH_ODDS_SENTINEL: float = 0.0


def _target_at(game_date: datetime.date) -> datetime.datetime:
    return (
        datetime.datetime.combine(game_date, datetime.time())
        + _COMMENCE_STANDIN
        - TRAINING_LOOKBACK
    )


def _is_blown(ev: float, line: float) -> bool:
    """A cached EV the swept archive can no longer price — pre-``get_ev``-fix residue."""
    return ev > BLOWN_ABS_CEILING or (line > 0 and ev > BLOWN_LINE_FACTOR * line)


def _resolve_row(
    archive: Archive,
    league: str,
    market: str,
    dist: str,
    cv: float,
    row: pd.Series,
    has_synth: bool,
) -> tuple[float, float, float, bool, bool]:
    """Re-derived ``(line, ev, odds, synthetic, archived)`` book columns for one
    cached row.

    A row the archive prices with a sane EV is recomputed from it (the backfill
    supersedes the seed); a row with no archived price keeps its clean cached
    book columns. A resolved EV that is blown (an archive value the magnitude
    sweep missed — inverted-moderate, or a point-in-time line below the latest
    one) or already NaN (a prior synthesis, or un-re-quotable pre-fix residue)
    is synthesized: ``Odds`` set to ``_SYNTH_ODDS_SENTINEL``, ``EV`` left NaN
    for ``pipeline._step_synthesize_odds`` to fill canonically. No matrix row
    may carry a blown or NaN ``EV``; a NaN ``EV`` crashes the LightGBMLSS fit.

    ``archived`` mirrors ``stats/base.py`` ``_resolve_player_market_odds``: a row
    is authentic iff the archive holds a positive line for it (``line > 0``),
    read *before* the no-price fallback overwrites ``line`` with the cached seed.
    The authentic mask further excludes synthesized rows, so an archived-but-blown
    row (``archived=True``, ``synthetic=True``) is correctly not counted.
    """
    game_date = pd.to_datetime(row["Date"]).date()
    at = _target_at(game_date)
    date = game_date.strftime("%Y-%m-%d")
    ev = archive.get_ev(league, market, date, row["Player"], at=at)
    line = archive.get_line(league, market, date, row["Player"], at=at)
    archived = line > 0
    priced = not np.isnan(ev) and line > 0
    if not priced:
        line, ev = row["Line"], row["EV"]
    if np.isnan(ev) or _is_blown(ev, line):
        return line, np.nan, _SYNTH_ODDS_SENTINEL, True, archived
    if priced:
        return line, ev, 1 - get_odds(line, ev, dist, cv=cv), False, archived
    cached_synth = row["Odds_synthetic"] if has_synth else False
    return line, ev, row["Odds"], cached_synth, archived


def _refresh_one(
    archive: Archive, league: str, market: str, M: pd.DataFrame
) -> tuple[int, float, float, int, int]:
    """Re-derive ``Line``/``Odds``/``EV``/``Archived`` from the archive for every
    cached row.

    Mirrors the ``Odds = 1 - get_odds(line, ev)`` derivation in ``stats/base.py``
    ``get_training_matrix`` so the injected columns equal a full rebuild's (see
    ``_resolve_row`` for the per-row policy). ``Archived`` is reconciled too so a
    newly backfilled real line lifts the authentic count the two-part support
    audit reads — refreshing prices without it leaves a real line flagged
    synthetic. Returns
    ``(n_changed, frac_half_before, frac_half_after, authentic_before, authentic_after)``
    where the coin-flip fraction (``Odds == 0.5``) is the degeneracy fingerprint
    the backfill clears and ``authentic = Archived & ~Odds_synthetic``.
    """
    cv = stat_cv[league].get(market, 1)
    dist = stat_dist.get(league, {}).get(market, "Gamma")
    has_synth = "Odds_synthetic" in M.columns
    lines, evs, odds, synth, arch = [], [], [], [], []
    for _, row in tqdm(M.iterrows(), total=len(M), desc=f"{league} {market}", leave=False):
        line, ev, od, sy, ar = _resolve_row(archive, league, market, dist, cv, row, has_synth)
        lines.append(line)
        evs.append(ev)
        odds.append(od)
        synth.append(sy)
        arch.append(ar)

    new_odds = np.asarray(odds, dtype=float)
    old_odds = M["Odds"].to_numpy(dtype=float)
    n_changed = int((~np.isclose(new_odds, old_odds)).sum())
    frac_before = float(np.isclose(old_odds, 0.5).mean())
    frac_after = float(np.isclose(new_odds, 0.5).mean())

    old_synth = M["Odds_synthetic"].to_numpy(dtype=bool) if has_synth else np.zeros(len(M), bool)
    new_arch = np.asarray(arch, dtype=bool)
    auth_before = int((M["Archived"].to_numpy(dtype=bool) & ~old_synth).sum())
    auth_after = int((new_arch & ~np.asarray(synth, dtype=bool)).sum())

    M["Line"] = lines
    M["EV"] = evs
    M["Odds"] = new_odds
    M["Odds_synthetic"] = synth
    M["Archived"] = new_arch
    return n_changed, frac_before, frac_after, auth_before, auth_after


def _all_cached_cells() -> list[tuple[str, str]]:
    """``(league, market)`` for every cached training matrix, recovered from its
    filename slug — the inverse of ``market_file_slug``: split the league prefix
    off the single underscore, turn hyphens back into spaces. The ``*_corr``
    feature matrices reverse to a ``"corr"`` market that carries no book columns;
    ``main`` skips them via its ``Odds``-column guard rather than by name."""
    root = pkg_resources.files(data) / "training_data"
    cells = []
    for path in sorted(root.iterdir(), key=lambda p: p.name):
        if not path.name.endswith(".parquet"):
            continue
        league, slug = path.name[: -len(".parquet")].split("_", 1)
        cells.append((league, slug.replace("-", " ")))
    return cells


@click.command()
@click.option("--league", default="NFL", help="League whose cached matrices to refresh.")
@click.option("--markets", default=None, help="Comma-separated market names (stat_map values).")
@click.option(
    "--all-cached",
    is_flag=True,
    help="Refresh every cached matrix across all leagues (ignores --league/--markets).",
)
@click.option("--dry-run", is_flag=True, help="Report the refresh delta; do not write the parquet.")
def main(league, markets, all_cached, dry_run):
    """Inject backfilled archive odds into cached training matrices (no feature
    rebuild). Follow with ``meditate --force`` to retrain against the real book."""
    if not all_cached and not markets:
        raise click.UsageError("provide --markets, or --all-cached to refresh every cached matrix")
    archive = Archive()
    cells = _all_cached_cells() if all_cached else [(league, m.strip()) for m in markets.split(",")]
    for lg, market in cells:
        path = pkg_resources.files(data) / f"training_data/{market_file_slug(lg, market)}.parquet"
        if not path.is_file():
            click.echo(f"  {lg} {market}: no cached matrix, skip")
            continue
        M = pd.read_parquet(path)
        if "Odds" not in M.columns:
            click.echo(f"  {lg} {market}: not a book matrix (no Odds column), skip")
            continue
        n_changed, frac_before, frac_after, auth_before, auth_after = _refresh_one(
            archive, lg, market, M
        )
        click.echo(
            f"  {lg} {market}: {n_changed}/{len(M)} rows changed; "
            f"Odds==0.5 {frac_before:.1%} -> {frac_after:.1%}; "
            f"authentic {auth_before} -> {auth_after}"
        )
        if not dry_run:
            M.to_parquet(path, compression="zstd", index=True)
    if dry_run:
        click.echo("DRY RUN — no parquet written.")


if __name__ == "__main__":
    main()
