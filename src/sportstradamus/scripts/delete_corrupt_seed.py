"""Delete corrupt ``ev`` rows from the archive — migration seed or blown magnitude.

Two corruption classes, two predicates:

* **Migration seed** (``--default``): the klepto→duckdb migration stamped a
  per-cell seed at game-date **midnight**. For count cells that price was
  inverted through the wrong distribution and blew up (``ev`` in the
  thousands–millions); for markets no book actually quotes, the median line was
  stamped as every book's ``ev`` (a degenerate 0.5 coin-flip). Only midnight-seed
  rows are touched; re-fetched and live rows are left intact. Removing a
  synthetic cell's corrupt seed lets ``training.pipeline`` fall through to its
  honest synthetic 0.5 (``Odds=0.5`` when ``get_ev`` returns ``nan``).

* **Blown magnitude** (``--blown-all-layers``): a grossly inflated ``ev``
  (``> 5×`` the line, or ``> 2000`` with no line) in **any** ``observed_at``
  layer. This is the residue of the ``get_ev`` inversion bugs (ZINB under≤gate
  runaway on the DFS write path; SkewNormal mean→∞ asymptote) that landed blown
  evs in live and re-fetched rows, not just the seed. Use it on cells whose live
  DFS / re-fetch layers were poisoned before the ``get_ev`` fix.

**Back up ``archive/archive.duckdb`` before running** — it mutates the DB in place.

    poetry run python -m sportstradamus.scripts.delete_corrupt_seed \
        --league NFL --markets "rushing tds,receiving tds,qb tds" --dry-run
    poetry run python -m sportstradamus.scripts.delete_corrupt_seed \
        --league NBA --markets "BLK,STL,FG3M" --blown-all-layers --dry-run
"""

from pathlib import Path

import click
import duckdb

# Matches Archive's default DB location (resolved relative to the repo cwd).
ARCHIVE_DB = Path("archive/archive.duckdb")

# Cross-book stddev below this means every book carries the same ev — the
# degenerate coin-flip stamp, not genuine book agreement.
DEGENERATE_STD = 1e-9
# No legitimate player-prop EV reaches this; used when no line exists to compare
# against (passing yards, the largest real market, tops out near 811).
BLOWN_ABS_CEILING = 2000.0
# ev beyond this multiple of the line is a distribution-inversion blowup. It is
# also the get_ev clamp ceiling (SN_MAX_MEAN_FACTOR): the fixed encoder caps every
# write at BLOWN_LINE_FACTOR × its own line and records that line, so MAX(line) over
# a slot bounds every legitimate ev. Anything above it is pre-fix residue.
BLOWN_LINE_FACTOR = 5.0
# Minimum distinct books in a seed group to declare it a degenerate coin-flip stamp.
_MIN_DEGENERATE_BOOKS = 2

_MIDNIGHT_SEED = "date_part('hour', observed_at) = 0 AND date_part('minute', observed_at) = 0"

# A row is magnitude-blown if its ev grossly exceeds the largest line any book
# offered for the slot, or, absent any positive line, an absolute ceiling. Keying
# on MAX(positive line) — not the latest line — spares a correctly-clamped ev from
# a higher-lined book when a later, lower line lands (the lines table has no book
# column), and the line > 0 filter excludes negative team-market lines (run lines,
# spreads) whose 5× ceiling is meaningless. Shared with the global sweep in
# ``scripts.sweep_runaway_odds``.
BLOWN_PREDICATE = f"""
    ev > {BLOWN_ABS_CEILING}
    OR ev > {BLOWN_LINE_FACTOR} * (
        SELECT MAX(l.line) FROM lines l
        WHERE l.league = odds.league AND l.market = odds.market
          AND l.game_date = odds.game_date AND l.entity = odds.entity
          AND l.line > 0)
"""

# A midnight-seed row is corrupt if it is magnitude-blown or sits in an
# all-books-identical (degenerate coin-flip) seed group.
_CORRUPT_SEED_WHERE = f"""
    league = ? AND market = ? AND {_MIDNIGHT_SEED}
    AND (
        {BLOWN_PREDICATE}
        OR EXISTS (
            SELECT 1 FROM odds s
            WHERE s.league = odds.league AND s.market = odds.market
              AND s.game_date = odds.game_date AND s.entity = odds.entity
              AND date_part('hour', s.observed_at) = 0
              AND date_part('minute', s.observed_at) = 0
            GROUP BY s.game_date, s.entity
            HAVING COUNT(DISTINCT s.book) >= {_MIN_DEGENERATE_BOOKS} AND stddev_pop(s.ev) < {DEGENERATE_STD}
        )
    )
"""

# Magnitude-blown rows in any observed_at layer (seed, re-fetch, or live).
_BLOWN_WHERE = f"league = ? AND market = ? AND ({BLOWN_PREDICATE})"


def _count(con, where: str, league: str, market: str) -> int:
    return con.execute(f"SELECT COUNT(*) FROM odds WHERE {where}", [league, market]).fetchone()[0]


def _delete(con, where: str, league: str, market: str) -> int:
    n = _count(con, where, league, market)
    if n:
        con.execute(f"DELETE FROM odds WHERE {where}", [league, market])
    return n


def count_corrupt_seed(con, league: str, market: str) -> int:
    """Number of corrupt midnight-seed rows for one ``(league, market)`` cell."""
    return _count(con, _CORRUPT_SEED_WHERE, league, market)


def delete_corrupt_seed(con, league: str, market: str) -> int:
    """Delete the corrupt midnight-seed rows for one cell; return how many."""
    return _delete(con, _CORRUPT_SEED_WHERE, league, market)


def count_blown(con, league: str, market: str) -> int:
    """Number of magnitude-blown rows (any layer) for one cell."""
    return _count(con, _BLOWN_WHERE, league, market)


def delete_blown(con, league: str, market: str) -> int:
    """Delete the magnitude-blown rows (any layer) for one cell; return how many."""
    return _delete(con, _BLOWN_WHERE, league, market)


@click.command()
@click.option("--league", required=True, help="League whose cells to clean.")
@click.option("--markets", required=True, help="Comma-separated market names (archive values).")
@click.option(
    "--blown-all-layers",
    is_flag=True,
    help="Delete magnitude-blown rows in any layer instead of only the midnight seed.",
)
@click.option("--dry-run", is_flag=True, help="Report counts; do not delete.")
def main(league, markets, blown_all_layers, dry_run):
    """Strip corrupt ev rows so a cell reads clean.

    Back up ``archive/archive.duckdb`` first — non-dry runs mutate it in place.
    """
    count_fn = count_blown if blown_all_layers else count_corrupt_seed
    delete_fn = delete_blown if blown_all_layers else delete_corrupt_seed
    kind = "blown (any layer)" if blown_all_layers else "corrupt seed"
    con = duckdb.connect(str(ARCHIVE_DB), read_only=dry_run)
    try:
        total = 0
        for market in (m.strip() for m in markets.split(",")):
            n = count_fn(con, league, market) if dry_run else delete_fn(con, league, market)
            click.echo(f"  {league} {market}: {'would delete' if dry_run else 'deleted'} {n}")
            total += n
        click.echo(f"{'DRY RUN — ' if dry_run else ''}{league}: {total} {kind} rows total")
    finally:
        con.close()


if __name__ == "__main__":
    main()
