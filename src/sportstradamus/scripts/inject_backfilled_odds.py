"""Re-resolve the odds-provenance block of cached training matrices from the
archive, without rebuilding the expensive feature columns.

Two situations leave a cached matrix stale. A historical backfill
(``scripts/backfill_historical_odds.py``) lands real two-sided prices into
``archive.duckdb`` that the cached rows never see, because ``_step_load_matrix``
only fetches game-dates *after* the cached cutoff each ``meditate`` run. And a
matrix built before the provenance block shipped carries none of
``QuoteSource``/``QuoteAuthenticity``/``QuoteBookCount``, so the first in-season
append concatenates two incompatible eras and ``incomplete_provenance_rows``
refuses the cell.

Both are repaired by re-running the training join's own resolver over the cached
rows. ``Stats.resolve_player_market_odds`` reads only ``stats.index`` and
``stats["Avg10"]`` from its frame — both cached — and its combo fallback needs
only the 300-day log window, not the profile frames built from it. So one
``window_short_logs`` call per gameday reproduces what a rebuild resolves, at one
batched archive query per gameday instead of a full feature rebuild.

The whole quote is written, prices included. Repairing the block alone would
strand a row whose class changed — one relabelled ``combo_ev_inversion`` while
its ``Odds`` stayed the neutral 0.5 reads as book evidence bought by a coin
flip, which is what the load-time provenance guard exists to prevent. Writing
the resolved ``Line`` is likewise what a rebuild produces: once the block is
present ``_clip_lines`` clips synthetic rows only, so authentic and derived
lines stay unclipped, and the next ``meditate`` re-trims from the repaired block.

    poetry run python -m sportstradamus.scripts.inject_backfilled_odds \
        --league NFL --markets "passing yards,attempts,completions" --dry-run

Or sweep every cached matrix at once (the ``*_corr`` feature matrices carry no
book columns and are skipped). ``--legacy-only`` narrows the sweep to the
matrices that predate the block, so repairing them cannot disturb the training
data of a cell that is already serving:

    poetry run python -m sportstradamus.scripts.inject_backfilled_odds \
        --all-cached --legacy-only
"""

import datetime
import importlib.resources as pkg_resources
from collections import Counter

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from sportstradamus import data
from sportstradamus.helpers.archive import TRAINING_LOOKBACK
from sportstradamus.helpers.io import market_file_slug
from sportstradamus.helpers.training_quotes import PROVENANCE_COLUMNS
from sportstradamus.stats import Stats, StatsMLB, StatsNBA, StatsNFL, StatsNHL, StatsWNBA

# 8 PM UTC commence-time stand-in from stats/base.py get_training_matrix. The
# archive read happens at this minus the training lookback; backfilled rows
# (observed_at 01:00) and the midnight seed both precede it, so the read returns
# the latest observation = the backfill.
_COMMENCE_STANDIN = datetime.timedelta(hours=20)

_LEAGUE_CLASSES = {
    "NBA": StatsNBA,
    "NFL": StatsNFL,
    "WNBA": StatsWNBA,
    "MLB": StatsMLB,
    "NHL": StatsNHL,
}

_REPAIRED_COLUMNS = ("Line", "Odds", "EV", "Archived", "Odds_synthetic", *PROVENANCE_COLUMNS)


def _target_at(game_date: datetime.date) -> datetime.datetime:
    return (
        datetime.datetime.combine(game_date, datetime.time())
        + _COMMENCE_STANDIN
        - TRAINING_LOOKBACK
    )


def _load_league(league: str):
    """Load one league's gamelog exactly as the training run does, minus the network."""
    cls = _LEAGUE_CLASSES[league]
    # Probable pitchers describe the live upcoming slate; a repair reads cached history only.
    stat_data = cls(load_live_pitchers=False) if cls is StatsMLB else cls()
    stat_data.load()
    stat_data.trim_gamelog()
    return stat_data


def resolve_cached_quotes(stat_data: Stats, market: str, M: pd.DataFrame) -> pd.DataFrame:
    """Re-resolve every cached row's quote through the training join's own resolver.

    Returns the canonical quote record for each row, aligned to ``M.index``.
    """
    game_dates = pd.to_datetime(M["Date"]).dt.date
    resolved = []
    for game_date, rows in tqdm(
        M.groupby(game_dates), unit="gameday", desc=f"{stat_data.league} {market}", leave=False
    ):
        stat_data.window_short_logs(game_date)
        players = rows["Player"]
        stats = pd.DataFrame({"Avg10": rows["Avg10"].to_numpy()}, index=players)
        stats = stats.loc[~stats.index.duplicated()]
        quotes = stat_data.resolve_player_market_odds(
            stats, market, game_date.strftime("%Y-%m-%d"), _target_at(game_date)
        )
        quotes = quotes.reindex(players)
        quotes.index = rows.index
        resolved.append(quotes)
    return pd.concat(resolved).reindex(M.index)


def _repair_one(stat_data: Stats, market: str, M: pd.DataFrame) -> str:
    """Write the re-resolved quotes into ``M`` and describe what moved."""
    quotes = resolve_cached_quotes(stat_data, market, M)
    legacy = "QuoteSource" not in M.columns
    authentic_before = int(M["Archived"].to_numpy(dtype=bool).sum())
    price_moved = int(
        (~np.isclose(quotes["Odds"].to_numpy(float), M["Odds"].to_numpy(float))).sum()
    )
    for column in _REPAIRED_COLUMNS:
        M[column] = quotes[column]

    sources = Counter(quotes["QuoteSource"])
    breakdown = ", ".join(f"{source} {count}" for source, count in sources.most_common())
    return (
        f"{len(M)} rows ({'legacy' if legacy else 'refresh'}); {breakdown}; "
        f"authentic {authentic_before} -> {int(M['Archived'].sum())}; "
        f"Odds moved on {price_moved}"
    )


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
@click.option("--league", default="NFL", help="League whose cached matrices to repair.")
@click.option("--markets", default=None, help="Comma-separated market names (stat_map values).")
@click.option(
    "--all-cached",
    is_flag=True,
    help="Repair every cached matrix across all leagues (ignores --league/--markets).",
)
@click.option(
    "--legacy-only",
    is_flag=True,
    help="Repair only matrices with no provenance block yet, leaving populated ones untouched.",
)
@click.option("--dry-run", is_flag=True, help="Report the repair delta; do not write the parquet.")
def main(league, markets, all_cached, legacy_only, dry_run):
    """Re-resolve cached training matrices' odds provenance from the archive (no
    feature rebuild). Follow with ``meditate`` to retrain against the repaired block."""
    if not all_cached and not markets:
        raise click.UsageError("provide --markets, or --all-cached to repair every cached matrix")
    cells = _all_cached_cells() if all_cached else [(league, m.strip()) for m in markets.split(",")]
    loaded = {}
    for lg, market in cells:
        path = pkg_resources.files(data) / f"training_data/{market_file_slug(lg, market)}.parquet"
        if not path.is_file():
            click.echo(f"  {lg} {market}: no cached matrix, skip")
            continue
        M = pd.read_parquet(path)
        if "Odds" not in M.columns:
            click.echo(f"  {lg} {market}: not a book matrix (no Odds column), skip")
            continue
        if legacy_only and "QuoteSource" in M.columns:
            click.echo(f"  {lg} {market}: provenance block already populated, skip")
            continue
        if lg not in loaded:
            click.echo(f"[{lg}] loading cached gamelogs...")
            loaded[lg] = _load_league(lg)
        click.echo(f"  {lg} {market}: {_repair_one(loaded[lg], market, M)}")
        if not dry_run:
            M.to_parquet(path, compression="zstd", index=True)
    if dry_run:
        click.echo("DRY RUN — no parquet written.")


if __name__ == "__main__":
    main()
