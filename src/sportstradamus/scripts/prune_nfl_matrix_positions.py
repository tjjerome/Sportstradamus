"""One-shot migration: prune cached NFL training matrices to position-eligible rows.

Stage A1.6. NFL market matrices historically pooled all four positions
(QB/WR/RB/TE) per gameday, so position-locked markets carried wrong-population
rows -- e.g. ``NFL_passing-yards.parquet`` read mean 38 because ~82% of rows were
WR/RB/TE players with 0 passing yards (the QB-only mean is ~216). The durable
write-side fix lives in :meth:`StatsNFL._market_position_filter`, which scopes all
*future* incremental rows; this script applies the same scoping once to the
already-cached parquets by dropping rows whose ``Player position`` code is not
eligible for the market.

Eligibility and the 1-based position codes come from the single source of truth
``NFL_MARKET_POSITIONS`` and ``StatsNFL.positions`` -- no hardcoded codes. Markets
absent from the map (the fantasy-points composites) are genuinely all-position and
are left untouched. Training parquets are gitignored, so this produces no commit;
it cleans the local cache and is the reproducible mechanism for any environment.

Usage:
    poetry run python -m sportstradamus.scripts.prune_nfl_matrix_positions
    poetry run python -m sportstradamus.scripts.prune_nfl_matrix_positions --dry-run
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm

from sportstradamus import data
from sportstradamus.stats import StatsNFL
from sportstradamus.stats.nfl import NFL_MARKET_POSITIONS

# Feature column holding the 1-based position code (QB=1, WR=2, RB=3, TE=4 per
# StatsNFL.positions), written by the player profile in stats/base.py.
_POSITION_COLUMN = "Player position"
# Target column whose mean we report as the cleanup's headline signal.
_RESULT_COLUMN = "Result"


@click.command()
@click.option("--dry-run", is_flag=True, help="Report row/mean changes without writing.")
def main(dry_run: bool) -> None:
    """Prune each position-scoped NFL training parquet to its eligible-position rows."""
    training_data_dir = Path(str(resources.files(data) / "training_data"))
    positions = StatsNFL().positions

    for market, eligible in tqdm(
        NFL_MARKET_POSITIONS.items(), unit="market", desc="Pruning NFL matrices"
    ):
        path = training_data_dir / f"NFL_{market.replace(' ', '-')}.parquet"
        if not path.is_file():
            tqdm.write(f"  skip {market}: no parquet at {path.name}")
            continue
        codes = {positions.index(p) + 1 for p in eligible}
        df = pd.read_parquet(path)
        pruned = df[df[_POSITION_COLUMN].isin(codes)].reset_index(drop=True)
        verb = "would prune" if dry_run else "pruned"
        tqdm.write(
            f"  {market:<22s} {verb}: {len(df):>6d} -> {len(pruned):>6d} rows, "
            f"mean {df[_RESULT_COLUMN].mean():7.2f} -> {pruned[_RESULT_COLUMN].mean():7.2f}  "
            f"(codes {sorted(codes)})"
        )
        if not dry_run:
            pruned.to_parquet(path, compression="zstd", index=True)


if __name__ == "__main__":
    main()
