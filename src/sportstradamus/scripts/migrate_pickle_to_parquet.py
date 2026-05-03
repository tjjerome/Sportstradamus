"""One-shot migration: history.dat / parlay_hist.dat -> parquet.

Run once after deploying the parquet writers. The legacy `.dat` files are
left in place as a safety net; delete them in a follow-up commit after a
verified prophecize + reflect + dashboard round-trip.

Usage:
    poetry run python -m sportstradamus.scripts.migrate_pickle_to_parquet
"""

from __future__ import annotations

import sys

import click
import pandas as pd

from sportstradamus.helpers.io import (
    HISTORY_PATH,
    HISTORY_PICKLE_PATH,
    PARLAY_HIST_PATH,
    PARLAY_HIST_PICKLE_PATH,
    read_history,
    read_parlay_hist,
    write_history,
    write_parlay_hist,
)


def _migrate(label: str, pickle_path, parquet_path, write_fn, read_fn) -> bool:
    if not pickle_path.is_file():
        click.echo(f"[skip] {label}: {pickle_path} missing")
        return True
    src = pd.read_pickle(pickle_path)
    click.echo(f"[{label}] read pickle: {len(src):,} rows, {len(src.columns)} cols")
    write_fn(src)
    rt = read_fn()
    click.echo(f"[{label}] wrote parquet: {parquet_path} ({len(rt):,} rows)")
    if len(src) != len(rt):
        click.echo(f"[{label}] ROW COUNT MISMATCH: pickle={len(src)} parquet={len(rt)}", err=True)
        return False
    src_cols = set(src.columns) - {"_date"}
    rt_cols = set(rt.columns)
    missing = src_cols - rt_cols
    if missing:
        click.echo(f"[{label}] missing cols after round-trip: {sorted(missing)}", err=True)
        return False
    return True


@click.command()
def main() -> None:
    ok = True
    ok &= _migrate(
        "history",
        HISTORY_PICKLE_PATH,
        HISTORY_PATH,
        write_history,
        read_history,
    )
    ok &= _migrate(
        "parlay_hist",
        PARLAY_HIST_PICKLE_PATH,
        PARLAY_HIST_PATH,
        write_parlay_hist,
        read_parlay_hist,
    )

    # Spot-check Offers round-trip on history (the heterogeneous list-of-tuple column).
    if HISTORY_PATH.is_file():
        df = read_history()
        if "Offers" in df.columns and df["Offers"].notna().any():
            sample = df["Offers"].dropna().iloc[0]
            click.echo(f"[history] Offers sample after round-trip: {sample!r}")
            if not isinstance(sample, list) or not all(isinstance(o, tuple) for o in sample):
                click.echo("[history] Offers did NOT round-trip as list[tuple]", err=True)
                ok = False

    if not ok:
        sys.exit(1)
    click.echo("Migration OK. Legacy .dat files left in place as backup.")


if __name__ == "__main__":
    main()
