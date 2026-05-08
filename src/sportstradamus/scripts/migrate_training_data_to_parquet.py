"""One-shot migration: ``training_data/{LEAGUE}_{MARKET}.csv`` -> parquet (zstd).

Each per-market CSV is converted in place to a sibling ``.parquet`` file with
zstd compression. Round-trip verified before the legacy CSV is removed.

The ``{LEAGUE}_corr.csv`` correlation matrices are skipped — they are a
different artifact and out of scope per the migration plan.

Usage:
    poetry run python -m sportstradamus.scripts.migrate_training_data_to_parquet
    # ...or keep the CSVs for safety:
    poetry run python -m sportstradamus.scripts.migrate_training_data_to_parquet --keep-csv
"""

from __future__ import annotations

import sys
from importlib import resources
from pathlib import Path

import click
import pandas as pd
import pandas.testing as pdt

from sportstradamus import data


def _human_size(n_bytes: int) -> str:
    size = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _migrate_one(csv_path: Path) -> tuple[bool, int, int]:
    parquet_path = csv_path.with_suffix(".parquet")
    df = pd.read_csv(csv_path, index_col=0)
    df.to_parquet(parquet_path, compression="zstd", index=True)

    rt = pd.read_parquet(parquet_path)
    try:
        pdt.assert_frame_equal(df, rt)
    except AssertionError as exc:
        click.echo(f"[FAIL] {csv_path.name}: round-trip mismatch", err=True)
        click.echo(str(exc).splitlines()[0], err=True)
        parquet_path.unlink(missing_ok=True)
        return False, 0, 0

    return True, csv_path.stat().st_size, parquet_path.stat().st_size


@click.command()
@click.option(
    "--keep-csv",
    is_flag=True,
    help="Leave the legacy .csv files in place after a successful round-trip.",
)
def main(keep_csv: bool) -> None:
    training_data_dir = Path(str(resources.files(data) / "training_data"))
    csvs = sorted(
        p
        for p in training_data_dir.glob("*.csv")
        if not p.name.endswith("_corr.csv")
    )
    if not csvs:
        click.echo("No training_data/*.csv files to migrate.")
        return

    click.echo(f"Migrating {len(csvs)} files in {training_data_dir}")

    total_csv = 0
    total_parquet = 0
    failures: list[str] = []
    for csv_path in csvs:
        ok, csv_bytes, pq_bytes = _migrate_one(csv_path)
        if not ok:
            failures.append(csv_path.name)
            continue
        total_csv += csv_bytes
        total_parquet += pq_bytes
        ratio = csv_bytes / pq_bytes if pq_bytes else float("inf")
        click.echo(
            f"  {csv_path.name:<40s} "
            f"{_human_size(csv_bytes):>8s} -> "
            f"{_human_size(pq_bytes):>8s}  ({ratio:4.1f}x)"
        )

    if failures:
        click.echo(f"\n{len(failures)} file(s) failed; nothing was deleted.", err=True)
        for name in failures:
            click.echo(f"  - {name}", err=True)
        sys.exit(1)

    click.echo(
        f"\nTotal: {_human_size(total_csv)} CSV -> "
        f"{_human_size(total_parquet)} parquet "
        f"({(total_csv / total_parquet) if total_parquet else 0:.1f}x)"
    )

    if keep_csv:
        click.echo("--keep-csv set; legacy CSVs left in place.")
        return

    for csv_path in csvs:
        csv_path.unlink()
    click.echo(f"Removed {len(csvs)} legacy CSV files.")


if __name__ == "__main__":
    main()
