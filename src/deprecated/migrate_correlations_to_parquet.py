"""One-shot migration: per-league correlation CSVs -> parquet (zstd).

Three artifact families are migrated:

1. ``data/training_data/{LEAGUE}_corr.csv`` — intermediate per-game record
   used as the warm-start cache by ``training/correlate.py``. Single-level
   integer index.
2. ``data/{LEAGUE}_corr_same_team.csv`` — within-team pair correlations.
   3-level MultiIndex ``(team, market_a, market_b)``.
3. ``data/{LEAGUE}_corr_opposing.csv`` — cross-team pair correlations.
   3-level MultiIndex ``(team, team_market, opp_market)``.

The unsuffixed ``data/{LEAGUE}_corr.csv`` files at the package root are
stale artifacts from a prior pipeline layout and have no current readers;
they are removed rather than converted.

Usage:
    poetry run python -m sportstradamus.scripts.migrate_correlations_to_parquet
    # ...or keep the CSVs for safety:
    poetry run python -m sportstradamus.scripts.migrate_correlations_to_parquet --keep-csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import pandas as pd
import pandas.testing as pdt

import sportstradamus

# Empty-file threshold. The legacy writer emits ``,R\n`` (3 bytes) when a
# league has no qualifying pairs; reading that with a MultiIndex spec fails,
# so we short-circuit to an empty parquet with the same column.
EMPTY_CSV_BYTE_THRESHOLD: int = 8


def _human_size(n_bytes: int) -> str:
    size = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _migrate_intermediate(csv_path: Path) -> tuple[bool, int, int]:
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


def _migrate_stratified(csv_path: Path) -> tuple[bool, int, int]:
    parquet_path = csv_path.with_suffix(".parquet")
    if csv_path.stat().st_size <= EMPTY_CSV_BYTE_THRESHOLD:
        pd.DataFrame(columns=["R"]).to_parquet(parquet_path, compression="zstd")
        return True, csv_path.stat().st_size, parquet_path.stat().st_size

    df = pd.read_csv(csv_path, index_col=[0, 1, 2])
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
    data_dir = Path(sportstradamus.__file__).parent / "data"
    training_data_dir = data_dir / "training_data"

    intermediates = sorted(training_data_dir.glob("*_corr.csv"))
    same_team = sorted(data_dir.glob("*_corr_same_team.csv"))
    opposing = sorted(data_dir.glob("*_corr_opposing.csv"))
    stale_unsuffixed = sorted(p for p in data_dir.glob("*_corr.csv") if p.parent == data_dir)

    plan: list[tuple[Path, str]] = []
    plan.extend((p, "intermediate") for p in intermediates)
    plan.extend((p, "stratified") for p in same_team)
    plan.extend((p, "stratified") for p in opposing)

    if not plan and not stale_unsuffixed:
        click.echo("No correlation CSVs to migrate.")
        return

    click.echo(f"Migrating {len(plan)} correlation file(s)")

    total_csv = 0
    total_parquet = 0
    failures: list[str] = []
    converted: list[Path] = []
    for csv_path, kind in plan:
        if kind == "intermediate":
            ok, csv_bytes, pq_bytes = _migrate_intermediate(csv_path)
        else:
            ok, csv_bytes, pq_bytes = _migrate_stratified(csv_path)
        if not ok:
            failures.append(csv_path.name)
            continue
        converted.append(csv_path)
        total_csv += csv_bytes
        total_parquet += pq_bytes
        ratio = csv_bytes / pq_bytes if pq_bytes else float("inf")
        rel = csv_path.relative_to(data_dir)
        click.echo(
            f"  {rel!s:<48s} "
            f"{_human_size(csv_bytes):>8s} -> "
            f"{_human_size(pq_bytes):>8s}  ({ratio:4.1f}x)"
        )

    if failures:
        click.echo(f"\n{len(failures)} file(s) failed; nothing was deleted.", err=True)
        for name in failures:
            click.echo(f"  - {name}", err=True)
        sys.exit(1)

    if total_parquet:
        click.echo(
            f"\nTotal: {_human_size(total_csv)} CSV -> "
            f"{_human_size(total_parquet)} parquet "
            f"({total_csv / total_parquet:.1f}x)"
        )

    if stale_unsuffixed:
        stale_bytes = sum(p.stat().st_size for p in stale_unsuffixed)
        click.echo(
            f"\nStale unsuffixed correlation files at package root "
            f"({_human_size(stale_bytes)} total — no current readers):"
        )
        for p in stale_unsuffixed:
            click.echo(f"  - {p.name} ({_human_size(p.stat().st_size)})")

    if keep_csv:
        click.echo("\n--keep-csv set; legacy CSVs left in place.")
        return

    for csv_path in converted:
        csv_path.unlink()
    for p in stale_unsuffixed:
        p.unlink()
    removed = len(converted) + len(stale_unsuffixed)
    click.echo(f"\nRemoved {removed} legacy CSV file(s).")


if __name__ == "__main__":
    main()
