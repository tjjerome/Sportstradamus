#!/usr/bin/env python3
"""One-shot migration: rename scoring columns in runtime parquet snapshots.

The 2026-06-14 scoring-column rename made the betting expected-value ``Model EV``
everywhere and the projected stat mean ``Projection``. Snapshots written before the
rename carry the old names; this rewrites each runtime snapshot in place with the
frame-appropriate names so the dashboard reads them without a full ``prophecize``
re-run. Idempotent — a snapshot already on the new names is skipped. The per-offer /
per-leg payloads are positional tuples, so only top-level columns are renamed.

Usage
-----
    poetry run python -m sportstradamus.scripts.migrate_scoring_cols
    poetry run python -m sportstradamus.scripts.migrate_scoring_cols --dry-run
"""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd
import pyarrow.parquet as pq

from sportstradamus.helpers.io import _atomic_write_parquet

# Offer-frame snapshot: the stat-mean ``Model EV`` becomes ``Projection`` while the
# offer EV multipliers ``Model`` / ``Books`` become the betting ``Model EV`` /
# ``Market EV``. pandas ``rename`` remaps original labels atomically, so the paired
# ``Model`` -> ``Model EV`` and ``Model EV`` -> ``Projection`` never collide.
_OFFER_MAP = {
    "Model EV": "Projection",
    "Books EV": "Market Projection",
    "Model": "Model EV",
    "Books": "Market EV",
    "Model P": "Win Prob",
    "Books P": "Market Prob",
    "Close Books P": "Close Market Prob",
    "Push P": "Push Prob",
    "Model STD": "Projection STD",
    "K": "Kelly",
}
# Parlay-frame snapshot: ``Model EV`` is already the parlay betting-EV (unchanged); only
# the book leg renames.
_PARLAY_MAP = {"Books EV": "Market EV"}
# History prediction level: the stat-mean pair renames; the Offers tuples are positional.
_HISTORY_MAP = {"Model EV": "Projection", "Books EV": "Market Projection"}

_FILE_MAPS: dict[str, dict[str, str]] = {
    "current_offers.parquet": _OFFER_MAP,
    "current_parlays.parquet": _PARLAY_MAP,
    "current_history.parquet": _OFFER_MAP,
    "history.parquet": _HISTORY_MAP,
    "parlay_hist.parquet": _PARLAY_MAP,
}


def _planned_renames(path: Path, rename_map: dict[str, str]) -> dict[str, str]:
    """Renames to apply, or empty when the snapshot is already migrated (schema-only read).

    A rename key that is also one of the target names can't distinguish a pre-migration
    snapshot from a migrated one — in the offer frame ``Model EV`` is both the old
    stat-mean (→ ``Projection``) and the new betting-EV name (from old ``Model``). So
    migration is gated on the keys that are *unambiguously* old; once those are gone the
    snapshot is treated as done and the ambiguous mapping never fires a second time.
    """
    names = set(pq.read_schema(path).names)
    values = set(rename_map.values())
    if not ({key for key in rename_map if key not in values} & names):
        return {}
    return {old: new for old, new in rename_map.items() if old in names}


@click.command()
@click.option(
    "--runtime-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Runtime snapshot directory (defaults to the packaged data/runtime).",
)
@click.option("--dry-run", is_flag=True, help="Report planned renames without writing.")
def main(runtime_dir: Path | None, dry_run: bool) -> None:
    base = runtime_dir or Path(__file__).resolve().parents[1] / "data" / "runtime"
    for fname, rename_map in _FILE_MAPS.items():
        path = base / fname
        if not path.exists():
            click.echo(f"{fname:32s} absent")
            continue
        present = _planned_renames(path, rename_map)
        if not present:
            click.echo(f"{fname:32s} already migrated")
            continue
        if dry_run:
            click.echo(f"{fname:32s} would rename {present}")
            continue
        df = pd.read_parquet(path).rename(columns=present)
        _atomic_write_parquet(df, path)
        click.echo(f"{fname:32s} renamed {list(present)}")


if __name__ == "__main__":
    main()
