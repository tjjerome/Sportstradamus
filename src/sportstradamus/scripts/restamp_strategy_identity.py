#!/usr/bin/env python3
"""One-shot restamp of strategy identity after the canonical-signature narrowing.

``StrategySpec.canonical_signature`` used to hash the whole spec, so widening a search axis
rotated every family's signature — and, through ``corner_fingerprint``, every stored corner
fingerprint. Narrowing the hash to the artifact-contract fields rotates it one last time. This
walks the artifact classes that store a signature and rewrites the rotated values in place
wherever the artifact's *implementation* is unchanged, so those cells need no retrain.

A cell is eligible only when its stored strategy slug, structural slug, implementation version and
artifact schema version still match the registry and its stored controls are still a registered
corner. That is exactly what the narrowed signature now encodes; anything else — notably the
SkewNormal cells carrying ``implementation_version`` 1 against today's 3 — is real drift, is left
untouched, and is reported as needing a retrain.

Every artifact class here is gitignored, so git is not the rollback: originals are copied to
``research/logs/restamp_identity/<stamp>/`` before anything is rewritten.

Usage
-----
    sportstradamus admin restamp-identity --dry-run
    sportstradamus admin restamp-identity
"""

from __future__ import annotations

import pickle
import shutil
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from importlib import resources as pkg_resources
from pathlib import Path
from typing import NamedTuple

import click
import pandas as pd

from sportstradamus import data
from sportstradamus.helpers.io import MODELS_DIR, market_file_slug, model_pickle_path
from sportstradamus.helpers.locks import TRAINING_ARTIFACTS_LOCK_FILE, hold_for_process
from sportstradamus.training.model_strategy import (
    BASE_STRUCTURAL_STRATEGY,
    CORNER_FINGERPRINT_CSV_COLUMN,
    MODEL_STRATEGY_MODEL_KEY,
    STRATEGY_SIGNATURE_CSV_COLUMN,
    build_artifact_identity,
    corner_fingerprint,
    get_strategy,
    parse_controls,
    resolve_report_identity,
    strategy_controls,
    validate_strategy_frame,
)
from sportstradamus.training.model_strategy.sweep import (
    NOMINEE_LEDGER_PATH,
    STRATEGY_RESEARCH_BOARD,
)

_TEST_SETS_DIR = pkg_resources.files(data) / "test_sets"

# The rotation only ever moves the signature and the fingerprint derived from it. A rebuild that
# moves anything else means the stored artifact disagrees with the registry about something the
# narrowed signature still covers, so it is drift, not rot.
_RESTAMPED_FIELDS = frozenset({"signature", "corner_fingerprint"})


class _IdentityColumns(NamedTuple):
    """Board/ledger CSV column names for the four identity fields, by role not position.

    Board and ledger spell the same identity under different column names.
    """

    signature: str
    fingerprint: str
    controls: str
    matrix: str


_BOARD_COLUMNS = _IdentityColumns(
    "strategy_signature", "corner_fingerprint", "controls_json", "matrix_hash"
)
_LEDGER_COLUMNS = _IdentityColumns(
    "strategy_signature",
    "strategy_corner_fingerprint",
    "strategy_controls_json",
    "strategy_matrix_hash",
)

# 15 min: long enough to outlast a meditate cell retrain or a confirm-walk step so the
# operator doesn't have to retry by hand; a lock stuck longer than that is a genuine hang.
_FLOCK_TIMEOUT_S = 900


def _implementation_moved(stored: Mapping) -> str:
    """Why the stored identity's registry-owned fields are no longer current, or ``""``.

    Deliberately does not check the cell: a pickle or row naming the wrong league/market is
    caught downstream, where the rebuilt identity moves a field outside :data:`_RESTAMPED_FIELDS`.
    """
    spec = get_strategy(str(stored["strategy_slug"]))
    structural = spec.slug if spec.is_structural else BASE_STRUCTURAL_STRATEGY
    for label, was, now in (
        ("structural_strategy", stored["structural_strategy"], structural),
        ("implementation_version", stored["implementation_version"], spec.implementation_version),
        (
            "artifact_schema_version",
            stored["artifact_schema_version"],
            spec.artifact_schema_version,
        ),
    ):
        if was != now:
            return f"{label} {was!r} -> {now!r}"
    if parse_controls(str(stored["controls_json"])) not in strategy_controls(spec):
        return "controls are no longer a registered corner"
    return ""


def _restamp_cell(league: str, market: str, backup: Path, dry_run: bool) -> str:
    """Restamp one cell's pickle and paired test-set CSV; returns why not, or ``""`` on success.

    Rebuilds through the same :func:`build_artifact_identity` the training pipeline stamps with,
    so status and split fingerprint come from the artifact's own structural payload rather than
    being copied forward blind. Both artifacts are rebuilt and revalidated in memory before either
    is written, so a cell that cannot be made valid leaves nothing half-migrated behind.
    """
    path = model_pickle_path(league, market)
    with open(path, "rb") as infile:
        filedict = pickle.load(infile)
    stored = filedict[MODEL_STRATEGY_MODEL_KEY]
    moved = _implementation_moved(stored)
    if moved:
        return moved
    rebuilt = asdict(
        build_artifact_identity(
            str(stored["strategy_slug"]),
            league,
            market,
            parse_controls(str(stored["controls_json"])),
            filedict.get("structural_calibration"),
            matrix_hash=str(stored["matrix_hash"]),
        )
    )
    rotated = {key for key, value in rebuilt.items() if stored.get(key) != value}
    if not rotated:
        return "already current"
    if not rotated <= _RESTAMPED_FIELDS:
        return f"identity moved beyond the signature: {sorted(rotated - _RESTAMPED_FIELDS)}"

    filedict[MODEL_STRATEGY_MODEL_KEY] = rebuilt
    resolve_report_identity(filedict, league, market)
    slug = market_file_slug(league, market)
    csv_path = Path(str(_TEST_SETS_DIR / f"{slug}.csv"))
    frame = pd.read_csv(csv_path) if csv_path.is_file() else None
    if frame is not None:
        frame[STRATEGY_SIGNATURE_CSV_COLUMN] = rebuilt["signature"]
        frame[CORNER_FINGERPRINT_CSV_COLUMN] = rebuilt["corner_fingerprint"]
        validate_strategy_frame(frame, league=league, market=market)
    if dry_run:
        return ""

    shutil.copy2(path, backup / f"{slug}.mdl")
    with open(path, "wb") as outfile:
        pickle.dump(filedict, outfile, -1)
    if frame is not None:
        shutil.copy2(csv_path, backup / f"{slug}.csv")
        frame.to_csv(csv_path, index=False)
    return ""


def _restamp_rows(path: Path, columns: _IdentityColumns, backup: Path, dry_run: bool) -> str:
    """Restamp a board or ledger CSV's identity columns; returns a one-line count summary.

    Rows whose implementation moved keep their rotated-away signature and stay correctly out of
    scope — the sweep's resume cache and the ledger's discount pairing both key on it.
    """
    signature_column, fingerprint_column, controls_column, matrix_column = columns
    frame = pd.read_csv(path, low_memory=False)
    restamped = 0
    for index, row in frame.iterrows():
        stored = {
            "strategy_slug": row["strategy_slug"],
            "structural_strategy": row["structural_strategy"],
            "implementation_version": row["strategy_implementation_version"],
            "artifact_schema_version": row["artifact_schema_version"],
            "controls_json": row[controls_column],
        }
        if _implementation_moved(stored):
            continue
        spec = get_strategy(str(row["strategy_slug"]))
        signature = spec.canonical_signature
        if row[signature_column] == signature:
            continue
        controls = parse_controls(str(row[controls_column]))
        frame.at[index, signature_column] = signature
        frame.at[index, fingerprint_column] = corner_fingerprint(
            spec, controls, str(row[matrix_column])
        )
        restamped += 1
    if not dry_run and restamped:
        shutil.copy2(path, backup / path.name)
        frame.to_csv(path, index=False)
    return f"{path.name}: {restamped}/{len(frame)} rows restamped"


@click.command()
@click.option("--dry-run", is_flag=True, help="Report what would change; write nothing.")
def main(dry_run: bool) -> None:
    """Rewrite rotated strategy signatures in place wherever the implementation is unchanged."""
    if not dry_run:
        hold_for_process(
            TRAINING_ARTIFACTS_LOCK_FILE,
            timeout_s=_FLOCK_TIMEOUT_S,
            label="restamp-identity",
            contention_hint="a meditate or confirm walk is writing the artifact set",
        )
    backup = (
        NOMINEE_LEDGER_PATH.parent
        / "logs"
        / "restamp_identity"
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    if not dry_run:
        backup.mkdir(parents=True, exist_ok=True)
        click.echo(f"backups -> {backup}")

    restamped, current, drifted = [], [], []
    for path in sorted(Path(str(MODELS_DIR)).glob("*.mdl")):
        league, _, slug = path.stem.partition("_")
        market = slug.replace("-", " ")
        reason = _restamp_cell(league, market, backup, dry_run)
        if not reason:
            restamped.append(f"{league} {market}")
        elif reason == "already current":
            current.append(f"{league} {market}")
        else:
            drifted.append(f"{league} {market}: {reason}")

    click.echo(f"\ncells restamped: {len(restamped)}")
    for line in restamped:
        click.echo(f"  {line}")
    click.echo(f"\ncells already current: {len(current)}")
    click.echo(f"cells needing a retrain: {len(drifted)}")
    for line in drifted:
        click.echo(f"  {line}")

    click.echo("")
    click.echo(_restamp_rows(STRATEGY_RESEARCH_BOARD, _BOARD_COLUMNS, backup, dry_run))
    click.echo(_restamp_rows(NOMINEE_LEDGER_PATH, _LEDGER_COLUMNS, backup, dry_run))
    if dry_run:
        click.echo("\n--dry-run: nothing written")


if __name__ == "__main__":
    main()
