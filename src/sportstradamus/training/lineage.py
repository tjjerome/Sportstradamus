"""Deterministic matrix manifest construction for quarantined full rebuilds."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

MATRIX_BUILDER_VERSION = "matrix-lineage-v1"
MATRIX_SCHEMA_VERSION = 1
MATRIX_TRIM_SEED = 42


def file_sha(path: Path) -> str | None:
    """SHA256 hex digest of a file's contents, or ``None`` if it does not exist."""
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _frame_sha(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(frame.columns), separators=(",", ":")).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def code_revision(repo_root: Path) -> str:
    """Return the checked-out revision without consulting a remote."""
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def validate_matrix_manifest(matrix_path: Path, matrix: pd.DataFrame) -> dict:
    """Validate a frozen parquet against its adjacent lineage manifest."""
    manifest_path = matrix_path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"frozen matrix manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    checks = {
        "builder_version": MATRIX_BUILDER_VERSION,
        "schema_version": MATRIX_SCHEMA_VERSION,
        "row_count": len(matrix),
        "feature_schema": list(matrix.columns),
        "matrix_sha256": file_sha(matrix_path),
    }
    mismatches = [name for name, expected in checks.items() if manifest.get(name) != expected]
    if mismatches:
        raise ValueError(
            f"frozen matrix manifest mismatch for {matrix_path}: {', '.join(mismatches)}"
        )
    return manifest


def write_matrix_manifest(
    matrix_path: Path,
    matrix: pd.DataFrame,
    stat_data,
    *,
    league: str,
    market: str,
    cutoff_date,
    repo_root: Path,
) -> Path:
    """Write lineage beside a quarantined parquet after its bytes are final."""
    config_root = repo_root / "src/sportstradamus/data/config"
    external_columns = [column for column in matrix if column.endswith("_asof")]
    snapshot_hashes = {}
    if league == "NFL":
        from sportstradamus.stats import nfl_fp_team_weekly, nfl_fp_weekly

        snapshot_hashes = {
            **nfl_fp_weekly.consumed_snapshot_hashes(),
            **nfl_fp_team_weekly.consumed_snapshot_hashes(),
        }
    manifest = {
        "builder_version": MATRIX_BUILDER_VERSION,
        "schema_version": MATRIX_SCHEMA_VERSION,
        "code_revision": code_revision(repo_root),
        "as_of": datetime.now(UTC).isoformat(),
        "league": league,
        "market": market,
        "query_hash": hashlib.sha256(
            json.dumps(
                {"league": league, "market": market, "cutoff_date": str(cutoff_date)},
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "input_hashes": {"gamelog": _frame_sha(stat_data.gamelog)},
        "external_snapshot_inventory": {
            "feature_coverage": {
                column: int(matrix[column].notna().sum()) for column in external_columns
            },
            "file_hashes": dict(sorted(snapshot_hashes.items())),
        },
        "dependency_hashes": dict(sorted(stat_data.model_dependency_inventory.items())),
        "config_hashes": {
            name: file_sha(config_root / name)
            for name in ("stat_meta.json", "feature_filter.json", "book_weights.json")
        },
        "seed": MATRIX_TRIM_SEED,
        "row_count": len(matrix),
        "feature_schema": list(matrix.columns),
        "matrix_sha256": file_sha(matrix_path),
    }
    manifest_path = matrix_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path
