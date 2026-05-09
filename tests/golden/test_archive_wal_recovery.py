"""Archive() must recover when DuckDB leaves a stale WAL on disk.

DuckDB <=1.1.x can crash on connect if a previous process was killed mid-run
and left a ``.wal`` whose CREATE TABLE entries replay against an
already-checkpointed catalog. The recovery path in
:class:`sportstradamus.helpers.archive.Archive` quarantines the bad WAL and
reconnects so the next pipeline run heals itself.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import duckdb
import pytest


def _build_corrupt_wal(tmp_path: Path) -> Path:
    """Construct an ``archive.duckdb`` whose WAL conflicts on replay.

    The recipe — distilled from the hands-on reproduction:

    1. Build ``archive.duckdb`` with the production schema and CHECKPOINT so
       the catalog is fully on disk and there is no WAL.
    2. In a separate subprocess, open a fresh DB, ``CREATE TABLE odds``,
       commit, then ``os._exit`` — this skips DuckDB's clean-shutdown
       checkpoint and leaves a real WAL that contains a bare CREATE TABLE.
    3. Copy that WAL on top of ``archive.duckdb.wal``. Now opening
       ``archive.duckdb`` reproduces the exact CatalogException seen in
       production.
    """
    archive_db = tmp_path / "archive.duckdb"

    con = duckdb.connect(str(archive_db))
    con.execute(
        "CREATE TABLE odds (league TEXT, market TEXT, game_date DATE, "
        "entity TEXT, book TEXT, ev DOUBLE, sample_ts TIMESTAMP)"
    )
    con.execute(
        "CREATE TABLE lines (league TEXT, market TEXT, game_date DATE, "
        "entity TEXT, line DOUBLE)"
    )
    con.execute("CHECKPOINT")
    con.close()

    victim_script = (
        "import duckdb, os\n"
        f'con = duckdb.connect({str(tmp_path / "victim.duckdb")!r})\n'
        'con.execute("CREATE TABLE odds (league TEXT, market TEXT, '
        "game_date DATE, entity TEXT, book TEXT, ev DOUBLE, "
        'sample_ts TIMESTAMP)")\n'
        "con.commit()\n"
        "os._exit(0)\n"
    )
    subprocess.run([sys.executable, "-c", victim_script], check=True)
    shutil.copy(tmp_path / "victim.duckdb.wal", tmp_path / "archive.duckdb.wal")
    return archive_db


def _reset_archive_singleton() -> None:
    from sportstradamus.helpers.archive import Archive

    inst = Archive._instance
    if inst is None:
        return
    con = getattr(inst, "_connection", None)
    if con is not None:
        with contextlib.suppress(Exception):
            con.close()
    inst._initialized = False
    Archive._instance = None


def test_archive_recovers_from_stale_wal(tmp_path, monkeypatch):
    archive_db = _build_corrupt_wal(tmp_path)
    wal_path = archive_db.with_name(archive_db.name + ".wal")
    assert wal_path.exists(), "fixture must leave a stale WAL in place"

    with pytest.raises(duckdb.CatalogException, match="Failure while replaying WAL"):
        duckdb.connect(str(archive_db))

    monkeypatch.setenv("SPORTSTRADAMUS_ARCHIVE_DB", str(archive_db))
    _reset_archive_singleton()
    try:
        from sportstradamus.helpers.archive import Archive

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            archive = Archive()

        recovery_warnings = [
            w for w in caught if "Discarded stale DuckDB WAL" in str(w.message)
        ]
        assert len(recovery_warnings) == 1, (
            f"expected exactly one recovery warning, got {[str(w.message) for w in caught]}"
        )

        quarantined = [
            p for p in tmp_path.iterdir()
            if p.name.startswith("archive.duckdb.wal.corrupt-")
        ]
        assert len(quarantined) == 1, (
            f"expected one quarantined WAL, got {sorted(p.name for p in tmp_path.iterdir())}"
        )

        assert archive._connection.execute("SELECT COUNT(*) FROM odds").fetchone() == (0,)
    finally:
        _reset_archive_singleton()
        monkeypatch.delenv("SPORTSTRADAMUS_ARCHIVE_DB", raising=False)


def test_unrelated_catalog_exception_propagates(tmp_path, monkeypatch):
    """Non-WAL CatalogExceptions must not be swallowed by the recovery path."""

    bogus_path = tmp_path / "not-a-db"
    bogus_path.write_bytes(b"this is not a duckdb file")

    monkeypatch.setenv("SPORTSTRADAMUS_ARCHIVE_DB", str(bogus_path))
    _reset_archive_singleton()
    try:
        from sportstradamus.helpers.archive import Archive

        with pytest.raises(Exception) as excinfo:
            Archive()
        assert "Failure while replaying WAL" not in str(excinfo.value)
    finally:
        _reset_archive_singleton()
        monkeypatch.delenv("SPORTSTRADAMUS_ARCHIVE_DB", raising=False)


