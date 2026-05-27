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
import time
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
        "CREATE TABLE lines (league TEXT, market TEXT, game_date DATE, entity TEXT, line DOUBLE)"
    )
    con.execute("CHECKPOINT")
    con.close()

    victim_script = (
        "import duckdb, os\n"
        f"con = duckdb.connect({str(tmp_path / 'victim.duckdb')!r})\n"
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

        recovery_warnings = [w for w in caught if "Discarded stale DuckDB WAL" in str(w.message)]
        assert len(recovery_warnings) == 1, (
            f"expected exactly one recovery warning, got {[str(w.message) for w in caught]}"
        )

        quarantined = [
            p for p in tmp_path.iterdir() if p.name.startswith("archive.duckdb.wal.corrupt-")
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


def test_archive_retries_when_another_process_holds_lock(tmp_path, monkeypatch):
    """A peer holding the DuckDB lock must trigger bounded retry, not immediate failure."""

    archive_db = tmp_path / "archive.duckdb"
    ready_file = tmp_path / "peer.ready"
    release_file = tmp_path / "peer.release"

    peer_script = (
        "import duckdb, pathlib, time\n"
        f"con = duckdb.connect({str(archive_db)!r})\n"
        f"pathlib.Path({str(ready_file)!r}).touch()\n"
        f"release = pathlib.Path({str(release_file)!r})\n"
        "while not release.exists():\n"
        "    time.sleep(0.05)\n"
        "con.close()\n"
    )
    peer = subprocess.Popen([sys.executable, "-c", peer_script])
    try:
        deadline = time.monotonic() + 10
        while not ready_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready_file.exists(), "peer process never acquired the lock"

        monkeypatch.setenv("SPORTSTRADAMUS_ARCHIVE_DB", str(archive_db))
        monkeypatch.setenv("SPORTSTRADAMUS_ARCHIVE_LOCK_WAIT_SECONDS", "30")
        _reset_archive_singleton()

        import threading

        from sportstradamus.helpers.archive import Archive

        # Release the lock shortly after Archive() starts retrying.
        def release_peer():
            time.sleep(3.0)
            release_file.touch()

        releaser = threading.Thread(target=release_peer)
        releaser.start()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                archive = Archive()
            lock_warnings = [w for w in caught if "is locked by another process" in str(w.message)]
            assert lock_warnings, (
                f"expected at least one lock-retry warning, got {[str(w.message) for w in caught]}"
            )
            assert archive._connection.execute("SELECT COUNT(*) FROM odds").fetchone() == (0,)
        finally:
            releaser.join()
            _reset_archive_singleton()
            monkeypatch.delenv("SPORTSTRADAMUS_ARCHIVE_LOCK_WAIT_SECONDS", raising=False)
            monkeypatch.delenv("SPORTSTRADAMUS_ARCHIVE_DB", raising=False)
    finally:
        release_file.touch()
        peer.wait(timeout=10)


def test_archive_lock_retry_gives_up_after_budget(tmp_path, monkeypatch):
    """When the lock is never released, retry must surface the original IOException."""

    archive_db = tmp_path / "archive.duckdb"
    ready_file = tmp_path / "peer.ready"
    release_file = tmp_path / "peer.release"

    peer_script = (
        "import duckdb, pathlib, time\n"
        f"con = duckdb.connect({str(archive_db)!r})\n"
        f"pathlib.Path({str(ready_file)!r}).touch()\n"
        f"release = pathlib.Path({str(release_file)!r})\n"
        "while not release.exists():\n"
        "    time.sleep(0.05)\n"
        "con.close()\n"
    )
    peer = subprocess.Popen([sys.executable, "-c", peer_script])
    try:
        deadline = time.monotonic() + 10
        while not ready_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready_file.exists(), "peer process never acquired the lock"

        monkeypatch.setenv("SPORTSTRADAMUS_ARCHIVE_DB", str(archive_db))
        # 1s budget — peer won't release in time, so we must fail fast.
        monkeypatch.setenv("SPORTSTRADAMUS_ARCHIVE_LOCK_WAIT_SECONDS", "1")
        _reset_archive_singleton()

        from sportstradamus.helpers.archive import Archive

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(duckdb.IOException, match="Could not set lock on file"):
                Archive()
    finally:
        release_file.touch()
        peer.wait(timeout=10)
        _reset_archive_singleton()
        monkeypatch.delenv("SPORTSTRADAMUS_ARCHIVE_LOCK_WAIT_SECONDS", raising=False)
        monkeypatch.delenv("SPORTSTRADAMUS_ARCHIVE_DB", raising=False)
