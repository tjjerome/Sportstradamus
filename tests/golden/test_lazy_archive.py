"""LazyArchive must defer DuckDB lock acquisition until first attribute access.

The production regression this guards: the Streamlit dashboard transitively
imports ``sportstradamus.stats``, which historically did
``archive = Archive()`` at module top. Because DuckDB takes an exclusive
file lock for the lifetime of a read-write connection, the long-lived
dashboard process held the archive lock forever — every overlapping cron
job (prophecize, confer, close-lines) then crashed on import.

These tests pin the LazyArchive contract: constructing the proxy and
importing modules that bind one MUST NOT open the DuckDB file. Only an
actual attribute read (``archive.get_ev``, ``archive.default_totals``,
etc.) is allowed to acquire the lock.
"""

from __future__ import annotations

import contextlib


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


def test_lazy_archive_construction_does_not_open_db(tmp_path, monkeypatch):
    archive_db = tmp_path / "archive.duckdb"
    monkeypatch.setenv("SPORTSTRADAMUS_ARCHIVE_DB", str(archive_db))
    _reset_archive_singleton()
    try:
        from sportstradamus.helpers.archive import Archive, LazyArchive

        proxy = LazyArchive()
        assert Archive._instance is None, "LazyArchive() must not construct the Archive singleton"
        assert not archive_db.exists(), (
            "LazyArchive() must not create the DuckDB file at construction time"
        )

        # First attribute access realizes the singleton and opens the DB.
        _ = proxy.default_totals
        assert Archive._instance is not None
        assert archive_db.exists()
    finally:
        _reset_archive_singleton()
        monkeypatch.delenv("SPORTSTRADAMUS_ARCHIVE_DB", raising=False)


def test_stats_module_import_does_not_open_archive(tmp_path, monkeypatch):
    """``from sportstradamus.stats.base import archive`` must not acquire the DuckDB lock.

    This is the exact import pattern every Stats subclass uses (mlb.py,
    nba.py, nfl.py, etc.) and the path the dashboard hits transitively.
    """
    archive_db = tmp_path / "archive.duckdb"
    monkeypatch.setenv("SPORTSTRADAMUS_ARCHIVE_DB", str(archive_db))
    _reset_archive_singleton()
    try:
        import importlib

        import sportstradamus.helpers.archive as A
        from sportstradamus.stats import base

        importlib.reload(base)

        assert A.Archive._instance is None, (
            "Importing sportstradamus.stats.base must not construct Archive"
        )
        assert type(base.archive).__name__ == "LazyArchive", (
            f"base.archive should be LazyArchive, got {type(base.archive).__name__}"
        )
        assert not archive_db.exists(), "Importing the module must not create the DuckDB file"
    finally:
        _reset_archive_singleton()
        monkeypatch.delenv("SPORTSTRADAMUS_ARCHIVE_DB", raising=False)


def test_lazy_archive_forwards_attribute_access(tmp_path, monkeypatch):
    """Every attribute read on the proxy must hit the live Archive singleton."""
    archive_db = tmp_path / "archive.duckdb"
    monkeypatch.setenv("SPORTSTRADAMUS_ARCHIVE_DB", str(archive_db))
    _reset_archive_singleton()
    try:
        from sportstradamus.helpers.archive import Archive, LazyArchive

        proxy = LazyArchive()
        # Touch default_totals on both — must return the same dict object.
        live = Archive()
        assert proxy.default_totals is live.default_totals
        # Method invocation through the proxy hits the live method.
        assert proxy.get_moneyline("NBA", "2026-01-01", "LAL") == 0.5
    finally:
        _reset_archive_singleton()
        monkeypatch.delenv("SPORTSTRADAMUS_ARCHIVE_DB", raising=False)
