"""Shared fixtures for the integration suite.

The integration suite exercises the full ``confer -> meditate -> prophecize``
pipeline on cached fixtures. It is opt-in via ``pytest -m integration``;
``pytest tests/golden/`` does not collect these tests.
"""

from __future__ import annotations

import contextlib
import importlib.resources as pkg_resources
from pathlib import Path

import pytest

from sportstradamus import data

INTEGRATION_DIR = Path(__file__).parent
FIXTURES_DIR = INTEGRATION_DIR / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Absolute path to the committed integration-suite fixtures."""
    return FIXTURES_DIR


@pytest.fixture
def reset_archive_singleton():
    """Force the next ``Archive()`` to re-initialize against the current cwd.

    ``Archive`` is a process-singleton that opens its DuckDB file once on
    first instantiation. Because several production modules call
    ``Archive()`` at import time, the singleton is bound before our test
    has a chance to ``chdir`` into a tmp working directory. Closing the
    existing connection and resetting ``_initialized`` lets the next call
    rerun ``__init__`` against the new cwd.
    """
    from sportstradamus.helpers.archive import Archive

    def _reset(inst):
        if inst is None:
            return
        con = getattr(inst, "_connection", None)
        if con is not None:
            with contextlib.suppress(Exception):
                con.close()
        inst._initialized = False

    _reset(Archive._instance)
    yield
    _reset(Archive._instance)


# Files in ``src/sportstradamus/data/`` that ``meditate`` rewrites mid-run.
# We snapshot their bytes at test setup and restore on teardown so the
# integration suite leaves no on-disk side effects.
_DATA_FILES_TO_PROTECT = ("book_weights.json", "upcoming_events.json")


def _data_dir_real_path() -> Path:
    """Resolve ``sportstradamus.data`` to a concrete filesystem ``Path``.

    ``importlib.resources.files`` may return a ``MultiplexedPath`` here,
    which is path-like but does not survive a ``Path(str(...))``
    round-trip. Walking one entry via ``iterdir`` gives us the real
    parent path on disk.
    """
    files = pkg_resources.files(data)
    for entry in files.iterdir():
        return Path(str(entry)).parent
    msg = "sportstradamus.data package contains no resources"
    raise RuntimeError(msg)


@pytest.fixture
def preserve_data_files():
    """Snapshot/restore mutable JSON config files in the ``data`` package.

    ``meditate`` rewrites ``book_weights.json``; ``confer --fixture-dir``
    rewrites ``upcoming_events.json``. The smoke test's stubs prevent
    those writes from carrying real production data, but the underlying
    file truncation still mutates the on-disk bytes — this fixture
    captures them up front and writes them back on teardown so the
    integration suite leaves no on-disk side effects.
    """
    data_root = _data_dir_real_path()
    snapshots = {
        name: (data_root / name).read_bytes()
        for name in _DATA_FILES_TO_PROTECT
        if (data_root / name).is_file()
    }
    yield
    for name, payload in snapshots.items():
        (data_root / name).write_bytes(payload)
