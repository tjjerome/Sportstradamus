"""Pin the CLAUDE.md hard rule: dashboard process must never hold the archive lock.

The Streamlit dashboard is the only long-lived process in the system.
DuckDB holds an exclusive file lock for the entire lifetime of any
read-write connection, so any archive connection the dashboard opens —
even accidentally via a module-level ``Archive()`` in a transitively-
imported module — blocks every cron job that needs to write odds.

This regression has bitten production twice already (first when
``stats/base.py`` instantiated ``Archive`` at import, then again when
the ``prediction/*`` modules did the same). The defensive proxy lives in
``helpers/archive.py`` as ``LazyArchive``; this test pins the rule so a
future refactor cannot silently regress.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import sys
from pathlib import Path

# Project source root: src/sportstradamus/
_PROJECT_SRC = Path(__file__).resolve().parents[2] / "src" / "sportstradamus"

# Dashboard entry surfaces. Add new top-level dashboard modules here when
# they appear under src/sportstradamus/.
_DASHBOARD_MODULES = (
    "sportstradamus.dashboard",
    "sportstradamus.dashboard_app",
    "sportstradamus.dashboard_data",
    "sportstradamus.dashboard_detail",
)


def _reset_archive_singleton() -> None:
    """Clear the Archive singleton and close any open DuckDB connection.

    Connection must be closed before nulling ``_instance``; leaving an open
    handle on the ``tmp_path`` database would cause the OS to refuse deletion
    of the temp directory on teardown on some platforms.
    """
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


def _clear_dashboard_imports() -> None:
    """Drop dashboard + stats modules from sys.modules so re-import is fresh.

    Other tests in the session may have already imported these modules, in
    which case ``importlib.import_module`` is a no-op and would never trigger
    the module-level code we want to observe. Clearing the cache forces a
    fresh execution.
    """
    prefixes = (
        "sportstradamus.dashboard",
        "sportstradamus.pages",
        "sportstradamus.stats",
        "sportstradamus.prediction",
    )
    for name in [m for m in list(sys.modules) if m.startswith(prefixes)]:
        del sys.modules[name]


def _import_page_file(path: Path) -> None:
    """Import a Streamlit page by filesystem path.

    Pages use emoji in their filenames (``1_🎯_Predictions_Today.py``)
    which are not valid Python identifiers, so they must be loaded via
    ``importlib.util.spec_from_file_location`` rather than a normal import.

    Streamlit-runtime side effects (``st.set_page_config`` outside a session,
    ``@st.cache_data`` registration) may raise when executed without a
    running Streamlit; we suppress those because the assertion we care about
    is purely the Archive singleton, not whether the page renders.
    """
    spec = importlib.util.spec_from_file_location(f"_pinned_page_{path.stem}", path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    with contextlib.suppress(Exception):
        spec.loader.exec_module(module)


def test_dashboard_imports_do_not_construct_archive(tmp_path, monkeypatch) -> None:
    """Importing every dashboard surface must leave ``Archive._instance`` is ``None``.

    Codifies the CLAUDE.md "Hard rules" entry: the dashboard never opens
    DuckDB. If this test fails, trace which module instantiates ``Archive()``
    at top level and switch it to ``LazyArchive`` (see
    ``sportstradamus.helpers.archive``).
    """
    monkeypatch.setenv("SPORTSTRADAMUS_ARCHIVE_DB", str(tmp_path / "archive.duckdb"))
    _reset_archive_singleton()
    _clear_dashboard_imports()
    try:
        from sportstradamus.helpers.archive import Archive

        assert Archive._instance is None, "test fixture failed to reset singleton"

        for name in _DASHBOARD_MODULES:
            with contextlib.suppress(Exception):
                importlib.import_module(name)

        for page_path in sorted((_PROJECT_SRC / "pages").glob("*.py")):
            if page_path.name == "__init__.py":
                continue
            _import_page_file(page_path)

        assert Archive._instance is None, (
            "Dashboard imports constructed the Archive singleton, which holds "
            "DuckDB's exclusive file lock for the lifetime of the dashboard "
            "process and blocks every cron job. Trace which module ran "
            "Archive() at top level and switch it to LazyArchive — see the "
            "Hard rules section of CLAUDE.md and the LazyArchive docstring "
            "in helpers/archive.py."
        )
    finally:
        _reset_archive_singleton()
        monkeypatch.delenv("SPORTSTRADAMUS_ARCHIVE_DB", raising=False)
