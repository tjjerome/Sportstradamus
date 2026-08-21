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
import pkgutil
import sys
from types import ModuleType

import pandas as pd


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


_CLEARED_PREFIXES = (
    "sportstradamus.dashboard",
    "sportstradamus.stats",
    "sportstradamus.prediction",
)


def _clear_dashboard_imports() -> dict[str, object]:
    """Drop dashboard + stats modules from sys.modules so re-import is fresh.

    Other tests in the session may have already imported these modules, in
    which case ``importlib.import_module`` is a no-op and would never trigger
    the module-level code we want to observe. Clearing the cache forces a
    fresh execution.

    Returns the removed ``{name: module}`` mapping so the caller can restore it
    on teardown via :func:`_restore_modules`. Without that restore, the
    freshly-imported (or now-missing) entries leak into later tests: any test
    that monkeypatches a symbol on one of these modules (e.g.
    ``prediction.correlation.find_correlation``) would patch a different module
    object than the one a fresh ``from ... import`` resolves, silently running
    the real function instead of the stub.
    """
    removed = {
        name: sys.modules[name] for name in list(sys.modules) if name.startswith(_CLEARED_PREFIXES)
    }
    for name in removed:
        del sys.modules[name]
    return removed


def _restore_modules(removed: dict[str, object]) -> None:
    """Undo :func:`_clear_dashboard_imports`: drop fresh re-imports, restore originals.

    Restoring ``sys.modules`` alone is not enough: the fresh-import phase also
    re-bound each fresh submodule as an *attribute* on its parent package (e.g.
    ``sportstradamus.prediction`` on the top package — the top package survives
    the clear, so its attribute now points at the fresh, about-to-be-discarded
    module object). ``monkeypatch.setattr`` with a dotted-string target resolves
    through those attributes, not ``sys.modules``, while a plain ``from x import
    y`` resolves through ``sys.modules`` — leaving them divergent means a later
    test patches a dead module while the code under test imports the real one.
    So re-point every parent attribute at the restored original, and drop
    attributes whose module was fresh-only (nothing to restore).
    """
    fresh = [m for m in list(sys.modules) if m.startswith(_CLEARED_PREFIXES)]
    for name in fresh:
        del sys.modules[name]
    sys.modules.update(removed)
    for name in set(fresh) | set(removed):
        parent_name, _, child = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is None:
            continue
        if name in sys.modules:
            setattr(parent, child, sys.modules[name])
        elif isinstance(getattr(parent, child, None), ModuleType):
            delattr(parent, child)


def _discover_dashboard_modules() -> list[str]:
    """Auto-discover all importable modules in the sportstradamus.dashboard package."""
    import sportstradamus.dashboard as _pkg

    discovered = []
    for info in pkgutil.walk_packages(
        path=_pkg.__path__,
        prefix=_pkg.__name__ + ".",
        onerror=lambda name: None,
    ):
        discovered.append(info.name)
    return [_pkg.__name__, *discovered]


def test_dashboard_imports_do_not_construct_archive(tmp_path, monkeypatch) -> None:
    """Importing every dashboard surface must leave ``Archive._instance`` is ``None``.

    Codifies the CLAUDE.md "Hard rules" entry: the dashboard never opens
    DuckDB. If this test fails, trace which module instantiates ``Archive()``
    at top level and switch it to ``LazyArchive`` (see
    ``sportstradamus.helpers.archive``).
    """
    monkeypatch.setenv("SPORTSTRADAMUS_ARCHIVE_DB", str(tmp_path / "archive.duckdb"))
    _reset_archive_singleton()
    removed_modules = _clear_dashboard_imports()
    try:
        from sportstradamus.helpers.archive import Archive

        assert Archive._instance is None, "test fixture failed to reset singleton"

        # A Streamlit page is a script: importing it runs its whole body, which
        # calls load_history()/load_parlay_hist() and then per-row CRPS over the
        # real history parquet — ~90s of work irrelevant to this test. Empty the
        # one reader both funnel through so each page short-circuits on its
        # ``if history.empty: st.stop()`` guard. The import statements (where a
        # top-level Archive() would fire) still execute, so the assertion below is
        # unaffected.
        monkeypatch.setattr(
            "sportstradamus.helpers.io.read_parquet_safe",
            lambda *_a, **_k: pd.DataFrame(),
        )

        # Importing a surface module executes its whole script body (Streamlit
        # calls included), so the walk covers exactly what a real page run
        # covers; runtime-context errors are suppressed because the assertion
        # is purely the Archive singleton.
        dashboard_module_names = _discover_dashboard_modules()
        for name in dashboard_module_names:
            with contextlib.suppress(Exception):
                importlib.import_module(name)

        assert Archive._instance is None, (
            "Dashboard imports constructed the Archive singleton, which holds "
            "DuckDB's exclusive file lock for the lifetime of the dashboard "
            "process and blocks every cron job. Trace which module ran "
            "Archive() at top level and switch it to LazyArchive — see the "
            "Hard rules section of CLAUDE.md and the LazyArchive docstring "
            "in helpers/archive.py."
        )
    finally:
        _restore_modules(removed_modules)
        _reset_archive_singleton()
        monkeypatch.delenv("SPORTSTRADAMUS_ARCHIVE_DB", raising=False)
