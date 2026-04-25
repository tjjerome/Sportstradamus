"""Back-compat shim: ``sportstradamus.stats`` → ``stats`` package.

The stats monolith was split into ``sportstradamus.stats.*`` in Phase 5 of
the maintainability refactor. Python's package-takes-precedence rule means
the ``stats/`` directory is loaded instead of this file at runtime, so this
shim is never actually imported. It is retained only as documentation of the
split and can be deleted once callers are verified.
"""
# This file is shadowed by the stats/ package and never loaded.
# See src/sportstradamus/stats/__init__.py for the public API.
