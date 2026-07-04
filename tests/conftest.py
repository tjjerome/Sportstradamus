"""Repo-wide pytest collection config.

``tests/deprecated/`` mirrors ``src/deprecated/``: tests for archived code
that has no live caller. Excluded from bare ``pytest`` / ``pytest tests/``
collection (kept out of the active suite) but still directly runnable —
``pytest tests/deprecated/`` — so the archived behavior stays verifiable.
See ``src/deprecated/README.md``.
"""

collect_ignore = ["deprecated"]
