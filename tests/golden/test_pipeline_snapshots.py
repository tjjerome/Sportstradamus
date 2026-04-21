"""Pipeline-level golden tests — placeholders.

These tests would assert that a small end-to-end pipeline run produces the
same output before and after refactoring. They are skipped because they
require fixture data we do not yet check in:

* ``test_meditate_single_market`` — needs a small cached training matrix
  (e.g., one market for one league with ~200 rows) committed under
  ``tests/golden/fixtures/training/``.
* ``test_confer_fixture`` — needs a recorded HTTP response from the odds
  API for one league + one market, plus a stub ``creds/keys.json``.
* ``test_prophecize_export`` — needs the union of the above plus a mocked
  Google Sheets client.

These will be filled in alongside Phases 3–5 once the modules they exercise
are touched. See the refactor plan at
/home/trevor/.claude/plans/my-codebase-has-gotten-toasty-bonbon.md.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Needs cached training-matrix fixture — see module docstring.")
def test_meditate_single_market_snapshot() -> None:
    raise NotImplementedError


@pytest.mark.skip(reason="Needs recorded HTTP fixture for odds API — see module docstring.")
def test_confer_archive_snapshot() -> None:
    raise NotImplementedError


@pytest.mark.skip(reason="Needs mocked Sheets client + stats fixture — see module docstring.")
def test_prophecize_export_snapshot() -> None:
    raise NotImplementedError
