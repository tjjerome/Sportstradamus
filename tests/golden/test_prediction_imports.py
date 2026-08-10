"""Import-surface guards for the prediction package.

Phase 3 Step 3 split ``beam_search_parlays`` into ``prediction/parlay.py``.
These tests pin the canonical import path and the package-level re-export
so future refactors can't silently break either.
"""

from __future__ import annotations


def test_beam_search_parlays_canonical_import() -> None:
    """Canonical path advertised in docs/ARCHITECTURE.md §Package Map."""
    from sportstradamus.prediction.parlay import beam_search_parlays

    assert callable(beam_search_parlays)


def test_beam_search_parlays_package_reexport() -> None:
    """Package-level re-export keeps ``from prediction import ...`` working."""
    from sportstradamus.prediction import beam_search_parlays

    assert callable(beam_search_parlays)


def test_parlay_helpers_canonical_import() -> None:
    """Parlay helpers live in payouts.py / joint.py — correlation.py is not their home."""
    from sportstradamus.prediction.joint import _nearest_psd
    from sportstradamus.prediction.payouts import (
        expected_payout_with_pushes,
        payout_curve_for,
    )

    assert callable(payout_curve_for)
    assert callable(_nearest_psd)
    assert callable(expected_payout_with_pushes)
