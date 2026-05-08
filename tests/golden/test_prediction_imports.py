"""Import-surface guards for the prediction package.

Phase 3 Step 3 split ``beam_search_parlays`` into ``prediction/parlay.py``.
These tests pin the canonical import path and the package-level re-export
so future refactors can't silently break either.
"""

from __future__ import annotations


def test_beam_search_parlays_canonical_import() -> None:
    """Canonical path advertised in CONTRIBUTING.md §Package Map."""
    from sportstradamus.prediction.parlay import beam_search_parlays

    assert callable(beam_search_parlays)


def test_beam_search_parlays_package_reexport() -> None:
    """Package-level re-export keeps ``from prediction import ...`` working."""
    from sportstradamus.prediction import beam_search_parlays

    assert callable(beam_search_parlays)


def test_correlation_backcompat_reexports() -> None:
    """Tests and downstream callers still import the helpers from correlation."""
    from sportstradamus.prediction.correlation import (
        _expected_payout_with_pushes,
        _nearest_psd,
        _payout_curve_for,
        beam_search_parlays,
        find_correlation,
    )

    assert callable(beam_search_parlays)
    assert callable(find_correlation)
    assert callable(_payout_curve_for)
    assert callable(_nearest_psd)
    assert callable(_expected_payout_with_pushes)
