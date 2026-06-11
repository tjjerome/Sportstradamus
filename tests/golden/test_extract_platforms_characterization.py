"""Characterization pin for ``dashboard_data._extract_platforms``.

Pins the unique-platform extraction before the nesting-sweep flatten: the
normalized ``Offers`` path (per-offer tuple slot 2, skipping non-list entries,
sub-3-length tuples, and falsy platform slots), the legacy ``Platform``-column
fallback, and the empty result when neither column is present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sportstradamus.dashboard.data import _extract_platforms


def test_extract_platforms_from_offers():
    history = pd.DataFrame(
        {
            "Offers": [
                [(18.5, 1.0, "Underdog"), (20.5, 2.0, "Sleeper")],
                [(8.5, 1.0, "Underdog")],
                np.nan,  # dropped by dropna
                [(1.5, 1.0, "")],  # falsy platform slot -> skipped
                [(2.5, 1.0)],  # len < 3 -> skipped
                "not-a-list",  # non-list entry -> skipped
            ]
        }
    )
    assert _extract_platforms(history) == ["Sleeper", "Underdog"]


def test_extract_platforms_platform_column_fallback():
    history = pd.DataFrame({"Platform": ["Underdog", "Sleeper", "Underdog", np.nan]})
    assert _extract_platforms(history) == ["Sleeper", "Underdog"]


def test_extract_platforms_offers_takes_precedence():
    history = pd.DataFrame(
        {
            "Offers": [[(5.5, 1.0, "PrizePicks")]],
            "Platform": ["Underdog"],
        }
    )
    assert _extract_platforms(history) == ["PrizePicks"]


def test_extract_platforms_neither_column():
    assert _extract_platforms(pd.DataFrame({"Other": [1, 2]})) == []
