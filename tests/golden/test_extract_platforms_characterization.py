"""Characterization pin for ``dashboard.data._extract_platforms``.

Pins the unique-platform extraction from the flat ``Platform`` column: dedup +
sort, NaN dropped, and the empty result when the column is absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sportstradamus.dashboard.data import _extract_platforms


def test_extract_platforms_from_platform_column():
    history = pd.DataFrame({"Platform": ["Underdog", "Sleeper", "Underdog", np.nan]})
    assert _extract_platforms(history) == ["Sleeper", "Underdog"]


def test_extract_platforms_no_platform_column():
    assert _extract_platforms(pd.DataFrame({"Other": [1, 2]})) == []
