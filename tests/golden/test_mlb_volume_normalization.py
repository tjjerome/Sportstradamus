"""Unit tests for MLB structural plate-appearance volume normalization.

The projector is structural (no trained model), so every path is testable
without training: slot->curve mapping, home/away selection, the unresolved-slot
fallback, and the bounded team offense adjustment.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from sportstradamus.stats import mlb


def test_slot_constants_are_well_formed():
    for curve in (mlb.SLOT_PA_HOME, mlb.SLOT_PA_AWAY, mlb.SLOT_STD):
        assert len(curve) == 9
        assert all(v > 0 for v in curve)
    # Leadoff bats more than the nine-hole; away teams bat a full ninth so out-PA them.
    assert mlb.SLOT_PA_HOME[0] > mlb.SLOT_PA_HOME[-1]
    assert mlb.SLOT_PA_AWAY[0] > mlb.SLOT_PA_AWAY[-1]
    assert all(a >= h for a, h in zip(mlb.SLOT_PA_AWAY, mlb.SLOT_PA_HOME, strict=True))
    expected_all = tuple(
        (h + a) / 2 for h, a in zip(mlb.SLOT_PA_HOME, mlb.SLOT_PA_AWAY, strict=True)
    )
    assert expected_all == mlb.SLOT_PA_ALL
    lo, hi = mlb.OFFENSE_ADJ_CLIP
    assert 0 < lo < 1 < hi
    assert 0 < mlb.LG_AVG_OBP < 0.5
    total_weight = mlb.OBP_ADJ_WEIGHT + mlb.MARKET_ADJ_WEIGHT
    assert total_weight == pytest.approx(1.0)
