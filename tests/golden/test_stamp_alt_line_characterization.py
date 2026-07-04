"""Pin for ``prediction.cli._stamp_alt_line`` (P8 Task 0.7).

Tolerance for flagging ``Alt Line`` is picked per-row from the cell's
distribution family: count stats (NegBin / ZINB / Poisson) move in 0.5 steps
so get the tighter ``_ALT_LINE_TOL_COUNT = 0.75`` tolerance; everything else
(continuous/yardage stats) gets ``_ALT_LINE_TOL_CONTINUOUS = 2.5``. A NaN
``Consensus Line`` (no archive line yet) always reads ``Alt Line = False``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sportstradamus.prediction import cli
from sportstradamus.prediction.cli import (
    _ALT_LINE_TOL_CONTINUOUS,
    _ALT_LINE_TOL_COUNT,
    _stamp_alt_line,
)


@pytest.fixture(autouse=True)
def _patch_stat_dist(monkeypatch):
    monkeypatch.setattr(
        cli,
        "stat_dist",
        {
            "NBA": {"assists": "NegBin", "points": "SkewNormal"},
            "NFL": {"rushing_yards": "Gamma"},
        },
    )


def test_tolerances_are_the_expected_values():
    assert _ALT_LINE_TOL_COUNT == 0.75
    assert _ALT_LINE_TOL_CONTINUOUS == 2.5


def _row(league, market, line, consensus_line):
    return {"League": league, "Market": market, "Line": line, "Consensus Line": consensus_line}


def test_count_market_inside_tolerance_not_flagged():
    offers = pd.DataFrame([_row("NBA", "assists", 5.5, 5.0)])  # diff 0.5 < 0.75
    out = _stamp_alt_line(offers)
    assert bool(out.iloc[0]["Alt Line"]) is False


def test_count_market_outside_tolerance_flagged():
    offers = pd.DataFrame([_row("NBA", "assists", 6.5, 5.0)])  # diff 1.5 > 0.75
    out = _stamp_alt_line(offers)
    assert bool(out.iloc[0]["Alt Line"]) is True


def test_continuous_market_inside_tolerance_not_flagged():
    offers = pd.DataFrame([_row("NFL", "rushing_yards", 62.0, 60.0)])  # diff 2.0 < 2.5
    out = _stamp_alt_line(offers)
    assert bool(out.iloc[0]["Alt Line"]) is False


def test_continuous_market_outside_tolerance_flagged():
    offers = pd.DataFrame([_row("NFL", "rushing_yards", 65.0, 60.0)])  # diff 5.0 > 2.5
    out = _stamp_alt_line(offers)
    assert bool(out.iloc[0]["Alt Line"]) is True


def test_nan_consensus_line_never_flagged_regardless_of_dist():
    offers = pd.DataFrame(
        [
            _row("NBA", "assists", 6.5, np.nan),
            _row("NFL", "rushing_yards", 90.0, np.nan),
        ]
    )
    out = _stamp_alt_line(offers)
    assert not out["Alt Line"].any()


def test_market_with_unknown_dist_routes_to_continuous_tolerance():
    """League/market absent from stat_dist -> dist is None -> not in _COUNT_DISTS
    -> continuous tolerance, same as an explicit SkewNormal/Gamma cell."""
    offers = pd.DataFrame([_row("NBA", "points", 22.0, 20.0)])  # diff 2.0, SkewNormal in fixture
    out = _stamp_alt_line(offers)
    assert bool(out.iloc[0]["Alt Line"]) is False  # within continuous tol (2.5)
    offers_far = pd.DataFrame([_row("NBA", "points", 24.0, 20.0)])  # diff 4.0
    out_far = _stamp_alt_line(offers_far)
    assert bool(out_far.iloc[0]["Alt Line"]) is True
