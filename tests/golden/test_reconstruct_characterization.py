"""Characterization pins for the distribution-math forecasting helpers in
``analysis``: ``reconstruct_prob``, ``reconstruct_quantile``, ``compute_crps_row``.

All three reconstruct a saved prediction's distribution from the row's
(Dist, CV, Model EV, Model Param, Gate) fields and evaluate it. These pins
freeze the exact float outputs across every distribution branch
(NegBin / ZINB / Gamma / ZAGamma), the ``Model Param``-missing CV fallback
(r = 1/cv, alpha = 1/cv**2), the zero-inflation gate adjustment, and the
missing-``Dist`` / missing-``Actual`` NaN guards — so the worst-first
decomposition of ``reconstruct_quantile`` and ``compute_crps_row`` cannot
drift the numbers.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sportstradamus.analysis import (
    compute_crps_row,
    reconstruct_prob,
    reconstruct_quantile,
)


def _row(dist, cv, ev, param, gate, actual=None):
    return {
        "Dist": dist,
        "CV": cv,
        "Projection": ev,
        "Model Param": param,
        "Gate": gate,
        "Actual": actual,
        "Step": 1,
    }


_ROWS = {
    "negbin": _row("NegBin", 0.5, 10.0, 4.0, np.nan, actual=8.0),
    "zinb": _row("ZINB", 0.5, 10.0, 4.0, 0.2, actual=8.0),
    "gamma": _row("Gamma", 0.4, 20.0, 6.25, np.nan, actual=18.0),
    "zagamma": _row("ZAGamma", 0.4, 20.0, 6.25, 0.15, actual=18.0),
    "negbin_paramnone": _row("NegBin", 0.5, 10.0, None, np.nan, actual=8.0),
    "gamma_paramnone": _row("Gamma", 0.4, 20.0, None, np.nan, actual=18.0),
    "nan_dist": _row(np.nan, 0.5, 10.0, 4.0, np.nan, actual=8.0),
}


_PROB_EXPECTED = {
    "negbin": 0.5340391631067969,
    "zinb": 0.6272313304854376,
    "gamma": 0.06561390606449677,
    "zagamma": 0.20577182015482226,
    "negbin_paramnone": 0.5693184456270783,
    "gamma_paramnone": 0.06561390606449675,
}


def test_reconstruct_prob_branches():
    for name, expected in _PROB_EXPECTED.items():
        assert reconstruct_prob(_ROWS[name], 9.5) == pytest.approx(expected, rel=1e-12)
    assert math.isnan(reconstruct_prob(_ROWS["nan_dist"], 9.5))


_QUANTILE_EXPECTED = {
    ("negbin", 0.1): 3.0,
    ("negbin", 0.5): 9.0,
    ("negbin", 0.9): 18.0,
    ("zinb", 0.1): 0.0,  # q <= gate (0.2) -> 0.0
    ("zinb", 0.5): 7.0,  # q_adj = (0.5 - 0.2) / 0.8
    ("zinb", 0.9): 17.0,
    ("gamma", 0.1): 10.674049447419465,
    ("gamma", 0.5): 18.944043518782863,
    ("gamma", 0.9): 30.691223205968015,
    ("zagamma", 0.1): 0.0,  # q <= gate (0.15) -> 0.0
    ("zagamma", 0.5): 17.268370451545504,
    ("zagamma", 0.9): 29.69433587140154,
    ("negbin_paramnone", 0.1): 2.0,
    ("negbin_paramnone", 0.5): 8.0,
    ("negbin_paramnone", 0.9): 20.0,
    ("gamma_paramnone", 0.1): 10.674049447419463,
    ("gamma_paramnone", 0.5): 18.944043518782873,
    ("gamma_paramnone", 0.9): 30.691223205968022,
}


def test_reconstruct_quantile_branches():
    for (name, q), expected in _QUANTILE_EXPECTED.items():
        assert reconstruct_quantile(_ROWS[name], q) == pytest.approx(expected, rel=1e-12)
    for q in (0.1, 0.5, 0.9):
        assert math.isnan(reconstruct_quantile(_ROWS["nan_dist"], q))


_CRPS_EXPECTED = {
    "negbin": 1.389270172749116,
    "zinb": 1.628012029760221,
    "gamma": 1.8640904063515675,
    "zagamma": 2.2985670275544705,
    "negbin_paramnone": 1.650089165789975,
    "gamma_paramnone": 1.864090406351563,
}


def test_compute_crps_row_branches():
    for name, expected in _CRPS_EXPECTED.items():
        assert compute_crps_row(_ROWS[name]) == pytest.approx(expected, rel=1e-9)
    assert math.isnan(compute_crps_row(_ROWS["nan_dist"]))
    # Missing Actual -> NaN even with a valid distribution.
    assert math.isnan(compute_crps_row(_row("NegBin", 0.5, 10.0, 4.0, np.nan)))
