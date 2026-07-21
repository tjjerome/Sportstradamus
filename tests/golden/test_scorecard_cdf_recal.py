"""§6.1 Rung C — Gate-4 applies the whole-CDF map ``g`` to the PIT before the KS.

The map is persisted as a constant ``PITRecalKnots`` JSON column on the scored test CSV
(mirroring ``DenomCol``/``GlobalMean``); the scorecard must warp every randomized-PIT draw
through ``g`` so ``g4_pit_ks`` reflects the *served* recalibrated predictive. A cell with no
such column behaves byte-identically to a pre-Rung-C cell.
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import skewnorm

from sportstradamus.training import posthoc
from sportstradamus.training.scorecard import (
    ACTUAL_COL,
    DECILE_COL,
    DEFAULT_PRED_COL,
    _apply_pit_recal_by_row,
    _randomized_pit_draws,
    _randomized_pit_ks,
    load_test_set,
)
from sportstradamus.training.ship_config import TARGET_NORM_NONE

_LOC, _SCALE, _ALPHA, _N = 20.0, 5.0, 3.0, 3000


def _under_dispersed_skewnorm():
    # Served predictive is SkewNormal(loc, scale, alpha); outcomes are drawn 1.7x wider, so
    # the served PIT piles into the tails (the under-dispersion signature Gate 4 fails on).
    rng = np.random.default_rng(0)
    y = skewnorm.rvs(_ALPHA, loc=_LOC, scale=_SCALE * 1.7, size=_N, random_state=rng)
    df = pd.DataFrame({"SN_Loc": _LOC, "SN_Scale": _SCALE, "SN_Alpha": _ALPHA}, index=range(_N))
    return df, y


def test_gate_g4_reflects_warped_pit():
    df, y = _under_dispersed_skewnorm()
    raw_ks = _randomized_pit_ks(df, "SkewNormal", y, strategy=TARGET_NORM_NONE)
    # g fit on this cell's own served PIT uniformizes it, so the warped Gate-4 KS must drop.
    served_pit = _randomized_pit_draws(df, "SkewNormal", y, strategy=TARGET_NORM_NONE)[0]
    blob = posthoc.fit_isotonic_pit(served_pit)
    warped = df.assign(PITRecalKnots=json.dumps(blob))
    warped_ks = _randomized_pit_ks(warped, "SkewNormal", y, strategy=TARGET_NORM_NONE)
    assert warped_ks < raw_ks
    assert warped_ks < 0.05


def test_gate_g4_identity_without_knots_column():
    # No PITRecalKnots column => no warp => the exact pre-Rung-C statistic. A lambda=0
    # (identity) blob must reproduce that same number, pinning the no-op both ways.
    df, y = _under_dispersed_skewnorm()
    bare = _randomized_pit_ks(df, "SkewNormal", y, strategy=TARGET_NORM_NONE)
    served_pit = _randomized_pit_draws(df, "SkewNormal", y, strategy=TARGET_NORM_NONE)[0]
    identity = df.assign(PITRecalKnots=json.dumps(posthoc.fit_isotonic_pit(served_pit, lam=0.0)))
    identity_ks = _randomized_pit_ks(identity, "SkewNormal", y, strategy=TARGET_NORM_NONE)
    assert identity_ks == bare


def test_load_test_set_preserves_pit_recal_knots(tmp_path):
    # report() scores a dumped CSV through load_test_set, which keeps only an allowlist of
    # columns. The constant PITRecalKnots map must survive that load, or the official Gate 4
    # silently scores the un-recalibrated cell. Both the deterministic A/B (pd.read_csv) and the
    # assign()-based tests above bypass load_test_set, so only this end-to-end load path catches it.
    rng = np.random.default_rng(1)
    n = 200
    blob = posthoc.fit_isotonic_pit(rng.beta(0.5, 0.5, size=3000))
    df = pd.DataFrame(
        {
            DECILE_COL: rng.uniform(10.0, 30.0, n),
            ACTUAL_COL: rng.uniform(0.0, 40.0, n),
            DEFAULT_PRED_COL: rng.uniform(10.0, 30.0, n),
            "SN_Loc": _LOC,
            "SN_Scale": _SCALE,
            "SN_Alpha": _ALPHA,
            "PITRecalKnots": json.dumps(blob),
        }
    )
    path = tmp_path / "WNBA_PA.csv"
    df.to_csv(path, index=False)
    loaded = load_test_set(path, DEFAULT_PRED_COL)
    probe = np.linspace(0.05, 0.95, n)
    np.testing.assert_array_equal(
        _apply_pit_recal_by_row(loaded, probe),
        posthoc.apply_cdf_recal(blob, probe),
    )
