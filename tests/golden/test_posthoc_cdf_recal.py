"""§6.1 Rung C — whole-CDF isotonic-PIT recalibration primitive.

A monotone map ``g`` fit on the validation PIT ``Z = F(Y|X)`` so ``g(Z)`` is Uniform;
applied to any CDF value ``u = F(point)`` it returns the recalibrated ``g(u)``. Distinct
from the single-line ``prob_recal_isotonic`` corrector: this reshapes the whole predictive
CDF (the alt-line ladder), not just the over-probability at one line.
"""

import numpy as np
from scipy import stats

from sportstradamus.training import posthoc


def _ks_uniform(u: np.ndarray) -> float:
    return float(stats.kstest(u, "uniform").statistic)


def test_isotonic_pit_map_flattens_under_dispersed_pit():
    # Beta(0.5, 0.5) is U-shaped: the PIT signature of an UNDER-dispersed predictive
    # (too narrow => outcomes land in the tails => F(y) piles up near 0 and 1). The
    # recalibration map is that PIT's own CDF, so applying it should uniformize.
    rng = np.random.default_rng(0)
    pit = rng.beta(0.5, 0.5, size=4000)
    blob = posthoc.fit_isotonic_pit(pit)
    recal = posthoc.apply_cdf_recal(blob, pit)
    assert _ks_uniform(recal) < 0.05 < _ks_uniform(pit)


def test_apply_cdf_recal_none_is_identity():
    # The no-blob path (a non-Rung-C cell): a missing map must be an exact no-op so a
    # pre-Rung-C pickle and a cell without the cdf_recal_isotonic slug serve byte-identically.
    u = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    np.testing.assert_array_equal(posthoc.apply_cdf_recal(None, u), u)


def test_isotonic_pit_blob_round_trips_through_json():
    # The map is persisted as a stringified-JSON CSV column and a pickle key; it must
    # be plain-typed (lists/floats) and reproduce the same transform after a round-trip.
    import json

    rng = np.random.default_rng(1)
    pit = rng.beta(0.6, 1.4, size=2000)  # right-skewed PIT (over-prediction signature)
    blob = posthoc.fit_isotonic_pit(pit)
    reloaded = json.loads(json.dumps(blob))
    u = np.linspace(0.0, 1.0, 101)
    np.testing.assert_allclose(
        posthoc.apply_cdf_recal(reloaded, u), posthoc.apply_cdf_recal(blob, u)
    )


def test_isotonic_pit_map_is_a_valid_cdf_transform():
    rng = np.random.default_rng(2)
    blob = posthoc.fit_isotonic_pit(rng.beta(0.5, 0.5, size=3000))
    u = np.linspace(0.0, 1.0, 201)
    g = posthoc.apply_cdf_recal(blob, u)
    assert np.all(np.diff(g) >= -1e-12)  # monotone non-decreasing
    assert g[0] == 0.0 and g[-1] == 1.0  # anchored to a full CDF
    assert g.min() >= 0.0 and g.max() <= 1.0


def test_isotonic_pit_lambda_zero_is_identity():
    # Convention g_lambda = lambda*g + (1-lambda)*identity: lambda=0 must be a pure
    # no-op (the scalar-replaced cell with no recalibration), lambda=1 full recal. The
    # pipeline's cross-fit selects lambda from a grid, so this contract is load-bearing.
    rng = np.random.default_rng(4)
    pit = rng.beta(0.5, 0.5, size=2000)  # badly miscalibrated, so identity != recal
    u = np.linspace(0.0, 1.0, 101)
    g0 = posthoc.apply_cdf_recal(posthoc.fit_isotonic_pit(pit, lam=0.0), u)
    np.testing.assert_allclose(g0, u, atol=1e-9)


def test_isotonic_pit_leaves_calibrated_pit_alone():
    # An already-uniform PIT (well-calibrated cell) must stay near-uniform — Rung C
    # should not distort a cell it cannot improve.
    rng = np.random.default_rng(3)
    pit = rng.uniform(0.0, 1.0, size=4000)
    recal = posthoc.apply_cdf_recal(posthoc.fit_isotonic_pit(pit), pit)
    assert _ks_uniform(recal) < 0.03


def test_select_pit_recal_recalibrates_miscalibrated_pit():
    rng = np.random.default_rng(5)
    pit = rng.beta(0.5, 0.5, size=3000)  # under-dispersed
    blob, cv_ks = posthoc.select_pit_recal(pit)
    assert blob is not None and blob["lam"] > 0.0  # recalibration selected
    assert cv_ks < _ks_uniform(pit)  # honest OOF KS still beats the raw PIT
    assert _ks_uniform(posthoc.apply_cdf_recal(blob, pit)) < 0.05  # served map flattens


def test_select_pit_recal_cross_fit_ks_is_honest_not_in_sample():
    # THE load-bearing guarantee: cv_ks is an out-of-fold estimate, so it can never be
    # better than the in-sample fit it would otherwise flatter. This is what stops a
    # high-DOF map from false-shipping a knife-edge cell.
    rng = np.random.default_rng(6)
    pit = rng.beta(0.5, 0.5, size=800)  # small n => in-sample optimism is visible
    blob, cv_ks = posthoc.select_pit_recal(pit)
    in_sample = _ks_uniform(posthoc.apply_cdf_recal(blob, pit))
    assert cv_ks >= in_sample
