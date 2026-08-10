"""§6.1 Rung C — whole-CDF isotonic-PIT recalibration primitive.

A monotone map ``g`` fit on the validation PIT ``Z = F(Y|X)`` so ``g(Z)`` is Uniform;
applied to any CDF value ``u = F(point)`` it returns the recalibrated ``g(u)``. Distinct
from the single-line ``prob_recal_isotonic`` corrector: this reshapes the whole predictive
CDF (the alt-line ladder), not just the over-probability at one line.

Also pins the two seams that apply ``g`` outside the scorecard: the ``model_prob``
inference warp (``over = 1 − g(F(line))``, a legacy no-blob pickle an exact no-op) and
the training pipeline's persisted test-set over-prob, which must be the same warped
quantity so Gates 1/5 judge the cell as served.
"""

import numpy as np
import pandas as pd
from scipy import stats

from sportstradamus.prediction.model_prob import _apply_cdf_recal_over
from sportstradamus.training import posthoc
from sportstradamus.training.pipeline import (
    _step_calibrate_dispersion,
    _step_compute_test_probabilities,
)


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


def test_isotonic_pit_map_does_not_duplicate_terminal_atom():
    # Discrete predictives can put an entire final equal-mass bin at PIT=1. The
    # bin itself then supplies the endpoint, so the explicit CDF anchor must not
    # append a second 1.0 domain knot.
    pit = np.concatenate((np.linspace(0.01, 0.89, 90), np.ones(10)))
    blob = posthoc.fit_isotonic_pit(pit)

    assert blob is not None
    assert np.all(np.diff(blob["x"]) > 0.0)
    assert blob["x"][-2] == np.nextafter(1.0, 0.0)
    assert blob["y"][-2] < 1.0
    assert blob["x"][-1] == 1.0
    assert blob["y"][-1] == 1.0


def test_isotonic_pit_lambda_zero_is_identity():
    # Convention g_lambda = lambda*g + (1-lambda)*identity: lambda=0 must be a pure
    # no-op (the scalar-replaced cell with no recalibration), lambda=1 full recal. The
    # pipeline's cross-fit selects lambda from a grid, so this contract is load-bearing.
    rng = np.random.default_rng(4)
    pit = rng.beta(0.5, 0.5, size=2000)  # badly miscalibrated, so identity != recal
    u = np.linspace(0.0, 1.0, 101)
    g0 = posthoc.apply_cdf_recal(posthoc.fit_isotonic_pit(pit, lam=0.0), u)
    np.testing.assert_allclose(g0, u, atol=1e-9)

    terminal_pit = np.concatenate((np.linspace(0.01, 0.89, 90), np.ones(10)))
    terminal_blob = posthoc.fit_isotonic_pit(terminal_pit, lam=0.0)
    left_of_one = np.nextafter(1.0, 0.0)
    assert posthoc.apply_cdf_recal(terminal_blob, np.array([left_of_one]))[0] == left_of_one


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


def test_crossfit_keeps_randomized_draws_grouped_by_source_row(monkeypatch):
    n = 50
    row_value = (np.arange(n) + 1.0) / (n + 2.0)
    draws = np.vstack([row_value, row_value + 1e-5])
    seen_train: list[np.ndarray] = []
    real_fit = posthoc.fit_isotonic_pit

    def recording_fit(pit, **kwargs):
        seen_train.append(np.asarray(pit))
        return real_fit(pit, **kwargs)

    monkeypatch.setattr(posthoc, "fit_isotonic_pit", recording_fit)
    posthoc._crossfit_pit_ks(draws, lam=0.5, k=5)

    assert len(seen_train) == 5
    for train in seen_train:
        for first, second in zip(draws[0], draws[1], strict=True):
            assert np.any(np.isclose(train, first, atol=1e-12, rtol=0.0)) == np.any(
                np.isclose(train, second, atol=1e-12, rtol=0.0)
            )


def test_apply_cdf_recal_over_matches_one_minus_g_of_cdf():
    # Inference serves ``over = 1 − g(F(line))``. The raw model over-prob is
    # ``1 − F(line)``, so the model_prob seam applies ``1 − g(1 − over_raw)`` — the
    # same map ``g`` the offline Gate 4 scored on the PIT.
    rng = np.random.default_rng(7)
    blob = posthoc.fit_isotonic_pit(rng.beta(0.5, 0.5, size=2000))
    cdf = rng.uniform(0.0, 1.0, size=50)  # F(line) for a batch of offers
    warped = _apply_cdf_recal_over(1.0 - cdf, blob)
    np.testing.assert_allclose(warped, 1.0 - posthoc.apply_cdf_recal(blob, cdf))


def test_apply_cdf_recal_over_none_is_identity():
    # A legacy pickle (no ``pit_recal_blob``) must serve byte-identically.
    over = np.array([0.1, 0.4, 0.9])
    np.testing.assert_array_equal(_apply_cdf_recal_over(over, None), over)


def test_calibrate_dispersion_rung_c_hands_served_val_params_downstream():
    # Rung C bypasses the scalar (c, skew) fit, but _step_calibrate_temperature and
    # _step_compute_test_probabilities still read the served (raw) val/test params off the
    # returned dict; leaving any None crashes get_odds (ev=None). Pins that contract.
    rng = np.random.default_rng(9)
    n = 500
    mean, sigma, alpha = np.full(n, 6.0), np.full(n, 3.0), np.full(n, 2.0)
    fused = {
        "r_test": None,
        "r_blend_val": None,
        "alpha_blend": None,
        "alpha_blend_val": None,
        "beta_blend_val": None,
        "phi_test": None,
        "phi_blend_val": None,
        "weighted_mean": mean,
        "weighted_mean_val": mean,
        "sn_sigma_blend_test": sigma,
        "sn_sigma_blend_val": sigma,
        "sn_alpha_blend_test": alpha,
        "sn_alpha_blend_val": alpha,
        "mix_blend_test": None,
        "mix_blend_val": None,
    }
    splits = {"y_validation": pd.DataFrame({"Result": rng.gamma(2.0, 3.0, n)})}
    out = _step_calibrate_dispersion(
        {}, fused, splits, "SkewNormal", 1.0, 0.0, 100.0, 0.5, posthoc_slug="cdf_recal_isotonic"
    )
    for key in (
        "val_weighted_mean_val",
        "sn_sigma_blend_val",
        "sn_alpha_blend_val",
        "sn_sigma_blend_test",
        "sn_alpha_blend_test",
    ):
        assert out[key] is not None, f"Rung C left {key} None — downstream get_odds will crash"
    assert out["c_opt"] == 1.0 and out["skew_cal"] == 0.0  # scalar bypassed
    assert out["pit_recal_blob"] is not None  # a map was fit


def _skewnormal_inputs():
    n = 300
    rng = np.random.default_rng(8)
    fused = {"weighted_mean": np.full(n, 18.0), "gate_blend_test": None}
    splits = {"B_test": pd.DataFrame({"Line": rng.uniform(8.0, 30.0, n)})}
    base = {"sn_sigma_blend_test": np.full(n, 5.0), "sn_alpha_blend_test": np.full(n, 2.0)}
    blob = posthoc.fit_isotonic_pit(rng.beta(0.5, 0.5, size=3000))
    return fused, splits, base, blob


def test_step_compute_test_probabilities_warps_over_through_g():
    # Gate 1 (Brier-skill) and Gate 5 (ECE) read the persisted ``P`` column, and a Rung-C
    # cell is *served* with the whole-CDF map applied, so the test-set over-prob the
    # pipeline records must be ``1 − g(F(line))``. A no-blob run must stay an exact no-op.
    fused, splits, base, blob = _skewnormal_inputs()
    raw = _step_compute_test_probabilities(
        fused, {**base, "pit_recal_blob": None}, splits, {}, "SkewNormal", 1
    )
    rerun = _step_compute_test_probabilities(
        fused, {**base, "pit_recal_blob": None}, splits, {}, "SkewNormal", 1
    )
    np.testing.assert_array_equal(raw, rerun)
    warped = _step_compute_test_probabilities(
        fused, {**base, "pit_recal_blob": blob}, splits, {}, "SkewNormal", 1
    )
    raw_under = raw[:, 0]  # F(line)
    np.testing.assert_allclose(warped[:, 1], 1.0 - posthoc.apply_cdf_recal(blob, raw_under))
    assert not np.allclose(warped[:, 1], raw[:, 1])  # the map is non-trivial on this PIT
