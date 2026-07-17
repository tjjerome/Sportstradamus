"""§6.1 Rung C — the training side persists the warped over-prob.

Gate 1 (Brier-skill) and Gate 5 (ECE) read the persisted ``P`` column, and a Rung-C cell
is *served* with the whole-CDF map applied, so the test-set over-prob the pipeline records
must be ``1 − g(F(line))`` — the same recalibrated CDF the inference seam serves. Otherwise
g1/g5 would judge the un-recalibrated cell and the ship decision would not reflect serving.
"""

import numpy as np
import pandas as pd

from sportstradamus.training import posthoc
from sportstradamus.training.pipeline import (
    _step_calibrate_dispersion,
    _step_compute_test_probabilities,
)

_N = 300


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
    rng = np.random.default_rng(8)
    fused = {"weighted_mean": np.full(_N, 18.0), "gate_blend_test": None}
    splits = {"B_test": pd.DataFrame({"Line": rng.uniform(8.0, 30.0, _N)})}
    base = {"sn_sigma_blend_test": np.full(_N, 5.0), "sn_alpha_blend_test": np.full(_N, 2.0)}
    blob = posthoc.fit_isotonic_pit(rng.beta(0.5, 0.5, size=3000))
    return fused, splits, base, blob


def test_step_compute_test_probabilities_warps_over_through_g():
    fused, splits, base, blob = _skewnormal_inputs()
    raw = _step_compute_test_probabilities(
        fused, {**base, "pit_recal_blob": None}, splits, {}, "SkewNormal", 1
    )
    warped = _step_compute_test_probabilities(
        fused, {**base, "pit_recal_blob": blob}, splits, {}, "SkewNormal", 1
    )
    raw_under = raw[:, 0]  # F(line)
    np.testing.assert_allclose(warped[:, 1], 1.0 - posthoc.apply_cdf_recal(blob, raw_under))
    assert not np.allclose(warped[:, 1], raw[:, 1])  # the map is non-trivial on this PIT


def test_step_compute_test_probabilities_no_blob_is_identity():
    fused, splits, base, _ = _skewnormal_inputs()
    a = _step_compute_test_probabilities(
        fused, {**base, "pit_recal_blob": None}, splits, {}, "SkewNormal", 1
    )
    b = _step_compute_test_probabilities(
        fused, {**base, "pit_recal_blob": None}, splits, {}, "SkewNormal", 1
    )
    np.testing.assert_array_equal(a, b)
