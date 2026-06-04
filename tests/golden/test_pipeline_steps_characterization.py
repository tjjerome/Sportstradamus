"""Characterization pins for the two over-CC ``train_market`` phase helpers in
``training.pipeline``: ``_step_fuse_predictions`` (model+book parameter blend per
distribution family) and ``_step_compute_diagnostics`` (the diag_* values written into
``filedict["diagnostics"]``).

Both branch on the distribution family, so every branch group is pinned: SkewNormal /
NegBin / ZINB / Gamma / ZAGamma. Inputs are synthetic (only the dict keys each helper
reads). ``calibration.fit_blend_weight`` is monkeypatched to record its ``(base_dist,
kwargs)`` and return a fixed 0.5 -- this freezes the per-family wiring without depending
on the optimizer, while ``fused_loc`` / ``get_odds`` / ``_zero_inflated_outcome_mean``
run for real. Expected values are literals captured from the pre-refactor code.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import sportstradamus.training.pipeline as pipe

_N = 8


@pytest.fixture
def fit_calls(monkeypatch):
    """Patch fit_blend_weight to record its call and return a fixed weight (0.5)."""
    calls: list[dict] = []

    def _fake(blending, *args, **kwargs):
        calls.append(
            {
                "base_dist": args[3] if len(args) > 3 else None,
                "kwargs": {
                    k: (round(float(np.mean(v)), 6) if isinstance(v, np.ndarray) else v)
                    for k, v in kwargs.items()
                },
            }
        )
        return 0.5

    monkeypatch.setattr(pipe.calibration, "fit_blend_weight", _fake)
    return calls


def _decoded() -> dict:
    base = np.linspace(8.0, 15.0, _N)
    return {
        "ev": base.copy(),
        "ev_validation": (base + 0.5).copy(),
        "sn_sigma_test": np.full(_N, 3.0),
        "sn_sigma_val": np.full(_N, 3.1),
        "sn_alpha_test": np.full(_N, 0.2),
        "sn_alpha_val": np.full(_N, 0.25),
        "alpha": np.full(_N, 2.0),
        "alpha_validation": np.full(_N, 2.1),
        "r": np.full(_N, 4.0),
        "r_validation": np.full(_N, 4.2),
        "gate_test": np.full(_N, 0.3),
        "gate_validation": np.full(_N, 0.32),
    }


def _splits() -> dict:
    return {
        "B_test": pd.DataFrame({"EV": np.linspace(8.5, 14.5, _N), "Line": np.linspace(9.0, 14.0, _N)}),
        "B_validation": pd.DataFrame({"EV": np.linspace(8.6, 14.6, _N)}),
        "y_validation": pd.DataFrame({"Result": np.linspace(7.0, 16.0, _N)}),
        "y_test": pd.DataFrame({"Result": np.linspace(7.5, 15.5, _N)}),
        "X_test": pd.DataFrame(
            {"MeanYr": np.linspace(8.0, 13.0, _N), "STDYr": np.full(_N, 4.0), "denom": np.linspace(9.0, 12.0, _N)}
        ),
    }


def _prob_params() -> pd.DataFrame:
    return pd.DataFrame(
        {"scale": np.full(_N, 0.3), "concentration": np.full(_N, 2.0), "total_count": np.full(_N, 4.0)}
    )


def _assert_fuse(out: dict, expected_means: dict):
    """Each named field's nanmean ≈ expected; fields absent from expected stay None."""
    for key, value in out.items():
        if key in expected_means:
            assert np.asarray(value, dtype=float).mean() == pytest.approx(expected_means[key]), key
        else:
            assert value is None, f"{key} expected None, got {value!r}"


def test_fuse_skewnormal_no_gate(fit_calls):
    out = pipe._step_fuse_predictions(_decoded(), _splits(), "SkewNormal", 0.9, 0.0, blending="logit")
    assert fit_calls[0]["base_dist"] == "SkewNormal"
    assert fit_calls[0]["kwargs"] == {"cv": 0.9, "model_sigma": 3.1, "model_skew_alpha": 0.25}
    _assert_fuse(out, {
        "model_weight": 0.5, "weighted_mean": 11.40086223,
        "sn_sigma_blend_test": 4.06124187, "sn_sigma_blend_val": 4.1882951,
        "sn_alpha_blend_test": 0.1, "sn_alpha_blend_val": 0.125,
    })


def test_fuse_skewnormal_with_gate(fit_calls):
    out = pipe._step_fuse_predictions(_decoded(), _splits(), "SkewNormal", 0.9, 0.3, blending="logit")
    assert fit_calls[0]["kwargs"]["gate_book"] == 0.3
    _assert_fuse(out, {
        "model_weight": 0.5, "weighted_mean": 11.40086223,
        "gate_blend_test": 0.3, "gate_blend_val": 0.3,
        "sn_sigma_blend_test": 4.06124187, "sn_sigma_blend_val": 4.1882951,
        "sn_alpha_blend_test": 0.1, "sn_alpha_blend_val": 0.125,
    })


def test_fuse_negbin(fit_calls):
    out = pipe._step_fuse_predictions(_decoded(), _splits(), "NegBin", 0.9, 0.0, blending="logit")
    assert fit_calls[0]["base_dist"] == "NegBin"
    assert fit_calls[0]["kwargs"] == {"model_alpha": 2.1, "model_r": 4.2, "cv": 0.9}
    _assert_fuse(out, {
        "model_weight": 0.5, "weighted_mean": 11.49875967,
        "r_test": 2.10818511, "r_blend_val": 2.1602469, "p_test": 0.1588969, "p_val": 0.15851274,
    })


def test_fuse_zinb(fit_calls):
    out = pipe._step_fuse_predictions(_decoded(), _splits(), "ZINB", 0.9, 0.3, blending="logit")
    assert fit_calls[0]["base_dist"] == "NegBin"
    assert fit_calls[0]["kwargs"]["gate_model"] == 0.32 and fit_calls[0]["kwargs"]["gate_book"] == 0.3
    _assert_fuse(out, {
        "model_weight": 0.5, "weighted_mean": 11.49875967,
        "gate_blend_test": 0.3, "gate_blend_val": 0.31,
        "r_test": 2.10818511, "r_blend_val": 2.1602469, "p_test": 0.1588969, "p_val": 0.15851274,
    })


def test_fuse_gamma(fit_calls):
    out = pipe._step_fuse_predictions(_decoded(), _splits(), "Gamma", 0.9, 0.0, blending="logit")
    assert fit_calls[0]["base_dist"] == "Gamma"
    _assert_fuse(out, {
        "model_weight": 0.5, "weighted_mean": 11.49534129,
        "alpha_blend": 1.61691423, "alpha_blend_val": 1.66672267,
        "beta_blend": 0.14599945, "beta_blend_val": 0.14573308,
    })


def test_fuse_zagamma(fit_calls):
    out = pipe._step_fuse_predictions(_decoded(), _splits(), "ZAGamma", 0.9, 0.3, blending="logit")
    assert fit_calls[0]["base_dist"] == "Gamma"
    assert fit_calls[0]["kwargs"]["gate_model"] == 0.32 and fit_calls[0]["kwargs"]["gate_book"] == 0.3
    _assert_fuse(out, {
        "model_weight": 0.5, "weighted_mean": 11.49534129,
        "gate_blend_test": 0.3, "gate_blend_val": 0.31,
        "alpha_blend": 1.61691423, "alpha_blend_val": 1.66672267,
        "beta_blend": 0.14599945, "beta_blend_val": 0.14573308,
    })


def _diag(dist, gate_blend_test):
    weighted_mean = np.linspace(9.0, 13.0, _N)
    y_proba = np.column_stack([np.linspace(0.6, 0.3, _N), np.linspace(0.4, 0.7, _N)])
    y_class = np.array([0, 1, 0, 1, 1, 0, 1, 1])
    player_stats = pd.Series(np.linspace(5.0, 20.0, 20))
    return pipe._step_compute_diagnostics(
        _splits(), _prob_params(), weighted_mean, y_proba, y_class, gate_blend_test,
        dist, 0.9, "denom", player_stats, 1,
    )


def _assert_diag_common(out):
    assert out["start_mean"] == pytest.approx(10.5)
    assert out["mean_line"] == pytest.approx(11.5)
    assert out["result_mean"] == pytest.approx(11.5)
    assert out["frac_ev_gt_line"] == pytest.approx(0.0)
    assert out["ev_meanyr_corr"] == pytest.approx(-1.0)
    assert out["result_meanyr_corr"] == pytest.approx(1.0)
    # N=8 < the 10-row diagnostic floor -> over_pct / cf collapse to nan
    assert math.isnan(out["over_pct_ev_gt"]) and math.isnan(out["over_pct_ev_lt"])
    assert math.isnan(out["cf_over_pct"])


def test_diag_skewnormal():
    out = _diag("SkewNormal", None)
    _assert_diag_common(out)
    assert out["shape_label"] == "scale"
    assert out["start_shape"] == pytest.approx(0.9)
    assert out["model_shape"] == pytest.approx(3.15)
    assert out["empirical_shape"] == pytest.approx(0.22770562459407903)
    assert out["model_ev"] == pytest.approx(11.0)


def test_diag_gamma():
    out = _diag("Gamma", None)
    _assert_diag_common(out)
    assert out["shape_label"] == "alpha"
    assert out["start_shape"] == pytest.approx(6.890625)
    assert out["model_shape"] == pytest.approx(2.0)
    assert out["empirical_shape"] == pytest.approx(7.162698412698414)
    assert out["model_ev"] == pytest.approx(11.0)


def test_diag_zagamma_applies_gate_to_ev():
    out = _diag("ZAGamma", np.full(_N, 0.3))
    assert out["shape_label"] == "alpha"
    assert out["model_ev"] == pytest.approx(7.699999999999999)  # 11.0 * (1 - 0.3)
    assert out["ev_minus_line"] == pytest.approx(-3.8000000000000003)
    assert out["median_ev_diff"] == pytest.approx(-3.800000000000001)


def test_diag_negbin():
    out = _diag("NegBin", None)
    _assert_diag_common(out)
    assert out["shape_label"] == "r"
    assert out["start_shape"] == pytest.approx(20.045454545454547)
    assert out["model_shape"] == pytest.approx(4.0)
    assert out["empirical_shape"] == pytest.approx(16.775092936802974)


def test_diag_zinb_applies_gate_to_ev():
    out = _diag("ZINB", np.full(_N, 0.3))
    assert out["shape_label"] == "r"
    assert out["model_ev"] == pytest.approx(7.699999999999999)
    assert out["ev_minus_line"] == pytest.approx(-3.8000000000000003)


def test_diag_over_pct_and_cf_nonnan_branch():
    """N=16 confident slate with EV above the line: exercises the IfExp true-side of
    over_pct_ev_gt and the cf get_odds counterfactual ratio (the cf-block extraction)."""
    n = 16
    weighted_mean = np.full(n, 14.0)
    line = np.linspace(9.0, 12.0, n)
    splits = {
        "B_test": pd.DataFrame({"EV": line, "Line": line}),
        "y_test": pd.DataFrame({"Result": np.linspace(10.0, 18.0, n)}),
        "X_test": pd.DataFrame({"MeanYr": np.linspace(8.0, 13.0, n), "STDYr": np.full(n, 4.0), "denom": np.full(n, 10.0)}),
    }
    prob_params = pd.DataFrame({"scale": np.full(n, 0.3), "concentration": np.full(n, 2.0), "total_count": np.full(n, 4.0)})
    y_proba = np.column_stack([np.full(n, 0.2), np.full(n, 0.8)])
    y_class = np.array([1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1])
    player_stats = pd.Series(np.linspace(5.0, 20.0, 20))

    out = pipe._step_compute_diagnostics(
        splits, prob_params, weighted_mean, y_proba, y_class, None, "Gamma", 0.9, "denom", player_stats, 1
    )
    assert out["frac_ev_gt_line"] == pytest.approx(1.0)
    assert out["over_pct_ev_gt"] == pytest.approx(0.875)
    assert math.isnan(out["over_pct_ev_lt"])  # no rows with ev <= line
    assert out["cf_over_pct"] == pytest.approx(1.0)
