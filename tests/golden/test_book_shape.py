"""WS2 book-shape foundation: ``fit_book_shape`` (the per-cell conditional shape fit)
and ``book_skewnormal_shape`` (the runtime curve eval feeding the SkewNormal book read).

These pin the additive, behavior-preserving foundation: the fit recovers planted
coefficients from synthetic history, the eval reproduces the fitted moments and clamps
infeasible skew, and an unfitted cell falls back to the constant-CV symmetric shape that
is bit-identical to today's book read.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import skewnorm

from sportstradamus.helpers import config
from sportstradamus.helpers.distributions import get_odds, skewnormal_loc_from_mean
from sportstradamus.training.calibration import fit_book_shape


def _synth_results(lines, a, b, skew_c, skew_d, n_per_bin, rng):
    """Draw SkewNormal outcomes whose conditional moments follow the planted curves.

    Each line bin draws ``n_per_bin`` samples with mean = line, variance = a·line^b, and
    skewness = skew_c + skew_d·line, so binning the result by line recovers the curves.
    """
    from sportstradamus.helpers.distributions import skewnormal_params_from_moments

    out_lines, out_results = [], []
    for line in lines:
        var = a * line**b
        skew = skew_c + skew_d * line
        sigma, alpha = skewnormal_params_from_moments(var, skew)
        loc = skewnormal_loc_from_mean(line, sigma, alpha)
        draws = skewnorm.rvs(float(alpha), loc=float(loc), scale=float(sigma),
                             size=n_per_bin, random_state=rng)
        out_results.append(draws)
        out_lines.append(np.full(n_per_bin, line))
    return np.concatenate(out_results), np.concatenate(out_lines)


def test_fit_book_shape_recovers_planted_coeffs():
    a, b, skew_c, skew_d = 1.3, 1.1, 0.9, -0.18
    lines = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
    rng = np.random.default_rng(12345)
    results, line_col = _synth_results(lines, a, b, skew_c, skew_d, 6000, rng)

    fit = fit_book_shape("NBA", "AST", results, line_col)

    assert fit["n_bins"] == len(lines)
    assert fit["a"] == pytest.approx(a, rel=0.10)
    assert fit["b"] == pytest.approx(b, abs=0.08)
    assert fit["skew_c"] == pytest.approx(skew_c, abs=0.12)
    assert fit["skew_d"] == pytest.approx(skew_d, abs=0.06)


def test_fit_book_shape_returns_none_when_too_few_bins():
    rng = np.random.default_rng(1)
    # Two bins clear the 120-row floor; the third (50 rows) is dropped -> below the bin floor.
    lines = np.concatenate([np.full(200, 1.5), np.full(200, 2.5), np.full(50, 3.5)])
    results = rng.poisson(lines).astype(float)

    assert fit_book_shape("NBA", "AST", results, lines) is None


def test_book_skewnormal_shape_fitted_moments_round_trip(monkeypatch):
    coeffs = {"a": 1.3, "b": 1.1, "skew_c": 0.9, "skew_d": -0.18, "n_bins": 9}
    monkeypatch.setitem(config.stat_meta, "TESTLG", {"TESTMK": {"cv": 0.5, "book_shape": coeffs}})
    mean = 4.0

    sigma, alpha = config.book_skewnormal_shape("TESTLG", "TESTMK", mean)

    var_t = coeffs["a"] * mean ** coeffs["b"]
    skew_t = coeffs["skew_c"] + coeffs["skew_d"] * mean
    loc = float(skewnormal_loc_from_mean(mean, sigma, alpha))
    m, v, s = skewnorm.stats(float(alpha), loc=loc, scale=float(sigma), moments="mvs")
    assert float(m) == pytest.approx(mean, abs=1e-6)
    assert float(v) == pytest.approx(var_t, abs=2e-3)
    assert float(s) == pytest.approx(skew_t, abs=2e-3)


def test_book_skewnormal_shape_unfitted_fallback_is_noop(monkeypatch):
    monkeypatch.setitem(config.stat_meta, "TESTLG", {"TESTMK": {"cv": 0.55}})
    mean = 3.0

    sigma, alpha = config.book_skewnormal_shape("TESTLG", "TESTMK", mean)

    assert float(alpha) == 0.0
    assert float(sigma) == pytest.approx(mean * 0.55, abs=1e-12)
    # The fallback shape feeds get_odds identically to today's constant-CV book read.
    p_new = get_odds(2.5, mean, "SkewNormal", sigma=sigma, skew_alpha=alpha)
    p_old = get_odds(2.5, mean, "SkewNormal", cv=0.55)
    assert p_new == pytest.approx(p_old, abs=1e-12)


def test_book_skewnormal_shape_clamps_infeasible_skew(monkeypatch):
    # A steep negative skew_d drives the linear skew far past the SkewNormal lower bound at
    # high mean (the DREB -1.55 extrapolation); the eval must stay finite and in-band.
    coeffs = {"a": 1.0, "b": 0.5, "skew_c": 0.8, "skew_d": -0.5, "n_bins": 5}
    monkeypatch.setitem(config.stat_meta, "TESTLG", {"TESTMK": {"cv": 0.5, "book_shape": coeffs}})
    mean = 6.0  # raw skew = 0.8 - 0.5*6 = -2.2, far below the -0.9953 bound

    sigma, alpha = config.book_skewnormal_shape("TESTLG", "TESTMK", mean)

    assert np.isfinite(float(alpha))
    loc = float(skewnormal_loc_from_mean(mean, sigma, alpha))
    _, v, s = skewnorm.stats(float(alpha), loc=loc, scale=float(sigma), moments="mvs")
    assert float(v) == pytest.approx(coeffs["a"] * mean ** coeffs["b"], abs=2e-3)
    assert -0.9953 < float(s) < 0.0
