"""WS2 book-shape foundation: ``fit_book_shape`` (the per-cell conditional shape fit)
and ``book_skewnormal_shape`` (the runtime curve eval feeding the SkewNormal book read).

These pin the additive, behavior-preserving foundation: the fit recovers planted
coefficients from synthetic history, the eval reproduces the fitted moments and clamps
infeasible skew, and an unfitted cell falls back to the constant-CV symmetric shape that
is bit-identical to today's book read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import skewnorm

from sportstradamus.helpers import config
from sportstradamus.helpers.distributions import (
    fused_loc,
    get_ev,
    get_odds,
    skewnormal_loc_from_mean,
)
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


# --- Step 2a: the distributions.py shape-injection contract -------------------
# get_ev accepts a fixed book sigma (evaluated once at the quoted line); fused_loc
# accepts a fitted book (sigma, skew) for the blend. Both default to today's shape.


def test_get_ev_fixed_sigma_round_trips():
    line, under, sigma, skew = 2.5, 0.62, 1.4, -0.4
    ev = get_ev(line, under, dist="SkewNormal", sigma=sigma, skew_alpha=skew)
    assert get_odds(line, ev, "SkewNormal", sigma=sigma, skew_alpha=skew) == pytest.approx(
        under, abs=1e-7
    )


def test_fused_loc_skewnormal_default_unchanged():
    ev, sig, skew, gate = fused_loc(
        0.3, np.array([10.0]), np.array([11.0]), 0.4, "SkewNormal",
        sigma=np.array([3.5]), skew_alpha=np.array([1.2]),
    )
    assert float(ev[0]) == pytest.approx(10.82731187571877, abs=1e-9)
    assert float(sig[0]) == pytest.approx(4.060653952180996, abs=1e-9)
    assert float(skew[0]) == pytest.approx(0.36, abs=1e-9)
    assert gate is None


def test_fused_loc_book_shape_noop_matches_default():
    args = (0.3, np.array([10.0]), np.array([11.0]), 0.4, "SkewNormal")
    kw = {"sigma": np.array([3.5]), "skew_alpha": np.array([1.2])}
    default = fused_loc(*args, **kw)
    explicit_noop = fused_loc(
        *args, **kw, book_sigma=np.array([11.0 * 0.4]), book_skew_alpha=np.array([0.0])
    )
    for a, b in zip(default[:3], explicit_noop[:3], strict=True):
        assert float(a[0]) == pytest.approx(float(b[0]), abs=1e-9)


def test_fused_loc_blends_book_skew():
    w, model_skew, book_skew = 0.3, 1.2, -0.8
    _, _, skew, _ = fused_loc(
        w, np.array([10.0]), np.array([11.0]), 0.4, "SkewNormal",
        sigma=np.array([3.5]), skew_alpha=np.array([model_skew]),
        book_sigma=np.array([4.0]), book_skew_alpha=np.array([book_skew]),
    )
    assert float(skew[0]) == pytest.approx(w * model_skew + (1 - w) * book_skew, abs=1e-9)


# --- Step 2c: model_prob book-over-prob uses the per-cell shape (at the book mean) -----
# Evaluating book_skewnormal_shape at the book mean (Market Projection) makes the unfitted
# no-op (mean*cv, 0) bit-identical to the legacy symmetric read — the behavior opts in per
# fitted cell with no explicit gate.

_OVER_SHAPE = {"a": 1.3, "b": 1.1, "skew_c": 0.6, "skew_d": -0.1, "n_bins": 9}


def test_book_over_prob_unfitted_is_legacy_symmetric():
    from sportstradamus.prediction.model_prob import _book_over_prob

    league, market = "WNBA", "AST"
    cv = config.stat_cv[league][market]
    assert config.stat_meta[league][market].get("book_shape") is None
    df = pd.DataFrame({"Line": [2.5, 1.5], "Market Projection": [2.2, 1.1]})

    got = _book_over_prob(df, "SkewNormal", cv, 0.5, None, league, market)

    expected = [
        1 - get_odds(ln, mp, "SkewNormal", cv=cv, step=0.5, sigma=mp * cv, skew_alpha=0, gate=None)
        for ln, mp in zip(df["Line"], df["Market Projection"], strict=True)
    ]
    assert list(got) == pytest.approx(expected, abs=1e-12)


def test_book_over_prob_fitted_uses_shape(monkeypatch):
    from sportstradamus.prediction.model_prob import _book_over_prob

    league, market = "WNBA", "AST"
    cv = config.stat_cv[league][market]
    monkeypatch.setitem(config.stat_meta[league][market], "book_shape", _OVER_SHAPE)
    df = pd.DataFrame({"Line": [2.5], "Market Projection": [2.2]})

    got = _book_over_prob(df, "SkewNormal", cv, 0.5, None, league, market)

    sigma, skew = config.book_skewnormal_shape(league, market, 2.2)
    expected = 1 - get_odds(
        2.5, 2.2, "SkewNormal", cv=cv, step=0.5,
        sigma=float(sigma), skew_alpha=float(skew), gate=None,
    )
    assert float(got.iloc[0]) == pytest.approx(expected, abs=1e-12)


# --- Settling verdict: the served blend collapses-to-cv (book shape does NOT enter) -------
# The WS2 settling experiment found the shaped book loses the served blend OOS, so
# _blend_with_book keeps the symmetric book leg — its output must be independent of a fitted
# book_shape, even though the book-only _book_over_prob path does read the shape.


def _sn_blend_df():
    return pd.DataFrame({
        "Projection": [2.2, 5.0],
        "Market Projection": [2.0, 4.5],
        "Line": [2.5, 4.5],
        "Model Sigma": [1.3, 2.1],
        "Model Skew": [0.4, -0.2],
    })


def test_blend_with_book_collapses_to_cv(monkeypatch):
    from sportstradamus.prediction.model_prob import _blend_with_book

    league, market = "WNBA", "AST"
    cv = config.stat_cv[league][market]

    df0 = _sn_blend_df()
    base0 = _blend_with_book(df0, "SkewNormal", 0.6, cv, 0.0, league, market)

    monkeypatch.setitem(config.stat_meta[league][market], "book_shape", _OVER_SHAPE)
    df1 = _sn_blend_df()
    base1 = _blend_with_book(df1, "SkewNormal", 0.6, cv, 0.0, league, market)

    assert list(base0) == pytest.approx(list(base1), abs=1e-12)
    assert list(df0["Model Sigma"]) == pytest.approx(list(df1["Model Sigma"]), abs=1e-12)
    assert list(df0["Model Skew"]) == pytest.approx(list(df1["Model Skew"]), abs=1e-12)
