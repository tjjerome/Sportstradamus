"""Unit tests for SkewNormal decode in :mod:`sportstradamus.prediction.model_prob`.

Pins the train/predict mirror for the SkewNormal branch added in P1 Task 5:

* Legacy pickles (no ``offset_meta`` / ``target_normalization`` keys) must decode
  through the ``ratio_meanyr`` strategy and produce ``Model EV ==
  loc * MeanYr_clipped + scale * MeanYr_clipped * delta * sqrt(2/pi)``,
  bit-identical to the pre-Task-5 hand-rolled formula.
* Centered-additive pickles (``method == "eb_additive"``) must decode through
  the EB-prior path: ``Model EV == EB_prior + loc + scale * delta * sqrt(2/pi)``
  with ``scale`` left in absolute residual units (no ``× MeanYr``).
* ``Model Skew`` must be finite under both paths — guards the FGA NaN dead end
  documented in ``docs/OVERCONFIDENCE_INVESTIGATION.md`` §3.4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sportstradamus.prediction.model_prob import _decode_skewnormal, _dispersion_calibrate
from sportstradamus.training.baselines import EB_SHRINKAGE_K, compute_eb_prior

# Floor applied to ``MeanYr`` before division. Mirrors the value pinned in
# ``training.baselines`` (and the legacy pipeline.py:466). Tests assert
# against this exact constant.
_MEANYR_FLOOR = 0.5


def _synth_player_stats(n: int = 6, rng_seed: int = 7) -> pd.DataFrame:
    """Feature frame with the columns the decode helper reads."""
    rng = np.random.default_rng(rng_seed)
    return pd.DataFrame(
        {
            "MeanYr": rng.uniform(2.0, 25.0, size=n),
            "MeanYr_nonzero": rng.uniform(2.0, 25.0, size=n),
            "GamesPlayed": rng.integers(1, 80, size=n).astype(float),
            "STDYr": rng.uniform(0.5, 5.0, size=n),
            "ZeroYr": np.zeros(n),
        }
    )


def _synth_prob_params(n: int = 6, rng_seed: int = 11) -> pd.DataFrame:
    """Synthetic LightGBMLSS predict-parameters frame."""
    rng = np.random.default_rng(rng_seed)
    return pd.DataFrame(
        {
            "loc": rng.uniform(0.5, 1.5, size=n),
            "scale": rng.uniform(0.1, 0.5, size=n),
            "alpha": rng.uniform(-2.0, 2.0, size=n),
        }
    )


def _expected_ratio_ev(prob_params: pd.DataFrame, X: pd.DataFrame, denom_col: str):
    """Replicates the pre-Task-5 hand-rolled formula for the ratio path."""
    meanyr_vals = X[denom_col].clip(lower=_MEANYR_FLOOR).to_numpy()
    loc_abs = prob_params["loc"].to_numpy() * meanyr_vals
    scale_abs = prob_params["scale"].to_numpy() * meanyr_vals
    alpha_sn = prob_params["alpha"].to_numpy()
    delta = alpha_sn / np.sqrt(1 + alpha_sn**2)
    return loc_abs + scale_abs * delta * np.sqrt(2 / np.pi), scale_abs, alpha_sn


def test_legacy_pickle_decodes_through_ratio_path_bit_identical():
    """No offset_meta + no target_normalization ⇒ same numbers as pre-Task-5 code."""
    X = _synth_player_stats()
    prob_params = _synth_prob_params()
    expected_ev, expected_sigma, expected_skew = _expected_ratio_ev(
        prob_params, X, denom_col="MeanYr"
    )

    out = _decode_skewnormal(
        prob_params.copy(),
        X,
        hist_gate=0.0,
        offset_meta=None,
        target_normalization="ratio_meanyr",
    )

    # Bitwise equality vs. legacy formula.
    np.testing.assert_array_equal(out["Projection"].to_numpy(), expected_ev)
    np.testing.assert_array_equal(out["Model Sigma"].to_numpy(), expected_sigma)
    np.testing.assert_array_equal(out["Model Skew"].to_numpy(), expected_skew)
    # Gate stays off when hist_gate <= publish threshold.
    assert "Model Gate" not in out.columns


def test_ratio_path_uses_meanyr_nonzero_when_hist_gate_exceeds_threshold():
    """High hist_gate switches denom column to MeanYr_nonzero (matches legacy)."""
    X = _synth_player_stats()
    prob_params = _synth_prob_params()
    # hist_gate > 0.05 -> MeanYr_nonzero; > 0.02 -> Model Gate populated.
    hist_gate = 0.30
    expected_ev, _, _ = _expected_ratio_ev(prob_params, X, denom_col="MeanYr_nonzero")

    out = _decode_skewnormal(
        prob_params.copy(),
        X,
        hist_gate=hist_gate,
        offset_meta=None,
        target_normalization="ratio_meanyr",
    )

    np.testing.assert_array_equal(out["Projection"].to_numpy(), expected_ev)
    np.testing.assert_array_equal(out["Model Gate"].to_numpy(), np.full(len(X), hist_gate))


def test_centered_additive_decodes_through_eb_offset_path():
    """EB centered: Model EV = EB_prior + loc + scale·delta·sqrt(2/pi)."""
    X = _synth_player_stats()
    prob_params = _synth_prob_params()
    global_mean = 8.69
    offset_meta = {
        "method": "eb_additive",
        "k": EB_SHRINKAGE_K,
        "global_mean": global_mean,
        "prior_col": "MeanYr",
        "games_col": "GamesPlayed",
    }

    eb = compute_eb_prior(
        X["MeanYr"].clip(lower=_MEANYR_FLOOR).to_numpy(),
        X["GamesPlayed"].clip(lower=0).to_numpy(),
        global_mean,
        EB_SHRINKAGE_K,
    )
    loc = prob_params["loc"].to_numpy()
    scale = prob_params["scale"].to_numpy()
    alpha_sn = prob_params["alpha"].to_numpy()
    delta = alpha_sn / np.sqrt(1 + alpha_sn**2)
    expected_ev = eb + loc + scale * delta * np.sqrt(2 / np.pi)

    out = _decode_skewnormal(
        prob_params.copy(),
        X,
        hist_gate=0.0,
        offset_meta=offset_meta,
        target_normalization="centered_additive_eb_meanyr_k10",
    )

    np.testing.assert_allclose(out["Projection"].to_numpy(), expected_ev, atol=1e-12)
    # Centered scale stays in absolute residual units — NOT × MeanYr.
    np.testing.assert_array_equal(out["Model Sigma"].to_numpy(), scale)
    # And critically: NOT equal to the ratio decode (which would be loc * MeanYr).
    meanyr_clipped = X["MeanYr"].clip(lower=_MEANYR_FLOOR).to_numpy()
    assert not np.allclose(out["Projection"].to_numpy(), loc * meanyr_clipped)


def test_centered_mean10_decodes_through_mean10_offset_path():
    """mean10: Model EV = Mean10 + loc + scale·delta·sqrt(2/pi); scale stays absolute.

    The centered_additive_mean10 decode path had no serve-side coverage — the gap behind
    the 2026-06-24 ~2x overconfidence ship.
    """
    X = _synth_player_stats()
    X["Mean10"] = np.random.default_rng(21).uniform(2.0, 25.0, len(X))
    prob_params = _synth_prob_params()
    offset_meta = {
        "method": "mean10_additive",
        "global_mean": 8.69,
        "prior_col": "Mean10",
        "prior_fallback_col": "MeanYr",
        "denom_col": "MeanYr",
    }
    loc = prob_params["loc"].to_numpy()
    scale = prob_params["scale"].to_numpy()
    alpha_sn = prob_params["alpha"].to_numpy()
    delta = alpha_sn / np.sqrt(1 + alpha_sn**2)
    # All Mean10 present (≥ floor) ⇒ baseline is Mean10, the denom fallback is unused.
    expected_ev = X["Mean10"].to_numpy() + loc + scale * delta * np.sqrt(2 / np.pi)

    out = _decode_skewnormal(
        prob_params.copy(),
        X,
        hist_gate=0.0,
        offset_meta=offset_meta,
        target_normalization="centered_additive_mean10",
    )

    np.testing.assert_allclose(out["Projection"].to_numpy(), expected_ev, atol=1e-12)
    # Centered scale stays in absolute residual units — NOT × MeanYr.
    np.testing.assert_array_equal(out["Model Sigma"].to_numpy(), scale)
    # And NOT the ratio decode (loc × MeanYr) — the family the bug confused it with.
    meanyr_clipped = X["MeanYr"].clip(lower=_MEANYR_FLOOR).to_numpy()
    assert not np.allclose(out["Projection"].to_numpy(), loc * meanyr_clipped)


def test_centered_additive_model_skew_is_finite():
    """Load-bearing: FGA dead end (docs §3.4) was NaN alpha at this layer."""
    X = _synth_player_stats(n=10, rng_seed=3)
    prob_params = _synth_prob_params(n=10, rng_seed=4)
    # Realistic alpha range including small-magnitude values; delta divides by
    # sqrt(1 + alpha^2) which is always >= 1, so no zero-divide is possible —
    # but the test pins that contract explicitly.
    prob_params["alpha"] = np.array([-3.0, -1.5, -0.5, -0.01, 0.0, 0.01, 0.5, 1.5, 3.0, 5.0])
    offset_meta = {
        "method": "eb_additive",
        "k": EB_SHRINKAGE_K,
        "global_mean": 8.69,
        "prior_col": "MeanYr",
        "games_col": "GamesPlayed",
    }

    out = _decode_skewnormal(
        prob_params.copy(),
        X,
        hist_gate=0.0,
        offset_meta=offset_meta,
        target_normalization="centered_additive_eb_meanyr_k10",
    )

    assert np.all(np.isfinite(out["Model Skew"].to_numpy())), (
        "Model Skew must be finite for every row — see "
        "docs/OVERCONFIDENCE_INVESTIGATION.md §3.4 (FGA NaN dead end)."
    )
    assert np.all(np.isfinite(out["Projection"].to_numpy()))
    assert np.all(np.isfinite(out["Model Sigma"].to_numpy()))


def test_ratio_path_model_skew_is_finite():
    """Same finite-Skew guard for the default ratio path."""
    X = _synth_player_stats(n=10, rng_seed=5)
    prob_params = _synth_prob_params(n=10, rng_seed=6)
    prob_params["alpha"] = np.array([-3.0, -1.5, -0.5, -0.01, 0.0, 0.01, 0.5, 1.5, 3.0, 5.0])

    out = _decode_skewnormal(
        prob_params.copy(),
        X,
        hist_gate=0.0,
        offset_meta=None,
        target_normalization="ratio_meanyr",
    )

    assert np.all(np.isfinite(out["Model Skew"].to_numpy()))
    assert np.all(np.isfinite(out["Projection"].to_numpy()))
    assert np.all(np.isfinite(out["Model Sigma"].to_numpy()))


def test_meanyr_floor_applied_in_ratio_path():
    """Tiny MeanYr clipped to 0.5 — must match the registry's _MEANYR_FLOOR."""
    X = pd.DataFrame(
        {
            "MeanYr": np.array([0.1, 0.0, 0.4, 10.0]),
            "MeanYr_nonzero": np.array([0.1, 0.0, 0.4, 10.0]),
            "GamesPlayed": np.array([10.0, 5.0, 30.0, 50.0]),
            "STDYr": np.array([1.0, 1.0, 1.0, 2.0]),
            "ZeroYr": np.zeros(4),
        }
    )
    prob_params = pd.DataFrame(
        {
            "loc": np.array([1.0, 1.0, 1.0, 1.0]),
            "scale": np.array([0.0, 0.0, 0.0, 0.0]),
            "alpha": np.array([0.0, 0.0, 0.0, 0.0]),
        }
    )

    out = _decode_skewnormal(
        prob_params.copy(),
        X,
        hist_gate=0.0,
        offset_meta=None,
        target_normalization="ratio_meanyr",
    )

    # loc=1 -> Model EV == MeanYr_clipped. First three should hit the 0.5 floor.
    np.testing.assert_array_equal(out["Projection"].to_numpy(), np.array([0.5, 0.5, 0.5, 10.0]))


def test_dispersion_calibrate_scales_skewnormal_sigma():
    """Route 1a-hybrid: SkewNormal ``Model Sigma`` is scaled by the dispersion factor
    after the blend, mirroring the count branch — the calibrated scale is what Gate 4
    scored on the test CSV and what the parlay builder must price.
    """
    offer_df = pd.DataFrame({"Model Sigma": [2.0, 3.0, 4.0], "Model Skew": [0.0, 1.0, -1.0]})
    _dispersion_calibrate(offer_df, "SkewNormal", 1.5, 0.0)
    np.testing.assert_allclose(offer_df["Model Sigma"].to_numpy(), [3.0, 4.5, 6.0])
    np.testing.assert_array_equal(offer_df["Model Skew"].to_numpy(), [0.0, 1.0, -1.0])


def test_dispersion_calibrate_unit_factor_leaves_skewnormal_sigma_untouched():
    offer_df = pd.DataFrame({"Model Sigma": [2.0, 3.0], "Model Skew": [0.0, 1.0]})
    _dispersion_calibrate(offer_df, "SkewNormal", 1.0, 0.0)
    np.testing.assert_array_equal(offer_df["Model Sigma"].to_numpy(), [2.0, 3.0])
    np.testing.assert_array_equal(offer_df["Model Skew"].to_numpy(), [0.0, 1.0])


def test_dispersion_calibrate_shifts_skewnormal_skew():
    """Lever 4a: the additive skew shift lands on ``Model Skew`` while sigma still scales."""
    offer_df = pd.DataFrame({"Model Sigma": [2.0, 3.0, 4.0], "Model Skew": [0.0, 1.0, -1.0]})
    _dispersion_calibrate(offer_df, "SkewNormal", 1.5, 0.8)
    np.testing.assert_allclose(offer_df["Model Sigma"].to_numpy(), [3.0, 4.5, 6.0])
    np.testing.assert_allclose(offer_df["Model Skew"].to_numpy(), [0.8, 1.8, -0.2])


def test_dispersion_calibrate_skew_only_does_not_early_return():
    """A skew-only cell (``c == 1``, ``s != 0``) must still apply — the guard checks both
    knobs, else the skew shift silently no-ops when the scale needs no change.
    """
    offer_df = pd.DataFrame({"Model Sigma": [2.0, 3.0], "Model Skew": [0.0, 1.0]})
    _dispersion_calibrate(offer_df, "SkewNormal", 1.0, 0.5)
    np.testing.assert_array_equal(offer_df["Model Sigma"].to_numpy(), [2.0, 3.0])
    np.testing.assert_allclose(offer_df["Model Skew"].to_numpy(), [0.5, 1.5])


def test_served_dispersion_dump_prices_identically_to_inference():
    """C1 round-trip: the SkewNormal predictive the scorecard decodes from the test-CSV
    dump prices byte-for-byte like the one inference serves after dispersion calibration.

    Routes both the inference params (blended sigma × c via the real
    ``_dispersion_calibrate``) and the gate-decoded params (persist's served-loc
    derivation + ``baselines.encode`` → scorecard ``_decode_sn_loc_scale``) through the
    real ``get_odds`` and asserts identical under-probabilities. A drift in the loc
    formula, the encode/decode inverse, or the apply order breaks it.
    """
    from sportstradamus.helpers.distributions import get_odds
    from sportstradamus.training.baselines import get_target_normalization
    from sportstradamus.training.scorecard import _decode_sn_loc_scale

    rng = np.random.default_rng(11)
    n = 12
    mean = rng.uniform(5.0, 25.0, n)
    blended_sigma = rng.uniform(1.5, 6.0, n)
    skew = rng.uniform(-0.5, 0.8, n)
    meanyr = rng.uniform(2.0, 20.0, n)
    line = mean + rng.uniform(-2.0, 2.0, n)
    c = 1.37

    # Inference: dispersion applied to the post-blend sigma, then priced.
    offer = pd.DataFrame({"Model Sigma": blended_sigma.copy(), "Model Skew": skew})
    _dispersion_calibrate(offer, "SkewNormal", c, 0.0)
    served_sigma = offer["Model Sigma"].to_numpy()
    under_inf = get_odds(line, mean, "SkewNormal", sigma=served_sigma, skew_alpha=skew)

    # Dump: persist holds the mean fixed, derives the scipy loc, re-encodes to model space.
    shift = (skew / np.sqrt(1 + skew**2)) * np.sqrt(2 / np.pi)
    served_loc = mean - served_sigma * shift
    strat = get_target_normalization("ratio_meanyr")
    X = pd.DataFrame({"MeanYr": meanyr})
    dumped = pd.DataFrame(
        {
            "SN_Loc": strat.encode_loc(served_loc, X, 0.0, "MeanYr"),
            "SN_Scale": strat.encode_scale(served_sigma, X, "MeanYr"),
            "SN_Alpha": skew,
            "MeanYr": meanyr,
        }
    )

    # Gate: decode, reconstruct the held-fixed mean, price through the same odds path.
    loc_g, scale_g = _decode_sn_loc_scale(dumped, "ratio_meanyr")
    decoded_mean = loc_g + scale_g * shift
    under_gate = get_odds(line, decoded_mean, "SkewNormal", sigma=scale_g, skew_alpha=skew)

    np.testing.assert_allclose(decoded_mean, mean, atol=1e-9)
    np.testing.assert_allclose(scale_g, served_sigma, atol=1e-9)
    np.testing.assert_allclose(under_gate, under_inf, atol=1e-12)


def test_served_skew_dump_prices_identically_to_inference():
    """Lever 4a round-trip: a nonzero ``skew_cal`` shift round-trips dump↔inference.

    The shift lands on ``Model Skew`` (inference) and on the dumped ``SN_Alpha`` + the
    served-loc derivation (persist, which derives loc from the *shifted* skew to hold the
    mean fixed). The gate decodes both; the priced under-probabilities must match. A drift
    in the apply order, the loc formula, or the encode/decode inverse breaks it.
    """
    from sportstradamus.helpers.distributions import get_odds, skewnormal_loc_from_mean
    from sportstradamus.training.baselines import get_target_normalization
    from sportstradamus.training.scorecard import _decode_sn_loc_scale

    rng = np.random.default_rng(13)
    n = 12
    mean = rng.uniform(5.0, 25.0, n)
    blended_sigma = rng.uniform(1.5, 6.0, n)
    skew = rng.uniform(-0.5, 0.8, n)
    meanyr = rng.uniform(2.0, 20.0, n)
    line = mean + rng.uniform(-2.0, 2.0, n)
    c, s = 1.37, 1.85

    # Inference: dispersion + skew applied to the post-blend params, then priced.
    offer = pd.DataFrame({"Model Sigma": blended_sigma.copy(), "Model Skew": skew.copy()})
    _dispersion_calibrate(offer, "SkewNormal", c, s)
    served_sigma = offer["Model Sigma"].to_numpy()
    served_skew = offer["Model Skew"].to_numpy()
    under_inf = get_odds(line, mean, "SkewNormal", sigma=served_sigma, skew_alpha=served_skew)

    # Dump: persist derives the scipy loc from the SHIFTED skew, holding the mean fixed.
    served_loc = skewnormal_loc_from_mean(mean, served_sigma, served_skew)
    strat = get_target_normalization("ratio_meanyr")
    X = pd.DataFrame({"MeanYr": meanyr})
    dumped = pd.DataFrame(
        {
            "SN_Loc": strat.encode_loc(served_loc, X, 0.0, "MeanYr"),
            "SN_Scale": strat.encode_scale(served_sigma, X, "MeanYr"),
            "SN_Alpha": served_skew,
            "MeanYr": meanyr,
        }
    )

    loc_g, scale_g = _decode_sn_loc_scale(dumped, "ratio_meanyr")
    alpha_g = dumped["SN_Alpha"].to_numpy()
    shift_g = (alpha_g / np.sqrt(1 + alpha_g**2)) * np.sqrt(2 / np.pi)
    decoded_mean = loc_g + scale_g * shift_g
    under_gate = get_odds(line, decoded_mean, "SkewNormal", sigma=scale_g, skew_alpha=alpha_g)

    assert not np.allclose(served_skew, skew)  # the shift actually moved the skew
    np.testing.assert_allclose(served_skew, skew + s, atol=1e-12)
    np.testing.assert_allclose(decoded_mean, mean, atol=1e-9)
    np.testing.assert_allclose(scale_g, served_sigma, atol=1e-9)
    np.testing.assert_allclose(under_gate, under_inf, atol=1e-12)
