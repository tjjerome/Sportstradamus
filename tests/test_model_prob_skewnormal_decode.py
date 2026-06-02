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

from sportstradamus.prediction.model_prob import _decode_skewnormal
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
    np.testing.assert_array_equal(out["Model EV"].to_numpy(), expected_ev)
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

    np.testing.assert_array_equal(out["Model EV"].to_numpy(), expected_ev)
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

    np.testing.assert_allclose(out["Model EV"].to_numpy(), expected_ev, atol=1e-12)
    # Centered scale stays in absolute residual units — NOT × MeanYr.
    np.testing.assert_array_equal(out["Model Sigma"].to_numpy(), scale)
    # And critically: NOT equal to the ratio decode (which would be loc * MeanYr).
    meanyr_clipped = X["MeanYr"].clip(lower=_MEANYR_FLOOR).to_numpy()
    assert not np.allclose(out["Model EV"].to_numpy(), loc * meanyr_clipped)


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
    assert np.all(np.isfinite(out["Model EV"].to_numpy()))
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
    assert np.all(np.isfinite(out["Model EV"].to_numpy()))
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
    np.testing.assert_array_equal(out["Model EV"].to_numpy(), np.array([0.5, 0.5, 0.5, 10.0]))
