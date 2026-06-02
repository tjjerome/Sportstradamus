"""Live-path verification for the centered_additive_eb_meanyr_k10 strategy.

Trains a deterministic SkewNormal model on cached NBA_FGA features, builds a
pickle-shaped filedict with offset_meta + target_normalization, and drives it
through model_prob._decode_skewnormal. Asserts Model Skew is finite for
every row (the FGA NaN dead end from docs/OVERCONFIDENCE_INVESTIGATION.md
3.4 must not recur) and Model EV matches the EB-additive formula.

Marked @pytest.mark.integration, no network -- relies on the cached
NBA_FGA.parquet fixture. Skips cleanly when the parquet is absent.
"""

from __future__ import annotations

import importlib.resources as pkg_resources

import numpy as np
import pandas as pd
import pytest

from sportstradamus import data
from sportstradamus.prediction.model_prob import _decode_skewnormal
from sportstradamus.skew_normal import SkewNormal as SkewNormalDist
from sportstradamus.stats import StatsNBA
from sportstradamus.training.baselines import (
    EB_SHRINKAGE_K,
    compute_eb_prior,
    get_target_normalization,
)
from sportstradamus.training.pipeline import (
    DETERMINISTIC_FIXED_PARAMS,
    DETERMINISTIC_SEED,
    fit_predict_params,
)

# Same subsample size as test_determinism_gate.py -- large enough to exercise
# tree building, small enough to keep the test < ~30s.
_GATE_N_ROWS = 4000

# High-volume quintile threshold for the amplification guard. Centered
# strategy must not collapse top-quintile EV below 70% of MeanYr.
_TOP_QUINTILE_EV_FRACTION_FLOOR = 0.7


@pytest.mark.integration
def test_centered_additive_skewnormal_decodes_with_finite_alpha_and_no_compression():
    parquet = pkg_resources.files(data) / "training_data/NBA_FGA.parquet"
    if not parquet.is_file():
        pytest.skip("cached NBA_FGA.parquet not present in this environment")

    M = pd.read_parquet(parquet)
    M = M.sort_values(["Date"]).head(_GATE_N_ROWS).reset_index(drop=True)

    cols = StatsNBA().get_stat_columns("FGA")
    X = M[cols].copy()
    for c in ("Home", "Player position"):
        if c in X.columns:
            X[c] = X[c].astype("category")
    y = M[["Result"]]

    n_train = int(len(M) * 0.7)
    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_test = X.iloc[n_train:].reset_index(drop=True)

    global_mean = float(np.ravel(y_train.to_numpy()).mean())
    denom_col = "MeanYr"
    strategy = get_target_normalization("centered_additive_eb_meanyr_k10")

    y_train_labels = np.ravel(y_train.to_numpy()).astype(float)
    y_train_labels = strategy.forward(y_train_labels, X_train, global_mean, denom_col)

    dist_obj = SkewNormalDist(stabilization="None", loss_fn="crps")
    params = {
        **DETERMINISTIC_FIXED_PARAMS,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "monotone_constraints": [0] * X_train.shape[1],
    }
    prob_params = fit_predict_params(
        dist_obj,
        "SkewNormal",
        X_train,
        y_train_labels,
        X_test,
        params=params,
        normalized=False,
        shape_ceiling=100.0,
        seed=DETERMINISTIC_SEED,
        offset_mode=True,
    )

    # Drive through the LIVE decode path (the same helper model_prob uses).
    offset_meta = {
        "method": "eb_additive",
        "k": EB_SHRINKAGE_K,
        "global_mean": global_mean,
        "prior_col": denom_col,
        "games_col": "GamesPlayed",
    }
    # Capture raw loc/scale/alpha BEFORE _decode_skewnormal mutates the frame
    # in place. The decode helper overwrites the "loc"/"scale" columns is a
    # no-op on this path (it only sets Model EV / Model Sigma / Model Skew),
    # but reading the raw arrays up front keeps the formula check independent.
    raw_loc = prob_params["loc"].to_numpy().copy()
    raw_scale = prob_params["scale"].to_numpy().copy()
    raw_alpha = prob_params["alpha"].to_numpy().copy()

    decoded = _decode_skewnormal(
        prob_params,
        X_test,
        hist_gate=0.0,
        offset_meta=offset_meta,
        target_normalization="centered_additive_eb_meanyr_k10",
    )

    # Assertion 1: Model Skew finite for every row (FGA NaN guard).
    assert decoded["Model Skew"].notna().all(), (
        "Model Skew (alpha) must be finite under the centered strategy -- "
        "NaN here is the FGA dead end from OVERCONFIDENCE_INVESTIGATION.md 3.4."
    )
    assert np.isfinite(decoded["Model Skew"].to_numpy()).all(), (
        "Model Skew must be finite (no inf) for every row."
    )

    # Assertion 2: Model EV matches EB_prior + loc + scale*delta*sqrt(2/pi).
    eb = compute_eb_prior(
        X_test[denom_col].clip(lower=0.5).to_numpy(),
        X_test["GamesPlayed"].clip(lower=0).to_numpy(),
        global_mean,
        EB_SHRINKAGE_K,
    )
    delta = raw_alpha / np.sqrt(1 + raw_alpha**2)
    expected_ev = eb + raw_loc + raw_scale * delta * np.sqrt(2 / np.pi)
    np.testing.assert_allclose(
        decoded["Model EV"].to_numpy(),
        expected_ev,
        atol=1e-9,
        err_msg=("Live decode formula does not match the train-side EB-additive transform."),
    )

    # Assertion 3: top-quintile EV not collapsed (anti-amplification guard).
    meanyr = X_test["MeanYr"].to_numpy()
    top_q = meanyr >= np.quantile(meanyr, 0.8)
    if top_q.sum() > 0:
        ratio = decoded["Model EV"].to_numpy()[top_q] / meanyr[top_q]
        assert ratio.mean() >= _TOP_QUINTILE_EV_FRACTION_FLOOR, (
            f"Top-quintile mean Model EV / MeanYr = {ratio.mean():.3f}; "
            f"centered strategy should not compress high-volume players below "
            f"{_TOP_QUINTILE_EV_FRACTION_FLOOR}x MeanYr."
        )

    # Assertion 4: overall magnitudes are sane.
    assert decoded["Model EV"].notna().all()
    assert decoded["Model EV"].mean() > 0
