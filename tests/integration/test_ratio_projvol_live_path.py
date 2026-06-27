"""Live-path verification for the ``ratio_projvol`` normalization.

Trains a deterministic SkewNormal model on the cached NBA_FGA features with the
``ratio_projvol`` forward transform, then drives the raw params through
``model_prob._decode_skewnormal`` — the same helper the betting path uses.
Asserts the decode reads the *persisted* projected-volume denominator (not the
MeanYr fallback) and that Model EV equals ``raw_sn_mean * denominator``.

Marked @pytest.mark.integration, no network — relies on the cached
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
from sportstradamus.training import baselines
from sportstradamus.training.pipeline import (
    DETERMINISTIC_FIXED_PARAMS,
    DETERMINISTIC_SEED,
    _step_build_splits,
    fit_predict_params,
)

_GATE_N_ROWS = 4000
_PROJ_MIN = "Player proj MIN mean"


@pytest.mark.integration
def test_ratio_projvol_live_decode_uses_persisted_projected_volume():
    parquet = pkg_resources.files(data) / "training_data/NBA_FGA.parquet"
    if not parquet.is_file():
        pytest.skip("cached NBA_FGA.parquet not present in this environment")
    M = pd.read_parquet(parquet).sort_values(["Date"]).head(_GATE_N_ROWS).reset_index(drop=True)
    if _PROJ_MIN not in M.columns:
        pytest.skip("fixture lacks the projected-volume column")

    splits = _step_build_splits(M, StatsNBA(), "FGA", "ratio_projvol")
    X_train, X_test = splits["X_train"], splits["X_test"].reset_index(drop=True)
    y_train_labels = splits["y_train_labels"].astype(float)
    global_mean = float(y_train_labels.mean())

    denom_col = baselines.resolve_denom_col(
        "ratio_projvol", "NBA", "FGA", X_train.columns, zero_inflated=False
    )
    strategy = baselines.get_target_normalization("ratio_projvol")
    y_t = strategy.forward(y_train_labels, X_train, global_mean, denom_col)

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
        y_t,
        X_test,
        params=params,
        normalized=True,
        shape_ceiling=100.0,
        seed=DETERMINISTIC_SEED,
    )

    raw_loc = prob_params["loc"].to_numpy().copy()
    raw_scale = prob_params["scale"].to_numpy().copy()
    raw_alpha = prob_params["alpha"].to_numpy().copy()

    offset_meta = {"method": "projvol_ratio", "denom_col": denom_col, "fallback_col": "MeanYr"}
    decoded = _decode_skewnormal(
        prob_params,
        X_test,
        hist_gate=0.0,
        offset_meta=offset_meta,
        target_normalization="ratio_projvol",
    )

    # Decode must multiply the raw SkewNormal mean by the projected-volume
    # denominator (with the MeanYr fallback), exactly as the train forward divided.
    denom = baselines._projvol_denominator(X_test, denom_col)
    delta = raw_alpha / np.sqrt(1 + raw_alpha**2)
    raw_sn_mean = raw_loc + raw_scale * delta * np.sqrt(2 / np.pi)
    expected_ev = raw_sn_mean * denom

    assert decoded["Projection"].notna().all()
    np.testing.assert_allclose(
        decoded["Projection"].to_numpy(),
        expected_ev,
        atol=1e-9,
        err_msg="Live decode did not multiply by the persisted projected-volume denominator.",
    )

    # Prove the decode used the proj denominator, not the MeanYr fallback path:
    # the two denominators differ on rows with a real projection, so the EVs must.
    meanyr_denom = X_test["MeanYr"].clip(lower=0.5).to_numpy()
    assert not np.allclose(raw_sn_mean * denom, raw_sn_mean * meanyr_denom)
