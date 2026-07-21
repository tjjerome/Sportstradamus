"""Live-path check for the count branch of ``_calibration_penalty`` (Lever 1).

Builds the per-trial served-PIT-KS closure exactly as ``train_market`` does for
a count cell under ``hpo_selection="calibrated"`` and evaluates it once on real
cached NBA_PF features: one LSS refit -> decode -> scalar dispersion-c fit ->
randomized-PIT KS. Asserts the closure returns a finite KS in [0, 1] for both a
plain NegBin and a joint ZINB (the gated frame path) — the contract
``_pick_calibrated_candidate`` relies on to rank trials.

Marked @pytest.mark.integration. Skips cleanly if the cached PF parquet is
absent (same policy as test_dpo_live_path).
"""

from __future__ import annotations

import importlib.resources as pkg_resources

import numpy as np
import pandas as pd
import pytest

from sportstradamus import data
from sportstradamus.stats import StatsNBA
from sportstradamus.training.pipeline import (
    DETERMINISTIC_FIXED_PARAMS,
    ZINB,
    NegativeBinomial,
    _calibration_penalty,
)

# Same subsample budget as the other live-path fixtures: one closure call is one
# 30-round fit, so two families stay well under the suite's per-test wall-time.
_GATE_N_ROWS = 4000


def _splits_and_params():
    parquet = pkg_resources.files(data) / "training_data/NBA_PF.parquet"
    if not parquet.is_file():
        pytest.skip("cached NBA_PF.parquet not present in this environment")

    M = pd.read_parquet(parquet)
    M = M.sort_values(["Date"]).head(_GATE_N_ROWS).reset_index(drop=True)

    cols = StatsNBA().get_stat_columns("PF")
    X = M.reindex(columns=cols).copy()
    for c in ("Home", "Player position"):
        if c in X.columns:
            X[c] = X[c].astype("category")
    y = np.ravel(M[["Result"]].to_numpy()).astype(float)

    n_train = int(len(M) * 0.7)
    splits = {
        "X_train": X.iloc[:n_train],
        "y_train_labels": y[:n_train],
        "X_validation": X.iloc[n_train:].reset_index(drop=True),
        "y_validation": pd.DataFrame({"Result": y[n_train:]}),
    }
    params = {
        **DETERMINISTIC_FIXED_PARAMS,
        "monotone_constraints": [0] * X.shape[1],
    }
    return splits, params


def _dist_info(dist: str, dist_obj) -> dict:
    # The count closure reads only these keys; target_normalization/global_mean/
    # denom_col are bound at closure build for the SkewNormal branch and unused here.
    return {
        "dist": dist,
        "dist_obj": dist_obj,
        "normalize": False,
        "shape_ceiling": 50.0,
        "offset_mode": False,
        "sn_param": "direct",
        "target_normalization": None,
        "global_mean": 0.0,
        "denom_col": "MeanYr",
    }


@pytest.mark.integration
@pytest.mark.parametrize(
    ("dist", "dist_obj_factory"),
    [
        ("NegBin", lambda: NegativeBinomial(stabilization="None", loss_fn="nll")),
        ("ZINB", lambda: ZINB(stabilization="None", loss_fn="nll")),
    ],
)
def test_count_closure_returns_finite_ks(dist, dist_obj_factory):
    splits, params = _splits_and_params()
    closure = _calibration_penalty(splits, _dist_info(dist, dist_obj_factory()))

    ks = closure(params)

    assert np.isfinite(ks)
    assert 0.0 <= ks <= 1.0
