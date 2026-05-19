# tests/integration/test_determinism_gate.py
"""P0.5 determinism gate: deterministic mode must be bit-reproducible.

Runs the pure fit/predict core twice on a fixed subsample of the real cached
NBA_FGA training matrix and asserts the test-split predicted distribution
parameters are exactly equal. No network (cached parquet + offline columns).
"""
import importlib.resources as pkg_resources

import numpy as np
import pandas as pd
import pytest

from sportstradamus import data
from sportstradamus.skew_normal import SkewNormal as SkewNormalDist
from sportstradamus.stats import StatsNBA
from sportstradamus.training.pipeline import (
    DETERMINISTIC_FIXED_PARAMS,
    DETERMINISTIC_SEED,
    fit_predict_params,
)

# Fixed subsample size: large enough to exercise real tree building, small
# enough to keep the gate < ~30s. Deterministic head() after a stable sort.
GATE_N_ROWS = 4000


@pytest.mark.integration
def test_deterministic_mode_is_bit_reproducible():
    parquet = pkg_resources.files(data) / "training_data/NBA_FGA.parquet"
    if not parquet.is_file():
        pytest.skip("cached NBA_FGA.parquet not present in this environment")

    M = pd.read_parquet(parquet)
    M = M.sort_values(["Date"]).head(GATE_N_ROWS).reset_index(drop=True)

    cols = StatsNBA().get_stat_columns("FGA")
    X = M[cols].copy()
    for c in ("Home", "Player position"):
        if c in X.columns:
            X[c] = X[c].astype("category")
    y = M[["Result"]]

    # Same temporal split shape train_market uses (no validation needed here).
    n_train = int(len(M) * 0.7)
    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_test = X.iloc[n_train:]

    # FGA is SkewNormal + ratio-normalized (global_mean >= 2.0 branch).
    y_train_labels = np.ravel(y_train.to_numpy()).astype(float)
    meanyr_train = X_train["MeanYr"].clip(lower=0.5).to_numpy()
    y_train_labels = np.clip(y_train_labels / meanyr_train, 0.01, None)

    def run():
        dist_obj = SkewNormalDist(stabilization="None", loss_fn="crps")
        return fit_predict_params(
            dist_obj, "SkewNormal", X_train, y_train_labels, X_test,
            params={**DETERMINISTIC_FIXED_PARAMS, "monotone_constraints": [0] * X_train.shape[1]},
            normalized=True, shape_ceiling=100.0, seed=DETERMINISTIC_SEED,
        )

    p1 = run()
    p2 = run()
    pd.testing.assert_frame_equal(p1, p2, check_exact=True)
