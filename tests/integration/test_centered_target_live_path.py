"""Live-path verification for the centered-additive SkewNormal strategies.

Trains a deterministic SkewNormal model on cached NBA_FGA features under each
``centered_additive`` target normalization, builds the pickle-shaped ``offset_meta`` from
the registry, and drives it through ``model_prob._decode_skewnormal`` with the
serve-derived ``offset_mode``. Guards the invariants a serve-decode drift has broken:

* ``Model Skew`` stays finite for every row (the FGA NaN dead end,
  ``docs/OVERCONFIDENCE_INVESTIGATION.md`` 3.4).
* The decoded mean is not inflated. A serve seeding drift (the prediction path deciding
  ``offset_mode`` from a hardcoded ``method`` string) doubled ``centered_additive_mean10``
  projections on 2026-06-24; a >= 1.5x median ``Projection/MeanYr`` fails here. This is the
  end-to-end backstop, parametrized over both centered strategies so the newer
  ``mean10_additive`` family is no longer uncovered.

Marked @pytest.mark.integration, no network -- relies on the cached NBA_FGA.parquet
fixture. Skips cleanly when the parquet is absent.
"""

from __future__ import annotations

import importlib.resources as pkg_resources

import numpy as np
import pandas as pd
import pytest

from sportstradamus import data
from sportstradamus.prediction.model_prob import _decode_skewnormal, _serve_offset_mode
from sportstradamus.skew_normal import SkewNormal as SkewNormalDist
from sportstradamus.stats import StatsNBA
from sportstradamus.training.baselines import get_target_normalization
from sportstradamus.training.pipeline import (
    DETERMINISTIC_FIXED_PARAMS,
    DETERMINISTIC_SEED,
    fit_predict_params,
)

# Same subsample size as test_determinism_gate.py -- large enough to exercise tree
# building, small enough to keep the test < ~30s.
_GATE_N_ROWS = 4000

# High-volume quintile threshold for the amplification guard. Centered strategy must not
# collapse top-quintile EV below 70% of MeanYr.
_TOP_QUINTILE_EV_FRACTION_FLOOR = 0.7

# A serve seeding/denom drift doubles the projection; the centered mean must track MeanYr.
_MAX_MEDIAN_INFLATION = 1.5


@pytest.mark.integration
@pytest.mark.parametrize("slug", ["centered_additive_eb_meanyr_k10", "centered_additive_mean10"])
def test_centered_skewnormal_live_decode_finite_and_uninflated(slug):
    parquet = pkg_resources.files(data) / "training_data/NBA_FGA.parquet"
    if not parquet.is_file():
        pytest.skip("cached NBA_FGA.parquet not present in this environment")

    M = pd.read_parquet(parquet).sort_values(["Date"]).head(_GATE_N_ROWS).reset_index(drop=True)
    cols = StatsNBA().get_stat_columns("FGA")
    # reindex (not M[cols]) matches production _step_build_splits: a feature_filter
    # addition not yet regenerated into the cached parquet fills NaN, not KeyError.
    X = M.reindex(columns=cols).copy()
    for c in ("Home", "Player position"):
        if c in X.columns:
            X[c] = X[c].astype("category")
    y = M[["Result"]]

    n_train = int(len(M) * 0.7)
    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_test = X.iloc[n_train:].reset_index(drop=True)

    global_mean = float(np.ravel(y_train.to_numpy()).mean())
    denom_col = "MeanYr"
    strategy = get_target_normalization(slug)
    offset_mode = strategy.start_mode_flag == "offset"

    y_train_labels = strategy.forward(
        np.ravel(y_train.to_numpy()).astype(float), X_train, global_mean, denom_col
    )

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
        offset_mode=offset_mode,
    )

    # Drive the LIVE decode with the registry-persisted offset_meta and the serve-derived
    # seeding flag — exactly what model_prob does at predict time.
    assert _serve_offset_mode("SkewNormal", slug) is offset_mode
    offset_meta = strategy.offset_meta(global_mean, denom_col)
    decoded = _decode_skewnormal(
        prob_params, X_test, hist_gate=0.0, offset_meta=offset_meta, target_normalization=slug
    )

    # FGA NaN guard.
    assert np.isfinite(decoded["Model Skew"].to_numpy()).all(), (
        "Model Skew must be finite for every row -- see "
        "docs/OVERCONFIDENCE_INVESTIGATION.md 3.4 (FGA NaN dead end)."
    )
    proj = decoded["Projection"].to_numpy()
    assert decoded["Projection"].notna().all() and (proj > 0).all()

    # Anti-inflation: the centered mean tracks MeanYr; a seeding/denom drift doubles it.
    meanyr = X_test["MeanYr"].to_numpy()
    median_ratio = float(np.median(proj / meanyr))
    assert median_ratio < _MAX_MEDIAN_INFLATION, (
        f"{slug}: median Projection/MeanYr = {median_ratio:.2f} >= {_MAX_MEDIAN_INFLATION} -- "
        "a serve seeding/denom drift (the 2026-06-24 ~2x overconfidence bug) is back."
    )

    # Anti-compression: top-quintile EV not collapsed below the MeanYr floor.
    top_q = meanyr >= np.quantile(meanyr, 0.8)
    if top_q.sum() > 0:
        ratio = proj[top_q] / meanyr[top_q]
        assert ratio.mean() >= _TOP_QUINTILE_EV_FRACTION_FLOOR, (
            f"{slug}: top-quintile mean Projection/MeanYr = {ratio.mean():.3f} < "
            f"{_TOP_QUINTILE_EV_FRACTION_FLOOR}; centered strategy compresses high-volume players."
        )
