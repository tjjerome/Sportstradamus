"""Live-path verification for the centered-parametrization SkewNormal.

Trains a deterministic CenteredSkewNormal on cached WNBA_PA features through
the production ``fit_lss_model`` / ``predict_lss_params`` path with
``sn_param="centered"`` and asserts the zero-serving-delta contract: the
predicted parameter frame is byte-compatible with a direct SkewNormal's
(``loc`` / ``scale`` / ``alpha`` columns), finite, and drives the same
downstream odds math.

Also asserts the determinism contract (bit-identical repeat fits) — the
model-strategy sweep ranks centered corners deterministically.

Marked @pytest.mark.integration. Skips cleanly if the cached PA parquet is
absent.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import math

import numpy as np
import pandas as pd
import pytest

from sportstradamus import data
from sportstradamus.helpers.distributions import get_odds
from sportstradamus.skew_normal_centered import CenteredSkewNormal
from sportstradamus.stats import StatsWNBA
from sportstradamus.training.pipeline import (
    DETERMINISTIC_FIXED_PARAMS,
    DETERMINISTIC_SEED,
    fit_lss_model,
    predict_lss_params,
)

_GATE_N_ROWS = 4000


def _load_pa_splits():
    parquet = pkg_resources.files(data) / "training_data/WNBA_PA.parquet"
    if not parquet.is_file():
        pytest.skip("cached WNBA_PA.parquet not present in this environment")

    M = pd.read_parquet(parquet)
    M = M.sort_values(["Date"]).head(_GATE_N_ROWS).reset_index(drop=True)

    cols = StatsWNBA().get_stat_columns("PA")
    # reindex (not M[cols]) matches production _step_build_splits: a feature_filter
    # addition not yet regenerated into the cached parquet fills NaN, not KeyError.
    X = M.reindex(columns=cols).copy()
    for c in ("Home", "Player position"):
        if c in X.columns:
            X[c] = X[c].astype("category")
    y = np.ravel(M[["Result"]].to_numpy()).astype(float)

    n_train = int(len(M) * 0.7)
    return X.iloc[:n_train], X.iloc[n_train:].reset_index(drop=True), y[:n_train]


def _fit_predict(X_train, X_test, y_train) -> pd.DataFrame:
    params = {
        **DETERMINISTIC_FIXED_PARAMS,
        "monotone_constraints": [0] * X_train.shape[1],
    }
    model = fit_lss_model(
        CenteredSkewNormal(loss_fn="nll"),
        "SkewNormal",
        X_train,
        y_train,
        params,
        normalized=False,
        shape_ceiling=100.0,
        seed=DETERMINISTIC_SEED,
        sn_param="centered",
    )
    return predict_lss_params(model, "SkewNormal", X_test, normalized=False, sn_param="centered")


@pytest.mark.integration
def test_centered_sn_live_path_reemits_direct_frame():
    X_train, X_test, y_train = _load_pa_splits()
    pp = _fit_predict(X_train, X_test, y_train)

    # Zero-serving-delta contract: every dist == "SkewNormal" consumer keys on
    # exactly this frame; a centered pickle must be indistinguishable here.
    assert list(pp.columns) == ["loc", "scale", "alpha"]
    assert len(pp) == len(X_test)
    assert pp.index.equals(X_test.index)

    loc = pp["loc"].to_numpy()
    scale = pp["scale"].to_numpy()
    alpha = pp["alpha"].to_numpy()
    assert np.isfinite(loc).all()
    assert np.isfinite(scale).all() and (scale > 0).all()
    assert np.isfinite(alpha).all()

    # Mean recovered from the direct frame must sit in the target's decade —
    # the centered mean head trained on raw PA (~25/game).
    delta = alpha / np.sqrt(1.0 + alpha**2)
    mean_back = loc + scale * delta * math.sqrt(2.0 / math.pi)
    assert 10.0 < float(mean_back.mean()) < 45.0

    # Serve parity: the SkewNormal odds path consumes the frame unchanged.
    # Closed interval: a deterministic 30-round fit can rail the sd head on a
    # few rows, saturating the CDF to exactly 0/1 — correct math the serve
    # path clips anyway; corner quality is the pilots' question, not this
    # contract's.
    line = np.round(mean_back) + 0.5
    under = get_odds(line, mean_back, "SkewNormal", cv=0.3, sigma=scale, skew_alpha=alpha, step=0.5)
    assert np.isfinite(under).all()
    assert ((under >= 0.0) & (under <= 1.0)).all()
    assert 0.05 < float(np.median(under)) < 0.95


@pytest.mark.integration
def test_centered_sn_live_path_is_deterministic():
    """Two fits with DETERMINISTIC_SEED must produce bit-identical parameters."""
    X_train, X_test, y_train = _load_pa_splits()
    pp_a = _fit_predict(X_train, X_test, y_train)
    pp_b = _fit_predict(X_train, X_test, y_train)
    pd.testing.assert_frame_equal(pp_a, pp_b, check_exact=True)
