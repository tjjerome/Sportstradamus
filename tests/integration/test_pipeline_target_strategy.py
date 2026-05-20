"""P1 Task 3: target_strategy plumbing through the SkewNormal pipeline.

Validates three load-bearing claims of the P1 plan:

* Default ``target_strategy="ratio_meanyr"`` is bitwise-identical to a
  no-strategy-arg call. The legacy code path is byte-frozen.
* Switching to ``centered_additive_eb_meanyr_k10`` actually changes the
  training behavior (predicted distribution params diverge).
* The centered strategy produces a ``loc`` with a different correlation to
  ``MeanYr`` than the ratio strategy — the explicit goal of P1 is to kill
  the multiplicative amplification path-wide artifact.

Uses the same cached NBA_FGA subsample as the determinism gate and skips
when the parquet is not present (no network, no fresh fetch).
"""
from __future__ import annotations

import importlib.resources as pkg_resources

import numpy as np
import pandas as pd
import pytest

from sportstradamus import data
from sportstradamus.skew_normal import SkewNormal as SkewNormalDist
from sportstradamus.stats import StatsNBA
from sportstradamus.training import baselines
from sportstradamus.training.pipeline import (
    DETERMINISTIC_FIXED_PARAMS,
    DETERMINISTIC_SEED,
    fit_predict_params,
)

# Same fixed subsample size as the determinism gate so the two share state.
GATE_N_ROWS = 4000


def _load_fixture():
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

    n_train = int(len(M) * 0.7)
    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_test = X.iloc[n_train:]

    y_train_labels = np.ravel(y_train.to_numpy()).astype(float)
    return X_train, y_train_labels, X_test


def _params(n_cols: int) -> dict:
    """Deterministic params matching the determinism gate's stochasticity probe."""
    return {
        **DETERMINISTIC_FIXED_PARAMS,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "monotone_constraints": [0] * n_cols,
    }


def _fit(y_labels, X_train, X_test):
    dist_obj = SkewNormalDist(stabilization="None", loss_fn="crps")
    return fit_predict_params(
        dist_obj,
        "SkewNormal",
        X_train,
        y_labels,
        X_test,
        params=_params(X_train.shape[1]),
        normalized=True,
        shape_ceiling=100.0,
        seed=DETERMINISTIC_SEED,
    )


def _fit_with_strategy(slug, X_train, y_raw, X_test):
    strat = baselines.get_strategy(slug)
    global_mean = float(np.mean(y_raw))
    y_t = strat.forward(y_raw, X_train, global_mean, "MeanYr")
    dist_obj = SkewNormalDist(stabilization="None", loss_fn="crps")
    normalized = strat.start_mode_flag == "normalized"
    offset_mode = strat.start_mode_flag == "offset"
    return fit_predict_params(
        dist_obj,
        "SkewNormal",
        X_train,
        y_t,
        X_test,
        params=_params(X_train.shape[1]),
        normalized=normalized,
        shape_ceiling=100.0,
        seed=DETERMINISTIC_SEED,
        offset_mode=offset_mode,
    )


@pytest.mark.integration
def test_ratio_meanyr_default_is_bitwise_identical_to_legacy_path():
    """Guard rail: ``ratio_meanyr`` strategy must reproduce the legacy inline
    transform exactly. Default ``meditate`` runs must stay byte-identical.
    """
    X_train, y_raw, X_test = _load_fixture()

    # Legacy inline path (mirrors pipeline.py SkewNormal target build prior
    # to Task 3): clip MeanYr -> divide -> clip ratio.
    meanyr_train = X_train["MeanYr"].clip(lower=0.5).to_numpy()
    y_legacy = np.clip(y_raw / meanyr_train, 0.01, None)
    legacy = _fit(y_legacy, X_train, X_test)

    via_strategy = _fit_with_strategy("ratio_meanyr", X_train, y_raw, X_test)

    pd.testing.assert_frame_equal(legacy, via_strategy, check_exact=True)


@pytest.mark.integration
def test_centered_strategy_diverges_from_ratio_strategy():
    """Switching strategies must actually move the model — otherwise the
    plumbing is a no-op and the A/B is meaningless.
    """
    X_train, y_raw, X_test = _load_fixture()

    ratio = _fit_with_strategy("ratio_meanyr", X_train, y_raw, X_test)
    centered = _fit_with_strategy(
        "centered_additive_eb_meanyr_k10", X_train, y_raw, X_test
    )

    # Predicted raw loc must differ — that's the GBDT learning a different
    # target. Use any-not-equal rather than mean-shift, because the centered
    # path predicts residuals (≈ 0) vs the ratio path (≈ 1.0).
    assert not np.allclose(
        ratio["loc"].to_numpy(),
        centered["loc"].to_numpy(),
        rtol=1e-6,
        atol=1e-6,
    ), "centered and ratio strategies produced identical predictions"


@pytest.mark.integration
def test_centered_loc_meanyr_correlation_differs_from_ratio_loc():
    """The whole P1 motivation: centered EB kills the multiplicative-
    amplification artifact, so corr(loc, MeanYr) must move appreciably
    between the two strategies on the same data.
    """
    X_train, y_raw, X_test = _load_fixture()

    ratio = _fit_with_strategy("ratio_meanyr", X_train, y_raw, X_test)
    centered = _fit_with_strategy(
        "centered_additive_eb_meanyr_k10", X_train, y_raw, X_test
    )

    meanyr_test = X_test["MeanYr"].to_numpy()
    corr_ratio = float(np.corrcoef(ratio["loc"].to_numpy(), meanyr_test)[0, 1])
    corr_centered = float(np.corrcoef(centered["loc"].to_numpy(), meanyr_test)[0, 1])

    # Different correlations to MeanYr is the load-bearing signal — the
    # ratio target builds in division by MeanYr, the centered target does
    # not, so the GBDT learns a structurally different mapping.
    assert abs(corr_ratio - corr_centered) > 0.05, (
        f"corr(loc, MeanYr) barely moved: ratio={corr_ratio:.3f}, "
        f"centered={corr_centered:.3f}"
    )
