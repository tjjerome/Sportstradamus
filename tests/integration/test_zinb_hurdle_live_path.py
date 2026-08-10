"""Live-path verification for the HurdleZINB ZINB replacement.

Trains a deterministic HurdleZINB on cached NBA_FG3M features and asserts
its predicted parameters drive the same downstream ZINB decode formula
``model_prob`` consumes (``base_ev = total_count * probs / (1 - probs)``,
``Model Gate = gate``, ``Model R = total_count``) into finite, sane values.

Downstream ``fused_loc`` (``helpers/distributions.py``) explicitly treats
the ``gate`` column as zero-inflation π — the structural-inflation
parameter, not the marginal zero rate. The HurdleZINB contract is that
``π + (1 − π) · NB(0) ≈ q`` per row, where ``q = P(Y = 0)`` from the
calibrated binary classifier — i.e. the predicted *total* zero rate
matches the calibrated full zero rate by construction. The test below
exercises that identity directly rather than asserting a mean-gate
threshold (which would conflate the inflation component with the
marginal under-fit framing in
docs/OVERCONFIDENCE_INVESTIGATION.md section 2).

Also asserts the determinism contract: two HurdleZINB.fit calls with the
same DETERMINISTIC_SEED produce bit-identical predicted parameters.

Marked @pytest.mark.integration. Skips cleanly if the cached FG3M parquet
is absent.
"""

from __future__ import annotations

import importlib.resources as pkg_resources

import numpy as np
import pandas as pd
import pytest

from sportstradamus import data
from sportstradamus.hurdle import HurdleZINB
from sportstradamus.stats import StatsNBA
from sportstradamus.training.pipeline import (
    DETERMINISTIC_FIXED_PARAMS,
    DETERMINISTIC_SEED,
)

# Same subsample size as test_determinism_gate / test_centered_target_live_path
# so the fixture wall-time stays under ~30s.
_GATE_N_ROWS = 4000

# Tolerance for the ZINB-identity reconstruction. The hurdle constructs π
# so π + (1-π)·NB(0) = q exactly per-row in continuous arithmetic; the
# clip(..., 0, 1) at the end is the only source of identity violation,
# triggered only when q < NB(0) (i.e. the binary clf disagrees with the
# NegBin's zero mass for a particular row). 0.02 is a generous mean
# tolerance.
_IDENTITY_RECONSTRUCTION_TOL: float = 0.02

# Sanity floor on the binary classifier's mean q. If q << hist_gate the
# clf is broken and π collapses everywhere, so the hurdle would degrade
# into a plain NegBin-on-positives. 0.10 is well below FG3M's ~0.33 true
# zero rate, leaving plenty of room for fit drift.
_CLF_Q_MIN: float = 0.10


@pytest.mark.integration
def test_hurdle_zinb_decodes_finite_and_reconstructs_zero_rate_identity():
    parquet = pkg_resources.files(data) / "training_data/NBA_FG3M.parquet"
    if not parquet.is_file():
        pytest.skip("cached NBA_FG3M.parquet not present in this environment")

    M = pd.read_parquet(parquet)
    M = M.sort_values(["Date"]).head(_GATE_N_ROWS).reset_index(drop=True)

    cols = StatsNBA().get_stat_columns("FG3M")
    # reindex (not M[cols]) matches production _step_build_splits: a feature_filter
    # addition not yet regenerated into the cached parquet fills NaN, not KeyError.
    X = M.reindex(columns=cols).copy()
    for c in ("Home", "Player position"):
        if c in X.columns:
            X[c] = X[c].astype("category")
    y = np.ravel(M[["Result"]].to_numpy()).astype(float)

    n_train = int(len(M) * 0.7)
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:].reset_index(drop=True)
    y_train = y[:n_train]

    params = {
        **DETERMINISTIC_FIXED_PARAMS,
        "monotone_constraints": [0] * X_train.shape[1],
    }

    model = HurdleZINB()
    model.fit(
        X_train,
        y_train,
        hp=params,
        rounds=int(params["opt_rounds"]),
        shape_ceiling=50.0,
        seed=DETERMINISTIC_SEED,
    )

    pp = model.predict(X_test, pred_type="parameters")

    # Contract: HurdleZINB outputs the same three columns a LightGBMLSS ZINB
    # would, so the downstream model_prob.py:252-257 decode is reused without
    # modification.
    assert list(pp.columns) == ["total_count", "probs", "gate"]
    assert len(pp) == len(X_test)
    assert pp.index.equals(X_test.index)

    total_count = pp["total_count"].to_numpy()
    probs = pp["probs"].to_numpy()
    gate = pp["gate"].to_numpy()

    # Drive the LIVE decode formula model_prob.py uses for ZINB. If any of
    # this is NaN/inf, model_prob will publish broken EVs.
    base_ev = total_count * probs / (1.0 - probs)
    assert np.isfinite(total_count).all(), "total_count must be finite for every row"
    assert np.isfinite(probs).all(), "probs must be finite for every row"
    assert ((probs > 0.0) & (probs < 1.0)).all(), "probs must lie in (0, 1) so 1-probs > 0"
    assert np.isfinite(base_ev).all(), (
        "base_ev = total_count * probs / (1 - probs) must be finite; "
        "non-finite values here surface as broken Model EV in the live slate."
    )
    assert (base_ev > 0).all(), "base_ev must be positive for a NegBin count"

    # Derived-π gate must lie in [0, 1] by construction.
    assert ((gate >= 0.0) & (gate <= 1.0)).all(), "gate must lie in [0, 1]"

    # The whole point of derived-π: per-row, the hurdle's *reconstructed*
    # P(Y = 0) under the ZINB identity must match the calibrated binary
    # classifier's q. This is the contract HurdleZINB exists to satisfy.
    nb0 = (1.0 - probs) ** total_count
    reconstructed_zero = gate + (1.0 - gate) * nb0
    q = model.clf.predict(X_test)
    q_mean = float(np.mean(q))
    assert q_mean >= _CLF_Q_MIN, (
        f"Binary clf mean q = {q_mean:.3f} is below {_CLF_Q_MIN:.2f} — "
        "the calibrated zero classifier is broken; derived-π would collapse "
        "everywhere."
    )
    diff = float(np.mean(np.abs(reconstructed_zero - q)))
    assert diff <= _IDENTITY_RECONSTRUCTION_TOL, (
        f"Per-row mean |π + (1−π)·NB(0) − q| = {diff:.4f} exceeds "
        f"{_IDENTITY_RECONSTRUCTION_TOL:.3f}. The hurdle's defining identity "
        "(reconstructed total P(Y=0) matches the calibrated classifier) is "
        "broken — π is mis-derived in src/sportstradamus/hurdle.py."
    )


@pytest.mark.integration
def test_hurdle_zinb_live_path_is_deterministic():
    """Two fits with DETERMINISTIC_SEED must produce bit-identical parameters.

    Mirrors the SkewNormal determinism guard in test_determinism_gate.py.
    Without this, the A/B compression_eval verdict for hurdle vs joint
    cannot be trusted — repeats the lesson from
    docs/CENTERED_TARGET_NEGATIVE_RESULT.md.
    """
    parquet = pkg_resources.files(data) / "training_data/NBA_FG3M.parquet"
    if not parquet.is_file():
        pytest.skip("cached NBA_FG3M.parquet not present in this environment")

    M = pd.read_parquet(parquet)
    M = M.sort_values(["Date"]).head(_GATE_N_ROWS).reset_index(drop=True)
    cols = StatsNBA().get_stat_columns("FG3M")
    # reindex (not M[cols]) matches production _step_build_splits: a feature_filter
    # addition not yet regenerated into the cached parquet fills NaN, not KeyError.
    X = M.reindex(columns=cols).copy()
    for c in ("Home", "Player position"):
        if c in X.columns:
            X[c] = X[c].astype("category")
    y = np.ravel(M[["Result"]].to_numpy()).astype(float)
    n_train = int(len(M) * 0.7)
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:].reset_index(drop=True)
    y_train = y[:n_train]

    params = {
        **DETERMINISTIC_FIXED_PARAMS,
        "monotone_constraints": [0] * X_train.shape[1],
    }

    def _fit_predict() -> pd.DataFrame:
        h = HurdleZINB()
        h.fit(
            X_train,
            y_train,
            hp=params,
            rounds=int(params["opt_rounds"]),
            shape_ceiling=50.0,
            seed=DETERMINISTIC_SEED,
        )
        return h.predict(X_test, pred_type="parameters")

    pp_a = _fit_predict()
    pp_b = _fit_predict()
    pd.testing.assert_frame_equal(pp_a, pp_b, check_exact=True)
