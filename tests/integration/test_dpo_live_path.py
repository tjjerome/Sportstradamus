"""Live-path verification for the Double Poisson (DPO) count family.

Trains a deterministic DPO on cached NBA_PF features (the Phase-0
DP-mandatory pilot cell) through the same ``fit_lss_model`` /
``predict_lss_params`` path production ``train_market`` uses, then drives
the predicted parameters through the serve-side decode chain
(``decode_predictive_mean`` -> ``get_odds`` / ``get_push_prob`` /
``fused_loc``) and asserts finite, sane values.

The defining DPO contract: ``mu`` is only the *approximate* mean, so the
serve decode must recover the exact truncated-series mean. Asserted here
by cross-checking the numpy serve kernel against the torch training
family on the same predicted parameters — the two kernels are
deliberately separate implementations (the serve import graph stays
torch-free), so drift between them fails loudly here before it can skew
a live slate.

Also asserts the determinism contract: two fits with DETERMINISTIC_SEED
produce bit-identical predicted parameters.

Marked @pytest.mark.integration. Skips cleanly if the cached PF parquet
is absent.
"""

from __future__ import annotations

import importlib.resources as pkg_resources

import numpy as np
import pandas as pd
import pytest

from sportstradamus import data
from sportstradamus.double_poisson import DoublePoisson, DoublePoissonTorch
from sportstradamus.helpers.distributions import (
    _DP_PHI_CEILING,
    _DP_PHI_FLOOR,
    decode_predictive_mean,
    fused_loc,
    get_odds,
    get_push_prob,
)
from sportstradamus.stats import StatsNBA
from sportstradamus.training.pipeline import (
    DETERMINISTIC_FIXED_PARAMS,
    DETERMINISTIC_SEED,
    fit_lss_model,
    predict_lss_params,
)

# Same subsample size as test_zinb_hurdle_live_path so the fixture
# wall-time stays under ~30s.
_GATE_N_ROWS = 4000

# numpy float64 serve kernel vs torch float32 training kernel on the same
# (mu, phi): both are truncated series over the same grid policy, so the
# exact means must agree to float32 precision.
_MEAN_CROSS_KERNEL_RTOL = 1e-3

# Rows cross-checked against the torch kernel (full-frame torch decode is
# needless — drift is systematic, not row-local).
_CROSS_KERNEL_N_ROWS = 32


def _load_pf_splits():
    parquet = pkg_resources.files(data) / "training_data/NBA_PF.parquet"
    if not parquet.is_file():
        pytest.skip("cached NBA_PF.parquet not present in this environment")

    M = pd.read_parquet(parquet)
    M = M.sort_values(["Date"]).head(_GATE_N_ROWS).reset_index(drop=True)

    cols = StatsNBA().get_stat_columns("PF")
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
        DoublePoisson(stabilization="None", loss_fn="nll"),
        "DPO",
        X_train,
        y_train,
        params,
        normalized=False,
        shape_ceiling=50.0,  # inert for DPO: phi is hard-bounded in-family
        seed=DETERMINISTIC_SEED,
    )
    return predict_lss_params(model, "DPO", X_test, normalized=False)


@pytest.mark.integration
def test_dpo_live_path_decodes_finite_and_recovers_exact_mean():
    X_train, X_test, y_train = _load_pf_splits()
    pp = _fit_predict(X_train, X_test, y_train)

    # Contract: the LightGBMLSS param frame carries exactly the two columns
    # every downstream consumer (decode_predictive_mean, scorecard
    # _infer_dist_from_columns via DP_MU/DP_PHI, model_prob) keys on.
    assert list(pp.columns) == ["mu", "phi"]
    assert len(pp) == len(X_test)
    assert pp.index.equals(X_test.index)

    mu = pp["mu"].to_numpy()
    phi = pp["phi"].to_numpy()
    assert np.isfinite(mu).all() and (mu > 0).all()
    assert np.isfinite(phi).all()
    assert ((phi >= _DP_PHI_FLOOR) & (phi <= _DP_PHI_CEILING)).all()

    decoded = decode_predictive_mean(pp, "DPO")
    ev = np.asarray(decoded.ev)
    assert np.isfinite(ev).all() and (ev > 0).all(), (
        "exact-mean decode must be finite/positive; non-finite values here "
        "surface as broken Projection in the live slate."
    )
    assert decoded.gate is None and decoded.r is None
    np.testing.assert_allclose(np.asarray(decoded.phi), phi, rtol=1e-9)

    # PF results run ~2/game; a deterministic 30-round fit must still land
    # the cohort mean in the right decade.
    assert 0.5 < float(ev.mean()) < 6.0

    # Defining DPO identity: numpy serve decode == torch training-family
    # exact mean on the same predicted parameters.
    import torch

    idx = np.linspace(0, len(pp) - 1, _CROSS_KERNEL_N_ROWS).astype(int)
    torch_mean = DoublePoissonTorch(
        torch.tensor(mu[idx]), torch.tensor(phi[idx])
    ).mean.numpy()
    np.testing.assert_allclose(ev[idx], torch_mean, rtol=_MEAN_CROSS_KERNEL_RTOL)

    # Serve parity: the model_prob chain (blend -> odds -> push) must stay
    # finite on real predicted parameters.
    line = np.round(ev) + 0.5
    under = get_odds(line, ev, "DPO", cv=0.5, phi=phi)
    assert np.isfinite(under).all()
    assert ((under > 0.0) & (under < 1.0)).all()

    mean_blend, phi_blend, gate_blend = fused_loc(
        0.6, ev, ev * 0.95, 0.5, "DPO", phi=phi
    )
    assert np.isfinite(mean_blend).all() and (mean_blend > 0).all()
    assert np.isfinite(phi_blend).all() and (phi_blend > 0).all()
    assert gate_blend is None

    push = get_push_prob(np.round(ev), ev, "DPO", cv=0.5, phi=phi)
    assert np.isfinite(push).all()
    assert ((push > 0.0) & (push < 1.0)).all(), (
        "integer lines on a count family must carry non-degenerate push mass"
    )


@pytest.mark.integration
def test_dpo_live_path_is_deterministic():
    """Two fits with DETERMINISTIC_SEED must produce bit-identical parameters.

    Mirrors the HurdleZINB determinism guard. Without this, deterministic
    board ranks for DPO corners cannot be trusted.
    """
    X_train, X_test, y_train = _load_pf_splits()
    pp_a = _fit_predict(X_train, X_test, y_train)
    pp_b = _fit_predict(X_train, X_test, y_train)
    pd.testing.assert_frame_equal(pp_a, pp_b, check_exact=True)
