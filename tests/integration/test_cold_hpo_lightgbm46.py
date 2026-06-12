"""Cold-start HPO must work under lightgbm>=4.6.

Regression guard for the optuna ``LightGBMPruningCallback`` / lightgbm
``cv_agg`` incompatibility. ``LightGBMLSS.hyper_opt`` adds a
``LightGBMPruningCallback`` whose validation name is hardcoded to ``"cv_agg"``;
lightgbm>=4.6 reports cv eval results under ``"valid"``, so that callback can
never match and raises. The training pipeline's cold-start path (no warm-start
pickle, the only path when ``data/models/`` is empty) must therefore route
through :func:`run_hyper_opt`, which omits the pruning callback, not
through ``LightGBMLSS.hyper_opt``.

Marked ``@pytest.mark.integration``: runs a small but real optuna + lightgbm
cv search. Offline, no network.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.integration
def test_cold_run_hyper_opt_returns_params_without_cv_agg_error():
    import lightgbm as lgb
    from lightgbmlss.model import LightGBMLSS

    from sportstradamus.skew_normal import SkewNormal
    from sportstradamus.training.hyperparams import run_hyper_opt

    rng = np.random.default_rng(0)
    n = 600
    x0 = rng.normal(size=n)
    X = np.column_stack([x0, rng.normal(size=n), rng.normal(size=n)])
    y = 5.0 + 2.0 * x0 + rng.normal(scale=1.0, size=n)
    dtrain = lgb.Dataset(X, label=y)

    model = LightGBMLSS(SkewNormal(stabilization="None", loss_fn="crps"))
    hp_dict = {
        "num_leaves": ["int", {"low": 8, "high": 31, "log": False}],
        "learning_rate": ["float", {"low": 0.05, "high": 0.2, "log": True}],
    }

    params = run_hyper_opt(
        model,
        hp_dict,
        dtrain,
        initial_params=None,
        num_boost_round=20,
        nfold=2,
        early_stopping_rounds=10,
        max_minutes=2,
        n_trials=2,
        silence=True,
    )

    assert "opt_rounds" in params
    assert int(params["opt_rounds"]) >= 1
    assert "num_leaves" in params
