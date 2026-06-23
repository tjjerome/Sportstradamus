"""End-to-end coverage for the calibration-aware HP levers (research brief).

Lever 1 has two halves that only exist together at train time:
  * `run_hyper_opt(top_k=K)` returns the K lowest-CRPS trials (a real optuna+lightgbm cv search).
  * `_select_calibrated_hp` refits each candidate and re-ranks by served validation PIT-KS.

Marked ``@pytest.mark.integration``: real lightgbm fits, offline, no network.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.integration
def test_run_hyper_opt_top_k_returns_ranked_candidate_list():
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

    candidates = run_hyper_opt(
        model,
        hp_dict,
        dtrain,
        num_boost_round=20,
        nfold=2,
        early_stopping_rounds=10,
        max_minutes=2,
        n_trials=4,
        silence=True,
        top_k=3,
    )

    assert isinstance(candidates, list)
    assert 1 <= len(candidates) <= 3
    losses = [c["cv_loss"] for c in candidates]
    assert losses == sorted(losses)  # ascending CV loss
    for cand in candidates:
        assert int(cand["opt_rounds"]) >= 1
        assert "num_leaves" in cand


@pytest.mark.integration
def test_select_calibrated_hp_scores_and_picks_skewnormal_candidate():
    from sportstradamus.skew_normal import SkewNormal
    from sportstradamus.training import baselines
    from sportstradamus.training.pipeline import DETERMINISTIC_FIXED_PARAMS, _select_calibrated_hp

    rng = np.random.default_rng(3)
    n = 1500
    mean_yr = rng.uniform(8.0, 25.0, size=n)
    signal = rng.normal(size=n)
    import pandas as pd

    X = pd.DataFrame(
        {
            "MeanYr": mean_yr,
            "STDYr": 0.4 * mean_yr,
            "ZeroYr": 0.0,
            "GlobalMean": mean_yr.mean(),
            "Signal": signal,
        }
    )
    rate = np.clip(1.0 + 0.15 * signal + rng.normal(scale=0.25, size=n), 0.2, None)
    y_raw = rate * mean_yr  # outcome ≈ per-unit rate × volume
    y_norm = y_raw / mean_yr  # ratio_meanyr forward

    cut = 1000
    splits = {
        "X_train": X.iloc[:cut],
        "y_train_labels": y_norm[:cut],
        "X_validation": X.iloc[cut:].reset_index(drop=True),
        "y_validation": pd.DataFrame({"Result": y_raw[cut:]}),
    }
    dist_info = {
        "dist": "SkewNormal",
        "dist_obj": SkewNormal(stabilization="None", loss_fn="crps"),
        "target_normalization": baselines.get_target_normalization("ratio_meanyr"),
        "global_mean": float(y_raw[:cut].mean()),
        "denom_col": "MeanYr",
        "normalize": True,
        "offset_mode": False,
        "shape_ceiling": None,
    }
    n_feat = X.shape[1]
    candidates = [
        {
            **DETERMINISTIC_FIXED_PARAMS,
            "opt_rounds": 20,
            "num_leaves": nl,
            "monotone_constraints": [0] * n_feat,
            "cv_loss": loss,
        }
        for nl, loss in [(8, 1.00), (31, 0.92), (63, 0.97)]
    ]

    winner = _select_calibrated_hp(candidates, splits, dist_info)

    assert "cv_loss" not in winner and "pit_ks" not in winner
    assert winner["num_leaves"] in (8, 31, 63)
    for cand in candidates:
        assert 0.0 <= cand["pit_ks"] <= 1.0  # every candidate scored a finite served PIT-KS
