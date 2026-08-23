"""End-to-end coverage for the calibration-aware HP levers (research brief).

Lever 1 (search-gated) has two halves that only exist together at train time:
  * `run_hyper_opt(calibration_penalty=…)` search-gates the Optuna objective on validation PIT-KS
    and returns every completed trial tagged with raw `cv_loss` + `pit_ks` (a real optuna+lightgbm
    cv search).
  * `_calibration_penalty` builds the per-trial served-PIT-KS closure; `_pick_calibrated_candidate`
    makes the final feasible pick.

Marked ``@pytest.mark.integration``: real lightgbm fits, offline, no network.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.integration
def test_run_hyper_opt_search_gate_returns_pit_ks_tagged_candidates():
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

    # Stand-in calibration gate: more leaves reads as worse-calibrated, so the search-gated
    # objective is exercised without a full per-trial refit (that path is covered below).
    def pit_ks(params):
        return 0.02 + 0.001 * params["num_leaves"]

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
        calibration_penalty=pit_ks,
        penalty_threshold=0.05,
    )

    assert isinstance(candidates, list) and candidates
    losses = [c["cv_loss"] for c in candidates]
    assert losses == sorted(losses)  # ascending raw CV-CRPS, not the penalized objective
    for cand in candidates:
        assert int(cand["opt_rounds"]) >= 1
        assert "num_leaves" in cand
        assert 0.0 <= cand["pit_ks"] <= 1.0  # the gate's PIT-KS is stashed per trial


@pytest.mark.integration
def test_calibration_penalty_scores_skewnormal_then_pick_selects(monkeypatch):
    import pandas as pd

    from sportstradamus.skew_normal import SkewNormal
    from sportstradamus.training import baselines
    from sportstradamus.training import pipeline as pipeline_mod
    from sportstradamus.training.pipeline import (
        DETERMINISTIC_FIXED_PARAMS,
        _calibration_penalty,
        _pick_calibrated_candidate,
    )
    from sportstradamus.training.scorecard import _gate4_pit_ks_threshold

    rng = np.random.default_rng(3)
    n = 1500
    mean_yr = rng.uniform(8.0, 25.0, size=n)
    signal = rng.normal(size=n)

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
        "hist_gate": 0.0,
        "normalize": True,
        "offset_mode": False,
        "shape_ceiling": None,
        "sn_param": "direct",
    }
    n_feat = X.shape[1]
    joint_fit_calls = []
    real_joint_fit = pipeline_mod.fit_skewnorm_dispersion_skew

    def _spy_joint_fit(*args, **kwargs):
        joint_fit_calls.append(1)
        return real_joint_fit(*args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "fit_skewnorm_dispersion_skew", _spy_joint_fit)
    served_pit_ks = _calibration_penalty(splits, dist_info)
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
    for cand in candidates:
        params = {k: v for k, v in cand.items() if k not in ("cv_loss", "pit_ks")}
        cand["pit_ks"] = served_pit_ks(params)
        assert 0.0 <= cand["pit_ks"] <= 1.0  # every candidate scored a finite served PIT-KS

    # C2: the SkewNormal closure fits (c, s) jointly per trial — the frame the gate scores —
    # not the retired c-only proxy.
    assert len(joint_fit_calls) == len(candidates)

    threshold = _gate4_pit_ks_threshold(len(splits["y_validation"]))
    winner = _pick_calibrated_candidate(candidates, threshold)
    assert winner in candidates
    assert winner["num_leaves"] in (8, 31, 63)
