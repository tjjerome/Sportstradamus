"""End-to-end coverage for the calibration-aware HP levers (research briefs).

Lever 1 (v3) has two halves that only exist together at train time:
  * `run_hyper_opt(calibration_penalty=…)` runs a raw-CV-loss objective with the trial's
    out-of-fold PIT-KS fed to the TPE sampler as a feasibility constraint, and returns every
    completed trial tagged with raw `cv_loss` + `pit_ks` (a real optuna+lightgbm cv search).
  * `_calibration_penalty` builds the per-trial served-PIT-KS evaluator over the cv fold
    boosters (`evaluate(params, *, cvbooster, folds, opt_rounds)` — no refit);
    `_pick_calibrated_candidate` makes the final 1-SE feasible pick.

Marked ``@pytest.mark.integration``: real lightgbm fits, offline, no network.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.integration
def test_run_hyper_opt_search_gate_returns_pit_ks_tagged_candidates():
    import lightgbm as lgb
    import pandas as pd
    from lightgbmlss.model import LightGBMLSS

    from sportstradamus.skew_normal import SkewNormal
    from sportstradamus.training.hyperparams import run_hyper_opt

    rng = np.random.default_rng(0)
    n = 600
    x0 = rng.normal(size=n)
    # A category-dtype column mirrors the production matrices (e.g. MLB "Player position"):
    # lightgbm auto-registers it as a named categorical feature, which breaks if the Dataset
    # is constructed before cv sees the trial params (the exp2 MLB arm-C crash).
    X = pd.DataFrame(
        {
            "x0": x0,
            "x1": rng.normal(size=n),
            "pos": pd.Categorical(rng.choice(["C", "1B", "OF"], size=n)),
        }
    )
    y = 5.0 + 2.0 * x0 + rng.normal(scale=1.0, size=n)
    dtrain = lgb.Dataset(X, label=y)

    model = LightGBMLSS(SkewNormal(stabilization="None", loss_fn="crps"))
    hp_dict = {
        "num_leaves": ["int", {"low": 8, "high": 31, "log": False}],
        "learning_rate": ["float", {"low": 0.05, "high": 0.2, "log": True}],
    }

    # Stand-in calibration gate: more leaves reads as worse-calibrated, so the constrained
    # sampler path is exercised without a real decode (that path is covered below). The
    # kwargs assert run_hyper_opt hands the evaluator the trial's own cv artifacts.
    def pit_ks(params, *, cvbooster, folds, opt_rounds):
        assert len(cvbooster.boosters) == len(folds) == 2
        assert sum(len(test_idx) for _, test_idx in folds) == n
        assert opt_rounds == params["opt_rounds"] >= 1
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
    import lightgbm as lgb
    import pandas as pd
    from lightgbmlss.model import LightGBMLSS

    from sportstradamus.helpers import set_model_start_values
    from sportstradamus.skew_normal import SkewNormal
    from sportstradamus.training import baselines
    from sportstradamus.training import pipeline as pipeline_mod
    from sportstradamus.training.hyperparams import _materialise_fold_datasets
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
    # Book EV for the train rows: a real quote near the outcome mean for most rows, with a
    # zero-EV band exercising the no-book fallback (model-only at weight 1) inside
    # _blend_oof_skewnormal.
    book_ev = mean_yr[:cut] * 1.02
    book_ev[:50] = 0.0
    splits = {
        "X_train": X.iloc[:cut],
        "y_train": pd.DataFrame({"Result": y_raw[:cut]}),
        "y_train_labels": y_norm[:cut],
        "B_train": pd.DataFrame({"Line": mean_yr[:cut], "Odds": 0.5, "EV": book_ev}),
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
        "cv": 0.4,
        "normalize": True,
        "offset_mode": False,
        "shape_ceiling": None,
        "sn_param": "direct",
    }
    n_feat = X.shape[1]
    sn_ks_probes = []
    real_sn_ks = pipeline_mod._served_sn_pit_ks

    def _spy_sn_ks(mean, sigma, skew, y, c, s, gate=None):
        sn_ks_probes.append((float(c), float(s)))
        return real_sn_ks(mean, sigma, skew, y, c, s, gate)

    monkeypatch.setattr(pipeline_mod, "_served_sn_pit_ks", _spy_sn_ks)

    blend_weights = []
    real_blend_fit = pipeline_mod.calibration.fit_blend_weight

    def _spy_blend_fit(*args, **kwargs):
        w = real_blend_fit(*args, **kwargs)
        blend_weights.append(float(np.asarray(w).ravel()[0]))
        return w

    monkeypatch.setattr(pipeline_mod.calibration, "fit_blend_weight", _spy_blend_fit)
    served_pit_ks = _calibration_penalty(splits, dist_info)

    # The trial artifacts the evaluator now consumes: a real cv per candidate with explicit
    # contiguous folds and return_cvbooster, exactly as run_hyper_opt builds them.
    model = LightGBMLSS(dist_info["dist_obj"])
    set_model_start_values(model, "SkewNormal", splits["X_train"], normalized=True)
    dtrain = lgb.Dataset(splits["X_train"], label=splits["y_train_labels"])
    dtrain.free_raw_data = False
    chunks = np.array_split(np.arange(cut), 2)
    folds = [(chunks[1], chunks[0]), (chunks[0], chunks[1])]

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
        cv_result = model.cv(
            dict(params),
            dtrain,
            num_boost_round=int(params["opt_rounds"]),
            folds=folds,
            fpreproc=_materialise_fold_datasets,
            return_cvbooster=True,
        )
        cand["pit_ks"] = served_pit_ks(
            params,
            cvbooster=cv_result["cvbooster"],
            folds=folds,
            opt_rounds=int(params["opt_rounds"]),
        )
        assert 0.0 <= cand["pit_ks"] <= 1.0  # every candidate scored a finite served PIT-KS

    # C2: the SkewNormal closure searches the skew dimension per trial — the frame the gate
    # scores — not the retired c-only (s = 0) proxy. The coarse sequential fit probes
    # interior s values by construction.
    assert any(s != 0.0 for _, s in sn_ks_probes)

    # R1-lite: every trial's measure blends with the book (exp2 MLB Goodhart fix) — one
    # fit_blend_weight per evaluator call, and the fitted weight is a real mixing fraction.
    assert len(blend_weights) == len(candidates)
    assert all(0.0 <= w <= 1.0 for w in blend_weights)

    threshold = _gate4_pit_ks_threshold(len(splits["y_validation"]))
    winner = _pick_calibrated_candidate(candidates, threshold, len(splits["X_train"]))
    assert winner in candidates
    assert winner["num_leaves"] in (8, 31, 63)
