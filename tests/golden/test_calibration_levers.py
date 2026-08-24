"""Calibration-aware HP levers (research briefs /tmp/researcher_calibration_hp.md,
/tmp/researcher_lever1_v3.md).

Lever 2 — count-branch dispersion calibration re-targeted from CRPS to the Gate-4
randomized-PIT KS (`_dispersion_pit_ks_loss`).
Lever 1 (v3) — calibration-constrained HP *search*: the Optuna objective is the raw CV loss
and the trial's out-of-fold PIT-KS reaches the TPE sampler as a feasibility constraint
(`TPESampler(constraints_func=…)`, Deb rule); the OOF param frame is harvested from the fold
boosters `model.cv` already trained (`predict_cv_oof_params`); the final pick is
`_pick_calibrated_candidate` (lowest-CRPS trial within the one-standard-error effective
threshold `max(threshold, min_pit_ks + 0.2606/sqrt(n))`).

Both are opt-in; production defaults (CRPS dispersion, loss-only selection) are unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_dispersion_pit_ks_loss_negbin_minimized_at_calibrating_scale():
    """The count PIT-KS loss is lower at the ``c`` that restores the true NegBin
    shape than at the over-dispersed blended shape."""
    from sportstradamus.training.pipeline import _dispersion_pit_ks_loss

    rng = np.random.default_rng(0)
    n, r_true, mu = 6000, 12.0, 5.0
    p_success = r_true / (r_true + mu)
    y = rng.negative_binomial(r_true, p_success, size=n).astype(float)

    r_blend = np.full(n, r_true / 2.0)  # half the true shape → too over-dispersed
    mean = np.full(n, mu)

    def loss(c):
        return _dispersion_pit_ks_loss(
            c,
            dist="NegBin",
            y_val_arr=y,
            val_weighted_mean=mean,
            gate_blend_val=None,
            r_blend_val=r_blend,
        )

    assert 0.0 <= loss(2.0) <= 1.5
    assert loss(2.0) < loss(1.0)


def test_dispersion_pit_ks_loss_gamma_holds_mean_fixed():
    """Gamma branch scales the shape while holding the mean (EV) fixed, so a
    mis-shaped blended alpha is corrected toward the calibrating ``c``."""
    from sportstradamus.training.pipeline import _dispersion_pit_ks_loss

    rng = np.random.default_rng(1)
    n, a_true, mu = 5000, 8.0, 4.0
    y = rng.gamma(shape=a_true, scale=mu / a_true, size=n)

    a_blend = np.full(n, a_true / 2.0)  # too few shape → too wide; c=2 restores it
    mean = np.full(n, mu)

    def loss(c):
        return _dispersion_pit_ks_loss(
            c,
            dist="Gamma",
            y_val_arr=y,
            val_weighted_mean=mean,
            gate_blend_val=None,
            alpha_blend_val=a_blend,
        )

    assert loss(2.0) < loss(1.0)


def test_pick_calibrated_candidate_prefers_lowest_loss_within_threshold():
    """Lever 1 policy, feasible region unchanged: among trials clearing the raw PIT-KS
    threshold, the lowest CRPS wins, and a sharp-but-miscalibrated trial never does —
    the 1-SE widening can only extend the threshold (tau_eff >= threshold), not shrink it."""
    from sportstradamus.training.pipeline import _pick_calibrated_candidate

    # SE = 0.2606/sqrt(10000) ≈ 0.0026, so tau_eff = max(0.05, 0.03 + 0.0026) = 0.05.
    candidates = [
        {"name": "sharp_miscalibrated", "cv_loss": 1.0, "pit_ks": 0.20},
        {"name": "calibrated_best", "cv_loss": 1.2, "pit_ks": 0.03},
        {"name": "calibrated_worse", "cv_loss": 1.5, "pit_ks": 0.04},
    ]
    picked = _pick_calibrated_candidate(candidates, threshold=0.05, n_rows=10000)
    assert picked["name"] == "calibrated_best"


def test_pick_calibrated_candidate_logs_headroom(caplog):
    """Experiment-B probe: one INFO line reports the feasible set the epsilon-constraint
    saw — min PIT-KS, feasible count at the raw threshold, the 1-SE effective threshold,
    and the CRPS price of the pick over the loss-only best. Diagnostic only."""
    import logging

    from sportstradamus.training.pipeline import _pick_calibrated_candidate

    # _pick_calibrated_candidate's logger sets propagate=False (helpers.get_logger),
    # so attach caplog's handler directly to it by its real name.
    target = logging.getLogger("sportstradamus.cli.sportstradamus.training.pipeline")
    target.addHandler(caplog.handler)
    try:
        _pick_calibrated_candidate(
            [
                {"cv_loss": 1.0, "pit_ks": 0.20},
                {"cv_loss": 1.2, "pit_ks": 0.03},
                {"cv_loss": 1.5, "pit_ks": 0.04},
            ],
            threshold=0.05,
            n_rows=10000,
        )
        _pick_calibrated_candidate(
            [
                {"cv_loss": 1.0, "pit_ks": 0.20},
                {"cv_loss": 1.4, "pit_ks": 0.11},
            ],
            threshold=0.05,
            n_rows=10000,
        )
    finally:
        target.removeHandler(caplog.handler)

    headroom = [r.getMessage() for r in caplog.records if "headroom" in r.getMessage()]
    assert len(headroom) == 2
    assert "min_pit_ks=0.0300" in headroom[0]
    assert "feasible=2/3" in headroom[0]
    assert "threshold=0.0500" in headroom[0]
    assert "tau_eff=0.0500" in headroom[0]  # min + SE (0.0326) below the raw threshold
    assert "crps_price=0.20000" in headroom[0]  # cv_loss(picked 1.2) - cv_loss(best 1.0)
    assert "min_pit_ks=0.1100" in headroom[1]
    assert "feasible=0/2" in headroom[1]
    assert "tau_eff=0.1126" in headroom[1]  # 0.11 + 0.2606/sqrt(10000)
    assert "crps_price=0.40000" in headroom[1]  # the 1-SE pick still reports its price


def test_pick_calibrated_candidate_infeasible_uses_one_se_rule():
    """When no trial clears the raw threshold, the pick is the lowest-CRPS trial within one
    KS sampling SE of the best-calibrated trial — never the bare min-pit_ks trial, the
    single most winner's-cursed of ~300 noisy KS draws (Lever-1-v3 brief, R3)."""
    from sportstradamus.training.pipeline import _pick_calibrated_candidate

    # SE = 0.2606/sqrt(970) ≈ 0.0084: 0.081 sits inside min 0.080 + SE, 0.095 outside.
    candidates = [
        {"name": "sharp_far", "cv_loss": 1.0, "pit_ks": 0.095},
        {"name": "min_ks", "cv_loss": 1.5, "pit_ks": 0.080},
        {"name": "tied_sharper", "cv_loss": 1.2, "pit_ks": 0.081},
    ]
    picked = _pick_calibrated_candidate(candidates, threshold=0.05, n_rows=970)
    assert picked["name"] == "tied_sharper"


def test_stabilization_arg_reaches_distribution_attribute():
    """Lever 4 guard: the --stabilization value must land on a real LightGBMLSS attribute,
    so a renamed constructor arg can't make the flag a silent no-op."""
    from sportstradamus.skew_normal import SkewNormal

    assert SkewNormal(stabilization="MAD", loss_fn="crps").stabilization == "MAD"
    assert SkewNormal(stabilization="None", loss_fn="crps").stabilization == "None"


def test_tpe_constraints_func_stores_pit_ks_violation_in_system_attrs():
    """Pin the optuna experimental ``constraints_func`` wiring (Lever-1-v3 brief, R3): with the
    same constraint shape ``run_hyper_opt`` builds, the study must store each completed trial's
    constraint tuple — pit_ks minus the threshold — under ``trial.system_attrs["constraints"]``
    (``optuna.samplers._base._CONSTRAINTS_KEY``, what TPESampler's Deb feasibility split reads).
    Breaks loudly if an optuna bump changes the experimental API."""
    import optuna
    from optuna.samplers import TPESampler

    tau = 0.05
    pit_by_trial = [0.02, 0.09, 0.05, 0.30]

    def objective(trial):
        trial.set_user_attr("pit_ks", pit_by_trial[trial.number])
        return trial.suggest_float("x", 0.0, 1.0)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        sampler=TPESampler(
            seed=0,
            constraints_func=lambda t: (t.user_attrs.get("pit_ks", float("inf")) - tau,),
        ),
        direction="minimize",
    )
    study.optimize(objective, n_trials=len(pit_by_trial))

    for trial in study.trials:
        stored = trial.system_attrs["constraints"]
        assert list(stored) == [pytest.approx(pit_by_trial[trial.number] - tau)]


def test_predict_cv_oof_params_matches_predict_lss_params_layout():
    """Lever-1-v3 R2 pin: the OOF param frame harvested from ``model.cv``'s fold boosters must
    carry the exact column layout and dtype ``predict_lss_params`` produces — that layout is
    what the calibrated closure's decode consumes — and cover every pooled row exactly once."""
    import lightgbm as lgb
    import pandas as pd
    from lightgbmlss.model import LightGBMLSS

    from sportstradamus.helpers import set_model_start_values
    from sportstradamus.skew_normal import SkewNormal
    from sportstradamus.training.hyperparams import _materialise_fold_datasets
    from sportstradamus.training.pipeline import predict_cv_oof_params, predict_lss_params

    rng = np.random.default_rng(0)
    n = 300
    mean_yr = rng.uniform(8.0, 25.0, size=n)
    X = pd.DataFrame(
        {
            "MeanYr": mean_yr,
            "STDYr": 0.4 * mean_yr,
            "ZeroYr": np.zeros(n),
            "Signal": rng.normal(size=n),
        }
    )
    y = np.clip(1.0 + 0.15 * X["Signal"].to_numpy() + rng.normal(scale=0.25, size=n), 0.2, None)

    model = LightGBMLSS(SkewNormal(stabilization="None", loss_fn="crps"))
    set_model_start_values(model, "SkewNormal", X, normalized=True)
    dtrain = lgb.Dataset(X, label=y)
    dtrain.free_raw_data = False

    params = {"num_leaves": 8, "learning_rate": 0.1, "min_child_samples": 20, "verbose": -1}
    chunks = np.array_split(np.arange(n), 2)
    folds = [(chunks[1], chunks[0]), (chunks[0], chunks[1])]
    cv_result = model.cv(
        dict(params),
        dtrain,
        num_boost_round=10,
        folds=folds,
        fpreproc=_materialise_fold_datasets,
        return_cvbooster=True,
    )
    oof = predict_cv_oof_params(
        cv_result["cvbooster"], folds, 10, X, model.start_values, model.dist
    )

    model.train(dict(params), dtrain, num_boost_round=10)
    ref = predict_lss_params(model, "SkewNormal", X, normalized=True)

    assert list(oof.columns) == list(ref.columns) == ["loc", "scale", "alpha"]
    assert list(oof.dtypes) == list(ref.dtypes)
    assert sorted(oof.index) == list(X.index)
    assert np.isfinite(oof.to_numpy()).all()
    assert (oof["scale"].to_numpy() > 0).all()


def test_predict_cv_oof_params_reemits_direct_columns_for_centered_sn():
    """Centered-parametrization pin (the exp2 NBA-PTS arm-C crash): the OOF frame from a
    ``CenteredSkewNormal`` model must re-emit the direct ``(loc, scale, alpha)`` columns the
    calibrated closure decodes — mirroring ``CenteredSkewNormal.predict_dist`` — not the raw
    ``(mean, sd, gamma1)`` heads."""
    import lightgbm as lgb
    import pandas as pd
    from lightgbmlss.model import LightGBMLSS

    from sportstradamus.helpers import set_model_start_values
    from sportstradamus.skew_normal_centered import CenteredSkewNormal
    from sportstradamus.training.hyperparams import _materialise_fold_datasets
    from sportstradamus.training.pipeline import predict_cv_oof_params

    rng = np.random.default_rng(1)
    n = 300
    mean_yr = rng.uniform(8.0, 25.0, size=n)
    X = pd.DataFrame(
        {
            "MeanYr": mean_yr,
            "STDYr": 0.4 * mean_yr,
            "ZeroYr": np.zeros(n),
            "Signal": rng.normal(size=n),
        }
    )
    y = np.clip(1.0 + 0.15 * X["Signal"].to_numpy() + rng.normal(scale=0.25, size=n), 0.2, None)

    model = LightGBMLSS(CenteredSkewNormal(loss_fn="nll"))
    set_model_start_values(model, "SkewNormal", X, normalized=True, sn_param="centered")
    dtrain = lgb.Dataset(X, label=y)
    dtrain.free_raw_data = False

    params = {"num_leaves": 8, "learning_rate": 0.1, "min_child_samples": 20, "verbose": -1}
    chunks = np.array_split(np.arange(n), 2)
    folds = [(chunks[1], chunks[0]), (chunks[0], chunks[1])]
    cv_result = model.cv(
        dict(params),
        dtrain,
        num_boost_round=10,
        folds=folds,
        fpreproc=_materialise_fold_datasets,
        return_cvbooster=True,
    )
    oof = predict_cv_oof_params(
        cv_result["cvbooster"], folds, 10, X, model.start_values, model.dist
    )

    assert list(oof.columns) == ["loc", "scale", "alpha"]
    assert sorted(oof.index) == list(X.index)
    assert np.isfinite(oof.to_numpy()).all()
    assert (oof["scale"].to_numpy() > 0).all()


def test_clamp_seed_params_rescues_sub_floor_log_param():
    """Warm-start regression: a pickle storing lambda_l1=0.0 (LightGBM's default, below the
    1e-6 log-scale floor) must be clamped up to the floor, not enqueued as-is — seeding 0.0 into
    a log=True distribution makes Optuna take log(0) and abort the whole study. In-bounds seeds
    pass through unchanged and 'none'-type (fixed) params are not seeded."""
    from sportstradamus.training.hyperparams import _clamp_seed_params

    hp_dict = {
        "num_threads": ["none", [8]],
        "lambda_l1": ["float", {"low": 1e-6, "high": 10, "log": True}],
        "num_leaves": ["int", {"low": 8, "high": 127, "log": False}],
    }
    seed = _clamp_seed_params(hp_dict, {"lambda_l1": 0.0, "num_leaves": 31, "num_threads": 8})

    assert seed["lambda_l1"] == 1e-6  # sub-floor log param clamped up — no log(0) abort
    assert seed["num_leaves"] == 31  # in-bounds seed unchanged
    assert "num_threads" not in seed  # 'none'-type fixed params are not seeded


def test_resolve_cell_knob_persists_per_cell_selection_and_honors_flag_override():
    """Durability: a cell's stat_meta training knob (hpo_selection, blending,
    count_dispersion_objective, zinb_mode) is what the production cron applies under the default 'auto' flag,
    so a tuned ship reproduces on retrain instead of darking out; an explicit flag forces every
    cell for a one-shot A/B."""
    from sportstradamus.training.cli import LOSS_AUTO, _resolve_cell_knob

    sm = {
        "WNBA": {
            "PR": {"hpo_selection": "calibrated"},
            "OREB": {"count_dispersion_objective": "pit_ks"},
            "FTM": {"zinb_mode": "hurdle"},
        }
    }
    assert _resolve_cell_knob(sm, "WNBA", "PR", "hpo_selection", "loss", LOSS_AUTO) == "calibrated"
    assert _resolve_cell_knob(sm, "WNBA", "PA", "hpo_selection", "loss", LOSS_AUTO) == "loss"
    assert _resolve_cell_knob(sm, "WNBA", "PR", "hpo_selection", "loss", "loss") == "loss"
    # count_dispersion_objective persists identically, so OREB's pit_ks ship reproduces on the cron
    assert (
        _resolve_cell_knob(sm, "WNBA", "OREB", "count_dispersion_objective", "crps", LOSS_AUTO)
        == "pit_ks"
    )
    assert (
        _resolve_cell_knob(sm, "WNBA", "BLST", "count_dispersion_objective", "crps", LOSS_AUTO)
        == "crps"
    )
    assert (
        _resolve_cell_knob(sm, "WNBA", "OREB", "count_dispersion_objective", "crps", "crps")
        == "crps"
    )
    # zinb_mode persists so a hurdle ship reproduces on the cron instead of darking back to joint
    assert _resolve_cell_knob(sm, "WNBA", "FTM", "zinb_mode", "joint", LOSS_AUTO) == "hurdle"
    assert _resolve_cell_knob(sm, "WNBA", "PA", "zinb_mode", "joint", LOSS_AUTO) == "joint"
    assert _resolve_cell_knob(sm, "WNBA", "FTM", "zinb_mode", "joint", "joint") == "joint"
