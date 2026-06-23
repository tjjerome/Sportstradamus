"""LightGBMLSS hyperparameter search utilities."""

import numpy as np
import torch

from sportstradamus.helpers import get_logger

logger = get_logger(__name__)


class _BoundedResponseFn:
    """Picklable callable that clamps a response function's output."""

    def __init__(self, orig_fn, ceiling):
        self.orig_fn = orig_fn
        self.ceiling = float(ceiling)

    def __call__(self, predt):
        return torch.clamp(self.orig_fn(predt), max=self.ceiling)


def _suggest_params(trial, hp_dict):
    hyper_params = {}
    for param_name, param_value in hp_dict.items():
        param_type = param_value[0]
        if param_type in ("categorical", "none"):
            hyper_params[param_name] = trial.suggest_categorical(param_name, param_value[1])
        elif param_type == "float":
            c = param_value[1]
            hyper_params[param_name] = trial.suggest_float(
                param_name, low=c["low"], high=c["high"], log=c["log"]
            )
        elif param_type == "int":
            c = param_value[1]
            hyper_params[param_name] = trial.suggest_int(
                param_name, low=c["low"], high=c["high"], log=c["log"]
            )
    if "boosting" not in hyper_params:
        hyper_params["boosting"] = trial.suggest_categorical("boosting", ["gbdt"])
    return hyper_params


# Lever 1 search-gate weight. The hinge is one-sided (zero once a trial clears the Gate-4 PIT-KS
# threshold), so a large weight only discourages the under-dispersed corner without distorting the
# feasible region — which stays ranked by pure CRPS (no pull toward the over-dispersed marginal
# predictor). 10x dominates the inter-trial CRPS spread (~0.03) at a typical gate excess (~0.05),
# so an uncalibrated-but-sharp trial loses to a calibrated one.
_CALIBRATION_PENALTY_WEIGHT = 10.0


def _penalized_objective(crps: float, pit_ks: float, threshold: float) -> float:
    """Search-gating HPO objective (Lever 1): CRPS plus a one-sided hinge on the validation PIT-KS.

    Zero hinge for calibrated trials (``pit_ks <= threshold``) so the feasible region is ranked by
    pure CRPS; infeasible trials are pushed up by their Gate-4 excess, steering the TPE sampler
    toward the wider-sigma calibrated region rather than only re-ranking the sharpest cluster.
    """
    return crps + _CALIBRATION_PENALTY_WEIGHT * max(0.0, pit_ks - threshold)


def _collect_calibrated_candidates(trials):
    """Every completed Lever-1 trial as a param dict tagged with its raw ``cv_loss`` and ``pit_ks``,
    sorted by CRPS — the candidate set :func:`_pick_calibrated_candidate` picks the final HP from.
    """
    completed = sorted(
        (t for t in trials if "pit_ks" in t.user_attrs),
        key=lambda t: t.user_attrs["cv_loss"],
    )
    candidates = []
    for trial in completed:
        params = dict(trial.params)
        params["opt_rounds"] = int(trial.user_attrs["opt_round"])
        params["cv_loss"] = float(trial.user_attrs["cv_loss"])
        params["pit_ks"] = float(trial.user_attrs["pit_ks"])
        candidates.append(params)
    return candidates


def run_hyper_opt(
    model,
    hp_dict,
    train_set,
    initial_params=None,
    num_boost_round=999,
    nfold=4,
    early_stopping_rounds=50,
    max_minutes=15,
    n_trials=100,
    silence=True,
    calibration_penalty=None,
    penalty_threshold=None,
):
    """Run Optuna HPO for LightGBMLSS.

    optuna 3.5's ``LightGBMPruningCallback`` hardcodes the cv validation name
    to "cv_agg" while lightgbm >=4.6 reports cv results under "valid", so that
    callback can never match and raises. Running with only early stopping
    sidesteps it; each trial stays bounded and, on a warm start, the seeded
    params are evaluated first, so the selected hyperparameters are unaffected.

    Default (``calibration_penalty is None``) returns the single best-CV-loss param dict and the
    objective is plain CV-CRPS — unchanged. When ``calibration_penalty`` is given (Lever 1, the
    SkewNormal calibrated path), the objective becomes the search-gating score
    ``CRPS + weight*max(0, pit_ks - penalty_threshold)`` — ``calibration_penalty(params)`` returns
    the trial's served validation PIT-KS — so the TPE sampler explores the wider-sigma region. The
    return is then every completed trial's param dict tagged with its raw ``cv_loss``
    and ``pit_ks``, sorted by ``cv_loss``, for :func:`_pick_calibrated_candidate`'s final pick.
    """
    # Deferred: lightgbm and optuna are heavy; keeping them out of the top-level
    # import keeps dashboard startup fast (training/ is not imported by the dashboard).
    import lightgbm as lgb
    import optuna
    from optuna.samplers import TPESampler

    def objective(trial):
        hyper_params = _suggest_params(trial, hp_dict)

        early_stopping_callback = lgb.early_stopping(
            stopping_rounds=early_stopping_rounds, verbose=False
        )

        cv_result = model.cv(
            hyper_params,
            train_set,
            num_boost_round=num_boost_round,
            nfold=nfold,
            callbacks=[early_stopping_callback],
            seed=None,
        )

        cv_losses = np.array(cv_result[f"valid {model.dist.loss_fn}-mean"])
        opt_rounds = int(np.argmin(cv_losses)) + 1
        trial.set_user_attr("opt_round", opt_rounds)
        crps = float(np.min(cv_losses))
        if calibration_penalty is None:
            return crps
        pit_ks = float(calibration_penalty({**hyper_params, "opt_rounds": opt_rounds}))
        trial.set_user_attr("cv_loss", crps)
        trial.set_user_attr("pit_ks", pit_ks)
        return _penalized_objective(crps, pit_ks, penalty_threshold)

    if silence:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        sampler=TPESampler(),
        direction="minimize",
        study_name="LightGBMLSS Hyper-Parameter Optimization",
    )

    if initial_params is not None:
        tunable_params = {k for k, v in hp_dict.items() if v[0] != "none"}
        seed_params = {k: v for k, v in initial_params.items() if k in tunable_params}
        seed_params["boosting"] = "gbdt"
        study.enqueue_trial(seed_params)

    study.optimize(objective, n_trials=n_trials, timeout=60 * max_minutes, show_progress_bar=True)

    logger.info("Hyper-Parameter Optimization finished.")
    logger.info("  Number of finished trials: %d", len(study.trials))

    if calibration_penalty is not None:
        return _collect_calibrated_candidates(study.trials)

    logger.info("  Best trial:")
    opt_param = study.best_trial
    opt_param.params["opt_rounds"] = int(
        study.trials_dataframe()["user_attrs_opt_round"][study.trials_dataframe()["value"].idxmin()]
    )

    logger.info("    Value: %s", opt_param.value)
    logger.info("    Params:")
    for key, value in opt_param.params.items():
        logger.info("    %s: %s", key, value)

    return opt_param.params
