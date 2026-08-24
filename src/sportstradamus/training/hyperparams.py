"""LightGBMLSS hyperparameter search utilities."""

import numpy as np
import torch

from sportstradamus.helpers import get_logger

logger = get_logger(__name__)


class _BoundedResponseFn:
    """Picklable callable that clamps a response function's output."""

    # Class-level default: instances pickled before the floor existed carry no
    # ``floor`` in their __dict__ and fall back here on unpickle.
    floor = None

    def __init__(self, orig_fn, ceiling, floor=None):
        self.orig_fn = orig_fn
        self.ceiling = float(ceiling)
        self.floor = None if floor is None else float(floor)

    def __call__(self, predt):
        return torch.clamp(self.orig_fn(predt), min=self.floor, max=self.ceiling)


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


def _clamp_seed_params(hp_dict: dict, initial_params: dict) -> dict:
    """Warm-start seed clamped into the current search space.

    A value stored in the pickle can fall outside today's bounds — notably
    ``lambda_l1``/``lambda_l2`` = 0.0 (LightGBM's default, below the ``1e-6`` log-scale
    floor) — which makes Optuna's ``enqueue_trial`` take ``log(0)`` and abort the whole
    study. Clamping each tunable seed into ``[low, high]`` honors the warm start instead.
    """
    seed = {}
    for name, (kind, spec) in hp_dict.items():
        if kind == "none" or name not in initial_params:
            continue
        seed[name] = min(max(initial_params[name], spec["low"]), spec["high"])
    return seed


def _materialise_fold_datasets(train_set, valid_set, params):
    """Rebuild cv's two folds as plain Datasets sharing the full frame's bin mappers.

    ``lgb.cv`` slices its folds with ``Dataset.subset``, and building a Booster over a subset
    corrupts the heap on wide frames: LightGBM 4.6 and 4.7 both abort inside ``LGBM_BoosterCreate``
    with a glibc heap message and no traceback, which cost the confirm walk both NFL carries
    nominees (docs/handoffs/dpo-confirm-crash.md). ``fpreproc`` is cv's own hook for replacing the
    fold data before that call, so its loop, early stopping and aggregation stay untouched, and
    referencing the full Dataset keeps the binning identical to the subset this replaces.
    """
    import lightgbm as lgb  # deferred for the same reason as run_hyper_opt's imports below

    full = train_set.reference
    data, label = full.get_data(), full.get_label()
    init_score = full.get_field("init_score").reshape(full.num_data(), -1, order="F")
    folds = []
    for fold in (train_set, valid_set):
        rows = np.asarray(fold.used_indices, dtype=np.int64)
        built = lgb.Dataset(
            data.take(rows, axis=0), label=label[rows], reference=full, free_raw_data=False
        )
        built.set_init_score(init_score[rows].ravel(order="F"))
        folds.append(built)
    return folds[0], folds[1], params


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
    objective is plain CV loss (CRPS, or nll for the count families) — unchanged. When
    ``calibration_penalty`` is given (Lever 1, calibrated HP selection), the objective stays the
    raw CV loss; each trial's served PIT-KS is measured out-of-fold —
    ``calibration_penalty(params, cvbooster=…, folds=…, opt_rounds=…)`` harvests the fold boosters
    ``model.cv`` already trained, so there is no per-trial refit — and reaches the TPE sampler as a
    feasibility constraint (``constraints_func``, Deb's rule: feasible trials model the "below"
    KDE first, infeasible ones rank by violation) instead of a scalarized hinge, so scarce
    feasibility can no longer collapse the search into a pure PIT-KS minimizer. The return is then
    every completed trial's param dict tagged with its raw ``cv_loss``
    and ``pit_ks``, sorted by ``cv_loss``, for :func:`_pick_calibrated_candidate`'s final pick.
    """
    # Deferred: lightgbm and optuna are heavy; keeping them out of the top-level
    # import keeps dashboard startup fast (training/ is not imported by the dashboard).
    import lightgbm as lgb
    import optuna
    from optuna.samplers import TPESampler

    # LightGBM drops the raw frame at construct time unless told otherwise, and
    # _materialise_fold_datasets reads it back off the Dataset to rebuild each fold. Set here
    # rather than at the caller so every entry point gets the working configuration.
    train_set.free_raw_data = False

    folds = None
    if calibration_penalty is not None:
        # Explicit contiguous folds (rows are date-ordered, so these are the same date blocks
        # cv's default produces) — lightgbm's kstep chunking silently drops the tail rows when
        # n % nfold != 0, and OOF pooling needs the exact held-out index sets. Row count comes
        # from the raw frame, NOT construct(): an early parameterless construct freezes the
        # Dataset so cv's later per-trial params (categorical_feature by name) force a re-init
        # against the converted array, where name lookup fails.
        chunks = np.array_split(np.arange(len(train_set.data)), nfold)
        folds = [(np.concatenate(chunks[:k] + chunks[k + 1 :]), chunks[k]) for k in range(nfold)]

    def objective(trial):
        hyper_params = _suggest_params(trial, hp_dict)

        early_stopping_callback = lgb.early_stopping(
            stopping_rounds=early_stopping_rounds, verbose=False
        )

        cv_result = model.cv(
            hyper_params,
            train_set,
            num_boost_round=num_boost_round,
            folds=folds,
            nfold=nfold,
            fpreproc=_materialise_fold_datasets,
            callbacks=[early_stopping_callback],
            seed=None,
            return_cvbooster=calibration_penalty is not None,
        )

        cv_losses = np.array(cv_result[f"valid {model.dist.loss_fn}-mean"])
        opt_rounds = int(np.argmin(cv_losses)) + 1
        trial.set_user_attr("opt_round", opt_rounds)
        crps = float(np.min(cv_losses))
        if calibration_penalty is not None:
            pit_ks = float(
                calibration_penalty(
                    {**hyper_params, "opt_rounds": opt_rounds},
                    cvbooster=cv_result["cvbooster"],
                    folds=folds,
                    opt_rounds=opt_rounds,
                )
            )
            trial.set_user_attr("cv_loss", crps)
            trial.set_user_attr("pit_ks", pit_ks)
        return crps

    if silence:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    if calibration_penalty is not None:
        # ``.get`` with inf: a trial that died before tagging pit_ks reads as infeasible,
        # not a crash. constraints_func is optuna-@experimental but 4-years-stable; its
        # wiring is pinned by tests/golden/test_calibration_levers.py.
        sampler = TPESampler(
            constraints_func=lambda t: (
                t.user_attrs.get("pit_ks", float("inf")) - penalty_threshold,
            )
        )
    else:
        sampler = TPESampler()

    study = optuna.create_study(
        sampler=sampler,
        direction="minimize",
        study_name="LightGBMLSS Hyper-Parameter Optimization",
    )

    if initial_params is not None:
        seed_params = _clamp_seed_params(hp_dict, initial_params)
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
