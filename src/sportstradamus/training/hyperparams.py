"""LightGBMLSS hyperparameter search utilities."""

import numpy as np
import torch


class _BoundedResponseFn:
    """Picklable callable that clamps a response function's output."""

    def __init__(self, orig_fn, ceiling):
        self.orig_fn = orig_fn
        self.ceiling = float(ceiling)

    def __call__(self, predt):
        return torch.clamp(self.orig_fn(predt), max=self.ceiling)


def warm_start_hyper_opt(
    model,
    hp_dict,
    train_set,
    initial_params,
    num_boost_round=999,
    nfold=4,
    early_stopping_rounds=50,
    max_minutes=15,
    n_trials=100,
    silence=True,
):
    """Run a shortened hyper_opt seeded with previous best parameters."""
    import lightgbm as lgb
    import optuna
    from optuna.integration import LightGBMPruningCallback
    from optuna.samplers import TPESampler

    tunable_params = {k for k, v in hp_dict.items() if v[0] != "none"}

    def objective(trial):
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

        pruning_callback = LightGBMPruningCallback(trial, model.dist.loss_fn)
        early_stopping_callback = lgb.early_stopping(
            stopping_rounds=early_stopping_rounds, verbose=False
        )

        cv_result = model.cv(
            hyper_params,
            train_set,
            num_boost_round=num_boost_round,
            nfold=nfold,
            callbacks=[pruning_callback, early_stopping_callback],
            seed=None,
        )

        opt_rounds = np.argmin(np.array(cv_result[f"valid {model.dist.loss_fn}-mean"])) + 1
        trial.set_user_attr("opt_round", int(opt_rounds))
        return np.min(np.array(cv_result[f"valid {model.dist.loss_fn}-mean"]))

    if silence:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = TPESampler()
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        study_name="LightGBMLSS Warm-Start Optimization",
    )

    # Enqueue previous best params as the first trial
    seed_params = {k: v for k, v in initial_params.items() if k in tunable_params}
    seed_params["boosting"] = "gbdt"
    study.enqueue_trial(seed_params)

    study.optimize(objective, n_trials=n_trials, timeout=60 * max_minutes, show_progress_bar=True)

    print("\nWarm-Start Hyper-Parameter Optimization finished.")
    print("  Number of finished trials: ", len(study.trials))
    print("  Best trial:")
    opt_param = study.best_trial
    opt_param.params["opt_rounds"] = int(
        study.trials_dataframe()["user_attrs_opt_round"][study.trials_dataframe()["value"].idxmin()]
    )

    print(f"    Value: {opt_param.value}")
    print("    Params: ")
    for key, value in opt_param.params.items():
        print(f"    {key}: {value}")

    return opt_param.params
