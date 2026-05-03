"""Per-market training pipeline: data loading, model fitting, calibration, diagnostics."""

import importlib.resources as pkg_resources
import json
import os
import pickle
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbmlss.distributions.NegativeBinomial import NegativeBinomial
from lightgbmlss.distributions.ZINB import ZINB
from lightgbmlss.model import LightGBMLSS
from scipy.optimize import minimize_scalar
from scipy.special import beta as beta_fn
from scipy.special import expit, logit
from scipy.stats import gamma, nbinom, norm, skewnorm
from sklearn.metrics import accuracy_score, log_loss, precision_score
from sklearn.model_selection import train_test_split

from sportstradamus import data
from sportstradamus.helpers import (
    fused_loc,
    get_ev,
    get_odds,
    set_model_start_values,
    stat_cv,
    stat_zi,
)
from sportstradamus.skew_normal import SkewNormal as SkewNormalDist
from sportstradamus.training.calibration import fit_book_weights, fit_model_weight
from sportstradamus.training.config import load_distribution_config, save_zi_config
from sportstradamus.training.data import trim_matrix
from sportstradamus.training.hyperparams import _BoundedResponseFn, warm_start_hyper_opt
from sportstradamus.training.report import report
from sportstradamus.training.shap import _scouting_shap_and_filter


def train_market(
    league: str,
    market: str,
    stat_data,
    force: bool,
    rebuild_filter: bool,
    archive,
    league_start_date,
) -> None:
    """Train or retrain one LightGBMLSS model for a single league/market pair.

    Loads the training matrix, selects the distribution, runs Optuna hyperparameter
    search, fits the model, applies dispersion calibration and temperature scaling,
    evaluates on the held-out test set, and saves the model pickle + training report.
    """
    stat_dist = load_distribution_config()
    stat_dist.setdefault(league, {})
    stat_zi.setdefault(league, {})

    if os.path.isfile(pkg_resources.files(data) / "book_weights.json"):
        with open(pkg_resources.files(data) / "book_weights.json") as infile:
            book_weights = json.load(infile)
    else:
        book_weights = {}

    book_weights.setdefault(league, {}).setdefault(market, {})
    book_weights[league][market] = fit_book_weights(league, market, stat_data, archive, book_weights)

    with open(pkg_resources.files(data) / "book_weights.json", "w") as outfile:
        json.dump(book_weights, outfile, indent=4)

    filename = "_".join([league, market]).replace(" ", "-")
    filepath = pkg_resources.files(data) / f"models/{filename}.mdl"
    need_model = True
    if os.path.isfile(filepath):
        with open(filepath, "rb") as infile:
            filedict = pickle.load(infile)
            model = filedict["model"]
            filedict["params"]
            dist = filedict["distribution"]
            cv = filedict["cv"]
            step = filedict["step"]

        need_model = False
    else:
        filedict = {}
        dist = None

    print(f"Training {league} - {market}")
    cv = stat_cv[league].get(market, 1)
    filepath = pkg_resources.files(data) / (f"training_data/{filename}.csv")

    if os.path.isfile(filepath):
        M = pd.read_csv(filepath, index_col=0)
        cutoff_date = pd.to_datetime(M["Date"]).max().date()
        M = M.loc[
            (pd.to_datetime(M.Date).dt.date <= cutoff_date)
            & (pd.to_datetime(M.Date).dt.date > league_start_date)
        ]
    else:
        cutoff_date = league_start_date
        M = pd.DataFrame()

    new_M = stat_data.get_training_matrix(market, cutoff_date)

    if new_M.empty and not force and not need_model:
        return

    M = pd.concat([M, new_M], ignore_index=True)
    if M.empty:
        print(f"  No usable training data for {league} {market}, skipping")
        return
    M.Date = pd.to_datetime(M.Date, format="mixed")
    if "Player" in M.columns:
        M = M.drop_duplicates(subset=["Player", "Date"], keep="last")
    step = M["Result"].drop_duplicates().sort_values().diff().min()
    _prep_gate = (
        stat_zi.get(league, {}).get(market, 0) if dist in ("ZINB", "ZAGamma") else 0
    )
    synthetic_mask = M.Odds.isna() | (M.Odds == 0)
    if "Odds_synthetic" in M.columns:
        synthetic_mask |= M["Odds_synthetic"].fillna(False)
    for i, row in M.loc[synthetic_mask].iterrows():
        if np.isnan(row["EV"]) or row["EV"] <= 0:
            M.loc[i, "Odds"] = 0.5
            M.loc[i, "EV"] = get_ev(
                M.loc[i, "Line"], 0.5, cv=cv, dist=dist, gate=_prep_gate or None
            )
            M.loc[i, "Odds_synthetic"] = True
        else:
            M.loc[i, "Odds"] = 1 - get_odds(
                row["Line"], row["EV"], dist, cv=cv, step=step, gate=_prep_gate or None
            )
            M.loc[i, "Odds_synthetic"] = False

    M = trim_matrix(M, 15000)
    M.to_csv(filepath)

    stat_data.save_comps()

    if rebuild_filter:
        print("  Scouting pass for filter rebuild...")
        diag = _scouting_shap_and_filter(league, market, M, stat_data)
        if diag is not None:
            print(
                f"  Filter: kept={diag['n_kept']} dropped={diag['n_dropped']} added={diag['n_added']}"
            )

    y = M[["Result"]]
    X = M[stat_data.get_stat_columns(market)]

    categories = ["Home", "Player position"]
    if "Player position" not in X.columns:
        categories.remove("Player position")
    for c in categories:
        X[c] = X[c].astype("category")

    categories = "name:" + ",".join(categories)

    # Temporal split: earliest 70% to train, latest 30% to test
    M_sorted = M.sort_values("Date")
    n = len(M_sorted)
    n_train = int(n * 0.7)
    train_idx = M_sorted.index[:n_train]
    test_idx = M_sorted.index[n_train:]

    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]

    X_test, X_validation, y_test, y_validation = train_test_split(
        X_test, y_test, test_size=0.5, random_state=25
    )

    B_train = M.loc[X_train.index, ["Line", "Odds", "EV"]]
    B_test = M.loc[X_test.index, ["Line", "Odds", "EV"]]
    B_validation = M.loc[X_validation.index, ["Line", "Odds", "EV"]]

    y_train_labels = np.ravel(y_train.to_numpy())

    # Distribution selection: mean < 2 → NegBin, mean >= 2 → SkewNormal
    threshold_dict = {"NBA": 60, "NFL": 10, "NHL": 60, "WNBA": 40, "MLB": 60}
    threshold = threshold_dict.get(league, 60)
    player_stats = (
        stat_data.gamelog.groupby(stat_data.log_strings.get("player"))
        .filter(lambda x: x[market].gt(0).sum() > threshold)
        .groupby(stat_data.log_strings.get("player"))[market]
    )

    zero_mask = y_train_labels == 0
    hist_gate = zero_mask.sum() / len(y_train_labels) if len(y_train_labels) > 0 else 0

    stat_zi[league][market] = hist_gate
    save_zi_config(stat_zi)

    player_stats = player_stats.apply(lambda x: x[x != 0]).groupby(level=0)

    global_mean = y_train_labels.mean()
    normalize = False
    denom_col = "MeanYr"

    if global_mean >= 2.0:
        dist = "SkewNormal"
        normalize = True
        dist_obj = SkewNormalDist(stabilization="None", loss_fn="crps")

        cv = (
            player_stats.std()
            / player_stats.mean()
            * player_stats.count()
            / player_stats.count().sum()
        ).sum()
        cv = max(cv, 0.05)
        shape_ceiling = None
        marginal_shape = None

        if hist_gate > 0.05:
            nonzero_mask = y_train_labels > 0
            X_train = X_train[nonzero_mask]
            y_train_labels = y_train_labels[nonzero_mask]
            denom_col = (
                "MeanYr_nonzero" if "MeanYr_nonzero" in X_train.columns else "MeanYr"
            )
        else:
            denom_col = "MeanYr"

        meanyr_train = X_train[denom_col].clip(lower=0.5).to_numpy()
        y_train_labels = y_train_labels / meanyr_train
        y_train_labels = np.clip(y_train_labels, 0.01, None)
    else:
        dist = "NegBin"
        if hist_gate > 0.02:
            dist = "ZINB"
        if dist == "NegBin":
            dist_obj = NegativeBinomial(stabilization="None", loss_fn="nll")
        else:
            dist_obj = ZINB(stabilization="None", loss_fn="nll")

        R_CAP = 50
        per_player_r = player_stats.mean() ** 2 / np.maximum(
            player_stats.var() - player_stats.mean(), 0.01
        )
        per_player_r = np.minimum(per_player_r, R_CAP)

        marginal_shape = max(float(np.quantile(per_player_r, 0.95)), 0.5)
        K_SHAPE = 2.0
        shape_ceiling = marginal_shape * K_SHAPE

        cv = (1 / per_player_r * player_stats.count() / player_stats.count().sum()).sum()
        cv = max(cv, 1 / shape_ceiling)

        cv = (
            player_stats.std()
            / player_stats.mean()
            * player_stats.count()
            / player_stats.count().sum()
        ).sum()
        cv = max(cv, 1 / np.sqrt(shape_ceiling))

    stat_cv[league][market] = cv
    with open(pkg_resources.files(data) / "stat_cv.json", "w") as f:
        json.dump(stat_cv, f, indent=4)

    # Under --rebuild-filter warm-starting from old pickle hyperparams is invalid
    opt_params = None if rebuild_filter else filedict.get("params")
    dtrain = lgb.Dataset(X_train, label=y_train_labels)

    # Bound shape response function (safety net)
    if dist in ("NegBin", "ZINB"):
        dist_obj.param_dict["total_count"] = _BoundedResponseFn(
            dist_obj.param_dict["total_count"], shape_ceiling
        )
    elif dist in ("Gamma", "ZAGamma"):
        dist_obj.param_dict["concentration"] = _BoundedResponseFn(
            dist_obj.param_dict["concentration"], shape_ceiling
        )

    model = LightGBMLSS(dist_obj)
    set_model_start_values(
        model, dist, X_train, shape_ceiling=shape_ceiling, normalized=normalize
    )

    col_list = list(X_train.columns)
    monotone = [0] * len(col_list)
    if dist in ("Gamma", "ZAGamma", "SkewNormal") and "MeanYr" in col_list:
        monotone[col_list.index("MeanYr")] = 1

    hp_search_space = {
        "feature_pre_filter": ["none", [False]],
        "num_threads": ["none", [8]],
        "max_depth": ["none", [-1]],
        "max_bin": ["none", [127]],
        "hist_pool_size": ["none", [9 * 1024]],
        "monotone_constraints": ["none", [monotone]],
        "path_smooth": ["float", {"low": 0, "high": 20, "log": False}],
        "num_leaves": ["int", {"low": 8, "high": 127, "log": False}],
        "lambda_l1": ["float", {"low": 1e-6, "high": 10, "log": True}],
        "lambda_l2": ["float", {"low": 1e-6, "high": 10, "log": True}],
        "min_child_samples": ["int", {"low": 30, "high": 150, "log": False}],
        "min_child_weight": [
            "float",
            {"low": 1e-3, "high": 0.75 * len(X_train) / 1000, "log": True},
        ],
        "learning_rate": ["float", {"low": 0.01, "high": 0.15, "log": True}],
        "feature_fraction": ["float", {"low": 0.4, "high": 1.0, "log": False}],
        "bagging_fraction": ["float", {"low": 0.4, "high": 1.0, "log": False}],
        "bagging_freq": ["none", [1]],
    }

    if opt_params is None or opt_params.get("opt_rounds") is None:
        opt_params = model.hyper_opt(
            hp_search_space,
            dtrain,
            num_boost_round=999,
            nfold=4,
            early_stopping_rounds=50,
            max_minutes=60,
            n_trials=300,
            silence=True,
        )
    else:
        opt_params = warm_start_hyper_opt(
            model, hp_search_space, dtrain, opt_params, n_trials=150, max_minutes=5
        )

    model.train(opt_params, dtrain, num_boost_round=opt_params["opt_rounds"])

    # Predictions and parameter extraction
    prob_params_train = pd.DataFrame()
    prob_params_validation = pd.DataFrame()
    prob_params = pd.DataFrame()

    idx = X_train.index
    set_model_start_values(model, dist, X_train, normalized=normalize)
    preds = model.predict(X_train, pred_type="parameters")
    preds.index = idx
    prob_params_train = pd.concat([prob_params_train, preds])

    idx = X_validation.index
    set_model_start_values(model, dist, X_validation, normalized=normalize)
    preds = model.predict(X_validation, pred_type="parameters")
    preds.index = idx
    prob_params_validation = pd.concat([prob_params_validation, preds])

    idx = X_test.index
    set_model_start_values(model, dist, X_test, normalized=normalize)
    preds = model.predict(X_test, pred_type="parameters")
    preds.index = idx
    prob_params = pd.concat([prob_params, preds])

    prob_params_train.sort_index(inplace=True)
    prob_params_train["result"] = y_train["Result"]
    prob_params_validation.sort_index(inplace=True)
    prob_params_validation["result"] = y_validation["Result"]
    prob_params.sort_index(inplace=True)
    prob_params["result"] = y_test["Result"]
    X_train.sort_index(inplace=True)
    B_train.sort_index(inplace=True)
    y_train.sort_index(inplace=True)
    X_test.sort_index(inplace=True)
    B_test.sort_index(inplace=True)
    y_test.sort_index(inplace=True)
    X_validation.sort_index(inplace=True)
    B_validation.sort_index(inplace=True)
    y_validation.sort_index(inplace=True)
    B_train.loc[B_train["Odds"] == 0, "Odds"] = 0.5
    B_test.loc[B_test["Odds"] == 0, "Odds"] = 0.5
    B_validation.loc[B_validation["Odds"] == 0, "Odds"] = 0.5

    r_validation = None
    r_test = None
    gate_test = None
    gate_validation = None
    sn_sigma_test = None
    sn_sigma_val = None
    sn_alpha_test = None
    sn_alpha_val = None

    if dist == "SkewNormal":
        loc_norm = prob_params["loc"].to_numpy()
        scale_norm = prob_params["scale"].to_numpy()
        alpha_sn = prob_params["alpha"].to_numpy()
        loc_norm_val = prob_params_validation["loc"].to_numpy()
        scale_norm_val = prob_params_validation["scale"].to_numpy()
        alpha_sn_val = prob_params_validation["alpha"].to_numpy()

        meanyr_test = X_test[denom_col].clip(lower=0.5).to_numpy()
        meanyr_val = X_validation[denom_col].clip(lower=0.5).to_numpy()
        loc_abs = loc_norm * meanyr_test
        scale_abs = scale_norm * meanyr_test
        loc_abs_val = loc_norm_val * meanyr_val
        scale_abs_val = scale_norm_val * meanyr_val

        delta = alpha_sn / np.sqrt(1 + alpha_sn**2)
        ev = loc_abs + scale_abs * delta * np.sqrt(2 / np.pi)
        delta_val = alpha_sn_val / np.sqrt(1 + alpha_sn_val**2)
        ev_validation = loc_abs_val + scale_abs_val * delta_val * np.sqrt(2 / np.pi)

        sn_sigma_test = scale_abs
        sn_sigma_val = scale_abs_val
        sn_alpha_test = alpha_sn
        sn_alpha_val = alpha_sn_val

        if hist_gate > 0.02:
            gate_test = np.full_like(ev, hist_gate)
            gate_validation = np.full_like(ev_validation, hist_gate)

    elif dist in ("NegBin", "ZINB"):
        r = prob_params["total_count"].to_numpy()
        p = prob_params["probs"].to_numpy()
        ev = r * p / (1 - p)
        r_validation = prob_params_validation["total_count"].to_numpy()
        p_validation = prob_params_validation["probs"].to_numpy()
        ev_validation = r_validation * p_validation / (1 - p_validation)
        alpha_validation = None
        if dist == "ZINB":
            gate_test = prob_params["gate"].to_numpy()
            gate_validation = prob_params_validation["gate"].to_numpy()
    elif dist in ("Gamma", "ZAGamma"):
        alpha = prob_params["concentration"].to_numpy()
        beta = prob_params["rate"].to_numpy()
        ev = alpha / beta
        alpha_validation = prob_params_validation["concentration"].to_numpy()
        beta_validation = prob_params_validation["rate"].to_numpy()
        ev_validation = alpha_validation / beta_validation
        if dist == "ZAGamma":
            gate_test = prob_params["gate"].to_numpy()
            gate_validation = prob_params_validation["gate"].to_numpy()

    if dist == "SkewNormal":
        base_dist = "SkewNormal"
    else:
        base_dist = "NegBin" if dist in ("NegBin", "ZINB") else "Gamma"
    book_ev_test = B_test["EV"].to_numpy()
    book_ev_val = B_validation["EV"].to_numpy()

    if dist == "SkewNormal":
        _zi_kwargs = dict(gate_book=hist_gate) if hist_gate > 0.02 else {}
        model_weight = fit_model_weight(
            ev_validation,
            book_ev_val,
            y_validation["Result"].to_numpy(),
            "SkewNormal",
            cv=cv,
            model_sigma=sn_sigma_val,
            model_skew_alpha=sn_alpha_val,
            **_zi_kwargs,
        )

        _zi_test = dict(gate_book=hist_gate) if hist_gate > 0.02 else {}
        _zi_val = dict(gate_book=hist_gate) if hist_gate > 0.02 else {}
        weighted_mean, sn_sigma_blend_test, sn_alpha_blend_test, gate_blend_test = fused_loc(
            model_weight,
            ev,
            book_ev_test,
            cv,
            "SkewNormal",
            sigma=sn_sigma_test,
            skew_alpha=sn_alpha_test,
            **_zi_test,
        )
        sn_sigma_blend_val, sn_alpha_blend_val = None, None
        _, sn_sigma_blend_val, sn_alpha_blend_val, gate_blend_val = fused_loc(
            model_weight,
            ev_validation,
            book_ev_val,
            cv,
            "SkewNormal",
            sigma=sn_sigma_val,
            skew_alpha=sn_alpha_val,
            **_zi_val,
        )

        r_test = None

    else:
        _zi_kwargs = {}
        if dist in ("ZINB", "ZAGamma") and hist_gate > 0:
            _zi_kwargs = dict(gate_model=gate_validation, gate_book=hist_gate)
        model_weight = fit_model_weight(
            ev_validation,
            book_ev_val,
            y_validation["Result"].to_numpy(),
            base_dist,
            model_alpha=alpha_validation,
            model_r=r_validation,
            cv=cv,
            **_zi_kwargs,
        )

        if dist in ("NegBin", "ZINB"):
            _zi_test = (
                dict(gate_model=gate_test, gate_book=hist_gate) if dist == "ZINB" else {}
            )
            _zi_val = (
                dict(gate_model=gate_validation, gate_book=hist_gate) if dist == "ZINB" else {}
            )
            r_blend_test, p_test, gate_blend_test = fused_loc(
                model_weight, ev, book_ev_test, cv, "NegBin", r=r, **_zi_test
            )
            weighted_mean = r_blend_test * (1 - p_test) / p_test
            r_test = r_blend_test

            r_blend_val, p_val, gate_blend_val = fused_loc(
                model_weight,
                ev_validation,
                book_ev_val,
                cv,
                "NegBin",
                r=r_validation,
                **_zi_val,
            )

        else:
            _zi_test = (
                dict(gate_model=gate_test, gate_book=hist_gate) if dist == "ZAGamma" else {}
            )
            _zi_val = (
                dict(gate_model=gate_validation, gate_book=hist_gate) if dist == "ZAGamma" else {}
            )
            alpha_blend, beta_blend, gate_blend_test = fused_loc(
                model_weight, ev, book_ev_test, cv, "Gamma", alpha=alpha, **_zi_test
            )
            weighted_mean = alpha_blend / beta_blend

            alpha_blend_val, beta_blend_val, gate_blend_val = fused_loc(
                model_weight,
                ev_validation,
                book_ev_val,
                cv,
                "Gamma",
                alpha=alpha_validation,
                **_zi_val,
            )
            r_test = None

    # Dispersion calibration
    y_class_val = (y_validation["Result"] >= B_validation["Line"]).astype(int).to_numpy()
    y_val_arr = y_validation["Result"].to_numpy()

    if dist == "SkewNormal":
        c_opt = 1.0
        val_weighted_mean_val, _, _, _ = fused_loc(
            model_weight,
            ev_validation,
            book_ev_val,
            cv,
            "SkewNormal",
            sigma=sn_sigma_val,
            skew_alpha=sn_alpha_val,
            **(dict(gate_book=hist_gate) if hist_gate > 0.02 else {}),
        )
    else:
        val_weighted_mean = (
            r_blend_val * (1 - p_val) / p_val
            if dist in ("NegBin", "ZINB")
            else alpha_blend_val / beta_blend_val
        )

        def dispersion_loss(c):
            if dist in ("NegBin", "ZINB"):
                r_cal = r_blend_val * c
                p_cal = r_cal / (r_cal + val_weighted_mean)

                k_max = int(max(y_val_arr.max() * 2, np.mean(val_weighted_mean) * 4, 30))
                k_vals = np.arange(k_max + 1)

                cdf = nbinom.cdf(k_vals[:, None], r_cal[None, :], p_cal[None, :])
                if gate_blend_val is not None:
                    cdf = gate_blend_val[None, :] + (1 - gate_blend_val[None, :]) * cdf

                indicator = (y_val_arr[None, :] <= k_vals[:, None]).astype(float)
                crps = np.sum((cdf - indicator) ** 2, axis=0)
            else:
                alpha_cal = alpha_blend_val * c
                scale_cal = val_weighted_mean / alpha_cal

                if gate_blend_val is not None:
                    x_max = max(y_val_arr.max() * 2, np.mean(val_weighted_mean) * 4)
                    x_grid = np.linspace(0, x_max, 500)
                    dx = x_grid[1] - x_grid[0]

                    cdf_grid = gamma.cdf(
                        x_grid[:, None], alpha_cal[None, :], scale=scale_cal[None, :]
                    )
                    cdf_grid = (
                        gate_blend_val[None, :] + (1 - gate_blend_val[None, :]) * cdf_grid
                    )

                    indicator = (y_val_arr[None, :] <= x_grid[:, None]).astype(float)
                    crps = np.sum((cdf_grid - indicator) ** 2, axis=0) * dx
                else:
                    F_y = gamma.cdf(y_val_arr, alpha_cal, scale=scale_cal)
                    F_y_a1 = gamma.cdf(y_val_arr, alpha_cal + 1, scale=scale_cal)
                    crps = (
                        y_val_arr * (2 * F_y - 1)
                        - val_weighted_mean * (2 * F_y_a1 - 1)
                        - scale_cal / beta_fn(0.5, alpha_cal)
                    )

            reg = 0.01 * np.log(c) ** 2
            return np.mean(crps) + reg

        if dist in ("NegBin", "ZINB"):
            mean_shape = np.mean(r_blend_val)
        else:
            mean_shape = np.mean(alpha_blend_val)
        max_c = shape_ceiling / mean_shape if mean_shape > 0 else 10.0
        upper_bound = min(10.0, max_c)

        disp_result = minimize_scalar(
            dispersion_loss, bounds=(0.1, upper_bound), method="bounded"
        )
        c_opt = disp_result.x

        if dist in ("NegBin", "ZINB"):
            r_test = r_test * c_opt
            r_blend_val = r_blend_val * c_opt
        else:
            alpha_blend = alpha_blend * c_opt
            beta_blend = alpha_blend / weighted_mean
            alpha_blend_val = alpha_blend_val * c_opt
            beta_blend_val = alpha_blend_val / val_weighted_mean

    # Test set probabilities with calibrated params
    if dist == "SkewNormal":
        y_proba_no_filt = get_odds(
            B_test["Line"].to_numpy(),
            weighted_mean,
            "SkewNormal",
            sigma=sn_sigma_blend_test,
            skew_alpha=sn_alpha_blend_test,
            gate=gate_blend_test,
        )
    elif dist in ("NegBin", "ZINB"):
        y_proba_no_filt = get_odds(
            B_test["Line"].to_numpy(), weighted_mean, dist, r=r_test, gate=gate_blend_test
        )
    else:
        y_proba_no_filt = get_odds(
            B_test["Line"].to_numpy(),
            weighted_mean,
            dist,
            alpha=alpha_blend,
            step=step,
            gate=gate_blend_test,
        )
    y_proba_no_filt = np.array([y_proba_no_filt, 1 - y_proba_no_filt]).transpose()

    # Temperature scaling calibration
    if dist == "SkewNormal":
        val_raw_under = get_odds(
            B_validation["Line"].to_numpy(),
            val_weighted_mean_val,
            "SkewNormal",
            sigma=sn_sigma_blend_val,
            skew_alpha=sn_alpha_blend_val,
            gate=gate_blend_val,
        )
    else:
        _r_val = r_blend_val if dist in ("NegBin", "ZINB") else None
        _alpha_val = alpha_blend_val if dist in ("Gamma", "ZAGamma") else None
        _gate_val = gate_blend_val if dist in ("ZINB", "ZAGamma") else None
        val_raw_under = get_odds(
            B_validation["Line"].to_numpy(),
            val_weighted_mean,
            dist,
            alpha=_alpha_val,
            step=step,
            r=_r_val,
            gate=_gate_val,
        )
    val_raw_over = 1 - val_raw_under

    val_raw_over_clipped = np.clip(val_raw_over, 1e-6, 1 - 1e-6)
    val_logits = logit(val_raw_over_clipped)

    def brier_loss(T):
        cal = expit(val_logits / T)
        brier = np.mean((cal - y_class_val) ** 2)
        reg = 0.01 * (T - 1) ** 2
        return brier + reg

    result_ts = minimize_scalar(brier_loss, bounds=(1.0, 10.0), method="bounded")
    T_opt = result_ts.x

    val_calibrated = expit(val_logits / T_opt)
    model_calib = 1 - np.mean((val_calibrated - y_class_val) ** 2)

    test_raw_over = y_proba_no_filt[:, 1]
    test_raw_over_clipped = np.clip(test_raw_over, 1e-6, 1 - 1e-6)
    test_calibrated_over = expit(logit(test_raw_over_clipped) / T_opt)
    y_proba_filt = np.array([1 - test_calibrated_over, test_calibrated_over]).transpose()

    stat_std = y.Result.std()

    y_class = (y_test["Result"] >= B_test["Line"]).astype(int)
    y_class = np.ravel(y_class.to_numpy())

    if dist == "SkewNormal":
        _raw_under = get_odds(
            B_test["Line"].to_numpy(),
            ev,
            "SkewNormal",
            sigma=sn_sigma_test,
            skew_alpha=sn_alpha_test,
            gate=gate_test,
        )
    elif dist in ("NegBin", "ZINB"):
        _raw_under = get_odds(B_test["Line"].to_numpy(), ev, dist, r=r, gate=gate_test)
    else:
        _raw_under = get_odds(
            B_test["Line"].to_numpy(), ev, dist, alpha=alpha, step=step, gate=gate_test
        )
    _raw_under = np.clip(_raw_under, 0, 1)
    y_proba_raw = np.array([_raw_under, 1 - _raw_under]).transpose()

    prec = np.zeros(3)
    acc = np.zeros(3)
    sharp = np.zeros(3)
    ll = np.zeros(3)
    over_pct = np.zeros(3)
    under_prec = np.zeros(3)

    for i, y_proba in enumerate([y_proba_raw, y_proba_no_filt, y_proba_filt]):
        y_pred = (y_proba > 0.5).astype(int)[:, 1]
        mask = np.max(y_proba, axis=1) > 0.54
        prec[i] = precision_score(y_class[mask], y_pred[mask])
        acc[i] = accuracy_score(y_class[mask], y_pred[mask])
        sharp[i] = np.std(y_proba[:, 1])
        ll[i] = log_loss(y_class, y_proba[:, 1])
        over_pct[i] = y_pred[mask].mean() / mask.mean() if mask.sum() > 0 else np.nan
        under_mask = mask & (y_pred == 0)
        under_prec[i] = (
            (y_class[under_mask] == 0).mean() if under_mask.sum() > 0 else np.nan
        )

    # Shape parameter diagnostics
    test_mean_yr = X_test["MeanYr"].mean()
    test_std_yr = X_test["STDYr"].mean()
    test_denom_mean = X_test[denom_col].mean() if dist == "SkewNormal" else test_mean_yr

    if dist == "SkewNormal":
        diag_start_shape = float(cv)
        scale_norm_mean = float(prob_params["scale"].mean())
        diag_model_shape = scale_norm_mean * test_denom_mean
        result_arr = y_test["Result"].to_numpy()
        diag_empirical_shape = float(result_arr.std() / max(result_arr.mean(), 1e-6))
        diag_shape_label = "scale"
    elif dist in ("Gamma", "ZAGamma"):
        diag_start_shape = float(
            np.clip((test_mean_yr / max(test_std_yr, 1e-6)) ** 2, 0.1, 100)
        )
        diag_model_shape = float(prob_params["concentration"].mean())
        per_player_emp_alpha = (
            player_stats.mean() / np.maximum(player_stats.std(), 0.01)
        ) ** 2
        diag_empirical_shape = float(np.median(per_player_emp_alpha))
        diag_shape_label = "alpha"
    elif dist in ("NegBin", "ZINB"):
        diag_start_shape = float(
            np.clip(test_mean_yr**2 / max(test_std_yr**2 - test_mean_yr, 1e-6), 1, 50)
        )
        diag_model_shape = float(prob_params["total_count"].mean())
        R_CAP = 50
        per_player_emp_r = player_stats.mean() ** 2 / np.maximum(
            player_stats.var() - player_stats.mean(), 0.01
        )
        per_player_emp_r = np.minimum(per_player_emp_r, R_CAP)
        diag_empirical_shape = float(np.median(per_player_emp_r))
        diag_shape_label = "r"

    diag_start_mean = float(test_mean_yr)
    diag_model_ev = float(weighted_mean.mean())
    diag_mean_line = float(B_test["Line"].mean())
    diag_ev_minus_line = float((weighted_mean - B_test["Line"].to_numpy()).mean())
    diag_result_mean = float(y_test["Result"].mean())

    _meanyr_arr = X_test["MeanYr"].to_numpy()
    _result_arr = y_test["Result"].to_numpy()
    diag_ev_meanyr_corr = float(np.corrcoef(_meanyr_arr, weighted_mean - _meanyr_arr)[0, 1])
    diag_result_meanyr_corr = float(
        np.corrcoef(_meanyr_arr, _result_arr - _meanyr_arr)[0, 1]
    )

    ev_minus_line_arr = weighted_mean - B_test["Line"].to_numpy()
    diag_median_ev_diff = float(np.median(ev_minus_line_arr))
    diag_frac_ev_gt_line = float((ev_minus_line_arr > 0).mean())

    ev_gt_mask = ev_minus_line_arr > 0
    ev_lt_mask = ev_minus_line_arr <= 0
    conf_mask = np.max(y_proba_no_filt, axis=1) > 0.54
    if (ev_gt_mask & conf_mask).sum() > 10:
        diag_over_pct_ev_gt = float(y_class[ev_gt_mask & conf_mask].mean())
    else:
        diag_over_pct_ev_gt = float("nan")
    if (ev_lt_mask & conf_mask).sum() > 10:
        diag_over_pct_ev_lt = float(y_class[ev_lt_mask & conf_mask].mean())
    else:
        diag_over_pct_ev_lt = float("nan")

    if dist == "SkewNormal":
        diag_cf_over_pct = float("nan")
    elif not np.isnan(diag_empirical_shape) and diag_empirical_shape > 0:
        emp_shape = np.full_like(weighted_mean, diag_empirical_shape)
        if dist in ("NegBin", "ZINB"):
            cf_under = get_odds(
                B_test["Line"].to_numpy(),
                weighted_mean,
                dist,
                r=emp_shape,
                gate=gate_blend_test,
            )
        else:
            cf_under = get_odds(
                B_test["Line"].to_numpy(),
                weighted_mean,
                dist,
                alpha=emp_shape,
                step=step,
                gate=gate_blend_test,
            )
        cf_over = 1 - cf_under
        cf_pred = (cf_over > 0.5).astype(int)
        cf_mask = np.maximum(cf_under, cf_over) > 0.54
        diag_cf_over_pct = (
            float(cf_pred[cf_mask].mean() / cf_mask.mean())
            if cf_mask.sum() > 10
            else float("nan")
        )
    else:
        diag_cf_over_pct = float("nan")

    filedict = {
        "model": model,
        "step": step,
        "stats": {
            "Accuracy": acc,
            "Over Prec": prec,
            "Under Prec": under_prec,
            "Over%": over_pct,
            "Sharpness": sharp,
            "NLL": ll,
        },
        "diagnostics": {
            "model_weight": model_weight,
            "model_calib": model_calib,
            "shape_label": diag_shape_label,
            "start_shape": diag_start_shape,
            "model_shape": diag_model_shape,
            "empirical_shape": diag_empirical_shape,
            "start_mean": diag_start_mean,
            "model_ev": diag_model_ev,
            "mean_line": diag_mean_line,
            "ev_minus_line": diag_ev_minus_line,
            "result_mean": diag_result_mean,
            "dispersion_cal": c_opt,
            "median_ev_diff": diag_median_ev_diff,
            "frac_ev_gt_line": diag_frac_ev_gt_line,
            "over_pct_ev_gt": diag_over_pct_ev_gt,
            "over_pct_ev_lt": diag_over_pct_ev_lt,
            "cf_over_pct": diag_cf_over_pct,
            "ev_meanyr_corr": diag_ev_meanyr_corr,
            "result_meanyr_corr": diag_result_meanyr_corr,
            "shape_ceiling": shape_ceiling,
            "marginal_shape": marginal_shape,
        },
        "params": opt_params,
        "distribution": dist,
        "cv": cv,
        "std": stat_std,
        "temperature": T_opt,
        "dispersion_cal": c_opt,
        "weight": model_weight,
        "r_book": None,
        "hist_gate": hist_gate,
        "shape_ceiling": shape_ceiling,
        "normalized": normalize,
        "expected_columns": list(X.columns),
    }

    X_test["Result"] = y_test["Result"]
    X_test["Line"] = B_test["Line"].values
    X_test["Blended_EV"] = weighted_mean
    if dist == "SkewNormal":
        X_test["EV"] = ev
        X_test["SN_Loc"] = prob_params["loc"]
        X_test["SN_Scale"] = prob_params["scale"]
        X_test["SN_Alpha"] = prob_params["alpha"]
        if hist_gate > 0.02:
            X_test["Gate"] = hist_gate
    elif dist in ("NegBin", "ZINB"):
        base_ev = (
            prob_params["total_count"] * prob_params["probs"] / (1 - prob_params["probs"])
        )
        X_test["EV"] = base_ev
        if dist == "ZINB":
            X_test["Gate"] = prob_params["gate"]
        X_test["R"] = prob_params["total_count"]
        X_test["NB_P"] = prob_params["probs"]
    elif dist in ("Gamma", "ZAGamma"):
        base_ev = prob_params["concentration"] / prob_params["rate"]
        X_test["EV"] = base_ev
        if dist == "ZAGamma":
            X_test["Gate"] = prob_params["gate"]
        X_test["Alpha"] = prob_params["concentration"]

    X_test["P"] = y_proba_filt[:, 1]

    filepath = pkg_resources.files(data) / f"test_sets/{filename}.csv"
    with open(filepath, "w") as outfile:
        X_test.to_csv(filepath)

    filepath = pkg_resources.files(data) / f"models/{filename}.mdl"
    with open(filepath, "wb") as outfile:
        pickle.dump(filedict, outfile, -1)
        del filedict
        del model

    report()
