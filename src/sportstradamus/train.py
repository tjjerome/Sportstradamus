from sportstradamus.stats import StatsMLB, StatsNBA, StatsNHL, StatsNFL, StatsWNBA
from sportstradamus.helpers import get_ev, get_odds, stat_cv, stat_zi, Archive, book_weights, feature_filter, set_model_start_values, fused_loc
import pickle
import importlib.resources as pkg_resources
from sportstradamus import data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logit, expit
from scipy.stats import poisson, nbinom, norm, gamma, fit
from sklearn.metrics import (
    precision_score,
    accuracy_score,
    log_loss
)
from scipy.special import softmax
import lightgbm as lgb
import pandas as pd
import click
import os
from datetime import datetime, timedelta
from tqdm import tqdm
from lightgbmlss.model import LightGBMLSS
from lightgbmlss.distributions.NegativeBinomial import NegativeBinomial
from lightgbmlss.distributions.ZINB import ZINB
from lightgbmlss.distributions.Gamma import Gamma
from lightgbmlss.distributions.ZAGamma import ZAGamma

import torch
import shap


class _BoundedResponseFn:
    """Picklable callable that clamps a response function's output."""
    def __init__(self, orig_fn, ceiling):
        self.orig_fn = orig_fn
        self.ceiling = float(ceiling)

    def __call__(self, predt):
        return torch.clamp(self.orig_fn(predt), max=self.ceiling)
import json
import warnings
pd.options.mode.chained_assignment = None
pd.set_option('future.no_silent_downcasting', True)
np.seterr(divide='ignore', invalid='ignore')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_distribution_config():
    """Load distribution configuration from stat_dist.json."""
    filepath = pkg_resources.files(data) / "stat_dist.json"
    if os.path.isfile(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}

def save_distribution_config(config):
    """Save distribution configuration to stat_dist.json."""
    filepath = pkg_resources.files(data) / "stat_dist.json"
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)

def load_zi_config():
    """Load zero-inflation gate configuration from stat_zi.json."""
    filepath = pkg_resources.files(data) / "stat_zi.json"
    if os.path.isfile(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}

def save_zi_config(config):
    """Save zero-inflation gate configuration to stat_zi.json."""
    filepath = pkg_resources.files(data) / "stat_zi.json"
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)

def warm_start_hyper_opt(model, hp_dict, train_set, initial_params,
                         num_boost_round=999, nfold=4,
                         early_stopping_rounds=50, max_minutes=15,
                         n_trials=100, silence=True):
    """Run a shortened hyper_opt seeded with previous best parameters."""
    import optuna
    from optuna.samplers import TPESampler
    from optuna.integration import LightGBMPruningCallback

    tunable_params = {k for k, v in hp_dict.items() if v[0] != "none"}

    def objective(trial):
        hyper_params = {}
        for param_name, param_value in hp_dict.items():
            param_type = param_value[0]
            if param_type == "categorical" or param_type == "none":
                hyper_params[param_name] = trial.suggest_categorical(param_name, param_value[1])
            elif param_type == "float":
                c = param_value[1]
                hyper_params[param_name] = trial.suggest_float(
                    param_name, low=c["low"], high=c["high"], log=c["log"])
            elif param_type == "int":
                c = param_value[1]
                hyper_params[param_name] = trial.suggest_int(
                    param_name, low=c["low"], high=c["high"], log=c["log"])

        if "boosting" not in hyper_params:
            hyper_params["boosting"] = trial.suggest_categorical("boosting", ["gbdt"])

        pruning_callback = LightGBMPruningCallback(trial, model.dist.loss_fn)
        early_stopping_callback = lgb.early_stopping(
            stopping_rounds=early_stopping_rounds, verbose=False)

        cv_result = model.cv(hyper_params, train_set,
                             num_boost_round=num_boost_round,
                             nfold=nfold,
                             callbacks=[pruning_callback, early_stopping_callback],
                             seed=None)

        opt_rounds = np.argmin(
            np.array(cv_result[f"valid {model.dist.loss_fn}-mean"])) + 1
        trial.set_user_attr("opt_round", int(opt_rounds))
        return np.min(np.array(cv_result[f"valid {model.dist.loss_fn}-mean"]))

    if silence:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = TPESampler()
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="minimize",
                                study_name="LightGBMLSS Warm-Start Optimization")

    # Enqueue previous best params as the first trial
    seed_params = {k: v for k, v in initial_params.items() if k in tunable_params}
    seed_params["boosting"] = "gbdt"
    study.enqueue_trial(seed_params)

    study.optimize(objective, n_trials=n_trials, timeout=60 * max_minutes,
                   show_progress_bar=True)

    print("\nWarm-Start Hyper-Parameter Optimization finished.")
    print("  Number of finished trials: ", len(study.trials))
    print("  Best trial:")
    opt_param = study.best_trial
    opt_param.params["opt_rounds"] = int(
        study.trials_dataframe()["user_attrs_opt_round"][
            study.trials_dataframe()["value"].idxmin()])

    print("    Value: {}".format(opt_param.value))
    print("    Params: ")
    for key, value in opt_param.params.items():
        print("    {}: {}".format(key, value))

    return opt_param.params

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

@click.command()
@click.option("--force/--no-force", default=False, help="Force update of all models")
@click.option("--league", type=click.Choice(["All", "NFL", "NBA", "MLB", "NHL", "WNBA"]), default="All",
              help="Select league to train on")
def meditate(force, league):
    global book_weights, archive, stat_structs

    np.random.seed(69)

    # Initialize stats and archive
    nba = StatsNBA()
    nfl = StatsNFL()
    wnba = StatsWNBA()
    # mlb = StatsMLB()
    # nhl = StatsNHL()

    stat_structs = {}
    archive = Archive()

    if datetime.today().date() > (nba.season_start - timedelta(days=7)) or league == "NBA":
        nba.load()
        nba.update()
        stat_structs.update({"NBA": nba})
    if datetime.today().date() > (nfl.season_start - timedelta(days=7)) or league == "NFL":
        nfl.load()
        nfl.update()
        stat_structs.update({"NFL": nfl})
    if datetime.today().date() > (wnba.season_start - timedelta(days=7)) or league == "WNBA":
        wnba.load()
        wnba.update()
        stat_structs.update({"WNBA": wnba})
    # if datetime.today().date() > (mlb.season_start - timedelta(days=7)) or league == "MLB":
    #     mlb.load()
    #     mlb.update()
    #     stat_structs.update({"MLB": mlb})
    # if datetime.today().date() > (nhl.season_start - timedelta(days=7)) or league == "NHL":
    #     nhl.load()
    #     nhl.update()
    #     stat_structs.update({"NHL": nhl})

    # === MARKET DEFINITIONS ===
    # Define all available betting markets for each league
    all_markets = {
        "NFL": [
            "targets",
            "carries",
            "attempts",
            "passing yards",
            "rushing yards",
            "receiving yards",
            "yards",
            "qb yards",
            "fantasy points prizepicks",
            "fantasy points underdog",
            "passing tds",
            "tds",
            "rushing tds",
            "receiving tds",
            "qb tds",
            "completions",
            "receptions",
            "interceptions",
            "sacks taken",
            "passing first downs",
            # "first downs",
            # "completion percentage"
        ],
        "NBA": [
            "MIN",
            "PTS",
            "REB",
            "AST",
            "PRA",
            "PR",
            "RA",
            "PA",
            "FG3M",
            "fantasy points prizepicks",
            "FG3A",
            "FTM",
            "FGM",
            "FGA",
            "STL",
            "BLK",
            "BLST",
            "TOV",
            "OREB",
            "DREB",
            "PF",
        ],
        "WNBA": [
            "MIN",
            "AST",
            "FG3M",
            "PA",
            "PR",
            "PTS",
            "RA",
            "REB",
            "OREB",
            "DREB",
            "FGA",
            "BLK",
            "STL",
            "BLST",
            "TOV",
            "FTM",
            "PRA",
            "fantasy points prizepicks"
        ],
        "MLB": [
            "plateAppearances",
            "pitches thrown",
            "pitching outs",
            "pitcher strikeouts",
            "hits allowed",
            "runs allowed",
            "walks allowed",
            # "1st inning runs allowed",
            # "1st inning hits allowed",
            "hitter fantasy score",
            "pitcher fantasy score",
            "hitter fantasy points underdog",
            "pitcher fantasy points underdog",
            "hits+runs+rbi",
            "total bases",
            "walks",
            "stolen bases",
            "hits",
            "runs",
            "rbi",
            "batter strikeouts",
            "singles",
            "doubles",
            "home runs"
        ],
        "NHL": [
            "timeOnIce",
            "shotsAgainst",
            "saves",
            "shots",
            "points",
            "goalsAgainst",
            "goalie fantasy points underdog",
            "skater fantasy points underdog",
            "blocked",
            "powerPlayPoints",
            "sogBS",
            # "fantasy points prizepicks",
            "hits",
            "goals",
            "assists",
            "faceOffWins",
        ]
    }
    if not league == "All":
        all_markets = {league: all_markets[league]}
    for league, markets in all_markets.items():
        stat_data = stat_structs.get(league)
        if stat_data is None:
            continue
        
        book_weights.setdefault(league, {}).setdefault("Moneyline", {})
        book_weights[league]["Moneyline"] = fit_book_weights(league, "Moneyline")

        book_weights.setdefault(league, {}).setdefault("Totals", {})
        book_weights[league]["Totals"] = fit_book_weights(league, "Totals")

        if league == "MLB":
            book_weights.setdefault(league, {}).setdefault("1st 1 innings", {})
            book_weights[league]["1st 1 innings"] = fit_book_weights(league, "1st 1 innings")
            book_weights.setdefault(league, {}).setdefault("pitcher win", {})
            book_weights[league]["pitcher win"] = fit_book_weights(league, "pitcher win")
            book_weights.setdefault(league, {}).setdefault("triples", {})
            book_weights[league]["triples"] = fit_book_weights(league, "triples")

        elif league == "NHL":
            stat_data.dump_goalie_list()

        with open(pkg_resources.files(data) / "book_weights.json", 'w') as outfile:
            json.dump(book_weights, outfile, indent=4)

        stat_data.update_player_comps()
        correlate(league, force)
        league_start_date = stat_data.trim_gamelog()

        for market in markets:
            stat_dist = load_distribution_config()
            stat_dist.setdefault(league, {})
            stat_zi.setdefault(league, {})

            if os.path.isfile(pkg_resources.files(data) / "book_weights.json"):
                with open(pkg_resources.files(data) / "book_weights.json", 'r') as infile:
                    book_weights = json.load(infile)
            else:
                book_weights = {}

            book_weights.setdefault(league, {}).setdefault(market, {})
            book_weights[league][market] = fit_book_weights(league, market)

            with open(pkg_resources.files(data) / "book_weights.json", 'w') as outfile:
                json.dump(book_weights, outfile, indent=4)

            filename = "_".join([league, market]).replace(" ", "-")
            filepath = pkg_resources.files(data) / f"models/{filename}.mdl"
            need_model = True
            if os.path.isfile(filepath):
                with open(filepath, 'rb') as infile:
                    filedict = pickle.load(infile)
                    model = filedict['model']
                    params = filedict['params']
                    dist = filedict['distribution']
                    cv = filedict['cv']
                    step = filedict['step']
                
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
                M = M.loc[(pd.to_datetime(M.Date).dt.date <= cutoff_date) & (pd.to_datetime(M.Date).dt.date > league_start_date)]
            else:
                cutoff_date = league_start_date
                M = pd.DataFrame()

            new_M = stat_data.get_training_matrix(market, cutoff_date)

            if new_M.empty and not force and not need_model:
                continue

            M = pd.concat([M, new_M], ignore_index=True)
            M.Date = pd.to_datetime(M.Date, format='mixed')
            if 'Player' in M.columns:
                M = M.drop_duplicates(subset=['Player', 'Date'], keep='last')
            step = M["Result"].drop_duplicates().sort_values().diff().min()
            _prep_gate = stat_zi.get(league, {}).get(market, 0) if dist in ("ZINB", "ZAGamma") else 0
            for i, row in M.loc[M.Odds.isna() | (M.Odds == 0)].iterrows():
                if np.isnan(row["EV"]) or row["EV"] <= 0:
                    M.loc[i, "Odds"] = 0.5
                    M.loc[i, "EV"] = get_ev(M.loc[i, "Line"], .5, cv=cv, dist=dist, gate=_prep_gate or None)
                else:
                    M.loc[i, "Odds"] = 1-get_odds(row["Line"], row["EV"], dist, cv=cv, step=step, gate=_prep_gate or None)

            M = trim_matrix(M, 15000)
            M.to_csv(filepath)

            # Write latest runtime comps to JSON for inspection
            stat_data.save_comps()

            y = M[['Result']]

            X = M[stat_data.get_stat_columns(market)]

            categories = ["Home", "Player position"]
            if "Player position" not in X.columns:
                categories.remove("Player position")
            for c in categories:
                X[c] = X[c].astype('category')

            categories = "name:"+",".join(categories)

            # === TEMPORAL SPLIT: earliest 70% of games (by date) to train, latest 30% to test ===
            # Sort by date, then split
            M_sorted = M.sort_values("Date")
            n = len(M_sorted)
            n_train = int(n * 0.7)
            train_idx = M_sorted.index[:n_train]
            test_idx = M_sorted.index[n_train:]

            X_train = X.loc[train_idx]
            y_train = y.loc[train_idx]
            X_test = X.loc[test_idx]
            y_test = y.loc[test_idx]

            # Keep test/validation split random as before
            X_test, X_validation, y_test, y_validation = train_test_split(
                X_test, y_test, test_size=0.5, random_state=25
            )

            B_train = M.loc[X_train.index, ['Line', 'Odds', 'EV']]
            B_test = M.loc[X_test.index, ['Line', 'Odds', 'EV']]
            B_validation = M.loc[X_validation.index, ['Line', 'Odds', 'EV']]

            y_train_labels = np.ravel(y_train.to_numpy())

            # === DISTRIBUTION SELECTION ===
            # Choose between NegBin (count data) or Gamma (continuous-like data)
            # Load from stat_dist.json or auto-select via NLL comparison
            # TODO Add Binomial dist for underdispersed markets like TDs?
            
            threshold_dict = {
                "NBA": 60,
                "NFL": 10,
                "NHL": 60,
                "WNBA": 40,
                "MLB": 60
            }
            threshold = threshold_dict.get(league, 60)
            player_stats = stat_data.gamelog.groupby(stat_data.log_strings.get("player")).filter(lambda x: x[market].gt(0).sum() > threshold).groupby(stat_data.log_strings.get("player"))[market]
            if market in stat_dist[league]:
                # Use previously selected distribution
                dist = stat_dist[league][market]
                hist_gate = stat_zi.get(league, {}).get(market, 0)
            else:
                dist, hist_gate = select_distribution(player_stats)
    
                # Save selected distribution for future runs
                stat_dist[league][market] = dist
                save_distribution_config(stat_dist)

                # Save historical zero rate for all distributions
                stat_zi[league][market] = hist_gate
                save_zi_config(stat_zi)
            
            player_stats = player_stats.apply(lambda x: x[x != 0])

            if dist in ["NegBin", "ZINB"]:
                if dist == "NegBin":
                    dist_obj = NegativeBinomial(stabilization="None", loss_fn="nll")
                else:
                    dist_obj = ZINB(stabilization="None", loss_fn="nll")

                # Marginal shape from training targets (full between+within-player variance)
                _nz = y_train_labels > 0
                _vy = y_train_labels[_nz] if _nz.sum() > 30 else y_train_labels
                _mu_m, _var_m = float(np.mean(_vy)), float(np.var(_vy))
                marginal_shape = max(_mu_m**2 / max(_var_m - _mu_m, 0.01), 0.5)
                K_SHAPE = 2.0
                shape_ceiling = marginal_shape * K_SHAPE

                cv = ((player_stats.var()-player_stats.mean())/player_stats.mean()**2*player_stats.count()/player_stats.count().sum()).sum()
                cv = max(cv, 1/shape_ceiling)
            else:
                if dist == "Gamma":
                    dist_obj = Gamma(stabilization="None", loss_fn="nll", response_fn="softplus")
                    nonzero_mask = y_train_labels > 0
                    X_train = X_train[nonzero_mask]
                    y_train = y_train[nonzero_mask]
                    y_train_labels = y_train_labels[nonzero_mask]
                else:
                    dist_obj = ZAGamma(stabilization="None", loss_fn="nll", response_fn="softplus")

                # Marginal shape from training targets (full between+within-player variance)
                _nz = y_train_labels > 0
                _vy = y_train_labels[_nz] if _nz.sum() > 30 else y_train_labels
                _mu_m, _var_m = float(np.mean(_vy)), float(np.var(_vy))
                marginal_shape = max((_mu_m / max(np.sqrt(_var_m), 0.01))**2, 0.5)
                K_SHAPE = 2.0
                shape_ceiling = marginal_shape * K_SHAPE

                cv = (player_stats.std()/player_stats.mean()*player_stats.count()/player_stats.count().sum()).sum()
                cv = max(cv, 1/np.sqrt(shape_ceiling))

            stat_cv[league][market] = cv
            with open(pkg_resources.files(data) / "stat_cv.json", "w") as f:
                json.dump(stat_cv, f, indent=4)
            
            opt_params = filedict.get("params")
            dtrain = lgb.Dataset(
                X_train, label=y_train_labels)
            
            # === BOUND SHAPE RESPONSE FUNCTION (safety net) ===
            if dist in ("NegBin", "ZINB"):
                dist_obj.param_dict["total_count"] = _BoundedResponseFn(dist_obj.param_dict["total_count"], shape_ceiling)
            elif dist in ("Gamma", "ZAGamma"):
                dist_obj.param_dict["concentration"] = _BoundedResponseFn(dist_obj.param_dict["concentration"], shape_ceiling)

            # === MODEL TRAINING ===
            # All distributions (NegBin, Gamma) use LightGBMLSS training
            model = LightGBMLSS(dist_obj)
            set_model_start_values(model, dist, X_train, shape_ceiling=shape_ceiling)

            hp_search_space = {
                "feature_pre_filter": ["none", [False]],
                "num_threads": ["none", [8]],
                "max_depth": ["none", [-1]],
                "max_bin": ["none", [127]],
                "hist_pool_size": ["none", [9*1024]],
                "path_smooth": ["float", {"low": 0, "high": 20, "log": False}],
                "num_leaves": ["int", {"low": 8, "high": 127, "log": False}],
                "lambda_l1": ["float", {"low": 1e-6, "high": 10, "log": True}],
                "lambda_l2": ["float", {"low": 1e-6, "high": 10, "log": True}],
                "min_child_samples": ["int", {"low": 30, "high": 150, "log": False}],
                "min_child_weight": ["float", {"low": 1e-3, "high": .75*len(X_train)/1000, "log": True}],
                "learning_rate": ["float", {"low": 0.001, "high": 0.15, "log": True}],
                "feature_fraction": ["float", {"low": 0.4, "high": 1.0, "log": False}],
                "bagging_fraction": ["float", {"low": 0.4, "high": 1.0, "log": False}],
                "bagging_freq": ["none", [1]]
            }

            if opt_params is None or opt_params.get("opt_rounds") is None:
                opt_params = model.hyper_opt(hp_search_space,
                                            dtrain,
                                            num_boost_round=999,
                                            nfold=4,
                                            early_stopping_rounds=50,
                                            max_minutes=60,
                                            n_trials=300,
                                            silence=True,
                                            )
            else:
                opt_params = warm_start_hyper_opt(model, hp_search_space, dtrain,
                                                  opt_params,
                                                  n_trials=200,
                                                  max_minutes=10)

            model.train(opt_params,
                        dtrain,
                        num_boost_round=opt_params["opt_rounds"]
                        )

            # === PREDICTIONS AND PARAMETER EXTRACTION ===
            # Generate predictions on all datasets and extract distribution parameters
            prob_params_train = pd.DataFrame()
            prob_params_validation = pd.DataFrame()
            prob_params = pd.DataFrame()

            # Use standard LightGBM prediction
            idx = X_train.index
            set_model_start_values(model, dist, X_train)
            preds = model.predict(X_train, pred_type="parameters")
            preds.index = idx
            prob_params_train = pd.concat([prob_params_train, preds])

            idx = X_validation.index
            set_model_start_values(model, dist, X_validation)
            preds = model.predict(X_validation, pred_type="parameters")
            preds.index = idx
            prob_params_validation = pd.concat([prob_params_validation, preds])

            idx = X_test.index
            set_model_start_values(model, dist, X_test)
            preds = model.predict(X_test, pred_type="parameters")
            preds.index = idx
            prob_params = pd.concat([prob_params, preds])

            prob_params_train.sort_index(inplace=True)
            prob_params_train['result'] = y_train['Result']
            prob_params_validation.sort_index(inplace=True)
            prob_params_validation['result'] = y_validation['Result']
            prob_params.sort_index(inplace=True)
            prob_params['result'] = y_test['Result']
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
            r_book = None  # NegBin market-level dispersion from historical data
            r_validation = None  # NegBin per-obs dispersion; None for Gamma
            r_test = None  # NegBin test-set dispersion for get_odds; None = use Poisson
            gate_test = None  # Zero-inflation gate for ZINB/ZAGamma; None for base dists
            gate_validation = None

            # Extract appropriate parameters based on distribution
            # ZINB inherits NegBin base params + gate; ZAGamma inherits Gamma + gate
            # For ZI dists, ev/ev_validation are BASE means (not deflated by gate)
            # so that fused_loc blends apples-to-apples with the book's base mean.
            if dist in ("NegBin", "ZINB"):
                r = prob_params["total_count"].to_numpy()
                p = prob_params["probs"].to_numpy()
                # PyTorch NegBin: probs = success probability → mean = n*p/(1-p)
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

            # === MODEL WEIGHTING AND PROBABILITY CALCULATION ===
            # fused_loc uses base distribution type (NegBin or Gamma) for blending
            # For ZI distributions, pass gate_model + hist_gate so fused_loc can
            # blend the gate alongside the base distribution parameters.
            # Book EVs on disk are already base means (get_ev with gate stores base
            # means), so no conversion is needed here.
            base_dist = "NegBin" if dist in ("NegBin", "ZINB") else "Gamma"
            book_ev_test = B_test["EV"].to_numpy()
            book_ev_val = B_validation["EV"].to_numpy()
            _zi_kwargs = {}
            if dist in ("ZINB", "ZAGamma") and hist_gate > 0:
                _zi_kwargs = dict(gate_model=gate_validation, gate_book=hist_gate)
            model_weight = fit_model_weight(ev_validation, book_ev_val, y_validation["Result"].to_numpy(), base_dist, model_alpha=alpha_validation, model_r=r_validation, cv=cv, **_zi_kwargs)

            if dist in ("NegBin", "ZINB"):
                # fused_loc blends both μ and r via geometric mean
                _zi_test = dict(gate_model=gate_test, gate_book=hist_gate) if dist == "ZINB" else {}
                _zi_val = dict(gate_model=gate_validation, gate_book=hist_gate) if dist == "ZINB" else {}
                r_blend_test, p_test, gate_blend_test = fused_loc(model_weight, ev, book_ev_test, cv, "NegBin", r=r, **_zi_test)
                weighted_mean = r_blend_test * (1 - p_test) / p_test
                r_test = r_blend_test

                r_blend_val, p_val, gate_blend_val = fused_loc(model_weight, ev_validation, book_ev_val, cv, "NegBin", r=r_validation, **_zi_val)

                test_alpha = None

            else:
                # Gamma / ZAGamma — precision-weighted blend
                _zi_test = dict(gate_model=gate_test, gate_book=hist_gate) if dist == "ZAGamma" else {}
                _zi_val = dict(gate_model=gate_validation, gate_book=hist_gate) if dist == "ZAGamma" else {}
                alpha_blend, beta_blend, gate_blend_test = fused_loc(model_weight, ev, book_ev_test, cv, "Gamma", alpha=alpha, **_zi_test)
                weighted_mean = alpha_blend / beta_blend

                alpha_blend_val, beta_blend_val, gate_blend_val = fused_loc(model_weight, ev_validation, book_ev_val, cv, "Gamma", alpha=alpha_validation, **_zi_val)
                r_test = None

                # Store alpha for downstream use
                test_alpha = alpha_blend

            # === DISPERSION CALIBRATION ===
            # Learn shape scaling factor on validation set to correct overconcentration.
            # Keeps mean fixed; only adjusts distributional width.
            y_class_val = (y_validation["Result"] >= B_validation["Line"]).astype(int).to_numpy()
            val_weighted_mean = r_blend_val * (1 - p_val) / p_val if dist in ("NegBin", "ZINB") else alpha_blend_val / beta_blend_val

            def dispersion_loss(c):
                if dist in ("NegBin", "ZINB"):
                    r_cal = r_blend_val * c
                    under = get_odds(B_validation["Line"].to_numpy(), val_weighted_mean, dist,
                                     r=r_cal, gate=gate_blend_val)
                else:
                    alpha_cal = alpha_blend_val * c
                    under = get_odds(B_validation["Line"].to_numpy(), val_weighted_mean, dist,
                                     alpha=alpha_cal, step=step, gate=gate_blend_val)
                over = 1 - under
                return np.mean((over - y_class_val) ** 2)

            disp_result = minimize_scalar(dispersion_loss, bounds=(0.1, 1.0), method='bounded')
            c_opt = disp_result.x

            # Apply dispersion calibration to test and validation params
            if dist in ("NegBin", "ZINB"):
                r_test = r_test * c_opt
                r_blend_val = r_blend_val * c_opt
            else:
                alpha_blend = alpha_blend * c_opt
                beta_blend = alpha_blend / weighted_mean
                test_alpha = alpha_blend
                alpha_blend_val = alpha_blend_val * c_opt
                beta_blend_val = alpha_blend_val / val_weighted_mean

            # Compute test set probabilities with calibrated params
            if dist in ("NegBin", "ZINB"):
                y_proba_no_filt = get_odds(B_test["Line"].to_numpy(), weighted_mean, dist, r=r_test, gate=gate_blend_test)
            else:
                y_proba_no_filt = get_odds(B_test["Line"].to_numpy(), weighted_mean, dist, alpha=alpha_blend, step=step, gate=gate_blend_test)
            y_proba_no_filt = np.array(
                [y_proba_no_filt, 1-y_proba_no_filt]).transpose()

            # === TEMPERATURE SCALING CALIBRATION ===
            # Fit temperature T on validation set: p_cal = sigmoid(logit(p_raw) / T)
            # T ≥ 1 ensures calibration only reduces confidence (pulls toward 0.5)
            _r_val = r_blend_val if dist in ("NegBin", "ZINB") else None
            _alpha_val = alpha_blend_val if dist in ("Gamma", "ZAGamma") else None
            _gate_val = gate_blend_val if dist in ("ZINB", "ZAGamma") else None
            # val_weighted_mean already computed before dispersion calibration (mean is invariant)
            val_raw_under = get_odds(B_validation["Line"].to_numpy(), val_weighted_mean, dist, alpha=_alpha_val, step=step, r=_r_val, gate=_gate_val)
            val_raw_over = 1 - val_raw_under

            val_raw_over_clipped = np.clip(val_raw_over, 1e-6, 1 - 1e-6)
            val_logits = logit(val_raw_over_clipped)

            def brier_loss(T):
                cal = expit(val_logits / T)
                return np.mean((cal - y_class_val) ** 2)

            result = minimize_scalar(brier_loss, bounds=(1.0, 10.0), method='bounded')
            T_opt = result.x

            # Evaluate calibration quality
            val_calibrated = expit(val_logits / T_opt)
            model_calib = 1 - np.mean((val_calibrated - y_class_val) ** 2)

            # Apply temperature scaling to test set
            test_raw_over = y_proba_no_filt[:, 1]
            test_raw_over_clipped = np.clip(test_raw_over, 1e-6, 1 - 1e-6)
            test_calibrated_over = expit(logit(test_raw_over_clipped) / T_opt)
            y_proba_filt = np.array(
                [1 - test_calibrated_over, test_calibrated_over]).transpose()

            stat_std = y.Result.std()

            # === TEST SET STATISTICS ===
            # Calculate model performance metrics on held-out test set
            y_class = (y_test["Result"] >=
                       B_test["Line"]).astype(int)
            y_class = np.ravel(y_class.to_numpy())

            prec = np.zeros(2)
            acc = np.zeros(2)
            sharp = np.zeros(2)
            ll = np.zeros(2)
            over_pct = np.zeros(2)
            under_prec = np.zeros(2)

            for i, y_proba in enumerate([y_proba_no_filt, y_proba_filt]):
                y_pred = (y_proba > .5).astype(int)[:, 1]
                mask = np.max(y_proba, axis=1) > 0.54
                prec[i] = precision_score(y_class[mask], y_pred[mask])
                acc[i] = accuracy_score(y_class[mask], y_pred[mask])

                sharp[i] = np.std(y_proba[:, 1])

                ll[i] = log_loss(y_class, y_proba[:, 1])

                # Directional diagnostics
                over_pct[i] = y_pred[mask].mean() if mask.sum() > 0 else np.nan
                under_mask = mask & (y_pred == 0)
                under_prec[i] = (y_class[under_mask] == 0).mean() if under_mask.sum() > 0 else np.nan

            # --- Shape parameter diagnostics ---
            # Three-way comparison: start values (method of moments) vs model vs empirical
            test_mean_yr = X_test['MeanYr'].mean()
            test_std_yr = X_test['STDYr'].mean()

            if dist in ("Gamma", "ZAGamma"):
                diag_start_shape = float(np.clip((test_mean_yr / max(test_std_yr, 1e-6)) ** 2, 0.1, 100))
                diag_model_shape = float(prob_params['concentration'].mean())
                test_mu = y_test['Result'].mean()
                test_std = y_test['Result'].std()
                diag_empirical_shape = float((test_mu / test_std) ** 2) if test_std > 0 else float('nan')
                diag_shape_label = "alpha"
            elif dist in ("NegBin", "ZINB"):
                diag_start_shape = float(np.clip(test_mean_yr ** 2 / max(test_std_yr ** 2 - test_mean_yr, 1e-6), 1, 50))
                diag_model_shape = float(prob_params['total_count'].mean())
                test_mu = y_test['Result'].mean()
                test_var = y_test['Result'].var()
                diag_empirical_shape = float(test_mu ** 2 / max(test_var - test_mu, 0.01))
                diag_shape_label = "r"

            # EV diagnostics: start mean vs blended model EV vs line vs actual results
            diag_start_mean = float(test_mean_yr)
            diag_model_ev = float(weighted_mean.mean())
            diag_mean_line = float(B_test['Line'].mean())
            diag_ev_minus_line = float((weighted_mean - B_test['Line'].to_numpy()).mean())
            diag_result_mean = float(y_test['Result'].mean())

            # --- Decomposition diagnostics ---
            # Median and fraction help distinguish mean bias from shape bias
            ev_minus_line_arr = weighted_mean - B_test['Line'].to_numpy()
            diag_median_ev_diff = float(np.median(ev_minus_line_arr))
            diag_frac_ev_gt_line = float((ev_minus_line_arr > 0).mean())

            # Conditional Over%: split by sign of (ev - line)
            ev_gt_mask = ev_minus_line_arr > 0
            ev_lt_mask = ev_minus_line_arr <= 0
            y_pred_nofilter = (y_proba_no_filt[:, 1] > 0.5).astype(int)
            conf_mask = np.max(y_proba_no_filt, axis=1) > 0.54
            if (ev_gt_mask & conf_mask).sum() > 10:
                diag_over_pct_ev_gt = float(y_pred_nofilter[ev_gt_mask & conf_mask].mean())
            else:
                diag_over_pct_ev_gt = float('nan')
            if (ev_lt_mask & conf_mask).sum() > 10:
                diag_over_pct_ev_lt = float(y_pred_nofilter[ev_lt_mask & conf_mask].mean())
            else:
                diag_over_pct_ev_lt = float('nan')

            # Counterfactual: what would Over% be with empirical shape?
            if not np.isnan(diag_empirical_shape) and diag_empirical_shape > 0:
                emp_shape = np.full_like(weighted_mean, diag_empirical_shape)
                if dist in ("NegBin", "ZINB"):
                    cf_under = get_odds(B_test["Line"].to_numpy(), weighted_mean, dist,
                                        r=emp_shape, gate=gate_blend_test)
                else:
                    cf_under = get_odds(B_test["Line"].to_numpy(), weighted_mean, dist,
                                        alpha=emp_shape, step=step, gate=gate_blend_test)
                cf_over = 1 - cf_under
                cf_pred = (cf_over > 0.5).astype(int)
                cf_mask = np.maximum(cf_under, cf_over) > 0.54
                diag_cf_over_pct = float(cf_pred[cf_mask].mean()) if cf_mask.sum() > 10 else float('nan')
            else:
                diag_cf_over_pct = float('nan')

            filedict = {
                "model": model,
                "step": step,
                "stats": {
                    "Accuracy": acc,
                    "Precision": prec,
                    "Under Prec": under_prec,
                    "Over%": over_pct,
                    "Sharpness": sharp,
                    "NLL": ll,
                    "Weight/Calib": [model_weight, model_calib]
                },
                "diagnostics": {
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
                "r_book": r_book,
                "hist_gate": hist_gate,
                "shape_ceiling": shape_ceiling,
            }

            X_test['Result'] = y_test['Result']
            X_test['Line'] = B_test['Line'].values
            X_test['Blended_EV'] = weighted_mean
            if dist in ("NegBin", "ZINB"):
                # Store base distribution mean: mean = r*p/(1-p)
                base_ev = prob_params['total_count'] * prob_params['probs'] / (1 - prob_params['probs'])
                X_test['EV'] = base_ev
                if dist == "ZINB":
                    X_test['Gate'] = prob_params['gate']
                X_test['R'] = prob_params['total_count']
                X_test['NB_P'] = prob_params['probs']
            elif dist in ("Gamma", "ZAGamma"):
                base_ev = prob_params['concentration'] / prob_params['rate']
                X_test['EV'] = base_ev
                if dist == "ZAGamma":
                    X_test['Gate'] = prob_params['gate']
                X_test['Alpha'] = prob_params['concentration']

            X_test['P'] = y_proba_filt[:, 1]

            # === SAVE TEST PREDICTIONS ===
            # Store predictions on test set for later analysis

            filepath = pkg_resources.files(data) / f"test_sets/{filename}.csv"
            with open(filepath, "w") as outfile:
                X_test.to_csv(filepath)

            filepath = pkg_resources.files(data) / f"models/{filename}.mdl"
            with open(filepath, "wb") as outfile:
                pickle.dump(filedict, outfile, -1)
                del filedict
                del model

            # === GENERATE TRAINING REPORT ===
            report()



# ============================================================================
# REPORTING AND ANALYSIS FUNCTIONS
# ============================================================================

def report():
    """Generate training report summarizing all model performance metrics."""
    model_list = [f.name for f in (pkg_resources.files(
        data)/"models/").iterdir() if ".mdl" in f.name]
    model_list.sort()
    with open(pkg_resources.files(data) / "stat_cv.json", "r") as f:
        stat_cv = json.load(f)
    with open(pkg_resources.files(data) / "stat_std.json", "r") as f:
        stat_std = json.load(f)
    stat_dist = load_distribution_config()
    stat_zi_local = load_zi_config()
    
    with open(pkg_resources.files(data) / "training_report.txt", "w") as f:
        league_models = {}  # {league: {market: model}} for summary tables
        prev_league = None
        for model_str in model_list:
            with open(pkg_resources.files(data) / f"models/{model_str}", "rb") as infile:
                model = pickle.load(infile)

            name = model_str.split("_")
            cv = model['cv']
            std = model.get('std',0)
            league = name[0]
            market = name[1].replace("-", " ").replace(".mdl", "")
            dist = model["distribution"]
            h_gate = model.get("hist_gate", 0)

            # Track models per league for summary table
            league_models.setdefault(league, {})[market] = model

            stat_cv.setdefault(league, {})
            stat_cv[league][market] = float(cv)

            stat_std.setdefault(league, {})
            stat_std[league][market] = float(std)
            
            # Save distribution selection
            stat_dist.setdefault(league, {})
            stat_dist[league][market] = dist

            # Save historical zero rate for all distributions
            stat_zi_local.setdefault(league, {})
            stat_zi_local[league][market] = h_gate

            f.write(f" {league} {market} ".center(60, "="))
            f.write("\n")
            f.write(f" Distribution Model: {dist}\n")
            f.write(f" Historical Zero Rate: {h_gate:.4f}\n")
            f.write(pd.DataFrame(model['stats'], index=[
                    ['No Filter', 'Filter']]).to_string(index=False))
            f.write("\n")

            if "diagnostics" in model:
                d = model["diagnostics"]
                sl = d["shape_label"]
                emp_shape = d.get('empirical_shape', 0.0)
                mod_shape = d.get('model_shape', 0.0)
                shape_ratio = mod_shape / max(emp_shape, 0.01) if not np.isnan(emp_shape) else float('nan')
                f.write(f" DIAG start_{sl}={d.get('start_shape', 0.0):.3f}"
                        f" model_{sl}={mod_shape:.3f}"
                        f" empirical_{sl}={emp_shape:.3f}"
                        f" shape_ratio={shape_ratio:.1f}x\n")
                f.write(f" DIAG start_mean={d.get('start_mean', 0.0):.2f}"
                        f" model_ev={d.get('model_ev', 0.0):.2f}"
                        f" mean_line={d.get('mean_line', 0.0):.2f}"
                        f" result_mean={d.get('result_mean', 0.0):.2f}\n")
                f.write(f" DIAG mean_ev_diff={d.get('ev_minus_line', 0.0):+.3f}"
                        f" median_ev_diff={d.get('median_ev_diff', 0.0):+.3f}"
                        f" frac_ev>line={d.get('frac_ev_gt_line', 0.0):.1%}\n")
                f.write(f" DIAG Over%|ev>line={d.get('over_pct_ev_gt', float('nan')):.3f}"
                        f" Over%|ev<line={d.get('over_pct_ev_lt', float('nan')):.3f}"
                        f" CF_Over%(emp_shape)={d.get('cf_over_pct', float('nan')):.3f}\n")
                f.write(f" DIAG shape_ceiling={d.get('shape_ceiling', 'N/A')}"
                        f" marginal_shape={d.get('marginal_shape', 'N/A')}"
                        f" dispersion_cal={d.get('dispersion_cal', 0.0):.3f}\n")

            if "params" in model:
                p = model["params"]
                f.write(f" HP rounds={p.get('opt_rounds', '?')}"
                        f" leaves={p.get('num_leaves', '?')}"
                        f" lr={p.get('learning_rate', 0):.4f}"
                        f" min_child={p.get('min_child_samples', '?')}"
                        f" L1={p.get('lambda_l1', 0):.2e}"
                        f" L2={p.get('lambda_l2', 0):.2e}\n")

            f.write("\n")

        # === PER-LEAGUE SUMMARY TABLES ===
        for league, markets in sorted(league_models.items()):
            f.write("\n" + "=" * 80 + "\n")
            f.write(f" {league} SUMMARY TABLE\n")
            f.write("=" * 80 + "\n")
            f.write(f"{'Market':<16} {'Dist':<8} {'Over%':>6} {'ShpR':>5}"
                    f" {'FracEV>':>7} {'MedDiff':>8}"
                    f" {'O%|EV>':>7} {'O%|EV<':>7} {'CF_O%':>6}\n")
            f.write("-" * 80 + "\n")
            for mkt, mdl in sorted(markets.items()):
                if "diagnostics" not in mdl:
                    continue
                d = mdl["diagnostics"]
                stats = mdl["stats"]
                dist_name = mdl.get("distribution", "?")[:6]
                over_pct_val = stats["Over%"][1]  # filtered
                emp_s = d.get('empirical_shape', 0.0)
                mod_s = d.get('model_shape', 0.0)
                sr = mod_s / max(emp_s, 0.01) if not np.isnan(emp_s) else float('nan')
                fev = d.get('frac_ev_gt_line', 0)
                med = d.get('median_ev_diff', 0)
                oeg = d.get('over_pct_ev_gt', float('nan'))
                oel = d.get('over_pct_ev_lt', float('nan'))
                cfo = d.get('cf_over_pct', float('nan'))
                sr_str = f"{sr:>4.1f}x" if not np.isnan(sr) else "  nan"
                f.write(f"{mkt:<16} {dist_name:<8} {over_pct_val:>6.3f} {sr_str}"
                        f" {fev:>7.1%} {med:>+8.3f}"
                        f" {oeg:>7.3f} {oel:>7.3f} {cfo:>6.3f}\n")
            f.write("\n")

    with open(pkg_resources.files(data) / "stat_cv.json", "w") as f:
        json.dump(stat_cv, f, indent=4)

    with open(pkg_resources.files(data) / "stat_std.json", "w") as f:
        json.dump(stat_std, f, indent=4)
    
    save_distribution_config(stat_dist)
    save_zi_config(stat_zi_local)


def see_features():
    """Analyze feature importance and correlations across all models using SHAP."""
    model_list = [f.name for f in (pkg_resources.files(data)/"models").iterdir() if ".mdl" in f.name]
    model_list.sort()
    feature_importances = []
    feature_correlations = []
    for model_str in tqdm(model_list, desc="Analyzing feature importances...", unit="market"):
        with open(pkg_resources.files(data) / f"models/{model_str}", "rb") as infile:
            filedict = pickle.load(infile)

        filepath = pkg_resources.files(
            data) / ("test_sets/" + model_str.replace(".mdl", ".csv"))
        M = pd.read_csv(filepath, index_col=0)

        y = M[['Result']]
        X = M.drop(columns=['Result', 'EV', 'P'])
        C = X.corrwith(M["Result"])
        
        # Drop distribution-specific parameters that aren't features
        dist = filedict["distribution"]
        if dist in ("Gamma", "ZAGamma"):
            X.drop(columns=["Alpha"], inplace=True, errors='ignore')
            C.drop(["Alpha"], inplace=True, errors='ignore')
        elif dist in ("NegBin", "ZINB"):
            X.drop(columns=["R", "NB_P"], inplace=True, errors='ignore')
            C.drop(["R", "NB_P"], inplace=True, errors='ignore')
        elif dist == "Mixture2G":
            X.drop(columns=["STD"], inplace=True, errors='ignore')
            C.drop(["STD"], inplace=True, errors='ignore')
        # Drop gate column for zero-inflated distributions
        if dist in ("ZINB", "ZAGamma"):
            X.drop(columns=["Gate"], inplace=True, errors='ignore')
            C.drop(["Gate"], inplace=True, errors='ignore')
        
        features = X.columns

        categories = ["Home", "Player position"]
        if "Player position" not in features:
            categories.remove("Player position")
        for c in categories:
            X[c] = X[c].astype('category')

        model = filedict['model']

        vals = np.zeros(len(X.columns))
        
        explainer = shap.TreeExplainer(model.booster)
        subvals = explainer.shap_values(X)
        
        # Aggregate |SHAP| across distribution outputs for multi-output models
        if isinstance(subvals, list):
            subvals = np.sum([np.abs(sv) for sv in subvals], axis=0)

        vals += np.mean(np.abs(subvals), axis=0)

        vals = vals/np.sum(vals)*100
        feature_importances.append(
            {k: v for k, v in list(zip(features, vals))})
        feature_correlations.append(C.to_dict())

    df = pd.DataFrame(feature_importances, index=[
                      market[:-4] for market in model_list]).fillna(0).infer_objects(copy=False).transpose()
    
    for league in ["NBA", "WNBA", "NFL", "NHL", "MLB"]:
        df[league + "_ALL"] = df[[col for col in df.columns if league in col]].mean(axis=1)

    df["ALL"] = df[[col for col in df.columns if "ALL" in col]].mean(axis=1)
    df.to_csv(pkg_resources.files(data) / "feature_importances.csv")
    pd.DataFrame(feature_correlations, index=[
                      market[:-4] for market in model_list]).T.to_csv(pkg_resources.files(data) / "feature_correlations.csv")

def filter_features():
    """Identify and filter low-importance features based on SHAP analysis."""
    shap_df = pd.read_csv(pkg_resources.files(data) / "feature_importances.csv", index_col=0)
    shap_df.drop(columns=[col for col in shap_df.columns if "ALL" in col], inplace=True)
    corr_df = pd.read_csv(pkg_resources.files(data) / "feature_correlations.csv", index_col=0)
    shap_df[shap_df < .3] = 0
    shap_df.fillna(0, inplace=True)
    corr_df = np.abs(corr_df)
    corr_df[corr_df < .1] = 0
    corr_df.fillna(0, inplace=True)

    leagues = list(set([col.split("_")[0] for col in shap_df.columns]))
    for league in leagues:
        feature_filter.setdefault(league, {})
        importances = shap_df.drop(index=feature_filter[league]["Common"]).rank(ascending=False, method="max")
        correlations = corr_df.drop(index=feature_filter[league]["Common"]).rank(ascending=False, method="max")
        
        markets = [col.split("_")[1] for col in shap_df.columns if col.split("_")[0] == league]
        for market in markets:
            if market not in feature_filter[league]:
                stats = []
                stats.extend([stat.replace(" short", "").replace(" growth", "") for stat in importances.loc[importances["_".join([league, market])] <= 30].index])
                stats.extend([stat.replace(" short", "").replace(" growth", "") for stat in correlations.loc[correlations["_".join([league, market])] <= 20].index])
                stats = list(set(stats))
                stats.sort()

                feature_filter[league][market.replace("-", " ")] = stats

    with open(pkg_resources.files(data) / "feature_filter.json", "w") as outfile:
        json.dump(feature_filter, outfile, indent=4)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def fit_book_weights(league, market):
    """Fit optimal weights for multiple sportsbooks using historical accuracy."""
    global book_weights
    warnings.simplefilter('ignore', UserWarning)
    print(f"Fitting Book Weights - {league}, {market}")
    df = archive.to_pandas(league, market)
    df = df[[col for col in df.columns if col != "pinnacle"]]
    if len([col for col in df.columns if col not in ["Line", "Result", "Over"]]) == 0:
        return {}
    stat_data = stat_structs[league]
    cv = stat_cv[league].get(market, 1)
    stat_dist = load_distribution_config()
    dist = stat_dist[league].get(market, "Poisson")
    
    if market == "Moneyline":
        log = stat_data.teamlog[[stat_data.log_strings["team"], stat_data.log_strings["date"], stat_data.log_strings["win"]]]
        log[stat_data.log_strings["date"]] = log[stat_data.log_strings["date"]].str[:10]
        df["Result"] = log.drop_duplicates([stat_data.log_strings["date"], stat_data.log_strings["team"]]).set_index([stat_data.log_strings["date"], stat_data.log_strings["team"]])[stat_data.log_strings["win"]]
        df.dropna(subset="Result", inplace=True)
        result = (df["Result"] == "W").astype(int)
        test_df = df.drop(columns='Result')

    elif market == "Totals":
        log = stat_data.teamlog[[stat_data.log_strings["team"], stat_data.log_strings["date"], stat_data.log_strings["score"]]]
        log[stat_data.log_strings["date"]] = log[stat_data.log_strings["date"]].str[:10]
        df["Result"] = log.drop_duplicates([stat_data.log_strings["date"], stat_data.log_strings["team"]]).set_index([stat_data.log_strings["date"], stat_data.log_strings["team"]])[stat_data.log_strings["score"]]
        df.dropna(subset="Result", inplace=True)
        result = df["Result"].astype(float)
        test_df = df.drop(columns='Result')
    
    elif market == "1st 1 innings":
        log = stat_data.gamelog.loc[stat_data.gamelog["starting pitcher"], ["opponent", stat_data.log_strings["date"], "1st inning runs allowed"]]
        log[stat_data.log_strings["date"]] = log[stat_data.log_strings["date"]].str[:10]
        df["Result"] = log.drop_duplicates([stat_data.log_strings["date"], "opponent"]).set_index([stat_data.log_strings["date"], "opponent"])["1st inning runs allowed"]
        df.dropna(subset="Result", inplace=True)
        result = df["Result"].astype(float)
        test_df = df.drop(columns='Result')
    
    else:
        log = stat_data.gamelog[[stat_data.log_strings["player"], stat_data.log_strings["date"], market]]
        log[stat_data.log_strings["date"]] = log[stat_data.log_strings["date"]].str[:10]
        df["Result"] = log.drop_duplicates([stat_data.log_strings["date"], stat_data.log_strings["player"]]).set_index([stat_data.log_strings["date"], stat_data.log_strings["player"]])[market]
        df.dropna(subset="Result", inplace=True)
        result = df["Result"].astype(float)
        test_df = df.drop(columns='Result')
        

    if market == "Moneyline":
        def objective(w, x, y):
            prob = np.exp(np.ma.average(np.ma.MaskedArray(np.log(x), mask=np.isnan(x)), weights=w, axis=1))
            return log_loss(y[~np.ma.getmask(prob)], np.ma.getdata(prob)[~np.ma.getmask(prob)])

    elif dist in ["NegBin", "ZINB", "Poisson"]:
        def objective(w, x, y):
            proj = np.array(np.exp(np.ma.average(np.ma.MaskedArray(np.log(x), mask=np.isnan(x)), weights=w, axis=1)))
            return -np.mean(poisson.logpmf(y.astype(int), proj))
        
    else:
        def objective(w, x, y):
            s = np.ma.MaskedArray(x*cv, mask=np.isnan(x))
            std = np.sqrt(1/np.ma.average(np.power(s,-2), weights=w, axis=1))
            proj = np.array(np.ma.average(np.ma.MaskedArray(x*np.power(s,-2), mask=np.isnan(x)), weights=w, axis=1)*std*std)
            return -np.mean(norm.logpdf(y, proj, std))
        
    if "Line" in test_df.columns:
        test_df.drop(columns=["Line"], inplace=True)

    x = test_df.loc[~test_df.isna().all(axis=1)].to_numpy()
    x[x<0] = np.nan
    y = result.loc[~test_df.isna().all(axis=1)].to_numpy()
    if len(x) > 9:
        prev_weights = book_weights.get(league, {}).get(market, {})
        guess = {}
        for book in test_df.columns:
            guess.update({book: prev_weights.get(book,1)})
            
        guess = list(guess.values())
        guess = np.clip(guess/np.sum(guess),0.005,.75)
        res = minimize(objective, guess, args=(x, y), bounds=[(0.001, 1)]*len(test_df.columns), tol=1e-8, method='TNC')
    
        return {k:res.x[i] for i, k in enumerate(test_df.columns)}
    else:
        return {}
    
def fit_model_weight(model_ev, odds_ev, result, dist, model_alpha=None, model_r=None, cv=None,
                     gate_model=None, gate_book=None):
    """
    Optimize the single blend weight between model predictions and
    bookmaker lines by maximizing log-likelihood on validation data.

    Returns a single float w in [0.05, 0.9].

    - NegBin: uses the logarithmic opinion pool — geometric mean of
      both μ and r with a single weight w.  The book's r is 1/cv.
    - Gamma: precision-weighted blend using model alpha and book
      alpha (1/cv²).

    When gate_model/gate_book are provided, the likelihood accounts for
    zero-inflation: P(y) = gate*I(y=0) + (1-gate)*base_pdf(y).
    """
    result = np.asarray(result, dtype=float)
    model_ev = np.asarray(model_ev, dtype=float)
    odds_ev = np.asarray(odds_ev, dtype=float)
    has_gate = gate_model is not None and gate_book is not None
    
    if dist == "NegBin":
        model_r_arr = np.asarray(model_r, dtype=float)
        result_int = result.astype(int)
        
        def objective(w):
            r_blend, p_blend, g_blend = fused_loc(w, model_ev, odds_ev, cv, "NegBin",
                                         r=model_r_arr, gate_model=gate_model, gate_book=gate_book)
            base_logpmf = nbinom.logpmf(result_int, r_blend, p_blend)
            if has_gate:
                # ZI likelihood: P(y=0) = g + (1-g)*NB(0), P(y>0) = (1-g)*NB(y)
                loglik = np.where(
                    result_int == 0,
                    np.log(g_blend + (1 - g_blend) * np.exp(base_logpmf)),
                    np.log(1 - g_blend) + base_logpmf
                )
                return -np.mean(loglik)
            return -np.mean(base_logpmf)

        res = minimize(objective, 0.5, bounds=[(0.05, 0.9)],
                       tol=1e-8, method='TNC')
        return res.x[0]
    else:
        model_alpha_arr = np.asarray(model_alpha, dtype=float)
        
        def objective(w):
            alpha, beta, g_blend = fused_loc(w, model_ev, odds_ev, cv, "Gamma",
                                   alpha=model_alpha_arr, gate_model=gate_model, gate_book=gate_book)
            base_logpdf = gamma.logpdf(result, alpha, scale=1 / beta)
            if has_gate:
                # ZA likelihood: P(y=0) = g, P(y>0) = (1-g)*Gamma(y)
                loglik = np.where(
                    result == 0,
                    np.log(np.clip(g_blend, 1e-12, None)),
                    np.log(1 - g_blend) + base_logpdf
                )
                return -np.mean(loglik)
            return -np.mean(base_logpdf)

        res = minimize(objective, .5, bounds=[(0.05, 0.9)], tol=1e-8, method='TNC')
        return res.x[0]

def select_distribution(player_stats):
    import warnings
    warnings.filterwarnings('ignore', 'overflow', RuntimeWarning)

    # Check data type properties (market-level, not per-player)
    sample = player_stats.first()
    is_integer = all(v == int(v) for v in sample)
    if is_integer:
        uniques = player_stats.apply(lambda x: x.unique().tolist()).explode().drop_duplicates().sort_values()
        step = uniques.diff().dropna().min() if len(uniques) > 1 else 1
    else:
        step = 0

    if not is_integer or step != 1:
        # Non-integer or non-unit-step data → Gamma family
        dist = "Gamma"
    else:
        # Per-player resolution = step / mean_of_nonzeros
        # High resolution (>0.2) → counting few discrete events → NegBin
        # Low resolution (≤0.2) → accumulating many events → quasi-continuous → Gamma
        def _player_resolution(x):
            nz = x[x > 0]
            return step / nz.mean() if len(nz) > 0 else np.nan
        resolutions = player_stats.apply(_player_resolution).dropna()
        resolution = resolutions.median()
        dist = "NegBin" if resolution > 0.2 else "Gamma"
        print(f"  Resolution: {resolution:.4f} ({'NegBin' if resolution > 0.2 else 'Gamma'})")

    # Check for zero inflation and compute historical gate
    observed_zeros = player_stats.agg(lambda x: x.eq(0).mean())

    if dist in ["NegBin", "ZINB"]:
        def _nb_mom(x):
            mu, var = x.mean(), x.var()
            if var <= mu:
                var = mu + 1e-6
            p = np.clip(mu / var, 1e-3, 1 - 1e-3)
            n = np.clip(mu * p / (1 - p), 0.1, 50)
            return (n, p)
        nb_fit = player_stats.apply(_nb_mom)
        base_zero_prob = nb_fit.apply(lambda row: nbinom.pmf(0, row[0], row[1]))
        p_zero = float(((observed_zeros - base_zero_prob) / (1 - base_zero_prob)).clip(0, 1).mean())
        if p_zero > 0.1:
            dist = "ZINB"
    else:
        gam_fit = player_stats.apply(lambda x: fit(gamma, x[x>0].astype(float), {"a":(0, 50), "scale":(0, 500)}).params)
        base_zero_prob = gam_fit.apply(lambda row: gamma.cdf(0.99, row[0], scale=row[2]))
        p_zero = float(((observed_zeros - base_zero_prob) / (1 - base_zero_prob)).clip(0, 1).mean())
        if p_zero > 0.05:
            dist = "ZAGamma"

    print(f"  Data type: {'integer (step={})'.format(int(step)) if is_integer else 'continuous'}")
    print(f"  Zero inflation - {p_zero:.4f}")
    print(f"  Selected: {dist}")

    return dist, p_zero

def count_training_rows(stat_data, market, start_date):
    """
    Estimate the number of training rows get_training_matrix would produce for
    a given market and start_date, using the archive and gamelog directly.

    Counts:
      - Archived rows: (date, player) entries in the archive with a real line
      - Non-archived rows: players above the 25th-percentile usage cutoff on
        each game day, excluding any that are already counted as archived

    This is an upper-bound estimate; trim_matrix may remove some rows afterward.
    """
    gamelog = stat_data.gamelog.drop_duplicates(
        subset=[stat_data.log_strings["game"], stat_data.log_strings["player"]], keep="last"
    ).copy()
    gamelog[stat_data.log_strings["date"]] = pd.to_datetime(gamelog[stat_data.log_strings["date"]]).dt.date
    gamelog = gamelog.loc[
        (gamelog[stat_data.log_strings["date"]] > start_date) &
        (gamelog[stat_data.log_strings["date"]] < datetime.today().date())
    ]
    if gamelog.empty:
        return 0

    usage_cutoff = gamelog[stat_data.usage_stat].quantile(0.25)
    league_archive = archive.archive.get(stat_data.league, {}).get(market, {})

    total = 0
    for game_date, players in gamelog.groupby(stat_data.log_strings["date"]):
        date_str = game_date.strftime("%Y-%m-%d")
        archived_players = set(league_archive.get(date_str, {}).keys())

        # Count unique players who actually played that day
        played = set(players[stat_data.log_strings["player"]].unique())
        archived_count = len(archived_players & played)

        # Count non-archived unique players above usage cutoff
        non_archived = players.drop_duplicates(
            subset=stat_data.log_strings["player"]
        ).loc[
            lambda df: ~df[stat_data.log_strings["player"]].isin(archived_players)
        ]
        non_archived_count = (non_archived[stat_data.usage_stat] > usage_cutoff).sum()

        total += archived_count + non_archived_count

    return total


def _histogram_weights(values, reference_values, min_reference=20):
    """Compute removal probabilities via histogram matching.

    Returns probability array aligned to values. Rows whose bin density
    exceeds the reference density are more likely to be removed.
    Falls back to uniform weights when reference data is insufficient.
    """
    if len(values) == 0:
        return np.array([])

    counts, bins = np.histogram(values)
    counts = counts / len(values)

    if len(reference_values) >= min_reference:
        ref_counts, _ = np.histogram(reference_values, bins)
        ref_counts = ref_counts / len(reference_values)
    else:
        ref_counts = np.zeros_like(counts)

    diff = np.clip(counts - ref_counts, 1e-8, None)
    p = np.zeros(len(values))
    for j, a in enumerate(bins[:-1]):
        p[values >= a] = diff[j]
    return p / np.sum(p)


def trim_matrix(M, min_rows=7500):
    """Remove data quality issues and prepare matrix for modeling.

    Trims outlier results, clips lines to a realistic range, balances
    the line distribution across positions, and balances over/under
    proportions.  All removal steps respect min_rows so that sparse-
    archive markets are not destroyed.
    """
    warnings.simplefilter('ignore', UserWarning)

    # --- 1. Fix DaysIntoSeason wrapping ---
    while any(M["DaysIntoSeason"] < 0) or any(M["DaysIntoSeason"] > 300):
        M.loc[M["DaysIntoSeason"] < 0, "DaysIntoSeason"] = M.loc[M["DaysIntoSeason"]
                                                                    < 0, "DaysIntoSeason"] - M["DaysIntoSeason"].min()
        M.loc[M["DaysIntoSeason"] > 300, "DaysIntoSeason"] = M.loc[M["DaysIntoSeason"]
                                                                    > 300, "DaysIntoSeason"] - M.loc[M["DaysIntoSeason"]
                                                                                                    > 300, "DaysIntoSeason"].min()

    # --- 2. Remove result outliers (archived rows always kept) ---
    M = M.loc[((M["Result"] >= M["Result"].quantile(.05)) & (
        M["Result"] <= M["Result"].quantile(.95))) | (M["Archived"] == 1)]

    # --- 3. Clip lines to a realistic range ---
    # Use archived range when coverage is good; otherwise fall back to
    # full-data percentiles so sparse-archive markets keep their natural
    # line distribution.
    archived_mask = M["Archived"] == 1
    n_archived = archived_mask.sum()
    if n_archived >= 50 and n_archived / len(M) > 0.10:
        line_floor = M.loc[archived_mask, "Line"].min()
        line_ceil = M.loc[archived_mask, "Line"].max()
    else:
        line_floor = M["Line"].quantile(0.05)
        line_ceil = M["Line"].quantile(0.95)
    M["Line"] = M["Line"].clip(line_floor, line_ceil)

    # --- 4. Balance line distribution ---
    # For each position (or overall), reduce the over-represented side of
    # the line distribution using histogram-matched removal weights.
    # Per-position archived median is used when >= 20 archived rows exist
    # for that position; otherwise the overall median is used.
    overall_target = (M.loc[archived_mask, "Line"].median()
                      if n_archived >= 5 else M["Line"].mean())

    def _balance_lines(M, pos_mask):
        budget = max(len(M) - min_rows, 0)
        if budget == 0:
            return M

        pos_archived = archived_mask & pos_mask
        n_pos_archived = pos_archived.sum()
        target = (M.loc[pos_archived, "Line"].median()
                  if n_pos_archived >= 20 else overall_target)

        non_arch = ~archived_mask & pos_mask
        less = M.loc[non_arch & (M["Line"] < target), "Line"]
        more = M.loc[non_arch & (M["Line"] > target), "Line"]

        n = min(abs(len(less) - len(more)), budget)
        if n == 0:
            return M

        if len(less) > len(more):
            ref = M.loc[pos_archived & (M["Line"] < target), "Line"]
            p = _histogram_weights(less.values, ref.values, min_reference=20)
            chopping_block = less.index
        else:
            ref = M.loc[pos_archived & (M["Line"] > target), "Line"]
            p = _histogram_weights(more.values, ref.values, min_reference=20)
            chopping_block = more.index

        n = min(n, len(chopping_block))
        cut = np.random.choice(chopping_block, n, replace=False, p=p)
        M.drop(cut, inplace=True)
        return M

    if "Player position" in M.columns:
        for i in M["Player position"].unique():
            M = _balance_lines(M, M["Player position"] == i)
    else:
        M = _balance_lines(M, pd.Series(True, index=M.index))

    # --- 5. Balance over/under proportions ---
    if n_archived < 10:
        return M.sort_values("Date")

    pushes = M.loc[M["Result"] == M["Line"]]
    push_rate = pushes["Archived"].sum() / M["Archived"].sum()
    M = M.loc[M["Result"] != M["Line"]]

    archived_no_push = M["Archived"] == 1
    if archived_no_push.sum() >= 20:
        target = (M.loc[archived_no_push, "Result"] >
                  M.loc[archived_no_push, "Line"]).mean()
    else:
        target = (M["Result"] > M["Line"]).mean()

    balance = (M["Result"] > M["Line"]).mean()
    budget = max(len(M) - min_rows, 0)
    n = min(2 * int(np.abs(target - balance) * len(M)), budget)

    if n > 0:
        if balance < target:
            chopping_block = M.loc[(M["Archived"] != 1) & (
                M["Result"] < M["Line"])].index
            p = (1/M.loc[chopping_block,
                            "MeanYr"].clip(0.1)).to_numpy()
            p = p/np.sum(p)
        else:
            chopping_block = M.loc[(M["Archived"] != 1) & (
                M["Result"] > M["Line"])].index
            p = (M.loc[chopping_block, "MeanYr"].clip(
                0.1)).to_numpy()
            p = p/np.sum(p)

        n = min(n, len(chopping_block))
        cut = np.random.choice(chopping_block, n, replace=False, p=p)
        M.drop(cut, inplace=True)

    # --- 6. Re-insert pushes at the correct proportion ---
    n = int(push_rate * len(M)) - pushes["Archived"].sum()
    chopping_block = pushes.loc[pushes["Archived"] == 0].index
    n = np.clip(n, None, len(chopping_block))
    if n > 0:
        cut = np.random.choice(chopping_block, n, replace=False)
        pushes.drop(cut, inplace=True)

    M = pd.concat([M, pushes]).sort_values("Date")

    return M

def correlate(league, force=False):
    """Calculate feature correlations with outcomes for feature engineering."""
    print(f"Correlating {league}...")
    tracked_stats = { 
        "NFL": {
            "QB": [
                "passing yards",
                "rushing yards",
                "qb yards",
                "fantasy points prizepicks",
                "fantasy points underdog",
                "passing tds",
                "rushing tds",
                "qb tds",
                "completions",
                "carries",
                "interceptions",
                "attempts",
                "sacks taken",
                "longest completion",
                "longest rush",
                "passing first downs",
                "first downs",
                "fumbles lost",
                "completion percentage"
            ],
            "RB": [
                "rushing yards",
                "receiving yards",
                "yards",
                "fantasy points prizepicks",
                "fantasy points underdog",
                "tds",
                "rushing tds",
                "receiving tds",
                "carries",
                "receptions",
                "targets",
                "longest rush",
                "longest reception",
                "first downs",
                "fumbles lost"
            ],
            "WR": [
                "receiving yards",
                "yards",
                "fantasy points prizepicks",
                "fantasy points underdog",
                "tds",
                "receiving tds",
                "receptions",
                "targets",
                "longest reception",
                "first downs",
                "fumbles lost"
            ],
            "TE": [
                "receiving yards",
                "yards",
                "fantasy points prizepicks",
                "fantasy points underdog",
                "tds",
                "receiving tds",
                "receptions",
                "targets",
                "longest reception",
                "first downs",
                "fumbles lost"
            ],
        },
        "NHL": {
            "G": [
                "saves",
                "goalsAgainst",
                "goalie fantasy points underdog"
            ],
            "C": [
                "points",
                "shots",
                "sogBS",
                "fantasy points prizepicks",
                "skater fantasy points underdog",
                "blocked",
                "hits",
                "goals",
                "assists",
                "faceOffWins",
                "timeOnIce",
            ],
            "W": [
                "points",
                "shots",
                "sogBS",
                "fantasy points prizepicks",
                "skater fantasy points underdog",
                "blocked",
                "hits",
                "goals",
                "assists",
                "faceOffWins",
                "timeOnIce",
            ],
            "D": [
                "points",
                "shots",
                "sogBS",
                "fantasy points prizepicks",
                "skater fantasy points underdog",
                "blocked",
                "hits",
                "goals",
                "assists",
                "faceOffWins",
                "timeOnIce",
            ]
        },
        "NBA": {
            "C": [
                "PTS",
                "REB",
                "AST",
                "PRA",
                "PR",
                "RA",
                "PA",
                "FG3M",
                "fantasy points prizepicks",
                "fantasy points underdog",
                "TOV",
                "BLK",
                "STL",
                "BLST",
                "FG3A",
                "FTM",
                "FGM",
                "FGA",
                "OREB",
                "DREB",
                "PF",
                "MIN"
            ],
            "P": [
                "PTS",
                "REB",
                "AST",
                "PRA",
                "PR",
                "RA",
                "PA",
                "FG3M",
                "fantasy points prizepicks",
                "fantasy points underdog",
                "TOV",
                "BLK",
                "STL",
                "BLST",
                "FG3A",
                "FTM",
                "FGM",
                "FGA",
                "OREB",
                "DREB",
                "PF",
                "MIN"
            ],
            "B": [
                "PTS",
                "REB",
                "AST",
                "PRA",
                "PR",
                "RA",
                "PA",
                "FG3M",
                "fantasy points prizepicks",
                "fantasy points underdog",
                "TOV",
                "BLK",
                "STL",
                "BLST",
                "FG3A",
                "FTM",
                "FGM",
                "FGA",
                "OREB",
                "DREB",
                "PF",
                "MIN"
            ],
            "F": [
                "PTS",
                "REB",
                "AST",
                "PRA",
                "PR",
                "RA",
                "PA",
                "FG3M",
                "fantasy points prizepicks",
                "fantasy points underdog",
                "TOV",
                "BLK",
                "STL",
                "BLST",
                "FG3A",
                "FTM",
                "FGM",
                "FGA",
                "OREB",
                "DREB",
                "PF",
                "MIN"
            ],
            "W": [
                "PTS",
                "REB",
                "AST",
                "PRA",
                "PR",
                "RA",
                "PA",
                "FG3M",
                "fantasy points prizepicks",
                "fantasy points underdog",
                "TOV",
                "BLK",
                "STL",
                "BLST",
                "FG3A",
                "FTM",
                "FGM",
                "FGA",
                "OREB",
                "DREB",
                "PF",
                "MIN"
            ]
        },
        "WNBA": {
            "G": [
                "PTS",
                "REB",
                "AST",
                "PRA",
                "PR",
                "RA",
                "PA",
                "FG3M",
                "fantasy points prizepicks",
                "fantasy points underdog",
                "TOV",
                "BLK",
                "STL",
                "BLST",
                "FG3A",
                "FTM",
                "FGM",
                "FGA",
                "OREB",
                "DREB",
                "PF",
                "MIN"
            ],
            "F": [
                "PTS",
                "REB",
                "AST",
                "PRA",
                "PR",
                "RA",
                "PA",
                "FG3M",
                "fantasy points prizepicks",
                "fantasy points underdog",
                "TOV",
                "BLK",
                "STL",
                "BLST",
                "FG3A",
                "FTM",
                "FGM",
                "FGA",
                "OREB",
                "DREB",
                "PF",
                "MIN"
            ],
            "C": [
                "PTS",
                "REB",
                "AST",
                "PRA",
                "PR",
                "RA",
                "PA",
                "FG3M",
                "fantasy points prizepicks",
                "fantasy points underdog",
                "TOV",
                "BLK",
                "STL",
                "BLST",
                "FG3A",
                "FTM",
                "FGM",
                "FGA",
                "OREB",
                "DREB",
                "PF",
                "MIN"
            ],
        },
        "MLB": {
            "P": [
                "pitcher strikeouts",
                "pitching outs",
                "pitches thrown",
                "hits allowed",
                "runs allowed",
                "1st inning runs allowed",
                "1st inning hits allowed",
                "pitcher fantasy score",
                "pitcher fantasy points underdog",
                "walks allowed"
            ],
            "B": [
                "hitter fantasy score",
                "hitter fantasy points underdog",
                "hits+runs+rbi",
                "total bases",
                "walks",
                "stolen bases",
                "hits",
                "runs",
                "rbi",
                "batter strikeouts",
                "singles",
                "doubles",
                "triples",
                "home runs"
            ]
        }
    }

    stats = tracked_stats[league]
    log = stat_structs[league]
    log_str = log.log_strings

    filepath = pkg_resources.files(data) / f"training_data/{league}_corr.csv"
    if os.path.isfile(filepath) and not force:
        matrix = pd.read_csv(filepath, index_col=0)
        matrix.DATE = pd.to_datetime(matrix.DATE, format="mixed")
        latest_date = matrix.DATE.max()
        matrix = matrix.loc[matrix.DATE >= datetime.today()-timedelta(days=300)]
    else:
        matrix = pd.DataFrame()
        latest_date = datetime.today()-timedelta(days=300)

    games = log.gamelog[log_str["game"]].unique()
    game_data = []

    for gameId in tqdm(games):
        game_df = log.gamelog.loc[log.gamelog[log_str["game"]] == gameId]
        gameDate = datetime.fromisoformat(game_df.iloc[0][log_str["date"]])
        if gameDate < latest_date or len(game_df[log_str["team"]].unique()) != 2:
            continue
        [home_team, away_team] = tuple(game_df.sort_values(log_str["home"], ascending=False)[log_str["team"]].unique())

        if league == "MLB":
            bat_df = game_df.loc[game_df['starting batter']]
            bat_df.position = "B" + bat_df.battingOrder.astype(str)
            bat_df.index = bat_df.position
            pitch_df = game_df.loc[game_df['starting pitcher']]
            pitch_df.position = "P"
            pitch_df.index = pitch_df.position
            game_df = pd.concat([bat_df, pitch_df])
        else:
            log.profile_market(log_str["usage"], date=gameDate)
            usage = pd.DataFrame(
                log.playerProfile[[f"{log_str.get('usage')} short", f"{log_str.get('usage_sec')} short"]])
            usage.reset_index(inplace=True)
            game_df = game_df.merge(usage, how="left")
            game_df = game_df.loc[game_df[log_str["position"]].apply(lambda x: isinstance(x, str))]
            game_df = game_df.fillna(0).infer_objects(copy=False)
            ranks = game_df.sort_values(f"{log_str.get('usage_sec')} short", ascending=False).groupby(
                [log_str["team"], log_str["position"]]).rank(ascending=False, method='first')[f"{log_str.get('usage')} short"].astype(int)
            game_df[log_str["position"]] = game_df[log_str["position"]] + \
                ranks.astype(str)
            game_df.index = game_df[log_str["position"]]

        homeStats = {}
        awayStats = {}
        for position in stats.keys():
            homeStats.update(game_df.loc[(game_df[log_str["team"]] == home_team) & game_df[log_str["position"]].str.contains(
                position), stats[position]].to_dict('index'))
            awayStats.update(game_df.loc[(game_df[log_str["team"]] == away_team) & game_df[log_str["position"]].str.contains(
                position), stats[position]].to_dict('index'))

        game_data.append({"TEAM": home_team} | {"DATE": gameDate.date()} |
            homeStats | {"_OPP_" + k: v for k, v in awayStats.items()})
        game_data.append({"TEAM": away_team} | {"DATE": gameDate.date()} |
            awayStats | {"_OPP_" + k: v for k, v in homeStats.items()})

    matrix = pd.concat([matrix, pd.json_normalize(game_data)], ignore_index=True)
    matrix.to_csv(filepath)

    big_c = {}
    matrix.drop(columns="DATE", inplace=True)
    matrix.fillna(0, inplace=True)
    for team in matrix.TEAM.unique():
        team_matrix = matrix.loc[matrix.TEAM == team].drop(columns="TEAM")
        team_matrix = team_matrix.loc[:,((team_matrix==0).mean() < .5)]
        team_matrix = team_matrix.reindex(sorted(team_matrix.columns), axis=1)
        c_spearman = team_matrix.corr(method='spearman', min_periods=int(len(team_matrix)*.75)).unstack()
        c = 2 * np.sin(np.pi / 6 * c_spearman)
        c = c.reindex(c.abs().sort_values(ascending=False).index).dropna()
        c = c.loc[c.abs()>0.05]
        big_c.update({team: c})

    pd.concat(big_c).to_csv((pkg_resources.files(data) / f"{league}_corr.csv"))

if __name__ == "__main__":
    warnings.simplefilter('ignore', UserWarning)
    meditate()
    see_features()
    filter_features()
