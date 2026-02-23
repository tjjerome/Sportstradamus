from sportstradamus.stats import StatsMLB, StatsNBA, StatsNHL, StatsNFL, StatsWNBA
from sportstradamus.helpers import get_ev, get_odds, stat_cv, Archive, book_weights, feature_filter, set_model_start_values
import pickle
import importlib.resources as pkg_resources
from sportstradamus import data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from scipy.optimize import minimize
from scipy.stats import poisson, skellam, norm
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
from lightgbmlss.distributions.Gaussian import Gaussian
from lightgbmlss.distributions.Poisson import Poisson
from lightgbmlss.distributions.Mixture import Mixture
from lightgbmlss.distributions.distribution_utils import DistributionClass
import shap
import json
import warnings
pd.options.mode.chained_assignment = None
pd.set_option('future.no_silent_downcasting', True)
np.seterr(divide='ignore', invalid='ignore')

# ============================================================================
# DISTRIBUTION CONFIGURATION
# ============================================================================
# Distribution parameters
dist_params_default = {
    "stabilization": "None",
    "response_fn": "softplus",
    "loss_fn": "nll"
}

# ============================================================================
# LOAD SPORTS DATA
# ============================================================================
# Load and initialize active sports leagues based on season dates

sports = []
nba = StatsNBA()
nba.load()
if datetime.today().date() > (nba.season_start - timedelta(days=7)):
    sports.append("NBA")
# mlb = StatsMLB()
# mlb.load()
# if datetime.today().date() > (mlb.season_start - timedelta(days=7)):
#     sports.append("MLB")
# # nhl = StatsNHL()
# nhl.load()
# if datetime.today().date() > (nhl.season_start - timedelta(days=7)):
#     sports.append("NHL")
nfl = StatsNFL()
nfl.load()
if datetime.today().date() > (nfl.season_start - timedelta(days=7)):
    sports.append("NFL")
wnba = StatsWNBA()
wnba.load()
if datetime.today().date() > (wnba.season_start - timedelta(days=7)):
    sports.append("WNBA")


stat_structs = {}
if "NBA" in sports:
    nba.update()
    stat_structs.update({"NBA": nba})
# if "MLB" in sports:
#     mlb.update()
#     stat_structs.update({"MLB": mlb})
# if "NHL" in sports:
#     nhl.update()
#     stat_structs.update({"NHL": nhl})
if "NFL" in sports:
    nfl.update()
    stat_structs.update({"NFL": nfl})
if "WNBA" in sports:
    wnba.update()
    stat_structs.update({"WNBA": wnba})

archive = Archive()

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

def select_best_distribution(y_data):
    """
    Select the best distribution (Poisson, Gaussian, or Mixture2G) based on NLL.
    
    Uses scipy/sklearn for comparison, returns LightGBMLSS distribution object.
    
    Parameters:
    -----------
    y_data : np.ndarray
        Training target values
    
    Returns:
    --------
    dist_name : str
        Distribution name ("Poisson", "Gaussian", or "Mixture2G")
    dist_obj : LightGBMLSS distribution object
        Distribution object for training with LightGBMLSS
    nll_scores : dict
        NLL scores for each distribution (for logging)
    """
    nll_scores = {}
    
    # === POISSON ===
    # Estimate rate parameter via MLE (sample mean for Poisson)
    rate = np.mean(y_data)
    nll_poisson = -np.mean(poisson.logpmf(y_data.astype(int), rate))
    nll_scores["Poisson"] = nll_poisson
    
    # === GAUSSIAN ===
    # Estimate mean and std via MLE
    mu = np.mean(y_data)
    std = np.std(y_data, ddof=1)
    nll_gaussian = -np.mean(norm.logpdf(y_data, mu, std))
    nll_scores["Gaussian"] = nll_gaussian
    
    # === MIXTURE OF 2 GAUSSIANS ===
    # Use sklearn GaussianMixture for fitting
    try:
        mixture_model = GaussianMixture(n_components=2, random_state=42, n_init=10)
        mixture_model.fit(y_data.reshape(-1, 1))
        
        # Calculate NLL: -1/n * sum(log(likelihood))
        nll_mixture = -np.mean(mixture_model.score_samples(y_data.reshape(-1, 1)))
        nll_scores["Mixture2G"] = nll_mixture
    except Exception as e:
        print(f"Warning: Failed to fit Mixture2G: {e}. Defaulting to Gaussian.")
        nll_scores["Mixture2G"] = np.inf
    
    # === SELECT BEST ===
    best_dist = min(nll_scores, key=nll_scores.get)
    
    # Map to LightGBMLSS distribution objects
    if best_dist == "Poisson":
        dist_obj = Poisson(**dist_params_default)
    elif best_dist == "Gaussian":
        dist_obj = Gaussian(**dist_params_default)
    else:  # Mixture2G
        dist_obj = Mixture(Gaussian(**dist_params_default), M=2)
    
    return best_dist, dist_obj, nll_scores


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

@click.command()
@click.option("--force/--no-force", default=True, help="Force update of all models")
@click.option("--league", type=click.Choice(["All", "NFL", "NBA", "MLB", "NHL", "WNBA"]), default="All",
              help="Select league to train on")
def meditate(force, league):
    global book_weights

    np.random.seed(69)

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
        # "MLB": [
        #     "plateAppearances",
        #     "pitches thrown",
        #     "pitching outs",
        #     "pitcher strikeouts",
        #     "hits allowed",
        #     "runs allowed",
        #     "walks allowed",
        #     # "1st inning runs allowed",
        #     # "1st inning hits allowed",
        #     "hitter fantasy score",
        #     "pitcher fantasy score",
        #     "hitter fantasy points underdog",
        #     "pitcher fantasy points underdog",
        #     "hits+runs+rbi",
        #     "total bases",
        #     "walks",
        #     "stolen bases",
        #     "hits",
        #     "runs",
        #     "rbi",
        #     "batter strikeouts",
        #     "singles",
        #     "doubles",
        #     "home runs"
        # ],
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
        # "NHL": [
        #     "timeOnIce",
        #     "shotsAgainst",
        #     "saves",
        #     "shots",
        #     "points",
        #     "goalsAgainst",
        #     "goalie fantasy points underdog",
        #     "skater fantasy points underdog",
        #     "blocked",
        #     "powerPlayPoints",
        #     "sogBS",
        #     # "fantasy points prizepicks",
        #     "hits",
        #     "goals",
        #     "assists",
        #     "faceOffWins",
        # ]
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

        for market in markets:
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

            print(f"Training {league} - {market}")
            cv = stat_cv[league].get(market, 1)
            filepath = pkg_resources.files(data) / (f"training_data/{filename}.csv")
            if os.path.isfile(filepath):
                M = pd.read_csv(filepath, index_col=0)
                cutoff_date = pd.to_datetime(M["Date"]).max().date()
                if league == "MLB":
                    start_date = (datetime.today()-timedelta(days=700)).date()
                elif league == "NFL":
                    start_date = (datetime.today()-timedelta(days=1200)).date()
                else:
                    start_date = (datetime.today()-timedelta(days=850)).date()
                M = M.loc[(pd.to_datetime(M.Date).dt.date <= cutoff_date) & (pd.to_datetime(M.Date).dt.date > start_date)]
            else:
                if league == "MLB":
                    cutoff_date = (datetime.today()-timedelta(days=700)).date()
                elif league == "NFL":
                    cutoff_date = (datetime.today()-timedelta(days=1200)).date()
                else:
                    cutoff_date = (datetime.today()-timedelta(days=850)).date()
                M = pd.DataFrame()

            new_M = stat_data.get_training_matrix(market, cutoff_date)

            if new_M.empty and not force and not need_model:
                continue

            M = pd.concat([M, new_M], ignore_index=True)
            M.Date = pd.to_datetime(M.Date, format='mixed')
            step = M["Result"].drop_duplicates().sort_values().diff().min()
            for i, row in M.loc[M.Odds.isna() | (M.Odds == 0)].iterrows():
                if np.isnan(row["EV"]) or row["EV"] <= 0:
                    M.loc[i, "Odds"] = 0.5
                    M.loc[i, "EV"] = M.loc[i, "Line"] if cv != 1 else get_ev(M.loc[i, "Line"], .5, cv)
                else:
                    M.loc[i, "Odds"] = 1-get_odds(row["Line"], row["EV"], cv=cv, step=step)

            M = trim_matrix(M)
            M.to_csv(filepath)

            y = M[['Result']]
            X = M[stat_data.get_stat_columns(market)]

            categories = ["Home", "Player position"]
            if "Player position" not in X.columns:
                categories.remove("Player position")
            for c in categories:
                X[c] = X[c].astype('category')

            categories = "name:"+",".join(categories)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            X_test, X_validation, y_test, y_validation = train_test_split(
                X_test, y_test, test_size=0.5, random_state=25
            )

            B_train = M.loc[X_train.index, ['Line', 'Odds', 'EV']]
            B_test = M.loc[X_test.index, ['Line', 'Odds', 'EV']]
            B_validation = M.loc[X_validation.index, ['Line', 'Odds', 'EV']]

            y_train_labels = np.ravel(y_train.to_numpy())

            # === DISTRIBUTION SELECTION ===
            # Choose between Poisson, Gaussian, or Mixture of 2 Gaussians
            # Load from stat_dist.json or auto-select via NLL comparison
            stat_dist = load_distribution_config()
            stat_dist.setdefault(league, {})
            
            if market in stat_dist[league] and not force:
                # Use previously selected distribution
                dist = stat_dist[league][market]
                if dist == "Poisson":
                    dist_obj = Poisson()
                elif dist == "Gaussian":
                    dist_obj = Gaussian()
                elif dist == "Mixture2G":
                    dist_obj = Mixture(Gaussian(), M=2)
                else:
                    dist = "Gaussian"
                    dist_obj = Gaussian()
            else:
                # Auto-select via NLL comparison
                dist, dist_obj, nll_scores = select_best_distribution(y_train_labels)
                
                # Save selected distribution for future runs
                stat_dist[league][market] = dist
                save_distribution_config(stat_dist)
                
                print(f"  Distribution scores - Poisson: {nll_scores.get('Poisson', np.inf):.4f}, Gaussian: {nll_scores.get('Gaussian', np.inf):.4f}, Mixture2G: {nll_scores.get('Mixture2G', np.inf):.4f}")
                print(f"  Selected: {dist}")
            
            
            opt_params = filedict.get("params")
            dtrain = lgb.Dataset(
                X_train, label=y_train_labels)
            
            # === MODEL TRAINING ===
            # All distributions (Gaussian, Poisson, Mixture2G) use LightGBMLSS training
            model = LightGBMLSS(dist_obj)
            set_model_start_values(model, dist, X_train)

            if opt_params is None or opt_params.get("opt_rounds") is None or force:
                params = {
                    "feature_pre_filter": ["none", [False]],
                    "num_threads": ["none", [8]],
                    "max_depth": ["int", {"low": 4, "high": 16, "log": False}],
                    "max_bin": ["none", [63]],
                    "hist_pool_size": ["none", [9*1024]],
                    "num_leaves": ["int", {"low": 23, "high": 256, "log": False}],
                    "lambda_l1": ["float", {"low": 1e-6, "high": 10, "log": True}],
                    "lambda_l2": ["float", {"low": 1e-6, "high": 10, "log": True}],
                    "min_child_samples": ["int", {"low": 30, "high": 100, "log": False}],
                    "min_child_weight": ["float", {"low": 1e-3, "high": .75*len(X_train)/1000, "log": True}],
                    "learning_rate": ["float", {"low": 0.001, "high": 0.1, "log": True}],
                    "feature_fraction": ["float", {"low": 0.4, "high": 1.0, "log": False}],
                    "bagging_fraction": ["float", {"low": 0.4, "high": 1.0, "log": False}],
                    "bagging_freq": ["none", [1]]
                }
                opt_params = model.hyper_opt(params,
                                            dtrain,
                                            num_boost_round=999,
                                            nfold=4,
                                            early_stopping_rounds=50,
                                            max_minutes=30,
                                            n_trials=100,
                                            silence=True,
                                            )

            model.train(opt_params,
                        dtrain,
                        num_boost_round=opt_params["opt_rounds"]
                        )

            # === PREDICTIONS AND PARAMETER EXTRACTION ===
            # Generate predictions on all datasets and extract distribution parameters
            prob_params_train = pd.DataFrame()
            prob_params_validation = pd.DataFrame()
            prob_params = pd.DataFrame()

            # All distributions (Gaussian, Poisson, Mixture2G) use standard LightGBM prediction
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
            cv = 1
            
            # Extract appropriate parameters based on distribution
            if dist == "Poisson":
                ev = prob_params["rate"].to_numpy()
                ev_train = prob_params_train["rate"].to_numpy()
                std_train = None
                ev_validation = prob_params_validation["rate"].to_numpy()
                std_validation = None
            elif dist == "Gaussian":
                ev = prob_params["loc"].clip(0.1).to_numpy()
                cv = y.Result.std()/y.Result.mean()
                ev_train = prob_params_train["loc"].to_numpy()
                std_train = prob_params_train["scale"]
                ev_validation = prob_params_validation["loc"].to_numpy()
                std_validation = prob_params_validation["scale"]
            elif dist == "Mixture2G":
                # Mixture of 2 Gaussians - each sample has feature-dependent mixture parameters
                # LightGBMLSS Mixture parameters: mix_prob_1, loc_1, scale_1, loc_2, scale_2
                mix_prob_1 = prob_params["mix_prob_1"].to_numpy()
                loc_1 = prob_params["loc_1"].to_numpy()
                scale_1 = prob_params["scale_1"].to_numpy()
                loc_2 = prob_params["loc_2"].to_numpy()
                scale_2 = prob_params["scale_2"].to_numpy()
                mix_prob_2 = 1.0 - mix_prob_1
                
                # Expected value is weighted average of means (per sample)
                ev = mix_prob_1 * loc_1 + mix_prob_2 * loc_2
                
                # Standard deviation from mixture variance (per sample)
                var_mixture = (mix_prob_1 * (scale_1**2 + (loc_1 - ev)**2) + 
                              mix_prob_2 * (scale_2**2 + (loc_2 - ev)**2))
                std_train_data = np.sqrt(var_mixture)
                
                # Do same for train and validation
                mix_prob_1_train = prob_params_train["mix_prob_1"].to_numpy()
                loc_1_train = prob_params_train["loc_1"].to_numpy()
                scale_1_train = prob_params_train["scale_1"].to_numpy()
                loc_2_train = prob_params_train["loc_2"].to_numpy()
                scale_2_train = prob_params_train["scale_2"].to_numpy()
                mix_prob_2_train = 1.0 - mix_prob_1_train
                
                ev_train = mix_prob_1_train * loc_1_train + mix_prob_2_train * loc_2_train
                var_mixture_train = (mix_prob_1_train * (scale_1_train**2 + (loc_1_train - ev_train)**2) + 
                                    mix_prob_2_train * (scale_2_train**2 + (loc_2_train - ev_train)**2))
                std_train = np.sqrt(var_mixture_train)
                
                mix_prob_1_val = prob_params_validation["mix_prob_1"].to_numpy()
                loc_1_val = prob_params_validation["loc_1"].to_numpy()
                scale_1_val = prob_params_validation["scale_1"].to_numpy()
                loc_2_val = prob_params_validation["loc_2"].to_numpy()
                scale_2_val = prob_params_validation["scale_2"].to_numpy()
                mix_prob_2_val = 1.0 - mix_prob_1_val
                
                ev_validation = mix_prob_1_val * loc_1_val + mix_prob_2_val * loc_2_val
                var_mixture_val = (mix_prob_1_val * (scale_1_val**2 + (loc_1_val - ev_validation)**2) + 
                                  mix_prob_2_val * (scale_2_val**2 + (loc_2_val - ev_validation)**2))
                std_validation = np.sqrt(var_mixture_val)
                
                cv = y.Result.std()/y.Result.mean()

            odds_ev = B_train["EV"].to_numpy()

            # === MODEL WEIGHTING AND PROBABILITY CALCULATION ===
            # Fit optimal weight between model and bookmaker EV
            # Calculate weighted predictions with temperature scaling for calibration
            model_weight = fit_model_weight(ev_validation, B_validation["EV"].to_numpy(), y_validation["Result"].to_numpy(), cv, std_validation)

            if dist == "Poisson":
                weighted_ev_train = np.exp(np.average(np.log(np.concatenate([[ev_train], [odds_ev]], axis=0)), weights=[model_weight, 1-model_weight], axis=0))
                weighted_ev_validation = np.exp(np.average(np.log(np.concatenate([[ev_validation], [B_validation["EV"].to_numpy()]], axis=0)), weights=[model_weight, 1-model_weight], axis=0))
                weighted_ev = np.exp(np.average(np.log(np.concatenate([[ev], [B_test["EV"].to_numpy()]], axis=0)), weights=[model_weight, 1-model_weight], axis=0))
                
                y_proba_no_filt = get_odds(B_test["Line"].to_numpy(), weighted_ev)
                y_proba_no_filt = np.array(
                    [y_proba_no_filt, 1-y_proba_no_filt]).transpose()

                res = minimize(lambda x: -np.mean(poisson.logpmf(y_validation["Result"].to_numpy(), weighted_ev_validation) if x == 1 else 
                                                  skellam.logpmf(y_validation["Result"].to_numpy(), (1/(2*x)+.5)*weighted_ev_validation, (1/(2*x)-.5)*weighted_ev_validation)), x0=[1], bounds=[(.1, 1)])
                model_temp = res.x[0]
                
                y_proba_filt = get_odds(B_test["Line"].to_numpy(), weighted_ev, temp=model_temp)
                y_proba_filt = np.array(
                    [y_proba_filt, 1-y_proba_filt]).transpose()
                
                kld_no_filt = -np.mean(poisson.logpmf(y_test["Result"].astype(int).to_numpy(), weighted_ev))
                kld_filt = kld_no_filt if model_temp == 1 else -np.mean(skellam.logpmf(y_test["Result"].to_numpy(), (1/(2*model_temp)+.5)*weighted_ev, (1/(2*model_temp)-.5)*weighted_ev))
                kld_train = -np.mean(poisson.logpmf(y_train["Result"].astype(int).to_numpy(), weighted_ev_train))
            elif dist in ["Gaussian", "Mixture2G"]:
                # All location-scale distributions use similar weighting logic
                s_train = np.concatenate([[std_train if std_train is not None else np.full_like(ev_train, 1.0)], [cv*odds_ev]], axis=0)
                std_train_weighted = np.sqrt(1/np.average(np.power(s_train,-2), weights=[model_weight, 1-model_weight], axis=0))
                weighted_ev_train = np.average(np.concatenate([[ev_train], [odds_ev]], axis=0)*np.power(s_train,-2), weights=[model_weight, 1-model_weight], axis=0)*std_train_weighted*std_train_weighted

                s_validation = np.concatenate([[std_validation if std_validation is not None else np.full_like(ev_validation, 1.0)], [cv*B_validation["EV"].to_numpy()]], axis=0)
                std_validation_weighted = np.sqrt(1/np.average(np.power(s_validation,-2), weights=[model_weight, 1-model_weight], axis=0))
                weighted_ev_validation = np.average(np.concatenate([[ev_validation], [B_validation["EV"].to_numpy()]], axis=0)*np.power(s_validation,-2), weights=[model_weight, 1-model_weight], axis=0)*std_validation_weighted*std_validation_weighted

                s = np.concatenate([[prob_params["scale"] if "scale" in prob_params.columns else np.full(len(prob_params), y_test["Result"].std())], [cv*B_test["EV"].to_numpy()]], axis=0)
                std_test = np.sqrt(1/np.average(np.power(s,-2), weights=[model_weight, 1-model_weight], axis=0))
                weighted_ev = np.average(np.concatenate([[ev], [B_test["EV"].to_numpy()]], axis=0)*np.power(s,-2), weights=[model_weight, 1-model_weight], axis=0)*std_test*std_test

                y_proba_no_filt = get_odds(B_test["Line"].to_numpy(), weighted_ev, cv, std_test, step=step)
                y_proba_no_filt = np.array(
                    [y_proba_no_filt, 1-y_proba_no_filt]).transpose()

                # Temperature optimization for Gaussian and Mixture2G (both use normal-based likelihood)
                res = minimize(lambda x: -np.mean(norm.logpdf(y_validation["Result"].to_numpy(), weighted_ev_validation, std_validation_weighted/x)), x0=[1], bounds=[(.1, 1)])
                model_temp = res.x[0]
                kld_no_filt = -np.mean(norm.logpdf(y_test["Result"].to_numpy(), weighted_ev, std_test))
                kld_filt = -np.mean(norm.logpdf(y_test["Result"].to_numpy(), weighted_ev, std_test/model_temp))
                kld_train = -np.mean(norm.logpdf(y_train["Result"].to_numpy(), weighted_ev_train, std_train_weighted))

                y_proba_filt = get_odds(B_test["Line"].to_numpy(), weighted_ev, cv, std_test, step=step, temp=model_temp)
                y_proba_filt = np.array(
                    [y_proba_filt, 1-y_proba_filt]).transpose()

            stat_std = y.Result.std()

            # === TEST SET STATISTICS ===
            # Calculate model performance metrics on held-out test set
            y_class = (y_test["Result"] >=
                       B_test["Line"]).astype(int)
            y_class = np.ravel(y_class.to_numpy())

            prec = np.zeros(2)
            acc = np.zeros(2)
            sharp = np.zeros(2)
            pit = np.zeros(2)
            ll = np.zeros(2)

            for i, y_proba in enumerate([y_proba_no_filt, y_proba_filt]):
                y_pred = (y_proba > .5).astype(int)[:, 1]
                mask = np.max(y_proba, axis=1) > 0.54
                prec[i] = precision_score(y_class[mask], y_pred[mask])
                acc[i] = accuracy_score(y_class[mask], y_pred[mask])

                sharp[i] = np.std(y_proba[:, 1])
                pit[i] = np.mean(y_proba[:, 1])

                ll[i] = log_loss(y_class, y_proba[:, 1])

            filedict = {
                "model": model,
                "step": step,
                "stats": {
                    "Accuracy": acc,
                    "Precision": prec,
                    "Sharpness": sharp,
                    "PIT": pit,
                    "NLL": ll,
                    "KLD": [kld_no_filt, kld_filt],
                    "Training Gap": [(kld_no_filt-kld_train)/kld_train, (kld_filt-kld_train)/kld_train],
                    "Weight/Temp": [model_weight, model_temp]
                },
                "params": opt_params,
                "distribution": dist,
                "cv": cv,
                "std": stat_std,
                "temperature": model_temp,
                "weight": model_weight
            }

            X_test['Result'] = y_test['Result']
            if dist == "Poisson":
                X_test['EV'] = prob_params['rate']
            elif dist == "Gaussian":
                X_test['EV'] = prob_params['loc']
                X_test['STD'] = prob_params['scale']
            elif dist == "Mixture2G":
                X_test['PI'] = prob_params['mix_prob_1']
                X_test['MU1'] = prob_params['loc_1']
                X_test['SIG1'] = prob_params['scale_1']
                X_test['MU2'] = prob_params['loc_2']
                X_test['SIG2'] = prob_params['scale_2']

            X_test['P'] = y_proba_filt[:, 1]

            filepath = pkg_resources.files(data) / f"test_sets/{filename}.csv"
            with open(filepath, "w") as outfile:
                X_test.to_csv(filepath)

            filepath = pkg_resources.files(data) / f"models/{filename}.mdl"
            with open(filepath, "wb") as outfile:
                pickle.dump(filedict, outfile, -1)
                del filedict
                del model

            # === SAVE TEST PREDICTIONS ===
            # Store predictions on test set for later analysis

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
    
    with open(pkg_resources.files(data) / "training_report.txt", "w") as f:
        for model_str in model_list:
            with open(pkg_resources.files(data) / f"models/{model_str}", "rb") as infile:
                model = pickle.load(infile)

            name = model_str.split("_")
            cv = model['cv']
            std = model.get('std',0)
            league = name[0]
            market = name[1].replace("-", " ").replace(".mdl", "")
            dist = model["distribution"]

            stat_cv.setdefault(league, {})
            stat_cv[league][market] = float(cv)

            stat_std.setdefault(league, {})
            stat_std[league][market] = float(std)
            
            # Save distribution selection
            stat_dist.setdefault(league, {})
            stat_dist[league][market] = dist

            f.write(f" {league} {market} ".center(90, "="))
            f.write("\n")
            f.write(f" Distribution Model: {dist}\n")
            f.write(pd.DataFrame(model['stats'], index=[
                    ['No Filter', 'Filter']]).to_string(index=False))
            f.write("\n\n")

    with open(pkg_resources.files(data) / "stat_cv.json", "w") as f:
        json.dump(stat_cv, f, indent=4)

    with open(pkg_resources.files(data) / "stat_std.json", "w") as f:
        json.dump(stat_std, f, indent=4)
    
    save_distribution_config(stat_dist)


def see_features():
    """Analyze feature importance and correlations across all models using SHAP."""
    model_list = [f.name for f in (pkg_resources.files(data)/"models").iterdir() if ".mdl" in f.name]
    model_list.sort()
    feature_importances = []
    feature_correlations = []
    features_to_filter = {}
    most_important = {}
    for model_str in tqdm(model_list, desc="Analyzing feature importances...", unit="market"):
        with open(pkg_resources.files(data) / f"models/{model_str}", "rb") as infile:
            filedict = pickle.load(infile)

        filepath = pkg_resources.files(
            data) / ("test_sets/" + model_str.replace(".mdl", ".csv"))
        M = pd.read_csv(filepath, index_col=0)

        y = M[['Result']]
        C = M.corr(numeric_only=True)["Result"]
        X = M.drop(columns=['Result', 'EV', 'P'])
        C = C.drop(['Result', 'EV', 'P'])
        
        # Drop distribution-specific parameters that aren't features
        dist = filedict["distribution"]
        if dist == "Gaussian":
            X.drop(columns=["STD"], inplace=True, errors='ignore')
            C.drop(["STD"], inplace=True, errors='ignore')
        elif dist == "Mixture2G":
            X.drop(columns=["STD"], inplace=True, errors='ignore')
            C.drop(["STD"], inplace=True, errors='ignore')
        
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
        
        # Handle different output shapes for different distributions
        if dist == "Gaussian":
            subvals = np.abs(subvals[0]) + np.abs(subvals[1])
        elif dist == "Poisson":
            # Poisson has 1 output, already correct shape
            pass
        else:
            pass

        vals = vals + np.mean(np.abs(subvals), axis=0)

        vals = vals/np.sum(vals)*100
        feature_importances.append(
            {k: v for k, v in list(zip(features, vals))})
        feature_correlations.append(C.to_dict())
        
        features_to_filter[model_str[:-4]] = list(features[vals == 0])
        most_important[model_str[:-4]] = list(features[np.argpartition(vals, -10)[-10:]])

    df = pd.DataFrame(feature_importances, index=[
                      market[:-4] for market in model_list]).fillna(0).infer_objects(copy=False).transpose()
    
    for league in ["NBA", "WNBA", "NFL", "NHL", "MLB"]:
        df[league + "_ALL"] = df[[col for col in df.columns if league in col]].mean(axis=1)
        most_important[league + "_ALL"] = list(df[league + "_ALL"].sort_values(ascending=False).head(10).index)

    df["ALL"] = df[[col for col in df.columns if "ALL" in col]].mean(axis=1)
    most_important["ALL"] = list(df["ALL"].sort_values(ascending=False).head(10).index)

    with open(pkg_resources.files(data) / "features_to_filter.json", "w") as outfile:
        json.dump(features_to_filter, outfile, indent=4)
    with open(pkg_resources.files(data) / "most_important_features.json", "w") as outfile:
        json.dump(most_important, outfile, indent=4)
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

    elif cv == 1:
        def objective(w, x, y):
            proj = np.array(np.exp(np.ma.average(np.ma.MaskedArray(np.log(x), mask=np.isnan(x)), weights=w, axis=1)))
            return -np.mean(poisson.logpmf(y.astype(int), proj))
        
    else:
        def objective(w, x, y):
            s = np.ma.MaskedArray(x*cv, mask=np.isnan(x))
            std = np.sqrt(1/np.ma.average(np.power(s,-2), weights=w, axis=1))
            proj = np.array(np.ma.average(np.ma.MaskedArray(x*np.power(s,-2), mask=np.isnan(x)), weights=w, axis=1)*std*std)
            return -np.mean(norm.logpdf(y, proj, std))
    
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

def fit_model_weight(model_ev, odds_ev, result, cv, model_std=None):
    """Optimize blend weight between model predictions and bookmaker lines."""
    if cv == 1:
        def objective(w, x, y):
            W = np.array([w, 1-w]).flatten()
            proj = np.exp(np.average(np.log(x), weights=W, axis=0))
            return -np.mean(poisson.logpmf(y.astype(int), proj))
        
    else:
        s = np.concatenate([[model_std], [cv*odds_ev]], axis=0)
        def objective(w, x, y):
            W = np.array([w, 1-w]).flatten()
            std = np.sqrt(1/np.average(np.power(s,-2), weights=W, axis=0))
            proj = np.average(x*np.power(s,-2), weights=W, axis=0)*std*std
            return -np.mean(norm.logpdf(y, proj, std))
        
    x = np.concatenate([[model_ev], [odds_ev]], axis=0)

    res = res = minimize(objective, .1, args=(x, result), bounds=[(0.05, 0.9)], tol=1e-8, method='TNC')
    return res.x[0]

def fit_filter(x, y_class):
    """Train logistic regression filter to identify high-confidence predictions."""
    filt = LogisticRegression(
                fit_intercept=False, solver='newton-cholesky', tol=1e-8, max_iter=500, C=.1).fit(x, y_class)

    return filt

def trim_matrix(M):
    """Remove data quality issues and prepare matrix for modeling."""
    warnings.simplefilter('ignore', UserWarning)
    # Trim the fat
    while any(M["DaysIntoSeason"] < 0) or any(M["DaysIntoSeason"] > 300):
        M.loc[M["DaysIntoSeason"] < 0, "DaysIntoSeason"] = M.loc[M["DaysIntoSeason"]
                                                                    < 0, "DaysIntoSeason"] - M["DaysIntoSeason"].min()
        M.loc[M["DaysIntoSeason"] > 300, "DaysIntoSeason"] = M.loc[M["DaysIntoSeason"]
                                                                    > 300, "DaysIntoSeason"] - M.loc[M["DaysIntoSeason"]
                                                                                                    > 300, "DaysIntoSeason"].min()

    M = M.loc[((M["Result"] >= M["Result"].quantile(.05)) & (
        M["Result"] <= M["Result"].quantile(.95))) | (M["Archived"] == 1)]
    M["Line"].clip(M.loc[(M["Archived"] == 1), "Line"].min(), M.loc[(M["Archived"] == 1), "Line"].max(), inplace=True)

    if "Player position" in M.columns:
        for i in M["Player position"].unique():
            target = M.loc[(M["Archived"] == 1) & (
                M["Player position"] == i), "Line"].median()

            less = M.loc[(M["Archived"] != 1) & (
                M["Player position"] == i) & (M["Line"] < target), "Line"]
            more = M.loc[(M["Archived"] != 1) & (
                M["Player position"] == i) & (M["Line"] > target), "Line"]

            n = np.clip(np.abs(less.count() - more.count()), None, np.clip(len(M) - 2000, 0, None))
            if n > 0:
                if less.count() > more.count():
                    chopping_block = less.index
                    counts, bins = np.histogram(less)
                    counts = counts/len(less)
                    actuals = M.loc[(M["Archived"] == 1) & (
                        M["Player position"] == i) & (M["Line"] < target), "Line"]
                    actual_counts, bins = np.histogram(
                        actuals, bins)
                    if len(actuals):
                        actual_counts = actual_counts / \
                            len(actuals)
                    diff = np.clip(
                        counts-actual_counts, 1e-8, None)
                    p = np.zeros(len(less))
                    for j, a in enumerate(bins[:-1]):
                        p[less >= a] = diff[j]
                    p = p/np.sum(p)
                else:
                    chopping_block = more.index
                    counts, bins = np.histogram(more)
                    counts = counts/len(more)
                    actuals = M.loc[(M["Archived"] == 1) & (
                        M["Player position"] == i) & (M["Line"] > target), "Line"]
                    actual_counts, bins = np.histogram(
                        actuals, bins)
                    if len(actuals):
                        actual_counts = actual_counts / \
                            len(actuals)
                    diff = np.clip(
                        counts-actual_counts, 1e-8, None)
                    p = np.zeros(len(more))
                    for j, a in enumerate(bins[:-1]):
                        p[more >= a] = diff[j]
                    p = p/np.sum(p)

                if n > len(chopping_block):
                    n = len(chopping_block)

                cut = np.random.choice(
                    chopping_block, n, replace=False, p=p)
                M.drop(cut, inplace=True)
    else:
        if len(M[(M["Archived"] == 1)]) > 4:
            target = M.loc[(M["Archived"] == 1), "Line"].median()
        else:
            target = M["Line"].mean()

        less = M.loc[(M["Archived"] != 1) & (M["Line"] < target), "Line"]
        more = M.loc[(M["Archived"] != 1) & (M["Line"] > target), "Line"]

        n = np.clip(np.abs(less.count() - more.count()), None, np.clip(len(M) - 2000, 0, None))
        if n > 0:
            if less.count() > more.count():
                chopping_block = less.index
                counts, bins = np.histogram(less)
                counts = counts/len(less)
                actuals = M.loc[(M["Archived"] == 1) & (
                    M["Line"] < target), "Line"]
                actual_counts, bins = np.histogram(
                    actuals, bins)
                if len(actuals):
                    actual_counts = actual_counts/len(actuals)
                diff = np.clip(
                    counts-actual_counts, 1e-8, None)
                p = np.zeros(len(less))
                for j, a in enumerate(bins[:-1]):
                    p[less >= a] = diff[j]
                p = p/np.sum(p)
            else:
                chopping_block = more.index
                counts, bins = np.histogram(more)
                counts = counts/len(more)
                actuals = M.loc[(M["Archived"] == 1) & (
                    M["Line"] > target), "Line"]
                actual_counts, bins = np.histogram(
                    actuals, bins)
                if len(actuals):
                    actual_counts = actual_counts/len(actuals)
                diff = np.clip(
                    counts-actual_counts, 1e-8, None)
                p = np.zeros(len(more))
                for j, a in enumerate(bins[:-1]):
                    p[more >= a] = diff[j]
                p = p/np.sum(p)

            if n > len(chopping_block):
                n = len(chopping_block)

            cut = np.random.choice(
                chopping_block, n, replace=False, p=p)
            M.drop(cut, inplace=True)

    if len(M.loc[(M["Archived"] == 1)]) < 10:
           return M.sort_values("Date")

    pushes = M.loc[M["Result"]==M["Line"]]
    push_rate = pushes["Archived"].sum()/M["Archived"].sum()
    M = M.loc[M["Result"]!=M["Line"]]
    target = (M.loc[(M["Archived"] == 1), "Result"] >
                M.loc[(M["Archived"] == 1), "Line"]).mean()
    
    balance = (M["Result"] > M["Line"]).mean()
    n = np.clip(2*int(np.abs(target - balance) * len(M)), None, np.clip(len(M) - 1600, 0, None))

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

        if n > len(chopping_block):
            n = len(chopping_block)

        cut = np.random.choice(chopping_block, n, replace=False, p=p)
        M.drop(cut, inplace=True)

    n = int(push_rate*len(M))-pushes["Archived"].sum()
    chopping_block = pushes.loc[pushes["Archived"]==0].index
    n = np.clip(n, None, len(chopping_block))
    if n > 0:
        cut = np.random.choice(chopping_block, n, replace=False)
        pushes.drop(cut, inplace=True)

    M = pd.concat([M,pushes]).sort_values("Date")

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
            game_df = game_df.merge(usage, how="left").fillna(0).infer_objects(copy=False)
            game_df = game_df.loc[game_df[log_str["position"]] != None]
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
        c = team_matrix.corr(min_periods=int(len(team_matrix)*.75)).unstack()
        # c = c.iloc[:int(len(c)/2)]
        # l1 = [i.split(".")[0] for i in c.index.get_level_values(0).to_list()]
        # l2 = [i.split(".")[0] for i in c.index.get_level_values(1).to_list()]
        # c = c.loc[[x != y for x, y in zip(l1, l2)]]
        c = c.reindex(c.abs().sort_values(ascending=False).index).dropna()
        c = c.loc[c.abs()>0.05]
        big_c.update({team: c})

    pd.concat(big_c).to_csv((pkg_resources.files(data) / f"{league}_corr.csv"))

if __name__ == "__main__":
    warnings.simplefilter('ignore', UserWarning)
    meditate()
    see_features()
    filter_features()
