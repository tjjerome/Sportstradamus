from sportstradamus.stats import StatsMLB, StatsNBA, StatsNHL, StatsNFL
from sportstradamus.helpers import get_ev
import pickle
import importlib.resources as pkg_resources
from sportstradamus import data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    precision_score,
    accuracy_score,
    brier_score_loss,
    mean_tweedie_deviance,
    log_loss
)
from scipy.stats import (
    norm,
    poisson,
    gamma,
    nbinom
)
import lightgbm as lgb
import pandas as pd
import click
import os
from lightgbmlss.model import LightGBMLSS
from lightgbmlss.distributions import (
    Gaussian,
    Poisson,
    Gamma,
    NegativeBinomial
)
from lightgbmlss.distributions.distribution_utils import DistributionClass
import shap
import json


@click.command()
@click.option("--force/--no-force", default=False, help="Force update of all models")
@click.option("--stats/--no-stats", default=False, help="Regenerate model reports")
@click.option("--alt/--no-alt", default=False, help="Generate alternate model sets")
@click.option("--league", type=click.Choice(["All", "NFL", "NBA", "MLB", "NHL"]), default="All",
              help="Select league to train on")
def meditate(force, stats, league, alt):

    with open(pkg_resources.files(data) / "stat_cv.json", "r") as f:
        stat_cv = json.load(f)

    dist_params = {
        "stabilization": "None",
        # "response_fn": "exp",
        "loss_fn": "nll"
    }

    distributions = {
        "Gaussian": Gaussian.Gaussian(**dist_params),
        "Poisson": Poisson.Poisson(**dist_params),
        "Gamma": Gamma.Gamma(**dist_params),
        "NegativeBinomial": NegativeBinomial.NegativeBinomial(**dist_params)
    }

    mlb = StatsMLB()
    mlb.load()
    # mlb.update()
    nba = StatsNBA()
    nba.load()
    # nba.update()
    nhl = StatsNHL()
    nhl.load()
    # nhl.update()
    nfl = StatsNFL()
    nfl.load()
    # nfl.update()

    all_markets = {
        "NBA": [
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
            "MIN",
        ],
        "NFL": [
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
            "carries",
            "receptions",
            "interceptions",
            "attempts",
            "targets",
        ],
        "NHL": [
            "points",
            "saves",
            "goalsAgainst",
            "shots",
            "sogBS",
            "fantasy points prizepicks",
            "goalie fantasy points underdog",
            "skater fantasy points underdog",
            "blocked",
            "hits",
            "goals",
            "assists",
            "faceOffWins",
            "timeOnIce",
        ],
        "MLB": [
            "pitcher strikeouts",
            "pitching outs",
            "pitches thrown",
            "hits allowed",
            "runs allowed",
            "walks allowed",
            "1st inning runs allowed",
            "1st inning hits allowed",
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
            "singles"
        ],
    }
    if not league == "All":
        all_markets = {league: all_markets[league]}
    for league, markets in all_markets.items():
        for market in markets:
            if league == "MLB":
                stat_data = mlb
            elif league == "NBA":
                stat_data = nba
            elif league == "NHL":
                stat_data = nhl
            elif league == "NFL":
                stat_data = nfl
            else:
                continue

            need_model = True
            filename = "_".join([league, market]).replace(" ", "-") + ".mdl"
            filepath = pkg_resources.files(data) / filename
            if os.path.isfile(filepath) and not force:
                if stats or alt:
                    with open(filepath, 'rb') as infile:
                        filedict = pickle.load(infile)
                        model = filedict['model']
                        params = filedict['params']
                        dist = filedict['distribution']
                        cv = filedict['cv']
                    need_model = False
                else:
                    continue

            print(f"Training {league} - {market}")
            cv = stat_cv[league].get(market, 1)
            filename = "_".join([league, market]).replace(" ", "-")
            filepath = pkg_resources.files(data) / (filename + ".csv")
            if os.path.isfile(filepath) and not force:
                M = pd.read_csv(filepath, index_col=0).dropna()
            else:
                M = stat_data.get_training_matrix(market, cv)
                M.to_csv(filepath)

            if M.empty:
                continue

            while any(M["DaysIntoSeason"] < 0) or any(M["DaysIntoSeason"] > 300):
                M.loc[M["DaysIntoSeason"] < 0, "DaysIntoSeason"] = M.loc[M["DaysIntoSeason"]
                                                                         < 0, "DaysIntoSeason"] - M["DaysIntoSeason"].min()
                M.loc[M["DaysIntoSeason"] > 300, "DaysIntoSeason"] = M.loc[M["DaysIntoSeason"]
                                                                           > 300, "DaysIntoSeason"] - M.loc[M["DaysIntoSeason"]
                                                                                                            > 300, "DaysIntoSeason"].min()

            M = M.loc[((M["Line"] >= M["Line"].quantile(.05)) & (
                M["Line"] <= M["Line"].quantile(.95))) | (M["Odds"] != 0.5)]

            y = M[['Result']]
            X = M.drop(columns=['Result'])

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            categories = ["Home", "Position"]
            if "Position" not in X.columns:
                categories.remove("Position")
            for c in categories:
                X[c] = X[c].astype('category')
                X_train[c] = X_train[c].astype('category')
                X_test[c] = X_test[c].astype('category')

            categories = "name:"+",".join(categories)

            params = {
                "feature_pre_filter": ["categorical", [False]],
                "boosting_type": ["categorical", ["gbdt"]],
                "max_depth": ["int", {"low": 2, "high": 63, "log": False}],
                # "max_bin": ["int", {"low": 63, "high": 4095, "log": False}],
                "num_leaves": ["int", {"low": 7, "high": 4095, "log": False}],
                "lambda_l1": ["float", {"low": 1e-8, "high": 10, "log": True}],
                "lambda_l2": ["float", {"low": 1e-8, "high": 10, "log": True}],
                "min_child_samples": ["int", {"low": 2, "high": 500, "log": True}],
                "min_child_weight": ["float", {"low": 1e-2, "high": len(X_train)*.8/1000, "log": True}],
                "path_smooth": ["float", {"low": 0, "high": len(X_train)/2, "log": False}],
                "learning_rate": ["float", {"low": 5e-2, "high": 0.5, "log": True}],
                "min_gain_to_split": ["float", {"low": 1e-8, "high": 40, "log": False}],
                "feature_fraction": ["float", {"low": 0.4, "high": 1.0, "log": False}],
                "bagging_fraction": ["float", {"low": 0.4, "high": 1.0, "log": False}],
                "bagging_freq": ["int", {"low": 1, "high": 15, "log": False}]
            }

            if need_model:
                y_train_labels = np.ravel(y_train.to_numpy())

                dtrain = lgb.Dataset(X_train, label=y_train_labels)

                lgblss_dist_class = DistributionClass()
                candidate_distributions = [Gaussian, Poisson]

                dist = lgblss_dist_class.dist_select(
                    target=y_train_labels, candidate_distributions=candidate_distributions, max_iter=100)

                dist = dist.loc[dist["nll"] > 0].iloc[0, 1]

                model = LightGBMLSS(distributions[dist])
                opt_param = model.hyper_opt(params,
                                            dtrain,
                                            num_boost_round=999,
                                            nfold=5,
                                            early_stopping_rounds=20,
                                            max_minutes=30,
                                            n_trials=180,
                                            silence=True,
                                            )
                opt_params = opt_param.copy()
                n_rounds = opt_params["opt_rounds"]
                del opt_params["opt_rounds"]

                model.train(opt_params,
                            dtrain,
                            num_boost_round=n_rounds
                            )

            elif alt:
                y_train_labels = np.ravel(y_train.to_numpy())

                dtrain = lgb.Dataset(X_train, label=y_train_labels)

                if dist == "Gaussian":
                    dist = "Gamma"
                    # X_train = X_train[y_train > 1]
                    # y_train = y_train[y_train > 1]
                    y_train_labels[y_train_labels < 2] = 2
                elif dist == "Poisson":
                    dist = "NegativeBinomial"
                else:
                    continue

                model = LightGBMLSS(distributions[dist])
                opt_param = model.hyper_opt(params,
                                            dtrain,
                                            num_boost_round=999,
                                            nfold=5,
                                            early_stopping_rounds=20,
                                            max_minutes=60,
                                            n_trials=500,
                                            silence=True,
                                            )
                opt_params = opt_param.copy()
                n_rounds = opt_params["opt_rounds"]
                del opt_params["opt_rounds"]

                model.train(opt_params,
                            dtrain,
                            num_boost_round=n_rounds
                            )

            prob_params_train = model.predict(X_train, pred_type="parameters")
            prob_params_train.index = y_train.index
            prob_params_train['result'] = y_train['Result']
            prob_params = model.predict(X_test, pred_type="parameters")
            prob_params.index = y_test.index
            prob_params['result'] = y_test['Result']
            cv = 1
            if dist == "Poisson":
                under = poisson.cdf(
                    X_train["Line"], prob_params_train["rate"])
                push = poisson.pmf(
                    X_train["Line"], prob_params_train["rate"])
                y_proba_train = under - push/2
                under = poisson.cdf(
                    X_test["Line"], prob_params["rate"])
                push = poisson.pmf(
                    X_test["Line"], prob_params["rate"])
                y_proba = under - push/2
                p = 1
                ev = prob_params["rate"]
                entropy = np.mean(
                    np.abs(poisson.cdf(y_test["Result"], prob_params["rate"])-.5))
            elif dist == "Gaussian":
                y_proba_train = norm.cdf(
                    X_train["Line"], prob_params_train["loc"], prob_params_train["scale"])
                y_proba = norm.cdf(
                    X_test["Line"], prob_params["loc"], prob_params["scale"])
                p = 0
                ev = prob_params["loc"]
                entropy = np.mean(
                    np.abs(norm.cdf(y_test["Result"], prob_params["loc"], prob_params["scale"])-.5))
                cv = y.Result.std()/y.Result.mean()
            elif dist == "Gamma":
                y_proba_train = gamma.cdf(
                    X_train["Line"], prob_params_train["concentration"], scale=1/prob_params_train["rate"])
                y_proba = gamma.cdf(
                    X_test["Line"], prob_params["concentration"], scale=1/prob_params["rate"])
                p = 2
                ev = prob_params["concentration"]/prob_params["rate"]
                entropy = np.mean(np.abs(gamma.cdf(
                    y_test["Result"], prob_params["concentration"], scale=1/prob_params["rate"])-.5))
                y_test.loc[y_test["Result"] == 0, 'Result'] = 1
            elif dist == "NegativeBinomial":
                y_proba_train = nbinom.cdf(
                    X_train["Line"], prob_params_train["total_count"], prob_params_train["probs"])
                y_proba = nbinom.cdf(
                    X_test["Line"], prob_params["total_count"], prob_params["probs"])
                p = 1
                ev = prob_params["total_count"] * \
                    (1-prob_params["probs"])/prob_params["probs"]
                entropy = np.mean(np.abs(norm.cdf(
                    y_test["Result"], prob_params["total_count"], prob_params["probs"])-.5))

            y_class = (y_train["Result"] >
                       X_train["Line"]).astype(int)
            y_proba_train = (1-y_proba_train).reshape(-1, 1)
            clf = LogisticRegressionCV().fit(y_proba_train, y_class)
            y_proba = (1-y_proba).reshape(-1, 1)
            y_proba = clf.predict_proba(y_proba)

            # y_proba = np.array([y_proba, 1-y_proba]).transpose()
            y_class = (y_test["Result"] >
                       X_test["Line"]).astype(int)
            y_class = np.ravel(y_class.to_numpy())

            y_pred = (y_proba > .5).astype(int)[:, 1]
            mask = np.max(y_proba, axis=1) > 0.54
            prec = precision_score(y_class[mask], y_pred[mask])
            acc = accuracy_score(y_class[mask], y_pred[mask])

            bs = brier_score_loss(y_class, y_proba[:, 1], pos_label=1)
            bs0 = brier_score_loss(
                y_class, 0.5*np.ones_like(y_class), pos_label=1)
            bs = 1 - bs/bs0

            dev = mean_tweedie_deviance(y_test, ev, power=p)
            dev0 = mean_tweedie_deviance(y_test, X_test["Line"], power=p)
            dev = 1 - dev/dev0

            ll = log_loss(y_class, y_proba[:, 1])
            ll0 = log_loss(y_class, 0.5*np.ones_like(y_class))
            ll = 1 - ll/ll0

            if cv == 1:
                entropy0 = np.mean(np.abs(poisson.cdf(
                    y_test["Result"], X_test["Line"].apply(get_ev, args=(.5,)))-0.5))
            else:
                entropy0 = np.mean(np.abs(norm.cdf(
                    y_test["Result"], X_test["Line"].apply(get_ev, args=(.5, cv)))-0.5))

            entropy = 1 - entropy/entropy0

            filedict = {
                "model": model,
                "filter": clf,
                "stats": {
                    "Accuracy": acc,
                    "Precision": prec,
                    "Entropy": entropy,
                    "Likelihood": ll,
                    "Brier Score": bs,
                    "Deviance": dev,
                },
                "params": params,
                "distribution": dist,
                "cv": cv
            }

            X_test['Result'] = y_test['Result']
            if dist == "Poisson":
                X_test['EV'] = prob_params['rate']
            elif dist == "Gaussian":
                X_test['EV'] = prob_params['loc']
                X_test['STD'] = prob_params['scale']
            elif dist == "Gamma":
                X_test['alpha'] = prob_params['concentration']
                X_test['beta'] = prob_params['rate']
            elif dist == "NegativeBinomial":
                X_test['N'] = prob_params['total_count']
                X_test['L'] = prob_params['probs']

            X_test['P'] = y_proba[:, 1]

            filename = "_".join(["test", league, market]
                                ).replace(" ", "-") + ".csv"

            filepath = pkg_resources.files(data) / filename
            with open(filepath, "wb") as outfile:
                X_test.to_csv(filepath)

            filename = "_".join([league, market]).replace(" ", "-") + ".mdl"

            filepath = pkg_resources.files(data) / filename
            with open(filepath, "wb") as outfile:
                pickle.dump(filedict, outfile, -1)
                del filedict
                del model

            report()
            see_features()


def report():
    model_list = [f.name for f in pkg_resources.files(
        data).iterdir() if ".mdl" in f.name]
    model_list.sort()
    with open(pkg_resources.files(data) / "stat_cv.json", "r") as f:
        stat_cv = json.load(f)
    with open(pkg_resources.files(data) / "training_report.txt", "w") as f:
        for model_str in model_list:
            with open(pkg_resources.files(data) / model_str, "rb") as infile:
                model = pickle.load(infile)

            name = model_str.split("_")
            cv = model['cv']
            league = name[0]
            market = name[1].replace("-", " ").replace(".mdl", "")
            dist = model["distribution"]
            if league not in stat_cv:
                stat_cv[league] = {}

            stat_cv[league][market] = cv

            f.write(f" {league} {market} ".center(65, "="))
            f.write("\n")
            f.write(f" Distribution Model: {dist}\n")
            f.write(pd.DataFrame(model['stats'], index=[
                    ['Stats']]).to_string(index=False))
            f.write("\n\n")

    with open(pkg_resources.files(data) / "stat_cv.json", "w") as f:
        json.dump(stat_cv, f, indent=4)


def see_features():
    model_list = [f.name for f in pkg_resources.files(
        data).iterdir() if ".mdl" in f.name]
    model_list.sort()
    feature_importances = []
    for model_str in model_list:
        with open(pkg_resources.files(data) / model_str, "rb") as infile:
            model = pickle.load(infile)

        filepath = pkg_resources.files(
            data) / ("test_" + model_str.replace(".mdl", ".csv"))
        M = pd.read_csv(filepath, index_col=0)

        y = M[['Result']]
        X = M.drop(columns=['Result', 'EV', 'P'])
        if model["distribution"] == "Gaussian":
            X.drop(columns=["STD"], inplace=True)
        features = X.columns

        categories = ["Home", "Position"]
        if "Position" not in features:
            categories.remove("Position")
        for c in categories:
            X[c] = X[c].astype('category')

        explainer = shap.TreeExplainer(model['model'].booster)
        vals = explainer.shap_values(X)
        if model["distribution"] == "Gaussian":
            vals = np.abs(vals[0]) + np.abs(vals[1])

        vals = np.mean(np.abs(vals), axis=0)
        vals = vals/np.sum(vals)*100

        feature_importances.append(
            {k: v for k, v in list(zip(features, vals))})

    df = pd.DataFrame(feature_importances, index=[
                      market[:-4] for market in model_list]).fillna(0).transpose()
    df.to_csv(pkg_resources.files(data) / "feature_importances.csv")


if __name__ == "__main__":
    meditate()
