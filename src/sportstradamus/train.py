from sportstradamus.stats import StatsMLB, StatsNBA, StatsNHL, StatsNFL
from sportstradamus.helpers import get_ev
import pickle
import importlib.resources as pkg_resources
from sportstradamus import data
import numpy as np
from sklearn.model_selection import train_test_split
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
import smogn


@click.command()
@click.option("--force/--no-force", default=False, help="Force update of all models")
@click.option("--stats/--no-stats", default=False, help="Regenerate model reports")
@click.option("--alt/--no-alt", default=False, help="Generate alternate model sets")
@click.option("--league", type=click.Choice(["All", "NFL", "NBA", "MLB", "NHL"]), default="All",
              help="Select league to train on")
def meditate(force, stats, league, alt):

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
    mlb.update()
    nba = StatsNBA()
    nba.load()
    nba.update()
    nhl = StatsNHL()
    nhl.load()
    nhl.update()
    nfl = StatsNFL()
    nfl.load()
    nfl.update()

    all_markets = {
        "NFL": [
            "passing yards",
            "rushing yards",
            "receiving yards",
            "yards",
            "qb yards",
            "fantasy points prizepicks",
            "fantasy points underdog",
            "fantasy points parlayplay",
            "passing tds",
            "rushing tds",
            "receiving tds",
            "tds",
            "qb tds",
            "completions",
            "carries",
            "receptions",
            "interceptions",
            "attempts",
            "targets",
        ],
        "MLB": [
            "pitcher strikeouts",
            "pitching outs",
            "pitches thrown",
            "hits allowed",
            "runs allowed",
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
            "walks allowed",
            "batter strikeouts",
            "hitter fantasy points parlay",
            "pitcher fantasy points parlay",
            "singles",
        ],
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
            "fantasy points parlay",
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
            "MIN",
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
            "goalie fantasy points parlay",
            "skater fantasy points parlay",
            "blocked",
            "hits",
            "goals",
            "assists",
            "faceOffWins",
            "timeOnIce",
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
                    need_model = False
                else:
                    continue

            print(f"Training {league} - {market}")
            filename = "_".join([league, market]).replace(" ", "-")
            filepath = pkg_resources.files(data) / (filename + ".csv")
            if os.path.isfile(filepath) and not force:
                M = pd.read_csv(filepath, index_col=0)
                M.to_csv(filepath)
            else:
                M = stat_data.get_training_matrix(market)
                M.to_csv(filepath)

            if M.empty:
                continue

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

            if need_model:
                if (y_train["Result"] <= 0).mean() >= .2:
                    m = X_train
                    m["Result"] = y_train["Result"]
                    m = smogn.smoter(data=m, y='Result', samp_method='balance')
                    y_train = m[['Result']]
                    X_train = m.drop(columns=['Result'])

                y_train = np.ravel(y_train.to_numpy())

                dtrain = lgb.Dataset(X_train, label=y_train)

                lgblss_dist_class = DistributionClass()
                candidate_distributions = [Gaussian, Poisson]

                dist = lgblss_dist_class.dist_select(
                    target=y_train, candidate_distributions=candidate_distributions, max_iter=100)

                dist = dist.loc[dist["nll"] > 0].iloc[0, 1]

                params = {
                    "boosting_type": ["categorical", ["gbdt"]],
                    "max_depth": ["int", {"low": 2, "high": 63, "log": False}],
                    "num_leaves": ["int", {"low": 7, "high": 4095, "log": False}],
                    "min_child_weight": ["float", {"low": 1e-2, "high": len(X)*.8/1000, "log": True}],
                    "feature_fraction": ["float", {"low": 0.4, "high": 1.0, "log": False}],
                    "bagging_fraction": ["float", {"low": 0.4, "high": 1.0, "log": False}],
                    "bagging_freq": ["int", {"low": 1, "high": 1, "log": False}]
                }

                model = LightGBMLSS(distributions[dist])
                opt_param = model.hyper_opt(params,
                                            dtrain,
                                            num_boost_round=999,
                                            nfold=5,
                                            early_stopping_rounds=20,
                                            max_minutes=30,
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

            elif alt:
                y_train = np.ravel(y_train.to_numpy())

                dtrain = lgb.Dataset(X_train, label=y_train)

                if dist == "Gaussian":
                    dist = "Gamma"
                    # X_train = X_train[y_train > 1]
                    # y_train = y_train[y_train > 1]
                    y_train[y_train < 2] = 2
                elif dist == "Poisson":
                    dist = "NegativeBinomial"
                else:
                    continue

                params = {
                    "boosting_type": ["categorical", ["gbdt"]],
                    "max_depth": ["int", {"low": 2, "high": 63, "log": False}],
                    "num_leaves": ["int", {"low": 7, "high": 4095, "log": False}],
                    "min_child_weight": ["float", {"low": 1e-2, "high": len(X)*.8/1000, "log": True}],
                    "feature_fraction": ["float", {"low": 0.4, "high": 1.0, "log": False}],
                    "bagging_fraction": ["float", {"low": 0.4, "high": 1.0, "log": False}],
                    "bagging_freq": ["int", {"low": 1, "high": 1, "log": False}]
                }

                model = LightGBMLSS(distributions[dist])
                opt_param = model.hyper_opt(params,
                                            dtrain,
                                            num_boost_round=999,
                                            nfold=5,
                                            early_stopping_rounds=20,
                                            max_minutes=30,
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

            X_test.loc[X_test["Line"] == 0, "Line"] = X_test.loc[X_test["Line"]
                                                                 == 0, "Avg10"].apply(np.ceil)-0.5
            X_test.loc[X_test["Line"] <= 0, "Line"] = 0.5
            prob_params = model.predict(X_test, pred_type="parameters")
            prob_params.index = y_test.index
            prob_params['result'] = y_test['Result']
            if dist == "Poisson":
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
                y_proba = norm.cdf(
                    X_test["Line"], prob_params["loc"], prob_params["scale"])
                p = 0
                ev = prob_params["loc"]
                entropy = np.mean(
                    np.abs(norm.cdf(y_test["Result"], prob_params["loc"], prob_params["scale"])-.5))
            elif dist == "Gamma":
                y_proba = gamma.cdf(
                    X_test["Line"], prob_params["concentration"], scale=1/prob_params["rate"])
                p = 2
                ev = prob_params["concentration"]/prob_params["rate"]
                entropy = np.mean(np.abs(gamma.cdf(
                    y_test["Result"], prob_params["concentration"], scale=1/prob_params["rate"])-.5))
                y_test.loc[y_test["Result"] == 0, 'Result'] = 1
            elif dist == "NegativeBinomial":
                y_proba = nbinom.cdf(
                    X_test["Line"], prob_params["total_count"], prob_params["probs"])
                p = 1
                ev = prob_params["total_count"] * \
                    (1-prob_params["probs"])/prob_params["probs"]
                entropy = np.mean(np.abs(norm.cdf(
                    y_test["Result"], prob_params["total_count"], prob_params["probs"])-.5))

            y_proba = np.array([y_proba, 1-y_proba]).transpose()
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

            entropy0 = np.mean(np.abs(poisson.cdf(
                y_test["Result"], X_test["Line"].apply(get_ev, args=(.5,)))-0.5))

            entropy = 1 - entropy/entropy0

            filedict = {
                "model": model,
                "stats": {
                    "Accuracy": acc,
                    "Precision": prec,
                    "Entropy": entropy,
                    "Likelihood": ll,
                    "Brier Score": bs,
                    "Deviance": dev,
                },
                "params": params,
                "distribution": dist
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
                X_test['P'] = prob_params['probs']

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


def report():
    model_list = [f.name for f in pkg_resources.files(
        data).iterdir() if ".mdl" in f.name]
    model_list.sort()
    with open(pkg_resources.files(data) / "training_report.txt", "w") as f:
        for model_str in model_list:
            with open(pkg_resources.files(data) / model_str, "rb") as infile:
                model = pickle.load(infile)

            name = model_str.split("_")
            league = name[0]
            market = name[1].replace("-", " ").replace(".mdl", "")
            dist = model["distribution"]

            f.write(f" {league} {market} ".center(65, "="))
            f.write("\n")
            f.write(f" Distribution Model: {dist}\n")
            f.write(pd.DataFrame(model['stats'], index=[
                    ['Stats']]).to_string(index=False))
            f.write("\n\n")


if __name__ == "__main__":
    meditate()
