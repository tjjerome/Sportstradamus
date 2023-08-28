from sportstradamus.stats import StatsMLB, StatsNBA, StatsNHL, StatsNFL
import pickle
import importlib.resources as pkg_resources
from sportstradamus import data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    accuracy_score,
    brier_score_loss,
    r2_score,
)
from scipy.stats import (
    norm,
    poisson
)
import lightgbm as lgb
import pandas as pd
import click
import os
from lightgbmlss.model import LightGBMLSS
from lightgbmlss.distributions import (
    Gaussian,
    Poisson
)
from lightgbmlss.distributions.distribution_utils import DistributionClass


@click.command()
@click.option("--force/--no-force", default=False, help="Force update of all models")
@click.option("--stats/--no-stats", default=False, help="Regenerate model reports")
@click.option("--league", type=click.Choice(["All", "NFL", "NBA", "MLB", "NHL"]), default="All",
              help="Select league to train on")
def meditate(force, stats, league):

    dist_params = {
        "stabilization": "None",
        "response_fn": "exp",
        "loss_fn": "nll"
    }

    distributions = {
        "Gaussian": Gaussian.Gaussian(**dist_params),
        "Poisson": Poisson.Poisson(**dist_params)
    }

    mlb = StatsMLB()
    mlb.load()
    mlb.update()
    # nba = StatsNBA()
    # nba.load()
    # nba.update()
    # nhl = StatsNHL()
    # nhl.load()
    # nhl.update()
    nfl = StatsNFL()
    nfl.load()
    nfl.update()

    all_markets = {
        "NFL": [
            "passing yards",
            "rushing yards",
            "receiving yards",
            "yards",
            "fantasy points prizepicks",
            "fantasy points underdog",
            "fantasy points parlayplay",
            "passing tds",
            "tds",
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
        # "NBA": [
        #     "PTS",
        #     "REB",
        #     "AST",
        #     "PRA",
        #     "PR",
        #     "RA",
        #     "PA",
        #     "FG3M",
        #     "fantasy score",
        #     "fantasy points parlay",
        #     "TOV",
        #     "BLK",
        #     "STL",
        #     "BLST",
        #     "FG3A",
        #     "FTM",
        #     "FGM",
        #     "FGA",
        #     "OREB",
        #     "DREB",
        #     "PF",
        #     "MIN",
        # ],
        # "NHL": [
        #     "points",
        #     "saves",
        #     "goalsAgainst",
        #     "shots",
        #     "sogBS",
        #     "fantasy score",
        #     "goalie fantasy points underdog",
        #     "skater fantasy points underdog",
        #     "goalie fantasy points parlay",
        #     "skater fantasy points parlay",
        #     "blocked",
        #     "hits",
        #     "goals",
        #     "assists",
        #     "faceOffWins",
        #     "timeOnIce",
        # ],
    }
    if not league == "All":
        all_markets = {league: all_markets[league]}
    for league, markets in all_markets.items():
        for market in markets:
            if league == "MLB":
                stat_data = mlb
            # elif league == "NBA":
            #     stat_data = nba
            # elif league == "NHL":
            #     stat_data = nhl
            elif league == "NFL":
                stat_data = nfl
            else:
                continue

            need_model = True
            filename = "_".join([league, market]).replace(" ", "-") + ".mdl"
            filepath = pkg_resources.files(data) / filename
            if os.path.isfile(filepath) and not force:
                if stats:
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
            if os.path.isfile(filepath):
                M = pd.read_csv(filepath, index_col=0)
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
                y_train = np.ravel(y_train.to_numpy())
                if len(X_train) < 500:
                    continue

                dtrain = lgb.Dataset(X_train, label=y_train)

                lgblss_dist_class = DistributionClass()
                candidate_distributions = [Gaussian, Poisson]

                dist = lgblss_dist_class.dist_select(
                    target=y_train, candidate_distributions=candidate_distributions, max_iter=100).iloc[0, 1]

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

            threshold = 0.545
            acc = 0
            null = 0
            preco = 0
            precu = 0
            bs = 0
            r2 = 0

            prob_params = model.predict(X_test, pred_type="parameters")
            if dist == "Poisson":
                under = poisson.cdf(
                    X_test["Line"].replace(0, 0.5), prob_params["rate"])
                push = poisson.pmf(
                    X_test["Line"].replace(0, 0.5), prob_params["rate"])
                y_proba = under - push/2
                ev = prob_params["rate"]
            elif dist == "Gaussian":
                y_proba = norm.cdf(
                    X_test["Line"].replace(0, 0.5), prob_params["loc"], prob_params["scale"])
                ev = prob_params["loc"]

            y_proba = np.array([y_proba, 1-y_proba]).transpose()
            y_class = (y_test["Result"] >
                       X_test["Line"].replace(0, 0.5)).astype(int)
            y_class = np.ravel(y_class.to_numpy())
            y_pred = (y_proba > threshold).astype(int)
            null = 1 - np.mean(np.sum(y_pred, axis=1))
            if null > .98:
                continue
            preco = precision_score(
                (y_class == 1).astype(int), y_pred[:, 1])
            precu = precision_score(
                (y_class == 0).astype(int), y_pred[:, 0])
            y_pred = y_pred[:, 1]
            acc = accuracy_score(
                y_class[np.max(y_proba, axis=1) > threshold], y_pred[np.max(
                    y_proba, axis=1) > threshold]
            )

            bs = brier_score_loss(y_class, y_proba[:, 1], pos_label=1)

            r2 = r2_score(y_test, ev)

            filedict = {
                "model": model,
                "stats": {
                    "Accuracy": acc,
                    "Null Points": null,
                    "Precision_Over": preco,
                    "Precision_Under": precu,
                    "Brier Score Loss": bs,
                    "R2 Score": r2
                },
                "params": params,
                "distribution": dist
            }

            X_test['Result'] = y_test['Result']
            X_test.reset_index(inplace=True, drop=True)
            if dist == "Poisson":
                X_test['EV'] = prob_params['rate']
            elif dist == "Gaussian":
                X_test['EV'] = prob_params['loc']
                X_test['STD'] = prob_params['scale']

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

            f.write(f" {league} {market} ".center(83, "="))
            f.write("\n")
            f.write(f" Distribution Model: {dist}\n")
            f.write(pd.DataFrame(model['stats'], index=[
                    ['Stats']]).to_string(index=False))
            f.write("\n\n")


if __name__ == "__main__":
    meditate()
