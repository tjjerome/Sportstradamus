from sportstradamus.stats import StatsMLB, StatsNBA, StatsNHL, StatsNFL
import pickle
import importlib.resources as pkg_resources
from sportstradamus import data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    accuracy_score,
    roc_auc_score,
    brier_score_loss
)
from scipy.stats import norm
import lightgbm as lgb
import pandas as pd
import click
import os
from lightgbmlss.model import LightGBMLSS
from lightgbmlss.distributions.Gaussian import Gaussian


@click.command()
@click.option("-f", "--force", default=False, help="Display progress bars")
@click.option("--league", type=click.Choice(["All", "NFL", "NBA", "MLB", "NHL"]), default="All",
              help="Select league to train on")
def meditate(force, league):
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
            elif league == "NBA":
                stat_data = nba
            elif league == "NHL":
                stat_data = nhl
            elif league == "NFL":
                stat_data = nfl
            else:
                continue

            filename = "_".join([league, market]).replace(" ", "-") + ".mdl"
            filepath = pkg_resources.files(data) / filename
            if os.path.isfile(filepath) and not force:
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

            M = M.loc[(M['Combo'] == 0) & (M['Rival'] == 0)
                      ].reset_index(drop=True)

            y = M[['Result']]
            X = M.drop(columns=['Result'])

            columns = ['Avg5', 'Avg10', 'AvgH2H', 'Meeting 1', 'Meeting 2', 'Meeting 3', 'Meeting 4', 'Meeting 5',
                       'Game 1', 'Game 2', 'Game 3', 'Game 4', 'Game 5', 'Game 6']
            for column in columns:
                X.loc[X[column] != 0, column] = X.loc[X[column]
                                                      != 0, column] + X.loc[X[column] != 0, "Line"]
            y['Result'] = y['Result'] + X['Line']
            X.drop(columns=['Last5', 'Last10', 'H2H',
                   'Combo', 'Rival'], inplace=True)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            y_train = np.ravel(y_train.to_numpy())
            if len(X_train) < 800:
                continue

            categories = ["Home", "Position"]
            if "Position" not in X.columns:
                categories.remove("Position")
            for c in categories:
                X[c] = X[c].astype('category')
                X_train[c] = X_train[c].astype('category')
                X_test[c] = X_test[c].astype('category')

            categories = "name:"+",".join(categories)

            dtrain = lgb.Dataset(X_train, label=y_train)

            params = {
                "boosting_type": ["categorical", ["gbdt"]],
                "max_depth": ["int", {"low": 2, "high": 63, "log": False}],
                "num_leaves": ["int", {"low": 7, "high": 4095, "log": False}],
                "min_child_weight": ["float", {"low": 1e-2, "high": len(X)*.8/1000, "log": True}],
                "feature_fraction": ["float", {"low": 0.4, "high": 1.0, "log": False}],
                "bagging_fraction": ["float", {"low": 0.4, "high": 1.0, "log": False}],
                "bagging_freq": ["int", {"low": 1, "high": 1, "log": False}]
            }

            model = LightGBMLSS(
                Gaussian(stabilization="None",
                         response_fn="exp",
                         loss_fn="nll"
                         )
            )
            opt_param = model.hyper_opt(params,
                                        dtrain,
                                        num_boost_round=999,
                                        nfold=5,
                                        early_stopping_rounds=20,
                                        max_minutes=20,
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
            roc = 0
            bs = 0

            prob_params = model.predict(X_test, pred_type="parameters")
            y_proba = norm.cdf(
                X_test["Line"], prob_params["loc"], prob_params["scale"])
            y_proba = np.array([y_proba, 1-y_proba]).transpose()
            y_test = (y_test["Result"] - X_test["Line"]).clip(0, 1).astype(int)
            y_test = np.ravel(y_test.to_numpy())
            y_pred = (y_proba > threshold).astype(int)
            null = 1 - np.mean(np.sum(y_pred, axis=1))
            if null > .98:
                continue
            preco = precision_score(
                (y_test == 1).astype(int), y_pred[:, 1])
            precu = precision_score(
                (y_test == 0).astype(int), y_pred[:, 0])
            y_pred = y_pred[:, 1]
            acc = accuracy_score(
                y_test[np.max(y_proba, axis=1) > threshold], y_pred[np.max(
                    y_proba, axis=1) > threshold]
            )
            roc = roc_auc_score(y_test[np.max(y_proba, axis=1) > threshold], y_pred[np.max(
                y_proba, axis=1) > threshold], average="weighted")

            bs = brier_score_loss(y_test, y_proba[:, 1], pos_label=1)

            filedict = {
                "model": model,
                "stats": {
                    "Accuracy": acc,
                    "Null Points": null,
                    "Precision_Over": preco,
                    "Precision_Under": precu,
                    "ROC_AUC": roc,
                    "Brier Score Loss": bs,
                },
                "params": params
            }

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

            f.write(f" {league} {market} ".center(89, "="))
            f.write("\n")
            f.write(pd.DataFrame(model['stats'], index=[
                    ['Normal', 'Combo', 'Rival']]).to_string())
            f.write("\n\n")


if __name__ == "__main__":
    meditate()
