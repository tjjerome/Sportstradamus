from sportstradamus.stats import StatsMLB, StatsNBA, StatsNHL
import pickle
import importlib.resources as pkg_resources
from sportstradamus import data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    accuracy_score,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.preprocessing import MaxAbsScaler
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
import lightgbm as lgb
import optuna
import pandas as pd
import click
import os


@click.command()
@click.option("-f", "--force", default=False, help="Display progress bars")
@click.option("--league", type=click.Choice(['All', 'NFL', 'NBA', 'MLB', 'NHL']), default='All',
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

    all_markets = {
        "MLB": [
            "pitcher strikeouts",
            "total bases",
            "batter strikeouts",
            "runs allowed",
            "hits",
            "pitching outs",
            "pitches thrown",
            "walks allowed",
            "hits allowed",
            "runs",
            "rbi",
            "singles",
            "hits+runs+rbi",
            "1st inning runs allowed",
        ],
        "NBA": [
            "PTS",
            "REB",
            "AST",
            "TOV",
            "STL",
            "FG3M",
            "BLK",
            "PRA",
            "PR",
            "RA",
            "PA",
            "BLST",
        ],
        "NHL": [
            "saves",
            "goalsAgainst",
            "shots",
            "goals",
            "assists",
            "points",
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
            else:
                continue

            filename = "_".join([league, market]).replace(" ", "-") + ".mdl"
            filepath = pkg_resources.files(data) / filename
            if os.path.isfile(filepath) and not force:
                continue

            print(f"Training {league} - {market}")
            X, y = stat_data.get_training_matrix(market)

            if X.empty:
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            if len(X_train) < 1000:
                continue

            y_train = np.ravel(y_train.to_numpy())
            y_test = np.ravel(y_test.to_numpy())

            opt_params = optimize(X, y)

            params = {
                'boosting_type': 'gbdt',
                'max_depth': 9,
                'n_estimators': 500,
                'num_leaves': 65,
                'min_child_weight': 5.9665759403609915,
                'feature_fraction': 0.8276862446751593,
                'bagging_fraction': 0.8821195174753188,
                'bagging_freq': 1,
                'is_unbalance': False
            } | opt_params

            trees = LGBMClassifier(**params)

            trees.fit(X_train, y_train)
            model = CalibratedClassifierCV(
                trees, cv=15, n_jobs=-1, method="sigmoid")
            model.fit(X_train, y_train)

            y_proba = model.predict_proba(X_test)

            thresholds = np.arange(0.5, 0.58, 0.001)
            acc = np.zeros_like(thresholds)
            null = np.zeros_like(thresholds)
            preco = np.zeros_like(thresholds)
            precu = np.zeros_like(thresholds)
            prec = np.zeros_like(thresholds)
            roc = np.zeros_like(thresholds)
            for i, threshold in enumerate(thresholds):
                y_pred = (y_proba > threshold).astype(int)
                preco[i] = precision_score(
                    (y_test == 1).astype(int), y_pred[:, 1])
                precu[i] = precision_score(
                    (y_test == -1).astype(int), y_pred[:, 0])
                y_pred = y_pred[:, 1] - y_pred[:, 0]
                null[i] = 1 - np.mean(np.abs(y_pred))
                acc[i] = accuracy_score(
                    y_test[np.abs(y_pred) > 0], y_pred[np.abs(y_pred) > 0]
                )
                prec[i] = precision_score(y_test, y_pred, average="weighted")
                roc[i] = roc_auc_score(y_test, y_pred, average="weighted")

            i = np.argmax(roc)
            j = np.argmax(acc)
            t1 = thresholds[i]
            t2 = thresholds[j]

            bs = brier_score_loss(y_test, y_proba[:, 1], pos_label=1)

            filedict = {
                "model": model,
                "threshold": (0.545, t1, t2),
                "edges": [np.floor(i * 2) / 2 for i in stat_data.edges][:-1],
                "stats": {
                    "Accuracy": (acc[26], acc[i], acc[j]),
                    "Null Points": (null[26], null[i], null[j]),
                    "Precision_Over": (preco[26], preco[i], preco[j]),
                    "Precision_Under": (precu[26], precu[i], precu[j]),
                    "ROC_AUC": (roc[26], roc[i], roc[j]),
                    "Brier Score Loss": (bs, bs, bs),
                },
            }

            filename = "_".join([league, market]).replace(" ", "-") + ".mdl"

            filepath = pkg_resources.files(data) / filename
            with open(filepath, "wb") as outfile:
                pickle.dump(filedict, outfile, -1)
                del filedict
                del model


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
            f.write(pd.DataFrame(model["stats"],
                    index=model["threshold"]).to_string())
            f.write("\n\n")


def optimize(X, y):

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    def objective(trial):
        # Define hyperparameters
        params = {
            'objective': 'binary',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'max_depth': trial.suggest_int('max_depth', 2, 63),
            'num_leaves': trial.suggest_int('num_leaves', 7, 4095),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-2, len(X_train)/1000, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'n_estimators': 9999999,
            'bagging_freq': 1,
            'is_unbalance': False,
            'metric': 'auc'
        }

        # Train model
        early_stopping = lgb.early_stopping(100)
        pruning_callback = optuna.integration.LightGBMPruningCallback(
            trial, "auc")
        model = lgb.train(params, lgb_train, valid_sets=lgb_val,
                          callbacks=[pruning_callback, early_stopping])

        # Return metrics
        y_pred = model.predict(X_val)
        y_pred = np.rint(y_pred)
        y_pred[y_pred == 0] = -1
        acc = accuracy_score(y_val, y_pred)
        # Return accuracy on validation set
        return acc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=1000)

    return study.best_params | {'n_estimators': study.best_trial.last_step}


if __name__ == "__main__":
    meditate()
    report()
