from propstradamus.stats import StatsMLB, StatsNBA, StatsNHL
import pickle
import importlib.resources as pkg_resources
from propstradamus import data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, roc_auc_score, brier_score_loss
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
import json


def meditate():
    mlb = StatsMLB()
    mlb.load()
    mlb.update()
    nba = StatsNBA()
    nba.load()
    nba.update()
    nhl = StatsNHL()
    nhl.load()
    nhl.update()

    all_markets = {'MLB': ['total bases', 'pitcher strikeouts', 'batter strikeouts', 'runs allowed',
                   'hits', 'pitching outs', 'walks allowed', 'hits allowed', 'runs',
                           'rbi', 'singles', 'hits+runs+rbi', '1st inning runs allowed'],
                   'NBA': ['PTS', 'REB', 'AST', 'TOV', 'STL', 'FG3M',
                           'BLK', 'PRA', 'PR', 'RA', 'PA', 'BLST'],
                   'NHL': ['saves', 'goalsAgainst', 'shots', 'goals',
                           'assists', 'points']
                   }
    for league, markets in all_markets.items():
        for market in markets:
            if league == 'MLB':
                X, y = mlb.get_training_matrix(market)
            elif league == 'NBA':
                X, y = nba.get_training_matrix(market)
            elif league == 'NHL':
                X, y = nhl.get_training_matrix(market)
            else:
                continue

            if X.empty:
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            if len(X_train) < 1000:
                continue

            scaler = MaxAbsScaler()

            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            y_train = np.ravel(y_train.to_numpy())
            y_test = np.ravel(y_test.to_numpy())

            # model = MLPClassifier(hidden_layer_sizes=(472, 112, 64),  # (472,112,64), (448,96,64), (416, 352, 56)
            #                       batch_size=128, tol=3.86e-5, max_iter=1000, alpha=0.001375,
            #                       beta_1=.96, beta_2=.958, learning_rate_init=.005685,
            #                       solver='adam', early_stopping=True, n_iter_no_change=300)  # .553r, .40a, .54u, .64o

            model = GradientBoostingClassifier(
                loss='log_loss', learning_rate=0.1, n_estimators=500, max_depth=10, max_features='log2')  # .547r, .39a, .61u, .55o

            # model = RandomForestClassifier(
            #     criterion='entropy', n_estimators=500, max_features=0.25, max_depth=26)  # .564r, .40a, .58u, .60o

            model = CalibratedClassifierCV(
                model, cv=10, n_jobs=-1, method='sigmoid')
            model.fit(X_train, y_train)

            y_proba = model.predict_proba(X_test)

            thresholds = np.arange(0.52, 0.58, 0.001)
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
                y_pred = y_pred[:, 1]-y_pred[:, 0]
                null[i] = 1-np.mean(np.abs(y_pred))
                acc[i] = accuracy_score(
                    y_test[np.abs(y_pred) > 0], y_pred[np.abs(y_pred) > 0])
                prec[i] = precision_score(y_test, y_pred, average='weighted')
                roc[i] = roc_auc_score(
                    y_test, y_pred, average='weighted')

            i = np.argmax(roc)
            j = np.argmax(acc)
            t1 = thresholds[i]
            t2 = thresholds[j]

            bs = brier_score_loss(y_test, y_proba[:, 1], pos_label=1)

            filedict = {'model': model,
                        'scaler': scaler,
                        'threshold': (.54, t1, t2),
                        'edges': [np.floor(i*2)/2 for i in mlb.edges][:-1],
                        'stats': {
                            'Accuracy': (acc[0], acc[i], acc[j]),
                            'Null Points': (null[0], null[i], null[j]),
                            'Precision_Over': (preco[0], preco[i], preco[j]),
                            'Precision_Under': (precu[0], precu[i], precu[j]),
                            'ROC_AUC': (roc[0], roc[i], roc[j]),
                            'Brier Score Loss': (bs, bs, bs)
                        }}

            filename = "_".join([league, market]).replace(" ", "-")+'.skl'

            filepath = (pkg_resources.files(data) / filename)
            with open(filepath, 'wb') as outfile:
                pickle.dump(filedict, outfile, -1)
                del filedict
                del model
                del scaler

    model_list = [f.name for f in pkg_resources.files(
        data).iterdir() if '.skl' in f.name]

    report = {}
    with open(pkg_resources.files(data) / 'training_report.txt', 'w') as f:

        for model_str in model_list:
            with open(pkg_resources.files(data) / model_str, 'rb') as infile:
                model = pickle.load(infile)

            name = model_str.split('_')
            league = name[0]
            market = name[1].replace('-', ' ').replace('.skl', '')

            f.write(f" {league} {market} ".center(72, '='))
            f.write("\n")
            f.write(pd.DataFrame(model['stats'],
                    index=model['threshold']).to_string())
            f.write("\n\n")


if __name__ == '__main__':
    meditate()
