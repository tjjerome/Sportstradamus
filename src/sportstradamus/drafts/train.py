import pickle
import importlib.resources as pkg_resources
from sportstradamus.drafts import data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    accuracy_score,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
from lightgbm import LGBMClassifier


def meditate():

    with open(pkg_resources.files(data) / 'BBM_teams.dat', 'rb') as infile:
        teams = pickle.load(infile)

    X = pd.DataFrame(teams)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LGBMClassifier(
        learning_rate=0.1,
        n_estimators=500,
        max_depth=-1,
        objective='binary'
    )

    model = CalibratedClassifierCV(
        model, cv=10, n_jobs=-1, method="sigmoid")
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)

    threshold = .5
    y_pred = (y_proba > threshold).astype(int)
    preco = precision_score(
        (y_test == 1).astype(int), y_pred[:, 1])
    precu = precision_score(
        (y_test == -1).astype(int), y_pred[:, 0])
    y_pred = y_pred[:, 1] - y_pred[:, 0]
    acc = accuracy_score(
        y_test[np.abs(y_pred) > 0], y_pred[np.abs(y_pred) > 0]
    )
    prec = precision_score(y_test, y_pred, average="weighted")
    roc = roc_auc_score(y_test, y_pred, average="weighted")

    bs = brier_score_loss(y_test, y_proba[:, 1], pos_label=1)

    filedict = {
        "model": model,
        "stats": {
            "Accuracy": acc,
            "Precision_Over": preco,
            "Precision_Under": precu,
            "ROC_AUC": roc,
            "Brier Score Loss": bs,
        },
    }

    filepath = pkg_resources.files(data) / 'REG_SEASON.mdl'
    with open(filepath, "wb") as outfile:
        pickle.dump(filedict, outfile, -1)
        del filedict
        del model


if __name__ == '__main__':
    meditate()
