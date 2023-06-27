from sportstradamus.stats import StatsNBA, StatsMLB
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_score,
    accuracy_score,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier
import importlib.resources as pkg_resources
from sportstradamus import data
import os
import pickle
import lightgbm as lgb
import optuna


# class for the tree-based/logistic regression pipeline


class TreeBasedLR:

    # initialization
    def __init__(self, forest_params, lr_params):

        # storing parameters
        self.forest_params = forest_params
        self.lr_params = lr_params

    # method for fitting the model
    def fit(self, X, y, sample_weight=None):

        # dict for finding the models

        # configuring the models
        self.lr = LogisticRegression(**self.lr_params)
        self.forest = LGBMClassifier(**self.forest_params)
        self.classes_ = np.unique(y)

        # first, we fit our tree-based model on the dataset
        self.forest.fit(X, y)

        # then, we apply the model to the data in order to get the leave indexes
        leaves = self.forest.predict(X, pred_leaf=True)

        # then, we one-hot encode the leave indexes so we can use them in the logistic regression
        self.encoder = OneHotEncoder()
        leaves_encoded = self.encoder.fit_transform(leaves)

        # and fit it to the encoded leaves
        self.lr.fit(leaves_encoded, y)

    # method for predicting probabilities
    def predict_proba(self, X):

        # then, we apply the model to the data in order to get the leave indexes
        leaves = self.forest.predict(X, pred_leaf=True)

        # then, we one-hot encode the leave indexes so we can use them in the logistic regression
        leaves_encoded = self.encoder.transform(leaves)

        # and fit it to the encoded leaves
        y_hat = self.lr.predict_proba(leaves_encoded)

        # retuning probabilities
        return y_hat

    # get_params, needed for sklearn estimators
    def get_params(self, deep=True):
        return {'forest_params': self.forest_params,
                'lr_params': self.lr_params,
                'forest_model': self.forest_model}


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
            'is_unbalance': trial.suggest_categorical('is_unbalance', [True, False]),
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

        # Return accuracy on validation set
        return roc_auc_score(y_val, y_pred)

    def objectiveLR(trial, tree_params):
        # Define hyperparameters
        params = {
            'solver': trial.suggest_categorical('solver', ['sag', 'newton-cg', 'newton-cholesky', 'lbfgs']),
            'C': trial.suggest_float('C', .001, 10, log=True),
            'fit_intercept': False
        }

        model = TreeBasedLR(tree_params, params)
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_val)

        return brier_score_loss(y_val, y_proba[:, 1], pos_label=1)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=1000)

    tree_params = study.best_params | {
        'n_estimators': study.best_trial.last_step-99}

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    studyLR = optuna.create_study(direction='minimize')
    studyLR.optimize(lambda x: objectiveLR(x, tree_params), n_trials=1000)

    return tree_params, studyLR.best_params


league = "NBA"
markets = ["PRA"]
for market in markets:
    # filename = "_".join([league, market]).replace(" ", "-") + ".mdl"
    # filepath = pkg_resources.files(data) / filename
    # with open(pkg_resources.files(data) / filename, 'rb') as infile:
    #     params = pickle.load(infile)['params']
    filename = "_".join([league, market]).replace(" ", "-")
    filepathX = pkg_resources.files(data) / (filename + "_X.csv")
    filepathy = pkg_resources.files(data) / (filename + "_y.csv")
    if os.path.isfile(filepathX):
        X = pd.read_csv(filepathX, index_col=0)
        y = pd.read_csv(filepathy, index_col=0)
    else:
        continue

    X = X.reset_index()
    y = np.ravel(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if len(X_train) < 1000:
        continue

    # scaler = MpltAbsScaler()

    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    # y_train = np.ravel(y_train.to_numpy())
    # y_test = np.ravel(y_test.to_numpy())

    # model = MLPClassifier(hidden_layer_sizes=(472, 112, 64),  # (472,112,64), (448,96,64), (416, 352, 56)
    #                       batch_size=128, tol=3.86e-5, mplt_iter=1000, alpha=0.001375,
    #                       beta_1=.96, beta_2=.958, learning_rate_init=.005685,
    #                       solver='adam', early_stopping=True, n_iter_no_change=300)  # .553r, .40a, .54u, .64o

    # model = GradientBoostingClassifier(
    #     loss="log_loss",
    #     learning_rate=0.1,
    #     n_estimators=500,
    #     mplt_depth=10,
    #     mplt_features="log2",
    # )  # .547r, .39a, .61u, .55o

    # model = RandomForestClassifier(
    #     criterion='entropy', n_estimators=500, mplt_features=0.25, mplt_depth=26)  # .564r, .40a, .58u, .60o

    # model.fit(X_train, y_train)

    tree_params, lr_params = optimize(X_train, y_train)

    trees = LGBMClassifier(**tree_params)
    lgblr = TreeBasedLR(tree_params, lr_params)

    trees.fit(X_train, y_train)
    lgblr.fit(X_train, y_train)
    modelCV5 = CalibratedClassifierCV(
        trees, cv=5, n_jobs=-1, method="sigmoid")
    modelCV5.fit(X_train, y_train)
    isoCV5 = CalibratedClassifierCV(
        trees, cv=5, n_jobs=-1, method="isotonic")
    isoCV5.fit(X_train, y_train)

    y_proba = trees.predict_proba(X_test)
    y_probalr = lgblr.predict_proba(X_test)
    y_CV5 = modelCV5.predict_proba(X_test)
    y_iCV5 = isoCV5.predict_proba(X_test)

    fop, mpv = calibration_curve(y_test, y_proba[:, 1], n_bins=20)
    bs = brier_score_loss(y_test, y_proba[:, 1], pos_label=1)
    auc = roc_auc_score(y_test, y_proba[:, 1])
    foplr, mpvlr = calibration_curve(y_test, y_probalr[:, 1], n_bins=20)
    bslr = brier_score_loss(y_test, y_probalr[:, 1], pos_label=1)
    auclr = roc_auc_score(y_test, y_probalr[:, 1])
    fopCV5, mpvCV5 = calibration_curve(y_test, y_CV5[:, 1], n_bins=20)
    bsCV5 = brier_score_loss(y_test, y_CV5[:, 1], pos_label=1)
    aucCV5 = roc_auc_score(y_test, y_CV5[:, 1])
    fopiCV5, mpviCV5 = calibration_curve(y_test, y_iCV5[:, 1], n_bins=20)
    bsiCV5 = brier_score_loss(y_test, y_iCV5[:, 1], pos_label=1)
    auciCV5 = roc_auc_score(y_test, y_iCV5[:, 1])

    fig = plt.figure()
    # plot perfectly calibrated
    plt.plot([0.35, 0.65], [0.35, 0.65], linestyle="--")
    # plot model reliability
    plt.plot(mpv, fop, marker=".", label="Uncalibrated")
    plt.plot(mpvlr, foplr, marker=".", label="Custom")
    plt.plot(mpvCV5, fopCV5, marker=".", label="Sigmoid, CV5")
    plt.plot(mpviCV5, fopiCV5, marker=".", label="Isotonic, CV5")
    plt.legend()
    plt.show()

    print(f"Uncalibrated: BS - {bs}, AUC - {auc}")
    print(f"Custom: BS - {bslr}, AUC - {auclr}")
    print(f"Sigmoid, CV5: BS - {bsCV5}, AUC - {aucCV5}")
    print(f"Isotonic, CV5: BS - {bsiCV5}, AUC - {auciCV5}")
