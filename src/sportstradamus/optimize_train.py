# %%
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import optuna
import importlib.resources as pkg_resources
import data
from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib import plot_edf
from optuna.visualization.matplotlib import plot_intermediate_values
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_slice
from optuna.visualization.matplotlib import plot_pareto_front
from sklearn.metrics import (
    precision_score,
    accuracy_score,
    roc_auc_score,
)

# %%
X = pd.read_csv(pkg_resources.files(data) / "X.csv", index_col=0)
y = pd.read_csv(pkg_resources.files(data) / "y.csv", index_col=0)
X = X.reset_index()
y = np.ravel(y)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

# %%


def objective(trial):
    # Define hyperparameters
    params = {
        'objective': 'binary',
        'max_bin': 1023,
        'n_estimators': 500,
        'verbosity': -1,
        'boosting_type': 'dart',
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 2, 128),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1.0),
        'bagging_fraction': trial.suggest_float("bagging_fraction", 0.4, 1.0),
        'bagging_freq': trial.suggest_int("bagging_freq", 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 1.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 1.0),
        'is_unbalance': trial.suggest_categorical('is_unbalance', [True, False]),
        'metric': 'auc'
    }

    # Train model
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")
    model = lgb.train(params, lgb_train, valid_sets=lgb_val, num_boost_round=1000,
                      callbacks=[pruning_callback])

    # Return metrics
    y_pred = model.predict(X_val)
    y_pred = np.rint(y_pred)
    y_pred[y_pred == 0] = -1
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, average="weighted")
    roc = roc_auc_score(y_val, y_pred, average="weighted")
    # Return accuracy on validation set
    return acc


# %%
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000, show_progress_bar=True)

# %%
