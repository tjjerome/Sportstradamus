# %%
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import optuna
import importlib.resources as pkg_resources
from sportstradamus import data
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
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")
    model = lgb.train(params, lgb_train, valid_sets=lgb_val,
                      callbacks=[pruning_callback, early_stopping])

    # Return metrics
    y_pred = model.predict(X_val)
    y_pred = np.rint(y_pred)
    y_pred[y_pred == 0] = -1
    acc = accuracy_score(y_val, y_pred)
    # Return accuracy on validation set
    return acc


# %%
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000, n_jobs=-1)
# %%
print(study.best_params)
# %%
