# %%
from lightgbmlss.model import *
from lightgbmlss.distributions.Gaussian import *
from sklearn.model_selection import train_test_split
import pandas as pd
from sportstradamus.drafts import data
import importlib.resources as pkg_resources
import numpy as np
from tqdm import tqdm
import lightgbm as lgb

import plotnine
from plotnine import *
plotnine.options.figure_size = (20, 10)
# %%
data_list = [f.name for f in pkg_resources.files(
    data).iterdir() if "teams.csv" in f.name]

df = pd.DataFrame()
for data_str in tqdm(data_list, desc='Loading Data Files', unit='file'):
    if df.empty:
        df = pd.read_csv(pkg_resources.files(
            data) / data_str, index_col=(0, 1))
    else:
        df = pd.concat(
            [df, pd.read_csv(pkg_resources.files(data) / data_str, index_col=(0, 1))])

# %%
df['Z'] = 0
df['NumSpikes'] = 0
for week in np.arange(1, 18):
    mu = df[f"W{week}_Points"].mean()
    sig = df[f"W{week}_Points"].std()
    df['Z'] = df['Z'] + ((df[f"W{week}_Points"]-mu)/sig - 91).clip(lower=1)
    df['NumSpikes'] = df['NumSpikes'] + \
        ((df[f"W{week}_Points"]-mu)/sig > 1.91).astype(int)
# %%
X = df.iloc[:, :36]
feature_names = X.columns
X = X.to_numpy()
y = df['Z'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)
dtrain = lgb.Dataset(X_train, label=y_train)
# %%
# candidate_distributions = [Gaussian, Cauchy, LogNormal, Gumbel]
lgblss = LightGBMLSS(
    Gaussian(stabilization="None",   # Options are "None", "MAD", "L2".
             # Function to transform the scale-parameter, e.g., "exp" or "softplus".
             response_fn="exp",
             # Loss function. Options are "nll" (negative log-likelihood) or "crps"(continuous ranked probability score).
             loss_fn="nll"
             )
)

# %%
param_dict = {
    # "eta":                      ["float", {"low": 1e-5,   "high": 1,     "log": True}],
    "max_depth":                ["int",   {"low": 2,      "high": 63,    "log": False}],
    # set to constant for this example
    "num_leaves":               ["int",   {"low": 7,    "high": 4095,   "log": False}],
    # set to constant for this example
    # "min_data_in_leaf":         ["int",   {"low": 20,     "high": 20,    "log": False}],
    # "min_gain_to_split":        ["float", {"low": 1e-8,   "high": 40,    "log": False}],
    "min_sum_hessian_in_leaf":  ["float", {"low": 1e-2,   "high": len(X)*.8/1000,   "log": True}],
    "subsample":                ["float", {"low": 0.4,    "high": 1.0,   "log": False}],
    "feature_fraction":         ["float", {"low": 0.4,    "high": 1.0,   "log": False}],
    "boosting":                 ["categorical", ["gbdt"]],
}

np.random.seed(123)
opt_param = lgblss.hyper_opt(param_dict,
                             dtrain,
                             # Number of boosting iterations.
                             num_boost_round=100,
                             nfold=5,                    # Number of cv-folds.
                             early_stopping_rounds=20,   # Number of early-stopping rounds
                             # Time budget in minutes, i.e., stop study after the given number of minutes.
                             max_minutes=10,
                             # The number of trials. If this argument is set to None, there is no limitation on the number of trials.
                             n_trials=None,
                             # Controls the verbosity of the trail, i.e., user can silence the outputs of the trail.
                             silence=False,
                             # Seed used to generate cv-folds.
                             seed=123,
                             # Seed for random number generator used in the Bayesian hyperparameter search.
                             hp_seed=None
                             )

# %%
np.random.seed(123)

opt_params = opt_param.copy()
n_rounds = opt_params["opt_rounds"]
del opt_params["opt_rounds"]

# Train Model with optimized hyperparameters
lgblss.train(opt_params,
             dtrain,
             num_boost_round=n_rounds
             )
# %%
pred_params = lgblss.predict(X_test,
                             pred_type="parameters")

# %%
pdp_df = pd.DataFrame(X_train, columns=feature_names)
lgblss.plot(pdp_df,
            parameter="rate",
            feature=feature_names[0],
            plot_type="Feature_Importance")
