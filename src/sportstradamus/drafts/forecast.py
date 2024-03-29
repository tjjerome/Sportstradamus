# %%
from lightgbmlss.model import *
from lightgbmlss.distributions.ZIPoisson import *
from sklearn.model_selection import train_test_split
import pandas as pd
from sportstradamus.drafts import data
import importlib.resources as pkg_resources
import numpy as np
from tqdm import tqdm
import lightgbm as lgb
import pickle
from matplotlib import pyplot as plt

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
    df['Z'] = df['Z'] + ((df[f"W{week}_Points"]-mu)/sig - 1.91).clip(lower=0)
    df['NumSpikes'] = df['NumSpikes'] + \
        ((df[f"W{week}_Points"]-mu)/sig > 1.91).astype(int)
# %%
X = df.iloc[:, :36]
feature_names = X.columns
X = X
y = df['NumSpikes']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)
dtrain = lgb.Dataset(X_train, label=y_train)
# %%
# candidate_distributions = [Gaussian, Cauchy, LogNormal, Gumbel]
lgblss = LightGBMLSS(
    ZIPoisson(stabilization="None",   # Options are "None", "MAD", "L2".
              # Function to transform the scale-parameter, e.g., "exp" or "softplus".
              response_fn="softplus",
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
                             num_boost_round=99999,
                             nfold=5,                    # Number of cv-folds.
                             early_stopping_rounds=20,   # Number of early-stopping rounds
                             # Time budget in minutes, i.e., stop study after the given number of minutes.
                             max_minutes=20,
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
# Save to file
with open(pkg_resources.files(data) / "WW.mdl", 'wb') as outfile:
    pickle.dump(lgblss, outfile)

# %%
pred_params = lgblss.predict(X_test,
                             pred_type="parameters")

# %%
# pdp_df = pd.DataFrame(X_test, columns=feature_names)
# Feature Importance of gate parameter
lgblss.plot(X_test,
            parameter="rate",
            plot_type="Feature_Importance")

# %%
# Partial Dependence Plot of rate parameter
lgblss.plot(X.loc[(X['NumQB'] < 3) & (X['NumRB'] > 4)],
            parameter="rate",
            feature="PriceRB",
            plot_type="Partial_Dependence")

# %%
n_bins = 10
data = df.loc[(df['NumQB'] < 3) & (df['NumTE'] < 5) & (df['NumWR'] < 11) & (
    df['NumWR'] > 5) & (df['NumRB'] > 3) & (df['NumRB'] < 8)]
X = data['draftValue1318']
if X.nunique() > n_bins:
    x, edges = pd.cut(X, n_bins, labels=np.arange(n_bins, 0, -1), retbins=True)
    edges = (edges[1:]+edges[:-1])/2
else:
    x = X
    edges = sorted(x.unique())
Y = data['NumSpikes']
y = pd.DataFrame(index=sorted(x.unique()), columns=['val'])
for i in x.unique():
    y.loc[i] = Y.loc[x == i].mean()

fig, ax1 = plt.subplots()

ax1.hist(X, bins=n_bins, color='tab:orange', alpha=.2)

ax2 = ax1.twinx()
ax2.plot(edges, y, marker='.')

fig.tight_layout()
plt.show()
# %%
