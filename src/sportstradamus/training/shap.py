"""SHAP-based feature importance computation and feature filter management."""

import importlib.resources as pkg_resources

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from sportstradamus import data
from sportstradamus import feature_selection as fs
from sportstradamus.helpers import feature_filter

# Distribution-specific columns to drop before SHAP analysis
_DIST_DROP_COLS = {
    "Gamma": ["Alpha"],
    "ZAGamma": ["Alpha", "Gate"],
    "NegBin": ["R", "NB_P"],
    "ZINB": ["R", "NB_P", "Gate"],
    "Mixture2G": ["STD"],
}


def _compute_shap_and_corr(model, test_df, distribution):
    """SHAP |val| + Pearson corr for one market test set. Returns (shap_dict_pct, corr_dict)."""
    X = test_df.drop(columns=["Result", "EV", "P"], errors="ignore")
    C = X.corrwith(test_df["Result"])

    drop_cols = _DIST_DROP_COLS.get(distribution, [])
    X.drop(columns=drop_cols, inplace=True, errors="ignore")
    C.drop(drop_cols, inplace=True, errors="ignore")

    features = X.columns
    for c in ("Home", "Player position"):
        if c in features:
            X[c] = X[c].astype("category")

    explainer = shap.TreeExplainer(model.booster)
    subvals = explainer.shap_values(X)
    if isinstance(subvals, list):
        subvals = np.sum([np.abs(sv) for sv in subvals], axis=0)

    vals = np.mean(np.abs(subvals), axis=0)
    total = np.sum(vals)
    if total > 0:
        vals = vals / total * 100

    return dict(zip(features, vals, strict=False)), C.to_dict()


def _refresh_all_aggregates(shap_df):
    """Rebuild ALL aggregate columns (league-level and global) in the SHAP DataFrame."""
    for col in [c for c in shap_df.columns if c.endswith("_ALL") or c == "ALL"]:
        shap_df.drop(columns=[col], inplace=True)
    for league in ["NBA", "WNBA", "NFL", "NHL", "MLB"]:
        cols = [c for c in shap_df.columns if c.startswith(league + "_")]
        if cols:
            shap_df[league + "_ALL"] = shap_df[cols].mean(axis=1)
    all_cols = [c for c in shap_df.columns if c.endswith("_ALL")]
    if all_cols:
        shap_df["ALL"] = shap_df[all_cols].mean(axis=1)
    return shap_df


def compute_market_importance(league, market, model, test_df, distribution):
    """Update one market column in feature_importances.csv + feature_correlations.csv.
    test_df must contain Result + features + (any dist params).
    """
    shap_dict, corr_dict = _compute_shap_and_corr(model, test_df, distribution)

    col_name = f"{league}_{market.replace(' ', '-')}"
    shap_path = pkg_resources.files(data) / "feature_importances.csv"
    corr_path = pkg_resources.files(data) / "feature_correlations.csv"

    shap_df = pd.read_csv(shap_path, index_col=0) if shap_path.is_file() else pd.DataFrame()
    corr_df = pd.read_csv(corr_path, index_col=0) if corr_path.is_file() else pd.DataFrame()

    if col_name in shap_df.columns:
        shap_df.drop(columns=[col_name], inplace=True)
    if col_name in corr_df.columns:
        corr_df.drop(columns=[col_name], inplace=True)

    shap_df = shap_df.join(pd.Series(shap_dict, name=col_name), how="outer").fillna(0)
    corr_df = corr_df.join(pd.Series(corr_dict, name=col_name), how="outer")

    shap_df = _refresh_all_aggregates(shap_df)

    shap_df.to_csv(shap_path)
    corr_df.to_csv(corr_path)


def see_features():
    """Batch: rebuild full feature_importances.csv + feature_correlations.csv from all saved models."""
    import pickle

    from tqdm import tqdm

    model_list = sorted(
        f.name for f in (pkg_resources.files(data) / "models").iterdir() if ".mdl" in f.name
    )
    feature_importances = []
    feature_correlations = []
    for model_str in tqdm(model_list, desc="Analyzing feature importances...", unit="market"):
        with open(pkg_resources.files(data) / f"models/{model_str}", "rb") as infile:
            filedict = pickle.load(infile)
        test_path = pkg_resources.files(data) / ("test_sets/" + model_str.replace(".mdl", ".csv"))
        test_df = pd.read_csv(test_path, index_col=0)
        shap_dict, corr_dict = _compute_shap_and_corr(
            filedict["model"], test_df, filedict["distribution"]
        )
        feature_importances.append(shap_dict)
        feature_correlations.append(corr_dict)

    df = (
        pd.DataFrame(feature_importances, index=[m[:-4] for m in model_list])
        .fillna(0)
        .infer_objects(copy=False)
        .transpose()
    )
    df = _refresh_all_aggregates(df)
    df.to_csv(pkg_resources.files(data) / "feature_importances.csv")
    pd.DataFrame(feature_correlations, index=[m[:-4] for m in model_list]).T.to_csv(
        pkg_resources.files(data) / "feature_correlations.csv"
    )


def _load_shap_corr_dfs():
    """Read SHAP + corr CSVs, drop ALL aggregate cols, return (shap_df, corr_df)."""
    sp = pkg_resources.files(data) / "feature_importances.csv"
    cp = pkg_resources.files(data) / "feature_correlations.csv"
    shap_df = pd.read_csv(sp, index_col=0) if sp.is_file() else pd.DataFrame()
    corr_df = pd.read_csv(cp, index_col=0) if cp.is_file() else pd.DataFrame()
    if not shap_df.empty:
        drop = [c for c in shap_df.columns if c == "ALL" or c.endswith("_ALL")]
        shap_df = shap_df.drop(columns=drop, errors="ignore").fillna(0)
    return shap_df, corr_df


def _save_feature_filter() -> None:
    """Write the current feature_filter dict to feature_filter.json."""
    import json

    with open(pkg_resources.files(data) / "feature_filter.json", "w") as outfile:
        json.dump(feature_filter, outfile, indent=4)


def _scouting_shap_and_filter(league, market, M, stat_data):
    """Train fixed-HP regression LightGBM on unfiltered features, write SHAP+corr columns
    via compute_market_importance, rewrite Filtered bucket via filter_market.
    Used only under --rebuild-filter, before the final Optuna training pass.
    """
    cols = stat_data.get_stat_columns(market, unfiltered=True)
    cols = [c for c in cols if c in M.columns]
    if not cols or "Result" not in M.columns:
        return None

    X = M[cols].copy()
    for c in ("Home", "Player position"):
        if c in X.columns:
            X[c] = X[c].astype("category")

    y = pd.to_numeric(M["Result"], errors="coerce").fillna(0).to_numpy()

    M_sorted = M.sort_values("Date")
    n_train = int(len(M_sorted) * 0.7)
    if n_train < 50:
        return None
    train_idx = M_sorted.index[:n_train]
    test_idx = M_sorted.index[n_train:]
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    pos_map = {ix: i for i, ix in enumerate(M.index)}
    y_train = y[[pos_map[i] for i in train_idx]]
    y_test = y[[pos_map[i] for i in test_idx]]

    dtrain = lgb.Dataset(
        X_train,
        label=y_train,
        free_raw_data=False,
        categorical_feature=[c for c in ("Home", "Player position") if c in X_train.columns],
    )
    params = dict(
        objective="regression",
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        verbose=-1,
        num_threads=8,
    )
    booster = lgb.train(params, dtrain, num_boost_round=200)

    class _Shim:
        pass

    shim = _Shim()
    shim.booster = booster

    test_df = X_test.copy()
    test_df["Result"] = y_test
    compute_market_importance(league, market, shim, test_df, distribution="regression")
    return filter_market(league, market)


def filter_market(league, market):
    """Per-market filter rebuild. Updates feature_filter[league]['Filtered'][market]
    in memory + on disk. Returns diagnostic dict.
    """
    shap_df, corr_df = _load_shap_corr_dfs()
    new_filtered, diag = fs.filter_market_features(league, market, feature_filter, shap_df, corr_df)

    # `feature_filter` is the same dict object as helpers.feature_filter (imported
    # by reference). Mutating it here is what `get_stat_columns` will see on the
    # next call — no clear/reload needed.
    feature_filter.setdefault(league, {})
    feature_filter[league].setdefault("Common", [])
    feature_filter[league].setdefault("Always", {"_default": []})
    feature_filter[league].setdefault("Filtered", {})
    feature_filter[league]["Filtered"][market] = new_filtered
    _save_feature_filter()
    return diag


def filter_features():
    """Batch: rebuild Filtered bucket for every (league, market) seen in SHAP CSV.
    Reuses per-market path so logic stays single-source.
    """
    shap_df, _ = _load_shap_corr_dfs()
    if shap_df.empty:
        return
    seen = set()
    for col in shap_df.columns:
        if "_" not in col:
            continue
        league, raw_market = col.split("_", 1)
        market = raw_market.replace("-", " ")
        if (league, market) in seen:
            continue
        seen.add((league, market))
        filter_market(league, market)
