"""Post-training SHAP importance diagnostics.

After the 2026-05-27 no-filter rewire, this module no longer drives feature
selection — ``get_stat_columns`` returns the full unfiltered candidate set
unconditionally. The remaining job is drift monitoring: compute |SHAP| per
feature on the held-out test set after each model trains and stash the result
in ``data/training/feature_importances.csv`` so the dashboard can show
importance trends over time.
"""

import importlib.resources as pkg_resources

import numpy as np
import pandas as pd
import shap

from sportstradamus import data


def _compute_shap_and_corr(model, test_df):
    """SHAP |val| + Pearson corr for one market test set. Returns (shap_dict_pct, corr_dict)."""
    # Allowlist from the trained booster — ignores diagnostic columns that
    # _step_persist_artifacts appends to test_df after fit.
    features = list(model.booster.feature_name())
    X = test_df[features].copy()
    C = X.corrwith(test_df["Result"])

    for c in ("Home", "Player position"):
        if c in X.columns:
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


def compute_market_importance(league: str, market: str, model, test_df) -> None:
    """Update one market column in feature_importances.csv + feature_correlations.csv.
    test_df must contain Result + features + (any dist params).
    """
    shap_dict, corr_dict = _compute_shap_and_corr(model, test_df)

    col_name = f"{league}_{market.replace(' ', '-')}"
    shap_path = pkg_resources.files(data) / "training" / "feature_importances.csv"
    corr_path = pkg_resources.files(data) / "training" / "feature_correlations.csv"

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


def see_features() -> None:
    """Batch-rebuild feature_importances.csv + feature_correlations.csv from all saved models."""
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
        shap_dict, corr_dict = _compute_shap_and_corr(filedict["model"], test_df)
        feature_importances.append(shap_dict)
        feature_correlations.append(corr_dict)

    df = (
        pd.DataFrame(feature_importances, index=[m[:-4] for m in model_list])
        .fillna(0)
        .infer_objects(copy=False)
        .transpose()
    )
    df = _refresh_all_aggregates(df)
    df.to_csv(pkg_resources.files(data) / "training" / "feature_importances.csv")
    pd.DataFrame(feature_correlations, index=[m[:-4] for m in model_list]).T.to_csv(
        pkg_resources.files(data) / "training" / "feature_correlations.csv"
    )


