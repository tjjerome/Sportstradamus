"""Shared feature-selection helpers.

Used by training pipeline (`filter_market` per-market rebuild under
`--rebuild-filter`) and the interactive `evaluate_model_features` script.
Single source of truth for composite scoring math.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression

from sportstradamus import data

warnings.filterwarnings("ignore")


# ─── Constants ─────────────────────────────────────────────────────────────────

META_COLS = {
    "Date",
    "Player",
    "Player team",
    "Result",
    "Line",
    "Odds",
    "EV",
    "Archived",
    "Blended_EV",
    "P",
    "Gate",
    "R",
    "NB_P",
    "Alpha",
    "SN_Loc",
    "SN_Scale",
    "SN_Alpha",
    "STD",
}

# Composite weights — same as evaluate_model_features.py
W_MODEL = dict(shap=0.35, corr=0.25, mi=0.20, stability=0.20)
W_NO_MODEL = dict(shap=0.00, corr=0.40, mi=0.35, stability=0.25)
REDUNDANCY_WEIGHT = 0.40

# Conservative defaults — err toward keeping features.
DROP_CUTOFF = 0.10  # composite below this is hard-dropped
ADD_THRESHOLD = 0.50  # candidate must score this high to come back
KEEP_FLOOR = 25  # always keep top-N regardless of score
KEEP_CAP = 60  # never keep more than this many Filtered features

# Eval-script-style defaults (interactive review). Kept here so the interactive
# script can import and override the conservative training-time defaults.
EVAL_DROP_CUTOFF = 0.30
EVAL_ADD_THRESHOLD = 0.38
EVAL_TOP_CANDIDATES = 10


# ─── Variant helpers ───────────────────────────────────────────────────────────


def strip_variant(col: str) -> str:
    return col.replace(" short", "").replace(" growth", "")


def variants_of(base: str, available_cols) -> list[str]:
    """Existing variants of a base feature: base, base short, base growth."""
    return [v for v in (base, base + " short", base + " growth") if v in available_cols]


# ─── Signal computation ────────────────────────────────────────────────────────


def compute_shap_scores(features: list[str], mkey: str, shap_df: pd.DataFrame) -> dict[str, float]:
    """Aggregate |SHAP| across base + short + growth variants per feature."""
    if shap_df.empty or mkey not in shap_df.columns:
        return {f: 0.0 for f in features}
    col = shap_df[mkey]
    out = {}
    for feat in features:
        variants = variants_of(feat, set(col.index))
        out[feat] = col[variants].fillna(0).sum() if variants else 0.0
    return out


def compute_corr_scores(features: list[str], mkey: str, corr_df: pd.DataFrame) -> dict[str, float]:
    """Max |corr| across variants from precomputed correlations CSV."""
    if corr_df.empty or mkey not in corr_df.columns:
        return {f: 0.0 for f in features}
    col = corr_df[mkey].abs()
    out = {}
    for feat in features:
        variants = variants_of(feat, set(col.index))
        out[feat] = col[variants].max() if variants else 0.0
    return out


def compute_from_training(
    features: list[str], train_df: pd.DataFrame, target: pd.Series
) -> tuple[dict, dict, dict]:
    """Corr + MI + temporal stability from training matrix.

    Returns (corr_scores, mi_scores, stability_scores).
    """
    if train_df.empty or len(target) < 50:
        zero = {f: 0.0 for f in features}
        return zero, zero, zero

    all_cols = set(train_df.columns)
    n = len(train_df)
    mid = n // 2

    # Best variant per feature = max abs-corr to target
    best_variant: dict[str, str] = {}
    for feat in features:
        cands = variants_of(feat, all_cols)
        if not cands:
            continue
        best_c, best_r = cands[0], 0.0
        for c in cands:
            r = train_df[c].corr(target)
            if abs(r) > best_r:
                best_c, best_r = c, abs(r)
        best_variant[feat] = best_c

    mi_input_cols = list(best_variant.values())
    if not mi_input_cols:
        zero = {f: 0.0 for f in features}
        return zero, zero, zero

    X_mi = train_df[mi_input_cols].copy()
    X_mi = X_mi.apply(pd.to_numeric, errors="coerce").fillna(0)

    mi_raw = mutual_info_regression(X_mi, target, random_state=42)
    mi_map = dict(zip(mi_input_cols, mi_raw, strict=False))

    corr_scores, mi_scores, stab_scores = {}, {}, {}
    for feat in features:
        cands = variants_of(feat, all_cols)
        if not cands:
            corr_scores[feat] = mi_scores[feat] = stab_scores[feat] = 0.0
            continue

        corr_scores[feat] = max((abs(train_df[c].corr(target)) for c in cands), default=0.0)

        bv = best_variant.get(feat, cands[0])
        mi_scores[feat] = mi_map.get(bv, 0.0)

        col_data = train_df[bv]
        early, late = col_data.iloc[:mid], col_data.iloc[mid:]
        t_early, t_late = target.iloc[:mid], target.iloc[mid:]
        r_early, _ = spearmanr(early.fillna(0), t_early)
        r_late, _ = spearmanr(late.fillna(0), t_late)
        re, rl = abs(r_early or 0), abs(r_late or 0)
        stab_scores[feat] = min(re, rl) / max(re, rl) if max(re, rl) > 0 else 0.0

    return corr_scores, mi_scores, stab_scores


def compute_redundancy(features: list[str], train_df: pd.DataFrame) -> dict[str, float]:
    """Max |corr| against all OTHER features. High = collinear."""
    if train_df.empty:
        return {f: 0.0 for f in features}
    all_cols = set(train_df.columns)
    feat_col = {}
    for feat in features:
        for v in (feat, feat + " short", feat + " growth"):
            if v in all_cols:
                feat_col[feat] = v
                break

    if not feat_col:
        return {f: 0.0 for f in features}

    sub = train_df[list(feat_col.values())].apply(pd.to_numeric, errors="coerce")
    corr_matrix = sub.corr().abs()
    rev = {v: k for k, v in feat_col.items()}
    corr_matrix = corr_matrix.rename(index=rev, columns=rev)

    out = {}
    for feat in features:
        if feat not in corr_matrix.index:
            out[feat] = 0.0
            continue
        row = corr_matrix.loc[feat].drop(labels=[feat], errors="ignore")
        out[feat] = row.max() if len(row) else 0.0
    return out


# ─── Composite ────────────────────────────────────────────────────────────────


def normalize_scores(score_dict: dict[str, float]) -> dict[str, float]:
    """Min-max normalize to [0, 1]. Constant vectors → 0.5."""
    if not score_dict:
        return {}
    vals = np.array(list(score_dict.values()))
    lo, hi = vals.min(), vals.max()
    if hi <= lo:
        return {k: 0.5 for k in score_dict}
    return {k: (v - lo) / (hi - lo) for k, v in score_dict.items()}


def composite_score(
    feat: str,
    shap_n: dict,
    corr_n: dict,
    mi_n: dict,
    stab_n: dict,
    redundancy: float,
    has_model: bool,
) -> float:
    w = W_MODEL if has_model else W_NO_MODEL
    raw = (
        w["shap"] * shap_n.get(feat, 0)
        + w["corr"] * corr_n.get(feat, 0)
        + w["mi"] * mi_n.get(feat, 0)
        + w["stability"] * stab_n.get(feat, 0)
    )
    return raw * (1.0 - REDUNDANCY_WEIGHT * redundancy)


# ─── Candidate discovery ─────────────────────────────────────────────────────


def discover_candidates(train_df: pd.DataFrame, locked: set[str], current: set[str]) -> list[str]:
    """Base feature names in training data not in `locked` (Common+Always)
    and not in `current` (Filtered). Drops near-constant columns.
    """
    if train_df.empty:
        return []
    all_cols = set(train_df.columns) - META_COLS
    base_names = {strip_variant(c) for c in all_cols}
    candidates = sorted(base_names - locked - current)
    out = []
    for feat in candidates:
        col = next(
            (c for c in (feat, feat + " short", feat + " growth") if c in train_df.columns), None
        )
        if col is None:
            continue
        vals = pd.to_numeric(train_df[col], errors="coerce").dropna()
        if len(vals) > 10 and vals.std() > 0:
            out.append(feat)
    return out


# ─── Per-market batch entry point ────────────────────────────────────────────


def market_key(league: str, market: str) -> str:
    return f"{league}_{market.replace(' ', '-')}"


def model_path(league: str, market: str):
    fn = f"{league}_{market}".replace(" ", "-")
    return pkg_resources.files(data) / f"models/{fn}.mdl"


def training_path(league: str, market: str):
    fn = f"{league}_{market}".replace(" ", "-")
    return pkg_resources.files(data) / f"training_data/{fn}.csv"


def _score_features(
    features: list[str],
    mkey: str,
    shap_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    train_df: pd.DataFrame,
    target: pd.Series,
    has_model: bool,
) -> tuple[dict, dict]:
    """Return (composite_scores, signal_breakdown)."""
    if not features:
        return {}, {}
    shap_raw = (
        compute_shap_scores(features, mkey, shap_df) if has_model else {f: 0.0 for f in features}
    )
    corr_pre = compute_corr_scores(features, mkey, corr_df)
    corr_train, mi_raw, stab_raw = compute_from_training(features, train_df, target)
    corr_raw = {f: max(corr_pre.get(f, 0), corr_train.get(f, 0)) for f in features}
    redund = compute_redundancy(features, train_df)

    shap_n = normalize_scores(shap_raw)
    corr_n = normalize_scores(corr_raw)
    mi_n = normalize_scores(mi_raw)
    stab_n = normalize_scores(stab_raw)

    scores = {}
    breakdown = {}
    for feat in features:
        r = redund.get(feat, 0.0)
        s = composite_score(feat, shap_n, corr_n, mi_n, stab_n, r, has_model)
        scores[feat] = s
        breakdown[feat] = dict(
            shap=shap_raw.get(feat, 0.0),
            corr=corr_raw.get(feat, 0.0),
            mi=mi_raw.get(feat, 0.0),
            stability=stab_raw.get(feat, 0.0),
            redundancy=r,
            composite=s,
        )
    return scores, breakdown


def filter_market_features(
    league: str,
    market: str,
    feature_filter: dict,
    shap_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    drop_cutoff: float = DROP_CUTOFF,
    add_threshold: float = ADD_THRESHOLD,
    keep_floor: int = KEEP_FLOOR,
    keep_cap: int = KEEP_CAP,
) -> tuple[list[str], dict]:
    """Compute new Filtered list for one market.

    Returns (new_filtered_list, diagnostic_dict). Mutates nothing.

    Pipeline:
      1. Score current Filtered features (composite of SHAP+corr+MI+stability,
         penalized by redundancy).
      2. Discover candidate features from training data not in Common/Always/Filtered;
         score them by corr+MI+stability only.
      3. Apply 3-gate decision: keep-floor (top-N always), drop-cutoff (hard floor),
         add-threshold (candidates only above this score), keep-cap (max size).
    """
    league_filter = feature_filter.get(league, {})
    common = set(league_filter.get("Common", []))
    always = league_filter.get("Always", {})
    market_key_clean = market.replace("-", " ")
    locked = common | set(always.get("_default", [])) | set(always.get(market_key_clean, []))
    current = list(league_filter.get("Filtered", {}).get(market_key_clean, []))
    # Strip any locked features from current (they're added back by get_stat_columns)
    current = [f for f in current if f not in locked]

    # Load training data + target
    tpath = training_path(league, market)
    if os.path.isfile(tpath):
        train_df = pd.read_csv(tpath, index_col=0)
        if "Date" in train_df.columns:
            train_df = train_df.sort_values("Date").reset_index(drop=True)
        target = (
            pd.to_numeric(train_df["Result"], errors="coerce").fillna(0)
            if "Result" in train_df.columns
            else pd.Series(dtype=float)
        )
        train_df = train_df.drop(
            columns=[c for c in META_COLS if c in train_df.columns], errors="ignore"
        )
    else:
        train_df = pd.DataFrame()
        target = pd.Series(dtype=float)

    mkey = market_key(league, market)
    has_model = os.path.isfile(model_path(league, market))

    # Score existing
    cur_scores, cur_breakdown = _score_features(
        current, mkey, shap_df, corr_df, train_df, target, has_model
    )

    # Discover + score candidates (no SHAP signal — they were not in trained model)
    candidates = discover_candidates(train_df, locked, set(current))
    cand_scores, cand_breakdown = _score_features(
        candidates, mkey, shap_df, corr_df, train_df, target, has_model=False
    )

    # Decision: rank current by composite; keep top floor; then add anything
    # at-or-above drop_cutoff; then add candidates >= add_threshold; cap at max.
    ranked_current = sorted(cur_scores.items(), key=lambda kv: -kv[1])

    keep = set()
    # Layer 1: floor
    for feat, _ in ranked_current[:keep_floor]:
        keep.add(feat)
    # Layer 2: anything above cutoff
    for feat, score in ranked_current:
        if score >= drop_cutoff:
            keep.add(feat)
    # Layer 3: candidates above add threshold
    ranked_cand = sorted(cand_scores.items(), key=lambda kv: -kv[1])
    for feat, score in ranked_cand:
        if score < add_threshold:
            break
        keep.add(feat)
    # Layer 4: cap (drop lowest-scoring beyond cap)
    if len(keep) > keep_cap:
        merged = {**cur_scores, **cand_scores}
        ranked_keep = sorted(keep, key=lambda f: -merged.get(f, 0.0))
        keep = set(ranked_keep[:keep_cap])

    new_filtered = sorted(keep)

    diag = dict(
        n_current=len(current),
        n_candidates=len(candidates),
        n_kept=len(new_filtered),
        n_dropped=len([f for f in current if f not in keep]),
        n_added=len([f for f in new_filtered if f not in current]),
        has_model=has_model,
        n_training=len(target),
        scores=cur_breakdown,
        candidate_scores=cand_breakdown,
    )
    return new_filtered, diag
