#!/usr/bin/env python3
"""
Evaluate feature sets for LightGBMLSS model training (feature_filter.json).

For each league/market, scores every market-specific feature using several
complementary signals and recommends which to keep, drop, or add.

Signals used
------------
  SHAP importance    – aggregate |SHAP| across base + short + growth variants,
                       from feature_importances.csv (requires a trained model)
  Correlation        – max |Pearson| against the target across variants,
                       from feature_correlations.csv or computed on-the-fly
  Mutual information – non-linear target association via sklearn MI regression,
                       computed from training data
  Temporal stability – min(|Spearman early|, |Spearman late|) / max(…),
                       how consistently the feature correlates across time
  Redundancy         – max |Pearson| of this feature against every OTHER current
                       feature; high redundancy penalises the composite score

Composite score (0–1)
  With model:    0.35·SHAP + 0.25·corr + 0.20·MI + 0.20·stability
  Without model: 0.40·corr + 0.35·MI   + 0.25·stability
The composite is then multiplied by (1 – 0.4·redundancy).

Common features are never evaluated or modified.
Model signals cannot evaluate features OUTSIDE the current filter list;
for candidate additions only correlation, MI, and stability are used.

Usage
-----
  poetry run python3 evaluate_model_features.py --league NBA
  poetry run python3 evaluate_model_features.py --league NFL --market targets passing-yards
  poetry run python3 evaluate_model_features.py --league NBA --save
  poetry run python3 evaluate_model_features.py --league NBA --threshold 0.25 --top-candidates 15
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import pickle
import warnings
import argparse
import importlib.resources as pkg_resources

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import spearmanr
from sportstradamus import data

warnings.filterwarnings("ignore")


# ─── Constants ─────────────────────────────────────────────────────────────────

META_COLS = {"Date", "Player", "Player team", "Result", "Line", "Odds", "EV"}

# Weights for composite scoring
W_MODEL   = dict(shap=0.35, corr=0.25, mi=0.20, stability=0.20)
W_NO_MODEL = dict(shap=0.00, corr=0.40, mi=0.35, stability=0.25)
REDUNDANCY_WEIGHT = 0.40   # how much to penalise for collinearity with peers

DEFAULT_DROP_THRESHOLD = 0.30    # features scoring below this are drop candidates
DEFAULT_ADD_THRESHOLD  = 0.38    # candidates scoring above this are add candidates
DEFAULT_TOP_CANDIDATES = 10      # max candidate additions to show per market


# ─── Helpers ───────────────────────────────────────────────────────────────────

def market_key(league: str, market: str) -> str:
    """Column name used in feature_importances / feature_correlations CSVs."""
    return f"{league}_{market.replace(' ', '-')}"


def model_path(league: str, market: str) -> str:
    filename = f"{league}_{market}".replace(" ", "-")
    return pkg_resources.files(data) / f"models/{filename}.mdl"


def training_path(league: str, market: str) -> str:
    filename = f"{league}_{market}".replace(" ", "-")
    return pkg_resources.files(data) / f"training_data/{filename}.csv"


def strip_variant(col: str) -> str:
    return col.replace(" short", "").replace(" growth", "")


def variants_of(base: str, available_cols) -> list[str]:
    """Return existing variants: base, base short, base growth."""
    return [v for v in [base, base + " short", base + " growth"]
            if v in available_cols]


def load_csv_safe(path):
    if os.path.isfile(path):
        return pd.read_csv(path, index_col=0)
    return pd.DataFrame()


# ─── Signal computation ────────────────────────────────────────────────────────

def compute_shap_scores(features: list[str], mkey: str,
                        shap_df: pd.DataFrame) -> dict[str, float]:
    """
    Aggregate SHAP importance across base + short + growth for each feature.
    Returns a dict {feature: raw_shap_sum}.  0.0 if column absent.
    """
    if shap_df.empty or mkey not in shap_df.columns:
        return {f: 0.0 for f in features}
    col = shap_df[mkey]
    scores = {}
    for feat in features:
        variants = [v for v in [feat, feat + " short", feat + " growth"]
                    if v in col.index]
        scores[feat] = col[variants].fillna(0).sum() if variants else 0.0
    return scores


def compute_corr_scores(features: list[str], mkey: str,
                        corr_df: pd.DataFrame) -> dict[str, float]:
    """
    Max |correlation| across variants from pre-computed correlation CSV.
    Falls back to 0 if unavailable.
    """
    if corr_df.empty or mkey not in corr_df.columns:
        return {f: 0.0 for f in features}
    col = corr_df[mkey].abs()
    scores = {}
    for feat in features:
        variants = [v for v in [feat, feat + " short", feat + " growth"]
                    if v in col.index]
        scores[feat] = col[variants].max() if variants else 0.0
    return scores


def compute_from_training(features: list[str], train_df: pd.DataFrame,
                          target: pd.Series) -> tuple[dict, dict, dict]:
    """
    Compute correlation, mutual information, and temporal stability from
    training data.  Returns (corr_scores, mi_scores, stability_scores).
    """
    if train_df.empty or len(target) < 50:
        zero = {f: 0.0 for f in features}
        return zero, zero, zero

    all_cols = set(train_df.columns)
    n = len(train_df)
    mid = n // 2

    # For MI we need a clean numeric matrix — use best variant per feature
    best_variant: dict[str, str] = {}
    for feat in features:
        cands = variants_of(feat, all_cols)
        if not cands:
            continue
        # pick the variant with highest abs-correlation to target
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

    # Mutual information
    mi_raw = mutual_info_regression(X_mi, target, random_state=42)
    mi_map = dict(zip(mi_input_cols, mi_raw))

    corr_scores, mi_scores, stability_scores = {}, {}, {}
    for feat in features:
        cands = variants_of(feat, all_cols)
        if not cands:
            corr_scores[feat] = mi_scores[feat] = stability_scores[feat] = 0.0
            continue

        # Correlation: max across variants
        corr_vals = [abs(train_df[c].corr(target)) for c in cands]
        corr_scores[feat] = max(corr_vals, default=0.0)

        # MI: from best variant
        bv = best_variant.get(feat, cands[0])
        mi_scores[feat] = mi_map.get(bv, 0.0)

        # Temporal stability: Spearman in first vs second half
        col_data = train_df[bv]
        early = col_data.iloc[:mid]
        late  = col_data.iloc[mid:]
        t_early = target.iloc[:mid]
        t_late  = target.iloc[mid:]
        r_early, _ = spearmanr(early.fillna(0), t_early)
        r_late, _  = spearmanr(late.fillna(0), t_late)
        re, rl = abs(r_early or 0), abs(r_late or 0)
        if max(re, rl) > 0:
            stability_scores[feat] = min(re, rl) / max(re, rl)
        else:
            stability_scores[feat] = 0.0

    return corr_scores, mi_scores, stability_scores


def compute_redundancy(features: list[str], train_df: pd.DataFrame) -> dict[str, float]:
    """
    For each feature, compute max |Pearson| against all OTHER features
    using the best available variant.  High values indicate collinearity.
    """
    if train_df.empty:
        return {f: 0.0 for f in features}
    all_cols = set(train_df.columns)
    # Map each feature to its best column (base preferred)
    feat_col = {}
    for feat in features:
        for variant in [feat, feat + " short", feat + " growth"]:
            if variant in all_cols:
                feat_col[feat] = variant
                break

    if not feat_col:
        return {f: 0.0 for f in features}

    sub = train_df[list(feat_col.values())].apply(pd.to_numeric, errors="coerce")
    corr_matrix = sub.corr().abs()
    # Rename back to base names
    rev = {v: k for k, v in feat_col.items()}
    corr_matrix = corr_matrix.rename(index=rev, columns=rev)

    redundancy = {}
    for feat in features:
        if feat not in corr_matrix.index:
            redundancy[feat] = 0.0
            continue
        row = corr_matrix.loc[feat].drop(labels=[feat], errors="ignore")
        redundancy[feat] = row.max() if len(row) else 0.0
    return redundancy


def normalize_scores(score_dict: dict[str, float]) -> dict[str, float]:
    """Min-max normalize a dict of scores to [0, 1]."""
    vals = np.array(list(score_dict.values()))
    lo, hi = vals.min(), vals.max()
    if hi <= lo:
        return {k: 0.5 for k in score_dict}
    return {k: (v - lo) / (hi - lo) for k, v in score_dict.items()}


def composite_score(feat: str, shap_n, corr_n, mi_n, stab_n,
                    redundancy: float, has_model: bool) -> float:
    w = W_MODEL if has_model else W_NO_MODEL
    raw = (w["shap"] * shap_n.get(feat, 0)
           + w["corr"] * corr_n.get(feat, 0)
           + w["mi"]   * mi_n.get(feat, 0)
           + w["stability"] * stab_n.get(feat, 0))
    return raw * (1.0 - REDUNDANCY_WEIGHT * redundancy)


# ─── Candidate discovery ───────────────────────────────────────────────────────

def discover_candidates(train_df: pd.DataFrame, common: set[str],
                        current: set[str]) -> list[str]:
    """
    Return base feature names present in training data but not in Common or
    the current market feature list.  Excludes metadata and near-constant cols.
    """
    if train_df.empty:
        return []
    all_cols = set(train_df.columns) - META_COLS
    base_names = {strip_variant(c) for c in all_cols}
    candidates = sorted(base_names - common - current)
    # Drop near-constant: std == 0 across base variant
    result = []
    for feat in candidates:
        col = next((c for c in [feat, feat + " short", feat + " growth"]
                    if c in train_df.columns), None)
        if col is None:
            continue
        vals = pd.to_numeric(train_df[col], errors="coerce").dropna()
        if len(vals) > 10 and vals.std() > 0:
            result.append(feat)
    return result


# ─── Per-market evaluation ─────────────────────────────────────────────────────

def evaluate_market(league: str, market: str, feature_filter: dict,
                    shap_df: pd.DataFrame, corr_df: pd.DataFrame,
                    drop_threshold: float, add_threshold: float,
                    top_candidates: int) -> dict:
    """
    Evaluate all features for one market.

    Returns a dict with:
      'current_scores'  – {feature: composite_score}
      'current_signals' – {feature: {shap, corr, mi, stability, redundancy}}
      'drop'            – list of features to consider removing
      'add'             – list of (feature, score) candidates to consider adding
    """
    common   = set(feature_filter.get(league, {}).get("Common", []))
    current  = list(feature_filter.get(league, {}).get(market, []))

    if not current:
        return {}

    mkey      = market_key(league, market)
    has_model = os.path.isfile(model_path(league, market))

    # ── Load training data ──
    tpath = training_path(league, market)
    if os.path.isfile(tpath):
        train_df = pd.read_csv(tpath, index_col=0)
        train_df = train_df.sort_values("Date").reset_index(drop=True)
        target   = pd.to_numeric(train_df["Result"], errors="coerce").fillna(0)
        train_df = train_df.drop(columns=[c for c in META_COLS if c in train_df.columns],
                                  errors="ignore")
    else:
        train_df = pd.DataFrame()
        target   = pd.Series(dtype=float)

    # ── Signal computation for existing features ──
    shap_raw  = compute_shap_scores(current, mkey, shap_df)
    corr_pre  = compute_corr_scores(current, mkey, corr_df)
    corr_train, mi_raw, stab_raw = compute_from_training(current, train_df, target)

    # Prefer pre-computed correlation; fall back to on-the-fly
    corr_raw = {f: max(corr_pre.get(f, 0), corr_train.get(f, 0)) for f in current}

    redundancy = compute_redundancy(current, train_df)

    # Normalize each signal across current features to [0,1]
    shap_n  = normalize_scores(shap_raw)
    corr_n  = normalize_scores(corr_raw)
    mi_n    = normalize_scores(mi_raw)
    stab_n  = normalize_scores(stab_raw)

    current_scores  = {}
    current_signals = {}
    for feat in current:
        r = redundancy.get(feat, 0.0)
        s = composite_score(feat, shap_n, corr_n, mi_n, stab_n, r, has_model)
        current_scores[feat] = s
        current_signals[feat] = dict(
            shap=shap_raw.get(feat, 0.0),
            corr=corr_raw.get(feat, 0.0),
            mi=mi_raw.get(feat, 0.0),
            stability=stab_raw.get(feat, 0.0),
            redundancy=r,
            composite=s,
        )

    drops = [f for f, s in current_scores.items() if s < drop_threshold]

    # ── Candidate evaluation ──
    candidates = discover_candidates(train_df, common, set(current))
    add_candidates = []
    if candidates and not train_df.empty:
        c_corr, c_mi, c_stab = compute_from_training(candidates, train_df, target)
        c_red  = compute_redundancy(candidates, train_df)
        # Normalize candidates against themselves
        c_corr_n = normalize_scores(c_corr)
        c_mi_n   = normalize_scores(c_mi)
        c_stab_n = normalize_scores(c_stab)
        zero = {f: 0.0 for f in candidates}
        for feat in candidates:
            r = c_red.get(feat, 0.0)
            s = composite_score(feat, zero, c_corr_n, c_mi_n, c_stab_n, r,
                                has_model=False)
            if s >= add_threshold:
                add_candidates.append((feat, s, dict(
                    corr=c_corr.get(feat, 0.0),
                    mi=c_mi.get(feat, 0.0),
                    stability=c_stab.get(feat, 0.0),
                    redundancy=r,
                    composite=s,
                )))
        add_candidates.sort(key=lambda x: -x[1])
        add_candidates = add_candidates[:top_candidates]

    return dict(
        current_scores=current_scores,
        current_signals=current_signals,
        drops=drops,
        add_candidates=add_candidates,
        has_model=has_model,
        n_training=len(target),
    )


# ─── Reporting ─────────────────────────────────────────────────────────────────

def print_market_report(league: str, market: str, result: dict,
                        drop_threshold: float) -> None:
    if not result:
        print(f"  (no features configured for {league}/{market})")
        return

    sigs = result["current_signals"]
    scores = result["current_scores"]
    has_model = result["has_model"]
    n_train = result["n_training"]
    drops = result["drops"]
    adds  = result["add_candidates"]

    model_tag = "[model]" if has_model else "[no model]"
    print(f"\n{'='*70}")
    print(f"  {league}  {market}  {model_tag}  (n={n_train:,})")
    print(f"{'='*70}")

    # Current features table
    hdr = f"  {'Feature':<45s} {'Score':>6s}"
    if has_model:
        hdr += f" {'SHAP':>6s}"
    hdr += f" {'Corr':>6s} {'MI':>6s} {'Stab':>6s} {'Redun':>6s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for feat in sorted(scores, key=lambda f: -scores[f]):
        sig = sigs[feat]
        s   = sig["composite"]
        tag = " <<<DROP" if feat in drops else ""
        row = f"  {feat:<45s} {s:>6.3f}"
        if has_model:
            row += f" {sig['shap']:>6.2f}"
        row += (f" {sig['corr']:>6.3f} {sig['mi']:>6.3f}"
                f" {sig['stability']:>6.3f} {sig['redundancy']:>6.3f}{tag}")
        print(row)

    # Drop summary
    if drops:
        print(f"\n  DROP candidates (score < {drop_threshold}):")
        for f in drops:
            print(f"    - {f}  (score={scores[f]:.3f})")
    else:
        print(f"\n  No drop candidates at threshold {drop_threshold}.")

    # Add candidates
    if adds:
        print(f"\n  ADD candidates (not in current filter):")
        print(f"  {'Feature':<45s} {'Score':>6s} {'Corr':>6s} {'MI':>6s} {'Stab':>6s}")
        print("  " + "-" * 78)
        for feat, s, sig in adds:
            print(f"  {feat:<45s} {s:>6.3f} {sig['corr']:>6.3f}"
                  f" {sig['mi']:>6.3f} {sig['stability']:>6.3f}")
    else:
        print("  No strong addition candidates found.")


# ─── Applying recommendations ──────────────────────────────────────────────────

def apply_recommendations(feature_filter: dict, league: str, market: str,
                           result: dict, drop_threshold: float,
                           add_threshold: float) -> list[str]:
    """
    Return an updated feature list for feature_filter[league][market]:
    - Removes drop candidates
    - Adds top candidates that exceed add_threshold
    """
    current = list(feature_filter[league][market])
    drops = set(result.get("drops", []))
    updated = [f for f in current if f not in drops]
    for feat, score, _ in result.get("add_candidates", []):
        if score >= add_threshold and feat not in updated:
            updated.append(feat)
    updated.sort()
    return updated


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and optimize feature_filter.json market feature sets")
    parser.add_argument("--league", required=True,
                        choices=["NFL", "NHL", "MLB", "WNBA", "NBA"],
                        help="League to evaluate")
    parser.add_argument("--market", nargs="+", default=None,
                        help="Market(s) to evaluate (default: all in league)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_DROP_THRESHOLD,
                        help=f"Drop threshold (default: {DEFAULT_DROP_THRESHOLD})")
    parser.add_argument("--add-threshold", type=float, default=DEFAULT_ADD_THRESHOLD,
                        help=f"Candidate addition threshold (default: {DEFAULT_ADD_THRESHOLD})")
    parser.add_argument("--top-candidates", type=int, default=DEFAULT_TOP_CANDIDATES,
                        help=f"Max additions to show per market (default: {DEFAULT_TOP_CANDIDATES})")
    parser.add_argument("--save", action="store_true",
                        help="Apply recommendations and save feature_filter.json")
    parser.add_argument("--no-add", action="store_true",
                        help="Do not suggest or apply feature additions")
    args = parser.parse_args()

    # ── Load shared data ──
    ff_path = pkg_resources.files(data) / "feature_filter.json"
    with open(ff_path) as f:
        feature_filter = json.load(f)

    shap_df = load_csv_safe(pkg_resources.files(data) / "feature_importances.csv")
    corr_df = load_csv_safe(pkg_resources.files(data) / "feature_correlations.csv")

    if not shap_df.empty:
        shap_df = shap_df.clip(lower=0)          # SHAP is always non-negative here
    if not corr_df.empty:
        corr_df = corr_df.abs()

    league_filter = feature_filter.get(args.league, {})
    all_markets = [m for m in league_filter if m != "Common"]

    if args.market:
        # Accept both space-form ("passing yards") and hyphen-form ("passing-yards")
        requested = {m.replace("-", " ") for m in args.market}
        markets = [m for m in all_markets if m in requested]
        unknown = requested - set(markets)
        if unknown:
            print(f"WARNING: markets not found in {args.league}: {unknown}")
    else:
        markets = all_markets

    if not markets:
        print(f"No markets to evaluate for {args.league}.")
        return

    add_threshold = 0.0 if args.no_add else args.add_threshold
    changes = {}

    for market in markets:
        result = evaluate_market(
            league=args.league,
            market=market,
            feature_filter=feature_filter,
            shap_df=shap_df,
            corr_df=corr_df,
            drop_threshold=args.threshold,
            add_threshold=add_threshold,
            top_candidates=0 if args.no_add else args.top_candidates,
        )
        print_market_report(args.league, market, result, args.threshold)

        if args.save and result:
            updated = apply_recommendations(
                feature_filter, args.league, market, result,
                args.threshold, add_threshold)
            old = feature_filter[args.league][market]
            if sorted(updated) != sorted(old):
                changes[market] = dict(
                    removed=[f for f in old if f not in updated],
                    added=[f for f in updated if f not in old],
                )
                feature_filter[args.league][market] = updated

    # ── Save ──
    if args.save:
        if changes:
            print(f"\n{'='*70}")
            print("SAVING changes to feature_filter.json")
            print(f"{'='*70}")
            for market, ch in changes.items():
                print(f"  {args.league}/{market}:")
                for f in ch["removed"]:
                    print(f"    - REMOVED: {f}")
                for f in ch["added"]:
                    print(f"    + ADDED:   {f}")
            with open(ff_path, "w") as f:
                json.dump(feature_filter, f, indent=4)
            print(f"\nSaved to {ff_path}")
        else:
            print("\nNo changes to save.")


if __name__ == "__main__":
    main()
