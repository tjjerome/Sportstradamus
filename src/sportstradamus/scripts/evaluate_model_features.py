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

import argparse
import importlib.resources as pkg_resources
import json
import os
import warnings

import pandas as pd

from sportstradamus import data
from sportstradamus.feature_selection import (
    EVAL_ADD_THRESHOLD as DEFAULT_ADD_THRESHOLD,
)
from sportstradamus.feature_selection import (
    EVAL_DROP_CUTOFF as DEFAULT_DROP_THRESHOLD,
)
from sportstradamus.feature_selection import (
    EVAL_TOP_CANDIDATES as DEFAULT_TOP_CANDIDATES,
)
from sportstradamus.feature_selection import (
    META_COLS,
    composite_score,
    compute_corr_scores,
    compute_from_training,
    compute_redundancy,
    compute_shap_scores,
    discover_candidates,
    market_key,
    model_path,
    normalize_scores,
    training_path,
)

warnings.filterwarnings("ignore")


def load_csv_safe(path):
    if os.path.isfile(path):
        return pd.read_csv(path, index_col=0)
    return pd.DataFrame()


# ─── Per-market evaluation ─────────────────────────────────────────────────────


def evaluate_market(
    league: str,
    market: str,
    feature_filter: dict,
    shap_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    drop_threshold: float,
    add_threshold: float,
    top_candidates: int,
) -> dict:
    """
    Evaluate all features for one market.

    Returns a dict with:
      'current_scores'  – {feature: composite_score}
      'current_signals' – {feature: {shap, corr, mi, stability, redundancy}}
      'drop'            – list of features to consider removing
      'add'             – list of (feature, score) candidates to consider adding
    """
    common = set(feature_filter.get(league, {}).get("Common", []))
    current = list(feature_filter.get(league, {}).get(market, []))

    if not current:
        return {}

    mkey = market_key(league, market)
    has_model = os.path.isfile(model_path(league, market))

    # ── Load training data ──
    tpath = training_path(league, market)
    if os.path.isfile(tpath):
        train_df = pd.read_csv(tpath, index_col=0)
        train_df = train_df.sort_values("Date").reset_index(drop=True)
        target = pd.to_numeric(train_df["Result"], errors="coerce").fillna(0)
        train_df = train_df.drop(
            columns=[c for c in META_COLS if c in train_df.columns], errors="ignore"
        )
    else:
        train_df = pd.DataFrame()
        target = pd.Series(dtype=float)

    # ── Signal computation for existing features ──
    shap_raw = compute_shap_scores(current, mkey, shap_df)
    corr_pre = compute_corr_scores(current, mkey, corr_df)
    corr_train, mi_raw, stab_raw = compute_from_training(current, train_df, target)

    # Prefer pre-computed correlation; fall back to on-the-fly
    corr_raw = {f: max(corr_pre.get(f, 0), corr_train.get(f, 0)) for f in current}

    redundancy = compute_redundancy(current, train_df)

    # Normalize each signal across current features to [0,1]
    shap_n = normalize_scores(shap_raw)
    corr_n = normalize_scores(corr_raw)
    mi_n = normalize_scores(mi_raw)
    stab_n = normalize_scores(stab_raw)

    current_scores = {}
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
        c_red = compute_redundancy(candidates, train_df)
        # Normalize candidates against themselves
        c_corr_n = normalize_scores(c_corr)
        c_mi_n = normalize_scores(c_mi)
        c_stab_n = normalize_scores(c_stab)
        zero = {f: 0.0 for f in candidates}
        for feat in candidates:
            r = c_red.get(feat, 0.0)
            s = composite_score(feat, zero, c_corr_n, c_mi_n, c_stab_n, r, has_model=False)
            if s >= add_threshold:
                add_candidates.append(
                    (
                        feat,
                        s,
                        dict(
                            corr=c_corr.get(feat, 0.0),
                            mi=c_mi.get(feat, 0.0),
                            stability=c_stab.get(feat, 0.0),
                            redundancy=r,
                            composite=s,
                        ),
                    )
                )
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


def print_market_report(league: str, market: str, result: dict, drop_threshold: float) -> None:
    if not result:
        print(f"  (no features configured for {league}/{market})")
        return

    sigs = result["current_signals"]
    scores = result["current_scores"]
    has_model = result["has_model"]
    n_train = result["n_training"]
    drops = result["drops"]
    adds = result["add_candidates"]

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
        s = sig["composite"]
        tag = " <<<DROP" if feat in drops else ""
        row = f"  {feat:<45s} {s:>6.3f}"
        if has_model:
            row += f" {sig['shap']:>6.2f}"
        row += (
            f" {sig['corr']:>6.3f} {sig['mi']:>6.3f}"
            f" {sig['stability']:>6.3f} {sig['redundancy']:>6.3f}{tag}"
        )
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
        print("\n  ADD candidates (not in current filter):")
        print(f"  {'Feature':<45s} {'Score':>6s} {'Corr':>6s} {'MI':>6s} {'Stab':>6s}")
        print("  " + "-" * 78)
        for feat, s, sig in adds:
            print(
                f"  {feat:<45s} {s:>6.3f} {sig['corr']:>6.3f}"
                f" {sig['mi']:>6.3f} {sig['stability']:>6.3f}"
            )
    else:
        print("  No strong addition candidates found.")


# ─── Applying recommendations ──────────────────────────────────────────────────


def apply_recommendations(
    feature_filter: dict,
    league: str,
    market: str,
    result: dict,
    drop_threshold: float,
    add_threshold: float,
) -> list[str]:
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
        description="Evaluate and optimize feature_filter.json market feature sets"
    )
    parser.add_argument(
        "--league",
        required=True,
        choices=["NFL", "NHL", "MLB", "WNBA", "NBA"],
        help="League to evaluate",
    )
    parser.add_argument(
        "--market", nargs="+", default=None, help="Market(s) to evaluate (default: all in league)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_DROP_THRESHOLD,
        help=f"Drop threshold (default: {DEFAULT_DROP_THRESHOLD})",
    )
    parser.add_argument(
        "--add-threshold",
        type=float,
        default=DEFAULT_ADD_THRESHOLD,
        help=f"Candidate addition threshold (default: {DEFAULT_ADD_THRESHOLD})",
    )
    parser.add_argument(
        "--top-candidates",
        type=int,
        default=DEFAULT_TOP_CANDIDATES,
        help=f"Max additions to show per market (default: {DEFAULT_TOP_CANDIDATES})",
    )
    parser.add_argument(
        "--save", action="store_true", help="Apply recommendations and save feature_filter.json"
    )
    parser.add_argument(
        "--no-add", action="store_true", help="Do not suggest or apply feature additions"
    )
    args = parser.parse_args()

    # ── Load shared data ──
    ff_path = pkg_resources.files(data) / "feature_filter.json"
    with open(ff_path) as f:
        feature_filter = json.load(f)

    shap_df = load_csv_safe(pkg_resources.files(data) / "feature_importances.csv")
    corr_df = load_csv_safe(pkg_resources.files(data) / "feature_correlations.csv")

    if not shap_df.empty:
        shap_df = shap_df.clip(lower=0)  # SHAP is always non-negative here
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
                feature_filter, args.league, market, result, args.threshold, add_threshold
            )
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
