#!/usr/bin/env python3
"""
Evaluate alternative feature sets for player comps.

For each target position group, this script:
1. Discovers all available numeric features with adequate coverage
2. Scores the current feature set as a baseline (equal weights)
3. Tests adding each unused feature (marginal gain)
4. Tests removing each current feature (marginal loss)
5. Runs greedy forward/backward selection to find the best feature set
6. Optionally optimizes weights for the best feature set found

Usage:
    poetry run python3 evaluate_comp_features.py --league NFL --position WR
    poetry run python3 evaluate_comp_features.py --league NHL --position C W
    poetry run python3 evaluate_comp_features.py --league NHL --position C W --optimize
    poetry run python3 evaluate_comp_features.py --league NBA --position P C F
    poetry run python3 evaluate_comp_features.py --league WNBA --position G F C
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import json
import importlib.resources as pkg_resources
from sportstradamus import data
from sportstradamus.stats import StatsNFL, StatsNHL, StatsNBA, StatsWNBA
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore")

from optimize_comp_weights import (
    build_weighted_comps,
    measure_comp_quality,
    precompute_market_lookups,
    optimize_position_weights,
    normalize_weights,
)


# ─── Scoring ───────────────────────────────────────────────────────────────────

def score_feature_set(features, full_profile, position, lookups,
                      min_comps=5, max_comps=15):
    """
    Score a feature set using equal weights.

    Returns (score, n_players) tuple.
    """
    player_opp_z, player_games, player_positions = lookups

    profile = full_profile[list(features)].copy()
    for col in profile.columns:
        profile[col] = pd.to_numeric(profile[col], errors='coerce')
    profile = profile.dropna()

    if len(profile) < min_comps + 2:
        return 0.0, len(profile)

    z = profile.apply(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0, axis=0)
    z = z.replace([np.inf, -np.inf], np.nan).dropna()

    if len(z) < min_comps + 2:
        return 0.0, len(z)

    weights = np.ones(len(features))
    comps = build_weighted_comps(z, weights, min_comps, max_comps)
    if not comps:
        return 0.0, len(z)

    comps_dict = {position: comps}
    score = measure_comp_quality(comps_dict, player_opp_z, player_games,
                                 player_positions, [position])
    return score, len(z)


# ─── Feature discovery ─────────────────────────────────────────────────────────

def discover_numeric_features(profile, min_coverage=0.5, min_unique=3):
    """Find all numeric features with adequate coverage."""
    features = []
    for col in profile.columns:
        if col.endswith("growth") or col.endswith("short"):
            continue
        try:
            vals = pd.to_numeric(profile[col], errors='coerce')
            coverage = vals.notna().mean()
            n_unique = vals.dropna().nunique()
            if coverage >= min_coverage and n_unique >= min_unique:
                features.append((col, coverage, n_unique))
        except Exception:
            pass
    return sorted(features, key=lambda x: -x[1])


# ─── NFL data loading ──────────────────────────────────────────────────────────

# Columns that are identifiers or intermediate calculations, not features
NFL_EXCLUDE = {
    "player", "player_id", "position", "team_name", "player_game_count",
    # Raw columns used only for derivation
    "dropbacks", "attempts", "routes", "targets", "scrambles",
    "designed_yards", "total_touches", "breakaway_yards",
    "contested_targets", "deep_contested_targets",
    "zone_grades_pass_route", "man_grades_pass_route",
    "center_medium_grades_pass",
    # Intermediate filter columns
    "snap_counts_offense", "snap_counts_defense", "snap_counts_special_teams",
}

NHL_EXCLUDE = {
    "playerName", "position", "team",
}

NBA_EXCLUDE = {
    "TEAM_ABBREVIATION", "PLAYER_ID", "GAME_ID", "SEASON_YEAR", "POS",
    "HOME", "OPP", "WL",
}

WNBA_EXCLUDE = NBA_EXCLUDE.copy()


def get_nfl_position_data(playerProfile, stats_obj, position):
    """Get position-filtered profile and pre-computed lookups for NFL."""
    filterStat = {"QB": "dropbacks", "RB": "attempts", "WR": "routes", "TE": "routes"}

    posProfile = playerProfile.loc[playerProfile.position == position].copy()
    posProfile[filterStat[position]] = (
        posProfile[filterStat[position]] / posProfile["player_game_count"])
    posProfile = posProfile.loc[
        posProfile[filterStat[position]] >= posProfile[filterStat[position]].quantile(.25)]

    gamelog = stats_obj.gamelog.copy()
    pos_gamelog = gamelog[gamelog["player display name"].isin(posProfile.index)]

    target_markets = {
        "QB": ["fantasy points underdog"],
        "RB": ["fantasy points underdog"],
        "WR": ["fantasy points underdog"],
        "TE": ["fantasy points underdog"],
    }

    lookups = precompute_market_lookups(
        pos_gamelog, "player display name", "opponent", "gameday",
        target_markets[position], "position", min_games=5)

    return posProfile, lookups


# ─── NHL data loading ──────────────────────────────────────────────────────────

def get_nhl_position_data(playerProfile, all_players, id_to_name, stats_obj, position):
    """Get position-filtered profile and pre-computed lookups for NHL."""
    pos_players = [p for p, v in all_players.items()
                   if v.get("position") == position and p in playerProfile.index]
    posProfile = playerProfile.loc[pos_players].copy()

    # Re-index by player name to match gamelog
    posProfile.index = posProfile.index.map(lambda x: id_to_name.get(x, x))
    posProfile = posProfile[~posProfile.index.duplicated(keep='first')]

    gamelog = stats_obj.gamelog.copy()
    pos_gamelog = gamelog[gamelog["playerName"].isin(posProfile.index)]

    target_markets = {
        "C": ["skater fantasy points underdog"],
        "W": ["skater fantasy points underdog"],
        "D": ["skater fantasy points underdog"],
        "G": ["goalie fantasy points underdog"],
    }

    lookups = precompute_market_lookups(
        pos_gamelog, "playerName", "opponent", "gameDate",
        target_markets[position], "position", min_games=5)

    return posProfile, lookups


# ─── NBA / WNBA data loading ───────────────────────────────────────────────────

def get_nba_position_data(playerProfile, playerDict, stats_obj, position):
    """Get position-filtered profile and pre-computed lookups for NBA."""
    pos_players = [p for p, v in playerDict.items()
                   if v.get("POS") == position and p in playerProfile.index]
    posProfile = playerProfile.loc[pos_players].copy()

    gamelog = stats_obj.gamelog.copy()
    pos_gamelog = gamelog[gamelog["PLAYER_NAME"].isin(posProfile.index)]

    lookups = precompute_market_lookups(
        pos_gamelog, "PLAYER_NAME", "OPP", "GAME_DATE",
        ["fantasy points prizepicks"], "POS", min_games=10)

    return posProfile, lookups


def get_wnba_position_data(playerProfile, playerDict, stats_obj, position):
    """Get position-filtered profile and pre-computed lookups for WNBA."""
    pos_players = [p for p, v in playerDict.items()
                   if v.get("POS") == position and p in playerProfile.index]
    posProfile = playerProfile.loc[pos_players].copy()

    gamelog = stats_obj.gamelog.copy()
    pos_gamelog = gamelog[gamelog["PLAYER_NAME"].isin(posProfile.index)]

    lookups = precompute_market_lookups(
        pos_gamelog, "PLAYER_NAME", "OPP", "GAME_DATE",
        ["fantasy points prizepicks"], "POS", min_games=8)

    return posProfile, lookups


# ─── Feature evaluation phases ─────────────────────────────────────────────────

def run_marginal_analysis(current_features, available_features, full_profile,
                          position, lookups, min_comps=5, max_comps=15):
    """
    Test adding each unused feature and removing each current feature.

    Returns (baseline_score, additions_list, removals_list).
    """
    baseline_score, baseline_n = score_feature_set(
        current_features, full_profile, position, lookups, min_comps, max_comps)
    print(f"  Baseline: {baseline_score:.5f} "
          f"({baseline_n} players, {len(current_features)} features)")

    unused = [f for f in available_features if f not in current_features]

    additions = []
    if unused:
        print(f"\n  Testing {len(unused)} feature additions...")
        for feat in tqdm(unused, desc="  Adding", leave=False):
            test_features = list(current_features) + [feat]
            score, n = score_feature_set(
                test_features, full_profile, position, lookups, min_comps, max_comps)
            additions.append((feat, score - baseline_score, score, n))
        additions.sort(key=lambda x: -x[1])

    removals = []
    if len(current_features) > 3:
        print(f"  Testing {len(current_features)} feature removals...")
        for feat in tqdm(current_features, desc="  Removing", leave=False):
            test_features = [f for f in current_features if f != feat]
            score, n = score_feature_set(
                test_features, full_profile, position, lookups, min_comps, max_comps)
            removals.append((feat, score - baseline_score, score, n))
        removals.sort(key=lambda x: -x[1])

    return baseline_score, additions, removals


def greedy_forward_selection(current_features, available_features, full_profile,
                             position, lookups, min_comps=5, max_comps=15,
                             max_additions=5, threshold=0.0005):
    """Greedily add features that improve the comp quality score."""
    features = list(current_features)
    unused = [f for f in available_features if f not in features]
    score, _ = score_feature_set(
        features, full_profile, position, lookups, min_comps, max_comps)
    history = [("baseline", score, len(features))]

    for step in range(max_additions):
        if not unused:
            break

        best_feat, best_delta = None, 0.0
        for feat in tqdm(unused, desc=f"  Fwd step {step+1}", leave=False):
            test = features + [feat]
            s, _ = score_feature_set(
                test, full_profile, position, lookups, min_comps, max_comps)
            delta = s - score
            if delta > best_delta:
                best_feat, best_delta = feat, delta

        if best_delta < threshold:
            print(f"    Step {step+1}: no feature improves score by >= {threshold}")
            break

        features.append(best_feat)
        unused.remove(best_feat)
        score += best_delta
        history.append((f"+{best_feat}", score, len(features)))
        print(f"    Step {step+1}: +{best_feat} -> {score:.5f} ({best_delta:+.5f})")

    return features, score, history


def greedy_backward_elimination(current_features, full_profile, position, lookups,
                                min_comps=5, max_comps=15, max_removals=5,
                                threshold=0.0005, min_features=5):
    """Greedily remove features that improve (or least hurt) the score."""
    features = list(current_features)
    score, _ = score_feature_set(
        features, full_profile, position, lookups, min_comps, max_comps)
    history = [("baseline", score, len(features))]

    for step in range(max_removals):
        if len(features) <= min_features:
            break

        best_feat, best_delta = None, 0.0
        for feat in tqdm(features, desc=f"  Bwd step {step+1}", leave=False):
            test = [f for f in features if f != feat]
            s, _ = score_feature_set(
                test, full_profile, position, lookups, min_comps, max_comps)
            delta = s - score
            if delta > best_delta:
                best_feat, best_delta = feat, delta

        if best_delta < threshold:
            print(f"    Step {step+1}: no removal improves score by >= {threshold}")
            break

        features.remove(best_feat)
        score += best_delta
        history.append((f"-{best_feat}", score, len(features)))
        print(f"    Step {step+1}: -{best_feat} -> {score:.5f} ({best_delta:+.5f})")

    return features, score, history


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate feature sets for player comps")
    parser.add_argument("--league", required=True, choices=["NFL", "NHL", "NBA", "WNBA"],
                        help="League to evaluate")
    parser.add_argument("--position", nargs="+", default=None,
                        help="Position(s) to evaluate (e.g., WR, C W); defaults to all")
    parser.add_argument("--optimize", action="store_true",
                        help="Run weight optimization on the best feature set")
    parser.add_argument("--maxiter", type=int, default=20,
                        help="Max DE iterations for --optimize (default: 20)")
    parser.add_argument("--max-additions", type=int, default=5,
                        help="Max features to add in forward selection (default: 5)")
    parser.add_argument("--max-removals", type=int, default=5,
                        help="Max features to remove in backward elimination (default: 5)")
    parser.add_argument("--save", action="store_true",
                        help="Save the best feature set to playerCompStats.json")
    args = parser.parse_args()

    with open(pkg_resources.files(data) / "playerCompStats.json", "r") as f:
        current_weights = json.load(f)

    # Load league data once
    if args.league == "NFL":
        print("Loading NFL data...")
        stats_obj = StatsNFL()
        stats_obj.load()
        full_profile = stats_obj.build_comp_profile()
        if full_profile.empty:
            print("ERROR: Could not load NFL data")
            return
        exclude = NFL_EXCLUDE
    elif args.league == "NHL":
        print("Loading NHL data...")
        stats_obj = StatsNHL()
        stats_obj.load()
        full_profile, all_players, id_to_name = stats_obj.build_comp_profile()
        exclude = NHL_EXCLUDE
    elif args.league == "NBA":
        print("Loading NBA data...")
        stats_obj = StatsNBA()
        stats_obj.load()
        full_profile, player_dict = stats_obj.build_comp_profile()
        exclude = NBA_EXCLUDE
    elif args.league == "WNBA":
        print("Loading WNBA data...")
        stats_obj = StatsWNBA()
        stats_obj.load()
        full_profile, player_dict = stats_obj.build_comp_profile()
        exclude = WNBA_EXCLUDE

    positions = args.position or list(current_weights.get(args.league, {}).keys())

    for position in positions:
        print(f"\n{'='*60}")
        print(f"{args.league} {position}")
        print(f"{'='*60}")

        if position not in current_weights.get(args.league, {}):
            print(f"  No current weights found for {args.league} {position}")
            continue

        current_features = list(current_weights[args.league][position].keys())
        print(f"  Current features ({len(current_features)}):")
        for feat in current_features:
            w = current_weights[args.league][position][feat]
            print(f"    {feat:<40s} {w}")

        # Get position data
        if args.league == "NFL":
            pos_profile, lookups = get_nfl_position_data(
                full_profile, stats_obj, position)
            min_comps = 5
        elif args.league == "NHL":
            pos_profile, lookups = get_nhl_position_data(
                full_profile, all_players, id_to_name, stats_obj, position)
            min_comps = 4 if position == "G" else 5
        elif args.league == "NBA":
            pos_profile, lookups = get_nba_position_data(
                full_profile, player_dict, stats_obj, position)
            min_comps = 5
        elif args.league == "WNBA":
            pos_profile, lookups = get_wnba_position_data(
                full_profile, player_dict, stats_obj, position)
            min_comps = 5

        player_opp_z, player_games, player_positions = lookups
        if not player_opp_z:
            print(f"  No market data for {position}, skipping")
            continue

        # Discover all available features
        available = discover_numeric_features(pos_profile)
        available_names = [f for f, _, _ in available
                          if f not in exclude]
        unused = [f for f in available_names if f not in current_features]
        print(f"\n  Available features: {len(available_names)}")
        print(f"  Unused candidates ({len(unused)}):")
        for f, cov, uniq in available:
            if f in unused:
                print(f"    {f:<40s} coverage={cov:.0%}  unique={uniq}")

        # ── Phase 1: Marginal analysis ──
        print(f"\n  Phase 1: Marginal Analysis")
        print(f"  {'-'*50}")
        baseline, additions, removals = run_marginal_analysis(
            current_features, available_names, pos_profile, position, lookups,
            min_comps=min_comps, max_comps=15)

        if additions:
            print(f"\n  Feature additions (sorted by improvement):")
            print(f"  {'Feature':<40s} {'Delta':>8s} {'Score':>8s} {'N':>5s}")
            for feat, delta, score, n in additions:
                marker = " <<<" if delta > 0.001 else ""
                print(f"  {feat:<40s} {delta:>+8.5f} {score:>8.5f} {n:>5d}{marker}")

        if removals:
            print(f"\n  Feature removals (sorted by improvement):")
            print(f"  {'Feature':<40s} {'Delta':>8s} {'Score':>8s} {'N':>5s}")
            for feat, delta, score, n in removals:
                marker = " <<<" if delta > 0.001 else ""
                print(f"  {feat:<40s} {delta:>+8.5f} {score:>8.5f} {n:>5d}{marker}")

        # ── Phase 2: Greedy forward selection ──
        print(f"\n  Phase 2: Greedy Forward Selection")
        print(f"  {'-'*50}")
        fwd_features, fwd_score, fwd_history = greedy_forward_selection(
            current_features, available_names, pos_profile, position, lookups,
            min_comps=min_comps, max_comps=15, max_additions=args.max_additions)

        # ── Phase 3: Greedy backward elimination ──
        print(f"\n  Phase 3: Greedy Backward Elimination")
        print(f"  {'-'*50}")
        best_features, best_score, bwd_history = greedy_backward_elimination(
            fwd_features, pos_profile, position, lookups,
            min_comps=min_comps, max_comps=15, max_removals=args.max_removals)

        # ── Summary ──
        print(f"\n  {'='*50}")
        print(f"  RESULTS for {args.league} {position}")
        print(f"  {'='*50}")
        print(f"  Baseline score: {baseline:.5f} ({len(current_features)} features)")
        print(f"  Best score:     {best_score:.5f} ({len(best_features)} features)")
        print(f"  Improvement:    {best_score - baseline:+.5f}")

        added = [f for f in best_features if f not in current_features]
        removed = [f for f in current_features if f not in best_features]
        if added:
            print(f"  Added:   {added}")
        if removed:
            print(f"  Removed: {removed}")
        if not added and not removed:
            print(f"  No changes recommended")

        print(f"\n  Best feature set:")
        for feat in best_features:
            tag = " (NEW)" if feat in added else ""
            print(f"    {feat}{tag}")

        # ── Phase 4: Optional weight optimization ──
        if args.optimize and (best_score > baseline or added or removed):
            print(f"\n  Phase 4: Weight Optimization on best feature set")
            print(f"  {'-'*50}")

            profile = pos_profile[best_features].copy()
            for col in profile.columns:
                profile[col] = pd.to_numeric(profile[col], errors='coerce')
            profile = profile.dropna()
            z_profile = profile.apply(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0, axis=0)
            z_profile = z_profile.replace([np.inf, -np.inf], np.nan).dropna()

            init_weights = np.ones(len(best_features))

            best_w, opt_score, eq_score = optimize_position_weights(
                z_profile, player_opp_z, player_games, player_positions,
                position, best_features, init_weights,
                min_comps=min_comps, max_comps=20,
                maxiter=args.maxiter, popsize=8)

            print(f"\n  Equal-weight score:     {eq_score:.5f}")
            print(f"  Optimized-weight score: {opt_score:.5f}")
            print(f"\n  Optimized weights:")
            for f, w in sorted(best_w.items(), key=lambda x: -x[1]):
                tag = " (NEW)" if f in added else ""
                print(f"    {f:<40s} {w:>5.1f}{tag}")

            if args.save:
                current_weights[args.league][position] = best_w

        elif args.save and (added or removed):
            # Save with equal weights normalized
            equal_w = {f: 1.0 for f in best_features}
            current_weights[args.league][position] = normalize_weights(equal_w)

    if args.save:
        filepath = pkg_resources.files(data) / "playerCompStats.json"
        with open(filepath, "w") as f:
            json.dump(current_weights, f, indent=4)
        print(f"\nSaved updated weights to {filepath}")


if __name__ == "__main__":
    main()
