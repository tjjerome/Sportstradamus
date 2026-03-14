"""
Optimize playerCompStats.json weights based on predictive power.

Jointly optimizes the full weight vector for each league/position using
scipy.optimize.differential_evolution. The objective function:
  1. Apply weight vector to z-scored player profiles
  2. Build BallTree comps using the weighted profiles
  3. Measure mean Spearman correlation of the comp signal vs actual outcomes
     across multiple target markets

Also runs per-feature diagnostics and reports the composite predictive score
for both current and optimized weights.
"""

import numpy as np
import pandas as pd
import pickle
import json
import importlib.resources as pkg_resources
from sklearn.neighbors import BallTree
from scipy.stats import zscore, spearmanr
from scipy.optimize import differential_evolution
from datetime import datetime, timedelta
from sportstradamus import data
from sportstradamus.stats import Stats, StatsNBA, StatsWNBA, StatsNFL, StatsNHL
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore")


def build_weighted_comps(z_profile, weights, min_comps=5, max_comps=15):
    """
    Build comps from a z-scored profile using a weight vector.
    
    Args:
        z_profile: DataFrame of z-scored player features (n_players x n_features)
        weights: array-like of weights, same length as features
        min_comps: minimum comps per player
        max_comps: maximum comps per player
    
    Returns:
        dict: {player: {"comps": [...], "distances": [...]}}
    """
    n_players = len(z_profile)
    if n_players < min_comps + 2:
        return {}
    
    weighted = z_profile.mul(np.sqrt(np.abs(weights)))
    knn = BallTree(weighted.values)
    return Stats._build_comps(knn, weighted, min_comps, max_comps)


def measure_comp_quality(comps_dict, player_opp_z_lookup, player_games_lookup,
                         player_positions, positions, min_comp_hits=2,
                         detailed=False):
    """
    Measure the predictive quality of comps using pre-computed z-score lookups.
    
    Args:
        comps_dict: {position: {player: {"comps": [...], "distances": [...]}}}
        player_opp_z_lookup: dict of market -> {(player, opp): mean_zscore}
        player_games_lookup: dict of market -> {player: [(opp, zscore), ...]}
        player_positions: dict of player -> position
        positions: list of valid position strings
        min_comp_hits: minimum comps with data for a prediction to count
        detailed: if True, return (score, details_dict) instead of just score
    
    Returns:
        Weighted mean Spearman correlation across markets (higher = better).
        If detailed=True, also returns a dict with per-market breakdown.
    """
    correlations = []  # (corr, n_obs) pairs
    details = {}
    
    for market in player_opp_z_lookup:
        opp_z = player_opp_z_lookup[market]
        game_data = player_games_lookup[market]
        
        predictions = []
        actuals = []
        
        for player, games in game_data.items():
            pos = player_positions.get(player)
            if pos is None or pos not in comps_dict:
                continue
            comp_data = comps_dict.get(pos, {}).get(player)
            if comp_data is None:
                continue
            
            comp_names = comp_data["comps"]
            comp_dists = comp_data["distances"]
            
            for opp, actual_z in games:
                weighted_z = 0.0
                total_weight = 0.0
                n_hits = 0
                
                for comp, dist in zip(comp_names, comp_dists):
                    if comp == player:
                        continue
                    key = (comp, opp)
                    if key in opp_z:
                        w = 1.0 / (1.0 + dist)
                        weighted_z += opp_z[key] * w
                        total_weight += w
                        n_hits += 1
                
                if n_hits >= min_comp_hits and total_weight > 0:
                    predictions.append(weighted_z / total_weight)
                    actuals.append(actual_z)
        
        if len(predictions) >= 50:
            corr, pval = spearmanr(predictions, actuals)
            if not np.isnan(corr):
                n_obs = len(predictions)
                correlations.append((corr, n_obs))
                details[market] = {"corr": corr, "pval": pval, "n": n_obs}
    
    if not correlations:
        return (0.0, details) if detailed else 0.0
    
    # Weight by sqrt(n) so larger-sample markets contribute more
    corrs = np.array([c for c, n in correlations])
    weights = np.array([np.sqrt(n) for c, n in correlations])
    weighted_mean = np.average(corrs, weights=weights)
    
    if detailed:
        return weighted_mean, details
    return weighted_mean


def precompute_market_lookups(gamelog, player_col, opp_col, date_col, markets,
                              position_col, min_games=10):
    """
    Pre-compute the z-score lookups that measure_comp_quality needs.
    This is expensive, so we do it once and reuse across many weight evaluations.
    
    Returns:
        player_opp_z_lookup: {market: {(player, opp): mean_zscore}}
        player_games_lookup: {market: {player: [(opp, zscore), ...]}}
        player_positions: {player: position}
    """
    player_positions = gamelog.groupby(player_col)[position_col].first().to_dict()
    player_opp_z_lookup = {}
    player_games_lookup = {}
    
    for market in markets:
        if market not in gamelog.columns:
            continue
        
        gl = gamelog[[player_col, opp_col, date_col, market, position_col]].dropna()
        if gl.empty:
            continue
        
        game_counts = gl.groupby(player_col).size()
        valid_players = game_counts[game_counts >= min_games].index
        gl = gl[gl[player_col].isin(valid_players)]
        if len(gl) < 100:
            continue
        
        gl = gl.copy()
        player_means = gl.groupby(player_col)[market].transform("mean")
        player_stds = gl.groupby(player_col)[market].transform("std")
        player_stds = player_stds.replace(0, np.nan)
        gl["zscore"] = (gl[market] - player_means) / player_stds
        gl = gl.dropna(subset=["zscore"])
        
        # (player, opp) -> mean z-score lookup
        opp_z = gl.groupby([player_col, opp_col])["zscore"].mean().to_dict()
        player_opp_z_lookup[market] = opp_z
        
        # player -> [(opp, zscore), ...] for each game
        games_by_player = {}
        for _, row in gl.iterrows():
            p = row[player_col]
            games_by_player.setdefault(p, []).append((row[opp_col], row["zscore"]))
        player_games_lookup[market] = games_by_player
    
    return player_opp_z_lookup, player_games_lookup, player_positions


def optimize_position_weights(z_profile, player_opp_z, player_games, player_positions,
                              position, features, current_weights, min_comps=5, max_comps=15,
                              maxiter=50, popsize=10, tol=1e-3):
    """
    Jointly optimize the full weight vector for a single position group.
    
    Uses differential evolution to find the weight vector that maximizes
    the composite predictive score across all markets.
    
    Args:
        z_profile: z-scored player profile DataFrame
        player_opp_z: pre-computed {market: {(player, opp): z}} 
        player_games: pre-computed {market: {player: [(opp, z), ...]}}
        player_positions: {player: position}
        position: position string
        features: list of feature names
        current_weights: array of current weights (used as a seed)
        min_comps, max_comps: comp bounds
        maxiter: max optimizer iterations
        popsize: population size multiplier for DE
        tol: convergence tolerance
    
    Returns:
        (best_weights_dict, best_score, current_score)
    """
    n_features = len(features)
    
    best_so_far = [0.0]
    
    def objective(weight_vec):
        comps = build_weighted_comps(z_profile, weight_vec, min_comps, max_comps)
        if not comps:
            return 0.0  # will be negated below
        comps_dict = {position: comps}
        score = measure_comp_quality(comps_dict, player_opp_z, player_games,
                                     player_positions, [position])
        if score > best_so_far[0]:
            best_so_far[0] = score
        return -score  # DE minimizes, we want to maximize
    
    # Evaluate current weights first
    current_score = -objective(np.array(current_weights))
    print(f"    Current weights score: {current_score:.5f}")
    
    # Bounds: each weight in [0.1, 8.0]
    bounds = [(0.1, 8.0)] * n_features
    
    # Seed the initial population with the current weights + random perturbations
    rng = np.random.default_rng(42)
    pop_count = popsize * n_features
    init_pop = rng.uniform(0.1, 8.0, size=(pop_count, n_features))
    init_pop[0] = np.array(current_weights)  # include current as first member
    # Add several perturbations of current
    for i in range(1, min(5, pop_count)):
        noise = rng.normal(0, 0.5, n_features)
        init_pop[i] = np.clip(np.array(current_weights) + noise, 0.1, 8.0)
    
    pbar = tqdm(total=maxiter, desc=f"    DE {position}", unit="gen",
                 bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]  best={postfix}")
    
    def de_callback(xk, convergence):
        pbar.set_postfix_str(f"{best_so_far[0]:.5f}")
        pbar.update(1)
    
    result = differential_evolution(
        objective,
        bounds=bounds,
        init=init_pop,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        seed=42,
        disp=False,
        polish=False,
        callback=de_callback
    )
    
    pbar.close()
    best_score = -result.fun
    best_vec = result.x
    print(f"    Optimized score: {best_score:.5f}")
    
    # Show per-market breakdown for the optimized weights
    opt_comps = build_weighted_comps(z_profile, best_vec, min_comps, max_comps)
    if opt_comps:
        _, market_details = measure_comp_quality(
            {position: opt_comps}, player_opp_z, player_games,
            player_positions, [position], detailed=True)
        if market_details:
            print(f"    {'Market':<35s} {'corr':>7s} {'p-value':>10s} {'n':>6s}")
            for mkt, info in sorted(market_details.items(), key=lambda x: -x[1]['corr']):
                sig = '***' if info['pval'] < 0.001 else '**' if info['pval'] < 0.01 else '*' if info['pval'] < 0.05 else ''
                print(f"    {mkt:<35s} {info['corr']:>+7.4f} {info['pval']:>10.2e} {info['n']:>5d} {sig}")
    
    # Normalize to [0.5, 6.0] scale
    best_weights = normalize_weights(dict(zip(features, best_vec)), min_w=0.5, max_w=6.0)
    
    return best_weights, best_score, current_score


def run_per_feature_diagnostic(z_profile, player_opp_z, player_games, player_positions,
                               position, features, min_comps=5, max_comps=15):
    """
    Run per-feature diagnostic: how predictive is each feature on its own?
    Returns dict of {feature: score}.
    """
    scores = {}
    for i, feature in enumerate(features):
        single_weights = np.zeros(len(features))
        single_weights[i] = 1.0
        comps = build_weighted_comps(z_profile, single_weights, min_comps, max_comps)
        if not comps:
            scores[feature] = 0.0
            continue
        comps_dict = {position: comps}
        score = measure_comp_quality(comps_dict, player_opp_z, player_games,
                                     player_positions, [position])
        scores[feature] = score
    return scores


# ─── League-specific data preparation ─────────────────────────────────────────

def prepare_nba(stats_obj, features_by_pos, target_markets):
    """Load and prepare NBA data, returning per-position optimization inputs."""
    stats_obj.load()

    playerProfile, playerDict = stats_obj.build_comp_profile()
    gamelog = stats_obj.gamelog.copy()

    position_data = {}
    for position in stats_obj.positions:
        if position not in features_by_pos:
            continue
        features = features_by_pos[position]
        pos_players = [p for p, v in playerDict.items()
                      if v["POS"] == position and p in playerProfile.index]
        if len(pos_players) < 10:
            continue
        pos_profile = playerProfile.loc[pos_players, features].dropna()
        z_profile = pos_profile.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
        pos_gamelog = gamelog[gamelog["PLAYER_NAME"].isin(pos_profile.index)]

        lookups = precompute_market_lookups(
            pos_gamelog, "PLAYER_NAME", "OPP", "GAME_DATE",
            target_markets, "POS", min_games=10)

        position_data[position] = (z_profile, lookups)

    return position_data


def prepare_wnba(stats_obj, features_by_pos, target_markets):
    """Load and prepare WNBA data."""
    stats_obj.load()

    playerProfile, playerDict = stats_obj.build_comp_profile()
    gamelog = stats_obj.gamelog.copy()

    position_data = {}
    for position in stats_obj.positions:
        if position not in features_by_pos:
            continue
        features = features_by_pos[position]
        pos_players = [p for p, v in playerDict.items()
                      if v["POS"] == position and p in playerProfile.index]
        if len(pos_players) < 10:
            continue
        pos_profile = playerProfile.loc[pos_players, features].replace([np.nan, np.inf, -np.inf], 0)
        z_profile = pos_profile.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
        pos_gamelog = gamelog[gamelog["PLAYER_NAME"].isin(pos_profile.index)]

        lookups = precompute_market_lookups(
            pos_gamelog, "PLAYER_NAME", "OPP", "GAME_DATE",
            target_markets, "POS", min_games=8)

        position_data[position] = (z_profile, lookups)

    return position_data


def prepare_nfl(stats_obj, features_by_pos, target_markets_by_pos):
    """Load and prepare NFL data."""
    stats_obj.load()
    
    playerProfile = stats_obj.build_comp_profile()
    if playerProfile.empty:
        return {}
    
    filterStat = {"QB": "dropbacks", "RB": "attempts", "WR": "routes", "TE": "routes"}
    gamelog = stats_obj.gamelog.copy()
    
    position_data = {}
    for position in ["QB", "RB", "WR", "TE"]:
        features = features_by_pos[position]
        positionProfile = playerProfile.loc[playerProfile.position == position].copy()
        positionProfile[filterStat[position]] = positionProfile[filterStat[position]] / positionProfile["player_game_count"]
        positionProfile = positionProfile.loc[positionProfile[filterStat[position]] >= positionProfile[filterStat[position]].quantile(.25)]
        positionProfile = positionProfile[features].dropna()
        
        if len(positionProfile) < 10:
            continue
        
        z_profile = positionProfile.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
        pos_gamelog = gamelog[gamelog["player display name"].isin(positionProfile.index)]
        
        lookups = precompute_market_lookups(
            pos_gamelog, "player display name", "opponent", "gameday",
            target_markets_by_pos[position], "position", min_games=5)
        
        position_data[position] = (z_profile, lookups)
    
    return position_data


def prepare_nhl(stats_obj, features_by_pos, target_markets_by_pos):
    """Load and prepare NHL data."""
    stats_obj.load()
    
    playerProfile, all_players, id_to_name = stats_obj.build_comp_profile()
    if playerProfile.empty:
        return {}
    
    gamelog = stats_obj.gamelog.copy()
    
    position_data = {}
    for position in ["C", "W", "D", "G"]:
        features = features_by_pos[position]
        pos_players = [p for p, v in all_players.items()
                      if v.get("position") == position and p in playerProfile.index]
        positionProfile = playerProfile.loc[pos_players, features].dropna()
        
        if len(positionProfile) < 10:
            continue
        
        # Re-index by player name so comps match gamelog
        positionProfile.index = positionProfile.index.map(lambda x: id_to_name.get(x, x))
        positionProfile = positionProfile[~positionProfile.index.duplicated(keep='first')]
        
        z_profile = positionProfile.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
        pos_names = list(positionProfile.index)
        pos_gamelog = gamelog[gamelog["playerName"].isin(pos_names)]
        
        lookups = precompute_market_lookups(
            pos_gamelog, "playerName", "opponent", "gameDate",
            target_markets_by_pos[position], "position", min_games=5)
        
        position_data[position] = (z_profile, lookups)
    
    return position_data


# ─── Normalization ─────────────────────────────────────────────────────────────

def normalize_weights(raw_weights, min_w=0.5, max_w=6.0):
    """Linearly map raw weight values to [min_w, max_w]."""
    values = np.array(list(raw_weights.values()))
    lo, hi = values.min(), values.max()
    if hi == lo or hi == 0:
        return {k: round((min_w + max_w) / 2, 1) for k in raw_weights}
    normalized = {}
    for k, v in raw_weights.items():
        w = min_w + (v - lo) / (hi - lo) * (max_w - min_w)
        normalized[k] = round(w, 1)
    return normalized


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Optimize player comp weights")
    parser.add_argument("--leagues", nargs="+", default=["NBA", "WNBA", "NFL", "NHL"],
                       help="Leagues to optimize (default: all)")
    parser.add_argument("--output", default=None,
                       help="Output file path (default: print to stdout)")
    parser.add_argument("--save", action="store_true",
                       help="Overwrite playerCompStats.json with optimized weights")
    parser.add_argument("--diagnostic-only", action="store_true",
                       help="Only run per-feature diagnostics, skip joint optimization")
    parser.add_argument("--maxiter", type=int, default=50,
                       help="Max iterations for differential evolution (default: 50)")
    parser.add_argument("--popsize", type=int, default=10,
                       help="Population size multiplier for DE (default: 10)")
    args = parser.parse_args()
    
    with open(pkg_resources.files(data) / "playerCompStats.json", "r") as f:
        current_weights = json.load(f)
    
    optimized = {}
    score_report = {}
    
    # ── NBA ──
    if "NBA" in args.leagues:
        print("\n" + "=" * 60)
        print("NBA")
        print("=" * 60)
        features_by_pos = {pos: list(current_weights["NBA"][pos].keys()) for pos in current_weights["NBA"]}
        # target_markets = ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV", "FTM"]
        target_markets = ["fantasy points prizepicks"]
        
        nba = StatsNBA()
        pos_data = prepare_nba(nba, features_by_pos, target_markets)
        
        nba_weights = {}
        for position in pos_data:
            if position not in current_weights["NBA"]:
                continue
            z_profile, lookups = pos_data[position]
            opp_z, games, positions = lookups
            if not opp_z:
                print(f"  {position}: no market data, skipping")
                nba_weights[position] = current_weights["NBA"][position]
                continue
            
            features = features_by_pos[position]
            cur_w = list(current_weights["NBA"][position].values())
            
            print(f"\n  Position: {position}")
            
            # Per-feature diagnostic
            diag = run_per_feature_diagnostic(z_profile, opp_z, games, positions,
                                              position, features)
            print("  Per-feature diagnostic:")
            for f, s in sorted(diag.items(), key=lambda x: -x[1]):
                print(f"    {f:30s}  {s:.4f}")
            
            if args.diagnostic_only:
                nba_weights[position] = current_weights["NBA"][position]
                continue
            
            # Joint optimization
            print(f"\n  Joint optimization ({position}):")
            best_w, best_s, cur_s = optimize_position_weights(
                z_profile, opp_z, games, positions,
                position, features, cur_w,
                min_comps=5, max_comps=20,
                maxiter=args.maxiter, popsize=args.popsize)
            nba_weights[position] = best_w
            score_report[f"NBA-{position}"] = (cur_s, best_s)
        
        if nba_weights and not args.diagnostic_only:
            optimized["NBA"] = nba_weights
    
    # ── WNBA ──
    if "WNBA" in args.leagues:
        print("\n" + "=" * 60)
        print("WNBA")
        print("=" * 60)
        features_by_pos = {pos: list(current_weights["WNBA"][pos].keys()) for pos in current_weights["WNBA"]}
        # target_markets = ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV"]
        target_markets = ["fantasy points prizepicks"]
        
        wnba = StatsWNBA()
        pos_data = prepare_wnba(wnba, features_by_pos, target_markets)
        
        wnba_weights = {}
        for position in pos_data:
            if position not in current_weights["WNBA"]:
                continue
            z_profile, lookups = pos_data[position]
            opp_z, games, positions = lookups
            if not opp_z:
                wnba_weights[position] = current_weights["WNBA"][position]
                continue
            
            features = features_by_pos[position]
            cur_w = list(current_weights["WNBA"][position].values())
            
            print(f"\n  Position: {position}")
            diag = run_per_feature_diagnostic(z_profile, opp_z, games, positions,
                                              position, features)
            print("  Per-feature diagnostic:")
            for f, s in sorted(diag.items(), key=lambda x: -x[1]):
                print(f"    {f:30s}  {s:.4f}")
            
            if args.diagnostic_only:
                wnba_weights[position] = current_weights["WNBA"][position]
                continue
            
            print(f"\n  Joint optimization ({position}):")
            best_w, best_s, cur_s = optimize_position_weights(
                z_profile, opp_z, games, positions,
                position, features, cur_w,
                min_comps=5, max_comps=20,
                maxiter=args.maxiter, popsize=args.popsize)
            wnba_weights[position] = best_w
            score_report[f"WNBA-{position}"] = (cur_s, best_s)
        
        if wnba_weights and not args.diagnostic_only:
            optimized["WNBA"] = wnba_weights
    
    # ── NFL ──
    if "NFL" in args.leagues:
        print("\n" + "=" * 60)
        print("NFL")
        print("=" * 60)
        features_by_pos = {pos: list(current_weights["NFL"][pos].keys()) for pos in ["QB", "RB", "WR", "TE"]}
        # target_markets_by_pos = {
        #     "QB": ["passing yards", "passing tds", "completions", "interceptions", "rushing yards", "tds"],
        #     "RB": ["rushing yards", "carries", "receptions", "receiving yards", "tds"],
        #     "WR": ["receiving yards", "receptions", "targets", "tds"],
        #     "TE": ["receiving yards", "receptions", "targets", "tds"]
        # }
        target_markets_by_pos = {
            "QB": ["fantasy points underdog"],
            "RB": ["fantasy points underdog"],
            "WR": ["fantasy points underdog"],
            "TE": ["fantasy points underdog"]
        }
        
        nfl_stats = StatsNFL()
        pos_data = prepare_nfl(nfl_stats, features_by_pos, target_markets_by_pos)
        
        nfl_weights = {}
        for position in ["QB", "RB", "WR", "TE"]:
            if position not in pos_data:
                print(f"\n  {position}: insufficient data, skipping")
                nfl_weights[position] = current_weights["NFL"][position]
                continue
            
            z_profile, lookups = pos_data[position]
            opp_z, games, positions = lookups
            if not opp_z:
                nfl_weights[position] = current_weights["NFL"][position]
                continue
            
            features = features_by_pos[position]
            cur_w = list(current_weights["NFL"][position].values())
            
            print(f"\n  Position: {position}")
            diag = run_per_feature_diagnostic(z_profile, opp_z, games, positions,
                                              position, features, min_comps=5, max_comps=15)
            print("  Per-feature diagnostic:")
            for f, s in sorted(diag.items(), key=lambda x: -x[1]):
                print(f"    {f:30s}  {s:.4f}")
            
            if args.diagnostic_only:
                nfl_weights[position] = current_weights["NFL"][position]
                continue
            
            print(f"\n  Joint optimization ({position}):")
            best_w, best_s, cur_s = optimize_position_weights(
                z_profile, opp_z, games, positions,
                position, features, cur_w,
                min_comps=5, max_comps=15,
                maxiter=args.maxiter, popsize=args.popsize)
            nfl_weights[position] = best_w
            score_report[f"NFL-{position}"] = (cur_s, best_s)
        
        optimized["NFL"] = nfl_weights
    
    # ── NHL ──
    if "NHL" in args.leagues:
        print("\n" + "=" * 60)
        print("NHL")
        print("=" * 60)
        features_by_pos = {pos: list(current_weights["NHL"][pos].keys()) for pos in ["C", "W", "D", "G"]}
        # target_markets_by_pos = {
        #     "C": ["points", "goals", "assists", "shots", "hits", "blocked", "faceOffWins"],
        #     "W": ["points", "goals", "assists", "shots", "hits", "blocked"],
        #     "D": ["points", "goals", "assists", "shots", "hits", "blocked"],
        #     "G": ["saves", "goalsAgainst"]
        # }
        target_markets_by_pos = {
            "C": ["skater fantasy points underdog"],
            "W": ["skater fantasy points underdog"],
            "D": ["skater fantasy points underdog"],
            "G": ["goalie fantasy points underdog"]
        }
        
        nhl = StatsNHL()
        pos_data = prepare_nhl(nhl, features_by_pos, target_markets_by_pos)
        
        nhl_weights = {}
        for position in ["C", "W", "D", "G"]:
            if position not in pos_data:
                print(f"\n  {position}: insufficient data, skipping")
                nhl_weights[position] = current_weights["NHL"][position]
                continue
            
            z_profile, lookups = pos_data[position]
            opp_z, games, positions = lookups
            if not opp_z:
                nhl_weights[position] = current_weights["NHL"][position]
                continue
            
            features = features_by_pos[position]
            cur_w = list(current_weights["NHL"][position].values())
            min_k = 4 if position == "G" else 5
            
            print(f"\n  Position: {position}")
            diag = run_per_feature_diagnostic(z_profile, opp_z, games, positions,
                                              position, features, min_comps=min_k, max_comps=15)
            print("  Per-feature diagnostic:")
            for f, s in sorted(diag.items(), key=lambda x: -x[1]):
                print(f"    {f:30s}  {s:.4f}")
            
            if args.diagnostic_only:
                nhl_weights[position] = current_weights["NHL"][position]
                continue
            
            print(f"\n  Joint optimization ({position}):")
            best_w, best_s, cur_s = optimize_position_weights(
                z_profile, opp_z, games, positions,
                position, features, cur_w,
                min_comps=min_k, max_comps=20,
                maxiter=args.maxiter, popsize=args.popsize)
            nhl_weights[position] = best_w
            score_report[f"NHL-{position}"] = (cur_s, best_s)
        
        optimized["NHL"] = nhl_weights
    
    # ── Summary ──
    if score_report:
        print("\n" + "=" * 60)
        print("COMPOSITE SCORE SUMMARY")
        print("=" * 60)
        print(f"  {'Group':<15s}  {'Current':>10s}  {'Optimized':>10s}  {'Change':>10s}")
        print(f"  {'-'*15}  {'-'*10}  {'-'*10}  {'-'*10}")
        for group, (cur, opt) in sorted(score_report.items()):
            delta = opt - cur
            print(f"  {group:<15s}  {cur:>10.5f}  {opt:>10.5f}  {delta:>+10.5f}")
        print()
        print("  Note: Scores are sqrt(n)-weighted mean Spearman correlations.")
        print("  A score of 0.05-0.10 across thousands of games is statistically")
        print("  significant (p << 0.001) and meaningful as one feature among 30+.")
        print("  Per-market p-values shown above (* p<.05, ** p<.01, *** p<.001).")
    
    # ── Weight comparison ──
    if optimized and not args.diagnostic_only:
        print("\n" + "=" * 60)
        print("WEIGHT COMPARISON: Current -> Optimized")
        print("=" * 60)
        
        for league in args.leagues:
            if league not in optimized:
                continue
            print(f"\n--- {league} ---")
            
            for position in optimized[league]:
                print(f"\n  {position}:")
                old = current_weights.get(league, {}).get(position, {})
                new = optimized[league][position]
                for feature in new:
                    old_val = old.get(feature, "N/A")
                    new_val = new[feature]
                    delta = ""
                    if isinstance(old_val, (int, float)):
                        diff = new_val - old_val
                        delta = f"  ({'+' if diff >= 0 else ''}{diff:.1f})"
                    print(f"    {feature:40s}  {str(old_val):>5s} -> {new_val:>5.1f}{delta}")
    
    # ── Output ──
    if not args.diagnostic_only:
        result = current_weights.copy()
        result.update(optimized)
        output_json = json.dumps(result, indent=4)
        
        if args.save:
            filepath = pkg_resources.files(data) / "playerCompStats.json"
            with open(filepath, "w") as f:
                f.write(output_json)
            print(f"\nSaved optimized weights to {filepath}")
        elif args.output:
            with open(args.output, "w") as f:
                f.write(output_json)
            print(f"\nSaved optimized weights to {args.output}")
        else:
            print("\nOptimized weights JSON:")
            print(output_json)


if __name__ == "__main__":
    main()
