"""Pairwise correlation scoring and beam-search parlay construction.

:func:`find_correlation` takes the flat list of scored offers produced by
:func:`process_offers`, groups them by game, looks up the pre-computed
stratified correlation matrices (``{LEAGUE}_corr_same_team.csv`` and
``{LEAGUE}_corr_opposing.csv``), and annotates each offer with its most
correlated team-mate and opponent legs.  It also calls
:func:`beam_search_parlays` to enumerate the top parlay combinations.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import re
import warnings
from itertools import combinations
from math import comb
from operator import itemgetter

import line_profiler
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import multivariate_normal, norm
from tqdm import tqdm

from sportstradamus import data
from sportstradamus.helpers import banned, stat_cv, stat_map, stat_std
from sportstradamus.spiderLogger import logger

# Legacy weighting from the unified-matrix era: same-team and cross pairs
# (where the team is the offensive actor) are weighted higher than the
# team's view of the opposing team's same-team pairs.
_OFFENSIVE_PAIR_WEIGHT: float = 0.75
_DEFENSIVE_PAIR_WEIGHT: float = 1.0 - _OFFENSIVE_PAIR_WEIGHT


def _team_slice(corr_df: pd.DataFrame, team: str) -> pd.Series:
    """Return the R series for one team from a stratified correlation DataFrame.

    Args:
        corr_df: Stratified correlation DataFrame with MultiIndex
            ``(team, market_a, market_b)`` and a single ``R`` column.
        team: Team abbreviation to look up.

    Returns:
        A Series indexed by ``(market_a, market_b)`` for the requested team,
        empty if the team has no entries.
    """
    try:
        return corr_df.loc[team]["R"]
    except KeyError:
        return pd.Series([], dtype=float)


def _build_game_corr_map(
    team: str,
    opp: str,
    c_same: pd.DataFrame,
    c_opp: pd.DataFrame,
) -> dict[tuple[str, str], float]:
    """Combine same-team and opposing matrices into one pair-keyed lookup.

    Keys preserve the legacy ``_OPP_`` prefix convention so the rest of
    :func:`find_correlation` can keep using string-keyed lookups:

    * ``(a, b)`` — same-team pair on ``team``.
    * ``(_OPP_a, _OPP_b)`` — same-team pair on ``opp`` (i.e. ``team`` faces it).
    * ``(a, _OPP_b)`` — cross pair where ``a`` is on ``team`` and ``b`` on ``opp``.

    Cross pairs are summed across the two perspectives (team's gamelog and
    opp's gamelog) to mirror the legacy unified-matrix arithmetic.
    """
    c_map: dict[tuple[str, str], float] = {}

    for (a, b), r in _team_slice(c_same, team).items():
        key = (a, b)
        c_map[key] = c_map.get(key, 0.0) + r * _OFFENSIVE_PAIR_WEIGHT

    for (a, b), r in _team_slice(c_same, opp).items():
        key = (f"_OPP_{a}", f"_OPP_{b}")
        c_map[key] = c_map.get(key, 0.0) + r * _DEFENSIVE_PAIR_WEIGHT

    for (a, b), r in _team_slice(c_opp, team).items():
        key = (a, f"_OPP_{b}")
        c_map[key] = c_map.get(key, 0.0) + r * _OFFENSIVE_PAIR_WEIGHT

    for (a, b), r in _team_slice(c_opp, opp).items():
        # opp file convention: level 1 is opp's market, level 2 is team's market.
        key = (b, f"_OPP_{a}")
        c_map[key] = c_map.get(key, 0.0) + r * _OFFENSIVE_PAIR_WEIGHT

    return c_map


@line_profiler.profile
def find_correlation(offers, stats, platform):
    """Annotate offers with correlation info and build parlay candidates.

    Groups scored offers by game, loads the league correlation CSV, computes
    pairwise boost modifiers, fills ``Team Correlation`` / ``Opp Correlation``
    columns, and calls :func:`beam_search_parlays` for each game.

    Args:
        offers: List of scored offer dicts from :func:`process_offers`.
        stats: ``{league: Stats}`` dict for active leagues.
        platform: DFS platform name (e.g. ``"Underdog"``).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: ``(offer_df, parlay_df)`` where
            ``offer_df`` is the full scored slate sorted by ``Model`` and
            ``parlay_df`` has beam-search parlay candidates.
    """
    logger.info("Finding Correlations")

    new_map = stat_map[platform].copy()
    warnings.simplefilter("ignore")

    df = pd.DataFrame(offers)
    versus_mask = df["Player"].str.contains(" vs. ")
    if not df.loc[versus_mask].empty:
        df.loc[versus_mask, "Team"] = df.loc[versus_mask].apply(
            lambda x: x["Team"].split("/")[0 if x["Bet"] == "Over" else 1], axis=1
        )
        df.loc[versus_mask, "Opponent"] = df.loc[versus_mask].apply(
            lambda x: x["Opponent"].split("/")[0 if x["Bet"] == "Over" else 1], axis=1
        )

    combo_mask = df["Team"].apply(lambda x: len(set(x.split("/"))) == 1)
    df.loc[combo_mask, "Team"] = df.loc[combo_mask, "Team"].apply(lambda x: x.split("/")[0])
    df.loc[combo_mask, "Opponent"] = df.loc[combo_mask, "Opponent"].apply(lambda x: x.split("/")[0])

    df["Team Correlation"] = ""
    df["Opp Correlation"] = ""
    parlay_df = pd.DataFrame(
        columns=[
            "Game",
            "Date",
            "League",
            "Platform",
            "Model EV",
            "Books EV",
            "Boost",
            "Rec Bet",
            "Leg 1",
            "Leg 2",
            "Leg 3",
            "Leg 4",
            "Leg 5",
            "Leg 6",
            "Legs",
            "P",
            "PB",
            "Fun",
            "Bet Size",
        ]
    )
    usage_str = {"NBA": "MIN", "WNBA": "MIN", "NFL": "snap pct", "NHL": "TimeShare"}
    tiebreaker_str = {
        "NBA": "USG_PCT short",
        "WNBA": "USG_PCT short",
        "NFL": "route participation short",
        "NHL": "Fenwick short",
    }
    positions = {
        "NBA": ["P", "C", "F", "W", "B"],
        "NFL": ["QB", "WR", "RB", "TE"],
        "NHL": ["C", "W", "D", "G"],
        "WNBA": ["G", "F", "C"],
    }
    payout_table = {  # equivalent payouts when insured picks are better
        "Underdog": [3.5, 6.5, 10.9, 20.2, 39.9],
        "PrizePicks": [3, 5.3, 10, 20.8, 38.8],
        "Sleeper": [1, 1, 1, 1, 1],
        "ParlayPlay": [1, 1, 1, 1, 1],
        "Chalkboard": [1, 1, 1, 1, 1],
    }

    for league in ["NFL", "NBA", "WNBA", "MLB", "NHL"]:
        league_df = df.loc[df["League"] == league]
        if league_df.empty:
            continue
        c_same = pd.read_csv(
            pkg_resources.files(data) / f"{league}_corr_same_team.csv",
            index_col=[0, 1, 2],
        )
        c_same.rename_axis(["team", "market", "correlation"], inplace=True)
        c_same.columns = ["R"]
        c_opp = pd.read_csv(
            pkg_resources.files(data) / f"{league}_corr_opposing.csv",
            index_col=[0, 1, 2],
        )
        c_opp.rename_axis(["team", "market", "correlation"], inplace=True)
        c_opp.columns = ["R"]
        stat_data = stats.get(league)
        team_mod_map = banned[platform][league]["team"]
        opp_mod_map = banned[platform][league]["opponent"]
        if platform == "Underdog":
            league_df["Boost"] = league_df["Boost"] / 1.78

        if league != "MLB":
            league_df["Player position"] = league_df["Player position"].apply(
                lambda x: (
                    positions[league][x - 1]
                    if isinstance(x, int)
                    else [positions[league][i - 1] for i in x]
                )
            )
            combo_df = league_df.loc[league_df.Player.str.contains(r"\+|vs.")]
            league_df = league_df.loc[~league_df.index.isin(combo_df.index)]
            player_df = league_df[["Player", "Team", "Player position"]]
            player_df.drop_duplicates(inplace=True)
            stat_data.profile_market(usage_str[league])
            usage = pd.DataFrame(
                stat_data.playerProfile[[usage_str[league] + " short", tiebreaker_str[league]]]
            )
            usage.reset_index(inplace=True)
            usage.rename(
                columns={
                    "player display name": "Player",
                    "playerName": "Player",
                    "PLAYER_NAME": "Player",
                },
                inplace=True,
            )
            player_df = player_df.merge(usage, how="left").fillna(0).infer_objects(copy=False)
            ranks = (
                player_df.sort_values(tiebreaker_str[league], ascending=False)
                .groupby(["Team", "Player position"])
                .rank(ascending=False, method="first")[usage_str[league] + " short"]
                .astype(int)
            )
            player_df["Player position"] = player_df["Player position"] + ranks.astype(str)
            player_df.index = player_df.Player
            player_df = player_df["Player position"].to_dict()
            league_df["Player position"] = league_df.Player.map(player_df)
        else:
            league_df["Player position"] = league_df["Player position"].apply(
                lambda x: (
                    ("B" + str(x) if x > 0 else "P")
                    if isinstance(x, int)
                    else ["B" + str(i) if i > 0 else "P" for i in x]
                )
            )

        if league == "NHL":
            new_map.update({"Points": "points", "Blocked Shots": "blocked", "Assists": "assists"})
        if league in ("NBA", "WNBA"):
            new_map.update({"Fantasy Points": "fantasy points prizepicks"})

        league_df["cMarket"] = league_df.apply(
            lambda x: (
                [
                    x["Player position"]
                    + "."
                    + new_map.get(x["Market"].replace("H2H ", ""), x["Market"].replace("H2H ", ""))
                ]
                if isinstance(x["Player position"], str)
                else [
                    p
                    + "."
                    + new_map.get(x["Market"].replace("H2H ", ""), x["Market"].replace("H2H ", ""))
                    for p in x["Player position"]
                ]
            ),
            axis=1,
        )

        league_df["Desc"] = (
            league_df[["Player", "Bet", "Line", "Market"]].astype(str).agg(" ".join, axis=1)
        )

        league_df["Desc"] = (
            league_df["Desc"]
            + " - "
            + league_df["Model P"].multiply(100).round(1).astype(str)
            + "%, "
            + league_df["Boost"].round(2).astype(str)
            + "x"
        )

        checked_teams = []
        teams = [team for team in league_df.Team.unique() if "/" not in team]
        for team in tqdm(teams, desc=f"Checking {league} games", unit="game"):
            if team in checked_teams:
                continue
            team_df = league_df.loc[league_df["Team"] == team]
            opp = team_df.Opponent.mode().values[0]
            date = team_df.Date.mode().values[0]
            opp_df = league_df.loc[league_df["Team"] == opp]
            if not opp_df.empty:
                opp_df["cMarket"] = opp_df.apply(
                    lambda x: ["_OPP_" + c for c in x["cMarket"]], axis=1
                )
            split_df = league_df.loc[
                league_df["Team"].str.contains("/")
                & (league_df["Team"].str.contains(team) | league_df["Team"].str.contains(opp))
            ]
            if not split_df.empty:
                split_df["cMarket"] = split_df.apply(
                    lambda x: [
                        ("_OPP_" + c) if (x["Team"].split("/")[d] == opp) else c
                        for d, c in enumerate(x["cMarket"])
                    ],
                    axis=1,
                )
            game_df = pd.concat([team_df, opp_df, split_df])
            game_df.drop_duplicates(subset="Desc", inplace=True)
            checked_teams.append(team)
            checked_teams.append(opp)

            c_map = _build_game_corr_map(team, opp, c_same, c_opp)

            game_df.reset_index(drop=True, inplace=True)
            game_dict = game_df.to_dict("index")

            idx = game_df.loc[
                (game_df["Books"] > 0.85)
                & (game_df["Books P"] >= 0.25)
                & (game_df["Model"] > 1)
                & (game_df["Model P"] >= 0.3)
            ].sort_values("K", ascending=False)
            idx = idx.drop_duplicates(subset=["Player", "Team", "Market", "Line"])
            idx = idx.groupby("Player").head(6)
            idx = (
                idx.sort_values(["Model", "Books"], ascending=False)
                .groupby("Team")
                .head(30)
                .sort_values(["Team", "Player"])
            )
            idx = (
                idx.sort_values(["Model", "Books"], ascending=False)
                .head(40)
                .sort_values(["Team", "Player"])
            )
            bet_df = idx.to_dict("index")

            C = np.eye(len(game_dict))
            M = np.zeros([len(game_dict), len(game_dict)])
            p_model = game_df["Model P"].to_numpy()
            p_books = game_df["Books P"].to_numpy()
            boosts = game_df["Boost"].to_numpy()
            V = p_model * (1 - p_model)
            V = V.reshape(len(game_dict), 1) * V
            V = np.sqrt(V)
            P = p_model.reshape(len(p_model), 1) * p_model
            Vb = p_books * (1 - p_books)
            Vb = Vb.reshape(len(game_dict), 1) * Vb
            Vb = np.sqrt(Vb)
            Pb = p_books.reshape(len(p_books), 1) * p_books
            for i, j in combinations(range(len(game_dict)), 2):
                leg1 = game_dict[i]
                leg2 = game_dict[j]
                cm1 = leg1["cMarket"]
                cm2 = leg2["cMarket"]
                b1 = [leg1["Bet"]] * len(cm1)
                b2 = [leg2["Bet"]] * len(cm2)
                n1 = leg1["Player"]
                n2 = leg2["Player"]

                if "vs." in n1:
                    b1[1] = "Under" if b1[0] == "Over" else "Over"

                if "vs." in n2:
                    b2[1] = "Under" if b2[0] == "Over" else "Over"

                rho = 0
                boost = 0 if (n1 in n2 or n2 in n1) else 1
                for xi, x in enumerate(cm1):
                    for yi, y in enumerate(cm2):
                        increment = c_map.get((x, y), c_map.get((y, x), 0))
                        if b1[xi] != b2[yi]:
                            increment = -increment
                        rho += increment

                        mod_map = team_mod_map if ("_OPP_" in x) == ("_OPP_" in y) else opp_mod_map

                        x_key = re.sub(r"[0-9]", "", x)
                        y_key = re.sub(r"[0-9]", "", y)
                        x_key = x_key.replace("_OPP_", "")
                        y_key = y_key.replace("_OPP_", "")

                        modifier = mod_map.get(frozenset([x_key, y_key]), [1, 1])
                        boost *= modifier[0] if b1[xi] == b2[yi] else modifier[1]

                C[i, j] = C[j, i] = rho / len(cm1) / len(cm2)
                M[i, j] = M[j, i] = boost

            EV = (
                np.multiply(
                    np.multiply(np.exp(np.multiply(C, V)), P),
                    boosts.reshape(len(boosts), 1) * M * boosts,
                )
                * payout_table[platform][0]
            )
            EVb = (
                np.multiply(
                    np.multiply(np.exp(np.multiply(C, Vb)), Pb),
                    boosts.reshape(len(boosts), 1) * M * boosts,
                )
                * payout_table[platform][0]
            )

            for i, offer in game_df.iterrows():
                indices = (EV[:, i] > 0.95) & (C[:, i] > 0.05) & (EVb[:, i] > 0.9)
                corr = game_df.loc[indices].copy()
                corr["Corr Mult"] = np.exp(C[indices, i] * V[indices, i])
                corr = corr.sort_values("Corr Mult", ascending=False).groupby("Player").head(1)
                same = corr.loc[corr["Team"] == offer["Team"]]
                other = corr.loc[corr["Team"] != offer["Team"]]
                df.loc[
                    (df["Player"] == offer["Player"]) & (df["Market"] == offer["Market"]),
                    "Team Correlation",
                ] = ", ".join(
                    (same["Desc"] + " (" + same["Corr Mult"].round(2).astype(str) + "x)").to_list()
                )
                df.loc[
                    (df["Player"] == offer["Player"]) & (df["Market"] == offer["Market"]),
                    "Opp Correlation",
                ] = ", ".join(
                    (
                        other["Desc"] + " (" + other["Corr Mult"].round(2).astype(str) + "x)"
                    ).to_list()
                )

            info = {
                "Game": "/".join(sorted([team, opp])),
                "Date": date,
                "League": league,
                "Platform": platform,
            }
            best_bets = []
            if not (platform in ["Chalkboard", "ParlayPlay"] and league == "MLB"):
                payouts = payout_table[platform]
                max_boost = 2.5 if platform == "Underdog" else 60
                best_bets = beam_search_parlays(
                    idx,
                    EV,
                    C,
                    M,
                    p_model,
                    p_books,
                    boosts,
                    payouts,
                    max_boost,
                    bet_df,
                    info,
                    team,
                    opp,
                )

                if len(best_bets) > 0:
                    bets = pd.DataFrame(best_bets)

                    df5 = (
                        pd.concat(
                            [
                                bets.sort_values("Model EV", ascending=False).head(300),
                                bets.sort_values("Rec Bet", ascending=False).head(300),
                                bets.sort_values("Fun", ascending=False).head(300),
                            ]
                        )
                        .drop_duplicates()
                        .sort_values("Model EV", ascending=False)
                    )

                    if len(df5) > 5:
                        rho_matrix = np.zeros([len(df5), len(df5)])
                        bets = df5["Bet ID"].to_list()
                        for i, j in tqdm(
                            combinations(range(len(df5)), 2),
                            desc="Filtering...",
                            leave=False,
                            total=comb(len(df5), 2),
                        ):
                            bet1 = bets[i]
                            bet2 = bets[j]
                            rho_cross = np.mean(C[np.ix_(bet1, bet2)])
                            rho_bet1 = np.mean(C[np.ix_(bet1, bet1)])
                            rho_bet2 = np.mean(C[np.ix_(bet2, bet2)])

                            rho_matrix[i, j] = np.clip(
                                rho_cross / np.sqrt(rho_bet1) / np.sqrt(rho_bet2), -1, 0.999
                            )

                        X = np.concatenate([row[i + 1 :] for i, row in enumerate(1 - rho_matrix)])
                        Z = linkage(X, "ward")
                        df5["Family"] = fcluster(Z, 3, criterion="maxclust")

                    else:
                        df5["Family"] = 1

                    parlay_df = pd.concat(
                        [parlay_df, df5.drop(columns="Bet ID")], ignore_index=True
                    )

    if platform == "Underdog":
        payouts = [0, 0, 3.5, 6.5, 6, 10, 25]
        parlay_df["Boost"] = parlay_df["Bet Size"].apply(lambda x: payouts[x]) * parlay_df["Boost"]

    return df.dropna(subset="Model").sort_values("Model", ascending=False), parlay_df


def beam_search_parlays(
    idx, EV, C, M, p_model, p_books, boosts, payouts, max_boost, bet_df, info, team, opp
):
    """Enumerate top parlay combinations via beam search.

    Extends parlays leg-by-leg up to ``len(payouts) + 1`` legs, keeping
    only the top ``K=1000`` candidates at each size by geometric-mean EV.
    Full correlated-probability evaluation runs on the survivors.

    Args:
        idx: DataFrame of candidate legs (filtered from the game DataFrame).
        EV: Pairwise EV matrix for the full game DataFrame.
        C: Pairwise correlation matrix.
        M: Pairwise boost-modifier matrix.
        p_model: Model probability vector.
        p_books: Books probability vector.
        boosts: Boost multiplier vector.
        payouts: Platform payout table list.
        max_boost: Max allowed product boost for a parlay.
        bet_df: Dict of leg metadata keyed by DataFrame index.
        info: ``{Game, Date, League, Platform}`` metadata dict.
        team: Home team abbreviation.
        opp: Away team abbreviation.

    Returns:
        list[dict]: Parlay candidate dicts ready for DataFrame construction.
    """
    K = 1000
    max_bet_size = len(payouts) + 1
    leg_indices = sorted(idx.index.to_numpy())
    leg_players = {i: bet_df[i]["Player"] for i in leg_indices}
    leg_teams = {i: bet_df[i]["Team"] for i in leg_indices}

    candidates = [(i,) for i in leg_indices]
    all_results = []

    for target_size in tqdm(
        range(2, max_bet_size + 1), desc=f"{info['League']}, {team}/{opp} Parlays", leave=False
    ):
        next_candidates = []

        for parlay in candidates:
            used_players = {leg_players[i] for i in parlay}
            last_idx = parlay[-1]

            for new_leg in leg_indices:
                if new_leg <= last_idx:
                    continue
                new_player = leg_players[new_leg]
                if new_player in used_players:
                    continue
                if any(new_player in p or p in new_player for p in used_players):
                    continue

                extended = (*parlay, new_leg)

                n_pairs = target_size * (target_size - 1) // 2
                ev_prod = np.prod(EV[np.ix_(extended, extended)][np.triu_indices(target_size, 1)])
                geo_mean = ev_prod ** (1 / n_pairs)
                if geo_mean < 1.05:
                    continue

                next_candidates.append((extended, geo_mean))

        next_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = next_candidates[:K]

        payout_base = payouts[target_size - 2]

        for parlay, _ in top_candidates:
            bet_id = parlay
            bet_size = target_size

            covers_team = any(team in leg_teams[i] for i in bet_id)
            covers_opp = any(opp in leg_teams[i] for i in bet_id)
            if not (covers_team and covers_opp):
                continue

            boost = np.prod(M[np.ix_(bet_id, bet_id)][np.triu_indices(bet_size, 1)]) * np.prod(
                boosts[np.ix_(bet_id)]
            )
            if boost <= 0.7 or boost > max_boost:
                continue

            pb = p_books[np.ix_(bet_id)]
            prev_pb = np.prod(pb) * boost * payout_base
            if prev_pb < 0.9:
                continue

            p = p_model[np.ix_(bet_id)]
            prev_p = np.prod(p) * boost * payout_base
            if prev_p < 1.5:
                continue

            SIG = C[np.ix_(bet_id, bet_id)]
            if any(np.linalg.eigvalsh(SIG) < 0.0001):
                continue

            payout = np.clip(payout_base * boost, 1, 100)
            p = payout * multivariate_normal.cdf(norm.ppf(p), np.zeros(bet_size), SIG)
            pb = p / prev_p * prev_pb
            units = (p - 1) / (payout - 1) / 0.05

            if units < 0.5 or p < 2 or pb < 0.9:
                continue

            bet = itemgetter(*bet_id)(bet_df)
            parlay_dict = info | {
                "Model EV": p,
                "Books EV": pb,
                "Boost": boost,
                "Rec Bet": units,
                "Leg 1": "",
                "Leg 2": "",
                "Leg 3": "",
                "Leg 4": "",
                "Leg 5": "",
                "Leg 6": "",
                "Legs": ", ".join([leg["Desc"] for leg in bet]),
                "Bet ID": bet_id,
                "P": prev_p,
                "PB": prev_pb,
                "Fun": np.sum(
                    [
                        3
                        - (
                            np.abs(leg["Line"])
                            / stat_std.get(info["League"], {}).get(leg["Market"], 1)
                        )
                        if ("H2H" in leg["Desc"])
                        else 2
                        - 1 / stat_cv.get(info["League"], {}).get(leg["Market"], 1)
                        + leg["Line"] / stat_std.get(info["League"], {}).get(leg["Market"], 1)
                        for leg in bet
                        if (leg["Bet"] == "Over") or ("H2H" in leg["Desc"])
                    ]
                ),
                "Bet Size": bet_size,
                "Leg Probs": tuple(bet_df[i]["Model P"] for i in bet_id),
                "Corr Pairs": tuple(SIG[np.triu_indices(bet_size, 1)]),
                "Boost Pairs": tuple(M[np.ix_(bet_id, bet_id)][np.triu_indices(bet_size, 1)]),
                "Indep P": float(np.prod(p_model[np.ix_(bet_id)]) * payout),
                "Indep PB": float(np.prod(p_books[np.ix_(bet_id)]) * payout),
            }
            for j in range(bet_size):
                parlay_dict["Leg " + str(j + 1)] = bet[j]["Desc"]

            all_results.append(parlay_dict)

        candidates = [p for p, _ in top_candidates]

    return all_results
