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
from typing import Literal

import line_profiler
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import multivariate_normal, norm
from tqdm import tqdm

from sportstradamus import data
from sportstradamus.helpers import banned, stat_cv, stat_map, stat_std, underdog_payouts
from sportstradamus.spiderLogger import logger

# Legacy weighting from the unified-matrix era: same-team and cross pairs
# (where the team is the offensive actor) are weighted higher than the
# team's view of the opposing team's same-team pairs.
_OFFENSIVE_PAIR_WEIGHT: float = 0.75
_DEFENSIVE_PAIR_WEIGHT: float = 1.0 - _OFFENSIVE_PAIR_WEIGHT

# --- Beam-search constants (audit §2.3 — promoted from inline magic numbers) -

# Beam width: max parlay candidates carried between sizes. Empirically large
# enough that the survivors at sizes 5/6 don't get pruned by ranking jitter.
_BEAM_WIDTH: int = 1000

# Per-step pairwise-EV floor. Below this, the candidate is too noisy to keep.
_PARLAY_GEO_MEAN_FLOOR: float = 1.05

# Boost-product gates: drop entries whose modifier product implies hard-banned
# pairs (low) or runaway promo stacking (high).
_MIN_PRODUCT_BOOST: float = 0.7
_MAX_BOOST_UNDERDOG: float = 2.5
_MAX_BOOST_OTHER: float = 60.0

# Final EV gates (post-copula evaluation).
_BOOKS_EV_FLOOR: float = 0.9
_MODEL_EV_PRECHECK_FLOOR: float = 1.5
_MODEL_EV_FINAL_FLOOR: float = 2.0
_KELLY_UNITS_FLOOR: float = 0.5

# PSD repair: floor on minimum eigenvalue. Sub-floor matrices are projected
# to the nearest PSD instead of dropped (audit §2.1, finding bullet 3).
_PSD_EIG_TOLERANCE: float = 1e-4

# Payout multiplier clip. Caps runaway boost products; matches legacy.
_PAYOUT_CLIP_LO: float = 1.0
_PAYOUT_CLIP_HI: float = 100.0

# Push-handling thresholds.
# Below this, a leg's push prob is treated as zero so the fast (analytical)
# mvn.cdf path runs. Set just above floating-point noise from ``get_push_prob``.
_PUSH_PROB_FLOOR: float = 1e-6
# Monte-Carlo sample count for the push-aware copula (audit follow-up).
_PUSH_MC_SAMPLES: int = 50_000

# Kelly sizing denominator: 5% bankroll per unit. Legacy.
_KELLY_BANKROLL_FRACTION: float = 0.05


def _legacy_underdog_overwrite_payouts() -> dict[int, float]:
    """Reproduce the audit §2.4 legacy line-498 overwrite table.

    Mixes insurance multipliers for sizes 2-3 with power multipliers for sizes
    4-6. Preserved verbatim so ``legacy=True`` runs match historical output.
    """
    return {2: 3.5, 3: 6.5, 4: 6.0, 5: 10.0, 6: 25.0}


def _legacy_underdog_search_payouts() -> list[float]:
    """Reproduce the audit §2.4 legacy in-search payout list (insurance line)."""
    return [3.5, 6.5, 10.9, 20.2, 39.9]


def _payout_curve_for(
    platform: str,
    contest_variant: Literal["power", "flex", "insurance", "rivals"],
    legacy: bool,
) -> tuple[list[float], dict[int, list[float]]]:
    """Build the (per-size search list, per-(size,misses) payout table) for a platform.

    The first return drives beam-search ranking (single multiplier per size,
    indexed ``[bet_size - 2]``). The second drives push-aware EV and the
    display ``Boost`` column (full payout curve indexed by miss count).

    Underdog pulls from ``data/underdog_payouts.json`` per the chosen
    ``contest_variant``. Other platforms (PrizePicks, Sleeper, ParlayPlay,
    Chalkboard) keep the legacy single-payout table.
    """
    if legacy and platform == "Underdog":
        search = _legacy_underdog_search_payouts()
        # Display table mirrors the legacy line-498 overwrite (mixed regime).
        legacy_overwrite = _legacy_underdog_overwrite_payouts()
        full_curve = {sz: [legacy_overwrite[sz], 0.0] for sz in legacy_overwrite}
        return search, full_curve

    legacy_tables: dict[str, list[float]] = {
        "PrizePicks": [3.0, 5.3, 10.0, 20.8, 38.8],
        "Sleeper": [1.0, 1.0, 1.0, 1.0, 1.0],
        "ParlayPlay": [1.0, 1.0, 1.0, 1.0, 1.0],
        "Chalkboard": [1.0, 1.0, 1.0, 1.0, 1.0],
    }
    if platform != "Underdog":
        lst = legacy_tables[platform]
        full = {i + 2: [lst[i], 0.0] for i in range(len(lst))}
        return lst, full

    variant_table = underdog_payouts[contest_variant]
    if contest_variant == "flex":
        full_curve = {int(sz): [float(v) for v in row] for sz, row in variant_table.items()}
    else:
        full_curve = {int(sz): [float(variant_table[sz]), 0.0] for sz in variant_table}

    sizes = sorted(full_curve.keys())
    min_size = sizes[0]
    max_size = sizes[-1]
    search = [full_curve[sz][0] for sz in range(min_size, max_size + 1)]
    return search, full_curve


def _nearest_psd(sigma: np.ndarray, tol: float = _PSD_EIG_TOLERANCE) -> np.ndarray:
    """Project a symmetric matrix to the nearest PSD via eigenvalue clipping.

    Symmetrize, clip eigenvalues at ``tol``, then rescale the diagonal back to
    1 so single-leg variances stay unit (the inputs are correlation matrices).

    Args:
        sigma: Symmetric ``(n, n)`` matrix from ``C[bet_id, bet_id]``.
        tol: Eigenvalue floor; matches the PSD acceptance threshold elsewhere.

    Returns:
        np.ndarray: PSD ``(n, n)`` matrix with unit diagonal.
    """
    sigma = (sigma + sigma.T) / 2
    eigvals, eigvecs = np.linalg.eigh(sigma)
    eigvals = np.clip(eigvals, tol, None)
    repaired = (eigvecs * eigvals) @ eigvecs.T
    diag_scale = 1.0 / np.sqrt(np.diag(repaired))
    return repaired * diag_scale[:, None] * diag_scale[None, :]


def _expected_payout_with_pushes(
    p_win: np.ndarray,
    p_push: np.ndarray,
    sigma: np.ndarray,
    bet_size: int,
    boost: float,
    payout_curve: dict[int, list[float]],
    rng: np.random.Generator | None = None,
) -> float:
    """Expected payout for a parlay where some legs may push.

    Samples ``_PUSH_MC_SAMPLES`` draws from the multivariate normal copula,
    classifies each leg as WIN / PUSH / LOSS via inverse-CDF cuts, and applies
    the variant payout curve at the resulting (effective_size, misses) cell.
    Pushes drop the parlay one leg per Underdog rules; an entry that pushes
    below the minimum bet size with no losses is treated as a refund (×1).

    Args:
        p_win: Per-leg chosen-side probability (already direction-adjusted).
        p_push: Per-leg push probability. Zeros where push is impossible.
        sigma: PSD-repaired correlation matrix for the parlay's legs.
        bet_size: Number of legs (``len(p_win)``).
        boost: Modifier-product boost for this parlay.
        payout_curve: ``{size: [mult_at_0_misses, mult_at_1_miss, ...]}``.
        rng: Optional ``np.random.Generator`` for deterministic tests.

    Returns:
        float: Expected payout, ready to be compared against the EV floor.
    """
    rng = rng if rng is not None else np.random.default_rng()
    samples = rng.multivariate_normal(np.zeros(bet_size), sigma, size=_PUSH_MC_SAMPLES)

    # Per leg: cuts split the standard normal into LOSS / PUSH / WIN bands so
    # the marginal probabilities match (p_lose, p_push, p_win) exactly.
    p_lose = np.clip(1.0 - p_win - p_push, 0.0, 1.0)
    cut_lose = norm.ppf(np.clip(p_lose, 1e-9, 1 - 1e-9))
    cut_push_top = norm.ppf(np.clip(p_lose + p_push, 1e-9, 1 - 1e-9))

    # Classification: 0 = LOSS, 1 = PUSH, 2 = WIN.
    classification = np.where(
        samples < cut_lose,
        0,
        np.where(samples < cut_push_top, 1, 2),
    )

    pushes = (classification == 1).sum(axis=1)
    losses = (classification == 0).sum(axis=1)
    eff_size = bet_size - pushes

    # Flat lookup table: lookup[size, misses] → payout multiplier. Sizes below
    # the parlay minimum are treated as refund (×1) iff there are no losses.
    max_idx = bet_size + 1
    lookup = np.zeros((max_idx, max_idx), dtype=float)
    for sz in range(2, bet_size + 1):
        curve = payout_curve.get(sz)
        if curve is None:
            continue
        for miss_idx, mult in enumerate(curve):
            if miss_idx < max_idx:
                lookup[sz, miss_idx] = float(mult)

    safe_size = np.clip(eff_size, 0, max_idx - 1)
    safe_miss = np.clip(losses, 0, max_idx - 1)
    payouts = lookup[safe_size, safe_miss]

    # Special-case the "all pushes (no losses)" outcome: refund.
    all_push_no_loss = (eff_size < 2) & (losses == 0)
    payouts = np.where(all_push_no_loss, 1.0, payouts)
    # Sub-minimum size with at least one loss → bust.
    payouts = np.where((eff_size < 2) & (losses > 0), 0.0, payouts)

    return float(np.clip(boost, 0.0, _PAYOUT_CLIP_HI) * payouts.mean())


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
def find_correlation(
    offers,
    stats,
    platform,
    *,
    contest_variant: Literal["power", "flex", "insurance", "rivals"] = "power",
    legacy: bool = False,
):
    """Annotate offers with correlation info and build parlay candidates.

    Groups scored offers by game, loads the league correlation CSV, computes
    pairwise boost modifiers, fills ``Team Correlation`` / ``Opp Correlation``
    columns, and calls :func:`beam_search_parlays` for each game.

    Args:
        offers: List of scored offer dicts from :func:`process_offers`.
        stats: ``{league: Stats}`` dict for active leagues.
        platform: DFS platform name (e.g. ``"Underdog"``).
        contest_variant: Underdog contest variant ("power", "flex",
            "insurance", "rivals"). Ignored for non-Underdog platforms.
            Default "power" matches the displayed Boost column historically;
            ranking changes vs. legacy because legacy ranked at the insurance
            line — use ``legacy=True`` for bit-for-bit historical behavior.
        legacy: When True, reproduce the pre-2026.05 pipeline verbatim — no
            PSD repair, no push-aware EV, mixed insurance/power Boost
            overwrite at line 498. Removed next release.

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
    # Search-list and full payout curve come from data/underdog_payouts.json
    # (Underdog) or the per-platform legacy table. ``search_payouts`` is the
    # single-multiplier-per-size list used inside the beam-search ranking;
    # ``full_payouts`` is the per-(size, miss-count) lookup driving push-aware
    # EV and the display Boost column.
    search_payouts, full_payouts = _payout_curve_for(platform, contest_variant, legacy)

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
            # ``Push P`` is added by :func:`model_prob` for integer-line discrete
            # markets; missing for combo legs etc. — fill 0.0 so the analytical
            # mvn.cdf path still runs for those legs.
            if "Push P" in game_df.columns:
                p_push = game_df["Push P"].fillna(0.0).to_numpy()
            else:
                p_push = np.zeros(len(game_df), dtype=float)
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
                * search_payouts[0]
            )
            EVb = (
                np.multiply(
                    np.multiply(np.exp(np.multiply(C, Vb)), Pb),
                    boosts.reshape(len(boosts), 1) * M * boosts,
                )
                * search_payouts[0]
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
                max_boost = _MAX_BOOST_UNDERDOG if platform == "Underdog" else _MAX_BOOST_OTHER
                best_bets = beam_search_parlays(
                    idx,
                    EV,
                    C,
                    M,
                    p_model,
                    p_books,
                    p_push,
                    boosts,
                    search_payouts,
                    full_payouts,
                    max_boost,
                    bet_df,
                    info,
                    team,
                    opp,
                    contest_variant=contest_variant,
                    legacy=legacy,
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

    if legacy and platform == "Underdog":
        # Legacy line-498 overwrite (audit §2.4): mixed insurance/power
        # multipliers indexed by bet size. Preserved verbatim under
        # ``legacy=True`` so historical exports stay reproducible.
        legacy_overwrite = _legacy_underdog_overwrite_payouts()
        parlay_df["Boost"] = (
            parlay_df["Bet Size"].apply(lambda x: legacy_overwrite.get(int(x), 0.0))
            * parlay_df["Boost"]
        )

    return df.dropna(subset="Model").sort_values("Model", ascending=False), parlay_df


def beam_search_parlays(
    idx,
    EV,
    C,
    M,
    p_model,
    p_books,
    p_push,
    boosts,
    payouts,
    full_payouts,
    max_boost,
    bet_df,
    info,
    team,
    opp,
    *,
    contest_variant: Literal["power", "flex", "insurance", "rivals"] = "power",
    legacy: bool = False,
):
    """Enumerate top parlay combinations via beam search.

    Extends parlays leg-by-leg up to ``len(payouts) + 1`` legs, keeping
    only the top ``_BEAM_WIDTH`` candidates at each size by geometric-mean EV.
    Full correlated-probability evaluation runs on the survivors.

    Args:
        idx: DataFrame of candidate legs (filtered from the game DataFrame).
        EV: Pairwise EV matrix for the full game DataFrame.
        C: Pairwise correlation matrix.
        M: Pairwise boost-modifier matrix.
        p_model: Model probability vector (chosen-side win probability).
        p_books: Books probability vector.
        p_push: Per-leg push probability (P(stat == line)). Zero for
            non-integer lines and for continuous-distribution markets.
        boosts: Boost multiplier vector.
        payouts: Search-list payout multipliers (length = max_size - 1),
            indexed by ``bet_size - 2``. Drives the ranking heuristic and the
            books-EV pre-check.
        full_payouts: Per-(size, miss-count) payout curve. Drives push-aware
            EV and the displayed ``Boost`` column.
        max_boost: Max allowed product boost for a parlay.
        bet_df: Dict of leg metadata keyed by DataFrame index.
        info: ``{Game, Date, League, Platform}`` metadata dict.
        team: Home team abbreviation.
        opp: Away team abbreviation.
        contest_variant: Underdog contest variant. Affects payout curve
            interpretation in :func:`_expected_payout_with_pushes`.
        legacy: When True, reproduce pre-2026.05 scoring (no PSD repair, no
            push-aware EV, bare modifier-product Boost in the output).

    Returns:
        list[dict]: Parlay candidate dicts ready for DataFrame construction.
    """
    K = _BEAM_WIDTH
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
                if geo_mean < _PARLAY_GEO_MEAN_FLOOR:
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
            if boost <= _MIN_PRODUCT_BOOST or boost > max_boost:
                continue

            pb = p_books[np.ix_(bet_id)]
            prev_pb = np.prod(pb) * boost * payout_base
            if prev_pb < _BOOKS_EV_FLOOR:
                continue

            p = p_model[np.ix_(bet_id)]
            prev_p = np.prod(p) * boost * payout_base
            if prev_p < _MODEL_EV_PRECHECK_FLOOR:
                continue

            SIG = C[np.ix_(bet_id, bet_id)]
            min_eig = np.min(np.linalg.eigvalsh(SIG))
            if legacy:
                # Legacy: drop non-PSD subsets instead of repairing.
                if min_eig < _PSD_EIG_TOLERANCE:
                    continue
            elif min_eig < _PSD_EIG_TOLERANCE:
                SIG = _nearest_psd(SIG)

            payout = np.clip(payout_base * boost, _PAYOUT_CLIP_LO, _PAYOUT_CLIP_HI)

            push_legs = p_push[np.ix_(bet_id)]
            has_pushes = bool(np.any(push_legs > _PUSH_PROB_FLOOR))

            if has_pushes and not legacy:
                p = _expected_payout_with_pushes(
                    p,
                    push_legs,
                    SIG,
                    bet_size,
                    boost=boost,
                    payout_curve=full_payouts,
                )
            else:
                p = payout * multivariate_normal.cdf(norm.ppf(p), np.zeros(bet_size), SIG)

            pb = p / prev_p * prev_pb
            units = (p - 1) / (payout - 1) / _KELLY_BANKROLL_FRACTION

            if units < _KELLY_UNITS_FLOOR or p < _MODEL_EV_FINAL_FLOOR or pb < _BOOKS_EV_FLOOR:
                continue

            bet = itemgetter(*bet_id)(bet_df)
            # Display Boost: under legacy, return the bare modifier product
            # (the post-search line-498 overwrite multiplies by per-size
            # payout). Under the new path, return the payout-inclusive value
            # so the column reflects the same EV that drove ranking.
            display_boost = boost if legacy else payout
            parlay_dict = info | {
                "Model EV": p,
                "Books EV": pb,
                "Boost": display_boost,
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
