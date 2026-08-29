"""Pairwise correlation scoring and parlay-construction entry point.

:func:`find_correlation` takes the flat list of scored offers produced by
:func:`process_offers`, groups them by game, looks up the pre-computed
stratified correlation matrices (``leagues/{league}/corr_same_team.parquet`` and
``leagues/{league}/corr_opposing.parquet``), and annotates each offer with its most
correlated team-mate and opponent legs.  It also calls
:func:`sportstradamus.prediction.parlay.beam_search_parlays` to enumerate the
top parlay combinations.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import re
import warnings
from itertools import combinations
from typing import Literal

import line_profiler
import numpy as np
import pandas as pd
from tqdm import tqdm

from sportstradamus import data
from sportstradamus.helpers import UNDERDOG_BOOST_BASELINE, banned, stat_map
from sportstradamus.prediction.parlay import (
    GameArrays,
    GameScoringContext,
    beam_search_parlays,
    resolve_leg_stat,
)
from sportstradamus.prediction.payouts import SLEEPER_FULL_REFUND_MAX_SIZE, payout_curve_for
from sportstradamus.spiderLogger import logger

# Legacy weighting from the unified-matrix era: same-team and cross pairs
# (where the team is the offensive actor) are weighted higher than the
# team's view of the opposing team's same-team pairs.
_OFFENSIVE_PAIR_WEIGHT: float = 0.75
_DEFENSIVE_PAIR_WEIGHT: float = 1.0 - _OFFENSIVE_PAIR_WEIGHT

# Boost-product caps for the parlay search (consumed by ``beam_search_parlays``
# via the ``max_boost`` arg). Underdog promo stacking is tighter than the rest.
_MAX_BOOST_UNDERDOG: float = 2.5
_MAX_BOOST_OTHER: float = 60.0


# Pre-filter gates for offers entering the per-game beam search. Hand-tuned to
# keep only book-supported, model-favored legs out of the cartesian explosion.
_OFFER_BOOKS_EV_FLOOR: float = 0.85
_OFFER_BOOKS_PROB_FLOOR: float = 0.25
_OFFER_MODEL_EV_FLOOR: float = 1.0
_OFFER_MODEL_PROB_FLOOR: float = 0.3

# Per-grouping caps that bound the candidate set fed to beam search.
_MAX_OFFERS_PER_PLAYER: int = 6
_MAX_OFFERS_PER_TEAM: int = 30
_MAX_OFFERS_PER_GAME: int = 40

# Display-correlation gates: when annotating an offer with top correlated legs,
# surface only partners whose EV / correlation clear these display thresholds.
_DISPLAY_MODEL_EV_FLOOR: float = 0.95
_DISPLAY_CORR_FLOOR: float = 0.05
_DISPLAY_BOOKS_EV_FLOOR: float = 0.9

# Top-N retained per sort key (Model EV / Rec Bet / Fun) before dedup.
# Three sort views deduped → up to ~3× this many survivors.
_PARLAY_TOP_N_PER_SORT: int = 300

# Per-league player-position resolution (consumed by _resolve_player_positions):
# the usage-profile stat that ranks players within a (team, position) group, its
# tiebreaker column, and the depth-chart position labels indexed by lineup slot.
# MLB is absent — it ranks by batting order instead.
_LEAGUE_USAGE_STAT = {"NBA": "MIN", "WNBA": "MIN", "NFL": "snap pct", "NHL": "TimeShare"}
_LEAGUE_USAGE_TIEBREAKER = {
    "NBA": "USG_PCT short",
    "WNBA": "USG_PCT short",
    "NFL": "route participation short",
    "NHL": "Fenwick short",
}
_LEAGUE_POSITIONS = {
    "NBA": ["P", "C", "F", "W", "B"],
    "NFL": ["QB", "WR", "RB", "TE"],
    "NHL": ["C", "W", "D", "G"],
    "WNBA": ["G", "F", "C"],
}


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


def _leg_bets(player: str, bet: str, n_markets: int) -> list[str]:
    """Per-cMarket Over/Under labels for a leg; the 2nd flips for ``vs.`` props."""
    bets = [bet] * n_markets
    if "vs." in player:
        bets[1] = "Under" if bet == "Over" else "Over"
    return bets


def _leg_pair_corr_boost(leg1, leg2, c_map, team_mod_map, opp_mod_map):
    """Correlation and boost modifier for one leg pair (the cm1×cm2 inner loops).

    Returns ``(rho, boost)`` with ``rho`` already averaged over the two cMarket
    lists, ready to drop symmetrically into the game's C and M matrices.
    """
    cm1 = leg1["cMarket"]
    cm2 = leg2["cMarket"]
    b1 = _leg_bets(leg1["Player"], leg1["Bet"], len(cm1))
    b2 = _leg_bets(leg2["Player"], leg2["Bet"], len(cm2))
    n1 = leg1["Player"]
    n2 = leg2["Player"]

    rho = 0
    boost = 0 if (n1 in n2 or n2 in n1) else 1
    for xi, x in enumerate(cm1):
        for yi, y in enumerate(cm2):
            increment = c_map.get((x, y), c_map.get((y, x), 0))
            if b1[xi] != b2[yi]:
                increment = -increment
            rho += increment

            mod_map = team_mod_map if ("_OPP_" in x) == ("_OPP_" in y) else opp_mod_map

            x_key = re.sub(r"[0-9]", "", x).replace("_OPP_", "")
            y_key = re.sub(r"[0-9]", "", y).replace("_OPP_", "")

            modifier = mod_map.get(frozenset([x_key, y_key]), [1, 1])
            boost *= modifier[0] if b1[xi] == b2[yi] else modifier[1]

    return rho / len(cm1) / len(cm2), boost


def _leg_opp_boost(game_df, platform):
    """Opposite-side payout multiplier per leg — Sleeper Flex devig input only.

    ``Boost_Over``/``Boost_Under`` mean different things per platform: for
    Sleeper they are raw decimal odds (books.py::get_sleeper); for Underdog
    they are baseline-scaled promo components (model_prob.py's
    UNDERDOG_BOOST_BASELINE multiply). Gate on platform explicitly rather
    than column presence alone, or Underdog's values would silently feed
    no_vig_odds as if they were fair decimal odds.
    """
    if platform != "Sleeper" or "Boost_Over" not in game_df.columns:
        return np.full(len(game_df), np.nan)
    return np.where(
        game_df["Bet"].to_numpy() == "Over",
        game_df["Boost_Under"].to_numpy(),
        game_df["Boost_Over"].to_numpy(),
    )


def _leg_shrinkage(game_df, platform, league, shrinkage_cache):
    """Per-leg Kelly shrinkage weight, cached by ``(league, canonical market)``.

    ``shrinkage_cache`` is caller-owned and shared across every game in one
    league's processing loop — ``resolve_market_shrinkage`` hits training/CLV
    I/O per call, and markets repeat heavily across a slate's legs, so this
    avoids re-resolving the same cell hundreds of times per league. Legs
    whose ``Market`` has no ``stat_map`` entry for this platform keep ``1.0``:
    those are game legs (moneylines, totals) with no per-market calibration
    concept, not no-evidence player cells — ``resolve_shrinkage``'s own
    no-evidence default is ``NO_EVIDENCE_SHRINKAGE`` (0.0).
    """
    from sportstradamus.strategies.underdog_pickem import resolve_market_shrinkage

    canonical = game_df["Market"].map(stat_map.get(platform, {}))
    out = np.ones(len(game_df), dtype=float)
    for i, market in enumerate(canonical):
        if not isinstance(market, str):
            continue
        key = (league, market)
        if key not in shrinkage_cache:
            shrinkage_cache[key] = resolve_market_shrinkage(league, market)[0]
        out[i] = shrinkage_cache[key]
    return out


def _build_correlation_matrices(
    game_df,
    game_dict,
    c_map,
    team_mod_map,
    opp_mod_map,
    search_payouts,
    platform,
    league,
    shrinkage_cache,
):
    """Per-game leg×leg correlation (C) / boost (M) matrices and EV grids.

    Returns a :class:`GameArrays` bundling the symmetric leg matrices ``C``/``M``,
    the model/book pairwise expected values ``EV``/``EVb``, the model std-dev
    outer product ``V``, and the per-leg probability / boost vectors beam search
    consumes.
    """
    C = np.eye(len(game_dict))
    M = np.zeros([len(game_dict), len(game_dict)])
    p_model = game_df["Win Prob"].to_numpy()
    p_books = game_df["Market Prob"].to_numpy()
    # ``Push P`` is added by :func:`model_prob` for integer-line discrete markets;
    # missing for combo legs etc. — fill 0.0 so the analytical mvn.cdf path runs.
    if "Push Prob" in game_df.columns:
        p_push = game_df["Push Prob"].fillna(0.0).to_numpy()
    else:
        p_push = np.zeros(len(game_df), dtype=float)
    boosts = game_df["Boost"].to_numpy()
    opp_boost = _leg_opp_boost(game_df, platform)
    shrinkage = _leg_shrinkage(game_df, platform, league, shrinkage_cache)
    V = p_model * (1 - p_model)
    V = V.reshape(len(game_dict), 1) * V
    V = np.sqrt(V)
    P = p_model.reshape(len(p_model), 1) * p_model
    Vb = p_books * (1 - p_books)
    Vb = Vb.reshape(len(game_dict), 1) * Vb
    Vb = np.sqrt(Vb)
    Pb = p_books.reshape(len(p_books), 1) * p_books
    for i, j in combinations(range(len(game_dict)), 2):
        rho, boost = _leg_pair_corr_boost(
            game_dict[i], game_dict[j], c_map, team_mod_map, opp_mod_map
        )
        C[i, j] = C[j, i] = rho
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
    return GameArrays(
        C=C,
        M=M,
        EV=EV,
        EVb=EVb,
        V=V,
        p_model=p_model,
        p_books=p_books,
        p_push=p_push,
        boosts=boosts,
        shrinkage=shrinkage,
        opp_boost=opp_boost,
    )


def _resolve_player_positions(league_df, league, stat_data):
    """Map each leg's numeric depth-chart slot to a position+rank label.

    Non-MLB: drop combo / ``vs.`` legs, rank players within (team, position) by
    the league's usage profile, and suffix the rank (``G1``/``G2``/...). MLB:
    batting order to ``B{n}`` (or ``P`` for pitchers). Returns the updated
    ``league_df`` (the non-MLB branch drops rows, so the caller must rebind).
    """
    if league != "MLB":
        league_df["Player position"] = league_df["Player position"].apply(
            lambda x: (
                _LEAGUE_POSITIONS[league][x - 1]
                if isinstance(x, int)
                else [_LEAGUE_POSITIONS[league][i - 1] for i in x]
            )
        )
        combo_df = league_df.loc[league_df.Player.str.contains(r"\+|vs.")]
        league_df = league_df.loc[~league_df.index.isin(combo_df.index)]
        player_df = league_df[["Player", "Team", "Player position"]]
        player_df.drop_duplicates(inplace=True)
        stat_data.profile_market(_LEAGUE_USAGE_STAT[league])
        usage = pd.DataFrame(
            stat_data.playerProfile[
                [_LEAGUE_USAGE_STAT[league] + " short", _LEAGUE_USAGE_TIEBREAKER[league]]
            ]
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
            player_df.sort_values(_LEAGUE_USAGE_TIEBREAKER[league], ascending=False)
            .groupby(["Team", "Player position"])
            .rank(ascending=False, method="first")[_LEAGUE_USAGE_STAT[league] + " short"]
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
    return league_df


def _build_cmarket(league_df, league, new_map):
    """Add the ``cMarket`` (position.corr-name) display column.

    Mutates the shared ``new_map`` with the league's correlation-name overrides
    (accumulating across leagues, matching the legacy single-dict behavior), then
    builds one ``position.market`` token per leg — a list, for multi-position
    combo legs.
    """
    if league == "NHL":
        new_map.update({"Points": "points", "Blocked Shots": "blocked", "Assists": "assists"})
    if league in ("NBA", "WNBA"):
        new_map.update({"Fantasy Points": "fantasy points prizepicks"})

    league_df["cMarket"] = league_df.apply(
        lambda x: (
            [x["Player position"] + "." + resolve_leg_stat(x["Market"], new_map)]
            if isinstance(x["Player position"], str)
            else [p + "." + resolve_leg_stat(x["Market"], new_map) for p in x["Player position"]]
        ),
        axis=1,
    )
    return league_df


def _assemble_game_frame(league_df, team):
    """Concatenate a team's legs with its opponent's and any split (combo) legs.

    Opponent and split legs get their ``cMarket`` tokens ``_OPP_``-prefixed so
    the correlation lookup can tell the two sides of the matchup apart. Returns
    ``(game_df, opp, date)``.
    """
    team_df = league_df.loc[league_df["Team"] == team]
    opp = team_df.Opponent.mode().values[0]
    date = team_df.Date.mode().values[0]
    opp_df = league_df.loc[league_df["Team"] == opp]
    if not opp_df.empty:
        opp_df["cMarket"] = opp_df.apply(lambda x: ["_OPP_" + c for c in x["cMarket"]], axis=1)
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
    game_df.drop_duplicates(subset=["Player", "Market", "Bet", "Line"], inplace=True)
    return game_df, opp, date


def _select_bet_offers(game_df):
    """Pre-filter and cap the legs that enter the beam search.

    Keep only book-supported, model-favored legs, dedup, then bound the candidate
    set per player / team / game so the cartesian parlay enumeration stays
    tractable.
    """
    idx = game_df.loc[
        (game_df["Market EV"] > _OFFER_BOOKS_EV_FLOOR)
        & (game_df["Market Prob"] >= _OFFER_BOOKS_PROB_FLOOR)
        & (game_df["Model EV"] > _OFFER_MODEL_EV_FLOOR)
        & (game_df["Win Prob"] >= _OFFER_MODEL_PROB_FLOOR)
    ].sort_values("Kelly", ascending=False)
    idx = idx.drop_duplicates(subset=["Player", "Team", "Market", "Line"])
    idx = idx.groupby("Player").head(_MAX_OFFERS_PER_PLAYER)
    idx = (
        idx.sort_values(["Model EV", "Market EV"], ascending=False)
        .groupby("Team")
        .head(_MAX_OFFERS_PER_TEAM)
        .sort_values(["Team", "Player"])
    )
    return (
        idx.sort_values(["Model EV", "Market EV"], ascending=False)
        .head(_MAX_OFFERS_PER_GAME)
        .sort_values(["Team", "Player"])
    )


def _corr_partners(legs: pd.DataFrame) -> list[dict]:
    """One structured record per correlated partner: player/market/bet/line/mult."""
    return [
        {
            "player": r["Player"],
            "market": r["Market"],
            "bet": r["Bet"],
            "line": float(r["Line"]),
            "mult": round(float(r["Corr Mult"]), 2),
        }
        for _, r in legs.iterrows()
    ]


def _annotate_correlation_columns(df, game_df, g):
    """Fill each offer's ``Corr Same`` / ``Corr Opp`` correlated-partner columns in ``df``.

    For every leg, surface its top correlated same-team and opponent partners
    (one per player) that clear the display EV / correlation gates. Each partner
    is a structured record (player/market/bet/line/mult), not a display string —
    ``leg_schema.leg_label`` renders it on demand.
    """
    EV, EVb, C, V = g.EV, g.EVb, g.C, g.V
    for i, offer in game_df.iterrows():
        indices = (
            (EV[:, i] > _DISPLAY_MODEL_EV_FLOOR)
            & (C[:, i] > _DISPLAY_CORR_FLOOR)
            & (EVb[:, i] > _DISPLAY_BOOKS_EV_FLOOR)
        )
        corr = game_df.loc[indices].copy()
        corr["Corr Mult"] = np.exp(C[indices, i] * V[indices, i])
        corr = corr.sort_values("Corr Mult", ascending=False).groupby("Player").head(1)
        same = corr.loc[corr["Team"] == offer["Team"]]
        other = corr.loc[corr["Team"] != offer["Team"]]
        # A (Player, Market) pair can span more than one row (Sleeper alt lines).
        # Broadcasting via a same-length Series (not a bare list, which numpy casts
        # to a 2D array and mis-shapes on an empty partner list) mirrors the old
        # scalar-string assignment's broadcast-to-every-matched-row semantics.
        mask = (df["Player"] == offer["Player"]) & (df["Market"] == offer["Market"])
        same_partners = _corr_partners(same)
        other_partners = _corr_partners(other)
        df.loc[mask, "Corr Same"] = pd.Series([same_partners] * mask.sum(), index=df.index[mask])
        df.loc[mask, "Corr Opp"] = pd.Series([other_partners] * mask.sum(), index=df.index[mask])


def _collect_game_corr(game_df, C, league, game_label, market_map):
    """Per-game upper-triangle correlation slice for the dashboard rail/constellation.

    Emits one row per distinct leg pair, keyed by ``Player|Market|Bet`` with the
    canonical ``Market`` code (``market_map`` mirrors the post-``process_offers``
    remap ``cli`` applies) so the slice joins ``current_offers.parquet``. ``C`` is
    positionally aligned to ``game_df`` (the caller reset its index). Correlation
    is line-independent, so same-key pairs at different lines collapse on the
    caller's dedup.
    """
    players = game_df["Player"].to_numpy()
    markets = game_df["Market"].to_numpy()
    bets = game_df["Bet"].to_numpy()
    rows = []
    for i, j in combinations(range(len(game_df)), 2):
        leg_i = f"{players[i]}|{market_map.get(markets[i], markets[i])}|{bets[i]}"
        leg_j = f"{players[j]}|{market_map.get(markets[j], markets[j])}|{bets[j]}"
        if leg_i == leg_j:
            continue
        leg_a, leg_b = sorted((leg_i, leg_j))
        rows.append(
            {
                "League": league,
                "Game": game_label,
                "leg_a": leg_a,
                "leg_b": leg_b,
                "rho": float(C[i, j]),
            }
        )
    return rows


def _append_parlay_rows(parlay_df, best_bets):
    """Dedup beam-search bets across the three sort views and append.

    Keeps the top ``_PARLAY_TOP_N_PER_SORT`` by Model EV / Rec Bet / Fun, dedups
    the union on parlay identity (``Bet ID``), and concatenates onto ``parlay_df``.
    """
    bets = pd.DataFrame(best_bets)
    df5 = (
        pd.concat(
            [
                bets.sort_values("Model EV", ascending=False).head(_PARLAY_TOP_N_PER_SORT),
                bets.sort_values("Rec Bet", ascending=False).head(_PARLAY_TOP_N_PER_SORT),
                bets.sort_values("Fun", ascending=False).head(_PARLAY_TOP_N_PER_SORT),
            ]
        )
        # subset=["Bet ID"]: a parlay's identity is its leg-index tuple. Full-row
        # dedup no longer works once ``legs`` (list[dict]) rides along — unhashable.
        .drop_duplicates(subset=["Bet ID"])
        .sort_values("Model EV", ascending=False)
    )
    return pd.concat([parlay_df, df5.drop(columns="Bet ID")], ignore_index=True)


def _append_story_context(
    story_sink, platform, league, game, date, g, bet_df, idx, search_payouts, full_payouts
):
    """Capture one game's scoring bundle for the story-menu generator.

    No-op unless ``story_sink`` is provided (the pickem variant-sweep caller and
    beam-search-only runs leave it None) and the game has bet-eligible legs. The
    bundle lets the generator price story subsets beam search never enumerates.
    """
    if story_sink is None or idx.empty:
        return
    story_sink.append(
        GameScoringContext(
            platform=platform,
            league=league,
            game=game,
            date=str(date),
            g=g,
            bet_df=bet_df,
            leg_indices=tuple(sorted(idx.index.to_numpy())),
            full_payouts=full_payouts,
            payout_base_by_size={
                s: search_payouts[s - 2] for s in range(2, len(search_payouts) + 2)
            },
            max_size=max(full_payouts),
        )
    )


@line_profiler.profile
def _process_league_games(
    df,
    league_df,
    league,
    platform,
    c_same,
    c_opp,
    team_mod_map,
    opp_mod_map,
    search_payouts,
    full_payouts,
    parlay_df,
    contest_variant,
    corr_sink,
    story_sink,
    market_map,
    stat_map,
):
    """Annotate correlations and beam-search parlays for every game in a league.

    For each (team, opponent) pairing: assemble the game frame, build the
    correlation / boost matrices, annotate ``df``'s display columns, collect the
    per-game correlation slice into ``corr_sink`` (when not None), and — unless
    the platform/league is parlay-ineligible — beam-search parlays onto
    ``parlay_df`` (returned).
    """
    checked_teams = []
    # Shared across every game this league processes below — resolve_market_shrinkage
    # hits training/CLV I/O per (league, market), and markets repeat heavily
    # across a slate, so caching here (not per-game) keeps it out of the hot path.
    shrinkage_cache: dict[tuple[str, str], float] = {}
    teams = [team for team in league_df.Team.unique() if "/" not in team]
    for team in tqdm(teams, desc=f"Checking {league} games", unit="game"):
        if team in checked_teams:
            continue
        game_df, opp, date = _assemble_game_frame(league_df, team)
        checked_teams.append(team)
        checked_teams.append(opp)

        c_map = _build_game_corr_map(team, opp, c_same, c_opp)
        game_df.reset_index(drop=True, inplace=True)
        game_dict = game_df.to_dict("index")
        idx = _select_bet_offers(game_df)
        bet_df = idx.to_dict("index")

        g = _build_correlation_matrices(
            game_df,
            game_dict,
            c_map,
            team_mod_map,
            opp_mod_map,
            search_payouts,
            platform,
            league,
            shrinkage_cache,
        )
        _annotate_correlation_columns(df, game_df, g)

        if corr_sink is not None:
            corr_sink.extend(
                _collect_game_corr(game_df, g.C, league, "/".join(sorted([team, opp])), market_map)
            )

        if platform in ["Chalkboard", "ParlayPlay"] and league == "MLB":
            continue
        info = {
            "Game": "/".join(sorted([team, opp])),
            "Date": date,
            "League": league,
            "Platform": platform,
        }
        max_boost = _MAX_BOOST_UNDERDOG if platform == "Underdog" else _MAX_BOOST_OTHER
        full_refund_below_size = SLEEPER_FULL_REFUND_MAX_SIZE if platform == "Sleeper" else None
        _append_story_context(
            story_sink,
            platform,
            league,
            info["Game"],
            date,
            g,
            bet_df,
            idx,
            search_payouts,
            full_payouts,
        )
        best_bets = beam_search_parlays(
            idx,
            g,
            search_payouts,
            full_payouts,
            max_boost,
            bet_df,
            info,
            team,
            opp,
            stat_map,
            contest_variant=contest_variant,
            full_refund_below_size=full_refund_below_size,
        )
        if best_bets:
            parlay_df = _append_parlay_rows(parlay_df, best_bets)
    return parlay_df


@line_profiler.profile
def find_correlation(
    offers,
    stats,
    platform,
    *,
    contest_variant: Literal["pooled", "power", "flex", "insurance", "rivals"] = "pooled",
    corr_sink: list | None = None,
    story_sink: list | None = None,
):
    """Annotate offers with correlation info and build parlay candidates.

    Groups scored offers by game, loads the league correlation parquets, computes
    pairwise boost modifiers, fills ``Corr Same`` / ``Corr Opp`` columns, and
    calls :func:`beam_search_parlays` for each game.

    Args:
        offers: List of scored offer dicts from :func:`process_offers`.
        stats: ``{league: Stats}`` dict for active leagues.
        platform: DFS platform name (e.g. ``"Underdog"``).
        contest_variant: Underdog payout pool. Default ``"pooled"`` combines
            ``power`` (sizes 2-3) and ``flex`` (sizes 4+) into one pool;
            the single-variant names are accepted for the ``pickem-build``
            path. Ignored for non-Underdog platforms.
        corr_sink: When provided, each game appends its upper-triangle
            correlation slice (``League, Game, leg_a, leg_b, rho``) to this list
            for the dashboard rail/constellation. The pickem variant-sweep caller
            leaves it None.
        story_sink: When provided, each game appends a
            :class:`~sportstradamus.prediction.parlay.GameScoringContext` to this
            list so the story-menu generator can price story subsets. The pickem
            variant-sweep caller leaves it None.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: ``(offer_df, parlay_df)`` where
            ``offer_df`` is the full scored slate sorted by ``Model EV`` and
            ``parlay_df`` has beam-search parlay candidates.
    """
    logger.info("Finding Correlations")

    new_map = stat_map[platform].copy()
    warnings.simplefilter("ignore")

    df = pd.DataFrame(offers)
    versus_mask = df["Player"].str.contains(" vs. ")
    if versus_mask.any():
        df.loc[versus_mask, "Team"] = df.loc[versus_mask].apply(
            lambda x: x["Team"].split("/")[0 if x["Bet"] == "Over" else 1], axis=1
        )
        df.loc[versus_mask, "Opponent"] = df.loc[versus_mask].apply(
            lambda x: x["Opponent"].split("/")[0 if x["Bet"] == "Over" else 1], axis=1
        )

    combo_mask = df["Team"].apply(lambda x: len(set(x.split("/"))) == 1)
    df.loc[combo_mask, "Team"] = df.loc[combo_mask, "Team"].apply(lambda x: x.split("/")[0])
    df.loc[combo_mask, "Opponent"] = df.loc[combo_mask, "Opponent"].apply(lambda x: x.split("/")[0])

    df["Corr Same"] = None
    df["Corr Opp"] = None
    # Depth-chart labels resolved per league below; combo legs keep "".
    df["Position"] = ""
    # Canonical matchup key shared by offers, parlays, and the corr slice so the
    # dashboard joins them. Distinct from the "Team vs Opp · Date" display label.
    df["Game"] = [
        "/".join(sorted([t, o])) for t, o in zip(df["Team"], df["Opponent"], strict=False)
    ]
    parlay_df = pd.DataFrame(
        columns=[
            "Game",
            "Date",
            "League",
            "Platform",
            "Model EV",
            "Market EV",
            "Boost",
            "Rec Bet",
            "legs",
            "Bet ID",
            "P",
            "PB",
            "Fun",
            "Bet Size",
            "Corr Pairs",
            "Boost Pairs",
            "Indep P",
            "Indep PB",
        ]
    )
    # ``search_payouts`` is the single-multiplier-per-size list used inside the
    # beam-search ranking; ``full_payouts`` is the per-(size, miss-count) lookup
    # driving push-aware EV and the display Boost column.
    search_payouts, full_payouts = payout_curve_for(platform, contest_variant)

    for league in ["NFL", "NBA", "WNBA", "MLB", "NHL"]:
        league_df = df.loc[df["League"] == league]
        if league_df.empty:
            continue
        league_dir = pkg_resources.files(data) / "leagues" / league.lower()
        c_same = pd.read_parquet(league_dir / "corr_same_team.parquet")
        c_same.rename_axis(["team", "market", "correlation"], inplace=True)
        c_same.columns = ["R"]
        c_opp = pd.read_parquet(league_dir / "corr_opposing.parquet")
        c_opp.rename_axis(["team", "market", "correlation"], inplace=True)
        c_opp.columns = ["R"]
        stat_data = stats.get(league)
        team_mod_map = banned[platform][league]["team"]
        opp_mod_map = banned[platform][league]["opponent"]
        if platform == "Underdog":
            league_df["Boost"] = league_df["Boost"] / UNDERDOG_BOOST_BASELINE

        league_df = _resolve_player_positions(league_df, league, stat_data)
        # _resolve_player_positions returns a slice the function otherwise drops;
        # persist the depth-chart labels onto the returned frame (combos absent
        # from its index keep "") so build_game_context can read Position.
        df.loc[league_df.index, "Position"] = league_df["Player position"]
        league_df = _build_cmarket(league_df, league, new_map)
        parlay_df = _process_league_games(
            df,
            league_df,
            league,
            platform,
            c_same,
            c_opp,
            team_mod_map,
            opp_mod_map,
            search_payouts,
            full_payouts,
            parlay_df,
            contest_variant,
            corr_sink,
            story_sink,
            stat_map[platform],
            stat_map,
        )

    return df.dropna(subset="Model EV").sort_values("Model EV", ascending=False), parlay_df
