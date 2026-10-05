"""Story-menu generator golden tests (P3a).

Exercises ``prediction.stories.menu.build_game_stories`` on hand-built
``GameScoringContext`` fixtures (synthetic ``GameArrays`` with explicit
correlation / probabilities, the same shape ``find_correlation`` hands the
generator). Pins the owner-locked acceptance criteria: ≤5 correlation-cluster
stories per game, no-signal games yield nothing, the two objectives are the true
argmaxes, the play-type cap holds, platforms are independent, the EV is the
real copula scorer (not a reimplementation), and each side's most prominent
strong leg leads a story that ranks ahead of the unled ones.
"""

from __future__ import annotations

import math
import re
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from sportstradamus.analysis import _leg_market_map
from sportstradamus.helpers import stat_map
from sportstradamus.prediction.joint import parlay_payout_prob, psd_or_none
from sportstradamus.prediction.parlay import GameArrays, GameScoringContext
from sportstradamus.prediction.payouts import payout_curve_for
from sportstradamus.prediction.stories import menu as menu_mod
from sportstradamus.prediction.stories.legs import validate_parlay_legs
from sportstradamus.prediction.stories.menu import (
    _MENU_EDGE_FLOOR,
    _story_prose,
    build_game_stories,
)
from sportstradamus.prediction.stories.pricing import _log_growth, score_subset


def _leg_keys(legs: list[dict]) -> frozenset:
    return frozenset((leg["player"], leg["bet"], leg["line"], leg["market"]) for leg in legs)


def _ctx(
    probs,
    corr,
    *,
    platform="Underdog",
    edges=None,
    teams=None,
    game="AAA/BBB",
    max_size=None,
    players=None,
    bets=None,
    kellys=None,
    stars=None,
    date="2026-06-12",
    commence="",
):
    """Build one synthetic GameScoringContext + its matching offers frame.

    Every leg is Kelly-positive unless ``kellys`` says otherwise; ``stars`` is
    the offers frame's ``Star`` column (all 0.0 by default, so a seed falls to
    the earlier player name).
    """
    n = len(probs)
    probs = np.asarray(probs, dtype=float)
    edges = [1.30] * n if edges is None else edges
    teams = teams or (["AAA"] * (n // 2) + ["BBB"] * (n - n // 2))
    players = players or [f"Player{i}" for i in range(n)]
    bets = bets or ["Over"] * n
    kellys = [0.25] * n if kellys is None else kellys
    stars = [0.0] * n if stars is None else stars
    names = ["Points", "Rebounds", "Assists", "Steals", "Blocks", "Turnovers", "FG3M", "FTM"]
    markets = [names[i % len(names)] for i in range(n)]
    g = GameArrays(
        C=np.asarray(corr, dtype=float),
        M=np.ones((n, n)) - np.eye(n),
        EV=np.zeros((n, n)),
        EVb=np.zeros((n, n)),
        V=np.zeros((n, n)),
        p_model=probs,
        p_books=probs.copy(),
        p_push=np.zeros(n),
        boosts=np.ones(n),
        shrinkage=np.ones(n),
        opp_boost=np.full(n, np.nan),
    )
    bet_df = {
        i: {
            "Model EV": edges[i],
            "Bet": bets[i],
            "Team": teams[i],
            "Player": players[i],
            "Market": markets[i],
            "Line": 5.5 + i,
            "Win Prob": probs[i],
            "Kelly": kellys[i],
            "Commence": commence,
        }
        for i in range(n)
    }
    search, full = payout_curve_for(platform, "pooled")
    if max_size is not None:
        full = {s: full[s] for s in full if s <= max_size}
    pay_base = {s: search[s - 2] for s in range(2, len(search) + 2) if s <= max(full)}
    sctx = GameScoringContext(
        platform=platform,
        league="NBA",
        game=game,
        date=date,
        g=g,
        bet_df=bet_df,
        leg_indices=tuple(range(n)),
        full_payouts=full,
        payout_base_by_size=pay_base,
        max_size=max(pay_base),
    )
    offers = pd.DataFrame(
        [
            {
                "Player": players[i],
                "Bet": bets[i],
                "Line": 5.5 + i,
                "Market": markets[i],
                "Game": game,
                "Team": teams[i],
                "Position": "",
                "Avg 5": 2.0,
                "DVPOA": 0.10,
                "Star": stars[i],
            }
            for i in range(n)
        ]
    )
    return sctx, offers


def _block_diag_corr(block_sizes, rho):
    """A correlation matrix of independent ρ-blocks (each block = one cluster)."""
    n = sum(block_sizes)
    corr = np.eye(n)
    start = 0
    for size in block_sizes:
        for i, j in combinations(range(start, start + size), 2):
            corr[i, j] = corr[j, i] = rho
        start += size
    return corr


def test_caps_at_five_stories():
    corr = _block_diag_corr([2] * 6, 0.4)  # six separable two-leg clusters
    # Interleave teams so every 2-leg block spans both sides — a single-team block
    # is not a valid parlay and the menu would (correctly) drop it.
    sctx, offers = _ctx([0.62] * 12, corr, teams=["AAA", "BBB"] * 6)
    out = build_game_stories([sctx], offers, pd.DataFrame(), None)
    assert out["story_id"].nunique() == 5
    assert set(out["objective"]) == {"builder", "moon"}
    assert len(out) == 10  # five stories × two objectives


def test_no_signal_games_yield_zero():
    # All legs below the edge floor → no strong legs → no stories.
    weak = _ctx([0.62, 0.61, 0.60], np.eye(3), edges=[1.0, 1.0, 1.0])
    assert build_game_stories([weak[0]], weak[1], pd.DataFrame(), None).empty
    # Strong legs but no correlated pair (identity ρ) → only singletons → nothing.
    # The lead's orphan fallback needs another player, and these are one player's
    # markets, so even the lead stays alone.
    uncorrelated = _ctx([0.66, 0.65, 0.64], np.eye(3), teams=["AAA"] * 3, players=["A"] * 3)
    assert build_game_stories([uncorrelated[0]], uncorrelated[1], pd.DataFrame(), None).empty


def test_edge_floor_excludes_weak_leg():
    # Legs 0,1 strong and correlated; leg 2 strong+correlated; leg 3 BELOW floor.
    corr = np.eye(4)
    corr[0, 1] = corr[1, 0] = 0.4
    corr[0, 2] = corr[2, 0] = 0.3
    corr[0, 3] = corr[3, 0] = 0.4
    sctx, offers = _ctx([0.66, 0.65, 0.64, 0.66], corr, edges=[1.30, 1.30, 1.30, 1.00])
    out = build_game_stories([sctx], offers, pd.DataFrame(), None)
    legs = [leg for row in out["legs"] for leg in row]
    assert not any(leg["player"] == "Player3" for leg in legs)  # sub-floor leg never appears


def test_kelly_zero_leg_is_never_strong():
    # A sub-1x or deep-alt payout zeroes Kelly, so the leg can't seed or join a story
    # even at twice the edge floor.
    sctx, _offers = _ctx([0.70, 0.66], np.eye(2), edges=[1.10, 1.30], kellys=[0.0, 0.25])
    assert menu_mod._strong_legs(sctx) == {1: 1.30}


def test_correlation_cluster_forms_over_exact_legs():
    corr = np.eye(2)
    corr[0, 1] = corr[1, 0] = 0.45
    sctx, offers = _ctx([0.62, 0.60], corr)
    out = build_game_stories([sctx], offers, pd.DataFrame(), None)
    assert out["story_id"].nunique() == 1
    for row in out["legs"]:
        assert {leg["player"] for leg in row} == {"Player0", "Player1"}


def test_cluster_holds_one_leg_per_player():
    """One hot player's four correlated markets must not swallow the cluster.

    Same-player legs correlate near 1, so without the rule the cap fills with
    one name and ``validate_parlay_legs`` (distinct players) has no valid
    subset left to pick.
    """
    sctx, _offers = _ctx(
        [0.66] * 7,
        _block_diag_corr([7], 0.4),
        players=["Star"] * 4 + ["Player4", "Player5", "Player6"],
    )
    edge = menu_mod._strong_legs(sctx)
    order = sorted(edge, key=lambda i: -edge[i])
    clusters = menu_mod._cluster_strong_legs(order, sctx, {})
    assert clusters == [([0, 4, 5, 6], "", "")]  # Star's strongest leg, then three teammates
    assert len({sctx.bet_df[i]["Player"] for i in clusters[0][0]}) == 4


def test_each_side_leads_a_story_and_led_stories_rank_first():
    # Over block {0,1,2} seeds on the star Player0 and Under block {3,4,5} on the
    # brighter Player4; the unled pair {6,7} is the richest cluster in the game.
    corr = _block_diag_corr([3, 3, 2], 0.3)
    sctx, offers = _ctx(
        [0.66, 0.65, 0.64, 0.70, 0.69, 0.68, 0.95, 0.95],
        corr,
        teams=["AAA", "BBB"] * 4,
        bets=["Over"] * 3 + ["Under"] * 3 + ["Over"] * 2,
        stars=[2.0, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0],
    )
    out = build_game_stories([sctx], offers, pd.DataFrame(), None)
    moon = out[out["objective"] == "moon"].set_index("story_id")
    assert moon.index.tolist() == ["AAA/BBB#0", "AAA/BBB#1", "AAA/BBB#2"]
    # Led stories first, ordered between themselves by Moon EV; the unled story
    # out-earns both yet ranks last, so EV alone no longer orders the menu.
    assert moon["lead_side"].tolist() == ["Under", "Over", ""]
    assert moon["lead_player"].tolist() == ["Player4", "Player0", ""]
    assert moon["star"].tolist() == [2.5, 2.0, 0.0]
    assert moon["model_ev"].iloc[2] > moon["model_ev"].iloc[0] > moon["model_ev"].iloc[1]
    # The lead sits in both presets of its story, on the side it leads.
    for _sid, grp in out[out["lead_side"] != ""].groupby("story_id"):
        side, player = grp["lead_side"].iloc[0], grp["lead_player"].iloc[0]
        for legs in grp["legs"]:
            assert (player, side) in {(leg["player"], leg["bet"]) for leg in legs}
    # The slate pass flags one led story on both of its rows.
    flagged = out[out["lead"]]
    assert flagged["story_id"].nunique() == 1
    assert len(flagged) == 2
    assert flagged["lead_side"].iloc[0] != ""


def test_led_story_headline_is_routed_to_its_lead(monkeypatch):
    leads = []

    def stub(_legs, _ctxs, lead=None):
        leads.append(lead)
        return ["Story"], 0, {}

    monkeypatch.setattr(menu_mod, "thesis_variants", stub)
    sctx, offers = _ctx(
        [0.66, 0.65, 0.70, 0.69],
        _block_diag_corr([2, 2], 0.4),
        teams=["AAA", "BBB"] * 2,
        bets=["Over", "Over", "Under", "Under"],
    )
    out = build_game_stories([sctx], offers, pd.DataFrame(), None)
    assert out["lead_side"].tolist() == ["Under", "Under", "Over", "Over"]
    assert leads == ["Player2", "Player0"]  # rank order: the richer Under-led story first


def test_orphan_lead_pairs_with_the_top_edge_leg_from_the_other_team():
    # The star Over (Player0, AAA) correlates with nothing, so its led cluster is a
    # singleton that prices nothing and borrows a leg. Player3 (AAA) has the top
    # edge but shares the lead's team; Player2 is the top-edge BBB leg.
    probs, edges, stars = [0.66, 0.62, 0.64, 0.65], [1.30, 1.20, 1.40, 1.50], [3.0, 0, 0, 0]
    sctx, offers = _ctx(
        probs, np.eye(4), teams=["AAA", "BBB", "BBB", "AAA"], edges=edges, stars=stars
    )
    out = build_game_stories([sctx], offers, pd.DataFrame(), None)
    assert out["story_id"].nunique() == 1
    assert out["lead_player"].unique().tolist() == ["Player0"]
    for legs in out["legs"]:
        assert {leg["player"] for leg in legs} == {"Player0", "Player2"}
    # One-sided game: any other player qualifies, so the top edge wins outright.
    sctx, offers = _ctx(probs, np.eye(4), teams=["AAA"] * 4, edges=edges, stars=stars)
    out = build_game_stories([sctx], offers, pd.DataFrame(), None)
    assert out["story_id"].nunique() == 1
    for legs in out["legs"]:
        assert {leg["player"] for leg in legs} == {"Player0", "Player3"}


def test_under_lead_survives_a_fully_correlated_pool():
    # Every pair correlates, so the Over-led cluster (the star Player0) absorbs the
    # whole pool and the Under lead (Player5, BBB) is left alone. It still gets its
    # story: it borrows the top-edge AAA leg — Player2 at 1.35, since Player1 and
    # Player3 out-edge it but share Player5's team — although the Over cluster
    # already claimed that leg.
    sctx, offers = _ctx(
        [0.66] * 6,
        _block_diag_corr([6], 0.3),
        teams=["AAA", "BBB"] * 3,
        bets=["Over"] * 5 + ["Under"],
        edges=[1.30, 1.45, 1.35, 1.40, 1.30, 1.30],
        stars=[2.0, 0, 0, 0, 0, 0],
    )
    edge = menu_mod._strong_legs(sctx)
    order = sorted(edge, key=lambda i: -edge[i])
    assert menu_mod._cluster_strong_legs(order, sctx, {"Over": 0, "Under": 5}) == [
        ([0, 1, 2, 3, 4], "Over", "Player0"),
        ([5], "Under", "Player5"),
    ]
    out = build_game_stories([sctx], offers, pd.DataFrame(), None)
    stories = out.drop_duplicates("story_id").set_index("lead_side")
    assert stories["lead_player"].to_dict() == {"Over": "Player0", "Under": "Player5"}
    for legs in out[out["lead_side"] == "Under"]["legs"]:
        assert {(leg["player"], leg["bet"]) for leg in legs} == {
            ("Player5", "Under"),
            ("Player2", "Over"),
        }
    for legs in out[out["lead_side"] == "Over"]["legs"]:
        assert ("Player0", "Over") in {(leg["player"], leg["bet"]) for leg in legs}


def test_same_team_led_cluster_borrows_an_other_team_leg():
    # Two-team game: the Under lead (Player2, BBB) correlates only with its own
    # team's Unders, so its cluster prices nothing under the both-teams rule. It
    # borrows the top-edge AAA leg (Player0) and the story spans both teams.
    sctx, offers = _ctx(
        [0.66] * 5,
        _block_diag_corr([2, 3], 0.4),
        teams=["AAA", "BBB", "BBB", "BBB", "BBB"],
        bets=["Over", "Over", "Under", "Under", "Under"],
    )
    out = build_game_stories([sctx], offers, pd.DataFrame(), None)
    assert set(out["lead_side"]) == {"Over", "Under"}
    under = out[out["lead_side"] == "Under"]
    assert under["story_id"].nunique() == 1
    assert under["lead_player"].iloc[0] == "Player2"
    team_of = dict(zip(offers["Player"], offers["Team"], strict=True))
    for legs in under["legs"]:
        pairs = {(leg["player"], leg["bet"]) for leg in legs}
        assert {("Player2", "Under"), ("Player0", "Over")} <= pairs
        assert {team_of[p] for p in _row_players(legs)} == {"AAA", "BBB"}


def test_one_lead_per_menu_alternates_sides_in_tip_order():
    # 2026-10-03 is an even ordinal, so Over is wanted first. ZZZ/YYY tips before
    # AAA/BBB although it sorts after it and is handed over second.
    corr = _block_diag_corr([2, 2], 0.4)

    def game(name, commence):
        return _ctx(
            [0.66, 0.65, 0.66, 0.65],
            corr,
            game=name,
            teams=[name[:3], name[-3:]] * 2,
            players=[f"{name}{i}" for i in range(4)],
            bets=["Over", "Over", "Under", "Under"],
            date="2026-10-03",
            commence=commence,
        )

    early, offers_early = game("ZZZ/YYY", "2026-10-03T17:00:00Z")
    late, offers_late = game("AAA/BBB", "2026-10-03T20:00:00Z")
    offers = pd.concat([offers_early, offers_late], ignore_index=True)
    out = build_game_stories([late, early], offers, pd.DataFrame(), None)
    assert out["lead"].dtype == bool
    menus = out[out["lead"]].groupby(["platform", "Date", "Game"])
    assert menus["story_id"].nunique().eq(1).all()
    assert menus["objective"].nunique().eq(2).all()  # both rows of the one story
    sides = out[out["lead"]].drop_duplicates("story_id").set_index("Game")["lead_side"]
    assert sides.to_dict() == {"ZZZ/YYY": "Over", "AAA/BBB": "Under"}


def test_commence_by_game_takes_the_first_non_empty_commence(monkeypatch):
    captured = {}
    real = menu_mod.assign_leads

    def spy(frame, commence_by_game):
        captured.update(commence_by_game)
        return real(frame, commence_by_game)

    monkeypatch.setattr(menu_mod, "assign_leads", spy)
    corr = _block_diag_corr([2], 0.4)
    # Sleeper rows carry no tip and come first; the Underdog context fills the key.
    sleeper, offers = _ctx([0.66, 0.65], corr, platform="Sleeper")
    underdog, _ = _ctx([0.66, 0.65], corr, commence="2026-06-12T23:00:00Z")
    untimed, _ = _ctx([0.66, 0.65], corr, game="CCC/DDD", teams=["CCC", "DDD"])
    build_game_stories([sleeper, underdog, untimed], offers, pd.DataFrame(), None)
    assert captured == {
        ("2026-06-12", "AAA/BBB"): "2026-06-12T23:00:00Z",
        ("2026-06-12", "CCC/DDD"): "",
    }


def test_objectives_are_true_argmaxes_and_share_legs():
    # Power-only cluster (sizes 2-3 ⇒ analytical mvn.cdf ⇒ deterministic; the flex
    # 4+ path is a 50k-sample Monte-Carlo whose EV jitters between scorings). A
    # strong anchor plus two thin legs makes the objectives diverge: the tight
    # high-prob pair compounds (Builder) while the wider set shoots the moon.
    corr = _block_diag_corr([3], 0.1)
    sctx, offers = _ctx([0.90, 0.55, 0.54], corr)
    out = build_game_stories([sctx], offers, pd.DataFrame(), None)
    builder = out[out["objective"] == "builder"].iloc[0]
    moon = out[out["objective"] == "moon"].iloc[0]

    # scipy's mvn.cdf is randomized QMC for dim ≥ 3, so EV carries ~1e-4 noise
    # between scorings; compare the chosen leg-SET (stable when the gap ≫ noise),
    # not exact EV. The 2-leg vs 3-leg EV gap here is ~0.24, far above the noise.
    # The menu argmaxes over *valid* subsets carrying the story's lead, so the
    # oracle must too — otherwise a single-team or lead-less subset could pose
    # as the "expected" argmax.
    assert (builder["lead_side"], builder["lead_player"]) == ("Over", "Player0")
    new_map = _leg_market_map(sctx.league, sctx.platform, stat_map)
    all_scores = [
        score_subset(c, sctx, new_map)
        for size in range(2, min(len(sctx.leg_indices), sctx.max_size) + 1)
        for c in combinations(sctx.leg_indices, size)
        if validate_parlay_legs([sctx.bet_df[i] for i in c])[0]
        and builder["lead_player"] in {sctx.bet_df[i]["Player"] for i in c}
    ]
    builder_legs = _leg_keys(builder["legs"])
    moon_legs = _leg_keys(moon["legs"])
    assert builder["model_ev"] <= moon["model_ev"]  # both from one scoring pass: Moon EV ≥ Builder
    assert moon_legs == _leg_keys(
        max(all_scores, key=lambda s: s["model_ev"])["legs"]
    )  # global EV argmax
    assert builder_legs == _leg_keys(
        max(all_scores, key=lambda s: s["G"])["legs"]
    )  # global log-growth argmax
    # This fixture diverges: tight strong pair builds, wider set shoots the moon.
    assert builder["bet_size"] < moon["bet_size"]
    assert builder_legs <= moon_legs  # shared legs allowed


def test_log_growth_closed_form():
    # p=0.55, payout=3.0 ⇒ b=2, full-Kelly f*=(0.55·3−1)/2=0.325, G=p·ln(1+b·f)+(1−p)·ln(1−f).
    p, payout = 0.55, 3.0
    b = payout - 1.0
    f = (p * payout - 1.0) / b
    expected = p * math.log(1.0 + b * f) + (1.0 - p) * math.log(1.0 - f)
    assert _log_growth(p, payout) == pytest.approx(expected)
    assert _log_growth(0.30, 3.0) == 0.0  # no edge ⇒ no growth


def test_play_type_cap_honored():
    corr = _block_diag_corr([6], 0.25)
    # Every pick sits above 0.64, where the 6-pick Flex's all-hit proxy (25 p^6) outranks
    # the 3-pick Power's (6.5 p^3) and so reaches the exact scorer through the shortlist.
    probs = [0.71, 0.70, 0.69, 0.68, 0.67, 0.66]
    # Underdog pooled reaches the 6-leg flex.
    ud = build_game_stories([_ctx(probs, corr)[0]], _ctx(probs, corr)[1], pd.DataFrame(), None)
    assert ud["bet_size"].max() == 6
    # A 3-cap platform (truncated payout table) never emits a 4+ leg story.
    capped_sctx, capped_offers = _ctx(probs, corr, max_size=3)
    capped = build_game_stories([capped_sctx], capped_offers, pd.DataFrame(), None)
    assert capped["bet_size"].max() <= 3


def test_platforms_are_independent():
    corr = _block_diag_corr([3], 0.3)
    ud_sctx, offers = _ctx([0.72, 0.70, 0.68], corr, platform="Underdog")
    sl_sctx, _ = _ctx([0.72, 0.70, 0.68], corr, platform="Sleeper")
    out = build_game_stories([ud_sctx, sl_sctx], offers, pd.DataFrame(), None)
    # Underdog menu present; Sleeper's [1.0, 1.0] placeholder can't clear breakeven.
    assert (out["platform"] == "Underdog").any()
    assert not (out["platform"] == "Sleeper").any()


def test_model_ev_is_the_real_copula_scorer():
    corr = _block_diag_corr([2], 0.4)
    sctx, _ = _ctx([0.66, 0.64], corr)
    bet_id = (0, 1)
    new_map = _leg_market_map(sctx.league, sctx.platform, stat_map)
    scored = score_subset(bet_id, sctx, new_map)
    g = sctx.g
    arr = np.asarray(bet_id)
    boost = float(g.M[0, 1] * g.boosts[0] * g.boosts[1])
    payout = float(np.clip(boost * sctx.payout_base_by_size[2], 1.0, 100.0))
    sig = psd_or_none(g.C[np.ix_(bet_id, bet_id)])
    direct = float(
        parlay_payout_prob(
            g.p_model[arr],
            g.p_push[arr],
            sig,
            2,
            boost,
            payout,
            sctx.full_payouts,
            sctx.payout_base_by_size[2],
        )
    )
    assert scored["model_ev"] == direct


def test_flex_loss_tier_prices_the_largest_multipliers_times_the_pair_modifier():
    # One sure loss on a 4-pick Flex: the 1-loss tier pays on the three largest pick
    # multipliers, whichever pick lost, times the pair-modifier product.
    sctx, _ = _ctx([1.0, 1.0, 0.0, 1.0], np.eye(4))
    sctx.g.boosts[:] = [0.9, 1.1, 1.2, 0.8]  # the loser holds the largest multiplier
    sctx.g.M[0, 1] = sctx.g.M[1, 0] = 0.85
    new_map = _leg_market_map(sctx.league, sctx.platform, stat_map)

    scored = score_subset((0, 1, 2, 3), sctx, new_map)

    one_loss_tier = sctx.full_payouts[4][1]
    assert scored["model_ev"] == pytest.approx(one_loss_tier * 1.2 * 1.1 * 0.9 * 0.85, rel=1e-3)


def test_edge_floor_value_is_owner_locked():
    assert _MENU_EDGE_FLOOR == 0.05


def _row_players(legs: list[dict]) -> list[str]:
    return [leg["player"] for leg in legs]


def test_presets_are_valid_parlays():
    # Every emitted Builder/Moon must be a valid DFS entry — distinct players and
    # both teams — because the menu enumerates only valid subsets.
    corr = _block_diag_corr([3], 0.3)
    sctx, offers = _ctx([0.66, 0.65, 0.64], corr, teams=["AAA", "BBB", "AAA"])
    out = build_game_stories([sctx], offers, pd.DataFrame(), None)
    assert not out.empty
    team_of = dict(zip(offers["Player"], offers["Team"], strict=True))
    for legs in out["legs"]:
        players = _row_players(legs)
        assert len(set(players)) == len(players)  # no repeated player
        assert len({team_of[p] for p in players}) >= 2  # both teams covered


def test_single_team_game_emits_single_team_story():
    # When the model's edge legs are ALL on one team, a single-team preset IS allowed —
    # the user completes it with a cross-game satellite leg in the editor. (Whole-game gate.)
    corr = _block_diag_corr([3], 0.3)
    sctx, offers = _ctx([0.66, 0.65, 0.64], corr, teams=["AAA", "AAA", "AAA"])
    out = build_game_stories([sctx], offers, pd.DataFrame(), None)
    assert not out.empty
    team_of = dict(zip(offers["Player"], offers["Team"], strict=True))
    for legs in out["legs"]:
        players = _row_players(legs)
        assert len(set(players)) == len(players)  # distinct players still required
        assert {team_of[p] for p in players} == {
            "AAA"
        }  # single-team preset, the game being one-sided


def test_unled_single_team_cluster_in_two_team_game_emits_no_story():
    # The game's edge spans both teams but each correlation cluster is one-sided and
    # none bridges them. The Over-led AAA cluster borrows the top-edge BBB leg
    # (Player2); the unled BBB cluster has no such recourse and emits nothing.
    corr = _block_diag_corr([2, 2], 0.4)
    sctx, offers = _ctx([0.66, 0.65, 0.64, 0.63], corr, teams=["AAA", "AAA", "BBB", "BBB"])
    out = build_game_stories([sctx], offers, pd.DataFrame(), None)
    assert out["story_id"].nunique() == 1
    assert out["lead_side"].eq("Over").all()
    team_of = dict(zip(offers["Player"], offers["Team"], strict=True))
    for legs in out["legs"]:
        players = _row_players(legs)
        assert {"Player0", "Player2"} <= set(players)
        assert {team_of[p] for p in players} == {"AAA", "BBB"}


def test_builder_and_moon_share_headline_and_dek():
    """One story, one headline, one dek — the mode chip must never swap the prose."""
    corr = _block_diag_corr([3], 0.1)
    sctx, offers = _ctx([0.90, 0.55, 0.54], corr)
    out = build_game_stories([sctx], offers, pd.DataFrame(), None)
    builder = out[out["objective"] == "builder"].iloc[0]
    moon = out[out["objective"] == "moon"].iloc[0]
    assert _leg_keys(builder["legs"]) != _leg_keys(moon["legs"])  # divergent presets
    for _sid, grp in out.groupby("story_id"):
        assert grp["headline"].nunique() == 1
        assert grp["dek"].nunique() == 1


def test_headline_names_only_players_shared_by_both_presets():
    corr = _block_diag_corr([3], 0.1)
    sctx, offers = _ctx([0.90, 0.55, 0.54], corr)
    out = build_game_stories([sctx], offers, pd.DataFrame(), None)
    for _sid, grp in out.groupby("story_id"):
        shared = set.intersection(*(set(_row_players(legs)) for legs in grp["legs"]))
        named = set(re.findall(r"Player\d+", grp["headline"].iloc[0]))
        assert named <= shared


def test_empty_core_guard_blanks_foreign_subject(monkeypatch):
    """Disjoint presets: a headline naming a player the two leg-sets don't share is blanked."""
    monkeypatch.setattr(
        menu_mod,
        "thesis_variants",
        lambda _legs, _ctxs, lead=None: (["Ghost rules"], 0, {"p": "Ghost"}),
    )
    sctx, offers = _ctx([0.66, 0.64], np.eye(2))
    presets = ({"bet_id": (0,)}, {"bet_id": (1,)})
    headline, dek = _story_prose(*presets, sctx, offers, {}, {}, {}, set(), "")
    assert headline == ""
    assert dek == ""  # a sub-2-leg core carries no cluster clause and no anchor row facts
    # An unnamed (game-script) subject survives the same disjoint presets.
    monkeypatch.setattr(
        menu_mod,
        "thesis_variants",
        lambda _legs, _ctxs, lead=None: (["The game tilts over"], 0, {"g": "X/Y"}),
    )
    headline, _dek = _story_prose(*presets, sctx, offers, {}, {}, {}, set(), "")
    assert headline == "The game tilts over"


def test_within_game_headline_dedup(monkeypatch):
    """Two stories seeded onto the same variant: the second bumps to the next one."""
    monkeypatch.setattr(
        menu_mod,
        "thesis_variants",
        lambda _legs, _ctxs, lead=None: (["Same story", "Second story"], 0, {}),
    )
    corr = _block_diag_corr([2, 2], 0.4)
    sctx, offers = _ctx([0.66] * 4, corr, teams=["AAA", "BBB"] * 2)
    out = build_game_stories([sctx], offers, pd.DataFrame(), None)
    assert out["story_id"].nunique() == 2
    assert out.drop_duplicates("story_id")["headline"].tolist() == ["Same story", "Second story"]


def test_headlines_dedupe_across_a_slate(monkeypatch):
    """Two games on one (platform, Date) draw distinct variants; another date starts fresh."""
    monkeypatch.setattr(
        menu_mod,
        "thesis_variants",
        lambda _legs, _ctxs, lead=None: (["Same story", "Second story"], 0, {}),
    )
    corr = _block_diag_corr([2], 0.4)
    first, offers = _ctx([0.66, 0.65], corr)
    second, _ = _ctx([0.66, 0.65], corr, game="CCC/DDD", teams=["CCC", "DDD"], players=["C", "D"])
    tomorrow, _ = _ctx(
        [0.66, 0.65],
        corr,
        game="EEE/FFF",
        teams=["EEE", "FFF"],
        players=["E", "F"],
        date="2026-06-13",
    )
    out = build_game_stories([first, second, tomorrow], offers, pd.DataFrame(), None)
    headlines = out.drop_duplicates("story_id").set_index("Game")["headline"]
    assert headlines.to_dict() == {
        "AAA/BBB": "Same story",
        "CCC/DDD": "Second story",
        "EEE/FFF": "Same story",
    }


def test_dek_is_deterministic_and_reads_core_correlation():
    corr = np.eye(2)
    corr[0, 1] = corr[1, 0] = 0.45
    sctx, offers = _ctx([0.62, 0.60], corr)
    first = build_game_stories([sctx], offers, pd.DataFrame(), None)
    second = build_game_stories([sctx], offers, pd.DataFrame(), None)
    assert first["dek"].tolist() == second["dek"].tolist()
    dek = first["dek"].iloc[0]
    assert dek
    assert "0.45" in dek  # the 2-leg core's mean pairwise rho, slot-formatted :.2f
    # The anchor's form/matchup clauses render only if offer_index keeps the
    # "Avg 5"/"DVPOA" facts — Player0 (max p_model) is the dek's subject.
    assert "Player0" in dek
