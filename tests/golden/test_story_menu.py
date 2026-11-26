"""Story-menu generator golden tests (P3a).

Exercises ``prediction.stories.menu.build_game_stories`` on hand-built
``GameScoringContext`` fixtures (synthetic ``GameArrays`` with explicit
correlation / probabilities, the same shape ``find_correlation`` hands the
generator). Pins the owner-locked acceptance criteria: ≤5 correlation-cluster
stories per game, no-signal games yield nothing, the two presets are the best
fitted value at their sizes, the play-type cap holds, platforms are independent,
the EV is the real copula scorer (not a reimplementation), and each side's most
prominent leg leads a story that ranks ahead of the unled ones. An Over leads
only in a shootout or a blowout. A game's story legs are its pool legs, ordered
by ``EV``, and no value floor drops a leg or a story.
"""

from __future__ import annotations

import dataclasses
import re
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from sportstradamus.prediction.joint import parlay_payout_prob, psd_or_none
from sportstradamus.prediction.parlay import GameArrays, GameScoringContext, leg_market_map
from sportstradamus.prediction.payouts import POWER_MAX_SIZE, payout_curve_for
from sportstradamus.prediction.stories import menu as menu_mod
from sportstradamus.prediction.stories.bank import why_bank
from sportstradamus.prediction.stories.legs import validate_parlay_legs
from sportstradamus.prediction.stories.menu import _story_prose, build_game_stories
from sportstradamus.prediction.stories.pricing import score_subset, shortlist


def _leg_keys(legs: list[dict]) -> frozenset:
    return frozenset((leg["player"], leg["bet"], leg["line"], leg["market"]) for leg in legs)


def _bet_id(legs: list[dict]) -> tuple[int, ...]:
    """A row's legs back to their fixture positions (``_ctx`` names leg i ``Player{i}``)."""
    return tuple(int(leg["player"].removeprefix("Player")) for leg in legs)


def _priced(sctx, bet_id, p) -> float:
    """The copula scorer called directly: the value of a unit staked when the legs win with ``p``.

    ``_ctx`` sets every pick multiplier and pair modifier to 1, so the payout is the
    size's base.
    """
    g = sctx.g
    arr = np.asarray(bet_id)
    size = len(bet_id)
    payout = sctx.payout_base_by_size[size]
    sig = psd_or_none(g.C[np.ix_(bet_id, bet_id)])
    return float(
        parlay_payout_prob(p[arr], g.p_push[arr], sig, size, 1.0, payout, sctx.full_payouts, payout)
    )


def _ctx(
    probs,
    corr,
    *,
    fits=None,
    evs=None,
    platform="Underdog",
    teams=None,
    game="AAA/BBB",
    max_size=None,
    players=None,
    bets=None,
    stars=None,
    date="2026-06-12",
    commence="",
):
    """Build one synthetic GameScoringContext + its matching offers frame.

    ``probs`` is each leg's claimed chance (``p_model``, ``Win Prob``) and ``fits``
    its fitted one (``p_fit``, ``Fit Prob``), the claimed chance unless given.
    ``evs`` is each leg's ``EV``: 0.90 unless given, under 1 like every leg on the
    live board, and tied, so an order by it falls to the earlier leg. ``stars`` is
    the offers frame's ``Star`` column (all 0.0 by default, so a seed falls to the
    earlier player name).
    """
    n = len(probs)
    probs = np.asarray(probs, dtype=float)
    fits = probs.copy() if fits is None else np.asarray(fits, dtype=float)
    evs = [0.90] * n if evs is None else evs
    teams = teams or (["AAA"] * (n // 2) + ["BBB"] * (n - n // 2))
    players = players or [f"Player{i}" for i in range(n)]
    bets = bets or ["Over"] * n
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
        p_fit=fits,
        p_books=probs.copy(),
        p_push=np.zeros(n),
        boosts=np.ones(n),
        shrinkage=np.ones(n),
        opp_boost=np.full(n, np.nan),
    )
    bet_df = {
        i: {
            "EV": evs[i],
            "Fit Prob": fits[i],
            "Pick": False,
            "Bet": bets[i],
            "Team": teams[i],
            "Player": players[i],
            "Market": markets[i],
            "Line": 5.5 + i,
            "Win Prob": probs[i],
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


def _context_frame(*games, date="2026-06-12", shape="shootout"):
    """Context frame giving each game one script; a shootout by default, so an Over may lead."""
    return pd.DataFrame([{"League": "NBA", "Game": g, "Date": date, "shape": shape} for g in games])


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
    # Every leg is a story leg, but no pair correlates (identity ρ) → only singletons →
    # nothing. The lead's orphan fallback needs another player, and these are one
    # player's markets, so even the lead stays alone.
    uncorrelated = _ctx([0.66, 0.65, 0.64], np.eye(3), teams=["AAA"] * 3, players=["A"] * 3)
    assert build_game_stories([uncorrelated[0]], uncorrelated[1], pd.DataFrame(), None).empty


def test_every_pool_leg_is_a_story_leg_at_its_ev():
    """A game's story legs are its pool legs, valued at their ``EV`` with no floor.

    Leg 0's EV is far under 1 and a zero Kelly would have barred it from the old
    menu; leg 1 sits outside the pool, so it stays out whatever its value.
    """
    sctx, _offers = _ctx([0.66, 0.70, 0.62], np.eye(3), evs=[0.78, 0.99, 0.93])
    sctx.bet_df[0]["Kelly"] = 0.0
    pool = dataclasses.replace(sctx, leg_indices=(2, 0))
    assert menu_mod._leg_evs(pool) == {2: 0.93, 0: 0.78}


def test_a_game_whose_every_leg_sits_under_ev_1_gets_a_menu():
    """No value floor on either number: every leg's EV and both presets' fitted and
    claimed values sit under 1, as on the live board, and the game keeps its story."""
    sctx, offers = _ctx(
        [0.42, 0.41, 0.40],
        _block_diag_corr([3], 0.3),
        fits=[0.40, 0.39, 0.38],
        evs=[0.77, 0.75, 0.73],
        teams=["AAA", "BBB", "AAA"],
    )
    out = build_game_stories([sctx], offers, pd.DataFrame(), None)
    assert out["story_id"].nunique() == 1
    assert (out["ev"] < 1).all()
    assert (out["model_ev"] < 1).all()


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
    subset left to pick. Star's highest-EV leg is the third of his four, and the
    claimed chances rank his first one highest.
    """
    sctx, _offers = _ctx(
        [0.70, 0.68, 0.62, 0.64, 0.66, 0.66, 0.66],
        _block_diag_corr([7], 0.4),
        evs=[0.80, 0.84, 0.95, 0.90, 0.86, 0.86, 0.86],
        players=["Star"] * 4 + ["Player4", "Player5", "Player6"],
    )
    ev = menu_mod._leg_evs(sctx)
    order = sorted(ev, key=lambda i: -ev[i])
    clusters = menu_mod._cluster_legs(order, sctx, {})
    assert clusters == [([2, 4, 5, 6], "", "")]  # Star's top-EV leg, then three teammates
    assert len({sctx.bet_df[i]["Player"] for i in clusters[0][0]}) == 4


def test_each_side_leads_a_story_and_led_stories_rank_first():
    # Over block {0,1,2} seeds on the star Player0 and Under block {3,4,5} on the
    # brighter Player4; the unled pair {6,7} is the richest cluster in the game by the
    # fit and the poorest by the claim, whose order the two led blocks reverse too.
    corr = _block_diag_corr([3, 3, 2], 0.3)
    sctx, offers = _ctx(
        [0.70, 0.69, 0.68, 0.66, 0.65, 0.64, 0.50, 0.50],
        corr,
        fits=[0.66, 0.65, 0.64, 0.70, 0.69, 0.68, 0.95, 0.95],
        teams=["AAA", "BBB"] * 4,
        bets=["Over"] * 3 + ["Under"] * 3 + ["Over"] * 2,
        stars=[2.0, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0],
    )
    out = build_game_stories([sctx], offers, _context_frame("AAA/BBB"), None)
    builder = out[out["objective"] == "builder"].set_index("story_id")
    assert builder.index.tolist() == ["AAA/BBB#0", "AAA/BBB#1", "AAA/BBB#2"]
    # Led stories first, ordered between themselves by the Builder's ev; the unled
    # story out-values both yet ranks last, so ev alone does not order the menu.
    assert builder["lead_side"].tolist() == ["Under", "Over", ""]
    assert builder["lead_player"].tolist() == ["Player4", "Player0", ""]
    assert builder["star"].tolist() == [2.5, 2.0, 0.0]
    assert builder["ev"].iloc[2] > builder["ev"].iloc[0] > builder["ev"].iloc[1]
    assert builder["model_ev"].iloc[1] > builder["model_ev"].iloc[0] > builder["model_ev"].iloc[2]
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


def test_unled_stories_rank_by_the_builders_ev():
    """Between unled stories the Builder's fitted value decides, not the Moon's or the claim.

    Six-leg cluster Y (Players 0-5) forms first; three-leg cluster X (Players 6-8)
    has the better Builder by the fit. Y's six-pick Moon out-values X, and the claim
    prefers Y's legs, so either of those orders would put Y first. The slate pass
    flags the story the menu ranks first.
    """
    sctx, offers = _ctx(
        [0.90] * 6 + [0.50] * 3,
        _block_diag_corr([6, 3], 0.3),
        fits=[0.80] * 6 + [0.85] * 3,
        teams=["AAA", "BBB"] * 4 + ["AAA"],
    )
    out = build_game_stories([sctx], offers, pd.DataFrame(), None)
    builder = out[out["objective"] == "builder"].set_index("story_id")
    moon = out[out["objective"] == "moon"].set_index("story_id")
    x, y = "AAA/BBB#0", "AAA/BBB#1"
    assert {leg["player"] for leg in builder.loc[x, "legs"]} == {"Player6", "Player7", "Player8"}
    assert builder.loc[x, "ev"] > builder.loc[y, "ev"]
    assert moon.loc[y, "ev"] > moon.loc[x, "ev"]
    assert builder.loc[y, "model_ev"] > builder.loc[x, "model_ev"]
    assert out.loc[out["lead"], "story_id"].unique().tolist() == [x]


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
    out = build_game_stories([sctx], offers, _context_frame("AAA/BBB"), None)
    assert out["lead_side"].tolist() == ["Under", "Under", "Over", "Over"]
    assert leads == ["Player2", "Player0"]  # rank order: the richer Under-led story first


def test_orphan_lead_pairs_with_the_top_ev_leg_from_the_other_team():
    # The star Over (Player0, AAA) correlates with nothing, so its led cluster is a
    # singleton that prices nothing and borrows a leg. Player3 (AAA) has the top EV
    # but shares the lead's team; Player2 is the top-EV BBB leg.
    probs, evs, stars = [0.66, 0.62, 0.64, 0.65], [0.80, 0.70, 0.90, 0.95], [3.0, 0, 0, 0]
    sctx, offers = _ctx(probs, np.eye(4), teams=["AAA", "BBB", "BBB", "AAA"], evs=evs, stars=stars)
    out = build_game_stories([sctx], offers, _context_frame("AAA/BBB"), None)
    assert out["story_id"].nunique() == 1
    assert out["lead_player"].unique().tolist() == ["Player0"]
    for legs in out["legs"]:
        assert {leg["player"] for leg in legs} == {"Player0", "Player2"}
    # One-sided game: any other player qualifies, so the top EV wins outright.
    sctx, offers = _ctx(probs, np.eye(4), teams=["AAA"] * 4, evs=evs, stars=stars)
    out = build_game_stories([sctx], offers, _context_frame("AAA/BBB"), None)
    assert out["story_id"].nunique() == 1
    for legs in out["legs"]:
        assert {leg["player"] for leg in legs} == {"Player0", "Player3"}


def test_under_lead_survives_a_fully_correlated_pool():
    # Every pair correlates, so the Over-led cluster (the star Player0) absorbs the
    # whole pool and the Under lead (Player5, BBB) is left alone. It still gets its
    # story: it borrows the top-EV AAA leg — Player2 at 0.85, since Player1 and
    # Player3 out-value it but share Player5's team — although the Over cluster
    # already claimed that leg.
    sctx, offers = _ctx(
        [0.66] * 6,
        _block_diag_corr([6], 0.3),
        teams=["AAA", "BBB"] * 3,
        bets=["Over"] * 5 + ["Under"],
        evs=[0.80, 0.95, 0.85, 0.90, 0.80, 0.80],
        stars=[2.0, 0, 0, 0, 0, 0],
    )
    ev = menu_mod._leg_evs(sctx)
    order = sorted(ev, key=lambda i: -ev[i])
    assert menu_mod._cluster_legs(order, sctx, {"Over": 0, "Under": 5}) == [
        ([0, 1, 2, 3, 4], "Over", "Player0"),
        ([5], "Under", "Player5"),
    ]
    out = build_game_stories([sctx], offers, _context_frame("AAA/BBB"), None)
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
    # borrows the top-EV AAA leg (Player0) and the story spans both teams.
    sctx, offers = _ctx(
        [0.66] * 5,
        _block_diag_corr([2, 3], 0.4),
        teams=["AAA", "BBB", "BBB", "BBB", "BBB"],
        bets=["Over", "Over", "Under", "Under", "Under"],
    )
    out = build_game_stories([sctx], offers, _context_frame("AAA/BBB"), None)
    assert set(out["lead_side"]) == {"Over", "Under"}
    under = out[out["lead_side"] == "Under"]
    assert under["story_id"].nunique() == 1
    assert under["lead_player"].iloc[0] == "Player2"
    team_of = dict(zip(offers["Player"], offers["Team"], strict=True))
    for legs in under["legs"]:
        pairs = {(leg["player"], leg["bet"]) for leg in legs}
        assert {("Player2", "Under"), ("Player0", "Over")} <= pairs
        assert {team_of[p] for p in _row_players(legs)} == {"AAA", "BBB"}


def test_one_lead_per_menu_and_the_brighter_over_lead_takes_the_over():
    # Two games leave one Over lead. AAA/BBB's Over lead is the brighter, so the game
    # takes it although ZZZ/YYY tips first.
    corr = _block_diag_corr([2, 2], 0.4)

    def game(name, commence, stars=None):
        return _ctx(
            [0.66, 0.65, 0.66, 0.65],
            corr,
            game=name,
            teams=[name[:3], name[-3:]] * 2,
            players=[f"{name}{i}" for i in range(4)],
            bets=["Over", "Over", "Under", "Under"],
            stars=stars,
            date="2026-10-03",
            commence=commence,
        )

    early, offers_early = game("ZZZ/YYY", "2026-10-03T17:00:00Z")
    late, offers_late = game("AAA/BBB", "2026-10-03T20:00:00Z", stars=[2.0, 0.0, 0.0, 0.0])
    offers = pd.concat([offers_early, offers_late], ignore_index=True)
    out = build_game_stories(
        [late, early], offers, _context_frame("AAA/BBB", "ZZZ/YYY", date="2026-10-03"), None
    )
    assert out["lead"].dtype == bool
    menus = out[out["lead"]].groupby(["platform", "Date", "Game"])
    assert menus["story_id"].nunique().eq(1).all()
    assert menus["objective"].nunique().eq(2).all()  # both rows of the one story
    sides = out[out["lead"]].drop_duplicates("story_id").set_index("Game")["lead_side"]
    assert sides.to_dict() == {"AAA/BBB": "Over", "ZZZ/YYY": "Under"}


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


def test_builder_and_moon_are_the_best_fitted_value_at_their_sizes():
    """Builder is the best ``ev`` among the Power sizes, Moon the best among the larger.

    The claimed chances run against the fitted ones, so presets picked on the claim
    would hold Player0 and Player1. The 4-pick Moon is a 50k-sample Monte-Carlo whose
    value jitters by about 0.01 between scorings; the fixture's best larger subset
    leads the next by 0.15, so the oracle's own scoring lands on the same legs.
    """
    sctx, offers = _ctx(
        [0.80, 0.75, 0.70, 0.45, 0.40],
        _block_diag_corr([5], 0.1),
        fits=[0.40, 0.45, 0.70, 0.75, 0.80],
        teams=["AAA", "BBB", "AAA", "BBB", "AAA"],
    )
    out = build_game_stories([sctx], offers, pd.DataFrame(), None).set_index("objective")
    stat_by_label = leg_market_map(sctx.league, sctx.platform)
    valid = [
        score_subset(c, sctx, stat_by_label)
        for size in range(2, 6)
        for c in combinations(range(5), size)
        if validate_parlay_legs([sctx.bet_df[i] for i in c])[0]
    ]
    best_power = max((s for s in valid if s["bet_size"] <= POWER_MAX_SIZE), key=lambda s: s["ev"])
    best_larger = max((s for s in valid if s["bet_size"] > POWER_MAX_SIZE), key=lambda s: s["ev"])
    assert _leg_keys(out.loc["builder", "legs"]) == _leg_keys(best_power["legs"])
    assert _leg_keys(out.loc["moon", "legs"]) == _leg_keys(best_larger["legs"])
    assert set(_bet_id(out.loc["builder", "legs"])) == {2, 3, 4}
    assert set(_bet_id(out.loc["moon", "legs"])) == {1, 2, 3, 4}


def test_preset_ties_go_to_the_smaller_builder_and_the_larger_moon(monkeypatch):
    """Ties on ``ev``: Builder takes the smaller subset, then the lower ``bet_id``; Moon
    takes the larger subset. The scorer is stubbed so the ties are exact."""
    values = {(0, 1, 4): 2.0, (2, 3): 2.0, (0, 3): 2.0, (0, 1, 2, 3): 3.0, (0, 1, 2, 3, 4): 3.0}
    monkeypatch.setattr(
        menu_mod,
        "score_subset",
        lambda bet_id, _sctx, _labels: {
            "bet_id": bet_id,
            "bet_size": len(bet_id),
            "ev": values.get(bet_id, 1.0),
        },
    )
    sctx, _offers = _ctx([0.66] * 5, np.eye(5), teams=["AAA", "BBB", "AAA", "BBB", "AAA"])
    builder, moon = menu_mod._best_subsets([0, 1, 2, 3, 4], sctx, {})
    assert builder["bet_id"] == (0, 3)
    assert moon["bet_id"] == (0, 1, 2, 3, 4)


def test_shortlist_prices_every_power_subset_and_the_top_larger_ones():
    """Every Power subset reaches the exact scorer; of the larger ones, the eight with the
    best independent proxy on the fitted chances, which run against the claimed ones."""
    sctx, _offers = _ctx(
        [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
        np.eye(7),
        fits=[0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60],
    )
    combos = [c for size in range(2, 7) for c in combinations(range(7), size)]
    larger = [c for c in combos if len(c) > POWER_MAX_SIZE]
    proxy = {c: np.prod(sctx.g.p_fit[list(c)]) * sctx.payout_base_by_size[len(c)] for c in larger}
    top = sorted(larger, key=lambda c: -proxy[c])[:8]
    assert proxy[top[-1]] > proxy[sorted(larger, key=lambda c: -proxy[c])[8]]
    picked = shortlist(combos, sctx)
    assert sorted(picked) == sorted([c for c in combos if len(c) <= POWER_MAX_SIZE] + top)


def test_play_type_cap_honored():
    corr = _block_diag_corr([6], 0.25)
    # Every pick sits high enough that the 6-pick Flex out-values the 4- and 5-pick ones,
    # so Moon takes all six.
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
    # Each platform prices the same legs on its own table. Sleeper pays each pick's own
    # multiplier, neutral here, so its presets value under 1 and it keeps its menu all
    # the same; Underdog's table pays 3.5x and 6.5x.
    by_platform = out.groupby("platform")["ev"]
    assert by_platform.max()["Sleeper"] < 1 < by_platform.min()["Underdog"]


def test_ev_is_the_copula_scorer_on_the_fitted_chances():
    sctx, _ = _ctx([0.66, 0.64], _block_diag_corr([2], 0.4), fits=[0.55, 0.58])
    stat_by_label = leg_market_map(sctx.league, sctx.platform)
    scored = score_subset((0, 1), sctx, stat_by_label)
    direct = _priced(sctx, (0, 1), sctx.g.p_fit)
    assert scored["ev"] == direct
    assert scored["win_prob"] == direct / sctx.payout_base_by_size[2]
    assert direct < _priced(sctx, (0, 1), sctx.g.p_model)


def test_rows_price_the_fit_beside_the_claim_and_stake_nothing():
    """``ev`` and ``joint_p`` price a preset on the fitted chances and ``model_ev`` prices
    the same legs on the claimed ones, so the two differ wherever the chances do. Each
    preset's fitted value clears 1, where full Kelly would stake it, yet ``kelly_stake``
    is 0.0: an entry gets no stake until the entries' money record supports one."""
    sctx, offers = _ctx(
        [0.70, 0.65, 0.60],
        _block_diag_corr([3], 0.3),
        fits=[0.50, 0.55, 0.60],
        teams=["AAA", "BBB", "AAA"],
    )
    out = build_game_stories([sctx], offers, pd.DataFrame(), None)
    assert len(out) == 2
    for row in out.to_dict("records"):
        bet_id = _bet_id(row["legs"])
        assert row["ev"] == pytest.approx(_priced(sctx, bet_id, sctx.g.p_fit), rel=1e-3)
        assert row["model_ev"] == pytest.approx(_priced(sctx, bet_id, sctx.g.p_model), rel=1e-3)
        assert row["joint_p"] == pytest.approx(row["ev"] / sctx.payout_base_by_size[len(bet_id)])
        assert 1 < row["ev"] < row["model_ev"] - 0.5
        assert row["kelly_stake"] == 0.0


def test_a_cluster_with_no_valid_larger_subset_gives_both_presets_the_same_legs():
    # Three legs hold no subset above Power's three picks, so Moon repeats the Builder,
    # numbers and all: one entry, priced once.
    sctx, offers = _ctx(
        [0.70, 0.65, 0.60],
        _block_diag_corr([3], 0.3),
        fits=[0.50, 0.55, 0.60],
        teams=["AAA", "BBB", "AAA"],
    )
    out = build_game_stories([sctx], offers, pd.DataFrame(), None).set_index("objective")
    assert _leg_keys(out.loc["builder", "legs"]) == _leg_keys(out.loc["moon", "legs"])
    numbers = ["ev", "model_ev", "joint_p", "bet_size"]
    assert out.loc["builder", numbers].tolist() == out.loc["moon", numbers].tolist()


def test_flex_loss_tier_prices_the_largest_multipliers_times_the_pair_modifier():
    # One sure loss on a 4-pick Flex: the 1-loss tier pays on the three largest pick
    # multipliers, whichever pick lost, times the pair-modifier product.
    sctx, _ = _ctx([1.0, 1.0, 0.0, 1.0], np.eye(4))
    sctx.g.boosts[:] = [0.9, 1.1, 1.2, 0.8]  # the loser holds the largest multiplier
    sctx.g.M[0, 1] = sctx.g.M[1, 0] = 0.85
    stat_by_label = leg_market_map(sctx.league, sctx.platform)

    scored = score_subset((0, 1, 2, 3), sctx, stat_by_label)

    one_loss_tier = sctx.full_payouts[4][1]
    assert scored["ev"] == pytest.approx(one_loss_tier * 1.2 * 1.1 * 0.9 * 0.85, rel=1e-3)


@pytest.mark.parametrize(
    ("shape", "led"),
    [
        ("shootout", {"Over", "Under"}),
        ("blowout", {"Over", "Under"}),
        ("grind", {"Under", ""}),
        ("coinflip", {"Under", ""}),
        ("even", {"Under", ""}),
        (None, {"Under", ""}),
    ],
)
def test_an_over_leads_only_a_shootout_or_a_blowout(shape, led):
    """Owner rule (2026-10-17): an Over-led story exists only in a game whose script runs
    hot or lopsided. A grind, a coinflip, an unquoted game ("even") or a game with no
    context row leads on its Under alone; its Over legs still ride in an unled story."""
    sctx, offers = _ctx(
        [0.66, 0.65, 0.70, 0.69],
        _block_diag_corr([2, 2], 0.4),
        teams=["AAA", "BBB"] * 2,
        bets=["Over", "Over", "Under", "Under"],
    )
    context = pd.DataFrame() if shape is None else _context_frame("AAA/BBB", shape=shape)
    out = build_game_stories([sctx], offers, context, None)
    assert set(out["lead_side"]) == led
    if "Over" not in led:
        unled_legs = [leg for legs in out.loc[out["lead_side"] == "", "legs"] for leg in legs]
        assert ("Player0", "Over") in {(leg["player"], leg["bet"]) for leg in unled_legs}


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
    # When the game's story legs are ALL on one team, a single-team preset IS allowed —
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
    # The game's story legs span both teams but each correlation cluster is one-sided
    # and none bridges them. The Over-led AAA cluster borrows the top-EV BBB leg
    # (Player2); the unled BBB cluster has no such recourse and emits nothing.
    corr = _block_diag_corr([2, 2], 0.4)
    sctx, offers = _ctx([0.66, 0.65, 0.64, 0.63], corr, teams=["AAA", "AAA", "BBB", "BBB"])
    out = build_game_stories([sctx], offers, _context_frame("AAA/BBB"), None)
    assert out["story_id"].nunique() == 1
    assert out["lead_side"].eq("Over").all()
    team_of = dict(zip(offers["Player"], offers["Team"], strict=True))
    for legs in out["legs"]:
        players = _row_players(legs)
        assert {"Player0", "Player2"} <= set(players)
        assert {team_of[p] for p in players} == {"AAA", "BBB"}


def _divergent_presets():
    """One story whose Builder is three Power picks and whose Moon adds a fourth."""
    corr = _block_diag_corr([4], 0.1)
    sctx, offers = _ctx([0.90, 0.75, 0.70, 0.65], corr, teams=["AAA", "BBB"] * 2)
    return build_game_stories([sctx], offers, pd.DataFrame(), None)


def test_builder_and_moon_share_headline_and_dek():
    """One story, one headline, one dek — the mode chip must never swap the prose."""
    out = _divergent_presets()
    builder = out[out["objective"] == "builder"].iloc[0]
    moon = out[out["objective"] == "moon"].iloc[0]
    assert _leg_keys(builder["legs"]) != _leg_keys(moon["legs"])  # divergent presets
    for _sid, grp in out.groupby("story_id"):
        assert grp["headline"].nunique() == 1
        assert grp["dek"].nunique() == 1


def test_headline_names_only_players_shared_by_both_presets():
    out = _divergent_presets()
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
    first = build_game_stories([sctx], offers, _context_frame("AAA/BBB"), None)
    second = build_game_stories([sctx], offers, _context_frame("AAA/BBB"), None)
    assert first["dek"].tolist() == second["dek"].tolist()
    dek = first["dek"].iloc[0]
    assert dek
    assert "0.45" in dek  # the 2-leg core's mean pairwise rho, slot-formatted :.2f
    # The anchor's form/matchup clauses render only if offer_index keeps the
    # "Avg 5"/"DVPOA" facts — Player0 is the Over lead and the leg with the highest
    # fitted chance, so it is the dek's subject either way.
    assert "Player0" in dek


def test_unled_dek_features_the_core_leg_with_the_highest_fitted_chance():
    """The claim ranks Player0's leg first and the fit Player1's; the dek features the fit's."""
    corr = np.eye(2)
    corr[0, 1] = corr[1, 0] = 0.45
    sctx, offers = _ctx([0.70, 0.60], corr, fits=[0.55, 0.65])
    out = build_game_stories([sctx], offers, pd.DataFrame(), None)
    assert out["lead_side"].eq("").all()
    assert set(re.findall(r"Player\d", out["dek"].iloc[0])) == {"Player1"}


def _led_pair_stories() -> pd.DataFrame:
    """One story per story_id on a slate led on both sides by the pair {Player0, Player1}.

    Each led cluster borrows the other team's leg, so both led stories price that
    pair; Player1's Under carries the higher fitted chance. Every offer runs 2.0 above
    its line in a matchup that gives the stat (``_ctx``), so those facts back an
    Over and cut against an Under.
    """
    corr = _block_diag_corr([2, 2], 0.4)
    sctx, offers = _ctx(
        [0.55, 0.70, 0.60, 0.66],
        corr,
        teams=["AAA", "BBB"] * 2,
        bets=["Over", "Under", "Over", "Over"],
    )
    out = build_game_stories([sctx], offers, _context_frame("AAA/BBB"), None)
    return out.drop_duplicates("story_id")


def test_led_story_dek_is_about_its_lead():
    """A led story's dek names its lead and nobody else; an unled story keeps the
    anchor with the highest fitted chance, which an anchor picked on that chance
    alone would make the Over-led dek's subject too (Player1)."""
    stories = _led_pair_stories()
    named = {
        (row["lead_side"], row["lead_player"]): set(re.findall(r"Player\d", row["dek"]))
        for _, row in stories.iterrows()
    }
    assert set(named) == {("Over", "Player0"), ("Under", "Player1"), ("", "")}
    assert len(stories) == 3
    for (side, lead), players in named.items():
        if lead:
            assert players <= {lead}, (side, lead, players)
    assert named[("Over", "Player0")] == {"Player0"}
    assert named[("", "")] == {"Player3"}
    over_dek = stories.set_index("lead_side").loc["Over", "dek"]
    above_for = why_bank()["dek"]["form"]["above_for"]
    assert any(v.format(p="Player0", dev="2", line="5.5") in over_dek for v in above_for)


def test_led_story_dek_never_argues_against_its_bet():
    """An Under lead running above the line in a matchup that gives the stat gets
    neither the form nor the matchup clause: the dek keeps the cluster read and
    names nobody (the per-offer Why keeps both reads, see test_stories)."""
    under_dek = _led_pair_stories().set_index("lead_side").loc["Under", "dek"]
    assert "0.40" in under_dek
    assert not re.findall(r"Player\d", under_dek)
