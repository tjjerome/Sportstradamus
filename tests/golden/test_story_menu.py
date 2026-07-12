"""Story-menu generator golden tests (P3a).

Exercises ``prediction.stories.menu.build_game_stories`` on hand-built
``GameScoringContext`` fixtures (synthetic ``GameArrays`` with explicit
correlation / probabilities, the same shape ``find_correlation`` hands the
generator). Pins the owner-locked acceptance criteria: ≤5 correlation-cluster
stories per game, no-signal games yield nothing, the two objectives are the true
argmaxes, the play-type cap holds, platforms are independent, and the EV is the
real copula scorer (not a reimplementation).
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
    _log_growth,
    _score_subset,
    _story_prose,
    build_game_stories,
)


def _leg_keys(legs: list[dict]) -> frozenset:
    return frozenset((leg["player"], leg["bet"], leg["line"], leg["market"]) for leg in legs)


def _ctx(
    probs, corr, *, platform="Underdog", edges=None, teams=None, game="AAA/BBB", max_size=None
):
    """Build one synthetic GameScoringContext + its matching offers frame."""
    n = len(probs)
    probs = np.asarray(probs, dtype=float)
    edges = [1.30] * n if edges is None else edges
    teams = teams or (["AAA"] * (n // 2) + ["BBB"] * (n - n // 2))
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
    )
    bet_df = {
        i: {
            "Model EV": edges[i],
            "Bet": "Over",
            "Team": teams[i],
            "Player": f"Player{i}",
            "Market": markets[i],
            "Line": 5.5 + i,
            "Win Prob": probs[i],
        }
        for i in range(n)
    }
    search, full = payout_curve_for(platform, "pooled", legacy=False)
    if max_size is not None:
        full = {s: full[s] for s in full if s <= max_size}
    pay_base = {s: search[s - 2] for s in range(2, len(search) + 2) if s <= max(full)}
    sctx = GameScoringContext(
        platform=platform,
        league="NBA",
        game=game,
        date="2026-06-12",
        g=g,
        bet_df=bet_df,
        leg_indices=tuple(range(n)),
        full_payouts=full,
        payout_base_by_size=pay_base,
        max_size=max(full),
    )
    offers = pd.DataFrame(
        [
            {
                "Player": f"Player{i}",
                "Bet": "Over",
                "Line": 5.5 + i,
                "Market": markets[i],
                "Game": game,
                "Team": teams[i],
                "Position": "",
                "Avg 5": 2.0,
                "DVPOA": 0.10,
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
    uncorrelated = _ctx([0.66, 0.65, 0.64], np.eye(3))
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


def test_correlation_cluster_forms_over_exact_legs():
    corr = np.eye(2)
    corr[0, 1] = corr[1, 0] = 0.45
    sctx, offers = _ctx([0.62, 0.60], corr)
    out = build_game_stories([sctx], offers, pd.DataFrame(), None)
    assert out["story_id"].nunique() == 1
    for row in out["legs"]:
        assert {leg["player"] for leg in row} == {"Player0", "Player1"}


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
    # The menu argmaxes over *valid* subsets only, so the oracle must too —
    # otherwise a single-team subset could pose as the "expected" argmax.
    new_map = _leg_market_map(sctx.league, sctx.platform, stat_map)
    all_scores = [
        _score_subset(c, sctx, new_map)
        for size in range(2, min(len(sctx.leg_indices), sctx.max_size) + 1)
        for c in combinations(sctx.leg_indices, size)
        if validate_parlay_legs([sctx.bet_df[i] for i in c])[0]
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
    probs = [0.66, 0.65, 0.64, 0.63, 0.62, 0.61]
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
    scored = _score_subset(bet_id, sctx, new_map)
    g = sctx.g
    arr = np.asarray(bet_id)
    boost = float(g.M[0, 1] * g.boosts[0] * g.boosts[1])
    payout = float(np.clip(boost * sctx.payout_base_by_size[2], 1.0, 100.0))
    sig = psd_or_none(g.C[np.ix_(bet_id, bet_id)], legacy=False)
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
            False,
        )
    )
    assert scored["model_ev"] == direct


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


def test_single_team_clusters_in_two_team_game_emit_no_story():
    # The game's edge spans both teams but each correlation cluster is one-sided and none
    # bridges them → the whole-game gate keeps both-teams required → no story emitted.
    corr = _block_diag_corr([2, 2], 0.4)
    sctx, offers = _ctx([0.66, 0.65, 0.64, 0.63], corr, teams=["AAA", "AAA", "BBB", "BBB"])
    assert build_game_stories([sctx], offers, pd.DataFrame(), None).empty


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
        menu_mod, "thesis_variants", lambda _legs, _ctxs: (["Ghost rules"], 0, {"p": "Ghost"})
    )
    sctx, offers = _ctx([0.66, 0.64], np.eye(2))
    headline, dek = _story_prose({"bet_id": (0,)}, {"bet_id": (1,)}, sctx, offers, {}, set())
    assert headline == ""
    assert dek == ""  # a sub-2-leg core carries no cluster clause and no anchor row facts
    # An unnamed (game-script) subject survives the same disjoint presets.
    monkeypatch.setattr(
        menu_mod, "thesis_variants", lambda _legs, _ctxs: (["The game tilts over"], 0, {"g": "X/Y"})
    )
    headline, _dek = _story_prose({"bet_id": (0,)}, {"bet_id": (1,)}, sctx, offers, {}, set())
    assert headline == "The game tilts over"


def test_within_game_headline_dedup(monkeypatch):
    """Two stories seeded onto the same variant: the second bumps to the next one."""
    monkeypatch.setattr(
        menu_mod,
        "thesis_variants",
        lambda _legs, _ctxs: (["Same story", "Second story"], 0, {}),
    )
    corr = _block_diag_corr([2, 2], 0.4)
    sctx, offers = _ctx([0.66] * 4, corr, teams=["AAA", "BBB"] * 2)
    out = build_game_stories([sctx], offers, pd.DataFrame(), None)
    assert out["story_id"].nunique() == 2
    assert out.drop_duplicates("story_id")["headline"].tolist() == ["Same story", "Second story"]


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
