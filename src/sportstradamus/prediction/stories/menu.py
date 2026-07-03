"""Per-game story menu: correlation-cluster stories with two starting parlays.

``build_game_stories`` turns each game's scoring bundle (``GameScoringContext``,
captured by ``find_correlation``'s ``story_sink``) into up to five data-driven
*stories* — greedy correlation clusters of the game's strong legs. Each story
emits two starting parlays over its cluster's legs (legs may be shared):

* **Bankroll Builder** — the leg-subset with the highest single-bet full-Kelly
  *log-growth* ``G`` (the bet that compounds bankroll fastest, not the biggest
  stake). Tends to the tight 2-3-leg core.
* **Shoot the Moon** — the leg-subset with the highest *model EV* inside the
  play-type cap (the widest high-edge set; usually a flex extension of Builder).

Pure and ``Archive``-free so the P3 dashboard rail can recompute it live. Subsets
are enumerated but scored in two phases — a cheap independent-joint proxy ranks
every subset, then only a shortlist is priced through the real Gaussian-copula
scorer (``parlay._parlay_payout_prob``), whose flex branch runs a 50k-sample
Monte-Carlo. The final argmax always uses the exact score, so fidelity holds while
the expensive MC stays bounded.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np
import pandas as pd

from sportstradamus.analysis import _leg_market_map
from sportstradamus.helpers import stat_map
from sportstradamus.leg_schema import build_leg
from sportstradamus.prediction.parlay import (
    _PAYOUT_CLIP_HI,
    _PAYOUT_CLIP_LO,
    _POWER_MAX_SIZE,
    GameScoringContext,
    _parlay_payout_prob,
    _psd_or_none,
    _resolve_leg_stat,
)
from sportstradamus.prediction.stories.context import GameCtx, ctxs_from_frame
from sportstradamus.prediction.stories.engine import thesis_variants
from sportstradamus.prediction.stories.legs import enrich_legs, validate_parlay_legs
from sportstradamus.prediction.stories.thesis import next_unique_variant
from sportstradamus.prediction.stories.why import story_dek

# A leg qualifies as a story seed when its per-$1 model EV clears this edge — the
# same 0.05 the unit-thesis gate and per-offer "why" already use.
_MENU_EDGE_FLOOR: float = 0.05
# Two strong legs join a cluster when |rho| over their pair clears this; matches
# the display-correlation floor in correlation.py.
_CLUSTER_RHO_FLOOR: float = 0.05
# Brute-force bound: 2**8 = 256 subsets keeps a cluster's enumeration instant.
_MAX_CLUSTER_LEGS: int = 8
# Owner-locked menu cap: at most five stories per (platform, game).
_MAX_STORIES: int = 5
# Drop a cluster whose best Shoot-the-Moon subset can't clear breakeven EV.
_MENU_MIN_MOON_EV: float = 1.0
# Exact-scored flex finalists per objective per cluster (Power subsets are all
# exact-scored cheaply via the analytical mvn.cdf, so they bypass the shortlist).
_SHORTLIST_K: int = 8

_STORY_COLS = [
    "platform",
    "League",
    "Game",
    "story_id",
    "objective",
    "headline",
    "legs",
    "joint_p",
    "model_ev",
    "kelly_stake",
    "bet_size",
    "Date",
    "dek",
]


def build_game_stories(
    story_ctxs: Sequence[GameScoringContext],
    offers: pd.DataFrame,
    context: pd.DataFrame,
    corr: list[dict] | None,
) -> pd.DataFrame:
    """One menu (≤5 stories × 2 objectives) per ``(platform, game)``.

    ``story_ctxs`` is the ``story_sink`` filled by ``find_correlation`` (one per
    game per platform); ``context``/``corr`` are the already-built
    ``current_game_context`` frame and ``current_game_corr`` slices, reused to
    headline each story via the P2 thesis engine.
    """
    ctxs = ctxs_from_frame(context, corr)
    rows: list[dict] = []
    for sctx in story_ctxs:
        rows.extend(_stories_for_game(sctx, offers, ctxs))
    return pd.DataFrame(rows, columns=_STORY_COLS)


def _stories_for_game(
    sctx: GameScoringContext, offers: pd.DataFrame, ctxs: Mapping[str, GameCtx]
) -> list[dict]:
    edge = _strong_legs(sctx)
    if len(edge) < 2:
        return []
    # Whole-game gate: only when the model's edge is entirely one-sided may a story
    # be a single-team preset (the user completes it with a satellite leg). A game
    # with edge on both teams still requires both-teams presets.
    require_both = (
        len({sctx.bet_df[i].get("Team") for i in edge if sctx.bet_df[i].get("Team")}) >= 2
    )
    new_map = _leg_market_map(sctx.league, sctx.platform, stat_map)
    scored = []
    for cluster in _cluster_strong_legs(edge, sctx.g.C):
        builder, moon = _best_subsets(cluster, sctx, new_map, require_both_teams=require_both)
        if builder is not None:
            scored.append((cluster, builder, moon))
    scored.sort(
        key=lambda cb: (
            -cb[2]["model_ev"],
            -max(edge[i] for i in cb[0]),
            cb[2]["bet_id"],
        )
    )
    rows: list[dict] = []
    seen: set[str] = set()
    for rank, (_cluster, builder, moon) in enumerate(scored[:_MAX_STORIES]):
        story_id = f"{sctx.game}#{rank}"
        headline, dek = _story_prose(builder, moon, sctx, offers, ctxs, seen)
        rows.append(_row(sctx, story_id, "builder", builder, headline, dek))
        rows.append(_row(sctx, story_id, "moon", moon, headline, dek))
    return rows


def _strong_legs(sctx: GameScoringContext) -> dict[int, float]:
    """Bet-eligible, model-favored legs mapped to their per-$1 edge.

    Every leg in ``leg_indices`` is already a player-prop or Rivals leg (game
    lines are L3-gated, not yet in the candidate set), so eligibility reduces to
    the strong-edge floor.
    """
    return {
        i: sctx.bet_df[i]["Model EV"]
        for i in sctx.leg_indices
        if sctx.bet_df[i]["Model EV"] - 1.0 >= _MENU_EDGE_FLOOR
    }


def _cluster_strong_legs(edge: Mapping[int, float], corr: np.ndarray) -> list[list[int]]:
    """Greedy ρ-graph clusters over the strong legs; singleton clusters dropped.

    Seeds on the strongest remaining leg and attaches any leg correlated with a
    current member, so an isolated strong leg forms no story (a thin game yields
    few or zero) and a correlated bundle forms exactly one.
    """
    remaining = set(edge)
    order = sorted(edge, key=lambda i: -edge[i])
    clusters = []
    while remaining:
        seed = next(i for i in order if i in remaining)
        cluster = _grow_cluster(seed, remaining, order, corr)
        if len(cluster) >= 2:
            clusters.append(sorted(cluster))
    return clusters


def _grow_cluster(
    seed: int, remaining: set[int], order: Sequence[int], corr: np.ndarray
) -> list[int]:
    """Attach legs correlated (|ρ| ≥ floor) with any member, strongest first, up to the cap."""
    cluster = [seed]
    remaining.discard(seed)
    changed = True
    while changed and len(cluster) < _MAX_CLUSTER_LEGS:
        changed = False
        for j in order:
            if j not in remaining:
                continue
            if any(abs(corr[j, m]) >= _CLUSTER_RHO_FLOOR for m in cluster):
                cluster.append(j)
                remaining.discard(j)
                changed = True
                if len(cluster) >= _MAX_CLUSTER_LEGS:
                    break
    return cluster


def _best_subsets(
    cluster: Sequence[int],
    sctx: GameScoringContext,
    new_map: dict,
    *,
    require_both_teams: bool = True,
) -> tuple[dict | None, dict | None]:
    """The (Builder, Moon) parlays for one cluster, or (None, None) if degenerate.

    Only **valid** parlays are enumerated, so the Builder/Moon picks are valid by
    construction. With ``require_both_teams`` (a two-team game) a one-team cluster
    yields ``(None, None)``; a one-sided game relaxes that, so its single-team
    cluster still produces a preset. Phase 1 ranks every candidate by a pure-numpy
    independent-joint proxy; phase 2 exact-scores only the shortlist (top-K by each
    objective ∪ all cheap Power subsets) through the copula scorer.
    """
    proxies = [
        (combo, *_independent(combo, sctx))
        for size in range(2, min(len(cluster), sctx.max_size) + 1)
        for combo in combinations(cluster, size)
        if validate_parlay_legs(
            [sctx.bet_df[i] for i in combo], require_both_teams=require_both_teams
        )[0]
    ]
    if not proxies:
        return None, None
    scored = [_score_subset(bet_id, sctx, new_map) for bet_id in _shortlist(proxies)]
    builder = min(scored, key=lambda s: (-s["G"], s["bet_size"], s["bet_id"]))
    moon = min(scored, key=lambda s: (-s["model_ev"], -s["bet_size"], s["bet_id"]))
    if moon["model_ev"] <= _MENU_MIN_MOON_EV:
        return None, None
    return builder, moon


def _shortlist(proxies: Sequence[tuple]) -> list[tuple[int, ...]]:
    """Distinct subsets worth an exact score: top-K by EV ∪ top-K by G ∪ all Power."""
    by_ev = sorted(proxies, key=lambda x: -x[1])[:_SHORTLIST_K]
    by_g = sorted(proxies, key=lambda x: -x[2])[:_SHORTLIST_K]
    power = [p for p in proxies if len(p[0]) <= _POWER_MAX_SIZE]
    picked = {p[0]: None for p in (*by_ev, *by_g, *power)}
    return list(picked)


def _independent(bet_id: Sequence[int], sctx: GameScoringContext) -> tuple[float, float]:
    """Cheap proxy: (independent EV, independent log-growth) — no copula, no MC."""
    p_ind = float(np.prod(sctx.g.p_model[np.asarray(bet_id)]))
    _boost, payout = _boost_payout(bet_id, sctx)
    return p_ind * payout, _log_growth(p_ind, payout)


def _score_subset(bet_id: Sequence[int], sctx: GameScoringContext, new_map: dict) -> dict:
    """Exact copula score for one subset (reuses parlay's gate-free scorer)."""
    size = len(bet_id)
    g = sctx.g
    arr = np.asarray(bet_id)
    boost, payout = _boost_payout(bet_id, sctx)
    sig = _psd_or_none(g.C[np.ix_(bet_id, bet_id)], legacy=False)
    model_ev = float(
        _parlay_payout_prob(
            g.p_model[arr],
            g.p_push[arr],
            sig,
            size,
            boost,
            payout,
            sctx.full_payouts,
            sctx.payout_base_by_size[size],
            False,
        )
    )
    win_prob = model_ev / payout if payout > 0 else 0.0
    return {
        "bet_id": tuple(bet_id),
        "bet_size": size,
        "model_ev": model_ev,
        "win_prob": win_prob,
        "G": _log_growth(win_prob, payout),
        "kelly_stake": _kelly_fraction(win_prob, payout),
        "legs": [
            build_leg(
                {
                    **sctx.bet_df[i],
                    "League": sctx.league,
                    "Game": sctx.game,
                    "Date": sctx.date,
                    "Platform": sctx.platform,
                    "Stat": _resolve_leg_stat(sctx.bet_df[i]["Market"], new_map),
                }
            )
            for i in bet_id
        ],
    }


def _boost_payout(bet_id: Sequence[int], sctx: GameScoringContext) -> tuple[float, float]:
    """Modifier-product boost and clipped payout multiplier for a subset (no admissibility gate)."""
    size = len(bet_id)
    g = sctx.g
    pairs = g.M[np.ix_(bet_id, bet_id)][np.triu_indices(size, 1)]
    boost = float(np.prod(pairs) * np.prod(g.boosts[np.asarray(bet_id)]))
    payout = float(
        np.clip(boost * sctx.payout_base_by_size[size], _PAYOUT_CLIP_LO, _PAYOUT_CLIP_HI)
    )
    return boost, payout


def _leg_dict(sctx: GameScoringContext, i: int) -> dict:
    """Minimal lowercase leg dict for ``enrich_legs`` — no ``build_leg`` needed here."""
    row = sctx.bet_df[i]
    return {"player": row["Player"], "bet": row["Bet"], "line": row["Line"], "market": row["Market"]}


def _kelly_fraction(p: float, payout: float) -> float:
    """Full-Kelly fraction of bankroll for a single bet; 0 when there's no edge."""
    b = payout - 1.0
    if b <= 0.0:
        return 0.0
    return max(0.0, (p * (b + 1.0) - 1.0) / b)


def _log_growth(p: float, payout: float) -> float:
    """Expected log-growth of a single parlay bet at its own full-Kelly fraction."""
    f = _kelly_fraction(p, payout)
    if f <= 0.0:
        return 0.0
    b = payout - 1.0
    return p * math.log1p(b * f) + (1.0 - p) * math.log1p(-f)


def _row(
    sctx: GameScoringContext,
    story_id: str,
    objective: str,
    sub: Mapping,
    headline: str,
    dek: str,
) -> dict:
    return {
        "platform": sctx.platform,
        "League": sctx.league,
        "Game": sctx.game,
        "story_id": story_id,
        "objective": objective,
        "headline": headline,
        "legs": sub["legs"],
        "joint_p": sub["win_prob"],
        "model_ev": sub["model_ev"],
        "kelly_stake": sub["kelly_stake"],
        "bet_size": sub["bet_size"],
        "Date": sctx.date,
        "dek": dek,
    }


def _story_prose(
    builder: Mapping,
    moon: Mapping,
    sctx: GameScoringContext,
    offers: pd.DataFrame,
    ctxs: Mapping[str, GameCtx],
    seen: set[str],
) -> tuple[str, str]:
    """One (headline, dek) per story — the mode chips swap legs, never the prose.

    The headline renders from the legs the two presets share, so a player it
    names is on whichever preset loads; disjoint presets (rare) render from
    the union and blank rather than name a player only one side carries.
    ``seen`` dedupes headlines within the (platform, game) menu.
    """
    per_sub = [[_leg_dict(sctx, i) for i in sub["bet_id"]] for sub in (builder, moon)]
    core = sorted(set(builder["bet_id"]) & set(moon["bet_id"]))
    parsed = [_leg_dict(sctx, i) for i in core] if core else per_sub[0] + per_sub[1]
    variants, vi, subject = thesis_variants(enrich_legs(parsed, offers), ctxs)
    dek = story_dek(core, sctx, offers)
    named = subject.get("p")
    if named and any(named not in {leg["player"] for leg in legs} for legs in per_sub):
        return "", dek
    headline = next_unique_variant(variants, vi, seen) if variants else ""
    if headline:
        seen.add(headline)
    return headline, dek
