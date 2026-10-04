"""Per-game story menu: correlation-cluster stories with two starting parlays.

``build_game_stories`` turns each game's scoring bundle (``GameScoringContext``,
captured by ``find_correlation``'s ``story_sink``) into up to five data-driven
*stories* — correlation clusters of the game's strong legs. Each story emits two
starting parlays over its cluster's legs (legs may be shared):

* **Bankroll Builder** — the leg-subset with the highest single-bet full-Kelly
  *log-growth* ``G`` (the bet that compounds bankroll fastest, not the biggest
  stake). Tends to the tight 2-3-leg core.
* **Shoot the Moon** — the leg-subset with the highest *model EV* inside the
  play-type cap (the widest high-edge set; usually a flex extension of Builder).

Ranked on edge alone the menu opens every game on a role player's Under (the
highest-probability leg on DFS x.5 lines), so two stories per game are *led*:
one grows from the most prominent strong Over and one from the most prominent
strong Under (``lead.lead_seeds``), the lead sits in both presets, led stories
rank first, and ``lead.assign_leads`` flags the one each menu headlines.

Pure and ``Archive``-free so the P3 dashboard rail can recompute it live. Subsets
are enumerated but scored in two phases (``pricing``) — a cheap independent-joint
proxy ranks every subset, then only a shortlist is priced through the real
Gaussian-copula scorer (``joint.parlay_payout_prob``), whose flex branch runs a
50k-sample Monte-Carlo. The final argmax always uses the exact score, so fidelity
holds while the expensive MC stays bounded.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np
import pandas as pd

from sportstradamus.analysis import _leg_market_map
from sportstradamus.helpers import stat_map
from sportstradamus.prediction.parlay import GameScoringContext
from sportstradamus.prediction.stories.context import GameCtx, ctxs_from_frame
from sportstradamus.prediction.stories.engine import thesis_variants
from sportstradamus.prediction.stories.lead import LEAD_SIDES, assign_leads, lead_seeds
from sportstradamus.prediction.stories.legs import (
    enrich_legs,
    lower_leg,
    offer_index,
    validate_parlay_legs,
)
from sportstradamus.prediction.stories.pricing import independent, score_subset, shortlist
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

# ``lead`` is written by ``assign_leads`` over the finished frame, not per row.
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
    "lead_side",
    "lead_player",
    "star",
    "lead",
]


def build_game_stories(
    story_ctxs: Sequence[GameScoringContext],
    offers: pd.DataFrame,
    context: pd.DataFrame,
    corr: list[dict] | None,
) -> pd.DataFrame:
    """One menu (≤5 stories × 2 objectives) per ``(platform, game)``, lead-flagged.

    ``story_ctxs`` is the ``story_sink`` filled by ``find_correlation`` (one per
    game per platform); ``offers`` carries ``Star`` (``lead.attach_prominence``);
    ``context``/``corr`` are the already-built ``current_game_context`` frame and
    ``current_game_corr`` slices, reused to headline each story via the P2
    thesis engine. Headlines dedupe across each (platform, date) slate, and
    ``assign_leads`` walks each date's games by tip: the first non-empty
    ``Commence`` on any context of the game, since Sleeper rows carry none.
    """
    ctxs = ctxs_from_frame(context, corr)
    seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    commence_by_game: dict[tuple[str, str], str] = {}
    rows: list[dict] = []
    for sctx in story_ctxs:
        key = (sctx.date, sctx.game)
        commence_by_game[key] = commence_by_game.get(key) or next(
            (rec["Commence"] for rec in sctx.bet_df.values() if rec["Commence"]), ""
        )
        rows.extend(_stories_for_game(sctx, offers, ctxs, seen[sctx.platform, sctx.date]))
    return assign_leads(pd.DataFrame(rows, columns=_STORY_COLS), commence_by_game)


def _stories_for_game(
    sctx: GameScoringContext,
    offers: pd.DataFrame,
    ctxs: Mapping[str, GameCtx],
    seen: set[str],
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
    index = offer_index(offers)
    stars = {}
    for i in edge:
        rec = sctx.bet_df[i]
        stars[i] = index.get((rec["Player"], rec["Bet"], rec["Line"]), {}).get("Star", 0.0)
    seeds = lead_seeds(edge, sctx.bet_df, stars)
    order = sorted(edge, key=lambda i: -edge[i])
    scored = []
    for cluster, side, lead in _cluster_strong_legs(order, sctx, seeds):
        priced = _price_cluster(
            cluster, side, lead, seeds.get(side), order, sctx, new_map, require_both=require_both
        )
        if priced is not None:
            scored.append(priced)
    # Led stories first, between themselves by Moon EV; then the rest by EV.
    scored.sort(
        key=lambda s: (s[3] == "", -s[2]["model_ev"], -max(edge[i] for i in s[0]), s[2]["bet_id"])
    )
    rows: list[dict] = []
    for rank, (_cluster, builder, moon, side, lead) in enumerate(scored[:_MAX_STORIES]):
        headline, dek = _story_prose(builder, moon, sctx, offers, index, ctxs, new_map, seen, lead)
        story = {
            "story_id": f"{sctx.game}#{rank}",
            "headline": headline,
            "dek": dek,
            "lead_side": side,
            "lead_player": lead,
            "star": max(stars[i] for i in set(builder["bet_id"]) | set(moon["bet_id"])),
        }
        rows.append(_row(sctx, "builder", builder, story))
        rows.append(_row(sctx, "moon", moon, story))
    return rows


def _strong_legs(sctx: GameScoringContext) -> dict[int, float]:
    """Bet-eligible, model-favored legs mapped to their per-$1 edge.

    Every leg in ``leg_indices`` is already a player prop (game lines are
    L3-gated, not yet in the candidate set), so eligibility is the strong-edge
    floor plus ``Kelly > 0``, which excludes payouts at or below 1x and above
    the favored cap so no -EV or deep-alt leg can seed or join a story.
    """
    return {
        i: sctx.bet_df[i]["Model EV"]
        for i in sctx.leg_indices
        if sctx.bet_df[i]["Model EV"] - 1.0 >= _MENU_EDGE_FLOOR and sctx.bet_df[i]["Kelly"] > 0
    }


def _cluster_strong_legs(
    order: Sequence[int], sctx: GameScoringContext, seeds: Mapping[str, int]
) -> list[tuple[list[int], str, str]]:
    """``(cluster, lead_side, lead_player)`` per ρ-graph cluster over ``order``.

    ``order`` is the game's strong legs, strongest edge first. The Over-led and
    Under-led clusters grow first from their ``seeds`` (each side's seed is kept
    out of the other's pool so both stories can exist) and are returned even as
    singletons: the caller lends a lead that prices nothing a partner leg. The
    greedy pass then seeds the rest on the strongest remaining leg, tagged
    ``("", "")``, where an isolated strong leg forms no story. A cluster attaches
    any leg correlated with a member.
    """
    players = {i: sctx.bet_df[i]["Player"] for i in order}
    remaining = set(order)
    clusters = []
    for side in LEAD_SIDES:
        if side in seeds:
            pool = remaining - set(seeds.values())
            cluster = _grow_cluster(seeds[side], pool, order, sctx.g.C, players)
            remaining -= set(cluster)
            clusters.append((sorted(cluster), side, players[seeds[side]]))
    while remaining:
        seed = next(i for i in order if i in remaining)
        cluster = _grow_cluster(seed, remaining, order, sctx.g.C, players)
        if len(cluster) >= 2:
            clusters.append((sorted(cluster), "", ""))
    return clusters


def _lead_partner(
    seed: int,
    cluster: Sequence[int],
    order: Sequence[int],
    sctx: GameScoringContext,
    require_both: bool,
) -> int | None:
    """The top-edge strong leg a lead borrows when its cluster prices nothing around it.

    From the other team when the game has strong legs on both (the preset must
    span both sides anyway), else from any other player. ``order`` is every
    strong leg by edge, so a leg another story already uses may serve twice.
    The pair may be uncorrelated: the dek's cluster clause stays quiet below
    its ρ floor, so the card never claims a bundle that isn't there.
    """
    field = "Team" if require_both else "Player"
    own = sctx.bet_df[seed][field]
    return next((j for j in order if j not in cluster and sctx.bet_df[j][field] != own), None)


def _grow_cluster(
    seed: int,
    remaining: set[int],
    order: Sequence[int],
    corr: np.ndarray,
    players: Mapping[int, str],
) -> list[int]:
    """Attach legs correlated (|ρ| ≥ floor) with any member, strongest first, up to the cap.

    One leg per player: a player's strongest leg claims their slot and their
    remaining legs stay in the pool for a later cluster. Same-player legs
    correlate near 1, so without the rule one hot player's five markets fill the
    cap and ``validate_parlay_legs`` (distinct players) finds no valid subset.
    """
    cluster = [seed]
    claimed = {players[seed]}
    remaining.discard(seed)
    changed = True
    while changed and len(cluster) < _MAX_CLUSTER_LEGS:
        changed = False
        for j in order:
            if j not in remaining or players[j] in claimed:
                continue
            if any(abs(corr[j, m]) >= _CLUSTER_RHO_FLOOR for m in cluster):
                cluster.append(j)
                claimed.add(players[j])
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
    must_include: int | None = None,
) -> tuple[dict | None, dict | None]:
    """The (Builder, Moon) parlays for one cluster, or (None, None) if degenerate.

    Only **valid** parlays are enumerated, so the Builder/Moon picks are valid by
    construction. With ``require_both_teams`` (a two-team game) a one-team cluster
    yields ``(None, None)``; a one-sided game relaxes that, so its single-team
    cluster still produces a preset. ``must_include`` (a led cluster's lead)
    restricts the enumeration to subsets carrying that leg, so both presets stay
    true argmaxes and the lead sits in their shared core by construction. Phase 1
    ranks every candidate by a pure-numpy independent-joint proxy; phase 2
    exact-scores only the shortlist (top-K by each objective ∪ all cheap Power
    subsets) through the copula scorer.
    """
    proxies = [
        (combo, *independent(combo, sctx))
        for size in range(2, min(len(cluster), sctx.max_size) + 1)
        for combo in combinations(cluster, size)
        if (must_include is None or must_include in combo)
        and validate_parlay_legs(
            [sctx.bet_df[i] for i in combo], require_both_teams=require_both_teams
        )[0]
    ]
    if not proxies:
        return None, None
    scored = [score_subset(bet_id, sctx, new_map) for bet_id in shortlist(proxies)]
    builder = min(scored, key=lambda s: (-s["G"], s["bet_size"], s["bet_id"]))
    moon = min(scored, key=lambda s: (-s["model_ev"], -s["bet_size"], s["bet_id"]))
    if moon["model_ev"] <= _MENU_MIN_MOON_EV:
        return None, None
    return builder, moon


def _price_cluster(
    cluster: list[int],
    side: str,
    lead: str,
    seed: int | None,
    order: Sequence[int],
    sctx: GameScoringContext,
    new_map: dict,
    *,
    require_both: bool,
) -> tuple[list[int], dict, dict, str, str] | None:
    """One cluster's (Builder, Moon) pair tagged with its lead, or ``None`` if neither prices.

    A lead whose cluster prices nothing around it (alone, or one-team in a
    two-team game) borrows a leg from ``order`` and is priced once more.
    """
    builder, moon = _best_subsets(
        cluster, sctx, new_map, require_both_teams=require_both, must_include=seed
    )
    if builder is None and seed is not None:
        partner = _lead_partner(seed, cluster, order, sctx, require_both)
        if partner is not None:
            cluster = [*cluster, partner]
            builder, moon = _best_subsets(
                cluster, sctx, new_map, require_both_teams=require_both, must_include=seed
            )
    if builder is None:
        return None
    return cluster, builder, moon, side, lead


def _row(sctx: GameScoringContext, objective: str, sub: Mapping, story: Mapping) -> dict:
    """One preset row: the subset's numbers under the story's shared prose and lead fields."""
    return {
        "platform": sctx.platform,
        "League": sctx.league,
        "Game": sctx.game,
        "objective": objective,
        "legs": sub["legs"],
        "joint_p": sub["win_prob"],
        "model_ev": sub["model_ev"],
        "kelly_stake": sub["kelly_stake"],
        "bet_size": sub["bet_size"],
        "Date": sctx.date,
        **story,
    }


def _story_prose(
    builder: Mapping,
    moon: Mapping,
    sctx: GameScoringContext,
    offers: pd.DataFrame,
    index: Mapping[tuple, Mapping],
    ctxs: Mapping[str, GameCtx],
    new_map: dict,
    seen: set[str],
    lead_player: str,
) -> tuple[str, str]:
    """One (headline, dek) per story — the mode chips swap legs, never the prose.

    The headline renders from the legs the two presets share, so a player it
    names is on whichever preset loads; disjoint presets (rare) render from
    the union and blank rather than name a player only one side carries. A
    led story's headline is routed to ``lead_player`` (``""`` for an unled
    one). ``index`` is the game's ``offer_index`` for the dek; ``seen`` dedupes
    headlines across the (platform, date) slate.
    """
    per_sub = [
        [lower_leg(sctx.bet_df[i], new_map) for i in sub["bet_id"]] for sub in (builder, moon)
    ]
    core = sorted(set(builder["bet_id"]) & set(moon["bet_id"]))
    parsed = [lower_leg(sctx.bet_df[i], new_map) for i in core] if core else per_sub[0] + per_sub[1]
    variants, vi, subject = thesis_variants(
        enrich_legs(parsed, offers), ctxs, lead=lead_player or None
    )
    dek = story_dek(core, sctx, index)
    named = subject.get("p")
    if named and any(named not in {leg["player"] for leg in legs} for legs in per_sub):
        return "", dek
    headline = next_unique_variant(variants, vi, seen) if variants else ""
    if headline:
        seen.add(headline)
    return headline, dek
