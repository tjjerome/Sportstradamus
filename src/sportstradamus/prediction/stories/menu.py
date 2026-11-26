"""Per-game story menu: correlation-cluster stories with two starting parlays.

``build_game_stories`` turns each game's scoring bundle (``GameScoringContext``,
captured by ``find_correlation``'s ``story_sink``) into up to five data-driven
*stories* — correlation clusters of the game's story legs, which are its pool legs
(``leg_indices``) ordered by ``EV``, with no floor. Each story emits two starting
parlays over its cluster's legs (legs may be shared), both picked by ``ev``, the
fitted value of a unit staked (the copula joint of the legs' fitted win chances
times the payout):

* **Bankroll Builder** — the Power-sized subset (two legs up to
  ``payouts.POWER_MAX_SIZE``) with the highest ``ev``; a tie goes to the smaller.
* **Shoot the Moon** — the larger subset, up to the play-type cap, with the
  highest ``ev``; a tie goes to the larger. A cluster with no valid larger subset
  repeats the Builder.

The picks rank rather than clear a bar, so every cluster with a valid parlay is a
story whatever its value, and the menu orders its stories by the Builder's
``ev``. Each row also prices the same entry on the model's claimed chances
(``model_ev``) and stakes nothing (``kelly_stake`` is 0).

Ranked on value alone a menu opens on whichever story values highest, star or
not, so two stories per game are *led*: one grows from the most prominent Over
(only in a shootout or a blowout, ``lead.OVER_LEAD_SHAPES``) and one from the most
prominent Under (``lead.lead_seeds``), the lead sits in both presets, led stories
rank first, and ``lead.assign_leads`` flags the one each menu headlines.

Pure and ``Archive``-free so the P3 dashboard rail can recompute it live. Every
valid subset is enumerated, but only some pay for the real Gaussian-copula scorer
(``joint.parlay_payout_prob``): every Power subset is priced exactly through its
analytical ``mvn.cdf``, while an independent-joint proxy ranks the larger ones
and only the top few run the flex branch's 50k-sample Monte-Carlo
(``pricing.shortlist``). The picks always read the exact score.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np
import pandas as pd

from sportstradamus.prediction.parlay import GameScoringContext, leg_market_map
from sportstradamus.prediction.payouts import POWER_MAX_SIZE
from sportstradamus.prediction.stories.context import GameCtx, ctxs_from_frame
from sportstradamus.prediction.stories.engine import thesis_variants
from sportstradamus.prediction.stories.lead import LEAD_SIDES, assign_leads, lead_seeds
from sportstradamus.prediction.stories.legs import (
    enrich_legs,
    lower_leg,
    offer_index,
    validate_parlay_legs,
)
from sportstradamus.prediction.stories.pricing import copula_ev, score_subset, shortlist
from sportstradamus.prediction.stories.thesis import next_unique_variant
from sportstradamus.prediction.stories.why import story_dek

# Two legs join a cluster when |rho| over their pair clears this; matches the
# display-correlation floor in correlation.py.
_CLUSTER_RHO_FLOOR: float = 0.05
# Brute-force bound: 2**8 = 256 subsets keeps a cluster's enumeration instant.
_MAX_CLUSTER_LEGS: int = 8
# Owner-locked menu cap: at most five stories per (platform, game).
_MAX_STORIES: int = 5

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
    "ev",
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
    ``assign_leads`` breaks its star ties by tip: the first non-empty
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
    ev = _leg_evs(sctx)
    if len(ev) < 2:
        return []
    # Whole-game gate: only when the game's story legs all sit on one team may a story
    # be a single-team preset (the user completes it with a satellite leg). A game
    # with legs on both teams still requires both-teams presets.
    require_both = len({team for i in ev if (team := sctx.bet_df[i].get("Team"))}) >= 2
    stat_by_label = leg_market_map(sctx.league, sctx.platform)
    index = offer_index(offers)
    stars = {}
    for i in ev:
        rec = sctx.bet_df[i]
        stars[i] = index.get((rec["Player"], rec["Bet"], rec["Line"]), {}).get("Star", 0.0)
    ctx = ctxs.get(sctx.game)
    seeds = lead_seeds(ev, sctx.bet_df, stars, ctx.shape if ctx else "even")
    order = sorted(ev, key=lambda i: -ev[i])
    scored = []
    for cluster, side, lead in _cluster_legs(order, sctx, seeds):
        priced = _price_cluster(
            cluster, seeds.get(side), order, sctx, stat_by_label, require_both=require_both
        )
        if priced is not None:
            scored.append((*priced, side, lead))
    # Led stories first, then each group by the Builder's fitted value.
    scored.sort(key=lambda s: (s[3] == "", -s[1]["ev"], -max(ev[i] for i in s[0]), s[1]["bet_id"]))
    rows: list[dict] = []
    for rank, (_cluster, builder, moon, side, lead) in enumerate(scored[:_MAX_STORIES]):
        headline, dek = _story_prose(
            builder, moon, sctx, offers, index, ctxs, stat_by_label, seen, lead
        )
        story = {
            "story_id": f"{sctx.game}#{rank}",
            "headline": headline,
            "dek": dek,
            "lead_side": side,
            "lead_player": lead,
            "star": max(stars[i] for i in set(builder["bet_id"]) | set(moon["bet_id"])),
        }
        # Each distinct preset is priced once: the copula scorer is noisy, so a Moon
        # that repeats the Builder would otherwise print two claims for one entry.
        claimed = {
            bet_id: copula_ev(bet_id, sctx, sctx.g.p_model)
            for bet_id in {builder["bet_id"], moon["bet_id"]}
        }
        rows.append(_row(sctx, "builder", builder, claimed[builder["bet_id"]], story))
        rows.append(_row(sctx, "moon", moon, claimed[moon["bet_id"]], story))
    return rows


def _leg_evs(sctx: GameScoringContext) -> dict[int, float]:
    """The game's story legs mapped to their ``EV``: every pool leg, with no floor.

    The pool (``leg_indices``) holds the game's priced legs, so each carries an
    ``EV`` (the fitted win chance times the side's payout). On the live board no
    leg reaches 1, so a floor on it would leave every menu empty.
    """
    return {i: sctx.bet_df[i]["EV"] for i in sctx.leg_indices}


def _cluster_legs(
    order: Sequence[int], sctx: GameScoringContext, seeds: Mapping[str, int]
) -> list[tuple[list[int], str, str]]:
    """``(cluster, lead_side, lead_player)`` per ρ-graph cluster over ``order``.

    ``order`` is the game's story legs, highest ``EV`` first. The Over-led and
    Under-led clusters grow first from their ``seeds`` (each side's seed is kept
    out of the other's pool so both stories can exist) and are returned even as
    singletons: the caller lends a lead that prices nothing a partner leg. The
    greedy pass then seeds the rest on the highest-EV remaining leg, tagged
    ``("", "")``, where an isolated leg forms no story. A cluster attaches any
    leg correlated with a member.
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
    """The top-EV leg a lead borrows when its cluster prices nothing around it.

    From the other team when the game has story legs on both (the preset must
    span both sides anyway), else from any other player. ``order`` is every
    story leg by ``EV``, so a leg another story already uses may serve twice.
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
    """Attach legs correlated (|ρ| ≥ floor) with any member, highest EV first, up to the cap.

    One leg per player: a player's highest-EV leg claims their slot and their
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
    stat_by_label: dict,
    *,
    require_both_teams: bool = True,
    must_include: int | None = None,
) -> tuple[dict | None, dict | None]:
    """The (Builder, Moon) parlays for one cluster, or (None, None) if no valid parlay.

    Only **valid** parlays are enumerated, so the Builder/Moon picks are valid by
    construction. With ``require_both_teams`` (a two-team game) a one-team cluster
    yields ``(None, None)``; a one-sided game relaxes that, so its single-team
    cluster still produces a preset. ``must_include`` (a led cluster's lead)
    restricts the enumeration to subsets carrying that leg, so the lead sits in
    both presets by construction. Both rank on ``ev`` whatever its level: the
    Builder is the best Power subset (a tie goes to the smaller subset, then the
    lower ``bet_id``) and the Moon the best larger one (a tie goes to the larger),
    else the Builder's own. Any valid subset holds a valid Power one, so the
    Builder always exists. ``pricing.shortlist`` bounds the exact scoring.
    """
    combos = [
        combo
        for size in range(2, min(len(cluster), sctx.max_size) + 1)
        for combo in combinations(cluster, size)
        if (must_include is None or must_include in combo)
        and validate_parlay_legs(
            [sctx.bet_df[i] for i in combo], require_both_teams=require_both_teams
        )[0]
    ]
    if not combos:
        return None, None
    scored = [score_subset(bet_id, sctx, stat_by_label) for bet_id in shortlist(combos, sctx)]
    builder = min(
        (s for s in scored if s["bet_size"] <= POWER_MAX_SIZE),
        key=lambda s: (-s["ev"], s["bet_size"], s["bet_id"]),
    )
    moon = min(
        (s for s in scored if s["bet_size"] > POWER_MAX_SIZE),
        key=lambda s: (-s["ev"], -s["bet_size"], s["bet_id"]),
        default=builder,
    )
    return builder, moon


def _price_cluster(
    cluster: list[int],
    seed: int | None,
    order: Sequence[int],
    sctx: GameScoringContext,
    stat_by_label: dict,
    *,
    require_both: bool,
) -> tuple[list[int], dict, dict] | None:
    """One cluster's (Builder, Moon) pair and the cluster priced, or ``None`` if neither prices.

    A lead whose cluster prices nothing around it (alone, or one-team in a
    two-team game) borrows a leg from ``order`` and is priced once more, so the
    returned cluster then carries that leg.
    """
    builder, moon = _best_subsets(
        cluster, sctx, stat_by_label, require_both_teams=require_both, must_include=seed
    )
    if builder is None and seed is not None:
        partner = _lead_partner(seed, cluster, order, sctx, require_both)
        if partner is not None:
            cluster = [*cluster, partner]
            builder, moon = _best_subsets(
                cluster, sctx, stat_by_label, require_both_teams=require_both, must_include=seed
            )
    if builder is None:
        return None
    return cluster, builder, moon


def _row(
    sctx: GameScoringContext, objective: str, sub: Mapping, model_ev: float, story: Mapping
) -> dict:
    """One preset row: the subset's numbers under the story's shared prose and lead fields.

    ``joint_p`` and ``ev`` are the entry's fitted chance and value; ``model_ev`` is
    the value the model claims for it.
    """
    return {
        "platform": sctx.platform,
        "League": sctx.league,
        "Game": sctx.game,
        "objective": objective,
        "legs": sub["legs"],
        "joint_p": sub["win_prob"],
        "ev": sub["ev"],
        "model_ev": model_ev,
        # An entry gets no stake until the entries' money record supports one
        # (docs/handoffs/profit-engine.md §5.3).
        "kelly_stake": 0.0,
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
    stat_by_label: dict,
    seen: set[str],
    lead_player: str,
) -> tuple[str, str]:
    """One (headline, dek) per story — the mode chips swap legs, never the prose.

    The headline renders from the legs the two presets share, so a player it
    names is on whichever preset loads; disjoint presets (rare) render from
    the union and blank rather than name a player only one side carries. A
    led story's headline and dek are routed to ``lead_player`` (``""`` for an
    unled one). ``index`` is the game's ``offer_index`` for the dek; ``seen``
    dedupes headlines across the (platform, date) slate.
    """
    per_sub = [
        [lower_leg(sctx.bet_df[i], stat_by_label) for i in sub["bet_id"]] for sub in (builder, moon)
    ]
    core = sorted(set(builder["bet_id"]) & set(moon["bet_id"]))
    parsed = (
        [lower_leg(sctx.bet_df[i], stat_by_label) for i in core]
        if core
        else per_sub[0] + per_sub[1]
    )
    variants, vi, subject = thesis_variants(
        enrich_legs(parsed, offers), ctxs, lead=lead_player or None
    )
    dek = story_dek(core, sctx, index, lead_player)
    named = subject.get("p")
    if named and any(named not in {leg["player"] for leg in legs} for legs in per_sub):
        return "", dek
    headline = next_unique_variant(variants, vi, seen) if variants else ""
    if headline:
        seen.add(headline)
    return headline, dek
