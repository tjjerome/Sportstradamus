"""Subset pricing for the story menu: the exact copula score and its cheap proxy.

Every price here is the value of a unit staked on a leg-subset: the copula joint
of its legs times the payout, through ``joint.parlay_payout_prob`` (the parlay
search's gate-free scorer). ``score_subset`` prices on the fitted win chances
(``GameArrays.p_fit``); ``copula_ev`` prices on any chances, which is how the menu
reads the model's claim (``p_model``) for the presets it keeps. ``shortlist``
bounds the exact scoring: every Power subset is cheap (the analytical
``mvn.cdf``), and the larger ones are ranked by an independent-joint proxy so
only the top few pay for the flex Monte-Carlo.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sportstradamus.leg_schema import build_leg
from sportstradamus.prediction.joint import parlay_payout_prob, psd_or_none
from sportstradamus.prediction.parlay import GameScoringContext, resolve_leg_stat
from sportstradamus.prediction.payouts import (
    PAYOUT_CLIP_HI,
    PAYOUT_CLIP_LO,
    POWER_MAX_SIZE,
)

# Exact-scored larger (flex) finalists per cluster; Power subsets are all
# exact-scored cheaply via the analytical mvn.cdf, so they bypass the cut.
_SHORTLIST_K: int = 8


def shortlist(combos: Sequence[tuple[int, ...]], sctx: GameScoringContext) -> list[tuple[int, ...]]:
    """Subsets worth an exact score: every Power subset, then the top-K larger ones by proxy."""
    power = [c for c in combos if len(c) <= POWER_MAX_SIZE]
    larger = sorted(
        (c for c in combos if len(c) > POWER_MAX_SIZE), key=lambda c: -_independent(c, sctx)
    )
    return power + larger[:_SHORTLIST_K]


def _independent(bet_id: Sequence[int], sctx: GameScoringContext) -> float:
    """Cheap proxy for the fitted value: independent all-hit chance times the payout."""
    _pair_modifier, payout = _modifier_payout(bet_id, sctx)
    return float(np.prod(sctx.g.p_fit[np.asarray(bet_id)])) * payout


def copula_ev(bet_id: Sequence[int], sctx: GameScoringContext, p: np.ndarray) -> float:
    """Value of a unit staked on ``bet_id`` when each leg wins with its chance in ``p``."""
    size = len(bet_id)
    g = sctx.g
    arr = np.asarray(bet_id)
    pair_modifier, payout = _modifier_payout(bet_id, sctx)
    return float(
        parlay_payout_prob(
            p[arr],
            g.p_push[arr],
            psd_or_none(g.C[np.ix_(bet_id, bet_id)]),
            size,
            g.boosts[arr],
            payout,
            sctx.full_payouts,
            sctx.payout_base_by_size[size],
            pair_modifier=pair_modifier,
        )
    )


def score_subset(bet_id: Sequence[int], sctx: GameScoringContext, stat_by_label: dict) -> dict:
    """Exact copula score for one subset on the fitted chances, with its canonical legs.

    ``ev`` is the fitted value of a unit staked and ``win_prob`` that value over
    the payout.
    """
    ev = copula_ev(bet_id, sctx, sctx.g.p_fit)
    _pair_modifier, payout = _modifier_payout(bet_id, sctx)
    return {
        "bet_id": tuple(bet_id),
        "bet_size": len(bet_id),
        "ev": ev,
        "win_prob": ev / payout,
        "legs": [
            build_leg(
                {
                    **sctx.bet_df[i],
                    "League": sctx.league,
                    "Game": sctx.game,
                    "Date": sctx.date,
                    "Platform": sctx.platform,
                    "Stat": resolve_leg_stat(sctx.bet_df[i]["Market"], stat_by_label),
                }
            )
            for i in bet_id
        ],
    }


def _modifier_payout(bet_id: Sequence[int], sctx: GameScoringContext) -> tuple[float, float]:
    """Pair-modifier product and clipped payout multiplier for a subset (no admissibility gate)."""
    size = len(bet_id)
    g = sctx.g
    pair_modifier = float(np.prod(g.M[np.ix_(bet_id, bet_id)][np.triu_indices(size, 1)]))
    boost = pair_modifier * np.prod(g.boosts[np.asarray(bet_id)])
    payout = float(np.clip(boost * sctx.payout_base_by_size[size], PAYOUT_CLIP_LO, PAYOUT_CLIP_HI))
    return pair_modifier, payout
