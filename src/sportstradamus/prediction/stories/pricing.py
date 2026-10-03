"""Subset pricing for the story menu: the exact copula score and its cheap proxy.

``_score_subset`` prices one leg-subset through ``joint.parlay_payout_prob`` (the
parlay search's gate-free scorer) while ``_independent`` and ``_shortlist`` rank
every subset by an independent-joint proxy so only a few finalists pay for the
exact score.
"""

from __future__ import annotations

import math
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

# Exact-scored flex finalists per objective per cluster (Power subsets are all
# exact-scored cheaply via the analytical mvn.cdf, so they bypass the shortlist).
_SHORTLIST_K: int = 8


def _shortlist(proxies: Sequence[tuple]) -> list[tuple[int, ...]]:
    """Distinct subsets worth an exact score: top-K by EV ∪ top-K by G ∪ all Power."""
    by_ev = sorted(proxies, key=lambda x: -x[1])[:_SHORTLIST_K]
    by_g = sorted(proxies, key=lambda x: -x[2])[:_SHORTLIST_K]
    power = [p for p in proxies if len(p[0]) <= POWER_MAX_SIZE]
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
    sig = psd_or_none(g.C[np.ix_(bet_id, bet_id)])
    model_ev = float(
        parlay_payout_prob(
            g.p_model[arr],
            g.p_push[arr],
            sig,
            size,
            boost,
            payout,
            sctx.full_payouts,
            sctx.payout_base_by_size[size],
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
                    "Stat": resolve_leg_stat(sctx.bet_df[i]["Market"], new_map),
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
    payout = float(np.clip(boost * sctx.payout_base_by_size[size], PAYOUT_CLIP_LO, PAYOUT_CLIP_HI))
    return boost, payout


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
