"""Live slip scoring for the dashboard builders — the one sanctioned live calc.

``score_slip`` prices an arbitrary ≤6-leg user slip by reusing the prediction
layer's Gaussian-copula scorer (``parlay._parlay_payout_prob``) over a
**block-diagonal** correlation matrix: within-game leg pairs take their ρ from
the ``current_game_corr`` slice, cross-game pairs are independent (ρ=0). This is
exactly the "pure numpy/scipy on ≤6 legs" exception the redesign spec carves out
of the precompute-first rule — both the same-game constellation builder and the
cross-game simple builder call it (a cross-game slip just makes the off-blocks
zero, so the joint collapses to the independent product).

Per-leg win/push probabilities and boosts are snapshotted from ``current_offers``
into the leg dicts, so scoring never re-reads the offers frame. Platform pricing
splits: Underdog uses the pooled Power/Flex payout table, Sleeper multiplies the
per-leg boosts (its correlation-discount factor is deferred — see ``DESIGN``/lane
brief — and surfaced as ``payout_approximate``). Money is ``Decimal``.

``slip_headline`` reuses the P2 thesis engine so the constellation builder's live
headline is a deterministic, path-independent function of the leg-set.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm

from sportstradamus.dashboard.legs import corr_key
from sportstradamus.prediction.parlay import (
    _PAYOUT_CLIP_HI,
    _PAYOUT_CLIP_LO,
    _POWER_MAX_SIZE,
    _parlay_payout_prob,
    _payout_curve_for,
    _psd_or_none,
)
from sportstradamus.prediction.stories.engine import thesis_variants
from sportstradamus.prediction.stories.legs import enrich_legs
from sportstradamus.strategies.kelly import fractional_kelly_stake

# norm.ppf blows up at the open-interval ends; clip win probs just inside.
_PROB_EPS: float = 1e-6

# Astrolabe crown reference maxima — fixed shared axis scale, spec §4.4c.
_CROWN_WIN: float = 0.30
_CROWN_EV: float = 0.12
_CROWN_KELLY: float = 0.03


@dataclass(frozen=True)
class SlipScore:
    """Live pricing of one user slip (≤6 legs)."""

    indep_p: float  # independent all-hit joint, ∏ per-leg p
    joint_p: float  # correlation-adjusted all-hit joint (Gaussian copula)
    payout: float  # payout multiplier per $1 staked
    model_ev: float  # expected payout per $1 (push/flex-aware)
    bet_size: int
    play_type: str  # Power | Flex | Sleeper
    stake: Decimal  # fractional-Kelly stake in dollars
    payout_approximate: bool  # Sleeper: correlation-discount factor pending


def score_slip(
    legs: Sequence[Mapping],
    corr: pd.DataFrame,
    *,
    platform: str,
    bankroll: Decimal,
    shrinkage: float = 1.0,
) -> SlipScore:
    """Price a user slip by reusing the parlay copula scorer.

    ``legs`` are canonical structured legs (``sportstradamus.leg_schema.LEG_FIELDS``);
    ``corr`` is the ``current_game_corr`` slice.
    """
    n = len(legs)
    p = np.clip(np.array([float(leg["win_prob"]) for leg in legs]), _PROB_EPS, 1 - _PROB_EPS)
    indep_p = float(np.prod(p))
    play_type, full_payouts, base, payout_approximate = _platform_pricing(platform, n)
    if n < 2 or base <= 0.0:
        return SlipScore(indep_p, indep_p, 0.0, 0.0, n, play_type, Decimal("0"), payout_approximate)

    sig = _psd_or_none(_block_diagonal_sig(legs, corr), legacy=False)
    joint_p = float(multivariate_normal.cdf(norm.ppf(p), np.zeros(n), sig))
    boost = float(np.prod([float(leg["boost"]) for leg in legs]))
    payout = float(np.clip(boost * base, _PAYOUT_CLIP_LO, _PAYOUT_CLIP_HI))
    push = np.array([float(leg.get("push_prob", 0.0) or 0.0) for leg in legs])
    model_ev = float(_parlay_payout_prob(p, push, sig, n, boost, payout, full_payouts, base, False))
    kelly_win = model_ev / payout if payout > 0 else 0.0
    stake = fractional_kelly_stake(
        bankroll=bankroll,
        win_prob=kelly_win,
        payout_multiplier=Decimal(repr(payout)),
        model_shrinkage=shrinkage,
    )
    return SlipScore(indep_p, joint_p, payout, model_ev, n, play_type, stake, payout_approximate)


def ev_lift(
    focus: Mapping,
    candidate: Mapping,
    corr: pd.DataFrame,
    *,
    platform: str,
    bankroll: float = 0.0,
    shrinkage: float = 1.0,
) -> float:
    """EV of {focus + candidate} minus focus alone (spec §4.1 Correlated tab).

    Both scored through the same copula path as the slip rail; ~milliseconds
    per candidate, so the Details tab computes live (no precompute).
    """
    pair = score_slip(
        [focus, candidate], corr, platform=platform, bankroll=bankroll, shrinkage=shrinkage
    )
    solo = score_slip([focus], corr, platform=platform, bankroll=bankroll, shrinkage=shrinkage)
    return pair.model_ev - solo.model_ev


def astrolabe_payload(score: SlipScore, *, nonce: int) -> dict:
    """JSON contract for the astrolabe component. Crowns are the fixed shared
    reference maxima (spec §4.4c): Win 30% / EV +12% / Kelly 3%.
    """
    kelly = (score.model_ev - 1) / (score.payout - 1) if score.payout > 1 else 0.0
    return {
        "legs": score.bet_size,
        "play_type": score.play_type,
        "payout": score.payout,
        "payout_approximate": score.payout_approximate,
        "win_corr": score.joint_p,
        "win_indep": score.indep_p,
        "ev": score.model_ev - 1,
        "kelly": max(kelly, 0.0),
        "crowns": {"win": _CROWN_WIN, "ev": _CROWN_EV, "kelly": _CROWN_KELLY},
        "nonce": nonce,
    }


def _platform_pricing(platform: str, n: int) -> tuple[str, dict, float, bool]:
    """(play_type, full payout curve, base multiplier, payout_approximate) for a platform/size.

    Underdog reads the pooled Power/Flex schedule (base = the per-size multiplier
    before the boost product). Sleeper has no per-size table — its payout is the
    boost product alone, so the base schedule is a flat 1.0 and the boost carries
    the multiplier (the correlation-discount factor is deferred).
    """
    if platform == "Sleeper":
        return "Sleeper", {n: [1.0, 0.0]}, 1.0, True
    search, full_curve = _payout_curve_for("Underdog", "pooled", legacy=False)
    base = search[n - 2] if 0 <= n - 2 < len(search) else 0.0
    play_type = "Power" if n <= _POWER_MAX_SIZE else "Flex"
    return play_type, full_curve, float(base), False


def _block_diagonal_sig(legs: Sequence[Mapping], corr: pd.DataFrame) -> np.ndarray:
    """Correlation matrix: within-game ρ from the slice, cross-game pairs ρ=0."""
    n = len(legs)
    sig = np.eye(n)
    if corr is None or corr.empty:
        return sig
    keys = [corr_key(leg) for leg in legs]
    games = [str(leg.get("game", "")) for leg in legs]
    rho_by_game = _rho_lookup(corr, set(games))
    for i in range(n):
        for j in range(i + 1, n):
            if games[i] and games[i] == games[j]:
                sig[i, j] = sig[j, i] = rho_by_game.get(
                    (games[i], frozenset((keys[i], keys[j]))), 0.0
                )
    return sig


def _rho_lookup(corr: pd.DataFrame, games: set[str]) -> dict:
    """Map ``(game, frozenset(leg_a, leg_b)) -> rho`` for the slip's games."""
    slice_df = corr.loc[corr["Game"].isin(games)]
    return {
        (str(row.Game), frozenset((row.leg_a, row.leg_b))): float(row.rho)
        for row in slice_df.itertuples(index=False)
    }


def slip_headline(legs: Sequence[Mapping], offers: pd.DataFrame, ctxs: Mapping) -> str:
    """Deterministic thesis headline for the slip's legs (path-independent).

    Reuses the P2 thesis engine; the variant is md5-seeded on the canonical
    leg-set, so identical legs always yield the identical string regardless of
    edit history. ``ctxs`` is the caller's ``ctxs_from_frame`` mapping.
    """
    if not legs:
        return ""
    # Canonicalize leg order so the headline is a pure function of the leg-set:
    # the engine's md5 seed already sorts, but routing tie-breaks can read order.
    ordered = sorted(legs, key=lambda leg: (leg["player"], leg["market"], leg["bet"], leg["line"]))
    variants, vi, _ = thesis_variants(enrich_legs(ordered, offers), ctxs)
    return variants[vi] if variants else ""
