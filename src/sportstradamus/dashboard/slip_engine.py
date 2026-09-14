"""Live slip scoring for the dashboard builders — the one sanctioned live calc.

``score_slip`` prices an arbitrary ≤6-leg user slip by reusing the prediction
layer's Gaussian-copula scorer (``joint.parlay_payout_prob``) over a
**block-diagonal** correlation matrix: within-game leg pairs take their ρ from
the game's :class:`GameCtx` (its ``current_game_corr`` slice, indexed by pair),
cross-game pairs are independent (ρ=0). This is
exactly the "pure numpy/scipy on ≤6 legs" exception the redesign spec carves out
of the precompute-first rule — both the same-game constellation builder and the
cross-game simple builder call it (a cross-game slip just makes the off-blocks
zero, so the joint collapses to the independent product).

Per-leg win/push probabilities and boosts are snapshotted from ``current_offers``
into the leg dicts, so scoring never re-reads the offers frame. Platform pricing:
both platforms read a real pooled payout schedule (Underdog Power/Flex, Sleeper
Max/Flex) that the boost product multiplies on top of. That product also carries
the platform's same-game pair modifiers from ``current_pair_modifiers`` — the same
leg-boosts × pair-modifiers product prophecize's story menu prices with — so the
live slip prices the way the app quotes it. A 0.0 modifier means the app refuses
the pair, and a slip holding one scores zero. Both pair lookups are dicts built once
per snapshot, so a price costs the copula, not a scan of the slate. Money is
``Decimal``.

``slip_headline`` reuses the P2 thesis engine so the constellation builder's live
headline is a deterministic, path-independent function of the leg-set.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm

from sportstradamus.dashboard.legs import corr_key
from sportstradamus.leg_schema import leg_label
from sportstradamus.prediction.joint import parlay_payout_prob, psd_or_none
from sportstradamus.prediction.payouts import (
    PAYOUT_CLIP_HI,
    PAYOUT_CLIP_LO,
    POWER_MAX_SIZE,
    SLEEPER_FULL_REFUND_MAX_SIZE,
    SLEEPER_MAX_SIZE,
    payout_curve_for,
)
from sportstradamus.prediction.stories.context import GameCtx
from sportstradamus.prediction.stories.engine import thesis_variants
from sportstradamus.prediction.stories.legs import enrich_legs
from sportstradamus.strategies.kelly import fractional_kelly_stake

# norm.ppf blows up at the open-interval ends; clip win probs just inside.
_PROB_EPS: float = 1e-6

# Astrolabe crown reference maxima — fixed shared axis scale, spec §4.4c.
_CROWN_WIN: float = 0.30
_CROWN_EV: float = 0.12
_CROWN_KELLY: float = 0.03

# Game -> frozenset of a leg pair's two ``corr_key`` keys -> that pair's ρ or payout modifier.
PairValues = Mapping[str, Mapping[frozenset, float]]

# current_pair_modifiers sentinel: this value means the platform refuses the pair
# outright, not just downweights it.
REFUSED_MODIFIER: float = 0.0


@dataclass(frozen=True)
class SlipScore:
    """Live pricing of one user slip (≤6 legs)."""

    indep_p: float  # independent all-hit joint, ∏ per-leg p
    joint_p: float  # correlation-adjusted all-hit joint (Gaussian copula)
    payout: float  # payout multiplier per $1 staked
    model_ev: float  # expected payout per $1 (push/flex-aware)
    bet_size: int
    play_type: str  # Power | Max | Flex
    stake: Decimal  # fractional-Kelly stake in dollars
    payout_approximate: bool  # always False; kept for the astrolabe JSON contract
    pair_mods: tuple[tuple[int, int, float], ...] = ()  # (i, j, modifier) per pair on record

    @property
    def banned(self) -> bool:
        """Whether the platform refuses a pair on the slip (a 0.0 modifier)."""
        return any(modifier == REFUSED_MODIFIER for _, _, modifier in self.pair_mods)


def score_slip(
    legs: Sequence[Mapping],
    ctxs: Mapping[str, GameCtx],
    mods: Mapping[str, PairValues],
    *,
    platform: str,
    bankroll: Decimal,
    shrinkage: float = 1.0,
) -> SlipScore:
    """Price a user slip by reusing the parlay copula scorer.

    ``legs`` are canonical structured legs (``sportstradamus.leg_schema.LEG_FIELDS``);
    ``ctxs`` is the per-game context map, whose ``rho`` the copula reads, and ``mods``
    the per-platform pair modifiers ``modifier_map`` indexes. A slip holding a pair
    ``platform`` refuses returns the zero score before the payout clip, which would
    lift its 0.0 back to 1x.
    """
    n = len(legs)
    p = np.clip(np.array([float(leg["win_prob"]) for leg in legs]), _PROB_EPS, 1 - _PROB_EPS)
    indep_p = float(np.prod(p))
    play_type, full_payouts, base, payout_approximate = _platform_pricing(platform, n)
    pair_mods = tuple(_same_game_pairs(legs, mods.get(platform, {})))
    unpriced = SlipScore(
        indep_p, indep_p, 0.0, 0.0, n, play_type, Decimal("0"), payout_approximate, pair_mods
    )
    if n < 2 or base <= 0.0 or unpriced.banned:
        return unpriced

    sig = psd_or_none(_block_diagonal_sig(legs, ctxs))
    joint_p = float(multivariate_normal.cdf(norm.ppf(p), np.zeros(n), sig))
    boost = float(
        np.prod([float(leg["boost"]) for leg in legs]) * np.prod([m for _, _, m in pair_mods])
    )
    payout = float(np.clip(boost * base, PAYOUT_CLIP_LO, PAYOUT_CLIP_HI))
    push = np.array([float(leg.get("push_prob", 0.0) or 0.0) for leg in legs])
    full_refund_below_size = SLEEPER_FULL_REFUND_MAX_SIZE if platform == "Sleeper" else None
    model_ev = float(
        parlay_payout_prob(
            p,
            push,
            sig,
            n,
            boost,
            payout,
            full_payouts,
            base,
            full_refund_below_size=full_refund_below_size,
        )
    )
    kelly_win = model_ev / payout if payout > 0 else 0.0
    stake = fractional_kelly_stake(
        bankroll=bankroll,
        win_prob=kelly_win,
        payout_multiplier=Decimal(repr(payout)),
        model_shrinkage=shrinkage,
    )
    return SlipScore(
        indep_p, joint_p, payout, model_ev, n, play_type, stake, payout_approximate, pair_mods
    )


def ev_lift(
    focus: Mapping,
    candidate: Mapping,
    ctxs: Mapping[str, GameCtx],
    mods: Mapping[str, PairValues],
    *,
    platform: str,
    bankroll: float = 0.0,
    shrinkage: float = 1.0,
) -> float:
    """EV of {focus + candidate} minus focus alone (spec §4.1 Correlated tab).

    Both scored through the same copula path as the slip rail; ~milliseconds
    per candidate, so the Details tab computes live (no precompute). A candidate
    the platform refuses beside the focus prices at $0, so it lifts nothing.
    """
    pair = score_slip(
        [focus, candidate], ctxs, mods, platform=platform, bankroll=bankroll, shrinkage=shrinkage
    )
    solo = score_slip(
        [focus], ctxs, mods, platform=platform, bankroll=bankroll, shrinkage=shrinkage
    )
    return pair.model_ev - solo.model_ev


def banned_partners(
    legs: Sequence[Mapping], mods: Mapping[str, PairValues], *, platform: str
) -> dict[str, str]:
    """Map each star ``platform`` won't pair with the slip to its card's "won't pair" text.

    Walks every slip leg, other-game satellites included, against the platform's 0.0
    pairs in that leg's own game; the star is the pair's other side. Two slip legs that
    refuse each other both land, and a same-player pair marks a slip player's other
    markets.
    """
    by_game = mods.get(platform, {})
    conflicts: dict[str, list[str]] = defaultdict(list)
    for leg in legs:
        key = corr_key(leg)
        for pair, modifier in by_game.get(leg["game"], {}).items():
            if modifier == REFUSED_MODIFIER and key in pair:
                (star,) = pair - {key}
                conflicts[star].append(leg_label(leg))
    return {
        star: f"{platform} won't pair this with {', '.join(labels)}"
        for star, labels in conflicts.items()
    }


def modifier_map(mods: pd.DataFrame) -> dict[str, dict[str, dict[frozenset, float]]]:
    """Index a ``current_pair_modifiers`` frame as ``Platform`` → ``Game`` → pair → modifier.

    The lookup ``score_slip`` and ``banned_partners`` read. Built once per snapshot, so a
    price never walks the slate's hundred thousand pairs.
    """
    out: dict[str, dict[str, dict[frozenset, float]]] = {}
    for platform, game, leg_a, leg_b, modifier in zip(
        mods["Platform"],
        mods["Game"],
        mods["leg_a"],
        mods["leg_b"],
        mods["modifier"].tolist(),
        strict=True,
    ):
        out.setdefault(platform, {}).setdefault(game, {})[frozenset((leg_a, leg_b))] = modifier
    return out


def astrolabe_payload(score: SlipScore) -> dict:
    """JSON contract for the astrolabe component.

    Crowns are the fixed shared reference maxima (spec §4.4c): Win 30% /
    EV +12% / Kelly 3%.
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
    }


def _platform_pricing(platform: str, n: int) -> tuple[str, dict, float, bool]:
    """(play_type, full payout curve, base multiplier, payout_approximate) for a platform/size.

    Both platforms read a real pooled schedule (Underdog Power/Flex, Sleeper
    Max/Flex, sportstradamus.prediction.payouts) — ``base`` is the per-size
    multiplier ``score_slip``'s ``boost * base`` composition applies on top of.
    Sizes above a platform's configured max (``SLEEPER_FLEX_CAP`` / Underdog's
    payout-table max, both 6 today) silently return ``base=0.0``, which
    ``score_slip``'s early-return guard turns into a zero-payout/zero-EV score
    rather than an error — pre-existing, symmetric across both platforms.
    """
    if platform == "Sleeper":
        search, full_curve = payout_curve_for("Sleeper", "pooled")
        base = search[n - 2] if 0 <= n - 2 < len(search) else 0.0
        play_type = "Max" if n <= SLEEPER_MAX_SIZE else "Flex"
        return play_type, full_curve, float(base), False
    search, full_curve = payout_curve_for("Underdog", "pooled")
    base = search[n - 2] if 0 <= n - 2 < len(search) else 0.0
    play_type = "Power" if n <= POWER_MAX_SIZE else "Flex"
    return play_type, full_curve, float(base), False


def _block_diagonal_sig(legs: Sequence[Mapping], ctxs: Mapping[str, GameCtx]) -> np.ndarray:
    """Correlation matrix: within-game ρ off each game's context, cross-game pairs ρ=0."""
    sig = np.eye(len(legs))
    rho = {game: ctx.rho for game, ctx in ctxs.items()}
    for i, j, value in _same_game_pairs(legs, rho):
        sig[i, j] = sig[j, i] = value
    return sig


def _same_game_pairs(legs: Sequence[Mapping], values: PairValues) -> list[tuple[int, int, float]]:
    """``(i, j, value)`` for each same-game leg pair ``values`` holds.

    A pair is keyed as a ``frozenset``, so it matches whichever order the slip lists
    its two legs in.
    """
    keys = [corr_key(leg) for leg in legs]
    pairs = []
    for i, j in combinations(range(len(legs)), 2):
        if legs[i]["game"] != legs[j]["game"]:
            continue
        value = values.get(legs[i]["game"], {}).get(frozenset((keys[i], keys[j])))
        if value is not None:
            pairs.append((i, j, value))
    return pairs


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
