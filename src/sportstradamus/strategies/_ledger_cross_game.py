"""Cross-game candidate builders for the simulated-bettor ledger.

Builds Underdog- or Sleeper-style Pick'em combinations that span >=2 distinct
games directly from already-scored single-leg offers, under an explicit
independence (rho=0) assumption between legs from different games. The
existing same-game parlay search (``prediction.parlay`` /
``prediction.correlation``) structurally cannot produce these -- it only ever
combines legs sharing one ``Game`` -- so this module is a deliberately
separate, independence-assumed path, not a duplicate of that search.
``build_cross_game_candidates`` fills the shared universe from the legs the
pick'em gate keeps, each read cut by its cell's trust;
``build_even_pick_candidates`` fills the ``even_picks`` persona's pool from the
slate date's Underdog even picks the product recommends, at their served read,
its entries dealt evenly over those legs with a game per leg. One pricer sizes
both. See ``docs/handoffs/sim-bettor-ledger.md`` §10 for the surrounding policy
design.
"""

from __future__ import annotations

import datetime
import hashlib
import math
from dataclasses import dataclass, replace
from decimal import Decimal

import numpy as np
import pandas as pd

from sportstradamus.helpers import archive_market, platform_payout, stat_map
from sportstradamus.leg_schema import build_leg, leg_label
from sportstradamus.prediction.payouts import (
    SLEEPER_FULL_REFUND_MAX_SIZE,
    expected_payout_with_pushes,
    payout_curve_for,
)
from sportstradamus.realized import recommended
from sportstradamus.strategies._ledger_selection import (
    BANKROLL_PER_REPLICATE,
    LedgerCandidate,
    _entropy_from,
)
from sportstradamus.strategies.kelly import fractional_kelly_stake
from sportstradamus.strategies.underdog_pickem import (
    PickemConfig,
    filter_legs,
    resolve_market_shrinkage,
)

_CROSS_GAME_BEAM_WIDTH: int = 200  # v1 default, narrower than the same-game search's beam
_MAX_ENTRY_SIZE: int = 6
# Underdog's raw multiplier on a pick it neither discounts nor boosts: an even pick
# (docs/underdog_api.md §6.8). The even_picks persona takes entries of these alone.
_EVEN_PICK_MULTIPLIER: float = 1.0
# Deals tried per entry size: a small slate has fewer than _CROSS_GAME_BEAM_WIDTH entries to deal.
_EVEN_PICK_DEALS_PER_SIZE: int = 4000

_POOLED_CURVES: dict[str, dict[int, list[float]]] = {
    "Underdog": payout_curve_for("Underdog", "pooled")[1],
    "Sleeper": payout_curve_for("Sleeper", "pooled")[1],
}


@dataclass(frozen=True)
class _ScoredLeg:
    idx: int
    player: str
    game: str
    win_prob: float  # what the leg is priced at: its served read, or that cut by its cell's trust
    push_prob: float
    book_devig: float  # "Market Prob" column passthrough
    line: float
    boost: float  # the platform's raw multiplier for this pick
    display: str  # leg_label(leg) -- canonical rendering, do not reformat it yourself
    canonical_leg: dict  # build_leg() output, with leg["stat"] patched to cell_market
    cell_market: str | None = None  # the model cell's market; None when stat_map lacks the label


def _served_legs(offers: pd.DataFrame, platform: str) -> list[_ScoredLeg]:
    """One leg per row of ``offers``, priced at its served ``Win Prob``."""
    out = []
    for i, row in offers.reset_index(drop=True).iterrows():
        leg = build_leg(row)
        cell_market = stat_map[platform].get(row["Market"])
        if cell_market:
            # stat_map names a market the same for every league, and the NHL gamelog
            # and cells call AST assists. Settlement reads the gamelog by this key.
            cell_market = archive_market(leg["league"], cell_market)
            leg["stat"] = cell_market
        out.append(
            _ScoredLeg(
                idx=i,
                player=leg["player"],
                game=leg["game"],
                win_prob=leg["win_prob"],
                push_prob=leg["push_prob"],
                book_devig=float(row["Market Prob"]),
                line=leg["line"],
                boost=leg["boost"],
                display=leg_label(leg),
                canonical_leg=leg,
                cell_market=cell_market,
            )
        )
    return out


def _score_legs(eligible: pd.DataFrame, platform: str) -> list[_ScoredLeg]:
    """``eligible``'s legs, each read pulled toward a coin flip by its cell's trust."""
    out = []
    for leg in _served_legs(eligible, platform):
        shrinkage, _source = resolve_market_shrinkage(leg.canonical_leg["league"], leg.cell_market)
        adj_p = 0.5 + (leg.win_prob - 0.5) * max(0.0, min(1.0, shrinkage))
        out.append(replace(leg, win_prob=adj_p))
    return out


def _expand(legs: list[_ScoredLeg], beam: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    by_idx = {leg.idx: leg for leg in legs}
    scored: list[tuple[tuple[int, ...], float]] = []
    for combo in beam:
        used_players = {by_idx[i].player for i in combo}
        last_idx = combo[-1]
        for leg in legs:
            if leg.idx <= last_idx or leg.player in used_players:
                continue
            if any(leg.player in p or p in leg.player for p in used_players):
                continue  # guards against "A. Player" vs "A. Player Jr." collisions
            extended = (*combo, leg.idx)
            if len({by_idx[i].game for i in extended}) < 2:
                continue  # single-game combos are the same-game search's job
            heuristic = math.prod(
                by_idx[i].win_prob for i in extended
            )  # cheap beam-pruning rank only
            scored.append((extended, heuristic))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [combo for combo, _ in scored[:_CROSS_GAME_BEAM_WIDTH]]


def _enumerate_cross_game_combos(legs: list[_ScoredLeg]) -> dict[int, list[tuple[int, ...]]]:
    beam = [(leg.idx,) for leg in legs]
    by_size: dict[int, list[tuple[int, ...]]] = {}
    for size in range(2, _MAX_ENTRY_SIZE + 1):
        beam = _expand(legs, beam)
        by_size[size] = beam
    return by_size


def _deal_even_pick_combos(
    legs: list[_ScoredLeg], rng: np.random.Generator
) -> dict[int, list[tuple[int, ...]]]:
    """Deal up to ``_CROSS_GAME_BEAM_WIDTH`` entries per size evenly over ``legs``, a game per leg.

    The shared beam keeps the entries with the highest joint read, which would stake this
    pool on its few strongest legs.
    """
    games = len({leg.game for leg in legs})
    by_size: dict[int, list[tuple[int, ...]]] = {}
    for size in range(2, min(_MAX_ENTRY_SIZE, games) + 1):
        combos: set[tuple[int, ...]] = set()
        for _ in range(_EVEN_PICK_DEALS_PER_SIZE):
            if len(combos) == _CROSS_GAME_BEAM_WIDTH:
                break
            picked = sorted(rng.choice(len(legs), size=size, replace=False).tolist())
            # A game per leg: Underdog taxes some same-game pairs and a cross-game record
            # carries pair_modifier 1.0, so only then is the table payout what the entry pays.
            # It is a player per leg too: the pool is one day's legs, and a player has one
            # game in a day.
            if len({legs[i].game for i in picked}) == size:
                combos.add(tuple(legs[i].idx for i in picked))
        by_size[size] = sorted(combos)
    return by_size


def _pricing_rng(date: datetime.date, run_slot: str) -> np.random.Generator:
    seed_seq = np.random.SeedSequence(_entropy_from(date, run_slot, salt="universe_pricing"))
    return np.random.default_rng(seed_seq)


def _price_combo(legs: tuple[_ScoredLeg, ...], rng: np.random.Generator, platform: str) -> float:
    """Expected payout per $1 on the legs' win probs, priced through the
    push-aware pooled curve rather than as the all-hit probability times the top
    tier, which would miss flex's partial-payout tiers.
    """
    n = len(legs)
    return expected_payout_with_pushes(
        p_win=np.array([leg.win_prob for leg in legs]),
        p_push=np.array([leg.push_prob for leg in legs]),
        sigma=np.eye(n),
        bet_size=n,
        boost=np.array([leg.boost for leg in legs]),
        payout_curve=_POOLED_CURVES[platform],
        rng=rng,
        full_refund_below_size=SLEEPER_FULL_REFUND_MAX_SIZE if platform == "Sleeper" else None,
    )


def _priced_candidates(
    legs: list[_ScoredLeg],
    by_size: dict[int, list[tuple[int, ...]]],
    config: PickemConfig,
    date: datetime.date,
    run_slot: str,
    platform: str,
) -> list[LedgerCandidate]:
    """A sized candidate per combo that clears ``config.min_ev`` on ``platform``'s pooled curve."""
    rng = _pricing_rng(date, run_slot)
    by_idx = {leg.idx: leg for leg in legs}

    out: list[LedgerCandidate] = []
    for size, combos in by_size.items():
        table_tier = _POOLED_CURVES[platform][size][0]
        for combo in combos:
            combo_legs = tuple(by_idx[i] for i in combo)
            ev_payout = _price_combo(combo_legs, rng, platform)
            ev = ev_payout - 1.0
            if ev < config.min_ev:
                continue
            # Sized, ranked and recorded as a same-game entry is
            # (underdog_pickem._row_to_entry): on what the entry pays when every pick
            # hits, and the win probability that payout makes worth the priced EV. The
            # all-hit probability on the bare table tier is blind to the pick multipliers
            # and the flex loss tiers, and sizes most flex entries to nothing.
            payout_mult = table_tier * math.prod(leg.boost for leg in combo_legs)
            joint_prob = ev_payout / payout_mult
            stake = fractional_kelly_stake(
                bankroll=BANKROLL_PER_REPLICATE,
                win_prob=joint_prob,
                payout_multiplier=Decimal(repr(payout_mult)),
                fraction=config.kelly_fraction,
                max_fraction_of_bankroll=config.max_stake_pct_bankroll,
            )
            games = {leg.game for leg in combo_legs}
            out.append(
                LedgerCandidate(
                    id=hashlib.sha1(
                        "|".join(sorted(leg.display for leg in combo_legs)).encode()
                    ).hexdigest()[:16],
                    contest_variant="power" if size <= 3 else "flex",
                    entry_size=size,
                    legs=tuple(leg.display for leg in combo_legs),
                    canonical_legs=tuple(leg.canonical_leg for leg in combo_legs),
                    players=frozenset(leg.player for leg in combo_legs),
                    game_span=len(games),
                    joint_prob=joint_prob,
                    payout_multiplier=payout_mult,
                    ev=ev,
                    stake=stake,
                    platform=platform,
                    lines=tuple(leg.line for leg in combo_legs),
                    model_probs=tuple(leg.win_prob for leg in combo_legs),
                    book_devig=tuple(leg.book_devig for leg in combo_legs),
                )
            )
    return out


def build_cross_game_candidates(
    offers_df: pd.DataFrame,
    config: PickemConfig,
    date: datetime.date,
    run_slot: str,
    platform: str = "Underdog",
) -> list[LedgerCandidate]:
    """Build independence-assumed (rho=0) parlay candidates spanning >=2 games.

    Filters ``offers_df`` (``Boost`` = the platform's raw pick multiplier)
    through the same leg gate ``construct_entries`` uses, beam-expands
    cross-game combos up to ``_MAX_ENTRY_SIZE`` legs, prices each via the
    pooled payout curve for ``platform``, and drops anything below
    ``config.min_ev``. See the module docstring for why this independence
    assumption is safe.

    Returns:
        One :class:`LedgerCandidate` per surviving combo, in no particular
        order (the caller's selection layer does the ranking/sampling).
    """
    eligible = filter_legs(offers_df, config)
    if eligible.empty:
        return []
    # A side the platform does not post carries a zero multiplier and no entry can hold
    # it. Left in, a Flex loss tier would drop that zero as the smallest multiplier and
    # price a payout nobody can collect.
    legs = _score_legs(eligible[eligible["Boost"] > 0], platform)
    by_size = _enumerate_cross_game_combos(legs)
    return _priced_candidates(legs, by_size, config, date, run_slot, platform)


def build_even_pick_candidates(
    offers_df: pd.DataFrame, config: PickemConfig, date: datetime.date, run_slot: str
) -> list[LedgerCandidate]:
    """Build the ``even_picks`` pool: cross-game entries of Underdog's recommended even picks.

    A leg is a row of Underdog's ``offers_df`` (``Boost`` = the raw pick multiplier)
    at exactly ``_EVEN_PICK_MULTIPLIER``, for a game on ``date``, whose side Receipts'
    rule recommends (:func:`sportstradamus.realized.recommended`). It is priced, sized
    and recorded at its served ``Win Prob``, through neither the pick'em leg gate nor
    the cell's trust: the bettor tests the recommendation as the dashboard shows it.
    Entries are dealt evenly over those legs, a game per leg, and each is priced and
    sized as :func:`build_cross_game_candidates` prices and sizes its own.

    Returns:
        One :class:`LedgerCandidate` per dealt entry at or above ``config.min_ev``,
        in no particular order.
    """
    payout = platform_payout(offers_df["Boost"], "Underdog")
    picks = offers_df[
        (offers_df["Boost"] == _EVEN_PICK_MULTIPLIER)
        & recommended(offers_df["Win Prob"], payout)
        # A leg gets one day, on its game-day read, and its entry settles at the next reflect.
        & (offers_df["Date"].astype(str).str[:10] == date.isoformat())
    ]
    legs = _served_legs(picks, "Underdog")
    # Seeded apart from the pricing stream, so the deal and the prices share no draws.
    deal_rng = np.random.default_rng(
        np.random.SeedSequence(_entropy_from(date, run_slot, salt="even_pick_deals"))
    )
    by_size = _deal_even_pick_combos(legs, deal_rng)
    return _priced_candidates(legs, by_size, config, date, run_slot, "Underdog")
