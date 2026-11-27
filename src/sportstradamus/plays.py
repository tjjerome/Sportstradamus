"""Plays: named rules that pick sides from the board, scored in money.

A play is a rule over the sides the apps posted. ``posted_sides`` lays those out, both sides
of every settled offer and each graded at its own payout, so a play may pick a side the board
never showed. ``play_record`` is the money table: what each play's legs returned, per league
and pick kind, alone and inside the Power entries they could have built
(``power_entry_returns``, exact). ``entry_record`` scores the engine's own graded entries the
same way. Pure.
"""

import math
from collections.abc import Callable
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from sportstradamus import live_fit, realized
from sportstradamus.helpers import (
    leg_kind,
    platform_payout,
    sleeper_payouts,
    underdog_payouts,
)
from sportstradamus.training.scorecard import _bootstrap_mean_ci_clustered

# The Power entries with the lowest hit rate a pick needs to break even: on Underdog's table
# .534, .536 and .537 a pick at 2, 3 and 4 picks, against .549 at 5.
POWER_ENTRY_SIZES: tuple[int, ...] = (2, 3, 4)

# What an all-hit entry of each size pays, before the legs' own multipliers.
_POWER_TABLES: dict[str, dict[int, float]] = {
    "Underdog": underdog_payouts["power"],
    "Sleeper": sleeper_payouts["power"],
}

# The money table is read every night, and a fixed-sample interval overstates what repeated
# looks show. The sequence on a play's excess over break-even holds at every look: its level
# matches the 95 % date-resampled bounds, and it is tightest near the legs a league's even
# picks reach in a 90-day window.
SEQUENCE_ALPHA = 0.05
SEQUENCE_DESIGN_LEGS = 1000

# The trailing game dates the nightly money table scores, the window main's money test reads
# a league's record over (docs/handoffs/profit-engine.md, 5.5).
PLAY_WINDOW_DAYS = 90
# Fixed, so the same history prints the same bounds wherever the table is built.
RECORD_SEED = 0

# The side columns the money table reads. A play's sides are copied once per pooling (by
# league, by kind, by neither), so the copies carry these alone.
_SCORED_SIDE_COLS = [
    "Date",
    "League",
    "Kind",
    "Platform",
    "Game",
    "Multiplier",
    "Win Prob",
    "Breakeven",
    "Hit",
    "Unit",
]

PLAY_RECORD_COLS: list[str] = [
    "play",
    "league",
    "kind",
    "n",
    "dates",
    "first_date",
    "last_date",
    "claim",
    "hit_rate",
    "breakeven_rate",
    "excess",
    "cs_lo",
    "cs_hi",
    "roi",
    "roi_lo",
    "roi_hi",
    *(f"power{size}_roi{bound}" for size in POWER_ENTRY_SIZES for bound in ("", "_lo", "_hi")),
]

ENTRY_RECORD_COLS: list[str] = [
    "league",
    "platform",
    "size",
    "n",
    "dates",
    "first_date",
    "last_date",
    "model_ev",
    "payout",
    "hit_rate",
    "breakeven_rate",
    "excess",
    "cs_lo",
    "cs_hi",
    "roi",
    "roi_lo",
    "roi_hi",
]


def _ev_rank(sides: pd.DataFrame) -> pd.Series:
    """As many legs as ``model_edge`` took on each date, league and kind: the highest ``EV``."""
    slate = sides.groupby(["Date", "League", "Kind"])
    taken = slate["Recommended"].transform("sum")
    # Ties keep row order, and a side the fit did not price has no rank.
    return slate["EV"].rank(method="first", ascending=False) <= taken


# ``model_edge`` is the board's own rule (``realized.recommended``), the cohort Receipts and
# the live gap read. ``under`` and ``all`` are controls that need no model: what the app's
# cut costs a bettor who takes every Under on an even pick, or every side it posts.
# ``ev_edge`` is the board's "+EV" mark (``realized.ev_marked``). ``ev_rank`` spends the
# board's leg count on the highest ``EV`` instead, and ``ev_pick`` is the board's own picks as
# each run flagged them (``live_fit.picks``). A new play goes last: ``play_record`` draws its
# bounds in this order, so the plays before it keep their numbers.
PLAYS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "model_edge": lambda sides: sides["Recommended"],
    "under": lambda sides: sides["Side"].eq("Under") & sides["Kind"].eq("underdog_even"),
    "all": lambda sides: pd.Series(True, index=sides.index),
    "ev_edge": lambda sides: realized.ev_marked(sides["Fit Prob Low"], sides["Payout"]),
    "ev_rank": _ev_rank,
    # NaN on a row scored before the flag existed.
    "ev_pick": lambda sides: sides["Pick"].eq(True),
}


def posted_sides(offers: pd.DataFrame) -> pd.DataFrame:
    """One row per side the app posted on each settled offer, graded at that side's payout.

    ``offers`` is ``realized.settled_offers`` output: one row per offer, at the side the model
    chose. The other side joins it where the app posted one (``Payout Over`` / ``Payout
    Under`` above zero; history carries both from 2026-10-05).

    Returns:
        Every input column, with ``Side`` (``Over`` / ``Under``), ``Multiplier`` (the side's
        own: Underdog's raw multiplier, Sleeper's decimal payout), ``Kind``
        (``helpers.leg_kind``), ``Game`` (the league and its two teams, the same from either
        team) and, priced per side, ``Payout``, ``Breakeven``, ``Hit``, ``Unit`` and
        ``Win Prob`` (the model's probability of that side). ``Recommended`` and ``Pick`` are
        False on a side the model did not choose. The EV fit prices the model's side alone,
        so ``live_fit.SCORE_COLS`` are NaN on the other side, and on every side of offers
        scored before the fit.
    """
    over = offers["Bet"].eq("Over")
    payout_chosen = offers["Payout Over"].where(over, offers["Payout Under"])
    payout_other = offers["Payout Under"].where(over, offers["Payout Over"])
    chosen = offers.assign(Side=offers["Bet"], Multiplier=offers["Boost"])
    # history_schema: a side's raw multiplier is its payout over the chosen side's, times Boost.
    other = offers.assign(
        Side=np.where(over, "Under", "Over"),
        Multiplier=payout_other * offers["Boost"] / payout_chosen,
        Recommended=False,
        Pick=False,
        **{"Win Prob": 1 - offers["Win Prob"]},
        **dict.fromkeys(live_fit.SCORE_COLS, np.nan),
    )
    sides = pd.concat([chosen, other[other["Multiplier"] > 0]], ignore_index=True)
    payout = platform_payout(sides["Multiplier"], sides["Platform"])
    hit = sides["Side"] == sides["Result"]
    teams = np.sort(sides[["Team", "Opponent"]].astype(str).to_numpy(), axis=1)
    return sides.assign(
        Kind=leg_kind(sides["Multiplier"], sides["Platform"]),
        Game=sides["League"] + ":" + teams[:, 0] + "/" + teams[:, 1],
        Payout=payout,
        Breakeven=1 / payout,
        Hit=hit,
        Unit=hit * payout - 1,
    )


def _elementary_symmetric(values: np.ndarray, top_degree: int) -> np.ndarray:
    """``e_0 .. e_top_degree`` of ``values``: ``e_k`` sums the product of every k-subset."""
    e = np.zeros(top_degree + 1)
    e[0] = 1.0
    for value in values:
        e[1:] += e[:-1] * value
    return e


def power_entry_returns(legs: pd.DataFrame) -> pd.DataFrame:
    """Mean return of every Power entry a day's legs could build with one leg per game.

    Exact, with no sampling. An entry of k legs from k games pays its platform's Power table
    times the legs' multipliers when all hit, so over every such entry the count is the k-th
    elementary symmetric polynomial of the games' leg counts, and the amount paid is the
    table times that polynomial of the games' summed winning multipliers. An entry holds one
    platform's legs; a date's platforms pool by entry. A pushed leg is in no entry: ``legs``
    holds settled sides only.

    Args:
        legs: Sides with ``Date``, ``Platform``, ``Game``, ``Multiplier`` and ``Hit``.

    Returns:
        One row per (``Date``, ``size``) that has an entry: ``entries`` and ``roi`` (the
        amount paid per entry, less the stake).
    """
    top_size = max(POWER_ENTRY_SIZES)
    rows = []
    for (date, platform), day in legs.groupby(["Date", "Platform"]):
        table = _POWER_TABLES[platform]
        games = day.assign(Won=day["Multiplier"] * day["Hit"]).groupby("Game")
        entries = _elementary_symmetric(games.size().to_numpy(float), top_size)
        won = _elementary_symmetric(games["Won"].sum().to_numpy(), top_size)
        rows += [
            (date, size, entries[size], table[size] * won[size])
            for size in POWER_ENTRY_SIZES
            if size in table and entries[size] > 0
        ]
    per_date = (
        pd.DataFrame(rows, columns=["Date", "size", "entries", "paid"])
        .groupby(["Date", "size"], as_index=False)
        .sum()
    )
    return per_date.assign(roi=per_date["paid"] / per_date["entries"] - 1).drop(columns="paid")


def mixture_rho(design_legs: int, alpha: float) -> float:
    """The mixture scale that makes the boundary tightest at the design sample (variance t/4)."""
    v0 = design_legs / 4
    # The boundary's derivative in rho vanishes where log((v0 + rho) / (rho alpha^2)) = v0 / rho;
    # that residual rises monotonically in rho, so the bracket holds its one root.
    return brentq(lambda rho: math.log((v0 + rho) / (rho * alpha**2)) - v0 / rho, 1e-3, 1e6)


def sequence_bounds(excess: np.ndarray, alpha: float, design_legs: int) -> tuple[float, float]:
    """The confidence sequence for the mean excess after these legs: mean +/- u(t/4) / t.

    A normal-mixture sequence (Howard, Ramdas, McAuliffe and Sekhon 2021, with Hoeffding's
    variance proxy t/4 for a hit less its break-even): it holds at every look, and is
    tightest at ``design_legs``.
    """
    t, rho = len(excess), mixture_rho(design_legs, alpha)
    v = t / 4
    radius = math.sqrt((v + rho) * math.log((v + rho) / (rho * alpha**2))) / t
    return float(excess.mean() - radius), float(excess.mean() + radius)


def _unit_bets_scored(bets: pd.DataFrame, rng: np.random.Generator) -> dict[str, float | str]:
    """Score legs and entries alike from their ``Date``, ``Hit``, ``Breakeven`` and ``Unit``."""
    dates = bets["Date"].to_numpy()
    excess = (bets["Hit"] - bets["Breakeven"]).to_numpy(float)
    cs_lo, cs_hi = sequence_bounds(excess, SEQUENCE_ALPHA, SEQUENCE_DESIGN_LEGS)
    roi, roi_lo, roi_hi = _bootstrap_mean_ci_clustered(bets["Unit"].to_numpy(float), dates, rng)
    return {
        "n": len(bets),
        "dates": bets["Date"].nunique(),
        "first_date": dates.min(),
        "last_date": dates.max(),
        "hit_rate": bets["Hit"].mean(),
        "breakeven_rate": bets["Breakeven"].mean(),
        "excess": excess.mean(),
        "cs_lo": cs_lo,
        "cs_hi": cs_hi,
        "roi": roi,
        "roi_lo": roi_lo,
        "roi_hi": roi_hi,
    }


def play_record(sides: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """The money table: what each play's legs returned, per league and pick kind.

    Args:
        sides: :func:`posted_sides` output, cut to the window to score.
        rng: Seeded generator behind the date-resampled bounds.

    Returns:
        ``PLAY_RECORD_COLS``: one row per play, league and kind with a leg, and the rows that
        pool every league or every kind as ``all``. ``n`` legs over ``dates`` game dates
        (``first_date`` to ``last_date``); ``claim``, the model's mean probability of the
        picked sides, beside their ``hit_rate`` and ``breakeven_rate`` (the mean of each
        leg's own break-even); ``excess``, the hit rate less break-even, with its sequence
        ``cs_lo`` / ``cs_hi`` (:func:`sequence_bounds`); ``roi``, the return on a unit a leg;
        and ``power{k}_roi``, the legs' mean :func:`power_entry_returns` of their date, so
        entries are staked as the legs are and a date counts by its legs. ``_lo`` / ``_hi``
        are 95 % bounds that resample whole game dates.
    """
    rows = []
    for play, rule in PLAYS.items():
        picked = sides.loc[rule(sides), _SCORED_SIDE_COLS]
        pooled = pd.concat(
            [
                picked,
                picked.assign(League="all"),
                picked.assign(Kind="all"),
                picked.assign(League="all", Kind="all"),
            ]
        )
        for (league, kind), legs in pooled.groupby(["League", "Kind"]):
            row = {
                "play": play,
                "league": league,
                "kind": kind,
                "claim": legs["Win Prob"].mean(),
                **_unit_bets_scored(legs, rng),
            }
            entries = power_entry_returns(legs)
            for size in POWER_ENTRY_SIZES:
                name = f"power{size}_roi"
                of_date = entries.loc[entries["size"] == size].set_index("Date")["roi"]
                row[name], row[f"{name}_lo"], row[f"{name}_hi"] = _bootstrap_mean_ci_clustered(
                    legs["Date"].map(of_date).to_numpy(float), legs["Date"].to_numpy(), rng
                )
            rows.append(row)
    return pd.DataFrame(rows, columns=PLAY_RECORD_COLS)


def trailing_record(offers: pd.DataFrame, now: datetime, rng: np.random.Generator) -> pd.DataFrame:
    """:func:`play_record` over the trailing ``PLAY_WINDOW_DAYS`` of ``now``: the nightly table.

    ``offers`` is ``realized.settled_offers`` output. Games before ``realized.COUNT_ONCE_FROM``
    stay out however far the window reaches: a prop there counts at every line it was scored at.
    """
    recent = realized.window(offers, PLAY_WINDOW_DAYS, now)
    recent = recent[pd.to_datetime(recent["Date"]) >= realized.COUNT_ONCE_FROM]
    return play_record(posted_sides(recent), rng)


def entry_record(entries: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """What the engine's own all-or-nothing entries returned, per league, platform and size.

    An entry counts once every leg settled as a hit or a miss (``Legs Resolved`` equals
    ``Bet Size``), and pays its ``Boost`` when none missed. One with an ungraded or a pushed
    leg pays at a smaller size, a payout these columns do not hold, and is left out.

    Args:
        entries: The parlay history's scalar columns (``Date``, ``League``, ``Platform``,
            ``Bet Size``, ``Boost``, ``Model EV``, ``Legs Resolved``, ``Misses``), cut to the
            window to score.
        rng: Seeded generator behind the date-resampled bounds.

    Returns:
        ``ENTRY_RECORD_COLS``: :func:`play_record`'s columns for entries in place of legs,
        with ``model_ev`` (the mean value the engine claimed for a unit staked) and
        ``payout`` (the mean an all-hit entry pays).
    """
    graded = entries[entries["Legs Resolved"] == entries["Bet Size"]]
    won = graded["Misses"] == 0
    graded = graded.assign(Hit=won, Breakeven=1 / graded["Boost"], Unit=won * graded["Boost"] - 1)
    rows = [
        {
            "league": league,
            "platform": platform,
            "size": size,
            "model_ev": part["Model EV"].mean(),
            "payout": part["Boost"].mean(),
            **_unit_bets_scored(part, rng),
        }
        for (league, platform, size), part in graded.groupby(["League", "Platform", "Bet Size"])
    ]
    return pd.DataFrame(rows, columns=ENTRY_RECORD_COLS)
