"""Per-offer trajectory of a DFS book's main line and of the fair line its price implies.

Collapses the archive's ladder — every ``(line, p_over)`` rung a book posted on every
poll — into one row per offer. Each poll contributes two numbers. The **main line** is
the rung priced nearest even money, so alt rungs coming and going cannot move the trend
(Sleeper posts a ladder whose lowest rung is not its line). The **fair line** is the
line at which the book's price would be even money, so a multiplier move with the line
held still registers (Underdog reprices touchdown props without moving the 0.5). A
posted number is a step function — a book holds it until it reposts — so both
trajectories carry the last poll forward instead of interpolating between polls.

Pure aggregation: it opens no archive and writes no file. The caller reads the ladder
via ``Archive.get_book_line_histories`` and persists the result. The snapshot's schema
lives with its path in ``helpers.io`` so the dashboard reader and this producer cannot
drift apart.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from sportstradamus.helpers.distributions import resolve_std
from sportstradamus.helpers.io import LINE_MOVEMENT_COLS, LINE_MOVEMENT_KEYS

# Sparkline width the Board grid renders. Fixed so every row plots on the same
# horizontal scale however often its book happened to be polled.
_MOVEMENT_POINTS = 12

# Rungs chained closer than this are one poll: a run stages a key's rungs under 10 s
# apart, and separate prophecize/confer runs land at least 410 s apart.
_POLL_GAP = pd.Timedelta(seconds=60)

# Fair lines round to hundredths before any diffing: fine enough for a touchdown price
# move (0.50 -> 0.66), coarse enough that interpolation float noise never counts as one.
_FAIR_DECIMALS = 2

# A tenth of a percent: finer than any payout step moves the implied over-probability.
_P_OVER_DECIMALS = 3

# A one-sided payout under 1.0x stores a raw breakeven past 0 or 1 (dfs_boost_probs),
# where the normal quantile is infinite; the clip holds a fair line within 2.33σ.
_P_FLOOR, _P_CEIL = 0.01, 0.99

_ARCHIVE_TO_OFFER = {
    "league": "League",
    "book": "Platform",
    "market": "Market",
    "entity": "Player",
}


def build_line_movement(observations: pd.DataFrame, offers: pd.DataFrame) -> pd.DataFrame:
    """Summarize each offer's ladder history into one joinable row.

    Rungs of one key staged within ``_POLL_GAP`` of each other form one poll, timed at
    its earliest rung. The poll's main rung is the one priced nearest even money; a tie
    goes to the higher line, as ``Archive.add_dfs`` breaks it. The fair line, in stat
    units rounded to ``_FAIR_DECIMALS``, is the line at which that price would be even:

    * With a partner — the nearest rung past even money, in the direction of even —
      interpolate linearly in p: ``L + (0.5 - p)·(L' - L)/(p' - p)``. It lands between
      the two lines, so a balanced rung flipping between straddling rungs barely moves
      it.
    * Without one, ``L + σ·Φ⁻¹(p)``: the median of a normal outcome with scale σ and
      P(X > L) = p, exactly L at even price. σ is the offer's ``resolve_std`` spread,
      so the centre is the book's but the scale is the model's; p is clipped to
      [``_P_FLOOR``, ``_P_CEIL``]. Not floored at 0 — a first-TD longshot sits 1-2σ
      below its line.

    Args:
        observations: ``Archive.get_book_line_histories`` output in any row order, one
            row per rung per poll: ``league, market, game_date, entity, book``,
            ``observed_at`` (naive UTC), ``line`` (stat units) and ``p_over`` (the
            rung's de-vigged over-probability).
        offers: The scored slate. Its ``(League, Platform, Market, Player, Date)``
            identity filters the observations; each key's first row supplies σ from
            ``Projection``, ``Projection STD`` and ``CV``.

    Returns:
        One row per offer key the archive observed, with columns ``LINE_MOVEMENT_COLS``.
        ``open_line``, ``close_line``, ``move`` and ``n_moves`` follow the main line;
        ``n_price_moves`` counts polls where the fair line changed while the main line
        held, and ``fair_move`` is the last fair line minus the first. ``series`` and
        ``fair_series`` are JSON lists of up to ``_MOVEMENT_POINTS`` main- and fair-line
        samples spaced evenly *in time* (confer slots are not evenly spaced, so
        even-in-index would distort the shape) between ``first_seen`` and ``last_seen``,
        each carrying the value posted at or before that instant; their first and last
        entries are the open and close values. ``changes`` is a JSON list of ``{t, line,
        p_over, fair}`` — ``t`` UTC ISO-8601 with its offset — for the first poll, every
        poll where either line changed, and the last poll. An offer seen once yields
        one-element series and zero moves. Column-stable and empty when either input
        is, or when no observation matches an offer.
    """
    if observations.empty or offers.empty:
        return pd.DataFrame(columns=LINE_MOVEMENT_COLS)

    obs = observations.rename(columns=_ARCHIVE_TO_OFFER)
    # game_date arrives as a DuckDB DATE; offers.Date is an ISO string.
    obs["Date"] = pd.to_datetime(obs["game_date"]).dt.strftime("%Y-%m-%d")
    obs = obs.merge(offers[LINE_MOVEMENT_KEYS].drop_duplicates(), on=LINE_MOVEMENT_KEYS)
    if obs.empty:
        return pd.DataFrame(columns=LINE_MOVEMENT_COLS)

    first_offers = offers.drop_duplicates(LINE_MOVEMENT_KEYS)
    spreads = first_offers[LINE_MOVEMENT_KEYS].assign(
        sigma=[
            resolve_std(std, ev, cv)
            for std, ev, cv in zip(
                first_offers["Projection STD"],
                first_offers["Projection"],
                first_offers["CV"],
                strict=True,
            )
        ]
    )
    polls = _main_rungs(obs).merge(spreads, on=LINE_MOVEMENT_KEYS, how="left")
    line, p = polls["line"], polls["p_over"]
    interpolated = line + (0.5 - p) * (polls["line_partner"] - line) / (polls["p_over_partner"] - p)
    from_spread = line + polls["sigma"] * norm.ppf(p.clip(_P_FLOOR, _P_CEIL))
    polls["fair"] = interpolated.where(polls["line_partner"].notna(), from_spread).round(
        _FAIR_DECIMALS
    )

    rows = [
        _movement_row(key, group) for key, group in polls.groupby(LINE_MOVEMENT_KEYS, sort=False)
    ]
    return pd.DataFrame(rows, columns=LINE_MOVEMENT_COLS)


def _main_rungs(obs: pd.DataFrame) -> pd.DataFrame:
    """Collapse rung rows into one row per poll: its main rung and that rung's partner.

    Vectorized over the whole slate, since a per-poll Python loop costs prophecize
    minutes. The partner sits past even money on the side toward even: the lowest rung
    above the main line priced under 0.5 when the main rung prices over, the highest
    rung below it priced over 0.5 when under. ``line_partner`` and ``p_over_partner``
    are NaN when the main rung prices exactly even or no rung qualifies. Rows come back
    in key-then-time order, each stamped with its poll's earliest ``observed_at``.
    """
    obs = obs.sort_values([*LINE_MOVEMENT_KEYS, "observed_at"], ignore_index=True)
    # A key's first row diffs to NaT, which compares False and so opens a poll.
    same_poll = obs.groupby(LINE_MOVEMENT_KEYS, sort=False)["observed_at"].diff() <= _POLL_GAP
    obs["poll"] = (~same_poll).cumsum()
    obs["observed_at"] = obs.groupby("poll")["observed_at"].transform("min")
    obs["from_even"] = (obs["p_over"] - 0.5).abs()

    ranked = obs.sort_values(["poll", "from_even", "line"], ascending=[True, True, False])
    rungs = ranked.drop_duplicates(["poll", "line"])
    main = rungs.drop_duplicates("poll")

    pairs = main[["poll", "line", "p_over"]].merge(
        rungs[["poll", "line", "p_over"]], on="poll", suffixes=("", "_partner")
    )
    above = (
        (pairs["p_over"] > 0.5)
        & (pairs["line_partner"] > pairs["line"])
        & (pairs["p_over_partner"] < 0.5)
    )
    below = (
        (pairs["p_over"] < 0.5)
        & (pairs["line_partner"] < pairs["line"])
        & (pairs["p_over_partner"] > 0.5)
    )
    pairs["line_gap"] = (pairs["line_partner"] - pairs["line"]).abs()
    partners = pairs[above | below].sort_values(["poll", "line_gap"]).drop_duplicates("poll")
    return main.merge(partners[["poll", "line_partner", "p_over_partner"]], on="poll", how="left")


def _movement_row(key: tuple, polls: pd.DataFrame) -> dict:
    lines = polls["line"].to_numpy(dtype=float)
    fair = polls["fair"].to_numpy(dtype=float)
    p_over = polls["p_over"].to_numpy(dtype=float)
    stamps = polls["observed_at"]
    first_seen = stamps.iloc[0]

    # Elapsed seconds, not raw epoch nanoseconds: an int64 timestamp exceeds
    # float64's exact-integer range, and a tick rounding below the first
    # observation would make searchsorted wrap to the end of the series.
    elapsed = (stamps - first_seen).dt.total_seconds().to_numpy()
    ticks = np.linspace(0.0, elapsed[-1], min(len(lines), _MOVEMENT_POINTS))
    sampled = np.searchsorted(elapsed, ticks, side="right") - 1

    line_moved = np.diff(lines) != 0
    fair_moved = np.diff(fair) != 0
    logged = np.r_[True, line_moved | fair_moved]
    logged[-1] = True
    changes = [
        {
            "t": stamp.tz_localize("UTC").isoformat(timespec="seconds"),
            "line": posted,
            "p_over": round(price, _P_OVER_DECIMALS),
            "fair": fair_line,
        }
        for stamp, posted, price, fair_line in zip(
            stamps[logged],
            lines[logged].tolist(),
            p_over[logged].tolist(),
            fair[logged].tolist(),
            strict=True,
        )
    ]

    return {
        **dict(zip(LINE_MOVEMENT_KEYS, key, strict=True)),
        "open_line": float(lines[0]),
        "close_line": float(lines[-1]),
        "move": float(lines[-1] - lines[0]),
        "n_moves": int(np.count_nonzero(line_moved)),
        "first_seen": first_seen,
        "last_seen": stamps.iloc[-1],
        "series": json.dumps(lines[sampled].tolist()),
        "n_price_moves": int(np.count_nonzero(fair_moved & ~line_moved)),
        "fair_move": round(float(fair[-1] - fair[0]), _FAIR_DECIMALS),
        "fair_series": json.dumps(fair[sampled].tolist()),
        "changes": json.dumps(changes),
    }
