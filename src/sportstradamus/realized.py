"""Realized performance, priced at what the platforms actually pay.

The one place realized performance is priced, so Receipts, Lab Diagnostics, nightly and
the research briefs read the same numbers. ``settled_offers`` grades each settled offer
at its platform payout (``helpers.platform_payout``), keeps only sides the platform
posted (a ``Boost == 0`` side was never a bet), counts an offer once per platform (the
same prop on Underdog and Sleeper is two bets), counts a prop whose line moved only at
the lines still posted when it was last scored (games from ``COUNT_ONCE_FROM``; earlier
games count every line posted) and flags ``Recommended`` by the one definition the story
menu shares. ``window``, ``by_split``, ``cohort_summary``, ``worst_month`` and
``calibration_summary`` slice and aggregate that frame; nightly persists
``compute_realized_by_side`` (``helpers.io.REALIZED_BY_SIDE_PATH``). Pure.
"""

from datetime import UTC, datetime
from itertools import pairwise

import numpy as np
import pandas as pd

from sportstradamus.analysis import annotate_offer_outcomes
from sportstradamus.helpers import MAX_FAVORED_PAYOUT, platform_payout

# 30 matches the graduation window; 90 gives the payout bands enough legs to read.
REALIZED_WINDOWS: tuple[int, ...] = (30, 90)
# From this game date an early closing stamp is released pre-game, so Scored At is the last scoring.
COUNT_ONCE_FROM = pd.Timestamp("2026-10-09")
# The owner-locked story / "why" edge floor. Pinned equal to menu._MENU_EDGE_FLOOR by test
# rather than imported, since that constant is private to the story menu.
RECOMMENDED_EDGE_MIN: float = 0.05
# The menu's Kelly cap: the menu never shows a leg paying more (Kelly is 0 there), so the
# recommended cohort stops where the menu does.
RECOMMENDED_PAYOUT_MAX: float = MAX_FAVORED_PAYOUT
# 2.5 is RECOMMENDED_PAYOUT_MAX; the bands show whether that cap should move.
PAYOUT_BAND_EDGES: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0)
# Reliability-diagram bin edges, reaching below 0.5 so the alt/ladder split has room to
# show tail bins (alt lines carry more extreme predicted probabilities).
CAL_BINS = np.arange(0.40, 1.01, 0.05)
REALIZED_BY_SIDE_COLS = [
    "computed_at",
    "window_days",
    "cohort",
    "split",
    "key",
    "side",
    "n",
    "hit_rate",
    "pred_rate",
    "book_rate",
    "breakeven_rate",
    "units",
    "roi",
]

# pd.cut bands are right-closed, so a payout sitting on an edge (the 2.5 cap) lands in the
# band that edge closes.
_PAYOUT_BAND_BINS = [-np.inf, *PAYOUT_BAND_EDGES, np.inf]
_PAYOUT_BAND_LABELS = [
    f"<={PAYOUT_BAND_EDGES[0]:.2f}",
    *(f"{lo:.2f}-{hi:.2f}" for lo, hi in pairwise(PAYOUT_BAND_EDGES)),
    f">{PAYOUT_BAND_EDGES[-1]:.2f}",
]
# One real offer per platform: the same prop posted on Underdog and Sleeper is two bets.
_OFFER_KEY = ["Date", "Player", "Market", "Line", "Bet", "Platform"]
# An offer without its line, or the side taken there: a moved line can flip the side.
_PROP_KEY = ["Date", "Player", "Market", "Platform"]
# The columns settled_offers adds, typed so an empty frame slices like a full one.
_PRICED_DTYPES = {
    "Payout": "float64",
    "Breakeven": "float64",
    "Hit": "bool",
    "Unit": "float64",
    "Recommended": "bool",
    "Quote": "object",
}
# Pinned so an empty run writes the same parquet schema as a full one; the rest are object.
_DTYPES = {
    "computed_at": "datetime64[ns]",
    "window_days": "int16",
    "n": "int64",
    **dict.fromkeys(
        ["hit_rate", "pred_rate", "book_rate", "breakeven_rate", "units", "roi"], "float64"
    ),
}
_CAL_SUMMARY_COLS = ["Cohort", "Alt Line", "Bin", "Predicted", "Actual", "N", "ECE", "ROI"]


def recommended(win_prob: pd.Series, payout: pd.Series) -> pd.Series:
    """Whether each side is recommended, by the one rule Receipts and the paper ledger share."""
    return (win_prob * payout - 1 >= RECOMMENDED_EDGE_MIN) & payout.between(
        1, RECOMMENDED_PAYOUT_MAX, inclusive="right"
    )


def settled_offers(history: pd.DataFrame) -> pd.DataFrame:
    """One row per settled offer the platform posted, priced at its platform payout.

    Keeps rows settled Over/Under with a chosen side and ``Win Prob`` on a posted side
    (``Boost > 0``), once per ``(Date, Player, Market, Line, Bet, Platform)``, with every
    input column (Lab and the CSV export read them), and adds ``Payout`` (full decimal
    payout), ``Breakeven`` (``1 / Payout``), ``Hit`` (bool), ``Unit`` (profit on a 1-unit
    stake), ``Recommended`` (an edge ``Win Prob x Payout - 1`` of at least
    ``RECOMMENDED_EDGE_MIN`` at a payout in ``(1, RECOMMENDED_PAYOUT_MAX]``) and ``Quote``
    (``"fallback"`` on the book-priced path, else ``"quoted"`` with a servable sportsbook
    quote, else ``"unquoted"``). Empty input comes back empty with those columns.

    A prop (``Date, Player, Market, Platform``) dated ``COUNT_ONCE_FROM`` or later counts
    only at the lines still posted when it was last scored: the rows carrying the latest
    ``Scored At`` among every row of the prop handed in, unposted and unsettled ones
    included. Rungs posted side by side share that stamp and stay separate legs; a line
    the platform moved off drops out, as does an unstamped row beside a stamped one. A
    prop never stamped, a frame with no ``Scored At`` column and every earlier game keep
    each posted line.

    Args:
        history: Flat prediction history (``history_schema.HISTORY_COLS``), ``Actual``
            filled by reflect. A frame already carrying ``Result`` (the dashboard's
            ``get_filtered_history``) is used as it stands: annotating is a row-wise apply.
    """
    if history.empty:
        return history.assign(
            **{col: pd.Series(dtype=dtype) for col, dtype in _PRICED_DTYPES.items()}
        )
    settled = history if "Result" in history.columns else annotate_offer_outcomes(history)
    keep = (
        settled["Result"].isin(("Over", "Under"))
        & settled["Bet"].notna()
        & settled["Win Prob"].notna()
        & (settled["Boost"] > 0)
    )
    # tail_scorecard's replayed test rows were never scored live and carry no stamp column.
    if "Scored At" in settled.columns:
        # Over every row, not only the posted and settled ones: when the last scoring is
        # of a side the platform never posted, the older line is gone all the same.
        last_scored = settled.groupby(_PROP_KEY)["Scored At"].transform("max")
        keep &= (
            (pd.to_datetime(settled["Date"]) < COUNT_ONCE_FROM)
            | settled["Scored At"].eq(last_scored)
            | last_scored.isna()
        )
    offers = settled[keep].drop_duplicates(subset=_OFFER_KEY)
    payout = platform_payout(offers["Boost"], offers["Platform"])
    # Derived, not annotate's Hit: that column is absent when no row resolved.
    hit = offers["Bet"] == offers["Result"]
    return offers.assign(
        Payout=payout,
        Breakeven=1 / payout,
        Hit=hit,
        Unit=hit * payout - 1,
        Recommended=recommended(offers["Win Prob"], payout),
        Quote=np.select(
            [offers["Model Version"].eq("book_fallback"), offers["Market Projection"].notna()],
            ["fallback", "quoted"],
            "unquoted",
        ),
    )


def window(offers: pd.DataFrame, days: int, now: datetime) -> pd.DataFrame:
    """The offers dated within the trailing ``days`` of ``now``.

    Receipts and the nightly ledger both cut their windows here, so a 30-day window holds
    the same offers everywhere. ``now`` is tz-stripped, so pass UTC (``datetime.now(UTC)``),
    the anchor nightly's live-metrics frame uses.
    """
    start = pd.Timestamp(now).tz_localize(None) - pd.Timedelta(days=days)
    return offers[pd.to_datetime(offers["Date"]) >= start]


def _realized_stats(groups) -> pd.DataFrame:
    """Realized stats per group of a ``settled_offers`` groupby, the one shared spec."""
    stats = groups.agg(
        n=("Hit", "size"),
        hit_rate=("Hit", "mean"),
        pred_rate=("Win Prob", "mean"),
        book_rate=("Market Prob", "mean"),
        payout=("Payout", "mean"),
        units=("Unit", "sum"),
    )
    # 1 / mean payout is n / sum(Payout): the hit rate at which flat 1-unit stakes break even.
    return stats.assign(breakeven_rate=1 / stats["payout"], roi=stats["units"] / stats["n"])


def by_split(offers: pd.DataFrame) -> pd.DataFrame:
    """Realized stats of ``settled_offers`` rows per (split, key, side).

    ``split`` names a grouping and ``key`` the offer's value in it: side (key ``"all"``),
    league, platform, alt_line, payout_band, cell (``League/Market``), model_version (NaN
    reads ``"legacy"``) or quote (``Quote``). No windows or cohorts: callers slice first.

    Returns:
        One row per group: ``split``, ``key``, ``side``, ``n``, ``hit_rate``, ``pred_rate``
        and ``book_rate`` (the mean model and book probabilities of the chosen side),
        ``payout`` (mean decimal payout), ``breakeven_rate`` (``n / sum(Payout)``, the hit
        rate at which flat 1-unit stakes break even), ``units`` (profit in 1-unit stakes)
        and ``roi`` (``units / n``).
    """
    split_keys = {
        "side": "all",
        "league": offers["League"],
        "platform": offers["Platform"],
        "alt_line": np.where(offers["Alt Line"].eq(True), "alt", "main"),
        "payout_band": pd.cut(
            offers["Payout"], _PAYOUT_BAND_BINS, labels=_PAYOUT_BAND_LABELS
        ).astype(str),
        "cell": offers["League"] + "/" + offers["Market"],
        "model_version": offers["Model Version"].fillna("legacy"),
        "quote": offers["Quote"],
    }
    long = offers.assign(**split_keys).melt(
        id_vars=["Bet", "Hit", "Win Prob", "Market Prob", "Payout", "Unit"],
        value_vars=list(split_keys),
        var_name="split",
        value_name="key",
    )
    return (
        _realized_stats(long.groupby(["split", "key", "Bet"]))
        .reset_index()
        .rename(columns={"Bet": "side"})
    )


def cohort_summary(offers: pd.DataFrame) -> dict[str, float]:
    """One cohort's realized stats: ``by_split``'s stats over all of ``offers`` at once.

    Keys ``n, hit_rate, pred_rate, book_rate, payout, breakeven_rate, units, roi``; an
    empty cohort reads ``n == 0`` and NaN elsewhere.
    """
    # One group holding every row keeps this on by_split's aggregation; reindexing to it
    # gives an empty cohort its row.
    stats = _realized_stats(offers.groupby(lambda _: "all")).reindex(["all"])
    return stats.fillna({"n": 0}).astype({"n": "int64"}).to_dict("records")[0]


def compute_realized_by_side(history: pd.DataFrame, *, now: datetime | None = None) -> pd.DataFrame:
    """Realized hit rate and ROI per bet side, long over windows, cohorts and splits.

    ``settled_offers(history)`` inside each ``REALIZED_WINDOWS`` trailing window, in cohort
    ``bettable`` (every row) and ``recommended`` (the ``Recommended`` rows), through
    ``by_split``. ``now`` anchors the windows and the ``computed_at`` stamp; it defaults to
    UTC now, stored tz-naive like the live-metrics frame.

    Returns:
        ``REALIZED_BY_SIDE_COLS`` in order, one row per (window_days, cohort, split, key,
        side). Empty, same dtypes, when nothing settles.
    """
    if history.empty:
        return pd.DataFrame(columns=REALIZED_BY_SIDE_COLS).astype(_DTYPES)
    offers = settled_offers(history)
    now_ts = pd.Timestamp(now or datetime.now(UTC)).tz_localize(None)
    frames = []
    for window_days in REALIZED_WINDOWS:
        recent = window(offers, window_days, now_ts)
        for cohort, rows in (("bettable", recent), ("recommended", recent[recent["Recommended"]])):
            frames.append(by_split(rows).assign(window_days=window_days, cohort=cohort))
    out = pd.concat(frames, ignore_index=True).assign(computed_at=now_ts)
    return out[REALIZED_BY_SIDE_COLS].astype(_DTYPES)


def worst_month(offers: pd.DataFrame) -> dict[str, str | float]:
    """The worst calendar month by realized units at the platform payout.

    ``{month, units, n, win_pct}`` for the ``YYYY-MM`` with the lowest summed ``Unit``,
    ties to the earliest month; ``{}`` when nothing settled.
    """
    if offers.empty:
        return {}
    # groupby sorts the YYYY-MM keys, so idxmin keeps the earliest of tied months.
    months = _realized_stats(offers.groupby(pd.to_datetime(offers["Date"]).dt.strftime("%Y-%m")))
    worst = months["units"].idxmin()
    return {
        "month": worst,
        "units": float(months.at[worst, "units"]),
        "n": int(months.at[worst, "n"]),
        "win_pct": float(months.at[worst, "hit_rate"]),
    }


def calibration_summary(offers: pd.DataFrame) -> pd.DataFrame:
    """Reliability rows per (cohort x alt-line split x ``Win Prob`` bin) of settled offers.

    ``Cohort`` is ``"posted"`` (every ``settled_offers`` row) or ``"recommended"``; ``Alt
    Line`` is the bool split, unstamped rows reading main. Each ``CAL_BINS`` bin carries its
    ``Predicted`` mean ``Win Prob``, ``Actual`` hit rate and ``N``. Each split carries its
    ``ECE`` over its binned rows and its ``ROI``, the mean ``Unit`` over every settled row
    in it: a ``Win Prob`` under the first bin edge counts toward ROI but not ECE. A split
    with no binned row is absent.
    """
    split = ["Cohort", "Alt Line"]
    rows = offers[["Win Prob", "Hit", "Unit"]].assign(**{"Alt Line": offers["Alt Line"].eq(True)})
    cohorts = pd.concat(
        [rows.assign(Cohort="posted"), rows[offers["Recommended"]].assign(Cohort="recommended")],
        ignore_index=True,
    )
    table = (
        cohorts.assign(
            Bin=pd.cut(cohorts["Win Prob"], CAL_BINS),
            ROI=cohorts.groupby(split)["Unit"].transform("mean"),
        )
        .groupby([*split, "Bin"], observed=True)
        .agg(
            Predicted=("Win Prob", "mean"),
            Actual=("Hit", "mean"),
            N=("Hit", "size"),
            ROI=("ROI", "first"),
        )
        .reset_index()
    )
    error = table["N"] * (table["Predicted"] - table["Actual"]).abs()
    per_split = table.assign(error=error).groupby(split)
    table["ECE"] = per_split["error"].transform("sum") / per_split["N"].transform("sum")
    return table.assign(Bin=table["Bin"].astype(str))[_CAL_SUMMARY_COLS]
