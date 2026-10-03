"""Realized ledger by bet side, priced at what the platforms actually pay.

Receipts grades every rec at a flat -110 and the nightly profit sim prices at fair odds,
so neither says whether the model's Over and Under picks made money at the Underdog and
Sleeper payouts they were posted at. This ledger prices each settled history offer at its
platform payout and drops sides the platform never posted (``Boost == 0``), which were
never bets. Pure; the caller persists it (``helpers.io.REALIZED_BY_SIDE_PATH``).
"""

from datetime import UTC, datetime
from itertools import pairwise

import numpy as np
import pandas as pd

from sportstradamus.analysis import annotate_offer_outcomes
from sportstradamus.helpers import UNDERDOG_BOOST_BASELINE

# 30 matches the graduation window; 90 gives the payout bands enough legs to read.
REALIZED_WINDOWS: tuple[int, ...] = (30, 90)
# The owner-locked story / "why" edge floor. Pinned equal to menu._MENU_EDGE_FLOOR by test
# rather than imported, since that constant is private to the story menu.
RECOMMENDED_EDGE_MIN: float = 0.05
# 2.5 is offer_records.MAX_FAVORED_PAYOUT; the bands show whether that cap should move.
PAYOUT_BAND_EDGES: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0)
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
# Pinned so an empty run writes the same parquet schema as a full one; the rest are object.
_DTYPES = {
    "computed_at": "datetime64[ns]",
    "window_days": "int16",
    "n": "int64",
    **dict.fromkeys(["hit_rate", "pred_rate", "book_rate", "units", "roi"], "float64"),
}


def _offers_by_split(history: pd.DataFrame) -> pd.DataFrame:
    """Settled, posted offers priced at the platform payout, one row per (offer x split).

    ``split`` names the grouping and ``key`` the offer's value in it. ``_unit`` is the
    realized profit on a 1-unit stake; ``_recommended`` flags a model edge at that payout
    of at least ``RECOMMENDED_EDGE_MIN``.
    """
    settled = annotate_offer_outcomes(history)
    offers = settled[
        settled["Result"].isin(("Over", "Under"))
        & settled["Bet"].notna()
        & settled["Win Prob"].notna()
        & (settled["Boost"] > 0)
    ].drop_duplicates(subset=_OFFER_KEY)
    # Underdog persists the raw multiplier over its flat baseline; Sleeper's Boost is
    # already the full decimal payout.
    payout = offers["Boost"] * np.where(
        offers["Platform"] == "Underdog", UNDERDOG_BOOST_BASELINE, 1.0
    )
    # Derived, not annotate's Hit: that column is absent when no row resolved.
    hit = offers["Bet"] == offers["Result"]
    split_keys = {
        "side": "all",
        "league": offers["League"],
        "platform": offers["Platform"],
        "alt_line": np.where(offers["Alt Line"].eq(True), "alt", "main"),
        "payout_band": pd.cut(payout, _PAYOUT_BAND_BINS, labels=_PAYOUT_BAND_LABELS).astype(str),
        "cell": offers["League"] + "/" + offers["Market"],
        "model_version": offers["Model Version"].fillna("legacy"),
    }
    return offers.assign(
        _date=pd.to_datetime(offers["Date"]),
        _hit=hit,
        _unit=hit * payout - 1,
        _recommended=offers["Win Prob"] * payout - 1 >= RECOMMENDED_EDGE_MIN,
        **split_keys,
    ).melt(
        id_vars=["_date", "_recommended", "Bet", "_hit", "Win Prob", "Market Prob", "_unit"],
        value_vars=list(split_keys),
        var_name="split",
        value_name="key",
    )


def compute_realized_by_side(history: pd.DataFrame, *, now: datetime | None = None) -> pd.DataFrame:
    """Realized hit rate and ROI per bet side, long over windows, cohorts and splits.

    An offer counts once per ``(Date, Player, Market, Line, Bet, Platform)`` when it
    settled Over/Under and the platform posted the chosen side. Cohort ``bettable`` is
    every such offer; ``recommended`` keeps those whose model edge at the platform payout,
    ``Win Prob x payout - 1``, is at least ``RECOMMENDED_EDGE_MIN``. ``split`` is one of
    side (key ``"all"``), league, platform, alt_line, payout_band, cell (``League/Market``)
    or model_version (NaN reads ``"legacy"``), and ``key`` the offer's value in it.

    Args:
        history: Flat prediction history (``history_schema.HISTORY_COLS``), ``Actual``
            filled by reflect. ``Win Prob`` and ``Market Prob`` are the model and book
            probabilities of the chosen side.
        now: Anchor for the trailing windows and the ``computed_at`` stamp; defaults to
            UTC now (stored tz-naive, like the live-metrics frame).

    Returns:
        ``REALIZED_BY_SIDE_COLS`` in order, one row per (window_days, cohort, split, key,
        side). ``hit_rate``, ``pred_rate`` and ``book_rate`` are the realized hit rate and
        the mean model and book probabilities; ``units`` is profit in 1-unit stakes at the
        platform payout and ``roi`` is ``units / n``. Empty, same dtypes, when nothing
        settles.
    """
    if history.empty:
        return pd.DataFrame(columns=REALIZED_BY_SIDE_COLS).astype(_DTYPES)
    by_split = _offers_by_split(history)
    # UTC, tz-stripped: the same anchor the live-metrics frame uses, so the two
    # nightly windows cover the same offers.
    now_ts = pd.Timestamp(now or datetime.now(UTC)).tz_localize(None)
    frames = []
    for window_days in REALIZED_WINDOWS:
        window = by_split[by_split["_date"] >= now_ts - pd.Timedelta(days=window_days)]
        for cohort, rows in (("bettable", window), ("recommended", window[window["_recommended"]])):
            stats = (
                rows.groupby(["split", "key", "Bet"])
                .agg(
                    n=("_hit", "size"),
                    hit_rate=("_hit", "mean"),
                    pred_rate=("Win Prob", "mean"),
                    book_rate=("Market Prob", "mean"),
                    units=("_unit", "sum"),
                )
                .reset_index()
            )
            frames.append(stats.assign(window_days=window_days, cohort=cohort))
    out = pd.concat(frames, ignore_index=True).rename(columns={"Bet": "side"})
    out = out.assign(roi=out["units"] / out["n"], computed_at=now_ts)
    return out[REALIZED_BY_SIDE_COLS].astype(_DTYPES)
