"""Realized-by-side panel: the model's Over and Under recs priced at the platforms' payouts.

Folded into the Receipts surface (``surfaces/receipts.py``) under the skeptic checks. Reads
the long ledger ``realized.compute_realized_by_side`` writes nightly and shows its
``recommended`` cohort — the legs the model actually told the owner to bet — as a headline
Over / Under ROI pair plus a themed grid over one trailing window and one split.
``by_side_grid`` is the pure shaping step; ``render_by_side`` draws the controls and the grid.
"""

import pandas as pd
import streamlit as st

from sportstradamus.dashboard.components.grid import render_themed_grid
from sportstradamus.helpers import UNDERDOG_BOOST_BASELINE
from sportstradamus.realized import REALIZED_WINDOWS, RECOMMENDED_EDGE_MIN

# Only the recommended cohort is shown: bettable-but-unrecommended legs were never picks.
_COHORT = "recommended"
# Ledger ``split`` value -> control label, also the grid's first-column header. The side
# split (key "all") is labelled "All": its own name would collide with the Over / Under column.
_SPLIT_LABELS = {
    "side": "All",
    "league": "League",
    "platform": "Platform",
    "alt_line": "Alt line",
    "payout_band": "Payout band",
    "cell": "Cell",
    "model_version": "Model version",
}
_WINDOW_LABELS = {days: f"{days}d" for days in REALIZED_WINDOWS}
_SIDE_ORDER = ["Over", "Under"]


def by_side_grid(df: pd.DataFrame, window: int, split: str) -> pd.DataFrame:
    """The recommended cohort at ``window`` days under ``split`` (a ledger split value), one
    row per (key, side), keys ascending and Over before Under, shaped for the themed grid.

    The first column is the split's label and holds the ledger ``key``; ``Hit%`` /
    ``Model%`` / ``Book%`` / ``ROI`` are the ledger fractions in percentage points. Empty
    in, empty out with the same columns.
    """
    rows = df[(df["window_days"] == window) & (df["cohort"] == _COHORT) & (df["split"] == split)]
    rows = (
        rows.assign(side=pd.Categorical(rows["side"], _SIDE_ORDER))
        .sort_values(["key", "side"])
        .reset_index(drop=True)
    )
    return pd.DataFrame(
        {
            _SPLIT_LABELS[split]: rows["key"],
            "Side": rows["side"].astype(str),
            "Bets": rows["n"],
            # The themed grid's percent formatter expects percentage points, not fractions.
            "Hit%": rows["hit_rate"] * 100,
            "Model%": rows["pred_rate"] * 100,
            "Book%": rows["book_rate"] * 100,
            "Units": rows["units"].round(1),
            "ROI": rows["roi"] * 100,
        }
    )


def render_by_side(df: pd.DataFrame) -> None:
    """Window + split controls, the headline Over / Under ROI pair and the themed grid."""
    if df.empty:
        st.caption("The realized-by-side ledger appears after the first nightly `reflect`.")
        return
    default_window = REALIZED_WINDOWS[0]
    window = (
        st.segmented_control(
            "Window",
            list(_WINDOW_LABELS),
            default=default_window,
            format_func=_WINDOW_LABELS.get,
            key="receipts_by_side_window",
        )
        or default_window
    )
    split = (
        st.segmented_control(
            "Split",
            list(_SPLIT_LABELS),
            default="side",
            format_func=_SPLIT_LABELS.get,
            key="receipts_by_side_split",
        )
        or "side"
    )
    st.caption(
        f"Recommended legs only — model edge ≥ {RECOMMENDED_EDGE_MIN:.0%} at the platform "
        "payout — one bet per platform, settled only. Payout is the platform's real "
        f"multiplier: Underdog boost × {UNDERDOG_BOOST_BASELINE} baseline, Sleeper decimal."
    )

    headline = df[
        (df["window_days"] == window) & (df["cohort"] == _COHORT) & (df["split"] == "side")
    ]
    for column, side in zip(st.columns(2), _SIDE_ORDER, strict=True):
        row = headline.loc[headline["side"] == side]
        if row.empty:
            column.metric(f"{side} ROI", "—")
            continue
        column.metric(
            f"{side} ROI",
            f"{row['roi'].iloc[0]:+.1%}",
            f"{row['units'].iloc[0]:+.1f}u · {row['n'].iloc[0]:,} bets",
            delta_color="normal",  # negative units render red — losers are shown, never hidden
        )

    grid = by_side_grid(df, window, split)
    if grid.empty:
        st.caption("No rows in this slice.")
        return
    render_themed_grid(
        grid,
        numeric_cols=["Bets", "Hit%", "Model%", "Book%", "Units", "ROI"],
        heatmap_col="ROI",
        heatmap_center=0.0,
        percent_cols=["Hit%", "Model%", "Book%", "ROI"],
        height=320,
        key="receipts_by_side_grid",
    )
