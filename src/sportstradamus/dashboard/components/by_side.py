"""Realized-by-side panel: the model's recommended Over and Under legs at the platforms' payouts.

Folded into the Receipts surface (``surfaces/receipts.py``) under the skeptic checks. The page
hands it ``realized.by_split`` of its window's recommended legs, so the sport switch, the
sidebar filters and the page window all reach it. It shows a headline Over / Under ROI pair
and a themed grid under one split. ``by_side_grid`` is the pure shaping step;
``render_by_side`` draws the split control and the grid.
"""

import pandas as pd
import streamlit as st

from sportstradamus.dashboard.components.grid import render_themed_grid
from sportstradamus.helpers import UNDERDOG_BOOST_BASELINE
from sportstradamus.realized import RECOMMENDED_EDGE_MIN, RECOMMENDED_PAYOUT_MAX

# ``by_split`` split -> control label, also the grid's first-column header. The side split
# (key "all") is labelled "All": its own name would collide with the Over / Under column.
_SPLIT_LABELS = {
    "side": "All",
    "league": "League",
    "platform": "Platform",
    "alt_line": "Alt line",
    "payout_band": "Payout band",
    "cell": "Cell",
    "model_version": "Model version",
    "quote": "Quote",
}
_SIDE_ORDER = ["Over", "Under"]
# Under a two-point claim over the book, Edge captured is a ratio of noise (real cells read
# −60 at a half-point floor).
_EDGE_CLAIM_MIN = 0.02
_EDGE_CAPTURED_HELP = (
    "The share of the model's claimed edge over the book that realized: (Hit − Book) / "
    "(Model − Book). 1 = the claim was right, 0 = the book was right."
)


def by_side_grid(stats: pd.DataFrame, split: str) -> pd.DataFrame:
    """``stats`` (a ``realized.by_split`` result) under ``split``, shaped for the themed grid.

    One row per (key, side), keys ascending and Over before Under. The first column is the
    split's label and holds the ``key``; ``Hit%`` / ``Breakeven%`` / ``Model%`` / ``Book%``
    / ``ROI`` are the fractions in percentage points. ``Edge captured`` is ``(hit_rate -
    book_rate) / (pred_rate - book_rate)``, NaN (a blank cell) where the model claims less
    than ``_EDGE_CLAIM_MIN`` over the book. Empty in, empty out with the same columns.
    """
    rows = stats[stats["split"] == split]
    rows = (
        rows.assign(side=pd.Categorical(rows["side"], _SIDE_ORDER))
        .sort_values(["key", "side"])
        .reset_index(drop=True)
    )
    claimed = rows["pred_rate"] - rows["book_rate"]
    return pd.DataFrame(
        {
            _SPLIT_LABELS[split]: rows["key"],
            "Side": rows["side"].astype(str),
            "Bets": rows["n"],
            # The themed grid's percent formatter expects percentage points, not fractions.
            "Hit%": rows["hit_rate"] * 100,
            "Breakeven%": rows["breakeven_rate"] * 100,
            "Model%": rows["pred_rate"] * 100,
            "Book%": rows["book_rate"] * 100,
            "Edge captured": ((rows["hit_rate"] - rows["book_rate"]) / claimed).where(
                claimed.abs() >= _EDGE_CLAIM_MIN
            ),
            "Units": rows["units"].round(1),
            "ROI": rows["roi"] * 100,
        }
    )


def render_by_side(stats: pd.DataFrame) -> None:
    """Split control, the headline Over / Under ROI pair and the themed grid over ``stats``."""
    if stats.empty:
        st.caption("No recommended legs settled in this window.")
        return
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
        f"payout, payouts capped at {RECOMMENDED_PAYOUT_MAX:g}x — one bet per platform, "
        "settled only, in the page window. Payout is the platform's real multiplier: Underdog "
        f"boost × {UNDERDOG_BOOST_BASELINE} baseline, Sleeper decimal; Breakeven% is the hit "
        f"rate those payouts need. Edge captured: {_EDGE_CAPTURED_HELP}"
    )

    headline = stats[stats["split"] == "side"]
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

    render_themed_grid(
        by_side_grid(stats, split),
        numeric_cols=[
            "Bets",
            "Hit%",
            "Breakeven%",
            "Model%",
            "Book%",
            "Edge captured",
            "Units",
            "ROI",
        ],
        heatmap_col="ROI",
        heatmap_center=0.0,
        header_help={"Edge captured": _EDGE_CAPTURED_HELP},
        percent_cols=["Hit%", "Breakeven%", "Model%", "Book%", "ROI"],
        decimal_cols=["Edge captured"],
        height=320,
        key="receipts_by_side_grid",
    )
