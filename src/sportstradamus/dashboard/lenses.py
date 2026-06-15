"""Board prophecy lenses — one-click preset filter views over the offers frame.

Each lens is a celestial-named preset that narrows the board to a betting stance
(sharp edges, longshots, model-vs-market disagreement or agreement) before the manual
filters apply. Pure: ``apply_lens(df, lens)`` returns a filtered frame; the registry
order is the control's display order. Divergence keys off the EV columns on the same
scale (``Model EV − Market EV`` via ``columns.edge_series``), never the probability.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from sportstradamus.dashboard.columns import edge_series

# "Sharp edges": tuned vs bankroll ROI — model EV over break-even with the market not
# strongly against it. A boost at/above this signals a line move, not genuine edge, so
# it doubles as the "Longshots" floor (the high-payout lottery sharp edges exclude).
_SHARP_EV = 1.02
_SHARP_BOOKS = 0.95
_LINE_MOVE_BOOST = 2.5
# Model-vs-market EV gap that counts as a disagreement; below it (with the market also
# positive) the two agree.
_DIVERGENCE_GAP = 0.10
# Market EV over break-even by this much ⇒ the market also rates the pick positive.
_MARKET_POSITIVE = 1.02
_EV_BREAK_EVEN = 1.0


def _sharp(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["Model EV"] > _SHARP_EV)
        & (df["Market EV"] > _SHARP_BOOKS)
        & (df["Boost"] <= _LINE_MOVE_BOOST)
    ]


def _longshots(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["Boost"] >= _LINE_MOVE_BOOST) & (df["Model EV"] > _EV_BREAK_EVEN)]


def _contrarian(df: pd.DataFrame) -> pd.DataFrame:
    return df[(edge_series(df) >= _DIVERGENCE_GAP) & (df["Model EV"] > _EV_BREAK_EVEN)]


def _consensus(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["Market EV"] > _MARKET_POSITIVE) & (edge_series(df).abs() <= _DIVERGENCE_GAP)]


LENSES: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "All": lambda df: df,
    "Sharp edges": _sharp,
    "Longshots": _longshots,
    "Contrarian": _contrarian,
    "Consensus": _consensus,
}


def apply_lens(df: pd.DataFrame, lens: str) -> pd.DataFrame:
    """Apply a named lens; an unknown lens or missing EV columns leave the frame unchanged."""
    fn = LENSES.get(lens)
    if fn is None or not {"Model EV", "Market EV", "Boost"}.issubset(df.columns):
        return df
    return fn(df)
