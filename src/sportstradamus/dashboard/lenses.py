"""Board prophecy lenses — one-click preset filter views over the offers frame.

You play against DFS, so both EV columns are scored against the app's payout: ``Model EV``
is the model's return per $1 and ``Market EV`` the consensus book's return at the same
payout, each break-even at 1.00. The lenses sort offers by betting stance — your edge,
high-payout longshots, and whether the book agrees the line is soft (Consensus) or not
(Contrarian) — before the manual filters apply. Pure: ``apply_lens(df, lens)`` returns a
filtered frame; the registry order is the control's display order.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

_EV_BREAK_EVEN = 1.0
# "Sharp edges": model EV over this clears bankroll ROI; the book must not be against it.
_SHARP_EV = 1.02
_BOOK_NOT_AGAINST = 0.98
# A boost at/above this signals a line move, not genuine edge, so sharp edges exclude it
# and it doubles as the "Longshots" floor (the high-payout lottery).
_LINE_MOVE_BOOST = 2.5


def _sharp(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["Model EV"] > _SHARP_EV)
        & (df["Market EV"] > _BOOK_NOT_AGAINST)
        & (df["Boost"] <= _LINE_MOVE_BOOST)
    ]


def _longshots(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["Boost"] >= _LINE_MOVE_BOOST) & (df["Model EV"] > _EV_BREAK_EVEN)]


def _contrarian(df: pd.DataFrame) -> pd.DataFrame:
    """Model finds +EV vs DFS that the consensus book misses (book scores it under break-even)."""
    return df[(df["Model EV"] > _EV_BREAK_EVEN) & (df["Market EV"] < _EV_BREAK_EVEN)]


def _consensus(df: pd.DataFrame) -> pd.DataFrame:
    """Model and book agree the DFS line is soft — both score it over break-even."""
    return df[(df["Model EV"] > _EV_BREAK_EVEN) & (df["Market EV"] > _EV_BREAK_EVEN)]


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
