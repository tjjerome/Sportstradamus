"""Board prophecy lenses — one-click preset filter views over the offers frame.

"Picks" is the board's picks (``Pick``, set by ``live_fit.picks``) and the Board's default.
The rest are views of a leg's standing. You play against DFS, so both columns they read are
returns per $1 at the app's payout: ``EV`` from the fitted win chance and ``Market EV`` from
the consensus book's. The app's cut leaves ``EV`` under 1.00 on nearly every leg, so the
lenses cut where the settled record separated legs, not at break-even
(docs/handoffs/profit-engine.md §5.3, "Lenses"). They sort offers by standing — the fit's
best legs, legs paying double or more, and whether the book agrees with the fit (Consensus)
or prices the leg well under it (Contrarian) — before the manual filters apply. Pure:
``apply_lens(df, lens)`` returns a filtered frame; the registry order is the control's
display order.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from sportstradamus.helpers import platform_payout

# "Sharp edges": about the fit's top leg in twenty...
_SHARP_EV = 0.93
# ...that the book's own price also puts near the top.
_SHARP_MARKET_EV = 0.95
# The two book lenses read further down the fit's ranking, about its top leg in ten.
_RANKED_EV = 0.92
_EV_BREAK_EVEN = 1.0
# "Contrarian": the book's return sits at least four points under the fit's.
_CONTRARIAN_MARKET_EV = 0.88
# "Longshots": the side pays at least double.
_LONGSHOT_PAYOUT = 2.0
# Sides that pay double rank lower on EV as a group, so their floor sits lower.
_LONGSHOT_EV = 0.86
_LENS_COLUMNS = {"EV", "Market EV", "Consensus Line", "Boost", "Platform"}


def _sharp(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["EV"] >= _SHARP_EV) & (df["Market EV"] >= _SHARP_MARKET_EV)]


def _longshots(df: pd.DataFrame) -> pd.DataFrame:
    payout = platform_payout(df["Boost"], df["Platform"])
    return df[(payout >= _LONGSHOT_PAYOUT) & (df["EV"] >= _LONGSHOT_EV)]


def _contrarian(df: pd.DataFrame) -> pd.DataFrame:
    """A sportsbook prices the leg well under break-even and the fit still ranks it high.

    Without a book line ``Market EV`` is the app's own price, which says nothing against
    the leg, so those legs stay out.
    """
    return df[
        (df["EV"] >= _RANKED_EV)
        & df["Consensus Line"].notna()
        & (df["Market EV"] < _CONTRARIAN_MARKET_EV)
    ]


def _consensus(df: pd.DataFrame) -> pd.DataFrame:
    """The book itself scores the leg over break-even, and the fit ranks it high."""
    return df[(df["EV"] >= _RANKED_EV) & (df["Market EV"] > _EV_BREAK_EVEN)]


def _tonight(df: pd.DataFrame) -> pd.DataFrame:
    """The soonest slate date in the frame — avoids wall-clock/timezone parsing
    entirely and degrades gracefully when the board spans more than one date.
    """
    if "Date" not in df.columns:
        return df
    return df[df["Date"] == df["Date"].min()]


LENSES: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "Picks": lambda df: df[df["Pick"]],
    "All": lambda df: df,
    "Tonight": _tonight,
    "Sharp edges": _sharp,
    "Longshots": _longshots,
    "Contrarian": _contrarian,
    "Consensus": _consensus,
}


def apply_lens(df: pd.DataFrame, lens: str) -> pd.DataFrame:
    """Apply a named lens; an unknown lens or a missing lens column leaves the frame unchanged."""
    fn = LENSES.get(lens)
    if fn is None or not _LENS_COLUMNS.issubset(df.columns):
        return df
    return fn(df)
