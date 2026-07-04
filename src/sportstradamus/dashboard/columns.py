"""Scoring-column semantics — the one home for what the snapshot's score columns mean.

You play against DFS (Underdog payout table / Sleeper boost), not against the book. So
``Model EV`` (``Win % × Boost``) is already your expected return per $1 staked on the app,
and the edge you capture is ``Model EV − 1`` (**Model Edge**). ``Market EV`` is the same
payout scored with the consensus book's probability, so ``Market EV − 1`` (**Consensus
Edge**) is how soft the book thinks the DFS line is — above 0 it agrees, below 0 you're
contrarian. This module relabels ``Win Prob`` to **Win %**, attaches header tooltips, and
derives the two edge columns. Display-only: the raw snapshot columns are unchanged.
"""

from __future__ import annotations

import pandas as pd

from sportstradamus.dashboard.narrative import match_label
from sportstradamus.helpers import market_display_name

MODEL_EDGE = "Model Edge"
CONSENSUS_EDGE = "Consensus Edge"
_EV_BREAK_EVEN = 1.0

LABELS = {"Win Prob": "Win %", CONSENSUS_EDGE: "Cons Edge"}

HELP = {
    "Win %": "Model's probability the pick hits.",
    MODEL_EDGE: "Your edge vs the DFS payout: Model EV − 1 (Win % × Boost − 1). +6% means $1 "
    "returns $1.06 on average against the app. This is what Kelly sizes.",
    "Cons Edge": "The consensus book's edge at the same DFS payout: Market EV − 1. Above 0% "
    "the book agrees the line is soft; below 0% the book disagrees (you're contrarian).",
    "Kelly": "Kelly edge — the bankroll fraction full-Kelly would stake on this leg.",
}


def _edge(df: pd.DataFrame, ev_col: str) -> pd.Series:
    return pd.to_numeric(df[ev_col], errors="coerce") - _EV_BREAK_EVEN


def add_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Append the derived ``Model Edge`` (``Model EV − 1``) and ``Consensus Edge``
    (``Market EV − 1``) columns, each only when its source EV column is present.
    """
    df = df.copy()
    if "Model EV" in df.columns:
        df[MODEL_EDGE] = _edge(df, "Model EV")
    if "Market EV" in df.columns:
        df[CONSENSUS_EDGE] = _edge(df, "Market EV")
    return df


def add_match_column(df: pd.DataFrame) -> pd.DataFrame:
    """Append ``Match`` (``"LVA v IND"`` / ``"LVA @ IND"``), the player-team-first
    matchup label ``match_label`` builds from ``Team``/``Opponent``/``Home``.
    """
    df = df.copy()
    df["Match"] = [
        match_label(t, o, h)
        for t, o, h in zip(df["Team"], df["Opponent"], df["Home"], strict=True)
    ]
    return df


def add_market_display(df: pd.DataFrame) -> pd.DataFrame:
    """Append ``Market Display``, the prose label for ``Market``'s slug. ``Market``
    itself is left unchanged — offer matching and lens/filter logic key off the slug.
    """
    df = df.copy()
    df["Market Display"] = [
        market_display_name(lg, m)
        for lg, m in zip(df["League"], df["Market"], strict=True)
    ]
    return df
