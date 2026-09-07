"""Scoring-column semantics — the one home for what the snapshot's score columns mean.

You play against DFS (Underdog payout table / Sleeper boost), not against the book. So
``Model EV`` (``Win % × Boost``) is already your expected return per $1 staked on the app,
and the edge you capture is ``Model EV − 1`` (**Model Edge**). ``Market EV`` is the same
payout scored with the consensus book's probability, so ``Market EV − 1`` (**Consensus
Edge**) is how soft the book thinks the DFS line is — above 0 it agrees, below 0 you're
contrarian. This module relabels ``Win Prob`` to **Win %**, attaches header tooltips, and
derives the two edge columns plus the Board's matchup, market-name, and line-movement
columns. Display-only: the raw snapshot columns are unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import pandas as pd

from sportstradamus.dashboard.components.spark_svg import movement_svg
from sportstradamus.dashboard.narrative import match_label
from sportstradamus.helpers import market_display_name
from sportstradamus.helpers.io import LINE_MOVEMENT_KEYS

MODEL_EDGE = "Model Edge"
CONSENSUS_EDGE = "Consensus Edge"
MOVE = "Move"
MOVE_SPARK = "Move Spark"
MOVE_TEXT = "Move Text"
_EV_BREAK_EVEN = 1.0

LABELS = {"Win Prob": "Win %", CONSENSUS_EDGE: "Cons Edge", "Market Display": "Market"}

HELP = {
    "Win %": "Model's probability the pick hits.",
    MODEL_EDGE: "Your edge vs the DFS payout: Model EV − 1 (Win % × Boost − 1). +6% means $1 "
    "returns $1.06 on average against the app. This is what Kelly sizes.",
    "Cons Edge": "The consensus book's edge at the same DFS payout: Market EV − 1. Above 0% "
    "the book agrees the line is soft; below 0% the book disagrees (you're contrarian).",
    "Kelly": "Kelly edge — the bankroll fraction full-Kelly would stake on this leg.",
    MOVE: "How far this row's Platform has moved its own line since the offer opened — the "
    "DFS book you'd bet, not the consensus. The trace runs green when the line moved toward "
    "your side and red when it moved away; a flat gray rule means the app never moved it.",
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
    ``Home`` defaults to away (``False``) when the column is absent, matching
    ``narrative.py``'s existing guard for the same optional-shaped column.
    """
    df = df.copy()
    home = df["Home"] if "Home" in df.columns else pd.Series(False, index=df.index)
    df["Match"] = [
        match_label(t, o, h) for t, o, h in zip(df["Team"], df["Opponent"], home, strict=True)
    ]
    return df


def _move_text(trajectory: Sequence[float]) -> str:
    """``"209.5 → 213.5"`` for the phone card, or ``""`` when it ended where it started.

    A round trip reads as no movement here on purpose: "2.5 → 2.5" spends a line of a
    390px card to say nothing. The desktop trace still draws the journey.
    """
    if len(trajectory) < 2 or trajectory[0] == trajectory[-1]:
        return ""
    return f"{trajectory[0]:.10g} → {trajectory[-1]:.10g}"


def add_line_movement(df: pd.DataFrame, movement: pd.DataFrame) -> pd.DataFrame:
    """Append the DFS book's line-movement columns from a ``current_line_movement`` frame.

    ``Move`` is the signed delta — the grid's cell *value*, so the column click-sorts —
    ``Move Spark`` the finished sparkline SVG its cellRenderer draws, and ``Move Text``
    the phone card's plain-text delta. An offer with no movement row gets NaN and empty
    strings, which every surface renders as no movement at all; on a same-day slate that
    is most of the board.
    """
    df = df.copy()
    keyed = movement.set_index(LINE_MOVEMENT_KEYS)
    joined = df.join(keyed[["move", "n_moves", "series"]], on=LINE_MOVEMENT_KEYS)
    df[MOVE] = pd.to_numeric(joined["move"], errors="coerce")
    # n_moves, not the net delta, is what tells a line that was never repriced from one
    # that wandered and came back — see movement_svg. Missing rows count as never.
    moves = pd.to_numeric(joined["n_moves"], errors="coerce").fillna(0).astype(int)
    sparks, texts = [], []
    for raw, bet, n_moves in zip(joined["series"], df["Bet"], moves, strict=True):
        trajectory = json.loads(raw) if isinstance(raw, str) else []
        sparks.append(movement_svg(trajectory, bet=bet, n_moves=n_moves))
        texts.append(_move_text(trajectory))
    df[MOVE_SPARK] = sparks
    df[MOVE_TEXT] = texts
    return df


def add_market_display(df: pd.DataFrame) -> pd.DataFrame:
    """Append ``Market Display``, the prose label for ``Market``'s slug. ``Market``
    itself is left unchanged — offer matching and lens/filter logic key off the slug.
    """
    df = df.copy()
    df["Market Display"] = [
        market_display_name(lg, m) for lg, m in zip(df["League"], df["Market"], strict=True)
    ]
    return df
