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

from sportstradamus.dashboard.components.spark_svg import movement_summary, movement_svg
from sportstradamus.dashboard.narrative import match_label
from sportstradamus.helpers import market_display_name
from sportstradamus.helpers.io import LINE_MOVEMENT_KEYS

MODEL_EDGE = "Model Edge"
CONSENSUS_EDGE = "Consensus Edge"
MOVE = "Move"
MOVE_SPARK = "Move Spark"
MOVE_SUMMARY = "Move Summary"
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
    MOVE: "How far the fair line has moved since the offer opened, in the stat's own units, on "
    "this row's Platform — the DFS app you'd bet, not the consensus. The fair line is where "
    "the app's price would be even money, so a multiplier change counts even while the posted "
    "line holds. The trace runs green when it moved toward your side (down for an Over, up "
    "for an Under), red when it moved away, and gray when it came back; a flat gray rule "
    "means neither the line nor its price ever moved.",
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


def _move_text(posted: Sequence[float]) -> str:
    """The phone card's ``"Line `209.5 → 213.5` ▲"``, or ``""`` when the posted line netted zero.

    The arrow is the posted line's own direction: the fair line's can run the other way, and
    this text names the posted line. A round trip reads as no movement here on purpose —
    "2.5 → 2.5" spends a line of a 390px card to say nothing — and so does a price-only
    move, which the phone leaves to the desktop trace.
    """
    if posted[0] == posted[-1]:
        return ""
    arrow = "▲" if posted[-1] > posted[0] else "▼"
    return f"Line `{posted[0]:.10g} → {posted[-1]:.10g}` {arrow}"


def add_line_movement(df: pd.DataFrame, movement: pd.DataFrame) -> pd.DataFrame:
    """Append the DFS book's line-movement columns from a ``current_line_movement`` frame.

    The fair line drives every encoding: ``Move`` is its signed delta (``fair_move``) — the
    grid cell's *value*, so the column click-sorts on it — and ``Move Spark`` the sparkline
    SVG its cellRenderer draws, titled with ``Move Summary`` (:func:`movement_summary`),
    which also rides along on its own for surfaces that print the words. ``Move Text`` is
    the phone card's posted-line text. An offer with no movement row, or one from a
    snapshot written before the fair-line columns, gets NaN and empty strings, which every
    surface renders as no movement at all; on a same-day slate that is most of the board.
    An old row's posted-only numbers never stand in: they would draw every price move as
    held and sort ``Move`` by a different quantity than the rest of the board.

    Args:
        df: Offers carrying the ``LINE_MOVEMENT_KEYS`` columns and ``Bet``.
        movement: A ``current_line_movement`` frame with every ``LINE_MOVEMENT_COLS``
            column, as ``dashboard.data.load_current_line_movement`` returns it.

    Returns:
        A copy of ``df``, index unchanged, with ``Move``, ``Move Spark``, ``Move Summary``
        and ``Move Text`` appended.
    """
    df = df.copy()
    keyed = movement.set_index(LINE_MOVEMENT_KEYS)
    joined = df.join(
        keyed[["fair_move", "fair_series", "series", "n_moves", "n_price_moves"]],
        on=LINE_MOVEMENT_KEYS,
    )
    df[MOVE] = pd.to_numeric(joined["fair_move"], errors="coerce")
    sparks, summaries, texts = [], [], []
    for fair_json, posted_json, bet, n_moves, n_price_moves in zip(
        joined["fair_series"],
        joined["series"],
        df["Bet"],
        joined["n_moves"],
        joined["n_price_moves"],
        strict=True,
    ):
        spark = summary = text = ""
        if isinstance(fair_json, str):
            fair, posted = json.loads(fair_json), json.loads(posted_json)
            summary = movement_summary(
                fair, posted, n_moves=int(n_moves), n_price_moves=int(n_price_moves)
            )
            spark = movement_svg(
                fair, bet=bet, n_changes=int(n_moves + n_price_moves), title=summary
            )
            text = _move_text(posted)
        sparks.append(spark)
        summaries.append(summary)
        texts.append(text)
    df[MOVE_SPARK] = sparks
    df[MOVE_SUMMARY] = summaries
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
