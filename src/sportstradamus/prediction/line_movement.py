"""Per-offer trajectory of the DFS book's own posted line.

Collapses the archive's raw ``(book, observed_at, line)`` observations into one
row per offer: where the number opened and closed, how far it travelled, how
many times it actually changed, and a fixed-width trajectory the Board grid
renders as a sparkline. A posted line is a step function — a book holds its
number until it reposts — so the trajectory carries the last known value
forward instead of interpolating between observations.

Pure aggregation: it opens no archive and writes no file. The caller reads the
observations via ``Archive.get_book_line_histories`` and persists the result.
The snapshot's schema lives with its path in ``helpers.io`` so the dashboard
reader and this producer cannot drift apart.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sportstradamus.helpers.io import LINE_MOVEMENT_COLS, LINE_MOVEMENT_KEYS

# Sparkline width the Board grid renders. Fixed so every row plots on the same
# horizontal scale however often its book happened to be polled.
_MOVEMENT_POINTS = 12

_ARCHIVE_TO_OFFER = {
    "league": "League",
    "book": "Platform",
    "market": "Market",
    "entity": "Player",
}


def build_line_movement(observations: pd.DataFrame, offers: pd.DataFrame) -> pd.DataFrame:
    """Summarize each offer's line history into one joinable row.

    Args:
        observations: ``Archive.get_book_line_histories`` output — columns
            ``league, market, game_date, entity, book, observed_at, line``.
        offers: The scored slate. Only its ``(League, Platform, Market, Player,
            Date)`` identity is read; observations outside it are dropped.

    Returns:
        One row per offer key the archive actually observed, with columns
        ``LINE_MOVEMENT_COLS``. ``series`` is a JSON list of floats: up to
        ``_MOVEMENT_POINTS`` samples spaced evenly *in time* (confer slots are
        not evenly spaced, so even-in-index would distort the shape) between
        ``first_seen`` and ``last_seen``, each carrying the last line posted at
        or before that instant. Its first and last entries are ``open_line`` and
        ``close_line``. An offer seen once yields a one-element series and a
        ``move`` of 0.0. Column-stable and empty when either input is.
    """
    if observations.empty or offers.empty:
        return pd.DataFrame(columns=LINE_MOVEMENT_COLS)

    obs = observations.rename(columns=_ARCHIVE_TO_OFFER)
    # game_date arrives as a DuckDB DATE; offers.Date is an ISO string.
    obs["Date"] = pd.to_datetime(obs["game_date"]).dt.strftime("%Y-%m-%d")
    obs = obs.merge(offers[LINE_MOVEMENT_KEYS].drop_duplicates(), on=LINE_MOVEMENT_KEYS)
    if obs.empty:
        return pd.DataFrame(columns=LINE_MOVEMENT_COLS)

    rows = [_movement_row(key, group) for key, group in obs.groupby(LINE_MOVEMENT_KEYS, sort=False)]
    return pd.DataFrame(rows, columns=LINE_MOVEMENT_COLS)


def _movement_row(key: tuple, group: pd.DataFrame) -> dict:
    group = group.sort_values("observed_at")
    lines = group["line"].to_numpy(dtype=float)
    stamps = pd.to_datetime(group["observed_at"])
    first_seen = stamps.iloc[0]

    # Elapsed seconds, not raw epoch nanoseconds: an int64 timestamp exceeds
    # float64's exact-integer range, and a tick rounding below the first
    # observation would make searchsorted wrap to the end of the series.
    elapsed = (stamps - first_seen).dt.total_seconds().to_numpy()
    ticks = np.linspace(0.0, elapsed[-1], min(len(lines), _MOVEMENT_POINTS))
    sampled = lines[np.searchsorted(elapsed, ticks, side="right") - 1]

    return {
        **dict(zip(LINE_MOVEMENT_KEYS, key, strict=True)),
        "open_line": float(lines[0]),
        "close_line": float(lines[-1]),
        "move": float(lines[-1] - lines[0]),
        "n_moves": int(np.count_nonzero(np.diff(lines))),
        "first_seen": first_seen,
        "last_seen": stamps.iloc[-1],
        "series": json.dumps([float(value) for value in sampled]),
    }
