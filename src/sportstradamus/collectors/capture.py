"""The store step the per-run lineup, weather and injury captures share."""

from __future__ import annotations

from collections.abc import Callable
from importlib.resources.abc import Traversable

import pandas as pd

from sportstradamus.helpers.io import write_capture_parquet
from sportstradamus.spiderLogger import logger


def capture_league(
    league: str,
    kind: str,
    build: Callable[[], pd.DataFrame],
    table_dir: Traversable,
    polled_at: pd.Timestamp,
) -> None:
    """Store the table ``build`` returns for one league; a failure is logged, never raised.

    Diagnostics: nothing that goes wrong in a capture may cost the board, or another league's
    capture, its run. So ``build`` (requests included) runs inside the isolation, a table with no
    rows writes no file, and any ``Exception`` is an ERROR with its traceback, "<league> <kind>
    capture not written". The file is :func:`write_capture_parquet`'s, under ``table_dir/<league>``.
    """
    try:
        rows = build()
        if not rows.empty:
            write_capture_parquet(rows, table_dir / league, polled_at)
    except Exception:
        logger.exception(f"{league} {kind} capture not written")
