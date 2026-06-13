"""Golden invariant: no cell marked ``shipped`` devel/main may fail the ship gate.

A served cell must clear all five offline gates (``ship == True`` in
``model_stats.parquet``; see ``training.scorecard``). ``meditate`` prunes the
pickle of any served cell that comes back failing, and
``generate-ship-config --branch devel`` refuses to validate while one exists.
This test pins the committed ``stat_meta.json`` against the same predicate so a
failing cell can never be marked served again.

``model_stats.parquet`` is runtime-recomputed (gitignored), so this skips in a
fresh CI checkout and bites where the parquet exists (dev machine, server).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sportstradamus.helpers.io import MODEL_STATS_PATH
from sportstradamus.training.graduation import served_cells_failing_ship
from sportstradamus.training.ship_config import STAT_META_PATH, load_stat_meta


def test_no_served_cell_fails_ship_gate():
    stats_path = Path(str(MODEL_STATS_PATH))
    if not stats_path.is_file():
        pytest.skip("model_stats.parquet absent (runtime-recomputed); invariant unverifiable here")
    meta = load_stat_meta(Path(str(STAT_META_PATH)))
    failing = served_cells_failing_ship(meta, stats_path)
    assert not failing, (
        "cells marked shipped devel/main but failing the ship gate "
        f"(demote to withheld in stat_meta.json): {failing}"
    )
