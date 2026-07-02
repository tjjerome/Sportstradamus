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

import pandas as pd
import pytest

from sportstradamus.helpers.io import MODEL_STATS_PATH
from sportstradamus.training.graduation import served_cells_failing_ship
from sportstradamus.training.ship_config import STAT_META_PATH, load_stat_meta


def test_no_served_cell_fails_ship_gate():
    stats_path = Path(str(MODEL_STATS_PATH))
    if not stats_path.is_file():
        pytest.skip("model_stats.parquet absent (runtime-recomputed); invariant unverifiable here")
    meta = load_stat_meta(Path(str(STAT_META_PATH)))

    # A dev box may carry a partial, single-league research parquet (e.g. a WNBA Ship-75
    # confirm run) rather than a full meditate snapshot. served_cells_failing_ship treats
    # any row it sees as authoritative -- correct for its other caller
    # (generate-ship-config, always run against a full server snapshot) but wrong here: a
    # league with zero rows at all is missing evidence, not passing evidence, so only
    # enforce once every league with a shipped cell actually has coverage.
    shipped_leagues = {
        league
        for league, markets in meta.items()
        for cell in markets.values()
        if cell.get("shipped") in ("devel", "main")
    }
    present_leagues = set(pd.read_parquet(stats_path, columns=["league"])["league"])
    missing = shipped_leagues - present_leagues
    if missing:
        pytest.skip(
            "model_stats.parquet does not cover every league with a shipped cell "
            f"(missing {sorted(missing)}) -- a partial local research snapshot, not a "
            "full meditate run; invariant unverifiable here"
        )

    failing = served_cells_failing_ship(meta, stats_path)
    assert not failing, (
        "cells marked shipped devel/main but failing the ship gate "
        f"(demote to withheld in stat_meta.json): {failing}"
    )
