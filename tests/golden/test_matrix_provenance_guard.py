"""``_step_load_matrix`` must refuse a matrix whose odds-provenance block is half populated.

Cached rows predating the block concat against fresh rows carrying it, and ``pd.concat``
back-fills history with NaN. The guard has to fire before the caller persists, or the bad
matrix lands on disk and every later run on that cell raises too.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

import sportstradamus.training.pipeline as pipe
from sportstradamus.helpers.training_quotes import PROVENANCE_COLUMNS

_LEAGUE_START = date(2020, 1, 1)


def _matrix(**overrides) -> pd.DataFrame:
    columns = {
        "Player": ["A", "B"],
        "Date": ["2025-01-01", "2025-01-02"],
        "Result": [1.0, 2.0],
        "Line": [1.5, 1.5],
        "Odds": [0.45, 0.55],
        "EV": [1.4, 1.6],
        "Archived": [True, False],
        "Odds_synthetic": [False, True],
        "QuoteSource": ["book_direct", "neutral_fallback"],
        "QuoteAuthenticity": ["authentic", "synthetic"],
        "QuoteSyntheticReason": [None, "no_usable_book_probability"],
        "QuoteObservedAt": ["2024-12-31T12:00:00", None],
        "QuoteBookCount": [2, 0],
    }
    columns.update(overrides)
    return pd.DataFrame(columns)


def _load(frame: pd.DataFrame, tmp_path):
    stat_data = SimpleNamespace(get_training_matrix=lambda market, cutoff: frame)
    return pipe._step_load_matrix(
        "MLB_runs-allowed",
        _LEAGUE_START,
        stat_data,
        "MLB",
        "runs allowed",
        deterministic=False,
        force=True,
        need_model=True,
        full_rebuild=True,
        matrix_output=tmp_path,
    )


def test_half_populated_provenance_block_is_refused(tmp_path):
    half = _matrix()
    half.loc[0, ["QuoteSource", "QuoteAuthenticity", "QuoteBookCount"]] = None

    with pytest.raises(ValueError) as excinfo:
        _load(half, tmp_path)

    message = str(excinfo.value)
    assert "MLB runs allowed" in message
    assert "1 of 2 rows" in message
    # The remedy must be runnable as printed: the repair is a module, not a console script.
    assert "python -m sportstradamus.scripts.inject_backfilled_odds" in message


def test_fully_populated_and_legacy_matrices_both_load(tmp_path):
    populated, _ = _load(_matrix(), tmp_path)
    assert len(populated) == 2

    legacy = _matrix().drop(columns=list(PROVENANCE_COLUMNS))
    loaded, _ = _load(legacy, tmp_path)
    assert len(loaded) == 2
