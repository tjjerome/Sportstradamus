"""Pin that re-trimming an already-trimmed matrix removes nothing from its cached rows.

``_step_persist_matrix`` runs ``trim_matrix`` over the whole matrix on every incremental
append. The Result band, the line clip and the push drop used to apply to every row, so each
pass cut a fresh slice of the unquoted tail: the band moves with the population, and a clipped
line that lands on its Result is a push the next pass drops. ``new_rows`` restricts those three
steps to the rows appended since the last trim; the floor-bounded balancing still sees the union.
"""

import numpy as np
import pandas as pd

from sportstradamus.training.data import trim_matrix

# Larger than any synthetic matrix here, so the balancing has no budget and the band, the
# clip and the push drop are the only steps that can remove rows.
FLOOR = 10_000


def _unquoted_heavy_matrix(prefix: str, seed: int, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    result = rng.poisson(3, n).astype(float)
    archived = rng.random(n) < 0.1
    return pd.DataFrame(
        {
            "Player": [f"{prefix}{i}" for i in range(n)],
            "Date": pd.Timestamp("2025-09-01") + pd.to_timedelta(rng.integers(0, 120, n), "D"),
            "Result": result,
            "Line": np.where(rng.random(n) < 0.5, result, result + 0.5),
            "Odds": 0.5,
            "EV": result,
            "Archived": archived,
            "DaysIntoSeason": rng.integers(0, 120, n),
            "MeanYr": rng.uniform(1, 8, n),
            "Player position": rng.integers(1, 4, n),
            "QuoteAuthenticity": np.where(archived, "authentic", "synthetic"),
        }
    )


def test_retrim_with_nothing_new_keeps_every_cached_row():
    raw = _unquoted_heavy_matrix("P", seed=3)
    once = trim_matrix(raw, FLOOR, seed=17)
    assert len(once) < len(raw)

    twice = trim_matrix(once, FLOOR, seed=17, new_rows=pd.Series(False, index=once.index))

    pd.testing.assert_frame_equal(twice.reset_index(drop=True), once.reset_index(drop=True))


def test_appended_rows_are_still_trimmed_beside_protected_cached_rows():
    cached = trim_matrix(_unquoted_heavy_matrix("P", seed=3), FLOOR, seed=17)
    fresh = _unquoted_heavy_matrix("F", seed=5)
    fresh["Date"] += pd.Timedelta(days=200)
    M = pd.concat([cached, fresh], ignore_index=True)
    new_rows = pd.Series([False] * len(cached) + [True] * len(fresh), index=M.index)

    trimmed = trim_matrix(M, FLOOR, seed=17, new_rows=new_rows)

    assert set(cached["Player"]) <= set(trimmed["Player"])
    assert len(trimmed) < len(M)
