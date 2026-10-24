"""Pin the contract of ``trim_matrix``: every step reads the line, never the outcome.

``_step_persist_matrix`` runs ``trim_matrix`` over the whole matrix on every incremental
append. The line clip applies only to the rows appended since the last trim (``new_rows``):
its bounds move with the population, so re-clipping cached rows cut a fresh slice of the
unquoted tail on every append. The floor-bounded line balancing still sees the union. No step
selects rows on ``Result``: the former 5th-95th percentile band and the over/under balancing
deleted stars' big games from the unquoted rows (truncation bias; nfl-ship15-recovery.md §11).
"""

import numpy as np
import pandas as pd

from sportstradamus.training.data import trim_matrix

# Larger than any synthetic matrix here, so the balancing has no budget and the line clip is
# the only step that can touch a row.
FLOOR = 10_000
# Small enough that the line balancing has budget on a 400-row matrix.
BALANCE_FLOOR = 200
# Unquoted results far above anything the Poisson(3) fixture produces: the stars' big games.
EXTREME_RESULTS = (40.0, 60.0)
# A synthetic line far outside the market's range, which only the line clip may touch.
EXTREME_LINE = 50.0


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


def _with_extreme_lines(M: pd.DataFrame) -> pd.DataFrame:
    """Give the first two unquoted rows a line the clip must pull back into range."""
    synthetic = M.index[~M["Archived"]][:2]
    M.loc[synthetic, "Line"] = EXTREME_LINE
    return M


def _with_tail_and_pushes(M: pd.DataFrame) -> pd.DataFrame:
    """Append five unquoted rows: two extreme results and three pushes (``Line == Result``)."""
    extra = M.iloc[:5].copy()
    extra["Player"] = [f"T{i}" for i in range(5)]
    extra["Archived"] = False
    extra["QuoteAuthenticity"] = "synthetic"
    extra.loc[extra.index[:2], "Result"] = EXTREME_RESULTS
    extra.loc[extra.index[2:], ["Result", "Line"]] = 3.0
    return pd.concat([M, extra], ignore_index=True)


def test_every_row_survives_below_the_floor_whatever_its_result():
    raw = _with_tail_and_pushes(_unquoted_heavy_matrix("P", seed=3))

    trimmed = trim_matrix(raw, FLOOR, seed=17)

    assert set(trimmed["Player"]) == set(raw["Player"])
    assert set(EXTREME_RESULTS) <= set(trimmed["Result"])


def test_kept_rows_do_not_depend_on_result():
    raw = _unquoted_heavy_matrix("P", seed=3)
    shuffled = raw.assign(Result=raw["Result"].sample(frac=1, random_state=0).to_numpy())

    kept = trim_matrix(raw, BALANCE_FLOOR, seed=17)
    kept_shuffled = trim_matrix(shuffled, BALANCE_FLOOR, seed=17)

    assert len(kept) < len(raw)
    assert kept["Player"].tolist() == kept_shuffled["Player"].tolist()


def test_retrim_with_nothing_new_keeps_every_cached_row():
    raw = _with_extreme_lines(_unquoted_heavy_matrix("P", seed=3))
    once = trim_matrix(raw, FLOOR, seed=17)
    assert (once.set_index("Player")["Line"] < EXTREME_LINE).all()

    twice = trim_matrix(once, FLOOR, seed=17, new_rows=pd.Series(False, index=once.index))

    pd.testing.assert_frame_equal(twice.reset_index(drop=True), once.reset_index(drop=True))


def test_appended_rows_are_clipped_beside_protected_cached_rows():
    cached = trim_matrix(_unquoted_heavy_matrix("P", seed=3), FLOOR, seed=17)
    fresh = _with_extreme_lines(_unquoted_heavy_matrix("F", seed=5))
    fresh["Date"] += pd.Timedelta(days=200)
    M = pd.concat([cached, fresh], ignore_index=True)
    new_rows = pd.Series([False] * len(cached) + [True] * len(fresh), index=M.index)

    trimmed = trim_matrix(M, FLOOR, seed=17, new_rows=new_rows).set_index("Player")

    pd.testing.assert_series_equal(
        trimmed.loc[cached["Player"], "Line"], cached.set_index("Player")["Line"]
    )
    assert (trimmed.loc[fresh["Player"], "Line"] < EXTREME_LINE).all()
