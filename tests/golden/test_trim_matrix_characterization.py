"""Characterization pin for ``training.data.trim_matrix`` (CC 17).

``trim_matrix`` removes data-quality issues from a training matrix in several
phases — DaysIntoSeason rebasing, result-outlier trim, archived line-clip,
per-position line balancing, over/under balancing, and push removal — and uses
``np.random.choice`` for the stochastic cuts. There is no existing test (the only
caller is the ``meditate`` training pipeline, which integration tests stub).

These two fixtures freeze the full output byte-for-byte (seeded RNG → deterministic)
so the CC decomposition can be proven behavior-preserving. The exact-output SHA is
the master pin; the readable invariants make a failure diagnosable. Two inputs cover
the two threshold regimes:

* ``_dense_matrix`` (300 rows, 120 archived, two positions): archived line-clip
  branch, per-position balancing with real cuts, over/under balancing, push removal,
  and the archived-row outlier exemption (idx 5 is a Result outlier but kept).
* ``_sparse_matrix`` (60 rows, 3 archived, no "Player position"): the clip-fallback,
  the mean ``overall_target`` fallback, the whole-frame balance branch, and the
  ``n_archived < MIN_BALANCE`` early return.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from sportstradamus.training.data import trim_matrix


def _dense_matrix() -> pd.DataFrame:
    i = np.arange(300)
    days = i.astype(float).copy()
    days[:3] = [-5.0, -3.0, -1.0]
    days[-3:] = [305.0, 310.0, 320.0]
    result = (i % 37).astype(float)
    result[5] = 999.0
    result[7] = -999.0
    line = ((i % 31) * 0.5).astype(float)
    push_idx = np.where(i % 10 == 0)[0]
    line[push_idx] = result[push_idx]
    archived = np.where(i % 5 < 2, 1, 0)
    meanyr = (i % 13).astype(float) + 1.0
    pos = np.where(i < 150, "RB", "WR")
    return pd.DataFrame(
        {
            "DaysIntoSeason": days,
            "Result": result,
            "Line": line,
            "Archived": archived,
            "MeanYr": meanyr,
            "Player position": pos,
            "Date": pd.date_range("2024-01-01", periods=300, freq="D"),
        }
    )


def _sparse_matrix() -> pd.DataFrame:
    i = np.arange(60)
    days = i.astype(float).copy()
    days[:2] = [-4.0, -2.0]
    days[-2:] = [305.0, 330.0]
    result = (i % 23).astype(float)
    result[4] = 500.0
    line = ((i % 19) * 0.5).astype(float)
    push_idx = np.where(i % 11 == 0)[0]
    line[push_idx] = result[push_idx]
    archived = np.where(i < 3, 1, 0)
    meanyr = (i % 7).astype(float) + 1.0
    return pd.DataFrame(
        {
            "DaysIntoSeason": days,
            "Result": result,
            "Line": line,
            "Archived": archived,
            "MeanYr": meanyr,
            "Date": pd.date_range("2024-02-01", periods=60, freq="D"),
        }
    )


def _sha(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=True).encode()).hexdigest()


def test_trim_matrix_dense_exact():
    np.random.seed(0)
    out = trim_matrix(_dense_matrix(), min_rows=180)

    assert out.shape == (260, 7)
    # idx 7 (low non-archived outlier) trimmed; idx 5 (high outlier but Archived) kept
    assert list(out.index[:8]) == [0, 1, 2, 3, 4, 5, 6, 8]
    assert out["DaysIntoSeason"].min() == 0.0
    assert out["DaysIntoSeason"].max() == 296.0
    assert int((out["Archived"] == 1).sum()) == 120  # archived rows never dropped
    assert float(out["Result"].sum()) == 5639.0
    assert _sha(out) == "12f3c2229c82d7d7542fe21b8eb5515d41914699b327a6f443017b0807def9d1"


def test_trim_matrix_sparse_fallback_exact():
    np.random.seed(0)
    out = trim_matrix(_sparse_matrix(), min_rows=40)

    assert out.shape == (51, 6)
    assert "Player position" not in out.columns
    assert out["DaysIntoSeason"].min() == 0.0
    assert out["DaysIntoSeason"].max() == 57.0
    assert float(out["Result"].sum()) == 512.0
    assert _sha(out) == "ef2d1799f2955d0cd68701b5323a808b7aab73a4584e6aa03b8f58de32e5e575"
