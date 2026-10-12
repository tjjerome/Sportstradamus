"""Tests for Phase 3 Step 1.b CLV bring-up.

Covers the per-segment Kelly shrinkage getter and the segments parquet
:func:`clv.persist_segments` writes.
"""

from __future__ import annotations

import pandas as pd
import pytest

from sportstradamus import clv


@pytest.fixture(autouse=True)
def _isolated_segments_parquet(tmp_path, monkeypatch):
    """Redirect segment persistence to a tmp_path so tests don't fight."""
    target = tmp_path / "clv_segments.parquet"
    monkeypatch.setattr(clv, "_segments_parquet_path", lambda: target)
    return target


def test_persist_segments_writes_parquet(_isolated_segments_parquet):
    rows = []
    # 25 over-rate-1 legs in NBA/points (segment beats close every time).
    for _ in range(25):
        rows.append(
            {
                "Player": "P",
                "League": "NBA",
                "Team": "BOS",
                "Date": "2026-05-04",
                "Market": "points",
                "Platform": "Underdog",
                "Line": 18.5,
                "Boost": 1.0,
                "Bet": "Over",
                "Win Prob": 0.6,
                "Market Prob": 0.5,
                "Close Market Prob": 0.7,
                "Market CLV": 0.2,
                "Model CLV": 0.1,
            }
        )
    df = pd.DataFrame(rows)
    summary = clv.summarize(df)
    assert summary["n"] == 25
    # summarize does not auto-persist; reflect calls persist_segments
    # explicitly with the full per-segment frame.
    assert not _isolated_segments_parquet.exists()
    clv.persist_segments(summary["all_segments"])
    assert _isolated_segments_parquet.exists()

    seg_df = pd.read_parquet(_isolated_segments_parquet)
    assert {"League", "Market", "Platform", "n", "market_clv", "frac_beat_close"} <= set(
        seg_df.columns
    )
    row = seg_df.loc[(seg_df["League"] == "NBA") & (seg_df["Market"] == "points")].iloc[0]
    assert int(row["n"]) == 25
    assert row["frac_beat_close"] == pytest.approx(1.0)


def test_get_segment_calibration_returns_fallback_when_missing(
    _isolated_segments_parquet,
):
    # No parquet on disk yet.
    shrink, n = clv.get_segment_calibration("NBA", "points")
    assert shrink == 1.0
    assert n == 0


def test_get_segment_calibration_remaps_frac_beat(_isolated_segments_parquet):
    pd.DataFrame(
        [
            {
                "League": "NBA",
                "Market": "points",
                "Platform": "Underdog",
                "n": 40,
                "market_clv": 0.05,
                "frac_beat_close": 0.75,
            }
        ]
    ).to_parquet(_isolated_segments_parquet, index=False)
    shrink, n = clv.get_segment_calibration("NBA", "points")
    # 0.75 → 2*(0.75-0.5) = 0.5
    assert shrink == pytest.approx(0.5)
    assert n == 40


def test_get_segment_calibration_clamps_below_half(_isolated_segments_parquet):
    pd.DataFrame(
        [
            {
                "League": "NBA",
                "Market": "points",
                "Platform": "Underdog",
                "n": 40,
                "market_clv": -0.05,
                "frac_beat_close": 0.30,
            }
        ]
    ).to_parquet(_isolated_segments_parquet, index=False)
    shrink, n = clv.get_segment_calibration("NBA", "points")
    assert shrink == 0.0
    assert n == 40
