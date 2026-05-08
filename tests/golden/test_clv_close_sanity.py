"""Tests for Phase 3 Step 1.a/1.b CLV bring-up.

Covers the close-snapshot freshness warning, the per-segment Kelly
shrinkage getter, and the side-effect parquet emitted by
:func:`clv.summarize`.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from sportstradamus import clv


@pytest.fixture(autouse=True)
def _isolated_segments_parquet(tmp_path, monkeypatch):
    """Redirect segment persistence to a tmp_path so tests don't fight."""
    target = tmp_path / "clv_segments.parquet"
    monkeypatch.setattr(clv, "_segments_parquet_path", lambda: target)
    return target


def _commence(hour: int = 19) -> datetime:
    return datetime(2026, 5, 4, hour, 0, tzinfo=UTC)


def test_check_close_sample_freshness_emits_one_warning_per_segment(caplog):
    samples = pd.DataFrame(
        [
            # Two stale legs in the same segment → one warning.
            {
                "league": "NBA",
                "market": "points",
                "date": "2026-05-04",
                "commence_time": _commence(),
                "sample_ts": _commence() - timedelta(minutes=180),
            },
            {
                "league": "NBA",
                "market": "points",
                "date": "2026-05-04",
                "commence_time": _commence(),
                "sample_ts": _commence() - timedelta(minutes=240),
            },
            # Fresh leg → no warning even though same segment.
            {
                "league": "NBA",
                "market": "rebounds",
                "date": "2026-05-04",
                "commence_time": _commence(),
                "sample_ts": _commence() - timedelta(minutes=30),
            },
        ]
    )
    logger = logging.getLogger("test.clv_freshness")
    logger.setLevel(logging.WARNING)
    with caplog.at_level(logging.WARNING, logger="test.clv_freshness"):
        out = clv.check_close_sample_freshness(samples, logger=logger)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "NBA" in warnings[0].getMessage()
    assert "points" in warnings[0].getMessage()
    assert len(out) == 1
    assert int(out.iloc[0]["n"]) == 2


def test_check_close_sample_freshness_skips_missing_timestamps():
    samples = pd.DataFrame(
        [
            {
                "league": "NBA",
                "market": "points",
                "date": "2026-05-04",
                "commence_time": _commence(),
                "sample_ts": pd.NaT,
            }
        ]
    )
    out = clv.check_close_sample_freshness(samples)
    assert out.empty


def test_check_close_sample_freshness_handles_empty_frame():
    out = clv.check_close_sample_freshness(pd.DataFrame())
    assert out.empty


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
                "Offers": [(18.5, 1.0, "Underdog", "Over", 0.6, 0.5, 0.7, 0.2, 0.1)],
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
    row = seg_df.loc[
        (seg_df["League"] == "NBA") & (seg_df["Market"] == "points")
    ].iloc[0]
    assert int(row["n"]) == 25
    assert row["frac_beat_close"] == pytest.approx(1.0)


def test_get_segment_calibration_returns_fallback_when_missing(
    _isolated_segments_parquet,
):
    # No parquet on disk yet.
    shrink, n = clv.get_segment_calibration("NBA", "points")
    assert shrink == 1.0
    assert n == 0


def test_get_segment_calibration_low_n_returns_fallback(_isolated_segments_parquet):
    pd.DataFrame(
        [
            {
                "League": "NBA",
                "Market": "points",
                "Platform": "Underdog",
                "n": 5,
                "market_clv": 0.05,
                "frac_beat_close": 1.0,
            }
        ]
    ).to_parquet(_isolated_segments_parquet, index=False)
    shrink, n = clv.get_segment_calibration("NBA", "points")
    assert shrink == 1.0
    assert n == 5


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


def test_get_segment_calibration_aggregates_platforms(_isolated_segments_parquet):
    pd.DataFrame(
        [
            {
                "League": "NBA",
                "Market": "points",
                "Platform": "Underdog",
                "n": 30,
                "market_clv": 0.04,
                "frac_beat_close": 0.6,
            },
            {
                "League": "NBA",
                "Market": "points",
                "Platform": "Sleeper",
                "n": 30,
                "market_clv": 0.02,
                "frac_beat_close": 0.8,
            },
        ]
    ).to_parquet(_isolated_segments_parquet, index=False)
    shrink, n = clv.get_segment_calibration("NBA", "points")
    # Weighted mean of frac_beat = 0.7  → 2*(0.7-0.5) = 0.4
    assert n == 60
    assert shrink == pytest.approx(0.4)


def test_lookback_warn_threshold_constant_is_stable():
    assert clv.CLOSE_LOOKBACK_WARN_MINUTES == 90
    # Exactly at the threshold → not stale.
    samples = pd.DataFrame(
        [
            {
                "league": "NBA",
                "market": "points",
                "date": "2026-05-04",
                "commence_time": _commence(),
                "sample_ts": _commence() - timedelta(minutes=90),
            }
        ]
    )
    out = clv.check_close_sample_freshness(samples)
    assert out.empty
    assert not np.isnan(clv.CLOSE_LOOKBACK_WARN_MINUTES)
