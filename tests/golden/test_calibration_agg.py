"""Golden pins for ``analysis.calibration_summary`` — the Receipts reliability agg.

Hand-built fixture with hand-computed ECE/ROI per split (see the arithmetic in each
test's comment); this is the same TDD style as ``test_profit_sim_precompute.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sportstradamus.analysis import JUICE_PAYOUT, calibration_summary

_CAL_SUMMARY_COLS = ["Alt Line", "Bin", "Predicted", "Actual", "N", "ECE", "ROI"]

# Standard split (Alt Line=False), 6 resolved rows across three bins. The third bin's
# prob (0.87) is deliberately off the 0.90 edge, not on it — np.arange(0.40, 1.01, 0.05)
# drifts that edge to ~0.8999999999999999, so a literal 0.90 value falls in the next
# bin up ((0.9, 0.95]) rather than (0.85, 0.9] as naive arithmetic would suggest.
#   bin (0.50, 0.55]: probs [0.52, 0.52, 0.55], hits [1, 0, 1] -> predicted=0.53, actual=2/3
#   bin (0.70, 0.75]: probs [0.72, 0.72],        hits [1, 1]    -> predicted=0.72, actual=1.0
#   bin (0.85, 0.90]: probs [0.87],              hits [0]       -> predicted=0.87, actual=0.0
_STD_PROBS = [0.52, 0.52, 0.55, 0.72, 0.72, 0.87]
_STD_HITS = [1, 0, 1, 1, 1, 0]

# Alt split (Alt Line=True), 4 resolved rows: one row (0.35) sits below the 0.40 bin
# floor so it drops out of binning but still counts toward the split's ROI. The other
# probs (0.82, 0.97) are deliberately off the arange bin edges for the same float-drift
# reason as the standard split's 0.87.
#   bin (0.80, 0.85]: probs [0.82, 0.82], hits [1, 0] -> predicted=0.82, actual=0.5
#   bin (0.95, 1.00]: probs [0.97],       hits [1]    -> predicted=0.97, actual=1.0
_ALT_PROBS = [0.35, 0.82, 0.82, 0.97]
_ALT_HITS = [1, 1, 0, 1]


def _rows(probs, hits, alt_line: bool, *, start_date="2026-06-01"):
    dates = pd.date_range(start_date, periods=len(probs)).strftime("%Y-%m-%d")
    return [
        {
            "Date": d,
            "Player": f"P{i}",
            "Market": "PTS",
            "League": "NBA",
            "Win Prob": p,
            "Bet": "Over",
            "Result": "Over" if h else "Under",
            "Alt Line": alt_line,
        }
        for i, (d, p, h) in enumerate(zip(dates, probs, hits, strict=True))
    ]


def _exploded(*, include_alt: bool = True, include_std: bool = True) -> pd.DataFrame:
    rows = []
    if include_std:
        rows += _rows(_STD_PROBS, _STD_HITS, alt_line=False)
    if include_alt:
        rows += _rows(_ALT_PROBS, _ALT_HITS, alt_line=True, start_date="2026-07-01")
    return pd.DataFrame(rows)


def test_columns_and_both_splits_present():
    df = calibration_summary(_exploded())
    assert list(df.columns) == _CAL_SUMMARY_COLS
    assert set(df["Alt Line"]) == {False, True}


def test_boundary_value_lands_in_lower_bin_not_upper():
    # 0.55 is the shared edge between (0.50, 0.55] and (0.55, 0.60]; pd.cut defaults to
    # right=True (half-open, left-exclusive/right-inclusive) so it must land in the lower bin.
    df = calibration_summary(_exploded())
    std = df.loc[~df["Alt Line"]].set_index("Bin")
    assert "(0.5, 0.55]" in std.index
    assert "(0.55, 0.6]" not in std.index
    row = std.loc["(0.5, 0.55]"]
    assert row["N"] == 3
    assert row["Predicted"] == pytest.approx(0.53, abs=1e-9)
    assert row["Actual"] == pytest.approx(2 / 3, abs=1e-9)


def test_standard_split_bin_values():
    df = calibration_summary(_exploded())
    std = df.loc[~df["Alt Line"]].set_index("Bin")
    assert len(std) == 3  # three populated bins, per the fixture design

    b72 = std.loc["(0.7, 0.75]"]
    assert b72["N"] == 2
    assert b72["Predicted"] == pytest.approx(0.72, abs=1e-9)
    assert b72["Actual"] == pytest.approx(1.0, abs=1e-9)

    b87 = std.loc["(0.85, 0.9]"]
    assert b87["N"] == 1
    assert b87["Predicted"] == pytest.approx(0.87, abs=1e-9)
    assert b87["Actual"] == pytest.approx(0.0, abs=1e-9)


def test_standard_split_ece_hand_computed():
    # N_binned = 3 + 2 + 1 = 6 (no sub-floor rows in the standard split here).
    # ECE = (3/6)*|0.53 - 2/3| + (2/6)*|0.72 - 1.0| + (1/6)*|0.87 - 0.0|
    expected = (3 / 6) * abs(0.53 - 2 / 3) + (2 / 6) * abs(0.72 - 1.0) + (1 / 6) * abs(0.87 - 0.0)
    df = calibration_summary(_exploded())
    std = df.loc[~df["Alt Line"]]
    # ECE is constant across a split's rows (one value per split, repeated per bin row).
    assert std["ECE"].nunique() == 1
    assert std["ECE"].iloc[0] == pytest.approx(expected, abs=1e-9)


def test_standard_split_roi_over_full_split_not_just_binned_rows():
    # All 6 standard rows clear the 0.40 floor, so ROI's denominator (6) matches the
    # binned-row count here; profits: hit -> +JUICE_PAYOUT, miss -> -1.
    profits = [(JUICE_PAYOUT if h else -1) for h in _STD_HITS]
    expected = sum(profits) / len(profits)
    df = calibration_summary(_exploded())
    std = df.loc[~df["Alt Line"]]
    assert std["ROI"].nunique() == 1
    assert std["ROI"].iloc[0] == pytest.approx(expected, abs=1e-9)


def test_alt_split_ece_denominator_excludes_sub_floor_row():
    # The 0.35 row clips to itself (already in [0,1]) but sits below the 0.40 bin floor,
    # so pd.cut assigns it no bin and groupby(observed=True) drops it from ECE's N.
    # N_binned = 2 (bin .80-.85) + 1 (bin .95-1.00) = 3, NOT 4.
    expected_ece = (2 / 3) * abs(0.82 - 0.5) + (1 / 3) * abs(0.97 - 1.0)
    df = calibration_summary(_exploded())
    alt = df.loc[df["Alt Line"]]
    assert alt["N"].sum() == 3
    assert alt["ECE"].nunique() == 1
    assert alt["ECE"].iloc[0] == pytest.approx(expected_ece, abs=1e-9)


def test_alt_split_roi_includes_the_sub_floor_row():
    # ROI's denominator is the full resolved split (4 rows), unlike ECE's binned-only 3 —
    # they are allowed to differ; this pins that they actually do for this fixture.
    profits = [(JUICE_PAYOUT if h else -1) for h in _ALT_HITS]
    expected_roi = sum(profits) / len(profits)  # denominator 4, not 3
    df = calibration_summary(_exploded())
    alt = df.loc[df["Alt Line"]]
    assert alt["ROI"].nunique() == 1
    assert alt["ROI"].iloc[0] == pytest.approx(expected_roi, abs=1e-9)
    assert alt["N"].sum() != len(_ALT_PROBS)  # ECE's N (3) != ROI's implicit denom (4)


def test_empty_split_contributes_no_rows_not_placeholder():
    # Every row is Alt Line=False; the True split must be absent entirely, not a
    # zero/NaN placeholder row.
    df = calibration_summary(_exploded(include_alt=False))
    assert set(df["Alt Line"]) == {False}
    # The False split itself is still computed correctly (matches the paired-fixture test).
    assert len(df) == 3


def test_empty_input_returns_empty_with_columns():
    df = calibration_summary(pd.DataFrame())
    assert df.empty
    assert list(df.columns) == _CAL_SUMMARY_COLS


def test_all_unresolved_returns_empty_with_columns():
    # Every row has Result == NaN (nothing settled yet) -> resolved-rows filter drops
    # everything before any aggregation runs.
    df = _exploded()
    df["Result"] = np.nan
    out = calibration_summary(df)
    assert out.empty
    assert list(out.columns) == _CAL_SUMMARY_COLS


def test_model_ev_fallback_used_when_win_prob_absent():
    # Same idiom as ev_threshold_record: falls back to Model EV when Win Prob is
    # missing/all-NaN. Model EV is Win Prob * payout and gets clipped to [0, 1] before
    # binning, mirroring receipts.py's existing Brier skeptic-check clip.
    df = _exploded()
    df = df.drop(columns=["Win Prob"])
    df["Model EV"] = [*_STD_PROBS, *_ALT_PROBS]
    out = calibration_summary(df)
    assert not out.empty
    assert set(out["Alt Line"]) == {False, True}
