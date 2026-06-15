"""Characterization for ``analysis.backfill_crps`` (precomputes the dashboard CRPS column).

CRPS is a pure function of a prediction's distribution and its realized ``Actual``, so the
nightly resolve fills it once and the diagnostics page only aggregates the column. This
pins: (1) filled values equal ``compute_crps_row`` per row, (2) rows missing ``Dist`` or
``Actual`` stay NaN, (3) the call is idempotent (no rewrite signal once every resolvable
row is filled), and (4) a frame without the distribution columns is a no-op.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sportstradamus.analysis import backfill_crps, compute_crps_row


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Dist": "NegBin",
                "CV": 0.5,
                "Projection": 6.0,
                "Model Param": 3.0,
                "Gate": np.nan,
                "Actual": 7.0,
            },
            {
                "Dist": "Gamma",
                "CV": 0.4,
                "Projection": 10.0,
                "Model Param": 5.0,
                "Gate": np.nan,
                "Actual": 12.0,
            },
            {
                "Dist": np.nan,
                "CV": 0.5,
                "Projection": 6.0,
                "Model Param": 3.0,
                "Gate": np.nan,
                "Actual": 5.0,
            },
            {
                "Dist": "NegBin",
                "CV": 0.5,
                "Projection": 6.0,
                "Model Param": 3.0,
                "Gate": np.nan,
                "Actual": np.nan,
            },
        ]
    )


def test_backfill_crps_matches_per_row_and_skips_unresolved():
    hist = _history()
    assert backfill_crps(hist) is True

    assert hist.loc[0, "CRPS"] == compute_crps_row(_history().loc[0])
    assert hist.loc[1, "CRPS"] == compute_crps_row(_history().loc[1])
    assert np.isfinite(hist.loc[0, "CRPS"])
    assert np.isfinite(hist.loc[1, "CRPS"])

    assert np.isnan(hist.loc[2, "CRPS"])  # missing Dist
    assert np.isnan(hist.loc[3, "CRPS"])  # unresolved Actual


def test_backfill_crps_is_idempotent():
    hist = _history()
    assert backfill_crps(hist) is True
    before = hist["CRPS"].to_list()

    assert backfill_crps(hist) is False  # nothing left to fill -> no parquet rewrite
    pd.testing.assert_series_equal(pd.Series(before), pd.Series(hist["CRPS"].to_list()))


def test_backfill_crps_no_distribution_columns_is_noop():
    assert backfill_crps(pd.DataFrame({"Foo": [1, 2]})) is False
