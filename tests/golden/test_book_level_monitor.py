"""Report-only book level on a cell's test-half sportsbook rows (I6d, prereg §13(iv)).

``report._book_level_columns`` reads the test-set CSV's ``Odds`` (the book UNDER
probability), ``Line``, ``Result`` and quote provenance, and puts the book's
calibration-in-the-large, pooled and per levelling class, on the model_stats row. No gate
reads these columns.
"""

import math
import sys
from unittest import mock

import pandas as pd
import pytest

from sportstradamus.helpers.io import market_file_slug
from sportstradamus.training.model_strategy import resolve_report_identity
from sportstradamus.training.report import _book_level_columns, _layer_gates_from_test_set

report_module = sys.modules["sportstradamus.training.report"]

_CLASSES = ("native", "redecoded", "ladder")
_CLASS_COLUMNS = (
    *(f"book_citl_{name}" for name in _CLASSES),
    *(f"book_share_{name}" for name in _CLASSES),
)


def _levelled_frame() -> pd.DataFrame:
    """Two authentic rows per levelling class, with two pick'em rows interleaved.

    Book over = 1 - Odds, so the authentic gaps (book over - y) are -.40 and .45 for
    ``book_direct``, .70 and -.62 for ``book_redecoded``, -.48 and .30 for
    ``book_ladder_repaired``; the pick'em rows' -.10 and .10 must not count.
    """
    return pd.DataFrame(
        {
            "Odds": [0.40, 0.10, 0.30, 0.48, 0.90, 0.55, 0.62, 0.70],
            "Line": [4.5, 3.5, 2.5, 6.5, 3.5, 4.5, 2.5, 6.5],
            "Result": [6.0, 9.0, 1.0, 7.0, 0.0, 3.0, 4.0, 2.0],
            "QuoteAuthenticity": [
                "authentic",
                "pickem",
                "authentic",
                "authentic",
                "pickem",
                "authentic",
                "authentic",
                "authentic",
            ],
            "QuoteSource": [
                "book_direct",
                "book_direct",
                "book_redecoded",
                "book_ladder_repaired",
                "book_direct",
                "book_direct",
                "book_redecoded",
                "book_ladder_repaired",
            ],
        }
    )


def test_book_citl_splits_the_sportsbook_rows_by_levelling_class():
    out = _book_level_columns(_levelled_frame())

    assert out["book_citl_offline"] == pytest.approx(-0.05 / 6)
    assert out["book_citl_native"] == pytest.approx(0.025)
    assert out["book_citl_redecoded"] == pytest.approx(0.04)
    assert out["book_citl_ladder"] == pytest.approx(-0.09)
    for name in _CLASSES:
        assert out[f"book_share_{name}"] == pytest.approx(1 / 3)


def test_a_push_counts_as_half_an_over():
    frame = pd.DataFrame(
        {
            "Odds": [0.30, 0.50],
            "Line": [4.0, 2.5],
            "Result": [4.0, 3.0],
            "QuoteAuthenticity": "authentic",
            "QuoteSource": "book_direct",
        }
    )
    # .70 - .5 at the push and .50 - 1 on the over; reading the push as an under gives .10,
    # as an over -.40.
    assert _book_level_columns(frame)["book_citl_offline"] == pytest.approx(-0.15)


def test_a_class_without_rows_reads_zero_share_and_no_level():
    frame = _levelled_frame().assign(QuoteSource="book_direct")

    out = _book_level_columns(frame)

    assert out["book_share_native"] == pytest.approx(1.0)
    assert out["book_share_redecoded"] == 0.0
    assert math.isnan(out["book_citl_redecoded"])


def test_a_dump_without_quote_source_fills_only_the_pooled_level():
    out = _book_level_columns(_levelled_frame().drop(columns="QuoteSource"))

    assert out["book_citl_offline"] == pytest.approx(-0.05 / 6)
    assert all(math.isnan(out[column]) for column in _CLASS_COLUMNS)


def test_a_dump_without_odds_reads_all_nan():
    out = _book_level_columns(_levelled_frame().drop(columns="Odds"))

    assert set(out) == {"book_citl_offline", *_CLASS_COLUMNS}
    assert all(math.isnan(value) for value in out.values())


def test_the_report_layers_the_monitor_from_the_dumped_test_set(tmp_path):
    frame = _levelled_frame().assign(MeanYr=3.0, Blended_EV=3.0)
    frame.to_csv(tmp_path / f"{market_file_slug('NBA', 'FGA')}.csv", index=False)
    identity = resolve_report_identity({"distribution": "SkewNormal"}, "NBA", "FGA")
    row = {}
    with (
        mock.patch.object(report_module, "_TEST_SETS_DIR", tmp_path),
        mock.patch.object(report_module, "compute_gates", return_value={"ship": True}),
    ):
        _layer_gates_from_test_set(row, identity, "NBA", "FGA")

    assert row["book_citl_offline"] == pytest.approx(-0.05 / 6)
    assert row["book_share_redecoded"] == pytest.approx(1 / 3)
    assert row["ship"] is True
