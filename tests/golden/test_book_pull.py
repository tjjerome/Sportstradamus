"""The book pull: a cell's served over-probability, pulled toward the sportsbook's on quoted rows.

``apply_book_pull`` serves ``(1 - lam)·p_book + lam·p_over``. The convex form keeps both ends
exact: ``lam`` 1 returns the model's price bit for bit, even near 0 and 1 where
``p_book + lam·(p_over - p_book)`` would round it, and ``lam`` 0 returns the book's. A row
without a book price keeps the model's. ``config.book_pull`` reads a cell's committed knob, which
training copies into the pickle; a cell without one is not pulled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sportstradamus.helpers import apply_book_pull, config

# Model and book over-probabilities: an ordinary row, a tie, rows near 0 and 1 (where the
# non-convex form rounds p_over), and a row without a book price.
_P_OVER = np.array([0.37, 0.5, 1e-9, 1 - 1e-9, 0.81])
_P_BOOK = np.array([0.55, 0.5, 0.9, 0.3, np.nan])
_HAS_BOOK = np.isfinite(_P_BOOK)

# A cell's planted stat_meta knob.
_PLANTED_PULL = 0.6


def test_full_weight_serves_the_model_bit_for_bit():
    assert np.array_equal(apply_book_pull(_P_OVER, _P_BOOK, 1.0), _P_OVER)


def test_zero_weight_serves_the_book():
    pulled = apply_book_pull(_P_OVER, _P_BOOK, 0.0)
    assert np.array_equal(pulled[_HAS_BOOK], _P_BOOK[_HAS_BOOK])


def test_a_row_without_a_book_price_keeps_the_model():
    pulled = apply_book_pull(_P_OVER, _P_BOOK, 0.0)
    assert np.array_equal(pulled[~_HAS_BOOK], _P_OVER[~_HAS_BOOK])


def test_half_weight_serves_the_midpoint():
    pulled = apply_book_pull(_P_OVER, _P_BOOK, 0.5)
    np.testing.assert_allclose(pulled[_HAS_BOOK], ((_P_OVER + _P_BOOK) / 2)[_HAS_BOOK])


def test_series_inputs_price_as_arrays():
    pulled = apply_book_pull(pd.Series(_P_OVER), pd.Series(_P_BOOK), 0.5)
    assert isinstance(pulled, np.ndarray)
    assert np.array_equal(pulled, apply_book_pull(_P_OVER, _P_BOOK, 0.5))


def test_config_reads_the_cell_knob(monkeypatch):
    # The loader writes an absent knob as None (PTS); a cell missing outright (REB) reads the same.
    monkeypatch.setitem(
        config.stat_meta, "NBA", {"PRA": {"book_pull": _PLANTED_PULL}, "PTS": {"book_pull": None}}
    )
    assert config.book_pull("NBA", "PRA") == _PLANTED_PULL
    assert config.book_pull("NBA", "PTS") is None
    assert config.book_pull("NBA", "REB") is None
