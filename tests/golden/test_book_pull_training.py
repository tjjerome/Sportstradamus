"""Pins the book pull in training: the dump's ``P`` is the price serving pulls toward the book.

On the standard path ``_structural_gate_inputs`` serves ``(1 - lam)·Odds + lam·P`` on rows with
an authentic sportsbook quote, where ``P`` is the over-probability after temperature and the
probability-stage post-hoc and ``Odds`` the training layer's book over-probability. A pickem or
derived row keeps ``P``, and so does every row of a cell without the knob. The stages before the
pull are stubbed: these pins hold where the pull lands, not how ``P`` is fit. The pickle carries
the knob for serving as a float, and only when the cell sets it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sportstradamus.helpers import apply_temperature
from sportstradamus.helpers.training_quotes import AUTHENTIC, DERIVED, PICKEM
from sportstradamus.training import pipeline as pipe
from sportstradamus.training.model_strategy import BASE_STRUCTURAL_STRATEGY
from tests.golden.test_model_version_attribution import _minimal_filedict_kwargs

# The model's over-probability before temperature and the book's, with each row's quote class.
_RAW_OVER = np.array([0.62, 0.41, 0.55, 0.30])
_BOOK_OVER = np.array([0.48, 0.52, 0.50, 0.35])
_QUOTES = pd.Series([AUTHENTIC, PICKEM, AUTHENTIC, DERIVED])
_AUTHENTIC = _QUOTES.eq(AUTHENTIC).to_numpy()
_TEMPERATURE = 1.3
_PULL = 0.4
_UNPULLED = apply_temperature(_RAW_OVER, _TEMPERATURE)


def _served_over(monkeypatch, book_pull):
    monkeypatch.setattr(
        pipe,
        "_step_compute_test_probabilities",
        lambda *args: np.column_stack((1 - _RAW_OVER, _RAW_OVER)),
    )
    monkeypatch.setattr(
        pipe, "_step_calibrate_temperature", lambda *args: (_TEMPERATURE, None, None, None)
    )
    splits = {"B_test": pd.DataFrame({"Odds": _BOOK_OVER}), "quote_authenticity_test": _QUOTES}
    return pipe._structural_gate_inputs(
        BASE_STRUCTURAL_STRATEGY,
        calibrated={},
        fused={},
        splits=splits,
        decoded={},
        dist="SkewNormal",
        step=1.0,
        experiment_context=None,
        posthoc_slug="none",
        mean_posthoc_blob=None,
        book_pull=book_pull,
    )["test_calibrated_over"]


def test_authentic_rows_pull_toward_the_book(monkeypatch):
    expected = (1 - _PULL) * _BOOK_OVER + _PULL * _UNPULLED
    np.testing.assert_allclose(_served_over(monkeypatch, _PULL)[_AUTHENTIC], expected[_AUTHENTIC])


def test_pickem_and_derived_rows_keep_the_served_price(monkeypatch):
    assert np.array_equal(_served_over(monkeypatch, _PULL)[~_AUTHENTIC], _UNPULLED[~_AUTHENTIC])


def test_a_cell_without_the_knob_is_not_pulled(monkeypatch):
    assert np.array_equal(_served_over(monkeypatch, None), _UNPULLED)


def test_the_pickle_carries_the_knob_as_a_float_only_when_set():
    kwargs = _minimal_filedict_kwargs()
    assert "book_pull" not in pipe._build_filedict(**kwargs)
    pulled = pipe._build_filedict(**kwargs, book_pull=1)["book_pull"]
    assert isinstance(pulled, float)
    assert pulled == 1
