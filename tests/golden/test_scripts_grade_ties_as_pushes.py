"""The two research scripts grade a tie as a push: dropped from the graded rows, never an Over.

Live grading (``realized.settled_offers``, the paper ledger) and training's labels already
read a result on the line as a push. These pins hold ``scripts/test_distributions.py`` and
``scripts/model_calibration/test_model_weight.py`` to the same rule, so a side-by-side with
``model_stats`` neither credits nor charges the model for a tie.
"""

from __future__ import annotations

import numpy as np
import pytest

from sportstradamus.scripts import test_distributions as td
from sportstradamus.scripts.model_calibration import test_model_weight as tmw

# A Gamma(10) cell projecting 20 against a line of 14 is a confident Over on every row.
_EV, _ALPHA, _LINE = 20.0, 10.0, 14.0


def _metrics(results):
    n = len(results)
    return td.compute_metrics(
        np.full(n, _EV), np.array(results, dtype=float), np.full(n, _LINE), _ALPHA, "Gamma"
    )


def test_distributions_accuracy_grades_settled_rows_only():
    # Twelve clears, four misses and eight ties: 12/16 right, the ties neither.
    res = _metrics([20.0] * 12 + [10.0] * 4 + [_LINE] * 8)
    assert res["accuracy"] == pytest.approx(0.75)
    assert res["n_confident"] == 16
    assert res["n_total"] == 24


def test_distributions_a_set_of_ties_grades_nothing():
    res = _metrics([_LINE] * 24)
    assert np.isnan(res["accuracy"])
    assert res["n_confident"] == 0


def test_model_weight_accuracy_drops_the_pushed_rows():
    data = {"result": np.array([20.0, 20.0, 10.0, 14.0, 14.0]), "line": np.full(5, 14.0)}
    acc, coverage = tmw._accuracy(np.full(5, 0.2), data)
    # Two clears right, one miss wrong, two pushes ungraded.
    assert acc == pytest.approx(2 / 3)
    assert coverage == pytest.approx(3 / 5)


def test_model_weight_brier_scores_settled_rows_only():
    data = {"result": np.array([20.0, 10.0, 14.0]), "line": np.full(3, 14.0)}
    # P(over) 0.8: (0.8 - 1)^2 and (0.8 - 0)^2; the push is dropped, not scored as 0 or 1.
    assert tmw._brier(np.full(3, 0.2), data) == pytest.approx((0.04 + 0.64) / 2)
