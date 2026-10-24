"""Pin the over/under label every training fit, metric and ship gate shares."""

import numpy as np
import pandas as pd

from sportstradamus.training.labels import PUSH, over_label


def test_tie_is_half_an_over():
    assert PUSH == 0.5
    np.testing.assert_array_equal(over_label([3, 2, 1], [2, 2, 2]), [1.0, 0.5, 0.0])


def test_missing_result_or_line_is_an_under():
    np.testing.assert_array_equal(over_label([np.nan, 3.0], [2.0, np.nan]), [0.0, 0.0])


def test_series_are_read_by_position():
    result = pd.Series([3.0, 2.0, 1.0], index=[2, 0, 1])
    line = pd.Series([2.0, 2.0, 0.0], index=[0, 1, 2])
    # Matching on the index instead would give [0.5, 0.0, 1.0].
    np.testing.assert_array_equal(over_label(result, line), [1.0, 0.5, 1.0])
