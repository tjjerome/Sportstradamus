"""Pin that a row with no dispersion history seeds its scale head at the batch's typical
dispersion, not at the clip floor.

``set_model_start_values`` seeds every row's scale head from its ``STDYr``. A player with one
game in the window has ``STDYr = 0``, and the ``1e-6`` clip floor handed the NLL a near-zero
sigma whose gradient blew the scale head up for every row sharing its leaves: the deterministic
WNBA PA fit in ``tests/integration/test_centered_sn_live_path.py`` drifted from a scale of 6.5
after one round to 62 after thirty, and its mean from 12.3 to 8.5, on 25 such rows in 2,800.
"""

import numpy as np
import pandas as pd
import pytest

from sportstradamus.helpers.distributions import set_model_start_values


class _Model:
    start_values = None


def _batch() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MeanYr": [12.0, 20.0, 8.0, 15.0],
            "STDYr": [4.0, 6.0, 0.0, 5.0],
            "ZeroYr": [0.0, 0.0, 0.0, 0.0],
        }
    )


@pytest.mark.parametrize(
    ("kwargs", "expected_scale"),
    [
        ({}, [4.0, 6.0, 5.0, 5.0]),
        ({"offset_mode": True}, [4.0, 6.0, 5.0, 5.0]),
        ({"normalized": True}, [4.0 / 12.0, 6.0 / 20.0, 5.0 / 8.0, 5.0 / 15.0]),
    ],
    ids=["raw", "offset", "normalized"],
)
def test_zero_dispersion_row_borrows_the_batch_median(kwargs, expected_scale):
    model = _Model()

    set_model_start_values(model, "SkewNormal", _batch(), **kwargs)

    np.testing.assert_allclose(model.start_values[:, 1], np.log(expected_scale))
