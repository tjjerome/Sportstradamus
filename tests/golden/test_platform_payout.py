"""``helpers.platform_payout`` — the one Boost-to-decimal-payout conversion for history rows."""

import numpy as np
import pandas as pd
import pytest

from sportstradamus.helpers import UNDERDOG_BOOST_BASELINE, platform_payout


def test_underdog_scales_by_the_baseline_and_sleeper_is_as_posted():
    assert platform_payout(1.0, "Underdog") == pytest.approx(UNDERDOG_BOOST_BASELINE)
    assert platform_payout(1.5, "Sleeper") == pytest.approx(1.5)
    assert platform_payout(0.0, "Underdog") == 0.0


def test_series_inputs_align_row_by_row():
    boost = pd.Series([1.0, 1.5, 0.56], index=[7, 8, 9])
    platform = pd.Series(["Underdog", "Sleeper", "Underdog"], index=[7, 8, 9])
    out = platform_payout(boost, platform)
    assert list(out.index) == [7, 8, 9]
    np.testing.assert_allclose(out, [UNDERDOG_BOOST_BASELINE, 1.5, 0.56 * UNDERDOG_BOOST_BASELINE])
