"""``helpers.platform_payout`` — the one Boost-to-decimal-payout conversion for history rows."""

from statistics import geometric_mean

import numpy as np
import pandas as pd
import pytest

from sportstradamus.helpers import UNDERDOG_BOOST_BASELINE, platform_payout, underdog_payouts

# The Underdog Power entry sizes the owner plays; the per-pick value is derived from them.
_ENTRY_SIZES_PLAYED = (3, 5, 6)


def test_baseline_is_the_per_pick_root_of_the_power_entries_played():
    # Pinned to the payout table so a table change moves the per-pick value on purpose.
    power = underdog_payouts["power"]
    roots = [power[size] ** (1 / size) for size in _ENTRY_SIZES_PLAYED]
    assert round(geometric_mean(roots), 2) == UNDERDOG_BOOST_BASELINE


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
