"""Unit tests for the ``ratio_projvol`` target normalization.

``ratio_projvol`` trains a SkewNormal cell on a per-volume rate
(``y / projected-volume``) and decodes ``rate * projected-volume``. It is
``ratio_meanyr`` with two differences: the denominator is a market-specific
projected-volume feature, and it falls back to ``MeanYr`` per-row where the
projection is missing (the volume projection is ``fillna(0)``-sentineled in
``stats/base.py``, so a missing projection shows up as ``0`` or ``NaN``, never a
real value). These tests pin the arithmetic, the fallback, and the per-cell
denominator resolution — no network, no fixture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sportstradamus.training import baselines
from sportstradamus.training.baselines import (
    _MEANYR_FLOOR,
    get_target_normalization,
    resolve_denom_col,
)

_PROJ_MIN = "Player proj MIN mean"


def _frame() -> pd.DataFrame:
    """Feature frame with a projected-volume column and a MeanYr fallback.

    Row 3 has a ``0`` projection (the ``fillna(0)`` sentinel) and row 4 a
    ``NaN`` projection; both must fall back to ``MeanYr``.
    """
    return pd.DataFrame(
        {
            _PROJ_MIN: [30.0, 24.0, 12.0, 0.0, np.nan],
            "MeanYr": [20.0, 18.0, 10.0, 8.0, 6.0],
        }
    )


def test_forward_divides_by_projected_volume():
    strat = get_target_normalization("ratio_projvol")
    X = _frame()
    y = np.array([15.0, 12.0, 6.0, 4.0, 3.0])

    out = strat.forward(y, X, global_mean=10.0, denom_col=_PROJ_MIN)

    # Rows 0-2 divide by the projection; rows 3-4 fall back to MeanYr.
    expected_denom = np.array([30.0, 24.0, 12.0, 8.0, 6.0])
    np.testing.assert_allclose(out, y / expected_denom)


def test_zero_and_nan_projection_fall_back_to_meanyr_not_floor():
    strat = get_target_normalization("ratio_projvol")
    X = _frame()
    y = np.array([15.0, 12.0, 6.0, 4.0, 3.0])

    out = strat.forward(y, X, global_mean=10.0, denom_col=_PROJ_MIN)

    # The 0/NaN rows must divide by MeanYr (8, 6), NOT by the 0.5 denom floor
    # (which would give y/0.5 = 8.0 and 6.0 — wildly inflated rates).
    assert out[3] == pytest.approx(4.0 / 8.0)
    assert out[4] == pytest.approx(3.0 / 6.0)
    assert out[3] != pytest.approx(4.0 / _MEANYR_FLOOR)


def test_decode_loc_is_exact_inverse_of_encode_loc():
    strat = get_target_normalization("ratio_projvol")
    X = _frame()
    loc = np.array([12.0, 9.0, 5.0, 3.0, 2.0])

    round_trip = strat.decode_loc(strat.encode_loc(loc, X, 10.0, _PROJ_MIN), X, 10.0, _PROJ_MIN)
    np.testing.assert_allclose(round_trip, loc)


def test_decode_scale_is_exact_inverse_of_encode_scale():
    strat = get_target_normalization("ratio_projvol")
    X = _frame()
    scale = np.array([4.0, 3.0, 2.0, 1.5, 1.0])

    round_trip = strat.decode_scale(strat.encode_scale(scale, X, _PROJ_MIN), X, _PROJ_MIN)
    np.testing.assert_allclose(round_trip, scale)


def test_offset_meta_persists_denominator_for_serve_mirror():
    strat = get_target_normalization("ratio_projvol")
    meta = strat.offset_meta(10.0, _PROJ_MIN)
    assert meta["method"] == "projvol_ratio"
    assert meta["denom_col"] == _PROJ_MIN


def test_start_mode_is_normalized():
    assert get_target_normalization("ratio_projvol").start_mode_flag == "normalized"


def test_ratio_projvol_is_registered_slug():
    assert "ratio_projvol" in baselines.TARGET_NORMALIZATION_SLUGS


def test_resolve_denom_col_maps_nfl_efficiency_markets():
    cols = ["Player proj carries mean", "Player proj targets mean", "Player proj attempts mean"]
    assert (
        resolve_denom_col("ratio_projvol", "NFL", "rushing-yards", cols, zero_inflated=False)
        == "Player proj carries mean"
    )
    assert (
        resolve_denom_col("ratio_projvol", "NFL", "receiving yards", cols, zero_inflated=False)
        == "Player proj targets mean"
    )
    assert (
        resolve_denom_col("ratio_projvol", "NFL", "completions", cols, zero_inflated=False)
        == "Player proj attempts mean"
    )


def test_resolve_denom_col_nba_wnba_default_to_minutes():
    cols = [_PROJ_MIN]
    assert resolve_denom_col("ratio_projvol", "NBA", "PTS", cols, zero_inflated=False) == _PROJ_MIN
    assert resolve_denom_col("ratio_projvol", "WNBA", "REB", cols, zero_inflated=False) == _PROJ_MIN


def test_resolve_denom_col_raises_on_volume_or_unmapped_market():
    # A volume market's own matrix has no projected-volume column for itself.
    with pytest.raises(ValueError, match="ratio_projvol"):
        resolve_denom_col("ratio_projvol", "NFL", "carries", ["MeanYr"], zero_inflated=False)
    # An NFL market with no volume mapping at all.
    with pytest.raises(ValueError, match="ratio_projvol"):
        resolve_denom_col("ratio_projvol", "NFL", "interceptions", ["MeanYr"], zero_inflated=False)


def test_every_mapped_denominator_is_a_forward_projection():
    # Leakage guard: every denominator must resolve to a pre-game projection
    # (``Player proj {vol} mean``), never a realized same-game volume — dividing
    # the target by its own outcome would leak it. A future map edit that points
    # at a realized column trips this.
    vols = [v for mkt in baselines._PROJVOL_VOLUME_MARKET.values() for v in mkt.values()]
    vols += list(baselines._PROJVOL_DEFAULT_VOLUME_MARKET.values())
    assert vols, "no projvol mappings registered"
    for vol in vols:
        assert f"Player proj {vol} mean".startswith("Player proj ")


def test_resolve_denom_col_non_projvol_unchanged():
    # ratio_meanyr keeps the MeanYr / MeanYr_nonzero behavior.
    cols = ["MeanYr", "MeanYr_nonzero"]
    assert resolve_denom_col("ratio_meanyr", "NBA", "PTS", cols, zero_inflated=False) == "MeanYr"
    assert (
        resolve_denom_col("ratio_meanyr", "NBA", "PTS", cols, zero_inflated=True)
        == "MeanYr_nonzero"
    )
    assert (
        resolve_denom_col("ratio_meanyr", "NBA", "PTS", ["MeanYr"], zero_inflated=True) == "MeanYr"
    )
