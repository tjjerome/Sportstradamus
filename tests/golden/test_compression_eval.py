"""Unit tests for the Phase-0 compression eval harness.

Exercises the numeric path (decile binning, compression ratio, scorecard,
ship/kill verdict) on synthetic test-set frames so no trained model, network,
or plotting backend is required.
"""

import numpy as np
import pandas as pd
import pytest

from sportstradamus.scripts.compression_eval import (
    MIN_TOP_DECILE_MAE_IMPROVEMENT,
    decile_table,
    load_test_set,
    scorecard,
    verdict,
)


def _compressed_frame(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Build a frame whose predictions are shrunk toward the global mean.

    Actuals span a wide MeanYr range; predictions pull each row halfway to the
    grand mean — the canonical compression pathology.
    """
    rng = np.random.default_rng(seed)
    meanyr = rng.uniform(2, 30, n)
    actual = meanyr + rng.normal(0, 3, n)
    grand = actual.mean()
    pred = grand + 0.5 * (actual - grand)
    return pd.DataFrame({"MeanYr": meanyr, "Result": actual, "EV": pred})


def test_decile_table_shape_and_monotone_bias():
    df = _compressed_frame()
    table = decile_table(df, "EV", n_deciles=10)
    assert len(table) == 10
    # Compression => top decile under-predicted (negative bias), bottom
    # decile over-predicted (positive bias).
    assert table.iloc[-1]["bias"] < 0
    assert table.iloc[0]["bias"] > 0


def test_compression_ratio_below_one_for_shrunk_predictions():
    card = scorecard(
        _compressed_frame(), "EV", strategy="t", league="NBA", market="PTS"
    )
    assert 0.45 < card.compression_ratio < 0.55
    assert card.top_decile_mae > 0
    assert card.top_decile_bias < 0


def test_perfect_predictions_have_unit_ratio():
    rng = np.random.default_rng(1)
    meanyr = rng.uniform(2, 30, 1000)
    df = pd.DataFrame({"MeanYr": meanyr, "Result": meanyr, "EV": meanyr})
    card = scorecard(df, "EV", strategy="t", league="NBA", market="PTS")
    assert card.compression_ratio == pytest.approx(1.0, abs=1e-9)
    assert card.global_mae == pytest.approx(0.0, abs=1e-9)


def test_verdict_ships_when_top_decile_improves():
    base = scorecard(
        _compressed_frame(seed=0), "EV", strategy="base", league="NBA", market="PTS"
    )
    # Candidate: predictions much closer to actual (less compression).
    df = _compressed_frame(seed=0)
    df["EV"] = df["Result"].mean() + 0.95 * (df["Result"] - df["Result"].mean())
    cand = scorecard(df, "EV", strategy="cand", league="NBA", market="PTS")
    ship, reason = verdict(base, cand)
    assert ship, reason


def test_verdict_kills_when_no_top_decile_gain():
    base = scorecard(
        _compressed_frame(seed=2), "EV", strategy="base", league="NBA", market="PTS"
    )
    ship, reason = verdict(base, base)
    assert not ship
    assert "KILL" in reason
    assert f"{MIN_TOP_DECILE_MAE_IMPROVEMENT:.0%}" in reason


def test_load_test_set_drops_nonfinite_and_validates_columns(tmp_path):
    good = pd.DataFrame(
        {"MeanYr": [1.0, 2.0, np.inf], "Result": [1.0, 2.0, 3.0], "EV": [1.0, 2.0, 3.0]}
    )
    p = tmp_path / "NBA_PTS.csv"
    good.to_csv(p, index=False)
    loaded = load_test_set(p, "EV")
    assert len(loaded) == 2

    bad = pd.DataFrame({"MeanYr": [1.0], "Result": [1.0]})
    bp = tmp_path / "NBA_AST.csv"
    bad.to_csv(bp, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_test_set(bp, "EV")
