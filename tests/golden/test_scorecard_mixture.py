"""Scorecard read paths for the 2-component Gaussian-mixture ("Mixture") family.

Mixture test-set CSVs carry the per-row params as ``MIX_Loc1`` / ``MIX_Loc2`` /
``MIX_Scale1`` / ``MIX_Scale2`` / ``MIX_W1`` (component-2 weight = ``1 − MIX_W1``)
in the cell's NORMALIZED space, plus the same constant ``DenomCol`` /
``GlobalMean`` columns SkewNormal uses — so both locs and both scales must
decode through the target-normalization registry before evaluating. Every
scorecard consumer routes through ``_infer_dist_from_columns`` →
``_pred_cdf_pmf`` / ``_pred_ppf``; these pins cover the inference, the
decode+CDF composition (PIT uniformity), the bisection PPF, the degenerate
one-component collapse to a plain normal, and an end-to-end ``gate_row``.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from sportstradamus.training.scorecard import (
    _infer_dist_from_columns,
    _ks_uniform,
    _pred_cdf_pmf,
    _pred_ppf,
    apply_thresholds,
    gate_row,
    load_test_set,
)

# All frames encode against ratio_meanyr: decoded loc/scale = raw * MeanYr
# (MeanYr kept well above the 0.5 decode floor so the multiply is exact).
_STRATEGY = "ratio_meanyr"
_MIX_COLS = {"MIX_Loc1", "MIX_Loc2", "MIX_Scale1", "MIX_Scale2", "MIX_W1"}


def _mix_frame(n: int = 200, seed: int = 4) -> pd.DataFrame:
    """Calibrated Mixture frame: ``Result`` drawn from each row's own decoded
    mixture (component picked ~Bernoulli(w1)), a ``Line`` near the mixture
    median, and a coin-flip book so the priced gates all compute.
    """
    rng = np.random.default_rng(seed)
    meanyr = rng.uniform(2.0, 10.0, n)
    w1 = rng.uniform(0.3, 0.9, n)
    raw_loc1 = rng.uniform(0.7, 0.9, n)
    raw_loc2 = rng.uniform(1.1, 1.4, n)
    raw_scale1 = rng.uniform(0.12, 0.22, n)
    raw_scale2 = rng.uniform(0.22, 0.35, n)
    loc1, loc2 = raw_loc1 * meanyr, raw_loc2 * meanyr
    scale1, scale2 = raw_scale1 * meanyr, raw_scale2 * meanyr
    from_first = rng.random(n) < w1
    result = np.where(from_first, rng.normal(loc1, scale1), rng.normal(loc2, scale2))
    df = pd.DataFrame(
        {
            "MeanYr": meanyr,
            "Result": result,
            "EV": w1 * loc1 + (1.0 - w1) * loc2,
            "MIX_Loc1": raw_loc1,
            "MIX_Loc2": raw_loc2,
            "MIX_Scale1": raw_scale1,
            "MIX_Scale2": raw_scale2,
            "MIX_W1": w1,
            "DenomCol": "MeanYr",
        }
    )
    median = _pred_ppf(df, "Mixture", 0.5, strategy=_STRATEGY)
    df["Line"] = np.round(median) + 0.5
    cdf_at_line, _ = _pred_cdf_pmf(df, "Mixture", df["Line"].to_numpy(), strategy=_STRATEGY)
    df["P"] = 1.0 - cdf_at_line
    df["Odds"] = 0.5
    return df


def test_infer_dist_and_load_test_set_keep_mix_columns(tmp_path):
    df = _mix_frame(50)
    assert _infer_dist_from_columns(df) == "Mixture"
    csv = tmp_path / "NBA_PTS.csv"
    df.to_csv(csv, index=False)
    loaded = load_test_set(csv, "EV")
    assert set(loaded.columns) >= _MIX_COLS
    assert _infer_dist_from_columns(loaded) == "Mixture"


def test_pred_cdf_pmf_bounded_cdf_and_zero_pmf():
    df = _mix_frame()
    y = df["Result"].to_numpy()
    cdf, pmf = _pred_cdf_pmf(df, "Mixture", y, strategy=_STRATEGY)
    assert np.all((cdf >= 0.0) & (cdf <= 1.0))
    # Continuous family: zero point mass routes the randomized PIT down the
    # single-deterministic-draw branch.
    np.testing.assert_array_equal(pmf, np.zeros_like(y))


def test_pred_ppf_finite_and_ordered():
    df = _mix_frame()
    q25 = _pred_ppf(df, "Mixture", 0.25, strategy=_STRATEGY)
    q75 = _pred_ppf(df, "Mixture", 0.75, strategy=_STRATEGY)
    assert np.all(np.isfinite(q25))
    assert np.all(np.isfinite(q75))
    assert np.all(q25 < q75)


def test_pit_uniform_pins_decode_and_cdf_composition():
    """Result is drawn from each row's own decoded mixture, so its PIT must be
    ~Uniform(0, 1); a decode that dropped the MeanYr denominator or scored only
    one component would blow the KS well past this bound (KS at n=200: E ≈ 0.06,
    α=0.05 critical ≈ 0.096).
    """
    df = _mix_frame()
    pit, _ = _pred_cdf_pmf(df, "Mixture", df["Result"].to_numpy(), strategy=_STRATEGY)
    assert _ks_uniform(pit) < 0.08


def test_degenerate_w1_reproduces_plain_normal_cdf():
    """w1=1 collapses the mixture to component 1: the CDF must equal the plain
    normal at the decoded loc/scale, with the dead component's params inert."""
    n = 64
    meanyr = np.linspace(2.0, 8.0, n)
    df = pd.DataFrame(
        {
            "MeanYr": meanyr,
            "MIX_Loc1": np.full(n, 1.1),
            "MIX_Loc2": np.full(n, 3.0),
            "MIX_Scale1": np.full(n, 0.2),
            "MIX_Scale2": np.full(n, 0.9),
            "MIX_W1": np.ones(n),
            "DenomCol": "MeanYr",
        }
    )
    y = 1.1 * meanyr + np.linspace(-1.0, 1.0, n)
    cdf, _ = _pred_cdf_pmf(df, "Mixture", y, strategy=_STRATEGY)
    np.testing.assert_allclose(cdf, norm.cdf(y, loc=1.1 * meanyr, scale=0.2 * meanyr), atol=1e-12)


def test_gate_row_end_to_end_mixture():
    """gate_row auto-detects MIX columns: analytical G4 + coverage populate and
    the calibrated frame clears the shape/bias gates through apply_thresholds."""
    row = apply_thresholds(
        gate_row(_mix_frame(600, seed=2), "EV", league="NBA", market="PTS", strategy=_STRATEGY)
    )
    assert row["g4_pit_ks"] is not None
    assert row["g4_iqr_pred"] > 0.0
    assert row["central50_coverage"] is not None
    assert row["central80_coverage"] is not None
    assert row["g2_pass"] and row["g3_pass"] and row["g4_pass"]
    assert isinstance(row["ship"], bool)
