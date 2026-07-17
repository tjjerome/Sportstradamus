"""Scorecard read paths for the Double Poisson ("DPO") family.

DPO test-set CSVs carry the natural params as ``DP_MU`` / ``DP_PHI`` columns
(the count analogue of NegBin's ``R`` / ``NB_P``) and no ``Gate`` column — the
family is gate-free. Every scorecard consumer routes through
``_infer_dist_from_columns`` → ``_pred_cdf_pmf`` / ``_pred_ppf``, so these pins
cover the inference, the bridges to the shared numpy kernel in
``helpers.distributions``, Gate-4 randomized-PIT calibration, the Gate-6
count-cohort over-leg, and an end-to-end ``gate_row``.
"""

import numpy as np
import pandas as pd

from sportstradamus.helpers.distributions import _dp_cdf_pmf, _dp_ppf
from sportstradamus.training.scorecard import (
    _GATE6_MARGIN,
    _gate6_legs,
    _infer_dist_from_columns,
    _pred_cdf_pmf,
    _pred_ppf,
    _randomized_pit_ks,
    apply_thresholds,
    gate_row,
    load_test_set,
)

# Low-mean / near-unit-precision band — the lattice regime DPO cells live in
# (NBA PF, NHL points per the WS-3 Phase-0 screen).
_MU_RANGE = (0.5, 4.0)
_PHI_RANGE = (0.6, 1.8)


def _dpo_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Calibrated DPO frame: ``Result`` drawn from each row's own ``(mu, phi)``
    by inverse-CDF sampling, with a calibrated over-probability ``P`` and a
    coin-flip book so the priced gates all compute.
    """
    rng = np.random.default_rng(seed)
    mu = rng.uniform(*_MU_RANGE, n)
    phi = rng.uniform(*_PHI_RANGE, n)
    y = _dp_ppf(rng.uniform(size=n), mu, phi)
    line = np.round(mu) + 0.5
    cdf_at_line, _ = _dp_cdf_pmf(np.floor(line), mu, phi)
    return pd.DataFrame(
        {
            "MeanYr": mu,
            "Result": y,
            "EV": mu,
            "DP_MU": mu,
            "DP_PHI": phi,
            "Line": line,
            "P": 1.0 - cdf_at_line,
            "Odds": np.full(n, 0.5),
        }
    )


def test_infer_dist_and_load_test_set_keep_dp_columns(tmp_path):
    df = _dpo_frame(50)
    assert _infer_dist_from_columns(df) == "DPO"
    csv = tmp_path / "NBA_PF.csv"
    df.to_csv(csv, index=False)
    loaded = load_test_set(csv, "EV")
    assert {"DP_MU", "DP_PHI"} <= set(loaded.columns)
    assert _infer_dist_from_columns(loaded) == "DPO"


def test_pred_cdf_pmf_matches_kernel():
    df = _dpo_frame(200, seed=2)
    y = df["Result"].to_numpy()
    cdf, pmf = _pred_cdf_pmf(df, "DPO", y, strategy="baseline")
    ref_cdf, ref_pmf = _dp_cdf_pmf(y, df["DP_MU"].to_numpy(), df["DP_PHI"].to_numpy())
    np.testing.assert_array_equal(cdf, ref_cdf)
    np.testing.assert_array_equal(pmf, ref_pmf)
    # Integer outcomes carry point mass — what routes the randomized PIT down
    # the discrete (multi-draw) branch.
    assert np.all(pmf > 0)


def test_pred_ppf_matches_kernel_and_round_trips():
    df = _dpo_frame(120, seed=3)
    mu = df["DP_MU"].to_numpy()
    phi = df["DP_PHI"].to_numpy()
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        k = _pred_ppf(df, "DPO", q, strategy="baseline")
        np.testing.assert_array_equal(k, _dp_ppf(q, mu, phi))
        cdf_at_k, _ = _dp_cdf_pmf(k, mu, phi)
        cdf_below, _ = _dp_cdf_pmf(k - 1.0, mu, phi)
        assert np.all(cdf_at_k >= q)
        assert np.all(cdf_below < q)


def test_randomized_pit_ks_calibrated_dpo_clears_gate4():
    """Well-calibrated DPO sample passes the Gate-4 KS; a 2x-too-narrow predictive fails it."""
    df = _dpo_frame(6000, seed=7)
    y = df["Result"].to_numpy()
    assert _randomized_pit_ks(df, "DPO", y, strategy="baseline") < 0.05
    narrow = df.assign(DP_PHI=df["DP_PHI"] * 4.0)  # variance mu/phi shrinks 4x
    assert _randomized_pit_ks(narrow, "DPO", y, strategy="baseline") > 0.10


def test_gate_row_end_to_end_dpo():
    """gate_row auto-detects DP columns: analytical G4 + coverage populate and the
    calibrated frame clears the shape/bias gates through apply_thresholds."""
    row = apply_thresholds(
        gate_row(_dpo_frame(600, seed=1), "EV", league="NBA", market="PF", strategy="baseline")
    )
    assert row["g4_pit_ks"] is not None
    assert row["g4_iqr_pred"] > 0.0
    assert row["central50_coverage"] is not None
    assert row["central80_coverage"] is not None
    assert row["g2_pass"] and row["g3_pass"] and row["g4_pass"]
    assert isinstance(row["ship"], bool)


def _dpo_bench_over_frame(n: int = 200, seed: int = 7) -> pd.DataFrame:
    """Mirror of test_scorecard's ``_count_legs_frame`` on DP params: the stable
    BENCH (low MeanYr) is over-predicted 1.3x vs a realized mean of 2.0 (above
    the ``_GATE6_OVER_MIN_MEAN`` guard), so the over-leg must fire iff DPO is in
    the Gate-6 count cohort.
    """
    rng = np.random.default_rng(seed)
    meanyr = np.concatenate([rng.uniform(0.5, 3.0, n // 2), rng.uniform(8.0, 20.0, n // 2)])
    mean10 = meanyr * (1.0 + rng.uniform(-0.05, 0.05, n))
    bench = meanyr <= np.quantile(meanyr, 0.25)
    return pd.DataFrame(
        {
            "MeanYr": meanyr,
            "Mean10": mean10,
            "Result": np.where(bench, 2.0, mean10) + rng.normal(0.0, 0.2, n),
            "Blended_EV": np.where(bench, 1.3 * 2.0, mean10),
            "Player": [f"P{i % 40}" for i in range(n)],
            "DP_MU": meanyr,
            "DP_PHI": np.ones(n),
        }
    )


def test_gate6_over_leg_covers_dpo_cells():
    g6 = _gate6_legs(_dpo_bench_over_frame(), "Blended_EV", league="WNBA", prior_g6_fired=None)
    assert g6["g6_over_ci_lo"] is not None
    assert g6["g6_over_ci_lo"] > 1.0 + _GATE6_MARGIN
