"""Unit tests for ``training.scorecard`` — the offline ship-gate harness.

Exercises the numeric path (decile binning, compression ratio, scorecard, the five
offline ship gates, and their deterministic-1/0 oracle) on synthetic test-set frames
so no trained model, network, or plotting backend is required.
"""

import numpy as np
import pandas as pd
import pytest

from sportstradamus.training.scorecard import (
    _GATE1_CI_HI_MAX,
    _GATE2_STAR_Z_MAX,
    _GATE3_BENCH_Z_MAX,
    _GATE4_IQR_RATIO_MIN,
    _GATE5_ECE_MAX,
    _SUPERSEDE_S3_Z_MIN,
    _ece_debias_offset,
    _gate1_brier_ci,
    _gate4_iqr_spread,
    _gate5_ece_debiased,
    _gate5_ece_equal_mass,
    _gate23_segment_match,
    _infer_dist_from_columns,
    _iqr_pred_analytical,
    _memmel_sharpe_z,
    _segment_masks,
    _supersede_paired_brier_ci,
    _supersede_paired_sharpe,
    _test_set_to_bet_frame,
    _zinb_ppf,
    apply_thresholds,
    decile_table,
    gate_row,
    load_test_set,
    scorecard,
    supersede_verdict,
    write_gate_scorecard,
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


def _priced_frame(n: int = 4000, seed: int = 7) -> pd.DataFrame:
    """A priced frame (P/Odds/Line present) with a calibrated-ish model and a noisy book."""
    rng = np.random.default_rng(seed)
    meanyr = rng.uniform(2, 30, n)
    line = meanyr.copy()
    p_true = rng.uniform(0.05, 0.95, n)
    outcomes = rng.uniform(size=n) < p_true
    result = np.where(outcomes, line + 1.0, line - 1.0)
    return pd.DataFrame(
        {
            "MeanYr": meanyr,
            "Result": result,
            "EV": meanyr,
            "Line": line,
            "P": p_true,  # model tracks the true over-rate
            "Odds": np.full(n, 0.5),  # book under-prob 0.5 (near-random)
        }
    )


# ---------------------------------------------------------------------------
# Compression diagnostics (decile table, std-ratio scorecard) — unchanged.
# ---------------------------------------------------------------------------


def test_decile_table_shape_and_monotone_bias():
    df = _compressed_frame()
    table = decile_table(df, "EV", n_deciles=10)
    assert len(table) == 10
    # Compression => top decile under-predicted (negative bias), bottom
    # decile over-predicted (positive bias).
    assert table.iloc[-1]["bias"] < 0
    assert table.iloc[0]["bias"] > 0


def test_compression_ratio_below_one_for_shrunk_predictions():
    card = scorecard(_compressed_frame(), "EV", strategy="t", league="NBA", market="PTS")
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


def test_scorecard_includes_bottom_quartile_bias_and_mean():
    # Bottom quartile (25 lowest-MeanYr rows of 100) over-predicted by a known
    # +2.0; all other rows predicted perfectly. The scorecard must surface the
    # bottom-quartile signed bias and that quartile's empirical (actual) mean.
    meanyr = np.arange(100, dtype=float)
    result = meanyr.copy()
    pred = result.copy()
    pred[:25] += 2.0  # over-predict the lowest quartile only
    df = pd.DataFrame({"MeanYr": meanyr, "Result": result, "EV": pred})
    card = scorecard(df, "EV", strategy="t", league="NBA", market="PTS")
    assert card.bottom_quartile_bias == pytest.approx(2.0)
    assert card.bottom_quartile_mean == pytest.approx(12.0)  # mean(0..24)
    assert card.bottom_quartile_mae == pytest.approx(2.0)


def test_brier_skill_score_positive_when_model_beats_book():
    card = scorecard(_priced_frame(seed=0), "EV", strategy="t", league="NBA", market="PTS")
    assert card.brier_skill_score is not None
    assert card.brier_skill_score > 0.1


def test_brier_skill_score_negative_when_book_beats_model():
    rng = np.random.default_rng(1)
    n = 4000
    meanyr = rng.uniform(2, 30, n)
    line = meanyr.copy()
    p_true = rng.uniform(0.05, 0.95, n)
    outcomes = rng.uniform(size=n) < p_true
    result = np.where(outcomes, line + 1.0, line - 1.0)
    df = pd.DataFrame(
        {
            "MeanYr": meanyr,
            "Result": result,
            "EV": meanyr,
            "Line": line,
            # Model is anti-correlated noise; book is the true probability so
            # book_over = 1 - Odds nails it.
            "P": 1.0 - p_true,
            "Odds": 1.0 - p_true,
        }
    )
    card = scorecard(df, "EV", strategy="t", league="NBA", market="PTS")
    assert card.brier_skill_score is not None
    assert card.brier_skill_score < 0


# ---------------------------------------------------------------------------
# Segment masks (shared by scorecard + Gates 2/3).
# ---------------------------------------------------------------------------


def test_segment_masks_partition_top_decile_and_bottom_quartile():
    df = pd.DataFrame({"MeanYr": np.arange(100, dtype=float)})
    star_mask, bench_mask = _segment_masks(df, n_deciles=10)
    assert star_mask.sum() == 10  # top decile
    assert bench_mask.sum() == 25  # bottom quartile
    assert not (star_mask & bench_mask).any()  # disjoint segments


# ---------------------------------------------------------------------------
# Gate 1 — Brier vs book, paired bootstrap CI.
# ---------------------------------------------------------------------------


def test_gate1_ci_below_zero_when_model_beats_book():
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=4000) < 0.5).astype(float)
    p_model = np.where(y == 1, 0.9, 0.1)  # close to truth
    p_book = np.full_like(y, 0.5)  # near-random
    mean, _lo, hi = _gate1_brier_ci(p_model, p_book, y, rng)
    assert mean < 0  # model Brier lower than book
    assert hi < 0  # 95% CI entirely below 0


def test_gate1_ci_above_zero_when_book_beats_model():
    rng = np.random.default_rng(1)
    y = (rng.uniform(size=4000) < 0.5).astype(float)
    p_model = np.where(y == 1, 0.1, 0.9)  # anti-correlated
    p_book = np.where(y == 1, 0.9, 0.1)  # nails it
    mean, lo, _hi = _gate1_brier_ci(p_model, p_book, y, rng)
    assert mean > 0
    assert lo > 0  # CI excludes 0 on the book's side


def test_gate1_oracle_equals_negative_book_brier():
    # Oracle p_model = y -> d_i = -(p_book - y)^2; the point mean is exactly
    # -mean(book Brier), and the CI sits entirely below 0.
    rng = np.random.default_rng(2)
    y = (rng.uniform(size=3000) < 0.5).astype(float)
    p_book = rng.uniform(0.2, 0.8, size=3000)
    mean, _lo, hi = _gate1_brier_ci(y, p_book, y, rng)
    assert mean == pytest.approx(-float(np.mean((p_book - y) ** 2)))
    assert hi < 0


# ---------------------------------------------------------------------------
# Gates 2/3 — segment bias-vs-spread match (denominator = segment sigma).
# ---------------------------------------------------------------------------


def test_gate23_segment_match_z_matches_known_bias():
    actual = np.arange(50, 150, dtype=float)
    mask = np.zeros(len(actual), dtype=bool)
    mask[:30] = True
    pred = actual.copy()
    pred[:30] += 2.0  # over-predict the segment by a constant +2.0
    _pred_mean, _true_mean, abs_diff, sigma, z = _gate23_segment_match(pred, actual, mask)
    expected_sigma = float(np.std(actual[:30], ddof=1))
    assert abs_diff == pytest.approx(2.0)
    assert sigma == pytest.approx(expected_sigma)
    assert z == pytest.approx(2.0 / expected_sigma)


def test_gate23_oracle_z_is_zero():
    actual = np.arange(50, 150, dtype=float)
    mask = np.zeros(len(actual), dtype=bool)
    mask[:30] = True
    *_, z = _gate23_segment_match(actual, actual, mask)  # oracle: pred = actual
    assert z == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Gate 4 — IQR spread (compression).
# ---------------------------------------------------------------------------


def test_gate4_iqr_ratio_below_one_for_compressed():
    df = _compressed_frame()
    _, _, ratio = _gate4_iqr_spread(df["Result"].to_numpy(), df["EV"].to_numpy())
    assert 0.45 < ratio < 0.55  # predictions shrunk to half-spread


def test_gate4_iqr_ratio_unit_for_perfect():
    x = np.arange(100, dtype=float)
    _, _, ratio = _gate4_iqr_spread(x, x)
    assert ratio == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Gate 4 analytical IQR (Operation Ship 75 Step 0.2) — pooled per-row q25/q75
# from the predicted distribution; replaces the broken point-IQR estimator on
# probabilistic markets. Brief at /tmp/researcher_g4_audit.md (Outcome B).
# ---------------------------------------------------------------------------


def test_zinb_ppf_zero_inflated_quantile_clips_to_zero():
    """Quantiles below the zero-inflation gate land at 0."""
    r = np.array([5.0, 5.0])
    nb_p = np.array([0.4, 0.4])
    gate = np.array([0.4, 0.4])  # 40% structural zeros
    # q=0.25 is well below π=0.4, so all rows return 0.
    out = _zinb_ppf(0.25, r, nb_p, gate)
    assert out.tolist() == [0.0, 0.0]


def test_zinb_ppf_above_gate_matches_rescaled_nbinom():
    """Above the gate the quantile inverts the rescaled NB tail."""
    from scipy.stats import nbinom

    r = np.array([5.0, 10.0])
    nb_p = np.array([0.4, 0.6])  # torch "probs"
    gate = np.array([0.2, 0.3])
    q = 0.75
    out = _zinb_ppf(q, r, nb_p, gate)
    # Reference: rescaled NB quantile per row.
    expected = np.array(
        [
            nbinom.ppf((q - 0.2) / (1 - 0.2), 5.0, 1.0 - 0.4),
            nbinom.ppf((q - 0.3) / (1 - 0.3), 10.0, 1.0 - 0.6),
        ]
    )
    assert np.allclose(out, expected)


def test_zinb_ppf_no_gate_matches_plain_nbinom():
    """gate=0 reduces ZINB to plain NB at every quantile."""
    from scipy.stats import nbinom

    r = np.array([5.0, 8.0, 12.0])
    nb_p = np.array([0.3, 0.5, 0.7])
    gate = np.zeros(3)
    for q in (0.25, 0.5, 0.75):
        out = _zinb_ppf(q, r, nb_p, gate)
        ref = nbinom.ppf(q, r, 1.0 - nb_p)
        assert np.allclose(out, ref), f"mismatch at q={q}"


def test_iqr_pred_analytical_negbin_matches_pooled_quantiles():
    """NegBin analytical IQR = IQR(concat(q25_per_row, q75_per_row))."""
    from scipy.stats import nbinom

    n = 500
    df = pd.DataFrame(
        {
            "R": np.full(n, 10.0),
            "NB_P": np.full(n, 0.5),
        }
    )
    iqr = _iqr_pred_analytical(df, "NegBin", strategy="ratio_meanyr")
    # All rows identical → pooled IQR equals the per-row IQR.
    q25 = nbinom.ppf(0.25, 10.0, 0.5)
    q75 = nbinom.ppf(0.75, 10.0, 0.5)
    assert iqr == pytest.approx(q75 - q25, abs=1e-9)


def test_iqr_pred_analytical_zinb_brief_worked_example():
    """ZINB(π=0.3, r=5, p=0.4) — pooled IQR matches the manual computation.

    F_ZINB(0) = 0.3 + 0.7 * (1-0.4)^5 = 0.3 + 0.7 * 0.07776 = 0.3544
    F_ZINB(1) = 0.3 + 0.7 * nbinom.cdf(1, 5, 0.6) per scipy convention.
    Since q25=0.25 < 0.3544 the per-row q25 = 0.
    q75=0.75 → (0.75 - 0.3)/0.7 = 0.6429 → nbinom.ppf(0.6429, 5, 0.6).
    """
    from scipy.stats import nbinom

    n = 1000
    df = pd.DataFrame(
        {
            "R": np.full(n, 5.0),
            "NB_P": np.full(n, 0.4),
            "Gate": np.full(n, 0.3),
        }
    )
    iqr = _iqr_pred_analytical(df, "ZINB", strategy="ratio_meanyr")
    q25 = 0.0
    q75 = float(nbinom.ppf((0.75 - 0.3) / (1 - 0.3), 5.0, 1.0 - 0.4))
    assert iqr == pytest.approx(q75 - q25, abs=1e-9)


def test_iqr_pred_analytical_skewnormal_ratio_strategy():
    """SkewNormal under ratio_meanyr: decoded_scale = SN_Scale * MeanYr."""
    from scipy.stats import skewnorm

    n = 500
    raw_loc = 1.0
    raw_scale = 0.5
    alpha = 2.0
    meanyr_value = 4.0
    df = pd.DataFrame(
        {
            "SN_Loc": np.full(n, raw_loc),
            "SN_Scale": np.full(n, raw_scale),
            "SN_Alpha": np.full(n, alpha),
            "MeanYr": np.full(n, meanyr_value),
            "Result": np.zeros(n),
            "EV": np.zeros(n),
        }
    )
    iqr = _iqr_pred_analytical(df, "SkewNormal", strategy="ratio_meanyr")
    decoded_loc = raw_loc * meanyr_value
    decoded_scale = raw_scale * meanyr_value
    q25 = float(skewnorm.ppf(0.25, alpha, loc=decoded_loc, scale=decoded_scale))
    q75 = float(skewnorm.ppf(0.75, alpha, loc=decoded_loc, scale=decoded_scale))
    assert iqr == pytest.approx(q75 - q25, abs=1e-6)


def test_iqr_pred_analytical_skewnormal_centered_strategy_passes_scale_through():
    """centered_additive_* strategies leave SN_Scale alone (no MeanYr multiply)."""
    from scipy.stats import skewnorm

    n = 500
    raw_loc = 1.5
    raw_scale = 2.0
    alpha = -1.0
    df = pd.DataFrame(
        {
            "SN_Loc": np.full(n, raw_loc),
            "SN_Scale": np.full(n, raw_scale),
            "SN_Alpha": np.full(n, alpha),
            "MeanYr": np.full(n, 4.0),  # would change result under ratio_meanyr
        }
    )
    iqr = _iqr_pred_analytical(df, "SkewNormal", strategy="centered_additive_mean10")
    q25 = float(skewnorm.ppf(0.25, alpha, loc=raw_loc, scale=raw_scale))
    q75 = float(skewnorm.ppf(0.75, alpha, loc=raw_loc, scale=raw_scale))
    assert iqr == pytest.approx(q75 - q25, abs=1e-6)


def test_iqr_pred_analytical_gamma_recovers_rate_from_ev():
    """Gamma analytical IQR uses scipy.stats.gamma; rate = Alpha / EV."""
    from scipy.stats import gamma as scipy_gamma

    n = 500
    a = 4.0
    rate = 2.0
    ev = a / rate
    df = pd.DataFrame(
        {
            "Alpha": np.full(n, a),
            "EV": np.full(n, ev),
        }
    )
    iqr = _iqr_pred_analytical(df, "Gamma", strategy="ratio_meanyr")
    q25 = float(scipy_gamma.ppf(0.25, a, scale=1.0 / rate))
    q75 = float(scipy_gamma.ppf(0.75, a, scale=1.0 / rate))
    assert iqr == pytest.approx(q75 - q25, abs=1e-6)


def test_gate4_iqr_spread_back_compat_point_iqr_without_df():
    """Old signature (no df / dist / strategy) keeps point-IQR semantics — the
    oracle row (pred = actual) still returns ratio = 1.0 so existing assertions
    on `g4_iqr_ratio_oracle` carry over.
    """
    x = np.arange(100, dtype=float)
    _, _, ratio = _gate4_iqr_spread(x, x)
    assert ratio == pytest.approx(1.0)


def test_gate4_iqr_spread_analytical_replaces_point_on_zinb():
    """When df + dist supplied, predicted IQR is the analytical pooled IQR.

    For a calibrated ZINB-like population, analytical IQR should match the
    actuals' IQR closely — the new gate stops measuring the point-prediction
    smoothing artifact.
    """
    from scipy.stats import nbinom

    rng = np.random.default_rng(1729)
    n = 4000
    r_arr = np.full(n, 5.0)
    p_arr = np.full(n, 0.4)
    gate_arr = np.zeros(n)
    # Calibrated actuals drawn from the matching NB.
    actuals = nbinom.rvs(5.0, 1.0 - 0.4, size=n, random_state=rng).astype(float)
    pred = np.full(n, float(actuals.mean()))  # point pred — smooth, deliberately
    df = pd.DataFrame({"Result": actuals, "EV": pred, "R": r_arr, "NB_P": p_arr, "Gate": gate_arr})
    iqr_pred, iqr_true, ratio = _gate4_iqr_spread(
        actuals, pred, df=df, dist="ZINB", strategy="ratio_meanyr"
    )
    assert iqr_true > 0
    # Analytical pooled q75 - q25 on the matching NB should equal the actuals' IQR
    # (integer support; pooled bag of identical per-row q25/q75 gives that exact diff).
    expected_q25 = float(nbinom.ppf(0.25, 5.0, 1.0 - 0.4))
    expected_q75 = float(nbinom.ppf(0.75, 5.0, 1.0 - 0.4))
    assert iqr_pred == pytest.approx(expected_q75 - expected_q25, abs=1e-9)
    assert ratio > 0.5  # passes the gate


def test_gate4_iqr_spread_degenerate_zero_over_zero_ships():
    """IQR_true = 0 AND IQR_pred = 0 → ratio 1.0 (the 0/0 convention)."""
    n = 200
    actuals = np.zeros(n)
    # ZINB with extreme structural zeros → IQR_pred also 0.
    df = pd.DataFrame(
        {
            "Result": actuals,
            "EV": np.zeros(n),
            "R": np.full(n, 1.0),
            "NB_P": np.full(n, 0.01),  # tiny mass on >0
            "Gate": np.full(n, 0.99),  # 99% structural zeros
        }
    )
    iqr_pred, iqr_true, ratio = _gate4_iqr_spread(
        actuals, np.zeros(n), df=df, dist="ZINB", strategy="ratio_meanyr"
    )
    assert iqr_true == 0.0
    assert iqr_pred == 0.0
    assert ratio == pytest.approx(1.0)


def test_gate4_iqr_spread_degenerate_nonzero_pred_zero_true_fails():
    """IQR_pred > 0 but IQR_true = 0 → ratio = inf (the gate fails)."""
    from scipy.stats import nbinom

    n = 200
    actuals = np.zeros(n)  # truth is fully zero
    df = pd.DataFrame(
        {
            "Result": actuals,
            "EV": np.full(n, 1.0),
            "R": np.full(n, 5.0),
            "NB_P": np.full(n, 0.5),
            "Gate": np.zeros(n),
        }
    )
    iqr_pred, iqr_true, ratio = _gate4_iqr_spread(
        actuals, np.full(n, 1.0), df=df, dist="ZINB", strategy="ratio_meanyr"
    )
    assert iqr_true == 0.0
    assert iqr_pred > 0
    # NB(5, 0.5) actuals at NB_P=0.5 have IQR > 0; pred can't match a degenerate truth.
    assert np.isinf(ratio)
    # Confirm scipy NB IQR is non-zero for this parameterization (sanity).
    assert nbinom.ppf(0.75, 5.0, 0.5) > nbinom.ppf(0.25, 5.0, 0.5)


def test_infer_dist_from_columns_dispatches_by_param_columns():
    """Distribution family inferred from the per-row param columns present."""
    sn = pd.DataFrame({"SN_Loc": [1.0], "SN_Scale": [0.5], "SN_Alpha": [0.0]})
    nb = pd.DataFrame({"R": [5.0], "NB_P": [0.4]})
    zinb = pd.DataFrame({"R": [5.0], "NB_P": [0.4], "Gate": [0.2]})
    gamma = pd.DataFrame({"Alpha": [4.0], "EV": [2.0]})
    zagamma = pd.DataFrame({"Alpha": [4.0], "EV": [2.0], "Gate": [0.3]})
    bare = pd.DataFrame({"Result": [0.0], "EV": [1.0]})
    assert _infer_dist_from_columns(sn) == "SkewNormal"
    assert _infer_dist_from_columns(nb) == "NegBin"
    assert _infer_dist_from_columns(zinb) == "ZINB"
    assert _infer_dist_from_columns(gamma) == "Gamma"
    assert _infer_dist_from_columns(zagamma) == "ZAGamma"
    assert _infer_dist_from_columns(bare) is None


def test_gate_row_uses_analytical_g4_when_dist_columns_present():
    """gate_row auto-detects ZINB columns and routes G4 through the analytical path.

    On a synthetic calibrated ZINB-like frame, the analytical g4_iqr_pred should
    differ from the point IQR of EV (which is degenerate at the mean).
    """
    from scipy.stats import nbinom

    rng = np.random.default_rng(31)
    n = 4000
    actuals = nbinom.rvs(5.0, 1.0 - 0.4, size=n, random_state=rng).astype(float)
    point_pred = np.full(n, float(actuals.mean()))
    df = pd.DataFrame(
        {
            "MeanYr": np.full(n, float(actuals.mean())),
            "Result": actuals,
            "EV": point_pred,
            "R": np.full(n, 5.0),
            "NB_P": np.full(n, 0.4),
            "Gate": np.zeros(n),
        }
    )
    row = gate_row(df, "EV", league="NBA", market="REB", strategy="ratio_meanyr")
    # Analytical replaces point on dist-aware rows: analytical pred IQR is the
    # integer NB IQR (non-zero), not the (==0) point IQR of a constant EV column.
    assert row["g4_iqr_pred"] > 0.0
    # Sanity: point IQR of the EV column truly is 0, so any non-zero g4_iqr_pred
    # proves the gate switched from the point estimator to the analytical one.
    point_iqr = float(np.percentile(point_pred, 75) - np.percentile(point_pred, 25))
    assert point_iqr == 0.0


# ---------------------------------------------------------------------------
# Gate 5 — equal-mass ECE.
# ---------------------------------------------------------------------------


def test_gate5_ece_zero_for_oracle():
    y = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    assert _gate5_ece_equal_mass(y, y) == pytest.approx(0.0)


def test_gate5_ece_large_for_confidently_wrong():
    y = np.zeros(1000)
    p_model = np.full(1000, 0.9)  # confident "over" on outcomes that never hit
    assert _gate5_ece_equal_mass(p_model, y) == pytest.approx(0.9, abs=1e-9)


def test_gate5_ece_small_for_calibrated_stream():
    rng = np.random.default_rng(3)
    p_model = rng.uniform(0.0, 1.0, 20000)
    y = (rng.uniform(size=20000) < p_model).astype(float)  # calibrated by construction
    assert _gate5_ece_equal_mass(p_model, y) < 0.05


# ---------------------------------------------------------------------------
# Gate 5 debiased ECE (Ship 75 Step 0.5) — Roelofs 2022 correction for the
# N-dependent upward binning bias documented in
# /tmp/researcher_lifecycle_gate_audit.md. The raw ECE estimator falsely
# fails up to 44.6% of perfectly calibrated NFL-N=240 cells; the debiased
# variant subtracts the null-distribution mean per cell.
# ---------------------------------------------------------------------------


def test_ece_debias_offset_returns_positive_finite():
    """The null-ECE bias is strictly positive for finite N (Roelofs 2022)."""
    rng = np.random.default_rng(0)
    p_model = rng.uniform(0.05, 0.95, 300)
    offset = _ece_debias_offset(p_model, n_resamples=50, rng=np.random.default_rng(1729))
    assert offset > 0
    assert np.isfinite(offset)


def test_ece_debias_offset_decreases_with_n():
    """Binning bias shrinks as N grows — matches the audit's N-dependence table."""
    rng = np.random.default_rng(0)
    p_small = rng.uniform(0.05, 0.95, 240)
    p_large = rng.uniform(0.05, 0.95, 2000)
    off_small = _ece_debias_offset(p_small, n_resamples=80, rng=np.random.default_rng(1729))
    off_large = _ece_debias_offset(p_large, n_resamples=80, rng=np.random.default_rng(1729))
    assert off_small > off_large  # smaller N → larger bias


def test_ece_debias_offset_deterministic_under_seed():
    """Same seed → byte-identical offset (no determinism gate breakage)."""
    rng = np.random.default_rng(0)
    p_model = rng.uniform(0.05, 0.95, 500)
    a = _ece_debias_offset(p_model, n_resamples=40, rng=np.random.default_rng(1729))
    b = _ece_debias_offset(p_model, n_resamples=40, rng=np.random.default_rng(1729))
    assert a == b


def test_gate5_ece_debiased_perfect_calibration_near_zero():
    """A perfectly calibrated stream's debiased ECE should be near zero —
    raw ECE minus the matched null bias cancels by construction.
    """
    rng = np.random.default_rng(0)
    p_model = rng.uniform(0.05, 0.95, 400)  # NFL-N regime where raw is biased high
    y = (rng.uniform(size=len(p_model)) < p_model).astype(float)
    raw = _gate5_ece_equal_mass(p_model, y)
    debiased = _gate5_ece_debiased(p_model, y, n_resamples=100, rng=np.random.default_rng(1729))
    assert raw > 0.04  # raw is non-trivially biased at N=400
    assert abs(debiased) < raw  # debias shrinks magnitude toward 0
    assert abs(debiased) < 0.03  # near-zero after correction


def test_gate5_ece_debiased_preserves_real_miscalibration():
    """A confidently-wrong model still has nonzero debiased ECE — the bias
    correction subtracts the null floor, not the genuine signal.
    """
    y = np.zeros(1000)
    p_model = np.full(1000, 0.9)
    raw = _gate5_ece_equal_mass(p_model, y)
    debiased = _gate5_ece_debiased(p_model, y, n_resamples=40, rng=np.random.default_rng(1729))
    assert raw == pytest.approx(0.9, abs=1e-9)
    # Null bias on a degenerate constant-p stream is small; debiased ≈ raw.
    assert abs(debiased - raw) < 0.05


# ---------------------------------------------------------------------------
# Supersede S3 — Memmel 2003 paired Sharpe inference. Replaces the bare
# `sharpe_candidate > sharpe_baseline` rule, which had ~50% Type-I rate per
# the audit. Per Memmel: z = (SR_c - SR_b) / SE(SR_diff) with closed-form
# variance using the paired correlation; one-sided ship at z > 1.645.
# ---------------------------------------------------------------------------


def test_memmel_sharpe_z_identical_returns_zero():
    """Two identical return streams → SR_diff = 0 → z exactly 0."""
    rng = np.random.default_rng(7)
    r = rng.normal(0.001, 0.02, 500)
    sr_b, sr_c, z = _memmel_sharpe_z(r, r.copy())
    assert sr_b == pytest.approx(sr_c)
    assert z == pytest.approx(0.0, abs=1e-9)


def test_memmel_sharpe_z_near_identical_models_under_critical():
    """Two near-identical models (tiny iid noise) — z should sit well below
    1.645 the vast majority of the time, fixing the audit's coin-flip rate.
    """
    rng = np.random.default_rng(11)
    base = rng.normal(0.001, 0.02, 1000)
    type1_rejects = 0
    for i in range(40):
        rng_i = np.random.default_rng(100 + i)
        b = base + rng_i.normal(0, 1e-4, 1000)
        c = base + rng_i.normal(0, 1e-4, 1000)
        _, _, z = _memmel_sharpe_z(b, c)
        if z > _SUPERSEDE_S3_Z_MIN:
            type1_rejects += 1
    # Under one-sided α=0.05, expected ~2 of 40. The audit's bare-comparison
    # rule rejects ~20 of 40 (50%). Tolerate up to 6 for sample-size noise.
    assert type1_rejects <= 6


def test_memmel_sharpe_z_positive_when_candidate_genuinely_better():
    """Candidate with shifted mean returns gets z > 1.645 at adequate N."""
    rng = np.random.default_rng(13)
    n = 1500
    b = rng.normal(0.0005, 0.02, n)
    c = b + rng.normal(0.0008, 0.005, n)  # genuine positive shift
    sr_b, sr_c, z = _memmel_sharpe_z(b, c)
    assert sr_c > sr_b
    assert z > _SUPERSEDE_S3_Z_MIN


def test_memmel_sharpe_z_correlation_tightens_se():
    """Highly-correlated paired returns yield a smaller SE than uncorrelated
    pairs at the same per-series SR — the whole point of using paired inference.
    """
    rng = np.random.default_rng(17)
    n = 1000
    common = rng.normal(0, 0.02, n)
    # Correlated pair: shared common signal + tiny independent noise.
    b_corr = common + rng.normal(0.0005, 0.001, n)
    c_corr = common + rng.normal(0.0006, 0.001, n)  # tiny mean shift
    # Independent pair: same marginals but no shared shock.
    b_ind = rng.normal(common.mean(), common.std(), n) + rng.normal(0.0005, 0.001, n)
    c_ind = rng.normal(common.mean(), common.std(), n) + rng.normal(0.0006, 0.001, n)
    _, _, z_corr = _memmel_sharpe_z(b_corr, c_corr)
    _, _, z_ind = _memmel_sharpe_z(b_ind, c_ind)
    # Same expected SR difference, but correlated pair gets a sharper z.
    assert abs(z_corr) > abs(z_ind)


def test_memmel_sharpe_z_handles_zero_variance_returns():
    """A zero-variance return series (Sharpe undefined) returns z=0 gracefully."""
    rng = np.random.default_rng(19)
    flat = np.zeros(200)
    r = rng.normal(0.001, 0.02, 200)
    _, _, z = _memmel_sharpe_z(flat, r)
    assert np.isfinite(z) or z == 0.0  # implementation must not crash


# ---------------------------------------------------------------------------
# gate_row — model + oracle assembly, column set, unpriced handling.
# ---------------------------------------------------------------------------


def test_gate_row_full_column_set_and_oracle_identities():
    row = gate_row(_priced_frame(), "EV", league="NBA", market="PTS", strategy="t")
    expected = {
        "league",
        "market",
        "strategy",
        "n_rows",
        "g1_brier_diff_mean",
        "g1_brier_diff_ci_lo",
        "g1_brier_diff_ci_hi",
        "g1_brier_diff_mean_oracle",
        "g1_brier_diff_ci_lo_oracle",
        "g1_brier_diff_ci_hi_oracle",
        "g1_brier_skill_score",
        "g2_star_pred_mean",
        "g2_star_true_mean",
        "g2_star_abs_diff",
        "g2_star_sigma",
        "g2_star_z",
        "g2_star_z_oracle",
        "g3_bench_pred_mean",
        "g3_bench_true_mean",
        "g3_bench_abs_diff",
        "g3_bench_sigma",
        "g3_bench_z",
        "g3_bench_z_oracle",
        "g4_iqr_pred",
        "g4_iqr_true",
        "g4_iqr_ratio",
        "g4_iqr_ratio_oracle",
        "g5_ece",
        "g5_ece_oracle",
        "g5_ece_null_bias",
        "g5_ece_debiased",
        "g5_ece_debiased_oracle",
    }
    assert set(row) == expected
    # Oracle identities: segment z = 0, IQR ratio = 1, ECE = 0, Brier diff < 0.
    assert row["g2_star_z_oracle"] == pytest.approx(0.0)
    assert row["g3_bench_z_oracle"] == pytest.approx(0.0)
    assert row["g4_iqr_ratio_oracle"] == pytest.approx(1.0)
    assert row["g5_ece_oracle"] == pytest.approx(0.0)
    assert row["g1_brier_diff_ci_hi_oracle"] < 0  # oracle beats the (imperfect) book


def test_gate_row_no_book_columns_blanks_gates_1_and_5():
    """No P/Odds/Line at all → both book-touching gates blank, price-free gates still run."""
    df = _compressed_frame()
    row = gate_row(df, "EV", league="NBA", market="AST", strategy="t")
    assert row["g1_brier_diff_mean"] is None
    assert row["g1_brier_skill_score"] is None
    assert row["g5_ece"] is None
    assert row["g2_star_z"] is not None
    assert row["g4_iqr_ratio"] is not None


def test_gate_row_no_odds_but_line_present_blanks_g1_only():
    """Book-unpriced cells (Line + P present, no Odds) → Gate 1 blank, Gate 5 still computes.

    Gate 5 is model-only calibration: it needs P + Line + Result but NOT the book's
    Odds, so it should still produce a value when the market has a posted line and a
    model probability but no book quote to compare Brier against.
    """
    rng = np.random.default_rng(11)
    n = 2000
    meanyr = rng.uniform(2, 30, n)
    line = meanyr.copy()
    p = rng.uniform(0.05, 0.95, n)
    outcomes = rng.uniform(size=n) < p
    result = np.where(outcomes, line + 1.0, line - 1.0)
    df = pd.DataFrame({"MeanYr": meanyr, "Result": result, "EV": meanyr, "Line": line, "P": p})
    row = gate_row(df, "EV", league="NBA", market="AST", strategy="t")
    assert row["g1_brier_diff_mean"] is None  # no Odds → Gate 1 auto-pass at verdict time
    assert row["g1_brier_skill_score"] is None
    assert row["g5_ece"] is not None  # Gate 5 computes on P + Line alone


def test_write_gate_scorecard_sorts_and_overwrites(tmp_path):
    rows = [
        gate_row(_compressed_frame(seed=1), "EV", league="NFL", market="yards", strategy="t"),
        gate_row(_compressed_frame(seed=2), "EV", league="NBA", market="PTS", strategy="t"),
    ]
    out = tmp_path / "scorecard.csv"
    df = write_gate_scorecard(rows, out)
    assert out.exists()
    assert list(df["league"]) == ["NBA", "NFL"]  # sorted by (league, market)
    reread = pd.read_csv(out)
    assert "g4_iqr_ratio" in reread.columns
    assert len(reread) == 2
    # Snapshot overwrites (not appends): a one-row rewrite leaves one row.
    write_gate_scorecard([rows[0]], out)
    assert len(pd.read_csv(out)) == 1


# ---------------------------------------------------------------------------
# apply_thresholds — strict pass/fail wiring on the starter combo.
# ---------------------------------------------------------------------------


def _clean_row(**overrides) -> dict[str, object]:
    """Build a gate_row-shaped dict that clears every strict threshold."""
    row = {
        "league": "NBA",
        "market": "PTS",
        "strategy": "t",
        "n_rows": 2000,
        "g1_brier_diff_mean": -0.05,
        "g1_brier_diff_ci_lo": -0.07,
        "g1_brier_diff_ci_hi": -0.03,
        "g1_brier_diff_mean_oracle": -0.25,
        "g1_brier_diff_ci_lo_oracle": -0.27,
        "g1_brier_diff_ci_hi_oracle": -0.23,
        "g1_brier_skill_score": 0.10,
        "g2_star_pred_mean": 25.0,
        "g2_star_true_mean": 25.2,
        "g2_star_abs_diff": 0.2,
        "g2_star_sigma": 5.0,
        "g2_star_z": 0.04,
        "g2_star_z_oracle": 0.0,
        "g3_bench_pred_mean": 7.0,
        "g3_bench_true_mean": 7.1,
        "g3_bench_abs_diff": 0.1,
        "g3_bench_sigma": 1.5,
        "g3_bench_z": 0.07,
        "g3_bench_z_oracle": 0.0,
        "g4_iqr_pred": 8.0,
        "g4_iqr_true": 12.0,
        "g4_iqr_ratio": 0.67,
        "g4_iqr_ratio_oracle": 1.0,
        "g5_ece": 0.05,
        "g5_ece_oracle": 0.0,
    }
    row.update(overrides)
    return row


def test_apply_thresholds_clean_cell_ships():
    out = apply_thresholds(_clean_row())
    for g in ("g1", "g2", "g3", "g4", "g5"):
        assert out[f"{g}_pass"], f"{g} failed unexpectedly"
    assert out["ship"]


def test_apply_thresholds_each_gate_fails_when_threshold_exceeded():
    fails = {
        "g1": _clean_row(g1_brier_diff_ci_hi=0.001),  # CI doesn't exclude 0 below
        "g2": _clean_row(g2_star_z=0.51),  # over the 0.5 cap
        "g3": _clean_row(g3_bench_z=0.51),
        "g4": _clean_row(g4_iqr_ratio=0.49),  # below the 0.5 floor
        "g5": _clean_row(g5_ece=0.076),  # over the 0.075 cap
    }
    for gate, row in fails.items():
        out = apply_thresholds(row)
        assert not out[f"{gate}_pass"], f"{gate} should have failed"
        assert not out["ship"]


def test_apply_thresholds_g1_no_odds_auto_passes():
    """G1 blank (no book Odds) ⇒ auto-pass — model wins by default."""
    out = apply_thresholds(
        _clean_row(
            g1_brier_diff_mean=None,
            g1_brier_diff_ci_lo=None,
            g1_brier_diff_ci_hi=None,
            g1_brier_diff_mean_oracle=None,
            g1_brier_skill_score=None,
        )
    )
    assert out["g1_pass"]
    assert out["ship"]


def test_apply_thresholds_g4_nan_fails_under_strict():
    """G4 blank (binary tds markets, IQR(Result)=0) fails under strict — flagged for revisit."""
    out = apply_thresholds(_clean_row(g4_iqr_pred=None, g4_iqr_true=None, g4_iqr_ratio=None))
    assert not out["g4_pass"]
    assert not out["ship"]


def test_apply_thresholds_g5_blank_fails():
    """G5 blank (no P or no Line) fails — not auto-pass; the cell couldn't compute calibration."""
    out = apply_thresholds(_clean_row(g5_ece=None))
    assert not out["g5_pass"]
    assert not out["ship"]


def test_strict_thresholds_are_pinned():
    """Lock the strict starter combo so an accidental tweak fails CI."""
    assert _GATE1_CI_HI_MAX == 0.0
    assert _GATE2_STAR_Z_MAX == 0.5
    assert _GATE3_BENCH_Z_MAX == 0.5
    assert _GATE4_IQR_RATIO_MIN == 0.5
    assert _GATE5_ECE_MAX == 0.075


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


def test_load_test_set_keeps_optional_columns_when_present(tmp_path):
    df = pd.DataFrame(
        {
            "MeanYr": [10.0, 12.0, 14.0],
            "Result": [11.0, 13.0, 15.0],
            "EV": [10.5, 12.5, 14.5],
            "P": [0.55, 0.60, 0.50],
            "Odds": [0.45, 0.40, 0.50],
            "Line": [10.0, 12.0, 14.0],
        }
    )
    p = tmp_path / "NBA_PTS.csv"
    df.to_csv(p, index=False)
    loaded = load_test_set(p, "EV")
    assert {"P", "Odds", "Line"}.issubset(loaded.columns)
    assert len(loaded) == 3


def test_load_test_set_handles_missing_optional_columns(tmp_path):
    df = pd.DataFrame({"MeanYr": [10.0, 12.0], "Result": [11.0, 13.0], "EV": [10.5, 12.5]})
    p = tmp_path / "NBA_PTS.csv"
    df.to_csv(p, index=False)
    loaded = load_test_set(p, "EV")
    card = scorecard(loaded, "EV", strategy="t", league="NBA", market="PTS")
    assert card.brier_skill_score is None


# ---------------------------------------------------------------------------
# Supersede gate — S1 + S2 + S3 (research -> devel, supersede an incumbent)
# ---------------------------------------------------------------------------


def _supersede_pair(
    n: int = 800,
    seed: int = 42,
    candidate_calibration_noise: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (baseline, candidate) test-set frames sharing the same events.

    Set-up: the **book** is flat (over-prob = 0.5 for every event) — half the
    rows are real over-edges (``p_true > 0.5``), half are real under-edges. The
    **candidate** has correctly-sided ``EV`` (over-edge ⇒ EV above line; under-
    edge ⇒ EV below) and probability ``p_true``. The **baseline** has the same
    EVs perturbed by Gaussian noise (so it picks the wrong side sometimes) and a
    mid-regressed probability. ``candidate_calibration_noise > 0`` injects noise
    into the candidate's ``EV`` instead — used to invert the win condition.
    """
    rng = np.random.default_rng(seed)
    meanyr = rng.uniform(2, 30, n)
    line = meanyr.copy()
    p_true = rng.uniform(0.20, 0.80, n)
    outcomes = rng.uniform(size=n) < p_true
    result = np.where(outcomes, line + 1.0, line - 1.0)
    odds = np.full(n, 0.5)
    # Correctly-sided EV — bet side aligns with the truth.
    correct_ev = np.where(p_true > 0.5, line + 1.0, line - 1.0)
    # Baseline EV is the correct side perturbed by enough noise that ~25% of
    # rows flip sides → baseline bets the wrong side on those events.
    baseline_ev = correct_ev + rng.normal(0, 1.2, n)
    baseline_p = 0.5 + 0.4 * (p_true - 0.5)  # mid-regressed prob
    # Candidate gets correctly-sided EV and tracks p_true exactly. Rejection
    # scenarios swamp its probability with Gaussian noise — large noise makes
    # the candidate's P uninformative and its Brier worse than baseline's.
    candidate_ev = correct_ev.copy()
    candidate_p = np.clip(p_true + rng.normal(0, candidate_calibration_noise, n), 1e-3, 1.0 - 1e-3)
    b_df = pd.DataFrame(
        {
            "MeanYr": meanyr,
            "Result": result.astype(float),
            "EV": baseline_ev,
            "Line": line,
            "Odds": odds,
            "P": baseline_p,
        }
    )
    c_df = pd.DataFrame(
        {
            "MeanYr": meanyr,
            "Result": result.astype(float),
            "EV": candidate_ev,
            "Line": line,
            "Odds": odds,
            "P": candidate_p,
        }
    )
    return b_df, c_df


def test_paired_brier_ci_positive_when_candidate_beats_baseline():
    # Candidate is well-calibrated; baseline is regressed toward 0.5 → candidate
    # Brier strictly lower per event ⇒ d_i > 0 ⇒ CI lo > 0. Need a decent N for
    # the bootstrap CI to be tight enough to strictly exclude 0.
    b_df, c_df = _supersede_pair(n=4000)
    res = _supersede_paired_brier_ci(b_df, c_df)
    assert res is not None
    n, mean, ci_lo, _ci_hi = res
    assert n == len(b_df)
    assert mean > 0
    assert ci_lo > 0


def test_paired_brier_ci_negative_when_baseline_beats_candidate():
    # Add enough noise to the candidate's P that it's WORSE than the baseline's
    # mid-regressed-but-still-signal P. Needs n large + noise large for the CI
    # to land strictly below 0.
    b_df, c_df = _supersede_pair(n=4000, candidate_calibration_noise=0.8)
    res = _supersede_paired_brier_ci(b_df, c_df)
    assert res is not None
    _, mean, _ci_lo, ci_hi = res
    assert mean < 0
    assert ci_hi < 0


def test_paired_brier_ci_returns_none_when_inputs_lack_p():
    df = pd.DataFrame({"MeanYr": [1.0], "Result": [1.0], "EV": [1.0], "Line": [1.0]})
    assert _supersede_paired_brier_ci(df, df) is None


def test_paired_brier_ci_returns_none_on_empty_intersection():
    b_df, c_df = _supersede_pair(n=50)
    # Disjoint indices — no shared events.
    c_df.index = c_df.index + 1000
    assert _supersede_paired_brier_ci(b_df, c_df) is None


def test_test_set_to_bet_frame_picks_ev_side_and_decimal_payout():
    # EV > Line ⇒ bet over ⇒ Hit = (Result >= Line); payout = 1/(1-Odds).
    df = pd.DataFrame(
        {
            "MeanYr": [10.0, 10.0],
            "Result": [12.0, 8.0],
            "EV": [11.0, 11.0],  # both EV > Line ⇒ both bet over
            "Line": [10.0, 10.0],
            "Odds": [0.4, 0.4],  # book under-prob 0.4 ⇒ over-prob 0.6
            "P": [0.6, 0.6],
        }
    )
    bets = _test_set_to_bet_frame(df, "EV")
    assert len(bets) == 2
    assert (bets["Platform"] == "Sleeper").all()
    # Boost = decimal odds = 1 / book_over_prob = 1 / 0.6 ≈ 1.667
    assert bets["Boost"].iloc[0] == pytest.approx(1.0 / 0.6)
    # Hit: row 0 Result >= Line ⇒ True; row 1 ⇒ False.
    assert bool(bets["Hit"].iloc[0]) is True
    assert bool(bets["Hit"].iloc[1]) is False


def test_test_set_to_bet_frame_returns_empty_without_odds():
    df = pd.DataFrame({"MeanYr": [1.0], "Result": [1.0], "EV": [1.0], "Line": [1.0], "P": [0.5]})
    bets = _test_set_to_bet_frame(df, "EV")
    assert bets.empty


def test_test_set_to_bet_frame_picks_under_when_ev_below_line():
    # EV < Line ⇒ bet under ⇒ Hit when Result < Line; book_under_prob = Odds.
    df = pd.DataFrame(
        {
            "MeanYr": [10.0],
            "Result": [8.0],
            "EV": [9.0],  # EV < Line ⇒ bet under
            "Line": [10.0],
            "Odds": [0.55],  # book under-prob 0.55 ⇒ decimal odds = 1/0.55
            "P": [0.4],
        }
    )
    bets = _test_set_to_bet_frame(df, "EV")
    assert bets["Boost"].iloc[0] == pytest.approx(1.0 / 0.55)
    # Hit: Result (8) < Line (10) ⇒ under wins.
    assert bool(bets["Hit"].iloc[0]) is True
    # Model probability on the UNDER side = 1 - P = 0.6.
    assert bets["Model P"].iloc[0] == pytest.approx(0.6)


def test_paired_sharpe_returns_finite_pair():
    b_df, c_df = _supersede_pair()
    res = _supersede_paired_sharpe(b_df, c_df, "EV")
    assert res is not None
    sb, sc, z = res
    assert np.isfinite(sb)
    assert np.isfinite(sc)
    assert np.isfinite(z)


def test_paired_sharpe_candidate_higher_when_better_calibrated():
    # Calibrated candidate vs regressed baseline ⇒ candidate Kelly stakes are
    # more aligned with true win probability ⇒ higher long-run Sharpe AND a
    # positive Memmel z (the paired-inference test that supersedes the bare
    # ``sc > sb`` rule).
    b_df, c_df = _supersede_pair(n=2000)
    res = _supersede_paired_sharpe(b_df, c_df, "EV")
    assert res is not None
    sb, sc, z = res
    assert sc > sb
    assert z > 0


def test_paired_sharpe_returns_none_on_empty_intersection():
    b_df, c_df = _supersede_pair(n=20)
    c_df.index = c_df.index + 1000
    assert _supersede_paired_sharpe(b_df, c_df, "EV") is None


def test_supersede_verdict_ships_when_all_three_pass():
    # A clean calibrated candidate that clears all 5 gates outright + beats a
    # regressed-toward-0.5 baseline on paired Brier + has higher Sharpe ⇒ ship.
    b_df, c_df = _supersede_pair(n=4000, seed=11)
    v = supersede_verdict(b_df, c_df, "EV", strategy="cand")
    assert v["s1_pass"] is True
    assert v["s2_pass"] is True
    assert v["s3_pass"] is True
    assert v["ship"] is True


def test_supersede_verdict_holds_when_candidate_worse():
    # Candidate worse than baseline ⇒ S2 fails (paired Brier CI negative) and
    # S3 fails (lower Sharpe). The verdict is HOLD even if S1 happens to pass.
    b_df, c_df = _supersede_pair(n=2000, seed=13, candidate_calibration_noise=0.4)
    v = supersede_verdict(b_df, c_df, "EV", strategy="cand")
    assert v["s2_pass"] is False
    assert v["ship"] is False


def test_supersede_verdict_holds_when_baseline_unpriced():
    # Baseline has no Odds column ⇒ S2 and S3 both return None ⇒ HOLD even if
    # the candidate would clear S1 on its own.
    _b, c_df = _supersede_pair(n=200, seed=21)
    b_unpriced = c_df.drop(columns=["Odds"])
    v = supersede_verdict(b_unpriced, c_df, "EV", strategy="cand")
    assert v["s2_pass"] is False
    assert v["s3_pass"] is False
    assert v["ship"] is False


# ---------------------------------------------------------------------------
# --live-window mode (Stage 0 deliverable 0.3)
# ---------------------------------------------------------------------------

import math
from datetime import datetime, timedelta

from click.testing import CliRunner

from sportstradamus.training.scorecard import (
    _history_to_eval_frame,
    _make_meanyr_lookup_from_gamelog,
    main,
)


def _build_live_offer(line, bet, model_p, books_p):
    return (line, 1.0, "Underdog", bet, model_p, books_p, float("nan"), float("nan"), float("nan"))


def _build_live_history_fixture(n: int = 60, market: str = "PTS") -> pd.DataFrame:
    rng = np.random.default_rng(13)
    rows = []
    today = datetime(2026, 5, 20)
    for idx in range(n):
        date = (today - timedelta(days=int(rng.integers(0, 25)))).strftime("%Y-%m-%d")
        line = float(rng.uniform(8.0, 30.0))
        bet = "Over" if rng.random() > 0.5 else "Under"
        model_p = float(rng.uniform(0.45, 0.65))
        books_p = float(rng.uniform(0.45, 0.55))
        actual = float(rng.normal(line, line * 0.18))
        rows.append(
            {
                "Player": f"Player_{idx}",
                "League": "NBA",
                "Team": "HOME",
                "Date": date,
                "Market": market,
                "Model EV": line + rng.normal(0, 1.5),
                "Books EV": line,
                "Dist": "SkewNormal",
                "CV": 0.3,
                "Model Param": line,
                "Gate": np.nan,
                "Temperature": 1.0,
                "Disp Cal": 1.0,
                "Step": "test",
                "Offers": [_build_live_offer(line, bet, model_p, books_p)],
                "Actual": actual,
            }
        )
    return pd.DataFrame(rows)


def test_history_to_eval_frame_renames_and_normalizes_columns():
    history = _build_live_history_fixture(n=40, market="PTS")
    lookup = lambda player, market, date: 22.0  # noqa: E731 — closure for fixture
    frame = _history_to_eval_frame(
        history, league="NBA", market="PTS", window_days=30, meanyr_lookup=lookup
    )
    assert list(frame.columns) == ["MeanYr", "Result", "EV", "P", "Odds", "Line"]
    assert (frame["MeanYr"] == 22.0).all()
    assert frame["EV"].notna().all()
    # Odds column is the book UNDER prob — flipped relative to the bet's side.
    # Since the lookup is constant and rows survive after dropna(), we should
    # have at least most of the input rows present.
    assert len(frame) > 0


def test_history_to_eval_frame_empty_history_returns_empty_schema():
    frame = _history_to_eval_frame(
        pd.DataFrame(),
        league="NBA",
        market="PTS",
        window_days=30,
        meanyr_lookup=lambda p, m, d: 0.0,
    )
    assert frame.empty
    assert list(frame.columns) == ["MeanYr", "Result", "EV", "P", "Odds", "Line"]


def test_history_to_eval_frame_filters_to_league_market_and_window():
    today = datetime(2026, 5, 20)
    rows = []
    # In-scope: NBA + PTS within window
    for idx in range(5):
        rows.append(
            {
                "Player": f"A_{idx}",
                "League": "NBA",
                "Date": today.strftime("%Y-%m-%d"),
                "Market": "PTS",
                "Model EV": 20.0,
                "Offers": [_build_live_offer(20.0, "Over", 0.55, 0.50)],
                "Actual": 22.0,
            }
        )
    # Out-of-scope league
    rows.append(
        {
            "Player": "B",
            "League": "WNBA",
            "Date": today.strftime("%Y-%m-%d"),
            "Market": "PTS",
            "Model EV": 20.0,
            "Offers": [_build_live_offer(20.0, "Over", 0.55, 0.50)],
            "Actual": 22.0,
        }
    )
    # Out-of-scope market
    rows.append(
        {
            "Player": "C",
            "League": "NBA",
            "Date": today.strftime("%Y-%m-%d"),
            "Market": "REB",
            "Model EV": 20.0,
            "Offers": [_build_live_offer(20.0, "Over", 0.55, 0.50)],
            "Actual": 22.0,
        }
    )
    # Out-of-scope date
    rows.append(
        {
            "Player": "D",
            "League": "NBA",
            "Date": (today - timedelta(days=120)).strftime("%Y-%m-%d"),
            "Market": "PTS",
            "Model EV": 20.0,
            "Offers": [_build_live_offer(20.0, "Over", 0.55, 0.50)],
            "Actual": 22.0,
        }
    )
    history = pd.DataFrame(rows)
    frame = _history_to_eval_frame(
        history,
        league="NBA",
        market="PTS",
        window_days=30,
        meanyr_lookup=lambda p, m, d: 18.0,
    )
    assert len(frame) == 5


def test_make_meanyr_lookup_returns_nan_when_gamelog_empty():
    lookup = _make_meanyr_lookup_from_gamelog(pd.DataFrame(), date_col="gameDate")
    assert math.isnan(lookup("AnyPlayer", "PTS", pd.Timestamp("2026-05-20")))


def test_make_meanyr_lookup_returns_nan_when_market_column_missing():
    gl = pd.DataFrame(
        {
            "playerName": ["Player_X"] * 5,
            "gameDate": pd.date_range("2026-04-01", periods=5, freq="D"),
            "REB": [10, 11, 12, 9, 8],
        }
    )
    lookup = _make_meanyr_lookup_from_gamelog(gl, date_col="gameDate")
    assert math.isnan(lookup("Player_X", "PTS", pd.Timestamp("2026-05-20")))


def test_make_meanyr_lookup_returns_mean_of_prior_year():
    gl = pd.DataFrame(
        {
            "playerName": ["Player_X"] * 4,
            "gameDate": [
                pd.Timestamp("2026-05-10"),
                pd.Timestamp("2026-05-12"),
                pd.Timestamp("2026-05-15"),
                pd.Timestamp("2026-05-19"),  # before the lookup date 2026-05-20
            ],
            "PTS": [10.0, 20.0, 30.0, 40.0],
        }
    )
    lookup = _make_meanyr_lookup_from_gamelog(gl, date_col="gameDate")
    val = lookup("Player_X", "PTS", pd.Timestamp("2026-05-20"))
    assert val == pytest.approx(25.0)


def test_live_window_cli_unknown_league_filter_errors(monkeypatch):
    runner = CliRunner()
    history = pd.DataFrame()
    monkeypatch.setattr("sportstradamus.training.scorecard.read_history", lambda: history)
    result = runner.invoke(main, ["--live-window", "30"])
    assert result.exit_code != 0
    assert "empty" in result.output.lower()


def test_live_window_cli_smoke_with_mock_stats(monkeypatch):
    """Full --live-window run with mocked Stats loading — no real gamelog needed."""
    history = _build_live_history_fixture(n=80, market="PTS")
    monkeypatch.setattr("sportstradamus.training.scorecard.read_history", lambda: history)
    monkeypatch.setattr(
        "sportstradamus.training.scorecard._load_league_stats_lookup",
        lambda league: lambda player, market, date: 20.0,
    )
    runner = CliRunner()
    result = runner.invoke(
        main, ["--live-window", "30", "--league", "NBA", "--market", "PTS", "--no-log"]
    )
    assert result.exit_code == 0, result.output
    assert "NBA_PTS" in result.output
    assert "live_30d" in result.output


def test_live_window_cli_rejects_conflicting_flags(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sportstradamus.training.scorecard.read_history",
        lambda: _build_live_history_fixture(n=10),
    )
    runner = CliRunner()
    fake_csv = tmp_path / "fake.csv"
    fake_csv.write_text("MeanYr,Result,EV\n1,1,1\n")
    result = runner.invoke(
        main,
        ["--live-window", "30", "--baseline", str(fake_csv), "--candidate", str(fake_csv)],
    )
    assert result.exit_code != 0
    assert "cannot combine" in result.output.lower()
