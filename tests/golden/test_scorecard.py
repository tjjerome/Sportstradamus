"""Unit tests for ``training.scorecard`` — the offline ship-gate harness.

Exercises the numeric path (decile binning, compression ratio, scorecard, the five
offline ship gates, and their deterministic-1/0 oracle) on synthetic test-set frames
so no trained model, network, or plotting backend is required.
"""

import hashlib
import importlib.resources as pkg_resources
import json
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from sportstradamus import data
from sportstradamus.helpers.distributions import _dp_cdf_pmf, _dp_ppf
from sportstradamus.training import group_conditional_cdf as receiving
from sportstradamus.training import posthoc
from sportstradamus.training.group_conditional_cdf._pool import fixed_pool_blob
from sportstradamus.training.model_strategy import (
    STRATEGY_SIGNATURE_CSV_COLUMN,
    STRATEGY_STATUS_CSV_COLUMN,
    STRUCTURAL_STRATEGY_CSV_COLUMN,
    artifact_identity_columns,
    build_artifact_identity,
    get_strategy,
    strategy_controls,
)
from sportstradamus.training.scorecard import (
    _GATE1_CI_HI_MAX,
    _GATE1_NONINF_MARGIN,
    _GATE2_STAR_Z_MAX,
    _GATE3_BENCH_Z_MAX,
    _GATE4_KS_NOISE_COEF,
    _GATE4_PIT_KS_DELTA,
    _GATE5_ECE_MAX,
    _GATE6_FIRE_ON,
    _GATE6_MARGIN,
    _GATE6_STAR_REF_BASKETBALL,
    _SUPERSEDE_S3_Z_MIN,
    ACTUAL_COL,
    DECILE_COL,
    DEFAULT_PRED_COL,
    TARGET_NORM_NONE,
    _apply_pit_recal_by_row,
    _decode_sn_loc_scale,
    _dispersion_diagnostics,
    _ece_debias_offset,
    _gate1_brier_ci,
    _gate4_iqr_spread,
    _gate4_pit_ks_threshold,
    _gate5_ece_debiased,
    _gate5_ece_equal_mass,
    _gate23_segment_match,
    _history_to_eval_frame,
    _infer_dist_from_columns,
    _iqr_pred_analytical,
    _ks_uniform,
    _make_meanyr_lookup_from_gamelog,
    _memmel_sharpe_z,
    _pred_cdf_pmf,
    _pred_midpit,
    _pred_ppf,
    _randomized_pit_draws,
    _randomized_pit_ks,
    _segment_masks,
    _signed_decode_strategy,
    _supersede_paired_brier_ci,
    _supersede_paired_sharpe,
    _tail_ks_uniform,
    _test_set_to_bet_frame,
    _zinb_ppf,
    apply_thresholds,
    compute_gates,
    decile_table,
    fit_skewnorm_dispersion_c,
    fit_skewnorm_dispersion_skew,
    gate_row,
    load_test_set,
    main,
    min_gate_slack,
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
            "Mean10": np.full(n, 4.0),  # location offset re-added to loc; IQR is location-free
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


def test_resolve_decode_strategy_substitutes_default_for_none(monkeypatch):
    """Withheld SkewNormal cells trained under the ``--target-strategy``
    default (the ``none`` slug) must decode with ``ratio_meanyr``, not the
    ``WITHHELD`` sentinel the ship-config projection collapses them to —
    otherwise the g4 IQR stays in normalized ratio-units and false-fails.
    """
    import sportstradamus.training.scorecard as sc

    fake_meta = {
        "NFL": {
            "carries": {
                "dist": "SkewNormal",
                "shipped": "withheld",
                "target_normalization": "none",
            },
            "receptions": {
                "dist": "SkewNormal",
                "shipped": "withheld",
                "target_normalization": "centered_additive_mean10",
            },
            "passing yards": {
                "dist": "SkewNormal",
                "shipped": "devel",
                "target_normalization": "ratio_meanyr",
            },
        }
    }
    monkeypatch.setattr(sc, "_cached_stat_meta", lambda: fake_meta)

    assert sc._resolve_decode_strategy("NFL", "carries") == "ratio_meanyr"
    # A real per-cell slug survives while withheld (not collapsed to WITHHELD).
    assert sc._resolve_decode_strategy("NFL", "receptions") == "centered_additive_mean10"
    assert sc._resolve_decode_strategy("NFL", "passing-yards") == "ratio_meanyr"


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
# Gate 6 — anti-shrinkage (all cells; the corr anchor, not the normalization, scopes it).
# ---------------------------------------------------------------------------


def _cohort_frame(
    n: int = 200, seed: int = 11, *, shrink: float = 0.8, anchored: bool = True
) -> pd.DataFrame:
    """A frame of stable players (recent form ≈ season baseline) whose ``Blended_EV`` is
    ``shrink``× their recent form. ``anchored`` ties ``Result`` to ``Mean10`` so the Gate-6
    ``corr`` anchor passes; set it False to model a MIN-like cell where recent form doesn't
    predict the outcome.
    """
    rng = np.random.default_rng(seed)
    meanyr = rng.uniform(2.0, 20.0, n)
    mean10 = meanyr * (1.0 + rng.uniform(-0.05, 0.05, n))  # all inside the ±0.12 stable band
    result = mean10 + rng.normal(0.0, 0.5, n) if anchored else rng.uniform(2.0, 20.0, n)
    return pd.DataFrame(
        {
            "MeanYr": meanyr,
            "Result": result,
            "Blended_EV": shrink * mean10,
            "Mean10": mean10,
            "Player": [f"P{i % 40}" for i in range(n)],
            "SN_Loc": np.zeros(n),
            "SN_Scale": np.ones(n),
            "SN_Alpha": np.zeros(n),
        }
    )


def test_gate6_flags_overshrunk_ratio_meanyr_cohort():
    """Gate 6 fails a ratio_meanyr SkewNormal cell whose stable stars are predicted well below
    their recent form — the defect the outcome-scored gates miss because the holdout shares it.
    """
    row = apply_thresholds(
        gate_row(
            _cohort_frame(shrink=0.8, anchored=True),
            "Blended_EV",
            league="WNBA",
            market="FGA",
            strategy="ratio_meanyr",
            decode_strategy="ratio_meanyr",
        )
    )
    assert row["g6_recent_corr"] >= _GATE6_FIRE_ON
    assert row["g6_star_ci_hi"] < row["g6_star_ref"] - _GATE6_MARGIN
    assert not row["g6_pass"]
    assert not row["ship"]


def test_gate6_exempts_unanchored_cell():
    """When recent form doesn't predict the outcome (corr below the anchor — the MIN case),
    Gate 6 auto-passes the same over-shrunk frame: recent form isn't a valid yardstick.
    """
    row = apply_thresholds(
        gate_row(
            _cohort_frame(shrink=0.8, anchored=False),
            "Blended_EV",
            league="WNBA",
            market="MIN",
            strategy="ratio_meanyr",
            decode_strategy="ratio_meanyr",
        )
    )
    assert row["g6_recent_corr"] < _GATE6_FIRE_ON
    assert row["g6_star_ci_hi"] is None
    assert row["g6_pass"]


def test_gate6_gates_non_ratio_meanyr_normalization():
    """Gate 6 is no longer cohort-scoped: a centered (non-ratio_meanyr) cell whose stable stars
    are over-shrunk is gated and fails — the WNBA-FGA mean-regression the narrow scope missed.
    """
    row = apply_thresholds(
        gate_row(
            _cohort_frame(shrink=0.8, anchored=True),
            "Blended_EV",
            league="WNBA",
            market="REB",
            strategy="centered_additive_mean10",
            decode_strategy="centered_additive_mean10",
        )
    )
    assert _GATE6_STAR_REF_BASKETBALL == 0.95  # relaxed from 0.98 — allow ~5% star shrinkage
    assert row["g6_star_ref"] == _GATE6_STAR_REF_BASKETBALL
    assert row["g6_star_ci_hi"] is not None
    assert row["g6_star_ci_hi"] < row["g6_star_ref"] - _GATE6_MARGIN
    assert not row["g6_pass"]
    assert not row["ship"]


def test_gate6_applies_regardless_of_distribution_family():
    """Gate 6 reads only the served EV against recent form / the outcome, so it fires for count
    families too: strip the SkewNormal params and it still computes on a bare data frame.
    """
    from sportstradamus.training.scorecard import _gate6_legs

    frame = _cohort_frame(shrink=0.8, anchored=True).drop(
        columns=["SN_Loc", "SN_Scale", "SN_Alpha"]
    )
    g6 = _gate6_legs(frame, "Blended_EV", league="WNBA")
    assert g6["g6_recent_corr"] >= _GATE6_FIRE_ON
    assert g6["g6_star_ci_hi"] is not None
    assert g6["g6_star_ci_hi"] < g6["g6_star_ref"] - _GATE6_MARGIN
    assert g6["g6_star_ref"] == _GATE6_STAR_REF_BASKETBALL
    assert g6["g6_citl_ci_hi"] is not None  # CITL runs regardless of distribution family


def test_gate6_hysteresis_and_over_constants_exist():
    from sportstradamus.training.scorecard import (
        _GATE6_FIRE_ON,
        _GATE6_KEEP_ON,
        _GATE6_OVER_MIN_MEAN,
    )

    # Deadband: a fresh cell starts judging at FIRE_ON, a flagged cell keeps down to KEEP_ON.
    assert _GATE6_FIRE_ON == 0.58
    assert _GATE6_KEEP_ON == 0.52
    assert _GATE6_KEEP_ON < _GATE6_FIRE_ON
    # Over-leg degenerate-count guard: only test a count bench whose realized mean clears 1.
    assert _GATE6_OVER_MIN_MEAN == 1.0


def test_gate6_anchored_hysteresis():
    from sportstradamus.training.scorecard import _gate6_anchored

    # Above fire-on: always judged, regardless of prior.
    assert _gate6_anchored(0.60, None) is True
    assert _gate6_anchored(0.58, False) is True
    # In the deadband: judged only if it fired last run.
    assert _gate6_anchored(0.56, True) is True
    assert _gate6_anchored(0.56, None) is False
    assert _gate6_anchored(0.56, False) is False
    # Below keep-on: never judged.
    assert _gate6_anchored(0.51, True) is False
    # Non-finite corr (degenerate cell): not anchored.
    assert _gate6_anchored(np.nan, True) is False


def _legs_frame(n=200, seed=11, *, ev_over_form=0.8, result_over_form=1.0, anchored=True):
    """Stable-star frame: ``Blended_EV = ev_over_form × Mean10`` and ``Result = result_over_form ×
    base + noise``, where ``base`` is ``Mean10`` (anchored — high corr) or an independent uniform
    (unanchored — low corr, the MIN case). Lets the recent-form leg and the CITL-under leg be
    driven independently.
    """
    rng = np.random.default_rng(seed)
    meanyr = rng.uniform(2.0, 20.0, n)
    mean10 = meanyr * (1.0 + rng.uniform(-0.05, 0.05, n))  # inside the ±0.12 stable band
    base = mean10 if anchored else rng.uniform(2.0, 20.0, n)
    return pd.DataFrame(
        {
            "MeanYr": meanyr,
            "Mean10": mean10,
            "Result": result_over_form * base + rng.normal(0.0, 0.3, n),
            "Blended_EV": ev_over_form * mean10,
            "Player": [f"P{i % 40}" for i in range(n)],
            "SN_Loc": np.zeros(n),
            "SN_Scale": np.ones(n),
            "SN_Alpha": np.zeros(n),
        }
    )


def _count_legs_frame(n=200, seed=7, *, bench_ev_over_result=1.3, bench_mean=2.0):
    """Count/ZINB frame (``R``/``NB_P``/``Gate`` ⇒ ``_infer_dist`` == ``"ZINB"``) whose stable
    BENCH (low MeanYr) is over-predicted vs a realized mean controllable above/below the guard.
    """
    rng = np.random.default_rng(seed)
    meanyr = np.concatenate([rng.uniform(0.5, 3.0, n // 2), rng.uniform(8.0, 20.0, n // 2)])
    mean10 = meanyr * (1.0 + rng.uniform(-0.05, 0.05, n))
    bench = meanyr <= np.quantile(meanyr, 0.25)
    result = np.where(bench, bench_mean, mean10) + rng.normal(0.0, 0.2, n)
    ev = np.where(bench, bench_ev_over_result * bench_mean, mean10)
    return pd.DataFrame(
        {
            "MeanYr": meanyr,
            "Mean10": mean10,
            "Result": result,
            "Blended_EV": ev,
            "Player": [f"P{i % 40}" for i in range(n)],
            "R": np.ones(n),
            "NB_P": np.full(n, 0.5),
            "Gate": np.zeros(n),
        }
    )


def _deadband_frame(n=2000, seed=0):
    """A frame whose ``corr(Mean10, Result)`` sits in the ``[0.52, 0.58)`` hysteresis deadband, so
    the recent-form leg is judged only when the prior run fired. Large ``n`` keeps the empirical
    corr tight around the tuned ~0.55.
    """
    rng = np.random.default_rng(seed)
    meanyr = rng.uniform(5.0, 20.0, n)
    mean10 = meanyr * (1.0 + rng.uniform(-0.05, 0.05, n))
    result = mean10 + rng.normal(0.0, 1.52 * mean10.std(), n)  # tuned so pearson r ≈ 0.55
    return pd.DataFrame(
        {
            "MeanYr": meanyr,
            "Mean10": mean10,
            "Result": result,
            "Blended_EV": 0.8 * mean10,
            "Player": [f"P{i % 80}" for i in range(n)],
            "SN_Loc": np.zeros(n),
            "SN_Scale": np.ones(n),
            "SN_Alpha": np.zeros(n),
        }
    )


def test_gate6_legs_recent_form_fires_when_anchored():
    from sportstradamus.training.scorecard import _gate6_legs

    g6 = _gate6_legs(
        _legs_frame(ev_over_form=0.8), "Blended_EV", league="WNBA", prior_g6_fired=None
    )
    assert g6["g6_recent_corr"] >= 0.58  # anchored at fire-on
    assert g6["g6_star_ci_hi"] is not None
    assert g6["g6_star_ci_hi"] < g6["g6_star_ref"] - _GATE6_MARGIN  # over-shrinks vs recent form


def test_gate6_legs_recent_form_exempt_below_keep_on():
    from sportstradamus.training.scorecard import _gate6_legs

    # Unanchored (Result independent of Mean10) ⇒ low corr ⇒ recent-form blank, but CITL still runs.
    g6 = _gate6_legs(
        _legs_frame(ev_over_form=0.8, anchored=False),
        "Blended_EV",
        league="WNBA",
        prior_g6_fired=None,
    )
    assert g6["g6_star_ci_hi"] is None  # recent-form exempt
    assert g6["g6_citl_ci_hi"] is not None  # CITL is not anchor-gated


def test_gate6_legs_citl_under_fires_against_outcome():
    from sportstradamus.training.scorecard import _gate6_legs

    # EV 0.85× the realized outcome on stable stars: CITL fires regardless of the recent-form leg.
    g6 = _gate6_legs(
        _legs_frame(ev_over_form=0.85, anchored=True),
        "Blended_EV",
        league="NBA",
        prior_g6_fired=None,
    )
    assert g6["g6_citl_ci_hi"] is not None
    assert g6["g6_citl_ci_hi"] < 1.0 - _GATE6_MARGIN


def test_gate6_legs_citl_silent_when_pred_not_below_outcome():
    from sportstradamus.training.scorecard import _gate6_legs

    # Pred below recent form but ABOVE the realized outcome (the outcome vindicates the regression).
    g6 = _gate6_legs(
        _legs_frame(ev_over_form=0.8, result_over_form=0.74, anchored=False),
        "Blended_EV",
        league="NBA",
        prior_g6_fired=None,
    )
    assert g6["g6_citl_ci_hi"] >= 1.0 - _GATE6_MARGIN  # does not fire


def test_gate6_legs_over_leg_fires_on_count_bench():
    from sportstradamus.training.scorecard import _gate6_legs

    g6 = _gate6_legs(
        _count_legs_frame(bench_ev_over_result=1.3),
        "Blended_EV",
        league="WNBA",
        prior_g6_fired=None,
    )
    assert g6["g6_over_ci_lo"] is not None
    assert g6["g6_over_ci_lo"] > 1.0 + _GATE6_MARGIN


def test_gate6_legs_over_leg_silent_under_degenerate_guard():
    from sportstradamus.training.scorecard import _gate6_legs

    # Same over-prediction but bench realized mean < 1 ⇒ guard blanks the over-leg.
    g6 = _gate6_legs(
        _count_legs_frame(bench_ev_over_result=1.3, bench_mean=0.4),
        "Blended_EV",
        league="WNBA",
        prior_g6_fired=None,
    )
    assert g6["g6_over_ci_lo"] is None


def test_gate6_legs_over_leg_blank_for_non_count_family():
    from sportstradamus.training.scorecard import _gate6_legs

    g6 = _gate6_legs(_legs_frame(), "Blended_EV", league="WNBA", prior_g6_fired=None)  # SkewNormal
    assert g6["g6_over_ci_lo"] is None


def test_gate6_passes_ors_three_legs():
    from sportstradamus.training.scorecard import _gate6_passes

    ok = {"g6_star_ci_hi": None, "g6_star_ref": None, "g6_citl_ci_hi": None, "g6_over_ci_lo": None}
    assert _gate6_passes(ok) is True  # all blank ⇒ auto-pass
    assert _gate6_passes({**ok, "g6_star_ci_hi": 0.80, "g6_star_ref": 0.95}) is False  # recent-form
    assert _gate6_passes({**ok, "g6_citl_ci_hi": 0.90}) is False  # CITL-under (< 1 - 0.03)
    assert _gate6_passes({**ok, "g6_citl_ci_hi": 0.99}) is True
    assert _gate6_passes({**ok, "g6_over_ci_lo": 1.10}) is False  # over (> 1 + 0.03)
    assert _gate6_passes({**ok, "g6_over_ci_lo": 1.00}) is True


def test_recent_form_fired_reads_prior_row():
    from sportstradamus.training.scorecard import recent_form_fired

    assert recent_form_fired({"g6_star_ci_hi": 0.80, "g6_star_ref": 0.95}) is True
    assert recent_form_fired({"g6_star_ci_hi": 0.99, "g6_star_ref": 0.95}) is False
    assert recent_form_fired({"g6_star_ci_hi": None, "g6_star_ref": None}) is False


def test_gate6_fails_ship_through_gate_row():
    # Anchored under-prediction: recent-form AND CITL fire, so g6 fails and ship is False.
    df = _legs_frame(ev_over_form=0.85, anchored=True, n=400)
    row = apply_thresholds(gate_row(df, "Blended_EV", league="NBA", market="x", strategy="x"))
    assert row["g6_citl_ci_hi"] is not None and row["g6_citl_ci_hi"] < 1.0 - _GATE6_MARGIN
    assert row["g6_pass"] is False
    assert row["ship"] is False


def test_gate_row_threads_prior_g6_fired():
    df = _deadband_frame()
    corr = np.corrcoef(df["Mean10"], df["Result"])[0, 1]
    assert 0.52 <= corr < 0.58  # precondition: the frame sits in the deadband
    judged = gate_row(
        df, "Blended_EV", league="WNBA", market="x", strategy="x", prior_g6_fired=True
    )
    exempt = gate_row(
        df, "Blended_EV", league="WNBA", market="x", strategy="x", prior_g6_fired=None
    )
    assert judged["g6_star_ci_hi"] is not None  # prior fired ⇒ kept on in the band
    assert exempt["g6_star_ci_hi"] is None  # no prior ⇒ exempt in the band


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
        "g1_clustered_ci_hi",
        "g1_brier_diff_ci_hi_standalone",
        "g1_brier_skill_score",
        "g2_star_pred_mean",
        "g2_star_true_mean",
        "g2_star_abs_diff",
        "g2_star_sigma",
        "g2_star_z",
        "g2_star_z_oracle",
        "g2_star_z_raw",
        "g3_bench_pred_mean",
        "g3_bench_true_mean",
        "g3_bench_abs_diff",
        "g3_bench_sigma",
        "g3_bench_z",
        "g3_bench_z_oracle",
        "g3_bench_z_raw",
        "g4_pit_ks",
        "g4_pit_ks_max",
        "g4_tail_pit_ks",
        "g4_iqr_pred",
        "g4_iqr_true",
        "g4_iqr_ratio",
        "g4_iqr_ratio_oracle",
        "central50_coverage",
        "central80_coverage",
        "g5_ece",
        "g5_ece_oracle",
        "g5_ece_null_bias",
        "g5_ece_debiased",
        "g5_ece_debiased_oracle",
        "g6_recent_corr",
        "g6_star_ratio",
        "g6_star_ci_hi",
        "g6_star_ref",
        "g6_citl_ratio",
        "g6_citl_ci_hi",
        "g6_over_ratio",
        "g6_over_ci_lo",
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


def test_gate1_uses_only_explicit_authentic_book_evidence():
    df = _priced_frame()
    authentic = np.arange(len(df)) % 2 == 0
    df["QuoteAuthenticity"] = np.where(authentic, "authentic", "synthetic")
    df.loc[~authentic, "P"] = 1.0 - (
        df.loc[~authentic, "Result"] >= df.loc[~authentic, "Line"]
    ).astype(float)

    full = gate_row(df, "EV", league="NBA", market="PTS", strategy="t")
    authentic_only = gate_row(
        df.loc[authentic].drop(columns="QuoteAuthenticity"),
        "EV",
        league="NBA",
        market="PTS",
        strategy="t",
    )

    assert full["g1_brier_diff_mean"] == pytest.approx(authentic_only["g1_brier_diff_mean"])
    assert full["g1_brier_diff_ci_lo"] == pytest.approx(authentic_only["g1_brier_diff_ci_lo"])
    assert full["g1_brier_diff_ci_hi"] == pytest.approx(authentic_only["g1_brier_diff_ci_hi"])
    assert full["g1_brier_skill_score"] == pytest.approx(authentic_only["g1_brier_skill_score"])


def test_retained_historical_passing_yards_gate1_reproduces():
    path = pkg_resources.files(data) / "test_sets/NFL_passing-yards.csv"
    with path.open("rb") as stream:
        assert (
            hashlib.file_digest(stream, "sha256").hexdigest()
            == "d9bc85adc753ea9aaa6fed183ada252b5c943a268366afc38673b56d6cea6b29"
        )
    frame = pd.read_csv(path, index_col=0)

    gates = compute_gates(frame, league="NFL", market="passing yards")

    assert gates["n_validation"] == 369
    assert gates["g1_brier_diff_mean"] == -0.0010
    assert gates["g1_brier_diff_ci_lo"] == -0.0051
    assert gates["g1_brier_diff_ci_hi"] == 0.0032
    assert gates["g1_clustered_ci_hi"] == 0.0033


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
        "g4_pit_ks": 0.03,
        "g4_pit_ks_max": 0.05,
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
        "g1": _clean_row(g1_brier_diff_ci_hi=0.01),  # beyond the 0.005 tie margin (mild-worse)
        "g2": _clean_row(g2_star_z=0.51),  # over the 0.5 cap
        "g3": _clean_row(g3_bench_z=0.51),
        "g4": _clean_row(g4_pit_ks=0.06),  # PIT-KS over the 0.05 threshold
        "g5": _clean_row(g5_ece=0.076),  # over the 0.075 cap
    }
    for gate, row in fails.items():
        out = apply_thresholds(row)
        assert not out[f"{gate}_pass"], f"{gate} should have failed"
        assert not out["ship"]


def test_apply_thresholds_g1_non_inferiority_margin():
    """G1 ships on the tie margin; ``g1_has_edge`` flags provable superiority separately."""
    # Clear win (ci_hi < 0): ships and has edge.
    win = apply_thresholds(_clean_row(g1_brier_diff_ci_hi=-0.03))
    assert win["g1_pass"] and win["g1_has_edge"]
    # Tight tie (0 <= ci_hi < margin): at least as good as the book ⇒ ships, but no edge.
    tie = apply_thresholds(_clean_row(g1_brier_diff_ci_hi=0.002))
    assert tie["g1_pass"] and not tie["g1_has_edge"] and tie["ship"]
    # Mild-worse beyond the margin: fails, no edge.
    worse = apply_thresholds(_clean_row(g1_brier_diff_ci_hi=0.01))
    assert not worse["g1_pass"] and not worse["g1_has_edge"] and not worse["ship"]


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


def test_apply_thresholds_g4_blank_pit_fails():
    """G4 blank (no per-row dist params ⇒ no g4_pit_ks) fails — no credit for absent evidence."""
    out = apply_thresholds(_clean_row(g4_pit_ks=None, g4_pit_ks_max=None))
    assert not out["g4_pass"]
    assert not out["ship"]


def test_apply_thresholds_g5_blank_fails():
    """G5 blank (no P or no Line) fails — not auto-pass; the cell couldn't compute calibration."""
    out = apply_thresholds(_clean_row(g5_ece=None))
    assert not out["g5_pass"]
    assert not out["ship"]


def test_strict_thresholds_are_pinned():
    """Lock the strict starter combo so an accidental tweak fails CI."""
    assert _GATE1_NONINF_MARGIN == 0.005
    assert _GATE1_CI_HI_MAX == 0.0
    assert _GATE2_STAR_Z_MAX == 0.5
    assert _GATE3_BENCH_Z_MAX == 0.5
    assert _GATE4_PIT_KS_DELTA == 0.05
    assert _GATE4_KS_NOISE_COEF == 1.358
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
            "P_PrePool": [0.56, 0.61, 0.51],
            "Odds": [0.45, 0.40, 0.50],
            "Line": [10.0, 12.0, 14.0],
        }
    )
    p = tmp_path / "NBA_PTS.csv"
    df.to_csv(p, index=False)
    loaded = load_test_set(p, "EV")
    assert {"P", "P_PrePool", "Odds", "Line"}.issubset(loaded.columns)
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


def _stamp_generic_identity(
    frame: pd.DataFrame,
    slug: str,
    controls: dict[str, str],
    *,
    league: str = "NBA",
    market: str = "PTS",
) -> pd.DataFrame:
    identity = build_artifact_identity(
        slug,
        league,
        market,
        controls,
        matrix_hash="e" * 64,
    )
    stamped = frame.copy()
    for column, value in artifact_identity_columns(identity.as_model_blob()).items():
        stamped[column] = value
    return stamped


def _encode_signed_skew_candidate(frame: pd.DataFrame, normalization: str) -> pd.DataFrame:
    encoded = frame.copy()
    if normalization == "ratio_meanyr":
        encoded["SN_Loc"] /= encoded["MeanYr"]
        encoded["SN_Scale"] /= encoded["MeanYr"]
    elif normalization == "centered_additive_mean10":
        encoded["Mean10"] = encoded["MeanYr"]
        encoded["SN_Loc"] -= encoded["Mean10"]
    else:
        raise ValueError(f"unsupported test normalization {normalization!r}")
    spec = get_strategy("SkewNormal")
    controls = next(
        corner for corner in strategy_controls(spec) if corner["normalization"] == normalization
    )
    return _stamp_generic_identity(encoded, spec.slug, controls)


def _supersede_pair(
    n: int = 800,
    seed: int = 42,
    candidate_calibration_noise: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (baseline, candidate) test-set frames sharing the same events.

    Set-up: outcomes are continuous ``Result ~ Normal(MeanYr, scale)`` (a SkewNormal with
    ``alpha=0``), so the frames carry the per-row distribution params real test-set CSVs do
    and Gate 4's randomized PIT can score predictive shape. The **line** sits at the
    ``1 − p_true`` quantile, making the book's flat over-prob 0.5 wrong by ``p_true − 0.5``.
    The **candidate** predicts the true distribution (calibrated: ``EV = MeanYr``,
    ``P = p_true``, params matching the draw) so it clears all five gates. The **baseline**
    perturbs ``EV`` (bets the wrong side sometimes) and mid-regresses ``P``.
    ``candidate_calibration_noise > 0`` swamps the candidate's ``P`` to invert the win.
    """
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    meanyr = rng.uniform(2, 30, n)
    scale = 4.0
    p_true = rng.uniform(0.20, 0.80, n)
    result = rng.normal(meanyr, scale)
    line = norm.ppf(1.0 - p_true, loc=meanyr, scale=scale)  # ⇒ P(Result > line) = p_true
    odds = np.full(n, 0.5)
    # Baseline bets the wrong side of the line on ~30% of rows (uniformly, not just the
    # low-edge ones) → its Kelly stakes lose there → lower Sharpe than the candidate, which
    # always sides with MeanYr. ``2·line − MeanYr`` reflects MeanYr across the line.
    flip = rng.random(n) < 0.30
    baseline_ev = np.where(flip, 2.0 * line - meanyr, meanyr) + rng.normal(0, 0.5, n)
    baseline_p = 0.5 + 0.4 * (p_true - 0.5)  # mid-regressed prob
    candidate_p = np.clip(p_true + rng.normal(0, candidate_calibration_noise, n), 1e-3, 1.0 - 1e-3)
    dist_cols = {"SN_Loc": meanyr, "SN_Scale": np.full(n, scale), "SN_Alpha": np.zeros(n)}
    event_cols = {
        "Player": [f"P{i % 80}" for i in range(n)],
        "Date": pd.date_range("2020-01-01", periods=n, freq="D"),
    }
    b_df = pd.DataFrame(
        {
            **event_cols,
            "MeanYr": meanyr,
            "Result": result,
            "EV": baseline_ev,
            "Line": line,
            "Odds": odds,
            "P": baseline_p,
            **dist_cols,
        }
    )
    c_df = pd.DataFrame(
        {
            **event_cols,
            "MeanYr": meanyr,
            "Result": result,
            "EV": meanyr.copy(),
            "Line": line,
            "Odds": odds,
            "P": candidate_p,
            **dist_cols,
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


def test_paired_brier_ci_returns_none_on_disjoint_events():
    b_df, c_df = _supersede_pair(n=50)
    c_df["Player"] = "different-" + c_df["Player"]
    assert _supersede_paired_brier_ci(b_df, c_df) is None


def test_supersede_pairing_ignores_row_order_and_range_index():
    b_df, c_df = _supersede_pair(n=500)
    expected_brier = _supersede_paired_brier_ci(b_df, c_df)
    expected_sharpe = _supersede_paired_sharpe(b_df, c_df, "EV")
    reordered = c_df.sample(frac=1.0, random_state=9).reset_index(drop=True)
    b_df.index = b_df.index + 5000

    actual_brier = _supersede_paired_brier_ci(b_df, reordered)
    actual_sharpe = _supersede_paired_sharpe(b_df, reordered, "EV")

    assert actual_brier == pytest.approx(expected_brier)
    assert actual_sharpe == pytest.approx(expected_sharpe)


def test_supersede_pairing_keeps_exact_shared_event_intersection():
    b_df, c_df = _supersede_pair(n=60)
    b_subset = b_df.iloc[:40].reset_index(drop=True)
    c_subset = c_df.iloc[20:].sample(frac=1.0, random_state=4).reset_index(drop=True)

    brier = _supersede_paired_brier_ci(b_subset, c_subset)
    sharpe = _supersede_paired_sharpe(b_subset, c_subset, "EV")

    assert brier is not None and brier[0] == 20
    assert sharpe is not None


@pytest.mark.parametrize("column", ["Player", "Date", "Line", "Result"])
def test_supersede_pairing_holds_when_event_identity_is_missing(column):
    b_df, c_df = _supersede_pair(n=50)
    c_df = c_df.drop(columns=[column])

    assert _supersede_paired_brier_ci(b_df, c_df) is None
    assert _supersede_paired_sharpe(b_df, c_df, "EV") is None


def test_supersede_pairing_does_not_pair_mismatched_outcomes():
    b_df, c_df = _supersede_pair(n=50)
    c_df["Result"] += 1.0

    assert _supersede_paired_brier_ci(b_df, c_df) is None
    assert _supersede_paired_sharpe(b_df, c_df, "EV") is None


def test_supersede_pairing_holds_on_ambiguous_duplicate_event_identity():
    b_df, c_df = _supersede_pair(n=50)
    b_df = pd.concat([b_df, b_df.iloc[[0]]], ignore_index=True)
    c_df = pd.concat([c_df, c_df.iloc[[0]]], ignore_index=True)

    assert _supersede_paired_brier_ci(b_df, c_df) is None
    assert _supersede_paired_sharpe(b_df, c_df, "EV") is None


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
    assert bets["Win Prob"].iloc[0] == pytest.approx(0.6)


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


def test_paired_sharpe_returns_none_on_disjoint_events():
    b_df, c_df = _supersede_pair(n=20)
    c_df["Date"] = c_df["Date"] + pd.Timedelta(days=1000)
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


@pytest.mark.parametrize(
    "normalization",
    ["ratio_meanyr", "centered_additive_mean10"],
)
def test_supersede_verdict_decodes_signed_candidate_normalization(normalization):
    baseline, candidate = _supersede_pair(n=1600, seed=31)
    signed_candidate = _encode_signed_skew_candidate(candidate, normalization)

    raw = apply_thresholds(
        gate_row(
            signed_candidate,
            "EV",
            league="NBA",
            market="PTS",
            strategy="candidate",
            decode_strategy=TARGET_NORM_NONE,
        )
    )
    verdict = supersede_verdict(
        baseline,
        signed_candidate,
        "EV",
        league="NBA",
        market="PTS",
        strategy="candidate",
    )

    assert raw["g4_pass"] is False
    assert verdict["s1_pass"] is True


def test_supersede_verdict_decodes_signed_centered_mixture():
    baseline, candidate = _supersede_pair(n=1600, seed=43)
    candidate = candidate.drop(columns=["SN_Loc", "SN_Scale", "SN_Alpha"])
    candidate["Mean10"] = candidate["MeanYr"]
    candidate["MIX_Loc1"] = 0.0
    candidate["MIX_Loc2"] = 0.0
    candidate["MIX_Scale1"] = 4.0
    candidate["MIX_Scale2"] = 4.0
    candidate["MIX_W1"] = 0.5
    spec = get_strategy("Mixture")
    controls = next(
        corner
        for corner in strategy_controls(spec)
        if corner["normalization"] == "centered_additive_mean10"
    )
    signed_candidate = _stamp_generic_identity(candidate, spec.slug, controls)

    raw = apply_thresholds(
        gate_row(
            signed_candidate,
            "EV",
            league="NBA",
            market="PTS",
            strategy="candidate",
            decode_strategy=TARGET_NORM_NONE,
        )
    )
    verdict = supersede_verdict(
        baseline,
        signed_candidate,
        "EV",
        league="NBA",
        market="PTS",
        strategy="candidate",
    )

    assert raw["g4_pass"] is False
    assert verdict["s1_pass"] is True


def test_supersede_verdict_legacy_candidate_uses_cell_stat_meta_decode(monkeypatch):
    import sportstradamus.training.scorecard as sc

    baseline, candidate = _supersede_pair(n=1600, seed=41)
    candidate["SN_Loc"] /= candidate["MeanYr"]
    candidate["SN_Scale"] /= candidate["MeanYr"]
    decode_calls = []

    def ratio_decode(league, market):
        decode_calls.append((league, market))
        return "ratio_meanyr"

    monkeypatch.setattr(sc, "_resolve_decode_strategy", ratio_decode)
    verdict = supersede_verdict(
        baseline,
        candidate,
        "EV",
        league="NBA",
        market="PTS",
        strategy="candidate",
    )

    assert decode_calls == [("NBA", "PTS")]
    assert verdict["s1_pass"] is True


def test_compute_gates_signed_identity_ignores_stale_stat_meta_decode(monkeypatch):
    import sportstradamus.training.scorecard as sc

    _, candidate = _supersede_pair(n=1600, seed=37)
    signed_candidate = _encode_signed_skew_candidate(candidate, "centered_additive_mean10")
    stat_meta_calls = []

    def stale_decode(league, market):
        stat_meta_calls.append((league, market))
        return "ratio_meanyr"

    monkeypatch.setattr(sc, "_resolve_decode_strategy", stale_decode)
    computed = compute_gates(
        signed_candidate,
        league="NBA",
        market="PTS",
        pred_col="EV",
    )
    explicit = apply_thresholds(
        gate_row(
            signed_candidate,
            "EV",
            league="NBA",
            market="PTS",
            strategy="meditate",
            decode_strategy="centered_additive_mean10",
        )
    )

    assert stat_meta_calls == []
    assert computed["g4_pit_ks"] == explicit["g4_pit_ks"]
    assert computed["g4_iqr_ratio"] == explicit["g4_iqr_ratio"]


def test_compute_gates_identity_absent_frame_uses_stat_meta_decode(monkeypatch):
    import sportstradamus.training.scorecard as sc

    _, candidate = _supersede_pair(n=800, seed=53)
    candidate["SN_Loc"] /= candidate["MeanYr"]
    candidate["SN_Scale"] /= candidate["MeanYr"]
    decode_calls = []

    def ratio_decode(league, market):
        decode_calls.append((league, market))
        return "ratio_meanyr"

    monkeypatch.setattr(sc, "_resolve_decode_strategy", ratio_decode)
    computed = compute_gates(candidate, league="NBA", market="PTS", pred_col="EV")

    assert decode_calls == [("NBA", "PTS")]
    assert computed["g4_pass"] is True


def test_diff_cli_preprint_passes_signed_decode_strategy(monkeypatch, tmp_path):
    import sportstradamus.training.scorecard as sc

    baseline, candidate = _supersede_pair(n=600, seed=47)
    baseline = _encode_signed_skew_candidate(baseline, "centered_additive_mean10")
    candidate = _encode_signed_skew_candidate(candidate, "centered_additive_mean10")
    baseline_path = tmp_path / "baseline.csv"
    candidate_path = tmp_path / "candidate.csv"
    baseline.to_csv(baseline_path, index=False)
    candidate.to_csv(candidate_path, index=False)
    decodes = []
    original_gate_row = sc.gate_row

    def capture_decode(*args, **kwargs):
        decodes.append(kwargs.get("decode_strategy"))
        return original_gate_row(*args, **kwargs)

    monkeypatch.setattr(sc, "gate_row", capture_decode)
    result = CliRunner().invoke(
        main,
        [
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(candidate_path),
            "--pred-col",
            "EV",
        ],
    )

    assert result.exit_code == 0, result.output
    assert decodes[:2] == ["centered_additive_mean10", "centered_additive_mean10"]


def test_signed_count_identity_uses_no_target_normalization():
    spec = get_strategy("DPO")
    frame = _stamp_generic_identity(pd.DataFrame(index=[0]), spec.slug, strategy_controls(spec)[0])

    assert _signed_decode_strategy(frame) == TARGET_NORM_NONE


# ---------------------------------------------------------------------------
# --live-window mode (Stage 0 deliverable 0.3)
# ---------------------------------------------------------------------------


def _build_live_offer(line, bet, model_p, books_p):
    """Return the offer-level columns for one flat history row (closing trio unfilled)."""
    return {
        "Line": line,
        "Boost": 1.0,
        "Platform": "Underdog",
        "Bet": bet,
        "Win Prob": model_p,
        "Market Prob": books_p,
        "Close Market Prob": float("nan"),
        "Market CLV": float("nan"),
        "Model CLV": float("nan"),
    }


def _build_live_history_fixture(n: int = 60, market: str = "PTS") -> pd.DataFrame:
    rng = np.random.default_rng(13)
    rows = []
    # Anchor to the real now: _history_to_eval_frame filters against datetime.now()'s
    # window, so a hardcoded date silently ages out of the window as the calendar advances.
    today = datetime.now()
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
                "Projection": line + rng.normal(0, 1.5),
                "Market Projection": line,
                "Dist": "SkewNormal",
                "CV": 0.3,
                "Model Param": line,
                "Gate": np.nan,
                "Temperature": 1.0,
                "Disp Cal": 1.0,
                "Step": "test",
                **_build_live_offer(line, bet, model_p, books_p),
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
    today = datetime.now()  # see _build_live_history_fixture: window filters against now()
    rows = []
    # In-scope: NBA + PTS within window
    for idx in range(5):
        rows.append(
            {
                "Player": f"A_{idx}",
                "League": "NBA",
                "Date": today.strftime("%Y-%m-%d"),
                "Market": "PTS",
                "Projection": 20.0,
                **_build_live_offer(20.0, "Over", 0.55, 0.50),
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
            "Projection": 20.0,
            **_build_live_offer(20.0, "Over", 0.55, 0.50),
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
            "Projection": 20.0,
            **_build_live_offer(20.0, "Over", 0.55, 0.50),
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
            "Projection": 20.0,
            **_build_live_offer(20.0, "Over", 0.55, 0.50),
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


def _skewnormal_frame(
    pred_scale: float, *, true_scale: float = 4.0, n: int = 8000, seed: int = 3
) -> pd.DataFrame:
    """SkewNormal (α=0 ⇒ Normal) frame whose predictive scale may differ from truth.

    Truth ~ Normal(0, ``true_scale``); the per-row predictive is Normal(0,
    ``pred_scale``). ``pred_scale == true_scale`` is calibrated; smaller is
    under-dispersed (intervals too narrow), larger is over-dispersed. No
    ``MeanYr`` column, so the decode leaves ``SN_Scale`` untouched.
    """
    rng = np.random.default_rng(seed)
    actual = rng.normal(0.0, true_scale, n)
    return pd.DataFrame(
        {
            "Result": actual,
            "EV": np.zeros(n),
            "SN_Loc": np.zeros(n),
            "SN_Scale": np.full(n, pred_scale),
            "SN_Alpha": np.zeros(n),
        }
    )


def test_pred_ppf_reconstructs_iqr_pred_analytical():
    """`_iqr_pred_analytical` must equal `_iqr` of the concatenated `_pred_ppf` bag.

    Guards the refactor that split the per-family quantile inversion out of the
    pooled-IQR helper — the two must stay numerically identical.
    """
    df = _skewnormal_frame(pred_scale=4.0)
    q25 = _pred_ppf(df, "SkewNormal", 0.25, strategy="baseline")
    q75 = _pred_ppf(df, "SkewNormal", 0.75, strategy="baseline")
    expected = np.percentile(np.concatenate([q25, q75]), 75) - np.percentile(
        np.concatenate([q25, q75]), 25
    )
    assert _iqr_pred_analytical(df, "SkewNormal", strategy="baseline") == pytest.approx(expected)


def test_dispersion_diagnostics_calibrated_is_near_nominal():
    """A correctly-scaled predictive covers at its nominal rate with a tiny PIT-KS."""
    df = _skewnormal_frame(pred_scale=4.0)
    pit_ks, tail_pit_ks, cov50, cov80 = _dispersion_diagnostics(
        df, "SkewNormal", df["Result"].to_numpy(), strategy="baseline"
    )
    assert cov50 == pytest.approx(0.50, abs=0.04)
    assert cov80 == pytest.approx(0.80, abs=0.04)
    assert pit_ks < 0.05
    assert tail_pit_ks <= pit_ks  # over-tail sub-supremum of the whole-CDF KS


def test_dispersion_diagnostics_flags_underdispersion():
    """Too-narrow predictive (half scale) → central coverage collapses, PIT-KS spikes.

    This is the NFL-receptions pathology: actuals fall *outside* the central
    interval far more than nominal, so coverage drops well below 0.50/0.80.
    """
    df = _skewnormal_frame(pred_scale=2.0)
    pit_ks, _tail, cov50, cov80 = _dispersion_diagnostics(
        df, "SkewNormal", df["Result"].to_numpy(), strategy="baseline"
    )
    assert cov50 < 0.35
    assert cov80 < 0.60
    assert pit_ks > 0.15


def test_dispersion_diagnostics_flags_overdispersion():
    """Too-wide predictive (double scale) → actuals cluster inside, coverage overshoots."""
    df = _skewnormal_frame(pred_scale=8.0)
    _, _, cov50, cov80 = _dispersion_diagnostics(
        df, "SkewNormal", df["Result"].to_numpy(), strategy="baseline"
    )
    assert cov50 > 0.65
    assert cov80 > 0.90


def test_dispersion_diagnostics_zinb_is_finite_and_bounded():
    """The count branch (ZINB mid-PIT + integer-quantile coverage) stays well-formed."""
    rng = np.random.default_rng(11)
    n = 3000
    df = pd.DataFrame(
        {
            "Result": rng.poisson(0.7, n).astype(float),
            "EV": np.full(n, 0.7),
            "R": np.full(n, 1.5),
            "NB_P": np.full(n, 0.4),
            "Gate": np.full(n, 0.3),
        }
    )
    assert _infer_dist_from_columns(df) == "ZINB"
    pit_ks, tail_pit_ks, cov50, cov80 = _dispersion_diagnostics(
        df, "ZINB", df["Result"].to_numpy(), strategy="baseline"
    )
    for v in (pit_ks, tail_pit_ks, cov50, cov80):
        assert np.isfinite(v) and 0.0 <= v <= 1.0
    assert tail_pit_ks <= pit_ks
    pit = _pred_midpit(df, "ZINB", df["Result"].to_numpy(), strategy="baseline")
    assert np.all((pit >= 0.0) & (pit <= 1.0))


def test_randomized_pit_ks_collapses_count_lattice():
    """On a calibrated low-count NegBin frame the mid-PIT KS is lattice-inflated while the
    randomized PIT KS is small — the fix that lets one Gate-4 threshold span count and
    continuous families (a continuous-tuned 0.05 would otherwise fail calibrated counts)."""
    from scipy.stats import nbinom

    rng = np.random.default_rng(7)
    r, nb_p = 1.5, 0.4  # scipy nbinom(r, 1 - nb_p); mean 1.0 — deep in the lattice regime
    y = nbinom.rvs(r, 1.0 - nb_p, size=6000, random_state=rng).astype(float)
    df = pd.DataFrame(
        {
            "Result": y,
            "EV": np.full_like(y, 1.0),
            "R": np.full_like(y, r),
            "NB_P": np.full_like(y, nb_p),
        }
    )
    assert _infer_dist_from_columns(df) == "NegBin"
    mid_ks = _ks_uniform(_pred_midpit(df, "NegBin", y, strategy="baseline"))
    rand_ks = _randomized_pit_ks(df, "NegBin", y, strategy="baseline")
    assert rand_ks < mid_ks - 0.02  # randomization removes the discreteness inflation
    assert rand_ks < 0.05  # calibrated cell clears the gate on the fixed statistic


def test_randomized_pit_ks_continuous_is_deterministic_ordinary_pit():
    """Continuous families have no point mass, so the randomized KS is the deterministic
    ``KS(F(y))`` — no draw dependence."""
    df = _skewnormal_frame(pred_scale=4.0)
    y = df["Result"].to_numpy()
    expected = _ks_uniform(_pred_midpit(df, "SkewNormal", y, strategy="baseline"))
    assert _randomized_pit_ks(df, "SkewNormal", y, strategy="baseline") == pytest.approx(expected)


def test_randomized_pit_applies_row_routed_cdf_maps():
    """QB/RB calibration maps are selected per row, not from the first CSV row."""
    import json

    from scipy.stats import norm

    from sportstradamus.training.structural_strategies import (
        AFFINE_STRATEGY,
    )

    y = np.array([-1.0, 0.0, 1.0, 2.0])
    identity = {"x": [0.0, 1.0], "y": [0.0, 1.0], "lam": 0.0}
    lower_half = {"x": [0.0, 0.5, 1.0], "y": [0.0, 0.25, 1.0], "lam": 1.0}
    frame = pd.DataFrame(
        {
            "SN_Loc": np.zeros(4),
            "SN_Scale": np.ones(4),
            "SN_Alpha": np.zeros(4),
            "MeanYr": np.ones(4),
            "P_PrePool": np.full(4, 0.5),
            "StructuralAdapterStrategy": AFFINE_STRATEGY,
            "StructuralRoute": ["QB", "RB", "QB", "RB"],
            "StructuralFallback": False,
            "PITRecalKnots": [
                json.dumps(identity),
                json.dumps(lower_half),
                json.dumps(identity),
                json.dumps(lower_half),
            ],
        }
    )
    spec = get_strategy(AFFINE_STRATEGY)
    legacy_payload = {
        "schema_version": spec.artifact_schema_version,
        "status": "active",
        "validation_audit": {"split_fingerprint_sha256": "c" * 64},
    }
    artifact_identity = build_artifact_identity(
        spec.slug,
        "NFL",
        "rushing yards",
        strategy_controls(spec)[0],
        legacy_payload,
        matrix_hash="d" * 64,
    )
    for column, value in artifact_identity_columns(artifact_identity.as_model_blob()).items():
        frame[column] = value

    draws = _randomized_pit_draws(frame, "SkewNormal", y, strategy="ratio_meanyr")
    raw = norm.cdf(y)
    expected = raw.copy()
    expected[[1, 3]] = np.interp(
        raw[[1, 3]],
        lower_half["x"],
        lower_half["y"],
    )

    assert len(draws) == 1
    np.testing.assert_allclose(draws[0], expected)

    mismatched_adapter = frame.assign(StructuralAdapterStrategy="none")
    with pytest.raises(ValueError, match="adapter CSV identity mismatch"):
        _randomized_pit_draws(
            mismatched_adapter,
            "SkewNormal",
            y,
            strategy="ratio_meanyr",
        )


def test_randomized_pit_ks_honors_skewnormal_zero_atom():
    """A gated SkewNormal is discrete at zero even though its positive head is continuous.

    This is the exact family served for high-zero NFL yardage cells.  Ignoring ``Gate``
    turns the zero-adjusted sample into a badly non-uniform ordinary PIT; randomizing the
    atom recovers calibration.
    """
    from scipy.stats import skewnorm

    rng = np.random.default_rng(451)
    n = 8000
    loc, scale, alpha, gate = 18.0, 10.0, 2.0, 0.2
    base = skewnorm.rvs(alpha, loc=loc, scale=scale, size=n, random_state=rng)
    y = np.where(rng.random(n) < gate, 0.0, base)
    frame = pd.DataFrame(
        {
            "SN_Loc": np.full(n, loc),
            "SN_Scale": np.full(n, scale),
            "SN_Alpha": np.full(n, alpha),
            "Gate": np.full(n, gate),
        }
    )

    gated_ks = _randomized_pit_ks(frame, "SkewNormal", y, strategy="none")
    ungated_ks = _randomized_pit_ks(frame.drop(columns="Gate"), "SkewNormal", y, strategy="none")

    assert gated_ks < 0.03
    assert ungated_ks > gated_ks + 0.08
    assert len(_randomized_pit_draws(frame, "SkewNormal", y, strategy="none")) == 25


def test_skewnormal_zero_atom_cdf_pmf_and_ppf_are_coherent():
    from scipy.stats import skewnorm

    frame = pd.DataFrame(
        {
            "SN_Loc": [10.0],
            "SN_Scale": [4.0],
            "SN_Alpha": [1.5],
            "Gate": [0.25],
        }
    )
    y = np.array([0.0])
    cdf, pmf = _pred_cdf_pmf(frame, "SkewNormal", y, strategy="none")
    base_zero = skewnorm.cdf(0.0, 1.5, loc=10.0, scale=4.0)
    assert cdf[0] == pytest.approx(0.25 + 0.75 * base_zero)
    assert pmf[0] == pytest.approx(0.25)

    # q=0.10 lies in the zero jump for these parameters; an upper quantile must
    # invert the rescaled positive-component CDF.
    assert _pred_ppf(frame, "SkewNormal", 0.10, strategy="none")[0] == 0.0
    q = 0.80
    expected = skewnorm.ppf((q - 0.25) / 0.75, 1.5, loc=10.0, scale=4.0)
    assert _pred_ppf(frame, "SkewNormal", q, strategy="none")[0] == pytest.approx(expected)


def test_gate4_pit_ks_threshold_takes_larger_of_delta_and_noise_floor():
    """Per-cell bound = max(δ=0.05, 1.358/√n): δ binds at large n, the KS noise floor at small."""
    assert _gate4_pit_ks_threshold(2000) == pytest.approx(_GATE4_PIT_KS_DELTA)  # floor ~0.030 < δ
    assert _gate4_pit_ks_threshold(100) == pytest.approx(_GATE4_KS_NOISE_COEF / 10.0)  # 0.1358 > δ
    assert np.isnan(_gate4_pit_ks_threshold(0))


def test_tail_ks_localizes_over_tail_miscalibration():
    """The reported over-tail KS is a sub-supremum of the whole-CDF KS (so always ≤ it):
    it localizes *where* the deviation lives. A self-correcting bulk bump leaves the
    over-tail clean (tail-KS ≈ 0 ≪ global); a deficit confined to the over-tail puts the
    global supremum in the tail (tail-KS ≈ global) — the alt-OVER mispricing that the
    whole-CDF gate nets away."""
    bulk = np.linspace(0.0, 1.0, 1000, endpoint=False)
    bulk[(bulk >= 0.4) & (bulk < 0.6)] = 0.4  # pile the middle fifth; CDF rejoins uniform by u=0.6
    assert _ks_uniform(bulk) > 0.15
    assert _tail_ks_uniform(bulk) < 0.01  # over-tail is undisturbed

    tail = np.linspace(0.0, 0.88, 1000, endpoint=False)  # no PIT mass above 0.88
    assert _tail_ks_uniform(tail) == pytest.approx(_ks_uniform(tail))  # global sup is in the tail
    assert _tail_ks_uniform(tail) > 0.1


def test_standalone_g1_column_scores_preblend_probs():
    """`g1_brier_diff_ci_hi_standalone` scores `P_standalone` on the fused rows.

    Blank when the column is absent; present and distinct when supplied. With a
    coin-flip standalone model and the frame's random (0.5) book, the standalone
    paired-Brier ties the book (upper bound 0), while the sharp fused model beats
    it (upper bound < 0) — so the standalone bound sits strictly above the fused.
    """
    df = _priced_frame()
    base = gate_row(df, "EV", league="NBA", market="PTS", strategy="t")
    assert base["g1_brier_diff_ci_hi_standalone"] is None

    df_sa = df.copy()
    df_sa["P_standalone"] = 0.5
    row = gate_row(df_sa, "EV", league="NBA", market="PTS", strategy="t")
    assert row["g1_brier_diff_ci_hi_standalone"] is not None
    assert row["g1_brier_diff_ci_hi_standalone"] > row["g1_brier_diff_ci_hi"]


def _sn_pit_ks(mean, sigma, skew, y, c, s):
    from sportstradamus.helpers.distributions import skewnormal_loc_from_mean
    from sportstradamus.training.scorecard import TARGET_NORM_NONE, _randomized_pit_ks

    scale = sigma * c
    alpha = skew + s
    df = pd.DataFrame(
        {
            "SN_Loc": skewnormal_loc_from_mean(mean, scale, alpha),
            "SN_Scale": scale,
            "SN_Alpha": alpha,
        }
    )
    return _randomized_pit_ks(df, "SkewNormal", y, strategy=TARGET_NORM_NONE)


def test_decode_sn_loc_scale_centered_additive_readds_location_offset():
    """centered_additive_mean10 PIT decode must re-add the Mean10 baseline to loc.

    Regression for the Gate-4 decode bug: the scorecard mirror returned the raw
    SN_Loc for centered strategies (correct for the location-free IQR, wrong for
    the location-sensitive PIT), mis-locating the predictive by the ~Mean10 offset
    and inflating pit_ks. Scale is still passed through unchanged for this strategy.
    """
    raw_loc = np.array([0.1, -0.2, 0.0])
    mean10 = np.array([2.4, 3.0, 1.8])  # all above the MeanYr floor; clip is a no-op
    df = pd.DataFrame(
        {
            "SN_Loc": raw_loc,
            "SN_Scale": np.array([1.5, 2.0, 1.0]),
            "Mean10": mean10,
            "MeanYr": np.array([2.5, 3.1, 1.9]),
        }
    )
    loc, scale = _decode_sn_loc_scale(df, "centered_additive_mean10")
    np.testing.assert_allclose(loc, mean10 + raw_loc)
    np.testing.assert_allclose(scale, df["SN_Scale"].to_numpy())


def test_decode_sn_loc_scale_uses_persisted_denom_col():
    """Gate-4 decode must use the cell's persisted DenomCol, not a hardcoded MeanYr.

    Regression for the Gate-4 denom bug: a zero-inflated SkewNormal cell encodes (and
    serves) against MeanYr_nonzero, but the decode hardcoded MeanYr, mis-scaling its
    dispersion by MeanYr/MeanYr_nonzero into spurious under-dispersion and a false g4
    fail. The pipeline now persists DenomCol; absent it (legacy CSVs) the decode falls
    back to MeanYr, its original scoring.
    """
    raw_loc = np.array([0.4, 0.5, 0.6])
    raw_scale = np.array([0.10, 0.12, 0.14])
    meanyr = np.array([2.0, 3.0, 4.0])
    meanyr_nonzero = np.array([3.0, 4.5, 6.0])  # > MeanYr, as on a zero-inflated cell
    base = {
        "SN_Loc": raw_loc,
        "SN_Scale": raw_scale,
        "MeanYr": meanyr,
        "MeanYr_nonzero": meanyr_nonzero,
    }

    loc_nz, scale_nz = _decode_sn_loc_scale(
        pd.DataFrame({**base, "DenomCol": "MeanYr_nonzero"}), "ratio_meanyr"
    )
    np.testing.assert_allclose(scale_nz, raw_scale * meanyr_nonzero)
    np.testing.assert_allclose(loc_nz, raw_loc * meanyr_nonzero)

    loc_legacy, scale_legacy = _decode_sn_loc_scale(pd.DataFrame(base), "ratio_meanyr")
    np.testing.assert_allclose(scale_legacy, raw_scale * meanyr)
    np.testing.assert_allclose(loc_legacy, raw_loc * meanyr)


def test_fit_skewnorm_dispersion_c_widens_underdispersed_symmetric():
    """A predictive half as wide as the truth must fit c ~ 2 to recover calibration.

    Symmetric case (skew=0): delta=0, so loc == mean and scaling the scale never
    moves the centre. y ~ Normal(mu, s); we hand the fitter the SAME mean but
    sigma = s/2 (under-dispersed), and the Gate-4-optimal c that makes the PIT
    Uniform is the one that restores the true scale, i.e. c ~ 2.
    """
    rng = np.random.default_rng(0)
    mu, s_true, n = 7.0, 3.0, 4000
    y = rng.normal(mu, s_true, n)
    c = fit_skewnorm_dispersion_c(np.full(n, mu), np.full(n, s_true / 2.0), np.zeros(n), y)
    assert abs(c - 2.0) < 0.25


def test_fit_skewnorm_dispersion_skew_recovers_underskew():
    """Lever 4a: a served predictive that collapsed to symmetry (``alpha = 0``) on
    right-skewed truth cannot be calibrated by scale alone — the joint ``(c, s)`` fit
    must inject a positive additive skew shift that materially beats the scale-only KS.
    """
    from scipy import stats as st

    rng = np.random.default_rng(0)
    a_true, n = 4.0, 4000
    y = st.skewnorm.rvs(a_true, size=n, random_state=rng)
    mean = np.full(n, float(y.mean()))
    sigma = np.full(n, float(y.std()))
    skew0 = np.zeros(n)

    c_only = fit_skewnorm_dispersion_c(mean, sigma, skew0, y)
    c, s = fit_skewnorm_dispersion_skew(mean, sigma, skew0, y)

    assert s > 1.0
    assert -3.0 <= s <= 3.0
    assert (
        _sn_pit_ks(mean, sigma, skew0, y, c, s)
        < _sn_pit_ks(mean, sigma, skew0, y, c_only, 0.0) - 0.02
    )


def test_fit_skewnorm_dispersion_skew_noop_on_calibrated():
    """No-op on an already-calibrated symmetric cell: the skew knob must not manufacture
    spurious asymmetry, so it returns ``c ~ 1``, ``s ~ 0`` and does not worsen the PIT.
    """
    rng = np.random.default_rng(1)
    mu, sd, n = 5.0, 2.0, 4000
    y = rng.normal(mu, sd, n)
    mean, sigma, skew0 = np.full(n, mu), np.full(n, sd), np.zeros(n)

    c, s = fit_skewnorm_dispersion_skew(mean, sigma, skew0, y)

    assert s == 0.0  # opt-in: a sub-margin gain falls back to the pure Lever-1 (c, 0)
    assert abs(c - 1.0) < 0.15
    assert _sn_pit_ks(mean, sigma, skew0, y, c, s) < 0.04


def test_fit_skewnorm_dispersion_skew_is_bit_reproducible():
    """The joint ``(c, s)`` fit must be bit-identical run-to-run — deterministic Nelder-Mead
    from fixed starts over the seeded randomized PIT. The dumped ``SN_Alpha`` carries ``s``,
    so the determinism gate's SkewNormal bit-identity depends on this fit being reproducible.
    """
    from scipy import stats as st

    rng = np.random.default_rng(2)
    y = st.skewnorm.rvs(3.0, size=3000, random_state=rng)
    mean, sigma, skew0 = (
        np.full(3000, float(y.mean())),
        np.full(3000, float(y.std())),
        np.zeros(3000),
    )

    assert fit_skewnorm_dispersion_skew(mean, sigma, skew0, y) == fit_skewnorm_dispersion_skew(
        mean, sigma, skew0, y
    )


def test_fit_skewnorm_dispersion_skew_sequential_freezes_scale_then_fits_skew():
    """Sequential variant (``joint=False``): fit the Lever-1 scale first, freeze it, then fit the
    additive skew on top by a 1-D search. With the served predictive carrying a *wrong-direction*
    base skew that scale cannot fix, the returned ``c`` is exactly the scale-only optimum (the joint
    fit may instead move ``c``) and the additive shift still materially beats scale-only.
    """
    from scipy import stats as st

    rng = np.random.default_rng(7)
    y = st.skewnorm.rvs(4.0, size=4000, random_state=rng)
    mean = np.full(4000, float(y.mean()))
    sigma = np.full(4000, float(y.std()))
    base = np.full(4000, -3.0)  # model fit a negative skew on right-skewed truth

    c_only = fit_skewnorm_dispersion_c(mean, sigma, base, y)
    c, s = fit_skewnorm_dispersion_skew(mean, sigma, base, y, joint=False)

    assert c == c_only  # scale frozen at the Lever-1 optimum, not re-optimized with s
    assert s > 0.0
    assert (
        _sn_pit_ks(mean, sigma, base, y, c, s)
        < _sn_pit_ks(mean, sigma, base, y, c_only, 0.0) - 0.01
    )


def test_fit_skewnorm_dispersion_skew_sequential_noop_on_calibrated():
    """The sequential variant must also reject a sub-margin skew on an already-calibrated cell,
    falling back to the pure Lever-1 ``(c, 0)`` like the joint fit does.
    """
    rng = np.random.default_rng(1)
    mu, sd, n = 5.0, 2.0, 4000
    y = rng.normal(mu, sd, n)
    mean, sigma, skew0 = np.full(n, mu), np.full(n, sd), np.zeros(n)

    c, s = fit_skewnorm_dispersion_skew(mean, sigma, skew0, y, joint=False)

    assert s == 0.0
    assert abs(c - 1.0) < 0.15


def test_joint_skew_fit_dominates_sequential_on_alpha_collapse():
    """Pins the coupling finding behind keeping the joint fit as the production default: on an
    ``alpha≈0`` collapsed predictive over right-skewed truth, the scale-only optimum sits at a ``c``
    where the skew gradient vanishes, so sequential (freeze that ``c``, then fit ``s``) is a no-op,
    while the joint fit co-moves ``c`` upward and injects the skew the cell needs. Joint must clear
    Gate-4 KS strictly below what sequential can reach here.
    """
    from scipy import stats as st

    rng = np.random.default_rng(0)
    y = st.skewnorm.rvs(4.0, size=4000, random_state=rng)
    mean = np.full(4000, float(y.mean()))
    sigma = np.full(4000, float(y.std()))
    skew0 = np.zeros(4000)

    c_seq, s_seq = fit_skewnorm_dispersion_skew(mean, sigma, skew0, y, joint=False)
    c_j, s_j = fit_skewnorm_dispersion_skew(mean, sigma, skew0, y, joint=True)

    assert s_seq == 0.0  # frozen at a skew-dead scale → sequential cannot help
    assert s_j > 1.0
    assert _sn_pit_ks(mean, sigma, skew0, y, c_j, s_j) < _sn_pit_ks(
        mean, sigma, skew0, y, c_seq, s_seq
    )


def test_min_gate_slack_is_the_binding_gate_headroom_when_all_pass():
    """The ship-margin scalar is the minimum per-gate headroom, each normalized to its own
    threshold; when every gate passes it is the tightest gate's fractional headroom (here Gate 4).
    """
    row = {
        "g1_brier_diff_ci_hi": 0.0,  # (0.005-0)/0.005 = 1.0
        "g2_star_z": 0.25,  # (0.5-0.25)/0.5 = 0.5
        "g3_bench_z": 0.0,  # 1.0
        "g4_pit_ks": 0.04,
        "g4_pit_ks_max": 0.05,  # (0.05-0.04)/0.05 = 0.2  <- binding
        "g5_ece_debiased": 0.05,  # (0.075-0.05)/0.075 = 0.333
    }
    assert min_gate_slack(row) == pytest.approx(0.2)


def test_min_gate_slack_negative_when_gate4_fails():
    """A failing gate drives the scalar below zero — the search must rank a non-shipping combo
    under a shipping one.
    """
    row = {
        "g1_brier_diff_ci_hi": 0.0,
        "g2_star_z": 0.25,
        "g3_bench_z": 0.0,
        "g4_pit_ks": 0.06,  # (0.05-0.06)/0.05 = -0.2
        "g4_pit_ks_max": 0.05,
        "g5_ece_debiased": 0.05,
    }
    assert min_gate_slack(row) < 0


def test_min_gate_slack_g1_blank_does_not_bind():
    """A blank Gate 1 (no book) auto-passes, so it must not bind the minimum — the scalar stays
    positive and equals the tightest *computed* gate (Gate 4 here), not a -inf from the blank.
    """
    row = {
        "g1_brier_diff_ci_hi": None,  # no Odds → auto-pass, non-binding
        "g2_star_z": 0.0,
        "g3_bench_z": 0.0,
        "g4_pit_ks": 0.04,
        "g4_pit_ks_max": 0.05,
        "g5_ece_debiased": 0.03,
    }
    assert min_gate_slack(row) == pytest.approx(0.2)


def test_eb_gate_decode_reconstructs_served_loc_with_dumped_global_mean():
    """``centered_additive_eb_meanyr_k10`` gate decode round-trips through the dumped GlobalMean.

    SN_Loc dumps as ``served_loc − EB_prior(MeanYr, GamesPlayed, global_mean, K)``; the gate
    must add the EB prior back using the *dumped* ``GlobalMean`` (not the hardcoded 0.0 the two
    older normalizations ignore) to recover ``served_loc`` byte-for-byte. Mirrors the
    prediction-side EB decode pinned in ``tests/test_model_prob_skewnormal_decode.py``, on the
    scorecard side — without the GlobalMean read the EB prior shrinks toward 0 and the decode
    silently corrupts every PIT/IQR the gate scores.
    """
    from sportstradamus.training.baselines import get_target_normalization

    rng = np.random.default_rng(17)
    n = 16
    global_mean = 8.4
    served_loc = rng.uniform(3.0, 22.0, n)
    served_scale = rng.uniform(1.0, 5.0, n)
    skew = rng.uniform(-1.0, 2.0, n)
    meanyr = rng.uniform(2.0, 25.0, n)
    games = rng.integers(1, 80, n).astype(float)

    strat = get_target_normalization("centered_additive_eb_meanyr_k10")
    X = pd.DataFrame({"MeanYr": meanyr, "GamesPlayed": games})
    dumped = pd.DataFrame(
        {
            "SN_Loc": strat.encode_loc(served_loc, X, global_mean, "MeanYr"),
            "SN_Scale": strat.encode_scale(served_scale, X, "MeanYr"),
            "SN_Alpha": skew,
            "MeanYr": meanyr,
            "GamesPlayed": games,
            "GlobalMean": global_mean,
        }
    )

    loc_g, scale_g = _decode_sn_loc_scale(dumped, "centered_additive_eb_meanyr_k10")
    np.testing.assert_allclose(loc_g, served_loc, atol=1e-9)
    np.testing.assert_allclose(scale_g, served_scale, atol=1e-9)
    # The EB prior was actually added back — decode is not the raw (encoded) loc.
    assert not np.allclose(loc_g, dumped["SN_Loc"].to_numpy())


def test_eb_gate_decode_zero_global_mean_fallback_differs_from_real_prior():
    """Guard the silent-corruption failure mode: decoding the same SN_Loc with the legacy
    GlobalMean-absent fallback (0.0) must NOT match the real-global-mean decode — proving the
    gate genuinely consumes the dumped GlobalMean rather than ignoring it like the older norms.
    """
    from sportstradamus.training.baselines import get_target_normalization

    rng = np.random.default_rng(23)
    n = 12
    global_mean = 11.0
    served_loc = rng.uniform(3.0, 20.0, n)
    meanyr = rng.uniform(2.0, 25.0, n)
    games = rng.integers(1, 60, n).astype(float)

    strat = get_target_normalization("centered_additive_eb_meanyr_k10")
    X = pd.DataFrame({"MeanYr": meanyr, "GamesPlayed": games})
    sn_loc = strat.encode_loc(served_loc, X, global_mean, "MeanYr")
    base = {"SN_Scale": np.ones(n), "SN_Alpha": np.zeros(n), "MeanYr": meanyr, "GamesPlayed": games}

    with_gm = pd.DataFrame({"SN_Loc": sn_loc, **base, "GlobalMean": global_mean})
    without_gm = pd.DataFrame({"SN_Loc": sn_loc, **base})

    loc_real, _ = _decode_sn_loc_scale(with_gm, "centered_additive_eb_meanyr_k10")
    loc_fallback, _ = _decode_sn_loc_scale(without_gm, "centered_additive_eb_meanyr_k10")

    np.testing.assert_allclose(loc_real, served_loc, atol=1e-9)
    assert not np.allclose(loc_real, loc_fallback)


def test_load_test_set_retains_eb_decode_columns(tmp_path):
    """``load_test_set`` curates feature columns, but the EB-prior decode needs ``GamesPlayed``
    and ``GlobalMean`` at gate time (mirroring how ``Mean10`` is retained for the mean10 decode).
    Without them the combination-search EB scoring raises ``KeyError`` instead of decoding.
    """
    df = pd.DataFrame(
        {
            "MeanYr": [5.0, 6.0, 7.0, 8.0],
            "Result": [4.0, 7.0, 6.0, 9.0],
            "Blended_EV": [5.1, 6.2, 6.8, 8.3],
            "SN_Loc": [0.1, 0.2, 0.3, 0.4],
            "SN_Scale": [1.0, 1.0, 1.0, 1.0],
            "SN_Alpha": [0.0, 0.0, 0.0, 0.0],
            "GamesPlayed": [10.0, 20.0, 30.0, 40.0],
            "GlobalMean": [6.5, 6.5, 6.5, 6.5],
            "some_feature": [1.0, 2.0, 3.0, 4.0],
        }
    )
    path = tmp_path / "WNBA_AST.csv"
    df.to_csv(path, index=False)

    out = load_test_set(path, "Blended_EV")

    assert "GamesPlayed" in out.columns
    assert "GlobalMean" in out.columns
    # Curation still drops non-decode feature columns.
    assert "some_feature" not in out.columns


def test_load_test_set_retains_nonzero_ratio_decode_contract(tmp_path):
    """The curated official scoring frame must retain the denominator named by DenomCol."""
    df = pd.DataFrame(
        {
            "MeanYr": [4.0, 5.0],
            "MeanYr_nonzero": [8.0, 10.0],
            "Result": [0.0, 9.0],
            "Blended_EV": [7.0, 9.0],
            "SN_Loc": [0.5, 0.6],
            "SN_Scale": [0.2, 0.3],
            "SN_Alpha": [1.0, 1.5],
            "DenomCol": ["MeanYr_nonzero", "MeanYr_nonzero"],
            "Gate": [0.2, 0.2],
        }
    )
    path = tmp_path / "NFL_receiving-yards.csv"
    df.to_csv(path, index=False)

    loaded = load_test_set(path, "Blended_EV")
    assert {"DenomCol", "MeanYr_nonzero"}.issubset(loaded.columns)
    loc, scale = _decode_sn_loc_scale(loaded, "ratio_meanyr")
    np.testing.assert_allclose(loc, df["SN_Loc"] * df["MeanYr_nonzero"])
    np.testing.assert_allclose(scale, df["SN_Scale"] * df["MeanYr_nonzero"])


def test_load_test_set_rejects_missing_or_variable_persisted_denom(tmp_path):
    base = pd.DataFrame(
        {
            "MeanYr": [4.0, 5.0],
            "Result": [3.0, 6.0],
            "Blended_EV": [4.0, 5.0],
            "SN_Loc": [0.5, 0.6],
            "SN_Scale": [0.2, 0.3],
            "SN_Alpha": [1.0, 1.5],
        }
    )
    missing = base.assign(DenomCol="Player projected carries mean")
    missing_path = tmp_path / "missing.csv"
    missing.to_csv(missing_path, index=False)
    with pytest.raises(ValueError, match="references missing column"):
        load_test_set(missing_path, "Blended_EV")

    variable = base.assign(DenomCol=["MeanYr", "MeanYr_nonzero"], MeanYr_nonzero=[8.0, 9.0])
    variable_path = tmp_path / "variable.csv"
    variable.to_csv(variable_path, index=False)
    with pytest.raises(ValueError, match="exactly one DenomCol"):
        load_test_set(variable_path, "Blended_EV")


# ---------------------------------------------------------------------------
# §6.1 Rung C — Gate 4 applies the whole-CDF map ``g`` to the PIT before the KS.
# The map is persisted as a constant ``PITRecalKnots`` JSON column on the scored
# test CSV (mirroring ``DenomCol``/``GlobalMean``); a cell without the column
# behaves byte-identically to a pre-Rung-C cell.
# ---------------------------------------------------------------------------

_RECAL_LOC, _RECAL_SCALE, _RECAL_ALPHA = 20.0, 5.0, 3.0


def _under_dispersed_skewnorm(n: int = 3000):
    # Served predictive is SkewNormal(loc, scale, alpha); outcomes are drawn 1.7x wider, so
    # the served PIT piles into the tails (the under-dispersion signature Gate 4 fails on).
    from scipy.stats import skewnorm

    rng = np.random.default_rng(0)
    y = skewnorm.rvs(
        _RECAL_ALPHA, loc=_RECAL_LOC, scale=_RECAL_SCALE * 1.7, size=n, random_state=rng
    )
    df = pd.DataFrame(
        {"SN_Loc": _RECAL_LOC, "SN_Scale": _RECAL_SCALE, "SN_Alpha": _RECAL_ALPHA}, index=range(n)
    )
    return df, y


def test_gate_g4_reflects_warped_pit():
    df, y = _under_dispersed_skewnorm()
    raw_ks = _randomized_pit_ks(df, "SkewNormal", y, strategy=TARGET_NORM_NONE)
    # g fit on this cell's own served PIT uniformizes it, so the warped Gate-4 KS must drop.
    served_pit = _randomized_pit_draws(df, "SkewNormal", y, strategy=TARGET_NORM_NONE)[0]
    blob = posthoc.fit_isotonic_pit(served_pit)
    warped = df.assign(PITRecalKnots=json.dumps(blob))
    warped_ks = _randomized_pit_ks(warped, "SkewNormal", y, strategy=TARGET_NORM_NONE)
    assert warped_ks < raw_ks
    assert warped_ks < 0.05


def test_gate_g4_identity_without_knots_column():
    # No PITRecalKnots column => no warp => the exact pre-Rung-C statistic. A lambda=0
    # (identity) blob must reproduce that same number, pinning the no-op both ways.
    df, y = _under_dispersed_skewnorm()
    bare = _randomized_pit_ks(df, "SkewNormal", y, strategy=TARGET_NORM_NONE)
    served_pit = _randomized_pit_draws(df, "SkewNormal", y, strategy=TARGET_NORM_NONE)[0]
    identity = df.assign(PITRecalKnots=json.dumps(posthoc.fit_isotonic_pit(served_pit, lam=0.0)))
    identity_ks = _randomized_pit_ks(identity, "SkewNormal", y, strategy=TARGET_NORM_NONE)
    assert identity_ks == bare


def test_load_test_set_preserves_pit_recal_knots(tmp_path):
    # report() scores a dumped CSV through load_test_set, which keeps only an allowlist of
    # columns. The constant PITRecalKnots map must survive that load, or the official Gate 4
    # silently scores the un-recalibrated cell. Both the deterministic A/B (pd.read_csv) and the
    # assign()-based tests above bypass load_test_set, so only this end-to-end load path catches it.
    rng = np.random.default_rng(1)
    n = 200
    blob = posthoc.fit_isotonic_pit(rng.beta(0.5, 0.5, size=3000))
    df = pd.DataFrame(
        {
            DECILE_COL: rng.uniform(10.0, 30.0, n),
            ACTUAL_COL: rng.uniform(0.0, 40.0, n),
            DEFAULT_PRED_COL: rng.uniform(10.0, 30.0, n),
            "SN_Loc": _RECAL_LOC,
            "SN_Scale": _RECAL_SCALE,
            "SN_Alpha": _RECAL_ALPHA,
            "PITRecalKnots": json.dumps(blob),
        }
    )
    path = tmp_path / "WNBA_PA.csv"
    df.to_csv(path, index=False)
    loaded = load_test_set(path, DEFAULT_PRED_COL)
    probe = np.linspace(0.05, 0.95, n)
    np.testing.assert_array_equal(
        _apply_pit_recal_by_row(loaded, probe),
        posthoc.apply_cdf_recal(blob, probe),
    )


# ---------------------------------------------------------------------------
# Double Poisson ("DPO") — DP_MU / DP_PHI param columns (the count analogue of
# NegBin's R / NB_P), gate-free family. Pins the column inference + load path,
# Gate-4 randomized-PIT calibration, the Gate-6 count-cohort over-leg, and an
# end-to-end gate_row. The kernel itself is pinned in tests/test_double_poisson.py.
# ---------------------------------------------------------------------------

# Low-mean / near-unit-precision band — the lattice regime DPO cells live in
# (NBA PF, NHL points per the WS-3 Phase-0 screen).
_DPO_MU_RANGE = (0.5, 4.0)
_DPO_PHI_RANGE = (0.6, 1.8)


def _dpo_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Calibrated DPO frame: ``Result`` drawn from each row's own ``(mu, phi)``
    by inverse-CDF sampling, with a calibrated over-probability ``P`` and a
    coin-flip book so the priced gates all compute.
    """
    rng = np.random.default_rng(seed)
    mu = rng.uniform(*_DPO_MU_RANGE, n)
    phi = rng.uniform(*_DPO_PHI_RANGE, n)
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
    """Mirror of ``_count_legs_frame`` on DP params: the stable BENCH (low MeanYr)
    is over-predicted 1.3x vs a realized mean of 2.0 (above the
    ``_GATE6_OVER_MIN_MEAN`` guard), so the over-leg must fire iff DPO is in the
    Gate-6 count cohort.
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
    from sportstradamus.training.scorecard import _gate6_legs

    g6 = _gate6_legs(_dpo_bench_over_frame(), "Blended_EV", league="WNBA", prior_g6_fired=None)
    assert g6["g6_over_ci_lo"] is not None
    assert g6["g6_over_ci_lo"] > 1.0 + _GATE6_MARGIN


# ---------------------------------------------------------------------------
# 2-component Gaussian mixture — MIX_Loc1/MIX_Loc2/MIX_Scale1/MIX_Scale2/MIX_W1
# in the cell's NORMALIZED space plus the same constant DenomCol/GlobalMean
# columns SkewNormal uses, so both locs and both scales must decode through the
# target-normalization registry before evaluating.
# ---------------------------------------------------------------------------

# All frames encode against ratio_meanyr: decoded loc/scale = raw * MeanYr
# (MeanYr kept well above the 0.5 decode floor so the multiply is exact).
_MIX_STRATEGY = "ratio_meanyr"
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
    median = _pred_ppf(df, "Mixture", 0.5, strategy=_MIX_STRATEGY)
    df["Line"] = np.round(median) + 0.5
    cdf_at_line, _ = _pred_cdf_pmf(df, "Mixture", df["Line"].to_numpy(), strategy=_MIX_STRATEGY)
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


def test_mixture_pit_uniform_pins_decode_and_cdf_composition():
    """Result is drawn from each row's own decoded mixture, so its PIT must be
    ~Uniform(0, 1); a decode that dropped the MeanYr denominator or scored only
    one component would blow the KS well past this bound (KS at n=200: E ≈ 0.06,
    α=0.05 critical ≈ 0.096). Zero point mass routes the randomized PIT down the
    single-deterministic-draw (continuous) branch.
    """
    df = _mix_frame()
    pit, pmf = _pred_cdf_pmf(df, "Mixture", df["Result"].to_numpy(), strategy=_MIX_STRATEGY)
    np.testing.assert_array_equal(pmf, np.zeros(len(df)))
    assert _ks_uniform(pit) < 0.08


def test_mixture_degenerate_w1_reproduces_plain_normal_cdf():
    """w1=1 collapses the mixture to component 1: the CDF must equal the plain
    normal at the decoded loc/scale, with the dead component's params inert."""
    from scipy.stats import norm

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
    cdf, _ = _pred_cdf_pmf(df, "Mixture", y, strategy=_MIX_STRATEGY)
    np.testing.assert_allclose(cdf, norm.cdf(y, loc=1.1 * meanyr, scale=0.2 * meanyr), atol=1e-12)


def test_gate_row_end_to_end_mixture():
    """gate_row auto-detects MIX columns: analytical G4 + coverage populate and
    the calibrated frame clears the shape/bias gates through apply_thresholds."""
    row = apply_thresholds(
        gate_row(_mix_frame(600, seed=2), "EV", league="NBA", market="PTS", strategy=_MIX_STRATEGY)
    )
    assert row["g4_pit_ks"] is not None
    assert row["g4_iqr_pred"] > 0.0
    assert row["central50_coverage"] is not None
    assert row["central80_coverage"] is not None
    assert row["g2_pass"] and row["g3_pass"] and row["g4_pass"]
    assert isinstance(row["ship"], bool)


# ---------------------------------------------------------------------------
# Official-scorecard parity guards for the NFL receiving-yards v3 two-part head.
# ---------------------------------------------------------------------------

_MATRIX_HASH = "a" * 64
_SPLIT_FINGERPRINT = "b" * 64


def _strategy_identity_columns() -> dict[str, object]:
    spec = get_strategy(receiving.CANDIDATE_NAME)
    legacy_payload = {
        "schema_version": receiving.SCHEMA_VERSION,
        "status": "active",
        "validation_audit": {"split_fingerprint_sha256": _SPLIT_FINGERPRINT},
    }
    identity = build_artifact_identity(
        spec.slug,
        "NFL",
        "receiving yards",
        strategy_controls(spec)[0],
        legacy_payload,
        matrix_hash=_MATRIX_HASH,
    )
    return artifact_identity_columns(identity.as_model_blob())


def _identity_map(lam: float) -> dict[str, object]:
    return {
        "kind": "isotonic_pit",
        "lam": lam,
        "x": [0.0, 1.0],
        "y": [0.0, 1.0],
    }


def _calibration_blob() -> dict[str, object]:
    curved = {
        "kind": "isotonic_pit",
        "lam": 1.0,
        "x": [0.0, 0.4, 1.0],
        "y": [0.0, 0.2, 1.0],
    }
    return {
        "kind": receiving.CANDIDATE_NAME,
        "schema_version": receiving.SCHEMA_VERSION,
        "line_probability_only": True,
        "temperature_fit_scope": "pre_map_raw_endpoint_settlement",
        "temperature": 1.4,
        "cdf": {
            "kind": "role_position_two_part_cdf",
            "role_boundary": {
                role: {
                    "kind": "two_part_role_boundary",
                    "intercept": -0.3 if role == "low" else 0.2,
                    "nonpositive": _identity_map(0.75),
                }
                for role in receiving.ROLE_VALUES
            },
            "positive": {
                f"{role}_pos{position}": curved.copy()
                for role in receiving.ROLE_VALUES
                for position in receiving.POSITION_CODES
            },
            "rb_boundary_residual": 0.25,
            "boundary_residual_positions": [3],
        },
        "probability_pool": fixed_pool_blob(),
    }


def _candidate_frame() -> pd.DataFrame:
    from scipy.stats import skewnorm

    result = np.array([0.0, 4.0, 8.0, 12.0, 20.0, 30.0])
    loc = np.array([6.0, 8.0, 10.0, 12.0, 18.0, 24.0])
    scale = np.array([5.0, 5.5, 6.0, 6.5, 8.0, 9.0])
    alpha = np.array([1.0, 1.3, 1.6, 1.9, 1.2, 0.8])
    gate = np.array([0.25, 0.20, 0.15, 0.10, 0.08, 0.05])
    base_f0 = skewnorm.cdf(0.0, alpha, loc=loc, scale=scale)
    f0 = gate + (1.0 - gate) * base_f0
    frame = pd.DataFrame(
        {
            "MeanYr": np.array([8.0, 10.0, 12.0, 15.0, 22.0, 28.0]),
            "Result": result,
            "Blended_EV": np.array([7.0, 9.0, 11.0, 14.0, 20.0, 26.0]),
            "Line": np.array([5.5, 7.5, 9.5, 11.5, 19.5, 27.5]),
            "P": np.array([0.53, 0.57, 0.51, 0.49, 0.55, 0.52]),
            "P_PrePool": np.array([0.58, 0.61, 0.54, 0.46, 0.60, 0.56]),
            "SN_Loc": loc,
            "SN_Scale": scale,
            "SN_Alpha": alpha,
            "Gate": gate,
            "StructuralAdapterStrategy": receiving.CANDIDATE_NAME,
            "StructuralRoute": np.array(["low", "low", "high", "high", "low", "high"]),
            "StructuralFallback": False,
            "StructuralCalibration": receiving.serialize_two_part_calibration(_calibration_blob()),
            "StructuralRole": np.array(["low", "low", "high", "high", "low", "high"]),
            "StructuralPosition": np.array([2, 3, 4, 2, 3, 4]),
            "StructuralF0": f0,
        }
    )
    for column, value in _strategy_identity_columns().items():
        frame[column] = value
    return frame


def test_receiving_v3_gate4_transforms_outcome_endpoints_before_randomization():
    frame = _candidate_frame()
    actual = frame["Result"].to_numpy(dtype=float)
    raw_upper, raw_mass = _pred_cdf_pmf(frame, "SkewNormal", actual, strategy="none")
    expected = receiving.two_part_randomized_pit(
        _calibration_blob(),
        raw_upper - raw_mass,
        raw_upper,
        frame["StructuralF0"].to_numpy(dtype=float),
        frame["StructuralRole"].to_numpy(),
        frame["StructuralPosition"].to_numpy(),
    )

    scored = np.asarray(_randomized_pit_draws(frame, "SkewNormal", actual, strategy="none"))
    np.testing.assert_array_equal(scored, expected)

    different_lines = frame.assign(Line=np.linspace(100.5, 600.5, len(frame)))
    rescored = np.asarray(
        _randomized_pit_draws(different_lines, "SkewNormal", actual, strategy="none")
    )
    np.testing.assert_array_equal(rescored, expected)


def test_load_test_set_retains_and_validates_receiving_v3_contract(tmp_path):
    path = tmp_path / "NFL_receiving-yards.csv"
    _candidate_frame().to_csv(path, index=False)

    loaded = load_test_set(path, "Blended_EV")

    assert {
        "StructuralAdapterStrategy",
        "StructuralCalibration",
        "StructuralRole",
        "StructuralPosition",
        "StructuralF0",
        "P_PrePool",
    }.issubset(loaded.columns)
    assert set(_strategy_identity_columns()).issubset(loaded.columns)
    assert loaded["StructuralCalibration"].nunique() == 1


def test_load_test_set_rejects_partial_or_identity_absent_receiving_contract(tmp_path):
    partial = _candidate_frame().drop(columns=STRATEGY_SIGNATURE_CSV_COLUMN)
    partial_path = tmp_path / "partial-identity.csv"
    partial.to_csv(partial_path, index=False)
    with pytest.raises(ValueError, match="StrategySignature"):
        load_test_set(partial_path, "Blended_EV")

    adapter_only = _candidate_frame().drop(columns=list(_strategy_identity_columns()))
    adapter_path = tmp_path / "adapter-only.csv"
    adapter_only.to_csv(adapter_path, index=False)
    with pytest.raises(ValueError, match="adapter strategy columns require generic"):
        load_test_set(adapter_path, "Blended_EV")


def test_load_test_set_rejects_inactive_or_mismatched_structural_identity(tmp_path):
    inactive = _candidate_frame()
    inactive[STRATEGY_STATUS_CSV_COLUMN] = "killed_fallback"
    inactive_path = tmp_path / "inactive.csv"
    inactive.to_csv(inactive_path, index=False)
    with pytest.raises(ValueError, match="inactive strategy artifact"):
        load_test_set(inactive_path, "Blended_EV")

    mismatched = _candidate_frame()
    mismatched[STRUCTURAL_STRATEGY_CSV_COLUMN] = "none"
    mismatched_path = tmp_path / "mismatched.csv"
    mismatched.to_csv(mismatched_path, index=False)
    with pytest.raises(ValueError, match="stale, mismatched, or wrong-cell"):
        load_test_set(mismatched_path, "Blended_EV")


def test_load_test_set_rejects_missing_or_wrong_schema_receiving_blob(tmp_path):
    frame = _candidate_frame()
    frame.loc[0, "StructuralAdapterStrategy"] = "none"
    path = tmp_path / "nonconstant-experiment.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="one constant nonmissing StructuralAdapterStrategy"):
        load_test_set(path, "Blended_EV")

    frame = _candidate_frame()
    frame.loc[0, "StructuralCalibration"] = None
    path = tmp_path / "missing-payload.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="nonmissing JSON value on every row"):
        load_test_set(path, "Blended_EV")

    frame = _candidate_frame()
    blob = json.loads(frame["StructuralCalibration"].iloc[0])
    blob["schema_version"] = 99
    frame["StructuralCalibration"] = json.dumps(blob)
    path = tmp_path / "wrong-schema.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unknown two-part calibration blob kind or schema"):
        load_test_set(path, "Blended_EV")


@pytest.mark.parametrize(
    ("column", "bad_value", "message"),
    [
        ("StructuralRole", "slot", "only nonmissing low/high"),
        ("StructuralPosition", 2.5, "integer league roster codes"),
        ("StructuralF0", 1.2, "finite probabilities"),
        ("SN_Scale", 0.0, "strictly positive"),
        ("P_PrePool", np.nan, "P_PrePool must be finite"),
    ],
)
def test_load_test_set_rejects_invalid_receiving_v3_row_fields(
    tmp_path, column, bad_value, message
):
    frame = _candidate_frame()
    frame.loc[0, column] = bad_value
    path = tmp_path / f"invalid-{column}.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        load_test_set(path, "Blended_EV")


def test_load_test_set_rejects_partial_receiving_v3_contract(tmp_path):
    frame = _candidate_frame().drop(columns="StructuralF0")
    path = tmp_path / "partial.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match=r"artifact missing required columns.*StructuralF0"):
        load_test_set(path, "Blended_EV")


def test_receiving_v3_gate4_rejects_f0_shape_drift():
    frame = _candidate_frame()
    frame["StructuralF0"] += 1e-4

    with pytest.raises(ValueError, match="does not match the persisted served SkewNormal shape"):
        _randomized_pit_draws(
            frame,
            "SkewNormal",
            frame["Result"].to_numpy(dtype=float),
            strategy="none",
        )


def test_receiving_v3_scorecard_contract_is_cell_scoped():
    with pytest.raises(ValueError, match="does not match the generic model-strategy identity"):
        gate_row(
            _candidate_frame(),
            "Blended_EV",
            league="NBA",
            market="PTS",
            strategy="none",
        )


def test_diff_cli_routes_explicit_cell_identity_to_receiving_v3_gate(tmp_path):
    path = tmp_path / "receiving-v3.csv"
    _candidate_frame().to_csv(path, index=False)

    result = CliRunner().invoke(
        main,
        [
            "--baseline",
            str(path),
            "--candidate",
            str(path),
            "--league",
            "NFL",
            "--market",
            "receiving-yards",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "supersede:" in result.output

    inferred = CliRunner().invoke(
        main,
        ["--baseline", str(path), "--candidate", str(path)],
    )
    assert inferred.exit_code == 0, inferred.output


def test_diff_cli_rejects_mixed_legacy_and_generic_identity(tmp_path):
    candidate = _candidate_frame()
    candidate_path = tmp_path / "candidate.csv"
    candidate.to_csv(candidate_path, index=False)

    legacy = candidate.drop(
        columns=[
            *list(_strategy_identity_columns()),
            "StructuralAdapterStrategy",
            "StructuralRoute",
            "StructuralFallback",
            "StructuralCalibration",
            "StructuralF0",
            "StructuralRole",
            "StructuralPosition",
        ]
    )
    legacy_path = tmp_path / "legacy.csv"
    legacy.to_csv(legacy_path, index=False)

    result = CliRunner().invoke(
        main,
        ["--baseline", str(legacy_path), "--candidate", str(candidate_path)],
    )

    assert result.exit_code != 0
    assert "cannot mix legacy and generic strategy identities" in result.output


def test_row_specific_calibration_payloads_score_each_row_against_its_own_fold_map():
    """A cross-fit artifact carries one payload per fold; each row is scored under its own.

    Splitting the frame's rows across two maps must reproduce, row for row, what scoring each
    half under its own constant-payload frame produces — that is what makes a cross-fit board
    row and a full-HPO one the same statistic.
    """
    frame = _candidate_frame()
    actual = frame["Result"].to_numpy(dtype=float)
    flat = _calibration_blob()
    flat["cdf"]["positive"] = {group: _identity_map(0.0) for group in flat["cdf"]["positive"]}
    payloads = np.where(
        np.arange(len(frame)) < 3,
        frame["StructuralCalibration"].iloc[0],
        receiving.serialize_two_part_calibration(flat),
    )
    mixed = frame.assign(StructuralCalibration=payloads)

    scored = np.asarray(_randomized_pit_draws(mixed, "SkewNormal", actual, strategy="none"))
    curved = np.asarray(_randomized_pit_draws(frame, "SkewNormal", actual, strategy="none"))
    flat_only = np.asarray(
        _randomized_pit_draws(
            frame.assign(StructuralCalibration=payloads[-1]), "SkewNormal", actual, strategy="none"
        )
    )

    np.testing.assert_array_equal(scored[:, :3], curved[:, :3])
    np.testing.assert_array_equal(scored[:, 3:], flat_only[:, 3:])
    assert not np.array_equal(curved[:, 3:], flat_only[:, 3:])


def test_row_specific_payloads_load_and_validate_every_distinct_blob(tmp_path):
    """Load-time validation is per payload, so a bad fold map fails closed like a bad constant."""
    frame = _candidate_frame()
    broken = json.loads(frame["StructuralCalibration"].iloc[0])
    broken["schema_version"] = 99
    frame["StructuralCalibration"] = np.where(
        np.arange(len(frame)) < 3, frame["StructuralCalibration"], json.dumps(broken)
    )
    path = tmp_path / "one-bad-fold.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="unknown two-part calibration blob kind or schema"):
        load_test_set(path, "Blended_EV")
