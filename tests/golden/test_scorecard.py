"""Unit tests for ``training.scorecard`` — the offline ship-gate harness.

Exercises the numeric path (decile binning, compression ratio, scorecard, the five
offline ship gates, and their deterministic-1/0 oracle) on synthetic test-set frames
so no trained model, network, or plotting backend is required.

One representative is kept per distinct behavior family (each gate + its oracle
identity, the dispersion/PIT-KS/tail-KS diagnostics, supersede S1/S2/S3, the
apply_thresholds wiring + strict-threshold pins, and the --live-window CLI). Mirror
cases and pure-internal numeric duplicates were removed in the 2026-06 test cull.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from sportstradamus.training.scorecard import (
    _GATE1_CI_HI_MAX,
    _GATE1_NONINF_MARGIN,
    _GATE2_STAR_Z_MAX,
    _GATE3_BENCH_Z_MAX,
    _GATE4_KS_NOISE_COEF,
    _GATE4_PIT_KS_DELTA,
    _GATE5_ECE_MAX,
    _SUPERSEDE_S3_Z_MIN,
    _decode_sn_loc_scale,
    _dispersion_diagnostics,
    _gate1_brier_ci,
    _gate4_iqr_spread,
    _gate5_ece_debiased,
    _gate5_ece_equal_mass,
    _gate23_segment_match,
    _infer_dist_from_columns,
    _iqr_pred_analytical,
    _ks_uniform,
    _memmel_sharpe_z,
    _pred_midpit,
    _randomized_pit_ks,
    _segment_masks,
    _tail_ks_uniform,
    _zinb_ppf,
    apply_thresholds,
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
# Compression diagnostics (decile table, std-ratio scorecard).
# ---------------------------------------------------------------------------


def test_compression_ratio_below_one_for_shrunk_predictions():
    card = scorecard(_compressed_frame(), "EV", strategy="t", league="NBA", market="PTS")
    assert 0.45 < card.compression_ratio < 0.55
    assert card.top_decile_mae > 0
    assert card.top_decile_bias < 0


def test_brier_skill_score_positive_when_model_beats_book():
    card = scorecard(_priced_frame(seed=0), "EV", strategy="t", league="NBA", market="PTS")
    assert card.brier_skill_score is not None
    assert card.brier_skill_score > 0.1


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


# ---------------------------------------------------------------------------
# Gate 4 — IQR spread (compression) + analytical pooled IQR (Operation Ship 75
# Step 0.2). Brief at /tmp/researcher_g4_audit.md (Outcome B).
# ---------------------------------------------------------------------------


def test_gate4_iqr_ratio_below_one_for_compressed():
    df = _compressed_frame()
    _, _, ratio = _gate4_iqr_spread(df["Result"].to_numpy(), df["EV"].to_numpy())
    assert 0.45 < ratio < 0.55  # predictions shrunk to half-spread


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


# ---------------------------------------------------------------------------
# Gate 5 — equal-mass ECE + debiased ECE (Ship 75 Step 0.5, Roelofs 2022).
# ---------------------------------------------------------------------------


def test_gate5_ece_large_for_confidently_wrong():
    y = np.zeros(1000)
    p_model = np.full(1000, 0.9)  # confident "over" on outcomes that never hit
    assert _gate5_ece_equal_mass(p_model, y) == pytest.approx(0.9, abs=1e-9)


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


# ---------------------------------------------------------------------------
# Supersede S3 — Memmel 2003 paired Sharpe inference (z > 1.645 to ship).
# ---------------------------------------------------------------------------


def test_memmel_sharpe_z_positive_when_candidate_genuinely_better():
    """Candidate with shifted mean returns gets z > 1.645 at adequate N."""
    rng = np.random.default_rng(13)
    n = 1500
    b = rng.normal(0.0005, 0.02, n)
    c = b + rng.normal(0.0008, 0.005, n)  # genuine positive shift
    sr_b, sr_c, z = _memmel_sharpe_z(b, c)
    assert sr_c > sr_b
    assert z > _SUPERSEDE_S3_Z_MIN


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
    }
    assert set(row) == expected
    # Oracle identities: segment z = 0, IQR ratio = 1, ECE = 0, Brier diff < 0.
    assert row["g2_star_z_oracle"] == pytest.approx(0.0)
    assert row["g3_bench_z_oracle"] == pytest.approx(0.0)
    assert row["g4_iqr_ratio_oracle"] == pytest.approx(1.0)
    assert row["g5_ece_oracle"] == pytest.approx(0.0)
    assert row["g1_brier_diff_ci_hi_oracle"] < 0  # oracle beats the (imperfect) book


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


# ---------------------------------------------------------------------------
# Supersede gate — S1 + S2 + S3 (research -> devel, supersede an incumbent).
# ---------------------------------------------------------------------------


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
    b_df = pd.DataFrame(
        {
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


# ---------------------------------------------------------------------------
# Dispersion diagnostics + randomized PIT-KS / tail-KS (Ship 75 binding
# constraint — under-dispersion detection across count + continuous families).
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# --live-window mode (Stage 0 deliverable 0.3).
# ---------------------------------------------------------------------------


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
