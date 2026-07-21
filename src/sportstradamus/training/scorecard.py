#!/usr/bin/env python3
"""Per-cell ship-gate scorecard for trained LightGBMLSS models.

Read by :func:`compute_gates` from the test-set CSV ``meditate`` dumps
(``data/test_sets/{LEAGUE}_{market}.csv``) and merged into the wide
``model_stats.parquet`` row by ``training.report.report()``. The five offline
gates (see ``docs/ship_gate.md``) plus the compression diagnostics live here so
the meditate path and the standalone A/B-test harness share one implementation.

Two entry points:
  * :func:`compute_gates` — pure-Python function returning the per-cell
    gate-column dict for inline use by ``training.report``. No file IO.
  * Click CLI — exercise the same numerics against arbitrary test sets in
    three modes: ``single`` (audit one or more cells; optional scorecard CSV
    output goes to a sandbox path, never ``data/training/``), ``diff``
    (baseline vs candidate, prints the supersede S1/S2/S3 verdict),
    ``--live-window`` (score the last N days of settled history). The CLI
    never writes ``model_stats.parquet`` — that file is owned by
    :func:`training.report.report`.

Usage
-----
  poetry run python3 -m sportstradamus.training.scorecard --league NBA
  poetry run python3 -m sportstradamus.training.scorecard \
      --league NBA --market PTS --strategy ratio_baseline --scatter
  poetry run python3 -m sportstradamus.training.scorecard \
      --baseline data/test_sets/NBA_PTS.csv --candidate /tmp/NBA_PTS_centered.csv
"""

from __future__ import annotations

import functools
import importlib.resources as pkg_resources
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import click
import numpy as np
import pandas as pd
import tabulate
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import gamma as _scipy_gamma
from scipy.stats import nbinom as _scipy_nbinom
from scipy.stats import norm as _scipy_norm
from scipy.stats import skewnorm as _scipy_skewnorm

from sportstradamus import data
from sportstradamus.helpers.distributions import (
    _dp_cdf_pmf,
    _dp_ppf,
    apply_cdf_recal,
    skewnormal_loc_from_mean,
)
from sportstradamus.helpers.integer_distribution import (
    RANDOMIZED_PIT_DRAWS,
    RANDOMIZED_PIT_SEED,
)
from sportstradamus.helpers.io import read_history
from sportstradamus.helpers.provenance import git_sha
from sportstradamus.training.baselines import get_target_normalization
from sportstradamus.training.group_conditional_cdf import (
    deserialize_two_part_calibration,
    ks_supremum,
    two_part_cdf_endpoints,
)
from sportstradamus.training.markets import ALL_MARKETS
from sportstradamus.training.model_strategy_artifacts import STRATEGY_IDENTITY_CSV_COLUMNS
from sportstradamus.training.model_strategy_frame import validate_strategy_frame
from sportstradamus.training.model_strategy_registry import (
    BASE_STRUCTURAL_STRATEGY,
    parse_controls,
)
from sportstradamus.training.ship_config import STAT_META_PATH, TARGET_NORM_NONE, load_stat_meta
from sportstradamus.training.structural_strategies import (
    AFFINE_STRATEGY,
    TWO_PART_STRATEGY,
)

# Ship gates (see docs/ship_gate.md). The promotion lifecycle is a 2x2:
# (set first baseline | supersede incumbent) x (research->devel offline |
# devel->main live).
#
#   * research -> devel, set baseline: the FIVE offline gates computed here —
#     Gate 1 Brier-vs-book paired bootstrap, Gates 2/3 star/bench bias-vs-spread
#     match (denominator = segment σ, NOT σ/sqrt(N) — SE collapses on large-N
#     low-variance bench segments), Gate 4 IQR spread, Gate 5 equal-mass ECE.
#     The per-cell metrics (plus an "oracle" bound) land on the wide
#     ``model_stats.parquet`` row via :func:`compute_gates`; ``apply_thresholds``
#     wires the strict starter pass/fail. Cells with no book Odds leave Gate 1
#     blank; the ship convention is that a blank Gate 1 auto-passes — no book
#     to beat, model wins by default. Gate 5 (model-only calibration) does NOT
#     use Odds, so it still computes for those cells; Gate 5 blank means
#     "couldn't compute" (no P or no Line), not auto-pass.
#   * research -> devel, supersede: pass all six + a paired Brier CI (current-new,
#     95% CI excludes 0 in the new model's favor) + a paired Sharpe improvement on a
#     backdated Kelly sim (supersede_verdict, diff mode).
#   * devel -> main: a profitability gate on live settled data (positive Kelly-sized
#     ROI to set a baseline; >= +0.5% ROI over >= 2 weeks to supersede an incumbent)
#     — see scripts/check_graduation.py.

# Bottom-mean QUARTILE = the bench segment (Gate 3); the top-mean DECILE (N_DECILES)
# is the star segment (Gate 2). Bench is pooled coarser on purpose — low-volume
# players generalize more than stars.
BOTTOM_QUARTILE_FRAC: float = 0.25

# Gate 1 / supersede-S2 paired bootstrap: resample count, RNG seed (fixed so the CI
# is reproducible — the repo has a determinism gate), and the 95% two-sided
# percentile bounds.
_GATE1_N_BOOT: int = 2000
_GATE1_SEED: int = 1729
_CI_LOW_PCT: float = 2.5
_CI_HIGH_PCT: float = 97.5

# Gate 5 ECE bins. EQUAL-MASS (qcut) — distinct from the equal-WIDTH bins in
# training/pipeline.py:_expected_calibration_error; betting cares about calibration
# where the predicted-probability mass actually sits, so equal-mass is the right cut.
_ECE_BINS: int = 10

# Strict starter thresholds for the research->devel set-baseline gate (see
# docs/ship_gate.md). The 5-gate row carries the raw measurements; these set
# pass/fail. A cell ships (research -> devel) iff all six pass.
#   G1 ci_hi  < δ    : non-inferiority — 95% CI upper bound below the statistical-tie
#                      margin δ (ensemble Brier at most δ worse than the book: a tight
#                      tie or a win passes; wildly-worse and underpowered-wide-CI fail)
#   G2 star  z < 0.5 : top-mean-decile bias under half the segment's stdev of outcomes
#   G3 bench z < 0.5 : bottom-quartile bias under half the segment's stdev of outcomes
#   G4 pit_ks < max(δ, 1.36/√n): randomized-PIT KS-uniformity of the predictive CDF —
#                      the worst-case alt-line probability mispricing — under the larger of
#                      the δ effect-size floor and the cell's KS sampling-noise floor
#   G5 ece    < 0.075: 10-bin equal-mass ECE under 7.5% (Roelofs-debiased: raw - null bias offset)
#   G6 star ratio    : (all cells; corr anchor scopes it) stable top-MeanYr pred / recent form,
#                      clustered-CI upper bound at/above the causal real-game floor — catches the
#                      mean-regression the holdout-scored G1-G5 are blind to; anchor-miss auto-pass
# G1 ships on the tie margin (intent: "ensemble at least as good as the book"); the
# stricter ci_hi < 0 (model provably beats the book) is retained as the reported,
# non-decisive ``g1_has_edge`` flag for sizing/prioritization. G2/G3 score the FUSED
# ``Blended_EV`` (what the parlay actually drafts); the raw-``EV`` model-compression view
# rides along as reported ``g2_star_z_raw`` / ``g3_bench_z_raw``.
_GATE1_NONINF_MARGIN: float = 0.005  # ~2% of the ~0.25 book Brier (~1 SE): max ensemble-vs-book Brier degradation called a tie
_GATE1_CI_HI_MAX: float = 0.0  # reported-only g1_has_edge threshold (provable superiority)
_GATE2_STAR_Z_MAX: float = 0.5
_GATE3_BENCH_Z_MAX: float = 0.5
_SHIP_GATES: tuple[str, ...] = ("g1", "g2", "g3", "g4", "g5", "g6")
# Gate 4 is the predictive-shape gate: KS distance of the randomized PIT from Uniform.
# PIT-KS = sup|F_model - F_true| = the worst-case probability error across all quantiles =
# the worst-case alt-line mispricing. δ = 0.05 is the effect-size floor (worst-case
# mispricing at most ~the house vig, the scale below which the model's own error is smaller
# than the edge it hunts); 1.358/√n is the per-cell KS α=0.05 critical value, so a cell is
# never failed below the miscalibration its sample size can resolve. The threshold is the
# larger of the two. The old IQR-ratio compression proxy was retired here: it conflated
# between- vs within-player spread and was fiat-blind on count cells (IQR(Result)=0).
_GATE4_PIT_KS_DELTA: float = 0.05
_GATE4_KS_NOISE_COEF: float = 1.358
_STRUCTURAL_ADAPTER_STRATEGY_COL = "StructuralAdapterStrategy"
_STRUCTURAL_CALIBRATION_COL = "StructuralCalibration"
_STRUCTURAL_ROLE_COL = "StructuralRole"
_STRUCTURAL_POSITION_COL = "StructuralPosition"
_STRUCTURAL_F0_COL = "StructuralF0"
_STRUCTURAL_ROUTE_COL = "StructuralRoute"
_STRUCTURAL_FALLBACK_COL = "StructuralFallback"
_TWO_PART_CONTRACT_COLUMNS: frozenset[str] = frozenset(
    {
        _STRUCTURAL_ADAPTER_STRATEGY_COL,
        _STRUCTURAL_CALIBRATION_COL,
        _STRUCTURAL_ROLE_COL,
        _STRUCTURAL_POSITION_COL,
        _STRUCTURAL_F0_COL,
        "SN_Loc",
        "SN_Scale",
        "SN_Alpha",
        "P",
        "P_PrePool",
        "Line",
    }
)
# Search range for the SkewNormal dispersion scalar c (fit_skewnorm_dispersion_c). Lower
# bound permits tightening an over-wide cell; upper bound matches the count branch's hard cap.
_DISPERSION_C_BOUNDS: tuple[float, float] = (0.1, 10.0)
# Clamp on the Lever-4a additive skew shift s (fit_skewnorm_dispersion_skew). |s| <= 3 keeps
# the served skewness well inside the SkewNormal's range and bounds the 2-param fit's capacity
# at ~2k calibration rows (the skewness MLE is only n^(1/4)-consistent near alpha=0 —
# Hallin & Ley 2014 — so an unbounded shift overfits the gate's own KS).
_DISPERSION_SKEW_BOUNDS: tuple[float, float] = (-3.0, 3.0)
# Deterministic warm starts for the joint (c, s) fit. The objective has a flat-gradient
# Fisher singularity at s = 0, so a single Nelder-Mead seeded there stalls; spanning negative,
# zero, and positive skew lets the arg-min escape it for either skew direction.
_DISPERSION_SKEW_STARTS: tuple[float, ...] = (-1.5, 0.0, 1.5, 3.0)
# Minimum PIT-KS improvement (vs scale-only) for the skew shift to be kept; below it the fit
# returns the pure Lever-1 (c, 0), byte-identical. Set to the measured val->test discount: a
# skew gain smaller than the discount is finite-sample noise that won't survive to the test
# gate, and on an already-calibrated cell the flat surface yields a spurious gain ~0.004.
_DISPERSION_SKEW_MIN_GAIN: float = 0.008
# Reported (not a ship term): the over-tail of the same randomized PIT, restricted to
# u >= _TAIL_PIT_FLOOR. The global Gate-4 KS is a sup over the whole CDF, so it nets
# compensating directional errors and a cell can pass g4 while mispricing the alt-OVER
# tail (receiving-tds: -4% at the standard over, +2% at the deep alt-over, ~0 net KS).
# This surfaces that blind spot for the fix-queue; it does not gate — the deep tail is
# too sample-starved per-cell to threshold without gaming the cutoff.
_TAIL_PIT_FLOOR: float = 0.80
_GATE5_ECE_MAX: float = 0.075

# Gate 5 debias — Roelofs (2022) "Mitigating Bias in Calibration Error
# Estimation". Equal-mass ECE is positively biased at finite N; the bias
# scales O(1/sqrt(N_bin)) and falsely fails ~45% of perfectly calibrated
# NFL-N≈240 cells per the lifecycle gate audit at /tmp/researcher_lifecycle_gate_audit.md.
# Fix: bootstrap-estimate the null bias by drawing y ~ Bernoulli(p_model)
# and subtracting the mean of those null ECEs from the raw ECE.
_GATE5_DEBIAS_RESAMPLES: int = 200
# Distinct from _GATE1_SEED so the null draws don't correlate with the
# Gate-1 paired-Brier bootstrap on the same probabilities.
_GATE5_DEBIAS_SEED: int = 9173

# Gate 6 (anti-shrinkage) — a sixth ship term applied to every cell. It catches a stable-star
# regression toward the global mean that the outcome-scored gates (g1-g5) can miss. The first
# case was the ratio_meanyr cohort: dividing the target by a 365-day MeanYr that conflates "high
# historical average" with "will regress" teaches a high-volume regression real games don't show,
# and g1-g5 — all scored against that same holdout — match its own suppressed stars (top-decile
# pred/Result ~ 1.0), blind to it. But the blind spot is general: g2's bias z divides by the
# outcome sigma, so on a high-variance stat a real proportional star shrinkage launders into a
# tiny z and passes. Gate 6 instead scores the model's STABLE top-MeanYr prediction against
# recent form (Mean10) — the one yardstick these artifacts don't suppress — and fails the cell
# when it sits below the causal real-game floor. The corr anchor (not the normalization or
# family) scopes it. [research-analyst /tmp/researcher_overshrinkage_gate.md, 2026-06-19;
# scope widened to all cells 2026-06-21]
# A "stable" player = recent form within this band of the season baseline; real production
# tracks recent form for them (noisy windows mean-revert, stable ones don't).
_GATE6_STABLE_BAND: float = 0.12  # |Mean10/MeanYr - 1| <= this
# corr(Mean10, Result) anchor with hysteresis: recent form is a valid yardstick only above it
# (exempts MIN ~0.3-0.5 and bursty counts, retains FGA/PR/PRA ~0.6). Two thresholds form a
# deadband so a near-0.55 cell can't flip ship state on a retrain wobble: a fresh cell starts
# being judged at FIRE_ON; a cell whose recent-form leg fired last run keeps being judged down to
# KEEP_ON. Below KEEP_ON (or no prior fire) the recent-form leg is exempt.
_GATE6_FIRE_ON: float = 0.58
_GATE6_KEEP_ON: float = 0.52
# Over-leg guard: count/ZINB bench over-prediction is only testable where the stable bench's
# realized mean clears this — below it Σ Result → 0 and the ratio is discreteness, not a defect.
_GATE6_OVER_MIN_MEAN: float = 1.0
# Tie band below the floor (~the house vig; the g4 delta=0.05 scale): a stable-star CI within
# this of the floor is a statistical tie, not over-shrinkage.
_GATE6_MARGIN: float = 0.03
# Below this many stable stars the cell is too sparse to test -> auto-pass (tiny NFL cells).
_GATE6_MIN_STAR_ROWS: int = 30
# Stable-form star floor the gate enforces: a stable high-volume player's served EV as a
# fraction of trailing-10 form. The causal value from the league gamelogs (6 WNBA / NBA / NFL
# seasons) runs ~0.99 and stat-invariant across basketball, but the gate floor sits below it to
# tolerate ~5% star shrinkage before failing (owner call 2026-06-21); NFL position-mixed yardage
# runs ~0.94. Recompute the causal value when the gamelog grows by a season.
_GATE6_STAR_REF_BASKETBALL: float = 0.95
_GATE6_STAR_REF_NFL: float = 0.94

# Breadth target — at least this fraction of each league's markets should clear the
# 5 gates (docs/ship_gate.md "Top priority"). Drives the per-league rollup.
BREADTH_TARGET_FRAC: float = 0.75

# Probability clip mirrors training/pipeline.py:_PROBA_CLIP so Brier never sees
# exact 0 or 1 from either model or book.
_PROBA_CLIP: float = 1e-6

# Default number of player-mean buckets. Deciles are the report's recommended
# slicing granularity for surfacing the compression cluster.
N_DECILES = 10

# Decile key. MeanYr is the player's season-to-date mean and is the only
# per-player signal present in the dumped test-set CSV (player id is dropped).
DECILE_COL = "MeanYr"
ACTUAL_COL = "Result"

# The ship gates score the fused ``Blended_EV`` — what the parlay actually drafts and what
# ``report.compute_gates`` scores in production (``report._SHIP_PRED_COL``). Default the CLI /
# A-B to that same column so a scorecard reflects the shipping decision rather than the raw
# model's pre-fusion compression. The raw-model-EV view stays available via ``--pred-col EV``
# and is always reported alongside as ``g2_star_z_raw`` / ``g3_bench_z_raw``.
DEFAULT_PRED_COL = "Blended_EV"

# Research artifacts live outside the package data dir — the run log is an append-only
# experiment journal, not shipped data. Climb scripts -> sportstradamus -> src -> repo
# root and write the log to <repo_root>/research/compression_eval/.
_REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_LOG_PATH = _REPO_ROOT / "research" / "compression_eval" / "compression_eval_log.csv"
SCATTER_DIR = Path("/tmp")

# Sandbox default for the CLI's --scorecard-out flag. ``training.report.report()`` is
# the only writer for the production ``model_stats.parquet``; the CLI's full-audit mode
# still writes a CSV snapshot for ad-hoc inspection, but it lands in /tmp by default so
# A/B-test runs never clobber the per-cell view ``meditate`` produces.
_SCORECARD_SANDBOX_DEFAULT = Path("/tmp/scorecard.csv")

# --live-window mode constants (Stage 0 deliverable 0.3).
# Look-back window for MeanYr computation from the per-league gamelog. Matches
# the deprecated stats path that used ~365 days as the season-to-date baseline.
_MEANYR_LOOKBACK_DAYS = 365
# CSV-shaped column set the existing scorecard() path consumes — preserved
# verbatim so live-mode reuses the offline harness unchanged.
_LIVE_EVAL_COLUMNS = ("MeanYr", "Result", "EV", "P", "Odds", "Line")

# Lookup callable signature used by the live-window adapter. Production
# implementation closes over Stats.gamelog; tests inject a deterministic mock.
MeanYrLookup = Callable[[str, str, pd.Timestamp], float]


@dataclass(frozen=True)
class Scorecard:
    """One experiment's compression summary, written as a run-log row."""

    timestamp: str
    git_sha: str
    strategy: str
    league: str
    market: str
    pred_col: str
    n_rows: int
    global_mae: float
    top_decile_mae: float
    top_decile_bias: float
    compression_ratio: float
    top_decile_compression_ratio: float
    pred_meanyr_corr: float
    result_meanyr_corr: float
    # Appended last so existing compression_eval_log.csv files (written before
    # this field existed) keep appending without breaking pandas concat reads.
    brier_skill_score: float | None
    # Bottom-mean-QUARTILE signed bias + that quartile's empirical (actual) mean,
    # and the top-mean-decile empirical mean. Together they make the calibration
    # ship gate (Universal decision threshold condition 4 / Gate 1) computable
    # directly from the logged scorecard: the bottom quartile drives the relative
    # + absolute low-volume check, the top-decile mean sizes the top-end absolute
    # backstop (top_decile_bias already carries the top-end bias). Appended last
    # for the same CSV-append back-compat reason.
    bottom_quartile_bias: float
    bottom_quartile_mean: float
    top_decile_mean: float
    # Bottom-quartile MAE — the bench-side companion to top_decile_mae. Track 2
    # (supersede) requires both segment MAEs to improve >= 5%, so the bench MAE must
    # be on the scorecard. Appended last for the same CSV-append back-compat reason.
    bottom_quartile_mae: float


def load_test_set(path: Path, pred_col: str) -> pd.DataFrame:
    """Load a dumped test-set CSV, keeping only the columns the harness needs.

    Args:
        path: Path to a ``{LEAGUE}_{market}.csv`` produced by ``meditate``.
        pred_col: Predicted-mean column to evaluate (``EV`` or ``Blended_EV``).

    Returns:
        Frame with ``MeanYr``, ``Result``, the prediction column, optional
        priced columns (``P``, ``Odds``, ``Line``), and any per-row
        distribution parameters present (``SN_Loc`` / ``SN_Scale`` /
        ``SN_Alpha`` for SkewNormal, ``MIX_Loc1`` / ``MIX_Loc2`` /
        ``MIX_Scale1`` / ``MIX_Scale2`` / ``MIX_W1`` for the 2-component
        Gaussian mixture, ``R`` / ``NB_P`` for NegBin/ZINB,
        ``DP_MU`` / ``DP_PHI`` for DPO, ``Alpha`` for Gamma/ZAGamma,
        ``Gate`` for the zero-inflated variants).
        Rows with non-finite values in any required column are dropped.

    Raises:
        ValueError: If a required column is missing from the CSV.
    """
    df = pd.read_csv(path)
    required = {DECILE_COL, ACTUAL_COL, pred_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing required columns: {sorted(missing)}")
    # Opportunistically keep brier-skill inputs when present; older CSVs without
    # them stay loadable and just skip the third gate downstream.
    # Keep both mean columns regardless of pred_col: g2/g3 ship-score the fused
    # ``Blended_EV`` while reporting the raw-``EV`` compression view, and the Gamma PIT
    # reads ``EV`` as a distribution parameter.
    optional = {
        "P",
        "P_PrePool",
        "P_standalone",
        "Odds",
        "Line",
        "Player",
        "Date",
        "EV",
        "Blended_EV",
    } & set(df.columns)
    # Per-row distribution params (Operation Ship 75 Step 0.2 G4 audit).
    # ``training/pipeline.py::_step_persist_artifacts`` dumps these at lines
    # 1191-1212; older CSVs predating that dump simply leave them out and the
    # gate falls back to the point-IQR estimator via `_infer_dist_from_columns`
    # returning None.
    decode_denom_cols: set[str] = set()
    if "DenomCol" in df.columns:
        denom_values = df["DenomCol"].dropna().astype(str).unique()
        if len(denom_values) != 1:
            raise ValueError(f"{path.name} must persist exactly one DenomCol value")
        denom_col = str(denom_values[0])
        if denom_col not in df.columns:
            raise ValueError(f"{path.name} DenomCol references missing column {denom_col!r}")
        decode_denom_cols = {"DenomCol", denom_col}

    decode_strategy_cols: set[str] = set()
    if "TargetNormalization" in df.columns:
        strategy_values = df["TargetNormalization"].dropna().astype(str).unique()
        if len(strategy_values) != 1:
            raise ValueError(f"{path.name} must persist exactly one TargetNormalization value")
        get_target_normalization(str(strategy_values[0]))
        decode_strategy_cols.add("TargetNormalization")

    dist_params = (
        (
            {
                "SN_Loc",
                "SN_Scale",
                "SN_Alpha",
                "MIX_Loc1",
                "MIX_Loc2",
                "MIX_Scale1",
                "MIX_Scale2",
                "MIX_W1",
                "Mean10",  # centered_additive_mean10 SkewNormal decode re-adds this baseline to loc
                "GamesPlayed",  # centered_additive_eb decode re-adds an EB prior over (MeanYr, GamesPlayed)
                "GlobalMean",  # …shrunk toward this persisted global mean
                "R",
                "NB_P",
                "DP_MU",
                "DP_PHI",
                "Alpha",
                "Gate",
                "PITRecalKnots",  # §6.1 Rung C whole-CDF map g; Gate 4 warps the PIT through it
                _STRUCTURAL_ADAPTER_STRATEGY_COL,
                _STRUCTURAL_CALIBRATION_COL,
                _STRUCTURAL_ROLE_COL,
                _STRUCTURAL_POSITION_COL,
                _STRUCTURAL_F0_COL,
                _STRUCTURAL_ROUTE_COL,
                _STRUCTURAL_FALLBACK_COL,
                *STRATEGY_IDENTITY_CSV_COLUMNS,
            }
            & set(df.columns)
        )
        | decode_denom_cols
        | decode_strategy_cols
    )
    out = df[sorted(required | optional | dist_params)].copy()
    # Filter non-finite rows on required columns only — missing P/Odds/Line rows
    # are filtered locally inside _brier_skill_score so older CSVs that lack
    # those columns still pass the harness.
    required_view = out[list(required)].replace([np.inf, -np.inf], np.nan)
    out = out[required_view.notna().all(axis=1)]
    validate_strategy_frame(out)
    _two_part_contract(out)
    return out


def decile_table(df: pd.DataFrame, pred_col: str, n_deciles: int = N_DECILES) -> pd.DataFrame:
    """Build the per-player-mean decile MAE/bias table.

    Args:
        df: Frame from :func:`load_test_set`.
        pred_col: Prediction column name.
        n_deciles: Number of equal-frequency ``MeanYr`` buckets.

    Returns:
        One row per decile with mean ``MeanYr``, count, MAE, signed bias
        (``pred - actual``), and mean predicted vs. actual.
    """
    work = df.copy()
    work["decile"] = pd.qcut(work[DECILE_COL].rank(method="first"), n_deciles, labels=False)
    err = work[pred_col] - work[ACTUAL_COL]
    work["abs_err"] = err.abs()
    work["bias"] = err
    grouped = work.groupby("decile")
    return pd.DataFrame(
        {
            "meanyr": grouped[DECILE_COL].mean(),
            "n": grouped.size(),
            "mae": grouped["abs_err"].mean(),
            "bias": grouped["bias"].mean(),
            "pred_mean": grouped[pred_col].mean(),
            "actual_mean": grouped[ACTUAL_COL].mean(),
        }
    ).reset_index()


def _compression_ratio(actual: np.ndarray, pred: np.ndarray) -> float:
    """Return ``std(pred) / std(actual)``; 1.0 = no compression, <1 = compressed."""
    a_std = float(np.std(actual))
    if a_std == 0.0:
        return float("nan")
    return float(np.std(pred)) / a_std


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, NaN-safe for degenerate (zero-variance) inputs."""
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _calibration_inputs(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(p_model, y)`` for the model-only calibration gate (Gate 5).

    Needs ``P`` (calibrated model over-probability), ``Line`` (the posted prop line)
    and ``Result`` to derive ``y = Result >= Line``. Independent of the book's
    ``Odds`` — Gate 5 only asks whether the MODEL's own probabilities are calibrated
    to outcomes; it does not compare against the book. Returns ``None`` when ``P`` or
    ``Line`` is missing or every row is non-finite.
    """
    needed = {"P", "Line"}
    if not needed.issubset(df.columns):
        return None
    sub = df[["P", "Line", ACTUAL_COL]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) == 0:
        return None
    y = (sub[ACTUAL_COL] >= sub["Line"]).astype(float).to_numpy()
    p_model = np.clip(sub["P"].to_numpy(), _PROBA_CLIP, 1 - _PROBA_CLIP)
    return p_model, y


def _priced_rows(df: pd.DataFrame) -> pd.DataFrame | None:
    """Rows usable for the priced Brier gates: finite P, Odds, Line, and outcome.

    The single source of truth for which test rows enter Gate 1, so ancillary
    columns (e.g. Player for the clustered bootstrap) align to the same rows.
    """
    if "Odds" not in df.columns:
        return None
    if not {"P", "Odds", "Line"}.issubset(df.columns):
        return None
    sub = df[["P", "Odds", "Line", ACTUAL_COL]].replace([np.inf, -np.inf], np.nan).dropna()
    return sub if len(sub) else None


def _brier_inputs(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Index] | None:
    """Return ``(p_model, p_book, y, index)`` for the priced Brier gates, or None.

    Layers the book's ``Odds`` (book under-probability ⇒ book over = ``1 - Odds``) on
    top of :func:`_calibration_inputs`. The row set is re-filtered to drop rows with
    non-finite ``Odds`` (so the Brier and ECE row sets can differ when some events
    have a posted line but no book quote). Returns ``None`` when ``Odds`` is missing
    entirely or every priced row is non-finite. Shared by
    :func:`_brier_skill_score` and Gate 1 (:func:`_gate1_brier_ci`); ``index`` is the
    surviving-row index so callers can align ancillary columns (Player) to the same
    rows without re-deriving the filter.
    """
    sub = _priced_rows(df)
    if sub is None:
        return None
    y = (sub[ACTUAL_COL] >= sub["Line"]).astype(float).to_numpy()
    p_model = np.clip(sub["P"].to_numpy(), _PROBA_CLIP, 1 - _PROBA_CLIP)
    p_book = np.clip(1.0 - sub["Odds"].to_numpy(), _PROBA_CLIP, 1 - _PROBA_CLIP)
    return p_model, p_book, y, sub.index


def _brier_skill_score(df: pd.DataFrame) -> float | None:
    """1 - brier(model_P) / brier(book_over) on the test set, or None if cols absent.

    Informational summary kept on the :class:`Scorecard` / run log; the Gate-1 ship
    signal is the paired bootstrap CI in :func:`_gate1_brier_ci`, not this ratio.
    """
    inputs = _brier_inputs(df)
    if inputs is None:
        return None
    p_model, p_book, y, _ = inputs
    brier_model = float(np.mean((p_model - y) ** 2))
    brier_book = float(np.mean((p_book - y) ** 2))
    if brier_book <= 0:
        return None
    return 1.0 - brier_model / brier_book


def _standalone_g1_hi(
    df: pd.DataFrame, p_book: np.ndarray, y: np.ndarray, index: pd.Index
) -> float | None:
    """Gate-1 CI upper bound for the *pre-blend* model probabilities (``P_standalone``).

    Scored on the same priced rows as the fused gate (``index`` from
    :func:`_brier_inputs`) against the same book, so the two upper bounds are
    directly comparable: it shows whether the standalone model or the book
    carries the fused pass. Report-only — never a ship term. ``None`` when the
    column is absent (CSVs predating the dump).
    """
    if "P_standalone" not in df.columns:
        return None
    p_sa = np.clip(df.loc[index, "P_standalone"].to_numpy(), _PROBA_CLIP, 1 - _PROBA_CLIP)
    return _gate1_brier_ci(p_sa, p_book, y, np.random.default_rng(_GATE1_SEED))[2]


def _segment_masks(df: pd.DataFrame, n_deciles: int = N_DECILES) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(star_mask, bench_mask)`` boolean arrays over ``df`` rows by ``MeanYr``.

    ``star`` = the top-mean decile (value-quantile threshold, ties included); ``bench``
    = the bottom-mean quartile (:data:`BOTTOM_QUARTILE_FRAC`, rank-based equal-frequency
    so it is never empty on tiny frames). Shared by :func:`scorecard` and the segment
    gates (:func:`_gate23_segment_se`) so both slice players identically.
    """
    star_mask = (df[DECILE_COL] >= df[DECILE_COL].quantile(1 - 1 / n_deciles)).to_numpy()
    rank = df[DECILE_COL].rank(method="first").to_numpy()
    n_bottom = max(1, int(np.ceil(BOTTOM_QUARTILE_FRAC * len(df))))
    bench_mask = rank <= n_bottom
    return star_mask, bench_mask


def scorecard(
    df: pd.DataFrame,
    pred_col: str,
    *,
    strategy: str,
    league: str,
    market: str,
    n_deciles: int = N_DECILES,
) -> Scorecard:
    """Compute the headline compression metrics for one test set.

    The ``*_meanyr_corr`` fields mirror ``training/report.py``'s
    ``ev_meanyr_corr`` / ``result_meanyr_corr`` definition
    (``corr(MeanYr, value - MeanYr)``) so the harness and the training report
    speak the same language.

    Args:
        df: Frame from :func:`load_test_set` (columns: ``MeanYr``, ``Result``,
            ``pred_col``, and optionally ``P``, ``Odds``, ``Line``).
        pred_col: Predicted-mean column to evaluate (``EV`` or ``Blended_EV``).
        strategy: Label written to the run log (e.g. ``"ratio_baseline"``).
        league: League tag written to the run log (e.g. ``"NBA"``).
        market: Market tag written to the run log (e.g. ``"PTS"``).
        n_deciles: Number of equal-frequency ``MeanYr`` buckets.

    Returns:
        A :class:`Scorecard` with global and per-decile compression metrics,
        including ``bottom_quartile_bias`` / ``bottom_quartile_mean`` and
        ``top_decile_mean`` for the calibration ship gate.
    """
    meanyr = df[DECILE_COL].to_numpy()
    actual = df[ACTUAL_COL].to_numpy()
    pred = df[pred_col].to_numpy()

    table = decile_table(df, pred_col, n_deciles)
    top = table.iloc[-1]
    star_mask, bq_mask = _segment_masks(df, n_deciles)
    bottom_quartile_bias = float(np.mean(pred[bq_mask] - actual[bq_mask]))
    bottom_quartile_mean = float(np.mean(actual[bq_mask]))
    bottom_quartile_mae = float(np.mean(np.abs(pred[bq_mask] - actual[bq_mask])))
    brier_skill = _brier_skill_score(df)

    return Scorecard(
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        git_sha=git_sha(),
        strategy=strategy,
        league=league,
        market=market,
        pred_col=pred_col,
        n_rows=len(df),
        global_mae=float(np.abs(pred - actual).mean()),
        top_decile_mae=float(top["mae"]),
        top_decile_bias=float(top["bias"]),
        compression_ratio=_compression_ratio(actual, pred),
        top_decile_compression_ratio=_compression_ratio(actual[star_mask], pred[star_mask]),
        pred_meanyr_corr=_corr(meanyr, pred - meanyr),
        result_meanyr_corr=_corr(meanyr, actual - meanyr),
        brier_skill_score=brier_skill,
        bottom_quartile_bias=bottom_quartile_bias,
        bottom_quartile_mean=bottom_quartile_mean,
        top_decile_mean=float(top["actual_mean"]),
        bottom_quartile_mae=bottom_quartile_mae,
    )


def _bootstrap_mean_ci(
    values: np.ndarray, rng: np.random.Generator, n_boot: int = _GATE1_N_BOOT
) -> tuple[float, float, float]:
    """Percentile bootstrap of the mean of ``values``: ``(mean, ci_lo, ci_hi)``.

    Resamples ``values`` with replacement ``n_boot`` times (seeded ``rng`` for a
    reproducible CI) and returns the point mean plus the 95% two-sided percentile
    bounds, specialized to the mean of an already-computed per-event statistic
    (Gate 1 here, supersede S2 later).
    Returns ``(nan, nan, nan)`` when ``values`` is empty after dropping non-finites.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draws[i] = values[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(draws, [_CI_LOW_PCT, _CI_HIGH_PCT])
    return float(values.mean()), float(lo), float(hi)


def _bootstrap_mean_ci_clustered(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = _GATE1_N_BOOT,
) -> tuple[float, float, float]:
    """Cluster (player) block bootstrap of the mean of ``values``.

    Resamples whole clusters with replacement, so within-cluster correlation is
    preserved and the CI is not anti-conservative on repeated-player panels.
    ``n_boot`` defaults to :data:`_GATE1_N_BOOT`. Returns ``(mean, ci_lo, ci_hi)``;
    ``(nan, nan, nan)`` if empty.
    """
    values = np.asarray(values, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    finite = np.isfinite(values)
    values, cluster_ids = values[finite], cluster_ids[finite]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    uniq = np.unique(cluster_ids)
    groups = [values[cluster_ids == c] for c in uniq]
    n_clusters = len(uniq)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        pick = rng.integers(0, n_clusters, n_clusters)
        draws[i] = np.concatenate([groups[j] for j in pick]).mean()
    lo, hi = np.percentile(draws, [_CI_LOW_PCT, _CI_HIGH_PCT])
    return float(values.mean()), float(lo), float(hi)


def _bootstrap_ratio_ci_clustered(
    num: np.ndarray,
    den: np.ndarray,
    cluster_ids: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = _GATE1_N_BOOT,
) -> tuple[float, float, float]:
    """Cluster (player) block bootstrap of the ratio of sums ``Σnum / Σden``.

    The ratio-of-sums is robust to the tiny denominators a mean-of-ratios blows up on (a
    bench player's ``Mean10`` of 1-2); resampling whole player clusters keeps repeated-player
    correlation from under-counting the CI. Returns ``(ratio, ci_lo, ci_hi)``;
    ``(nan, nan, nan)`` if empty.
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    finite = np.isfinite(num) & np.isfinite(den)
    num, den, cluster_ids = num[finite], den[finite], cluster_ids[finite]
    if len(num) == 0 or den.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    uniq = np.unique(cluster_ids)
    groups = {c: np.where(cluster_ids == c)[0] for c in uniq}
    n_clusters = len(uniq)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        pick = np.concatenate([groups[uniq[j]] for j in rng.integers(0, n_clusters, n_clusters)])
        draws[i] = num[pick].sum() / den[pick].sum()
    lo, hi = np.percentile(draws, [_CI_LOW_PCT, _CI_HIGH_PCT])
    return float(num.sum() / den.sum()), float(lo), float(hi)


def _gate1_brier_ci_clustered(
    p_model: np.ndarray,
    p_book: np.ndarray,
    y: np.ndarray,
    cluster_ids: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Gate 1 paired Brier CI under a player-clustered bootstrap."""
    d = (p_model - y) ** 2 - (p_book - y) ** 2
    return _bootstrap_mean_ci_clustered(d, cluster_ids, rng)


def _iqr(values: np.ndarray) -> float:
    """Inter-quartile range (75th - 25th percentile) of ``values``."""
    p75, p25 = np.percentile(np.asarray(values, dtype=float), [75, 25])
    return float(p75 - p25)


# Gate 4 analytical IQR (Operation Ship 75 Step 0.2) — Outcome B per the
# research brief at /tmp/researcher_g4_audit.md. The old gate computed
# IQR(point_pred) / IQR(actual), a sharpness-vs-calibration category error
# (Gneiting & Raftery 2007): point predictions are conditional means and
# always lose to realized-outcome spread on the IQR. The fix is to compute
# the pooled IQR of the predictive distribution per-row, by inverting the
# per-row CDF at q = 0.25 / 0.75, then taking the IQR of the concatenated
# bag. Validated against sample-based IQR with M=1000 on five real cells
# (/tmp/g4_iqr_experiment.py) — agrees to <=0.001 on ZINB, within 10% on
# SkewNormal.


# torch's NegativeBinomial(probs=p) treats p as "probability of success" and
# counts FAILURES; scipy.stats.nbinom(n, p) treats p the same way but counts
# successes-needed before n failures, so the per-trial parameter is mirrored:
# scipy_p = 1 - torch_probs. Mapping documented in /tmp/researcher_g4_audit.md.
def _zinb_ppf(q: float, r: np.ndarray, nb_p: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """Vectorized inverse CDF for ZINB(``gate``, ``r``, ``nb_p``).

    Mixture: ``F(k) = π + (1−π)·F_NB(k; r, p)`` where ``π = gate``. Quantiles
    at ``q ≤ π`` land at 0; above the gate, rescale to the conditional NB.
    NegBin params follow PyTorch / LightGBMLSS (``probs``); scipy uses the
    mirrored ``1 − probs`` for its second argument.
    """
    r = np.asarray(r, dtype=float)
    nb_p = np.asarray(nb_p, dtype=float)
    gate = np.asarray(gate, dtype=float)
    # q <= π: quantile in the structural-zero point mass.
    in_gate = q <= gate
    scaled_q = np.where(in_gate, 0.0, (q - gate) / np.maximum(1.0 - gate, 1e-12))
    nb_q = _scipy_nbinom.ppf(scaled_q, r, 1.0 - nb_p)
    return np.where(in_gate, 0.0, nb_q)


def _mix_cdf(
    y: np.ndarray,
    w1: np.ndarray,
    loc1: np.ndarray,
    scale1: np.ndarray,
    loc2: np.ndarray,
    scale2: np.ndarray,
) -> np.ndarray:
    """CDF of the 2-component Gaussian mixture: ``w1·Φ₁(y) + (1−w1)·Φ₂(y)``."""
    return w1 * _scipy_norm.cdf(y, loc=loc1, scale=scale1) + (1.0 - w1) * _scipy_norm.cdf(
        y, loc=loc2, scale=scale2
    )


# Mixture-quantile bisection: every quantile lies within ±8·max(scale) of the component
# locs (normal mass beyond 8σ ≈ 6e-16, under float64 CDF resolution), and 80 halvings
# shrink that bracket past any further float64 refinement.
_MIX_PPF_BRACKET_SIGMAS: float = 8.0
_MIX_PPF_BISECT_ITERS: int = 80


def _mix_ppf(
    q: float,
    w1: np.ndarray,
    loc1: np.ndarray,
    scale1: np.ndarray,
    loc2: np.ndarray,
    scale2: np.ndarray,
) -> np.ndarray:
    """Vectorized inverse CDF for the 2-component Gaussian mixture.

    A mixture quantile has no closed form; the CDF is continuous and strictly
    increasing, so bisect per row inside the ±8σ bracket around the components.
    """
    smax = np.maximum(scale1, scale2)
    lo = np.minimum(loc1, loc2) - _MIX_PPF_BRACKET_SIGMAS * smax
    hi = np.maximum(loc1, loc2) + _MIX_PPF_BRACKET_SIGMAS * smax
    for _ in range(_MIX_PPF_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        below = _mix_cdf(mid, w1, loc1, scale1, loc2, scale2) < q
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def _infer_dist_from_columns(df: pd.DataFrame) -> str | None:
    """Identify the distribution family from per-row parameter columns.

    Returns one of ``"Mixture"``, ``"SkewNormal"``, ``"NegBin"`, ``"ZINB"``,
    ``"DPO"``, ``"Gamma"``, ``"ZAGamma"`` based on which params
    ``training/pipeline.py`` ``_step_persist_artifacts`` dumped into the
    test-set CSV (~lines 1191-1212). ``None`` for legacy / synthetic frames
    missing every distribution param — those keep the back-compat point-IQR
    semantics.
    """
    cols = set(df.columns)
    if {"MIX_Loc1", "MIX_Loc2", "MIX_Scale1", "MIX_Scale2", "MIX_W1"} <= cols:
        return "Mixture"
    if {"SN_Loc", "SN_Scale", "SN_Alpha"} <= cols:
        return "SkewNormal"
    if {"DP_MU", "DP_PHI"} <= cols:
        return "DPO"
    if {"Alpha", "EV"} <= cols and "R" not in cols:
        return "ZAGamma" if "Gate" in cols else "Gamma"
    if {"R", "NB_P"} <= cols:
        return "ZINB" if "Gate" in cols else "NegBin"
    return None


# SkewNormal (and Mixture) strategies the gate can decode from the dumped params. ``ratio_meanyr``,
# ``ratio_projvol``, and ``centered_additive_mean10`` ignore ``GlobalMean``;
# ``ratio_projvol`` multiplies by the persisted ``DenomCol`` (a projected-volume column,
# with the same per-row MeanYr fallback the served path uses);
# ``centered_additive_eb_meanyr_k10`` re-adds ``GlobalMean`` from the persisted column
# (fallback 0.0).
_SN_DECODE_STRATEGIES: frozenset[str] = frozenset(
    {
        "ratio_meanyr",
        "ratio_projvol",
        "centered_additive_mean10",
        "centered_additive_eb_meanyr_k10",
    }
)


def _decode_sn_loc_scale(df: pd.DataFrame, strategy: str) -> tuple[np.ndarray, np.ndarray]:
    """Decode raw SkewNormal ``loc`` / ``scale`` to EV-space per strategy.

    Dispatches through the canonical ``baselines`` registry so the gate scores the
    same absolute predictive that ``prediction.model_prob`` prices, rather than a
    hand-rolled mirror that can drift. ``ratio_meanyr`` multiplies both by the cell's
    denominator; ``centered_additive_mean10`` re-adds the Mean10 baseline to ``loc`` —
    irrelevant for the location-free IQR but load-bearing for the PIT, which is where
    the old hand-rolled mirror silently dropped the offset. ``centered_additive_eb_meanyr_k10``
    re-adds an empirical-Bayes prior that shrinks toward ``global_mean``, which the
    pipeline persists per-row as the constant ``GlobalMean`` column; without it the
    prior shrinks toward 0 and the decode silently corrupts the PIT. Ratio/Mean10
    decodes ignore ``global_mean``, so the legacy 0.0 fallback is correct for them.

    The denominator must match the one the cell encoded (and serves) with: a
    zero-inflated SkewNormal cell uses ``MeanYr_nonzero``, not ``MeanYr``, so the
    pipeline persists the choice as the constant ``DenomCol`` column. Decoding a
    nonzero-denominator cell against ``MeanYr`` mis-scales its dispersion and fails
    Gate 4 on a predictive the betting path never priced.
    """
    raw_loc = df["SN_Loc"].to_numpy(dtype=float)
    raw_scale = df["SN_Scale"].to_numpy(dtype=float)
    if strategy not in _SN_DECODE_STRATEGIES:
        return raw_loc, raw_scale
    global_mean = float(df["GlobalMean"].iloc[0]) if "GlobalMean" in df.columns else 0.0
    denom_col = str(df["DenomCol"].iloc[0]) if "DenomCol" in df.columns else "MeanYr"
    strat = get_target_normalization(strategy)
    return (
        strat.decode_loc(raw_loc, df, global_mean, denom_col),
        strat.decode_scale(raw_scale, df, denom_col),
    )


def _decode_mix_params(
    df: pd.DataFrame, strategy: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode 2-component Gaussian-mixture params to EV space per strategy.

    Returns ``(w1, loc1, scale1, loc2, scale2)``. The pipeline persists the MIX
    locs/scales in the cell's NORMALIZED space exactly like ``SN_Loc`` /
    ``SN_Scale``, so both component locs run through ``decode_loc`` and both
    scales through ``decode_scale`` with the same ``GlobalMean`` / ``DenomCol``
    fallbacks as :func:`_decode_sn_loc_scale`. The component weight is a
    probability — scale-free, returned as-is.
    """
    w1 = df["MIX_W1"].to_numpy(dtype=float)
    loc1 = df["MIX_Loc1"].to_numpy(dtype=float)
    scale1 = df["MIX_Scale1"].to_numpy(dtype=float)
    loc2 = df["MIX_Loc2"].to_numpy(dtype=float)
    scale2 = df["MIX_Scale2"].to_numpy(dtype=float)
    if strategy not in _SN_DECODE_STRATEGIES:
        return w1, loc1, scale1, loc2, scale2
    global_mean = float(df["GlobalMean"].iloc[0]) if "GlobalMean" in df.columns else 0.0
    denom_col = str(df["DenomCol"].iloc[0]) if "DenomCol" in df.columns else "MeanYr"
    strat = get_target_normalization(strategy)
    return (
        w1,
        strat.decode_loc(loc1, df, global_mean, denom_col),
        strat.decode_scale(scale1, df, denom_col),
        strat.decode_loc(loc2, df, global_mean, denom_col),
        strat.decode_scale(scale2, df, denom_col),
    )


def _pred_ppf(df: pd.DataFrame, dist: str, q: float, *, strategy: str) -> np.ndarray:
    """Per-row inverse CDF ``F⁻¹(q)`` of the predictive distribution.

    Shared by the Gate-4 pooled IQR and the PIT / coverage over-dispersion
    diagnostics so both read the same per-row params the training pipeline
    dumped into the test-set CSV. ``q`` is a scalar in ``(0, 1)``; the ZINB /
    ZAGamma mixtures invert through the custom routines (a quantile at or below
    the zero gate lands on 0).
    """
    if dist == "SkewNormal":
        loc, scale = _decode_sn_loc_scale(df, strategy)
        alpha = df["SN_Alpha"].to_numpy(dtype=float)
        if "Gate" not in df.columns:
            return _scipy_skewnorm.ppf(q, alpha, loc=loc, scale=scale)

        # Exact inverse of the served zero-adjusted predictive.  The positive-component
        # SkewNormal still has mathematical support below zero, so its CDF contributes to
        # the left limit at the zero atom.  Quantiles inside that jump are exactly zero;
        # quantiles on either side invert the appropriately rescaled base CDF.
        gate = np.clip(df["Gate"].to_numpy(dtype=float), 0.0, 1.0)
        base_at_zero = _scipy_skewnorm.cdf(0.0, alpha, loc=loc, scale=scale)
        left = (1.0 - gate) * base_at_zero
        right = left + gate
        denom = np.maximum(1.0 - gate, 1e-12)
        base_q = np.where(q < left, q / denom, (q - gate) / denom)
        base_q = np.clip(base_q, 0.0, 1.0)
        base_ppf = _scipy_skewnorm.ppf(base_q, alpha, loc=loc, scale=scale)
        return np.where((q >= left) & (q <= right), 0.0, base_ppf)
    if dist == "Mixture":
        return _mix_ppf(q, *_decode_mix_params(df, strategy))
    if dist == "NegBin":
        r = df["R"].to_numpy(dtype=float)
        p = df["NB_P"].to_numpy(dtype=float)
        return _scipy_nbinom.ppf(q, r, 1.0 - p)
    if dist == "ZINB":
        r = df["R"].to_numpy(dtype=float)
        p = df["NB_P"].to_numpy(dtype=float)
        gate = df["Gate"].to_numpy(dtype=float)
        return _zinb_ppf(q, r, p, gate)
    if dist == "DPO":
        mu = df["DP_MU"].to_numpy(dtype=float)
        phi = df["DP_PHI"].to_numpy(dtype=float)
        return _dp_ppf(q, mu, phi)
    if dist == "Gamma":
        # rate = concentration / EV; scale = 1 / rate = EV / concentration.
        a = df["Alpha"].to_numpy(dtype=float)
        ev = df["EV"].to_numpy(dtype=float)
        return _scipy_gamma.ppf(q, a, scale=ev / np.maximum(a, 1e-12))
    if dist == "ZAGamma":
        # Mixture of point-mass at 0 (gate) + Gamma on Y>0. Per-row inversion
        # mirrors _zinb_ppf: q ≤ π → 0, else rescaled Gamma quantile.
        a = df["Alpha"].to_numpy(dtype=float)
        ev = df["EV"].to_numpy(dtype=float)
        gate = df["Gate"].to_numpy(dtype=float)
        scale = ev / np.maximum(a, 1e-12)
        rescaled = np.where(gate >= q, 0.0, (q - gate) / np.maximum(1 - gate, 1e-12))
        return np.where(gate >= q, 0.0, _scipy_gamma.ppf(rescaled, a, scale=scale))
    raise ValueError(f"Unknown distribution family for analytical IQR: {dist!r}")


def _iqr_pred_analytical(df: pd.DataFrame, dist: str, *, strategy: str) -> float:
    """Pooled analytical IQR of the predicted distribution across all rows.

    Per-row ``q25 = F⁻¹(0.25)`` and ``q75 = F⁻¹(0.75)`` (via :func:`_pred_ppf`)
    are concatenated and the bag's ``percentile(75) − percentile(25)`` is the
    pooled IQR — a discrete approximation to the mixture-predictive IQR,
    validated against sample-based pooled IQR to ≤ 0.001 on count families.
    """
    q25 = _pred_ppf(df, dist, 0.25, strategy=strategy)
    q75 = _pred_ppf(df, dist, 0.75, strategy=strategy)
    return _iqr(np.concatenate([q25, q75]))


def _pred_cdf_pmf(
    df: pd.DataFrame, dist: str, y: np.ndarray, *, strategy: str
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row predictive CDF ``F(y)`` and outcome point-mass ``P(Y=y)``.

    The point mass is 0 everywhere for the continuous families (SkewNormal,
    Mixture, Gamma) and the per-row probability of the realized integer for the
    count / zero-inflated families. Shared by the mid-PIT and the randomized PIT
    so both read the same family parameterization; the zero-inflated mixtures
    fold the gate into both terms.
    """
    y = np.asarray(y, dtype=float)
    if dist == "SkewNormal":
        loc, scale = _decode_sn_loc_scale(df, strategy)
        alpha = df["SN_Alpha"].to_numpy(dtype=float)
        cdf = _scipy_skewnorm.cdf(y, alpha, loc=loc, scale=scale)
        if "Gate" not in df.columns:
            return cdf, np.zeros_like(y)

        # SkewNormal yardage cells above the historical-zero threshold are trained on
        # positive rows and served as gate·delta_0 + (1-gate)·SkewNormal.  For y >= 0 this
        # is exactly the CDF used by helpers.get_odds; below zero the point mass has not
        # occurred yet.  Randomizing the atom at y == 0 is required for an exact PIT.
        gate = np.clip(df["Gate"].to_numpy(dtype=float), 0.0, 1.0)
        cdf = np.where(y < 0.0, (1.0 - gate) * cdf, gate + (1.0 - gate) * cdf)
        pmf = np.where(np.isclose(y, 0.0), gate, 0.0)
        return cdf, pmf
    if dist == "Mixture":
        return _mix_cdf(y, *_decode_mix_params(df, strategy)), np.zeros_like(y)
    if dist in ("NegBin", "ZINB"):
        r = df["R"].to_numpy(dtype=float)
        p = df["NB_P"].to_numpy(dtype=float)
        cdf = _scipy_nbinom.cdf(y, r, 1.0 - p)
        pmf = _scipy_nbinom.pmf(y, r, 1.0 - p)
        if dist == "ZINB":
            gate = df["Gate"].to_numpy(dtype=float)
            cdf = gate + (1.0 - gate) * cdf
            pmf = np.where(y == 0, gate + (1.0 - gate) * pmf, (1.0 - gate) * pmf)
        return cdf, pmf
    if dist == "DPO":
        mu = df["DP_MU"].to_numpy(dtype=float)
        phi = df["DP_PHI"].to_numpy(dtype=float)
        return _dp_cdf_pmf(y, mu, phi)
    if dist in ("Gamma", "ZAGamma"):
        a = df["Alpha"].to_numpy(dtype=float)
        ev = df["EV"].to_numpy(dtype=float)
        cdf = _scipy_gamma.cdf(y, a, scale=ev / np.maximum(a, 1e-12))
        if dist == "ZAGamma":
            gate = df["Gate"].to_numpy(dtype=float)
            cdf = np.where(y == 0, gate, gate + (1.0 - gate) * cdf)
            return cdf, np.where(y == 0, gate, 0.0)
        return cdf, np.zeros_like(y)
    raise ValueError(f"Unknown distribution family for PIT: {dist!r}")


def _pred_midpit(df: pd.DataFrame, dist: str, y: np.ndarray, *, strategy: str) -> np.ndarray:
    """Per-row mid-PIT ``F(y) − ½·P(Y=y)`` of the actual outcome ``y``.

    Continuous families collapse to the ordinary PIT ``F(y)``; the count /
    zero-inflated families subtract half the realized point mass (Czado-Gneiting-Held
    2009). Reported alongside the coverage as a diagnostic; the gate statistic is the
    randomized PIT (:func:`_randomized_pit_draws`), which is the lattice-exact version.
    """
    cdf, pmf = _pred_cdf_pmf(df, dist, y, strategy=strategy)
    return cdf - 0.5 * pmf


def _ks_uniform(values: np.ndarray) -> float:
    ordered = np.sort(np.clip(values, 0.0, 1.0))
    if len(ordered) == 0:
        return float("nan")
    return ks_supremum(ordered)


def _tail_ks_uniform(values: np.ndarray, floor: float = _TAIL_PIT_FLOOR) -> float:
    """KS distance from Uniform restricted to the over-tail ``u >= floor`` of the PIT.

    The empirical CDF is the global one (the sub-``floor`` mass still counts toward each
    tail point's rank); only the supremum is taken over the over-tail. Reads as the
    worst-case alt-OVER mispricing — the region where boosted parlay legs live, which the
    whole-CDF Gate-4 KS can net to zero.
    """
    u = np.sort(np.clip(values, 0.0, 1.0))
    n = len(u)
    if n == 0:
        return float("nan")
    mask = u >= floor
    if not np.any(mask):
        return 0.0
    idx = np.arange(1, n + 1)
    d_plus = float(np.max(idx[mask] / n - u[mask]))
    d_minus = float(np.max(u[mask] - (idx[mask] - 1) / n))
    return max(d_plus, d_minus)


def _two_part_contract(
    df: pd.DataFrame,
) -> tuple[Mapping[str, object], np.ndarray, np.ndarray, np.ndarray] | None:
    """Validate and decode the two-part row contract when it is present."""
    # style: allow-complexity — flat row-contract validator; each branch is one
    # distinct contract rule, so splitting would only scatter the checks.
    identity, _ = validate_strategy_frame(df)
    if identity is None or (identity.structural_strategy != TWO_PART_STRATEGY):
        return None

    missing = _TWO_PART_CONTRACT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"two-part scorecard contract missing columns: {sorted(missing)}")
    calibration = df[_STRUCTURAL_CALIBRATION_COL]
    if calibration.isna().any() or calibration.astype(str).nunique() != 1:
        raise ValueError("StructuralCalibration must be one constant nonmissing JSON value")
    blob = deserialize_two_part_calibration(str(calibration.iloc[0]))

    roles = df[_STRUCTURAL_ROLE_COL].to_numpy()
    if (
        not all(isinstance(value, str) for value in roles)
        or not np.isin(roles, ("low", "high")).all()
    ):
        raise ValueError("StructuralRole must contain only nonmissing low/high strings")

    raw_positions = pd.to_numeric(df[_STRUCTURAL_POSITION_COL], errors="coerce").to_numpy(
        dtype=float
    )
    positions = raw_positions.astype(int, casting="unsafe", copy=False)
    if not np.isfinite(raw_positions).all() or not np.array_equal(raw_positions, positions):
        raise ValueError("StructuralPosition must contain integer league roster codes")

    f0 = pd.to_numeric(df[_STRUCTURAL_F0_COL], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(f0).all() or np.any((f0 < 0.0) | (f0 > 1.0)):
        raise ValueError("StructuralF0 must contain finite probabilities in [0, 1]")

    for column in ("SN_Loc", "SN_Scale", "SN_Alpha", "P", "P_PrePool", "Line"):
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"two-part {column} must be finite on every row")
        if column == "SN_Scale" and np.any(values <= 0.0):
            raise ValueError("two-part SN_Scale must be strictly positive")
        if column in {"P", "P_PrePool"} and np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"two-part {column} must lie in [0, 1]")
    if "Gate" in df.columns:
        gate = pd.to_numeric(df["Gate"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(gate).all() or np.any((gate < 0.0) | (gate >= 1.0)):
            raise ValueError("two-part Gate must be finite and lie in [0, 1)")
    return blob, f0, roles.astype(str), positions


def _two_part_cdf_endpoints(
    df: pd.DataFrame,
    dist: str,
    y: np.ndarray,
    *,
    strategy: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Rebuild and transform the two outcome CDF endpoints for the two-part strategy."""
    contract = _two_part_contract(df)
    if contract is None:
        return None
    if dist != "SkewNormal":
        raise ValueError("two-part scorecard contract requires SkewNormal row parameters")
    if "PITRecalKnots" in df.columns:
        raise ValueError("two-part must use StructuralCalibration, not PITRecalKnots")

    blob, persisted_f0, roles, positions = contract
    raw_f0, _ = _pred_cdf_pmf(
        df,
        dist,
        np.zeros(len(df), dtype=float),
        strategy=strategy,
    )
    if not np.allclose(persisted_f0, raw_f0, rtol=1e-12, atol=1e-12):
        raise ValueError("StructuralF0 does not match the persisted served SkewNormal shape")

    raw_upper, raw_mass = _pred_cdf_pmf(df, dist, y, strategy=strategy)
    raw_lower = raw_upper - raw_mass
    return two_part_cdf_endpoints(
        blob,
        raw_lower,
        raw_upper,
        persisted_f0,
        roles,
        positions,
    )


def _apply_pit_recal_by_row(df: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    """Apply either one cell-level PIT map or auditable row-routed maps."""
    array = np.asarray(values, dtype=float)
    if "PITRecalKnots" not in df.columns:
        return array
    payloads = df["PITRecalKnots"]
    if payloads.isna().any():
        raise ValueError("PITRecalKnots must be present on every scored row")
    out = np.empty_like(array)
    payload_strings = payloads.astype(str)
    for payload in payload_strings.unique():
        mask = payload_strings.eq(payload).to_numpy()
        blob = json.loads(payload)
        out[mask] = apply_cdf_recal(blob, array[mask])
    return out


def _randomized_pit_draws(
    df: pd.DataFrame, dist: str, y: np.ndarray, *, strategy: str
) -> list[np.ndarray]:
    """Seeded randomized-PIT samples used by Gate 4 and its over-tail diagnostic.

    Each integer's probability jump is spread by ``V ~ U(0,1)``:
    ``F(y−1) + V·P(Y=y) = F(y) − (1−V)·P(Y=y)``, exactly Uniform(0, 1) under calibration
    for any discrete family (Brockwell 2007). Continuous families have ``P(Y=y)=0`` so a
    single deterministic PIT ``F(y)`` is returned; the count families return
    :data:`RANDOMIZED_PIT_DRAWS` seeded draws so the statistics built on them stay
    reproducible.
    """
    identity, _ = validate_strategy_frame(df)
    structural_strategy = (
        identity.structural_strategy if identity is not None else BASE_STRUCTURAL_STRATEGY
    )
    if structural_strategy == TWO_PART_STRATEGY:
        receiving_endpoints = _two_part_cdf_endpoints(df, dist, y, strategy=strategy)
        assert receiving_endpoints is not None
        lower, upper = receiving_endpoints
        rng = np.random.default_rng(RANDOMIZED_PIT_SEED)
        return [
            lower + rng.random(len(lower)) * (upper - lower) for _ in range(RANDOMIZED_PIT_DRAWS)
        ]
    if structural_strategy not in {
        BASE_STRUCTURAL_STRATEGY,
        AFFINE_STRATEGY,
    }:
        raise ValueError(f"no scorecard adapter for structural strategy {structural_strategy!r}")

    cdf, pmf = _pred_cdf_pmf(df, dist, y, strategy=strategy)
    # §6.1 Rung C: warp each PIT draw through the served whole-CDF map g (identity unless the
    # cell persisted a PITRecalKnots column), so Gate 4 scores the recalibrated predictive.
    upper = _apply_pit_recal_by_row(df, cdf)
    if not np.any(pmf > 0):
        return [upper]
    # For F*=g∘F, map the two sides of a discrete jump before randomizing it:
    # F*(y-) + V[P*(Y=y)] = g(F(y)-p) + V[g(F(y))-g(F(y)-p)].  Applying g after
    # randomization is not equivalent when g is nonlinear across the atom.
    lower = _apply_pit_recal_by_row(df, cdf - pmf)
    rng = np.random.default_rng(RANDOMIZED_PIT_SEED)
    n = len(cdf)
    return [lower + rng.random(n) * (upper - lower) for _ in range(RANDOMIZED_PIT_DRAWS)]


def _randomized_pit_ks(df: pd.DataFrame, dist: str, y: np.ndarray, *, strategy: str) -> float:
    """Whole-CDF KS distance from Uniform of the randomized PIT — the Gate-4 statistic.

    One KS threshold spans continuous and count cells because the randomized PIT (see
    :func:`_randomized_pit_draws`) is exactly Uniform under calibration for either; the
    non-randomized mid-PIT would fail well-calibrated count cells on discreteness alone.
    Averaged over the seeded draws so the gate is reproducible.
    """
    draws = _randomized_pit_draws(df, dist, y, strategy=strategy)
    return float(np.mean([_ks_uniform(u) for u in draws]))


def _served_sn_pit_ks(
    mean: np.ndarray,
    sigma: np.ndarray,
    skew: np.ndarray,
    y: np.ndarray,
    c: float,
    s: float,
    gate: np.ndarray | float | None = None,
) -> float:
    """Gate-4 randomized-PIT KS of the served SkewNormal under scale ``c`` and skew shift ``s``.

    Holds the (blended) mean fixed and re-derives ``loc`` per row through the shared
    :func:`helpers.distributions.skewnormal_loc_from_mean` — the same formula the betting
    path uses — so the fit optimizes the exact gate statistic the test-set CSV is later
    scored against; no re-derived PIT math. ``c`` scales the scale, ``s`` shifts the shape
    (``alpha + s``). The Lever-1 and Lever-4a fits both minimize this.
    """
    scale = sigma * c
    alpha = skew + s
    df = pd.DataFrame(
        {
            "SN_Loc": skewnormal_loc_from_mean(mean, scale, alpha),
            "SN_Scale": scale,
            "SN_Alpha": alpha,
        }
    )
    if gate is not None:
        df["Gate"] = gate
    return _randomized_pit_ks(df, "SkewNormal", y, strategy=TARGET_NORM_NONE)


def fit_skewnorm_dispersion_c(
    mean: np.ndarray,
    sigma: np.ndarray,
    skew: np.ndarray,
    y: np.ndarray,
    *,
    bounds: tuple[float, float] = _DISPERSION_C_BOUNDS,
    gate: np.ndarray | float | None = None,
) -> float:
    """Fit the scale multiplier ``c`` that minimizes the Gate-4 randomized-PIT KS (Lever 1).

    The served SkewNormal predictive holds the (blended) mean fixed and scales the scale by
    ``c`` (shape left untouched). See :func:`_served_sn_pit_ks` for the shared objective.
    """
    mean = np.asarray(mean, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    skew = np.asarray(skew, dtype=float)
    y = np.asarray(y, dtype=float)

    return float(
        minimize_scalar(
            lambda c: _served_sn_pit_ks(mean, sigma, skew, y, c, 0.0, gate),
            bounds=bounds,
            method="bounded",
        ).x
    )


def fit_skewnorm_dispersion_skew(
    mean: np.ndarray,
    sigma: np.ndarray,
    skew: np.ndarray,
    y: np.ndarray,
    *,
    bounds: tuple[float, float] = _DISPERSION_C_BOUNDS,
    skew_bounds: tuple[float, float] = _DISPERSION_SKEW_BOUNDS,
    joint: bool = True,
    gate: np.ndarray | float | None = None,
) -> tuple[float, float]:
    """Fit the scale ``c`` and additive skew shift ``s`` minimizing the Gate-4 KS (Lever 4a).

    The served SkewNormal collapses to ``alpha ≈ 0`` (the Fisher-information singularity at the
    symmetric point — Hallin & Ley 2014, *Bernoulli* 20(3)), so a multiplicative skew adjustment
    injects nothing; the shift is additive, ``alpha + s``.

    Two fit orders, selected by ``joint``:

    * ``joint=True`` (default) — ``c`` and ``s`` are optimized *together* by multi-start
      Nelder-Mead. They are coupled (the scale-only optimum widens once skew is admitted), and the
      objective is flat at ``s = 0``, so the fit is restarted across :data:`_DISPERSION_SKEW_STARTS`
      and the arg-min taken. The scale is free to move off its scale-only value.
    * ``joint=False`` — *sequential* "dispersion then skew": the Lever-1 scale ``c`` is fit and
      frozen, then ``s`` is fit alone on top by a 1-D bounded search. The returned ``c`` is exactly
      the scale-only optimum.

    Both warm-start from the scale-only ``c`` and apply the same opt-in margin: a calibrated
    ``(c, s)`` that fails to beat scale-only by at least :data:`_DISPERSION_SKEW_MIN_GAIN` returns
    ``(c0, 0.0)`` — the pure Lever-1 result. ``s = 0`` recovers scale-only either way.
    """
    mean = np.asarray(mean, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    skew = np.asarray(skew, dtype=float)
    y = np.asarray(y, dtype=float)
    c0 = fit_skewnorm_dispersion_c(mean, sigma, skew, y, bounds=bounds, gate=gate)
    ks_scale_only = _served_sn_pit_ks(mean, sigma, skew, y, c0, 0.0, gate)

    if joint:

        def _ks(params: np.ndarray) -> float:
            c, s = params
            if not (bounds[0] <= c <= bounds[1] and skew_bounds[0] <= s <= skew_bounds[1]):
                return 1.0
            return _served_sn_pit_ks(mean, sigma, skew, y, c, s, gate)

        best = min(
            (
                minimize(
                    _ks,
                    x0=np.array([c0, s0]),
                    method="Nelder-Mead",
                    options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 400},
                )
                for s0 in _DISPERSION_SKEW_STARTS
            ),
            key=lambda r: r.fun,
        )
        c, s = float(np.clip(best.x[0], *bounds)), float(np.clip(best.x[1], *skew_bounds))
    else:
        s_seq = minimize_scalar(
            lambda s: _served_sn_pit_ks(mean, sigma, skew, y, c0, s, gate),
            bounds=skew_bounds,
            method="bounded",
        ).x
        c, s = c0, float(np.clip(s_seq, *skew_bounds))

    ks_cal = _served_sn_pit_ks(mean, sigma, skew, y, c, s, gate)
    if ks_cal > ks_scale_only - _DISPERSION_SKEW_MIN_GAIN:
        return c0, 0.0
    return c, s


def _gate4_pit_ks_threshold(n: int) -> float:
    """Per-cell Gate-4 bound: ``max(δ, 1.358/√n)`` — the larger of the effect-size floor
    (worst-case alt-line mispricing ``δ`` ≈ the vig) and the cell's KS α=0.05 critical
    value (never fail a cell below the miscalibration its ``n`` can resolve).
    """
    if n <= 0:
        return float("nan")
    return max(_GATE4_PIT_KS_DELTA, _GATE4_KS_NOISE_COEF / float(np.sqrt(n)))


def _dispersion_diagnostics(
    df: pd.DataFrame, dist: str, actual: np.ndarray, *, strategy: str
) -> tuple[float, float, float, float]:
    """Gate-4 statistic + reported triple: ``(pit_ks, tail_pit_ks, central50, central80)``.

    ``pit_ks`` is the whole-CDF randomized-PIT KS distance from Uniform — the Gate-4 ship
    statistic (the worst-case alt-line mispricing). ``tail_pit_ks`` is the same statistic
    restricted to the over-tail (:func:`_tail_ks_uniform`): reported, not a ship term, it
    names the alt-OVER mispricing the whole-CDF sup can net away. The two central-interval
    coverages name the *direction* a KS cannot: a coverage below its nominal level
    (0.50 / 0.80) means the predictive is too narrow (under-dispersed), above means too
    wide. Coverage is nominal-lumpy on the discrete count families (integer-quantile
    endpoints), exact on the continuous SkewNormal / Mixture cells.
    """
    draws = _randomized_pit_draws(df, dist, actual, strategy=strategy)
    pit_ks = float(np.mean([_ks_uniform(u) for u in draws]))
    tail_pit_ks = float(np.mean([_tail_ks_uniform(u) for u in draws]))
    lo50 = _pred_ppf(df, dist, 0.25, strategy=strategy)
    hi50 = _pred_ppf(df, dist, 0.75, strategy=strategy)
    lo80 = _pred_ppf(df, dist, 0.10, strategy=strategy)
    hi80 = _pred_ppf(df, dist, 0.90, strategy=strategy)
    cov50 = float(np.mean((actual >= lo50) & (actual <= hi50)))
    cov80 = float(np.mean((actual >= lo80) & (actual <= hi80)))
    return pit_ks, tail_pit_ks, cov50, cov80


def _gate1_brier_ci(
    p_model: np.ndarray, p_book: np.ndarray, y: np.ndarray, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Gate 1 — paired bootstrap of the per-event (model - book) Brier difference.

    ``d_i = (p_model_i - y_i)^2 - (p_book_i - y_i)^2``. Lower Brier is better, so a
    bootstrap-mean CI entirely below 0 means the model beats the book. Returns
    ``(mean, ci_lo, ci_hi)``.
    """
    d = (p_model - y) ** 2 - (p_book - y) ** 2
    return _bootstrap_mean_ci(d, rng)


def _gate23_segment_match(
    pred: np.ndarray, actual: np.ndarray, mask: np.ndarray
) -> tuple[float, float, float, float, float]:
    """Gates 2/3 — bias-vs-spread match of predicted vs true mean on one segment.

    Returns ``(pred_mean, true_mean, abs_diff, sigma, z)`` where ``sigma =
    std(actual[mask], ddof=1)`` is the within-segment spread of true outcomes and
    ``z = abs_diff / sigma`` measures the bias in units of that spread. The
    denominator is the segment's STANDARD DEVIATION, not its standard error
    (``sigma / sqrt(N)``) — Gate 3's bench segment has hundreds-to-thousands of
    low-variance rows, so SE collapses to ~0 and the gate would fire on a negligible
    bias; sigma keeps the yardstick at "what a typical event in the segment looks
    like" regardless of N. ``sigma`` / ``z`` are ``nan`` for an empty or zero-variance
    segment.
    """
    seg_pred = pred[mask]
    seg_actual = actual[mask]
    n = len(seg_actual)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    pred_mean = float(seg_pred.mean())
    true_mean = float(seg_actual.mean())
    abs_diff = abs(pred_mean - true_mean)
    sigma = float(np.std(seg_actual, ddof=1)) if n > 1 else float("nan")
    z = abs_diff / sigma if sigma and np.isfinite(sigma) and sigma > 0 else float("nan")
    return pred_mean, true_mean, abs_diff, sigma, z


def _raw_segment_z(
    df: pd.DataFrame, actual: np.ndarray, star_mask: np.ndarray, bench_mask: np.ndarray
) -> tuple[float | None, float | None]:
    """Star/bench bias z's computed on the raw model mean ``EV`` — reported beside the
    fused ship gates.

    Gates 2/3 score the fused ``Blended_EV`` (what the parlay actually drafts); this is the
    same z on the raw ``EV``, the model-compression view book fusion masks, so a passing
    cell shows whether the model itself is uncompressed or the book is carrying it. Returns
    ``(None, None)`` for a frame without an ``EV`` column.
    """
    if "EV" not in df.columns:
        return None, None
    raw = _zero_inflated_mean(df, df["EV"].to_numpy(dtype=float))
    return (
        _gate23_segment_match(raw, actual, star_mask)[4],
        _gate23_segment_match(raw, actual, bench_mask)[4],
    )


def _gate4_iqr_spread(
    actual: np.ndarray,
    pred: np.ndarray,
    *,
    df: pd.DataFrame | None = None,
    dist: str | None = None,
    strategy: str | None = None,
) -> tuple[float, float, float]:
    """Gate 4 — IQR spread (compression) over the full population.

    When ``df`` + ``dist`` are supplied (the typical real-cell call from
    :func:`gate_row`), the predicted IQR is the **pooled analytical IQR** of
    the per-row predictive distribution — `_iqr_pred_analytical`. Without them
    (oracle row, legacy synthetic frames) the predicted IQR falls back to the
    point estimator ``_iqr(pred)`` so the oracle (`pred = actual`) keeps
    returning ``ratio = 1.0``.

    Degenerate handling (Ship 75 Step 0.4): when ``iqr_true = 0``, the
    ``iqr_pred = 0`` cell gets ``ratio = 1.0`` (perfectly-matched degenerate
    truth), but ``iqr_pred > 0`` gets ``ratio = inf`` (the gate fails — model
    spreads where truth is point-mass).
    """
    iqr_true = _iqr(actual)
    if df is not None and dist is not None:
        iqr_pred = _iqr_pred_analytical(df, dist, strategy=strategy or _DECODE_FALLBACK_STRATEGY)
    else:
        iqr_pred = _iqr(pred)
    if iqr_true == 0:  # noqa: SIM108 — doubly-nested ternary is unreadable here
        ratio = 1.0 if iqr_pred == 0 else float("inf")
    else:
        ratio = iqr_pred / iqr_true
    return iqr_pred, iqr_true, ratio


def _gate5_ece_equal_mass(p_model: np.ndarray, y: np.ndarray, n_bins: int = _ECE_BINS) -> float:
    """Gate 5 — equal-mass expected calibration error.

    Bins ``p_model`` into ``n_bins`` equal-mass bins (``pd.qcut``) and returns the
    mass-weighted ``sum_b (n_b / N) * |mean(p_model in b) - mean(y in b)|``. Equal
    MASS (not equal width as in training/pipeline.py) so the bins track where the
    predicted-probability mass actually sits. Returns ``nan`` for an empty input.
    """
    total = len(p_model)
    if total == 0:
        return float("nan")
    # duplicates="drop" collapses ties (e.g. a spike of identical probabilities, or
    # the deterministic 0/1 oracle) into fewer bins rather than raising. A fully
    # degenerate vector (all probabilities equal) leaves no valid edges -> qcut
    # returns all-NaN; fall back to a single bin over every row.
    bins = np.asarray(pd.qcut(p_model, n_bins, labels=False, duplicates="drop"), dtype=float)
    valid = ~np.isnan(bins)
    if not valid.any():
        return abs(float(p_model.mean()) - float(y.mean()))
    ece = 0.0
    for b in np.unique(bins[valid]):
        m = bins == b
        n = int(m.sum())
        ece += (n / total) * abs(float(p_model[m].mean()) - float(y[m].mean()))
    return float(ece)


def _ece_debias_offset(
    p_model: np.ndarray,
    *,
    n_resamples: int = _GATE5_DEBIAS_RESAMPLES,
    rng: np.random.Generator | None = None,
) -> float:
    """Roelofs (2022) Monte-Carlo bias correction for the equal-mass ECE.

    The binned ECE estimator is positively biased at finite N — even a
    perfectly calibrated model produces ``_gate5_ece_equal_mass > 0``
    because sampling noise in each bin's mean-``y`` inflates
    ``|mean_p - mean_y|``. Per the lifecycle gate audit at
    ``/tmp/researcher_lifecycle_gate_audit.md`` this falsely fails up to
    44.6 % of perfectly calibrated NFL-N≈240 cells.

    The fix: estimate the bias as the expected ECE under the null
    hypothesis that ``p_model`` IS the calibration, by drawing
    ``y ~ Bernoulli(p_model)`` ``n_resamples`` times and averaging the
    resulting ECEs. Subtract that offset from the raw ECE in
    :func:`_gate5_ece_debiased`. Returns ``nan`` on empty input.
    """
    if rng is None:
        rng = np.random.default_rng(_GATE5_DEBIAS_SEED)
    p = np.clip(np.asarray(p_model, dtype=float), 0.0, 1.0)
    if len(p) == 0:
        return float("nan")
    eces = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        y_null = (rng.uniform(size=len(p)) < p).astype(float)
        eces[i] = _gate5_ece_equal_mass(p, y_null)
    return float(np.nanmean(eces))


def _gate5_ece_debiased(
    p_model: np.ndarray,
    y: np.ndarray,
    *,
    n_resamples: int = _GATE5_DEBIAS_RESAMPLES,
    rng: np.random.Generator | None = None,
) -> float:
    """Raw equal-mass ECE minus the Roelofs (2022) null-distribution offset.

    ``raw_ECE - mean_null_ECE`` removes the binning-noise floor while
    preserving genuine miscalibration signal. Falls back to the raw ECE
    if either term is non-finite (degenerate inputs).
    """
    raw = _gate5_ece_equal_mass(p_model, y)
    if not np.isfinite(raw):
        return raw
    offset = _ece_debias_offset(p_model, n_resamples=n_resamples, rng=rng)
    if not np.isfinite(offset):
        return raw
    return float(raw - offset)


# Scatter plot output settings. DPI and figure size are presentation choices,
# not model parameters; named here so they are easy to find and adjust.
_SCATTER_DPI: int = 110
_SCATTER_FIG_INCHES: tuple[int, int] = (7, 7)

# Synthetic base date for the supersede S3 Kelly sim. The sim needs monotonic
# per-event dates; using a fixed epoch keeps the date series deterministic and
# independent of wall-clock time. Any fixed date works — only ordering matters.
_SUPERSEDE_S3_BASE_DATE: pd.Timestamp = pd.Timestamp("2026-01-01")

_DECODE_FALLBACK_STRATEGY: str = "ratio_meanyr"


def _round_gate_value(v: float | None) -> float | None:
    """Round to 4 dp; map None / non-finite to a blank CSV cell."""
    if v is None or not np.isfinite(v):
        return None
    return round(float(v), 4)


@functools.lru_cache(maxsize=1)
def _cached_stat_meta() -> dict:
    """Memoize raw stat_meta so a full-audit loop hits disk once."""
    return load_stat_meta(Path(str(STAT_META_PATH)))


def _resolve_decode_strategy(league: str, market_stem: str) -> str:
    """Look up the per-cell training strategy for a SkewNormal decode mirror.

    ``market_stem`` is the file-slug form (e.g. ``fantasy-points-prizepicks``)
    that lives in the test_set CSV name; ``stat_meta.json`` keys are the raw
    market names with spaces, so we reverse the hyphenation done by
    :func:`helpers.io.market_file_slug`.

    Reads the strategy straight from ``stat_meta.json`` rather than the
    ship-config projection: :func:`load_ship_config` collapses every withheld
    cell to the ``WITHHELD`` sentinel, which hides the ``ratio_meanyr``
    transform the cell actually trained under and leaves the g4 IQR decode in
    normalized ratio-units. A ``none`` value means the cell trained under the
    historic ``ratio_meanyr`` fallback (see
    ``ship_config.resolve_flag_target_normalization``).
    """
    market_with_spaces = market_stem.replace("-", " ")
    target_norm = (
        _cached_stat_meta()
        .get(league, {})
        .get(market_with_spaces, {})
        .get("target_normalization", TARGET_NORM_NONE)
    )
    return _DECODE_FALLBACK_STRATEGY if target_norm == TARGET_NORM_NONE else target_norm


def _signed_decode_strategy(df: pd.DataFrame) -> str | None:
    identity, _ = validate_strategy_frame(df)
    if identity is None:
        return None
    return parse_controls(identity.controls_json).get("normalization", TARGET_NORM_NONE)


def _decode_strategy_for_frame(df: pd.DataFrame, league: str = "", market: str = "") -> str | None:
    signed = _signed_decode_strategy(df)
    if signed is not None:
        return signed
    if league and market:
        return _resolve_decode_strategy(league, market.replace(" ", "-"))
    return None


def _zero_inflated_mean(df: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    """Recover E[Y] = (1 - π)·μ for the bias gates on zero-inflated count cells.

    ZINB/ZAGamma and gated SkewNormal store the BASE-distribution mean in ``EV``: the betting path factors
    the zero-inflation gate out of EV and reapplies it only when pricing over/under
    probabilities (``get_odds``). Gates 2/3 compare the predicted mean against the
    zero-INCLUSIVE empirical mean, so they must reapply the gate too — otherwise the
    prediction is overstated by ``1/(1-π)``, inflating the bias most where the gate is
    large (bench players; star goal-line backs). An ungated SkewNormal has no ``Gate``
    column and therefore remains unchanged.
    """
    if _infer_dist_from_columns(df) in ("ZINB", "ZAGamma", "SkewNormal") and "Gate" in df:
        return pred * (1.0 - df["Gate"].to_numpy(dtype=float))
    return pred


def _gate6_anchored(corr: float, prior_g6_fired: bool | None) -> bool:
    if not np.isfinite(corr):
        return False
    if corr >= _GATE6_FIRE_ON:
        return True
    return corr >= _GATE6_KEEP_ON and bool(prior_g6_fired)


_GATE6_BLANK_LEGS: dict[str, float | None] = {
    "g6_recent_corr": None,
    "g6_star_ratio": None,
    "g6_star_ci_hi": None,
    "g6_star_ref": None,
    "g6_citl_ratio": None,
    "g6_citl_ci_hi": None,
    "g6_over_ratio": None,
    "g6_over_ci_lo": None,
}


def _gate6_legs(
    df: pd.DataFrame,
    pred_col: str,
    *,
    league: str,
    prior_g6_fired: bool | None = None,
) -> dict[str, float | None]:
    """Gate 6 (anti-shrinkage): three one-sided legs on the stable top/bottom-MeanYr segments, each
    reading the served ``pred_col`` (so all are normalization- and family-agnostic):

    * **recent-form** (star vs ``Mean10``): the original leg, gated by the corr anchor with
      hysteresis — catches the ``ratio_meanyr`` holdout-corruption class CITL is blind to.
    * **CITL-under** (star vs ``Result``): calibration-in-the-large, run on *every* cell (the
      outcome is a valid yardstick without the anchor) — catches outcome under-shrinkage that
      Gate 2's σ-normalization launders.
    * **over** (bench vs ``Result``): count/ZINB bench over-prediction, guarded by a realized
      segment-mean floor so it can't fire on degenerate rare counts.

    Returns the ``g6_*`` measurement subset; a leg's keys are ``None`` where it can't or shouldn't
    test (the gate auto-passes a blank leg).
    """
    if not {"Mean10", "Player"}.issubset(df.columns):
        return dict(_GATE6_BLANK_LEGS)
    work = df[(df[DECILE_COL] > 0) & (df["Mean10"] > 0)]
    meanyr = work[DECILE_COL].to_numpy()
    mean10 = work["Mean10"].to_numpy()
    result = work[ACTUAL_COL].to_numpy()
    pred = _zero_inflated_mean(work, work[pred_col].to_numpy(dtype=float))
    players = work["Player"].to_numpy()
    rng = np.random.default_rng(_GATE1_SEED)
    out = dict(_GATE6_BLANK_LEGS)

    corr = _corr(mean10, result)
    out["g6_recent_corr"] = corr if np.isfinite(corr) else None

    stable = np.abs(mean10 / meanyr - 1.0) <= _GATE6_STABLE_BAND
    star = stable & (meanyr >= np.quantile(meanyr, 1.0 - BOTTOM_QUARTILE_FRAC))
    if int(star.sum()) >= _GATE6_MIN_STAR_ROWS:
        citl, _, citl_hi = _bootstrap_ratio_ci_clustered(
            pred[star], result[star], players[star], rng
        )
        out["g6_citl_ratio"], out["g6_citl_ci_hi"] = citl, citl_hi
        if _gate6_anchored(corr, prior_g6_fired):
            ratio, _, ratio_hi = _bootstrap_ratio_ci_clustered(
                pred[star], mean10[star], players[star], rng
            )
            out["g6_star_ratio"], out["g6_star_ci_hi"] = ratio, ratio_hi
            out["g6_star_ref"] = (
                _GATE6_STAR_REF_NFL if league == "NFL" else _GATE6_STAR_REF_BASKETBALL
            )

    if _infer_dist_from_columns(df) in ("NegBin", "ZINB", "DPO"):
        bench = stable & (meanyr <= np.quantile(meanyr, BOTTOM_QUARTILE_FRAC))
        if (
            int(bench.sum()) >= _GATE6_MIN_STAR_ROWS
            and result[bench].mean() >= _GATE6_OVER_MIN_MEAN
        ):
            over, over_lo, _ = _bootstrap_ratio_ci_clustered(
                pred[bench], result[bench], players[bench], rng
            )
            out["g6_over_ratio"], out["g6_over_ci_lo"] = over, over_lo
    return out


def gate_row(
    df: pd.DataFrame,
    pred_col: str,
    *,
    league: str,
    market: str,
    strategy: str,
    decode_strategy: str | None = None,
    prior_g6_fired: bool | None = None,
) -> dict[str, object]:
    """Compute the offline ship gates for one cell — a model row plus an oracle row.

    The oracle assumes the model predicted the true score exactly (``pred = Result``;
    over-probability ``1 if Result>=Line else 0``), giving each gate's idealistic
    bound: Gate 1 diff = -book Brier, Gates 2/3 ``z = 0``, Gate 4 ratio ``1.0``, Gate 5
    ``ece = 0``. The σ / IQR_true denominators equal the model row, so the oracle
    columns size each gate's natural threshold. Measurement-only — no pass/fail. Gate
    1 is **blank when ``Odds`` is missing** (no book to beat); the ship convention is
    that a blank Gate 1 **auto-passes** — model wins by default. Gate 5 needs only
    ``P`` + ``Line`` (not ``Odds``), so it still computes for book-unpriced cells; a
    blank Gate 5 means "couldn't compute" (no P or no Line), NOT auto-pass.
    """
    identity, _ = validate_strategy_frame(df)
    if identity is not None:
        same_league = not league or league == identity.league
        same_market = not market or market.replace(" ", "-") == identity.market.replace(" ", "-")
        if not (same_league and same_market):
            raise ValueError("scorecard cell does not match the generic model-strategy identity")
        league = league or identity.league
        market = market or identity.market
    actual = df[ACTUAL_COL].to_numpy()
    pred = df[pred_col].to_numpy()
    bias_pred = _zero_inflated_mean(df, pred)
    star_mask, bench_mask = _segment_masks(df)

    # Gates 2/3 — model on the fused mean (pred_col is Blended_EV in production); the
    # oracle (pred = actual) zeroes abs_diff / z, sigma unchanged. The raw-EV view rides
    # along as a reported diagnostic so the model-compression signal isn't masked by fusion.
    g2_pred, g2_true, g2_abs, g2_sigma, g2_z = _gate23_segment_match(bias_pred, actual, star_mask)
    _, _, _, _, g2_z_oracle = _gate23_segment_match(actual, actual, star_mask)
    g3_pred, g3_true, g3_abs, g3_sigma, g3_z = _gate23_segment_match(bias_pred, actual, bench_mask)
    _, _, _, _, g3_z_oracle = _gate23_segment_match(actual, actual, bench_mask)
    g2_z_raw, g3_z_raw = _raw_segment_z(df, actual, star_mask, bench_mask)

    # Gate 4 — randomized-PIT KS-uniformity of the predictive CDF (whole-distribution
    # calibration; the worst-case alt-line mispricing), with its per-cell threshold. The
    # IQR-ratio + central-interval coverage ride along as reported diagnostics — coverage
    # names the direction (under/over-dispersion) the KS scalar can't. All route through the
    # per-row distribution params, so they're blank on legacy/synthetic frames that lack
    # them. ``decode_strategy`` is the persisted training normalization (signed CSV
    # controls, or stat_meta for an identity-absent legacy frame); ``strategy`` is only
    # the run label kept for the row.
    g4_dist = _infer_dist_from_columns(df)
    decode_for_g4 = decode_strategy or strategy
    g_pit_ks = g_tail_pit_ks = g_cov50 = g_cov80 = None
    if g4_dist is not None:
        g4_iqr_pred, g4_iqr_true, g4_ratio = _gate4_iqr_spread(
            actual, pred, df=df, dist=g4_dist, strategy=decode_for_g4
        )
        g_pit_ks, g_tail_pit_ks, g_cov50, g_cov80 = _dispersion_diagnostics(
            df, g4_dist, actual, strategy=decode_for_g4
        )
    else:
        g4_iqr_pred, g4_iqr_true, g4_ratio = _gate4_iqr_spread(actual, pred)
    g4_pit_ks_max = _gate4_pit_ks_threshold(len(df))
    _, _, g4_ratio_oracle = _gate4_iqr_spread(actual, actual)

    # Gate 1 — paired Brier vs book. Needs Odds; blank ⇒ "no book to beat, model wins
    # by default" (the auto-pass convention is doc'd at the module-header and applied
    # at verdict-wiring time). Oracle p_model = y (the deterministic 1/0 prediction).
    brier_in = _brier_inputs(df)
    if brier_in is None:
        g1_mean = g1_lo = g1_hi = g1_mean_o = g1_lo_o = g1_hi_o = bss = None
        g1_clustered_hi = g1_standalone_hi = None
    else:
        p_model_b, p_book, y_b, priced_index = brier_in
        g1_mean, g1_lo, g1_hi = _gate1_brier_ci(
            p_model_b, p_book, y_b, np.random.default_rng(_GATE1_SEED)
        )
        g1_mean_o, g1_lo_o, g1_hi_o = _gate1_brier_ci(
            y_b, p_book, y_b, np.random.default_rng(_GATE1_SEED)
        )
        bss = _brier_skill_score(df)
        g1_clustered_hi = None
        if "Player" in df.columns:
            g1_clustered_hi = _gate1_brier_ci_clustered(
                p_model_b,
                p_book,
                y_b,
                df.loc[priced_index, "Player"].to_numpy(),
                np.random.default_rng(_GATE1_SEED),
            )[2]
        g1_standalone_hi = _standalone_g1_hi(df, p_book, y_b, priced_index)

    # Gate 5 — model-only calibration. Needs P + Line (NOT Odds) — Gate 5 checks the
    # model's probabilities against outcomes; the book doesn't enter. Blank only if
    # P or Line is missing entirely; that's "couldn't compute", NOT auto-pass.
    cal_in = _calibration_inputs(df)
    if cal_in is None:
        g5_ece = g5_ece_o = g5_ece_db = g5_ece_db_o = g5_ece_bias = None
    else:
        p_model_c, y_c = cal_in
        g5_ece = _gate5_ece_equal_mass(p_model_c, y_c)
        g5_ece_o = _gate5_ece_equal_mass(y_c, y_c)
        # Roelofs (2022) bias-correction. Per the lifecycle gate audit the
        # raw equal-mass ECE falsely fails ~45% of perfectly calibrated
        # NFL-N≈240 cells; the debiased variant is the one apply_thresholds
        # actually checks. The model-row offset uses the model's own
        # probabilities; the oracle row uses the deterministic 0/1 prediction
        # ``p = y`` and so its null offset is structurally tiny.
        g5_ece_bias = _ece_debias_offset(p_model_c)
        g5_ece_db = (
            float(g5_ece - g5_ece_bias)
            if np.isfinite(g5_ece) and np.isfinite(g5_ece_bias)
            else g5_ece
        )
        oracle_bias = _ece_debias_offset(y_c)
        g5_ece_db_o = (
            float(g5_ece_o - oracle_bias)
            if np.isfinite(g5_ece_o) and np.isfinite(oracle_bias)
            else g5_ece_o
        )

    g6 = _gate6_legs(df, pred_col, league=league, prior_g6_fired=prior_g6_fired)

    r = _round_gate_value
    return {
        "league": league,
        "market": market,
        "strategy": strategy,
        "n_rows": len(df),
        "g1_brier_diff_mean": r(g1_mean),
        "g1_brier_diff_ci_lo": r(g1_lo),
        "g1_brier_diff_ci_hi": r(g1_hi),
        "g1_brier_diff_mean_oracle": r(g1_mean_o),
        "g1_brier_diff_ci_lo_oracle": r(g1_lo_o),
        "g1_brier_diff_ci_hi_oracle": r(g1_hi_o),
        "g1_clustered_ci_hi": r(g1_clustered_hi),
        "g1_brier_diff_ci_hi_standalone": r(g1_standalone_hi),
        "g1_brier_skill_score": r(bss),
        "g2_star_pred_mean": r(g2_pred),
        "g2_star_true_mean": r(g2_true),
        "g2_star_abs_diff": r(g2_abs),
        "g2_star_sigma": r(g2_sigma),
        "g2_star_z": r(g2_z),
        "g2_star_z_oracle": r(g2_z_oracle),
        "g2_star_z_raw": r(g2_z_raw),
        "g3_bench_pred_mean": r(g3_pred),
        "g3_bench_true_mean": r(g3_true),
        "g3_bench_abs_diff": r(g3_abs),
        "g3_bench_sigma": r(g3_sigma),
        "g3_bench_z": r(g3_z),
        "g3_bench_z_oracle": r(g3_z_oracle),
        "g3_bench_z_raw": r(g3_z_raw),
        "g4_pit_ks": r(g_pit_ks),
        "g4_pit_ks_max": r(g4_pit_ks_max),
        "g4_tail_pit_ks": r(g_tail_pit_ks),
        "g4_iqr_pred": r(g4_iqr_pred),
        "g4_iqr_true": r(g4_iqr_true),
        "g4_iqr_ratio": r(g4_ratio),
        "g4_iqr_ratio_oracle": r(g4_ratio_oracle),
        "central50_coverage": r(g_cov50),
        "central80_coverage": r(g_cov80),
        "g5_ece": r(g5_ece),
        "g5_ece_oracle": r(g5_ece_o),
        "g5_ece_null_bias": r(g5_ece_bias),
        "g5_ece_debiased": r(g5_ece_db),
        "g5_ece_debiased_oracle": r(g5_ece_db_o),
        **{k: r(v) for k, v in g6.items()},
    }


def _g1_within_tie_margin(hi: float | None) -> bool:
    """Gate-1 ship test (non-inferiority): the paired-Brier CI upper bound sits below
    the statistical-tie margin :data:`_GATE1_NONINF_MARGIN` — i.e. we are 95%
    confident the fused ensemble's Brier is at most ``δ`` worse than the book's. A
    tight tie or a win passes; a wildly-worse or underpowered (wide-CI) cell fails. A
    blank bound (no ``Odds``) auto-passes — there is no book to beat.
    """
    return hi is None or hi < _GATE1_NONINF_MARGIN


def _below_zero_ci_bound(hi: float | None) -> bool:
    """Reported ``g1_has_edge`` flag: the CI upper bound is below 0 (the model
    *provably* beats the book), not merely within the tie margin. Non-decisive — the
    ship gate is :func:`_g1_within_tie_margin`.

    ``hi`` is stored rounded to 4 dp and round() keeps the sign bit, so a
    genuinely-negative bound in (-5e-5, 0) — e.g. receiving-yards' -0.00004 — lands on
    -0.0, where a plain ``-0.0 < 0.0`` is False. A negative-signed zero still beat the
    book, so treat it as below the bound. A blank bound (no ``Odds``) is True — no
    book to beat.
    """
    if hi is None:
        return True
    return hi < _GATE1_CI_HI_MAX or (hi == 0.0 and bool(np.signbit(hi)))


def _gate6_passes(row: Mapping[str, object]) -> bool:
    """Gate-6 ship test: pass iff none of the three one-sided legs fires. Each leg auto-passes when
    blank (not applicable / untestable) — the deliberate blank-is-pass, unlike g2-g5.

    * recent-form: ``star_ci_hi`` (CI UB of ``Σpred/ΣMean10``) at/above ``star_ref − margin``.
    * CITL-under: ``citl_ci_hi`` (UB of ``Σpred/ΣResult``) at/above ``1 − margin``.
    * over: ``over_ci_lo`` (LB of bench ``Σpred/ΣResult``) at/below ``1 + margin``.
    """
    star_hi, star_ref = row.get("g6_star_ci_hi"), row.get("g6_star_ref")
    citl_hi = row.get("g6_citl_ci_hi")
    over_lo = row.get("g6_over_ci_lo")
    recent_ok = star_hi is None or star_ref is None or star_hi >= star_ref - _GATE6_MARGIN
    citl_ok = citl_hi is None or citl_hi >= 1.0 - _GATE6_MARGIN
    over_ok = over_lo is None or over_lo <= 1.0 + _GATE6_MARGIN
    return recent_ok and citl_ok and over_ok


def recent_form_fired(row: Mapping[str, object]) -> bool:
    """Whether the recent-form leg flagged this cell on the row's run. Seeds Gate-6 hysteresis from
    the prior ``model_stats`` row using ``g6_star_ci_hi``/``g6_star_ref`` (columns that predate this
    rework, so the first run after it lands is still seeded — cold-start safe).
    """
    hi, ref = row.get("g6_star_ci_hi"), row.get("g6_star_ref")
    return hi is not None and ref is not None and hi < ref - _GATE6_MARGIN


def apply_thresholds(row: dict[str, object]) -> dict[str, object]:
    """Augment a :func:`gate_row` row with per-gate ``*_pass`` flags + overall ``ship``.

    Gate 1 is a non-inferiority test (:data:`_GATE1_NONINF_MARGIN`); ``g1_has_edge``
    reports the stricter provable-superiority result without gating on it. Blank-cell
    semantics — distinct because the gates fail for different structural reasons:

    * Gate 1 blank (no ``Odds``): **auto-pass** — no book to beat, model wins by
      default. The only "no book data" auto-pass.
    * Gate 2/3/5 blank: **fail** — the cell couldn't compute the gate (e.g. missing
      ``P`` / ``Line``), and we don't credit the model for absence of evidence.
    * Gate 4 blank (no per-row distribution params ⇒ no ``g4_pit_ks``): **fail** — the
      predictive-shape gate is undefined without the persisted distribution, and we don't
      credit the model for absence of evidence.
    """
    out = dict(row)
    g1_pass = _g1_within_tie_margin(out.get("g1_brier_diff_ci_hi"))
    g2 = out.get("g2_star_z")
    g2_pass = g2 is not None and g2 < _GATE2_STAR_Z_MAX
    g3 = out.get("g3_bench_z")
    g3_pass = g3 is not None and g3 < _GATE3_BENCH_Z_MAX
    g4 = out.get("g4_pit_ks")
    g4_max = out.get("g4_pit_ks_max")
    g4_pass = g4 is not None and g4_max is not None and g4 < g4_max
    # Gate 5 reads the Roelofs-debiased ECE (raw - null offset). Falls back
    # to raw ECE if the debiased column is absent (synthetic golden frames).
    g5 = out.get("g5_ece_debiased", out.get("g5_ece"))
    g5_pass = g5 is not None and g5 < _GATE5_ECE_MAX
    # Gate 6 (anti-shrinkage): pass iff none of the three one-sided legs fires; each blank leg is a
    # "not applicable" auto-pass, the opposite of the g2-g5 blank-is-fail convention.
    g6_pass = _gate6_passes(out)
    out["g1_pass"] = g1_pass
    out["g1_has_edge"] = _below_zero_ci_bound(out.get("g1_brier_diff_ci_hi"))
    out["g2_pass"] = g2_pass
    out["g3_pass"] = g3_pass
    out["g4_pass"] = g4_pass
    out["g5_pass"] = g5_pass
    out["g6_pass"] = g6_pass
    out["ship"] = all((g1_pass, g2_pass, g3_pass, g4_pass, g5_pass, g6_pass))
    return out


def _normalized_gate_slack(value: float | None, threshold: float) -> float:
    """Headroom of one upper-bounded gate in units of its own threshold: ``(threshold - value) /
    threshold`` (positive ⇒ passing). A ``None`` value is a hard fail (the gate couldn't compute),
    so it returns ``-inf`` and binds the minimum.
    """
    return -np.inf if value is None else (threshold - value) / threshold


def min_gate_slack(row: dict[str, object]) -> float:
    """Single continuous ship-margin scalar: the minimum per-gate headroom across the six gates,
    each normalized to its own threshold so they compare. ``> 0`` ⇔ every gate passes with room,
    and the value is the binding (tightest) gate's fractional headroom.

    The combination search maximizes this over ``(normalization, loss, calibration-mode)``: it is
    a *ranking* signal only — the authoritative ship decision is :func:`apply_thresholds` on the
    real-HPO scorecard. A blank Gate 1 (no book) or Gate 6 (off-cohort / untestable) auto-passes
    and so does not bind; blank Gate 2/3/4/5 are hard fails (``-inf``).
    """
    hi = row.get("g1_brier_diff_ci_hi")
    g4, g4_max = row.get("g4_pit_ks"), row.get("g4_pit_ks_max")
    # Gate 6 auto-pass (blank leg) never binds (+inf). recent-form / CITL are lower-bounded (the
    # value must clear a floor, so headroom is value − floor); over is upper-bounded.
    star_hi, star_ref = row.get("g6_star_ci_hi"), row.get("g6_star_ref")
    citl_hi = row.get("g6_citl_ci_hi")
    over_lo = row.get("g6_over_ci_lo")
    return min(
        np.inf if hi is None else (_GATE1_NONINF_MARGIN - hi) / _GATE1_NONINF_MARGIN,
        _normalized_gate_slack(row.get("g2_star_z"), _GATE2_STAR_Z_MAX),
        _normalized_gate_slack(row.get("g3_bench_z"), _GATE3_BENCH_Z_MAX),
        -np.inf if g4 is None or g4_max is None else (g4_max - g4) / g4_max,
        _normalized_gate_slack(row.get("g5_ece_debiased", row.get("g5_ece")), _GATE5_ECE_MAX),
        np.inf
        if star_hi is None or star_ref is None
        else (star_hi - (star_ref - _GATE6_MARGIN)) / (star_ref - _GATE6_MARGIN),
        np.inf if citl_hi is None else (citl_hi - (1.0 - _GATE6_MARGIN)) / (1.0 - _GATE6_MARGIN),
        np.inf if over_lo is None else _normalized_gate_slack(over_lo, 1.0 + _GATE6_MARGIN),
    )


# Identity columns gate_row attaches that the inline caller already owns on
# the row it's merging into — strip them off in compute_gates so the merge
# can't fight the parent row over (league, market).
_GATE_ROW_IDENTITY_KEYS = ("league", "market", "strategy")


def compute_gates(
    test_set_df: pd.DataFrame,
    *,
    league: str,
    market: str,
    strategy: str = "meditate",
    pred_col: str = DEFAULT_PRED_COL,
    prior_g6_fired: bool | None = None,
) -> dict[str, object]:
    """Per-cell ship-gate column dict for inline use by ``training.report``.

    Wraps :func:`gate_row` + :func:`apply_thresholds`, strips the identity
    fields the parent row already owns, and renames ``n_rows`` to
    ``n_validation`` so the merge into the wide stats row uses the column
    name the parquet schema documents.

    Args:
        test_set_df: A frame produced by :func:`load_test_set`.
        league: League code (``"NBA"``, ``"NFL"``, ...).
        market: Market name in either slug (``"rushing-tds"``) or
            human-readable (``"rushing tds"``) form. Spaces are converted
            to hyphens before the per-cell training-strategy lookup.
        strategy: Run label written into the row's ``strategy`` field
            before it's stripped. Defaults to ``"meditate"`` so the inline
            caller doesn't have to think about a label that's never read.
        pred_col: Predicted-mean column to evaluate (``"Blended_EV"`` = fused ship gate; ``"EV"`` = raw model).
        prior_g6_fired: Whether the recent-form leg fired on the prior run, seeding the Gate-6
            anchor hysteresis. ``None`` (no prior / cold start) judges at the fire-on threshold.

    Returns:
        Dict carrying every gate measurement, oracle bound, per-gate
        ``g{1..5}_pass`` flag, and the overall ``ship`` verdict. Excludes
        ``league``, ``market``, ``strategy``; renames ``n_rows`` →
        ``n_validation``.
    """
    market_stem = market.replace(" ", "-")
    decode_strategy = _decode_strategy_for_frame(test_set_df, league, market_stem)
    row = apply_thresholds(
        gate_row(
            test_set_df,
            pred_col,
            league=league,
            market=market_stem,
            strategy=strategy,
            decode_strategy=decode_strategy,
            prior_g6_fired=prior_g6_fired,
        )
    )
    gate_only = {k: v for k, v in row.items() if k not in _GATE_ROW_IDENTITY_KEYS}
    if "n_rows" in gate_only:
        gate_only["n_validation"] = gate_only.pop("n_rows")
    return gate_only


def write_gate_scorecard(rows: list[dict[str, object]], out_path: Path) -> pd.DataFrame:
    """Write the per-cell five-gate scorecard snapshot to CSV, one row per cell.

    Overwrites ``out_path`` each call — a *snapshot* of the latest audit (the six
    gate metrics, the oracle bound, and the per-gate ``*_pass`` flags + overall
    ``ship``), distinct from the append-only run log. Rows are sorted by
    ``(league, market)`` for a stable git diff. Caller pre-applies thresholds via
    :func:`apply_thresholds` so the ``ship`` column is always present on rows.
    """
    df = pd.DataFrame(rows).sort_values(["league", "market"]).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    return df


def _print_breadth_rollup(scorecard_df: pd.DataFrame) -> None:
    """Print per-league SHIP counts against the breadth target."""
    for league in sorted(scorecard_df["league"].unique()):
        sub = scorecard_df[scorecard_df["league"] == league]
        n_pass = int(sub["ship"].sum())
        n_markets = len(ALL_MARKETS.get(league, {}))
        target = int(np.ceil(BREADTH_TARGET_FRAC * n_markets)) if n_markets else 0
        flag = "OK" if n_pass >= target else "SHORT"
        click.echo(
            f"  {league}: {n_pass}/{n_markets} markets ship "
            f"(target >={target} for {BREADTH_TARGET_FRAC:.0%}; {len(sub)} evaluated) [{flag}]"
        )


def _print_ship_summary(rows: list[dict[str, object]]) -> None:
    """Bottom-line SHIP/KILL verdict per cell, so a sweep's result reads without
    scrolling back through the per-cell decile tables. ANSI-styled cells misalign
    inside a tabulate table, so the table stays plain text and only the tally is colored.
    """
    summary = []
    for row in rows:
        ships = bool(row["ship"])
        failed = [g for g in _SHIP_GATES if not row.get(f"{g}_pass", True)]
        summary.append(
            [
                f"{row['league']}_{row['market']}",
                "SHIP" if ships else "KILL",
                "-" if ships else " ".join(failed),
            ]
        )
    click.echo("\nSHIP SUMMARY")
    click.echo(
        tabulate.tabulate(summary, headers=["cell", "verdict", "failed gates"], tablefmt="github")
    )
    n_ship = sum(bool(row["ship"]) for row in rows)
    click.secho(f"{n_ship}/{len(rows)} cells ship", fg="green" if n_ship == len(rows) else "red")


def _gate_headline(row: dict[str, object]) -> str:
    """One-line per-cell gate summary for stdout, with SHIP / KILL verdict.

    Cells with no book ``Odds`` get a ``(no Odds; G1 auto-pass)`` note. When the row
    has been augmented by :func:`apply_thresholds`, the line ends with ``[SHIP]``
    or ``[KILL: g3 g4 ...]`` naming the failing gates.
    """

    def f(key: str, fmt: str = "+.4f") -> str:
        v = row.get(key)
        return "nan" if v is None else format(float(v), fmt)

    head = (
        f"G1 brier_diff {f('g1_brier_diff_mean')} "
        f"[{f('g1_brier_diff_ci_lo')}, {f('g1_brier_diff_ci_hi')}]  "
        f"G2 star_z {f('g2_star_z', '.2f')}  G3 bench_z {f('g3_bench_z', '.2f')}  "
        f"G4 iqr_ratio {f('g4_iqr_ratio', '.3f')}  "
        f"G5 ece_db {f('g5_ece_debiased', '.4f')} (raw {f('g5_ece', '.4f')})"
    )
    if row.get("g6_star_ci_hi") is not None:
        head += f"  G6 star_hi {f('g6_star_ci_hi', '.3f')} / ref {f('g6_star_ref', '.3f')}"
    if row.get("g1_brier_diff_mean") is None:
        head += "  (no Odds; G1 auto-pass)"
    if "ship" in row:
        if row["ship"]:
            head += "  [SHIP]"
        else:
            failed = [g for g in _SHIP_GATES if not row.get(f"{g}_pass", True)]
            head += f"  [KILL: {' '.join(failed)}]"
    return head


# Supersede gate (S2/S3) constants — see docs/ship_gate.md "research -> devel,
# supersede an incumbent: S1 + S2 + S3".
# Distinct from _GATE1_SEED so S2's bootstrap doesn't correlate with Gate 1's
# resampling on the same events.
_SUPERSEDE_S2_SEED: int = 2027
# Book implied-probability floor/ceiling for the S3 Kelly sim — a book quote at
# 0.01 implies a 100x payout that swamps a noisy model probability; clip to the
# range a real prop bet actually quotes at.
_SUPERSEDE_BOOK_P_FLOOR: float = 0.05
_SUPERSEDE_BOOK_P_CEILING: float = 0.95
# Initial bankroll for the paired Sharpe sim — units arbitrary; the gate compares
# Sharpe ratios via Memmel inference, so absolute dollars are immaterial.
_SUPERSEDE_S3_INITIAL_BANKROLL: float = 1000.0

# S3 Memmel (2003) paired Sharpe z critical value. Per the lifecycle gate audit
# the bare ``sharpe_c > sharpe_b`` rule has ~50 % Type-I rate; switching to a
# one-sided z > 1.645 test at α = 0.05 restores nominal coverage. Reference:
# Memmel (2003), "Performance Hypothesis Testing with the Sharpe Ratio".
_SUPERSEDE_S3_Z_MIN: float = 1.645
_SUPERSEDE_EVENT_KEY_COLUMNS: tuple[str, ...] = ("Player", "Date", "Line", ACTUAL_COL)


def _align_supersede_events(
    baseline: pd.DataFrame, candidate: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Return exact shared events in deterministic order, or ``None`` when identity is unsafe.

    CSV row numbers are serialization details, not event identity. Pair on the
    player, normalized event date, quoted line, and realized result instead. The
    result belongs in the key so two rows whose outcomes disagree can never be
    compared as the same event. A duplicate key in either side is ambiguous once
    the original matrix index has been discarded, so fail closed rather than
    manufacture an ordinal pairing that depends on row order.
    """
    required = set(_SUPERSEDE_EVENT_KEY_COLUMNS)
    if not required.issubset(baseline.columns) or not required.issubset(candidate.columns):
        return None

    def event_keys(frame: pd.DataFrame) -> pd.MultiIndex | None:
        players = frame["Player"].astype("string")
        dates = pd.to_datetime(frame["Date"], errors="coerce", utc=True)
        lines = pd.to_numeric(frame["Line"], errors="coerce")
        results = pd.to_numeric(frame[ACTUAL_COL], errors="coerce")
        if (
            players.isna().any()
            or players.str.len().eq(0).any()
            or dates.isna().any()
            or not np.isfinite(lines.to_numpy(dtype=float)).all()
            or not np.isfinite(results.to_numpy(dtype=float)).all()
        ):
            return None
        return pd.MultiIndex.from_arrays(
            [players.astype(str), dates, lines.astype(float), results.astype(float)],
            names=_SUPERSEDE_EVENT_KEY_COLUMNS,
        )

    baseline_keys = event_keys(baseline)
    candidate_keys = event_keys(candidate)
    if baseline_keys is None or candidate_keys is None:
        return None
    shared = baseline_keys.unique().intersection(candidate_keys.unique()).sort_values()
    if shared.empty:
        return None

    baseline_mask = baseline_keys.isin(shared)
    candidate_mask = candidate_keys.isin(shared)
    baseline_shared_keys = baseline_keys[baseline_mask]
    candidate_shared_keys = candidate_keys[candidate_mask]
    if baseline_shared_keys.duplicated().any() or candidate_shared_keys.duplicated().any():
        return None

    baseline_aligned = baseline.loc[baseline_mask].copy()
    candidate_aligned = candidate.loc[candidate_mask].copy()
    baseline_aligned.index = baseline_shared_keys
    candidate_aligned.index = candidate_shared_keys
    return (
        baseline_aligned.reindex(shared).reset_index(drop=True),
        candidate_aligned.reindex(shared).reset_index(drop=True),
    )


def _supersede_paired_brier_ci(
    b_df: pd.DataFrame, c_df: pd.DataFrame
) -> tuple[int, float, float, float] | None:
    """S2 — paired Brier CI on exact shared events from two test sets.

    ``d_i = brier_baseline_i - brier_candidate_i`` per shared event. Positive ``d``
    ⇒ candidate has lower Brier; the gate fires when the 95% percentile CI of
    ``mean(d)`` strictly excludes 0 from below (``ci_lo > 0``). Returns
    ``(n_shared, mean, ci_lo, ci_hi)`` or ``None`` if either frame lacks the
    requisite columns or the intersection is empty.
    """
    if "P" not in b_df.columns or "P" not in c_df.columns:
        return None
    aligned = _align_supersede_events(b_df, c_df)
    if aligned is None:
        return None
    b_aligned, c_aligned = aligned
    p_b = pd.to_numeric(b_aligned["P"], errors="coerce").to_numpy(dtype=float)
    p_c = pd.to_numeric(c_aligned["P"], errors="coerce").to_numpy(dtype=float)
    usable = np.isfinite(p_b) & np.isfinite(p_c)
    if not usable.any():
        return None
    b_aligned = b_aligned.loc[usable]
    c_aligned = c_aligned.loc[usable]
    p_b = np.clip(p_b[usable], _PROBA_CLIP, 1.0 - _PROBA_CLIP)
    p_c = np.clip(p_c[usable], _PROBA_CLIP, 1.0 - _PROBA_CLIP)
    y = (
        c_aligned["Result"].astype(float).to_numpy() >= c_aligned["Line"].astype(float).to_numpy()
    ).astype(float)
    brier_b = (p_b - y) ** 2
    brier_c = (p_c - y) ** 2
    d = brier_b - brier_c
    rng = np.random.default_rng(_SUPERSEDE_S2_SEED)
    mean, lo, hi = _bootstrap_mean_ci(d, rng)
    return len(b_aligned), mean, lo, hi


def _test_set_to_bet_frame(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """Adapt a test-set frame to the per-bet schema ``simulate_kelly_all`` expects.

    For each event the model picks the **EV side** (``over`` if ``pred >= Line`` else
    ``under``). The bet's ``Model P`` is the model's probability on that side;
    ``Boost`` is the decimal-odds payout ``1 / clip(book_p)`` and
    ``Platform="Sleeper"`` makes
    :func:`sportstradamus.strategies.profit_sim.compute_payout` return the net
    ``boost - 1``; the sim's Kelly branch rebuilds decimal odds as ``net + 1`` (so
    the fraction is unchanged) and settles winners at net, giving each event a
    well-defined Kelly stake and return. Synthetic monotonic dates make each event
    its own "day" so the resulting return series has per-event resolution.
    Returns an empty frame when ``Odds`` / ``P`` / ``Line`` are absent.
    """
    if _calibration_inputs(df) is None or "Odds" not in df.columns:
        return pd.DataFrame()
    sub = (
        df[["P", "Odds", "Line", ACTUAL_COL, pred_col]].replace([np.inf, -np.inf], np.nan).dropna()
    )
    if sub.empty:
        return pd.DataFrame()
    p_model_over = np.clip(sub["P"].to_numpy(dtype=float), _PROBA_CLIP, 1.0 - _PROBA_CLIP)
    p_book_over = np.clip(
        1.0 - sub["Odds"].to_numpy(dtype=float),
        _SUPERSEDE_BOOK_P_FLOOR,
        _SUPERSEDE_BOOK_P_CEILING,
    )
    pred = sub[pred_col].to_numpy(dtype=float)
    line = sub["Line"].to_numpy(dtype=float)
    result = sub[ACTUAL_COL].to_numpy(dtype=float)

    bet_over = pred >= line
    p_model = np.where(bet_over, p_model_over, 1.0 - p_model_over)
    p_book = np.where(bet_over, p_book_over, 1.0 - p_book_over)
    payout_decimal = 1.0 / p_book
    hit_over = result >= line
    hit = np.where(bet_over, hit_over, ~hit_over)

    n = len(sub)
    dates = [(_SUPERSEDE_S3_BASE_DATE + pd.Timedelta(days=int(i))).date() for i in range(n)]
    return pd.DataFrame(
        {
            "Player": [f"E{i}" for i in range(n)],
            "Market": "supersede",
            "Platform": "Sleeper",
            "Boost": payout_decimal,
            "Win Prob": p_model,
            "Model EV": p_model,
            "Kelly": p_model * payout_decimal,
            "Market EV": 1.0,
            "Hit": hit,
            "_date": dates,
        }
    )


def _memmel_sharpe_z(b_returns: np.ndarray, c_returns: np.ndarray) -> tuple[float, float, float]:
    """Memmel (2003) closed-form test for the difference of paired Sharpe ratios.

    Given paired return series with means μ_b, μ_c, stdevs σ_b, σ_c, and
    Pearson correlation ρ, the z-statistic for ``SR_c - SR_b`` is::

        Var(SR_c - SR_b) ≈ (1 / T) * (
            2 * (1 - ρ)
            + 0.5 * (SR_b² + SR_c²)
            - SR_b * SR_c * (ρ² + 0.5 * (1 + ρ²))
        )
        z = (SR_c - SR_b) / sqrt(Var)

    Replaces the bare ``sharpe_c > sharpe_b`` rule, which the lifecycle gate
    audit measured at ~50 % Type-I rate. Returns ``(SR_b, SR_c, z)``.
    Degenerate inputs (zero variance, length < 2, non-finite variance) return
    ``z = 0`` rather than raising — the caller treats those as "no
    information, hold".

    Reference: Memmel, C. (2003), "Performance Hypothesis Testing with the
    Sharpe Ratio", Finance Letters 1:21–23.
    """
    b = np.asarray(b_returns, dtype=float)
    c = np.asarray(c_returns, dtype=float)
    n = min(len(b), len(c))
    if n < 2:
        return 0.0, 0.0, 0.0
    b = b[:n]
    c = c[:n]
    sd_b = float(np.std(b, ddof=0))
    sd_c = float(np.std(c, ddof=0))
    sr_b = float(np.mean(b)) / sd_b if sd_b > 0 else 0.0
    sr_c = float(np.mean(c)) / sd_c if sd_c > 0 else 0.0
    if sd_b == 0.0 or sd_c == 0.0:
        return sr_b, sr_c, 0.0
    rho = float(np.corrcoef(b, c)[0, 1])
    if not np.isfinite(rho):
        rho = 0.0
    var = (1.0 / n) * (
        2.0 * (1.0 - rho)
        + 0.5 * (sr_b**2 + sr_c**2)
        - sr_b * sr_c * (rho**2 + 0.5 * (1.0 + rho**2))
    )
    if not np.isfinite(var) or var <= 0.0:
        return sr_b, sr_c, 0.0
    z = (sr_c - sr_b) / float(np.sqrt(var))
    if not np.isfinite(z):
        return sr_b, sr_c, 0.0
    return sr_b, sr_c, float(z)


def _supersede_paired_sharpe(
    b_df: pd.DataFrame, c_df: pd.DataFrame, pred_col: str
) -> tuple[float, float, float] | None:
    """S3 — paired Sharpe + Memmel z from a Kelly-all sim on shared events.

    Returns ``(sharpe_baseline, sharpe_candidate, memmel_z)`` or ``None``
    when either frame can't be adapted to a bet frame (no ``Odds`` column
    or empty intersection). The verdict layer ships on ``z > 1.645``; the
    raw Sharpe pair is preserved for the headline + scorecard row.
    """
    # Imported here to keep ``compression_eval``'s import surface minimal — the
    # supersede path is the only one that needs the profit-sim lib.
    from sportstradamus.strategies.profit_sim import extract_sim_returns, simulate_kelly_all

    aligned = _align_supersede_events(b_df, c_df)
    if aligned is None:
        return None
    b_aligned, c_aligned = aligned
    needed = {"P", "Odds", "Line", ACTUAL_COL, pred_col}
    if not needed.issubset(b_aligned.columns) or not needed.issubset(c_aligned.columns):
        return None
    b_numeric = b_aligned[list(needed)].apply(pd.to_numeric, errors="coerce")
    c_numeric = c_aligned[list(needed)].apply(pd.to_numeric, errors="coerce")
    usable = np.isfinite(b_numeric.to_numpy(dtype=float)).all(axis=1) & np.isfinite(
        c_numeric.to_numpy(dtype=float)
    ).all(axis=1)
    if not usable.any():
        return None
    b_bets = _test_set_to_bet_frame(b_aligned.loc[usable], pred_col)
    c_bets = _test_set_to_bet_frame(c_aligned.loc[usable], pred_col)
    if b_bets.empty or c_bets.empty:
        return None
    b_sim = simulate_kelly_all(
        b_bets, prob_col="Win Prob", initial_bankroll=_SUPERSEDE_S3_INITIAL_BANKROLL
    )
    c_sim = simulate_kelly_all(
        c_bets, prob_col="Win Prob", initial_bankroll=_SUPERSEDE_S3_INITIAL_BANKROLL
    )
    b_returns = extract_sim_returns(b_sim, _SUPERSEDE_S3_INITIAL_BANKROLL)
    c_returns = extract_sim_returns(c_sim, _SUPERSEDE_S3_INITIAL_BANKROLL)
    sr_b, sr_c, z = _memmel_sharpe_z(b_returns, c_returns)
    return float(sr_b), float(sr_c), float(z)


def supersede_verdict(
    b_df: pd.DataFrame,
    c_df: pd.DataFrame,
    pred_col: str,
    *,
    league: str = "",
    market: str = "",
    strategy: str = "candidate",
) -> dict[str, object]:
    """Run S1 + S2 + S3 on a row-aligned baseline / candidate pair.

    Returns a dict with all gate measurements plus ``ship`` (the AND of all
    three). Missing inputs propagate as ``None``/``False`` for the affected
    gates; the ``ship`` verdict is conservative — a ``None``-flagged gate
    fails. See `docs/ship_gate.md`'s "research -> devel, supersede an incumbent"
    table for the rules.
    """
    out: dict[str, object] = {}
    decode_strategy = _decode_strategy_for_frame(c_df, league, market)
    c_row = apply_thresholds(
        gate_row(
            c_df,
            pred_col,
            league=league,
            market=market,
            strategy=strategy,
            decode_strategy=decode_strategy,
        )
    )
    out["s1_pass"] = bool(c_row.get("ship", False))

    s2 = _supersede_paired_brier_ci(b_df, c_df)
    if s2 is None:
        out.update(s2_n=None, s2_mean=None, s2_ci_lo=None, s2_ci_hi=None, s2_pass=False)
    else:
        n, mean, lo, hi = s2
        out.update(s2_n=n, s2_mean=mean, s2_ci_lo=lo, s2_ci_hi=hi, s2_pass=lo > 0)

    s3 = _supersede_paired_sharpe(b_df, c_df, pred_col)
    if s3 is None:
        out.update(
            s3_sharpe_baseline=None,
            s3_sharpe_candidate=None,
            s3_memmel_z=None,
            s3_pass=False,
        )
    else:
        sb, sc, z = s3
        out.update(
            s3_sharpe_baseline=sb,
            s3_sharpe_candidate=sc,
            s3_memmel_z=z,
            s3_pass=z > _SUPERSEDE_S3_Z_MIN,
        )

    out["ship"] = bool(out["s1_pass"] and out["s2_pass"] and out["s3_pass"])
    return out


def _supersede_headline(v: dict[str, object]) -> str:
    """One-line stdout headline for a supersede verdict."""

    def f(key: str, fmt: str = "+.4f") -> str:
        x = v.get(key)
        return "nan" if x is None else format(float(x), fmt)

    parts = [
        f"S1={'PASS' if v.get('s1_pass') else 'FAIL'}",
        f"S2 d_mean {f('s2_mean')} [{f('s2_ci_lo')}, {f('s2_ci_hi')}] "
        f"n={v.get('s2_n', 'nan')} → {'PASS' if v.get('s2_pass') else 'FAIL'}",
        f"S3 sharpe base→cand {f('s3_sharpe_baseline', '.3f')} → "
        f"{f('s3_sharpe_candidate', '.3f')} (z {f('s3_memmel_z', '+.2f')}) "
        f"→ {'PASS' if v.get('s3_pass') else 'FAIL'}",
        f"[{'SUPERSEDE' if v.get('ship') else 'HOLD'}]",
    ]
    return "  ".join(parts)


def append_run_log(card: Scorecard, log_path: Path) -> None:
    """Append a scorecard row to the cross-session run log CSV.

    The :class:`Scorecard` schema only grows (new fields are appended last), so a
    log first written under an older, shorter header keeps that stale header while
    later rows carry the extra trailing columns — a ragged file ``pd.read_csv``
    rejects. When the on-disk header no longer matches the current columns,
    re-header the whole file (back-filling new columns as NaN for historical rows)
    before appending; otherwise take the plain append fast path.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([asdict(card)])
    if not log_path.exists():
        row.to_csv(log_path, index=False)
        return
    with log_path.open() as fh:
        on_disk_header = fh.readline().rstrip("\r\n")
    if on_disk_header == ",".join(row.columns):
        row.to_csv(log_path, mode="a", header=False, index=False)
        return
    migrated = pd.read_csv(log_path).reindex(columns=row.columns)
    pd.concat([migrated, row], ignore_index=True).to_csv(log_path, index=False)


def write_scatter(df: pd.DataFrame, pred_col: str, out_path: Path, title: str) -> None:
    """Write a predicted-vs-actual scatter colored by MeanYr decile.

    matplotlib is imported lazily so the numeric path (and its unit tests) does
    not require a display backend or the optional plotting dependency.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    work = df.copy()
    work["decile"] = pd.qcut(work[DECILE_COL].rank(method="first"), N_DECILES, labels=False)
    fig, ax = plt.subplots(figsize=_SCATTER_FIG_INCHES)
    sc = ax.scatter(
        work[ACTUAL_COL], work[pred_col], c=work["decile"], cmap="viridis", s=8, alpha=0.4
    )
    lim = [0, float(max(work[ACTUAL_COL].max(), work[pred_col].max()))]
    ax.plot(lim, lim, "r--", linewidth=1, label="y = x (perfect)")
    ax.set_xlabel("Actual")
    ax.set_ylabel(f"Predicted ({pred_col})")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.colorbar(sc, ax=ax, label="MeanYr decile")
    fig.tight_layout()
    fig.savefig(out_path, dpi=_SCATTER_DPI)
    plt.close(fig)


def _print_table(table: pd.DataFrame) -> None:
    click.echo(
        f"{'decile':>6} {'meanyr':>8} {'n':>6} {'mae':>8} {'bias':>8} {'pred':>8} {'actual':>8}"
    )
    for _, r in table.iterrows():
        click.echo(
            f"{int(r['decile']):>6} {r['meanyr']:>8.2f} {int(r['n']):>6} "
            f"{r['mae']:>8.3f} {r['bias']:>+8.3f} {r['pred_mean']:>8.2f} "
            f"{r['actual_mean']:>8.2f}"
        )


def _resolve_test_sets(test_sets_dir: Path, league: str | None, market: str | None) -> list[Path]:
    """Resolve the CSV files to evaluate from --league/--market filters."""
    paths = sorted(test_sets_dir.glob("*.csv"))
    if league:
        paths = [p for p in paths if p.stem.startswith(f"{league}_")]
    if market:
        paths = [p for p in paths if p.stem == f"{league}_{market}".replace(" ", "-")]
    return paths


def _history_to_eval_frame(
    history: pd.DataFrame,
    league: str,
    market: str,
    window_days: int,
    meanyr_lookup: MeanYrLookup,
) -> pd.DataFrame:
    """Project history.parquet rows into the offline scorecard()-shaped frame.

    Filters to ``(league, market)`` settled offers within ``window_days`` and
    constructs the CSV-shaped columns the existing harness expects:
    ``MeanYr`` from the injected lookup; ``Result`` from ``Actual``; ``EV``
    from prediction-level ``Model EV`` (raw-stat units); ``P`` normalized to
    model OVER-probability and ``Odds`` normalized to book UNDER-probability
    so the existing :func:`_brier_skill_score` semantics hold unchanged.
    """
    if history.empty:
        return pd.DataFrame(columns=list(_LIVE_EVAL_COLUMNS))
    dated = history.copy()
    cutoff = pd.Timestamp(datetime.now(UTC).date()) - pd.Timedelta(days=window_days)
    dated["_date"] = pd.to_datetime(dated["Date"], errors="coerce")
    mask = (
        (dated["League"] == league)
        & (dated["Market"] == market)
        & dated["Actual"].notna()
        & dated["_date"].notna()
        & (dated["_date"] >= cutoff)
    )
    subset = dated.loc[mask].copy()
    if subset.empty:
        return pd.DataFrame(columns=list(_LIVE_EVAL_COLUMNS))

    over_mask = subset["Bet"].eq("Over").to_numpy()
    model_p = subset["Win Prob"].to_numpy()
    books_p = subset["Market Prob"].to_numpy()
    out = pd.DataFrame(
        {
            "MeanYr": [
                meanyr_lookup(player, market, date)
                for player, date in zip(subset["Player"], subset["_date"], strict=False)
            ],
            "Result": subset["Actual"].astype(float).to_numpy(),
            "EV": subset["Projection"].astype(float).to_numpy(),
            "P": np.where(over_mask, model_p, 1.0 - model_p),
            "Odds": np.where(over_mask, 1.0 - books_p, books_p),
            "Line": subset["Line"].astype(float).to_numpy(),
        }
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out.reset_index(drop=True)


def _make_meanyr_lookup_from_gamelog(gamelog: pd.DataFrame, date_col: str) -> MeanYrLookup:
    """Closure that returns the player's prior-365-day mean of the market column.

    ``date_col`` is the gamelog's date-column name (varies per league via
    ``Stats.log_strings["date"]``). The closure returns NaN when the player is
    absent, the market column is missing, or the look-back window is empty.
    """
    if gamelog is None or gamelog.empty:
        return lambda player, market, date: float("nan")
    gl = gamelog.copy()
    gl[date_col] = pd.to_datetime(gl[date_col], errors="coerce")

    def lookup(player: str, market: str, date: pd.Timestamp) -> float:
        if market not in gl.columns:
            return float("nan")
        window_start = date - pd.Timedelta(days=_MEANYR_LOOKBACK_DAYS)
        prior = gl[
            (gl.get("playerName", gl.get("player display name", "")) == player)
            & (gl[date_col] < date)
            & (gl[date_col] >= window_start)
        ]
        if prior.empty:
            return float("nan")
        return float(prior[market].mean())

    return lookup


@functools.cache
def _load_league_stats_lookup(league: str) -> MeanYrLookup:
    """Load the league's Stats class once and return a MeanYr lookup callable.

    Caches across calls within a process so multi-league --live-window runs pay
    the gamelog load cost once per league. Returns a NaN-only lookup when the
    league or gamelog is unavailable so the live-window mode degrades gracefully.
    """
    from sportstradamus.nightly import LEAGUE_CLASSES

    stats_cls = LEAGUE_CLASSES.get(league)
    if stats_cls is None:
        return lambda player, market, date: float("nan")
    obj = stats_cls()
    try:
        obj.load()
    except Exception:
        return lambda player, market, date: float("nan")
    gamelog = getattr(obj, "gamelog", pd.DataFrame())
    date_col = getattr(obj, "log_strings", {}).get("date", "gameDate")
    return _make_meanyr_lookup_from_gamelog(gamelog, date_col)


def _print_live_scorecard(card: Scorecard, stem: str, pred_col: str) -> None:
    """Print the live-window scorecard summary in the same shape as offline mode."""
    click.echo(f"\n=== {stem}  ({pred_col}, n={card.n_rows}) ===")
    click.echo(
        f"strategy={card.strategy}  "
        f"global_mae={card.global_mae:.3f}  "
        f"top_decile_mae={card.top_decile_mae:.3f}  "
        f"top_decile_bias={card.top_decile_bias:+.3f}  "
        f"bottom_quartile_bias={card.bottom_quartile_bias:+.3f}  "
        f"compression_ratio={card.compression_ratio:.3f} "
        f"(top {card.top_decile_compression_ratio:.3f})"
    )
    click.echo(
        f"result_meanyr_corr={card.result_meanyr_corr:+.3f}  "
        f"pred_meanyr_corr={card.pred_meanyr_corr:+.3f}"
    )
    if card.brier_skill_score is not None:
        click.echo(f"brier_skill_score={card.brier_skill_score:+.3f}")


def _resolve_live_cells(
    history: pd.DataFrame, league: str | None, market: str | None
) -> list[tuple[str, str]]:
    """Return distinct ``(league, market)`` pairs present in history matching filters."""
    if history.empty:
        return []
    settled = history[history["Actual"].notna()]
    if settled.empty:
        return []
    if league:
        settled = settled[settled["League"] == league]
    if market:
        settled = settled[settled["Market"] == market]
    return sorted({(row.League, row.Market) for row in settled.itertuples(index=False)})


def _resolve_diff_cell(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    league: str | None,
    market: str | None,
) -> tuple[str, str]:
    baseline_identity, _ = validate_strategy_frame(baseline)
    candidate_identity, _ = validate_strategy_frame(candidate)
    if (baseline_identity is None) != (candidate_identity is None):
        raise click.UsageError("Diff mode cannot mix legacy and generic strategy identities.")
    if baseline_identity is None:
        return league or "", (market or "").replace(" ", "-")

    baseline_cell = (
        baseline_identity.league,
        baseline_identity.market.replace(" ", "-"),
    )
    candidate_cell = (
        candidate_identity.league,
        candidate_identity.market.replace(" ", "-"),
    )
    if baseline_cell != candidate_cell:
        raise click.UsageError("Diff mode requires baseline and candidate from the same cell.")
    requested_cell = (league or baseline_cell[0], (market or baseline_cell[1]).replace(" ", "-"))
    if requested_cell != baseline_cell:
        raise click.UsageError("Diff mode cell does not match the generic strategy identity.")
    return requested_cell


@click.command()
@click.option("--league", default=None, help="Filter test sets by league (e.g. NBA).")
@click.option("--market", default=None, help="Single market stem (requires --league).")
@click.option(
    "--pred-col",
    type=click.Choice(["EV", "Blended_EV"]),
    default=DEFAULT_PRED_COL,
    help="Predicted-mean column to evaluate. Blended_EV = fused ship gate (default); EV = raw model.",
)
@click.option("--strategy", default="unlabeled", help="Strategy label for the run log.")
@click.option("--deciles", default=N_DECILES, show_default=True, help="Number of buckets.")
@click.option("--scatter/--no-scatter", default=False, help="Write a scatter PNG to /tmp.")
@click.option(
    "--test-sets-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Override the test_sets directory (defaults to the package data dir).",
)
@click.option(
    "--baseline",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Diff mode: baseline test-set CSV.",
)
@click.option(
    "--candidate",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Diff mode: candidate test-set CSV (compared against --baseline).",
)
@click.option("--no-log", is_flag=True, default=False, help="Skip appending to the run log.")
@click.option(
    "--scorecard/--no-scorecard",
    "write_scorecard",
    default=True,
    help="On a full audit, write the per-cell five-gate scorecard CSV (model + oracle).",
)
@click.option(
    "--scorecard-out",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Gate scorecard CSV path. Defaults to /tmp/scorecard.csv on a full audit "
        "(no --league/--market filter); a filtered run only writes when this flag "
        "is given explicitly. The production model_stats.parquet is owned by "
        "training.report.report() and is never touched by this CLI."
    ),
)
@click.option(
    "--live-window",
    type=int,
    default=None,
    help=(
        "Score the last N days of settled offers from history.parquet instead "
        "of test_sets CSVs. Reuses the offline decile-bias path; strategy label "
        "becomes `live_{N}d` unless --strategy is given."
    ),
)
def main(
    league: str | None,
    market: str | None,
    pred_col: str,
    strategy: str,
    deciles: int,
    scatter: bool,
    test_sets_dir: Path | None,
    baseline: Path | None,
    candidate: Path | None,
    no_log: bool,
    write_scorecard: bool,
    scorecard_out: Path | None,
    live_window: int | None,
) -> None:
    """Score compression on dumped test sets, diff two strategies, or score live data."""
    # style: allow-complexity — scorecard entrypoint: a 3-mode dispatcher
    # (live-window / baseline-candidate / test-set sweep), each a guarded
    # sequential block. The residual CC is mode selection plus click UsageError
    # input validation, not nested logic.
    log_path = Path(str(RUN_LOG_PATH))

    if live_window is not None:
        if baseline or candidate:
            raise click.UsageError("--live-window cannot combine with --baseline/--candidate.")
        if test_sets_dir is not None:
            raise click.UsageError("--live-window does not use --test-sets-dir.")
        live_strategy = strategy if strategy != "unlabeled" else f"live_{live_window}d"
        history = read_history()
        if history.empty:
            raise click.UsageError("history.parquet is empty; nothing to score.")
        cells = _resolve_live_cells(history, league, market)
        if not cells:
            raise click.UsageError("No settled offers match the --league/--market filters.")
        for cell_league, cell_market in cells:
            lookup = _load_league_stats_lookup(cell_league)
            frame = _history_to_eval_frame(history, cell_league, cell_market, live_window, lookup)
            if frame.empty:
                click.echo(f"{cell_league}_{cell_market}: no offers in last {live_window}d.")
                continue
            # history.parquet carries only the raw model mean (Model EV -> EV); the fused
            # Blended_EV lives in the dumped test-set CSVs, so live monitoring scores EV.
            card = scorecard(
                frame,
                "EV",
                strategy=live_strategy,
                league=cell_league,
                market=cell_market,
                n_deciles=deciles,
            )
            _print_live_scorecard(card, f"{cell_league}_{cell_market}", "EV")
            if not no_log:
                append_run_log(card, log_path)
        return

    if baseline or candidate:
        if not (baseline and candidate):
            raise click.UsageError("--baseline and --candidate must be given together.")
        if bool(league) != bool(market):
            raise click.UsageError(
                "Diff mode requires --league and --market together when identifying a cell."
            )
        b_df = load_test_set(baseline, pred_col)
        c_df = load_test_set(candidate, pred_col)
        diff_league, diff_market = _resolve_diff_cell(b_df, c_df, league, market)
        baseline_decode = _decode_strategy_for_frame(b_df, diff_league, diff_market)
        candidate_decode = _decode_strategy_for_frame(c_df, diff_league, diff_market)
        b_row = apply_thresholds(
            gate_row(
                b_df,
                pred_col,
                league=diff_league,
                market=diff_market,
                strategy="baseline",
                decode_strategy=baseline_decode,
            )
        )
        c_row = apply_thresholds(
            gate_row(
                c_df,
                pred_col,
                league=diff_league,
                market=diff_market,
                strategy=strategy,
                decode_strategy=candidate_decode,
            )
        )
        click.echo(f"baseline : {baseline.name}")
        _print_table(decile_table(b_df, pred_col, deciles))
        click.echo(_gate_headline(b_row))
        click.echo(f"\ncandidate: {candidate.name}")
        _print_table(decile_table(c_df, pred_col, deciles))
        click.echo(_gate_headline(c_row))
        verdict = supersede_verdict(
            b_df,
            c_df,
            pred_col,
            league=diff_league,
            market=diff_market,
            strategy=strategy,
        )
        click.echo(f"\nsupersede: {_supersede_headline(verdict)}")
        return

    resolved_dir = test_sets_dir or Path(str(pkg_resources.files(data) / "test_sets"))
    if not resolved_dir.exists():
        raise click.UsageError(f"No test_sets directory at {resolved_dir}. Run `meditate` first.")
    paths = _resolve_test_sets(resolved_dir, league, market)
    if not paths:
        raise click.UsageError("No matching test-set CSVs found.")

    rows: list[dict[str, object]] = []
    for path in paths:
        stem = path.stem
        lg, _, mkt = stem.partition("_")
        df = load_test_set(path, pred_col)
        card = scorecard(df, pred_col, strategy=strategy, league=lg, market=mkt, n_deciles=deciles)
        decode_strategy = _decode_strategy_for_frame(df, lg, mkt)
        row = apply_thresholds(
            gate_row(
                df,
                pred_col,
                league=lg,
                market=mkt,
                strategy=strategy,
                decode_strategy=decode_strategy,
            )
        )
        rows.append(row)
        click.echo(f"\n=== {stem}  ({pred_col}, n={card.n_rows}) ===")
        _print_table(decile_table(df, pred_col, deciles))
        click.echo(
            f"compression_ratio={card.compression_ratio:.3f} "
            f"(top {card.top_decile_compression_ratio:.3f})  "
            f"result_meanyr_corr={card.result_meanyr_corr:+.3f}  "
            f"pred_meanyr_corr={card.pred_meanyr_corr:+.3f}"
        )
        click.echo(_gate_headline(row))
        if scatter:
            out = SCATTER_DIR / f"compression_{stem}_{pred_col}.png"
            write_scatter(df, pred_col, out, f"{stem} — {strategy}")
            click.echo(f"scatter: {out}")
        if not no_log:
            append_run_log(card, log_path)

    if rows:
        _print_ship_summary(rows)

    # Gate scorecard snapshot. Only a FULL audit (no league/market filter) auto-writes
    # the sandbox scorecard, so a filtered run can't clobber it down to a subset; a
    # filtered run still writes when --scorecard-out is given explicitly. The
    # production model_stats.parquet is owned by training.report.report() and is
    # never written from this CLI path.
    if write_scorecard and rows:
        if scorecard_out is not None:
            sc_path: Path | None = scorecard_out
        elif league is None and market is None:
            sc_path = _SCORECARD_SANDBOX_DEFAULT
        else:
            sc_path = None
        if sc_path is not None:
            sc_df = write_gate_scorecard(rows, sc_path)
            click.echo(f"\nGate scorecard ({len(sc_df)} cells): {sc_path}")
            _print_breadth_rollup(sc_df)


if __name__ == "__main__":
    main()
