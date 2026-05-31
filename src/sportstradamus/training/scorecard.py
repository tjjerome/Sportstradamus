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
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import click
import numpy as np
import pandas as pd
from scipy.stats import gamma as _scipy_gamma
from scipy.stats import nbinom as _scipy_nbinom
from scipy.stats import skewnorm as _scipy_skewnorm

from sportstradamus import data
from sportstradamus.analysis import explode_offers
from sportstradamus.helpers.io import read_history
from sportstradamus.training.baselines import _MEANYR_FLOOR as _SN_DENOM_FLOOR
from sportstradamus.training.markets import ALL_MARKETS
from sportstradamus.training.ship_config import STAT_META_PATH, STRATEGY_NONE, load_stat_meta

# ---------------------------------------------------------------------------
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
#   * research -> devel, supersede: pass all five + a paired Brier CI (current-new,
#     95% CI excludes 0 in the new model's favor) + a paired Sharpe improvement on a
#     backdated Kelly sim (supersede_verdict, diff mode).
#   * devel -> main: a profitability gate on live settled data (positive Kelly-sized
#     ROI to set a baseline; >= +0.5% ROI over >= 2 weeks to supersede an incumbent)
#     — see scripts/check_graduation.py.
# ---------------------------------------------------------------------------

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
# pass/fail. A cell ships (research -> devel) iff all five pass.
#   G1 ci_hi  < 0    : 95% CI strictly excludes 0 below (model Brier < book's at 95%)
#   G2 star  z < 0.5 : top-mean-decile bias under half the segment's stdev of outcomes
#   G3 bench z < 0.5 : bottom-quartile bias under half the segment's stdev of outcomes
#   G4 iqr_ratio > 0.5: prediction IQR at least half the truth's (<= 50% compression)
#   G5 ece    < 0.075: 10-bin equal-mass ECE under 7.5% (Roelofs-debiased: raw - null bias offset)
_GATE1_CI_HI_MAX: float = 0.0
_GATE2_STAR_Z_MAX: float = 0.5
_GATE3_BENCH_Z_MAX: float = 0.5
_GATE4_IQR_RATIO_MIN: float = 0.5
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

# Raw model EV is the cleanest view of the model's own compression; Blended_EV
# mixes in the bookmaker line and masks it. Default to the raw model column.
DEFAULT_PRED_COL = "EV"

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


def _git_sha() -> str:
    """Return the short HEAD SHA, or ``"unknown"`` outside a git tree."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_test_set(path: Path, pred_col: str) -> pd.DataFrame:
    """Load a dumped test-set CSV, keeping only the columns the harness needs.

    Args:
        path: Path to a ``{LEAGUE}_{market}.csv`` produced by ``meditate``.
        pred_col: Predicted-mean column to evaluate (``EV`` or ``Blended_EV``).

    Returns:
        Frame with ``MeanYr``, ``Result``, the prediction column, optional
        priced columns (``P``, ``Odds``, ``Line``), and any per-row
        distribution parameters present (``SN_Loc`` / ``SN_Scale`` /
        ``SN_Alpha`` for SkewNormal, ``R`` / ``NB_P`` for NegBin/ZINB,
        ``Alpha`` for Gamma/ZAGamma, ``Gate`` for the zero-inflated variants).
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
    optional = {"P", "Odds", "Line"} & set(df.columns)
    # Per-row distribution params (Operation Ship 75 Step 0.2 G4 audit).
    # ``training/pipeline.py::_step_persist_artifacts`` dumps these at lines
    # 1191-1212; older CSVs predating that dump simply leave them out and the
    # gate falls back to the point-IQR estimator via `_infer_dist_from_columns`
    # returning None.
    dist_params = {
        "SN_Loc",
        "SN_Scale",
        "SN_Alpha",
        "R",
        "NB_P",
        "Alpha",
        "Gate",
    } & set(df.columns)
    out = df[sorted(required | optional | dist_params)].copy()
    # Filter non-finite rows on required columns only — missing P/Odds/Line rows
    # are filtered locally inside _brier_skill_score so older CSVs that lack
    # those columns still pass the harness.
    required_view = out[list(required)].replace([np.inf, -np.inf], np.nan)
    return out[required_view.notna().all(axis=1)]


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


def _brier_inputs(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return ``(p_model, p_book, y)`` for the priced Brier gates, or None if absent.

    Layers the book's ``Odds`` (book under-probability ⇒ book over = ``1 - Odds``) on
    top of :func:`_calibration_inputs`. The row set is re-filtered to drop rows with
    non-finite ``Odds`` (so the Brier and ECE row sets can differ when some events
    have a posted line but no book quote). Returns ``None`` when ``Odds`` is missing
    entirely or every priced row is non-finite. Shared by
    :func:`_brier_skill_score` and Gate 1 (:func:`_gate1_brier_ci`).
    """
    if "Odds" not in df.columns:
        return None
    needed = {"P", "Odds", "Line"}
    if not needed.issubset(df.columns):
        return None
    sub = df[["P", "Odds", "Line", ACTUAL_COL]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) == 0:
        return None
    y = (sub[ACTUAL_COL] >= sub["Line"]).astype(float).to_numpy()
    p_model = np.clip(sub["P"].to_numpy(), _PROBA_CLIP, 1 - _PROBA_CLIP)
    p_book = np.clip(1.0 - sub["Odds"].to_numpy(), _PROBA_CLIP, 1 - _PROBA_CLIP)
    return p_model, p_book, y


def _brier_skill_score(df: pd.DataFrame) -> float | None:
    """1 - brier(model_P) / brier(book_over) on the test set, or None if cols absent.

    Informational summary kept on the :class:`Scorecard` / run log; the Gate-1 ship
    signal is the paired bootstrap CI in :func:`_gate1_brier_ci`, not this ratio.
    """
    inputs = _brier_inputs(df)
    if inputs is None:
        return None
    p_model, p_book, y = inputs
    brier_model = float(np.mean((p_model - y) ** 2))
    brier_book = float(np.mean((p_book - y) ** 2))
    if brier_book <= 0:
        return None
    return 1.0 - brier_model / brier_book


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
        git_sha=_git_sha(),
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


def _iqr(values: np.ndarray) -> float:
    """Inter-quartile range (75th - 25th percentile) of ``values``."""
    p75, p25 = np.percentile(np.asarray(values, dtype=float), [75, 25])
    return float(p75 - p25)


# ---------------------------------------------------------------------------
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
# ---------------------------------------------------------------------------


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


def _infer_dist_from_columns(df: pd.DataFrame) -> str | None:
    """Identify the distribution family from per-row parameter columns.

    Returns one of ``"SkewNormal"``, ``"NegBin"``, ``"ZINB"``, ``"Gamma"``,
    ``"ZAGamma"`` based on which params ``training/pipeline.py``
    ``_step_persist_artifacts`` dumped into the test-set CSV (~lines
    1191-1212). ``None`` for legacy / synthetic frames missing every
    distribution param — those keep the back-compat point-IQR semantics.
    """
    cols = set(df.columns)
    if {"SN_Loc", "SN_Scale", "SN_Alpha"} <= cols:
        return "SkewNormal"
    if {"Alpha", "EV"} <= cols and "R" not in cols:
        return "ZAGamma" if "Gate" in cols else "Gamma"
    if {"R", "NB_P"} <= cols:
        return "ZINB" if "Gate" in cols else "NegBin"
    return None


# MeanYr floor for ratio_meanyr decode is canonical at
# `training.baselines._MEANYR_FLOOR` (imported at the top of this module as
# `_SN_DENOM_FLOOR`); the analytical-IQR mirror must agree with the training
# pipeline or the predicted IQR drifts from what the model actually output.

# Strategies that decode SkewNormal scale by multiplying with MeanYr_clipped,
# mirroring training.baselines._ratio_decode_scale. centered_additive_*
# strategies leave SN_Scale alone (baselines._centered_eb_decode_scale and
# _centered_mean10_decode_scale return raw scale).
_RATIO_LIKE_STRATEGIES: frozenset[str] = frozenset({"ratio_meanyr"})


def _decode_sn_loc_scale(df: pd.DataFrame, strategy: str) -> tuple[np.ndarray, np.ndarray]:
    """Decode raw SkewNormal ``loc`` / ``scale`` to EV-space per strategy.

    Mirrors ``training.baselines.TargetStrategy.decode_loc`` / ``decode_scale``
    for the strategies wired in ``_STRATEGIES`` there. ``ratio_meanyr``
    multiplies both by ``MeanYr.clip(_SN_DENOM_FLOOR)``; the
    ``centered_additive_*`` strategies leave ``scale`` alone (location decode
    is irrelevant for IQR, which is location-free for SkewNormal).
    """
    raw_loc = df["SN_Loc"].to_numpy(dtype=float)
    raw_scale = df["SN_Scale"].to_numpy(dtype=float)
    if strategy in _RATIO_LIKE_STRATEGIES and "MeanYr" in df.columns:
        meanyr = df["MeanYr"].clip(lower=_SN_DENOM_FLOOR).to_numpy(dtype=float)
        return raw_loc * meanyr, raw_scale * meanyr
    return raw_loc, raw_scale


def _iqr_pred_analytical(df: pd.DataFrame, dist: str, *, strategy: str) -> float:
    """Pooled analytical IQR of the predicted distribution across all rows.

    Per-row ``q25 = F⁻¹(0.25)`` and ``q75 = F⁻¹(0.75)`` are evaluated via
    ``scipy.stats.<dist>.ppf`` (custom inversion for the ZINB / ZAGamma
    mixtures). The arrays are concatenated and the bag's
    ``percentile(75) − percentile(25)`` is the pooled IQR — a discrete
    approximation to the mixture-predictive IQR, validated against
    sample-based pooled IQR to ≤ 0.001 on count families.
    """
    if dist == "SkewNormal":
        loc, scale = _decode_sn_loc_scale(df, strategy)
        alpha = df["SN_Alpha"].to_numpy(dtype=float)
        q25 = _scipy_skewnorm.ppf(0.25, alpha, loc=loc, scale=scale)
        q75 = _scipy_skewnorm.ppf(0.75, alpha, loc=loc, scale=scale)
    elif dist == "NegBin":
        r = df["R"].to_numpy(dtype=float)
        p = df["NB_P"].to_numpy(dtype=float)
        q25 = _scipy_nbinom.ppf(0.25, r, 1.0 - p)
        q75 = _scipy_nbinom.ppf(0.75, r, 1.0 - p)
    elif dist == "ZINB":
        r = df["R"].to_numpy(dtype=float)
        p = df["NB_P"].to_numpy(dtype=float)
        gate = df["Gate"].to_numpy(dtype=float)
        q25 = _zinb_ppf(0.25, r, p, gate)
        q75 = _zinb_ppf(0.75, r, p, gate)
    elif dist == "Gamma":
        # rate = concentration / EV; scale = 1 / rate = EV / concentration.
        a = df["Alpha"].to_numpy(dtype=float)
        ev = df["EV"].to_numpy(dtype=float)
        scale = ev / np.maximum(a, 1e-12)
        q25 = _scipy_gamma.ppf(0.25, a, scale=scale)
        q75 = _scipy_gamma.ppf(0.75, a, scale=scale)
    elif dist == "ZAGamma":
        # Mixture of point-mass at 0 (gate) + Gamma on Y>0. Per-row inversion
        # mirrors _zinb_ppf: q ≤ π → 0, else rescaled Gamma quantile.
        a = df["Alpha"].to_numpy(dtype=float)
        ev = df["EV"].to_numpy(dtype=float)
        gate = df["Gate"].to_numpy(dtype=float)
        scale = ev / np.maximum(a, 1e-12)
        rescaled_25 = np.where(gate >= 0.25, 0.0, (0.25 - gate) / np.maximum(1 - gate, 1e-12))
        rescaled_75 = np.where(gate >= 0.75, 0.0, (0.75 - gate) / np.maximum(1 - gate, 1e-12))
        q25 = np.where(gate >= 0.25, 0.0, _scipy_gamma.ppf(rescaled_25, a, scale=scale))
        q75 = np.where(gate >= 0.75, 0.0, _scipy_gamma.ppf(rescaled_75, a, scale=scale))
    else:
        raise ValueError(f"Unknown distribution family for analytical IQR: {dist!r}")
    return _iqr(np.concatenate([q25, q75]))


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
        iqr_pred = _iqr_pred_analytical(df, dist, strategy=strategy or "ratio_meanyr")
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
    normalized ratio-units. A ``none`` strategy means the cell took the
    ``--target-strategy`` default — ``ratio_meanyr``.
    """
    market_with_spaces = market_stem.replace("-", " ")
    strategy = (
        _cached_stat_meta()
        .get(league, {})
        .get(market_with_spaces, {})
        .get("strategy", STRATEGY_NONE)
    )
    return _DECODE_FALLBACK_STRATEGY if strategy == STRATEGY_NONE else strategy


def gate_row(
    df: pd.DataFrame,
    pred_col: str,
    *,
    league: str,
    market: str,
    strategy: str,
    decode_strategy: str | None = None,
) -> dict[str, object]:
    """Compute the five offline gates for one cell — a model row plus an oracle row.

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
    actual = df[ACTUAL_COL].to_numpy()
    pred = df[pred_col].to_numpy()
    star_mask, bench_mask = _segment_masks(df)

    # Gates 2/3 — model; the oracle (pred = actual) zeroes abs_diff / z, sigma unchanged.
    g2_pred, g2_true, g2_abs, g2_sigma, g2_z = _gate23_segment_match(pred, actual, star_mask)
    _, _, _, _, g2_z_oracle = _gate23_segment_match(actual, actual, star_mask)
    g3_pred, g3_true, g3_abs, g3_sigma, g3_z = _gate23_segment_match(pred, actual, bench_mask)
    _, _, _, _, g3_z_oracle = _gate23_segment_match(actual, actual, bench_mask)

    # Gate 4 — IQR spread; the oracle (pred = actual) gives ratio 1.0 via the
    # point-IQR fallback. The model row routes through the analytical
    # pooled-IQR estimator when per-row distribution params are present in df,
    # else falls back to the point estimator for back-compat (synthetic frames
    # in the golden tests). ``decode_strategy`` is the per-cell training
    # strategy (stat_meta lookup); ``strategy`` is just the run label kept
    # for the row.
    g4_dist = _infer_dist_from_columns(df)
    decode_for_g4 = decode_strategy or strategy
    if g4_dist is not None:
        g4_iqr_pred, g4_iqr_true, g4_ratio = _gate4_iqr_spread(
            actual, pred, df=df, dist=g4_dist, strategy=decode_for_g4
        )
    else:
        g4_iqr_pred, g4_iqr_true, g4_ratio = _gate4_iqr_spread(actual, pred)
    _, _, g4_ratio_oracle = _gate4_iqr_spread(actual, actual)

    # Gate 1 — paired Brier vs book. Needs Odds; blank ⇒ "no book to beat, model wins
    # by default" (the auto-pass convention is doc'd at the module-header and applied
    # at verdict-wiring time). Oracle p_model = y (the deterministic 1/0 prediction).
    brier_in = _brier_inputs(df)
    if brier_in is None:
        g1_mean = g1_lo = g1_hi = g1_mean_o = g1_lo_o = g1_hi_o = bss = None
    else:
        p_model_b, p_book, y_b = brier_in
        g1_mean, g1_lo, g1_hi = _gate1_brier_ci(
            p_model_b, p_book, y_b, np.random.default_rng(_GATE1_SEED)
        )
        g1_mean_o, g1_lo_o, g1_hi_o = _gate1_brier_ci(
            y_b, p_book, y_b, np.random.default_rng(_GATE1_SEED)
        )
        bss = _brier_skill_score(df)

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
        "g1_brier_skill_score": r(bss),
        "g2_star_pred_mean": r(g2_pred),
        "g2_star_true_mean": r(g2_true),
        "g2_star_abs_diff": r(g2_abs),
        "g2_star_sigma": r(g2_sigma),
        "g2_star_z": r(g2_z),
        "g2_star_z_oracle": r(g2_z_oracle),
        "g3_bench_pred_mean": r(g3_pred),
        "g3_bench_true_mean": r(g3_true),
        "g3_bench_abs_diff": r(g3_abs),
        "g3_bench_sigma": r(g3_sigma),
        "g3_bench_z": r(g3_z),
        "g3_bench_z_oracle": r(g3_z_oracle),
        "g4_iqr_pred": r(g4_iqr_pred),
        "g4_iqr_true": r(g4_iqr_true),
        "g4_iqr_ratio": r(g4_ratio),
        "g4_iqr_ratio_oracle": r(g4_ratio_oracle),
        "g5_ece": r(g5_ece),
        "g5_ece_oracle": r(g5_ece_o),
        "g5_ece_null_bias": r(g5_ece_bias),
        "g5_ece_debiased": r(g5_ece_db),
        "g5_ece_debiased_oracle": r(g5_ece_db_o),
    }


def apply_thresholds(row: dict[str, object]) -> dict[str, object]:
    """Augment a :func:`gate_row` row with per-gate ``*_pass`` flags + overall ``ship``.

    Applies the strict starter thresholds (:data:`_GATE1_CI_HI_MAX` etc.). Blank-cell
    semantics — distinct because the gates fail for different structural reasons:

    * Gate 1 blank (no ``Odds``): **auto-pass** — no book to beat, model wins by
      default. The only "no book data" auto-pass.
    * Gate 2/3/5 blank: **fail** — the cell couldn't compute the gate (e.g. missing
      ``P`` / ``Line``), and we don't credit the model for absence of evidence.
    * Gate 4 blank (``IQR(Result) = 0``): **fail** under this strict pass; the
      compression yardstick is structurally undefined for sparse ``tds``-style
      binary markets, flagged in ``docs/operation_ship_75.md`` Step 0.4 for revisit.
    """
    out = dict(row)
    g1_hi = out.get("g1_brier_diff_ci_hi")
    g1_pass = g1_hi is None or g1_hi < _GATE1_CI_HI_MAX
    g2 = out.get("g2_star_z")
    g2_pass = g2 is not None and g2 < _GATE2_STAR_Z_MAX
    g3 = out.get("g3_bench_z")
    g3_pass = g3 is not None and g3 < _GATE3_BENCH_Z_MAX
    g4 = out.get("g4_iqr_ratio")
    g4_pass = g4 is not None and g4 > _GATE4_IQR_RATIO_MIN
    # Gate 5 reads the Roelofs-debiased ECE (raw - null offset). Falls back
    # to raw ECE if the debiased column is absent (synthetic golden frames).
    g5 = out.get("g5_ece_debiased", out.get("g5_ece"))
    g5_pass = g5 is not None and g5 < _GATE5_ECE_MAX
    out["g1_pass"] = g1_pass
    out["g2_pass"] = g2_pass
    out["g3_pass"] = g3_pass
    out["g4_pass"] = g4_pass
    out["g5_pass"] = g5_pass
    out["ship"] = g1_pass and g2_pass and g3_pass and g4_pass and g5_pass
    return out


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
        pred_col: Predicted-mean column to evaluate (``"EV"`` by default).

    Returns:
        Dict carrying every gate measurement, oracle bound, per-gate
        ``g{1..5}_pass`` flag, and the overall ``ship`` verdict. Excludes
        ``league``, ``market``, ``strategy``; renames ``n_rows`` →
        ``n_validation``.
    """
    market_stem = market.replace(" ", "-")
    decode_strategy = _resolve_decode_strategy(league, market_stem)
    row = apply_thresholds(
        gate_row(
            test_set_df,
            pred_col,
            league=league,
            market=market_stem,
            strategy=strategy,
            decode_strategy=decode_strategy,
        )
    )
    gate_only = {k: v for k, v in row.items() if k not in _GATE_ROW_IDENTITY_KEYS}
    if "n_rows" in gate_only:
        gate_only["n_validation"] = gate_only.pop("n_rows")
    return gate_only


def write_gate_scorecard(rows: list[dict[str, object]], out_path: Path) -> pd.DataFrame:
    """Write the per-cell five-gate scorecard snapshot to CSV, one row per cell.

    Overwrites ``out_path`` each call — a *snapshot* of the latest audit (the five
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
    if row.get("g1_brier_diff_mean") is None:
        head += "  (no Odds; G1 auto-pass)"
    if "ship" in row:
        if row["ship"]:
            head += "  [SHIP]"
        else:
            failed = [g for g in ("g1", "g2", "g3", "g4", "g5") if not row.get(f"{g}_pass", True)]
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


def _supersede_paired_brier_ci(
    b_df: pd.DataFrame, c_df: pd.DataFrame
) -> tuple[int, float, float, float] | None:
    """S2 — paired Brier CI on the row-aligned intersection of two test sets.

    ``d_i = brier_baseline_i - brier_candidate_i`` per shared event. Positive ``d``
    ⇒ candidate has lower Brier; the gate fires when the 95% percentile CI of
    ``mean(d)`` strictly excludes 0 from below (``ci_lo > 0``). Returns
    ``(n_shared, mean, ci_lo, ci_hi)`` or ``None`` if either frame lacks the
    requisite columns or the intersection is empty.
    """
    if _calibration_inputs(b_df) is None or _calibration_inputs(c_df) is None:
        return None
    shared = b_df.index.intersection(c_df.index)
    if len(shared) == 0:
        return None
    b_aligned = b_df.loc[shared]
    c_aligned = c_df.loc[shared]
    p_b = np.clip(b_aligned["P"].astype(float).to_numpy(), _PROBA_CLIP, 1.0 - _PROBA_CLIP)
    p_c = np.clip(c_aligned["P"].astype(float).to_numpy(), _PROBA_CLIP, 1.0 - _PROBA_CLIP)
    # Outcome ``y`` is event-level (Result vs Line), so the candidate frame's view
    # is authoritative — the supersede test is "did the new model do better on the
    # same events". If the two frames disagree on (Result, Line) for shared rows
    # the alignment is moot; bias the comparison toward the candidate's labels.
    y = (
        c_aligned["Result"].astype(float).to_numpy() >= c_aligned["Line"].astype(float).to_numpy()
    ).astype(float)
    brier_b = (p_b - y) ** 2
    brier_c = (p_c - y) ** 2
    d = brier_b - brier_c
    rng = np.random.default_rng(_SUPERSEDE_S2_SEED)
    mean, lo, hi = _bootstrap_mean_ci(d, rng)
    return len(shared), mean, lo, hi


def _test_set_to_bet_frame(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """Adapt a test-set frame to the per-bet schema ``simulate_kelly_all`` expects.

    For each event the model picks the **EV side** (``over`` if ``pred >= Line`` else
    ``under``). The bet's ``Model P`` is the model's probability on that side;
    ``Boost`` is the decimal-odds payout ``1 / clip(book_p)`` so
    :func:`sportstradamus.strategies.profit_sim.compute_payout` (under
    ``Platform="Sleeper"``, which returns ``boost``) yields a payout > 1 and the
    sim's Kelly branch doesn't early-exit. Synthetic monotonic dates make each
    event its own "day" so the resulting return series has per-event resolution.
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
            "Model P": p_model,
            "Model": p_model,
            "K": p_model * payout_decimal,
            "Books": 1.0,
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

    shared = b_df.index.intersection(c_df.index)
    if len(shared) == 0:
        return None
    b_bets = _test_set_to_bet_frame(b_df.loc[shared], pred_col)
    c_bets = _test_set_to_bet_frame(c_df.loc[shared], pred_col)
    if b_bets.empty or c_bets.empty:
        return None
    b_sim = simulate_kelly_all(
        b_bets, prob_col="Model P", initial_bankroll=_SUPERSEDE_S3_INITIAL_BANKROLL
    )
    c_sim = simulate_kelly_all(
        c_bets, prob_col="Model P", initial_bankroll=_SUPERSEDE_S3_INITIAL_BANKROLL
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
    c_row = apply_thresholds(
        gate_row(c_df, pred_col, league=league, market=market, strategy=strategy)
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
    """Pretty-print the decile table to stdout."""
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
    exploded = explode_offers(history)
    if exploded.empty:
        return pd.DataFrame(columns=list(_LIVE_EVAL_COLUMNS))
    cutoff = pd.Timestamp(datetime.now(UTC).date()) - pd.Timedelta(days=window_days)
    exploded["_date"] = pd.to_datetime(exploded["Date"], errors="coerce")
    mask = (
        (exploded["League"] == league)
        & (exploded["Market"] == market)
        & exploded["Actual"].notna()
        & exploded["_date"].notna()
        & (exploded["_date"] >= cutoff)
    )
    subset = exploded.loc[mask].copy()
    if subset.empty:
        return pd.DataFrame(columns=list(_LIVE_EVAL_COLUMNS))

    over_mask = subset["Bet"].eq("Over").to_numpy()
    model_p = subset["Model P"].to_numpy()
    books_p = subset["Books P"].to_numpy()
    out = pd.DataFrame(
        {
            "MeanYr": [
                meanyr_lookup(player, market, date)
                for player, date in zip(subset["Player"], subset["_date"], strict=False)
            ],
            "Result": subset["Actual"].astype(float).to_numpy(),
            "EV": subset["Model EV"].astype(float).to_numpy(),
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
    exploded = explode_offers(history)
    if exploded.empty:
        return []
    settled = exploded[exploded["Actual"].notna()]
    if settled.empty:
        return []
    if league:
        settled = settled[settled["League"] == league]
    if market:
        settled = settled[settled["Market"] == market]
    return sorted({(row.League, row.Market) for row in settled.itertuples(index=False)})


@click.command()
@click.option("--league", default=None, help="Filter test sets by league (e.g. NBA).")
@click.option("--market", default=None, help="Single market stem (requires --league).")
@click.option(
    "--pred-col",
    type=click.Choice(["EV", "Blended_EV"]),
    default=DEFAULT_PRED_COL,
    help="Predicted-mean column to evaluate. EV = raw model (default).",
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
            card = scorecard(
                frame,
                pred_col,
                strategy=live_strategy,
                league=cell_league,
                market=cell_market,
                n_deciles=deciles,
            )
            _print_live_scorecard(card, f"{cell_league}_{cell_market}", pred_col)
            if not no_log:
                append_run_log(card, log_path)
        return

    if baseline or candidate:
        if not (baseline and candidate):
            raise click.UsageError("--baseline and --candidate must be given together.")
        b_df = load_test_set(baseline, pred_col)
        c_df = load_test_set(candidate, pred_col)
        b_row = apply_thresholds(
            gate_row(b_df, pred_col, league="", market="", strategy="baseline")
        )
        c_row = apply_thresholds(gate_row(c_df, pred_col, league="", market="", strategy=strategy))
        click.echo(f"baseline : {baseline.name}")
        _print_table(decile_table(b_df, pred_col, deciles))
        click.echo(_gate_headline(b_row))
        click.echo(f"\ncandidate: {candidate.name}")
        _print_table(decile_table(c_df, pred_col, deciles))
        click.echo(_gate_headline(c_row))
        verdict = supersede_verdict(b_df, c_df, pred_col, strategy=strategy)
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
        # Per-cell training strategy comes from stat_meta (the source of
        # truth for what `meditate` trained the cell with); the CLI
        # `--strategy` value is just the run label written into the row.
        decode_strategy = _resolve_decode_strategy(lg, mkt)
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
