#!/usr/bin/env python3
"""Offline regression-toward-the-mean diagnostic for trained LightGBMLSS models.

Reads the ``data/test_sets/{LEAGUE}_{market}.csv`` artifacts that ``meditate``
already dumps (no network, no model reload) and quantifies prediction
compression: the structural GBDT bias where high-mean players are
under-predicted and low-mean players over-predicted.

Primary signal is the per-player-mean decile table — rows binned by ``MeanYr``
(player season-to-date mean), reporting MAE and signed bias per decile. A
monotone negative bias rising across the top deciles is the compression
signature. The compression ratio ``std(pred) / std(actual)`` summarizes it in
one number (1.0 = no compression; Wheeler 2012 measured ~7.7x on raw NBA PPG).

Two modes:
  * single  — score one or more test sets: a per-cell five-gate scorecard (with an
              "oracle" bound) to data/tier0_scorecard.csv, plus the compression run log.
  * diff    — compare a candidate test set against a baseline, reporting the gate
              metrics for both (the supersede ship verdict lands in a later phase).

Usage
-----
  poetry run python3 -m sportstradamus.scripts.compression_eval --league NBA
  poetry run python3 -m sportstradamus.scripts.compression_eval \
      --league NBA --market PTS --strategy ratio_baseline --scatter
  poetry run python3 -m sportstradamus.scripts.compression_eval \
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

from sportstradamus import data
from sportstradamus.analysis import explode_offers
from sportstradamus.helpers.io import read_history
from sportstradamus.training.markets import ALL_MARKETS

# ---------------------------------------------------------------------------
# Ship gates (see docs/ship_gate.md). The promotion lifecycle is a 2x2:
# (set first baseline | supersede incumbent) x (research->devel offline |
# devel->main live).
#
#   * research -> devel, set baseline: the FIVE offline gates computed here —
#     Gate 1 Brier-vs-book paired bootstrap, Gates 2/3 star/bench bias-vs-spread
#     match (denominator = segment σ, NOT σ/sqrt(N) — SE collapses on large-N
#     low-variance bench segments), Gate 4 IQR spread, Gate 5 equal-mass ECE.
#     This module is MEASUREMENT-ONLY for these five: the per-cell metrics (plus
#     an "oracle" bound) go to tier0_scorecard.csv with no pass/fail. Thresholds
#     (k, the Gate-1 CI rule, the IQR floor, the ECE ceiling) are chosen after
#     reading the numbers, then the overall verdict is wired back in. Cells with
#     no book Odds leave Gate 1 blank; the ship convention is that a blank Gate 1
#     auto-passes — no book to beat, model wins by default. Gate 5 (model-only
#     calibration) does NOT use Odds, so it still computes for those cells; Gate 5
#     blank means "couldn't compute" (no P or no Line), not auto-pass.
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
# percentile bounds. Seed matches zinb_routing_diagnostics._DEFAULT_SEED.
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
#   G5 ece    < 0.075: 10-bin equal-mass ECE under 7.5%
_GATE1_CI_HI_MAX: float = 0.0
_GATE2_STAR_Z_MAX: float = 0.5
_GATE3_BENCH_Z_MAX: float = 0.5
_GATE4_IQR_RATIO_MIN: float = 0.5
_GATE5_ECE_MAX: float = 0.075

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

# Per-cell gate scorecard SNAPSHOT (not the append-only run log). write_gate_scorecard
# overwrites this on every full audit with one row per evaluated cell: the five gate
# metrics for the model alongside an "oracle" column set (model = the true score
# exactly) that bounds each gate. Measurement-only — no pass/fail until thresholds
# are set. Readable as plain CSV without re-running the audit.
TIER0_SCORECARD_PATH = pkg_resources.files(data) / "training" / "tier0_scorecard.csv"

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
        Frame with ``MeanYr``, ``Result`` and the prediction column, rows with
        non-finite values in any of the three dropped.

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
    out = df[sorted(required | optional)].copy()
    # Filter non-finite rows on required columns only — missing P/Odds/Line rows
    # are filtered locally inside _brier_skill_score so older CSVs that lack
    # those columns still pass the harness.
    required_view = out[list(required)].replace([np.inf, -np.inf], np.nan)
    out = out[required_view.notna().all(axis=1)]
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
    bounds. Sibling of ``zinb_routing_diagnostics._bootstrap_ci``, specialized to the
    mean of an already-computed per-event statistic (Gate 1 here, supersede S2 later).
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


def _gate4_iqr_spread(actual: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    """Gate 4 — IQR spread (compression) over the full population.

    Returns ``(iqr_pred, iqr_true, ratio)`` with ``ratio = iqr_pred / iqr_true``
    (1.0 = no compression, <1 = predictions compressed). IQR-robust sibling of the
    std-based :func:`_compression_ratio`.
    """
    iqr_pred = _iqr(pred)
    iqr_true = _iqr(actual)
    ratio = iqr_pred / iqr_true if iqr_true > 0 else float("nan")
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


def gate_row(
    df: pd.DataFrame, pred_col: str, *, league: str, market: str, strategy: str
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

    # Gate 4 — IQR spread; the oracle (pred = actual) gives ratio 1.0.
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
        g5_ece = g5_ece_o = None
    else:
        p_model_c, y_c = cal_in
        g5_ece = _gate5_ece_equal_mass(p_model_c, y_c)
        g5_ece_o = _gate5_ece_equal_mass(y_c, y_c)

    def r(v: float | None) -> float | None:
        """Round to 4 dp; map None / non-finite to a blank CSV cell."""
        if v is None or not np.isfinite(v):
            return None
        return round(float(v), 4)

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
    g5 = out.get("g5_ece")
    g5_pass = g5 is not None and g5 < _GATE5_ECE_MAX
    out["g1_pass"] = g1_pass
    out["g2_pass"] = g2_pass
    out["g3_pass"] = g3_pass
    out["g4_pass"] = g4_pass
    out["g5_pass"] = g5_pass
    out["ship"] = g1_pass and g2_pass and g3_pass and g4_pass and g5_pass
    return out


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
        f"G4 iqr_ratio {f('g4_iqr_ratio', '.3f')}  G5 ece {f('g5_ece', '.4f')}"
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
# RATIOS (sharpe_new > sharpe_current), so absolute dollars are immaterial.
_SUPERSEDE_S3_INITIAL_BANKROLL: float = 1000.0


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
        c_aligned["Result"].astype(float).to_numpy()
        >= c_aligned["Line"].astype(float).to_numpy()
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
    sub = df[["P", "Odds", "Line", ACTUAL_COL, pred_col]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
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
    base = pd.Timestamp("2026-01-01")
    dates = [(base + pd.Timedelta(days=int(i))).date() for i in range(n)]
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


def _supersede_paired_sharpe(
    b_df: pd.DataFrame, c_df: pd.DataFrame, pred_col: str
) -> tuple[float, float] | None:
    """S3 — paired Sharpe from a Kelly-all sim on the shared events.

    Returns ``(sharpe_baseline, sharpe_candidate)`` or ``None`` when either
    frame can't be adapted to a bet frame (no ``Odds`` column or empty
    intersection).
    """
    # Imported here to keep ``compression_eval``'s import surface minimal — the
    # supersede path is the only one that needs the profit-sim lib.
    from sportstradamus.strategies.profit_sim import simulate_kelly_all, summarize_runs

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
    s_b = summarize_runs(b_sim, _SUPERSEDE_S3_INITIAL_BANKROLL)["sharpe"]
    s_c = summarize_runs(c_sim, _SUPERSEDE_S3_INITIAL_BANKROLL)["sharpe"]
    return float(s_b), float(s_c)


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
        out.update(s3_sharpe_baseline=None, s3_sharpe_candidate=None, s3_pass=False)
    else:
        sb, sc = s3
        out.update(s3_sharpe_baseline=sb, s3_sharpe_candidate=sc, s3_pass=sc > sb)

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
        f"{f('s3_sharpe_candidate', '.3f')} "
        f"→ {'PASS' if v.get('s3_pass') else 'FAIL'}",
        f"[{'SUPERSEDE' if v.get('ship') else 'HOLD'}]",
    ]
    return "  ".join(parts)


def append_run_log(card: Scorecard, log_path: Path) -> None:
    """Append a scorecard row to the cross-session run log CSV."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([asdict(card)])
    header = not log_path.exists()
    row.to_csv(log_path, mode="a", header=header, index=False)


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
    fig, ax = plt.subplots(figsize=(7, 7))
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
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _print_table(table: pd.DataFrame) -> None:
    """Pretty-print the decile table to stdout."""
    click.echo(
        f"{'decile':>6} {'meanyr':>8} {'n':>6} {'mae':>8} " f"{'bias':>8} {'pred':>8} {'actual':>8}"
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
    help="Gate scorecard CSV path (default data/tier0_scorecard.csv; only a full audit auto-writes).",
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
        b_row = apply_thresholds(gate_row(b_df, pred_col, league="", market="", strategy="baseline"))
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
        row = apply_thresholds(gate_row(df, pred_col, league=lg, market=mkt, strategy=strategy))
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
    # the canonical scorecard, so a filtered run can't clobber it down to a subset; a
    # filtered run still writes when --scorecard-out is given explicitly.
    if write_scorecard and rows:
        if scorecard_out is not None:
            sc_path: Path | None = scorecard_out
        elif league is None and market is None:
            sc_path = Path(str(TIER0_SCORECARD_PATH))
        else:
            sc_path = None
        if sc_path is not None:
            sc_df = write_gate_scorecard(rows, sc_path)
            click.echo(f"\nGate scorecard ({len(sc_df)} cells): {sc_path}")
            _print_breadth_rollup(sc_df)


if __name__ == "__main__":
    main()
