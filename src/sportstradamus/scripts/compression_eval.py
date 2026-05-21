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
  * single  — score one or more test sets, append a scorecard to the run log.
  * diff    — compare a candidate test set against a baseline and emit a
              ship/kill verdict against the Phase-0 threshold.

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

# Phase-0 ship gate (see plan): a strategy ships only if it cuts top-mean-decile
# MAE by at least this fraction without regressing global MAE beyond the
# tolerance below. Sourced from the attached report's "decision threshold".
MIN_TOP_DECILE_MAE_IMPROVEMENT = 0.05
MAX_GLOBAL_MAE_REGRESSION = 0.01

# Brier-skill third gate (see CLAUDE.md "Kelly & blending" and the plan).
# A strategy ships only if its brier_skill_score does not regress vs the
# baseline (any worsening = KILL, even if the MAE gates pass).
MAX_BRIER_SKILL_REGRESSION: float = 0.0

# Bottom-decile over-prediction gate (Universal decision threshold condition 4 /
# Gate 1). Added after the §7a pre-check found top-decile MAE "wins" that were
# really low-volume over-prediction (FG3M bottom decile predicted ~3.4x actual).
# A candidate KILLs if its bottom-mean-decile signed bias is more positive than
# the baseline's, or if its magnitude exceeds this fraction of that decile's
# empirical mean — floored so low-mean count cells aren't held to an impossibly
# tight bound. Tolerances are tunable; the direction (a top-decile win must not
# be financed by bottom-decile over-prediction) is fixed.
BOTTOM_DECILE_BIAS_MAGNITUDE_FRAC: float = 0.10
BOTTOM_DECILE_BIAS_ABS_FLOOR: float = 0.05

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

RUN_LOG_PATH = pkg_resources.files(data) / "compression_eval_log.csv"
SCATTER_DIR = Path("/tmp")

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
    # Bottom-mean-decile signed bias and that decile's empirical (actual) mean,
    # appended last for the same CSV-back-compat reason. They make the plan's
    # bottom-decile ship gate (Universal decision threshold condition 4 / Gate 1)
    # computable directly from the logged scorecard, not just the printed table.
    bottom_decile_bias: float
    bottom_decile_mean: float


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


def _brier_skill_score(df: pd.DataFrame) -> float | None:
    """1 - brier(model_P) / brier(book_over) on the test set, or None if cols absent.

    Requires ``P`` (calibrated model over-probability), ``Odds`` (book
    under-probability so book over = ``1 - Odds``), ``Line`` and
    ``Result``. The binary outcome is ``Result >= Line``. Returns
    ``None`` if any column is missing or has all-NaN values, so older
    CSVs without these columns are handled gracefully.
    """
    needed = {"P", "Odds", "Line"}
    if not needed.issubset(df.columns):
        return None
    sub = df[["P", "Odds", "Line", ACTUAL_COL]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) == 0:
        return None
    y = (sub[ACTUAL_COL] >= sub["Line"]).astype(float).to_numpy()
    p_model = np.clip(sub["P"].to_numpy(), _PROBA_CLIP, 1 - _PROBA_CLIP)
    p_book = np.clip(1.0 - sub["Odds"].to_numpy(), _PROBA_CLIP, 1 - _PROBA_CLIP)
    brier_model = float(np.mean((p_model - y) ** 2))
    brier_book = float(np.mean((p_book - y) ** 2))
    if brier_book <= 0:
        return None
    return 1.0 - brier_model / brier_book


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
        including ``bottom_decile_bias`` and ``bottom_decile_mean`` for the
        bottom-mean-decile ship gate.
    """
    meanyr = df[DECILE_COL].to_numpy()
    actual = df[ACTUAL_COL].to_numpy()
    pred = df[pred_col].to_numpy()

    table = decile_table(df, pred_col, n_deciles)
    top = table.iloc[-1]
    bottom = table.iloc[0]
    top_mask = df[DECILE_COL] >= df[DECILE_COL].quantile(1 - 1 / n_deciles)
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
        top_decile_compression_ratio=_compression_ratio(
            actual[top_mask.to_numpy()], pred[top_mask.to_numpy()]
        ),
        pred_meanyr_corr=_corr(meanyr, pred - meanyr),
        result_meanyr_corr=_corr(meanyr, actual - meanyr),
        brier_skill_score=brier_skill,
        bottom_decile_bias=float(bottom["bias"]),
        bottom_decile_mean=float(bottom["actual_mean"]),
    )


def verdict(baseline: Scorecard, candidate: Scorecard) -> tuple[bool, str]:
    """Apply the offline ship gate comparing a candidate to a baseline.

    Returns:
        ``(ship, reason)``. ``ship`` is True only if all four gate conditions
        hold: (1) top-decile MAE improves by at least
        :data:`MIN_TOP_DECILE_MAE_IMPROVEMENT`; (2) global MAE does not regress
        by more than :data:`MAX_GLOBAL_MAE_REGRESSION`; (3) ``brier_skill_score``
        does not regress (skipped if either card lacks it); (4) the bottom-mean-
        decile signed bias is not more positive than the baseline's and its
        magnitude stays within :data:`BOTTOM_DECILE_BIAS_MAGNITUDE_FRAC` of that
        decile's empirical mean (floored at :data:`BOTTOM_DECILE_BIAS_ABS_FLOOR`).
        Condition (4) blocks a top-decile win bought by low-volume
        over-prediction — the §7a failure mode.
    """
    if baseline.top_decile_mae == 0:
        return False, "baseline top-decile MAE is zero; cannot compute improvement"
    top_impr = (baseline.top_decile_mae - candidate.top_decile_mae) / baseline.top_decile_mae
    global_reg = (candidate.global_mae - baseline.global_mae) / baseline.global_mae

    if top_impr < MIN_TOP_DECILE_MAE_IMPROVEMENT:
        return False, (
            f"KILL: top-decile MAE improved {top_impr:+.1%} "
            f"(need >= {MIN_TOP_DECILE_MAE_IMPROVEMENT:.0%})"
        )
    if global_reg > MAX_GLOBAL_MAE_REGRESSION:
        return False, (
            f"KILL: global MAE regressed {global_reg:+.1%} "
            f"(max {MAX_GLOBAL_MAE_REGRESSION:.0%})"
        )
    if baseline.brier_skill_score is not None and candidate.brier_skill_score is not None:
        delta = candidate.brier_skill_score - baseline.brier_skill_score
        if delta < -MAX_BRIER_SKILL_REGRESSION:
            return False, (
                f"KILL: brier_skill_score regressed "
                f"{baseline.brier_skill_score:+.3f} → "
                f"{candidate.brier_skill_score:+.3f} (Δ {delta:+.3f})"
            )
    if candidate.bottom_decile_bias > baseline.bottom_decile_bias:
        return False, (
            f"KILL: bottom-decile bias worsened "
            f"{baseline.bottom_decile_bias:+.3f} → {candidate.bottom_decile_bias:+.3f} "
            f"(low-volume over-prediction must not increase)"
        )
    bias_bound = max(
        BOTTOM_DECILE_BIAS_MAGNITUDE_FRAC * candidate.bottom_decile_mean,
        BOTTOM_DECILE_BIAS_ABS_FLOOR,
    )
    if abs(candidate.bottom_decile_bias) > bias_bound:
        return False, (
            f"KILL: bottom-decile bias {candidate.bottom_decile_bias:+.3f} exceeds "
            f"calibration bound ±{bias_bound:.3f} (10% of decile mean "
            f"{candidate.bottom_decile_mean:.3f}, floor {BOTTOM_DECILE_BIAS_ABS_FLOOR:.2f})"
        )
    return True, (
        f"SHIP: top-decile MAE {top_impr:+.1%}, global MAE {global_reg:+.1%}"
        + (
            f", brier_skill {baseline.brier_skill_score:+.3f} → "
            f"{candidate.brier_skill_score:+.3f}"
            if baseline.brier_skill_score is not None and candidate.brier_skill_score is not None
            else ""
        )
    )


def append_run_log(card: Scorecard, log_path: Path) -> None:
    """Append a scorecard row to the cross-session run log CSV."""
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


def _print_live_scorecard(card: object, stem: str, pred_col: str) -> None:
    """Print the live-window scorecard summary in the same shape as offline mode."""
    click.echo(f"\n=== {stem}  ({pred_col}, n={card.n_rows}) ===")
    click.echo(
        f"strategy={card.strategy}  "
        f"global_mae={card.global_mae:.3f}  "
        f"top_decile_mae={card.top_decile_mae:.3f}  "
        f"top_decile_bias={card.top_decile_bias:+.3f}  "
        f"bottom_decile_bias={card.bottom_decile_bias:+.3f}  "
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
        b_card = scorecard(
            b_df, pred_col, strategy="baseline", league="", market="", n_deciles=deciles
        )
        c_card = scorecard(
            c_df, pred_col, strategy=strategy, league="", market="", n_deciles=deciles
        )
        click.echo(f"baseline : {baseline.name}")
        _print_table(decile_table(b_df, pred_col, deciles))
        click.echo(f"\ncandidate: {candidate.name}")
        _print_table(decile_table(c_df, pred_col, deciles))
        ship, reason = verdict(b_card, c_card)
        click.echo(
            f"\ncompression_ratio  base={b_card.compression_ratio:.3f}  "
            f"cand={c_card.compression_ratio:.3f}"
        )
        click.echo(reason)
        raise SystemExit(0 if ship else 1)

    resolved_dir = test_sets_dir or Path(str(pkg_resources.files(data) / "test_sets"))
    if not resolved_dir.exists():
        raise click.UsageError(f"No test_sets directory at {resolved_dir}. Run `meditate` first.")
    paths = _resolve_test_sets(resolved_dir, league, market)
    if not paths:
        raise click.UsageError("No matching test-set CSVs found.")

    for path in paths:
        stem = path.stem
        lg, _, mkt = stem.partition("_")
        df = load_test_set(path, pred_col)
        card = scorecard(df, pred_col, strategy=strategy, league=lg, market=mkt, n_deciles=deciles)
        click.echo(f"\n=== {stem}  ({pred_col}, n={card.n_rows}) ===")
        _print_table(decile_table(df, pred_col, deciles))
        click.echo(
            f"global_mae={card.global_mae:.3f}  "
            f"top_decile_mae={card.top_decile_mae:.3f}  "
            f"top_decile_bias={card.top_decile_bias:+.3f}  "
            f"bottom_decile_bias={card.bottom_decile_bias:+.3f}  "
            f"compression_ratio={card.compression_ratio:.3f} "
            f"(top {card.top_decile_compression_ratio:.3f})"
        )
        click.echo(
            f"result_meanyr_corr={card.result_meanyr_corr:+.3f}  "
            f"pred_meanyr_corr={card.pred_meanyr_corr:+.3f}"
        )
        if card.brier_skill_score is not None:
            click.echo(f"brier_skill_score={card.brier_skill_score:+.3f}")
        if scatter:
            out = SCATTER_DIR / f"compression_{stem}_{pred_col}.png"
            write_scatter(df, pred_col, out, f"{stem} — {strategy}")
            click.echo(f"scatter: {out}")
        if not no_log:
            append_run_log(card, log_path)


if __name__ == "__main__":
    main()
