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

import importlib.resources as pkg_resources
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import click
import numpy as np
import pandas as pd

from sportstradamus import data

# Phase-0 ship gate (see plan): a strategy ships only if it cuts top-mean-decile
# MAE by at least this fraction without regressing global MAE beyond the
# tolerance below. Sourced from the attached report's "decision threshold".
MIN_TOP_DECILE_MAE_IMPROVEMENT = 0.05
MAX_GLOBAL_MAE_REGRESSION = 0.01

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
    out = df[[DECILE_COL, ACTUAL_COL, pred_col]].copy()
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
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
    work["decile"] = pd.qcut(
        work[DECILE_COL].rank(method="first"), n_deciles, labels=False
    )
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
    """
    meanyr = df[DECILE_COL].to_numpy()
    actual = df[ACTUAL_COL].to_numpy()
    pred = df[pred_col].to_numpy()

    table = decile_table(df, pred_col, n_deciles)
    top = table.iloc[-1]
    top_mask = df[DECILE_COL] >= df[DECILE_COL].quantile(1 - 1 / n_deciles)

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
    )


def verdict(baseline: Scorecard, candidate: Scorecard) -> tuple[bool, str]:
    """Apply the Phase-0 ship gate comparing a candidate to a baseline.

    Returns:
        ``(ship, reason)``. ``ship`` is True only if top-decile MAE improves by
        at least :data:`MIN_TOP_DECILE_MAE_IMPROVEMENT` and global MAE does not
        regress by more than :data:`MAX_GLOBAL_MAE_REGRESSION`.
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
    return True, (
        f"SHIP: top-decile MAE {top_impr:+.1%}, global MAE {global_reg:+.1%}"
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
        f"{'decile':>6} {'meanyr':>8} {'n':>6} {'mae':>8} "
        f"{'bias':>8} {'pred':>8} {'actual':>8}"
    )
    for _, r in table.iterrows():
        click.echo(
            f"{int(r['decile']):>6} {r['meanyr']:>8.2f} {int(r['n']):>6} "
            f"{r['mae']:>8.3f} {r['bias']:>+8.3f} {r['pred_mean']:>8.2f} "
            f"{r['actual_mean']:>8.2f}"
        )


def _resolve_test_sets(
    test_sets_dir: Path, league: str | None, market: str | None
) -> list[Path]:
    """Resolve the CSV files to evaluate from --league/--market filters."""
    paths = sorted(test_sets_dir.glob("*.csv"))
    if league:
        paths = [p for p in paths if p.stem.startswith(f"{league}_")]
    if market:
        paths = [p for p in paths if p.stem == f"{league}_{market}".replace(" ", "-")]
    return paths


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
) -> None:
    """Score compression on dumped test sets, or diff two strategies."""
    log_path = Path(str(RUN_LOG_PATH))

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
        raise click.UsageError(
            f"No test_sets directory at {resolved_dir}. Run `meditate` first."
        )
    paths = _resolve_test_sets(resolved_dir, league, market)
    if not paths:
        raise click.UsageError("No matching test-set CSVs found.")

    for path in paths:
        stem = path.stem
        lg, _, mkt = stem.partition("_")
        df = load_test_set(path, pred_col)
        card = scorecard(
            df, pred_col, strategy=strategy, league=lg, market=mkt, n_deciles=deciles
        )
        click.echo(f"\n=== {stem}  ({pred_col}, n={card.n_rows}) ===")
        _print_table(decile_table(df, pred_col, deciles))
        click.echo(
            f"global_mae={card.global_mae:.3f}  "
            f"top_decile_mae={card.top_decile_mae:.3f}  "
            f"top_decile_bias={card.top_decile_bias:+.3f}  "
            f"compression_ratio={card.compression_ratio:.3f} "
            f"(top {card.top_decile_compression_ratio:.3f})"
        )
        click.echo(
            f"result_meanyr_corr={card.result_meanyr_corr:+.3f}  "
            f"pred_meanyr_corr={card.pred_meanyr_corr:+.3f}"
        )
        if scatter:
            out = SCATTER_DIR / f"compression_{stem}_{pred_col}.png"
            write_scatter(df, pred_col, out, f"{stem} — {strategy}")
            click.echo(f"scatter: {out}")
        if not no_log:
            append_run_log(card, log_path)


if __name__ == "__main__":
    main()
