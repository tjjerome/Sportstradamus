"""Empirical calibration audit for beam-search parlay candidates.

Reads ``data/parlay_hist.dat`` (written by ``prophecize`` and resolved by
``reflect``), filters to the last ``--days`` (default 90) of resolved
parlays, recovers each parlay's predicted joint probability from
``Model EV`` and the platform payout table, buckets by predicted
probability deciles, and computes the empirical hit rate per decile from
the resolved ``Legs`` / ``Misses`` columns.

A "hit" is a parlay where every counted leg covered (``Misses == 0`` and
``Legs == Bet Size``) — pushed legs are excluded by ``check_bet`` in
:mod:`sportstradamus.analysis` before the row is counted.

Outputs (timestamped with today's date):

* ``docs/PARLAY_CALIBRATION_<YYYY-MM-DD>.png`` — predicted vs empirical
  hit rate by decile, with a 45-degree reference line.
* ``docs/PARLAY_CALIBRATION_<YYYY-MM-DD>.csv`` — decile table with the
  predicted-probability bin edges, mean predicted probability, empirical
  hit rate, sample count, and Wilson 95% confidence bounds.

If ``parlay_hist.dat`` is missing or contains zero resolved rows in the
window, the script writes an empty CSV plus a placeholder PNG explaining
the data gap, so the audit document can still reference the artifacts.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
from datetime import date, datetime, timedelta
from pathlib import Path

import click
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sportstradamus import data

# Underdog/PrizePicks per-size payout multipliers used inside
# ``beam_search_parlays`` to convert a joint probability into ``Model EV``.
# Indexed by ``bet_size - 2`` (i.e. a 2-leg parlay reads index 0).
PAYOUT_TABLE: dict[str, list[float]] = {
    "Underdog": [3.5, 6.5, 10.9, 20.2, 39.9],
    "PrizePicks": [3.0, 5.3, 10.0, 20.8, 38.8],
    "Sleeper": [1.0, 1.0, 1.0, 1.0, 1.0],
    "ParlayPlay": [1.0, 1.0, 1.0, 1.0, 1.0],
    "Chalkboard": [1.0, 1.0, 1.0, 1.0, 1.0],
}

# Mirrors the ``np.clip(payout_base * boost, 1, 100)`` line in
# ``beam_search_parlays`` so the inverse computation stays in sync.
PAYOUT_FLOOR: float = 1.0
PAYOUT_CEIL: float = 100.0

# Number of equal-frequency buckets for the calibration plot. Ten matches
# the reliability-diagram convention used elsewhere in the repo.
N_DECILES: int = 10

# Minimum resolved parlays per decile before we plot a point. Below this,
# the empirical hit rate is too noisy to interpret.
MIN_DECILE_SAMPLES: int = 20

DOCS_DIR: Path = Path(__file__).resolve().parents[3] / "docs"


def _recovered_payout(row: pd.Series) -> float:
    """Reconstruct ``clip(payout_base * boost, 1, 100)`` for a stored row."""
    platform = row.get("Platform")
    bet_size = row.get("Bet Size")
    boost = row.get("Boost", 1.0)
    if platform not in PAYOUT_TABLE or pd.isna(bet_size):
        return float("nan")
    idx = int(bet_size) - 2
    table = PAYOUT_TABLE[platform]
    if idx < 0 or idx >= len(table):
        return float("nan")
    return float(np.clip(table[idx] * float(boost or 1.0), PAYOUT_FLOOR, PAYOUT_CEIL))


def _recover_joint_prob(row: pd.Series) -> float:
    """Recover the copula-based predicted joint probability.

    ``beam_search_parlays`` stores ``Model EV = payout * mvn.cdf(...)``
    where ``payout = clip(payout_base * boost, 1, 100)``. Inverting gives
    ``joint_prob = Model EV / payout``.
    """
    payout = _recovered_payout(row)
    model_ev = row.get("Model EV")
    if pd.isna(payout) or payout == 0 or pd.isna(model_ev):
        return float("nan")
    return float(model_ev) / payout


def _recover_indep_joint_prob(row: pd.Series) -> float:
    """Recover the independence-assumption joint probability.

    ``Indep P = prod(p_model_legs) * payout`` is also stored on each row.
    Comparing this against the copula-based ``Joint P`` shows whether the
    correlation modeling is improving or hurting calibration.
    """
    payout = _recovered_payout(row)
    indep_p = row.get("Indep P")
    if pd.isna(payout) or payout == 0 or pd.isna(indep_p):
        return float("nan")
    return float(indep_p) / payout


def _wilson_bounds(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence bounds for a binomial proportion."""
    if n == 0:
        return float("nan"), float("nan")
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _load_resolved_parlays(window_start: date) -> pd.DataFrame:
    """Load ``parlay_hist.dat`` and filter to resolved rows in the window.

    A resolved row has non-null ``Legs`` and ``Misses`` (filled by
    ``reflect``). Rows whose ``Date`` is before ``window_start`` or whose
    platform/bet-size is unknown are dropped.
    """
    path = pkg_resources.files(data) / "parlay_hist.dat"
    if not Path(str(path)).is_file():
        return pd.DataFrame()

    df = pd.read_pickle(path)
    if df.empty:
        return df

    required = {"Date", "Platform", "Bet Size", "Boost", "Model EV", "Legs", "Misses"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"parlay_hist.dat missing columns: {sorted(missing)}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    df = df.loc[df["Date"].notna() & (df["Date"] >= window_start)]
    df = df.loc[df["Legs"].notna() & df["Misses"].notna()]
    df["Joint P"] = df.apply(_recover_joint_prob, axis=1)
    df = df.loc[df["Joint P"].between(0.0, 1.0, inclusive="both")]
    if "Indep P" in df.columns:
        df["Indep Joint P"] = df.apply(_recover_indep_joint_prob, axis=1)
    else:
        df["Indep Joint P"] = float("nan")
    df["Hit"] = ((df["Misses"] == 0) & (df["Legs"] == df["Bet Size"])).astype(int)
    return df


def _bucket_by_decile(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket parlays into ``N_DECILES`` equal-frequency bins by ``Joint P``."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "decile",
                "p_low",
                "p_high",
                "predicted",
                "empirical",
                "n",
                "hits",
                "ci_low",
                "ci_high",
            ]
        )

    quantiles = np.linspace(0, 1, N_DECILES + 1)
    edges = np.unique(np.quantile(df["Joint P"], quantiles))
    if len(edges) < 2:
        edges = np.array([df["Joint P"].min(), df["Joint P"].max() + 1e-9])

    df = df.copy()
    df["decile"] = pd.cut(df["Joint P"], bins=edges, include_lowest=True, labels=False)

    rows = []
    for decile, grp in df.groupby("decile", dropna=True):
        n = len(grp)
        hits = int(grp["Hit"].sum())
        ci_low, ci_high = _wilson_bounds(hits, n)
        indep_pred = (
            float(grp["Indep Joint P"].mean()) if "Indep Joint P" in grp.columns else float("nan")
        )
        rows.append(
            {
                "decile": int(decile) + 1,
                "p_low": float(edges[int(decile)]),
                "p_high": float(edges[int(decile) + 1]),
                "predicted": float(grp["Joint P"].mean()),
                "predicted_indep": indep_pred,
                "empirical": hits / n,
                "n": n,
                "hits": hits,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    return pd.DataFrame(rows).sort_values("decile").reset_index(drop=True)


def _per_size_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate predicted-vs-empirical at each ``Bet Size`` level.

    Localizes calibration damage. The audit revealed top-decile overconfidence
    is concentrated where ``mvn.cdf`` is least reliable (large ``Bet Size``
    with tight correlations), so per-size MACE separates "code path noise"
    from "single-leg model miscalibration".
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "bet_size",
                "n",
                "hits",
                "predicted_copula",
                "predicted_indep",
                "empirical",
                "gap_copula",
                "gap_indep",
            ]
        )

    rows = []
    for bet_size, grp in df.groupby("Bet Size", dropna=True):
        n = len(grp)
        hits = int(grp["Hit"].sum())
        emp = hits / n
        pred_cop = float(grp["Joint P"].mean())
        pred_ind = (
            float(grp["Indep Joint P"].mean()) if "Indep Joint P" in grp.columns else float("nan")
        )
        rows.append(
            {
                "bet_size": int(bet_size),
                "n": n,
                "hits": hits,
                "predicted_copula": pred_cop,
                "predicted_indep": pred_ind,
                "empirical": emp,
                "gap_copula": pred_cop - emp,
                "gap_indep": pred_ind - emp,
            }
        )
    return pd.DataFrame(rows).sort_values("bet_size").reset_index(drop=True)


def _plot_calibration(buckets: pd.DataFrame, png_path: Path, n_total: int) -> None:
    """Write the predicted-vs-empirical reliability diagram to ``png_path``."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="perfect")

    plottable = buckets.loc[buckets["n"] >= MIN_DECILE_SAMPLES]
    if not plottable.empty:
        ax.errorbar(
            plottable["predicted"],
            plottable["empirical"],
            yerr=[
                plottable["empirical"] - plottable["ci_low"],
                plottable["ci_high"] - plottable["empirical"],
            ],
            fmt="o",
            capsize=3,
            color="C0",
            label=f"observed ({len(plottable)} deciles >= {MIN_DECILE_SAMPLES})",
        )
        for _, row in plottable.iterrows():
            ax.annotate(
                f"n={int(row['n'])}",
                (row["predicted"], row["empirical"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    else:
        ax.text(
            0.5,
            0.5,
            "no decile met the minimum sample threshold\n"
            f"(total resolved parlays in window: {n_total})",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
            color="firebrick",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted joint probability (decile mean)")
    ax.set_ylabel("Empirical hit rate")
    ax.set_title(f"Parlay calibration audit — n={n_total} resolved parlays")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


@click.command()
@click.option(
    "--days",
    type=int,
    default=90,
    show_default=True,
    help="Window size in days, ending today.",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=DOCS_DIR,
    show_default=True,
    help="Directory to write PNG and CSV outputs.",
)
def main(days: int, out_dir: Path) -> None:
    """Run the parlay-calibration audit and write artifacts to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().date()
    window_start = today - timedelta(days=days)
    stamp = today.strftime("%Y-%m-%d")
    png_path = out_dir / f"PARLAY_CALIBRATION_{stamp}.png"
    csv_path = out_dir / f"PARLAY_CALIBRATION_{stamp}.csv"
    by_size_path = out_dir / f"PARLAY_CALIBRATION_BY_SIZE_{stamp}.csv"

    df = _load_resolved_parlays(window_start)
    n_total = len(df)
    buckets = _bucket_by_decile(df)
    by_size = _per_size_summary(df)

    buckets.to_csv(csv_path, index=False)
    by_size.to_csv(by_size_path, index=False)
    _plot_calibration(buckets, png_path, n_total)

    if n_total == 0:
        click.echo(
            f"No resolved parlays found in window [{window_start} .. {today}]. "
            f"Wrote placeholder artifacts to {out_dir}."
        )
        return

    overall = df["Hit"].mean()
    mean_pred = df["Joint P"].mean()
    indep_mean = df["Indep Joint P"].mean()
    click.echo(
        f"n={n_total} resolved parlays | mean copula={mean_pred:.3f} "
        f"| mean indep={indep_mean:.3f} | empirical={overall:.3f} "
        f"| gap_copula={mean_pred - overall:+.3f} | gap_indep={indep_mean - overall:+.3f}"
    )
    if not by_size.empty:
        click.echo("Per bet size:")
        for _, row in by_size.iterrows():
            click.echo(
                f"  size={int(row['bet_size'])} n={int(row['n']):>6} "
                f"copula={row['predicted_copula']:.3f} "
                f"indep={row['predicted_indep']:.3f} "
                f"emp={row['empirical']:.3f} "
                f"gap_cop={row['gap_copula']:+.3f} gap_ind={row['gap_indep']:+.3f}"
            )
    click.echo(f"Wrote {png_path}, {csv_path}, and {by_size_path}.")


if __name__ == "__main__":
    main()
