"""Run powershap on one (league, market) cell and print the selected features.

After the 2026-05-27 no-filter rewire, this script is a standalone diagnostic
for evaluating powershap as an alternative selection method on a single cell.
It is NOT wired into meditate; production uses the full unfiltered candidate
set per researcher Option C / Akhiat & Touchanti 2024.

Usage:
    poetry run python scripts/powershap_eval.py
    poetry run python scripts/powershap_eval.py --market "receiving tds"
    poetry run python scripts/powershap_eval.py --sweep "0.01,0.05,0.10,0.25,0.50"
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import sys
from pathlib import Path

import click
import lightgbm as lgb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sportstradamus import data  # noqa: E402
from sportstradamus.stats import StatsNBA, StatsNFL, StatsWNBA  # noqa: E402

_STAT_CTORS = {"NBA": StatsNBA, "NFL": StatsNFL, "WNBA": StatsWNBA}

# Mirror the 70/30 temporal split used by the old scouting pass / production
# _step_build_splits, so the powershap fit sees the same training subset
# production would use.
_SCOUTING_TRAIN_FRACTION = 0.7

# Fixed-HP scouting params translated to LGBMRegressor argument names.
_SCOUTING_SKLEARN_PARAMS = dict(
    objective="regression",
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=50,
    colsample_bytree=0.8,
    subsample=0.8,
    subsample_freq=1,
    verbose=-1,
    n_jobs=8,
    n_estimators=200,
)


def _load_matrix(league: str, market: str) -> pd.DataFrame:
    """Read the cached training matrix parquet for one cell."""
    slug = market.replace(" ", "-")
    path = pkg_resources.files(data) / f"training_data/{league}_{slug}.parquet"
    if not Path(str(path)).is_file():
        raise FileNotFoundError(
            f"No cached training matrix at {path}. Run `meditate --league {league}` first."
        )
    M = pd.read_parquet(path)
    M.Date = pd.to_datetime(M.Date, format="mixed")
    if "Player" in M.columns:
        M = M.drop_duplicates(subset=["Player", "Date"], keep="last")
    return M


def _build_xy(
    M: pd.DataFrame, stat_data, market: str
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Reproduce the scouting-pass feature set + temporal split inputs."""
    cols = stat_data.get_stat_columns(market)
    cols = [c for c in cols if c in M.columns]
    if not cols or "Result" not in M.columns:
        raise ValueError("Cell has no candidate columns or no Result in matrix.")

    X = M[cols].copy()
    for c in ("Home", "Player position"):
        if c in X.columns:
            X[c] = X[c].astype("category")
    y = pd.to_numeric(M["Result"], errors="coerce").fillna(0).to_numpy()

    M_sorted = M.sort_values("Date")
    n_train = int(len(M_sorted) * _SCOUTING_TRAIN_FRACTION)
    if n_train < 50:
        raise ValueError(f"Too few training rows: {len(M_sorted)}")
    train_idx = M_sorted.index[:n_train]
    X_train = X.loc[train_idx]
    y_train = y[[M.index.get_loc(i) for i in train_idx]]
    return X_train, y_train, cols


def _run_powershap(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    power_alpha: float,
    power_iterations: int,
    automatic: bool,
) -> list[str]:
    """Fit powershap on the scouting train split, return the selected feature names."""
    from powershap import PowerShap

    model = lgb.LGBMRegressor(**_SCOUTING_SKLEARN_PARAMS)
    ps = PowerShap(
        model=model,
        power_alpha=power_alpha,
        power_req_iterations=power_iterations,
        automatic=automatic,
    )
    cat_cols = [c for c in ("Home", "Player position") if c in X_train.columns]
    X_safe = X_train.copy()
    for c in cat_cols:
        X_safe[c] = X_safe[c].cat.codes
    ps.fit(X_safe, y_train)
    mask = ps.get_support()
    return [c for c, keep in zip(X_train.columns, mask, strict=False) if keep]


def _save_selection(league: str, market: str, selected: list[str]) -> None:
    """Persist the selected feature list to /tmp/powershap_eval/."""
    slug = market.replace(" ", "-")
    out_dir = Path("/tmp/powershap_eval")
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{league}_{slug}_powershap.json").write_text(
        "[\n" + ",\n".join(f'  "{f}"' for f in sorted(selected)) + "\n]\n"
    )
    print(f"\n  Wrote selection list to {out_dir}/{league}_{slug}_powershap.json")


@click.command()
@click.option("--league", default="NFL", show_default=True)
@click.option("--market", default="passing first downs", show_default=True)
@click.option(
    "--power-alpha",
    default=0.01,
    show_default=True,
    help="powershap p-value threshold (lower = stricter = fewer features).",
)
@click.option(
    "--power-iterations",
    default=10,
    show_default=True,
    help="Min powershap re-seeded iterations (each is one full LightGBM fit).",
)
@click.option(
    "--automatic/--no-automatic",
    default=True,
    show_default=True,
    help="Let powershap auto-iterate until power threshold is met.",
)
@click.option(
    "--sweep",
    default="",
    show_default=False,
    help="Comma-separated alpha values to sweep (e.g. '0.01,0.05,0.10,0.25,0.50'). "
    "When set, runs powershap once per alpha and reports feature-count for each.",
)
def main(
    league: str,
    market: str,
    power_alpha: float,
    power_iterations: int,
    automatic: bool,
    sweep: str,
) -> None:
    """Run powershap on one (league, market) cell, print selected features."""
    if league not in _STAT_CTORS:
        raise click.UsageError(f"Unsupported league: {league}")

    print(f"Loading {league} cached gamelogs...")
    stat_data = _STAT_CTORS[league]()
    stat_data.load()
    stat_data.update_player_comps()

    print(f"Loading training matrix for {league} {market}...")
    M = _load_matrix(league, market)
    X_train, y_train, cols = _build_xy(M, stat_data, market)
    print(f"  matrix: {len(M)} rows, {len(cols)} candidate columns")
    print(f"  train split: {len(X_train)} rows")

    alphas = [float(a) for a in sweep.split(",") if a.strip()] if sweep else [power_alpha]
    last_selected: list[str] = []
    for alpha in alphas:
        print(
            f"\nRunning powershap (alpha={alpha}, "
            f"iter={power_iterations}, automatic={automatic})..."
        )
        selected = _run_powershap(X_train, y_train, alpha, power_iterations, automatic)
        print(f"  alpha={alpha}: kept {len(selected)} of {len(cols)} candidates")
        last_selected = selected

    if len(alphas) > 1:
        print("\n--- Sweep summary ---")
        for alpha in alphas:
            # Re-run isn't free; the loop above already printed per-alpha counts.
            pass

    print(f"\nFinal selection ({len(last_selected)} features):")
    for f in sorted(last_selected)[:30]:
        print(f"  {f}")
    if len(last_selected) > 30:
        print(f"  ... and {len(last_selected) - 30} more")

    _save_selection(league, market, last_selected)


if __name__ == "__main__":
    main()
