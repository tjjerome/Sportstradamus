#!/usr/bin/env python3
"""Route count cells to plain-NB / hurdle-NB / Double Poisson before a family build.

WS-3 Phase 0, the §6.6 entry bar. Read-only with respect to the models and the
training pipeline: this script only loads cached training-data parquets, splits
them honestly in time, and prints one routing verdict per (league, market) cell.

The split is chronological by ``Date``, first 70% as training — mirroring
``training/pipeline.py:_step_build_splits``. The pipeline sub-splits the trailing
30% *randomly* into test/validation, so no purely temporal validation set exists;
here that whole 30% window is scored to keep the zero-count test's power.

Two diagnostics per cell, both under a per-player moment-fit Negative Binomial
null (players with fewer than ``_MIN_PLAYER_ROWS`` training rows, and validation
players unseen in training, pool into a global fit):

1. Zero-modification routing (Wilson & Einbeck 2019 style, non-asymptotic): the
   observed validation zero count scored two-sided against its Poisson-binomial
   null over the per-row NB zero probabilities. p >= 0.05 routes ``plain-NB``; a
   significant excess routes ``hurdle-NB``; a significant deficit routes
   ``DP-mandatory`` (no zero-inflation gate can subtract zeros).

2. Dispersion escalation bar for the Double Poisson build: (a) Dunn-Smyth
   randomized-quantile-residual variance < 0.70 under the NB fit, AND (b) a small
   deterministic Poisson-objective LightGBM tracks the realized top decile more
   closely than the NB per-player means. ``bar_pass`` = (a) AND (b).

Dev-only; never ships (devel-ship-curator denylist). Upserts one row per cell to
``data/research/count_family_screen.parquet``.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import json
import logging
import math
from pathlib import Path

import click
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy import stats
from tqdm import tqdm

from sportstradamus import data

logger = logging.getLogger(__name__)

# The three Double Poisson pilots from the WS-3 plan; the default cell set.
_PILOT_CELLS: tuple[tuple[str, str], ...] = (("NBA", "PF"), ("WNBA", "TOV"), ("NHL", "points"))
# stat_meta.json dist labels that mark a count cell (--all-count universe).
_COUNT_DISTS = frozenset({"ZINB", "NegBin", "Negative Binomial"})
_TARGET_COLUMN = "Result"
# Mirrors training/pipeline.py:_TRAIN_FRACTION so the screen trains on the same window
# the pipeline would; everything after the cut is the temporal validation window.
_TRAIN_FRACTION = 0.7
# Below this many training rows a player's moment fit is noise -> pooled global fit.
_MIN_PLAYER_ROWS = 5
# Floor on var - mean so the NB size r stays finite for under-dispersed players.
_VAR_EXCESS_FLOOR = 1e-6
# Moment-fit size r clipped to this range (guards degenerate per-player fits).
_R_MIN, _R_MAX = 0.5, 50.0
# Two-sided significance level for the zero-modification routing test.
_ROUTE_ALPHA = 0.05
# The exact O(n^2) Poisson-binomial DP convolution stays sub-second up to here;
# larger validation sets use the normal approximation with continuity correction.
_EXACT_DP_MAX_N = 5000
# RQR variance below this = the NB fit over-disperses -> Double Poisson territory.
_RQR_VAR_BAR = 0.70
# Bootstrap resamples for the RQR-variance CI (spec minimum 200).
_N_BOOTSTRAP = 200
# 90% two-sided percentile CI.
_RQR_CI_PCT = (5.0, 95.0)
# Fixed seed so the randomized residuals, bootstrap, and GBM reproduce across runs.
_SEED = 1729
# Clip PIT draws away from {0, 1} so norm.ppf stays finite.
_U_EPS = 1e-10
# Small fixed Poisson GBM for the top-decile tracking check (spec: ~200 trees).
_GBM_TREES = 200
# Top decile of realized outcomes.
_TOP_DECILE_Q = 0.9
# Target + book columns: never GBM features (the pipeline excludes them from X too).
_NON_FEATURE_COLUMNS = frozenset({_TARGET_COLUMN, "Line", "Odds", "EV"})

ROUTE_CHOICES: tuple[str, str, str] = ("plain-NB", "hurdle-NB", "DP-mandatory")

REQUIRED_COLUMNS: tuple[str, ...] = (
    "league",
    "market",
    "n_val",
    "zero_obs",
    "zero_null_mean",
    "zero_pvalue",
    "route",
    "rqr_var",
    "rqr_ci_lo",
    "rqr_ci_hi",
    "gbm_top_decile_gap",
    "nb_top_decile_gap",
    "bar_pass",
)


def _data_child(*parts: str) -> Path:
    # data/ is a namespace package (MultiplexedPath); resolve concrete children
    # by traversing before str(), never str() the multiplexed root itself.
    node = pkg_resources.files(data)
    for part in parts:
        node = node / part
    return Path(str(node))


def _matrix_path(league: str, market: str) -> Path:
    # NFL market names carry spaces; the cached parquets hyphenate them.
    return _data_child("training_data", f"{league}_{market.replace(' ', '-')}.parquet")


def count_cells(
    leagues: tuple[str, ...], markets: tuple[str, ...], *, all_count: bool
) -> list[tuple[str, str]]:
    """Resolve the (league, market) cells to screen.

    ``--all-count`` -> every count-dist cell in stat_meta.json; league/market
    filters -> that subset of the count universe; neither -> the three Double
    Poisson pilots. Cells without a cached training matrix are dropped with a
    warning.
    """
    meta = json.loads(_data_child("config", "stat_meta.json").read_text())
    universe = [
        (league, market)
        for league, cells in meta.items()
        for market, info in cells.items()
        if info.get("dist") in _COUNT_DISTS
    ]
    if not all_count:
        if leagues or markets:
            universe = [
                (lg, mk)
                for lg, mk in universe
                if (not leagues or lg in leagues) and (not markets or mk in markets)
            ]
        else:
            universe = list(_PILOT_CELLS)
    for league, market in universe:
        if not _matrix_path(league, market).is_file():
            logger.warning("No cached training matrix for %s %s; skipping.", league, market)
    return [cell for cell in universe if _matrix_path(*cell).is_file()]


def _moment_nb(y: np.ndarray) -> tuple[float, float]:
    """Moment-fit NB as (r, p) with p = mean / (mean + r), so P(0) = (1 - p) ** r."""
    mean = float(y.mean())
    var = float(y.var(ddof=1)) if y.size > 1 else 0.0
    r = float(np.clip(mean**2 / max(var - mean, _VAR_EXCESS_FLOOR), _R_MIN, _R_MAX))
    return r, mean / (mean + r)


def _player_nb_params(train: pd.DataFrame, val_players: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Per-validation-row NB (r, p) arrays from each player's training moment fit."""
    global_rp = _moment_nb(train[_TARGET_COLUMN].to_numpy(dtype=float))
    fits = {
        player: _moment_nb(group.to_numpy(dtype=float))
        for player, group in train.groupby("Player")[_TARGET_COLUMN]
        if group.size >= _MIN_PLAYER_ROWS
    }
    params = np.array([fits.get(player, global_rp) for player in val_players])
    return params[:, 0], params[:, 1]


def poisson_binomial_pvalue(
    zero_probs: np.ndarray, k_obs: int, *, exact: bool | None = None
) -> tuple[float, float]:
    """Two-sided p-value for observing ``k_obs`` zeros under a Poisson-binomial null.

    The null distribution of the total zero count over independent per-row zero
    probabilities is Poisson-binomial: exact DP convolution when the row count is
    <= ``_EXACT_DP_MAX_N`` (force a branch with ``exact``), else a normal
    approximation with continuity correction. Both branches return
    ``min(1, 2 * min(P(X <= k), P(X >= k)))``.

    Returns:
        ``(null_mean, pvalue)``.
    """
    n = len(zero_probs)
    null_mean = float(zero_probs.sum())
    if exact is None:
        exact = n <= _EXACT_DP_MAX_N
    if exact:
        pmf = np.zeros(n + 1)
        pmf[0] = 1.0
        for j, prob in enumerate(zero_probs, start=1):
            pmf[1 : j + 1] = pmf[1 : j + 1] * (1.0 - prob) + pmf[:j] * prob
            pmf[0] *= 1.0 - prob
        p_le = float(pmf[: k_obs + 1].sum())
        p_ge = float(pmf[k_obs:].sum())
    else:
        sd = math.sqrt(float((zero_probs * (1.0 - zero_probs)).sum()))
        p_le = float(stats.norm.cdf((k_obs + 0.5 - null_mean) / sd))
        p_ge = float(stats.norm.sf((k_obs - 0.5 - null_mean) / sd))
    return null_mean, min(1.0, 2.0 * min(p_le, p_ge))


def route_zero_mod(zero_obs: int, null_mean: float, pvalue: float) -> str:
    """Map the zero-count test to a family route.

    Two-sided p >= ``_ROUTE_ALPHA`` -> ``plain-NB`` (no zero modification needed).
    Significant excess zeros -> ``hurdle-NB`` (a gate can add zeros). Significant
    deficit -> ``DP-mandatory``: no zero-inflation gate can *subtract* zeros, only
    a family change can.
    """
    if pvalue >= _ROUTE_ALPHA:
        return "plain-NB"
    return "hurdle-NB" if zero_obs > null_mean else "DP-mandatory"


def rqr_z(y: np.ndarray, r: np.ndarray, p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Dunn-Smyth randomized quantile residuals under per-row NB(r, p).

    ``p`` follows this module's parameterization (p = mean / (mean + r)); scipy's
    nbinom success probability is its complement. A well-calibrated fit gives
    residuals with variance ~1; an over-dispersed fit compresses them below 1.
    """
    p_scipy = 1.0 - p
    hi = stats.nbinom.cdf(y, r, p_scipy)
    lo = np.where(y > 0, stats.nbinom.cdf(y - 1, r, p_scipy), 0.0)
    u = rng.uniform(lo, hi)
    return stats.norm.ppf(np.clip(u, _U_EPS, 1.0 - _U_EPS))


def _rqr_var_with_ci(z: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    """Point estimate and bootstrap 90% percentile CI of the RQR variance."""
    idx = rng.integers(0, z.size, size=(_N_BOOTSTRAP, z.size))
    draws = z[idx].var(axis=1, ddof=1)
    ci_lo, ci_hi = np.percentile(draws, _RQR_CI_PCT)
    return float(z.var(ddof=1)), float(ci_lo), float(ci_hi)


def _top_decile_gaps(
    train: pd.DataFrame, val: pd.DataFrame, nb_means: np.ndarray
) -> tuple[float, float]:
    """|mean prediction - mean realized| within the realized top decile, GBM vs NB.

    A small deterministic Poisson-objective GBM on the matrix's numeric feature
    columns. If it tracks the realized top decile more closely than the per-player
    NB means, conditional signal exists that the moment fit compresses — the
    second half of the escalation bar.
    """
    feature_cols = [
        c for c in train.select_dtypes(include=[np.number]).columns if c not in _NON_FEATURE_COLUMNS
    ]
    model = LGBMRegressor(
        objective="poisson",
        n_estimators=_GBM_TREES,
        random_state=_SEED,
        deterministic=True,
        force_row_wise=True,
        verbosity=-1,
    )
    model.fit(train[feature_cols], train[_TARGET_COLUMN])
    preds = model.predict(val[feature_cols])
    y_val = val[_TARGET_COLUMN].to_numpy(dtype=float)
    top = y_val >= np.quantile(y_val, _TOP_DECILE_Q)
    realized = float(y_val[top].mean())
    return abs(float(preds[top].mean()) - realized), abs(float(nb_means[top].mean()) - realized)


def screen_cell(league: str, market: str, frame: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Run both diagnostics on one cell's cached matrix and return its verdict row."""
    frame = frame[frame[_TARGET_COLUMN].notna()].sort_values("Date")
    n_train = int(len(frame) * _TRAIN_FRACTION)
    train, val = frame.iloc[:n_train], frame.iloc[n_train:]
    r, p = _player_nb_params(train, val["Player"])
    y_val = val[_TARGET_COLUMN].to_numpy(dtype=float)

    zero_obs = int((y_val == 0).sum())
    null_mean, pvalue = poisson_binomial_pvalue((1.0 - p) ** r, zero_obs)

    rqr_var, ci_lo, ci_hi = _rqr_var_with_ci(rqr_z(y_val, r, p, rng), rng)
    nb_means = r * p / (1.0 - p)
    gbm_gap, nb_gap = _top_decile_gaps(train, val, nb_means)
    return {
        "league": league,
        "market": market,
        "n_val": len(val),
        "zero_obs": zero_obs,
        "zero_null_mean": null_mean,
        "zero_pvalue": pvalue,
        "route": route_zero_mod(zero_obs, null_mean, pvalue),
        "rqr_var": rqr_var,
        "rqr_ci_lo": ci_lo,
        "rqr_ci_hi": ci_hi,
        "gbm_top_decile_gap": gbm_gap,
        "nb_top_decile_gap": nb_gap,
        "bar_pass": bool(rqr_var < _RQR_VAR_BAR and gbm_gap < nb_gap),
    }


def _upsert_row(row: dict, out_path: Path) -> None:
    """Replace the cell's prior row (sweep-board style) so partial runs stay current."""
    frame = pd.DataFrame([row], columns=list(REQUIRED_COLUMNS))
    if out_path.exists():
        prior = pd.read_parquet(out_path)
        keep = prior[~((prior["league"] == row["league"]) & (prior["market"] == row["market"]))]
        frame = pd.concat([keep, frame], ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)


def _split_commas(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(part.strip() for value in values for part in value.split(",") if part.strip())


@click.command()
@click.option(
    "--league",
    "leagues",
    multiple=True,
    help="League filter; repeatable or comma-separated.",
)
@click.option(
    "--market",
    "markets",
    multiple=True,
    help="Market filter; repeatable or comma-separated.",
)
@click.option(
    "--all-count",
    is_flag=True,
    help="Screen every count-dist cell in stat_meta.json with a cached matrix.",
)
def main(leagues: tuple[str, ...], markets: tuple[str, ...], all_count: bool) -> None:
    """Screen count cells: zero-modification routing + Double Poisson escalation bar."""
    cells = count_cells(_split_commas(leagues), _split_commas(markets), all_count=all_count)
    if not cells:
        click.echo("No count cells matched the selection.")
        return
    rng = np.random.default_rng(_SEED)
    out_path = _data_child("research", "count_family_screen.parquet")
    rows = []
    for league, market in tqdm(cells, desc="Screening count cells"):
        row = screen_cell(league, market, pd.read_parquet(_matrix_path(league, market)), rng)
        rows.append(row)
        _upsert_row(row, out_path)
    board = pd.DataFrame(rows, columns=list(REQUIRED_COLUMNS))
    click.echo(board.to_string(index=False, float_format="{:.4f}".format))
    click.echo(f"Upserted {len(rows)} cells into {out_path}")


if __name__ == "__main__":
    main()
