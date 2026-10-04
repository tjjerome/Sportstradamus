"""Information test for the tail scorecard: does the model know anything at the money?

On authentic held-out test rows the ladder priced, the model-only forecast (the pre-blend
mean ``EV`` and ``P_standalone`` at the row's line, never the fused ``P`` or
``Blended_EV``, which already hold the book leg) is fitted beside the market in a
forecast-encompassing regression (Chong & Hendry 1986), each estimator with an intercept:

- linear: ``Actual ~ a + b_model * EV + b_market * line``;
- logistic: ``y ~ a + b_market * logit(p_market) + b_model * logit(P_standalone)``;
- the ablation: the log-loss the logistic fit gains from its model term.

The market is read at two references on the same rows. Decision time: each sportsbook's
latest ``under_prob`` at the row's line by the DFS platforms' last poll of the row,
averaged over books, and the consensus line then. Training time: ``1 - Odds`` and
``Book_EV``, the 12:00 UTC quote. Their difference is the reference-timing skew. League
and overall rows pool their cells with fixed effects (a cell intercept and market slope,
one common b_model), so cross-cell level differences are not credited to the model. Every
CI resamples whole days. No through-origin slope of (y - m) on (p - m): a mis-levelled
market reference, like the NFL count quotes, contaminates it.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from scipy.special import expit, logit

from sportstradamus.helpers.training_quotes import AUTHENTIC, DFS_PLATFORM_BOOKS
from sportstradamus.training.scorecard import (
    _CI_HIGH_PCT,
    _CI_LOW_PCT,
    _bootstrap_mean_ci_clustered,
)

# Spec section 6: at least 2,000 multinomial day resamples behind every CI.
DAY_RESAMPLES = 2000

# R3's floors for a cell in a fit (c23, c26): below them b_model's CI spans any reading.
INFO_MIN_ROWS = 100
INFO_MIN_DAYS = 5

# A market price this flat cannot carry its own slope beside the cell intercept (R3, c26).
_MARKET_SD_MIN = 1e-4

# Probabilities are clipped before the logit so a 0 or 1 stays finite (R3).
_PROB_CLIP = 1e-3

# Newton's method on the logit converges quadratically; a step this small has converged.
_NEWTON_TOL = 1e-10
_NEWTON_MAX_STEPS = 50

# The market line each reference regresses the realized stat on; its over-price is
# ``p_<reference>``.
_REFERENCE_LINES = {"decision": "consensus", "training": "Book_EV"}

_DECISION_QUOTE_SQL = """
WITH quotes AS (
    SELECT k.rid, o.book, arg_max(o.under_prob, o.observed_at) AS under_prob
    FROM info_rows k JOIN odds o ON o.league = $league AND o.market = $market
        AND o.game_date = k.game_date AND o.entity = k.player AND o.line = k.line
        AND o.observed_at <= k.t_dec AND o.under_prob IS NOT NULL
    WHERE NOT list_contains($dfs, o.book)
    GROUP BY ALL
)
SELECT rid, 1 - avg(under_prob) AS p_decision FROM quotes GROUP BY rid
"""


def information_rows(
    con: duckdb.DuckDBPyConnection,
    league: str,
    market: str,
    rows: pd.DataFrame,
    rungs: pd.DataFrame,
) -> pd.DataFrame:
    """A cell's authentic, settled test rows the ladder priced, with both market references.

    Args:
        con: The read-only archive connection.
        league: The cell's league.
        market: The cell's market, as the archive keys it.
        rows: The cell's test rows in the scored window, indexed by ``rid``.
        rungs: Their ladder rungs: ``rid``, ``last_poll`` and the decision-time
            ``consensus`` line.

    Returns:
        The authentic, non-push ``rows`` with a sportsbook quote at their own line by
        decision time, plus ``consensus``; the over-prices ``p_decision`` (the mean
        sportsbook quote then) and ``p_training`` (``1 - Odds``); ``y``, 1.0 when the
        result went over; and the clipped log-odds ``logit_model`` (``P_standalone``),
        ``logit_decision`` and ``logit_training``.
    """
    # Decision time is the platforms' last poll of the row, and the consensus line depends
    # only on the poll time, so the latest-polled rung carries both.
    latest = rungs.sort_values("last_poll").drop_duplicates("rid", keep="last").set_index("rid")
    info = rows.join(latest[["last_poll", "consensus"]], how="inner")
    info = info[info["QuoteAuthenticity"].eq(AUTHENTIC) & info["Result"].ne(info["Line"])]
    keys = pd.DataFrame(
        {
            "rid": info.index,
            "game_date": pd.to_datetime(info["Date"]).dt.date,
            "player": info["Player"],
            "line": info["Line"],
            "t_dec": info["last_poll"],
        }
    )
    con.register("info_rows", keys)
    params = {"league": league, "market": market, "dfs": sorted(DFS_PLATFORM_BOOKS)}
    info = info.join(con.execute(_DECISION_QUOTE_SQL, params).df().set_index("rid"), how="inner")
    info = info.assign(y=info["Result"].gt(info["Line"]).astype(float), p_training=1 - info["Odds"])
    prices = {"model": "P_standalone", "decision": "p_decision", "training": "p_training"}
    return info.assign(
        **{
            f"logit_{name}": logit(info[column].clip(_PROB_CLIP, 1 - _PROB_CLIP))
            for name, column in prices.items()
        }
    )


def information_table(info: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """b_model with day-clustered CIs and the model term's log-loss gain, per scope.

    One row per (scope, key, reference) for each cell clearing ``INFO_MIN_ROWS`` and
    ``INFO_MIN_DAYS`` with a moving market price at both references, and for each league
    and overall, pooled over those cells with cell fixed effects. A cell below the floors
    carries only its counts.

    Args:
        info: ``information_rows`` frames with ``cell`` and ``League``.
        rng: Drives every day resample.

    Returns:
        ``scope``, ``key``, ``split`` (``"information"``), ``band`` (the reference),
        ``info_rows``, ``info_days``, ``info_cells``; each estimator's ``b_model`` with its
        CI bounds (``lin_*``, ``logit_*``) and, on a cell, its ``b_market``; ``ll_gain``,
        the mean per-row log-likelihood gain of the model term in nats, in sample, with a
        CI that resamples days at the fitted coefficients.
    """
    per_cell = info.groupby("cell").agg(
        rows=("y", "size"),
        days=("Date", "nunique"),
        **{f"sd_{reference}": (f"p_{reference}", "std") for reference in _REFERENCE_LINES},
    )
    fits = per_cell.index[
        per_cell["rows"].ge(INFO_MIN_ROWS)
        & per_cell["days"].ge(INFO_MIN_DAYS)
        & per_cell.filter(like="sd_").min(axis=1).gt(_MARKET_SD_MIN)
    ]
    pooled = info[info["cell"].isin(fits)]
    scopes = [
        ("overall", "all", pooled),
        *(("league", league, rows) for league, rows in pooled.groupby("League")),
        *(("cell", cell, rows) for cell, rows in info.groupby("cell")),
    ]
    records = []
    for scope, key, rows in scopes:
        counts = {
            "scope": scope,
            "key": key,
            "split": "information",
            "info_rows": len(rows),
            "info_days": rows["Date"].nunique(),
            "info_cells": rows["cell"].nunique(),
        }
        if rows.empty or (scope == "cell" and key not in fits):
            records.append(counts)
            continue
        records += [
            counts | {"band": reference, **_estimates(rows, reference, rng)}
            for reference in _REFERENCE_LINES
        ]
    return pd.DataFrame(records)


def _estimates(rows: pd.DataFrame, reference: str, rng: np.random.Generator) -> dict[str, float]:
    """Both encompassing fits and the ablation at one market reference."""
    days = rows["Date"].to_numpy()
    actual = rows["Result"].to_numpy(float)
    design = _design(rows, _REFERENCE_LINES[reference], "EV")
    linear = np.linalg.lstsq(design, actual, rcond=None)[0]
    linear_draws = _day_resamples(
        design, actual - design @ linear, np.ones(len(actual)), linear, days, rng
    )
    y = rows["y"].to_numpy(float)
    design = _design(rows, f"logit_{reference}", "logit_model")
    logistic = _logit_fit(design, y)
    p_full = expit(design @ logistic)
    logistic_draws = _day_resamples(design, y - p_full, p_full * (1 - p_full), logistic, days, rng)
    p_market = expit(design[:, :-1] @ _logit_fit(design[:, :-1], y))
    gain = np.log(np.where(y == 1, p_full / p_market, (1 - p_full) / (1 - p_market)))
    ll_gain, ll_lo, ll_hi = _bootstrap_mean_ci_clustered(gain, days, rng, DAY_RESAMPLES)
    one_cell = rows["cell"].nunique() == 1  # a pooled fit has one market slope per cell
    estimates = {"ll_gain": ll_gain, "ll_gain_lo": ll_lo, "ll_gain_hi": ll_hi}
    for name, beta, draws in (
        ("lin", linear, linear_draws),
        ("logit", logistic, logistic_draws),
    ):
        lo, hi = np.percentile(draws[:, -1], [_CI_LOW_PCT, _CI_HIGH_PCT])
        estimates |= {
            f"{name}_b_model": beta[-1],
            f"{name}_b_model_lo": lo,
            f"{name}_b_model_hi": hi,
            f"{name}_b_market": beta[1] if one_cell else np.nan,
        }
    return estimates


def _design(rows: pd.DataFrame, market: str, model: str) -> np.ndarray:
    """Cell fixed effects (an intercept and a market slope per cell), the model term last."""
    cells = pd.get_dummies(rows["cell"], dtype=float).to_numpy()
    return np.hstack([cells, cells * rows[[market]].to_numpy(float), rows[[model]].to_numpy(float)])


def _logit_fit(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta = np.zeros(design.shape[1])
    for _ in range(_NEWTON_MAX_STEPS):
        p = expit(design @ beta)
        step = np.linalg.solve((design * (p * (1 - p))[:, None]).T @ design, design.T @ (y - p))
        beta += step
        if np.abs(step).max() < _NEWTON_TOL:
            return beta
    raise ArithmeticError(f"logit fit did not converge in {_NEWTON_MAX_STEPS} Newton steps")


def _day_resamples(
    design: np.ndarray,
    residual: np.ndarray,
    curvature: np.ndarray,
    beta: np.ndarray,
    days: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """``DAY_RESAMPLES`` multinomial day resamples of a fitted coefficient vector.

    Each resample takes one Newton step from the full-sample fit on per-day sums of the
    score and curvature (Andrews 2002's k-step bootstrap with k = 1): exact for least
    squares, first-order for the logit, and it never refits the pooled fixed-effect design
    row by row. A resample that drops every day of a cell leaves that cell's terms where
    they were (the pseudo-inverse) while b_model still moves.

    Returns:
        ``(DAY_RESAMPLES, n_coefficients)`` resampled coefficients.
    """
    _, day = np.unique(days, return_inverse=True)
    members = [day == d for d in range(day.max() + 1)]
    score = np.stack([design[m].T @ residual[m] for m in members])
    curvature_sums = np.stack([(design[m] * curvature[m, None]).T @ design[m] for m in members])
    weights = rng.multinomial(len(members), np.full(len(members), 1 / len(members)), DAY_RESAMPLES)
    inverse = np.linalg.pinv(np.tensordot(weights, curvature_sums, axes=1), hermitian=True)
    return beta + np.einsum("bij,bj->bi", inverse, weights @ score)
