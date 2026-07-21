"""Affine mean-correction and raw grouped-CDF kernels (rushing corner).

These functions are moved verbatim from the rushing affine calibrator. The only
change is that the position codes are threaded in as ``codes`` rather than read
from a frozen module literal, so the engine iterates the set discovered from data
(and persisted as the ``position_cdf`` keys) instead of a hard-coded tuple.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import skewnorm
from sklearn.model_selection import GroupKFold

from sportstradamus.helpers.distributions import skewnormal_loc_from_mean
from sportstradamus.training.group_conditional_cdf._contracts import (
    MARGINAL_MEAN_FLOOR,
    PIT_LAMBDA_TIE_TOLERANCE,
    PIT_LAMBDAS,
    PLAYER_CV_FOLDS,
    AffinePredictive,
    GroupCdfBlob,
)
from sportstradamus.training.group_conditional_cdf._maps import fit_pit_map, ks_uniform


def fit_affine(mean: np.ndarray, result: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(mean, result, 1)
    return float(intercept), float(slope)


def apply_affine(
    pred: AffinePredictive,
    intercept: float,
    slope: float,
    floor: float = MARGINAL_MEAN_FLOOR,
) -> AffinePredictive:
    mean = np.clip(intercept + slope * pred.marginal_mean, floor, None)
    return AffinePredictive(mean, pred.sigma, pred.alpha, pred.gate)


def subset_predictive(pred: AffinePredictive, rows: np.ndarray) -> AffinePredictive:
    return AffinePredictive(*(getattr(pred, field)[rows] for field in pred.__dataclass_fields__))


def raw_cdf_endpoints(pred: AffinePredictive, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    conditional = pred.marginal_mean / (1.0 - pred.gate)
    loc = skewnormal_loc_from_mean(conditional, pred.sigma, pred.alpha)
    base = skewnorm.cdf(point, pred.alpha, loc=loc, scale=pred.sigma)
    upper = np.where(
        point < 0.0,
        (1.0 - pred.gate) * base,
        pred.gate + (1.0 - pred.gate) * base,
    )
    atom = np.where(point == 0.0, pred.gate, 0.0)
    return upper - atom, upper


def mapped_cdf_endpoints(pred, point, positions, maps, codes):
    lower, upper = raw_cdf_endpoints(pred, point)
    mapped_lower, mapped_upper = np.empty_like(lower), np.empty_like(upper)
    for code in codes:
        mask = positions == code
        blob = maps[str(code)]
        mapped_lower[mask] = np.interp(lower[mask], blob["x"], blob["y"])
        mapped_upper[mask] = np.interp(upper[mask], blob["x"], blob["y"])
    return np.clip(mapped_lower, 0.0, 1.0), np.clip(mapped_upper, 0.0, 1.0)


def mapped_over(pred, line, positions, maps, codes):
    _, mapped_under = mapped_cdf_endpoints(pred, line, positions, maps, codes)
    return 1.0 - mapped_under


def fit_group_maps(pred, result, positions, players, uniforms, codes):
    lower, upper = raw_cdf_endpoints(pred, result)
    draws = lower[None, :] + uniforms * (upper - lower)[None, :]
    maps: dict[str, GroupCdfBlob] = {}
    for code in codes:
        mask = positions == code
        lam = select_lambda(draws[:, mask], players[mask])
        maps[str(code)] = fit_pit_map(draws[:, mask], lam)
    return maps


def select_lambda(draws: np.ndarray, players: np.ndarray) -> float:
    splitter = GroupKFold(n_splits=PLAYER_CV_FOLDS)
    scores: dict[float, float] = {}
    for lam in PIT_LAMBDAS:
        oof = np.empty_like(draws)
        for train, hold in splitter.split(np.zeros(draws.shape[1]), groups=players):
            blob = fit_pit_map(draws[:, train], lam)
            oof[:, hold] = np.interp(draws[:, hold], blob["x"], blob["y"])
        scores[lam] = float(np.mean([ks_uniform(sample) for sample in oof]))
    minimum = min(scores.values())
    return next(lam for lam in PIT_LAMBDAS if scores[lam] <= minimum + PIT_LAMBDA_TIE_TOLERANCE)
