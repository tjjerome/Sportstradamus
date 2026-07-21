"""Endpoint-preserving CDF-map kernels for the group-conditional engine.

Both NFL corners fit a monotone map from raw CDF quantiles to recalibrated ones,
but by two non-interchangeable implementations. The receiving corner delegates to
:func:`sportstradamus.training.posthoc.fit_isotonic_pit` (blob kind
``isotonic_pit``); the rushing corner uses a hand-inlined empirical-CDF fit (blob
kind ``group_empirical_cdf``). Their terminal-knot handling differs, so they are
not bit-identical and ``map_impl`` selects the one a corner needs. The two
Kolmogorov-Smirnov helpers likewise differ: the receiving path filters nonfinite
values and rejects an empty draw; the rushing path does neither.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from sportstradamus.training.group_conditional_cdf._contracts import (
    PIT_BINS,
    EndpointMapBlob,
    GroupCdfBlob,
)
from sportstradamus.training.posthoc import fit_isotonic_pit


def fit_endpoint_map(samples: np.ndarray, lam: float) -> EndpointMapBlob:
    """Fit the receiving ``isotonic_pit`` endpoint map at a shrink ``lam``."""
    fitted = fit_isotonic_pit(samples, n_bins=PIT_BINS, lam=lam)
    if fitted is None:
        raise ValueError("two-part map has insufficient conditional support")
    fitted["lam"] = float(lam)
    return cast(EndpointMapBlob, fitted)


def apply_endpoint_map(blob: EndpointMapBlob, values: np.ndarray) -> np.ndarray:
    return np.clip(np.interp(values, blob["x"], blob["y"]), 0.0, 1.0)


def fit_pit_map(draws: np.ndarray, lam: float) -> GroupCdfBlob:
    """Fit the rushing ``group_empirical_cdf`` map from a draw-by-row matrix."""
    pit = np.sort(np.asarray(draws, dtype=float).reshape(-1), kind="stable")
    coverage = (np.arange(len(pit)) + 0.5) / len(pit)
    x_knots, y_knots = [0.0], [0.0]
    for group_pit, group_coverage in zip(
        np.array_split(pit, PIT_BINS), np.array_split(coverage, PIT_BINS), strict=True
    ):
        x_value = float(group_pit.mean())
        if x_value <= x_knots[-1]:
            continue
        x_knots.append(x_value)
        y_knots.append(lam * float(group_coverage.mean()) + (1.0 - lam) * x_value)
    if x_knots[-1] < 1.0:
        x_knots.append(1.0)
        y_knots.append(1.0)
    else:
        y_knots[-1] = 1.0
    return {"kind": "group_empirical_cdf", "lam": lam, "x": x_knots, "y": y_knots}


def ks_uniform_finite(values: np.ndarray) -> float:
    """Receiving KS: drop nonfinite values, reject an empty draw."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("PIT guard requires at least one finite value")
    ordered = np.sort(np.clip(finite, 0.0, 1.0))
    ranks = np.arange(1, len(ordered) + 1)
    return float(
        max(
            np.max(ranks / len(ordered) - ordered),
            np.max(ordered - (ranks - 1) / len(ordered)),
        )
    )


def ks_uniform(values: np.ndarray) -> float:
    """Rushing KS: no finite filter, no empty check."""
    ordered = np.sort(np.clip(values, 0.0, 1.0))
    ranks = np.arange(1, len(ordered) + 1)
    return float(
        max(
            np.max(ranks / len(ordered) - ordered),
            np.max(ordered - (ranks - 1) / len(ordered)),
        )
    )


def mean_ks(draws: np.ndarray) -> float:
    if draws.ndim != 2 or draws.shape[1] == 0:
        raise ValueError("PIT guard requires a nonempty draw-by-row matrix")
    return float(np.mean([ks_uniform_finite(draw) for draw in draws]))
