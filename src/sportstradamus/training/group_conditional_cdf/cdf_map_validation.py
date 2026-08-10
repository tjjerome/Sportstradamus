"""Shared portable-CDF validation for structural structural calibrators."""

from __future__ import annotations

from collections.abc import Collection, Mapping

import numpy as np


def cdf_map_blob_is_valid(
    blob: Mapping[str, object], *, kind: str, allowed_lambdas: Collection[float]
) -> bool:
    """Return whether a CDF-map blob has canonical monotone unit-interval knots."""
    x_values = np.asarray(blob.get("x"), dtype=float)
    y_values = np.asarray(blob.get("y"), dtype=float)
    lam = blob.get("lam")
    return bool(
        blob.get("kind") == kind
        and isinstance(lam, (int, float))
        and float(lam) in allowed_lambdas
        and x_values.ndim == y_values.ndim == 1
        and len(x_values) == len(y_values)
        and len(x_values) >= 2
        and np.isfinite(x_values).all()
        and np.isfinite(y_values).all()
        and x_values[0] == y_values[0] == 0.0
        and x_values[-1] == y_values[-1] == 1.0
        and np.all(np.diff(x_values) > 0.0)
        and np.all(np.diff(y_values) >= 0.0)
        and np.all((x_values >= 0.0) & (x_values <= 1.0))
        and np.all((y_values >= 0.0) & (y_values <= 1.0))
    )
