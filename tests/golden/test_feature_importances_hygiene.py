"""Golden invariant: the drift-monitoring CSVs carry no double-prefix residue.

A fixed bug once emitted ``Player Player X_asof`` feature rows alongside the
canonical ``Player X_asof`` rows (``training.shap``); the stale duplicates are
all-zero residue. ``see_features()`` purges them on a full rebuild, and this
test pins both CSVs so a stale row can never creep back in (a regressed prefix
bug, or ``compute_market_importance``'s outer-join accumulating them again).

The CSVs are runtime-recomputed (gitignored), so this skips in a fresh checkout
and bites where they exist (dev machine, server).
"""

from __future__ import annotations

import importlib.resources as pkg_resources

import pandas as pd
import pytest

from sportstradamus import data

_STALE_PREFIX = "Player Player "


@pytest.mark.parametrize("csv_name", ["feature_importances.csv", "feature_correlations.csv"])
def test_no_double_prefix_rows(csv_name):
    path = pkg_resources.files(data) / "training" / csv_name
    if not path.is_file():
        pytest.skip(f"{csv_name} absent (runtime-recomputed); invariant unverifiable here")
    df = pd.read_csv(path, index_col=0)
    stale = [str(i) for i in df.index if str(i).startswith(_STALE_PREFIX)]
    assert not stale, f"{csv_name} carries {len(stale)} double-prefix residue rows: {stale[:5]}"
