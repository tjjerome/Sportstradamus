"""New-schema to legacy-vocabulary translation for Fantasy Points rows.

The 104 aggregate output columns the NFL models train on are frozen inside
the model pickles and sliced strictly at serve time, so a column that stops
appearing is a ``KeyError`` in production. The rebuilt API publishes none of
those names. Rather than rename the feature set and retrain, this module
copies each column the recipes read back to its legacy camelCase name on the
way to parquet.

``config/fantasypoints_column_map.json`` holds the translation, keyed by
logical file kind. It was derived by matching *values* — not names — against
the last legacy snapshot, so a wrong guess would have shown up as a value
mismatch rather than a plausible-looking column.

Three sections, all additive, so one parquet serves both the frozen feature
set and anything built on the wider schema later:

* ``rename`` copies a column under its legacy name, applying any ``scale``.
  Those carry the conventions the API dropped: rates stored as fractions
  where it reports percents, sack yardage stored as a loss.
* ``derive`` builds a legacy rate from a ``__raw`` numerator/denominator
  pair, which is exact where the API's displayed rate is pre-rounded.
* ``bucket`` re-nests the flat ``{bucket}_{stat}`` columns into the single
  JSON cell the legacy schema carried, so the bucket-parsing aggregators read
  archived legacy snapshots and new pulls alike.
"""

from __future__ import annotations

import functools
import importlib.resources as pkg_resources
import json
from pathlib import Path

import pandas as pd

from sportstradamus import data

COLUMN_MAP_PATH = Path(str(pkg_resources.files(data) / "config" / "fantasypoints_column_map.json"))


@functools.cache
def load_column_map() -> dict[str, dict]:
    """Return the parsed column map, read once per process."""
    with COLUMN_MAP_PATH.open() as f:
        return json.load(f)


def map_key(context: str, tool: str) -> str | None:
    """Return the column-map key for one routed catalog entry, or ``None``.

    ``player`` reads the bare tool entry; ``opponent`` the ``_opp`` defensive
    mirror; ``team`` prefers a ``_team`` entry and falls back to the bare one
    for the kinds whose offensive view is the unsuffixed one.
    """
    mapping = load_column_map()
    candidates = {
        "player": (tool,),
        "team": (f"{tool}_team", tool),
        "opponent": (f"{tool}_opp",),
    }.get(context, (tool,))
    return next((key for key in candidates if key in mapping), None)


def apply_column_map(df: pd.DataFrame, context: str, tool: str) -> pd.DataFrame:
    """Add the legacy-named columns this file kind's map entry describes.

    A kind with no map entry is returned untouched — that is how a tool
    collected only for the wider schema stays out of the frozen feature set.
    """
    key = map_key(context, tool)
    if key is None:
        return df
    entry = load_column_map()[key]
    scales = entry.get("scale", {})
    for new_col, legacy_col in entry.get("rename", {}).items():
        if new_col not in df.columns:
            continue
        values = _coerce_numeric(df, new_col)
        scale = scales.get(new_col)
        df[legacy_col] = values * scale if scale is not None else df[new_col]
    for legacy_col, expr in entry.get("derive", {}).items():
        series = _evaluate(df, expr)
        if series is not None:
            df[legacy_col] = series
    bucket = entry.get("bucket")
    if bucket:
        df["bucket"] = _build_bucket_cells(df, bucket)
    return df


def _coerce_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def _evaluate(df: pd.DataFrame, expr: dict) -> pd.Series | None:
    """Resolve one map expression — ``{"col": x}`` or ``{"num": x, "den": y}``.

    ``num`` may be a list, which sums the named columns first: a few legacy
    totals only exist in the new schema as the sum of their parts.

    Returns ``None`` when a referenced column is absent, which is how a tool
    that stops publishing a split degrades to a missing legacy column rather
    than a wrong one.
    """
    if "col" in expr:
        col = expr["col"]
        return _coerce_numeric(df, col) if col in df.columns else None
    num_cols = expr["num"] if isinstance(expr["num"], list) else [expr["num"]]
    den_col = expr["den"]
    if den_col not in df.columns or any(c not in df.columns for c in num_cols):
        return None
    num = sum(_coerce_numeric(df, c) for c in num_cols)
    den = _coerce_numeric(df, den_col)
    return num.divide(den).where(den > 0)


def _build_bucket_cells(df: pd.DataFrame, bucket_map: dict) -> list[dict]:
    """Re-nest the flat per-bucket columns into one legacy ``bucket`` dict per row.

    A bucket is emitted for a row only when at least one of its rates
    resolved — i.e. the player actually saw that split. The legacy files
    carried buckets the same way, and the aggregators read the rate keys, so
    a routes-only entry would be dead weight in every parquet.
    """
    resolved = {
        name: {stat: (_evaluate(df, expr), "num" in expr) for stat, expr in inner.items()}
        for name, inner in bucket_map.items()
    }
    cells: list[dict] = []
    for pos in range(len(df)):
        cell: dict[str, dict] = {}
        for name, stats in resolved.items():
            values = {}
            has_rate = False
            for stat, (series, is_rate) in stats.items():
                if series is None or pd.isna(series.iat[pos]):
                    continue
                values[stat] = series.iat[pos].item()
                has_rate = has_rate or is_rate
            if has_rate:
                cell[name] = values
        cells.append(cell)
    return cells
