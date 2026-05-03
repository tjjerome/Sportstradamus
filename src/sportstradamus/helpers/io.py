"""Atomic parquet/JSON IO + schema converters for the data hot path.

All writers stage to a `<path>.tmp` and then `os.replace()` to the target path,
so the dashboard never reads a torn file when a pipeline writes mid-render.

The `Offers` column in history is in-memory `list[tuple]` of mixed types.
PyArrow needs a `list<struct>` for that, so the converters round-trip
tuple <-> dict at the parquet boundary. Every other consumer of the column
keeps its tuple-indexed semantics.
"""

import importlib.resources as pkg_resources
import json
from pathlib import Path

import pandas as pd

from sportstradamus import data

# --- Canonical artifact paths ---
# Dashboard reads these; pipelines write these. Single source of truth.
HISTORY_PATH = pkg_resources.files(data) / "history.parquet"
PARLAY_HIST_PATH = pkg_resources.files(data) / "parlay_hist.parquet"
CURRENT_OFFERS_PATH = pkg_resources.files(data) / "current_offers.parquet"
CURRENT_PARLAYS_PATH = pkg_resources.files(data) / "current_parlays.parquet"
CURRENT_META_PATH = pkg_resources.files(data) / "current_meta.json"
MODEL_STATS_PATH = pkg_resources.files(data) / "model_stats.parquet"

# Legacy pickle paths kept only for the one-shot migration script.
HISTORY_PICKLE_PATH = pkg_resources.files(data) / "history.dat"
PARLAY_HIST_PICKLE_PATH = pkg_resources.files(data) / "parlay_hist.dat"

# Field order for the Offers struct must match the in-memory tuple positions.
_OFFER_FIELDS = ("line", "boost", "platform", "bet", "model_p", "books_p")

# Tuple-typed columns in parlay_hist that round-trip as homogeneous float lists.
_PARLAY_LIST_COLS = ("Leg Probs", "Corr Pairs", "Boost Pairs", "Markets", "Players")


def _atomic_write_parquet(df: pd.DataFrame, path) -> None:
    path = Path(str(path))
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, engine="pyarrow", index=False)
    tmp.replace(path)


def _atomic_write_json(obj, path) -> None:
    path = Path(str(path))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, default=str)
    tmp.replace(path)


def read_parquet_safe(path) -> pd.DataFrame:
    """Return parquet contents or an empty DataFrame if the file is absent."""
    p = Path(str(path))
    if not p.is_file():
        return pd.DataFrame()
    return pd.read_parquet(p, engine="pyarrow")


# ---------------------------------------------------------------------------
# History.Offers <-> parquet struct
# ---------------------------------------------------------------------------


def _offer_tuple_to_dict(offer):
    if not isinstance(offer, tuple | list) or len(offer) != len(_OFFER_FIELDS):
        return None
    return dict(zip(_OFFER_FIELDS, offer, strict=False))


def _offer_dict_to_tuple(offer):
    if not isinstance(offer, dict):
        return offer
    return tuple(offer.get(f) for f in _OFFER_FIELDS)


def _offers_for_parquet(offers):
    if not isinstance(offers, list):
        return []
    return [d for d in (_offer_tuple_to_dict(o) for o in offers) if d is not None]


def _offers_from_parquet(offers):
    if offers is None:
        return []
    return [_offer_dict_to_tuple(o) for o in offers]


def write_history(df: pd.DataFrame) -> None:
    """Atomically write the prediction history parquet."""
    out = df.copy()
    if "Offers" in out.columns:
        out["Offers"] = out["Offers"].apply(_offers_for_parquet)
    _atomic_write_parquet(out, HISTORY_PATH)


def read_history() -> pd.DataFrame:
    df = read_parquet_safe(HISTORY_PATH)
    if df.empty:
        return df
    if "Offers" in df.columns:
        df["Offers"] = df["Offers"].apply(_offers_from_parquet)
    return df


# ---------------------------------------------------------------------------
# Parlay history homogeneous tuple cols <-> parquet list<float>
# ---------------------------------------------------------------------------


def _seq_to_list(v):
    if isinstance(v, tuple):
        return list(v)
    return v


def _list_to_tuple(v):
    if isinstance(v, list):
        return tuple(v)
    return v


def write_parlay_hist(df: pd.DataFrame) -> None:
    """Atomically write the parlay history parquet."""
    out = df.copy()
    # _date is a transient column added by analysis.compute_parlay_metrics; never persist it.
    out = out.drop(columns=[c for c in ("_date",) if c in out.columns])
    for col in _PARLAY_LIST_COLS:
        if col in out.columns:
            out[col] = out[col].apply(_seq_to_list)
    _atomic_write_parquet(out, PARLAY_HIST_PATH)


def read_parlay_hist() -> pd.DataFrame:
    df = read_parquet_safe(PARLAY_HIST_PATH)
    if df.empty:
        return df
    for col in _PARLAY_LIST_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_list_to_tuple)
    return df
