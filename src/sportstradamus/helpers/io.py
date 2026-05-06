"""Atomic parquet/JSON IO + schema converters for the data hot path.

All writers stage to a ``<path>.tmp`` and then ``os.replace()`` to the target
path, so the dashboard never reads a torn file when a pipeline writes
mid-render.

The ``Offers`` column in history is in-memory ``list[tuple]`` of mixed types.
PyArrow needs a ``list<struct>`` for that, so the converters round-trip
tuple <-> dict at the parquet boundary. Every other consumer of the column
keeps its tuple-indexed semantics.

Reads prefer parquet but fall back to the legacy ``.dat`` pickle when
parquet is absent — this lets installs that haven't run the one-shot
migration script keep working. Writes target parquet only; ``.dat`` files
are read-only legacy.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sportstradamus import data

# --- Canonical artifact paths ---
HISTORY_PATH = pkg_resources.files(data) / "history.parquet"
PARLAY_HIST_PATH = pkg_resources.files(data) / "parlay_hist.parquet"
CURRENT_OFFERS_PATH = pkg_resources.files(data) / "current_offers.parquet"
CURRENT_PARLAYS_PATH = pkg_resources.files(data) / "current_parlays.parquet"
CURRENT_META_PATH = pkg_resources.files(data) / "current_meta.json"
MODEL_STATS_PATH = pkg_resources.files(data) / "model_stats.parquet"

# Legacy pickle paths kept as read-only fallback until the parquet migration
# has run on every install.
HISTORY_PICKLE_PATH = pkg_resources.files(data) / "history.dat"
PARLAY_HIST_PICKLE_PATH = pkg_resources.files(data) / "parlay_hist.dat"

# Field order for the Offers struct must match the in-memory tuple positions.
# CLV adds the trailing three (close_books_p, market_clv, model_clv); legacy
# six-tuples are zero-padded with NaN on read.
_OFFER_FIELDS = (
    "line",
    "boost",
    "platform",
    "bet",
    "model_p",
    "books_p",
    "close_books_p",
    "market_clv",
    "model_clv",
)
_LEGACY_OFFER_LEN = 6

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


def _read_pickle_fallback(path) -> pd.DataFrame:
    """Return pickle contents or an empty DataFrame if the file is absent."""
    p = Path(str(path))
    if not p.is_file():
        return pd.DataFrame()
    return pd.read_pickle(p)


# ---------------------------------------------------------------------------
# History.Offers <-> parquet struct
# ---------------------------------------------------------------------------


def _pad_legacy_offer(offer):
    """Pad a 6-tuple legacy offer up to the current 9-field shape with NaN."""
    if len(offer) == _LEGACY_OFFER_LEN:
        return (*tuple(offer), np.nan, np.nan, np.nan)
    return tuple(offer)


def _offer_tuple_to_dict(offer):
    if not isinstance(offer, tuple | list):
        return None
    if len(offer) == _LEGACY_OFFER_LEN:
        offer = _pad_legacy_offer(offer)
    if len(offer) != len(_OFFER_FIELDS):
        return None
    return dict(zip(_OFFER_FIELDS, offer, strict=False))


def _offer_dict_to_tuple(offer):
    if not isinstance(offer, dict):
        if isinstance(offer, tuple | list) and len(offer) == _LEGACY_OFFER_LEN:
            return _pad_legacy_offer(offer)
        return offer
    return tuple(offer.get(f, np.nan) for f in _OFFER_FIELDS)


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
    """Read the prediction history. Falls back to ``history.dat`` on miss."""
    df = read_parquet_safe(HISTORY_PATH)
    if df.empty:
        df = _read_pickle_fallback(HISTORY_PICKLE_PATH)
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
    out = out.drop(columns=[c for c in ("_date",) if c in out.columns])
    for col in _PARLAY_LIST_COLS:
        if col in out.columns:
            out[col] = out[col].apply(_seq_to_list)
    _atomic_write_parquet(out, PARLAY_HIST_PATH)


def read_parlay_hist() -> pd.DataFrame:
    """Read the parlay history. Falls back to ``parlay_hist.dat`` on miss."""
    df = read_parquet_safe(PARLAY_HIST_PATH)
    if df.empty:
        df = _read_pickle_fallback(PARLAY_HIST_PICKLE_PATH)
    if df.empty:
        return df
    for col in _PARLAY_LIST_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_list_to_tuple)
    return df


# ---------------------------------------------------------------------------
# Upcoming-events ledger (close-line scheduler input)
# ---------------------------------------------------------------------------

UPCOMING_EVENTS_PATH = pkg_resources.files(data) / "upcoming_events.json"


def read_upcoming_events() -> list[dict]:
    """Return the list of upcoming events, or empty list if the file is absent."""
    p = Path(str(UPCOMING_EVENTS_PATH))
    if not p.is_file():
        return []
    with p.open() as f:
        return json.load(f)


def write_upcoming_events(events: list[dict]) -> None:
    """Atomically write the upcoming-events ledger as JSON."""
    cleaned = [e for e in events if isinstance(e, dict) and not _has_nan(e.values())]
    _atomic_write_json(cleaned, UPCOMING_EVENTS_PATH)


def _has_nan(values) -> bool:
    return any(isinstance(v, float) and math.isnan(v) for v in values)
