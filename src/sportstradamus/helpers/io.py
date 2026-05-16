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
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sportstradamus import data

# --- Canonical artifact paths ---
# Dashboard reads these; pipelines write these. Single source of truth.
HISTORY_PATH = pkg_resources.files(data) / "history.parquet"
PARLAY_HIST_PATH = pkg_resources.files(data) / "parlay_hist.parquet"
CURRENT_OFFERS_PATH = pkg_resources.files(data) / "current_offers.parquet"
CURRENT_PARLAYS_PATH = pkg_resources.files(data) / "current_parlays.parquet"
CURRENT_META_PATH = pkg_resources.files(data) / "current_meta.json"
CURRENT_HISTORY_PATH = pkg_resources.files(data) / "current_history.parquet"
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


def _atomic_write_parquet(df: pd.DataFrame, path, compression: str | None = None) -> None:
    path = Path(str(path))
    tmp = path.with_suffix(path.suffix + ".tmp")
    if compression is None:
        df.to_parquet(tmp, engine="pyarrow", index=False)
    else:
        df.to_parquet(tmp, engine="pyarrow", index=False, compression=compression)
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
    # _date is a transient column added by analysis.compute_parlay_metrics; never persist it.
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


# ---------------------------------------------------------------------------
# Per-league gamelog: parquet for the two DataFrames + JSON/parquet sidecar
# for the heterogeneous ``players`` payload.
# ---------------------------------------------------------------------------


def _gamelog_paths(league: str):
    league = league.lower()
    base = pkg_resources.files(data)
    return {
        "gamelog": base / f"{league}_gamelog.parquet",
        "teamlog": base / f"{league}_teamlog.parquet",
        "players_json": base / f"{league}_players.json",
        "players_parquet": base / f"{league}_players.parquet",
        "legacy_pickle": base / f"{league}_data.dat",
    }


def _coerce_object_columns_to_str(df: pd.DataFrame) -> pd.DataFrame:
    """Cast object columns with mixed concrete types to string.

    Pandas tolerates int+str in a single object column (MLB ``batSide`` has
    24k rows where the value is the int ``0`` instead of an L/R/S string).
    PyArrow does not — it raises on conversion. Coercing to string preserves
    every distinguishable value and keeps NaN as null.
    """
    for col in df.select_dtypes(include="object").columns:
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        types = {type(v) for v in non_null}
        if len(types) > 1:
            df[col] = df[col].where(df[col].isna(), df[col].astype(str))
    return df


def _json_default(o):
    """Encode numpy scalars and pandas Timestamps the players dict tends to leak."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        v = float(o)
        return None if math.isnan(v) else v
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return str(o)


def _stringify_keys(obj):
    """Recursively cast dict keys to str so JSON can serialise int/numpy keys."""
    if isinstance(obj, dict):
        return {
            (str(int(k)) if isinstance(k, np.integer) else str(k)): _stringify_keys(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_stringify_keys(v) for v in obj]
    return obj


def _restore_int_keys(obj):
    """Restore numeric dict keys after a JSON round-trip.

    JSON only allows string keys; the league code paths use int year and int
    player-id keys. Convert back any key that parses cleanly as an int. Mixed
    keys like ``'2023-24'`` (NBA seasons) are left as strings.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            new_k = k
            if isinstance(k, str):
                stripped = k.lstrip("-")
                if stripped.isdigit():
                    new_k = int(k)
            out[new_k] = _restore_int_keys(v)
        return out
    if isinstance(obj, list):
        return [_restore_int_keys(v) for v in obj]
    return obj


def _write_players(players, paths) -> None:
    """Write ``players`` as parquet (DataFrame) or JSON (dict)."""
    json_path = Path(str(paths["players_json"]))
    parquet_path = Path(str(paths["players_parquet"]))
    if isinstance(players, pd.DataFrame):
        # Preserve the index — NFL's players DataFrame is indexed by player name.
        tmp = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
        players.to_parquet(tmp, engine="pyarrow", compression="zstd")
        tmp.replace(parquet_path)
        if json_path.is_file():
            json_path.unlink()
        return
    payload = _stringify_keys(players if isinstance(players, dict) else {})
    tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, default=_json_default)
    tmp.replace(json_path)
    if parquet_path.is_file():
        parquet_path.unlink()


def _read_players(paths):
    parquet_path = Path(str(paths["players_parquet"]))
    if parquet_path.is_file():
        return pd.read_parquet(parquet_path, engine="pyarrow")
    json_path = Path(str(paths["players_json"]))
    if json_path.is_file():
        with json_path.open() as f:
            return _restore_int_keys(json.load(f))
    return None


def read_gamelog(league: str) -> dict:
    """Return ``{"players", "gamelog", "teamlog"}`` for the league.

    Reads parquet first; falls back to ``{league}_data.dat`` pickle if the
    parquet files are absent. Mirrors ``read_history`` so installs that
    haven't run the migration script keep working.
    """
    paths = _gamelog_paths(league)
    gp = Path(str(paths["gamelog"]))
    if gp.is_file():
        gamelog = pd.read_parquet(gp, engine="pyarrow")
        teamlog_path = Path(str(paths["teamlog"]))
        teamlog = (
            pd.read_parquet(teamlog_path, engine="pyarrow")
            if teamlog_path.is_file()
            else pd.DataFrame()
        )
        players = _read_players(paths)
        if players is None:
            players = {}
        return {"players": players, "gamelog": gamelog, "teamlog": teamlog}

    legacy = Path(str(paths["legacy_pickle"]))
    if legacy.is_file():
        with legacy.open("rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            return {
                "players": obj.get("players", {}),
                "gamelog": obj.get("gamelog", pd.DataFrame()),
                "teamlog": obj.get("teamlog", pd.DataFrame()),
            }
        # NFL legacy format pre-dates the dict wrapper: a bare gamelog DataFrame.
        return {"players": {}, "gamelog": obj, "teamlog": pd.DataFrame()}

    return {"players": {}, "gamelog": pd.DataFrame(), "teamlog": pd.DataFrame()}


def write_gamelog(league: str, gamelog: pd.DataFrame, teamlog: pd.DataFrame, players) -> None:
    """Atomic-write per-league gamelog/teamlog parquet + players sidecar."""
    paths = _gamelog_paths(league)
    gamelog = _coerce_object_columns_to_str(gamelog.copy())
    _atomic_write_parquet(gamelog, paths["gamelog"], compression="zstd")
    if isinstance(teamlog, pd.DataFrame) and not teamlog.empty:
        teamlog = _coerce_object_columns_to_str(teamlog.copy())
        _atomic_write_parquet(teamlog, paths["teamlog"], compression="zstd")
    _write_players(players, paths)
