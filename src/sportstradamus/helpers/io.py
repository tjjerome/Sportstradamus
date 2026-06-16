"""Atomic parquet/JSON IO + schema converters for the data hot path.

All writers stage to a ``<path>.tmp`` and then ``os.replace()`` to the target
path, so the dashboard never reads a torn file when a pipeline writes
mid-render.

The ``Offers`` column in history is in-memory ``list[tuple]`` of mixed types.
PyArrow needs a ``list<struct>`` for that, so the converters round-trip
tuple <-> dict at the parquet boundary. Every other consumer of the column
keeps its tuple-indexed semantics.

Writes target parquet only. The legacy ``.dat`` klepto pickles have been
removed from the data package; readers no longer fall back to them.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sportstradamus import data

# Layout: data/config/ (live configs), data/runtime/ (job state),
# data/training/ (training outputs), data/leagues/{league}/ (per-league files),
# data/models/ (model pickles).
_RUNTIME_DIR = pkg_resources.files(data) / "runtime"
_TRAINING_DIR = pkg_resources.files(data) / "training"
HISTORY_PATH = _RUNTIME_DIR / "history.parquet"
PARLAY_HIST_PATH = _RUNTIME_DIR / "parlay_hist.parquet"
CURRENT_OFFERS_PATH = _RUNTIME_DIR / "current_offers.parquet"
CURRENT_PARLAYS_PATH = _RUNTIME_DIR / "current_parlays.parquet"
CURRENT_GAME_CORR_PATH = _RUNTIME_DIR / "current_game_corr.parquet"
CURRENT_GAME_CONTEXT_PATH = _RUNTIME_DIR / "current_game_context.parquet"
CURRENT_GAME_STORIES_PATH = _RUNTIME_DIR / "current_game_stories.parquet"
# Per-offer deep-dive detail prerender (comps-vs-opponent, volume trend, SHAP "other
# stats"), keyed by (League, Date, Player, Market, Opponent). Written by prophecize.
CURRENT_OFFER_DETAILS_PATH = _RUNTIME_DIR / "current_offer_details.parquet"
CURRENT_PICKEM_PATH = _RUNTIME_DIR / "current_pickem.parquet"
CURRENT_META_PATH = _RUNTIME_DIR / "current_meta.json"
MODEL_STATS_PATH = _TRAINING_DIR / "model_stats.parquet"
# Plain-text mirror of model_stats.parquet for filesystem browsing in VSCode.
# Atomically regenerated from the parquet on every training write; never read
# by code (parquet is the authoritative source on disagreement).
MODEL_STATS_CSV_PATH = _TRAINING_DIR / "model_stats.csv"
LIVE_METRICS_PATH = _RUNTIME_DIR / "live_metrics_per_market.parquet"
# Precomputed strategy x horizon profit-sim grid (Receipts reads it instead of
# running the Monte-Carlo backtest at page load). Written nightly by reflect.
PROFIT_SIM_SUMMARY_PATH = _RUNTIME_DIR / "profit_sim_summary.parquet"
# User-built slips saved from the dashboard ("Lock it in"); graded by nightly.
USER_SLIPS_PATH = _RUNTIME_DIR / "user_slips.parquet"

# Root for trained model pickles. model_pickle_path builds the per-cell path
# from this root; prune_model_pickle deletes one to dark-out a withheld cell.
MODELS_DIR = pkg_resources.files(data) / "models"

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


def market_file_slug(league: str, market: str) -> str:
    """Slugify a ``(league, market)`` pair for on-disk filenames.

    Joins league and market with an underscore and replaces spaces with hyphens,
    e.g. ``("NFL", "rushing tds") -> "NFL_rushing-tds"``. Shared by the model
    pickle, training-data, and test-set path builders so they all agree.

    Args:
        league: League code (e.g. ``"NBA"``).
        market: Market stem, possibly containing spaces.

    Returns:
        The slug string (no directory, no extension).
    """
    return "_".join([league, market]).replace(" ", "-")


def model_pickle_path(league: str, market: str) -> Path:
    """Return the production model-pickle path for a ``(league, market)`` cell.

    Built from :func:`market_file_slug`, e.g.
    ``("NFL", "rushing tds") -> models/NFL_rushing-tds.mdl``. This is the single
    path both the training writer and the prediction loader use, and the one
    ``prune_model_pickle`` deletes to dark-out a withheld cell.

    Args:
        league: League code (e.g. ``"NBA"``).
        market: Market stem, possibly containing spaces.

    Returns:
        The ``.mdl`` path under ``data/models/`` (not guaranteed to exist).
    """
    return Path(str(MODELS_DIR / f"{market_file_slug(league, market)}.mdl"))


def prune_model_pickle(league: str, market: str) -> bool:
    """Delete a cell's production model pickle so inference skips that market.

    Used by ``meditate`` to dark-out a cell marked ``shipped="withheld"`` in
    ``stat_meta.json``: with no pickle on disk, ``model_prob`` returns ``[]``
    and the market is not scored.

    Args:
        league: League code.
        market: Market stem.

    Returns:
        ``True`` if a pickle existed and was removed, ``False`` if none was present.
    """
    path = model_pickle_path(league, market)
    existed = path.exists()
    path.unlink(missing_ok=True)
    return existed


def _atomic_tmp(path) -> tuple[Path, Path]:
    """Resolve ``(target, target.tmp)`` for an atomic write, creating the parent.

    Snapshot dirs (``runtime/``, ``training/``) are gitignored and so absent on a
    fresh checkout or install; the write-temp-then-rename fails on a missing
    parent. Mirrors the mkdir the model-pickle writer does in training/pipeline.py.
    """
    path = Path(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path, path.with_suffix(path.suffix + ".tmp")


def _atomic_write_parquet(df: pd.DataFrame, path, compression: str | None = None) -> None:
    path, tmp = _atomic_tmp(path)
    df.to_parquet(tmp, engine="pyarrow", index=False, compression=compression)
    tmp.replace(path)


def _atomic_write_json(obj, path) -> None:
    path, tmp = _atomic_tmp(path)
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, default=str)
    tmp.replace(path)


def _atomic_write_csv(df: pd.DataFrame, path) -> None:
    path, tmp = _atomic_tmp(path)
    df.to_csv(tmp, index=False)
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


def _pad_legacy_offer(offer):
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
    """Read the prediction history."""
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
    """Read the parlay history."""
    df = read_parquet_safe(PARLAY_HIST_PATH)
    if df.empty:
        return df
    for col in _PARLAY_LIST_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_list_to_tuple)
    return df


# ---------------------------------------------------------------------------
# User-built slips (dashboard "Lock it in" -> nightly grading)
# ---------------------------------------------------------------------------


def read_user_slips() -> pd.DataFrame:
    """Return saved user slips, or an empty DataFrame when none are on disk."""
    return read_parquet_safe(USER_SLIPS_PATH)


def upsert_user_slip(row: dict) -> None:
    """Insert a slip or replace the existing one with the same ``slip_id``.

    "Lock it in" on a fresh slip appends; editing a locked slip and re-locking
    replaces its row in place (a re-lock resets the grading columns to pending).
    """
    existing = read_user_slips()
    if not existing.empty and "slip_id" in existing.columns:
        existing = existing.loc[existing["slip_id"] != row["slip_id"]]
        combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        combined = pd.DataFrame([row])
    _atomic_write_parquet(combined, USER_SLIPS_PATH)


def write_user_slips(df: pd.DataFrame) -> None:
    """Atomically rewrite the whole user-slips table (nightly grading writeback)."""
    _atomic_write_parquet(df, USER_SLIPS_PATH)


def delete_user_slip(slip_id: str) -> None:
    """Drop one slip by ``slip_id`` (the sidebar "Delete"); no-op if it's gone."""
    existing = read_user_slips()
    if existing.empty or "slip_id" not in existing.columns:
        return
    _atomic_write_parquet(existing.loc[existing["slip_id"] != slip_id], USER_SLIPS_PATH)


# ---------------------------------------------------------------------------
# Upcoming-events ledger (close-line scheduler input)
# ---------------------------------------------------------------------------

UPCOMING_EVENTS_PATH = _RUNTIME_DIR / "upcoming_events.json"


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
    base = pkg_resources.files(data) / "leagues" / league
    return {
        "gamelog": base / "gamelog.parquet",
        "teamlog": base / "teamlog.parquet",
        "players_json": base / "players.json",
        "players_parquet": base / "players.parquet",
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

    Returns empty DataFrames and an empty ``players`` dict when the parquet
    files are absent.
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
