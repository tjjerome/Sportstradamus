"""Shared data loading and state for the Sportstradamus dashboard.

All surfaces import from here to get cached DataFrames and filters.
"""

import importlib.resources as pkg_resources
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import streamlit as st

from sportstradamus import data
from sportstradamus.analysis import (
    _migrate_flat_history,
    explode_offers,
)
from sportstradamus.helpers.io import (
    CURRENT_GAME_CONTEXT_PATH,
    CURRENT_GAME_CORR_PATH,
    CURRENT_GAME_STORIES_PATH,
    CURRENT_META_PATH,
    CURRENT_OFFERS_PATH,
    CURRENT_PARLAYS_PATH,
    CURRENT_PICKEM_PATH,
    HISTORY_PATH,
    MODEL_STATS_PATH,
    PARLAY_HIST_PATH,
    USER_SLIPS_PATH,
    read_history,
    read_parlay_hist,
    read_parquet_safe,
    read_user_slips,
    write_history,
)

# Column names that differ across league gamelog parquets.
# Keys: player, date, opp (None if not available), home (None if not available).
GAMELOG_SCHEMA = {
    "NBA": {
        "file": "leagues/nba/gamelog.parquet",
        "player": "PLAYER_NAME",
        "date": "GAME_DATE",
        "opp": "OPP",
        "home": "HOME",
    },
    "WNBA": {
        "file": "leagues/wnba/gamelog.parquet",
        "player": "PLAYER_NAME",
        "date": "GAME_DATE",
        "opp": "OPP",
        "home": "HOME",
    },
    "MLB": {
        "file": "leagues/mlb/gamelog.parquet",
        "player": "playerName",
        "date": "gameDate",
        "opp": "opponent",
        "home": "home",
    },
    "NHL": {
        "file": "leagues/nhl/gamelog.parquet",
        "player": "playerName",
        "date": "gameDate",
        "opp": "opponent",
        "home": "home",
    },
    "NFL": {
        "file": "leagues/nfl/gamelog.parquet",
        "player": "player display name",
        "date": None,
        "opp": None,
        "home": None,
    },
}

PRED_BANNER_COLOR = "#1f4e79"  # deep teal
STATS_BANNER_COLOR = "#2d6a4f"  # forest green

# Stats-page time-window menu: label -> lookback days (None = all time).
TIMEFRAME_OPTIONS = {
    "All time": None,
    "Last 7 days": 7,
    "Last 30 days": 30,
    "Last 3 months": 91,
    "Last 6 months": 183,
    "Last year": 365,
}

# Cache TTL for parquet/JSON loaders. mtime is the primary invalidation signal;
# this is a safety net so a cache entry can't outlive 10 min even if the host
# filesystem ever misreports mtime.
_CACHE_TTL_SECONDS = 600

# TTL for static config loaders (stat_map.json). No mtime key — these files
# change only on deploy, so a 1-hour TTL is sufficient.
_STATIC_CONFIG_TTL_SECONDS = 3600


def _mtime(path) -> float:
    """Return the parquet/JSON file's mtime, or 0.0 if absent.

    Used as a cache key so Streamlit re-reads when cron rewrites a snapshot.
    Passed by value into private ``_load_*_cached`` helpers; the function
    body never touches the value, but Streamlit hashes it for the key.
    """
    p = Path(str(path))
    return p.stat().st_mtime if p.is_file() else 0.0


def format_ts(ts: str) -> str:
    """Convert a raw timestamp string to a friendly local-time label.

    Handles UTC ISO strings (with Z suffix), naive local strings, and fallback values.
    Returns a human-readable format like "May 15 at 10:42 PM".
    """
    if not ts or ts in ("no run on record", "no meditate run on record"):
        return ts
    ts_norm = ts.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(ts_norm)
    except ValueError:
        try:
            dt = datetime.strptime(ts_norm, "%Y-%m-%d %H:%M")
        except ValueError:
            return ts
    local_dt = dt.astimezone()
    return local_dt.strftime("%b %-d at %-I:%M %p")


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner="Loading prediction history...")
def _load_history_cached(mtime: float) -> pd.DataFrame:
    """Cached read of the prediction-history parquet.

    ``mtime`` is the file modification time; Streamlit hashes it as part of
    the cache key so a fresh cron write invalidates the entry.
    """
    history = read_history()
    if history.empty:
        return history

    # Migrate old flat schema → normalized (one row per prediction, Offers list)
    if "Offers" not in history.columns:
        history = _migrate_flat_history(history)
        write_history(history)

    # Ensure prediction-level columns exist for backward compatibility
    for col in ["Dist", "CV", "Model Param", "Gate", "Temperature", "Disp Cal", "Step", "Actual"]:
        if col not in history.columns:
            history[col] = np.nan
    return history


def load_history() -> pd.DataFrame:
    """Load prediction history from parquet (mtime-keyed cache).

    Migrates old flat schema (no Offers column) to the normalized one-row-per-
    prediction shape with an Offers list, then writes the migrated frame back.
    Offers round-trip as list[tuple] (CLV-aware 9-tuples; 6-tuples padded).
    """
    return _load_history_cached(_mtime(HISTORY_PATH))


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner="Loading parlay history...")
def _load_parlays_cached(mtime: float) -> pd.DataFrame:
    parlays = read_parlay_hist()
    if parlays.empty:
        return parlays

    # Backward compat: ensure correlation/Indep columns exist for older runs.
    for col in ["Corr Pairs", "Boost Pairs", "Indep P", "Indep PB"]:
        if col not in parlays.columns:
            parlays[col] = np.nan
    return parlays


def load_parlays() -> pd.DataFrame:
    """Parlay history from parquet, keyed on file mtime so cron rewrites invalidate the cache."""
    return _load_parlays_cached(_mtime(PARLAY_HIST_PATH))


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner="Loading current offers...")
def _load_current_offers_cached(mtime: float) -> pd.DataFrame:
    return read_parquet_safe(CURRENT_OFFERS_PATH)


def load_current_offers() -> pd.DataFrame:
    """Today's scored offers from the latest ``prophecize`` snapshot."""
    return _load_current_offers_cached(_mtime(CURRENT_OFFERS_PATH))


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner="Loading current parlays...")
def _load_current_parlays_cached(mtime: float) -> pd.DataFrame:
    return read_parquet_safe(CURRENT_PARLAYS_PATH)


def load_current_parlays() -> pd.DataFrame:
    """Today's parlay candidates from the latest ``prophecize`` snapshot."""
    return _load_current_parlays_cached(_mtime(CURRENT_PARLAYS_PATH))


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner="Loading correlation slices...")
def _load_current_game_corr_cached(mtime: float) -> pd.DataFrame:
    return read_parquet_safe(CURRENT_GAME_CORR_PATH)


def load_current_game_corr() -> pd.DataFrame:
    """Per-game leg-pair correlation slices from the latest ``prophecize`` snapshot.

    Columns ``League, Game, leg_a, leg_b, rho`` with leg key ``Player|Market|Bet``;
    joins ``current_offers`` on the canonical ``Game`` key. Feeds the slip rail
    copula, the constellation, and the swap dialog.
    """
    return _load_current_game_corr_cached(_mtime(CURRENT_GAME_CORR_PATH))


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner="Loading story menu...")
def _load_current_game_stories_cached(mtime: float) -> pd.DataFrame:
    return read_parquet_safe(CURRENT_GAME_STORIES_PATH)


def load_current_game_stories() -> pd.DataFrame:
    """Per-(platform, game) story menu from the latest ``prophecize`` snapshot.

    Keys ``platform, League, Game, story_id, objective`` with ``legs`` a JSON
    leg-desc list; ``headline``/``joint_p``/``model_ev``/``kelly_stake``/
    ``bet_size`` per row. Seeds the Slips constellation builder (Bankroll
    Builder / Shoot the Moon).
    """
    return _load_current_game_stories_cached(_mtime(CURRENT_GAME_STORIES_PATH))


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner="Loading game context...")
def _load_current_game_context_cached(mtime: float) -> pd.DataFrame:
    return read_parquet_safe(CURRENT_GAME_CONTEXT_PATH)


def load_current_game_context() -> pd.DataFrame:
    """Per-game context (total/spread/shape/pos_edges) from the latest snapshot.

    One row per ``League, Game, Date``; feeds the live thesis regen in the
    constellation builder via ``slip_engine.slip_headline``.
    """
    return _load_current_game_context_cached(_mtime(CURRENT_GAME_CONTEXT_PATH))


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner=False)
def _load_user_slips_cached(mtime: float) -> pd.DataFrame:
    return read_user_slips()


def load_user_slips() -> pd.DataFrame:
    """User-saved slips (dashboard 'Lock it in'); graded nightly. mtime-keyed cache."""
    return _load_user_slips_cached(_mtime(USER_SLIPS_PATH))


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner="Loading pickem entries...")
def _load_current_pickem_cached(mtime: float) -> pd.DataFrame:
    return read_parquet_safe(CURRENT_PICKEM_PATH)


def load_current_pickem() -> pd.DataFrame:
    """Today's Underdog Pick'em entries from the latest ``prophecize`` snapshot."""
    return _load_current_pickem_cached(_mtime(CURRENT_PICKEM_PATH))


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner=False)
def _load_gamelog_cached(league: str, mtime: float) -> pd.DataFrame:
    schema = GAMELOG_SCHEMA.get(league)
    if schema is None:
        return pd.DataFrame()
    return read_parquet_safe(pkg_resources.files(data) / schema["file"])


def load_gamelog(league: str) -> pd.DataFrame:
    """Load the full season gamelog parquet for a league (mtime-keyed cache).

    Returns an empty DataFrame if the league is unrecognised or the file is
    missing.  Callers are responsible for filtering to the relevant player and
    stat column.
    """
    schema = GAMELOG_SCHEMA.get(league)
    gamelog_path = pkg_resources.files(data) / schema["file"] if schema else None
    return _load_gamelog_cached(league, _mtime(gamelog_path) if gamelog_path else 0.0)


@st.cache_data(ttl=_CACHE_TTL_SECONDS)
def _load_current_meta_cached(mtime: float) -> dict:
    if not CURRENT_META_PATH.is_file():
        return {}
    try:
        with open(CURRENT_META_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def load_current_meta() -> dict:
    """Snapshot metadata (timestamp, leagues, platforms, row counts; mtime-keyed)."""
    return _load_current_meta_cached(_mtime(CURRENT_META_PATH))


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner="Loading model training stats...")
def _load_model_stats_cached(mtime: float) -> pd.DataFrame:
    return read_parquet_safe(MODEL_STATS_PATH)


def load_model_stats() -> pd.DataFrame:
    """Per-(league, market, metric_row) training diagnostics from ``meditate``."""
    return _load_model_stats_cached(_mtime(MODEL_STATS_PATH))


def render_banner(kind: Literal["predictions", "stats"], subtitle: str = "") -> None:
    """Render a colored section banner so Predictions vs Stats are visually distinct."""
    if kind == "predictions":
        color, label = PRED_BANNER_COLOR, "Predictions"
    else:
        color, label = STATS_BANNER_COLOR, "Stats"
    sub = f" — {subtitle}" if subtitle else ""
    st.markdown(
        f'<div style="background:{color};padding:10px 14px;border-radius:6px;'
        f'color:white;margin-bottom:14px;font-size:14px">'
        f"<b>{label}</b>{sub}</div>",
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=_STATIC_CONFIG_TTL_SECONDS, show_spinner="Loading stat map...")
def load_stat_map() -> dict:
    """Load the stat name mapping config (static, long-lived cache)."""
    with open(pkg_resources.files(data) / "config" / "stat_map.json") as f:
        return json.load(f)


def load_resolve_meta() -> dict:
    """Load nightly resolution metadata (last run time, counts).

    Returns a dict with keys: last_run, history_resolved, parlays_resolved.
    Returns empty dict if file doesn't exist (nightly hasn't run yet).
    """
    meta_path = pkg_resources.files(data) / "runtime" / "resolve_meta.json"
    try:
        with open(meta_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def sport_filtered(df: pd.DataFrame) -> pd.DataFrame:
    """Narrow ``df`` to the app-level sport switch (``st.session_state["sport"]``).

    Returns ``df`` unchanged when "All" is selected or there is no ``League``
    column. Empty-result handling stays with the caller — each surface words
    its own message.
    """
    sport = st.session_state.get("sport", "All")
    if sport == "All" or "League" not in df.columns:
        return df
    return df.loc[df["League"] == sport]


def get_filtered_history(
    history: pd.DataFrame,
    leagues: list | None = None,
    platforms: list | None = None,
    markets: list | None = None,
    date_range: tuple | None = None,
    min_model_p: float | None = None,
) -> pd.DataFrame:
    """Explode offers and apply sidebar filters.

    Returns a per-offer DataFrame with columns: all prediction-level cols +
    Line, Boost, Platform, Bet, Win Prob, Market Prob, Result, Hit, Model EV, Market EV, Kelly.
    """
    df = explode_offers(history)
    if df.empty:
        return df

    df = df.dropna(subset=["Result"])
    df = df.loc[df["Result"] != "Push"]

    if leagues:
        df = df.loc[df["League"].isin(leagues)]
    if platforms:
        df = df.loc[df["Platform"].isin(platforms)]
    if markets:
        df = df.loc[df["Market"].isin(markets)]
    if date_range:
        df["_date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
        df = df.loc[(df["_date"] >= date_range[0]) & (df["_date"] <= date_range[1])]
    if min_model_p is not None:
        prob_col = "Win Prob" if "Win Prob" in df.columns and df["Win Prob"].notna().any() else "Model EV"
        df = df.loc[df[prob_col] >= min_model_p]

    return df


def get_prediction_history(
    history: pd.DataFrame,
    leagues: list | None = None,
    date_range: tuple | None = None,
) -> pd.DataFrame:
    """Return prediction-level rows (no explosion) for CRPS/coverage analysis.

    Filters by league and date range but does NOT explode offers.
    """
    df = history.copy()

    if leagues:
        df = df.loc[df["League"].isin(leagues)]
    if date_range:
        df["_date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
        df = df.loc[(df["_date"] >= date_range[0]) & (df["_date"] <= date_range[1])]

    return df


def _extract_platforms(history):
    """Extract unique platform names from the Offers column (or legacy Platform)."""
    if "Offers" in history.columns:
        return sorted(
            {
                str(offer[2])
                for offers in history["Offers"].dropna()
                if isinstance(offers, list)
                for offer in offers
                if len(offer) >= 3 and offer[2]
            }
        )
    if "Platform" in history.columns:
        return sorted(set(history["Platform"].dropna().unique()))
    return []


def sidebar_filters(
    history: pd.DataFrame,
    parlays: pd.DataFrame | None = None,
    key_prefix: str = "",
) -> dict:
    """Render sidebar filters and return filter values."""
    if st.sidebar.button(
        "Refresh data",
        key=f"{key_prefix}refresh_data",
        help="Force-reload all parquet snapshots, bypassing the data cache.",
    ):
        st.cache_data.clear()
        st.rerun()
    st.sidebar.header("Filters")

    if not history.empty:
        dates = pd.to_datetime(history["Date"], errors="coerce").dropna()
        min_date = dates.min().date()
        max_date = dates.max().date()
    else:
        min_date = datetime.today().date() - timedelta(days=365)
        max_date = datetime.today().date()

    date_range = st.sidebar.date_input(
        "Date range",
        value=(max(min_date, max_date - timedelta(days=90)), max_date),
        min_value=min_date,
        max_value=max_date,
        key=f"{key_prefix}date_range",
    )
    if len(date_range) == 1:
        date_range = (date_range[0], max_date)

    leagues = sorted(history["League"].dropna().unique()) if not history.empty else []
    selected_leagues = st.sidebar.multiselect(
        "Leagues", leagues, default=leagues, key=f"{key_prefix}leagues"
    )

    platforms = _extract_platforms(history)
    selected_platforms = st.sidebar.multiselect(
        "Platforms", platforms, default=platforms, key=f"{key_prefix}platforms"
    )

    if not history.empty and "Dist" in history.columns:
        coverage = history["Dist"].notna().mean()
        st.sidebar.metric("Distribution Data Coverage", f"{coverage:.0%}")

    return {
        "date_range": date_range,
        "leagues": selected_leagues,
        "platforms": selected_platforms,
    }


def load_resolved_history_or_stop() -> pd.DataFrame:
    """Load prediction history for a stats page, or ``st.stop()`` the page if absent.

    Shared stats-page preamble: stops with a warning when no history exists,
    otherwise captions the last-resolved timestamp and returns the frame.
    """
    history = load_history()
    if history.empty:
        st.warning("No prediction history found.")
        st.stop()
    meta = load_resolve_meta()
    if meta.get("last_run"):
        st.caption(f"Data last resolved: {format_ts(meta['last_run'])}")
    return history


def filtered_history_or_stop(history: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply sidebar ``filters`` to ``history``, or ``st.stop()`` the page if nothing remains."""
    df = get_filtered_history(
        history,
        leagues=filters["leagues"],
        platforms=filters["platforms"],
        date_range=filters["date_range"],
    )
    if df.empty:
        st.info("No resolved predictions match the current filters.")
        st.stop()
    return df
