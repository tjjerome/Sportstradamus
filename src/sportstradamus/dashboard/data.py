"""Shared data loading and state for the Sportstradamus dashboard.

All surfaces import from here to get cached DataFrames and filters.
"""

import importlib.resources as pkg_resources
import json
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sportstradamus import data
from sportstradamus.analysis import annotate_offer_outcomes
from sportstradamus.dashboard.components.lab_filters import FILTER_AXES
from sportstradamus.helpers.io import (
    CALIBRATION_SUMMARY_PATH,
    CURRENT_GAME_CONTEXT_PATH,
    CURRENT_GAME_CORR_PATH,
    CURRENT_GAME_STORIES_PATH,
    CURRENT_META_PATH,
    CURRENT_OFFER_DETAILS_PATH,
    CURRENT_OFFERS_PATH,
    CURRENT_PARLAYS_PATH,
    HISTORY_PATH,
    MODEL_STATS_PATH,
    PARLAY_HIST_PATH,
    PROFIT_SIM_SUMMARY_PATH,
    USER_SLIPS_PATH,
    parlay_hist_mtime,
    read_history,
    read_parlay_hist,
    read_parquet_safe,
    read_user_slips,
)
from sportstradamus.history_schema import PREDICTION_KEY, PREDICTION_LEVEL_COLS
from sportstradamus.prediction.stories.context import ctxs_from_frame

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
    Passed by value into private ``_load_*_cached`` helpers alongside the
    path itself; mtime alone is not a unique cache key across tests, which
    each point the same path constant at a different tmp file — two
    ``_load_*_cached`` calls a test apart can otherwise collide on a stale
    entry. The function body doesn't always touch either value directly,
    but Streamlit hashes both as the key.
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
def _load_history_cached(path: Path, mtime: float) -> pd.DataFrame:
    """Cached read of the prediction-history parquet.

    ``path`` + ``mtime`` form the cache key: path alone so two different
    paths (e.g. two tests' tmp fixtures) never share an entry; mtime so a
    fresh cron write invalidates it. ``read_history()`` re-resolves the path
    itself, so ``path`` isn't touched in the body beyond keying the cache.
    """
    history = read_history()
    if history.empty:
        return history

    # Ensure prediction-level columns exist for backward compatibility
    for col in [*PREDICTION_LEVEL_COLS, "Actual"]:
        if col not in history.columns:
            history[col] = np.nan
    return history


def load_history() -> pd.DataFrame:
    """Load prediction history from parquet (path+mtime-keyed cache).

    Flat one-row-per-(prediction x book offer) schema; prediction-level
    columns are duplicated across every offer row for the same prediction.
    """
    return _load_history_cached(HISTORY_PATH, _mtime(HISTORY_PATH))


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner="Loading parlay history...")
def _load_parlays_cached(path: Path, mtime: float, columns: tuple[str, ...] | None) -> pd.DataFrame:
    parlays = read_parlay_hist(columns=list(columns) if columns else None)
    if parlays.empty:
        return parlays

    if columns is None:
        # Full reads back-fill correlation/Indep columns absent from older runs; a
        # projected read asks for an explicit, known-present column set, so it skips this.
        for col in ["Corr Pairs", "Boost Pairs", "Indep P", "Indep PB"]:
            if col not in parlays.columns:
                parlays[col] = np.nan
    return parlays


def load_parlays(columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Parlay history from parquet, keyed on path+mtime so cron rewrites invalidate the cache.

    ``columns`` projects the read to a scalar subset (Lab Correlations does this — the
    full 1.7M-row history carries multi-GB ``list<float>`` struct columns it never plots).
    """
    key = tuple(columns) if columns else None
    return _load_parlays_cached(PARLAY_HIST_PATH, parlay_hist_mtime(), key)


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner="Loading current offers...")
def _load_current_offers_cached(path: Path, mtime: float) -> pd.DataFrame:
    return read_parquet_safe(path)


def load_current_offers() -> pd.DataFrame:
    """Today's scored offers from the latest ``prophecize`` snapshot."""
    return _load_current_offers_cached(CURRENT_OFFERS_PATH, _mtime(CURRENT_OFFERS_PATH))


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner="Loading current parlays...")
def _load_current_parlays_cached(path: Path, mtime: float) -> pd.DataFrame:
    return read_parquet_safe(path)


def load_current_parlays() -> pd.DataFrame:
    """Today's parlay candidates from the latest ``prophecize`` snapshot."""
    return _load_current_parlays_cached(CURRENT_PARLAYS_PATH, _mtime(CURRENT_PARLAYS_PATH))


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner="Loading correlation slices...")
def _load_current_game_corr_cached(path: Path, mtime: float) -> pd.DataFrame:
    return read_parquet_safe(path)


def load_current_game_corr() -> pd.DataFrame:
    """Per-game leg-pair correlation slices from the latest ``prophecize`` snapshot.

    Columns ``League, Game, leg_a, leg_b, rho`` with leg key ``Player|Market|Bet``;
    joins ``current_offers`` on the canonical ``Game`` key. Feeds the slip rail
    copula, the constellation, and the swap dialog.
    """
    return _load_current_game_corr_cached(CURRENT_GAME_CORR_PATH, _mtime(CURRENT_GAME_CORR_PATH))


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner="Loading story menu...")
def _load_current_game_stories_cached(path: Path, mtime: float) -> pd.DataFrame:
    return read_parquet_safe(path)


def load_current_game_stories() -> pd.DataFrame:
    """Per-(platform, game) story menu from the latest ``prophecize`` snapshot.

    Keys ``platform, League, Game, story_id, objective`` with ``legs`` a JSON
    leg-desc list; ``headline``/``joint_p``/``model_ev``/``kelly_stake``/
    ``bet_size`` per row. Seeds the Slips constellation builder (Bankroll
    Builder / Shoot the Moon).
    """
    return _load_current_game_stories_cached(
        CURRENT_GAME_STORIES_PATH, _mtime(CURRENT_GAME_STORIES_PATH)
    )


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner=False)
def _load_profit_sim_summary_cached(path: Path, mtime: float) -> pd.DataFrame:
    return read_parquet_safe(path)


def load_profit_sim_summary() -> pd.DataFrame:
    """Precomputed strategy x horizon profit-sim grid from the latest ``reflect`` run.

    Columns ``Strategy, Horizon, ROI, Sharpe, Max Drawdown, Win%``. Receipts reads
    this instead of running the Monte-Carlo backtest at page load; empty when
    ``reflect`` has not written it yet.
    """
    return _load_profit_sim_summary_cached(PROFIT_SIM_SUMMARY_PATH, _mtime(PROFIT_SIM_SUMMARY_PATH))


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner=False)
def _load_calibration_summary_cached(path: Path, mtime: float) -> pd.DataFrame:
    return read_parquet_safe(path)


def load_calibration_summary() -> pd.DataFrame:
    """Precomputed reliability (prob bin x alt-line split) grid from the latest ``reflect`` run.

    Columns ``Alt Line, Bin, Predicted, Actual, N, ECE, ROI``. Receipts reads this
    instead of re-binning history at page load; empty when ``reflect`` has not
    written it yet.
    """
    return _load_calibration_summary_cached(
        CALIBRATION_SUMMARY_PATH, _mtime(CALIBRATION_SUMMARY_PATH)
    )


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner="Loading game context...")
def _load_current_game_context_cached(path: Path, mtime: float) -> pd.DataFrame:
    return read_parquet_safe(path)


def load_current_game_context() -> pd.DataFrame:
    """Per-game context (total/spread/shape/pos_edges) from the latest snapshot.

    One row per ``League, Game, Date``; feeds the live thesis regen in the
    constellation builder via ``slip_engine.slip_headline``.
    """
    return _load_current_game_context_cached(
        CURRENT_GAME_CONTEXT_PATH, _mtime(CURRENT_GAME_CONTEXT_PATH)
    )


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner=False)
def _load_game_ctxs_cached(ctx_mtime: float, corr_mtime: float) -> dict:
    return ctxs_from_frame(load_current_game_context(), load_current_game_corr())


def load_game_ctxs() -> dict:
    """Per-game :class:`GameCtx` map, built once per snapshot instead of every rerun.

    ``ctxs_from_frame`` walks the whole ``current_game_corr`` slice (tens of thousands of
    rows) to attach each game's rho, so rebuilding it on every constellation click was the
    Games surface's biggest per-rerun cost. Cache-keyed on both source files' mtimes so a
    fresh ``prophecize`` snapshot still refreshes it.
    """
    return _load_game_ctxs_cached(_mtime(CURRENT_GAME_CONTEXT_PATH), _mtime(CURRENT_GAME_CORR_PATH))


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner="Loading offer details...")
def _load_current_offer_details_cached(path: Path, mtime: float) -> pd.DataFrame:
    return read_parquet_safe(path)


def load_current_offer_details() -> pd.DataFrame:
    """Prerendered deep-dive detail rows from the latest ``prophecize`` snapshot.

    One row per ``League, Date, Player, Market, Opponent`` with JSON-string
    ``comps_vs_opp`` / ``volume_trend`` / ``other_stats`` columns the deep-dive
    Comps and Other-stats tabs decode. Empty when the sidecar is absent.
    """
    return _load_current_offer_details_cached(
        CURRENT_OFFER_DETAILS_PATH, _mtime(CURRENT_OFFER_DETAILS_PATH)
    )


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner=False)
def _load_user_slips_cached(path: Path, mtime: float) -> pd.DataFrame:
    return read_user_slips()


def load_user_slips() -> pd.DataFrame:
    """User-saved slips (dashboard 'Lock it in'); graded nightly. path+mtime-keyed cache."""
    return _load_user_slips_cached(USER_SLIPS_PATH, _mtime(USER_SLIPS_PATH))


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


_CORR_MARKET_SUMMARY_COLUMNS = ["market_a", "market_b", "rho_mean", "n_teams", "scope"]


def _corr_market_summary_path(league: str) -> Path:
    league_dir = pkg_resources.files(data) / "leagues" / league.lower()
    return Path(str(league_dir / "corr_market_summary.parquet"))


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner=False)
def _load_corr_market_summary_cached(league: str, mtime: float) -> pd.DataFrame:
    df = read_parquet_safe(_corr_market_summary_path(league))
    return df if not df.empty else pd.DataFrame(columns=_CORR_MARKET_SUMMARY_COLUMNS)


def load_corr_market_summary(league: str) -> pd.DataFrame:
    """Market-pair mean-correlation summary for a league (mtime-keyed cache).

    Columns ``market_a, market_b, rho_mean, n_teams, scope`` (scope is
    ``same_team`` or ``opposing``), written by ``training.correlate``'s
    ``_write_corr_outputs`` alongside the dashboard-forbidden per-team corr
    parquets. Returns an empty DataFrame for a league with no corr data
    generated yet — callers caption that rather than crash.
    """
    path = _corr_market_summary_path(league)
    return _load_corr_market_summary_cached(league, _mtime(path))


@st.cache_data(ttl=_CACHE_TTL_SECONDS)
def _load_current_meta_cached(path: Path, mtime: float) -> dict:
    if not path.is_file():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def load_current_meta() -> dict:
    """Snapshot metadata (timestamp, leagues, platforms, row counts; path+mtime-keyed)."""
    return _load_current_meta_cached(CURRENT_META_PATH, _mtime(CURRENT_META_PATH))


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner="Loading model training stats...")
def _load_model_stats_cached(path: Path, mtime: float) -> pd.DataFrame:
    return read_parquet_safe(path)


def load_model_stats() -> pd.DataFrame:
    """Per-(league, market, metric_row) training diagnostics from ``meditate``."""
    return _load_model_stats_cached(MODEL_STATS_PATH, _mtime(MODEL_STATS_PATH))


@st.cache_data(ttl=_STATIC_CONFIG_TTL_SECONDS, show_spinner="Loading stat map...")
def load_stat_map() -> dict:
    """Load the stat name mapping config (static, long-lived cache)."""
    with open(pkg_resources.files(data) / "config" / "stat_map.json") as f:
        return json.load(f)


# Axes that are always present on a stat_meta.json cell; the rest of
# FILTER_AXES (blending, hpo_selection, count_dispersion_objective, zinb_mode)
# are sparse Optuna search axes that default to "none" when absent.
_ALWAYS_PRESENT_AXES = ("dist", "target_normalization", "posthoc", "shipped")


@st.cache_data(ttl=_STATIC_CONFIG_TTL_SECONDS, show_spinner="Loading stat meta...")
def load_stat_meta() -> pd.DataFrame:
    """One row per ``(league, market)`` cell of ``stat_meta.json``, sparse axes defaulted.

    Columns: ``league, market`` + :data:`~sportstradamus.dashboard.components.lab_filters.FILTER_AXES`
    — ``dist, target_normalization, posthoc, blending, hpo_selection,
    count_dispersion_objective, zinb_mode, shipped``. Feeds the shared Lab
    filter panel (``lab_filters.render_lab_filters``); a committed-config read,
    so no archive/pipeline dependency.
    """
    with open(pkg_resources.files(data) / "config" / "stat_meta.json") as f:
        raw: dict[str, dict[str, dict]] = json.load(f)

    rows = []
    for league, markets in raw.items():
        for market, cell in markets.items():
            row = {"league": league, "market": market}
            for axis in FILTER_AXES:
                row[axis] = cell[axis] if axis in _ALWAYS_PRESENT_AXES else cell.get(axis, "none")
            rows.append(row)

    return pd.DataFrame(rows, columns=["league", "market", *FILTER_AXES])


@st.cache_data(ttl=_STATIC_CONFIG_TTL_SECONDS, show_spinner=False)
def load_stat_tooltips() -> dict[str, dict[str, str]]:
    """Per-league ``{stat: tooltip}`` glosses for the deep-dive Other-stats tab."""
    with open(pkg_resources.files(data) / "config" / "stat_tooltips.json") as f:
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

    Matches a ``League`` or lowercase ``league`` column (the training stats frame uses the
    latter). Returns ``df`` unchanged when "All" is selected or neither column is present.
    Empty-result handling stays with the caller — each surface words its own message.
    """
    sport = st.session_state.get("sport", "All")
    if sport == "All":
        return df
    league_col = next((c for c in ("League", "league") if c in df.columns), None)
    if league_col is None:
        return df
    return df.loc[df[league_col] == sport]


def get_filtered_history(
    history: pd.DataFrame,
    leagues: list | None = None,
    platforms: list | None = None,
    markets: list | None = None,
    date_range: tuple | None = None,
    min_model_p: float | None = None,
) -> pd.DataFrame:
    """Annotate offer outcomes and apply sidebar filters.

    Returns a per-offer DataFrame with columns: all prediction-level cols +
    Line, Boost, Platform, Bet, Win Prob, Market Prob, Result, Hit, Model EV, Market EV, Kelly.
    """
    df = annotate_offer_outcomes(history)
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
        prob_col = (
            "Win Prob" if "Win Prob" in df.columns and df["Win Prob"].notna().any() else "Model EV"
        )
        df = df.loc[df[prob_col] >= min_model_p]

    return df


def get_prediction_history(
    history: pd.DataFrame,
    leagues: list | None = None,
    date_range: tuple | None = None,
) -> pd.DataFrame:
    """Return one row per prediction (deduped on the identity key) for CRPS/coverage analysis.

    Filters by league and date range, then collapses the flat one-row-per-offer
    frame down to one row per :data:`~sportstradamus.history_schema.PREDICTION_KEY`.
    """
    df = history.copy()

    if leagues:
        df = df.loc[df["League"].isin(leagues)]
    if date_range:
        df["_date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
        df = df.loc[(df["_date"] >= date_range[0]) & (df["_date"] <= date_range[1])]

    return df.drop_duplicates(subset=PREDICTION_KEY, keep="last")


def _extract_platforms(history):
    """Extract unique platform names from the Platform column."""
    if "Platform" in history.columns:
        return sorted(set(history["Platform"].dropna().unique()))
    return []


def sidebar_filters(
    history: pd.DataFrame,
    key_prefix: str = "",
    *,
    time_window_key: str | None = None,
) -> dict:
    """Render sidebar filters and return filter values.

    When ``time_window_key`` is set, a Time-window selectbox renders under the same
    Filters header and the returned dict's ``cutoff`` is the date that window starts at
    (``None`` for "All time" or when no window is requested). This folds the two Lab
    pages' formerly page-local time window into the one shared filter section.
    """
    if st.sidebar.button(
        "Refresh data",
        key=f"{key_prefix}refresh_data",
        help="Force-reload all parquet snapshots, bypassing the data cache.",
    ):
        st.cache_data.clear()
        st.rerun()
    st.sidebar.header("Filters")

    cutoff = None
    if time_window_key is not None:
        time_window = st.sidebar.selectbox(
            "Time window", list(TIMEFRAME_OPTIONS.keys()), index=0, key=time_window_key
        )
        window_days = TIMEFRAME_OPTIONS[time_window]
        if window_days is not None:
            cutoff = datetime.today().date() - timedelta(days=window_days)

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

    return {
        "date_range": date_range,
        "leagues": selected_leagues,
        "platforms": selected_platforms,
        "cutoff": cutoff,
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
