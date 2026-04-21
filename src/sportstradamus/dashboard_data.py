"""Shared data loading and state for the Sportstradamus dashboard.

All pages import from here to get cached DataFrames and filters.
"""

import importlib.resources as pkg_resources
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from sportstradamus import data
from sportstradamus.analysis import (
    _migrate_flat_history,
    check_bet,
    explode_offers,
    resolve_history,
)
from sportstradamus.stats import StatsMLB, StatsNBA, StatsNFL, StatsNHL, StatsWNBA

LEAGUE_CLASSES = {
    "NBA": StatsNBA,
    "WNBA": StatsWNBA,
    "MLB": StatsMLB,
    "NFL": StatsNFL,
    "NHL": StatsNHL,
}


@st.cache_data(ttl=3600, show_spinner="Loading prediction history...")
def load_history():
    """Load prediction history in normalized schema.

    If old flat schema is detected (no Offers column), migrates automatically
    and saves the migrated data back to pickle.
    """
    filepath = pkg_resources.files(data) / "history.dat"
    if not os.path.isfile(filepath):
        return pd.DataFrame()
    history = pd.read_pickle(filepath)

    # Migrate old flat schema → normalized (one row per prediction, Offers list)
    if "Offers" not in history.columns:
        history = _migrate_flat_history(history)
        history.to_pickle(filepath)

    # Ensure prediction-level columns exist for backward compatibility
    for col in ["Dist", "CV", "Model Param", "Gate", "Temperature", "Disp Cal", "Step", "Actual"]:
        if col not in history.columns:
            history[col] = np.nan

    return history


@st.cache_data(ttl=3600, show_spinner="Loading parlay history...")
def load_parlays():
    """Load parlay history."""
    filepath = pkg_resources.files(data) / "parlay_hist.dat"
    if not os.path.isfile(filepath):
        return pd.DataFrame()
    parlays = pd.read_pickle(filepath)

    # Backward compat
    for col in ["Corr Pairs", "Boost Pairs", "Indep P", "Indep PB"]:
        if col not in parlays.columns:
            parlays[col] = np.nan

    return parlays


@st.cache_resource(show_spinner="Loading league stats...")
def load_stats():
    """Load Stats objects from cached pickle files (no API calls).

    API updates are handled by the nightly script (poetry run nightly),
    not at dashboard startup. This keeps the dashboard load fast (~2 s).
    """
    stats = {}
    for lg, cls in LEAGUE_CLASSES.items():
        try:
            obj = cls()
            obj.load()
            if hasattr(obj, "gamelog") and not obj.gamelog.empty:
                stats[lg] = obj
        except Exception:
            pass
    return stats


@st.cache_data(ttl=3600, show_spinner="Loading stat map...")
def load_stat_map():
    with open(pkg_resources.files(data) / "stat_map.json") as f:
        return json.load(f)


def load_resolve_meta():
    """Load nightly resolution metadata (last run time, counts).

    Returns a dict with keys: last_run, history_resolved, parlays_resolved.
    Returns empty dict if file doesn't exist (nightly hasn't run yet).
    """
    meta_path = pkg_resources.files(data) / "resolve_meta.json"
    try:
        with open(meta_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def resolve_and_save(history, stats):
    """Resolve pending predictions (fill Actual) and save back to pickle."""
    if "Actual" not in history.columns:
        history["Actual"] = np.nan

    pending_count = history["Actual"].isna().sum()
    if pending_count == 0:
        return history

    history = resolve_history(history, stats)

    filepath = pkg_resources.files(data) / "history.dat"
    history.to_pickle(filepath)

    return history


def resolve_parlays_and_save(parlays, stats, stat_map):
    """Resolve pending parlays and save back to pickle."""
    if parlays.empty or "Legs" not in parlays.columns:
        return parlays

    unresolved = parlays.loc[parlays["Legs"].isna()]
    if len(unresolved) == 0:
        return parlays

    from tqdm import tqdm

    tqdm.pandas()
    results = unresolved.progress_apply(
        lambda bet: check_bet(bet, stats, stat_map), axis=1
    ).to_list()
    parlays.loc[parlays["Legs"].isna(), ["Legs", "Misses"]] = results

    filepath = pkg_resources.files(data) / "parlay_hist.dat"
    parlays.to_pickle(filepath)

    return parlays


def get_filtered_history(
    history, leagues=None, platforms=None, markets=None, date_range=None, min_model_p=None
):
    """Explode offers and apply sidebar filters.

    Returns a per-offer DataFrame with columns: all prediction-level cols +
    Line, Boost, Platform, Bet, Model P, Books P, Result, Hit, Model, Books, K.
    """
    # Explode normalized schema into one row per offer
    df = explode_offers(history)
    if df.empty:
        return df

    # Drop pushes and unresolved
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
        prob_col = "Model P" if "Model P" in df.columns and df["Model P"].notna().any() else "Model"
        df = df.loc[df[prob_col] >= min_model_p]

    return df


def get_prediction_history(history, leagues=None, date_range=None):
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
    """Extract unique platform names from Offers column."""
    platforms = set()
    if "Offers" in history.columns:
        for offers in history["Offers"].dropna():
            if isinstance(offers, list):
                for offer in offers:
                    if len(offer) >= 3 and offer[2]:
                        platforms.add(str(offer[2]))
    elif "Platform" in history.columns:
        platforms = set(history["Platform"].dropna().unique())
    return sorted(platforms)


def sidebar_filters(history, parlays=None, key_prefix=""):
    """Render sidebar filters and return filter values."""
    st.sidebar.header("Filters")

    # Date range
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

    # League filter
    leagues = sorted(history["League"].dropna().unique()) if not history.empty else []
    selected_leagues = st.sidebar.multiselect(
        "Leagues", leagues, default=leagues, key=f"{key_prefix}leagues"
    )

    # Platform filter (extracted from Offers in normalized schema)
    platforms = _extract_platforms(history)
    selected_platforms = st.sidebar.multiselect(
        "Platforms", platforms, default=platforms, key=f"{key_prefix}platforms"
    )

    # Data coverage indicator
    if not history.empty and "Dist" in history.columns:
        coverage = history["Dist"].notna().mean()
        st.sidebar.metric("Distribution Data Coverage", f"{coverage:.0%}")

    return {
        "date_range": date_range,
        "leagues": selected_leagues,
        "platforms": selected_platforms,
    }
