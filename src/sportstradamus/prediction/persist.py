"""Snapshot writers for the prediction pipeline.

`prophecize` writes the current run's scored offers and parlay candidates as
parquet snapshots that the Streamlit dashboard reads. Files land via atomic
rename so the dashboard never sees a torn read mid-run.
"""

from __future__ import annotations

import datetime
from collections.abc import Iterable

import numpy as np
import pandas as pd

from sportstradamus.helpers.io import (
    CURRENT_META_PATH,
    CURRENT_OFFERS_PATH,
    CURRENT_PARLAYS_PATH,
    _atomic_write_json,
    _atomic_write_parquet,
)

# Display columns kept in current_offers.parquet. The dashboard reads:
# - Offer details: League, Date, Team, Opponent, Player, Market, Platform, Bet, Line, Boost
# - Scoring: Model P (hit probability), Model (edge), Books, Model EV, Model STD, Push P
# - Context: Avg 5, Avg H2H, Moneyline, O/U, DVPOA
# - Correlations: Team Correlation, Opp Correlation
# - Stat key: Stat (for history lookups)
# - Distribution (for PDF/PMF): Dist, CV, Gate, and shape parameters (Model R, Alpha, Sigma, Skew)
#
# Dropped internal columns: Model Param (generic workaround), Temperature, Disp Cal, Step,
# Books P, K, Player position (not needed for rendering).
_OFFER_KEEP_COLS = [
    "League",
    "Date",
    "Team",
    "Opponent",
    "Player",
    "Market",
    "Platform",
    "Bet",
    "Line",
    "Boost",
    "Model P",
    "Model",
    "Books",
    "Model EV",
    "Model STD",
    "Push P",
    "Avg 5",
    "Avg H2H",
    "Moneyline",
    "O/U",
    "DVPOA",
    "Team Correlation",
    "Opp Correlation",
    "Stat",
    "Dist",
    "CV",
    "Gate",
    "Model R",
    "Model Alpha",
    "Model Sigma",
    "Model Skew",
]

# A row with no boost, no model edge, and no book edge carries no signal —
# it is a scoring artifact and is dropped before the snapshot is written.
_OFFER_SIGNAL_COLS = ["Boost", "Model", "Books"]

# Internal correlation cols dropped from current_parlays.parquet — these are
# scoring artifacts not useful for human review. `Indep P` is kept (independence
# joint probability) so users can compare against the correlation-aware joint.
_PARLAY_DROP_COLS = ["Leg Probs", "Corr Pairs", "Boost Pairs", "Indep PB"]


def write_current_offers(
    offers: pd.DataFrame,
    parlays: pd.DataFrame,
    leagues: Iterable[str],
    platforms: Iterable[str],
    contest_variant: str = "pooled",
    stats_dict: dict | None = None,
) -> None:
    """Write the current-run snapshot atomically.

    `offers` should be the concatenated post-`process_offers` per-platform
    DataFrames (with a `Platform` column already attached). `parlays` is the
    deduped post-loop parlay_df. `contest_variant` is the Underdog payout
    variant the parlays were scored under, recorded in meta so the dashboard
    can display which payout schedule the EV column reflects. Empty inputs
    are still written so the dashboard reflects the most recent run.

    The `stats_dict` parameter is unused (retained for API compatibility).
    The dashboard loads player history directly from cached gamelog parquets.
    """
    offers_out = _normalize_offers(offers)
    parlays_out = _normalize_parlays(parlays)

    _atomic_write_parquet(offers_out, CURRENT_OFFERS_PATH)
    _atomic_write_parquet(parlays_out, CURRENT_PARLAYS_PATH)
    _atomic_write_json(
        {
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "leagues": sorted(set(leagues)),
            "platforms": sorted(set(platforms)),
            "contest_variant": contest_variant,
            "offer_rows": len(offers_out),
            "parlay_rows": len(parlays_out),
        },
        CURRENT_META_PATH,
    )


def _normalize_offers(offers: pd.DataFrame) -> pd.DataFrame:
    if offers is None or offers.empty:
        return pd.DataFrame(columns=_OFFER_KEEP_COLS)
    df = offers.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop rows with no signal (edge or boost).
    signal_cols = [c for c in _OFFER_SIGNAL_COLS if c in df.columns]
    if signal_cols:
        signal = df[signal_cols].fillna(0)
        df = df.loc[(signal != 0).any(axis=1)]

    # Reshape distribution parameters for the dashboard. The scoring pipeline
    # returns "Model Param" (the shape value) but the dashboard needs it split
    # into distribution-specific columns (Model R, Model Alpha, etc.) for scipy.
    if "Model Param" in df.columns and "Dist" in df.columns:
        df["Model R"] = np.where(df["Dist"].isin(("NegBin", "ZINB")), df["Model Param"], np.nan)
        df["Model Alpha"] = np.where(df["Dist"].isin(("Gamma", "ZAGamma")), df["Model Param"], np.nan)
        df["Model Sigma"] = np.where(df["Dist"] == "SkewNormal", df["Model Param"], np.nan)

    # Ensure Gate is present (zero-inflation probability for ZINB, ZAGamma).
    if "Gate" not in df.columns:
        df["Gate"] = np.nan
    if "Model Skew" not in df.columns:
        df["Model Skew"] = np.nan

    # Keep only columns the dashboard needs; drop internal scoring artifacts.
    keep = [c for c in _OFFER_KEEP_COLS if c in df.columns]
    df = df[keep]
    if "Model" in df.columns:
        df = df.sort_values("Model", ascending=False, ignore_index=True)
    return df


def _normalize_parlays(parlays: pd.DataFrame) -> pd.DataFrame:
    if parlays is None or parlays.empty:
        return pd.DataFrame()
    df = parlays.drop(columns=[c for c in _PARLAY_DROP_COLS if c in parlays.columns])
    df = df.replace([np.inf, -np.inf], np.nan)
    if "Model EV" in df.columns:
        df = df.sort_values("Model EV", ascending=False, ignore_index=True)
    return df
