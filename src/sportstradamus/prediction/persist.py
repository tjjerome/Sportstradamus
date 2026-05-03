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

# Display columns kept in current_offers.parquet. Internal model-tuning cols
# (`Model Param`, `Dist`, `CV`, `Gate`, `Temperature`, `Disp Cal`, `Step`,
# `Model P`, `Books P`, `K`, `Player position`) are dropped — the dashboard
# is a renderer, not a re-scorer.
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
    "Model EV",
    "Model",
    "Books",
    "Avg 5",
    "Avg H2H",
    "Moneyline",
    "O/U",
    "DVPOA",
]

# Internal correlation cols dropped from current_parlays.parquet — these are
# scoring artifacts not useful for human review (matches the old Sheets path).
_PARLAY_DROP_COLS = ["Leg Probs", "Corr Pairs", "Boost Pairs", "Indep P", "Indep PB"]


def write_current_offers(
    offers: pd.DataFrame,
    parlays: pd.DataFrame,
    leagues: Iterable[str],
    platforms: Iterable[str],
) -> None:
    """Write the current-run snapshot atomically.

    `offers` should be the concatenated post-`process_offers` per-platform
    DataFrames (with a `Platform` column already attached). `parlays` is the
    deduped post-loop parlay_df. Empty inputs are still written so the
    dashboard reflects the most recent run.
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
            "offer_rows": int(len(offers_out)),
            "parlay_rows": int(len(parlays_out)),
        },
        CURRENT_META_PATH,
    )


def _normalize_offers(offers: pd.DataFrame) -> pd.DataFrame:
    if offers is None or offers.empty:
        return pd.DataFrame(columns=_OFFER_KEEP_COLS)
    df = offers.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    keep = [c for c in _OFFER_KEEP_COLS if c in df.columns]
    df = df[keep]
    if "Model EV" in df.columns:
        df = df.sort_values("Model EV", ascending=False, ignore_index=True)
    return df


def _normalize_parlays(parlays: pd.DataFrame) -> pd.DataFrame:
    if parlays is None or parlays.empty:
        return pd.DataFrame()
    df = parlays.drop(columns=[c for c in _PARLAY_DROP_COLS if c in parlays.columns])
    df = df.replace([np.inf, -np.inf], np.nan)
    if "Model EV" in df.columns:
        df = df.sort_values("Model EV", ascending=False, ignore_index=True)
    return df
