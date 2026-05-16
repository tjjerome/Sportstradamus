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
    CURRENT_HISTORY_PATH,
    CURRENT_META_PATH,
    CURRENT_OFFERS_PATH,
    CURRENT_PARLAYS_PATH,
    _atomic_write_json,
    _atomic_write_parquet,
)

# Display columns kept in current_offers.parquet. Internal model-tuning cols
# (`Model Param`, `Dist`, `CV`, `Gate`, `Temperature`, `Disp Cal`, `Step`,
# `Books P`, `K`, `Player position`) are dropped — the dashboard is a
# renderer, not a re-scorer. `Model P` (hit probability), `Model STD`, and
# the correlation strings are kept so the dashboard can show them in the
# main table and the per-row detail popup.
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


def write_player_history(offers_df: pd.DataFrame, stats_dict: dict) -> None:
    """Write per-game player stat history for offers in the current run.

    For each unique (Player, League, Stat) in offers_df, extract all game-level
    stat values from the Stats object's short_gamelog (full season window).
    Writes a long-format parquet with columns: Player, League, Market, GameDate,
    Opponent, StatValue, GameNum (1=oldest).

    Silent guards: skips players/stats that don't exist in stats_dict or gamelog.
    """
    history_rows = []
    for (player, league, stat), group in offers_df.groupby(
        ["Player", "League", "Stat"], dropna=False
    ):
        if not isinstance(stat, str) or pd.isna(stat):
            continue
        stat_obj = stats_dict.get(league)
        if stat_obj is None or not hasattr(stat_obj, "short_gamelog"):
            continue

        player_col = stat_obj.log_strings.get("player")
        date_col = stat_obj.log_strings.get("date")
        opp_col = stat_obj.log_strings.get("opponent")
        if not all([player_col, date_col, opp_col, stat in stat_obj.short_gamelog.columns]):
            continue

        player_games = stat_obj.short_gamelog[stat_obj.short_gamelog[player_col] == player]
        if player_games.empty:
            continue

        for game_num, (_, row) in enumerate(player_games.iterrows(), 1):
            history_rows.append(
                {
                    "Player": player,
                    "League": league,
                    "Market": stat,
                    "GameDate": row[date_col],
                    "Opponent": row[opp_col],
                    "StatValue": row[stat],
                    "GameNum": game_num,
                }
            )

    if history_rows:
        history_df = pd.DataFrame(history_rows)
        _atomic_write_parquet(history_df, CURRENT_HISTORY_PATH)
    else:
        _atomic_write_parquet(
            pd.DataFrame(
                columns=[
                    "Player",
                    "League",
                    "Market",
                    "GameDate",
                    "Opponent",
                    "StatValue",
                    "GameNum",
                ]
            ),
            CURRENT_HISTORY_PATH,
        )


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

    If `stats_dict` is provided, also writes current_history.parquet with
    per-game stat values for each player+stat combo in the offers.
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

    if stats_dict:
        write_player_history(offers_out, stats_dict)


def _normalize_offers(offers: pd.DataFrame) -> pd.DataFrame:
    if offers is None or offers.empty:
        return pd.DataFrame(columns=_OFFER_KEEP_COLS)
    df = offers.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    signal_cols = [c for c in _OFFER_SIGNAL_COLS if c in df.columns]
    if signal_cols:
        signal = df[signal_cols].fillna(0)
        df = df.loc[(signal != 0).any(axis=1)]
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
