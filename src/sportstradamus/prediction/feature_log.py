"""Serve-time feature log: what each model saw for every player it scored.

``model_prob`` upserts one row per scored player and game, so train/serve parity can
be checked against the training matrix and a leg can be re-scored offline under
another model file. Diagnostics only: the dashboard never reads it, and no row reaches
``history.parquet`` or a ``current_*`` snapshot.

Layout is ``FEATURE_LOG_DIR/date=<game date>/<LEAGUE>_<market>.parquet``, keyed by
:data:`FEATURE_LOG_KEY`. A rescoring replaces the row, so the latest scoring wins as in
``history.parquet``; it is not always ``prophecize``'s, because the live loads behind
``ledger-commit`` and ``bet pickem`` score through ``model_prob`` too. ``Market`` is the
model cell, which names the pickle. History carries the same code except on NBA and
WNBA, whose ``... underdog`` markets are served by the ``... prizepicks`` cell.

Beside the key, a row carries what ``model_prob`` prices a leg from, apart from the
pickle, the committed ``stat_meta`` book shapes and the offer's line:

* ``Scored At`` (naive UTC) and the serving constants ``Model Version``, ``Step``,
  ``Model Weight`` and ``Hist Gate``. The last is the runtime zero-rate gate, which no
  pickle records.
* The frame fed to the model, column for column. Parquet keeps a categorical as its
  category values, which is what LightGBM matches on.
* The decision-time book leg as the blend consumes it: ``Market Projection``,
  ``Books STD`` and the ``Quote *`` provenance, plus ``Quote Over Prob`` and
  ``Quote EV`` so the leg can be re-inverted under another pickle's shape.
* The model's per-player outputs before the book blend or any offer line: the raw
  distribution parameters and their decoded ``Projection`` / ``Model *`` columns.
* ``Opponent Pitcher`` (MLB only): the probable starter the features were built on.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from sportstradamus.helpers.io import (
    FEATURE_LOG_DIR,
    _atomic_write_parquet,
    market_file_slug,
    read_parquet_safe,
)

if TYPE_CHECKING:
    from sportstradamus.helpers.training_quotes import TrainingQuote
    from sportstradamus.stats.base import Stats

FEATURE_LOG_KEY = ["League", "Market", "Platform", "Player", "Date"]
# One weekly retrain cycle plus the 14 days prior model files are kept, with slack.
FEATURE_LOG_RETENTION_DAYS = 30


def upsert_feature_log(
    league: str,
    market: str,
    platform: str,
    offers: list[dict],
    stat_data: Stats,
    player_stats: pd.DataFrame,
    prob_params: pd.DataFrame,
    quotes: Sequence[TrainingQuote | None],
    *,
    model_version: str,
    step: float,
    model_weight: float,
    hist_gate: float,
) -> None:
    """Upsert one scored market's rows into their game-date partitions.

    A player with no dated offer is not logged: a combo leg's component is fed to the
    model but served nowhere, so it has no game to file under.

    Args:
        league: League key.
        market: Model cell, already through ``normalize_market``.
        platform: DFS platform the offers came from.
        offers: The market's raw offers. They give each player's game date (the last
            offer wins, as in ``model_prob``) and, for MLB, team.
        stat_data: The league's ``Stats``, read only for MLB's probable pitchers.
        player_stats: Player-indexed frame fed to the model, with the book-leg columns
            ``model_prob`` adds.
        prob_params: Player-indexed model outputs, raw and decoded.
        quotes: The quote behind each ``player_stats`` row's book leg, ``None`` where
            the blend runs model-only.
        model_version: Identity of the serving pickle.
        step: The pickle's line step.
        model_weight: The pickle's model share of the book blend.
        hist_gate: Zero-rate gate the serve read from ``stat_zi``.
    """
    game_dates = {offer["Player"]: offer["Date"] for offer in offers}
    rows = pd.DataFrame(
        {
            "League": league,
            "Market": market,
            "Platform": platform,
            "Player": player_stats.index,
            "Date": player_stats.index.map(game_dates),
            "Scored At": pd.Timestamp.now("UTC").tz_localize(None),
            "Model Version": model_version,
            "Step": step,
            "Model Weight": model_weight,
            "Hist Gate": hist_gate,
            "Quote Over Prob": [quote.over_probability if quote else np.nan for quote in quotes],
            "Quote EV": [quote.ev if quote else np.nan for quote in quotes],
        },
        index=player_stats.index,
    )
    if league == "MLB":
        teams = {offer["Player"]: offer["Team"] for offer in offers}
        rows["Opponent Pitcher"] = [
            stat_data.upcoming_games.get(teams.get(player), {}).get("Opponent Pitcher")
            for player in rows.index
        ]
    rows = rows.join(player_stats).join(prob_params).dropna(subset=["Date"])

    slug = market_file_slug(league, market)
    for game_date, day in rows.groupby("Date"):
        path = Path(str(FEATURE_LOG_DIR)) / f"date={game_date}" / f"{slug}.parquet"
        logged = read_parquet_safe(path)
        merged = day if logged.empty else pd.concat([logged, day], ignore_index=True)
        _atomic_write_parquet(
            merged.drop_duplicates(FEATURE_LOG_KEY, keep="last"), path, compression="zstd"
        )


def prune_feature_log() -> None:
    """Delete the game-date partitions older than ``FEATURE_LOG_RETENTION_DAYS``."""
    cutoff = pd.Timestamp.today().date() - pd.Timedelta(days=FEATURE_LOG_RETENTION_DAYS)
    # ISO dates order as text, so a partition name is compared without being parsed.
    for partition in Path(str(FEATURE_LOG_DIR)).glob("date=*"):
        if partition.name < f"date={cutoff}":
            shutil.rmtree(partition)
