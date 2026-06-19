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
    CURRENT_GAME_CONTEXT_PATH,
    CURRENT_GAME_CORR_PATH,
    CURRENT_GAME_STORIES_PATH,
    CURRENT_META_PATH,
    CURRENT_OFFER_DETAILS_PATH,
    CURRENT_OFFERS_PATH,
    CURRENT_PARLAYS_PATH,
    CURRENT_PICKEM_PATH,
    _atomic_write_json,
    _atomic_write_parquet,
)
from sportstradamus.prediction.stories import STORIES_VERSION

# Display columns kept in current_offers.parquet. The dashboard reads:
# - Offer details: League, Date, Team, Opponent, Home (is Team the host — orders the
#   "{home} vs {away}" matchup label), Game (canonical matchup key),
#   Player, Market, Platform, Bet, Line, Consensus Line (weighted-avg book line from
#   archive.get_line — the Model-tab consensus marker, distinct from the DFS app Line), Boost
# - Scoring: Win Prob (hit probability), Model EV (edge), Market EV, Projection (stat mean),
#   Kelly, Projection STD, Push Prob
# - Context: Avg 5, Avg H2H, Moneyline, O/U, DVPOA, Position (depth-chart label)
# - Correlations: Team Correlation, Opp Correlation
# - Narrative: Why (precomputed "the case" string, prediction/stories)
# - Stat key: Stat (for history lookups)
# - Distribution (for PDF/PMF): Dist, CV, Gate, and shape parameters (Model R, Alpha, Sigma, Skew)
#
# Dropped internal columns: Model Param (generic workaround), Temperature, Disp Cal, Step,
# Market Prob, Player position (numeric depth slot — the resolved label rides in Position).
_OFFER_KEEP_COLS = [
    "League",
    "Date",
    "Team",
    "Opponent",
    "Home",
    "Game",
    "Player",
    "Market",
    "Platform",
    "Bet",
    "Line",
    "Consensus Line",
    "Boost",
    "Win Prob",
    "Model EV",
    "Market EV",
    "Projection",
    "Kelly",
    "Projection STD",
    "Push Prob",
    "Avg 5",
    "Avg H2H",
    "Moneyline",
    "O/U",
    "DVPOA",
    "Position",
    "Team Correlation",
    "Opp Correlation",
    "Why",
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
_OFFER_SIGNAL_COLS = ["Boost", "Model EV", "Market EV"]

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
) -> None:
    """Write the current-run snapshot atomically.

    `offers` should be the concatenated post-`process_offers` per-platform
    DataFrames (with a `Platform` column already attached). `parlays` is the
    deduped post-loop parlay_df. `contest_variant` is the Underdog payout
    variant the parlays were scored under, recorded in meta so the dashboard
    can display which payout schedule the EV column reflects. Empty inputs
    are still written so the dashboard reflects the most recent run.
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
            "stories_version": STORIES_VERSION,
        },
        CURRENT_META_PATH,
    )


def write_current_game_corr(corr_rows: list[dict] | pd.DataFrame) -> None:
    """Write the per-game correlation slices snapshot atomically.

    ``corr_rows`` is the collector list filled by ``find_correlation`` (one row
    per game leg pair: ``League, Game, leg_a, leg_b, rho``). Deduped on the
    matchup + leg-pair key — the same pair recurs across platforms and across a
    player's multiple lines, and correlation is line-independent. Written even
    when empty so the dashboard reflects the latest run.
    """
    df = pd.DataFrame(corr_rows, columns=["League", "Game", "leg_a", "leg_b", "rho"])
    if not df.empty:
        df = df.drop_duplicates(subset=["League", "Game", "leg_a", "leg_b"], ignore_index=True)
    _atomic_write_parquet(df, CURRENT_GAME_CORR_PATH)


def write_current_game_context(context: pd.DataFrame) -> None:
    """Write the per-game context snapshot atomically.

    ``context`` is the prebuilt ``stories.build_game_context`` frame (one row per
    ``(League, Game, Date)`` — game shape, derived spread, favorite, positional
    DVPOA edges). The caller builds it once and feeds the same frame to the
    thesis pass, so headline and artifact share one view. Written even when empty
    so the dashboard reflects the latest run.
    """
    _atomic_write_parquet(context, CURRENT_GAME_CONTEXT_PATH)


def write_current_game_stories(stories: pd.DataFrame) -> None:
    """Write the per-game story-menu snapshot atomically.

    ``stories`` is the ``stories.build_game_stories`` frame: up to five stories
    per ``(platform, game)``, each with a Bankroll Builder and a Shoot-the-Moon
    parlay (one row per ``objective``). ``legs`` is a JSON list of leg-desc
    strings (round-trips through ``parse_leg`` for the dashboard's live thesis
    regen); ``kelly_stake`` is a standalone full-Kelly bankroll *fraction* — the
    dashboard rail re-sizes it into Decimal dollars against the live bankroll.
    ``build_game_stories`` returns a column-stable frame even with no stories, so
    an empty slate still writes a header-only snapshot the dashboard can read.
    """
    _atomic_write_parquet(stories, CURRENT_GAME_STORIES_PATH)


def write_current_offer_details(details: pd.DataFrame) -> None:
    """Write the per-offer deep-dive detail prerender atomically.

    ``details`` is the ``stories.build_offer_details`` frame: one row per
    ``(League, Date, Player, Market, Opponent)`` with ``comps_vs_opp`` /
    ``volume_trend`` / ``other_stats`` JSON payloads the dashboard renders without
    recomputing. Column-stable even on an empty slate, so a header-only snapshot
    still writes for the dashboard to read.
    """
    _atomic_write_parquet(details, CURRENT_OFFER_DETAILS_PATH)


def write_current_pickem(entries: pd.DataFrame) -> None:
    """Write the current-run Underdog Pick'em entries snapshot atomically.

    `entries` is the frame from `strategies._pickem_emit.entries_to_frame`
    (bankroll-independent fields; the dashboard sizes stakes from a user
    bankroll). Written even when empty so the dashboard reflects the latest run.
    """
    _atomic_write_parquet(entries, CURRENT_PICKEM_PATH)


def _normalize_offers(offers: pd.DataFrame) -> pd.DataFrame:
    if offers is None or offers.empty:
        return pd.DataFrame(columns=_OFFER_KEEP_COLS)
    df = offers.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    signal_cols = [c for c in _OFFER_SIGNAL_COLS if c in df.columns]
    if signal_cols:
        signal = df[signal_cols].fillna(0)
        df = df.loc[(signal != 0).any(axis=1)]

    df = _split_dist_params(df)

    # Ensure Gate is present (zero-inflation probability for ZINB, ZAGamma).
    if "Gate" not in df.columns:
        df["Gate"] = np.nan
    if "Model Skew" not in df.columns:
        df["Model Skew"] = np.nan

    keep = [c for c in _OFFER_KEEP_COLS if c in df.columns]
    df = df[keep]
    if "Model EV" in df.columns:
        df = df.sort_values("Model EV", ascending=False, ignore_index=True)
    return df


def _split_dist_params(df: pd.DataFrame) -> pd.DataFrame:
    """Split the generic "Model Param" shape value into distribution-specific
    columns (Model R / Model Alpha / Model Sigma) the dashboard's scipy calls
    expect. No-op when the scoring pipeline didn't attach Model Param / Dist.
    """
    if "Model Param" not in df.columns or "Dist" not in df.columns:
        return df
    df["Model R"] = np.where(df["Dist"].isin(("NegBin", "ZINB")), df["Model Param"], np.nan)
    df["Model Alpha"] = np.where(df["Dist"].isin(("Gamma", "ZAGamma")), df["Model Param"], np.nan)
    df["Model Sigma"] = np.where(df["Dist"] == "SkewNormal", df["Model Param"], np.nan)
    return df


def _normalize_parlays(parlays: pd.DataFrame) -> pd.DataFrame:
    if parlays is None or parlays.empty:
        return pd.DataFrame()
    df = parlays.drop(columns=[c for c in _PARLAY_DROP_COLS if c in parlays.columns])
    df = df.replace([np.inf, -np.inf], np.nan)
    if "Model EV" in df.columns:
        df = df.sort_values("Model EV", ascending=False, ignore_index=True)
    return df
