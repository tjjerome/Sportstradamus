"""Narrative display values derived from the prophecize snapshots.

Pure lookups the Tonight and Game surfaces use to turn the precomputed thesis
and game-context artifacts into per-game display text — no Streamlit, no I/O, so
they unit-test directly. Both key on the canonical ``Game`` slash key
(``"NYK/SAS"``) plus ``Date`` so a doubleheader's two games stay distinct.
"""

from __future__ import annotations

import pandas as pd


def top_thesis(parlays: pd.DataFrame, *, game: str, date) -> str:
    """The headline of the highest-EV parlay for one game (``""`` when none).

    ``parlays`` is ``current_parlays``; the top per-leg-set thesis is the
    ``Thesis`` of the max-``Model EV`` row in ``(Game, Date)``.
    """
    if parlays.empty or "Thesis" not in parlays.columns or "Model EV" not in parlays.columns:
        return ""
    sub = parlays.loc[
        (parlays["Game"] == game) & (parlays["Date"].astype(str) == str(date))
    ].dropna(subset=["Model EV"])
    if sub.empty:
        return ""
    thesis = sub.loc[sub["Model EV"].idxmax(), "Thesis"]
    return str(thesis) if pd.notna(thesis) else ""


def context_strip(ctx_df: pd.DataFrame, *, game: str, date) -> dict | None:
    """Total / derived spread / favorite / shape for one game, or ``None`` if absent.

    ``ctx_df`` is ``current_game_context`` (keyed ``League, Game, Date``). The
    caller formats — ``fav_team`` may be ``None`` on an even game (spread 0).
    """
    if ctx_df.empty:
        return None
    sub = ctx_df.loc[(ctx_df["Game"] == game) & (ctx_df["Date"].astype(str) == str(date))]
    if sub.empty:
        return None
    row = sub.iloc[0]
    return {
        "game_total": float(row["game_total"]),
        "spread": float(row["spread"]),
        "fav_team": row["fav_team"],
        "shape": row["shape"],
    }
