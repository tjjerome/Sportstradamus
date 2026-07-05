"""Narrative display values and cross-surface handoffs from the prophecize snapshots.

Pure lookups the Tonight, Games, and deep-dive surfaces use to turn the precomputed
thesis and game-context artifacts into per-game display text — no Streamlit, no I/O, so
they unit-test directly. Most key on the canonical ``Game`` slash key (``"NYK/SAS"``)
plus ``Date`` so a doubleheader's two games stay distinct.
"""

from __future__ import annotations

import pandas as pd

from sportstradamus.dashboard import theme

# Game-shape gloss, shared by the Games banner and the deep-dive context strip.
SHAPE_HELP = (
    "Projected game script: shootout (high total), grind (low total), blowout "
    "(lopsided), or coinflip (tight). It tilts which counting stats run hot."
)

# Colorblind-safe side cues (spec §3.2): shape (up/down triangle) plus color both
# carry Over/Under, so the aggrid JsCode cell renderer in Phase B can key off shape
# alone if color is unavailable. Sourced from theme.GREEN/RED, not re-hardcoded.
_ARROW_UP = (
    f'<svg class="ar" viewBox="0 0 9 10" width="9" height="10">'
    f'<path d="M4.5 0 9 10 0 10Z" fill="{theme.GREEN}"/></svg>'
)
_ARROW_DOWN = (
    f'<svg class="ar" viewBox="0 0 9 10" width="9" height="10">'
    f'<path d="M4.5 10 9 0 0 0Z" fill="{theme.RED}"/></svg>'
)


def bet_arrow(bet: str) -> str:
    """Colorblind-safe side cue: shape + color both carry Over/Under."""
    return _ARROW_UP if bet == "Over" else _ARROW_DOWN


def match_label(team: str, opp: str, home: bool) -> str:
    """Player-team-first matchup: 'LVA @ IND' away, 'LVA v IND' home (spec §3.4)."""
    return f"{team} {'v' if home else '@'} {opp}"


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


def home_away(group: pd.DataFrame) -> tuple[str, str]:
    """``(home, away)`` team codes for one game's offer rows.

    The home side is the one whose rows carry ``Home == True`` (a model feature);
    without that column — snapshots written before it was surfaced — the canonical
    ``Game`` slash key (alphabetical) orders the matchup so it still renders
    deterministically. Dedups the two per-team orderings of one game to a single label.
    """
    teams = str(group["Game"].iloc[0]).split("/")
    if "Home" in group.columns:
        hosts = group.loc[group["Home"].astype(object).eq(True), "Team"]
        if not hosts.empty:
            home = str(hosts.iloc[0])
            away = next((t for t in teams if t != home), "")
            if away:
                return home, away
    return (teams[0], teams[1]) if len(teams) == 2 else (teams[0] if teams else "", "")


def context_strip(ctx_df: pd.DataFrame, *, game: str, date) -> dict | None:
    """Total / derived spread / favorite / shape / favorite win prob for one game.

    ``None`` if the game is absent. ``ctx_df`` is ``current_game_context`` (keyed
    ``League, Game, Date``). The caller formats — ``fav_team`` may be ``None`` and
    ``ml_fav_prob`` NaN on a game with no moneyline (an even game has spread 0).
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
        "baseline_total": float(row["baseline_total"]),
        "ml_fav_prob": float(row["ml_fav_prob"]),
    }
