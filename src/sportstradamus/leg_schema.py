"""Canonical structured-leg schema shared by prediction, grading, and dashboard.

One leg = one dict with LEG_FIELDS keys; stored in parquet as list<struct>
columns (pandas cells of list[dict]). Replaces the retired Desc string
round-trip — display strings are rendered on demand via leg_label, never stored.
"""

from collections.abc import Mapping

import pandas as pd

LEG_FIELDS = (
    "player",
    "team",
    "market",
    "stat",
    "bet",
    "line",
    "league",
    "game",
    "date",
    "platform",
    "win_prob",
    "boost",
    "push_prob",
    "kelly",
)

_OFFER_KEYS = {
    "player": "Player",
    "team": "Team",
    "market": "Market",
    "bet": "Bet",
    "league": "League",
    "game": "Game",
    "platform": "Platform",
}


def _numeric_or_default(row: Mapping, col: str, default: float) -> float:
    """Read a numeric offer column, treating a missing key or NaN as default."""
    value = row.get(col, default)
    return default if pd.isna(value) else float(value)


def build_leg(row: Mapping) -> dict:
    """Snapshot an offer row (dict or Series) into one canonical leg record."""
    leg = {field: str(row[col]) for field, col in _OFFER_KEYS.items()}
    stat = row.get("Stat")
    leg["stat"] = str(stat) if pd.notna(stat) and stat else str(row["Market"])
    leg["date"] = str(row["Date"])[:10]
    leg["line"] = float(row["Line"])
    leg["win_prob"] = float(row["Win Prob"])
    leg["boost"] = _numeric_or_default(row, "Boost", 1.0)
    leg["push_prob"] = _numeric_or_default(row, "Push Prob", 0.0)
    leg["kelly"] = _numeric_or_default(row, "Kelly", 0.0)
    return leg


def leg_label(leg: Mapping) -> str:
    """Human display string for a leg. Render-only — never parsed back."""
    return f"{leg['player']} {leg['bet']} {leg['line']:.10g} {leg['market']}"
