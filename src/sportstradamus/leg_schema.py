"""Canonical structured-leg schema shared by prediction, grading, and dashboard.

One leg = one dict with LEG_FIELDS keys; stored in parquet as list<struct>
columns (pandas cells of list[dict]). Replaces the retired Desc string
round-trip — display strings are rendered on demand via leg_label, never stored.
"""

from collections.abc import Mapping

import pandas as pd

from sportstradamus.helpers import market_display_name

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

# Only the fields the dual-shape helpers below actually read off a raw offer
# row. The rest of LEG_FIELDS (stat/league/game/date/platform/push_prob) is
# canonical-only — every call site that needs one of those only ever holds a
# canonical leg, never a raw offer row, so it indexes directly (leg["game"])
# instead of going through leg_field.
_FIELD_TO_OFFER_COL = {
    "player": "Player",
    "team": "Team",
    "market": "Market",
    "bet": "Bet",
    "line": "Line",
    "win_prob": "Win Prob",
    "boost": "Boost",
    "kelly": "Kelly",
}


def leg_field(leg: Mapping, field: str, default=None):
    """Read one leg field off either a canonical leg (lowercase) or a raw
    current_offers row (Title-Case) — the shared identity/display helpers
    (corr_key, find_offer_idx, validate_parlay_legs, the constellation's node
    fields) run on both a slip's own legs and not-yet-picked offer-row
    candidates from the same pool.
    """
    if field in leg:
        return leg[field]
    return leg.get(_FIELD_TO_OFFER_COL[field], default)


def is_model_liked(leg: Mapping) -> bool:
    """Whether a leg's Kelly edge is positive — the star-vs-satellite split
    the constellation, satellite picker, and slip builder all filter on.
    Runs on either a canonical leg or a raw current_offers row via leg_field.
    """
    return float(leg_field(leg, "kelly", 0.0) or 0.0) > 0


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
    market = market_display_name(leg.get("league", ""), leg["market"])
    return f"{leg['player']} {leg['bet']} {leg['line']:.10g} {market}"
