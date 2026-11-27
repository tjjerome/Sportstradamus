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
    # The EV stamp's read of the leg (prediction/ev_stamp.py): the fitted win chance and
    # its EV, None on a leg the fit did not price, and whether the board called it a pick.
    "fit_prob",
    "ev",
    "pick",
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
# row — league among them, because the constellation resolves a board's display
# name for its labels, hovers and cards off raw candidate rows as readily as off
# slip legs; the platform and the fit's three too, because a star's size, card and
# pick flag read them off either shape. The rest of LEG_FIELDS
# (stat/game/date/push_prob/kelly) is canonical-only: every call site that needs one of
# those only ever holds a canonical leg, never a raw offer row, so it indexes
# directly (leg["game"]) instead of going through leg_field.
_FIELD_TO_OFFER_COL = {
    "player": "Player",
    "team": "Team",
    "market": "Market",
    "bet": "Bet",
    "line": "Line",
    "league": "League",
    "platform": "Platform",
    "win_prob": "Win Prob",
    "boost": "Boost",
    "fit_prob": "Fit Prob",
    "ev": "EV",
    "pick": "Pick",
}


def leg_field(leg: Mapping, field: str, default=None):
    """Read one leg field off either a canonical leg (lowercase) or a raw
    current_offers row (Title-Case) — the shared identity/display helpers
    (corr_key, find_offer_idx, validate_parlay_legs, the constellation's node
    fields) run on both a slip's own legs and not-yet-added offer-row
    candidates from the same pool.
    """
    if field in leg:
        return leg[field]
    return leg.get(_FIELD_TO_OFFER_COL[field], default)


def leg_field_float(leg: Mapping, field: str, default: float = 0.0) -> float:
    """``leg_field`` as a float, coalescing a missing/``None``/falsy read to ``default``.

    A leg present in the mapping with an explicit ``None`` bypasses
    ``leg_field``'s own default (it only applies when the key is absent), so
    the ``or default`` here is load-bearing, not decorative.
    """
    return float(leg_field(leg, field, default) or default)


def is_pick(leg: Mapping) -> bool:
    """Whether the board called the leg one of its picks (``live_fit.picks``) — the
    star-vs-the-rest split the constellation draws on. Runs on either a canonical
    leg or a raw current_offers row via leg_field; a leg stored before the flag
    existed reads as no pick.
    """
    return bool(leg_field(leg, "pick", False))


def _numeric_or_default(row: Mapping, col: str, default: float | None) -> float | None:
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
    leg["fit_prob"] = _numeric_or_default(row, "Fit Prob", None)
    leg["ev"] = _numeric_or_default(row, "EV", None)
    pick = row.get("Pick")
    leg["pick"] = bool(pd.notna(pick) and pick)
    return leg


def leg_label(leg: Mapping) -> str:
    """Human display string for a leg. Render-only — never parsed back."""
    market = market_display_name(leg.get("league", ""), leg["market"])
    return f"{leg['player']} {leg['bet']} {leg['line']:.10g} {market}"
