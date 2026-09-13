"""Parlay-leg parsing, stat-category vocabulary, and offer enrichment.

``enrich_legs`` joins canonical lowercase-keyed legs (from ``lower_leg`` here
or ``sportstradamus.leg_schema.build_leg``) back to the scored offers frame to
attach each leg's canonical market, game, team, and depth-chart position — the
fields the archetype engine routes on. ``_stat_category`` maps a market to the
coarse category the phrase bank is keyed by, and ``narrative_side`` reads the
market valence the same table sets: both live here because the negative-market
list is the one place that knowledge is written down.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from sportstradamus.leg_schema import leg_field
from sportstradamus.prediction.parlay import resolve_leg_stat
from sportstradamus.prediction.stories.context import Leg

# Bet-Under is the thriving side for these markets (mistake / damage-allowed
# counts), so narrative valence flips relative to the bet direction. Exact
# lowercase internal slugs; batter "walks" and "pitcher strikeouts" stay positive.
_NEGATIVE_MARKETS: frozenset[str] = frozenset(
    {
        "tov",
        "interceptions",
        "sacks taken",
        "fumbles lost",
        "goalsagainst",
        "walks allowed",
        "runs allowed",
        "hits allowed",
        "1st inning hits allowed",
        "batter strikeouts",
    }
)

# Map a leg market to a coarse stat category so the bank can pick imagery that
# fits the stat. Needles cover every leg vocabulary in play: canonical codes
# ("PRA", "FG3M"), Underdog display names ("Pts + Rebs + Asts"), and Sleeper
# snake keys ("pts_reb_ast"), across NBA/WNBA/NFL/NHL/MLB.
_STAT_CATEGORY = {
    "scoring": (
        "point",
        "pts",
        "pra",
        "pr",
        "pa",
        "p+",
        "3-p",
        "3pt",
        "three",
        "threes",
        "fg3",
        "fgm",
        "fga",
        "fg_",
        "ftm",
        "free throw",
        "pass yd",
        "passing yards",
        "pass_yds",
        "pass td",
        "passing td",
        "rush yd",
        "rushing yards",
        "rush_yds",
        "rec yd",
        "receiving yards",
        "rec_yds",
        "kicking points",
        "goal",
        "shots on goal",
        "sog",
        "total bases",
        "hits",
        "rbi",
        "runs",
    ),
    "boards": ("rebound", "reb", "board", "ra", "pr"),
    "playmaking": (
        "assist",
        "ast",
        "pa",
        "playmak",
        "dish",
        "completions",
        "pass att",
        "receptions",
        "targets",
    ),
    "stops": (
        "steal",
        "stl",
        "block",
        "blk",
        "blst",
        "stocks",
        "tackle",
        "sack",
        "interception",
        "blocked",
    ),
    "k's": (
        "strikeout",
        "pitcher strikeouts",
        "pitcher_strikeouts",
        "ks",
        "_k",
        "strikeouts",
        "saves",
        "outs",
    ),
}

# Offer columns kept per offer_index record, beyond the (Player, Bet, Line)
# match key — read by enrich_legs and the story dek's anchor clauses
# ("Avg 5", "DVPOA", "Opp Hand", and "Lineup" exist only for the latter; the
# dek says which batting slot and against which hand, never the hitter's side).
_OFFER_ENRICH_COLS = (
    "Market",
    "Game",
    "Team",
    "Position",
    "Win Prob",
    "Avg 5",
    "DVPOA",
    "Opp Hand",
    "Lineup",
)


def _stat_category(market: str) -> str:
    m = (market or "").lower()
    # Negative markets resolve first: their substrings ("goal", "runs", "hits",
    # "interception", ...) would otherwise collide with thriving-stat needles.
    if m in _NEGATIVE_MARKETS:
        return "mistakes"
    for cat, needles in _STAT_CATEGORY.items():
        if any(n in m for n in needles):
            return cat
    return "production"


def narrative_side(leg: Leg) -> str:
    """The leg's thriving direction: its bet, flipped on a negative market."""
    if not leg.negative:
        return leg.bet
    return "Under" if leg.bet == "Over" else "Over"


def lower_leg(row: Mapping, new_map: dict) -> dict:
    """Map a canonical uppercase-keyed leg row to the lowercase keys ``enrich_legs`` wants.

    ``new_map`` is the platform's display-name → slug map (``analysis._leg_market_map``);
    ``stat`` is the slug behind the display ``Market``, resolved the same way the
    persisted leg's ``Stat`` field is so a leg that misses the offers join still
    categorizes off the slug. Pass ``{}`` where only the display fields are read.
    """
    return {
        "player": row["Player"],
        "bet": row["Bet"],
        "line": row["Line"],
        "market": row["Market"],
        "stat": resolve_leg_stat(row["Market"], new_map),
    }


def enrich_legs(parsed: list[dict], offers: pd.DataFrame) -> list[Leg]:
    """Attach each leg's offer context (canonical market, game, team, position).

    Joins on ``(Player, Bet, Line)`` against the offers frame's uppercase
    columns; ``parsed`` legs carry the canonical lowercase schema keys
    (``player``/``bet``/``line``/``market``, plus the ``stat`` slug both
    ``lower_leg`` and ``leg_schema.build_leg`` resolve). A leg with no matching
    offer falls back to that slug rather than its platform display name, because
    both tables below key on the slug: "INTs Thrown" would categorize as
    ``production`` and lose the valence flip ``narrative_side`` reads off it.
    Such a leg carries no game/team/position (it simply can't anchor a
    unit/stack).
    """
    idx = offer_index(offers)
    out: list[Leg] = []
    for leg in parsed:
        match = idx.get((leg["player"], leg["bet"], leg["line"])) or {}
        market = match.get("Market") or leg.get("stat") or leg["market"]
        win_prob = match.get("Win Prob")
        out.append(
            Leg(
                player=leg["player"],
                bet=leg["bet"],
                line=leg["line"],
                market=market,
                game=match.get("Game"),
                team=match.get("Team"),
                position=match.get("Position"),
                category=_stat_category(market),
                negative=market.lower() in _NEGATIVE_MARKETS,
                win_prob=None if win_prob is None or pd.isna(win_prob) else float(win_prob),
            )
        )
    return out


def offer_index(offers: pd.DataFrame) -> dict[tuple, dict]:
    """Map ``(Player, Bet, Line)`` to its first matching offer record.

    Shared by :func:`enrich_legs` here and the story dek's anchor lookup in
    ``why.py``. ``setdefault`` keeps the first offer on a duplicate key.
    """
    if offers is None or offers.empty or not {"Player", "Bet", "Line"}.issubset(offers.columns):
        return {}
    keep = ["Player", "Bet", "Line", *(c for c in _OFFER_ENRICH_COLS if c in offers.columns)]
    idx: dict[tuple, dict] = {}
    for rec in offers[keep].to_dict("records"):
        idx.setdefault((rec["Player"], rec["Bet"], rec["Line"]), rec)
    return idx


def validate_parlay_legs(
    legs: Sequence[Mapping], *, require_both_teams: bool = True
) -> tuple[bool, str]:
    """Whether a same-game leg-set is a valid DFS entry; ``(ok, reason_if_not)``.

    Underdog/Sleeper reject a same-game parlay that repeats a player (two markets
    on one name) or sits entirely on one team. Shared by the dashboard slip
    editor's lock-in gate and the story-menu preset generator, so a seeded preset
    is valid by construction. ``leg_field`` reads player/team off either a
    canonical lowercase leg or a raw uppercase ``current_offers`` row; a leg
    without a team can't satisfy the both-sides rule on its own.

    ``require_both_teams=False`` relaxes only the both-sides rule (never the
    distinct-player rule): the menu sets it for a game whose model edge is entirely
    one-sided, so it can still offer that team's preset — the user adds the second
    team from another game (a satellite leg) in the editor.
    """
    players = [leg_field(leg, "player") for leg in legs]
    if len(set(players)) < len(players):
        return False, "Two legs share a player — each leg needs a distinct player."
    if require_both_teams and len({t for leg in legs if (t := leg_field(leg, "team"))}) < 2:
        return False, "Add a leg from the other team — a parlay needs both sides."
    return True, ""
