"""Parlay-leg parsing, stat-category vocabulary, and offer enrichment.

``parse_leg`` recovers a leg's Player/Bet/Line/Market from its display string
(``"{Player} {Bet} {Line} {Market} - {Model P}%, {Boost}x"``, built in
``prediction/correlation.py``). ``enrich_legs`` joins those parsed legs back to
the scored offers frame to attach each leg's canonical market, game, team, and
depth-chart position — the fields the archetype engine routes on. ``_stat_category``
maps a market to the coarse category the phrase bank is keyed by.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

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
# ("Avg 5" / "DVPOA" exist only for the latter).
_OFFER_ENRICH_COLS = ("Market", "Game", "Team", "Position", "Win Prob", "Avg 5", "DVPOA")


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


def enrich_legs(parsed: list[dict], offers: pd.DataFrame) -> list[Leg]:
    """Attach each leg's offer context (canonical market, game, team, position).

    Joins on ``(Player, Bet, Line)`` against the offers frame's uppercase
    columns; ``parsed`` legs carry the canonical lowercase schema keys
    (``player``/``bet``/``line``/``market``). A leg with no matching offer
    keeps its own market and carries no game/team/position (it simply can't
    anchor a unit/stack).
    """
    idx = offer_index(offers)
    out: list[Leg] = []
    for leg in parsed:
        match = idx.get((leg["player"], leg["bet"], leg["line"]))
        market = (match.get("Market") if match else None) or leg["market"]
        win_prob = match.get("Win Prob") if match else None
        out.append(
            Leg(
                player=leg["player"],
                bet=leg["bet"],
                line=leg["line"],
                market=market,
                game=match.get("Game") if match else None,
                team=match.get("Team") if match else None,
                position=match.get("Position") if match else None,
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


def parse_leg(leg: str) -> dict | None:
    """Splitting on ``" Over "`` / ``" Under "`` yields the player name even when
    it contains spaces.  Returns ``None`` for anything that does not parse.
    """
    if not isinstance(leg, str) or not leg.strip():
        return None
    head = leg.split(" - ", 1)[0].strip()
    for bet in ("Over", "Under"):
        token = f" {bet} "
        if token not in head:
            continue
        player, rest = head.split(token, 1)
        rest = rest.strip().split()
        if not rest:
            return None
        try:
            line = float(rest[0])
        except ValueError:
            return None
        market = " ".join(rest[1:]).strip()
        if not player.strip() or not market:
            return None
        return {"Player": player.strip(), "Bet": bet, "Line": line, "Market": market}
    return None


def validate_parlay_legs(
    legs: Sequence[Mapping], *, require_both_teams: bool = True
) -> tuple[bool, str]:
    """Whether a same-game leg-set is a valid DFS entry; ``(ok, reason_if_not)``.

    Underdog/Sleeper reject a same-game parlay that repeats a player (two markets
    on one name) or sits entirely on one team. Shared by the dashboard slip
    editor's lock-in gate and the story-menu preset generator, so a seeded preset
    is valid by construction. Legs carry the offer-row ``Player``/``Team``; a leg
    without a ``Team`` can't satisfy the both-sides rule on its own.

    ``require_both_teams=False`` relaxes only the both-sides rule (never the
    distinct-player rule): the menu sets it for a game whose model edge is entirely
    one-sided, so it can still offer that team's preset — the user adds the second
    team from another game (a satellite leg) in the editor.
    """
    players = [leg["Player"] for leg in legs]
    if len(set(players)) < len(players):
        return False, "Two legs share a player — each leg needs a distinct player."
    if require_both_teams and len({leg.get("Team") for leg in legs if leg.get("Team")}) < 2:
        return False, "Add a leg from the other team — a parlay needs both sides."
    return True, ""
