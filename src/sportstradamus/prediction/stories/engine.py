"""The v2 prophecy engine: route a leg-set + game context to a headline.

``route`` is the heart — a pure classifier that picks one of four story
archetypes from the actual legs in a slip and the game it belongs to, with no
dependency on the legacy parlay-family clustering:

* **player** — one player demonstrably drives the slip (a *unique* leg-majority,
  ≥ 2 legs). The uniqueness gate is the fix for v1's arbitrary alphabetical
  "star" on no-standout slips.
* **stack** — a correlated same-game bundle (≥ 3 legs, ≥ 2 players, mean
  bet-signed ρ over the slip's pairs clears the floor).
* **unit** — a position group on one team with a real DVPOA matchup edge.
* **game-script** — the game shape itself is the story; the universal fallback
  that never names an arbitrary player.

``thesis`` turns the routed archetype into a rendered headline via the phrase
bank; the variant is a deterministic md5 of the leg-set + date, so a snapshot
always renders the same copy yet the same matchup rotates day to day.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import combinations

from sportstradamus.prediction.stories.bank import bank_cell
from sportstradamus.prediction.stories.context import GameCtx, Leg

# League → phrase-bank voice. NBA and WNBA share the basketball voice; an unknown
# league reads straight from the league-neutral ``shared`` cells.
_VOICE_BY_LEAGUE = {
    "NBA": "basketball",
    "WNBA": "basketball",
    "NFL": "football",
    "NHL": "hockey",
    "MLB": "baseball",
}

# Archetype firing gates (named per CLAUDE.md §9). A player must hold a *unique*
# leg majority of at least _PLAYER_MIN_LEGS; a stack needs _STACK_MIN_LEGS
# correlated legs averaging at least _STACK_MEAN_RHO; a unit needs
# _UNIT_MIN_LEGS legs in one (team, group, direction) whose matchup edge clears
# _UNIT_EDGE_FLOOR (the same 0.05 floor the per-offer "why" uses).
_PLAYER_MIN_LEGS: int = 2
_STACK_MIN_LEGS: int = 3
_STACK_MEAN_RHO: float = 0.10
_UNIT_MIN_LEGS: int = 2
_UNIT_EDGE_FLOOR: float = 0.05


def route(legs: Sequence[Leg], ctxs: Mapping[str, GameCtx]) -> tuple[str, dict]:
    """Pick the archetype and its slot subject for a leg-set. Pure, bank-free."""
    if not legs:
        return "game-script", {}
    game = _primary_game(legs)
    ctx = ctxs.get(game)
    label = game or ""
    glegs = _primary_legs(legs, game)

    player = _try_player(glegs, label)
    if player is not None:
        return "player", player
    stack = _try_stack(glegs, ctx, label)
    if stack is not None:
        return "stack", stack
    unit = _unit_subject(glegs, ctx, label)
    if unit is not None:
        return "unit", unit
    return "game-script", {"g": label}


def _primary_legs(legs: Sequence[Leg], game: str | None) -> list[Leg]:
    same = [leg for leg in legs if leg.game == game]
    return same or list(legs)


def _try_player(legs: Sequence[Leg], label: str) -> dict | None:
    player, count, n = _dominant_player(legs)
    if player is not None and count >= _PLAYER_MIN_LEGS and 2 * count >= n:
        return {"p": player, "g": label}
    return None


def _try_stack(legs: Sequence[Leg], ctx: GameCtx | None, label: str) -> dict | None:
    if ctx is None or len(legs) < _STACK_MIN_LEGS or _distinct_players(legs) < 2:
        return None
    if _mean_rho(legs, ctx) < _STACK_MEAN_RHO:
        return None
    return {"n": len(legs), "p": _anchor(legs), "g": label}


def _primary_game(legs: Sequence[Leg]) -> str | None:
    games = Counter(leg.game for leg in legs if leg.game)
    if not games:
        return None
    return sorted(games, key=lambda g: (-games[g], g))[0]


def _dominant_player(legs: Sequence[Leg]) -> tuple[str | None, int, int]:
    """The uniquely most-legged player (or ``None`` on a tie), its count, and n."""
    counts = Counter(leg.player for leg in legs)
    n = len(legs)
    ranked = counts.most_common()
    if len(ranked) == 1:
        return ranked[0][0], ranked[0][1], n
    (p1, c1), (_p2, c2) = ranked[0], ranked[1]
    return (p1, c1, n) if c1 > c2 else (None, 0, n)


def _distinct_players(legs: Sequence[Leg]) -> int:
    return len({leg.player for leg in legs})


def _leg_id(leg: Leg) -> str:
    return f"{leg.player}|{leg.market}|{leg.bet}"


def _mean_rho(legs: Sequence[Leg], ctx: GameCtx) -> float:
    ids = [_leg_id(leg) for leg in legs]
    vals = [ctx.rho.get(frozenset({a, b})) for a, b in combinations(ids, 2)]
    found = [v for v in vals if v is not None]
    return sum(found) / len(found) if found else 0.0


def _anchor(legs: Sequence[Leg]) -> str:
    counts = Counter(leg.player for leg in legs)
    return sorted(counts, key=lambda p: (-counts[p], p))[0]


def _pos_group(position: str | None) -> str:
    return re.sub(r"\d+$", "", position or "")


def _unit_subject(legs: Sequence[Leg], ctx: GameCtx | None, label: str) -> dict | None:
    if ctx is None or not ctx.pos_edges:
        return None
    for (team, grp, bet), count in _unit_groups(legs).items():
        if count >= _UNIT_MIN_LEGS and _unit_favorable(ctx, team, grp, bet):
            return {"team": team, "grp": grp, "opp": _opponent(label, team), "g": label}
    return None


def _unit_groups(legs: Sequence[Leg]) -> Counter:
    """Count legs per ``(team, position-group, direction)`` — a unit candidate."""
    groups: Counter = Counter()
    for leg in legs:
        if leg.team and leg.position:
            groups[(leg.team, _pos_group(leg.position), leg.bet)] += 1
    return groups


def _unit_favorable(ctx: GameCtx, team: str, grp: str, bet: str) -> bool:
    """A group's DVPOA edge clears the floor in the leg's direction."""
    dvpoa = ctx.pos_edges.get(team, {}).get(grp, {}).get("dvpoa")
    if dvpoa is None:
        return False
    return dvpoa >= _UNIT_EDGE_FLOOR if bet == "Over" else dvpoa <= -_UNIT_EDGE_FLOOR


def _opponent(game: str, team: str) -> str:
    others = [side for side in (game or "").split("/") if side != team]
    return others[0] if others else ""


def thesis_variants(
    legs: Sequence[Leg], ctxs: Mapping[str, GameCtx]
) -> tuple[list[str], int, dict]:
    """Rendered variant list, the seeded pick index, and slot subject for a leg-set.

    The slate-uniqueness pass bumps the index among the rendered variants on a
    collision, so the engine hands back the whole cell, not just one string.
    """
    if not legs:
        return [], 0, {}
    archetype, subject = route(legs, ctxs)
    game = _primary_game(legs)
    ctx = ctxs.get(game)
    sub_legs = _subject_legs(legs, game, archetype, subject)
    shape = ctx.shape if ctx else "even"
    league = ctx.league if ctx else ""
    cell = bank_cell(
        _VOICE_BY_LEAGUE.get(league, "shared"),
        archetype,
        shape,
        _direction(sub_legs),
        _modal_category(sub_legs),
    )
    rendered = [variant.format(**subject) for variant in cell]
    if not rendered:
        return [], 0, subject
    idx = int(hashlib.md5(_seed(game, legs, ctx, archetype).encode()).hexdigest(), 16) % len(
        rendered
    )
    return rendered, idx, subject


def _subject_legs(
    legs: Sequence[Leg], game: str | None, archetype: str, subject: dict
) -> list[Leg]:
    glegs = _primary_legs(legs, game)
    if archetype == "player":
        return [leg for leg in glegs if leg.player == subject.get("p")] or glegs
    return glegs


def _direction(legs: Sequence[Leg]) -> str:
    overs = sum(1 for leg in legs if leg.bet == "Over")
    return "Over" if overs * 2 >= len(legs) else "Under"


def _modal_category(legs: Sequence[Leg]) -> str:
    cats = Counter(leg.category for leg in legs)
    return cats.most_common(1)[0][0] if cats else "production"


def _seed(game: str | None, legs: Sequence[Leg], ctx: GameCtx | None, archetype: str) -> str:
    date = ctx.date if ctx else ""
    shape = ctx.shape if ctx else "even"
    ids = "|".join(sorted(_leg_id(leg) for leg in legs))
    return f"{game or ''}|{ids}|{date}|{shape}|{archetype}"
