"""Per-family prophecy headline: ``attach_parlay_theses`` and its machinery.

One headline per ``(League, Game, Family)``, written onto every row of the
family. Built from the family's driving player (star-carousel ranking), the
leg-majority direction, the modal stat category, and the game shape (``O/U`` +
``Moneyline``, joined on ``Game``; ``Moneyline`` is the team's implied win
probability in ``(0, 1)``). A slate-uniqueness pass guarantees no two families
in the same ``(League, Date)`` share a thesis. The variant pick is a stable md5
seed including ``Date`` — deterministic per snapshot, rotating day to day.
"""

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass

import pandas as pd

from sportstradamus.prediction.stories.bank import _bank_cell, _stat_category
from sportstradamus.prediction.stories.legs import parse_leg

# Per-league raw-game-total bands: at/above the high band a slate plays as a
# shootout, at/below the low band as a rockfight/grind. Bands sit on the live
# O/U column (NBA/WNBA points in the 200s, NHL goals ~6, NFL points ~45, MLB
# runs ~9); reasonable not tuned — the thesis voice, not a wager, rides them.
_TOTAL_BANDS: dict[str, tuple[float, float]] = {
    "NBA": (235.0, 215.0),
    "WNBA": (170.0, 155.0),
    "NFL": (49.0, 41.0),
    "NHL": (6.5, 5.5),
    "MLB": (9.5, 7.5),
}
# Fallback band when a league has no entry — split around a generic midpoint so
# an unknown league still classifies high/low instead of always "even".
_DEFAULT_TOTAL_BAND: tuple[float, float] = (1.10, 0.90)

# How far a side's implied win probability must sit from a coin flip (0.5) for
# the game to read as lopsided (blowout / runaway) rather than tight
# (slugfest / coinflip). 0.18 ⇒ ~68% favorite, a clearly favored side.
_ML_LOPSIDED_MARGIN: float = 0.18


def attach_parlay_theses(parlays: pd.DataFrame, offers: pd.DataFrame) -> pd.DataFrame:
    """Add a ``Thesis`` column: one prophecy headline per ``(League, Game, Family)``.

    The headline is constant within a family but written onto every row of it.
    Built from the family's driving player (star-carousel ranking over its legs),
    the leg-majority direction, the modal stat category, and the game's shape
    (from ``offers`` ``O/U`` + ``Moneyline``, joined on ``Game``). A
    slate-uniqueness pass guarantees no two families in the same ``(League,
    Date)`` share a thesis. The frame is mutated in place and returned.
    """
    if parlays.empty:
        parlays["Thesis"] = pd.Series(dtype=str)
        return parlays

    shapes = _game_shapes(offers)
    theses: dict[tuple, _Thesis] = {}
    for key, fam_rows in parlays.groupby(["League", "Game", "Family"], dropna=False):
        theses[key] = _family_thesis(key, fam_rows, shapes)
    _dedupe_slate(theses)

    keyed = list(zip(parlays["League"], parlays["Game"], parlays["Family"], strict=True))
    parlays["Thesis"] = [theses[k].text if k in theses else "" for k in keyed]
    return parlays


@dataclass
class _Thesis:
    """A family's headline plus the slots a uniqueness bump needs to vary it."""

    text: str
    cell: list[str]
    index: int
    player: str
    game_label: str
    date: str
    tag: str  # distinguishing suffix source (player / stat) once a cell exhausts

    def render(self, idx: int) -> str:
        return self.cell[idx].format(p=self.player, g=self.game_label)


def _game_shapes(offers: pd.DataFrame) -> dict[str, str]:
    """Map each ``Game`` to a named shape from its total + win-prob spread.

    Returns ``{}`` when ``offers`` lacks the ``Game`` key (not yet a snapshot
    column in early P2) or the context columns; families then fall back to the
    league-agnostic ``"even"`` shape.
    """
    if offers.empty or "Game" not in offers.columns:
        return {}
    shapes: dict[str, str] = {}
    for game, grp in offers.groupby("Game", dropna=True):
        league = grp["League"].iloc[0] if "League" in grp.columns else ""
        total = pd.to_numeric(grp.get("O/U"), errors="coerce").dropna()
        mls = pd.to_numeric(grp.get("Moneyline"), errors="coerce").dropna()
        shapes[game] = _classify_shape(
            league,
            total.median() if not total.empty else None,
            (mls - 0.5).abs().max() if not mls.empty else None,
        )
    return shapes


def _classify_shape(league: str, total: float | None, ml_margin: float | None) -> str:
    """Name the game script from the total band and the win-prob margin."""
    lopsided = ml_margin is not None and ml_margin >= _ML_LOPSIDED_MARGIN
    tight = ml_margin is not None and ml_margin < _ML_LOPSIDED_MARGIN
    if lopsided:
        return "blowout"
    high, low = _TOTAL_BANDS.get(league, _DEFAULT_TOTAL_BAND)
    if total is not None and total >= high:
        return "shootout"
    if total is not None and total <= low:
        return "grind"
    if tight:
        return "coinflip"
    return "even"


@dataclass
class _FamilyLegs:
    """Per-(player) leg accumulation for one family, feeding the star ranking."""

    by_player: dict[str, list[tuple[str, str, float]]]  # player -> [(bet, market, line)]
    markets: dict[str, set]
    totals: Counter
    market_lines: dict[str, list]


def _collect_family_legs(fam_rows: pd.DataFrame) -> _FamilyLegs:
    leg_cols = [c for c in fam_rows.columns if c.startswith("Leg ")]
    by_player: dict[str, list] = defaultdict(list)
    markets: dict[str, set] = defaultdict(set)
    totals: Counter = Counter()
    market_lines: dict[str, list] = defaultdict(list)
    for _, row in fam_rows.iterrows():
        for col in leg_cols:
            leg = parse_leg(row.get(col))
            if not leg:
                continue
            by_player[leg["Player"]].append((leg["Bet"], leg["Market"], leg["Line"]))
            markets[leg["Player"]].add(leg["Market"])
            totals[leg["Player"]] += 1
            market_lines[leg["Market"]].append(leg["Line"])
    return _FamilyLegs(by_player, markets, totals, market_lines)


def _line_pct(legs: _FamilyLegs, market: str, line: float) -> float:
    vals = legs.market_lines[market]
    return sum(1 for v in vals if v <= line) / len(vals) if vals else 0.0


def _stardom(legs: _FamilyLegs, player: str) -> tuple:
    """Star-carousel sort key: market breadth, biggest line-vs-market, volume, name.

    Reuses the ranking idea of the old dashboard bank — a featured player is
    offered in more markets and carries higher lines within them.
    """
    best_pct = max(
        (_line_pct(legs, mk, ln) for _, mk, ln in legs.by_player[player]),
        default=0.0,
    )
    return (len(legs.markets[player]), best_pct, legs.totals[player], player)


def _family_thesis(key: tuple, fam_rows: pd.DataFrame, shapes: dict[str, str]) -> _Thesis:
    league, game, _family = key
    date = str(fam_rows["Date"].iloc[0]) if "Date" in fam_rows.columns else ""
    label = _game_label(game, league)
    legs = _collect_family_legs(fam_rows)
    if not legs.totals:
        return _Thesis("", [], 0, "", label, date, "")

    player = max(legs.totals, key=lambda p: _stardom(legs, p))
    player_legs = legs.by_player[player]
    overs = sum(1 for b, _, _ in player_legs if b == "Over")
    direction = "Over" if overs * 2 >= len(player_legs) else "Under"
    category = Counter(_stat_category(m) for _, m, _ in player_legs).most_common(1)[0][0]
    shape = shapes.get(game, "even")

    cell = _bank_cell(shape, direction, category)
    seed_src = f"{game}|{_family}|{player}|{date}|{shape}".encode()
    index = int(hashlib.md5(seed_src).hexdigest(), 16) % len(cell)
    thesis = _Thesis(cell[index], cell, index, player, label, date, player)
    thesis.text = thesis.render(index)
    return thesis


def _game_label(game: str, league: str) -> str:
    """Human game label; falls back to the league when ``Game`` is unkeyed."""
    return game if isinstance(game, str) and game.strip() else (league or "the slate")


def _dedupe_slate(theses: dict[tuple, _Thesis]) -> None:
    """Guarantee distinct theses within each ``(League, Date)`` slate.

    Walks each slate's families in key order; on a collision, advances the
    later family to the next variant in its cell (cycling), and if the whole
    cell is exhausted appends a distinguishing tag so the text still differs.
    Mutates the ``_Thesis`` objects in place.
    """
    slates: dict[tuple, list[tuple]] = defaultdict(list)
    for key, thesis in theses.items():
        slates[(key[0], thesis.date)].append(key)

    for fam_keys in slates.values():
        seen: set[str] = set()
        for key in sorted(fam_keys, key=lambda k: tuple(str(part) for part in k)):
            thesis = theses[key]
            if not thesis.text:
                continue
            _bump_until_unique(thesis, seen)
            seen.add(thesis.text)


def _bump_until_unique(thesis: _Thesis, seen: set[str]) -> None:
    for step in range(len(thesis.cell)):
        idx = (thesis.index + step) % len(thesis.cell)
        candidate = thesis.render(idx)
        if candidate not in seen:
            thesis.index, thesis.text = idx, candidate
            return
    base = thesis.render(thesis.index)
    suffix = 2
    while f"{base} — {thesis.tag}" in seen:
        thesis.tag = f"{thesis.tag} ({suffix})"
        suffix += 1
    thesis.text = f"{base} — {thesis.tag}"
