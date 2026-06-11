"""Family-label phrase bank and game-story helpers.

Hosts the family_labels_for_game star carousel and its sub-functions,
temporarily parked here until a later phase moves it pipeline-side.
"""

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass

import pandas as pd

from sportstradamus.dashboard.legs import parse_leg

# Map a market name to a coarse stat category so the phrase bank can pick a
# verb that fits the stat ("dominates the glass" only makes sense for boards).
# Needles cover every leg vocabulary in play: pretty names ("3-Pointers
# Made"), Sleeper snake keys ("threes_made"), and canonical codes ("FG3M").
_STAT_CATEGORY = {
    "scoring": (
        "point",
        "pts",
        "3-p",
        "3pt",
        "three",
        "fg3",
        "fg ",
        "fga",
        "fgm",
        "pass yd",
        "pass td",
        "rush yd",
        "rec yd",
        "goal",
        "shots",
    ),
    "boards": ("rebound", "reb", "board", "glass"),
    "playmaking": ("assist", "ast", "playmak"),
    "k's": ("strikeout", "pitcher k", "ks", "_k", "saves"),
}

# Deterministic, offline phrase bank keyed by (direction, stat_category).
# A stable hash of (player, family) selects the variant so names never flip
# between Streamlit reruns.
_PHRASES = {
    ("Over", "scoring"): ["{p} goes nuclear", "{p} can't miss", "{p} fills it up"],
    ("Over", "boards"): ["{p} dominates the glass", "{p} crashes the boards", "{p} cleans up"],
    ("Over", "playmaking"): ["{p} runs the show", "{p} dishes all night", "{p} sets the table"],
    ("Over", "k's"): ["{p} mows them down", "{p} racks up Ks", "{p} is unhittable"],
    ("Over", "production"): ["{p} goes off", "{p} stuffs the stat sheet", "{p} takes over"],
    ("Under", "scoring"): ["{p} ice cold", "{p} held in check", "{p} can't buy a bucket"],
    ("Under", "boards"): ["{p} boxed out", "{p} off the glass", "{p} disappears inside"],
    ("Under", "playmaking"): ["{p} can't find anyone", "{p} bottled up", "{p} held quiet"],
    ("Under", "k's"): ["{p} gets hit around", "{p} no swing-and-miss", "{p} laboring"],
    ("Under", "production"): ["{p} struggles", "{p} no-shows", "{p} quiet night"],
}


def _stat_category(market: str) -> str:
    """Coarse stat bucket for the phrase bank."""
    m = (market or "").lower()
    for cat, needles in _STAT_CATEGORY.items():
        if any(n in m for n in needles):
            return cat
    return "production"


def _phrase(player: str, family: float, direction: str, category: str) -> str:
    """Pick a stable phrase-bank variant for a family headliner."""
    bank = _PHRASES.get((direction, category)) or _PHRASES[(direction, "production")]
    seed = f"{player}|{family}".encode()
    idx = int(hashlib.md5(seed).hexdigest(), 16) % len(bank)
    return bank[idx].format(p=player)


@dataclass
class _GameLegStats:
    """Per-game leg accumulation feeding the star-carousel family labels.

    All four are built together by :func:`_collect_family_legs` and consumed
    together by the stardom ranking and label-naming helpers.
    """

    fam_legs: dict[float, dict[str, list]]
    player_markets: dict[str, set]
    player_total: Counter
    market_lines: dict[str, list]


def family_labels_for_game(game_group: pd.DataFrame) -> dict[float, str]:
    """Map each family in one game's parlays to a fun, distinct headline.

    "Star carousel": rank the game's players by stardom — number of distinct
    stat markets they're offered in (stars get props everywhere), tie-broken by
    how big their biggest line is relative to that market (stars carry high
    lines).  Hand the #1 star to the family they most drive (most legs), the
    #2 star to the next family, and so on, so every family gets a recognizable,
    different headliner.  Families left after the stars are exhausted fall
    through to their own next-best player — never a neutral placeholder.
    """
    leg_cols = [c for c in game_group.columns if c.startswith("Leg ")]
    families = sorted(game_group["Family"].dropna().unique())
    if not families:
        return {}

    stats = _collect_family_legs(game_group, leg_cols, families)
    if not stats.player_total:
        return dict.fromkeys(families, "Mixed bag")

    ranked = sorted(
        stats.player_total, key=lambda p: _player_stardom(stats, families, p), reverse=True
    )
    labels, taken = _assign_stars_to_families(stats, families, ranked)
    _fill_remaining_families(stats, families, labels, taken)
    return labels


def _collect_family_legs(
    game_group: pd.DataFrame, leg_cols: list[str], families: list[float]
) -> _GameLegStats:
    """Tally per-(family, player) legs and per-player/-market breadth for one game."""
    fam_legs: dict[float, dict[str, list]] = {f: defaultdict(list) for f in families}
    player_markets: dict[str, set] = defaultdict(set)
    player_total: Counter = Counter()
    market_lines: dict[str, list] = defaultdict(list)

    for _, row in game_group.iterrows():
        fam = row["Family"]
        if fam not in fam_legs:
            continue
        for col in leg_cols:
            p = parse_leg(row.get(col))
            if not p:
                continue
            fam_legs[fam][p["Player"]].append((p["Bet"], p["Market"], p["Line"]))
            player_markets[p["Player"]].add(p["Market"])
            player_total[p["Player"]] += 1
            market_lines[p["Market"]].append(p["Line"])

    return _GameLegStats(fam_legs, player_markets, player_total, market_lines)


def _line_pct(stats: _GameLegStats, market: str, line: float) -> float:
    """Fraction of this market's seen lines at or below ``line`` (0 if unseen)."""
    vals = stats.market_lines[market]
    return sum(1 for v in vals if v <= line) / len(vals) if vals else 0.0


def _player_stardom(stats: _GameLegStats, families: list[float], player: str) -> tuple:
    """Stardom sort key: market breadth, biggest line-vs-market, volume, name.

    All three numeric components are free proxies for "this is a featured player".
    """
    best_pct = max(
        (
            _line_pct(stats, mk, ln)
            for fam in families
            for _, mk, ln in stats.fam_legs[fam].get(player, [])
        ),
        default=0.0,
    )
    return (len(stats.player_markets[player]), best_pct, stats.player_total[player], player)


def _family_label_name(stats: _GameLegStats, fam: float, player: str) -> str:
    """Phrase-bank headline for ``player`` as the face of family ``fam``."""
    legs = stats.fam_legs[fam][player]
    over = sum(1 for b, _, _ in legs if b == "Over")
    direction = "Over" if over * 2 >= len(legs) else "Under"
    category = Counter(_stat_category(m) for _, m, _ in legs).most_common(1)[0][0]
    return _phrase(player, fam, direction, category)


def _assign_stars_to_families(
    stats: _GameLegStats, families: list[float], ranked: list[str]
) -> tuple[dict[float, str], set[str]]:
    """#1 star -> family they most drive, #2 -> next family, ... Returns (labels, taken)."""
    labels: dict[float, str] = {}
    taken: set[str] = set()
    for player in ranked:
        if len(labels) == len(families):
            break
        cands = [f for f in families if f not in labels and player in stats.fam_legs[f]]
        if not cands:
            continue
        fam = max(cands, key=lambda f: (len(stats.fam_legs[f][player]), -f))
        labels[fam] = _family_label_name(stats, fam, player)
        taken.add(player)
    return labels, taken


def _fill_remaining_families(
    stats: _GameLegStats, families: list[float], labels: dict[float, str], taken: set[str]
) -> None:
    """Give each unlabeled family its own best-remaining (then any) player; mutates in place."""
    for fam in families:
        if fam in labels:
            continue
        pool = sorted(
            stats.fam_legs[fam], key=lambda p: _player_stardom(stats, families, p), reverse=True
        )
        pick = next((p for p in pool if p not in taken), pool[0] if pool else None)
        labels[fam] = _family_label_name(stats, fam, pick) if pick else "Mixed bag"
        if pick:
            taken.add(pick)
