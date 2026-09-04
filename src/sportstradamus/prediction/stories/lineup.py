"""MLB lineup facts as offer columns: batting side, opposing hand, posted-vs-usual slot.

The story layer reads only ``current_offers`` columns, so tonight's lineup facts
have to become columns before ``why.py`` can turn them into clauses. Both sources
are the loaded ``StatsMLB``: ``players`` is the ``{id: {name, bats, throws}}``
registry, resolved by name because that is the only player key the offers frame
carries, and ``upcoming_games[team]`` holds the probable opposing starter plus the
posted batting order.
"""

import re

import pandas as pd

# Depth-chart label for an MLB hitter, "B1".."B9" — the batting slot resolved in
# correlation._resolve_player_positions. Other leagues reuse "B" for bench, so the
# League check in attach_lineup_columns is what keeps an NBA "B1" out of this.
_BATTING_SLOT = re.compile(r"B[1-9]")

# {slot} vocabulary: the top of the order has a name of its own, the rest read as
# plain ordinals.
_SLOT_WORDS = ("leadoff", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th")

_LINEUP_COLUMNS = ("Bats", "Opp Hand", "Lineup")


def batting_slot(position: object) -> str | None:
    """The ``{slot}`` word for a ``B1``-``B9`` depth label, or ``None`` if it isn't one."""
    label = str(position)
    if not _BATTING_SLOT.fullmatch(label):
        return None
    return _SLOT_WORDS[int(label[1]) - 1]


def attach_lineup_columns(offers: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Add ``Bats``, ``Opp Hand``, and ``Lineup`` for MLB hitters; ``""`` everywhere else.

    ``Lineup`` is ``"posted"`` when the hitter appears in tonight's posted order
    and ``"usual"`` when his slot came from ``get_depth``'s modal-slot fallback,
    so the prose can say which it is; the posted-order match is by name, exactly
    as ``get_depth`` makes it. ``Opp Hand`` stays blank when the game carries no
    probable starter. Pitchers, combo legs, and other leagues get blank strings so
    the parquet columns keep a plain string dtype, and so does every row when MLB
    is out of season and never loaded. The frame is mutated in place and returned.
    """
    mlb = stats.get("MLB")
    if offers.empty or mlb is None:
        blank = pd.Series(dtype=str) if offers.empty else ""
        for column in _LINEUP_COLUMNS:
            offers[column] = blank
        return offers

    bats_by_name, throws_by_name = _hands_by_name(mlb.players)
    bats, opp_hand, lineup = [], [], []
    for league, team, player, position in offers[
        ["League", "Team", "Player", "Position"]
    ].itertuples(index=False, name=None):
        if league != "MLB" or batting_slot(position) is None:
            bats.append("")
            opp_hand.append("")
            lineup.append("")
            continue
        game = mlb.upcoming_games.get(team, {})
        starter = game.get("Opponent Pitcher")
        bats.append(bats_by_name.get(player, ""))
        opp_hand.append(throws_by_name.get(starter, "") if starter else "")
        lineup.append("posted" if player in game.get("Batting Order", []) else "usual")
    offers["Bats"], offers["Opp Hand"], offers["Lineup"] = bats, opp_hand, lineup
    return offers


def _hands_by_name(players: dict) -> tuple[dict[str, str], dict[str, str]]:
    """Batting sides and throwing hands from the player registry, keyed by name.

    Split by role rather than kept as one record per name: the registry is
    id-keyed and names collide across roles, so a pitcher named Luis Garcia
    would otherwise shadow the batter of that name and cost him his side.
    Same-role duplicates stay last-wins.
    """
    return (
        {p["name"]: p["bats"] for p in players.values() if "bats" in p},
        {p["name"]: p["throws"] for p in players.values() if "throws" in p},
    )
