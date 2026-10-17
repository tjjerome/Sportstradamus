"""Star prominence and the lead story each game's menu headlines.

Ranked on edge alone, a game's menu anchors on its highest-probability leg,
which on DFS x.5 lines is usually a role player's Under. These pure pieces
re-anchor it on the game's stars from snapshot columns alone, no new data:
``attach_prominence`` scores every offer's player, ``lead_seeds`` picks the leg
an Over-led and an Under-led story grow from, and ``assign_leads`` flags the one
story per ``(platform, Game, Date)`` the Games and Tonight tabs headline, with
Over leads filling toward half of each date's games. An Over leads only a game
whose script runs hot or lopsided (``OVER_LEAD_SHAPES``).
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

# Rank-1 depth labels: correlation._LEAGUE_POSITIONS groups plus the usage rank
# within team x group (NBA "B" is the big, not the bench). No MLB entry: B1..B9 is
# the batting slot, not usage (owner decision 2026-10-03).
STAR_RANK_ONE: dict[str, frozenset[str]] = {
    "NBA": frozenset({"P1", "W1", "B1", "C1", "F1"}),
    "WNBA": frozenset({"G1", "F1", "C1"}),
    "NFL": frozenset({"QB1", "WR1", "RB1", "TE1"}),
    "NHL": frozenset({"C1", "W1", "D1", "G1"}),
}

# Over first: menu._cluster_strong_legs grows the led clusters in this order, so the
# Over-led one claims its legs before the Under-led one.
LEAD_SIDES: tuple[str, str] = ("Over", "Under")
# Owner rule (2026-10-17): an Over leads a game only when its script (``GameCtx.shape``)
# runs hot or lopsided; in a grind, a coinflip or an unquoted game the Over legs ride
# only in unled stories, so a story never sells an Over against its own game read.
OVER_LEAD_SHAPES: frozenset[str] = frozenset({"shootout", "blowout"})


def attach_prominence(offers: pd.DataFrame) -> pd.DataFrame:
    """Add ``Star`` in [0, 3]: rank-1 depth + line percentile + market breadth.

    ``line_pct`` ranks the player's median ``Line`` within ``(League, Market)``
    so alt-line rungs count once, and ``breadth`` is the player's distinct
    markets over the league's slate maximum; both pool the two platforms. They
    exist for the star the minutes rank buries (A'ja Wilson posts as F3): a top
    line on a full market menu can outscore a rank-1 label. Combo legs score on
    those terms like any player while their blank ``Position`` keeps depth at
    zero. The frame is mutated in place and returned.
    """
    if offers.empty:
        offers["Star"] = pd.Series(dtype="float64")
        return offers
    median_line = offers.groupby(["League", "Market", "Player"])["Line"].median()
    line_pct = median_line.groupby(level=["League", "Market"]).rank(pct=True, method="average")
    n_markets = offers.groupby(["League", "Player"])["Market"].nunique()
    breadth = n_markets / n_markets.groupby(level="League").transform("max")
    depth = [
        position in STAR_RANK_ONE.get(league, frozenset())
        for league, position in zip(offers["League"], offers["Position"], strict=True)
    ]
    terms = (
        offers[["League", "Market", "Player"]]
        .join(line_pct.rename("line_pct"), on=["League", "Market", "Player"])
        .join(breadth.rename("breadth"), on=["League", "Player"])
        .assign(depth=depth)
    )
    offers["Star"] = terms["depth"] + terms["line_pct"] + terms["breadth"]
    return offers


def lead_seeds(
    edge: Mapping[int, float],
    bet_df: Mapping[int, Mapping],
    star_of: Mapping[int, float],
    shape: str,
) -> dict[str, int]:
    """Per side, the strong leg id its lead story grows from.

    ``edge`` maps the game's strong legs to their ``Model EV`` and ``star_of``
    maps them to ``Star``. The most prominent leg wins, then the bigger edge,
    then the alphabetically earlier player; a side with no strong leg is left
    out, and so is Over unless the game's ``shape`` is in ``OVER_LEAD_SHAPES``.
    """
    seeds = {}
    for side in LEAD_SIDES:
        if side == "Over" and shape not in OVER_LEAD_SHAPES:
            continue
        legs = [i for i in edge if bet_df[i]["Bet"] == side]
        if legs:
            seeds[side] = min(legs, key=lambda i: (-star_of[i], -edge[i], bet_df[i]["Player"]))
    return seeds


def assign_leads(
    stories: pd.DataFrame, commence_by_game: Mapping[tuple[str, str], str]
) -> pd.DataFrame:
    """Flag ``lead`` on one story per ``(platform, Game, Date)``.

    ``_pick_sides`` gives each ``(Date, Game)`` one side for both platforms,
    with Over leads filling toward half of each date's games. Each platform
    flags its story led on that side, else its story led on the other side,
    else its best Shoot-the-Moon EV story, on both objective rows, so both
    platforms agree whenever both offer the side. ``commence_by_game`` is each
    game's tip keyed ``(Date, Game)``, ``""`` when untimed, so a series posting
    one Game on two dates reads each date's own tip. Flags read only the frame
    and those tips, so reruns reproduce them. The frame is mutated in place and
    returned.
    """
    moon = stories[stories["objective"] == "moon"]
    picks = _pick_sides(moon, commence_by_game)
    on_pick = [
        side == picks[date, game]
        for side, date, game in zip(moon["lead_side"], moon["Date"], moon["Game"], strict=True)
    ]
    top = (
        moon.assign(on_pick=on_pick, led=moon["lead_side"] != "")
        .sort_values(
            ["on_pick", "led", "model_ev", "story_id"], ascending=[False, False, False, True]
        )
        .drop_duplicates(["platform", "Date", "Game"])
    )
    story_key = ["platform", "Date", "story_id"]
    stories["lead"] = pd.MultiIndex.from_frame(stories[story_key]).isin(
        pd.MultiIndex.from_frame(top[story_key])
    )
    return stories


def _pick_sides(
    moon: pd.DataFrame, commence_by_game: Mapping[tuple[str, str], str]
) -> dict[tuple[str, str], str | None]:
    """The side each ``(Date, Game)`` leads on: Over leads fill toward half of each date's games.

    ``moon`` is the Shoot-the-Moon row of every story, and a game is led on a
    side when either platform has a story led on it. A game led on one side
    keeps that side, and a game led on neither maps to ``None``. Games led on
    both take Over until the date's Over leads reach half of its games, and the
    rest take Under. Half counts every game the date has a story for and rounds
    up, and the Over-only games count toward it. Rounded down, a date's lone
    two-sided game (a Thursday or Monday night NFL game) would lead on Under
    every time. The games with the highest ``star`` on an Over-led story take
    Over first, ties by tip from ``commence_by_game`` (untimed ``""`` last),
    then by game key.
    """
    picks: dict[tuple[str, str], str | None] = {}
    for date, day in moon.groupby("Date"):
        sides_by_game = day.groupby("Game")["lead_side"].agg(set).map(lambda sides: sides - {""})
        over_star = day[day["lead_side"] == "Over"].groupby("Game")["star"].max()
        two_sided = sorted(
            (game for game, sides in sides_by_game.items() if len(sides) == 2),
            key=lambda game: (
                -over_star[game],
                commence_by_game[date, game] == "",
                commence_by_game[date, game],
                game,
            ),
        )
        half = (len(sides_by_game) + 1) // 2
        over_only = sum(sides == {"Over"} for sides in sides_by_game)
        for game, sides in sides_by_game.items():
            if len(sides) < 2:
                picks[date, game] = next(iter(sides), None)
        for overs_taken, game in enumerate(two_sided, start=over_only):
            picks[date, game] = "Over" if overs_taken < half else "Under"
    return picks
