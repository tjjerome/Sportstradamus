"""Star prominence and the lead story each game's menu headlines.

Ranked on edge alone, a game's menu anchors on its highest-probability leg,
which on DFS x.5 lines is usually a role player's Under. These pure pieces
re-anchor it on the game's stars from snapshot columns alone, no new data:
``attach_prominence`` scores every offer's player, ``lead_seeds`` picks the leg
an Over-led and an Under-led story grow from, and ``assign_leads`` flags the one
story per ``(platform, Game, Date)`` the Games and Tonight tabs headline,
alternating Over and Under down each date's slate.
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

# assign_leads indexes this by date-ordinal parity, so an even date opens on Over.
LEAD_SIDES: tuple[str, str] = ("Over", "Under")


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
    edge: Mapping[int, float], bet_df: Mapping[int, Mapping], star_of: Mapping[int, float]
) -> dict[str, int]:
    """Per side, the strong leg id its lead story grows from.

    ``edge`` maps the game's strong legs to their ``Model EV`` and ``star_of``
    maps them to ``Star``. The most prominent leg wins, then the bigger edge,
    then the alphabetically earlier player; a side with no strong leg is left
    out.
    """
    seeds = {}
    for side in LEAD_SIDES:
        legs = [i for i in edge if bet_df[i]["Bet"] == side]
        if legs:
            seeds[side] = min(legs, key=lambda i: (-star_of[i], -edge[i], bet_df[i]["Player"]))
    return seeds


def assign_leads(
    stories: pd.DataFrame, commence_by_game: Mapping[tuple[str, str], str]
) -> pd.DataFrame:
    """Flag ``lead`` on one story per ``(platform, Game, Date)``, alternating sides.

    Each date walks its games in tip order from ``commence_by_game``, keyed
    ``(Date, Game)`` so a series posting one Game on two dates orders each date
    by its own tip; untimed games (``""``) walk after the timed ones, by key.
    Over is wanted first on an even date ordinal and Under on an odd one. A game
    takes the wanted side when either platform has a story led on it, and only
    then does the turn pass; a game led on one side only takes that side and
    leaves the turn. Each platform flags its story led on the game's side, else
    its story led on the other side, else its best Shoot-the-Moon EV story, on
    both objective rows. Flags read only snapshot keys, the date and which sides
    exist, so reruns reproduce them and both platforms agree whenever both offer
    the side. The frame is mutated in place and returned.
    """
    moon = stories[stories["objective"] == "moon"]
    picks: dict[tuple[str, str], str | None] = {}
    for date, day in moon.groupby("Date"):
        sides_by_game = day.groupby("Game")["lead_side"].agg(set)
        turn = pd.Timestamp(date).toordinal()
        for game in sorted(
            sides_by_game.index,
            key=lambda g: (commence_by_game[date, g] == "", commence_by_game[date, g], g),
        ):
            sides = sides_by_game[game] - {""}
            want = LEAD_SIDES[turn % 2]
            if want in sides:
                picks[date, game] = want
                turn += 1
            else:
                # The wanted side is missing, so at most the other one is led here.
                picks[date, game] = next(iter(sides), None)
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
