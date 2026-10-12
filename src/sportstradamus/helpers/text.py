"""String normalization utilities.

The name-normalization path is load-bearing: sportsbooks spell player
names inconsistently (accents, suffixes, parenthesized team tags) and
``remove_accents`` is the canonical entry point that the archive, the
stats loaders, and the prop-export writer all run names through.
"""

import datetime
import functools
import re
import unicodedata

import statsapi as mlb

from sportstradamus.helpers.config import name_map

# How far ahead to fetch MLB schedules when building the probable-pitcher dict.
MLB_PITCHER_LOOKAHEAD_DAYS = 7

# The MLB feed's detailed game states whose box score will not change again. A game called
# early stays "Completed Early" for good. "Game Over" is left out: its box score is not
# final yet, and the feed lists that game again as "Final".
MLB_FINAL_BOX_SCORE_STATES = frozenset({"Final", "Completed Early"})

# The MLB feed's detailed game states that end a game or call it off; a cause, when one
# is given, follows a colon ("Postponed: Rain"). Every other state is a game still to
# come or under way: "Warmup" and "Delayed Start: Rain" both come before first pitch.
MLB_FINISHED_STATES = MLB_FINAL_BOX_SCORE_STATES | {
    "Game Over",
    "Postponed",
    "Cancelled",
    "Forfeit",
}

# Team abbreviation corrections applied by both DFS scrapers. PDX→POR makes
# Portland's WNBA team match the NBA code (Sleeper uses the airport code PDX). Kept
# for Sleeper's wire format only — abbreviations.json now emits POR, so nothing
# writes PDX into the archive; it used to, and every archived Portland total read
# back as the league default.
ABBR_MAP = {"WSH": "WAS", "GS": "GSW", "PHO": "PHX", "NOP": "NO", "AZ": "ARI", "PDX": "POR"}


def remove_accents(input_str):
    """Normalize a player name to the project's canonical spelling.

    Strips accents, parenthesized substrings, and positional suffixes
    ("Jr", "Sr", "II"-"IV"), then runs the result through ``name_map`` to
    resolve known alternate spellings. Handles the two combo-prop syntaxes
    ("A + B" and "A vs B") by normalizing each side independently.

    Args:
        input_str: The raw name as seen on a sportsbook or in a league feed.

    Returns:
        The canonical name, or ``""`` when ``input_str`` is ``None``.
    """
    if input_str is None:
        return ""
    nfkd_form = unicodedata.normalize("NFKD", input_str)
    out_str = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    out_str = out_str.replace(".", "")
    for substr in [" Jr", " Sr", " II", " III", " IV"]:
        if out_str.endswith(substr):
            out_str = out_str.replace(substr, "")
    # Curly apostrophe -> ASCII so name_map keys match; the literal is data
    # (flagged by RUF001 once that code is un-ratcheted — intentional).
    out_str = out_str.replace("’", "'")
    out_str = re.sub(r"[\(\[].*?[\)\]]", "", out_str).strip().title()
    if "+" in out_str:
        names = out_str.split("+")
        out_str = " + ".join([name_map.get(n.strip(), n.strip()) for n in names])
    elif " vs " in out_str:
        names = out_str.split(" vs ")
        out_str = " vs. ".join([name_map.get(n.strip(), n.strip()) for n in names])
    else:
        out_str = name_map.get(out_str, out_str)
    return out_str


def mlb_game_is_over(status: str) -> bool:
    """Whether the MLB feed's detailed game state says the game is finished or called off."""
    return status.split(":")[0] in MLB_FINISHED_STATES


@functools.cache
def get_mlb_pitchers():
    """Return a ``{team_abbr: pitcher_name}`` dict for the next week of MLB.

    A game counts until it is over, so today's starter stays listed through warm-up.
    Hits mlb-statsapi once per process; subsequent calls return the cache.
    The LA/ANA/ARI/WAS aliases exist because different sportsbooks spell
    those franchises differently from the league feed.
    """
    mlb_games = mlb.schedule(
        start_date=datetime.date.today(),
        end_date=(datetime.date.today() + datetime.timedelta(days=MLB_PITCHER_LOOKAHEAD_DAYS)),
    )
    mlb_teams = mlb.get("teams", {"sportId": 1})
    pitchers = {}
    for game in mlb_games:
        if not mlb_game_is_over(game["status"]):
            abbrs = _resolve_team_abbrs(game, mlb_teams)
            if abbrs is None:
                continue
            away_team, home_team = abbrs
            _assign_probable_pitchers(game, away_team, home_team, pitchers)

    pitchers["LA"] = pitchers.get("LAD", "")
    pitchers["ANA"] = pitchers.get("LAA", "")
    pitchers["ARI"] = pitchers.get("AZ", "")
    pitchers["WAS"] = pitchers.get("WSH", "")
    return pitchers


def _resolve_team_abbrs(game, mlb_teams):
    """Map a game's away/home ids to abbreviations; None if either is ambiguous."""
    away_team = [
        team["abbreviation"] for team in mlb_teams["teams"] if team["id"] == game["away_id"]
    ]
    home_team = [
        team["abbreviation"] for team in mlb_teams["teams"] if team["id"] == game["home_id"]
    ]
    if len(away_team) == 1 and len(home_team) == 1:
        return away_team[0], home_team[0]
    return None


def _assign_probable_pitchers(game, away_team, home_team, pitchers):
    """For game one of a series, record each side's probable starter (first seen wins)."""
    if game["game_num"] != 1:
        return
    if "away_probable_pitcher" in game and away_team not in pitchers:
        pitchers[away_team] = remove_accents(game["away_probable_pitcher"])
    if "home_probable_pitcher" in game and home_team not in pitchers:
        pitchers[home_team] = remove_accents(game["home_probable_pitcher"])
