"""String normalization and small collection utilities.

The name-normalization path is load-bearing: sportsbooks spell player
names inconsistently (accents, suffixes, parenthesized team tags) and
``remove_accents`` is the canonical entry point that the archive, the
stats loaders, and the prop-export writer all run names through.
"""

import datetime
import re
import unicodedata

import numpy as np
import pandas as pd
import statsapi as mlb

from sportstradamus.helpers.config import name_map


def remove_accents(input_str):
    """Normalize a player name to the project's canonical spelling.

    Strips accents, parenthesized substrings, and positional suffixes
    ("Jr", "Sr", "II"–"IV"), then runs the result through ``name_map`` to
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


def merge_dict(a, b, path=None):
    """Recursively merge ``b`` into ``a``, in place. Returns ``a``.

    The archive uses a few keys (``Line``, ``EV``) with domain-specific
    merge semantics rather than the default overwrite — ``Line`` lists
    extend, ``EV`` dicts update, everything else gets overwritten.
    """
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dict(a[key], b[key], [*path, str(key)])
            elif np.all(a[key] == b[key]):
                pass  # same leaf value
            elif key == "Line":
                a[key].extend(b[key])
            elif key == "EV":
                a[key].update(b[key])
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a


def get_trends(x):
    """Return per-column slope of the trailing 5 rows of ``x`` as a Series."""
    if len(x) < 3:
        trend = np.zeros(len(x.columns))
    else:
        trend = np.polyfit(np.arange(0, len(x.tail(5))), x.tail(5), 1)[0]
    return pd.Series(trend, index=x.columns)


def hmean(items):
    """Harmonic mean of numeric ``items``, ignoring zeros and strings.

    Returns ``0`` when no finite non-zero values are present (callers rely
    on the zero sentinel rather than a raised exception).
    """
    total = 0
    count = 0
    for i in items:
        if (i != 0) and type(i) is not str:
            count += 1
            total += 1 / i

    if total != 0:
        return count / total
    else:
        return 0


_mlb_pitchers_cache = None


def get_mlb_pitchers():
    """Return a ``{team_abbr: pitcher_name}`` dict for the next week of MLB.

    Hits mlb-statsapi once per process; subsequent calls return the cache.
    The LA/ANA/ARI/WAS aliases exist because different sportsbooks spell
    those franchises differently from the league feed.
    """
    global _mlb_pitchers_cache
    if _mlb_pitchers_cache is not None:
        return _mlb_pitchers_cache

    mlb_games = mlb.schedule(
        start_date=datetime.date.today(),
        end_date=(datetime.date.today() + datetime.timedelta(days=7)),
    )
    mlb_teams = mlb.get("teams", {"sportId": 1})
    pitchers = {}
    for game in mlb_games:
        if game["status"] in ["Pre-Game", "Scheduled"]:
            awayTeam = [
                team["abbreviation"] for team in mlb_teams["teams"] if team["id"] == game["away_id"]
            ]
            homeTeam = [
                team["abbreviation"] for team in mlb_teams["teams"] if team["id"] == game["home_id"]
            ]
            if len(awayTeam) == 1 and len(homeTeam) == 1:
                awayTeam = awayTeam[0]
                homeTeam = homeTeam[0]
            else:
                continue
            if game["game_num"] == 1:
                if "away_probable_pitcher" in game and awayTeam not in pitchers:
                    pitchers[awayTeam] = remove_accents(game["away_probable_pitcher"])
                if "home_probable_pitcher" in game and homeTeam not in pitchers:
                    pitchers[homeTeam] = remove_accents(game["home_probable_pitcher"])

    pitchers["LA"] = pitchers.get("LAD", "")
    pitchers["ANA"] = pitchers.get("LAA", "")
    pitchers["ARI"] = pitchers.get("AZ", "")
    pitchers["WAS"] = pitchers.get("WSH", "")
    _mlb_pitchers_cache = pitchers
    return pitchers
