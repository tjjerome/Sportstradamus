"""The live leagues' ``Stats``, loaded and updated for a scoring run."""

from sportstradamus.helpers import odds_budget
from sportstradamus.spiderLogger import logger
from sportstradamus.stats.base import Stats
from sportstradamus.stats.mlb import StatsMLB
from sportstradamus.stats.nba import StatsNBA
from sportstradamus.stats.nfl import StatsNFL
from sportstradamus.stats.nhl import StatsNHL
from sportstradamus.stats.wnba import StatsWNBA


def load_live_leagues() -> tuple[dict[str, Stats], list[str]]:
    """Load every live league's ``Stats`` and update its gamelog from the league API.

    One league's bad update must not cost every league its board: a league whose update
    raises sits the run out. It is not scored on the gamelog as cached, because the day's
    schedule arrives only with an update: without it every player reads as away and an
    MLB hitter faces no pitcher, on a board that would look like any other.

    Returns:
        ``(stats, failed)``: the ``Stats`` of the live leagues that updated, by league
        name, and the names of the live leagues whose update raised.
    """
    stats, failed = {}, []
    for cls in (StatsNBA, StatsNFL, StatsWNBA, StatsMLB, StatsNHL):
        struct = cls()
        struct.load()
        if not odds_budget.league_is_live(struct.league, struct.season_start):
            continue
        try:
            struct.update()
        except Exception:
            logger.exception(f"{struct.league} update failed, league not scored this run")
            failed.append(struct.league)
        else:
            stats[struct.league] = struct
    return stats, failed
