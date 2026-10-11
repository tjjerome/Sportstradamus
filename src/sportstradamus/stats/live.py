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

    One league's failure must not cost every league its board: a league whose constructor,
    load or update raises sits the run out. It is not scored on the gamelog as cached,
    because the day's schedule arrives only with an update: without it every player reads
    as away and an MLB hitter faces no pitcher, on a board that would look like any other.

    Returns:
        ``(stats, failed)``: the ``Stats`` of the live leagues that updated, by league
        name, and the names of the leagues that raised.
    """
    stats, failed = {}, []
    # Named beside its class: a league whose constructor raises has no instance to name it.
    for league, cls in (
        ("NBA", StatsNBA),
        ("NFL", StatsNFL),
        ("WNBA", StatsWNBA),
        ("MLB", StatsMLB),
        ("NHL", StatsNHL),
    ):
        try:
            struct = cls()
            struct.load()
            if not odds_budget.league_is_live(league, struct.season_start):
                continue
            struct.update()
        except Exception:
            logger.exception(f"{league} update failed, league not scored this run")
            failed.append(league)
        else:
            stats[league] = struct
    return stats, failed
