"""NBA and WNBA season-to-date player tables as of a day, stored per day.

``LeagueDashPlayerBioStats`` (usage, true shooting) and ``LeagueDashPlayerShotLocations``
(attempts and makes by zone) sum a season's games through their ``DateTo``, on stats.nba.com and
stats.wnba.com alike. The comp pool reads them as of the day before its date and the gamelog
update as of yesterday, so a rebuilt row's comps and the served row's read the same tables. Each
day's two payloads are stored raw, gzipped, at
``data/leagues/<league>/season_tables/<season>/<day>.json.gz`` (the shot table repeats its
column names, which parquet refuses): a rebuild asks the league's host for a day once.
``SynergyPlayTypes`` takes no date range, so ``StatsNBA`` reads it to date, undated.
"""

import gzip
import json
from datetime import date

import nba_api.stats.endpoints as nba
import pandas as pd

from sportstradamus.helpers.io import LEAGUES_DIR, _atomic_tmp
from sportstradamus.stats import nba_client

# The scale each league's shot table is read at. The NBA zeroes a zone's points per shot below
# half an attempt a game; the WNBA reads only zone shares, which are scale-free, from totals:
# stats.wnba.com's per-game table misstates the corner threes and its per-40 table every zone.
_SHOT_PER_MODE = {"NBA": "PerGame", "WNBA": "Totals"}


def read(league: str, season: str | int, day: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The season's player bios and shot locations, counting its games through ``day``.

    Args:
        league: ``"NBA"`` or ``"WNBA"``.
        season: The endpoints' season: ``"2025-26"`` for the NBA, the year for the WNBA.
        day: The last day counted; a day before the season's first game reads empty tables.

    Returns:
        ``(bios, shots)``: one row per player each, ``shots`` keeping the endpoint's repeated
        ``FGM`` / ``FGA`` / ``FG_PCT`` columns, one triple per zone.

    Raises:
        NBAStatsError: A fetch failed every retry; nothing is stored.
    """
    path = (
        LEAGUES_DIR / league.lower() / "season_tables" / str(season) / f"{day.isoformat()}.json.gz"
    )
    try:
        with gzip.open(path, "rt") as f:
            tables = json.load(f)
    except FileNotFoundError:
        league_id, host = nba_client.LEAGUE_SOURCES[league]
        date_to = day.strftime("%m/%d/%Y")
        tables = {
            "bios": nba_client.fetch(
                nba.leaguedashplayerbiostats.LeagueDashPlayerBioStats,
                host=host,
                league_id=league_id,
                season=season,
                date_to_nullable=date_to,
            ).get_normalized_dict()["LeagueDashPlayerBioStats"],
            "shots": nba_client.fetch(
                nba.leaguedashplayershotlocations.LeagueDashPlayerShotLocations,
                host=host,
                league_id_nullable=league_id,
                season=season,
                season_type_all_star="Regular Season",
                distance_range="By Zone",
                per_mode_detailed=_SHOT_PER_MODE[league],
                date_to_nullable=date_to,
            ).get_dict()["resultSets"],
        }
        path, tmp = _atomic_tmp(path)
        with gzip.open(tmp, "wt") as f:
            json.dump(tables, f)
        tmp.replace(path)
    shots = tables["shots"]
    return pd.DataFrame(tables["bios"]), pd.DataFrame(
        shots["rowSet"], columns=shots["headers"][1]["columnNames"]
    )
