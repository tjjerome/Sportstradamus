"""NBA season-to-date player tables as of a day, stored per day.

``LeagueDashPlayerBioStats`` (usage, true shooting) and ``LeagueDashPlayerShotLocations``
(attempts and makes by zone) sum a season's games through their ``DateTo``. The comp pool reads
them as of the day before its date and the gamelog update as of yesterday, so a rebuilt row's
comps and the served row's read the same tables. Each day's two payloads are stored raw,
gzipped, at ``data/leagues/nba/season_tables/<season>/<day>.json.gz`` (the shot table repeats
its column names, which parquet refuses): a rebuild asks stats.nba.com for a day once.
``SynergyPlayTypes`` takes no date range, so ``StatsNBA`` reads it to date, undated.
"""

import gzip
import json
from datetime import date

import nba_api.stats.endpoints as nba
import pandas as pd

from sportstradamus.helpers.io import LEAGUES_DIR, _atomic_tmp
from sportstradamus.stats import nba_client

SEASON_TABLES_DIR = LEAGUES_DIR / "nba" / "season_tables"


def read(season: str, day: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The season's player bios and shot locations, counting its games through ``day``.

    Args:
        season: The endpoints' season, ``"2025-26"``.
        day: The last day counted; a day before the season's first game reads empty tables.

    Returns:
        ``(bios, shots)``: one row per player each, ``shots`` keeping the endpoint's repeated
        ``FGM`` / ``FGA`` / ``FG_PCT`` columns, one triple per zone.

    Raises:
        NBAStatsError: A fetch failed every retry; nothing is stored.
    """
    path = SEASON_TABLES_DIR / season / f"{day.isoformat()}.json.gz"
    try:
        with gzip.open(path, "rt") as f:
            tables = json.load(f)
    except FileNotFoundError:
        date_to = day.strftime("%m/%d/%Y")
        tables = {
            "bios": nba_client.fetch(
                nba.leaguedashplayerbiostats.LeagueDashPlayerBioStats,
                season=season,
                date_to_nullable=date_to,
            ).get_normalized_dict()["LeagueDashPlayerBioStats"],
            "shots": nba_client.fetch(
                nba.leaguedashplayershotlocations.LeagueDashPlayerShotLocations,
                season=season,
                season_type_all_star="Regular Season",
                distance_range="By Zone",
                per_mode_detailed="PerGame",
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
