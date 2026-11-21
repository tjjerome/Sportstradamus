"""Per-game skater fields from the NHL stats service's game reports.

Power-play and short-handed ice time, faceoffs taken, and power-play shots and share come
from the league's own reports, not from moneypuck: moneypuck's situation splits hold 5-on-4
and 4-on-5 play only, so time and shots at other strengths fall out of them. On 70 games of
2023-24 to 2026-27, 11 with a 5-on-3, the two disagree on power-play time for 281 of 2,520
skater-games (by up to 170 seconds), short-handed time for 258 and power-play shots for 53.
Faceoffs taken agree, and come from the same reports so that every field has one source.
"""

import json
from datetime import date
from time import sleep

import pandas as pd
from tqdm import tqdm

from sportstradamus.spiderLogger import logger
from sportstradamus.stats.base import scraper

_REPORT_URL = "https://api.nhle.com/stats/rest/en/skater/{report}"
# Each report's fields, by the gamelog column they fill. Each report lists every skater of a
# game it has published. A report is read whole: a filter on one of its fields (ppTimeOnIce>0)
# slowed a request from about a second to 15 (2026-11-20).
_REPORTS = {
    "faceoffwins": {"totalFaceoffs": "faceOffsTaken"},
    "timeonice": {"ppTimeOnIce": "powerPlayTimeOnIce", "shTimeOnIce": "shortHandedTimeOnIce"},
    "powerplay": {"ppShots": "powerPlayShots", "ppTimeOnIcePctPerGame": "powerPlayShare"},
}
REPORT_COLUMNS = [column for fields in _REPORTS.values() for column in fields.values()]
# The service reports seconds; the gamelog keeps ice time in minutes, as timeOnIce.
_SECONDS_COLUMNS = ["powerPlayTimeOnIce", "shortHandedTimeOnIce"]
# The service returns at most 100 rows a request, whatever the limit asks, so a report is
# read page by page, in an order that holds still between pages.
_PAGE_SORT = json.dumps(
    [{"property": "gameId", "direction": "ASC"}, {"property": "playerId", "direction": "ASC"}]
)
# The service answers 429, with a minute's Retry-After, past about 50 requests in 7 seconds;
# 150 requests a quarter-second apart went through clean.
_REQUEST_GAP_S = 0.25
# Games are asked for two weeks at a time: about 100 games, 110 requests.
WINDOW_DAYS = 14


def game_player_key(frame: pd.DataFrame) -> pd.MultiIndex:
    """(gameId, playerId) row key; the stored playerId is object-typed, so both sides cast."""
    return pd.MultiIndex.from_arrays(
        [frame["gameId"].astype(int), pd.to_numeric(frame["playerId"])]
    )


def _window_values(game_ids: list[int], max_attempts: int) -> pd.DataFrame | None:
    """REPORT_COLUMNS of each player every report lists in ``game_ids``, keyed (gameId, playerId).

    ``None`` once the service stops answering.
    """
    ids = ",".join(map(str, game_ids))
    params = {
        "isAggregate": "false",
        "isGame": "true",
        "limit": -1,
        "sort": _PAGE_SORT,
        "cayenneExp": f"gameId in ({ids})",
    }
    frames = []
    for report, fields in _REPORTS.items():
        rows = []
        while True:
            sleep(_REQUEST_GAP_S)
            body = scraper.get(
                _REPORT_URL.format(report=report),
                max_attempts=max_attempts,
                params=params | {"start": len(rows)},
            )
            if not body:
                return None
            rows += body["data"]
            if not body["data"] or len(rows) >= body["total"]:
                break
        frame = pd.DataFrame(rows, columns=["gameId", "playerId", *fields])
        frames.append(frame.set_index(["gameId", "playerId"]).rename(columns=fields))
    # A game one report lists before another waits for a later call rather than read 0.
    values = pd.concat(frames, axis=1, join="inner")
    values[_SECONDS_COLUMNS] /= 60
    return values


def fill_report_columns(
    gamelog: pd.DataFrame,
    seasons: set[int] | None = None,
    since: date | None = None,
    max_attempts: int = 3,
) -> pd.DataFrame:
    """``gamelog`` with REPORT_COLUMNS filled on the rows of every game the service published.

    A published game's rows take its reports' values. Its goalies, whom the skater reports
    leave out, read 0, as does a skater's share of a power play his team never had. Rows of a
    game the service has not published yet keep NaN, as do those of every window after one the
    service failed to answer; a later call fills them, asking only for games still lacking.

    Args:
        gamelog: The NHL gamelog.
        seasons: Season start years to fill, e.g. ``{2025}``; ``None`` is every season.
        since: The earliest game date to fill; ``None`` is every date.
        max_attempts: Tries per request before the fill stops and the rest wait.

    Returns:
        The gamelog, holding every REPORT_COLUMNS column.
    """
    log = gamelog.reindex(columns=gamelog.columns.union(REPORT_COLUMNS, sort=False))
    lacking = log[REPORT_COLUMNS].isna().any(axis=1)
    if seasons is not None:
        # An NHL game id leads with its season's start year: 2025020001 -> 2025.
        lacking &= (log["gameId"] // 1_000_000).isin(seasons)
    if since is not None:
        lacking &= pd.to_datetime(log["gameDate"]).dt.date >= since
    games = log.loc[lacking, ["gameId", "gameDate"]].drop_duplicates("gameId")
    days = pd.to_datetime(games["gameDate"])
    windows = games["gameId"].groupby((days - days.min()).dt.days // WINDOW_DAYS)
    for _, game_ids in tqdm(windows, desc="Getting NHL report stats"):
        values = _window_values(game_ids.tolist(), max_attempts)
        if values is None:
            logger.warning("NHL stats service stopped answering; the unfilled rows wait")
            break
        published = lacking & log["gameId"].isin(values.index.get_level_values("gameId"))
        log.loc[published, REPORT_COLUMNS] = (
            values.reindex(index=game_player_key(log.loc[published]), columns=REPORT_COLUMNS)
            .fillna(0)
            .to_numpy()
        )
    return log
