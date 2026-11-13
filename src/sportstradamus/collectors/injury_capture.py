"""Per-run capture of every league's injury report, read from ESPN.

ESPN's injury feeds (``site.api.espn.com``, keyless) list each team's injured players with a
status, the injury and an expected return. They hold the current report only, no history, so
the record starts with the first capture. Each ``prophecize`` run stores one row per listed
player at ``INJURY_CAPTURE_DIR/<league>/date=YYYY-MM-DD/HHMMSS.parquet``, named by the run's
``Scored At`` (UTC). Every league every run, in season or not: a preseason injury still decides
who plays on opening day.

A row the feed marks Active is dropped: in the NFL feed those are news items, not injuries. Ids
and names are ESPN's, not the league's; the model build that reads these files owns the name
resolver.
"""

from __future__ import annotations

import re
from functools import partial

import pandas as pd
import requests

from sportstradamus.collectors.capture import capture_league
from sportstradamus.helpers.io import INJURY_CAPTURE_DIR
from sportstradamus.helpers.scraping import CAPTURE_TIMEOUT_S

INJURY_FEEDS = {
    "NFL": "football/nfl",
    "NBA": "basketball/nba",
    "WNBA": "basketball/wnba",
    "NHL": "hockey/nhl",
    "MLB": "baseball/mlb",
}
INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/{feed}/injuries"
# Column order and dtype, pinned so a league's files read together: an all-null column left to
# pyarrow is typed ``null``.
INJURY_DTYPES = {
    "polled_at": "datetime64[ns]",
    "team": "string",
    "athlete_id": "Int64",
    "name": "string",
    "position": "string",
    "status": "string",
    "type": "string",
    "updated": "datetime64[ns]",
    "comment": "string",
    "body_part": "string",
    "side": "string",
    "return_date": "string",
    "fantasy_status": "string",
}
# The athlete object carries no id; its www.espn.com player links do. No slash is required
# after the digits: the NHL and MLB player card ends at the id.
_ATHLETE_ID = re.compile(r"/id/(\d+)")


def _athlete_id(links: list[dict]) -> int | None:
    matches = (_ATHLETE_ID.search(link["href"]) for link in links)
    return next((int(match[1]) for match in matches if match), None)


def injury_rows(payload: dict, polled_at: pd.Timestamp) -> pd.DataFrame:
    """One row per listed player but the Active ones; a field the feed leaves out is NA.

    ``updated`` is the row's own ``date``, naive UTC.
    """
    listed = (
        (injury, injury["athlete"], injury.get("details", {}))
        for team in payload["injuries"]
        for injury in team["injuries"]
        if injury["status"] != "Active"
    )
    rows = [
        {
            "polled_at": polled_at,
            "team": athlete["team"]["abbreviation"],
            "athlete_id": _athlete_id(athlete["links"]),
            "name": athlete["displayName"],
            "position": athlete["position"]["abbreviation"],
            # The row's status, not the athlete's: most Out rows' athlete status reads Active.
            "status": injury["status"],
            "type": injury["type"]["name"],
            "updated": pd.Timestamp(injury["date"]).tz_convert(None),
            "comment": injury["shortComment"],
            "body_part": details.get("type"),
            "side": details.get("side"),
            "return_date": details.get("returnDate"),
            "fantasy_status": details.get("fantasyStatus", {}).get("abbreviation"),
        }
        for injury, athlete, details in listed
    ]
    return pd.DataFrame(rows, columns=list(INJURY_DTYPES)).astype(INJURY_DTYPES)


def _fetch_injuries(feed: str, polled_at: pd.Timestamp) -> pd.DataFrame:
    response = requests.get(INJURIES_URL.format(feed=feed), timeout=CAPTURE_TIMEOUT_S)
    response.raise_for_status()
    return injury_rows(response.json(), polled_at)


def capture_injuries(scored_at: pd.Timestamp) -> None:
    """Write every league's injury report, one parquet per league.

    A league whose feed lists nobody writes nothing.
    """
    for league, feed in INJURY_FEEDS.items():
        build = partial(_fetch_injuries, feed, scored_at)
        capture_league(league, "injury", build, INJURY_CAPTURE_DIR, scored_at)
