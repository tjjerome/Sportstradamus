"""Shared response-parsing and dated-snapshot routing for cumulative sources.

Cleaning the Glass (NBA) and Baseball Savant (MLB) both serve tabular
season-to-date data and write it to the same ``{grain}_data/{league}/{source}/
{season}/{date}/{tool}.parquet`` layout — only the league and source-dir differ.
This module holds that shared knowledge so the two source packages stay thin
wiring. The exact tools + columns are pinned when the endpoints are captured
(see the per-source docs); the parser makes no column assumptions.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from sportstradamus import data
from sportstradamus.collectors.catalog import EndpointSpec

_DATA_BASE = Path(data.__path__[0])
PLAYER_DATA_BASE = _DATA_BASE / "player_data"
TEAM_DATA_BASE = _DATA_BASE / "team_data"

_PLAYER_PREFIX = "player_"
_TEAM_PREFIX = "team_"

# JSON payload keys that have held the row list on the tables checked so far,
# tried in order. CSV / DataFrame bodies skip this path.
_JSON_ROW_KEYS = ("rows", "data", "values", "stats", "leaderboard")


def parse_tabular_response(body: object) -> pd.DataFrame:
    """Turn a collector response into a DataFrame, tolerating four body shapes.

    - a ready DataFrame (``Scrape.get_csv`` already parsed the CSV),
    - CSV / delimited text (a paid export downloaded as text),
    - a bare list of row dicts,
    - a JSON object carrying the rows under one of :data:`_JSON_ROW_KEYS`.

    Returns an empty DataFrame for an empty body or an unrecognised shape so the
    runner records an ``empty`` result rather than crashing the catalog walk.
    """
    if isinstance(body, pd.DataFrame):
        return body
    if isinstance(body, str):
        text = body.strip()
        return pd.read_csv(io.StringIO(text)) if text else pd.DataFrame()
    if isinstance(body, list):
        return pd.DataFrame.from_records([r for r in body if isinstance(r, dict)])
    if isinstance(body, dict):
        for key in _JSON_ROW_KEYS:
            node = body.get(key)
            if isinstance(node, list):
                return pd.DataFrame.from_records([r for r in node if isinstance(r, dict)])
    return pd.DataFrame()


def dated_path_for(
    spec: EndpointSpec, *, season: int, date: str, league: str, source_dir: str
) -> Path:
    """Route one entry to ``{player,team}_data/{league}/{source_dir}/{season}/{date}/{tool}.parquet``.

    Raises:
        ValueError: If the entry name lacks a ``player_`` / ``team_`` prefix —
            captured by the runner as a routing failure rather than silently
            filed under the wrong grain.
    """
    grain, tool = _route_player_team(spec)
    base = PLAYER_DATA_BASE if grain == "player" else TEAM_DATA_BASE
    return base / league / source_dir / str(season) / date / f"{tool}.parquet"


def _route_player_team(spec: EndpointSpec) -> tuple[str, str]:
    if spec.name.startswith(_PLAYER_PREFIX):
        return "player", spec.name[len(_PLAYER_PREFIX) :]
    if spec.name.startswith(_TEAM_PREFIX):
        return "team", spec.name[len(_TEAM_PREFIX) :]
    raise ValueError(
        f"entry {spec.name!r} must be named player_<tool> or team_<tool> so it "
        "routes to player_data / team_data."
    )
