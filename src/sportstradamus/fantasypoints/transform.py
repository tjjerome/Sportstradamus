"""JSON-to-parquet pipeline for Fantasy Points tool responses.

Every observed FP Data Suite response carries its row data at
``content.table.rows.values`` as a list of flat dicts, with each row
including ``gameSeason`` / ``gameWeek`` columns. Columns are
camelCase with a source prefix (``playerStats...``, ``teamStats...``,
``opponentStats...``, ``game*``).

This module turns that response into a pandas DataFrame and routes
the parquet output to ``player_data/`` vs ``team_data/`` based on
which context the catalog entry was generated for
(``player_<tool>`` / ``team_<tool>`` / ``opponent_<tool>``).
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import json
from pathlib import Path

import pandas as pd

from sportstradamus import data

# Base of the package data tree. Matches the layout the existing
# Stats classes use (``data/player_data/{LEAGUE}/{YEAR}/``); we just
# add a sibling ``team_data/`` for team/opponent contexts.
_DATA_BASE = Path(str(pkg_resources.files(data)))
PLAYER_DATA_BASE = _DATA_BASE / "player_data"
TEAM_DATA_BASE = _DATA_BASE / "team_data"

# Catalog entries are named ``{context}_{slug_underscore}`` (see
# discover.py). ``opponent`` rows are defensive aggregates — still
# team-level data — so they land under team_data with an ``_opp``
# suffix to keep them separate from the offensive view of the same
# tool.
_PLAYER_PREFIX = "player_"
_TEAM_PREFIX = "team_"
_OPPONENT_PREFIX = "opponent_"


def parse_table_response(payload: dict) -> pd.DataFrame:
    """Extract ``content.table.rows.values`` from an FP response into a DataFrame.

    Columns whose cells contain a ``list`` or ``dict`` are serialised
    to JSON strings so the parquet writer doesn't have to materialise
    nested arrow types (which work but balloon the file size and make
    downstream pandas reads slower).

    Args:
        payload: Decoded JSON body returned by any
            ``POST /v2/ds/{league}/tools/{context}/{slug}`` call.

    Returns:
        DataFrame with one row per ``values[]`` entry. Returns an
        empty DataFrame if the response has no rows (e.g. an empty
        tool or an unexpected shape).
    """
    rows = _extract_rows(payload)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(rows)
    for col in df.columns:
        if df[col].map(lambda v: isinstance(v, list | dict)).any():
            df[col] = df[col].map(_to_json_string)
    return df


def parquet_path_for_spec(
    spec_name: str,
    *,
    season: int,
    week: int,
    league: str = "NFL",
) -> Path:
    """Compute the on-disk parquet path for one catalog entry's snapshot.

    Routing rules:

    - ``player_<tool>``   → ``player_data/{league}/{season}/<tool>_week_NN.parquet``
    - ``team_<tool>``     → ``team_data/{league}/{season}/<tool>_week_NN.parquet``
    - ``opponent_<tool>`` → ``team_data/{league}/{season}/<tool>_opp_week_NN.parquet``

    Args:
        spec_name: Catalog entry name (``{context}_{slug_underscore}``).
        season: NFL season year, used as the directory name.
        week: NFL week (1-18), zero-padded into the filename.
        league: League code (default ``NFL``). The directory uses the
            upper-case form to match the existing
            ``data/player_data/NFL/`` convention.

    Raises:
        ValueError: When ``spec_name`` doesn't start with a known
            context prefix — discover-generated entries always do;
            hand-imported entries that don't match either prefix
            should be renamed before they're parquetised.
    """
    if spec_name.startswith(_PLAYER_PREFIX):
        tool = spec_name[len(_PLAYER_PREFIX) :]
        base = PLAYER_DATA_BASE / league / str(season)
        filename = f"{tool}_week_{week:02d}.parquet"
    elif spec_name.startswith(_TEAM_PREFIX):
        tool = spec_name[len(_TEAM_PREFIX) :]
        base = TEAM_DATA_BASE / league / str(season)
        filename = f"{tool}_week_{week:02d}.parquet"
    elif spec_name.startswith(_OPPONENT_PREFIX):
        tool = spec_name[len(_OPPONENT_PREFIX) :]
        base = TEAM_DATA_BASE / league / str(season)
        filename = f"{tool}_opp_week_{week:02d}.parquet"
    else:
        raise ValueError(
            f"Catalog name {spec_name!r} doesn't start with "
            f"player_/team_/opponent_ — can't route its parquet."
        )
    return base / filename


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write the DataFrame to ``path``, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _extract_rows(payload: dict) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    table = payload.get("content", {}).get("table", {})
    rows_node = table.get("rows")
    if isinstance(rows_node, dict):
        values = rows_node.get("values", [])
    elif isinstance(rows_node, list):
        values = rows_node
    else:
        values = []
    return [v for v in values if isinstance(v, dict)]


def _to_json_string(value):
    if isinstance(value, list | dict):
        return json.dumps(value, separators=(",", ":"), default=str)
    return value
