"""JSON-to-parquet pipeline for Fantasy Points tool responses.

Every observed FP Data Suite response carries its row data at
``content.table.rows.values`` as a list of flat dicts, with each row
including ``gameSeason`` / ``gameWeek`` columns. Columns are
camelCase with a source prefix (``playerStats...``, ``teamStats...``,
``opponentStats...``, ``game*``).

This module turns that response into a pandas DataFrame and routes
the parquet output to ``player_data/`` vs ``team_data/`` based on
the spec's context. Routing prefers (in order): the catalog name
prefix (``player_`` / ``team_`` / ``opponent_`` — what
``discover.py`` writes), then the URL path
(``.../tools/{context}/{slug}`` — covers hand-imported entries
from before the prefix convention existed), then the first segment
of ``output_subdir``.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pandas as pd

from sportstradamus import data
from sportstradamus.fantasypoints.catalog import EndpointSpec

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

# Contexts we know how to route. Matches `_KNOWN_CONTEXTS` in
# discover.py — anything else (``other``) routes to team_data with
# the raw context as a filename suffix, since we have no evidence
# of where ``other``-context tools should live.
_CONTEXT_TO_PREFIX = {
    "player": _PLAYER_PREFIX,
    "team": _TEAM_PREFIX,
    "opponent": _OPPONENT_PREFIX,
}

# URL fragment that identifies the routing context on hand-imported
# entries that predate the discover-prefix convention. Pattern:
# ``/v2/ds/{league}/tools/{context}/{slug}`` — captures both.
_TOOL_URL_RE = re.compile(r"/tools/(?P<context>[^/]+)/(?P<slug>[^/?#]+)")


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
    spec: EndpointSpec,
    *,
    season: int,
    week: int,
    league: str = "NFL",
) -> Path:
    """Compute the on-disk parquet path for one catalog entry's snapshot.

    Routing rules (applied in order):

    1. **Name prefix** — ``player_X`` → ``player_data/.../X_week_NN``,
       ``team_X`` → ``team_data/.../X_week_NN``, ``opponent_X`` →
       ``team_data/.../X_opp_week_NN``. Discover-generated entries
       always match this case.
    2. **URL path** — for hand-imported entries that pre-date the
       prefix convention, parse ``/tools/{context}/{slug}`` out of
       ``spec.url`` and route on ``context``.
    3. **output_subdir** — fall back to the first path segment of
       the catalog's ``output_subdir`` field (e.g. ``team/line_matchups``
       → ``team``).

    Args:
        spec: Catalog entry.
        season: NFL season year, used as the directory name.
        week: NFL week (1-18), zero-padded into the filename.
        league: League code (default ``NFL``).

    Raises:
        ValueError: When none of the three routing sources match a
            known context. Rename the spec (recommended:
            ``player_/team_/opponent_<slug>``) and re-run.
    """
    context, tool = _route_spec(spec)
    if context == "player":
        base = PLAYER_DATA_BASE / league / str(season)
        filename = f"{tool}_week_{week:02d}.parquet"
    elif context == "team":
        base = TEAM_DATA_BASE / league / str(season)
        filename = f"{tool}_week_{week:02d}.parquet"
    elif context == "opponent":
        base = TEAM_DATA_BASE / league / str(season)
        filename = f"{tool}_opp_week_{week:02d}.parquet"
    else:
        raise ValueError(
            f"Catalog entry {spec.name!r} (url={spec.url!r}, "
            f"output_subdir={spec.output_subdir!r}) has no recognisable "
            f"context. Rename it player_/team_/opponent_<slug>."
        )
    return base / filename


def _route_spec(spec: EndpointSpec) -> tuple[str, str]:
    """Return ``(context, tool_slug)`` for one spec, trying three routing sources."""
    for prefix, ctx in (
        (_PLAYER_PREFIX, "player"),
        (_TEAM_PREFIX, "team"),
        (_OPPONENT_PREFIX, "opponent"),
    ):
        if spec.name.startswith(prefix):
            return ctx, spec.name[len(prefix) :]
    match = _TOOL_URL_RE.search(urlsplit(spec.url).path)
    if match and match["context"] in _CONTEXT_TO_PREFIX:
        return match["context"], match["slug"].replace("-", "_")
    first_segment = (spec.output_subdir or "").split("/", 1)[0]
    if first_segment in _CONTEXT_TO_PREFIX:
        tool_slug = spec.name.removeprefix(f"{first_segment}_")
        return first_segment, tool_slug
    return "", spec.name


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write the DataFrame to ``path``, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _extract_rows(payload: dict) -> list[dict]:
    """Return the row list from an FP response, tolerating both dict and list shapes.

    FP uses ``rows: {count, values: [...]}`` in most tools but falls back to a
    bare list in a few legacy endpoints — handle both so callers don't need to.
    """
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


def _to_json_string(value: Any) -> Any:
    if isinstance(value, list | dict):
        return json.dumps(value, separators=(",", ":"), default=str)
    return value
