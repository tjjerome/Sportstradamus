"""JSON-to-parquet pipeline for Fantasy Points tool responses.

FP Data Suite responses carry row data in one of two shapes:

- v2 ``/values`` endpoints (every discover-generated entry):
  ``content.rows.values``.
- Legacy tool endpoints (hand-imported entries that pre-date v2):
  ``content.table.rows.values``.

Each row is a flat dict with ``gameSeason`` / ``gameWeek`` columns;
column names are camelCase with a source prefix (``playerStats...``,
``teamStats...``, ``opponentStats...``, ``game*``).

This module turns either response into a pandas DataFrame and routes
the parquet output to ``player_data/`` vs ``team_data/`` based on
the spec's context. Routing prefers (in order): the catalog name
prefix (``player_`` / ``team_`` / ``opponent_`` — what
``discover.py`` writes), then the URL path
(``.../tools/{context}/{slug}`` — covers hand-imported entries
from before the prefix convention existed), then the first segment
of ``output_subdir``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pandas as pd

from sportstradamus import data
from sportstradamus.fantasypoints.catalog import EndpointSpec

# Base of the package data tree. ``sportstradamus.data`` is a namespace
# package — its ``importlib.resources.files()`` returns a
# ``MultiplexedPath`` whose ``str()`` is ``MultiplexedPath('/real/path')``
# (literally with the wrapper text), not the path itself. Using that
# directly silently writes parquets into a directory named
# ``MultiplexedPath('...')`` and the user can't find anything.
# ``__path__[0]`` gives the real filesystem path of the package's
# first search location, which is what we want.
_DATA_BASE = Path(data.__path__[0])
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

# Per-mode filename suffix. Empty for ``weekly`` (so existing parquets
# don't have to be renamed) and ``postseason`` (which gets its own
# ``week_19..22`` subfolder — no name collision possible). Only
# ``season_to_date`` keeps an explicit suffix because it shares the
# regular-season ``week_NN`` folder with ``weekly``.
_MODE_SUFFIX = {
    "weekly": "",
    "season_to_date": "_s2d",
    "postseason": "",
}

# NFL regular season is 18 weeks; postseason rounds (wildcard,
# divisional, conference, super bowl) are surfaced as continuation
# weeks 19..22 on disk so the per-week folder structure stays flat.
# Mirrors the ``NFL_REGULAR_SEASON_WEEKS`` constant in
# :mod:`fantasypoints.cli` — duplicated here rather than imported to
# keep ``transform`` free of CLI dependencies.
_REGULAR_SEASON_WEEKS = 18


def parse_table_response(payload: dict) -> pd.DataFrame:
    """Extract the FP response's row list into a DataFrame.

    Reads ``content.rows.values`` (v2 ``/values`` endpoints) or
    ``content.table.rows.values`` (legacy endpoints) — see
    :func:`_extract_rows`. Columns whose cells contain a ``list`` or
    ``dict`` are serialised to JSON strings so the parquet writer
    doesn't have to materialise nested arrow types (which work but
    balloon the file size and make downstream pandas reads slower).

    Args:
        payload: Decoded JSON body returned by any
            ``POST /v2/ds/{league}/tools/{context}/{slug}/values`` call.

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
    mode: str = "weekly",
) -> Path:
    """Compute the on-disk parquet path for one catalog entry's snapshot.

    Layout: ``{base}/{league}/{season}/week_NN/{tool}{suffix}{opp}.parquet`` —
    one subfolder per week so 45 per-tool files stay grouped instead
    of cluttering the season directory. Opponent-context tools keep
    the ``_opp`` suffix in the filename so they don't collide with
    the offensive view of the same tool.

    Mode shapes the path differently depending on whether the mode
    needs to coexist with ``weekly`` in the same folder:

    - ``weekly`` → ``week_NN/{tool}.parquet``.
    - ``season_to_date`` → ``week_NN/{tool}_s2d.parquet`` (shares the
      regular-season folder; suffix prevents collision with weekly).
    - ``postseason`` → ``week_{NN+18}/{tool}.parquet`` (1 → week_19
      wildcard, 2 → week_20 divisional, 3 → week_21 conf champ,
      4 → week_22 super bowl). Postseason rounds live in their own
      folders so the per-week tree stays flat and no filename suffix
      is needed.

    Routing rules (applied in order):

    1. **Name prefix** — ``player_X`` → ``player_data/...``,
       ``team_X`` → ``team_data/...``, ``opponent_X`` →
       ``team_data/.../X_opp...``. Discover-generated entries
       always match this case.
    2. **URL path** — for hand-imported entries that pre-date the
       prefix convention, parse ``/tools/{context}/{slug}`` out of
       ``spec.url`` and route on ``context``.
    3. **output_subdir** — fall back to the first path segment of
       the catalog's ``output_subdir`` field.

    Args:
        spec: Catalog entry.
        season: NFL season year, used as the directory name.
        week: NFL week (1-18 for regular season; 1-4 for postseason
            in ``--mode postseason``, which maps to folder weeks 19-22).
        league: League code (default ``NFL``).
        mode: ``weekly`` (default) | ``season_to_date`` | ``postseason``.
            Drives both the folder number and the filename suffix.

    Returns:
        Absolute path to the parquet file (not yet created).

    Raises:
        ValueError: When none of the three routing sources match a
            known context, or when ``mode`` is not recognised.
    """
    context, tool = _route_spec(spec)
    mode_suffix = _MODE_SUFFIX.get(mode)
    if mode_suffix is None:
        raise ValueError(f"Unknown mode {mode!r}. Use weekly / season_to_date / postseason.")
    folder_week = week + _REGULAR_SEASON_WEEKS if mode == "postseason" else week
    week_dir = f"week_{folder_week:02d}"
    if context == "player":
        base = PLAYER_DATA_BASE / league / str(season) / week_dir
        filename = f"{tool}{mode_suffix}.parquet"
    elif context == "team":
        base = TEAM_DATA_BASE / league / str(season) / week_dir
        filename = f"{tool}{mode_suffix}.parquet"
    elif context == "opponent":
        base = TEAM_DATA_BASE / league / str(season) / week_dir
        filename = f"{tool}_opp{mode_suffix}.parquet"
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
    """Write ``df`` to ``path`` as a parquet file, creating parent dirs as needed.

    Args:
        df: DataFrame to serialise.
        path: Destination path (absolute). Parent directories are created
            automatically if they don't exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _extract_rows(payload: dict) -> list[dict]:
    """Return the row list from an FP response, tolerating both response shapes.

    Two observed shapes for ``rows``:

    - v2 ``/values`` endpoints: ``content.rows.values`` (no ``table``
      wrapper). This is what every discover-generated entry hits.
    - Legacy tool endpoints: ``content.table.rows.values`` — kept for
      hand-imported entries that predate the v2 contract.

    ``rows`` itself is normally ``{count, values: [...]}`` but a few
    legacy endpoints return a bare list — handle both so callers
    don't need to.
    """
    if not isinstance(payload, dict):
        return []
    content = payload.get("content", {})
    if not isinstance(content, dict):
        return []
    rows_node = content.get("rows")
    if rows_node is None and isinstance(content.get("table"), dict):
        rows_node = content["table"].get("rows")
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
