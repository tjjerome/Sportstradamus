"""JSON-to-parquet pipeline for Fantasy Points tool responses.

The API answers each tool with a bare JSON array of flat, snake_case row
dicts. Every row also carries a ``__raw`` sub-object holding the counts
behind the displayed rates — the numerators and denominators the
aggregation recipes need in order to pool weeks correctly.

Turning one response into the frame the stats layer expects is three steps,
all so that layer never learns the upstream schema changed: ``__raw`` is
flattened up into the row, the identity columns are synthesised from the
API's own ids, and every column the recipes read is copied to its legacy
camelCase name by :mod:`collectors.fantasypoints.column_map`.

Parquet output routes to ``player_data/`` vs ``team_data/`` from the
catalog name prefix (``player_`` / ``team_`` / ``opponent_``), falling
back to the first segment of ``output_subdir``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from sportstradamus import data
from sportstradamus.collectors.catalog import EndpointSpec
from sportstradamus.collectors.fantasypoints.column_map import apply_column_map

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

# Contexts we know how to route. Anything else routes to team_data with
# the raw context as a filename suffix, since we have no evidence of where
# such tools should live.
_CONTEXT_TO_PREFIX = {
    "player": _PLAYER_PREFIX,
    "team": _TEAM_PREFIX,
    "opponent": _OPPONENT_PREFIX,
}

# Per-file-kind new -> legacy column translation, keyed the same way the
# stats layer's FILE_KINDS are.
# Sub-object on every new-API row holding the counts behind the rates.
_RAW_KEY = "__raw"

# Identity columns the stats layer keys on. The new API publishes one
# 3-letter team code per row and a GSIS-style player id under a per-tool
# name, so both team id columns are filled with the abbreviation itself:
# ``_build_team_abbreviation_map`` then resolves to an identity mapping and
# the recipes' groupby key keeps working unchanged.
_NEW_PLAYER_ID_COLS = ("passer_id", "rusher_id", "receiver_id", "player_id", "gsis_id")
_PLAYER_ID_COL = "playerPlayerId"
_TEAM_ID_COL = "teamTeamId"
_TEAM_ABBR_COL = "teamAbbreviation"

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
# :mod:`collectors.fantasypoints.source` — duplicated here rather than
# imported to keep ``transform`` free of source/CLI dependencies.
_REGULAR_SEASON_WEEKS = 18


def parse_table_response(
    payload: object,
    *,
    spec: EndpointSpec | None = None,
    season: int | None = None,
    week: int | None = None,
) -> pd.DataFrame:
    """Turn one tool response into the DataFrame the stats layer expects.

    Flattens each row's ``__raw`` counts up alongside the displayed rates,
    synthesises the identity columns, then copies every mapped column to
    its legacy name (see the module docstring). Columns whose cells hold a
    ``list`` or ``dict`` are serialised to JSON strings so the parquet
    writer doesn't materialise nested arrow types, which work but balloon
    the file and slow downstream reads.

    Args:
        payload: Decoded JSON body — a bare list of row dicts.
        spec: Catalog entry that produced the body. Without it the frame
            is returned untranslated, which is what the ad-hoc
            ``import-curl`` preview wants.
        season: NFL season the request covered. Stamped onto every row as
            ``gameSeason``; a response never echoes its own filters, and the
            stats layer's metadata pass drops a whole kind that lacks it.
        week: NFL week the request covered, stamped as ``gameWeek``.

    Returns:
        DataFrame with one row per response entry, empty when the response
        carries no rows.
    """
    rows = _extract_rows(payload)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame.from_records([_flatten_raw(row) for row in rows])
    df = _add_identity_columns(df)
    if season is not None:
        df["gameSeason"] = season
    if week is not None:
        df["gameWeek"] = week
    if spec is not None:
        df = apply_column_map(df, *_route_spec(spec))
    for col in df.columns:
        if df[col].map(lambda v: isinstance(v, list | dict)).any():
            df[col] = df[col].map(_to_json_string)
    return df


def _flatten_raw(row: dict) -> dict:
    """Lift a row's ``__raw`` counts up beside its displayed rates.

    Displayed values win on a name collision: where both carry ``games`` or
    ``dropbacks`` they agree, and the displayed one is what the legacy
    column was matched against.
    """
    raw = row.get(_RAW_KEY)
    flat = {k: v for k, v in row.items() if k != _RAW_KEY}
    if isinstance(raw, dict):
        flat = {**{k: v for k, v in raw.items() if k not in flat}, **flat}
    return flat


def _add_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Synthesise the identity columns the stats layer groups and joins on.

    ``playerPlayerId`` only ever serves as a groupby key before the result
    is re-keyed to the player's name, so the API's GSIS-style id is a valid
    substitute. Both team id columns get the 3-letter abbreviation, which
    makes ``_build_team_abbreviation_map`` an identity mapping.
    """
    if "team" in df.columns:
        df[_TEAM_ID_COL] = df["team"]
        df[_TEAM_ABBR_COL] = df["team"]
    id_col = next((c for c in _NEW_PLAYER_ID_COLS if c in df.columns), None)
    if id_col is not None:
        df[_PLAYER_ID_COL] = df[id_col]
    if "name" in df.columns:
        names = df["name"].astype(str).str.split(n=1)
        df["playerFirstName"] = names.str[0]
        df["playerLastName"] = names.str[1].fillna("")
    if "position" in df.columns:
        df["playerPosition"] = df["position"]
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
       ``team_data/.../X_opp...``.
    2. **output_subdir** — fall back to the first path segment of
       the catalog's ``output_subdir`` field, for hand-imported
       entries that pre-date the prefix convention.

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
    """Return ``(context, tool_slug)`` for one spec, preferring the name prefix."""
    for prefix, ctx in (
        (_PLAYER_PREFIX, "player"),
        (_TEAM_PREFIX, "team"),
        (_OPPONENT_PREFIX, "opponent"),
    ):
        if spec.name.startswith(prefix):
            return ctx, spec.name[len(prefix) :]
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


def _extract_rows(payload: object) -> list[dict]:
    """Return the row list from a tool response.

    The API answers with a bare JSON array. An error page or an expired
    session yields some other shape, which becomes zero rows here and an
    ``empty`` run result with a response preview attached.
    """
    if isinstance(payload, list):
        return [v for v in payload if isinstance(v, dict)]
    return []


def _to_json_string(value: Any) -> Any:
    if isinstance(value, list | dict):
        return json.dumps(value, separators=(",", ":"), default=str)
    return value
