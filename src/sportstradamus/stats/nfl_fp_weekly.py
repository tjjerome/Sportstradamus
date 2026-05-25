"""FantasyPoints weekly-snapshot parquet loader for the NFL pipeline.

The Phase-1 PFF -> FantasyPoints migration consumed FP *season-aggregate*
CSVs (one per stat-type per season). The hooks here read the
*weekly-snapshot* parquets the operator is downloading -- one snapshot
per (season, snapshot_week), each holding **per-player-per-game** rows
for the games played in that week (NOT season-to-date totals).

Spot-check on the 2022 backfill confirms per-game grain:
``2022/week_01/receiving_basic.parquet`` row for Justin Jefferson shows
``targets=11 receptions=9 yards=184`` (his actual week-1 line) and
``2022/week_02/receiving_basic.parquet`` shows ``targets=12 receptions=6
yards=48`` (his week-2 line). Each row is one player's stat line for one
game. The ``gameWeek`` column matches the snapshot directory week.

Time-capsule semantics under per-game grain
-------------------------------------------

To get "season-to-date through week N" for player-comp inputs or
training features, **concatenate snapshots 1..N** with
:func:`load_through`, then aggregate (sum numerators, sum denominators
for rate stats). For leakage-safe queries on a game in week ``W``, pass
``W - 1`` -- the loader never returns rows from the target game itself.

Use :func:`load_window` when the desired range is not anchored at week 1
(e.g. "last 4 weeks of form" for trend features).

Layout on disk::

    data/player_data/NFL/{season}/
        {season-aggregate}.csv         <-- Phase-1 inputs (legacy)
        week_{NN}/
            efficiency.parquet
            fantasy_points_allowed.parquet  <-- see PLACEHOLDER caveat
            fantasy_points_scored.parquet   <-- see PLACEHOLDER caveat
            ...                             <-- 24 file_kinds total

Most file_kinds publish per-player-per-game rows. Two kinds currently
land as 1-row placeholders showing a single (team, opponent, position)
tile with full-season-looking numbers -- see
:data:`PLACEHOLDER_SINGLE_TILE_KINDS`. Their final semantics will be
confirmed once the FP fetcher in :mod:`sportstradamus.fantasypoints`
finishes parameterizing those endpoints.

This module is read-only. It does NOT wire into ``build_comp_profile``
or ``base_profile`` yet -- that swap happens in the Phase 1.5 / 2 / 3 / 4
plans (see ``.claude/plans/nfl-fp-phase-2-3-4-followups.md``) once the
historical seasons (2022-2024) and the in-flight 2025 season finish
backfilling.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from importlib import resources as pkg_resources
from importlib.resources.abc import Traversable
from pathlib import Path

import pandas as pd

from sportstradamus import data
from sportstradamus.spiderLogger import logger

# Logical name -> on-disk basename. Logical names are the public handle
# every consumer should reference; basenames stay encapsulated here so a
# future FP schema rename doesn't ripple through call sites.
FILE_KINDS: dict[str, str] = {
    "efficiency": "efficiency.parquet",
    "fantasy_points_allowed": "fantasy_points_allowed.parquet",
    "fantasy_points_scored": "fantasy_points_scored.parquet",
    "fpts_scored_report": "fpts_scored_report.parquet",
    "offense_snap_share_report": "offense_snap_share_report.parquet",
    "offense_snaps": "offense_snaps.parquet",
    "passing_advanced": "passing_advanced.parquet",
    "passing_basic": "passing_basic.parquet",
    "passing_depth": "passing_depth.parquet",
    "qb_coverage_matchup": "qb_coverage_matchup.parquet",
    "receiving_advanced": "receiving_advanced.parquet",
    "receiving_basic": "receiving_basic.parquet",
    "receiving_man_vs_zone": "receiving_man_vs_zone.parquet",
    "receiving_route_share_report": "receiving_route_share_report.parquet",
    "receiving_routes_run": "receiving_routes_run.parquet",
    "receiving_separation_by_alignment": "receiving_separation_by_alignment.parquet",
    "receiving_separation_by_breaks": "receiving_separation_by_breaks.parquet",
    "receiving_separation_by_coverage": "receiving_separation_by_coverage.parquet",
    "receiving_separation_by_routes": "receiving_separation_by_routes.parquet",
    "receiving_target_share_report": "receiving_target_share_report.parquet",
    "rushing_advanced": "rushing_advanced.parquet",
    "rushing_basic": "rushing_basic.parquet",
    "rushing_bell_cow": "rushing_bell_cow.parquet",
    "wr_coverage_matchup": "wr_coverage_matchup.parquet",
}

# File_kinds whose current parquets contain a single placeholder row
# (a tile pull rather than the full per-game enumeration). Aggregating
# these through :func:`load_window` will produce a useless result until
# the fetcher's endpoint parameterization is finalized. Consumers should
# special-case or skip these for now; the constant lets a downstream
# auditor flag the anomaly without re-discovering it.
PLACEHOLDER_SINGLE_TILE_KINDS: frozenset[str] = frozenset(
    {"fantasy_points_allowed", "fantasy_points_scored"}
)

# Snapshot directory naming convention. The folder ``week_{NN}`` holds
# per-game rows for week NN of the parent season; the data inside is
# NOT season-to-date. Two-digit zero-padded; week numbers run 1..18 in
# regular season (playoff snapshots, if/when they appear, extend this).
_WEEK_DIR_RE = re.compile(r"^week_(\d{2})$")
_WEEK_DIR_FMT = "week_{week:02d}"
_NFL_DIR = "player_data/NFL"


def snapshot_dir(season: int, snapshot_week: int) -> Traversable | None:
    """Return the snapshot directory for ``(season, snapshot_week)`` or None.

    Args:
        season: NFL season year (e.g. 2024).
        snapshot_week: Zero-padded week number within that season (1–18+).

    Returns:
        An importlib-resources path object pointing at the
        ``week_{NN}/`` directory, or ``None`` if the directory is absent.
        ``None`` is the expected interim state for weeks that haven't
        been downloaded yet.
    """
    week_dir = _WEEK_DIR_FMT.format(week=snapshot_week)
    directory = pkg_resources.files(data) / f"{_NFL_DIR}/{season}/{week_dir}"
    return directory if os.path.exists(directory) else None


def available_snapshots(season: int) -> list[int]:
    """Return sorted list of snapshot weeks present on disk for ``season``.

    Args:
        season: NFL season year (e.g. 2024).

    Returns:
        Sorted list of integer week numbers whose ``week_{NN}/``
        directories exist under the season root. Empty if none.
    """
    season_dir = pkg_resources.files(data) / f"{_NFL_DIR}/{season}"
    if not os.path.exists(season_dir):
        return []

    weeks: list[int] = []
    for entry in Path(str(season_dir)).iterdir():
        match = _WEEK_DIR_RE.match(entry.name)
        if match:
            weeks.append(int(match.group(1)))
    weeks.sort()
    return weeks


def load_snapshot(
    season: int,
    snapshot_week: int,
    file_kind: str,
) -> pd.DataFrame | None:
    """Load one parquet from the ``(season, snapshot_week)`` directory.

    Returns the per-player-per-game rows for the games played in
    ``snapshot_week`` of ``season``. To stitch multiple weeks together
    into a season-to-date view, use :func:`load_through` or
    :func:`load_window` instead.

    Args:
        season: NFL season year (e.g. 2024).
        snapshot_week: Exact snapshot week number to load.
        file_kind: Logical kind name from :data:`FILE_KINDS`
            (e.g. ``"receiving_basic"``).

    Returns:
        The parquet contents as a DataFrame, or ``None`` if the snapshot
        directory or the requested file is absent, or if the parquet
        cannot be parsed. Returns an empty DataFrame when the parquet
        exists but holds no rows (FP sometimes writes empty files
        during in-flight weeks).

    Raises:
        ValueError: If ``file_kind`` is not a key in :data:`FILE_KINDS`.
    """
    if file_kind not in FILE_KINDS:
        raise ValueError(
            f"unknown FP weekly file_kind: {file_kind!r}; valid kinds: {sorted(FILE_KINDS)}"
        )

    directory = snapshot_dir(season, snapshot_week)
    if directory is None:
        return None

    path = directory / FILE_KINDS[file_kind]
    if not os.path.exists(path):
        return None

    try:
        return pd.read_parquet(path)
    except Exception as exc:
        logger.warning(
            "failed to read %s for season=%s week=%s: %s",
            file_kind,
            season,
            snapshot_week,
            exc,
        )
        return None


def load_window(
    season: int,
    start_week: int,
    end_week: int,
    file_kind: str,
) -> pd.DataFrame | None:
    """Concatenate per-game snapshots over ``[start_week, end_week]`` inclusive.

    The returned DataFrame stacks per-player-per-game rows from every
    snapshot whose week is in the closed interval. Snapshots that are
    missing or empty are silently skipped; the function returns
    ``None`` only when no snapshot in the window produced any rows.

    The caller is responsible for aggregation. For rate stats, the
    correct pattern is ``sum(numerator) / sum(denominator)`` across the
    concatenated rows -- see ``.claude/plans/nfl-fp-phase-2-3-4-
    followups.md`` for the per-kind denominator recommendations.

    Args:
        season: NFL season year (e.g. 2024).
        start_week: Inclusive lower bound on snapshot week.
        end_week: Inclusive upper bound on snapshot week.
        file_kind: Logical kind name from :data:`FILE_KINDS`.

    Returns:
        Concatenated DataFrame of every available snapshot in the
        window, or ``None`` if the window resolved to zero non-empty
        snapshots.

    Raises:
        ValueError: If ``file_kind`` is not a key in :data:`FILE_KINDS`,
            or if ``start_week > end_week``.
    """
    if file_kind not in FILE_KINDS:
        raise ValueError(
            f"unknown FP weekly file_kind: {file_kind!r}; valid kinds: {sorted(FILE_KINDS)}"
        )
    if start_week > end_week:
        raise ValueError(
            f"start_week={start_week} must be <= end_week={end_week}"
        )

    frames: list[pd.DataFrame] = []
    for week in available_snapshots(season):
        if not start_week <= week <= end_week:
            continue
        df = load_snapshot(season, week, file_kind)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_through(
    season: int,
    last_week: int,
    file_kind: str,
) -> pd.DataFrame | None:
    """Concatenate snapshots 1..``last_week`` inclusive -- season-to-date helper.

    Thin convenience wrapper over :func:`load_window` anchored at week 1.
    The leakage-safe time-capsule pattern for a game in week ``W`` is::

        load_through(season, W - 1, kind)

    so the target game itself is excluded from the lookback.

    Args:
        season: NFL season year (e.g. 2024).
        last_week: Inclusive upper bound on snapshot week. Pass
            ``target_game_week - 1`` for leakage-safe lookbacks.
        file_kind: Logical kind name from :data:`FILE_KINDS`.

    Returns:
        Concatenated DataFrame of every available snapshot from week 1
        through ``last_week``, or ``None`` if no snapshots in the range
        produced any rows.

    Raises:
        ValueError: If ``file_kind`` is not a key in :data:`FILE_KINDS`,
            or if ``last_week < 1``.
    """
    if last_week < 1:
        raise ValueError(f"last_week={last_week} must be >= 1")
    return load_window(season, 1, last_week, file_kind)


def load_all_snapshots(
    season: int,
    file_kind: str,
) -> dict[int, pd.DataFrame]:
    """Load every available snapshot of ``file_kind`` for ``season``.

    Args:
        season: NFL season year (e.g. 2024).
        file_kind: Logical kind name from :data:`FILE_KINDS`.

    Returns:
        Mapping of snapshot week → DataFrame for every week that is
        present on disk and parses without error. Production paths use
        :func:`load_window` / :func:`load_through`; this returns the
        snapshots un-concatenated for diagnostics and snapshot-drift
        analysis.
    """
    out: dict[int, pd.DataFrame] = {}
    for week in available_snapshots(season):
        df = load_snapshot(season, week, file_kind)
        if df is not None and not df.empty:
            out[week] = df
    return out


def snapshot_inventory(seasons: Iterable[int] | None = None) -> pd.DataFrame:
    """Return a DataFrame inventorying which snapshots+files exist on disk.

    Used by operator tools (and future tests) to confirm a download is
    complete before swapping production over.

    Args:
        seasons: Iterable of season years to include, or ``None`` to scan
            every numeric subdirectory under the NFL player-data root.

    Returns:
        DataFrame with columns ``["season", "snapshot_week",
        *FILE_KINDS]``. Boolean kind columns are ``True`` when the
        corresponding parquet file exists in that snapshot directory.
        Returns an empty DataFrame (correct columns, zero rows) when no
        snapshots are found.
    """
    if seasons is None:
        nfl_dir = pkg_resources.files(data) / _NFL_DIR
        if not os.path.exists(nfl_dir):
            return pd.DataFrame(columns=["season", "snapshot_week", *FILE_KINDS])
        season_entries = sorted(
            int(entry.name) for entry in Path(str(nfl_dir)).iterdir() if entry.name.isdigit()
        )
    else:
        season_entries = sorted(seasons)

    rows: list[dict[str, object]] = []
    for season in season_entries:
        for week in available_snapshots(season):
            directory = snapshot_dir(season, week)
            if directory is None:
                continue
            row: dict[str, object] = {"season": season, "snapshot_week": week}
            for kind, basename in FILE_KINDS.items():
                row[kind] = os.path.exists(directory / basename)
            rows.append(row)

    return pd.DataFrame(rows, columns=["season", "snapshot_week", *FILE_KINDS])
