"""FantasyPoints weekly-snapshot parquet loader for the NFL pipeline.

The Phase-1 PFF -> FantasyPoints migration consumed FP *season-aggregate*
CSVs (one per stat-type per season). The hooks here read the
*weekly-snapshot* parquets that the operator is in the process of
downloading -- one snapshot per (season, snapshot_week), each holding
season-to-date player rows as FP knew them at the end of that snapshot
week. With these snapshots we can build true time-capsule features
(season-to-date as-of-game-date) and close the comp-candidate-pool
leakage flagged in Phase 1.

Layout on disk (post-Phase-1K canonical slot):

::

    data/player_data/NFL/{season}/
        {season-aggregate}.csv         <-- Phase-1 inputs (legacy)
        week_{NN}/
            efficiency.parquet
            fantasy_points_allowed.parquet
            fantasy_points_scored.parquet
            ...                        <-- 24 file_kinds total

This module is read-only. It does NOT wire into ``build_comp_profile`` or
``base_profile`` yet -- that swap happens in the Phase 1.5 / 2 / 3 / 4
plans (see ``.claude/plans/nfl-fp-phase-2-3-4-followups.md``) once the
2025 weekly download is complete and the historical seasons (2022-2024)
are backfilled with their per-season per-snapshot data. Until then the
hooks return whatever's present so downstream callers can be developed
against partial fixtures.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from importlib import resources as pkg_resources

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

# Logical kinds that key on team rather than player. Defense-allowed
# fantasy points are aggregated per (opponent, position) tile, not per
# attacking player. Consumers should join these on ``opponentTeamId``
# rather than ``playerPlayerId``.
TEAM_LEVEL_KINDS: frozenset[str] = frozenset({"fantasy_points_allowed"})

# Snapshot directory naming convention. The folder ``week_{NN}`` holds the
# FP database state at the end of week NN of the parent season's reporting
# calendar. Two-digit zero-padded; week numbers run 1..18 in regular
# season (playoff snapshots, if/when they appear, would extend this).
_WEEK_DIR_RE = re.compile(r"^week_(\d{2})$")
_WEEK_DIR_FMT = "week_{week:02d}"
_NFL_DIR = "player_data/NFL"


def snapshot_dir(season: int, snapshot_week: int) -> object | None:
    """Return the snapshot directory for ``(season, snapshot_week)`` or None.

    Returns the importlib-resources path object (so the caller can keep
    using ``/`` joins) when the directory exists; returns ``None`` if the
    snapshot has not been downloaded yet. ``None`` is the expected
    interim state for in-flight seasons.

    Args:
        season: NFL season year (e.g. 2024).
        snapshot_week: Zero-padded week number within that season (1–18+).

    Returns:
        An importlib-resources path object pointing at the
        ``week_{NN}/`` directory, or ``None`` if the directory is absent.
    """
    week_dir = _WEEK_DIR_FMT.format(week=snapshot_week)
    directory = pkg_resources.files(data) / f"{_NFL_DIR}/{season}/{week_dir}"
    return directory if os.path.exists(directory) else None


def available_snapshots(season: int) -> list[int]:
    """Return sorted list of snapshot weeks present on disk for ``season``.

    Empty list means no weekly snapshots have been downloaded for this
    season; callers should fall back to the season-aggregate CSVs at the
    season's top-level directory.

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
    for entry in os.listdir(season_dir):
        match = _WEEK_DIR_RE.match(entry)
        if match:
            weeks.append(int(match.group(1)))
    weeks.sort()
    return weeks


def resolve_asof_snapshot(season: int, target_week: int) -> int | None:
    """Return the largest available snapshot week N with N <= ``target_week``.

    This is the time-capsule resolver: a feature being built for a game
    in week ``target_week`` of ``season`` should consult the snapshot
    returned here, never anything more recent. Returns ``None`` when no
    snapshot at or before ``target_week`` exists for this season.

    Args:
        season: NFL season year (e.g. 2024).
        target_week: The game week for which features are being built.
            The resolver never returns a snapshot newer than this.

    Returns:
        The largest snapshot week <= ``target_week`` that exists on disk,
        or ``None`` if no eligible snapshot is present.
    """
    candidates = [w for w in available_snapshots(season) if w <= target_week]
    return max(candidates) if candidates else None


def load_snapshot(
    season: int,
    snapshot_week: int,
    file_kind: str,
) -> pd.DataFrame | None:
    """Load one parquet from the ``(season, snapshot_week)`` directory.

    Returns ``None`` when the snapshot directory or the requested file is
    missing. Returns an empty DataFrame when the parquet exists but
    holds no rows (FP sometimes writes empty files during in-flight
    weeks). Callers should treat ``None`` and empty the same.

    Args:
        season: NFL season year (e.g. 2024).
        snapshot_week: Exact snapshot week number to load (not a target;
            use :func:`load_asof` for time-capsule resolution).
        file_kind: Logical kind name from :data:`FILE_KINDS`
            (e.g. ``"efficiency"``).

    Returns:
        The parquet contents as a DataFrame, or ``None`` if the snapshot
        directory or the requested file is absent, or if the parquet
        cannot be parsed.

    Raises:
        ValueError: If ``file_kind`` is not a key in :data:`FILE_KINDS`.
    """
    if file_kind not in FILE_KINDS:
        raise ValueError(
            f"unknown FP weekly file_kind: {file_kind!r}; " f"valid kinds: {sorted(FILE_KINDS)}"
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


def load_asof(
    season: int,
    target_week: int,
    file_kind: str,
) -> pd.DataFrame | None:
    """Load the time-capsule snapshot of ``file_kind`` as-of ``target_week``.

    Resolves to the largest snapshot week N <= ``target_week`` (via
    :func:`resolve_asof_snapshot`) and returns that parquet, or ``None``
    if no eligible snapshot exists. This is the entry point a leakage-
    safe training-feature builder should call: pass the target game's
    ``(season, week)`` and you get FP's view of the world strictly
    before that game.

    Args:
        season: NFL season year (e.g. 2024).
        target_week: The game week. The loader returns FP's view of the
            world as of the most recent snapshot at or before this week.
        file_kind: Logical kind name from :data:`FILE_KINDS`
            (e.g. ``"efficiency"``).

    Returns:
        The parquet contents as a DataFrame for the resolved snapshot
        week, or ``None`` if no eligible snapshot exists or the file is
        absent/unreadable.
    """
    snapshot_week = resolve_asof_snapshot(season, target_week)
    if snapshot_week is None:
        return None
    return load_snapshot(season, snapshot_week, file_kind)


def load_all_snapshots(
    season: int,
    file_kind: str,
) -> dict[int, pd.DataFrame]:
    """Load every available snapshot of ``file_kind`` for ``season``.

    Returns ``{snapshot_week: DataFrame}`` for the snapshots that exist
    and parsed successfully. Useful for offline analysis (snapshot drift
    diagnostics, feature stability checks) -- the production code paths
    use :func:`load_asof` instead.

    Args:
        season: NFL season year (e.g. 2024).
        file_kind: Logical kind name from :data:`FILE_KINDS`
            (e.g. ``"efficiency"``).

    Returns:
        Mapping of snapshot week → DataFrame for every week that is
        present on disk and parses without error. Empty dict when no
        snapshots exist for the season.
    """
    out: dict[int, pd.DataFrame] = {}
    for week in available_snapshots(season):
        df = load_snapshot(season, week, file_kind)
        if df is not None and not df.empty:
            out[week] = df
    return out


def snapshot_inventory(seasons: Iterable[int] | None = None) -> pd.DataFrame:
    """Return a DataFrame inventorying which snapshots+files exist on disk.

    Columns: ``season``, ``snapshot_week``, plus one boolean column per
    logical file_kind (True iff that parquet is present in the snapshot
    directory). ``seasons=None`` scans every season subdir present.

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
        season_entries = sorted(int(name) for name in os.listdir(nfl_dir) if name.isdigit())
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
