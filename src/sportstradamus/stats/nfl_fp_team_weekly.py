"""FantasyPoints weekly team-snapshot parquet loader for the NFL pipeline.

Sibling to :mod:`sportstradamus.stats.nfl_fp_weekly` (which reads the
player-grain snapshots at ``data/player_data/NFL/``). This module reads
the team-grain snapshots that the operator is downloading into
``data/team_data/NFL/{season}/week_{NN}/``. Each parquet here has one
row per (team, game) -- i.e. **per-game team aggregates**, NOT
season-to-date totals. Same per-game grain as the sibling player loader;
the time-capsule semantics described here mirror that module's.

Layout on disk::

    data/team_data/NFL/{season}/
        week_{NN}/
            coverage_matrix.parquet
            coverage_matrix_opp.parquet
            fantasy_points_scored.parquet
            fantasy_points_scored_opp.parquet
            line_matchups.parquet            <-- this-week-only forward look
            passing_advanced.parquet
            passing_advanced_opp.parquet
            ...                              <-- 20 file_kinds total

The ``*_opp`` variants are defense-faced equivalents of the offensive
parquets (e.g. ``rushing_advanced_opp`` = "what this team's defense has
allowed on the ground"). Consumers should join offensive features on
the team's own ``teamTeamId`` and defensive features on the opponent's
``teamTeamId``.

Time-capsule semantics
----------------------

To get "season-to-date through week N" for any rolling team feature,
:func:`load_through` concatenates snapshots 1..N and the caller
aggregates. For leakage-safe queries on a game in week ``W``, pass
``W - 1`` -- the loader never returns rows from the target game itself.
Use :func:`load_window` when the desired range is not anchored at
week 1 (e.g. "last 4 weeks of defense form").

Aggregation guidance for downstream consumers
---------------------------------------------

Most team-level stats are **rate stats with a denominator** -- they
should be aggregated as ``sum(numerator) / sum(denominator)`` across
the concatenated games, NOT as ``mean(rate)`` across games. The
mean-of-rates form silently up-weights low-volume games (a snowed-out
17-attempt rushing game gets the same vote as a 35-attempt division
game). Recommended denominators per family:

- passing_*: weight by ``teamStatsPassingDropbacksTotal``
- receiving_*: weight by ``teamStatsReceivingRoutesTotal`` (or
  ``teamStatsReceivingTargetsTotal`` for target-share metrics)
- rushing_*: weight by ``teamStatsRushingAttemptsTotal``
- coverage_matrix*: weight per-scheme % by
  ``teamStatsPassingDropbacksTotal``; weight per-scheme YPRR by the
  scheme's own dropback count column
- run_pass_report: weight per-``bucket`` pass% by snaps in that bucket
- proe_report: ``sum(actual_dropbacks - expected_dropbacks) /
  sum(expected_dropbacks)``; the per-game ratio is biased

A few kinds are **matchup-dependent forecasts**, not historical
aggregates -- they're FP's expert view of the upcoming game and must
NOT be averaged. Pull the row at the target game's snapshot directly
via :func:`load_snapshot` and join on ``(team, opponent, week)``:

- line_matchups -- both ``teamStats*`` (this team's protection-line
  baseline) and ``opponentStats*`` (the opponent's pass-rush profile
  for this matchup) are forward-looking and game-specific
  (final semantics pending fetcher-side schema confirmation; the
  ``opponentStats*`` interpretation is currently best-guess and should
  be re-checked once the team-data backfill populates)

And a few are **game outcomes** that average game-to-game (the
denominator is just games):

- fantasy_points_scored / fantasy_points_scored_opp -- team-level
  FP-scored / FP-allowed; mean across in-window games is appropriate
  (or median to dampen blow-out tails)

The detailed per-column categorization lives in
``.claude/plans/nfl-fp-phase-2-3-4-followups.md`` (Phase 1.5 / 2
sections).

Like its sibling, this module is read-only and not yet wired into
``base_profile`` or ``get_stats``. Consumers land in Phase 2 (team /
defense profile augmentation) once the team-data download is fully
backfilled.
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

# Logical kind name -> on-disk basename. Same encapsulation pattern as
# the player-grain loader: callers reference logical names so a future
# FP schema rename ripples through this constant only.
FILE_KINDS: dict[str, str] = {
    "coverage_matrix": "coverage_matrix.parquet",
    "coverage_matrix_opp": "coverage_matrix_opp.parquet",
    "fantasy_points_scored": "fantasy_points_scored.parquet",
    "fantasy_points_scored_opp": "fantasy_points_scored_opp.parquet",
    "line_matchups": "line_matchups.parquet",
    "passing_advanced": "passing_advanced.parquet",
    "passing_advanced_opp": "passing_advanced_opp.parquet",
    "passing_basic": "passing_basic.parquet",
    "passing_basic_opp": "passing_basic_opp.parquet",
    "proe_report": "proe_report.parquet",
    "proe_report_opp": "proe_report_opp.parquet",
    "receiving_advanced": "receiving_advanced.parquet",
    "receiving_advanced_opp": "receiving_advanced_opp.parquet",
    "receiving_basic": "receiving_basic.parquet",
    "receiving_basic_opp": "receiving_basic_opp.parquet",
    "run_pass_report": "run_pass_report.parquet",
    "rushing_advanced": "rushing_advanced.parquet",
    "rushing_advanced_opp": "rushing_advanced_opp.parquet",
    "rushing_basic": "rushing_basic.parquet",
    "rushing_basic_opp": "rushing_basic_opp.parquet",
}

# Logical kinds that are forward-looking matchup forecasts rather than
# historical aggregates. Concatenating these across weeks is WRONG --
# consumers must call :func:`load_snapshot` for the target week directly.
MATCHUP_FORECAST_KINDS: frozenset[str] = frozenset({"line_matchups"})

# Logical kinds whose ``*_opp`` mirror is the defense-faced equivalent.
# Useful for callers that want to iterate "all paired offensive kinds"
# without enumerating manually.
PAIRED_OFFENSE_DEFENSE_KINDS: frozenset[str] = frozenset(
    {
        "coverage_matrix",
        "fantasy_points_scored",
        "passing_advanced",
        "passing_basic",
        "proe_report",
        "receiving_advanced",
        "receiving_basic",
        "rushing_advanced",
        "rushing_basic",
    }
)

# Snapshot directory naming convention. Same scheme the player loader
# uses; FP publishes both data tiers under matched week numbers.
_WEEK_DIR_RE = re.compile(r"^week_(\d{2})$")
_WEEK_DIR_FMT = "week_{week:02d}"
_TEAM_DIR = "team_data/NFL"


def snapshot_dir(season: int, snapshot_week: int) -> Traversable | None:
    """Return the team-snapshot directory for ``(season, snapshot_week)``.

    Args:
        season: NFL season year (e.g. 2024).
        snapshot_week: Zero-padded week number within that season.

    Returns:
        An importlib-resources path object pointing at the
        ``week_{NN}/`` directory under ``team_data/NFL/{season}/``, or
        ``None`` if the directory is absent.
    """
    week_dir = _WEEK_DIR_FMT.format(week=snapshot_week)
    directory = pkg_resources.files(data) / f"{_TEAM_DIR}/{season}/{week_dir}"
    return directory if os.path.exists(directory) else None


def available_snapshots(season: int) -> list[int]:
    """Return sorted list of team-snapshot weeks present on disk.

    Args:
        season: NFL season year (e.g. 2024).

    Returns:
        Sorted list of integer week numbers whose ``week_{NN}/``
        directories exist under the season root. Empty if none.
    """
    season_dir = pkg_resources.files(data) / f"{_TEAM_DIR}/{season}"
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
    """Load one team-grain parquet from the ``(season, snapshot_week)`` dir.

    Returns the per-team-per-game rows for the games played in
    ``snapshot_week`` of ``season``. To stitch multiple weeks together
    into a season-to-date view, use :func:`load_through` or
    :func:`load_window` instead. For matchup-forecast kinds
    (:data:`MATCHUP_FORECAST_KINDS`), this single-snapshot accessor IS
    the right entry point -- those rows must not be concatenated.

    Args:
        season: NFL season year (e.g. 2024).
        snapshot_week: Exact snapshot week number to load.
        file_kind: Logical kind name from :data:`FILE_KINDS`
            (e.g. ``"line_matchups"``).

    Returns:
        The parquet contents as a DataFrame, or ``None`` if the
        snapshot directory or the requested file is missing. Returns an
        empty DataFrame when the parquet exists but holds no rows
        (FP sometimes writes empty files during in-flight weeks).

    Raises:
        ValueError: If ``file_kind`` is not a known team-grain kind.
    """
    if file_kind not in FILE_KINDS:
        raise ValueError(
            f"unknown FP team weekly file_kind: {file_kind!r}; valid kinds: {sorted(FILE_KINDS)}"
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
            "failed to read team %s for season=%s week=%s: %s",
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
    """Concatenate per-game team snapshots over ``[start_week, end_week]`` inclusive.

    The returned DataFrame stacks per-team-per-game rows from every
    snapshot whose week is in the closed interval. Snapshots that are
    missing or empty are silently skipped; the function returns
    ``None`` only when no snapshot in the window produced any rows.

    The caller is responsible for aggregation. For rate stats, the
    correct pattern is ``sum(numerator) / sum(denominator)`` across the
    concatenated rows -- see the module docstring for the per-kind
    denominator recommendations. For matchup-forecast kinds
    (:data:`MATCHUP_FORECAST_KINDS`), call :func:`load_snapshot`
    directly instead -- aggregating those rows is a bug.

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
            f"unknown FP team weekly file_kind: {file_kind!r}; valid kinds: {sorted(FILE_KINDS)}"
        )
    if start_week > end_week:
        raise ValueError(f"start_week={start_week} must be <= end_week={end_week}")

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
    """Load every available team-snapshot of ``file_kind`` for ``season``.

    Args:
        season: NFL season year (e.g. 2024).
        file_kind: Logical kind name from :data:`FILE_KINDS`.

    Returns:
        ``{snapshot_week: DataFrame}`` for snapshots that exist and
        parsed successfully. Production paths use :func:`load_window` /
        :func:`load_through`; this returns the snapshots un-concatenated
        for diagnostics and snapshot-drift analysis.
    """
    out: dict[int, pd.DataFrame] = {}
    for week in available_snapshots(season):
        df = load_snapshot(season, week, file_kind)
        if df is not None and not df.empty:
            out[week] = df
    return out


def snapshot_inventory(seasons: Iterable[int] | None = None) -> pd.DataFrame:
    """Inventory which team-snapshots+files exist on disk.

    Args:
        seasons: Optional iterable of season years to scan. ``None``
            scans every season subdir present under ``team_data/NFL/``.

    Returns:
        DataFrame with columns ``season``, ``snapshot_week``, plus one
        boolean per logical file_kind (True iff that parquet is present
        in the snapshot directory). Used by operator tools to confirm a
        download is complete before swapping production over.
    """
    if seasons is None:
        team_dir = pkg_resources.files(data) / _TEAM_DIR
        if not os.path.exists(team_dir):
            return pd.DataFrame(columns=["season", "snapshot_week", *FILE_KINDS])
        season_entries = sorted(
            int(entry.name) for entry in Path(str(team_dir)).iterdir() if entry.name.isdigit()
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
