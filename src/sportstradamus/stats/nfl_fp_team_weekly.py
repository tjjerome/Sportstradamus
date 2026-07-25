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

from sportstradamus.stats.nfl_fp_snapshots import SnapshotStore

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


_store = SnapshotStore(base_dir="team_data/NFL", file_kinds=FILE_KINDS, label="FP team weekly")

# Expose the configured store's methods as this module's public loader API
# (the stdlib-`random` idiom). The identical block in the sibling player
# module is justified parallelism -- same API, different config, no logic --
# so the duplicate-code check is silenced for it.
# pylint: disable=duplicate-code
snapshot_dir = _store.snapshot_dir
available_snapshots = _store.available_snapshots
load_snapshot = _store.load_snapshot
load_window = _store.load_window
load_window_or_empty = _store.load_window_or_empty
load_through = _store.load_through
load_all_snapshots = _store.load_all_snapshots
snapshot_inventory = _store.snapshot_inventory
enable_snapshot_cache = _store.enable_cache
disable_snapshot_cache = _store.disable_cache
consumed_snapshot_hashes = _store.consumed_hashes
