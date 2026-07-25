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
confirmed once the FP fetcher in :mod:`sportstradamus.collectors.fantasypoints`
finishes parameterizing those endpoints.

This module is read-only. It does NOT wire into ``build_comp_profile``
or ``base_profile`` yet -- that swap happens in the Phase 1.5 / 2 / 3 / 4
plans (see ``.claude/plans/nfl-fp-phase-2-3-4-followups.md``) once the
historical seasons (2022-2024) and the in-flight 2025 season finish
backfilling.
"""

from __future__ import annotations

from sportstradamus.stats.nfl_fp_snapshots import SnapshotStore

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


_store = SnapshotStore(base_dir="player_data/NFL", file_kinds=FILE_KINDS, label="FP weekly")

# Expose the configured store's methods as this module's public loader API
# (the stdlib-`random` idiom). The identical block in the sibling team module
# is justified parallelism -- same API, different config, no logic -- so the
# duplicate-code check is silenced for it.
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
