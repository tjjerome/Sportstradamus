"""Phase-1.5 bridge: aggregate weekly FP snapshots into the comp-profile shape.

The season-CSV loader (:mod:`sportstradamus.stats.nfl_fp_loader`,
:func:`load_one_year`) produces one column per FP CSV column with the
``pass_adv_*`` / ``rec_adv_*`` / ``rush_adv_*`` / ``rush_bellcow_*`` /
``off_snaps_*`` / ``eff_*`` / ``qb_cov_*`` / ``tm_cov_off_*`` prefix
scheme; downstream
(:func:`sportstradamus.stats.nfl_fp_loader.derive_comp_metrics` and
``build_comp_profile``) consumes that wide schema directly.

This module reproduces the same wide schema from per-game weekly parquets
exposed by :mod:`sportstradamus.stats.nfl_fp_weekly`, so
``build_comp_profile`` can call :func:`load_through_one_year` for a
target ``(season, week)`` and get a leakage-clean lookback profile that
plugs into the same downstream code path as :func:`load_one_year`.

The bridge is a single recipe table — :data:`_AGGREGATE_RECIPES` — that
encodes, per output column, which file_kind to read and which aggregation
pattern (sum / weighted_rate / weighted_mean / game_mean from
:mod:`nfl_fp_aggregation`) turns the per-game rows into a season-to-date
value. Each output column is named with the same prefix the season-CSV
loader uses, so the downstream merge with the season-CSV path or with
:func:`derive_comp_metrics` is column-name compatible.

A small number of season-CSV columns are computed by formula rather than a
single primitive — :func:`_derive_aggregate_metrics` handles those after
the recipe table runs. Examples: ``pass_adv_ANY_A`` (Adjusted Net Yards
per Attempt) needs the standard formula
``(YDS - SACK_YDS + 20*TD - 45*INT) / (ATT + SACK)`` over season sums;
``rec_sep_cov_WIN_RATE.4`` selects the deep-coverage bucket.

The bridge is read-only with respect to the season-CSV path — they share
no module state and either can be called independently. Phase-1.5
wiring in ``build_comp_profile`` switches between them based on whether
a ``target_game_date`` was passed.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sportstradamus.helpers.text import remove_accents
from sportstradamus.spiderLogger import logger
from sportstradamus.stats import nfl_fp_team_weekly_aggregate, nfl_fp_weekly
from sportstradamus.stats.nfl_fp_aggregation import (
    PLAYER_GROUP_COL,
    game_mean,
    total_sum,
    weighted_mean,
    weighted_rate,
)
from sportstradamus.stats.nfl_fp_loader import (
    _COMP_POSITIONS,
    _RB_ROLE_ALIASES,
    _join_nfl_players,
    _load_pbp_named_by_player,
    derive_comp_metrics,
)

# ANY/A formula constants per the standard pro-football-reference definition.
# Touchdowns are worth 20 yards; interceptions cost 45. Same constants used
# inside the FP season CSV's ``ANY/A`` column.
_ANY_A_TD_BONUS = 20
_ANY_A_INT_PENALTY = 45

# Coverage-scheme percentage columns in qb_coverage_matchup.parquet are
# named ``playerStatsCoverageScheme{Scheme}PassingDropbacksPercentage``;
# the MAN bucket is the per-game fraction of dropbacks faced against
# man coverage. Cover2/3/4/6 buckets exist but aren't load-bearing for
# the current comp config.
_QB_COV_MAN_PCT_COL = "playerStatsCoverageSchemeManPassingDropbacksPercentage"
_QB_COV_DROPBACKS_COL = "playerStatsPassingDropbacksTotal"

# In the season-CSV separation-by-coverage file, the 5th sub-stratum
# (auto-suffixed ``.4`` on the WIN_RATE column) is the simpler "deep"
# bucket. The weekly equivalent is the per-game cover3 + cover4 rolled
# rows; absent those, the qb_coverage_matchup ``Cover3``/``Cover4`` PPR
# totals are not the same quantity. Phase-1.5 v1 leaves
# ``rec_sep_cov_WIN_RATE.4`` empty in the weekly path -- only the
# alignment overall (sep_overall, win_rate_overall) lands here. Deep-bucket
# coverage signal returns in Phase-2 when team-data fetcher fills in.


@dataclass(frozen=True)
class _Recipe:
    """One output column derived from one file_kind via one aggregation pattern.

    Attributes:
        output_col: Column name in the aggregated frame; uses the same
            prefix scheme as the season-CSV loader's output.
        file_kind: Logical name from
            :data:`sportstradamus.stats.nfl_fp_weekly.FILE_KINDS`.
        pattern: One of ``"sum"`` / ``"weighted_rate"`` /
            ``"weighted_mean"`` / ``"game_mean"``. Maps to the helper of
            the same name in :mod:`nfl_fp_aggregation`.
        args: Source-column tuple. ``("col",)`` for sum/game_mean;
            ``("num", "den")`` for weighted_rate/weighted_mean.
    """

    output_col: str
    file_kind: str
    pattern: str
    args: tuple[str, ...]


# Per-output-column recipes, grouped by file_kind for readability. Each
# entry maps one season-CSV column the comp profile consumes downstream to
# the per-game weekly columns + aggregation pattern that reconstruct it.
# Pattern choice is per the guidance in
# ``nfl-fp-phase-2-3-4-followups.md`` "Aggregation guidance for downstream
# consumers":
#
# * sum            -> raw season-to-date totals (Pattern A for tallies)
# * weighted_rate  -> sum(num)/sum(den) when both raw totals are exposed
# * weighted_mean  -> sum(rate*weight)/sum(weight) when only the rate is
#                     exposed per game (aDOT, TWT_pct, RPO_pct, etc.)
# * game_mean      -> Pattern C, mean of per-game outcomes
#
# Recipes that need a multi-column formula (ANY/A, sep_routes_mean) are
# computed in :func:`_derive_aggregate_metrics` after this table runs.
_AGGREGATE_RECIPES: tuple[_Recipe, ...] = (
    # passing_advanced -- QB primary file
    _Recipe("pass_adv_DB", "passing_advanced", "sum", ("playerStatsPassingDropbacksTotal",)),
    _Recipe("pass_adv_SCRM", "passing_advanced", "sum", ("playerStatsPassingScramblesTotal",)),
    _Recipe("pass_adv_ATT", "passing_advanced", "sum", ("playerStatsPassingAttemptsTotal",)),
    _Recipe("pass_adv_YDS", "passing_advanced", "sum", ("playerStatsPassingYardsTotal",)),
    _Recipe("pass_adv_TD", "passing_advanced", "sum", ("playerStatsPassingTouchdownsTotal",)),
    _Recipe(
        "pass_adv_AY",
        "passing_advanced",
        "sum",
        ("playerStatsPassingYardsAirTotal",),
    ),
    _Recipe(
        "pass_adv_FD",
        "passing_advanced",
        "sum",
        ("playerStatsFirstDownsPassing",),
    ),
    _Recipe("pass_adv_INT", "passing_advanced", "sum", ("playerStatsPassingInterceptionsTotal",)),
    _Recipe("pass_adv_SACK", "passing_advanced", "sum", ("playerStatsPassingSackedTotal",)),
    _Recipe(
        "pass_adv_SACK_YDS",
        "passing_advanced",
        "sum",
        ("playerStatsPassingSackedYardsLost",),
    ),
    _Recipe(
        "pass_adv_CPOE",
        "passing_advanced",
        "weighted_rate",
        (
            "playerStatsPassingCompletionsOverExpected",
            "playerStatsPassingAttemptsTotal",
        ),
    ),
    _Recipe(
        "pass_adv_OFFTARGET_pct",
        "passing_advanced",
        "weighted_mean",
        (
            "playerStatsPassingOffTargetThrowAttemptsPercentage",
            "playerStatsPassingAttemptsTotal",
        ),
    ),
    _Recipe(
        "pass_adv_HERO_pct",
        "passing_advanced",
        "weighted_mean",
        (
            "playerStatsPassingHeroThrowPercentage",
            "playerStatsPassingAttemptsTotal",
        ),
    ),
    _Recipe(
        "pass_adv_CATCHABLE_pct",
        "passing_advanced",
        "weighted_mean",
        (
            "playerStatsPassingAttemptsCatchablePercentage",
            "playerStatsPassingAttemptsTotal",
        ),
    ),
    _Recipe(
        "pass_adv_aDOT",
        "passing_advanced",
        "weighted_mean",
        (
            "playerStatsPassingAverageDepthOfTarget",
            "playerStatsPassingAttemptsTotal",
        ),
    ),
    _Recipe(
        "pass_adv_DEEP_pct",
        "passing_advanced",
        "weighted_rate",
        (
            "playerStatsPassingDeepThrowAttemptsTotal",
            "playerStatsPassingAttemptsTotal",
        ),
    ),
    _Recipe(
        "pass_adv_PrROE",
        "passing_advanced",
        "weighted_rate",
        (
            "playerStatsPassingPressuredOverExpected",
            "playerStatsPassingDropbacksTotal",
        ),
    ),
    _Recipe(
        "pass_adv_TTT",
        "passing_advanced",
        "weighted_mean",
        (
            "playerStatsPassingAverageTimeToThrow",
            "playerStatsPassingDropbacksTotal",
        ),
    ),
    _Recipe(
        "pass_adv_TWT_pct",
        "passing_advanced",
        "weighted_mean",
        (
            "playerStatsPassingTowThrowPercentage",
            "playerStatsPassingAttemptsTotal",
        ),
    ),
    _Recipe(
        "pass_adv_RPO_pct",
        "passing_advanced",
        "weighted_mean",
        (
            "playerStatsPassingRunPassOptionPercentage",
            "playerStatsPassingDropbacksTotal",
        ),
    ),
    # passing_basic -- only ``YDS.1`` (QB rushing yards) is consumed
    # downstream; the auto-suffixed name comes from FP CSV having two
    # ``YDS`` columns (passing + rushing). The weekly parquet exposes
    # rushing yards as a separately-named column.
    _Recipe(
        "pass_basic_YDS.1",
        "passing_basic",
        "sum",
        ("playerStatsRushingYardsTotal",),
    ),
    # rushing_basic -- RB volume input + goal-line carry totals
    _Recipe(
        "rush_basic_ATT",
        "rushing_basic",
        "sum",
        ("playerStatsRushingAttemptsTotal",),
    ),
    _Recipe(
        "rush_basic_INSIDE5_ATT",
        "rushing_basic",
        "sum",
        ("playerStatsInside5RushingAttemptsTotal",),
    ),
    _Recipe(
        "rush_basic_INSIDE10_ATT",
        "rushing_basic",
        "sum",
        ("playerStatsInside10RushingAttemptsTotal",),
    ),
    _Recipe(
        "rush_basic_INSIDE20_ATT",
        "rushing_basic",
        "sum",
        ("playerStatsInside20RushingAttemptsTotal",),
    ),
    # rushing_advanced -- RB volume + efficiency + scheme split
    _Recipe(
        "rush_adv_YDS",
        "rushing_advanced",
        "sum",
        ("playerStatsRushingYardsTotal",),
    ),
    _Recipe(
        "rush_adv_TD",
        "rushing_advanced",
        "sum",
        ("playerStatsRushingTouchdownsTotal",),
    ),
    _Recipe(
        "rush_adv_FD",
        "rushing_advanced",
        "sum",
        ("playerStatsFirstDownsRushing",),
    ),
    _Recipe(
        "rush_adv_EXP_YDS",
        "rushing_advanced",
        "sum",
        ("playerStatsRushingYardsExplosiveTotal",),
    ),
    _Recipe(
        "rush_adv_Success_pct",
        "rushing_advanced",
        "weighted_mean",
        (
            "playerStatsRushingAttemptsSuccessPercentage",
            "playerStatsRushingAttemptsTotal",
        ),
    ),
    _Recipe(
        "rush_adv_EXP_RUN_pct",
        "rushing_advanced",
        "weighted_mean",
        (
            "playerStatsRushingRunsExplosivePercentage",
            "playerStatsRushingAttemptsTotal",
        ),
    ),
    _Recipe(
        "rush_adv_MTF_ATT",
        "rushing_advanced",
        "weighted_rate",
        (
            "playerStatsRushingMissedTacklesForcedTotal",
            "playerStatsRushingAttemptsTotal",
        ),
    ),
    _Recipe(
        "rush_adv_YACO_ATT",
        "rushing_advanced",
        "weighted_rate",
        (
            "playerStatsRushingYardsAfterContactTotal",
            "playerStatsRushingAttemptsTotal",
        ),
    ),
    _Recipe(
        "rush_adv_YBC_ATT",
        "rushing_advanced",
        "weighted_mean",
        (
            "playerStatsRushingYardsBeforeContactPerAttempt",
            "playerStatsRushingAttemptsTotal",
        ),
    ),
    _Recipe(
        "rush_adv_STUFF_pct",
        "rushing_advanced",
        "weighted_mean",
        (
            "playerStatsRushingAttemptsStuffsPercentage",
            "playerStatsRushingAttemptsTotal",
        ),
    ),
    _Recipe(
        "rush_adv_ZONE_pct",
        "rushing_advanced",
        "weighted_mean",
        (
            "playerStatsRushingConceptZoneAttemptsPercentage",
            "playerStatsRushingAttemptsTotal",
        ),
    ),
    _Recipe(
        "rush_adv_MAN_pct",
        "rushing_advanced",
        "weighted_mean",
        (
            "playerStatsRushingConceptManAttemptsPercentage",
            "playerStatsRushingAttemptsTotal",
        ),
    ),
    # rushing_bell_cow -- RB market-share inputs
    _Recipe(
        "rush_bellcow_ATT_pct",
        "rushing_bell_cow",
        "weighted_mean",
        ("marketShareRushingAttemptsTotal", "teamStatsRushingAttemptsTotal"),
    ),
    _Recipe(
        "rush_bellcow_TGT_pct",
        "rushing_bell_cow",
        "weighted_mean",
        ("marketShareReceivingTargetsTotal", "teamStatsReceivingTargetsTotal"),
    ),
    _Recipe(
        "rush_bellcow_XFP_pct",
        "rushing_bell_cow",
        "weighted_rate",
        ("marketShareXfpPprTotal", "teamStatsXfpPprTotal"),
    ),
    # receiving_basic -- volume input shared with RB derivation + red-zone target totals
    _Recipe(
        "rec_basic_REC",
        "receiving_basic",
        "sum",
        ("playerStatsReceivingReceptionsTotal",),
    ),
    _Recipe(
        "rec_basic_INSIDE10_TGT",
        "receiving_basic",
        "sum",
        ("playerStatsInside10ReceivingTargetsTotal",),
    ),
    _Recipe(
        "rec_basic_INSIDE20_TGT",
        "receiving_basic",
        "sum",
        ("playerStatsInside20ReceivingTargetsTotal",),
    ),
    # receiving_advanced -- WR / TE / RB primary file
    _Recipe("rec_adv_RTE", "receiving_advanced", "sum", ("playerStatsReceivingRoutesTotal",)),
    _Recipe("rec_adv_TGT", "receiving_advanced", "sum", ("playerStatsReceivingTargetsTotal",)),
    _Recipe(
        "rec_adv_TD",
        "receiving_advanced",
        "sum",
        ("playerStatsReceivingTouchdownsTotal",),
    ),
    _Recipe(
        "rec_adv_FD",
        "receiving_advanced",
        "sum",
        ("playerStatsFirstDownsReceiving",),
    ),
    _Recipe(
        "rec_adv_YAC",
        "receiving_advanced",
        "sum",
        ("playerStatsReceivingYardsAfterCatchTotal",),
    ),
    _Recipe(
        "rec_adv_YAC_after_contact",
        "receiving_advanced",
        "sum",
        ("playerStatsReceivingYardsAfterContactTotal",),
    ),
    _Recipe(
        "rec_adv_DEEP_TGT",
        "receiving_advanced",
        "sum",
        ("playerStatsReceivingDeepTargetTargetsTotal",),
    ),
    _Recipe(
        "rec_adv_YPRR",
        "receiving_advanced",
        "weighted_rate",
        (
            "playerStatsReceivingYardsTotal",
            "playerStatsReceivingRoutesTotal",
        ),
    ),
    _Recipe(
        "rec_adv_TPRR",
        "receiving_advanced",
        "weighted_rate",
        (
            "playerStatsReceivingTargetsTotal",
            "playerStatsReceivingRoutesTotal",
        ),
    ),
    _Recipe(
        "rec_adv_AY_Share",
        "receiving_advanced",
        "weighted_mean",
        (
            "marketShareReceivingYardsAir",
            "playerStatsReceivingRoutesTotal",
        ),
    ),
    _Recipe(
        "rec_adv_TGT_pct",
        "receiving_advanced",
        "weighted_mean",
        (
            "marketShareReceivingTargetsTotal",
            "playerStatsReceivingRoutesTotal",
        ),
    ),
    _Recipe(
        "rec_adv_DRP_pct",
        "receiving_advanced",
        "weighted_rate",
        (
            "playerStatsReceivingDropsTotal",
            "playerStatsReceivingTargetsTotal",
        ),
    ),
    _Recipe(
        "rec_adv_CATCH_pct",
        "receiving_advanced",
        "weighted_rate",
        (
            "playerStatsReceivingTargetsCatchableTotal",
            "playerStatsReceivingTargetsTotal",
        ),
    ),
    _Recipe(
        "rec_adv_DESIGN_pct",
        "receiving_advanced",
        "weighted_rate",
        (
            "playerStatsReceivingTargetedReadDesignTotal",
            "playerStatsReceivingTargetsTotal",
        ),
    ),
    _Recipe(
        "rec_adv_1READ_pct",
        "receiving_advanced",
        "weighted_rate",
        (
            "playerStatsReceivingTargetedReadFirstTotal",
            "playerStatsReceivingTargetsTotal",
        ),
    ),
    _Recipe(
        "rec_adv_aDOT",
        "receiving_advanced",
        "weighted_mean",
        (
            "playerStatsReceivingAverageDepthOfTarget",
            "playerStatsReceivingTargetsTotal",
        ),
    ),
    _Recipe(
        "rec_adv_MTF_REC",
        "receiving_advanced",
        "weighted_rate",
        (
            "playerStatsReceivingMissedTacklesForcedTotal",
            "playerStatsReceivingReceptionsTotal",
        ),
    ),
    _Recipe(
        "rec_adv_CTGT_pct",
        "receiving_advanced",
        "weighted_rate",
        (
            "playerStatsReceivingTargetsContestedTotal",
            "playerStatsReceivingTargetsTotal",
        ),
    ),
    _Recipe(
        "rec_adv_CTGT_REC",
        "receiving_advanced",
        "sum",
        ("playerStatsReceivingReceptionsContested",),
    ),
    # receiving_routes_run -- alignment shares
    _Recipe(
        "rec_routes_WIDE_RTE_pct",
        "receiving_routes_run",
        "weighted_rate",
        (
            "playerStatsReceivingAlignmentWideRoutesTotal",
            "playerStatsReceivingRoutesTotal",
        ),
    ),
    _Recipe(
        "rec_routes_SLOT_RTE_pct",
        "receiving_routes_run",
        "weighted_rate",
        (
            "playerStatsReceivingAlignmentSlotRoutesTotal",
            "playerStatsReceivingRoutesTotal",
        ),
    ),
    # ``INLINE`` and ``BACKFIELD`` routes aren't broken out as totals in
    # receiving_routes_run; receiving_advanced exposes them as per-game
    # percentages of the player's routes that week.
    _Recipe(
        "rec_routes_INLINE_RTE_pct",
        "receiving_advanced",
        "weighted_mean",
        (
            "playerStatsReceivingAlignmentInlineRoutesPercentage",
            "playerStatsReceivingRoutesTotal",
        ),
    ),
    # receiving_man_vs_zone -- the ``YPRR.1`` (man) and ``.2`` (zone)
    # season-CSV columns; man/zone routes split via the ``bucket`` field
    # in the weekly parquet rather than separate columns.
    _Recipe(
        "rec_mz_YPRR_overall",
        "receiving_man_vs_zone",
        "weighted_rate",
        (
            "playerStatsReceivingYardsTotal",
            "playerStatsReceivingRoutesTotal",
        ),
    ),
    # receiving_separation_by_alignment -- overall row
    _Recipe(
        "rec_sep_align_overall_SEP_SCORE",
        "receiving_separation_by_alignment",
        "weighted_mean",
        (
            "playerStatsReceivingSeparationScorePercentage",
            "playerStatsReceivingSeparationRoutesTotal",
        ),
    ),
    _Recipe(
        "rec_sep_align_overall_WIN_RATE",
        "receiving_separation_by_alignment",
        "weighted_mean",
        (
            "playerStatsReceivingSeparationWinsPercentage",
            "playerStatsReceivingSeparationRoutesTotal",
        ),
    ),
    # offense_snap_share_report -- player offense snap %
    _Recipe(
        "off_snaps_Snap_pct",
        "offense_snap_share_report",
        "weighted_mean",
        (
            "marketShareSnapsOffenseTotal",
            "playerStatsGamesPlayed",
        ),
    ),
    # offense_snaps -- raw offensive snap total (denominator for the
    # TE ``block_pct_proxy`` derivation in ``_derive_aggregate_metrics``)
    # plus red-zone snap volume / share
    _Recipe(
        "off_snaps_TOTAL",
        "offense_snaps",
        "sum",
        ("playerStatsSnapsOffenseTotal",),
    ),
    _Recipe(
        "off_snaps_INSIDE10",
        "offense_snaps",
        "sum",
        ("playerStatsInside10SnapsOffenseTotal",),
    ),
    _Recipe(
        "off_snaps_INSIDE20",
        "offense_snaps",
        "sum",
        ("playerStatsInside20SnapsOffenseTotal",),
    ),
    _Recipe(
        "off_snaps_INSIDE10_pct",
        "offense_snaps",
        "weighted_mean",
        (
            "marketShareInside10SnapsOffenseTotal",
            "playerStatsGamesPlayed",
        ),
    ),
    # efficiency -- WR / RB fantasy efficiency + xFP + workload totals
    _Recipe(
        "eff_FP_G",
        "efficiency",
        "game_mean",
        ("playerStatsFantasyPointsPpr",),
    ),
    _Recipe(
        "eff_YFS_G",
        "efficiency",
        "game_mean",
        ("playerStatsScrimmageYardsTotal",),
    ),
    _Recipe(
        "eff_XFP",
        "efficiency",
        "game_mean",
        ("playerStatsXfpPprTotal",),
    ),
    _Recipe(
        "eff_TOUCHES",
        "efficiency",
        "sum",
        ("playerStatsScrimmageTouchesTotal",),
    ),
    # qb_coverage_matchup -- QB-side man-coverage exposure. Per-game
    # rows in this file are (QB, opponent, week); for the comp profile
    # we want a per-QB season-to-date weighted mean across all matchups.
    _Recipe(
        "qb_cov_QB_MAN_pct",
        "qb_coverage_matchup",
        "weighted_mean",
        (_QB_COV_MAN_PCT_COL, _QB_COV_DROPBACKS_COL),
    ),
    # Phase 3+4 asof additions -- QB pressure & throw-away & route-share signal.
    # These are recipe-only additions that the Phase 3+4
    # ``_compute_fp_asof_features`` helper on ``StatsNFL`` projects to the
    # ``Player {col}_asof`` namespace in the training matrix.
    _Recipe(
        "pass_adv_PrDB_pct",
        "passing_advanced",
        "weighted_rate",
        (
            "playerStatsPassingPressuredTotal",
            "playerStatsPassingDropbacksTotal",
        ),
    ),
    _Recipe(
        "pass_adv_TA_pct",
        "passing_advanced",
        "weighted_rate",
        (
            "playerStatsPassingIncompletionsThrownAway",
            "playerStatsPassingDropbacksTotal",
        ),
    ),
    _Recipe(
        "rec_route_share_pct",
        "receiving_route_share_report",
        "game_mean",
        ("marketShareReceivingRoutesTotal",),
    ),
)

# Recipes that share a denominator with the season-CSV's separation-by-
# routes file. Aggregating per (player, route_bucket) and meaning the
# SEP_SCORE across routes gives the per-player ``rec_sep_route_SEP_SCORE``
# family the season CSV exposes as ``.0`` ... ``.6`` auto-suffixed cols.
# The FP weekly parquet packs route-stratified SEP data inside the
# JSON-encoded ``bucket`` column rather than top-level fields; the
# parser below mirrors Phase 2's ``_rpr_bucket_pass_rate`` shape
# (`stats/nfl_fp_team_weekly_aggregate.py`).
_SEP_BY_ROUTES_KIND = "receiving_separation_by_routes"
_SEP_BY_ROUTES_VALUE_KEY = "playerStatsReceivingSeparationScorePercentage"
_SEP_BY_ROUTES_WEIGHT_KEY = "playerStatsReceivingSeparationRoutesTotal"

# Canonical route-bucket order for the ``rec_sep_route_SEP_SCORE`` positional
# suffix. The season-CSV publishes these columns left-to-right in the order
# below (``Overall`` first per FP's convention, then the per-route slices in
# the order the original CSV header carried them). Pinning the order here
# instead of falling back to ``sorted()`` keeps the ``.0`` / ``.1`` / ``.N``
# pairing stable across runs AND across the season-CSV / weekly-parquet
# paths, so any future consumer that references a specific suffix sees the
# same route in both. Unknown route keys observed in the JSON but absent
# from this tuple still get a suffix (appended in observation order) -- a
# new route bucket adds a column rather than silently dropping data.
_ROUTE_BUCKET_ORDER: tuple[str, ...] = (
    "bucketReceivingSeparationRouteOverall",
    "bucketReceivingSeparationRouteSlant",
    "bucketReceivingSeparationRouteOut",
    "bucketReceivingSeparationRouteInDig",
    "bucketReceivingSeparationRouteHitch",
    "bucketReceivingSeparationRouteComeback",
    "bucketReceivingSeparationRouteCorner",
    "bucketReceivingSeparationRoutePost",
    "bucketReceivingSeparationRouteGo",
    "bucketReceivingSeparationRouteCrossers",
    "bucketReceivingSeparationRouteFlat",
    "bucketReceivingSeparationRouteScreens",
    "bucketReceivingSeparationRouteBackfield",
)

# Phase 3+4 per-coverage YPRR (wr_coverage_matchup). One column per
# scheme exposes the per-game ``ReceivingYardsPerRoute`` rate plus
# ``ReceivingRoutesTotal`` denominator. ``_aggregate_per_coverage_yprr``
# weight-averages across games per (player, scheme) and rolls Cover 2/3/4/6
# into the ``zone`` bucket for sample-sufficiency on low-route players.
# Per-shell entries (``cover2`` .. ``cover6``) emit the un-folded shell rate
# alongside the folded ``zone`` rollup -- both are useful: the rollup for
# low-route players where individual shells are sparse, the per-shell for
# matchup-specific signal when the defense leans on one coverage.
_COVERAGE_YPRR_SCHEMES: dict[str, tuple[str, ...]] = {
    "man": ("Man",),
    "zone": ("Cover2", "Cover3", "Cover4", "Cover6"),
    "cover2": ("Cover2",),
    "cover3": ("Cover3",),
    "cover4": ("Cover4",),
    "cover6": ("Cover6",),
}

# Phase 3+4 per-coverage separation (receiving_separation_by_coverage).
# Bucket JSON has scheme-keyed inner dicts with
# ``playerStatsReceivingSeparationScorePercentage`` and
# ``...RoutesTotal``. ``_aggregate_separation_by_coverage`` parses the JSON,
# emits one column per scheme. ``rec_sep_vs_man`` and ``rec_sep_vs_zone``
# are the FP-pre-aggregated rollups (Zone bucket is one of the inner keys --
# FP publishes a pre-bucketed Zone alongside Man); ``rec_sep_vs_redzone``
# and ``rec_sep_vs_cover{2,3,4,6}`` are the per-shell un-folded rates.
_SEP_BY_COVERAGE_KIND = "receiving_separation_by_coverage"
_SEP_BY_COVERAGE_SCHEMES: dict[str, str] = {
    "man": "bucketReceivingSeparationMan",
    "zone": "bucketReceivingSeparationZone",
    "redzone": "bucketReceivingSeparationRedZone",
    "cover2": "bucketReceivingSeparationCover2",
    "cover3": "bucketReceivingSeparationCover3",
    "cover4": "bucketReceivingSeparationCover4",
    "cover6": "bucketReceivingSeparationCover6",
}
_SEP_BY_COVERAGE_VALUE_KEY = "playerStatsReceivingSeparationScorePercentage"
_SEP_BY_COVERAGE_WEIGHT_KEY = "playerStatsReceivingSeparationRoutesTotal"

# Per-alignment separation (receiving_separation_by_alignment.bucket).
# The FP parquet emits one bucket per alignment family with the same
# SEP_SCORE / WIN_RATE / Routes denominator schema as the by-coverage
# helper above. The top-level row already produces the overall rate
# (``rec_sep_align_overall_*``); this bucket helper produces the per-
# alignment slices that the comp profile + asof surface can read.
# ``Overall`` is intentionally omitted from this map -- it duplicates the
# top-level recipe outputs and would just bloat the wide frame.
_SEP_BY_ALIGNMENT_KIND = "receiving_separation_by_alignment"
_SEP_BY_ALIGNMENT_BUCKETS: dict[str, str] = {
    "WIDE": "bucketReceivingSeparationWide",
    "SLOT": "bucketReceivingSeparationSlot",
    "INLINE": "bucketReceivingSeparationInline",
    "BACKFIELD": "bucketReceivingSeparationBackfield",
}
_SEP_BY_ALIGNMENT_SEP_KEY = "playerStatsReceivingSeparationScorePercentage"
_SEP_BY_ALIGNMENT_WIN_KEY = "playerStatsReceivingSeparationWinsPercentage"
_SEP_BY_ALIGNMENT_WEIGHT_KEY = "playerStatsReceivingSeparationRoutesTotal"

# Per-coverage YPRR via the receiving_man_vs_zone bucket. The top-level
# row produces an overall yards-per-route via ``rec_mz_YPRR_overall``;
# this bucket helper produces the per-shell rates (Man / Zone /
# SingleHigh / TwoHigh). The bucket key sets are FP-pre-aggregated --
# Cover2/3/4/6 are NOT broken out here (they live in
# ``wr_coverage_matchup`` per-scheme columns instead, which the
# ``_aggregate_per_coverage_yprr`` helper above already consumes).
# ``Overall`` is intentionally omitted -- duplicates the top-level recipe.
_MZ_BUCKET_KIND = "receiving_man_vs_zone"
_MZ_BUCKETS: dict[str, str] = {
    "MAN": "bucketMan",
    "ZONE": "bucketZone",
    "SINGLEHIGH": "bucketSingleHigh",
    "TWOHIGH": "bucketTwoHigh",
}
_MZ_VALUE_KEY = "playerStatsReceivingAveragesPerRouteYardsTotal"
_MZ_WEIGHT_KEY = "playerStatsReceivingRoutesTotal"

# Metadata cols carried straight through (filtered, not aggregated). One
# representative row per player is sufficient since these don't change
# week-to-week.
_PLAYER_METADATA_COLS = (
    "playerFirstName",
    "playerLastName",
    "playerPosition",
    "teamAbbreviation",
)


def load_through_one_year(season: int, target_week: int) -> pd.DataFrame:
    """Aggregate per-game FP snapshots into a leakage-clean comp profile.

    Reads weeks ``1..target_week`` of ``season`` from
    :func:`nfl_fp_weekly.load_through`, applies :data:`_AGGREGATE_RECIPES`
    per file_kind, joins PBP/NGS aggregates the same way
    :func:`nfl_fp_loader.load_one_year` does, and returns a DataFrame
    with the same downstream shape (indexed by accent-stripped player
    name, with ``position``, ``player_game_count``, the prefixed FP
    metrics, the derived columns from :func:`derive_comp_metrics`, and
    age/height/bmi).

    Convenience wrapper for :func:`load_window_one_year` anchored at
    week 1. Use the windowed primitive when the caller needs a non-
    anchored slice (e.g. the comp profile's pre-week-5 fallback that
    pulls only the prior season's back half).

    The caller is responsible for the time-capsule semantics -- pass
    ``target_week = actual_target_week - 1`` to make the target game's
    own snapshot invisible to the comp pool.

    Args:
        season: NFL season year.
        target_week: Inclusive last week of the lookback window. For a
            leakage-safe comp profile being computed for week N, pass
            ``N - 1``.

    Returns:
        DataFrame keyed by accent-stripped player name. Empty if no
        snapshots are available for the window.
    """
    return load_window_one_year(season, 1, target_week)


def load_window_one_year(season: int, start_week: int, end_week: int) -> pd.DataFrame:
    """Aggregate per-game FP snapshots for a single closed week range.

    Convenience wrapper for :func:`load_multi_window_one_year` with a
    one-window list. Used when the caller wants a single-season window
    (e.g. legacy fallback paths that aggregate season-by-season).
    """
    return load_multi_window_one_year([(season, start_week, end_week)])


def load_multi_window_one_year(
    windows: Sequence[tuple[int, int, int]],
) -> pd.DataFrame:
    """Aggregate per-game FP snapshots across one or more (season, start, end) windows.

    Pools raw per-game rows from every window BEFORE applying the
    per-player aggregation. A player with games across two windows
    (e.g. 2024 weeks 11-18 + 2025 weeks 1-2 for a pre-week-5 comp
    lookback) gets one aggregated row whose numerator + denominator
    sums span every per-game appearance -- the "blend" semantics the
    Phase-1.5 pre-cutoff rule asks for.

    PBP/NGS aggregates and age/height/bmi are joined for the LATEST
    season in the window set (the one whose comp pool we're really
    computing for). Players who only appear in earlier seasons inherit
    NaN on the PBP columns, which the downstream ``fillna(0)`` chain
    in ``update_player_comps`` absorbs.

    Args:
        windows: List of ``(season, start_week, end_week)`` triples.
            Empty list returns an empty DataFrame. Windows with
            ``start_week > end_week`` are skipped silently.

    Returns:
        DataFrame keyed by accent-stripped player name with the same
        column shape as :func:`load_window_one_year`. Empty if every
        window produced no snapshots.
    """
    windows = [w for w in windows if w[1] <= w[2]]
    if not windows:
        return pd.DataFrame()

    by_player_id = _aggregate_recipes(windows)
    if by_player_id.empty:
        return pd.DataFrame()

    by_player_id = _combine_bucket_aggregates(by_player_id, windows)
    return _finalize_player_frame(by_player_id, windows)


def _broadcast_team_coverage(
    profile: pd.DataFrame, windows: Sequence[tuple[int, int, int]]
) -> pd.DataFrame:
    """Project the team-grain ``off_faced_{man,zone}_pct`` rates onto every player row.

    The KNN comp config (``data/config/playerCompStats.json``) weights WR
    on ``off_faced_man_pct`` and TE on ``off_faced_zone_pct`` -- the share
    of own-team dropbacks the player's offense faces vs each coverage
    shell. These rates exist only at team grain in the FP weekly drop;
    the player-comp profile broadcasts them onto each player via the
    player's ``Team`` so the KNN distance metric can see them.

    Uses the Pattern-A-only entry point on the team bridge to avoid
    pulling matchup-specific ``lm_*`` columns (Pattern B) into the
    comp profile -- those are forecasts for a specific game, not
    season-to-date features.

    Args:
        profile: Per-player wide frame indexed by accent-stripped name
            with a ``Team`` column carrying the player's team abbreviation.
        windows: Same ``(season, start_week, end_week)`` tuples the
            recipe aggregation pooled, so the broadcast rate aligns
            with the rest of the player's season-to-date features.

    Returns:
        ``profile`` with two columns added (or left NaN-filled): the
        ``off_faced_man_pct`` and ``off_faced_zone_pct`` values for
        each player's team. Missing-team rows are NaN.
    """
    if profile.empty or "Team" not in profile.columns:
        return profile
    team_features = nfl_fp_team_weekly_aggregate.load_pattern_a_team_features(windows)
    if team_features.empty:
        return profile
    keep_cols = [
        c for c in ("off_faced_man_pct", "off_faced_zone_pct") if c in team_features.columns
    ]
    if not keep_cols:
        return profile
    return profile.merge(team_features[keep_cols], left_on="Team", right_index=True, how="left")


def _aggregate_recipes(windows: Sequence[tuple[int, int, int]]) -> pd.DataFrame:
    """Apply :data:`_AGGREGATE_RECIPES`, grouping reads by file_kind to avoid duplicate IO.

    Pools per-game rows across every window before per-recipe aggregation
    so a player's stats from multiple seasons land in one aggregate row.
    """
    by_kind: dict[str, list[_Recipe]] = {}
    for recipe in _AGGREGATE_RECIPES:
        by_kind.setdefault(recipe.file_kind, []).append(recipe)

    out = pd.DataFrame()
    for file_kind, recipes in by_kind.items():
        df = _load_kind_multi(windows, file_kind)
        if df.empty:
            continue
        kind_frame = _apply_recipes(df, recipes)
        if kind_frame.empty:
            continue
        out = out.combine_first(kind_frame) if not out.empty else kind_frame
    return out


def _load_kind_multi(windows: Sequence[tuple[int, int, int]], file_kind: str) -> pd.DataFrame:
    """Concatenate per-game rows from every (season, start, end) window for ``file_kind``."""
    frames = []
    for season, start_week, end_week in windows:
        df = nfl_fp_weekly.load_window_or_empty(season, start_week, end_week, file_kind)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _apply_recipes(df: pd.DataFrame, recipes: Sequence[_Recipe]) -> pd.DataFrame:
    """Run each recipe against the file's per-game DataFrame, return a wide frame."""
    out = pd.DataFrame()
    for recipe in recipes:
        if not all(col in df.columns for col in recipe.args):
            continue
        series = _dispatch(df, recipe)
        if series is None or series.empty:
            continue
        out[recipe.output_col] = series
    return out


def _dispatch(df: pd.DataFrame, recipe: _Recipe) -> pd.Series | None:
    """Route a recipe to the right helper in :mod:`nfl_fp_aggregation`."""
    if recipe.pattern == "sum":
        (col,) = recipe.args
        return total_sum(df, col)
    if recipe.pattern == "weighted_rate":
        num_col, den_col = recipe.args
        return weighted_rate(df, num_col, den_col)
    if recipe.pattern == "weighted_mean":
        value_col, weight_col = recipe.args
        return weighted_mean(df, value_col, weight_col)
    if recipe.pattern == "game_mean":
        (col,) = recipe.args
        return game_mean(df, col, group_col=PLAYER_GROUP_COL)
    logger.warning("unknown aggregation pattern %r for %s", recipe.pattern, recipe.output_col)
    return None


def _iter_bucket_payloads(df: pd.DataFrame) -> Iterator[tuple[object, dict[str, object]]]:
    """Yield ``(player_id, payload)`` for each row whose JSON ``bucket`` parses to a dict.

    Shared preamble for the separation/coverage aggregators — each reads the
    same per-game ``bucket`` JSON column but extracts different inner keys.
    """
    for player_id, bucket_str in zip(df[PLAYER_GROUP_COL], df["bucket"], strict=True):
        if not isinstance(bucket_str, str) or not bucket_str:
            continue
        try:
            payload = json.loads(bucket_str)
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict):
            yield player_id, payload


def _weighted_mean_wide(long_rows: list[dict], bucket_col: str) -> pd.DataFrame:
    """Volume-weighted mean of (value, weight) per (player, bucket_col), pivoted wide."""
    work = pd.DataFrame(long_rows)
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work["weight"] = pd.to_numeric(work["weight"], errors="coerce")
    work = work.dropna(subset=["value", "weight"])
    if work.empty:
        return pd.DataFrame()
    grouped = work.groupby([PLAYER_GROUP_COL, bucket_col], dropna=False)
    weighted_num = grouped.apply(
        lambda g: (g["value"] * g["weight"]).sum(min_count=1),
        include_groups=False,
    )
    weighted_den = grouped["weight"].sum(min_count=1)
    per_bucket = weighted_num.divide(weighted_den).where(weighted_den != 0, other=np.nan)
    return per_bucket.unstack(bucket_col)


def _explode_scheme_long_rows(df, schemes, value_key, weight_key, bucket_col):
    """Long-form (player, bucket_col, value, weight) rows from a JSON ``bucket``
    column, reading each named scheme's inner dict by key. Skips non-dict inners
    and rows missing the value or weight key.
    """
    long_rows: list[dict[str, object]] = []
    for player_id, payload in _iter_bucket_payloads(df):
        for bucket_name, key in schemes.items():
            inner = payload.get(key)
            if not isinstance(inner, dict):
                continue
            value = inner.get(value_key)
            weight = inner.get(weight_key)
            if value is None or weight is None:
                continue
            long_rows.append(
                {
                    PLAYER_GROUP_COL: player_id,
                    bucket_col: bucket_name,
                    "value": value,
                    "weight": weight,
                }
            )
    return long_rows


def _explode_route_long_rows(df, value_key, weight_key):
    """Long-form (player, route_bucket, value, weight) rows from a JSON ``bucket``
    column whose top-level keys are route names (not a fixed scheme map).
    """
    long_rows: list[dict[str, object]] = []
    for player_id, payload in _iter_bucket_payloads(df):
        for route_key, route_payload in payload.items():
            if not isinstance(route_payload, dict):
                continue
            value = route_payload.get(value_key)
            weight = route_payload.get(weight_key)
            if value is None or weight is None:
                continue
            long_rows.append(
                {
                    PLAYER_GROUP_COL: player_id,
                    "route_bucket": route_key,
                    "value": value,
                    "weight": weight,
                }
            )
    return long_rows


def _order_and_suffix_routes(wide: pd.DataFrame) -> pd.DataFrame:
    """Match the season-CSV's ``.0``, ``.1`` ... ``.N`` auto-suffix scheme.

    Order by ``_ROUTE_BUCKET_ORDER`` first so suffix N maps to the same route both
    here and in the legacy season-CSV path. Unknown route keys (not in the canonical
    tuple) append after, in observation order, so a new route bucket grows the family
    rather than silently dropping data.
    """
    known = [b for b in _ROUTE_BUCKET_ORDER if b in wide.columns]
    unknown = [b for b in wide.columns if b not in _ROUTE_BUCKET_ORDER]
    ordered = known + unknown
    wide = wide.reindex(ordered, axis=1)
    rename: dict[object, str] = {ordered[0]: "rec_sep_route_SEP_SCORE"}
    for i, bucket in enumerate(ordered[1:], start=1):
        rename[bucket] = f"rec_sep_route_SEP_SCORE.{i}"
    return wide.rename(columns=rename)


def _fold_coverage_schemes(df, schemes):
    """Pattern-A numerator/denominator (YPR*Routes, Routes) summed across the given
    coverage schemes' per-game columns. Returns ``(num, den)`` Series, or
    ``(None, None)`` when no scheme column is present.
    """
    num = None
    den = None
    for scheme in schemes:
        ypr_col = f"playerStatsCoverageScheme{scheme}ReceivingYardsPerRoute"
        routes_col = f"playerStatsCoverageScheme{scheme}ReceivingRoutesTotal"
        if ypr_col not in df.columns or routes_col not in df.columns:
            continue
        ypr = pd.to_numeric(df[ypr_col], errors="coerce")
        routes = pd.to_numeric(df[routes_col], errors="coerce")
        contrib_num = (ypr * routes).where(routes > 0, other=0)
        contrib_den = routes.where(routes > 0, other=0)
        num = contrib_num if num is None else num.add(contrib_num, fill_value=0)
        den = contrib_den if den is None else den.add(contrib_den, fill_value=0)
    return num, den


def _combine_bucket_aggregates(by_player_id: pd.DataFrame, windows) -> pd.DataFrame:
    """combine_first the five per-bucket aggregate frames (each a disjoint column
    family) onto the recipe base, in their fixed precedence order.
    """
    for aggregate in (
        _aggregate_separation_by_routes,
        _aggregate_per_coverage_yprr,
        _aggregate_separation_by_coverage,
        _aggregate_separation_by_alignment_bucket,
        _aggregate_man_vs_zone_bucket,
    ):
        frame = aggregate(windows)
        if not frame.empty:
            by_player_id = by_player_id.combine_first(frame)
    return by_player_id


def _finalize_player_frame(by_player_id: pd.DataFrame, windows) -> pd.DataFrame:
    """Join metadata + game counts, project to the name index, derive metrics, filter
    to comp positions, broadcast team coverage, and combine the PBP frame.
    """
    metadata = _player_metadata(windows)
    if metadata.empty:
        return pd.DataFrame()
    by_player_id = by_player_id.join(metadata, how="left")

    games_played = _player_game_counts(windows)
    if not games_played.empty:
        by_player_id["G"] = games_played

    by_player_id = _project_to_name_index(by_player_id)
    if by_player_id.empty:
        return by_player_id

    by_player_id = _derive_aggregate_metrics(by_player_id)
    if "POS" in by_player_id.columns:
        by_player_id.loc[by_player_id["POS"].isin(_RB_ROLE_ALIASES), "POS"] = "RB"
        by_player_id = by_player_id.loc[by_player_id["POS"].isin(_COMP_POSITIONS)]
    by_player_id = by_player_id.rename(columns={"POS": "position", "G": "player_game_count"})
    by_player_id = _broadcast_team_coverage(by_player_id, windows)
    by_player_id = derive_comp_metrics(by_player_id)

    target_season = max(w[0] for w in windows)
    pbp = _load_pbp_named_by_player(target_season)
    if pbp is not None and not by_player_id.empty:
        by_player_id = by_player_id.combine_first(pbp)

    return _join_nfl_players(by_player_id)


def _aggregate_separation_by_routes(
    windows: Sequence[tuple[int, int, int]],
) -> pd.DataFrame:
    """Per-route SEP score per player, output as ``rec_sep_route_SEP_SCORE.*`` columns.

    The season CSV publishes per-route SEP buckets as
    ``rec_sep_route_SEP_SCORE``, ``.1``, ``.2`` ... ``.N`` auto-suffixed
    columns. The FP weekly parquet packs the route-stratified SEP data
    inside the JSON-encoded ``bucket`` column (one game row carries one
    JSON cell mapping route names to per-route stat dicts) rather than
    exposing it as top-level fields. This helper parses the JSON cell-
    by-cell, explodes one long-form row per (player, game, route_bucket),
    then computes the volume-weighted mean per (player, route_bucket)
    across every pooled window before pivoting wide. A player with
    games in two seasons gets one per-bucket aggregate -- the "blend"
    semantics the Phase-1.5 pre-cutoff rule asks for.

    Sparse-by-design: each game's JSON only carries route buckets the
    player actually ran. Empty-stats buckets (missing
    ``playerStatsReceivingSeparationScorePercentage``) are skipped at
    row explosion time, leaving the wide-pivot column NaN.
    """
    df = _load_kind_multi(windows, _SEP_BY_ROUTES_KIND)
    if df.empty or "bucket" not in df.columns or PLAYER_GROUP_COL not in df.columns:
        return pd.DataFrame()
    long_rows = _explode_route_long_rows(df, _SEP_BY_ROUTES_VALUE_KEY, _SEP_BY_ROUTES_WEIGHT_KEY)
    if not long_rows:
        return pd.DataFrame()
    wide = _weighted_mean_wide(long_rows, "route_bucket")
    if wide.empty:
        return wide
    return _order_and_suffix_routes(wide)


def _aggregate_per_coverage_yprr(
    windows: Sequence[tuple[int, int, int]],
) -> pd.DataFrame:
    """Per-(player, coverage-bucket) YPRR from ``wr_coverage_matchup`` snapshots.

    Emits two columns:

    * ``rec_yprr_vs_man`` -- season-to-date Yards Per Route Run when faced
      with man coverage (Pattern A weighted rate over the per-game
      ``ReceivingYardsPerRoute`` rate, weighted by ``ReceivingRoutesTotal``).
    * ``rec_yprr_vs_zone`` -- analogous rate bucketing
      Cover 2 / 3 / 4 / 6 together. The FP per-game file exposes one column
      per scheme; this helper folds the four zone schemes into a single
      Pattern-A rate (sum of YPR * Routes across the zone schemes, divided
      by sum of Routes across the same).

    Bucketing Cover2/3/4/6 into zone follows the Phase 3+4 evaluation brief:
    the per-scheme cells get sparse on low-route players, but the
    man-vs-zone split is the load-bearing decomposition for the
    pass-catcher matchup signal.
    """
    df = _load_kind_multi(windows, "wr_coverage_matchup")
    if df.empty or PLAYER_GROUP_COL not in df.columns:
        return pd.DataFrame()

    pieces: dict[str, pd.Series] = {}
    for bucket_name, schemes in _COVERAGE_YPRR_SCHEMES.items():
        num, den = _fold_coverage_schemes(df, schemes)
        if num is None or den is None:
            continue
        work = pd.DataFrame(
            {PLAYER_GROUP_COL: df[PLAYER_GROUP_COL], "num": num, "den": den}
        ).dropna(subset=[PLAYER_GROUP_COL])
        grouped = work.groupby(PLAYER_GROUP_COL)
        num_sum = grouped["num"].sum(min_count=1)
        den_sum = grouped["den"].sum(min_count=1)
        pieces[f"rec_yprr_vs_{bucket_name}"] = num_sum.divide(den_sum).where(
            den_sum > 0, other=np.nan
        )

    if not pieces:
        return pd.DataFrame()
    return pd.DataFrame(pieces)


def _aggregate_separation_by_coverage(
    windows: Sequence[tuple[int, int, int]],
) -> pd.DataFrame:
    """Per-(player, coverage-bucket) separation score from sep-by-coverage snapshots.

    The FP ``receiving_separation_by_coverage`` parquet packs per-scheme
    separation stats inside a JSON-encoded ``bucket`` column rather than as
    top-level fields, mirroring the route-bucketed shape ``_aggregate_
    separation_by_routes`` handles. This helper parses the JSON cell-by-
    cell for every key in :data:`_SEP_BY_COVERAGE_SCHEMES` (Man / Zone are
    FP-pre-aggregated rollups; RedZone is a situational slice; Cover 2/3/4/6
    are per-shell un-foldings), computes the routes-weighted mean per
    (player, scheme), and emits one ``rec_sep_vs_{name}`` column per scheme.

    Skips inner buckets missing the value or weight key so partially-empty
    rows don't poison the aggregation.
    """
    df = _load_kind_multi(windows, _SEP_BY_COVERAGE_KIND)
    if df.empty or "bucket" not in df.columns or PLAYER_GROUP_COL not in df.columns:
        return pd.DataFrame()
    long_rows = _explode_scheme_long_rows(
        df,
        _SEP_BY_COVERAGE_SCHEMES,
        _SEP_BY_COVERAGE_VALUE_KEY,
        _SEP_BY_COVERAGE_WEIGHT_KEY,
        "scheme",
    )
    if not long_rows:
        return pd.DataFrame()
    wide = _weighted_mean_wide(long_rows, "scheme")
    if wide.empty:
        return wide
    return wide.rename(columns={name: f"rec_sep_vs_{name}" for name in _SEP_BY_COVERAGE_SCHEMES})


def _aggregate_separation_by_alignment_bucket(
    windows: Sequence[tuple[int, int, int]],
) -> pd.DataFrame:
    """Per-(player, alignment) separation SEP_SCORE + WIN_RATE from bucket JSON.

    The top-level ``receiving_separation_by_alignment`` row already produces
    ``rec_sep_align_overall_SEP_SCORE`` and ``rec_sep_align_overall_WIN_RATE``
    via the recipe table. The per-alignment buckets (Wide / Slot / Inline /
    Backfield) live inside the JSON ``bucket`` column; this helper parses
    them and emits ``rec_sep_align_{ALIGN}_SEP_SCORE`` +
    ``rec_sep_align_{ALIGN}_WIN_RATE`` per alignment.

    Same per-bucket weighted-mean shape as
    :func:`_aggregate_separation_by_coverage`; uses the routes-total
    inner key as the weight. ``Overall`` is intentionally not in
    :data:`_SEP_BY_ALIGNMENT_BUCKETS` -- duplicates the top-level recipe.
    """
    df = _load_kind_multi(windows, _SEP_BY_ALIGNMENT_KIND)
    if df.empty or "bucket" not in df.columns or PLAYER_GROUP_COL not in df.columns:
        return pd.DataFrame()

    long_rows: list[dict[str, object]] = []
    for player_id, payload in _iter_bucket_payloads(df):
        for align_name, key in _SEP_BY_ALIGNMENT_BUCKETS.items():
            inner = payload.get(key)
            if not isinstance(inner, dict):
                continue
            sep_value = inner.get(_SEP_BY_ALIGNMENT_SEP_KEY)
            win_value = inner.get(_SEP_BY_ALIGNMENT_WIN_KEY)
            weight = inner.get(_SEP_BY_ALIGNMENT_WEIGHT_KEY)
            if weight is None:
                continue
            long_rows.append(
                {
                    PLAYER_GROUP_COL: player_id,
                    "alignment": align_name,
                    "sep_value": sep_value,
                    "win_value": win_value,
                    "weight": weight,
                }
            )
    if not long_rows:
        return pd.DataFrame()

    work = pd.DataFrame(long_rows)
    work["sep_value"] = pd.to_numeric(work["sep_value"], errors="coerce")
    work["win_value"] = pd.to_numeric(work["win_value"], errors="coerce")
    work["weight"] = pd.to_numeric(work["weight"], errors="coerce")
    work = work.dropna(subset=["weight"])
    if work.empty:
        return pd.DataFrame()

    grouped = work.groupby([PLAYER_GROUP_COL, "alignment"], dropna=False)
    weight_sum = grouped["weight"].sum(min_count=1)
    sep_num = grouped.apply(
        lambda g: (g["sep_value"] * g["weight"]).sum(min_count=1),
        include_groups=False,
    )
    win_num = grouped.apply(
        lambda g: (g["win_value"] * g["weight"]).sum(min_count=1),
        include_groups=False,
    )
    sep_wide = sep_num.divide(weight_sum).where(weight_sum != 0, other=np.nan).unstack("alignment")
    win_wide = win_num.divide(weight_sum).where(weight_sum != 0, other=np.nan).unstack("alignment")
    sep_wide = sep_wide.rename(
        columns={a: f"rec_sep_align_{a}_SEP_SCORE" for a in _SEP_BY_ALIGNMENT_BUCKETS}
    )
    win_wide = win_wide.rename(
        columns={a: f"rec_sep_align_{a}_WIN_RATE" for a in _SEP_BY_ALIGNMENT_BUCKETS}
    )
    return sep_wide.join(win_wide, how="outer")


def _aggregate_man_vs_zone_bucket(
    windows: Sequence[tuple[int, int, int]],
) -> pd.DataFrame:
    """Per-(player, coverage-shell) YPRR from receiving_man_vs_zone.bucket JSON.

    The top-level row already produces ``rec_mz_YPRR_overall`` via the
    recipe table. The per-shell buckets (Man / Zone / SingleHigh / TwoHigh)
    live inside the JSON ``bucket`` column; each inner dict exposes the
    per-game ``AveragesPerRouteYardsTotal`` rate and the
    ``ReceivingRoutesTotal`` denominator (see
    :data:`_MZ_VALUE_KEY` / :data:`_MZ_WEIGHT_KEY`). Helper emits
    ``rec_mz_YPRR_{MAN,ZONE,SINGLEHIGH,TWOHIGH}`` per player.

    Same per-bucket weighted-mean shape as the by-coverage / by-alignment
    helpers above. ``Overall`` is intentionally not in
    :data:`_MZ_BUCKETS` -- duplicates the top-level recipe.

    Independent from the ``rec_mz_YPRR.1`` / ``rec_mz_YPRR.2`` placeholder
    pair that :func:`_derive_aggregate_metrics` still emits: those placeholders
    prevent ``derive_comp_metrics`` from short-circuiting on a missing
    ``man_yprr_diff`` / ``zone_yprr_diff`` input when per-bucket data is sparse;
    this helper adds the real per-shell rates alongside them.
    """
    df = _load_kind_multi(windows, _MZ_BUCKET_KIND)
    if df.empty or "bucket" not in df.columns or PLAYER_GROUP_COL not in df.columns:
        return pd.DataFrame()
    long_rows = _explode_scheme_long_rows(
        df,
        _MZ_BUCKETS,
        _MZ_VALUE_KEY,
        _MZ_WEIGHT_KEY,
        "shell",
    )
    if not long_rows:
        return pd.DataFrame()
    wide = _weighted_mean_wide(long_rows, "shell")
    if wide.empty:
        return wide
    return wide.rename(columns={s: f"rec_mz_YPRR_{s}" for s in _MZ_BUCKETS})


def _player_metadata(windows: Sequence[tuple[int, int, int]]) -> pd.DataFrame:
    """Carry First/Last/POS/Team from the most-recent snapshot in the window set.

    Picks the latest (season, week) each player appeared across every
    pooled window; mid-season trades resolve to the player's last-known
    team rather than the season-opener.
    """
    # offense_snaps is the highest-coverage kind, with one row per player who
    # took a single offensive snap. receiving_basic is the fallback for
    # weeks where snap data didn't fetch.
    for kind in ("offense_snaps", "receiving_basic", "passing_advanced", "rushing_basic"):
        df = _load_kind_multi(windows, kind)
        if df.empty or PLAYER_GROUP_COL not in df.columns:
            continue
        if not all(col in df.columns for col in _PLAYER_METADATA_COLS):
            continue
        if "gameWeek" not in df.columns or "gameSeason" not in df.columns:
            continue
        sorted_df = df.sort_values(["gameSeason", "gameWeek"], kind="stable")
        latest = sorted_df.drop_duplicates(subset=PLAYER_GROUP_COL, keep="last")
        meta = latest.set_index(PLAYER_GROUP_COL)[list(_PLAYER_METADATA_COLS)].copy()
        meta.columns = ["First", "Last", "POS", "Team"]
        return meta
    return pd.DataFrame()


def _player_game_counts(windows: Sequence[tuple[int, int, int]]) -> pd.Series:
    """Per-player count of games appeared across every pooled lookback window."""
    df = _load_kind_multi(windows, "offense_snaps")
    if df.empty or PLAYER_GROUP_COL not in df.columns:
        return pd.Series(dtype="int64")
    games = df.groupby(PLAYER_GROUP_COL).size()
    games.name = "G"
    return games


def _project_to_name_index(by_player_id: pd.DataFrame) -> pd.DataFrame:
    """Re-key from playerPlayerId to accent-stripped ``First Last`` to match load_one_year."""
    if "First" not in by_player_id.columns or "Last" not in by_player_id.columns:
        return pd.DataFrame()
    name = (by_player_id["First"].astype(str) + " " + by_player_id["Last"].astype(str)).apply(
        remove_accents
    )
    by_player_id = by_player_id.assign(Name=name)
    by_player_id = by_player_id.dropna(subset=["Name"])
    by_player_id.index = by_player_id["Name"]
    by_player_id.index.name = None
    # Drop intermediate name columns; ``Name`` survives for downstream
    # ``build_comp_profile`` re-indexing.
    by_player_id = by_player_id.drop(columns=["First", "Last"])
    return by_player_id.loc[~by_player_id.index.duplicated(keep="last")]


def _derive_aggregate_metrics(profile: pd.DataFrame) -> pd.DataFrame:
    """Compute season-CSV columns that need a multi-input formula.

    Three reconstructions land here -- the rest of the recipe table maps
    1:1 to a single aggregation pattern:

    * ``pass_adv_ANY_A`` -- Adjusted Net Yards per Attempt, standard
      pro-football-reference formula over season sums.
    * ``rec_adv_YPRR`` aliasing into ``rec_mz_YPRR.1`` / ``.2`` -- the
      man/zone split needs a per-bucket aggregation that hasn't been
      threaded through the recipe table yet; for v1 we leave the
      derive_comp_metrics ``man_yprr_diff`` / ``zone_yprr_diff``
      branches NaN. They aren't in the current ``playerCompStats.json``
      config so the gate isn't load-bearing.
    * ``qb_cov_COV_GRADE`` -- FP doesn't expose a coverage-grade
      equivalent at game grain; left NaN.

    Args:
        profile: Wide frame indexed by accent-stripped player name with
            the recipe-table outputs already populated.

    Returns:
        ``profile`` with the additional formula-derived columns merged
        in. Missing inputs leave the derived column as NaN rather than
        raising.
    """
    yards = profile.get("pass_adv_YDS")
    sack_yards = profile.get("pass_adv_SACK_YDS")
    tds = profile.get("pass_adv_TD")
    ints = profile.get("pass_adv_INT")
    attempts = profile.get("pass_adv_ATT")
    sacks = profile.get("pass_adv_SACK")
    if all(s is not None for s in (yards, sack_yards, tds, ints, attempts, sacks)):
        num = yards - sack_yards + _ANY_A_TD_BONUS * tds - _ANY_A_INT_PENALTY * ints
        den = attempts + sacks
        profile["pass_adv_ANY_A"] = num.divide(den).where(den > 0, other=np.nan)

    overall_yprr = profile.get("rec_mz_YPRR_overall")
    if overall_yprr is not None:
        # Without per-bucket split, ``YPRR.1`` (man) and ``.2`` (zone)
        # fall back to the overall as a non-informative placeholder so
        # downstream ``man_yprr_diff`` / ``zone_yprr_diff`` evaluate to
        # zero rather than NaN. They aren't in the current comp
        # config; the placeholder keeps derive_comp_metrics from
        # short-circuiting on a missing input.
        profile["rec_mz_YPRR.1"] = overall_yprr
        profile["rec_mz_YPRR.2"] = overall_yprr

    if "rec_sep_align_overall_SEP_SCORE" in profile.columns:
        profile["rec_sep_align_SEP_SCORE"] = profile["rec_sep_align_overall_SEP_SCORE"]
    if "rec_sep_align_overall_WIN_RATE" in profile.columns:
        profile["rec_sep_align_WIN_RATE"] = profile["rec_sep_align_overall_WIN_RATE"]

    if "qb_cov_QB_MAN_pct" in profile.columns:
        # FP weekly doesn't expose an equivalent of the FP-season-CSV
        # "COV GRADE"; leave NaN and the downstream fillna(0) absorbs it.
        profile["qb_cov_COV_GRADE"] = np.nan

    # TE block-pct proxy: pass snaps the player did NOT run a route on,
    # normalized by total offensive snaps. The legacy season-CSV path
    # at ``nfl_pbp_agg._te_block_pct_proxy`` computed the same ratio
    # from two CSVs that the FP weekly drop replaced; keeping the
    # ``block_pct_proxy`` name preserves the rolling-mean transform in
    # ``nfl_fp_loader.TRANSFORM_FEATURES`` (otherwise the comp config
    # weight on this column would NaN out on every TE row).
    snaps = profile.get("off_snaps_TOTAL")
    routes = profile.get("rec_adv_RTE")
    if snaps is not None and routes is not None:
        diff = (snaps - routes.fillna(0)).clip(lower=0)
        profile["block_pct_proxy"] = diff.divide(snaps).where(snaps > 0, other=np.nan)

    return profile


__all__ = (
    "load_multi_window_one_year",
    "load_through_one_year",
    "load_window_one_year",
)
