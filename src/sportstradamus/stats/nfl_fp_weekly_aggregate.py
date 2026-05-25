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

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sportstradamus.helpers.text import remove_accents
from sportstradamus.spiderLogger import logger
from sportstradamus.stats import nfl_fp_weekly
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
    _Recipe(
        "pass_adv_DB", "passing_advanced", "sum", ("playerStatsPassingDropbacksTotal",)
    ),
    _Recipe(
        "pass_adv_SCRM", "passing_advanced", "sum", ("playerStatsPassingScramblesTotal",)
    ),
    _Recipe("pass_adv_ATT", "passing_advanced", "sum", ("playerStatsPassingAttemptsTotal",)),
    _Recipe("pass_adv_YDS", "passing_advanced", "sum", ("playerStatsPassingYardsTotal",)),
    _Recipe("pass_adv_TD", "passing_advanced", "sum", ("playerStatsPassingTouchdownsTotal",)),
    _Recipe(
        "pass_adv_INT", "passing_advanced", "sum", ("playerStatsPassingInterceptionsTotal",)
    ),
    _Recipe(
        "pass_adv_SACK", "passing_advanced", "sum", ("playerStatsPassingSackedTotal",)
    ),
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
        "pass_adv_aDOT",
        "passing_advanced",
        "weighted_mean",
        (
            "playerStatsPassingAverageDepthOfTarget",
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
    # rushing_basic -- RB volume input
    _Recipe(
        "rush_basic_ATT",
        "rushing_basic",
        "sum",
        ("playerStatsRushingAttemptsTotal",),
    ),
    # rushing_advanced -- RB efficiency inputs
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
        "rush_adv_STUFF_pct",
        "rushing_advanced",
        "weighted_mean",
        (
            "playerStatsRushingAttemptsStuffsPercentage",
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
    # receiving_basic -- volume input shared with RB derivation
    _Recipe(
        "rec_basic_REC",
        "receiving_basic",
        "sum",
        ("playerStatsReceivingReceptionsTotal",),
    ),
    # receiving_advanced -- WR / TE / RB primary file
    _Recipe(
        "rec_adv_RTE", "receiving_advanced", "sum", ("playerStatsReceivingRoutesTotal",)
    ),
    _Recipe(
        "rec_adv_TGT", "receiving_advanced", "sum", ("playerStatsReceivingTargetsTotal",)
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
    # efficiency -- WR / RB fantasy efficiency
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
    # qb_coverage_matchup -- QB-side man-coverage exposure. Per-game
    # rows in this file are (QB, opponent, week); for the comp profile
    # we want a per-QB season-to-date weighted mean across all matchups.
    _Recipe(
        "qb_cov_QB_MAN_pct",
        "qb_coverage_matchup",
        "weighted_mean",
        (_QB_COV_MAN_PCT_COL, _QB_COV_DROPBACKS_COL),
    ),
)

# Recipes that share a denominator with the season-CSV's separation-by-
# routes file. Aggregating per (player, route_bucket) and meaning the
# SEP_SCORE across routes gives the per-player ``rec_sep_route_SEP_SCORE``
# family the season CSV exposes as ``.0`` ... ``.6`` auto-suffixed cols.
_SEP_BY_ROUTES_KIND = "receiving_separation_by_routes"
_SEP_BY_ROUTES_VALUE_COL = "playerStatsReceivingSeparationByScoreStep"
_SEP_BY_ROUTES_WEIGHT_COL = "playerStatsReceivingSeparationRoutesTotal"

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

    sep_routes = _aggregate_separation_by_routes(windows)
    if not sep_routes.empty:
        by_player_id = by_player_id.combine_first(sep_routes)

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
    by_player_id = derive_comp_metrics(by_player_id)

    target_season = max(w[0] for w in windows)
    pbp = _load_pbp_named_by_player(target_season)
    if pbp is not None and not by_player_id.empty:
        by_player_id = by_player_id.combine_first(pbp)

    by_player_id = _join_nfl_players(by_player_id)
    return by_player_id


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


def _load_kind_multi(
    windows: Sequence[tuple[int, int, int]], file_kind: str
) -> pd.DataFrame:
    """Concatenate per-game rows from every (season, start, end) window for ``file_kind``."""
    frames = []
    for season, start_week, end_week in windows:
        df = _load_kind(season, start_week, end_week, file_kind)
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


def _aggregate_separation_by_routes(
    windows: Sequence[tuple[int, int, int]],
) -> pd.DataFrame:
    """Per-route SEP score per player, output as ``rec_sep_route_SEP_SCORE.*`` columns.

    The season CSV publishes per-route SEP buckets as
    ``rec_sep_route_SEP_SCORE``, ``.1``, ``.2`` ... ``.N`` auto-suffixed
    columns. The weekly parquet stratifies routes via the ``bucket``
    column; this helper pivots them so the downstream
    ``derive_comp_metrics`` per-route mean
    (``sep_routes_mean = mean(rec_sep_route_SEP_SCORE*)``) yields the
    same number as the season-CSV path. Pools per-game rows across every
    window before pivoting -- a player with games in two seasons gets
    one per-bucket aggregate.
    """
    df = _load_kind_multi(windows, _SEP_BY_ROUTES_KIND)
    if df.empty:
        return pd.DataFrame()
    if "bucket" not in df.columns or _SEP_BY_ROUTES_VALUE_COL not in df.columns:
        return pd.DataFrame()

    work = df.dropna(subset=["bucket", _SEP_BY_ROUTES_VALUE_COL])
    if work.empty:
        return pd.DataFrame()

    grouped = work.groupby([PLAYER_GROUP_COL, "bucket"], dropna=False)
    weighted_num = grouped.apply(
        lambda g: (g[_SEP_BY_ROUTES_VALUE_COL] * g[_SEP_BY_ROUTES_WEIGHT_COL]).sum(
            min_count=1
        ),
        include_groups=False,
    )
    weighted_den = grouped[_SEP_BY_ROUTES_WEIGHT_COL].sum(min_count=1)
    per_bucket = weighted_num.divide(weighted_den).where(weighted_den != 0, other=np.nan)
    wide = per_bucket.unstack("bucket")
    if wide.empty:
        return wide

    # Match the season-CSV's ``.0``, ``.1`` ... ``.N`` auto-suffix scheme:
    # first bucket gets no suffix, subsequent buckets ``.1`` ``.2`` ...
    columns = list(wide.columns)
    rename: dict[object, str] = {columns[0]: "rec_sep_route_SEP_SCORE"}
    for i, bucket in enumerate(columns[1:], start=1):
        rename[bucket] = f"rec_sep_route_SEP_SCORE.{i}"
    return wide.rename(columns=rename)


def _load_kind(
    season: int, start_week: int, end_week: int, file_kind: str
) -> pd.DataFrame:
    """Read one file_kind for the closed week range, log + swallow loader errors.

    Normalises the loader's ``DataFrame | None`` return to an empty
    DataFrame so downstream recipe dispatch doesn't have to special-case
    the None branch (every recipe already short-circuits on ``empty``).
    """
    try:
        df = nfl_fp_weekly.load_window(season, start_week, end_week, file_kind)
    except Exception as exc:
        logger.warning(
            "load_window(%s, %s, %s, %s) failed: %s",
            season,
            start_week,
            end_week,
            file_kind,
            exc,
        )
        return pd.DataFrame()
    return df if df is not None else pd.DataFrame()


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
    by_player_id = by_player_id.loc[~by_player_id.index.duplicated(keep="last")]
    return by_player_id


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

    return profile


__all__ = (
    "load_through_one_year",
    "load_window_one_year",
    "load_multi_window_one_year",
)
