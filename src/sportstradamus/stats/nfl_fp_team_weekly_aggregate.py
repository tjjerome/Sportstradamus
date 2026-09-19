"""Phase-2 bridge: aggregate weekly FP team-snapshots into teamProfile / defenseProfile features.

Sibling to :mod:`sportstradamus.stats.nfl_fp_weekly_aggregate` at team grain.
The recipe table here encodes, per output column, which team-grain file_kind
to read and which aggregation pattern reduces the per-game rows into a
season-to-date team feature. Both ``base_profile`` (training) and any
inference-time consumer route reads through :func:`load_team_and_defense_features`
so the training-vs-inference aggregation strategies can't drift.

Pattern dispatches handled by this module:

* ``weighted_rate`` -- Pattern A reduction via
  :func:`sportstradamus.stats.nfl_fp_aggregation.weighted_rate` at team grain
  (``sum(num) / sum(den)`` per teamTeamId across the window).
* ``weighted_mean`` -- Pattern A variant via
  :func:`sportstradamus.stats.nfl_fp_aggregation.weighted_mean` when the
  per-game row exposes only the rate (e.g. success%, stuffs%).
* ``proe_formula`` -- custom Pattern A. PROE is
  ``sum(actual_dropbacks - expected_dropbacks) / sum(expected_dropbacks)``;
  the per-game ratio is biased.
* ``rpr_bucket_pass_rate`` -- JSON-parse ``run_pass_report``'s ``bucket``
  column to extract per-situation snap counts, then compute pass% as
  ``sum(pass_snaps) / (sum(pass_snaps) + sum(rush_snaps))`` per team.

Pattern C (per-opponent x position fantasy_points_allowed) is **deferred
to Phase 2b** -- the FP fetcher currently writes a 1-row season-to-date
stub per snapshot (flagged in
:data:`sportstradamus.stats.nfl_fp_weekly.PLACEHOLDER_SINGLE_TILE_KINDS`).
Once the endpoint is parameterized properly, add a recipe to this
module routed through
:func:`sportstradamus.stats.nfl_fp_aggregation.game_mean` and re-key on
``(opponent, position)``.

Output: two abbreviation-indexed DataFrames suitable for joining into
``teamProfile`` / ``defenseProfile``. Missing-team rows are absent from
the result; :func:`sportstradamus.stats.base._profile_rows_for_teams`
reindexes them as all-NaN at consumer side.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from sportstradamus.spiderLogger import logger
from sportstradamus.stats import nfl_fp_team_weekly, nfl_fp_weekly
from sportstradamus.stats.nfl_fp_aggregation import (
    TEAM_GROUP_COL,
    weighted_mean,
    weighted_rate,
)

# Abbreviation column the FP team-grain parquets carry alongside teamTeamId.
# Used to re-key each window's rows onto the abbreviation index that
# teamProfile / defenseProfile uses (matches the NFL gamelog's "team" col).
_TEAM_ABBR_COL = "teamAbbreviation"

# PROE source columns. PFR-style formula:
# ``sum(actual_dropbacks - expected_dropbacks) / sum(expected_dropbacks)``.
# The per-game ratio is biased; the sum form is the season-to-date PROE.
_PROE_ACTUAL_COL = "teamStatsPassingDropbacksTotal"
_PROE_EXPECTED_COL = "teamStatsPassingDropbacksExpected"

# run_pass_report snap columns. In the archived legacy snapshots each
# per-game row's ``bucket`` cell is a JSON object keyed by bucket name whose
# nested entries carry these two; the current API answers one filtered
# request per bucket and puts them at the top level of its own kind.
_RPR_BUCKET_PASS = "teamStatsSnapsOffensePass"
_RPR_BUCKET_RUSH = "teamStatsSnapsOffenseRush"
_RPR_BUCKET_TOTAL = "teamStatsSnapsOffenseTotal"

# Spec-aligned subset of FP's situational buckets. ``Inside20`` ≈ redzone,
# ``Inside10`` ≈ goalline; Leading / Trailing capture game-state pass
# tendency; FirstDown / ThirdDown capture down-state. Other available
# buckets (FirstHalf/SecondHalf, Under5/5To9/Over10) are skipped here as
# largely collinear with the chosen subset. Each entry names the output
# suffix, the legacy nested-bucket key, and the per-bucket file kind the
# current API's filtered pulls land in.
_RPR_BUCKETS: tuple[tuple[str, str, str], ...] = (
    ("redzone", "bucketInside20", "run_pass_report_redzone"),
    ("goalline", "bucketInside10", "run_pass_report_goalline"),
    ("neutral", "bucketNeutral", "run_pass_report_neutral"),
    ("leading", "bucketLeading", "run_pass_report_leading"),
    ("trailing", "bucketTrailing", "run_pass_report_trailing"),
    ("first_down", "bucketFirstDown", "run_pass_report_first_down"),
    ("third_down", "bucketThirdDown", "run_pass_report_third_down"),
)


@dataclass(frozen=True)
class _TeamRecipe:
    """One output column derived from one team-grain file_kind via one aggregation pattern.

    Attributes:
        output_col: Column name in the aggregated frame (no Team / Defense
            prefix -- the prefix is added at base_profile join time via the
            existing ``add_prefix("Team ")`` / ``add_prefix("Defense ")``
            pipeline in stats/base.py).
        file_kind: Logical kind name from
            :data:`sportstradamus.stats.nfl_fp_team_weekly.FILE_KINDS`.
        pattern: Dispatch key: ``"weighted_rate"`` / ``"weighted_mean"`` /
            ``"proe_formula"`` / ``"rpr_bucket_pass_rate"``.
        args: Source-column tuple. ``(num, den)`` for weighted_rate /
            ``proe_formula``; ``(value, weight)`` for weighted_mean;
            ``(bucket_key,)`` for rpr_bucket_pass_rate.
        grain: ``"team"`` -> output goes into team_features frame.
            ``"defense"`` -> output goes into defense_features frame.
    """

    output_col: str
    file_kind: str
    pattern: str
    args: tuple
    grain: str


# Recipes that reference a column absent from the snapshot's schema are
# silently skipped in _dispatch; this lets the table stay optimistic without
# crashing on per-season schema drift.
_RECIPES: tuple[_TeamRecipe, ...] = (
    # coverage_matrix -> teamProfile (off scheme exposure)
    _TeamRecipe(
        "off_faced_man_pct",
        "coverage_matrix",
        "weighted_rate",
        (
            "teamStatsCoverageSchemeManPassingDropbacksTotal",
            "teamStatsPassingDropbacksTotal",
        ),
        "team",
    ),
    _TeamRecipe(
        "off_faced_zone_pct",
        "coverage_matrix",
        "weighted_rate",
        (
            "teamStatsCoverageSchemeZonePassingDropbacksTotal",
            "teamStatsPassingDropbacksTotal",
        ),
        "team",
    ),
    _TeamRecipe(
        "off_faced_single_high_pct",
        "coverage_matrix",
        "weighted_rate",
        (
            "teamStatsCoverageSchemeSingleHighPassingDropbacksTotal",
            "teamStatsPassingDropbacksTotal",
        ),
        "team",
    ),
    _TeamRecipe(
        "off_faced_two_high_pct",
        "coverage_matrix",
        "weighted_rate",
        (
            "teamStatsCoverageSchemeTwoHighPassingDropbacksTotal",
            "teamStatsPassingDropbacksTotal",
        ),
        "team",
    ),
    # coverage_matrix_opp -> defenseProfile (def scheme rate)
    _TeamRecipe(
        "def_man_pct",
        "coverage_matrix_opp",
        "weighted_rate",
        (
            "opponentStatsCoverageSchemeManPassingDropbacksTotal",
            "opponentStatsPassingDropbacksTotal",
        ),
        "defense",
    ),
    _TeamRecipe(
        "def_zone_pct",
        "coverage_matrix_opp",
        "weighted_rate",
        (
            "opponentStatsCoverageSchemeZonePassingDropbacksTotal",
            "opponentStatsPassingDropbacksTotal",
        ),
        "defense",
    ),
    _TeamRecipe(
        "def_single_high_pct",
        "coverage_matrix_opp",
        "weighted_rate",
        (
            "opponentStatsCoverageSchemeSingleHighPassingDropbacksTotal",
            "opponentStatsPassingDropbacksTotal",
        ),
        "defense",
    ),
    _TeamRecipe(
        "def_two_high_pct",
        "coverage_matrix_opp",
        "weighted_rate",
        (
            "opponentStatsCoverageSchemeTwoHighPassingDropbacksTotal",
            "opponentStatsPassingDropbacksTotal",
        ),
        "defense",
    ),
    # rushing_advanced -> teamProfile rush efficiency
    _TeamRecipe(
        "rush_yaco_per_att",
        "rushing_advanced",
        "weighted_rate",
        (
            "teamStatsRushingYardsAfterContactTotal",
            "teamStatsRushingAttemptsTotal",
        ),
        "team",
    ),
    _TeamRecipe(
        "rush_ybc_per_att",
        "rushing_advanced",
        "weighted_rate",
        (
            "teamStatsRushingYardsBeforeContactTotal",
            "teamStatsRushingAttemptsTotal",
        ),
        "team",
    ),
    _TeamRecipe(
        "rush_success_pct",
        "rushing_advanced",
        "weighted_mean",
        (
            "teamStatsRushingAttemptsSuccessPercentage",
            "teamStatsRushingAttemptsTotal",
        ),
        "team",
    ),
    _TeamRecipe(
        "rush_stuff_pct",
        "rushing_advanced",
        "weighted_mean",
        (
            "teamStatsRushingAttemptsStuffsPercentage",
            "teamStatsRushingAttemptsTotal",
        ),
        "team",
    ),
    _TeamRecipe(
        "rush_explosive_pct",
        "rushing_advanced",
        "weighted_mean",
        (
            "teamStatsRushingRunsExplosivePercentage",
            "teamStatsRushingAttemptsTotal",
        ),
        "team",
    ),
    # rushing_advanced_opp -> defenseProfile rush defense
    _TeamRecipe(
        "def_rush_yaco_allowed_per_att",
        "rushing_advanced_opp",
        "weighted_rate",
        (
            "opponentStatsRushingYardsAfterContactTotal",
            "opponentStatsRushingAttemptsTotal",
        ),
        "defense",
    ),
    _TeamRecipe(
        "def_rush_ybc_allowed_per_att",
        "rushing_advanced_opp",
        "weighted_rate",
        (
            "opponentStatsRushingYardsBeforeContactTotal",
            "opponentStatsRushingAttemptsTotal",
        ),
        "defense",
    ),
    _TeamRecipe(
        "def_rush_success_allowed_pct",
        "rushing_advanced_opp",
        "weighted_mean",
        (
            "opponentStatsRushingAttemptsSuccessPercentage",
            "opponentStatsRushingAttemptsTotal",
        ),
        "defense",
    ),
    _TeamRecipe(
        "def_rush_stuff_pct",
        "rushing_advanced_opp",
        "weighted_mean",
        (
            "opponentStatsRushingAttemptsStuffsPercentage",
            "opponentStatsRushingAttemptsTotal",
        ),
        "defense",
    ),
    # proe_report -> teamProfile PROE (custom sum-form formula)
    _TeamRecipe(
        "proe",
        "proe_report",
        "proe_formula",
        (_PROE_ACTUAL_COL, _PROE_EXPECTED_COL),
        "team",
    ),
)


def _rpr_bucket_recipes() -> tuple[_TeamRecipe, ...]:
    """Two recipes per bucket -- one per era, both writing the same output column.

    The filtered per-bucket kind exists only for weeks pulled from the
    current API; the nested ``bucket`` cell only for archived legacy
    snapshots. Whichever is on disk for a given week produces the column and
    the other yields nothing, so a window of either era aggregates the same.
    """
    return tuple(
        recipe
        for suffix, bucket_key, bucket_kind in _RPR_BUCKETS
        for recipe in (
            _TeamRecipe(
                output_col=f"rp_{suffix}_pass_pct",
                file_kind=bucket_kind,
                pattern="weighted_rate",
                args=(_RPR_BUCKET_PASS, _RPR_BUCKET_TOTAL),
                grain="team",
            ),
            _TeamRecipe(
                output_col=f"rp_{suffix}_pass_pct",
                file_kind="run_pass_report",
                pattern="rpr_bucket_pass_rate",
                args=(bucket_key,),
                grain="team",
            ),
        )
    )


_ALL_RECIPES: tuple[_TeamRecipe, ...] = _RECIPES + _rpr_bucket_recipes()


def load_team_and_defense_features(
    pattern_a_windows: Sequence[tuple[int, int, int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate weekly FP team snapshots into team / defense feature frames.

    Pattern A kinds pool per-game team rows across every
    ``(season, start_week, end_week)`` window, then aggregate per the recipe.

    The windows-list contract lets the NFL caller apply the Phase-1.5
    lookback rule (post-week-4 = current-season only;
    pre-week-5 = prior season weeks 11..18 BLENDED with current-season
    partial, raw rows pooled before per-team groupby). Both branches of
    that rule express as a ``windows`` list -- the bridge stays family-
    agnostic about cutoffs and weights.

    Output frames are indexed by ``teamAbbreviation`` (e.g. ``"PHI"``,
    ``"KC"``) to match the abbreviation-indexed teamProfile / defenseProfile
    built by ``stats/base.py:base_profile``.

    Args:
        pattern_a_windows: Per-window ``(season, start_week, end_week)``
            tuples to pool for Pattern A rate-stat reads. Windows where
            ``start_week > end_week`` are skipped silently. Empty list
            short-circuits to no features.

    Returns:
        Tuple ``(team_features, defense_features)`` of DataFrames, each
        indexed by team abbreviation. Empty DataFrames if no usable
        snapshots were found.
    """
    if not pattern_a_windows:
        return pd.DataFrame(), pd.DataFrame()

    return (
        _aggregate_pattern_a(pattern_a_windows, grain="team"),
        _aggregate_pattern_a(pattern_a_windows, grain="defense"),
    )


def _recipes_by_kind(grain: str) -> dict[str, list[_TeamRecipe]]:
    by_kind: dict[str, list[_TeamRecipe]] = {}
    for recipe in _ALL_RECIPES:
        if recipe.grain != grain:
            continue
        by_kind.setdefault(recipe.file_kind, []).append(recipe)
    return by_kind


def _pool_windows(
    pattern_a_windows: Sequence[tuple[int, int, int]], file_kind: str
) -> pd.DataFrame:
    """Pool before the per-team groupby so blended windows are treated as one sample.

    Each window is re-keyed from its own season's ``teamTeamId`` to the team
    abbreviation *before* the concat: the two snapshot eras don't share an id
    space (archived legacy pulls key teams numerically, current pulls key them
    by abbreviation), so pooling the raw ids would split one team across two
    groups and silently drop whichever era the map didn't cover.

    Windows where ``start_week > end_week``, whose snapshots are absent, or
    whose season has no usable abbreviation map contribute nothing. Empty frame
    when no window yields rows.
    """
    frames = []
    for season, start_week, end_week in pattern_a_windows:
        df = nfl_fp_team_weekly.load_window_or_empty(season, start_week, end_week, file_kind)
        if df.empty:
            continue
        abbr = _build_team_abbreviation_map(season)
        if abbr.empty:
            continue
        keyed = df.assign(**{TEAM_GROUP_COL: df[TEAM_GROUP_COL].map(abbr)})
        frames.append(keyed.loc[keyed[TEAM_GROUP_COL].notna()])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _aggregate_pattern_a(
    pattern_a_windows: Sequence[tuple[int, int, int]],
    *,
    grain: Literal["team", "defense"],
) -> pd.DataFrame:
    """Apply every recipe whose ``grain`` matches over pooled per-window per-game rows.

    For each ``file_kind`` referenced by matching recipes, loads all windows and
    concatenates the raw per-game rows — abbreviation-keyed by ``_pool_windows``,
    pooled across windows before the per-team groupby so blended windows are
    treated as one sample — then dispatches to ``_apply_recipes``.

    Args:
        pattern_a_windows: ``(season, start_week, end_week)`` tuples to pool.
            Windows where ``start_week > end_week`` are skipped.
        grain: ``"team"`` -> collects team-grain recipes; ``"defense"`` -> defense-grain.

    Returns:
        DataFrame indexed by team abbreviation. Empty when no usable snapshots found.
    """
    if not pattern_a_windows:
        return pd.DataFrame()

    out = pd.DataFrame()
    for file_kind, recipes in _recipes_by_kind(grain).items():
        pooled = _pool_windows(pattern_a_windows, file_kind)
        if pooled.empty:
            continue
        kind_frame = _apply_recipes(pooled, recipes)
        if kind_frame.empty:
            continue
        out = kind_frame if out.empty else out.combine_first(kind_frame)

    return out


def _apply_recipes(df: pd.DataFrame, recipes: Sequence[_TeamRecipe]) -> pd.DataFrame:
    """Route each recipe to its dispatch function and collect results into a wide frame.

    Recipes that reference missing columns are silently skipped (``_dispatch``
    returns ``None``); the caller gets a frame with however many columns survived.

    Args:
        df: Raw pooled per-game rows for one ``file_kind``.
        recipes: Subset of ``_ALL_RECIPES`` filtered to one ``file_kind`` and ``grain``.

    Returns:
        Wide DataFrame indexed by ``TEAM_GROUP_COL``, which ``_pool_windows`` has
        already re-keyed to the team abbreviation, one column per surviving
        recipe. Empty when all recipes return ``None``.
    """
    out = pd.DataFrame()
    for recipe in recipes:
        series = _dispatch(df, recipe)
        if series is None or series.empty:
            continue
        out[recipe.output_col] = series
    return out


def _missing_cols(df: pd.DataFrame, cols: Sequence[str]) -> bool:
    return any(c not in df.columns for c in cols)


def _dispatch(df: pd.DataFrame, recipe: _TeamRecipe) -> pd.Series | None:
    """Route a recipe to the right helper -- skip if required columns missing."""
    if recipe.pattern == "weighted_rate":
        num_col, den_col = recipe.args
        if _missing_cols(df, recipe.args):
            return None
        return weighted_rate(df, num_col, den_col, group_col=TEAM_GROUP_COL)
    if recipe.pattern == "weighted_mean":
        value_col, weight_col = recipe.args
        if _missing_cols(df, recipe.args):
            return None
        return weighted_mean(df, value_col, weight_col, group_col=TEAM_GROUP_COL)
    if recipe.pattern == "proe_formula":
        actual_col, expected_col = recipe.args
        if _missing_cols(df, recipe.args):
            return None
        grouped = df.groupby(TEAM_GROUP_COL, dropna=False)
        actual = grouped[actual_col].sum(min_count=1)
        expected = grouped[expected_col].sum(min_count=1)
        out = (actual - expected).divide(expected).where(expected > 0, other=np.nan)
        out.name = recipe.output_col
        return out
    if recipe.pattern == "rpr_bucket_pass_rate":
        (bucket_key,) = recipe.args
        return _rpr_bucket_pass_rate(df, bucket_key)
    logger.warning("unknown team-recipe pattern %r for %s", recipe.pattern, recipe.output_col)
    return None


def _rpr_bucket_pass_rate(df: pd.DataFrame, bucket_key: str) -> pd.Series | None:
    """JSON-parse run_pass_report's bucket column, return per-team pass% for ``bucket_key``.

    The bucket column carries a per-row JSON object with nested entries for
    each situational split (Inside10, Inside20, Leading, Trailing, etc.).
    Each split exposes ``teamStatsSnapsOffensePass`` and (sometimes)
    ``teamStatsSnapsOffenseRush``. Per-team pass% =
    ``sum(pass_snaps) / (sum(pass_snaps) + sum(rush_snaps))`` across all
    games in the window. Returns ``None`` when the bucket column is absent
    or no team has any snaps in the bucket.
    """
    if "bucket" not in df.columns or TEAM_GROUP_COL not in df.columns:
        return None
    parsed = df["bucket"].apply(_parse_bucket_json)
    pass_snaps = parsed.apply(lambda d: _bucket_snap(d, bucket_key, _RPR_BUCKET_PASS))
    rush_snaps = parsed.apply(lambda d: _bucket_snap(d, bucket_key, _RPR_BUCKET_RUSH))
    work = pd.DataFrame(
        {
            "_pass": pd.to_numeric(pass_snaps, errors="coerce").fillna(0),
            "_rush": pd.to_numeric(rush_snaps, errors="coerce").fillna(0),
            TEAM_GROUP_COL: df[TEAM_GROUP_COL].values,
        }
    )
    grouped = work.groupby(TEAM_GROUP_COL, dropna=False)
    pass_sum = grouped["_pass"].sum()
    total = pass_sum + grouped["_rush"].sum()
    out = pass_sum.divide(total).where(total > 0, other=np.nan)
    out.name = f"rpr_{bucket_key}"
    return out


def _bucket_snap(parsed: object, bucket_key: str, snap_field: str) -> float:
    """Pull a single snap count out of a parsed bucket JSON dict."""
    if not isinstance(parsed, dict):
        return 0.0
    bucket = parsed.get(bucket_key)
    if not isinstance(bucket, dict):
        return 0.0
    return float(bucket.get(snap_field, 0) or 0)


def _parse_bucket_json(raw: object) -> dict:
    """Tolerantly parse one row's ``bucket`` JSON cell -- ``{}`` on failure."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


# Module-level cache: teamTeamId -> teamAbbreviation, keyed by season.
# Populated lazily on first call to _build_team_abbreviation_map; avoids
# re-reading parquets on repeated calls in a training loop.
_ABBR_MAP_CACHE: dict[int, pd.Series] = {}


def _build_team_abbreviation_map(season: int) -> pd.Series:
    """Map ``teamTeamId`` -> ``teamAbbreviation`` for the given season.

    Sourced from ``line_matchups`` (the only team-grain kind that carries
    both keys -- all other team-grain kinds expose ``teamTeamId`` paired
    with ``teamLocation`` / ``teamNickname`` but not the abbreviation).
    Falls back to a player-grain kind (``offense_snaps``) when no team-grain
    line_matchups snapshot is populated; the player-grain bundle is more
    likely to be backfilled and always carries ``teamAbbreviation``.
    Caches per season on the module object so repeat calls in a training
    loop don't re-read parquets.
    """
    cache = _ABBR_MAP_CACHE
    if season in cache:
        return cache[season]

    mapping = _abbr_from_team_grain(season)
    if mapping is None:
        mapping = _abbr_from_player_grain(season)

    if mapping is None or mapping.empty:
        cache[season] = pd.Series(dtype="object", name=_TEAM_ABBR_COL)
    else:
        cache[season] = mapping
    return cache[season]


def _abbr_from_team_grain(season: int) -> pd.Series | None:
    """Try to build the teamTeamId -> abbreviation map from a team-grain snapshot.

    ``coverage_matrix`` is the preferred source: current pulls put the
    abbreviation on every team row, which makes this an identity map.
    ``line_matchups`` is the fallback for the archived legacy seasons, whose
    team parquets pair ``teamTeamId`` with location / nickname and nowhere
    else carry the abbreviation. It is consulted second rather than first
    because the kind is retired — any file still on disk for a season that
    has since been re-pulled holds the superseded id space.
    """
    for week in nfl_fp_team_weekly.available_snapshots(season):
        for kind in ("coverage_matrix", "line_matchups"):
            df = nfl_fp_team_weekly.load_snapshot(season, week, kind)
            if df is None or df.empty:
                continue
            if TEAM_GROUP_COL not in df.columns or _TEAM_ABBR_COL not in df.columns:
                continue
            return (
                df[[TEAM_GROUP_COL, _TEAM_ABBR_COL]]
                .dropna()
                .drop_duplicates(subset=[TEAM_GROUP_COL])
                .set_index(TEAM_GROUP_COL)[_TEAM_ABBR_COL]
            )
    return None


def _abbr_from_player_grain(season: int) -> pd.Series | None:
    """Fall back to a player-grain kind (``offense_snaps``) for the mapping.

    Player-grain parquets always carry both ``playerTeamId`` (a per-row
    foreign key) and ``teamAbbreviation``. Using ``offense_snaps`` because
    it has the broadest player coverage; receiving_basic / passing_basic
    work too but only cover players who took those snaps.
    """
    for week in nfl_fp_weekly.available_snapshots(season):
        df = nfl_fp_weekly.load_snapshot(season, week, "offense_snaps")
        if df is None or df.empty:
            continue
        # Player-grain frames key team via ``playerTeamId`` not ``teamTeamId``.
        # Resolve via whichever id column the frame actually exposes.
        id_col = next(
            (c for c in ("teamTeamId", "playerTeamId") if c in df.columns),
            None,
        )
        if id_col is None or _TEAM_ABBR_COL not in df.columns:
            continue
        return (
            df[[id_col, _TEAM_ABBR_COL]]
            .dropna()
            .drop_duplicates(subset=[id_col])
            .set_index(id_col)[_TEAM_ABBR_COL]
        )
    return None


__all__ = ("load_team_and_defense_features",)
