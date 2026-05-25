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

Pattern B (``line_matchups``) is handled inline in
:func:`_load_line_matchups`: single-snapshot lookup, no aggregation; the
row IS the feature for the target game. Team-side ``teamStats*`` columns
go to teamProfile under the team's abbreviation; opponent-side
``opponentStats*`` columns go to defenseProfile under the opponent's
abbreviation.

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
# Used to re-key per-teamId aggregates back to the abbreviation index that
# teamProfile / defenseProfile uses (matches the NFL gamelog's "team" col).
_TEAM_ABBR_COL = "teamAbbreviation"

# Opponent abbreviation column line_matchups carries instead of
# ``opponentTeamId``. Defense-side Pattern B rows index by this directly
# since the value is already a team-abbreviation key matching the
# defenseProfile index in stats/base.py.
_OPPONENT_ABBR_COL = "opponentAbbreviation"

# PROE source columns. PFR-style formula:
# ``sum(actual_dropbacks - expected_dropbacks) / sum(expected_dropbacks)``.
# The per-game ratio is biased; the sum form is the season-to-date PROE.
_PROE_ACTUAL_COL = "teamStatsPassingDropbacksTotal"
_PROE_EXPECTED_COL = "teamStatsPassingDropbacksExpected"

# run_pass_report bucket subkeys. Each per-game row's ``bucket`` cell is a
# JSON object carrying nested entries keyed by bucket name; each nested
# entry has teamStatsSnapsOffensePass / teamStatsSnapsOffenseRush.
_RPR_BUCKET_PASS = "teamStatsSnapsOffensePass"
_RPR_BUCKET_RUSH = "teamStatsSnapsOffenseRush"

# Spec-aligned subset of FP's situational buckets. ``Inside20`` ≈ redzone,
# ``Inside10`` ≈ goalline; Leading / Trailing capture game-state pass
# tendency; FirstDown / ThirdDown capture down-state. Other available
# buckets (FirstHalf/SecondHalf, Under5/5To9/Over10) are skipped here as
# largely collinear with the chosen subset.
_RPR_BUCKETS: tuple[tuple[str, str], ...] = (
    ("redzone", "bucketInside20"),
    ("goalline", "bucketInside10"),
    ("neutral", "bucketNeutral"),
    ("leading", "bucketLeading"),
    ("trailing", "bucketTrailing"),
    ("first_down", "bucketFirstDown"),
    ("third_down", "bucketThirdDown"),
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
    """Synthesize run_pass_report per-bucket pass% recipes from :data:`_RPR_BUCKETS`."""
    return tuple(
        _TeamRecipe(
            output_col=f"rp_{suffix}_pass_pct",
            file_kind="run_pass_report",
            pattern="rpr_bucket_pass_rate",
            args=(bucket_key,),
            grain="team",
        )
        for suffix, bucket_key in _RPR_BUCKETS
    )


_ALL_RECIPES: tuple[_TeamRecipe, ...] = _RECIPES + _rpr_bucket_recipes()


def load_team_and_defense_features(
    pattern_a_windows: Sequence[tuple[int, int, int]],
    pattern_b_snapshot: tuple[int, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate weekly FP team snapshots into team / defense feature frames.

    Pattern A kinds pool per-game team rows across every
    ``(season, start_week, end_week)`` window, then aggregate per the
    recipe. Pattern B (``line_matchups``) reads the single snapshot at
    ``pattern_b_snapshot`` -- the row IS the feature for the target game.

    The two-input contract lets the NFL caller apply the Phase-1.5
    lookback rule (post-week-4 = current-season only;
    pre-week-5 = prior season weeks 11..18 BLENDED with current-season
    partial, raw rows pooled before per-team groupby). Both branches of
    that rule express as a ``windows`` list -- the bridge stays family-
    agnostic about cutoffs and weights.

    Output frames are indexed by ``teamAbbreviation`` (e.g. ``"PHI"``,
    ``"KC"``) to match the abbreviation-indexed teamProfile / defenseProfile
    built by ``stats/base.py:base_profile``. The abbreviation map is
    sourced from the most recent season in ``pattern_a_windows`` (or
    ``pattern_b_snapshot[0]`` if Pattern A is empty); team relocations
    (OAK->LV, STL->LA) are resolved to the team's *current* abbreviation
    because the underlying ``teamTeamId`` is stable across seasons.

    Args:
        pattern_a_windows: Per-window ``(season, start_week, end_week)``
            tuples to pool for Pattern A rate-stat reads. Windows where
            ``start_week > end_week`` are skipped silently. Empty list
            short-circuits to no Pattern A features.
        pattern_b_snapshot: ``(season, week)`` of the target game's
            line_matchups snapshot. Pattern B output is empty when the
            snapshot is missing or has no rows.

    Returns:
        Tuple ``(team_features, defense_features)`` of DataFrames, each
        indexed by team abbreviation. Empty DataFrames if no usable
        snapshots were found.
    """
    abbr_season = _pick_abbr_season(pattern_a_windows, pattern_b_snapshot)
    abbr_map = _build_team_abbreviation_map(abbr_season)
    if abbr_map.empty:
        return pd.DataFrame(), pd.DataFrame()

    team_features = _aggregate_pattern_a(
        pattern_a_windows, grain="team", abbr_map=abbr_map
    )
    defense_features = _aggregate_pattern_a(
        pattern_a_windows, grain="defense", abbr_map=abbr_map
    )
    team_b, defense_b = _load_line_matchups(*pattern_b_snapshot, abbr_map=abbr_map)
    if not team_b.empty:
        team_features = (
            team_b if team_features.empty else team_features.join(team_b, how="outer")
        )
    if not defense_b.empty:
        defense_features = (
            defense_b
            if defense_features.empty
            else defense_features.join(defense_b, how="outer")
        )
    return team_features, defense_features


def _pick_abbr_season(
    pattern_a_windows: Sequence[tuple[int, int, int]],
    pattern_b_snapshot: tuple[int, int],
) -> int:
    """Pick the season to source the teamTeamId -> abbreviation map from.

    Prefer the most recent season in the Pattern A windows; fall back to
    the Pattern B snapshot season when Pattern A is empty. Either choice
    converges to the same mapping for NFL teams that haven't relocated;
    for teams that have (OAK->LV in 2020), the most-recent season gives
    the current-day abbreviation -- which matches what teamProfile /
    defenseProfile use for live game lookups.
    """
    if pattern_a_windows:
        return max(season for season, _, _ in pattern_a_windows)
    return pattern_b_snapshot[0]


def _aggregate_pattern_a(
    pattern_a_windows: Sequence[tuple[int, int, int]],
    *,
    grain: Literal["team", "defense"],
    abbr_map: pd.Series,
) -> pd.DataFrame:
    """Apply every recipe whose ``grain`` matches over pooled per-window per-game rows.

    For each ``file_kind`` referenced by matching recipes, loads all windows,
    concatenates the raw per-game rows (pooling across windows before the per-team
    groupby so blended windows are treated as one sample), then dispatches to
    ``_apply_recipes``. Results are re-keyed from ``teamTeamId`` to team
    abbreviation via ``abbr_map``.

    Args:
        pattern_a_windows: ``(season, start_week, end_week)`` tuples to pool.
            Windows where ``start_week > end_week`` are skipped.
        grain: ``"team"`` -> collects team-grain recipes; ``"defense"`` -> defense-grain.
        abbr_map: Series mapping ``teamTeamId`` -> ``teamAbbreviation`` for re-keying.

    Returns:
        DataFrame indexed by team abbreviation. Empty when no usable snapshots found.
    """
    if not pattern_a_windows:
        return pd.DataFrame()

    by_kind: dict[str, list[_TeamRecipe]] = {}
    for recipe in _ALL_RECIPES:
        if recipe.grain != grain:
            continue
        by_kind.setdefault(recipe.file_kind, []).append(recipe)

    out = pd.DataFrame()
    for file_kind, recipes in by_kind.items():
        frames = []
        for season, start_week, end_week in pattern_a_windows:
            if start_week > end_week:
                continue
            df = _load_kind_window(season, start_week, end_week, file_kind)
            if not df.empty:
                frames.append(df)
        if not frames:
            continue
        pooled = pd.concat(frames, ignore_index=True)
        kind_frame = _apply_recipes(pooled, recipes)
        if kind_frame.empty:
            continue
        out = kind_frame if out.empty else out.combine_first(kind_frame)

    if out.empty:
        return out

    mapping = abbr_map.to_dict()
    out.index = [mapping.get(idx) for idx in out.index]
    out = out.loc[out.index.notna()] if isinstance(out.index, pd.Index) else out
    return out


def _apply_recipes(df: pd.DataFrame, recipes: Sequence[_TeamRecipe]) -> pd.DataFrame:
    """Route each recipe to its dispatch function and collect results into a wide frame.

    Recipes that reference missing columns are silently skipped (``_dispatch``
    returns ``None``); the caller gets a frame with however many columns survived.

    Args:
        df: Raw pooled per-game rows for one ``file_kind``.
        recipes: Subset of ``_ALL_RECIPES`` filtered to one ``file_kind`` and ``grain``.

    Returns:
        Wide DataFrame indexed by ``TEAM_GROUP_COL`` (``teamTeamId``), one column
        per surviving recipe. Empty when all recipes return ``None``.
    """
    out = pd.DataFrame()
    for recipe in recipes:
        series = _dispatch(df, recipe)
        if series is None or series.empty:
            continue
        out[recipe.output_col] = series
    return out


def _dispatch(df: pd.DataFrame, recipe: _TeamRecipe) -> pd.Series | None:
    """Route a recipe to the right helper -- skip if required columns missing."""
    if recipe.pattern == "weighted_rate":
        num_col, den_col = recipe.args
        if num_col not in df.columns or den_col not in df.columns:
            return None
        return weighted_rate(df, num_col, den_col, group_col=TEAM_GROUP_COL)
    if recipe.pattern == "weighted_mean":
        value_col, weight_col = recipe.args
        if value_col not in df.columns or weight_col not in df.columns:
            return None
        return weighted_mean(df, value_col, weight_col, group_col=TEAM_GROUP_COL)
    if recipe.pattern == "proe_formula":
        actual_col, expected_col = recipe.args
        if actual_col not in df.columns or expected_col not in df.columns:
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
    logger.warning(
        "unknown team-recipe pattern %r for %s", recipe.pattern, recipe.output_col
    )
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


def _load_line_matchups(
    season: int,
    target_week: int,
    abbr_map: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pattern B -- single-snapshot lookup, split team-side / opp-side cols.

    ``line_matchups`` carries one row per matchup that week. Each row has
    both the team's own protection-line baseline (``teamStats*``) and the
    opponent's pass-rush profile (``opponentStats*``). Team-side cols join
    into teamProfile under the team's abbreviation (resolved from
    ``teamTeamId`` via ``abbr_map``); opp-side cols join into defenseProfile
    under the opponent's abbreviation (taken directly from the
    ``opponentAbbreviation`` column FP writes on line_matchups rows).
    """
    df = nfl_fp_team_weekly.load_snapshot(season, target_week, "line_matchups")
    if df is None or df.empty or TEAM_GROUP_COL not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    mapping = abbr_map.to_dict()
    team_cols = [c for c in df.columns if c.startswith("teamStats")]
    opp_cols = [c for c in df.columns if c.startswith("opponentStats")]

    team_df = _line_matchups_team_side(df, team_cols, mapping)
    opp_df = _line_matchups_opp_side(df, opp_cols)
    return team_df, opp_df


def _line_matchups_team_side(
    df: pd.DataFrame,
    value_cols: list[str],
    mapping: dict[int, str],
) -> pd.DataFrame:
    """Project the team-side of line_matchups (own OL) to an abbreviation-indexed frame.

    Uses ``teamTeamId`` resolved via ``mapping`` because the team-grain
    line_matchups frame carries ``teamTeamId`` + ``teamAbbreviation`` for
    its own side -- both work as the index, but resolving via the id keeps
    one canonical source-of-truth path with the Pattern-A aggregation.
    """
    if not value_cols:
        return pd.DataFrame()
    side = df[[TEAM_GROUP_COL, *value_cols]].copy()
    side.index = [mapping.get(idx) for idx in side[TEAM_GROUP_COL]]
    side = side.loc[side.index.notna()] if isinstance(side.index, pd.Index) else side
    side = side.drop(columns=[TEAM_GROUP_COL])
    side = side.add_prefix("lm_")
    return side


def _line_matchups_opp_side(
    df: pd.DataFrame,
    value_cols: list[str],
) -> pd.DataFrame:
    """Project the opp-side of line_matchups (opp DL) to an abbreviation-indexed frame.

    Uses ``opponentAbbreviation`` directly because line_matchups does NOT
    carry ``opponentTeamId`` -- the abbreviation is the only opponent key
    FP writes. Skips the abbr_map round-trip entirely.
    """
    if not value_cols or _OPPONENT_ABBR_COL not in df.columns:
        return pd.DataFrame()
    side = df[[_OPPONENT_ABBR_COL, *value_cols]].copy()
    side.index = side[_OPPONENT_ABBR_COL].values
    side = side.loc[side.index.notna()] if isinstance(side.index, pd.Index) else side
    side = side.drop(columns=[_OPPONENT_ABBR_COL])
    side = side.add_prefix("lm_def_")
    return side


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
    """Try to build the teamTeamId -> abbreviation map from ``line_matchups``."""
    for week in nfl_fp_team_weekly.available_snapshots(season):
        df = nfl_fp_team_weekly.load_snapshot(season, week, "line_matchups")
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


def _load_kind_window(
    season: int,
    start_week: int,
    end_week: int,
    file_kind: str,
) -> pd.DataFrame:
    """Read one ``file_kind`` for the closed ``[start_week, end_week]`` range.

    Catches and logs any loader exception rather than propagating it so that a
    single missing or corrupt snapshot doesn't abort the entire profile build.
    Callers should treat an empty return as "no data available" and continue.

    Args:
        season: NFL season year (e.g. 2024).
        start_week: First week of the range, inclusive.
        end_week: Last week of the range, inclusive. Must be >= ``start_week``.
        file_kind: Logical kind name from ``nfl_fp_team_weekly.FILE_KINDS``.

    Returns:
        Concatenated per-game rows for the range. Empty DataFrame on any error
        or when the snapshot directory contains no matching files.
    """
    if start_week > end_week:
        return pd.DataFrame()
    try:
        df = nfl_fp_team_weekly.load_window(season, start_week, end_week, file_kind)
    except Exception as exc:
        logger.warning(
            "team load_window(%s, %s, %s, %s) failed: %s",
            season,
            start_week,
            end_week,
            file_kind,
            exc,
        )
        return pd.DataFrame()
    return df if df is not None else pd.DataFrame()


__all__ = ("load_team_and_defense_features",)
