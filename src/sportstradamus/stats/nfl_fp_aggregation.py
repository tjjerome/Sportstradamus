"""Per-game → season-to-date aggregation helpers for FP weekly snapshots.

The FP weekly loaders (:mod:`sportstradamus.stats.nfl_fp_weekly` and
:mod:`sportstradamus.stats.nfl_fp_team_weekly`) return concatenated
per-game rows; this module turns that DataFrame into the rolled-up
feature values consumers actually need.

Three aggregation patterns map to the kinds of stats FP publishes; the
docstrings on the loader modules spell out which file_kinds need which
pattern, and Phase-1.5 / Phase-2 build features by dispatching through
the four helpers here:

- :func:`weighted_rate` — Pattern A, the default for rate stats.
  ``sum(numerator) / sum(denominator)`` per group. The right way to
  derive a season-to-date rate from per-game totals; mean-of-rates
  silently up-weights low-volume games.
- :func:`total_sum` — raw season-to-date counts (no rate derivation).
  The simplest aggregation; correct for any stat that's a pure tally.
- :func:`game_mean` — Pattern C, average of per-game outcomes.
  Use for stats whose only meaningful denominator is "games played"
  (team-level FP scored / FP allowed). Distinct from Pattern A in that
  there's no underlying volume to weight against.
- :func:`matchup_lookup` — Pattern B, single-row pass-through.
  No aggregation; pulls the row(s) matching a (team, opponent) key
  for matchup-forecast kinds (`line_matchups`, etc.).

Both `base_profile` (training) and `get_stats` (inference) should
route every FP-derived feature through these helpers so the two
paths can't drift on aggregation strategy. Phase 1.5 wires up the
first consumer; Phase 2/3/4 reuse the same helpers.

All helpers tolerate empty / missing columns by returning empty
results or NaN -- they never raise on the data shape, only on
caller-side argument errors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Default grouping keys for the two FP grains. Consumers can override
# (e.g. `group_col="teamTeamId"` when aggregating defense-side player
# data keyed on opponent). Keeping these as constants documents the
# expected primary keys.
PLAYER_GROUP_COL = "playerPlayerId"
TEAM_GROUP_COL = "teamTeamId"


def weighted_rate(
    df: pd.DataFrame,
    numerator_col: str,
    denominator_col: str,
    *,
    group_col: str = PLAYER_GROUP_COL,
) -> pd.Series:
    """Compute ``sum(numerator) / sum(denominator)`` per group.

    The right way to derive a season-to-date rate from per-game total
    columns. For YPRR, pass ``numerator_col="playerStatsReceivingYardsTotal"``
    and ``denominator_col="playerStatsReceivingRoutesTotal"``: the
    output is per-player YPRR over the concatenated window, weighted
    by routes-run as it should be (a 1-route game doesn't get the same
    vote as a 35-route game).

    Groups whose summed denominator is zero (or where the group has no
    rows) produce ``NaN`` in the output rather than divide-by-zero.

    Args:
        df: Per-game rows, typically from :func:`nfl_fp_weekly.load_through`.
        numerator_col: Column to sum on the numerator side.
        denominator_col: Column to sum on the denominator side.
        group_col: Group key. Defaults to ``"playerPlayerId"`` for
            player-grain frames; pass ``"teamTeamId"`` for team-grain.

    Returns:
        Series indexed by ``group_col`` value with the per-group rate.
        Empty Series if ``df`` is empty or any required column is
        missing.
    """
    if df.empty or numerator_col not in df.columns or denominator_col not in df.columns:
        return pd.Series(dtype="float64", name=f"{numerator_col}_per_{denominator_col}")

    grouped = df.groupby(group_col, dropna=False)
    num = grouped[numerator_col].sum(min_count=1)
    den = grouped[denominator_col].sum(min_count=1)
    out = num.divide(den).where(den != 0, other=np.nan)
    out.name = f"{numerator_col}_per_{denominator_col}"
    return out


def total_sum(
    df: pd.DataFrame,
    count_col: str,
    *,
    group_col: str = PLAYER_GROUP_COL,
) -> pd.Series:
    """Sum ``count_col`` per group -- raw season-to-date totals.

    The simplest aggregation: per-group sum across every per-game row
    in the window. Correct for any tally (targets, receptions, yards,
    routes-run, snaps).

    Args:
        df: Per-game rows.
        count_col: Column to sum.
        group_col: Group key. Defaults to ``"playerPlayerId"``.

    Returns:
        Series indexed by ``group_col`` value with the per-group sum.
        Empty Series if ``df`` is empty or ``count_col`` is missing.
    """
    if df.empty or count_col not in df.columns:
        return pd.Series(dtype="float64", name=count_col)

    out = df.groupby(group_col, dropna=False)[count_col].sum(min_count=1)
    out.name = count_col
    return out


def game_mean(
    df: pd.DataFrame,
    value_col: str,
    *,
    group_col: str = TEAM_GROUP_COL,
) -> pd.Series:
    """Mean of ``value_col`` over the per-game rows for each group.

    Use for stats whose denominator is "games played" -- typically
    team-level FP scored / FP allowed. Distinct from
    :func:`weighted_rate` in that there's no per-game volume to weight
    against; each game contributes equally.

    The default ``group_col`` is the team key because Pattern C is
    overwhelmingly used at team grain. Override for player-grain
    usage.

    Args:
        df: Per-game rows.
        value_col: Column to average.
        group_col: Group key. Defaults to ``"teamTeamId"``.

    Returns:
        Series indexed by ``group_col`` value with the per-group mean.
        Empty Series if ``df`` is empty or ``value_col`` is missing.
    """
    if df.empty or value_col not in df.columns:
        return pd.Series(dtype="float64", name=f"{value_col}_per_game")

    out = df.groupby(group_col, dropna=False)[value_col].mean()
    out.name = f"{value_col}_per_game"
    return out


def matchup_lookup(
    df: pd.DataFrame,
    where_col: str,
    where_value: object,
) -> pd.DataFrame:
    """Pick rows matching ``df[where_col] == where_value`` -- no aggregation.

    Pattern B entry point. Matchup-forecast kinds like ``line_matchups``
    publish one row per (team, week) that's already game-specific --
    the row IS the feature. Concatenating these across weeks would
    destroy the matchup signal; this helper just filters.

    The caller is responsible for picking the right snapshot before
    calling (via :func:`nfl_fp_weekly.load_snapshot` or
    :func:`nfl_fp_team_weekly.load_snapshot`, NOT ``load_through`` /
    ``load_window``).

    Args:
        df: Single-snapshot frame (NOT a concatenated window).
        where_col: Column to filter on (e.g. ``"teamTeamId"``).
        where_value: Value to match.

    Returns:
        Filtered DataFrame. Empty when no row matches, when ``df`` is
        empty, or when ``where_col`` is missing.
    """
    if df.empty or where_col not in df.columns:
        return df.iloc[0:0].copy()
    return df.loc[df[where_col] == where_value].copy()
