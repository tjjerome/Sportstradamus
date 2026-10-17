"""Training data preparation: matrix trimming."""

import warnings

import numpy as np
import pandas as pd

from sportstradamus.helpers.training_quotes import SYNTHETIC
from sportstradamus.stats.base import SEASON_MAX_DAYS

# Quantile bounds for the line clip when archived data is sparse.
_LINE_CLIP_LOW: float = 0.05
_LINE_CLIP_HIGH: float = 0.95

# Minimum archived rows AND minimum fraction of archived rows required before
# the line-clip bounds are derived from the archived distribution (a smaller
# or sparser archive falls back to the full-data percentiles so we don't over-
# restrict thin markets).
_MIN_ARCHIVED_FOR_CLIP: int = 50
_MIN_ARCHIVED_FRACTION_FOR_CLIP: float = 0.10

# Minimum archived rows required before an archived-derived line target is trusted
# over the full-data fallback: overall median and per-position median respectively.
_MIN_ARCHIVED_FOR_MEDIAN_LINE: int = 5
_MIN_POS_ARCHIVED_FOR_MEDIAN: int = 20

# Minimum reference-set size before histogram-matching weights are trusted;
# below this the reference is treated as empty and uniform weights are used.
_MIN_HISTOGRAM_REFERENCE: int = 20

# Rides through trim_matrix marking the rows appended since the matrix was last trimmed. Cached
# rows were clipped when they were new; re-clipping them against bounds that move with the
# population cut a fresh slice of the unquoted tail on every append.
_NEW_ROW_COLUMN: str = "__NewRow"


def _histogram_weights(
    values, reference_values, min_reference=_MIN_HISTOGRAM_REFERENCE
) -> np.ndarray:
    """Compute removal probabilities via histogram matching.

    Returns probability array aligned to values. Rows whose bin density
    exceeds the reference density are more likely to be removed.
    Falls back to uniform weights when reference data is insufficient.
    """
    if len(values) == 0:
        return np.array([])

    counts, bins = np.histogram(values)
    counts = counts / len(values)

    if len(reference_values) >= min_reference:
        ref_counts, _ = np.histogram(reference_values, bins)
        ref_counts = ref_counts / len(reference_values)
    else:
        ref_counts = np.zeros_like(counts)

    diff = np.clip(counts - ref_counts, 1e-8, None)
    p = np.zeros(len(values))
    for j, a in enumerate(bins[:-1]):
        p[values >= a] = diff[j]
    return p / np.sum(p)


def _rebase_days_into_season(M: pd.DataFrame) -> pd.DataFrame:
    while any(M["DaysIntoSeason"] < 0) or any(M["DaysIntoSeason"] > SEASON_MAX_DAYS):
        M.loc[M["DaysIntoSeason"] < 0, "DaysIntoSeason"] = (
            M.loc[M["DaysIntoSeason"] < 0, "DaysIntoSeason"] - M["DaysIntoSeason"].min()
        )
        M.loc[M["DaysIntoSeason"] > SEASON_MAX_DAYS, "DaysIntoSeason"] = (
            M.loc[M["DaysIntoSeason"] > SEASON_MAX_DAYS, "DaysIntoSeason"]
            - M.loc[M["DaysIntoSeason"] > SEASON_MAX_DAYS, "DaysIntoSeason"].min()
        )
    return M


def _clip_lines(M: pd.DataFrame, archived_mask, n_archived: int, new_mask) -> pd.DataFrame:
    if (
        n_archived >= _MIN_ARCHIVED_FOR_CLIP
        and n_archived / len(M) > _MIN_ARCHIVED_FRACTION_FOR_CLIP
    ):
        line_floor = M.loc[archived_mask, "Line"].min()
        line_ceil = M.loc[archived_mask, "Line"].max()
    else:
        line_floor = M["Line"].quantile(_LINE_CLIP_LOW)
        line_ceil = M["Line"].quantile(_LINE_CLIP_HIGH)
    clip_mask = (
        M["QuoteAuthenticity"].eq(SYNTHETIC)
        if "QuoteAuthenticity" in M
        else pd.Series(True, index=M.index)
    ) & new_mask
    M.loc[clip_mask, "Line"] = M.loc[clip_mask, "Line"].clip(line_floor, line_ceil)
    return M


def _balance_lines(M, pos_mask, archived_mask, overall_target, min_rows, rng):
    budget = max(len(M) - min_rows, 0)
    if budget == 0:
        return M

    pos_archived = archived_mask & pos_mask
    n_pos_archived = pos_archived.sum()
    target = (
        M.loc[pos_archived, "Line"].median()
        if n_pos_archived >= _MIN_POS_ARCHIVED_FOR_MEDIAN
        else overall_target
    )

    non_arch = ~archived_mask & pos_mask
    less = M.loc[non_arch & (M["Line"] < target), "Line"]
    more = M.loc[non_arch & (M["Line"] > target), "Line"]

    n = min(abs(len(less) - len(more)), budget)
    if n == 0:
        return M

    if len(less) > len(more):
        ref = M.loc[pos_archived & (M["Line"] < target), "Line"]
        p = _histogram_weights(less.values, ref.values)
        chopping_block = less.index
    else:
        ref = M.loc[pos_archived & (M["Line"] > target), "Line"]
        p = _histogram_weights(more.values, ref.values)
        chopping_block = more.index

    n = min(n, len(chopping_block))
    cut = rng.choice(chopping_block, n, replace=False, p=p)
    M.drop(cut, inplace=True)
    return M


def trim_matrix(
    M: pd.DataFrame,
    min_rows: int = 7500,
    seed: int | None = None,
    *,
    new_rows: pd.Series | None = None,
) -> pd.DataFrame:
    """Clip lines to the market's range and balance the line distribution across positions.

    Every step reads the line, never the outcome: selecting rows on ``Result`` truncated the
    unquoted tail (stars' big games) and biased the fit (nfl-ship15-recovery.md §11). The
    balancing respects ``min_rows`` so sparse-archive markets are not destroyed.

    ``new_rows`` is a boolean mask over ``M.index`` marking the rows appended since the
    matrix was last trimmed (``None``: every row). The line clip applies only to those rows,
    so re-trimming a cached matrix with nothing new returns it unchanged; the balancing still
    sees every row.
    """
    warnings.simplefilter("ignore", UserWarning)
    M = M.assign(**{_NEW_ROW_COLUMN: True if new_rows is None else new_rows})
    if seed is None:
        rng = np.random
    else:
        stable_keys = [column for column in ("Date", "Player") if column in M]
        if stable_keys:
            M = M.sort_values(stable_keys, kind="stable").reset_index(drop=True)
        rng = np.random.default_rng(seed)

    M = _rebase_days_into_season(M)

    archived_mask = M["Archived"] == 1
    n_archived = archived_mask.sum()
    M = _clip_lines(M, archived_mask, n_archived, M[_NEW_ROW_COLUMN])

    overall_target = (
        M.loc[archived_mask, "Line"].median()
        if n_archived >= _MIN_ARCHIVED_FOR_MEDIAN_LINE
        else M["Line"].mean()
    )

    if "Player position" in M.columns:
        for i in M["Player position"].unique():
            M = _balance_lines(
                M, M["Player position"] == i, archived_mask, overall_target, min_rows, rng
            )
    else:
        M = _balance_lines(
            M, pd.Series(True, index=M.index), archived_mask, overall_target, min_rows, rng
        )

    return M.sort_values("Date").drop(columns=_NEW_ROW_COLUMN)
