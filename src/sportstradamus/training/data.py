"""Training data preparation: row counting and matrix trimming."""

import warnings
from datetime import datetime

import numpy as np
import pandas as pd


def count_training_rows(stat_data, market, start_date, archive) -> int:
    """Estimate the number of training rows get_training_matrix would produce for
    a given market and start_date, using the archive and gamelog directly.

    Counts:
      - Archived rows: (date, player) entries in the archive with a real line
      - Non-archived rows: players above the 25th-percentile usage cutoff on
        each game day, excluding any that are already counted as archived

    This is an upper-bound estimate; trim_matrix may remove some rows afterward.
    """
    gamelog = stat_data.gamelog.drop_duplicates(
        subset=[stat_data.log_strings["game"], stat_data.log_strings["player"]], keep="last"
    ).copy()
    gamelog[stat_data.log_strings["date"]] = pd.to_datetime(
        gamelog[stat_data.log_strings["date"]]
    ).dt.date
    gamelog = gamelog.loc[
        (gamelog[stat_data.log_strings["date"]] > start_date)
        & (gamelog[stat_data.log_strings["date"]] < datetime.today().date())
    ]
    if gamelog.empty:
        return 0

    usage_cutoff = gamelog[stat_data.usage_stat].quantile(0.25)
    league_archive = archive.archive.get(stat_data.league, {}).get(market, {})

    total = 0
    for game_date, players in gamelog.groupby(stat_data.log_strings["date"]):
        date_str = game_date.strftime("%Y-%m-%d")
        archived_players = set(league_archive.get(date_str, {}).keys())

        played = set(players[stat_data.log_strings["player"]].unique())
        archived_count = len(archived_players & played)

        non_archived = players.drop_duplicates(subset=stat_data.log_strings["player"]).loc[
            lambda df: ~df[stat_data.log_strings["player"]].isin(archived_players)
        ]
        non_archived_count = (non_archived[stat_data.usage_stat] > usage_cutoff).sum()

        total += archived_count + non_archived_count

    return total


def _histogram_weights(values, reference_values, min_reference=20) -> np.ndarray:
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


def trim_matrix(M: pd.DataFrame, min_rows: int = 7500) -> pd.DataFrame:
    """Remove data quality issues and prepare matrix for modeling.

    Trims outlier results, clips lines to a realistic range, balances
    the line distribution across positions, and balances over/under
    proportions.  All removal steps respect min_rows so that sparse-
    archive markets are not destroyed.
    """
    warnings.simplefilter("ignore", UserWarning)

    # --- 1. Fix DaysIntoSeason wrapping ---
    while any(M["DaysIntoSeason"] < 0) or any(M["DaysIntoSeason"] > 300):
        M.loc[M["DaysIntoSeason"] < 0, "DaysIntoSeason"] = (
            M.loc[M["DaysIntoSeason"] < 0, "DaysIntoSeason"] - M["DaysIntoSeason"].min()
        )
        M.loc[M["DaysIntoSeason"] > 300, "DaysIntoSeason"] = (
            M.loc[M["DaysIntoSeason"] > 300, "DaysIntoSeason"]
            - M.loc[M["DaysIntoSeason"] > 300, "DaysIntoSeason"].min()
        )

    # --- 2. Remove result outliers (archived rows always kept) ---
    M = M.loc[
        ((M["Result"] >= M["Result"].quantile(0.05)) & (M["Result"] <= M["Result"].quantile(0.95)))
        | (M["Archived"] == 1)
    ]

    # --- 3. Clip lines to a realistic range ---
    # Use archived range when coverage is good; otherwise fall back to
    # full-data percentiles so sparse-archive markets keep their natural
    # line distribution.
    archived_mask = M["Archived"] == 1
    n_archived = archived_mask.sum()
    if n_archived >= 50 and n_archived / len(M) > 0.10:
        line_floor = M.loc[archived_mask, "Line"].min()
        line_ceil = M.loc[archived_mask, "Line"].max()
    else:
        line_floor = M["Line"].quantile(0.05)
        line_ceil = M["Line"].quantile(0.95)
    M["Line"] = M["Line"].clip(line_floor, line_ceil)

    # --- 4. Balance line distribution ---
    overall_target = M.loc[archived_mask, "Line"].median() if n_archived >= 5 else M["Line"].mean()

    def _balance_lines(M, pos_mask):
        budget = max(len(M) - min_rows, 0)
        if budget == 0:
            return M

        pos_archived = archived_mask & pos_mask
        n_pos_archived = pos_archived.sum()
        target = M.loc[pos_archived, "Line"].median() if n_pos_archived >= 20 else overall_target

        non_arch = ~archived_mask & pos_mask
        less = M.loc[non_arch & (M["Line"] < target), "Line"]
        more = M.loc[non_arch & (M["Line"] > target), "Line"]

        n = min(abs(len(less) - len(more)), budget)
        if n == 0:
            return M

        if len(less) > len(more):
            ref = M.loc[pos_archived & (M["Line"] < target), "Line"]
            p = _histogram_weights(less.values, ref.values, min_reference=20)
            chopping_block = less.index
        else:
            ref = M.loc[pos_archived & (M["Line"] > target), "Line"]
            p = _histogram_weights(more.values, ref.values, min_reference=20)
            chopping_block = more.index

        n = min(n, len(chopping_block))
        cut = np.random.choice(chopping_block, n, replace=False, p=p)
        M.drop(cut, inplace=True)
        return M

    if "Player position" in M.columns:
        for i in M["Player position"].unique():
            M = _balance_lines(M, M["Player position"] == i)
    else:
        M = _balance_lines(M, pd.Series(True, index=M.index))

    # --- 5. Balance over/under proportions ---
    if n_archived < 10:
        return M.sort_values("Date")

    pushes = M.loc[M["Result"] == M["Line"]]
    push_rate = pushes["Archived"].sum() / M["Archived"].sum()
    M = M.loc[M["Result"] != M["Line"]]

    archived_no_push = M["Archived"] == 1
    if archived_no_push.sum() >= 20:
        target = (M.loc[archived_no_push, "Result"] > M.loc[archived_no_push, "Line"]).mean()
    else:
        target = (M["Result"] > M["Line"]).mean()

    balance = (M["Result"] > M["Line"]).mean()
    budget = max(len(M) - min_rows, 0)
    n = min(2 * int(np.abs(target - balance) * len(M)), budget)

    if n > 0:
        if balance < target:
            chopping_block = M.loc[(M["Archived"] != 1) & (M["Result"] < M["Line"])].index
            p = (1 / M.loc[chopping_block, "MeanYr"].clip(0.1)).to_numpy()
            p = p / np.sum(p)
        else:
            chopping_block = M.loc[(M["Archived"] != 1) & (M["Result"] > M["Line"])].index
            p = (M.loc[chopping_block, "MeanYr"].clip(0.1)).to_numpy()
            p = p / np.sum(p)

        n = min(n, len(chopping_block))
        cut = np.random.choice(chopping_block, n, replace=False, p=p)
        M.drop(cut, inplace=True)

    # --- 6. Re-insert pushes at the correct proportion ---
    n = int(push_rate * len(M)) - pushes["Archived"].sum()
    chopping_block = pushes.loc[pushes["Archived"] == 0].index
    n = np.clip(n, None, len(chopping_block))
    if n > 0:
        cut = np.random.choice(chopping_block, n, replace=False)
        pushes.drop(cut, inplace=True)

    M = pd.concat([M, pushes]).sort_values("Date")

    return M
