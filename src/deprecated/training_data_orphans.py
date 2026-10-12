# ARCHIVED 2026-10-11 from src/sportstradamus/training/data.py
# Reason: count_training_rows had no caller in src/sportstradamus/, tests/ or scripts/, so it
#         is archived with the usage cutoff only it read. The Archive method only it called
#         (archived_players_by_date) is archived in dfs_under_boost.py.
# Last live SHA: a63b9107
# Original imports (now unresolved here):
#   from datetime import datetime
#   import pandas as pd

# From src/sportstradamus/training/data.py. Only count_training_rows read this cutoff.
#
# Quantile cutoff for counting non-archived rows: players below the 25th
# percentile of usage on each game day are excluded (marginal players rarely
# accrue a sportsbook line, so their rows add noise without adding signal).
_USAGE_QUANTILE_CUTOFF: float = 0.25


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

    usage_cutoff = gamelog[stat_data.usage_stat].quantile(_USAGE_QUANTILE_CUTOFF)
    archived_by_date = archive.archived_players_by_date(stat_data.league, market)

    total = 0
    for game_date, players in gamelog.groupby(stat_data.log_strings["date"]):
        date_str = game_date.strftime("%Y-%m-%d")
        archived_players = archived_by_date.get(date_str, set())

        played = set(players[stat_data.log_strings["player"]].unique())
        archived_count = len(archived_players & played)

        non_archived = players.drop_duplicates(subset=stat_data.log_strings["player"]).loc[
            lambda df: ~df[stat_data.log_strings["player"]].isin(archived_players)
        ]
        non_archived_count = (non_archived[stat_data.usage_stat] > usage_cutoff).sum()

        total += archived_count + non_archived_count

    return total
