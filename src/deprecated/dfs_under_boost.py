# ARCHIVED 2026-08-29 from src/sportstradamus/helpers/archive.py
# Reason: add_dfs now prices one-sided DFS offers at their payout-implied
#         breakeven (dfs_boost_probs); fabricating the symmetric under twin
#         let moved/discounted lines enter the archive as fair 50/50 quotes.
# Last live SHA: e1246576
# Original imports (now unresolved here):
#   (none)


def _dfs_under_boost(over, boost_under):
    return boost_under if boost_under and boost_under > 0 else over
