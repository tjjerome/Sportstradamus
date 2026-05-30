# FP Data Coverage Matrix

_as of 2026-05-25, branch model-research, FP weekly snapshots in `src/sportstradamus/data/player_data/NFL/2025/week_01/`_

## Summary

- Total FP parquet kinds: **24**
- Total source columns surveyed (excluding meta/ID cols): **604**
- Total bucket-JSON top-level keys discovered across 10 JSON-bearing kinds: **43**  (`grid`: 4 kinds × 1 game-index key + redundant inner schema; `bucket`: 6 kinds × 2–13 route/scheme/alignment keys)
- Columns with all 5 surface bits OFF: **537 / 604**  (cheap-win pool)
- Columns reaching at least surface 4 (in_training_matrix): **34**
- Total meta columns excluded across all kinds (dedup'd union): **41** distinct names

Audit is read-only; no source files were modified.

**Caveats**

- `fantasy_points_allowed` and `fantasy_points_scored` are flagged `PLACEHOLDER_SINGLE_TILE_KINDS` in `nfl_fp_weekly.py:102` — they currently carry **one placeholder row** rather than per-game enumeration. No recipe touches them today and no recipe should be added until the fetcher's endpoint parameterisation is finalised.
- `NFL_rushing-yards.parquet` is **not present** under `data/training_data/`. Surface-4 detection used the union across `NFL_passing-yards`, `NFL_receiving-yards`, `NFL_fantasy-points-prizepicks`, `NFL_passing-tds`, `NFL_receiving-tds` (5 matrices). All 21 asof + 3 delta + 2 trend/breakout names land identically in every NFL training matrix scanned, so the union vs single-matrix counts agree.
- The four team-grain `off_faced_*` columns appear at surfaces 4 and 5 (in the matrix as `Team off_faced_*` / `Defense off_faced_*`), but **none of the 24 audited player-grain parquets is their source**. They are aggregated from `data/team_data/NFL/<season>/week_NN/coverage_matrix*.parquet` via `nfl_fp_team_weekly_aggregate._RECIPES`. The team-grain audit is out of scope for this document.
- `playerStatsGamesPlayed` is **not** treated as a meta/ID column — it is the per-row game-count denominator for `off_snaps_Snap_pct` (recipe) and several `derive_comp_metrics` `per_game` derivations. It appears in the stat tables, marked `in_recipe yes` only in the one kind where a recipe references it directly (`offense_snap_share_report`).
- The `in_recipe` column lists up to 3 recipe outputs per raw column. A raw column can be both a numerator for one recipe and a denominator/weight for several others; surfaces 2-5 evaluate across all recipe outputs the column feeds (union). For example, `playerStatsPassingDropbacksTotal` is the primary for `pass_adv_DB` AND the denominator for `pass_adv_PrROE` / `pass_adv_PrDB_pct` / `pass_adv_TA_pct` / `pass_adv_RPO_pct`, so it shows surface-4 yes via the asof names of those rate stats.
- `in_comp` shows the position-tagged comp key in the form `RB(success_rate via rush_adv_Success_pct)` when the raw column feeds a `derive_comp_metrics` formula whose output appears under one of the `NFL.{QB,RB,WR,TE}` keys in `playerCompStats.json`.

## Per-kind tables

### efficiency (1,144 rows × 73 cols; 53 stat rows below, 20 meta cols excluded)

| column | in_recipe | in_asof | in_comp | in_matrix | in_filter |
|---|---|---|---|---|---|
| `playerStatsSnapsOffenseTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingRoutesTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingTargetsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingReceptionsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingYardsAfterCatchTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingYardsAfterContactTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingYardsAirTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingAveragesPerRouteYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingAveragesPerTargetYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingAveragesPerReceptionYardsAfterCatch` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingAveragesPerReceptionYardsAfterContact` ← UNUSED | no | no | no | no | no |
| `playerStatsScrimmageTouchesTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsScrimmageYardsTotal` | yes (eff_YFS_G(primary (game_mean))) | no | yes (RB(eff_YFS_G)) | no | no |
| `playerStatsScrimmageYardsAfterContactTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsScrimmageTouchdownsExpectedTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsScrimmageAveragesPerTouchYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsScrimmageAveragesPerTouchYardsAfterContact` ← UNUSED | no | no | no | no | no |
| `playerStatsXfpPprTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNonPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsPpr` | yes (eff_FP_G(primary (game_mean))) | no | yes (WR(eff_FP_G)) | no | no |
| `playerStatsFantasyPointsHalfPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsHalfPprRb` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsPprTe` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsDraftKings` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsFanDuel` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsCbs` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNfl` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsEspn` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsYahoo` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsFfpc` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNffc` ← UNUSED | no | no | no | no | no |
| `playerStatsGamesPlayed` ← UNUSED | no | no | no | no | no |
| `marketShareXfpPprTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingAttemptsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingYardsPerAttempt` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingYardsAfterContactTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingYardsAfterContactPerAttempt` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingMissedTacklesForcedTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingMissedTacklesForcedPerAttempt` ← UNUSED | no | no | no | no | no |
| `playerStatsScrimmageMissedTacklesForcedTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsScrimmageAveragesPerTouchMissedTacklesForced` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingReceptionsExplosiveTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingReceptionsExplosivePercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingTouchdownsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsScrimmageTouchesExplosivePercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsScrimmageTouchdownsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingMissedTacklesForcedTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingMissedTacklesForcedPerReception` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingRunsExplosiveTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingRunsExplosivePercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingTouchdownsTotal` ← UNUSED | no | no | no | no | no |

### fantasy_points_allowed (1 rows × 40 cols; 29 stat rows below, 11 meta cols excluded) — **PLACEHOLDER_SINGLE_TILE_KIND** (single row; no recipe consumes today)

| column | in_recipe | in_asof | in_comp | in_matrix | in_filter |
|---|---|---|---|---|---|
| `playerStatsPassingDropbacksTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingAttemptsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingCompletionsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingTouchdownsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingPasserRating` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingAttemptsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingYardsPerAttempt` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingTouchdownsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingTargetsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingReceptionsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingTouchdownsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingAveragesPerTargetYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNonPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsHalfPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsHalfPprRb` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsPprTe` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsDraftKings` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsFanDuel` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsCbs` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNfl` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsEspn` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsYahoo` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsFfpc` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNffc` ← UNUSED | no | no | no | no | no |
| `teamStatsGamesPlayed` ← UNUSED | no | no | no | no | no |

### fantasy_points_scored (1 rows × 40 cols; 29 stat rows below, 11 meta cols excluded) — **PLACEHOLDER_SINGLE_TILE_KIND** (single row; no recipe consumes today)

| column | in_recipe | in_asof | in_comp | in_matrix | in_filter |
|---|---|---|---|---|---|
| `playerStatsPassingDropbacksTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingAttemptsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingCompletionsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingTouchdownsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingPasserRating` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingAttemptsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingYardsPerAttempt` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingTouchdownsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingTargetsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingReceptionsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingTouchdownsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingAveragesPerTargetYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNonPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsHalfPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsHalfPprRb` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsPprTe` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsDraftKings` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsFanDuel` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsCbs` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNfl` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsEspn` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsYahoo` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsFfpc` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNffc` ← UNUSED | no | no | no | no | no |
| `teamStatsGamesPlayed` ← UNUSED | no | no | no | no | no |

### fpts_scored_report (1,144 rows × 27 cols; 15 stat rows below, 12 meta cols excluded)

| column | in_recipe | in_asof | in_comp | in_matrix | in_filter |
|---|---|---|---|---|---|
| `playerStatsGamesPlayed` ← UNUSED | no | no | no | no | no |
| **`grid`** (JSON) | see Bucket-JSON sub-table below | n/a | n/a | n/a | n/a |
| `playerStatsFantasyPointsNonPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsHalfPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsHalfPprRb` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsPprTe` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsDraftKings` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsFanDuel` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsCbs` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNfl` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsEspn` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsYahoo` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsFfpc` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNffc` ← UNUSED | no | no | no | no | no |

### offense_snap_share_report (1,144 rows × 15 cols; 3 stat rows below, 12 meta cols excluded)

| column | in_recipe | in_asof | in_comp | in_matrix | in_filter |
|---|---|---|---|---|---|
| `playerStatsGamesPlayed` | yes (off_snaps_Snap_pct(weight)) | yes (snap_share_asof) | yes (RB(off_snaps_Snap_pct), TE(off_snaps_Snap_pct)) | yes (snap_share_asof, snap_share_2wk_trend) | yes (snap_share_asof, snap_share_2wk_trend) |
| **`grid`** (JSON) | see Bucket-JSON sub-table below | n/a | n/a | n/a | n/a |
| `marketShareSnapsOffenseTotal` | yes (off_snaps_Snap_pct(value)) | yes (snap_share_asof) | yes (RB(off_snaps_Snap_pct), TE(off_snaps_Snap_pct)) | yes (snap_share_asof, snap_share_2wk_trend) | yes (snap_share_asof, snap_share_2wk_trend) |

### offense_snaps (1,144 rows × 62 cols; 32 stat rows below, 30 meta cols excluded)

| column | in_recipe | in_asof | in_comp | in_matrix | in_filter |
|---|---|---|---|---|---|
| `playerStatsGamesPlayed` ← UNUSED | no | no | no | no | no |
| `teamStatsSnapsOffenseTotal` ← UNUSED | no | no | no | no | no |
| `teamStatsSnapsOffensePass` ← UNUSED | no | no | no | no | no |
| `teamStatsSnapsOffenseRush` ← UNUSED | no | no | no | no | no |
| `teamStatsInside5SnapsOffenseTotal` ← UNUSED | no | no | no | no | no |
| `teamStatsInside10SnapsOffenseTotal` ← UNUSED | no | no | no | no | no |
| `teamStatsInside20SnapsOffenseTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsSnapsOffenseTotal` | yes (off_snaps_TOTAL(primary)) | no | yes (TE(block_pct_proxy via off_snaps_TOTAL)) | no | no |
| `playerStatsSnapsOffensePass` ← UNUSED | no | no | no | no | no |
| `playerStatsSnapsOffenseRush` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNonPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsHalfPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsHalfPprRb` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsPprTe` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsDraftKings` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsFanDuel` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsCbs` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNfl` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsEspn` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsYahoo` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsFfpc` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNffc` ← UNUSED | no | no | no | no | no |
| `playerStatsInside10SnapsOffenseTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsInside20SnapsOffenseTotal` ← UNUSED | no | no | no | no | no |
| `marketShareSnapsOffenseTotal` ← UNUSED | no | no | no | no | no |
| `marketShareSnapsOffensePass` ← UNUSED | no | no | no | no | no |
| `marketShareSnapsOffenseRush` ← UNUSED | no | no | no | no | no |
| `marketShareInside10SnapsOffenseTotal` ← UNUSED | no | no | no | no | no |
| `marketShareInside20SnapsOffenseTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsInside5SnapsOffenseTotal` ← UNUSED | no | no | no | no | no |
| `marketShareInside5SnapsOffenseTotal` ← UNUSED | no | no | no | no | no |

### passing_advanced (1,144 rows × 100 cols; 65 stat rows below, 35 meta cols excluded)

| column | in_recipe | in_asof | in_comp | in_matrix | in_filter |
|---|---|---|---|---|---|
| `playerStatsGamesPlayed` ← UNUSED | no | no | no | no | no |
| `playerStatsSnapsOffenseTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingOpportunitiesTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingAttemptsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNonPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsHalfPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsHalfPprRb` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsPprTe` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsDraftKings` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsFanDuel` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsCbs` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNfl` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsEspn` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsYahoo` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsFfpc` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNffc` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingDropbacksTotal` | yes (pass_adv_DB(primary), pass_adv_PrROE(denominator), pass_adv_TWT_pct(denominator (via attempts; weight)) +3 more) | yes (qb_press_over_expected_asof, qb_two_way_throw_pct_asof, qb_rpo_pct_asof, qb_pressure_dropback_pct_asof, qb_throw_away_pct_asof) | yes (QB(dropbacks_per_game via pass_adv_DB), QB(scrambles_per_dropback via pass_adv_DB), QB(pass_adv_PrROE), QB(pass_adv_TWT_pct), QB(pass_adv_RPO_pct)) | yes (qb_press_over_expected_asof, qb_two_way_throw_pct_asof, qb_rpo_pct_asof, qb_pressure_dropback_pct_asof, qb_throw_away_pct_asof) | yes (qb_press_over_expected_asof, qb_two_way_throw_pct_asof, qb_rpo_pct_asof, qb_pressure_dropback_pct_asof, qb_throw_away_pct_asof) |
| `playerStatsPassingAttemptsTotal` | yes (pass_adv_ATT(primary), pass_adv_CPOE(denominator), pass_adv_aDOT(weight) +1 more) | yes (qb_cpoe_asof, qb_cpoe_4wk_delta, qb_two_way_throw_pct_asof) | yes (QB(pass_adv_ANY_A via pass_adv_ATT), QB(pass_adv_CPOE), QB(pass_adv_aDOT), QB(pass_adv_TWT_pct)) | yes (qb_cpoe_asof, qb_cpoe_4wk_delta, qb_two_way_throw_pct_asof) | yes (qb_cpoe_asof, qb_cpoe_4wk_delta, qb_two_way_throw_pct_asof) |
| `playerStatsPassingAttemptsCatchablePercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingAttemptsInEndzoneTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingCompletionsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingCompletionsPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingCompletionsAdjustedPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingCompletionsOverExpected` | yes (pass_adv_CPOE(numerator)) | yes (qb_cpoe_asof, qb_cpoe_4wk_delta) | yes (QB(pass_adv_CPOE)) | yes (qb_cpoe_asof, qb_cpoe_4wk_delta) | yes (qb_cpoe_asof, qb_cpoe_4wk_delta) |
| `playerStatsPassingIncompletionsBatted` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingDropsYardsLost` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingDropsPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingYardsTotal` | yes (pass_adv_YDS(primary)) | no | yes (QB(pass_adv_ANY_A via pass_adv_YDS)) | no | no |
| `playerStatsPassingYardsPerAttempt` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingYardsAfterCatchPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingYardsAdjustedNetPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingYardsAirTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingTouchdownsTotal` | yes (pass_adv_TD(primary)) | no | yes (QB(pass_adv_ANY_A via pass_adv_TD)) | no | no |
| `playerStatsPassingSackedTotal` | yes (pass_adv_SACK(primary)) | no | yes (QB(pass_adv_ANY_A via pass_adv_SACK)) | no | no |
| `playerStatsPassingSackedFaultTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingSackedPressuredPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingSackedYardsLost` | yes (pass_adv_SACK_YDS(primary)) | no | yes (QB(pass_adv_ANY_A via pass_adv_SACK_YDS)) | no | no |
| `playerStatsPassingSackedPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingScramblesTotal` | yes (pass_adv_SCRM(primary)) | no | yes (QB(scrambles_per_dropback via pass_adv_SCRM)) | no | no |
| `playerStatsPassingScramblesYards` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingPressuredTotal` | yes (pass_adv_PrDB_pct(numerator)) | yes (qb_pressure_dropback_pct_asof) | no | yes (qb_pressure_dropback_pct_asof) | yes (qb_pressure_dropback_pct_asof) |
| `playerStatsPassingPressuredPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingPressuredExpected` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingPressuredOverExpected` | yes (pass_adv_PrROE(numerator)) | yes (qb_press_over_expected_asof) | yes (QB(pass_adv_PrROE)) | yes (qb_press_over_expected_asof) | yes (qb_press_over_expected_asof) |
| `playerStatsPassingRunPassOptionPercentage` | yes (pass_adv_RPO_pct(value)) | yes (qb_rpo_pct_asof) | yes (QB(pass_adv_RPO_pct)) | yes (qb_rpo_pct_asof) | yes (qb_rpo_pct_asof) |
| `playerStatsPassingOffTargetThrowAttemptsPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingDeepThrowAttemptsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingDeepThrowAttemptsPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingHeroThrowPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingTargetedReadFirstPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingTargetedReadCheckdownPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingThrowAccuracyHighlyAccuratePercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingAverageTimeToThrow` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingAverageTimeToSack` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingAverageTimeToScramble` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingAverageTimeToPressure` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingAverageDepthOfTarget` | yes (pass_adv_aDOT(value)) | no | yes (QB(pass_adv_aDOT)) | no | no |
| `playerStatsPassingPasserRating` ← UNUSED | no | no | no | no | no |
| `playerStatsFirstDownsPassing` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingIncompletionsSpiked` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingScramblesTouchdowns` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingTowThrowPercentage` | yes (pass_adv_TWT_pct(value)) | yes (qb_two_way_throw_pct_asof) | yes (QB(pass_adv_TWT_pct)) | yes (qb_two_way_throw_pct_asof) | yes (qb_two_way_throw_pct_asof) |
| `playerStatsPassingIncompletionsThrownAway` | yes (pass_adv_TA_pct(numerator)) | yes (qb_throw_away_pct_asof) | no | yes (qb_throw_away_pct_asof) | yes (qb_throw_away_pct_asof) |
| `playerStatsPassingInterceptionsTotal` | yes (pass_adv_INT(primary)) | no | yes (QB(pass_adv_ANY_A via pass_adv_INT)) | no | no |

### passing_basic (1,144 rows × 55 cols; 36 stat rows below, 19 meta cols excluded)

| column | in_recipe | in_asof | in_comp | in_matrix | in_filter |
|---|---|---|---|---|---|
| `playerStatsGamesPlayed` ← UNUSED | no | no | no | no | no |
| `playerStatsSnapsOffenseTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNonPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsHalfPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsHalfPprRb` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsPprTe` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsDraftKings` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsFanDuel` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsCbs` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNfl` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsEspn` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsYahoo` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsFfpc` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNffc` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingOpportunitiesTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingAttemptsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingYardsTotal` | yes (pass_basic_YDS.1(primary)) | no | no | no | no |
| `playerStatsRushingTouchdownsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingDropbacksTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingAttemptsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingCompletionsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingCompletionsPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingYardsPerAttempt` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingYardsAdjustedNetPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingScramblesTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingScramblesYards` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingPasserRating` ← UNUSED | no | no | no | no | no |
| `playerStatsFirstDownsPassing` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingTouchdownsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingSackedTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingSackedYardsLost` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingSackedPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingScramblesTouchdowns` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingInterceptionsTotal` ← UNUSED | no | no | no | no | no |

### passing_depth (1,144 rows × 48 cols; 13 stat rows below, 35 meta cols excluded)

| column | in_recipe | in_asof | in_comp | in_matrix | in_filter |
|---|---|---|---|---|---|
| `playerStatsGamesPlayed` ← UNUSED | no | no | no | no | no |
| **`bucket`** (JSON) | see Bucket-JSON sub-table below | n/a | n/a | n/a | n/a |
| `playerStatsSnapsOffenseTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingDropbacksTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingAttemptsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingAttemptsCatchablePercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingCompletionsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingTouchdownsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingHeroThrowTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingPasserRating` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingTowThrowTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingInterceptionsTotal` ← UNUSED | no | no | no | no | no |

### qb_coverage_matchup (1,144 rows × 29 cols; 18 stat rows below, 11 meta cols excluded)

| column | in_recipe | in_asof | in_comp | in_matrix | in_filter |
|---|---|---|---|---|---|
| `playerStatsGamesPlayed` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsCoverageSchemeManFantasyPointsPprTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsCoverageSchemeCover3FantasyPointsPprTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsCoverageSchemeCover4FantasyPointsPprTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsCoverageSchemeCover6FantasyPointsPprTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsCoverageSchemeCover2FantasyPointsPprTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingDropbacksTotal` | yes (qb_cov_QB_MAN_pct(weight)) | yes (qb_man_coverage_dropbacks_pct_asof) | yes (QB(qb_cov_QB_MAN_pct)) | yes (qb_man_coverage_dropbacks_pct_asof) | yes (qb_man_coverage_dropbacks_pct_asof) |
| `playerStatsCoverageSchemeManPassingDropbacksTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsCoverageSchemeManPassingDropbacksPercentage` | yes (qb_cov_QB_MAN_pct(value)) | yes (qb_man_coverage_dropbacks_pct_asof) | yes (QB(qb_cov_QB_MAN_pct)) | yes (qb_man_coverage_dropbacks_pct_asof) | yes (qb_man_coverage_dropbacks_pct_asof) |
| `playerStatsCoverageSchemeCover2PassingDropbacksTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsCoverageSchemeCover2PassingDropbacksPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsCoverageSchemeCover3PassingDropbacksTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsCoverageSchemeCover3PassingDropbacksPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsCoverageSchemeCover4PassingDropbacksTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsCoverageSchemeCover4PassingDropbacksPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsCoverageSchemeCover6PassingDropbacksTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsCoverageSchemeCover6PassingDropbacksPercentage` ← UNUSED | no | no | no | no | no |

### receiving_advanced (1,144 rows × 98 cols; 63 stat rows below, 35 meta cols excluded)

| column | in_recipe | in_asof | in_comp | in_matrix | in_filter |
|---|---|---|---|---|---|
| `playerStatsGamesPlayed` ← UNUSED | no | no | no | no | no |
| `playerStatsSnapsOffenseTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingRoutesTotal` | yes (rec_adv_RTE(primary), rec_adv_YPRR(denom/weight), rec_adv_TPRR(denom/weight) +3 more) | yes (rec_yprr_asof, rec_yprr_4wk_delta, rec_tprr_asof, rec_ay_share_asof, target_share_asof) | yes (TE(block_pct_proxy via rec_adv_RTE), RB(rec_adv_YPRR), WR(rec_adv_YPRR), TE(rec_adv_YPRR), RB(rec_adv_TPRR), WR(rec_adv_TPRR), TE(rec_adv_TPRR), WR(rec_adv_AY_Share), RB(rec_adv_TGT_pct), TE(rec_routes_INLINE_RTE_pct)) | yes (rec_yprr_asof, rec_yprr_4wk_delta, rec_tprr_asof, rec_ay_share_asof, target_share_asof, target_share_breakout_flag) | yes (rec_yprr_asof, rec_yprr_4wk_delta, rec_tprr_asof, rec_ay_share_asof, target_share_asof, target_share_breakout_flag) |
| `playerStatsReceivingTargetsTotal` | yes (rec_adv_TGT(primary), rec_adv_TPRR(numerator), rec_adv_DRP_pct(denominator) +4 more) | yes (rec_tprr_asof) | yes (RB(rec_adv_TPRR), WR(rec_adv_TPRR), TE(rec_adv_TPRR), WR(rec_adv_DRP_pct), TE(rec_adv_DRP_pct), WR(rec_adv_1READ_pct), TE(rec_adv_1READ_pct), WR(rec_adv_aDOT)) | yes (rec_tprr_asof) | yes (rec_tprr_asof) |
| `playerStatsReceivingTargetsContestedTotal` | yes (rec_adv_CTGT_pct(numerator)) | no | no | no | no |
| `playerStatsReceivingTargetsCatchableTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingTargetsCatchablePercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingTargetsPerRoute` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingReceptionsTotal` | yes (rec_adv_MTF_REC(weight)) | no | yes (RB(rec_adv_MTF_REC)) | no | no |
| `playerStatsReceivingReceptionsContested` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingReceptionsPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingYardsTotal` | yes (rec_adv_YPRR(numerator)) | yes (rec_yprr_asof, rec_yprr_4wk_delta) | yes (RB(rec_adv_YPRR), WR(rec_adv_YPRR), TE(rec_adv_YPRR)) | yes (rec_yprr_asof, rec_yprr_4wk_delta) | yes (rec_yprr_asof, rec_yprr_4wk_delta) |
| `playerStatsReceivingYardsAfterCatchTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingYardsAfterContactTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingYardsAirTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingTouchdownsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingMissedTacklesForcedTotal` | yes (rec_adv_MTF_REC(numerator)) | no | yes (RB(rec_adv_MTF_REC)) | no | no |
| `playerStatsReceivingMissedTacklesForcedPerReception` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingAlignmentSlotRoutesPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingAlignmentWideRoutesPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingTargetedReadFirstTotal` | yes (rec_adv_1READ_pct(numerator)) | no | yes (WR(rec_adv_1READ_pct), TE(rec_adv_1READ_pct)) | no | no |
| `playerStatsReceivingAverageDepthOfTarget` | yes (rec_adv_aDOT(value)) | no | yes (WR(rec_adv_aDOT)) | no | no |
| `playerStatsReceivingDeepTargetTargetsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingPasserRatingWhenTargeted` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingAveragesPerRouteYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingAveragesPerTargetYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingAveragesPerTargetYardsOverExpected` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingAveragesPerReceptionYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingAveragesPerReceptionYardsAfterCatch` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingAveragesPerReceptionYardsAfterContact` ← UNUSED | no | no | no | no | no |
| `playerStatsFirstDownsReceiving` ← UNUSED | no | no | no | no | no |
| `playerStatsXfpPprTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsXfpPprReceiving` ← UNUSED | no | no | no | no | no |
| `playerStatsOffenseThreatsPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNonPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsHalfPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsHalfPprRb` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsPprTe` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsDraftKings` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsFanDuel` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsCbs` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNfl` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsEspn` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsYahoo` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsFfpc` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNffc` ← UNUSED | no | no | no | no | no |
| `playerStatsInside20ReceivingTargetsTotal` ← UNUSED | no | no | no | no | no |
| `marketShareReceivingRoutesTotal` ← UNUSED | no | no | no | no | no |
| `marketShareReceivingTargetsTotal` | yes (rec_adv_TGT_pct(value)) | yes (target_share_asof) | yes (RB(rec_adv_TGT_pct)) | yes (target_share_asof, target_share_breakout_flag) | yes (target_share_asof, target_share_breakout_flag) |
| `marketShareReceivingYardsTotal` ← UNUSED | no | no | no | no | no |
| `marketShareReceivingYardsAir` | yes (rec_adv_AY_Share(value)) | yes (rec_ay_share_asof) | yes (WR(rec_adv_AY_Share)) | yes (rec_ay_share_asof) | yes (rec_ay_share_asof) |
| `marketShareReceivingTouchdownsTotal` ← UNUSED | no | no | no | no | no |
| `marketShareReceivingTargetedReadFirst` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingAlignmentInlineRoutesPercentage` | yes (rec_routes_INLINE_RTE_pct(value)) | no | yes (TE(rec_routes_INLINE_RTE_pct)) | no | no |
| `playerStatsReceivingTargetsInEndzone` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingReceptionsHero` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingAlignmentBackfieldRoutesPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingTouchdownsInEndzone` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingTargetedReadDesignTotal` | yes (rec_adv_DESIGN_pct(numerator)) | no | no | no | no |
| `playerStatsReceivingTargetedReadDesignPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingDropsTotal` | yes (rec_adv_DRP_pct(numerator)) | no | yes (WR(rec_adv_DRP_pct), TE(rec_adv_DRP_pct)) | no | no |
| `playerStatsReceivingAveragesPerTargetDropsTotal` ← UNUSED | no | no | no | no | no |

### receiving_basic (1,144 rows × 41 cols; 28 stat rows below, 13 meta cols excluded)

| column | in_recipe | in_asof | in_comp | in_matrix | in_filter |
|---|---|---|---|---|---|
| `playerStatsSnapsOffenseTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingTargetsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingReceptionsTotal` | yes (rec_basic_REC(primary)) | no | yes (RB(total_touches_per_game via rec_basic_REC)) | no | no |
| `playerStatsReceivingReceptionsPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingAveragesPerTargetYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingAveragesPerReceptionYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNonPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsHalfPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsHalfPprRb` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsPprTe` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsDraftKings` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsFanDuel` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsCbs` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNfl` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsEspn` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsYahoo` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsFfpc` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNffc` ← UNUSED | no | no | no | no | no |
| `playerStatsGamesPlayed` ← UNUSED | no | no | no | no | no |
| `marketShareReceivingTargetsTotal` ← UNUSED | no | no | no | no | no |
| `marketShareReceivingYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingTouchdownsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsInside10ReceivingTargetsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsInside20ReceivingTargetsTotal` ← UNUSED | no | no | no | no | no |
| `marketShareReceivingTouchdownsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingFumblesTotal` ← UNUSED | no | no | no | no | no |

### receiving_man_vs_zone (1,144 rows × 57 cols; 21 stat rows below, 36 meta cols excluded)

| column | in_recipe | in_asof | in_comp | in_matrix | in_filter |
|---|---|---|---|---|---|
| `playerStatsGamesPlayed` ← UNUSED | no | no | no | no | no |
| **`bucket`** (JSON) | see Bucket-JSON sub-table below | n/a | n/a | n/a | n/a |
| `playerStatsReceivingRoutesTotal` | yes (rec_mz_YPRR_overall(denominator)) | no | no | no | no |
| `playerStatsReceivingTargetsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingTargetsPerRoute` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingReceptionsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingYardsTotal` | yes (rec_mz_YPRR_overall(numerator)) | no | no | no | no |
| `playerStatsReceivingAveragesPerRouteYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNonPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsHalfPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsHalfPprRb` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsPprTe` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsDraftKings` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsFanDuel` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsCbs` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNfl` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsEspn` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsYahoo` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsFfpc` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNffc` ← UNUSED | no | no | no | no | no |

### receiving_route_share_report (1,144 rows × 15 cols; 3 stat rows below, 12 meta cols excluded)

| column | in_recipe | in_asof | in_comp | in_matrix | in_filter |
|---|---|---|---|---|---|
| `playerStatsGamesPlayed` ← UNUSED | no | no | no | no | no |
| **`grid`** (JSON) | see Bucket-JSON sub-table below | n/a | n/a | n/a | n/a |
| `marketShareReceivingRoutesTotal` | yes (rec_route_share_pct(primary (game_mean))) | yes (route_share_asof) | no | yes (route_share_asof) | yes (route_share_asof) |

### receiving_routes_run (1,144 rows × 62 cols; 27 stat rows below, 35 meta cols excluded)

| column | in_recipe | in_asof | in_comp | in_matrix | in_filter |
|---|---|---|---|---|---|
| `playerStatsGamesPlayed` ← UNUSED | no | no | no | no | no |
| `playerStatsSnapsOffenseTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingRoutesTotal` | yes (rec_routes_WIDE_RTE_pct(denominator), rec_routes_SLOT_RTE_pct(denominator)) | no | yes (WR(rec_routes_WIDE_RTE_pct), WR(rec_routes_SLOT_RTE_pct), TE(rec_routes_SLOT_RTE_pct)) | no | no |
| `playerStatsReceivingTargetsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingTargetsPerRoute` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingAlignmentInlineRoutesTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingAlignmentInlineTargetsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingAlignmentSlotRoutesTotal` | yes (rec_routes_SLOT_RTE_pct(numerator)) | no | yes (WR(rec_routes_SLOT_RTE_pct), TE(rec_routes_SLOT_RTE_pct)) | no | no |
| `playerStatsReceivingAlignmentWideRoutesTotal` | yes (rec_routes_WIDE_RTE_pct(numerator)) | no | yes (WR(rec_routes_WIDE_RTE_pct)) | no | no |
| `marketShareReceivingRoutesTotal` ← UNUSED | no | no | no | no | no |
| `marketShareReceivingAlignmentInlineRoutesTotal` ← UNUSED | no | no | no | no | no |
| `marketShareReceivingAlignmentInlineRoutesPerDropback` ← UNUSED | no | no | no | no | no |
| `marketShareReceivingAlignmentSlotRoutesTotal` ← UNUSED | no | no | no | no | no |
| `marketShareReceivingAlignmentSlotRoutesPerDropback` ← UNUSED | no | no | no | no | no |
| `marketShareReceivingAlignmentWideRoutesTotal` ← UNUSED | no | no | no | no | no |
| `marketShareReceivingAlignmentWideRoutesPerDropback` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingAlignmentBackfieldRoutesTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingAlignmentBackfieldTargetsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingAlignmentBackfieldYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingAveragesPerRouteYardsTotal` ← UNUSED | no | no | no | no | no |
| `marketShareReceivingAlignmentBackfieldRoutesTotal` ← UNUSED | no | no | no | no | no |
| `marketShareReceivingAlignmentBackfieldRoutesPerDropback` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingAlignmentSlotTargetsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingAlignmentSlotYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingAlignmentWideTargetsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingAlignmentWideYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingAlignmentInlineYardsTotal` ← UNUSED | no | no | no | no | no |

### receiving_separation_by_alignment (1,144 rows × 46 cols; 17 stat rows below, 29 meta cols excluded)

| column | in_recipe | in_asof | in_comp | in_matrix | in_filter |
|---|---|---|---|---|---|
| `playerStatsGamesPlayed` ← UNUSED | no | no | no | no | no |
| **`bucket`** (JSON) | see Bucket-JSON sub-table below | n/a | n/a | n/a | n/a |
| `playerStatsReceivingTargetsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingTargetsPerRoute` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingReceptionsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingYardsAirTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingTouchdownsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingSeparationRoutesTotal` | yes (rec_sep_align_overall_SEP_SCORE(weight), rec_sep_align_overall_WIN_RATE(weight)) | yes (sep_overall_asof) | yes (WR(sep_overall via rec_sep_align_overall_SEP_SCORE), WR(win_rate_overall via rec_sep_align_overall_WIN_RATE)) | yes (sep_overall_asof) | yes (sep_overall_asof) |
| `playerStatsReceivingSeparationRoutesDepthPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingSeparationScorePercentage` | yes (rec_sep_align_overall_SEP_SCORE(value)) | yes (sep_overall_asof) | yes (WR(sep_overall via rec_sep_align_overall_SEP_SCORE)) | yes (sep_overall_asof) | yes (sep_overall_asof) |
| `playerStatsReceivingSeparationByScoreStep` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingSeparationWinsPercentage` | yes (rec_sep_align_overall_WIN_RATE(value)) | no | yes (WR(win_rate_overall via rec_sep_align_overall_WIN_RATE)) | no | no |
| `playerStatsReceivingAveragesPerRouteYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingSeparationByScoreOpen` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingSeparationByScoreNegative` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingSeparationByScoreWideOpen` ← UNUSED | no | no | no | no | no |

### receiving_separation_by_breaks (1,144 rows × 32 cols; 3 stat rows below, 29 meta cols excluded)

| column | in_recipe | in_asof | in_comp | in_matrix | in_filter |
|---|---|---|---|---|---|
| `playerStatsGamesPlayed` ← UNUSED | no | no | no | no | no |
| **`bucket`** (JSON) | see Bucket-JSON sub-table below | n/a | n/a | n/a | n/a |
| `playerStatsReceivingSeparationRoutesTotal` ← UNUSED | no | no | no | no | no |

### receiving_separation_by_coverage (1,144 rows × 32 cols; 3 stat rows below, 29 meta cols excluded)

| column | in_recipe | in_asof | in_comp | in_matrix | in_filter |
|---|---|---|---|---|---|
| `playerStatsGamesPlayed` ← UNUSED | no | no | no | no | no |
| **`bucket`** (JSON) | see Bucket-JSON sub-table below | n/a | n/a | n/a | n/a |
| `playerStatsReceivingSeparationRoutesTotal` ← UNUSED | no | no | no | no | no |

### receiving_separation_by_routes (1,144 rows × 31 cols; 4 stat rows below, 27 meta cols excluded)

| column | in_recipe | in_asof | in_comp | in_matrix | in_filter |
|---|---|---|---|---|---|
| `playerStatsReceivingTargetsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingSeparationRoutesTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsGamesPlayed` ← UNUSED | no | no | no | no | no |
| **`bucket`** (JSON) | see Bucket-JSON sub-table below | n/a | n/a | n/a | n/a |

### receiving_target_share_report (1,144 rows × 15 cols; 3 stat rows below, 12 meta cols excluded)

| column | in_recipe | in_asof | in_comp | in_matrix | in_filter |
|---|---|---|---|---|---|
| `playerStatsGamesPlayed` ← UNUSED | no | no | no | no | no |
| `marketShareReceivingTargetsTotal` ← UNUSED | no | no | no | no | no |
| **`grid`** (JSON) | see Bucket-JSON sub-table below | n/a | n/a | n/a | n/a |

### rushing_advanced (1,144 rows × 68 cols; 47 stat rows below, 21 meta cols excluded)

| column | in_recipe | in_asof | in_comp | in_matrix | in_filter |
|---|---|---|---|---|---|
| `playerStatsGamesPlayed` ← UNUSED | no | no | no | no | no |
| `playerStatsSnapsOffenseTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsXfpPprTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNonPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsHalfPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsHalfPprRb` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsPprTe` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsDraftKings` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsFanDuel` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsCbs` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNfl` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsEspn` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsYahoo` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsFfpc` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNffc` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingAttemptsTotal` | yes (rush_adv_Success_pct(weight), rush_adv_EXP_RUN_pct(weight), rush_adv_MTF_ATT(denominator) +2 more) | yes (rb_success_rate_asof, rb_success_rate_4wk_delta, rb_mtf_per_att_asof, rb_yaco_per_att_asof) | yes (RB(success_rate via rush_adv_Success_pct), RB(rush_adv_MTF_ATT)) | yes (rb_success_rate_asof, rb_success_rate_4wk_delta, rb_mtf_per_att_asof, rb_yaco_per_att_asof) | yes (rb_success_rate_asof, rb_success_rate_4wk_delta, rb_mtf_per_att_asof, rb_yaco_per_att_asof) |
| `playerStatsRushingAttemptsSuccessPercentage` | yes (rush_adv_Success_pct(value)) | yes (rb_success_rate_asof, rb_success_rate_4wk_delta) | yes (RB(success_rate via rush_adv_Success_pct)) | yes (rb_success_rate_asof, rb_success_rate_4wk_delta) | yes (rb_success_rate_asof, rb_success_rate_4wk_delta) |
| `playerStatsRushingYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingYardsPerAttempt` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingYardsBeforeContactPerAttempt` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingYardsAfterContactTotal` | yes (rush_adv_YACO_ATT(numerator)) | yes (rb_yaco_per_att_asof) | no | yes (rb_yaco_per_att_asof) | yes (rb_yaco_per_att_asof) |
| `playerStatsRushingYardsAfterContactPerAttempt` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingYardsAfterContactPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingYardsExplosiveTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingYardsExplosivePercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingRunsExplosivePercentage` | yes (rush_adv_EXP_RUN_pct(value)) | no | no | no | no |
| `playerStatsFirstDownsRushing` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingAttemptsStuffsPercentage` | yes (rush_adv_STUFF_pct(value)) | no | no | no | no |
| `playerStatsRushingMissedTacklesForcedTotal` | yes (rush_adv_MTF_ATT(numerator)) | yes (rb_mtf_per_att_asof) | yes (RB(rush_adv_MTF_ATT)) | yes (rb_mtf_per_att_asof) | yes (rb_mtf_per_att_asof) |
| `playerStatsRushingMissedTacklesForcedPerAttempt` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingConceptZoneAttemptsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingConceptZoneAttemptsPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingConceptZoneYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingConceptZoneYardsPerAttempt` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingConceptZoneAttemptsSuccessPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingConceptManAttemptsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingConceptManAttemptsPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingConceptManAttemptsSuccessPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingConceptManYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingConceptManYardsPerAttempt` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingTouchdownsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingTouchdownsPercentage` ← UNUSED | no | no | no | no | no |
| `marketShareInside5RushingAttemptsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingConceptZoneTouchdownsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingFumblesTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingConceptManTouchdownsTotal` ← UNUSED | no | no | no | no | no |

### rushing_basic (1,144 rows × 68 cols; 49 stat rows below, 19 meta cols excluded)

| column | in_recipe | in_asof | in_comp | in_matrix | in_filter |
|---|---|---|---|---|---|
| `playerStatsGamesPlayed` ← UNUSED | no | no | no | no | no |
| `playerStatsSnapsOffenseTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingTargetsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingReceptionsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingReceptionsPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingTouchdownsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingAveragesPerTargetYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingAveragesPerReceptionYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNonPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsHalfPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsHalfPprRb` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsPprTe` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsDraftKings` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsFanDuel` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsCbs` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNfl` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsEspn` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsYahoo` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsFfpc` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNffc` ← UNUSED | no | no | no | no | no |
| `playerStatsInside20ReceivingTargetsTotal` ← UNUSED | no | no | no | no | no |
| `marketShareReceivingTargetsTotal` ← UNUSED | no | no | no | no | no |
| `marketShareReceivingYardsTotal` ← UNUSED | no | no | no | no | no |
| `marketShareReceivingTouchdownsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsInside10ReceivingTargetsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingScramblesTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingScramblesYards` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingAttemptsTotal` | yes (rush_basic_ATT(primary)) | no | yes (RB(total_touches_per_game via rush_basic_ATT)) | no | no |
| `playerStatsRushingYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingYardsPerAttempt` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingRunsOneOrMorePercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingRunsThreeOrMorePercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingRunsFiveOrMorePercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingRunsTenOrMorePercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingRunsFifteenOrMorePercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsInside20RushingAttemptsTotal` ← UNUSED | no | no | no | no | no |
| `marketShareInside20RushingAttemptsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingTouchdownsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsInside5RushingAttemptsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsInside10RushingAttemptsTotal` ← UNUSED | no | no | no | no | no |
| `marketShareInside5RushingAttemptsTotal` ← UNUSED | no | no | no | no | no |
| `marketShareInside10RushingAttemptsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingRunsTwentyOrMorePercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingFumblesTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingRunsThirtyOrMorePercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsPassingScramblesTouchdowns` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingFumblesTotal` ← UNUSED | no | no | no | no | no |

### rushing_bell_cow (1,144 rows × 49 cols; 29 stat rows below, 20 meta cols excluded)

| column | in_recipe | in_asof | in_comp | in_matrix | in_filter |
|---|---|---|---|---|---|
| `playerStatsFantasyPointsNonPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsHalfPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsHalfPprRb` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsPprTe` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsDraftKings` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsFanDuel` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsCbs` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNfl` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsEspn` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsYahoo` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsFfpc` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsNffc` ← UNUSED | no | no | no | no | no |
| `playerStatsGamesPlayed` ← UNUSED | no | no | no | no | no |
| `teamStatsSnapsOffenseTotal` ← UNUSED | no | no | no | no | no |
| `teamStatsPassingDropbacksTotal` ← UNUSED | no | no | no | no | no |
| `teamStatsRushingAttemptsTotal` | yes (rush_bellcow_ATT_pct(weight)) | no | yes (RB(bellcow_score via rush_bellcow_ATT_pct)) | no | no |
| `teamStatsReceivingTargetsTotal` | yes (rush_bellcow_TGT_pct(weight)) | no | yes (RB(bellcow_score via rush_bellcow_TGT_pct)) | no | no |
| `teamStatsXfpPprTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsSnapsOffenseTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingRoutesTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingTargetsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsXfpPprTotal` ← UNUSED | no | no | no | no | no |
| `marketShareSnapsOffenseTotal` ← UNUSED | no | no | no | no | no |
| `marketShareReceivingRoutesTotal` ← UNUSED | no | no | no | no | no |
| `marketShareReceivingTargetsTotal` | yes (rush_bellcow_TGT_pct(value)) | no | yes (RB(bellcow_score via rush_bellcow_TGT_pct)) | no | no |
| `marketShareXfpPprTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsRushingAttemptsTotal` ← UNUSED | no | no | no | no | no |
| `marketShareRushingAttemptsTotal` | yes (rush_bellcow_ATT_pct(value)) | no | yes (RB(bellcow_score via rush_bellcow_ATT_pct)) | no | no |

### wr_coverage_matchup (1,144 rows × 35 cols; 24 stat rows below, 11 meta cols excluded)

| column | in_recipe | in_asof | in_comp | in_matrix | in_filter |
|---|---|---|---|---|---|
| `playerStatsReceivingRoutesTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsCoverageSchemeCover4ReceivingRoutesTotal` | yes (rec_yprr_vs_zone(helper-denominator)) | yes (rec_yprr_vs_zone_asof) | no | yes (rec_yprr_vs_zone_asof) | yes (rec_yprr_vs_zone_asof) |
| `playerStatsCoverageSchemeCover4ReceivingRoutesPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsGamesPlayed` ← UNUSED | no | no | no | no | no |
| `playerStatsReceivingAveragesPerRouteYardsTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsFantasyPointsPpr` ← UNUSED | no | no | no | no | no |
| `playerStatsCoverageSchemeCover2ReceivingRoutesTotal` | yes (rec_yprr_vs_zone(helper-denominator)) | yes (rec_yprr_vs_zone_asof) | no | yes (rec_yprr_vs_zone_asof) | yes (rec_yprr_vs_zone_asof) |
| `playerStatsCoverageSchemeCover2ReceivingRoutesPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsCoverageSchemeCover2ReceivingYardsPerRoute` | yes (rec_yprr_vs_zone(helper-numerator (YPR*Routes))) | yes (rec_yprr_vs_zone_asof) | no | yes (rec_yprr_vs_zone_asof) | yes (rec_yprr_vs_zone_asof) |
| `playerStatsCoverageSchemeCover2FantasyPointsPprTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsCoverageSchemeCover3ReceivingRoutesTotal` | yes (rec_yprr_vs_zone(helper-denominator)) | yes (rec_yprr_vs_zone_asof) | no | yes (rec_yprr_vs_zone_asof) | yes (rec_yprr_vs_zone_asof) |
| `playerStatsCoverageSchemeCover3ReceivingRoutesPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsCoverageSchemeCover6ReceivingRoutesTotal` | yes (rec_yprr_vs_zone(helper-denominator)) | yes (rec_yprr_vs_zone_asof) | no | yes (rec_yprr_vs_zone_asof) | yes (rec_yprr_vs_zone_asof) |
| `playerStatsCoverageSchemeCover6ReceivingRoutesPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsCoverageSchemeManReceivingRoutesTotal` | yes (rec_yprr_vs_man(helper-denominator)) | yes (rec_yprr_vs_man_asof) | no | yes (rec_yprr_vs_man_asof) | yes (rec_yprr_vs_man_asof) |
| `playerStatsCoverageSchemeManReceivingRoutesPercentage` ← UNUSED | no | no | no | no | no |
| `playerStatsCoverageSchemeManReceivingYardsPerRoute` | yes (rec_yprr_vs_man(helper-numerator (YPR*Routes))) | yes (rec_yprr_vs_man_asof) | no | yes (rec_yprr_vs_man_asof) | yes (rec_yprr_vs_man_asof) |
| `playerStatsCoverageSchemeManFantasyPointsPprTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsCoverageSchemeCover3ReceivingYardsPerRoute` | yes (rec_yprr_vs_zone(helper-numerator (YPR*Routes))) | yes (rec_yprr_vs_zone_asof) | no | yes (rec_yprr_vs_zone_asof) | yes (rec_yprr_vs_zone_asof) |
| `playerStatsCoverageSchemeCover3FantasyPointsPprTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsCoverageSchemeCover4ReceivingYardsPerRoute` | yes (rec_yprr_vs_zone(helper-numerator (YPR*Routes))) | yes (rec_yprr_vs_zone_asof) | no | yes (rec_yprr_vs_zone_asof) | yes (rec_yprr_vs_zone_asof) |
| `playerStatsCoverageSchemeCover4FantasyPointsPprTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsCoverageSchemeCover6FantasyPointsPprTotal` ← UNUSED | no | no | no | no | no |
| `playerStatsCoverageSchemeCover6ReceivingYardsPerRoute` | yes (rec_yprr_vs_zone(helper-numerator (YPR*Routes))) | yes (rec_yprr_vs_zone_asof) | no | yes (rec_yprr_vs_zone_asof) | yes (rec_yprr_vs_zone_asof) |

## Bucket-JSON unpacks

Sampled 5 non-null rows per kind in `2025/week_01/` for top-level key enumeration. Inner-key sets are stable across the 5 samples in every kind except `passing_depth.bucket` (`bucket10To19` appears in 1/5 — the depth-bucket schema may grow as more games stress the snapshot).

### `grid` (4 kinds: `fpts_scored_report`, `offense_snap_share_report`, `receiving_route_share_report`, `receiving_target_share_report`)

Top-level keys: `"1"` (a per-game tile keyed by game index — only one game per week-01 row, so only one numeric key in the sample) plus a `"key"` string identifying the grid layout. Inner keys under `"1"` are **identical to the parquet's flat top-level columns**:

- `fpts_scored_report.grid["1"]` inner: `playerStatsFantasyPointsPpr`, `…HalfPpr`, `…DraftKings`, … — already exposed as flat columns.
- `offense_snap_share_report.grid["1"]` inner: `marketShareSnapsOffenseTotal`, `playerPlayerId` — already flat.
- `receiving_route_share_report.grid["1"]` inner: `marketShareReceivingRoutesTotal`, `playerStatsGamesPlayed` — already flat.
- `receiving_target_share_report.grid["1"]` inner: `marketShareReceivingTargetsTotal`, `playerStatsGamesPlayed` — already flat.

**No recipe parses any `grid` column.** Every `grid` key is unused, but the inner stats are **not** cheap-win candidates — they duplicate flat columns already exposed at parquet root. `grid` is a week-series replay of the row's own stats and adds no new metric surface.

### `passing_depth.bucket` (sparse — `bucketOverall` 5/5, `bucket10To19` 1/5)

| nested key | parsed today? | proposed output | notes |
|---|---|---|---|
| `bucketOverall` | no (UNUSED) | (no new metric vs. flat top-level fields) | inner echoes play-context, not stat-bearing once the recipe consumes the flat columns |
| `bucket10To19` | no (UNUSED) | `pass_depth_10To19_{ATT,CMP,YDS,TD}` family | depth-band passing breakouts; sparse in week 1 (1/5) — **confirm full schema across a full season before wiring**, the depth-band names are likely `bucketShort` / `bucket10To19` / `bucket20To29` / `bucketDeep` |
| _other depth buckets if present in later weeks_ | no | `pass_depth_*` family | requires multi-week sampling to enumerate completely |

### `receiving_man_vs_zone.bucket` (5 keys, all 5/5)

| nested key | parsed today? | proposed output | notes |
|---|---|---|---|
| `bucketOverall` | no (UNUSED) | (duplicates `rec_mz_YPRR_overall` already produced from top-level) | redundant |
| `bucketMan` | no (UNUSED) | `rec_mz_YPRR_MAN` | **replaces current placeholder** — today `rec_mz_YPRR.1` is filled with `overall` value as a non-informative stub (see `_derive_aggregate_metrics`); this bucket has the real per-coverage rate |
| `bucketZone` | no (UNUSED) | `rec_mz_YPRR_ZONE` | same — replaces `rec_mz_YPRR.2` placeholder |
| `bucketSingleHigh` | no (UNUSED) | `rec_mz_YPRR_SINGLEHIGH` | new — single-high safety look |
| `bucketTwoHigh` | no (UNUSED) | `rec_mz_YPRR_TWOHIGH` | new — two-high safety look |

### `receiving_separation_by_alignment.bucket` (5 keys, all 5/5)

| nested key | parsed today? | proposed output | notes |
|---|---|---|---|
| `bucketReceivingSeparationOverall` | no (UNUSED) | duplicates `rec_sep_align_overall_SEP_SCORE` + `…_WIN_RATE` from top-level | redundant |
| `bucketReceivingSeparationWide` | no (UNUSED) | `rec_sep_align_WIDE_SEP_SCORE`, `…_WIN_RATE` | WR-specific separation signal |
| `bucketReceivingSeparationSlot` | no (UNUSED) | `rec_sep_align_SLOT_SEP_SCORE`, `…_WIN_RATE` | slot WR / nickel-relevant |
| `bucketReceivingSeparationInline` | no (UNUSED) | `rec_sep_align_INLINE_SEP_SCORE`, `…_WIN_RATE` | TE-specific |
| `bucketReceivingSeparationBackfield` | no (UNUSED) | `rec_sep_align_BACKFIELD_SEP_SCORE`, `…_WIN_RATE` | RB-receiver-relevant |

### `receiving_separation_by_breaks.bucket` (6 keys, all 5/5) — **entire file unused today (by design)**

**Design note**: the breaks decomposition (Horizontal / Vertical / Static / Shallow / Backfield) and the routes decomposition (Slant / Out / Dig / Hitch / Go / etc., parsed by `_aggregate_separation_by_routes`) measure overlapping things — every NFL route IS a sequence of breaks, so the break-style axis is a coarser projection of the per-route signal. The phase-1.5 comp-config design (`/home/trevor/.claude/plans/before-we-move-on-squishy-squid.md`) explicitly dropped `sep_breaks_mean` as collinear with the kept `sep_routes_mean` — that's why no helper was ever written and the file currently has zero recipe wiring.

**Open question**: which axis is the better feature is not settled. The routes axis is what was wired; the breaks axis is what was deferred. A future ablation could re-test whether the breaks decomposition captures separation difficulty that the routes axis misses (e.g. "the player struggles on horizontal breaks regardless of route family"). Wiring both at the same time would over-saturate the KNN distance metric — the comp-config design phase made the right call to pick one, not the right call about *which* one.

| nested key | parsed today? | proposed output | notes |
|---|---|---|---|
| `bucketReceivingSeparationBreaksOverall` | no (UNUSED) | `rec_sep_breaks_overall_SEP_SCORE` | redundant with `rec_sep_align_overall_SEP_SCORE` even if helper added |
| `bucketReceivingSeparationBreaksHorizontal` | no (UNUSED) | `rec_sep_breaks_HORIZ_SEP_SCORE` | horizontal-stem routes |
| `bucketReceivingSeparationBreaksVertical` | no (UNUSED) | `rec_sep_breaks_VERT_SEP_SCORE` | vertical-stem routes |
| `bucketReceivingSeparationBreaksStatic` | no (UNUSED) | `rec_sep_breaks_STATIC_SEP_SCORE` | no-break routes |
| `bucketReceivingSeparationBreaksShallowUnderneath` | no (UNUSED) | `rec_sep_breaks_SHALLOW_SEP_SCORE` | drag / short shallow routes |
| `bucketReceivingSeparationBreaksBackfield` | no (UNUSED) | `rec_sep_breaks_BACKFIELD_SEP_SCORE` | RB-receiver |

### `receiving_separation_by_coverage.bucket` (8 keys, all 5/5)

| nested key | parsed today? | derived output | notes |
|---|---|---|---|
| `bucketReceivingSeparationMan` | **YES** (helper) | `rec_sep_vs_man` → `sep_vs_man_asof` | wired in `_aggregate_separation_by_coverage` |
| `bucketReceivingSeparationZone` | **YES** (helper) | `rec_sep_vs_zone` → `sep_vs_zone_asof` | wired same |
| `bucketReceivingSeparationRedZone` | no (UNUSED) | `rec_sep_vs_redzone` | situational — TD-relevant for end-zone targets |
| `bucketReceivingSeparationCover2` | no (UNUSED) | `rec_sep_vs_cover2` | per-shell granularity (today's Zone is the pre-folded combo) |
| `bucketReceivingSeparationCover3` | no (UNUSED) | `rec_sep_vs_cover3` | same |
| `bucketReceivingSeparationCover4` | no (UNUSED) | `rec_sep_vs_cover4` | same |
| `bucketReceivingSeparationCover6` | no (UNUSED) | `rec_sep_vs_cover6` | same |
| `bucketReceivingSeparationOverall` | no (UNUSED) | duplicates `rec_sep_align_overall_SEP_SCORE` from sibling file | redundant |

### `receiving_separation_by_routes.bucket` (13 keys, all 5/5) — all parsed by helper

All 13 route keys are parsed by `_aggregate_separation_by_routes`; helper emits `rec_sep_route_SEP_SCORE.0..12` per `_ROUTE_BUCKET_ORDER`. Suffix `.0` (Overall) is consumed by the comp metric `sep_routes_mean` after the per-route mean. Suffixes `.1..12` are produced but only `.0` feeds the comp config today; the others are present in the wide frame but **not consumed by comp / asof / filter / training matrix** — they could be added to any of those surfaces with no helper-side work.

| nested key | parsed today? | output column | downstream consumer |
|---|---|---|---|
| `bucketReceivingSeparationRouteOverall` | YES (helper) | `rec_sep_route_SEP_SCORE` | → `sep_routes_mean` (NFL.WR + NFL.TE comp keys, after route-family mean) |
| `bucketReceivingSeparationRouteSlant` | YES (helper) | `rec_sep_route_SEP_SCORE.1` | produced but unused downstream (cheap surface-3 / surface-2 win — add to `playerCompStats.json` or `_FP_ASOF_COLUMN_MAP`) |
| `bucketReceivingSeparationRouteOut` | YES (helper) | `rec_sep_route_SEP_SCORE.2` | produced but unused downstream (cheap surface-3 / surface-2 win — add to `playerCompStats.json` or `_FP_ASOF_COLUMN_MAP`) |
| `bucketReceivingSeparationRouteInDig` | YES (helper) | `rec_sep_route_SEP_SCORE.3` | produced but unused downstream (cheap surface-3 / surface-2 win — add to `playerCompStats.json` or `_FP_ASOF_COLUMN_MAP`) |
| `bucketReceivingSeparationRouteHitch` | YES (helper) | `rec_sep_route_SEP_SCORE.4` | produced but unused downstream (cheap surface-3 / surface-2 win — add to `playerCompStats.json` or `_FP_ASOF_COLUMN_MAP`) |
| `bucketReceivingSeparationRouteComeback` | YES (helper) | `rec_sep_route_SEP_SCORE.5` | produced but unused downstream (cheap surface-3 / surface-2 win — add to `playerCompStats.json` or `_FP_ASOF_COLUMN_MAP`) |
| `bucketReceivingSeparationRouteCorner` | YES (helper) | `rec_sep_route_SEP_SCORE.6` | produced but unused downstream (cheap surface-3 / surface-2 win — add to `playerCompStats.json` or `_FP_ASOF_COLUMN_MAP`) |
| `bucketReceivingSeparationRoutePost` | YES (helper) | `rec_sep_route_SEP_SCORE.7` | produced but unused downstream (cheap surface-3 / surface-2 win — add to `playerCompStats.json` or `_FP_ASOF_COLUMN_MAP`) |
| `bucketReceivingSeparationRouteGo` | YES (helper) | `rec_sep_route_SEP_SCORE.8` | produced but unused downstream (cheap surface-3 / surface-2 win — add to `playerCompStats.json` or `_FP_ASOF_COLUMN_MAP`) |
| `bucketReceivingSeparationRouteCrossers` | YES (helper) | `rec_sep_route_SEP_SCORE.9` | produced but unused downstream (cheap surface-3 / surface-2 win — add to `playerCompStats.json` or `_FP_ASOF_COLUMN_MAP`) |
| `bucketReceivingSeparationRouteFlat` | YES (helper) | `rec_sep_route_SEP_SCORE.10` | produced but unused downstream (cheap surface-3 / surface-2 win — add to `playerCompStats.json` or `_FP_ASOF_COLUMN_MAP`) |
| `bucketReceivingSeparationRouteScreens` | YES (helper) | `rec_sep_route_SEP_SCORE.11` | produced but unused downstream (cheap surface-3 / surface-2 win — add to `playerCompStats.json` or `_FP_ASOF_COLUMN_MAP`) |
| `bucketReceivingSeparationRouteBackfield` | YES (helper) | `rec_sep_route_SEP_SCORE.12` | produced but unused downstream (cheap surface-3 / surface-2 win — add to `playerCompStats.json` or `_FP_ASOF_COLUMN_MAP`) |

## Cheap-win candidate list

Priority-ordered by estimated LOC ascending. "Recipe LOC" assumes a single `_Recipe(...)` tuple addition to `_AGGREGATE_RECIPES` in `nfl_fp_weekly_aggregate.py` (4 LOC for `sum`/`game_mean`, 8 LOC for `weighted_rate` / `weighted_mean`). "Helper LOC" assumes the pattern of an existing bucket-parser (40–60 LOC for a new helper, ~10 LOC to add a key to an existing one). Each Tier-1 entry that you also want in the asof matrix takes ~3 more LOC (one map entry in `_FP_ASOF_COLUMN_MAP` plus the asof name in `feature_filter.json`).

### Tier 1 — top-level columns, single `_Recipe` row (~4–8 LOC each)

Flat top-level columns already in the parquet — the column exists, the recipe just doesn't reference it yet.

| parquet kind | unused column(s) | proposed recipe output | pattern | LOC | note |
|---|---|---|---|---|---|
| receiving_advanced | `playerStatsReceivingTouchdownsTotal` | `rec_adv_TD` | sum | 4 | TD-market feature; trivial |
| receiving_advanced | `playerStatsFirstDownsReceiving` | `rec_adv_FD` | sum | 4 | first-downs market |
| receiving_advanced | `playerStatsReceivingYardsAfterCatchTotal` | `rec_adv_YAC` | sum | 4 | YAC distinct from total yards |
| receiving_advanced | `playerStatsReceivingYardsAfterContactTotal` | `rec_adv_YAC_after_contact` | sum | 4 | broken-tackle yards |
| receiving_advanced | `playerStatsReceivingDeepTargetTargetsTotal` | `rec_adv_DEEP_TGT` | sum | 4 | deep-target volume; pairs with aDOT |
| receiving_advanced | `playerStatsReceivingTargetsCatchableTotal` + `…TargetsTotal` | `rec_adv_CATCH_pct` | weighted_rate | 8 | catchable-target share |
| receiving_advanced | `playerStatsReceivingReceptionsContested` | `rec_adv_CTGT_REC` | sum | 4 | high-contested-target receivers |
| rushing_advanced | `playerStatsRushingYardsTotal` | `rush_adv_YDS` | sum | 4 | recipe currently has only rate stats for rushing — adding the level is the lowest-LOC win |
| rushing_advanced | `playerStatsFirstDownsRushing` | `rush_adv_FD` | sum | 4 | first-downs market |
| rushing_advanced | `playerStatsRushingTouchdownsTotal` | `rush_adv_TD` | sum | 4 | TD-market feature |
| rushing_advanced | `playerStatsRushingYardsBeforeContactPerAttempt` + `…AttemptsTotal` | `rush_adv_YBC_ATT` | weighted_mean | 8 | YBC is the o-line side of yaco_per_att |
| rushing_advanced | `playerStatsRushingYardsExplosiveTotal` | `rush_adv_EXP_YDS` | sum | 4 | explosive yardage; pairs with `rush_adv_EXP_RUN_pct` |
| rushing_advanced | `playerStatsRushingConceptZoneAttemptsPercentage` + `…AttemptsTotal` | `rush_adv_ZONE_pct` | weighted_mean | 8 | zone-scheme share — defenses split by scheme |
| rushing_advanced | `playerStatsRushingConceptManAttemptsPercentage` + `…AttemptsTotal` | `rush_adv_MAN_pct` | weighted_mean | 8 | gap-scheme share |
| passing_advanced | `playerStatsPassingYardsAirTotal` | `pass_adv_AY` | sum | 4 | air yards (vs. total yards) |
| passing_advanced | `playerStatsFirstDownsPassing` | `pass_adv_FD` | sum | 4 | first-downs market |
| passing_advanced | `playerStatsPassingDeepThrowAttemptsTotal` + `…AttemptsTotal` | `pass_adv_DEEP_pct` | weighted_rate | 8 | deep-attempt share |
| passing_advanced | `playerStatsPassingAverageTimeToThrow` + `…DropbacksTotal` | `pass_adv_TTT` | weighted_mean | 8 | time-to-throw (oline + scheme proxy) |
| passing_advanced | `playerStatsPassingOffTargetThrowAttemptsPercentage` + `…AttemptsTotal` | `pass_adv_OFFTARGET_pct` | weighted_mean | 8 | accuracy signal |
| passing_advanced | `playerStatsPassingHeroThrowPercentage` + `…AttemptsTotal` | `pass_adv_HERO_pct` | weighted_mean | 8 | high-difficulty throw rate |
| passing_advanced | `playerStatsPassingAttemptsCatchablePercentage` + `…AttemptsTotal` | `pass_adv_CATCHABLE_pct` | weighted_mean | 8 | placement-quality signal |
| offense_snaps | `playerStatsInside10SnapsOffenseTotal` | `off_snaps_INSIDE10` | sum | 4 | RZ snap volume — goal-line backs / WRs |
| offense_snaps | `playerStatsInside20SnapsOffenseTotal` | `off_snaps_INSIDE20` | sum | 4 | RZ snap volume |
| offense_snaps | `marketShareInside10SnapsOffenseTotal` + `playerStatsGamesPlayed` | `off_snaps_INSIDE10_pct` | weighted_mean | 8 | RZ snap share |
| rushing_bell_cow | `marketShareXfpPprTotal` + `teamStatsXfpPprTotal` | `rush_bellcow_XFP_pct` | weighted_rate | 8 | expected-FP market-share; complements ATT/TGT% bellcow components |
| efficiency | `playerStatsXfpPprTotal` | `eff_XFP` | game_mean | 4 | xFP isolates volume + air yards from realised outcome |
| efficiency | `playerStatsScrimmageTouchesTotal` | `eff_TOUCHES` | sum | 4 | total touches workload denominator |
| receiving_basic | `playerStatsInside10ReceivingTargetsTotal`, `…Inside20…` | `rec_basic_INSIDE10_TGT`, `_INSIDE20_TGT` | sum | 8 | red-zone target volume for TD markets |
| rushing_basic | `playerStatsInside5RushingAttemptsTotal`, `…Inside10…`, `…Inside20…` | `rush_basic_INSIDE{5,10,20}_ATT` | sum | 12 | goal-line carry volume for rushing-TD markets |

### Tier 2 — extend an existing bucket helper (~5–25 LOC each)

| parquet kind | unused bucket key(s) | helper to extend | proposed output(s) | LOC |
|---|---|---|---|---|
| receiving_separation_by_coverage | `bucketReceivingSeparationRedZone` | `_aggregate_separation_by_coverage` (add scheme entry) | `rec_sep_vs_redzone` | 5 |
| receiving_separation_by_coverage | `bucketReceivingSeparationCover{2,3,4,6}` | `_aggregate_separation_by_coverage` (un-fold the zone bucket) | `rec_sep_vs_cover{2,3,4,6}` | 15 |
| wr_coverage_matchup | per-scheme YPRR for Cover2/3/4/6 (already read but currently rolled into one `zone` bucket) | extend `_aggregate_per_coverage_yprr` to emit per-scheme outputs | `rec_yprr_vs_cover{2,3,4,6}` | 20 |
| receiving_separation_by_routes (already parses 13 keys) | `.1..12` already produced — only surface 3 / 4 / 5 wiring is needed | n/a (no helper change) — add to `playerCompStats.json` and `_FP_ASOF_COLUMN_MAP` | per-route asof family (e.g. `rec_sep_route_GO_asof`) | 5 per route added (config edits only) |

### Tier 3 — new bucket-parser helper (~40–60 LOC each)

Entire JSON columns currently dead. The two existing JSON helpers (`_aggregate_separation_by_routes`, `_aggregate_separation_by_coverage`) provide copy-paste templates.

| parquet kind | bucket column | proposed helper name | output family | LOC |
|---|---|---|---|---|
| receiving_separation_by_alignment | `bucket` (4 useful keys after dropping Overall) | `_aggregate_separation_by_alignment_bucket` | `rec_sep_align_{WIDE,SLOT,INLINE,BACKFIELD}_{SEP_SCORE,WIN_RATE}` (8 cols) | 50 |
| receiving_separation_by_breaks | `bucket` (5 useful keys after dropping Overall) — entire file dead **by design** (the breaks axis was dropped during comp-config design as collinear with the routes axis — see bucket sub-table for full design note). Wiring it is cheap to implement (~50 LOC) but **not a clean win**: the comp KNN would over-saturate if breaks and routes are both included. Open question worth a future ablation, not a sure-thing cheap win. | `_aggregate_separation_by_breaks_bucket` | `rec_sep_breaks_{HORIZ,VERT,STATIC,SHALLOW,BACKFIELD}_SEP_SCORE` (5 cols) | 50 (mechanical) + ablation budget |
| receiving_man_vs_zone | `bucket` (4 useful keys after dropping Overall) | `_aggregate_man_vs_zone_bucket` | `rec_mz_YPRR_{MAN,ZONE,SINGLEHIGH,TWOHIGH}` — also **replaces** the current placeholder `rec_mz_YPRR.1/.2` | 50 |
| passing_depth | `bucket` (sparse; needs multi-week schema discovery first) | `_aggregate_passing_depth_bucket` | `pass_depth_{SHORT,MED,DEEP}_{ATT,CMP,YDS,TD}` family | 60 (after schema discovery) |

### Tier 4 — out-of-scope here, listed for context

`fantasy_points_allowed.parquet` and `fantasy_points_scored.parquet` would feed defense-grain or game-script features once they enumerate per (team, opponent, week) instead of single-tile placeholders. Estimate post-fetcher-fix: a defense-profile helper similar to `_aggregate_recipes` grouped on opponent-position, ~80–120 LOC. Out of cheap-win pool until upstream snapshot shape stabilises.

## Meta columns excluded from coverage count

Applied uniformly across all 24 kinds. A column matching one of the names below is treated as roster / schedule / play-context metadata rather than a stat value, and is excluded from the M (source columns surveyed) count above.

**Identity / roster**: `playerPlayerId`, `playerFirstName`, `playerLastName`, `playerPosition`, `teamAbbreviation`, `teamLocation`, `teamNickname`, `teamTeamId`, `teamIsHomeTeam`, `opponentAbbreviation`, `opponentTeamId`, `opponentLocation`, `opponentNickname`, `opponentConference`, `opponentDivisionName`.

**Game / schedule**: `gameGameId`, `gameSeason`, `gameSeasonTypeSeasonTypeId`, `gameWeek`.

**Play-context** (descriptors of the in-game play row, not a player stat): `playPlayId`, `playStartClockQuarter`, `playDownNumber`, `playOffensePersonnelKey`, `playPassDropbackTypeName`, `playPassPassResultName`, `playPassTargetedRouteFamily`, `playPassThrowTypeParent`, `playPassReceiverSeparationName`, `playDefenseMiddleOfTheFieldLookPreName`, `playDefenseMiddleOfTheFieldLookPostName`, `playDefenseCoverageSchemeParent`, `playPlayerAlignmentFamily`, `playPlayerAlignmentSide`, `playOffensePrimaryConceptName`.

**Categorical labels** (string bucket labels, not numeric stats): `playerStatsPassingPlayActionLabel`, `playerStatsPassingPressuredLabel`, `playerStatsPassingBlitzedLabel`, `playerStatsReceivingTargetsContestedLabel`, `playerStatsReceivingTargetsCatchableLabel`.

**JSON team/opponent metadata arrays** (roster-shape, not stat values): `teamsPlayedFor`, `opponentsPlayed`.

**Other**: `key` (top-level grid metadata string in `grid`-bearing kinds — identifies the grid layout, no stat value).

**Ambiguous case — kept in stat tables, not excluded**: `playerStatsGamesPlayed` is a row-level game count, used by `off_snaps_Snap_pct` as a denominator and by several `derive_comp_metrics` `per-game` formulas. It is listed in the stat tables and marked `in_recipe yes` only in the kinds where a recipe references it (today: `offense_snap_share_report` only).
