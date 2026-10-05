# I6a: count a push as half an Over in training's labels

Inventory, replay and design, 2026-10-05, branch `devel` at `06d5bb39`. Design note only; no code
was changed. It is the record behind the owner ask in `docs/handoffs/honest-receipts.md` §8.

## Summary (ten lines)

1. **Rule.** One function, `over_label(result, line)` in a new `training/labels.py`, returns 1 for an Over, 0 for an Under and 0.5 for a tie. Every fit and score that averages the label uses it as is; every hit-or-miss statistic (log-loss, AUC, accuracy, precision, the simulated-bet frame) leaves tie rows out.
2. **The ship gates' own label changes in the same commit, and that flips no verdict.** Measured on all 78 served cells: 73 ship today, 73 ship with the gates' label at 0.5, both on the stored test sets and in the retrain projection.
3. **The alternative that leaves the gates alone is the one that flips a verdict.** Once a cell is retrained under the new label and graded under the old one, WNBA TOV fails Gate 5 (0.0204 to 0.1005 against a 0.075 line) and NFL targets lands at 0.0685. Dropping ties instead flips NFL completions (stored sets) and NBA BLST (after a retrain) on Gate 1. Outside the recommended path, NBA PR would turn from fail to pass on Gate 1 (0.0053 to 0.0049, line 0.005) if it were retrained before this change lands and graded after.
4. **How often.** A tie is 2.34% of held-out rows (6,487 of 276,706), 0.73% of the rows Gate 1 reads, 2.31% of validation rows. It is 5% or more in 26 of 78 cells and reaches 18.8% (WNBA TOV).
5. **What it changes.** The temperature moves by more than 0.01 in 43 cells. Served P(Over) falls 0.51 points on average and far more in tie-heavy cells with a Platt or isotonic map: NFL qb tds 11.5, WNBA TOV 9.2, NFL targets 8.1, NBA DREB 6.6. The lane brief's "at most 0.2 pp" is a pooled figure for recommended legs, not a per-cell bound.
6. **The lane brief undercounts the work.** It names two label sites in `pipeline.py`. There are 3 there, 4 in `scorecard.py` and 4 in the structural strategies. Handed a 0.5 label today, Platt (14 served cells) and the legacy mode stats raise, and `_compute_metrics` silently counts a tie as an Under, which would corrupt Kelly shrinkage.
7. **Staking moves.** Kelly shrinkage shifts by more than 0.01 in 11 cells; NBA AST goes from 0 to 0.013 and so starts staking; NBA FG3M falls from 0.177 to 0.115.
8. **Cost.** `pipeline.py` does not grow (about five lines out, five in). No new class, no option flag, no strategy-version bump.
9. **Unknown.** The replay refits only the temperature and the post-hoc map on the stored model, so "no flips" is a projection, not a retrain. In WNBA TOV the new rule moves the read at half-point lines from 3.2 points above the hit rate to 5.8 below; pooled over the 26 tie-heavy cells it improves (+1.37 to -0.47).
10. **Not established.** The supersede checks S2 and S3 (no baseline and candidate pair was replayed), the structural strategies (no cell uses them), the cross-validated Platt path with a 0.5 label (no served cell uses it), and the effect on live recommendations.

## How to read this note

Every number is tagged **measured** (with the script that produced it) or **inferred** (with the file and line it was read from). The scripts are kept on the dev box only, in `~/backups/sportstradamus/2026-10-04-honest-receipts/main/wave4/i6a/`, and run with `poetry run python <script>` from the repository root; none writes to the repository. Paths are relative to `/home/trevor/Sportstradamus/`.

Words used throughout:

| Word | Meaning |
|---|---|
| tie | `Result == Line`. A bet on that line pushes: the stake comes back. |
| today | the label `1 if Result >= Line else 0`. A tie counts as an Over win. |
| half | the approved label: 1 above the line, 0 below, 0.5 at a tie. |
| drop | tie rows left out of the fit or score. |
| served cell | a (league, market) with a model file `src/sportstradamus/data/models/{LEAGUE}_{market}.mdl`. There are 78. |
| held-out rows | `src/sportstradamus/data/test_sets/{LEAGUE}_{market}.csv`, the rows the ship gates grade. |
| T | the temperature: one number per cell that pulls the model's probabilities toward 50%, fit on validation rows. |
| post-hoc map | an optional second map fit on validation rows after T. Four kinds read the label or sit at that stage: Platt, cross-validated Platt, isotonic, and one anchored to the book. |
| g1 | Gate 1: upper end of the interval on (model Brier minus book Brier) on rows with a real sportsbook price. Must be below 0.005. A cell with no priced rows passes. |
| g5 | Gate 5: calibration error on all held-out rows, with a simulated chance level subtracted. Must be below 0.075. |

## 1. Inventory

### 1.1 Which side is already right

The probability side already counts a tie as half an Over. Every distribution family returns `P(under) = P(X < line) + P(X == line) / 2` (inferred: `helpers/distributions.py:570-580` negative binomial, `:583-596` double Poisson, `:599-610` SkewNormal, `:613-624` Gamma, `:653-657` mixture, `:706` Poisson). The label side does not. So today a model with a perfect distribution is graded as under-reading the Over on every integer line, by half the tie mass, and the temperature and post-hoc map are fit to remove a gap that is only a difference of convention.

Measured sign of that: NFL sacks-taken has no post-hoc map and 17.5% ties. Its Gate 5 number is 0.0696 under today's label (line 0.075) and -0.0095 under the half label (`m3_gates.py`, Table A3).

### 1.2 Sites that derive an over/under outcome from a result and a line

"Takes 0.5 today?" was tested by calling the production function with a 0.5 label on WNBA TOV validation rows (measured: `m5_consumers.py`), unless marked inferred.

| # | File:line | Expression | What consumes it | Takes 0.5 today? | Probability convention on the other side |
|---|---|---|---|---|---|
| 1 | `training/pipeline.py:3212` (`_step_calibrate_temperature`) | `(y_validation.Result >= B_validation.Line).astype(int)` | Fit: temperature (`_brier_temperature_loss` `:2847`, called `:3250`) | yes | half-split, then T |
| 1a | same array, `:3256` | | Metric: `model_calib` (1 minus Brier, stored only) | yes (inferred: plain arithmetic) | same |
| 1b | same array, `:4281-4287` | | Fit: post-hoc map `posthoc.fit_posthoc` | isotonic yes; book-anchored yes (reads no label); **Platt raises; cross-validated Platt raises** | same, after T |
| 1c | same array, `:5002-5008` | | Metrics: `_step_compute_skill_metrics` `:1939` to `_compute_metrics` `:659` on priced validation rows: Brier, log-loss, AUC, calibration error, accuracy, precision both sides, over rates, and from them `brier_skill_score` and `kelly_shrinkage` | **silently wrong**: `.astype(int)` at `:662` turns 0.5 into 0, so a tie becomes an Under | model: same; book: the stored quote |
| 2 | `training/pipeline.py:4304-4307` (`_structural_gate_inputs`) | `(Result >= Line).astype(int)` on validation | Metrics: `model_calib` and the skill metrics, for the structural strategies | as 1a and 1c | group-conditional settlement, also half-split (inferred: `group_conditional_cdf/_line_head.py`) |
| 3 | `training/pipeline.py:5009-5011` (`train_market`) | `(y_test.Result >= B_test.Line).astype(int)` | Metrics: legacy mode stats (`_step_compute_mode_stats` `:2006`: precision, accuracy, log-loss, under precision). Stored in the model file under `stats`; nothing outside `pipeline.py` reads them (measured: `git grep`) | **raises** | half-split, three stages of calibration |
| 3a | same array | | Metrics: `_diag_over_pcts` `:2129`, the `over_pct_ev_gt` / `over_pct_ev_lt` columns | yes | none (compares expected value with the line) |
| 4 | `training/scorecard.py:538` (`_calibration_inputs`) | `(Result >= Line).astype(float)` | **Gate 5** (`_gate5_ece_debiased`), the `ece_*` columns, the oracle rows | yes | stored `P` column of the test set (half-split, calibrated) |
| 5 | `training/scorecard.py:593` (`_brier_inputs`) | same | **Gate 1** (`_gate1_brier_ci`), `brier_skill_score` on held-out rows, `_standalone_g1_hi` | yes | model: stored `P`; book: `1 - Odds`, the quoted price with the margin removed |
| 6 | `training/scorecard.py:2447-2449` (`_supersede_paired_brier_ci`) | same | Supersede check S2: paired Brier of candidate against incumbent | yes (inferred: plain arithmetic; not executed) | stored `P` of each test set |
| 7 | `training/scorecard.py:2493` (`_test_set_to_bet_frame`) | `hit_over = result >= line`; side from `pred >= line` `:2489` | Supersede check S3: simulated Kelly bets, binary `Hit` column | needs a win or a loss; a tie is neither | stored `P` |
| 8 | `training/group_conditional_cdf/_pipeline_steps_shared.py:129` | `(result >= line).astype(float)` | Metrics for the structural sweep: out-of-fold Brier intervals, calibration error, rate error | yes (inferred) | group-conditional, half-split |
| 9 | `training/group_conditional_cdf/_pipeline_steps_two_part.py:107`, `:399` | same | Fit: `fit_two_part_groupcdf` temperature; support audit class counts (`_pipeline_steps_two_part_support.py:224`, `:237`, `np.bincount(outcome.astype(int))`) | **temperature raises** (guard `_validation.py:160-161`); the audit would count a tie as an Under (inferred) | same |
| 10 | `training/group_conditional_cdf/_fit_affine.py:84` | `(y >= lines).astype(float)` | Fits: `fit_temperature_affine`, `fit_probability_pool` | temperature yes; **pool raises** (guard `probability_pool.py:58-59`) | same |

No other file in `training/` derives the outcome. `calibration.py` (blend weight), the mean and dispersion corrections, the PIT map, `shap.py`, `report.py` and `model_strategy/` read the result as a number, never as over or under (inferred: read of each file; `git grep` for `>= .*Line`, `> .*Line`, `Result.*Line`).

### 1.3 Sites that already treat a tie differently, and why

| File:line | Expression | Why it differs | Change? |
|---|---|---|---|
| `training/data.py:194-215` (`_balance_over_under`) | strict `Result > Line` | Tie rows are split off first (`:195`, `:197`) and added back at the archived push rate. The strict test never sees a tie. | no |
| `analysis.py:93-102`, `:133-144` | Over / Under / Push | Live settlement: a tie is a push. | no |
| `realized.py:82`, `nightly.py:250`, `:282` | keeps Over and Under | Receipts count settled bets only. | no |
| `strategies/_ledger_settlement.py`, `prediction/payouts.py` | leg push | Parlay settlement. | no |
| `scripts/tail_scorecard.py:221`, `:251-252` | three-way; ties dropped | Written after the tie problem was known. | no |
| `scripts/backtest_combo_quotes.py:102`, `:123`; `scripts/plot_parlay_hist.py:121-123` | skip ties | Same. | no |
| `scripts/test_distributions.py:223`; `scripts/model_calibration/test_model_weight.py:165`, `:301`, `:385`, `:432` | `>=` | Standalone diagnostics; write nothing a gate or a model reads. | no (see 4.6) |
| `dashboard/components/deep_dive_charts.py:162`, `:263`; `form_spark.py:56` | `value >= line` | Display: a tie is drawn as a hit in a player's form chart. | no (flag only, see 4.6) |
| `stats/base.py:2488`; `helpers/training_quotes.py:126` | `<` on a half-point subline; a rescale of a probability | Not outcome labels. | no |

## 2. How often a tie happens

Measured: `m1_tie_rates.py` (12 seconds). Held-out rows are loaded the way the gates load them (`scorecard.load_test_set`). Validation rows were reached without retraining: the temperature-ridge harness (`s02_replay.py`, kept beside these scripts' parent folder under `temp_ridge/`) had already rebuilt each served cell's validation split from the cached training matrix and the served model file; the label it captured equals `Result >= Line` in 78 of 78 cells.

| Scope | Cells | Held-out rows | Ties | Tie % | Gate-1 rows | Gate-1 ties | Gate-1 tie % | Validation rows | Val ties | Val tie % | Cells at 5% or more |
|---|---|---|---|---|---|---|---|---|---|---|---|
| All served cells | 78 | 276,706 | 6,487 | 2.34 | 180,038 | 1,315 | 0.73 | 276,249 | 6,390 | 2.31 | 26 |

| League | Cells | Held-out rows | Ties | Tie % | Gate-1 tie % | Val tie % | Cells at 5% or more | Highest cell % |
|---|---|---|---|---|---|---|---|---|
| MLB | 15 | 102,948 | 174 | 0.17 | 0.13 | 0.15 | 0 | 4.31 |
| NBA | 18 | 38,745 | 2,390 | 6.17 | 4.50 | 6.03 | 9 | 13.88 |
| NFL | 19 | 31,223 | 984 | 3.15 | 2.35 | 3.18 | 6 | 18.60 |
| NHL | 12 | 73,304 | 499 | 0.68 | 0.13 | 0.63 | 2 | 7.19 |
| WNBA | 14 | 30,486 | 2,440 | 8.00 | 1.91 | 8.06 | 9 | 18.76 |

| Distribution family | Cells | Held-out rows | Ties | Tie % | Gate-1 tie % | Val tie % | Cells at 5% or more | Highest cell % |
|---|---|---|---|---|---|---|---|---|
| Double Poisson (count) | 27 | 126,598 | 3,230 | 2.55 | 0.66 | 2.49 | 9 | 18.76 |
| Negative binomial (count) | 9 | 49,910 | 328 | 0.66 | 0.47 | 0.62 | 3 | 7.38 |
| Zero-inflated negative binomial (count) | 11 | 44,501 | 984 | 2.21 | 0.04 | 2.33 | 5 | 18.60 |
| All count families | 47 | 221,009 | 4,542 | 2.06 | 0.50 | 2.04 | 17 | 18.76 |
| SkewNormal (continuous) | 31 | 55,697 | 1,945 | 3.49 | 2.84 | 3.41 | 9 | 17.49 |

Two things to carry forward. Ties are rare where Gate 1 looks (0.73%) because books mostly quote half-point lines; they are common where only Gate 5 looks, on integer pick'em lines. And 9 of the 14 highest-tie cells have no priced rows at all, so Gate 5 is the only gate that reads their label. Table A1 at the end lists all 78 cells.

## 3. Replay

### 3.1 Method and its limits

`m2_refit.py` (5 seconds) starts from the harness's stored validation probabilities and raw held-out probabilities for each served cell and refits, under each label, the two things in the calibration chain that read it: the temperature (current objective, Brier alone, bounds 1 to 10) and the Platt or isotonic map. It then applies both to the held-out rows. `m3_gates.py` (1 minute 50) runs the production `scorecard.compute_gates` on each cell's test set, swapping in-process only the two functions that build the gates' label. `m4_summary.py` tabulates.

What is held fixed: the booster, the blend weight, the mean, dispersion and PIT corrections, and the rows. A real retrain re-tunes the booster and adds rows. So every "after a retrain" number below is a projection.

Checks on the harness (measured: `m2_refit.py`):

| Check | Result |
|---|---|
| Cells replayed | 78 of 78, all rows matched to the stored test set |
| Refit with the old penalised objective against the T in the model file | within 0.012 in every cell; differs by more than 0.0001 in 5 (NBA BLST, NBA RA, NFL qb tds, NFL yards, NHL powerPlayPoints) |
| Soft-label Platt on a binary label against today's Platt | coefficients equal to 6e-15 |
| Baseline gates against `model_stats.parquet` `ship` | 78 of 78 agree (73 ship) |
| Replayed held-out P against the stored `P` column | exact in 65 cells; 13 differ, largest row difference NBA FG3M 0.168, NBA PA 0.047, NHL powerPlayPoints 0.029, NBA BLST 0.016, under 0.004 in the other 9 |

Because of the last row, arms are compared replay against replay, and the stored test sets are graded separately.

### 3.2 Ship gates, three ways, on the stored test sets

This is what happens to a cell that has not been retrained when only the gates' label changes. `report()` regrades every model file's stored test set at the end of each league pass (inferred: `training/report.py:100`, `:322`), so this state begins at the first `meditate` after the change for every cell at once.

Gates 2, 3, 4 and 6 read neither a label nor `P`; their metrics shift by exactly 0 in every arm (measured: `m3_gates.py`, `m4_gate_shift.csv`).

| Gate | Metric | Cells | Half: mean shift | Half: largest rise | Half: largest fall | Half: flips | Drop: mean shift | Drop: largest rise | Drop: largest fall | Drop: flips |
|---|---|---|---|---|---|---|---|---|---|---|
| g1 | interval upper end | 48 | +0.0006 | +0.0218 NBA FG3M (-0.0384 to -0.0166) | -0.0061 NFL receptions | none | +0.0007 | +0.0204 NBA FG3M | -0.0069 NHL skater fantasy | **NFL completions pass to fail (0.0041 to 0.0053)** |
| g2 | star z | 78 | 0 | 0 | 0 | none | 0 | 0 | 0 | none |
| g3 | bench z | 78 | 0 | 0 | 0 | none | 0 | 0 | 0 | none |
| g4 | PIT distance | 78 | 0 | 0 | 0 | none | 0 | 0 | 0 | none |
| g5 | debiased calibration error | 78 | -0.0062 | +0.0595 NFL targets (0.0000 to 0.0595) | -0.0791 NFL sacks-taken (0.0696 to -0.0095) | none | -0.0053 | +0.0447 NFL qb tds | -0.0622 NFL sacks-taken | none |
| g6 | three one-sided legs | 78 | 0 | 0 | 0 | none | 0 | 0 | 0 | none |
| ship | cells shipping | 78 | 73 to 73 | | | **none** | 73 to 72 | | | NFL completions |

How many cells move (half label): g1 by more than 0.0005 in 17 of 48, by more than 0.005 in 4; g5 by more than 0.0005 in 49 of 78, by more than 0.005 in 33, by more than 0.02 in 17.

Cells nearest a line in this state (half label, stored test sets). The gates are seeded, so these are the numbers `report()` will print as long as the test-set files do not change.

| Cell | Gate | Today | Half label | Line | Verdict |
|---|---|---|---|---|---|
| NFL qb tds | g5 | 0.0149 | 0.0625 | 0.075 | pass, pass |
| NFL targets | g5 | 0.0000 | 0.0595 | 0.075 | pass, pass |
| WNBA TOV | g5 | 0.0203 | 0.0333 | 0.075 | pass, pass |
| NFL completions | g1 | 0.0041 | 0.0048 | 0.005 | pass, pass |
| NBA REB | g1 | 0.0040 | 0.0041 | 0.005 | pass, pass |
| NBA PR | g1 | 0.0056 | 0.0052 | 0.005 | fail, fail |
| WNBA RA | g1 | 0.0086 | 0.0080 | 0.005 | fail, fail |

Why NFL qb tds and NFL targets rise: their Platt and isotonic maps were fit to today's label, so their stored probabilities sit about half the tie mass too high for the half label. A retrain removes it (next table).

### 3.3 Ship gates after a retrain (projection)

Measured: `m2_refit.py`, `m3_gates.py`, `m4_summary.py`. "Before" is the replay under today's label graded under today's label.

| Fits use | Gates use | Cells shipping | g1 mean shift | g1 largest rise | g5 mean shift | g5 largest rise | Verdict flips, by name |
|---|---|---|---|---|---|---|---|
| half | half | 73 to 73 | +0.0009 | +0.0235 NBA FG3M | -0.0078 | +0.0211 WNBA FTM | **none** |
| half | today | 73 to 72 | +0.0003 | +0.0032 NFL interceptions | +0.0044 | +0.0801 WNBA TOV | **WNBA TOV g5 pass to fail (0.0204 to 0.1005)** |
| drop | drop | 73 to 72 | +0.0011 | +0.0226 NBA FG3M | -0.0069 | +0.0246 WNBA FTM | **NBA BLST g1 pass to fail (0.0016 to 0.0060)** |
| drop | today | 73 to 71 | +0.0004 | +0.0044 NFL interceptions | +0.0049 | +0.0924 WNBA TOV | **NBA BLST g1 fail (0.0052); WNBA TOV g5 fail (0.1128)** |
| today, temperature refit without the penalty | half | 73 to 74 | +0.0008 | +0.0242 NBA FG3M | -0.0042 | +0.0595 NFL targets | **NBA PR g1 fail to pass (0.0053 to 0.0049)** |

The last row is a cell retrained on today's code and then graded with the new gate label. It cannot happen if the change lands as one commit, because a cell retrained after the commit is fit under half.

The cells nearest a line, after a retrain, in each arm. Bold marks a fail.

| Cell | Gate | Line | Before (fits today, gates today) | After (fits half, gates half) | Alternative (fits half, gates today) | Ties dropped in both |
|---|---|---|---|---|---|---|
| NFL qb-tds | g5 | 0.075 | 0.0169 | 0.0094 | 0.0393 | 0.0131 |
| NFL targets | g5 | 0.075 | -0.0000 | 0.0049 | 0.0685 | 0.0021 |
| WNBA TOV | g5 | 0.075 | 0.0204 | 0.0126 | **0.1005** | 0.0196 |
| NFL sacks-taken | g5 | 0.075 | 0.0703 | -0.0077 | 0.0709 | 0.0081 |
| NFL completions | g1 | 0.005 | 0.0041 | 0.0023 | 0.0025 | 0.0026 |
| NBA BLST | g1 | 0.005 | 0.0016 | 0.0037 | 0.0038 | **0.0060** |
| NBA REB | g1 | 0.005 | 0.0029 | 0.0026 | 0.0025 | 0.0028 |
| NBA PR | g1 | 0.005 | **0.0053** | **0.0052** | **0.0059** | **0.0055** |
| WNBA RA | g1 | 0.005 | **0.0086** | **0.0087** | **0.0095** | **0.0089** |

Table A3 gives g1 and g5 for all 78 cells in every arm.

### 3.4 Temperature and served probability

Measured: `m2_refit.py`, `m4_summary.py`. "Today" here is the refit with the current objective under today's label, so the shift isolates the label. Table A2 gives every cell.

| Quantity | Value |
|---|---|
| Cells where T moves by more than 0.01 | 43 of 78 |
| Cells where T moves by more than 0.1 | 22 (T falls in 31 of the 43, rises in 12) |
| Mean change in P(Over), all held-out rows | -0.51 points (drop: -0.54) |
| Mean absolute change | 0.61 points (drop: 0.65) |
| Cells with mean absolute change above 0.1 / 1 / 5 points | 41 / 23 / 4 |

| League | Cells | Mean change, pp | Mean absolute change, pp | Same, half-point lines only, pp | Largest cell | Its mean absolute change, pp |
|---|---|---|---|---|---|---|
| MLB | 15 | -0.004 | 0.020 | 0.018 | runs-allowed | 1.01 |
| NBA | 18 | -1.09 | 1.29 | 1.17 | DREB | 6.61 |
| NFL | 19 | -1.07 | 1.12 | 0.58 | qb tds | 11.45 |
| NHL | 12 | -0.08 | 0.18 | 0.14 | hits | 2.18 |
| WNBA | 14 | -1.88 | 2.23 | 1.92 | TOV | 9.16 |

| Post-hoc map of the cell | Cells | Mean change, pp | Mean absolute change, pp | Largest cell, pp |
|---|---|---|---|---|
| Platt (reads the label) | 14 | -1.71 | 1.72 | 11.45 |
| isotonic on the probability (reads the label) | 8 | -1.81 | 1.81 | 8.14 |
| isotonic on the mean | 8 | -0.99 | 1.50 | 4.44 |
| none | 39 | -0.12 | 0.20 | 3.30 |
| ratio-of-expectations on the mean | 3 | -0.23 | 0.61 | 1.66 |
| isotonic on the distribution | 6 | -0.02 | 0.23 | 0.84 |

The seventeen cells that move most:

| Cell | Post-hoc map | Val tie % | T today | T half | Mean change in P(Over), pp | Same, half-point lines only, pp |
|---|---|---|---|---|---|---|
| NFL qb-tds | Platt | 23.0 | 2.023 | 1.886 | -11.45 | -11.30 |
| WNBA TOV | Platt | 18.4 | 1.583 | 1.100 | -9.16 | -9.00 |
| NFL targets | isotonic on the probability | 16.1 | 1.000 | 1.000 | -8.14 | -6.48 |
| NBA DREB | isotonic on the probability | 13.5 | 1.147 | 1.137 | -6.61 | -6.01 |
| WNBA STL | isotonic on the mean | 12.3 | 1.794 | 1.023 | -3.51 | -2.67 |
| WNBA AST | Platt | 8.8 | 1.142 | 1.122 | -4.37 | -4.36 |
| WNBA BLST | isotonic on the mean | 16.8 | 1.857 | 1.163 | -2.12 | -1.74 |
| WNBA OREB | none | 12.5 | 1.819 | 1.308 | -2.82 | -2.98 |
| NBA OREB | none | 13.1 | 1.772 | 1.218 | -2.54 | -2.31 |
| NBA FTM | Platt | 5.8 | 2.063 | 1.750 | -2.90 | -2.91 |
| NBA RA | isotonic on the probability | 4.4 | 1.482 | 1.523 | -2.28 | -2.25 |
| NHL hits | isotonic on the probability | 4.4 | 1.831 | 1.451 | -2.18 | -1.90 |
| NFL completions | Platt | 3.5 | 4.209 | 3.957 | -1.76 | -1.78 |
| WNBA BLK | none | 5.9 | 1.446 | 1.303 | -1.69 | -1.78 |
| NBA BLST | ratio-of-expectations on the mean | 10.8 | 1.406 | 1.153 | -0.63 | -0.43 |
| WNBA FTM | none | 8.4 | 2.117 | 1.733 | -0.90 | -0.86 |
| NFL carries | Platt | 3.1 | 6.646 | 5.170 | -1.51 | -1.50 |

The temperature and the post-hoc map are one setting per cell, so the shift lands on every row of the cell, including rows on half-point lines that can never tie (last column). The largest shifts are in cells whose Platt or isotonic map reads the label.

### 3.5 Does the new rule make the probabilities better where a tie cannot happen?

Measured: `m6_halfpoint_and_null.py` (47 seconds). A half-point line cannot tie, so all three label rules grade those held-out rows identically. Only the served probability differs. "Read minus hit" is mean P(Over) minus the share of Overs.

| Rows | Cells | Rows | Read minus hit, today | Read minus hit, half | Brier today | Brier half | Log-loss today | Log-loss half |
|---|---|---|---|---|---|---|---|---|
| Half-point lines, all cells | 78 | 236,855 | +0.76 pp | +0.48 pp | 0.1974 | 0.1974 | 0.5753 | 0.5751 |
| Half-point lines, the 26 cells with 5% ties or more | 26 | 26,987 | +1.37 pp | -0.47 pp | 0.2298 | 0.2292 | 0.6514 | 0.6500 |
| Half-point lines, the other 52 cells | 52 | 209,868 | +0.69 pp | +0.60 pp | 0.1933 | 0.1933 | 0.5655 | 0.5654 |
| Integer lines, tie rows left out (see caveat) | 69 | 33,374 | +1.98 pp | +0.35 pp | 0.2213 | 0.2199 | 0.6324 | 0.6294 |

Caveat on the last row: on an integer line P(Over) includes half the tie mass, so it is not meant to equal the win rate among settled bets exactly. The row shows direction only.

Pooled, the new rule is no worse on any of these and slightly better in the tie-heavy cells. Cell by cell it is mixed: the absolute gap at half-point lines shrinks in 25 cells and grows in 28; Brier there improves in 28 and worsens in 23. The cells with the largest moves (Table A5 has all 26):

| Cell | Half-point rows | Read minus hit, today | Read minus hit, half | Brier shift |
|---|---|---|---|---|
| WNBA TOV | 905 | +3.21 pp | **-5.78 pp** | +0.0023 (worse) |
| NFL qb tds | 155 | +5.69 pp | -5.61 pp | -0.0007 |
| NFL targets | 945 | +3.42 pp | -3.06 pp | -0.0029 (better) |
| NBA DREB | 859 | +3.46 pp | -2.55 pp | -0.0012 |
| WNBA AST | 1,349 | +3.82 pp | -0.54 pp | -0.0015 |
| WNBA OREB | 1,372 | +3.75 pp | +0.77 pp | -0.0020 |
| WNBA BLK | 1,878 | +4.41 pp | +2.63 pp | -0.0015 |

Reading (inferred, not measured): a cell has one post-hoc map for all its lines. Under today's label the integer-line rows pull that map up by half the tie mass, which over-reads the half-point lines. Under half the two conventions agree, so what remains at half-point lines is the model's own error by line type, which one map cannot fix. In WNBA TOV that leftover is large and of the opposite sign.

### 3.6 Kelly shrinkage

Measured: `m2_refit.py`, `m4_kelly.csv`. Kelly shrinkage is the model's Brier skill against the book on priced validation rows, clipped to 0..1 (inferred: `training/pipeline.py:1939-1990`); live staking multiplies by it (inferred: `training/report.py:443`).

| Quantity | Value |
|---|---|
| Cells with priced validation rows | 48 |
| Cells with Kelly shrinkage above 0, today / half | 42 / 44 |
| Cells crossing zero | NBA AST 0 to 0.0127 (ships, so it starts staking); MLB runs-allowed 0 to 0.0014 (fails Gate 4, so it does not) |
| Cells moving by more than 0.001 / 0.01 | 22 / 11 |
| Largest moves | NBA FG3M 0.177 to 0.115; NFL receptions 0.030 to 0.056; NHL hits 0.058 to 0.078; NHL skater fantasy 0.089 to 0.108; NBA STL 0.114 to 0.096 |

Table A4 lists the 22 cells.

### 3.7 Post-hoc map variants and the Gate 5 chance level

Variants were replayed on the 19 cells that have a Platt or isotonic map and validation ties (measured: `m2_refit.py`, `m3_gates.py`):

| Temperature fit on | Platt or isotonic fit on | Cells shipping | Verdict flips against the first row | Gate 5 under the half gate label: NFL qb-tds / NFL targets / NBA DREB / WNBA TOV |
|---|---|---|---|---|
| half | half, as a soft label (**recommended**) | 73 | none | 0.0094 / 0.0049 / 0.0124 / 0.0126 |
| half | today's rule kept | 74 | NBA PR fail to pass | 0.0632 / 0.0595 / 0.0389 / 0.0333 |
| half | tie rows dropped | 73 | none | -0.0022 / 0.0302 / 0.0149 / 0.0194 |

Gate 5 subtracts a chance level simulated by drawing each outcome as a coin with the model's probability (inferred: `scorecard.py:1665`). A 0.5 label is less noisy than a coin, so that level is slightly too generous. Measured (`m6_halfpoint_and_null.py`): the excess is at most 0.0041 (NFL sacks-taken), above 0.002 in 2 cells and above 0.001 in 4, against a 0.075 line.

## 4. Design

### 4.1 The helper

New file `src/sportstradamus/training/labels.py`, about 15 lines, one constant and one function:

```python
"""Over/under outcome label shared by training's fits, metrics and ship gates."""

import numpy as np

# A tie pushes and the stake comes back, so it is half an Over. Matches
# helpers.distributions, whose P(under) is P(X < line) + P(X == line) / 2.
PUSH = 0.5


def over_label(result, line) -> np.ndarray:
    """Return 1.0 where ``result`` beats ``line``, 0.0 where it falls short, ``PUSH`` at a tie.

    Positional: ``result`` and ``line`` are same-length arrays or Series in the same row order.
    """
    result = np.asarray(result, dtype=float)
    line = np.asarray(line, dtype=float)
    return np.where(result == line, PUSH, (result > line).astype(float))
```

Semantics, each one the same as today except the tie:

| Case | Today | Helper |
|---|---|---|
| result above line | 1 | 1.0 |
| result below line | 0 | 0.0 |
| result equals line (exact float equality, the same test `>=` and `data.py:195` use) | 1 | 0.5 |
| result or line missing | 0 | 0.0 |
| alignment | pandas refuses to compare two Series whose indexes differ in order, so label order and row order already coincide at `:3212` and `:5010` (inferred from pandas' comparison rule; `:4304` is already positional) | positional |

A caller that needs settled rows writes `settled = y != PUSH`. No second function, no class, no parameter choosing the rule.

### 4.2 Who calls it, and what each consumer does with a tie

The principle: a consumer that takes the mean of the label gets the 0.5. A consumer that counts hits and misses gets settled rows only, because a pushed bet is neither and because that is what the live receipts already count (`realized.py:82`).

| Site | Edit | Tie handling | Why |
|---|---|---|---|
| `pipeline.py:3212` | `y_class_val = over_label(splits["y_validation"]["Result"], B_validation["Line"])` | 0.5 | Temperature is a Brier fit; already accepts it. |
| `posthoc.py:376` (`_platt_coeffs`) | the two `LogisticRegression(...).fit(feat, y)` calls go through one private `_logistic_fit(feat, y, **kwargs)` that enters each row once as an Over with weight `y` and once as an Under with weight `1 - y` (six lines; shape in `m2_refit.py:soft_platt`) | 0.5 | Keeps one target for the whole chain. Equal to today's fit on a binary label to 6e-15. The penalised branch already uses `_bernoulli_nll`, valid for a fractional label (inferred: `posthoc.py:398`). The `len(np.unique(y)) < 2` guards at `:310` and `:333` keep their meaning: a constant label. |
| `posthoc._fit_isotonic :301` | none | 0.5 | Already accepts it. |
| `pipeline.py:659` (`_compute_metrics`) | drop `.astype(int)`; Brier becomes `np.mean((probs - y) ** 2)`; `settled = y != PUSH`; log-loss, AUC, accuracy and both precisions read `probs[settled]`, `y[settled].astype(int)` | 0.5 for Brier, calibration error (inferred to accept it: `pipeline.py:642`, a binned mean) and `empirical_over_rate`; settled rows for the rest | Brier feeds `brier_skill_score` and Kelly shrinkage and must use the same label as the fit. The others are hit-or-miss statistics and sklearn refuses a 0.5. |
| `pipeline.py:4304-4307` | four lines become `y_class_val = over_label(splits["y_validation"]["Result"], splits["B_validation"]["Line"])` | 0.5 | Same consumers as above. |
| `pipeline.py:5009-5011` | three lines become `y_class = over_label(splits["y_test"]["Result"], splits["B_test"]["Line"])` | 0.5 into `_diag_over_pcts` | Already accepts it. |
| `pipeline.py:2006` (`_step_compute_mode_stats`) | first line `settled = y_class != PUSH`, then index its four inputs by it | settled rows | Precision, accuracy, log-loss. Keeping today's rule would go on counting a tie as an Over win. |
| `scorecard.py:538`, `:593`, `:2447` | `over_label(sub[ACTUAL_COL], sub["Line"])` | 0.5 | Gate 5, Gate 1 and S2 are means of a squared or absolute error. See 4.3. |
| `scorecard.py:2493` (`_test_set_to_bet_frame`) | drop tie rows before building the frame | settled rows | A simulated bet that pushes returns the stake; `Hit` stays binary. |
| `group_conditional_cdf/_pipeline_steps_shared.py:129`, `_pipeline_steps_two_part.py:107`, `:399`, `_fit_affine.py:84` | `over_label(...)`; the three "binary 0/1" guards (`_validation.py:160`, `probability_pool.py:58`, `_line_head.py:45`) become "within 0 and 1"; the support audit counts settled rows | 0.5 in the fits, settled rows in the class counts | See 4.4. |

Line count for `pipeline.py` (5,117 lines today): the three expressions give back five lines; the import and the two settled-row masks take about five. Net zero, give or take a line. `scorecard.py` edits are in place.

Fallback if the soft-label Platt is unwanted: fit Platt and isotonic on settled rows only. Measured, it gives the same ship verdicts as the soft label (3.7). It costs a mask at the `fit_posthoc` call in `pipeline.py` and makes the post-hoc map chase the win rate among settled bets while the temperature chases the half label, two targets in one chain. Not recommended.

### 4.3 The gates' own label

Recommended: Gate 1, Gate 5 and S2 switch to the half label in the same commit as the fits.

| Option | Stored test sets (cells not yet retrained) | After a retrain (projection) |
|---|---|---|
| **Gates on half, same commit (recommended)** | 73 ship, no flip | 73 ship, no flip |
| Gates left as today (the alternative) | 73 ship, no flip (nothing changes) | **72 ship: WNBA TOV fails Gate 5 at 0.1005**; NFL targets at 0.0685 |
| Gates drop ties | 72 ship: NFL completions fails Gate 1 at 0.0053 | 72 ship: NBA BLST fails Gate 1 at 0.0060 |

The alternative is offered because the brief asks for it, and it is available at no code cost: leave `scorecard.py` untouched. Its price is that every tie-heavy cell is then fit to one convention and graded on another, and the grade error is half the tie mass. Gate 5 is where that shows.

One asymmetry stays in Gate 1 under any rule (inferred, not measured): a sportsbook's two-way price on an integer line is the chance of an Over among settled bets, while the model's probability and the half label both include half the tie mass. The book is therefore graded slightly off its own convention, by about (tie share) times (its probability minus 0.5) per integer-line row, which is under 0.02 in probability for a 20% tie share and under 0.0005 in Brier. Gate-1 rows tie 0.73% of the time.

### 4.4 The structural strategies

No served cell and no configured cell uses a two-part or affine strategy (measured: 0 of 78 model files, 0 of 93 entries in `data/config/stat_meta.json`). They are still in the sweep's pool, so a sweep can put one on any cell.

Recommended: convert their four label sites in the same change. If they stay on `>=`, a cell later moved to one is fit with a tie as an Over and graded with a tie as a half, which is the mismatched pairing measured in 3.3. The cost is four expressions, three guard messages and one class count, in code this replay could not exercise. If the owner would rather not touch unexercised code, the cheaper choice is to leave them and record the gap in the lane brief; nothing served changes either way.

### 4.5 No strategy-version bump

`StrategySpec.implementation_version` is part of each strategy's signature (inferred: `training/model_strategy/registry.py:137`, `specs.py:167`, `:279`). A bump would rotate the signature of every family at once; `report()` would then skip the gates for every stored model until it is retrained (inferred: `training/report.py`, `_identity_block`), and serving checks the same signature when it loads a model (inferred: `prediction/model_prob.py:343-345`, not executed). That changes which cells are served, which is outside this change.

The nearest precedent, the removal of the temperature penalty in an earlier commit (`a7ca8e21`), did not bump and relied on "a cell picks the change up at its next meditate". This design does the same. The standing rule in `docs/handoffs/low_weight_models.md:21-22` ("a bug fix that changes how an artifact is produced is an `implementation_version` bump") points the other way; the owner should confirm the precedent applies.

None of the files touched is on the research-gate list (`.claude/research_gated.txt` holds four distribution-family files), and no distribution family or dispersion mechanism changes.

### 4.6 What stays as it is

| Thing | Why |
|---|---|
| `helpers/distributions.py` | Already counts a tie as half. It is the side the label is being brought into line with. |
| `training/data.py:194-215` | Removes ties before its strict test. |
| Blend weight, mean and dispersion corrections, PIT map | Read no over/under label. |
| Gates 2, 3, 4, 6 | Read no over/under label. |
| Gate 5's simulated chance level | Leaving the coin-flip simulation costs at most 0.0041 of leniency (3.7). Changing it would add code for no verdict. |
| `analysis.py`, `realized.py`, `nightly.py`, ledger settlement, payouts | Already treat a tie as a push. |
| `scripts/tail_scorecard.py`, `backtest_combo_quotes.py`, `plot_parlay_hist.py` | Already drop ties. |
| `scripts/test_distributions.py`, `scripts/model_calibration/test_model_weight.py`, the `data/research/` snapshot | Standalone diagnostics and a frozen research copy; nothing reads their output. Worth a later one-line switch to the helper, not part of this change. |
| Dashboard form charts | Draw a tie as a hit. Display only. Flagged for the owner; not part of this change. |
| The stored `stats` block of the model file | Read by nothing outside `pipeline.py`. Deleting it would shrink `pipeline.py`; noted, not proposed here. |

### 4.7 Tests that fail before the change and pass after it

No existing test pins the tie rule; the fixtures are almost all continuous and never tie (inferred: read of the files below).

| File | New or changed assertion |
|---|---|
| `tests/golden/test_labels.py` (new) | `over_label([3, 2, 1], [2, 2, 2])` is `[1.0, 0.5, 0.0]`; a missing result gives 0.0; a Series with a shuffled index is read by position. Fails today: the module does not exist. |
| `tests/golden/test_calibration_levers.py` | On a constructed validation frame with ties, `_step_calibrate_temperature` returns a label containing 0.5 and a T different from the tie-as-Over fit. |
| `tests/test_posthoc.py`, `tests/golden/test_platt_cv.py` | `fit_posthoc("prob_recal_platt", ...)` and `"prob_recal_platt_cv"` accept a label containing 0.5 (both raise today, measured); on a binary label the coefficients equal today's to 1e-9. |
| `tests/golden/test_pipeline_artifact_parity.py` or a new case beside it | `_compute_metrics` on rows with ties: Brier uses 0.5, `empirical_over_rate` counts a tie as half, accuracy and precision ignore tie rows. Today a 0.5 becomes an Under (measured). `_step_compute_mode_stats` does not raise on a label with ties. |
| `tests/golden/test_scorecard.py` | `_brier_inputs` and `_calibration_inputs` return 0.5 for a tie row; the oracle rows still score zero error; `_test_set_to_bet_frame` leaves a tie row out (the comment at `:1678-1694`, "Hit = (Result >= Line)", is rewritten). |
| `tests/golden/test_two_part_groupcdf.py`, `test_affine_groupcdf.py` | `fit_probability_pool` and the two-part temperature accept 0.5; the assertions on the "binary 0/1" guard messages change. Only if 4.4 is taken. |

Existing Platt tests that compare floats exactly may see differences near 1e-15 from the weighted fit and need a tolerance.

### 4.8 Documents that state the old rule

| File | What to revise in place |
|---|---|
| `docs/ship_gate.md:60` | Gate 1 formula written with `Result >= Line`. |
| `docs/ship_gate.md:203` | Oracle row: "over-prob = 1 if Result>=Line else 0". |
| `training/scorecard.py:527`, `:1901` | Docstrings stating the same. |
| `CLAUDE.md`, model_stats column table | Scoring, Discrimination, Over rates and Kelly rows: say Brier and over rates count a tie as half and that log-loss, AUC, accuracy and precision are on settled rows. |
| `docs/handoffs/honest-receipts.md:251-253` | I6a says "both label sites in `pipeline.py` (at most 0.2 pp)". Replace with the real scope and the per-cell range. |
| `docs/archive/researcher_train_serve_skew.md:176-203`, `:775-776` | Archive; leave, it is a dated record. |

### 4.9 What becomes incomparable across the change

| Stored thing | Comparable across the change? |
|---|---|
| Model file: `temperature`, `posthoc_blob`, `metrics`, `stats`, `diagnostics.model_calib` | No, until the cell is retrained. The `model_version` hash does not record the label rule (inferred: `pipeline.py:2307`), so nothing in the file says which rule made it. |
| Test-set CSV | Schema unchanged: it stores `Result` and `Line`, no label column (inferred: `pipeline.py:2590-2700`). Its `P` column is old-rule until the cell is retrained. |
| `model_stats.parquet` | Mixed from the first `report()` after the change: the gate columns (`g1_*`, `g5_*`, `ece_*`, `ship`) are new-rule for every cell at once; the columns copied from the model file (Scoring, Discrimination, Over rates, Kelly) are old-rule until each cell retrains. |
| Sweep board and nominee-ledger rows | Their g1 and g5 slack was scored under the old label. Comparing a new candidate against an old board row mixes rules. Not replayed. |
| `empirical_over_rate` | Falls by half the tie share in every cell at retrain (9 points in WNBA TOV). A drop in this column is the rule, not the data. |

Suggested order of work, all in one commit so no cell is ever fit under one rule and graded under the other: the helper and its test; `posthoc.py`; `pipeline.py`; `scorecard.py`; the structural sites if taken; documents. Then one `meditate` per league as usual.

## 5. Unknowns and risks, ranked

| # | Risk or unknown | Evidence | What would settle it |
|---|---|---|---|
| 1 | **Per-cell probability shifts are large and not always an improvement.** Served P(Over) falls by up to 11.5 points in a cell. In WNBA TOV the read at half-point lines goes from 3.2 points over the hit rate to 5.8 under, and Brier on those 905 rows worsens by 0.0023. Recommendations in NFL qb tds, WNBA TOV, NFL targets and NBA DREB will lean to the Under. | measured: 3.4, 3.5 | Compare held-out Brier at half-point lines per cell after the real retrain; `m6_halfpoint_and_null.py` re-runs on the new test sets with one path change. |
| 2 | **"No verdict flips" is a projection.** The booster, blend weight and rows are held fixed. NBA PR sits at 0.0052 against a 0.005 line and WNBA RA at 0.0087; both already fail. | measured: 3.1, 3.3 | The retrain itself. |
| 3 | **The window between the commit and each cell's retrain.** The gates regrade stored test sets at once. NFL qb tds reads 0.0625 and NFL targets 0.0595 on Gate 5 (line 0.075) until retrained. They pass, with a margin of about 0.013. | measured: 3.2 | Retrain NFL first, or accept the margin. |
| 4 | **Staking changes.** Kelly shrinkage moves by more than 0.01 in 11 cells; NBA AST starts staking at 0.013. | measured: 3.6 | Owner review of Table A4 before the retrain. |
| 5 | **Gate 1 grades the book slightly off its own convention on integer lines.** Bounded under 0.0005 in Brier on affected rows; not measured. | inferred: 4.3 | Needs the tie mass per quote, which the test sets do not store. |
| 6 | **S2 and S3 were not replayed.** No baseline and candidate pair was at hand. S2 is arithmetic and should accept 0.5; S3 needs the tie rows dropped. | inferred: `scorecard.py:2421-2500` | One supersede run on a tie-heavy cell after the change. |
| 7 | **Structural strategies and cross-validated Platt are converted blind.** No served or configured cell uses either. | measured: counts in 4.4; `m5_consumers.py` shows where they raise | Their unit tests in 4.7. |
| 8 | **13 cells' replay does not reproduce the stored P exactly** (NBA FG3M by up to 0.168 on a row). Arms are compared replay to replay, so the shifts stand, but the absolute "after" values for those cells are less certain. | measured: 3.1 | Not needed for the decision. |
| 9 | **Old and new numbers will sit side by side** in `model_stats.parquet` and on the sweep board with nothing marking which rule made them. | inferred: 4.9 | A dated line in the lane brief; no code. |
| 10 | **A tie is exact float equality.** A line stored as 2.0000001 would not tie. Same as today's `>=` and as `data.py:195`. | inferred | None needed. |

## Scripts and outputs

| Script | Runtime | Produces |
|---|---|---|
| `m1_tie_rates.py` | 12 s | `m1_tie_rates.csv`, `m1_by_league.csv`, `m1_by_dist.csv`, `m1_by_family.csv` |
| `m2_refit.py` | 5 s | `m2_cells.csv`, `m2_rows/{cell}.parquet` (P per arm on the test-set rows) |
| `m3_gates.py` | 1 min 50 s | `m3_gates.csv` (its docstring also names `m3_flips.csv`, which it does not write; flips are in `m4_flips.csv`) |
| `m4_summary.py` | seconds | `m4_summary.log`, `m4_gate_shift.csv`, `m4_flips.csv`, `m4_dP_by_league.csv`, `m4_temperature.csv`, `m4_kelly.csv` |
| `m5_consumers.py` | seconds | `m5_consumers.csv` |
| `m6_halfpoint_and_null.py` | 47 s | `m6.log`, `m6_halfpoint.csv`, `m6_null_offset.csv` |
| `m7_tables.py` | seconds | `appendix.md` (the tables below) |
| `m8_assemble.py` | seconds | this note: fills three per-cell tables of `design_body.md` from the CSVs and appends `appendix.md` |

`m2_refit.py` and `m1_tie_rates.py` read the temperature-ridge harness's replay files (`~/backups/sportstradamus/2026-10-04-honest-receipts/main/temp_ridge/replay/`).

## Appendix: every cell

### Table A1. Tie rate, every served cell (measured: `m1_tie_rates.py`)

Sorted by held-out tie share. Gate-1 rows are the held-out rows with a real sportsbook price; a dash means the cell has none, so Gate 1 passes by default and only Gate 5 reads its label.

| cell | family | post-hoc map | held-out rows | ties | tie % | Gate-1 rows | Gate-1 ties | Gate-1 tie % | validation rows | val ties | val tie % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| WNBA_TOV | DPO | prob_recal_platt | 2255 | 423 | 18.76 | 0 | 0 | - | 2245 | 413 | 18.40 |
| NFL_qb-tds | ZINB | prob_recal_platt | 457 | 85 | 18.60 | 0 | 0 | - | 417 | 96 | 23.02 |
| NFL_sacks-taken | SkewNormal | none | 446 | 78 | 17.49 | 0 | 0 | - | 398 | 61 | 15.33 |
| WNBA_BLST | DPO | isotonic_mean | 2150 | 365 | 16.98 | 0 | 0 | - | 2145 | 361 | 16.83 |
| NFL_targets | SkewNormal | prob_recal_isotonic | 2783 | 425 | 15.27 | 0 | 0 | - | 2815 | 452 | 16.06 |
| WNBA_DREB | DPO | isotonic_mean | 2194 | 321 | 14.63 | 0 | 0 | - | 2194 | 309 | 14.08 |
| NBA_OREB | ZINB | none | 2132 | 296 | 13.88 | 0 | 0 | - | 2180 | 285 | 13.07 |
| NBA_DREB | DPO | prob_recal_isotonic | 2061 | 279 | 13.54 | 0 | 0 | - | 2114 | 285 | 13.48 |
| WNBA_STL | DPO | isotonic_mean | 2254 | 295 | 13.09 | 0 | 0 | - | 2246 | 276 | 12.29 |
| NBA_BLST | DPO | roe_mean | 2192 | 267 | 12.18 | 327 | 6 | 1.83 | 2219 | 240 | 10.82 |
| NBA_FGM | SkewNormal | none | 2102 | 256 | 12.18 | 0 | 0 | - | 2089 | 303 | 14.50 |
| NBA_FG3A | SkewNormal | none | 2157 | 256 | 11.87 | 0 | 0 | - | 2186 | 248 | 11.34 |
| NBA_FG3M | DPO | isotonic_mean | 2193 | 258 | 11.76 | 1777 | 243 | 13.67 | 2224 | 236 | 10.61 |
| WNBA_OREB | ZINB | none | 2211 | 254 | 11.49 | 0 | 0 | - | 2156 | 270 | 12.52 |
| WNBA_FGA | SkewNormal | cdf_recal_isotonic | 2243 | 189 | 8.43 | 0 | 0 | - | 2257 | 175 | 7.75 |
| WNBA_AST | DPO | prob_recal_platt | 2191 | 167 | 7.62 | 663 | 8 | 1.21 | 2191 | 192 | 8.76 |
| NFL_passing-tds | NegBin | isotonic_mean | 447 | 33 | 7.38 | 421 | 33 | 7.84 | 385 | 20 | 5.19 |
| WNBA_FTM | ZINB | none | 2203 | 159 | 7.22 | 0 | 0 | - | 2174 | 183 | 8.42 |
| NHL_sogBS | DPO | none | 2391 | 172 | 7.19 | 0 | 0 | - | 2411 | 154 | 6.39 |
| NFL_receptions | SkewNormal | none | 2936 | 195 | 6.64 | 2415 | 145 | 6.00 | 2970 | 190 | 6.40 |
| NFL_passing-first-downs | SkewNormal | none | 407 | 27 | 6.63 | 0 | 0 | - | 355 | 29 | 8.17 |
| WNBA_BLK | ZINB | none | 2243 | 132 | 5.88 | 0 | 0 | - | 2257 | 133 | 5.89 |
| NBA_AST | SkewNormal | none | 2207 | 117 | 5.30 | 1666 | 116 | 6.96 | 2176 | 102 | 4.69 |
| NBA_REB | NegBin | none | 2227 | 113 | 5.07 | 1954 | 99 | 5.07 | 2226 | 98 | 4.40 |
| NBA_FTM | NegBin | prob_recal_platt | 2202 | 111 | 5.04 | 226 | 0 | 0.00 | 2200 | 127 | 5.77 |
| NHL_shotsAgainst | SkewNormal | none | 679 | 34 | 5.01 | 0 | 0 | - | 726 | 25 | 3.44 |
| NFL_completions | SkewNormal | prob_recal_platt | 442 | 21 | 4.75 | 428 | 21 | 4.91 | 397 | 14 | 3.53 |
| WNBA_FG3M | DPO | isotonic_mean | 2138 | 100 | 4.68 | 705 | 18 | 2.55 | 2114 | 115 | 5.44 |
| NHL_hits | DPO | prob_recal_isotonic | 2461 | 108 | 4.39 | 1460 | 44 | 3.01 | 2321 | 102 | 4.39 |
| MLB_pitches-thrown | SkewNormal | cdf_recal_isotonic | 766 | 33 | 4.31 | 0 | 0 | - | 822 | 28 | 3.41 |
| NBA_MIN | SkewNormal | none | 1966 | 81 | 4.12 | 0 | 0 | - | 1955 | 74 | 3.79 |
| NHL_blocked | DPO | none | 3036 | 113 | 3.72 | 1986 | 2 | 0.10 | 2960 | 97 | 3.28 |
| NBA_RA | DPO | prob_recal_isotonic | 2217 | 81 | 3.65 | 1655 | 69 | 4.17 | 2216 | 98 | 4.42 |
| NBA_STL | DPO | none | 2135 | 70 | 3.28 | 801 | 46 | 5.74 | 2156 | 54 | 2.50 |
| NFL_carries | DPO | prob_recal_platt | 1303 | 42 | 3.22 | 1026 | 42 | 4.09 | 1253 | 39 | 3.11 |
| NBA_PR | SkewNormal | prob_recal_platt | 2195 | 60 | 2.73 | 1793 | 60 | 3.35 | 2170 | 52 | 2.40 |
| NFL_rushing-tds | ZINB | none | 1294 | 28 | 2.16 | 0 | 0 | - | 1283 | 34 | 2.65 |
| NBA_PTS | SkewNormal | none | 2246 | 48 | 2.14 | 1979 | 45 | 2.27 | 2224 | 50 | 2.25 |
| NBA_PA | NegBin | prob_recal_isotonic | 2189 | 44 | 2.01 | 1599 | 44 | 2.75 | 2195 | 47 | 2.14 |
| NFL_interceptions | DPO | isotonic_mean | 455 | 9 | 1.98 | 428 | 9 | 2.10 | 402 | 11 | 2.74 |
| NBA_PRA | SkewNormal | prob_recal_platt | 2212 | 43 | 1.94 | 1951 | 43 | 2.20 | 2214 | 33 | 1.49 |
| MLB_runs-allowed | ZINB | prob_recal_platt | 1101 | 20 | 1.82 | 969 | 5 | 0.52 | 1140 | 20 | 1.75 |
| NFL_attempts | SkewNormal | prob_recal_platt | 441 | 8 | 1.81 | 427 | 8 | 1.87 | 393 | 1 | 0.25 |
| WNBA_RA | SkewNormal | prob_recal_isotonic | 2146 | 15 | 0.70 | 449 | 15 | 3.34 | 2137 | 5 | 0.23 |
| NHL_shots | DPO | none | 7995 | 55 | 0.69 | 7218 | 2 | 0.03 | 7961 | 54 | 0.68 |
| MLB_hits | DPO | none | 10344 | 69 | 0.67 | 10339 | 69 | 0.67 | 10300 | 62 | 0.60 |
| NFL_rushing-yards | SkewNormal | cdf_recal_isotonic | 1352 | 8 | 0.59 | 1115 | 8 | 0.72 | 1310 | 10 | 0.76 |
| NFL_receiving-yards | SkewNormal | prob_recal_isotonic | 2993 | 16 | 0.53 | 2430 | 16 | 0.66 | 3007 | 14 | 0.47 |
| WNBA_PA | SkewNormal | prob_recal_platt | 2094 | 10 | 0.48 | 568 | 10 | 1.76 | 2122 | 8 | 0.38 |
| WNBA_PR | SkewNormal | none | 2121 | 10 | 0.47 | 803 | 10 | 1.25 | 2141 | 8 | 0.37 |
| NBA_BLK | ZINB | prob_recal_platt | 2239 | 8 | 0.36 | 1317 | 8 | 0.61 | 2263 | 13 | 0.57 |
| MLB_total-bases | NegBin | none | 10046 | 26 | 0.26 | 10046 | 26 | 0.26 | 10037 | 15 | 0.15 |
| NHL_skater-fantasy-points-underdog | SkewNormal | none | 2040 | 5 | 0.25 | 234 | 5 | 2.14 | 1945 | 5 | 0.26 |
| NFL_fantasy-points-underdog | SkewNormal | cdf_recal_isotonic | 2983 | 7 | 0.23 | 0 | 0 | - | 2963 | 5 | 0.17 |
| MLB_hits+runs+rbi | DPO | none | 9875 | 23 | 0.23 | 9875 | 23 | 0.23 | 9870 | 32 | 0.32 |
| NBA_fantasy-points-prizepicks | SkewNormal | none | 1873 | 2 | 0.11 | 292 | 2 | 0.68 | 1892 | 1 | 0.05 |
| MLB_hits-allowed | SkewNormal | roe_mean | 1054 | 1 | 0.09 | 1054 | 1 | 0.09 | 1101 | 2 | 0.18 |
| NHL_points | DPO | none | 11836 | 9 | 0.08 | 8042 | 9 | 0.11 | 11728 | 14 | 0.12 |
| NFL_receiving-tds | ZINB | none | 2989 | 2 | 0.07 | 0 | 0 | - | 3011 | 4 | 0.13 |
| NHL_assists | DPO | none | 11789 | 3 | 0.03 | 7976 | 3 | 0.04 | 11649 | 3 | 0.03 |
| MLB_runs | DPO | none | 9817 | 1 | 0.01 | 9663 | 1 | 0.01 | 9748 | 0 | 0.00 |
| MLB_rbi | NegBin | none | 10179 | 1 | 0.01 | 10171 | 1 | 0.01 | 10188 | 0 | 0.00 |
| MLB_walks | DPO | none | 8765 | 0 | 0.00 | 8700 | 0 | 0.00 | 8786 | 0 | 0.00 |
| NFL_fantasy-points-prizepicks | SkewNormal | none | 2851 | 0 | 0.00 | 0 | 0 | - | 2824 | 0 | 0.00 |
| MLB_doubles | DPO | none | 10082 | 0 | 0.00 | 7753 | 0 | 0.00 | 10227 | 0 | 0.00 |
| WNBA_MIN | SkewNormal | cdf_recal_isotonic | 2043 | 0 | 0.00 | 0 | 0 | - | 2027 | 2 | 0.10 |
| MLB_home-runs | ZINB | none | 10004 | 0 | 0.00 | 9901 | 0 | 0.00 | 10032 | 0 | 0.00 |
| NFL_yards | SkewNormal | roe_mean | 2900 | 0 | 0.00 | 0 | 0 | - | 2962 | 0 | 0.00 |
| MLB_pitcher-fantasy-points-underdog | DPO | none | 496 | 0 | 0.00 | 0 | 0 | - | 548 | 0 | 0.00 |
| NFL_qb-yards | SkewNormal | none | 401 | 0 | 0.00 | 0 | 0 | - | 360 | 0 | 0.00 |
| NHL_timeOnIce | SkewNormal | none | 1860 | 0 | 0.00 | 0 | 0 | - | 1966 | 0 | 0.00 |
| MLB_stolen-bases | NegBin | prob_recal_platt | 8446 | 0 | 0.00 | 8446 | 0 | 0.00 | 8450 | 0 | 0.00 |
| NHL_powerPlayPoints | NegBin | prob_recal_isotonic | 10831 | 0 | 0.00 | 6617 | 0 | 0.00 | 10710 | 0 | 0.00 |
| MLB_singles | DPO | none | 9745 | 0 | 0.00 | 9660 | 0 | 0.00 | 9800 | 0 | 0.00 |
| NHL_goals | ZINB | none | 17628 | 0 | 0.00 | 17268 | 0 | 0.00 | 17560 | 0 | 0.00 |
| NHL_goalie-fantasy-points-underdog | SkewNormal | cdf_recal_isotonic | 758 | 0 | 0.00 | 152 | 0 | 0.00 | 799 | 1 | 0.13 |
| NFL_tds | NegBin | isotonic_mean | 3343 | 0 | 0.00 | 3293 | 0 | 0.00 | 3274 | 0 | 0.00 |
| MLB_batter-strikeouts | DPO | prob_recal_platt | 2228 | 0 | 0.00 | 0 | 0 | - | 2380 | 0 | 0.00 |

### Table A2. Temperature and served probability, every served cell (measured: `m2_refit.py`)

`T in model file` was fit with the old penalty; `T today`, `T half` and `T drop` are refits with the current objective (Brier alone) under each label. dP is the change in P(Over) on the cell's held-out rows, `half` minus `today`, in percentage points, with the temperature and the Platt or isotonic map both refit. Sorted by mean absolute dP.

| cell | post-hoc map | T in model file | T today | T half | T shift | T drop | mean dP pp | mean abs dP pp | max abs dP pp | mean dP at half-point lines pp | mean dP (drop) pp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NFL_qb-tds | prob_recal_platt | 1.370 | 2.023 | 1.886 | -0.137 | 1.475 | -11.45 | 11.45 | 12.45 | -11.30 | -10.81 |
| WNBA_TOV | prob_recal_platt | 1.205 | 1.583 | 1.100 | -0.483 | 1.000 | -9.16 | 9.16 | 10.10 | -9.00 | -10.32 |
| NFL_targets | prob_recal_isotonic | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | -8.14 | 8.14 | 18.96 | -6.48 | -6.05 |
| NBA_DREB | prob_recal_isotonic | 1.091 | 1.147 | 1.137 | -0.010 | 1.000 | -6.61 | 6.61 | 9.92 | -6.01 | -6.60 |
| WNBA_STL | isotonic_mean | 1.256 | 1.794 | 1.023 | -0.771 | 1.000 | -3.51 | 4.44 | 12.27 | -2.67 | -3.69 |
| WNBA_AST | prob_recal_platt | 1.063 | 1.142 | 1.122 | -0.020 | 1.029 | -4.37 | 4.37 | 4.63 | -4.36 | -4.27 |
| WNBA_BLST | isotonic_mean | 1.290 | 1.857 | 1.163 | -0.694 | 1.000 | -2.12 | 3.74 | 10.32 | -1.74 | -2.95 |
| WNBA_OREB | none | 1.338 | 1.819 | 1.308 | -0.511 | 1.171 | -2.82 | 3.30 | 7.32 | -2.98 | -3.92 |
| NBA_OREB | none | 1.295 | 1.772 | 1.218 | -0.554 | 1.082 | -2.54 | 3.25 | 8.31 | -2.31 | -3.49 |
| NBA_FTM | prob_recal_platt | 1.336 | 2.063 | 1.750 | -0.313 | 1.649 | -2.90 | 2.90 | 3.30 | -2.91 | -3.37 |
| NBA_RA | prob_recal_isotonic | 1.203 | 1.482 | 1.523 | 0.040 | 1.466 | -2.28 | 2.28 | 8.82 | -2.25 | -2.21 |
| NHL_hits | prob_recal_isotonic | 1.298 | 1.831 | 1.451 | -0.380 | 1.329 | -2.18 | 2.18 | 11.69 | -1.90 | -2.82 |
| NFL_completions | prob_recal_platt | 1.333 | 4.209 | 3.957 | -0.252 | 3.915 | -1.76 | 1.78 | 2.30 | -1.78 | -1.74 |
| WNBA_BLK | none | 1.287 | 1.446 | 1.303 | -0.143 | 1.258 | -1.69 | 1.70 | 2.34 | -1.78 | -2.27 |
| NBA_BLST | roe_mean | 1.153 | 1.406 | 1.153 | -0.253 | 1.060 | -0.63 | 1.66 | 4.43 | -0.43 | -0.92 |
| WNBA_FTM | none | 1.384 | 2.117 | 1.733 | -0.384 | 1.593 | -0.90 | 1.58 | 4.37 | -0.86 | -1.32 |
| NFL_carries | prob_recal_platt | 1.508 | 6.646 | 5.170 | -1.476 | 5.022 | -1.51 | 1.54 | 4.50 | -1.50 | -1.64 |
| NFL_rushing-tds | none | 1.126 | 1.170 | 1.087 | -0.084 | 1.040 | -1.39 | 1.39 | 1.66 | -1.37 | -2.18 |
| NBA_PR | prob_recal_platt | 1.172 | 1.430 | 1.416 | -0.013 | 1.408 | -1.20 | 1.20 | 1.49 | -1.19 | -1.23 |
| NFL_interceptions | isotonic_mean | 1.270 | 2.075 | 1.671 | -0.405 | 1.522 | -0.42 | 1.11 | 4.84 | -0.33 | -0.63 |
| NBA_PA | prob_recal_isotonic | 1.153 | 1.331 | 1.290 | -0.041 | 1.277 | -1.08 | 1.08 | 3.33 | -1.07 | -1.14 |
| MLB_runs-allowed | prob_recal_platt | 1.271 | 2.550 | 3.384 | 0.835 | 3.112 | -0.79 | 1.01 | 6.00 | -0.70 | -0.72 |
| WNBA_FG3M | isotonic_mean | 1.075 | 1.129 | 1.027 | -0.103 | 1.000 | -0.81 | 1.00 | 2.13 | -0.85 | -1.04 |
| NBA_STL | none | 1.238 | 1.714 | 1.502 | -0.212 | 1.463 | -0.73 | 0.89 | 2.79 | -0.69 | -0.89 |
| WNBA_FGA | cdf_recal_isotonic | 1.129 | 1.216 | 1.308 | 0.092 | 1.180 | -0.15 | 0.84 | 1.63 | -0.10 | +0.06 |
| NBA_FG3M | isotonic_mean | 1.000 | 1.000 | 1.092 | 0.092 | 1.000 | -0.29 | 0.80 | 1.98 | -0.18 | +0.00 |
| NHL_sogBS | none | 1.000 | 1.000 | 1.056 | 0.056 | 1.000 | -0.31 | 0.77 | 1.23 | -0.27 | +0.00 |
| NBA_PRA | prob_recal_platt | 1.341 | 2.660 | 2.676 | 0.016 | 2.650 | -0.74 | 0.74 | 0.85 | -0.74 | -0.72 |
| NHL_blocked | none | 1.168 | 1.435 | 1.302 | -0.133 | 1.238 | -0.05 | 0.67 | 2.18 | +0.04 | -0.08 |
| NFL_passing-tds | isotonic_mean | 1.274 | 2.379 | 2.015 | -0.363 | 1.845 | -0.11 | 0.64 | 2.58 | -0.12 | -0.18 |
| NBA_REB | none | 1.100 | 1.259 | 1.381 | 0.122 | 1.313 | -0.10 | 0.56 | 2.07 | -0.07 | -0.05 |
| NHL_shots | none | 1.123 | 1.407 | 1.294 | -0.114 | 1.260 | +0.03 | 0.48 | 1.89 | +0.06 | +0.04 |
| NBA_PTS | none | 1.074 | 1.265 | 1.386 | 0.121 | 1.360 | -0.22 | 0.41 | 2.05 | -0.18 | -0.18 |
| NBA_AST | none | 1.075 | 1.211 | 1.298 | 0.087 | 1.276 | -0.13 | 0.39 | 1.54 | -0.14 | -0.10 |
| NFL_sacks-taken | none | 1.000 | 1.000 | 1.033 | 0.033 | 1.000 | +0.02 | 0.35 | 0.73 | -0.04 | +0.00 |
| NBA_BLK | prob_recal_platt | 1.370 | 1.669 | 1.645 | -0.024 | 1.639 | -0.29 | 0.29 | 0.56 | -0.29 | -0.38 |
| WNBA_PA | prob_recal_platt | 1.124 | 1.344 | 1.311 | -0.032 | 1.305 | -0.19 | 0.26 | 1.33 | -0.20 | -0.21 |
| NBA_FG3A | none | 1.016 | 1.027 | 1.002 | -0.025 | 1.000 | -0.02 | 0.26 | 0.55 | -0.05 | -0.03 |
| NFL_receiving-yards | prob_recal_isotonic | 1.266 | 3.806 | 3.758 | -0.048 | 3.755 | -0.23 | 0.23 | 0.37 | -0.23 | -0.24 |
| NFL_attempts | prob_recal_platt | 1.298 | 3.310 | 3.233 | -0.077 | 3.229 | -0.12 | 0.13 | 0.40 | -0.13 | -0.14 |
| WNBA_RA | prob_recal_isotonic | 1.145 | 1.347 | 1.337 | -0.010 | 1.335 | -0.12 | 0.12 | 0.95 | -0.12 | -0.14 |
| NFL_fantasy-points-underdog | cdf_recal_isotonic | 1.066 | 1.114 | 1.123 | 0.009 | 1.119 | +0.03 | 0.09 | 0.18 | +0.03 | +0.02 |
| MLB_hits-allowed | roe_mean | 1.124 | 1.295 | 1.311 | 0.015 | 1.307 | -0.02 | 0.08 | 0.27 | -0.02 | -0.02 |
| NFL_receiving-tds | none | 1.041 | 1.053 | 1.049 | -0.004 | 1.046 | -0.08 | 0.08 | 0.09 | -0.08 | -0.13 |
| WNBA_PR | none | 1.278 | 2.036 | 2.007 | -0.029 | 2.004 | +0.01 | 0.08 | 0.32 | +0.01 | +0.01 |
| NHL_skater-fantasy-points-underdog | none | 1.091 | 1.129 | 1.124 | -0.006 | 1.122 | -0.05 | 0.08 | 0.11 | -0.06 | -0.07 |
| MLB_hits | none | 1.067 | 1.136 | 1.127 | -0.009 | 1.125 | +0.05 | 0.08 | 0.18 | +0.05 | +0.06 |
| NFL_receptions | none | 1.381 | 3.439 | 3.506 | 0.067 | 3.376 | +0.00 | 0.06 | 0.23 | -0.00 | -0.00 |
| WNBA_MIN | cdf_recal_isotonic | 1.010 | 1.015 | 1.011 | -0.004 | 1.010 | +0.01 | 0.05 | 0.08 | +0.01 | +0.01 |
| NFL_rushing-yards | cdf_recal_isotonic | 1.422 | 4.052 | 3.993 | -0.059 | 3.990 | +0.01 | 0.04 | 0.31 | +0.01 | +0.01 |
| NHL_goalie-fantasy-points-underdog | cdf_recal_isotonic | 1.211 | 1.469 | 1.468 | -0.001 | 1.468 | -0.00 | 0.01 | 0.02 | -0.00 | -0.00 |
| MLB_total-bases | none | 1.053 | 1.146 | 1.147 | 0.001 | 1.146 | +0.00 | 0.01 | 0.02 | +0.00 | +0.00 |
| NHL_assists | none | 1.000 | 1.000 | 1.000 | -0.000 | 1.000 | -0.00 | 0.00 | 0.00 | -0.00 | -0.00 |
| MLB_stolen-bases | prob_recal_platt | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | -0.00 | 0.00 | 0.00 | -0.00 | +0.00 |
| MLB_batter-strikeouts | prob_recal_platt | 1.101 | 1.292 | 1.292 | 0.000 | 1.292 | -0.00 | 0.00 | 0.00 | -0.00 | +0.00 |
| NHL_goals | none | 1.196 | 1.266 | 1.266 | 0.000 | 1.266 | +0.00 | 0.00 | 0.00 | +0.00 | +0.00 |
| WNBA_DREB | isotonic_mean | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | +0.00 | 0.00 | 0.00 | +0.00 | +0.00 |
| MLB_hits+runs+rbi | none | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | +0.00 | 0.00 | 0.00 | +0.00 | +0.00 |
| MLB_home-runs | none | 1.177 | 1.243 | 1.243 | 0.000 | 1.243 | +0.00 | 0.00 | 0.00 | +0.00 | +0.00 |
| MLB_pitcher-fantasy-points-underdog | none | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | +0.00 | 0.00 | 0.00 | +0.00 | +0.00 |
| MLB_pitches-thrown | cdf_recal_isotonic | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | +0.00 | 0.00 | 0.00 | +0.00 | +0.00 |
| MLB_rbi | none | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | +0.00 | 0.00 | 0.00 | +0.00 | +0.00 |
| MLB_runs | none | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | +0.00 | 0.00 | 0.00 | +0.00 | +0.00 |
| MLB_singles | none | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | +0.00 | 0.00 | 0.00 | +0.00 | +0.00 |
| MLB_walks | none | 1.004 | 1.006 | 1.006 | 0.000 | 1.006 | +0.00 | 0.00 | 0.00 | +0.00 | +0.00 |
| NBA_FGM | none | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | +0.00 | 0.00 | 0.00 | +0.00 | +0.00 |
| NBA_MIN | none | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | +0.00 | 0.00 | 0.00 | +0.00 | +0.00 |
| NFL_yards | roe_mean | 1.115 | 1.204 | 1.204 | 0.000 | 1.204 | +0.00 | 0.00 | 0.00 | +0.00 | +0.00 |
| NBA_fantasy-points-prizepicks | none | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | +0.00 | 0.00 | 0.00 | +0.00 | +0.00 |
| NFL_fantasy-points-prizepicks | none | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | +0.00 | 0.00 | 0.00 | +0.00 | +0.00 |
| NHL_timeOnIce | none | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | +0.00 | 0.00 | 0.00 | +0.00 | +0.00 |
| MLB_doubles | none | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | +0.00 | 0.00 | 0.00 | +0.00 | +0.00 |
| NFL_qb-yards | none | 1.223 | 1.400 | 1.400 | 0.000 | 1.400 | +0.00 | 0.00 | 0.00 | +0.00 | +0.00 |
| NHL_shotsAgainst | none | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | +0.00 | 0.00 | 0.00 | +0.00 | +0.00 |
| NFL_tds | isotonic_mean | 1.026 | 1.034 | 1.034 | 0.000 | 1.034 | +0.00 | 0.00 | 0.00 | +0.00 | +0.00 |
| NHL_powerPlayPoints | prob_recal_isotonic | 1.012 | 1.016 | 1.016 | 0.000 | 1.016 | +0.00 | 0.00 | 0.00 | +0.00 | +0.00 |
| NHL_points | none | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | +0.00 | 0.00 | 0.00 | +0.00 | +0.00 |
| NFL_passing-first-downs | none | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | +0.00 | 0.00 | 0.00 | +0.00 | +0.00 |

### Table A3. Gate 1 and Gate 5, every served cell (measured: `m3_gates.py`)

g1 is the upper end of the interval on (model Brier minus book Brier); it must be below 0.005, and a dash means the cell has no priced rows (Gate 1 passes by default). g5 is the debiased calibration error; it must be below 0.075. The first nine columns grade the stored test set as it is and change only the gates' label. Columns starting with R grade the replayed probabilities (temperature and post-hoc map refit under the named training label): `today` = training label and gate label as today, `half+half` = both 0.5 at a tie, `half fits, gates as today` = the alternative that leaves the gates alone.

| cell | g1 today | g1 half | g1 drop | g5 today | g5 half | g5 drop | ships today | ships half | ships drop | R g1 today | R g1 half+half | R g5 today | R g5 half+half | R g5 half fits, gates as today | R ships half+half | R ships half fits, gates as today | R ships drop+drop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MLB_batter-strikeouts | - | - | - | 0.0006 | 0.0006 | 0.0006 | yes | yes | yes | - | - | 0.0007 | 0.0007 | 0.0007 | yes | yes | yes |
| MLB_doubles | -0.0005 | -0.0005 | -0.0005 | 0.0033 | 0.0033 | 0.0033 | yes | yes | yes | -0.0005 | -0.0005 | 0.0033 | 0.0033 | 0.0033 | yes | yes | yes |
| MLB_hits | 0.0007 | 0.0005 | 0.0006 | 0.0073 | 0.0091 | 0.0077 | yes | yes | yes | 0.0005 | 0.0003 | 0.0039 | 0.0062 | 0.0044 | yes | yes | yes |
| MLB_hits+runs+rbi | -0.0005 | -0.0006 | -0.0006 | 0.0079 | 0.0087 | 0.0088 | yes | yes | yes | -0.0005 | -0.0006 | 0.0079 | 0.0087 | 0.0079 | yes | yes | yes |
| MLB_hits-allowed | -0.0001 | -0.0000 | 0.0000 | -0.0105 | -0.0109 | -0.0088 | yes | yes | yes | -0.0007 | -0.0006 | -0.0087 | -0.0089 | -0.0084 | yes | yes | yes |
| MLB_home-runs | -0.0467 | -0.0467 | -0.0467 | 0.0033 | 0.0033 | 0.0033 | yes | yes | yes | -0.0467 | -0.0467 | 0.0040 | 0.0040 | 0.0040 | yes | yes | yes |
| MLB_pitcher-fantasy-points-underdog | - | - | - | 0.0096 | 0.0096 | 0.0096 | NO | NO | NO | - | - | 0.0096 | 0.0096 | 0.0096 | NO | NO | NO |
| MLB_pitches-thrown | - | - | - | 0.0110 | 0.0020 | 0.0098 | yes | yes | yes | - | - | 0.0110 | 0.0020 | 0.0110 | yes | yes | yes |
| MLB_rbi | -0.0002 | -0.0002 | -0.0002 | 0.0102 | 0.0102 | 0.0102 | yes | yes | yes | -0.0002 | -0.0002 | 0.0102 | 0.0102 | 0.0102 | yes | yes | yes |
| MLB_runs | -0.0009 | -0.0009 | -0.0010 | 0.0142 | 0.0142 | 0.0143 | yes | yes | yes | -0.0009 | -0.0009 | 0.0142 | 0.0142 | 0.0142 | yes | yes | yes |
| MLB_runs-allowed | 0.0047 | 0.0041 | 0.0042 | 0.0006 | 0.0025 | -0.0039 | NO | NO | NO | 0.0047 | 0.0037 | 0.0006 | -0.0016 | -0.0015 | NO | NO | NO |
| MLB_singles | 0.0002 | 0.0002 | 0.0002 | -0.0002 | -0.0002 | -0.0002 | yes | yes | yes | 0.0002 | 0.0002 | -0.0002 | -0.0002 | -0.0002 | yes | yes | yes |
| MLB_stolen-bases | -0.0135 | -0.0135 | -0.0135 | 0.0082 | 0.0082 | 0.0082 | yes | yes | yes | -0.0135 | -0.0135 | 0.0082 | 0.0082 | 0.0082 | yes | yes | yes |
| MLB_total-bases | -0.0009 | -0.0011 | -0.0011 | 0.0003 | 0.0005 | 0.0004 | yes | yes | yes | -0.0009 | -0.0011 | 0.0030 | 0.0032 | 0.0031 | yes | yes | yes |
| MLB_walks | -0.0000 | -0.0000 | -0.0000 | -0.0032 | -0.0032 | -0.0032 | yes | yes | yes | -0.0000 | -0.0000 | -0.0030 | -0.0030 | -0.0030 | yes | yes | yes |
| NBA_AST | 0.0010 | -0.0023 | -0.0022 | -0.0033 | -0.0122 | -0.0120 | yes | yes | yes | 0.0013 | -0.0019 | -0.0011 | -0.0101 | 0.0027 | yes | yes | yes |
| NBA_BLK | -0.0354 | -0.0347 | -0.0345 | 0.0003 | -0.0005 | 0.0003 | yes | yes | yes | -0.0355 | -0.0346 | 0.0003 | 0.0000 | 0.0009 | yes | yes | yes |
| NBA_BLST | 0.0032 | 0.0033 | 0.0042 | 0.0326 | -0.0065 | -0.0061 | yes | yes | yes | 0.0016 | 0.0037 | 0.0278 | -0.0074 | 0.0340 | yes | yes | NO |
| NBA_DREB | - | - | - | 0.0134 | 0.0389 | 0.0359 | yes | yes | yes | - | - | 0.0134 | 0.0124 | 0.0532 | yes | yes | yes |
| NBA_FG3A | - | - | - | 0.0379 | -0.0066 | -0.0080 | yes | yes | yes | - | - | 0.0379 | -0.0057 | 0.0382 | yes | yes | yes |
| NBA_FG3M | -0.0384 | -0.0166 | -0.0180 | 0.0451 | 0.0021 | 0.0099 | yes | yes | yes | -0.0396 | -0.0161 | 0.0297 | -0.0054 | 0.0301 | yes | yes | yes |
| NBA_FGM | - | - | - | 0.0383 | -0.0056 | 0.0024 | yes | yes | yes | - | - | 0.0383 | -0.0056 | 0.0383 | yes | yes | yes |
| NBA_FTM | -0.0004 | -0.0004 | -0.0004 | 0.0018 | 0.0006 | 0.0054 | yes | yes | yes | -0.0004 | 0.0002 | 0.0018 | 0.0053 | 0.0140 | yes | yes | yes |
| NBA_MIN | - | - | - | 0.0093 | 0.0037 | 0.0010 | yes | yes | yes | - | - | 0.0093 | 0.0037 | 0.0093 | yes | yes | yes |
| NBA_OREB | - | - | - | 0.0311 | -0.0035 | 0.0074 | yes | yes | yes | - | - | 0.0251 | -0.0068 | 0.0359 | yes | yes | yes |
| NBA_PA | 0.0034 | 0.0030 | 0.0030 | -0.0030 | -0.0093 | -0.0097 | yes | yes | yes | 0.0034 | 0.0035 | -0.0030 | -0.0013 | 0.0079 | yes | yes | yes |
| NBA_PR | 0.0056 | 0.0052 | 0.0054 | 0.0189 | 0.0212 | 0.0195 | NO | NO | NO | 0.0053 | 0.0052 | 0.0186 | 0.0206 | 0.0183 | NO | NO | NO |
| NBA_PRA | 0.0020 | 0.0015 | 0.0019 | 0.0081 | 0.0118 | 0.0133 | yes | yes | yes | 0.0019 | 0.0013 | 0.0076 | 0.0093 | 0.0066 | yes | yes | yes |
| NBA_PTS | 0.0039 | 0.0039 | 0.0040 | 0.0087 | 0.0117 | 0.0109 | yes | yes | yes | 0.0034 | 0.0031 | 0.0075 | 0.0110 | 0.0073 | yes | yes | yes |
| NBA_RA | 0.0016 | 0.0022 | 0.0025 | 0.0059 | 0.0011 | 0.0011 | yes | yes | yes | 0.0016 | 0.0026 | 0.0058 | 0.0099 | 0.0250 | yes | yes | yes |
| NBA_REB | 0.0040 | 0.0041 | 0.0039 | 0.0235 | 0.0043 | 0.0037 | yes | yes | yes | 0.0029 | 0.0026 | 0.0249 | 0.0006 | 0.0258 | yes | yes | yes |
| NBA_STL | -0.0295 | -0.0215 | -0.0205 | 0.0111 | 0.0057 | 0.0059 | yes | yes | yes | -0.0310 | -0.0224 | 0.0071 | 0.0042 | 0.0080 | yes | yes | yes |
| NBA_fantasy-points-prizepicks | -0.0064 | -0.0083 | -0.0074 | 0.0134 | 0.0139 | 0.0137 | yes | yes | yes | -0.0064 | -0.0083 | 0.0134 | 0.0139 | 0.0134 | yes | yes | yes |
| NFL_attempts | 0.0024 | 0.0018 | 0.0018 | 0.0069 | 0.0024 | 0.0055 | yes | yes | yes | 0.0025 | 0.0020 | 0.0069 | 0.0031 | 0.0077 | yes | yes | yes |
| NFL_carries | -0.0004 | -0.0022 | -0.0020 | 0.0042 | 0.0079 | 0.0084 | yes | yes | yes | -0.0004 | -0.0016 | 0.0042 | 0.0067 | 0.0073 | yes | yes | yes |
| NFL_completions | 0.0041 | 0.0048 | 0.0053 | -0.0017 | 0.0040 | 0.0007 | yes | yes | NO | 0.0041 | 0.0023 | -0.0014 | -0.0035 | -0.0068 | yes | yes | yes |
| NFL_fantasy-points-prizepicks | - | - | - | 0.0068 | 0.0068 | 0.0068 | yes | yes | yes | - | - | 0.0068 | 0.0068 | 0.0068 | yes | yes | yes |
| NFL_fantasy-points-underdog | - | - | - | -0.0028 | -0.0033 | -0.0040 | yes | yes | yes | - | - | -0.0018 | -0.0022 | -0.0017 | yes | yes | yes |
| NFL_interceptions | -0.0656 | -0.0608 | -0.0588 | 0.0638 | 0.0539 | 0.0550 | yes | yes | yes | -0.0737 | -0.0649 | 0.0410 | 0.0419 | 0.0497 | yes | yes | yes |
| NFL_passing-first-downs | - | - | - | 0.0030 | -0.0127 | 0.0082 | yes | yes | yes | - | - | 0.0030 | -0.0127 | 0.0030 | yes | yes | yes |
| NFL_passing-tds | -0.0260 | -0.0119 | -0.0107 | 0.0323 | 0.0186 | 0.0092 | yes | yes | yes | -0.0305 | -0.0158 | 0.0122 | 0.0037 | 0.0144 | yes | yes | yes |
| NFL_qb-tds | - | - | - | 0.0149 | 0.0625 | 0.0596 | yes | yes | yes | - | - | 0.0169 | 0.0094 | 0.0393 | yes | yes | yes |
| NFL_qb-yards | - | - | - | 0.0159 | 0.0159 | 0.0159 | yes | yes | yes | - | - | 0.0202 | 0.0202 | 0.0202 | yes | yes | yes |
| NFL_receiving-tds | - | - | - | 0.0071 | 0.0068 | 0.0065 | yes | yes | yes | - | - | 0.0064 | 0.0063 | 0.0066 | yes | yes | yes |
| NFL_receiving-yards | 0.0014 | 0.0011 | 0.0011 | 0.0043 | 0.0042 | 0.0039 | yes | yes | yes | 0.0014 | 0.0012 | 0.0043 | 0.0060 | 0.0070 | yes | yes | yes |
| NFL_receptions | 0.0030 | -0.0031 | -0.0028 | 0.0426 | 0.0537 | 0.0529 | yes | yes | yes | -0.0031 | -0.0088 | 0.0132 | 0.0176 | 0.0133 | yes | yes | yes |
| NFL_rushing-tds | - | - | - | 0.0032 | -0.0013 | 0.0002 | yes | yes | yes | - | - | -0.0008 | 0.0005 | 0.0081 | yes | yes | yes |
| NFL_rushing-yards | 0.0028 | 0.0027 | 0.0027 | 0.0298 | 0.0276 | 0.0268 | yes | yes | yes | -0.0028 | -0.0028 | -0.0026 | -0.0030 | -0.0024 | yes | yes | yes |
| NFL_sacks-taken | - | - | - | 0.0696 | -0.0095 | 0.0074 | yes | yes | yes | - | - | 0.0703 | -0.0077 | 0.0709 | yes | yes | yes |
| NFL_targets | - | - | - | -0.0000 | 0.0595 | 0.0307 | yes | yes | yes | - | - | -0.0000 | 0.0049 | 0.0685 | yes | yes | yes |
| NFL_tds | -0.0227 | -0.0227 | -0.0227 | 0.0059 | 0.0059 | 0.0059 | yes | yes | yes | -0.0227 | -0.0227 | 0.0057 | 0.0057 | 0.0057 | yes | yes | yes |
| NFL_yards | - | - | - | 0.0146 | 0.0146 | 0.0146 | yes | yes | yes | - | - | 0.0091 | 0.0091 | 0.0091 | yes | yes | yes |
| NHL_assists | -0.0013 | -0.0014 | -0.0014 | 0.0009 | 0.0009 | 0.0010 | yes | yes | yes | -0.0013 | -0.0014 | 0.0009 | 0.0009 | 0.0009 | yes | yes | yes |
| NHL_blocked | 0.0002 | 0.0002 | 0.0002 | 0.0094 | 0.0003 | -0.0004 | yes | yes | yes | -0.0002 | -0.0001 | 0.0097 | -0.0006 | 0.0091 | yes | yes | yes |
| NHL_goalie-fantasy-points-underdog | -0.0030 | -0.0030 | -0.0030 | 0.0383 | 0.0383 | 0.0383 | NO | NO | NO | -0.0036 | -0.0036 | 0.0230 | 0.0230 | 0.0230 | NO | NO | NO |
| NHL_goals | -0.0071 | -0.0071 | -0.0071 | 0.0097 | 0.0097 | 0.0097 | yes | yes | yes | -0.0072 | -0.0072 | 0.0088 | 0.0088 | 0.0088 | yes | yes | yes |
| NHL_hits | -0.0108 | -0.0150 | -0.0147 | 0.0127 | 0.0127 | 0.0163 | yes | yes | yes | -0.0108 | -0.0141 | 0.0127 | -0.0036 | 0.0096 | yes | yes | yes |
| NHL_points | 0.0003 | 0.0002 | 0.0002 | 0.0030 | 0.0034 | 0.0041 | yes | yes | yes | 0.0003 | 0.0002 | 0.0030 | 0.0034 | 0.0030 | yes | yes | yes |
| NHL_powerPlayPoints | -0.0007 | -0.0007 | -0.0007 | -0.0018 | -0.0018 | -0.0018 | yes | yes | yes | -0.0008 | -0.0008 | -0.0016 | -0.0016 | -0.0016 | yes | yes | yes |
| NHL_shots | -0.0046 | -0.0046 | -0.0046 | -0.0029 | -0.0031 | -0.0020 | yes | yes | yes | -0.0044 | -0.0046 | 0.0037 | 0.0030 | -0.0003 | yes | yes | yes |
| NHL_shotsAgainst | - | - | - | 0.0056 | -0.0058 | -0.0093 | yes | yes | yes | - | - | 0.0056 | -0.0058 | 0.0056 | yes | yes | yes |
| NHL_skater-fantasy-points-underdog | -0.0066 | -0.0111 | -0.0135 | 0.0070 | 0.0058 | 0.0082 | yes | yes | yes | -0.0074 | -0.0116 | 0.0101 | 0.0084 | 0.0096 | yes | yes | yes |
| NHL_sogBS | - | - | - | 0.0532 | 0.0180 | 0.0286 | yes | yes | yes | - | - | 0.0532 | 0.0183 | 0.0534 | yes | yes | yes |
| NHL_timeOnIce | - | - | - | 0.0119 | 0.0119 | 0.0119 | yes | yes | yes | - | - | 0.0119 | 0.0119 | 0.0119 | yes | yes | yes |
| WNBA_AST | 0.0023 | 0.0026 | 0.0023 | 0.0032 | 0.0252 | 0.0254 | yes | yes | yes | 0.0023 | 0.0026 | 0.0032 | -0.0025 | 0.0161 | yes | yes | yes |
| WNBA_BLK | - | - | - | 0.0319 | 0.0113 | 0.0093 | yes | yes | yes | - | - | 0.0350 | 0.0118 | 0.0319 | yes | yes | yes |
| WNBA_BLST | - | - | - | 0.0606 | 0.0058 | 0.0252 | yes | yes | yes | - | - | 0.0499 | 0.0047 | 0.0663 | yes | yes | yes |
| WNBA_DREB | - | - | - | 0.0451 | 0.0004 | -0.0007 | yes | yes | yes | - | - | 0.0451 | 0.0004 | 0.0451 | yes | yes | yes |
| WNBA_FG3M | -0.0263 | -0.0216 | -0.0214 | 0.0109 | 0.0056 | 0.0009 | yes | yes | yes | -0.0269 | -0.0210 | 0.0103 | 0.0065 | 0.0113 | yes | yes | yes |
| WNBA_FGA | - | - | - | 0.0287 | 0.0061 | 0.0077 | yes | yes | yes | - | - | 0.0299 | -0.0007 | 0.0313 | yes | yes | yes |
| WNBA_FTM | - | - | - | 0.0014 | 0.0102 | 0.0157 | yes | yes | yes | - | - | -0.0024 | 0.0187 | -0.0080 | yes | yes | yes |
| WNBA_MIN | - | - | - | 0.0114 | 0.0114 | 0.0114 | yes | yes | yes | - | - | 0.0113 | 0.0114 | 0.0114 | yes | yes | yes |
| WNBA_OREB | - | - | - | 0.0398 | -0.0052 | -0.0020 | yes | yes | yes | - | - | 0.0372 | -0.0052 | 0.0410 | yes | yes | yes |
| WNBA_PA | 0.0030 | 0.0032 | 0.0033 | 0.0081 | 0.0085 | 0.0125 | yes | yes | yes | 0.0025 | 0.0030 | 0.0099 | 0.0098 | 0.0090 | yes | yes | yes |
| WNBA_PR | 0.0029 | 0.0033 | 0.0034 | 0.0180 | 0.0171 | 0.0159 | yes | yes | yes | 0.0009 | 0.0015 | -0.0011 | -0.0005 | -0.0010 | yes | yes | yes |
| WNBA_RA | 0.0086 | 0.0080 | 0.0080 | 0.0138 | 0.0112 | 0.0112 | NO | NO | NO | 0.0086 | 0.0087 | 0.0138 | 0.0068 | 0.0086 | NO | NO | NO |
| WNBA_STL | - | - | - | 0.0413 | -0.0041 | 0.0041 | yes | yes | yes | - | - | 0.0217 | 0.0000 | 0.0572 | yes | yes | yes |
| WNBA_TOV | - | - | - | 0.0203 | 0.0333 | 0.0361 | yes | yes | yes | - | - | 0.0204 | 0.0126 | 0.1005 | yes | NO | yes |

### Table A4. Kelly shrinkage, cells that move by more than 0.001 (measured: `m2_refit.py`, `m4_summary.py`)

Kelly shrinkage is the model's Brier skill against the book on priced validation rows, clipped to 0..1; staking multiplies by it.

| cell | priced validation rows | of which ties | Kelly shrinkage today | half | shift | drop |
|---|---|---|---|---|---|---|
| NBA_FG3M | 1745 | 218 | 0.1768 | 0.1153 | -0.0614 | 0.1115 |
| NBA_STL | 789 | 36 | 0.1141 | 0.0959 | -0.0183 | 0.0889 |
| WNBA_FG3M | 687 | 18 | 0.1576 | 0.1420 | -0.0156 | 0.1381 |
| NFL_passing-tds | 367 | 20 | 0.2190 | 0.2040 | -0.0150 | 0.1985 |
| NFL_interceptions | 382 | 11 | 0.2529 | 0.2390 | -0.0140 | 0.2300 |
| NBA_BLST | 360 | 6 | 0.0369 | 0.0305 | -0.0064 | 0.0263 |
| NBA_BLK | 1280 | 13 | 0.1517 | 0.1476 | -0.0041 | 0.1448 |
| NFL_completions | 385 | 14 | 0.0044 | 0.0014 | -0.0031 | 0.0010 |
| NBA_RA | 1605 | 67 | 0.0103 | 0.0090 | -0.0013 | 0.0082 |
| MLB_hits | 10299 | 62 | 0.0002 | 0.0013 | +0.0011 | 0.0012 |
| MLB_runs-allowed | 997 | 4 | 0.0000 | 0.0014 | +0.0014 | 0.0005 |
| NBA_PRA | 1935 | 33 | 0.0044 | 0.0060 | +0.0016 | 0.0058 |
| NBA_fantasy-points-prizepicks | 334 | 1 | 0.1629 | 0.1646 | +0.0017 | 0.1641 |
| NBA_PA | 1554 | 47 | 0.0020 | 0.0049 | +0.0028 | 0.0048 |
| NBA_REB | 1900 | 80 | 0.0103 | 0.0134 | +0.0031 | 0.0123 |
| NHL_goalie-fantasy-points-underdog | 179 | 1 | 0.0543 | 0.0577 | +0.0033 | 0.0573 |
| NFL_carries | 979 | 39 | 0.0401 | 0.0514 | +0.0113 | 0.0504 |
| WNBA_AST | 665 | 9 | 0.0037 | 0.0161 | +0.0124 | 0.0148 |
| NBA_AST | 1609 | 101 | 0.0000 | 0.0127 | +0.0127 | 0.0100 |
| NHL_skater-fantasy-points-underdog | 219 | 5 | 0.0887 | 0.1077 | +0.0190 | 0.1081 |
| NHL_hits | 1396 | 34 | 0.0584 | 0.0776 | +0.0192 | 0.0740 |
| NFL_receptions | 2450 | 153 | 0.0300 | 0.0559 | +0.0259 | 0.0523 |

### Table A5. Held-out rows at half-point lines, the 26 cells with at least 5% ties (measured: `m6_halfpoint_and_null.py`)

A half-point line cannot tie, so every label rule grades these rows the same way; only the served probability differs.

| cell | held-out tie % | half-point rows | Over hit rate | read minus hit, today pp | read minus hit, half pp | Brier today | Brier half | Brier shift |
|---|---|---|---|---|---|---|---|---|
| WNBA_TOV | 18.8 | 905 | 53.9 | +3.21 | -5.78 | 0.2385 | 0.2408 | +0.0023 |
| NFL_qb-tds | 18.6 | 155 | 59.4 | +5.69 | -5.61 | 0.2375 | 0.2368 | -0.0007 |
| NFL_sacks-taken | 17.5 | 147 | 55.8 | -4.92 | -4.96 | 0.2072 | 0.2075 | +0.0003 |
| WNBA_BLST | 17.0 | 1098 | 46.8 | -0.35 | -2.08 | 0.2324 | 0.2299 | -0.0025 |
| NFL_targets | 15.3 | 945 | 52.0 | +3.42 | -3.06 | 0.2062 | 0.2033 | -0.0029 |
| WNBA_DREB | 14.6 | 875 | 51.9 | -2.23 | -2.23 | 0.2293 | 0.2293 | +0.0000 |
| NBA_OREB | 13.9 | 1171 | 42.5 | +1.46 | -0.85 | 0.2351 | 0.2345 | -0.0006 |
| NBA_DREB | 13.5 | 859 | 51.1 | +3.46 | -2.55 | 0.2135 | 0.2123 | -0.0012 |
| WNBA_STL | 13.1 | 1379 | 44.6 | +1.57 | -1.10 | 0.2418 | 0.2419 | +0.0000 |
| NBA_BLST | 12.2 | 1278 | 48.2 | -0.43 | -0.86 | 0.2367 | 0.2367 | -0.0000 |
| NBA_FGM | 12.2 | 751 | 48.7 | -0.81 | -0.81 | 0.2328 | 0.2328 | +0.0000 |
| NBA_FG3A | 11.9 | 916 | 46.4 | +1.30 | +1.25 | 0.2271 | 0.2272 | +0.0001 |
| NBA_FG3M | 11.8 | 1314 | 51.2 | +0.48 | +0.30 | 0.2355 | 0.2353 | -0.0003 |
| WNBA_OREB | 11.5 | 1372 | 37.7 | +3.75 | +0.77 | 0.2248 | 0.2228 | -0.0020 |
| WNBA_FGA | 8.4 | 931 | 52.8 | -1.00 | -1.11 | 0.2285 | 0.2282 | -0.0003 |
| WNBA_AST | 7.6 | 1349 | 51.0 | +3.82 | -0.54 | 0.2428 | 0.2414 | -0.0015 |
| NFL_passing-tds | 7.4 | 352 | 49.4 | -0.14 | -0.27 | 0.2502 | 0.2507 | +0.0005 |
| WNBA_FTM | 7.2 | 1335 | 40.5 | +5.22 | +4.36 | 0.2366 | 0.2361 | -0.0005 |
| NHL_sogBS | 7.2 | 1076 | 62.0 | -4.51 | -4.77 | 0.2131 | 0.2127 | -0.0004 |
| NFL_receptions | 6.6 | 2029 | 48.1 | +1.98 | +1.98 | 0.2487 | 0.2487 | -0.0000 |
| NFL_passing-first-downs | 6.6 | 150 | 48.7 | +7.89 | +7.89 | 0.2037 | 0.2037 | +0.0000 |
| WNBA_BLK | 5.9 | 1878 | 24.4 | +4.41 | +2.63 | 0.1738 | 0.1723 | -0.0015 |
| NBA_AST | 5.3 | 1466 | 51.8 | +0.54 | +0.39 | 0.2417 | 0.2418 | +0.0001 |
| NBA_REB | 5.1 | 1460 | 52.4 | -1.45 | -1.52 | 0.2462 | 0.2459 | -0.0003 |
| NBA_FTM | 5.0 | 1479 | 43.5 | +1.41 | -1.50 | 0.2385 | 0.2386 | +0.0001 |
| NHL_shotsAgainst | 5.0 | 317 | 55.5 | -0.35 | -0.35 | 0.2203 | 0.2203 | +0.0000 |
