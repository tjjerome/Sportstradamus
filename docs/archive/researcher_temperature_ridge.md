# In-repo research brief: should the `0.01·(T − 1)²` penalty in the Brier temperature fit be removed, or its weight set by cross-validation? — 2026-10-04

## TL;DR

- **Verdict: GO. Remove the penalty (weight 0). Do not build a cross-validated weight.** Both of the owner's conditions hold on criteria written down before the results. (a) Held-out log-loss and Brier at the sportsbook line improve (Holm-adjusted p = .003) and the recommended-leg gap falls. (b) All 73 served cells that pass the six offline gates today still pass, and the 5 that fail today still fail: no pass-to-fail flip, no fail-to-pass flip, no new near-miss.
- **How much better.** At the sportsbook line (180,038 held-out rows, 48 cells, 409 game days): log-loss −0.00039 [−0.00057, −0.00023], Brier −0.00018 [−0.00025, −0.00011]. It is an NFL effect (log-loss −0.0049 [−0.0068, −0.0031]) with smaller NBA and WNBA ones; MLB and NHL do not move. On the tail scorecard the recommended-leg gap goes 12.9 → 10.1 pp at the 1.78 Underdog baseline in the code (Δ −2.8 pp [−7.0, +0.2], p = .09) and 13.1 → 9.7 pp at the 1.83 baseline of locked decision 8 (Δ −3.3 pp [−6.2, −1.3], p = .003). Recommended volume falls 28 % (24 % at 1.83), return per pick goes −10.3 % → −6.2 %, units lost go −118 → −51.
- **The penalty is not a small-sample guard.** It is a fixed weight on a per-row mean loss, so it never fades as rows accumulate: it pulls NHL goals to the same T at 100 validation rows and at 17,560. It costs most in the smallest cells (held-out log-loss −0.014 without it under 500 validation rows, −0.007 at 500–1,500, −0.0015 at 1,500–3,500, nothing above). It would earn its keep only below roughly 100–200 validation rows; the smallest served cell has 355. Cross-validation itself picks a weight of 0.003 or less in 44 of the 56 cells where the weight changes anything.
- **The cost, stated plainly: alternate-line probabilities get worse in four NFL cells.** Log-loss over all 40,568 settled DFS rungs rises +0.0059 [+0.0014, +0.0092] (p = .002), about four to six times what the earlier brief measured on live legs. All of it is the alternate rungs of NFL receptions, rushing yards, interceptions and passing TDs, where the favoured side's read falls well below its hit rate (receptions: .744 → .616 against .771). No gate can see this: the two gates that read the probability read it at the main line. The same four cells are where the money is saved. Their recommended legs lose about 20 % per pick today, and removing the penalty cuts those legs from 398 to 116.
- **Mechanics.** One line in `_brier_temperature_loss`. T is stored in each model file, so the change reaches a cell at its next retrain. Condition (b) was verified without retraining, by replaying the calibration chain from each stored booster (the replay reproduces the stored `P` to 1e-9 in 65 of 78 cells); the same replay attributes any flip at the retrain. This is not the cure: without the penalty the recommended legs still read 10 pp above their hit rate and still lose about 6 % per pick.

## 1. Decision question and the statistic that decides it

**Decision.** Honest-receipts locked decision 14: "The temperature penalty is dropped only if the models are better without it and every cell that passes the gates today still passes." The owner's rule in his words: "Drop the temperature if you think we're better without and the models still pass the gates." Read as two conditions:

- (a) the models are measurably better without the penalty, on criteria fixed before looking;
- (b) every (league, market) cell that passes the six offline ship gates today still passes them with the unpenalised temperature.

GO needs both. A cross-validated weight is the allowed middle answer if it meets both. Anything else is KILL: keep the penalty.

**What is being decided.** The temperature T divides the over-probability's log-odds: `p' = expit(logit(p) / T)`. Its reciprocal 1/T is the calibration slope of Cox (1958, doi:10.1093/biomet/45.3-4.562). T is bounded to [1, 10], so it can only flatten a probability toward .5, never sharpen it. `_brier_temperature_loss` (`src/sportstradamus/training/pipeline.py:2845–2850` in the working tree, 2841–2846 at HEAD; an unrelated in-flight edit added four lines above it) adds `0.01 * (T - 1) ** 2` to the mean Brier, and `_step_calibrate_temperature` (`pipeline.py:3198`, the `minimize_scalar` call at 3245–3249) minimises the sum. The penalty entered in commit `d43b65c9` (2026-03-29) with one line of rationale: "L2 penalty toward T=1 (no correction)".

**Statistics that decide it.** For (a): the paired difference, candidate minus penalised, in held-out log-loss and Brier on sportsbook-priced rows, and in the recommended-leg gap on the tail scorecard, each with a day-clustered bootstrap interval. For (b): `scorecard.compute_gates` on each served cell's stored test set with `P` replaced by the candidate's probability.

**Terms used throughout.**

| Term | Meaning |
|---|---|
| Penalised | As shipped: weight 0.01. Reproduces every stored T |
| Unpenalised | Weight 0 |
| Cross-validated | Weight chosen per cell from {0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1} by 5-fold player-disjoint out-of-fold Brier, ties to the larger weight |
| One-SE | Sensitivity arm: the largest weight whose out-of-fold Brier is within one fold standard error of the best |
| Temperature-last cell | No probability-stage post-hoc follows T (56 of 78 cells). A change in T reaches the served probability in full |
| Post-hoc cell | A Platt or isotonic recalibration of the probability (`prob_recal_platt`, `prob_recal_isotonic`) is fit after T (22 of 78) and absorbs most of a change in T |
| Moved cell | Unpenalised T differs from penalised T by more than 0.001: 56 of 78 (36 temperature-last, 20 post-hoc). The other 22 fit T = 1 either way |
| Gate population | Held-out test rows with a real sportsbook price (`QuoteAuthenticity == authentic`): the rows gate 1 scores |
| Tail scorecard | `sportstradamus admin tail-scorecard`: every held-out test row re-served at each Underdog and Sleeper rung the archive held, then passed through the live recommendation rule |
| Recommended leg | A scorecard leg the live rule recommends (edge of at least 5 % at the platform payout) |
| Gap | Mean of (read − hit) over recommended legs, in percentage points. Positive means the model overstates |
| Main rung, alternate rung | A DFS rung whose line equals, or differs from, the held-out row's sportsbook line |

## 2. Pre-registration

Written to `PREREG.md` in the sandbox at 22:47 CDT (file timestamp). At that point the replay had reproduced the penalised chain only: no unpenalised T, no gate under another T and no tail-scorecard arm had been computed in this session. The earlier brief's figures (R3, Finding 11) were known.

- **Arms.** As in the terms table. Every arm keeps the `(1.0, 10.0)` bounds and the all-validation-rows fit population. Held fixed in every arm: the booster, the blend weight, the mean corrector, the dispersion and skew scalars and the PIT map. Refit per arm: T, then the probability post-hoc where the cell has one.
- **Primary endpoints**, each a paired difference against the penalised arm:
  - P1a: log-loss on the gate population, pooled over cells, row-weighted.
  - P1b: Brier, same pooling.
  - P2: the recommended-leg gap on the tail scorecard, both cohorts on the same day weights.
- **Inference.** Day-clustered bootstrap, 4,000 resamples, seed 0; one calendar date is one cluster across all cells. Holm (1979) over 3 endpoints × 2 arms.
- **"Measurably better"**, verbatim: "at least one primary endpoint improves with a Holm-adjusted p < .05 (Holm over the 3 endpoints × 2 arms FREE and CV) and no primary endpoint worsens with an unadjusted 95% CI excluding 0. Both rules lean toward keeping the status quo." (FREE is the unpenalised arm, CV the cross-validated one.)
- **Secondary.** Recommended volume, return per pick at platform payouts, total overstatement Σ(read − hit), bulk log-loss and bulk gap over all rung rows, and the per-cell change in validation Brier skill and `kelly_shrinkage`.
- **Condition (b).** `compute_gates` on each served cell's stored test-set frame with `P` replaced. It fails if any cell that ships under the penalised arm does not ship under the candidate. Near miss: gate 1 CI upper bound above 0.004, or gate 5 above 0.060, under the candidate where the penalised arm was not.
- **Verdict rule**, verbatim: "GO iff FREE is measurably better and has no pass→fail flip. CV-weight iff FREE fails either condition and CV satisfies both. Otherwise KILL (keep the penalty)."

Not pre-registered, so exploratory wherever they appear: the rerun at the 1.83 Underdog baseline, the main-rung against alternate-rung split, the cell-size bins, the gate pass windows and the supersession yardstick.

One reading is stricter than the rule above: every primary endpoint individually significant. On that reading P2 falls short at the 1.78 baseline (p = .09) and clears at 1.83 (p = .003). The reader should weigh that; the verdict below follows the rule as written.

## 3. Data actually used

- **Cells.** All 78 (league, market) cells with a model file in `data/models/`, a stored test set and a cached training matrix whose sha256 equals the model file's `matrix_hash`: MLB 15, NBA 18, NFL 19, NHL 12, WNBA 14. Thirteen further test sets have no model file and are out of scope.
- **Rows.** Validation rows per cell run from 355 (NFL passing first downs) to 17,560 (NHL goals): 8 cells under 500, 9 at 500–1,500, 46 at 1,500–3,500, 15 above. Held-out test rows: 276,706 in 78 cells over 432 game days. Gate population: 180,038 rows in 48 cells over 409 days. The other 30 cells have no usable sportsbook-priced test rows; gate 1 is blank for them and passes automatically.
- **Tail scorecard.** 32 cells have rungs (MLB 14, NFL 13, WNBA 5): 40,568 settled rung rows from 2026-08-30 to 2026-09-28, 23 game days (MLB 17, NFL 10; WNBA contributes 30 rungs). 12,148 are main rungs and 28,420 alternate. The penalised arm reproduces the first recorded CLI run exactly: 40,568 rungs, 1,149 recommended, read .653, hit .524. NBA and NHL test sets predate the columns the scorecard needs, so those leagues have no tail read.
- **Code.** Production functions, unmodified: `pipeline._step_calibrate_and_serve` and its steps, `posthoc.fit_posthoc`, `scorecard.compute_gates` and `supersede_verdict`, `tail_pricing.price_cell` and `serve`, `tail_scorecard.replay_live_rule`, `realized.settled_offers`. The archive was opened read-only. Nothing under `src/`, `tests/`, `data/models/`, `model_stats.parquet` or `stat_meta.json` was written. No model was retrained.
- **The 1.83 rerun.** `UNDERDOG_BOOST_BASELINE` is 1.78 in the code. The sensitivity run patches it to 1.83 in process. A first attempt patched nothing (its log shows `1.83 1.78` and results identical to 1.78) and was discarded; the numbers below come from the corrected run, whose log shows `1.83 1.83` and whose recommended count changes from 1,149 to 1,447.

## 4. Key findings

### Finding 1: the calibration chain can be replayed offline, and the replay is a faithful test of this change

The order of steps in `train_market`, from `_step_calibrate_and_serve`:

| # | Step | Fit on | Reads T? | Where the result lives |
|---|---|---|---|---|
| 0 | Booster (LightGBMLSS) | First 70 % of the matrix by date | No | Model file |
| 1 | Blend weight (`_step_fuse_predictions`) | Sportsbook-priced validation rows; fewer than 10 clusters gives weight 1.0 | No | Model file |
| 2 | Mean corrector (`_step_correct_fused_mean`: `roe_mean`, `isotonic_mean`) | Validation rows | No | Model file (post-hoc blob) |
| 3 | Dispersion scalar, skew, PIT map (`_step_calibrate_dispersion`) | Validation rows | No | Model file |
| 4 | **Temperature** (`_step_calibrate_temperature`) | **All validation rows**: priced rows fused at the blend weight, unpriced rows model-only | — | Model file |
| 5 | Probability post-hoc (`prob_recal_platt`, `prob_recal_isotonic`) | Validation rows after T; the book leg on priced rows only | Yes | Model file (post-hoc blob) |
| 6 | Validation skill: `brier_skill_score`, `kelly_shrinkage` | Priced validation rows after steps 4–5 | Yes | `model_stats.parquet` |
| 7 | Steps 1–5 applied to the held-out test half | Nothing is fit | Yes | Test-set CSV |

The last 30 % of the matrix by date is split about evenly into validation and test by a hash of (Player, Date). The two halves cover the same dates, so "held-out" below means out of sample, not out of time.

- **Nothing fit before T reads T.** The only things fit after it are the probability post-hoc and the validation skill numbers. Refitting steps 4–6 from the stored booster is therefore the whole effect of this change on a given model file. A retrain would add a new booster and confound the comparison.
- **What is stored.** The model file holds the booster, the blend weight, the post-hoc blob, dispersion, skew, PIT map, T, `matrix_hash`, `trained_at` and the controls. The test-set CSV holds the held-out rows with `P` (final: after fusion, T and the probability post-hoc), `P_standalone` (model-only, before the blend), `P_PrePool` (before the book pool; structural strategies only), `QuoteAuthenticity`, `Odds` (the book's under-probability, so `p_book = 1 − Odds`), `Book_EV`, `Line` and `Result`. Validation rows and their predictions are stored nowhere.
- **Why the replay is exact anyway.** The split is a deterministic function of the matrix, every model file's `matrix_hash` equals the sha256 of the cached matrix on disk, and the booster is in the file. Rebuilding the split, predicting with the stored booster and running the pipeline's own chain reproduces:
  - the stored T within 0.0004 in 76 of 78 cells (NFL yards differs by 0.0013, NBA BLST by 0.012);
  - the stored test-set `P` to 1e-9 in 65 cells and to 1e-3 in 6 more. Seven drift further, two of them materially: NBA FG3M (mean absolute difference .027; the blend weight replays as .766 against .809 stored) and NBA BLST (.004);
  - the gates: recomputed from the stored CSVs, they equal `model_stats.parquet` in all 78 cells.
- **Two things the replay needs.** First, 59 of the 78 model files predate commit `0c1cf950` (2026-09-28), which changed how `set_model_start_values` seeds rows with no history; the replay rebuilds the old seeding in process for those files. Second, the cached matrix's index does not match the CSV's row order, so rows are aligned on (Player, Date). Every CSV row matched in every cell, with unique keys and identical `Line`, `Result`, `Odds` and authenticity.
- **How the inexact cells are handled.** Every arm is compared replay to replay, on the same upstream constants; the "today" columns show the stored values. For the two material cells a cross-check rescales the stored `P` directly, `expit(T_stored · logit(P) / T_new)`. NBA BLST still passes (gate 1 .0015 unpenalised, .0016 cross-validated; gate 5 .027). NBA FG3M has T = 1 in every arm, so nothing changes.
- **Cost.** About 15 minutes of wall time for the 78-cell replay on 4–7 workers, and one minute for all arms and gates.
- **The sandbox-retrain route was not needed.** Retraining a subset with `--frozen-matrix-dir` and `--artifact-output` would be the weaker test here: it draws a new booster and new hyperparameters, so a gate that moved could not be attributed to the temperature.

This answers the question left open in R3 (open question 5): T and the stages after it can be refit without a full `meditate` retrain, for as long as the cached matrix is the one the model file was trained on.

### Finding 2: the penalty is a fixed bias toward T = 1, not a small-sample regulariser

- **Mechanism.** The loss is a per-row mean and the weight is not divided by the number of rows. The penalised fit therefore stops where the Brier's slope in T equals −0.02·(T − 1), at every sample size. For NFL receptions the penalty at the unpenalised optimum (T = 3.44) would be 0.0595, against a total Brier gain of 0.0083 available from flattening (.2566 at T = 1, .2483 at T = 3.44). The penalty, not the data, sets T.
- **It does not fade with sample size.** Subsampling the validation rows, the penalised T for NFL receptions averages 1.37–1.38 at every size from 100 rows to 2,970, and for NHL goals 1.18–1.20 from 100 to 17,560. By analogy with a likelihood: a fixed weight λ on a mean log-likelihood acts like a Gaussian prior on T with standard deviation 1/√(2λn), which is 0.36 at 385 rows, 0.13 at 2,970 and 0.05 at 17,560. A prior that tightens as evidence accumulates is backwards.
- **How far it pulls.**

  | League | Served cells | T = 1 in both arms | T moves | Unpenalised T above 1.5× penalised | Post-hoc cells | Validation rows, median (range) |
  |---|---:|---:|---:|---:|---:|---|
  | MLB | 15 | 8 | 7 | 1 | 3 | 9,748 (548–10,300) |
  | NBA | 18 | 4 | 14 | 2 | 7 | 2,190 (1,892–2,263) |
  | NFL | 19 | 4 | 15 | 8 | 6 | 1,283 (355–3,274) |
  | NHL | 12 | 5 | 7 | 0 | 2 | 2,686 (726–17,560) |
  | WNBA | 14 | 1 | 13 | 2 | 4 | 2,165 (2,027–2,257) |
  | All | 78 | 22 | 56 | 13 | 22 | |

  The 13 cells, penalised → unpenalised: NFL carries 1.51 → 6.65, NFL completions 1.33 → 4.21, NFL rushing yards 1.42 → 4.05, NFL receiving yards 1.27 → 3.81, NFL receptions 1.38 → 3.44, NFL attempts 1.30 → 3.31, NBA PRA 1.34 → 2.66, MLB runs allowed 1.27 → 2.55, NFL passing TDs 1.27 → 2.38, WNBA FTM 1.38 → 2.12, NFL interceptions 1.27 → 2.08, NBA FTM 1.34 → 2.06, WNBA PR 1.28 → 2.04. Seven of them are post-hoc cells (carries, completions, receiving yards, attempts, PRA, runs allowed, NBA FTM), where the refit Platt or isotonic map absorbs the change (Finding 4).
- **T never falls.** Unpenalised T is at least the penalised T in all 78 cells, so in a temperature-last cell the change can lower a favoured side's read and never raise it.
- **Agreement with R3.** R3 Finding 11 refit T on the test half and reported 5.03 (receptions), 6.79 (rushing yards), 4.39 (carries), 4.11 (receiving yards), 3.57 (attempts), 10 (interceptions, passing TDs) and 4.85 (runs allowed). The test-half optimum in this replay is 5.02, 6.61, 4.49, 4.10, 3.58, 10, 10 and 4.74. The T a retrain would store is the validation-half fit in Appendix A, which is lower (receptions 3.44, rushing yards 4.05).
- **Side note, out of scope.** 21 cells would fit T below 1, sharpening, if the lower bound allowed it (NFL targets .67, MLB pitcher fantasy points .72, NFL passing first downs .81, NBA FG3M .83). Every arm keeps the bound at 1.

### Finding 3 (condition b): no cell changes its gate verdict

- **Only gates 1 and 5 read `P`.** Gate 1 is the paired Brier difference against the book on sportsbook-priced rows, whose 95 % CI upper bound must be below 0.005. Gate 5 is the debiased equal-mass ECE (Roelofs et al. 2022, arXiv:2012.08668; [48] in `operation_ship_references.md`), which must be below 0.075. Gates 2, 3, 4 and 6 read the predictive mean and CDF, which the temperature does not touch. Their statistics are identical across arms in all 78 cells (largest absolute difference: 0).
- **Result.**

  | Arm | Pass all six | Fail | Pass → fail | Fail → pass |
  |---|---:|---:|---|---|
  | Today (stored test sets, equal to `model_stats.parquet`) | 73 | 5 | — | — |
  | Unpenalised | 73 | 5 | none | none |
  | Cross-validated | 73 | 5 | none | none |
  | One-SE (sensitivity) | 72 | 6 | NBA BLST (gate 1 .0068) | none |

- **The five failing cells fail for reasons the temperature does not reach.** MLB pitcher fantasy points (gate 4), MLB runs allowed (gate 4: .0501 against .0500), NHL goalie fantasy points (gate 4), NBA PR (gate 1: .0056 today, .0053 unpenalised, still above .005) and WNBA RA (gate 1: .0086; a post-hoc cell, unchanged).
- **Near-misses and thin margins.** No new near-miss appears. Removing the penalty clears two gate 5 near-misses and widens most thin gate 1 margins.

  | Cell | Gate | Today | Unpenalised | Cross-validated | Reading |
  |---|---|---:|---:|---:|---|
  | NFL sacks taken | 5 (< .075) | .0696 | .0703 | .0703 | No change: T = 1 in every arm, and the 0.0007 is replay drift (replayed penalised: .0703). The thinnest margin in the fleet |
  | NFL interceptions | 5 | .0638 | .0410 | .0449 | Near-miss cleared |
  | WNBA BLST | 5 | .0606 | .0499 | .0501 | Near-miss cleared |
  | NFL receptions | 5 | .0426 | .0132 | .0134 | Wider |
  | NFL completions | 1 (< .005) | .0041 | .0041 | .0041 | Post-hoc cell; unchanged |
  | NBA REB | 1 | .0040 | .0029 | .0032 | Wider |
  | NBA PTS | 1 | .0039 | .0034 | .0035 | Wider |
  | NBA PA | 1 | .0034 | .0034 | .0034 | Post-hoc cell; unchanged |
  | NBA BLST | 1 | .0032 | .0016 | .0017 | Wider (replayed penalised: .0036) |
  | NFL receptions | 1 | .0030 | −.0031 | −.0026 | Wider |
  | WNBA PA | 1 | .0030 | .0025 | .0026 | Wider |
  | WNBA PR | 1 | .0029 | .0009 | .0010 | Wider |
  | NFL rushing yards | 1 | .0028 | −.0028 | −.0022 | Wider |
  | NFL attempts | 1 | .0024 | .0025 | .0024 | Post-hoc cell; 0.0001 worse |
  | NBA AST | 1 | .0010 | .0013 | .0011 | 0.0003 worse |
  | NHL shots | 5 | −.0029 | .0037 | .0000 | Worse, far from .075 |
  | NFL QB yards | 5 | .0159 | .0202 | .0193 | Worse, far from .075 |
  | WNBA BLK | 5 | .0319 | .0350 | .0327 | Worse, far from .075 |

- **Robustness to noise in T (exploratory).** T is itself an estimate. Each of the 54 passing temperature-last cells had its gates recomputed along the bootstrap range of its unpenalised T and across a grid of T from 1 to 10.
  - At the 5th percentile of the unpenalised T no cell fails. At the 95th one does: NFL sacks taken (gate 5 .0766 at T = 1.235).
  - **NFL sacks taken is the one cell where the penalty protects a gate.** It passes only while T stays at or below 1.17, because gate 5 rises with T and the cell has no book. Both arms fit T = 1.0 today. Across resamples of its 398 validation rows, the chance of fitting T above 1.17 is 3.2 % penalised and 11.5 % unpenalised.
  - Five cells fail at T = 1 and pass only because the temperature flattens them: NFL receptions (needs T ≥ 1.29; the penalised 1.38 sits 0.09 above that edge), NFL rushing yards (≥ 1.23), NBA BLST (≥ 1.11), NFL interceptions (≥ 1.05) and WNBA PR (≥ 1.05). For these, the penalty is what makes the margin thin.
  - Summed over cells, the expected number of noise-driven failures per full retrain is 0.07 penalised and 0.13 unpenalised. Nearly all of the unpenalised figure is NFL sacks taken.
- **The full 78-cell table is Appendix A.**

### Finding 4 (condition a, at the sportsbook line): log-loss and Brier improve, almost entirely in NFL

Paired difference against the penalised arm, held-out rows, day-clustered bootstrap. Negative is better.

| Scope | Cells | Rows | Days | Δ log-loss, unpenalised | Δ Brier, unpenalised | Δ log-loss, cross-validated |
|---|---:|---:|---:|---|---|---|
| **Gate population (P1a, P1b)** | 48 | 180,038 | 409 | **−0.00039 [−0.00057, −0.00023]** | **−0.00018 [−0.00025, −0.00011]** | −0.00035 [−0.00050, −0.00021] |
| NFL | 9 | 11,983 | 100 | −0.00492 [−0.00678, −0.00313] | −0.00205 [−0.00282, −0.00128] | −0.00426 [−0.00579, −0.00280] |
| WNBA | 5 | 3,188 | 171 | −0.00109 [−0.00201, −0.00020] | −0.00051 [−0.00096, −0.00009] | −0.00051 [−0.00132, +0.00025] |
| NBA | 13 | 17,337 | 80 | −0.00059 [−0.00091, −0.00030] | −0.00026 [−0.00040, −0.00014] | −0.00054 [−0.00079, −0.00031] |
| MLB | 12 | 96,577 | 137 | −0.00000 [−0.00010, +0.00008] | −0.00000 [−0.00003, +0.00003] | −0.00001 [−0.00010, +0.00006] |
| NHL | 9 | 50,953 | 162 | +0.00004 [−0.00015, +0.00023] | −0.00003 [−0.00009, +0.00004] | +0.00001 [−0.00016, +0.00017] |
| Temperature-last cells | 31 | 148,014 | 408 | −0.00047 [−0.00071, −0.00027] | −0.00021 [−0.00031, −0.00013] | −0.00042 [−0.00062, −0.00025] |
| Post-hoc cells | 17 | 32,024 | 407 | −0.00005 [−0.00008, −0.00002] | −0.00002 [−0.00004, −0.00001] | −0.00005 [−0.00007, −0.00002] |
| Cells replayed to 1e-9 only | 39 | 162,904 | 409 | −0.00041 [−0.00061, −0.00022] | −0.00018 [−0.00027, −0.00011] | −0.00036 [−0.00053, −0.00021] |
| All held-out rows, priced or not | 78 | 276,706 | 432 | −0.00045 [−0.00062, −0.00029] | −0.00018 [−0.00024, −0.00012] | −0.00040 [−0.00054, −0.00027] |

- **Levels on the gate population** (log-loss, Brier): book .59937, .20729; penalised .58033, .19980; unpenalised .57993, .19962; cross-validated .57998, .19964; one-SE .58111, .19997.
- **Multiplicity.** Holm over the six pre-registered tests: P1a and P1b are at an adjusted p of .003 in both arms (the bootstrap floor is .0005 unadjusted).
- **Size.** Pooled, the change is 2 % of the model's log-loss lead over the book (.0190). In NFL it is material: the lead goes from .0489 to .0538.
- **Per cell.** 36 of the 48 cells change: 24 better, 12 worse. Best: −0.0323 (NFL interceptions). Worst: +0.00072 (NBA AST).
- **Sharpness at the main line.** On all held-out rows the favoured side's mean read against its hit rate: NFL receptions .580 against .525 penalised, .533 unpenalised; NFL rushing yards .581 against .526, then .530; WNBA PR .589 against .553, then .558. WNBA FTM goes from 1.9 pp over (.613 against .594) to 1.8 pp under (.576).
- **Post-hoc cells barely move, by construction.** A Platt map refit after a larger T re-expands what the T flattened, and an isotonic map is unaffected by a monotone rescaling of its input apart from its knots. The mean absolute change in `P` is .0006 across the 20 moved post-hoc cells (largest: WNBA PA .0049) against .016 across the 36 moved temperature-last cells (largest: NFL rushing yards .050, NFL receptions .047). On all held-out rows the post-hoc cells get worse by +0.00007 [+0.00003, +0.00012]: visible to the bootstrap, nil in practice.
- **More shrinkage is the wrong direction.** The one-SE arm is worse than today: +0.00079 [+0.00062, +0.00096] on the gate population.

### Finding 5 (condition a, on the tail scorecard): the recommended-leg gap falls about 3 pp and volume about a quarter

Day-clustered bootstrap over 23 game days. "Posted payouts only" keeps the legs whose side the live history shows was actually posted; the rest are priced at an assumed payout.

| | Penalised | Unpenalised | Δ unpenalised [95 % CI], p | Cross-validated | Δ cross-validated [95 % CI], p |
|---|---:|---:|---|---:|---|
| **Underdog baseline 1.78 (the code today; pre-registered)** | | | | | |
| Recommended legs | 1,149 | 827 | −28.0 % [−31.6, −23.2] | 869 | −24.4 % [−26.9, −20.3] |
| Mean read | .653 | .640 | | .642 | |
| Hit rate | .524 | .539 | | .546 | |
| **Gap (P2)** | **+12.90 pp** | **+10.07 pp** | **−2.83 pp [−6.98, +0.20], p = .09** | +9.61 pp | −3.30 pp [−5.64, −0.27], p = .04 |
| Return per pick | −10.3 % | −6.2 % | +4.1 pp [−0.8, +11.1], p = .17 | −5.3 % | +5.0 pp [+0.1, +8.6], p = .05 |
| Return per pick, posted payouts only | −14.4 % | −6.5 % | +7.9 pp [+2.5, +13.4], p = .02 | −6.3 % | +8.1 pp [+2.7, +13.3], p = .02 |
| Units, one per leg | −118.2 | −51.2 | +67.1 [+4.9, +153.8], p = .03 | −46.4 | +71.8 [+5.8, +150.8], p = .03 |
| Total overstatement, in legs | 148.2 | 83.3 | −64.9 [−136.8, −11.0], p = .004 | 83.5 | −64.8 [−129.2, −9.3], p = .005 |
| **Underdog baseline 1.83 (locked decision 8; sensitivity)** | | | | | |
| Recommended legs | 1,447 | 1,105 | −23.6 % [−27.4, −20.6] | 1,138 | −21.4 % [−23.6, −18.9] |
| Gap | +13.06 pp | +9.74 pp | −3.33 pp [−6.22, −1.26], p = .003 | +9.58 pp | −3.48 pp [−5.53, −1.51], p = .001 |
| Return per pick | −10.9 % | −6.1 % | +4.9 pp [+1.7, +9.6], p = .004 | −5.8 % | +5.2 pp [+2.2, +8.1], p = .002 |
| Return per pick, posted payouts only | −16.9 % | −9.5 % | +7.4 pp [+3.3, +10.5], p = .009 | −9.7 % | +7.2 pp [+4.2, +9.8], p = .004 |
| Units, one per leg | −158.3 | −67.1 | +91.2 [+15.7, +188.1], p = .003 | −65.9 | +92.4 [+14.0, +183.3], p = .003 |
| Total overstatement, in legs | 189.0 | 107.6 | −81.4 [−161.1, −17.2], p < .001 | 109.0 | −80.0 [−155.5, −14.5], p < .001 |

- **Multiplicity at the pre-registered baseline.** Holm-adjusted P2: .071 cross-validated, .0875 unpenalised. P2 does not worsen in either arm, so with P1a and P1b it meets the pre-registered rule. Taken alone it is suggestive at 1.78 and significant at 1.83.
- **By league (1.78).** NFL: 904 → 642 legs (−29 %), gap 15.8 → 12.5 pp, units −136.8 → −67.8 (Δ +69.0 [+12.3, +159.3]). MLB: 240 → 182 legs (−24 %), gap 3.1 → 2.1 pp, units +15.1 → +14.7 (Δ −0.4 [−7.3, +5.8]). In MLB the change removes a quarter of the recommended legs and moves no money either way.
- **By platform (1.78).** Underdog: 626 → 504 legs, gap 13.4 → 11.8 pp. Sleeper: 523 → 323 legs, gap 12.3 → 7.5 pp.
- **What moves.** 737 legs are recommended in both arms; on those same legs the read drops from .651 to .642 against a hit rate of .543, so they still lose 5.8 % per pick. 412 legs drop out: read .656, hit .490, −18.3 % per pick, −75.4 units; 367 of them are still held by the live rule and now fall below the edge threshold. 90 legs come in, all but one in temperature-last cells: read .627, hit .511, −9.3 % per pick, −8.3 units. The legs that come in are not reads that went up: in temperature-last cells no settled leg reads higher without the penalty (0 of 13,745). They are the live rule choosing differently once reads fall: 66 are a different rung or side of a player-market the rule already held, and 24 are player-markets it did not hold before. Every recommended leg in every arm reads above .5.
- **Where the saving is.** Four NFL cells (receptions, rushing yards, interceptions, passing TDs) go from 398 recommended legs losing 81.6 units to 116 losing 21.3. That is 60 of the 67 units saved. Every other cell together goes from 751 legs losing 36.7 units to 711 losing 29.9. Per cell: NFL receptions 214 → 91 legs, NFL rushing yards 147 → 4, NFL interceptions 25 → 18, NFL passing TDs 12 → 3, MLB total bases 102 → 61.
- **Against the earlier estimate.** R3 put the gap effect at −1.0 to −1.2 pp, from an extra temperature fit walk-forward and applied on top of live legs. This brief replaces T inside the chain and scores held-out rows in the same window, and finds −2.8 to −3.3 pp. The live effect should sit between the two (Reality checks).

### Finding 6: the cost is real and no gate can see it: alternate-line probabilities get worse in four NFL cells

Log-loss over every settled rung, recommended or not. Positive is worse.

| Scope | Rungs | Penalised | Unpenalised | Δ unpenalised [95 % CI] | Δ cross-validated [95 % CI] |
|---|---:|---:|---:|---|---|
| All settled rungs | 40,568 | .5765 | .5824 | **+0.0059 [+0.0014, +0.0092], p = .002** | +0.0046 [+0.0012, +0.0070] |
| NFL main rungs | 3,037 | .7064 | .6989 | −0.0075 [−0.0122, −0.0032] | −0.0055 [−0.0094, −0.0020] |
| NFL alternate rungs | 17,495 | .6212 | .6357 | +0.0145 [+0.0089, +0.0174] | +0.0112 [+0.0077, +0.0131] |
| MLB main rungs | 9,097 | .6240 | .6238 | −0.0002 [−0.0005, +0.0001] | −0.0002 [−0.0005, +0.0001] |
| MLB alternate rungs | 10,909 | .4293 | .4302 | +0.0009 [−0.0003, +0.0023] | +0.0006 [−0.0002, +0.0017] |
| All rungs except the alternate rungs of the four NFL cells | 35,658 | | | −0.0016 [−0.0025, −0.0001] | |

- **All of the cost is four cells' alternate rungs.** NFL receptions (2,281 alternate rungs, +0.0674 each), NFL rushing yards (2,326, +0.0552), NFL interceptions (97, +0.1065) and NFL passing TDs (206, +0.0203) add +0.0073 to the pooled figure. Everything else improves by 0.0014.
- **What goes wrong there.** On receptions' alternate rungs the favoured side hits .771. The penalised read is .744 and the unpenalised read is .616. On rushing yards the hit rate is .708, against reads of .669 and .572. By distance from the projection, NFL rungs within a quarter of a standard deviation are unchanged (−0.0012), and the cost grows with distance: +0.0137 at 0.75–1.0, +0.0262 at 1.0–1.5, +0.0292 beyond 1.5.
- **Why.** T is fit at the main sportsbook line and applied to every rung. At the main line the model's disagreement with the line carries little information in these cells, so the data ask for heavy flattening. At a far alternate rung the probability is driven by how far the line sits from the projection, and that carries real information. One scalar cannot serve both. The slope that minimises log-loss on each rung type, fitted on those same rungs (descriptive, in sample):

  | Cell | Penalised slope 1/T | Unpenalised slope | Best slope, main rungs (n) | Best slope, alternate rungs (n) | Δ log-loss on alternate rungs, unpenalised |
  |---|---:|---:|---|---|---:|
  | NFL receptions | .72 | .29 | .07 (524) | .81 (2,281) | +.0674 |
  | NFL rushing yards | .70 | .25 | .00 (210) | .89 (2,326) | +.0552 |
  | NFL interceptions | .79 | .48 | .00 (102) | 1.97 (97) | +.1065 |
  | NFL passing TDs | .79 | .42 | .28 (104) | .74 (206) | +.0203 |
  | NFL QB yards | .82 | .71 | .00 (84) | .19 (890) | −.0188 |
  | NFL yards | .90 | .83 | .07 (115) | .49 (1,770) | −.0106 |
  | MLB total bases | .95 | .87 | .70 (1,487) | .95 (3,898) | +.0011 |
  | MLB hits | .94 | .88 | .76 (1,501) | 1.03 (1,527) | +.0022 |

  The penalty lands near the alternate-rung optimum in the first four cells by accident: it was never fit to those rungs. In NFL QB yards and NFL yards even the alternate rungs want more flattening than either arm gives. This is the same reading as R3 Finding 11: "No single price-blind temperature is right for both."
- **The cost is one-sided.** Over all settled rungs the favoured side already reads 1.2 pp below its hit rate with the penalty and 3.0 pp below without it. The overstatement lives only in the selected tail. Removing the penalty trims the selected tail by making the bulk more cautious; it does not create new overstated legs.
- **It is not where the money is.** The alternate-rung information the penalty preserves does not show up as profitable legs. In those four cells the penalised arm recommends 173 alternate-rung legs, which hit .457 against a read of .625 and lose 17.9 % per pick. The far favourites the model reads correctly are the ones the platform prices correctly too.
- **Who does pay.** Any consumer of the probability at a far rung in those cells, for example the displayed win probability, parlay leg scoring and ladder pricing in the `dfs-products` lane. None of these was measured here.
- **Against the earlier estimate.** R3 measured +0.0009 to +0.0016 on live posted legs with an extra temperature applied on top. Replacing T in the chain and scoring every held-out rung gives +0.0059, and NFL alone +0.0113 against R3's +0.0029.

### Finding 7: for which cell sizes the penalty matters

Held-out change without the penalty, by validation-set size, on the 36 temperature-last moved cells. The last three columns are the bootstrap standard deviation of the calibration slope 1/T in each arm and the slope shift the penalty imposes (medians over the cells in the row).

| Validation rows | Cells | Δ log-loss, all held-out rows | Δ log-loss, gate population | SD of slope, unpenalised | SD of slope, penalised | Slope shift from the penalty |
|---|---:|---|---|---:|---:|---:|
| Under 500 | 3 | −0.0137 [−0.0212, −0.0066] | −0.0213 [−0.0338, −0.0112] | .185 | .063 | .305 |
| 500–1,500 | 4 | −0.0073 [−0.0103, −0.0042] | −0.0045 [−0.0076, −0.0014] | .092 | .044 | .131 |
| 1,500–3,500 | 23 | −0.0015 [−0.0020, −0.0009] | −0.0028 [−0.0038, −0.0018] | .064 | .030 | .114 |
| Over 3,500 | 6 | +0.00005 [−0.00016, +0.00027] | +0.00006 [−0.00015, +0.00027] | .028 | .016 | .051 |

- **The penalty hurts most where it was meant to help.** It does halve the sampling noise in the slope. The bias it adds is larger at every size: under 500 rows the squared bias is about .09 against a variance saving of about .03.
- **Learning curve.** Validation rows were subsampled by player to a ladder of sizes, T was fit in each arm, and each fit was scored on the cell's full test set. Regret is the test log-loss above the best T for that test set, averaged over cells.

  | Validation rows used | Cells | Regret, penalised | Regret, unpenalised | Regret, cross-validated |
  |---:|---:|---:|---:|---:|
  | 100 | 78 | .00382 | .00372 | .00425 |
  | 200 | 78 | .00345 | .00236 | .00299 |
  | 400 | 72 | .00322 | .00169 | .00211 |
  | 800 | 67 | .00228 | .00090 | .00110 |
  | 1,600 | 61 | .00160 | .00057 | .00069 |
  | 3,200 | 16 | .00013 | .00018 | .00018 |
  | 6,400 | 15 | .00008 | .00011 | .00010 |

  The arms tie at 100 rows; in temperature-last cells alone the penalty is ahead there (.00353 against .00377). From 200 rows the unpenalised fit is better. The two bottom rows hold only the large MLB and NHL cells, where the right T is close to 1 and the arms differ by 0.00004.
- **Where the penalty is free insurance.** In the 22 cells whose best test T is at most 1.05 the penalty is better at every size, by a little: 0.0014 at 100 rows, 0.0004 at 400, 0.0001 at 1,600. In the 56 cells whose best T is above 1.05 it costs 0.0007 at 100 rows, 0.0023 at 400 and 0.0015 at 1,600.
- **Answer.** As a guard against over-fitting T, the penalty would pay only below roughly 100–200 validation rows. No served cell is that small: the smallest has 355, and the eight under 500 are all NFL quarterback markets. At served sizes it is a small premium in the cells that want T = 1 and a large cost in the cells that do not.
- **The upper bound becomes reachable.** With the penalty no cell fits T above 1.51. Without it, the share of bootstrap resamples that hit the T = 10 bound is 13 % for NFL rushing yards, 6 % for NFL passing TDs and 2 % for NFL interceptions (and 35 % for NFL carries and 12 % for NFL attempts, both post-hoc cells where it is absorbed). At T = 10 a cell reads close to .5 everywhere. It then recommends nothing and still passes its gates (NFL rushing yards at T = 10: gate 1 −.0033, gate 5 −.0064). The gates cannot tell a calibrated cell from a mute one.

### Finding 8: a cross-validated weight works, and it is not worth building

- **What it picks.** In the 56 moved cells the chosen weight is 0 in 2 cells, 0.0001 in 10, 0.0003 in 17, 0.001 in 9, 0.003 in 6, 0.01 (today's) in 2 and 0.03 or more in 10. All ten of the last group are cells where T is within 0.15 of 1 in every arm. Cross-validation never asks for more penalty where the temperature matters.
- **Against the unpenalised fit, head to head.** Gate-population log-loss: unpenalised better by 0.00004 [0.00001, 0.00007]. Recommended-leg gap: unpenalised worse by 0.47 pp [−1.72, +2.38]. Return per pick: unpenalised worse by 0.85 pp [−4.4, +3.0]. Log-loss over all rungs: unpenalised worse by 0.0013 [0.0002, 0.0023]. The two are indistinguishable on every money endpoint. Cross-validation softens the alternate-rung cost by a fifth (+0.0046 against +0.0059) and does not remove it.
- **Where it is worst.** At 100 validation rows the cross-validated arm has the highest regret of the three (.00425). Tuning a penalty needs data, and the cells that would need the penalty are the ones without data.
- **The one-SE rule fails.** It is worse than today at the sportsbook line (+0.00079) and it flips NBA BLST to fail.
- **Build cost.** Unpenalised: delete one line. Cross-validated: a fold loop, a nine-value grid constant, a tie rule and a test, about 40 lines for one maintainer to own, for no measurable gain over the one-line change.

### Finding 9: side effects on stake sizing and on the supersession bar

- **Stake multipliers rise by more than 0.001 in 11 cells and none changes sign.** `kelly_shrinkage` is the validation Brier skill against the book, clipped to [0, 1], and T is fit on the same rows. Of the 48 cells with a book, 42 are positive in both arms. The sum goes from 2.451 to 2.524. The largest moves: NFL receptions .0120 → .0300, NFL rushing yards .0031 → .0169, NBA STL .1062 → .1141, NFL passing TDs .2113 → .2190, NBA BLST .0310 → .0369, NFL interceptions .2477 → .2529, NBA REB .0077 → .0103, WNBA PR .0151 → .0177. The held-out skill agrees in direction (NFL receptions, test rows: .0057 → .0288), so this is not only in-sample optimism.
- **What that does to money.** The unit totals in Finding 5 count one unit per leg. Under Kelly sizing a receptions leg is staked 2.5× larger without the penalty, while the cell recommends 57 % fewer legs. In stake-weighted terms the receptions saving is roughly a wash (multiplier × legs: 2.6 penalised against 2.7 unpenalised); the rushing yards saving survives because the cell nearly stops recommending.
- **The supersession bar is tilted by the change.** `supersede_verdict` compares a candidate's test-set dump with the incumbent's stored dump. Scoring each cell's unpenalised probabilities against its own stored ones, as if the change were a candidate: all 52 of the 56 moved cells that pass today pass S1; 9 pass S2 (NFL rushing yards +0.0074, NFL interceptions +0.0069, NFL passing TDs +0.0051, NFL receptions +0.0048, WNBA PR +0.0022, WNBA STL +0.0021, NHL goalie fantasy points, NFL yards, MLB hits); 5 pass S3; 2 would formally supersede (NFL rushing yards, WNBA PR). Two post-hoc cells are significantly worse by a small amount (WNBA PA −0.0005, NFL QB TDs −0.0004).
  - First reading: cell by cell this change clears the project's own replacement bar in 2 of 56 cells. It is a pooled improvement driven by NFL, not 56 small wins.
  - Second reading: after the change, any candidate trained under the new rule is compared with an incumbent dump made under the old one, and can pass S2 on the temperature rule alone.

### Finding 10: what the literature says about a regularised or small-sample temperature

1. **Temperature scaling as defined has no penalty.** Guo et al. (2017, arXiv:1706.04599) fit one scalar on a held-out validation set by negative log-likelihood. They found the one-parameter form the most reliable of the Platt-style variants they tried, and the many-parameter variants prone to over-fit. One parameter is the regulariser.
2. **Calibration maps are regularised when they have many parameters, and then the strength is cross-validated.** Kull et al. (2019, arXiv:1910.12656) add L2 and off-diagonal-and-intercept penalties to Dirichlet calibration and choose their weights by an inner 3-fold cross-validation (their Appendix D.1). Temperature scaling is the one-parameter special case of that family and gets no penalty. Ridge logistic regression with a cross-validated ridge goes back to le Cessie & van Houwelingen (1992).
3. **The classical small-sample regulariser for a sigmoid calibrator fades with sample size.** Platt (1999) replaces the 0/1 targets with (N₊ + 1)/(N₊ + 2) and 1/(N₋ + 2), a uniform-prior estimate whose pull vanishes as the counts grow. Lin, Lin & Weng (2007, doi:10.1007/s10994-007-5018-6) keep those targets in their corrected algorithm. The weight here does the opposite: it is constant per row, so its total pull grows with the sample.
4. **Low-parameter scaling is the sample-efficient choice, and small sets still make it noisy.** Niculescu-Mizil & Caruana (2005, doi:10.1145/1102351.1102430) find Platt scaling better than isotonic regression when the calibration set is small, on the order of a thousand cases or fewer. Kumar, Liang & Ma (2019, arXiv:1909.10155) show scaling methods need far fewer samples than binning. Mozafari et al. (2018, arXiv:1810.11586) note that "when the validation set does not contain enough correctly and misclassified samples, TS finds the suboptimal T value". Zhang, Kailkhura & Han (2020, arXiv:2003.07329) treat data efficiency as a design goal for the same reason.
5. **Shrinkage does not rescue small samples, and a tuned penalty is least reliable where it is most wanted.** Van Calster et al. (2020, doi:10.1177/0962280220921415) find shrinkage improves the calibration slope on average and varies widely from sample to sample, most at small sizes. Riley et al. (2021, doi:10.1016/j.jclinepi.2020.12.005) trace this to tuning parameters estimated with large uncertainty. Steyerberg et al. (2004, doi:10.1002/sim.1844) recommend the simplest recalibration when the updating set is small. Finding 8 shows the same pattern here: the cross-validated arm has the highest regret at 100 rows.
6. **How precisely a slope can be estimated is a sample-size question with a known form.** Riley et al. (2021, doi:10.1002/sim.9025) derive the standard error of the calibration slope for a validation sample. The measured standard deviations here, .19 under 500 rows and .03 above 3,500, are of that kind.
7. **Calibration fit on one population does not carry to a shifted one.** Ovadia et al. (2019, arXiv:1906.02530) show post-hoc temperature scaling degrades under dataset shift. Hébert-Johnson et al. (2018, arXiv:1711.08513) show that calibration overall implies nothing about subpopulations; the boosting view of the same idea is [44] in `operation_ship_references.md` (Globus-Harris et al. 2023, arXiv:2301.13767). In the hierarchy of Van Calster et al. (2016, doi:10.1016/j.jclinepi.2015.12.005), a slope of 1 is only "weak" calibration. Main line against alternate line is exactly such a split (Finding 6).
8. **Selecting on an estimate guarantees disappointment after selection** (Smith & Winkler 2006, doi:10.1287/mnsc.1050.0451). A calibration change can trim the recommended-leg gap. It cannot close it.
9. **The one-standard-error rule** (Hastie, Tibshirani & Friedman 2009, doi:10.1007/978-0-387-84858-7; `lambda.1se` in Friedman, Hastie & Tibshirani 2010, doi:10.18637/jss.v033.i01) prefers the more regularised model within one standard error of the best. Here "more regularised" means "closer to T = 1", the direction the data reject.

No source was found that recommends a fixed, sample-size-independent L2 pull on a single temperature. The literature's answer to "regularise the temperature?" is that one parameter needs none, and that where a penalty is used its strength is tuned and scales with the data.

## 5. Recommendation and routing protocol

**GO: remove the penalty.** By the pre-registered rule the unpenalised fit is measurably better and flips no gate. The cross-validated weight also meets both conditions; it is not recommended, because it is indistinguishable from the unpenalised fit on every money endpoint and costs code the one-line change does not (Finding 8).

This is a trade, and the owner should take it knowing what is given up: alternate-line probabilities in four NFL cells get worse (Finding 6), recommended volume falls by about a quarter, and NFL rushing yards nearly stops recommending. KILL remains a defensible answer for an owner who weighs the alternate-rung read above the recommended-leg read.

**The code change.** In `src/sportstradamus/training/pipeline.py`, function `_brier_temperature_loss` (lines 2845–2850 in the working tree, 2841–2846 at HEAD): delete `reg = 0.01 * (T - 1) ** 2`, return the Brier alone, and correct the docstring.

```python
def _brier_temperature_loss(T: float, val_logits: np.ndarray, y_class_val: np.ndarray) -> float:
    """Brier score at temperature ``T``. Pure."""
    return np.mean((expit(val_logits / T) - y_class_val) ** 2)
```

- **Leave alone in this change:** the `(1.0, 10.0)` bounds, the fit population, and `TEMPERATURE_REGULARIZATION = 0.01` in `training/group_conditional_cdf/_contracts.py:41` (used by `_line_head._optimize_temperature` for the two-part and affine structural strategies). No served cell uses a structural strategy, so neither condition could be tested there (open question 2).
- **Tests.** No test calls `_brier_temperature_loss` or pins a temperature it fitted; the fixtures that mention a temperature hard-code one. One small golden test would freeze the decision: on synthetic log-odds with a true slope of 0.3, the fitted T is near 3.3 (the penalised fit returns about 1.4).
- **Do not bump `implementation_version`.** A bump orphans every model file and serving then skips them, which is a withhold by another name. Model files from before and after the change are told apart by `trained_at` and the date in `Model Version`.

**When it takes effect.** At each cell's next `meditate`. Serving reads T from the model file (`prediction/model_prob.py:1255`), so nothing changes until that file is rewritten. Writing a refit T into existing model files is possible in principle and not recommended: it would be a second writer of model files outside `meditate`, and NFL, where nearly all of the effect is, is retrained weekly.

**How the retrain verifies condition (b).**

1. Before the retrain, copy `model_stats.parquet` aside. `report()` rewrites it.
2. Retrain as usual.
3. Compare `ship`, `g1_brier_diff_ci_hi` and `g5_ece_debiased` per cell, before and after.
4. For any cell that passed before and fails after, replay the new model file's calibration at weight 0.01 and at weight 0 from the same validation log-odds (the method of Finding 1). If it passes at 0.01 and fails at 0, the change caused the failure, condition (b) is broken for that cell, and the owner's rule says restore the penalty. If it fails at both, the new booster caused it and the temperature is not the reason.
5. Watch first: NFL sacks taken (its T must stay at or below 1.17), NFL completions (gate 1 .0041), and the five cells that need T above a floor (Finding 3).
6. Run `sportstradamus admin tail-scorecard` before and after for the money read. Tag receipts by `Model Version` and do not pool across the change (honest-receipts §6).

**What to expect in numbers.** These come from replaying today's boosters; a retrain fits new boosters, so the realised T will differ.

| League | Cells whose T moves | Largest changes in T | At the sportsbook line | On the tail scorecard |
|---|---|---|---|---|
| NFL | 15 of 19 | Receptions 1.38 → 3.4 (bootstrap 5–95 %: 2.5–5.6), rushing yards 1.42 → 4.1 (2.4–10), passing TDs 1.27 → 2.4 (1.3–10), interceptions 1.27 → 2.1 (1.2–5.4) | Log-loss −0.005. Gate 1 upper bound falls by .006 in receptions and rushing yards. Gate 5 falls in interceptions (.064 → .041), passing TDs (.032 → .012) and receptions (.043 → .013) | Recommended legs −24 to −29 %, gap −3.3 to −3.8 pp, receptions −57 %, rushing yards close to none. Log-loss over alternate rungs +0.015 |
| NBA | 14 of 18 | STL 1.24 → 1.71, OREB 1.30 → 1.77, BLST 1.16 → 1.41, PTS 1.07 → 1.27 | Log-loss −0.0006 | No read yet |
| WNBA | 13 of 14 | FTM 1.38 → 2.12, PR 1.28 → 2.04, BLST 1.29 → 1.86, STL 1.26 → 1.79 | Log-loss −0.0011 | 30 rungs; no read |
| MLB | 7 of 15 | Hits allowed 1.12 → 1.30, total bases 1.05 → 1.15 | No change | Recommended legs −22 to −24 %, money unchanged |
| NHL | 7 of 12 | Shots 1.12 → 1.41, blocked 1.17 → 1.44, goalie fantasy points 1.21 → 1.47 | No change | No read |

Fleet-wide: 73 of 78 cells keep passing; recommended legs fall by a quarter; the recommended-leg gap falls from about 13 pp to about 10 pp on the scorecard (expect −1 to −3 pp live); log-loss over all DFS rungs rises by about 0.006. If NFL rushing yards shows almost no recommended legs after its retrain, that is this change and not a bug.

**When to reverse.** Restore the line if step 4 attributes a pass-to-fail flip to the temperature, or if four weekly scorecard runs show the recommended-leg gap no lower than before while log-loss over all rungs stays more than 0.005 worse.

**Routing.**

| Item | Verdict | Kind |
|---|---|---|
| Remove the penalty in `_brier_temperature_loss` | GO | Engineering: one line, takes effect at the retrain |
| Cross-validated weight | Do not build | Meets both conditions; no gain over the line above |
| One-SE weight, or any larger weight | KILL | Worse than today; flips NBA BLST |
| A temperature that differs between main and alternate rungs | Open | Research: unproven. R3 found a fit on DFS-rung rows did not help |
| The structural strategies' copy of the penalty | Open | Untested; no served cell |

## 6. What was tried and failed (everything run)

- **One-SE weight.** Worse than today on every pooled endpoint at the sportsbook line; one pass-to-fail flip.
- **Rescaling the stored `P` directly** instead of replaying, for the inexact cells. It agrees with the replay and every cell still passes. It cannot be used for post-hoc cells, where the probability post-hoc sits on top of T.
- **The supersession bar as a per-cell yardstick.** Two of 56 cells clear it (Finding 9).
- **The 1.83 baseline.** Strengthens P2. Not pre-registered as primary.
- **Gates along each cell's T bootstrap band and across a T grid.** One exposure, NFL sacks taken (Finding 3).
- **Not run.** No retrain. No fixed intermediate weight such as 0.001. No change to the lower bound. No arm for the structural strategies. No NBA or NHL tail read.

## 7. Reality checks

1. **This is not the cure.** Without the penalty the recommended legs still read 10 pp above their hit rate and lose about 6 % per pick; the legs recommended in both arms still lose 5.8 %. The gap is selection on model–market disagreement (R3 Finding 1; Smith & Winkler 2006), and a flatter temperature trims it by about a quarter.
2. **The benefit is narrow.** Four NFL cells account for 60 of the 67 units saved, by recommending 71 % fewer legs. Outside them the change is 751 → 711 legs and 7 units.
3. **The cost is on the product surface.** Alternate-line reads in those four cells get worse by 0.02–0.11 log-loss per rung, and no gate measures it. If ladder pricing or far-rung legs matter more than recommended main-line legs, KILL is the better answer.
4. **P2 rests on 23 game days, 10 of them NFL.** With that few clusters a percentile bootstrap interval is too narrow (Cameron, Gelbach & Miller 2008, doi:10.1162/rest.90.3.414). P1a and P1b rest on 409 days and do not share the problem.
5. **Held-out here is out of sample, not out of time.** Validation and test rows cover the same dates. On the scorecard's main rungs the best NFL slope is near 0, flatter than even the unpenalised fit, which suggests the right T is not stable from the validation window to the scorecard window. R3's walk-forward estimate on live legs (−1.0 to −1.2 pp) is the out-of-time read, and it is smaller than this brief's.
6. **Thirteen cells do not replay exactly**, two materially. Restricting to the 65 exact cells leaves the pooled result unchanged (−0.00041 on the gate population).
7. **A retrain refits the booster.** Gate outcomes after the retrain will differ from Appendix A for reasons unrelated to the temperature. Step 4 of the protocol exists to separate the two.
8. **NFL sacks taken carries more retrain risk without the penalty:** 11.5 % against 3.2 % that noise alone fits a T that fails gate 5.
9. **A cell can go mute.** NFL rushing yards has a 13 % chance of fitting T = 10 on a resample. It would still pass its gates and would recommend nothing.
10. **Stake sizes move with it.** `kelly_shrinkage` rises 2.5× in NFL receptions. The flat-unit totals do not capture that.
11. **NFL book quotes are themselves inconsistent** (`ev` and `under_prob` disagree; honest-receipts §6, I6d). Every fused NFL number here inherits that.
12. **MLB pays in volume for nothing.** A quarter of its recommended legs disappear and the money does not move.

## 8. Open questions / caveats (for the plan's "Open questions")

1. **A distance-aware temperature.** The data want a slope near 0 at NFL main rungs and 0.7–2.0 at the alternate rungs of four cells. R3 found a pooled fit on DFS-rung rows did not help (gap +0.20 pp, bulk log-loss worse). Whether a slope that depends on distance from the projection would is a research question. It belongs with I6d (NFL at-market information) and with the second half of I6c (the fit population).
2. **The structural strategies' copy** (`TEMPERATURE_REGULARIZATION`, `_line_head._optimize_temperature`). Left at 0.01, a two-part or affine candidate in the sweep is fit under the old rule and compared with temperature-last candidates under the new one. No served cell depends on it. Its golden tests check fold counts and bounds, not a fitted value, as far as a grep shows.
3. **The lower bound.** 21 cells would fit T below 1. Not tested; a separate decision.
4. **NBA and NHL tail reads.** Their test sets predate the scorecard's columns. Read P2 for those leagues after their season-start retrain re-dumps them.
5. **Stale evidence after the change.** Stored corner verdicts in `research/confirm_nominee_gates.csv` and every incumbent's stored test-set dump were produced under the penalised rule. No identity hash moves. Re-dump an incumbent under the new code before a supersession walk on any of the 36 temperature-last moved cells (Finding 9).
6. **A fixed intermediate weight** (for example 0.001) was not run. The cross-validated choices span 0 to 0.1 by cell, so no single constant is supported.
7. **`kelly_shrinkage` is fit on the rows T is fit on.** Whether the stake multiplier should come from held-out rows is a sizing question outside this brief.
8. **Re-read P2 after four weekly scorecard runs,** tagged by `Model Version`, before treating the −3 pp as established.
9. **The replay harness lives in a session scratch directory** and will be lost with `/tmp`. Copy it out if step 4 of the retrain protocol is wanted (the path is at the end of this brief).
10. **Parlays were not measured.** The tail scorecard scores single legs. Parlay construction scores its legs with the same probability, so lower far-rung reads in the four NFL cells will change which parlays are built; the size and sign of that are unknown.
11. **The integration suite was not run** (dispatch constraint). `tests/integration/test_end_to_end.py` trains in fake mode and may pin calibration outputs.

## 9. Appendix A: every served cell, gates today and without the penalty

No row changes its verdict. "Today" is the stored test set, equal to `model_stats.parquet`. The penalised T shown is the replayed fit, which equals the stored T within 0.0004 in 76 cells (Finding 1). A dagger marks a value inside the pre-registered near-miss band (gate 1 above .004, gate 5 above .060). Gates 2, 3, 4 and 6 do not read `P` and are identical in every arm. Where the note says the replay does not reproduce the stored `P`, the "today" value and the unpenalised value sit on different bases; compare the unpenalised value with the replayed penalised one given in the note.

| Cell | Post-hoc stage | Validation rows | T: penalised → unpenalised (cross-validated) | Six gates: today → unpenalised / cross-validated | Gate 1 CI upper, passes below .005: today → unpenalised (cross-validated) | Gate 5 debiased ECE, passes below .075: today → unpenalised (cross-validated) | Note |
|---|---|---:|---|---|---|---|---|
| MLB batter-strikeouts | Platt, after T | 2,380 | 1.101 → 1.292 (1.174) | PASS → PASS / PASS | no book | +.0006 → +.0007 (+.0006) |  |
| MLB doubles | — | 10,227 | 1.000 → 1.000 (1.000) | PASS → PASS / PASS | −.0005 → −.0005 (−.0005) | +.0033 → +.0033 (+.0033) |  |
| MLB hits | — | 10,300 | 1.067 → 1.136 (1.131) | PASS → PASS / PASS | +.0007 → +.0005 (+.0005) | +.0073 → +.0039 (+.0042) |  |
| MLB hits+runs+rbi | — | 9,870 | 1.000 → 1.000 (1.000) | PASS → PASS / PASS | −.0005 → −.0005 (−.0005) | +.0079 → +.0079 (+.0079) |  |
| MLB hits-allowed | mean (roe), before T | 1,101 | 1.124 → 1.295 (1.250) | PASS → PASS / PASS | −.0001 → −.0007 (−.0006) | −.0105 → −.0087 (−.0097) |  |
| MLB home-runs | — | 10,032 | 1.177 → 1.243 (1.240) | PASS → PASS / PASS | −.0467 → −.0467 (−.0467) | +.0033 → +.0040 (+.0038) |  |
| MLB pitcher-fantasy-points-underdog | — | 548 | 1.000 → 1.000 (1.000) | FAIL (gate 4) → FAIL (gate 4) / FAIL (gate 4) | no book | +.0096 → +.0096 (+.0096) |  |
| MLB pitches-thrown | CDF (isotonic), before T | 822 | 1.000 → 1.000 (1.000) | PASS → PASS / PASS | no book | +.0110 → +.0110 (+.0110) |  |
| MLB rbi | — | 10,188 | 1.000 → 1.000 (1.000) | PASS → PASS / PASS | −.0002 → −.0002 (−.0002) | +.0102 → +.0102 (+.0102) |  |
| MLB runs | — | 9,748 | 1.000 → 1.000 (1.000) | PASS → PASS / PASS | −.0009 → −.0009 (−.0009) | +.0142 → +.0142 (+.0142) |  |
| MLB runs-allowed | Platt, after T | 1,140 | 1.271 → 2.550 (2.267) | FAIL (gate 4) → FAIL (gate 4) / FAIL (gate 4) | +.0047† → +.0047† (+.0047†) | +.0006 → +.0006 (+.0007) |  |
| MLB singles | — | 9,800 | 1.000 → 1.000 (1.000) | PASS → PASS / PASS | +.0002 → +.0002 (+.0002) | −.0002 → −.0002 (−.0002) |  |
| MLB stolen-bases | Platt, after T | 8,450 | 1.000 → 1.000 (1.000) | PASS → PASS / PASS | −.0135 → −.0135 (−.0135) | +.0082 → +.0082 (+.0082) |  |
| MLB total-bases | — | 10,037 | 1.053 → 1.146 (1.121) | PASS → PASS / PASS | −.0009 → −.0009 (−.0009) | +.0003 → +.0030 (+.0020) |  |
| MLB walks | — | 8,786 | 1.004 → 1.006 (1.000) | PASS → PASS / PASS | −.0000 → −.0000 (−.0000) | −.0032 → −.0030 (−.0035) |  |
| NBA AST | — | 2,176 | 1.075 → 1.211 (1.130) | PASS → PASS / PASS | +.0010 → +.0013 (+.0011) | −.0033 → −.0011 (−.0036) |  |
| NBA BLK | Platt, after T | 2,263 | 1.370 → 1.669 (1.646) | PASS → PASS / PASS | −.0354 → −.0355 (−.0355) | +.0003 → +.0003 (+.0003) |  |
| NBA BLST | mean (roe), before T | 2,219 | 1.164 → 1.406 (1.379) | PASS → PASS / PASS | +.0032 → +.0016 (+.0017) | +.0326 → +.0278 (+.0283) | replay does not reproduce the stored P exactly (largest row difference 0.0164); compare the unpenalised value with the replayed penalised one: gate 1 +.0036, gate 5 +.0337 |
| NBA DREB | isotonic, after T | 2,114 | 1.091 → 1.147 (1.091) | PASS → PASS / PASS | no book | +.0134 → +.0134 (+.0134) |  |
| NBA FG3A | — | 2,186 | 1.016 → 1.027 (1.000) | PASS → PASS / PASS | no book | +.0379 → +.0379 (+.0382) |  |
| NBA FG3M | mean (isotonic), before T | 2,224 | 1.000 → 1.000 (1.000) | PASS → PASS / PASS | −.0384 → −.0396 (−.0396) | +.0451 → +.0297 (+.0297) | replay does not reproduce the stored P exactly (largest row difference 0.1678); compare the unpenalised value with the replayed penalised one: gate 1 −.0396, gate 5 +.0297 |
| NBA FGM | — | 2,089 | 1.000 → 1.000 (1.000) | PASS → PASS / PASS | no book | +.0383 → +.0383 (+.0383) |  |
| NBA FTM | Platt, after T | 2,200 | 1.336 → 2.063 (1.913) | PASS → PASS / PASS | −.0004 → −.0004 (−.0004) | +.0018 → +.0018 (+.0018) |  |
| NBA MIN | — | 1,955 | 1.000 → 1.000 (1.000) | PASS → PASS / PASS | no book | +.0093 → +.0093 (+.0093) |  |
| NBA OREB | — | 2,180 | 1.295 → 1.772 (1.703) | PASS → PASS / PASS | no book | +.0311 → +.0251 (+.0237) |  |
| NBA PA | isotonic, after T | 2,195 | 1.153 → 1.331 (1.316) | PASS → PASS / PASS | +.0034 → +.0034 (+.0034) | −.0030 → −.0030 (−.0030) | replay does not reproduce the stored P exactly (largest row difference 0.0465); compare the unpenalised value with the replayed penalised one: gate 1 +.0034, gate 5 −.0030 |
| NBA PR | Platt, after T | 2,170 | 1.172 → 1.430 (1.402) | FAIL (gate 1) → FAIL (gate 1) / FAIL (gate 1) | +.0056† → +.0053† (+.0053†) | +.0189 → +.0186 (+.0186) |  |
| NBA PRA | Platt, after T | 2,214 | 1.341 → 2.660 (2.409) | PASS → PASS / PASS | +.0020 → +.0019 (+.0019) | +.0081 → +.0076 (+.0077) |  |
| NBA PTS | — | 2,224 | 1.074 → 1.265 (1.197) | PASS → PASS / PASS | +.0039 → +.0034 (+.0035) | +.0087 → +.0075 (+.0078) |  |
| NBA RA | isotonic, after T | 2,216 | 1.204 → 1.482 (1.404) | PASS → PASS / PASS | +.0016 → +.0016 (+.0016) | +.0059 → +.0058 (+.0059) | replay does not reproduce the stored P exactly (largest row difference 0.0022); compare the unpenalised value with the replayed penalised one: gate 1 +.0016, gate 5 +.0059 |
| NBA REB | — | 2,226 | 1.100 → 1.259 (1.215) | PASS → PASS / PASS | +.0040 → +.0029 (+.0032) | +.0235 → +.0249 (+.0246) |  |
| NBA STL | — | 2,156 | 1.238 → 1.714 (1.634) | PASS → PASS / PASS | −.0295 → −.0310 (−.0310) | +.0111 → +.0071 (+.0075) |  |
| NBA fantasy-points-prizepicks | — | 1,892 | 1.000 → 1.000 (1.000) | PASS → PASS / PASS | −.0064 → −.0064 (−.0064) | +.0134 → +.0134 (+.0134) |  |
| NFL attempts | Platt, after T | 393 | 1.298 → 3.310 (2.636) | PASS → PASS / PASS | +.0024 → +.0025 (+.0024) | +.0069 → +.0069 (+.0069) |  |
| NFL carries | Platt, after T | 1,253 | 1.508 → 6.646 (6.646) | PASS → PASS / PASS | −.0004 → −.0004 (−.0004) | +.0042 → +.0042 (+.0042) |  |
| NFL completions | Platt, after T | 397 | 1.333 → 4.209 (4.209) | PASS → PASS / PASS | +.0041† → +.0041† (+.0041†) | −.0017 → −.0014 (−.0014) |  |
| NFL fantasy-points-prizepicks | — | 2,824 | 1.000 → 1.000 (1.000) | PASS → PASS / PASS | no book | +.0068 → +.0068 (+.0068) |  |
| NFL fantasy-points-underdog | CDF (isotonic), before T | 2,963 | 1.066 → 1.114 (1.015) | PASS → PASS / PASS | no book | −.0028 → −.0018 (−.0040) |  |
| NFL interceptions | mean (isotonic), before T | 402 | 1.270 → 2.075 (1.869) | PASS → PASS / PASS | −.0656 → −.0737 (−.0722) | +.0638† → +.0410 (+.0449) |  |
| NFL passing-first-downs | — | 355 | 1.000 → 1.000 (1.000) | PASS → PASS / PASS | no book | +.0030 → +.0030 (+.0030) |  |
| NFL passing-tds | mean (isotonic), before T | 385 | 1.274 → 2.379 (1.980) | PASS → PASS / PASS | −.0260 → −.0305 (−.0298) | +.0323 → +.0122 (+.0150) |  |
| NFL qb-tds | Platt, after T | 417 | 1.371 → 2.023 (1.908) | PASS → PASS / PASS | no book | +.0149 → +.0169 (+.0169) | replay does not reproduce the stored P exactly (largest row difference 0.0023); compare the unpenalised value with the replayed penalised one: gate 1 —, gate 5 +.0149 |
| NFL qb-yards | — | 360 | 1.223 → 1.400 (1.364) | PASS → PASS / PASS | no book | +.0159 → +.0202 (+.0193) |  |
| NFL receiving-tds | — | 3,011 | 1.041 → 1.053 (1.013) | PASS → PASS / PASS | no book | +.0071 → +.0064 (+.0088) |  |
| NFL receiving-yards | isotonic, after T | 3,007 | 1.266 → 3.806 (2.667) | PASS → PASS / PASS | +.0014 → +.0014 (+.0014) | +.0043 → +.0043 (+.0043) |  |
| NFL receptions | — | 2,970 | 1.381 → 3.439 (2.763) | PASS → PASS / PASS | +.0030 → −.0031 (−.0026) | +.0426 → +.0132 (+.0134) |  |
| NFL rushing-tds | — | 1,283 | 1.126 → 1.170 (1.126) | PASS → PASS / PASS | no book | +.0032 → −.0008 (+.0032) |  |
| NFL rushing-yards | CDF (isotonic), before T | 1,310 | 1.422 → 4.052 (3.060) | PASS → PASS / PASS | +.0028 → −.0028 (−.0022) | +.0298 → −.0026 (+.0017) |  |
| NFL sacks-taken | — | 398 | 1.000 → 1.000 (1.000) | PASS → PASS / PASS | no book | +.0696† → +.0703† (+.0703†) | replay does not reproduce the stored P exactly (largest row difference 0.0037); compare the unpenalised value with the replayed penalised one: gate 1 —, gate 5 +.0703 |
| NFL targets | isotonic, after T | 2,815 | 1.000 → 1.000 (1.000) | PASS → PASS / PASS | no book | −.0000 → −.0000 (−.0000) |  |
| NFL tds | mean (isotonic), before T | 3,274 | 1.026 → 1.034 (1.018) | PASS → PASS / PASS | −.0227 → −.0227 (−.0227) | +.0059 → +.0057 (+.0062) |  |
| NFL yards | mean (roe), before T | 2,962 | 1.116 → 1.204 (1.164) | PASS → PASS / PASS | no book | +.0146 → +.0091 (+.0116) | replay does not reproduce the stored P exactly (largest row difference 0.0004); compare the unpenalised value with the replayed penalised one: gate 1 —, gate 5 +.0147 |
| NHL assists | — | 11,649 | 1.000 → 1.000 (1.000) | PASS → PASS / PASS | −.0013 → −.0013 (−.0013) | +.0009 → +.0009 (+.0009) |  |
| NHL blocked | — | 2,960 | 1.168 → 1.435 (1.404) | PASS → PASS / PASS | +.0002 → −.0002 (−.0002) | +.0094 → +.0097 (+.0095) |  |
| NHL goalie-fantasy-points-underdog | CDF (isotonic), before T | 799 | 1.211 → 1.469 (1.401) | FAIL (gate 4) → FAIL (gate 4) / FAIL (gate 4) | −.0030 → −.0036 (−.0037) | +.0383 → +.0230 (+.0266) |  |
| NHL goals | — | 17,560 | 1.196 → 1.266 (1.263) | PASS → PASS / PASS | −.0071 → −.0072 (−.0072) | +.0097 → +.0088 (+.0087) |  |
| NHL hits | isotonic, after T | 2,321 | 1.298 → 1.831 (1.799) | PASS → PASS / PASS | −.0108 → −.0108 (−.0108) | +.0127 → +.0127 (+.0127) |  |
| NHL points | — | 11,728 | 1.000 → 1.000 (1.000) | PASS → PASS / PASS | +.0003 → +.0003 (+.0003) | +.0030 → +.0030 (+.0030) |  |
| NHL powerPlayPoints | isotonic, after T | 10,710 | 1.012 → 1.016 (1.004) | PASS → PASS / PASS | −.0007 → −.0008 (−.0008) | −.0018 → −.0016 (−.0016) | replay does not reproduce the stored P exactly (largest row difference 0.0286); compare the unpenalised value with the replayed penalised one: gate 1 −.0008, gate 5 −.0016 |
| NHL shots | — | 7,961 | 1.123 → 1.407 (1.302) | PASS → PASS / PASS | −.0046 → −.0044 (−.0046) | −.0029 → +.0037 (+.0000) |  |
| NHL shotsAgainst | — | 726 | 1.000 → 1.000 (1.000) | PASS → PASS / PASS | no book | +.0056 → +.0056 (+.0056) |  |
| NHL skater-fantasy-points-underdog | — | 1,945 | 1.091 → 1.129 (1.058) | PASS → PASS / PASS | −.0066 → −.0074 (−.0060) | +.0070 → +.0101 (+.0053) |  |
| NHL sogBS | — | 2,411 | 1.000 → 1.000 (1.000) | PASS → PASS / PASS | no book | +.0532 → +.0532 (+.0532) |  |
| NHL timeOnIce | — | 1,966 | 1.000 → 1.000 (1.000) | PASS → PASS / PASS | no book | +.0119 → +.0119 (+.0119) |  |
| WNBA AST | Platt, after T | 2,191 | 1.063 → 1.142 (1.012) | PASS → PASS / PASS | +.0023 → +.0023 (+.0023) | +.0032 → +.0032 (+.0032) |  |
| WNBA BLK | — | 2,257 | 1.287 → 1.446 (1.375) | PASS → PASS / PASS | no book | +.0319 → +.0350 (+.0327) |  |
| WNBA BLST | mean (isotonic), before T | 2,145 | 1.290 → 1.857 (1.757) | PASS → PASS / PASS | no book | +.0606† → +.0499 (+.0501) |  |
| WNBA DREB | mean (isotonic), before T | 2,194 | 1.000 → 1.000 (1.000) | PASS → PASS / PASS | no book | +.0451 → +.0451 (+.0451) |  |
| WNBA FG3M | mean (isotonic), before T | 2,114 | 1.075 → 1.129 (1.042) | PASS → PASS / PASS | −.0263 → −.0269 (−.0259) | +.0109 → +.0103 (+.0111) |  |
| WNBA FGA | CDF (isotonic), before T | 2,257 | 1.129 → 1.216 (1.177) | PASS → PASS / PASS | no book | +.0287 → +.0299 (+.0295) |  |
| WNBA FTM | — | 2,174 | 1.384 → 2.117 (2.059) | PASS → PASS / PASS | no book | +.0014 → −.0024 (−.0039) |  |
| WNBA MIN | CDF (isotonic), before T | 2,027 | 1.010 → 1.015 (1.000) | PASS → PASS / PASS | no book | +.0114 → +.0113 (+.0119) |  |
| WNBA OREB | — | 2,156 | 1.338 → 1.819 (1.795) | PASS → PASS / PASS | no book | +.0398 → +.0372 (+.0370) |  |
| WNBA PA | Platt, after T | 2,122 | 1.124 → 1.344 (1.318) | PASS → PASS / PASS | +.0030 → +.0025 (+.0026) | +.0081 → +.0099 (+.0096) |  |
| WNBA PR | — | 2,141 | 1.278 → 2.036 (1.954) | PASS → PASS / PASS | +.0029 → +.0009 (+.0010) | +.0180 → −.0011 (−.0008) |  |
| WNBA RA | isotonic, after T | 2,137 | 1.145 → 1.347 (1.292) | FAIL (gate 1) → FAIL (gate 1) / FAIL (gate 1) | +.0086† → +.0086† (+.0086†) | +.0138 → +.0138 (+.0138) |  |
| WNBA STL | mean (isotonic), before T | 2,246 | 1.256 → 1.794 (1.695) | PASS → PASS / PASS | no book | +.0413 → +.0217 (+.0221) |  |
| WNBA TOV | Platt, after T | 2,245 | 1.205 → 1.583 (1.338) | PASS → PASS / PASS | no book | +.0203 → +.0204 (+.0204) |  |

## 10. Bibliography

| # | Source | Identifier |
|---|---|---|
| B1 | Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML* | arXiv:1706.04599 |
| B2 | Kull, M., Perello-Nieto, M., Kängsepp, M., Silva Filho, T., Song, H. & Flach, P. (2019). Beyond temperature scaling: obtaining well-calibrated multiclass probabilities with Dirichlet calibration. *NeurIPS* (Appendix D.1: hyperparameters by inner 3-fold cross-validation) | arXiv:1910.12656 |
| B3 | Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In *Advances in Large Margin Classifiers*, MIT Press, 61–74 | no DOI (book chapter) |
| B4 | Lin, H.-T., Lin, C.-J. & Weng, R. C. (2007). A note on Platt's probabilistic outputs for support vector machines. *Machine Learning* 68(3):267–276 | doi:10.1007/s10994-007-5018-6 |
| B5 | Niculescu-Mizil, A. & Caruana, R. (2005). Predicting good probabilities with supervised learning. *ICML*, 625–632 | doi:10.1145/1102351.1102430 |
| B6 | Kumar, A., Liang, P. & Ma, T. (2019). Verified uncertainty calibration. *NeurIPS* | arXiv:1909.10155 |
| B7 | Mozafari, A. S., Gomes, H. S., Leão, W., Janny, S. & Gagné, C. (2018). Attended temperature scaling: a practical approach for calibrating deep neural networks | arXiv:1810.11586 |
| B8 | Zhang, J., Kailkhura, B. & Han, T. Y.-J. (2020). Mix-n-Match: ensemble and compositional methods for uncertainty calibration in deep learning. *ICML* | arXiv:2003.07329 |
| B9 | Van Calster, B., van Smeden, M., De Cock, B. & Steyerberg, E. W. (2020). Regression shrinkage methods for clinical prediction models do not guarantee improved performance: simulation study. *Statistical Methods in Medical Research* 29(11):3166–3178 | doi:10.1177/0962280220921415 (arXiv:1907.11493) |
| B10 | Riley, R. D., Snell, K. I. E., Martin, G. P. et al. (2021). Penalization and shrinkage methods produced unreliable clinical prediction models especially when sample size was small. *Journal of Clinical Epidemiology* 132:88–96 | doi:10.1016/j.jclinepi.2020.12.005 |
| B11 | Riley, R. D., Debray, T. P. A., Collins, G. S. et al. (2021). Minimum sample size for external validation of a clinical prediction model with a binary outcome. *Statistics in Medicine* 40(19):4230–4251 | doi:10.1002/sim.9025 |
| B12 | Steyerberg, E. W., Borsboom, G. J. J. M., van Houwelingen, H. C., Eijkemans, M. J. C. & Habbema, J. D. F. (2004). Validation and updating of predictive logistic regression models: a study on sample size and shrinkage. *Statistics in Medicine* 23(16):2567–2586 | doi:10.1002/sim.1844 |
| B13 | Van Calster, B., Nieboer, D., Vergouwe, Y., De Cock, B., Pencina, M. J. & Steyerberg, E. W. (2016). A calibration hierarchy for risk models was defined: from utopia to empirical data. *Journal of Clinical Epidemiology* 74:167–176 | doi:10.1016/j.jclinepi.2015.12.005 |
| B14 | Cox, D. R. (1958). Two further applications of a model for binary regression. *Biometrika* 45(3–4):562–565 | doi:10.1093/biomet/45.3-4.562 |
| B15 | Ovadia, Y., Fertig, E., Ren, J., Nado, Z., Sculley, D., Nowozin, S., Dillon, J. V., Lakshminarayanan, B. & Snoek, J. (2019). Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. *NeurIPS* | arXiv:1906.02530 |
| B16 | Hébert-Johnson, Ú., Kim, M. P., Reingold, O. & Rothblum, G. N. (2018). Multicalibration: calibration for the (computationally-identifiable) masses. *ICML* | arXiv:1711.08513 |
| B17 | Globus-Harris, I. et al. (2023). Multicalibration as boosting for regression ([44] in `operation_ship_references.md`) | arXiv:2301.13767 |
| B18 | Roelofs, R., Cain, N., Shlens, J. & Mozer, M. C. (2022). Mitigating bias in calibration error estimation. *AISTATS* ([48] in `operation_ship_references.md`) | arXiv:2012.08668 |
| B19 | Smith, J. E. & Winkler, R. L. (2006). The optimizer's curse: skepticism and postdecision surprise in decision analysis. *Management Science* 52(3):311–322 | doi:10.1287/mnsc.1050.0451 |
| B20 | Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*, 2nd ed., Springer (§7.10, the one-standard-error rule) | doi:10.1007/978-0-387-84858-7 |
| B21 | Friedman, J., Hastie, T. & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software* 33(1):1–22 | doi:10.18637/jss.v033.i01 |
| B22 | le Cessie, S. & van Houwelingen, J. C. (1992). Ridge estimators in logistic regression. *Applied Statistics* 41(1):191–201 | doi:10.2307/2347628 (publication details confirmed; the DOI was not confirmed by search) |
| B23 | Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics* 6(2):65–70 | JSTOR 4615733 |
| B24 | Cameron, A. C., Gelbach, J. B. & Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. *Review of Economics and Statistics* 90(3):414–427 | doi:10.1162/rest.90.3.414 |
| B25 | Memmel, C. (2003). Performance hypothesis testing with the Sharpe ratio. *Finance Letters* 1:21–23 (the S3 leg of `supersede_verdict`) | no DOI |
| P1 | Prior in-repo brief R3: train/serve skew and tail calibration (Finding 11, open question 5) | `docs/archive/researcher_train_serve_skew.md` |
| P2 | Lane brief: locked decisions 8 and 14, I6c, the tail scorecard | `docs/handoffs/honest-receipts.md` §2, §6, §7 |
| P3 | Commits: the penalty `d43b65c9` (2026-03-29); the start-value seeding change `0c1cf950` (2026-09-28) | git |

Scratch scripts and outputs (read-only reproductions; every table above comes from these scripts or from one-off queries over their outputs): `/tmp/claude-1000/-home-trevor-Sportstradamus/7c225c1a-7c1e-4cde-bede-cf7b5fe0cce8/scratchpad/temp_ridge/`. `PREREG.md` (the pre-registration), `s01_inventory.py` (cells, hashes), `s02_replay.py` (calibration-only replay; captures the validation log-odds), `s03_arms.py` (arms, post-hoc refits, gates), `s03b_pooled.py` (P1a, P1b), `s04_tail_arms.py` and `s04b_tail_boot.py` (tail scorecard per arm, P2 and the secondaries; `TAIL_DIR` and `UD_BASELINE` select the baseline), `s05_cellsize.py` (learning curve, slope noise), `s06_band.py` and `s07_window.py` (gate robustness, supersession yardstick), `s08_table.py` (Appendix A), `s09_rungtype.py` (main against alternate rungs). Outputs: `replay/`, `arms/`, `tail/`, `tail183/`, `cellsize/`.
