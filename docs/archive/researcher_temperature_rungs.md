# In-repo research brief: how should served P(over) be calibrated so it is right at posted alternate rungs without getting worse at the main line? 2026-11-22

## TL;DR

- **Verdict: GO on a reference-anchored temperature. It is a serving and replay change with no training change.** The served probability at any rung L becomes

  `P(L) = expit( logit p_raw(L) − logit p_raw(m) + logit clip(q_m, .01, .99) )`

  - `p_raw` is the over-probability after the whole-CDF recal and before temperature.
  - `m` is the offer's reference line: the sportsbook consensus, else the line of record (`Archive.get_reference_line`), else the offer's own line.
  - `q_m` is today's served probability at `m`: temperature T, then any PROB_STAGE post-hoc, fitted exactly as now.

  At L = m the formula gives today's P exactly. Away from m it keeps the model's own shape and moves only its location, by the same logit shift today's calibration applies at the reference line. Nothing is refit, so pickles, offline gates, `model_stats` and pickle identity all stay unchanged, and no `implementation_version` bump is needed.
- **Out-of-sample effect.** The formula has no fitted parameter. T and the post-hoc come from validation, so the test-half numbers are already honest. Units are Brier ×1e-4 against served, with 95 % intervals from 500 resamples of whole game dates.
  - **P1** (own line): exactly 0 in every league, by construction.
  - **P2** (still-posted rungs):

    | Pickles | P2 change [95 %] | Dates |
    |---|---|---|
    | NFL candidate | −188 [−208, −155] | 22 |
    | NFL control | −178 [−196, −152] | 22 |
    | NBA (dev) | −47 [−56, −36] | 27 |
    | WNBA (dev) | −37 [−51, −22] | 17 |
    | NHL (dev) | −4.1 [−5.5, −2.7] | 48 |
    | MLB (dev) | −5.1 [−9.2, −1.4] | 37 |

  - **NFL forward window**: −205 (candidate) and −202 (control), over 8 dates.
  - **Kill check**: no league and no cell reads KILL. The worst cell point estimates are NHL hits +34 and NFL fantasy points +15 to +18, both with intervals that cross zero.
- **Every alternative reads KILL on the fixed yardstick.** Each design was cross-fit by ISO-week halves of the test half (fit on one half, score the other, then swap). Each improves P2 by 93–197, but:
  - T = 1, a scalar T fit on many thresholds, a scalar T fit on real rungs, caps of 1.5 and 2, and a P-only two-regime map all worsen NFL P1 by +6 to +47. Each P1 interval lies wholly above zero.
  - A distance-dependent T and a fitted lean-removal pass the pooled NFL read but kill NFL receptions on P1 (+8.2 [0.5, 16.5] and +4.4 [0.3, 8.6]). The lean-removal also kills NHL and MLB on P1 (+101, +40). The distance-dependent T can also price the rungs of one prop out of order.
- **Owner call needed (10-04 rule).** For a player with no sportsbook line, `m` is the pick'em platform's main rung. Anchoring the rungs there makes alt-rung prices depend on where Underdog put its main line. Two ways to resolve it:
  - Confirm that this is "evaluation-point anchoring", not "blending": training's T is already fit at those same lines.
  - Or restrict `m` to the sportsbook consensus (`Archive.get_line`). Rows without one keep today's chain. This keeps 86–89 % of the NFL gain (71 % NBA, 77 % WNBA, 70 % NHL, but only 4 % of MLB's small gain).
- **PROB_STAGE post-hocs.** Keep them inside `q_m` for this change, so P1 is untouched and the read isolates the rung fix. Clip `q_m` to [.01, .99] so an isotonic 0 or 1 at the main line cannot pin a whole ladder. Retire `prob_recal_isotonic` afterwards as a separately read stat_meta change: dropping post-hocs reads P1-neutral or better (NBA −6.6 [−12.3, −1.3]; NFL targets −27 / −20) and P2 the same or slightly better.

## 1. Decision question and the statistic that decides it

**Decision.** Choose one calibrator design for the over-probability, with its exact formula, fit population and code sites, so that the served P is right at posted alternate rungs and no worse at the main line.

**Statistic.** `sportstradamus ship priced` (src/sportstradamus/scripts/priced_score.py):

- P1: pick'em and authentic rows at their own line.
- P2: Underdog and Sleeper rungs still posted at the prop's last poll.
- Pushes are dropped, and every interval comes from 500 resamples of whole game dates.
- The primary is `p2_dbrier` when P2 spans 20 or more dates, else `p1_dmse`.
- **KILL** if any judged statistic spanning 20 or more dates has its interval wholly above zero.
- **KEEP** if the primary's interval is wholly below zero and nothing kills.
- **NEUTRAL** otherwise.

A design that buys P2 with any significant P1 Brier loss therefore reads KILL, however large the P2 gain.

**What the defect is.** `training/pipeline.py:_step_calibrate_temperature` (l. 3261) fits one T in `_TEMPERATURE_BOUNDS = (1, 10)` by Brier on validation rows at each row's own line. Serving (`prediction/model_prob.py` l. 1260–1261) and the replay (`scripts/tail_pricing.serve` l. 228–231) apply that T and the PROB_STAGE post-hoc at every rung.

## 2. Data and method

- **NFL arms.** `$L/runs/nfl_verdict_guard/{candidate,control}/models`, using their `test_sets` (test half: 2024-12 to 2026-10-26, 89–97 dates) and `forward` dumps. `$L` = `/home/trevor/backups/sportstradamus/2026-10-21-live-gap-2/partc`.
- **Other leagues.** Dev's served pickles and test sets under `src/sportstradamus/data/{models,test_sets}`.
- **Pricing.** Archive opened read-only. Rungs come from `tail_pricing.price_cell`, cut by `realized_replay.still_posted`. Every probability is re-served through `tail_pricing.serve_live` with the pickle's own knobs. P1 and P2 are scored exactly as `priced_score` scores them (Brier at the line, pushes dropped, intervals from `scorecard._bootstrap_mean_ci_clustered` with `key_rng`).
- **Cross-fit for designs with fitted parameters.** Fit on the dates of one ISO-week parity, score the other, then swap, so every scored row comes from a fit that never saw its week.
- **Fit populations.**
  - **A** (training-like): every priced row at its own line and at `floor(line + k·sd) + .5` for k ∈ {±.5, ±1, ±1.5, ±2}, scored from the realized `Result`.
  - **B**: the fit half's P1 rows plus its real still-posted rungs.
- **Scripts and outputs.** In `/tmp/claude-1000/-home-trevor-Sportstradamus/a1d00aa1-f9ab-405c-90b2-280a0565af2e/scratchpad/rx/`:
  - `xfit2.py`, `summ.py`, `percell.py`;
  - `x2_{nfl_cand,nfl_ctl,dev_NBA,dev_WNBA,dev_NHL,dev_MLB,fwd_candidate,fwd_control}_changes.parquet`, one row per scored row, design and statistic;
  - `x2_*_params.csv`, the fitted parameters per half.

## 3. Key Findings

1. **One scalar T cannot serve both regimes, and the cross-fit shows it out of sample.** Refitting today's own-line T by cross-fit leaves NFL P1 flat (−1.2 [−5.1, 3.1]) and worsens P2 by +133 [104, 160] (control +116 [69, 164]; forward +90 and +118). That is the defect, reproduced on held-out weeks. In logit space, temperature is a calibration slope (Cox 1958, doi:10.1093/biomet/45.3-4.562; Guo et al. 2017, arXiv:1706.04599). For a near-logistic CDF, logit P(Y > L) ≈ (μ − L)/s, so dividing by T is the same as serving a distribution T times wider. NFL's T of 3–10 therefore serves alt rungs from a distribution 3–10 times too wide.

2. **The main-line problem is location noise, not shape, which is why T is the wrong tool.** At a market line the model's lean carries no information: NBA encompassing β is −.11 [−.29, +.11] and NFL's is small and unstable (researcher_market_centre_anchor.md). Let the model's location error relative to the market be e, with spread σ. The Brier cost of that error at a rung δ from the line is about φ(δ/σ)²·var(e)/σ².
   - At δ = 0, NFL attempts' 72e-4 cost implies sd(e)/σ ≈ .21.
   - The same noise costs about 26e-4 at δ = 1σ and about 1e-4 at 2σ.
   - T removes this noise at the line only by inflating the scale, which wrecks every rung where location barely matters.
   - The structural fix moves location per row and leaves the shape alone.

   This is the threshold-calibration point of Gneiting, Balabdaoui & Raftery (2007, doi:10.1111/j.1467-9868.2007.00587.x): calibration at one threshold does not imply calibration at others. Sahoo et al. (2021, NeurIPS 34) show that average calibration leaves threshold decisions miscalibrated unless the forecaster is calibrated at the thresholds the decisions use. Multicalibration makes the same point for subgroups (Hébert-Johnson et al. 2018, arXiv:1711.08513): the own-line pool is one subgroup, and alt rungs are another.

3. **Every scalar or P-only design reads KILL on P1 in NFL** (candidate test half, cross-fit):

   | Design | P1 change [95 %] | P2 change [95 %] | Reads |
   |---|---|---|---|
   | (b) T = 1 | +47.2 [34.4, 64.1] | −156.7 [−180.0, −120.3] | KILL |
   | (a) scalar T fit on population A | +43.5 [31.5, 57.7] | −160.4 [−183.4, −131.5] | KILL |
   | (a) scalar T fit on population B (real rungs) | +26.1 [17.7, 36.1] | −162.8 [−182.6, −139.3] | KILL |
   | (d) cap T at 1.5 | +15.7 [9.1, 22.9] | −139.5 | KILL |
   | (d) cap T at 2 | +6.3 [2.8, 11.5] | −93.2 | KILL |
   | P-only two-regime map, z·(1 − a·e^{−z²/2w²}) | +39.9 [28.2, 54.5] | −160.0 | KILL |

   The control arm is the same: T = 1 gives P1 +50.9, the cap at 2 gives +5.0 [1.1, 8.9]. A map that sees only P cannot tell ".6 at the main line" (which should go to .5) from ".6 one rung off" (which should stay). The threshold-weighted CRPS route of Gneiting & Ranjan (2011, doi:10.1198/jbes.2010.08110), which integrates Brier over a chosen set of thresholds, therefore cannot rescue a single scalar either: whatever weight it puts on the main line, one slope still trades P1 against P2.

4. **The two-regime designs that need `m` work only when the main line's calibration is left alone.**
   - **Distance-dependent T**, `T(d) = 1 + (T0 − 1)·exp(−d²/2h²)` with d = (L − m)/sd:
     - NFL: P1 −6.5 [−15.4, 2.1], P2 −173.1.
     - It kills NFL receptions on P1 (+8.2 [0.5, 16.5]).
     - Because T varies with L, it can price the rungs of one prop out of order: a lower line gets a lower P(over) whenever the model leans against the main line.
   - **Fitted lean removal**, `expit((z(L) − λ·z(m))/T1)`:
     - NFL: P1 −6.2 [−14.1, 0.7], P2 −197.5.
     - It kills receptions on P1 (+4.4 [0.3, 8.6]).
     - It kills NHL (+101 [94, 110]) and MLB (+40 [37, 42]) on P1, where a joint λ fit throws away real main-line skill.
   - **Shared lesson**: refitting the main-line calibration and the rung shape together lets the rung data pull the main line off its optimum.

5. **The reference-anchored temperature keeps the main line exact and wins the rungs in every league.** `logit P(L) = logit p_raw(L) − logit p_raw(m) + logit q_m` (no fitted parameter; called "keepm1" in the scripts):

   | Pickles | P1 rows | P2 rungs / dates | P1 | P2 [95 %] | P2 at alt rungs | Rule |
   |---|---|---|---|---|---|---|
   | NFL candidate, test | 13,333 | 31,594 / 22 | 0 | −188.0 [−207.9, −155.1] | −226.6 | KEEP |
   | NFL control, test | 13,304 | 31,357 / 22 | 0 | −177.5 [−195.6, −151.9] | −213.7 | KEEP |
   | NFL candidate, forward | | / 8 | 0 | −205.4 [−246.7, −179.6] | −246.1 | NEUTRAL (< 20 dates) |
   | NFL control, forward | | / 8 | 0 | −201.5 [−237.1, −180.4] | −241.4 | NEUTRAL (< 20 dates) |
   | NBA, dev served | 3,625 | 14,941 / 27 | 0 | −46.7 [−56.1, −36.1] | −55.0 | KEEP |
   | WNBA, dev served | 3,069 | 3,105 / 17 | 0 | −37.2 [−51.1, −21.8] | −45.7 | NEUTRAL (< 20 dates) |
   | NHL, dev served | 41,159 | 30,293 / 48 | 0 | −4.1 [−5.5, −2.7] | −8.3 | KEEP |
   | MLB, dev served | 94,859 | 37,757 / 37 | 0 | −5.1 [−9.2, −1.4] | −8.8 | KEEP |

   - **Per cell**: no P2 interval lies wholly above zero in any arm or window.
   - **Largest NFL cell gains** (test half): attempts −613, completions −445, receptions −245, carries −295, rushing yards −280, receiving yards −116.
   - **Adverse point estimates**, all with intervals crossing zero: NHL hits +34.0, NFL fantasy points +15.0 / +18.2 on the test half (but −25 / −18 forward), NHL skater fantasy +6.8, MLB hits-allowed +1.8. In these cells T > 1 was also widening a shape that is too sharp. The whole-CDF recal (Rung C) is the right home for that correction (Kuleshov et al. 2018, arXiv:1807.00263), not T.
   - **Shape of the formula**: it is a cumulative-link model (McCullagh 1980, doi:10.1111/j.2517-6161.1980.tb01109.x). Each row's intercept comes from today's reference-line calibration, and the slope across thresholds is the model's own. It separates location from shape in the way EMOS fits a predictive distribution over all thresholds rather than one (Gneiting et al. 2005, doi:10.1175/MWR2904.1).

6. **Fitting a shape temperature T_s is not worth it yet.** The variant `logit q_m + (z(L) − z(m))/T_s` was tried two ways:
   - T_s fit on population A: pooled P2 is about equal to the parameter-free form (NFL −187.9).
     - It kills NHL assists (+1.8, interval above zero).
     - It hits the 0.5 bound in low-count cells (MLB home runs, NHL goals), where the k·sd offsets collapse onto the same half-integer lines.
   - T_s fit on real rungs: MLB pooled +1.7.

   The parameter-free form is as good and cannot overfit.

7. **The PROB_STAGE post-hocs earn nothing at either population.** Taking the post-hoc out of `q_m` ("keepm_noph" in the scripts) gives:
   - NFL P1: −0.9 [−3.4, 1.4] (candidate) and −3.2 [−7.1, 0.4] (control).
   - NBA P1: −6.6 [−12.3, −1.3]. NBA PRA's Platt map costs it.
   - NFL targets P1: −27 / −20; receiving yards: +0.3 / −12.7 (isotonic cells).
   - P2: the same or 1–11 better.

   This agrees with [[prob-recal-isotonic-rung-mispricing]]. Keeping post-hocs inside `q_m` is a choice for read isolation, not for accuracy.

8. **Most of the gain comes from rows that have a sportsbook reference.** Splitting the P2 gain by the test row's quote class:

   | League | Authentic rows: share of rungs, share of gain | Mean change, authentic | Mean change, not authentic |
   |---|---|---|---|
   | NFL | 79 % of rungs, 86–89 % of gain | −204 | −126 / −95 |
   | NBA | 71 % of gain | −68 | −27 |
   | WNBA | 77 % of gain | | |
   | NHL | 70 % of gain | | |
   | MLB | 4 % of gain | | |

   A strict reading of the 10-04 rule, with `m` limited to the sportsbook consensus, keeps most of NFL's gain. The non-authentic replay rows use the test row's line as `m`, and for synthetic rows that is an Avg10 line, not the platform's main rung that serving would use. Their figure is therefore indicative only.

## 4. Recommendation

**Formula.** For every scored rung, with `p_raw` the over-probability after `apply_cdf_recal` and before temperature:

```
q_m   = posthoc(apply_temperature(p_raw(m), T))       # today's served P at the reference line
shift = logit(p_raw(m)) − logit(clip(q_m, .01, .99))  # REF_ANCHOR_CLIP = .01 (named constant)
P(L)  = expit(logit(p_raw(L)) − shift)
```

- `m` is the offer's reference line: `Archive.get_reference_line(league, archive_market(league, market), date, player)` (sportsbook consensus, else line of record). Where that is missing, `m` is the offer's own line, which reproduces today's chain exactly.
- The two-part and affine candidate overrides still replace `cal_over` after this step, as they do today.
- **No fitted parameter and no training change.** `_step_calibrate_temperature`, `_TEMPERATURE_BOUNDS`, the post-hoc fits, test-set `P`, the gates and `model_stats` stay as they are.
- **Universal.** It needs no per-cell stat_meta knob: no league and no cell reads KILL. Because it lives in serving, it reaches every league when production pulls the code, not at each league's next `meditate`. The NFL send does not wait for it: a pickle served by old code behaves exactly as today.

**Code sites.**

1. **`prediction/model_prob.py`, l. 1255–1261.** After `_model_over_and_push`:
   - Compute `raw_ref` with a second `_model_over_and_push` on `offer_df.assign(Line=ref_line)` (push output discarded), then `_apply_cdf_recal_over`.
   - Compute `q_m` with `apply_temperature` then `_apply_prob_posthoc` on `raw_ref`.
   - Replace the two current lines with the shift formula.

   This is one small helper, in the shared distributions helper beside `apply_temperature`, so serving and the replay call the same code.
2. **Reference line before scoring.** `prediction/cli.py` l. 350–361 attaches `Consensus Line` / `Reference Line` only after scoring. Move the `get_reference_line` lookup ahead of `process_offers`, so each offer (Underdog alt rungs are `{**offer, "Line": rung}` copies in `books/underdog.py` l. 349–358) carries `Reference Line` into `model_prob`. Alternatively, `model_prob` can read it through `LazyArchive`. Either way, do not add an `Archive()` at module level (dashboard lock rule).
3. **`scripts/tail_pricing.serve`, l. 228–231.**
   - Add a `ref_line` array: the row's `Line` for test rows (P1, no change) and `Test Line` for priced rungs.
   - Apply the same helper.
   - `serve_live` passes `rows["Test Line"]` when present, else `rows["Line"]`.
   - `price_cell`'s `p_serving` and `p_training` both inherit it, and so do `priced_score` and `realized_replay`.
4. **`scripts/feature_parity.score`, l. 296–303.** No change. Its `weight / temperature` sensitivity is measured at the reference line, where the formula is unchanged.
5. **For the A/B only:** a switch that lets one arm serve under the old chain. Either a `serve(..., anchored=False)` keyword used by a lane script, or a temporary `ship priced --baseline-unanchored`. Delete the switch after the verdict.

**PROB_STAGE post-hocs.**

- In this change they stay inside `q_m`, so the main line is exactly today's.
- Then retire `prob_recal_isotonic` in a separate, separately read stat_meta edit (`posthoc: none`). The cells are NBA DREB/RA/PA, NHL hits/PPP, NFL receiving yards/targets and WNBA PTS/RA.
- Leave `prob_recal_platt` (guarded by a9f4bf70) alone.

## 5. Pre-registration block (paste into the lane STATE.md before the real-chain read)

```
PRE-REGISTRATION: reference-anchored temperature (brief /tmp/researcher_temperature_rungs.md), written 2026-11-22
Correction to the planned read: the fix changes no training step, so refits "with and without the fix" give
bit-identical pickles. Do NOT refit for this read. A/B the serving formula on fixed pickles.
Arms (same pickles, two serving formulas):
  BASE = today's chain: P(L) = posthoc(expit(logit p_raw(L) / T)) at every rung.
  FIX  = P(L) = expit(logit p_raw(L) - logit p_raw(m) + logit clip(q_m, .01, .99)), q_m = BASE at m,
         m = test row's Line (P1) / Test Line (P2 rungs) in the replay.
Pickle sets: (1) NFL nfl_verdict_guard candidate, (2) its control, (3) dev's served pickles, every league
  (NBA, WNBA, NHL, MLB, NFL), (4) any pickle set the NFL send would ship.
Windows: test half from 2024-01-01 (PRIMARY); forward window (CHECK, descriptive: 8 P2 dates < 20, so it
  reads NEUTRAL by rule and cannot KEEP or KILL).
Statistics: ship priced as fixed: p1_dmse, p1_dbrier, p2_dbrier; 500 whole-game-date resamples, seed 1729;
  primary p2_dbrier where P2 spans >= 20 dates, else p1_dmse.
Verdict rule (unchanged): KILL if any judged statistic spanning >= 20 dates has its interval wholly above
  zero; KEEP if the primary is wholly below zero and nothing kills; NEUTRAL otherwise. Read per league and
  per cell. GO = no league KILL and NFL KEEP on both arms.
Expected (from this brief, same pickles): P1 changes identically 0 except rows whose q_m is outside
  [.01, .99]; P2 NFL about -180 (+-30), NBA about -47, NHL about -4, MLB about -5, WNBA NEUTRAL (17 dates).
  A per-cell P2 interval wholly above zero anywhere is a surprise: stop and read that cell.
Secondary (report only): (i) the money bar: realized-replay posted Underdog STANDARD 1.83 legs whose line
  differs from the reference line (the only 1.83 legs whose P moves): count recommended per arm, hit rate
  vs .546, date-clustered interval; (ii) recommended priced alt legs per arm: count and ROI at posted price;
  (iii) watch cells NHL hits, NHL skater fantasy points, NFL fantasy points, MLB hits-allowed.
```

## 6. Reality checks

- **Effect size and where it holds.**
  - P2 gains are large only where T was large: NFL, NBA PRA, WNBA PR. NFL pooled is about −180 to −205 ×1e-4 (from a served P2 Brier of about 1,960, roughly a 10 % cut), and attempts and completions gain −300 to −750.
  - Where T ≈ 1 (most NHL and MLB cells) the formula is nearly the identity, so the gain is −4 to −5. That is real but small.
  - These are offline replays on archive ladders: 22 NFL dates and one season's ladder era. That is enough for the rule, but it is not a live record.
- **Build cost.** About 0.5–1 engineering day: one shared helper, about 10 lines in `model_prob`, moving the reference-line lookup ahead of scoring, about 10 lines in `tail_pricing.serve`, and a temporary A/B switch. No retrain, no send and no identity change. This is an engineering project, not a research one: the method is a reparameterized cumulative-link intercept and has no fitted parameter.
- **What could make it wrong.**
  - **The serving reference line differs from the replay's `Test Line`.** The serving `m` is the reference line at scoring time, while the replay uses the test row's line of record. Line movement between the two shifts the anchor. Most props move ≤ 1 point, but the read cannot see it.
  - **A main line that is wrong by itself drags its whole ladder.** If `q_m` is wrong, for example a stale consensus line or a capped T with a sign-flipped Platt map, every rung inherits the error. Today the error stays at the one line. The a9f4bf70 guard and the [.01, .99] clip bound this risk.
  - **Some cells genuinely want T > 1 at the rungs.** NHL hits and NFL fantasy points look like this. There the formula serves a slightly too-sharp shape. The fix belongs in Rung C (the whole-CDF map), not in a new T. If the real read kills one of these cells, add a per-cell `shape_temperature` then, not now.

## 7. Risks

- **Offline gates and `model_stats`.** Nothing changes: no column, no gate, no Gate-1 interval. They read P at the own line, which is exactly unchanged, and the PIT, which does not read T. The gates were blind to the defect and are blind to the fix. Only `ship priced` P2, the realized replay and live alt-rung legs move. `model_stats.temperature` keeps its meaning: it now sets the reference-line probability only. A T at the cap is no longer a rung hazard, but it still flags a main line the model cannot price.
- **Supersession and Gate 1.** `supersede_verdict` and Gate 1 read offline gates on the test dump, so they are unaffected and no cell owes a re-verdict. **Gate 2** (live graduation) reads live legs. If it reads alt rungs, the deploy date is a break in its series and should be recorded.
- **Money at the 1.83 bar.** Where the Underdog main line equals the reference line, the served P and the recommendation are unchanged. Where the DFS main line sits off the sportsbook consensus, the standard pick's P now follows the model's shape from `m`. In NFL that happens on about 38 % of player-market-days, by about 0.7 ([[dfs-main-line-eval-only]]). When T is high, a half-point gap moves P by up to about .10 on low-count stats (receptions, sd ≈ 2) and by far less on yardage. That is enough to create new recommended 1.83 legs on the "line side". That is the one classic edge the 10-29 work flagged to read forward, and it is unproven. Read secondary (i) before trusting those legs. Recommended priced alt legs should fall, because today's flattening makes longshots look likely.
- **10-04 owner rule.** With `m` taken as the pick'em main rung on rows that have no sportsbook, the DFS line decides where the model's median sits. That is a centre anchor in effect, and centre anchoring was KILLED on 10-29, partly because of the 10-04 rule. Two ways forward:
  - The owner confirms that anchoring the existing calibration at the line training already calibrates at is allowed.
  - Or ship with `m = Archive.get_line` (sportsbook only), with no consensus line meaning today's chain. That keeps about 87 % of the NFL gain and stays compliant on any reading.
- **Cells where it should not apply.** None is required by the evidence. Watch NHL hits (P2 +34; already a known low-read cell) and NFL fantasy points (+15 to +18 on the test half, −18 to −25 forward). If either reads KILL, take it out with the sportsbook-only `m` or a Rung C refit, not with a per-cell T.

## 8. Open questions for §8.2

1. Should a shape temperature `T_s`, or the Rung C map, absorb the "T > 1 at rungs" cells (NHL hits, fantasy points)? Fit it on real ladder rungs once the validation window lies inside the ladder era. The synthetic k·sd offsets are unreliable in low-count cells.
2. The money read: do 1.83 legs created where the DFS line sits off the consensus actually win (break-even .546)? This is the "line side" hypothesis, and the fix starts acting on it automatically.
3. Retiring `prob_recal_isotonic` and the NBA PRA Platt map: re-judge each as a separate read on P1. The evidence here says P1-neutral or better.
4. With the rung problem removed, should `_TEMPERATURE_BOUNDS` stay at (1, 10)? A T at the cap now only means "no skill at the line". It might be cleaner to report that than to fit it.

## 9. Bibliography

| Source | Identifier | Used for |
|---|---|---|
| Cox, D. R. (1958). Two further applications of a model for binary regression. *Biometrika* 45(3–4) | doi:10.1093/biomet/45.3-4.562 | 1/T as a calibration slope |
| Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. (2017). On calibration of modern neural networks. ICML | arXiv:1706.04599 | Temperature scaling |
| Gneiting, T., Balabdaoui, F., Raftery, A. E. (2007). Probabilistic forecasts, calibration and sharpness. *JRSS-B* 69(2) | doi:10.1111/j.1467-9868.2007.00587.x | Threshold calibration of predictive CDFs |
| Sahoo, R., Zhao, S., Chen, A., Ermon, S. (2021). Reliable decisions with threshold calibration. NeurIPS 34 | https://papers.nips.cc/paper/2021/hash/0e65972dce68dad4d52d063967f0a705-Abstract.html | Average calibration ≠ calibration at decision thresholds |
| Hébert-Johnson, U., Kim, M. P., Reingold, O., Rothblum, G. N. (2018). Multicalibration. ICML | arXiv:1711.08513 | Pooled calibration does not carry to a subgroup (alt rungs) |
| Matheson, J. E., Winkler, R. L. (1976). Scoring rules for continuous probability distributions. *Management Science* 22(10) | doi:10.1287/mnsc.22.10.1087 | CRPS as integrated Brier over thresholds |
| Gneiting, T., Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA* 102(477) | doi:10.1198/016214506000001437 | Brier and CRPS propriety |
| Gneiting, T., Ranjan, R. (2011). Comparing density forecasts using threshold- and quantile-weighted scoring rules. *JBES* 29(3) | doi:10.1198/jbes.2010.08110 | Threshold-weighted CRPS (design e) |
| Gneiting, T., Raftery, A. E., Westveld, A. H., Goldman, T. (2005). Calibrated probabilistic forecasting using EMOS and minimum CRPS estimation. *Monthly Weather Review* 133(5) | doi:10.1175/MWR2904.1 | Location and scale recalibrated separately over all thresholds |
| Kuleshov, V., Fenner, N., Ermon, S. (2018). Accurate uncertainties for deep learning using calibrated regression. ICML | arXiv:1807.00263 | Whole-CDF (Rung C) recalibration as the home for shape fixes |
| McCullagh, P. (1980). Regression models for ordinal data. *JRSS-B* 42(2) | doi:10.1111/j.2517-6161.1980.tb01109.x | Cumulative-link structure of the recommended formula |
| Allen, S., Ginsbourger, D., Ziegel, J. (2023). Evaluating forecasts for high-impact events using transformed kernel scores | arXiv:2202.12732 | Weighted scores for a region of thresholds |
| Smith, J. E., Winkler, R. L. (2006). The optimizer's curse. *Management Science* 52(3) | doi:10.1287/mnsc.1050.0451 | Selection on main-line disagreement (money side) |
| In-repo: docs/archive/researcher_temperature_ridge.md (partc); researcher_market_centre_anchor.md; memory temperature_cap_flattens_rungs, prob_recal_isotonic_rung_mispricing, dfs_main_line_eval_only | — | Prior verdicts, owner rules, β at market |
