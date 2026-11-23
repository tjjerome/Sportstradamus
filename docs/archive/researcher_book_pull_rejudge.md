# In-repo research brief P: re-judging the book pull and the 0.20 blend-weight floor on priced rows (live-gap-2 Part C, Wave 4), 2026-11-21

The owner's ruling for these two `gbdt_feature_study.md` §15 items: #0d yields, and each item is kept only if priced-row Brier improves and the forward realized replay is no worse. This brief says which read decides each item, on which dates, and how a KEEP would be served. It was produced read-only.

## TL;DR

- **4.1 book pull: GO for one offline read per league; NO-GO for any build before the read KEEPs.**
  - Candidate: the arm each league sends, plus P′ = M + λ_L(P − M) on rows with an authentic quote. λ_L is one number per league, fit on held-out rows before the 10-29 cut.
  - Judged by `ship priced`: p2_dbrier is primary (p1_dbrier below 20 rung dates). ROI per recommended leg can only kill.
  - Expected: NEUTRAL in NHL and likely in NFL; NBA is the plausible KEEP.
- **The read must use post-1h dumps (085d0bcc).** The pull's inputs all moved with the re-quote. The live λ\* on today's versions stays report-only: it falls below 20 scored dates before the imminent sends, and those pre-1h pickles are about to retire.
- **4.2 weight floor: NO-GO until NFL's rebuild with 1h quotes and Wave 3.**
  - If passing yards still fits w < .20, run the walk without the floor, ranked by priced Brier against the book instead of slack. Judge the pick on later dates against what production serves now.
  - Expected: NEUTRAL. Replace the floor with this comparison before the next NBA/NHL walk.
- **A KEEP serves** as the league's λ in a `stat_meta.json` knob, copied into each pickle at retrain. One shared function applies it in `pipeline.py`, `model_prob.py` and `tail_pricing.serve`, with a golden parity test. It is a learned market-anchored layer (#0d), admitted only by the owner's ruling.

| Item | Run | Primary | Decision | Order vs brief M |
|---|---|---|---|---|
| 4.1 | sent arm with and without the pull, forward rows (NFL: test half) | p2_dbrier, else p1_dbrier | `ship priced`; ROI kill-only | on top of what the league sends (Wave 3 candidate if M KEEPs) |
| 4.2 | floor-free walk, ranked by p1_dbrier vs book; pick vs `--baseline-book` on later dates | `ship priced`'s primary | same; KEEP → `shipped` flip | after NFL's post-1h + Wave 3 rebuild; moot if w ≥ .20 |

## Key findings

1. **The pull is a two-forecast combination with a closed-form optimum.**
   - λ\* = Σ(y−M)(P−M)/Σ(P−M)². This is the Bates–Granger weight under the Granger–Ramanathan constraint (weights sum to one, no intercept): doi:10.2307/3008764; doi:10.1002/for.3980030207.
   - With D = P − M and y − M = λ₀D + ε, serving λ̂ = λ₀ + e changes Brier by (e² − (1−λ₀)²)E[D²]. So the pull pays only where estimation error is smaller than the true pull.
   - Per-cell SE on own-line rows is .4–.7, so the fit uses one λ per league (Smith & Wallis 2009, doi:10.1111/j.1468-0084.2008.00541.x; Claeskens et al. 2016, doi:10.1016/j.ijforecast.2015.12.005; Diebold & Shin 2019, doi:10.1016/j.ijforecast.2018.09.006).
   - It is a linear pool of survival functions, so it stays monotone across rungs; λ = 1 is the identity.
   - Pools of calibrated forecasts are uncalibrated (Ranjan & Gneiting 2010, doi:10.1111/j.1467-9868.2009.00726.x), so λ is fit on outcomes.
2. **The §15 evidence was measured on priced legs, but on retired versions.**
   - October's pull cut quoted NFL+NHL Brier by 62.5 and 99 × 10⁻⁴; fold 2 ran backward in time.
   - It cut recommended legs 32–63% and ROI per leg by about 2 pp, with no interval (lead-time brief Findings 5–7).
   - λ moved from .011 to .290 across one version step, so it never transfers between versions.
3. **Since I6d, the disagreement sits at rungs and converted lines, where M is our own decode** (`book_over_prob`, `prediction/offer_records.py:119`; track §3.4 X3). Outcome-free counts, current versions through 11-15:

   | League | E[D²] quoted | Alt rungs | Converted main | Native main |
   |---|---|---|---|---|
   | NBA | .0056 | .0121 (76% of ΣD²) | .0028 | .0017 |
   | NFL | .0117 | .0171 (49%) | .0171 (43%) | .0024 |
   | NHL ex-goals | .0013 | .0014 | — | .0012 (74%) |

   Held-out rows at their own lines sit 5× (NFL) to 16× (NBA PRA) closer to the book, so P2 carries the power.
4. **The 1h re-quote moves the pull's inputs.**
   - Training now reads quotes at start − 1 h (`stats/game_starts.py`, `TRAINING_LOOKBACK`). It used to read them at 12:00 UTC: a median 24 h before an NFL kickoff, and before NBA's first poll (lead-time Finding 7).
   - These move: in-season NBA/NHL authenticity, w (`pipeline.py:3530–3569`), T (`:3232`), the book-CITL posthoc (`training/posthoc.py:128`), Gate 1, P1 membership, M = 1 − `Odds`, and the replay's book leg (`tail_pricing.py:313`).
   - These do not: P2 membership, the history's decision-time `Market Prob`, and outcomes.
   - Today's `model_stats` w and Gate 1 are pre-1h, and the read uses neither.
   - A fresher book earns more weight (lead-time Lever 3: NFL w −3–7%). That absorbs part of the gain, so the pull counts only as an increment.
5. **Live reads on today's versions cannot reach a verdict.**
   - Dev history (synced 11-16) holds NBA v20261029 and NHL v20261030 on 18 dates each. A 5-date burn-in plus 20 scored dates needs about 25, and the imminent sends end both stretches first (a send resets λ\*, live-gap-2 §6).
   - The cell mix also changed mid-stretch (NBA pulls 11-15/16; NHL rung leg from 11-11).
   - The owner's 10-30 reading of #0d bars live weeks as a primary gate (track §8.2 #0d).
6. **The floor stands in for "adds nothing over the book"; the priced rule measures that directly.**
   - The floor binds only in walks:
     - `confirm.py:87–92` reverts;
     - `confirm_nominees.py:181–185` drops corners (NaN and 1.0 exempt);
     - `confirm.py:138–150` waives S2/S3 because both models "sit on the book… ~0 by construction".
   - A corner at the .05 bound (`calibration.py:31`) ties the book, so `ship priced` returns NEUTRAL by construction. Yet the floor also vetoes .05–.20 corners that might win.
   - Without the floor, slack ranking (`confirm_nominees.py:88`) re-picks a bound corner.
   - Passing yards sits at the bound on 9 of 13 corners and has been held since 09-29 (`nfl-ship15-recovery.md` §5, §6 decision 5).

## Recommendation

### 4.1 pre-registration

- **Arms.**
  - Base: the arm the league sends (the Wave 3 candidate where M KEEPs, else the control), refit at the 10-29 cut on the post-1h matrix at served parameters.
  - Candidate: the base dumps with the pull applied. No refit, so refit noise cancels.
  - M: on P1, 1 − `Odds`. On a P2 rung, `book_over_prob` at that rung, with `Book_EV` as `Market Projection` (serving's `Market EV`, `model_prob.py:1194`).
  - Pickem and synthetic rows are unchanged and stay in the scored rows at dBrier 0; quoted-only means are reported alongside.
  - λ = 1 for:
    - NHL goals (constant live book leg; lead-time Finding 10.1);
    - NHL shots/assists (rung leg since 11-11), unless `price_dumps` replays that leg.
- **Fit.** λ_L is the closed-form λ\* on the base arm's held-out authentic P1 rows before the cut, pooled over the cells the league serves and clipped to [0, 1]. NFL is judged on its test half: expanding by date after a 10-date burn-in.
- **Statistics** (`ship priced`, `scripts/priced_score.py`): p2_dbrier and p1_dbrier, ×10⁻⁴, pushes dropped, 95% over 500 whole-date resamples.
  - Primary: p2_dbrier at ≥ 20 rung dates, else p1_dbrier.
  - p1_dmse ≡ 0 here. Using p1_dbrier in its place is the only deviation from `judge()` (`priced_score.py:215–223`).
  - NFL's test half has 22 rung dates, which is borderline: count them first.
- **Money (kill-only).** Rerun the `realized-replay --posted` selection (`realized.recommended`) under P′. Statistic: flat-stake ROI per recommended leg, candidate minus base, same intervals. Also report legs, hit, `hit_lb` and `gap_delta`.
- **Rule** (as `ship priced`):
  - KEEP: the primary lies wholly below 0 and nothing judged lies wholly worse.
  - KILL: any judged statistic lies wholly worse.
  - Otherwise NEUTRAL, which is also forced under 20 dates.
  - Futility: if the outcome-free MDE exceeds the full-pull gain E[D²], record NEUTRAL without reading outcomes.
- **Report-only:**
  - per-cell and logit λ (Satopää et al. 2014, doi:10.1016/j.ijforecast.2013.09.009);
  - encompassing β (Harvey et al. 1998, doi:10.1080/07350015.1998.10524759);
  - the line-class split;
  - the Murphy split (doi:10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2);
  - dBrier by week of version age.
- **Dates.**
  - The fit uses dates before the cut.
  - The forward check runs from the day after the matrix build (NBA 10-29, NHL 10-30) to the rebuilt matrix's newest settled date. These rows are replayed, so sends do not touch them.
  - The live λ\* (report-only) uses one served version: from its first serving date to the day before the league's next send, never across one.

### Serving a KEEP

- **Code.** Add `apply_book_pull(p_over, p_book, lam)` in `helpers/distributions.py` beside `apply_temperature` (`:917`): `p_book + lam·(p_over − p_book)` where `p_book` is finite, else `p_over`. Call it:
  - after the posthoc at `model_prob.py:1247`, with `Market EV`, on quoted rows;
  - in `pipeline.py` after temperature and posthoc, on authentic test rows, so the gates score what serves;
  - in `tail_pricing.serve` (`:161`) on its `authentic` mask, so `RECONSTRUCTION_TOL` holds.
- **State.**
  - A per-cell `stat_meta.json` knob `book_pull` holds the league's λ from that send's read. Absent means inert. It sits outside the signed identity, like `book_leg`.
  - The pipeline copies it into the pickle, and serving reads the pickle (as for T).
  - It is refit at every league send; a send without a read clears it.
  - Rejected: per-cell pickle fits. At SE .4–.7 they leave λ = 1 almost everywhere.
- **Parity.** A golden test: `model_prob`, `serve` and the pipeline dump agree on quoted, pickem and NHL-goals rows at λ = .5. Also add a book-at-rung P2 path to `ship priced`, which 4.2 needs too.
- **#0d.** This is a learned market-anchored layer, but a model-side, pre-argmax combination on every quoted leg. It is not the killed post-argmax selection shrink, trust layer or DFS-line anchor (track §3.4).
- **Cost.** About 1.5–2 dev-days plus a retrain per league. It cannot ride NBA's or NHL's imminent Part C sends, so it waits for their next ones. NFL's send can carry it only if the read and build land first.

### 4.2 pre-registration

- **When.**
  - After NFL's post-1h + Wave 3 rebuild. The re-quote cannot lift a weight at the bound; the plain mean (`mean_anchor`) may (study §4.4).
  - Receptions' recipe change leaves the floor alone: its w sits at the .90 cap.
  - If rebuilt passing yards fits w ≥ .20, judge it like any withheld cell (`ship priced --baseline-book`) and stop.
- **Walk.**
  - Run `ship sweep --confirm` without `--min-model-weight`.
  - Rank six-gate passers by p1_dbrier against the book on the earlier half of the test-half dates.
  - Judge the pick on the later half against `--baseline-book`: P = 1 − `Odds`, mean `Book_EV`, authentic rows. That is what `book_fallback_prob` serves now. Use the forward rows as a check.
  - Selecting and judging on the same rows is the optimizer's curse (Smith & Winkler 2006, doi:10.1287/mnsc.1050.0451).
  - Primary: `ship priced`'s p1_dmse, since `--baseline-book` has no P2 (`priced_score.py:292`). Money is checked as in 4.1.
- **Re-judge together**, one comparison each, at their league's post-1h rebuild:
  - **Held by the floor:** NFL passing yards and MLB runs allowed (withheld); NFL attempts (wave 1; served now at .389; `low_weight_models.md` 93–119).
  - **Six-gate corners under .20 that a floored walk drops:** MLB hits allowed, MLB batter strikeouts, NFL rushing yards, WNBA REB, NHL skater fantasy (`research/confirm_nominee_gates.csv`).
  - **Served under .20 by ordinary retrains:** NFL tds (.119) and MLB hits allowed (.104).
  - MLB and WNBA at their pre-season sends.
- **Rule change.** Retire the w-floor for the priced ranking before the next NBA/NHL walk. The re-quote gives formerly book-less cells (w = 1.0, exempt) fitted weights that can land under .20.

## Power (80%, two-sided 5%)

| Read | Size | MDE ×10⁻⁴ | Gain at λ₀ = .3 | Expect |
|---|---|---|---|---|
| NBA quoted legs | 300–770 legs/date, 20 dates | 20–35 | 27–40 | possible KEEP |
| NHL ex-goals | ~670 legs/date, 20 dates | 13 | 6 (max 13) | NEUTRAL |
| NFL test half, P1 | ~12k rows, 85–104 dates | 13 | 12 | borderline |
| Passing yards P1 | 425 rows, 85 dates | ≈100 | ≈30 | NEUTRAL |

Variance structure comes from pre-I6d legs at a fixed λ = .2 (DEFF 4.7 NBA, 4.1 NHL; seed 1729). Recompute the P2 MDE on post-1h dumps before reading.

## Reality checks

- **Size.** October's gain came at an NFL E[D²] of about .0135. A full pull now gains at most E[D²] (56 × 10⁻⁴ NBA, 13 × 10⁻⁴ NHL), before the re-quote absorbs part of it.
- **Money.** The pre-I6d pull cut legs and ROI per leg, so the likeliest KILL is the money check.
- **Version age.** The information share decays within one to two weeks (track #0d, exploratory), so a λ fixed at the send under-pulls late. If dBrier grows by week, refitting through newer rows (Wave 5) is the lever.
- **NBA fit rows** are mostly last April–June (PRA 702/214/28) plus about 8 in-season dates, so a playoff λ may not transfer.
- **Few clusters.** At 20–25 dates, pairs bootstraps under-cover (Cameron, Gelbach & Miller 2008, doi:10.1162/rest.90.3.414; MacKinnon & Webb 2017, doi:10.1002/jae.2508; Webb 2023, doi:10.1111/caje.12661). A narrow KEEP earns a second send only on later dates.
- **Spent windows.** Each read spends its window (Dwork et al. 2015, doi:10.1126/science.aaa9375). A pre-fit λ scored out of sample is the Giacomini–White method comparison (doi:10.1111/j.1468-0262.2006.00718.x; West 1996, doi:10.2307/2171956).
- **Gate bias.** Fitting λ on the sent pickle's test half biases the gates by about SE²·E[D²], which is negligible.

## Open questions (→ §8.2)

1. Does the pull's gain grow with version age? Then it is staleness, and Wave 5 owns it.
2. `ship priced` has no book price at rungs (for `--baseline-book` or the pull).
3. NHL goals' live book leg is constant; fix it before any read includes goals.
4. Does `price_dumps` replay the NHL shots/assists rung leg?

## What I checked

- Code at devel 71702fc1 (file:line as cited); `ship priced` and `ship_gate.md` "The priced-row verdict".
- Post-I6d live legs: outcome-free counts, E[D²] and line classes only.
- Pre-1h test halves (`partb/before_phase4/*/test_sets/`): authentic rows by month.
- Nothing was trained, swept or written.

## Bibliography

| Source | Identifier |
|---|---|
| Bates & Granger 1969, The combination of forecasts, Oper. Res. Q. 20(4) | doi:10.2307/3008764 |
| Granger & Ramanathan 1984, Improved methods of combining forecasts, J. Forecasting 3(2) | doi:10.1002/for.3980030207 |
| Harvey, Leybourne & Newbold 1998, Tests for forecast encompassing, JBES 16(2) | doi:10.1080/07350015.1998.10524759 |
| Clements & Harvey 2010, Forecast encompassing tests and probability forecasts, J. Appl. Econometrics 25(6) | doi:10.1002/jae.1097 |
| Ranjan & Gneiting 2010, Combining probability forecasts, JRSS-B 72(1) | doi:10.1111/j.1467-9868.2009.00726.x |
| Smith & Wallis 2009, A simple explanation of the forecast combination puzzle, OBES 71(3) | doi:10.1111/j.1468-0084.2008.00541.x |
| Claeskens, Magnus, Vasnev & Wang 2016, The forecast combination puzzle, IJF 32(3) | doi:10.1016/j.ijforecast.2015.12.005 |
| Diebold & Shin 2019, Partially-egalitarian LASSO for forecast combination, IJF 35(4) | doi:10.1016/j.ijforecast.2018.09.006 |
| Satopää et al. 2014, Combining multiple probability predictions using a simple logit model, IJF 30(2) | doi:10.1016/j.ijforecast.2013.09.009 |
| Murphy 1973, A new vector partition of the probability score, J. Appl. Meteor. 12(4) | doi:10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2 |
| Giacomini & White 2006, Tests of conditional predictive ability, Econometrica 74(6) | doi:10.1111/j.1468-0262.2006.00718.x |
| West 1996, Asymptotic inference about predictive ability, Econometrica 64(5) | doi:10.2307/2171956 |
| Cameron, Gelbach & Miller 2008, Bootstrap-based improvements for inference with clustered errors, REStat 90(3) | doi:10.1162/rest.90.3.414 |
| MacKinnon & Webb 2017, Wild bootstrap inference for wildly different cluster sizes, J. Appl. Econometrics 32(2) | doi:10.1002/jae.2508 |
| Webb 2023, Reworking wild bootstrap-based inference for clustered errors, Can. J. Econ. 56(3) | doi:10.1111/caje.12661 |
| Smith & Winkler 2006, The optimizer's curse, Management Science 52(3) | doi:10.1287/mnsc.1050.0451 |
| Dwork et al. 2015, The reusable holdout, Science 349(6248) | doi:10.1126/science.aaa9375 |
