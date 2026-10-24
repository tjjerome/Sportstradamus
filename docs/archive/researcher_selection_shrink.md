# In-repo research brief: selection-conditional shrink of served probabilities toward the sportsbook consensus (winner's-curse correction)

**Question:** should the served probability of a recommended leg be shrunk toward the sportsbook consensus by a factor fitted on the realized selected subset, and if so at what grain, with which estimator, and where (before or after argmax)? Options compared: (a) status quo with the 2.5x cap, (b) a higher floor, (c) `kelly_shrinkage` as lambda, (d) unconditional `prob_recal_book_citl`, (e) argmax-EV (design note only).
**Date:** 2026-10-03 · branch `devel` · read-only. No `reflect`, no DuckDB, nothing written under `src/`. Scripts and pickles are in `/tmp/claude-1000/-home-trevor-Sportstradamus/7c225c1a-7c1e-4cde-bede-cf7b5fe0cce8/scratchpad/research/` (`base.py` loader, `walk.py` walk-forward, `boot.py` bootstrap, `regress.py`/`mech.py` mechanism, `oracle.py`, `stack.py`, `scores.py`).

**VERDICT: KILL.** Do not shrink served probabilities toward the book at any grain (global, league, cell), with either estimator (scalar lambda or logistic on the logit difference), or at either placement (before or after argmax). Keep (a). No option clears the acceptance bar out of sample, and a hindsight oracle fitted on the evaluation window itself does not clear it either.

---

## TL;DR

- **Every option fails out of sample.** Primary split (fit < 2026-09-01, eval 2026-09-01..10-02, cohort re-derived after the shrink): the best shrink is per-cell lambda with an evidence floor, **+1.7 pp ROI** (day-clustered 95% CI [-0.3, +3.4]); global lambda = 0.14 gives **+1.3 pp** [-1.1, +3.2]. The Over hit rate among survivors is at most 0.475, against a required 0.50 and +3 pp. A hindsight grid over global (lambda_quoted, lambda_unquoted) and caps of {2.5, 2.25, 2.0, 1.9}, all fitted on the eval window, also passes nothing. Its best cell is +2.9 pp with Over hit 0.501, at cap 2.0.
- **There is nothing for a shrink factor to tune.** In the post-fix era the served `Win Prob` carries essentially no information beyond the market on the legs the rule selects. lambda_sel = **-0.10 ± 0.10** on eval (0.006 on the clean-era fit), and the logistic slope is gamma = **0.01 ± 0.15**. Realized hit rates track the book in every bin of model-book disagreement. An honest stack anchored on the market (book, model and payout price, fitted on all rows) recommends **45 legs where the status quo recommends 3,469**.
- **64% of recommended legs have no sportsbook quote.** For those legs `Market Prob` is the payout-implied fill, so a book shrink cannot reach them. They hit **0.466 (Over) and 0.514 (Under) against a 0.610 read**, so most of the cohort's overconfidence is out of the lever's reach.
- **The prescribed split straddles the book-leg fix.** The empirical era boundary is 2026-08-31 (`6c21665c`, `8bebd0e3`): the daily share of NaN `Market Projection` rows is 0–1% through 08-30 and 74% on 08-31. Before the fix the "book" includes platform self-quotes and flat-0.5 fills. Pre-fix lambda_all is 0.39 and post-fix it is 0.16, so any lambda fitted before September was fitted against a book that no longer exists.
- **Mechanism.** MLB is pure selection on noise: its unconditional calibration holds, and on selected quoted legs the hit rate equals the book. NFL is selection plus a model bias: unconditional Under overconfidence is **+3.6 pp on 10,004 legs**, and the disagreement with the book is anti-informative (lambda_all **-0.14 ± 0.03**). NFL routes to training, not to a serving shrink.

---

## Key Findings

### 1. Which rows carry a real book quote (binding for any book-anchored lever)

**From 2026-08-31 on:**
- **Quoted rows.** `Market Projection` notna means the model path found a servable quote: a `book_direct` same-line cohort with at least one non-DFS sportsbook, or a combo component sum (`prediction/book_quotes.py:_has_serving_support`, `_MIN_FALLBACK_REAL_BOOKS = 1`). On these rows `Market Prob` is that quote decoded at the **offered** line through the cell's fitted book shape (`offer_records.book_over_prob`), flipped for Under. Rows with `Model Version == "book_fallback"` also rest on a servable quote, with `Win Prob ≡ Market Prob`.
- **Unquoted rows.** NaN `Market Projection` rows carry the fill `dfs_boost_probs(Boost_Over, Boost_Under)` (`finalize_records`, offer_records.py:256–266): a proportional devig when both sides are posted, raw 1/π and its complement when only one is. They survive only if `|Win Prob − Market Prob| ≤ UNQUOTED_BOOK_DISAGREEMENT_MAX = 0.15`.

**Before 2026-08-31, nothing certifies a quote.** `Market Projection` is non-null on 99.6–100% of rows. The pre-`8bebd0e3` book leg fell back through three sources in order:
1. the DFS-inclusive `archive.get_ev` composite;
2. the combo markets;
3. the platform's own payout self-quote.

On top of that, the pre-`6c21665c` NaN fill was a flat 0.5 (`_BOOK_PRIOR_PROB`), and the fallback path averaged means across lines. That last one is the poisoning: in the fit window, 4,266 recommended fallback Unders claimed a book of 0.673 and hit 0.496. **No history column separates real quotes from self-quotes in that era.**

**Caveat even on real-quote rows.** At a DFS rung line, `Market Prob` is an extrapolation of the book's quote at its own line, not a book price at that line. The eval shows the cost: on quoted "book-edge" Unders (payout about 1.47) our decode read 0.790 and the legs hit 0.672. The platform's own price (implied about 0.68) was right.

### 2. Selection mechanism: optimizer's curse on a forecast that is calibrated in the large but not conditionally

A forecast that is calibrated conditional on its own value (E[y | m, π] = m) does not suffer when a rule selects on m·π. The curse appears exactly where the reliability curve is flatter than 45° inside the selected region. This is the gap between an **unbiased** estimate and a **posterior-mean** (calibrated) estimate, and the Bayesian posterior cures it (Smith & Winkler 2006, DOI 10.1287/mnsc.1050.0451; Capen, Clapp & Campbell 1971, SPE-2993-PA; Thaler 1988, DOI 10.1257/jep.2.1.191; Efron 2011, DOI 10.1198/jasa.2011.tm11181; van Zwet & Cator 2021, DOI 10.1111/stan.12241).

Calibration in the large is exact here. On eval bettable quoted rows the model reads 0.591/0.644 (Over/Under) against hits of 0.595/0.636. But the information share collapses in the tail.

**Write D = Win Prob − Market Prob.** lambda is the through-origin LS slope of (y − b) on D, where b is the book probability (`Market Prob`). The SE is day-clustered.

| window (book regime) | lambda_all quoted | lambda_sel quoted | logistic on selected: alpha / gamma |
|---|---|---|---|
| fit 07-04..08-31 (pre-fix book) | 0.390 ± 0.042 | 0.140 ± 0.042 | −0.505 ± 0.055 / 0.603 ± 0.073 |
| eval 09-01..10-02 (post-fix) | 0.161 ± 0.061 | **−0.100 ± 0.102** | −0.087 ± 0.128 / **0.010 ± 0.151** |
| unquoted eval, vs payout-implied price | 0.213 ± 0.072 | 0.044 ± 0.121 | — |

Binned by D on eval quoted rows, the hit rate follows the book, never the model:

| D bin | hit | book | model |
|---|---|---|---|
| D > 0.2 | 0.366 | 0.403 | 0.661 |
| (0.15, 0.2] | 0.460 | 0.456 | 0.626 |
| (0.1, 0.15] | 0.516 | 0.506 | 0.626 |
| D ≤ −0.1 | 0.727 | 0.743 | 0.586 |

**Selection on X is legitimate, but the linear form is misspecified.** Selection is on predictors, not outcomes, so a well-specified E[y | m, b] fitted on all rows is unaffected by selection (Dawid 1994, IMS LNMS 24:211–220; Senn 2008, DOI 10.1198/000313008X331530). Here lambda_sel < lambda_all in both regimes, so the information share decays in the tail. A fit on the selected subset targets the right region, but it returns lambda ≈ 0. At that value the "shrink" is just "replace the model with the market".

**Proper scores agree.** On the clean out-of-sample window (09-16..10-02), the book beats the served model on **all** 29,919 quoted rows: log-loss 0.6350 vs 0.6414, Brier 0.2224 vs 0.2254. The payout-implied price beats it on all 38,416 unquoted rows: 0.6385 vs 0.6429 (Gneiting & Raftery 2007, DOI 10.1198/016214506000001437). This is the adverse-selection setting. The data is consistent with the platforms pricing rungs off sportsbook feeds and quoting the bigger payout on the side they think is less likely; that is an inference, not something observed (Glosten & Milgrom 1985, DOI 10.1016/0304-405X(85)90044-3; Levitt 2004, DOI 10.1111/j.1468-0297.2004.00207.x).

The 2.0–2.5x payout band is the curse at its maximum. The claimed edge is noise multiplied by π, and recommended Overs there hit 0.361–0.378 against a 0.54 read.

### 3. Whether the 30-day numbers need a model-bias component: MLB no, NFL yes

The table covers recommended legs on or after 2026-09-04. "Bettable pred−hit" is unconditional; "quoted sel hit/book" is the selected legs with a real quote.

| league / side | n | hit | model | book | bettable pred−hit | quoted sel hit / book |
|---|---|---|---|---|---|---|
| MLB Over | 1023 | 0.437 | 0.597 | 0.474 | +0.003 | 0.463 / 0.457 |
| MLB Under | 2181 | 0.518 | 0.615 | 0.510 | +0.006 | 0.556 / 0.550 |
| NFL Over | 902 | 0.447 | 0.623 | 0.495 | +0.015 | **0.331 / 0.448** |
| NFL Under | 1720 | 0.458 | 0.601 | 0.466 | **+0.036** (n=10,004) | 0.426 / 0.457 |

- **MLB is selection alone.** It is calibrated unconditionally, and on quoted selected legs the hit rate equals the book.
- **NFL needs a model-bias component.** The unconditional Under overconfidence is about 7 SE. The model's disagreement with the book is anti-informative, with lambda_all −0.14 ± 0.03 and −0.27 ± 0.13 on selected legs.
- **The NFL book anchor also misses on the selected Overs.** The decoded book reads 0.448 and the legs hit 0.331 (n=266), so selection is also picking rows where the anchor is wrong. Late role or injury news is the usual suspect; this is unverified.
- **The pooled 30-day cohort mixes both regimes.** By leg count it is 53% MLB (selection alone) and 44% NFL (selection plus bias).
- **The 90-day `book_rate` mixes two book definitions.** For example, Under reads 0.586 against a 0.504 hit. That is the pre-fix poisoning, not a post-fix book failure: on the eval window the recommended Under book is 0.495 against a 0.501 hit.

### 4. Walk-forward result: no estimator, grain, or anchor passes

The full table is below. Four structural reasons:
1. The shrink can only touch the 36% of the cohort with a real quote.
2. The eval-optimal lambda for quoted rows is 0. A quoted-only sweep is monotone, from +1.5 pp at lambda = 0 down to 0 at lambda = 1, with Over hit never above 0.470.
3. Survivors stay mostly unquoted. For example, survivors of global lambda (S1) still read 0.611 on Overs while hitting 0.469.
4. A higher floor makes things **worse** (−1.6 to −10.5 pp). The claimed edge is anti-correlated with the outcome because high edge means a high payout, which means the platform priced the side as unlikely.

**Clean-era robustness split** (fit 08-31..09-15, eval 09-16..10-02, one book regime): lambda_sel = 0.006, the shrinks give +0.3 to +2.1 pp, and nothing passes. A weekly rolling refit clips lambda to 0 every week, with mixed results (−1.9, +3.7 and −6.3 pp).

### 5. `kelly_shrinkage` is in the wrong units to serve as lambda

Under the linear pool p = b + lambda·D, the Brier-optimal lambda is lambda* = E[D(y−b)]/E[D²]. The skill score of the served p against the book then satisfies:

`BSS = (2·lambda* − 1)·E[D²] / E[(y−b)²]`

It verifies exactly on eval quoted rows: lambda* = 0.161 gives BSS = −0.0097 by both routes. So **a positive `kelly_shrinkage` = clip(BSS) certifies lambda* > 0.5**. A cell with BSS of 0.01 and a small D can have lambda* ≈ 1, and using BSS as lambda would shrink it to 0.01. Option (c) lands near the eval-optimal lambda ≈ 0 by accident, not by design: +0.7 to +1.6 pp.

Kelly's anchor is also 1/payout, not the book (`strategies/kelly.anchored_win_prob`). Shrinking toward 1/payout by a factor s is algebraically the same as raising the edge floor to 0.05/s, which is option (b), and (b) hurts.

### 6. Placement: post-argmax if this is ever built

The linear shrink commutes with the complement, so pre-argmax and post-argmax agree on every row whose side does not flip. At lambda = 0.14, pre-argmax flips the side on **4,384 of 51,747** eval quoted rows (8.5%) but adds only **2** recommendable legs. Those 2 assume the standard Underdog 1.78x payout on the other side, since history does not carry both payouts.

The flips would move the inputs to Gate 2. `precision_over_live` and `precision_under_live` are computed on all posted `Bet` rows (`nightly._side_precision`), so they change. Post-argmax changes neither `Bet` nor Gate 2.

---

## Walk-forward table: primary split (OOS = eval ≥ 2026-09-01; IS = fit < 2026-09-01)

**How to read it:**
- "pred" is the mean **shrunk** probability of the survivors.
- "O-count" is recommended Overs as a share of (a)'s 1,995.
- "Pass" means all four criteria hold: ΔROI ≥ +3 pp, O-count ≥ 0.60, Over hit ≥ 0.50, and Under ROI not worse than (a).
- Shrinks touch quoted rows only, and fallback rows are unchanged, unless the row says otherwise.

| Option | OOS n O/U | OOS hit O/U | OOS pred O/U | OOS ROI O/U | OOS ROI | ΔROI pp [95% CI] | O-count | Pass | IS n | IS hit O/U | IS ROI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) status quo, cap 2.5 | 1995 / 4120 | 0.456 / 0.501 | 0.612 / 0.611 | −17.0% / −9.2% | −11.7% | 0 | 1.00 | — | 10616 | 0.490 / 0.508 | −9.8% |
| (b) floor 0.10 | 1036 / 1822 | 0.429 / 0.476 | 0.621 / 0.609 | −19.7% / −9.7% | −13.3% | −1.6 [−4.7, +1.3] | 0.52 | no | 8099 | 0.488 / 0.515 | −8.5% |
| (b) floor 0.15 (= IS-chosen) | 439 / 652 | 0.442 / 0.434 | 0.628 / 0.615 | −13.6% / −13.5% | −13.5% | −1.8 [−9.3, +5.7] | 0.22 | no | 6323 | 0.490 / 0.517 | −8.1% |
| (b) floor 0.25 | 138 / 166 | 0.399 / 0.355 | 0.647 / 0.669 | −16.0% / −27.4% | −22.2% | −10.5 [−18.9, +0.4] | 0.07 | no | 3616 | 0.485 / 0.502 | −11.3% |
| (c) `kelly_shrinkage`, NaN → 1 | 1430 / 2964 | 0.463 / 0.524 | 0.611 / 0.627 | −16.4% / −8.4% | −11.0% | +0.7 [−1.8, +2.8] | 0.72 | no | 10775 | 0.483 / 0.518 | −10.8% |
| (c') `kelly_shrinkage`, NaN → 0.01 | 1372 / 2919 | 0.469 / 0.527 | 0.612 / 0.627 | −15.5% / −7.9% | −10.3% | +1.4 [−1.2, +3.3] | 0.69 | no | 7384 | 0.444 / 0.517 | −11.0% |
| (d) book CITL per cell (fit-window intercepts; flipped sides dropped) | 1710 / 3510 | 0.460 / 0.509 | 0.621 / 0.622 | −17.5% / −9.1% | −11.8% | −0.1 [−1.7, +1.2] | 0.86 | no | 9573 | 0.490 / 0.517 | −9.5% |
| (d') book CITL (eval-window intercepts, outcome-free) | 1702 / 3977 | 0.464 / 0.537 | 0.636 / 0.633 | −17.5% / −6.3% | −9.7% | +2.0 [−0.4, +4.0] | 0.85 | no | 11241 | 0.494 / 0.518 | −9.6% |
| S1 lambda global, selected fit (0.140) | 1376 / 2923 | 0.469 / 0.525 | 0.611 / 0.625 | −15.4% / −8.0% | −10.4% | +1.3 [−1.1, +3.2] | 0.69 | no | 7315 | 0.433 / 0.519 | −10.8% |
| S1' lambda global, all quoted fit (0.390) | 1431 / 2961 | 0.465 / 0.515 | 0.609 / 0.616 | −15.8% / −8.5% | −10.9% | +0.9 [−1.3, +2.4] | 0.72 | no | 8350 | 0.472 / 0.510 | −10.9% |
| S2 lambda per league (MLB 0.137, WNBA 0.163; NFL has no fit rows → global) | 1376 / 2924 | 0.469 / 0.525 | 0.611 / 0.625 | −15.4% / −8.0% | −10.4% | +1.3 [−1.1, +3.2] | 0.69 | no | 7316 | 0.433 / 0.519 | −10.8% |
| S3 logit offset, selected (alpha −0.505, gamma 0.603) | 1367 / 2653 | 0.464 / 0.513 | 0.611 / 0.613 | −16.4% / −8.5% | −11.1% | +0.6 [−2.3, +2.7] | 0.69 | no | 4919 | 0.464 / 0.500 | −9.6% |
| S3' logit, side alpha (O −0.658, U −0.446, gamma 0.706) | 1366 / 2661 | 0.464 / 0.512 | 0.611 / 0.613 | −16.3% / −8.6% | −11.2% | +0.5 [−2.4, +2.7] | 0.68 | no | 4946 | 0.469 / 0.500 | −9.4% |
| S3'' logit offset, all quoted (alpha −0.001, gamma 0.406) | 1446 / 2995 | 0.469 / 0.516 | 0.609 / 0.617 | −15.1% / −8.5% | −10.7% | +1.0 [−1.0, +2.5] | 0.72 | no | 8667 | 0.469 / 0.512 | −11.1% |
| S4 lambda per cell (≥ 150 selected fit rows, else league, else global; 11 cells) | 1580 / 2966 | 0.475 / 0.524 | 0.617 / 0.621 | −14.7% / −7.6% | −10.0% | +1.7 [−0.3, +3.4] | 0.79 | no | 6925 | 0.445 / 0.516 | −10.0% |
| ref: lambda = 0 on quoted (book only) | 1372 / 2978 | 0.469 / 0.533 | 0.612 / 0.633 | −15.5% / −7.8% | −10.2% | +1.5 [−1.2, +3.5] | 0.69 | no | 7552 | 0.443 / 0.522 | −11.0% |
| S1 + no recommendation without a real quote | 50 / 334 | 0.560 / 0.608 | 0.640 / 0.733 | +1.2% / −7.7% | −6.5% | +5.2 [−13.1, +23.3] | **0.03** | no | 7287 | 0.432 / 0.520 | −10.7% |
| (a) + no recommendation without a real quote | 669 / 1531 | 0.435 / 0.479 | 0.615 / 0.609 | −18.7% / −11.0% | −13.4% | −1.6 [−6.4, +3.8] | 0.34 | no | 10588 | 0.490 / 0.508 | −9.7% |
| (b2) cap 2.0 (not a shrink) | 1361 / 3096 | 0.498 / 0.535 | 0.647 / 0.635 | −14.8% / −7.1% | −9.5% | +2.3 [+0.3, +3.8] | 0.68 | no (Over hit) | 9101 | 0.494 / 0.520 | −10.6% |
| S5 market-anchored stack (book + model + 1/π, all rows) | 4285 / 3688 | 0.600 / 0.627 | 0.767 / — | −15.5% / −16.5% | −16.0% | −4.3 | — | no | 4499 | 0.502 / 0.496 | −9.5% |

The S5 primary result is garbage: its unquoted arm was fitted on 570 pre-fix rows. In the clean split S5 recommends **45 legs, against (a)'s 3,469** (ROI −8.9%).

**Overfit and era are visible in the in-sample column.** S1, S4 and lambda = 0 all have *worse* in-sample ROI than (a) (−10.8, −10.0 and −11.0% against −9.8%). Shrinking toward the poisoned pre-fix book hurt, while the same move post-fix helps a little. Cap 2.0 is the reverse: −0.8 pp in sample, +2.3 pp out of sample (+3.3 pp [+1.2, +5.4] in the clean split, Over hit 0.485). It is the most consistent single lever, but it is still a fail.

**Two readings of "bettable Over count".** I read it as recommended Overs after the shrink relative to (a). The literal ledger meaning (all posted Over offers) cannot change under a post-argmax shrink, which would make that criterion vacuous. Under that vacuous reading, "S1 + no recommendation without a real quote" would pass the other three criteria. It does so on 384 legs (6% of (a)) with ΔROI CI [−13, +23] pp, so the evidence cannot resolve it either way. It is not a defensible GO.

---

## Recommendation / routing protocol

1. **KILL the selection-conditional shrink: no serving-time formula ships.** Grain, estimator and placement are all moot, because the out-of-sample value is lambda ≈ 0. A future build only makes sense if the lever is reopened. If it is, use this form:
   - **Formula:** after the `_MAX_CONFIDENCE` clip and before `Model EV`/`Kelly` in `finalize_records`, on rows with a real quote: `Win Prob' = Market Prob + λ_L · (Win Prob − Market Prob)`.
   - **Grain:** λ_L per league, keyed on `Model Version`.
   - **Fit:** through-origin LS on the trailing post-fix recommended quoted legs.
   - **Evidence floor:** at least 1,000 selected quoted legs, and a day-clustered 95% CI with a lower bound above 0 (otherwise λ = 0, never 1).
   - **Placement:** post-argmax only (Finding 6).
2. **(b) Higher floor: KILL.** It is out-of-sample negative at every level, because claimed edge is anti-correlated with realized outcome.
3. **(c) `kelly_shrinkage` as lambda: KILL.** It is in the wrong units (Finding 5). Leave it where it is, as Kelly's stake shrink toward 1/payout. That already sizes these legs near zero, and it is the real bankroll safety valve.
4. **(d) Unconditional `prob_recal_book_citl` on every cell: KILL as a selection fix.** Out of sample, fit-window intercepts give −0.1 pp (primary) and +2.8 pp (clean); eval-window intercepts give +2.0 / +0.1 pp. The size and sign of the gain are unstable. The lean is not the problem; the lack of tail information is. It remains a per-cell board slug for Gate-1 lean cells (`[[g1_citl_lean_book_anchored_posthoc]]`).
5. **(e) Argmax-EV side selection, design note: do not build on `Win Prob`.**
   - Doubling the candidate set raises the optimizer's curse (Smith & Winkler 2006), and the bigger payout is the side the platform prices as less likely.
   - Run it only on an honest, market-anchored probability. In the clean era that probability recommends about 1% of today's legs. That makes (e) a book-versus-platform price comparator (Kaunitz, Zhong & Kreiner 2017, arXiv:1710.02824), not a model lever.
   - Backtest it with this exact walk-forward once `Payout Over`/`Payout Under` have 30 days of history. They are absent from the dev-box history today (commit `ad599623` landed 2026-10-03).
6. **Route the NFL model bias to training.** The unconditional Under overconfidence (+3.6 pp) and the negative information share belong to the NFL lanes (§6.9 NFL routing, `nfl-ship15-recovery.md`), not a serving patch.
7. **Stacking with `model_weight`.** The served `Win Prob` already carries the training blend: `fused_loc` at weight w, fitted by CRPS/NLL/1-SE on all authentic validation rows. A lambda fitted on the served p is the residual trust after that blend, so it is not a double count if three conditions hold:
   - fit it on served p, never on pre-blend model p;
   - key it to `Model Version` and refit after any retrain that moves w;
   - never import a quantity that already contains w (`kelly_shrinkage` is BSS of the fused predictive).

   Measured lambda_sel does not track w: for w in (0.3, 0.6] it is −0.01 ± 0.26; in (0.6, 0.85] it is −0.14 ± 0.11; in (0.85, 0.99] it is 0.54 ± 0.28 (n=186). w is a population proper-score weight and does not transfer to the selected tail, so it must not set lambda.

## Reality checks

- **Effect size and regime.** The ceiling for a *global* shrink toward the book on this data is the lambda = 0 quoted result, +1.5 pp. Choosing lambda per cell reaches +1.7 pp. Both are about half the bar, with CIs that include 0. It holds for the post-fix book on 31 eval days (MLB-heavy, early NFL, WNBA playoffs). NBA is absent; the October season will differ, so treat it as a bet that this transfers.
- **What could make the KILL wrong.**
  - The NFL anti-information may be a transient of the 2026 feature-pipeline breaks (`[[nfl_line_matchups_leak_and_fp_2026_cliff]]`, `[[nfl_rebuild_player_membership_shift]]`). After the NFL retrains, lambda_sel could turn positive.
  - The eval window is short and seasonal.
  - The decoded book at rung lines is itself imperfect (Finding 1). A better anchor (sportsbook alt-line quotes at the rung) could make a shrink worth more than this data shows.
- **Build cost if reopened.** This is an engineering project, not research. It needs one block in `finalize_records`, a nightly lambda table written by `reflect`, and a golden test. It cannot be validated until about 60 days of single-regime post-fix history exist (around 2026-11-01).

## What to monitor live (realized_by_side, nightly)

- **Add a `quote` split** (quoted / unquoted / fallback) to `realized._offers_by_split`. Today `book_rate` pools sportsbook and payout-implied anchors, which is how the 90-day "book" ended up at 0.586 against 0.504 realized.
- **Track the selection residual.** For recommended legs by league × side, track `hit_rate − book_rate` and the trailing-30-day lambda_sel with its day-clustered CI. Reopen this lever only when the CI lower bound is above 0.
- **Cover what Gate 2 cannot see.** Gate 2's `precision_*_live` uses all posted `Bet` rows (about 0.59/0.64), not the recommended cohort, so it cannot see this failure. The realized ledger is the only monitor for it.
- **Pre-register the payout-band read for the cap (owner packet, not a session flip).** Proposed trigger: two consecutive monthly 30-day reads in which the recommended 2.0–2.5 band's ROI sits at least 5 pp below the 1.5–2.0 band on **both** sides. The current 30-day gap is 6.6 pp on Over and 9.0 pp on Under. The 90-day gap is 5.1 pp on Over and 0.2 pp on Under.

## Open questions / caveats

1. **Product framing (owner).** Honest, market-anchored probabilities recommend about 1% of today's legs. The acceptance bar keeps at least 60% of Overs with a hit rate of at least 0.50, which presumes there is edge to keep; this data says there is none on the selected subset. Should the `recommended` label and the story menu floor (`RECOMMENDED_EDGE_MIN = 0.05`) keep firing on unproven claimed edge?
2. **Unquoted legs (64% of the cohort) get full trust (lambda = 1).** This is `[[no-evidence-full-trust-pattern]]`. The data says the payout-implied price is sharper there, but simply requiring a quote makes ROI worse (−1.6 pp), because quoted disagreements are the most adversely selected. This is unresolved, and it is the larger lever by count.
3. **Is a sportsbook alt-line quote at the rung available in the archive?** If so, it would be a better `Market Prob` for rung payouts than the shape decode. That is a §6.11 (WS-4) question.
4. **Re-run this walk-forward around 2026-11-01** within one book regime, with NBA in sample. NFL has no fit-window rows in the prescribed split, so per-league lambda for NFL was untestable here.
5. **Look-ahead in (c).** `model_stats.parquet` is the current file, not a point-in-time snapshot.

## Bibliography

| # | Source | Identifier |
|---|---|---|
| S1 | Smith, J. E., Winkler, R. L. (2006). The optimizer's curse: skepticism and postdecision surprise in decision analysis. *Management Science* 52(3), 311–322. | DOI 10.1287/mnsc.1050.0451 |
| S2 | Capen, E. C., Clapp, R. V., Campbell, W. M. (1971). Competitive bidding in high-risk situations. *Journal of Petroleum Technology* 23(6), 641–653. | SPE-2993-PA (DOI 10.2118/2993-PA) |
| S3 | Thaler, R. H. (1988). Anomalies: the winner's curse. *Journal of Economic Perspectives* 2(1), 191–202. | DOI 10.1257/jep.2.1.191 |
| S4 | Efron, B. (2011). Tweedie's formula and selection bias. *JASA* 106(496), 1602–1614. | DOI 10.1198/jasa.2011.tm11181 |
| S5 | Dawid, A. P. (1994). Selection paradoxes of Bayesian inference. *Multivariate Analysis and Its Applications*, IMS Lecture Notes–Monograph Series 24, 211–220. | IMS LNMS vol. 24 on Project Euclid (no DOI confirmed) |
| S6 | Senn, S. (2008). A note concerning a selection "paradox" of Dawid's. *The American Statistician* 62(3), 206–210. | DOI 10.1198/000313008X331530 |
| S7 | van Zwet, E. W., Cator, E. A. (2021). The significance filter, the winner's curse and the need to shrink. *Statistica Neerlandica* 75(4), 437–452. | DOI 10.1111/stan.12241; arXiv:2009.09440 |
| S8 | Andrews, I., Kitagawa, T., McCloskey, A. (2024). Inference on winners. *Quarterly Journal of Economics* 139(1), 305–358. | DOI 10.1093/qje/qjad043 |
| S9 | Glosten, L. R., Milgrom, P. R. (1985). Bid, ask and transaction prices in a specialist market with heterogeneously informed traders. *Journal of Financial Economics* 14(1), 71–100. | DOI 10.1016/0304-405X(85)90044-3 |
| S10 | Levitt, S. D. (2004). Why are gambling markets organised so differently from financial markets? *Economic Journal* 114(495), 223–246. | DOI 10.1111/j.1468-0297.2004.00207.x |
| S11 | Kaunitz, L., Zhong, S., Kreiner, J. (2017). Beating the bookies with their own numbers — and how the online sports betting market is rigged. | arXiv:1710.02824 |
| S12 | Hubáček, O., Šourek, G., Železný, F. (2019). Exploiting sports-betting market using machine learning. *International Journal of Forecasting* 35(2), 783–796. | DOI 10.1016/j.ijforecast.2019.01.001 |
| S13 | Walsh, C., Joshi, A. (2024). Machine learning for sports betting: should model selection be based on accuracy or calibration? *Machine Learning with Applications* 16, 100539. | arXiv:2303.06021 |
| S14 | Baker, R. D., McHale, I. G. (2013). Optimal betting under parameter uncertainty: improving the Kelly criterion. *Decision Analysis* 10(3), 189–199. | DOI 10.1287/deca.2013.0271 |
| S15 | Uhrín, M., Šourek, G., Hubáček, O., Železný, F. (2021). Optimal sports betting strategies in practice: an experimental review. *IMA Journal of Management Mathematics*. | DOI 10.1093/imaman/dpaa029; arXiv:2107.08827 |
| S16 | Cox, D. R. (1958). Two further applications of a model for binary regression. *Biometrika* 45(3–4), 562–565 (logistic calibration slope and intercept). | DOI 10.1093/biomet/45.3-4.562 |
| S17 | Van Calster, B. et al. (2019). Calibration: the Achilles heel of predictive analytics. *BMC Medicine* 17, 230. | DOI 10.1186/s12916-019-1466-7 |
| S18 | Gneiting, T., Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA* 102(477), 359–378. | DOI 10.1198/016214506000001437 |
| S19 | Dmochowski, J. P. (2026). The profit-bias identity in sports betting: bookmaker profit as the public's prediction error (preprint, not peer-reviewed: realized margin is indistinguishable from the hold on 1,139 MLB games). | arXiv:2609.06739 |

**In-repo prior art engaged:**
- Repo bibliography: [47] Efron–Morris; [56] Ranjan–Gneiting; [75] Gneiting–Balabdaoui–Raftery.
- Blend-weight brief: B9 Satopää et al. 2014 (logit combination), at `docs/archive/researcher_blend_weight_slug.md`.
- Memory notes: `[[g1_citl_lean_book_anchored_posthoc]]`, `[[no-evidence-full-trust-pattern]]`, `[[proxy_goodhart_under_search]]`, `[[blend_weight_book_riding_selection]]`, `[[gate1_pickem_cohort_bookless]]`, `[[dfs_pickem_lines_are_real_quotes]]`, `[[training-live-metric-mismatch]]`.
- Plan sections: §3.3 and §6.10 of `docs/handoffs/model_improvement_track.md`.

**Artifacts read:**
- `scratchpad/realized_by_side.parquet`, reproduced exactly from history: 90-day recommended Over 5,656 legs at 0.470/0.663; Under 11,237 legs at 0.504/0.642.
- `data/runtime/history.parquet`: 533,411 rows, of which 277,957 are settled, posted and deduplicated offers between 2026-07-04 and 10-02.
- `data/training/model_stats.parquet` (74 cells).
- Code: `realized.py`, `prediction/offer_records.py`, `prediction/book_quotes.py`, `prediction/model_prob.py`, `training/posthoc.py`, `strategies/kelly.py`, `nightly.py`, `training/graduation.py`.
- Commits `6c21665c`, `8bebd0e3`, `1d0d7ba9`, `ad599623`.
