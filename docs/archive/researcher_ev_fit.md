# In-repo research brief: the EV fit (estimator, "+EV" mark, stakes, side, scoreboard, line translation)

**Question.** How phase 1's fitted "EV" should be estimated, marked and staked under estimation error, which side a leg should take, what the model lane's scoreboard should publish, and how a sportsbook's price at one line should be moved to the app's line. **Written** 2026-11-25. **Scope.** Read-only. The analysis used the frozen history (games through 2026-11-24) and the archive opened `read_only=True`; nothing was written to either checkout. Scripts and outputs live in `~/backups/sportstradamus/2026-11-25-profit-engine/research/ev_fit/`, written `ev_fit/` below.

**Terms.**
- **App's fair price:** the two posted payouts de-vigged proportionally, pf = (1/po)/(1/po + 1/pu).
- **Kinds:** Underdog even (equal payouts), Underdog modified payout, and Sleeper.
- **Classes:** in class A a sportsbook prices the app's exact line. In class B books price only other lines, and in class C no book quotes the prop. A quote counts only if the scoring run could see it (finding 2) and it is at most 90 minutes old.
- **Model's side:** the side the served model favours, which is today's rule (`finalize_records`).
- **Fallback rows:** withheld cells whose `Win Prob` is the book consensus (`Model Version = book_fallback`).
- **dLL:** the app's log loss minus the fit's, in nats per leg × 10⁴; positive means the fit is better.
- **n0:** the prior's strength, in pseudo-legs.
- **Says / paid:** the mean fitted value against the mean realized payout per unit staked.

Every interval is 95 % and resamples whole game dates (2,000 draws, seed 1729). The walk-forward fits on dates before D and scores D. Its test dates run 2026-10-23 to 11-24 (33 dates, 117,289 offers in NBA, NFL and NHL).

## TL;DR

- **The replication is exact, but two of the audit's inputs were wrong.**
  - Every audit weight and band reproduces.
  - The audit's class-A book term includes a closing-quote batch that 397 of 426 scoring runs never saw. On quotes the run could see, NBA's raw-book weight is 0.53 (.32), not 0.80 (.28), and NBA's class shares are 21/27/52 %, not 33/39/28 %.
  - The served quote's median age of 79 minutes comes from cron timing, not from the 1-hour training quote.
- **Q1: GO, with a prior.**
  - The plan's fit as written (unpenalized, per stratum) loses to the app's price out of sample: dLL −2.6 [−12.0, +5.9]. Its 3,018 legs valued at 1 or more paid 0.863.
  - A Gaussian prior centred on the no-evidence point and worth n0 = 4,000 legs gives +5.1 [+1.3, +8.8]. It is no worse than the app in any league × kind.
  - The gains sit in NHL and in NBA modified-payout picks. NBA even, NBA Sleeper and NFL stay at the app's price.
- **Q2 and Q6: the over-claim at the top is the optimizer's curse, and "+EV" needs a lower bound.**
  - In a simulated world with no information, the plan's fit marks 986 false "+EV" legs over 33 dates. The prior cuts that to 7, and a one-sided 95 % lower bound cuts it to 0.
  - With the prior, the 49 legs at a point value of 1 or more paid only 0.787 [0.612, 0.953].
  - Mark "+EV" only when the lower bound clears 1. On this window no leg qualifies.
- **Q7: the side follows the model.** On the 28 % of offers where the fit's value side differs from the model's side, the fit's side returns 2.7 pp less a leg [−5.7, +0.3]. No margin, bar or bound on a flip recovers that loss. The bands the plan quotes were computed with the fit choosing the side.
- **Q4, Q3 and Q5:**
  - **Q4:** the served-P weight in class A is fine for prediction, but it is not "the model's weight". The scoreboard should publish the model term fitted on served rows only and judge changes as paired same-leg differences; the minimum detectable change today is 0.31.
  - **Q3:** stake half of Kelly evaluated at the lower bound. That is zero on this window.
  - **Q5:** translate with each book's own ladder slope, which is outcome-free. On class B it earns 0.26 against the engine's 0.07 in NBA, a paired gain of +0.20 (SE .11).

## 1. Answers, one per question

**Q1, the estimator. Verdict: GO, one penalized logistic fit per league × kind × class.** Fit P(Over) with one row per offer. Use the plan's terms plus one split: the model term is fitted on served rows and a separate term on fallback rows.

The prior is Gaussian at the no-evidence point (c = 0, a = 1, every w = 0) and is worth n0 = 4,000 legs. Re-choose n0 monthly from {2,000; 4,000; 8,000} by walk-forward log loss. Keep the window at 90 days, equally weighted, and keep the floor at 400 legs.

The evidence for each sub-question (tables E3 and E4):
- **(a) c and a.** A free slope a matters on modified-payout and Sleeper picks: fixing a = 1 scores −6.2. These picks show a favourite–longshot pattern: longshots priced at 3.2 or more hit 0.199 against the app's 0.219, and favourites at 1.3 or less hit 0.758 against 0.735 (Snowberg & Wolfers 2010). On even picks a is not identified, so the fit has only c there. The prior keeps both c and a near (0, 1) unless the data insist.
- **(b) One row per offer, not stacked sides.** This follows from algebra; I did not run it. Stacking duplicates each outcome. With c = 0 it returns the same point estimates and standard errors understated by √2. With c ≠ 0 the stacked rows cannot give P(Over) + P(Under) = 1 unless a side indicator is added, and that reproduces the single-row model.
- **(c) Class A's model term measured against the app or against the book.** The two tie (+4.9 vs +4.9). Choose "against the app": the weight then reads as what the served number adds beyond the fresh book at a fixed app price, and the zero point is the same in both.
- **(d) Pooling.** Pooling kinds hurts: +3.5 pooled against +4.9 separate, and −16.1 without the prior. Pooling across leagues was not tested, and the leagues differ in what they carry. The prior already does the useful pooling, toward the app's price.
- **(e) The penalty, the step that matters.** n0 = 250 to 16,000 traces +0.9, +2.4, +3.9, +4.9, +5.1, +4.7, +3.7. A nested walk-forward choice gives +4.6 [+0.8, +8.5] and lands on 4,000 to 8,000.
- **(f) Window and floor.** Half-lives of 14 and 30 days score +4.4 and +4.7 against +4.9 at equal weights. Floors of 200, 400 and 1,000 score +5.0, +4.9 and +5.2. With the prior in place the floor barely binds.
- **The fallback split.** It is neutral overall (−0.07 [−0.75, +0.69]) and helps NHL class B (+5.1 [+1.2, +10.0]). It is required for Q4.

**Q2, the over-claim at the top. Verdict: the cause is selection on estimation noise. Correct it with the prior plus a lower-bound "+EV" mark.**

The simulation redraws outcomes from known truths and reruns the same walk-forward:
- With outcomes drawn from the app's fair price (no information anywhere), the plan's unpenalized fit marks 986 legs "+EV" in 33 dates. They say 1.038 and are truly worth 0.893. That is the same signature as the real data, so coefficient noise under selection alone produces the plan's over-claim (Smith & Winkler 2006).
- The prior cuts the false marks to 27 (n0 = 2,000) or 7 (n0 = 4,000). A one-sided 95 % bound cuts them to 0.
- In a world as informative as the data allow, only 0.13 % of legs are truly +EV. There the n0 = 4,000 point rule marks 37 legs, worth 0.991 on average (38 % truly ≥ 1). A one-SE bound marks 2.3 legs, 75 % truly ≥ 1.

On real outcomes with the prior, the bands are honest up to a value of 0.97: 0.872 → 0.889, 0.916 → 0.915, 0.948 → 0.933, 0.980 → 0.948. The 49 legs at a point value of 1 or more pay 0.787, below even the simulation's 0.991. So a second, smaller cause sits on top of the noise. On offers where the fit and the model pick different sides, the fit over-states its side most where the model disagrees most with the app: 1.2 pp in the smallest quarter of disagreements, 2.5 pp in the largest. Clipping the model term did not repair this (+0.14 [−0.27, +0.54]).

**What "+EV" requires:** a one-sided 95 % lower bound on the value of at least 1, using the fitted logit's standard error. On this window that marks no leg, while a one-SE bound marks 1 to 3. The board therefore ranks by point EV ("best available"), and the mark stays rare until the evidence grows.

**Q6, acceptance and kill lines. Verdict: section 3, written before any code.** The recommended specification passes all six tests on the frozen data. The plan's specification fails AT1, AT2 and AT4.

**Q7, which side a leg takes. Verdict: the board's side follows the model (today's rule). The fit values that side and does not choose it.**
- The fit and the model agree on 71.4 % of offers.
- On the 33,494 offers where they disagree (n0 = 4,000), the fit's side returns −12.8 % and the model's side −10.1 %. The difference is −2.7 pp [−5.7, +0.3]: negative or flat in every league × kind, and credibly negative in NFL Sleeper (−9.0 pp [−15.9, −0.4]).
- At the top, the 1,290 flipped legs that the prototypes' fit valued at 1 or more returned −17.9 %. The same offers on the model's side returned −6.4 %.
- The fit over-states the chance of its own side on these offers by 1 to 3 pp in every league × kind.
- A margin does not help (table E8). A gap of 0.05 between the two sides' values flips 4,210 offers for +2.0 pp [−3.7, +8.2]. A value bar of 1.00 to 1.06, or a lower-bound flip, moves 0 to 8 offers. Taking the model's own value side (model P × payout) is clearly worse: −5.1 pp [−7.7, −2.6] on 39,405 flips.

**Q4, double counting and the scoreboard. Verdict: the fit is sound for prediction, but the class-A served-P weight is not the model's weight. Publish the model term on served rows, and judge every step on the same legs.**
- *Prediction.* Regressing on the served P and the raw book together is sound. The logistic re-weights what the two share, and replacing served P with `Standalone Prob` in class A changes nothing (−0.14 [−0.34, +0.06]).
- *Meaning.* In class A, w_model measures what the served number adds beyond the fresh book. That includes the blend's stale and translated copy of the book, so it is not the model's information. In NBA class A the served term is 0.20 (.17) and the standalone term 0.06 (.08).
- *Fallback rows.* About a third of rows are fallback rows whose "model" is the book (35 % of class A, 37 % of B, 29 % of C). Fit them in their own term.
- **Publish two statistics per league × kind × class** (and per market on request):
  1. the coefficient of the model term on served rows, with its date-clustered SE and its leg and date counts (Cameron & Miller 2015);
  2. its walk-forward log-loss gain, with a date-resampled interval.
- **Judge a candidate** (a model version, a translation) by the paired difference on the same legs, with a date-bootstrap SE. Two independent periods are the wrong comparison.
- **Detectable changes** (two-sided 5 %, power 80 %, that is 2.8 × SE): the NBA class-B weight is 0.36 over two independent 4-week periods and 0.25 over 8 weeks. Paired on today's 22 NBA dates it is 0.31.
- **The plan's kill line,** "within one SE of today's":
  - Read against independent periods, it keeps a step that does nothing 24 % of the time.
  - Read on the paired difference's own SE, that rate falls to 16 %, and a true 2-SE gain survives 84 % of the time.

**Q3, stakes. Verdict: stake half of Kelly evaluated at the lower-bound win chance. Cap exposure per leg across a slate's entries.**
- *Singles.* f = ½ · max(0, (p_lo·m − 1)/(m − 1)) with p_lo = sig(η̂ − 1.645·se). This is the robust Kelly stake for an interval of win chances: growth increases in p, so the worst case sits at p_lo (Sun & Boyd 2018). Halving guards against the misspecification that Q2 shows (Uhrín et al. 2021).
- *Power entries of k legs.* Use P_lo = Π sig(η̂_i − 1.645·se_i). Treating every leg as at its own bound is conservative and exact when the legs share one stratum's coefficients.
- *Legs shared across entries.* Cap the summed stake of all entries holding a leg at the largest stake any one of them would take alone. The exact alternative maximizes expected log wealth over the joint leg outcomes at p_lo (Whitrow 2007; Grant, Johnstone & Kwon 2008).
- *Why not shrinkage.* The Baker–McHale-style shrinkage, k = e²/(e² + var e), and quarter Kelly both staked the 49 point-EV legs, which returned −21.3 % [−38.8, −4.7]. The lower-bound rule staked nothing.
- *What the table must hold.* It needs each stratum's coefficient covariance, not only the standard errors: c and a are correlated, so the SEs alone cannot give se(η̂).

**Q5, translating a book's price to another line. Verdict, ranked:**
1. Interpolate the same book's two bracketing rungs on the logit scale wherever they exist.
2. Otherwise, shift the nearest quote by that market's slope in logit P per unit of line, estimated from archived adjacent rungs (outcome-free).
3. Do not use the model's own shape (step 2 of the plan's M2), and do not use today's `Market Prob`.

The supporting numbers (tables E12 and E13):
- **Outcome-free accuracy.** Within one book's ladder, interpolation misses a held-out rung by 0.018 logits and the slope by 0.023. The model's SD misses by 0.047 to 0.088, and an unshifted quote by 0.31.
- **Weights on outcomes (NBA).** On class-B legs the slope earns 0.26 (.16), against 0.07 (.10) for the engine and 0.24 (.13) for the model's SD. On the legs where all three exist it earns 0.59 (.21).
- **Head to head.** Pooled with the engine's number, the slope takes the weight: slope 0.53 (.14), engine −0.21 (.09).
- **Walk-forward.** Adding the slope as class B's book term gains +3.4 [+0.4, +6.6] in NFL class B and +1.9 [−2.5, +5.8] in NBA.
- **Coverage.**
  - Interpolation reaches only 4 % of class-B legs.
  - The slope reaches 69 %: nine NBA markets and three NFL yardage markets.
  - NHL has no ladder data, so it has no translation.
- **Freshness.** Weights are earned on fresh quotes: the slope gets 0.52 (.15) at 20 minutes or less and 0.12 (.31) at 45 to 90 minutes.

## 2. The recommended estimator

```
ROWS   one per settled posted offer of Underdog and Sleeper (pushes dropped); y = 1 if Over hit.
       Book inputs are the ones the scoring run saw: compute them at scoring time and STORE them
       in history (class, book_A, books, quote age) so the nightly fit never re-joins the archive.
INPUT  po, pu         posted payouts (gross per unit) of Over and Under; skip the fit if one side is missing
       pf  = (1/po) / (1/po + 1/pu);  x0 = logit(pf)            # every probability clipped to [0.001, 0.999]
       kind = sleeper | ud_even (po == pu) | ud_priced
       served = Model Version != "book_fallback";  pm = model P(Over) (Win Prob, or 1 - Win Prob if Bet = Under)
       book_A = plain mean over books of each book's latest visible price at the app's exact line
                (odds row at that line, else that book's ladder rung there), quote age <= 90 min
       cls = A if book_A exists; B if any visible book quote <= 90 min at another line; else C
DESIGN offset x0; columns
       c       1
       a1      x0                                   # omitted where x0 is constant (ud_even)
       w_book  logit(book_A) - x0                   # class A only (class B gains one when M2 earns it)
       w_model served       * (logit(pm) - x0)
       w_fb    (1 - served) * (logit(pm) - x0)
MODEL  logit P(Over) = x0 + X b;  b = 0 is the no-evidence point (c = 0, a = 1, w = 0)
PRIOR  b_j ~ Normal(0, 1/lambda_j),  lambda_j = n0 * 0.25 * mean(X_j^2 over the training rows),  n0 = 4000
FIT    nightly, per league x kind x class, on rows with Date in [D - 90 d, D); Newton / IRLS on
       log-likelihood + log prior. Fewer than 400 rows: no fit (b = 0).
       Monthly: n0 <- argmin over {2000, 4000, 8000} of the summed walk-forward log loss of the last 28 dates.
STORE  per stratum: b; Sigma = inv(X'WX + diag(lambda)); the date-clustered sandwich (for display and
       the scoreboard); n_legs, n_dates, window, n0, lambda; walk-forward dLL of the last 28 dates with interval.
SCORE  eta = x0 + x'b;  se = sqrt(x' Sigma x);  side = the model's side (Q7)
       p_side = sig(+eta) for Over, sig(-eta) for Under;  m = that side's payout
       EV = p_side * m;  p_lo = sig(+-eta - 1.645 * se);  EV_lo = p_lo * m
       "+EV" mark iff EV_lo >= 1;   Kelly (single) = 0.5 * max(0, (EV_lo - 1) / (m - 1))
NO-EVIDENCE DEFAULT  b = 0 and se = 0: p = pf, EV = pf_side * m (< 1 by the app's cut), no mark, no stake.
       Applies below the floor, to one-sided rungs, to offers with no Win Prob, and to any stratum under kill K2.
ENTRY  k legs from k games, payout G = T_k * prod(m_i):  P_lo = prod p_lo,i;  mark iff G * P_lo >= 1;
       stake = 0.5 * max(0, (G * P_lo - 1) / (G - 1)); per slate, the summed stake of the entries holding
       any one leg <= the largest stake among them.
```

## 3. Acceptance test (frozen history, walk-forward, last 33 dates) and kill lines

| # | Test | Threshold | Recommended (n0 = 4,000) | Plan as written |
|---|---|---|---|---|
| AT1 | Overall dLL | ≥ +2, lower bound > 0 | +5.1 [+1.3, +8.8] PASS | −2.6 [−12.0, +5.9] FAIL |
| AT2 | Each league × kind (all have ≥ 2,000 legs) | point ≥ 0 and lower bound ≥ −8 (≈ a 2 pp bias on an even pick) | points +0.4 to +10.7, lowest bound −6.1 PASS | NFL Sleeper −25.4 [−70.7]; NBA even −11.8 [−27.4] FAIL |
| AT3 | Bulk honesty, model's side, each value band with ≥ 150 legs | \|says − paid\| ≤ .02, or the interval covers says | 4 of 4 bands PASS (table E6) | n/a (other side rule) |
| AT4 | "+EV" mark with no information (simulation, 3 runs) | ≤ 1 marked leg per 33 dates | 0 PASS | 986 FAIL |
| AT5 | Rank play minus today's rule, paired by date (per date × league × kind, the top N legs by EV on the model's side, N = today's count) | lower bound ≥ −3 pp a leg | +1.1 pp [−2.5, +5.0] PASS | not run |
| AT6 | Code parity | the shipped fit reproduces AT1 within ±0.3 and AT2's table on this frozen history | engineering check | |

**Kill lines.** Each is read nightly on the live walk-forward once at least 28 new scored dates exist.
- **K1.** The overall dLL's upper bound is below 0. Every stratum then reverts to the no-evidence default and Q1 reopens.
- **K2.** A league × kind's dLL upper bound is below 0. That stratum takes the default until its trailing dLL turns positive.
- **K3.** After 50 or more marked legs, the "+EV" set's upper bound on paid is below 1. Raise z to 2.33 or double n0.
- **K4.** The rank play minus today's rule has an upper bound below 0. The fit ranks worse than the rule it replaces, so the specification is wrong.
- **K5.** The fit's side beats the model's side on disagreeing offers with a lower bound above 0. Reopen Q7.
- **Specification is wrong** if AT1 or AT2 fails on a re-run, if K1 or K4 fires, or if the monthly n0 choice runs to the grid's top (8,000 or more) for two months, which would mean the fit has become the app's price.

No stratum needs a forced default beyond these rules. NBA even, NBA Sleeper and all of NFL already sit within half a point of the app's price.

## 4. Evidence (scripts in `ev_fit/`)

**E1. Replication** (`s02_replicate.py`, `.out`).
- All audit numbers reproduce exactly.
- Underdog raw same-line book: NBA 0.80 (.28), NFL 1.39 (.57), NHL 0.83 (.21).
- Class-C model: 0.01 (.09), 0.20 (.13), 0.59 (.12).
- Class-B translated book: NBA 0.15 (.10), NFL −0.06 (.10).
- Walk-forward bands, model input (73,565 legs): .885/.892, .919/.909, .949/.928, then .983/.864, 1.012/.906 and 1.052/.969 at the top.
- Walk-forward bands, raw book (58,049 legs): .921/.909, .951/.929, .983/.930, 1.011/.955.
- Market names: one rename, NBA "fantasy points underdog" → "prizepicks" (931 legs). No book quotes it under either name, so joining on the raw name lost no book-quoted legs.

**E2. What the scoring run saw** (`s03_quote_age.py`, `s04_visibility.py`, `s05_weights_visible.py`).
- *Where the served quote comes from.* It is always the newest archive row the run could read: 100 % of served stamps match an archive stamp.
- *Why it is a median 79 minutes old* (NBA 80). Every scoring run fires at :50, and NBA's last readable quote is usually the :30 broad confer an hour and twenty minutes earlier. The closing capture for games 10 minutes out is stamped 0 to 13 s before the run, and only 29 of 426 runs read it. That accounts for all 42 % of rows where the archive held a newer row.

Weights on visible quotes against the audit's:

| Underdog class A, raw book weight | Audit | Visible | Visible, ladder rungs added | Class shares A/B/C, audit → visible (with rungs) |
|---|---|---|---|---|
| NBA | 0.80 (.28) | 0.53 (.32) | 0.53 (.23), 13,166 legs | 33/39/28 → 21/27/52 (28/21/52) |
| NFL | 1.39 (.57) | 1.39 (.58) | 1.24 (.47) | 19/41/39 → 17/37/46 (18/36/46) |
| NHL | 0.83 (.21) | 0.92 (.23) | 0.92 (.23) | 48/17/35 → 40/14/47 (40/14/47) |

**E3. Specifications, walk-forward** (`s06_q1_specs.py`, `s10_nested_n0.py`). Figures are dLL ×10⁴, then the legs valued ≥ 1 on the fit's best side and what they paid.

| Spec | dLL | Valued ≥ 1: legs, paid |
|---|---|---|
| Plan, unpenalized | −2.6 [−12.0, +5.9] | 3,018, 0.863 [0.818, 0.924] |
| c fixed at 0 | +3.0 [−3.2, +8.6] | 1,432, 0.896 |
| a fixed at 1 | −6.2 [−15.2, +1.6] | 3,174, 0.855 |
| Prior n0 = 500 / 1,000 / 2,000 | +2.4 / +3.9 / +4.9 [+0.1, +9.4] | 921 / 473 / 186, 0.861 / 0.818 / 0.886 |
| **Prior n0 = 4,000** | **+5.1 [+1.3, +8.8]** | 57, 0.709 [0.536, 0.886] |
| Prior n0 = 8,000 / 16,000 | +4.7 [+1.8, +7.5] / +3.7 [+1.7, +5.6] | 11 / 1 |
| n0 = 2,000, model term against the book | +4.9 [+0.2, +9.3] | 178, 0.890 |
| n0 = 2,000, kinds pooled | +3.5 [−2.8, +9.2] | 670, 0.918 |
| n0 = 2,000, half-life 14 / 30 days | +4.4 / +4.7 | 70 / 107 |
| n0 = 2,000, floor 1,000 / 200 | +5.2 / +5.0 | 176 / 186 |
| Nested n0 choice (one for all) | +4.6 [+0.8, +8.5] | picks 4,000 or 8,000 on 83 % of legs |

**E4. Recommended spec by league × kind** (`s06_per_kind.csv`). Figures are legs, then dLL [95 %].

| | Even | Modified payout | Sleeper |
|---|---|---|---|
| NBA | 9,408: +0.4 [−5.7, +6.5] | 35,217: +7.4 [+3.0, +11.4] | 25,571: +0.7 [−5.0, +6.2] |
| NFL | 2,270: +0.5 [−3.0, +3.9] | 7,124: +1.3 [−3.5, +6.0] | 3,582: +1.0 [−4.8, +7.0] |
| NHL | 2,882: +2.2 [−6.1, +9.6] | 14,867: +8.4 [−3.2, +20.1] | 16,368: +10.7 [−2.0, +23.4] |

**E5. Marks under known truths** (`s07_q2_sim.py`, `s07b_q2_sim_lcb.py`, `.csv`). Each world has 3 runs; the figures are legs marked per 33 dates, with their mean true value.

| Mark rule | Null: plan / n0 2,000 / n0 4,000 | Informative truth: plan / 2,000 / 4,000 |
|---|---|---|
| point ≥ 1 | 986 (0.893) / 27 (0.889) / 7 (0.891) | 1,460 (0.938) / 97 (0.984) / 37 (0.991) |
| one-SE bound ≥ 1 | 77 / 1 / 0 | 184 (0.953) / 8 (1.038) / 2.3 (1.046) |
| 95 % bound ≥ 1 | 6 / 0 / 0 | 32 (0.979) / 0.7 (1.110) / 0 |

**E6. Model's-side bands at n0 = 4,000** (`s13_variants.py`). Figures are says → paid [95 %].
- [0, .90): 73,323 legs, .872 → .889 [.881, .897]
- [.90, .94): 41,289 legs, .916 → .915 [.903, .928]
- [.94, .97): 2,455 legs, .948 → .933 [.898, .974]
- [.97, 1): 173 legs, .980 → .948 [.813, 1.087]
- ≥ 1: 49 legs, 1.027 → .787 [.612, .953]

Clipping the model term at ±0.25 or ±0.5 changes dLL by +0.14 or +0.10 (both not significant) and leaves the top band unchanged.

**E7. Q7 legs the fit values ≥ thr** (`s08_q7_side.py`, prototypes' unpenalized fit). Each cell gives legs and return [95 %]. The last column is the fit's side minus the model's side on the flipped legs.

| thr | On the model's side | Flipped to the other side | The same flips, played on the model's side | Difference |
|---|---|---|---|---|
| 0.98 | 3,140: −8.1 % | 2,153: −18.7 % [−23.9, −12.8] | −5.5 % | −13.3 pp [−21.7, −2.1] |
| 1.00 | 1,728: −10.6 % | 1,290: −17.9 % [−23.6, −9.2] | −6.4 % | −11.6 pp [−21.6, +5.2] |
| 1.03 | 673: −12.1 % | 582: −18.8 % | −6.8 % | −12.0 pp [−25.5, +9.3] |

On the whole board (n0 = 4,000, 33,494 disagreeing offers), fit side minus model side per leg, by league × kind:
- NBA: Sleeper −1.8 [−6.8, +2.9]; modified payout −4.1 [−10.5, +2.4]; even −1.0 [−7.8, +6.7]
- NFL: Sleeper −9.0 [−15.9, −0.4]; modified payout −2.1 [−10.9, +6.2]; even −5.1 [−17.4, +10.3]
- NHL: Sleeper +0.3 [−6.8, +7.9]; modified payout −3.6 [−11.8, +5.1]; even −4.8 [−18.9, +8.5]

The fit's stated chance on its side, against the realized hit, by quarter of the model's disagreement: .474/.462, .469/.462, .462/.450, .442/.417.

**E8. Side rules on the whole board, n0 = 4,000** (`s12_q7_rules.py`). Every rule picks one side for every offer.
- Whole-board return: today's rule −10.08 % [−10.77, −9.41]; fit's maximum −10.84 %.
- Per flipped leg against the model's side:

| Rule | Offers flipped | Rule minus the model's side, per flipped leg |
|---|---|---|
| Fit's maximum | 33,494 | −2.7 pp [−5.7, +0.3] |
| Value gap 0.05 / 0.10 / 0.20 | 4,210 / 199 / 4 | +2.0 [−3.7, +8.2] / −10.3 [−40.3, +20.2] / noise |
| Value bar 1.00 / 1.03 | 8 / 3 | noise |
| Lower-bound flip, z = 1 / 1.645 | 1 / 0 | |
| Model's value side (model P × payout) | 39,405 | −5.1 pp [−7.7, −2.6] |

**E9. Class A and fallback rows** (`s14_q4_scoreboard.py`, `s15_fallback_split.py`; all apps, date-clustered).
- *Served P vs `Standalone Prob`, beside the book* (rows that carry both):
  - NBA, 8,900 legs: book 0.72 (.31) + served 0.20 (.17), or book 0.78 (.32) + standalone 0.06 (.08)
  - NFL: served 0.17 (.29), standalone 0.01 (.11)
  - NHL: served 0.00 (.20), standalone 0.06 (.13)
- *Model term on served rows only*, classes A / B / C:
  - NBA: 0.10 (.15) / 0.01 (.10) / −0.05 (.07)
  - NFL: −0.02 (.17) / 0.03 (.07) / 0.14 (.10)
  - NHL: −0.01 (.13) / 0.59 (.27) / 0.48 (.08)
- *Fallback term:* NBA A −0.65 (.27), NFL A 1.82 (.77), NHL C 1.22 (.47).

**E10. Precision of the class-B weight** (`s14_q4_scoreboard.py`).
- *Today's SE.* Engine number: NBA .098 (22 dates, 9,534 legs) and NFL .106 (7 dates). Slope: NBA .159 and NFL .267.
- *The paired step (NBA).* Engine 0.05 against slope 0.25: a difference of +0.20 with paired SE .11, against .20 if the two estimates were independent (their correlation is .73).
- *Detectable change,* projected at 6.5 NBA and 3 NFL game dates a week:

| Comparison | 4 weeks | 8 weeks | 13 weeks |
|---|---|---|---|
| NBA engine term, two independent periods | 0.36 | 0.25 | 0.20 |
| NFL engine term, two independent periods | 0.32 | 0.23 | 0.18 |
| NBA slope term, two independent periods | 0.61 | 0.43 | 0.34 |

**E11. Stakes** (`s16_q3_stakes.py`, n0 = 4,000, model's side).
- The fitted logit's SE has a median of .029 to .033.
- Singles: point, quarter and shrunk Kelly each staked 49 legs, which returned −21.3 % [−38.8, −4.7]. Bankroll log growth over the window was −0.34 (full), −0.08 (quarter) and −0.08 (shrunk). The lower-bound rule staked 0.
- 2-pick Underdog Power pairs of even picks: 21 of 259 pairs were priced at 1 or more and returned +16.7 % [−46.2, +92.5]. The lower-bound rule staked 0.
- Even legs fitted above .55 hit .421 (19 legs).

**E12. Class-B weights by translation** (`s11_q5_translate.py`). Figures are weight (SE) and legs.

| | Engine `Market Prob` | Rung interpolation | Book slope | Model SD | Unshifted |
|---|---|---|---|---|---|
| NBA | 0.07 (.10) 9,534 | none bracket | 0.26 (.16) 9,514 | 0.24 (.13) 4,371 | 0.01 (.10) |
| NFL | −0.04 (.11) 3,116 | 1.14 (.72) 792 | 0.30 (.27) 2,866 | −0.12 (.09) 4,136 | 0.25 (.22) |
| NHL | −0.16 (.33) 2,897 | none | unusable (one rung pair) | 0.48 (.29) 2,964 | −0.02 (.15) |
| NBA, common legs (3,738) | 0.17 (.12) | | 0.59 (.21) | 0.14 (.14) | −0.18 (.27) |
| 3 leagues, line distance 1 | 0.14 (.14) | | 0.47 (.20) | 0.22 (.15) | 0.01 (.09) |
| 3 leagues, quote ≤ 20 min / 45–90 min | 0.16 (.08) / −0.27 (.19) | | 0.52 (.15) / 0.12 (.31) | −0.02 / −0.05 | −0.02 / 0.18 |

Encompassing, three leagues pooled:
- engine −0.21 (.09) with slope +0.53 (.14), 11,123 legs;
- engine +0.08 (.07) with model SD −0.07 (.09).

**E13. Outcome-free translation error** (`s11_q5_translate.py`, logit RMSE).
- Held-out interior rung of one book's ladder:
  - NBA (11,155 rungs): interpolation .018, slope .023, model SD .047, unshifted .312
  - NFL (1,406 rungs): .019, .023, .088, .314
- Other-line books on class-A legs, against the same-line price:
  - NBA: slope .100, model SD .114, unshifted .471
  - NFL: .072, .090, .226
- The slope's cross-book error equals the spread between books at the same line.

## 5. Reality checks

- **The prize is small.**
  - +5 × 10⁻⁴ nats a leg is a better probability, not a profit: the rank play still loses 7.4 % a leg, against 8.5 % for today's rule.
  - No specification found a set of legs that credibly returns 1 or more.
  - Expect "+EV" marks and Kelly stakes to be near zero until the evidence grows.
  - The plan's "every `Kelly > 0` reader follows without a change" therefore means those readers go quiet. The board is carried by rank.
- **This is an engineering project, not a research one.** The fit is penalized logistic recalibration (Cox 1958; Steyerberg et al. 2004; Satopää et al. 2014): about 100 lines and one nightly job.
- **The data window is the binding risk.** The analysis covers 33 test dates in November, with only 14 NFL dates. Its NBA classes depend on cron timing that is about to change (E2).
- **What could make this wrong:**
  - the closing-capture fix raising the class-A weights back toward the audit's numbers;
  - a regime change at the trade deadline or the playoffs;
  - MLB and WNBA, which are absent from the window.

## 6. Not settled, and what would settle it

1. **The class-A NBA weight with the closing quote visible** (0.53 now, 0.80 in the audit). It needs 2 to 4 weeks of live rows after the cron fix, when the final scoring pass runs after close-lines writes.
2. **NHL class B.** NHL has no alternate-line ladders. The model-SD translation earns 0.48 (.29), which is too weak to ship, and no new data is allowed. NHL class B stays without a book term.
3. **The top over-claim beyond the noise** (E6, ≥ 1). It shrinks with the model's disagreement. A model term that varies by market or by size of disagreement needs about 3 more months of top-band legs to fit.
4. **Partial pooling across leagues** (a hierarchical prior centred on a pooled fit) was not run.
5. **Whether the stake rule grows a bankroll.** It is untestable here, because it staked nothing.
6. **The slope table** used every date. Its inputs are outcome-free, but production should build it from earlier dates only. It also needs at least 100 rung pairs per market; NFL passing touchdowns has only 6.
7. **Which covariance to use for se.** The frozen-data marks used the model-based covariance. The date-clustered one may be larger, which only makes the mark rarer.

## 7. Where this differs from the earlier briefs

- **R1 trust layer** (`researcher_trust_layer.md`, killed 2026-10-04). This fit is the same family: a conditional model of the hit on all posted legs, applied after the side is chosen, so selection is ignorable given its inputs.
  - *What is new.* R1 died because the model added nothing out of sample (its ablation). The owner has since ruled that the fitted number leads even where only the app's price survives.
  - *What still binds.* R1's finding holds for NBA and NFL here too (E4 and E9). R1's recommendation to fit on all posted legs before selection is followed.
- **Selection shrink** (killed 2026-10-03). That lever was fitted on the selected legs, with λ_sel ≈ 0. This fit is not fitted on selected legs.
- **w_ne and λ_DFS** (`researcher_gates_vs_pickem_profit.md`, killed as abstention). Their algebra matches the prior's pull toward the app. The owner now treats the no-evidence value as deliberate, so the few marks it leaves are the intended outcome. Their evidence still binds as a fact: honest probabilities leave almost nothing above break-even in NBA and NFL.
- **Book pull** (`researcher_book_pull_rejudge.md`). λ\* is a two-forecast weight. This fit subsumes it per stratum and on outcomes, and the plan retires `book_pull` in M1.

## 8. Bibliography

| Source | Identifier | Used for |
|---|---|---|
| Kelly, J. L. (1956). A new interpretation of information rate. *Bell System Technical Journal* 35(4) | DOI 10.1002/j.1538-7305.1956.tb03809.x | stake formula |
| Cox, D. R. (1958). Two further applications of a model for binary regression. *Biometrika* 45(3/4) | DOI 10.1093/biomet/45.3-4.562 | intercept and slope recalibration |
| Stern, H. (1991). On the probability of winning a football game. *The American Statistician* 45(3) | DOI 10.1080/00031305.1991.10475798 | translating a line through a fitted spread |
| Harvey, Leybourne & Newbold (1998). Tests for forecast encompassing. *JBES* | DOI 10.1080/07350015.1998.10524759 | encompassing tests |
| Steyerberg et al. (2004). Validation and updating of predictive logistic regression models: sample size and shrinkage. *Statistics in Medicine* 23(16) | DOI 10.1002/sim.1844 | penalized updating of a base forecast |
| Smith & Winkler (2006). The optimizer's curse. *Management Science* 52(3) | DOI 10.1287/mnsc.1050.0451 | Q2 mechanism |
| Whitrow, C. (2007). Algorithms for optimal allocation of bets on many simultaneous events. *JRSS-C* 56(5) | DOI 10.1111/j.1467-9876.2007.00594.x | slates with shared legs |
| Grant, Johnstone & Kwon (2008). Optimal betting strategies for simultaneous games. *Decision Analysis* 5(1) | DOI 10.1287/deca.1080.0106 | simultaneous bets |
| Gelman, Jakulin, Pittau & Su (2008). A weakly informative default prior for logistic regression. *Ann. Appl. Stat.* 2(4) | DOI 10.1214/08-AOAS191 | prior on logistic coefficients |
| Ranjan & Gneiting (2010). Combining probability forecasts. *JRSS-B* 72(1) | DOI 10.1111/j.1467-9868.2009.00726.x | pools need fitting on outcomes |
| Snowberg & Wolfers (2010). Explaining the favorite–long shot bias. *J. Political Economy* 118(4) | NBER w15923, DOI 10.3386/w15923 | free slope a |
| Clements & Harvey (2010). Forecast encompassing tests and probability forecasts. *J. Applied Econometrics* | DOI 10.1002/jae.1097 | encompassing with probability forecasts |
| Efron (2011). Tweedie's formula and selection bias. *JASA* 106(496) | DOI 10.1198/jasa.2011.tm11181 | selection bias |
| Baker & McHale (2013). Optimal betting under parameter uncertainty: improving the Kelly criterion. *Decision Analysis* 10(3) | DOI 10.1287/deca.2013.0271 | shrunk Kelly (compared) |
| Satopää et al. (2014). Combining multiple probability predictions using a simple logit model. *IJF* 30(2) | DOI 10.1016/j.ijforecast.2013.09.009 | logit pooling, slope a > 1 |
| Cameron & Miller (2015). A practitioner's guide to cluster-robust inference. *J. Human Resources* 50(2) | DOI 10.3368/jhr.50.2.317 | date-clustered SEs |
| Chu, Wu & Swartz (2018). Modified Kelly criteria. *JQAS* 14(1) | DOI 10.1515/jqas-2017-0122 | Kelly variants |
| Sun & Boyd (2018, rev. 2021). Distributional robust Kelly gambling | arXiv:1812.10371 | Kelly at the lower bound |
| Uhrín, Šourek, Hubáček & Železný (2021). Optimal sports betting strategies in practice: an experimental review. *IMA J. Management Mathematics* | DOI 10.1093/imaman/dpaa029, arXiv:2107.08827 | fractional Kelly in practice |
