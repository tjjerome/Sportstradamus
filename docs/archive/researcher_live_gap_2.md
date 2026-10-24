# In-repo research brief (live-gap-2 §4.1): pre-registered statistics, populations and Holm family for E1–E4, a sanity read of E5's alarm, and what the data in hand already decides (2026-10-21)

**Prior.** R3 (`docs/archive/researcher_train_serve_skew.md`).

**Data.** Every number was measured on dev data on 2026-10-21 unless marked *inferred*:
- history synced 10-21 03:29 CDT;
- the matrices and pickles from the 10-18 retrains;
- the feature log, 10-05 → 10-26.

Appendix A has the command for each number.

**Terms.**
- **CITL** (calibration-in-the-large of the book) = mean(P_book,over − y_over). Positive means the book over-claims the Over.
- **y** = `training/labels.over_label`: 1 if Result > Line, ½ on a push, 0 otherwise.
- **"Priced"** = has a sportsbook quote (`QuoteAuthenticity == "authentic"`).

## TL;DR

- **H3 is close to decided by the data in hand, and the defect is not limited to the NFL.**
  - The NFL count cells' training book leg is off the outcome level by 4.7–27.8 points. These are the held-out rows that fit the blend weight and score Gate 1. On the Gate-1 rows: receptions +8.6, carries +4.7, tds −11.9, passing tds −17.6, interceptions −27.8.
  - The 2026-season rows are levelled (|CITL| ≤ 3.0). Live `Market Prob` is within 1.2 points on receptions, carries, interceptions and tds.
  - Across the 38 served cells with ≥ 100 priced held-out rows, offline Brier skill against the book tracks the book's level error (Spearman ρ = .75).
  - On 10 of the 11 cells with |CITL| > .05, re-levelling the book by its own mean drops the offline BSS from .05–.26 to ≤ .052.
  - Preview of the pinned H3 statistic: level error .102 offline vs .017 live.
  - So I6d opens. Its scope is book-leg levelling in every league. The first candidate it should pre-register is refitting w and Gate 1 on levelled rows.
- **E1 is pre-registered as a forward-honest test at one origin, with the served hyperparameters pinned.**
  - W is every row dated on or after d_W, the date of row ⌊0.9n⌋. Every arm trains only on rows before d_W.
  - Decision arm (amendment): **C**, the booster refit on all pre-W rows with arm A's calibrators transplanted. B (f = 0.85) is reported but does not decide.
  - Statistic: the paired difference in Brier score of the served P on priced W rows, with a 95% CI from a game-date block bootstrap. Yards is scored on all 2,088 W rows.
  - The minimum detectable effect at 80% power (MDE80) is .0006–.0014 Brier on the four priced cells and .0055 on yards. In practice "3 of 5" means "3 of the 4 priced cells".
  - R3 found no decay in booster information, and the calibration chain damps booster changes by w/T (.17–.19 on receptions and carries). Both predict NFL nulls, so H1 will likely fail, but cheaply.
- **E2 as written would not measure gate optimism.**
  - Today's `--holdout-blind` requires `--deterministic`, which trains a different booster: `DETERMINISTIC_FIXED_PARAMS` (30 rounds, 31 leaves, lr 0.1).
  - Pre-registered instead: re-gate the **served** booster, cross-fitting only its calibrators. The read is descriptive:
    - pass→fail flips against the stored verdicts;
    - a slack shift with both halves cross-fit the same way;
    - the Gate-1 CI-method read: 39 cells have a Gate 1; all 39 pass with an iid or player-clustered CI, 38 date-clustered, 34–35 two-way; 31 cells have no Gate 1;
    - E1-A's forward gates.
- **H4 needs a population amendment; the Holm family has two tests.**
  - 3,255 of the 3,611 offline rows in the > .10 band are pre-2026 NFL count rows priced against the mis-levelled book. The literal read would therefore measure H3.
  - Primary population: rows where the opponent is levelled. That is 356 band rows, counted without looking at outcomes. Power is .65–.85 for the live −.115 [−.164, −.062].
  - Holm covers {H3, H4}, one-sided, family-wise error rate (FWER) .05. H1 is decided by its own count rule, whose false-adopt rate is ≈ 1.5 × 10⁻⁴ under the global null. E2, E3's parity reads and E5 are descriptive.
- **E5's 1 pp alarm reads only the sd of the pre-blend model P** (`feature_parity.py:157–163`, `:238`).
  - Served ΔP ≈ (w/T) × raw ΔP, a factor of .19–.83 on the E1 cells. The alarm is therefore a conservative bug detector, but it cannot see a level shift.
  - Flag on RMS(ΔP) instead, and rank repairs by served RMS × live volume.
  - Score NFL through the current pickle: only 4–36 rows per cell fall inside the matrix window under their own version's file.

## 1. Decision questions and the statistics that decide them

| H | Decision it unblocks | Deciding statistic (pinned) | §3 rule | Amendment (owner decides) | Data in hand |
|---|---|---|---|---|---|
| H1 | Adopt "refit the booster on all rows for serving" in the week-3 batch, or close the train fraction / refit as a lever | Per cell: d̄ = mean[(P_A − y)² − (P_C − y)²] over the scored W rows, with P the served over-probability at the matrix `Line`. 95% CI from a game-date block bootstrap (4,000 draws). Win: CI lower bound > 0. Loss: CI upper bound < 0 | Kept: ≥ 3 of 5 wins and 0 losses | C is the decision arm; B (0.85) is descriptive. This removes §3's "or" fork | Power in §3.2. R3 predicts NFL nulls |
| H2 | Owner ask §8.3 (whether re-gated cells leave `devel`). The blind walk ships regardless | Number of served cells whose `ship` flips pass→fail when re-gated on the validation half, with the served booster and cross-fit calibrators | Kept, read as descriptive | Use the served booster, not today's deterministic one. Add E1-A's forward gates | The Gate-1 CI method alone flips 0–5 of 39 |
| H3 | Open I6d (count-quote levelling), or route the count cells to H4 | L_off and L_live = row-weighted mean of per-cell \|CITL\|. Offline uses the served model's priced held-out rows (both halves); live uses settled quoted main lines. H3 holds if L_off > .05, L_live ≤ .02, and D = L_off − L_live > 0 survives Holm | Kept; pinned | None | Preview (unblinded): L_off .102, L_live .017 |
| H4 | Propose a disagreement-band monitor column (never a gate), or call the live band a population shift | BSS of served P vs the book on priced test rows with \|P − P_book\| > .10. One-sided date-block test of BSS < 0, Holm-adjusted | Population amended | Primary population = rows with a levelled opponent. The literal Gate-1 rows become secondary | 356 band rows (counted blind) |
| E5 | Which cells get a serving-side feature repair this week | RMS(ΔP) of the model's over-probability, serve-time vs matrix features, same file, same line, same book leg. Alarm at > 1 pp. Served-chain RMS sets repair order | I6e threshold | Flag on RMS, not on sd alone | Damping table in §3.6 |

**Design choices that change a verdict, and the choice pre-registered**

| Choice | Pre-registered | Why |
|---|---|---|
| E1 decision arm (§3 says "0.85 or refit-on-all") | **C** | C is the protocol week 2 would build (lane §5) and changes only the booster. B mixes booster recency with calibration rows that are half as many and later. An unadjusted "or" is a forking path. |
| E1 hyperparameters | The served pickle's θ in every arm; no HPO | Unseeded TPE (`hyperparams.py:147`) makes two runs of the same protocol differ (B27). θ_served was tuned only on rows before the served 0.7 cut, all of which precede d_W. |
| E1 cluster unit | Game date, with NFL week as a sensitivity check | Measured design effect ≈ 1 on arm contrasts, and date ≈ week. 36 dates vs 14 weeks matters with few clusters (B22, B23). The house rule is day-clustered. A cell whose verdict differs between the two counts as neither a win nor a loss. |
| E1 rows for yards | All W rows (no book) | Only 38 priced W rows, on 4 dates. The paired arm contrast needs no book. |
| E2 booster | Served booster; calibrators cross-fit | Today's blind run trains `DETERMINISTIC_FIXED_PARAMS`, so its flips would measure the booster difference. |
| H3 offline rows | The served model's priced held-out rows, both halves | These rows fit w (validation half) and score Gate 1 (test half), which is the mechanism H3 names. The 2026-only rows are levelled and would answer a different question. |
| H3 pooling | Row-weighted mean of per-cell \|CITL\| | Signs differ by cell (receptions +, tds and interceptions −), so a signed pool cancels. |
| H4 population | Rows where the opponent is levelled | 90% of the literal band rows face a mis-levelled opponent (Finding 3). |
| Holm family | {H3, H4} | These are the only standalone directional claims. H1 has its own, stricter rule. |
| E5 alarm | RMS(ΔP) > 1 pp | The sd cannot see a uniform shift, and a uniform shift moves legs across the 5% edge bar. |

## 2. Key Findings

### Finding 1: offline skill against the book is, cell by cell, the book's level error

This finding is exploratory (post hoc, 38 cells).

**Association.** Across the 38 served cells with ≥ 100 priced held-out rows:
- Spearman ρ between |CITL_book| on the held-out window and offline BSS (`model_stats`) is **.75** (p ≈ 5 × 10⁻⁸).
- The 11 cells with |CITL| > .05 have median BSS **.144**.
- The other 27 have median **.002**, and none exceeds .035.

**Mechanism.** If the levelled book has mean error 0, shifting it by δ adds exactly δ² to its Brier score. The skill score is therefore inflated by δ²/BS_book. This is Murphy's (1973) reliability term (B16); CITL follows Van Calster et al. 2019 (B18). Re-levelling the book by its own mean (one parameter, in-sample on thousands of rows) removes most of the skill.

The table below uses the test (Gate-1) rows.

| Cell | n | CITL | BSS observed | δ²/BS_book | BSS vs re-levelled book | Served w |
|---|---|---|---|---|---|---|
| NFL interceptions | 450 | −.278 | .254 | .225 | .038 | .900 |
| NBA BLK | 1,318 | −.218 | .168 | .176 | −.010 | .818 |
| NBA STL | 640 | −.221 | .195 | .178 | .021 | .900 |
| NFL passing tds | 451 | −.176 | .121 | .117 | .005 | .900 |
| NBA FG3M | 1,602 | −.169 | .126 | .122 | .005 | .840 |
| NBA BLST | 118 | −.173 | .140 | .109 | .034 | .858 |
| NFL tds | 3,496 | −.118 | .131 | .084 | .052 | .832 |
| MLB home runs | 7,572 | +.126 | .256 | .121 | **.154** | .723 |
| NFL receptions | 2,529 | +.086 | .046 | .030 | .017 | .824 |
| NFL carries | 1,078 | +.047 | .026 | .009 | .017 | .861 |
| WNBA FG3M | 862 | −.068 | .047 | .018 | .030 | .900 |
| *Levelled comparators:* NFL receiving yards, rushing yards, attempts, completions; NHL shots; MLB hits | 454–7,813 | ≤ .026 | −.001 to .021 | ≤ .003 | −.002 to .019 | .33–.88 |

**Reading.** On these 11 cells, Gate 1 and w were fit against a broken opponent. Their offline "skill" is mostly the opponent's error. MLB home runs is the exception: something beyond level is wrong there (see Open question 2). R3 §9 #2 and #12 predicted this mechanism; it is now measured.

### Finding 2: both legs of H3 measured on the NFL count cells

The live window is 2026-09-21 → 10-20, settled quoted main lines (`realized.settled_offers`). Offline CIs use a date-block bootstrap with 1,000 draws (2,000 for MLB). Seasonal columns cover the held-out window, both halves, with the matrix `Odds` read as the book's over probability (`pipeline.py:932`).

| Cell | Offline CITL, Gate-1 rows [95% CI] | Held-out CITL: 2024 / 2025 / 2026 season | Live CITL of `Market Prob` [95% CI], n | w / offline BSS | Live BSS (lane §2.4) |
|---|---|---|---|---|---|
| receptions | +.086 [.062, .106] | +.087 / +.106 / −.002 | −.012 [−.036, .006], 2,568 | .82 / .049 | −.043 |
| carries | +.047 [.019, .072] | +.081 / +.058 / −.002 | −.008 [−.096, .069], 408 | .86 / .048 | −.083 |
| attempts | +.026 [−.014, .061] | +.018 / +.042 / +.008 | −.058 [−.161, .017], 389 | .32 / .014 | −.034 |
| completions | +.026 [−.015, .063] | +.018 / +.019 / −.030 | −.044 [−.128, .052], 236 | .64 / .008 | −.046 |
| interceptions | −.278 [−.336, −.217] | −.288 / −.319 / +.021 | −.001 [−.116, .081], 180 | .90 / .218 | −.073 |
| passing tds | −.176 [−.226, −.128] | −.270 / −.206 / −.022 | −.029 [−.124, .010], 186 | .90 / .185 | — |
| tds | −.119 [−.145, −.090] | −.189 / −.151 / +.009 | +.005 [−.041, .035], 536 | .83 / .144 | −.042 |
| receiving yards | +.004 [−.013, .020] | +.013 / +.012 / −.002 | −.011, 1,669 | .33 / .017 | −.005 |
| rushing yards | +.020 [−.011, .049] | +.033 / +.016 / +.040 | **+.060 [.035, .093]**, 942 | .42 / .016 | +.006 |

- **Where the defect sits.** All rows are `book_direct`. In 2026 the median number of books drops from 7–8 to 2–6, consistent with live-polled rows. The pre-2026 rows are the defect, and they make up 80–84% of each count cell's priced held-out rows (receptions 808 of 5,117 rows are 2026).
- **Preview of the pinned statistic** (7 measurable cells; Gate-1 rows only): L_off = **.102**, L_live = **.017**, D = .085.
  - The per-cell rule holds on receptions, interceptions and tds.
  - Carries (.047 offline) and passing tds (.029 live) are near misses.
- **Blend weight.** Across the 9 NFL priced cells, the served w tracks the opponent's error (ρ = .85). The weight cap of .90 binds often (NBA RA and REB sit at .90 with levelled books), so w is not only a level-error artifact.

### Finding 3: the live disagreement band is the count cells' band, and the literal E4 population would re-measure H3

**Live.** NFL, quoted main lines, 7,401 rows over 13 dates.

| Band | n | BSS |
|---|---|---|
| ≤ .02 | 1,723 | −.0023 |
| .02–.05 | 2,151 | +.0001 |
| .05–.10 | 1,792 | −.0083 |
| > .10 | 1,735 | **−.1153 [−.1644, −.0617]** (4,000 date-block draws) |

- Count cells hold **87%** of the > .10 rows: 1,514 rows, BSS −.120 [−.183, −.061].
- Yardage/combo cells: 221 rows, BSS −.088 [−.202, +.070].
- Counting each prop once (games ≥ 10-09 only): −.109 on 815 rows. Moved-line multi-counting is therefore not the driver.
- Versions in the band: 0917: 432 rows, 0927: 857, 0929: 50, 1003: 10, 1015: 240, 1018: 146.

**Offline, counted blind** (no outcomes read). Priced test rows of the 10 served NFL priced cells. The > .10 band has 3,611 rows over 103 dates and 471 players:

| Group | > .10 rows |
|---|---|
| Count cells, pre-2026 | 3,255 |
| Count cells, 2026 | 205 |
| Yardage, pre-2026 | 145 |
| Yardage, 2026 | 6 |

The literal Gate-1 band is therefore 90% rows where the opponent is off by 5–32 points. That read is predictably positive and answers H3, not H4.

Selection concentrates exactly where model and market disagree. This is the optimizer's curse (B28) and R3's rank-1 mechanism.

### Finding 4: today's `--holdout-blind` is a ranking mode with its own booster, so E2 needs a served-booster re-gate

What the code does today:
- `--holdout-blind` requires `--deterministic` (`cli.py:168–171`).
- It cannot be combined with `--frozen-matrix-dir` or `--artifact-output` (`cli.py:174–180`).
- `--deterministic` replaces every hyperparameter with `DETERMINISTIC_FIXED_PARAMS` (`pipeline.py:270–284`, `:1601–1602`).
- It evaluates only the validation half. The test half is absent from the run (`pipeline.py:1162–1170`).
- It cross-fits the calibration chain in 5 player-disjoint folds and leaves the GBDT untouched (`:4483–4495`, `:4536–4614`; selected at `:4990`).
- It writes to `data/test_sets/deterministic/{strategy}/` (`:2709–2733`).

The sweep nominates on these deterministic rows (`confirm.py:204–262`). The walk then ships the first full-HPO nominee that passes on the test half (`confirm.py:1250–1293`).

Literature:
- Selecting on the evaluation set biases the estimate (Cawley & Talbot 2010, B2). Varma & Simon (B3) found tuned classifiers on null data showing 38–42% CV error against a true 50%.
- Reusing one holdout adaptively overfits it (Dwork et al. 2015, B4). The walk's first-to-pass rule, the Gate-4 retries, the supersede comparisons and the manual flips are all adaptive queries of the same rows.
- The blind walk (rank on the cross-fit half, read the test half once) is the nested remedy.

E2 can only size the test-half selection relative to a validation half that the sweep also looked at, albeit through a different booster. E2 therefore stays descriptive.

### Finding 5: a forward-honest test of a refit protocol evaluates the whole procedure at an origin, with every fit strictly before it

- Tashman (B5) distinguishes fixed-origin from rolling-origin evaluation, and "updating" from "recalibrating" at the origin. The rolling forecasting origin is the textbook form (Hyndman & Athanasopoulos, B6, §5.10).
- Rolling setups leak when data travel from the test period into training (Hewamalage et al., B7).
- K-fold validity needs uncorrelated errors (Bergmeir et al., B8). The (Player, Date) hash split interleaves rows of the same games, and structured data needs blocked splits (Roberts et al., B9). This is why the same-period split flatters the calibrators.
- Giacomini & White (B11) frame the comparison as one between forecasting methods (model plus estimation scheme plus window), which is what E1 compares.
- Pesaran & Timmermann (B10): under structural breaks, the choice of window trades bias against variance. E1's arms are expanding windows. B and C add recent rows without dropping old ones, so they can only lose through their calibrators: B's are thinner, C's are transplanted.
- Cawley & Talbot: the evaluation must contain the calibration step. That is why C carries A's calibrators rather than refitting them on rows its booster saw.

### Finding 6: arm contrasts must hold training randomness fixed, and the CI covers only row sampling

- Comparing learning algorithms needs training-set and seed variance accounted for (Dietterich, B25; Nadeau & Bengio, B26). Hyperparameter-search variance alone can match benchmark effect sizes (Bouthillier et al., B27).
- With θ pinned, LightGBM's default seeds and the pickle's `num_threads: 8`, a fit is reproducible to float noise (`feature_parity.REPRODUCE_TOL = 1e-6`, `feature_parity.py:83`).
- The CI is conditional on the realized arms. It says how E1's fixed protocols compare at this origin, not across refits.

### Finding 7: the paired Brier difference on identical rows needs no book, and date clustering makes it a Diebold–Mariano test on date-aggregated losses

- Brier is strictly proper (Gneiting & Raftery, B15). On identical rows the book term cancels, so "Brier vs book" between two arms is the plain paired Brier difference.
- A date-block bootstrap of the mean difference is the Diebold–Mariano test (B12) on per-date mean loss differentials. Harvey–Leybourne–Newbold's correction (B13) applies when dates are few.
- Panel versions with cross-sectional dependence inside a period are in Qu, Timmermann & Zhu (B14).
- **Measured design effects:**

  | Contrast | Clustering | Design effect |
  |---|---|---|
  | Arm vs arm (10-15 → 10-18 contrast) | date | 0.8–1.2 |
  | Arm vs arm | player | ≈ iid |
  | Arm vs arm | NFL week | ≈ date |
  | Model vs book | date | 1.0–1.3 |

- **Few clusters.** NFL W has 14 weeks, and the E3 join has 3–5 dates. Both are in the regime where pairs-cluster inference under-covers (CGM 2008, B19; MacKinnon & Webb, B22; MacKinnon, Nielsen & Webb, B23).
- Two-way date × player CIs: CGM 2011 (B20) and Owen's pigeonhole bootstrap (B21, slightly conservative). Block bootstrap: Künsch (B24).

### Finding 8: power

- E1: the table in §3.2.
- E4 primary population (356 band rows):
  - The live per-row loss-difference sd in the band is .165 and the book's Brier score .231, giving SE ≈ .037 × design effect (1.0–1.3).
  - Power for the live −.115 is .65–.85, depending on the design effect and on whether H4's Holm step is .025 or .05. Power for −.05 is about .3.
- BSS is a ratio, and its sampling variance and small-sample bias are worked out by Bradley, Schwartz & Hashino (B17). They need hundreds of pairs; 356 is at the floor.

### Finding 9: Gate 1's CI method moves verdicts at the margin, and 31 of 70 served cells have no Gate 1

- 39 served cells have a computable Gate 1 (the other 31 are blank and auto-pass). The ship CI is a 95% two-sided iid bootstrap whose upper bound must be < .005 (`scorecard.py:1542–1552`, `:713–733`, `:146`). Pass counts by CI method:

  | CI method | Cells passing (of 39) |
  |---|---|
  | iid (the ship CI) | 39 |
  | player-clustered | 39 |
  | date-clustered | 38 |
  | two-way | 34–35 |

  The two-way CI flips NBA RA (.0063), NFL attempts (.0051), NFL yards (.0411) and WNBA PR (.0060). MLB runs allowed sits on the bar at .0050.
- NFL yards' Gate 1 rests on 19 rows, 18 players and 4 dates. Its iid upper bound is −.0012; its date-clustered upper bound is +.0223.

### Finding 10: E5's alarm reads the raw model P on its sd, and the NFL window under each row's own version is nearly empty

- `over_probability` is "the model's own over-probability … from its decoded pre-blend outputs" (`feature_parity.py:157–163`).
- `flag = dp.std() > PARITY_SD_ALARM` (`:238`, `:78`). The mean `dp_mean` is reported but not flagged.
- **NFL.** Feature-log rows by version:

  | Version | Rows |
  |---|---|
  | 0927 | 2,183 |
  | 0929 | 303 |
  | 1003 | 60 |
  | 1015 | 1,190 |
  | 1018 | 1,562 |

  `models/prior` keeps only the NFL 20260929 and 20261015 files. Rows scorable through their own file inside the matrix window: receptions 36, receiving yards 36, targets 24, the rest 4–19.
- **NHL** is ample: shots 1,423, points 1,745, goals 3,173, assists 1,700.
- **Fixed-file NFL parity is feasible.** The current 20261018 pickles' `expected_columns` are a subset of the logged frames for receptions, receiving yards, rushing yards, carries, interceptions, tds and yards. Completions is missing 42 columns, and qb yards has no pickle.

### Finding 11: R3's prior predicts an E1 null on NFL, and the calibration chain damps whatever the booster gains

- R3 (Findings 14–15, §9 #2): NFL booster information is ≈ 0 at every distance from the training cutoff.
- A full 10-15 → 10-18 retrain moved the served P by sd .014 against a raw-P sd of .056 on receptions, and .005 against .028 on carries. The calibration chain damps booster changes by about w/T (Finding 10, §3.6).
- NHL is damped far less (.6–.8). If recency matters (concept drift; Gama et al., B30), it shows on NHL first.

## 3. Pre-registration (routing protocol)

### 3.1 Multiplicity

**Holm family (B1).** m = 2, FWER α = .05, both tests one-sided.
- p-values come from 4,000 date-block bootstrap draws, seed 1729, floor 1/4,000.
- The smaller p is compared with .025, the larger with .05.

| Test | Hypothesis | Statistic, null side | Status |
|---|---|---|---|
| T1 | H3 | D = L_off − L_live ≤ 0. Offline dates and live dates are resampled independently in each draw | **Unblinded**: previewed today (D ≈ .085) |
| T2 | H4 | BSS > .10 band ≥ 0 on the primary population. Band membership is fixed per row | Blind: only counts were read |

**Outside the family**
- **H1** is decided by its count rule: per-cell two-sided 95% date-block CIs; ≥ 3 wins and 0 losses.
  - Under independence and the global null, a false adoption needs ≥ 3 one-sided rejections at .025 out of 5: probability **1.5 × 10⁻⁴**. This is far stricter than Holm.
  - A spurious veto under the null has probability 1 − .975⁵ = 12%, which is conservative toward not adopting.
- **Descriptive, carrying no p-value into any decision:**
  - E1 secondaries;
  - all of E2;
  - E3 (b)–(e);
  - E4's secondary populations and strata;
  - E5;
  - Findings 1 and 3 (post hoc).

### 3.2 E1: train fraction and refit (decides H1)

**Freeze first.** Copy each cell's matrix into the worktree's frozen-matrix directory and assert that its sha256 equals the served pickle's `model_strategy.matrix_hash` (all 70 served cells match today). Also freeze `stat_calibration.json` (cv, std, zi, book_shape) so every arm reads the same constants.

**Windows.** Sort the matrix stably by date (mergesort). d_W = Date of row ⌊0.9n⌋. **W** = all rows with Date ≥ d_W, so a date is never split. **P** = rows with Date < d_W.

| Cell | n | Served 0.7 cut | d_W | W rows | W dates | W players | W priced (2025 season / 2026) | W synthetic / pickem | A trains to | B trains to |
|---|---|---|---|---|---|---|---|---|---|---|
| NFL receptions | 20,000 | 2024-12-15 | 2025-12-14 | 2,189 | 36 (14 weeks) | 404 | 1,732 (924 / 808) | 291 / 166 | 2024-10-20 (P held-out 5,344; 4,599 priced) | 2025-09-14 (2,672; 2,331 priced) |
| NFL yards | 20,000 | 2024-12-08 | 2025-12-14 | 2,088 | 36 | 390 | 38 (0 / 38) | 1,786 / 264 | 2024-10-13 (0 priced) | 2025-09-07 |
| NFL carries | 9,256 | 2024-12-08 | 2025-12-14 | 966 | 36 | 182 | 639 (378 / 261) | 174 / 153 | 2024-10-17 (2,487; 1,936 priced) | 2025-09-07 (1,244; 1,019 priced) |
| NHL shots | 55,587 | 2025-12-11 | 2026-03-29 | 5,645 | 81 (75 priced) | 529 | 4,077 (3,525 / 552) | 1,201 / 367 | 2025-11-08 (14,983) | 2026-01-12 (7,492) |
| NHL points | 101,863 | 2025-12-13 | 2026-03-30 | 10,345 | 80 (62 priced) | 790 | 3,927 (3,840 / 87) | 5,554 / 864 | 2025-11-11 (27,456) | 2026-01-11 (13,728) |

**Arms and which rows each layer is fit on.** Every arm uses the served corner (distribution, normalization, post-hoc slugs, knobs) and the served θ, including `opt_rounds` and `num_threads: 8`. There is no HPO.

| Layer (pickle key) | A: f 0.7 on P | B: f 0.85 on P | C: refit on all of P |
|---|---|---|---|
| Booster (`model`) | P rows before P's 0.7 cut | P rows before P's 0.85 cut | All of P, same θ and rounds |
| Validation / test split | (Player, Date) hash of P's last 30% | Hash of P's last 15%, the rows just before W and disjoint from it | none |
| Blend weight (`weight`, `r_book`) | A's validation half, priced rows | B's validation half | Copied from A |
| Dispersion / skew (`dispersion_cal`, `skew_cal`, `shape_ceiling`) | A's validation half | B's validation half | Copied from A |
| PIT recal (`pit_recal_blob`) | same as above | same as above | Copied from A |
| Temperature (`temperature`) | same as above | same as above | Copied from A |
| Prob-stage post-hoc (`posthoc`, `posthoc_blob`) | same as above | same as above | Copied from A |
| Normalization (`normalized`, `offset_meta`, `target_normalization`), `cv`, `step`, `hist_gate` | From A's run and the frozen calibration file | Same frozen constants | Copied from A |
| Scored on | All of W | All of W | All of W |

C is A's file with only the booster swapped. That is exactly the "select and calibrate at 0.7, refit the booster on all rows for serving" protocol. Every W row is scored through the pipeline's test-dump path into a test-set-shaped CSV, so `compute_gates` and the scorecard read it unchanged.

**Primary statistic.** Per cell, d̄ = mean[(P_A − y)² − (P_C − y)²]. Positive means C is better.
- P = the served over-probability at the matrix `Line`.
- y is push ½.
- Rows:
  - priced W rows for receptions, carries, NHL shots and NHL points (pickem, synthetic and derived rows excluded);
  - **all W rows for yards**, which has no book: only 38 priced rows on 4 dates, and the paired contrast needs no book.
- Report ΔBSS = d̄ / mean Brier_book alongside, for the priced cells.

**CI and rule.**
- Game-date block bootstrap, 4,000 draws, two-sided 95% percentile.
- NFL week-block (Tuesday-start weeks) as a sensitivity check. A cell whose win/loss verdict differs between date and week blocks counts as neither.
- H1 holds if there are ≥ 3 wins of 5 and 0 losses. Then the refit-on-all protocol is adopted in the week-3 batch.
- Otherwise H1 is not a lever.
- A pass that comes only through B does **not** declare H1. B's result becomes a lead for a new pre-registration.

**Secondaries (descriptive).**
- B vs A and B vs C, same statistic.
- **Raw booster** per arm: `P_standalone` Brier and the model's NLL of `Result` on the same rows.
- Gate 4 (`g4_pit_ks`) per arm on W.
- The tail scorecard's selected gap per arm. It is descriptive below `DESCRIPTIVE_MIN_SELECTED = 100` (`graduation.py:53`). `TEST_SETS_DIR` is hardcoded (`tail_scorecard.py:59`), so run it in the worktree with the arm's files in place.
- W ∩ 2026 season: receptions 808 rows, carries 261, shots 552, points 87, yards 1,064.
- Optional second origin, robustness only: d_W′ = row ⌊0.8n⌋, window [d_W′, d_W).

**How to interpret the mechanism read (fixed in advance).** If C's raw booster beats A's on W (NLL or Brier CI excludes 0) but C's served P does not, the transplanted calibrators are the bottleneck. That is a week-2 design note (recalibrate after the refit), not an H1 pass.

**Reproduction check (blocking).** Before any arm runs, the harness retrains each cell at f = 0.7 on the full frozen matrix with θ_served. It must reproduce the served test-set `P` and `P_standalone` within 1e-6 and the served `split_fingerprint`. If it misses, stop: the harness is not production.

**Power.** Planning SE = the paired 10-15 → 10-18 served-P contrast on shared W ∩ test rows (larger of iid and date SE), scaled by √(n_test / n_W).

| Cell | Scored rows | Planning SE | MDE80 | Power at ΔBrier .001 / .002 / .003 / .005 | MDE80 if sd(ΔP) doubles |
|---|---|---|---|---|---|
| NFL receptions | 1,732 | .00034 | .0010 | .83 / 1.00 / 1.00 / 1.00 | .0019 |
| NFL carries | 639 | .00021 | .0006 | 1.00 / 1.00 / 1.00 / 1.00 | .0012 |
| NHL shots | 4,077 | .00020 | .0006 | 1.00 / 1.00 / 1.00 / 1.00 | .0011 |
| NHL points | 3,927 | .00051 | .0014 | .50 / .98 / 1.00 / 1.00 | .0028 |
| NFL yards | 2,088 (all) | .00195 | .0055 | .08 / .18 / .34 / .73 | .0109 |

For scale, the live deficit is about .0105 Brier on receptions (BSS −.043 × .245) and about .020 on carries. E1 can see 5–10% of it.

**Cost (inferred).**
- About a day of worktree code:
  - a fraction override and a date cutoff;
  - a θ pin that hands the pickle's parameters straight to LightGBM, following the hurdle branch at `pipeline.py:1603–1620`;
  - the C file swap;
  - a W dump.
- Then 1–3 hours of compute for 15 arm fits plus 5 reproduction fits.
- The existing `--frozen-matrix-dir` path runs full HPO: 300 trials and 60 minutes per fit (`pipeline.py:1622–1634`).

### 3.3 E2: gate optimism (H2; descriptive)

| | Served verdict (`ship=True` on all 70) | Today's `--deterministic --holdout-blind` | Pre-registered E2 |
|---|---|---|---|
| Booster | Full-HPO θ, first 70% | `DETERMINISTIC_FIXED_PARAMS` | **Served** booster: load the pickle's `model`, or a θ_served retrain verified within 1e-6 |
| Rows | Test half | Validation half (test half absent) | V = validation half; T = test half for the symmetric read |
| Calibrators | Fit once on the validation half | 5-fold player-disjoint, out-of-fold over V | Out-of-fold over V; out-of-fold over T for the symmetric read |
| Prior use of these rows | Walk ships the first nominee to pass on T, plus Gate-4 retries, supersede comparisons and manual flips | The sweep ranks corners here | The sweep ranked a different booster on V |

- **Population.** The 70 served cells (`shipped != "withheld"`): NFL 18, NBA 15, MLB 14, WNBA 12, NHL 11.
- **Reads (all descriptive):**
  1. **Flip count.** Number of cells with ship(V) = False, broken down by gate and league.
  2. **Symmetric slack shift.** Per cell and gate, slack(T, cross-fit) − slack(V, cross-fit). Report the median and the count with T > V. Without selection, about half would have T > V. Report the binomial reference with "descriptive" written next to it.
  3. **Gate-1 CI method on both halves.** Upper bounds under iid, player, date and two-way pigeonhole CIs; count of cells crossing .005.
  4. **Forward arm (amendment).** E1-A's own test-half gates vs its W gates, for 5 cells.
- **Pre-specified flags:**
  - NFL count cells: Gate 1 faces a broken opponent on both halves (Finding 1). Report their Gate 1 on 2026-season rows as well.
  - MLB and WNBA keep the outcome-band trim until their 2027 preseason rebuilds (nfl-ship15-recovery §11), so read them as a separate trimmed population.
  - The 31 cells with no Gate 1 cannot be read on Gate 1.
- **Why no test enters Holm:**
  - adoption of the blind walk is decided regardless;
  - both halves have been looked at;
  - cells within a league share players and dates;
  - the only decision left (owner ask 3) is a policy call.

### 3.4 E3: same-rows decomposition, and the H3 decision

**Join.**
- Live side: `settled_offers(read_history())`, NFL and NHL.
- Feature log: read each `date=*/…parquet` with its own column list (schemas differ per file, so reading the whole directory raises `ArrowInvalid`).
- Join history to the feature log on (League, Market = model cell, Platform, Player, Date).
  - For NBA/WNBA, map history's `… underdog` market to the cell's `… prizepicks` by cell name, never through `normalize_market`.
  - Many history rows (lines, rungs, sides) join to one feature-log row.
- Then join to the cell's frozen matrix on (Player, Date). The NFL matrix ends 10-15; NHL ends 10-17.
- Feasibility: 7,041 live NFL rows from 10-05 → 10-20, of which 6,981 have a feature-log row. Rows that also have a matrix row:

  | Cell | Joined rows | of which quoted |
  |---|---|---|
  | receiving yards | 955 | 923 |
  | receptions | 873 | 856 |
  | rushing yards | 479 | 467 |
  | carries | 306 | 261 |
  | yards | 206 | 173 |
  | tds | 163 | 158 |
  | attempts | 162 | 155 |
  | passing tds | 154 | 150 |
  | completions | 120 | 120 |
  | qb yards | 112 | 109 |
  | interceptions | 107 | 105 |

  These rows span **only 3–5 game dates**.

**Populations.**
- Q: quoted main lines (`Quote == "quoted"`, `Alt Line` not True).
- A: alt rungs (`Alt Line == True`).
- U: unquoted (`Quote == "unquoted"`).
- Excluded: `fallback` (the book-priced path).

**(a) H3, the decision.**
- **Cells.** Served NFL count-stat cells with ≥ 100 priced held-out rows offline and ≥ 100 Q rows live: attempts, carries, completions, interceptions, passing tds, receptions, tds.
- **Offline rows.** The served split's priced held-out rows (validation and test halves).
  - P_book,over = matrix `Odds` (training convention: over probability, `pipeline.py:932`).
  - The test CSV stores 1 − that (`:2642`).
- **Live rows.** Q rows from 2026-09-21 to the last settled date at run time.
  - P_book,over = `Market Prob` if Bet == Over, else 1 − `Market Prob` (it is the chosen side's probability).
  - Pushes are void.
- **Statistics.** CITL per cell and source. L = Σ n_c |CITL_c| / Σ n_c.
- **Rule.** H3 holds if L_off > .05, L_live ≤ .02, and T1 rejects.
  - Per-cell versions of the rule set I6d's scope.
  - If H3 fails, the count cells route to H4.
  - L_live is biased upward by noise in small cells (about .003–.005), which makes the ≤ .02 leg conservative.

**(b) Model-P parity (descriptive).**
- ΔP_raw = `over_probability` with serve-time features minus `over_probability` with matrix features. Use the same file: the current 20261018 pickle for the 7 feasible NFL cells, each row's own version for NHL. Same `Line`, same `Hist Gate`.
- ΔP_served = the same through the full serving chain, at the DFS line with the logged book leg (R3's `parity_lib.serve`).
- Report mean, sd and RMS with player-clustered CIs. There is no date inference on 3–5 dates.

**(c) Book-leg parity on the joined rows (descriptive).** Compare three book probabilities, each against outcomes at its own line:
- matrix `Odds` at the matrix `Line`;
- logged `Quote Over Prob` at `Quote Line`;
- `Market Prob` at the DFS line.

**(d) Population split (descriptive).** Live BSS of `Win Prob` vs `Market Prob` by Q / A / U over the full window, with date-block CIs. Also the recommended-leg gap per population.

**(e) Label (descriptive).** Push share per population: offline scores a push as ½, live voids it.

### 3.5 E4: disagreement-band gates (decides H4)

- **Bands.** |P − P_book| with right-closed edges `[-1e-9, .02, .05, .10, 1.0]`, identical to the live table.
  - Offline P = test-set `P` (the served chain, over side).
  - P_book = 1 − `Odds`.
  - y is push ½.
- **BSS per band** = 1 − Σ(P − y)² / Σ(P_book − y)², the `_brier_skill_score` convention (`scorecard.py:599–613`). Use the delta-method SE (B17) as a cross-check.
- **Primary population (amendment).** Priced test rows of served NFL priced cells where the opponent is levelled, chosen on prior knowledge rather than on any band result:
  - every such cell's rows dated ≥ 2026-09-01;
  - receiving yards and rushing yards rows from every season. The memory note `nfl_count_quote_columns_disagree` (2026-08-31) already established these as levelled.
  - Counted blind, the bands hold 1,762 / 2,275 / 948 / **356** rows.
- **Rule.** H4 holds if T2 rejects (BSS of the > .10 band < 0, Holm-adjusted). Then propose the band read as a monitor column in an owner packet; never a gate. Otherwise the live loss is a population shift, and E3 carries the verdict (Finding 2 is the leading candidate for the count cells).
- **Descriptive extras:**
  - whether the offline two-sided CI overlaps the live −.115 [−.164, −.062];
  - strata: count 2026 (205 band rows) vs yardage (151);
  - the literal population (all priced test rows; 3,611 band rows), expected positive because of H3;
  - NHL served cells (no live counterpart).

### 3.6 E5: sanity read of the parity alarm

| Cell | w | T | sd ΔP served / raw (paired 10-15 → 10-18) | Ratio | w/T |
|---|---|---|---|---|---|
| NFL receptions | .824 | 4.93 | .0140 / .0558 | .25 | .17 |
| NFL carries | .861 | 4.66 | .0053 / .0283 | .19 | .18 |
| NHL shots | .900 | 1.22 | .0123 / .0148 | .83 | .74 |
| NHL points | .642 | 1.02 | .0266 / .0445 | .60 | .63 |
| NFL yards | .05 (1.0 on unpriced rows) | 1.24 | .0743 / .0635 | 1.17 | — |

1. The alarm reads the raw P. That is a conservative bug detector: it flags skews that a later re-fit T could unmask. Keep 1 pp on raw RMS as the trigger, and **rank repairs by served RMS × live volume**.
2. sd cannot see a level shift. A 2 pp uniform shift has sd ≈ 0 and moves legs across the 5% edge bar. Flag on RMS = √(mean² + sd²), or on |mean| as well as sd, with a player-clustered CI on the mean.
3. Sampling noise: SE(sd) ≈ sd/√(2(n − 1)). That is ±12% at n = 36 and ±2.4% at n ≈ 870. Read "above 1 pp" from the CI's lower bound.
4. NFL needs the fixed-file mode (current pickle on both frames). `check_version` scores through each row's own file, and `--versions` compares versions on the logged frame only. R3's `c16_parity.py` already does the fixed-file read.

### 3.7 Threats to validity and pre-specified exclusions

| Threat | What it touches | Handling, fixed in advance |
|---|---|---|
| Version mixing since the 10-18 sends: band rows span 0917–1018, and the 10-18 T values (receptions 4.93) flatten P | E3 and E4 live comparators | H3's live leg (book level) does not depend on the version. Report the band pooled and by version. No live read of the 10-18 versions before ≥ 5 game days. The 11-17 Gate-2 read is the control. |
| Quote-class relabel, Aug → Oct (pickem class) | E1, E2, E4 | Current matrices only, with the hash checked: all 70 served pickles match, and all five leagues now carry pickem rows. Never mix test sets dumped before the relabel. |
| Trimmed vs untrimmed (lever T rebuilt NFL, NBA and NHL untrimmed on 10-18; MLB and WNBA stay trimmed until 2027) | E2; any read that pools across the 10-18 boundary | MLB and WNBA reported separately. Paired shared-row reads only across lever T. |
| Push label (offline ½ since `33ce9b28`; live void) | E3, E4 cross-reads | Offline primary uses ½. Also report an offline variant without pushes, and the push shares. |
| Matrix window slides at each rebuild; the served split is bound to `matrix_hash` | E1, E2 | Freeze matrices and `stat_calibration.json`. Abort a cell if its hash moved. |
| A moved line counted N times (games before 10-09) | Live reads | `settled_offers` counts once from 10-09. The band result holds on games ≥ 10-09 alone (−.109, 815 rows). H3's live leg is reported on both windows. |
| Underdog boost 1.78 vs 1.83 | Selected gaps, tail scorecard | Brier, BSS and CITL do not use price. Selection reads use per-row era baselines (`tail_scorecard.py:69–70`) and are never pooled across eras. |
| Historical count-quote mis-levelling | E1 absolute BSS, E2 Gate 1, E4 literal population | E1's primary statistic needs no book. E2 flags the affected cells. E4's primary population excludes them. |
| The served corner was chosen by a walk that read W's rows | E1 absolute gates, E2 forward arm | All arms share the corner, so the contrast is biased toward A (conservative for H1). Forward gates are not fresh. |
| HPO noise (unseeded TPE; isolated runs repeat full HPO, `pipeline.py:745`) | E1, E2 | θ pinned; reproduction check. |
| Unstable date sort (quicksort, `pipeline.py:1122`) | E1 | W is cut at a date boundary. Inside P, production's own cut is left as it is. |
| Few clusters (NFL W: 14 weeks; E3 join: 3–5 dates) | E1 NFL CIs, E3 | Date-block primary with week sensitivity. E3 uses player clusters only. If a verdict hinges on ≤ 15 clusters, use a wild cluster bootstrap-t (Webb weights) as the tie-break (B22, B23). |
| Feature-log retention (30 days) and test-set overwrite at retrain | E3, E5, E2, E4 | Run before any retrain, and snapshot `data/test_sets/`. Dev keeps what it has synced. |
| Exclusions | All | Rows with missing `Result`, `Line` or (where used) `Odds`; class filters as stated above. No outlier trimming, no exclusion based on outcomes, no band or window changes after results are seen. |

## 4. Reality checks

- **H3 / I6d: a large effect, and the fix route is mostly engineering.**
  - The cheapest candidate is a calibrator-only refit: fit w and Gate 1 on rows with a levelled opponent. Each cell keeps its booster and its other constants.
  - What it buys (inferred): on the 11 cells, w should fall toward the .32–.64 seen on levelled cells. That moves the served P toward the book exactly where the edge rule selects, and recommended volume on those cells drops. That is the owner's trade-off.
  - Gate 1 re-graded on levelled rows rests on 150–1,400 rows per NFL cell, and some cells may fail it. A shipped verdict that rests on Gate 1 against a broken book is not evidence of skill.
  - Re-levelling the archive itself is a research project; the root cause is unknown.
- **H1.**
  - The prior is low, inferred at < 25%. NFL booster information is ≈ 0 (R3), and the chain damps booster moves to about a fifth on NFL count cells. A pass needs 3 of {receptions, carries, shots, points}.
  - Even a pass is worth about .001 Brier, around 10% of the receptions live deficit.
  - Run it anyway: about a day of harness work plus hours of compute settles the "stale booster" question.
- **H4.** Power is .65–.85 only for the live-sized effect. A null offline does not clear the band; on count cells H3 is the shift.
- **E2.** The flip count mixes selection with half-to-half noise. The symmetric slack shift isolates selection only partly, because the validation half was also looked at.
- **What could make this brief wrong:**
  - A CITL difference between offline and live could come from the outcome definition (lines, pushes) rather than the producer. E3 (c) checks this on the same rows.
  - The 2026 levelling might be the median-line structure (book prices near even at the book's own line) rather than a fixed producer.
  - The confirmatory H3 window is longer than the preview's.

## 5. Open questions and caveats

1. **Root cause of each mis-levelled book leg.** Candidates:
   - NFL count rows from 2022–2025 (historical backfill vs live polling);
   - NBA BLK, STL, FG3M and BLST in the 2025-26 held-out rows;
   - WNBA FG3M.

   For each, check `under_prob` vs `ev` vs `get_odds` by book and season (memory `nfl_count_quote_columns_disagree`; R3 §9 #12).
2. **MLB home runs: both producers are off, in opposite directions.**
   - Training `Odds` from 08-01: +.117 on 583 rows.
   - Live `Market Prob` over 08-01 → 10-05: **−.115 [−.155, −.075]** on 425 rows over 56 dates.
   - MLB hits is levelled (−.001 on 18,055 rows); stolen bases is +.041 [−.017, .096].
   - Re-levelling leaves home runs' offline BSS at .154, so the problem is more than level.
3. **Live rushing yards `Market Prob`** over-claims the Over by +.060 [.035, .093]. Check whether the DFS-line decode is at fault, using E3 (c).
4. **NBA live book level is unknown.** BLK, STL and FG3M have 16–60 rows on one date. Read again after about 3 weeks.
5. **Gate 1's CI method** (iid ships; two-way flips 4–5 of 39) is an owner question. It is not in H2's rule.
6. **The blend-weight cap of .90 binds** with levelled books too (NBA RA, REB). A w refit on levelled rows is a necessary test, not a sure fix.
7. **E1 tests one origin.** The second origin (80–90%) is the cheapest robustness read.
8. **NHL:** held-out |CITL| ≤ .025 everywhere, so NHL is clean. E1's NHL cells are the cleanest test of H1.

## 6. Bibliography

| # | Source | Identifier |
|---|---|---|
| B1 | Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scand. J. Statist.* 6(2):65–70 | JSTOR 4615733 |
| B2 | Cawley, G. C. & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *JMLR* 11:2079–2107 | jmlr.org/papers/v11/cawley10a.html |
| B3 | Varma, S. & Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics* 7:91 | doi:10.1186/1471-2105-7-91 |
| B4 | Dwork, C., Feldman, V., Hardt, M., Pitassi, T., Reingold, O. & Roth, A. (2015). The reusable holdout: Preserving validity in adaptive data analysis. *Science* 349(6248):636–638 | doi:10.1126/science.aaa9375 |
| B5 | Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: an analysis and review. *IJF* 16(4):437–450 | doi:10.1016/S0169-2070(00)00065-0 |
| B6 | Hyndman, R. J. & Athanasopoulos, G. *Forecasting: Principles and Practice*, 3rd ed., §5.10 Time series cross-validation. OTexts | otexts.com/fpp3/tscv.html |
| B7 | Hewamalage, H., Ackermann, K. & Bergmeir, C. (2023). Forecast evaluation for data scientists: common pitfalls and best practices. *Data Min. Knowl. Discov.* 37(2):788–832 | doi:10.1007/s10618-022-00894-5; arXiv:2203.10716 |
| B8 | Bergmeir, C., Hyndman, R. J. & Koo, B. (2018). A note on the validity of cross-validation for evaluating autoregressive time series prediction. *CSDA* 120:70–83 | doi:10.1016/j.csda.2017.11.003 |
| B9 | Roberts, D. R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* 40(8):913–929 | doi:10.1111/ecog.02881 |
| B10 | Pesaran, M. H. & Timmermann, A. (2007). Selection of estimation window in the presence of breaks. *J. Econometrics* 137(1):134–161 | doi:10.1016/j.jeconom.2006.03.010 |
| B11 | Giacomini, R. & White, H. (2006). Tests of conditional predictive ability. *Econometrica* 74(6):1545–1578 | doi:10.1111/j.1468-0262.2006.00718.x |
| B12 | Diebold, F. X. & Mariano, R. S. (1995). Comparing predictive accuracy. *JBES* 13(3):253–263 | doi:10.1080/07350015.1995.10524599 |
| B13 | Harvey, D., Leybourne, S. & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *IJF* 13(2):281–291 | doi:10.1016/S0169-2070(96)00719-4 |
| B14 | Qu, R., Timmermann, A. & Zhu, Y. (2024). Comparing forecasting performance with panel data. *IJF* 40(3):918–941 | doi:10.1016/j.ijforecast.2023.08.001 |
| B15 | Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA* 102(477):359–378 | doi:10.1198/016214506000001437 |
| B16 | Murphy, A. H. (1973). A new vector partition of the probability score. *J. Appl. Meteorol.* 12(4):595–600 | doi:10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2 |
| B17 | Bradley, A. A., Schwartz, S. S. & Hashino, T. (2008). Sampling uncertainty and confidence intervals for the Brier score and Brier skill score. *Wea. Forecasting* 23(5):992–1006 | doi:10.1175/2007WAF2007049.1 |
| B18 | Van Calster, B. et al. (2019). Calibration: the Achilles heel of predictive analytics. *BMC Medicine* 17:230 | doi:10.1186/s12916-019-1466-7 |
| B19 | Cameron, A. C., Gelbach, J. B. & Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. *REStat* 90(3):414–427 | doi:10.1162/rest.90.3.414 |
| B20 | Cameron, A. C., Gelbach, J. B. & Miller, D. L. (2011). Robust inference with multiway clustering. *JBES* 29(2):238–249 | doi:10.1198/jbes.2010.07136 |
| B21 | Owen, A. B. (2007). The pigeonhole bootstrap. *Ann. Appl. Stat.* 1(2):386–411 | doi:10.1214/07-AOAS122 |
| B22 | MacKinnon, J. G. & Webb, M. D. (2017). Wild bootstrap inference for wildly different cluster sizes. *J. Appl. Econometrics* 32(2):233–254 | doi:10.1002/jae.2508 |
| B23 | MacKinnon, J. G., Nielsen, M. Ø. & Webb, M. D. (2023). Cluster-robust inference: A guide to empirical practice. *J. Econometrics* 232(2):272–299 | doi:10.1016/j.jeconom.2022.04.001; arXiv:2205.03285 |
| B24 | Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *Ann. Statist.* 17(3):1217–1241 | doi:10.1214/aos/1176347265 |
| B25 | Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Comput.* 10(7):1895–1923 | doi:10.1162/089976698300017197 |
| B26 | Nadeau, C. & Bengio, Y. (2003). Inference for the generalization error. *Mach. Learn.* 52(3):239–281 | doi:10.1023/A:1024068626366 |
| B27 | Bouthillier, X. et al. (2021). Accounting for variance in machine learning benchmarks. *MLSys* 3 | arXiv:2103.03098 |
| B28 | Smith, J. E. & Winkler, R. L. (2006). The optimizer's curse. *Mgmt. Sci.* 52(3):311–322 | doi:10.1287/mnsc.1050.0451 |
| B29 | Clements, M. P. & Harvey, D. I. (2010). Forecast encompassing tests and probability forecasts. *J. Appl. Econometrics* 25(6):1028–1062 (I6d acceptance) | doi:10.1002/jae.1097 |
| B30 | Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M. & Bouchachia, A. (2014). A survey on concept drift adaptation. *ACM Comput. Surv.* 46(4):44 | doi:10.1145/2523813 |

Verification: B2–B23, B25–B27 and B30 were checked today by search or by doi.org resolution. B1, B24, B28 and B29 are carried over from R3's verified bibliography.

## Appendix A: data-in-hand checks

Run each from the repo root as `.venv/bin/python - <<'EOF' … EOF`. Each starts with:

```python
import json, hashlib, pickle, numpy as np, pandas as pd, pyarrow.parquet as pq
from pathlib import Path
D = Path("src/sportstradamus/data"); rng = np.random.default_rng(1729)
```

**A1. Served inventory and matrix-hash equality** (70 / 70)

```python
from sportstradamus.helpers.io import model_pickle_path
sm = json.load(open(D/"config/stat_meta.json"))
served = [(lg, m) for lg, c in sm.items() for m, v in c.items() if isinstance(v, dict) and v.get("shipped") != "withheld"]
print(len(served), sum(pickle.load(open(model_pickle_path(lg, m), "rb"))["model_strategy"]["matrix_hash"]
      == hashlib.sha256((D/f"training_data/{lg}_{m.replace(' ', '-')}.parquet").read_bytes()).hexdigest() for lg, m in served))
```

**A2. E1 windows** (the §3.2 table)

```python
for lg, m in [("NFL","receptions"),("NFL","yards"),("NFL","carries"),("NHL","shots"),("NHL","points")]:
    M = pd.read_parquet(D/f"training_data/{lg}_{m}.parquet", columns=["Player","Date","QuoteAuthenticity"])
    M["Date"] = pd.to_datetime(M["Date"]); M = M.sort_values("Date", kind="mergesort"); n = len(M)
    dW = M.Date.iloc[int(0.9*n)]; W, P = M[M.Date >= dW], M[M.Date < dW]
    print(lg, m, n, M.Date.iloc[int(0.7*n)].date(), dW.date(), len(W), W.Date.nunique(), W.Player.nunique(),
          P.Date.iloc[int(0.7*len(P))].date(), P.Date.iloc[int(0.85*len(P))].date())
    print(pd.crosstab(np.where(W.Date.dt.month >= 8, W.Date.dt.year, W.Date.dt.year - 1), W.QuoteAuthenticity))
```

**A3. E1 power proxy.** Uses `BK = Path.home()/"backups/sportstradamus/2026-10-17-train-live-gap"` and the `dW` values from A2. For each cell:
- merge `after/test_sets_all/{cell}.csv` with `before/…` on Player, Date, Line and Result;
- keep Date ≥ d_W, priced rows (non-priced for yards);
- d = (P_after − y)² − (P_before − y)²;
- report the sd of ΔP on P and on P_standalone, and SE(mean d) as iid, date-block and player-block (2,000 draws).

**A4. Gate-1 CI methods** (31 blank; 39 / 39 / 38 / 34–35). For each served test CSV:
- take priced rows with P and Odds present, skipping cells with fewer than 10 players;
- d = (P − y)² − ((1 − Odds) − y)²;
- report the 97.5th percentile of the bootstrap mean under four schemes: iid, player blocks, date blocks, and pigeonhole (independent multinomial weights on dates and on players, row weight = product), 1,000 draws each;
- count cells with that percentile < .005.

**A5. H3 preview** (Finding 2)

```python
from sportstradamus.helpers.io import read_history
from sportstradamus.realized import settled_offers
so = settled_offers(read_history()); so["Date"] = pd.to_datetime(so["Date"])
L = so[(so.League=="NFL") & (so.Date>"2026-09-20") & (so.Date<="2026-10-20") & (so.Quote=="quoted") & (so["Alt Line"]!=True)].copy()
L["pbo"] = np.where(L.Bet=="Over", L["Market Prob"], 1-L["Market Prob"]); L["yo"] = np.where(L.Bet=="Over", L.Hit, ~L.Hit.astype(bool)).astype(float)
print(L.groupby("Market").apply(lambda f: pd.Series({"n": len(f), "citl": (f.pbo-f.yo).mean()}), include_groups=False))
for m in ["receptions","carries","attempts","completions","interceptions","passing tds","tds"]:
    t = pd.read_csv(D/f"test_sets/NFL_{m.replace(' ', '-')}.csv", usecols=["Result","Line","Odds","QuoteAuthenticity"])
    a = t[t.QuoteAuthenticity=="authentic"]; y = (a.Result > a.Line) + 0.5*(a.Result == a.Line)
    print(m, len(a), ((1 - a.Odds) - y).mean())   # CIs: date-block bootstrap of the same mean
```

**A6. Held-out level by season and the cross-league association** (Findings 1 and 2). For each served cell:
- sort the matrix by date (mergesort) and take the 0.7 cut;
- take priced rows dated ≥ the cut;
- CITL = mean(Odds − y), using matrix `Odds` as the over probability, by season and overall;
- join `model_stats.parquet` on (league, market) for `brier_skill_score` and `model_weight`;
- `scipy.stats.spearmanr(|CITL|, bss)` over cells with ≥ 100 rows.

**A7. Offline BSS vs the re-levelled book** (Finding 1)

```python
t = pd.read_csv(D/"test_sets/NFL_interceptions.csv", usecols=["Result","Line","Odds","P","QuoteAuthenticity"])
a = t[t.QuoteAuthenticity=="authentic"]; y = (a.Result > a.Line) + 0.5*(a.Result == a.Line); pb = 1 - a.Odds; dl = (pb - y).mean()
print(1 - ((a.P-y)**2).mean()/((pb-y)**2).mean(), dl**2/((pb-y)**2).mean(), 1 - ((a.P-y)**2).mean()/((pb-dl-y)**2).mean())
```

**A8. Live band table, CI and composition** (Finding 3). Use L from A5:
- y = Hit;
- band = `pd.cut(|Win Prob − Market Prob|, [-1e-9, .02, .05, .10, 1.0])`;
- BSS as a ratio of sums;
- date-block CI by resampling per-date sums (4,000 draws);
- split by count vs other cells, by `Model Version`, and by Date ≥ 10-09.

**A9. Blind E4 counts.** Over served NFL priced cells' test rows:
- band = `pd.cut(|P − (1 − Odds)|, …)`;
- tabulate by count/yardage group × (Date ≥ 2026-08-30) × band;
- report row, date and player counts only, without reading outcomes.

**A10. E3 join feasibility.**
- Feature log: read `src/sportstradamus/data/runtime/feature_log/date=*/NFL_*.parquet`, one file at a time with columns taken from `pq.read_schema`.
- Join `settled_offers` rows (NFL, Date 10-05 → 10-20) on League, Market, Platform, Player and Date.
- Then join each cell's matrix on (Player, Date); count rows, dates and quoted rows.

**A11. Cross-league live level** (Open question 2). Same as A5 with:
- League MLB, Market `home runs`, `stolen bases` or `hits`, Date 08-01 → 10-05;
- matrix side: `MLB_home-runs.parquet`, priced rows with Date ≥ 08-01, mean(`Odds` − y).

**A12. Offline Gate-1 and w.** From `model_stats.parquet`, select columns `brier_skill_score`, `model_weight`, `n_authentic_validation`, `g1_brier_diff_mean` and `g1_brier_diff_ci_hi` for the NFL cells.

---

## Load-bearing conclusions for the plan

1. **The H3 mechanism is measured. Offline Brier skill against the book is the book's level error on 11 served cells** (ρ = .75 over 38 cells; re-levelled BSS ≤ .052 on 10 of the 11). The pinned statistic previews at L_off .102 vs L_live .017.
   → `docs/handoffs/live-gap-2.md` §2 (new fact 6) and the §3 H3 row (pinned statistic; expected verdict "holds; I6d opens").
2. **I6d's scope becomes book-leg levelling in every league:**
   - NFL count rows 2022–2025;
   - NBA BLK, STL, FG3M, BLST;
   - MLB home runs (training +.117 vs live −.115);
   - WNBA FG3M.

   Candidates to pre-register there:
   - (a) refit w and Gate 1 on levelled rows only (calibrators only, no booster retrain);
   - (b) re-level or relabel mis-levelled archive rows as `derived`;
   - (c) a per-cell book-CITL monitor column.

   → `docs/handoffs/honest-receipts.md` §6 I6d, `live-gap-2.md` §5 (week 2) and the §7 row "Count-quote producer".
3. **H4 population amendment.** Primary population = priced test rows where the opponent is levelled (2026-season rows plus receiving and rushing yards from all seasons; 356 band rows). The literal population is secondary. Power .65–.85 for −.115.
   → `live-gap-2.md` §3 H4 row and §4.3 E4.
4. **Holm family m = 2** (T1 H3, T2 H4; one-sided; FWER .05). H1 is decided by its count rule (false adopt 1.5 × 10⁻⁴). E2, E3 (b)–(e) and E5 are descriptive.
   → `live-gap-2.md` §3, as a note under the table.
5. **E1 design.**
   - Design: arms trained before d_W, scored on W, θ_served pinned, a blocking reproduction check.
   - **Decision arm C (amendment)**; B is descriptive.
   - Clustering: date-block primary, week-block sensitivity.
   - Yards scored on all W rows.
   - MDE80 .0006–.0014 on priced cells (yards .0055).
   - R3 prior: likely null on NFL.

   → `live-gap-2.md` §3 H1 row and §4.3 E1. A pending entry for "train fraction / refit" goes into `docs/handoffs/model_improvement_track.md` §3.4.
6. **E2 correction.** Today's `--holdout-blind` requires `--deterministic` and trains `DETERMINISTIC_FIXED_PARAMS` (`cli.py:168–171`, `pipeline.py:1601–1602`). Re-gate the served booster with cross-fit calibrators. The reads are descriptive:
   - flip count;
   - symmetric slack shift;
   - Gate-1 CI method;
   - E1-A's forward gates.

   NFL count cells' Gate 1 is flagged, and MLB/WNBA are read as a trimmed population.
   → `live-gap-2.md` §4.3 E2 and §8 ask 3.
7. **E5 alarm.** It reads only the sd of the raw P (`feature_parity.py:157–163`, `:238`). Flag on RMS (or |mean| as well as sd), rank repairs by served RMS (≈ (w/T) × raw) × volume, and use fixed-file current-pickle parity for NFL.
   → `live-gap-2.md` §4.3 E5 and `honest-receipts.md` §6 I6e.
8. **Research holes:**
   - root cause of each mis-levelled producer;
   - the MLB home-runs scale problem beyond level;
   - the live rushing-yards `Market Prob` at +.060;
   - whether the w cap of .90 binds independently of level.

   → `model_improvement_track.md` §8.2.
9. **Cross-league caveats:**
   - NHL book legs are levelled (|CITL| ≤ .025), so NHL is the cleanest H1 test;
   - NBA live book level is unreadable for about 3 weeks;
   - MLB and WNBA stay trimmed until 2027, so their gates are read separately;
   - NFL has only 14 week-clusters.

   → `model_improvement_track.md` §7.4.
10. **New owner ask: whether Gate 1's ship CI becomes date-clustered or two-way.** Today 4–5 of 39 cells flip under two-way. NFL yards' Gate 1 rests on 19 rows (date-clustered upper bound +.022).
    → `live-gap-2.md` §8 (a new ask). This is not part of H2's rule.
