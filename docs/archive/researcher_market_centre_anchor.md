# Market centre vs evaluation on the recommended pick'em legs

In-repo research brief (research-analyst), 2026-10-29. It asks what should change in the offline evaluation and in the served probability so that offline predicts live on the recommended DFS pick'em legs, in what order, and under which pre-registered test.

## TL;DR

- **The REPORT's "edge only where the market is thin" is wrong; the axis is market line vs non-market line.**
  - The calibrated "not posted" tail is mostly the model graded against lines it built itself: `max(Avg10, 0.5)` and lines inverted from expected values. NBA has zero unposted sportsbook-priced October test rows.
  - At a market median, the model's disagreement with the book carries no information in NBA (encompassing slope β −.11 [−.29, +.11]), a small and unstable amount in NFL, and real information in NHL (+.67 [+.39, +.94]).
  - Live hit rates on recommended 1.83 legs follow the same order: NBA .508, NFL .509, NHL .545.
  - The loss is the optimizer's curse (Smith & Winkler 2006, doi:10.1287/mnsc.1050.0451).
- **Evaluation.**
  - Replace the posted-flag gate with a *market-median tail gate* built from existing test-set columns.
  - It ranks the leagues the way live does, but it reads 4–7.5 pp high, so it may only screen cells out.
  - Importance weighting: KILL. The problem is concept shift, not covariate shift.
- **Serving.**
  - DFS-line centre anchor: KILL. The owner ruled it out on 10-04, and in NBA it would fit to near-full strength, which amounts to abstaining.
  - The principled lever that stays within the rules is a blend weight fitted on near-even sportsbook rows (`w_ne`). It is research first and reaches only quoted legs, 45% of the recommended legs.
- **Priced legs leak more than the 1.83 legs.** Since 10-09 the recommended priced legs (2,418, twice the 1.83 count) returned −9.7%, against −4.6% for the 1.83 legs. The record gate must cover them with an ROI bound.
- **Order.**
  1. L1, Part 5's live-record gate, built as specified (fixed-sample 90% bound) and checked weekly. It is the cheapest lever and cannot leak, but it releases nothing today. 1–1.5 engineering days.
  2. L2, the market-median screen. 1.5–2 engineering days.
  3. L3, the `w_ne` prototype. 0.5–1 research day.

## Key Findings

1. **The calibrated "not posted" tail was graded at non-market lines.**
   - Composition of the October test half (served test sets, posting flags from `history.parquet`):
     - NBA's unposted rows are 4,124 synthetic, 1,023 derived, 18 pick'em and 0 sportsbook-priced.
     - NHL's are 9,559 synthetic.
   - The synthetic line is `max(Avg10, 0.5)` priced at .5 (`stats/base.py:2360`; `training/pipeline.py:892–948`).
   - Within the posted-1.83 NBA player-games, the standalone tail (q ≥ .574) gap depends on which line a row carries:
     - synthetic lines: .000 (n 116);
     - derived lines: −.016 (176);
     - sportsbook lines: −.130 (126);
     - pick'em lines: −.146 (204).
   - NFL's unposted sportsbook rows are lopsided: the book gives the model's side .867.
   - Calibration on a pool does not carry to a subgroup the selector can identify (Hébert-Johnson et al. 2018, arXiv:1711.08513).
2. **At a market median the disagreement is mostly noise.**
   - The test regresses hit minus b on q minus b, where b is the book's probability on the model's side, over all sportsbook rows within .06 of even. Intervals are date-block bootstrap at 90% (Clements & Harvey 2010, doi:10.1002/jae.1097).
   - Standalone β:
     - NBA: −.11 [−.29, +.11] (3,081 rows, 71 dates).
     - NFL: +.14 [+.01, +.28] before 2026, −.19 [−.47, +.04] in Jan–Sep 2026, +.33 [+.14, +.47] in October.
     - NHL: +.67 [+.39, +.94].
   - Matched October legs (posted 1.83, near-even price): the tail gap grows with disagreement while hit minus book stays flat at about +.01 to +.03. This is the winner's curse (Milgrom & Weber 1982, doi:10.2307/1911865).

     | Disagreement | Tail gap |
     |---|---|
     | .05–.09 | −.045 |
     | .09–.14 | −.080 |
     | > .14 | −.212 |

   - For a typical recommended leg (claimed .615) to break even at 1.83, a cell needs β ≈ .046 / .115 ≈ .40.
3. **The fused weight is too high at near-even prices in NBA.**
   - Fused β on near-even rows:
     - NBA: +.36 [+.08, +.62], so the model is weighted about 3× too heavily there.
     - NHL: +.96.
     - NFL: +1.04 before 2026, −.05 in Jan–Sep 2026, +1.29 in October.
   - w is one number per cell, fitted on all sportsbook validation rows (`training/pipeline.py:3541`; `training/calibration.py:678–692`). Lopsided rows dominate that fit, and on them model and book agree.
   - The established fix is weights that change with the regime (Deutsch, Granger & Teräsvirta 1994, doi:10.1016/0169-2070(94)90049-3; Elliott & Timmermann 2005, doi:10.1111/j.1468-2354.2005.00361.x).
4. **Mechanisms (Q3).**

   | Mechanism | Separating read | Verdict |
   |---|---|---|
   | M5. Selection on the largest disagreements at a median line | Gap grows with disagreement; hit minus book flat (F2) | **Supported** |
   | M1. Liquid markets are more efficient than thin ones | It needs thin-market median lines. October has about 20 unposted near-even tail rows. | **Untested.** Prior from game lines only (Krieger & Davis 2024, doi:10.1007/s12197-023-09656-5) |
   | M2. Late news moved the line after the features were cut | Gap −.186 where Underdog's line equals the noon line, −.152 where it differs | **Contradicted** as the main cause |
   | M3. Noon book leg vs tip-time book leg | The standalone P carries the gap with no book leg in it | **Contradicted**; M3 can move only w's level |
   | M4. Outcome-band trimming biases stars | The pickles are post-lever-T. Overs fail too: Overs −.132, Unders −.191 on NBA 1.83 legs. Unposted high-line rows are calibrated: NBA −.001, NFL +.005. | **Contradicted** |
   | M6. The platform posts even money where its own median is sharp (Glosten & Milgrom 1985, doi:10.1016/0304-405X(85)90044-3) | NFL near-even β: +.46 [+.11, +.82] on priced-only lines vs +.18 [−.03, +.44] on posted-1.83 lines | **Exploratory lead** |

5. **Why not importance weighting or a posted flag.**
   - The miss sits in P(hit | q, line type) at the same features. That is concept shift; importance weighting fixes only a shift in P(x) (Shimodaira 2000, doi:10.1016/S0378-3758(00)00115-4; Sugiyama et al. 2007, JMLR 8:985–1005; Moreno-Torres et al. 2012, doi:10.1016/j.patcog.2011.06.019).
   - The weights would also be extreme: near-even sportsbook rows are 277 of 7,375 NBA October test rows.
   - The posted flag is only a proxy for a market median. It needs a `history.parquet` join, its payouts start in October, and its sportsbook tail is thin: 126 NBA, 330 NFL and 10 NHL rows.
   - Filtering on information known at forecast time keeps the evaluation honest; filtering on outcomes would not (Lerch et al. 2017, doi:10.1214/16-STS588).
6. **Even the market-median gate is optimistic.** Its fused tail hit rates, quoted and pick'em analogs, against live:

   | League | Quoted analog | Pick'em analog | Live |
   |---|---|---|---|
   | NBA | .549 | .548 | .508 |
   | NFL | .582 | .578 | .509 |
   | NHL | .591 | .620 | .545 |

   Two causes are visible:
   - **The grading line.** The same NBA player-games miss by −.090 graded at the noon book line and by −.162 at Underdog's line, which sits on average +.27 above the book's.
   - **In-period calibration.** The calibrators are fitted on validation rows hash-interleaved with the test rows (`training/pipeline.py:1066–1225`).
7. **Record-gate statistics (Part 5 item 1).** I simulated the share of a cell's legs released, by its true hit rate. Assumptions: 4 legs per cell per day (live active cells run 1–18 per date), a 150-day season, weekly looks.

   | Rule | .50 | .52 | .546 | .58 | .60 |
   |---|---|---|---|---|---|
   | Part 5: 30-day fixed-n 90% bound | .011 | .033 | .101 | .304 | .462 |
   | Same, 95% bound | .004 | .014 | .050 | .187 | .316 |
   | Anytime-valid normal mixture (Howard et al. 2021, doi:10.1214/20-AOS1991) | .000 | .000 | .002 | .055 | .212 |

   - **Look frequency.** Daily looks are worse. At 10 legs a day over 120 days, 20.6% of true-.52 cells are released at some point, against 11.7% with weekly looks.
   - **Clustering.** Date clustering inside a cell is negligible (design effect 0.96–1.02). A record pooled across leagues has a design effect of 1.45 by player-date (Cameron, Gelbach & Miller 2008, doi:10.1162/rest.90.3.414).
   - **Today.** No cell clears. The best is NHL skater fantasy: hit .540 on 224 legs, lower bound .497.
   - **Priced legs.** Recommended legs at payouts outside 1.8–1.9 are the larger exposure. Since 10-09, count-once: 2,418 legs, hit .483 against a break-even of .533, ROI −9.7% (90% lower bound −12.2%). By league: NBA −10.9% (1,024 legs), NFL −10.6% (911), NHL −1.9% (417). The .5464 hit-rate bound does not apply at other payouts; mean unit profit does.

## Recommendation / routing protocol

| Lever | Change | Acceptance (pre-registered) | Kill | Cost |
|---|---|---|---|---|
| **L1 record gate, now** | A cell's recommended flags show only while its *shadow record* clears the bound. The shadow record is every leg the unchanged rule would recommend: 1.83 payouts, counted once from 10-09, trailing 30 days. The bound is a one-sided 90% lower bound > .5464 on ≥ 100 legs over ≥ 10 dates. Priced legs get the same rule on mean unit profit, lower bound > 0. Recomputed weekly. | 4 weeks: released legs hit ≥ .546 on ≥ 300 legs, or the list stays empty | Released legs hit < .52 on ≥ 300 | 1–1.5 engineering days |
| **L2 market-median screen, now** | Report-only pick'em eligibility, never in `ship`. Eligible = L2 pass and L1 release. Frozen definitions in the list below. | 6 weeks, cells with ≥ 50 live legs: Spearman ρ(offline tail gap, live gap) > 0, and the L2 bound ≤ 7 pp above live hit | ρ ≤ 0, or eligible cells hit < .52 on ≥ 300 | 1.5–2 engineering days |
| **L3 `w_ne`, research first** | A second blend weight, used when the sportsbook quote is within .06 of even. Cross-fitted on validation rows dated before 10-09; quoted legs only. | Near-even Brier improves (CI excludes 0) in ≥ 2 of 3 leagues, and the NBA and NFL near-even tail gap closes by ≥ .04. After the build: a 4-week Diebold–Mariano Brier test shows Win Prob no worse than Market Prob on quoted 1.83 legs (doi:10.1080/07350015.1995.10524599). | Either research condition fails | 0.5–1 research day, then 2–3 engineering days plus a retrain per league |
| L4 DFS-line centre anchor | — | — | **KILL** | — |
| L5 importance weighting | — | — | **KILL** (F5) | — |

**L2 frozen definitions.**
- **Quoted analog:** sportsbook rows with book P(over) within .06 of .5.
- **Pick'em analog:** pick'em rows in cells whose test-half over-rate lies in [.40, .60].
- **Tail:** fused q ≥ 1.05/1.83.
- **Statistic:** a date-clustered one-sided 90% lower bound against .5464.
- **Floors:** ≥ 100 tail rows over ≥ 20 dates per cell. Below that, the cell inherits its league's verdict if the league has ≥ 300 rows.
- **Reported beside it:** ±.04 and ±.08 band sensitivity, near-even Gate 1, and β.

**Why L4 dies.**
- The owner's 10-04 rule forbids it.
- With NBA β ≈ 0 it fits to near-full strength, which is abstention.
- The even-money line is the median, so a mean anchor on a right-skewed family leans Under (Gneiting 2011, doi:10.1198/jasa.2011.r10138).
- Reopen only if the rule is reversed and a cell's β lower bound clears .40.

**What the levers must not touch.**
- L1 leaves Win Prob, training, Gate 1, the blend fit and `realized.recommended()` (`realized.py:90`) alone. That function defines the shadow record. The gate is only a release mask downstream of it, so it can never lock itself shut.
- L2 uses DFS lines only to evaluate.
- L3 fits on no pick'em row and leaves unquoted legs alone (676 of 1,221 recommended legs).

**Routing today.**
- L2 retrospectively fails NBA (bound .520 quoted, .528 pick'em), leaves NFL borderline (.545, .546) and passes NHL (.567, .599).
- Expected L3 weights: NBA ≈ 0, NFL small, NHL ≈ current w.
- L1 already keeps the REPORT's NFL volume cells off: live carries, receptions and QB-yards legs hit .41–.44.

**Owner decides.**
1. L1, accepting an empty list today.
2. L2 as pick'em eligibility. It evaluates against DFS lines, which the 10-04 rule allows, and leaves the 09-29 Gate 1 decision alone.
3. Whether L3's research step runs.
4. Confirm L4 stays dead. The plan's "market-centre anchor" wording conflicts with the 10-04 rule.
5. Withdraw the "edge where the market is thin" claim.
6. What the board shows when honest probabilities recommend almost nothing.
7. Whether a cell's record survives a retrain.

## Reality checks

- **No lever here creates profit.**
  - L1 turns −4.6% ROI on 1,221 legs at 1.83 (hit .521 [.491, .554]) and −9.7% on 2,418 priced legs into near-zero exposure.
  - Profit needs β ≳ .40 at the median. Only NHL shows that offline, and its live legs still hit .545.
  - L3 cannot create edge where β ≈ 0. At best it keeps NHL's quoted legs and makes NBA's Win Prob honest, which means fewer NBA picks.
- **The bands were chosen after the reads.** The sensitivity report is the guard.
- **Thin history.** NBA's market-median rows are 2026 only, and NFL's β flips sign across eras.
- **Transfer is a bet.** MLB and WNBA were not read.
- **Simple simulations.** They assume independent legs and a constant hit rate.
- **No DFS-specific literature.** I found no primary literature on pick'em line efficiency; the market citations are about game lines and sportsbooks.

## Open questions / caveats (for §8.2)

1. What gives NHL information at the median and NBA none: features, market depth or stat shape?
2. M6. Does the platform's choice between even money and a priced lean select against the model? Testing it needs Boost-derived payouts on pre-October rows.
3. Should a record be kept per platform? Sleeper's recommended 1.83 legs returned −10.4% (228 legs), against −3.2% for Underdog's (993).
4. How the 4–7.5 pp optimism splits between the grading line and in-period calibration. A forward-dated validation split would separate them.
5. Optional upgrades, not recommended now:
   - Multiplicity control (Holm 1979; e-BH, Wang & Ramdas 2022, doi:10.1111/rssb.12489). The cost that matters is exposure per losing cell, which F7 bounds near 3%.
   - Partial pooling toward the league rate (Efron & Morris 1973, doi:10.1080/01621459.1973.10481350; Brown 2008, doi:10.1214/07-AOAS138).

## Evidence read

- **Lane folder:**
  - `REPORT.md` 418–444, 496–533 and 581–640.
  - In `research/`, the `.txt` and `.py` files of the gap_decomposition, nba_matched_legs, nba_population_split, nfl_population_split, offline_vs_live_tail, nba_forward_test, dfs_market_anchor_read, dfs_anchor_splits, dfs_anchor_side_check and dfs_line_vs_book_line reads.
- **Code:**
  - `prediction/model_prob.py` 857–944 and 1032–1250
  - `helpers/distributions.py` 917–928 and 970–1117
  - `training/pipeline.py` 151–154, 892–948, 1066–1225, 2599–2754, 3232–3295 and 3522–3600
  - `training/calibration.py` 678–692
  - `stats/base.py` 2266–2295 and 2360
  - `helpers/training_quotes.py` 430–475
  - `realized.py` 1–175
- **Docs:**
  - `ship_gate.md` line 232
  - `live-gap-2.md` §8
  - `model_improvement_track.md` §3.4 and §8.2
  - Plan Part 5, lines 141–182
- **New reads:** seven scratch scripts (six reads and the record-gate simulation); the handback lists their paths.

## Bibliography

| Source | Identifier |
|---|---|
| Smith & Winkler 2006, Management Science 52(3), optimizer's curse | doi:10.1287/mnsc.1050.0451 |
| Milgrom & Weber 1982, Econometrica 50(5), winner's curse | doi:10.2307/1911865 |
| Clements & Harvey 2010, J. Appl. Econometrics 25(6), encompassing for probability forecasts | doi:10.1002/jae.1097 |
| Deutsch, Granger & Teräsvirta 1994, IJF 10(1), changing combination weights | doi:10.1016/0169-2070(94)90049-3 |
| Elliott & Timmermann 2005, IER 46(4), combination under regime switching | doi:10.1111/j.1468-2354.2005.00361.x |
| Gneiting 2011, JASA 106(494), making and evaluating point forecasts | doi:10.1198/jasa.2011.r10138; arXiv:0912.0902 |
| Shimodaira 2000, JSPI 90(2), covariate shift weighting | doi:10.1016/S0378-3758(00)00115-4 |
| Sugiyama, Krauledat & Müller 2007, importance-weighted CV | JMLR 8:985–1005 |
| Moreno-Torres et al. 2012, Pattern Recognition 45(1), dataset shift | doi:10.1016/j.patcog.2011.06.019 |
| Lerch et al. 2017, Statistical Science 32(1), forecaster's dilemma | doi:10.1214/16-STS588 |
| Hébert-Johnson et al. 2018, ICML, multicalibration | arXiv:1711.08513 |
| Krieger & Davis 2024, J. Econ. Finance 48(2):263–279, visibility and NFL line efficiency | doi:10.1007/s12197-023-09656-5 |
| Glosten & Milgrom 1985, JFE 14(1), adverse selection in quoting | doi:10.1016/0304-405X(85)90044-3 |
| Howard, Ramdas, McAuliffe & Sekhon 2021, Ann. Statist. 49(2), confidence sequences | doi:10.1214/20-AOS1991 |
| Cameron, Gelbach & Miller 2008, REStat 90(3), clustered inference | doi:10.1162/rest.90.3.414 |
| Holm 1979, Scand. J. Statist. 6(2), sequential rejection | JSTOR 4615733 |
| Wang & Ramdas 2022, JRSSB 84(3), FDR with e-values | doi:10.1111/rssb.12489 |
| Efron & Morris 1973, JASA 68(341), empirical Bayes | doi:10.1080/01621459.1973.10481350 |
| Brown 2008, Ann. Appl. Statist. 2(1), in-season batting averages | doi:10.1214/07-AOAS138 |
| Diebold & Mariano 1995, JBES 13(3), comparing predictive accuracy | doi:10.1080/07350015.1995.10524599 |
