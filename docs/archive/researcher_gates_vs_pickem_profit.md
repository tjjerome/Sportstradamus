# In-repo research brief — why gate-passing models lose on recommended pick'em legs, and which lever to build (L1 / L2 / L3)

**Question (owner, 2026-10-29 evening, verbatim):** "This sounds to me like miscalibrated models, which
should have been handled by the gates, so I'm still a little confused." Operationally: decompose the
recommended 1.83 tail gap (claimed Win Prob minus realized hit, games 10-09..10-26, count-once) per
league; test whether any statistic computable before 10-09 ranks the served cells by their live gap;
decide L1 / L2 / L3 / something else, each with a pre-registered test, a cost and what it cannot fix.

**Date:** 2026-10-29, 21:00–22:00 CDT. **Scope:** read-only. No production file, pickle,
`stat_meta.json` entry or git ref was touched; no read needed the archive. Inputs: dev
`data/runtime/history.parquet` (games through 10-26), the 10-18 test-set dumps
`data/test_sets/prior/*__20261018*` ("v1018", the version that served), `$L/i6d/before/<LG>/test_sets/`
("before"), `data/test_sets/` ("current", the 10-29 post-I6d retrains), the 10-17 `model_stats.parquet`
backup, `data/training/tail_scorecard.parquet`. Scripts and outputs: `$L/research/gates_vs_profit_reads/`
r00–r13 (`$L` = `/home/trevor/backups/sportstradamus/2026-10-21-live-gap-2`).

**Conventions.** gap = hit − claimed (negative = the model reads high) unless a table says otherwise;
q = chosen-side probability, b = the book's probability for the same side; "market line" = a
sportsbook quote within .06 of even, or a DFS pick'em line (price pinned at .5); recommended = q ≥
1.05/1.83 = .574 at a 1.83 payout (`realized.recommended`); break-even .5464; 90 % date-cluster
bootstrap intervals (Cameron, Gelbach & Miller 2008). Encompassing β here is fused-side with an
intercept, so it differs from `MARKET_CENTRE_BRIEF.md`'s standalone-side numbers; both put NBA and NFL far below .40.

---

## TL;DR (plain English, for the owner)

Partly, yes: the models are too confident, but only on the kind of pick we recommend, and no gate
looks at that kind of pick on its own (recommended picks claimed about 61 % and won about 51 % in NBA
and NFL and 55 % in NHL, against 54.6 % needed to break even). Across everything in the test data a
60 % call comes true about 60 % of the time in all three leagues, and all 30 served stats pass the
confidence gate with room to spare; but 78–93 % of those confident test calls sit on lines the
pipeline made up from a player's recent average or on lines where the sportsbook already agreed, not
on the coin-flip lines we bet. The picks we recommend sit almost entirely on lines the pick'em app or
a sportsbook set as a coin flip, and we take the ones where the model disagrees with that line the
most; in NBA and NFL the model knows little the line does not already know, so its biggest
disagreements are mostly its biggest mistakes, and the old test data showed this as soon as those
lines were read on their own. NHL's shortfall did not show in the old test data, which points at the
comparison-player leak being rebuilt now, but that stays unproven until the rebuild lands. No gate
number could have singled out the losing stats in advance: at the standard 1.83 payout they all lose
about the same, and if anything the stats that looked best against the sportsbook did worse, probably
because of the sportsbook-price bug fixed this week. So re-tuning the model's confidence will not make
money (every honest version of it simply stops making these picks); the fix that holds up is to hide a
stat's recommendations until its own record of recommended picks beats break-even, which today hides
all of them, and to keep looking for information the line does not already have.

## TL;DR (technical)

- **H1 rejected; H2 and H6 are one finding and carry NBA and NFL; NHL is unexplained.** Waterfall on
  recommended 1.83 legs: NBA −.096 = marginal +.003, market line −.051, residual −.048 [−.087, −.002];
  NFL −.109 = −.001, −.078, −.030 [−.086, +.012]; NHL −.062 = +.004, +.006, −.072 [−.103, −.035]. Market
  lines are 80 / 94 / 98 % of live recommended legs but 17 / 6 / 7 % of the offline ≥ .574 tail, where
  the pipeline's own synthetic/derived lines (55 / 61 / 40 %) are calibrated or under-confident. Gate 5
  debiased ECE max .024 against .075; 0 of 30 cells fail.
- **Nothing ranks the losing cells.** At 1.83 (n = 7) live gaps are homogeneous (Cochran Q 1.4 on 6 df,
  τ = 0). At all payouts (n = 21, τ .052, reliability .55) no statistic survives Holm over 38; the
  strongest point the wrong way (L2 quoted-row gap ρ +.71, n 11; near-even BSS +.65, n 12; Gate 1 BSS
  +.48, n 15; L2 gap −.47 in L2's own convention, n 16, its pre-registered KILL side). The tail
  scorecard agrees with live per league, not per cell (ρ −.17, n 21). Ranking needs ≈ 46 cells at
  today's reliability, or ≈ 540 legs per cell plus ≈ 31 cells.
- **The league-level market-median gap was the readable signal.** Read at the live claimed level, its
  90 % lower bound gave NBA .508 and NFL .454 (both under .5464; live .508, .509) and NHL .544, a
  0.25 pp miss that flips to a pass on the 10-27 and 10-29 NHL generations (live .545). On the 10-29
  versions now serving it still reads NBA .494–.501 and NFL .474–.499.
- **L1 GO; L2 only as a league-level, gap-form, report-only read (~0.5 day); L3 KILL; λ_DFS KILL.**
  L3 fails its pre-registered acceptance in both generations (served v1018: Brier 0 of 3 leagues, NBA
  tail closes .030, NFL .008; post-I6d: 1 of 3, .044 / .028). It is λ_sel's algebra fit before
  selection, the principled version (Dawid 1994, doi:10.1214/lnms/1215463797; Efron 2011,
  doi:10.1198/jasa.2011.tm11181), and fails anyway: abstention where strong, under-shrinkage where
  mild. λ_DFS on the 56 % unquoted legs fits .29–.35 in NBA/NFL (0 of 320 recommended legs survive,
  the #0d class) and .98–.99 in NHL (342 of 347 survive, gap unchanged).
- **Why the gates miss it:** they certify probability quality on the bulk next to the book. Gate 1 is a
  non-inferiority average, Gate 5 is marginal calibration, T is fit on all validation rows, and
  calibration certifies no information (Foster & Vohra 1998, doi:10.1093/biomet/85.2.379). Profit at a
  fixed payout needs calibration conditional on the market's number on the selected legs: the
  multicalibration / decision-calibration property (Hébert-Johnson et al. 2018, arXiv:1711.08513; Zhao
  et al. 2021, arXiv:2107.05719). No gate measures it, and the data say NBA and NFL lack it (offline
  pick'em β NBA .02, NFL .18; post-I6d near-even standalone β NBA −.01, NFL .12).

---

## Key findings

**1. The calibration the gates check is fine (H1 rejected).** On the version-matched v1018 test sets
(30 cells with live recommended legs, rows before 10-09) the fused ≥ .574 tail misses by −.003 (NBA,
10,126 rows), −.012 (NFL, 10,592) and +.003 (NHL, 50,495). Every bin from .574 to .70+ is within .025
(r02). Gate-5 debiased ECE peaks at .024 against .075, and 0 of 30 cells fail. Applying the pooled
offline miss to each live leg (the H1 term) moves expected hit by +.003 / −.001 / +.004.

Calibration on average is necessary and nowhere near sufficient:
- Foster & Vohra (1998, *Biometrika* 85(2):379–390, doi:10.1093/biomet/85.2.379) build a forecaster
  that is asymptotically calibrated on any sequence with no knowledge of the process.
- DeGroot & Fienberg (1983, *The Statistician* 32(1/2):12–22, doi:10.2307/2987588) rank calibrated
  forecasters by refinement.
- Murphy's partition (1973, doi:10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2) puts reliability and
  resolution in separate Brier terms. Gate 5 bounds reliability only.
- Gneiting, Balabdaoui & Raftery's "sharpness subject to calibration" (§14 [75]) says the same.

**2. The shortfall lives on market lines, which the gates barely contain (H2).** Pre-10-09 fused tail
(q ≥ .574) gap by line type (r02):

| League | near-even book | pick'em | own line (synthetic/derived) | book agrees ≥ .06 |
|---|---|---|---|---|
| NBA | −.057 [−.082, −.029] n 698 | −.092 [−.129, −.051] n 996 | +.034 n 5,540 | −.034 n 2,365 |
| NFL | −.098 [−.185, −.007] n 135 | −.115 [−.180, −.040] n 511 | +.005 n 6,429 | −.022 n 3,461 |
| NHL | +.011 [−.009, +.030] n 1,551 | −.007 [−.028, +.011] n 1,828 | +.006 n 19,957 | +.002 n 27,153 |

Composition. Offline: share of ≥ .574 rows. Live: share of recommended 1.83 legs.

| League | offline: market lines | offline: own line | live: market lines (near-even + pick'em-only) |
|---|---|---|---|
| NBA | 16.7 % | 54.7 % | 79.7 % (45.2 + 34.5) |
| NFL | 6.1 % | 60.7 % | 94.3 % (45.9 + 48.4) |
| NHL | 6.7 % | 39.5 % | 98.2 % (9.5 + 88.7) |

This is the dispatch's trap at full size: the model is calibrated against its own trailing average,
and those rows dominate every pooled read.

The subgroup framing fits:
- Hébert-Johnson, Kim, Reingold & Rothblum (2018, ICML, PMLR 80:1939–1948, arXiv:1711.08513) ask for
  calibration on every subgroup a class can identify. The subgroup that matters here is "the line came
  from a market and sits near even".
- Hansen et al. (§14 [45], arXiv:2406.06487) find GBDTs often already multicalibrated over groups
  their own features define. The market's number is not such a feature, so no such guarantee holds.
- Dawid (1982, *JASA* 77(379):605–610, doi:10.1080/01621459.1982.10477856): a coherent forecaster
  expects calibration on subsequences selected with information it uses. The recommended set is
  selected with the line, which carries information the model does not condition on.

**3. The waterfall (the statistic that decides).** For each live recommended 1.83 leg (r03):
- E_A = claimed + the offline pooled miss at the leg's bin (what Gate 5 or a pooled read sees);
- E_B = claimed + the offline miss at its (line class, bin);
- residual = hit − E_B.

| League | legs (dates) | claimed | hit | gap [90 %] | H1 marginal | H2 market line | residual [90 %] | H5, inside the residual |
|---|---|---|---|---|---|---|---|---|
| NBA | 478 (7) | .604 | .508 | −.096 [−.133, −.050] | +.003 | −.051 | −.048 [−.087, −.002] | +.016 |
| NFL | 320 (7) | .618 | .509 | −.109 [−.175, −.051] | −.001 | −.078 | −.030 [−.086, +.012] | +.036 |
| NHL | 391 (18) | .607 | .545 | −.062 [−.092, −.025] | +.004 | +.006 | −.072 [−.103, −.035] | n/a (no quoted leg on a differing line) |

ROI at 1.83 is −6.7 % (NBA), −6.8 % (NFL) and −0.2 % (NHL). Recommended priced legs lose more and are
not decomposed here: −10.9 % on 1,024 NBA legs, −10.6 % on 911 NFL legs, −1.9 % on 417 NHL legs.

The pre-10-09 test data, read on the right line types, already held about half of NBA's gap and about
three-quarters of NFL's. NHL's gap is all residual. Descriptively, the residual holds (r03, r06, r12):
- **H5 line mismatch helps:** legs whose DFS line differs from the book's line show residual +.061
  (NBA) and +.085 (NFL).
- **Version:** NFL's 09-27 pickles −.043 against −.024 for 10-18. NHL: 08-06 −.136, 08-29 −.014,
  09-02 −.097, 10-15 −.105, 10-18 −.064.
- **Platform:** Sleeper −.076 against Underdog −.035 (NBA); −.050 against −.026 (NFL). NHL runs the
  other way (n 28).
- **Side:** recommended Unders at market lines miss more in NBA (−.142 [−.225, −.069]) and NFL (−.135
  [−.216, −.071]) than Overs (−.074 [−.129, +.020]; −.047 [−.073, +.051]). NHL is reversed (Overs
  −.094, n 45; Unders −.053).
- **NFL live level:** the served mean ran 7.3 % of the line below the realized mean [+4.7, +9.7],
  while the DFS lines sat at the median (over-rate at the line .495). NBA −2.5 % [−6.1, +1.1], NHL
  +1.8 % [−1.8, +5.7]. Offline, NFL's weak side at market lines was the Over (−.173, n 144, against
  Unders −.093), so the NFL Under loss is live-only.
- **In-period calibration:** w and T are fit on validation rows hash-interleaved with the test rows
  (`training/pipeline.py:_step_build_splits`), so offline calibration is optimistic by construction.

**4. H6 is H2 seen through selection, not a separate cause.** Both describe E[y | q, b] at market
lines. Live, all 1.83 legs at market lines, by disagreement q − b (r07):

| q − b | legs | claimed | b | hit | gap | hit − b |
|---|---|---|---|---|---|---|
| < .03 | 1,884 | .520 | .514 | .490 | −.030 | −.024 |
| .03–.074 (not selected) | 1,620 | .543 | .493 | .535 | −.008 | +.042 |
| .074–.10 | 531 | .576 | .491 | .533 | −.043 | +.042 |
| .10–.15 | 517 | .618 | .496 | .509 | −.109 | +.013 |
| > .15 | 65 | .697 | .491 | .400 | −.297 | −.091 |

Hit stays near the book's number (hit − b from −.02 to +.04) while the claimed edge grows, so the
selected tail's miss scales with the disagreement. That is the optimizer's curse (Smith & Winkler
2006, *Management Science* 52(3):311–322, doi:10.1287/mnsc.1050.0451), the winner's curse of
competitive bidding (Capen, Clapp & Campbell 1971, doi:10.2118/2993-PA; Thaler 1988,
doi:10.1257/jep.2.1.191): picking the largest estimated edges picks the largest estimation errors.

Forecast encompassing (Chong & Hendry 1986, doi:10.2307/2297611; Harvey, Leybourne & Newbold 1998,
doi:10.1080/07350015.1998.10524759) regresses (y − b) on (q − b). β = 0 means the book contains
everything the model knows.

| Read | NBA | NFL | NHL |
|---|---|---|---|
| live near-even β | +.21 [−.48, +.82] | +.03 [−.53, +1.62] | +1.97 [+.65, +3.35] (unstable, n 522) |
| live pick'em-only β | −.36 | −.27 | +.32 [−.16, +.90] |
| live Brier-optimal pull λ*, near-even | .25 | .14 | .89 |
| live Brier-optimal pull λ*, pick'em-only | −.33 | .13 | .57 [.35, .81] |
| offline pick'em β, v1018 pre-10-09 (b = .5) | .02 [−.36, +.47] | .18 [−.30, +.62] | .99 [.92, 1.08] |

The NBA and NFL pick'em verdict was already in the pre-10-09 data.

- **Not a consequence of H3:** NBA's offline market-line weakness is mid-season (days 29+, −.078 on
  v1018).
- **Not caused by H4:** pick'em tails do not move across generations (NBA −.092 → −.076, NFL −.115 →
  −.110, NHL −.007 → −.003), and the post-I6d near-even standalone β is −.007 [−.130, +.145] in NBA and
  .116 in NFL. H4 amplified it (finding 7).
- **Slope, not level:** the miss grows with disagreement instead of sitting at a constant offset. That
  is missing information, not a bias, which is why a re-centring (the killed L4) cannot fix it. An
  even-money line is a median, not a mean (Gneiting 2011, doi:10.1198/jasa.2011.r10138).
- **Falsifiers:** (a) a flat gap across disagreement bins; (b) hit − b reliably above 0 at q − b ≥
  .10; (c) a live near-even β lower bound above .40 in NBA or NFL on ≥ 2,000 legs (the #0d scale).
  None holds today, and (c) cannot be read at today's sample.

**5. Why the gates could not see it: by design, not by bug.**
- **Gate 1** is a non-inferiority Brier test against the book on authentic rows (`ci_hi < .005`), a
  Diebold–Mariano-style average (Diebold & Mariano 1995, doi:10.1080/07350015.1995.10524599). A fused
  P that is mostly the book ties it with zero information. Its rows are where model and book agree:
  mean model–market distance .050, against .120 on recommended legs (`docs/ship_gate.md`).
- **Gate 5** is marginal calibration over all rows (finding 1).
- **T** (`_step_calibrate_temperature`) is fit by Brier on all validation rows at each row's own line,
  the same synthetic-heavy population. Served T medians are NBA 1.56, NFL 1.42 and NHL 1.02; a
  pick'em-only T would be 3.1–3.6 (NBA), 8.9–14.3 (NFL) and 1.0 (NHL) (r09).
- **Live Gate 2** grades all settled offers, not the recommended ones.

The gates certify "safe to serve as a probability next to the book", and these models are. Profit at
a fixed payout needs calibration on the decisions the probabilities induce. Zhao, Kim, Sahoo, Ma &
Ermon (2021, arXiv:2107.05719) call this decision calibration. The forecasting literature separates
the two properties:
- Betting returns are not a valid measure of accuracy (Wunderlich & Memmert 2020, *IJF*
  36(2):713–722, doi:10.1016/j.ijforecast.2019.08.009).
- Profit comes from being right where you differ from the bookmaker: Hubáček, Šourek & Železný 2019,
  *IJF* 35(2):783–796, doi:10.1016/j.ijforecast.2019.01.001 (decorrelate from the odds); Hubáček &
  Šír 2023, *IJF* 39(2):691–719, doi:10.1016/j.ijforecast.2022.02.001, arXiv:2010.12508 (profit with an
  inferior but decorrelated model).
- Bookmaker prices are strong forecasts in their own right (Štrumbelj 2014, *IJF* 30(4):934–943,
  doi:10.1016/j.ijforecast.2014.02.008; Kovalchik 2016, *JQAS* 12(3):127–138,
  doi:10.1515/jqas-2015-0059).

`docs/ship_gate.md` already says "the gates score the bulk, not the recommended tail."

**6. The gate-ranking test: nothing ranks the losers (r04, r04b, r12).**

*1.83 cohort.* n = 7 cells with ≥ 50 live recommended legs and a v1018 set: NBA PR, PRA and RA; NFL
fantasy UD and yards; NHL hits and skater fantasy. The live gaps are homogeneous: pooled .081 (claimed
− hit), Cochran Q 1.4 on 6 df (p .97), DerSimonian–Laird τ = 0 (DerSimonian & Laird 1986,
doi:10.1016/0197-2456(86)90046-2), reliability 0. Of 19 statistics the best is the stored g4_pit_ks,
ρ −.68 (p .09, Holm 1.0). There is nothing to rank.

*All-payout cohort* (the `nightly.recommended_gap_by_cell` definition), n = 21. The cells do differ:
Q 56.5 on 20 df, τ .052, mean sampling sd .047, reliability .55. Of 38 statistics none survives Holm
(Holm 1979, *Scand. J. Statist.* 6(2):65–70, JSTOR 4615733); 1.9 false positives are expected at
.05. The strongest point the wrong way, with ρ measured against the live gap as claimed − hit:

| statistic (v1018 rows before 10-09; "stored" = the 10-17 `model_stats`) | n | ρ | p | Holm |
|---|---|---|---|---|
| L2 tail gap, quoted near-even rows only | 11 | +.71 | .015 | .55 |
| near-even BSS | 12 | +.65 | .022 | .82 |
| offline mean disagreement at near-even quotes (the one right-signed) | 12 | +.55 | .067 | 1.0 |
| Gate 1 BSS, recomputed | 15 | +.48 | .071 | 1.0 |
| L2 tail gap as specified (= −.47 in L2's offline-gap vs live-gap convention) | 16 | +.47 | .064 | 1.0 |
| stored kelly_shrinkage | 16 | +.45 | .082 | 1.0 |
| stored brier_skill_score | 16 | +.44 | .087 | 1.0 |
| encompassing β, fused, near-even | 12 | +.35 | .27 | 1.0 |
| Gate 5 ECE, debiased | 21 | −.10 | .67 | 1.0 |

A positive ρ on a skill statistic means more offline skill came with a bigger live miss.

The likely mechanism: before I6d, offline BSS tracked the book leg's level error (ρ .75, memory
`book_level_error_inflates_offline_bss`). The most mis-levelled cells looked most skilled, earned
higher w, and disagreed most with the correctly priced live market. On the post-I6d sets Gate 1 BSS
turns to the right sign (ρ −.34, n 16, p .20); the L2 gap stays wrong-signed (+.32, n 18).

The existing tail scorecard (10-18 versions, archived DFS rungs) agrees with live per league: NFL
+.086 against +.130, NHL +.070 against +.070 (claimed − hit). It does not agree per cell (ρ −.17, n 21,
p .46).

*What sample would decide it.* A reliability of .55 attenuates any observed ρ by √.55 (Spearman 1904,
doi:10.2307/1412159). To detect a true ρ of .5 at 80 % power and one-sided α .05 takes ≈ 46 cells
(≈ 58 two-sided; Bonett & Wright 2000, doi:10.1007/BF02294183); the three leagues have 24 cells with
≥ 50 recommended legs. Reliability .8 needs ≈ 540 legs per cell (≈ 4 weeks of NBA or ≈ 8–10 weeks of
NFL or NHL at today's all-payout rate), and then ≈ 31 cells. At 1.83 there is no between-cell signal
to find.

**7. H4: the I6d bug inflated offline information; I6f is the NHL suspect; the version trap.**

Same cells, rows before 10-09, near-even sportsbook rows (r05), v1018 → current:
- NHL standalone β .978 [.915, 1.034] → .572 [.311, .824]; fused 1.113 → .847; near-even BSS .047 →
  .002.
- NBA standalone .247 [.114, .376] → −.007 [−.130, +.145]; fused .893 → .363.
- NFL standalone .085 → .116; its near-even tail gap −.098 → −.019 (n 287), the one place the retrain
  may have helped.
- Pick'em tails do not move, because I6d does not touch pick'em rows.

The mis-levelled book leg made the model look informative at even money, most in NHL and NBA, and fed
a blend weight that was too high.

*Version trap (correcting the dispatch).* `i6d/before/` matches the live record for NBA only, whose
`before/` sets are the 10-18 dumps. NFL's and NHL's `before/` sets are the 10-27 retrains, which
served no settled leg (history ends at game date 10-26). Recommended 1.83 legs by serving version:
- NBA: 10-18 93 % (the rest book fallback).
- NFL: 09-27 44 %, 10-18 33 %, 10-15 14 %, 09-29 and 10-03 8 %.
- NHL: 10-18 41 %, 08-29 23 %, 10-15 20 %, 09-02 9 %, 08-06 7 %.

Only 10-18 and 10-27 dumps exist, so v1018 is exact for NBA and the nearest available for NFL and NHL.

*I6f (the NHL comp pool built from season-end summaries)* is unmeasured. Its fingerprint would be
offline information that live lacks, and NHL pick'em lines show exactly that: offline λ .98–.99 and
β .99, against live pick'em-only λ* .57 [.35, .81] and recommended pick'em hit .545 on a claimed .607.
Several NHL DFS lines are also lopsided rather than medians (test over-rate at the line: assists .281,
goals .126, points .394, skater fantasy .395). Hits is a real median (.475) and shows an offline tail
gap of −.076 with λ .56 (r10).

*When I6f becomes measurable.*
- **Offline:** once the cold matrix pass, `meditate --league NHL --force` and the new test sets land.
  The pass started 16:20 CDT 10-29 and was on market 5 of 15 at 21:10 CDT, about an hour per market,
  so roughly the evening of 10-30 to 10-31. Rerun r05, r08, r09 and r13 on the same NHL cells, rows
  before 10-09. If I6f carried NHL's residual, pick'em λ and β fall toward ≈ .6 and the league gap
  widens from −.045 toward the live −.062.
- **Live:** ≈ 1,000 NHL pick'em 1.83 legs (≈ 2.5–3 weeks after the send) give a λ* as sharp as
  today's. Skip each version's first four days (#0d: information share falls from .47–.61 to .05–.14
  within one to two weeks).

**8. H3 (season start) carries little.**
- **NBA:** live legs are days 0–6, but offline the market-line tail is no worse at season start (current
  sets: days 0–28 −.065, n 418, which are the live window's own test rows; days 29+ −.088).
- **NHL:** live −.092 (days 7–13, n 65), −.062 (14–20, n 172), −.049 (21–28, n 154), with overlapping
  intervals. Offline, days 0–28 and days 29+ both read +.001.
- **NFL:** live legs are all past day 28, so H3 cannot apply.

At most a few points for NHL and about zero for NBA. No season-start lever.

**9. H5 (line mismatch) is the one place the model held up.** It is folded into finding 3 (+.016 NBA,
+.036 NFL). Recommended quoted legs on the book line's side of a differing DFS line: n 95 over 12
dates, hit .621 [.509, .722], claimed .618, Market Prob .547, ROI +13.6 %. Same-line recommended legs
hit .487 [.442, .531] (ROI −10.6 %), and unrecommended legs on the book's side hit only .482 (r11).
The pattern matches Kaunitz, Zhong & Kreiner (2017, arXiv:1710.02824): the edge sits where one price
disagrees with the consensus. It is a lead to read forward in L1's shadow record, not a lever: the
interval includes break-even.

**10. L3 `w_ne` fails its pre-registered acceptance in both generations (r08).** The acceptance,
verbatim from the dispatch: "near-even Brier improves (CI excludes 0) in >= 2 of 3 leagues AND the
NBA/NFL near-even recommended-tail gap closes by >= .04."

| generation | λ, fused form (NBA / NFL / NHL) | Brier improves, CI excludes 0 | NBA tail closes by (rows left) | NFL tail closes by (rows left) | verdict |
|---|---|---|---|---|---|
| v1018 (served) | .67 / .67 / 1.05 | 0 of 3 | .030 (335) | .008 (73) | FAIL |
| current (post-I6d) | .26 / .72 / .63 | 1 of 3 (NBA +.0015 [+.0001, +.0028]) | .044 (56) | .028 (180) | FAIL |

The standalone-form weights on current rows (NBA .006, NFL .13, NHL .41) match the expected "NBA ≈ 0,
NFL small, NHL ≈ current".

Live quoted near-even legs (1,312 NBA, 962 NFL, 522 NHL):
- With the v1018 λ, NBA Brier improves .0014 [.0003, .0023]; NFL and NHL do not move significantly.
- With the v1018 λ, NBA's 216 recommended legs become 85 (hit .553, gap −.039) and NFL's 147 become 98
  (gap −.139).
- With the current λ, NBA keeps 5 legs (abstention) and NFL 107 (gap −.136).

*Head-on 1: is `w_ne` λ_sel again?* The algebra is the same, Market Prob + λ(Win Prob − Market Prob).
The fit differs: λ_sel was fit on the selected legs, `w_ne` on all near-even rows before selection.
Fitting before selection is the principled correction, because a correctly specified posterior mean is
immune to selection (Dawid 1994, doi:10.1214/lnms/1215463797); Efron's Tweedie correction (2011,
doi:10.1198/jasa.2011.tm11181) is fit on the whole population and applied to the selected tail. It
fails anyway, for two reasons:
- The offline population's λ is not the live tail's: offline NBA .26–.67 and NFL .67–.72, against
  live near-even λ* .25 and .14 and λ_sel ≈ −.10 on the selected legs.
- One λ is linear in the disagreement, while the live miss is steepest at the extreme. A mild λ
  under-shrinks (NFL keeps a −.14 gap); a strong one abstains (NBA 216 → 5).

So it is not a disguised retry, but it ends in the same place. Reopen only on #0d's trigger.

**11. The unquoted 56 %: only L1 acts usefully (r09, r10).** Recommended pick'em-only 1.83 legs are NBA
165 (35 %), NFL 155 (48 %) and NHL 347 (89 %). They hit .467 / .490 / .545 against a claimed .608 /
.606 / .607. `w_ne` cannot touch them because they carry no quote.

λ_DFS shrinks toward even, q′ = .5 + λ(P − .5), cross-fit over five date blocks on pick'em rows
before 10-09:
- **Fit:** λ NBA .32–.35 (out-of-sample Brier gain +.003 to +.006, CI ≥ 0), NFL .29 (14 dates, CI
  spans 0), NHL .98–.99.
- **Live Brier:** improves in NBA (+.008 [+.0001, +.015]) and NFL (+.004 [+.0004, +.014]), not NHL.
- **Live recommended legs:** 0 of 165 NBA and 0 of 155 NFL survive, which is abstention. 342 of 347
  NHL survive with the gap unchanged (−.068).
- **T_pickem** behaves the same way (T 3.1–3.6 NBA, 8.9–14.3 NFL, 1.0 NHL).

λ_DFS is a learned pull toward the platform's price, the #0d class, and here it reduces to
abstention, which L1 delivers without a probability layer. In NHL it does nothing because the offline
pick'em rows look fully informative (finding 7).

**12. The league-level, gap-form L2 is the readable offline signal (r12, r13).** L2's frozen
definitions, pooled per league, rows before 10-09. "Implied" = live claimed mean + the gap's one-sided
90 % lower bound.

| League | generation | rows (dates) | claimed | hit | hit lower bound → L2 as specified | gap [lower bound] | implied live hit lower bound | live hit |
|---|---|---|---|---|---|---|---|---|
| NBA | v1018 | 1,694 (77) | .631 | .554 | .536 → fail | −.078 [−.097] | .508 | .508 |
| NFL | v1018 | 621 (50) | .660 | .539 | .494 → fail | −.121 [−.165] | .454 | .509 |
| NHL | v1018 | 1,164 (155) | .631 | .586 | .568 → **pass** | −.045 [−.062] | .544 | .545 |
| NBA / NFL / NHL | current (10-29, now serving) | 2,339 / 646 / 1,445 | .623 / .658 / .623 | .536 / .557 / .589 | — | −.088 / −.101 / −.034 | .494 / .474 / .554 | — |

The gap form called NBA and NFL clearly. It caught NHL by 0.25 pp, and the same read passes NHL on the
10-27 and 10-29 generations (.554–.561), so it would not reliably have caught NHL. The hit form
passes NHL because the offline tail claims .631 against .607 live. This is three leagues in one
window: suggestive, not validation.

---

## Recommendation / routing protocol

| Lever | Verdict | Pre-registered test | Cost | What it cannot fix |
|---|---|---|---|---|
| **L1** live-record release gate (as in `MARKET_CENTRE_BRIEF.md` Part 5: shadow record of `realized.recommended()` legs, trailing 30 days, count-once, one-sided 90 % lower bound > .5464 on ≥ 100 legs over ≥ 10 dates; priced legs: mean unit profit lower bound > 0; weekly) | **GO, build first.** It is the only lever that grades the population that loses money, it reaches quoted, unquoted and priced legs alike, and it cannot leak. Retrospectively it releases nothing: league hit lower bounds NBA .480, NFL .470, NHL .521. | Frozen before the build. Operating characteristics as simulated (share of a cell's legs released: .011 at a true .50, .101 at .546, .462 at .60). Forward: within 4 weeks, released legs hit ≥ .546 on ≥ 300 legs or the list stays empty; KILL if released legs hit < .52 on ≥ 300. Weekly release lists must reproduce from the frozen script. | 1–1.5 engineering days. Optional pooled variant +0.5 day. | It creates no edge. Win Prob stays over-confident on market lines. Release is close to a lottery among break-even cells (≈ 10 % of a true-.546 cell's legs). It cannot rank cells in its first weeks, and it uses the realized record, so the owner should confirm it sits outside the 10-04 rule. |
| **L2** market-median screen | **Reframe.** The per-cell form is not supported: its retrospective ρ is −.47 (n 16), its own pre-registered KILL side, and at 1.83 there is no cell signal. Build only the league-level gap form as a report-only column beside the tail scorecard. Never a gate. | Over the next 6 league-windows (two 30-day windows per league), the league flag (implied live hit lower bound against .5464) must agree with the live recommended record's side of .5464 in ≥ 5 of 6, and the offline and live gaps must sit within .03 in ≥ 4 of 6. KILL at ≤ 3 of 6, or misses > .05 in ≥ 3 of 6. | ≈ 0.5 day, against 1.5–2 days for the per-cell spec. | Report-only, with no per-cell resolution. Blind to leaks that inflate the test set (an I6f-type look-ahead) and to live-only drift (NFL's live level). A pass is not an edge: NHL passes it now and sits at break-even live. |
| **L3** `w_ne` | **KILL.** Fails its own acceptance in both generations (finding 10). | As pre-registered; result FAIL / FAIL. | 0 (avoids 0.5–1 research day plus 2–3 engineering days and a retrain per league). | n/a |
| **λ_DFS / T_pickem** (live-gap-2 item 5.2) | **KILL.** Abstention by another name in NBA and NFL; no change in NHL; the #0d class. | As run: out-of-sample Brier plus live recommended-leg effect (finding 11). | 0 | n/a |

**L1 optional pooled variant.** Release on each cell's partially pooled estimate: the cell shrunk to
its league mean using DerSimonian–Laird τ² each week (empirical Bayes, Efron & Morris 1973, §14 [47]).
At 1.83 τ = 0 today, so per-cell records are noise around the league mean; the pooled form releases
on league evidence and lets a cell stand apart once τ > 0. Run the same operating-characteristics
simulation before building it. If the release looks are ever made more often than weekly, a
time-uniform bound (Howard et al. 2021, doi:10.1214/20-AOS1991) is the conservative choice.

**Something else** (no build unless stated):
1. **Answer to the owner:** the plain-English TL;DR above.
2. **No season-start lever** (H3 small, finding 8).
3. **NHL I6f re-measure** on the rebuilt NHL test sets (finding 7). Prediction to test: pick'em λ and
   β fall toward ≈ .6, and the league L2 gap widens toward −.06.
4. **Calibration population ≠ decision population (research hole).** T and Gate 5 are fit and scored on
   rows that are 40–61 % the pipeline's own lines; the bet population is 80–98 % market lines. The
   report-only fix is the L2 gap form. Do not refit T or a post-hoc layer on market lines for profit:
   in NBA and NFL that is abstention, in NHL a no-op. A displayed "market-line Win Prob" for honesty
   alone is the owner's call.
5. **Leads, not levers:** book's-side line discrepancy (finding 9); NFL's live Under loss and level
   (finding 3); information the line lacks (#0e (iv)).
6. **Expectation for the 10-29 versions now serving:** the league gap form still reads NBA .494 and NFL
   .474 on rows before 10-09 (.501 and .499 on all rows), below .5464, so expect L1 to keep NBA and NFL
   shut. NFL's post-I6d near-even tail (−.019, n 287) is the one hint of change; L1's shadow record is
   where it will show.

---

## Reality checks

- **Effect size.** L1 turns −6.7 % / −6.8 % / −0.2 % ROI on recommended 1.83 legs (NBA / NFL / NHL) and
  −10.9 % / −10.6 % / −1.9 % on priced legs into near-zero exposure. It makes no money. That holds
  while at-market information stays near zero (NBA and NFL now); if a retrain adds real information,
  L1 is how it gets released.
- **Sample.** The H2 term rests on thousands of pre-10-09 rows and is the stable part. The live side is
  7 NBA dates (season days 0–6) and 7 NFL dates, with 140 of 320 NFL legs on one Sunday under one
  version. Every live interval is wide and the residual is the fragile part.
- **Version coverage.** Exact for NBA; for NFL and NHL, v1018 covers only 33 % and 41 % of legs.
- **The gate-ranking null is a power statement as much as a finding.** At all payouts the wrong-signed
  skill statistics are suggestive (p .015–.09), not established. At 1.83 the cells really are alike.
- **L2's league read is n = 3**, and the NHL call sits within 0.25 pp of the line.
- **What could make this wrong.**
  - The post-I6d versions carry information the 10-18 versions lacked (NFL near-even hint).
  - The I6f rebuild changes NHL either way.
  - DFS platforms shade some lines (Levitt 2004, doi:10.1111/j.1468-0297.2004.00207.x), though live
    over-rates at the DFS line sit at .494–.498 in all three leagues.
  - The owner rules that L1's use of the realized record conflicts with the 10-04 rule.

## Open questions / caveats (→ §8.2)

1. **#0e (iv), why NHL shows at-market information offline and NBA none:** partly answered. Much of
   NHL's was the I6d book leg (standalone β .98 → .57); the rest waits on I6f.
2. **NFL's live level:** the served mean ran 7.3 % of the line low in weeks 6–8, with Unders losing
   live and Overs offline. Read it on the post-I6d versions before acting.
3. **Platform:** Sleeper legs miss more than Underdog legs in NBA and NFL (#0e (v)); not decomposed.
4. **Priced legs** are the larger exposure and are not decomposed here; L1's priced rule covers them,
   L2 and L3 do not.
5. **Per-cell agreement with live cannot be measured at 1.83 for months** (τ = 0; ≈ 46 cells needed).
   #0e's guard "four weekly runs agree with live per cell" will be noise-dominated; read agreement per
   league.
6. **In-period calibration optimism** (w and T fit on hash-interleaved validation rows) is a known
   design property, unquantified here.

---

## Bibliography

| # | Source | Identifier |
|---|---|---|
| 1 | Dawid, A. P. (1982). The well-calibrated Bayesian. *JASA* 77(379), 605–610. | doi:10.1080/01621459.1982.10477856 |
| 2 | DeGroot, M. H. & Fienberg, S. E. (1983). The comparison and evaluation of forecasters. *The Statistician* 32(1/2), 12–22. | doi:10.2307/2987588 |
| 3 | Gneiting, T., Balabdaoui, F. & Raftery, A. E. (2007). Probabilistic forecasts, calibration and sharpness. *JRSS-B* 69(2), 243–268. (§14 [75]) | doi:10.1111/j.1467-9868.2007.00587.x |
| 4 | Foster, D. P. & Vohra, R. V. (1998). Asymptotic calibration. *Biometrika* 85(2), 379–390. | doi:10.1093/biomet/85.2.379 |
| 5 | Murphy, A. H. (1973). A new vector partition of the probability score. *J. Applied Meteorology* 12(4), 595–600. | doi:10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2 |
| 6 | Hébert-Johnson, Ú., Kim, M. P., Reingold, O. & Rothblum, G. N. (2018). Multicalibration: calibration for the (computationally-identifiable) masses. *ICML*, PMLR 80:1939–1948. | arXiv:1711.08513 |
| 7 | Zhao, S., Kim, M. P., Sahoo, R., Ma, T. & Ermon, S. (2021). Calibrating predictions to decisions: a novel approach to multi-class calibration. | arXiv:2107.05719 |
| 8 | Hansen, K. et al. (2024). Are GBDTs already multicalibrated? (§14 [45]) | arXiv:2406.06487 |
| 9 | Chong, Y. Y. & Hendry, D. F. (1986). Econometric evaluation of linear macro-economic models. *Rev. Economic Studies* 53(4), 671–690. | doi:10.2307/2297611 |
| 10 | Harvey, D., Leybourne, S. & Newbold, P. (1998). Tests for forecast encompassing. *JBES* 16(2), 254–259. | doi:10.1080/07350015.1998.10524759 |
| 11 | Diebold, F. X. & Mariano, R. S. (1995). Comparing predictive accuracy. *JBES* 13(3), 253–263. | doi:10.1080/07350015.1995.10524599 |
| 12 | Smith, J. E. & Winkler, R. L. (2006). The optimizer's curse: skepticism and postdecision surprise in decision analysis. *Management Science* 52(3), 311–322. | doi:10.1287/mnsc.1050.0451 |
| 13 | Capen, E. C., Clapp, R. V. & Campbell, W. M. (1971). Competitive bidding in high-risk situations. *J. Petroleum Technology* 23(6), 641–653. | doi:10.2118/2993-PA |
| 14 | Thaler, R. H. (1988). Anomalies: the winner's curse. *J. Economic Perspectives* 2(1), 191–202. | doi:10.1257/jep.2.1.191 |
| 15 | Dawid, A. P. (1994). Selection paradoxes of Bayesian inference. *IMS Lecture Notes – Monograph Series* 24, 211–220. | doi:10.1214/lnms/1215463797 |
| 16 | Efron, B. (2011). Tweedie's formula and selection bias. *JASA* 106(496), 1602–1614. | doi:10.1198/jasa.2011.tm11181 |
| 17 | Štrumbelj, E. (2014). On determining probability forecasts from betting odds. *IJF* 30(4), 934–943. | doi:10.1016/j.ijforecast.2014.02.008 |
| 18 | Kovalchik, S. A. (2016). Searching for the GOAT of tennis win prediction. *JQAS* 12(3), 127–138. | doi:10.1515/jqas-2015-0059 |
| 19 | Hubáček, O., Šourek, G. & Železný, F. (2019). Exploiting sports-betting market using machine learning. *IJF* 35(2), 783–796. | doi:10.1016/j.ijforecast.2019.01.001 |
| 20 | Hubáček, O. & Šír, G. (2023). Beating the market with a bad predictive model. *IJF* 39(2), 691–719. | doi:10.1016/j.ijforecast.2022.02.001; arXiv:2010.12508 |
| 21 | Wunderlich, F. & Memmert, D. (2020). Are betting returns a useful measure of accuracy in (sports) forecasting? *IJF* 36(2), 713–722. | doi:10.1016/j.ijforecast.2019.08.009 |
| 22 | Kaunitz, L., Zhong, S. & Kreiner, J. (2017). Beating the bookies with their own numbers — and how the online sports betting market is rigged. | arXiv:1710.02824 |
| 23 | Levitt, S. D. (2004). Why are gambling markets organised so differently from financial markets? *Economic Journal* 114(495), 223–246. | doi:10.1111/j.1468-0297.2004.00207.x |
| 24 | Gneiting, T. (2011). Making and evaluating point forecasts. *JASA* 106(494), 746–762. | doi:10.1198/jasa.2011.r10138 |
| 25 | Cameron, A. C., Gelbach, J. B. & Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. *Rev. Economics and Statistics* 90(3), 414–427. | doi:10.1162/rest.90.3.414 |
| 26 | DerSimonian, R. & Laird, N. (1986). Meta-analysis in clinical trials. *Controlled Clinical Trials* 7(3), 177–188. | doi:10.1016/0197-2456(86)90046-2 |
| 27 | Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scand. J. Statistics* 6(2), 65–70. | JSTOR 4615733 |
| 28 | Spearman, C. (1904). The proof and measurement of association between two things. *Am. J. Psychology* 15(1), 72–101. | doi:10.2307/1412159 |
| 29 | Bonett, D. G. & Wright, T. A. (2000). Sample size requirements for estimating Pearson, Kendall and Spearman correlations. *Psychometrika* 65(1), 23–28. | doi:10.1007/BF02294183 |
| 30 | Howard, S. R., Ramdas, A., McAuliffe, J. & Sekhon, J. (2021). Time-uniform, nonparametric, nonasymptotic confidence sequences. *Annals of Statistics* 49(2), 1055–1080. | doi:10.1214/20-AOS1991 |
| 31 | Efron, B. & Morris, C. (1973). Stein's estimation rule and its competitors. (§14 [47]) | — |

All DOIs above were resolved against Crossref on 2026-10-29 except Holm (JSTOR stable ID) and the
arXiv-only items (resolved on arxiv.org / proceedings.mlr.press).

*Repo-internal evidence (not literature):* `docs/ship_gate.md` (Gate 1 intent, "the gates score the
bulk"), `docs/handoffs/model_improvement_track.md` §3.4, §6.10, §8.2 #0d/#0e/#0f,
`docs/handoffs/live-gap-2.md` §5, `$L/research/MARKET_CENTRE_BRIEF.md` (L1/L2/L3 definitions,
record-gate simulation), `training/pipeline.py:_step_calibrate_temperature` and `_step_build_splits`;
memories `selection_shrink_toward_book_kill`, `book_level_error_inflates_offline_bss`,
`live_gap_root_cause_market_line_selection`, `research_verdicts_ledger`.

## Files (all under `$L/research/gates_vs_profit_reads/`)

`common.py` (loaders, CIs); `r00_live_inventory` (legs by version, quote, platform, date);
`r01_live_cells` (+ .csv; per-cell live table); `r02_offline_reliability` (+ `r02_offline_v1018.parquet`,
`r02_h1_cells.csv`; H1/H2); `r03_decomposition` (+ .csv; the waterfall); `r04_gate_ranking`
(+ `r04_gate_ranking_table.csv`, `r04_spearman_{183_only,all_payouts}.csv`); `r04b_gate_ranking_versions`
(+ `r04b_gate_ranking_{before,current}.txt`, `r04_gate_ranking_table_{before,current}.csv`,
`r04_spearman_{before,current}_*.csv`); `r05_h4_versions` (+ .csv); `r06_h3_season_start`; `r07_h6_stress`;
`r08_l3_wne` (+ `r08_l3_wne_{v1018,current}.csv`); `r09_pickem_lambda_dfs`; `r10_nhl_pickem_cells`;
`r11_line_discrepancy_legs`; `r12_l2_league_and_sides`; `r13_l2_league_generations` (each `.py` + `.txt`).
