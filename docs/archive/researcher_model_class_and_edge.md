# Model class and the edge: are GBDTs the right tool, is there another way to beat the books, should the pool grow, or should the project restart?

In-repo research brief (research-analyst), 2026-10-29 (reads 20:40–22:05 CDT). It ranks the five branches the dispatch set (A keep and fix, B swap or add a learner, C widen the pool, D pivot to where the pick'em line lags the books, E restart) against an information ceiling estimated from this project's own matrices, gamelogs, test sets, live legs and odds archive.

**Scope.** Read-only. Nothing under `src/` was written. No model, no `stat_meta.json` entry, no plan or doc and no git ref was touched. The archive was read only through the copy `/tmp/arch_nhl_i6f_20261029.duckdb` with `SPORTSTRADAMUS_ARCHIVE_READ_ONLY=1`. Scripts and outputs are in `$L/research/model_class_reads/` (r00–r08), where `$L` = `/home/trevor/backups/sportstradamus/2026-10-21-live-gap-2`. The sibling brief `/tmp/researcher_gates_vs_pickem_profit.md` owns the bet-selection levers L1–L3. It rules L1 GO, L2 report-only and L3 KILL. This brief cites its numbers and does not repeat them.

**Conventions.**
- BSS = Brier skill against the book.
- Break-even (BE) is .5464 at the 1.83 per-pick payout (since 2026-10-05) and .5618 at 1.78 (before). The two eras are never pooled.
- Intervals are 90% date-cluster bootstraps (Cameron, Gelbach & Miller 2008, doi:10.1162/rest.90.3.414).
- Each prop is counted once.
- "Rungs" are the archive `ladder` table's per-book de-vigged P(over) at each quoted line. They are read at the pick'em line by exact match or by PAV plus logit interpolation between bracketing lines (Breeden & Litzenberger 1978, house ref [80]).
- "Market Prob" is the served book probability: `1 − get_odds(Line, Market Projection, dist, cv)`. It is a decode of the composite book mean through the cell's family, not a price.
- Owner-facing times are CDT.

---

## TL;DR

1. **GBDTs are not your problem.**
   - Your model reads the same public facts the sportsbooks read.
   - On lines the books price near even, it adds no usable information in the NBA, little in the NFL, and some only in the NHL.
   - A different learner fed the same inputs would hit the same wall. Published benchmarks put the best alternatives within a few percent of a tuned GBDT.
2. **There is room in principle, but almost none of it is reachable.**
   - Someone who knew each player's true expected output could win about 60% of NBA and WNBA picks and 55% of NFL picks.
   - In the NHL and MLB the books already sit at the limit a single game allows, except NHL shots and blocked shots.
   - Most of that room cannot be known before the game, and the books' own prices barely move between when we pick and tip-off.
   - Today's model captures 0–7% of it.
3. **The one other route with positive evidence uses the books instead of beating them.**
   - The rule: take a pick'em leg only when the sportsbooks' own prices at the pick'em line lean 55% or more to one side.
   - In three overlapping backtests those legs won 57–62%, against a 54.6–56.2% break-even.
   - But only about five a day qualify, none of the reads clears break-even with confidence, and NHL legs lost.
   - The rule is model-free, which your current policy forbids. Run it as a paper test you approve, not as a rebuild.
4. **Blending in more projections the way you blend the consensus line will not help** unless a new source knows something the books don't.
   - Projections built from the same public stats are already inside the book's price.
   - A learned blender that switches on game context is the trust layer you killed on 10-04.
   - The only blend change with a measured payoff is fixing how the NHL book price is converted at the pick'em line.
5. **Do not start over.**
   - The learner is one swappable module.
   - The three-year odds archive, the stat collectors, the six-gate test harness and the live ledger are the expensive parts, and no restart can replace them.
   - A restart would reach the same ceiling months from now.
6. **Next: three tests with pass/fail rules fixed in advance, about a week of work each.** Start the clock on the paper test of the books-versus-pick'em rule first. Then run a learner-and-information check that settles question 1 on your own data, and the NHL book-price fix.

---

## Decision table

Grades:
- A: several peer-reviewed or benchmark results in this regime.
- B: peer-reviewed in an adjacent regime, or an in-house read whose interval excludes the bar.
- C: in-house backtests with overlapping samples or post-hoc filters, whose interval touches the bar.
- D: practitioner claim with data.
- E: practitioner claim without data.

Research project = unproven, may not transfer. Engineering project = known method, build cost given.

| Rank | Branch | Expected edge against the ceiling | Evidence | Cost | Cannot fix | Type | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | A. Keep the stack; fix inputs, selection, evaluation | No ceiling gain; fewer, honest legs; profit only where at-median β > 0 (NHL, leak-caveated) | B | Days to 3 weeks per item | NBA β ≈ 0; tail volume; news the market sees first | Engineering (+ L-3 research) | **GO** |
| 2 | D. Books' lean against the pick'em line | .570–.622 on ~5 legs a date against BE .546–.562; no read clears BE with confidence; NHL negative | C (house), E (practitioner) | ~1 week to build; 5–10 months to decide | Volume; platform limits; owner's no-model-free rule | Research | **Report-only test X2, owner-gated** |
| 3 | B. Swap or add a learner | Δβ ≈ 0; Brier ±.001 | A (benchmarks), B (house) | 1–3 weeks per challenger | Missing information; selection on disagreement | Research, likely negative | **Audit X1; no build** |
| 4 | C. Widen the pool | ≈ 0 from same-data members; ≤ ~1% ROI from crowds in mature markets; NHL decode fix ≈ .002 Brier | A, B, C | ~1 week (NHL); months (projections) | Information the books already price | Mixed | **KILL except X3** |
| 5 | E. Restart | Same ceiling | — | Months | Everything above | — | **KILL** |

Detail per branch, in rank order:

**A. Keep the stack; fix inputs, selection and evaluation**
- Expected edge against the ceiling:
  - Raises no ceiling.
  - Removes false model–book disagreements, so live hit rates on fewer legs move toward offline.
  - Profit is plausible only where at-median β > 0. That is NHL today, and possibly inflated by the comps leak that I6f is rebuilding.
  - NBA stays at or below break-even.
- Evidence grade: B (house β, capture, gap waterfall; the sibling's L1).
- Cost:
  - Days to 3 weeks per item.
  - L1 gate: 1–1.5 engineering days (sibling).
  - NHL decode: about 1 week.
  - L-3 inactives/usage features: 2–4 weeks, plus a timestamp-leakage brief.
- What it cannot fix:
  - NBA's β ≈ 0.
  - The tail's thin volume.
  - News the market sees first.
- Type: engineering. L-3's information content is a research question.
- **Verdict: Rank 1, GO.** It is the only branch inside the owner's rules that serves picks.

**D. Pivot to where the pick'em line lags the books**
- Expected edge against the ceiling:
  - Rungs lean ≥ .55 on 1.4% of live props, about 5 legs a date.
  - Those legs won .570–.622 (n 71–314) against BE .546–.562.
  - ROI would be +2% to +11% if real.
  - The NHL subsets lost (n 11–14).
- Evidence grade: C (house); E for practitioner lag claims; B for the Kaunitz template.
- Cost:
  - About 1 week to build a report-only ledger.
  - 1,000–1,600 legs (5–10 months) to decide; a sequential kill can come earlier.
- What it cannot fix:
  - Volume.
  - Platform limits and line pulls.
  - The owner's no-model-free rule.
- Type: research project.
- **Verdict: Rank 2, report-only forward test X2, owner-gated.**

**B. Swap or add a learner class**
- Expected edge against the ceiling:
  - Δβ ≈ 0; Brier ±.001.
  - Benchmark gaps between top tabular learners are a few percent of error.
  - The needed move is β from ≈ 0 to ≳ .40.
- Evidence grade: A for "no universally better tabular learner"; B for "no information gain here" (the GPBoost prototype failed; capture is 0–7%).
- Cost:
  - 1–3 weeks per challenger on 4–5 cells.
  - TabPFN-2.5 weights are non-commercial.
  - TabPFN v2's validated range (≤ 10k rows, ≤ 500 columns) covers only 14 of 66 served cells by rows, and no NFL cell by columns.
- What it cannot fix: missing information; selection on disagreement.
- Type: research project with a likely negative answer.
- **Verdict: Rank 3, one audit (X1) to close it; no build.**

**C. Widen the pool**
- Expected edge against the ceiling:
  - Same-data members (baselines, opportunity × rate, other learners): ≈ 0, because the book encompasses them.
  - Crowd or projection members: about 1% ROI at best in mature markets, against a 4.6-pp pick'em hurdle.
  - A raw-rung book leg at the pick'em line: +.0003 Brier [−.0004, +.0011], which is nothing. NHL is the exception at +.0023 [+.0002, +.0040].
  - A context-gated stacker is R1, already killed.
  - Extremizing is wrong when the book encompasses the model.
- Evidence grade: A (combination theory); B (Brown & Reade; Spann & Skiera); C (house r08).
- Cost:
  - NHL fix: about 1 week.
  - Projections: months of forward archiving, because there is no point-in-time archive.
- What it cannot fix: information the books already price.
- Type: mixed.
- **Verdict: Rank 4, KILL as an edge lever, except the NHL book-price fix (X3).**

**E. Restart**
- Expected edge against the ceiling: the same ceiling.
- Evidence grade: —.
- Cost: months for a solo maintainer.
- What it cannot fix: everything above.
- **Verdict: Rank 5, KILL.**

---

## 1. The information ceiling, per league

**Question.** How much better than the sportsbook could any forecaster be, given how noisy a single game is? Where does the current stack sit inside that room?

**Method** (r01, r05, r07; NHL from the lane's pre-I6f matrix copy):

1. **Model.** Y = μ + ε with E[ε | μ] = 0.
   - The book's information on a row is the in-sample OLS fit of Y on (1, Line, decoded book mean, logit book P(over)), on authentic sportsbook rows. That is four parameters on ≥ 300 rows.
   - Its residual variance MSE_lin is the book's mean-space error.
   - The raw decoded mean is not used as the forecast. For heavy-tailed counts it is the family-consistent mean that reproduces the price, not a point forecast. Example: NBA REB at line 7.5 with P(over) .67 has a decoded mean of 13.6. The raw-mean ratio for NBA REB is 1.880; on the book's information it is 1.289.
2. **Floor.** F = a lower bound on the irreducible noise E[Var(Y | μ)].
   - Poisson for counts.
   - Compound Poisson for yardage, with per-touch variance from the data: receiving 80.9, rushing 33.1, passing 86.2 yd² per touch.
   - Binomial with at-bats known, for MLB.
   - Realistic floor F_R = F + V_u. V_u = Cov(e, u)² / Var(u) is the part of the book's error explained by opportunity surprise u. u is realized minutes / time on ice / snap share / plate appearances / batters faced / QB attempts, minus the player's trailing 10-game mean. A model with no faster news than the book cannot know it.
3. **Oracle at the book's median line.** With r² = MSE_lin / F − 1:
   - BSS\* = (2/π)·asin(r²/(1+r²)).
   - Hit\* = ½ + atan(r)/π on every leg.
   - Share of legs an oracle could bet at 1.83 = 2(1 − Φ(.1166/r)).
4. **Capture.** The out-of-sample incremental R² of the model's own mean over the same book information, on the held-out test sets (authentic rows), as a share of the knowable part (MSE_lin − F_R)/MSE_lin.
5. **Proven-knowable lower bound.** What the market itself learns between our decision time and kickoff at the pick'em line: the RMS of the rung move, expressed in BSS units (rms²/.25) (r02).

**Numbers** (medians over served cells with ≥ 300 authentic rows and an opportunity history; ICC context from `data/icc/` for NBA, NFL and WNBA):

| League | ICC (between-player share) | Upper bound, Poisson floor: BSS\* (hit\*) | Realistic, + opportunity surprise: BSS\* (hit\*) | Oracle's share of legs above BE at 1.83 | Model's capture of the realistic knowable part | At-median β (standalone, market-centre brief) | Proven-knowable: decision → kickoff |
|---|---|---|---|---|---|---|---|
| NBA | .27–.51 | .103 (.632) | **.062 (.601)** | .72 | **2%** (incr. R² .002) | −.11 [−.29, +.11] | rms .0105 → .0004 |
| WNBA | .37–.57 | .103 (.631) | **.076 (.612)** | .75 | **7%** (.009) | — (season over) | — |
| NFL | .41–.63 | .093 (.625) | **.029 (.549)** | .36 | **≈ 0%** (.001) | +.14 pre-2026 / −.19 Jan–Sep / +.33 Oct | rms .0051 → .0001 |
| NHL | no parquet: gamelog + matrices | .000; shots .064, blocked .079 | **.000; shots .043, blocked .059** | 0 (shots .66) | 3% on shots/blocked | +.67 [+.39, +.94] | rms .0061 → .00015 |
| MLB | no parquet: gamelog + matrices | .000 (binomial floor ≈ .035) | **≈ .000–.03** | ≈ 0 | 1% | — | — |

**Reading.**

1. **Low-count markets have no room at all; the book already sits on the noise floor.**
   - NHL goals, assists and points: MSE_lin / F_P .97–.995.
   - MLB batter counts: .82–.995 against the Poisson floor, and about .9–1.06 against the binomial floor.
   - NFL tds, passing tds and interceptions: .80–.96.
   - No learner can beat a book that sits at the irreducible noise.
2. **Mid- and high-count markets leave room in principle.** NBA and WNBA realistic BSS\* is .06–.08, NFL .03, NHL shots and blocked .04–.06.
   - These are upper bounds: every error above the Poisson-plus-opportunity floor is treated as knowable.
   - Pace, game script beyond minutes, and correlated makes are treated as knowable, so NFL carries' .196 is surely too high.
   - Combos understate their floor (the PRA, H+R+RBI and RBI covariance), so their ratios are not ceilings.
   - Opportunity surprise alone explains 2–7% of the book's squared error in single-stat NBA, WNBA and MLB markets, 23–26% in PR and PRA, and 57% in NFL completions (r05).
3. **The proven-knowable part is tiny.** The market learns .0001–.0004 BSS units between decision and kickoff, and the lead-time brief measured .0006–.0013 Brier.
   - The true knowable room lies between that floor and the realistic upper bound.
4. **The current stack sits near the floor.**
   - It captures 0–7% of the realistic upper bound.
   - Its at-median β is ≈ 0 in NBA.
   - On the same 1,669 settled quoted props at the pick'em line (r08), the live Brier scores are: Standalone Prob alone .2638, served Win Prob .2511, Market Prob .2510, raw rungs .2499, coin .25. On live market lines the model alone is worse than a coin, and the blend is rescued by the book.
   - The sibling's offline pick'em β (NBA .02, NFL .18) and post-I6d near-even β (NBA −.01, NFL .12) agree.

---

## 2. Key Findings

### 1. The binding constraint is information, not the learner

- The model trains on about 145–544 public features per cell. Player-prop lines are not features; game lines are (`stats/base.py:get_stat_columns`, `feature_filter.json`).
- **NBA.** The realistic ceiling leaves room (BSS\* .062), but the model's incremental R² over the book is .002. Its at-median β is −.11 [−.29, +.11]. A typical recommended leg (claimed .615) breaks even at 1.83 only with β ≳ .40 (market-centre brief).
- **NHL.** The low-count markets have no room. The model's β of .67 must therefore sit in shots, blocked and hits, in the decode, or in the comps leak that I6f is rebuilding (sibling brief).
- **What this means.** A learner swap cannot move these numbers unless the inputs change. Benchmark gaps between learners are measured in a few percent of error (Finding 2). The gap between β ≈ 0 and β ≈ .40 is a change in information, not in fit.
- The market-efficiency literature sets the prior:
  - Bookmakers set informed prices (Levitt 2004, doi:10.1111/j.1468-0297.2004.00207.x).
  - News is impounded quickly and fully (Croxson & Reade 2014, doi:10.1111/ecoj.12033).
  - NBA absence news is what early lines miss (Dare, Dennis & Paul 2015, doi:10.1016/j.frl.2015.02.004). That is information the model has no faster feed for.

### 2. No benchmark predicts a learner swap will create edge, and the project's history agrees

**Tabular benchmarks:**
- Trees remain the state of the art on medium-sized tabular data (Grinsztajn, Oyallon & Varoquaux 2022, arXiv:2207.08815, house [67]).
- XGBoost beat the deep models even on the datasets of the papers that proposed them. Deep plus XGBoost ensembles beat XGBoost alone only modestly (Shwartz-Ziv & Armon 2022, Information Fusion 81, doi:10.1016/j.inffus.2021.11.011; arXiv:2106.03253).
- "There is still no universally superior solution" (FT-Transformer: Gorishniy et al. 2021, NeurIPS, arXiv:2106.11959).
- SAINT claims a small average edge over GBDTs (Somepalli et al. 2021, arXiv:2106.01342).
- No universal winner: GBDTs are better on skewed and heavy-tailed features, and TabPFN is strong on small data (McElfresh et al. 2023, arXiv:2305.02997, house [68]).
- TabArena (Erickson et al. 2025, arXiv:2506.16791): "gradient-boosted trees are still strong contenders". Deep learning catches up only with large time budgets and ensembling. Foundation models excel on smaller datasets.

**TabPFN:**
- v2 (Hollmann et al. 2025, *Nature* 637:319–326, doi:10.1038/s41586-024-08328-6; house [66] lists vol. 636, the article PDF 637) dominates at ≤ 10,000 samples and ≤ 500 features.
  - Served matrices are 3.1k–120k rows: NBA ≈ 17.8k, WNBA ≈ 20k, MLB median 62k, NHL median 39k, NFL median 9.5k with 544 columns.
  - So v2's validated range covers 14 of 66 served cells by rows, and no NFL cell by columns. It is a sweet spot, not a hard cap.
- TabPFN-2.5 (Grinsztajn et al. 2025, arXiv:2511.08667) is "built for datasets with up to 50,000 data points and 2,000 features". It claims to lead TabArena.
  - Its weights are under a non-commercial licence (research and internal evaluation only; commercial terms through Prior Labs).
  - So it cannot serve a betting product without a commercial licence. The v2 weights are Apache-2.0 plus attribution.

**Distributional boosting siblings:**
- XGBoostLSS and LightGBMLSS are the incumbent class (März 2019, arXiv:1907.03178; März & Kneib 2022, arXiv:2204.00778).
- NGBoost's natural gradient fixes a multi-parameter optimization problem the stack already stabilized; the plan skipped it (Duan et al. 2020, ICML, arXiv:1910.03225).
- CatBoost virtual ensembles give knowledge uncertainty (Malinin, Prokhorenkova & Ustimenko 2021, ICLR, arXiv:2006.10562). That could flag out-of-distribution rows, such as a role change, as a selection guard. It adds no information.

**GPBoost** (Sigrist 2022, *JMLR* 23(232):1–46, arXiv:2004.02653):
- Its advantage is grouped random effects with high-cardinality groups.
- The project already encodes player identity through EB-shrunk means and KNN comps.
- Its NBA prototype failed: it did not beat the EB offset, and its top-volume bias was −2.5 (`docs/archive/nba_overconfidence_investigation.md`).
- The B3 "MZINB vs GPBoost" fork was superseded by the B1.5 mean-head pivot and never re-entered (`docs/archive/gbdt_mean_regression_context.md`).

**Bayesian hierarchical and state-space player rates:**
- Sources: Efron & Morris 1975, doi:10.1080/01621459.1975.10479864; Brown 2008, doi:10.1214/07-AOAS138; Jensen, McShane & Wyner 2009, doi:10.1214/09-BA424; Glickman & Stern 1998, doi:10.1080/01621459.1998.10474084.
- These shrink noisy rates well. The gain is in small samples and early season, where the book also regresses.
- The plan's §6.6 small-sample ladder (EB, then TabPFN head-to-head, then HB) already sequences them for calibration, not edge.
- Metric-reliability tools (Franks et al. 2016, doi:10.1515/jqas-2016-0098) say the same: stability and discrimination of a rate are limited by the sample, not the model.

**Simulation:**
- A possession-level Markov simulation forecast as well as other statistical methods, and bookmaker odds were the best forecasters (Štrumbelj & Vračar 2012, doi:10.1016/j.ijforecast.2011.01.004).
- A simulation re-expresses the same inputs. It is not new information.

**House limits.** LightGBMLSS binds four ways (§6.6): a closed family, weak scale/shape gradients, Fisher pathologies, and a shared schedule. The escalations (ordinal threshold stack, mixture head, comp-PMF) are shape and calibration fixes for g4. None addresses edge.

### 3. Market-relative formulations are the right objective, and on this data they read ≈ 0

- Hubáček, Šourek & Železný 2019 (*IJF* 35(2):783–796, doi:10.1016/j.ijforecast.2019.01.001):
  - Penalizing correlation with the bookmaker's implied probabilities made NBA models (2007–2014) profitable where accuracy-only models were not.
  - The objective is the part of the outcome the book gets wrong.
  - This project's version of that test is the at-median encompassing β and r07's incremental R² over book information. Both read ≈ 0 in NBA and NFL.
  - A residual learner on this data would learn ≈ 0 and default to the book.
- Profit is a poor accuracy measure: profits arise by luck or design without accuracy (Wunderlich & Memmert 2020, *IJF* 36(2):713–722, doi:10.1016/j.ijforecast.2019.08.009).
- Calibration-based model selection beat accuracy-based selection on NBA betting ROI (+34.7% against −35.2%) (Walsh & Joshi 2024, *Machine Learning with Applications* 16:100539, arXiv:2303.06021).
  - That is one season and one market (grade C). It supports the house calibration focus, but calibration certifies no information.
- Treat any 22-date read as a hypothesis:
  - Significant single-season "inefficiencies" appear even under full efficiency and do not persist (Winkelmann, Ötting, Deutscher & Makarewicz 2024, *J. Sports Econ.* 25(1):54–97, doi:10.1177/15270025231204997).
  - A published tennis "buzz" edge mostly vanished after one outlier bet was corrected. Its one surviving strategy stopped working after 2020 (Clegg & Cartlidge 2025, *IJF* 41(2):798–802, arXiv:2306.01740).

### 4. A bigger pool helps only with new information; extremizing and context gating are wrong here

- **When combining helps.**
  - Combination gains scale with the information members do not share. The optimal weight on a member another encompasses is 0 (Bates & Granger 1969, doi:10.2307/3008764; Clemen 1989, doi:10.1016/0169-2070(89)90012-5; Wang, Hyndman, Li & Kang 2023, *IJF* 39(4):1518–1547, doi:10.1016/j.ijforecast.2022.11.005).
  - The test is encompassing (Harvey, Leybourne & Newbold 1998, doi:10.1080/07350015.1998.10524759; Clements & Harvey 2010, doi:10.1002/jae.1097). The book encompasses the model in NBA.
- **The combination puzzle.** Estimated weights' error eats small gains (Smith & Wallis 2009, doi:10.1111/j.1468-0084.2008.00541.x; Claeskens et al. 2016, doi:10.1016/j.ijforecast.2015.12.005).
  - A third or fourth member from the same data adds weight-estimation noise and no signal.
- **Pool form.**
  - The log pool in use is the externally Bayesian choice (Genest & Zidek 1986, house [54]).
  - A linear pool of calibrated members is over-dispersed (under-confident) and needs recalibration (Hora 2004, house [58]; Ranjan & Gneiting 2010, doi:10.1111/j.1467-9868.2009.00726.x).
  - The house BLP probe was a wash (§3.4).
- **Extremizing is wrong here.**
  - It is justified by information diversity among forecasters (Baron et al. 2014, *Decision Analysis* 11(2):133–145, doi:10.1287/deca.2014.0293; Satopää et al. 2014, *IJF* 30(2):344–356, doi:10.1016/j.ijforecast.2013.09.009; Satopää, Pemantle & Ungar 2016, *JASA* 111(516):1623–1633, doi:10.1080/01621459.2015.1100621).
  - When one member's information contains the others' (book ⊇ model in NBA), the optimal aggregate is that member. Extremizing adds exactly the overconfidence the live tail shows.
- **Context-gated stacking.**
  - Sources: Yao, Vehtari, Simpson & Gelman 2018, doi:10.1214/17-BA1091; Deutsch, Granger & Teräsvirta 1994, doi:10.1016/0169-2070(94)90049-3; Elliott & Timmermann 2005, doi:10.1111/j.1468-2354.2005.00361.x.
  - A gate on market observables is the R1 trust layer killed 2026-10-04 (#0d: "model-free in disguise").
  - A gate on non-market context still faces the puzzle.
- **External crowds and projections.**
  - Aggregated amateur tips held information not in soccer prices: +1.317% average return over 68,339 events (Brown & Reade 2019, *EJOR* 272(3):1073–1081, doi:10.1016/j.ejor.2018.07.015).
  - Betting odds and prediction markets beat tipsters on their own (Spann & Skiera 2009, *J. Forecasting* 28(1):55–72, doi:10.1002/for.1091).
  - That is the best documented case, and it is about 1% ROI in a mature market. The pick'em hurdle is 4.6 pp of hit rate (about 9% of stake).
  - No point-in-time projection archive exists in the repo; FantasyPoints snapshots are stats, not projections. A projection member therefore needs months of forward archiving before its first honest test.

### 5. The "anti-calibrated Market Prob" is the decode, not the market

- Market Prob decodes the composite book mean through the cell's family at the pick'em line.
- **The decode leans.** On 4,912 quoted live legs since 10-05 it sits −.017 below the raw rungs on average (MAD .0245) (r03).
  - Count families lean Under: NBA DPO −.045, NBA NegBin −.074, NFL DPO −.042, NHL DPO −.037, MLB DPO −.027, WNBA DPO −.038.
  - SkewNormal is about 0 (NBA +.006, NFL −.003).
- **Its leans are mostly manufactured.**
  - Market Prob ≥ .55: 804 legs hit .535 [.506, .562], ROI −2.7%.
  - On the 597 legs where Market Prob leans ≥ .05 but the rungs sit within .03 of even, its side hits .516.
  - The raw rungs' own ≥ .55 leans hit .59–.62 (Finding 6).
- **In Brier the decode costs almost nothing:** +.0003 [−.0004, +.0011] on 4,930 settled props without NHL goals or MLB home runs (r08). Most legs sit near .5 either way.
  - NHL is the exception: +.0023 [+.0002, +.0040] on 931 props (the #0g hurdle clamp plus the DPO decode).
- **The book set is soft.** The 2026 archive is soft US books. Pinnacle is in it only through 2024-03-20 and Circa only through 2023-07, so "sharp books" here means a soft-book consensus.
  - The only prop comparison found (practitioner, grade D) puts Pinnacle's NFL prop closing Brier .0015–.0018 below DraftKings', on about 6,300 lines. Sharp and soft books differ little on props.
- **De-vigging** is proportional (multiplicative) per book on two-way rungs; a one-sided rung assumes a 6.5% hold (`helpers/distributions.py:no_vig_odds`, called from `moneylines._devig_ladder`). See Štrumbelj 2014, doi:10.1016/j.ijforecast.2014.02.008, and Clarke, Kovalchik & Ingram 2017, house [55].
  - At near-even lines the method moves P by < .005.
  - On 0.5-line longshot props the one-sided residual is +.01 to +.04 (#0f).
- **Quote age matters.** On the ≥ .55 legs:
  - Rungs ≤ 1 h old: n 61, hit .656.
  - Rungs > 3 h old: n 24, hit .458.
  - The 90th-percentile age is 21 h.
  - This is a post-hoc split. The forward test freezes it as a secondary arm.
- **Reconciliation.** The practitioner claim is "de-vigged books beat pick'em". On this data it is directionally right, but only where the books actually lean, on fresh quotes. Market Prob's leans are a decode artifact and must never stand in for the books' view.

### 6. The books' lean at the pick'em line is the only positive edge signal, and it is thin

**Three overlapping reads**, count-once, date-clustered (bars: .5464 at 1.83, .5618 at 1.78):

| Read | Window | Leg universe | P_side ≥ .55: n, hit [90% CI] | ROI |
|---|---|---|---|---|
| r02 exact line | games 10-09…10-26 | scored legs, decision-time rungs | 71, .606 [.511, .684] | +10.8% |
| r02 interpolated | same | same | 74, .622 [.529, .694] | +13.8% |
| r03 interpolated | 10-05…10-26, 16 dates | scored legs, decision-time rungs | 88, .591 [.500, .660] | +8.2% |
| r04b archive backtest | 07-12…10-26, 61 dates | Underdog main rung at even at 11:00 CDT; one-sided-looking props removed | 314, .570 [.525, .614] (BE .560) | +1.8% |
| r04b at ≥ .58 | same | same | 157, .611 [.559, .660] | +9.0% |

**Splits:**
- r03, NBA Underdog: 56 legs at .661. NBA Sleeper: 13 at .462. NHL: 12 at about .42.
- r03, model agrees: 55 at .582. Model disagrees: 33 at .606. The model adds nothing on these legs.
- r03, rungs from ≥ 3 books: 72 at .611 [.537, .681].
- r04b by era: 1.78, 268 at .582. 1.83, only 46 at .500 (NBA 33 at .545, NHL 11 at .364).
- r04b WNBA: 255 at .588 [.542, .632]. Its lower bound sits below .5618.

**Calibration of the rungs at Underdog's line** (r04b): (.50, .52] .503; (.52, .55] .516; (.55, .58] .529; (.58, .62] .621; (.62, .65] .585. Leans under .55 carry nothing, and they are 95–99% of props (r04b 95.5%, live r03 98.6%).

**Artifact caught.**
- Before filtering, r04 read P_side ≥ .55 on 3,628 legs at .827.
- MLB 0.5-line props carried it: home runs 1,592 at .889, stolen bases 885 at .886, doubles 503 at .847.
- These are one-sided "higher only" offers stored at an even price, not two-sided picks. A practitioner post shows the same pattern on PrizePicks alt lines (grade E).
- Any backtest of this rule must exclude them.

**Lag** (r04b, Underdog polls 11:00–18:00 CDT the same day):
- Where the books lean ≥ .55, Underdog later moved the leg 35% of the time, with median about 2 h. 72% of those moves went toward the books (n 107).
- Near-even legs moved 46% of the time, toward the books 51%.
- The book side hit .535 when Underdog later moved toward it, against .500 when not (lean > .52; n 471 / 252).
- Underdog does follow the books with a lag, but the polls (about 14 a day) cannot resolve the 15–30-minute lags practitioners claim (XCLSV 2026, grade E).

**Precedent.**
- Betting soft books against the cross-book consensus made money in football until the accounts were restricted (Kaunitz, Zhong & Kreiner 2017, arXiv:1710.02824). Limits are the practical failure mode.
- The side the books moved toward between decision and kickoff (|dp| ≥ .02) hit .546 [.407, .660] (n 97). The model bet that side .371 of the time (r02).

**Power.**
- To establish a true .58 against .5464 at one-sided α .10 and power .8 needs about 980 legs, or about 1,200 with date clustering.
- At about 5 qualifying legs a date that is 5–8 months, and 10 months at α .05.

### 7. Structural, simulation and hierarchical members carry no information the book lacks

- **Opportunity × efficiency.**
  - It is a re-expression of features the GBDT already has (minutes, usage and rate features).
  - It adds information only through a better opportunity forecast than the book's, which means news.
  - T5's multiplicative factorization was killed for +27% predictive-variance inflation (Goodman 1960, house [1]; §3.4).
  - r05 bounds the stake: unanticipated opportunity is 2–7% of the book's squared error in single-stat markets.
- **Simulation and Bayesian hierarchical rates.** These are calibration tools, not edge (Finding 2).
- **DFS contest strategy** (Haugh & Singal 2021, *Management Science* 67(1):72–92, doi:10.1287/mnsc.2019.3528) gets its edge from opponent modelling in guaranteed-prize contests. It does not transfer to fixed-payout pick'em.

### 8. A restart costs months and buys nothing a module swap does not

The inventory is in §4. The learner-specific surface is the LightGBMLSS fit and predict paths, plus four custom distribution classes (`skew_normal.py` 196, `skew_normal_centered.py` 171, `double_poisson.py` 233, `hurdle.py` 309 lines). The plan's module footprint already names the swap path: a new `is_*` flag, a `model_prob` load branch, and an extended determinism gate. A swap takes weeks. A restart re-collects the same data, re-learns the same lessons, and lands on the ceiling in §1.

---

## 3. Recommendation and routing protocol

**Branch ranking:**
1. **A, GO.** Adopt the sibling's L1 live-record gate; retrospectively it releases nothing in any league (lower bounds NBA .480, NFL .470, NHL .521 against .5464). Fix the NHL book price (X3). Brief, then pilot, L-3 inactives/usage. NBA cannot be made profitable this way; it can be made honest.
2. **D, report-only forward test (X2).** Owner-gated, because acting on it would be the model-free engine the owner ruled out. A paper ledger is research, not an engine.
3. **B, one audit (X1), no build.** Expected outcome: KILL for edge. Shape and calibration escalations stay in §6.6.
4. **C, KILL** as an edge lever, except X3. Do not add projection, baseline or opportunity members. Do not extremize. Do not gate the pool on market context.
5. **E, KILL.**

### Pre-registered experiments, ranked

Rank 1 is X2 because its clock is the long pole. If the owner declines X2, X1 moves to rank 1.

**X2. Report-only paper test: the books' lean at the pick'em line (Branch D, question 2).**
- **Rule** (frozen before the first read):
  - A live Underdog or Sleeper pick'em leg qualifies when the latest per-book rung snapshot at or before `Scored At` places one side at P ≥ .55 at the pick'em line. Use the exact-line mean when quoted, else PAV plus logit interpolation between bracketing rungs, with no extrapolation.
  - The pick is that side.
  - One leg per prop per date.
  - It is graded at its own era and platform break-even: 1.83 → .5464; Sleeper at its payout. Never pool eras.
  - Excluded: 0.5-line props of rare-event markets (MLB home runs, stolen bases, doubles, triples, singles, walks, runs, RBI, hits; NHL goals, power-play points; NFL tds), and any prop whose books put a side above .65.
- **Arms:**
  - Primary: the rule.
  - Secondary, Holm over three: ≥ 3 books; rung age ≤ 1 h; model agrees.
  - NHL and each platform are reported separately.
- **Statistic:**
  - Hit rate against break-even with a date-cluster bootstrap CI.
  - Weekly looks through an anytime-valid confidence sequence (Howard, Ramdas, McAuliffe & Sekhon 2021, doi:10.1214/20-AOS1991).
- **Accept:** after ≥ 1,000 qualifying primary legs over ≥ 60 dates, the 90% lower bound exceeds break-even, or the confidence sequence's lower bound crosses it earlier.
- **Kill:** at any weekly look, the confidence sequence's upper bound falls below break-even; or the point hit is ≤ .53 after 400 legs.
- **Harness.** This is not `ship scorecard`, which grades test-set rows, not live legs. It is `realized.py`'s settled legs plus a read-only archive copy, computed nightly. There is no serving change and no stake.
- **Cost:** about 1 week. Decision horizon 5–10 months. Research project.

**X1. Learner and information audit (Branch B, question 1, and the same-data half of question 3).**
- **Cells** (frozen):
  - NBA AST, NBA PRA and NBA FG3M (β ≈ 0 league).
  - NFL receptions (small n).
  - NHL shots (β > 0 control).
- **Fit rule.** Each arm is fit on the cell's cached matrix rows dated before the first test-set date, and predicts the existing test-set rows. Research directory only; no `src/` edit.
- **Arms:**
  - **Arm 0:** the incumbent's test-set CSV, copied to `$L/research/model_class_reads/x1/`.
  - **Arm 1, a different learner.** CatBoost with virtual ensembles (Apache-2.0), or TabM / FT-Transformer (MIT). TabPFN v2 only on cells with ≤ 10k rows and ≤ 500 columns. TabPFN-2.5 only if its licence's evaluation clause is confirmed. It predicts the cell's mean, and P(over) uses the incumbent's served shape columns, so only the location learner changes.
  - **Arm 2, market-relative (Hubáček's objective in its simplest form).** LightGBM on the residual Y − B̂, where B̂ is r07's book-information prediction on authentic rows; forecast = B̂ + residual. This is a gray zone under #0d for any serving use, so it is offline only here.
- **Primary statistics:**
  1. Out-of-sample incremental R² of the arm's mean over book information on authentic test rows, with a date-cluster CI (r07 method).
     - Incumbent values: NBA AST .009, PRA .000, FG3M .002; NFL receptions .000; NHL shots .003.
  2. At-median encompassing β of the arm's standalone P against the book P, on sportsbook rows within .06 of even, with a date-block CI (the market-centre brief's statistic).
- **Secondary statistic:** `sportstradamus ship scorecard --baseline x1/<cell>__arm0.csv --candidate x1/<cell>__armK.csv --scorecard-out x1/<cell>__armK_scorecard.csv` on fused copies.
- **Accept (Branch B stays open):** in ≥ 2 of the 3 NBA cells, some arm meets all three conditions:
  - Its incremental-R² 90% lower bound is ≥ .005 above the incumbent's point.
  - Its β lower bound is > 0 and above the incumbent's point.
  - No gate regresses.
- **Kill (close Branch B for edge; TabPFN and learner-swap rows become "calibration/shape only"):** in all five cells, both arms' incremental-R² and β intervals cover the incumbent's.
- **Prior:** about 80% kill.
- **Cost:** about 1 week of main-session work. Research project with a likely negative answer, worth running because it answers the owner with house data.

**X3. NHL book price at the pick'em line (Branches A and C, engineering).**
- **Candidate.** Where at least two books' rungs bracket or equal the pick'em line, the book leg becomes the raw rung P. It is mapped to the family's book mean by solving the cell's family at that line with the book cv, so the parameter-space `fused_loc` pool is kept. Otherwise the current decode stands.
- **Grading:**
  1. Replay on settled NHL props since 10-05, r08 method, paired against the served Market Prob and Win Prob.
  2. `ship scorecard` on NHL test-set CSV copies, with the book mean re-solved on rows whose decode clamps.
- **Accept:**
  - Book-leg Brier difference (candidate − served): 90% upper bound < 0.
  - Fused Brier difference: upper bound ≤ 0.
  - No NHL gate flips to fail on the CSV copies.
- **Kill:** the book-leg CI covers 0 after 1,500 settled props, or any gate flip.
- **Note.** A naive logit pool of `Standalone Prob` with the raw rung at the same w is worse than the served chain (−.0041 [−.0062, −.0022], r08), so do not change the pool form.
- **Cost:** about 1 week. Research-gated by #0g; this brief and r08 are the evidence.

**Do not build:**
- A context-gated stacker on market observables (R1).
- Extremizing or temperature-sharpening of the pool.
- An opportunity × rate member or replacement.
- A projection member without a point-in-time archive.
- TabPFN-2.5 in production without a commercial licence.
- A restart.

---

## 4. Restart verdict and asset inventory

**Verdict: KILL.** The parts a restart would throw away are the parts that are not broken, and the part that limits profit (information) comes with the data, not the code.

| Asset | Size | Replaceable? |
|---|---|---|
| Odds / lines / ladder archive (DuckDB) | 2.85 GB, polls since 2023 | **No.** Decision-time prop prices cannot be re-bought; Underdog and Sleeper history is not for sale; Odds API history costs credits |
| Live ledger (`history.parquet`, `parlay_hist/`, `realized.py`) | — | **No.** It is the only live evidence (count-once from 10-09) |
| Gamelogs, 5 leagues | 118 MB | Re-fetchable; weeks of collector work |
| Collectors (FantasyPoints, CTG, Savant) | 4.1k lines | Rebuildable |
| Training matrices, 66 served cells, 145–544 columns | 1.3 GB | Rebuildable from gamelogs and archive (NFL cold rebuild about 6–7 h) |
| Test sets / models | 6.3 GB / 449 MB | Regenerated by training |
| Code: `src/` | 92k lines (training 25.6k, stats 14.2k, dashboard 12.3k, scripts 11.7k, prediction 7.7k, helpers 6.4k, collectors 4.1k, strategies 2.6k, books 0.6k) | The learner-specific part is about 0.9k lines of distribution classes plus the fit and HPO paths in `training/pipeline.py`; the rest is learner-agnostic |
| Tests | 76k lines | Encode the regressions already paid for |
| Six-gate harness, sweep/confirm engine, I6d levelling | — | Rebuildable, but each encodes months of lessons (Gate 4 PIT-KS, Gate 6 anti-shrinkage, the pickem cohort, book levelling) |
| History | 2,758 commits since 2023-05-05 | — |

**Costs compared:**
- A learner swap on a few cells: 1–3 weeks.
- Branch D's tooling: about 1 week on the existing archive.
- A restart to parity (collection, features, distributional training, fusion, gates, serving): several months for one person, ending at the same §1 ceiling.

A narrower "restart" the owner could consider is a scope cut, not a rewrite: fewer leagues and cells, focused where β > 0. That needs no new code.

---

## 5. Reality checks

**The ceiling is an upper bound.**
- It treats every error above the Poisson-plus-opportunity floor as knowable.
- Pace, game script beyond minutes, and correlated makes are not in the floor, so NFL carries' .196 and NBA's .062 are too high.
- MLB's Poisson floor is too high, because hits are binomial in about 4 AB. The binomial version is ≈ .035.
- Combos understate their floor.
- If the floor were much too low everywhere, the room would shrink further and every conclusion here would strengthen.

**β and capture are measured on test sets whose book rows I6d just levelled.** The post-I6d retrains moved NBA served BSS to ±.02 (#0f), consistent with β ≈ 0. A retrain could move β by about ±.1, not to .40.

**The tail-rule evidence is weak in exactly the era that pays.**
- The reads overlap: r02 ⊂ r03 in time, and r04b shares October.
- The filters were chosen after r04's MLB artifact. The (.65 cap, rare-event list) are frozen in X2.
- The 1.83-era reads disagree: NBA .661 at scoring time against .545 at 11:00 CDT.
- NHL is negative in every read (n 11–14).
- WNBA's 255 legs are 1.78-era, and the WNBA market is thinner (practitioners say softer, grade E). They may not transfer.

**Platform response.**
- Underdog moved 35% of leaning legs within hours.
- Pick'em platforms pull or adjust lines on news, and limit accounts (Kaunitz 2017's experience with books).
- Even if X2 passes, the dollar value is about 5 legs a day at +4–10% ROI.

**Projected effects.**
- X1: no change in β (prior about 80%).
- X2: a positive point estimate of +2 to +6 pp over break-even is likely; a conclusive pass in 2026 is unlikely.
- X3: about .002 Brier on NHL quoted props; the tail effect is unknown.

**What would make this brief wrong:**
- A new information source the books price late: confirmed inactives faster than the line, private minutes projections, or a sharp-book feed. That source would reopen A (L-3) and C (as a member).
- X1 finding β > 0 in NBA with a different learner, which would contradict every benchmark and the GPBoost history.

---

## 6. Open questions / caveats (for §8.2)

1. **#0e(iv), NHL.** The low-count markets (goals, assists, points) have no room at all, yet NHL's at-median β is +.67. Read β by NHL market after the I6f comps rebuild. If it survives only on shots, blocked and hits, it is real room. If it collapses, it was the leak.
2. **Reachability of the NBA room.** Decompose the NBA book residual by inactive and usage news (L-3) before any feature build. Opportunity surprise is 2–7% of single-stat book error and 23–26% of PR/PRA, and how much of it a timestamp-honest feed could know is unmeasured.
3. **WNBA 2027.** WNBA has the largest tail-rule sample (255 at .588, 1.78-era). Re-read it at the 2027 opener against .5464.
4. **No sharp book after 2024.** Pinnacle was in the archive only through 2024-03-20 and Circa through 2023-07. A sharp feed could strengthen D; cost and terms are unknown.
5. **Lag resolution.** The archive's roughly 14 Underdog polls a day cannot measure sub-hour lags. Practitioner claims of 15–30 minutes stay untested.
6. **TabPFN-2.5 licence.** Confirm whether its "internal evaluation" clause covers X1 before using it.
7. **House reference [66]** gives *Nature* vol. 636 for Hollmann et al. 2025; the article PDF gives 637:319–326.

---

## 7. Bibliography

| # | Source | Identifier |
|---|---|---|
| 1 | Grinsztajn, L., Oyallon, E. & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on tabular data? *NeurIPS* (house [67]) | arXiv:2207.08815 |
| 2 | Shwartz-Ziv, R. & Armon, A. (2022). Tabular data: deep learning is not all you need. *Information Fusion* 81 | doi:10.1016/j.inffus.2021.11.011; arXiv:2106.03253 |
| 3 | McElfresh, D. et al. (2023). When do neural nets outperform boosted trees on tabular data? *NeurIPS* (house [68]) | arXiv:2305.02997 |
| 4 | Gorishniy, Y., Rubachev, I., Khrulkov, V. & Babenko, A. (2021). Revisiting deep learning models for tabular data. *NeurIPS* | arXiv:2106.11959 |
| 5 | Somepalli, G. et al. (2021). SAINT: improved neural networks for tabular data via row attention and contrastive pre-training | arXiv:2106.01342 |
| 6 | Erickson, N., Purucker, L., Tschalzev, A., Holzmüller, D., Desai, P. M., Salinas, D. & Hutter, F. (2025). TabArena: a living benchmark for machine learning on tabular data | arXiv:2506.16791 |
| 7 | Hollmann, N. et al. (2025). Accurate predictions on small data with a tabular foundation model. *Nature* 637:319–326 (house [66]) | doi:10.1038/s41586-024-08328-6 |
| 8 | Grinsztajn, L. et al. (2025). TabPFN-2.5: advancing the state of the art in tabular foundation models (non-commercial weights licence) | arXiv:2511.08667 |
| 9 | Duan, T. et al. (2020). NGBoost: natural gradient boosting for probabilistic prediction. *ICML* | arXiv:1910.03225 |
| 10 | März, A. (2019). XGBoostLSS: an extension of XGBoost to probabilistic forecasting | arXiv:1907.03178 |
| 11 | März, A. & Kneib, T. (2022). Distributional gradient boosting machines | arXiv:2204.00778 |
| 12 | Malinin, A., Prokhorenkova, L. & Ustimenko, A. (2021). Uncertainty in gradient boosting via ensembles. *ICLR* | arXiv:2006.10562 |
| 13 | Sigrist, F. (2022). Gaussian process boosting. *JMLR* 23(232):1–46 | arXiv:2004.02653 |
| 14 | Hubáček, O., Šourek, G. & Železný, F. (2019). Exploiting sports-betting market using machine learning. *IJF* 35(2):783–796 | doi:10.1016/j.ijforecast.2019.01.001 |
| 15 | Kaunitz, L., Zhong, S. & Kreiner, J. (2017). Beating the bookies with their own numbers — and how the online sports betting market is rigged | arXiv:1710.02824 |
| 16 | Wunderlich, F. & Memmert, D. (2020). Are betting returns a useful measure of accuracy in (sports) forecasting? *IJF* 36(2):713–722 | doi:10.1016/j.ijforecast.2019.08.009 |
| 17 | Walsh, C. & Joshi, A. (2024). Machine learning for sports betting: should model selection be based on accuracy or calibration? *Machine Learning with Applications* 16:100539 | arXiv:2303.06021 |
| 18 | Štrumbelj, E. (2014). On determining probability forecasts from betting odds. *IJF* 30(4):934–943 | doi:10.1016/j.ijforecast.2014.02.008 |
| 19 | Štrumbelj, E. & Vračar, P. (2012). Simulating a basketball match with a homogeneous Markov model and forecasting the outcome. *IJF* 28(2):532–542 | doi:10.1016/j.ijforecast.2011.01.004 |
| 20 | Winkelmann, D., Ötting, M., Deutscher, C. & Makarewicz, T. (2024). Are betting markets inefficient? Evidence from simulations and real data. *J. Sports Economics* 25(1):54–97 | doi:10.1177/15270025231204997 |
| 21 | Clegg, L. & Cartlidge, J. (2025). Not feeling the buzz: correction study of mispricing and inefficiency in online sportsbooks. *IJF* 41(2):798–802 | arXiv:2306.01740 |
| 22 | Bates, J. M. & Granger, C. W. J. (1969). The combination of forecasts. *Operational Research Quarterly* 20(4):451–468 | doi:10.2307/3008764 |
| 23 | Clemen, R. T. (1989). Combining forecasts: a review and annotated bibliography. *IJF* 5(4):559–583 | doi:10.1016/0169-2070(89)90012-5 |
| 24 | Smith, J. & Wallis, K. F. (2009). A simple explanation of the forecast combination puzzle. *Oxford Bull. Econ. Stat.* 71(3):331–355 | doi:10.1111/j.1468-0084.2008.00541.x |
| 25 | Claeskens, G., Magnus, J. R., Vasnev, A. L. & Wang, W. (2016). The forecast combination puzzle: a simple theoretical explanation. *IJF* 32(3):754–762 | doi:10.1016/j.ijforecast.2015.12.005 |
| 26 | Wang, X., Hyndman, R. J., Li, F. & Kang, Y. (2023). Forecast combinations: an over 50-year review. *IJF* 39(4):1518–1547 | doi:10.1016/j.ijforecast.2022.11.005; arXiv:2205.04216 |
| 27 | Harvey, D. I., Leybourne, S. J. & Newbold, P. (1998). Tests for forecast encompassing. *JBES* 16(2):254–259 | doi:10.1080/07350015.1998.10524759 |
| 28 | Clements, M. P. & Harvey, D. I. (2010). Forecast encompassing tests and probability forecasts. *J. Appl. Econometrics* 25(6):1028–1062 | doi:10.1002/jae.1097 |
| 29 | Genest, C. & Zidek, J. (1986). Combining probability distributions. *Statistical Science* 1(1):114–135 | house [54] |
| 30 | Hora, S. C. (2004). Probability judgments for continuous quantities. *Management Science* 50(5):597–604 | house [58] |
| 31 | Ranjan, R. & Gneiting, T. (2010). Combining probability forecasts. *JRSS-B* 72(1):71–91 (house [56]) | doi:10.1111/j.1467-9868.2009.00726.x |
| 32 | Baron, J., Mellers, B. A., Tetlock, P. E., Stone, E. & Ungar, L. H. (2014). Two reasons to make aggregated probability forecasts more extreme. *Decision Analysis* 11(2):133–145 | doi:10.1287/deca.2014.0293 |
| 33 | Satopää, V. A., Baron, J., Foster, D. P., Mellers, B. A., Tetlock, P. E. & Ungar, L. H. (2014). Combining multiple probability predictions using a simple logit model. *IJF* 30(2):344–356 | doi:10.1016/j.ijforecast.2013.09.009 |
| 34 | Satopää, V. A., Pemantle, R. & Ungar, L. H. (2016). Modeling probability forecasts via information diversity. *JASA* 111(516):1623–1633 | doi:10.1080/01621459.2015.1100621; arXiv:1406.2148 |
| 35 | Yao, Y., Vehtari, A., Simpson, D. & Gelman, A. (2018). Using stacking to average Bayesian predictive distributions. *Bayesian Analysis* 13(3):917–1003 | doi:10.1214/17-BA1091 |
| 36 | Deutsch, M., Granger, C. W. J. & Teräsvirta, T. (1994). The combination of forecasts using changing weights. *IJF* 10(1):47–57 | doi:10.1016/0169-2070(94)90049-3 |
| 37 | Elliott, G. & Timmermann, A. (2005). Optimal forecast combination under regime switching. *Int. Econ. Rev.* 46(4) | doi:10.1111/j.1468-2354.2005.00361.x |
| 38 | Brown, A. & Reade, J. J. (2019). The wisdom of amateur crowds: evidence from an online community of sports tipsters. *EJOR* 272(3):1073–1081 | doi:10.1016/j.ejor.2018.07.015 |
| 39 | Spann, M. & Skiera, B. (2009). Sports forecasting: a comparison of the forecast accuracy of prediction markets, betting odds and tipsters. *J. Forecasting* 28(1):55–72 | doi:10.1002/for.1091 |
| 40 | Efron, B. & Morris, C. (1975). Data analysis using Stein's estimator and its generalizations. *JASA* 70(350):311–319 | doi:10.1080/01621459.1975.10479864 |
| 41 | Brown, L. D. (2008). In-season prediction of batting averages: a field test of empirical Bayes and Bayes methodologies. *Ann. Appl. Stat.* 2(1):113–152 | doi:10.1214/07-AOAS138 |
| 42 | Jensen, S. T., McShane, B. B. & Wyner, A. J. (2009). Hierarchical Bayesian modeling of hitting performance in baseball. *Bayesian Analysis* 4(4):631–652 | doi:10.1214/09-BA424 |
| 43 | Glickman, M. E. & Stern, H. S. (1998). A state-space model for National Football League scores. *JASA* 93(441):25–35 | doi:10.1080/01621459.1998.10474084 |
| 44 | Franks, A. M., D'Amour, A., Cervone, D. & Bornn, L. (2016). Meta-analytics: tools for understanding the statistical properties of sports metrics. *JQAS* 12(4):151–165 | doi:10.1515/jqas-2016-0098 |
| 45 | Haugh, M. B. & Singal, R. (2021). How to play fantasy sports strategically (and win). *Management Science* 67(1):72–92 | doi:10.1287/mnsc.2019.3528 |
| 46 | Levitt, S. D. (2004). Why are gambling markets organised so differently from financial markets? *Economic Journal* 114(495):223–246 | doi:10.1111/j.1468-0297.2004.00207.x |
| 47 | Croxson, K. & Reade, J. J. (2014). Information and efficiency: goal arrival in soccer betting. *Economic Journal* 124(575):62–91 | doi:10.1111/ecoj.12033 |
| 48 | Dare, W. H., Dennis, S. A. & Paul, R. J. (2015). Player absence and betting lines in the NBA. *Finance Research Letters* 13:130–136 | doi:10.1016/j.frl.2015.02.004 |
| 49 | Moskowitz, T. J. (2021). Asset pricing and sports betting. *J. Finance* 76(6):3153–3209 | doi:10.1111/jofi.13082 |
| 50 | Smith, J. E. & Winkler, R. L. (2006). The optimizer's curse. *Management Science* 52(3):311–322 | doi:10.1287/mnsc.1050.0451 |
| 51 | Howard, S. R., Ramdas, A., McAuliffe, J. & Sekhon, J. (2021). Time-uniform, nonparametric, nonasymptotic confidence sequences. *Ann. Statist.* 49(2) | doi:10.1214/20-AOS1991 |
| 52 | Cameron, A. C., Gelbach, J. B. & Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. *REStat* 90(3):414–427 | doi:10.1162/rest.90.3.414 |
| 53 | Clarke, S. R., Kovalchik, S. & Ingram, M. (2017). Adjusting bookmaker's odds to allow for overround. *Am. J. Sports Sci.* 5(6):45–49 (house [55]) | doi:10.11648/j.ajss.20170506.12 |
| 54 | Breeden, D. T. & Litzenberger, R. H. (1978). Prices of state-contingent claims implicit in option prices. *J. Business* 51(4):621–651 (house [80]) | doi:10.1086/296025 |
| 55 | Goodman, L. A. (1960). On the exact variance of products. *JASA* 55(292):708–713 | house [1] |

**Practitioner sources (graded):**

| # | Source | Claim | Grade |
|---|---|---|---|
| P1 | Chanchani, S. (2024). Measuring sportsbook accuracy: analysis of NFL player props using line deviations. shaanchanchani.github.io/articles/sportsbook-props | De-vigged closing Brier across four books on about 6,300 NFL prop lines: Pinnacle .2474 against DraftKings .2492; all within .002; no significance tests | D |
| P2 | XCLSV Media (2026). PrizePicks vs Underdog Pick'em vs DraftKings Pick6 | Pick'em lines lag sportsbook prop moves by about 15–30 min in peak windows; no data shown | E |
| P3 | PropsBot (2026), PropCruncher (2026) comparison pages | Softest pick'em pricing in lower-profile leagues and role players; Underdog 0.5–1.5 pts friendlier on NBA (vendor data, method unstated) | E |
| P4 | EV Bettors (X, 2024-04) | PrizePicks and Underdog set lines independently; shop both | E |
| P5 | dev.to, "PrizePicks gave me nine pass-yard lines for one quarterback. Eight are over-only" (2026) | One-sided alt lines are common; this corroborates r04's MLB 0.5-line artifact | E |
| P6 | MIT Technology Review (2017-10-19) on Kaunitz et al. | The authors' bookmaker accounts were restricted once they won | D (journalism) |

---

## 8. Files written

All under `$L/research/model_class_reads/`:

- `r00_fantasy_weights.{py,txt}`: fantasy-scoring weights recovered by least squares.
- `r01_ceiling.{py,txt,csv}`: Poisson and opportunity-known floors on the decoded mean (superseded for the headline by r07).
- `r01b_book_mean_diag.{py,txt,csv}`: decoded-mean bias.
- `r02_ladder_at_dfs_line.{py,txt}` and `r02_legs.parquet`: rungs at the pick'em line on live legs since 10-09; the decision → kickoff move.
- `r03_ladder_rule_extended.{py,txt}` and `r03_legs.parquet`: the same since 10-05; the decode gap by family.
- `r04_books_vs_underdog_backtest.{py,txt}` and `r04_props.parquet`: archive backtest, July–October; contains the MLB one-sided artifact.
- `r04b_backtest_two_sided.{py,txt}`: the backtest restricted to plausible two-sided props; Underdog lag.
- `r05_opportunity_ceiling.{py,txt,csv}`: the opportunity-surprise floor; the MLB binomial variant.
- `r06_capture.{py,txt,csv}`: mean-space capture on raw decoded means. Superseded by r07, because the decoded mean distorts the heavy-tailed families.
- `r07_ceiling_book_info.{py,txt,csv}`: the headline ceiling on book information, plus capture.
- `r08_decode_vs_rung_brier.{py,txt}`: served decode against raw rungs and the naive logit pool, on settled props.
- `logs/2026-10-29/kelly.jsonl`: an incidental log from importing `sportstradamus` helpers. It is not an output.
