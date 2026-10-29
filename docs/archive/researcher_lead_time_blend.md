# In-repo research brief: should the blend weight and temperature depend on lead time? (I6f, NFL offers scored days ahead)

**Question.** The owner, 2026-10-28: "When we get early predictions we're using the best information available and that's fine. We may just want to weigh the books less in the blend or find another way to calibrate the predictions to account for that difference." The brief answers three questions:
- Should the fused probability's blend weight w and temperature T depend on lead time, the hours from scoring to kickoff?
- Which statistic decides whether to build such a schedule?
- Do early picks win less? This is live-gap-2 §5 "Next" item 4, marked "untested".

**Date and scope.** 2026-10-29. Read-only. No code, model, config or doc was changed, and the production archive was never opened.

**Data sources.**
- **Decision-time legs (population A).** `realized.settled_offers` over the dev copy of production `history.parquet` (the 10-26 sync).
  - NFL and NHL games from 2026-10-04 to 10-25 that carry a `Scored At`.
  - One row per prop at its last scoring, counted once from 10-09 games.
- **Every stored scoring (population B).** The distinct (key, Line, Platform, Scored At) rows across every `history.parquet` snapshot under `~/backups/sportstradamus/`, plus the final one.
  - Graded with the final `Actual`. The latest scoring per prop-line is kept within each lead bin.
  - The snapshots are session backups. B therefore covers only the game days a snapshot happened to catch: 8 per early bin.
- **Lead.** Kickoff minus `Scored At`.
  - Kickoff comes from `clv.commence_times`: Underdog's `Commence`, which Sleeper rows borrow from their team-day.
  - Bins: game day (under 24 h), 1 day, 2 days, 3+ days.
- **Archive replay.** Sportsbook prop rows from the read-only copy `/tmp/arch_nfl_20261028.duckdb`.
  - NFL 2026: 12 markets, games 09-10 to 10-25, rows cut at kickoff.
  - NHL 2025-26: the backfilled 01:00 and 23:00 UTC reads.
  - Outcomes come from history `Actual` (NFL) and `data/leagues/nhl/gamelog.parquet` (NHL).
- **Code** was read at devel `7644c380`.

**Conventions.**
- Probabilities are over-side unless the text says "chosen side".
- **Gap:** mean served chosen-side probability minus hit rate.
- **ROI:** mean(hit × posted payout − 1), using `realized` conventions.
- **CIs:** 95% percentile cluster bootstrap, 2,000 draws, seed 1729. Clustered by game date unless "game clusters" (date × team pair) is stated.
- **λ\*:** the Brier-optimal pull of the served probability W away from the book M, λ\* = Σ(y − M)(W − M) / Σ(W − M)². At λ = 1 the served W is used as is; at λ = 0 the book is served.
- **Tags:** [measured] means computed here, with the script listed in Appendix A. [inferred] means reasoned from measured numbers or from code.

---

## Verdict

**KILL a lead-time schedule for w and T, in all four forms asked about:**
- a serving-time w(lead);
- a T(lead);
- "weigh the books less early";
- a training refit against an earlier book cutoff.

**Early picks do not win less.**
- NFL recommended legs graded at their first appearance (24 h or more before kickoff) overstate their hit rate by +.109, against +.121 on game day. The difference is −.012 [−.068, +.026].
- They returned −6.6% against −9.0%, a difference of +2.4 pp [−2.7, +11.6].
- On the same prop, line and model version, the served probability is no less accurate early: the Brier difference is −.0006 [−.0018, +.0009].

**The sportsbook improves toward kickoff, but only slightly.** Its Brier falls by .0006–.0013 on a .224 base, and its calibration slope stays at .95–.97 at every read.

**The owner's direction holds against game day, not against what is served.**
- The Brier-optimal pull toward the model is larger early than on game day: λ\* is .32 against .16 on same-version pairs.
- Both values sit far below the served blend, which keeps all of its distance from the book (λ = 1) at every lead.

**A per-lead weight is worth almost nothing.**
- Over one common weight it gains 3.5 × 10⁻⁵ Brier in-sample and nothing walk-forward (±6 × 10⁻⁵).
- It moves the recommended gap by −0.1 to +0.4 pp.

**Weighing the book less early makes things worse.** At λ = 1.25 for early legs, quoted-leg Brier worsens by .003–.004 in both walk-forward folds.

**Today's fitted w is already an early-book weight.**
- Training's book read sits a median 24 h before kickoff.
- Under efficient prices, the Bates–Granger argument limits the lead effect on the optimal weight to 3–7% relative.

**The level of w is the real problem.**
- w is too high at every lead. Fixing that level is the target of the levelled I6d retrain already in flight.
- That fix should improve quoted-leg Brier, not the recommended tail.

**The deciding statistic is the paired Δλ.** A schedule earns a tenth of Gate 1's δ (5 × 10⁻⁴ Brier) only if Δλ ≥ .38. Today's 95% upper bound is .18–.27.

### Table 1. The per-lead read, NFL (NHL scores 99.9% of its legs on game day)

| | Game day (< 24 h) | 1 day | 2 days | 3+ days |
|---|---|---|---|---|
| Legs, every scoring (B); game days | 9,099; 9 | 5,103; 8 | 3,263; 8 | 3,346; 8 |
| Legs at decision time (A, `realized`) | 8,985 | 244 | 213 | 155 |
| Recommended legs (B) | 1,215 | 581 | 426 | 491 |
| Recommended gap | +.121 [+.093, +.153] | +.078 [+.021, +.105] | +.105 [+.052, +.175] | +.150 [+.110, +.186] |
| Recommended ROI | −9.0% [−12.9, −4.6] | −2.7% [−7.1, +10.6] | −5.5% [−17.0, +5.1] | −12.2% [−19.5, −6.7] |
| All-legs gap | −.016 [−.032, +.013] | −.024 [−.039, +.011] | −.024 [−.034, +.006] | −.000 [−.031, +.038] |
| Served BSS against the book, quoted legs | −.043 | −.034 | −.022 | −.057 |
| Served calibration slope (SE) | .94 (.05) | 1.13 (.06) | 1.07 (.05) | .84 (.16) |
| Encompassing β_W / β_M (SE) | .11 (.04) / .83 (.05) | .32 (.14) / .67 (.20) | .43 (.08) / .69 (.10) | .05 (.28) / .92 (.26) |
| Book Brier minus its own close (archive replay, 2,776 props, 13 dates) | close = .2238 | c−24h +.0011 [+.0002, +.0020] | c−48h +.0009 [+.0002, +.0015] | c−72h +.0006 [−.0003, +.0015]; c−96h +.0013 [+.0007, +.0022] |

**Pooled early (24 h or more) against game day, NFL:**
- Recommended gap: −.012 [−.068, +.026].
- ROI: +2.4 pp [−2.7, +11.6].
- Paired served Brier, same version: −.0006 [−.0018, +.0009].
- Paired book Brier, same version: +.0040 [+.0007, +.0076].
- λ\*: .32 against .16, so Δλ = +.16 [+.06, +.27] (game clusters).
- Per-lead over common λ: 3.5 × 10⁻⁵ in-sample, ±6 × 10⁻⁵ walk-forward.

Sources: `read1.py`, `read2.py`, `read5.py`, `read8.py`, `early_vs_gameday.py`, `replay.py`, `schedule_value.py`, `sim.py` [measured].

---

## TL;DR

- **Early picks do not win less.**
  - Recommended NFL legs first scored 1–3+ days out read +.078 / +.105 / +.150 over their hit rate, against +.121 on game day.
  - Pooled, the early cohort is −.012 [−.068, +.026] on the gap and +2.4 pp [−2.7, +11.6] on ROI.
  - The weakest pocket is 3+ day Overs (hit .371, n 105, 4 game days). The 3+ day bin carries 8 game days and its CI overlaps game day.
- **The book barely improves toward kickoff and stays calibrated.**
  - 96 h out the consensus is .0013 Brier worse than its close; 72 h out it is .0006 (n.s.).
  - The calibration slope is .95–.97 at every read. There is no horizon miscalibration of the kind Page & Clemen 2013 (doi:10.1111/j.1468-0297.2012.02561.x) find months from expiry.
- **The owner's direction holds against game day, not against what is served.**
  - Early λ\* is .22–.32 and game-day λ\* .12–.16, so the early optimum keeps more of the model.
  - Both are far below 1: the served probability strays 3–8 times too far from the book at every lead.
  - Encompassing at decision time: book β .83 (.05) against served β .11 (.04) in NFL. In NHL without goals the book β is 1.21 (.15) against .10 (.13).
- **A schedule is worth about nothing. The level of w is worth something, but it belongs to I6d.**
  - Per-lead over common λ: 3.5 × 10⁻⁵ Brier in-sample, ±6 × 10⁻⁵ walk-forward.
  - A common pull toward the book: −.006 to −.010 Brier on quoted legs. Yet it leaves the recommended tail no better (gap +0.2 to +1.4 pp, cohort 32–63% smaller), as R3 and the selection-shrink KILL predict.
- **Ranked levers.**
  - (1) No schedule, plus a report-only λ\* read beside I6d's M3 on the levelled versions.
  - (2) Capture each line's first-scoring probabilities in history. About 1 dev-day. It is the only build here, and it is the owner's call.
  - (3) Align training's book read with decision time: DEFER. If ever done, anchor on kickoff and never on `date + 20 h`.
  - KILL: w(lead), T(lead), weighing the books less early, an earlier-cutoff refit, and WS-1's "time-decay discount" for NFL and NHL.
- **Instrument hazards found along the way:**
  - NHL goals' constant book leg flips the NHL encompassing read.
  - `Close Market Prob` is not the same instrument as `Market Prob`.
  - The archive holds in-play polls.
  - The NFL serving chain lost to the book on its two game days with `Standalone Prob`, and NFL attempts' T sits at the cap of 10.

---

## 1. Decision and the statistic that decides it

- **Decision.** Should w or T be a function of lead time, in serving, in training, or both?
- **Estimand.** λ\*(L), the Brier-optimal pull of served W from book M at lead L, on quoted legs.
  - λ\*(L) < 1 means the served probability strays too far from the book at that lead.
  - A schedule is worth building only if λ\* differs across leads by enough to pay for estimating it.
- **Value of a schedule.** Replacing one common λ with two equal-size lead bins gains about (Δλ / 2)² · E[(W − M)²] Brier.
  - E[(W − M)²] = .0137 on the paired panel [measured: `schedule_value.py`].
  - A gain of δ/10 = 5 × 10⁻⁴ therefore needs Δλ ≥ .38. Here δ is a tenth of Gate 1's non-inferiority margin of .005 (`docs/ship_gate.md`). Anything smaller is below what the house acts on.
- **Deciding statistic.** The paired Δλ = λ\*(≥ 24 h) − λ\*(< 24 h).
  - Population: the same prop, line and platform, quoted at both ends, under the same Model Version.
  - CI: game-cluster bootstrap, with a date-cluster read beside it.
  - Rule: KILL while the 95% upper bound is below .38.
  - Today: +.10 [+.02, +.18] on all pairs and +.16 [+.06, +.27] on same-version pairs (game clusters). The date-cluster interval for all pairs is [−.07, +.14].
- **Confirmation if it ever opens.** A walk-forward Giacomini–White test of per-lead against common λ: out-of-sample Brier, with estimation error included.
  - This is the test for a conditional combination rule (Giacomini & White 2006, doi:10.1111/j.1468-0262.2006.00718.x).
  - Diebold & Mariano (1995, doi:10.1080/07350015.1995.10524599) is its unconditional special case.
- **Why the recommended gap does not decide.**
  - It is the money metric, but it moves with selection.
  - At 400–600 recommended legs per bin its CI is ±.04–.06, too wide for a lead contrast.
  - It is the non-inferiority guard, not the decider.

---

## 2. Key findings

### Finding 1: the house grades the last scoring; the owner sees the first

- **A, the decision-time population** [measured: `read1.py`, `build.py`].
  - NFL: 8,985 of 9,597 legs were scored on game day, at a median lead of 0.2 h. The early bins hold 244, 213 and 155.
  - From 10-09 on, 376 of 9,157 NFL legs are early (4.1%). 192 of those come from the pulled fantasy points PrizePicks cell. Recommended early legs: 69 of 1,231.
  - NHL: 21,467 of 21,488 legs were scored on game day.
- **This is by construction** [inferred: `prediction/cli.py:451–486`, `realized.py:97`].
  - `_upsert_history` keeps one row per (key, Line, Platform), so each re-scoring overwrites the one before.
  - `realized.last_scored_lines` counts a prop once, at its last scoring.
  - A prop is therefore early in A only when it stopped being scored 24 h or more before kickoff: it left the board, or its cell was pulled.
  - `First Scored At` survives a re-score. The first scoring's probabilities do not.
- **B, every scoring** [measured: `read1.py`, `read3.py`].
  - NFL: 5,103 / 3,263 / 3,346 legs scored 1 / 2 / 3+ days out (8 game days each), against 9,099 on game day.
  - 3,142 NFL props carry two or more scorings, spanning a median 71 h.
  - NHL: 194 legs at 1 day and none beyond; the median span is 9 h.
- **Consequence.** "Do early picks win less" is a question about B.
  - B is opportunistic: its early scorings exist only where a session snapshotted `history.parquet` between the early and the game-day scoring.
  - 10-18 and 10-25 hold 95% of the NFL pairs (5,026 of 5,271).

### Finding 2: early picks do not win less

- **Pooled, NFL recommended (B)** [measured: `early_vs_gameday.py`].
  - Early (n 1,498): gap +.109, ROI −6.6%.
  - Game day (n 1,215): gap +.121, ROI −9.0%.
  - Differences: −.012 [−.068, +.026] on the gap and +2.4 pp [−2.7, +11.6] on ROI, over 9 dates.
  - By side. Overs: early gap +.121, ROI −10.7% (n 358), against game day +.162, −19.2% (n 301). Unders: +.106, −5.3% (n 1,140), against +.108, −5.7% (n 914).
- **Per bin** (Table 1) [measured: `read5.py`].
  - Only the 3+ day bin reads worse than game day: gap +.150 against +.121.
  - That bin is driven by Overs: hit .371 against .592 predicted, n 105 on 4 days. Its CI overlaps game day.
- **Status flips on the same prop and line** [measured: `read5.py`, `read3.py`].
  - Of 661 legs recommended early, 336 stay recommended on game day; their early side hit .485 (ROI −7.1%).
  - The 325 that drop out hit .511 (ROI −3.2%).
  - Sides flip on .089 of pairs.
- **What it means.** The overstatement exists at every lead and is about the same size at each one.
  - It is selection on model–market disagreement, which R3 put at 89% of the gap.
  - It is the optimizer's curse (Smith & Winkler 2006, doi:10.1287/mnsc.1050.0451), not a lead effect.

### Finding 3: the sportsbook gains little information between four days out and kickoff, and stays calibrated

- **NFL archive replay** [measured: `replay.py`].
  - Balanced panel: 2,776 props, 13 dates, 158 games. Rows observed at or after kickoff are dropped.
  - Consensus Brier minus its own close:
    - c−96h: +.0013 [+.0007, +.0022]
    - c−72h: +.0006 [−.0003, +.0015]
    - c−48h: +.0009 [+.0002, +.0015]
    - c−24h: +.0011 [+.0002, +.0020]
    - training's own read: +.0010 [+.0001, +.0019]
  - The close scores .2238. Between 20% and 29% of props' modal book line moved between the read and the close.
  - Over 16 dates [measured: `replay_bydate.py`]: c−72h +.0004 [−.0003, +.0012]; the training read +.0009 [.0000, +.0016]. On 10-18 and 10-25, where the history pairs sit, c−72h is −.0008 [−.0029, +.0012].
- **Calibration** [measured: `replay_cal.py`]. The logistic slope is .954–.973 (SE about .065) and CITL +.005 to +.010 at every read, on the same panel. There is no horizon effect.
- **NHL 2025-26** [measured: `replay_nhl.py`].
  - 105,739 props over 211 dates.
  - The 01:00 UTC read (about 22 h before puck drop) is .0020 [+.0014, +.0027] worse than the 23:00 UTC read.
  - Goals: +.0043 [+.0029, +.0058]. 23:00 UTC can sit after early puck drops, so this is an upper bound.
  - Shots +.0004. Assists, points and blocked shots about +.0001.
- **Literature.** The direction is textbook; the size is small.
  - Gandar, Dare, Brown & Zuber 1998 (doi:10.1111/0022-1082.155346): NBA open-to-close line moves make the line a better predictor, because informed money arrives late.
  - Moskowitz 2021 (doi:10.1111/jofi.13082): open-to-close moves carry momentum that the outcome partly reverses — a small effect.
  - Croxson & Reade 2014 (doi:10.1111/ecoj.12033): news is impounded fast and fully.
  - Dare, Dennis & Paul 2015 (doi:10.1016/j.frl.2015.02.004): opening lines miss when a player is absent.
  - Gandar et al. 1988 (doi:10.1111/j.1540-6261.1988.tb02617.x): NFL spreads pass statistical rationality tests.
  - Brown & Sauer 1993 (doi:10.1111/j.1540-6261.1993.tb04751.x): prices still carry a non-fundamental component, so "efficient" below means approximately so.
- **Where the gain lives** [inferred]. The NFL information gain sits in the weekly injury cycle: Wednesday–Friday reports and the 90-minute inactives. In NHL it sits in goals: lineups and goalies.

### Finding 4: on the same prop and line, the served probability holds and the book improves, mostly through line conversion

- **All pairs** [measured: `read3.py`, date clusters].
  - NFL pairs (same prop, line and platform; early 24 h or more, game day under 24 h): 5,271 pairs on 5 game days, at median leads of 71 h and 0.4 h.
  - Served Brier, early minus game day: +.0007 [+.0002, +.0020].
  - Book, quoted at both ends (3,609 pairs): +.0033 [+.0014, +.0041].
  - With 5 date clusters these intervals under-cover.
- **Same Model Version at both ends** [measured: `read8.py`, game clusters]. 3,160 pairs, 60% of the total.
  - Served: −.0006 [−.0018, +.0009].
  - Book: +.0040 [+.0007, +.0076] (2,108 quoted pairs).
- **The book's early loss sits where its quote is converted from another line** [measured: `read6.py`].
  - Quote line ≠ offer line at both ends: +.0044 [+.0018, +.0073] (n 2,424; |offer − quote| about 6).
  - Quote line = offer line at both ends: +.0003 [−.0016, +.0026] (n 1,095).
  - [inferred] A converted early price compounds the book's staleness with the decode distance. A native early price is as good as the game-day one on this panel.
- **10-25, the one NFL game day with `Standalone Prob` at both ends** (12 games) [measured: `read7.py`].
  - The model alone was *better* early: −.0060 [−.0121, −.0006].
  - The served chain: −.0009 [−.0024, +.0007].
  - The 10-23 builder change confounds this: 1,797 of 2,011 early scorings used the old builder.
- **The panel is the low-news subset** [measured: `linemove.py`].
  - 66% of single-line NFL props move their DFS line between the early and the game-day scoring: Underdog 76%, Sleeper 60%. Receiving yards move 88% of the time, by 8.4 yards on average; touchdowns never move.
  - The move goes toward the outcome 58% of the time (n 1,462).
  - Pairs need an unchanged line, so they cover the props with little news. They likely understate the early book's information loss on the full early population [inferred].

### Finding 5: the optimal pull toward the model is larger early, but far below the served blend at every lead

**Table 2. Blend-weight reads** [measured: `read4.py`, `read8.py`, `read2.py`, `nhl_goals_check.py`]

| Read | n | λ\* of served W against the book | Model-alone w\* (`Standalone Prob` against the book) | Served w (mean) |
|---|---|---|---|---|
| NFL decision time, game day, quoted | 7,318; 43 games | .083 [−.057, +.237] | .331 [.086, .608] (1,941; 2 days) | .59–.61 |
| NFL paired, early | 3,953; 27 games | .222 [.018, .431] | .593 [.249, .952] (1,737; 10-25 only) | .615 |
| NFL paired, game day | same pairs | .120 [−.044, .298] | .385 [.064, .728] (same) | .595 |
| NFL paired, same version: early / game day | 2,108 | .323 [.108, .575] / .162 [−.051, .395] | | |
| NHL decision time, game day, quoted, **goals excluded** | 13,279 | .404 [.141, .670] | .348 [−.218, .873] (3,616) | .693 |
| NHL with goals (constant book leg, Finding 10) | 15,518 | .700 [.570, .826] | .584 [.234, .940] | .703 |

- **Encompassing at decision time** [measured: `read2.py`, `nhl_goals_check.py`; date-clustered logit].
  - NFL, game day: β_W .11 (.04), β_M .83 (.05).
  - NHL, goals excluded: β_W .10 (.13), β_M 1.21 (.15).
  - In both leagues the book encompasses the served probability on quoted legs at decision time (Harvey, Leybourne & Newbold 1998, doi:10.1080/07350015.1998.10524759; Clements & Harvey 2010, doi:10.1002/jae.1097).
  - The early NFL bins read β_W .32 (.14) / .43 (.08) / .05 (.28).
- **Bates–Granger under efficient prices bounds the lead effect** [inferred, derivation].
  - Let M_c be the close and M_L the book at lead L. If prices are a martingale, M_c = M_L + ε, where ε is uncorrelated with everything known at L, including the model's probability S.
  - Then Cov(y − M_L, S − M_L) = Cov(y − M_c, S − M_c), and Var(S − M_L) = Var(S − M_c) − Var(ε).
  - So w\*(L) = w\*(c) · D / (D − Var ε), with D = E[(S − M_c)²] (Bates & Granger 1969, doi:10.2307/3008764).
  - With D = .0203 (the model alone against the book) and Var ε = .0006–.0013 (Finding 3's Brier gaps), the model's optimal weight is 3–7% higher, relative, at 1–4 days than at the close. Example: .33 becomes .34–.35 [measured: `schedule_value.py`].
  - The sign matches the owner. The size is negligible.
  - The paired λ\* step (+.10 to +.16) is larger than this bound. It sits on 2 Sundays (date-cluster CI [−.07, +.14]) and is not resolved.
- **Model version moves λ more than lead does** [measured: `sim.py`].
  - Fitted on games up to 10-19 (versions before lever T), λ is .011. Fitted on games from 10-22, it is .290.
  - That version step of +.28 compares with a lead step of +.10 to +.16.

### Finding 6: the schedule is worth 3.5 × 10⁻⁵ Brier in-sample and nothing out of sample

- **In-sample** [measured: `schedule_value.py`].
  - On 3,953 quoted pairs, λ\* is .222 early, .120 on game day and .172 common.
  - E[(W − M)²] is .0138 early and .0135 on game day.
  - Per-lead over common λ gains 3.5 × 10⁻⁵ Brier.
- **Walk-forward** [measured: `sim.py`]. Fit on games up to 10-19 and test from 10-22, then the reverse.
  - Per-lead against common λ: +.00006 and −.00005 quoted-leg Brier.
  - Recommended gap: −0.1 and +0.4 pp. ROI: +0.2 and −1.0 pp.
- **Minimum detectable effect** [inferred].
  - The per-leg Brier difference between the two rules has an SD of about .006, from (Δλ/2) × RMS(W − M) = .05 × .117.
  - Detecting 3.5 × 10⁻⁵ needs roughly 1–2 × 10⁵ independent paired legs.
  - At about 2,000 quoted pairs per Sunday slate, that is 55–115 Sundays, or three to six regular seasons.
- **Literature.** This is the expected outcome.
  - Deutsch, Granger & Teräsvirta 1994 (doi:10.1016/0169-2070(94)90049-3) is the template for weights that switch on an observable.
  - Stock & Watson 2004 (doi:10.1002/for.928) found the most successful combinations are the least sensitive to the recent performance of their inputs.
  - Timmermann 2006 (doi:10.1016/S1574-0706(05)01004-9) surveys why time-varying weights rarely beat fixed ones out of sample.
  - Smith & Wallis 2009 (doi:10.1111/j.1468-0084.2008.00541.x) and Claeskens, Magnus, Vasnev & Wang 2016 (doi:10.1016/j.ijforecast.2015.12.005) trace this to weight estimation error, which here is as large as the gain.
  - If per-group weights are ever fitted, shrink them toward the common weight (Diebold & Shin 2019, doi:10.1016/j.ijforecast.2018.09.006).

### Finding 7: weighing the book less early hurts, and training's w is already an early-book weight

- **"Weigh the book less early"** [measured: `sim.py`]. Tested as P' = M + 1.25 (W − M) on quoted legs scored 24 h or more out, with game day served as is.
  - Quoted-leg Brier: .24188 against .23820 in fold 1 and .23786 against .23484 in fold 2.
  - The early segment alone: .24392 against .23818 and .24194 against .23552.
  - Recommended legs grow 33% and 17% at the same overstatement: gap −0.2 / +0.8 pp, ROI +1.6 / −0.5 pp.
  - It moves away from the measured optimum: the served blend is already above λ\* at every lead.
- **Training's book leg is an early book** [measured: `replay.py`; code `stats/base.py:2471–2472`, `helpers/archive.py:98`].
  - It is the newest row at or before 12:00 UTC on the game date: `date + 20 h − TRAINING_LOOKBACK_HOURS (8)`.
  - In the replay that read sits a median 24.2 h before kickoff and scores .0010 worse than the close. It carries the information of a c−24h read.
  - w and T are fit on validation rows against that book (`training/calibration.py:678`, `training/pipeline.py:3232`).
  - So today's w already is an early-book weight. By Bates–Granger, the weight that should be lower is game day's, by the same 3–7%.
- **Serving reads the newest quotes** (`prediction/book_quotes.py:143`, no `at=`) [measured: `read5.py`].
  - Median quote age at scoring: 0.2 h on game day, 9.3 h at 1 day, 24.3 h at 2 days, 0.3 h at 3+ days.
  - Books per quote: median 4 in every bin.

### Finding 8: temperature shows no lead dependence

- **Calibration slope by lead** [measured: `read2.py`].
  - NFL (B, all legs), game day through 3+ days: .94 (.05), 1.13 (.06), 1.07 (.05), .84 (.16). There is no monotone pattern (Cox 1958, doi:10.1093/biomet/45.3-4.562).
  - NHL game day: 1.05 (.03).
- **T is already placed correctly.**
  - It applies `expit(logit(p) / T)` after the pool (`helpers/distributions.py:917`), bounded to [1, 10] (`training/pipeline.py:3229`).
  - Recalibrating the combination is the right place for it (Ranjan & Gneiting 2010, doi:10.1111/j.1467-9868.2009.00726.x).
- **The weather contrast.**
  - Postprocessing fits per-lead calibrators as standard (Vannitsem et al. 2021, doi:10.1175/BAMS-D-19-0308.1, arXiv:2004.06582).
  - Schulz & Lerch 2022 (doi:10.1175/MWR-D-21-0150.1, arXiv:2106.09512) "fit a separate model for each lead time", accepting forecasts that are non-seamless across leads, because numerical weather prediction skill decays steeply with lead.
  - Here neither input's skill moves measurably over 1–4 days. A per-lead calibrator would have nothing to learn [inferred].

### Finding 9: "robust to moving lines" is about the DFS line, not the blend

- **Line moves against the early pick** [measured: `linemove.py`, `linemove2.py`].
  - 66% of single-line NFL props move their DFS line between the first early scoring and the game-day scoring.
  - The early chosen side catches the move (the line shifts to that side's benefit) 61.6% [58.3, 64.9] of the time (1,462 moved props, 42 games).
  - Recommended early legs catch it only 52.2% [43.7, 60.3] (n 251).
  - Hit rates of early recommended legs: .527 when the line moved for them, .517 when against, .561 when it did not move. All claimed about .61–.63.
- **Interpretation** [inferred].
  - The model leans on the books and picks the side they favour; the platforms then converge toward the books.
  - The recommended set, which holds the largest disagreements with the payout, does not inherit that edge.
  - A stale-line rule would be a selection lever, outside this brief (§6 item 7).

### Finding 10: instrument hazards met on the way (each changes a read; none changes the verdict)

1. **NHL goals serves a constant book leg** [measured: `nhl_goals_check.py`; this corroborates the live-gap-2 REPORT, 10-28 23:40 CDT].
   - 64.6% of NHL goals quoted decision-time legs share one `Market Prob`.
   - With goals included, NHL looks model-led: λ\* .70, β_W .83.
   - Without goals it matches NFL: λ\* .40 [.14, .67], β_W .10 (.13), β_M 1.21 (.15).
2. **`Close Market Prob` and `Market Prob` are different instruments** [measured: `read1.py`, item 3].
   - At a lead of about 0 they sit .069 RMS apart.
   - The "close" scores worse than the decision-time book on NFL game day: .2292 against .2256. That gap is an instrument difference, not information.
   - Never pair the two in a lead or CLV contrast.
3. **The archive holds in-play polls** [measured: `inplay_check.py`, `sunday_polls.py`; code `moneylines.py:669–700`].
   - The props poll does not skip started events. 3.6% of NFL rows with a known kickoff were observed 1–3 h after it.
   - Sunday dates before 10-05 carry 19:00 and 20:00 UTC rows.
   - `clv.commence_times` gives rows without a `Commence` (scored before 10-05) a 20:00 UTC kickoff (`clv.py:43`), which falls after the 17:00 UTC starts. Those NFL Sunday closes may be in-play reads [inferred].
4. **The NFL serving chain lost to the book on its two `Standalone Prob` game days** [measured: `chain.py`; 13 games].
   - Served minus book: +.0092 [+.0019, +.0159]. Served minus the model alone: +.0023 [−.0050, +.0099].
   - Median T is 4.20, and NFL attempts sits at T = 10.0, the cap.
   - NHL: served minus the model alone is −.0015 [−.0029, −.0001]. Served minus book is −.0025 [−.0058, +.0007], or −.0002 [−.0026, +.0023] without goals.

---

## 3. Recommendation: ranked levers

**Table 3.** Projected effects are on the recommended-cohort gap (pp). Costs are in dev-days.

| Rank | Lever | Verdict | Δ recommended gap | Cost | Type |
|---|---|---|---|---|---|
| 1 | No lead schedule; read the post-I6d level λ\* beside I6d M3 | **GO (a read)** | 0 from the read | 0.25 | research read |
| 2 | Capture each line's first-scoring probabilities in history | **Owner's call** (build only to monitor early picks) | 0 (measurement) | ~1 | engineering |
| 3 | Align training's book read with the decision-time book | **DEFER** (fold into a re-inject done for another reason) | < 0.2 | 2–3 plus a retrain per league | engineering |
| 4 | Serving w(lead) step | **KILL** | −0.1 to +0.4 | ~3 plus the research gate | engineering build of a refuted idea |
| 5 | T(lead) | **KILL** | ~0 [inferred] | 2–3 | same |
| 6 | Weigh the books less early | **KILL** | −0.2 to +0.8; Brier worse | 1–2 | refuted |
| 7 | Earlier-cutoff training refit | **KILL** | ~0 [inferred] | 3–5 plus retrains | research, unpromising |
| (WS-1) | "Time-decay discount" on early legs (§6.10 step 3) | **KILL for NFL and NHL** | n/a | n/a | refuted |

### Lever 1. No lead schedule; read the level instead (rank 1)

- **Exact change.** None: one w and one T per cell, as fit. Add λ\* as a report-only line beside I6d's M3, the fixed-effect logistic encompassing test on model-only probabilities.
- **Pre-registered read.**
  - **Statistic:** λ\* of served `Win Prob` against `Market Prob`, over-side.
  - **Population:** `realized.settled_offers` with `Market Projection` present (not book-fallback), lead under 24 h, Model Versions trained on the levelled matrices.
    - NFL as is.
    - NHL separately, with goals excluded until its serve-time decode is fixed.
  - **CI:** game-cluster bootstrap, 4,000 draws, seed 1729, with a date-cluster read reported beside it.
  - **Rule:**
    - CI covers 1: the level is right-sized; close the item.
    - Upper bound below 1: w still sits above the live optimum. Route to a blend-level brief on why validation-fit w over-weights the model against the live book. Do not route to a lead schedule.
    - Lower bound above 1: the model is under-used; same route.
  - **Minimum n:** NFL, 4 post-send Sundays and at least 5,000 quoted legs. NHL, 10 game days and at least 5,000 legs.
  - **Read date:** the Tuesday after the fourth post-send NFL Sunday (11-24 if the NFL send lands by 10-31).
- **Projected effect.** The read itself changes nothing.
  - What the retrain will likely do, using the common-λ walk-forward as a proxy [inferred, `sim.py`]: quoted-leg Brier −.006 to −.010, recommended legs −32% to −63%, recommended gap +0.2 to +1.4 pp (no better). Plan for a better-calibrated bulk and a smaller, not more honest, tail.
- **Must not touch:**
  - I6d's frozen decode table and M1 bars.
  - The Gate-2 thresholds (owner-only).
  - No flat shrink of served probabilities (killed).

### Lever 2. Capture first-scoring probabilities in history (rank 2; the owner's call)

- **Exact change.**
  - In `prediction/cli.py::_upsert_history`, carry `First Win Prob`, `First Market Prob`, `First Standalone Prob` and `First Quote Line` forward exactly as `First Scored At` is carried: `groupby(key).transform("first")` before the dedup.
  - Add the four names to `history_schema.DECISION_COLS`.
- **Why.**
  - Today a re-score overwrites the early probabilities. The early-pick question can only be answered from ad hoc snapshots: B's 3,953 quoted pairs came from two Sundays.
  - With the columns, every line first scored 24 h or more out carries its own paired early read. That keeps Table 1 and the Δλ read current every Sunday.
- **Pre-registered test (integrity).**
  - Run two consecutive fake-mode prophecize runs.
  - A re-scored key must hold run 1's `Win Prob` in `First Win Prob` and run 2's in `Win Prob`.
  - A new key must hold equal values in both.
  - The closed-row-wins dedup must behave as today.
  - Live use: Lever 4's re-open read.
- **Projected effect:** 0 (measurement only).
- **Cost:** ~1 dev-day: two files plus a test. The columns are additive, so dashboard readers are unaffected.
- **Must not touch:**
  - `Win Prob`, the side rule, `realized.recommended`.
  - The closed-row-wins dedup.
  - The inference path.

### Lever 3. Align training's book read with the decision-time book (rank 3; DEFER)

This is the item already open in honest-receipts §6 item 3: "the book-leg cutoff aligned to decision time where ladder polls exist (`TRAINING_LOOKBACK_HOURS`)".

- **Exact change.**
  - Replace `target_at = date + 20 h − 8 h` (`stats/base.py:2471–2472`) with each game's kickoff minus 30 minutes from the league schedule, taking the newest row before it.
  - Never anchor on `date + 20 h` itself: Sunday 19:00 and 20:00 UTC polls are in-play for 17:00 UTC kickoffs (Finding 10).
  - Backfilled seasons barely move. Their only pre-kickoff game-day snapshot for afternoon games is 01:00 UTC.
- **Projected effect** [inferred: Bates–Granger with Var ε .0006–.0013].
  - The fitted w on NFL cells falls 3–7% relative.
  - Quoted-leg Brier moves on the order of 10⁻⁵. The recommended gap moves under 0.2 pp.
- **Pre-registered test.** None for money: the effect is below any detectable size. If it is built:
  - Every chosen `observed_at` must precede its kickoff (zero in-play rows).
  - Report the share of matrix `Odds` rows changed per cell.
  - Report w per cell through I6d's M2. Read it at that retrain.
- **Cost:** 2–3 dev-days, plus one re-inject and one retrain per league.
- **Must not touch:**
  - Serving's book read.
  - I6d's decode table and bars.
  - The 15:00 UTC game-line cutoff (`GAME_LINE_TRAINING_CUTOFF`), which is a separate I6f item.

### Lever 4. Serving w(lead) step (KILL)

- **What it would be.** Per-cell w for leads of 24 h or more and under 24 h, or a multiplier on w for the early bin. It would apply where `_blend_with_book` reads the pickle's weight (`prediction/model_prob.py:1070`, `:1198`), keyed on hours to kickoff.
- **Why KILL.**
  - (i) The Δλ upper bound of .18–.27 is below the .38 a schedule needs.
  - (ii) Walk-forward, per-lead against common λ moves Brier by ±6 × 10⁻⁵ and the gap by −0.1 / +0.4 pp.
  - (iii) It could only be fit on live legs. Backfilled seasons hold game-day snapshots only, so each training row has one book read. That makes w(lead) an outcome-fitted serving layer, the class of the killed shrink.
- **Pre-registered re-open.**
  - Data: post-I6d versions with Lever 2's columns.
  - Condition: the same-version paired Δλ has a lower bound above .38 under date clusters, with 8 or more NFL Sundays and 8,000 or more quoted pairs.
  - Before any serving change, a walk-forward Giacomini–White test of per-lead against common λ must then pass. Per-bin weights are shrunk toward the common one (Diebold & Shin 2019).
    - Brier gain of at least 5 × 10⁻⁴.
    - Date-cluster lower bound above 0.
    - Recommended-gap upper bound at most +1 pp.
  - Earliest read: after the regular season (January 2027).
- **Projected effect:** −0.1 to +0.4 pp.
- **Cost:** ~3 dev-days, plus the research gate (a live serving-distribution change, §8.2).
- **Must not touch:**
  - The side rule, the 0.05 edge floor, `realized.recommended`.
  - No flat shrink.

### Lever 5. T(lead) (KILL)

- **What it would be.** T for early and for game-day scorings separately, applied in `apply_temperature` (`prediction/model_prob.py:1230`).
- **Why KILL.**
  - No lead pattern in the calibration slope (Finding 8).
  - T already sits after the pool.
  - Weather's per-lead calibrators answer a skill decay this data does not show.
- **Pre-registered re-open.** The early-minus-game-day calibration slope on captured post-I6d pairs has a date-cluster CI excluding 0, over 8 or more Sundays. Read in January 2027.
- **Projected effect:** about 0 pp [inferred].
- **Cost:** 2–3 dev-days.
- **Must not touch:** `_TEMPERATURE_BOUNDS`; the Brier fit of T.

### Lever 6. Weigh the books less early (KILL)

- **What it would be.** The tested form is P' = M + 1.25 (W − M) for leads of 24 h or more, the probability-space equivalent of a larger w.
- **Why KILL.**
  - It worsens quoted-leg Brier by .003–.004 overall and .006 on early legs in both folds.
  - It adds 17–33% more early recommendations at the same overstatement.
  - It moves away from the measured optimum, since λ\* is below 1 at every lead.
- **Pre-registered re-open.** Lever 1's read, run on early legs, shows λ\*(early) with a lower bound above 1, meaning the model is under-used early. Read at Lever 1's date with Lever 2's columns.
- **Projected effect:** −0.2 to +0.8 pp.
- **Cost:** 1–2 dev-days.
- **Must not touch:** as Lever 4.

### Lever 7. Earlier-cutoff training refit (KILL)

- **What it would be.** A second (w, T) per cell, fit against a book read at kickoff − 72 h and served for scorings 48 h or more out.
- **Why KILL.**
  - Training's read is already about c−24h (Finding 7).
  - Moving from c−24h to c−72h changes the book by Var ε of about .0003, about 1% in w.
  - Backfilled seasons have no c−72h read at all.
- **Pre-registered re-open.** On the season-end replay, the c−72h read is at least .002 Brier worse than training's read (four times today), with a date-cluster lower bound above 0. Read in January 2027.
- **Projected effect:** about 0 pp [inferred].
- **Cost:** 3–5 dev-days, plus retrains.
- **Must not touch:** I6d's levelled read.

### WS-1's "time-decay discount" (§6.10 step 3)

- **For NFL and NHL: do not discount early legs.** Pooled early ROI is +2.4 pp [−2.7, +11.6] against game day, and the gap difference is −.012 [−.068, +.026].
- Other leagues were not read.

---

## 4. How a lead-conditional w or T differs from the killed shrink and the trust layer

| | Selection shrink (KILL 2026-10-03) | Trust layer R1 (KILL 2026-10-04) | w(lead) or T(lead) (this brief) |
|---|---|---|---|
| Where it acts | After the side choice: `Win Prob' = Market + λ(Win − Market)` on recommended quoted legs | After the side choice: a learned `Trust Prob` from market observables and served p | Before the side choice: the `fused_loc` weight or `apply_temperature`, keyed on hours to kickoff |
| Fit on | Recommended legs | Posted legs | All quoted legs, split by lead |
| Estimand | Calibration of the selected tail | An honest posted-leg probability | The pool weight at each lead |
| Effect on the recommended set | Only removes legs | Only removes legs | Moves sides and edges on every quoted leg |
| What killed it | λ_sel of −.10 out of sample | Market observables carry all of served p | Δλ too small to pay (3.5 × 10⁻⁵); null walk-forward; fittable only on live legs |

**The shared lesson.**
- Every variant fit on outcomes pulls the served probability toward the book.
- That improves quoted-leg Brier and leaves the recommended tail as overstated as before. In the sim, a common λ shrinks the cohort 32–63% and moves its gap by +0.2 to +1.4 pp.
- The tail is selection on disagreement (R3: 89%). No split by lead changes that.

**What is genuinely different about w(lead).**
- It acts before the side choice and on every leg, so it is a conditional forecast combination, not a selection correction (Deutsch, Granger & Teräsvirta 1994).
- That is why it was worth testing.
- Its ceiling comes from how the book's information arrives, not from selection, and that ceiling is about 10⁻⁵ Brier.

---

## 5. Reality checks

**The 3–4 week window (NFL about 370, NHL about 100 recommended legs a week).**
- **Decision time (`realized`).**
  - Early legs are 5.6% of NFL recommended legs (69 of 1,231 since 10-09) and 0.1% of NHL ones (1 of 783).
  - That gives about 60–85 NFL and fewer than 1 NHL early recommended legs in 3–4 weeks.
  - A gap CI at n of about 70 over 4 Sundays is about ±.12. Nothing is readable.
- **With Lever 2's capture.**
  - The every-scoring panel adds about 150–200 early recommended NFL legs and about 2,000 quoted pairs per Sunday slate. For scale, 10-18 and 10-25 gave 2,601 and 2,425 pairs, about 75% of them quoted.
  - In 3–4 weeks: the early-against-game-day gap difference to about ±.06, and Δλ to about ±.06 with game clusters.
  - That is enough to keep confirming Δλ < .38. It is never enough to test a schedule.
- **The schedule's own effect** (3.5 × 10⁻⁵ Brier) needs 1–2 × 10⁵ paired legs: 55–115 NFL Sundays. It cannot be validated in 2026.
- **What the window can read.** The post-I6d level λ\* on about 1,800 quoted NFL game-day legs a week (Lever 1), to about ±.15 after 4 weeks. That separates "level fixed" (about 1) from "still too high" (about .1).

**Few clusters.**
- 95% of NFL pairs sit on two Sundays, 10-18 and 10-25. The λ\* panel has 27 games on 5 dates.
- Game-level clusters give the narrow intervals. Date clusters (5) put Δλ at [−.07, +.14].
- The few-cluster bootstrap under-covers, worse with clusters of very unequal size (Cameron, Gelbach & Miller 2008, doi:10.1162/rest.90.3.414; MacKinnon & Webb 2017, doi:10.1002/jae.2508).
- The KILL does not rest on the narrow intervals: even the point estimates leave the schedule at 3.5 × 10⁻⁵.

**What could make the KILL wrong.**
- More late news raises Var ε: injury-heavy slates, late-season weather.
- Even at Var ε = .004, three times the largest measured, w\*(L) / w\*(close) is 1.25 and the schedule is worth about 1 × 10⁻⁴. That is still a fifth of δ/10.
- NFL weeks 12–18 are where to look, through Lever 2's columns [inferred].

**Pre-I6d models.**
- Every read here comes from versions trained on the un-levelled book leg.
- The level (λ\*) will move with the retrain. The lead contrast is a property of how the book's information arrives and should not [inferred].

**Pair selection.**
- Pairs need an unchanged DFS line. 66% of single-line props move theirs, toward the outcome 58% of the time.
- The paired panel is therefore the low-news subset and understates the early book's loss [inferred].
- The per-bin calibration in Table 1 covers the full population and agrees.

**Version and builder changes.**
- NFL versions changed on 10-18: 40% of pairs straddle it, and the same-version subset agrees in sign.
- Early scorings before 10-23 17:50 UTC used the old feature builder.
- `Standalone Prob` exists only from 10-22, which covers two NFL game days.

**Leagues not read.**
- NBA, WNBA and MLB were not read for lead.
- NHL has no early scorings, so its whole contribution is the level read and the instrument hazards.

**Model-alone weight against the pickle's w.**
- The pickle's w is a distribution-level log-pool weight, followed by T and post-hoc stages.
- The model-alone w\* is a linear-pool weight on probabilities: .33 for NFL on game day against a served .61.
- Compare their direction, not their values.

---

## 6. Open questions and caveats (candidates for §8.2)

1. **Post-I6d level (Lever 1's read).** Does the levelled retrain bring λ\* to 1, or does validation-fit w still over-weight the model against the live book? The version step from 10-19 to 10-22 moved λ from .01 to .29, more than any lead step did.
2. **Capture (Lever 2).** This is the owner's call. Without it, "do early picks win less" stays answerable only from ad hoc snapshots.
3. **In-play archive rows.**
   - The props poll stores started events (`moneylines.py:669–700`): 3.6% of NFL rows with a known kickoff were observed 1–3 h after it.
   - Any archive read anchored at or after kickoff needs a kickoff cut.
   - `clv.commence_times` gives rows scored before 10-05 a 20:00 UTC kickoff, after Sunday's 17:00 UTC starts. Audit those NFL Sunday closes before any CLV verdict on pre-10-05 NFL rows.
4. **`Close Market Prob` and `Market Prob` are different instruments.**
   - They sit .069 RMS apart at a lead of about 0.
   - The close scores worse than the decision-time book on NFL game day: .2292 against .2256.
   - Never pair them in a lead or CLV contrast.
5. **The NFL serving chain against the book.**
   - Served minus book was +.0092 [+.0019, +.0159] on 2 game days. Served minus the model alone was +.0023 (n.s.).
   - NFL attempts sits at T = 10.0, the cap.
   - Re-read after the I6d retrain. If the gap holds, the suspect is the T fit, which is Brier against a mis-levelled book.
6. **NHL goals' constant book leg** (live-gap-2 REPORT, 10-28 23:40 CDT) flips every NHL blend read.
   - λ\* falls from .70 to .40 without goals; β_W falls from .83 to .10.
   - Drop goals from NHL blend reads until the serve-time decode is fixed.
7. **Stale-line selection.**
   - The early chosen side catches the DFS line's later move 61.6% [58.3, 64.9] of the time; recommended early legs catch it only 52.2% [43.7, 60.3].
   - A rule preferring early legs where the books have already moved away from the DFS line would be a selection lever, not a blend lever.
   - It needs its own read before any brief: DFS-line CLV of recommended legs, by lead.
8. **10-25's model-alone oddity.**
   - `Standalone Prob` scored better early than on game day: −.0060 [−.0121, −.0006], on one game day and confounded by the builder change.
   - If it recurs on captured post-I6d pairs, the suspect is the game-day feature build, not the blend.

---

## 7. Bibliography

| # | Source | Identifier | Used for |
|---|---|---|---|
| B1 | Bates, J. M. & Granger, C. W. J. (1969). The combination of forecasts. *Operational Research Quarterly* 20(4):451–468 | doi:10.2307/3008764 (also doi:10.1057/jors.1969.103) | Weights follow error variances; the bound on w(lead) |
| B2 | Deutsch, M., Granger, C. W. J. & Teräsvirta, T. (1994). The combination of forecasts using changing weights. *Int. J. Forecasting* 10(1):47–57 | doi:10.1016/0169-2070(94)90049-3 | Weights that switch on an observable |
| B3 | Timmermann, A. (2006). Forecast combinations. *Handbook of Economic Forecasting* vol. 1, ch. 4, 135–196 | doi:10.1016/S1574-0706(05)01004-9 | Time-varying weights and estimation error |
| B4 | Stock, J. H. & Watson, M. W. (2004). Combination forecasts of output growth in a seven-country data set. *J. Forecasting* 23(6):405–430 | doi:10.1002/for.928 | The best combinations are the least sensitive to recent performance |
| B5 | Smith, J. & Wallis, K. F. (2009). A simple explanation of the forecast combination puzzle. *Oxford Bull. Econ. Stat.* 71(3):331–355 | doi:10.1111/j.1468-0084.2008.00541.x | Weight estimation error eats small gains |
| B6 | Claeskens, G., Magnus, J. R., Vasnev, A. L. & Wang, W. (2016). The forecast combination puzzle: a simple theoretical explanation. *Int. J. Forecasting* 32(3):754–762 | doi:10.1016/j.ijforecast.2015.12.005 | Same |
| B7 | Diebold, F. X. & Shin, M. (2019). Machine learning for regularized survey forecast combination: partially-egalitarian LASSO and its derivatives. *Int. J. Forecasting* 35(4):1679–1691 | doi:10.1016/j.ijforecast.2018.09.006 | Shrink per-group weights toward the common one |
| B8 | Giacomini, R. & White, H. (2006). Tests of conditional predictive ability. *Econometrica* 74(6):1545–1578 | doi:10.1111/j.1468-0262.2006.00718.x | The confirmatory test for a lead-conditioned rule |
| B9 | Diebold, F. X. & Mariano, R. S. (1995). Comparing predictive accuracy. *JBES* 13(3):253–263 | doi:10.1080/07350015.1995.10524599 | The unconditional paired comparison |
| B10 | Harvey, D. I., Leybourne, S. J. & Newbold, P. (1998). Tests for forecast encompassing. *JBES* 16(2):254–259 | doi:10.1080/07350015.1998.10524759 | Encompassing |
| B11 | Clements, M. P. & Harvey, D. I. (2010). Forecast encompassing tests and probability forecasts. *J. Appl. Econometrics* 25(6):1028–1062 | doi:10.1002/jae.1097 | Logit encompassing for probabilities |
| B12 | Ranjan, R. & Gneiting, T. (2010). Combining probability forecasts. *JRSS-B* 72(1):71–91 | doi:10.1111/j.1467-9868.2009.00726.x | Recalibrate the combination (T after the pool) |
| B13 | Cox, D. R. (1958). Two further applications of a model for binary regression. *Biometrika* 45(3–4):562–565 | doi:10.1093/biomet/45.3-4.562 | Calibration slope |
| B14 | Smith, J. E. & Winkler, R. L. (2006). The optimizer's curse: skepticism and postdecision surprise in decision analysis. *Management Science* 52(3):311–322 | doi:10.1287/mnsc.1050.0451 | Overstatement of the selected tail at every lead |
| B15 | Gandar, J., Zuber, R., O'Brien, T. & Russo, B. (1988). Testing rationality in the point spread betting market. *J. Finance* 43(4):995–1008 | doi:10.1111/j.1540-6261.1988.tb02617.x (JSTOR doi:10.2307/2328148) | NFL spreads pass statistical rationality tests |
| B16 | Brown, W. O. & Sauer, R. D. (1993). Fundamentals or noise? Evidence from the professional basketball betting market. *J. Finance* 48(4):1193–1209 | doi:10.1111/j.1540-6261.1993.tb04751.x | Prices carry a non-fundamental part; martingale is approximate |
| B17 | Gandar, J. M., Dare, W. H., Brown, C. R. & Zuber, R. A. (1998). Informed traders and price variations in the betting market for professional basketball games. *J. Finance* 53(1):385–401 | doi:10.1111/0022-1082.155346 | Open-to-close moves make lines better predictors |
| B18 | Dare, W. H., Dennis, S. A. & Paul, R. J. (2015). Player absence and betting lines in the NBA. *Finance Research Letters* 13:130–136 | doi:10.1016/j.frl.2015.02.004 | Absence news is what early lines miss |
| B19 | Moskowitz, T. J. (2021). Asset pricing and sports betting. *J. Finance* 76(6):3153–3209 | doi:10.1111/jofi.13082 | Open-to-close momentum, partly reversed; small |
| B20 | Croxson, K. & Reade, J. J. (2014). Information and efficiency: goal arrival in soccer betting. *Economic Journal* 124(575):62–91 | doi:10.1111/ecoj.12033 | News impounded fast and fully |
| B21 | Page, L. & Clemen, R. T. (2013). Do prediction markets produce well-calibrated probability forecasts? *Economic Journal* 123(568):491–513 | doi:10.1111/j.1468-0297.2012.02561.x | Calibration decays with horizon (months); absent at 1–4 days here |
| B22 | Levitt, S. D. (2004). Why are gambling markets organised so differently from financial markets? *Economic Journal* 114(495):223–246 | doi:10.1111/j.1468-0297.2004.00207.x | Bookmakers set informed prices |
| B23 | Vannitsem, S. et al. (2021). Statistical postprocessing for weather forecasts: review, challenges, and avenues in a big data world. *BAMS* 102(3):E681–E699 | doi:10.1175/BAMS-D-19-0308.1; arXiv:2004.06582 | Lead-dependent postprocessing where skill decays with lead |
| B24 | Schulz, B. & Lerch, S. (2022). Machine learning methods for postprocessing ensemble forecasts of wind gusts: a systematic comparison. *Monthly Weather Review* 150(1):235–257 | doi:10.1175/MWR-D-21-0150.1; arXiv:2106.09512 | "A separate model for each lead time"; non-seamless across leads |
| B25 | Cameron, A. C., Gelbach, J. B. & Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. *REStat* 90(3):414–427 | doi:10.1162/rest.90.3.414 | Few-cluster under-coverage |
| B26 | MacKinnon, J. G. & Webb, M. D. (2017). Wild bootstrap inference for wildly different cluster sizes. *J. Appl. Econometrics* 32(2):233–254 | doi:10.1002/jae.2508 | Unequal clusters (two Sundays hold 95% of the pairs) |

Metadata for B1–B7 and B15–B24 was checked against the Crossref API (`https://api.crossref.org/works/{doi}`) on 2026-10-28/29. B5, B6 and B8–B14, B22, B25 and B26 are carried verbatim from the house briefs (`docs/archive/researcher_live_gap_2.md`, `researcher_i6d_book_leg_levelling.md`, `researcher_blend_weight_slug.md`).

Search sources consulted:
- [Page & Clemen 2013, author PDF](https://people.duke.edu/~clemen/bio/Published%20Papers/45.PredictionMarkets-Page&Clemen-EJ-2013.pdf)
- [Schulz & Lerch, ar5iv](https://ar5iv.arxiv.org/html/2106.09512)
- [Vannitsem et al., arXiv:2004.06582](https://arxiv.org/pdf/2004.06582)
- [Dare, Dennis & Paul, IDEAS](https://ideas.repec.org/a/eee/finlet/v13y2015icp130-136.html)
- [Stock & Watson, IDEAS](https://ideas.repec.org/a/jof/jforec/v23y2004i6p405-430.html)
- [Moskowitz, EconPapers](https://econpapers.repec.org/RePEc:bla:jfinan:v:76:y:2021:i:6:p:3153-3209)
- [Timmermann, EconPapers](https://econpapers.repec.org/RePEc:eee:ecofch:1-04)
- [Diebold & Shin, EconPapers](https://econpapers.repec.org/RePEc:nbr:nberwo:24967)
- [Gandar, Dare, Brown & Zuber, EconBiz](https://econbiz.de/Record/informed-traders-and-price-variations-in-the-betting-market-for-professional-basketball-games-gandar-john/10005214276)
- [Gandar et al. 1988, Scilit](https://scilit.com/publications/c990659542ad56b524e376acaa847c94)
- [Big Ideas in Sports Analytics, arXiv:2301.04001](https://arxiv.org/pdf/2301.04001)

---

## Appendix A: re-runnable commands

**Directory.** All scripts live in `/tmp/claude-1000/-home-trevor-Sportstradamus/dd99c1c3-a36f-4829-a8ba-72de8c688fea/scratchpad/lead_time/`, with outputs in its `out/`. The directory is session scratch, which a crash wipes; copy it (for example to `~/backups/sportstradamus/2026-10-21-live-gap-2/research/lead_time_scripts/`) before relying on it.

**How to run.**
```
cd /tmp/claude-1000/-home-trevor-Sportstradamus/dd99c1c3-a36f-4829-a8ba-72de8c688fea/scratchpad/lead_time
SPORTSTRADAMUS_ARCHIVE_READ_ONLY=1 nice -n 10 /home/trevor/Sportstradamus/.venv/bin/python <script>.py > out/<script>.txt
```

**Run order:**
1. `extract_archive.py`: from the read-only archive copy only, writes `out/arch_nfl_books.parquet` and `out/nfl_last_dfs.parquet`.
2. `build.py`: writes `out/A.parquet` and `out/B.parquet`.
3. `final_close.py`.
4. The reads, in any order.

**Scripts and outputs** (each script writes `out/<script>.txt`):

| Script | Output |
|---|---|
| `read1.py` | Items 1–3: counts, calibration and the book against the close per bin |
| `read2.py` | Slopes, encompassing, model-alone w\*, the back-out check |
| `read3.py` | Paired early against game day; item 5 |
| `read4.py` | λ\* by bin, pairs and cell |
| `read5.py` | Recommended economics, quote age, status flips |
| `read6.py` | Line-conversion split |
| `read7.py` | 10-25 Standalone pairs |
| `read8.py` | Same-version pairs and Δλ |
| `early_vs_gameday.py` | Pooled early against game day, recommended |
| `sim.py` | Walk-forward policies |
| `schedule_value.py` | Schedule value; Bates–Granger bound |
| `chain.py` | Served against model alone against book |
| `nhl_goals_check.py` | NHL with and without goals |
| `linemove.py`, `linemove2.py` | DFS line moves |
| `replay.py` (also `replay.parquet`), `replay_bydate.py`, `replay_cal.py`, `replay_nhl.py` | Archive replays |
| `inplay_check.py`, `sunday_polls.py` | In-play rows |

`common.py` and `read4_lib.py` hold the shared loaders and the vectorised cluster bootstrap.

---

## Load-bearing conclusions for the plan

1. **→ `live-gap-2.md` §5 "Next" item 4** (I6f, NFL offers scored days ahead): close it.
   - Early picks do not win less. Pooled NFL early-recommended gap: −.012 [−.068, +.026] against game day. ROI: +2.4 pp [−2.7, +11.6].
   - The paired same-version served Brier is unchanged: −.0006 [−.0018, +.0009].
   - No lead-time w or T schedule.
2. **→ `model_improvement_track.md` §3.4 lever verdicts**, new row: "Lead-time-conditional blend weight or temperature (I6f): **KILL 2026-10-29**".
   - Δλ is +.10 to +.16, upper bound .18–.27, against the .38 needed to earn δ/10.
   - Per-lead over common λ: 3.5 × 10⁻⁵ Brier in-sample, ±6 × 10⁻⁵ walk-forward.
   - Weighing the book less early worsens Brier by .003–.004.
   - Training's book read is already about c−24h.
   - Reopen only on Lever 4's rule. Brief: `researcher_lead_time_blend.md`.
3. **→ §6.10 WS-1 step 3**: the "time-decay discount" candidate is dead for NFL and NHL on the same evidence.
4. **→ `honest-receipts.md` §6 item 3** (I6f "Also open", `TRAINING_LOOKBACK_HOURS`): DEFER.
   - Worth 3–7% relative in w and about 10⁻⁵ Brier.
   - If it is ever done, anchor on kickoff − 30 min from the schedule, never on `date + 20 h` (in-play polls). Fold it into a re-inject done for another reason.
5. **→ §8.2 #0f / #0e** (I6d acceptance): add the report-only post-I6d λ\* read beside M3.
   - NFL, plus NHL without goals. Game clusters, 4 Sundays, ≥ 5,000 legs, read about 11-24.
   - Expect the levelled retrain to improve quoted-leg Brier (proxy −.006 to −.010) and shrink the recommended cohort 32–63% without a better gap.
6. **→ §8.2, a new instrument-hazard note:**
   - `Close Market Prob` ≠ `Market Prob` (.069 RMS at lead 0).
   - The archive stores in-play polls (3.6% of NFL rows), and clv's 20:00 UTC stand-in sits after Sunday early kickoffs.
   - The NFL chain lost to the book (+.0092 [+.0019, +.0159]) with attempts at the T cap. Re-read after I6d.
7. **→ §7.4 cross-league caveats:**
   - Lead effects are NFL-only: NHL scores 99.9% of its legs on game day.
   - NHL goals' constant book leg flips every NHL blend read (λ\* .70 → .40, β_W .83 → .10 without goals).
   - NHL goals is the one market whose book gains materially near puck drop (+.0043).
8. **→ `live-gap-2.md` §8 owner asks**: Lever 2 (capture first-scoring probabilities, about 1 dev-day) is the owner's call. It is the only way to keep "do early picks win less" monitored.
