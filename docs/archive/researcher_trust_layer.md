# In-repo research brief R1: a selection-aware probability layer ("Trust Prob") fit on all posted settled legs, applied post-argmax

**Question (verbatim):** "Should Sportstradamus add a selection-aware probability layer, fit on all posted settled legs and applied after the side is chosen, whose `Trust Prob` drives recommendations, Kelly and parlay legs; if so, which pre-registered specification, at what threshold, with what out-of-sample volume and ROI at the platform payout? KILL if no specification beats the served `Win Prob` on the selected tail out of sample, or if the served model probability has no material partial contribution."
**Date:** 2026-10-04 · branch `devel` @ `efe5106e` · R1 of the honest-Receipts plan (`~/.claude/plans/we-have-a-problem-deep-boot.md`) · read-only: archive opened `duckdb.connect(..., read_only=True)`, nothing written under `src/`, no pickles, no config, no git. Scripts and intermediates: `/tmp/claude-1000/-home-trevor-Sportstradamus/7c225c1a-7c1e-4cde-bede-cf7b5fe0cce8/scratchpad/r1/` (`prereg.py` frozen spec, `feats.py` decision-time archive join, `walk.py` walk-forward, `cfit.py` monotone logistic, `evalw.py` pre-registered evaluation, `ablate.py`, `selcomp.py`, `regime.py`, `sens.py`, `fallback.py`, `fallback2.py`). Every script ran in under 70 s on the full data; nothing was subsampled.

**VERDICT: KILL.** Do not build `prediction/trust.py`, in either tier. Both eligible layers beat served `Win Prob` on proper scores and remove most of the recommended-set overconfidence, but they do it **without the model**. Dropping served `Win Prob` does not significantly worsen out-of-sample log-loss for either class: logistic +0.00038 nats [−0.00011, +0.00088], one-sided p = 0.072; GBM −0.00013 [−0.00055, +0.00023], p = 0.75. That is the pre-registered kill rule ("a layer that ignores the model is model-free in disguise"). Independently, no class × threshold clears Tier B: 0 of 12 after Holm.

---

## TL;DR

- **An honest probability is achievable; a model-built one is not, on this data.** On today's served-p recommended legs (n = 3,819 out of sample), served p reads 0.609 against a 0.476 hit, a gap of +13.4 pp [+9.8, +16.1]. The constrained logistic cuts the gap to +2.6 pp [−0.5, +5.3] and the GBM to +3.0 pp [+0.1, +5.7]. The same layers with served p removed do as well: +1.1 and +1.8 pp. On those legs Trust Prob keeps only 13–16% of the model's deviation from the market reference, and the realized hit sits 1.9 pp *below* that reference. The overstatement belongs to the selector, not the model. On the `book_fallback` class, where `Win Prob` *is* the decoded sportsbook consensus, edge ≥ 5% selection still harvests +17.8 pp all-time (main-session §6 measurement, 2026-10-04) and +22.5 pp post-fix (n = 26). In every quote class the platform's own price sits within 1–8 pp of the recommended-tail hit rate (Finding 12).
- **The kill rule fires on the ablation (K2) for both eligible classes.** Served p carries a small real signal across all posted legs. The in-sample day-clustered Wald test on the served-p block gives χ² = 184 on 11 df, and a minimal market anchor gains +0.00035 nats out of sample (p = 0.023). The signal sits on quoted Unders, payouts ≤ 1.5, main lines, MLB hits, singles and walks, and freshly retrained versions. Given the decision-time market observables the selector already sees, it adds nothing out of sample. On the recommended tail it makes log-loss worse (−0.0022 nats). Excluding legs the platform would void does not rescue it, and neither does any selection-ROI comparison (every with-p minus no-p CI covers 0).
- **Tier B fails 12 of 12 after Holm.** The best raw cell is the logistic at Trust EV − 1 ≥ 0.08: 43 legs in 26 days (1.75 MLB legs/day, none elsewhere), ROI +26.5% [−6.2, +50.2], raw p = 0.05, Holm p = 0.60.
  - It rests on one day: 24 of the 92 legs at the 0.05 threshold fall on 2026-09-23, behind a single stale pre-lineup sportsbook slot.
  - Excluding 09-23, the 0.08 tail is +15.4% [−13, +51] on 27 legs.
  - At Underdog per-pick payouts of 1.81 or 1.87 it becomes +14.8% or −0.7%.
  - The GBM loses at every threshold (−9.3% to −22.9%). Its own tail is still overconfident by +7 to +19 pp: estimation noise in a flexible fit re-creates a smaller optimizer's curse (Smith & Winkler 2006, DOI 10.1287/mnsc.1050.0451).
- **Tier A is moot under the kill.** The logistic would pass A1 (own-tail gap CI covers 0) at 0.05/0.08 (n = 92 and 43), A2 and A3. The GBM fails A1 at every threshold. Under the plan's stricter success number (|mean Trust Prob − hit| ≤ 3 pp *and* a CI covering 0), both classes fail A1 at every threshold: the logistic's own-tail gaps are +5.4, +8.3, +3.8 and −5.8 pp. A market-anchored `Trust Prob`, even display-only, is the model-free engine the owner ruled out.
- **What R1 hands on instead:**
  - the per-league volume/ROI frontier the owner asked for;
  - a regime map of where the model still informs and where it is anti-informative, routed to R3/I6 as model-fix targets;
  - an evidence-backed abstain rule for leagues with no fit;
  - a retrain finding (exploratory): the model's information share is 0.47–0.61 in a version's first days and 0.05–0.14 one to two weeks later;
  - a grading defect for the realized ledger: about 5% of MLB hitter legs are non-starters, which Sleeper voids and we grade;
  - a pre-registered confirmatory re-run for mid-November on I2's decision-time columns.

---

## 0. Pre-registration (frozen before any outcome look at the archive features)

**Method rules (verbatim from the R1 brief, `r1_prompt.md` §Method rules):** "Post-fix data only (Date ≥ 2026-08-31, the book-leg fix era); walk-forward (expanding window, weekly refits, never fit on the evaluation week, every feature cut at decision time); day-clustered CIs (block bootstrap by Date, ≥ 2,000 resamples); pre-register the candidate set, Holm across it, report everything tried; price at the platform payout via `sportstradamus.realized.settled_offers` (platform payout, posted sides only, per-platform dedup; `Recommended` = `Win Prob × Payout − 1 ≥ 0.05` and `1 < Payout ≤ 2.5`; columns `Payout, Breakeven, Hit, Unit, Recommended, Quote ∈ {quoted, unquoted, fallback}`); NFL reported separately; KILL is a valid verdict; primary sources with DOIs / arXiv IDs."

**Frozen specification** (`r1/prereg.py`, written 2026-10-04 16:01 UTC; the only outcome-based knowledge before freezing was the prior selection-shrink brief):

- **Population.** `settled_offers(history)` with Date ≥ 2026-08-31, restricted to legs with a decision time.
- **Decision time.** `t_dec` = the platform's last `ladder` poll of that exact (player, market, line). Prophecize stages that poll immediately before scoring (`prediction/scoring.py:_match_league_offers` → `archive.add_dfs`), and history keeps the last scored snapshot per (key, Line, Platform) (`prediction/cli.py:_upsert_history`). Every archive feature is cut at `observed_at ≤ t_dec`.
- **Walk-forward.**
  - Eval weeks: W1 09-07..13, W2 09-14..20, W3 09-21..27, W4 09-28..10-04 (data ends 10-02).
  - Each week is fit on every leg dated before the week starts (expanding from 08-31), refit weekly.
  - Hyper-parameters are fixed in `prereg.py` and were never tuned on an eval week.
- **Inputs (w-free except served p).**
  - Market prices: logit served p; logit `Market Prob` on quoted/fallback rows; logit platform-implied chosen-side probability at `t_dec` (ladder `p_over`); logit breakeven `1/Payout`; logit same-line sportsbook rung price at `t_dec` with a missing flag.
  - Line and book context: w-free tail position `z_side`; platform main-line movement since open; sportsbook median-line movement; log book count; line spread across books; log hours since the last sportsbook quote; log hours the line has been on the board.
  - Model Version: age in days, plus a missing flag for fallback/legacy.
  - Categoricals: side, alt, platform, payout band (≤1.5, 1.5–2.0, 2.0–2.5, >2.5), league, cell.
  - The DFS platform's own price and line enter in their own right, not only through served p and book p. Without them a layer could not separate the `book_fallback` class, where `Win Prob ≡ Market Prob`. They are: the exact-line ladder implied price at `t_dec`, the rung breakeven `1/Payout` and payout band, the DFS-line-vs-consensus distance (`z_side`, `z_abs`), the alt flag, the same-line sportsbook rung, platform main-line movement, and a p × fallback interaction. (Checked against the coordinator's 2026-10-04 requirement: all were in the frozen set, so this is not a deviation.)
- **Classes (at most three).**
  - (i) Identity, q = p.
  - (ii) L2 logistic, C = 0.1 on standardized inputs, intercept unpenalized. Named interactions of logit p with: quoted, fallback, each payout band, alt, side, each league, platform. Cell enters as an L2-shrunk fixed effect.
  - (iii) LightGBM (Ke et al. 2017), binary objective: 300 trees, lr 0.03, depth 3, 7 leaves, min_child_samples 400, bagging 0.8, feature fraction 0.8, λ2 = 5, league and cell categorical. Monotone +1 on logit p and on the four market probabilities of the same event (`monotone_constraints_method="advanced"`).
- **Ablation.** The same class with served p and every p interaction removed. Test: Δ log-loss (no-p minus full) on all posted eval legs, day-clustered bootstrap, one-sided α = 0.05. A class that fails is disqualified, and if every non-identity class fails the verdict is KILL.
- **Selection.** `q × Payout − 1 ≥ t` for t ∈ {0, .02, .05, .08}, with payout in (1, 2.5] (the menu window, `MAX_FAVORED_PAYOUT`). Applied post-argmax; `Bet` is never changed.
- **Inference and tiers.**
  - Resample eval dates, B = 2,000, seed 1729.
  - Holm across 3 classes × 4 thresholds (12 one-sided tests of ROI > 0 at the platform payout; Holm 1979).
  - Tier A: A1 own-tail gap CI covers 0; A2 log-loss on all posted legs no worse than served p; A3 gap on the served-p recommended set significantly smaller than `Win Prob`'s.
  - Tier B: Holm-adjusted p < 0.05.
- **No-fit default.** A league with fewer than 2,000 fit legs or fewer than 5 distinct fit dates abstains (q = NaN, never recommended, Kelly 0). It never falls back to full trust (`[[no-evidence-full-trust-pattern]]`).

**Disclosed deviations.**

1. **Monotonicity mechanism for (ii).** The rule (non-decreasing in served p) was frozen, but the mechanism was not.
   - The unconstrained fit had negative slopes in served logit p on 4, 62, 102 and 87 of 105–168 observed indicator combinations in W1–W4. The cause is the anti-information region: payout > 2.5, alt lines, NFL/WNBA, quoted.
   - Class (ii) was therefore fit with linear constraints, slope ≥ 0 on every observed combination (SLSQP; on fallback rows the constraint is on the total derivative, since `Market Prob ≡ Win Prob` there). It converged every week with the constraints binding.
   - The unconstrained fit is reported as an ineligible reference. Out of sample it is indistinguishable (log-loss 0.63863 vs 0.63858).
2. **Tail position.** The dispatch lists `|Line − Projection| / Projection STD` but also bars any input containing w. `Projection` is the fused mean, so I used a w-free centre: the sportsbook median line at `t_dec`, else the platform main line, else the line itself, scaled by `CV × centre`. The w-containing version was not tested.
3. **Model Weight (w)** is not an input anywhere (the dispatch's w rule). The prior brief's descriptive check that λ does not track w stands.

## 1. Data actually used

| Item | Value |
|---|---|
| Settled posted legs, 2026-08-31..10-02 | 131,798 (`settled_offers`), 50 cells (MLB 17, NFL 14, NHL 10, WNBA 9), 32 dates |
| Dropped: no decision time | 74 (NFL Underdog legs with no exact-line ladder poll) |
| Fit sizes W1..W4 | 19,230 / 47,633 / 87,704 / 125,834 |
| Eval legs (W1–W4) | 112,494; league fitted: 98,388 (MLB 91,215 · NFL 5,832 · WNBA 1,341) on 26 days, 40 cells |
| Abstained by the no-fit rule | NFL W1–W2 (9,073), WNBA W2–W3 (4,311), NHL W4 (722) |
| Served-p recommended legs, eval, fitted leagues | 3,819 (all leagues incl. abstained: MLB 2,827 · NFL 2,501 · WNBA 134 · NHL 34) |
| Decision-time join | exact-line platform poll for 99.94% of legs (median 9 polls); on unquoted legs the last poll's implied price equals the persisted `Market Prob` on 64% (71% within 0.005); sportsbook consensus at `t_dec` 58.6%; same-line sportsbook rung 43.4% (main 51–77%, alt 2–17%) |
| League mix | 93% of fitted eval legs are MLB. NFL has 5 fitted eval dates. NBA is absent (season starts late October) |
| Quote class (fitted eval) | quoted 36,299 · unquoted 58,098 · fallback 3,991 (MLB 2,992, WNBA W4 999). A further 3,401 fallback legs abstained (WNBA W2–W3 3,208, NHL W4 193) |
| Where fallback lives post-fix | MLB pitcher cells (strikeouts, walks allowed, pitching outs; 3–4% of MLB legs); WNBA from 09-14 (73–75% of WNBA legs); NHL W4 (27%); NFL none |
| Fallback era split | 4,425 of the 4,463 all-time recommended fallback legs predate the window |

History today carries none of the I2 decision columns (`Scored At`, `Consensus Line`, `Model Weight`, `Quote *`, `Payout Over/Under`), so quote class is `Market Projection` notna plus `Model Version`, and decision time is the proxy above.

---

## Key Findings

### 1. Both layers are better probabilities than served p, on all posted legs and on the recommended tail (A2, A3 pass)

Out of sample, fitted leagues, n = 98,388, Δ = layer minus served p (negative is better), day-clustered 95% CI. Log-loss and Brier are strictly proper, so a lower value means a better probability, not just a better ranking (Gneiting & Raftery 2007, DOI 10.1198/016214506000001437).

| spec | log-loss | Brier | ΔLL vs p |
|---|---|---|---|
| (i) identity (served p) | 0.64155 | 0.22519 | 0 |
| (ii) monotone L2 logistic | 0.63858 | 0.22360 | −0.00297 [−0.00496, −0.00089] |
| (iii) monotone GBM | 0.63790 | 0.22350 | −0.00365 [−0.00548, −0.00174] |
| ablation (ii) without p | 0.63896 | 0.22362 | −0.00258 [−0.00471, −0.00038] |
| ablation (iii) without p | 0.63777 | 0.22346 | −0.00378 [−0.00570, −0.00174] |

On the served-p recommended set (n = 3,819; hit 0.476):

| spec | mean prob | gap | paired reduction vs p | log-loss |
|---|---|---|---|---|
| served p | 0.609 | +13.4 [+9.8, +16.1] | — | 0.722 |
| logistic | 0.502 | +2.6 [−0.5, +5.3] | 10.8 [9.8, 11.6] | 0.680 |
| GBM | 0.506 | +3.0 [+0.1, +5.7] | 10.4 [9.5, 11.1] | 0.680 |
| logistic without p | 0.487 | +1.1 [−2.0, +3.8] | 12.2 [11.3, 13.0] | 0.679 |

This is the optimizer's-curse correction working as theory says it should. The selector's inputs (p, payout, quote class and the rest) are all observed, so the selection is ignorable given X (Rosenbaum & Rubin 1983, DOI 10.1093/biomet/70.1.41). A conditional model E[y | X] fit on every posted leg is therefore unbiased on any subset selected by a function of X (Dawid 1994; Senn 2008, DOI 10.1198/000313008X331530). Heckman-type selection bias needs selection on unobservables (Heckman 1979, DOI 10.2307/1912352), and there is none here. Ranking on the posterior mean instead of the noisy estimate is the cure Smith & Winkler (2006) prescribe. The KILL condition "no specification beats the served `Win Prob` on the selected tail" therefore does **not** fire: at t = 0.08 the trust-selected set even beats the served-selected set by +40.7 pp ROI [+12.2, +63.1] (paired).

### 2. The ablation fails: given what the selector already sees, served p adds nothing out of sample (KILL rule K2)

This ablation is a forecast-encompassing test. The null is that the market-observables forecast encompasses the model (Harvey, Leybourne & Newbold 1998, DOI 10.1080/07350015.1998.10524759; for probability forecasts under the log score, Clements & Harvey 2010, DOI 10.1002/jae.1097). It is evaluated on walk-forward forecasts, so estimation error is included (Giacomini & White 2006, DOI 10.1111/j.1468-0262.2006.00718.x; Diebold & Mariano 1995, DOI 10.1080/07350015.1995.10524599).

| Δ log-loss = LL(no p) − LL(with p); > 0 means served p informs | logistic | GBM |
|---|---|---|
| all posted legs (pre-registered test) | +0.00038 [−0.00011, +0.00088], p = 0.072 | −0.00013 [−0.00055, +0.00023], p = 0.75 |
| non-fallback legs, n = 94,397 (the scope where the ablation is defined; on fallback rows p ≡ book, so the no-p model still sees p) | +0.00040 [−0.00011, +0.00090], p = 0.071 | −0.00009 [−0.00052, +0.00028], p = 0.67 |
| fallback legs, n = 3,991 (reported for completeness; not a test of served p, because the book input *is* served p) | +0.00003 [−0.00135, +0.00145], p = 0.50 | −0.00111 [−0.00183, −0.00031] (no-p better) |
| served-p recommended legs | −0.00046 [−0.00307, +0.00168] | −0.00089 [−0.00295, +0.00092] |
| quoted / unquoted | +0.00076 (p = .015) / +0.00018 (p = .30) | +0.00005 / −0.00017 |
| Under / Over | +0.00072 (p = .005) / −0.00033 | −0.00015 / −0.00008 |
| W1 / W2 / W3 / W4 | −0.00054 / +0.00020 / +0.00113 (p = .0005) / +0.00068 | −0.00094 (no-p significantly better) / +0.00012 / +0.00021 / −0.00038 |
| MLB 09-07..16, legs the platform voids excluded | −0.00067 [−0.00117, −0.00017] (no-p better) | −0.00096 [−0.00149, −0.00045] |

The signal is real but small, unstable, and absent where decisions are made:

- **In sample** (all post-fix legs, day-clustered): the served-p block is overwhelmingly significant, χ² = 184.5 on 11 df, p = 1e-33. The main slope is 0.33 ± 0.07 in logit units, or 0.19 ± 0.05 without interactions (z = 3.8). Much of the χ² is anti-information, for example the p × payout > 2.5 term at −2.88 ± 0.76.
- **Minimal market anchor** (book, platform-implied, breakeven, quote class, side), the S5-like world: p adds +0.00035 nats [+0.00000, +0.00069] out of sample, p = 0.023. On the recommended tail it costs −0.0022 nats (p_one-sided = 0.97).
- **Decision level**: the with-p minus no-p selection ROI CIs cover 0 at every threshold. The logistic's best is +18.7 pp [−7.6, +38.7] at t = 0.08; the GBM is near 0 throughout. Excluding 09-23, the no-p logistic is ahead at t = 0.05 (+4.2% vs −2.3%).

So the model's information beyond the market is largely what the structural observables (cell, side, payout band, tail position) already carry. The selection literature has a consistent prior here: betting odds are the best available forecasts (Spann & Skiera 2009, DOI 10.1002/for.1091; Štrumbelj 2014, DOI 10.1016/j.ijforecast.2014.02.008), and the residual a targeted adjustment captures is mainly the favourite–longshot structure (Snowberg & Wolfers 2010, DOI 10.1086/655844; Goto, Takeishi & Yairi 2026, arXiv:2604.17194, preprint).

### 3. On the recommended tail Trust Prob collapses to the market

Take non-fallback served-p recommended legs, with market reference m = the sportsbook decode on quoted legs and the platform's own price on unquoted legs:

| | value |
|---|---|
| p − m | +11.4 pp |
| hit − m | **−1.9 pp** |
| layer keep share Σ(q−m)(p−m)/Σ(p−m)² | logistic 0.13, GBM 0.16 (logistic without p 0.02) |

The realized information share on those legs (λ_sel, the prior brief's estimator: through-origin slope of y − m on p − m) is negative in every league × side and quote class. It is significantly negative on alt lines (−0.35 ± 0.10) and in the 2.0–2.5 payout band (−0.25 ± 0.10); see Finding 6. Curing the curse cannot create edge. Where E[y | X] is the market, an honest layer recommends almost nothing, and what it recommends is market structure, not model. The `book_fallback` class, where no model enters at all, shows the same tail overstatement at full size (Finding 12).

### 4. Tier B fails everywhere, and the one promising cell is a single day

Out of sample, fitted leagues, 26 days. ROI at the platform payout (Underdog Boost × 1.78, Sleeper posted), day-clustered 95% CI, Holm across the 12:

| class | t | n | legs/day | hit | mean q | gap (pp) | payout | ROI | raw p | Holm p | A1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| identity | 0 | 8,577 | 330 | .526 | .620 | +9.4 [+7.2, +11.2] | 1.74 | −10.8% [−13.8, −7.3] | 1 | 1 | no |
| identity | .02 | 6,197 | 238 | .511 | .615 | +10.4 [+7.9, +12.3] | 1.79 | −10.9% [−14.3, −6.9] | 1 | 1 | no |
| identity | .05 | 3,819 | 147 | .476 | .609 | +13.4 [+9.8, +16.1] | 1.85 | −14.0% [−19.0, −8.0] | 1 | 1 | no |
| identity | .08 | 2,353 | 90.5 | .461 | .607 | +14.6 [+10.7, +17.9] | 1.90 | −14.2% [−20.5, −7.3] | 1 | 1 | no |
| logistic | 0 | 605 | 23.3 | .597 | .651 | +5.4 [+1.3, +9.5] | 1.63 | −5.4% [−12.5, +1.9] | .92 | 1 | no |
| logistic | .02 | 264 | 10.2 | .606 | .689 | +8.3 [+0.9, +16.4] | 1.58 | −7.3% [−20.8, +5.6] | .86 | 1 | no |
| logistic | .05 | 92 | 3.5 | .685 | .722 | +3.8 [−6.2, +16.5] | 1.57 | +5.1% [−17.2, +21.8] | .32 | 1 | yes |
| logistic | .08 | 43 | 1.7 | .791 | .733 | −5.8 [−18.3, +11.5] | 1.60 | +26.5% [−6.2, +50.2] | .05 | .60 | yes |
| GBM | 0 | 1,165 | 44.8 | .549 | .622 | +7.2 [+2.9, +11.3] | 1.70 | −9.3% [−16.3, −1.9] | .99 | 1 | no |
| GBM | .02 | 612 | 23.5 | .529 | .623 | +9.3 [+4.2, +13.7] | 1.73 | −11.8% [−19.6, −3.1] | .99 | 1 | no |
| GBM | .05 | 222 | 8.5 | .500 | .622 | +12.2 [+5.4, +19.6] | 1.79 | −13.7% [−26.7, −1.3] | .99 | 1 | no |
| GBM | .08 | 80 | 3.1 | .425 | .611 | +18.6 [+10.3, +28.3] | 1.87 | −22.9% [−41.6, −6.1] | .99 | 1 | no |

The logistic's 0.05/0.08 tail is fragile in three independent ways:

- **One day carries it.** 24 of the 92 legs at t = 0.05 are on 2026-09-23 (MLB Unders on Coby Mayo, Gunnar Henderson, Myles Straw, Pete Alonso and others). They were selected because a single 19:30 UTC sportsbook slot priced them as near-certain Unders. For example, DraftKings had Mayo's hits Under at 0.803 while three books at 22:12, after `t_dec`, had 0.41–0.43; and seven different hitters carried an identical 0.9036 home-run-Under price. That is a stale pre-lineup quote, not a model edge. Ex-09-23: t = 0.05 gives −2.3% [−23.9, +18.9] on 68 legs; t = 0.08 gives +15.4% [−13.3, +51.4] on 27 legs.
- **The payout convention moves it.** At Underdog per-pick payouts of 1.81 / 1.87 (the live Power roots; R2 owns the convention), the logistic at 0.08 becomes +14.8% [−15.8, +35.0] on 54 legs and −0.7% [−23.8, +18.9] on 107 legs. Nothing passes at any convention.
- **Composition.** The largest group in the 0.05 tail (28 of 92 legs) is MLB quoted Unders, where the decoded book (0.77) and rung (0.72) sit far above the platform's 0.57. Quoted Overs in the same tail read 0.79 and hit 0.57, so the layer is overconfident there too.

The GBM's failure is the textbook second-order curse. A flexible fit's estimation noise is largest in sparse tail regions, so ranking on q·π selects its own positive errors. Smith & Winkler's posterior-mean argument applies to the layer as much as to the model, and inference on such winners needs the corrections of Andrews, Kitagawa & McCloskey (2024, DOI 10.1093/qje/qjad043). Boosted trees also give distorted probabilities without a calibration stage (Niculescu-Mizil & Caruana 2005, DOI 10.1145/1102351.1102430). A post-hoc calibrator would be one more fitted stage, with its own noise, in the same sparse tail.

### 5. How this differs from every killed form, and why it still fails

| Killed form (`docs/archive/researcher_selection_shrink.md`) | Its estimator | What this layer adds | Outcome here |
|---|---|---|---|
| S1/S2/S4 scalar λ toward the book (global / league / cell) | E[y] = b + λ(p − b), one λ per grain, quoted rows only | Conditions on the selector's observables (payout band, quote class incl. unquoted via the platform price, side, alt, tail position, rung price, book count and spread, movement, staleness, cell), fit on all posted legs | Removes the curse (A3) and confirms λ is not constant (λ_all +0.27 at payout ≤ 1.5 vs −0.21 at 2.0–2.5); no model information is left to keep |
| S3 logistic on the logit difference | one slope + intercept | named interactions, monotone constraint | same |
| (b) higher edge floor | ranks on p·π | re-ranks on q·π | trust-ranked tail beats served-ranked (−5.4% vs −10.8% at t = 0), still negative |
| (c) `kelly_shrinkage` as λ | wrong units (BSS) | not used | — |
| (d) unconditional book CITL | intercept only | conditional | — |
| S5 market-anchored stack (book + model + 1/π, all rows) | linear, no conditioning, no archive | decision-time archive features, interactions, monotone GBM, abstain default | S5-like minimal anchor: 36 legs in 26 days at t = 0 (cf. S5's 45 of 3,469); the full layer finds 605, which lose −5.4% |

The killed shrink approximates the posterior-mean correction with one regression slope. That treats the selection as happening on noise in p alone, and it is misspecified because the slope moves with the selector's own inputs. This layer is the general version of that correction, estimated in the selector's coordinates. Its residual risks are the ones the results show: estimation noise in the tail (GBM), and nothing for it to preserve when the model has no conditional information.

### 6. Regime map: where the model still informs (route to R3 / I6 as model-fix targets, never pulls)

Out-of-sample window, non-fallback legs:

- **λ_all / λ_sel**: the realized information share (through-origin slope of y − m on p − m, day-clustered SE) on all posted legs and on served-p recommended legs.
- **keep**: the logistic's out-of-sample share of p − m. Low keep means Trust Prob sits far below served p, back at the market reference; this is the plan's "where Trust Prob sits far below served p" map.
- **gap**: served-p recommended gap, served p → logistic.

| segment | n | λ_all | λ_sel | keep | gap p → q (pp) |
|---|---|---|---|---|---|
| MLB Over / Under | 26,683 / 61,540 | 0.22 ± 0.08 / 0.23 ± 0.10 | −0.21 ± 0.17 / −0.09 ± 0.20 | 0.31 / 0.32 | +15.7 → +3.6 / +10.7 → +1.4 |
| NFL Over / Under | 2,047 / 3,785 | −0.00 ± 0.13 / 0.07 ± 0.05 | −0.09 ± 0.26 / −0.22 ± 0.13 | 0.12 / 0.20 | +14.6 → +1.6 / +18.8 → +6.8 |
| quoted / unquoted | 36,299 / 58,098 | 0.26 ± 0.11 / 0.16 ± 0.09 | −0.23 ± 0.15 / −0.08 ± 0.14 | 0.24 / 0.30 | +14.6 → +1.3 / +12.8 → +3.0 |
| payout ≤ 1.5 | 62,511 | **0.27 ± 0.07** | +0.06 ± 0.25 | 0.37 | +11.0 → +0.3 |
| payout 1.5–2.0 | 30,562 | 0.17 ± 0.08 | −0.11 ± 0.13 | 0.17 | +12.3 → +2.0 |
| payout 2.0–2.5 | 1,256 | **−0.21 ± 0.09** | **−0.25 ± 0.10** | 0.07 | +16.3 → +4.2 |
| main / alt line | 72,311 / 22,086 | 0.26 ± 0.06 / **0.01 ± 0.08** | −0.10 ± 0.12 / **−0.35 ± 0.10** | 0.27 / 0.31 | +12.5 → +1.8 / +18.3 → +7.2 |
| MLB hits | 13,207 | **0.38 ± 0.12** | **+0.47 ± 0.19** | 0.43 | +5.1 → −4.3 |
| MLB singles / walks | 5,425 / 8,758 | **0.57 ± 0.16 / 0.59 ± 0.14** | +0.04 ± 0.22 / n < 150 | 0.51 / 0.35 | +9.8 → +3.5 / +12.0 → +0.2 |
| MLB total bases / runs | 17,162 / 9,791 | 0.19 ± 0.12 / 0.16 ± 0.13 | −0.05 ± 0.28 / −0.25 ± 0.26 | 0.27 / 0.28 | +10.6 → +0.2 / +11.8 → +2.5 |
| MLB rbi | 9,583 | 0.10 ± 0.13 | n < 150 | 0.31 | +18.7 → +10.1 |
| MLB hits+runs+rbi | 18,917 | **0.00 ± 0.13** | **−0.71 ± 0.28** | 0.24 | +15.9 → +4.9 |
| MLB hits allowed / runs allowed | 2,577 / 1,790 | 0.23 ± 0.13 / −0.06 ± 0.10 | −0.12 ± 0.22 / −0.32 ± 0.26 | 0.23 / 0.08 | +16.5 → +4.2 / +18.4 → +6.6 |
| Model Version unseen in the fit (fresh retrain) | 17,327 | **0.58 ± 0.10** | — | 0.33 | +6.9 → −1.5 |
| Model Version seen | 77,070 | 0.13 ± 0.06 | — | 0.27 | +14.2 → +3.1 |

Model-fix targets for R3, ranked by where the selected tail actually sits:

1. **The 2.0–2.5 payout band and alt lines.** Served p is anti-informative there (λ < 0), and these are where recommendations concentrate (1,043 of 1,256 band-2.0–2.5 legs are recommended). This is R3's DFS-rung decode and tail-calibration items (rung `Market Prob` from the consensus shape, tail reliability by z and alt flag, I6c).
2. **NFL quoted legs** (λ_all −0.13 Over, −0.23 Under; NFL Under hit 0.526 vs served 0.587 vs market 0.551). This is the NFL Under bias by Model Version (R3; prior brief Finding 3).
3. **MLB hits+runs+rbi and runs allowed** (λ_all ≈ 0, λ_sel −0.71 on HRR). Composite and summed stats: R3's combo-sum and quote-skew items.
4. **Where the model works:** MLB hits, singles and walks, payouts ≤ 1.5, main lines, quoted MLB Unders (0.51 ± 0.16). These are the cells to protect in any retrain.

### 7. Retrains: the information share decays with version age (exploratory, two retrains)

λ_all by MLB version family and calendar week:

| version | first days | week +1 | week +2 |
|---|---|---|---|
| 20260903 | 0.47 ± 0.18 (wk 08-31) | 0.23 ± 0.09 | 0.05 ± 0.10 |
| 20260917 | 0.61 ± 0.11 (09-17..20) | 0.14 ± 0.15 | 0.13 ± 0.78 |

NFL 20260829 read 0.36 ± 0.11 in its first NFL week and 20260917 read 0.05 ± 0.08.

- **The pooled layer under-trusts fresh versions.** It keeps 0.33 of p − m on unseen-version legs, whose realized λ is 0.58.
- **Serving staleness is not the mechanism.** `prophecize` calls `Stats.update()` every run (`prediction/cli.py:232`). The pattern fits concept drift, where the fitted input-outcome relation goes stale between retrains (Gama et al. 2014, DOI 10.1145/2523813), but two retrains do not establish it.
- **A post-hoc lp × age interaction did not help out of sample.** It is unstable week to week (+0.036, −0.015, −0.005, −0.006 per day) and gives Δ log-loss −0.00008 vs without. It is therefore not evidence for a fix, only a pre-registration candidate for the November re-run.
- **Implication for any layer keyed to Model Version:** a retrain is a regime change. Pooling across versions mis-states trust exactly in the days after a retrain, when the model is most informative.

### 8. No-fit default: abstain; pooled cross-league transfer is unsafe

Applying the pooled fit to the league-weeks the rule abstained on gives:

| league-week | n | Δ log-loss vs served p (logistic / GBM) | t = 0 selections (logistic / GBM) |
|---|---|---|---|
| NFL W1 | 4,706 | +0.0091 (worse) / −0.0107 | 7 legs +30.7% / 47 legs −21.6% (gap +16.8 pp) |
| NFL W2 | 4,367 | −0.0133 / −0.0106 | 203 legs −17.2% (gap +14.3 pp) / 186 legs −20.2% (gap +16.9 pp) |
| WNBA W2 | 1,947 | +0.0044 / +0.0049 (both worse) | 6 legs −23.6% / 7 legs +1.1% |
| WNBA W3 | 2,364 | −0.0111 / −0.0108 | 0 / 10 legs +3.4% |
| NHL W4 | 722 | −0.0071 / −0.0107 | 0 / 0 |

The pooled fit's sign flips across league-weeks, and where it selects in volume (NFL W2) it loses 17–20% with a 14–17 pp gap.

"No evidence" must resolve to abstain (q = NaN, no recommendation, Kelly 0), not to the pooled fit and never to full trust. This is the reject option (Chow 1970, DOI 10.1109/TIT.1970.1054406), applied per league. NBA would abstain until it has ≥ 2,000 settled posted legs over ≥ 5 dates.

### 9. Phantom gate: no edge is being truncated; do not persist phantom rows

On unquoted legs, bin by D = p − implied. Realized hit − implied is about +0.01 to +0.02 for D in (0.02, 0.10], then falls back to −0.006 at D in (0.12, 0.15]. The layer's q − implied keeps rising (+0.022 → +0.033), but that is the layer extrapolating, not an edge the outcomes support. The plan's I2 condition ("persist phantom rows if Trust Prob still rises near |D| = 0.15") is formally met by the layer and refuted by the outcomes. Leave the gate and the history schema as they are.

### 10. NFL, reported separately (W3–W4 fitted, 5 dates, 5,832 legs)

- **Layers vs served p:** both layers beat served p by about 0.015–0.018 nats (logistic −0.0176 [−0.0294, −0.0141]). The no-p versions match them (−0.0171 / −0.0150), and the ablation is +0.00058 [−0.00065, +0.00242].
- **Recommended tail:** served p's gap is +17.3 pp [+14.1, +32.8] on 938 legs (hit 0.434); the layers bring it to +5.0 and +4.7 pp.
- **Trust selections lose badly in NFL:**
  - logistic, t = 0: 57 legs, hit 0.333, −37.5% [−66.2, −30.4];
  - GBM, t = 0: 192 legs, −29.6%.
- **Reliability of the CIs:** five day-clusters is below the range where cluster-bootstrap percentile intervals hold size (Cameron, Gelbach & Miller 2008, DOI 10.1162/rest.90.3.414), so treat every NFL CI as indicative only.

### 11. A grading defect for the realized ledger (not R1's lane; flagged because it biases every Receipts number)

Sleeper's rules: a "Batter must start and record a plate appearance", otherwise the pick is DNP and voided (support.sleeper.com, article 9047931). `annotate_offer_outcomes` grades any game-log row.

- **Size.** On the dev game log (through 2026-09-16), 5.4% of post-fix MLB hitter legs are non-starters or 0-PA. On those legs Unders "hit" 0.893 and Overs 0.225.
- **Effect.** Removing them moves Sleeper's served-p recommended hitter ROI from −6.8% to −10.9% (995 legs, 73 voids).
- **Unverified:** Underdog's rule.
- **Not a driver of this verdict.** Trust selections carry fewer voids (2.5%) than served-p selections (5.2%), and excluding voids makes served p's contribution worse, not better (Finding 2).

### 12. The fallback class: the same curse with no model in the loop (coordinator's main-session §6 measurement)

On `book_fallback` rows (`Model Version == "book_fallback"`, served when a cell has no pickle or no player matched: `prediction/scoring.py:_score_market`), `Win Prob` is the sportsbook consensus decoded to the DFS line. So `Win Prob ≡ Market Prob`, and no model enters. The rows still pass through the same `finalize_records` tail as model rows, which means a raw `Kelly` on the decoded-book probability and menu eligibility.

The main-session §6 measurement (2026-10-04, all-time, `settled_offers`, posted sides, platform payout) finds:
- the class is calibrated on all posted legs (n 57,317, read .600, hit .594);
- it is overstated by **+17.8 pp** on the recommended subset (n 4,463, read .670, hit .492, payout 1.85, ROI −9.9%);
- the quoted and unquoted recommended cohorts overstate by +16.1 and +11.4 pp.

The pre-registered window reproduces it (out of sample, fitted leagues, served-p recommended legs, gap in pp with a day-clustered CI for served p):

| quote class | posted n | recommended n | hit | platform implied | breakeven | served p (gap) | logistic (gap) | GBM (gap) |
|---|---|---|---|---|---|---|---|---|
| fallback | 3,991 | 26 | .385 | .465 | .533 | .610 (+22.5 [−1.4, +43.0]) | .536 (+15.2) | .530 (+14.5) |
| quoted | 36,299 | 1,076 | .468 | .476 | .542 | .614 (+14.6 [+6.5, +19.8]) | .481 (+1.3) | .487 (+1.8) |
| unquoted | 58,098 | 2,717 | .480 | .494 | .555 | .607 (+12.8 [+9.9, +16.0]) | .509 (+3.0) | .513 (+3.3) |

What this settles:

- **The overstatement is a property of the selector, not of the model.** Selecting on (probability × payout − 1) ≥ 5% picks the legs where the probability's error against the platform's own price is largest. A probability that is unbiased on average is then biased on the selected subset. That is the optimizer's curse in its textbook form (Smith & Winkler 2006, DOI 10.1287/mnsc.1050.0451), and here no model is involved.
  - In every quote class the platform's implied price sits within 1–8 pp of the recommended-tail hit rate, below breakeven, while the served probability is 13–23 pp high.
  - All three classes are calibrated in bulk: within 1–2 pp on all posted legs all-time (§6: fallback .600 read vs .594 hit, quoted .621 vs .613, unquoted .633 vs .621), and out of sample in this window (fallback .621 vs .627, quoted .639 vs .635, unquoted .639 vs .628). The gap is created by the selection.
  - On all fallback rows, binned by D = p − implied: where the decoded book leads the platform by 0–5 pp, it points the right way (hit − implied +0.018 and +0.042). In the next bin, D in (0.05, 0.10], it turns against itself: hit − implied is −0.063 while p − implied is +0.068 (n = 199, spread over 21 days, mostly pitcher strikeouts, walks allowed and pitching outs). Beyond 0.10 there are only 38 legs. This mirrors the unquoted phantom finding (Finding 9).
  - The reversal bin has no alt lines. Its sportsbook quotes are older (median 5.5 h since the last quote, against 2.4–3.4 h where the decode holds) and its DFS line sits further from the consensus line (mean |z| 0.50 against 0.19–0.40). A stale consensus and a decode across a line gap both fit; this is the book-quote and DFS-decode work in R3, not a model problem.
  - The consensus-versus-outlier logic of Kaunitz, Zhong & Kreiner (2017, arXiv:1710.02824) is to bet where one price strays from the consensus. It does not carry over here: once the decoded consensus exceeds the DFS price by more than 5 pp, the DFS price is the one that holds. That fits Levitt's (2004, DOI 10.1111/j.1468-0297.2004.00207.x) evidence that a price-setter prices to be right rather than to balance its book. This is a bet on mechanism, not a test.
- **Implication (1): the platform's own price and line were in the frozen feature set** (Pre-registration, inputs). Even with them, the layers do not beat the decoded book's proper scores on fallback rows: logistic +0.00047 [−0.00335, +0.00371], GBM +0.00330 [−0.00101, +0.00755] nats, both not significantly worse. On the 26-leg fallback tail they close about a third of the gap (+22.5 → +15.2 / +14.5 pp). A plausible reason, untested: 3,991 fitted fallback legs (with most WNBA fallback weeks abstained) are too few for the pooled p × fallback terms to learn the D > 0.05 reversal.
- **Implication (2): the ablation is undefined on fallback rows, so the pre-registered test is re-read without them.** On the 94,397 non-fallback legs: logistic +0.00040 [−0.00011, +0.00090], p = 0.071; GBM −0.00009 [−0.00052, +0.00028], p = 0.67. The fallback row is in the Finding 2 table for completeness (logistic +0.00003, GBM −0.00111 nats), and the fallback class has its own frontier rows (pooled by quote class, and per league). The verdict is unchanged.
- **The class mostly predates the window.**
  - All-time recommended fallback legs by month: 2026-07, 443; 2026-08, 3,982; post-fix, 38 (hit .395, read .619, ROI −26.0%).
  - Fallback share of posted legs: 73% in July, 27% in August, 6% in September, 29% on 10-01..02 (two days).
  - The all-time 17.8 pp is therefore mostly a pre-fix-era number. The post-fix number agrees in sign and size on very little data.
- **A `Trust Prob` on fallback rows would be model-free by construction**, whatever K2 says, because those rows have no model input.
  - The layers select more fallback legs than served p does (logistic t = 0: 169 against 68; GBM 216), and that slice is flat to negative: logistic +0.2% [−10.7, +13.0], GBM −9.9% [−19.5, −0.2]. See the by-quote-class frontier.

---

## Recommendation / routing protocol

1. **KILL I4 entirely (I4a–I4f).**
   - Do not build: `prediction/trust.py`, the `Trust Prob` / `Trust EV` / `Rec Rule` columns, the `policy_v2` persona, or the Receipts `trust_rate`.
   - `Bet`, `Win Prob`, the menu, Kelly and the parlay engine stay on today's path.
   - R2 should not wait on Trust Prob leg marginals; none will exist.
2. **Route the regime map (Finding 6) to R3 → I6 as model-fix targets**, in this order: the 2.0–2.5 band and alt rungs (decode and tail calibration), NFL quoted Unders (bias by version), composite MLB stats (HRR, runs allowed). Protect MLB hits, singles and walks and the ≤ 1.5 band in any retrain. These are never grounds to pull, demote or withhold a cell.
3. **No-fit default, as a standing rule for any probability consumer:** abstain (NaN, Kelly 0) below 2,000 legs and 5 dates per league. Pooled transfer failed on NFL W1/W2 and WNBA W2 (Finding 8).
4. **Retrain handling for any future layer or monitor:** treat a new Model Version as a regime change. Report fresh-version legs separately, and never let a pooled fit set trust for a version it has not seen (Finding 7).
5. **Keep the phantom gate; do not persist phantom rows** (Finding 9).
6. **Standing monitor:** the I1d "edge captured" column `(hit − book)/(pred − book)` on the recommended cohort is λ_sel. Its trailing-30-day day-clustered CI is the one live number that would reopen this lever (below).
7. **Two items that R1's kill leaves unfixed:**
   - **Kelly's no-evidence path.** I4d was its vehicle; `resolve_shrinkage` still returns `clip(live_bss)` from a live segment of any size when training BSS is missing (`[[no-evidence-full-trust-pattern]]`). It needs its own fix: Kelly sizing on an estimated edge has to shrink with the estimate's uncertainty (Baker & McHale 2013, DOI 10.1287/deca.2013.0271), and a tiny live segment carries the most uncertainty.
   - **The DNP grading defect** (Finding 11), for the realized/Receipts lane.
8. **`book_fallback` recommendations are a model-free engine today. This is an owner decision, flagged and not decided here.**
   - Fallback rows are scored through the same `finalize_records` tail, so the menu can flag and size them on the decoded sportsbook consensus alone.
   - Record: 4,463 legs all-time at −9.9%, and 38 post-fix at −26.0% (Finding 12).
   - By the owner's own rule (no model-free engine), they should not carry an edge flag. Removing it is not a model pull, because no model serves those rows.
   - The main session should confirm how the menu and `strategies/kelly.py` treat `book_fallback` rows. The unfixed no-evidence Kelly path (item 7) is the risk on exactly these cells.

## Volume / ROI frontier per league (owner decision 4)

- **Volume:** legs per league-day over that league's eval days. Abstained days count 0; nothing is floored.
- **Columns:** zero = days with no legs; ROI [95% CI] at the platform payout, day-clustered. Every class at every grid threshold; artifact `r1/frontier.csv`.
- **NHL** has 3 eval days, so its CIs carry no information.

| class | t | MLB (24 days, all fitted) | NFL (11 days, 5 fitted) | WNBA (13 days, 4 fitted) | NHL (3 days, 0 fitted) |
|---|---|---|---|---|---|
| identity | 0 | 292/day, −10.0% [−13.8, −6.0] | 364/day (median 98), −11.4% [−19.0, −7.4] | 22.7/day, −1.1% [−16.9, +15.8] | 28/day, −5.6% [−26.7, +19.1] |
| identity | .02 | 203/day, −10.1% [−14.8, −5.7] | 303/day (median 82), −12.2% [−21.1, −8.6] | 17.0/day, +1.7% [−15.7, +19.3] | 20.3/day, −3.4% [−28.9, +27.3] |
| identity | .05 | 118/day, −12.5% [−20.0, −5.9] | 227/day (median 60), −14.9% [−24.7, −10.5] | 10.3/day, +6.2% [−13.3, +31.8] | 11.3/day, −5.0% [−35.2, +90.0] |
| identity | .08 | 66.8/day, −12.8% [−21.5, −4.9] | 172/day (median 42), −14.1% [−26.1, −7.2] | 5.5/day (3 zero), +11.3% [−16.4, +41.9] | 3.3/day, −27.7% [−55.8, +90.0] |
| logistic | 0 | 22.8/day (median 18, 0 zero), −2.0% [−8.5, +4.4] | 5.2/day (6 zero), −37.5% [−66.2, −30.4] | 0 | 0 (abstain) |
| logistic | .02 | 10.0/day (7.5, 1 zero), −2.1% [−15.0, +9.9] | 2.1/day (6 zero), −62.3% | 0 | 0 |
| logistic | .05 | 3.7/day (2, 4 zero), +7.6% [−14.3, +23.5] | 0.36/day (9 zero), −50.2% | 0 | 0 |
| logistic | .08 | 1.75/day (1, 8 zero), +29.5% [−0.5, +50.0] | 0.09/day (10 zero), −100% | 0 | 0 |
| GBM | 0 | 40.3/day (31, 2 zero), −5.1% [−11.6, +1.2] | 17.5/day (6 zero), −29.6% [−45.2, −24.9] | 0.38/day, −28.8% | 0 |
| GBM | .02 | 20.1/day (17.5, 2 zero), −6.6% [−14.9, +1.6] | 11.5/day, −30.1% | 0.15/day | 0 |
| GBM | .05 | 6.3/day (4, 3 zero), −7.8% [−23.3, +6.2] | 6.4/day, −25.2% [−100, −0.4] | 0.08/day | 0 |
| GBM | .08 | 2.0/day (2, 9 zero), −14.7% [−38.8, +14.5] | 3.0/day, −34.7% | 0 | 0 |

**How to read it.** The single-league MLB logistic at 0.08 (+29.5%, lower bound −0.5%) is not Holm-protected and is the 2026-09-23 stale-quote cluster (Finding 4). The honest menu the current models support is 0–4 MLB legs a day with no statistical support, and nothing in NFL, WNBA or NHL. MLB's postseason ends in about four weeks, which takes that league's volume to zero.

**By quote class (the `book_fallback` class reported separately).** Fitted leagues, pooled; legs/day over the 26 eval days; ROI [95% CI] at the platform payout, day-clustered. Artifact: `r1/frontier_by_quote.csv`.

| class | t | quoted: n (per day), ROI | unquoted: n (per day), ROI | fallback: n (per day), ROI |
|---|---|---|---|---|
| identity | 0 | 2,222 (85.5), −12.0% [−18.7, −3.5] | 6,287 (241.8), −10.2% [−13.7, −7.0] | 68 (2.6), −22.0% [−50.4, +9.4] |
| identity | .02 | 1,635 (62.9), −12.8% [−20.3, −1.9] | 4,512 (173.5), −10.1% [−14.2, −6.8] | 50 (1.9), −21.2% [−57.8, +19.1] |
| identity | .05 | 1,076 (41.4), −14.6% [−23.9, −0.3] | 2,717 (104.5), −13.7% [−19.4, −8.7] | 26 (1.0), −25.6% [−69.2, +23.4] |
| identity | .08 | 729 (28.0), −14.3% [−25.8, +3.8] | 1,606 (61.8), −14.3% [−21.3, −9.0] | 18 (0.7), −1.7% [−56.2, +52.2] |
| logistic | 0 | 126 (4.8), −5.7% [−37.7, +13.7] | 310 (11.9), −8.2% [−15.8, −0.3] | 169 (6.5), +0.2% [−10.7, +13.0] |
| logistic | .02 | 88 (3.4), −2.8% [−38.5, +15.1] | 105 (4.0), −12.4% [−25.9, +3.6] | 71 (2.7), −5.4% [−29.7, +19.2] |
| logistic | .05 | 43 (1.7), +17.8% [−22.7, +41.4] | 28 (1.1), −10.6% [−29.6, +11.7] | 21 (0.8), +0.1% [−62.9, +44.7] |
| logistic | .08 | 28 (1.1), +32.3% [−23.7, +57.0] | 8 (0.3), −0.7% [−52.0, +34.5] | 7 (0.3), +34.3% [−55.4, +102.4] |
| GBM | 0 | 218 (8.4), −16.6% [−35.0, +8.0] | 731 (28.1), −6.9% [−13.3, −0.3] | 216 (8.3), −9.9% [−19.5, −0.2] |
| GBM | .02 | 131 (5.0), −21.7% [−45.0, +7.0] | 357 (13.7), −7.9% [−16.6, +3.5] | 124 (4.8), −12.5% [−26.1, +0.6] |
| GBM | .05 | 58 (2.2), −30.0% [−66.1, +1.3] | 112 (4.3), −7.2% [−22.4, +9.7] | 52 (2.0), −9.7% [−32.7, +8.7] |
| GBM | .08 | 26 (1.0), −31.0% [−100, +48.9] | 38 (1.5), −19.3% [−40.0, −1.5] | 16 (0.6), −18.5% [−74.0, +23.4] |

The logistic's positive quoted tail at .05/.08 is the 2026-09-23 MLB quoted-Under cluster (Finding 4).

**The `book_fallback` class per league per day** (same denominators: MLB 24 days, WNBA 13, NHL 3; NFL has no fallback legs). Identity counts every posted day, abstained weeks included, because identity is today's served path. Artifact: `r1/frontier_fallback_by_league.csv`.

| class | t | MLB fallback: n (per day, zero days), ROI | WNBA fallback | NHL fallback |
|---|---|---|---|---|
| identity | 0 | 62 (2.6, 9), −26.0% [−56.8, +5.3] | 29 (2.2, 2), +25.0% [−3.9, +55.2] | 1 leg |
| identity | .02 | 46 (1.9, 9), −22.2% [−59.6, +18.2] | 18 (1.4, 4), +1.4% [−31.6, +42.5] | 0 |
| identity | .05 | 26 (1.1, 13), −25.6% [−70.8, +24.0] | 4 (0.3, 9) | 0 |
| identity | .08 | 18 (0.75, 15), −1.7% [−57.5, +52.5] | 1 (0.08, 12) | 0 |
| logistic | 0 | 169 (7.0, 4), +0.2% [−11.4, +13.7] | 0 | 0 |
| logistic | .02 | 71 (3.0, 5), −5.4% [−30.3, +18.3] | 0 | 0 |
| logistic | .05 | 21 (0.9, 11), +0.1% [−62.4, +45.8] | 0 | 0 |
| logistic | .08 | 7 (0.3, 18), +34.3% [−62.4, +99.5] | 0 | 0 |
| GBM | 0 | 212 (8.8, 3), −9.8% [−19.4, +0.3] | 4 (0.3, 9), −11.0% | 0 |
| GBM | .02 | 122 (5.1, 5), −11.1% [−24.5, +2.3] | 2 (0.15, 11) | 0 |
| GBM | .05 | 51 (2.1, 10), −7.9% [−30.4, +11.8] | 1 (0.08, 12) | 0 |
| GBM | .08 | 16 (0.7, 16), −18.5% [−74.0, +23.0] | 0 | 0 |

On fallback legs no class has a lower bound above zero, at any threshold, in any league. The largest point estimate on more than 20 legs is WNBA's served-path fallback at t = 0: +25.0% on 29 legs, lower bound −3.9%. MLB's served-path fallback at the same threshold is −26.0% on 62 legs.

## Reality checks

- **Regime.**
  - The window is 26 eval days, 93% MLB, with five NFL dates, playoff WNBA and preseason NHL. NBA is absent.
  - The kill rests mainly on MLB.
  - The November re-run will be NFL, NBA and NHL. It is a bet that this transfers.
- **Power.** The logistic ablation's p = 0.072 is close. With more days a 0.0004-nat effect could become "significant". It would still not be *material*: about 13% of what the layer gains over served p, with no decision-level effect (every with-p minus no-p selection ROI CI covers 0) and a negative effect on the tail.
- **Proxies.** Decision time is the last exact-line platform poll. On 36% of unquoted legs the persisted price differs from that poll, so some features may post-date the actual decision by up to one prophecize cycle. That biases toward the market features, not against the model, and it is the main reason for the I2 re-run. Quote class is a `Market Projection` proxy.
- **Clusters.** 26 day-clusters (5 for NFL) is the regime where percentile cluster bootstraps can over-reject (Cameron, Gelbach & Miller 2008). That makes the Tier-B failures robust. The A2/A3 passes may be slightly optimistic, but A3's effect (10 pp) is far outside any plausible size distortion.
- **Payout convention.** Grading uses the 1.78 Underdog per-pick convention. At 1.81–1.87 nothing passes (Finding 4).
- **Build cost if reopened.** This is an engineering project, not research: about 250 lines in `prediction/trust.py`, a nightly refit in `nightly.py`, four consumer edits, golden tests and a D6 A/B. It is not worth any of that while K2 fails.

## What was tried and failed

| Attempt | Result |
|---|---|
| Unconstrained L2 logistic (pre-registered form) | Ineligible: negative served-p slope on up to 102 of 155 indicator combinations per week. Out of sample it is identical to the constrained fit |
| Monotone L2 logistic, class (ii) | A2 ✓, A3 ✓, A1 ✓ only at t ≥ .05 (n ≤ 92), and ✗ everywhere under the plan's ≤ 3 pp reading; K2 ✗ (p = .072); Tier B ✗ |
| Monotone GBM, class (iii) | A2 ✓, A3 ✓, A1 ✗ at every t (own-tail gap +7 to +19 pp); K2 ✗; Tier B ✗; ROI negative at every t |
| Minimal market anchor (S5-like; diagnostic) | p adds +0.00035 nats (p = .023) overall, −0.0022 on the tail; selects 36 legs in 26 days |
| Pooled cross-league transfer (rejected no-fit option) | Worse than served p on NFL W1 and WNBA W2; NFL W2 selections −17% / −20% |
| Post-hoc lp × version-age | No out-of-sample gain (−0.00008 nats); unstable sign |
| Void-excluded grading (MLB ≤ 09-16) | p's contribution turns significantly negative |
| Underdog payout 1.81 / 1.87 | No pass at any threshold |
| Leave 2026-09-23 out | Tail ROI falls from +26.5% to +15.4% (t = .08), from +5.1% to −2.3% (t = .05) |

## Open questions / caveats

1. **Confirmatory re-run, mid-November** (pre-registered now; same `prereg.py`, same 12-test Holm family). What it will add:
   - `Scored At`: exact decision time, replacing the last-poll proxy.
   - `Quote Source/Authenticity/Books/Line/Observed At`: exact quote class and staleness, plus a direct test of the stale-single-book failure in Finding 4.
   - `Consensus Line`: an exact w-free tail centre.
   - `Payout Over/Under`: an exact devig on quoted legs and the argmax-EV comparator.
   - About 6 weeks of NFL, NHL's regular season, and NBA once it clears the no-fit rule.
   - One new pre-registered interaction, lp × version age (Finding 7).
   - `Model Weight` stays out of the inputs.
2. **Reopen trigger** (both must hold in the confirmatory window):
   - (a) K2 passes for a pre-registered class: Δ log-loss of the ablation > 0 at one-sided 5%, day-clustered.
   - (b) The live λ_sel (the I1d edge-captured column) has a trailing-30-day CI lower bound above 0.
   Until then this lever stays dead, and the selection fix lives in R3/I6. No `prediction/trust.py` spec is given, because the brief asks for one only if the verdict is not KILL. If the trigger fires, the spec is the plan's I4a sketch, with three additions: `r1/prereg.py` as the frozen feature and class definition, abstain as the no-fit default, and per-version regime handling (Finding 7).
3. **Is the version-age decay real?** Two retrains only. It matters for the weekly `meditate` cadence. R3 should test it on the test sets (λ by days since train date), not on live legs alone.
4. **The stale-quote failure mode.** One 19:30 UTC sportsbook slot on 09-23 drove a quarter of the 0.05 tail. Whether the Odds API returns pre-lineup or placeholder prices for scratched players is a data-quality question for the book-quote lane.
5. **DNP grading** (Finding 11). Verify Underdog's MLB rule, then decide whether `annotate_offer_outcomes` should void non-starters on Sleeper. That is a Receipts/realized change, so the owner decides.
6. **Fallback class in the confirmatory run.** Once I2's `Quote Source` persists, the class label becomes exact. The November re-run should report fallback separately in the ablation (where it is undefined) and in the frontier. WNBA (about 75% fallback in its playoffs) is out of season by then. The live fallback population will be whatever NBA/NHL/NFL cells lack a pickle.

## Bibliography

| # | Source | Identifier |
|---|---|---|
| T1 | Smith, J. E., Winkler, R. L. (2006). The optimizer's curse: skepticism and postdecision surprise in decision analysis. *Management Science* 52(3), 311–322. | DOI 10.1287/mnsc.1050.0451 |
| T2 | Dawid, A. P. (1994). Selection paradoxes of Bayesian inference. IMS Lecture Notes–Monograph Series 24, 211–220. | IMS LNMS vol. 24 (no DOI confirmed) |
| T3 | Senn, S. (2008). A note concerning a selection "paradox" of Dawid's. *The American Statistician* 62(3), 206–210. | DOI 10.1198/000313008X331530 |
| T4 | Rosenbaum, P. R., Rubin, D. B. (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika* 70(1), 41–55. | DOI 10.1093/biomet/70.1.41 |
| T5 | Heckman, J. J. (1979). Sample selection bias as a specification error. *Econometrica* 47(1), 153–161. | DOI 10.2307/1912352 |
| T6 | Harvey, D. I., Leybourne, S. J., Newbold, P. (1998). Tests for forecast encompassing. *Journal of Business & Economic Statistics* 16(2), 254–259. | DOI 10.1080/07350015.1998.10524759 |
| T7 | Clements, M. P., Harvey, D. I. (2010). Forecast encompassing tests and probability forecasts. *Journal of Applied Econometrics* 25(6), 1028–1062. | DOI 10.1002/jae.1097 |
| T8 | Diebold, F. X., Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics* 13(3), 253–263. | DOI 10.1080/07350015.1995.10524599 |
| T9 | Giacomini, R., White, H. (2006). Tests of conditional predictive ability. *Econometrica* 74(6), 1545–1578. | DOI 10.1111/j.1468-0262.2006.00718.x |
| T10 | Gneiting, T., Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA* 102(477), 359–378. | DOI 10.1198/016214506000001437 |
| T11 | Cameron, A. C., Gelbach, J. B., Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. *Review of Economics and Statistics* 90(3), 414–427. | DOI 10.1162/rest.90.3.414 |
| T12 | Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics* 6(2), 65–70. | JSTOR 4615733 |
| T13 | Andrews, I., Kitagawa, T., McCloskey, A. (2024). Inference on winners. *Quarterly Journal of Economics* 139(1), 305–358. | DOI 10.1093/qje/qjad043 |
| T14 | Ke, G. et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. *Advances in NeurIPS* 30 (monotone constraints per the `lightgbm` package). | proceedings.neurips.cc/paper/2017 (6449f44a…) |
| T15 | Niculescu-Mizil, A., Caruana, R. (2005). Predicting good probabilities with supervised learning. *ICML '05*, 625–632. | DOI 10.1145/1102351.1102430 |
| T16 | Chow, C. K. (1970). On optimum recognition error and reject tradeoff. *IEEE Transactions on Information Theory* 16(1), 41–46. | DOI 10.1109/TIT.1970.1054406 |
| T17 | Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., Bouchachia, A. (2014). A survey on concept drift adaptation. *ACM Computing Surveys* 46(4), 44. | DOI 10.1145/2523813 |
| T18 | Snowberg, E., Wolfers, J. (2010). Explaining the favorite–long shot bias: is it risk-love or misperceptions? *Journal of Political Economy* 118(4), 723–746. | DOI 10.1086/655844 |
| T19 | Spann, M., Skiera, B. (2009). Sports forecasting: a comparison of the forecast accuracy of prediction markets, betting odds and tipsters. *Journal of Forecasting* 28(1), 55–72. | DOI 10.1002/for.1091 |
| T20 | Štrumbelj, E. (2014). On determining probability forecasts from betting odds. *International Journal of Forecasting* 30(4), 934–943. | DOI 10.1016/j.ijforecast.2014.02.008 |
| T21 | Kaunitz, L., Zhong, S., Kreiner, J. (2017). Beating the bookies with their own numbers — and how the online sports betting market is rigged. | arXiv:1710.02824 |
| T22 | Goto, K., Takeishi, N., Yairi, T. (2026). Forecast sports outcomes under efficient market hypothesis: odds-only and generalised linear models (preprint, not peer-reviewed). | arXiv:2604.17194 |
| T23 | Levitt, S. D. (2004). Why are gambling markets organised so differently from financial markets? *Economic Journal* 114(495), 223–246. | DOI 10.1111/j.1468-0297.2004.00207.x |
| T24 | Baker, R. D., McHale, I. G. (2013). Optimal betting under parameter uncertainty: improving the Kelly criterion. *Decision Analysis* 10(3), 189–199. | DOI 10.1287/deca.2013.0271 |
| T25 | Sleeper Player Picks rules (vendor documentation; DNP: "Batter must start and record a plate appearance"). | support.sleeper.com/en/articles/9047931 |

**In-repo prior art engaged:**
- Main-session §6 measurement, 2026-10-04 (`scratchpad/s6_before_after.md`): all-time recommended cohort by quote class. Its fallback overstatement (read .670, hit .492) is addressed in Finding 12.
- `docs/archive/researcher_selection_shrink.md`: the killed forms, the λ estimator, S5, the era boundary.
- Plan file: R1, I2, I4, decisions 3–4.
- `docs/handoffs/model_improvement_track.md` §6.10 and §8.2.
- Memory notes: `[[no-evidence-full-trust-pattern]]`, `[[selection_shrink_toward_book_kill]]`, `[[proxy_goodhart_under_search]]`, `[[blend_weight_book_riding_selection]]`, `[[dfs_pickem_lines_are_real_quotes]]`.

**Artifacts** (scratchpad `r1/`):
- `settled_postfix.parquet`, `arch_feats.parquet`, `model_frame.parquet`, `walk_oos.parquet`, `eval_rows.parquet`.
- `sel_table.csv`, `frontier.csv`, `frontier_by_quote.csv`, `frontier_fallback_by_league.csv`, `walk_logit_coefs.csv`, `walk_notes.csv`, `mlb_pa_join.parquet`.
- Code read: `realized.py`, `prediction/offer_records.py`, `prediction/scoring.py`, `prediction/cli.py`, `helpers/archive.py`, `history_schema.py`, `strategies/kelly.py`.
