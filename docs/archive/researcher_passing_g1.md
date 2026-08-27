# In-repo research brief — NFL passing yards, the Gate-1 wall

**Question:** find a *general*, board-sweepable lever that ships the last unshipped NFL cell
(passing yards, blocked solely by Gate 1) with no gate loosening, no market-specific tailoring,
and no decay as data accrues. Rank by cost-to-ship; adjudicate small-n-safe recalibration,
floor-weight semantics, data accrual, and the owner's continuous-hurdle suggestion.

**Date:** 2026-08-26 · branch `devel` · read-only. Nothing under `src/`, `data/config/`, or
`docs/` was written; no model was trained. All numbers below are re-derived with the repo's own
`training.scorecard` functions on artifacts already on disk. Working scripts:
`/tmp/claude-1000/-home-trevor-Sportstradamus/e42aed04-884a-4031-a139-0204cffabe07/scratchpad/g1/`.

---

## TL;DR

- **The dispatch's central premise is wrong in a way that changes the answer.** The best corner is
  *not* "95% book." At the nominal `w = 0.05` floor the served probability retains **46.1%** of the
  standalone model's departure from the book (`rms|p − p_book|` 0.098 vs the standalone's 0.213).
  `fused_loc`'s SkewNormal branch is a **precision-weighted** blend, so `w_eff = 1/(1 + ((1−w)/w)·
  (σ_m/σ_b)²)`; a row where the GBDT emits a small σ seizes the blend regardless of `w`. The grid
  floor cannot express what `crps_1se` is asking for.
- **Gate 1 is a disagreement-radius test at fixed n.** Because `d_i = (p_m−p_b)(p_m+p_b−2y)` and
  `|p_m+p_b−2y| ≈ 1` near 0.5, `sd(d) ≈ rms|p_m − p_b|`, so the gate admits at most
  **`rms|p_model − p_book| ≲ δ√n/1.96`** even for a *perfectly calibrated* model. At n = 325 that
  budget is **0.046**. The best corner sits at 0.098 — 2.1× over. The 2026-07-20 full-HPO artifact
  that PASSED sat at 0.041 — just inside. This is one number that explains every result in the file
  and it is a general, cross-league fact (n = 2000 ⇒ budget 0.114; NFL is hard because n is small).
- **Not a recalibration-error wall.** Under a truth = book Monte-Carlo at the pipeline's own fit size
  (n_fit = 266), an oracle (zero-estimation-error) map gives `E[ci_hi] = 0.0026`; the shipped
  unregularized 2-param Platt gives `0.0069`. So the estimation-error term is worth **+0.0043 of
  ci_hi** — real, and recoverable. But shrinking the map *toward identity* is the wrong target and
  measurably hurts (λ=20 → ci_hi 0.0150 vs 0.0065). **The transferable DOF is the slope; the
  non-transferable DOF is the intercept.**
- **Highest-value general lever: penalize the Platt intercept, select the penalty by out-of-fold
  log-loss.** New PROB_STAGE slug (λ=0 recovers today's `prob_recal_platt` exactly, so it is a
  superset). Measured across **59 cells** with a usable dump under the pipeline's own 5-fold
  player-disjoint cross-fit: pass-rate 81.4% → **86.4%**, median `ci_hi` 0.00140 → 0.00138,
  **0 cells regressed**, 3 flipped KILL→PASS (`NFL passing yards`, `NFL completions`, `WNBA AST`).
  On the small-n cohort (n<600, 16 cells) median `ci_hi` **halves**, 0.00484 → 0.00255, pass 50% →
  62.5%. Passing yards: **0.0065 → 0.0027**.
- **But every route to a pass on this cell is a route to looking more like the book, and the owner
  must ship it knowing that.** The model has *no measurable incremental signal* here: the pooled
  log-odds weight on the model is `ŝ = 0.035` (LR = 0.22, p ≈ 0.32; player-clustered 95% CI
  [0.000, 0.227] with 33% of resamples at exactly 0), OOF log-loss gain **+0.00034 nats/row**,
  BSS +0.0016. `kelly_shrinkage = clip(BSS,0,1) ≈ 0.002` ⇒ the cell would ship and essentially never
  stake. That is a breadth counter, not money. Gate 1 is a non-inferiority test and this is textbook
  **assay sensitivity** (ICH E10 [86]).
- **Cheapest action first, and it is nearly free:** §6.8 records a 2026-07-20 **isolated full-HPO +
  independent-holdout six-gate PASS** for this exact cell (g1 −0.0010, row CI [−0.0051, 0.0032],
  clustered 0.0033, w = 0.1093). I re-scored its on-disk artifact
  (`data/test_sets/NFL_passing-yards.csv`, n = 369): **ci_hi +0.00321, PASS, confirmed today**. The
  recipe and command are recorded verbatim; the quarantined pickle is gone from `/tmp` but one
  `meditate` reproduces it. Several stage changes have landed since (post-fusion mean corrector,
  joint `(c,s)`, ±8 skew bounds), so it must be re-earned — but **re-run before you build.**
- **Owner's continuous-hurdle idea: DEAD, plainly.** `P(Result = 0) = 0.128%` (3 rows of 2342);
  `P(<10) = 0.30%`. There is no zero atom, the hurdle parameter is not identified, and g4/g5 already
  pass. No g1 mechanism exists. Redirect the build.
- **Wait-for-data is honest but slow and only ~coin-flip.** The scored frame grows **~80 rows per
  NFL season (~4.6 per in-season week), not 30/week** — 30/week is the *matrix* rate; 70% goes to
  train and half the remainder to the fit split. P(pass) ≈ 43% now, 49% after one season, 63% after
  five. The bottleneck is uncertainty in the true μ_d, not half-width shrink.

---

## Key findings

### 1. Reproduction: the board numbers are real, and the frame matters

Re-scoring the sweep's own dump with `scorecard._gate1_brier_ci` reproduces the board exactly.

| frame | n priced | over-rate | `rms\|p−p_book\|` | g1 mean | g1 iid ci_hi | clustered ci_hi | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| board `deterministic/ratio_meanyr` (cross-fit validation) | 325 | 0.5477 | **0.098** | +0.00697 | **+0.01900** | +0.02101 | KILL |
| production `test_sets/NFL_passing-yards.csv` (true holdout, 2026-07-20 artifact) | 369 | 0.4986 | **0.041** | −0.00098 | **+0.00321** | +0.00329 | **PASS** |

Board row 2805 (`ratio_meanyr crps direct nll none`) reports `g1_brier_diff_ci_hi = 0.0190` and
`standalone = 0.0589`; my re-score returns 0.01900 and 0.05890. The harness is not the problem.

Two structural facts the dispatch did not state and that the plan should carry:

- **The board runs holdout-blind.** `model_strategy/sweep.py` passes `--holdout-blind`, which sets
  `eval_mask = validation_mask` (`pipeline.py:1150`) and routes scoring through
  `_step_crossfit_calibrate_and_serve` — a **5-fold, player-disjoint** cross-fit
  (`_CALIBRATION_FOLDS = 5`, `_calibration_folds` hashes `Player`). Every calibrator (blend weight,
  temperature, post-hoc) is fit on ~266 rows and applied to ~67. The board number is honest but it is
  **not** the frame a full-HPO confirm scores.
- **The ship gate reads the iid bootstrap.** `apply_thresholds` keys on `g1_brier_diff_ci_hi`;
  `g1_clustered_ci_hi` is reported-only. (Already noted in
  `docs/archive/researcher_blend_weight_slug.md`; re-confirmed.)

The two split halves of the *current* matrix have over-rates **0.5494 (validation, n=324)** and
**0.4639 (holdout, n=360)** — a 2.2 SE gap on a market whose true over-rate is ~0.50. Hold that
number; §4 turns it into the load-bearing mechanism.

### 2. Gate 1 is an admissible-disagreement-radius test, and this is the whole wall

`d_i = (p_m − y)² − (p_b − y)² = (p_m − p_b)(p_m + p_b − 2y)`. On a market where both
probabilities live near 0.5, `|p_m + p_b − 2y| ≈ 1`, so `sd(d) ≈ rms|p_m − p_b|` and the bootstrap
half-width is `≈ 1.96·rms|p_m − p_b|/√n`. Setting that equal to the margin δ = 0.005 gives, for a
model whose point estimate is a dead tie (`d̄ = 0`):

> **`rms|p_model − p_book| ≲ δ√n / 1.96`**

Measured, on the same 325 rows (`expD.py`, `expA.py`):

| served probability | `rms\|p−p_book\|` | `sd(d)` | g1 mean | g1 ci_hi |
|---|---:|---:|---:|---:|
| book itself | 0.000 | 0.000 | 0 | 0 |
| **budget at n = 325** | **0.046** | 0.046 | — | 0.005 |
| logit-shrink to book, s = 0.10 | 0.035 | 0.035 | +0.00030 | +0.00434 |
| logit-shrink to book, s = 0.15 | 0.049 | 0.049 | +0.00110 | +0.00675 |
| naked fused (board corner) | **0.098** | 0.104 | +0.00697 | +0.01900 |
| raw blend before `(c,s)` calibration | 0.268 | — | +0.03820 | +0.06675 |
| standalone model | 0.213 | 0.219 | +0.03472 | +0.05890 |

Budget by n: 325 → 0.046; 500 → 0.057; 800 → 0.072; 1200 → 0.088; 2000 → 0.114. This is the precise
sense in which "NFL is the hard league" (§3.2) — not a modelling deficiency, an **assay-power**
property of a fixed-δ non-inferiority test at small n [86][87][88]. It also explains why a cell can
be *more right* than the book and still fail: the gate charges for disagreement and only refunds it
through a sufficiently negative `d̄`. To pass at the board's `sd(d) = 0.104` this cell would need
`d̄ < −0.0070` — a Brier skill of +2.8% vs the book, ~18× what it has.

### 3. The nominal blend weight does not mean what the dispatch assumes

`helpers/distributions.py:924-953`, SkewNormal branch:

```
prec_m = 1/σ_m² ;  prec_b = 1/σ_b²
total_prec  = w·prec_m + (1−w)·prec_b
blended_loc = (w·loc_m·prec_m + (1−w)·loc_b·prec_b)/total_prec
blended_σ   = 1/√total_prec
```

so the realized weight on the model's location is `w_eff = 1/(1 + ((1−w)/w)(σ_m/σ_b)²)`. At
`w = 0.05`: `σ_m/σ_b = 1 → w_eff 0.05`; `0.5 → 0.174`; `0.35 → 0.301`; `0.10 → 0.840`.

Measured on the dump: the fused probability retains **46.1%** of the standalone's departure from the
book (`rms` 0.098 / 0.213); the OLS of `logit p_fused` on `logit p_sa` + `logit p_book` loads
`b_model = +0.588`; 18.8% of rows sit more than 5 probability points from the book and 3.4% more
than 30. **A corner labelled "95% book" is nothing of the sort.** Inverting the observed `w_eff ≈
0.33` gives `σ_m/σ_b ≈ 0.33`, so reaching a genuine 10% probability-space retention would need
`w ≈ 0.012` — a 4× extension below the current `_MODEL_WEIGHT_MIN = 0.05`.

**This is the mechanical reason `crps_1se` "wins" at the floor and still fails.** The 1-SE parsimony
rule (shipped, `/tmp/researcher_blend_weight_slug.md`) correctly detects a flat loss path and asks
for the minimum weight; the grid floor then hands it 33% of the model anyway.

### 4. Decomposing the 0.0056 — (a) model leg, (b) Platt map, (c) CI half-width

Answering the dispatch's question 2 directly. Because the exact best corner's dump was overwritten
by the sweep's next corner, I decompose the corner I *can* re-score (`ratio_meanyr crps direct nll
none`, board ci_hi 0.0190) and reconstruct the Platt step honestly under the pipeline's own 5-fold
player-disjoint cross-fit. The emulation reproduces the real corner's behaviour closely (naked
0.0120 → platt 0.0056 on the board corner; 0.0190 → 0.0065 here).

| stage | `rms\|p−p_book\|` | mean `d` | ci_hi | half-width |
|---|---:|---:|---:|---:|
| naked fused (post-`(c,s)`, post-temperature) | 0.098 | +0.00697 | 0.01899 | 0.01203 |
| + cross-fit Platt (2-param, `C=1e6`, today's slug) | 0.053 | **+0.00098** | **0.00652** | **0.00553** |
| + cross-fit **intercept-penalised** Platt | 0.022 | +0.00002 | **0.00197** | 0.00195 |

**Answer to (a)/(b)/(c): at the Platt corner, ~85% of `ci_hi` is CI half-width** (0.00553 of 0.00652)
and the point estimate is a dead tie (+0.00098). The same split almost certainly holds at the real
0.0056 corner. **But "it's all half-width" does not imply "only n can help"** — the half-width is
`1.96·rms|p−p_book|/√n`, so it is *equally* a function of how far the served probability departs
from the book. Shrinking the departure shrinks the half-width one-for-one. That is why the
intercept-penalised map takes the half-width from 0.0055 to 0.0020 at fixed n.

**The Platt map's estimation error, isolated** (`expF.py`, Monte-Carlo with truth = book,
n_fit = 266, n_apply = 325, 800 reps):

| map | E[mean d] | E[ci_hi] | P(pass) |
|---|---:|---:|---:|
| identity (naked) | +0.00977 | 0.02130 | 0.8% |
| **Platt 2-param (shipped)** | +0.00224 | **0.00687** | 42.6% |
| slope-only 1-param | +0.00073 | 0.00311 | 84.1% |
| slope-only, ridge λ=2 | +0.00061 | 0.00291 | 86.4% |
| oracle slope (zero estimation error) | −0.00027 | 0.00177 | 100.0% |
| oracle Platt (zero estimation error) | −0.00133 | 0.00256 | 99.9% |

So: **the recal-error term is worth +0.0043 of `ci_hi` (0.00687 − 0.00256), and ~88% of it is the
intercept.** The gate needs 0.0006 of it.

### 5. Why the intercept is the non-transferable degree of freedom — and the intercept-penalised slug

The fitted Platt slope on this cell is `a = 0.0358` with intercept `b = +0.192`. Three observations:

1. **The slope is doing the shrinking.** `a ≈ 0.035` maps `sd(logit p_fused) = 1.51` to `0.052` —
   exactly the book's `0.051`. `prob_recal_platt` on this cell *is* the book-collapse, reached by the
   slope. (Note: the repo's Platt is `expit(a·logit x + b)` on an already-probabilistic input, so
   `a=1, b=0` **is** the identity — Kull et al.'s "logistic calibration can uncalibrate a calibrated
   classifier" critique [89] applies to Platt-on-scores, not to this parameterisation.)
2. **The intercept estimates the over-rate, which is not learnable at this n.** Per-fold fitted
   intercepts: +0.117, +0.152, +0.297, +0.249, +0.140 — while the fit-fold over-rate ranges
   0.529–0.573 and the *apply*-fold over-rate ranges **0.431–0.607**. Fold 0's full Platt fits a
   **negative slope** (`a = −0.024`), i.e. it inverts the model. Pure noise.
3. **Quantified transfer cost.** Validation-half over-rate 0.5494, true-holdout-half over-rate
   0.4639. An intercept learned on one and applied to the other costs
   `(0.5494−0.5)(0.5494+0.5−2·0.4639) = +0.00601` of Brier — **larger than the entire gate margin.**
   This is a concrete, named mechanism for the board→confirm g1 gap the ledger measures
   (131 paired rows: median +0.0013, sd 0.0113, tail to +0.042).

The literature says exactly this. Guo et al. [90] show **temperature scaling — the single-parameter,
intercept-free variant of Platt scaling — beats full/vector scaling** precisely because the extra
parameters overfit a small held-out calibration set. In clinical prediction, Janssen et al. [91] and
Su et al. [92] find simpler updating (fewer re-estimated parameters) wins on small new samples;
Vergouwe et al. [93] give the formal answer — a **closed testing procedure that escalates the
updating method** (nothing → intercept only → intercept+slope → refit) **only when the new sample
carries enough evidence.** Copas [94] and Van Houwelingen & Le Cessie [95] are the shrinkage
ancestors: anticipate that a fitted map degrades out of sample and pre-shrink it. Riley et al. [96]
show n ≈ 300 is far below what is needed to pin calibration-in-the-large and slope jointly.

**The house-fit form of that idea:** a Platt map with an **L2 prior on the intercept only** (slope
free), λ chosen per cell by the **same 5-fold player-disjoint out-of-fold criterion the pipeline
already runs** in `posthoc.select_pit_recal` — but scored on **log-loss**, a proper scoring rule that
never touches the gate statistic or the book comparison. λ = 0 recovers `prob_recal_platt` bit-for-bit;
λ = ∞ is intercept-free (temperature-style). It self-tunes as data accrues: large-n cells already
prefer small λ.

**Measured across every deterministic dump on disk (59 cells, `expG.py`/`expH.py`):**

| policy | median ci_hi | pass-rate | mean ci_hi |
|---|---:|---:|---:|
| naked (no prob recal) | +0.00427 | 55.9% | +0.00289 |
| fixed λ = 0 (**today's `prob_recal_platt`**) | +0.00140 | 81.4% | −0.00236 |
| fixed λ = 20 | +0.00138 | 84.7% | −0.00277 |
| fixed λ = ∞ (intercept-free) | +0.00157 | 81.4% | −0.00202 |
| **OOF-log-loss-selected λ (proposed)** | **+0.00138** | **86.4%** | **−0.00303** |

- **Small-n cohort (n < 600, 16 cells):** median ci_hi 0.00484 → **0.00255**; pass 50.0% → **62.5%**.
- **Large-n cohort (n ≥ 600, 43 cells):** median 0.00126 → 0.00097; pass 93.0% → **95.3%**.
- **Flipped KILL→PASS:** `NFL passing yards` (0.00651 → 0.00275), `NFL completions`, `WNBA AST`.
- **Flipped PASS→KILL: none (0/59).**
- **Anti-Goodhart check** (per the `proxy_goodhart_under_search` memory): the CV criterion is
  log-loss, evaluated out-of-fold; it dominates every *fixed* λ on the gate statistic it never saw,
  and selects λ=∞ on 24/59 cells, an interior λ on 31/59, and λ=0 on only 4/59. The map is
  **not** a universal book-collapse: `rms|p−p_book|` after CV-λ ranges from 0.022
  (passing yards — rank 1 of 59, the *least* model-bearing cell on the board) up to 0.32
  (`NFL interceptions`, whose ci_hi is −0.052). It collapses only where there is nothing to carry.

### 6. What actually carries the Gate-1 statistic: nine degenerate rows

This is the finding I did not expect and it belongs in the plan.

The served probability on this cell is 90% inside `[0.456, 0.581]` — but **9 of 325 rows (2.8%) are
pinned at `P < 0.01` or `P > 0.99`**, several at exactly 0.0 / 1.0. Sorting by `|d_i|`:

| P | p_book | y | d_i | share of mean(d) |
|---:|---:|---:|---:|---:|
| 0.000001 | 0.4920 | 1 | +0.74192 | +0.00228 |
| 0.000001 | 0.4918 | 1 | +0.74176 | +0.00228 |
| 0.997217 | 0.5055 | 0 | +0.73891 | +0.00227 |

**Three rows carry 98.1% of the Gate-1 point estimate.** Five rows carry 73.5% of its variance; nine
carry 81.0%. One wrong degenerate row is worth **46% of the entire 0.005 gate margin**.

And **NFL passing yards has the highest rate of degenerate served probabilities of all 59 cells**
(2.77%; median across cells is 0.0000; only 22/59 have any, only 7/23 SkewNormal dumps).

The mechanism is §3's precision blend: 15 of 333 rows (4.5%) are served at **< 0.25× the book scale**
and 7 at **< 0.10×**, where `w_eff → 0.84+` and `blended_σ → σ_m/√w`. The predictive becomes a
near-point-mass and the over-probability saturates. A nominally-5% blend is being seized by an
over-confident GBDT σ on a handful of rows.

A downstream symmetric probability clamp is a blunt tool — clamping to `[0.25, 0.75]` still leaves
`ci_hi = 0.00856`, because a wrong row at 0.25 still costs +0.30. The fix belongs **upstream, on
`σ_m` before the precision blend**. Flooring the served scale at `κ × book scale` (a proxy for the
real fix, `expL.py`):

| κ | rows floored | extreme P | g1 mean | g1 ci_hi | g4 pit_ks (<0.0744) | cov50 | g5 debiased |
|---:|---:|---:|---:|---:|---:|---:|---:|
| — (today) | 0 | 2.77% | +0.00686 | 0.01896 | 0.0421 | 0.538 | +0.0320 |
| 0.30 | 15 | 1.54% | +0.00483 | 0.01578 | 0.0421 | 0.538 | +0.0301 |
| 0.50 | 15 | 0.92% | +0.00308 | 0.01315 | 0.0421 | 0.538 | +0.0188 |
| 0.80 | 18 | 0.92% | **+0.00226** | 0.01187 | 0.0421 | 0.541 | +0.0204 |

It removes **two-thirds of the g1 point estimate** and costs nothing on g4/g5 — but it does not move
the half-width, so it is **necessary-looking and insufficient alone**.

### 7. Hypothesis tested and refuted: the ±8 skew widening is not the culprit

The obvious suspect was the joint `(c, skew_cal)` calibrator — fit to minimise **Gate 4's own
PIT-KS** — plus the `_DISPERSION_SKEW_BOUNDS = (−8, 8)` widening that landed the same day. This
cell's board corner carries `skew_cal = −1.22`, and the shipped 2026-07-20 artifact was essentially
symmetric (`SN_Alpha ≈ +0.02`). Undoing the calibration on the same fitted model (`expJ.py`):

| variant | `rms\|p−p_book\|` | g1 mean | g1 ci_hi | g4 pit_ks |
|---|---:|---:|---:|---:|
| served (c = 1.319, s = −1.219) — today | 0.098 | +0.00686 | 0.01896 | **0.0421** |
| scale-only (c = 1.319, s = 0) | 0.225 | +0.01866 | 0.04262 | 0.2153 |
| skew-only (c = 1, s = −1.219) | 0.154 | +0.01698 | 0.03455 | 0.1048 |
| raw blend (c = 1, s = 0) | 0.268 | +0.03820 | 0.06675 | 0.2388 |

**Refuted.** The joint `(c,s)` fit improves g1 *and* g4 substantially; the raw blend is far worse on
both. The skew calibration is repairing a genuinely mis-shaped predictive, not causing the wall. Do
not touch the ±8 widening on this evidence.

### 8. Does the model carry any edge here at all? No — and this is the honest ceiling

Fitting the book-anchored pool `logit q = logit p_book + s·(logit p_fused − logit p_book)` by MLE
(`expD.py`):

- pooled `ŝ = 0.0351`; LR vs `s = 0` is **0.223**, p ≈ **0.32**;
- OOF log-loss gain **+0.00034 nats/row** (0.69470 → 0.69435);
- player-clustered bootstrap of `ŝ`: median 0.039, 95% CI **[0.000, 0.227]**, **33.3% of resamples
  return exactly 0**;
- per-fold cross-fit `s*`: 0.000, 0.020, 0.081, 0.049, 0.045;
- `brier_book = 0.25077`; a constant at this frame's own over-rate scores 0.24773.

The book on this cell is `p ∈ [0.4367, 0.5434]`, `sd = 0.0127` — a near-degenerate median quote, and
still unbeatable. **Any lever that ships this cell ships it by looking like the book.** That is the
classical **assay-sensitivity** failure of a non-inferiority design [86][87][88]: a trial against an
active control passes trivially when the experimental arm *is* the control. The project's existing
defences hold — a tie passes by doctrine (`docs/ship_gate.md`), `kelly_shrinkage = clip(BSS,0,1) ≈
0.002` sizes it to nothing, and the 14-day live Gate-2 soak is the real filter — but the owner should
book this as **coverage, not edge**, exactly as the predecessor brief concluded for `crps_1se`.

The distinction that keeps Candidate A legitimate: its λ is selected by an **out-of-fold proper
scoring rule**, never by `ci_hi`. That is the same line that killed the ci_hi-minimising weight
objective (assay-sensitivity leak), and it is *more* conservative than the existing precedent —
`select_pit_recal` already selects λ by out-of-fold **Gate-4 KS**, which is the gate's own functional.

### 9. Data accrual — the honest distribution, and a correction to the accrual rate

**The dispatch's ~30 rows/week is the matrix rate, not the scored-frame rate.** `_TRAIN_FRACTION =
0.7` and the held-out block is split ~50/50 by `Player+Date` hash, so a new row reaches the scored
frame with probability 0.15. The matrix holds 2342 rows over 4.4 seasons ≈ **530 QB rows/season**, so
the scored frame grows **≈ 80 rows/season ≈ 4.6 per in-season week** — a **6.5× slower** clock.

Taking the best corner's structure as `sd(d) ≈ 0.044`, `d̄ ≈ 0.0056 − 1.96·0.044/√325 = +0.00082`
(SE 0.00244), and holding the model fixed:

| true μ_d | n needed for ci_hi < 0.005 | extra NFL seasons |
|---:|---:|---:|
| −0.0010 | 207 | already |
| 0.0000 | 297 | already |
| +0.0005 | 367 | 0.5 |
| +0.0010 | 465 | 1.8 |
| +0.0020 | 826 | 6.3 |
| +0.0030 | 1859 | 19.3 |

The point estimate is not known to better than ±0.0024, so the honest object is the pass-probability
curve (μ_d ~ N(d̄, SE); realised d̄_n ~ N(μ_d, sd/√n)):

| extra seasons | n | P(ci_hi < 0.005) |
|---:|---:|---:|
| 0 | 325 | **43.1%** |
| 1 | 404 | 48.8% |
| 2 | 484 | 53.5% |
| 3 | 564 | 57.1% |
| 5 | 722 | 63.0% |
| 8 | 961 | 69.1% |
| 12 | 1279 | 74.0% |
| 20 | 1915 | 80.0% |

**Waiting is not a plan.** The curve is flat because the binding uncertainty is in μ_d, not in the
half-width; it asymptotes at `P(μ_d < 0.005) ≈ 96%` only after decades. One season of accrual buys
about **6 percentage points**. Note also that retraining on more data changes the model, so μ_d is
not actually fixed — the table is the requested "hold the point estimate" projection, not a forecast.

### 10. The continuous-hurdle question: no mechanism, close the lane

Measured on the full matrix (`expE.py`, n = 2342): `mean = 226.9`, `sd = 70.4`, **`P(Result = 0) =
0.128%` (3 rows)**, `P(<10) = 0.30%`, `P(<50) = 0.85%`, `P(<100) = 3.2%`; the three zeros are the
entire left atom. `P(Result < 0.25 × Line) = 0.85%`.

A two-part / hurdle model splits `P(Y = 0)` from `f(Y | Y > 0)`. With `π̂ = 0.0013` on n = 2342, the
zero-part logistic has ~3 events — the parameter is not identified in any useful sense (Riley's
events-per-variable arithmetic [96] puts the usable floor orders of magnitude higher), and the
positive-part fit is numerically the same fit as today's. There is no channel through which it
touches g1: the gate is decided by rows near the median line, and 90% of served probabilities already
sit in `[0.456, 0.581]`. This also matches the repo's own precedent — the B1 ZTNB refutation and the
§8.2 note that "positive-only KS ≈ full KS, so ZAGamma/hurdle would fix the wrong defect" for the
yards cells. **Verdict: DEAD. Redirect the build.** (The two zero-atom cells that shipped —
receiving/rushing TDs — have `P(0)` two to three orders of magnitude larger.)

### 11. Question 5 — the other small-n co-riding ideas, adjudicated

| idea | verdict | evidence |
|---|---|---|
| **Shrink the model leg toward the book inside the blend** (rather than recalibrating after) | **This is the right instinct, but it is already what `crps_1se` does — and the grid floor blocks it.** See §3: `w = 0.05` yields 46% probability-space retention. The actionable version is Candidate C (extend the grid floor), not a new objective. | §3; `_MODEL_WEIGHT_MIN = 0.05` |
| **Book-anchored logit pool with a fitted weight** (a prob-stage twin of the blend) | Works spectacularly (cross-fit ci_hi **0.0028**, PASS) — **and that is the warning.** The fitted `s → 0.035`, so it ships the book. Same family as the killed ci_hi-minimising objective in effect, if not in objective. **Do not build as a standalone slug**; Candidate A reaches the same place through a proper-score criterion and a map that cannot special-case the book. | §8, `expC.py` |
| **Ridge-penalise the Platt map toward *identity*** | **KILL — wrong shrinkage target.** λ=1 → 0.0064; λ=5 → 0.0071; λ=20 → 0.0150; λ=200 → 0.0189 (≈ naked). The map is doing real bias correction; shrinking it toward identity undoes it. | `expC.py` |
| **Beta calibration** (Kull et al. [89]) | **KILL at this n** — 3 parameters, cross-fit ci_hi 0.0074, worse than 2-param Platt (0.0065). More DOF is the wrong direction here. | `expC.py` |
| **Bagged / ensembled recalibration maps** (B = 200) | **No-op.** ci_hi 0.00639 vs Platt's 0.00652. Bagging a smooth, low-variance MLE returns approximately the MLE; there is no instability for it to average away. | `expC.py` |
| **Leave-one-out / cross-fit *fitting* of the recal map** | **Category confusion.** LOO/CV changes the *estimate of performance*, not the apply-time map, so it removes no apply-time variance. It is, however, exactly the right machinery for **selecting λ** — which is Candidate A. The pipeline already cross-fits the map itself under `--holdout-blind`. | §1; `posthoc.select_pit_recal` |
| **Isotonic prob-recal** (`prob_recal_isotonic`, already a slug) | Middling here — cross-fit 0.0132 vs Platt 0.0065. Consistent with the standing small-n result that isotonic over-fits below a few hundred points [89][97][98]. | `expC.py` |

---

## Recommendation — ranked by cost-to-ship, with pre-registered kill criteria

### 0. Re-run the recorded 2026-07-20 recipe before building anything · ~0.2 session

§6.8 records an **isolated full-HPO + independent-holdout six-gate PASS** for this exact cell, and I
re-scored its artifact today: `data/test_sets/NFL_passing-yards.csv`, n = 369, `g1 mean −0.00098`,
`iid ci_hi +0.00321`, `clustered +0.00329` — **PASS**, `rms|p−p_book| = 0.041`, inside the 0.046
budget. The command is recorded verbatim in the plan (SkewNormal / `ratio_meanyr` / dist CRPS /
blend NLL / `--hpo-selection loss` / `--sn-param direct` / no stabilization / no posthoc, landing at
`w = 0.1093`, temperature 1.154, dispersion 0.9908).

Note carefully: **the same corner is on today's board at `ci_hi = 0.0190`** (row 2805). The gap is
frame (cross-fit validation vs true holdout) plus every stage change since 2026-07-20 (post-fusion
mean corrector, joint `(c,s)`, ±8 skew bounds — per the `corner_verdicts_are_not_code_scoped`
memory). One `meditate` settles whether the pass survives the current code.

- **Projected:** binary. Either it reproduces (`ci_hi ≈ 0.003`, ship) or it does not.
- **Kill criterion:** if the fresh full-HPO run's `g1_brier_diff_ci_hi ≥ 0.005` on the true holdout,
  the 2026-07-20 evidence is superseded — record that in the plan and proceed to Candidate 1.

### 1. BUILD — `prob_recal_platt_cv`: intercept-penalised Platt, λ by out-of-fold log-loss · ~1 session

The general lever. A new `PROB_STAGE` slug in `training/posthoc.py`:

- Fit `logit q = a·logit p + b` by penalised MLE with an **L2 prior on `b` only** (slope free).
- Select λ per cell from a fixed grid — `(0, 1, 5, 20, 100, ∞)` — by **out-of-fold log-loss** under
  the *same* 5-fold player-disjoint folds `_calibration_folds` already builds, mirroring
  `select_pit_recal`'s structure exactly. Ties break toward larger λ (more shrinkage).
- Register in `model_strategy/specs.py` alongside `prob_recal_platt` / `prob_recal_isotonic`.
- λ = 0 is bit-identical to today's `prob_recal_platt`, so the pool strictly grows.
- Persist `(a, b, λ)` in the blob; `apply_posthoc`'s `platt` branch already applies `expit(a·logit x
  + b)` unchanged — **no inference-path change**, which keeps this inside §7.3's cheapest change class.

**Projected effect (measured, 59 cells, `expG.py`/`expH.py`):** NFL passing yards board `ci_hi`
0.00651 → **0.00275**. Repo-wide pass-rate 81.4% → 86.4%, **0/59 regressions**, 3 KILL→PASS. Small-n
cohort median `ci_hi` halves.

**Regime where it holds:** cells whose held-out fit split is a few hundred rows and whose over-rate
is therefore unlearnable. It is近 a no-op above n ≈ 2000, where the CV correctly selects small λ.

**Kill criterion (pre-register):** re-sweep NFL passing yards + the 16-cell small-n cohort with the
new slug. Revert if **either** (a) passing yards' best-corner board `ci_hi ≥ 0.0040` — a 20% margin
under the bar, chosen to absorb the ledger's median board→confirm g1 shift of +0.0013 — **or**
(b) any currently-shipped cell's best-corner `ci_hi` regresses by more than +0.0010.

### 2. BUILD (conditionally) — a book-relative scale floor in the SkewNormal precision blend · ~1 session + a full re-sweep

Fixes a real defect regardless of whether it ships this cell: 4.5% of rows are served at < 0.25× the
book scale, 2.8% saturate to 0/1, and **three such rows carry 98% of the g1 point estimate**. Floor
`σ_m` at `κ·σ_b` inside `fused_loc`'s SkewNormal branch before forming `prec_m`, so `w_eff` is capped
at `1/(1 + ((1−w)/w)κ²)`. One named constant.

**Projected:** g1 mean +0.00686 → +0.00226 at κ = 0.8 (`ci_hi` 0.0190 → 0.0119); g4 `pit_ks`
unchanged at 0.0421, `cov50` 0.538 → 0.541, g5 improves. **Insufficient alone** — pair with
Candidate 1.

**Reality check on cost:** `fused_loc` is shared by training *and* serving, so it is one edit site —
but it changes the served predictive of **every SkewNormal cell** (20+ shipped). That means a full
re-sweep plus supersession re-checks, and it is the reason this ranks below Candidate 1 despite
addressing the more fundamental defect. My test floors the *post-`(c,s)`* served scale as a proxy;
the real edit is upstream of the blend and will not reproduce these numbers exactly.

**Kill criterion:** if, at the chosen κ, **≥ 2 currently-shipped SkewNormal cells** lose any gate on
the re-sweep, revert. Also revert if κ must exceed 1.0 to move passing yards — a floor above the book's
own scale is not a guard, it is a rewrite of the blend.

### 3. CONSIDER (owner policy call) — extend the blend-weight grid below 0.05 · ~0.5 session

`_MODEL_WEIGHT_MIN = 0.05` yields 46% probability-space model retention (§3). Reaching a genuine 10%
needs `w ≈ 0.012`. A log-spaced grid down to ~0.01 would let `crps_1se` express what it is already
asking for.

**Risk, stated plainly:** the floor is the last structural backstop against a total book-collapse on
*every* cell. Lowering it broadens the assay-sensitivity exposure repo-wide, not just here. The 1-SE
rule is the guard, but it is a statistical guard, not a structural one. **This is an owner decision,
not a session edit.**

**Kill criterion:** if a re-sweep drives **> 25%** of currently-shipped cells to the new floor, the
old floor was load-bearing — restore it.

### 4. WAIT-FOR-DATA — legitimate, slow, ~coin-flip

P(pass) 43% today, 49% after the 2026 season, 63% after five. Accrual is ~80 scored rows/season. If
nothing is built, the honest expected pass date is **"sometime in the next five seasons, at even
odds after one."** Do not represent this as a plan.

### 5. DEAD-END — continuous hurdle / two-part for this cell

`P(Result = 0) = 0.128%`, three rows. No identified zero part, no g1 channel. **Close the lane.**

---

## Reality checks

- **This buys coverage, not money.** `ŝ = 0.035`, LR p ≈ 0.32, OOF log-loss gain 0.0003 nats/row,
  BSS +0.0016 ⇒ `kelly_shrinkage ≈ 0.002`. Shipping passing yards moves NFL 19/20 → 20/20 and stakes
  approximately nothing. If the goal is the breadth counter, say so; if the goal is EV, this cell is
  not where it is.
- **Candidate 1's repo-wide numbers come from a broad screen, not a controlled A/B.** The 59 dumps
  were written by different corners at different times under different code revisions, and some are
  true-holdout frames while the recent ones are cross-fit validation frames. The *direction* is
  robust (0/59 regressions, and the small-n/large-n split is exactly what theory predicts); the
  *magnitudes* will move under a clean re-sweep. Read direction, not magnitude — the
  `deterministic_ab_g4_oversell` memory applies.
- **The board is not the ship number.** The ledger's 131 paired confirm-vs-board g1 rows have median
  +0.0013, sd 0.0113, and a tail to +0.042. A board `ci_hi` of 0.00275 has real but not overwhelming
  headroom; a board `ci_hi` of 0.0045 would not. That is why Candidate 1's kill bar is 0.0040, not
  0.0050.
- **Candidate 1 does not create edge and cannot.** It removes an estimation-variance term worth
  +0.0043 of `ci_hi`. If the underlying `d̄` were materially positive it would not save the cell.
  Here `d̄` is a dead tie (+0.00098), which is why removing the variance is decisive.
- **What would make this recommendation wrong.** (i) If the OOF log-loss criterion turns out to
  select λ adversarially *under sweep pressure* across many corners — the `proxy_goodhart_under_search`
  failure mode. My check is static across 59 cells and one λ grid; it has not been validated with the
  sweep optimising over λ *and* 15 corners simultaneously. (ii) If the intercept is genuinely
  load-bearing on some cohort I did not sample (team-level markets with no `Player` column fall back
  to date-hashed folds and were largely excluded by the n ≥ 80 filter). (iii) If a future gate scores
  the *probability* leg against the *mean* leg for coherence — every `PROB_STAGE` slug already
  decouples `P` from `Blended_EV`, and 22 shipped cells carry that decoupling today, but a coherence
  gate would put all of them in scope at once.
- **§7.4 cross-league caveat, generalised.** The disagreement budget `δ√n/1.96` is the cleanest
  statement of why an NFL verdict does not transfer to NBA and vice versa: an NBA cell at n ≈ 2000
  has a budget of 0.114, 2.5× the NFL cell's 0.046. A lever that "fixes g1" on NFL may be measuring
  nothing on NBA, and an NBA g1 pass says nothing about whether the same recipe survives NFL n.

---

## Open questions / caveats to carry into the plan

1. **Does the OOF-log-loss λ selection survive optimisation pressure?** Validate the proxy *under
   search* (15 corners × 6 λ), not just statically, before trusting the 86.4% figure.
2. **Should the degenerate-served-probability rate become a reported diagnostic?**
   `frac(P < 0.01 or P > 0.99)` is currently invisible; it ranks passing yards 1/59 and it is the
   direct cause of its g1 point estimate. A one-column addition to `model_stats` would surface the
   `fused_loc` precision-seizure class across every SkewNormal cell (7/23 dumps affected).
3. **Is `_MODEL_WEIGHT_MIN = 0.05` the right object at all?** It is a precision weight, not a
   probability weight, and its realized meaning varies per cell with `σ_m/σ_b`. A weight expressed in
   probability space (or a reported `w_eff`) would make the sweep's blending axis interpretable.
4. **The board→confirm g1 gap now has a named mechanism** (the Platt intercept transferring across a
   val/holdout over-rate gap of 0.549 → 0.464). Worth checking whether the 14 shipped
   `prob_recal_platt` cells show a larger measured confirm-vs-board g1 shift than the
   `posthoc: none` cells — a 20-minute ledger query that would either confirm or refute the mechanism
   at scale.
5. **Does the temperature stage still earn its place?** On this cell `T ≈ 1.02` — effectively a
   no-op — and `prob_recal_platt` subsumes it exactly (`expit(a·logit(expit(logit p /T)) + b) =
   expit((a/T)·logit p + b)`). The `bounds=(1.0, 10.0)` constraint means it can only soften. Two
   stacked calibrators fit on the same 266 rows is one more DOF than the analysis in §5 accounts for.
6. **`P_standalone` is a strong routing diagnostic that nothing routes on.** Across passing-yards
   corners it ranges 0.044 (`centered_additive_eb_meanyr_k10`) to 0.282 (`ratio_projvol`) — a 6×
   spread in standalone model quality that the board records and the slack ranking ignores.

---

## Bibliography

Existing repo citations built on rather than re-derived: [47] Efron & Morris (shrinkage), [48]
Roelofs et al. (ECE debias), [54] Genest & Zidek (opinion pools), [56] Ranjan & Gneiting (pooled
calibration), [71] López de Prado (CPCV/embargo), [75] Gneiting/Balabdaoui/Raftery (calibration and
sharpness). New identifiers below continue the numbering in `docs/operation_ship_references.md`
(highest existing = [85]).

| # | Source | Identifier | Used for |
|---|---|---|---|
| [86] | ICH E10, *Choice of Control Group and Related Issues in Clinical Trials* (2000) | ICH Harmonised Tripartite Guideline E10 | Assay sensitivity + the constancy assumption — a non-inferiority test against a control passes vacuously when the experimental arm *is* the control |
| [87] | Fleming, T. R. (2008). Current issues in non-inferiority trials. *Statistics in Medicine* 27(3), 317–332 | DOI 10.1002/sim.2855 | Margin choice, assay sensitivity, why a fixed δ is an evidence bar and not a quality bar |
| [88] | D'Agostino, R. B., Massaro, J. M., Sullivan, L. M. (2003). Non-inferiority trials: design concepts and issues. *Statistics in Medicine* 22(2), 169–186 | DOI 10.1002/sim.1425 | The `ci_hi < δ` construction and its power/n coupling — the source of the disagreement-radius result |
| [89] | Kull, M., Silva Filho, T. M., Flach, P. (2017). Beyond sigmoids: how to obtain well-calibrated probabilities from binary classifiers with beta calibration. *Electronic Journal of Statistics* 11(2), 5052–5080 | DOI 10.1214/17-EJS1338SI | Beta calibration (tested, killed at this n); the identity-not-in-family critique and why it does *not* apply to the repo's logit-parameterised Platt |
| [90] | Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML* | arXiv:1706.04599 | Temperature scaling = the single-parameter, intercept-free Platt variant; it beats vector/matrix scaling precisely because extra parameters overfit a small calibration set |
| [91] | Janssen, K. J. M., Moons, K. G. M., Kalkman, C. J., Grobbee, D. E., Vergouwe, Y. (2008). Updating methods improved the performance of a clinical prediction model in new patients. *J. Clinical Epidemiology* 61(1), 76–86 | DOI 10.1016/j.jclinepi.2007.04.018 | "Recalibration in the large" (intercept) vs logistic recalibration (intercept+slope) — the canonical decomposition my §5 measurement uses |
| [92] | Su, T.-L., Jaki, T., Hickey, G. L., Buchan, I., Sperrin, M. (2018). A review of statistical updating methods for clinical prediction models. *Statistical Methods in Medical Research* 27(1), 185–197 | DOI 10.1177/0962280215626466 | Survey: simpler updating dominates on small new samples |
| [93] | Vergouwe, Y., Nieboer, D., Oostenbrink, R., et al. (2017). A closed testing procedure to select an appropriate method for updating prediction models. *Statistics in Medicine* 36(28), 4529–4539 | DOI 10.1002/sim.7179 · PMID 27891652 | The formal escalate-only-with-evidence procedure that Candidate 1's λ-by-CV implements in shrinkage form |
| [94] | Copas, J. B. (1983). Regression, prediction and shrinkage (with discussion). *JRSS-B* 45(3), 311–354 | DOI 10.1111/j.2517-6161.1983.tb01258.x | Anticipate out-of-sample degradation and pre-shrink the fitted map — the shrinkage ancestor |
| [95] | Van Houwelingen, J. C., Le Cessie, S. (1990). Predictive value of statistical models. *Statistics in Medicine* 9(11), 1303–1325 | DOI 10.1002/sim.4780091109 | Heuristic shrinkage factor; the `p/n` optimism scale my Monte-Carlo measures empirically |
| [96] | Riley, R. D., Debray, T. P. A., Collins, G. S., et al. (2021). Minimum sample size for external validation of a clinical prediction model with a binary outcome. *Statistics in Medicine* 40(19), 4230–4251 | DOI 10.1002/sim.9025 | n required to pin calibration-in-the-large and slope separately — why n ≈ 300 cannot support both, and why the hurdle's 3-event zero part is unidentified |
| [97] | Niculescu-Mizil, A., Caruana, R. (2005). Predicting good probabilities with supervised learning. *ICML*, 625–632 | DOI 10.1145/1102351.1102430 | Platt vs isotonic crossover as a function of calibration-set size — supports the isotonic verdict at n ≈ 325 |
| [98] | Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In *Advances in Large Margin Classifiers*, MIT Press, 61–74 | — | The original 2-parameter sigmoid map; the slug's ancestor |
| [99] | Kumar, A., Liang, P., Ma, T. (2019). Verified uncertainty calibration. *NeurIPS* | arXiv:1909.10155 | Sample complexity of calibration; why measured calibration error at small n is dominated by estimation variance |
| [100] | Gupta, C., Ramdas, A. (2021). Distribution-free calibration guarantees for histogram binning without sample splitting. *ICML* | arXiv:2105.04656 | Distribution-free guarantees and the cost of sample splitting — the backdrop for cross-fit λ selection |
| [101] | Vaicenavicius, J., Widmann, D., Andersson, C., Lindsten, F., Roll, J., Schön, T. B. (2019). Evaluating model calibration in classification. *AISTATS*, PMLR 89:3459–3467 | arXiv:1902.06977 | Finite-sample bias/variance of calibration assessment — companion to the repo's existing [48] |
| [102] | van der Laan, L., Ulloa-Pérez, E., Carone, M., Luedtke, A. (2023). Causal isotonic calibration for heterogeneous treatment effects. *ICML*, PMLR 202 | arXiv:2306.05986 | **Cross-calibration**: cross-fitted calibrators using all data with no hold-out set — the formal justification for the pipeline's `--holdout-blind` cross-fit and for Candidate 1's λ selection |
| [103] | Smith, J., Wallis, K. F. (2009). A simple explanation of the forecast combination puzzle. *Oxford Bulletin of Economics and Statistics* 71(3), 331–355 | DOI 10.1111/j.1468-0084.2008.00541.x | Estimation error in combination weights — already cited in the `crps_1se` brief; re-anchors §3's "the grid floor cannot express what the 1-SE rule asks for" |
| [104] | Claeskens, G., Magnus, J. R., Vasnev, A. L., Wang, W. (2016). The forecast combination puzzle: a simple theoretical explanation. *International J. Forecasting* 32(3), 754–762 | DOI 10.1016/j.ijforecast.2015.12.005 | Why an equal/reference-anchored combination beats an optimally-estimated one at small n |
| [105] | Diebold, F. X., Pauly, P. (1990). The use of prior information in forecast combination. *International J. Forecasting* 6(4), 503–508 | DOI 10.1016/0169-2070(90)90028-A | Bayesian shrinkage of estimated combination weights toward a prior — the book-as-prior framing |
| [106] | Diebold, F. X., Shin, M. (2019). Machine learning for regularized survey forecast combination: partially-egalitarian LASSO and its derivatives. *International J. Forecasting* 35(4), 1679–1691 | DOI 10.1016/j.ijforecast.2018.09.006 | Modern regularised combination — penalise *toward the reference*, not toward zero; the direct analogue of §5's "wrong shrinkage target" result |
| [107] | Van Calster, B., McLernon, D. J., van Smeden, M., Wynants, L., Steyerberg, E. W. (2019). Calibration: the Achilles heel of predictive analytics. *BMC Medicine* 17, 230 | DOI 10.1186/s12916-019-1466-7 | The calibration hierarchy (mean / weak / moderate / strong); places "intercept vs slope" in a standard taxonomy |
| [108] | Riley, R. D., Collins, G. S. (2023). Stability of clinical prediction models developed using statistical or machine learning methods. *Biometrical Journal* 65(8), 2200302 | DOI 10.1002/bimj.202200302 | Instability of fitted models and their calibration maps at small n — the per-fold sign-flipping slope in §5 is a textbook instance |
