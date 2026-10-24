# In-repo research brief — count-branch blend **location**: is the log-opinion pool the lever?

**Question this brief answers.** Gate 6's CITL leg (`Σ served_mean / Σ Result`, upper CI ≥ 0.97)
blocks NFL tds. Is the count branch's geometric (logarithmic-opinion) pooling of the location the
cause, is the `_MODEL_WEIGHT_MAX = 0.9` cap the cause, or is the book leg the cause — and what is
the **market-agnostic** change that closes it without any currently-shipping cell losing `ship`?

**Date:** 2026-08-25. **Discharges:** the CLAUDE.md research-first gate and
`model_improvement_track.md` §8.2 for a §6.5 blend-structure change.
**Status: read-only.** Nothing under `src/` was written. Every number below is either re-derived
from the operator's arm artifacts in `/tmp/scratch/count-mean/` or computed on
`src/sportstradamus/data/test_sets/*.csv` + `src/sportstradamus/data/models/*.mdl` with the
production `training.scorecard` gate code. The screen harness is preserved at
`/tmp/researcher_count_blend_location_screen/`.

---

## TL;DR

1. **Do not reopen §6.5. The pool-structure NO-GO stands, and this cohort strengthens it.**
   Swapping the count location pool geometric → arithmetic moves the all-row served CITL *away*
   from 1.0 on **19 of 29** pooled count cells and costs **NHL hits** its `ship`. The geometric
   pool's extra shrinkage is doing real calibration work on every cell whose model over-predicts
   (MLB home runs 1.032 → **1.311**, NHL goals 0.961 → **1.132**, MLB stolen bases 0.992 → **1.157**).
   The "harmonize with the continuous branch" version — precision-weighted pooling — is strictly
   **worse** than what ships today (NFL tds `g6_citl_ci_hi` 0.8983 → **0.7419**, `g4` 0.0431 →
   **0.0649** fail). **KILL both pool-shape variants.**

2. **The operator's location-vs-width distinction is legitimate as an *observation* and
   rationalizing as a *reopen argument*.** §6.5 genuinely never probed location and Gate 6's CITL
   leg genuinely never scored those probes — but the correct destination for the observation is
   **§6.1 Rung A (post-hoc mean corrector), not §6.5**, because the defect is not the pool operator,
   it is **which side of the pool the corrector sits on**.

3. **The lever: the mean-stage corrector is applied to the wrong side of the pool.**
   `pipeline.py:4276-4283` fits `roe_mean`/`isotonic_mean` on the *model* mean and applies it
   **before** `_step_fuse_predictions`; `model_prob.py:1532` mirrors it before `_blend_with_book`.
   Ranjan & Gneiting (2010) Thm 1 and Gneiting & Ranjan (2013) say a non-trivial pool of calibrated
   components is itself **uncalibrated** — you must recalibrate the *output*, not the inputs. The
   repo recalibrates the inputs, and the log pool then re-decalibrates them by the measured factor
   `Σ μ_m·ρ^{1−w} / Σ μ_m`. Moving the corrector to the fused mean **ships both test cells** and
   improves Gate 4 and Gate 5 at the same time: NFL tds `g6_citl_ci_hi` 0.8527 → **1.1322**,
   `g4_pit_ks` 0.0463 → **0.0171**, `g5` 0.0351 → 0.0226 → **SHIP**; NFL interceptions 0.8928 →
   **1.0760**, `g4` 0.0839 → **0.0337** → **SHIP**.

4. **Blast radius of that move is 11 cells, enumerable today, and 4 of the 9 shipping ones are
   mathematical no-ops** (`w = 1.0`, no authentic quote ⇒ fused mean *is* the model mean). All 9
   retain `ship` in the offline screen. Contrast: a pool-shape change touches all 29 pooled count
   cells; the `_MODEL_WEIGHT_MAX` cap binds on exactly **4** served cells (NBA STL, NFL passing tds,
   NHL shots, WNBA FG3M).

5. **Three framing corrections you should take before building anything.** (a) The **power de-vig
   is dead code** — no call site in the repo passes `method="power"`, so it cannot be over-correcting
   NFL tds. (b) "33 of 48 cells have a book Brier worse than a constant" is **17 of 30** once you
   restrict to the authentic rows the blend actually pools; the rest is the synthetic class you
   already excluded. (c) The NFL tds book has the **best ranking signal in the cell**
   (`roc_auc` 0.7341 vs the model's 0.7149 and the served blend's 0.7227) — it is mis-*levelled*,
   not uninformative, so "a bad book deserves 0% weight" is the wrong inference.

---

## Key findings

### 1. The geometric pool is not a mistake — it is the exact density-level LOP for a count family

For densities in an exponential family with a common carrier, the logarithmic opinion pool
`f̄ ∝ Π fᵢ^{wᵢ}` is the member of that family whose **natural parameter is the weighted average of
the components' natural parameters** (Genest & Zidek 1986, doi:10.1214/ss/1177013825; the
externally-Bayesian characterisation is Genest, McConway & Schervish 1986,
doi:10.1214/aos/1176349934). Poisson's natural parameter is `log μ`, so the LOP of
`Poisson(μ_m)` and `Poisson(μ_b)` is exactly `Poisson(μ_m^w μ_b^{1−w})` — the geometric mean of the
rates. `fused_loc`'s `mu = exp(w·log(ev_a) + (1−w)·log(ev_b))` is therefore the *right* operator for
a count location, not an approximation of a "real" arithmetic one.

The same is true of the continuous branch, which the operator reads as "arithmetic". A Gaussian's
natural parameters are `(μ/σ², −1/2σ²)`; the LOP of two Gaussians is Gaussian with
precision `w·τ_m + (1−w)·τ_b` and mean `(w·τ_m·μ_m + (1−w)·τ_b·μ_b)/(w·τ_m + (1−w)·τ_b)` — which is
verbatim `fused_loc`'s Gamma/SkewNormal branch. **Both branches are the same operator.** They differ
in location behaviour only because the link differs: the Gaussian LOP's location is a weighted
*arithmetic* mean (so it lies in the convex hull of the two means), while the Poisson/NegBin LOP's
location is a weighted *geometric* mean (so by AM-GM it is strictly below the convex combination
whenever the two legs disagree).

**Consequence.** "Make the count branch match the continuous branch" is not "switch to arithmetic";
the literal match is the precision-weighted rule. I measured it and it is the worst option on the
table (§3, `prec w` row) because a NegBin's variance shrinks with its mean, so precision weighting
hands the *lower* leg more weight and compounds the deficit.

### 2. The LOP is not mean-preserving, and the deficit is a closed-form, systematic multiplier

Write `ρᵢ = ev_book,ᵢ / ev_model,ᵢ`. Then

```
served_geo,i  = μ_m,i · ρᵢ^(1−w)          served_arith,i = μ_m,i · (w + (1−w)·ρᵢ)
CITL_served / CITL_model = Σᵢ μ_m,i·fᵢ / Σᵢ μ_m,i
```

so the pool applies a *deterministic* haircut to whatever mean calibration the model has. This is
the calibration-in-the-large level of the Van Calster et al. (2016) hierarchy
(doi:10.1016/j.jclinepi.2015.12.005) — the coarsest level, and the one Gate 6's CITL leg tests.
The literature statement is Hora (2004, doi:10.1287/mnsc.1040.0205) and Gneiting & Ranjan (2013,
doi:10.1214/13-EJS823): the linear pool is mean-preserving but **over**-dispersed on disagreement;
the log pool **sharpens** and is not mean-preserving. Ranjan & Gneiting (2010,
doi:10.1111/j.1467-9868.2009.00726.x) Thm 1 is the general result: any non-trivial pool of distinct
calibrated forecasts is uncalibrated.

On NFL tds `median ρ = 0.219` (arm `e1b`) and the identity checks out to three decimals:
`0.219^0.10 = 0.8577` × model CITL 0.9173 = 0.787 vs the measured all-row served CITL **0.7844**.

### 3. Decomposition of the NFL tds Gate-6 gap, measured on the *gate statistic*

Every row below is a counterfactual re-scored through the production
`scorecard.compute_gates`, not a hand calculation. The harness reproduces the recorded
`scorecard.csv` row **exactly** on the identity arm (all six gates to 4 dp) and reproduces
`model_stats.parquet`'s `g4_pit_ks` and `g6_citl_ci_hi` on **35 of 35** resolved count cells, so
the counterfactuals are trustworthy at fixed `R`, fixed `dispersion_cal`, fixed temperature.

Arm `e1b` — NFL tds, NegBin + `isotonic_mean`, matrix `7918c1b8`, fitted `w = 0.9000` (on the cap):

| counterfactual | `g6_citl_ci_hi` (≥0.97) | `g4_pit_ks` (<0.05) | `g5_ece_deb` | `g1_ci_hi` | `g2` | `g3` | ship |
|---|---|---|---|---|---|---|---|
| **geometric @ w (as served)** | **0.8983** ✗ | 0.0431 | 0.0179 | −0.0234 | 0.1284 | 0.0332 | ✗ |
| arithmetic @ w | 0.9682 ✗ | 0.0343 | 0.0097 | −0.0228 | 0.0849 | 0.0699 | ✗ |
| precision-weighted @ w | 0.7419 ✗ | 0.0649 ✗ | 0.0323 | −0.0236 | 0.2223 | 0.0582 | ✗ |
| geometric @ w = 0.95 | 0.9713 ✓ | 0.0339 | 0.0113 | −0.0231 | 0.0832 | 0.0670 | ✓ |
| arithmetic @ w = 0.95 | 1.0091 ✓ | 0.0292 | 0.0116 | −0.0227 | 0.0596 | 0.0871 | ✓ |
| geometric @ w = 1.00 (book out) | 1.0494 ✓ | 0.0246 | 0.0040 | −0.0225 | 0.0342 | 0.1044 | ✓ |
| book level-recal + geometric @ w | 1.0583 ✓ | 0.0245 | 0.0094 | −0.0235 | 0.0292 | 0.0886 | ✓ |

**Three corrections to the framing this table forces.**

* The operator's counterfactual table reports the **all-row** CITL (0.8471 for arithmetic), not the
  gate statistic. Gate 6 scores the *stable top-MeanYr-quartile* segment through a player-clustered
  bootstrap **upper** bound. On the real statistic the arithmetic pool reaches **0.9682** — it misses
  by 0.0018, not "roughly half the deficit". Read the gate, not the aggregate.
* The `_MODEL_WEIGHT_MAX` cap and the pool shape are **near-identical in size** (+0.0730 vs +0.0699)
  and roughly additive.
* **No quasi-arithmetic mean can pass this gate at `w = 0.9` if the model is under-calibrated.**
  Every weighted mean `M_φ(μ_m, μ_b; w)` with `μ_b < μ_m` satisfies the internality property
  `M_φ ≤ μ_m` (Hardy, Littlewood & Pólya 1934, *Inequalities*, Thm 16). So `served ≤ model`, and
  reaching 0.97 requires the *model's* star-segment CITL to already clear 0.97 (it does here:
  1.0494) **and** the pool to give the book almost nothing. That is the exact algebraic reason no
  pre-fusion mean corrector can reach the gate.

### 4. Cohort safety KILLS the arithmetic pool

All-row served CITL at each cell's own fitted `w`, 29 pooled count cells (the correct blast-radius
read for a pool-shape change — every pooled count cell moves):

| cell | `w` | `ρ` | model CITL | geo (served) | arith | Δ |
|---|---|---|---|---|---|---|
| MLB home runs | 0.708 | 0.231 | 1.626 | 1.032 | **1.311** | +0.280 |
| NHL goals | 0.761 | 0.157 | 1.372 | 0.961 | **1.132** | +0.171 |
| MLB stolen bases | 0.772 | 2.068 | 0.839 | 0.992 | **1.157** | +0.164 |
| NBA BLK | 0.800 | 0.127 | 1.279 | 1.036 | **1.148** | +0.112 |
| NHL powerPlayPoints | 0.392 | 1.983 | 0.701 | 0.990 | 1.052 | +0.063 |
| NHL hits | 0.776 | 1.509 | 0.962 | 1.030 | 1.066 | +0.036 |
| … 23 others | | | | | | ≤ +0.013 |

`|arith − 1| > |geo − 1|` on **19 of 29**. Re-scoring all six gates: `arith @ w` **loses `ship` on
NHL hits** (`g6_over_ci_lo` 1.0192 → 1.0305 against a 1.03 threshold) and gains nothing.
Cohort-wide the arithmetic pool is a wash on the other gates (median Δ: `g4` +0.0000, `g5` +0.0001,
`g1` +0.0000) and mildly worse on the bench gate (`g3` median +0.0062, mean +0.0187).

**The reason the geometric pool looks good here is a coincidence of two biases cancelling, and you
should not rely on it** — the model systematically over-predicts on low-mean count cells and the
book sits below it, so the LOP's extra shrinkage happens to correct the model. NFL tds is the cell
where the model *under*-predicts and the book is far below it, so the same shrinkage compounds the
error twice. That is a fragile arrangement, but replacing it wholesale is a net loss today.

### 5. Gate 6's CITL leg binds almost nowhere else — NFL tds is an outlier, not a class

Across the **35** resolved count cells, **zero** fail the CITL leg; the smallest margin is NHL
powerPlayPoints at 0.9734. The lane record already established that the fusion haircut has median
1.0000 across 48 count cells with Spearman(mean(y), haircut) = −0.024. This brief adds the gate-level
version: **the blend's location behaviour is not a cohort problem.** Any change justified only by
NFL tds is, by the operator's own market-agnosticism constraint, out of scope; the change must be
justified by a *structural* argument that happens to bind hardest there. Finding 8 is that argument.

### 6. The book leg: mis-levelled, not uninformative — and the level errors go both ways

On authentic rows only (what the blend pools), across 30 count cells:

* median book `roc_auc` **0.5647**, median model `roc_auc` **0.5784** — the book carries real
  ordering signal;
* median logit calibration slope of the book's over-probability = **0.7153** (< 1) — the book's
  probabilities are systematically *too extreme*, i.e. they should be shrunk toward the base rate;
* book Brier worse than a constant base rate on **17 of 30** (29/44 if you include the synthetic
  rows the blend never pools — that is where "33 of 48" comes from);
* `book_mean_p / base_rate` ranges 0.40 (NBA BLK) to 2.48 (MLB home runs), median ≈ 1.05. **There is
  no cohort-wide directional bias.**

NFL tds and NFL interceptions are the two extremes: level gap **0.283** and **0.389**, worst in the
cohort. But NFL tds' book `roc_auc` is **0.7341**, higher than the model's 0.7149 and the served
blend's 0.7227 — the book knows who scores, it just prices the level ~3.5× low. A Poisson GLM of
`Result` on `log(ev_book)` gives slope **1.147 (se 0.066)** and intercept `exp(a) = 7.32`:
overwhelmingly a *multiplicative level* error, not a shape error, and
`Σy / Σ ev_book = 4.918` against `Σy / Σ ev_model = 1.095`.

**Verified negatives on the book leg.** (a) The archived `ev` round-trips **exactly** (ratio 1.0000
at the 10th/50th/90th percentiles) through `get_ev(line, Odds, cv, dist)` with the arm's own
`cv`/family — there is **no decode drift** of the kind in `[[serve_decode_drift_offset_mode]]`.
(b) The power/logarithmic de-vig (Clarke, Kovalchik & Ingram 2017, doi:10.11648/j.ajss.20170506.12)
is **never invoked**: `no_vig_odds` has 9 call sites and every one uses the default
`method="proportional"`. It is built, committed, and dead. It cannot be over-correcting anything.
The 3.5× level error is in the raw consensus quote or the entity/market population behind it, and
I **could not determine** its root cause from the dumps alone — that needs an archive-side audit,
which is out of scope for a read-only brief.

### 7. `_MODEL_WEIGHT_MAX = 0.9` binds on four cells, and relaxing it is the wrong lever anyway

Census of `weight` across the 73 production pickles:

| fitted `w` | count | what it means |
|---|---|---|
| exactly **1.0** | 28 | `_fit_nonsn_weight`'s `else` branch — **no authentic quote in validation**, so no book at all |
| exactly **0.9** | 4 | NBA STL (DPO), NFL passing tds (NegBin), NHL shots (DPO), WNBA FG3M (DPO) — *on the cap* |
| exactly **0.05** | 3 | MLB batter strikeouts, NFL attempts, NFL completions — on the floor |
| interior | 38 | the fit found an interior optimum; a cap change cannot move them |

A cap change is therefore **self-limiting**: only the 4 boundary cells move, and in my screen all
three resolved ones (NBA STL, NFL passing tds, NHL shots) retain `ship` at both `w = 0.95` and
`w = 1.00`. That is the cheapest possible blast radius.

**And yet it is the wrong lever.** Three reasons:

* The objective that would move `w` is `fit_model_weight`'s clamped NLL (or `fit_model_weight_crps`)
  on the *full predictive*, not the mean. It has no term that knows about Gate 6, so a wider box buys
  a Gate-6 improvement only by accident — and Claeskens, Magnus, Vasnev & Wang (2016,
  doi:10.1016/j.ijforecast.2015.12.005) show that estimated combination weights are random, which
  *biases* the combination even when the components are unbiased. A cap is a defensible shrinkage
  device; 0.9 is an arbitrary value for it, and moving it to 0.95 or 1.0 is equally arbitrary.
* On NFL tds it throws away the highest-`roc_auc` signal in the cell (Finding 6).
* It does nothing for the general defect (Finding 8), so you would be back here on the next cell.

### 8. **The actual defect: the mean corrector is on the wrong side of the pool**

`pipeline.py:4276-4283` fits the MEAN_STAGE corrector on `decoded["ev_validation"]` and applies it
to `decoded["ev"]` **before** `_step_fuse_predictions`; the in-code comment states the intent
explicitly ("correct both test and validation means BEFORE fusion so the correction flows through
the blend"). `model_prob.py:1525-1534` mirrors it ("Mean-stage post-hoc correction before blending,
mirroring train_market"). **The correction does not flow through the blend** — Finding 2 shows the
pool multiplies it by `Σ μ_m·ρ^{1−w}/Σ μ_m`, which on NFL tds is 0.855.

This is precisely the arrangement Ranjan & Gneiting (2010) Thm 1 forbids: recalibrating the
components and then pooling leaves the pool uncalibrated. Their remedy — and Gneiting & Ranjan's
(2013) — is to recalibrate the **combination**. §6.5's own reopen note already cites this rule
("co-fit … never sequence the two halves"), but it was written about the book *shape*; the same
theorem applies to the mean corrector's placement, and that placement has never been probed.

**Measured, on the arm whose corrector is affine and therefore exactly invertible** (`e1`, NFL tds,
NegBin + `roe_mean`, `a` 0.0613 / `b` 0.9386, `w` 0.8805). "none (reconstructed)" un-applies the
affine and re-pools; the post-fusion rows apply a 2-fold **player-disjoint cross-fit** corrector to
the fused mean:

| corrector stage | `g6_citl_ci_hi` | `g4_pit_ks` (<0.05) | `g5_ece_deb` | `g1_ci_hi` | `g2` | `g3` | ship |
|---|---|---|---|---|---|---|---|
| `none` (reconstructed) | 0.7876 | 0.0727 ✗ | 0.0382 | −0.0199 | 0.1682 | 0.0525 | ✗ |
| **pre-fusion `roe_mean` (as run)** | 0.8527 | 0.0463 | 0.0351 | −0.0209 | 0.1381 | 0.0586 | ✗ |
| **post-fusion `roe_mean`** | **1.1322** | **0.0171** | 0.0226 | −0.0200 | 0.0438 | 0.1667 | **✓** |
| post-fusion `isotonic_mean` | 1.2093 | 0.0178 | 0.0081 | −0.0211 | 0.0793 | 0.1117 | ✓ |
| pre + post `roe_mean` | 1.1454 | 0.0171 | 0.0227 | −0.0205 | 0.0523 | 0.1577 | ✓ |

NFL interceptions (`e3`, `roe_mean`, `a` 0.5398 / `b` 0.1962, `w` 0.9000):

| corrector stage | `g6_citl_ci_hi` | `g4_pit_ks` (<0.0689) | `g5_ece_deb` (<0.075) | `g1_ci_hi` | ship |
|---|---|---|---|---|---|
| `none` (reconstructed) | 0.8300 | 0.1420 ✗ | 0.0784 ✗ | −0.0749 | ✗ |
| pre-fusion `roe_mean` (as run) | 0.8928 | 0.0839 ✗ | 0.0437 | −0.0824 | ✗ |
| **post-fusion `roe_mean`** | **1.0760** | **0.0337** | 0.0552 | −0.0798 | **✓** |
| post-fusion `isotonic_mean` | 1.1405 | 0.0421 | 0.0950 ✗ | −0.0675 | ✗ |

The move improves Gate 4 and Gate 5 as a side effect, which is the tell that it is a real structural
fix and not a Gate-6-specific hack: correcting the mean of the object that Gate 4 actually scores
(the served predictive) is strictly more coherent than correcting the mean of an object no gate
reads. It is also **family-agnostic by construction** — the corrector is a map on a mean vector, and
every branch of `fused_loc` returns a mean or a mean-equivalent (`weighted_mean` for NegBin/DPO,
`blended_ev` for SkewNormal, with `skewnormal_loc_from_mean` already the shared re-encode).

**Blast radius: 11 cells, enumerable now** (`stat_meta.json` `posthoc ∈ MEAN_STAGE`):

| cell | dist | `shipped` | slug | fitted `w` | effect of the move |
|---|---|---|---|---|---|
| NFL passing tds | NegBin | devel | `isotonic_mean` | 0.900 | real — **primary control** |
| NBA FG3M | DPO | devel | `isotonic_mean` | 0.809 | real |
| NBA BLST | DPO | devel | `roe_mean` | 0.705 | real |
| WNBA FG3M | DPO | devel | `isotonic_mean` | 0.900 | real |
| MLB hits allowed | SkewNormal | devel | `isotonic_mean` | 0.121 | real — **continuous-branch control** |
| NFL qb yards | SkewNormal | devel | `isotonic_mean` | 1.000 | **exact no-op** |
| WNBA BLST | DPO | devel | `isotonic_mean` | 1.000 | **exact no-op** |
| WNBA DREB | DPO | devel | `isotonic_mean` | 1.000 | **exact no-op** |
| WNBA STL | DPO | devel | `isotonic_mean` | 1.000 | **exact no-op** |
| NBA FGA | SkewNormal | withheld | `isotonic_mean` | — | candidate |
| NFL interceptions | ZINB | withheld | `roe_mean` | — | candidate |

All nine `devel` cells retain `ship` in the offline screen. The tightest is **NFL passing tds**
(`g4` 0.0498 → 0.0618 against a 0.0694 threshold; `g2` 0.1487 → 0.3643 against 0.5) — that is your
canary. Note the screen applies the cross-fit corrector *on top of* the already-corrected served
mean for the eight `isotonic_mean` cells (their correctors are not cleanly invertible), so those
rows are a "pre + post" read, not the exact "post only" counterfactual; treat them as directional.

### 9. Book-leg level recalibration also works, and is **not** cohort-safe

A 2-fold player-disjoint cross-fit Poisson GLM `E[Y] = exp(a + b·log(ev_book))`, applied to the book
leg before the (unchanged, geometric) pool:

| cell | multiplier on the book | before | after | verdict |
|---|---|---|---|---|
| NFL tds | ×4.56 | `g6` 0.8983 ✗ | **1.0583 ✓, `g4` 0.0245** | gains `ship` |
| NFL interceptions | ×3.78 | 0.8928 ✗ | **1.0079 ✓, `g4` 0.0490** | gains `ship` |
| NHL shots | ×0.86 | `g4` 0.0157 | 0.0072 | improves, keeps `ship` |
| NBA STL | ×1.33 | `g4` 0.0348 | 0.0287 | improves, keeps `ship` |
| NFL passing tds | ×1.40 | — | — | neutral |
| **MLB home runs** | ×0.43 | `g6` 1.3518 ✓ | **0.9291 ✗** | **loses `ship`** |
| **MLB stolen bases** | ×0.57 | 1.1357 ✓ | **0.9235 ✗** | **loses `ship`** |
| **NHL hits** | ×0.70 | 1.0476 ✓ | **0.9632 ✗** | **loses `ship`** |

Two things follow. First, under the hard "no shipping cell loses `ship`" bar this is a **KILL as a
default**. Second — and worth an owner decision — those three cells' Gate-6 passes are *borrowed
from an over-stated book leg*: their models under-predict and a book quoting 1.5–2.1× the model's
mean is carrying them over the line. Losing `ship` there is arguably a **detection**, not a
regression. That is the operator's call, not mine.

Note also the coherence trap: recalibrating the book to be mean-unbiased and then pooling
*geometrically* leaves the pool systematically **under**-biased, because two mean-calibrated legs
pooled by AM-GM give a low answer. If you ever do recalibrate the book leg, the pool must move to
arithmetic in the same change, or you have re-created this brief's problem one layer down.

### 10. Vincentization is the principled third option, and it is not worth building today

Quantile averaging (Vincentization; Genest 1992, doi:10.1214/aos/1176348676) has exactly the
property the question asks for: because `Q̄(u) = w·Q_m(u) + (1−w)·Q_b(u)`, its mean is
`w·μ_m + (1−w)·μ_b` (**mean-preserving**), while its dispersion does *not* pick up the linear pool's
disagreement term (**not over-dispersed**). Busetti (2017, doi:10.1111/obes.12163) establishes
empirically and analytically that its properties sit **between** the linear and logarithmic pools;
Lichtendahl, Grushka-Cockayne & Winkler (2013, doi:10.1287/mnsc.1120.1667) show the average quantile
forecast is always sharper than the average probability forecast and wins in practice under both
over- and under-confidence.

**But** the quantile average of two lattice distributions is supported on a non-integer lattice, so
it does not return a NegBin/DPO `(r, p)` / `(μ, φ)` pair and breaks the served-object contract
(`_stage_family_shape_columns`, `get_odds`, `model_prob` decode, the §7.3 round-trip). The
implementable surrogate is a **moment-matched Vincentization**: set the served mean to the linear
pool of means and the served scale to the linear pool of *standard deviations* (not variances), then
re-invert into the family. That is a genuine research project with a real build cost, and Finding 8
delivers the same Gate-6 outcome for a fraction of it. **Park it; do not build it now.**

### 11. The dispersion pool must **not** travel with the location fix

`fused_loc` log-pools `r` (NegBin/ZINB) and `phi` (DPO) with the same `w`. Three independent reasons
to leave it alone:

* The recommended fix (Finding 8) does not touch `fused_loc` at all, so the question is moot for the
  lead arm.
* Every counterfactual in §3 and §8 holds `R` fixed and Gate 4 **improves** — the dispersion pool is
  not the binding constraint on either test cell.
* R4/Exp-3 (`count_dispersion_flip.md`) already closed the count dispersion objective as null: 98
  paired cross-fit corners, `g6_pass` identical on all 98, `g1_brier_diff_ci_hi` median Δ 0.0000.

If a later arm *does* move the location pool, then the dispersion pool must move with it for a
different reason: `r_blend = exp(w·log r + (1−w)·log(1/cv))` is a natural-parameter average of a
*shape*, and mixing an arithmetic location with a geometric shape is not any pool — it is a
parameter kludge with no calibration guarantee. Change both or neither.

### 12. Reality check the gates cannot give you: there is no discrimination floor

NFL interceptions ships in four of my counterfactuals — and its `roc_auc` is **0.4993 (served) /
0.5215 (model-only) / 0.5030 (book)**. Nobody in that cell has ranking signal, and Gate 1 still
passes at `ci_hi = −0.0794` because the *book* is worse than a coin. The six offline gates test
calibration, bias and non-inferiority; **none of them tests discrimination.** Any arm on this lane
must carry `roc_auc` as a reported KILL criterion, exactly as the lane record's interceptions
refutation did. Treat a `ship` at `roc_auc < 0.55` as a gate failure regardless of what
`apply_thresholds` says.

---

## Recommendation

### Verdict on the §6.5 reopen question — asked directly, answered directly

**You are rationalizing about §6.5, and you are right about the underlying observation.**

* The **reopen argument is not supported.** The §6.5 NO-GO was about the pooling *operator*, and my
  cohort screen says changing the operator is a net negative today (19/29 CITL worse, NHL hits loses
  `ship`, precision-weighting strictly worse than the incumbent). The "location vs width" and
  "Gate 6 vs Gate 4" distinctions are real, but they do not rescue the operator change — they just
  mean nobody had measured it. Now somebody has, and the answer is the same NO-GO for a new reason.
  **§6.5 stays closed. Do not spend a retrain on a pool-shape arm.**
* The **observation is correct and lands in §6.1.** The count branch really does carry a systematic
  location haircut that no pre-fusion corrector can survive, Gate 6's CITL leg really is the only
  gate that sees it, and the fix is a *stage-order* change to the §6.1 Rung A corrector — which is
  not a blend-structure change at all, does not touch `fused_loc`, and therefore does not require
  reopening §6.5.

### Routing protocol (market-agnostic; applies to every league, market and family)

**R1 — Move the MEAN_STAGE corrector from pre-fusion to post-fusion.** One rule, no per-cell
branching, no thresholds:

> The mean-stage corrector is fit on the **fused** validation mean against `Result` and applied to
> the **fused** test/live mean, before dispersion calibration and before the temperature fit.

Three code sites, all already parameterised by family:
`training/pipeline.py:4276-4283` (fit + apply moves after `_step_fuse_predictions`),
`training/pipeline.py:_step_calibrate_dispersion` (ordering — `c` must be fit against the corrected
mean), `prediction/model_prob.py:1525-1534` (`_apply_mean_posthoc` moves after
`_blend_with_book`). The pickle contract is unchanged (`posthoc` + `posthoc_blob` already persist);
`§7.3` requires the round-trip test on both sides in the same change.

**R2 — Nothing else changes.** `fused_loc` untouched. `_MODEL_WEIGHT_MAX` untouched. The book leg
untouched. The dispersion pool untouched. Each of those is a separate, later, independently-gated
decision.

**R3 — Adjudication of "does a bad book deserve 10% of the served mean?"** The repo **already has**
a principled per-cell book-quality-conditioned weight: `fit_model_weight` is a maximum-likelihood
estimate of exactly that quantity, fit out-of-sample on validation. Do **not** build a second weight
keyed on Brier skill or any other gate-adjacent statistic — that is the `[[proxy_goodhart_under_search]]`
failure mode verbatim (a static-fidelity proxy that survives a static check and diverges under
optimisation pressure), and here it would be worse, because the proxy would key on the same
outcome variable the gates score. If you later decide the MLE weight is over-shrunk, the honest
change is to the *box* (`_MODEL_WEIGHT_MAX`), justified by an estimation-error argument
(Claeskens et al. 2016) and validated on the 4 boundary cells — not a new heuristic. **Per-row**
book-quality weighting is a separate research project and needs an out-of-sample demonstration
that the row-level quality signal exists before any build.

---

## Pre-registered experiment design

Ordered cheapest-first, one axis per arm. Every arm: `--frozen-matrix-dir` + `--artifact-output`,
full 300-trial HPO, ~25 min per cell (per the lane record's corrected cost).

### Arm 0 — offline screen, zero retrains, ~2 min per cell (**run this first, always**)

The harness is already built and validated; it lives at
`/tmp/researcher_count_blend_location_screen/`. Recipe, so it can be rebuilt from scratch:

1. Read the dump + the matching `.mdl`; take `w = blob["weight"]`, `fam = blob["distribution"]`.
2. Recover the book leg by inverting the served pool:
   `ev_b = exp((log(Blended_EV) − w·log(EV))/(1−w))` on rows where the two differ (rows with
   `QuoteAuthenticity != "authentic"` are served at `w = 1` and drop out automatically).
3. Recover the temperature: `T = median(logit(1 − get_odds(served params)) / logit(P))`.
   Its within-cell spread is ~1e-14 — if it isn't, the dump and pickle are from different runs and
   the cell must be excluded (the `count_dispersion_flip.md` honesty protocol).
4. Build the counterfactual mean, rebuild `NB_P = μ/(R+μ)` (or `DP_MU = _dp_mu_from_mean(μ, φ)`),
   recompute `P = expit(logit(1 − get_odds(new params))/T)`.
5. `scorecard.compute_gates(frame, league=…, market=…)`.
6. **Validity check, mandatory:** the identity counterfactual must reproduce the recorded
   `scorecard.csv` / `model_stats.parquet` row. Mine does on 35/35 count cells and 3/3 arms.

This screen is exact for Gates 2, 3 and 6 (pure location), exact-at-fixed-`R` for Gate 4, and
fixed-temperature for Gates 1 and 5. It is *not* a substitute for a retrain — `c`, `T` and `w` all
refit in production — but it will tell you the sign and roughly the magnitude for free.

### Arm 1 — NFL tds, post-fusion `isotonic_mean` (primary test case)

Byte-identical `StrategyControlsJSON` to the lane's `e1b` arm on matrix `7918c1b8`, differing only
in the corrector's stage.

* **SHIP:** all six gates pass **and** `roc_auc ≥ 0.70` (the `e1b` incumbent's 0.7227 is the bar —
  do not accept a Gate-6 pass bought with rank collapse).
* **AMBIGUOUS (go to Arm 1b):** `g6_citl_ci_hi ∈ [0.94, 0.97)`.
* **KILL:** `g6_citl_ci_hi < 0.94`, **or** `g4_pit_ks ≥ 0.05`, **or** `g1_brier_diff_ci_hi ≥ 0.005`,
  **or** `roc_auc < 0.65`. A KILL here means the offline screen's +0.28 on `g6` did not survive a
  validation-fit corrector, and the lane closes.

*Screen says:* `g6` 1.13 (`roe_mean`) / 1.21 (`isotonic_mean`), `g4` 0.017, ship. Expect the live
number lower — see Reality checks.

### Arm 1b (only if Arm 1 is ambiguous) — NFL tds, post-fusion `roe_mean`

Affine is lower-variance than isotonic and val→test degrades less. Same triggers.

### Arm 2 — **NFL passing tds** (control: currently `devel`, count branch, `w` on the cap)

The tightest control in the set (`g4` slack 0.0196, and the screen moves it to 0.0076 slack).

* **PASS:** retains `ship`.
* **KILL the whole lane:** loses `ship`. A market-agnostic change that demotes a shipping count cell
  is not shippable, full stop.

### Arm 3 — **MLB hits allowed** (control: continuous branch, SkewNormal, `w = 0.121`)

This is the arm that proves market-agnosticism: a SkewNormal cell where the book carries ~88% of the
served location. If the move is coherent, this cell should be near-neutral (the screen shows
`g6` 1.0701 → 1.0507, `g2` 0.0210 → 0.0785, `g3` 0.1201 → 0.0616, ship retained).

* **PASS:** retains `ship`.
* **KILL:** loses `ship`, or `g4_pit_ks` degrades by > 0.010 — the SkewNormal re-encode
  (`skewnormal_loc_from_mean` at fixed `SN_Scale`/`SN_Alpha`) is the one place where a mean-only
  correction can silently move the shape.

### Arm 4 — **the no-op assertion** (a golden test, not a retrain)

`NFL qb yards`, `WNBA BLST`, `WNBA DREB`, `WNBA STL` all fit `w = 1.0` because
`_fit_nonsn_weight` found no authentic validation quote. For them the fused mean **is** the model
mean, so the move must be *bit-identical*. Assert `Blended_EV` unchanged to 1e-12 in
`tests/golden/`. If it is not, the implementation has a bug in the `w = 1` path and nothing else in
the lane is trustworthy.

### Arm 5 — **NBA BLST + NBA FG3M + WNBA FG3M** (remaining live controls)

Run only after Arms 1–4 clear. Each must retain `ship`. Cheapest last because the screen shows all
three comfortably clear.

### Contingent Arm 6 — book-leg level recalibration (**do not run unless Arms 1–5 all fail**)

If the post-fusion move dies, this is the only other lever with measured Gate-6 power.
Pre-registered KILL, non-negotiable: **any of MLB stolen bases / NHL hits / MLB home runs losing
`ship` kills the arm** (the screen says all three will). Do not run it as an experiment; run it as a
*decision packet* asking the owner whether three cells whose Gate-6 pass is borrowed from a
mis-levelled book should keep shipping.

### Explicitly NOT to be run

* Any `fused_loc` pool-shape arm (geometric → arithmetic, or → precision-weighted). Screened, KILLED
  cohort-wide, §5 above.
* Any `_MODEL_WEIGHT_MAX` arm as a *lead*. It is cheap and safe but it does not generalise and it
  discards the best-ranking signal in the flagship cell.
* Any per-cell or per-row book-quality weight heuristic. R3 above.

---

## Reality checks

**Effect size, and the regime where it holds.** The +0.28 on `g6_citl_ci_hi` and the −0.029 on
`g4_pit_ks` come from a corrector fit **on the test dump** under a 2-fold player-disjoint cross-fit.
Production fits on validation and evaluates on test — a strictly harder problem. This repo has been
burned by exactly this twice (`[[deterministic_ab_g4_oversell]]`,
`[[board_confirm_gap_root_cause]]`: 7/7 ship at ≥ +0.07 slack, 0/5 at ≤ +0.05). Discount hard. The
mitigating fact is that `roe_mean` is a **two-parameter affine** map — about the lowest-variance
corrector that exists — so the val→test gap should be far smaller than for a 300-trial HPO corner.
`isotonic_mean` has more capacity and correspondingly more val→test risk; that is why Arm 1b exists.
NFL tds' margin in the screen (1.13 against 0.97) is large enough to absorb a substantial discount;
NFL interceptions' (1.076) is not, and it should not be trusted.

**The move over-corrects in the direction Gate 6 does not police.** Post-fusion, NFL tds' star-segment
CITL upper bound goes to 1.13–1.21 — the served mean now runs *above* the outcome on stars. Gate 6's
CITL leg is one-sided (under-prediction only) and its over leg is bench-scoped and guarded by
`mean(Result) ≥ 1`, so **nothing catches this**. The lane record already flags two live cells in this
state (NFL receiving tds 1.639, WNBA FTM 1.156). Expect the post-fusion move to add cells to that
list. If the owner cares, the report-only companion to add is the *all-row* served CITL, which this
brief computes for all 44 count dumps in
`/tmp/researcher_count_blend_location_screen/cohort_out.csv`.

**Research project vs engineering project.** R1 is an **engineering project**: known method
(recalibrate the combination, not the components — Ranjan & Gneiting 2010), three call sites, no new
concepts, an existing corrector, an unchanged pickle contract, an enumerable 11-cell blast radius and
a 4-cell bit-identity assertion. Estimated cost: one focused session plus ~6 confirm arms
(~2.5 h of compute). Findings 9 (book recalibration) and 10 (Vincentization) are **research
projects** with unproven transfer and, in the book case, a measured cohort cost.

**What would make this recommendation wrong.**

1. *The offline screen is optimistic about the corrector and Arm 1 lands at `g6` ≈ 0.93.* Then the
   pre→post move is real but too small, and the lane closes rather than escalating to a pool change
   (which §5 has already killed).
2. *The dispersion refit eats the gain.* My screen applies the mean correction **on top of** the
   already-dispersion-calibrated `R`. Production would fuse → correct → fit `c`. If `c` re-widens to
   restore the CRPS optimum, `g4` may not improve as much as 0.0463 → 0.0171 suggests. Pre-register
   the ordering (`fuse → mean-correct → dispersion-calibrate → temperature`) so the arm is
   interpretable either way.
3. *`w` and the corrector are now sequenced, not co-fit.* `_fit_nonsn_weight` fits `w` on the
   **uncorrected** legs; the corrector then fits on the fused mean. §6.5's own note (citing Ranjan &
   Gneiting 2010 Thm 1) warns against sequencing two halves of a calibration. The sequential version
   is the cheap first arm; if Arm 1 is ambiguous, a co-fit of `(w, corrector)` on one held-out
   objective is the escalation — and that *is* a §6.5 reopen, so it needs a fresh brief.
4. *The `(1−π)` contract.* `[[posthoc_mean_miscontracted_on_gated]]` records that `roe_mean`
   double-applies `(1−π)` on ZINB / gated-SkewNormal cells, which is why MEAN_STAGE is currently
   suppressed there. Post-fusion the corrector would be fit on the fused **base** mean while
   `_zero_inflated_mean` has Gates 2/3/6 read `(1−π_blend)·base`. **The fit target and the gate
   target must be made to agree in the same change**, or you will reproduce the mis-contract one
   stage later. This is the single most likely implementation bug in R1.
5. *NBA PF is not helped.* PF's book is synthetic (1 of 2123 rows pooled), so the fused mean is the
   model mean and the move is a no-op. Its wall — one `posthoc` slot, two jobs — is untouched by
   everything in this brief.

---

## Open questions / caveats

* **Why is NFL tds' book 4.9× low on the mean and 3.5× low on the over-probability?** Not decode
  drift (archived `ev` round-trips exactly), not the power de-vig (dead code). The GLM says a clean
  multiplicative level error (slope 1.147 ± 0.066). I **could not determine** the cause from the
  dumps; it needs a read-only archive audit of the NFL tds `odds`/`lines` rows — which books, which
  observation dates, whether `under_prob` is a native de-vigged quote or a value back-derived from a
  stored `ev` by `migrate_archive_shapefree.py` (that path decodes **with** `book_gate`, while
  `training_quotes._authentic_quote` re-inverts **without** it; the asymmetry is the right order of
  magnitude, but the ZINB cells in my cohort show no systematic level-gap signature, so it is a
  hypothesis, not a finding). Same audit covers NFL interceptions (3.7×), NBA BLK (0.40×) and
  MLB home runs (2.48×).
* **Three shipping cells' Gate-6 pass is borrowed from a mis-levelled book** (MLB stolen bases,
  NHL hits, MLB home runs; the book runs 1.5–2.1× above the model and the pool lifts the served mean
  across the line). Owner decision, not a session edit.
* **The six gates have no discrimination floor** (Finding 12). Carry `roc_auc` as an explicit KILL on
  every arm in this lane, and consider whether the gate set needs a report-only companion.
* **`_pred_cdf_pmf`'s Gamma/ZAGamma branch reads `EV`, not `Blended_EV`** (`scorecard.py`), so on a
  Gamma cell Gate 4 would score the *model's* mean rather than the served one. There are currently
  **zero** Gamma/ZAGamma cells in `stat_meta.json` (43 SkewNormal / 25 DPO / 17 ZINB / 8 NegBin), so
  this is latent, not live — but any future Gamma cell would silently escape both this lane's fix and
  Gate 4's location coverage.
* **The offline screen cannot see `w`, `c` or `T` refitting.** It is a sign-and-magnitude tool. Every
  number in §3 and §8 carries that caveat.
* **NFL tds' test set is a single season (2025)**, so no temporal contrast is available to date the
  book-level anomaly. The cross-league caveat applies: n = 2466 for NFL tds against ~2200 for a
  typical NBA/WNBA cell is fine, but NFL interceptions is n = 389 and nothing measured on it should
  be treated as a second data point.
* **Vincentization is parked, not refuted** (Finding 10). If a future cohort presents a genuine
  location *and* width defect together, the moment-matched quantile pool is the principled joint fix
  and Busetti (2017) is the right starting point.

---

## Bibliography

| # | Source | Identifier | Used for |
|---|---|---|---|
| 1 | Genest, C. & Zidek, J. V. (1986). "Combining Probability Distributions: A Critique and an Annotated Bibliography." *Statistical Science* 1(1), 114–135. | doi:10.1214/ss/1177013825 | LOP definition; the citation already in `fused_loc`'s docstring |
| 2 | Genest, C., McConway, K. J. & Schervish, M. J. (1986). "Characterization of Externally Bayesian Pooling Operators." *Annals of Statistics* 14(2), 487–501. | doi:10.1214/aos/1176349934 | LOP is the externally-Bayesian pool; geometric-average characterisation |
| 3 | Ranjan, R. & Gneiting, T. (2010). "Combining Probability Forecasts." *JRSS-B* 72(1), 71–91. | doi:10.1111/j.1467-9868.2009.00726.x | Thm 1: any non-trivial pool of calibrated forecasts is uncalibrated ⇒ recalibrate the *combination* |
| 4 | Gneiting, T. & Ranjan, R. (2013). "Combining predictive distributions." *Electronic Journal of Statistics* 7, 1747–1782. | doi:10.1214/13-EJS823 | Linear pool over-dispersed; log pool sharpens; generalized / spread-adjusted / beta pools |
| 5 | Hora, S. C. (2004). "Probability Judgments for Continuous Quantities: Linear Combinations and Calibration." *Management Science* 50(5), 597–604. | doi:10.1287/mnsc.1040.0205 | Linear combination degrades calibration of already-calibrated components |
| 6 | Lichtendahl, K. C. Jr., Grushka-Cockayne, Y. & Winkler, R. L. (2013). "Is It Better to Average Probabilities or Quantiles?" *Management Science* 59(7), 1594–1611. | doi:10.1287/mnsc.1120.1667 | Quantile average is always sharper than the probability average |
| 7 | Busetti, F. (2017). "Quantile Aggregation of Density Forecasts." *Oxford Bulletin of Economics and Statistics* 79(4), 495–512. | doi:10.1111/obes.12163 | Vincentization sits *between* the linear and logarithmic pools |
| 8 | Genest, C. (1992). "Vincentization Revisited." *Annals of Statistics* 20(2), 1137–1142. | doi:10.1214/aos/1176348676 | Quantile aggregation preserves functional form under location-scale |
| 9 | Bordley, R. F. (1982). "A Multiplicative Formula for Aggregating Probability Assessments." *Management Science* 28(10), 1137–1148. | doi:10.1287/mnsc.28.10.1137 | Axiomatic derivation of the multiplicative (log) pool |
| 10 | Claeskens, G., Magnus, J. R., Vasnev, A. L. & Wang, W. (2016). "The forecast combination puzzle: A simple theoretical explanation." *IJF* 32(3), 754–762. | doi:10.1016/j.ijforecast.2015.12.005 | Estimated weights are random ⇒ the combination is biased ⇒ a shrinkage cap is defensible, 0.9 is arbitrary |
| 11 | Winkler, R. L., Grushka-Cockayne, Y., Lichtendahl, K. C. Jr. & Jose, V. R. R. (2019). "Probability Forecasts and Their Combination: A Research Perspective." *Decision Analysis* 16(4), 239–260. | doi:10.1287/deca.2019.0391 | Survey framing of pool choice vs recalibration |
| 12 | Van Calster, B., Nieboer, D., Vergouwe, Y., De Cock, B., Pencina, M. J. & Steyerberg, E. W. (2016). "A calibration hierarchy for risk models was defined: from utopia to empirical data." *J. Clin. Epidemiol.* 74, 167–176. | doi:10.1016/j.jclinepi.2015.12.005 | Calibration-in-the-large is the coarsest level — exactly Gate 6's CITL leg |
| 13 | Clarke, S., Kovalchik, S. & Ingram, M. (2017). "Adjusting Bookmaker's Odds to Allow for Overround." *American Journal of Sports Science* 5(6), 45–49. | doi:10.11648/j.ajss.20170506.12 | The power de-vig `_power_devig_exponent` implements — confirmed unwired |
| 14 | Brockwell, A. E. (2007). "Universal residuals: A multivariate transformation." *Statistics & Probability Letters* 77(14), 1473–1478. | doi:10.1016/j.spl.2007.02.008 | Randomized PIT — Gate 4's statistic (already cited in-repo) |
| 15 | Czado, C., Gneiting, T. & Held, L. (2009). "Predictive Model Assessment for Count Data." *Biometrics* 65(4), 1254–1261. | doi:10.1111/j.1541-0420.2009.01191.x | Mid-PIT / non-randomized PIT for count predictives (already cited in-repo) |
| 16 | Hardy, G. H., Littlewood, J. E. & Pólya, G. (1934). *Inequalities*, Cambridge University Press, Thm 16 (internality of quasi-arithmetic means). | ISBN 978-0521358804 | No weighted mean of `(μ_m, μ_b)` with `μ_b < μ_m` can exceed `μ_m` |

**In-repo evidence, not literature** (so the distinction is explicit): Findings 3–9, 11 and 12 are
measurements on this repo's own artifacts —
`/tmp/scratch/count-mean/{e1,e1b,e3}/artifacts/`, `src/sportstradamus/data/test_sets/*.csv`,
`src/sportstradamus/data/models/*.mdl`, `src/sportstradamus/data/training/model_stats.parquet`,
`src/sportstradamus/data/config/stat_meta.json` — scored with the production
`sportstradamus.training.scorecard`. Harness and the 44-cell cohort table:
`/tmp/researcher_count_blend_location_screen/`.
