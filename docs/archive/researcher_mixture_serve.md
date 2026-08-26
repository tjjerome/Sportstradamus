# In-repo research brief — should the Mixture family be promoted from research-only to serve+confirm?

**Question:** §8.2 hole #0a says "put Mixture back in the pool the day it can serve." Is that day now?
Is the proposed serve build (decode + moment-matched blend + scalar dispersion + book-leg→Normal)
statistically sound, is it complete, and does the projected lift justify the build cost?

**Scope:** `docs/handoffs/model_improvement_track.md` §6.6 / §8.2 hole #0a, family-escalation axis.
**Date:** 2026-08-25. **Mode:** read-only (a board sweep is live; nothing was trained, no config touched).

---

## TL;DR

1. **KILL the serve build as scoped — hold Mixture research-only, with a pre-registered reopen
   trigger.** The verdict is driven by fresh ground truth, not by a defect in the family: on the
   live board (`strategy_research_board.csv`, swept 2026-08-22 → 2026-08-26) **every near-term
   addressable withheld continuous cell already has a shipping corner on a confirmable family**
   except two — and those two are exactly the cells the current Mixture implementation is
   *unsafe* on.
2. **Hole #0a's premise is stale.** NBA FGA now has **4 board-shipping DPO corners**
   (best `discounted_slack +0.053`, `g4 0.0283`), so the July "+0.150 Mixture" is superseded.
   NFL passing yards has **0/17 shipping corners and fails g1 on 100% of them**
   (`g1_ci_hi +0.0060` vs the 0.005 bar) — and the house rule is explicit that a family/shape
   lever moves g4 and **cannot manufacture g1 edge** (`[[cdf_recal_nonstationary_pit]]`,
   `[[nfl_volume_cells_feature_mature]]`). Mixture cannot ship passing yards.
3. **The only genuinely g4-bound continuous residuals left are NFL receiving yards (0/15, g4 fails
   79%, g6 57%) and NFL rushing yards (g4 78%, g6 71%, every "shipping" corner has negative
   `discounted_slack`). Both carry `zi = 0.127 / 0.116` — and the Mixture branch publishes no
   zero gate anywhere** (`pipeline._step_decode_predictions` forces `gate_test = None`;
   `scorecard._pred_cdf_pmf` returns `zeros_like(y)` for Mixture). Training drops the zero rows
   (`zero_adjusted_continuous` includes Mixture) but nothing re-attaches the atom, so a Mixture
   corner on those cells serves `E[Y|Y>0]` — a measured **+20.8% / +14.2% mean inflation** that
   **Gates 2, 3 and 6 are structurally unable to detect** (implied `g2 z ≈ 0.36 / 0.25` against a
   0.5 bar; Gate 6's only applicable leg is one-sided *under*).
4. **Even with the gate built, the measured ceiling does not clear the bar on receiving yards.**
   Three independent estimates agree: the repo's own oracle sweep (Mixture KS **0.062**), the July
   board's conditional Mixture corner (**0.0552**), and my player-disjoint cross-fit residual screen
   (GMM-2 heldout KS **0.0942** marginal / **0.1026** on `Y>0`) — against a 0.05 bar, with g6
   co-failing.
5. **Three no-regret items that are worth doing regardless of the verdict**, all cheap:
   (a) a **fail-fast guard on `dist: "Mixture"` in `stat_meta.json`** — I reproduced
   `TypeError: 'NoneType' object is not subscriptable` from `get_ev`/`get_odds` and traced **~8 book-leg
   call sites outside `model_prob`** (confer's `_prop_book`, `archive.add_dfs`,
   `archive._devig_over`, `training_quotes` via `stats.base._resolve_gameday_quotes`,
   `stats.*.check_combo_markets`, `clv`, `analysis`) that a one-line hand edit would take down;
   (b) the **`_scale_mixture` about-the-mean correction** (verified exact vs. the current saturating
   form); (c) the **cross-fit residual-shape screen** below as a standing, seconds-per-cell
   read-only predictor of mixture headroom, so the reopen trigger can be evaluated without
   spending board budget.

---

## Key findings

### 1. The fitted 2-component structure is a *scale mixture*, not a bimodal regime mixture

§6.6 motivates the Mixture head as covering "blowout/benching regimes no single-mode family
represents." The data does not support that story.

I decoded the served SkewNormal predictive from each cell's production test dump
(`scorecard._decode_sn_loc_scale` via the `baselines` registry, honouring `DenomCol` / `GlobalMean`),
formed the standardized residual `z = (Result − mean)/sd`, split **player-disjoint** in half, fitted
on one half and scored KS-vs-Uniform on the other:

| cell | n | zi | as-is `N(0,1)` | GMM M=2 | GMM M=3 | StudentT | fitted `w₂` | minor-comp sd | `V_between/V_total` |
|---|---|---|---|---|---|---|---|---|---|
| NFL receiving yards | 2238 | 0.127 | 0.1429 | 0.0942 | 0.0869 | **0.0900** | 0.082 | 12.4× | 0.171 |
| NFL rushing yards | 1012 | 0.116 | 0.0895 | **0.0800** | 0.0799 | 0.1159 | 0.060 | 19.1× | 0.304 |
| NFL passing yards | 369 | 0.005 | 0.0690 | 0.0615 | 0.0830 | **0.0598** | 0.16 | 2.8× | 0.053 |
| NBA FGA | 1873 | 0.000 | 0.0743 | **0.0477** | 0.0386 | 0.0563 | 0.008 | 23.3× | 0.133 |
| NBA PTS | 2251 | 0.025 | 0.0395 | 0.0397 | 0.0198 | **0.0364** | 0.008 | 9.5× | 0.000 |

Every cell fits `w₂ ∈ [0.008, 0.16]` with the minor component's scale **3–23× the major's**. That is
a two-point discretization of a scale-mixture (heavy-tail) density, not a location mixture of two
regimes. Two consequences:

* **A Student-t LSS head is a materially cheaper competitor that ties or beats the mixture on 3 of
  5 cells** (receiving, passing, NBA PTS). StudentT is 3 parameters vs 5, ships natively in the
  pinned `lightgbmlss 0.6.1` (`distributions/StudentT.py`), has closed-form scipy CDF/PPF, no
  component-permutation or collapse pathology, exact scale semantics for `dispersion_cal`, and
  blends through the *existing* SkewNormal precision-weighted machinery with no new operator.
  **The in-repo StudentT NO-GO (`[[cdf_recal_nonstationary_pit]]`) does not cover this**: it was
  measured as a *global shape hand-fit on the test set*, whereas a StudentT **LSS head** carries a
  per-row conditional scale. That distinction is exactly what the same memory says matters ("a
  **global** monotone warp — or a **single global** shape parameter — cannot be heavy at low mean and
  near-Gaussian at high mean at once"), and it is not a refutation of the head.
* Refitting a *unimodal* family to the residual is actively harmful: "Normal refit" (0.2576) and
  "SkewNormal refit" (0.2536) are **worse than leaving the residual alone** (0.1429) on receiving
  yards, because a moment/ML fit to a heavy-tailed sample is dominated by the tail and wrecks the
  bulk. That is a clean, quantified statement of why a scalar `dispersion_cal` — which *is* a scale
  refit — is the wrong instrument on these cells, and it corroborates §8.2 #9(d).

The mean-conditional diagnosis in `[[cdf_recal_nonstationary_pit]]` reproduces and is severe:
`P(PIT > 0.95)` by served-mean quartile is **0.200 / 0.097 / 0.111 / 0.118** (receiving) and
**0.229 / 0.091 / 0.071 / 0.067** (rushing) against a nominal 0.050. A *conditional* mixture whose
weights and scales are functions of `x` is the one family class that can be heavy at low mean and
Gaussian at high mean — so the theoretical case for Mixture over any global-shape family is real.
It is the *magnitude* that fails, not the mechanism.

### 2. Fusion by location shift is exact at `w = 1` and degrades as `w → 0` — but it errs on the safe side

`pipeline._fuse_mixture` fits `model_weight` by scoring the mixture through the SkewNormal machinery
with `sigma = mixture sd, alpha = 0` (a moment-matched normal), then applies
`_shift_mixture(mix, weighted_mean − ev)` — both component locations shifted by the pooled-minus-model
delta, scales and weights untouched.

**What is right.** Because the mixture mean is linear in the component locations, the shift makes the
served mixture mean **exactly** `weighted_mean`. At `w = 1` (and on every non-authentic-quote row,
which `_fuse_mixture` pins to `w = 1`), `fused_loc` returns `blended_ev = ev_a` with `α = 0`, so the
shift is **identically zero and the fusion is exact** — no approximation at all. **15 of the 29
SkewNormal cells in `model_stats.parquet` sit at `model_weight = 1.0`.**

**What is approximated.** With `α = 0` on both legs, `fused_loc`'s "precision-weighted" SkewNormal
branch *is* the Gaussian logarithmic opinion pool (Genest & Zidek 1986, DOI 10.1214/ss/1177013825;
the Gaussian product-of-experts identity, Hinton 2002, DOI 10.1162/089976602760128018): it returns
`σ_blend = (w/σ_m² + (1−w)/σ_b²)^{−1/2}`. The mixture branch discards that width term entirely.
The omitted per-row factor is `ρ = (w + (1−w)k²)^{−1/2}` with `k = σ_m/σ_b`. Measuring `k` on the
production dumps (`σ_b = EV·cv`, `σ_m ≈ served sd`):

| cell | `k` p10 | `k` med | `k` p90 | `ρ` span at `w=0.5` | `ρ` span at `w=0.05` |
|---|---|---|---|---|---|
| NFL receiving yards | 0.366 | 0.794 | 1.064 | 1.33 → 0.97 | 2.36 → 1.10 |
| NFL rushing yards | 0.234 | 0.794 | 1.153 | 1.38 → 0.93 | 2.4 → 1.0 |
| NBA FGA | 0.603 | 0.785 | 0.980 | 1.20 → 1.00 | 1.6 → 1.0 |

So the location-shift blend omits a **per-row width factor spanning ~35% at `w = 0.5` and up to
~2.4× at `w = 0.05`**. A single scalar `dispersion_cal` absorbs the geometric level of `ρ` but not
its row-to-row spread — and the residual is correlated with `σ_m/σ_b`, i.e. exactly the
heteroscedasticity Gate 4 prices.

**Is the omission wrong?** Not obviously. The combination literature says the log-linear pool is
**under**-dispersed and the linear pool **over**-dispersed relative to the ideal
(Ranjan & Gneiting 2010, DOI 10.1111/j.1467-9868.2009.00726.x; Gneiting & Ranjan 2013,
DOI 10.1214/13-EJS823 — they introduce the spread-adjusted and beta-transformed linear pools
precisely to repair this). Since the repo's dominant symptom is under-dispersion, *not* contracting
is accidentally the safer error. So the honest verdict on 1(a) is: **acceptable for a pilot,
provably exact at `w = 1`, and it should be scoped to `w ≈ 1` cells rather than fixed.**

**The principled tractable alternative, if it is ever fixed**, is the Gaussian-sum /
component-wise conjugate pool (Alspach & Sorenson 1972, DOI 10.1109/TAC.1972.1100034): a Gaussian
mixture times a Gaussian *is* a Gaussian mixture, so pool each component with the book separately
(`μ_k' = (w·prec_k·μ_k + (1−w)·prec_b·μ_b)/(w·prec_k + (1−w)·prec_b)`, `prec_k' = w·prec_k + (1−w)·prec_b`)
and re-weight `w_k' ∝ w_k·Z_k` by each component's agreement with the book. ~15 lines, per-row
correct, and it reweights toward the component the book supports — which is the one thing the
location shift cannot do. Note it inherits the LogOP's under-dispersion, so it is only an
improvement if paired with the existing `dispersion_cal`.

### 3. `dispersion_cal` on a mixture is a saturating, semantics-breaking control — and there is an exact fix

`_scale_mixture` multiplies both component scales by `c`, holding the component *locations* fixed.
Total variance is `V(c) = c²·V_within + V_between` with `V_between = w₁w₂(μ₁−μ₂)²`. Measured
`V_between/V_total` is 0.17–0.36 on these cells, so:

| cell shape | sd(c=0.1)/sd | sd(c=0.5)/sd | sd(c=2)/sd | hard floor `√(V_b/V_t)` |
|---|---|---|---|---|
| receiving-like (`V_b/V_t = 0.31`) | 0.423 | 0.615 | 1.868 | **0.413** |
| rushing-like (`V_b/V_t = 0.36`) | 0.558 | 0.692 | 1.757 | **0.552** |
| FGA-like (`V_b/V_t = 0.19`) | 0.376 | 0.591 | 1.898 | **0.364** |

Three consequences:

* `dispersion_cal` **is no longer a width multiplier** on this family. The `model_stats` column
  semantics documented in `CLAUDE.md` ("`shape_ratio ≈ 1.0` ⇒ well-calibrated dispersion,
  `< 1.0` ⇒ over-dispersed") silently change meaning, as do any drift monitors reading it.
* The narrowing direction **saturates**: no `c` can bring the served sd below ~0.41–0.55×. The
  optimizer then rails at the bracket bound (0.1), and `model_prob._serving_dispersion_cal` treats
  `dispersion_cal ≤ 0.1005` as a **diverged fit and serves the unscaled shape** — a silent
  train/serve divergence (training scored `c ≈ 0.1`, serving uses `c = 1.0`) that no gate can see.
  *Severity check:* the current SkewNormal cohort's minimum `dispersion_cal` is 0.726, so no live
  cell needs > 45% narrowing today. This is a correctness/semantics defect, not a live blocker.
* I did **not** reproduce multimodality in `PIT-KS(c)` — on a representative configuration
  `minimize_scalar(bounded)` found the same optimum as a 0.05-step grid. So the Brent-local-optimum
  concern is theoretical, not observed. Do not overclaim it.

**Exact fix (recommended if the build ever proceeds, ~3 lines + serve mirror):** scale about the
mixture mean — `m = w₁μ₁ + w₂μ₂`, `μ_k → m + c(μ_k − m)`, `σ_k → c·σ_k`. This is a pure scale
transform of the random variable: **mean-preserving, shape-preserving in the standardized sense
(skew/kurtosis/bimodality invariant), and scales total sd by exactly `c`** — verified numerically
(c = 0.5 / 0.8 / 1.25 / 2.0 → sd ratio 0.5000 / 0.8000 / 1.2500 / 2.0000). It does **not** violate
the docstring's intent ("per-component scaling would let the optimizer trade tail mass between
modes"): a single scalar affine map preserves component separation *in units of sd*.

### 4. Book leg mapped to `"Normal"` is correct — and the mapping is NOT one site

I verified that `get_odds(line, ev, "Normal", cv=…)` is **bit-identical** to
`get_odds(line, ev, "SkewNormal", sigma=None, skew_alpha=None)`, which is exactly the leg
`_fuse_mixture` uses (`fused_loc` with `book_sigma=None` → `σ_b = ev_b·cv`, `α = 0`). So
`Mixture → "Normal"` is the right map. **Do not map to `"SkewNormal"`** — `model_prob._book_over_prob`
branches on `dist == "SkewNormal"` into `book_skewnormal_shape`, the WS-1/WS-2 per-cell *fitted
asymmetric* book curve, which training's mixture blend never uses; that would be a train/serve
shape break.

**But the map has to be applied everywhere `stat_dist[league][market]` reaches `get_ev`/`get_odds`.**
`stat_dist` is a derived view of `stat_meta.json` (`helpers/config.py`), and a confirmed Mixture
corner persists `"dist": "Mixture"` (its `persist` map includes `dist`). I reproduced the failure:

```
get_odds(45.5, 40.0, "Mixture", cv=0.7, step=0.5, gate=None)
  → TypeError: 'NoneType' object is not subscriptable      # _mixnorm_odds(..., mix["w1"], ...) with mix=None
get_ev(45.5, 0.52, 0.7, dist="Mixture")
  → TypeError: 'NoneType' object is not subscriptable
```

Traced call sites that would fire (`dist` read from config, `mix` never supplied):

| site | job | consequence |
|---|---|---|
| `moneylines.py:817-819` `_prop_book` | **confer cron** | crashes the odds fetch |
| `helpers/archive.py:1154` `add_dfs` | prophecize / DFS write | crashes |
| `helpers/archive.py:886` → `_devig_over` | **close-lines cron** | `_devig_over` catches only `(ValueError, RuntimeError)`; `TypeError` escapes |
| `helpers/training_quotes.py:233,333` via `stats/base.py:2212` | **matrix rebuild** | kills a league regen |
| `stats/{base,nba,nhl,mlb}.py` `check_combo_markets` / `profile_market` | prediction + training | crashes |
| `prediction/model_prob.py:310,355` `_book_cell_params`/`_book_over_prob` | prophecize | crashes |
| `clv.py:164,174` | CLV backfill | crashes |

`training/calibration.py:fit_book_weights` is **safe** — `_make_book_objective` falls through to the
Normal branch for any non-count family, which is the same convention I am recommending.

Why this has stayed latent: during a board run the family arrives via `meditate --dist Mixture`
(`dist_override`), which `_step_select_distribution` consumes **without persisting** under
`--deterministic`, so every config-reading path still sees `SkewNormal`. The trap only arms at
persist/serve time. Note `stat_meta` `dist` is human-forceable
(`[[dist_selection_forceable_via_stat_meta]]`) — **a hand edit today would take down confer.**

### 5. Two more serve-path holes the dispatch's build list does not name

* **`_serve_offset_mode` is hardcoded to SkewNormal** (`model_prob.py:590-593`:
  `return dist == "SkewNormal" and get_target_normalization(...).start_mode_flag == "offset"`).
  Training seeds a Mixture with `offset_mode = strategy.start_mode_flag == "offset"`, and
  `_mixture_start_values` branches on it (`loc = zeros` vs `loc = mu`). Two of Mixture's four
  `_NORMS` values are offset strategies, and **NBA FGA's incumbent normalization is
  `centered_additive_eb_meanyr_k10`** — an offset strategy. Because LightGBMLSS adds
  `start_values` back as `init_score` at predict time, a wrong seed is a silent per-row location
  error of order `MeanYr`. This is precisely the `[[serve_decode_drift_offset_mode]]` failure class
  (~2× overconfidence), and it lands on the flagship pilot cell.
* **`_model_predictive_sd` has no Mixture branch** — it falls through to `np.clip(model_ev, 0.5, None)`,
  i.e. it uses the *mean* as the predictive SD for `_sanitize_book_ev`'s plausibility band. Needs a
  `_mixture_moments`-derived branch. `_sanitize_model_ev` likewise clamps `Projection` without
  touching the component locations, so the clamp and the served mixture would disagree unless the
  location shift is computed from the *clamped* mean.

Neutral/OK: `get_push_prob` returns 0 for Mixture (verified — correct for a continuous family);
`_mixnorm_ppf` has **no production caller** today, so the ladder risk is latent, and its
±8·max(scale) bracket with 80 bisections is safe for a Gaussian mixture (the goldens pin a 1e-4
round trip); `_mixnorm_cdf` saturates cleanly to 0/1 at extreme alt lines with no underflow trap.
One parity nit to pin with a golden: training's Mixture `get_odds` calls omit `step` (defaults to 1)
while serve passes the pickle's empirical `step`; identical for integer-target cells, divergent on a
0.5-granularity cell. `predict_dist` always draws 1000 samples/row even for `pred_type="parameters"`
(`mixture_distribution_utils.predict_dist` → `draw_samples`), so a Mixture cell adds a measurable
(if small) charge to the `[[prophecize_runtime_lane]]` budget.

### 6. The zero-inflation hole is the decisive safety finding

`pipeline.py:4010`: `zero_adjusted_continuous = dist in ("SkewNormal", "Mixture")` — Mixture **drops
zero rows from training** when `hist_gate > NONZERO_DENOM_GATE (0.05)` and decodes against
`MeanYr_nonzero`. But:

* `_step_decode_predictions` for Mixture explicitly sets `gate_test = gate_validation = None`;
* `_fuse_mixture` passes no gate to `fused_loc`;
* `_step_persist_artifacts` writes no `Gate` column on the Mixture branch;
* `scorecard._pred_cdf_pmf` returns `_mix_cdf(...), np.zeros_like(y)` for `"Mixture"` — no gate term;
* `model_prob._zi_kwargs` returns `{}` for anything that is not SkewNormal/ZINB/ZAGamma.

So on a cell with `zi > 0.05`, a Mixture corner **fits `Y|Y>0`, is scored on all rows including the
zeros, and serves an un-gated CDF with support on negative yardage**. Measured on the production
test frames:

| cell | test-frame zero rate | `mean(Y)` | `mean(Y\|Y>0)` | inflation | implied `g2 z` (bar 0.5) | all-row CITL ratio (g6 fires < 0.97) |
|---|---|---|---|---|---|---|
| NFL receiving yards | 0.172 | 30.10 | 36.36 | **1.208×** | 0.361 **pass** | 1.208 **pass** |
| NFL rushing yards | 0.125 | 34.50 | 39.41 | **1.142×** | 0.245 **pass** | 1.142 **pass** |
| NBA PTS | 0.032 | 12.89 | 13.31 | 1.033× | 0.093 pass | 1.033 pass |

**A +14–21% systematic over-prediction passes Gates 2, 3 and 6 by construction** — Gate 2's σ
denominator launders proportional bias on a high-variance stat (the blind spot `ship_gate.md` itself
documents), and Gate 6's applicable legs are the one-sided *under* legs (the "over" leg is
count/ZINB-only, `_GATE6_OVER_MIN_MEAN` guarded). It surfaces only as live over-betting of the OVER.
Partial mitigation: the book leg pulls the location back at `w < 1` — but `_fuse_mixture` pins
`w = 1` on every non-authentic-quote row.

There is also a *second-order* worry that cuts at the evidence itself: on those cells the fitted
Gaussian components leak probability below zero, and that leakage is what absorbs the zero outcomes
in the PIT. So part of the July receiving-yards `g4 = 0.0552` may be a **support artifact rather
than a tail model**. That should be checked before any Mixture g4 number on a gated cell is trusted.

### 7. Divergence and degeneracy risk under full HPO (Q2)

The three guardrails are the right *kind* of intervention and are theoretically sufficient for
well-posedness — Hathaway (1985, DOI 10.1214/aos/1176349557) proves that constraining the ratio of
component scales to `≥ c > 0` restores a consistent, well-defined constrained MLE for normal
mixtures, whose unconstrained likelihood is unbounded (Day 1969, DOI 10.1093/biomet/56.3.463;
Kiefer & Wolfowitz 1956, DOI 10.1214/aoms/1177728066). Residual risks:

1. **The clamp is loose.** `[0.02, 20]×label_std` implies a scale-**ratio** bound of 1000. Hathaway's
   theory holds for any `c > 0`, but the practical literature is explicit that a non-trivial `c` is
   what actually suppresses spurious maxima (Ingrassia 2004, DOI 10.1007/s10260-004-0092-4;
   Ingrassia & Rocci 2007, DOI 10.1016/j.csda.2006.10.011). *If* the build proceeds: tighten toward
   `σ_min ≥ 0.10 × label_std`, or adopt the Chen–Tan–Zhang (2008, *Statistica Sinica* 18:443–465)
   variance penalty, which yields a consistent penalized MLE.
2. **Weight saturation is unguarded.** `mix.param_dict["mix_prob"] = softmax_fn` on a boosted logit
   pair has no floor; a component driven to `w ≈ 0` stops receiving gradient but is **still served**
   with whatever parameters the last non-degenerate round left. Add a confirm-time diagnostic:
   `min_i min(w₁ᵢ, 1−w₁ᵢ)` and `frac(min(w) < 1e-3)`.
3. **Per-row label inconsistency is the practical form of label switching.** The mixture likelihood
   is invariant to component permutation (Redner & Walker 1984, DOI 10.1137/1026034;
   Stephens 2000, DOI 10.1111/1467-9868.00265), and `tests/golden/test_mixture_training_response.py`
   pins exactly that equivariance — it does **not** pin that component 1 is "the bulk" *uniformly
   across `x`*. If `sign(loc₁ − loc₂)` flips over the feature space, the boosted heads are chasing a
   discontinuous target. Cheap diagnostic: `frac(sign(loc₁ − loc₂) != modal sign)`.
4. **Diagonal-Hessian coupling is the mechanism behind the σ→0/NaN history.** The pinned
   `lightgbmlss 0.6.1` `Mixture` defaults `hessian_mode="individual"` — a per-parameter second
   derivative, i.e. a diagonal approximation to an observed information matrix that is strongly
   non-diagonal when components overlap (the classical reason EM crawls on poorly-separated
   mixtures). When a component's responsibility → 0 or 1 the diagonal entry collapses and the Newton
   step explodes. `lambda_l2 ≥ 1.0` is the correct bound (leaf value `= −Σg/(Σh + λ)`), and
   `min_child_weight ≥ 0.1` backs it. **Untried and cheap: `hessian_mode="grouped"`.**
   Note `stabilization="L2"` divides grad/hess by their RMS over rows — a global per-head rescale.
   With heavy-tailed gradients (which is precisely this family's regime) the RMS is inflated by a
   few rows and the rest of the batch's effective step collapses; `"MAD"` is the robust choice and
   is `gamboostLSS`'s default. Worth one A/B if the build proceeds.
5. **Board→confirm gap mechanisms specific to Mixture.** (i) The board pins fixed small HPs; a
   300-trial full-HPO search explores exactly the deep-tree / low-`min_child_weight` region where the
   collapse lives — the M1 σ-head precedent. (ii) Mixture is **excluded from both confirm-time
   rescue paths**: `pipeline.py:4783` skips the `calibrated` HP-selection closure for Mixture (and
   warns the pin is inert), and `confirm.py:875` excludes Mixture from the g4-only calibrated retry.
   A Mixture nominee that near-misses g4 at confirm simply dies, where a SkewNormal nominee gets two
   attempts. **This asymmetry alone makes a Mixture board leader materially less likely to convert
   than its slack implies.** (iii) Five boosted heads share one round schedule and learning rate —
   the §6.6 "one loss and round schedule shared by all heads" bind; the weight heads need far fewer
   rounds than the location heads. Noncyclical / one-parameter-at-a-time updates are the published
   remedy (Thomas et al. 2018, DOI 10.1007/s11222-017-9754-6).

### 8. Small-sample overfitting at n ≈ 300–2200 (Q3)

The general framework is sound and well-precedented: mixtures as *predictive* densities are the
standard fix for under-dispersion in probabilistic forecasting — BMA is literally a mixture
predictive introduced to repair ensemble under-dispersion (Raftery et al. 2005,
DOI 10.1175/MWR2906.1), and a 2-component mixture EMOS significantly outperforms its single-family
benchmarks on wind speed (Baran & Lerch 2016, DOI 10.1002/env.2380, arXiv:1507.06517).
Mixture heads are first-class in the LSS framework this repo runs (März & Kneib 2022,
arXiv:2204.00778; the library's own `Mixture` docstring cites Bishop 1994, Aston NCRG/94/004 and
Jang, Gu & Poole 2017 for the Gumbel-softmax it replaces).

The transfer caveat is sample size and mode collapse. Those forecasting results sit at
n ≈ 10³–10⁵ *independent* training pairs. Here:

* **NBA FGA** (n 1873, 355 players) and **NFL receiving yards** (2238, 382) are a defensible regime.
* **NFL passing yards is not**: n = 369 with **60 unique players**. Estimating a rare-component
  weight of ~0.16 there is ~59 effective observations, clustered. And it is precisely the cell whose
  "+0.100 slack, only family ever to pass all six" is the strongest board evidence — the textbook
  small-n selection-optimism trap. Note also that the "is the second component real?" question has
  **non-standard asymptotics**: the naive LRT for M=1 vs M=2 is invalid at the boundary; use the
  modified LRT (Chen, Chen & Kalbfleisch 2001, DOI 10.1111/1467-9868.00273) or a parametric
  bootstrap (McLachlan 1987, DOI 10.2307/2347790). This is the same class of trap as the Vuong
  misuse the house already guards against.
* Mode collapse under NLL is a documented, general MDN failure — not a torch-version accident
  (Makansi et al. 2019, arXiv:1906.03631; Hjorth & Nabney 1999, DOI 10.1049/cp:19991120). The repo's
  guardrails address the *explosion* direction; they do not address a component that quietly dies.

### 9. Ground truth: the board says the addressable set is empty (or unsafe)

Withheld cohort as of today: **19 cells, 14 continuous-class**, of which 8 have `zi ≤ 0.02`.
Near-term shippable (excluding D1/D2-activation-gated MLB/NHL): NBA FGA, NFL passing yards,
WNBA MIN, WNBA PRA, WNBA fantasy points prizepicks — plus the two gated NFL yards cells.
Live board, cross-fit validation rows, swept 2026-08-22 → 2026-08-26:

| cell | rows | best slack | best family | `g4 / bar` | shipping corners | gate fail-rate across corners |
|---|---|---|---|---|---|---|
| NBA FGA | 29 | +0.053 | **DPO** | 0.0283 / 0.050 | **4** (disc. +0.053) | g1 .86 · g4 .41 · g6 .48 |
| WNBA MIN | 29 | +0.076 | SkewNormal | 0.0266 / 0.050 | **23** (disc. +0.059) | g4 .14 |
| WNBA REB | 48 | +0.032 | SkewNormal | 0.0315 / 0.050 | **14** (disc. +0.032) | g4 .17 |
| WNBA PTS | 45 | +0.012 | ZINB | 0.0459 / 0.050 | **9** | g4 .51 · g6 .47 |
| WNBA PRA | 49 | +0.022 | SkewNormal | 0.0363 / 0.050 | **7** (disc. +0.022) | g4 .20 |
| NFL rushing yards | 49 | +0.060 | SkewNormal | 0.0470 / 0.050 | 4 — **all disc. −0.032** | **g4 .78 · g6 .71** |
| NFL receiving yards | 15 | −0.120 | SkewNormal | 0.0560 / 0.050 | **0** | **g4 .79 · g6 .57** · g1 .43 |
| NFL passing yards | 17 | −0.200 | SkewNormal | 0.0438 / 0.074 | **0** | **g1 1.00** · g4 .18 |
| WNBA fp prizepicks | 41 | −1.000 | SkewNormal | 0.0144 / 0.050 | 0 | **g2 .59** · g1 .85 |

Reading it against the routing rule "spend a family lever only on a cell whose binding gate is g4":

* **NBA FGA, WNBA MIN/REB/PRA/PTS** — g4 is already cleared by confirmable families. No lever needed.
* **NFL passing yards** — g1-bound on 100% of corners. **No family can fix g1.** The July
  "+0.100, both gate-passing corners were Mixture" is superseded evidence; the archive repair
  (§3.2) moved this cell's book leg since then.
* **WNBA fantasy points prizepicks** — g2/g1-bound: a location problem, not a shape problem.
* **NFL receiving + rushing yards** — genuinely g4-bound. **These are the only two cells Mixture
  could help, and both carry `zi ≈ 0.12` — the exact cohort where the family has no zero gate.**
  Both also co-fail g6, which a shape lever does not move.

---

## Recommendation

### KILL the serve build now. Hold Mixture at `_RESEARCH` capabilities.

Not because the family is unsound — because the addressable set has emptied out from under it, and
the two cells that remain are both (a) unsafe under the current implementation and (b) beyond the
family's measured ceiling. §8.2 #0a should be **closed as superseded**, not left open.

Concretely, on the terms hole #0a was written:

> "on the first holdout-blind run it was the rank-1 corner on 3 of the 4 cells searched — NFL passing
> tds +0.202, NBA FGA +0.150, NFL passing yards +0.100"

* NFL passing tds **shipped** (`NegBin` / `isotonic_mean` / `pit_ks`, 2026-08-25 ledger).
* NBA FGA has **4 board-shipping DPO corners** at `discounted_slack +0.053`.
* NFL passing yards is **g1-walled on 100% of 17 corners** — outside any family lever's reach.

### Pre-registered reopen trigger (mechanical, evaluable by the WS-0 monthly sweep)

Reopen the serve build when a single cell satisfies **all four**:

1. **g4 is the sole binding gate** across the cell's board corners (g1/g2/g3/g5/g6 all pass on the
   best corner); and
2. the cell has **no shipping corner with positive `discounted_slack`** on a confirmable family; and
3. **`g4_pit_ks / g4_pit_ks_max ≤ 1.20`** on the best corner (i.e. the excess is inside the ~20% a
   mixture has actually been observed to close); and
4. either **`zi ≤ 0.02`**, or the zero-gate work in the build list below is funded.

Two cells at once ⇒ the build clears its cost. One cell ⇒ prefer a **StudentT LSS head** (Finding 1)
as the cheaper first escalation; it is 3 parameters, closed-form, already in the venv, and blends
through machinery that exists.

### No-regret items (do these regardless — total ≈ 0.75 session)

1. **Fail-fast on a Mixture cell in `stat_meta`.** Add a single `book_family(dist)` helper in
   `helpers/config.py` (Mixture → `"Normal"`, everything else identity) and either call it at the
   ~8 book-leg boundaries, or — cheaper, and sufficient today — raise a clear error in
   `ship_config._validate_cell` when a cell's `dist` is a family with no book-leg mapping. Today a
   one-line hand edit to `stat_meta.json` silently arms a `TypeError` in **confer**, **close-lines**
   and **matrix regen**. Pin with a golden that loops every `dist` value in `stat_meta` through
   `get_ev(line, 0.5, cv, dist=dist)`.
2. **Fix `_scale_mixture` to scale about the mixture mean** (Finding 3) even while the family is
   research-only, so board `dispersion_cal` values are comparable across families and future
   evidence is not measured through a saturating control.
3. **Adopt the cross-fit residual-shape screen as the standing family-headroom diagnostic.** It runs
   in seconds per cell from an existing test dump, needs no training, and directly answers "would a
   heavier-tail family close this cell's g4 gap?" — for `{as-is, GMM M=2, GMM M=3, StudentT}` under a
   player-disjoint split. This is what makes the reopen trigger evaluable without spending board
   budget on a family that cannot confirm.

### If the build is funded anyway — the complete list (the dispatch's list is ~60% of it)

| # | Item | Why the dispatch missed it |
|---|---|---|
| 1 | `_decode_mixture` in `model_prob.py` mirroring `_decode_mixture_frame` | named |
| 2 | Blend: location-shift both components by pooled − model; compute the delta **after** `_sanitize_model_ev` | named; ordering is new |
| 3 | `dispersion_cal` scaling — **about the mean**, not component scales | named, but the form is wrong |
| 4 | Book leg → `"Normal"` **via a shared helper at all ~8 sites**, not just `model_prob` | named as one site; it is eight |
| 5 | **`_serve_offset_mode` must stop hardcoding SkewNormal** | not named — `[[serve_decode_drift_offset_mode]]` class, hits NBA FGA |
| 6 | **Zero gate**: either build it into decode/fuse/persist/scorecard/serve, or add an applicability predicate `hist_gate ≤ GATE_PUBLISH_THRESHOLD` | not named — this is the safety-critical one |
| 7 | `_model_predictive_sd` Mixture branch (`_mixture_moments` sd) | not named |
| 8 | `_annotate_display_shape` Mixture branch (dashboard `Model Param` / `Projection STD`) | not named |
| 9 | `correct_fused_mean` post-fusion re-shift on the serve side (training already does it, `pipeline.py:3007`) | not named |
| 10 | Live-path integration test mirroring `test_dpo_live_path.py` (§7.3 hard ship gate) | implied |
| 11 | Golden pinning the `step` argument parity between training's Mixture `get_odds` and serve's | not named |
| 12 | Confirm-time degeneracy diagnostics: min component weight, `sign(loc₁−loc₂)` consistency, component-scale ratio | not named |

**Build cost: ~3.5–4 sessions + 1 pilot session.** This is an **engineering project**, not a research
project — the training half is landed and the method is known — but item 6 alone is a genuine
modelling decision (how does a Gaussian mixture represent a zero atom?), which pushes part of it
back into research.

### The ZINB/NegBin identity merge — yes, defer it (Q5)

It is a **refactor coupling, not a correctness dependency**. The serving/blend layer collapses
`NegBin` and `ZINB` into one branch (`dist in ("NegBin","ZINB")` at `_blend_with_book`,
`_model_over_and_push`, `_dispersion_calibrate`, `_annotate_display_shape`, `_model_predictive_sd`)
while the registry carries them as two specs with distinct persisted controls. Adding a sixth
serving family is the natural moment to consolidate — which is exactly the plan's own **S5**
machinery-simplification item ("consolidate the per-family serve branches … into one dispatch table
in `helpers/distributions` … family #7 becomes one registry entry plus its torch class"), explicitly
scheduled **post-campaign** under a "nothing lands mid-sweep" rule. A Mixture serve build is purely
additive beside the existing branches; deferring costs one extra `elif` at each of ~6 sites.

**The one thing that must not be deferred is item 4** — the book-leg family map is not a refactor,
it is a live `TypeError`. Implementing it as a single shared helper is a down payment on S5 and
closes the crash class in one place.

---

## Reality checks

* **The July board numbers are not evidence about today's board.** The archive repair (§3.2) moved
  the book leg on the NFL cells, `_authentic_quote` gating changed
  (`[[archive_under_prob_written_gated]]`), and the mean corrector moved post-fusion
  (2026-08-25). `[[corner_verdicts_are_not_code_scoped]]` is exactly this hazard. NFL passing
  yards going from "both gate-passing corners were Mixture" to "g1 fails on 100% of 17 corners"
  is the observed instance.
* **A `ships=True` board row is a candidate flag, not a ship.** Several "shipping corners" above
  carry negative `discounted_slack` (NFL rushing yards −0.032 on all four; WNBA PRA's
  `cdf_recal_isotonic` −0.060). `[[crossfit_board_ships_optimistic]]`,
  `[[board_confirm_gap_root_cause]]`. My statement that "the addressable set has emptied" is
  strongest for NBA FGA / WNBA MIN / WNBA REB and weakest for NFL rushing yards.
* **A board sweep was running while I read the board.** Rows are dated 2026-08-22 → 2026-08-26
  (receiving yards was swept today). Re-derive the table from §3 before acting.
* **My residual screen is a marginal, cross-fit proxy, not a conditional fit.** It fits one global
  mixture to standardized residuals; the real LightGBMLSS head fits per-row parameters and can beat
  it. It is a *lower* bound on conditional headroom and an *upper* bound on what a global shape
  family can do. It agrees with the repo's own oracle sweep to within ~0.03 KS on receiving yards,
  which is the reason to trust its direction, not its third decimal.
* **The StudentT recommendation is a bet, not a result.** It rests on a marginal-residual cross-fit
  on 5 cells with stale served parameters. It is cheap enough to test (one registry entry + the
  existing SkewNormal serve machinery), which is the whole argument for testing it *before*
  funding the mixture serve path — not for shipping it on this evidence.
* **What would make this brief wrong.** (a) NFL rushing yards fails its full-HPO confirm on every
  corner and re-enters the board as a clean g4-only cell with `g4/bar ≈ 1.05` — then trigger
  conditions 1–3 fire and the build is back on the table (condition 4 still needs the zero gate).
  (b) The WNBA board's shipping corners fail to convert at confirm, re-opening 3–5 cells at once.
  (c) The g6 co-failure on receiving/rushing yards turns out to be an artifact of the served mean
  (per `[[gate_off_probe_confounds_mean]]` — pin the mean before attributing), leaving those cells
  g4-only after all.

---

## Open questions / caveats

1. **Is the July receiving-yards `g4 = 0.0552` a tail model or a support artifact?** On a `zi = 0.127`
   cell with no gate, the fitted Gaussian components leak mass below zero and that leakage absorbs
   the zero outcomes in the PIT. Until that is decomposed (`F_mix(0)` per row vs the empirical zero
   rate), no Mixture g4 number on a gated cell should be trusted. Cheap: it needs one board corner
   re-run with the mixture params dumped.
2. **Does a StudentT LSS head clear g4 where the global StudentT oracle did not?** The in-repo
   NO-GO measured a global hand-fit shape; a per-row conditional scale is a different object and my
   cross-fit says it may be the better of the two escalations on 3 of 5 cells. Resolving this
   reorders the whole §6.6 continuous escalation ladder (it would also bear on the SHASH ask).
3. **Is `hessian_mode="grouped"` better than the default `"individual"` for this family?** Never
   tried in-repo; it changes the curvature approximation from diagonal-per-parameter to
   joint-per-parameter-type, which is the coupling that drives mixture instability.
4. **Should Mixture's exclusion from the `calibrated` HP-selection closure and the g4-only retry be
   fixed before, not after, a pilot?** As it stands a Mixture nominee has strictly fewer confirm-time
   rescue paths than a SkewNormal nominee, which biases every board-vs-confirm comparison against
   it. If Mixture is ever re-piloted, either build the closure or record the handicap explicitly.
5. **Does `stabilization="L2"` hurt a heavy-tailed-gradient head?** L2 divides by the RMS over rows;
   MAD is `gamboostLSS`'s robust default. §8.2 #9(c) says promote `--stabilization` to a swept axis
   only if MAD/L2 ever wins on ≥1 cell — this family is the natural first test.
6. **Does the `dist: "Mixture"` crash class have siblings?** The audit here was Mixture-specific.
   Any future family added to `stat_meta` before its book-leg map exists reproduces it exactly; a
   generic golden over every `stat_meta` `dist` value is the durable fix.

---

## Bibliography

| # | Source | Identifier | Used for |
|---|---|---|---|
| 1 | Hathaway, R.J. (1985). A constrained formulation of maximum-likelihood estimation for normal mixture distributions. *Ann. Statist.* 13(2):795–800 | DOI 10.1214/aos/1176349557 | Scale-ratio constraint restores a consistent MLE; the repo's clamp is "Hathaway-style" but loose |
| 2 | Day, N.E. (1969). Estimating the components of a mixture of normal distributions. *Biometrika* 56(3):463–474 | DOI 10.1093/biomet/56.3.463 | Unbounded normal-mixture likelihood / degenerate maximizer |
| 3 | Kiefer, J. & Wolfowitz, J. (1956). Consistency of the MLE in the presence of infinitely many incidental parameters. *Ann. Math. Statist.* 27(4):887–906 | DOI 10.1214/aoms/1177728066 | Foundational unboundedness/consistency result |
| 4 | Ingrassia, S. (2004). A likelihood-based constrained algorithm for multivariate normal mixture models. *Stat. Methods Appl.* 13:151–166 | DOI 10.1007/s10260-004-0092-4 | Spurious maxima; a non-trivial ratio bound is what suppresses them in practice |
| 5 | Ingrassia, S. & Rocci, R. (2007). Constrained monotone EM algorithms for finite mixture of multivariate Gaussians. *CSDA* 51(11):5339–5351 | DOI 10.1016/j.csda.2006.10.011 | Constrained estimation procedures |
| 6 | Chen, J., Tan, X. & Zhang, R. (2008). Inference for normal mixtures in mean and variance. *Statistica Sinica* 18:443–465 | — (vol/pages) | Penalized likelihood on component variances; consistent penalized MLE |
| 7 | Redner, R.A. & Walker, H.F. (1984). Mixture densities, maximum likelihood and the EM algorithm. *SIAM Review* 26(2):195–239 | DOI 10.1137/1026034 | Identifiability only up to permutation |
| 8 | Stephens, M. (2000). Dealing with label switching in mixture models. *JRSS-B* 62(4):795–809 | DOI 10.1111/1467-9868.00265 | Label switching; predictive density is permutation-invariant, the heads are not |
| 9 | Chen, H., Chen, J. & Kalbfleisch, J.D. (2001). A modified likelihood ratio test for homogeneity in finite mixture models. *JRSS-B* 63(1):19–29 | DOI 10.1111/1467-9868.00273 | Valid M=1 vs M=2 test; the naive LRT is invalid at the boundary |
| 10 | McLachlan, G.J. (1987). On bootstrapping the likelihood ratio test statistic for the number of components in a normal mixture. *JRSS-C* 36(3):318–324 | DOI 10.2307/2347790 | Parametric-bootstrap alternative for the component-count test |
| 11 | McLachlan, G.J. & Peel, D. (2000). *Finite Mixture Models*. Wiley | DOI 10.1002/0471721182 | Standard reference; spurious local maximizers (§3.9) |
| 12 | Frühwirth-Schnatter, S. (2006). *Finite Mixture and Markov Switching Models*. Springer | DOI 10.1007/978-0-387-35768-3 | Identifiability, label switching, practical estimation |
| 13 | Bishop, C.M. (1994). Mixture density networks. Aston Univ. Tech. Report NCRG/94/004 | NCRG/94/004 | The MDN construction the pinned `lightgbmlss.Mixture` cites |
| 14 | Makansi, O. et al. (2019). Overcoming limitations of mixture density networks. *CVPR* | arXiv:1906.03631 | Mode collapse and training instability of NLL-trained MDNs |
| 15 | Hjorth, L.U. & Nabney, I.T. (1999). Regularisation of mixture density networks. *ICANN* | DOI 10.1049/cp:19991120 | Variance-collapse regularization for MDNs |
| 16 | Jang, E., Gu, S. & Poole, B. (2017). Categorical reparameterization with Gumbel-softmax. *ICLR* | arXiv:1611.01144 | The stochastic weight response the repo replaces with `softmax_fn` |
| 17 | März, A. & Kneib, T. (2022). Distributional Gradient Boosting Machines | arXiv:2204.00778 | The LSS framework; mixture densities as first-class heads |
| 18 | März, A. (2019). XGBoostLSS — an extension of XGBoost to probabilistic forecasting | arXiv:1907.03178 | Predecessor of the pinned library |
| 19 | Rigby, R.A. & Stasinopoulos, D.M. (2005). GAMLSS. *JRSS-C* 54(3):507–554 | DOI 10.1111/j.1467-9876.2005.00510.x | Distributional-regression foundation |
| 20 | Mayr, A. et al. (2012). GAMLSS for high-dimensional data via boosting. *JRSS-C* 61(3):403–427 | DOI 10.1111/j.1467-9876.2011.01033.x | Boosted distributional regression; the MAD/L2 stabilization lineage |
| 21 | Thomas, J. et al. (2018). Gradient boosting for distributional regression: … noncyclical updates. *Stat. Comput.* 28:673–687 | DOI 10.1007/s11222-017-9754-6 | Shared round schedule across heads is a known defect; noncyclical updates are the remedy |
| 22 | Genest, C. & Zidek, J.V. (1986). Combining probability distributions: a critique and an annotated bibliography. *Statist. Sci.* 1(1):114–135 | DOI 10.1214/ss/1177013825 | LinOP vs LogOP; the operator `fused_loc` implements |
| 23 | Ranjan, R. & Gneiting, T. (2010). Combining probability forecasts. *JRSS-B* 72(1):71–91 | DOI 10.1111/j.1467-9868.2009.00726.x | A linear pool of calibrated forecasts is over-dispersed; recalibrate the combination |
| 24 | Gneiting, T. & Ranjan, R. (2013). Combining predictive distributions. *EJS* 7:1747–1782 | DOI 10.1214/13-EJS823 | Spread-adjusted / beta-transformed linear pools; dispersion of pooled predictives |
| 25 | Hinton, G.E. (2002). Training products of experts by minimizing contrastive divergence. *Neural Comput.* 14(8):1771–1800 | DOI 10.1162/089976602760128018 | Gaussian product-of-experts = precision-weighted pool |
| 26 | Alspach, D.L. & Sorenson, H.W. (1972). Nonlinear Bayesian estimation using Gaussian sum approximations. *IEEE TAC* 17(4):439–448 | DOI 10.1109/TAC.1972.1100034 | Gaussian-mixture × Gaussian is a Gaussian mixture: the correct mixture-book pool |
| 27 | Raftery, A.E., Gneiting, T., Balabdaoui, F. & Polakowski, M. (2005). Using BMA to calibrate forecast ensembles. *MWR* 133(5):1155–1174 | DOI 10.1175/MWR2906.1 | A mixture predictive introduced specifically to repair under-dispersion |
| 28 | Baran, S. & Lerch, S. (2016). Mixture EMOS model for calibrating ensemble forecasts of wind speed. *Environmetrics* 27(2):116–130 | DOI 10.1002/env.2380 · arXiv:1507.06517 | 2-component mixture significantly beats single-family EMOS on calibration |
| 29 | Gneiting, T., Balabdaoui, F. & Raftery, A.E. (2007). Probabilistic forecasts, calibration and sharpness. *JRSS-B* 69(2):243–268 | DOI 10.1111/j.1467-9868.2007.00587.x | Maximize sharpness subject to calibration; the PIT paradigm Gate 4 implements |
| 30 | Gneiting, T. & Raftery, A.E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA* 102(477):359–378 | DOI 10.1198/016214506000001437 | Proper scoring; already in the repo's citation base |
| 31 | Brockwell, A.E. (2007). Universal residuals: a multivariate transformation. *Statist. Probab. Lett.* 77(2):143–147 | DOI 10.1016/j.spl.2007.02.008 | Randomized PIT — Gate 4's statistic |

**In-repo evidence read for this brief (not literature):**
`docs/handoffs/model_improvement_track.md` §3/§6.6/§7.1/§7.3/§7.4/§8.1/§8.2/§10 ·
`docs/ship_gate.md` · `src/sportstradamus/training/pipeline.py`
(`_continuous_dist_obj`, `_step_select_distribution`, `_decode_mixture_frame`, `_shift_mixture`,
`_scale_mixture`, `_mixture_moments`, `_fuse_mixture`, `_calibrate_mixture_dispersion`,
`_step_compute_test_probabilities`, `_step_calibrate_temperature`, `_step_persist_artifacts`,
`_step_correct_fused_mean`) · `src/sportstradamus/training/scorecard.py`
(`_pred_cdf_pmf`, `_decode_mix_params`, `_mix_cdf`, `_randomized_pit_ks`) ·
`src/sportstradamus/helpers/distributions.py` (`get_ev`, `get_odds`, `get_push_prob`, `fused_loc`,
`_mixnorm_*`, `_mixture_start_values`, `set_model_start_values`) ·
`src/sportstradamus/prediction/model_prob.py` (`_serve_offset_mode`, `_build_prob_params`,
`_decode_model_params`, `_book_cell_params`, `_book_over_prob`, `_blend_with_book`,
`_model_predictive_sd`, `_sanitize_model_ev`, `_dispersion_calibrate`, `_serving_dispersion_cal`,
`_model_over_and_push`) · `src/sportstradamus/training/model_strategy/{specs,registry,confirm}.py` ·
`src/sportstradamus/helpers/{config,archive,training_quotes}.py` · `moneylines.py` ·
`.venv/…/lightgbmlss/distributions/{Mixture,mixture_distribution_utils,distribution_utils}.py` (0.6.1) ·
`data/research/strategy_research_board.csv` (swept 2026-08-22 → 2026-08-26) ·
`data/training/model_stats.parquet` · `data/config/{stat_meta,stat_calibration}.json` ·
`data/test_sets/{NFL_receiving-yards,NFL_rushing-yards,NFL_passing-yards,NBA_FGA,NBA_PTS}.csv` ·
memories `[[cdf_recal_nonstationary_pit]]`, `[[serve_decode_drift_offset_mode]]`,
`[[dist_selection_forceable_via_stat_meta]]`, `[[crossfit_board_ships_optimistic]]`,
`[[board_confirm_gap_root_cause]]`, `[[corner_verdicts_are_not_code_scoped]]`,
`[[mean_corrector_belongs_post_fusion]]`, `[[nfl_volume_cells_feature_mature]]`,
`[[gate_off_probe_confounds_mean]]`, `[[prophecize_runtime_lane]]`.
