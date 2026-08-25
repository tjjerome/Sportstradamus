# In-repo research brief — closing the Gate-4 gap on low-mean count cells

**Question:** the g4-failing ZINB/NegBin/DPO cohort over-predicts P(Y=0) by +0.03…+0.11, the
post-hoc dispersion scalar `c` pins at its bound, and an offline `(c, g)` gate-shift probe closes
much of the gap. Is the zero-inflation vestigial, are these cells conditionally under-dispersed,
which family should replace ZINB, how should a load-bearing gate be recalibrated, and what is the
smallest experiment that decides it?

**Date:** 2026-08-24. **Scope:** read-only. No production file, model pickle, `stat_meta.json`
entry, or git ref was touched. All new numbers below were computed from the committed test-set
dumps in `src/sportstradamus/data/test_sets/` and the committed ledger
`research/confirm_nominee_gates.csv` using the production decode
(`training.scorecard._pred_cdf_pmf`, `training.pipeline._count_pit_frame`). Probe scripts:
`/tmp/claude-1000/-home-trevor-Sportstradamus/318e838c-2de7-4af1-9434-68d82047eddf/scratchpad/rb/`.

---

## TL;DR

- **The gate-shift result is confounded and the "vestigial zero-inflation" reading is mostly
  wrong.** Dropping or shrinking π rescales the served mean by `1/(1−π)`. A probe that holds the
  served mean fixed and moves *only* the gate buys 0.005–0.009 KS. A probe that moves *only* the
  mean (gate and shape untouched) takes **all four** g4-failing ZINB cells below their Gate-4
  thresholds, and adding the gate on top of a free mean buys ≈ 0.000. On NFL interceptions:
  `c` alone 0.1133 → `(c, gate)` 0.1044 → **mean-only 0.0620** (threshold 0.0698).
- **Three distinct pathologies, not one.** Measured conditional dispersion
  `φ̂ = (Var(y) − Var(m))/mean(y)` with the Cameron–Trivedi t-statistic: **NBA PF is genuinely
  sub-Poisson** (φ̂ 0.78, t −8.70) — structurally unreachable by NegBin/ZINB at any parameter;
  **NFL tds and NFL interceptions are equidispersed with a stationary ~26 % predictive-mean
  deficit**; **MLB runs-allowed is over-dispersed** (φ̂ 1.57, t +9.44) with a mean slope of 0.09.
  A single "count Gate-4 lever" cannot exist.
- **The lever this cohort needs already exists, is already a swept axis, and is structurally
  disabled on gated families.** `posthoc: roe_mean` / `isotonic_mean` (§6.1 Rung A) is fit on the
  ZINB **base** mean `r·p/(1−p)` against the zero-**inclusive** `Result`, and the gate is then
  applied a second time downstream. Simulated on real dumps with the exact production contract:
  on NHL goals (π̄ 0.79) it drives the served CITL from 0.997 to **0.260**; on NBA BLK (π̄ 0.53),
  0.972 → **0.441**. Gate-aware refitting turns the same corrector into the fix (NFL tds 0.762 →
  1.056). No live cell is currently harmed — only 1 of 32 gated cells carries a mean posthoc and
  it is `withheld` — so this is a **suppressed lever, not an active bug**.
- **Family verdict:** keep DPO, retire *joint* ZINB, keep *hurdle* ZINB for genuinely inflated
  cells, and **do not build CMP**. A ZI component can only *add* zeros (π ≥ 0); a hurdle can also
  express zero *deflation* (Mullahy 1986; Feng 2021 doi:10.1186/s40488-021-00121-4), which is the
  direction six of the nine probed cells actually need. The repo's own 2026-05 ZTNB refutation
  (`q < NB(0)` on ~65 % of FG3M rows) is the same finding observed from the other side.
- **Ship arithmetic — two convertible cells, not nine.** NFL tds and NBA PF are *pass-all-but-g4*
  on the ledger with a real Gate-1 win (`ci_hi` −0.021 and −0.108…−0.188). NBA TOV (2/7 g1 pass)
  and NHL goalsAgainst (1/10) are Gate-1 walled — everything there is correctness only.
  **The dispatch's premise that NFL interceptions fails Gate 1 is contradicted by the four most
  recent ledger rows** (g1 pass, `ci_hi` ≈ −0.077, BSS +0.24…+0.30); it fails g4 and **g6** (0/10).
  Run **one** experiment first: NFL tds, `--dist NegBin --posthoc roe_mean`, ~35 min.

---

## Key findings

### 1. Killing the ZINB gate is a mean rescale in disguise; the mean is the parameter doing the work

The served ZINB predictive is `π·δ₀ + (1−π)·NB(μ)` with `E[Y] = (1−π)·μ`
(`pipeline._zero_inflated_outcome_mean`, `scorecard._zero_inflated_mean`, and
`scorecard._pred_cdf_pmf` all agree). Forcing `π → 0` therefore multiplies the served mean by
`1/(1−π)` — a factor of 1.02 on NBA PF (π̄ 0.017) and **4.76 on NHL goals** (π̄ 0.79). That single
fact reproduces the entire "low-π cells love it / high-π cells are destroyed by it" split in the
local diagnostic without any appeal to zero-inflation.

I re-ran the probe with the confound removed: `k` scales the **zero-inclusive** served mean, `g`
shifts the gate in logit space **at fixed served mean**, `c` scales the family shape. Fit on one
random half, scored on the disjoint half, median of 6 seeded splits
(`rb/mean_vs_gate.py`; held-out randomized-PIT KS via `scorecard._randomized_pit_ks`):

| cell | π̄ | CITL | thr | as-shipped | `c` only | `(c,g)` @ fixed mean | **`k` only** | `(c,k)` | `(c,g,k)` | fitted `k` |
|---|---|---|---|---|---|---|---|---|---|---|
| NFL interceptions | 0.135 | 0.738 | 0.0698 | 0.1199 | 0.1133 | 0.1044 | **0.0620** ✓ | 0.0605 ✓ | 0.0605 ✓ | 1.48 |
| NBA PF | 0.017 | 0.953 | 0.0500 | 0.0680 | 0.0634 | 0.0620 | **0.0437** ✓ | 0.0382 ✓ | 0.0339 ✓ | 1.07 |
| NFL tds | 0.414 | 0.737 | 0.0500 | 0.0553 | 0.0533 | 0.0462 | **0.0299** ✓ | 0.0287 ✓ | 0.0289 ✓ | 1.51 |
| NBA TOV | 0.077 | 0.933 | 0.0500 | 0.0606 | 0.0577 | 0.0528 | **0.0265** ✓ | 0.0233 ✓ | 0.0241 ✓ | 1.09 |
| MLB runs-allowed | 0.060 | 0.953 | 0.0500 | 0.0529 | 0.0562 | 0.0597 | **0.0365** ✓ | 0.0430 | 0.0430 | 1.06 |
| NBA BLK (passes) | 0.527 | 1.036 | 0.0500 | 0.0463 | 0.0415 | 0.0330 | 0.0398 | 0.0388 | 0.0340 | 1.16 |
| WNBA FTM (passes) | 0.369 | 1.156 | 0.0500 | 0.0465 | 0.0425 | 0.0428 | 0.0470 | 0.0362 | 0.0363 | 0.96 |
| NHL goals (passes) | 0.790 | 0.961 | 0.0500 | 0.0366 | 0.0349 | 0.0109 | 0.0227 | 0.0219 | 0.0090 | 1.31 |
| NBA OREB (control) | 0.149 | 1.062 | 0.0500 | 0.0360 | 0.0321 | 0.0294 | 0.0410 | 0.0355 | 0.0349 | 0.98 |

Read the last four columns together. With the mean free, the gate parameter contributes
0.000–0.006 on every cell except NHL goals (π̄ 0.79, where π genuinely *is* a shape parameter of
the positive part) — and NHL goals already passes Gate 4 by 0.013. **The zero-inflation weight is
not the binding parameter on any failing cell.**

Corroborating: `CITL = Σ(1−π)μ / ΣResult` is 0.737 on NFL tds and 0.738 on NFL interceptions —
the served predictive mean is 26 % below the realized outcome mean. That is also visible in the
gate rows: the two NFL tds *NegBin* arms recorded in `count_dispersion_flip.md` KILL with
`g6_citl_ci_hi` 0.783 / 0.805, i.e. Gate 6's calibration-in-the-large leg is measuring the same
defect from a different angle.

### 2. Why the mean bias is invisible to Gate 1 but fatal to Gate 4

The served over-probability `P` is `get_odds(Line, weighted_mean, dist, r, gate)` **followed by a
temperature fit and an optional `prob_recal_*` corrector**
(`pipeline._step_compute_test_probabilities` → `_step_calibrate_temperature`). Temperature is a
one-parameter logit scaler fit at the quoted line, so it absorbs a mean bias *at that single
threshold* and nowhere else. Measured on NFL tds: empirical over-rate 0.197 vs served `P` 0.177 —
a 2-point miss at the line — while the predictive mean is 26 % low.

This is the textbook separation between **threshold/exceedance calibration** and **probabilistic
calibration** (Gneiting, Balabdaoui & Raftery 2007, doi:10.1111/j.1467-9868.2007.00587.x;
Gneiting & Ranjan 2013, doi:10.1214/13-EJS823): a forecast can be calibrated at one exceedance
threshold and badly miscalibrated as a distribution. Gate 4 is the probabilistic-calibration test
and Gate 1 is (via temperature) close to a threshold-calibration test, so **the architecture
guarantees this failure mode exists** and Gate 4 is doing exactly the job it was designed for.

Practical consequence for the risk assessment: lifting the mean should improve g4 and g6 while
being roughly *neutral* on g1/g5, because the temperature refit will simply move closer to 1. That
is an unusually favourable risk profile for a calibration change — but it is a prediction, and
E1 below is designed to falsify it.

### 3. NBA PF is genuinely conditionally under-dispersed; NegBin and ZINB cannot represent it

**Statistic.** The right primary read is the variance-decomposition dispersion index at the served
conditional mean `m`:

```
φ̂ = ( Var(y) − Var(m) ) / mean(y)          [ = 1 under conditional Poisson ]
```

from `Var(Y) = E[Var(Y|X)] + Var(E[Y|X])`. Its bias direction is what makes it usable here: GBT
leaf-averaging attenuates `m`, so `Var(m)` under-states `Var(E[Y|X])` and **φ̂ is biased upward**.
`φ̂ < 1` is therefore *conservative* evidence of genuine sub-Poisson conditional variance — mean
misspecification cannot manufacture it, only mask it.

The **secondary** read is the Cameron & Trivedi (1990, doi:10.1016/0304-4076(90)90014-K)
regression-based test: regress `((yᵢ − mᵢ)² − yᵢ)/mᵢ` on a constant; the t-statistic is
asymptotically N(0,1) under equidispersion and is genuinely two-sided, so it detects
under-dispersion. It requires only the mean–variance relation under the alternative, not a full
distribution. (Dean & Lawless 1989, doi:10.1080/01621459.1989.10478792 is the score-test sibling;
Dunn & Smyth 1996, doi:10.1080/10618600.1996.10474708 randomized quantile residuals are the same
information in the idiom Stage B1.5 already used in this repo.)

**Do not use the raw Pearson dispersion `mean((y−m)²/m)` alone.** `E[(y−m)²] = Var(Y|x) + bias²`,
so any mean error inflates it, and at low `m` the `1/m` weighting is dominated by near-zero rows.
NBA PF is the demonstration: raw Pearson **1.31** (reads over-dispersed) vs isotonic-mean-corrected
Pearson **0.81** vs variance-decomposition **0.78**. Same data, opposite verdict.

Measured on the g4-relevant cells (`rb/cond_disp.py`, n as shown, `_iso` = after an in-sample
isotonic mean recalibration, `floor` = the minimum dispersion index attainable by *any*
distribution on ℤ≥0 at these means):

| cell | family | n | CITL | slope | φ̂ | φ̂ᵢₛₒ | CT t | CT tᵢₛₒ | Pearson raw | floor | reading |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NBA PF | ZINB (joint) | 2250 | 0.953 | 0.873 | 0.767 | 0.780 | +0.45 | **−8.70** | 1.310 | 0.083 | **sub-Poisson** |
| NBA TOV | ZINB (joint) | 2196 | 0.933 | 1.032 | 0.955 | 0.902 | +1.62 | −1.76 | 1.648 | 0.125 | mildly sub-Poisson |
| NFL tds | ZINB (joint) | 2466 | 0.737 | 1.394 | 1.114 | 1.029 | +1.20 | −0.32 | 1.319 | 0.733 | equidispersed |
| NFL interceptions | ZINB (joint) | 378 | 0.738 | −0.166 | 1.045 | 1.032 | +1.39 | +0.30 | — | 0.464 | equidispersed |
| MLB runs-allowed | ZINB (hurdle) | 1101 | 0.953 | 0.090 | 1.453 | 1.568 | +9.73 | **+9.44** | 1.815 | 0.068 | **over-dispersed** |
| NBA OREB (control) | ZINB (joint) | 2132 | 1.062 | 0.782 | 0.793 | 0.864 | +1.37 | −3.15 | 1.270 | 0.177 | sub-Poisson, passing |

A NegBin has `Var = μ + μ²/r ≥ μ` for every `r`, so no NegBin or ZINB body can reach φ = 0.78.
**The `c → 10` bound-pinning reported in the local diagnostic is the signature of exactly this**:
`c` scales `r`, `r → ∞` *is* the Poisson limit, so a pinned `c` means the fit wants a variance the
family's floor forbids. That is a free, already-computed mis-family detector.

**Cohort scale.** 25 of 46 count cells read φ̂ᵢₛₒ < 1. The strongest, with their current family:
NHL sogBS 0.65 / t −22.4 (DPO), MLB batter-strikeouts 0.68 / −15.1 (DPO), **NFL passing-tds 0.77 /
−4.16 (NegBin, shipped devel, g4 0.0447 — one retrain from failing)**, **NFL rushing-tds 0.78 /
−7.16 (ZINB, shipped devel)**, WNBA BLST 0.79 (DPO), WNBA TOV 0.79 (DPO), **NFL receiving-tds 0.80
/ −10.6 (ZINB hurdle, shipped devel, and CITL 1.64)**. The pattern worth naming: **the NFL
touchdown markets are systematically sub-Poisson and are all on families that cannot represent
it.** Separately, NHL sogBS predicts `P(Y=0) = 0.0224` against an empirical zero rate of **exactly
0.0000** — a hard positive floor no dispersion knob can fix; that one is a support problem
(shifted/truncated), not a dispersion problem.

**Cohort-level correlates of `g4_pit_ks`** (34 cells with a ledger gate row, Spearman):
`|zero-mass gap|` **+0.607**, `|zero z|` +0.484, `|φ̂ᵢₛₒ − 1|` +0.448, `|CITL − 1|` **+0.150**.
So the zero-mass gap is the proximate cause of Gate-4 failure cohort-wide, and it has *at least
three* upstream causes — mean bias, conditional dispersion, family support. The local diagnostic's
"the KS supremum sits at the bottom of the lattice" is confirmed and generalises; its
interpretation as zero-inflation does not.

### 4. Vestigial zero-inflation is a well-documented pathology, and the Vuong test cannot detect it

Warton (2005, doi:10.1002/env.702) compared marginal count models on 20 multivariate datasets
(1672 variables) and found the plain negative binomial best-fitting *without* zero inflation — the
canonical "many zeros ≠ zero inflation" result. Blasco-Moreno et al. (2019,
doi:10.1111/2041-210X.13185) separate false / random / structural zeros and show that only
structural zeros justify a ZI component. Perumean-Chaney et al. (2013,
doi:10.1080/00949655.2012.668550) show by simulation that zero-inflation and over-dispersion are
mutually confounded, so a mis-specified dispersion is routinely absorbed by π and vice versa.

**Identifiability at π → 0.** The ZI weight sits on the boundary of its parameter space, so
(a) the Wald standard error is not usable, (b) the likelihood-ratio null is a 50:50 mixture
`½χ²₀ + ½χ²₁`, not `χ²₁` (Self & Liang 1987, doi:10.1080/01621459.1987.10478472; Chernoff 1954),
and (c) the information matrix approaches singularity along the π direction — the same *shape* of
pathology as the α = 0 SkewNormal Fisher singularity this repo already fought (Hallin & Ley 2014),
so the intuition transfers.

**The Vuong test is invalid for this.** Wilson (2015, doi:10.1016/j.econlet.2014.12.029) shows a
non-zero-inflated model is neither strictly nor partially non-nested in its ZI counterpart (ZINB
reduces to NB at π = 0), so Vuong's criteria fail; he further shows the test cannot identify
zero *deflation*, producing inconsistent conclusions in exactly the direction this cohort sits.
**Concrete implication:** the Schwarz-corrected Vuong emitted by `zinb-routing-diagnostics` into
`data/zinb_routing/{LEAGUE}_diagnostics.parquet` must stay descriptive-only and must never route a
cell. (This matches the standing plan note and the 2026-05 rescope verdict.)

**The correct in-sample family** is the score-test line — van den Broek (1995, doi:10.2307/2532959)
for ZIP; Deng & Paul (2000, doi:10.2307/3315965) generalising to discrete GLMs (reducing to van den
Broek in the Poisson case); Jansakul & Hinde (2002, doi:10.1016/S0167-9473(01)00104-9) for ZIP with
covariates; Ridout, Hinde & Demétrio (2001, doi:10.1111/j.0006-341X.2001.00219.x) for ZIP-vs-ZINB.
All of them test a *fitted GLM* null and assume the mean model is correct — precisely the
assumption that fails on this cohort — so none is the right primary tool here.

**Recommended per-cell rule (computable today, no refit, no boundary problem):** the
frozen-parameter form of Wilson & Einbeck's zero-modification test (2019,
doi:10.1177/1471082X18762277). Score the cell's **ZI-free counterpart** (gate forced to 0) on the
held-out split with parameters frozen, and compare observed to expected zeros. Under H₀ the zero
count is Poisson-binomial with exact variance, so

```
z₀ = ( O₀ − Σᵢ p̂ᵢ(0) ) / sqrt( Σᵢ p̂ᵢ(0)·(1 − p̂ᵢ(0)) )
```

is two-sided, needs no asymptotics in the parameters (they are frozen out-of-sample), detects
deflation as well as inflation, and has none of the nesting or boundary problems. Route:
`|z₀| < 2` → no zero modification, drop the gate; `z₀ > +3` → genuine excess zeros, keep a gate and
use **hurdle** not joint; `z₀ < −3` → zero **deflation**, a ZI model is structurally wrong and DPO
(or CMP) is mandatory.

**Failure modes, stated plainly.**
1. *It is a residual-pattern test, not a family test.* A mean bias or an unmodelled dispersion
   defect fires it. DHARMa's own documentation makes this warning explicitly for its simulation
   analogue `testZeroInflation`. **It must be run after the mean is calibrated**, or it will
   mis-route exactly the NFL tds / interceptions class.
2. *Rows are clustered by player,* so the Poisson-binomial variance is under-stated and `|z₀|` is
   optimistic. Use the player-clustered bootstrap the repo already applies in Gates 1 and 6.
3. *At small π the test is under-powered by construction* — absence of evidence, not evidence of
   absence.
4. *Test-then-choose biases downstream inference* (Campbell 2021, doi:10.1111/2041-210X.13559,
   arXiv:1911.00115). In this repo the downstream object is a ship gate on a disjoint split, so the
   exposure is selection overfitting rather than invalid p-values. Mitigate by making the screen an
   **entry criterion for a pre-registered confirm**, never a selector on the scored split.

### 5. The mean corrector exists, is already swept, and is mis-contracted on gated families

`_POSTHOC = ("none", "prob_recal_isotonic", "prob_recal_platt", "roe_mean", "isotonic_mean")` is a
search axis on the ZINB, NegBin and DPO base specs
(`training/model_strategy/specs.py`). The mean-stage corrector is applied to `decoded["ev"]`
**before** fusion, so — unlike the docstring's "it deliberately does not touch dispersion (Gate 4)"
might suggest — it *does* flow into `fused["weighted_mean"]` and therefore into `NB_P` / `DP_MU`
and the Gate-4 CDF. Good: it is exactly the right stage.

The defect is the **target**. `pipeline` calls

```python
mean_posthoc_blob = posthoc.fit_posthoc(posthoc_slug, decoded["ev_validation"], val_result)
```

and for ZINB `decoded.ev = r·p/(1−p)` — the **base** NegBin mean with the gate excluded
(`helpers.distributions.decode_predictive_mean`) — while `val_result` is the zero-**inclusive**
outcome. The fitted map therefore absorbs the `(1−π)` factor, and `(1−π)` is then applied a second
time by `_pred_cdf_pmf` / `get_odds` / `_zero_inflated_mean`.

I simulated that exact contract on the real dumps (`rb/roe_double_gate.py`: fit the affine on the
earliest 60 % by date, apply, re-gate, score the latest 40 %):

| cell | π̄ | served CITL now | CITL under the **as-coded** corrector | CITL under a **gate-aware** corrector |
|---|---|---|---|---|
| NHL goals | 0.790 | 0.997 | **0.260** | 1.058 |
| NBA BLK | 0.527 | 0.972 | **0.441** | 0.895 |
| NFL tds | 0.414 | 0.762 | **0.629** | 1.056 |
| NFL interceptions | 0.135 | 0.721 | 0.819 | 0.950 |
| NBA TOV | 0.077 | 0.910 | 0.911 | 0.958 |
| NBA PF | 0.017 | 0.932 | 0.947 | 0.962 |

The damage scales with π̄, exactly as the mechanism predicts. Exposure is 17 ZINB + 15 gated
SkewNormal cells (SkewNormal cells above `GATE_PUBLISH_THRESHOLD` carry the same `Gate` column and
the same `_zero_inflated_mean` treatment). **Currently exactly one of those 32 carries a
mean-stage posthoc — NFL interceptions, `shipped: withheld` — so nothing live is harmed.** The
board did its job: it tried the corrector, the corrector wrecked the cell, the corner lost. The
cost is a *suppressed lever*, and it is suppressed hardest precisely on the cells that need it
most.

Weak corroborating signal (adoption rates, not proof): mean-stage posthoc is selected on 6/25 DPO
cells (24 %, ungated) and 1/17 ZINB cells (5.9 %, gated).

### 6. The correction shape differs by cell — affine is right for PF, not obviously for NFL tds

CITL by quintile of the served mean:

| quintile | NFL tds | NFL interceptions | NBA PF | NBA TOV | NBA OREB (control) |
|---|---|---|---|---|---|
| Q1 (low) | 1.038 | 0.662 | 0.899 | 0.783 | 0.837 |
| Q2 | 0.939 | 0.617 | 0.975 | 0.963 | 1.026 |
| Q3 | 0.831 | 0.581 | 0.944 | 0.992 | 1.079 |
| Q4 | 0.604 | 0.899 | 0.941 | 0.950 | 1.109 |
| Q5 (high) | 0.727 | 1.077 | 0.985 | 0.926 | 1.116 |

NBA PF is a nearly flat ~5 % deficit — an affine or scale corrector is well shaped. NFL tds is
range-dependent and non-monotone (worst at Q4), so an affine recovers only part of it and an
isotonic map is the better-shaped tool — but §6.1 Rung A's house rule, grounded in ref [48]
(Roelofs et al. 2022, low-base-rate calibration-error bias), says **affine ROE only at NFL count
means**. Sweep both; pre-register affine as primary and treat `isotonic_mean` as the second corner.

NFL interceptions is the warning case: its calibration slope on the dump is **−0.166** and its
quintile CITL is non-monotone. A mean corrector fit there largely flattens the prediction toward
the marginal mean, which would erase whatever ranking the mean head has. Any interceptions arm must
be read against g1 and `roc_auc`, not just g4.

NBA OREB, the control, has a genuine *slope* defect in the **opposite** direction (over-predicts
the top). That is why a global scale harms it, and it is why nothing here can be a default.

### 7. Transfer is governed by stationarity of the bias — and that is a testable entry gate

Honest temporal probe (`rb/temporal_k.py`): fit the mean scale on the **earliest 60 %** of each
dump by date, score the **latest 40 %**. This is strictly harsher than random half-splits and is
the closest read-only analogue of the production validation → test discount.

| cell | CITL (fit era) | CITL (score era) | fitted k | KS base | **KS after** | thr | verdict |
|---|---|---|---|---|---|---|---|
| NFL tds | 0.721 | 0.762 | 1.468 | 0.0549 | **0.0265** | 0.050 | pass, 2× margin |
| NBA PF | 0.967 | 0.932 | 1.069 | 0.0694 | **0.0393** | 0.050 | pass |
| NFL interceptions | 0.751 | 0.721 | 1.369 | 0.1342 | **0.0620** | 0.110 | pass |
| NBA TOV | 0.950 | 0.910 | 1.074 | 0.0680 | **0.0394** | 0.050 | pass (g1-walled) |
| NHL goals | 0.937 | 0.997 | 1.332 | 0.0340 | 0.0259 | 0.050 | improves, already passing |
| NBA BLK | 1.084 | 0.972 | 1.073 | 0.0689 | 0.0577 | 0.050 | improves, still fails split |
| **NBA OREB** | 1.119 | 0.984 | 0.941 | 0.0476 | **0.0663** | 0.050 | **HARMED — loses a pass** |
| **WNBA FTM** | 1.183 | 1.118 | 0.911 | 0.0419 | **0.0568** | 0.050 | **HARMED** |
| **MLB runs-allowed** | 0.912 | 1.019 | 1.094 | 0.0303 | **0.0659** | 0.065 | **HARMED** |

The rule falls straight out: **the corrector transfers exactly where the bias is large and
stationary across the fit/score boundary, and harms where it is not.** OREB (1.119 → 0.984) and
MLB runs-allowed (0.912 → 1.019) flip sign between eras; WNBA FTM starts at KS 0.0419 with nothing
to fix. All three are correctly excluded by a stationarity screen. This measured harm set is the
strongest argument in the brief that the fix must be a per-cell option, never a default.

### 8. Ledger reality: what any Gate-4 fix can actually convert

Newest rows per cell in `research/confirm_nominee_gates.csv`:

| cell | rows | g1 pass | g4 pass | g6 pass | best g4 | pass-all-but-g4 rows | g1 `ci_hi` (median) | verdict |
|---|---|---|---|---|---|---|---|---|
| **NFL tds** | 7 | **7/7** | 1/7 | 6/7 | 0.0203 (stale matrix) | **5** | −0.0208 | **convertible; g4 gap 0.0044** |
| **NBA PF** | 3 | **3/3** | 0/3 | 3/3 | 0.0766 | **3** | −0.1159 | **convertible; g4 gap 0.027** |
| NFL interceptions | 10 | 9/10 | 0/10 | **0/10** | 0.1061 | 0 | −0.0775 | needs g4 **and** g6 |
| NBA TOV | 7 | **2/7** | 4/7 | 3/7 | 0.0204 | 0 | +0.0138 | **Gate-1 walled** |
| NHL goalsAgainst | 10 | **1/10** | 2/10 | 10/10 | 0.0349 | 1 | +0.0128 | **Gate-1 walled** |
| MLB runs-allowed | 0 | — | — | — | — | — | — | no ledger rows; shipped, g4 0.0465 |

Three corrections to the dispatch's framing, all load-bearing:

1. **NFL interceptions does not fail Gate 1** on any recent row — it passes 9/10 with `ci_hi`
   ≈ −0.077 and BSS +0.24…+0.30. It fails **Gate 6** on 10/10 (`g6_citl_ci_hi` 0.932 / 0.630 —
   under-prediction, i.e. the same mean deficit). Verify with one `ship scorecard` read before
   enrolling or excluding it.
2. **NBA TOV's Gate 4 is already solved** by the family axis (DPO 0.0204, NegBin 0.0239) and it
   still does not ship, because g1 fails. No amount of Gate-4 work converts TOV.
3. **NBA PF's §6.6 "g5 co-failing" note is stale** — its three most recent rows pass g5. PF is
   currently a pure g4 blocker with the largest Gate-1 cushion in the cohort.

Also worth carrying: the ledger's `g4_pit_ks` and the dump-scored `g4_pit_ks` disagree on PF
(0.0775 vs 0.0680) and interceptions (0.1849 vs 0.1199) — `count_dispersion_flip.md` already flags
that these dumps and gate rows come from different runs. **Use the dumps for mechanism, the ledger
for pass/fail.**

---

## Recommendation / routing protocol

### A. Per-cell screen (offline, free, run before any confirm)

Compute on the cell's validation split (or, offline, on a temporal split of its dump). All four
legs are one-liners over `(y, m)` plus the served CDF.

| leg | statistic | threshold | routes to |
|---|---|---|---|
| **S1 mean** | `CITL = Σm/Σy`, player-clustered bootstrap CI | \|CITL−1\| ≥ 0.04 **and** CI excludes 1 | mean-stage posthoc |
| **S2 stationarity** | split the fit window in half by date; two half-window CITLs | same side of 1 **and** \|ΔCITL\| < 0.06 | **veto** if it fails — this is what excludes OREB / runs-allowed |
| **S3 dispersion** | `φ̂ = (Var(y)−Var(m))/mean(y)` + Cameron–Trivedi t | φ̂ < 0.90 and t < −3 | DPO (sub-Poisson); φ̂ > 1.20 and t > +3 → NegBin/hurdle |
| **S4 zero modification** | frozen-parameter `z₀` on the **ZI-free** counterpart | \|z₀\| < 2 → drop gate; > +3 → hurdle; < −3 → DPO mandatory | family |

Order matters: **run S1/S2 before S4**, because a mean bias fires the zero test (finding 4,
failure mode 1). Never let S4 fire on an uncalibrated mean.

Free bonus detector already in the pipeline: **a count cell whose `dispersion_cal` fit pins at its
`c` bound is mis-familied.** `c → 10` on a NegBin/ZINB means the fit wants sub-Poisson variance the
family forbids; `c → 0.1` means it wants more spread than the shape head can give.

### B. Family routing

| condition | family | machinery |
|---|---|---|
| φ̂ < 0.90 (sub-Poisson) | **DPO** | **exists** — swept family, `_DP_PHI_CEILING = 25`, exact-series normalizer, 14 ships |
| φ̂ ∈ [0.90, 1.20], z₀ ∈ [−2, +2] | **NegBin** | **exists** — swept family |
| z₀ > +3 (genuine excess zeros) | **ZINB `zinb_mode: hurdle`** | **exists** — swept axis |
| any | **joint ZINB** | **retire as a default.** 0/132 board corners, best slack −0.166; gate head unidentified under NLL (repo's own finding); a ZI weight can only *add* zeros while 6/9 probed cells need the other direction |
| — | **CMP** | **do not build.** See reality checks |
| — | **ZTNB-hurdle** | **do not re-propose.** Analytically killed in B1.1 (`q < NB(0)` on ~65 % of FG3M rows) — and that failure is itself the zero-deflation this brief measures |

### C. The mean lever

Route S1∧S2 passers to the **existing** `posthoc` axis: `roe_mean` primary at NFL count means
(house rule, ref [48]), `isotonic_mean` as the second corner elsewhere.

Two ways to make it work on a gated cell, in cost order:

- **Zero-code path (recommended for the first experiment): route the cell to NegBin or DPO first.**
  On a gate-free family `decoded.ev` *is* the predictive mean, so the corrector is correctly
  targeted today. Both `dist` and `posthoc` are already swept axes; this is a two-axis corner, not
  a code change.
- **Six-line path (only if a genuinely gated cell needs it): fix the MEAN_STAGE contract.** Fit and
  apply the corrector on the zero-inclusive mean `(1−π)·ev`, then divide `(1−π)` back out before it
  reaches `_stage_family_shape_columns`. Localised to `pipeline._train_market_core` (~lines
  4276–4283) plus a golden. This changes a serving distribution, so it is research-gated — **this
  brief discharges that gate**, but it is not required by the experiment below and should not be
  built ahead of E1's verdict.

### D. High-π cells — KILL the mixture-weight recalibration build

There is no high-π Gate-4 problem to solve. All three high-π cells pass Gate 4 today (NHL goals
0.0366, NBA BLK 0.0463, WNBA FTM 0.0465, dump-scored, against 0.05). The only high-π failing cell,
NFL tds (π̄ 0.41), is a mean-deficit cell whose gate contributes ≈ 0.000 once the mean is free.
**Do not build a post-hoc recalibrator for a mixture weight.**

If a high-π cell ever does fail g4, the defensible procedure is the one this repo already has:
**derived-π hurdle** — a calibrated binary zero classifier `q̂(x)` on validation, then
`π = clip((q̂ − NB(0))/(1 − NB(0)), 0, 1)`. That is post-hoc recalibration of a **probability** with
a proper calibrator (Platt/isotonic on a binary event), which is standard and well-founded. A free
logit shift on a mixture weight is a shape parameter with no calibration target of its own — the
canonical unidentified knob that Goodharts under search pressure, which this repo has already been
bitten by (`[[proxy_goodhart_under_search]]`: a static-fidelity ≤ 0.004 proxy diverged 0.02 → 0.15
as a live constraint). Two standing repo facts to carry: the derived-π gate is **π_zi, not q**
(`[[p2_hurdle_zinb_verdict]]`), and hurdle cells **bypass Optuna entirely**, so a `calibrated`
HP-selection pin is inert there (§8.2 open-Q #9(e)).

Also: **do not re-propose Rung C (isotonic-PIT / IDR) for count cells.** It is built, and it is a
confirmed dead end on the low-mean lattice (`[[rung_c_whole_cdf_recal]]`); the monotone map
degrades the lattice, and its g4↔g5 tension is real (WNBA DREB g4 0.0306 ✓ / g5 0.0961 ✗).

---

## Experiment design (pre-registered)

### E1 — NFL tds. Run this first. One cell, one arm, ~35 min.

```
--dist NegBin --posthoc roe_mean --target-normalization none \
--dist-training-loss nll --blending-loss-fn nll --count-dispersion-objective crps
```
on the pinned matrix `7918c1b8…`, full HPO. **Baseline is the recorded ZINB incumbent row**
(2026-08-23, g4 0.0544, g1 `ci_hi` −0.0208, BSS 0.159) — no second run needed.

*Why first:* it is the only cell that is pass-all-but-g4 on the **current** matrix hash with a real
Gate-1 win; the gap is the smallest in the cohort (0.0044); the offline temporal-transfer estimate
is 0.0265 (full-frame 0.0209), a 5× margin over the gap; it needs **no new code**; and NegBin is
adequate because the cell measures equidispersed (φ̂ 1.03, CT t −0.32).

*Record:* all six gates, `g4_pit_ks`, `g4_tail_pit_ks`, `g6_citl_ci_hi`, `g1_brier_diff_ci_hi`,
`roc_auc`, **`model_weight`**, and CITL computed **both** ways (`Σ EV/ΣResult` model-only and
`Σ Blended_EV·(1−π)/ΣResult` served). The two CITLs are what make a failure attributable.

| outcome | reading | next |
|---|---|---|
| all six pass | **SHIP.** Flip `dist: NegBin`, `posthoc: roe_mean`, `shipped: devel` | run E2 |
| g4 < 0.05 and served CITL ∈ [0.97, 1.03] but another gate fails | mechanism confirmed, cell not convertible | run E2 |
| **served CITL ∈ [0.97, 1.03] and g4 ≥ 0.05** | **KILL the whole direction.** The zero-mass gap is not mean-driven and the offline probe was an artifact of fitting on the scored split | stop; do not run E2/E3 |
| served CITL still < 0.95 while model-only CITL ≈ 1.0 | **dilution**, not mechanism failure — the blend re-shrank the correction | the lever is §6.5 blend structure (closed, research-gated), not post-hoc; do not spend E2/E3 |
| g1 `ci_hi` ≥ 0.005 or g6 fires | the mean lift is buying calibration with edge | abandon the mean lever on this cohort; route to the family axis |

### E2 — NBA PF. Only if E1's mechanism confirms. One cell, two arms, ~70 min.

- **Arm A:** `--dist DPO --posthoc roe_mean`
- **Arm B:** `--dist DPO --posthoc none`

*Why PF:* largest Gate-1 cushion in the cohort (`ci_hi` −0.108…−0.188, BSS 0.29–0.63), 3/3
pass-all-but-g4, largest g4 gap (0.027), flat correction shape (quintile CITL 0.90–0.99 → affine is
well specified), and it is **the only cell with a measured structural family mismatch**
(φ̂ 0.78 sub-Poisson on a NegBin body).

*Reads:* A ships and B doesn't ⇒ mean-driven. B ships ⇒ family-driven, and the earlier
16-corner DPO kill on PF was matrix/HP-bound rather than structural. Neither ⇒ PF's residual is
neither mean nor family; stop escalating PF under §8.1 matrix exhaustion.

### E3 — NFL interceptions. Only if E1 ships. One cell, one arm, ~35 min.

`--dist NegBin --posthoc roe_mean`. Must clear **g4 and g6 together**. Two preconditions:
(i) settle the Gate-1 premise with one `ship scorecard` read — the dispatch and the ledger
disagree; (ii) accept that the cell's mean has a **negative** calibration slope (−0.166), so a mean
corrector partly flattens it toward the marginal — read `roc_auc` and `g1_brier_diff_ci_hi`, not
just g4. n = 378, so treat any pass as provisional pending a second matrix.

### Do not run

**NBA TOV** (2/7 g1 pass; g4 already solved at 0.0204 by DPO — a Gate-4 fix ships nothing),
**NHL goalsAgainst** (1/10 g1 pass), **MLB runs-allowed** (fails the S2 stationarity leg;
CITL flips 0.912 → 1.019 and the temporal probe *harms* it, 0.0303 → 0.0659).

### Controls

**NBA OREB** and **NBA BLK** must retain `ship`. Neither passes the S1∧S2 screen, so neither should
ever be enrolled; the S2 stationarity leg is the veto that keeps this from becoming a default.

---

## Reality checks

- **Effect size and regime.** The mean-only gains (0.03–0.06 KS) hold *only* where the CITL bias is
  ≥ 4 % and stationary across a temporal split. Outside that regime the identical intervention
  **harms**, and I measured both sides: NBA OREB 0.0476 → 0.0663, WNBA FTM 0.0419 → 0.0568, MLB
  runs-allowed 0.0303 → 0.0659, all on the honest temporal split, all losing a Gate-4 pass. This is
  a per-cell option or it is nothing.
- **The probe is not the production object.** My `k` is a single scalar fit on the scored dump's own
  halves (and, in the temporal arm, on its earlier era). Production fits an affine or isotonic map
  on the *validation* split and serves a different era. The observed val → test discount on the
  SkewNormal `(c, s)` analogue is **+0.008–0.010 KS**. NFL tds' temporal margin (0.0265 vs 0.050)
  survives that comfortably; **NBA PF's mean-only margin (0.0393 vs 0.050) does not clearly
  survive** — which is precisely why E2 carries the family arm.
- **The single most likely reason E1 under-performs is blend dilution.** `roe_mean` corrects the
  model mean *before* fusion, so the realised move in the served mean is scaled by `model_weight`.
  NFL tds' model-only gated mean is 0.903 of the outcome while its fused mean is 0.737 — the book
  is pulling it down by 0.17. If `model_weight` is low, a model-side corrector recovers only part
  of what the probe's post-fusion `k` moved. E1's dual-CITL recording is designed to detect this,
  and the answer if it happens is **§6.5 blend structure** (closed, research-gated), not more
  post-hoc.
- **This is the lever the plan is explicitly skeptical of.** §6.1 Rung A's operator note says
  post-hoc mean correction edits the central tendency the model should learn, and should be a last
  resort held to the g1 BSS guardrail. That skepticism is right and the kill triggers encode it.
  The counter-argument specific to this cohort: the deficit is **stationary** (NFL interceptions
  0.742 / 0.714 / 0.760 across three time-ordered thirds of its dump; NFL tds 0.690 / 0.751 /
  0.775), so it is a systematic bias rather than drift, which is the one regime in which a
  post-hoc mean map is a legitimate estimator rather than a patch.
- **Build cost is near zero, deliberately.** E1/E2/E3 use only existing swept axes (`dist`,
  `posthoc`). The one code item — the MEAN_STAGE gate-aware contract — is ~6 lines plus a golden
  and is **not on the critical path**. Nothing here proposes a new distribution class, a new
  post-hoc stage, or a new gate.
- **What would make the whole direction wrong.** (i) The CITL deficit is a selection artifact of
  which rows carry offers — the test rows *are* the served population, so I doubt it, but it is
  untested. (ii) The gain is an artifact of fitting on the scored split; the temporal arm is my
  best defence and it is still within-dump. (iii) Blend dilution (above). (iv) A mean lift that
  buys g4 at the cost of g1 — the E1 secondary kill trigger.
- **Do not read magnitudes from any small-HP A/B** (`[[deterministic_ab_g4_oversell]]`). E1–E3 are
  full-HPO confirms, so this applies to my offline probes, not to the experiments.
- **CMP is genuinely the better under-dispersed family and still should not be built.** The
  mean-parametrized CMP (Huang 2017, doi:10.1177/1471082X17697749, arXiv:1606.03214) attains the
  minimum-variance two-point limit for any mean (arXiv:2011.07503), which the double Poisson does
  not. But the *measured* need here is φ̂ ≈ 0.78, i.e. a DP φ ≈ 1.3 against an implementation
  ceiling of 25 — nowhere near where the approximation matters. CMP would cost a new torch class, a
  normalizing-series lookup or asymptotic expansion (Gaunt et al. 2019,
  doi:10.1007/s10463-017-0629-6), and the full §7.3 nine-site serve wiring. **Research project;
  DPO is the engineering project and it is already done.**
- **The dispersion floor bounds what any family swap can buy at low mean.** The most
  under-dispersed law on ℤ≥0 with mean μ is the two-point law on ⌊μ⌋/⌈μ⌉ (arXiv:2011.07503), giving
  a dispersion index of `1 − μ` for μ < 1. At NFL tds' μ ≈ 0.24 the floor is 0.73, so a family swap
  has at most ~27 % of the variance to play with there. On NBA PF (μ ≈ 2.1, floor 0.083) the family
  axis has real room; on the NFL TD markets it does not, which is another reason those cells are
  mean stories.

---

## Open questions / caveats

1. **`model_weight` on NFL tds is unknown to me** and it determines how much of a pre-fusion mean
   correction survives. Read it from the confirm log before interpreting E1.
2. **Which side of the blend owns the deficit varies by cell.** Model-only vs fused gated CITL:
   NFL tds 0.903 → 0.737 (blend-driven), MLB runs-allowed 1.024 → 0.953 (blend-driven), NFL
   interceptions 0.753 → 0.738 (model-driven), NBA PF 0.943 → 0.953 and NBA TOV 0.906 → 0.933
   (blend slightly *helps*). A blend-driven deficit is a §6.5 question, not a §6.1 one.
3. **NFL receiving-tds is a shipped `devel` cell with served CITL 1.64** — the served mean is 64 %
   above the outcome mean. Gates 2/3 miss it because σ is large at μ = 0.16; Gate 6's CITL leg is
   one-sided (under-prediction only) and its over leg is guarded by `mean(Result) ≥ 1`. **A
   low-mean count cell that over-predicts currently has no gate.** Worth an owner look; it is not
   part of this lane's ask.
4. **NHL sogBS predicts P(Y=0) = 0.0224 against an empirical zero rate of exactly 0.** That is a
   hard positive floor, i.e. a support problem (shifted / zero-truncated), not a dispersion one. No
   φ knob fixes it. Separate item.
5. **NFL passing-tds (NegBin, devel, g4 0.0447) and NFL rushing-tds (ZINB, devel) both read
   sub-Poisson** (φ̂ 0.77 / 0.78, CT t −4.2 / −7.2) on families that cannot represent it. They pass
   Gate 4 today by a thin margin. Candidates for a DPO supersession lane, one cell per confirm.
6. **My zero z-test numbers are computed on the served predictive (gate on)**, which is the
   calibration read. The routing form of the test (finding 4) evaluates the **ZI-free** counterpart
   and I computed that only via the existing nine-cell `gate_off_probe`, as KS rather than as z₀.
   Someone should compute the gate-off z₀ across the cohort before using it to route.
7. **The 50:50 boundary mixture and score-test literature is cited but not exercised here** — I did
   not run a van den Broek / Deng–Paul score test on any cell, because the mean-misspecification
   finding makes an in-sample GLM-null test the wrong first tool. If the mean is fixed and a cell
   still shows excess zeros, that is when the score test earns its place.
8. **I could not verify a journal version of Huang's arbitrarily-under-dispersed CMP paper** — it is
   cited by arXiv ID only.
9. **The dump-vs-ledger `g4_pit_ks` disagreement on NBA PF and NBA TOV persists** (PF 0.0680 vs
   0.0775). `count_dispersion_flip.md` already records that these dumps and gate rows come from
   different runs. Every number in this brief that comes from a dump is labelled as such.

---

## Bibliography

| # | Reference | Identifier |
|---|---|---|
| 1 | Wilson, P. (2015). The misuse of the Vuong test for non-nested models to test for zero-inflation. *Economics Letters* 127, 51–53. | doi:10.1016/j.econlet.2014.12.029 |
| 2 | Wilson, P. & Einbeck, J. (2019). A new and intuitive test for zero modification. *Statistical Modelling* 19(4). | doi:10.1177/1471082X18762277 |
| 3 | Warton, D. I. (2005). Many zeros does not mean zero inflation: comparing the goodness-of-fit of parametric models to multivariate abundance data. *Environmetrics* 16(3), 275–289. | doi:10.1002/env.702 |
| 4 | Blasco-Moreno, A., Pérez-Casany, M., Puig, P., Morante, M. & Castells, E. (2019). What does a zero mean? Understanding false, random and structural zeros in ecology. *Methods in Ecology and Evolution* 10(7), 949–959. | doi:10.1111/2041-210X.13185 |
| 5 | Perumean-Chaney, S. E., Morgan, C., McDowall, D. & Aban, I. (2013). Zero-inflated and overdispersed: what's one to do? *J. Statistical Computation and Simulation* 83(9), 1671–1683. | doi:10.1080/00949655.2012.668550 |
| 6 | Campbell, H. (2021). The consequences of checking for zero-inflation and overdispersion in the analysis of count data. *Methods in Ecology and Evolution* 12(4), 665–680. | doi:10.1111/2041-210X.13559; arXiv:1911.00115 |
| 7 | van den Broek, J. (1995). A score test for zero inflation in a Poisson distribution. *Biometrics* 51(2), 738–743. | doi:10.2307/2532959 |
| 8 | Deng, D. & Paul, S. R. (2000). Score tests for zero inflation in generalized linear models. *Canadian J. Statistics* 28(3), 563–570. | doi:10.2307/3315965 |
| 9 | Jansakul, N. & Hinde, J. P. (2002). Score tests for zero-inflated Poisson models. *Computational Statistics & Data Analysis* 40(1), 75–96. | doi:10.1016/S0167-9473(01)00104-9 |
| 10 | Ridout, M., Hinde, J. & Demétrio, C. G. B. (2001). A score test for testing a zero-inflated Poisson regression model against zero-inflated negative binomial alternatives. *Biometrics* 57(1), 219–223. | doi:10.1111/j.0006-341X.2001.00219.x |
| 11 | Self, S. G. & Liang, K.-Y. (1987). Asymptotic properties of MLE and LR tests under nonstandard conditions. *JASA* 82(398), 605–610. | doi:10.1080/01621459.1987.10478472 |
| 12 | Cameron, A. C. & Trivedi, P. K. (1990). Regression-based tests for overdispersion in the Poisson model. *J. Econometrics* 46(3), 347–364. | doi:10.1016/0304-4076(90)90014-K |
| 13 | Dean, C. & Lawless, J. F. (1989). Tests for detecting overdispersion in Poisson regression models. *JASA* 84(406), 467–472. | doi:10.1080/01621459.1989.10478792 |
| 14 | Dunn, P. K. & Smyth, G. K. (1996). Randomized quantile residuals. *J. Computational and Graphical Statistics* 5(3), 236–244. | doi:10.1080/10618600.1996.10474708 |
| 15 | Czado, C., Gneiting, T. & Held, L. (2009). Predictive model assessment for count data. *Biometrics* 65(4), 1254–1261. | doi:10.1111/j.1541-0420.2009.01191.x |
| 16 | Gneiting, T., Balabdaoui, F. & Raftery, A. E. (2007). Probabilistic forecasts, calibration and sharpness. *JRSS-B* 69(2), 243–268. | doi:10.1111/j.1467-9868.2007.00587.x |
| 17 | Gneiting, T. & Ranjan, R. (2013). Combining predictive distributions. *Electronic J. Statistics* 7, 1747–1782. | doi:10.1214/13-EJS823 |
| 18 | Efron, B. (1986). Double exponential families and their use in generalized linear regression. *JASA* 81(395), 709–721. | doi:10.1080/01621459.1986.10478327 |
| 19 | Zou, Y., Geedipally, S. R. & Lord, D. (2013). Evaluating the double Poisson generalized linear model. *Accident Analysis & Prevention* 59, 497–505. | doi:10.1016/j.aap.2013.07.017 |
| 20 | Huang, A. (2017). Mean-parametrized Conway–Maxwell–Poisson regression models for dispersed counts. *Statistical Modelling* 17(6), 359–380. | doi:10.1177/1471082X17697749; arXiv:1606.03214 |
| 21 | Huang, A. (2020). On arbitrarily underdispersed Conway–Maxwell–Poisson distributions. *(journal version unverified)* | arXiv:2011.07503 |
| 22 | Shmueli, G., Minka, T. P., Kadane, J. B., Borle, S. & Boatwright, P. (2005). A useful distribution for fitting discrete data: revival of the Conway–Maxwell–Poisson distribution. *JRSS-C* 54(1), 127–142. | doi:10.1111/j.1467-9876.2005.00474.x |
| 23 | Sellers, K. F. & Shmueli, G. (2010). A flexible regression model for count data. *Annals of Applied Statistics* 4(2), 943–961. | doi:10.1214/09-AOAS306 |
| 24 | Sellers, K. F. & Morris, D. S. (2017). Underdispersion models: models that are "under the radar". *Communications in Statistics — Theory and Methods* 46(24), 12075–12086. | doi:10.1080/03610926.2017.1291976 |
| 25 | Gaunt, R. E., Iyengar, S., Olde Daalhuis, A. B. & Simsek, B. (2019). An asymptotic expansion for the normalising constant of the Conway–Maxwell–Poisson distribution. *Ann. Inst. Statistical Mathematics* 71, 163–180. | doi:10.1007/s10463-017-0629-6 |
| 26 | Mullahy, J. (1986). Specification and testing of some modified count data models. *J. Econometrics* 33(3), 341–365. | doi:10.1016/0304-4076(86)90002-3 |
| 27 | Feng, C. X. (2021). A comparison of zero-inflated and hurdle models for modeling zero-inflated count data. *J. Statistical Distributions and Applications* 8:8. | doi:10.1186/s40488-021-00121-4 |
| 28 | Winkelmann, R. (1995). Duration dependence and dispersion in count-data models. *J. Business & Economic Statistics* 13(4), 467–474. | doi:10.1080/07350015.1995.10524620 |
| 29 | Bourguignon, M. & Weiß, C. H. et al. (2021). A simple and useful regression model for underdispersed count data based on Bernoulli–Poisson convolution. *Statistical Papers*. | doi:10.1007/s00362-021-01253-0 |
| 30 | Kuleshov, V., Fenner, N. & Ermon, S. (2018). Accurate uncertainties for deep learning using calibrated regression. *ICML*. | arXiv:1807.00263 |
| 31 | Henzi, A., Ziegel, J. F. & Gneiting, T. (2021). Isotonic distributional regression. *JRSS-B* 83(5), 963–993. | doi:10.1111/rssb.12450; arXiv:1909.03725 |
| 32 | Dheur, V. & Ben Taieb, S. (2023). A large-scale study of probabilistic calibration in neural network regression. | arXiv:2306.02738 |
| 33 | Hartig, F. DHARMa: residual diagnostics for hierarchical regression models (`testZeroInflation`). | CRAN package `DHARMa` |
| 34 | Belitz, K. & Stackelberg, P. E. (2021). Evaluation of six methods for correcting bias in estimates from ensemble tree machine-learning models. *Environmental Modelling & Software* 139, 105006. | doi:10.1016/j.envsoft.2021.105006 |
| 35 | Roelofs, R. et al. (2022). Mitigating bias in calibration error estimation. *(repo ref [48])* | — |
| 36 | März, A. (2019/2022). LightGBMLSS — an extension of LightGBM to probabilistic modelling. | arXiv:1912.03384 |
| 37 | Hallin, M. & Ley, C. (2014). Skew-symmetric distributions and Fisher information: the double sin of the skew-normal. *Bernoulli* 20(3), 1432–1453. | doi:10.3150/13-BEJ528 |

*Repo-internal evidence referenced above (not literature):* `docs/ship_gate.md`;
`docs/handoffs/model_improvement_track.md` §6.1, §6.6, §7.3, §8.1, §8.2;
`docs/handoffs/count_dispersion_flip.md`; memories `[[p2_hurdle_zinb_verdict]]`,
`[[project_b1_ztnb_rescope]]`, `[[rung_c_whole_cdf_recal]]`, `[[proxy_goodhart_under_search]]`,
`[[deterministic_ab_g4_oversell]]`, `[[ledger_ship_is_matrix_scoped]]`.
