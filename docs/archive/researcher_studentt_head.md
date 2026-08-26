# In-repo research brief — should Sportstradamus build a StudentT LSS head as a new continuous family?

**Question:** §8.2 hole #0a's fallback says "if a cell is genuinely g4-bound, a **StudentT LSS head** is the
cheaper continuous escalation — own brief + pilot before any build." Is the pilot evidence there? Does a
conditional-scale StudentT head close NFL receiving yards' Gate-4 wall?

**Scope:** `docs/handoffs/model_improvement_track.md` §6.6 / §8.2 hole #0a, family-escalation axis;
`docs/handoffs/shape_bound_triage.md` prior-art table.
Written 2026-08-26. Mode: read-only w.r.t. production (an NFL interceptions sweep was live throughout;
no `stat_meta`, model pickle, config, or serving path was touched. All fits ran at `nthread=4` in
`/tmp/claude-…/scratchpad`).

---

## TL;DR

1. **NO-GO on the StudentT build.** In a player-and-time-honest pilot on the cached NFL yards matrices,
   a LightGBMLSS `StudentT` head **never beats the incumbent SkewNormal recipe on any of the three
   cells at any blend weight**, and never reaches the decision number (holdout `pit_ks ≤ 0.045`).
   Receiving yards, best StudentT arm: **0.0616**; incumbent SkewNormal recipe: **0.0527** (screen),
   **0.0561** (production full-HPO, ledger 2026-08-26T10:49). Bar = 0.050.
2. **The mechanism is decisive and generalizes: the binding defect on these cells is *asymmetry on a
   non-negative support*, not tail weight.** A symmetric heavy tail buys right-tail mass only by paying
   an equal amount of *impossible* left-tail mass. Measured sub-zero leakage `E[F_base(0)]` at w=1:
   SkewNormal **0.062**, StudentT-NLL **0.161** — 2.6×. On a cell with `gate = 0.127` and a 17.2% holdout
   zero rate, that over-allocates ~9 points of probability to the y ≤ 0 region no outcome can occupy.
3. **The StudentT head has an internal, unavoidable trade between PIT and the mean.** Its NLL is a
   redescending M-estimator (Lange, Little & Taylor 1989, DOI 10.1080/01621459.1989.10478852), so the
   `loc` head fits a robust centre, not `E[Y]`. The two best-PIT StudentT arms carry a
   **−24.6% / −26.1% served-mean error** (21.8–22.2 vs 29.47 actual) — an instant Gate-2/3/6 kill.
   Switching to CRPS fixes the mean and **freezes the `df` head** at a single global value
   (p10 8.00 / median 8.16 / p90 8.31) — i.e. it degenerates into precisely the *global* shape oracle the
   in-repo NO-GO (`[[cdf_recal_nonstationary_pit]]`) already killed.
4. **The real lever is one constant.** `scorecard._DISPERSION_SKEW_BOUNDS = (-3.0, 3.0)` clamps the
   Lever-4a additive skew shift, and **receiving yards rails it on every full-HPO nominee**
   (`skew_cal` = 2.99892…2.999998 across all six 2026-08-26 confirms; 6 of 30 SkewNormal cells in
   `model_stats.parquet` rail it). Widening the bracket, val-fitted → holdout-scored, walks receiving
   yards' `pit_ks` **0.0568 → 0.0511 (cap 5) → 0.0475 (cap 8) → 0.0453 (cap 12)** at w=0.90, at **zero
   cost to g1/g2/g3/g5** (g1 `ci_hi` −0.00025 → −0.00042, BSS +0.0117 → +0.0123, debiased ECE
   0.0002 → −0.0012, g2/g3/CITL **bit-identical** — the (c,s) fit holds the served mean fixed).
   Projected onto the production nominee: **0.0570 → ≈0.0477 at cap 8.**
5. **StudentT ties SkewNormal exactly where it cannot ship, and loses everywhere it could.** On NFL
   passing yards (zero rate 0.000 on the holdout, near-symmetric CLT-shaped target) StudentT-CRPS 0.0402
   vs SkewNormal 0.0396 — a tie. Passing yards is g1-walled on 100% of its corners, so no family lever
   ships it. That is the regime statement: **the t head is competitive on un-gated symmetric cells and
   dominated on gated right-skewed ones**, and the gated right-skewed ones are the whole residual cohort.

---

## Key findings

### 1. Pilot design — what was replicated and what was simplified

Everything is derived from the cached production matrices
(`src/sportstradamus/data/training_data/NFL_{receiving,rushing,passing}-yards.parquet`) and reproduces the
production split and decode contract exactly:

| Production mechanic | Replicated? | Source |
|---|---|---|
| Temporal 70/30 split on `Date`, then `hash(Player, Date) < 2^63` → validation / eval | **yes, verbatim** | `pipeline._step_build_splits` |
| `ratio_meanyr` forward transform, `_MEANYR_FLOOR = 0.5`, `_RATIO_TARGET_FLOOR = 0.01` | **yes** | `baselines._ratio_forward` |
| Denominator = `resolve_denom_col(..., zero_inflated=True)` → **`MeanYr_nonzero`** | **yes** | `baselines.resolve_denom_col` |
| Zero rows dropped from TRAIN when `hist_gate > NONZERO_DENOM_GATE (0.05)` | **yes** | `pipeline._step_select_distribution:4025` |
| Scale-head clamp `[0.02, 10] × IQR/1.349`, alpha clamp ±30 | **yes** | `pipeline._skewnormal_dist_obj` |
| Per-row `start_values` re-seeded before every predict | **yes** | `pipeline.predict_lss_params` |
| Blend = `fused_loc` SkewNormal branch (precision-weighted pool, book leg `σ_b = ev_b·cv`, `α_b = 0`) | **yes, transcribed line-for-line** | `helpers.distributions.fused_loc:925-953` |
| `test_weight = where(authentic, w, 1.0)`; blended gate `= w_row · hist_gate` (book leg ungated, `gate_book = 0.0`) | **yes** | `pipeline._fuse_skewnormal:3428-3462` |
| Gated CDF `F = g + (1−g)F_base` for `y ≥ 0`, atom `P(Y=0) = g`, randomized over the atom, 25 seeded draws | **yes** | `scorecard._pred_cdf_pmf:1051-1065`, `_randomized_pit_draws` |
| `(c, skew_cal)` fitted on **validation**, scored on **eval** | **yes** | `scorecard.fit_skewnorm_dispersion_skew` |
| **Optuna HP search (300 trials)** | **no** — one fixed modest HP set shared by every arm | simplification |
| Temperature / `posthoc` / `hpo_selection` / structural corners | **no** | simplification |

**The simplification is validated, not assumed.** At the production skew cap (±3) the screen reproduces
the full-HPO ledger to three decimals on two independent corners:

| corner | w | screen `pit_ks` | production full-HPO `g4_pit_ks` | screen `c` | production `dispersion_cal` | `skew_cal` |
|---|---|---|---|---|---|---|
| `ratio_meanyr` / crps / direct / nll / none (incumbent, ledger row 2026-08-26T10:49) | 0.900 | **0.0568** | **0.0561** | 1.619 | 1.589 | railed 2.99999 both |
| `ratio_projvol` / nll / direct / crps / prob_recal_platt (ledger 2026-08-26T07:44) | 0.050 | **0.0717** | **0.0723** | 1.726 | 1.606 | railed 2.99996 both |

Per `[[deterministic_ab_g4_oversell]]` the honest reading is *direction and within-model delta*, not level —
but a 0.0006–0.0007 level agreement on two corners is unusually strong for a screen, and it is what makes
the cap projection in Finding 5 quotable.

### 2. StudentT loses on both gated cells and ties on the un-gated one

Validation-fitted scalar calibration → eval-scored randomized PIT-KS. Best row per arm across
`w ∈ {0.05, 0.10, 0.175, 0.50, 0.74, 0.90, 1.00}`. `n_eval` = 2250 / 1006 / 371.

**NFL receiving yards** (`gate = 0.1270`, holdout zero rate 0.1716, holdout mean 29.47, bar 0.050):

| arm | calibration | best w | eval `pit_ks` | eval tail-KS | cov50 | cov80 | served mean | mean error |
|---|---|---|---|---|---|---|---|---|
| **SkewNormal crps + (c, s)** — production recipe | c, s(±3) | 1.00 | **0.0527** | 0.0372 | 0.494 | 0.878 | 27.66 | −6.1% |
| SkewNormal crps + (c, s) | c, s(±3) | 0.90 | 0.0568 | 0.0568 | 0.466 | 0.852 | 26.44 | −10.3% |
| SkewNormal nll + (c, s) | c, s(±3) | 1.00 | 0.0601 | 0.0416 | 0.486 | 0.864 | 31.42 | +6.6% |
| **StudentT nll, MAD stabilization** | c | 1.00 | **0.0616** | 0.0587 | 0.509 | 0.856 | **21.77** | **−26.1%** |
| StudentT nll, `df ∈ [3, 50]` | c | 1.00 | 0.0721 | 0.0721 | 0.452 | 0.830 | **21.78** | **−26.1%** |
| StudentT nll | c | 1.00 | 0.0722 | 0.0722 | 0.441 | 0.830 | **22.23** | **−24.6%** |
| StudentT crps, `df ∈ [3, 50]` | c | 0.90 | 0.0740 | 0.0720 | 0.474 | 0.837 | 25.93 | −12.0% |
| StudentT crps | c | 0.90 | 0.0741 | 0.0707 | 0.474 | 0.839 | 25.95 | −11.9% |
| SkewNormal crps, **c-only** (t's handicap) | c | 0.90 | 0.0729 | 0.0688 | 0.515 | 0.850 | 26.44 | −10.3% |
| ZAGamma head (screen-grade, Finding 7) | c | 1.00 | 0.0553 | — | — | — | 28.37 | −3.7% |

**NFL rushing yards** (`gate = 0.1135`, holdout zero rate 0.1163, `n_eval` 1006):

| arm | best w | eval `pit_ks` |
|---|---|---|
| **SkewNormal nll + (c, s)** | 0.50 | **0.0606** |
| SkewNormal nll + (c, s) | 0.10 | 0.0623 |
| SkewNormal crps + (c, s) | 0.50 | 0.0694 |
| SkewNormal nll, c-only | 0.175 | 0.0740 |
| StudentT nll | 0.05 | 0.0802 |
| StudentT crps | 0.175 | 0.0808 |
| StudentT nll, MAD | 0.175 | 0.0829 |

**NFL passing yards** (`gate = 0.0006`, holdout zero rate **0.0000**, `n_eval` 371 — the un-gated control):

| arm | best w | eval `pit_ks` |
|---|---|---|
| SkewNormal crps + (c, s) | 0.175 | **0.0396** |
| SkewNormal crps, c-only | 0.175 | 0.0397 |
| **StudentT crps** | 0.90 | **0.0402** |
| StudentT crps, `df ∈ [3, 50]` | 1.00 | 0.0402 |
| StudentT nll, MAD | 0.175 | 0.0428 |
| StudentT nll | 1.00 | 0.0535 |

Three readings:

* **StudentT does beat a shape-free scalar recalibration.** On receiving yards the c-only SkewNormal is
  0.0729–0.0836 while StudentT reaches 0.0616–0.0741. The Mixture brief's cross-fit residual screen
  (`/tmp/researcher_mixture_serve.md`, Finding 1) replicates: *a heavier tail is a real improvement over
  refitting a scale.* That is not the comparison that decides a build.
* **The production comparison is against SkewNormal + the additive skew shift**, and StudentT loses to it
  by 17% (receiving), 32% (rushing) on the KS scale, at every blend weight.
* **On the one cell where the target is symmetric and un-gated, the two families tie.** That is the
  regime boundary, and it is on the wrong side of the residual cohort.

### 3. Why: a symmetric heavy tail is the wrong instrument on a non-negative right-skewed target

The mechanism table (w=1.00, receiving yards, validation-fitted calibration):

| arm | `c` | `s` | `E[F_base(0)]` — sub-zero leakage | mean PIT of the zero rows | served mean |
|---|---|---|---|---|---|
| SkewNormal crps + (c, s) | 1.677 | **+3.0 (railed)** | **0.0619** | 0.1412 | 27.66 |
| SkewNormal crps, c-only | 1.203 | 0 | 0.0984 | 0.1699 | 27.66 |
| StudentT crps | 1.189 | — | 0.0962 | 0.1654 | 27.03 |
| StudentT nll, MAD | 1.464 | — | 0.0739 | 0.1829 | 21.77 |
| **StudentT nll** | 1.223 | — | **0.1606** | 0.2046 | 22.23 |

Both families are unbounded below and the served predictive is
`F(y) = gate + (1 − gate)·F_base(y)` for `y ≥ 0` — so `(1 − gate)·F_base(0)` is probability mass parked on
**negative yardage**, which is unreachable for a season-total receiving-yards outcome. Total mass at or
below zero: SkewNormal `0.127 + 0.873×0.062 = 0.181` against a holdout zero rate of **0.172** — nearly
exact. StudentT-NLL: `0.127 + 0.873×0.161 = 0.267` — a **9.5-point over-allocation**.

The railed positive `skew_cal` is doing *two* jobs at once: it thickens the right tail **and** it pushes
mass off the negative half-line. A symmetric family gets the first and pays for it on the second. This is
the classic support-plus-asymmetry argument: the skew-normal was introduced precisely to add an asymmetry
degree of freedom to a Gaussian body (Azzalini 1985; Azzalini & Capitanio 1999,
DOI 10.1111/1467-9868.00194), and NFL yardage is documented as strongly skewed with a spike at/near zero
(Glazer, Parast & Hooten 2025, *The American Statistician*, DOI 10.1080/00031305.2025.2604812; the same
paper also names the yardage-rounding measurement error this repo's half-point `step` handling partly
absorbs). Modern athlete-performance work reaches for a **skew-t** for exactly this reason and explicitly
flags that the standard skew-t still ties the two tails together (Griffin et al., arXiv:2405.17214).

### 4. The StudentT head's own pathologies are live in this data, not theoretical

**(a) NLL fits a robust centre, not the mean — and the blend's currency is the mean.**
The t log-likelihood is Huber-like: its score is redescending, so extreme observations are down-weighted
(Lange, Little & Taylor 1989, DOI 10.1080/01621459.1989.10478852). On a right-skewed target the resulting
`loc` sits near the conditional median. `E[T] = loc` for `df > 1`, so decode is *arithmetically* correct
and *statistically* biased: −24.6% (T-nll) and −26.1% (T-nll-MAD) on the served mean. The repo's
`_fuse_skewnormal` / `fused_loc` / `_blend_with_book` all take `ev_a` = the model's mean, and Gates 2, 3
and 6 all score the fused mean. A −25% mean is not a calibration nit — it is three failed gates.

**(b) CRPS repairs the mean and kills the `df` head.** `lightgbmlss` sets the Hessian to 1 on the CRPS
path (`StudentT` docstring: *"if 'crps' is used, the Hessian is set to 1 … using the CRPS disregards any
variation in the curvature"*), so the `df` head receives gradient with no curvature and converges to a
single global value:

| arm | `df` p10 | `df` median | `df` p90 | frac `df > 30` | frac `df < 4` |
|---|---|---|---|---|---|
| StudentT **crps** | 8.00 | **8.16** | 8.31 | 0.000 | 0.020 |
| StudentT **nll** | 8.45 | **61.0** | 286.4 | **0.716** | 0.040 |
| StudentT nll, `df ∈ [3, 50]` | 9.83 | 39.7 | 50.0 | 0.618 | 0.020 |

A frozen `df` is a **global** tail index with a conditional scale — which is materially the object the
in-repo StudentT NO-GO already measured. The distinction the Mixture brief drew ("a global hand-fit shape
oracle is not a conditional-scale LSS head") is correct in principle and **empty in practice on the CRPS
path**, which is the loss the incumbent recipe uses. On the NLL path the head *is* conditional — and it
sends 72% of rows to `df > 30`, i.e. effectively Normal, while the minority tail rows drive the mean bias.
That is the textbook two-sided `df` instability: weakly identified, with an unbounded likelihood at the
boundary of the parameter space (Fernández & Steel 1999, Biometrika 86(1):153–167,
DOI 10.1093/biomet/86.1.153) and improper posteriors under the obvious non-informative priors
(Fonseca, Ferreira & Migon 2008, Biometrika 95(2):325–333, DOI 10.1093/biomet/asn001). Practitioner
packages respond by hard-bounding `ν` — rugarch 2.1, fGarch 2, GAS 4 — with no agreement on the value
(arXiv:2510.09785; see also arXiv:1910.01398 for measured df-estimation instability in Student-t GARCH).

**(c) The clamp precedent works but does not rescue the arm.** `lightgbmlss`'s `exp_fn_df`
(`utils.py`: `exp(raw) + 1e-6 + 2.0`) already enforces `df > 2`, so the variance always exists — the
"df→small blows the variance" half of the dispatch's worry is pre-solved by the library. The
`df → ∞` half is not, and it is the one that fires. Mirroring `hyperparams._BoundedResponseFn` with
`floor=3.0, ceiling=50.0` (arm `T_nll_c3_50`) changed the answer by **0.0001** (0.0722 → 0.0721): the
clamp is correct hygiene and not a lever.

**(d) `stabilization="MAD"` is the one StudentT knob that helped**, moving nll from 0.0722 to 0.0616 —
consistent with the boosted-GAMLSS literature, where MAD is `gamboostLSS`'s robust default because L2
divides gradients by an RMS that heavy-tailed rows inflate (Mayr et al. 2012,
DOI 10.1111/j.1467-9876.2011.01033.x; Hofner, Mayr & Schmid 2016, DOI 10.18637/jss.v074.i01). It did not
fix the mean (−26.1%), and it is a **free-standing finding for §8.2 #9(c)**: MAD beat L2 and "None" on this
family and cell by 15% on PIT-KS, so the `--stabilization` axis now has one measured win behind it.

### 5. The blend does **not** dilute a tail — but it does not help either, and the real lever is the ±3 skew clamp

**Answer to "does low `w` dilute the StudentT tail?" — No, and that is the point.** `fused_loc` is a
*parameter* pool, not a density pool: it precision-weights `(loc, σ)` and linearly blends the shape
(`α_blend = w·α_model + (1−w)·0`). A StudentT analogue would pool the moment-matched sd and re-express
the result as `t(df_model, μ_blend, σ_blend/√(df/(df−2)))`, so **the served tail index is 100% the model's
at every `w`, including the 0.05 floor**. The blend moves only location and width. What the blend *does* do
is (i) pull the location toward the book — helpful for g1, and (ii) shrink the served zero gate to
`w · hist_gate` (the SkewNormal book leg is ungated, `gate_book = 0.0`, `pipeline._fuse_skewnormal:3429`),
so at `w = 0.05` the served zero mass is **0.0064** against a 17.2% holdout zero rate. That is why every
arm's PIT-KS degrades monotonically as `w → 0` (SkewNormal 0.0527 → 0.0717; StudentT 0.0722 → 0.0886), and
it is a *shared* penalty that no family choice changes.

Meanwhile the incumbent's actual constraint is one module-level constant:

```python
# scorecard.py:181-188
_DISPERSION_C_BOUNDS: tuple[float, float] = (0.1, 10.0)
_DISPERSION_SKEW_BOUNDS: tuple[float, float] = (-3.0, 3.0)
```

**Receiving yards rails `s` at +3.0 on every single full-HPO nominee** (`skew_cal` 2.998982, 2.999210,
2.999958, 2.999989, 2.999998 in `research/confirm_nominee_gates.csv`, 2026-08-23 → 2026-08-26), and so do
5 other SkewNormal cells in `model_stats.parquet` (NFL rushing yards 2.999892, NFL fantasy points
prizepicks 2.999926, NFL fantasy points underdog 2.999511, NFL receptions 2.999839, NFL yards 2.999712,
NHL skater fantasy points underdog 2.998108; NBA AST 2.388 and NFL targets 2.583 are near-railed).

Widening the bracket, honestly val-fitted → eval-scored:

| `s` cap | w=0.90 `c` | w=0.90 val KS | **w=0.90 eval KS** | w=0.175 eval KS | w=0.05 eval KS | w=1.00 eval KS |
|---|---|---|---|---|---|---|
| 1 | 1.309 | 0.0675 | 0.0679 | 0.0820 | 0.0859 | 0.0769 |
| 2 | 1.491 | 0.0570 | 0.0629 | 0.0745 | 0.0785 | 0.0643 |
| **3 (today)** | 1.619 | 0.0512 | **0.0568** | 0.0681 | 0.0717 | 0.0527 |
| **5** | 1.732 | 0.0449 | **0.0511** | 0.0622 | 0.0653 | 0.0414 |
| **8** | 1.798 | 0.0411 | **0.0475** | 0.0593 | 0.0620 | 0.0351 |
| 12 | 1.833 | 0.0390 | 0.0453 | 0.0579 | 0.0609 | 0.0345 |
| 30 | 1.850 | 0.0377 | 0.0443 | 0.0574 | 0.0604 | 0.0332 |

The val→eval gap stays ≈0.006 and does **not** widen with the cap — at cap 30 the validation gain
(0.0512 → 0.0377) stops transferring (eval 0.0568 → 0.0443) exactly where the theory says it should:
`δ = α/√(1+α²)` is 0.949 at α=3, 0.992 at α=8, 0.9994 at α=30, so past ~8 the shape barely moves and only
noise is being fitted. **The transferable gain saturates at cap ≈ 8.**

And it is free on every other gate (same screen, w=0.90, line-only temperature fitted on validation):

| `s` cap | eval `pit_ks` | g1 mean | g1 `ci_hi` | BSS | g5 debiased ECE | g2 star z | g3 bench z | CITL |
|---|---|---|---|---|---|---|---|---|
| 3 | 0.0568 | −0.00295 | −0.00025 | 0.0117 | +0.0002 | 0.1053 | 0.0970 | 0.8972 |
| 5 | 0.0511 | −0.00307 | −0.00040 | 0.0122 | −0.0007 | 0.1053 | 0.0970 | 0.8972 |
| 8 | 0.0475 | −0.00310 | −0.00042 | 0.0123 | −0.0012 | 0.1053 | 0.0970 | 0.8972 |
| 12 | 0.0453 | −0.00311 | −0.00046 | 0.0123 | −0.0015 | 0.1053 | 0.0970 | 0.8972 |

g2 / g3 / CITL are **bit-identical** across caps by construction: `fit_skewnorm_dispersion_skew` re-derives
`loc` from a held-fixed mean, so the skew shift is mean-preserving. g1 and g5 both improve slightly. This is
the rare shape lever that touches exactly one gate.

**Projection onto the production nominee.** Using the within-model ratio (cap 8 / cap 3 = 0.0475/0.0568 =
0.836) and the ledger's best g4-only corner (`ratio_projvol` / nll / direct / nll / prob_recal_platt,
w=0.900, 2026-08-26T05:42: g1 `ci_hi` +0.0005, g2 0.0858, g3 0.0156, **g4 0.0570**, g5 −0.0061, g6 pass):

| `s` cap | projected `g4_pit_ks` | vs bar 0.050 |
|---|---|---|
| 3 (today) | 0.0570 | fail |
| 5 | 0.0513 | coin flip |
| **8** | **0.0477** | **pass, ~0.0023 margin** |
| 12 | 0.0455 | pass, no extra transferable gain |

**This is not gate-loosening.** `_DISPERSION_SKEW_BOUNDS` is the *calibrator's search bracket*, not a gate
threshold; `_GATE4_PIT_KS_DELTA` and the `1.358/√n` noise floor are untouched, and a widened candidate
still has to clear `pit_ks < 0.050` on an untouched holdout. §8.1's "never loosen a gate to hit breadth"
is not engaged. §4's owner-only rule *is* engaged in spirit because the constant lives in `scorecard.py`
beside the gate constants — treat it as an owner sign-off item, not a session edit.

### 6. The bracket's own rationale survives the widening — with one caveat to preserve

The in-code justification is: *"|s| ≤ 3 keeps the served skewness well inside the SkewNormal's range and
bounds the 2-param fit's capacity at ~2k calibration rows (the skewness MLE is only n^(1/4)-consistent
near alpha=0 — Hallin & Ley 2014 — so an unbounded shift overfits the gate's own KS)."*

Both halves need re-reading against the measurement:

* **The Hallin & Ley singularity is at α = 0, not at α = 8.** The skew-normal Fisher information is singular
  *at symmetry*, which is what degrades the skewness estimate to n^(1/4) (or worse for generalized families)
  — Hallin & Ley 2014, *Bernoulli* 20(3):1432–1453, DOI 10.3150/13-BEJ528; arXiv:1209.4177. A cell whose
  fit **rails away from zero** is the opposite regime. What *does* bite at large |α| is the well-known
  monotone-likelihood divergence of the skew-normal MLE (`α̂ → ±∞` with positive probability;
  Azzalini & Capitanio 1999, DOI 10.1111/1467-9868.00194) — and that is a *flatness*, not an
  overfit: the measured eval KS flattens at 0.0453 → 0.0443 from cap 12 → 30 while validation keeps
  falling, which is that flatness showing up as unearned validation gain.
* **The overfit worry is real but bounded, and it is measurable.** A finite cap is the right control; the
  evidence says the correct value is **≈8** (where transfer stops), not 30 (uncapped) and not 3.
  `_DISPERSION_SKEW_MIN_GAIN = 0.008` already refuses a shift that does not beat scale-only by the
  measured val→test discount, and the cap-3 → cap-8 gain (0.0093 eval / 0.0101 val) clears it.

### 7. The positive-support family alternative is *not* better — and that matters for the routing rule

Since the diagnosis is "the cell wants a strongly right-skewed, near-half-normal shape on `[0, ∞)`", the
obvious alternative is a zero-adjusted Gamma — which is **already wired end-to-end on the serve side**
(`get_ev` / `get_odds` `_gamma_odds` with the ZA gate, `fused_loc` Gamma branch, `_blend_with_book` else-branch,
`_zi_kwargs` ZAGamma, `_model_predictive_sd`, `_annotate_display_shape`, `decode_predictive_mean`,
`scorecard._pred_cdf_pmf`) and carries `capabilities=_SERVE_ONLY` in `specs._COMPATIBILITY_SPECS`. Only the
training branch is missing (`_step_select_distribution` routes `dist not in ("SkewNormal","Mixture")` to the
count branch, and `Gamma` is not in `_FORCEABLE_DISTS`).

I screened it. **ZAGamma reaches 0.0553 at w=1.00 and 0.0729 at w=0.90** — i.e. it ties the cap-3 SkewNormal
and loses badly to the cap-8 one (0.0351 / 0.0475). Its served mean is the best of any arm (28.37 vs 29.47,
−3.7%). Screen-grade only (untuned seeding, one HP set), so read it as *not obviously better*, not as a kill.

The conclusion that survives: **the amount of right skew is the lever, and the incumbent family can already
supply it once the clamp is lifted.** No new family is required to fix receiving yards.

### 8. Corroboration: the trained α head is inert, so the post-hoc scalar *is* the shape instrument

The screen reproduces §6.6's headline diagnosis independently: the SkewNormal `alpha` head trains to
≈0 on this cell (p10 −0.134, median **+0.006**, p90 +0.285; 0% railed at the ±30 training clamp) while the
post-hoc `skew_cal` rails at its ±3 bound. So on receiving yards the entire served skewness is a **single
cell-level scalar**, and the boosted per-row shape head contributes nothing — the α=0 Fisher singularity
(Arellano-Valle & Azzalini 2008, DOI 10.1007/s00184-007-0131-x; Hallin & Ley 2014) live in the fitted model,
exactly as §6.6 states. Two consequences worth carrying:

* the centered-SN rung (`sn_param: "centered"`) targets the *head*, and on this cell the head is not where
  the shape lives — which is consistent with the two centered nominees confirming at 0.0572 / 0.0581, no
  better than direct;
* widening the scalar's bracket is therefore not "more post-hoc" — it is the only place this cell's shape
  is representable at all today.

---

## Recommendation

### Verdict: **NO-GO on the StudentT LSS head.** Do not add a sixth trainable family for this cohort.

Close §8.2 hole #0a's StudentT fallback as **piloted and refuted for the gated right-skewed cohort**, with a
narrow, pre-registered survival clause (below). The reason is not that the family is unsound — it is that on
every cell where a family lever could ship, StudentT is dominated by a one-constant change to machinery that
already exists, and on the one cell where it ties, no family lever can ship.

### Routing protocol for the continuous g4-bound cohort (implementable)

Apply in order; stop at the first pass.

| Step | Condition on the cell | Action |
|---|---|---|
| **0** | any SkewNormal cell failing only g4 | check `skew_cal` in `model_stats.parquet` / the confirm ledger |
| **1** | `\|skew_cal\| ≥ 2.99` (railed) **and** g4 is the sole failing gate | **widen `_DISPERSION_SKEW_BOUNDS` to (−8, +8)** and re-confirm. Owner sign-off (constant lives beside the gate constants). Expected: `pit_ks × ≈0.84`. Zero cost to g1/g2/g3/g5; g2/g3/CITL provably unchanged |
| **2** | `\|skew_cal\| < 2.99` (not railed) and g4-bound | the cell is not skew-clamp-bound — stay on the existing §6.1/§6.2 axes; do **not** widen |
| **3** | still g4-bound after step 1, `zi ≤ 0.02`, holdout residual near-symmetric (cov50 ≈ cov80 direction agrees) | StudentT is *permissible* as a research arm — but note that such a cell is, empirically, also g1-walled |
| **4** | still g4-bound, `zi > 0.05`, right-skewed | **route to skew-t / SHASH**, not to StudentT. This is the family class the evidence points at |

**Blast radius of step 1 today: 7 cells.** NFL receiving yards (withheld — a first-ship, Tier-0 gates only)
plus 6 currently `shipped: "devel"` cells (NFL rushing yards, receptions, yards, fantasy points ×2, NHL
skater fantasy points underdog), which are **supersessions** and must clear `supersede_verdict`'s S1/S2/S3,
not just the six gates. Sequence receiving yards first — it is the only one with no incumbent to beat.

### If the verdict is ever reopened — the complete StudentT build list

Named for completeness; every item is real work the dispatch's sketch under-counts.

| # | Item | Detail |
|---|---|---|
| 1 | `_FORCEABLE_DISTS` + `_resolve_dist` | add `"StudentT"` (`pipeline.py:159`), else `stat_meta` rejects it |
| 2 | `_continuous_dist_obj` branch | `StudentT(stabilization=…, response_fn="exp", loss_fn=…)`; **`_BoundedResponseFn` on `scale`** (mirror the SN `[0.02, 10]×IQR/1.349` clamp) **and on `df`** (`floor=3.0, ceiling=50.0`). Note `exp_fn_df` already guarantees `df > 2` |
| 3 | `zero_adjusted_continuous` | must include `"StudentT"` — **the Mixture kill's zero-gate hole** (`pipeline.py:4024`). Receiving yards `zi = 0.127`, rushing `0.113`, so this is load-bearing, not hypothetical |
| 4 | `set_model_start_values` + `_skewnormal_start_values` sibling | 3-column raw seed `[log(df₀−2), loc, log(scale)]`; must honour `normalized` / `offset_mode`. **`_serve_offset_mode` hardcodes `dist == "SkewNormal"`** (`model_prob.py:590`) — the `[[serve_decode_drift_offset_mode]]` class fires on any non-SN continuous family |
| 5 | `decode_predictive_mean` | `ev = loc` (exact for `df > 1`); publish `sigma`-equivalent **sd = `scale·√(df/(df−2))`**, and persist `df` as a new pickle field + a `Model DF` dump column (`_build_filedict`, legacy default, byte-identical round-trip test) |
| 6 | `fused_loc` StudentT branch | pool **moment-matched sd**, not raw `scale` (a t `scale` is not a σ); re-express as `t(df_model, μ_blend, sd_blend/√(df/(df−2)))`. Decide and document whether `df` blends (recommend: no — the book leg has no tail index) |
| 7 | `get_odds` / `get_ev` | `scipy.stats.t` branch with the half-point `step` correction and the ZA gate, mirroring `_skewnormal_odds`; `get_ev`'s `brentq` bracket must stay monotone in the mean |
| 8 | `_blend_with_book`, `_dispersion_calibrate` (`Model Sigma` analogue), `_model_predictive_sd`, `_annotate_display_shape`, `_zi_kwargs`, `_serving_dispersion_cal` | six serve sites (DPO needed nine) |
| 9 | `scorecard._pred_cdf_pmf` + `_decode_sn_loc_scale` sibling + `_served_*_pit_ks` calibrator | Gate-4 must price the served t, including the gate and the zero atom |
| 10 | `specs.py` FamilySpec | axes `{dist, normalization, dist_training_loss, blending_loss_fn, posthoc}`; **`stabilization` should be a real axis here** (MAD won by 15% — Finding 4d); persist `dist` + `df` policy; `capabilities` starts `_RESEARCH` |
| 11 | `ship_config._validate_cell` / `CONTINUOUS_DISTS` | add to the continuous set so the `target_normalization != "none"` invariant applies |
| 12 | Live-path integration test | mirroring `tests/integration/test_zinb_hurdle_live_path.py` — §7.3 hard ship gate |
| 13 | Confirm-time diagnostics | `frac(df > 30)`, `frac(df < 4)`, and **`mean(served)/mean(Result)`** — the −25% mean bias must fail loud, not surface as a g2 near-miss |

**Cost: ~3 sessions + 1 pilot session.** This is an **engineering project** (known method, existing scaffold),
not a research project — which is exactly why the NO-GO is about payoff, not risk.

### Next-cheapest receiving-yards levers, ranked

1. **Widen `_DISPERSION_SKEW_BOUNDS` to ±8** (~1 constant + goldens + owner sign-off + 1 confirm).
   Projected `g4 0.0570 → ≈0.0477`. Every other gate unchanged or better. **This is the recommendation.**
2. **Re-confirm the `ratio_projvol` / `prob_recal_platt` / w=0.90 corner under the widened cap** — it is the
   cell's only nominee that fails *only* g4 (g1 `ci_hi` +0.0005, g6 pass). Carry the
   `[[ratio_projvol_refuted]]` caveat: that normalization is refuted where volume can hit zero, and NFL
   targets can be zero, so a `ratio_meanyr` arm must be run beside it.
3. **skew-t / SHASH** as the family escalation if 1–2 fail. SHASH separates skewness (ε) from tail weight (δ)
   in one family (Jones & Pewsey 2009, DOI 10.1093/biomet/asp053), which is precisely the two-axis control
   this evidence asks for; §6.6's skew-t deferral (torch lacks `betainc`) does not apply to SHASH.
4. **A trainable ZAGamma branch** — cheapest *new* family by build cost (serve path complete, ~1 session),
   but measured no better than the cap-3 incumbent, so only worth it if 1–3 all fail.
5. **Gate re-estimation is ruled out.** Replacing the published `hist_gate` (train zero rate 0.127) with the
   validation zero rate (0.173) makes eval KS *worse* (0.0526 → 0.0540); a freely-fitted gate buys only
   0.0526 → 0.0507. The train/holdout zero-rate drift is real but is not the binding defect.

---

## Reality checks

* **Effect size and its regime.** The cap-8 projection (`0.0570 → ≈0.0477`) is a *within-model ratio*
  transported from a fixed-HP screen to a 300-trial full-HPO artifact. It holds because the screen matched
  production's cap-3 KS to 0.0006–0.0007 on two independent corners **and** matched its fitted `c` (1.619 vs
  1.589) and its railed `s`. It would not hold if the full-HPO fit changed the *shape* of the residual, and a
  ~0.0023 projected margin is thin. **Read it as "worth exactly one confirm", not as a ship.**
* **The StudentT NO-GO is strongest on the gated cells and weakest on passing yards.** On passing yards
  StudentT-CRPS (0.0402) edges SkewNormal-c-only (0.0397 is a tie) and my n is 371. If a future *un-gated,
  near-symmetric, g4-only* continuous cell appears that is not g1-walled, this brief does not cover it —
  step 3 of the routing protocol is the honest carve-out.
* **My screen is not a confirm.** One fixed HP set, no Optuna, no temperature on the KS arms, no posthoc, no
  structural corners, no cross-fit. `[[deterministic_ab_g4_oversell]]` and
  `[[crossfit_board_ships_optimistic]]` both apply. Every absolute level here is a screen level.
* **The g1/g5 probe railed its temperature at the bound (T = 4.0)** on my low-capacity model, so its absolute
  BSS (+0.012) and CITL (0.897) are screen artifacts. The *deltas across `s` caps* are the finding, and they
  are computed inside one fitted model with everything else held fixed.
* **A sweep was live the whole time.** `research/confirm_nominee_gates.csv` and
  `strategy_research_board.csv` were being written during this session (board `swept_at`
  2026-08-26T09:41, ledger rows through 10:49). Re-derive the receiving-yards standings from §3 before
  acting; the six confirms I read may not be the last six.
* **What would make this brief wrong.** (a) The full-HPO fit at cap 8 lands at 0.051–0.053 rather than 0.048
  — the projection is a ratio, and a 4% error in it flips the verdict on *this cell* (it would not
  resurrect StudentT). (b) NFL receiving yards turns out to be g6-bound as well once the served mean is
  pinned (`[[gate_off_probe_confounds_mean]]` — my screen's CITL 0.897 is model-capacity, but the
  incumbent artifact does currently fail g6). (c) Widening the cap to ±8 regresses one of the six
  `shipped: devel` railed cells at supersession — which is why step 1 is sequenced one cell at a time,
  receiving yards first.
* **Research project vs engineering project.** The StudentT build is a fully-specified *engineering*
  project (13 items, ~3–4 sessions, no unknowns). The skew-t/SHASH escalation in step 4 is a *research*
  project — hand-rolled torch class, no in-repo precedent for the density, and unproven on this data.
  The skew-cap widening is neither: it is a one-line owner decision with measured evidence.

---

## Open questions / caveats

1. **Is ±8 the right cap, or should the bracket be data-adaptive?** The transfer saturates at ~8 on
   receiving yards. Whether that number is cell-specific (it is a function of how far the fitted `α` must
   travel, which depends on `zi` and the target's skewness) or universal is unmeasured. A defensible
   alternative is to keep the bracket wide and lean harder on `_DISPERSION_SKEW_MIN_GAIN` plus the existing
   val→test discount — but that trades a hard bound for a soft one on the gate's own statistic.
2. **Does the widened cap regress any of the 6 railed `shipped: devel` cells?** All six would re-fit to a
   larger `s`. The mean is preserved by construction, so g2/g3/g6 are safe; g1/g5 improved slightly here but
   were not measured on those cells. Each needs its own S1/S2/S3 supersession pass.
3. **`_model_predictive_sd` returns the SkewNormal `scale`, not the sd.** At `α = 8` the true sd is
   `scale·√(1 − 2δ²/π) ≈ 0.60·scale`, so `_sanitize_book_ev`'s plausibility band is ~66% too wide and gets
   wider as the cap opens. Pre-existing, cosmetic today, but it is a real train/serve semantic drift that a
   widened cap amplifies.
4. **Should `--stabilization` become a swept axis now?** §8.2 #9(c) says "only if MAD/L2 ever wins on ≥1
   cell". MAD beat both L2 and None on the StudentT arm by 15% on PIT-KS. That is one win on a family that
   is not being built — arguably it discharges the YAGNI condition, arguably it does not. Owner call.
5. **Is the ZAGamma screen's 0.0553 real?** My Gamma arm used untuned start values and one HP set (its first
   parametrization diverged outright). If the routing protocol ever reaches step 4, re-screen it properly
   before costing the build.
6. **The July "upper-tail under-dispersion" diagnosis needs restating.** Under today's archive, blend, and
   `(c, s)` calibration, receiving yards' `tail_pit_ks` equals its whole-CDF `pit_ks` on most corners (so the
   sup *is* in the over-tail), but the incumbent's `cov80 = 0.851 > 0.80` says the far tail is now **too
   heavy**, not too light — the July reading predates the joint `(c, s)` fit. The residual is asymmetry plus
   support, not raw tail weight. Worth a one-line correction wherever the old framing is quoted.
7. **Does the `zi` gate belong on the *fused* side?** The book leg is ungated (`gate_book = 0.0`), so a
   low-`w` corner serves a near-zero zero-mass on a 17%-zero cell. That is a structural blend question, not
   a family question, and it caps how good any family can look at `w → 0` (every arm here degraded
   monotonically as `w` fell). It is the natural companion to `[[mean_corrector_belongs_post_fusion]]`.

---

## Bibliography

| # | Source | Identifier | Used for |
|---|---|---|---|
| 1 | Lange, K., Little, R.J.A. & Taylor, J.M.G. (1989). Robust statistical modeling using the t distribution. *JASA* 84(408):881–896 | DOI 10.1080/01621459.1989.10478852 | t-likelihood is a redescending M-estimator; the `loc` head fits a robust centre, not `E[Y]` — the −25% mean bias |
| 2 | Fernández, C. & Steel, M.F.J. (1999). Multivariate Student-t regression models: pitfalls and inference. *Biometrika* 86(1):153–167 | DOI 10.1093/biomet/86.1.153 | Likelihood unbounded at the boundary of the t parameter space; global ML maximization is vacuous |
| 3 | Fonseca, T.C.O., Ferreira, M.A.R. & Migon, H.S. (2008). Objective Bayesian analysis for the Student-t regression model. *Biometrika* 95(2):325–333 | DOI 10.1093/biomet/asn001 | `df` is weakly identified; improper priors → improper posteriors; motivates a hard `df` bound |
| 4 | Azzalini, A. (1985). A class of distributions which includes the normal ones. *Scand. J. Statist.* 12(2):171–178 | JSTOR 4615982 | The skew-normal: an asymmetry parameter on a Gaussian body |
| 5 | Azzalini, A. & Capitanio, A. (1999). Statistical applications of the multivariate skew-normal distribution. *JRSS-B* 61(3):579–602 | DOI 10.1111/1467-9868.00194 | SN MLE diverges to `α̂ = ±∞` with positive probability — the flatness behind the cap-12→30 non-transfer |
| 6 | Azzalini, A. & Capitanio, A. (2003). Distributions generated by perturbation of symmetry, with emphasis on a multivariate skew t-distribution. *JRSS-B* 65(2):367–389 | DOI 10.1111/1467-9868.00391 | The skew-t: skewness **and** tail weight, the family this evidence points at |
| 7 | Hallin, M. & Ley, C. (2014). Skew-symmetric distributions and Fisher information: the double sin of the skew-normal. *Bernoulli* 20(3):1432–1453 | DOI 10.3150/13-BEJ528 · arXiv:1209.4177 | Fisher singularity is **at α = 0**; n^(1/4) rate there — the ±3 cap's stated rationale, re-scoped |
| 8 | Arellano-Valle, R.B. & Azzalini, A. (2008). The centred parametrization for the multivariate skew-normal distribution. *Metrika* 68:31–46 (with the CP/DP inversion) | DOI 10.1007/s00184-007-0131-x | The α=0 singularity the centered-SN rung targets; corroborates Finding 8 |
| 9 | Jones, M.C. & Pewsey, A. (2009). Sinh-arcsinh distributions. *Biometrika* 96(4):761–780 | DOI 10.1093/biomet/asp053 | SHASH separates skewness (ε) from tail weight (δ) — the two-axis control the routing protocol's step 4 wants |
| 10 | Jones, M.C. & Faddy, M.J. (2003). A skew extension of the t-distribution, with applications. *JRSS-B* 65(1):159–174 | DOI 10.1111/1467-9868.00378 | Alternative skew-t construction |
| 11 | Zhu, D. & Galbraith, J.W. (2010). A generalized asymmetric Student-t distribution with application to financial econometrics. *J. Econometrics* 157(2):297–305 | DOI 10.1016/j.jeconom.2010.01.013 | Two independent tail parameters — the only t-family variant that could hold "heavy right, bounded left" |
| 12 | Rigby, R.A. & Stasinopoulos, D.M. (2005). Generalized additive models for location, scale and shape. *JRSS-C* 54(3):507–554 | DOI 10.1111/j.1467-9876.2005.00510.x | The distributional-regression foundation |
| 13 | Stasinopoulos, M.D., Rigby, R.A. & De Bastiani, F. (2018). GAMLSS: a distributional regression approach. *Statistical Modelling* 18(3–4):248–273 | DOI 10.1177/1471082X18759144 | Family-selection practice in GAMLSS; skew/kurtosis families |
| 14 | Kneib, T., Silbersdorff, A. & Säfken, B. (2023). Rage against the mean — a review of distributional regression approaches. *Econometrics and Statistics* 26:99–123 | DOI 10.1016/j.ecosta.2021.07.006 | Survey; family choice as the dominant modelling decision |
| 15 | Mayr, A., Fenske, N., Hofner, B., Kneib, T. & Schmid, M. (2012). GAMLSS for high-dimensional data — a flexible approach based on boosting. *JRSS-C* 61(3):403–427 | DOI 10.1111/j.1467-9876.2011.01033.x | Boosted distributional regression; the MAD/L2 stabilization lineage |
| 16 | Hofner, B., Mayr, A. & Schmid, M. (2016). gamboostLSS: an R package for model building and variable selection in the GAMLSS framework. *JSS* 74(1):1–31 | DOI 10.18637/jss.v074.i01 | MAD is the package default for robustness — Finding 4(d) |
| 17 | Thomas, J., Mayr, A., Bischl, B., Schmid, M., Smith, A. & Hofner, B. (2018). Gradient boosting for distributional regression: faster tuning and improved variable selection via noncyclical updates. *Stat. Comput.* 28:673–687 | DOI 10.1007/s11222-017-9754-6 | One round schedule shared by all heads is a known defect — the `df`-head freeze under CRPS |
| 18 | März, A. & Kneib, T. (2022). Distributional gradient boosting machines | arXiv:2204.00778 | The LightGBMLSS framework this repo runs |
| 19 | März, A. (2019). XGBoostLSS — an extension of XGBoost to probabilistic forecasting | arXiv:1907.03178 | Predecessor; the `loss_fn="crps"` Hessian-1 convention |
| 20 | Gneiting, T., Balabdaoui, F. & Raftery, A.E. (2007). Probabilistic forecasts, calibration and sharpness. *JRSS-B* 69(2):243–268 | DOI 10.1111/j.1467-9868.2007.00587.x | Maximize sharpness subject to calibration — the PIT paradigm Gate 4 implements |
| 21 | Gneiting, T. & Raftery, A.E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA* 102(477):359–378 | DOI 10.1198/016214506000001437 | Proper scoring; already in the repo's base |
| 22 | Brockwell, A.E. (2007). Universal residuals: a multivariate transformation. *Statist. Probab. Lett.* 77(2):143–147 | DOI 10.1016/j.spl.2007.02.008 | Randomized PIT — Gate 4's statistic, replicated verbatim for the zero atom |
| 23 | Jordan, A., Krüger, F. & Lerch, S. (2019). Evaluating probabilistic forecasts with scoringRules. *JSS* 90(12):1–37 | DOI 10.18637/jss.v090.i12 | Closed-form CRPS for the Student-t — the analytic alternative to lightgbmlss's sampled CRPS |
| 24 | Ranjan, R. & Gneiting, T. (2010). Combining probability forecasts. *JRSS-B* 72(1):71–91 | DOI 10.1111/j.1467-9868.2009.00726.x | Recalibrate the combination, not the legs — why the `(c, s)` fit sits post-fusion |
| 25 | Genest, C. & Zidek, J.V. (1986). Combining probability distributions: a critique and an annotated bibliography. *Statist. Sci.* 1(1):114–135 | DOI 10.1214/ss/1177013825 | The pooling operator `fused_loc` implements |
| 26 | Hinton, G.E. (2002). Training products of experts by minimizing contrastive divergence. *Neural Comput.* 14(8):1771–1800 | DOI 10.1162/089976602760128018 | Gaussian product-of-experts = the precision-weighted pool; why the parameter pool keeps the model's shape at every `w` |
| 27 | Glazer, A.K., Parast, L. & Hooten, M.B. (2025). Beyond the yard line: accommodating rounded sports data in statistical models. *The American Statistician* | DOI 10.1080/00031305.2025.2604812 | NFL yardage is rounded, skewed, and spiked — the domain fact behind the support argument |
| 28 | Griffin, J.E. et al. (2024). Modelling between- and within-season trajectories in elite athletic performance data | arXiv:2405.17214 | Skew-t for athlete performance, with the explicit caveat that its two tails are tied together |
| 29 | Charalampos, S. et al. (2019). The effects of degrees-of-freedom estimation in the asymmetric GARCH model with Student-t innovations | arXiv:1910.01398 | Measured `df`-estimation instability in a comparable LSS-style setting |
| 30 | (preprint) The pitfalls of continuous heavy-tailed distributions in high-frequency data analysis (2025) | arXiv:2510.09785 | Package-level `df` lower bounds (rugarch 2.1, fGarch 2, GAS 4) — no agreed value; supports a hard clamp |

**In-repo evidence read for this brief (not literature):**
`docs/handoffs/model_improvement_track.md` §3/§3.4/§6.6/§7.1/§7.2/§7.3/§7.4/§8.1/§8.2/§10 ·
`docs/ship_gate.md` · `docs/handoffs/shape_bound_triage.md` ·
`docs/archive/researcher_mixture_serve.md` (= `/tmp/researcher_mixture_serve.md`) ·
`src/sportstradamus/data/research/briefs/researcher_nfl_breadth_20260719.md` ·
`src/sportstradamus/training/pipeline.py` (`_step_build_splits`, `_step_select_distribution`,
`_skewnormal_dist_obj`, `_continuous_dist_obj`, `_resolve_dist`, `_FORCEABLE_DISTS`,
`_step_decode_predictions`, `_fuse_skewnormal`, `_step_calibrate_dispersion`, `predict_lss_params`) ·
`src/sportstradamus/training/scorecard.py` (`_pred_cdf_pmf`, `_randomized_pit_draws`, `_ks_uniform`,
`_tail_ks_uniform`, `_dispersion_diagnostics`, `fit_skewnorm_dispersion_skew`,
`_DISPERSION_SKEW_BOUNDS`, `_DISPERSION_SKEW_MIN_GAIN`, `_gate4_pit_ks_threshold`) ·
`src/sportstradamus/training/baselines.py` (`_ratio_*`, `resolve_denom_col`, `_TARGET_NORMALIZATIONS`) ·
`src/sportstradamus/training/hyperparams.py` (`_BoundedResponseFn`) ·
`src/sportstradamus/training/model_strategy/specs.py` (`_BASE_SPECS`, `_COMPATIBILITY_SPECS`, `_RESEARCH`) ·
`src/sportstradamus/training/ship_config.py` (`_validate_cell`) ·
`src/sportstradamus/helpers/distributions.py` (`fused_loc`, `get_odds`, `get_ev`, `_skewnormal_odds`,
`_gamma_odds`, `decode_predictive_mean`, `set_model_start_values`, `GATE_PUBLISH_THRESHOLD`,
`NONZERO_DENOM_GATE`) ·
`src/sportstradamus/prediction/model_prob.py` (`_blend_with_book`, `_dispersion_calibrate`,
`_serving_dispersion_cal`, `_model_predictive_sd`, `_zi_kwargs`, `_serve_offset_mode`,
`_decode_model_params`, `_annotate_display_shape`) ·
`.venv/…/lightgbmlss/distributions/StudentT.py` + `utils.py` (`exp_fn_df`, `softplus_fn_df`) 0.6.1 ·
`research/confirm_nominee_gates.csv` (rows 2026-08-23 → 2026-08-26T10:49) ·
`src/sportstradamus/data/research/strategy_research_board.csv` (swept 2026-08-26T09:41) ·
`src/sportstradamus/data/training/model_stats.parquet` ·
`src/sportstradamus/data/config/{stat_meta,stat_calibration}.json` ·
`src/sportstradamus/data/training_data/NFL_{receiving,rushing,passing}-yards.parquet` ·
memories `[[cdf_recal_nonstationary_pit]]`, `[[serve_decode_drift_offset_mode]]`,
`[[deterministic_ab_g4_oversell]]`, `[[crossfit_board_ships_optimistic]]`,
`[[board_confirm_gap_root_cause]]`, `[[ratio_projvol_refuted]]`,
`[[gate_off_probe_confounds_mean]]`, `[[mean_corrector_belongs_post_fusion]]`,
`[[nfl_volume_cells_feature_mature]]`, `[[corner_verdicts_are_not_code_scoped]]`.
