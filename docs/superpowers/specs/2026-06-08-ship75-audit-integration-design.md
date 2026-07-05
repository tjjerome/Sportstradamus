# Design: Fold the full-distribution audit into Operation Ship 75

**Status:** Approved skeleton; this spec is the detailed edit plan for review before touching the canonical docs.
**Scope:** Documentation only — restructure and de-drift `docs/operation_ship_75.md`, extend
`docs/sportstradamus_roadmap_v2.md`, add citations to `docs/operation_ship_references.md`. No code
changes. Every code-bearing item lands as a *flagged lever*, research-gated where it changes a
distribution family or dispersion mechanism (CLAUDE.md research-first).

---

## 1. Context and goal

The user commissioned an independent audit ("Unused Levers for Full-Distribution Accuracy in a DFS
Player-Prop Model"). It is dense, well-cited, and largely additive to a sound plan. The task is to
fold its genuinely-new, model-breadth paths into Operation Ship 75, route the non-marginal-breadth
work to the v2 roadmap (deferred to just after Ship 75), and de-drift the now-stale parts of the
Ship 75 doc in the same pass.

Before folding, the audit's claims about current code state were verified against the source. Two
headline claims were found **stale** and one **partially stale**; the corrections below are
load-bearing for the edits.

### 1.1 Verification findings (audit claim vs actual code)

| Audit claim | Actual code | Verdict |
|---|---|---|
| §5.8 "`fused_loc` fuses only location" | `fused_loc` ([distributions.py:493-598](../../../src/sportstradamus/helpers/distributions.py)) blends summary *parameters*: NegBin log-blends μ and r; SkewNormal precision-blends loc+σ and linear-shrinks skew; Gamma precision-blends | **Imprecise**, but the substance (immature pooling) is correct — see §1.2 |
| §5.8 "use log pool not the linear pool" | Already a log/precision pool, not a linear mixture | **Stale as stated** — but the deeper immaturities are real (§1.2) |
| §2.1 "scalar (c,s) is the only post-hoc calibration; add isotonic" | Isotonic exists on the single-line over-prob (`prob_recal_isotonic`) and on the mean (`isotonic_mean`) in [posthoc.py](../../../src/sportstradamus/training/posthoc.py); **not** on the full PIT/CDF | **Partial** — isotonic-PIT/IDR over the whole alt-line ladder is genuinely new; frame as an extension |
| §3 root-cause "SkewNormal gets no dispersion calibration" (Ship 75 §3, present tense) | Since fixed: [pipeline.py:1534](../../../src/sportstradamus/training/pipeline.py) fits joint `(c_opt, skew_cal)` for SkewNormal; [model_prob.py:624](../../../src/sportstradamus/prediction/model_prob.py) applies it | Ship 75 §3 is **stale** (describes a fixed bug); de-drift in place |
| §5.8 de-vig power method | `no_vig_odds` ([distributions.py:81-98](../../../src/sportstradamus/helpers/distributions.py)) uses the proportional method `o/(o+u)`; one-sided lines fabricate a flat 6.5% under | **Real delta** — power method matters on asymmetric/longshot lines |
| §5.1 `ratio_projvol` normalization | Unbuilt (Ship 75 agrees) | **Real, both agree** |

### 1.2 The five `fused_loc` immaturities (load-bearing for §5.3 of the doc)

The user specifically flagged the pooling as immature and correct to. A close read confirms five
concrete weaknesses:

1. **Parameter blend ≠ density pool.** The docstring calls the NegBin path a "logarithmic opinion
   pool (Genest & Zidek 1986)," but it log-blends `(μ, r)` ([distributions.py:558](../../../src/sportstradamus/helpers/distributions.py)).
   A true LOP multiplies the PMFs/PDFs pointwise and renormalizes. Parameter-blending coincides with
   the LOP only in the Gaussian case — so SkewNormal's loc/σ precision-blend *is* a correct weighted
   Gaussian LOP, but **NegBin/ZINB is an approximation wearing the LOP label**.
2. **Crude book distribution.** Book side = symmetric `N(ev_b, (ev_b·cv)²)`, skew forced to 0
   ([distributions.py:569,574](../../../src/sportstradamus/helpers/distributions.py)), with `cv` a single
   per-cell constant from `stat_calibration.json` ([model_prob.py:186](../../../src/sportstradamus/prediction/model_prob.py)) —
   not line-specific. The actual two-way de-vigged price never shapes the book's spread or skew.
3. **Skew is a linear shrink.** `blended_skew = w·model_skew` ([distributions.py:581](../../../src/sportstradamus/helpers/distributions.py)) —
   leaning on the book decays skew toward symmetric; no third-moment pooling.
4. **The pool can only sharpen, never widen.** Precision/log pooling is ≤ the least-dispersed
   component; widening the under-dispersed model is bolted on afterward as the separate scalar
   `c_opt` (`dispersion_cal`). No flexibly-dispersive capacity in the pool itself.
5. **No noise model on the book price.** `w` (model_weight) is a learned per-cell constant; there is
   no per-observation book precision derived from how noisy `p_book` is in that market.

Conclusion: the blend axis is a confirmed **primary** Ship-75 lever (it attacks under-dispersion
directly), not a maybe. It is elevated in the restructure.

### 1.3 User routing decisions (collected)

- **Integration depth:** restructure §5 (around the audit's framing), not surgical adds.
- **Axis count:** **four** axes — `normalization → model/loss → blend → calibration` — each
  independently swappable per cell. (User correction; the original "three axes" bundled model/loss
  into blend.)
- **Deferral scope:** defer non-marginal-breadth only. Roadmap gets copula/dependence, conformal
  alt-line ladder, CLV dashboard, multi-task-NN. Ship 75 keeps everything that moves a cell past the
  five gates.
- **Gate methodology:** diagnostics only — add advisory report-only diagnostics; **do not** fold in
  BH-FDR / Deflated-Sharpe / a CLV ship-gate (left out of Ship 75 entirely).
- **De-drift:** fix stale statements in place.

---

## 2. The four-axis reframe

A served predictive is built in four independently-swappable per-cell stages. Restructured §5 leads
with this and maps every lever onto a stage:

| Axis | What it controls | Values | Tier today |
|---|---|---|---|
| **Normalization** | target transform the model fits | `ratio_meanyr`, `centered_additive_mean10`, `centered_additive_eb_meanyr_k10`, `ratio_projvol` (unbuilt) | retrain; searched |
| **Model/loss** | how the model trains its own predictive | `dist_training_loss` (nll/crps); training-time variance/MMD-to-uniform-PIT regularizer (unbuilt) | retrain; loss searched, regularizer unbuilt |
| **Blend** | how the model predictive meets the book | `blending_loss_fn` (nll/crps for w-fit); `fused_loc` pooling; book-distribution construction (de-vig + recovery); BLP recalibration wrapper (unbuilt); p_book noise model (unbuilt); w_book schedule (unbuilt) | retrain (weight) + research-gated (structure) |
| **Calibration** | post-hoc on the served predictive | location (`roe_mean`/`isotonic_mean`), scale+shape (scalar `c`/`skew_cal`), full-CDF (isotonic-PIT / IDR — unbuilt), `prob_recal_*` | auto-fit / free |

The two modeling **heads** have opposite calibration problems (Czado-Gneiting-Held 2009): the
continuous SkewNormal head is **under-dispersed** (PIT U-shaped, needs widening); the count
ZINB/NegBin/ZAGamma suite **over-covers** (PIT inverted-U, needs narrowing). "Widen everything" and
"narrow everything" are both wrong. The NegBin family's conditional variance is bounded below by its
mean, so the deepest count cells need a both-directions family (a structural ceiling post-hoc cannot
clear).

Three research-gated tracks feed the model axis and are documented as their own subsections, not as a
fifth axis: **family/distribution** (the model's head), **small-sample/hierarchical**, and **features**.

---

## 3. Restructured `operation_ship_75.md` §5

Preserve all working machinery (research board, operating loop, supersede, no-defer policy). Move the
old lever cascade onto the four axes. New section layout:

```
§5  The lever stack — four axes × two heads
    intro: served predictive = normalization → model/loss → blend → calibration (4 swappable
           per-cell stages); continuous head widens, count head narrows; NegBin var ≥ mean ⇒
           deepest count cells need a both-directions family.

  §5.0  Strategy research board + operating loop          [KEEP — update 3→4 axes]
  §5.1  Lever 0 — re-score, promote free passers          [KEEP]
  §5.2  Calibration axis — the post-hoc ladder (free→cheap, both heads, opposite directions)
  §5.3  Blend axis — the under-built lever (ELEVATED)     [RESEARCH-GATED structure]
  §5.4  Normalization axis                                [KEEP + build ratio_projvol]
  §5.5  Model/loss axis                                   [NEW — split out of blend]
  §5.6  Family / distribution axis                        [RESEARCH-GATED]
  §5.7  Small-sample / hierarchical layer
  §5.8  Features — leakage-safe player-level              [KEEP old Lever 3]
  §5.9  Lever cap / matrix-exhaustion policy              [KEEP — update to new axes]
```

### 3.1 §5.0 board + operating loop (keep, minor edit)

Unchanged except: every "three axes" → "four axes"; update the orthogonality statement to
`target_normalization ⊥ {dist_training_loss} ⊥ {blending_loss_fn, fused_loc/BLP, book-recovery} ⊥
{posthoc, dispersion_cal, skew_cal}`; update the executable/unbuilt axis table to four rows (§2 of
this spec). The GridSampler still searches the retrain corners (normalization × dist_training_loss ×
blending_loss_fn); calibration auto-fit; blend *structure* and the variance regularizer are
research-gated builds, not corners yet.

### 3.2 §5.2 Calibration axis — the post-hoc ladder

Consolidates old Lever 1 (scalar dispersion/skew) and old Lever 2 (mean correction) into one ordered
ladder, applied to the served predictive, both heads in opposite directions:

- **Rung A — location** (shipped): `roe_mean` (affine) / `isotonic_mean`, `MEAN_STAGE` in posthoc.py.
  Bias cells (g2/g3). Affine ROE only at low count means (isotonic tails overfit). **Operator note —
  skeptical:** post-hoc correction of the *predicted mean* is held to a higher bar than width/shape
  recalibration — it edits the central tendency the model is supposed to learn, so a cell that ships
  only because its mean was patched is suspect. Prefer fixing location at the source (normalization
  §5.4, features §5.8) first; carry mean correction as available-but-last-resort, and ship a
  mean-corrected cell only if it also holds the g1 BSS guardrail and survives the val→test discount.
- **Rung B — scale + shape** (shipped): scalar joint `(c, skew_cal)`, fit jointly against PIT-KS
  inside `_step_calibrate_dispersion`. Widen SkewNormal / narrow count. (The Hallin & Ley α=0
  singularity rationale and the joint-vs-sequential dominance move here, retained from current §5.)
- **Rung C — full CDF** (new, free): isotonic-PIT recalibration (Kuleshov 2018) and IDR-as-recalibrator
  (Henzi, Ziegel & Gneiting 2021) — a monotone map on the *whole* predictive CDF, recalibrating the
  entire alt-line ladder, not just the single-line over-prob the existing `prob_recal_isotonic` fixes.
  Prefer isotonic/IDR over conformal calibration on the count lattice (conformal yields discontinuous
  randomized CDFs — bad for pricing a ladder; Marx 2022).
- **Existing `prob_recal_*`** retained as-is (single-line probability recalibration).
- **Note:** calibration is close to exhausted as a *breadth* lever on its own — it fixes
  width/shape/location, never signal or location-scale structure.

Go/No-Go and if-it-fails branches carry over from the current Lever 1/Lever 2 blocks (BSS guardrail
≤0.01, g5 < 0.075, inference-path round-trip test).

### 3.3 §5.3 Blend axis — the under-built lever (elevated)

The largest new section. Built from the audit §5.8 plus the verified immaturities (§1.2). **Structure
changes here are research-gated** (they alter the dispersion mechanism).

- **Current state + the five immaturities** (§1.2 verbatim, with file:line refs). Corrects the doc's
  prior "blend structure is fixed / precision-weighted pool" line.
- **Fitting half — one de-vigged point → a distribution:**
  - Replace proportional de-vig with the **power (logarithmic) method** in `no_vig_odds` (preserves
    [0,1], handles favorite-longshot bias on asymmetric/anytime-TD lines; Clarke, Kovalchik & Ingram
    2017). Flag cells with `|p_over − 0.5| > 0.3` for extra downstream shrinkage.
  - Recover the book *distribution* by fixing the model's shape `(σ̂, α̂)` / `(θ̂, π̂)` and solving the
    single location parameter so the model-shaped CDF passes through the de-vigged point (1-D
    root-find). The line is a **median**, not a mean — which is why model skew matters in the inversion.
  - Count tail case (anytime-TD): `λ = −log(1−p)` / NegBin `μ = θ((1−p)^(−1/θ)−1)` are
    ill-conditioned (`dλ/dp = 1/(1−p)` blows up as p→1); regularize toward the model's μ̂ and cap
    `|μ_book − μ̂| ≤ K·SD`. (Practitioner-grade, flagged as such.)
- **Pooling half:**
  - Keep the log pool as the base operator; **fix the NegBin/ZINB path to a real density LOP**
    (grid-multiply the PMFs and renormalize) instead of the parameter log-blend.
  - Add the **beta-transformed linear pool (BLP)** recalibration wrapper (Ranjan & Gneiting 2010):
    `F^BLP = B_{α,β}(w·F_model + (1−w)·F_book)`, flexibly dispersive (can narrow *or* widen), fit
    **outside** the five gate calculations (it changes the gate inputs, not the gates). This is the
    principled under-dispersion fix that supersedes the bolt-on scalar `c_opt` for the blend.
  - **Conjugate noise model on `p_book`:** treat `logit(p_book)` as a Gaussian observation on the
    model's logit-CDF at the line, per-cell variance from residual studies → precision weight; "book
    is noisy in this market" falls out as lower precision.
  - **Learn `w`** by minimizing CRPS (continuous) / log-score (count), shrunk per-cell toward a global
    prior ∝ 1/n_cell; do not hard-code book = truth (props are soft).
  - **Time-varying `w_book`** ramp toward close (CLV framing — the de-vigged *close*, not the open).
    (The CLV-edge *dashboard* defers to roadmap; the weight schedule stays here.)
- **Caveat folded in** (audit §7.10): do not "fix" under-dispersion with a raw linear pool — its
  widening is disagreement-driven, not legitimate uncertainty, and degrades sharpness + the KS/ECE
  gates. Use log pool + BLP.
- **Go/No-Go:** PIT-KS crosses below threshold with g1 BSS drop ≤0.01 and g5 < 0.075, on validation,
  before any ship; inference-path round-trip test for the new served objects (de-vig method, recovered
  book params, BLP coefficients).

### 3.4 §5.4 Normalization axis (keep + endorse build)

Carry the current normalization content. Endorse and schedule `ratio_projvol`: target = `y /
projected-volume` (a per-minute / per-carry / per-target rate), decode = `rate × projected-volume`;
on the count side use `log(volume)` as a **GLM offset**, delta-method or Monte-Carlo back to totals
scale for the gate. Volume projections already exist as `proj_*` features. Names it the most likely
structural unlock for the NFL volume cells (efficiency × opportunity separation).

### 3.5 §5.5 Model/loss axis (new — split out of blend)

- `dist_training_loss` nll vs crps per cell (Gebetsberger 2018: min-CRPS more robust to
  misspecification, ML slightly more efficient under correct spec; keep the per-cell winner). Note the
  honest caveat that LightGBMLSS's CRPS path sets the Hessian to 1 (first-order), which is *why* a
  properly-curved CRPS head (NGBoost natural gradient) is the narrow-use Hail-Mary in §5.6/roadmap.
- **Training-time variance / soft-calibration regularizer** (the lever Ship 75 §8 already flags as
  "untried, not refuted") — concrete form: an MMD-to-uniform-PIT penalty (Chung 2021) or a held-out
  variance penalty to widen σ where the model is overconfident. Resolves the §8 wording from "floated,
  never run" to a defined, unbuilt model/loss lever.
- CRPS-stacking of the model's own CDF variants across loss × transform (Gneiting & Ranjan 2013), via a
  log/beta-transformed pool (same dispersion logic as §5.3, not a raw linear pool).

### 3.6 §5.6 Family / distribution axis (research-gated)

Reframe the old Lever 4b/Lever 5 by the per-head ceiling logic. Every item here needs a
`research-analyst` brief before build (family/dispersion-mechanism change).

- **Continuous, by escalating expressiveness:** centered-parametrization SkewNormal / skew-t
  (Arellano-Valle & Azzalini 2008) — a *loss-function* change that removes the α=0 Fisher-information
  singularity at the source (distinct from, and complementary to, the post-hoc additive `skew_cal`
  patch); **try first** among family moves. Then SHASH / Johnson SU (Jones & Pewsey 2009) — 4-param,
  separately governing skew and kurtosis — for heavy-kurtosis cells (NBA AST). Then skew-t / Student-t
  for the heaviest tails. **Explicit caveat:** centered parametrization fixes the *singularity*, not
  the *tails*; a cell needing both routes to SHASH/JSU/skew-t.
- **Count structural ceiling:** CMP (Sellers & Shmueli 2010), Generalized Poisson (Harris 2012),
  Double Poisson (Efron 1986) — span over- *and* under-dispersion. **Correct the §7a mis-shelving:**
  CMPμ is currently filed under §7a as a *mean-compression* family; it is also (and primarily here) the
  *dispersion-direction* fix for over-covering count cells. Note the CMP infinite normalizing constant
  must be truncated and round-trip-tested on the live path (audit §7.5).
- **Per-cell ZI-vs-hurdle-vs-plain-NB-vs-CMP screen** on the honest val→test PIT — stop defaulting ZINB
  on cells that are not genuinely zero-inflated (inflates variance → over-coverage). Hurdle already
  exists via `zinb_mode`; this is a one-field edit + retrain.
- **Tweedie / generalized-gamma** heads for zero-mass continuous cells (NFL RB2 rush yards).
- Existing per-cell pivots fold here: `zinb_mode: hurdle`, monotone priors (`monotone_priors.json`),
  per-position split (T11).

### 3.7 §5.7 Small-sample / hierarchical layer (the NFL wall)

- **EB-shrink the distributional parameters** (μ, σ, ν, τ) per player toward a per-position mean, CV'd
  shrinkage strength — cheapest, stays in LightGBMLSS. (Refines the existing player-level-features
  lever with a parameter-shrinkage variant.)
- **TabPFN v2** head-to-head on the small-n NFL/WNBA cells (Hollmann 2025) — native full predictive
  distribution, no per-cell tuning, ≤~10k-row sweet spot; a *per-cell tool for small cells*, not the
  data-rich NBA. Recalibrate its output through the existing PIT gate. Try **before** the full
  hierarchical build (lower friction, same Bayesian small-n logic).
- **Hierarchical-Bayes** layer (player ⊂ position ⊂ team) — research-gated escalation if EB shrinkage
  and TabPFN are insufficient. (The multi-task NN sharing a trunk across cells/leagues defers to
  roadmap.)

### 3.8 §5.8 Features (keep old Lever 3)

Unchanged: leakage-safe `MeanYr_expanding_shifted` + opponent-defense interaction + blowout flag; SHAP
< 0.001 ⇒ revert. The edge-not-width cells.

### 3.9 §5.9 Lever cap / matrix-exhaustion policy (keep, update)

Keep the no-defer / matrix-exhaustion policy. Update the matrix to the four axes + the research-gated
tracks: a cell leaves the Ship-75 board only after failing across normalization × model/loss × blend ×
calibration **plus** the family and hierarchical tracks. Zero cells qualify today.

---

## 4. De-drift edits to `operation_ship_75.md` (in place)

1. **§3 root-cause** — rewrite the present-tense "SkewNormal receives no dispersion calibration ...
   early-returns c_opt = 1.0" passage to past tense: the scalar joint `(c, skew_cal)` fix is built and
   shipped; the *residual* under-dispersion that the scalar cannot reach is what the calibration ladder
   (Rung C), the blend axis (BLP), and the normalization axis now target. Removes the §3 ↔ §1d/§5
   contradiction.
2. **§5 search-space blend bullet** — replace "the blend *structure* is fixed" and the
   "precision-weighted pool + per-cell model_weight" description with the corrected state: a
   full-but-immature parameter-space pool (the five immaturities), now an active lever per §5.3.
3. **§1c** — append advisory **report-only** diagnostics to the existing companion list
   (`central50/80_coverage`, `g4_tail_pit_ks`): Anderson-Darling PIT (tail-weighted), conditional /
   stratified PIT-KS (by mean-decile / blowout / B2B / position / home-away), non-randomized PIT for
   the count head (lower-variance lattice diagnostic; Czado-Gneiting-Held), CRPS reliability
   decomposition (Arnold et al. 2024). All advisory — the five gates are unchanged. **No** BH-FDR,
   Deflated-Sharpe, or CLV ship-gate (per the diagnostics-only decision).
4. **§8 holes** — update hole #6 (block-bootstrap / clustered-g1 backlog) to name **CPCV + player/date
   embargo** (López de Prado 2018) as the concrete method once the `game_date` join is fixed; keep it a
   validation refinement, not a gate change. Fold the audit's variance-regularizer into the existing
   "untried, not refuted" note as a now-defined model/loss lever (cross-ref §5.5). **Collapse the dated
   `Resolved …` / `Updated …` blockquote accretion** in §8 into current-state prose — these dated
   build-log blocks are the exact STYLE_GUIDE §16 smell the `docs-style.py` hook guards; the de-drift
   converts them to "current status" prose (which hole is open vs resolved), with the dated history left
   to git.
5. **§7a** — add the one-line correction that CMPμ is also the count dispersion-direction fix
   (cross-ref §5.6), so the family is no longer mis-shelved as mean-compression-only.
6. **§10 reading list** — cross-ref the new roadmap-deferred items and `operation_ship_references.md`
   for the new citations.

---

## 5. Additions to `sportstradamus_roadmap_v2.md` (the deferred tail)

Deferred to just after Ship 75 reaches 75% breadth. Pointers, not restatements.

- **Phase 1.2 follow-ups** — add: **copula layer over the calibrated marginals** for parlay pricing
  (PIT-transform residuals → fit Gaussian/t-copula within same-game leg groups → sample and invert →
  EB-shrink per-pair correlations), plus a **dependence diagnostic** (average pairwise rank correlation
  of residual PITs within same-game groups vs the under-independence prediction). This refines the
  existing shipped Gaussian copula; it is the "parlay correlation testing" the user flagged for
  deferral. Note the asymmetry that makes it worth building: sportsbooks tax SGP correlation, the DFS
  pick'em apps largely do not.
- **Phase 6 (Modeling Refinements)** — add to the deferred-tail list: distributional conformal
  prediction + CQR for the alt-line ladder (Chernozhukov 2021; Romano 2019); the CLV CRPS-edge
  dashboard (model vs de-vigged *closing* distribution, per market, weekly); TabPFN-as-platform and the
  multi-task NN shared-trunk backbone (cross-cell/cross-league pooling); spliced/Pareto-tail and
  normalizing-flow heads for residual heavy tails. Cross-ref Ship 75 §5.6/§5.7 (where the *per-cell,
  small-n* uses of TabPFN and conformal stay) and `operation_ship_90.md`.
- **Decisions & Trade-offs table** — one row: "Full-distribution audit — marginal-breadth levers
  folded into Ship 75 §5; parlay-dependence + conformal-ladder + backbone swings deferred here."

---

## 6. Additions to `operation_ship_references.md`

Append the audit's new citations (the ones not already present) to the existing `[1]`–`[48]` list,
keeping the numbering scheme: Arellano-Valle & Azzalini 2008/2013; Jones & Pewsey 2009; Sellers &
Shmueli 2010; Harris 2012; Efron 1986; Ranjan & Gneiting 2010; Genest & Zidek 1986; Gneiting & Ranjan
2013; Czado-Gneiting-Held 2009; Kuleshov 2018; Henzi-Ziegel-Gneiting 2021; Chernozhukov 2021; Romano
2019; Vovk 2018; Gebetsberger 2018; Chung 2021; Hollmann 2025 (TabPFN); Clarke-Kovalchik-Ingram 2017;
Arnold et al. 2024; López de Prado 2018; Grinsztajn 2022 / McElfresh 2023. Citations live here, not
inline in the plan (per the existing doc convention).

---

## 7. Full routing table (every audit item → destination)

| Audit item | Destination | Tier |
|---|---|---|
| §2.1 isotonic-PIT recalibration | Ship 75 §5.2 Rung C | free |
| §2.1 IDR as recalibrator | Ship 75 §5.2 Rung C | free |
| §2.1 conformal predictive distributions / CQR / DCP | roadmap Phase 6 | deferred |
| §2.2 centered-parametrization SkewNormal | Ship 75 §5.6 | research-gated |
| §2.2 SHASH / Johnson SU | Ship 75 §5.6 | research-gated |
| §2.2 skew-t / Student-t | Ship 75 §5.6 | research-gated |
| §2.2 NGBoost (narrow) | Ship 75 §5.5 note + roadmap Hail-Mary | deferred |
| §2.3 CRPS-vs-NLL per cell | Ship 75 §5.5 | retrain |
| §2.3 variance / MMD-to-uniform-PIT regularizer | Ship 75 §5.5 + de-drift §8 | unbuilt |
| §3.1 CMP / GenPoisson / Double-Poisson | Ship 75 §5.6 + §7a correction | research-gated |
| §3.2 isotonic/IDR for counts | Ship 75 §5.2 Rung C | free |
| §3.3 ZI-vs-hurdle-vs-NB-vs-CMP screen | Ship 75 §5.6 | retrain |
| §3.4 non-randomized PIT diagnostic | Ship 75 §1c advisory | free |
| §4 copula over marginals + dependence diagnostic | roadmap Phase 1.2 follow-ups | deferred |
| §5.1 ratio_projvol + GLM offset | Ship 75 §5.4 (build) | retrain |
| §5.2 EB-shrink distributional params | Ship 75 §5.7 | retrain |
| §5.2 full hierarchical Bayes | Ship 75 §5.7 | research-gated |
| §5.2 multi-task neural net | roadmap Phase 6 | deferred |
| §5.3 Anderson-Darling / stratified PIT / CRPS reliability decomp | Ship 75 §1c advisory | free |
| §5.4 BH-FDR / Deflated-Sharpe | **not folded** (diagnostics-only decision) | out |
| §5.5 CPCV + embargo | Ship 75 §8 hole #6 (validation) | refine |
| §5.5 CLV gate | roadmap (advisory/monitor only; dashboard Phase 6) | deferred |
| §5.6 CRPS-stacking | Ship 75 §5.5 | retrain |
| §5.6 multicalibration | Ship 75 §1c advisory (subgroup) | free |
| §5.6 Tweedie / generalized-gamma | Ship 75 §5.6 | research-gated |
| §5.7 TabPFN v2 (small-n cells) | Ship 75 §5.7 | research-gated |
| §5.7 TabPFN as platform / multi-task NN | roadmap Phase 6 | deferred |
| §5.7 keep-GBDT prior | Ship 75 §5.7 note | — |
| §5.8 power de-vig | Ship 75 §5.3 (touches no_vig_odds) | small build |
| §5.8 book-distribution recovery | Ship 75 §5.3 | research-gated |
| §5.8 NegBin density LOP fix | Ship 75 §5.3 | research-gated |
| §5.8 BLP recalibration wrapper | Ship 75 §5.3 | research-gated |
| §5.8 conjugate p_book noise model | Ship 75 §5.3 | research-gated |
| §5.8 time-varying w_book | Ship 75 §5.3 (dashboard → roadmap) | retrain |
| §6 staged roadmap | informs Ship 75 §5 cheapest-first ordering | — |
| §7 risks/caveats | folded into the relevant §5 subsections | — |
| §8 references | operation_ship_references.md | — |

---

## 8. Out of scope (explicit)

- **No code changes.** This is documentation. Code-bearing items land as flagged/research-gated levers.
- **No gate-battery changes.** The five gates are unchanged; only advisory report-only companions are
  added. BH-FDR, Deflated-Sharpe, non-randomized-PIT-*as-gate*, and a CLV ship-gate are explicitly not
  adopted.
- **No per-cell ship verdicts.** The doc points at `stat_meta.json` / `model_stats.csv` / the sweep
  board for live counts (unchanged convention).
- **No restatement of deferred content in Ship 75.** The roadmap is the home of record for the
  deferred tail; Ship 75 cross-refs.

---

## 9. Verification (documentation gates)

This is doc work, so the gates are doc-drift, not pytest:

- All internal cross-references (`§5.x`, file:line links, roadmap anchors) resolve.
- No surviving contradiction between §3, §1d, and §5 on the SkewNormal dispersion-cal state (the
  de-drift target).
- The "four axes" language is consistent across §5.0, the axis table, and §5.9.
- One canonical home per fact: deferred items appear in the roadmap with only a pointer from Ship 75;
  citations only in `operation_ship_references.md`.
- Changelog discipline per STYLE_GUIDE §16 (caveman one-liners, newest-first, capped).
- The `docs-style.py` hook nudge is addressed, not ignored.
- Sanity: the doc still reads as a single coherent plan, not a plan with an audit stapled on.

---

## 10. Execution note

Edits sequence cleanly as: (1) Ship 75 §5 restructure, (2) Ship 75 de-drift (§3/§1c/§7a/§8/§10), (3)
roadmap-v2 additions, (4) references append. Each is an independent edit set on a different doc region;
the writing-plans step will order them and define the per-edit checks. No subagents needed (single
contributor, documentation, no module boundaries crossed).
