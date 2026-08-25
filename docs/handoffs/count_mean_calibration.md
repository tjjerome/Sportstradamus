# Lane record — count Gate-4 mean calibration: lever real, both cells walled here

Status: **closed.** The mean lever **works** — it converts Gate 4 on both cells the brief called
convertible — and neither cell shipped *in this lane*, for two different structural reasons.
**NFL tds has since shipped:** its wall was the mean corrector sitting on the wrong side of the
model↔book pool, and the successor lane
[mean_corrector_stage_order.md](mean_corrector_stage_order.md) moved the corrector post-fusion and
took the cell through all six gates. NBA PF's `posthoc`-slot wall is untouched and still open.
Context spine: CLAUDE.md,
[docs/ARCHITECTURE.md](../ARCHITECTURE.md),
[model_improvement_track.md](model_improvement_track.md) §6.1 Rung A / §6.5.
Research brief (mechanism, literature, per-cell measurements):
[researcher_count_gate4_zeroinflation.md](../archive/researcher_count_gate4_zeroinflation.md).
Predecessor lane: [count_dispersion_flip.md](count_dispersion_flip.md).

## The question

Gate 4 fails on 5 of 16 ZINB count cells and 0 of 33 NegBin/DPO. The research brief established
that the failure is a **predictive-mean** deficit, not zero-inflation: the served ZINB mean is
`(1−π)·μ`, so moving the gate silently rescales the mean, and an offline probe that moved only the
mean took all four failing ZINB cells under threshold. This lane asked whether that survives a
full-HPO confirm on the current matrices.

Seven arms, all `--frozen-matrix-dir` + `--artifact-output` sandboxes, all full 300-trial HPO.

## Verdict

**The lever is real and reproducible.** At fixed family and matrix, adding a mean-stage corrector
moves Gate 4 across its threshold on both convertible cells:

| cell | family | `posthoc` | `g4_pit_ks` | thr |
|---|---|---|---|---|
| NFL tds | NegBin | `none` | 0.0700 | 0.0500 |
| NFL tds | NegBin | `roe_mean` | **0.0463** ✓ | 0.0500 |
| NFL tds | NegBin | `isotonic_mean` | **0.0431** ✓ | 0.0500 |
| NBA PF | DPO | `none` | 0.0627 | 0.0500 |
| NBA PF | DPO | `roe_mean` | **0.0366** ✓ | 0.0500 |
| NBA PF | NegBin | `prob_recal_isotonic` | 0.0784 | 0.0500 |
| NBA PF | NegBin | `roe_mean` | 0.0596 | 0.0500 |

Two leagues, two families, both corrector shapes. That closes the mechanism question the brief
opened: **the count Gate-4 gap is a mean deficit and a mean corrector closes it.**

**Neither cell shipped in this lane, and the two walls are unrelated.** NFL tds' wall was
stage order and is now broken; NBA PF's is the single-valued `posthoc` slot and still stands.

### NFL tds was blend-bound on Gate 6 — since fixed by moving the corrector post-fusion

Full results against the honest full-HPO incumbent (ledger `seed/incumbent`, `elapsed_s` 3705 —
*not* the 80 s cross-fit board rows, whose g6 pass is optimistic):

| | ZINB incumbent | NegBin `none` | NegBin `roe_mean` | **NegBin `isotonic_mean`** |
|---|---|---|---|---|
| `g4_pit_ks` (≤0.0500) | 0.0592 ✗ | 0.0700 ✗ | 0.0463 ✓ | **0.0431 ✓** |
| `g4_tail_pit_ks` | 0.0569 | 0.0691 | 0.0437 | **0.0384** |
| `g1_brier_diff_ci_hi` | −0.0227 | −0.0216 | −0.0209 | **−0.0234** |
| `brier_skill_score` | 0.1779 | — | 0.1494 | 0.1654 |
| `g5_ece_debiased` | 0.0065 | 0.0336 | 0.0351 | 0.0179 |
| `g6_citl_ci_hi` (≥0.97) | 0.8647 ✗ | 0.8049 ✗ | 0.8527 ✗ | **0.8983 ✗** |
| `roc_auc` | 0.7336 | — | 0.7145 | 0.7227 |
| model CITL / served CITL | 0.903 / 0.737 | 0.718 / 0.628 | 0.929 / 0.768 | 0.917 / 0.784 |

Gate 6's CITL leg needs `Σ Blended_EV/ΣResult` upper-bounded at ≥ 0.97. The corrector gets the
**model** mean to 0.92–0.93, but fusion carries only 0.83–0.86 of it through, so a corrector placed
*before* the pool lands served ≈ 0.86 however well it fits. This lane read that residual as the
blend structure; it is the corrector's **stage**, and the successor lane reaches
`g6_citl_ci_hi` 1.0956 on this same corner by correcting the pooled mean instead of the model leg.
§6.5 stays closed — the pool operator is not what changed.

`isotonic_mean` beats `roe_mean` on g4, g4_tail, g1, BSS, g5, g6 and `roc_auc` simultaneously,
exactly as §6 of the research brief predicted from NFL tds' range-dependent correction shape. The
affine fit is intercept-dominated (`a` 0.0613, `b` 0.9386 on a mean of 0.17), so it lifts the low
quintiles that were already over 1.0: served-mean quintile CITL spread **0.995** under `roe_mean`
versus **0.547** under `isotonic_mean`.

### NBA PF is posthoc-slot-bound on Gate 1

`posthoc` is single-valued and mutually exclusive by design. PF needs it for two jobs at once:

| `posthoc` (NegBin unless noted) | `g4_pit_ks` (≤0.0500) | `g1_brier_diff_ci_hi` (<0) | BSS | `g5_ece_debiased` |
|---|---|---|---|---|
| `prob_recal_isotonic` — ledger recipe | 0.0784 ✗ | **−0.1303 ✓** | **+0.345** | 0.0055 ✓ |
| `roe_mean` | 0.0596 ✗ | +0.0532 ✗ | −0.141 | 0.0986 ✗ |
| `roe_mean`, DPO | **0.0366 ✓** | +0.1209 ✗ | −0.321 | 0.0877 ✗ |
| `none`, DPO | 0.0627 ✗ | +0.1123 ✗ | −0.298 | 0.1079 ✗ |

The `prob_recal_isotonic` arm reproduces the ledger's NegBin rows (`g1_ci_hi` −0.108…−0.188,
BSS +0.294…+0.632, `g5` 0.007–0.013, `n_validation` 2123) — so the harness is sound and **PF's
entire Gate-1 cushion comes from the prob-stage recalibrator, worth a 0.49 BSS swing.** Selecting
any mean-stage slug removes it. One slot, two jobs, no corner that satisfies both.

`cdf_recal_isotonic` is the only slug that claims both jobs, and it is **inert on count families
by construction**: `pipeline.py` gates the whole-CDF stage on
`posthoc_slug in CDF_STAGE and dist == "SkewNormal"`, and the registry refuses the corner because
it would train identically to `none` under a different fingerprint. There is no zero-code escape.

### NFL interceptions is refuted, exactly as pre-registered

`NegBin` + `roe_mean` on `74dbd8a0`: `g4_pit_ks` 0.0839 (thr 0.0689) ✗, `g6_citl_ci_hi` 0.8928 ✗,
and the decisive number — **`roc_auc` 0.4993.** The fitted affine is `a` 0.5398, `b` **0.1962**:
the corrector collapsed the model onto the marginal mean and erased its ranking. The research
brief's §6 warning (calibration slope −0.166 ⇒ a mean corrector flattens toward the marginal ⇒
read `roc_auc`, not just g4) was precise. Do not retry a mean corrector on this cell.

## Corrections to the brief's framing

- **`--target-normalization none` is not a valid CLI value.** The E1 command in the brief fails at
  argument parsing. Drop the flag: `--bypass-withholding` forces `TARGET_NORM_NONE` on every
  non-SkewNormal cell and the count branch ignores the slug regardless.
- **Arms cost ~25 min, not 35** — 300 Optuna trials, trial-count-bound well inside the 3600 s cap.
- **The fusion haircut is recipe-dependent, not a cell property.** It moved 0.874 → 0.826 → 0.855
  across three NFL tds arms and 0.981 → 0.894 on interceptions. Two predictions in this lane were
  built on a shipped dump's haircut and both were wrong in the same direction. Measure it per arm;
  never carry it across recipes. NBA PF is the exception where it held (0.996 → 1.000).
- **Blend dilution is not a low-mean or per-league property.** Across all 48 count cells with a
  shipped dump the median haircut is **1.0000** and Spearman(`mean(y)`, haircut) is **−0.024**.
  NFL tds (0.816) is an outlier, not a class. The other severe cells — MLB home runs 0.635,
  NHL goals 0.700, NBA BLK 0.810 — are all cells whose *model* CITL is 1.28–1.63, where the blend
  is correcting an over-prediction and the haircut is the blend working.
- **The brief's "do not spend E2" dilution row does not generalize.** Its rationale is blend
  shrinkage; PF's blend demonstrably does not shrink. E2 was run on that basis and produced the
  lane's second confirmation of the mechanism.
- **The 16-corner DPO kill on PF is not settled here.** At fixed `posthoc: roe_mean`, DPO costs
  0.18 BSS against NegBin (−0.321 vs −0.141) — a clean single-axis read that DPO is the worse body
  for this cell. Whether DPO paired with a prob-stage corrector recovers the Gate-1 cushion is
  untested; every DPO arm here ran without one.

## Residue

- **NFL tds' best known corner is now its `stat_meta` pin.** `dist: NegBin`,
  `posthoc: isotonic_mean`, `dist_training_loss: nll`, `blending_loss_fn: nll`,
  `count_dispersion_objective: crps` on matrix `7918c1b8` beat the shipped `ZINB` / `none` pin on
  five of six gates here and failed only g6. The successor lane re-ran exactly this corner with the
  corrector post-fusion, cleared g6, and persisted it as `shipped: devel`.
- **Two live cells over-predict with no gate to catch them.** NFL receiving tds serves CITL 1.639
  and WNBA FTM 1.156, both `shipped: devel`. Gate 6's CITL leg is one-sided (under-prediction
  only). Flagged in the research brief's §11 and still open.
- **`MLB home runs` serves CITL 1.032 off a model CITL of 1.626** — the blend is doing all the
  calibration work on a `devel` cell. Worth an owner look; a blend-structure change would expose it.
- The mean lever remains a per-cell swept option and is correctly suppressed on gated families
  (the MEAN_STAGE mis-contract in §4 of the research brief is unchanged and was not on the critical
  path). No code changed in this lane.
