# Model Overconfidence Investigation — FGA-under / FG3M-over

**Date:** 2026-05-17 → 2026-05-19
**Status:** Investigation paused; all exploratory code reverted to baseline
commit `64a26b9`. No production code changed.
**Scope:** NBA props model publishing systematically overconfident bets —
FGA always toward the Under, FG3M strongly toward the Over.

This is an audit-style record of everything tried, the evidence for each
conclusion, the dead ends (and *why* they were dead ends), and a concrete
resume plan. It is written so the work does not have to be re-derived.

---

## 1. Symptom (ground truth)

Source: the real published output the production server generated,
`src/sportstradamus/data/current_offers.parquet` (278 rows, pulled from the
remote server during the investigation so we analysed what production
actually publishes, not a training holdout).

| Market | Dist | Live offers | Bet side | mean(Model EV) | mean(Line) | mean(Model P) | Notes |
|---|---|---|---|---|---|---|---|
| **FGA** | SkewNormal | 3 | **3 Under, 0 Over** | 12.6 | 18.2 | 0.84 (67% at 0.9 cap) | `Model Skew`=NaN every row |
| **FG3M** | ZINB | 44 | **42 Over, 2 Under** | 2.74 | 1.25 | 0.68 | recent 5-game form ≈ 1.2 |

Other SkewNormal markets (PTS/REB/AST/PR/PRA/PA/RA) in the same slate were
**balanced** (Model P ≈ 0.59, none at the cap, mean(EV−Line) −0.78…+0.77).
Other ZINB markets (STL/BLK/BLST/TOV) were over-skewed like FG3M but with
small n. So FGA (SkewNormal) and FG3M (ZINB) are the visibly pathological
markets, with distinct mechanisms.

Key qualitative facts established early and never contradicted:
- FGA elite, high-volume starters (SGA line 21.5, Wembanyama 18.5) get
  `Model EV` ≈ 12–13 — far below the line *and* below their own recent form.
- `Model Skew` (the SkewNormal `alpha`/skew parameter) is **NaN** for every
  live FGA row, although `Model EV`/`Model Sigma` are populated.
- FG3M `Model EV` ≈ 2.7 is roughly **double** the line and the players'
  recent form.

---

## 2. Confirmed root cause — ZINB gate under-fit (FG3M, path-wide)

**This is the one finding that reproduced cleanly every time and is ready to
fix.**

### Mechanism

`ZINB` (zero-inflated negative binomial) models `P(0) = π + (1−π)·NB(0)`,
where `π` is the structural-zero "gate". In the production pipeline the gate
is one of three jointly-fit LightGBMLSS parameters (`total_count`, `probs`,
`gate`) optimised under NLL. Offline replay of every NBA ZINB model on its
saved holdout showed the learned per-offer gate converges to roughly **half**
the true structural-zero rate:

| Market | actual zero rate | `hist_gate` (correct) | **learned per-offer gate (mean)** | P(over@line) pred vs actual |
|---|---|---|---|---|
| FG3M | 0.332 | 0.337 | **0.188** | 0.79 vs 0.44 |
| PF   | 0.139 | 0.164 | **0.024** | 0.97 vs 0.44 |
| TOV  | 0.317 | 0.337 | **0.126** | 0.86 vs 0.44 |
| STL  | 0.484 | 0.485 | **0.234** | 0.76 vs 0.42 |
| OREB | 0.454 | 0.447 | **0.184** | 0.80 vs 0.36 |
| BLST | 0.317 | 0.310 | **0.164** | 0.83 vs 0.42 |
| BLK  | 0.637 | 0.628 | **0.319** | 0.67 vs 0.34 |
| FTM  | 0.442 | 0.450 | **0.165** | 0.50 vs 0.38 |

`hist_gate` (the unconditional historical zero rate) matches reality, so the
data is fine — the *learned* gate head is structurally biased low. Because
the ZINB mean is `(1−gate)·base`, the model compensates with a lower base, so
the **mean** looks roughly right while the **distribution shape** is wrong:
far too little mass at/near zero ⇒ `P(over@line)` inflated at every line ⇒
confident Overs in every ZINB market. FG3M is simply the most-published one.

The gate is **unidentified under joint NLL with a flexible count head**
(classic ZINB identifiability issue): the count component can absorb zeros, so
the optimiser trades gate vs. count and settles on a too-low gate. Bounding
`total_count` (`r`) and reseeding the gate start value were both prototyped
and **did not** fix it (the count ceiling didn't bind; the start value is
overridden by boosting). The fix must change the estimation structure.

### Recommended fix (designed, specced, not yet implemented)

A **consistent two-stage ZINB with a derived gate** — keeps true
zero-inflation semantics (the NB still emits its own sampling zeros; the gate
is *inflation only*), and is statistically correct (no hurdle/ZINB
double-count):

1. A separately-trained **calibrated binary classifier** estimates the
   *observable total* zero probability `q = P(Y = 0)` (well-identified — this
   is what the joint NLL gate gets wrong).
2. A NegBin supplies the count shape (`total_count`, `probs`); `NB(0)` is its
   natural sampling-zero mass.
3. **Derive** the structural inflation gate from the exact ZINB identity
   `q = π + (1−π)·NB(0)` ⟹ `π = clip((q − NB(0)) / (1 − NB(0)), 0, 1)`.

Then `P(0) = π + (1−π)·NB(0) = q` by construction (no double count). A wrapper
class exposes `.predict(..., pred_type="parameters")` returning the *same*
`total_count/probs/gate` columns as `ZINB`, so **all downstream code
(`get_odds`, `fused_loc`, `get_ev`, the `model_prob` ZINB branch) is
unchanged.** Literature basis: hurdle vs zero-inflated distinction
(zero-truncated count vs. untruncated); two-stage estimation of the
observable total-zero probability sidesteps the EM normally needed for latent
structural zeros.

Full task-by-task TDD spec (Phase B, "SUPERSEDED → derived-π" section):
`docs/superpowers/plans/2026-05-18-fga-fg3m-overconfidence-fix.md`.

---

## 3. FGA-under — investigated extensively; root cause NOT resolved

The bulk of the effort went here and produced mostly **negative results**.
Documenting them so they are not repeated.

### 3.1 Hypotheses tested and ruled out

| Hypothesis | Test | Outcome |
|---|---|---|
| SkewNormal scale/dispersion under-predicted (overconfident spread) | offline replay; CRPS→NLL + bounded scale prototype | **Refuted.** FGA raw holdout Brier 0.21 (best in cohort); not a dispersion problem. Prototype reverted. |
| `Result/MeanYr` ratio normalization causes regression-to-global-mean | corr(predicted loc, MeanYr) path-wide sweep | Real *slope* artifact (corr −0.37…−0.87 across all SkewNormal markets) but see below — not the dominant level cause. |
| Additive empirical-Bayes per-player offset fixes it (mixed-effects approximation) | full TDD implementation (Phase A) + retrain + holdout | **Inconclusive / dead end** — see §3.2. Reverted. |
| Bug: `set_model_start_values` offset-mode seeded `loc=0` | controlled isolation experiment | Confirmed a regression; fixing to `loc=mu` halved bias in isolation but did not reproduce on retrain. |
| GPBoost mixed-effects (per-player random intercept) — the literature's canonical remedy | deterministic GPBoost prototype | **Failed.** Did not beat the EB offset; top-volume bias −2.5. Its "flat fixed-effect" decomposition was a GPBoost-*internal* FE/RE artifact, not the production model. |
| SHAP feature filter drops volume features | plain-LGBM filtered vs full-feature, importance | **Refuted.** All volume features kept; `Mean10` is the #1 feature by ~15×. |
| Monotone constraints / heavy Optuna regularization flatten the location head | 2×2 (crps/nll × prod/light params) attribution probe | **Refuted.** Every SkewNormal config tracked volume well. |
| CRPS loss (Hessian≡1) flattens the SkewNormal location | same attribution probe | **Refuted.** CRPS configs recovered volume spread ≥ plain L2. |

### 3.2 The decisive negative result

In a **clean, deterministic** harness the *production-equivalent* SkewNormal
model (CRPS + production Optuna params + additive-EB offset, prior re-added at
predict) is **well-calibrated**: predicted volume-quintile spread 7.4 vs
actual 8.3, `meanAbsBias` **+0.12**, tracks volume correctly. A plain L2
regressor on absolute FGA also tracks volume well.

So the SkewNormal training stage is **probably not** the source of the live
FGA bias. The earlier "it's broken" conclusions came from **non-reproducible
offline runs** (see §4).

### 3.3 The methodological blocker — non-reproducibility

The same nominal SkewNormal configuration produced top-volume-quintile bias
of **−0.48, −0.92, −1.3, −2.0, −2.5, and +0.12** across different
harnesses/runs. Suspected sources:

- LightGBM / LightGBMLSS seeding not pinned.
- Per-row `start_values` broadcasting in LightGBMLSS predict.
- Optuna nondeterminism.
- `train_market`'s Optuna **starved** in time-boxed offline runs (smoke
  retrains completed only 3–18 trials vs. the deployed model's 374 rounds),
  producing degenerate hyperparameters.

**Until offline evaluation is deterministic, no training-side change to FGA
can be validated.** This is the single highest-leverage prerequisite for any
resumed FGA work.

### 3.4 The under-investigated lead (most aligned with the symptom)

`Model Skew` (SkewNormal `alpha`) is **NaN for every live FGA row**, yet
offline replay proved the trained model predicts *valid* alpha on saved
features. The NaN therefore arises in the **live prediction path**
(`src/sportstradamus/prediction/model_prob.py`) — candidate causes: live
`playerStats` feature/column misalignment, `set_model_start_values` seeding
differing live vs. train, or the post-model `temperature ≈ 1.37` /
`weight = 0.9` bookmaker blend. This was observed early, labelled
"secondary", and **prematurely descoped** while effort went to the training
target. It is the stage most consistent with the actual user-reported symptom
(EV ≈ 12.5 vs line ≈ 18, 100% Under at the cap) and was never given a proper
root-cause pass.

---

## 4. Recommended resume plan (priority order)

1. **Deterministic evaluation harness (prerequisite).** Pin
   LightGBM/LightGBMLSS seeds, fix `start_values`, deterministic
   train/test split, controlled or fixed-param Optuna for evaluation. Add a
   determinism gate: run the same config twice → bit-identical predictions.
   Nothing else about FGA is verifiable until this exists. (GPBoost's harness
   already demonstrated bit-identical determinism is achievable.)

2. **Live `model_prob.py` FGA diagnosis.** Deterministically reproduce the
   end-to-end live path: raw SkewNormal params → decode → `fused_loc` (w=0.9
   bookmaker blend) → `dispersion_cal` → `temperature`. Identify which stage
   collapses EV relative to the line and **why `Model Skew` is NaN live but
   valid on saved features.** This directly targets the symptom.

3. **Implement Root Cause B (ZINB derived-π gate).** Independent of FGA,
   evidence is solid and reproducible, downstream code is unchanged, and the
   task-by-task spec already exists. This is the safest, highest-confidence
   win.

---

## 5. Artifact pointers

- This report: `docs/OVERCONFIDENCE_INVESTIGATION.md` (tracked).
- Concise hand-back: `docs/superpowers/plans/2026-05-19-overconfidence-findings-handback.md`.
- Detailed implementation spec (Phase A abandoned; **Phase B derived-π still
  valid**): `docs/superpowers/plans/2026-05-18-fga-fg3m-overconfidence-fix.md`.
- Investigation history / decisions:
  `~/.claude/plans/my-models-seem-to-jazzy-minsky.md`.
- Project memory: `overconfidence-investigation` (so future sessions don't
  repeat the SkewNormal dead end).
- Code state: baseline `64a26b9`; Phase-A commits (`fb04444`, `c8b7020`,
  `518a544`, `e910bcc`, `f159ad9`) were reset out. `gpboost` was
  `pip install`ed into the venv for a prototype (pyproject/lock untouched).
- Throwaway diagnostic scripts were under `/tmp` (`protoA2_meanrev.py`,
  `protoB2_hurdle.py`, `isolate_fga.py`, `sweep_fga_k.py`, `gpb_harness.py`,
  `attr_diag.py`, `fga_exp.py`) — ephemeral; re-derive from this report.

## 6. One-line takeaway

ZINB gate under-fit (FG3M-over) is real, reproducible, and ready to fix via
the derived-π two-stage design. The FGA-under training rework was a dead end
built on non-reproducible metrics; resume FGA only after (a) a deterministic
eval harness and (b) a proper root-cause pass on the live `model_prob.py`
path (`Model Skew`=NaN + bookmaker blend).
