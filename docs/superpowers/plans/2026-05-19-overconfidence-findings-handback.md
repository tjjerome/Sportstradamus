# FGA-under / FG3M-over overconfidence — consolidated findings & hand-back

**Status:** PAUSED at user request. All Phase-A code reverted; repo is at clean
baseline commit `64a26b9` ("add plan"), working tree clean. No production code
changes remain. (The local gitignored `NBA_FGA.mdl` was left as an experimental
artifact; user is pulling a production model — not a concern.)

This document is the decision-ready summary so work can resume later without
re-deriving anything.

---

## 1. The symptom (ground truth — trustworthy)

From the real published output `src/sportstradamus/data/current_offers.parquet`:

- **FGA (SkewNormal):** 3/3 live offers = **Under**, `Model P` pinned at the
  0.9 cap; `Model EV` ≈ 12.5 vs line ≈ 18 (≈30% below line, below recent
  form); **`Model Skew` (SkewNormal alpha) = NaN` on every FGA row** while
  Model EV/Sigma are populated.
- **FG3M (ZINB):** 42/44 = **Over**; `Model EV` ≈ 2.74 vs line ≈ 1.25 and vs
  recent 5-game form ≈ 1.2.

## 2. Confirmed findings (high confidence, reproducible)

- **Root Cause B — ZINB gate under-fit (path-wide, deterministic).** Offline
  replay of every NBA ZINB pickle: the learned per-offer `gate` ≈ **half** the
  true structural-zero rate in every market (FG3M 0.19 vs 0.33; PF 0.02 vs
  0.14; TOV 0.13 vs 0.32; STL 0.23 vs 0.48; …). `hist_gate` itself is correct,
  so the defect is the jointly-fit ZINB gate head under `nll`. Effect:
  P(over@line) inflated everywhere (predicted 0.67–0.97 vs actual 0.34–0.44)
  → confident Overs across all ZINB markets. This reproduced cleanly every
  time. **The derived-π two-stage construction is the sound fix** (see §5).

- **SkewNormal training is probably NOT the FGA root cause.** In a *clean
  deterministic* harness the production-equivalent SkewNormal model
  (crps + prod Optuna params + additive-EB offset, prior re-added at predict)
  is well-calibrated: predicted vol-quintile spread 7.4 vs actual 8.3,
  meanAbsBias **+0.12**, tracks volume. A plain L2 regressor on absolute FGA
  also tracks volume well (`Mean10` is the #1 feature by ~15×). Features, the
  SHAP filter, monotone constraints, and CRPS/Hessian were each tested and
  **ruled out** as the cause of any flat fixed-effect.

- The "flat fixed-effect" reported mid-investigation was a **GPBoost-internal
  FE/RE decomposition artifact**, not a property of the production model. (My
  error — conflated GPBoost's own latent split with the deployed model.)

## 3. What went wrong / open problems

- **Offline evaluation is non-reproducible.** The same nominal SkewNormal
  config produced top-volume-quintile bias of −0.48, −0.92, −1.3, −2.0, −2.5,
  and +0.12 across harnesses/runs. Suspected sources: lightgbmlss/LightGBM
  seeding, per-row `start_values` broadcasting, Optuna nondeterminism, and
  `train_market` Optuna **starvation** in time-boxed offline runs (smoke runs
  completed only 3–18 Optuna trials vs the deployed model's 374 rounds).
  **Until this is fixed, no training-side change can be validated.**

- **`Model Skew` = NaN in the live path was prematurely descoped.** Offline
  replay showed the trained FGA model predicts *valid* alpha on saved
  features, yet the live `current_offers.parquet` has alpha = NaN for every
  FGA row. This points at the live `model_prob.py` SkewNormal path
  (feature/column alignment, `set_model_start_values` seeding live vs train,
  or the `temperature=1.37` / `weight=0.9` bookmaker blend) — the stage most
  aligned with the actual user-reported symptom and **least investigated.**

## 4. Recommended next steps (priority order)

1. **Determinism harness (prerequisite, small, high-leverage).** Seed
   LightGBM/lightgbmlss, fix `start_values`, deterministic split, controlled
   Optuna (or fixed params for eval). Add a determinism gate (same config run
   twice → bit-identical). Nothing else is verifiable until this exists.
2. **Live `model_prob.py` FGA diagnosis.** Deterministically reproduce the
   live path end-to-end (raw SkewNormal params → decode → `fused_loc` w=0.9
   book blend → `dispersion_cal` → `temperature`). Find which stage collapses
   EV vs line and **why `Model Skew`=NaN live but not on saved features.**
   This targets the actual symptom.
3. **Root Cause B (ZINB derived-π gate).** Independent of FGA, evidence is
   solid and reproducible. Detailed task-by-task spec already written (Phase B,
   superseded to the derived-π design) at
   `docs/superpowers/plans/2026-05-18-fga-fg3m-overconfidence-fix.md` — see the
   "Phase B SUPERSEDED" note: `gate = clip((q − NB(0))/(1 − NB(0)), 0, 1)`
   with a calibrated zero classifier `q`. Reuses all downstream ZINB code.

## 5. Pointers

- Prior detailed implementation spec (Phase A now abandoned; **Phase B
  derived-π section still valid**):
  `docs/superpowers/plans/2026-05-18-fga-fg3m-overconfidence-fix.md`
- Canonical investigation plan/history:
  `/home/trevor/.claude/plans/my-models-seem-to-jazzy-minsky.md`
- Throwaway diagnostic scripts were under `/tmp` (`protoA2_meanrev.py`,
  `protoB2_hurdle.py`, `isolate_fga.py`, `sweep_fga_k.py`, `gpb_harness.py`,
  `attr_diag.py`, `fga_exp.py`) — **ephemeral, may not persist**; re-derive
  from this doc if needed. `gpboost` was `pip install`ed into the venv for a
  prototype (pyproject/lock untouched); harmless, remove if undesired.

## 6. One-line takeaway

Root Cause B (ZINB gate) is real, reproducible, and ready to fix via the
derived-π two-stage design. The FGA "Root Cause A" training rework was a dead
end built on non-reproducible metrics; the real FGA lever is (a) deterministic
evaluation and (b) the live `model_prob.py` path (Model Skew=NaN + book
blend), which still needs a proper root-cause pass.
