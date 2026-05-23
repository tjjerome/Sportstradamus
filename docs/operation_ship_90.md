# Operation Ship 90 — Stub

> **Status:** stub. Filled in as Operation Ship 75 lands and we see
> which cells resisted. This doc captures ideas as they surface so they
> aren't lost.

## North Star

**90% of markets per covered league** past Tier-0 quality gates:

- NBA ≥ 19/21
- WNBA ≥ 17/18
- NFL ≥ 18/20

Likely paired with a **Gate-1 tightening** (e.g. G2/G3 z-bound from
0.5 → 0.3; G5 ECE from 0.075 → 0.05). The user decides which gates
tighten and when, based on Ship 75 results.

## Likely targets

- The **`deferred-90`** cells from Ship 75 (lever-exhausted after
  Steps 0–4).
- The **`degenerate`** cells from Ship 75 — NFL receiving-tds /
  rushing-tds / tds with `IQR_true = 0` — if Step 0.4 didn't resolve
  them.
- Re-entry candidates for the **CMPμ / marginalized-hurdle / MZINB
  family build** per the Stage B1.5 §7a re-entry condition
  (conditional Dunn–Smyth RQR variance < 0.70 AND Poisson GBM tracking
  the top decile while NB compresses). See
  [`operation_ship_references.md`](operation_ship_references.md) §7.
- Cells that ship under Ship 75 but live-soak regresses
  (Gate-2 re-entry).

## Ideas to research closer to start time

- **T11 — per-position model split.** Deferred from Track A (see
  [references doc](operation_ship_references.md) §9). Enabled by
  Stage A1.6 NFL position-scoping work. Train separate model per
  (position, market) instead of pooled cross-position with categorical
  feature. Selective, not wholesale — split only where eligible-position
  marginals diverge materially (rushing-yards QB-scramble ~19 vs
  RB-workhorse ~37). Tight receiving stays pooled. Min-row guard +
  fallback to pooled+categorical below threshold.
- **Marginalized hurdle** ([ref 25]) for genuinely-inflated ZINB cells
  if Ship 75 doesn't get them (NFL passing-tds, interceptions). Smaller
  delta than MZINB; the literal joint-fit version of the two-stage
  hurdle currently shipped. **Consumes `_ZeroTruncatedNB`** (unwired
  block kept in [`hurdle.py`](../src/sportstradamus/hurdle.py)).
- **MZINB** ([ref 27]) only for true structural-zero excess.
- **CMPμ** ([ref 5]) for genuinely-underdispersed cells (only if Step 5
  re-entry condition fires — Ship 75 §7a verdict says it probably
  won't). Mean-parameterized Conway-Maxwell-Poisson with log-link μ,
  orthogonal dispersion ν. Custom distribution; precomputed (μ, ν) → λ
  look-up grid with bilinear interpolation, refreshed once per market.
  Ceiling ~3–8% top-decile MAE improvement.
- **GBDT mean-head replacement candidates** (deferred from Track A4):
  - **T2 CatBoost ordered TS** ([ref 17]): only published GBDT
    mechanism with a proof of unbiasedness for high-cardinality
    categoricals (`player_id`). Caveat: proof is for log-loss /
    squared-error, not SkewNormal NLL.
  - **T4 MEGB / GBMixed**: EM/BLUP mixed-effects boosting that fixes
    the bias GPBoost was criticized for in [9]. MEGB on CRAN +
    github.com/rid4stat/MEGB.
  - **T10 PGBM** ([ref 21]): mean + variance from a single ensemble
    without parametric distribution; avoids SkewNormal shape bound.
- **Calibration-on-scale (CQR / LCMQR)** ([refs 14, 15]) once the mean
  is dialed in. Post-hoc per-player-decile calibration of *scale*;
  orthogonal to mean correction.
- **gbex** ([ref 18]) — Generalized Pareto tail boosting on
  exceedances, layered on the LSS body. Good parallel to T3 (spliced
  tail head, deferred from A2 due to breadth priority).
- **FAGTB adversarial penalty against MeanYr decile** ([refs 19, 20]).
  Quantile-bucket MeanYr (10 deciles); adversary predicts decile from
  residual; penalize loc gradient by adversary loss.
- **Inference-cost optimizations** once per-cell calibration objects
  stack up. The pickle file size grows with each new key
  (`temperature`, `bias_correction`, `cqr_blob`, etc.); load time on
  `prophecize` may regress.

## Open questions to address pre-launch

- **Gate-1 tightening — before or after Ship 90 starts?** Tighten
  early forces some Ship 75 graduates back into queue; tighten late
  defers the question. Decision deferred to user based on Ship 75
  results.
- **Are there profitable markets currently locked out by the existing
  gates that should NOT be tightened?** Profit-sim
  (`dashboard pages/3,4,6`) has the answer. Check before tightening
  any gate.
- **The `model-research` → `devel-foundation` → `devel` → `main`
  branch ladder.** Old plan §"Branches & model-promotion flow"
  describes this; under Ship 75 we ship per-cell directly from
  `model-research` via the `devel-ship-curator` agent. If Ship 90
  needs a more disciplined staging pipeline, re-instate the foundation
  branch.
- **Live-data drift detection.** Stage 0 instrumentation persists
  rolling 7d/30d metrics. By Ship 90 launch we should have ≥ 6 months
  of live data per cell — strong enough to detect drift / regime
  changes (e.g. NBA 3PA trends [28]).

## Out of scope

- Anything that requires changing the LightGBMLSS framework upstream.
- Adding new leagues (Ship 90 covers NBA / WNBA / NFL only).
- Replacing the GBDT base learner wholesale (out of scope until the
  full lever stack — Steps 1–5 from Ship 75 — has been exhausted on
  every cell).

## Excluded markets register

Cells removed from the denominator if Step 0.4 of Ship 75 can't fix
them (TBD as Ship 75 lands):

- (placeholder) NFL receiving-tds — `IQR_true = 0`
- (placeholder) NFL rushing-tds — `IQR_true = 0`
- (placeholder) NFL tds — `IQR_true = 0`

If any of these are excluded, the Ship 90 target adjusts accordingly
(e.g. NFL ≥ 17/20 if 1 cell excluded).

## Reading list (when Ship 90 starts)

1. [`docs/operation_ship_75.md`](operation_ship_75.md) — what shipped /
   what didn't (the `deferred-90` register)
2. [`docs/operation_ship_references.md`](operation_ship_references.md)
   — research history
3. [`docs/archive/gbdt_mean_regression_*.md`](archive/) — original
   plan + context (full prose for anything cited)
4. [`docs/ship_gate.md`](ship_gate.md) — current Tier-0 / Tier-1
   thresholds
