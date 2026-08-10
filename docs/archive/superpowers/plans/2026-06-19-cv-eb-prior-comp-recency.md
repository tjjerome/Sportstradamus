# CV the §6.3 feature constants: `_EXPANDING_EB_PRIOR_K` + `_COMP_RECENCY_HALFLIFE_DAYS`

Next-session plan. Both §6.3 feature constants shipped as documented placeholders; the discipline
(ledger "cross-validated … do not invent a value") is to CV before fixing them.

## Constants (independent — CV separately)

- **`_EXPANDING_EB_PRIOR_K = 10.0`** (`base.py:60`, used `base.py:1606`). Efron-Morris shrink of
  `MeanYr_expanding_eb` toward the per-position baseline: `shrink = n/(n+K)`. Bigger K shrinks to
  the prior longer — sparse weekly NFL likely wants a larger K than dense NBA. Affects every league
  with expanding features.
- **`_COMP_RECENCY_HALFLIFE_DAYS = 45.0`** (`base.py:72`, used `base.py:1748`). Comp recency weight
  `exp(-days_ago/halflife)` feeding `comps z recent` / `comps trend`. Shorter = more recent-form.
  Comp-pool leagues only (NBA/WNBA/NFL/NHL; MLB excluded).

## Method — tiered, held-out only (no test-refit)

Feature constants ⇒ a faithful eval recomputes the affected column at each grid point. Tier it:

1. **Cheap feature-signal sweep** (narrows the grid). Per league, on a CPCV split with a
   player/date embargo (López de Prado), recompute *only* the one column at each grid point and
   measure its out-of-fold signal vs the target (single-feature OOF corr / R²; no model retrain —
   both columns are closed-form over cached inputs). Grids: K ∈ {3,5,10,20,40};
   halflife ∈ {14,30,45,90,180} days.
2. **Model confirm** on the top 1–2 per league: regen the column at that value → `meditate
   --deterministic` → `training.scorecard` on the volume + 6 just-shipped cells, vs the 10/45
   placeholder. Held-out gate / Brier / EV-RMSE.

## Decision

Per-league optima differ materially ⇒ promote to a per-league dict (allowed parallel block: same
shape, genuinely different knowledge). Cluster near 10/45 ⇒ keep the global, mark CV-confirmed.

## Scope

- New dev-only driver `scripts/cv_feature_constants.py` (resumable) — **hard-exclude from devel**
  (like `regen_ab_batch.py`). Touches `base.py` + the ledger. Preserve parquets (`.regen_backup/`).
- Reuse `get_training_matrix`, `trim_matrix`, `meditate --deterministic`, `training.scorecard`.

## Also pending (§6.3 follow-up)

- Re-confirm WNBA fantasy-points (omitted from the confirm groups).
- `PrimeTime` / `RestDiff`: revert vs backfill (globally inert pending the gametime/rest backfill).
- Investigate the 4 A/B regressors: NBA_STL, WNBA_OREB, NFL qb-yards, NFL receiving-tds.
