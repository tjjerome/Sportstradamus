# Model-strategy reachability repairs

> Status: ACTIVE — recommendations 1/2/3/5 landed; awaiting the isolated targeted
> sweeps for the three NFL acceptance cells (§7).

## 1. Mission

Repair the generic strategy-research pipeline so registered structural methods,
historically valid incumbents, and genuinely different confirmation mechanisms
remain reachable from holdout-blind sweep through fresh retrain.

NFL receiving yards, rushing yards, and passing yards are mandatory acceptance
cells, but the implementation must operate through registry metadata and
applicability rules rather than market-specific branches. Passing all six gates
is an empirical outcome, not a code-test assertion.

## 2. Read first

Read these after `CLAUDE.md` and `CONTRIBUTING.md`:

1. [`model_improvement_track.md`](model_improvement_track.md) — strategy, gate,
   and experiment contracts.
2. [`matrix-provenance-and-crossfit-gates.md`](../archive/matrix-provenance-and-crossfit-gates.md)
   — frozen-matrix and cross-fit constraints.
3. `src/sportstradamus/training/model_strategy/` — registry, sweep, confirmation,
   and artifact identity.
4. `src/sportstradamus/training/pipeline.py`, `structural_context.py`, and
   `group_conditional_cdf/` — structural fit and calibration flow.
5. `src/sportstradamus/training/scorecard.py` — persisted distribution contract
   and six-gate scoring.

Before editing, inspect `git status --short` and preserve unrelated matrix,
role-registry, or shipping work already in the tree.

## 3. Verify before you trust

Run these checks at the start. If their output contradicts this brief, the
output wins; update the brief for a minor drift or stop and ask the owner when
the implementation contract materially changed.

```bash
git status --short
pgrep -af "[m]editate|model.strategy.(sweep|confirm)" || true
rg -n "holdout-blind does not support structural|not spec.is_structural" \
  src/sportstradamus/training
rg -n "SEED_CORNERS|_known_good_corners|_incumbent_corner" \
  src/sportstradamus/training/model_strategy
rg -n "_board_lane|_count_class_backup|CONFIRM_TOP_K" \
  src/sportstradamus/training/model_strategy/confirm.py
```

The defects to verify are structural exclusion from holdout-blind search,
whole-fit structural state leaking into cross-fit artifacts, absent targeted
seeds, omitted defaults hiding incumbents, and rank-only top-K nomination.

## 4. Locked scope

Implement recommendations 1, 2, 3, and 5 only:

1. Fold-stable structural search and confirmation.
2. Mandatory current-matrix evaluation seeds.
3. Historical-default incumbent reconstruction.
5. Mechanism-diverse confirmation nominees.

Do not change matrix construction or rebuild policy, gate thresholds,
negative-board confirmation policy, automatic shipping state, dirty-tree
policy, or provenance policy. Do not tune recipes against retained holdout
outcomes. Generalize through strategy registry data, not league/market
conditionals; the three NFL cells belong only in the seed declarations and
acceptance run.

## 5. Stage plan

### Stage 1 — Structural holdout-blind support

Remove the explicit structural rejection from holdout-blind training and the
structural exclusions from sweep family and priority-corner discovery. Existing
capability and applicability contracts remain authoritative.

Build routes, role thresholds, gate rates, and affine expert sets once from the
original train/full-validation split. One structural activation decision applies
to the entire trial:

- An inactive context fails the corner; never score a pooled/base fallback
  under a structural identity.
- Affine keeps one train-derived expert set. Failure in any calibration fold
  aborts the entire corner.
- Two-part uses one grouping. Force `role_by_position` through the support audit
  for every outer calibration-fit partition and pin it only if all pass.
  Otherwise test forced `role_only` across all partitions. Fail when neither is
  universally supported. Use the pinned grouping for every fold and the final
  whole-validation fit.

Normalize fit-dependent structural row output under one internal
`structural_rows` mapping with `test_route`, `prepool_over`,
`pit_recal_by_row`, `two_part_calibration_by_row`, `two_part_f0`,
`two_part_role`, and `two_part_position`.

Non-applicable fields are `None`. Cross-fit and restore the original row order
for every fit-dependent field. The final whole-validation fit continues to
supply the serving pickle's constants and validation audit.

Extend the two-part scorecard contract to accept row-specific
`StructuralCalibration` payloads in deterministic cross-fit CSVs. Deserialize
each unique payload once and apply it only to its rows. Continue accepting the
single constant payload emitted by ordinary full-HPO training.

Bump both structural strategy implementation versions. Bump the two-part
artifact schema because its deterministic CSV representation changes. Existing
structural board rows and cached deterministic artifacts must become stale and
be regenerated; base strategy artifacts remain compatible.

Acceptance: both structural strategies complete holdout-blind training when
supported; every scored row uses calibration state fitted without that row; no
fold changes grouping or experts; row-specific payloads/maps validate; and
constant full-HPO structural artifacts remain valid.

### Stage 2 — Seed and incumbent reachability

Split seed semantics into:

- `MANDATORY_SWEEP_CORNERS`: evaluated on the current matrix but never
  synthesized directly by confirmation.
- `CONFIRM_EVIDENCE_CORNERS`: independently proven full-HPO recipes available
  to sweep and confirm.

Register these mandatory sweep corners:

- NFL receiving yards: the two-part structural spec's exact fixed controls.
- NFL rushing yards: the affine structural spec's exact fixed controls.
- NFL passing yards: `SkewNormal`, `ratio_meanyr`, CRPS distribution loss,
  direct parametrization, NLL blending, and `posthoc=none`.

Keep the existing passing-TDs DPO recipe in `CONFIRM_EVIDENCE_CORNERS`.

Enqueue, before sampler suggestions: mandatory cell corner, confirm-evidence
corner, then reconstructed incumbent. Deduplicate by the current matrix-bound
corner fingerprint. Seed declarations contain recipes only; a matrix change
therefore produces a new evaluation identity while exact resume can reuse an
existing row.

Structural mandatory corners may reach confirmation only through validated
board rows carrying a structural split fingerprint. Preserve the positive-board
requirement; mandatory evaluation is not a negative-board confirm exception.

Acceptance: all three recipes precede sampler trials; exact resume reuses a
fingerprint while a matrix change forces reevaluation; confirmation cannot
synthesize a structural seed; and the passing-TDs evidence seed remains.

### Stage 3 — Historical incumbent defaults

Add a validated strategy-level `INCUMBENT_CONTROL_DEFAULTS` registry containing
effective historical behavior, not CLI `auto` sentinels:

- SkewNormal: `ratio_meanyr`, CRPS loss, direct parametrization, canonical
  default blending, and no post-hoc.
- Mixture: `ratio_meanyr`, canonical default blending, and no post-hoc; retain
  its fixed NLL loss.
- ZINB: joint mode, CRPS dispersion, canonical default blending, and no post-hoc.
- NegBin/DPO: CRPS dispersion, canonical default blending, and no post-hoc.
- Structural strategies remain fully described by fixed controls.

Reconstruct controls in this order:

1. Strategy fixed controls.
2. Explicit persisted cell values.
3. Historical effective defaults.
4. Failure if any required control remains unresolved.

Validate the completed recipe against `strategy_controls(spec)`. Never coerce
an explicit incompatible or obsolete value.

Acceptance: legacy passing-yards metadata reconstructs CRPS/direct; explicit
values override defaults; count defaults reproduce effective CLI behavior;
invalid recipes return no incumbent; and tests bind defaults to CLI behavior.

### Stage 5 — Mechanism-diverse confirmation

Define two internal keys:

- Primary mechanism: `strategy_slug`.
- Gate-4 mechanism: strategy slug plus normalization, distribution loss,
  SkewNormal parametrization, ZINB mode, count-dispersion objective, blending
  loss, and post-hoc only for mean, whole-CDF, or structural stages.

Treat `posthoc=none` and probability-only recalibrators as the same Gate-4
mechanism because they do not change the predictive PIT distribution.

Order positive board nominees by:

1. Highest-ranked candidate.
2. Highest-ranked candidates with unseen primary mechanisms.
3. Candidates with unseen Gate-4 mechanisms.
4. Remaining candidates in ordinary rank order.

Keep `CONFIRM_TOP_K` for the normal board lane and retain the integer-target
count-class backup as an additional insert-only nominee. Preserve cross-fit-only
eligibility, positive-rank filtering, fingerprint deduplication,
evidence/incumbent ordering, and stop-on-first-success behavior.

Acceptance: near-identical ZINB corners cannot crowd out positive DPO or
structural rows; distribution-shaping alternatives precede probability-only
variants; rank order remains when no alternative exists; and nonpositive cells
receive no confirm walk.

## 6. Identity and failure contracts

- `split_fingerprint` represents the complete stable validation membership,
  never an individual calibration fold.
- Structural confirmation candidates originate from a board row and retain its
  split fingerprint.
- Matrix pinning verifies the frozen source hash equals the candidate matrix hash
  before training.
- Full-HPO model, CSV, and model-stats identities reproduce the expected
  structural split fingerprint.
- Matrix, split, controls, implementation-version, or artifact-schema mismatch
  fails closed.

## 7. Verification

Add focused tests for structural support and stitching, row-specific scorecard
contracts, seed ordering, incumbent reconstruction, nominee diversity, and
matrix/split identity mismatches. Then run:

```bash
poetry run ruff check src/sportstradamus/
poetry run pytest tests/golden/
poetry run pytest -m integration -n0
```

Run isolated targeted sweeps for NFL receiving, rushing, and passing yards only
after the test suite is green and no other sweep or `meditate` owns the training
artifacts. Acceptance requires each mandatory row to carry the current matrix
identity, supported structural rows to complete holdout-blind scoring, and
unsupported folds to fail rather than emit fallback scores. Run normal full-HPO
confirmation only for positive nominees.

## 8. Stop conditions

Stop and ask the owner before changing a gate, confirming a board-negative
recipe, modifying matrix construction, altering shipping state, or overwriting
another agent's work. If a structural method cannot be made honestly cross-fit
without a new artifact representation beyond the row contract above, record the
specific incompatibility before expanding scope.

## 9. Session definition of done

All requested stages are implemented without excluded recommendations; lint,
golden, and integration suites pass; targeted evidence is isolated; shipping
state is unchanged; and the ledger records the commit, checks, structural
versions, and next empirical action.

## 10. Ledger

Append one line per implementation session, newest first:

`YYYY-MM-DD · stage N · what landed (SHA) · checks ✓/✓/✓ · next: <one clause>`

2026-08-03 · recs 1/2/3/5 · holdout-blind structural cross-fit via a pinned two-part
grouping and a general structural-row stitch, mandatory sweep seeds split from
confirm-only evidence, incumbent reconstruction through control defaults, Gate-4
mechanism-diverse nominees; two-part artifact schema 4, affine artifact schema 2 (was
declared 1 against a writer stamping 2 — every fresh affine artifact was being
rejected), both structural specs implementation v2 (d4730ae) · checks ✓/✓/✓ · next:
isolated targeted sweeps for NFL receiving, rushing, and passing yards
