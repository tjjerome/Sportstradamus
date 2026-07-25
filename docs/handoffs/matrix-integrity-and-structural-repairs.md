# Matrix integrity and structural-calibration repairs

> Status: COMPLETE (Stages 0–5 executed; Stage 5 retains honest KILL evidence)

> Current checkpoint: two real repeated-build checks rejected their inputs before retraining. The first
> exposed unordered team-feature columns and tied player-depth ranks; the second exposed a few-ULP
> `EV`/`Odds` drift from unordered latest-per-book archive rows. After stable book ordering was added, fresh
> concurrent read-only rebuilds C/D matched exactly in frame contents, dtypes, column order, parquet bytes,
> stable manifest fields, and manifest SHA for attempts (`1bfbe4bf...`, 2,463 rows), carries (`c43479a0...`,
> 6,627 rows), and targets (`8700a730...`, 13,978 rows), each with 477 columns. The earlier dependency
> artifacts remain non-authoritative and will not be relabeled. Corrected full-HPO retraining from the
> accepted C matrix root is complete in the isolated dependency quarantine: attempts, carries, and targets
> all pass the strict `volume-v1` loader, source-matrix SHA binding, schema, timestamp, version, distribution,
> and isolated test-CSV checks. Stage 2 is complete: latest-per-book fields stay row-coherent; direct
> probabilities use the outcome-independent `modal-nearest-median-v1` same-line cohort; EV inversion and
> fallbacks are explicitly derived/synthetic; scratch, append, and repair share one pure resolver; and
> structural pooling consumes only explicit `QuoteAuthenticity == "authentic"`. Stage 3 cleanup is active.
> The re-derived Stage 3 active-cell audit is also complete: all 93 legacy matrices fail only because the
> canonical tree predates explicit provenance, with 15 authentic-supported, 62 partial, and 16 bookless
> cells under the legacy compatibility projection. The new auditor additionally identifies 12 matrices
> with invalid authentic quotes, 19 with runaway EV residue, and 11 NHL matrices with position leakage.
> Runtime repair now rejects runaway archive EV before it can enter any scratch/append/repair matrix, and
> the complete 15-market NHL policy is fail-closed. The first quarantined `timeOnIce` candidate was
> rejected after its diff exposed `_step_synthesize_odds` re-inverting every already-resolved non-authentic
> quote and overwriting source provenance. The mask now touches only missing/zero Odds, preservation is
> regression-pinned. Corrected candidate B is clean in quarantine: `timeOnIce` has 62,202 rows and
> SHA `0193cc93...` with only eligible positions, while `shotsAgainst` has 5,543 all-goalie rows and
> SHA `cb44cc0f...`; every quote is honestly classified bookless/synthetic and both manifests validate.
> Independent cold rebuild C matches B exactly in frame contents, dtypes, column order, parquet bytes,
> SHA, and every stable manifest field (`as_of` is the sole expected difference). Both identities pass
> the auditor with zero violations and are accepted. Isolated full-HPO from immutable candidate C is
> complete under `volume-v1`: both artifacts pass strict identity, matrix binding, distribution, schema,
> version, timestamp, and test-set checks. Canonical matrices, models, and test sets remain untouched.
> The dependent goalie-only candidate A is complete from that isolated dependency root: `saves` has
> 6,194 rows / SHA `e8f11ac0...`, `goalsAgainst` has 5,794 / `aa7f6e09...`, and `goalie fantasy points
> underdog` has 5,802 / `82429686...`. All three are position `G` only, carry all six provenance fields on
> every row, and pass the auditor with zero duplicate, nonfinite, invalid/runaway quote, provenance, or
> position violations. The full old/new identity, raw-outcome, feature, position, quote-source,
> synthetic-rate, and coverage diff is recorded below. Candidate B is now rebuilding independently with
> the identical recipe. `saves` now matches A/B exactly in frame contents, dtypes, column order, parquet
> bytes, SHA, and stable manifest fields (`as_of` alone differs), so its identity is accepted;
> `goalsAgainst` now also matches exactly and is accepted at SHA `aa7f6e09...`; goalie fantasy
> points now matches exactly at SHA `82429686...` as well. The complete three-cell downstream goalie
> set is independently reproduced and accepted.
> The first prioritized `NFL_tds` candidate correctly failed on its first 2021 slate: the accepted volume
> dependencies expect external snapshot features whose local history starts later, while inference used a
> strict column slice before it could reproduce the upstream matrices' historical zero representation.
> Snapshot-only dependency inference now reindexes missing historical features to zero, matching the
> accepted volume training rows; live serving remains strict. The focused loader/lineage suite passes
> 16/16. Corrected `NFL_tds` candidate B is clean at 16,451 rows / SHA `a4b17f4e...`, with complete
> provenance and zero auditor violations. Independent repeat C matches B exactly in frame contents,
> dtypes, column order, parquet bytes, SHA, and every stable manifest field (`as_of` alone differs), so
> the corrected `NFL_tds` identity is accepted. MLB cold preflight now
> distinguishes structural hitter volume from pitcher-model dependencies and slugifies dependency paths
> consistently. Its first residue candidate was rejected before data access because construction queried
> the live probable-pitcher slate; frozen/full rebuild construction is now offline. Candidate B has
> completed `pitches thrown` at 7,116 rows / SHA `f1ed47e8...`: it is honestly classified as fully
> bookless/synthetic, has complete provenance, and passes the auditor with zero violations. Independent
> candidate C matches it exactly in frame contents, dtypes, column order, parquet bytes, SHA, and every
> stable manifest field (`as_of` alone differs), so the `pitches thrown` identity is accepted. Candidate B
> also completed `total bases` at 64,493 rows / SHA `e87573fd...`: all rows are coherent direct-book
> quotes with complete provenance and zero audit violations. Its first audit exposed only a one-ULP
> comparison error at the valid exact `5x line` EV boundary; the shared runaway guard now preserves the
> strict-greater-than contract with a regression pin. Candidate C is independently rebuilding both MLB
> cells. Both now match candidate B exactly in frame contents, dtypes, column order, parquet bytes, SHA,
> and every stable manifest field (`as_of` alone differs), so both MLB identities are accepted. The seven
> NHL matrices with legacy goalie leakage are running as partitioned skater-only builds from the accepted
> isolated `timeOnIce` dependency. `goals` is accepted after an exact A/B repeat at SHA `68e129f1...`.
> Candidate A is also clean for `points` (`2f6c988f...`, 79,861 rows), `faceOffWins` (`105a259e...`,
> 24,558), `hits` (`821ddcb6...`, 16,541), `assists` (`9d80fe49...`, 83,640), and
> `powerPlayPoints` (`73a8dc92...`, 83,622): all five have complete provenance, only skater position
> codes 1/2/3, and zero auditor violations. Independent B repeats for all five are auditor-clean and
> match A exactly in frame contents, dtypes, column order, parquet bytes, SHA, and all stable manifest
> fields, so those identities are accepted. The much denser `skater fantasy points underdog` candidate A
> has also completed cleanly at 14,906 rows / SHA `3075a0fc...`: it has complete provenance, only skater
> position codes 1/2/3, and zero auditor violations. A repeat of that seventh skater market was stopped
> at 221/558 before output after re-reading the written acceptance: Stage 3 requires the complete
> fail-closed position policy, active-cell audit, and representative clean rebuild evidence, not an exact
> repeat of every historically contaminated NHL cell. With three independently reproduced goalie
> markets and six independently reproduced skater markets, Stage 3 is complete. Stage 4 is also
> complete: affine schema 2 persists its exact expected code set, train-scale-aware bounds,
> model-only unseen-position fallback, explicit quote authenticity, and five outer-fold support
> audits. Only eligible authentic rows fit/use the book pool; non-authentic and unseen-position rows
> stay model-only. Position arrays are integer-validated before code discovery, declared `role_only`
> is typed, and unsupported config combinations fail before fit. Legacy affine schema 1 retains its
> fixed QB/RB maps/bounds and original application semantics; receiving-v3 remains serve-exact.
> The focused structural regression sweep passes 88 tests and full source lint is clean. Stage 5
> quarantined candidate A/B reconstruction is complete for the four named NFL markets. Each B parquet
> is byte-identical to A and each manifest differs only in expected `as_of`: passing yards
> `dba0b6f2...` / 2,420 rows, passing TDs `f3b929c9...` / 2,467, receiving yards
> `9f3f3f8f...` / 14,953, and rushing yards `7b9f7bbb...` / 6,573. Both roots pass the
> auditor with zero violations. The owner then authorized the four isolated frozen-recipe HPO
> retrains. Passing yards and passing TDs completed and were independently scored before a host power
> outage cleared the `/tmp` quarantine; receiving yards was interrupted at trial 28, while the initial
> concurrent rushing-yards process had already stopped on a native allocator failure before trial 1.
> Passing yards honestly failed Gate 1; passing TDs passed all six gates. Exact hashes and scorecard
> evidence remain in the ledger, but the lost artifacts are not being treated as available. Recovery is
> complete in a persistent workspace quarantine: the independently repeated current-workflow matrices
> are accepted at passing yards `c81e3337...`, passing TDs `0bda0f41...`, receiving yards
> `f59954c4...`, and rushing yards `2e055464...`. The installed LightGBM binary and relevant
> LightGBMLSS modules match their cached wheels byte-for-byte. Rushing yards passes the generic and
> selected-feature audits, its exact frozen matrix completes the original deterministic 30-round fit,
> and the unresolved native abort is isolated to the original full four-fold HPO/CV path rather than
> matrix construction or the base fit. The first identity-stable passing-yards recovery completed
> 102 trials (best objective `0.876253`) but honestly failed Gate 1 at `+0.0212`
> `[+0.0092,+0.0346]`. Diagnosis found that the scorecard and generic fusion path still treated
> synthetic neutral probabilities as bookmaker evidence: the 363 authentic rows were only `+0.0024`,
> while 15 synthetic rows contributed `+0.4754` each on average. Generic fusion now fits/applies
> book influence only on explicitly authentic rows, non-authentic rows remain model-only, test
> artifacts persist authenticity, and Gate 1 uses only authentic book evidence. The focused repair
> suite passes 172 tests. Its exact-recipe retry completed 122 trials (best `0.886175`) and reduced
> Gate 1 to `+0.0030` `[-0.0040,+0.0102]`; Gates 2–5 pass, but the authentic-only upper bound still
> exceeds the owner threshold, so this remains an honest KILL rather than permission to tune.
> Passing TDs then completed the unchanged 300-trial DPO/`roe_mean` recipe (best trial 211,
> objective `1.46732`) and independently re-earned SHIP: Gate 1 is `-0.0503`
> `[-0.0689,-0.0314]`, with all other gates passing. The isolated artifact is bound to the accepted
> `0bda0f41...` matrix and `volume-v1`; all 22 synthetic test rows are model-only. Receiving yards
> then completed 106 trials under its unchanged one-hour receiving-v3 recipe (best trial 32,
> `0.655224`). All 12 nested structural guards pass, including the formerly failing
> position-positive and CITL guards, and independent held-out scoring is SHIP with Gate 1 `-0.0001`
> `[-0.0023,+0.0022]`. The final unchanged rushing affine attempt again aborted in packaged four-fold
> CV before trial 1. An operational one-worker/two-arena containment also ended in a kernel-recorded
> native SIGSEGV before trial 1; neither attempt emitted an artifact. With the accepted matrix
> byte-reproducible/auditor-clean and its direct LightGBMLSS fit already proven, rushing remains an
> honest execution KILL under the owner's no-LightGBM/no-packaged-code boundary. Stage 5 is therefore
> closed with two SHIPs, one gate KILL, and one execution KILL; failure evidence has not been relabeled
> or used to weaken a recipe. Canonical matrices, models, test sets, calibration files, and archive data
> remain untouched.

## 1. Mission & money logic

Make training matrices deterministic, provenance-bearing build products, then repair the grouped-CDF
and NHL position contracts that would corrupt a fresh retrain. Success means a cold rebuild on
canonical inputs can independently re-earn all six gates for NFL passing yards, passing TDs,
receiving yards, and rushing yards—not an unsupported claim of byte-identical legacy recovery.

This lane protects model evidence from serving-state changes, archive migration, synthetic book
probabilities, random trimming, and position contamination. Another agent owns the broad
`model_strategy` board/TPE plan; this lane does not compete with it.

## 2. Read first

1. `CLAUDE.md`, `CONTRIBUTING.md`, and `STYLE_GUIDE.md`.
2. `docs/handoffs/model_improvement_track.md` near the popular-market evidence and structured-integer
   owner block (currently near lines 1051 and 1191).
3. `docs/ship_gate.md`; gate semantics and thresholds are owner-owned.
4. `stats/base.py`, `training/pipeline.py`, and `training/data.py`.
5. `helpers/archive.py` and `scripts/inject_backfilled_odds.py`.
6. `training/group_conditional_cdf/`, `training/structural_context.py`, and `training/role_specs.py`.
7. `stats/nfl.py` and `stats/nhl.py`.

## 3. Verify before you trust

If output contradicts this brief, output wins. Fix minor drift here; stop on a material contract change.

```bash
git status --short
git log --oneline -8

# Current supporting-model placement.
find src/sportstradamus/data/models -maxdepth 1 -type f -printf '%f\n' \
  | sort | rg '^NFL_(attempts|carries|targets)\.mdl$' || true
find src/sportstradamus/data/old_models -maxdepth 1 -type f -printf '%f\n' \
  | sort | rg '^NFL_(attempts|carries|targets)\.mdl$' || true

# Matrix/provenance and NHL position inventory.
poetry run python - <<'PY'
from pathlib import Path
import pandas as pd
root = Path("src/sportstradamus/data/training_data")
active, synth = [], []
for p in sorted(root.glob("*.parquet")):
    try:
        f = pd.read_parquet(p)
    except Exception:
        continue
    if {"Player","Date","Result","Line","Odds","EV","Archived"} <= set(f):
        active.append(p.name)
        synth += ["Odds_synthetic" in f]
print("active", len(active), "with_synthetic_provenance", sum(synth))
for slug in ("goalsAgainst","saves","shotsAgainst","goalie-fantasy-points-underdog"):
    p = root / f"NHL_{slug}.parquet"
    if p.exists():
        f = pd.read_parquet(p)
        print(slug, len(f), f["Player position"].value_counts().sort_index().to_dict(),
              "archived", int(f["Archived"].sum()))
PY

poetry run pytest -q tests/golden/test_two_part_groupcdf.py \
  tests/golden/test_affine_experiment.py \
  tests/golden/test_structural_conditional_gate_serving.py \
  tests/golden/test_integer_distribution.py
```

Audit snapshot, 2026-07-22 (re-derive; never hard-code): 93 active matrices, 11 with
`Odds_synthetic`; carries/targets absent from the serving model root; `NHL_goalsAgainst` had
12,635 rows across all four position codes while all 1,590 archived rows were goalies. A scratch NFL
build failed on its first 2021 gameday because the attempts dependency requires external columns whose
local history begins in 2022. Re-derived Stage 0 found no duplicate `(Player, Date)` identities,
six nonfinite/invalid-archived rows in one cell, no explicit quote-source columns in any active matrix,
and legacy compatibility counts of 1,221,593 authentic, 7 explicitly synthetic, and 829,845 bookless
rows (missing `Odds_synthetic` remains unclassified, so those counts are not evidence of authenticity).

## 4. Locked decisions

- Do not edit the external board plan or broad
  `training/model_strategy/{specs,registry,sweep,confirm}.py`; document/coordinate any needed
  target-lattice interface.
- Never overwrite current matrices while developing. Build into quarantine and compare first.
- Never delete/move model artifacts merely to make resolution work. Served/withheld state is not a
  feature-dependency registry.
- Preserve receiving-v3 and affine-v1 readers. Corrected affine behavior needs a new
  implementation/schema identity; an old blob must retain old meaning.
- Preserve existing NFL yardage settlement under its existing identity. Do not enroll it on
  low-count/integer cross-sport cells. Exact integer transfer remains owner-blocked pending a push
  settlement choice.
- Never weaken gates, support floors, artifact parity, or tolerances.
- Exact legacy reproduction may be claimed only when every raw input and dependency is identified and
  hashed. Otherwise establish and re-earn a reproducible new baseline.

## 5. Module footprint

Primary scope is `stats/{base,nfl,nhl}.py`, `helpers/archive.py`,
`training/{pipeline,data,structural_context,structural_strategies}.py`,
`scripts/inject_backfilled_odds.py`, `training/group_conditional_cdf/`, a small manifest/audit
module, and focused tests. Touch `prediction/model_prob.py` only for old/new structural artifact
parity and follow the serving discipline in `model_improvement_track.md`.

## 6. Stage plan

### Stage 0 — Pin failures and add a read-only auditor

- Report duplicate identity, finite/range failures, target lattice, positions,
  authentic/synthetic/bookless counts, archive coverage, and quote sources.
- Add tests proving dependency resolution cannot vary with serving-model pruning.
- Add archive fixtures requiring line/probability cohort coherence.
- Add affine tests for non-NFL codes, unseen/fallback codes, and mixed authentic/synthetic rows; every
  output must be initialized and finite.
- Add complete NHL goalie/skater market-filter tests.
- Stop if a policy choice would require looking at heldout outcomes.

**Acceptance:** Confirmed defects have failing characterization tests; the existing focused baseline
still passes independently.

### Stage 1 — Deterministic lineage and stable feature dependencies

Current blocker: the serving `attempts` artifact and all three `old_models` candidates lack
`model_version`, `trained_at`, `dependency_identity`, and a training cutoff. The new preflight refuses
them. Owner selected newly trained `volume-v1` dependencies; the three raw matrices now validate in
quarantine and isolated full-HPO training is in progress. Stages 2–5 remain unstarted until the
dependency namespace passes validation.

- Add a full-rebuild mode distinct from `--force`; it must not read/append the cached matrix and must
  accept a quarantine output path.
- Resolve attempts/carries/targets from a versioned dependency namespace/manifest, never the prunable
  serving root. Inspect `old_models`, but validate identity, distribution, schema, and cutoff before
  considering migration.
- Preflight dependencies and expected columns before looping gamedays; one actionable report replaces
  a late KeyError.
- Reconstruct historical season/week locally from cached gamelog/schedule data; network collection is
  not a rebuild dependency.
- Freeze a pre-snapshot feature policy. Do not blindly zero missing expected columns; optional external
  families need availability indicators or a model trained for that missingness regime.
- Give trimming a local seed and stable ordering.
- Emit a manifest with builder/schema version, code revision, as-of, relevant input/query hashes,
  external snapshot inventory, dependency/config hashes, seed, row count, feature schema, and matrix SHA.

**Acceptance:** Two builds from identical frozen inputs have identical rows and matrix SHA. If
authoritative carries/targets artifacts are unavailable, finish infrastructure and ask the owner to
choose legacy recovery versus a newly re-earned feature baseline.

### Stage 2 — Canonical quote resolution and honest provenance

- Query latest rows per book once; derive line, under probability, EV, timestamps, and book count from
  a coherent cohort. A probability must correspond to the selected line. Freeze the consensus-line
  policy before model outcomes are inspected.
- Persist quote source, authenticity, synthetic reason, observed-at, and book count. `Archived`
  becomes a compatibility projection, not “positive line existed.”
- Stored direct probability can be authentic. EV inversion, model fallback, and 0.5 are derived or
  synthetic and stay distinguishable.
- Fix `_step_synthesize_odds`: Odds derived from EV are synthetic.
- Make repair, append, and scratch paths call one pure resolver; remove duplicate inversion logic.
- Never blanket-fill provenance nulls with zero.

**Acceptance:** Fixture-based repair/append/scratch book fields match exactly; structural pools receive
only explicitly authentic rows.

### Stage 3 — NHL/archive cleanup and quarantined rebuilds

- Add one complete NHL position map. At minimum shotsAgainst, saves, goalsAgainst, and goalie-fantasy
  markets are goalie-only; skater markets exclude goalies.
- Audit every active cell. Prioritize NFL `tds`, MLB legacy EV/shape-free residue, and matrices missing
  synthetic provenance.
- Classify authentic-supported, partial, and bookless cells. Bookless cells may model outcomes but may
  not advertise 0.5 as bookmaker evidence or fit a book pool.
- Quarantine rebuilt cells and report old/new identity, raw-outcome, feature, position, quote-source,
  synthetic-rate, and coverage differences.

**Acceptance:** No goalie-only row has a non-goalie position; invalid archive values cannot enter a
matrix; every row has a source classification; the auditor exits nonzero on violations.

### Stage 4 — Safe generalized grouped-CDF

- New affine schema persists the expected code set from context through fit/apply/validation.
- Implement declared model-only `pooled_fallback`, or fail before fit if fallback is unsupported.
  Never partially assign an `np.empty_like` result.
- Affine accepts explicit authenticity. Fit temperature/pool on eligible outer-training rows and apply
  the pool only to authentic rows; persist support/fold audits.
- Make new-schema affine bounds train-scale-aware while legacy bounds retain legacy meaning.
- Unsupported `StrategyConfig` combinations fail loudly; typing includes implemented `role_only`;
  validate positions before integer-code discovery.
- Add NBA/WNBA/NHL code-shape, integer rejection, unseen fallback, and legacy affine-v1/receiving-v3
  round-trip tests.

**Acceptance:** No generalized path depends on NFL position literals; legacy decode is identical; new
artifacts identify groups/protocol/provenance; synthetic books cannot affect a fitted/applied pool.

### Stage 5 — Rebuild and re-earn the four gates

1. Freeze the four recipes from `stat_meta.json` and the commands in `model_improvement_track.md`;
   this lane does not search alternatives.
2. Build quarantined matrices/manifests and complete row/feature/quote diffs before retraining.
3. Cold full-HPO retrain and independently score all six gates. Give passing yards extra attention
   because its prior Gate-1 margin was narrow.
4. Receiving/rushing use the corrected structural identity, never an old-version relabel.
5. Record matrix/artifact hashes and honest pass/fail. Promotion remains an owner decision.

**Acceptance:** Rebuild SHA is reproducible; artifact/CSV/scorecard/reload/serving parity passes; all six
gates pass for a first-ship claim. Failure is valid evidence, not permission to tune the same holdout.

## 7. Working rules and stop conditions

- Keep the 1.3 GB DuckDB read-only except in an explicitly backed-up migration.
- Prefer vectorized cohort queries; no network dependency in reconstruction.
- Preserve other-agent edits; inspect `git diff` before each patch.
- Raw observed quotes remain separate from derived features. Applicability never enrolls a strategy.
- Stop before promotion/overwrite, choosing integer settlement, treating `old_models` as authoritative
  without lineage, changing old blob meaning, editing board-search scope, weakening a guard, or adding a
  paid/network rebuild dependency.
- Conflict order: current output > repo law > owner gate/handoff docs > this brief.

## 8. Session definition of done

- Review every touched `.py`; run focused tests, `poetry run ruff check src/sportstradamus/`, then
  `poetry run pytest tests/golden/`.
- Run `poetry run pytest -m integration -n0` when integration behavior is reached.
- No canonical data/artifact mutation without separate authorization.
- Update status and append one ledger line. Never push `devel` directly.

## 9. Ledger (append-only, newest first)

- 2026-07-25 · Session plan closed, protected state verified, and recovery residue removed ·
  all 255 protected canonical files match their pre-training SHA-256 and size/mtime/mode
  fingerprints exactly. Full source Ruff and `git diff --check` pass; integration is 30/30; the two
  edited non-golden characterization files are 10/10; final golden is 3,831 passed / 1 expected
  xfail / 1 unchanged runtime failure in `test_ship_gate_invariant` from the local
  `model_stats.parquet`. The task-owned duplicate-code failure was consolidated/rationale-scoped
  and its gate passes. Touched source was reviewed, and 205 MB / 97 recovery-only quarantine files
  were deleted while retaining the workflow regression tests. No board-sweep plan, packaged
  LightGBM/LightGBMLSS code, canonical matrix/model/test/calibration file, or archive data changed ·
  next: commit this scoped working tree; no promotion or push
- 2026-07-25 · Final unchanged rushing attempt retained as an execution KILL, not matrix regression ·
  the exact frozen SkewNormal/ratio/CRPS/NLL affine recipe against accepted SHA `2e055464...` again
  aborted in packaged four-fold CV before trial 1 with `malloc_consolidate(): invalid chunk size`.
  A source-free operational containment (`OMP_THREAD_LIMIT=1`, two malloc arenas) also failed before
  trial 1; the kernel recorded a native SIGSEGV in libc. Neither attempt emitted a model or test CSV.
  The matrix remains byte-reproducible and auditor-clean, its selected 207-feature frame is finite,
  the installed LightGBM/LightGBMLSS files match their cached wheels, and the direct deterministic
  LightGBMLSS fit on this exact matrix completes. No LightGBM, LightGBMLSS, search-space, packaged
  code, or canonical data was changed. Under the owner boundary, further native-CV modification is
  out of scope, so Stage 5 closes with this honest execution KILL rather than a false first-ship
  claim · next: final repository validation, canonical fingerprint comparison, cleanup, and commit
- 2026-07-25 · Corrected-identity receiving-v3 retry re-earned structural and held-out acceptance ·
  the unchanged frozen SkewNormal/ratio/CRPS/NLL receiving-v3 search completed 106 trials in its
  one-hour budget (best trial 32, objective `0.655224`) and emitted isolated artifact SHA
  `aee0d007...`, test CSV SHA `c84a44d...`, and scorecard SHA `14a2a6f4...`. The artifact has the
  expected 427-feature schema, exact `volume-v1` binding to accepted matrix SHA `f59954c4...`, and
  active structural schema 3. All 12 nested validation guards pass, including
  `gate4_position_positive` and `gate6_citl`; validation Gate 1 is `-0.00221`, global PIT-KS
  `0.0341`, and CITL ratio CI `[0.8818,0.9927]`. Independent scoring on 1,846 authentic held-out
  quotes reports Gate 1 `-0.0001` `[-0.0023,+0.0022]`; all remaining gates pass and the verdict is
  SHIP. All 407 synthetic rows are model-only to floating-point precision · next: run the unchanged
  rushing affine recipe against its accepted matrix, then diagnose/retry only if it regresses
- 2026-07-25 · Authentic-only passing-TD retry re-earned all six gates · the unchanged frozen
  DPO/`roe_mean` recipe completed all 300 trials in 44:17 (best trial 211, objective `1.46732`) and
  emitted isolated artifact SHA `4202bbba...`, test CSV SHA `2e830d65...`, and scorecard SHA
  `a2fb4382...`. The artifact has the expected 435-feature schema, `volume-v1` dependency identity,
  and exact binding to accepted matrix SHA `0bda0f41...`. Its 22 synthetic test rows are model-only
  with zero fused-minus-model mean error. Independent scoring on 366 authentic quotes reports
  Gate 1 `-0.0503` `[-0.0689,-0.0314]`; all remaining gates pass and the verdict is SHIP. No
  canonical or packaged-code write occurred · next: run receiving yards under its unchanged
  receiving-v3 structural recipe and diagnose before retry if its structural audit still fails
- 2026-07-25 · Authentic-only passing-yards retry removed false evidence but still failed honestly ·
  the unchanged frozen recipe completed 122 trials (best trial 72, objective `0.886175`) and emitted
  isolated artifact SHA `dd0efced...` plus test CSV SHA `f63b197d...`, both bound to matrix
  `c81e3337...`. All 15 non-authentic test rows use their model mean exactly (maximum fused-minus-
  model mean error `5.7e-14`), and Gate 1 now scores only 363 explicit authentic quotes. The
  independent result is Gate 1 `+0.0030` `[-0.0040,+0.0102]`; Gates 2–5 pass, but the CI upper bound
  exceeds the `+0.005` owner threshold, so the verdict remains KILL. The remaining gap is genuine
  authentic evidence with the blend already at its frozen minimum model weight `0.05`; the plan
  forbids using this result to search another recipe · next: run and score passing TDs under the
  shared provenance repair, then diagnose any regression before the structural markets
- 2026-07-25 · Passing-yards Gate-1 regression localized to synthetic rows posing as book evidence ·
  the identity-stable frozen retry completed 102 trials (best trial 23, objective `0.876253`) and
  emitted isolated artifact SHA `1a2055fe...`, then honestly failed Gate 1 at `+0.0212`
  `[+0.0092,+0.0346]` while Gates 2–5 passed. Identity-aligned decomposition showed authentic
  quotes at only `+0.00241` fused-minus-book Brier; 15 synthetic neutral-price rows averaged
  `+0.47544` and created the rejection. This violated the Stage 2 rule that synthetic/bookless
  probabilities are not bookmaker evidence. Generic fusion now fits on authentic validation quotes
  only and applies model-only fallback to every derived/synthetic row; artifacts persist explicit
  authenticity and Gate 1 filters to it. The shared-path, persistence, and scorecard regression
  suite passes 172 tests with lint clean. The recipe/search space remains frozen and no LightGBM
  code or installed package changed · next: rerun passing yards on the same accepted matrix, then
  independently score all gates
- 2026-07-25 · Rushing-yards recovery audit found no matrix or installed-package corruption ·
  both accepted cold builds are byte-identical at SHA `2e055464...`; the manifest and generic auditor
  validate all 6,573 rows with no duplicate, nonfinite, quote, provenance, position, or runaway-EV
  violation. The actual 207-feature model frame is finite with stable dtypes and the same feature
  contract and position cohorts as canonical data. The installed LightGBM native library and relevant
  LightGBMLSS modules match their cached wheels byte-for-byte, and `pip check` is clean. Most
  importantly, the original deterministic 30-round LightGBMLSS fit completed on this exact accepted
  frozen matrix, with execution deliberately stopped immediately after fit and before prediction or
  persistence. No artifact or canonical write occurred. The outstanding failure is therefore confined
  to the untouched full four-fold HPO/CV execution, not demonstrated matrix corruption · next: resume
  the remaining original frozen Stage 5 recipes sequentially, starting with passing yards
- 2026-07-25 · Original pre-recovery rushing-yards path separated build success from HPO failure ·
  the untouched deterministic path completed its 30-round LightGBMLSS fit and reached the expected
  affine provenance guard on the legacy canonical matrix. The exact frozen full-HPO command against
  accepted matrix SHA `2e055464...` then reproduced the pre-trial native abort
  (`malloc_consolidate(): invalid chunk size`) with 35 GiB host memory available; no trial, artifact,
  test CSV, or canonical write occurred. Per Stage 5, this is retained as failure evidence and is not
  permission to edit LightGBM or search an alternative recipe · next: continue the remaining frozen
  Stage 5 retries sequentially, beginning with passing yards
- 2026-07-25 · Post-503 LightGBM experimentation fully rolled back at owner direction ·
  `training/hyperparams.py`, the original LightGBMLSS construction, HPO search settings,
  monotone constraints, cache/thread settings, and related tests are restored to the pre-recovery
  code; no installed package was modified. Fourteen focused matrix/lineage/trim/live-parity/holdout
  tests pass. A bounded deterministic rushing-yards smoke completed the original 30-round model fit
  without native failure, then correctly failed closed because the legacy canonical matrix lacks the
  explicit quote provenance required by affine calibration. No canonical artifact or matrix was
  written · next: resume Stage 5 exactly as specified with the accepted immutable quarantined matrix
  and the original frozen full-HPO recipe, one market at a time
- 2026-07-25 · Stage 5 repair cycle confirmed identity-unstable held-out assignment · calibration/
  test assignment now hashes immutable Player/Date identity instead of reshuffling shared rows when a
  matrix grows, and a golden pins insertion stability. Rushing-yards also reproducibly aborts inside
  LightGBMLSS CV before trial 1 through two operational defects: the 9,216 MB per-fold histogram
  cache exceeds the 39 GiB host's safe native-allocation envelope, and some sampled tree corners make
  LightGBM reject an empty right split. The exact long corner completes 997 rounds at objective
  `0.8180640340` with a 512 MB cache; CRPS CV is serialized to avoid allocator corruption, and
  `LightGBMError` now fails only its Optuna trial so the study continues. Parameter space, time budget,
  round limit, early stop, and objective remain frozen. No rushing artifact exists yet · next: rerun
  rushing under the repaired execution path, then score and retry every market affected by the shared
  split change
- 2026-07-25 · Current-workflow receiving-yards baseline failed closed before artifact persistence ·
  the frozen SkewNormal/ratio/receiving-v3 search completed 110 trials (best trial 85, objective
  `0.653185`) without allocator, lineage, or canonical-write error. During post-fit structural
  validation, `_two_part_candidate_oof_audit` rejected the candidate on
  `gate4_position_positive` and `gate6_citl`; no model or test CSV was emitted, and the failed log is
  retained. This is honest fail-closed evidence from the corrected structural identity, not permission
  to weaken its guards or relabel an old artifact · next: complete the rushing-yards baseline, then
  diagnose/repair this structural failure and the passing-yards Gate-1 regression at workflow level
- 2026-07-25 · Current-workflow passing-TD baseline re-earned all six gates · the frozen DPO/none/
  `roe_mean` study completed the full 300-trial cap (best trial 201, objective `1.46646`) and emitted
  artifact SHA `14c7ff5743925c3eae590df799754fb181c58ebc737c519fe152643430fc82d6` plus test CSV
  SHA `9056a874ceb769c8aca383bde343c041a9437c8bcd3b1a0c99f5a6352a8715ed`. It carries the
  expected 435-feature schema, `volume-v1` dependency identity, and exact binding to accepted matrix
  SHA `0bda0f41...`. Independent scoring on 370 untouched rows reports Gate-1 Brier delta `-0.0628`,
  95% CI `[-0.0809,-0.0439]`, and SHIP with no failed gate · next: complete receiving- and
  rushing-yards frozen baselines before repairing/retrying the passing-yards failure
- 2026-07-25 · Current-workflow passing-yards baseline HPO completed and honestly failed Gate 1 ·
  the isolated frozen study completed 111 trials (best trial 104, objective `0.888094`) and emitted
  artifact SHA `cebb299641dc3aac12da2ccca2338cbca38345179e36c9f9a37d7063a95d419e` plus test CSV
  SHA `18cf1695fc617e1dc81daa5f6696660df68ba3b6d13f5ad8e80b3c58c36f9d82`. The artifact has the
  expected SkewNormal/ratio recipe, 390-feature schema, `volume-v1` dependency identity, and exact
  binding to accepted matrix SHA `c81e3337...`. Independent scoring on 363 untouched rows reports
  fused-minus-book Brier `+0.0160`, 95% CI `[+0.0052,+0.0281]`; Gates 2–5 pass, but positive Gate 1
  makes the verdict KILL. This reproduces the pre-outage failure and is retained as baseline evidence,
  not promoted or tuned in place · next: complete the other three frozen baselines, then repair the
  identity-unstable holdout/feature contract at workflow level and retry every failing cell
- 2026-07-24 · Current-workflow Stage 5 matrix identities independently reproduced and accepted ·
  both cold roots are auditor-clean and match exactly in frame contents, dtypes, column order, parquet
  bytes, SHA, and every stable manifest field; `as_of` is the sole expected manifest difference.
  Accepted identities are passing yards 2,420 rows /
  `c81e3337d0707d00e1343046a47e00884f2002366fdbfd057429aa6e36310b0e`, passing TDs 2,467 /
  `0bda0f417207792761231b4c4e3c60a7fa38f257bbd735ebf79a9864ae71ea42`, receiving yards
  14,953 / `f59954c434ff48763c1cbd4e33af84cab8f59fd37930956a1796484a597e53fe`, and rushing yards
  6,573 / `2e05546470dc7d10e76cba1a1ad34a350c1b6642387347082b7c6df4dd108674`.
  All four have complete quote provenance and zero identity, numeric, quote, or position violations ·
  next: sequential isolated full-HPO with the four frozen recipes, independent six-gate scoring, and
  workflow-level diagnosis/repair plus fresh retry for any regression
- 2026-07-24 · Recovered Stage 5 passing matrices completed cleanly from the current `volume-v1`
  namespace · the persistent full rebuild produced passing yards at 2,420 rows / SHA
  `c81e3337d0707d00e1343046a47e00884f2002366fdbfd057429aa6e36310b0e` and passing TDs at
  2,467 / `0bda0f417207792761231b4c4e3c60a7fa38f257bbd735ebf79a9864ae71ea42`. Both are QB-only,
  carry complete quote provenance, and pass the auditor with zero duplicate, nonfinite, range,
  runaway-EV, authentic-quote, provenance, or position violations. Their row counts reproduce the
  pre-outage accepted builds while their new hashes honestly bind the re-earned current-workflow
  dependencies; receiving- and rushing-yards rebuilds are active in the same persistent quarantine ·
  next: finish, audit, and independently repeat all four current Stage 5 identities
- 2026-07-24 · Recovered current-workflow `volume-v1` namespace complete · targets completed its solo
  frozen study after 63 trials (best trial 21, objective `0.382812`) and persists artifact SHA
  `5bbc8209fae5f0ee73d7a45ff5102de78cff18e9ae06385218e95df028ccd1b1` plus isolated test CSV SHA
  `e89472a14cc32a9ca838e443ed950eeff558ff5ec2ea8245463c238f477dee67`. It passes the strict loader,
  expected 427-feature schema, timestamp/cutoff, SkewNormal/ratio recipe, and exact binding to current
  targets matrix SHA `f9dfcdf8...`. Attempts, carries, and targets are now all re-earned in persistent
  quarantine, and the protected 255-file canonical content/metadata fingerprints remain unchanged ·
  next: rebuild and exactly repeat the four Stage 5 matrices from this recovered dependency namespace
- 2026-07-24 · Targets dependency HPO restarted solo after second host failure · the crash occurred
  after attempts/carries had already finalized, so valid artifacts were preserved and not rerun.
  Targets restarted alone from accepted matrix SHA `f9dfcdf8...` with the frozen
  SkewNormal/ratio/CRPS/blend-NLL/loss/direct/no-stabilization recipe into persistent `volume-v1`.
  The study is healthy at 25 completed trials / 22m47s, best trial 21 with objective `0.382812`;
  no allocator, lineage, or canonical-write error is present · next: let the one-hour budget close,
  strictly validate the artifact, then rebuild the four Stage 5 matrices
- 2026-07-24 · Crash recovery found attempts/carries HPO complete and strictly valid · both concurrent
  studies had finalized before the host failed: attempts completed 89 trials (best 66, objective
  `0.325157`) and persists artifact SHA
  `2d102635dc93cf03694ce7d7b1b62bd6629ca6ad8c41e3740dfeb405bd43b1ef`; carries completed 39
  (best 32, `0.392236`) at SHA
  `f21e4fde45bd12517f03048258808e0b6fec0ab0e569d11e0feb236dcdc6aa41`. Both pass the strict
  `volume-v1` loader, expected 390/447 feature schemas, timestamps/cutoff, SkewNormal/ratio recipe,
  and exact bindings to the accepted current matrix SHAs. The protected 255-file canonical content and
  metadata fingerprints are unchanged. No valid work was rerun; targets will run alone to remove
  concurrent native-memory pressure · next: complete and validate targets `volume-v1` HPO
- 2026-07-24 · Current-workflow Stage 1 recovery matrices accepted after exact cold repeats · all
  three provenance-bearing 483-column matrices are manifest-valid and reproduce byte-for-byte, with
  `as_of` the sole manifest difference: attempts 2,401 rows / SHA
  `b0671c2205f8495546e5c2a0899709e753d619fd50841470ad24ddd5fd0d0002`, carries 6,444 /
  `2678e87d7e0667977e575afe417fbfb3b0e232790825d9dfe3bd9def5124c963`, and targets 14,144 /
  `f9dfcdf891181447dc1a41db8c9ce06c841fa3525ab57bda1da3dfc07c7f799d`. These supersede the lost
  pre-Stage-2 dependency inputs rather than weakening provenance to reproduce them. The accepted root,
  repeat root, logs, and 255-file canonical fingerprint are persistent under `.stage5-recovery` · next:
  isolated full-HPO re-earning of the three `volume-v1` dependencies
- 2026-07-24 · Recovery correctly rejected an obsolete Stage 1 identity · fresh current-workflow
  `attempts` and `carries` builds are manifest-valid but cannot reproduce the pre-Stage-2 accepted
  matrices: they contain 483 columns instead of 477 because the current pipeline adds the five explicit
  quote-provenance fields plus `Odds_synthetic`, and repaired quote eligibility changes the row sets to
  2,401 and 6,444 respectively. Both use the same recorded gamelog/config hashes and code revision, so
  this is expected stage evolution rather than outage corruption or nondeterminism. Recovery will not
  strip provenance or restore the earlier resolver merely to recover old hashes; it will independently
  reproduce and HPO-bind a new current-workflow volume baseline before rebuilding Stage 5 · next:
  complete targets and exact cold repeats for all three current dependency matrices
- 2026-07-24 · Power-outage recovery started; lost quarantine is not treated as evidence-in-hand ·
  repository edits and this handoff survived, but the host cleared all `/tmp` Stage 1/3/5 matrices,
  dependency artifacts, HPO artifacts, CSVs, scorecards, logs, and protected-tree fingerprint files.
  No persisted copy of the accepted volume dependencies or Stage 5 matrices was found elsewhere.
  Receiving yards had reached 28 completed trials (best trial 24, CV loss `0.653673`) when power was
  lost, so it produced no accepted artifact. Recovery will use a persistent workspace quarantine:
  rebuild the three volume dependencies and require their recorded matrix/artifact identities, then
  rebuild the four Stage 5 matrices and require their recorded SHAs before restarting only the missing
  or owner-requested retry work. The owner also authorized workflow-level diagnosis/fixes and fresh
  retries for any regressed market, followed by scoped cleanup and commit; canonical promotion remains
  out of scope · next: re-earn the lost dependency and Stage 5 inputs from cached raw data
- 2026-07-24 · Stage 5 passing diagnosis recorded without tuning · the accepted rebuild and prior
  passing-yards evidence have nearly identical temporal 30% tails (725 shared rows), but the positional
  seeded validation/test split is not identity-stable: only 197 test identities are shared, with 167
  former test rows moving to validation and 165 former validation rows moving to test. The rebuilt
  artifact also retains 390 features versus the prior artifact's 241 (149 newly active as-of fields:
  100 player, 28 defense, 21 team). On the 189 shared rows whose book line did not change, the new
  standalone model is still substantially worse than book (`+0.120445` Brier), so quote repair and the
  three corrected shared line rows are not the primary cause. The blend hit its minimum model weight
  (`0.05`) but could not rescue Gate 1. This is failure evidence, not authorization for feature
  ablation, split repair, tuning, or another passing retrain · next: finish the two structural-market
  runs, then preserve all four scorecards without promotion
- 2026-07-24 · Stage 5 passing pair complete; structural pair in progress · isolated passing yards
  completed 39 trials (best trial 21, CV loss `0.909283`) and fails Gate 1 on 363 untouched test rows:
  fused-minus-book Brier `+0.018691`, 95% CI `[+0.0068,+0.0312]`. Passing TDs completed 288 trials
  (best trial 191, CV loss `1.46597`) and passes Gate 1 on 370 rows at `-0.0630`, 95% CI
  `[-0.0811,-0.0439]`; the independent scorecard classifies it SHIP. Both signed artifacts validate
  against the accepted matrix SHAs, and the protected 255-file canonical content/metadata fingerprint
  is unchanged. Receiving yards is running its frozen one-hour recipe. The concurrent rushing-yards
  process failed in the native allocator before trial 1 and produced no artifact; its log is retained,
  and it will receive a fresh cold solo run after receiving releases memory · next: complete and score
  receiving yards, then rerun and score rushing yards alone
- 2026-07-24 · Stage 5 HPO authorized; frozen-run config isolation repaired before launch · owner
  granted fresh permission for the four frozen-recipe HPO retrains. Preflight then caught that
  `--artifact-output` redirected models/test CSVs but the distribution and book-shape stages still
  attempted to persist fitted `zi`, `cv`, and `book_shape` into canonical
  `stat_calibration.json`. Isolated runs now keep those fitted values in process memory and suppress
  all three canonical config writes. Two regression pins plus adjacent artifact/lineage coverage
  pass (27 tests); focused lint and `git diff --check` are clean. A content+metadata fingerprint of
  255 protected canonical matrices/models/test sets/config files was captured before training. The
  frozen passing pair is next, with a maximum of two simultaneous HPO jobs.

- 2026-07-23 · Stage 5 matrix identity accepted; paused before HPO permission boundary · candidate B
  completed auditor-clean and exactly reproduced all four candidate-A parquets byte-for-byte:
  passing yards `dba0b6f2...`, passing TDs `f3b929c9...`, receiving yards `9f3f3f8f...`, and
  rushing yards `7b9f7bbb...`. Rows, dtypes, column order, quote/provenance counts, position cohorts,
  and target lattices therefore match exactly; each manifest differs only in the expected `as_of`
  timestamp and every stable field is identical. The four immutable candidate-B matrices are ready
  for the frozen recipes. Per owner instruction, no HPO retrain has started; fresh permission is
  required to proceed. Canonical matrices, models, test sets, archive, and board-search files remain
  untouched.

- 2026-07-23 · Stage 5 four-market candidate A complete and auditor-clean · isolated cold rebuilds
  completed for NFL passing yards (2,420 rows, SHA `dba0b6f2...`, QB only, 63.3% authentic),
  passing TDs (2,467, `f3b929c9...`, QB only, 61.7%), receiving yards (14,953,
  `9f3f3f8f...`, WR/RB/TE only, 57.9%), and rushing yards (6,573, `7b9f7bbb...`, QB/RB only,
  58.8%). The root audit reports zero duplicate identities, nonfinite values, out-of-range odds,
  runaway EVs, invalid authentic rows, or missing provenance columns; all four targets have an
  integer lattice. Candidate B exact repeats are next. No HPO or canonical write has started.

- 2026-07-23 · Stage 5 candidate A active; empty historical gameday is now fail-safe · the first
  concurrent NFL rebuild exposed a real reconstruction gap on 2021-11-29: passing-market outcomes
  existed, but no as-of feature rows survived the snapshot/depth reconstruction, so the empty stats
  frame had no `Archived` quote column and both passing builds stopped. `get_training_matrix()` now
  treats this expected empty-day condition as zero contributed observations and continues before the
  quote/provenance join. A dedicated regression plus the adjacent lineage and quote-resolution suites
  pass (22 tests), and focused lint is clean. The failed runs wrote no matrix artifacts and their logs
  are retained in quarantine. Passing yards and passing TDs restarted with the fix; receiving and
  rushing continued unaffected. No canonical data was read as an append base or written.

- 2026-07-23 · Stage 4 complete; safe generalized grouped-CDF accepted · affine schema 2 now persists
  the expected position-code set from structural context through fit, validation, serialization, and
  apply; integer validation happens before discovery; NBA/WNBA/NHL-style code shapes are regression
  pinned. Unseen positions use a declared raw-CDF/unpooled model-only fallback, eliminating the
  partially initialized `np.empty_like` path. Fit/apply require explicit authenticity; the fitted book
  pool sees only eligible authentic outer-train rows and applies only to eligible authentic rows,
  while fold/code/authentic support is persisted in `fit_audit`. Schema-2 intercept bounds follow the
  validation target scale; schema-1 fixed bounds and QB/RB meaning remain unchanged. `role_only` is in
  `GroupingKind`, unsupported `StrategyConfig` combinations fail before dispatch, fractional codes
  fail before discovery, and legacy affine-v1 plus receiving-v3 round trips remain serve-exact.
  Focused final sweep: 88 passed; `ruff check src/sportstradamus/` and `git diff --check` clean · next:
  Stage 5 quarantined four-market matrix rebuild/repeat, then owner-authorized HPO.

- 2026-07-23 · Stage 3 complete; unnecessary seventh skater repeat stopped · re-reading the written
  acceptance showed that exact A/B reproduction of every historically contaminated NHL market was an
  over-scoped local gate, not a plan requirement. The active `skater fantasy points underdog` B repeat
  was stopped cleanly at 221/558 before it wrote an output; its completed clean candidate A remains
  quarantined at 14,906 rows / SHA `3075a0fc...`. Stage 3 already has substantially more than the
  representative evidence required: the complete fail-closed NHL position map and active-cell audit,
  three exact repeated goalie identities (`saves`, `goalsAgainst`, goalie fantasy points), and six exact
  repeated skater identities (`goals`, `points`, `faceOffWins`, `hits`, `assists`,
  `powerPlayPoints`), all auditor-clean with explicit provenance. Stage 3 acceptance is therefore met;
  Stage 4 is active. No canonical data was written.

- 2026-07-23 · Stage 3 five NHL skater identities accepted; fantasy candidate A clean/repeat B active ·
  independent B repeats for `points`, `faceOffWins`, `hits`, `assists`, and `powerPlayPoints` all pass
  the auditor and match A exactly in frame contents, dtypes, column order, parquet bytes, SHA, and every
  stable manifest field (`as_of` alone differs), accepting SHAs `2f6c988f...`, `105a259e...`,
  `821ddcb6...`, `9d80fe49...`, and `73a8dc92...`. The 558-slate
  `skater fantasy points underdog` candidate A then completed in 5h14m33s with 14,906 rows / SHA
  `3075a0fc...`. It has zero duplicate, nonfinite, invalid/runaway quote, provenance, or NHL-position
  violations; positions are only 1/2/3; sources are 453 direct-book authentic, 14,029 combo-EV derived,
  and 424 model-fallback synthetic. Against the legacy matrix it adds the six quote-provenance fields
  plus projected `timeOnIce` mean/std, removes all 102 invalid-position rows, preserves all 6,382
  overlapping raw results, adds 8,524 identities, and omits 9,798 legacy identities under the frozen
  rebuild boundary. Its independent B repeat is active from the identical isolated dependency root.
  Canonical matrices remain untouched.

- 2026-07-23 · Stage 3 NHL `goals` accepted; five more skater candidates clean · independent `goals`
  candidate B passes the auditor and matches A exactly in frame contents, dtypes, column order, parquet
  bytes, SHA `68e129f1...`, and all stable manifest fields (`as_of` alone differs). Candidate A also
  completed cleanly for `points` (79,861 rows / `2f6c988f...`), `faceOffWins` (24,558 /
  `105a259e...`), `hits` (16,541 / `821ddcb6...`), `assists` (83,640 / `9d80fe49...`), and
  `powerPlayPoints` (83,622 / `73a8dc92...`). Each has complete quote provenance, zero duplicate,
  nonfinite, invalid/runaway quote, provenance, or NHL-position violations, and only position codes
  1/2/3. Against the legacy matrices they add the six quote-provenance fields plus projected
  `timeOnIce` mean/std and remove all 757, 6,950, 110, 1,732, and 3,100 invalid-position rows,
  respectively. Their fresh B repeats are active; `skater fantasy points underdog` candidate A
  remains active. All outputs are quarantined and canonical data is untouched.

- 2026-07-23 · Stage 3 NHL downstream goalie set accepted · `goalie fantasy points underdog` candidate
  B passes the auditor with position G only and zero violations, and exactly matches A in frame contents,
  dtypes, column order, parquet bytes, SHA
  `82429686a66adc4466e538457d18b0afe039f092288b74b3807632685232a19a`, and all stable manifest
  fields (`as_of` alone differs). Together with the already accepted exact repeats for `saves` and
  `goalsAgainst`, all three downstream goalie matrices are closed; the accepted upstream `shotsAgainst`
  identity was already independently reproduced · next: finish the seven skater rebuild/repeat gates
- 2026-07-23 · Stage 3 NHL skater `goals` candidate A clean; repeat B active · the quarantined rebuild
  has 117,307 rows / 164 columns / SHA
  `68e129f13c6ed4695a00d62560292aa7003221c69afec25a0b6d098a9a8659c8`, complete provenance,
  no goalie positions, and zero duplicate, nonfinite, invalid/runaway quote, odds-range, provenance, or
  NHL-policy violations. Relative to canonical it retains 116,414 identities, adds 893, omits 4,936,
  removes all 3,965 legacy goalie rows, preserves the 0..4 integer target lattice, and adds the six quote
  provenance plus two current `timeOnIce` dependency fields. Authentic coverage is 96.12%; repeat B is
  running from the identical dependency root and recipe · next: require exact A/B reproduction before
  accepting this skater identity
- 2026-07-23 · Stage 3 corrected MLB `total bases` identity accepted; MLB priority closed · candidate C
  passes the auditor with zero violations and exactly matches B in frame contents, dtypes, column order,
  parquet bytes, SHA `e87573fde9f29f2dd6116769c06e6d35facbeff5f6b103af8c03bde67357a479`, and every
  stable manifest field; only `as_of` differs. The one-ULP strict-boundary repair did not alter the
  resolved 64,493-row matrix. Both prioritized MLB residue identities are independently reproduced and
  accepted · next: close the final goalie repeat and the seven partitioned skater rebuild/repeat gates
- 2026-07-23 · Stage 3 full active-cell audit re-derived after numeric-boundary repair · all 93 canonical
  cells were reread without mutation. The matrix-level inventory is unchanged: 15 authentic-supported,
  62 partial, 16 bookless, all 93 missing explicit provenance, 12 with invalid authentic quotes, 19 with
  runaway EV residue, 11 NHL cells with position leakage, and one with nonfinite core data. Row-level
  runaway accounting is now 12,929 overall; canonical MLB `total bases` has 10,865 true runaway rows,
  while 400 values previously counted there were only one-ULP representations of the valid exact `5x`
  boundary · next: retain the corrected strict-greater-than audit contract for all remaining rebuilds
- 2026-07-23 · Stage 3 NHL `goalsAgainst` identity accepted · independent candidate B passes the
  auditor with position G only and zero violations, and exactly matches A in frame contents, dtypes,
  column order, parquet bytes, SHA
  `aa7f6e09bcca7e270d60204845dae120bf21410511a9291c3ff079fd9fba11a5`, and every stable
  manifest field; only `as_of` differs. The repeat process has advanced to its final goalie fantasy cell
  · next: require the same exact-repeat gate for `goalie fantasy points underdog`
- 2026-07-23 · Stage 3 corrected MLB `pitches thrown` identity accepted · candidate C passes the auditor
  with zero violations and exactly matches B in frame contents, dtypes, column order, parquet bytes, SHA
  `f1ed47e8d80986da3d286334b70a4776e922b666d1553c89f82c912d4e5cacaa`, and every stable
  manifest field; only the expected `as_of` timestamp differs. The accepted identity remains 7,116 rows
  / 141 columns and honestly bookless/model-fallback on every row. Repeat C has advanced to `total bases`
  · next: require the same exact-repeat gate for `total bases`
- 2026-07-23 · Stage 3 contaminated NHL skater cleanup partitioned execution active · after accepting
  the corrected `NFL_tds` identity, a fresh quarantined build started for all seven remaining leakage cells: `assists`,
  `faceOffWins`, `goals`, `hits`, `points`, `powerPlayPoints`, and `skater fantasy points underdog`.
  Its serialized first cell projected at roughly two hours, so the process was stopped before writing
  any artifact; the same full recipes will be partitioned across the slots freed by the finishing repeat
  gates; six one-cell candidate-A partitions are active on the 16-core host. The much denser skater-fantasy
  partition was paused before writing at 8% because it materially throttled all six simpler cells; it remains
  queued for a low-contention full rebuild and repeat. Candidate A uses only the accepted isolated NHL volume dependencies; canonical matrices remain
  untouched · next: run, audit/diff, and independently reproduce every one of the seven cells
- 2026-07-23 · Stage 3 corrected `NFL_tds` identity accepted · independent candidate C passes the
  auditor with zero violations and exactly matches B in frame contents, dtypes, column order, parquet
  bytes, SHA `a4b17f4e8344f8bf4649829b7c875d23fee80c4c95b1db4a45b9ac7f7a75b4cd`, and every stable
  manifest field; only the expected `as_of` timestamp differs. The accepted identity remains 16,451 rows
  / 495 columns with 75.50% authentic coverage and a 0..4 integer target lattice · next: complete the
  MLB and NHL repeat gates, plus the seven contaminated NHL skater-cell rebuilds, before closing Stage 3
- 2026-07-23 · Stage 3 corrected MLB residue candidate B complete; repeat C active · `total bases`
  produced 64,493 rows / 144 columns / SHA
  `e87573fde9f29f2dd6116769c06e6d35facbeff5f6b103af8c03bde67357a479`, with 64,493 authentic
  same-line direct-book quotes, complete provenance, and zero duplicate, nonfinite, invalid/runaway,
  odds-range, or provenance violations. Relative to canonical it retains all 64,490 identities, adds
  three, preserves the 0..19 integer outcome lattice, and adds only the six provenance columns. The
  initial audit flagged 3,943 values that were exactly the allowed `5x line` ceiling but serialized one
  ULP high; the common strict `>5x` guard now tolerates only that numeric boundary (a `1e-9` excess still
  fails), with quote/auditor tests 16/16 and lint clean. Fresh repeat C is rebuilding both MLB cells from
  the same offline recipe · next: require exact B/C frame, dtype, column, parquet-byte, SHA, and
  stable-manifest agreement before accepting either MLB identity
- 2026-07-23 · Stage 3 NHL `saves` identity accepted · independent candidate B exactly matches A in
  frame contents, dtypes, column order, parquet bytes, SHA
  `e8f11ac034ada64579207135ebd7aca0dd3b6c50643fc99b45efa6596218ce28`, and every stable manifest
  field; only the expected `as_of` timestamp differs. The repeated three-cell process has advanced to
  `goalsAgainst` · next: require the same exact-repeat gate for `goalsAgainst` and goalie fantasy points
- 2026-07-23 · Stage 3 corrected `NFL_tds` candidate B complete; repeat C active · the full 314-gameday
  rebuild produced 16,451 rows / 495 columns / SHA
  `a4b17f4e8344f8bf4649829b7c875d23fee80c4c95b1db4a45b9ac7f7a75b4cd` and passes the auditor with
  zero duplicate, nonfinite, invalid/runaway quote, or provenance violations. It contains 12,420
  authentic direct-book rows and 4,031 explicitly synthetic/bookless rows (3,825 model fallback, 206
  neutral fallback), for 75.50% authentic coverage. Relative to canonical it retains 16,281 identities,
  adds 170, omits 162, preserves the 0..4 integer outcome lattice and eligible WR/RB/TE position shape,
  replaces six retired PBP-as-of columns, and adds current dependency/provenance features. Candidate C is
  running independently with the identical recipe and accepted volume root · next: require exact B/C
  frame, dtype, column, parquet-byte, SHA, and stable-manifest agreement before acceptance
- 2026-07-23 · Stage 3 corrected MLB `pitches thrown` candidate complete; `total bases` active · the
  334-gameday cold rebuild produced 7,116 rows / 141 columns / SHA
  `f1ed47e8d80986da3d286334b70a4776e922b666d1553c89f82c912d4e5cacaa`. All 7,116 rows explicitly
  report `model_fallback` / synthetic / zero books, so the previously bookless cell no longer advertises
  neutral probability as bookmaker evidence. It passes the auditor with zero duplicate, nonfinite,
  invalid/runaway quote, or provenance violations. Relative to canonical it retains all 6,940 old
  identities, adds 176, preserves the raw-outcome ceiling (new range 36..103 versus old 37..103), and
  adds only the six provenance columns. `total bases` is now building in the same quarantine candidate ·
  next: audit/diff `total bases`, then independently reproduce both MLB cells before acceptance
- 2026-07-23 · Stage 3 first MLB residue candidate rejected; frozen construction made offline · the
  `pitches thrown,total bases` process failed before loading cached data because `StatsMLB` constructed
  the live upcoming probable-pitcher map. Deterministic, frozen-matrix, and full-rebuild construction now
  skips that serving-only network state, while ordinary serving/update construction remains unchanged.
  MLB dependency preflight now correctly requires `pitches thrown` only for pitcher targets (hitter cells
  use structural plate appearances), and dependency filenames share the canonical market slugger so
  space-containing names resolve consistently. Focused loader/lineage contracts pass 16/16 and lint is
  clean; rejected path A is preserved and residue candidate B is active in fresh quarantine · next:
  complete and audit/diff `pitches thrown` and `total bases`, then independently reproduce them
- 2026-07-23 · Stage 3 first `NFL_tds` candidate rejected; historical dependency alignment repaired ·
  candidate A failed closed on its first 2021 slate because the accepted attempts/carries/targets models
  expect external snapshot features whose local inventory starts later. Inspection of accepted upstream
  matrices confirmed those absent pre-snapshot features were represented as zero during training (for
  example every 2021 attempts row has zero `Player age_asof`, `Team proe`, and `Defense def_man_pct`).
  Snapshot-only cold dependency inference now reindexes absent expected columns with the same zero value;
  live inference retains strict missing-column failure. Loader/lineage tests pass 14/14 and lint is clean;
  rejected path A is preserved, and candidate B is running from accepted isolated NFL dependencies · next:
  audit/diff candidate B, then independently reproduce it before acceptance
- 2026-07-23 · Stage 3 dependent NHL goalie candidate A complete; repeat B active · quarantined A
  passes the auditor for all three cells with zero violations: `saves` is 6,194 rows / 162 columns / SHA
  `e8f11ac034ada64579207135ebd7aca0dd3b6c50643fc99b45efa6596218ce28`, `goalsAgainst` is 5,794 /
  162 / `aa7f6e09bcca7e270d60204845dae120bf21410511a9291c3ff079fd9fba11a5`, and `goalie fantasy points
  underdog` is 5,802 / 162 / `82429686a66adc4466e538457d18b0afe039f092288b74b3807632685232a19a`.
  Every row is a goalie with complete provenance. Relative to canonical, the builds add the six explicit
  quote columns; retain 4,099 / 3,886 / 4,027 identities; add 2,095 / 1,908 / 1,775 newly covered goalie
  identities; and omit 10,799 / 8,749 / 10,889 old identities, including the old matrices' 10,231 /
  8,141 / 9,940 ineligible skater rows. New authentic coverage is 53.39% / 15.24% / 5.70%, and explicit
  non-authentic rates are 46.61% / 84.76% / 94.30%. Independent cold repeat B is running with the same
  immutable dependencies and recipe · next: require exact A/B frame, dtype, column, parquet-byte, SHA,
  and stable-manifest agreement before accepting the identities
- 2026-07-23 · Stage 3 dependent NHL goalie rebuild active · quarantined candidate A has completed
  `saves` (6,194 rows / 162 columns; 3,307 authentic direct-book and 2,887 explicitly synthetic rows)
  and `goalsAgainst` (5,794 rows / 162 columns; 883 authentic direct-book and 4,911 explicitly derived
  combo-EV rows). Both contain only position `G`, have complete provenance, and pass the matrix auditor
  with zero violations. The old matrices contained 10,799 and 8,749 identities absent from the new
  policy-clean builds respectively, predominantly ineligible skater rows. Candidate A remains provisional
  while `goalie fantasy points underdog` builds; acceptance requires a full audit/diff and independent
  cold repeat · next: finish candidate A, audit/diff all three cells, then reproduce the complete build
- 2026-07-23 · Stage 3 NHL volume dependency HPO complete and strictly validated · the frozen
  300-trial/60-minute recipe completed 22 `timeOnIce` trials (best observed objective `1.23536`) and 203
  `shotsAgainst` trials (best `3.20571`) before their per-market time caps. The isolated artifacts have
  SHA `fbffa9cd97f1311a3e9c06f90d748acf4fc1dbafe8ac020af31e73681c978cfd` (`timeOnIce`, 136 features,
  9,330 test rows) and `d23d52d6ca598da3ac0a15ef5a87f8c82faec6b424812473fc2447482aa784d3`
  (`shotsAgainst`, 133 features, 831 test rows); both pass the strict `volume-v1` loader and bind exactly
  to the accepted matrix SHAs. Canonical artifacts were not read as dependencies or overwritten · next:
  quarantined goalie-only downstream rebuilds with this dependency root, then matrix audit/diff
- 2026-07-23 · Stage 3 isolated NHL dependency HPO active · the full 300-trial study is running from
  immutable accepted matrix root `/tmp/sportstradamus-stage3-nhl-volume-c` into isolated artifact root
  `/tmp/sportstradamus-stage3-nhl-dependencies/volume-v1`; 11 `timeOnIce` trials have completed without
  failure and the current best objective is `1.23545`. No canonical matrix, model, or test CSV is an
  input/output target · next: finish both dependency studies and strictly validate artifact identity
- 2026-07-23 · Stage 3 NHL volume matrices accepted for isolated HPO · independent cold rebuilds B/C
  match exactly in frame contents, dtypes, column order, parquet bytes, and all stable manifest fields;
  only the expected `as_of` timestamps differ. Accepted `timeOnIce` is 62,202 rows / 162 columns / SHA
  `0193cc939f8dfbcb15c8cd4d774fc536091d1ebf893ba0ec916fce244dfb1c0f`; accepted `shotsAgainst` is
  5,543 rows / 162 columns / SHA `cb44cc0f61b4c637d7b3d9cc076b2606371d10fae42a7d1148e219dde0b02e28`.
  Repeat C passes the matrix auditor with zero duplicate, nonfinite, invalid/runaway quote, provenance,
  or NHL-position violations · next: authorized full-HPO retrain from immutable C into isolated
  `volume-v1`, then strict artifact/matrix identity validation
- 2026-07-22 · Stage 3 corrected NHL volume candidate B complete; repeat C active · quarantined
  `timeOnIce` is 62,202 rows / 162 columns / SHA `0193cc939f8dfbcb15c8cd4d774fc536091d1ebf893ba0ec916fce244dfb1c0f`
  with positions C/W/D/G all eligible for the shared market; `shotsAgainst` is 5,543 rows / 162 columns /
  SHA `cb44cc0f61b4c637d7b3d9cc076b2606371d10fae42a7d1148e219dde0b02e28` with position G only. Both
  have complete provenance, zero invalid/runaway quote or position violations, and honestly report
  bookless model fallback rather than authentic book support. A fresh cold repeat is running before
  these identities can be accepted or used for isolated HPO · next: require exact B/C reproducibility
- 2026-07-22 · Stage 3 first NHL volume candidate rejected · the raw 558-slate `timeOnIce` rebuild
  completed in quarantine, but its diff showed all 62,202 resolved non-authentic rows relabeled from
  their canonical resolver sources to `pipeline_ev_inversion` because `_step_synthesize_odds` treated
  `Odds_synthetic=True` as a request to invert a second time. The pipeline now synthesizes only missing
  or zero Odds and preserves already-resolved source/authenticity/reason/timestamp/book-count fields;
  focused pipeline/rebuild contracts 30 passed and lint clean. The rejected candidate is excluded from
  HPO · next: rebuild corrected `timeOnIce`/`shotsAgainst`, audit and repeat-check their identities
- 2026-07-22 · Stage 3 active-cell audit and ingress guards complete · all 93 active legacy matrices were
  read without mutation: 15 classify authentic-supported, 62 partial, and 16 bookless; every cell lacks
  the new explicit provenance, 12 cells contain invalid authentic rows, 19 contain runaway EV residue,
  and 11 NHL cells contain position-ineligible rows. `NFL_tds` is partial (12,677 authentic / 3,766
  bookless); MLB `pitches thrown` is bookless and `total bases` carries 11,265 runaway authentic rows.
  The pure resolver now excludes the existing `>2000 or >5x line` runaway class, the auditor reports it,
  and the complete active NHL registry is fail-closed (goalie-only, skater-only, or shared `timeOnIce`);
  focused Stage 3 contracts 25 passed / 2 expected xfails · next: quarantined NHL volume rebuilds and
  old/new matrix diffs
- 2026-07-22 · Stage 2 complete · coherent latest-per-book archive rows feed the frozen
  `modal-nearest-median-v1` same-line policy; matrices persist source, authenticity, reason, observation
  time, and book count without null sentinels; EV inversion is synthetic, repair/append/scratch parity is
  fixture-pinned for direct/derived/bookless rows, and structural pooling uses explicit authenticity only;
  focused quote/archive/structural suite 91 passed and lint clean · next: Stage 3 NHL position map,
  invalid-archive exclusion, full active-cell audit, and quarantined cell diffs
- 2026-07-22 · Stage 1 complete; corrected attempts dependency re-earned and full set validated · attempts
  best objective `0.319525`, artifact SHA
  `87788f5e104a7d851dc15cef9eb8ed94e7f69074191bde75560bd6d83df81ced`, accepted matrix SHA
  `1bfbe4bf241af724308713ddafd71be38f3a7ecf3abd88c7c7cafd22b6a6b6aa`, and 390-feature schema;
  attempts/carries/targets all pass strict identity, matrix binding, SkewNormal, model version/timestamp,
  and isolated CSV checks · next: Stage 2 canonical quote resolver and provenance parity
- 2026-07-22 · corrected carries dependency re-earned and validated · full-HPO best objective
  `0.391189`; artifact SHA `797ffc2011ffd605fac4d968fdc1019f4caab91c4bfb861a1c0fbdad61a17437`
  carries strict `volume-v1` identity, SkewNormal family, 447-feature schema, training timestamp, and
  accepted carries matrix SHA `c43479a08c03f5cb86795569d0430a09c6801dd773794cd61b25ee36abce4e16`;
  isolated test CSV is present · next: complete attempts full-HPO
- 2026-07-22 · corrected targets dependency re-earned and validated · full-HPO best objective
  `0.403399`; artifact SHA `65e53b89bf678cfbfc55fc8d2e69db578555d90fb4385a5f61cc126815ddac50`
  carries strict `volume-v1` identity, SkewNormal family, 427-feature schema, training timestamp, and
  accepted targets matrix SHA `8700a7309e18317f85291cfffb6835c038cd322ecb12e15ab60ad83f0aa14447`;
  isolated test CSV is present · next: complete carries full-HPO
- 2026-07-22 · owner authorized corrected Stage 1 HPO · accepted C matrices are the immutable inputs and
  `/tmp/sportstradamus-dependencies-fixed/volume-v1` is the isolated artifact destination; canonical
  matrices, models, and test sets remain out of scope · next: run and validate attempts/carries/targets
  full-HPO artifacts against the accepted matrix SHAs
- 2026-07-22 · Stage 1 matrices accepted; owner-paused before HPO · fresh concurrent read-only C/D
  rebuilds match exactly in rows, dtypes, column order, parquet bytes, stable manifest fields, and SHA:
  attempts `1bfbe4bf241af724308713ddafd71be38f3a7ecf3abd88c7c7cafd22b6a6b6aa` (2,463), carries
  `c43479a08c03f5cb86795569d0430a09c6801dd773794cd61b25ee36abce4e16` (6,627), targets
  `8700a7309e18317f85291cfffb6835c038cd322ecb12e15ab60ad83f0aa14447` (13,978), all 477 columns ·
  next: wait for explicit owner permission before corrected full-HPO `volume-v1` retraining
- 2026-07-22 · Stage 1 corrected A/B acceptance also failed honestly · targets matched exactly, while
  attempts had 391 `EV` and 303 `Odds` values and carries had 700 `EV` and 500 `Odds` values differ by
  only 1e-14–1e-16; latest-per-book archive rows now end in stable book order before weighted floating-
  point reduction and the ordering is regression-covered · next: rebuild a fresh independent pair,
  compare exact rows/bytes/SHA, then pause for owner permission before HPO
- 2026-07-22 · Stage 1 repeated-build acceptance failed honestly · an independent real rebuild found
  process-dependent team-feature ordering plus 1 targets, 10 carries, and 5 attempts depth values
  changed by tied-rank iteration order; stable ordering, canonical full-rebuild columns, and concurrent
  read-only archive connections are now regression-covered · next: finish corrected A/B rebuild,
  require exact rows/bytes/SHA, then retrain dependencies from the accepted matrix identity
- 2026-07-22 · Stage 1 frozen-input boundary tightened · dependency loading now requires training
  timestamp and source-matrix SHA; frozen-matrix training verifies adjacent manifest builder/schema,
  rows, feature schema, and parquet SHA; rebuild manifests hash every consumed FP snapshot file ·
  next: finish and validate attempts full-HPO artifact
- 2026-07-22 · Stage 1 dependency matrices rebuilt · quarantine has attempts 2,463 rows,
  carries 6,627, targets 13,978; each has 477 columns, no duplicate player/date keys or infinities,
  explicit unavailable public-enrichment state, and matching manifest row/schema/SHA fields ·
  next: isolated full-HPO `volume-v1` training
- 2026-07-22 · Stage 1 snapshot policy frozen · rebuilds use cached gamelog schedules, weekly
  FantasyPoints snapshots, and packaged player metadata; unfrozen PBP/NGS and live ID enrichment are
  disabled with an explicit unavailable feature · next: complete quarantined volume matrices
- 2026-07-22 · owner chose newly re-earned dependency baseline · building attempts/carries/targets
  from cached raw inputs into quarantine; legacy artifacts remain non-authoritative · next: train and
  validate `volume-v1`
- 2026-07-22 · blocked-state validation · ruff clean; focused 64 passed/7 expected xfails;
  integration 30 passed; full golden 3,778 passed/8 xfailed with one unrelated current-data ship-gate
  invariant failure after three matrix-repair regressions were fixed · next: owner dependency choice
- 2026-07-22 · Stage 1 infrastructure complete; owner-blocked · full rebuild is cache-free,
  quarantine-only, locally seeded, manifest-bearing, and `volume-v1`-only; real NFL preflight rejected
  all attempts/carries/targets candidates for missing lineage · next: owner chooses recovery vs re-earn
- 2026-07-22 · Stage 0 complete · read-only auditor added; 93/93 legacy matrices fail explicit quote
  provenance, one cell has six invalid archived rows; archive/affine/NHL defects pinned as strict xfails
  while the independent 50-test structural baseline stays green · next: Stage 1 infrastructure
- 2026-07-22 · Stage 0 audit reproduced · 93 active matrices, 11 with synthetic-provenance
  column; focused structural baseline 50 passed; goalie-market contamination and dependency-root
  coupling confirmed · next: land read-only auditor and characterization pins
- 2026-07-22 · handoff created · originating audit: 93 focused tests passed, no code/canonical data
  modified · next: stage 0 re-derive state and pin failing contracts
