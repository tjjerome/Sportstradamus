# NFL passing-yards Gate-1 improvement handoff

## Goal

Improve the current NFL passing-yards model until fresh, honestly earned evidence
passes all six held-out gates, without selecting changes against the retained
held-out outcomes, weakening a gate, or restoring the obsolete Wednesday
evaluation protocol.

Rushing yards is not part of this goal. Its diagnostic held-out scorecard passes
all six gates, although its official pipeline run still has a separate
pre-holdout affine probability-pool stability KILL.

## Workspace and durable evidence

- Repository: `/home/trevor/Sportstradamus`
- Branch: `feature/model-improvement`
- Stage 5 evidence:
  `src/sportstradamus/data/research/stage5-recovery-20260725-v1/`
- Passing diagnosis:
  `src/sportstradamus/data/research/stage5-recovery-20260725-v1/PASSING_GATE1_DIAGNOSIS.md`
- Exact identity bridge:
  `src/sportstradamus/data/research/stage5-recovery-20260725-v1/passing/fresh-gate1-bridge.json`
- Row-level identity ledger:
  `src/sportstradamus/data/research/stage5-recovery-20260725-v1/passing/fresh-identity-ledger.csv`
- Current model and test rows:
  `src/sportstradamus/data/research/stage5-recovery-20260725-v1/artifacts/passing-yards/`
- Current scorecard:
  `src/sportstradamus/data/research/stage5-recovery-20260725-v1/artifacts/passing-yards/NFL_passing-yards.scorecard.csv`
- HPO log:
  `src/sportstradamus/data/research/stage5-recovery-20260725-v1/logs/passing-yards-hpo.log`
- Accepted reusable matrix:
  `src/sportstradamus/data/training_data/NFL_passing-yards.parquet`

Do not rebuild the passing, attempts, carries, or targets matrices unless a hash
or manifest integrity check fails.

## Current accepted input and frozen comparison

- Matrix rows: 2,420
- Matrix columns: 493
- Matrix SHA-256:
  `79703d1f365acdb091c303260c8673d09ed1d3201346ffc8b327b54e28795a4c`
- Matrix A/B rebuilds were byte-identical, manifest-valid, and auditor-clean.
- Quote cohorts: 1,531 authentic and 889 synthetic matrix rows.
- Frozen comparison recipe:
  - distribution: `SkewNormal`
  - target normalization: `ratio_meanyr`
  - distribution loss: `crps`
  - blend loss: `nll`
  - HPO selection: `loss`
  - SkewNormal parameterization: `direct`
  - stabilization: none
  - posthoc: none

The frozen comparison completed 68 trials in 1:00:42. Trial 44 was best at
objective `0.889819`.

## Current held-out result

- Test rows: 378
- Authentic Gate-1 rows: 363
- Synthetic/model-only rows: 15
- Gate 1 fused-minus-book Brier: `+0.0003`
- Gate-1 row CI: `[-0.0067,+0.0074]`
- Gate-1 player-clustered CI-high: `+0.0085`
- Gate 2: pass
- Gate 3: pass
- Gate 4: pass
- Gate 5: pass
- Gate 6: pass
- Verdict: **KILL on Gate 1 only**

Artifact bindings:

- Model SHA-256:
  `5fa2c6aae52142416e573b6703ef4c6efeff1923c2b2d896143f07f00136a0a8`
- Test CSV SHA-256:
  `197424599d393516a274cbbe1213e36929ce5cd3f1e7a0697c095c5c3ee6d33a`
- Scorecard SHA-256:
  `daab373e26a657350923dce1a24d8f995006036f8b3674aa8b118ebe0efc2267`

## Why Wednesday passed

The retained Wednesday scorecard reported Gate 1 `-0.0010`, row CI
`[-0.0051,+0.0032]`, and clustered CI-high `+0.0033`.

That PASS does not survive the corrected evidence protocol. Rescoring the
unchanged Wednesday predictions gives:

| Evidence protocol | Rows | Gate 1 | Clustered CI-high |
| --- | ---: | ---: | ---: |
| Legacy positional holdout and all finite Odds | 369 | `-0.0010` | `+0.0033` |
| Stable Player+Date identity membership | 192 | `+0.0017` | `+0.0067` |
| Stable identity and authentic quotes only | 184 | `+0.0028` | `+0.0077` |

The old PASS was therefore mostly an evaluation-membership artifact, not a
better model that was subsequently lost.

The exact historical-to-current point-estimate bridge is:

| Component | Gate-1 change |
| --- | ---: |
| Identity cohort | `-0.0011532` |
| Outcome and line | `0` |
| Book probability and eligibility | `+0.0029494` |
| Standalone model probability | `+0.0006755` |
| Fusion and calibration | `-0.0011570` |

The contributions sum to the observed `+0.0013147` change with a
`2.17e-19` remainder. Within the bookmaker component, authentic eligibility is
`+0.0026451`; actual quote repricing is only `+0.0003043`.

Conclusion: stable identity assignment removed a favorable legacy cohort, and
synthetic neutral probabilities are no longer credited as bookmaker evidence.
Those are correct workflow changes and must not be reversed.

## Workflow repair already completed

The scorecard CSV loader was dropping persisted `QuoteAuthenticity` even though
the in-memory Gate-1 scorer filtered to authentic evidence. It now preserves
that column, so CLI scoring reproduces the exact bridge endpoint. Do not undo
this repair or compare against a scorecard that includes synthetic quotes as
bookmaker evidence.

## Productive improvement target

The problem is narrow: improve authentic-row incremental over/under probability
quality versus an efficient bookmaker while preserving the already-passing
distribution and segment gates.

Use training and validation evidence to investigate:

- whether the generalized strategy developed after the frozen comparison is
  appropriate for passing yards;
- validation-only standalone probability quality by player, temporal, line,
  usage, and quote-support cohorts;
- calibration/fusion behavior when the model has real incremental signal;
- whether the 149 added as-of, matchup, team, and dependency features contribute
  stable validation lift or only noise;
- model capacity, feature regularization, and probability calibration selected
  solely through training/CV and validation;
- robustness across deterministic identity-derived folds or temporal
  validation windows before spending another held-out scorecard.

The historical artifact had 241 model features. The current model retains all
241 and adds 149 current features, for 390 selected features. This is a useful
validation-side ablation boundary, not permission to select a subset using the
held-out result.

## Hard boundaries

- Do not tune against either retained held-out test set or its gate result.
- Do not weaken the `+0.005` Gate-1 CI-high threshold.
- Do not restore positional random splitting.
- Do not treat derived or synthetic quotes as authentic bookmaker evidence.
- Do not alter settled outcomes, lines, identity membership, or quote labels.
- Do not modify installed LightGBM or LightGBMLSS code.
- Do not overwrite canonical models, test sets, calibration files, archives, or
  the board-sweep plan without explicit authorization.
- Do not relabel the Wednesday artifact as current evidence.
- Do not rebuild accepted matrices without a demonstrated integrity failure.
- Keep experiments isolated until a strategy is preregistered from
  training/validation evidence.
- Per owner direction, do not add regression tests as a substitute for doing the
  model research correctly.

## Recommended execution sequence

1. Verify the accepted matrix, current artifact, scorecard, and bridge hashes.
2. Inspect training/CV and validation-only probability residuals; do not open
   the held-out outcomes for selection.
3. Preregister a small, causal set of strategy/feature/calibration comparisons.
4. Evaluate them using deterministic identity-derived or temporal
   training/validation folds.
5. Select one candidate without consulting held-out performance.
6. Run one fresh full HPO/retrain through the normal `meditate` pipeline.
7. Persist model, test rows, scorecard, recipe, logs, hashes, and environment
   identity under `src/sportstradamus/data/research/`.
8. Score honestly. Promote nothing automatically.

## Required next-agent report

- Exact strategy and why training/validation evidence selected it.
- Matrix, dependency, model, test, and scorecard hashes.
- Validation performance and stability across cohorts/folds.
- Fresh held-out values for all six gates.
- Clear SHIP/KILL verdict without threshold reinterpretation.
- A statement confirming that no retained held-out result influenced selection.
