# Plan: Mitigate GBDT Regression-Toward-the-Mean in the Training Pipeline

> **Multi-session handoff doc.** Status below is updated each session. The
> attached source report lives in the originating session; this file is the
> durable plan + progress log for the project on branch
> `claude/fix-gbdt-mean-regression-GcY1g` (PR #46 → `devel`).
>
> **All phases of this plan are worked and documented in PR #46.** Each
> phase's code, the harness run-log entries, and the status updates below land
> as commits on that one PR; do not open separate PRs per phase.

## Status / progress log

| Phase | State | Notes |
|---|---|---|
| **P0 — offline eval harness** | ✅ done (PR #46) | `src/sportstradamus/scripts/compression_eval.py` + `tests/golden/test_compression_eval.py`. ruff clean, 6 unit tests pass, CLI single+diff smoke-tested on synthetic data. Full `poetry` gates NOT run in the build env — network policy blocks the PyTorch CPU wheel source so `poetry install` fails on `torch`; needs a normal-network run before merge. |
| **P0.5 — determinism gate** | ✅ done (PR #46) | Opt-in `meditate --deterministic` (debug-only, never publish) + `tests/integration/test_determinism_gate.py`. Pure helpers `seed_everything` / `fit_lss_model` / `predict_lss_params` / `fit_predict_params` in `pipeline.py`; under `--deterministic`: RNGs pinned (random/numpy/torch + `torch.use_deterministic_algorithms`), Optuna swapped for `DETERMINISTIC_FIXED_PARAMS`, input frozen to cached parquet. Persistent writes are **redirected to `data/{test_sets,models}/deterministic/`** (training parquet + whole-suite `report()` stay fully suppressed) so a `--deterministic` run produces consumable artifacts without ever overwriting production paths. Gate runs on real cached `NBA_FGA.parquet` (4000 rows, ~5s) with stochastic LightGBM (`feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=1`) so it actually tests the seeding mechanism — different seed produces `loc` max-abs diff ~0.34, same seed bit-identical. Default `meditate` byte-identical. P1 unblocked. |
| P1 — centered-target bridge (SkewNormal) | ⬜ next | This *is* the additive empirical-Bayes offset the overconfidence investigation already tried (Phase A). It was found **inconclusive due to non-reproducibility, not wrong** — deterministically it was well-calibrated (+0.12 bias). Proceed under `--deterministic` so the P0 harness diff verdict is trustworthy. |
| P2 — `init_score` baseline (NegBin/ZINB) | ⬜ | Pair with the **confirmed, reproducible ZINB derived-π gate fix** (separate existing spec); see annotation in priority list. |
| P3–P10 | ⬜ | see priority list; P10 (GPBoost) already prototyped and failed deterministically — annotated below |

**Start next session here:** P1 (centered-target bridge, SkewNormal). Run the
candidate strategy with `meditate --deterministic`; outputs land in
`data/test_sets/deterministic/{LEAGUE}_{market}.csv` and
`data/models/deterministic/{LEAGUE}_{market}.mdl`. Score with
`compression_eval --test-sets-dir src/sportstradamus/data/test_sets/deterministic`
(or diff-mode against the production baseline CSV) so the P0 harness verdict
is trustworthy.
Keep the default strategy = current production behavior. Per the
overconfidence investigation, also confirm any training-side win actually
propagates to the live `model_prob.py` output before declaring it fixed (see
§"Live-path confound").

## Context

LightGBMLSS predictions in this repo compress toward the global mean: high-volume
players (e.g. NBA Anthony Edwards PTS) are systematically under-predicted, low-volume
over-predicted. The source report explains this is structural to gradient-boosted
trees (leaf averaging + shrinkage + no extrapolation), not a bug.

The repo **already implements the exact "season-mean ratio" workaround the report
identifies as underperforming**, and only for one of two branches:

- **SkewNormal branch** (`global_mean >= 2.0`, e.g. NBA PTS): target is
  `Result / MeanYr` clipped to `[0.01, ∞)`; at inference `loc`/`scale` are multiplied
  back by `MeanYr`. This is the multiplicative-amplification trap from report §5 /
  Key Finding #2 — a small downward bias in ratio space becomes a large absolute
  under-prediction for high-mean players.
- **NegBin/ZINB branch** (`global_mean < 2.0`): **raw counts, no normalization at
  all.** The user confirms compression is visible here too. Per report
  "Implementing in LightGBMLSS" step #4, count families cannot be additively
  centered — they need an `init_score`/baseline injection instead.

Neither branch uses `init_score`/`base_margin`, per-player centering, leakage-safe
player target encoding, sample weighting, or post-hoc isotonic calibration.
`player_id` is not a model feature. Diagnostics (`report.py`) already track
`ev_meanyr_corr` / `result_meanyr_corr` / `shape_ratio` but there is **no
per-player-mean-decile slicing**, which is the cleanest compression signature.

Scope (per user): **all leagues, all markets** — both branches must be addressed.
New ML deps (gpboost/pymer4) are acceptable in later phases if needed. Gate every
experiment with an **offline eval harness built first**.

Constraint discovered: the build container is a fresh clone — `data/training_data/`,
`data/test_sets/`, `data/models/`, `data/player_data/` do not exist, and
`poetry install` fails because the network policy blocks the PyTorch CPU wheel
source. The harness must therefore operate on artifacts a `meditate` run produces
(it already writes `data/test_sets/{LEAGUE}_{market}.csv` with X_test features
incl. `MeanYr`, predicted distribution params, `Result`, `Line`), not on
pre-cached matrices. The harness itself needs no network.

This is a multi-session project: build measurement infrastructure once, then work
down a priority list of interventions, shipping the first that closes the gap.

## Findings folded in from the overconfidence investigation

`docs/OVERCONFIDENCE_INVESTIGATION.md` (and its hand-back) ran a parallel,
deeper pass on two of the worst-compressed markets (NBA **FGA** SkewNormal,
**FG3M** ZINB). Its conclusions directly change this plan's risk profile.
Read both before resuming. The load-bearing ones:

1. **Offline evaluation was non-reproducible — this is the dominant blocker.**
   The same nominal SkewNormal config produced top-volume-quintile bias of
   −0.48, −0.92, −1.3, −2.0, −2.5, **and +0.12** across runs. Suspected
   sources: unpinned LightGBM/LightGBMLSS seeds, per-row `start_values`
   broadcasting in LightGBMLSS predict, Optuna nondeterminism, and `meditate`
   Optuna **starvation** in time-boxed offline runs (3–18 trials vs. the
   deployed model's hundreds). The P0 harness scores the CSVs `meditate`
   dumps, but `meditate` itself is the non-deterministic stage — so the
   harness's ship/kill verdict is currently noise. **No P1+ strategy can be
   validated until a determinism gate exists** (new §"Determinism prerequisite").

2. **P1 was already attempted and is not refuted.** The "centered-target
   bridge" (replace `y/MeanYr` with `y − baseline`, add baseline back to `loc`
   only) is the additive empirical-Bayes per-player offset the investigation
   built and reverted in its Phase A. It was **inconclusive, not wrong**: in a
   clean *deterministic* harness the production-equivalent SkewNormal model
   *with* the additive-EB offset was well-calibrated (predicted vol-quintile
   spread 7.4 vs actual 8.3, meanAbsBias **+0.12**, tracked volume). So P1
   remains the highest-leverage lever — but the +0.12 result also means the
   SkewNormal *training* stage may not be the dominant source of the live
   symptom (see #4). Treat P1 as "re-run under determinism and measure", not
   "implement a known fix".

3. **The `Result/MeanYr` slope artifact is corroborated but is not the level
   cause.** corr(predicted loc, MeanYr) was −0.37…−0.87 across *all*
   SkewNormal markets — a real multiplicative-amplification artifact, exactly
   what P1 targets. But the investigation's decisive negative result says this
   slope is "not the dominant level cause" of the live under-prediction. P1 is
   still worth doing; do not expect it alone to close the live gap.

4. **Live-path confound — the plan is purely training-side; the strongest
   unexplained lead is in prediction.** `Model Skew` (SkewNormal `alpha`) is
   **NaN for every live FGA row**, while offline replay proves the trained
   model emits *valid* alpha on saved features. The defect is therefore in the
   live path: `src/sportstradamus/prediction/model_prob.py` — feature/column
   alignment, `set_model_start_values` seeding live vs. train, the `fused_loc`
   `weight≈0.9` bookmaker blend, or `temperature≈1.37`. **A training-side
   compression fix that never reaches the published EV is the FGA dead end
   repeated.** Every shipped strategy must be verified end-to-end on the live
   path, not just on the dumped test set (new §"Live-path confound").

5. **ZINB gate under-fit is a confirmed, reproducible win adjacent to P2.**
   Separate from mean compression: the jointly-fit ZINB `gate` head converges
   to ≈ half the true structural-zero rate in every NBA ZINB market (FG3M 0.19
   vs 0.33; PF 0.02 vs 0.14; …), inflating `P(over@line)` everywhere. The fix
   (a derived-π two-stage ZINB, downstream code unchanged) is fully specced at
   `docs/superpowers/plans/2026-05-18-fga-fg3m-overconfidence-fix.md` (Phase B
   "SUPERSEDED → derived-π"). It is the single highest-confidence item across
   both projects. Fold it into P2.

6. **GPBoost (P10) was already prototyped deterministically and failed** — did
   not beat the EB offset, top-volume bias −2.5; its "flat fixed-effect" was a
   GPBoost-internal FE/RE artifact, not a property of the production model. Do
   not re-attempt P10 naively (annotation in priority list).

## Determinism prerequisite (P0.5 — blocks P1+)

Before any target/baseline strategy is A/B'd, make offline evaluation
bit-reproducible, or the P0 ship/kill verdict is meaningless (finding #1):

- Pin LightGBM and LightGBMLSS seeds; pin the train/test split seed.
- Make `set_model_start_values` row-broadcasting deterministic (it is also a
  suspected non-determinism source *and* the P1 `loc`-start change touches it).
- Use fixed hyperparameters (or a controlled, seeded, non-starved Optuna) for
  evaluation runs so a smoke retrain is comparable to the deployed model.
- **Determinism gate:** run the same strategy/config through `meditate` twice
  and assert bit-identical predicted parameters / test-set CSVs before any
  decile-table comparison is trusted. (The investigation's GPBoost harness
  already demonstrated bit-identical determinism is achievable here.)

This is small, high-leverage, and strictly precedes P1. Add it as a gate the
harness or a thin wrapper enforces, not a one-off manual check.

## Live-path confound (verify every shipped strategy end-to-end)

This plan optimizes the training stage; the overconfidence investigation shows
the live symptom may originate downstream (finding #4). For any strategy that
clears the P0 threshold on dumped test sets, before promoting it to default
also confirm it survives the live `model_prob.py` path: raw distribution
params → decode → `fused_loc` (`weight≈0.9` book blend) → `dispersion_cal` →
`temperature`. In particular resolve why `Model Skew`=NaN live but valid on
saved features — a strategy that fixes decile bias offline but is then
flattened by the book blend or NaN-ed in decode has not fixed the user-visible
problem.

## Critical files

| File | Role | Key lines |
|---|---|---|
| `src/sportstradamus/training/pipeline.py` | target build, dist select, training, denorm, test_set dump | 245–324 (branch/target), 328 (`lgb.Dataset` — `init_score` injection point), 341/394–409 (`set_model_start_values`), 345–346 (MeanYr monotone), 439–452 (SkewNormal denorm), ~960/981 (test_set dump) |
| `src/sportstradamus/training/report.py` | diagnostics → `training_report.txt`, `model_stats.parquet` | `ev_meanyr_corr`/`result_meanyr_corr` (~850), `write_model_stats` |
| `src/sportstradamus/stats/base.py` | baseline features + target | 676–702 (`MeanYr`, `Mean10`, `*_Ratio`), 1005/1011/1082 (`Result`) |
| `src/sportstradamus/stats/nba.py` | NBA `MIN`, `USG_PCT`, per-48 stats | 127–135, 359, 366 |
| `src/sportstradamus/helpers/distributions.py` | `set_model_start_values` (loc=1.0 in ratio space) | 425–504 |
| `src/sportstradamus/skew_normal.py` | custom SkewNormal (location-scale, supports negatives) | 30–199 |
| `src/sportstradamus/scripts/compression_eval.py` | **P0 harness** — decile table, compression ratio, run log, diff verdict | — |
| `src/sportstradamus/prediction/model_prob.py` | **Live-path confound** — where the FGA symptom (Model Skew=NaN, EV≪line) actually appears; verify shipped strategies here | SkewNormal decode, `fused_loc` w≈0.9 blend, `temperature`≈1.37 |
| `docs/superpowers/plans/2026-05-18-fga-fg3m-overconfidence-fix.md` | Existing task-by-task spec for the **ZINB derived-π gate** fix (pair with P2) | Phase B "SUPERSEDED → derived-π" |

## Architectural principle (applies to all phases)

Make the **target/baseline transform a single configurable strategy**, selected by
a CLI flag on `meditate` (and a matching env var for the harness), defaulting to
current behavior. Every experiment becomes a new strategy value, not a destructive
rewrite. This is what makes the multi-session A/B tractable and keeps `devel`
shippable between sessions. Centralize the forward transform, the inverse
(de-norm) transform, and the inference-time mirror so train/predict cannot drift.
The inference mirror lives in `stats/base.py:get_stats`; any new baseline must be
computed there identically and leakage-safe.

## Phase 0 — Offline eval harness (DONE)

Delivered in `src/sportstradamus/scripts/compression_eval.py` (`click` CLI),
reading `data/test_sets/{LEAGUE}_{market}.csv` (no network), emitting:

1. **Per-player-mean-decile table**: bin rows by `MeanYr` decile; per decile
   MAE, bias (mean signed error `pred − actual`), prediction-vs-actual mean.
   Compression signature = monotone negative bias rising across top deciles.
2. **Compression ratio**: `std(predicted_mean) / std(actual)` overall and top-decile
   (report cites Wheeler's 7.7× as the pathological end; 1.0 = no compression).
3. **`result_meanyr_corr` vs `pred_meanyr_corr`** (mirrors report.py definitions).
4. **Scatter PNG**: predicted vs actual, colored by `MeanYr` decile, y=x reference.
5. A **scorecard** appended to a run log (`data/compression_eval_log.csv`) keyed
   by strategy name + git SHA, so cross-session comparison is mechanical.
6. `--baseline`/`--candidate` **diff mode**: prints the delta + ship/kill verdict
   and exits non-zero on KILL.

**Universal decision threshold (every experiment):** ship a strategy only if it
reduces **top-mean-decile MAE by ≥ 5%** vs the current production strategy without
worsening **global MAE by > 1%** and without worsening `brier_skill_score` on the
existing report. Otherwise kill it and move to the next priority.

Outstanding for P0: real-data validation — run `poetry run meditate --league NBA`
for one high-mean (PTS) and one low-mean market in an env with normal network,
confirm the decile table/scatter show the known top-decile under-diagonal cluster,
and run the full `poetry` quality gates.

**P0.5 (determinism gate) now sits between P0 and P1** — see §"Determinism
prerequisite". It is the overconfidence investigation's #1 finding and is a
hard precondition for trusting any P1+ diff verdict.

## Priority list of interventions (work down until threshold met)

Mapped from the report's LightGBMLSS-specific order, adapted to both branches.
Each is a new target/baseline strategy behind the configurable flag; each is
gated by the harness.

**P1 — Centered-target bridge (SkewNormal branch).** Report's #1, single highest
leverage. Replace `y / MeanYr` with `y − baseline` where `baseline` is a
leakage-safe player baseline (start with existing `MeanYr`/`Mean10`; verify it is
prior-games-only — if it includes the current game, fix the leak first). Train
SkewNormal on the centered residual (location-scale family supports negatives —
fits cleanly). At inference add `baseline` back to **`loc` only**; `scale`/`alpha`
unchanged (kills the multiplicative amplification at pipeline.py:439–452). Update
`set_model_start_values` (loc start → 0, not 1.0) and the `get_stats` mirror.
*Expected: large.* **Investigation note:** this is the Phase-A additive-EB
offset — inconclusive (non-reproducible), *not* refuted; deterministically it
was well-calibrated (+0.12 bias). Blocked on P0.5. The `loc`-start change
touches the same `start_values` broadcasting flagged as a non-determinism
source — the investigation also found the offset-mode `loc=0` seeding was a
confirmed regression bug (fixing toward the per-row prior halved bias in
isolation); in centered space residual mean ≈ 0 so a 0 start is semantically
right, but verify the broadcast is per-row and deterministic, not a degenerate
global 0. Then verify it survives the live path (§"Live-path confound").

**P2 — `init_score` player baseline (NegBin/ZINB branch).** Report's #4 — count
families can't be centered. Inject the log-link of the player baseline as
per-row `init_score` on the count/location parameter via the `lgb.Dataset` at
pipeline.py:328; booster learns only the deviation. Verify LightGBMLSS supports
per-parameter `init_score` on a small sample first; if fiddly, fall back to a
strong leakage-safe target-encoded player-baseline feature (P5) plus reduced
regularization (P6) for this branch. *Expected: large.*

> **Pair P2 with the ZINB derived-π gate fix (confirmed, reproducible,
> already specced).** The overconfidence investigation proved a *distinct*
> ZINB defect from compression: the jointly-fit `gate` head learns ≈ half the
> true structural-zero rate path-wide, inflating `P(over@line)` everywhere.
> `init_score` on the count base does not fix the gate. The sound fix is a
> derived-π two-stage ZINB (calibrated zero classifier `q`; `gate =
> clip((q − NB(0))/(1 − NB(0)), 0, 1)`), keeping all downstream ZINB code
> unchanged. Task-by-task spec:
> `docs/superpowers/plans/2026-05-18-fga-fg3m-overconfidence-fix.md`
> (Phase B "SUPERSEDED → derived-π"). It is the highest-confidence,
> reproducible win across both projects and is independent of the
> determinism blocker — consider doing it first within the count branch.

**P3 — Rate decomposition (NBA + any league with a clean volume driver).** Report's
#2. Center the *rate* (`stat / MIN` for NBA using `nba.py` `MIN`; analogous volume
driver per league where one exists) and multiply by a separately projected volume
at inference. Stack on P1/P2. Skip for leagues with no stable volume analog.
*Expected: large where applicable.*

**P4 — Verify distribution family per branch.** Report's #3. Confirm centered →
SkewNormal/Student-t; raw count → NegBin/ZINB via init_score; rate → positive
continuous. Cheap sanity gate before deeper work.

**P5 — Leakage-safe target-encoded player features.** Report's #5. Expanding-window
`groupby(player).expanding().mean().shift(1)` player (and player×opponent) encoding
added in `stats/base.py`. Helps both branches; also the fallback for P2.
*Expected: medium.*

**P6 — Reduce tree regularization slightly.** Report's #6. Widen Optuna ranges:
larger `num_leaves`/`max_depth`, smaller `min_child_samples`/`min_child_weight`
(pipeline.py:348–368). Re-check decile bias. *Expected: small.*

**P7 — Isotonic post-hoc calibration on the location parameter.** Report's #8.
Fit `IsotonicRegression(out_of_bounds="clip")` of actual vs predicted location on
the existing validation split; apply at inference. Cheap polish. *Expected: small.*

**P8 — Sample weighting (upweight high-target games).** Report's #9. LightGBM
`sample_weight`, ≤2× at top end. Last resort; re-check global calibration.
*Expected: small, with tradeoff.*

**P9 — MERF-style iteration (hand-rolled).** Report's #10. Wrap P1/P2 in an
alternating fit-residual / re-estimate-shrunken-per-player-baseline loop to
convergence. More engineering; only if one-shot baseline proves too crude.

**P10 — GPBoost / mixed-effects migration.** Report's last resort. Only if P9 is
exhausted/unstable or LSS flexibility proves unnecessary. New dependency
(`gpboost`); user pre-approved deps for a phase that needs it. Treat as a separate
multi-session sub-project with its own plan. **Investigation note:** GPBoost
was already prototyped deterministically for FGA and **failed** — it did not
beat the additive-EB offset (top-volume bias −2.5), and its "flat
fixed-effect" decomposition turned out to be a GPBoost-internal FE/RE artifact,
not a property of the production model. Do not re-attempt naively; if revisited,
treat the prior negative result as the baseline to beat.

## Session handoff

- One strategy/experiment per session where feasible (aligns with CLAUDE.md
  "one module per session"); commit + push to `claude/fix-gbdt-mean-regression-GcY1g`
  and update the harness run log so the next session sees the scorecard history.
- Keep the default strategy = current production behavior until an experiment
  clears the threshold, so `devel`-tracking production is never regressed
  mid-project.
- Record each experiment's scorecard verdict (ship/kill) in the run log committed
  to the repo (not a scratch doc), and update the **Status / progress log** table
  at the top of this file.

## Verification (every code session)

- `poetry run ruff check src/sportstradamus/`
- `poetry run pytest tests/golden/` (incl. `test_compression_eval.py`)
- `poetry run pytest -m integration` (fake-mode, no network)
- Regenerate CLI help snapshots if `meditate` flags change:
  `REGENERATE_SNAPSHOTS=1 poetry run pytest tests/golden/test_cli_help.py`
- Determinism gate (P0.5): `poetry run pytest tests/integration/test_determinism_gate.py -v -m integration`
  must pass (proves seeded LightGBM under `--deterministic` is bit-reproducible).
  Candidate strategies are A/B'd under `meditate --deterministic` so the
  scorecard delta is trustworthy.
- Functional gate: harness scorecard delta vs current strategy meets the P0
  threshold before a strategy is promoted to default.
- Live-path gate: the promoted strategy is confirmed end-to-end through
  `model_prob.py` (no `Model Skew`=NaN, EV not collapsed by the book blend),
  not only on the dumped test set.
