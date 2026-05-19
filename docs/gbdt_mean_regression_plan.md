# Plan: Mitigate GBDT Regression-Toward-the-Mean in the Training Pipeline

> **Multi-session handoff doc.** Status below is updated each session. The
> attached source report lives in the originating session; this file is the
> durable plan + progress log for the project on branch
> `claude/fix-gbdt-mean-regression-GcY1g` (PR #46 → `devel`).

## Status / progress log

| Phase | State | Notes |
|---|---|---|
| **P0 — offline eval harness** | ✅ done (PR #46) | `src/sportstradamus/scripts/compression_eval.py` + `tests/golden/test_compression_eval.py`. ruff clean, 6 unit tests pass, CLI single+diff smoke-tested on synthetic data. Full `poetry` gates NOT run in the build env — network policy blocks the PyTorch CPU wheel source so `poetry install` fails on `torch`; needs a normal-network run before merge. |
| P1 — centered-target bridge (SkewNormal) | ⬜ next | |
| P2 — `init_score` baseline (NegBin/ZINB) | ⬜ | |
| P3–P10 | ⬜ | see priority list |

**Start next session here:** P1. Keep the default strategy = current
production behavior; gate with the P0 harness diff mode.

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
*Expected: large.*

**P2 — `init_score` player baseline (NegBin/ZINB branch).** Report's #4 — count
families can't be centered. Inject the log-link of the player baseline as
per-row `init_score` on the count/location parameter via the `lgb.Dataset` at
pipeline.py:328; booster learns only the deviation. Verify LightGBMLSS supports
per-parameter `init_score` on a small sample first; if fiddly, fall back to a
strong leakage-safe target-encoded player-baseline feature (P5) plus reduced
regularization (P6) for this branch. *Expected: large.*

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
multi-session sub-project with its own plan.

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
- Functional gate: harness scorecard delta vs current strategy meets the P0
  threshold before a strategy is promoted to default.
