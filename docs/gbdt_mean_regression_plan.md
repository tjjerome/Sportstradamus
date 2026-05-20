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
| **P1 — centered-target bridge (SkewNormal)** | ✅ done (PR #46), result: **FGA-only SHIP, family-wide KILL** | Two centered-target variants A/B'd path-wide under `meditate --deterministic` against the `ratio_meanyr` baseline. (a) `centered_additive_eb_meanyr_k10` (Phase-A's EB(MeanYr, K=10)): FGA SHIPS (+5.3% top-decile MAE, brier_skill +0.096→+0.112), every other SkewNormal market KILLs (PTS −3.5%, PA −4.1%, PR −2.9%, RA −2.2%, FG3A −3.8%, FGM −2.6%, MIN +3.7%, PRA +0.8%, REB +0.2%, fantasy-points-prizepicks brier_skill regressed). (b) `centered_additive_mean10` (trailing-10 baseline, more responsive to recent form — added post-Phase-A as the obvious "level shifts with form" hypothesis): **every SkewNormal market KILLs** including FGA (+4.6% — close but under the 5% bar), with PA −6.6% and PR −6.7% notably *worse* than Phase-A. Count-family markets (FG3M, FTM, OREB, PF, STL, TOV, BLK, BLST) showed exactly 0% delta under both strategies as expected — the centered-target transform is a no-op for NegBin/ZINB. **Both runs together confirm the OVERCONFIDENCE_INVESTIGATION §3.2 "decisive negative result", strengthened: the SkewNormal level bias is not the dominant compression cause path-wide, regardless of baseline horizon (long-term EB or short-term trailing-10).** FGA is genuinely special — its win comes from EB(MeanYr) capturing structural shot-volume; Mean10 is too noisy for FGA itself and not the right lever for volume-shifting markets either. Default `--target-strategy=ratio_meanyr` stays. The infrastructure (`baselines.py`, the registry, the offset_meta pickle field, the brier_skill gate, the live-path test) is reusable for P3 (rate decomposition) and P2 (init_score baseline for count markets) — the next levers worth trying based on the path-wide negative result. |
| **P2.B — HurdleZINB (derived-π gate)** | ✅ done (PR #46), result: **6/8 NBA ZINB markets SHIP** | New `meditate --zinb-mode=hurdle` (orthogonal to `--target-strategy`; default `joint` stays byte-identical to pre-P2.B). `HurdleZINB` (in `src/sportstradamus/hurdle.py`) is a two-stage drop-in for joint ZINB: calibrated binary classifier estimates `q = P(Y=0)`; NegBin LightGBMLSS on `Y>0` supplies count shape; structural-inflation π derived from the ZINB identity `π = clip((q − NB(0))/(1 − NB(0)), 0, 1)` (NOT the simpler `gate = 1 − p_nonzero` from the original Phase B spec — corrected because downstream `fused_loc` in `helpers/distributions.py` explicitly treats `gate` as zero-inflation, not marginal P(Y=0)). Returned `total_count/probs/gate` columns match the LightGBMLSS ZINB contract so `model_prob` ZINB decode (lines 252-257) is untouched and legacy pickles still load via `getattr(model, "is_hurdle", False)`. Path-wide A/B under `meditate --deterministic --league NBA --zinb-mode hurdle` against the joint baseline: **SHIP** on FG3M (+9.7% top-decile MAE, brier_skill +0.115→+0.290), OREB (+44.9%, +0.019→+0.109), PF (+19.2%, −0.238→−0.002), TOV (+26.8%, −0.049→+0.058), BLK (+40.4%, +0.237→+0.299), BLST (+11.6%, −0.002→+0.093). **KILL** on FTM (+1.3%, under 5% bar) and STL (global MAE +14.1% regression). The joint ZINB had per-row catastrophic blowups in mid-deciles on BLK/OREB/PF/BLST under deterministic mode (compression_ratio 24–5357×; predicted means up to 1437) that the hurdle eliminates entirely — global MAE drops 60–99% on those markets. **Default stays `--zinb-mode=joint`** — shipping the infrastructure + verdict here; the per-market routing question (FTM/STL stay joint, the rest move to hurdle) is a follow-up. **Note on the verdict criterion**: the parent plan said "predicted gate mean ≈ hist_gate" — that criterion was mis-stated under derived-π semantics. Derived-π gate is π_zi (the inflation parameter), structurally ≤ q with equality only in the zero-truncated-NB limit. For FG3M (positives mean ≈ 2.2) NB(0) ≈ 0.20 even with a well-fit NB, so derived-π gate ≈ 0.17 (similar to joint's 0.18) but the *total reconstructed P(Y=0)* matches `q ≈ 0.33` exactly by construction. The meaningful SHIP/KILL signal is the downstream compression_eval verdict on `P(over@line)` proxies (top-decile MAE + brier_skill_score), not gate mean. New `tests/integration/test_zinb_hurdle_live_path.py` asserts the identity reconstruction `π + (1−π)·NB(0) ≈ q` per-row (mean tolerance 0.02) and two-run bit-identity under `DETERMINISTIC_SEED`. Determinism gate extended with a parallel hurdle assertion. |
| **P2.A — `init_score` baseline (NegBin/ZINB)** | ✅ closed: **DEAD** | In-process spike on FG3M: LightGBMLSS accepts per-row `init_score` (as a length-2n flat array, `[log_EB, zeros]` per-parameter concatenation) without raising — but the produced predictions are **byte-identical** to a plain NegBin fit, every decile. Either LightGBMLSS overrides init_score with its own `start_values` seeding, or the 30-round deterministic fit converges to the same answer regardless of starting point. Either way, the bias signature does not move. Also: FG3M's plain-NegBin top-decile bias is already −0.013 — there is no meaningful compression signature on the count-branch NegBin mean to fix; the overconfidence was the gate, which P2.B already addresses. **P2.A is dead** on the count branch as a one-line `init_score` transform. Per parent plan, the fallback is P5 (leakage-safe target-encoded player-baseline feature) or P3 (rate decomposition) — both require their own design sessions. |
| P3–P10 | ⬜ | see priority list; P10 (GPBoost) already prototyped and failed deterministically — annotated below |

**Start next session here:** P2.A is dead and P2.B ships. Candidates queued
for the next session, in priority order:

1. **Per-market routing decision for P2.B.** FTM and STL kill under hurdle;
   the other 6 NBA ZINB markets ship. A simple `data/zinb_mode_per_market.json`
   config plus a per-market lookup in `pipeline.py` would let us ship hurdle
   only where it wins. Tractable, ~1 session.
2. **P3 — rate decomposition.** Center `stat / MIN` for NBA and project
   volume separately. The natural next experiment for SkewNormal
   volume-shifting markets (PA/PR/PRA) per the
   `CENTERED_TARGET_NEGATIVE_RESULT.md` discussion §5.
3. **Live-path NaN root cause.** `Model Skew = NaN` for live FGA rows
   despite valid offline replay (`OVERCONFIDENCE_INVESTIGATION.md` §3.4) —
   "tackle when diminishing returns hit" criterion now met since P1+P2.A
   are dead and P2.B ships.

The Phase B spec at
`docs/superpowers/plans/2026-05-18-fga-fg3m-overconfidence-fix.md` (Phase B
"SUPERSEDED → derived-π") was the source for the HurdleZINB design but
needed one math correction: the gate returned by `.predict` is π_zi
(derived from the ZINB identity), NOT `1 − p_nonzero`. Future readers of
the Phase B spec should mentally swap that line.

**Deterministic-mode hardening (fixed in PR #46 post-P1):**
`meditate --deterministic` now auto-implies `--force` (in `cli.py:meditate`)
because the input-freeze leaves `new_M` empty, which would otherwise
short-circuit `train_market` at the `if new_M.empty and not force and not
need_model: return` line whenever a prior model pickle exists. Also
hardened: `stat_zi.json` and `feature_filter.json` writes are skipped under
`--deterministic` — those configs flow in-memory through the current run
but never get persisted, so the crippled deterministic hyperparameters
can't mutate production config.

**P1 follow-ups (closed or deferred):**
1. ~~**Promote `centered_additive_eb_meanyr_k10` to default.**~~ **DEAD** —
   the path-wide A/B shows the win is FGA-only; every other SkewNormal
   market regresses. Default stays on `ratio_meanyr`. Centered strategy
   remains available as an opt-in for FGA-specific runs (or future
   per-market strategy selection if that's ever built).
2. ~~**Path-wide A/B**~~ ✅ done (above): FGA SHIP, rest KILL.
3. **Live-path NaN root cause (deferred — "diminishing-returns trigger")**:
   `tests/integration/test_centered_target_live_path.py` guards that the
   centered strategy's decode is finite, but the original FGA
   `Model Skew=NaN` symptom (per
   `docs/OVERCONFIDENCE_INVESTIGATION.md` §3.4) was an
   *existing-model + live `model_prob.py`* failure, distinct from
   anything P1 touched. **Tackle this when further model-improvement
   work (P2 init_score baseline, P3 rate decomposition, etc.) starts
   yielding diminishing returns** — the live-path NaN bug is likely
   responsible for a chunk of the published-EV pathology that
   training-side fixes can't reach. Don't lose track of it.
4. **`EB_SHRINKAGE_K` tuning** — Phase-A "Task A7" follow-up. Only
   relevant if a per-market FGA centered_additive deployment ships and
   shows residual mid-volume over-prediction. Low priority now that the
   path-wide A/B is closed.

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
leverage. Replace `y / MeanYr` with `y − baseline`. Train SkewNormal on the
centered residual (location-scale family supports negatives — fits cleanly).
At inference add `baseline` back to **`loc` only**; `scale`/`alpha` unchanged
(kills the multiplicative amplification at pipeline.py:612–615). Update
`set_model_start_values` (loc start → 0, not 1.0) and the `get_stats` mirror.
*Expected: large.*

**Baseline = the Phase-A EB prior, first cut.** The overconfidence
investigation's Phase A used `baseline = EB(MeanYr, GamesPlayed, K=10)` where
`EB_prior = (GamesPlayed·MeanYr + K·global_mean) / (GamesPlayed + K)` (a
one-line James-Stein/empirical-Bayes shrinkage of season-to-date `MeanYr`
toward the global mean — stops noisy low-sample players from blowing up the
centered residual). `EB_SHRINKAGE_K = 10.0`. **Not raw `MeanYr`; not `Mean10`.**
The Phase-A spec lives at `docs/superpowers/plans/2026-05-18-fga-fg3m-overconfidence-fix.md`
(`compute_eb_prior` helper, `EB_SHRINKAGE_K`); replicate it as the first P1
strategy under P0.5 determinism — this is the documented prior win and the
cleanest single-variable re-validation.

**Configurable baseline-source strategy.** Per the architectural principle,
the baseline itself is a strategy value, not a hardcoded choice. Wire a
`TargetBaseline` plug (e.g. enum / strategy object) with at least these
candidates: `eb_meanyr_k10` (Phase-A, default for P1), `meanyr` (raw),
`mean10` (raw trailing-10), `blended` (`α·Mean10 + (1−α)·EB(MeanYr)`). First
P1 ship attempt uses `eb_meanyr_k10` only; the others are cheap A/B
follow-ups through the same `compression_eval` harness if Phase-A clears the
threshold (or if it doesn't — Mean10 is the obvious next try). Tune
`EB_SHRINKAGE_K` only after baseline-source is settled.

**Leakage audit is task zero.** [`stats/base.py:682`](src/sportstradamus/stats/base.py#L682)
computes `stats["MeanYr"] = playergames.groupby(...).mean()`. Verify
`playergames` is strictly `< game_date`, not `≤`. The audit applies more
strictly to `Mean10` than to `MeanYr`: an off-by-one that includes the
current game contaminates ~10% of a 10-game window vs ~1% of a season —
add a regression test asserting `MeanYr`/`Mean10` at `game_date` equals the
expected expanding-then-shifted aggregate.

**Investigation notes** (don't rediscover these):
1. Phase A was *inconclusive (non-reproducible)*, **not refuted** —
   deterministically the production-equivalent additive-EB SkewNormal model
   was well-calibrated (+0.12 top-quintile bias, vol-spread 7.4 vs actual
   8.3). P0.5 is the determinism gate that makes the Phase-A re-run a
   trustworthy A/B.
2. The `loc`-start=0 change touches `set_model_start_values` — the
   investigation flagged the offset-mode `loc=0` seeding as a confirmed
   regression bug (fixing to `loc=mu` halved bias in isolation). In
   centered space the residual mean ≈ 0 so `loc=0` is semantically right
   *if* the per-row broadcast is deterministic, not a degenerate global 0.
   Verify with a unit test on `set_model_start_values` under the centered
   path (sized to `len(X)`, not scalar 0).
3. After Phase-A's offline win, **confirm end-to-end through
   `prediction/model_prob.py`** before declaring P1 done — the
   user-published EV path is where the FGA symptom actually lives
   (`Model Skew=NaN` live, w≈0.9 book blend, temperature). See
   §"Live-path confound".

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

### Tooling note: `gh` is a userspace install on this workstation

PR #46 CI status and review-comment monitoring use the GitHub CLI. `gh` is
**not a system package** — it lives at `~/.local/bin/gh` (installed
2026-05-19 via the official static tarball release, since `sudo apt` would
have required an interactive password). Future sessions must ensure
`~/.local/bin` is on `PATH`; on this workstation it already is (set in
`~/.profile` / `~/.bashrc`), but a sandboxed or non-login shell may not
inherit it. If `gh --version` fails, run:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Authentication is also a one-time setup the user completes locally
(`gh auth login` interactive, or `export GH_TOKEN=…` from a PAT with `repo`
scope). Agent sessions don't have credentials by default — if `gh api …`
returns `HTTP 401`, the user needs to re-auth.

### Per-session rules

- One strategy/experiment per session where feasible (aligns with CLAUDE.md
  "one module per subagent" — the per-strategy scope discipline, not the
  serial execution); commit + push to `claude/fix-gbdt-mean-regression-GcY1g`
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
