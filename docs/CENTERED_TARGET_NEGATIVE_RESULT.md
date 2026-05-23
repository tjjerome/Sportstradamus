# Centered-Target Family — Path-Wide Negative Result (Handoff for Research)

**Date:** 2026-05-20
**Branch:** `claude/fix-gbdt-mean-regression-GcY1g` (PR #46 → `devel`)
**Status:** P1 of `docs/archive/gbdt_mean_regression_plan.md` (superseded 2026-05-23 by `docs/operation_ship_75.md`) is complete; result is a
strong negative finding. Handing off to a research agent to figure out where
to take the project next.
**Lineage:** Direct continuation of `docs/OVERCONFIDENCE_INVESTIGATION.md`
(Phase A specifically) under a now-trustworthy determinism gate.

This document is self-contained. A researcher should be able to read it
without prior context, understand what was tested, what was ruled out, and
what open questions remain. Concrete numbers are inlined; pointers to code
and artifacts are at the bottom.

---

## 1. One-line takeaway

The "centered additive target" family — replace SkewNormal's
multiplicative `Result / MeanYr` target with `Result − baseline`, decode
`loc + baseline` — **ships on NBA FGA only and is dead on every other
SkewNormal market**, regardless of baseline horizon. The earlier
OVERCONFIDENCE_INVESTIGATION §3.2 "decisive negative result" (SkewNormal
training stage is probably NOT the dominant source of the live FGA bias)
is now reproduced and strengthened across the path.

## 2. What we were trying to learn

LightGBMLSS predictions in this repo compress toward the global mean:
high-volume players under-predicted, low-volume over-predicted
(`docs/archive/gbdt_mean_regression_plan.md` §Context). The hypothesis underlying
Phase A in `docs/OVERCONFIDENCE_INVESTIGATION.md` was that the SkewNormal
branch's multiplicative target `y / MeanYr` (the only branch where
normalization is currently applied) imposes a **multiplicative-amplification
artifact**: a small downward bias in ratio space becomes a large absolute
under-prediction for high-mean players (cf. corr(predicted loc, MeanYr)
ranged −0.37…−0.87 across all SkewNormal markets).

Phase A built an additive empirical-Bayes offset to replace the ratio
target, but the result was **inconclusive due to non-reproducibility**: the
same nominal config produced top-volume-quintile bias of −0.48, −0.92, −1.3,
−2.0, −2.5, and **+0.12** across different runs. The +0.12 came from one
clean deterministic harness; production-equivalent runs landed all over the
map.

So P0.5 was added to the parent plan: a bit-reproducible determinism gate
(`meditate --deterministic`, pinned seeds, fixed hyperparameters, frozen
input). With that in place, P1 re-tested Phase A's idea under conditions
where the SHIP/KILL verdict is trustworthy. The negative finding documented
here was previously suspected but not provable.

## 3. What we built (so a researcher knows what's available)

All on branch `claude/fix-gbdt-mean-regression-GcY1g` (PR #46), tested and
green:

- **`src/sportstradamus/scripts/compression_eval.py`** (P0): per-mean-decile
  table, compression ratio, ship/kill verdict against three gates —
  top-decile MAE improvement ≥ 5%, global MAE regression ≤ 1%,
  brier_skill_score (model vs book Brier ratio) not worse. CLI:
  `compression_eval --baseline <baseline.csv> --candidate <cand.csv>`.
- **`meditate --deterministic`** (P0.5): bit-reproducible offline eval.
  Pinned LightGBM/torch/numpy RNGs (`seed_everything`), fixed
  `DETERMINISTIC_FIXED_PARAMS`, frozen training input (no live fetch),
  outputs go to `data/{test_sets,models}/deterministic/{strategy}/` so A/B
  runs don't clobber each other. Auto-implies `--force` so the cache short-
  circuit doesn't make runs silent no-ops. Skips writes to `feature_filter.
  json` / `stat_zi.json` so crippled-quality runs can't mutate prod config.
- **`src/sportstradamus/training/baselines.py`** (P1): single-source-of-
  truth strategy registry. Each strategy bundles `forward(y, X, ...)`,
  `decode_loc(loc, X, ...)`, `decode_scale(scale, X, ...)`,
  `start_mode_flag`, and `offset_meta(...)` (persisted to the pickle so the
  prediction-side mirror in `model_prob.py` recomputes the baseline
  identically). Three strategies registered:
  - `ratio_meanyr` (default, bitwise-equivalent to legacy production)
  - `centered_additive_eb_meanyr_k10` (Phase-A, EB-shrunk MeanYr, K=10)
  - `centered_additive_mean10` (trailing-10 raw mean with MeanYr fallback)
- **`tests/integration/test_centered_target_live_path.py`**: drives a
  deterministic SkewNormal model through the actual `model_prob.py` decode
  helper and asserts `Model Skew` is finite + EV matches the formula. This
  is the guard for the FGA "Model Skew = NaN live" dead end from
  OVERCONFIDENCE_INVESTIGATION §3.4.

The infrastructure is pluggable: a new candidate strategy is ~30 lines in
`baselines.py` plus a few unit tests; the rest of the pipeline picks it up
automatically via `--target-strategy <slug>`.

## 4. Results

### 4.1 A/B methodology

Two `meditate --deterministic --league NBA` runs against the cached
training parquet:

1. Baseline: `--target-strategy ratio_meanyr` (legacy production).
2. Candidate A: `--target-strategy centered_additive_eb_meanyr_k10`.
3. Candidate B: `--target-strategy centered_additive_mean10`.

Each run trains every NBA market and dumps a per-market test-set CSV with
`Result`, `Line`, `Odds`, `P` (calibrated model over-prob), and `EV`.
`compression_eval --baseline <ratio> --candidate <cand>` scores each market
independently.

The runs are bit-reproducible (proven by `tests/integration/
test_determinism_gate.py` shipping with P0.5). A re-run produces byte-
identical CSVs at the same seed; a different seed produces measurably
different output, so the gate genuinely tests the seeding mechanism.

### 4.2 Full path-wide table

`top-decile MAE improvement` (positive = candidate has lower MAE = better;
5% is the SHIP threshold). NB: `compression_eval`'s verdict text uses the
phrase *"top-decile MAE improved X%"* with the sign attached — `+5.3%`
means a 5.3% drop in MAE (good); `−3.5%` means a 3.5% rise (bad).

| Market | Dist (assumed) | A: `eb_meanyr_k10` | B: `mean10` | Notes |
|---|---|---|---|---|
| **FGA** | SkewNormal | **+5.3% SHIP** | +4.6% KILL (under bar) | FGA is the only market that ships; EB(MeanYr) is genuinely better than Mean10 for it |
| PTS | SkewNormal | −3.5% KILL | −1.5% KILL | Mean10 less bad but still worse than ratio |
| PA (PTS+AST) | SkewNormal | −4.1% KILL | **−6.6% KILL** | Mean10 substantially *worse* than EB |
| PR (PTS+REB) | SkewNormal | −2.9% KILL | **−6.7% KILL** | Mean10 substantially *worse* than EB |
| PRA (PTS+REB+AST) | SkewNormal | +0.8% KILL | +1.6% KILL | Both negligible |
| RA (REB+AST) | SkewNormal | −2.2% KILL | −0.4% KILL | Both KILL |
| AST | SkewNormal | (not in EB summary table) | +1.5% KILL | Small bump, under bar |
| REB | SkewNormal | +0.2% KILL | −0.8% KILL | Negligible |
| DREB | SkewNormal? | (not in EB summary table) | −4.4% KILL | Got worse under Mean10 |
| FGM | SkewNormal | −2.6% KILL | −1.5% KILL | Both KILL |
| FG3A | SkewNormal | −3.8% KILL | +3.9% KILL | Mean10 ~7pp better, still under bar |
| MIN | SkewNormal | +3.7% KILL | +3.6% KILL | Both small, both under bar |
| fantasy-points-prizepicks | SkewNormal | brier_skill regressed → KILL | 0% (no-op?) | EB regressed Brier; Mean10 0% suggests this market may not be SkewNormal in the cached parquet — worth confirming |
| FG3M, FTM, OREB, PF, STL, TOV, BLK, BLST | NegBin / ZINB | 0% (no-op) | 0% (no-op) | Centered-target transform doesn't apply to count families — confirmed |

### 4.3 What's surprising in the table

1. **FGA is the only ship and it's barely.** +5.3% just clears a 5% bar.
   And Mean10 (which on paper should be a *more* responsive baseline for
   form-driven markets) gets WORSE FGA performance than EB(MeanYr) at +4.6%.
2. **PA and PR get *substantially* worse under Mean10 than under EB.** PA
   −4.1% → −6.6%, PR −2.9% → −6.7%. The trailing-10 baseline actively hurts
   the volume-shifting markets it was supposed to help. This is the
   strongest negative signal in the experiment.
3. **AST is the only non-FGA market where Mean10 produces a small
   improvement (+1.5%)**, far below the ship bar. Suggests AST has *some*
   form-recency component but it's not the dominant compression cause.
4. **MIN improves marginally under both (+3.6%, +3.7%)**, never clearing
   5%. MIN is the closest the strategies come to working on a non-FGA
   market — likely because minutes are partly structural (starter vs bench
   role).
5. **The "Phase A might still work — it was inconclusive, not refuted"
   framing from OVERCONFIDENCE_INVESTIGATION is now refuted *with caveat*.**
   The +0.12 top-quintile bias that motivated re-trying Phase A was real
   and reproducible — it's the FGA number. It just doesn't generalize.

## 5. Interpretation

The hypothesis the centered-target family was testing: "the multiplicative
amplification (`loc` × `MeanYr`) is the dominant cause of compression for
high-volume players; replace the target with `y − baseline` to kill the
amplification factor, and bias should drop across the path."

The result rejects that hypothesis for every SkewNormal market except FGA.

**Why FGA likely shipped:**

NBA shot volume (FGA) is *structural*: it's set by role, rotation, and team
scheme. A given player's FGA per game is remarkably stable across a season
once their role is established. The Phase-A EB(MeanYr) baseline captures
this stable per-player level extremely well — the residual `y − EB_prior`
has small variance and is approximately mean-zero, exactly the input shape
SkewNormal handles well. The ratio target `y / MeanYr` distorts this signal
through the multiplicative encoding.

**Why every other SkewNormal market killed:**

PTS, PA, PR, etc. depend on `volume × efficiency`, both of which move. A
player can have a stable FGA but a 30% swing in FG% game-to-game, which
yields a large swing in PTS. Subtracting a stable EB(MeanYr) baseline
leaves a residual that still has the full efficiency-variance signal AND
loses the multiplicative-amplification handle that was at least partially
calibrating to per-player scale. Net: residual is bigger and harder to
predict than the original `y / MeanYr` target. The ratio target apparently
imposes a useful prior on per-player scale that the centered target
discards.

This is consistent with the OVERCONFIDENCE_INVESTIGATION §3.2 finding:
"the SkewNormal training stage is **probably not** the source of the live
FGA bias" — re-stated: the SkewNormal *level* bias is not the dominant
compression cause for the path, even though it *is* the dominant cause for
FGA specifically.

**What this means for the parent project's priority list:**

The priority list in `docs/archive/gbdt_mean_regression_plan.md` queued several
other levers in order. Re-reading them in light of this negative result:

- **P2 — `init_score` baseline for NegBin/ZINB markets.** *Different
  branch entirely.* Count families have a structurally different
  compression mode (no normalization at all currently). Worth doing on its
  own merits; not affected by this finding. Paired in the plan with the
  reproducible ZINB derived-π gate fix.
- **P3 — Rate decomposition** (center `stat / MIN`, project volume
  separately). *This was the right candidate for the volume-shifting
  markets that just killed.* The negative result here strengthens the case
  for P3 — neither MeanYr nor Mean10 is the right baseline because both
  conflate volume and efficiency; rate decomposition tackles them
  separately. The natural next experiment.
- **P5 — Leakage-safe target-encoded player features.** Would inject
  per-player baseline as a *feature*, not a target transform. Tree can use
  it or ignore it. Lower-risk than a target rewrite.
- **P6 — Reduce tree regularization.** GBDT shrinkage is the structural
  cause of leaf averaging. Smaller leaves / more leaves widens the
  prediction range. Cheap to test, modest expected gain.
- **P7 — Isotonic post-hoc calibration on `loc`.** Pure polish.
- **P10 — GPBoost / mixed-effects.** Already prototyped deterministically
  in OVERCONFIDENCE_INVESTIGATION and *failed* (top-volume bias −2.5);
  don't re-attempt naively.

## 6. Open questions for the research agent

Ranked by likely leverage:

1. **What IS the dominant compression cause for non-FGA SkewNormal
   markets, if not the level bias?** The level-bias hypothesis is now
   thoroughly rejected. Likely candidates:
   - GBDT leaf-averaging at the extremes (structural to gradient boosting,
     not specific to LightGBMLSS or the target encoding)
   - Volume-efficiency entanglement (PTS = FGA × FG% — single-target
     learning conflates both)
   - Feature gaps (the model lacks features that would let it
     extrapolate at the tails — e.g. pace, lineup, opposing-team adjusted)
   - Loss-function mismatch (CRPS may be reasonable on average but
     under-penalize tail errors)
   - The SkewNormal distribution shape itself (skew=0 start, scale
     bounded by `exp` response — could be limiting alpha learning)
2. **Can we make "FGA-like" vs "PTS-like" quantitative?** Currently the
   characterization is qualitative ("FGA is structural, PTS is form-
   driven"). A defensible quantitative criterion (e.g. ratio of per-player
   season-std to per-player-cohort-std; intraclass correlation; …) would
   let us either (a) selectively apply centered_additive_eb_meanyr_k10 on
   FGA-like markets without a one-off CLI dance, or (b) rule out per-
   market strategy selection if no clean cutoff exists.
3. **Does P3 (rate decomposition) generalize where centered-target
   failed?** Volume-shifting markets need separate volume and efficiency
   projections. The infrastructure is already there to add a
   `rate_decomposition` strategy to `baselines.py` and A/B it the same
   way. The volume column for NBA is `MIN` (already in features); other
   leagues need analogs.
4. **What does the per-decile bias structure look like AFTER the centered
   strategy on the markets that killed?** The summary table shows
   top-decile MAE % improvement, but the underlying decile tables
   (printed by `compression_eval` in single-market mode) carry more
   signal — is the bias signature *shifted* (e.g. now over-prediction at
   the low end) or just larger everywhere? Pulling the full decile tables
   for the killed markets (NBA_PTS, NBA_PA, NBA_PR) and comparing
   side-by-side against the baseline might surface what the model is
   actually doing differently.
5. **Is the live-path bug (Model Skew=NaN, `model_prob.py`, FGA) a
   bigger contributor than any training-side fix?** Filed in
   `docs/archive/gbdt_mean_regression_plan.md` as "tackle when diminishing returns
   hit" — but if training-side levers have run out (P1 family dead, P3/P5/
   P6 expected to be marginal), this becomes the next-most-leveraged
   investigation. OVERCONFIDENCE_INVESTIGATION §3.4 has the symptoms
   characterized.
6. **Sanity-check the `fantasy-points-prizepicks` 0% result.** It showed
   exactly 0% delta under Mean10 (suggesting the strategy was a no-op),
   which would mean the market falls in the NegBin/ZINB branch
   (`global_mean < 2.0`) — but a fantasy-points aggregate should have mean
   in the 15-30 range. Either the cached parquet for this market is
   degenerate, or there's a routing bug. Worth a 10-minute confirmation.

## 7. What's locked

- **`--target-strategy=ratio_meanyr` is the default and is staying.**
  Promoting `centered_additive_eb_meanyr_k10` would regress most markets.
- **Centered-target infrastructure stays in.** The registry, the
  brier_skill_score gate, the live-path test, the deterministic-mode
  hardening are all reusable for the next round of experiments. A
  researcher does not have to rebuild this scaffolding.
- **Default `meditate` invocation is byte-identical to pre-P1 production**
  — confirmed by `tests/integration/test_pipeline_target_strategy.py` which
  asserts a no-arg call equals the explicit `"ratio_meanyr"` call to
  `pd.testing.assert_frame_equal(check_exact=True)`. The branch is
  shippable to `devel` whenever the user is ready; the negative result
  doesn't rollback anything.

## 8. Artifact pointers

| What | Where |
|---|---|
| Branch | `claude/fix-gbdt-mean-regression-GcY1g` (32 commits ahead of `origin`, not pushed) |
| Master plan (durable) | `docs/operation_ship_75.md` (this doc historicized under `docs/archive/gbdt_mean_regression_plan.md`) |
| OVERCONFIDENCE_INVESTIGATION (lineage) | `docs/OVERCONFIDENCE_INVESTIGATION.md` |
| Phase-A original spec | `docs/superpowers/plans/2026-05-18-fga-fg3m-overconfidence-fix.md` |
| P0.5 determinism design | `docs/superpowers/specs/2026-05-19-p0.5-determinism-gate-design.md` |
| Strategy registry | `src/sportstradamus/training/baselines.py` |
| Pipeline dispatch | `src/sportstradamus/training/pipeline.py` (target_strategy plumbing throughout `train_market`) |
| Prediction-side mirror | `src/sportstradamus/prediction/model_prob.py` `_decode_skewnormal` helper |
| Offline harness | `src/sportstradamus/scripts/compression_eval.py` |
| Determinism gate test | `tests/integration/test_determinism_gate.py` |
| Live-path NaN guard | `tests/integration/test_centered_target_live_path.py` |
| Pipeline target-strategy test | `tests/integration/test_pipeline_target_strategy.py` |
| Unit tests for the strategies | `tests/test_centered_target.py` (28 tests) |
| A/B CSVs on disk | `src/sportstradamus/data/test_sets/deterministic/{ratio_meanyr,centered_additive_eb_meanyr_k10,centered_additive_mean10}/NBA_*.csv` |
| Compression eval run log | `src/sportstradamus/data/compression_eval_log.csv` (sparse; A/Bs were diff-mode and not auto-logged) |

## 9. How to re-run an A/B

```bash
# Baseline (production behavior, bit-reproducible)
poetry run meditate --deterministic --league NBA --target-strategy ratio_meanyr --reset-markets NBA:FGA

# Candidate (replace slug for any registered strategy in baselines.STRATEGY_SLUGS)
poetry run meditate --deterministic --league NBA --target-strategy <slug> --reset-markets NBA:FGA

# Path-wide verdict
for mkt in $(ls src/sportstradamus/data/test_sets/deterministic/ratio_meanyr/NBA_*.csv); do
  m=$(basename "$mkt" .csv)
  poetry run python -m sportstradamus.scripts.compression_eval \
    --baseline "src/sportstradamus/data/test_sets/deterministic/ratio_meanyr/${m}.csv" \
    --candidate "src/sportstradamus/data/test_sets/deterministic/<slug>/${m}.csv" \
    --strategy <slug>
done
```

`--reset-markets` only clears the SHAP filter for the named markets; it
does *not* scope which markets are trained — `meditate` always trains
everything for the league (a known quirk worth a fix later).

Wall time: ~10-15 minutes per `meditate --deterministic --league NBA` run
on cached parquet. `compression_eval` is ~1 sec per market.
