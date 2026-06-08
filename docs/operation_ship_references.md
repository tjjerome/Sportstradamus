# Operation Ship 75 — References

> **Preserves the load-bearing facts from the deprecated
> [`gbdt_mean_regression_plan.md`](archive/gbdt_mean_regression_plan.md) /
> [`gbdt_mean_regression_context.md`](archive/gbdt_mean_regression_context.md)**
> so future sessions don't lose context after the new plan
> ([`operation_ship_75.md`](operation_ship_75.md)) supersedes them. Full
> prose for any item below lives in the archived docs at the line
> citations. Research briefs cited as `/tmp/researcher_*.md` are
> distilled here; their full bodies are not preserved (scratch).

## Table of contents

1. [Phase P0 / P0.5 — offline harness + determinism gate](#1-phase-p0--p05--offline-harness--determinism-gate)
2. [Phase P1 — centered-target SkewNormal A/B](#2-phase-p1--centered-target-skewnormal-ab)
3. [Phase P2.A — `init_score` baseline DEAD](#3-phase-p2a--init_score-baseline-dead)
4. [Phase P2.B — HurdleZINB with derived-π gate](#4-phase-p2b--hurdlezinb-with-derived-π-gate)
5. [Stage 0 — Live-data instrumentation](#5-stage-0--live-data-instrumentation)
6. [Stage B1 — ZTNB refutation + routing diagnostics](#6-stage-b1--ztnb-refutation--routing-diagnostics)
7. [Stage B1.5 §7a — likelihood-vs-features pre-check](#7-stage-b15-7a--likelihood-vs-features-pre-check)
8. [Stage B1.6 — 2026-05-22 breadth research verdict](#8-stage-b16--2026-05-22-breadth-research-verdict)
9. [Stage A1 / A1.5 / A1.6 — ICC diagnostics + T5 kill](#9-stage-a1--a15--a16--icc-diagnostics--t5-kill)
10. [Branch / PR / commit refs](#10-branch--pr--commit-refs)
11. [Critical files map](#11-critical-files-map)
12. [Per-change-type inference-path checklist](#12-per-change-type-inference-path-checklist)
13. [Cross-league caveats](#13-cross-league-caveats)
14. [References [1]–[48]](#14-references)

---

## 1. Phase P0 / P0.5 — offline harness + determinism gate

**State:** ✅ done (PR #46).

- [`training/scorecard.py`](../src/sportstradamus/training/scorecard.py)
  + [`tests/golden/test_scorecard.py`](../tests/golden/test_scorecard.py)
  (promoted from the original `scripts/compression_eval.py` harness in `98e1b45`).
- Opt-in `meditate --deterministic`:
  - RNGs pinned (random / numpy / torch + `torch.use_deterministic_algorithms`)
  - Optuna swapped for `DETERMINISTIC_FIXED_PARAMS`
  - Input frozen to cached parquet
  - Persistent writes redirected to `data/{test_sets,models}/deterministic/`
- Gate runs on real cached `NBA_FGA.parquet` (4000 rows, ~5s) with
  stochastic LightGBM. Different seed → max-abs diff ~0.34; same seed
  bit-identical. Default `meditate` is byte-identical (deterministic
  mode is opt-in).
- Full prose:
  [`docs/archive/gbdt_mean_regression_context.md`](archive/gbdt_mean_regression_context.md)
  §Status / progress log row P0 / P0.5.

## 2. Phase P1 — centered-target SkewNormal A/B

**State:** ✅ done. **Result: FGA-only SHIP, family-wide KILL.**

Two centered-target variants A/B'd path-wide under `--deterministic`
vs `ratio_meanyr`:

| Strategy | Verdict | Numbers |
|---|---|---|
| `centered_additive_eb_meanyr_k10` (EB(MeanYr, K=10)) | FGA SHIPS, rest KILL | FGA +5.3% top-decile MAE, BSS +0.096→+0.112; PTS −3.5%, PA −4.1%, PR −2.9%, RA −2.2%, FG3A −3.8%, FGM −2.6%, MIN +3.7%, PRA +0.8%, REB +0.2% |
| `centered_additive_mean10` (trailing-10) | every market KILLs | FGA +4.6% (under 5% bar), PA −6.6%, PR −6.7% |

Count-family markets show exactly 0% delta under both (transform no-op
for NegBin / ZINB). Confirms the archived NBA-overconfidence investigation §3.2: the
SkewNormal level bias is not the dominant compression cause path-wide
regardless of baseline horizon. FGA is genuinely special (EB(MeanYr)
captures structural shot-volume; Mean10 too noisy).

**Default stays `--target-strategy=ratio_meanyr`.** Infrastructure
(`baselines.py`, registry, `offset_meta` pickle field, brier_skill
gate, live-path test) is **reusable** for future phases.

Full prose:
[`docs/archive/nba_overconfidence_investigation.md`](archive/nba_overconfidence_investigation.md)
(the centered-target negative result) and archived context P1 row.

## 3. Phase P2.A — `init_score` baseline DEAD

**State:** ✅ closed. **Result: byte-identical to plain NegBin.**

In-process FG3M spike: LightGBMLSS accepts per-row `init_score`
(length-2n flat `[log_EB, zeros]` per-parameter concat) without error,
but predictions are byte-identical to a plain NegBin fit, every decile.
Either LightGBMLSS overrides `init_score` with its own `start_values`,
or the 30-round deterministic fit converges regardless of start.

FG3M's plain-NegBin top-decile bias is already −0.013 — no meaningful
compression on the count-branch NegBin mean; the overconfidence was the
gate, which P2.B addresses. **DEAD** as a one-line `init_score`
transform.

## 4. Phase P2.B — HurdleZINB with derived-π gate

**State:** ✅ done. **Result: 6/8 NBA ZINB markets SHIP.**

New `meditate --zinb-mode=hurdle` (orthogonal to `--target-strategy`;
default `joint` byte-identical to pre-P2.B).
[`HurdleZINB`](../src/sportstradamus/hurdle.py):

- Calibrated binary classifier estimates `q = P(Y=0)`
- NegBin LightGBMLSS on `Y>0` supplies count shape
- Structural-inflation π derived via
  `π = clip((q − NB(0))/(1 − NB(0)), 0, 1)` (NOT the simpler
  `gate = 1 − p_nonzero` — corrected because downstream `fused_loc`
  treats `gate` as zero-inflation, not marginal P(Y=0))
- Returns `total_count` / `probs` / `gate` matching the ZINB contract
  so `model_prob` decode (~lines 252-257) is untouched
- Legacy pickles load via `getattr(model, "is_hurdle", False)`

**Path-wide A/B vs joint:**

| SHIP | Market | Improvement |
|---|---|---|
| ✅ | FG3M | +9.7%, BSS +0.115→+0.290 |
| ✅ | OREB | +44.9%, BSS +0.019→+0.109 |
| ✅ | PF | +19.2%, BSS −0.238→−0.002 |
| ✅ | TOV | +26.8%, BSS −0.049→+0.058 |
| ✅ | BLK | +40.4%, BSS +0.237→+0.299 |
| ✅ | BLST | +11.6%, BSS −0.002→+0.093 |
| KILL | FTM | +1.3% (under bar) |
| KILL | STL | global MAE +14.1% regression |

Joint ZINB had per-row catastrophic blowups in mid-deciles on
BLK / OREB / PF / BLST under deterministic mode (compression_ratio
24–5357×; predicted means up to 1437); the hurdle eliminates entirely
— global MAE drops 60–99% there.

**Default stays `--zinb-mode=joint`** (per-market routing is now wired
via `ship_config.json` object form with `zinb_mode` field; see
[`training/ship_config.py`](../src/sportstradamus/training/ship_config.py)
`resolve_cell_zinb_mode`).

Live-path test:
[`tests/integration/test_zinb_hurdle_live_path.py`](../tests/integration/test_zinb_hurdle_live_path.py)
asserts `π + (1−π)·NB(0) ≈ q` per-row (tol 0.02) + two-run
bit-identity. Determinism gate has parallel hurdle assertion.

## 5. Stage 0 — Live-data instrumentation

**State:** ✅ done (PR #46). Five deliverables shipped:

| # | Deliverable | Where |
|---|---|---|
| 0.1 | `compute_book_brier_skill_score` (book as reference, not chance) | [analysis.py](../src/sportstradamus/analysis.py) — 8 unit tests, hand-ref within 1e-6 |
| 0.2 | `_compute_live_metrics` Step 6 in `nightly.py`; writes `data/live_metrics_per_market.parquet` (locked 10-col / 2-window schema) | [nightly.py](../src/sportstradamus/nightly.py) — 6 round-trip tests |
| 0.3 | `scorecard --live-window N` + `_history_to_eval_frame` + per-league `_load_league_stats_lookup` | [training/scorecard.py](../src/sportstradamus/training/scorecard.py) — 8 new tests |
| 0.4 | `check_graduation` CLI: joins Gate 1 × Gate 2, `_classify_lifecycle` → {not-shipped, in-test, graduated, demoted} | [scripts/check_graduation.py](../src/sportstradamus/scripts/check_graduation.py) — 11 tests, 7 parametrized |
| 0.5 | `backfill_live_metrics` — walks history backwards, day-precision dedup | [scripts/backfill_live_metrics.py](../src/sportstradamus/scripts/backfill_live_metrics.py) — 5 tests |

All three always-on gates green; 30 new tests. Track-A/B graduation
lookups are now a parquet read.

## 6. Stage B1 — ZTNB refutation + routing diagnostics

**State:** ✅ done. **Result: ZTNB REFUTED; 0/23 cells route to
`hurdle_nb_ztnb`.**

**B1.1 — ZTNB likelihood fix: hypothesis REFUTED.** Zero-truncated NB
(`_ZeroTruncatedNB`) is correct in isolation (scipy-verified) but
**incompatible with the frozen derived-π hurdle decode.** On FG3M (a
P2.B SHIP), the ZTNB count component implies `NB(0) ≈ 0.41` vs
observed `q ≈ 0.31`, so on **65% of rows `NB(0) > q`** → π clips to 0
→ identity breaks (`test_zinb_hurdle_live_path` diff 0.136 vs 0.02
tol). The fix would regress the 6 P2.B SHIP markets.

`_ZeroTruncatedNB` kept as an unwired, test-covered block for the
MZINB head (B3) if it's ever revived.

**B1.2 — routing diagnostics:** read-only
`scripts/zinb_routing_diagnostics.py` writes
`data/zinb_routing/{LEAGUE}_diagnostics.parquet` for all 23 cells.

**Marginal split** (NB the conditional pass in §7 overrides this):

- 13 → CMP (underdispersed, var/mean ≤ 1.3): NBA STL, BLST, TOV,
  OREB, PF; WNBA BLK, STL, BLST, TOV, OREB; NFL tds, rushing tds,
  receiving tds.
- 10 → MZINB (overdispersed + mild inflation, var/mean 1.35–7.9):
  NBA FG3M, FTM, BLK; WNBA FG3M, FTM; NFL passing tds, qb tds,
  interceptions, sacks taken, passing first downs.

**The blanket ZINB label is wrong for ≥ 13 markets.**

## 7. Stage B1.5 §7a — likelihood-vs-features pre-check

**State:** ✅ done. **Verdict: FEATURES, not likelihood. Family build
DEFERRED.**

Poisson-GBM pre-check on 9 cells (NBA FTM/STL/TOV/FG3M, WNBA
STL/FG3M/FTM, NFL interceptions/rushing-tds). Each fit with a throwaway
in-memory Poisson GBM (LightGBM `objective="poisson"`, seed 1729, no
pickle saved) and scored against the production NB/ZINB baseline.

**Findings:**

1. **Top-decile compression is distribution-family-INVARIANT.** Poisson
   top-decile CR (`std(pred)/std(actual)`) 0.16–0.35 vs production
   NB/ZINB 0.12–0.37 — indistinguishable; both severely compressed.
   A model with **no overdispersion freedom at all** compresses the
   high-mean tail just as hard. Textbook ensemble-tree dynamic-range
   bias ([3], [4], [30]) — the regularized leaf-average itself.
   **CMPμ / MZINB / hurdle re-parameterize dispersion or the zero gate
   and leave the boosted mean head untouched.**

2. **Only 2/9 cells pass ≥ 5%, and both are upward mean-bias, not
   dispersion.** NBA FG3M (+6.3%) and NFL rushing-tds (+10.3%) "SHIP"
   but show uniform upward bias — the production model over-predicts,
   worst at the *bottom* decile (FG3M predicts 0.68 threes where actual
   is 0.20). Trivial bias re-centering recovers 41–47% of the gain with
   no family change.

3. **No CMPμ candidate among NBA / WNBA cells.** Conditional Dunn–Smyth
   RQR variance collapses marginal var/mean toward 1 once the ~280-col
   feature set conditions the mean: STL 1.17→1.08, TOV 1.16→1.04, WNBA
   STL 0.99→1.00 — **equi-dispersed**, failing the
   `CMPμ iff conditional < 0.90 AND marginal < 1.0` gate.

**Pivot:** Track B's next step is a ~1–2 week feature/bias track, not
the family build. The 13-cell `cmp` label and 10-cell inflated label
from B1.2 are both **unsupported as a family-build trigger** by this
evidence.

**Re-entry condition** (per cell): build CMPμ / marginalized hurdle
**only** on a cell that still kills after the cheap fixes AND
conditional RQR variance < 0.70 AND Poisson GBM tracks the top decile
while NB compresses.

## 8. Stage B1.6 — 2026-05-22 breadth research verdict

**State:** ✅ research done; build pending. Brief was
`/tmp/researcher_breadth_75.md`. New references [43]–[48] added.

**Bottleneck.** Every failing cell is a low-mean count market and fails
on the absolute bias gates (bottom-quartile over-prediction +
top-decile under-prediction), not the BSS floor. Family swaps are dead
per §7a — only two layers touch the mean head: **post-hoc correction
of the prediction** and **features that hand the tree the per-player
level it averages away.**

**Ranked levers:**

1. **Post-hoc per-decile / isotonic / affine mean-bias correction** —
   RANK 1. The *only* method that moves both gated bands by
   construction (a monotone map pulls bottom-band over-prediction down
   and top-band under-prediction up at once). Prefer **isotonic** where
   miscalibration is curved; plain affine / ROE captures the uniform
   part. **Per-decile multicalibration** ([44]) is the formal frame
   but a fallback — trained GBDTs are often already near-multicalibrated
   ([45]).

2. **Leakage-safe expanding-mean / EB-shrunk player feature** — RANK 2
   (the deferred P5). `groupby(player_id).expanding().mean().shift(1)`
   for stat and stat × opponent. Optionally James-Stein / EB-shrunk
   (Efron–Morris [47]). Regularized target encoding beats one-hot
   ([43]).

3. **Opponent-defense × player + blowout / garbage-time** — RANK 3.
   Narrowest, most build cost; targets FG3M low-volume overshoot.

4. **BSS-floor gate-tuning toward −0.02** — mop-up only. Policy lever
   for the ~1 bias-passes/BSS-fails cell per league (NBA PF, NFL
   interceptions).

**Per-league nearest-miss targets** (the cells closest to passing):

- **NBA (need +3 at the time of research):** TOV / BLST / OREB / PF
  (STL, FG3M backups) — near-equi-dispersed conditionally, want
  post-hoc, not family.
- **WNBA (need +4):** TOV / BLST / OREB / STL — same family as NBA
  cheap wins but **re-validate, never assume** (half the games:
  ~40/season makes isotonic tails and expanding-mean feature noisier;
  fall back to affine ROE / strongly-shrunk player mean).
- **NFL (need +2):** rushing-tds + receiving-tds via **affine ROE,
  not isotonic / per-decile** (too few positive events at those means:
  interceptions 0.5, TD zero-rates 0.78–0.92 [48]). Keep rushing-yards,
  qb-tds, sacks-taken on a per-position bench.

**Flag:** NBA / WNBA AST are SkewNormal, not count cells — post-hoc +
player feature apply, but count-family reasoning does not.

> **Note (2026-05-23):** Operation Ship 75 uses a stricter scorecard
> (g1–g5 including IQR + ECE) than the Tier-0 the research verdict
> targeted. The verdict's gap counts (NBA +3, WNBA +4, NFL +2) reflect
> the old absolute-bias gates only. The new scorecard shows wider gaps
> (NBA +9, WNBA +4, NFL +10) because G4 (IQR ratio) and G5 (ECE) are
> failing many additional cells. Ship 75 Step 0 audits G4 first — that
> may flip many cells without invoking the post-hoc lever at all.

## 9. Stage A1 / A1.5 / A1.6 — ICC diagnostics + T5 kill

**State:** ✅ done. **Result: family clusters AMBIGUOUS; T5 KILLED
wholesale; A2 pivots to T3 tail head (deferred behind breadth).**

**Stage A1 — ICC₁ table (36 cells).** Read-only
`scripts/icc_diagnostics.py` (console `icc-diagnostics`) writes
`data/icc/{NBA,WNBA,NFL}_icc.parquet`.

**Routing verdict:** 25 ambiguous, 10 eb_centering, 1 tail_extension.

- NBA (ICC 0.27–0.51): only PA 0.514 → eb, DREB 0.274 → tail; rest
  ambiguous. **FGA 0.489**, PTS 0.473.
- WNBA (0.37–0.57): slightly *higher* than NBA, not noisier
  (4-season pooling stable).
- NFL (0.41–0.79): qb-yards 0.790, carries 0.666, targets 0.507,
  rushing-yards 0.502 → eb. After A1.6 position-split cleanup:
  qb-yards 0.790 → 0.423 (the 0.790 was a QB-vs-RB
  between-position artifact).

**ICC does NOT predict the P1 EB ship/kill** — FGA SHIPPED at 0.489
while PA (highest NBA ICC 0.514) KILLED.

**Stage A1.5 — factor-ICC de-risk of T5.**

Band verdict (median ICC(volume) − median ICC(efficiency)):

- NBA gap +0.232 → MIXED (volume clause FAILED — 0/3 NBA volume
  factors reach 0.5)
- WNBA gap +0.456 → CONFIRMED but low-confidence (single computable
  efficiency factor)
- NFL gap +0.291 → CONFIRMED

**Literature OVERRIDES the band verdict.** Goodman's exact
variance-of-products [1]:
`CV²(XY) = CV²(X) + CV²(Y) + CV²(X)·CV²(Y) ≥ CV²(X) + CV²(Y)`.
On actual NBA top-mean-decile PTS player-seasons, direct PTS modeling
gives within-player-season CV **0.334**, but recomposed
FGA × (PTS/FGA) gives CV **0.423 — +27% predictive-variance
inflation** on the priced cell. Recomposition discards the structural
negative covariance and re-inflates the tail.

**T5 KILLED as a wholesale multiplicative architecture.** A2 pivots
to **T3** (spliced / Pareto-tail or normalizing-flow head) as the
primary build (deferred behind 75% breadth).

**Stage A1.6 — NFL position-split + WNBA test fix.**

- `NFL_MARKET_POSITIONS` constant + `_market_position_filter` hook
  (no-op default in [`stats/base.py`](../src/sportstradamus/stats/base.py),
  NFL override in [`stats/nfl.py`](../src/sportstradamus/stats/nfl.py)).
- Cached parquets cleaned via
  [`scripts/prune_nfl_matrix_positions.py`](../src/sportstradamus/scripts/prune_nfl_matrix_positions.py):
  passing-yards 15000 → 2646 rows, mean 38.1 → 215.9.
- New Stage A4 entry **T11**: per-position model-split bias experiment
  (deferred to Ship 90 — see [`operation_ship_90.md`](operation_ship_90.md)).

## 10. Branch / PR / commit refs

- **Active branch (Operation Ship 75):** `model-research`.
- **Prior PR:** #46 → `devel`; HEAD before Ship 75 rework `fbec3cc`.
- **Earlier breadth-led docs rework HEAD:** `6e913b1`.
- **Latest shipped via old plan:** 13 NBA + 10 WNBA + 13 NFL baselines
  locked in `data/ship_config.json` (`b5d2609` / `c9fcf01` / `fbec3cc`).
  P2.B HurdleZINB (`cee5625`). P1 strategy infrastructure (`1d0e65e`).
- **Note:** the locked-baseline numbers above used the *old* Tier-0
  spec (absolute bias + BSS ≥ 0). Under the new five-gate scorecard
  ([`training/scorecard.py`](../src/sportstradamus/training/scorecard.py),
  adds G4 IQR ratio and G5 ECE), some of these no longer ship. See
  Step 0 audit in
  [`operation_ship_75.md`](operation_ship_75.md).

## 11. Critical files map

Copy of "Critical files" table from the deprecated plan, line numbers
last-verified against `fbec3cc` (HEAD before Ship 75 rework):

| File | Role | Key lines |
|---|---|---|
| [`src/sportstradamus/training/pipeline.py`](../src/sportstradamus/training/pipeline.py) | target build, dist select, training, denorm, test_set dump | 245–324 (branch/target), 328 (`lgb.Dataset` init_score injection), 341 / 394–409 (`set_model_start_values`), 345–346 (MeanYr monotone), 348–368 (Optuna search space), 439–452 (SkewNormal denorm), ~960 / 981 (test_set dump) |
| [`src/sportstradamus/training/report.py`](../src/sportstradamus/training/report.py) | diagnostics → `model_stats.parquet` (+ `.csv` mirror) | `ev_meanyr_corr` / `result_meanyr_corr` (~850), `write_model_stats` |
| [`src/sportstradamus/stats/base.py`](../src/sportstradamus/stats/base.py) | baseline features + target; inference-time mirror lives here | 597 (`get_stats`), 676–702 (`MeanYr`, `Mean10`, `*_Ratio`), 1005 / 1011 / 1082 (`Result`) |
| [`src/sportstradamus/stats/nba.py`](../src/sportstradamus/stats/nba.py) | NBA `MIN`, `USG_PCT`, per-48 stats | 127–135, 359, 366 |
| [`src/sportstradamus/helpers/distributions.py`](../src/sportstradamus/helpers/distributions.py) | `set_model_start_values`; `fused_loc` (book blend) | 425–504 |
| [`src/sportstradamus/skew_normal.py`](../src/sportstradamus/skew_normal.py) | custom SkewNormal (location-scale, supports negatives) | 30–199 |
| [`src/sportstradamus/hurdle.py`](../src/sportstradamus/hurdle.py) | HurdleZINB (Stage 2 ZTNB lives here, unwired) | ~201 (NegBin loss) |
| [`src/sportstradamus/training/scorecard.py`](../src/sportstradamus/training/scorecard.py) | P0 harness (promoted from `scripts/compression_eval.py` in `98e1b45`) — decile table, compression ratio, run log, diff verdict, gate scorecard, `supersede_verdict` | 84–149 (gate constants), 927 (`_gate4_iqr_spread`), 1110 (`gate_row`), 1295 (`apply_thresholds`), 1389 (`write_gate_scorecard`), 1641 (`supersede_verdict`), 1989 (`main`) |
| [`src/sportstradamus/prediction/model_prob.py`](../src/sportstradamus/prediction/model_prob.py) | Live-path confound — shipped strategies must survive end-to-end | SkewNormal decode ~276, NegBin/ZINB decode 259–272, hurdle dispatch 205, `fused_loc` w≈0.9 blend, `temperature` ≈ 1.37 |
| [`src/sportstradamus/training/ship_config.py`](../src/sportstradamus/training/ship_config.py) | per-cell `ship_config.json` loader + `resolve_cell_strategy` + `resolve_cell_zinb_mode` + `WITHHELD` | 37 (`WITHHELD`), 90 (`load_ship_config`), 117 (`resolve_cell_strategy`), 145 (`resolve_cell_zinb_mode`) |
| [`src/sportstradamus/training/baselines.py`](../src/sportstradamus/training/baselines.py) | strategy registry | 243–268 (`_STRATEGIES`), 272 (`STRATEGY_SLUGS`), 278 (`ZINB_MODES`) |

## 12. Per-change-type inference-path checklist

Every change must land its inference-side mirror in the same PR before
promotion. Gate 1 lets a change into the test window; this checklist is
what makes that window safe. Copy from the deprecated plan §Inference
path:

| Change type | Inference-side work | Precedent |
|---|---|---|
| **Training-only** (loss change, monotone constraint, ZTNB likelihood, per-parameter Optuna, sample reweighting) | None. Output schema unchanged. | Stage B1 ZTNB attempt: only loss changed. |
| **New target / baseline strategy** | Inverse decode in `model_prob.py` `_decode_skewnormal` via `baselines.STRATEGY_REGISTRY[strategy].decode_loc/decode_scale`; matching `*_Ratio` feature in `get_stats`; `target_strategy` + `offset_meta` keys round-trip. | P1 `centered_additive_*`. |
| **New distribution head** (T3 spliced/Pareto, MZINB, CMPμ, PGBM, MEGB, gbex) | (a) new decode block in [`model_prob.py:259-272`](../src/sportstradamus/prediction/model_prob.py); (b) `get_ev` / `get_odds` / `fused_loc` / `set_model_start_values` accept the new `dist`; (c) `dist` in `_build_filedict` + legacy fallback; (d) new live-path test mirroring [`test_zinb_hurdle_live_path.py`](../tests/integration/test_zinb_hurdle_live_path.py). | P2.B HurdleZINB. |
| **Post-hoc calibration object** (isotonic on loc, CQR/LCMQR, **`bias_correction` from Ship 75 Step 1**) | Pickle as a new key (`isotonic` / `cqr` / `temperature` / `bias_correction` precedent); load in `model_prob`, apply after decode (before/after `fused_loc` per what's calibrated); byte-identical round-trip test. | `temperature` field in `_build_filedict`. |
| **New player-level feature** (Ship 75 Step 2 expanding-mean, Step 3 opponent-defense) | Column in BOTH `get_training_matrix` and `get_stats`, computed identically, leakage-safe; same dtype/index; add to `feature_filter.json` whitelist. | `MeanYr` / `Mean10` / `*_Ratio` (`base.py:676-702`), `test_meanyr_mean10_leakage.py`. |
| **Multi-head factorization** (T5, deferred) | `prophecize` loads N pickles/market; `model_prob` Monte Carlos; `fused_loc` multi-output blend; new `factor_pickles: dict[str, Path]` on parent pickle; new live-path test. **Largest inference-side change in plan.** | None in-repo. |
| **Different model class** (CatBoost ordered TS, MEGB, GPBoost) | New `is_catboost`/`is_gpboost` flag; `model_prob` + `prediction/__init__.py` load path branch; determinism gate extended; adapt if no LSS `predict(pred_type="parameters")` API. | P2.B `is_hurdle`. |

**Hard ship gate:** any change requiring inference-side work must have
a passing live-path integration test under `tests/integration/`
**before promotion to production**.

**Pickle-schema discipline:** every new field needs (1) reader site in
`model_prob.py`, (2) legacy default fallback
(`filedict.get("new_key", default)`), (3) round-trip test asserting
byte-identical predictions. Current fields written by
`_build_filedict`: `model`, `step`, `stats`, `metrics`, `diagnostics`,
`params`, `distribution`, `cv`, `std`, `temperature`, `dispersion_cal`,
`weight`, `r_book`, `hist_gate`, `shape_ceiling`, `normalized`,
`offset_meta`, `target_strategy`, `zinb_mode`, `is_hurdle`,
`expected_columns`. Ship 75 adds: `bias_correction`.

## 13. Cross-league caveats

Copy from the deprecated plan §Cross-league caveats. Read before
running any cross-league A/B:

1. **NFL sample sizes are an order of magnitude smaller than NBA**
   (~17 vs ~82 games/player/season). EB(MeanYr, K=10) is aggressive
   shrinkage at that size — re-derive `K = σ²_within / σ²_between`
   per league. NFL K may be much lower (or EB transform may fail on
   form-volatile NFL markets).
2. **NFL stats are position-locked.** `Player position` is already a
   categorical (`pipeline.py:_step_build_splits`). Cross-player models
   per market may not transfer cleanly (QB vs WR don't share "passing
   yards"); per-position scoping shipped in Stage A1.6 for NFL.
3. **WNBA shares NBA's structure but has half the games / season.**
   EB K=10 probably fine but verify. The per-100-poss factorization
   (T5-basketball) is KILLED so the transfer point is moot. WNBA has
   no `FGM` or `FG3A` markets (confirmed against the `stat_dist`
   config view of `stat_meta.json`);
   `FTM_per_FGA` + `FG3M_per_FGA` are the available efficiency
   factors for ICC diagnostics.
4. **The scorecard A/B harness is league-agnostic but file
   paths are league-specific.** Cached parquets at
   `data/training_data/{LEAGUE}_{market}.parquet`; deterministic test
   sets at `data/test_sets/deterministic/{strategy}/{LEAGUE}_{market}.csv`.
5. **Determinism gate currently covers NBA only.** Two
   `test_deterministic_mode_*` tests use NBA_FGA + NBA_FG3M
   ([`tests/integration/test_determinism_gate.py:37,102`](../tests/integration/test_determinism_gate.py)).
   Before a cross-league change, add parallel assertions on WNBA_FGA
   + WNBA_FG3M + a representative NFL market — else the cross-league
   verdict is noise.
6. **For low-mean NFL markets** (interceptions ~0.5, sacks ~1.5),
   the ZINB diagnostic formulae may need to compute on `log(1+Y)`
   [22]; the asymptotic Vuong degrades badly at very low means.
   Wilson-Einbeck's non-asymptotic test should be the only one trusted
   for NFL interceptions / sacks.
7. **Two-track parallelism holds across leagues.** Track A and B
   touch different distribution branches and markets; workable in
   parallel per league. Shared resource is the read-only
   `scorecard` harness.
8. **Low-mean conditional-dispersion diagnostics need a non-Pearson
   estimator — trust the Dunn–Smyth RQR over Pearson at mean ≲ 0.11.**
   In B1.5 the df-corrected Pearson and RQR variance diverged at very
   low NFL means (rushing-tds: Pearson 0.57 vs RQR 0.96; interceptions:
   Pearson 1.38 vs RQR 1.04). A future low-mean NFL pass should
   bootstrap the RQR variance and/or use deviance-based dispersion.
9. **Post-hoc bias correctors at NFL count means must be affine ROE,
   not isotonic / per-decile.** At interceptions ~0.5 and TD zero-rates
   0.78–0.92, isotonic tails and per-bin (multicalibration) correctors
   overfit; percent-calibration error is worst in low-base-rate groups
   [48]. Reserve isotonic / per-decile for the higher-mean NBA / WNBA
   count cells.

## 14. References

Renumbered from the deprecated context doc.

[1] Goodman, L. A. (1960). On the exact variance of products. *JASA* 55(292), 708–713.
[2] Buchanan, S., et al. Field-goal-percentage reliability in basketball — citation TBD; full DOI in archived context.
[3] Mentch, L., Zhou, S. (2020). Randomization as regularization. *JMLR* — on tree-ensemble dynamic-range bias.
[4] Wager, S. (2020). Subsampling and ensembling regression trees. Statistics & Probability Letters — leaf-averaging compression.
[5] Huang, A. (2017). Mean-parameterized Conway-Maxwell-Poisson regression for dispersed counts. *Statistical Modelling* 17(4-5), 359–380.
[6] Casella, G., Berger, R. L. (2002). *Statistical Inference* — empirical-Bayes shrinkage K.
[7] Aigner, D. J., Hirschberg, J. G. (1985). Aggregate vs disaggregate forecasts.
[8] Sprangers, O., et al. (2021). Probabilistic Gradient Boosting Machines. *KDD*.
[9] Sigrist, F. (2021). GPBoost critique — mixed-effects boosting bias.
[10] Dunn, P. K., Smyth, G. K. (1996). Randomized quantile residuals. *J Comp Graph Stat*.
[11] Daly, F., Gaunt, R. E. (2016). The Conway-Maxwell-Poisson distribution — normalizing-constant truncation.
[12] Sellers, K. F., Shmueli, G. (2010). A flexible regression model for count data based on the CMP distribution.
[13] Stein, M. L. (2013). Limitations on low-rank approximations — sampling variability of efficiency rates.
[14] Romano, Y., Patterson, E., Candès, E. (2019). Conformalized quantile regression. *NeurIPS*.
[15] Sesia, M., Candès, E. (2020). Localized CMQR (LCMQR).
[16] Sellers, K. F., Borle, S., Shmueli, G. (2012). The CMP distribution — equi-dispersion behavior.
[17] Prokhorenkova, L., et al. (2018). CatBoost: unbiased boosting with categorical features. *NeurIPS*.
[18] Velthoen, J., et al. (2023). gbex: gradient boosting for extremes.
[19] Beutel, A., et al. (2019). Fairness-aware GBT (FAGTB).
[20] Goodfellow, I., et al. (2014). GANs — adversarial framework.
[21] Sprangers, O., Schelter, S. (2022). PGBM — probabilistic gradient boosting.
[22] Wilson, P., Einbeck, J. (2018). Zero-inflation hypothesis tests for count data.
[23] Wilson, P. (2015). Bootstrap zero-inflation test.
[24] Wilson, P. (2015). Schwarz-corrected Vuong + the nested-at-γ=0 critique.
[25] Smith, V. A., Preisser, J. S. (2014). Marginalized hurdle models.
[26] Long, D. L., Preisser, J. S. (2015). Marginalized hurdle design choices.
[27] Preisser, J. S., Stamm, J. W. (2010). Marginalized ZINB (MZINB).
[28] Basketball-Reference league-wide 3PA / 3P% history.
[29] Kuhn, M., Johnson, K. (2019). Feature engineering and selection — protocol-lock-before-refit.
[30] Friedman, J. H. (2001). Greedy function approximation — leaf-averaging mechanism.
[31] Feng, C., et al. (2020). Randomized quantile residuals for count GOF.
[43] Pargent, F., et al. (2022). Regularized target encoding outperforms traditional methods for high-cardinality categorical features. *Comp Stat* — DOI 10.1007/s00180-022-01207-6.
[44] Globus-Harris, I., et al. (2023). Multicalibration as boosting. *arXiv:2301.13767*.
[45] Hansen, K., et al. (2024). Are GBDTs already multicalibrated? *arXiv:2406.06487*.
[46] Reserved.
[47] Efron, B., Morris, C. (1973). Stein's estimation rule and its competitors — James-Stein / empirical-Bayes shrinkage.
[48] Roelofs, R., et al. (2022). Mitigating bias in calibration error estimation — low-base-rate sensitivity.

*Added by the full-distribution audit (marginal-breadth levers folded into [`operation_ship_75.md`](operation_ship_75.md) §5):*

[49] Czado, C., Gneiting, T., Held, L. (2009). Predictive model assessment for count data. *Biometrics* 65(4), 1254–1261.
[50] Hallin, M., Ley, C. (2014). Skew-symmetric distributions and Fisher information — the double sin of the skew-normal. *Bernoulli* 20(3), 1432–1462. *(arXiv:1209.4177)*
[51] Kuleshov, V., Fenner, N., Ermon, S. (2018). Accurate uncertainties for deep learning using calibrated regression. *ICML*. *(arXiv:1807.00263)*
[52] Henzi, A., Ziegel, J., Gneiting, T. (2021). Isotonic distributional regression. *JRSS-B* 83(5), 963–993.
[53] Marx, C., et al. (2022). Modular conformal calibration. *(arXiv:2206.11468)*
[54] Genest, C., Zidek, J. (1986). Combining probability distributions — a critique and annotated bibliography. *Statistical Science* 1(1), 114–135.
[55] Clarke, S., Kovalchik, S., Ingram, M. (2017). Adjusting bookmaker's odds to allow for overround. *Am. J. Sports Science* 5(6), 45–49.
[56] Ranjan, R., Gneiting, T. (2010). Combining probability forecasts — the beta-transformed linear pool. *JRSS-B* 72(1), 71–91.
[57] Gneiting, T., Ranjan, R. (2013). Combining predictive distributions. *Electronic Journal of Statistics* 7, 1747–1782.
[58] Hora, S. C. (2004). Probability judgments for continuous quantities — linear-pool dispersion. *Management Science* 50(5), 597–604.
[59] Gebetsberger, M., et al. (2018). Estimation methods for nonhomogeneous regression — CRPS vs maximum likelihood. *Monthly Weather Review* 146(12), 4323–4338.
[60] Chung, Y., et al. (2021). Beyond pinball loss — quantile methods for calibrated uncertainty. *NeurIPS*.
[61] Arellano-Valle, R. B., Azzalini, A. (2008). The centred parametrization for the multivariate skew-normal distribution. *J. Multivariate Analysis* 99, 1362–1382.
[62] Jones, M. C., Pewsey, A. (2009). Sinh-arcsinh distributions. *Biometrika* 96(4), 761–780.
[63] Harris, T., Yang, Z., Hardin, J. (2012). Modeling underdispersed count data with generalized Poisson regression. *Stata Journal* 12(4), 736–747.
[64] Efron, B. (1986). Double exponential families and their use in GLM regression. *JASA* 81(395), 709–721.
[65] Gelman, A., Hill, J. (2007). *Data Analysis Using Regression and Multilevel/Hierarchical Models.* Cambridge UP.
[66] Hollmann, N., et al. (2025). Accurate predictions on small data with a tabular foundation model (TabPFN v2). *Nature* 636, 319–326.
[67] Grinsztajn, L., Oyallon, E., Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on tabular data? *NeurIPS*. *(arXiv:2207.08815)*
[68] McElfresh, D., et al. (2023). When do neural nets outperform boosted trees on tabular data? *NeurIPS*. *(arXiv:2305.02997)*
[69] Arnold, S., Walz, E.-M., Ziegel, J., Gneiting, T. (2024). Decompositions of the mean CRPS. *Electronic Journal of Statistics* 18, 4992–5044.
[70] Chernozhukov, V., Wüthrich, K., Zhu, Y. (2021). Distributional conformal prediction. *PNAS* 118(48). *(arXiv:1909.07889)*
[71] López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley (ch. 7 purging/embargo, ch. 12 CPCV).

---

> **For full prose**, see the archived
> [`gbdt_mean_regression_plan.md`](archive/gbdt_mean_regression_plan.md)
> and
> [`gbdt_mean_regression_context.md`](archive/gbdt_mean_regression_context.md).
> Line citations in the tables above are accurate as of HEAD `fbec3cc`.
