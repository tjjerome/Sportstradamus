# Lane record — mean corrector stage order: recalibrate the pool, not its inputs

Status: **closed, shipped.** NFL tds passes all six gates on a full-HPO run for the first time.
The change is one market-agnostic rule with no per-cell branching. Four of five live `MEAN_STAGE`
controls retain `ship`; MLB hits allowed drops Gate 4, and its re-sweep found three corners that
ship under the new order but promoted none — the supersession baseline predates the change.
Context spine: CLAUDE.md, [docs/ARCHITECTURE.md](../ARCHITECTURE.md),
[model_improvement_track.md](model_improvement_track.md) §6.1 Rung A.
Research brief (pooling literature, cohort screen, per-cell counterfactuals):
[researcher_count_blend_location.md](../archive/researcher_count_blend_location.md).
Predecessor lane: [count_mean_calibration.md](count_mean_calibration.md).

## The question

The predecessor lane proved at full HPO that a mean-stage corrector converts Gate 4 on NFL tds
(`g4_pit_ks` 0.0700 → 0.0431) and lifts the *model* CITL to 0.92, while the *served* CITL topped out
at 0.86 — so the cell still failed Gate 6 and stayed `withheld`. This lane asked whether that
residual is the blend structure or the corrector's **position** relative to the blend.

## The change — one rule

> The mean-stage corrector is fit on the **fused served** validation mean against `Result` and
> applied to the **fused served** test/live mean — after fusion, before dispersion calibration and
> before the temperature fit.

`fused_loc` pools a count location geometrically (Genest & Zidek 1986), which is the exact
density-level logarithmic opinion pool for a count family and correct as an operator. But a LOP is
**not mean-preserving**: by AM–GM the pooled mean sits below the arithmetic one by `ρ^(1−w)` where
`ρ = μ_book/μ_model`. Correcting before the pool therefore hands the pool a corrected mean and gets
only `ρ^(1−w)` of the correction back — on NFL tds, 0.855 of it. This is precisely the arrangement
Ranjan & Gneiting (2010, doi:10.1111/j.1467-9868.2009.00726.x) Thm 1 forbids: a non-trivial pool of
calibrated components is itself uncalibrated, so the object to recalibrate is the **combination**.

"Served" means the mean the gates score: `(1−π_blend)·base` on a zero-inflated family, `base`
otherwise. Fitting on the served mean and dividing the gate back out before re-encoding reconciles
the fit target with the gate target, which closes `posthoc_mean_miscontracted_on_gated` by
construction rather than leaving it as policy.

`training/posthoc.py:correct_fused_mean` is the single entry point both `pipeline` and `model_prob`
call, so the two stages cannot drift. `fused_loc`, `_MODEL_WEIGHT_MAX`, the book leg and the
dispersion pool are untouched, and the pickle contract is unchanged.

**CSV contract change:** `EV` now carries the *uncorrected* model mean and `Blended_EV` carries the
correction. Every outcome gate reads `Blended_EV` (`report._SHIP_PRED_COL`); the only `EV` readers
are the reported-only `g2/g3_star_z_raw` and `_pred_cdf_pmf`'s Gamma branch (no Gamma cells exist).

## Verdict — NFL tds ships

Full HPO, 300 Optuna trials, matrix `7918c1b8` — byte-identical to the predecessor lane's `e1b`
arm, so this is a clean single-axis read of the stage order.

| | ZINB incumbent | pre-fusion `isotonic_mean` (`e1b`) | **post-fusion `isotonic_mean`** |
|---|---|---|---|
| `g6_citl_ci_hi` (≥0.97) | 0.8647 ✗ | 0.8983 ✗ | **1.0956 ✓** |
| `g4_pit_ks` (≤0.0500) | 0.0592 ✗ | 0.0431 ✓ | **0.0242 ✓** |
| `g4_tail_pit_ks` | 0.0569 | 0.0384 | **0.0196** |
| `g5_ece_debiased` (<0.075) | 0.0065 | 0.0179 | **0.0083** |
| `g1_brier_diff_ci_hi` (<0.005) | −0.0227 | −0.0234 | **−0.0221** |
| `brier_skill_score` | 0.1779 | 0.1654 | **0.1641** |
| `g2_star_z` / `g3_bench_z` (<0.5) | — | — | **0.0025 / 0.0651** |
| `roc_auc` (served) | 0.7336 | 0.7227 | **0.7119** |
| `ship` | ✗ | ✗ | **✓** |

The fusion haircut inverts: **0.855 → 1.281**. Served CITL 0.9314 against a model-only 0.7271 —
the pool now carries the correction instead of eating it.

Gate 6 is partly self-satisfying on a `MEAN_STAGE` cell, exactly as pre-registered: the corrector's
job is to equalise `Σpred` and `Σy`, which is what the CITL leg measures. The independent evidence
that the move is structural is that **Gate 4 and Gate 5 improve simultaneously with Gate 1 flat** —
a corrector fit on the mean cannot target the randomized-PIT KS.

Persisted corner: `dist: NegBin`, `posthoc: isotonic_mean`, `count_dispersion_objective: crps`,
`blending: nll`, `target_normalization: none`, `shipped: devel`.

## Controls — 4 pass, 1 drop

Each control ran with no CLI control flags, resolving its recipe from `stat_meta.json`, on its own
pinned matrix at full HPO.

| cell | dist / slug | `w` | `g4_pit_ks` before → after | `roc_auc` before → after | verdict |
|---|---|---|---|---|---|
| NFL passing tds | NegBin / `isotonic_mean` | 0.90 | 0.0498 → 0.0452 | 0.548 → 0.476 | **ship** |
| NBA BLST | DPO / `roe_mean` | 0.70 | 0.0191 → 0.0210 | 0.513 → 0.630 | **ship** |
| NBA FG3M | DPO / `isotonic_mean` | 0.77 | 0.0257 → 0.0156 | 0.602 → 0.601 | **ship** |
| WNBA FG3M | DPO / `isotonic_mean` | 0.90 | 0.0354 → 0.0262 | 0.618 → 0.679 | **ship** |
| MLB hits allowed | SkewNormal / `isotonic_mean` | 0.12 | 0.0323 → **0.0540** | 0.516 → 0.579 | **drop** |

The four `w = 1.0` cells (NFL qb yards, WNBA BLST/DREB/STL) reproduce **every gate value
bit-identically** on a deterministic pre/post run, as the identity pool requires. Their `EV` and
`P_standalone` columns move by design; `Blended_EV` differs on 5 rows out of 8 890, all at the
degenerate `mean = 0` boundary where the isotonic corrector's clip range shifts by one knot. No gate
moves.

NFL passing tds' `roc_auc` was investigated and cleared: the **uncorrected** model mean on that arm
scores 0.4831 and production's model leg scores 0.4964, so the cell has no ranking power in either
arrangement and the corrector did not flatten it. Its `brier_skill_score` stays at +0.146 — this
cell beats the book on calibration, never on ranking.

## Why MLB hits allowed dropped — isotonic quantizes the served location

`isotonic_mean` is a step function. Before the change it acted on the model leg and the book leg
re-separated the plateaus inside the pool; after it, the plateaus **are** the served mean.

| dump | `EV` distinct | `Blended_EV` distinct |
|---|---|---|
| MLB hits allowed, old order | 20 / 1066 | 1066 / 1066 |
| MLB hits allowed, **new order** | 1066 / 1066 | **20 / 1066** |
| NFL qb yards (`w = 1`, no pool) | 23 / 287 | 23 / 287 |

On a continuous family a quantized location clumps the randomized PIT, which is what Gate 4 scores —
hence 0.0323 → 0.0540. Count families discretize in `get_odds` anyway, so the same corrector
*improved* Gate 4 on all three DPO/NegBin controls. MLB hits allowed is the cohort's worst case: its
book carries 88% of the served location, so pre-fusion the plateaus were invisible.

The lever is generic and already per-cell — `roe_mean` is affine and strictly monotone, so it never
quantizes, and the sweep chooses between the two per cell. Nothing here needs a threshold or a
family branch.

### The re-corner found three shipping corners and promoted none

`ship sweep --league MLB --market "hits allowed" --confirm` at the normal §7.1 bar: 48 board
corners in 58 min, then a full-HPO confirm walk over four nominees on matrix `4e933457`.

| nominee | recipe | `g4_pit_ks` | six gates | `roc_auc` |
|---|---|---|---|---|
| board +0.095 | SkewNormal · `centered_additive_eb_meanyr_k10` · crps · centered · **`cdf_recal_isotonic`** | 0.0318 | **ship** | 0.528 |
| board +0.092 | SkewNormal · `ratio_meanyr` · nll · direct · **`cdf_recal_isotonic`** | 0.0328 | **ship** | 0.560 |
| board +0.031 | DPO · pit_ks · nll · **`roe_mean`** | 0.0487 | **ship** | 0.588 |
| seed/incumbent | SkewNormal · `ratio_meanyr` · nll · centered · `isotonic_mean` | **0.0540** | **fail g4** | 0.579 |

All three alternatives clear the official six-gate ship under the new stage order, and **none of
them is a `MEAN_STAGE` step function** — two use the whole-CDF `cdf_recal_isotonic`, one the affine
`roe_mean`. That is the quantization escape the mechanism predicts.

None was promoted: a live cell runs the **supersession** test, not the fresh-ship confirm, and all
three failed S2 (paired mean edge, CI straddles 0) and S3 (Memmel paired-Sharpe `z` +0.90 / +0.95 /
+1.47 against a 1.645 bar) after passing S1. **The baseline those legs score against is the
incumbent's stored dump, produced under the old stage order** — so the walk compared candidates
under the new code with an incumbent under the old. The fourth nominee, the incumbent's own corner,
was skipped as "ledger already holds this corner's verdict on this matrix"; that stored verdict
(`g4_pit_ks` 0.0323, `ship` True) is the old-code one, and Arm 3 retrained exactly that corner on
exactly that matrix under the new code at **0.0540**.

**Generalizable trap: a corner verdict is scoped to its matrix, not to the pipeline.** A stage
change invalidates every stored verdict and every supersession baseline without moving
`strategy_matrix_hash`, so the ledger's skip-if-known rule silently reuses a stale row. Cross-ref
`[[ledger_ship_is_matrix_scoped]]`, which is the same failure keyed on the matrix instead.

**Left for the operator.** The cell keeps its pin and stays `devel`; the sweep restored the
incumbent's `stat_meta` entry, pickle and dump byte-identically. The next production `meditate`
retrains it under the new code, `report()` writes `ship=False`, and the warn-only serve-iff-ship
policy puts the cull-or-promote decision where it belongs. Promoting one of the three corners above
is a one-line `stat_meta.json` edit plus a retrain; the DPO/`roe_mean` corner is the one whose
`roc_auc` (0.588) also beats the incumbent's.

## Corrections to the brief's framing

- **There is no suppression code for `MEAN_STAGE` on gated families.** The docs described it as
  "structurally disabled on ZINB"; the code tests the slug alone and `model_strategy/specs.py`
  offers `roe_mean`/`isotonic_mean` to the ZINB spec. The suppression was de-facto (no live gated
  cell carried a mean slug), so the gate-aware contract had to be *written*, not un-suppressed.
- **`_fuse_negbin` and `_fuse_gamma` never set `weighted_mean_val`.** Only `_fuse_skewnormal` and
  `_fuse_dpo` did; the count branch re-derived it inside `_calibrate_count_dispersion`. The
  derivation moved into the `_fuse_*` branches that own it and the duplicate was deleted.
- **Re-syncing `β = α/μ` for Gamma is unnecessary.** `_calibrate_count_dispersion` already
  re-derives `beta_blend_val` from `weighted_mean_val`, and `_stage_family_shape_columns` re-derives
  `NB_P`, `DP_MU` and `SN_Loc` from `weighted_mean`. Only Mixture, whose components carry the
  location, needs an explicit re-encode.
- **"Byte-identical dumps" is the wrong acceptance test for the `w = 1` cells.** The same change
  redefines the `EV` column, so the honest assertion is that every *gate value* is identical.
- **Arm 0's temperature-spread check reads ~1e-7 on count cells, not ~1e-14**, and 1e-4…1e-3 on
  SkewNormal, where the loc encode/decode round-trip and the `step = 0.5` discretization both enter.
  The stronger validity check is the one that held exactly: the identity re-score reproduced every
  recorded `model_stats.parquet` gate value to a delta of 0.0 on all nine cells.

## Residue

- **NFL tds' book quote is deflated ~8× by a gated-write / ungated-read asymmetry.**
  `odds.under_prob` is not a native de-vigged quote: it is the stored `ev` re-derived **through**
  the zero-inflation gate. On 30 000 sampled NFL tds rows the gated re-derivation matches the stored
  value to a median absolute difference of **0.00031** while the gate-free one misses by **0.142**;
  passing tds gives 0.00205 against 0.107. It holds across all 12 sportsbooks and every year
  2023–2026. `helpers/training_quotes.py:_authentic_quote` then inverts it **ungated** by explicit
  design, so the round trip is `ev --gate--> under_prob --no gate--> ev'` with `ev' ≪ ev`. That is
  where `ρ = 0.22` and the 0.855 haircut came from: stored book `ev` averages 0.25–0.46 against a
  model mean of 0.17, yet the re-inverted book mean is ≈0.04. Cohort-wide — every zero-inflated cell
  writes `under_prob` the same way. `_authentic_quote`'s own docstring already concedes "the book
  leg of `fused_loc` is unsound for gated cells either way"; this measurement identifies why, and
  makes it a repairable write-path defect rather than a distribution-family limitation. **Not
  actioned** — a repair moves the book leg on every gated cell, and three shipping cells (MLB home
  runs, MLB stolen bases, NHL hits) are recorded as borrowing their Gate-6 pass from a mis-levelled
  book leg. Owner decision packet.
- **Post-fusion correction makes the served mean coarse on `isotonic_mean` cells.** NFL tds serves
  34 distinct means over 2466 offers (was 2466). It costs nothing on the gates or on `roc_auc`
  (served 0.7125 against 0.6920 in production) but the served EV is granular, which matters for
  staking.
- **Gate 6's CITL leg remains one-sided**, and this change adds cells to the over-predicting list it
  cannot catch (NFL receiving tds serves CITL 1.639, WNBA FTM 1.156; both `devel`). Unchanged from
  the predecessor lane and still open.
- **`_pred_cdf_pmf`'s Gamma/ZAGamma branch reads `EV`, not `Blended_EV`**
  ([scorecard.py:1084](../../src/sportstradamus/training/scorecard.py#L1084)). Latent, not live —
  zero Gamma cells exist today — but the `EV` contract change makes it wrong rather than merely
  inconsistent. Fix it before any Gamma cell ships.
- **`fused_loc`'s SkewNormal branch inlines `skewnormal_loc_from_mean` twice** instead of calling
  it, despite that function's docstring claiming otherwise.
- **Killed and not to be retried**, all cohort-screened in the research brief: the pool operator
  itself — geometric → arithmetic moves served CITL *away* from 1.0 on 19 of 29 pooled count cells
  and costs NHL hits its `ship`, and precision-weighted is strictly worse than what ships today;
  `_MODEL_WEIGHT_MAX` as a lead (binds on 4 cells, does not generalize); and a second
  book-quality-conditioned weight, since `fit_model_weight` already is one and a Brier-keyed second
  is `proxy_goodhart_under_search` verbatim.
