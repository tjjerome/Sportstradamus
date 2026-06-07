# Operation Ship 75

> **Home of record for the model-research push to 75% breadth.** Research verdicts,
> citations, and the inference-path checklist live in
> [`operation_ship_references.md`](operation_ship_references.md). Current gate
> thresholds live in [`ship_gate.md`](ship_gate.md). The next-rung stub is
> [`operation_ship_90.md`](operation_ship_90.md). This document was rewritten clean on
> 2026-06-03 after the Gate-4 PIT-KS redefinition reset the board; the prior (bloated)
> revision is recoverable from git history (`git show 5c4a335:docs/operation_ship_75.md`).

## North Star

**Get ≥ 75% of each covered league's markets past the five offline ship gates (g1–g5)
in [`training/scorecard.py`](../src/sportstradamus/training/scorecard.py).**

| League | Target | Markets |
|---|---|---|
| NBA | ≥ 16 / 21 | 76% |
| WNBA | ≥ 14 / 18 | 78% |
| NFL | ≥ 15 / 20 | 75% |

A cell counts toward the numerator once it is `shipped ∈ {"devel", "main"}` in
[`stat_meta.json`](../src/sportstradamus/data/config/stat_meta.json) — the production
server tracks `devel`, so a Gate-1-clearing cell in its 14-day Gate-2 soak is already
live and already counts (ship-incrementally). Mantra: *don't let perfect be the enemy
of good.* The gate certifies **deployable**; the Kelly sizer and the live soak certify
**profitable**.

**Failure is not an option at the operation level.** Individual *levers* are allowed to
fail — when one doesn't move a cell, we scrap that path and take the next. The plan is
built so that every league has more independent levers than it has cells to flip, and
the failure branches are written down in advance.

## Purpose — beat the DFS apps; the sharp book is an asset, not an adversary

Read this before anything about a "gate." It is the lens for the whole operation and the
correction that had to be repeated most often during the gate audits.

**We are beating the DFS pick'em apps (Underdog, PrizePicks, Sleeper), not the
sportsbooks.** Those apps price slates of over/under and alt-line picks that players
combine into parlays. We win by handing the parlay builder **more accurate, better-
calibrated probabilities than the app's implied prices** — across the standard line and
the whole alt-line ladder it offers.

**The sharp sportsbook consensus is an asset we blend into, never a wall to beat.** It is
a weighted, closing, de-vigged consensus — by construction it sits very close to the true
probability, so beating it *standalone* is genuinely hard and is **not the goal**.
`fused_loc` blends our model with that sharp line (the ensemble that actually prices our
offers) precisely so we inherit its sharpness and add whatever marginal signal the model
carries. The book is there to be *used*.

**Calibration is the product.** The parlay builder needs well-shaped full predictive
distributions to price alt-lines and to assemble correlated legs without compounding
mispricing. That is why **Gate 4 (PIT-KS calibration) is the real quality bar** — and why
the under-dispersion fix (§5 L1) is the centre of the plan.

**Gate 1 is a no-regression guardrail, not a "beat the book" demand.** It is a
*non-inferiority* test: it certifies that blending our model into the sharp line does not
make the ensemble *worse* than the line alone. A tie passes — that is the entire point.
It follows that:

- A cell that fails Gate 1 is one where our current model is **regressing the blend** —
  adding noise, not signal (typically a small-sample, under-calibrated cell). The fix is a
  better model (calibration via L1, real signal via L3), or leaning harder on the book in
  the blend. It is **never** evidence that "the market is too efficient to win," and
  **never** a reason to loosen a gate.
- A sharp book therefore can never be a wall. The worst case is a cell where the model
  adds nothing and we ride the book line (a tie) — and even then the calibration work (g4)
  still matters, because the parlay builder still needs a well-shaped distribution around
  that line.

## Current standings (2026-06-07, post L0 + L1 ships)

| League | Shipped (`devel`) | Target | Gap |
|---|---|---|---|
| NBA | 11 / 21 | 16 | **−5** |
| WNBA | 8 / 18 | 14 | **−6** |
| NFL | 5 / 20 | 15 | **−10** |
| **Total** | **24 / 59 (41%)** | 45 | **−21** |

Source of truth: the `shipped` field in `stat_meta.json` (verified directly, not the
stale `model_stats.parquet`, which still carries the retired IQR-g4 columns). Shipped
cells today:

- **NBA (11):** BLK, BLST, FG3M, MIN, OREB, PA, PR, PRA, PTS, RA, REB
- **WNBA (8):** BLK, FG3M, FGA, MIN, PA, PR, PRA, TOV
- **NFL (5):** interceptions, receiving-tds, rushing-tds, targets, tds

The board bottomed at **19/59 (32%)** on 2026-06-03 after the Gate-4 redefinition and has
recovered to 24/59 via **Lever 0** (free re-score promotes: WNBA BLK/FG3M/TOV) and
**Lever 1** (post-hoc scale calibration: NBA RA/MIN). This is a deliberate, correct
tightening — not a regression. The previous "43/59 = 73%" was scored under a Gate 4
(`IQR(EV)/IQR(Result) > 0.5`) that turned out to be measuring sharpness, not calibration,
and waved through under-dispersed cells. The redefinition to a randomized-PIT KS
calibration gate (commit `2f1ecd4`) demoted 24 false passes (`cb077db`). The board is now
honest. Getting back above 75% is a model-quality job, and the binding constraint is
precisely identified (§3).

---

## 1. What is already done

Two years of audits and research converge on one clean state: **the gates are now
trustworthy, the book prices are now honest, and the dead-end model levers are known.**
The detail is preserved in `operation_ship_references.md` and in git history at the
commits named below; the compressed ledger:

### 1a. Gate-integrity audits — the gates now measure the right thing

Every gate was independently audited; several were found to be measuring artifacts and
fixed. Net result: a cell that fails a gate today is failing on real model quality, not
on a scorer bug.

| Audit | Verdict | Status today |
|---|---|---|
| **G4 IQR ratio** (orig) | Sharpness-vs-calibration category error; later, decode-strategy and EV-representation artifacts (`carries` g4 0.072 with +0.23 Brier skill gave it away) | **Retired** — replaced by PIT-KS (§1c). The IQR ratio survives as the report-only `g4_iqr_ratio`. |
| **G5 equal-mass ECE** | Positively biased at finite N (false-fails up to 44% of calibrated NFL-N cells) | **Fixed & live** — Monte-Carlo null-bias offset (Roelofs 2022); gate reads `g5_ece_debiased`. |
| **G2 / G3 bias gates** | ZINB/ZAGamma stored base-distribution μ; gate compared it to the zero-*inclusive* segment mean, overstating by `1/(1−π)` | **Fixed & live** — `_zero_inflated_mean` reapplies π; gates score the fused `Blended_EV`. |
| **G1 strict-superiority** | Over-specified intent #3 ("at least as good as the book") | **Reframed & live** — non-inferiority `ci_hi < 0.005` (§1c). Strict superiority survives as the reported `g1_has_edge`. |
| **G1 selection-gate proposal** (Step 0.10) | Replacing/augmenting g1 with a precision-on-value-picks gate is a garden-of-forking-paths multiplicity trap; 0/6 NFL cells survive a Bonferroni-corrected test | **KEEP g1 as-is.** Standing rule: never gate on a model-conditioned statistic. |
| **G1 −0.0 rounding** | `round()` preserves the sign bit; a genuinely-negative CI bound in (−5e-5, 0) false-failed | **Fixed & live** — `np.signbit` counts negative-zero as below the bound. |

### 1b. Archive repair — the book we grade against is now real

The archived per-book `ev` for the passing/count families was a legacy klepto-era seed
(every book = the consensus median line ⇒ `p_book ≡ 0.5`, a coin flip), not a real
price. We re-fetched real two-sided historical prices (Odds API, `backfill_historical_odds.py`),
injected them without a feature rebuild (`inject_backfilled_odds.py`, now
rebuild-equivalent after two latent-bug fixes), and retrained. This is **permanent,
load-bearing infrastructure** — every g1 verdict now grades against an honest book.

The repair also confirmed *how sharp the asset is*: with honest two-sided prices the
continuous NFL passing/rushing-volume lines sit at the median, so `p_book ≈ 0.5` and
`brier_book ≈ 0.25` by construction — the book is genuinely sharp there, and the model
does not beat it *standalone*. Per §Purpose that is exactly what makes it a good blend
ingredient, **not a wall**: the bar on those cells is not "beat the book" but "be
calibrated and carry enough signal that blending the model in doesn't regress the line."
NFL is the hard league because its small samples (≈300–1000 rows/cell) make *that* bar
harder to clear — not because any market is unwinnable (§5, §6).

### 1c. Current gate definitions (the bar we ship against)

Full thresholds and rationale in [`ship_gate.md`](ship_gate.md); the five gates as they
stand:

| # | Gate | Statistic | Threshold |
|---|---|---|---|
| 1 | Brier vs book (non-inferiority) | paired-bootstrap 95% CI of `(p_model−y)² − (p_book−y)²` | `ci_hi < 0.005` |
| 2 | Star σ-match | `\|mean(Blended_EV) − mean(Result)\| / std` on top-mean decile | `z < 0.5` |
| 3 | Bench σ-match | same on bottom-mean quartile | `z < 0.5` |
| 4 | **PIT-KS calibration** | `KS(randomized-PIT, Uniform)` of the predictive CDF | `pit_ks < max(0.05, 1.358/√n)` |
| 5 | Equal-mass debiased ECE | 10 equal-mass `p_model` bins | `ece < 0.075` |

Gate 4 is the one that reset the board. `pit_ks = sup\|F_model − F_true\|` **is** the
worst-case alt-line mispricing, and the randomized PIT (Brockwell 2007) is exactly
Uniform under calibration for count *and* continuous families, so one threshold spans
both. Report-only companions name the *direction* a KS scalar can't:
`central50_coverage` / `central80_coverage` (below nominal ⇒ predictive too narrow),
`g4_tail_pit_ks` (alt-over wobble), `g1_has_edge`, `betting_active`.

### 1d. Model-lever research — what is dead, what is alive

| Lever | Verdict | Source |
|---|---|---|
| Centered-target normalization (`centered_additive_mean10`) | **REOPENED — alive (overturns the P1 "dead" call).** P1 ruled it out as a *mean/level-compression* fix (FGA-only ship) — a different objective, scored before the Gate-4 PIT-KS redefinition. It was **never tried as a PIT-calibration lever**, and the 2026-06-07 honest sweep finds it **systematically out-calibrates `ratio_meanyr` on Gate 4**: deterministic 5/5 on WNBA AST/PTS/RA/REB/DREB + NBA AST/DREB (stand-ins, real-HPO confirm pending). A leading ship axis, not dead. | Phase P1 → §5 sweep 2026-06-07 |
| `init_score` warm-start baseline | **Dead** — byte-identical to plain NegBin. | Phase P2.A |
| ZTNB-hurdle likelihood | **Refuted** — incompatible with the derived-π decode; would regress the 6 shipped hurdle markets. | Stage B1 |
| T5 multiplicative factorization (volume × efficiency) | **Killed** — Goodman variance-of-products gives +27% predictive-variance inflation on the priced cell. | Stage A1.5 |
| Family build for *mean* compression (CMPμ / MZINB) | **On the table — conditional.** Top-decile mean compression is distribution-family-*invariant* (the tree leaf-average itself, refs [3][4][30]), so re-parameterizing dispersion alone won't move it — re-entry is *gated on the §7a condition*, not shelved. | Stage B1.5 §7a |
| HurdleZINB (per-cell mode) | **Alive** — shipped; 6/8 NBA ZINB markets. Available per-cell via `zinb_mode`. | Phase P2.B |
| Post-hoc **mean** correction (`roe_mean` / `isotonic_mean`) | **Alive & shipped** — `MEAN_STAGE` in [`posthoc.py`](../src/sportstradamus/training/posthoc.py); flipped NFL passing-tds + interceptions. | Stage B1.6 / Step 1 |
| Post-hoc **probability** recalibration (`prob_recal_*`) | **Alive** — `PROB_STAGE` built, available per-cell. | posthoc.py |
| Post-hoc **scale / dispersion** correction | **GO — route 1a-hybrid** (fit in `_step_calibrate_dispersion` via `scorecard._randomized_pit_ks`; reuse `dispersion_cal` field + apply; count branch CRPS→PIT-KS too). Reverses the v1 "route 1b" after C0 fixed the decode bug 1b dodged. Levi closed-form σ-scaling is a dead end (diverges 5–7000× on skewed cells). | §5 L1 / brief `researcher_lever1_strategy.md` |
| Player-level features (expanding-mean, EB-shrunk, opp-defense) | **Alive, unbuilt** — RANK 2/3 in the breadth verdict. | Stage B1.6 |
| Per-position model split (NFL) | **Alive, on the table (T11)** — a live NFL lever now, not held for Ship 90; pull forward whenever the binding league needs it. | Stage A1.6 |

---

## 2. The reframe

The old plan was organized around *gate audits* because, at the time, we did not trust
the gates. That work is finished (§1a). **The gates are now correct, so from here the
plan is organized around model-quality levers**, cheapest-first, each shipping per cell
on a clean re-score.

The single most important consequence of the Gate-4 reset: the dominant failure mode
across all three leagues is no longer bias, ECE, or Brier — it is **predictive
under-dispersion** (the per-row distribution is too narrow), which the new PIT-KS gate
measures directly and the old IQR gate hid.

---

## 3. Diagnosis: systematic under-dispersion is the binding constraint

A fresh full-board re-score (all 59 test CSVs through `compute_gates`, the exact
production path) shows the failure modes are overwhelmingly concentrated on Gate 4, and
the coverage diagnostics point one direction: **too narrow.**

**Failure-mode census of the 40 withheld cells:**

| Primary failure | Count | Lever |
|---|---|---|
| **Gate 4 only** (g1/g2/g3/g5 all pass) | 24 | dispersion calibration (§5 L1) |
| Gate 4 + Gate 1 (marginal g1, `ci_hi` 0.007–0.018) | 6 (all NFL) | dispersion + edge (§5 L1→L3) |
| Gate 4 + Gate 2/3 (bias) | 3 | mean correction then dispersion (§5 L2→L1) |
| Multi-gate (g1+g3+g4) | 2 (NFL passing-first-downs, qb-yards) | hardest; features + per-position |
| Gate 1 only / Gate 1+5 (edge) | 2 (WNBA STL; NBA PF also g5) | edge / features (§5 L3) |
| **Pass now but un-promoted** | 3 (WNBA BLK, FG3M, TOV) | free (§5 L0) |

Sum = 40 withheld. The signal is stark: **24 cells fail Gate 4 and nothing else**, and a
further 6 fail Gate 4 plus a *marginal* Gate 1. Fix the predictive width and most of the
board moves.

**The coverage evidence (representative cells, nominal central-50 = 0.50):**

| Cell | family | `pit_ks` | `central50` | reading |
|---|---|---|---|---|
| NBA PTS (shipped) | SkewNormal | 0.046 | 0.45 | mildly narrow, squeaks under 0.05 |
| NBA AST | SkewNormal | 0.102 | 0.38 | under-dispersed |
| NBA STL | ZINB | 0.051 | 0.81 | a hair over (count lattice) |
| WNBA DREB | SkewNormal | **0.504** | 0.23 | severely under-dispersed |
| NFL receptions | SkewNormal | **0.432** | 0.24 | severely under-dispersed |
| NFL passing-yards | SkewNormal | 0.122 | 0.38 | under-dispersed, g1 ties |

Every SkewNormal cell — *including the ones that ship* (PA/PR/PRA/PTS/REB sit at
`central50 ≈ 0.44–0.46`) — is under-dispersed; it is only a matter of degree. The
shipped ones squeak under `pit_ks < 0.05`; the withheld ones don't.

### Root cause (found in code, 2026-06-03)

The SkewNormal family receives **no dispersion calibration at all**, by two hardcoded
exclusions:

1. [`training/pipeline.py:_step_calibrate_dispersion`](../src/sportstradamus/training/pipeline.py)
   (~line 1490) **early-returns for `dist == "SkewNormal"` with `c_opt = 1.0`** — it
   never fits the CRPS-minimizing scale factor it fits for NegBin/ZINB/Gamma.
2. [`prediction/model_prob.py:675`](../src/sportstradamus/prediction/model_prob.py)
   applies `dispersion_cal` only `if dispersion_cal != 1.0 and dist != "SkewNormal"`.

So every SkewNormal cell runs on the raw GBDT scale — exactly the leaf-averaged,
dynamic-range-compressed scale that refs [3][4][30] predict is too narrow. The count
families *do* get `dispersion_cal` (CRPS-fit), but the hair's-breadth count failures
(NBA FTM 0.066, STL 0.051, TOV 0.051) show the **CRPS objective is not the PIT-KS
objective** — minimizing CRPS does not guarantee a calibrated PIT.

This is good news: the binding constraint has a named, surgical root cause in existing,
tested machinery, and the lever that addresses it (§5 L1) has the highest leverage in
the plan.

---

## 4. Why this is solvable (the headroom)

The diagnosis is not "the models are bad" — it is "the predictive width is uncalibrated
on the family that makes up most of the board." The g1 (Brier) margins on the only-g4
cells are comfortable (NBA AST `ci_hi` −0.028, DREB −0.064), meaning the blended ensemble
already *ties-or-beats the book on a proper score* (Gate 1 satisfied); they fail only
because the predictive shape that prices alt-lines is too tight — a pure calibration miss,
the exact thing the parlay builder needs fixed. Widening a too-tight predictive toward calibration is a
move with headroom on both the gate (PIT-KS down) and, plausibly, Brier (sharper tails).

The risk is the mirror image: over-widening pushes probabilities toward 0.5, which can
*reduce* Brier skill (g1) and shift ECE (g5). So L1 is fit to a calibration target and
guard-railed on g1/g5 — see §5.

---

## 5. The lever stack (cheapest-first; each ships per cell on a clean re-score)

Each lever names its **mechanism**, **targets**, a **go/no-go** measured on validation
*before* any ship, and an explicit **if-it-fails** branch. Levers are independent enough
that a failure on one does not block the others.

### The search space is three axes, not one cascade — and we have explored ~one

The lever cascade below (L0–L5) is the depth-first exploration of **one** of three
orthogonal axes. A served predictive is `(normalization → model+loss → blend → calibration)`,
and three of those stages are independently swappable per cell
(`target_normalization ⊥ {posthoc, dispersion_cal, skew_cal} ⊥ {dist_training_loss, blending_loss_fn}`). Naming them:

- **Normalization** — the target transform the model fits. *Well-defined* (`ratio_meanyr`,
  `centered_additive_mean10`, `centered_additive_eb_meanyr_k10`) but badly *under-explored*:
  the EB / hierarchical-shrinkage strategy is built, decode-tested, and assigned to **zero**
  production cells, and **volume normalization does not exist yet**. Modeling `points`
  (or `points / season-mean`) conflates *efficiency × opportunity*. A `ratio_projvol`
  strategy — target `= y / projected-volume` (a per-minute / per-carry / per-target rate),
  decode `= rate × projected-volume` — separates the stable efficiency signal from the
  matchup-driven volume signal. The volume projections **already exist** (`proj_*` features:
  projected carries/targets/minutes) — they are used as *features* but never as the
  normalization *denominator*. This is the most likely structural unlock for the NFL
  volume cells (a `rushing-yards` g1 block is plausibly a volume/efficiency conflation the
  book prices and we do not).

- **Calibration** — post-hoc on the served predictive: **{none, dispersion, skew-joint,
  skew-sequential}**. This is the axis worked to date (L1 `dispersion_cal`, L4a `skew_cal`).
  Two fit *orders* exist for the scale-and-skew pair `(c, s)` and they are **not** equivalent:
  *joint* optimizes `(c, s)` together; *sequential* fits the Lever-1 scale `c`, freezes it,
  then fits the additive skew `s` on top. On the binding under-skew cells **joint strictly
  dominates** — the coupling is structural (Hallin & Ley's Fisher-information singularity at
  `alpha ≈ 0`): the scale-only KS optimum lands at a `c` where the skew gradient vanishes, so
  sequential is stuck at `s = 0` there, while joint co-moves `c` upward *and* injects skew
  (synthetic check 2026-06-07: scale-only KS 0.064 → sequential 0.064 → joint 0.015).
  Sequential earns its keep only where the trained model already carries a wrong/weak
  *directional* skew at a scale-active `c` (it then touches it up); joint is the prior. Because
  every mode here is a post-hoc transform of a *fixed* trained predictive, the whole calibration
  axis is **free to sweep** — no retrain. The honest combination search below does *not* yet exploit
  this: it reads the pipeline's one val-fit mode off each dump, because an honest sweep must fit the
  modes on a dumped *validation* predictive, not the test rows (re-fitting on test is the optimism that
  sank the first screen — §5 search). Calibration is close to exhausted as a *breadth* lever on its own
  — it fixes width/shape, never signal or location-scale structure.

- **Blending** — how the model's distribution meets the book: the training **loss** (`nll`
  vs `crps`, upstream — it shapes the predictive that gets blended) × the **blend** itself
  (`fused_loc`: precision-weighted pool + per-cell `model_weight`). Both frozen today
  (loss set per family, blend structure fixed). Under-explored.

**Nothing is deferred. Every withheld cell and every lever is a live Ship-75 candidate.** The
`deferred-90` / "defer" / Lever-cap tags are **retired for the duration of this operation.** They
were always per-axis verdicts — almost always the *calibration axis under a single normalization* —
and the 2026-06-07 sweep showed how badly that mis-reads: cells the screen stamped `deferred-90`
ship under a *different* normalization. A cell leaves the Ship-75 board **only** after it has
actually failed on **all three axes** (normalization × calibration × blending) **plus** the
hierarchical layer — and as of 2026-06-07 that is true of **zero cells**. Any "defer",
"deferred-90", "cannot reach", or "efficient-market wall" wording that survives below is the *old*
per-axis verdict, kept for its evidence but **superseded by this policy**; read none of it as final.
The bar for parking a cell is axis-exhaustion across the whole matrix, never a single screen.

**Next phase after the calibration pass — the combination search (smart, per-market; not a
brute-force grid).** The naive read — cross every axis value for every cell,
`normalization {4} × calibration {4} × loss {2} = 32` retrains × ~38 cells — is the wrong shape,
because the three axes are **not equally expensive**:

- **Calibration is post-hoc and free.** Every calibration mode is a transform of an *already
  trained* predictive's val/test arrays (§5 calibration bullet), fit in milliseconds. It must
  **not** sit in the training loop. Given one trained model, sweep all four modes and keep the
  best — that collapses the `×4` calibration factor to zero retrains.
- **Only `normalization × loss` costs a train** — ≈ `(2–4 cell-applicable norms) × 2 losses`
  = **4–8 trains per cell**, not 32.

So the search is, **per market**: an **Optuna study over the categorical retrain grid**, built
2026-06-07 as [`training/model_strategy_driver.py`](../src/sportstradamus/training/model_strategy_driver.py)
(entry point `model-strategy-driver`). Its `SEARCH_SPACE` is four axes in pipeline's `[kind, spec]`
idiom plus a `stage` tag: the **retrain** axes `normalization × dist_training_loss × blending_loss_fn`
form a `GridSampler` grid (exhaustive + deterministic — the right tool for ≤12 discrete corners; the
`[kind, spec, stage]` shape flips to TPE the moment a continuous axis lands), and the one **post-hoc**
axis — `calibration` (the free 4-mode sweep, fanned out off each trained corner with no extra train) —
is scored off each trained dump. `blending_loss_fn` rides the *retrain* tier (not post-hoc) because the
blend weight is fit *inside* meditate: a `--blending-loss-fn crps` run refits `w` by CRPS during
training. The brief's free post-hoc `w`-refit needs the dump to carry the pre-blend components, which
the deterministic flow does not persist yet; until it does, a `crps` blend costs a train. Each
**trial** is one `--deterministic`
train (bit-reproducible, fast, never published); the objective is the **min-gate slack** of the
best post-hoc row — a single scalar positive iff the corner ships and larger the more gate headroom
it has, so it optimizes "ships, with margin" across all five gates at once rather than chasing g4
alone. Per cell the study returns one board row per `(retrain corner × calibration mode)`, ranked by
slack.

*What the built ranker actually scores (and the calibration-honesty trap it avoids).* The shipped
`training/model_strategy_search.py` ranks each normalization by the **honest val-fit→test gate row**: the
deterministic dump already carries the pipeline's own **validation-fit** joint calibration, so the
ranker just calls `gate_row` on it — the *same* code production ships on — and reads
`min_gate_slack` off the result. No test re-fit, so fidelity is by construction. An earlier build
instead **re-fit the four calibration modes on the trial's test rows** (even with an OOS
split-half) and reported the best; that path oversold the screen by ~0.008 KS and is removed. The
*design* sweep — try `{none, dispersion, skew-joint, skew-sequential}` and keep the best — is
still cheap when honest (post-hoc calibration **holds the blended mean fixed** via
`skewnormal_loc_from_mean`, so **Gates 2/3 are calibration-invariant**, **Gate 4 is exact from the
SkewNormal params** via `_served_sn_pit_ks`, and **Gates 1/5 barely move**, so only Gate 4 varies
across modes — no `P` re-pricing). But an *honest* sweep must fit each mode on a dumped
**validation** predictive, not the test set; until that val dump exists, each trial uses the
pipeline's one val-fit mode as-is. `scorecard.sweep_calibration_modes` is **now wired into the
driver** (via `gate_rows_by_calibration_mode` → `model_strategy_search.score_calibration_modes`): it
recovers each dump's pre-calibration blended `(mean, sigma, skew)` — dividing the served scale by the
pipeline's auto-fit `dispersion_cal` and subtracting `skew_cal`, both read from the pickle — and fits
all four modes over it, no retrain. **Honesty caveat:** today that recovered predictive is the *test*
dump (the deterministic flow dumps only the test split), so the in-driver sweep is an **in-sample
ranking signal** — the same optimism (~0.008 KS) the earlier test-refit screen carried, acceptable
*only* because the driver ranks and the real-HPO 5-gate scorecard ships. The honest-val upgrade is to
recover from a dumped **validation** predictive; the recovery code is identical, only the dump split
changes.

**The deterministic study only ranks; the real-HPO scorecard ships.** The WNBA-AST lesson
(2026-06-07, corrected): on **like-for-like** normalization the deterministic stand-in **tracks**
real HPO — under the default `ratio_meanyr` the honest deterministic Gate-4 KS is **0.126**, matching
real HPO's **0.123** (no "worse model beats better" paradox; the apparent paradox was the dishonest
test re-fit reporting 0.039, now gone). The ranker's value is that it surfaced a *different*
normalization that ships honestly: under `centered_additive_mean10` WNBA AST clears all five gates
(g4 = 0.047 knife-edge, g1 −0.017, g5 0.049) where `ratio_meanyr` fails g4 hard. That is a
normalization swap to **confirm under real HPO**, not a calibration trick. The search is therefore a
cheap *ranker*: take the **top-K (K≈2–3) corners per cell**, re-run each under **real HPO**, and ship
the first that clears the official 5-gate scorecard (`model_stats.parquet`) — never the deterministic
score. The knife-edge g4 (0.047 vs 0.05) is exactly the kind of margin the val→test discount can
erase, so the real-HPO confirm is mandatory.

Sweep the **whole board**, shipping cells included (a shipped cell may have a better corner than
the scale-only default it settled for). The calibration-proof cells the L4a screen isolated
(heavy-tail NBA AST / NFL passing-yards; g1-blocked NFL receiving/rushing-yards; the
centered-strategy WNBA DREB / NFL receptions) are the highest-value rows but not the limit.
Built 2026-06-07: the per-market Optuna driver (`model-strategy-driver`), the `--dist-training-loss`
and `--blending-loss-fn` `meditate` flags, the forced 4-mode calibration sweep (wrapping
`training/scorecard.py`'s gate computation), and the **`crps` blend objective** (`fit_model_weight_crps`,
registered in `BLENDING_SLUGS`). The crps build cleared its gate: the empirical clamp-bite check
(`/tmp/clamp_bite_check.py`, the brief's decisive #1) found the `-20` logpdf clamp bites on the blended
predictive of every heavy-tail SkewNormal cell (0.18–1.79% of rows, mean 0.77%, concentrated on the
under-dispersed binding cells), so it is not the "~0 everywhere" KILL picture. The win is *modest* and
per-cell — guardrail-2 (the per-cell search scoring) is the arbiter of whether any cell adopts it.
Still unbuilt: the `ratio_projvol` `TargetNormalization`, the honest-val dump that makes the
calibration sweep OOS, and the **free post-hoc `w`-refit** (so `crps` blend stops costing a train —
needs the dump to carry the pre-blend components). A genuinely structural fourth lever — a
**hierarchical Bayesian** layer that
generalizes the dormant EB prior (learn the shrinkage per group, pool the full predictive across
player ← position ← team/league rather than the mean only) — is the research-gated answer to the
small-sample NFL wall and is scoped by a `research-analyst` brief before any build.

**The operating loop (per parameter, per cell).** Once an axis is wired, the per-cell workflow is
fixed — and the ship bar differs for a withheld cell vs an incumbent:

1. **Driver board = candidate generator.** `model-strategy-driver` returns the ranked board
   (deterministic train + in-sample calibration sweep). A `ships=True` row is a *candidate flag*,
   never a ship — the real gate is the full-HPO official 5-gate scorecard in `model_stats.parquet`.
   Carry the **top-K (2–3)** corners per cell forward.

2. **Withheld cell → real-HPO confirm → ship to devel.** Set the winning corner's strategy
   (`target_normalization` + calibration mode + `dist_training_loss`) in `stat_meta.json`, run a
   full-HPO `meditate`, read the official scorecard. A clean **5/5** → flip
   `shipped: "withheld" → "devel"`. Ship the first top-K corner that clears; if the winner is just
   the cell's current default strategy it is a straight confirm (no strategy edit). The knife-edge
   cells (g4 within ~0.003 of 0.05) are exactly where the val→test discount bites, so the confirm is
   mandatory — never ship the deterministic score.

3. **Incumbent (already-shipped) cell with a better corner → supersede, a higher bar.** A shipped
   cell does **not** re-ship on a fresh 5/5 — the candidate must *beat the incumbent*. Built as
   `scorecard.supersede_verdict(baseline, candidate)` (`docs/ship_gate.md`, "supersede an
   incumbent"), three gates AND'd:
   - **S1** — candidate clears the standard 5-gate scorecard standalone;
   - **S2** — paired Brier CI lower-bound > 0 (candidate statistically sharper on the shared rows);
   - **S3** — paired Kelly-Sharpe Memmel-z > min (candidate's simulated returns statistically sharper).

   All three → `SUPERSEDE` (swap the strategy in `stat_meta.json`); any fail → `HOLD` (keep the
   incumbent). Both sides need full-HPO, row-aligned test dumps; CLI
   `python -m sportstradamus.training.scorecard --baseline … --candidate …`. The S2/S3
   beat-incumbent asymmetry is deliberate — it stops strategy-churn on noise.

**Two caveats on "wire in a new parameter."** (a) **Research-gate** — a parameter that changes a
*distribution family or dispersion mechanism* needs a `research-analyst` brief before it is wired or
built (CLAUDE.md research-first); a plain knob (a normalization slug, a loss choice) does not.
(b) **Wiring an axis-value ≠ it sweeps.** A value can sit in `SEARCH_SPACE` yet not sweep until its
machinery exists, and its *cost tier* can shift once it does. `blending_loss_fn` carried `crps` as a
defined value before `fit_model_weight_crps` existed; building it (2026-06-07, after the clamp-bite
check cleared) made it sweepable — but on the *retrain* tier, not the brief's free post-hoc tier,
because the deterministic dump doesn't yet carry the pre-blend components a free `w`-refit needs. So
"wire it in" is sometimes "wire the axis, build the value, decide the tier — then it sweeps."

**Sweep in flight (2026-06-07, preliminary).** The first honest pass — the built normalization
ranker ([`training/model_strategy_search.py`](../src/sportstradamus/training/model_strategy_search.py): deterministic
trains × both decodable normalizations, scored on val-fit→test gates) — is running across the
withheld SkewNormal cells of the covered leagues. The early signal is strong and consistent:
`centered_additive_mean10` systematically out-calibrates `ratio_meanyr` on Gate 4, flipping a batch
of withheld cells (on the first 8: WNBA AST/PTS/RA/REB/DREB and NBA AST/DREB, several with real
margin below 0.05, not knife-edge). These are **deterministic stand-in** verdicts — each ships only
after its real-HPO confirm — but they corroborate the thesis above that **normalization, not
calibration, is the dominant unexplored axis** (and they confirm this section's own prediction that
the "centered-strategy" cells like WNBA DREB resolve once the normalization axis is tried). The
Optuna `GridSampler` driver + the `dist_training_loss`/`blending_loss_fn` flags + the 4-mode
calibration sweep landed 2026-06-07; the `ratio_projvol` normalization and the hierarchical-Bayes
layer remain unbuilt. This *preliminary* pass predates the driver — it is the normalization slice alone.

**Is the search matrix well-defined? — the executable state, honestly.** The *method* is well-defined
and proven: per cell, enumerate the executable axis-values, train one `--deterministic` model each,
score the **honest val-fit→test gate row** (the production gate path), rank by `min_gate_slack`, and
confirm the top-K under real HPO before any ship. As of 2026-06-07 the driver controls three of the
four axis dimensions (normalization, dist-loss, blend-loss as the retrain grid; calibration swept free);
the executable state per axis:

| Axis | Values defined | Executable today | Build to put the rest on the table |
|---|---|---|---|
| **Normalization** | `ratio_meanyr`, `centered_additive_mean10`, `centered_additive_eb_meanyr_k10`, `ratio_projvol` | **3 of 4** — `ratio_meanyr` + `centered_additive_mean10` + the **EB** slug now have a Gate-4 SkewNormal decode (`scorecard._decode_sn_loc_scale`, EB off the dumped `GlobalMean`) | **build** `ratio_projvol` (target = y / projected-volume) |
| **Calibration** | `none`, `dispersion`, `skew-joint`, `skew-sequential` | **swept by the driver (in-sample)** — `gate_rows_by_calibration_mode` recovers the blended predictive off each dump and *compares* all four modes per corner; today the dump is the **test** split, so it is an in-sample ranking signal (real-HPO confirms) | dump the **validation** predictive so the same sweep is OOS |
| **Loss / blending** | loss `nll`/`crps`; blend weights | **dist-loss: 2** (`--dist-training-loss nll\|crps`) + **blend-loss: 2** (`--blending-loss-fn nll\|crps`; `fit_model_weight_crps` built 2026-06-07) — both retrain-tier | free post-hoc `w`-refit (so `crps` blend stops costing a train); expose the blend weights |

As of 2026-06-07 the driver gives the search **three independent retrain degrees of freedom**
(`normalization × dist_training_loss × blending_loss_fn`, the `GridSampler` grid) **plus the free
post-hoc calibration sweep** (all four modes compared per corner) — it controls and compares along
every axis, where the pre-driver state only *auto-fit* calibration and held loss/blend fixed. The
earlier "one DOF — a line of 2 points" framing is fully superseded: the visited set is a
`(3 norms × 2 dist-losses × 2 blend-losses × 4 cal-modes) = 48`-row board per cell once a sweep is run.
What remains genuinely unbuilt is narrow and is *fidelity / cost*, not new axes: the `ratio_projvol`
normalization value (the one missing axis-value), the honest-val calibration dump (today's sweep is
in-sample), the free post-hoc `w`-refit (so a `crps` blend stops costing a train), and the
research-gated hierarchical-Bayes layer. Nothing in the space is ruled out — only those are not-yet-wired.

### Lever 0 — Re-score every withheld cell under the current gates; promote free passers

**Mechanism.** The Gate-4 redefinition was applied as *demotions only*; cells that the
*old* gate killed but the *new* gate passes were never re-promoted. The fresh re-score
finds three: **WNBA BLK (`pit_ks` 0.037), FG3M (0.030), TOV (0.038)** all pass all five
gates today but sit at `shipped: "withheld"`.

**Targets.** WNBA +3 → 8/18. Cost: three one-line `stat_meta.json` edits + the
`supersede_verdict()` check (these are first-baselines, so Tier-0 absolute gates only).

**Go/No-Go.** Re-run the official scorecard (not just this sweep) to confirm the three
still pass; promote. Run the same sweep monthly so the demote-only asymmetry never hides
a free pass again.

**If it fails.** N/A — this is bookkeeping, not modelling. (If the official scorecard
disagrees with the sweep on any of the three, that disagreement is itself a scorer bug
to chase before anything else ships.)

### Lever 1 — Post-hoc dispersion calibration (the main event)

**Mechanism.** Fit a per-cell scale correction so the predictive PIT is Uniform, applied
after decode and `fused_loc`, leak-free (fit on validation, applied to the disjoint test
split). Two implementation routes, cheaper first:

- **1a — Extend the existing `dispersion_cal` to SkewNormal.** Remove the early-return
  in `_step_calibrate_dispersion` and the `dist != "SkewNormal"` guard in `model_prob.py`;
  fit `c_opt` to scale `SN_Scale` (sigma). Reuses the pickle field, the inference apply
  site, and the determinism harness already in place.
- **1b — A new `SCALE_STAGE` in `posthoc.py`.** Mirror `MEAN_STAGE`/`PROB_STAGE`:
  `{"scale_pit", ...}`, fit on the fused predictive to minimize PIT-KS (or match
  `central50`/`central80` coverage) directly, selected per cell via the `posthoc` field.
  Cleaner separation; works for count families too.

**Re-target the objective.** The count-branch `dispersion_cal` currently minimizes CRPS;
the gate is PIT-KS. Re-target the fit to **coverage / PIT-KS** (or add it as a
tiebreaker). This is what should flip the hair's-breadth count cells (FTM/STL/TOV).

**Targets.** The 24 "Gate-4-only" cells, plus a likely assist on the 6 "g4+marginal-g1"
NFL cells. Expected: NBA +6–9, WNBA +6–8, NFL +3–5.

**Go/No-Go (per cell, on validation, before ship):**
- PASS: `pit_ks` crosses below `max(0.05, 1.358/√n)` **and** `g1` Brier-skill drops by
  ≤ 0.01 (the §1.4-class guardrail) **and** `g5` ECE stays < 0.075.
- The gate scores the per-row params in the test-CSV dump, so the calibrated scale
  **must be written to that dump** (`_step_persist_artifacts`) and applied identically
  at inference — this is a "post-hoc calibration object" on the inference-path checklist
  (references §12): pickle key + legacy fallback + byte-identical round-trip test.

**If it fails** (widening flips g4 but breaks g1/g5, or doesn't move `pit_ks` because the
mislocation is in the mean/skew not the scale):
- Route bias-direction cells (g2/g3 also failing) through **Lever 2 first** — a
  mislocated mean reads as poor coverage that scale can't fix.
- Route cells where scale calibration trades g4 for g1 to **Lever 3** (features that
  earn the sharper predictive honestly rather than inflating it).
- Severe cells (WNBA DREB 0.50, NFL receptions 0.43, qb-yards 0.23) likely need more
  than a scalar — escalate to Lever 4 (per-cell family/shape) and flag as research holes
  (§8): is the SkewNormal *shape* (tail weight), not just its width, wrong here?

#### Lever 1 — BUILT 2026-06-06 (route 1a-hybrid, served-predictive)

The dispersion fit is implemented and gate-green (golden + integration + a dump↔inference
round-trip unit test). **Operator call this session: Gate 4 certifies the SERVED (blended ×
c) predictive — the distribution the parlay builder actually prices — not the model-only
shape.** Consequences threaded end to end:

- **Fit** ([`scorecard.fit_skewnorm_dispersion_c`](../src/sportstradamus/training/scorecard.py)):
  minimizes the gate's own `_randomized_pit_ks` over a scale multiplier `c` on the served
  predictive (blended mean held fixed, scipy loc derived via the shared
  `helpers.distributions.skewnormal_loc_from_mean`). No PIT-math duplication.
- **Calibrate** (`_step_calibrate_dispersion` SkewNormal branch): drops the `c=1.0`
  early-return; emits served `sn_sigma_blend_{test,val}` = blended × c, mirroring the count
  branch's `r_test`. `_step_compute_test_probabilities` + temperature now read the served sigma
  so the `P` column matches inference.
- **Dump** (`_step_persist_artifacts`): writes the served (blended, c-baked) loc/scale/skew,
  re-encoded to normalized space via the new `baselines.encode_loc/encode_scale` (exact
  inverses of `decode_*`), so the scorecard's existing decode recovers the served EV params
  byte-for-byte. C0's decode + the strategy plumbing are untouched.
- **Apply** ([`model_prob._dispersion_calibrate`](../src/sportstradamus/prediction/model_prob.py)):
  drops the `dist == "SkewNormal"` guard; `Model Sigma *= dispersion_cal` after the blend
  (current code order = the served sigma). Existing `dispersion_cal` pickle field + legacy
  fallback reused.
- **Real-data validation** (NBA AST, `--deterministic` so it's a deliberately low-quality
  model, not a ship verdict): the fit returned **c = 1.158**; held-out test `pit_ks`
  dropped **0.1149 (c=1) → 0.0867 (c=1.158)**, with the test-optimal c = 1.29 → 0.063. So
  widening monotonically reduces miscalibration on the disjoint split and the val-fit c is
  conservative vs test-optimal, exactly the ~20–30% val→test discount the brief predicted.
  **Production ship counts await the C3 real-HPO retrain loop** (deterministic HPO is too weak
  to clear 0.05 on its own).
- **Note for C3:** because Gate 4 now scores the *blended* predictive (wider than model-only —
  the book widens the too-narrow model), the model-only board (`board_postC0.csv`) and its
  tier map are **pessimistic**; the real flip counts come from the retrain, as always planned.

#### Lever 1 — SHIPPED 2 cells (RA, MIN); C1 ceiling found at moderate cells — 2026-06-06

- **Blocker (found + fixed).** `data/models/` is empty in a fresh checkout, so every local
  retrain takes the *cold* HPO path — and `lightgbm 4.6` broke it: optuna's
  `LightGBMPruningCallback` hardcodes the cv validation name to `"cv_agg"`, but lightgbm ≥ 4.6
  reports cv eval under `"valid"`, so the callback raises. Fallout of the local-only lightgbm
  bump `ff75d41` (on `model-research`, **not** `origin/devel` — the server runs 4.2 and is
  unaffected). Fix: consolidate the cold and warm paths into one
  [`hyperparams.tune_hyperparameters(initial_params=None)`](../src/sportstradamus/training/hyperparams.py)
  (cold = warm minus the seed enqueue; both omit the broken pruner — the same TPE-over-cv loop
  lightgbmlss `hyper_opt` runs), redirect the pipeline cold caller, add
  `tests/integration/test_cold_hpo_lightgbm46.py` as the regression guard.
- **NBA RA shipped.** Real HPO (98 trials, `--bypass-withholding`, `ratio_meanyr`) + C1 →
  `dispersion_cal = 1.077`; **g4 `pit_ks` 0.0530 → 0.0339** (clears 0.05), g1 brier-skill
  0.030 → 0.033 (improved), g5 ECE 0.032 — **5/5 gates, `ship=True`**. `dispersion_cal > 1`
  confirms the g4 flip is *caused* by C1's widening, not a luckier HPO draw. Flipped
  `shipped: withheld → devel`. **NBA 9 → 10, board 22 → 23/59.**
- **NBA MIN shipped.** Real HPO + C1 → `dispersion_cal = 1.102`; **g4 `pit_ks` 0.0549 → 0.0298**,
  all 5 gates, g1 brier-skill 0.33 (model dominates the book). Flipped `withheld → devel`.
  **NBA 10 → 11, board 23 → 24/59.** Its blended scale-only floor (0.023) sits *below* its
  model-only floor (0.033) — the precision-weighted book blend was benign here.
- **C2-remainder done.** The SkewNormal determinism gate is extended to WNBA (PTS) and NFL
  (receiving-yards) via a shared `_assert_skewnormal_params_bit_reproducible`; 3/3 cross-league
  bit-identity holds — the Lever-1 ship prerequisite.
- **AST — C1 ceiling (did not ship).** Real HPO + C1 fit a large `c = 1.277` and pulled g4
  `pit_ks` 0.1024 → 0.0812, but **the scale-only floor is 0.063 > 0.05**: sweeping `c` on the
  served predictive bottoms at total c ≈ 1.39, then over-disperses. No uniform scale calibrates
  AST — the residual is *shape*, not width, and unlike MIN the blend *lifts* AST's floor (book
  and model disagree on the conditional), so widening can't recover it. **Verdict: hair's-breadth
  g4 cells (RA 0.003, MIN 0.005 gap) ship on C1; moderate cells (AST 0.052 gap) need L4 (shape —
  a skew-aware post-hoc or a heavier tail), not more width.** The model-only floor proxy (DREB
  0.036 / FGM 0.043 / FG3A 0.032, all < 0.05) is suggestive but *not* sufficient — the gate
  scores the blend and AST shows the blend is the wildcard, so each still needs its own retrain.
  *(Scope correction, 2026-06-07: the C1-scale ceiling under `ratio_meanyr` is a real tried result,
  but the "residual is shape → AST needs L4" conclusion was itself premature — the §5 sweep ships
  AST under the `centered_additive_mean10` normalization (det g4 0.042), which reshapes the residual.
  The normalization axis resolves it; nothing here rules AST out.)*

#### Lever 4a — skew calibration: research + applicability screen — 2026-06-07

Lever 4a = a **second** post-hoc knob beside L1's scale `c`: an additive shift `s` on the
served SkewNormal shape `alpha`, fit *jointly* with `c` by minimizing the same Gate-4
randomized-PIT KS. It stays inside the SkewNormal family (reuses `dispersion_cal` machinery,
adds a sibling `skew_cal` field) — an engineering change, **not** a new distribution head.

- **Research brief `/tmp/researcher_l4_shape.md` (research-gated; mechanism = GO).** Additive
  (not multiplicative — the model fits `alpha ≈ 0`, so `alpha × k` injects no skew) and *joint*
  (not sequential — `c` and `s` are coupled). The `alpha ≈ 0` fit is *expected*, not a bug: the
  SkewNormal direct parameterization has a **Fisher-information singularity at `alpha = 0`**
  (Hallin & Ley 2014, *Bernoulli* 20(3), arXiv:1209.4177), so the per-row skew head collapses to
  symmetry. Fit `s` in the **centered-skewness** metric (clamp `|s| ≤ 3`), warm-start `(c,0)`,
  **fit on val**, ship only at **val `pit_ks < 0.040`** (the val→test discount is +0.008–0.010;
  measured by split-half).
- **NBA AST — the L4a screen routed it `deferred-90` (heavy tail); the sweep refutes that.** Under
  its own `ratio_meanyr` normalization the residual is a Gaussian-thin heavy *tail* the SkewNormal
  cannot reach by `(c,s)` alone (`z₉₉ = +4.9σ`, excess-kurtosis ≈ 591; johnsonSU hits 0.0345 where the
  joint `(c,s)` floors at 0.0488, gap 0.014 = the tail; split-half honest fit 0.060 > 0.05). That is a
  real **calibration-axis-under-`ratio_meanyr`** verdict — and it is *normalization-specific*: the
  2026-06-07 sweep ships NBA AST under `centered_additive_mean10` (deterministic g4 0.042), because the
  centered target reshapes the residual the tail lived in. So `deferred-90` was premature. A skew-t head
  (L5) remains the fallback **only if** the normalization + real-HPO route also fails, and L5 stays a
  *batch* call (≥ 3 cells before a shared skew-t) — not a per-cell rule-out here.
- **Applicability screen (brief open-q#5).** Ran the johnsonSU-vs-joint-`(c,s)` floor on every
  withheld SkewNormal cell **decoded with its own `target_normalization`** (the brief's scratch
  hardcoded `ratio_meanyr`, which mis-decoded the `centered_additive_mean10` cells — that is what
  manufactured WNBA DREB's spurious −37.8 served mean). A cell is a **true L4a candidate** only
  where `joint(c,s) ≈ johnsonSU` (skew is the residual, tail already calibrated) **and** `joint <
  0.040` (survives the discount). Then layered the current g1 (Brier non-inferiority) as a ship
  proxy — L4a barely moves g1, so a cell that fails g1 now won't ship on calibration alone:

  | cell | scale-only ks | joint (c,s) ks | johnsonSU ks | g1_ci_hi | route |
  |---|---|---|---|---|---|
  | WNBA AST | 0.0513 | 0.0397 | 0.0437 | −0.027 ✓ | **L4a ship** |
  | NFL yards | 0.0613 | 0.0184 | 0.0195 | −0.017 ✓ | **L4a ship** |
  | NFL fantasy-prizepicks | 0.0540 | 0.0170 | 0.0393 | −0.046 ✓ | **L4a ship** |
  | NFL fantasy-underdog | 0.0750 | 0.0209 | 0.0453 | −0.011 ✓ | **L4a ship** |
  | NFL receiving-yards | 0.0751 | 0.0333 | 0.0342 | +0.0075 ✗ | g4-fixable; g1 not yet cleared → on the table for L3 features + normalization |
  | NFL rushing-yards | 0.0739 | 0.0343 | 0.0367 | +0.0101 ✗ | g4-fixable; g1 not yet cleared → on the table for L3 + normalization |
  | NBA AST | 0.0629 | 0.0488 | 0.0345 | −0.038 ✓ | screen stamped heavy-tail `deferred-90` — **refuted by the 2026-06-07 sweep: ships under `centered_additive_mean10`, det g4 0.042** |
  | NFL passing-yards | 0.0800 | 0.0463 | 0.0236 | — | heavy tail on the calibration axis (n=377) — *on the table; normalization/blending untried* |
  | WNBA DREB | 0.0933 | 0.0564 | 0.1225 | — | screen said neither → **refuted by the 2026-06-07 sweep: ships under `centered_additive_mean10`, det g4 0.035** |
  | NFL receptions | 0.1244 | 0.0738 | 0.5255 | — | neither on the calibration axis — *on the table; in the running sweep, normalization untried* |

  **Verdict: BUILD L4a** — 4 g1-clean, g4-L4a-fixable ships (WNBA AST + NFL yards / fantasy-pp /
  fantasy-ud) from one ~½-day engineering build; **3 of the 4 are NFL** (the binding league, 5/20).
  This overturns the brief's "NO-GO on building it for AST as the breadth play" — that NO-GO was
  explicitly conditioned on AST being the *only* target; the screen (which the brief itself asked
  for) found the breadth. The L1-shippable batch (scale-only `< 0.05`) is unchanged: NBA {DREB
  0.036, FGM 0.043, FG3A 0.032}, WNBA {PTS 0.046, RA 0.030, REB 0.035, fantasy-pp 0.047}, NFL
  {carries 0.047, sacks-taken 0.045}. (All in-sample test-set floors — the honest verdict is the
  val→test retrain, never the proxy.)

  **Honest-retrain reconciliation (2026-06-07, first cell).** WNBA AST is the first screen row
  taken through the honest val-fit→test retrain (the combination search of §5). The screen's
  in-sample `joint(c,s) = 0.0397` was decoded under the cell's own `ratio_meanyr` normalization;
  the honest val-fit→test KS under `ratio_meanyr` is **0.126** — the in-sample fit oversold by
  ~0.086, an order of magnitude past the assumed +0.008–0.010 discount. The cell nonetheless
  **ships**, but under the *other* decodable normalization: `centered_additive_mean10` lands an
  honest g4 = **0.047** (5/5; g1 −0.017, g5 0.049), pending real-HPO confirm. Two lessons: (1) the
  in-sample screen floors are unreliable ship predictors for *overconfident* cells (WNBA AST
  central-50 coverage 0.376) — the val→test discount is cell- and normalization-specific and can be
  an order of magnitude larger than the uniform estimate; (2) the decisive axis here is
  **normalization, not calibration** — the same `(c,s)` machinery ships the cell under one target
  transform and fails it under another, which is exactly why the honest sweep scores *both*
  normalizations per cell rather than trusting the screened one. (An earlier search build re-fit
  `(c,s)` on the test rows and reported 0.039, reproducing the in-sample optimism; that dishonest
  path was removed — commit `10306ee`.)

  **Every row above is a live Ship-75 candidate — none is parked.** The screen varied only `(c, s)`;
  it did not touch normalization or blending. The sweep has already flipped NBA AST and WNBA DREB
  (under `centered_additive_mean10`); NFL receptions and passing-yards stay on the table for the
  **volume/rate normalization** (`ratio_projvol` — a per-carry / per-minute target reshapes the
  residual the calibration knob can't reach), the dormant **EB** normalization, and the **blending**
  axis. All of them are seed inputs for the combination search (§5); nothing waits in a Ship-90
  holding pen.

#### Lever 1 — research verdict (2026-06-06; brief `researcher_lever1_strategy.md` supersedes the v1 `researcher_dispersion_cal.md`)

**GO.** A textbook post-hoc scale calibration (engineering, not a research bet: Levi et al.
2022 arXiv:1905.11659; Kuleshov et al. 2018 arXiv:1807.00263). Two independent research passes
plus the C0 fix converged the build:

- **Route 1a-hybrid (reverses the v1 "route 1b" call).** Fit the dispersion scalar `c` *inside*
  [`_step_calibrate_dispersion`](../src/sportstradamus/training/pipeline.py) (remove the line-1512
  SkewNormal early-return), **objective = `scorecard._randomized_pit_ks` on the decoded
  predictive** for *both* branches, reusing the existing `dispersion_cal` pickle field + the
  [`_dispersion_calibrate`](../src/sportstradamus/prediction/model_prob.py) apply (drop its
  `dist == "SkewNormal"` clause; `Model Sigma *= dispersion_cal` after decode). Why 1a-hybrid
  now beats 1b: (1) C0 fixed the decode bug 1b was meant to dodge; (2) the fit objective *must*
  be the gate's own KS — a `SCALE_STAGE` in posthoc would either import scorecard (1a-hybrid in
  disguise) or re-duplicate the PIT math (the exact bug class C0 closed); (3) `dispersion_cal`
  is **orthogonal to `posthoc`**, so a cell stacks `roe_mean` (L2) **and** a scale fit (L1) —
  1b's single-slug `posthoc` structurally cannot, and the mean-then-width cells are a large part
  of the board. Constrain the fit to `g1_ci_hi ≤ baseline+0.01` and `g5 < 0.075`.
- **Do NOT use the Levi closed-form** σ-scaling (`α²=mean(resid²/σ²)`): empirically diverges
  5–7000× from the KS-optimal `c` on these skewed cells (WNBA DREB α=7158 vs c=1.43). Keep the
  KS `minimize_scalar`.
- **C0 decode bug — DONE** (this session): `_decode_sn_loc_scale` now dispatches the canonical
  `baselines` decode; `load_test_set` retains `Mean10`. DREB 0.504→0.153, receptions
  0.432→0.184 (verified), both still moderate-fail g4 → they route **L2 (`roe_mean`) → L1**
  (orthogonal stacking). Plain under-dispersion + mild low-loc bias; **Lever 5 not triggered.**
- **The count branch is systematically over-wide — a one-change-helps-21 lever.** All 21
  ZINB/NegBin cells over-cover (central-50 0.70–0.87) and every one improves under `c>1`;
  re-targeting the count `dispersion_cal` fit CRPS→PIT-KS tightens the whole branch (corrects
  v1's "coverage is wrong-sign" overstatement: PIT-KS and over-coverage agree on *direction*
  here; PIT-KS stays the objective because coverage-matching would over-narrow some lattices).
- **Whole-board tier map** (c-grid; discount ~20–30% for val→test, always retrain):
  - **Tier A — scalar flips:** NBA +6 (AST/DREB/FGM/FG3A/MIN/RA), WNBA +3 (PTS/REB/RA).
  - **Tier B — count CRPS→PIT-KS:** NBA +3 (FTM/STL/TOV), WNBA +3 (BLST/FTM/OREB), NFL +1 (passing-tds).
  - **Tier C — mean-then-width (L2→L1 stack):** WNBA DREB/AST, NFL receptions/receiving-yards/rushing-yards.
  - **Tier D — no scalar saves (needs c=1.5–2.0 ⇒ structural):** NFL attempts (0.214), qb-yards
    (0.227), completions, carries, receiving/rushing-yards — g1, not width, binds. → Lever 3/4d.
- **Fragility: 6 shipped cells fail a fresh re-score — a threshold cliff, not staleness**
  (pickle/CSV mtimes Δ=0.0m; PIT reproducible to 6 dp). NBA PRA/WNBA PA/WNBA PRA miss by
  +0.0002–0.0011 (literal flapping at 0.05); NBA PA +0.0073 (L1 rescues at c=1.05); **NFL
  interceptions actually passes g4, fails g1; NFL targets (g2+g4) is the one genuine
  regression.** See the new §8 hole #0 — the hysteresis-vs-demote policy is an **operator call**.
- **Honest breadth after L1 + count-retarget (post-discount), CALIBRATION AXIS ONLY:** NBA
  ~15–17 (**clears 16**), WNBA ~12–14 (**at-risk, zero margin** — L0 already banked), NFL ~7–9.
  NFL falling short of 15 *on the calibration axis* is **not a ceiling** — it is the signal
  that NFL's remaining cells need the **normalization** axis (volume / `ratio_projvol`: the g1
  gap on `attempts`/`carries`/`receiving-yards`/`rushing-yards`/`qb-yards` is most likely an
  efficiency-vs-volume conflation a rate target separates) and/or the **blending** axis, both
  untried. Do **not** pre-name these `deferred-90`; they are queued for the combination search,
  then the hierarchical layer if that stalls. The breadth number is a *calibration-axis*
  estimate, not a league verdict.

### Lever 2 — Post-hoc mean correction (already built)

**Mechanism.** `roe_mean` (affine) / `isotonic_mean` `MEAN_STAGE` correctors, selected
per cell via `posthoc` on `stat_meta.json`. Already shipped (NFL passing-tds,
interceptions).

**Targets.** The Gate-2/3 (bias) failures: NBA FGA (`g2_z` 0.67), NBA
fantasy-points-prizepicks (0.78), NFL fantasy-points-prizepicks (`g3_z` 0.52), NFL
qb-yards (`g3_z` 1.06). Run **before** Lever 1 on these — fix location, then width.

**Go/No-Go.** `g2_z`/`g3_z` < 0.5 on validation with the §1.4 BSS guardrail (reject if
Brier skill drops > 0.01). At NFL count means use affine ROE only (isotonic tails
overfit at low base rates, ref [48]).

**If it fails.** The bias is conditional, not a global affine/monotone shift → Lever 3
(opponent-defense interaction) is the structural fix.

### Lever 3 — Leakage-safe player-level features

**Mechanism.** Two features the leaf-average compresses away, mirrored exactly in
training and `get_stats` (leakage-safe, strict `<` date filter):
- **3a** `MeanYr_expanding_shifted` = `groupby(player_id).expanding().mean().shift(1)`,
  plus a `× opponent_team` variant; optional EB/James-Stein shrinkage (ref [47]) where
  expanding is sparse.
- **3b** opponent-defense × player interaction (`profile_market`) + a blowout/garbage-time
  flag from the book moneyline+spread.

**Targets.** The edge-problem cells where the model needs *more signal*, not more width:
the NFL g1+g4 volume markets (attempts, carries, completions, receiving-yards,
rushing-yards) and WNBA STL / NBA PF.

**Go/No-Go.** Retrain → re-score; ship per passer. SHAP importance < 0.001 ⇒ the feature
is inert for that cell → revert, don't carry dead features. Extend
`test_meanyr_mean10_leakage.py` before any ship.

**If it fails.** For NFL, escalate to Lever 4's per-position split (the breadth verdict's
note that QB/RB/WR don't share a "rushing-yards" generative process). For a cell where the
model still regresses the blend after features, lean harder on the book in the blend (ride
the sharp line; it ships if calibration holds) or document it (§7) — never "unwinnable."

### Lever 4 — Per-cell pivots (family / mode / shape / per-position)

For cells that survive Levers 1–3, the per-cell toolkit, each a one-field edit + retrain
behind `supersede_verdict()`:
- **4a** `zinb_mode: hurdle` for genuinely-inflated count cells (built, P2.B).
- **4b** family swap for cells whose *shape* (not width) is wrong — only on the §7a
  re-entry condition (still kills after the cheap fixes **and** conditional RQR variance
  < 0.70 **and** Poisson-GBM tracks the top decile while NB compresses).
- **4c** monotone priors (`monotone_priors.json`, layered default→league→market) to force
  mechanically-implausible splits out and stop Optuna wasting trials — for the NFL
  small-n volume cells. BSS guardrail per cell; wrong-sign prior is *worse* than none, so
  commit only priors with mechanical meaning (volume shares, plays-per-game).
- **4d** per-position model split (T11) — a live NFL lever (no longer held for Ship 90): train
  separate (position, market) models where eligible-position marginals diverge materially; min-row
  guard + fallback to pooled+categorical.

**Lever cap.** A cell that fails Levers 1–4 is exhausted *on the calibration + feature axis only* —
it stays a **live Ship-75 candidate** and is handed to the combination search (the normalization ×
calibration × blending matrix, §5) and then the hierarchical layer, with a one-line note naming the
axis already tried. It does **not** receive a `deferred-90` tag — that tag is **retired for this
operation** (§5 policy). A cell becomes Ship-90 territory only after the *whole matrix* plus the
hierarchical layer has actually failed it; **zero cells qualify today.** Gate-definition changes
never count as a lever.

### Lever 5 — Distribution / tail rebuild (sequenced last — on the table)

The T3 spliced/Pareto-tail or Student-t LSS head, or the CMPμ/MZINB family build. Heavy
inference-side work (references §12, "new distribution head"). Score only on cells that
reach it. Note this is the *calibration-axis* terminal lever; a cell that needs it (a genuine
heavy tail like NBA AST) may instead be reachable on the **normalization** axis (a volume/rate
target can change the residual-tail shape) — try the cheaper combination search before
committing to a new distribution head.

---

## 6. Per-league path to 75%

The arithmetic, grounded in the 2026-06-03 re-score. "Available" counts the cells a lever
can plausibly flip; the target gap is in parentheses. **Caveat for the reader:** the per-cell
`pit_ks` figures below are *in-sample* test-set floors from the L0/L4a screen and run
optimistic — the honest combination-search sweep (§5), now re-grounding every withheld cell
on its val-fit→test gates across both decodable normalizations, supersedes them. Treat the
counts as the pre-sweep estimate; the sweep's board is authoritative.

### NBA — 11/21, need +5 → 16

- **L1 dispersion** available on Gate-4-only cells: AST, DREB, FG3A, FGM, FTM, STL, TOV
  (MIN and RA already shipped via L1).
- **L2 then L1** on FGA, fantasy-points-prizepicks (g2 bias + g4).
- **L3** on PF (g1 + g5 — the one genuinely hard NBA cell).
- **Verdict: comfortable.** +5 of those seven L1 cells clears the target; FGA/FP/PF are
  backups, not load-bearing.

### WNBA — 8/18, need +6 → 14

- **L0 free promotes — DONE:** BLK, FG3M, TOV shipped → 8/18.
- **L1 / normalization** available on Gate-4-only cells: AST, BLST, DREB, FTM, OREB, PTS,
  RA, REB, fantasy-points-prizepicks. DREB screened severe (0.50) on the calibration axis but
  **ships under `centered_additive_mean10` in the sweep** (det 0.035) — not discounted. AST likewise
  ships under centered (val-fit→test g4 0.047), the sweep's lead candidate, pending real-HPO confirm.
- **L3** on STL (g1 edge).
- **Verdict: achievable.** +6 of the eight realistic L1/normalization cells = 14.

### NFL — 5/20, need +10 → 15 (the binding league)

- **L1 dispersion, g1 already passes** on **5** tractable cells: passing-tds,
  passing-yards, yards, fantasy-points-underdog, sacks-taken → **10/20**. (A 6th g4-only
  cell, receptions, screened severe at `pit_ks` 0.43 on the calibration axis — it is in the running
  sweep with the normalization axis untried, so it stays on the table.)
- **L1 + edge** on 6 g1+g4 cells: the 5 continuous-volume markets (attempts, carries,
  completions, receiving-yards, rushing-yards) + qb-tds. g1 is *marginal* (`ci_hi`
  0.007–0.018) and these are also under-dispersed, so L1 *might* pull g1 under 0.005 as a
  side effect — **uncertain, this is the crux** (§8 hole).
- **L3 / L4d** (features, per-position split) on the same 6 if L1 doesn't carry g1.
- **Hardest:** qb-yards, passing-first-downs (multi-gate), receptions (severe g4).
- **Verdict: hard, with a real failure mode.** L1 reliably gets NFL to ~10. The last +5
  depend on either (a) L1 incidentally fixing the marginal g1s by tightening calibration,
  or (b) L3/L4 giving the model enough real signal that blending it in stops regressing the
  sharp line (the g1 guardrail). The bar is *don't regress the blend*, not *beat the book*
  (§Purpose). **Plan for failure here:** if NFL stalls at 11–13, escalate to L4d
  (per-position split), and for any cell the model can't improve, lean on the book in the
  blend — that ships if calibration holds. No cell is shelved to `deferred-90` (the tag is retired
this operation), and never loosen a gate.

---

## 7. Failure protocol

- **Per-lever:** each lever above has an explicit go/no-go and an if-it-fails branch.
  When a lever's go/no-go fails on a cell, record it (lever-attempt +1) and take the
  branch. Do not grind a dead lever.
- **Per-cell:** push every cell until it ships or has actually failed across the **whole matrix**
  — all three axes (normalization × calibration × blending) plus the hierarchical layer. Four
  failed *calibration/feature* levers is **not** an exit: the cell moves to the combination search
  and the hierarchical layer with a one-line note naming the axes tried. No easy out, no infinite
  grind, and **no `deferred-90` shelf** — the only ways off the Ship-75 board are genuine
  matrix-wide exhaustion (zero cells today) or the operator's explicit, documented denominator call.
- **Operation-level: failure is not an option.** The §6 arithmetic shows NBA and WNBA
  clear with L0+L1 alone, with backups. NFL is the one league that can genuinely stall;
  its escalation ladder (L1 → L3 → L4d per-position) is deep enough to reach 15, and for
  any cell whose model can't be made to add to the blend, the fallback is to ride the sharp
  book line (ships if calibration holds); excluding a cell from the denominator is the
  **operator's** call, made explicitly and documented — never resolved by quietly loosening
  a gate.
- **Never loosen a gate to hit breadth.** `_GATE4_PIT_KS_DELTA` and friends are
  effect-size floors (vig-scale), not breadth knobs. Standing rule from Step 0.10: any
  search over bet-definition knobs must be multiplicity-corrected before it informs a
  ship, and we never gate on a model-conditioned statistic.

---

## 8. Research holes (flag for `research-analyst` before betting the plan on them)

These are the points where the plan rests on an *assumption* a focused study should
confirm or kill. Each is a place a lever could fail; knowing early reorders the queue.

> **Resolved 2026-06-06** (brief `researcher_dispersion_cal.md`, distilled into the §5 Lever 1
> verdict block): holes **#1** (feasible `c` exists for moderate cells; trade bounded, g5 binds
> first), **#2** (skip was a-priori, not a tested negative), **#3** (PIT-KS re-target flips the
> count cells; coverage is the wrong objective), and **#5** (the "severe" cells are a decode
> artifact, not a shape problem) are answered. A **new prerequisite hole #0** surfaced:
>
> 0. **The `centered_additive_*` Gate-4 PIT decode bug** — **RESOLVED (C0, this session).**
>    `_decode_sn_loc_scale` now dispatches the canonical `baselines` decode and `load_test_set`
>    retains `Mean10`; DREB 0.504→0.153, receptions 0.432→0.184. No shipped cell moved.
>
> **New hole #0b (highest priority — gate-policy, operator call): Gate-4 baseline hysteresis.**
> A fresh re-score fails 6 cells the `shipped` field still calls `devel` — 3 by +0.0002–0.0011,
> literal flapping at the hard 0.05 cutoff (a finite-sample KS statistic is a step function at
> the boundary; arXiv:2503.11673). Options: (a) an asymmetric **hysteresis band** — demote a
> *baselined* cell only at `pit_ks ≥ 0.05 + ε` (ε≈0.005, vig-scale), first-ship stays strict
> 0.05 (this is a deployment-stability tolerance à la arXiv:2403.19871, **not** a breadth
> loosening — pin with a golden test + reconcile with the monthly `gate-status` auto-demote);
> (b) ship L1 first (moves the 4 salvageable cliff cells to comfortable margin) and accept the
> 2 genuine demotions (NFL targets g2+g4, interceptions g1). **Decision pending — it gates
> trust in the ship-incrementally premise.**
>
> Holes #4 (do the marginal NFL g1s improve under calibration) and #6 (block-bootstrap backlog)
> remain open.
>
> **Updated 2026-06-07 (honest combination-search sweep).** Two methodological results from the
> first honest val-fit→test retrains (the §5 sweep), both bearing on how much to trust the
> in-sample screen that seeds this plan:
>
> - **The in-sample screen floors oversell, and not by a fixed discount.** The L4a screen and the
>   §6 per-cell `pit_ks` columns are 2-parameter fits on the *test* rows. For WNBA AST the
>   in-sample `joint(c,s) = 0.0397` became an honest val→test **0.126** under the cell's own
>   `ratio_meanyr` normalization — an 0.086 oversell, an order of magnitude past the assumed
>   +0.008–0.010. The oversell tracks model overconfidence (WNBA AST central-50 coverage 0.376),
>   so the floors are *rank* hints, not ship predictions. The honest sweep (val-fit→test gates,
>   both normalizations, deterministic ranker → real-HPO confirm) replaces them; nothing ships on
>   an in-sample floor. This *refines* hole #1: post-hoc scale **does** move PIT-KS with g1/g5
>   intact (WNBA AST 0.126→0.047, g1 −0.017, g5 0.049) — but only under the right normalization and
>   with a cell-specific val→test discount.
> - **Normalization is a first-class ship axis, often decisive over calibration.** WNBA AST ships
>   under `centered_additive_mean10` (0.047) and fails under `ratio_meanyr` (0.126) with the *same*
>   calibration machinery. The sweep scores both decodable normalizations per cell; the §6 counts,
>   computed against each cell's single default normalization, are a lower bound.
>
> **Predictive-variance regularization — not currently queued, but untried and NOT ruled out.** A
> lever was floated to "escape a local minimum where a worse (deterministic) model out-calibrates the
> sharper HPO model." *That specific premise* was a measurement artifact — an earlier search build
> re-fit calibration on the *test* rows and reported 0.039 for WNBA AST; the honest deterministic
> stand-in matches real HPO on like-for-like (g4 0.126 vs 0.123 under `ratio_meanyr`), so there is no
> worse-beats-better paradox to escape, and we are not prioritizing it. But the underlying
> overconfidence is real (WNBA AST central-50 coverage 0.376) and **no variance-regularization
> experiment has actually been run** — so the lever is *unproven, not refuted*. If the normalization
> and post-hoc-calibration axes (which the sweep exercises first) leave a cell overconfident, a
> training-time variance penalty is a legitimate untried option to pick back up.

1. **Will post-hoc scale calibration actually move PIT-KS without breaking g1/g5?**
   The central assumption of Lever 1. A quick study on 3–4 representative cells (one
   hair's-breadth, one moderate, one severe) before the full build: fit `c_opt` to
   coverage on validation, measure the g1/g5 trade. If widening reliably trades g4 for
   g1, the plan pivots to Lever 3 sooner. **Highest-priority hole.**
2. **Was the SkewNormal dispersion-calibration exclusion deliberate?** Two hardcoded
   skips (§3) suggest someone tried and backed out. Check git blame / archived context
   for a prior negative result before re-enabling — we may be about to repeat a known
   failure. (Cheap; do it first.)
3. **CRPS-vs-PIT-KS objective gap.** Quantify how much re-targeting the count-branch
   dispersion fit from CRPS to coverage moves the hair's-breadth ZINB cells (FTM/STL/TOV).
   If small, the count cells need a different lever than the SkewNormal ones.
4. **Do the marginal NFL g1s improve when dispersion is calibrated?** The difference
   between NFL reaching 10 and reaching 15. A targeted before/after Brier-CI on the 5
   g1+g4 volume cells after a candidate L1 fit.
5. **Severe-coverage cells: width or shape?** WNBA DREB (0.50), NFL receptions (0.43),
   qb-yards (0.23) are too far off for a scalar. Is SkewNormal's *tail weight* wrong
   (→ Student-t / spliced tail, Lever 5) or is it a feature/mislocation problem? A PIT
   histogram (not just the KS scalar) per cell answers this.
6. **The block-bootstrap / clustered-g1 backlog.** The test-set CSVs still lack a
   `game_date` column on combined-stat cells, blocking the player-clustered Gate-1 recheck
   and a closing-line-value gate. Resolving the training-data join unlocks both and is the
   one principled selection-style criterion worth building (Step 0.10).

---

## 9. Verification & supersession (unchanged guardrails)

**Always-on, every commit / PR:**

```bash
poetry run ruff check src/sportstradamus/
poetry run pytest tests/golden/          # incl. scorecard / gate tests
poetry run pytest -m integration         # fake-mode, no network
```

**Per-lever:** any change with an inference-side mirror (Lever 1 scale object, Lever 3
features) needs its live-path integration test green **before promotion** (references
§12). The determinism gate
([`test_determinism_gate.py`](../tests/integration/test_determinism_gate.py)) must stay
green; extend it to WNBA + NFL before any Lever-1+ ship — cross-league determinism is the
old plan's hard-won lesson.

**Tier-1 supersession.** Every change to a *baselined* (already-shipped) cell must clear
`supersede_verdict()` (S1 five gates + S2 paired-Brier CI + S3 paired-Sharpe) before
replacing the baseline. First-ships use Tier-0 absolute gates only. The
`devel-ship-curator` agent carves every per-cell PR; confirm its denylist covers any new
research scaffolding.

**`refactoring-specialist`** runs on every touched Python file before any push (CLAUDE.md
hard rule).

---

## 10. Reading list & cross-references

1. [`CONTRIBUTING.md`](../CONTRIBUTING.md) §Package Map
2. [`docs/STYLE_GUIDE.md`](STYLE_GUIDE.md)
3. [`docs/ship_gate.md`](ship_gate.md) — current g1–g5 thresholds (authoritative)
4. [`docs/operation_ship_references.md`](operation_ship_references.md) — research verdicts,
   citations [1]–[48], critical-files map, the per-change-type inference-path checklist
5. [`docs/operation_ship_90.md`](operation_ship_90.md) — next-rung stub; the levers it lists
   (T11 per-position, T3 tail head, CMPμ) are **on the Ship-75 table too** — pulled forward as
   needed, not reserved

Done = each league shows ≥ 75% (`shipped ∈ {devel, main}`) on a fresh scorecard, with
every promotion having cleared its go/no-go and, for baselined cells, `supersede_verdict()`.
