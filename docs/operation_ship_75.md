# Operation Ship 75

> **Home of record for the model-research push to 75% breadth.** Research verdicts,
> citations, and the inference-path checklist live in
> [`operation_ship_references.md`](operation_ship_references.md). Current gate
> thresholds live in [`ship_gate.md`](ship_gate.md). The next-rung stub is
> [`operation_ship_90.md`](operation_ship_90.md). Rewritten clean on 2026-06-03 after the
> Gate-4 PIT-KS redefinition reset the board, and amended continuously since; prior revisions
> are recoverable from git history (`git show 5c4a335:docs/operation_ship_75.md`).

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
fixing under-dispersion is central to the plan (§3, §5).

**Gate 1 is a no-regression guardrail, not a "beat the book" demand.** It is a
*non-inferiority* test: it certifies that blending our model into the sharp line does not
make the ensemble *worse* than the line alone. A tie passes — that is the entire point.
It follows that:

- A cell that fails Gate 1 is one where our current model is **regressing the blend** —
  adding noise, not signal (typically a small-sample, under-calibrated cell). The fix is a
  better model (calibration §5.2, real signal via features §5.8), or leaning harder on the book in
  the blend. It is **never** evidence that "the market is too efficient to win," and
  **never** a reason to loosen a gate.
- A sharp book therefore can never be a wall. The worst case is a cell where the model
  adds nothing and we ride the book line (a tie) — and even then the calibration work (g4)
  still matters, because the parlay builder still needs a well-shaped distribution around
  that line.

## Current standings

Per-cell status is not restated here — it drifts on every ship. Two canonical sources,
both live:

- **Ship state** (release surface per cell, all 59): the `shipped` field in
  [`stat_meta.json`](../src/sportstradamus/data/config/stat_meta.json) — `"withheld"` /
  `"devel"` / `"main"`. Committed, so git carries the history of every flip.
- **Per-cell gate numbers** (g1–g5, PIT-KS slack, `ship`): `data/training/model_stats.csv`,
  the VSCode-browseable mirror of the authoritative `model_stats.parquet`, rewritten by
  every `meditate`. Column dictionary in CLAUDE.md §"Training stats".

**The bar.** 75% of each covered league = **NBA 16 / 21, WNBA 14 / 18, NFL 15 / 20**. NFL
is the binding league (§6).

**Why the board is smaller than it once was — and that is correct.** A prior "43/59 = 73%"
was scored under a Gate 4 (`IQR(EV)/IQR(Result) > 0.5`) that measured sharpness, not
calibration, and waved through under-dispersed cells. The redefinition to a randomized-PIT
KS calibration gate (commit `2f1ecd4`) demoted the false passes (`cb077db`); the board is
now honest. Getting back above 75% is a model-quality job, and the binding constraint is
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
`g4_tail_pit_ks` (alt-over wobble), `g1_has_edge`, `betting_active`. Advisory diagnostics to add
(report-only, never a gate — the five gates are unchanged): Anderson-Darling PIT (tail-weighted, where
the alt-line ladder lives), conditional / stratified PIT-KS (by mean-decile / blowout / position /
home-away — the formal star/bench check), the non-randomized PIT for the count head (lower-variance on
the lattice; Czado-Gneiting-Held), and a CRPS reliability decomposition (miscalibration / discrimination
/ uncertainty; Arnold et al. 2024).

### 1d. Model-lever research — what is dead, what is alive

| Lever | Verdict | Source |
|---|---|---|
| Centered-target normalization (`centered_additive_mean10`) | **REOPENED — alive (overturns the P1 "dead" call).** P1 ruled it out as a *mean/level-compression* fix (FGA-only ship), a different objective scored before the Gate-4 PIT-KS redefinition; it was **never tried as a PIT-calibration lever**. The honest sweep (§5) finds it out-calibrates `ratio_meanyr` on Gate 4 for several withheld cells — a leading ship axis, not dead. Per-cell candidate status is the sweep board, never an in-sample floor. | Phase P1 → §5 sweep |
| `init_score` warm-start baseline | **Dead** — byte-identical to plain NegBin. | Phase P2.A |
| ZTNB-hurdle likelihood | **Refuted** — incompatible with the derived-π decode; would regress the 6 shipped hurdle markets. | Stage B1 |
| T5 multiplicative factorization (volume × efficiency) | **Killed** — Goodman variance-of-products gives +27% predictive-variance inflation on the priced cell. | Stage A1.5 |
| Family build (CMPμ / MZINB; SHASH / Johnson-SU; CMP / Generalized-Poisson) | **On the table — research-gated (§5.6).** Two distinct jobs the prior framing conflated: (a) top-decile *mean* compression is distribution-family-*invariant* (the tree leaf-average itself, refs [3][4][30]), so re-parameterizing dispersion alone won't move it; (b) but a both-directions count family (CMP / Generalized-Poisson) *is* the fix for the over-covering count cells the NegBin variance floor can't reach, and SHASH/JSU is the heavy-kurtosis continuous fix — the audit's correction to the mean-compression-only shelving. | §5.6 |
| HurdleZINB (per-cell mode) | **Alive** — shipped; 6/8 NBA ZINB markets. Available per-cell via `zinb_mode`. | Phase P2.B |
| Post-hoc **mean** correction (`roe_mean` / `isotonic_mean`) | **Alive & shipped** — `MEAN_STAGE` in [`posthoc.py`](../src/sportstradamus/training/posthoc.py); flipped NFL passing-tds + interceptions. | Stage B1.6 / Step 1 |
| Post-hoc **probability** recalibration (`prob_recal_*`) | **Alive** — `PROB_STAGE` built, available per-cell. | posthoc.py |
| Post-hoc **scale / dispersion** correction | **GO — route 1a-hybrid** (fit in `_step_calibrate_dispersion` via `scorecard._randomized_pit_ks`; reuse `dispersion_cal` field + apply; count branch CRPS→PIT-KS too). Reverses the v1 "route 1b" after C0 fixed the decode bug 1b dodged. Levi closed-form σ-scaling is a dead end (diverges 5–7000× on skewed cells). | §5.2 Rung B / brief `researcher_lever1_strategy.md` |
| Player-level features (expanding-mean, EB-shrunk, opp-defense) | **Alive, unbuilt** — RANK 2/3 in the breadth verdict. | Stage B1.6 |
| Per-position model split (NFL) | **Alive, on the table (T11)** — a live NFL lever now, not held for Ship 90; pull forward whenever the binding league needs it. | Stage A1.6 |

---

## 2. The reframe

The old plan was organized around *gate audits* because, at the time, we did not trust
the gates. That work is finished (§1a). **The gates are now correct, so from here the
plan is organized around model-quality levers**, cheapest-first, each shipping per cell
on a clean re-score.

The single most important consequence of the Gate-4 reset: the dominant *gate-failure
symptom* across all three leagues is no longer bias, ECE, or Brier — it is **predictive
under-dispersion** (the per-row distribution is too narrow), which the new PIT-KS gate
measures directly and the old IQR gate hid. Which *lever* best fixes that symptom is an
open tension (§3).

---

## 3. Diagnosis: under-dispersion is the dominant symptom

A fresh full-board re-score (all 59 test CSVs through `compute_gates`, the exact
production path) shows the failure modes are overwhelmingly concentrated on Gate 4, and
the coverage diagnostics point one direction: **too narrow.**

**Failure-mode census of the 40 withheld cells:**

| Primary failure | Count | Lever |
|---|---|---|
| **Gate 4 only** (g1/g2/g3/g5 all pass) | 24 | calibration §5.2 / normalization §5.4 |
| Gate 4 + Gate 1 (marginal g1, `ci_hi` 0.007–0.018) | 6 (all NFL) | calibration §5.2 → features §5.8 / blend §5.3 |
| Gate 4 + Gate 2/3 (bias) | 3 | mean §5.2-A then scale §5.2-B |
| Multi-gate (g1+g3+g4) | 2 (NFL passing-first-downs, qb-yards) | hardest; features §5.8 + per-position §5.6 |
| Gate 1 only / Gate 1+5 (edge) | 2 (WNBA STL; NBA PF also g5) | features §5.8 |
| **Pass now but un-promoted** | 3 (WNBA BLK, FG3M, TOV) | free (§5.1) |

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

### Root cause

The original root cause — the SkewNormal family received **no dispersion calibration at all** (two
hardcoded exclusions: `_step_calibrate_dispersion` early-returned `c_opt = 1.0` for SkewNormal, and
`model_prob._dispersion_calibrate` skipped it) — **is fixed**: the pipeline now fits the joint
`(c, skew_cal)` scale/skew against PIT-KS for SkewNormal and applies it on the served path (§5.2 Rung B,
shipped). What remains is the *residual* under-dispersion the scalar can't reach. Every SkewNormal cell
started on the raw GBDT scale — leaf-averaged, dynamic-range-compressed, too narrow (refs [3][4][30]) —
and the scalar `c` widens it, but on the moderate/severe cells the miss is *shape* or *location*, not
pure scale. The count families get `dispersion_cal` too, but the original objective was CRPS, not PIT-KS
(re-targeted in §5.2 Rung B) — minimizing CRPS does not guarantee a calibrated PIT.

**Which axis is primary is not assumed — the honest search adjudicates.** Two axes address the same
too-narrow PIT from different angles:

- **Calibration (§5.2).** Scale (and skew) the served SkewNormal width to a Uniform PIT. Ships the
  hair's-breadth cells (NBA RA, MIN) outright, but is close to exhausted as a *breadth* lever on its own.
- **Normalization (§5.4).** A centered / rate target reshapes the residual so the predictive is
  calibratable at all. The honest sweep finds `centered_additive_mean10` systematically out-calibrates
  `ratio_meanyr` on Gate 4 and ships moderate/severe cells (NBA AST, WNBA DREB) that **no scalar width
  fix reaches** — which is why §5 calls normalization "often decisive."

Plus the two axes the original diagnosis omitted: the **blend (§5.3)** — a flexibly-dispersive BLP pool
that widens the served predictive at the source — and **model/loss (§5.5)**. The honest combination
search (§5.0) scores all four axes per cell and lets the gate decide; read §3's heading as the *symptom*
(too-narrow PIT), not a verdict that calibration is the fix.

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
*reduce* Brier skill (g1) and shift ECE (g5). So calibration (§5.2) is fit to a calibration target and
guard-railed on g1/g5 — see §5.

---

## 5. The lever stack (cheapest-first; each ships per cell on a clean re-score)

Each lever names its **mechanism**, **targets**, a **go/no-go** measured on validation
*before* any ship, and an explicit **if-it-fails** branch. Levers are independent enough
that a failure on one does not block the others.

### The served predictive — four axes × two heads

A served predictive is built in four stages, `normalization → model/loss → blend → calibration`, each
independently swappable per cell
(`target_normalization ⊥ {dist_training_loss, variance_reg} ⊥ {blending_loss_fn, fused_loc/BLP, book-recovery} ⊥ {posthoc, dispersion_cal, skew_cal}`).
The subsections below (§5.1–§5.9) implement these four axes plus three research-gated tracks (family,
hierarchical, features) that feed the model stage.

The two modeling **heads** fail calibration in *opposite* directions (Czado-Gneiting-Held 2009), so
there is no single "widen everything" or "narrow everything" fix:

- **Continuous (SkewNormal)** — *under-dispersed* (PIT U-shaped, central-50 below nominal). Needs
  widening.
- **Count (ZINB / NegBin / ZAGamma)** — *over-covers* (PIT inverted-U, central-50 above nominal). Needs
  narrowing — and the negative-binomial conditional variance is bounded below by its mean, so a cell
  whose truth is near-equidispersed or tighter is *forced* too wide: the deepest count cells need a
  both-directions family (§5.6), a structural ceiling post-hoc cannot clear.

The four axes, each detailed in its own subsection:

- **Normalization** (§5.4) — the target transform the model fits (`ratio_meanyr`,
  `centered_additive_mean10`, `centered_additive_eb_meanyr_k10`, and the unbuilt `ratio_projvol`).
  Retrain; searched.
- **Model/loss** (§5.5) — how the model trains its own predictive: `dist_training_loss` (`nll`/`crps`)
  and the unbuilt training-time variance / soft-calibration regularizer. Retrain; loss searched.
- **Blend** (§5.3) — how the model predictive meets the book: `blending_loss_fn`, the `fused_loc`
  pooling operator, the book-distribution construction, and the unbuilt BLP recalibration wrapper. The
  under-built axis (§5.3 details the five current immaturities). Retrain (weight) + research-gated
  (structure).
- **Calibration** (§5.2) — post-hoc on the served predictive, a location → scale+shape → full-CDF
  ladder. Auto-fit / free. Close to exhausted as a *breadth* lever on its own — it fixes
  width/shape/location, never signal or location-scale structure.

**The strategy research board — interface.** The per-market searcher is
[`training/model_strategy_driver.py`](../src/sportstradamus/training/model_strategy_driver.py)
(entry point `model-strategy-driver`). `model-strategy-driver --board` searches the default
covered-league board (the withheld SkewNormal cells the lever can reach) and writes the ranked
board to `data/research/strategy_research_board.csv`, appended after each cell so an interrupt keeps
partial progress; `--league L --market M [--out PATH]` runs a single cell. Each corner trains one
`meditate --deterministic --target-normalization … --dist-training-loss {nll|crps} --blending-loss-fn
{nll|crps} --bypass-withholding`, where `--deterministic` pins the RNGs and the fixed fast
hyperparameters and writes to a sandbox (`research/models/deterministic/` +
`data/test_sets/deterministic/`) so a trial never clobbers a production market.

**Read the board for results — this plan does not restate them.** Per-cell slack, ship verdict, and
Gate-4 PIT-KS drift every run and live in `strategy_research_board.csv` (and in `model_stats.csv`
once a cell is confirmed). What the plan fixes is the *method* and its current executable state:

| Axis | Values | Executable today | Unbuilt |
|---|---|---|---|
| **Normalization** (retrain) | `ratio_meanyr`, `centered_additive_mean10`, `centered_additive_eb_meanyr_k10`, `ratio_projvol` | 3 of 4 carry a Gate-4 SkewNormal decode (`scorecard._decode_sn_loc_scale`; EB off the dumped `GlobalMean`) | `ratio_projvol` (target = y / projected-volume) |
| **Model/loss** (retrain) | dist-loss `nll`/`crps`; variance / soft-cal regularizer | dist-loss via `--dist-training-loss` | variance / MMD-to-uniform-PIT regularizer |
| **Blend** (retrain weight + research-gated structure) | blend-loss `nll`/`crps`; `fused_loc` pool; book-distribution recovery; BLP wrapper; p_book noise | blend-loss via `--blending-loss-fn` (`fit_model_weight_crps` built); current `fused_loc` parameter pool | density-LOP fix, power de-vig, book recovery, BLP, p_book noise; free post-hoc `w`-refit |
| **Calibration** (auto-fit, not searched) | location (`roe_mean`/`isotonic_mean`); scale+shape (`dispersion_cal` + joint `skew_cal`); full-CDF (isotonic-PIT / IDR) | location + scale+shape shipped, baked into the dump, read by the gate | isotonic-PIT / IDR full-CDF recal; honest-val mode sweep |

The driver searches the **retrain** axes (normalization × dist-loss × blend-loss) as a `GridSampler`
grid (≤12 discrete corners — exhaustive and deterministic; the `[kind, spec, stage]` `SEARCH_SPACE`
flips to TPE the moment a continuous axis lands). Calibration is auto-fit per corner, **not** a searched
axis; the blend *structure* (BLP, book-recovery) and the variance regularizer are research-gated builds,
not corners yet. The one
genuinely structural unbuilt lever — a research-gated **hierarchical-Bayes** layer (learn the
shrinkage per group; pool the full predictive across player ← position ← team rather than the mean
only) — is the answer to the small-sample NFL wall and is scoped by a `research-analyst` brief
before any build.

**Nothing is deferred. Every withheld cell and every lever is a live Ship-75 candidate.** The
`deferred-90` / "defer" / Lever-cap tags are **retired for the duration of this operation.** They
were always per-axis verdicts — almost always the *calibration axis under a single normalization* —
and the honest sweep showed how badly that mis-reads: cells the screen stamped `deferred-90`
ship under a *different* normalization. A cell leaves the Ship-75 board **only** after it has
actually failed on **all four axes** (normalization × model/loss × blend × calibration) **plus** the
family and hierarchical tracks — and that is true of **zero cells** today. Any "defer",
"deferred-90", "cannot reach", or "efficient-market wall" wording that survives below is the *old*
per-axis verdict, kept for its evidence but **superseded by this policy**; read none of it as final.
The bar for parking a cell is axis-exhaustion across the whole matrix, never a single screen.

**Implementation.** The cost asymmetry is the design's core: a calibration mode is a post-hoc
transform fit in milliseconds, so it never sits in the training loop — only `normalization × loss`
costs a train (≈ a handful of trains per cell, not a full cross-product). Per market the driver runs
an Optuna `GridSampler` over the categorical retrain grid; each **trial** is one `--deterministic`
train (bit-reproducible, fast, never published), and the objective is the negative **min-gate
slack** — a single scalar, positive iff the corner ships and larger the more headroom it has across
all five gates at once, so it optimizes "ships, with margin," not Gate 4 alone.

Each corner is scored by the **honest val-fit→test gate row**
([`model_strategy_search._score_normalization`](../src/sportstradamus/training/model_strategy_search.py)):
the deterministic dump already carries the pipeline's own validation-fit joint calibration, so the
ranker calls `scorecard.gate_row` on it — the *same* code production ships on — and reads
`min_gate_slack`. No test re-fit, so the deterministic score tracks real HPO on like-for-like by
construction. An earlier build instead re-fit the four calibration modes on each trial's *test* rows
and reported the best; that oversold the screen and was removed (commit `10306ee`). The driver is now
a faithful fixed-HP replica of the production HPO pipeline — same calibration, same gate, same decode
— the only differences being the fixed hyperparameters in place of the Optuna search and the sandbox
write locations.

**The deterministic study only ranks; the real-HPO scorecard ships.** Take the top-K (K ≈ 2–3)
corners per cell, re-run each under real HPO, and ship the first that clears the official 5-gate
scorecard (`model_stats.csv`) — never the deterministic score. Knife-edge cells (g4 within ~0.003 of
0.05) are exactly where the val→test discount bites, so the confirm is mandatory. Sweep the **whole**
board, shipped cells included — a shipped cell may have a better corner than the scale-only default
it settled for.

**The operating loop (per parameter, per cell).** Once an axis is wired, the per-cell workflow is
fixed — and the ship bar differs for a withheld cell vs an incumbent:

1. **Driver board = candidate generator.** `model-strategy-driver` returns the ranked board
   (one deterministic train per corner, scored on the honest val-fit→test gate). A `ships=True` row
   is a *candidate flag*, never a ship — the real gate is the full-HPO official 5-gate scorecard
   (`model_stats.csv` / `.parquet`). Carry the **top-K (2–3)** corners per cell forward.

2. **Withheld cell → real-HPO confirm → ship to devel.** Set the winning corner's
   `target_normalization` (and `posthoc`) in `stat_meta.json` and pass any non-default
   `--dist-training-loss` / `--blending-loss-fn` to a full-HPO `meditate`, then read the official
   scorecard. A clean **5/5** → flip
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
defined value before `fit_model_weight_crps` existed; building it made it sweepable — but on the
*retrain* tier, not the brief's free post-hoc tier,
because the deterministic dump doesn't yet carry the pre-blend components a free `w`-refit needs. So
"wire it in" is sometimes "wire the axis, build the value, decide the tier — then it sweeps."

### §5.1 — Lever 0: re-score every withheld cell; promote free passers

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

### §5.2 — Calibration axis: the post-hoc ladder (location → scale+shape → full-CDF)

Post-hoc transforms on the served predictive, free or near-free (fit on validation, applied to the
disjoint test split). One ordered ladder, both heads in *opposite* directions; close to exhausted as a
*breadth* lever alone (fixes width/shape/location, never signal or location-scale structure).

**Rung A — location (`roe_mean` / `isotonic_mean`, shipped).** Affine ROE / isotonic `MEAN_STAGE`
correctors, selected per cell via `posthoc` in `stat_meta.json` (shipped: NFL passing-tds,
interceptions). Targets the Gate-2/3 bias failures (NBA FGA `g2_z` 0.67, NFL qb-yards `g3_z` 1.06). At
NFL count means use affine ROE only (isotonic tails overfit at low base rates, ref [48]). **Operator
note — skeptical:** post-hoc correction of the *predicted mean* edits the central tendency the model is
supposed to learn, so a cell that ships only because its mean was patched is suspect; prefer fixing
location at the source (normalization §5.4, features §5.8) first, carry mean correction as
available-but-last-resort, and ship a mean-corrected cell only if it also holds the g1 BSS guardrail and
survives the val→test discount.

**Rung B — scale + shape (joint `(c, skew_cal)`, shipped — route 1a-hybrid).** Fit a per-cell scale `c`
so the predictive PIT is Uniform, applied after decode and `fused_loc`, inside
[`_step_calibrate_dispersion`](../src/sportstradamus/training/pipeline.py) with objective =
[`scorecard._randomized_pit_ks`](../src/sportstradamus/training/scorecard.py), reusing the
`dispersion_cal` field and the
[`model_prob._dispersion_calibrate`](../src/sportstradamus/prediction/model_prob.py) apply. The additive
skew `skew_cal` on the served SkewNormal `alpha` (the code's "Lever 4a") is fit *jointly* with `c`
against the same PIT-KS — additive not multiplicative because the direct parameterization has a
Fisher-information singularity at `alpha = 0` (Hallin & Ley 2014, arXiv:1209.4177), so `alpha × k`
injects no skew; joint strictly dominates sequential on the binding under-skew cells (the scale-only KS
optimum lands where the skew gradient vanishes). Fit in centered-skewness space (clamp `|s| ≤ 3`),
warm-start `(c, 0)`, ship at val `pit_ks < 0.040` (val→test discount +0.008–0.010). Fit, dump, and gate
all act on the *served* (model × book) predictive — `c` scales the served sigma, re-encoded to
normalized space (`baselines.encode_*`) so the scorecard decode recovers it byte-for-byte. The Levi
closed-form σ-scaling is a dead end (diverges 5–7000× from the KS-optimal `c` on skewed cells; keep the
KS `minimize_scalar`). **Re-target the count objective:** the count-branch `dispersion_cal` minimized
CRPS, but the gate is PIT-KS — re-targeting the fit to PIT-KS is the one change that tightens the whole
over-wide count branch (all 21 ZINB/NegBin cells over-cover).

**Rung C — full CDF (isotonic-PIT / IDR, new, free).** The scalar `(c, s)` is the bottom of an
expressiveness ladder: a monotone spline on the PIT (Kuleshov 2018) or isotonic distributional
regression as a recalibrator (Henzi, Ziegel & Gneiting 2021) recalibrates the *whole* predictive CDF —
the entire alt-line ladder the parlay builder prices — not just the single-line over-probability the
existing `prob_recal_isotonic` fixes. It acts on `Z = F(Y|X)`, the randomized PIT for counts, so it
ports to both heads (widen SkewNormal / narrow count) unchanged. Prefer isotonic / IDR over conformal
*calibration* on the count lattice — conformal yields discontinuous randomized CDFs (Marx 2022), bad for
pricing a ladder. (Conformal *predictive distributions* for the alt-line ladder are a separate, deferred
item — roadmap v3 §8 deferred register.)

**Go/No-Go (per cell, on validation, before ship).** `pit_ks` crosses below `max(0.05, 1.358/√n)`
**and** g1 Brier-skill drops by ≤ 0.01 (the §1.4-class guardrail) **and** g5 ECE stays < 0.075. Any
served calibration object (scale, skew, the Rung-C map) must be written to the test-CSV dump
(`_step_persist_artifacts`) and applied identically at inference — pickle key + legacy fallback +
byte-identical round-trip test (references §12).

**If it fails** (widening flips g4 but breaks g1/g5, or `pit_ks` won't move because the mislocation is
mean/skew not scale). Route bias-direction cells (g2/g3 failing) through Rung A first; route cells that
trade g4 for g1 to features (§5.8, signal not width); where neither scale nor skew clears g4 the residual
is *shape* the SkewNormal can't reach by `(c, s)` — try the normalization axis (§5.4) and the blend axis
(§5.3, BLP), escalate to a family rebuild (§5.6) only on matrix-wide exhaustion. Do not pre-tag such
cells dead (§5.9).

### §5.3 — Blend axis: the under-built lever

Of the four axes this is the least developed, and it bears directly on the dominant symptom: the blend
is where the sharp book's information meets the model's too-narrow predictive. **Structure changes here
are research-gated** (they alter the dispersion mechanism — `research-analyst` brief first).

**Current state — `fused_loc` is a full but immature parameter-space pool, not a density pool.**
[`fused_loc`](../src/sportstradamus/helpers/distributions.py) blends summary *parameters*, with five
weaknesses:

1. **Parameter blend ≠ density pool.** The NegBin path log-blends `(μ, r)` and calls itself a
   "logarithmic opinion pool (Genest & Zidek 1986)," but a true LOP multiplies the PMFs/PDFs pointwise
   and renormalizes — parameter-blending coincides with the LOP only in the Gaussian case. So the
   SkewNormal loc/σ precision-blend *is* a correct weighted Gaussian LOP, but NegBin/ZINB is an
   approximation wearing the label.
2. **Crude book distribution.** The book side is a symmetric `N(ev_b, (ev_b·cv)²)` with `cv` a single
   per-cell constant from `stat_calibration.json` — not line-specific; skew forced to 0. The actual
   two-way de-vigged price never shapes the book's spread or skew.
3. **Skew is a linear shrink.** `blended_skew = w·model_skew` — leaning on the book decays skew toward
   symmetric; no third-moment pooling.
4. **The pool can only sharpen, never widen.** Precision/log pooling is ≤ the least-dispersed component;
   widening the under-dispersed model is bolted on afterward as the separate scalar `c_opt` (Rung B). No
   flexibly-dispersive capacity in the pool itself.
5. **No noise model on the book price.** `w` (`model_weight`) is a learned per-cell constant; no
   per-observation book precision from how noisy `p_book` is in that market.

The de-vig itself (`no_vig_odds`) is **proportional** (`o/(o+u)`), and one-sided lines fabricate a flat
6.5% under.

**Fitting half — turn one de-vigged point into a distribution.** The book gives one point on the implied
CDF (the line's de-vigged over/under split). (a) Replace proportional de-vig with the **power
(logarithmic) method** — preserves [0,1] and handles favourite-longshot bias on asymmetric / anytime-TD
lines (Clarke, Kovalchik & Ingram 2017), where proportional / Shin can return out-of-range
probabilities; flag cells with `|p_over − 0.5| > 0.3` for extra downstream shrinkage. (b) Recover the
book *distribution* by fixing the model's shape `(σ̂, α̂)` / `(θ̂, π̂)` and solving the single location so
the model-shaped CDF passes through the de-vigged point (1-D root-find; the line is a *median*, not a
mean, which is why skew matters). (c) Count tail case (anytime-TD): `λ = −log(1−p)` / NegBin
`μ = θ((1−p)^(−1/θ)−1)` are ill-conditioned (`dλ/dp = 1/(1−p)` blows up as p→1) — regularize toward the
model's `μ̂` and cap `|μ_book − μ̂| ≤ K·SD`.

**Pooling half.** (a) Keep the log pool as base operator but **fix NegBin/ZINB to a real density LOP**
(grid-multiply the PMFs and renormalize) instead of the parameter log-blend. (b) Add the
**beta-transformed linear pool (BLP)** wrapper `F^BLP = B_{α,β}(w·F_model + (1−w)·F_book)` (Ranjan &
Gneiting 2010): flexibly dispersive — it can narrow *or* widen the pool, learned from history — and is
fit **outside** the five gate calculations (it changes the gate inputs, not the gates). This is the
principled under-dispersion fix that supersedes the bolt-on scalar widener. **Do not** substitute a raw
*linear* pool: its widening is disagreement-driven, not legitimate uncertainty, and degrades sharpness
and the KS/ECE gates when book and model disagree (Gneiting & Ranjan 2013; Hora 2004). (c) **Conjugate
noise on `p_book`:** treat `logit(p_book)` as a Gaussian observation on the model's logit-CDF at the
line, per-cell variance from residual studies → precision weight; "book is noisy in this market" falls
out as lower precision. (d) Learn `w` by CRPS (continuous) / log-score (count), shrunk per-cell toward a
global prior ∝ 1/n_cell; do not hard-code book = truth (props are soft). (e) **Time-varying `w_book`**
ramp toward close (the de-vigged *close* is the best probability estimate, but at *close*). The CLV-edge
*dashboard* defers to the roadmap v3 §8 register; the weight schedule stays here.

**Go/No-Go.** `pit_ks` below threshold with g1 BSS drop ≤ 0.01 and g5 < 0.075, on validation, before
ship; an inference-path round-trip test for every new served object (de-vig method, recovered book
params, BLP coefficients).

**If it fails.** A cell the blend can't widen into calibration without a g1 hit is signal-starved →
features (§5.8) or normalization (§5.4); a genuine heavy tail the BLP can't reach → family (§5.6).

### §5.4 — Normalization axis

The target transform the model fits — well-defined but under-explored. `ratio_meanyr`,
`centered_additive_mean10`, `centered_additive_eb_meanyr_k10` carry a Gate-4 SkewNormal decode; the EB /
hierarchical-shrinkage strategy is built, decode-tested, and assigned to **zero** production cells.

**Build `ratio_projvol` (the most likely NFL volume unlock).** Modeling `points` (or `points /
season-mean`) conflates *efficiency × opportunity*. A `ratio_projvol` strategy — target =
`y / projected-volume` (a per-minute / per-carry / per-target rate), decode = `rate × projected-volume`
— separates the stable efficiency signal from the matchup-driven volume signal; on the count side use
`log(volume)` as a **GLM offset** and delta-method (or Monte-Carlo) back to totals scale for the gate.
The volume projections **already exist** (`proj_*` features: projected carries/targets/minutes) — used
as *features*, never as the normalization *denominator*. A `rushing-yards` g1 block is plausibly a
volume/efficiency conflation the book prices and we do not. Decode + leakage tests before any ship.

### §5.5 — Model/loss axis

How the model trains its own predictive, upstream of the blend.

**Training loss `nll` vs `crps` per cell.** Min-CRPS is more robust to misspecification, max-likelihood
slightly more efficient under correct specification (Gebetsberger 2018); keep the per-cell winner.
Honest caveat: LightGBMLSS's CRPS path sets the Hessian to 1 (first-order, discards loss curvature),
which is *why* a properly-curved CRPS head (NGBoost's natural gradient) is the narrow-use Hail-Mary
(§5.6 / roadmap v3 §8), not the default.

**Training-time variance / soft-calibration regularizer (unbuilt; the lever §8 flags as "untried, not
refuted").** Concrete form: an MMD-to-uniform-PIT penalty (Chung 2021) or a held-out variance penalty
that widens σ where the model is overconfident — attacking under-dispersion at the source rather than
post-hoc. Pick it up if normalization and the blend axis leave a cell overconfident.

**CRPS-stacking** the model's own CDF variants across loss × transform (Gneiting & Ranjan 2013), via a
log / beta-transformed pool (same dispersion logic as §5.3, never a raw linear pool).

### §5.6 — Family / distribution axis (research-gated)

The model's distributional head. Every item here changes a family or dispersion mechanism → a
`research-analyst` brief gates the build (CLAUDE.md research-first). Each is a one-field edit + retrain
behind `supersede_verdict()`.

**Continuous, by escalating expressiveness.** (a) **Centered-parametrization SkewNormal / skew-t**
(Arellano-Valle & Azzalini 2008) — a *loss-function* change that removes the `alpha = 0`
Fisher-information singularity at the source (distinct from, and complementary to, the post-hoc additive
`skew_cal` patch of §5.2); **try first** among family moves. It fixes the *singularity*, **not the
tails**. (b) **SHASH / Johnson SU** (Jones & Pewsey 2009) — 4-parameter, separately governing skew and
kurtosis — for the heavy-kurtosis cells (NBA AST) the centered family still leaves too thin. (c)
**skew-t / Student-t** for the heaviest tails.

**Count — the structural ceiling.** The negative-binomial conditional variance is bounded below by its
mean, so over-covering count cells that won't narrow under recalibration need a both-directions family:
**COM-Poisson** (Sellers & Shmueli 2010), **Generalized Poisson** (Harris 2012, cheaper — no infinite
normalizing constant), or **Double Poisson** (Efron 1986). COM-Poisson's infinite `Z(λ,ν)` must be
truncated and round-trip-tested on the live path. This corrects the §1d lever-table shelving: `CMPμ` is filed there (the 'Family build' row)
as a *mean-compression* family, but it is also — and here primarily — the count *dispersion-direction*
fix. Run a per-cell **plain-NB vs hurdle vs ZINB vs COM-Poisson** screen on the honest val→test PIT;
stop defaulting ZINB on cells that aren't genuinely zero-inflated (a ZINB mixture inflates variance to
fit zeros a single process already explains, feeding the over-coverage). Hurdle already exists via
`zinb_mode` (built, P2.B).

**Other.** Tweedie / generalized-gamma heads for zero-mass continuous cells (NFL RB2 rush yards).
Monotone priors (`monotone_priors.json`, layered default→league→market) for the NFL small-n volume cells
— commit only priors with mechanical meaning (a wrong-sign prior is worse than none). Per-position model
split (T11) where eligible-position marginals diverge materially (min-row guard + fallback to
pooled+categorical).

### §5.7 — Small-sample / hierarchical layer (the NFL wall)

Partial pooling dominates both no-pooling and complete-pooling at n ≈ 300–1000 per group (Gelman & Hill
2007). Cheapest-first:

- **EB-shrink the distributional parameters** (μ, σ, ν, τ) per player toward a per-position mean, with
  cross-validated shrinkage strength — stays in the LightGBMLSS stack.
- **TabPFN v2 head-to-head** on the small-n NFL / WNBA cells (Hollmann 2025): a tabular foundation model
  returning a native full predictive distribution with no per-cell tuning, sweet-spot ≤ ~10k rows — a
  per-cell tool for the *small* cells, not the data-rich NBA (where it loses to tuned GBDTs). Recalibrate
  its output through the existing PIT gate; judge on the same honest val→test ship criterion. Try
  *before* the full hierarchical build (lower friction, same Bayesian small-n logic). Keep GBDTs the
  backbone — the data is tabular and medium-sized, GBDT's home turf (Grinsztajn 2022; McElfresh 2023).
- **Hierarchical-Bayes** layer (player ⊂ position ⊂ team) — research-gated escalation if EB shrinkage and
  TabPFN are insufficient. (The multi-task shared-trunk NN that pools across cells/leagues defers to the
  roadmap v3 §8 register.)

### §5.8 — Features: leakage-safe player-level

Two features the leaf-average compresses away, mirrored exactly in training and `get_stats`
(leakage-safe, strict `<` date filter): **3a** `MeanYr_expanding_shifted` =
`groupby(player_id).expanding().mean().shift(1)`, plus a `× opponent_team` variant and optional
EB / James-Stein shrinkage (ref [47]) where expanding is sparse; **3b** opponent-defense × player
interaction (`profile_market`) + a blowout / garbage-time flag from the book moneyline+spread. Targets
the edge-problem cells where the model needs *more signal*, not more width: the NFL g1+g4 volume markets
(attempts, carries, completions, receiving-yards, rushing-yards) and WNBA STL / NBA PF. Retrain →
re-score, ship per passer; SHAP importance < 0.001 ⇒ inert → revert, don't carry dead features. Extend
`test_meanyr_mean10_leakage.py` before any ship. If features don't carry g1 for an NFL cell, escalate to
the per-position split (§5.6); for a cell the model still can't improve, lean harder on the book in the
blend (ride the sharp line; ships if calibration holds) — never "unwinnable."

### §5.9 — Lever cap & matrix-exhaustion policy

A cell that fails one axis is exhausted *on that axis only* — it stays a **live Ship-75 candidate** and
moves to the next axis, then to the family / hierarchical tracks, with a one-line note naming the axes
already tried. The only ways off the board are genuine matrix-wide exhaustion across **all four axes**
(normalization × model/loss × blend × calibration) **plus** the family and hierarchical tracks — true of
**zero cells** today — or the operator's explicit, documented denominator call. No `deferred-90` tag
(zero cells qualify for Ship-90 territory today); the heaviest tail-head rebuilds (spliced/Pareto-tail,
MZINB) are the deferred long-shots (roadmap v3 §8 deferred register), tried only after the cheaper
family (§5.6) and normalization (§5.4) moves. Gate-definition changes never count as a lever.

---

## 6. Per-league path to 75%

Live counts are in `stat_meta.json`; per-cell gate numbers and current ship candidates are
in `model_stats.csv` and the §5 sweep board. What follows is the durable routing — which
cells each lever can plausibly flip, and the per-league verdict — not a snapshot of the count.

### NBA → 16 / 21

- **Calibration (§5.2)** on the Gate-4-only cells: AST, DREB, FG3A, FGM, FTM, STL, TOV (MIN and RA
  already shipped via the scale fit).
- **§5.2 Rung A then Rung B** on FGA, fantasy-points-prizepicks (g2 bias + g4).
- **Features (§5.8)** on PF (g1 + g5 — the one genuinely hard NBA cell).
- **Verdict: comfortable.** Five of those seven calibration cells clear the target; FGA/FP/PF are
  backups, not load-bearing.

### WNBA → 14 / 18

- **Free promotes (§5.1)** banked the first re-score passers.
- **Calibration (§5.2) / normalization (§5.4)** on the Gate-4-only cells: AST, BLST, DREB, FTM, OREB,
  PTS, RA, REB, fantasy-points-prizepicks. Cells the calibration axis alone can't reach route to the
  **normalization** axis (`centered_additive_mean10`) in the §5 sweep; current candidate status is the
  sweep board, never an in-sample floor.
- **Features (§5.8)** on STL (g1 edge).
- **Verdict: achievable** — six of the eight realistic calibration/normalization cells.

### NFL → 15 / 20 (the binding league)

- **Calibration (§5.2), g1 already passes** on the tractable cells: passing-tds, passing-yards,
  yards, fantasy-points-underdog, sacks-taken. A 6th g4-only cell, receptions, screened severe on the
  calibration axis but stays on the table with the normalization (§5.4) and blend (§5.3) axes untried.
- **Calibration + edge** on the g1+g4 cells: the continuous-volume markets (attempts, carries,
  completions, receiving-yards, rushing-yards) + qb-tds. g1 is *marginal* and these are also
  under-dispersed, so the scale fit *might* pull g1 under the threshold as a side effect — **uncertain,
  this is the crux** (§8 hole #4). The blend axis (§5.3 BLP) and normalization (§5.4 `ratio_projvol`)
  are the structural follow-ups here.
- **Features (§5.8) / per-position split (§5.6)** on the same cells if calibration doesn't carry g1.
- **Hardest:** qb-yards, passing-first-downs (multi-gate), receptions (severe g4).
- **Verdict: hard, with a real failure mode.** Calibration reliably gets NFL most of the way; the last
  stretch depends on either (a) the scale fit incidentally fixing the marginal g1s by tightening
  calibration, or (b) features / blend / per-position giving the model enough real signal that blending
  it in stops regressing the sharp line. The bar is *don't regress the blend*, not *beat the book*
  (§Purpose). **Plan for failure:** if NFL stalls, escalate to the per-position split (§5.6) and the
  blend rebuild (§5.3), and for any cell the model can't improve, lean on the book in the blend — that
  ships if calibration holds. No cell is shelved (§5.9 standing policy); never loosen a gate.

---

## 7. Failure protocol

- **Per-lever:** each lever above has an explicit go/no-go and an if-it-fails branch.
  When a lever's go/no-go fails on a cell, record it (lever-attempt +1) and take the
  branch. Do not grind a dead lever.
- **Per-cell:** push every cell until it ships or has actually failed across the **whole matrix**
  — all four axes (normalization × model/loss × blend × calibration) plus the family and hierarchical tracks (§5.9
  standing policy). Four failed *calibration/feature* levers is **not** an exit: the cell moves to
  the combination search and the hierarchical layer with a one-line note naming the axes tried. The
  only ways off the Ship-75 board are genuine matrix-wide exhaustion (zero cells today) or the
  operator's explicit, documented denominator call.
- **Operation-level: failure is not an option.** The §6 arithmetic shows NBA and WNBA
  clear with §5.1+§5.2 alone, with backups. NFL is the one league that can genuinely stall;
  its escalation ladder (§5.2 → §5.8 → §5.6 per-position, plus the §5.3 blend rebuild) is deep enough
  to reach 15, and for
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

**Highest-priority open hole — #0b, Gate-4 baseline hysteresis (gate-policy, operator call).** A fresh
re-score fails several cells the `shipped` field still calls `devel`, some by a hair (a finite-sample KS
statistic is a step function at the hard 0.05 cutoff; arXiv:2503.11673). Two options, both the
operator's explicit call (never a quiet gate-loosen): (a) an asymmetric **hysteresis band** — demote a
*baselined* cell only at `pit_ks ≥ 0.05 + ε` (ε≈0.005, vig-scale; first-ship stays strict 0.05), a
deployment-stability tolerance (arXiv:2403.19871) pinned by a golden test and reconciled with the
monthly `gate-status` auto-demote; (b) ship the §5.2 scale fit first (moves the salvageable cliff cells
to comfortable margin) and accept the genuine demotions. Pending — it gates trust in the
ship-incrementally premise.

The methodological findings that seeded this plan are now baked into §5 and need no separate hole: the
in-sample 2-parameter screen oversells the honest val→test gate (cell-specific, not a fixed discount), so
nothing ships on an in-sample floor (§5.0 operating loop); normalization is a first-class ship axis,
often decisive over calibration (§5.4); and the predictive-variance regularizer is untried, not refuted
— now a model/loss lever (§5.5).

**The numbered holes, current status** (resolutions cross-ref §5; only #4 and #6 remain open — #0b
above is the highest-priority open hole):

1. *Will post-hoc scale calibration move PIT-KS without breaking g1/g5?* **Resolved** — yes,
   refined by the honest sweep: it does, but only under the right normalization and with a
   cell-specific val→test discount (WNBA AST 0.126→0.047, g1 −0.017, g5 0.049).
2. *Was the SkewNormal dispersion-cal exclusion deliberate?* **Resolved** — an a-priori skip,
   not a tested negative.
3. *CRPS-vs-PIT-KS objective gap on the count branch.* **Resolved** — the PIT-KS re-target
   flips the count cells; coverage is the wrong objective (over-narrows some lattices).
4. **Open — do the marginal NFL g1s improve when dispersion is calibrated?** The difference
   between NFL reaching 10 and reaching 15. A targeted before/after Brier-CI on the 5 g1+g4
   volume cells after a candidate §5.2 fit.
5. *Severe-coverage cells: width or shape?* **Resolved** — a decode artifact (C0), not a shape
   problem; plain under-dispersion + mild low-loc bias, and the severe cells (WNBA DREB, NFL
   receptions) ship under the centered normalization rather than via a new tail head.
6. **Open — the block-bootstrap / clustered-g1 backlog.** The test-set CSVs still lack a
   `game_date` column on combined-stat cells, blocking the player-clustered Gate-1 recheck and
   a closing-line-value gate. Resolving the training-data join unlocks both; the concrete method is
   **CPCV + a player/date embargo** ([71] López de Prado 2018), kept as a validation refinement, not a
   gate change. It is the one principled selection-style criterion worth building (Step 0.10).

---

## 9. Verification & supersession (unchanged guardrails)

**Always-on, every commit / PR:**

```bash
poetry run ruff check src/sportstradamus/
poetry run pytest tests/golden/          # incl. scorecard / gate tests
poetry run pytest -m integration         # fake-mode, no network
```

**Per-lever:** any change with an inference-side mirror (§5.2 scale object, §5.8
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
   citations [1]–[71], critical-files map, the per-change-type inference-path checklist
5. [`docs/operation_ship_90.md`](operation_ship_90.md) — next-rung stub; the levers it lists
   (T11 per-position, T3 tail head, CMPμ) are **on the Ship-75 table too** — pulled forward as
   needed, not reserved
6. [`docs/sportstradamus_roadmap_v3.md`](sportstradamus_roadmap_v3.md) — the swimlane master
   index. The parlay copula / dependence layer is now the `parlay-dependence` lane
   ([`docs/handoffs/parlay-dependence.md`](handoffs/parlay-dependence.md), gated on calibrated
   marginals); the conformal alt-line ladder, CLV-edge dashboard, TabPFN-as-platform /
   multi-task-NN backbone, and heaviest tail-head rebuilds live in its §8 deferred register

Done = each league shows ≥ 75% (`shipped ∈ {devel, main}`) on a fresh scorecard, with
every promotion having cleared its go/no-go and, for baselined cells, `supersede_verdict()`.
