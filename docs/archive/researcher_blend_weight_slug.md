# In-repo research brief — a general blend-weight fitting objective (§6.5-gated `blending` slug)

**Question:** which general blend-weight objective drives `w` toward the book on a no-edge cell
(so Gate-1's non-inferiority CI collapses) while keeping `w` high on a cell with real edge — with
no market-specific logic and no gate changes? Adjudicate (a) `brier_line`, (b) uncertainty-shrunk
`w`, (c) the forecast-combination / opinion-pool literature.

**Date:** 2026-08-25 · branch `devel` · read-only (a board sweep was live; nothing under `src/` or
`data/config/` was written).

---

## TL;DR

- **KILL (a) `brier_line`.** On the target cell it is close to a **no-op**: the Brier-at-line optimum
  and the CRPS optimum coincide at `λ* = 0.255`, and the incumbent fit already serves `λ_loc = 0.26`.
  A proper-score fit is *already* at the Brier optimum. Worse, it moves the fit onto the gate's own
  functional, and the expected one-parameter selection optimism (~0.5 SE ≈ **0.0010–0.0015**) is the
  same order as the entire shortfall this cell needs to close (0.0010). Non-inferiority with a
  tunable shrink-to-control knob is the classic assay-sensitivity failure (ICH E10).
- **KILL (b) as posed** (`w' = w·n/(n+k)`). `n` is a poor proxy for evidence: NFL receiving yards
  (n=2235) has weight-evidence `t = +3.03` while NBA MIN (n=2087) has `t = +10.69`. Same `n`,
  4× the evidence. The functional form is undetermined and `k` has no anchor.
- **BUILD (c), specific form: a one-standard-error parsimony rule on the existing CRPS path** —
  new slug `crps_1se`. Grid `w` over `[0.05, 0.9]`, take `ŵ* = argmin CRPS`, compute the
  **player-clustered paired bootstrap SE of the loss *difference*** `L_i(ŵ*) − L_i(w_min)`, then
  return the **smallest** `w` whose mean loss sits within 1 SE of the minimum. One global constant
  (κ = 1 SE). No gate change, no new served object, no market logic. This is Diebold–Pauly (1990)
  shrinkage of estimated combination weights toward a prior, with the book as the prior; motivated
  by the forecast-combination puzzle (Smith & Wallis 2009 DOI 10.1111/j.1468-0084.2008.00541.x;
  Claeskens et al. 2016 DOI 10.1016/j.ijforecast.2015.12.005).
- **Measured discrimination of the proposed rule** (location-only probe, 11 cells): it sends
  NFL passing yards → `λ 0.02` and NFL completions → `λ 0.03` (the two g1-marginal cells) while
  holding NBA MIN at 0.65, NBA fantasy-points-prizepicks at 0.57, WNBA PTS at 0.44, NBA PRA at 0.41.
  Exactly the asked-for behavior, driven only by evidence-per-unit-of-noise.
- **Two corrections to the dispatch framing, both load-bearing.** (1) The ship gate reads
  `g1_brier_diff_ci_hi` — the **iid** paired bootstrap. `g1_clustered_ci_hi` is reported-only
  (`scorecard.apply_thresholds` line 2063). (2) On the **production** test frame NFL passing yards
  **already passes g1** (iid ci_hi 0.0032, clustered 0.0033, BSS +0.0039). The 0.0060 failure is on
  the board's `crossfit_validation` frame. The point estimate is a tie on both frames; **the failure
  is CI width, not accuracy.**
- **Honest economics.** This buys **coverage, not edge.** `kelly_shrinkage = clip(BSS, 0, 1)`, so a
  book-leaned cell sizes to ~0.5% model weight. That safety valve is already built and must stay.

---

## Key findings

### 1. The g1 failure on this cohort is a precision problem, not an accuracy problem

Reproducing the gate on `data/test_sets/NFL_passing-yards.csv` (369 rows, 366 priced, 59 players):

| statistic | value |
|---|---|
| empirical over-rate at the line | 0.4986 |
| `p_book` mean / sd | 0.5003 / 0.0211 |
| `brier_book` | 0.25164 |
| served `P` mean / sd | 0.5132 / 0.0454 |
| g1 paired diff, mean | **−0.00098** |
| g1 iid ci_hi (the gate) | **+0.00321** |
| g1 clustered ci_hi | +0.00329 |
| `brier_skill_score` (fused) | **+0.00389** |
| `P_standalone` BSS / clustered ci_hi | **−0.0972 / +0.0466** |

The point estimate is essentially exactly zero on both frames (board: `g1_brier_skill = −0.0003`).
The clustered half-width is ≈0.0042 (production) and ≈0.0059 (board crossfit) against a 0.0050
margin. Certifying non-inferiority at the *same* effect size would need `(0.0059/0.0050)² = 1.39×`
the clusters — roughly 84 instead of ~60. The alternative is a deviation ~15% smaller. The repo
already encodes this reading: `confirm._G1_RETRY_NOISE_BAND = 0.002` and this cell's miss is 0.0010.

Also note `brier_book ≈ 0.25` is **not** evidence that the book is sharp. On a balanced two-sided
prop the book's over-probability at its own line is ≈0.5 by construction (measured sd 0.021), so
`brier_book → 0.25` for *any* book quality. The book's skill lives entirely in where it put the
line, which g1 conditions on and therefore cannot see. Gate 1 on this cell is literally "beat a coin
flip at a line that makes the outcome a coin flip." (This corrects, in the useful direction, the
`[[project_passing_book_degenerate]]` framing and the §6.9 "brier_book ≈ 0.250 ⇒ sharp book" line —
both readings of 0.25 are unfalsifiable from `brier_book` alone.)

### 2. `ci_hi` is proportional to the model's probability deviation — closed form

Write `Δ = p_model − p_book`. The per-event paired Brier difference is exactly

```
d_i = Δ_i² + 2·Δ_i·(p_book,i − y_i)
```

so under a uniform shrink `p_λ = p_book + λΔ`:

```
ci_hi(λ) = λ²·E[Δ²]  −  2λ·E[Δ(y − ½)]  +  1.96·λ·sd(Δ(p_book−y))/√m
```

with `ci_hi(0) = 0` exactly. The λ-linear coefficient is `[1.96·sd/√m − 2·E[Δ(y−½)]]` — positive
iff the model's directional reward per unit of boldness is below ~1 SE. So:

* **no-edge cell** → `ci_hi(λ)` is monotone increasing, minimized at the floor;
* **real-edge cell** → the linear term is negative and shrinking *hurts*.

Measured on the production frame (player-clustered bootstrap, seed 1729, 2000 draws):

| λ | g1 mean | g1 clustered ci_hi | BSS |
|---|---|---|---|
| 1.00 (as served) | −0.00098 | +0.00329 | +0.00389 |
| 0.80 | −0.00106 | +0.00237 | +0.00420 |
| 0.50 | −0.00092 | +0.00123 | +0.00364 |
| 0.30 | −0.00065 | +0.00064 | +0.00259 |
| 0.00 | 0 | 0 | 0 |

`ci_hi` is monotone and near-linear in λ; **BSS is flat** (+0.0039 → +0.0042 over λ∈[0.5, 1.0], an
0.00008 change in Brier against a clustered SE of ~0.0021). Signal-to-noise for *locating* the
Brier optimum is ≈0.04. This is the forecast-combination puzzle in its exact textbook shape.

### 3. Brier-at-line and CRPS pick the *same* weight — so (a) buys nothing the gate can't already see

Reconstructing the two legs from the dump (book leg = the legacy symmetric constant-CV normal that
`fused_loc` actually uses — `stat_calibration.json` has `book_shape: None` for this cell, and §6.5
records that production books stay symmetric; `ev_b = Line / (1 + cv·Φ⁻¹(Odds))`, `cv = 0.40157`),
then sweeping the location weight λ at the served scale:

```
CRPS  argmin λ = 0.255   gain vs book-only +0.3215   paired clustered SE 0.2798   t = +1.15
Brier argmin λ = 0.255   gain vs book-only +0.00428  paired clustered SE 0.0028   t = +1.52
```

The two objectives agree to the grid resolution, and the incumbent served blend is already there:
the recovered effective location weight is **λ_loc median 0.26** (q10 0.11, q90 0.41), computed as
`(Blended_EV − ev_b)/(EV − ev_b)`. So `brier_line` would return approximately the weight the
incumbent CRPS/NLL fit already returns. **Candidate (a) is a near-no-op on the cell it was proposed
to fix.**

Corroborating: across 261 matched board pairs that differ only in `blending_loss_fn`, the median
|Δ `g1_brier_diff_ci_hi`| is **0.0006**; on the NFL passing-yards pair it is **exactly 0.0000**
(rows 1729/1730 are byte-identical on g1–g5, `g4_pit_ks`, `dispersion_cal` and `skew_cal`). 25% of
NFL pairs are fully inert. **A swap between two proper scores is a low-amplitude knob.** The lever
that moves this cohort is not *which* proper score — it is *how much estimation error you are
willing to pay for*.

### 4. The fitted weight is a high-variance statistic exactly on the cells that fail

Cluster-bootstrapping the CRPS argmin (400 resamples, player clusters):

| cell | K | λ*(full) | bootstrap λ* median | 90% interval | sd |
|---|---|---|---|---|---|
| NFL passing yards | 59 | 0.26 | 0.26 | **[0.08, 0.48]** | **0.120** |
| NFL completions | 61 | 0.30 | 0.30 | [0.12, 0.46] | 0.113 |
| NBA PTS | 342 | 0.50 | 0.50 | [0.38, 0.62] | 0.077 |
| NBA fantasy points prizepicks | 354 | 0.82 | 0.84 | [0.76, 0.90] | 0.037 |
| NBA MIN | 358 | 0.92 | 0.92 | [0.86, 0.98] | 0.038 |

The fitted weight on the failing cell has a 90% interval spanning a **6× range** and 3× the sd of
the strong cells. This is precisely the condition under which Claeskens et al. (2016,
DOI 10.1016/j.ijforecast.2015.12.005) show that treating estimated weights as fixed is invalid — the
combination becomes biased and its variance exceeds the fixed-weight case — and under which
Smith & Wallis (2009, DOI 10.1111/j.1468-0084.2008.00541.x) attribute the puzzle to finite-sample
weight-estimation error. Diebold & Pauly (1990, IJF 6(4) 503–508,
DOI 10.1016/0169-2070(90)90028-A) is the canonical prescription: shrink the estimated weight toward
a prior, with the prior/LS weights as polar cases and the shrinkage intensity set by prior precision.

### 5. The 1-SE rule discriminates cleanly across cells with no market logic

Same probe, all cells with a SkewNormal dump and an authentic book. `t` is the paired clustered
t-statistic of the CRPS gain at `λ*` over the book-only blend; `λ_1SE` is the smallest λ within one
paired clustered SE of the minimum:

| cell | n | K | λ* | t (CRPS) | **λ_1SE (CRPS)** |
|---|---|---|---|---|---|
| **NFL passing yards** | 366 | 59 | 0.25 | +1.16 | **0.02** |
| **NFL completions** | 377 | 61 | 0.30 | +1.20 | **0.03** |
| NFL receiving yards | 2235 | 382 | 0.23 | +3.03 | 0.10 |
| NFL rushing yards | 1011 | 177 | 0.35 | +3.99 | 0.18 |
| NBA PTS | 2251 | 342 | 0.50 | +2.30 | 0.18 |
| WNBA RA | 2232 | 160 | 0.72 | +4.26 | 0.38 |
| NBA PRA | 2212 | 338 | 0.75 | +4.63 | 0.41 |
| WNBA PTS | 2239 | 158 | 0.76 | +5.56 | 0.44 |
| NBA fantasy points prizepicks | 2030 | 354 | 0.83 | +9.91 | 0.57 |
| NBA MIN | 2087 | 358 | 0.92 | +10.69 | 0.65 |

*(NFL attempts returned a degenerate `λ* = 0, gain = 0` — its dumped `EV` reconstructs to the book
leg; treat as a reconstruction artifact of a stale 2026-07 CSV, not a finding.)*

The rule sends exactly the two g1-marginal cells to the floor and leaves the strong cells at
0.38–0.65. **The projected crossfit ci_hi on NFL passing yards at λ ≈ 0.05 is
`0.0060 × (0.05/0.26) ≈ 0.0012`** under the §2 linear rule — a comfortable pass — while the
reconstructed Brier at λ = 0.05 is 0.24998 vs `brier_book` 0.25203, i.e. the cell keeps a small
**positive** skill (~+0.008) rather than becoming a pure book echo.

### 6. Frames: honest at the row level, but a real selection channel on the gate's functional

Read from `training/pipeline.py`:

* **Production (`meditate`, `holdout_blind=False`).** `_step_build_splits` takes the temporal 30%
  tail, then splits it by `hash(Player, Date) < _VALIDATION_HASH_THRESHOLD` into `*_validation` and
  `*_test`. Every calibrator in `_step_calibrate_and_serve` — **including `fit_blend_weight`** — is
  fit on `*_validation` and applied to `*_test`. The scorecard scores `*_test`. Rows are disjoint,
  so a `brier_line` fit is **not** scored in-sample. But the split is **row-level, not
  player-level**: validation and test share players and eras.
* **Board (`holdout_blind=True`).** `X_test` *is* `X_validation`, and
  `_step_crossfit_calibrate_and_serve` refits the whole chain on 4/5 **player-disjoint** folds
  (`_calibration_folds` hashes the player; `_CALIBRATION_FOLDS = 5`) and applies out-of-fold. The
  board number for any new slug is therefore honestly OOF. **The board is the right place to test.**

So: **not a mechanical leak.** But the residual optimism is not negligible. The SE of g1 on this
cell is ~0.0021 (production) / ~0.0030 (board); a smooth one-parameter minimization of that same
functional buys on the order of half an SE, i.e. **0.0010–0.0015** — 20–30% of the whole 0.005
margin and larger than the 0.0010 the cell is short. And the dominant optimism channel already
exists: the corner that reaches production is chosen by maximizing gate slack over 17 corners
(Cawley & Talbot 2010, JMLR 11:2079–2107 — selection over an evaluation statistic produces bias of
the same magnitude as real method differences). Adding a knob whose sole purpose is to move that
statistic compounds it.

This is the assay-sensitivity problem from non-inferiority trial design: a non-inferiority test is
uninterpretable if the "treatment" can be tuned toward the control (ICH E10 §1.5; D'Agostino,
Massaro & Sullivan 2003, *Stat Med* 22(2) 169–186, DOI 10.1002/sim.1425 — including the serial
"biocreep" failure mode, which is what "route the whole cohort through a book-lean slug" would be).
The fix is not to abandon the shrinkage — it is to **drive the shrinkage with a functional the gate
does not score.** §3 shows that costs nothing: CRPS and Brier-at-line pick the same weight.

### 7. §6.5's NO-GO does not bind this cohort — and the plan says so explicitly

From §6.5 and `[[pooling_half_blp_nogo]]`: the operator weight-challenge (probe-v2) jointly re-fit
weight and dispersion by **CRPS + PIT-KS hinge**, "DID explore low weights (`λ_A`→0.63, `w_mix`→0.40
on book-heavy NBA RA/PR/AST) and every one still loses OOS (0/9 cells clear the gate)". The gate
in "0/9 clear the gate" is **Gate 4 PIT-KS**, on a **g4-bound over-wide cohort at `w ≈ 0.90`**
(NBA RA/PR/AST, NFL fantasy-underdog/carries, WNBA DREB). The stated root cause — "at `w ≈ 0.90` the
served predictive is already the model's and near-calibrated, so the beta wrapper has no
decalibration to repair" — does not describe this cell: NFL passing yards is at `λ_loc = 0.26`,
passes g4 with 41% headroom (`pit_ks 0.0438` vs `max 0.0744`), and fails **only** g1. §6.5 itself
says "Reopen only for a genuinely over-wide cohort at `w<0.9` … re-probing per cohort — never
generalize the NO-GO." **This is a different cohort, a different gate, and a different mechanism.**

Two §6.5 pre-commitments *do* transfer and are carried into the recipe below: fit on a proper score
(CRPS, never PIT-KS — Gneiting et al. 2005 [74]), and keep a **collapse guard** so the new object
must strictly beat the incumbent or reduce to it.

---

## Recommendation — the exact fitting recipe

**Verdict: build one new `blending` slug, `crps_1se`. Do not build `brier_line`. Do not build a
`ci_hi`-minimizing objective under any name.**

```
slug:      "crps_1se"   (added to calibration.BLENDING_SLUGS; DEFAULT_BLENDING stays "nll")
frame:     splits["*_validation"], authentic quotes only
           (_split_quote_authenticity_mask — unchanged; never fit on synthetic/derived rows)
loss:      L_i(w) = the existing per-observation CRPS of the blended predictive
           (fit_model_weight_crps's objective verbatim: same fused_loc, same gate plumbing,
            same family branches). NOT the Brier at the line. NOT PIT-KS.
path:      w over a fixed grid, np.linspace(0.05, 0.9, 35)  (0.025 steps)
           — a 1-SE rule needs the whole path, so grid, not the scalar TNC in _minimize_weight
argmin:    ŵ* = grid[argmin_w mean_i L_i(w)]
SE:        player-clustered PAIRED bootstrap of the DIFFERENCE
             D_i = L_i(ŵ*) − L_i(w_min),  resample whole player clusters,
             SE = sd of the bootstrap means.
           Reuse scorecard._bootstrap_mean_ci_clustered's cluster-resampling idiom.
           Cluster key = players_validation, falling back to dates_validation
           (identical to _calibration_folds' fallback for team markets).
           2000 draws, fixed seed (the repo has a determinism gate).
rule:      ŵ = min { w in grid : mean_i L_i(w) <= mean_i L_i(ŵ*) + KAPPA * SE }
constant:  KAPPA = 1.0    # one global module-level constant, one reason comment. No per-cell tuning.
guard:     the rule is a RESTRICTION, never an expansion: if no w < ŵ* clears the band,
           return ŵ* exactly, byte-identical to the "crps" slug.
DPO:       fit_dpo_weight must learn the slug too — it branches on `blending == "crps"` and
           would otherwise silently fall back to nll on every DPO cell.
```

Nothing else changes. `w` still rides the pickle to `prediction/model_prob.py` as a scalar in
`[0.05, 0.9]`; there is no new served object, no inference-path change, no gate change, and no
market-specific branch.

**Why the paired-difference SE and not the marginal SE.** Chen & Yang (2021, *Stats* 4(4) 868–892,
DOI 10.3390/stats4040051) evaluate the 1-SE rule directly and find its central weakness is that
"the standard error of the CV curve itself is not the standard error of prediction error
differences," with SE estimation bias of "50–100% upwards or downwards in various situations."
Because the losses at different `w` are computed on the *same rows* and are highly correlated, the
paired difference SE is both much smaller and far better estimated than the marginal SE. Using the
paired clustered bootstrap answers their objection head-on. Their second finding — the 1-SE rule
usually *loses* on prediction accuracy while *winning* on parsimonious selection — is exactly the
trade being made here, and should be stated in the plan as the price, not hidden.

**Why the book, not equal weights, is the shrinkage prior.** Diebold & Pauly shrink toward the
simple average because their components are peers. Here the book is the *incumbent benchmark*, and
`w_min = 0.05` is already the repo's encoding of "maximum trust in the book." Shrinking to the
existing bound requires no new parameter and no new served semantics. Liu, Hao & Wang (2024, *OBES*
86(3) 714–741, DOI 10.1111/obes.12590) formalize the general case as a double shrinkage — toward
equal weights and toward zero — and find the two dominate in different regimes; our single
shrinkage toward the benchmark is the degenerate two-forecast case of the zero-weight arm.

**Diagnostic, not objective.** Keep the Brier-at-line optimum as a *reported* number next to the
fitted `w` if it is cheap. It is a useful cross-check (§3 shows the two agree), but it must never be
the thing minimized.

**Observability ask (cheap, high-value).** The board CSV carries no `model_weight` and no
`g1_brier_diff_ci_hi_standalone`. Without them nobody reading the board can tell whether a corner
passed g1 because the model is good or because `w` collapsed. Add both to the board/ledger column
set before this slug enters the sweep grid. Otherwise the slug is a silent BSS-launderer.

---

## Reality checks

**Projected effect size, and its regime.** On NFL passing yards the §2 linear rule projects board
`ci_hi 0.0060 → ~0.0012` at λ ≈ 0.05, a comfortable pass. **This is a projection through a
location-only proxy, not a measurement.** Three things it does not model:
1. The real `w` moves the *scale* and *skew* too. `fused_loc`'s SkewNormal branch is
   **precision-weighted**, so `λ_loc = w·prec_m / (w·prec_m + (1−w)·prec_b) ≠ w`; a model much
   sharper than the book keeps meaningful location weight even at `w = 0.05`.
2. The chain **downstream** of the blend refits: `dispersion_cal` (currently **1.284** on this
   cell — the served predictive relies on widening), `skew_cal` (−1.136), the PIT map, the
   temperature, and `prob_recal_platt`. Platt is fit by log-loss on validation, so it will not
   *expand* deviations the data does not support, but it will re-center them.
3. The floor is a **weight** floor, not a **deviation** floor. Empirically the three cells already
   sitting at `w = 0.05` land at clustered ci_hi of −0.0277 (MLB batter strikeouts), +0.0006
   (NFL attempts) and +0.0039 (NFL completions) — a 6× spread. `w = 0.05` does not mean `Δp = 0`.

**The cross-sectional correlation runs the "wrong" way — do not misread it.** Across the 46 cells in
`model_stats.parquet` with a g1 measurement, `corr(model_weight, g1_clustered_ci_hi) = −0.32`:
higher `w` associates with *lower* ci_hi. That is confounding, not refutation — the high-`w` cells
are the ones with real edge (or no book). The causal claim in §2 is within-cell, and the λ-sweep
measures it directly. But a plan reader looking only at the cross-section would conclude the
opposite, so state it.

**g4 risk (the main cross-cell hazard) — but favorable on this cell.** A scalar `w` couples
location-trust and shape-trust; lowering it hands the predictive shape to the book's symmetric
constant-CV normal. On a cell whose model shape is right, that trades a g1 pass for a g4 fail.
On NFL passing yards it happens to run the *other* way: the reconstructed book σ (mean 88.7) is
**wider** than the served σ (mean 78.8), and `dispersion_cal = 1.284` shows the served predictive is
already being widened — so a lower `w` widens the base and the calibrator has less work to do.
Do not generalize that to other cells; measure it.

**g2/g3/g6 risk.** Gates 2/3 score the fused `Blended_EV`; Gate 6 exists precisely to catch stable-star
regression toward a global mean. Dragging the served mean onto the book line *is* a mean-regression
mechanism, and it is real if the book shades stars. It is not automatic: NFL completions ships at
`w = 0.05` with g6 passing. But g6 is the gate most likely to fire on a control cell, and it must be
read in the pilot.

**Synthetic-book risk — already mitigated, do not weaken it.** `_split_quote_authenticity_mask`
excludes non-authentic quotes from the weight fit and forces `w = 1.0` on them at apply time, so the
slug cannot shrink toward a synthetic 0.5 row. The 8 bookless NFL cells and the 17–21% synthetic
dilution on priced cells (§6.9) are unchanged by this proposal. If that mask is ever relaxed, this
slug becomes a mechanism for shipping cells by agreeing with a coin flip.

**Economic reality check — coverage, not edge.** `kelly_shrinkage = clip(brier_skill_score, 0, 1)`.
A cell shipped at the weight floor with BSS ≈ 0.004–0.008 sizes to ~0.5% model weight; it will
almost never generate an EV edge because `model_ev ≈ book_ev` by construction. What it *does* buy is
real but narrow: a calibrated predictive for a market that currently has none (parlay correlation
legs, alt-line/ladder pricing, coverage of a hole in the board). The owner's ruling — "as long as it
passes the calibration gates it's fine to ride the book line" — is coherent with that and with the
§6.9 terminal fallback, but the plan should record the trade in those words. The Kelly safety valve
is the thing that makes the ruling safe; do not remove or floor it.

**Research project vs engineering project.** This is an **engineering project**, not a research bet.
The method (1-SE selection on a proper-score path, clustered paired SE) is standard and the
mechanism is measured in-house. Build cost: one new branch in `calibration.py` (~40 lines: grid
path + clustered paired bootstrap + the min-within-band rule), the slug added to `BLENDING_SLUGS`,
one line in `fit_dpo_weight`, plus the two board columns. No inference-path change. The *uncertain*
part is not the method — it is whether the projected λ→ci_hi transfer survives the downstream
calibration chain on real retrains. That is what the pilot measures.

**What would make this recommendation wrong.**
- If the crossfit g1 failure is a **direction** problem rather than a magnitude problem — the model
  systematically wrong on the validation era in a way the test era does not share — shrinking passes
  the gate mechanically while the underlying model stays bad. The tell is
  `g1_brier_diff_ci_hi_standalone` (**+0.0466** here, i.e. the standalone model is decisively worse
  than the book) plus the model's +6.1% mean bias (`EV` mean 233.15 vs recovered book 219.66). Both
  say "the model does not know this market," and the honest read of a `crps_1se` ship is *the book
  is doing the work.* That is a defensible ship, but it is not a model improvement.
- If the 1-SE band turns out to be wide enough on *large* cells to move them too. The measured
  λ_1SE of 0.38–0.65 on the strong cells says no, but that is the location-only proxy; the real
  path includes the scale term and could be flatter.
- If `κ = 1.0` proves too aggressive, the correct response is to **fix κ globally by evidence**
  (e.g. calibrate so that no cell with `t > 2.5` moves by more than 0.05) — never per-cell, never
  per-league. A per-cell κ reintroduces exactly the selection channel §6 warns about.

---

## Acceptance evidence — pilot design, pass and kill criteria

**Pilot cells (7).** The negative controls matter more than the positives.

| role | cells | expected behavior |
|---|---|---|
| target | NFL passing yards, NFL completions | `w` moves to/near the floor; `g1_pass` flips False→True; g2–g6 hold |
| mid | NBA PTS, NFL rushing yards | `w` moves modestly (Δλ ≈ 0.1–0.3); `ship` unchanged |
| control | NBA MIN, NBA fantasy points prizepicks, WNBA PTS | `w` moves < 0.05; every gate unchanged; BSS unchanged |

**Run it on the board first** (`holdout_blind`), because `_step_crossfit_calibrate_and_serve` scores
the new slug out-of-fold on player-disjoint folds — the only frame in the system that prices the
slug's own estimation error honestly. Then confirm the two targets under full HPO.

**Pass =** all of:
1. Both target cells reach `ship == True` on the board crossfit frame **and** survive a full-HPO
   confirm (board `ships` is a cross-fit pass, not a confirm verdict — `[[crossfit_board_ships_optimistic]]`).
2. On the three controls: no gate flips pass→fail, `|Δ min_gate_slack|` inside the deterministic-A/B
   noise band, and `|Δ w| < 0.05`.
3. On the targets, `brier_skill_score` stays **≥ 0** at the fitted `w`. A cell that gets *worse* than
   the book after shrinking toward the book is not a shrinkage problem — it is a broken reconstruction
   or a broken book leg.
4. The slug is byte-identical to `crps` on at least one cell where the loss path has curvature ≫ SE
   (a golden-test-able collapse guard).

**Kill =** any of:
1. A control cell loses `ship` — the rule is not selective enough; κ is wrong or the SE is misestimated.
2. A target buys g1 at the cost of g4 or g6 — the scalar `w` is too blunt for this cohort and the
   residual routes to §6.1 (mean corrector / Rung C) or §6.6, per §6.5's own routing.
3. The measured board `ci_hi` improvement on the targets is < half the §2 projection — the downstream
   calibration chain is restoring the deviation, and the lever is post-fusion, not in the blend.
4. `w` collapses to the floor on more than ~25% of the full sweep grid. That would mean the rule is
   firing on evidence, not noise, and the repo would be shipping a book echo across the board. In
   that case, tighten κ globally, do not scope the slug.

---

## Open questions / caveats to carry into the plan

1. **The two frames disagree by 0.0028 on this cell.** Production test says ci_hi 0.0032 (pass);
   board crossfit says 0.0060 (fail). Both are "held out," but the crossfit number includes the
   fold-to-fold variability of every calibrator and the production number does not. Which is the
   *right* number for a ship decision is a live question the plan has not settled. If the crossfit
   number is the honest one, several currently-shipped cells are optimistic; if the production number
   is, this cohort may not need the slug at all. **Resolve before building.**
2. **The gate reads the iid bound, not the clustered one.** `apply_thresholds` gates on
   `g1_brier_diff_ci_hi`; `g1_clustered_ci_hi` is reported-only. On a repeated-player panel the iid
   bootstrap is anti-conservative. Measured across the 46 cells with both, the gap is small
   (median +0.0001) and exactly one cell (WNBA RA: iid 0.0046, clustered 0.0053) passes on iid and
   would fail on clustered. Small, but it is a live inconsistency between the code and the way the
   gate is described in prose.
3. **`_MODEL_WEIGHT_MIN = 0.05` is a weight floor, not a deviation floor.** Whether the floor should
   be lowered (or made family-dependent via the precision ratio) for a cohort that wants a true
   book echo is untested. Do **not** bundle it with this slug.
4. **NFL attempts' 2026-07 test CSV reconstructs degenerately** (`EV` ≡ recovered book leg). Either
   the CSV is stale or that cell's dump has a decode issue. Worth a look independently of this lane.
5. **A `crps_1se` ship is not a model improvement and should not set a supersession baseline** on
   the strength of g1 alone. `[[corner_verdicts_are_not_code_scoped]]` and
   `[[board_slack_not_a_supersession_signal]]` both apply.
6. **Cross-league.** The rule is n- and cluster-count-sensitive by design, and NFL cells have ~60
   player clusters vs NBA's ~350. The rule will therefore be *systematically more aggressive on NFL*
   — which is correct under the combination-puzzle logic but means NFL becomes a book-lean league in
   aggregate. That is a program-level decision, not a per-cell one, and should be recorded as such.

---

## Bibliography

| # | Source | Identifier |
|---|---|---|
| B1 | Claeskens, G., Magnus, J. R., Vasnev, A. L., Wang, W. (2016). The forecast combination puzzle: a simple theoretical explanation. *International Journal of Forecasting* 32(3), 754–762. | DOI 10.1016/j.ijforecast.2015.12.005 |
| B2 | Smith, J., Wallis, K. F. (2009). A simple explanation of the forecast combination puzzle. *Oxford Bulletin of Economics and Statistics* 71(3), 331–355. | DOI 10.1111/j.1468-0084.2008.00541.x |
| B3 | Diebold, F. X., Pauly, P. (1990). The use of prior information in forecast combination. *International Journal of Forecasting* 6(4), 503–508. | DOI 10.1016/0169-2070(90)90028-A |
| B4 | Liu, L., Hao, X., Wang, Y. (2024). Solving the forecast combination puzzle using double shrinkages. *Oxford Bulletin of Economics and Statistics* 86(3), 714–741. | DOI 10.1111/obes.12590 |
| B5 | Chen, Y., Yang, Y. (2021). The one standard error rule for model selection: does it work? *Stats* 4(4), 868–892. | DOI 10.3390/stats4040051 |
| B6 | Breiman, L. (1996). Stacked regressions. *Machine Learning* 24(1), 49–64. | DOI 10.1007/BF00117832 |
| B7 | Yao, Y., Vehtari, A., Simpson, D., Gelman, A. (2018). Using stacking to average Bayesian predictive distributions (with discussion). *Bayesian Analysis* 13(3), 917–1007. | DOI 10.1214/17-BA1091; arXiv:1704.02030 |
| B8 | Cawley, G. C., Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *JMLR* 11, 2079–2107. | jmlr.org/papers/v11/cawley10a.html |
| B9 | Satopää, V. A., Baron, J., Foster, D. P., Mellers, B. A., Tetlock, P. E., Ungar, L. H. (2014). Combining multiple probability predictions using a simple logit model. *International Journal of Forecasting* 30(2), 344–356. | DOI 10.1016/j.ijforecast.2013.09.009 |
| B10 | Lichtendahl, K. C. Jr., Grushka-Cockayne, Y., Jose, V. R. R., Winkler, R. L. (2017/2018). Bayesian ensembles of binary-event forecasts: when is it appropriate to extremize or anti-extremize? | arXiv:1705.02391; DOI 10.48550/arXiv.1705.02391 |
| B11 | ICH E10 (2000). *Choice of Control Group and Related Issues in Clinical Trials* — assay sensitivity in active-control non-inferiority trials. | ICH Harmonised Tripartite Guideline E10 (database.ich.org/sites/default/files/E10_Guideline.pdf) |
| B12 | D'Agostino, R. B. Sr., Massaro, J. M., Sullivan, L. M. (2003). Non-inferiority trials: design concepts and issues — the encounters of academic consultants in statistics. *Statistics in Medicine* 22(2), 169–186. | DOI 10.1002/sim.1425 |
| B13 | Hastie, T., Tibshirani, R., Friedman, J. (2009). *The Elements of Statistical Learning*, 2nd ed., §7.10 — origin of the one-standard-error rule (after Breiman, Friedman, Olshen & Stone 1984, *CART*). | ISBN 978-0-387-84857-0 |
| B14 | Genest, C., Zidek, J. (1986). Combining probability distributions: a critique and annotated bibliography. *Statistical Science* 1(1), 114–135. | repo ref [54] |
| B15 | Ranjan, R., Gneiting, T. (2010). Combining probability forecasts — the beta-transformed linear pool. *JRSS-B* 72(1), 71–91. | repo ref [56] |
| B16 | Gneiting, T., Ranjan, R. (2013). Combining predictive distributions. *Electronic Journal of Statistics* 7, 1747–1782. | repo ref [57] |
| B17 | Gneiting, T., Raftery, A. E., Westveld, A. H., Goldman, T. (2005). Calibrated probabilistic forecasting using ensemble model output statistics and minimum CRPS estimation. *Monthly Weather Review* 133(5), 1098–1118. | DOI 10.1175/MWR2904.1; repo ref [74] |

**In-repo prior art engaged:** `docs/handoffs/model_improvement_track.md` §6.5 (CLOSED cohort, reopen
conditions, pre-committed design), §6.9 (book state audit; terminal book-lean fallback, lines
~1504–1505), §7.1 / `docs/ship_gate.md` (the six gates), §8.2 (research-first);
`[[pooling_half_blp_nogo]]` (weight-challenge probe-v2, 0/9 **g4** OOS wins);
`[[book_distribution_audit_nogo]]`, `[[book_skew_shape_borrow_refuted]]`,
`[[project_passing_book_degenerate]]`, `[[crossfit_board_ships_optimistic]]`,
`[[deterministic_ab_g4_oversell]]`.

**Artifacts read:** `src/sportstradamus/data/research/strategy_research_board.csv` (NFL passing
yards rows 1729–1745, swept 2026-08-22, `code_rev a4b93c49`; NFL rushing yards 2105–2153);
`src/sportstradamus/data/training/model_stats.parquet` (74 cells, 46 with g1);
`src/sportstradamus/data/test_sets/NFL_passing-yards.csv` and 10 sibling dumps;
`src/sportstradamus/data/config/stat_calibration.json`; `training/calibration.py`,
`training/pipeline.py`, `training/scorecard.py`, `training/model_strategy/confirm.py`,
`helpers/distributions.py:fused_loc`.
