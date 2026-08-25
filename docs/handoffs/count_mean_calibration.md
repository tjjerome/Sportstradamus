# Count Gate-4 — the mean is the lever

> Status: **OPEN, ready to run.** Research gate discharged by
> [researcher_count_gate4_zeroinflation.md](../archive/researcher_count_gate4_zeroinflation.md)
> (635 lines — read it, it carries the literature and the per-cell measurements this brief only
> summarizes). First experiment is **E1, NFL tds, ~35 min, no new code.** Predecessor lane:
> [count_dispersion_flip.md](count_dispersion_flip.md) (closed — the dispersion scalar is
> exhausted).

## 1. Mission, and what it is worth

Close Gate 4 on the count cohort. Be honest about the size of the prize: **two cells are
convertible — NFL tds and NBA PF.** Everything else in the cohort is either already passing,
Gate-1 walled, or fails the entry screen. If you find yourself building infrastructure, you have
lost the plot; the whole first experiment is two existing CLI flags.

## 2. Read first, in order

1. [count_dispersion_flip.md](count_dispersion_flip.md) — what is already dead. Do not re-test the
   `crps` → `pit_ks` count-dispersion objective; it is a measured null three ways.
2. [researcher_count_gate4_zeroinflation.md](../archive/researcher_count_gate4_zeroinflation.md)
   — the research brief. §§1–3 for the mechanism,
   §7 for the transfer evidence, §E1–E3 for the pre-registered design.
3. [docs/ship_gate.md](../ship_gate.md) — Gate 4 is a randomized-PIT KS, Gate 6's CITL leg is
   `Σpred/ΣResult` on star players and is one-sided (under-prediction only).
4. [model_improvement_track.md](model_improvement_track.md) §6.1 Rung A (the mean corrector and the
   operator's standing skepticism of it), §6.5 (blend structure, closed), §8.2 open questions.

## 3. The mechanism — it is the mean, not the zeros

Gate 4's supremum on every failing count cell sits at the bottom of the lattice: predicted
`P(Y=0)` exceeds empirical by +0.05 to +0.11, which is 80–99 % of the whole statistic, and
cohort-wide `|zero-mass gap|` correlates with `g4_pit_ks` at Spearman **+0.607**. That much is
solid and reproducible.

**The obvious reading of it is wrong.** Zero-inflation is not the cause. The served ZINB mean is
`(1−π)·μ`, so shrinking or dropping the gate silently rescales the mean by `1/(1−π)` — 1.02× on
NBA PF, 4.76× on NHL goals. A probe that moves the gate *at fixed served mean* buys 0.005–0.009
KS. A probe that moves only the mean takes **all four** failing ZINB cells under threshold, and
adding the gate on top of a free mean buys ≈ 0.000:

| cell | π̄ | CITL | thr | as-shipped | `c` only | `(c,g)` @ fixed mean | **`k` (mean) only** | fitted `k` |
|---|---|---|---|---|---|---|---|---|
| NFL interceptions | 0.135 | 0.738 | 0.0698 | 0.1199 | 0.1133 | 0.1044 | **0.0620** ✓ | 1.48 |
| NBA PF | 0.017 | 0.953 | 0.0500 | 0.0680 | 0.0634 | 0.0620 | **0.0437** ✓ | 1.07 |
| NFL tds | 0.414 | 0.737 | 0.0500 | 0.0553 | 0.0533 | 0.0462 | **0.0299** ✓ | 1.51 |
| NBA TOV | 0.077 | 0.933 | 0.0500 | 0.0606 | 0.0577 | 0.0528 | **0.0265** ✓ | 1.09 |
| NBA OREB *(control)* | 0.149 | 1.062 | 0.0500 | 0.0360 | **0.0321** | 0.0294 | 0.0410 | 0.98 |

The NFL cells serve a predictive mean **26 % below** the realized outcome mean. Gate 6's CITL leg
is measuring the same defect from the other side (NFL tds' NegBin arms KILL at `g6_citl_ci_hi`
0.783 / 0.805; interceptions sits at 0.932 against a 0.97 line). **One defect, two gate
failures.**

**Why Gate 1 never saw it.** The served over-probability passes through a temperature fit at the
quoted line, which is a one-parameter logit scaler. It absorbs a mean bias *at that single
threshold* and nowhere else — on NFL tds, empirical over-rate 0.197 vs served 0.177, a 2-point
miss, while the predictive mean is 26 % low. This is the textbook split between threshold and
probabilistic calibration (Gneiting, Balabdaoui & Raftery 2007
doi:10.1111/j.1467-9868.2007.00587.x). The architecture *guarantees* this failure mode exists and
Gate 4 is doing exactly the job it was built for.

Practical consequence: lifting the mean should improve g4 and g6 while being roughly neutral on
g1/g5, because the temperature refit just moves closer to 1. Favourable risk profile — but it is a
prediction, and E1 is designed to falsify it.

## 4. Why the failing cohort is entirely ZINB

Every Gate-4 failure among 49 count cells is a ZINB: **5 of 16 ZINB fail, 0 of 33 NegBin+DPO**
(Fisher p = 0.0023). It is not sample size — median dump 2250 vs 2237 — and median `g4_pit_ks` is
0.0402 ZINB vs 0.0238 DPO, 0.0218 NegBin.

That association is real but it is a *symptom*, and the chain matters because it tells you where
to intervene:

> The mean-stage corrector (`posthoc: roe_mean` / `isotonic_mean`) is fit on `decoded.ev`, which
> for a ZINB is the **base** NegBin mean `r·p/(1−p)` with the gate excluded — while the target
> `val_result` is the zero-**inclusive** outcome. The fitted map absorbs `(1−π)`, and `(1−π)` is
> then applied a second time downstream. Simulated on real dumps with the production contract, it
> drives NHL goals' served CITL from 0.997 to **0.260** and NBA BLK's from 0.972 to **0.441**.
> So on gated families the corrector wrecks the cell, the board correctly rejects that corner, and
> the cell's mean deficit goes uncorrected forever.

Adoption rates corroborate: mean-stage posthoc is selected on **6/25 DPO cells (24 %, ungated)**
and **1/17 ZINB cells (5.9 %, gated)** — and that one is `withheld`, so **no live cell is
harmed.** This is a *suppressed lever*, not an active bug, and it is suppressed hardest on exactly
the cells that need it.

## 5. Three pathologies, not one

There is no single "count Gate-4 lever". Measured conditional dispersion
`φ̂ = (Var(y) − Var(m))/mean(y)` with the Cameron–Trivedi (1990) t-statistic:

| cell | φ̂ᵢₛₒ | CT t | Pearson raw | reading | route |
|---|---|---|---|---|---|
| NBA PF | 0.780 | **−8.70** | 1.310 | **sub-Poisson** | DPO (NegBin/ZINB structurally cannot reach it) |
| NBA TOV | 0.902 | −1.76 | 1.648 | mildly sub-Poisson | — (Gate-1 walled) |
| NFL tds | 1.029 | −0.32 | 1.319 | equidispersed | mean lever, NegBin is adequate |
| NFL interceptions | 1.032 | +0.30 | — | equidispersed | mean lever |
| MLB runs allowed | 1.568 | **+9.44** | 1.815 | over-dispersed | — (fails the S2 veto) |

**Use `φ̂`, never raw Pearson `mean((y−m)²/m)` alone.** `E[(y−m)²] = Var(Y|x) + bias²`, so any mean
error inflates it and at low `m` the `1/m` weighting is dominated by near-zero rows. NBA PF reads
1.31 raw (over-dispersed) and 0.78 corrected (sub-Poisson) — **same data, opposite verdict.** `φ̂`
is biased *upward* by GBT leaf-averaging, so `φ̂ < 1` is conservative evidence: mean
misspecification can mask genuine sub-Poisson variance but cannot manufacture it.

**Free mis-family detector already in the pipeline:** a count cell whose `dispersion_cal` fit pins
at its `c` bound is mis-familied. `c → 10` on a NegBin/ZINB means the fit wants a variance below
the family's floor (`Var = μ + μ²/r ≥ μ` for every `r`); `c → 0.1` means it wants more spread than
the shape head can give. The bound-pinning reported in the predecessor lane is this signature.

## 6. Entry screen — run it before spending any confirm

All four legs are one-liners over `(y, m)` plus the served CDF. Compute on the validation split,
or offline on a temporal split of the cell's dump.

| leg | statistic | threshold | routes to |
|---|---|---|---|
| **S1 mean** | `CITL = Σm/Σy`, player-clustered bootstrap CI | \|CITL−1\| ≥ 0.04 **and** CI excludes 1 | mean-stage posthoc |
| **S2 stationarity** | split the fit window in half by date; two half-window CITLs | same side of 1 **and** \|ΔCITL\| < 0.06 | **VETO if it fails** |
| **S3 dispersion** | `φ̂` + Cameron–Trivedi t | φ̂ < 0.90 ∧ t < −3 → DPO; φ̂ > 1.20 ∧ t > +3 → NegBin/hurdle | family |
| **S4 zero modification** | frozen-parameter `z₀` on the **ZI-free** counterpart | \|z₀\|<2 drop gate; >+3 hurdle; <−3 DPO mandatory | family |

**Order matters: S1/S2 before S4.** A mean bias fires the zero test, so running S4 on an
uncalibrated mean mis-routes exactly the NFL tds / interceptions class. `z₀` is the frozen-parameter
zero-modification test (Wilson & Einbeck 2019, doi:10.1177/1471082X18762277) —
`z₀ = (O₀ − Σp̂ᵢ(0)) / sqrt(Σp̂ᵢ(0)(1−p̂ᵢ(0)))`, two-sided, no boundary problem, detects deflation.
Use the player-clustered bootstrap; raw Poisson-binomial variance is optimistic.

**Do not use a Vuong test to route zero-inflation.** A non-ZI model is neither strictly nor
partially non-nested in its ZI counterpart, so Vuong's criteria fail outright, and it cannot
identify zero *deflation* — the direction most of this cohort actually sits (Wilson 2015,
doi:10.1016/j.econlet.2014.12.029). The Schwarz-corrected Vuong in
`data/zinb_routing/{LEAGUE}_diagnostics.parquet` stays descriptive-only and must never route a
cell.

**S2 is the leg that keeps this from becoming a default.** On an honest temporal split (fit
earliest 60 %, score latest 40 %) the same intervention *harms* three cells out of a Gate-4 pass:
NBA OREB 0.0476 → 0.0663, WNBA FTM 0.0419 → 0.0568, MLB runs allowed 0.0303 → 0.0659. All three
flip CITL sign between eras or start with nothing to fix. **This is a per-cell option or it is
nothing.**

## 7. Family routing

| condition | family | machinery |
|---|---|---|
| φ̂ < 0.90 (sub-Poisson) | **DPO** | exists — swept family, `_DP_PHI_CEILING = 25`, 14 ships |
| φ̂ ∈ [0.90, 1.20], z₀ ∈ [−2, +2] | **NegBin** | exists — swept family |
| z₀ > +3 (genuine excess zeros) | **ZINB `zinb_mode: hurdle`** | exists — swept axis |
| — | **joint ZINB** | **retire as a default** — 0/132 board corners, gate head unidentified under NLL, and a ZI weight can only *add* zeros while most of this cohort needs the other direction |
| — | **CMP** | **do not build.** Genuinely the better under-dispersed family, but the measured need is φ̂ ≈ 0.78 ⇒ DP φ ≈ 1.3 against a ceiling of 25 — nowhere near where the approximation matters. Costs a torch class, a normalizing-series expansion, and the full §7.3 nine-site serve wiring |
| — | **ZTNB-hurdle** | **do not re-propose.** Analytically killed in B1.1 (`q < NB(0)` on ~65 % of FG3M rows) — and that failure *is* the zero deflation measured here, seen from the other side |

**Dispersion floor.** The most under-dispersed law on ℤ≥0 with mean μ is the two-point law on
⌊μ⌋/⌈μ⌉, giving index `1 − μ` for μ < 1. At NFL tds' μ ≈ 0.24 the floor is 0.73, so a family swap
has at most ~27 % of the variance to play with — another reason the TD markets are mean stories.
At NBA PF's μ ≈ 2.1 the floor is 0.083 and the family axis has real room.

## 8. The mean lever — two paths, cheapest first

**Zero-code path (use this for E1/E2/E3): route the cell to NegBin or DPO first.** On a gate-free
family `decoded.ev` *is* the predictive mean, so `roe_mean` / `isotonic_mean` is correctly targeted
today. Both `dist` and `posthoc` are already swept axes — this is a two-axis corner, not a code
change.

**Six-line path (only if a genuinely gated cell needs it): fix the MEAN_STAGE contract.** Fit and
apply the corrector on the zero-inclusive mean `(1−π)·ev`, then divide `(1−π)` back out before it
reaches `_stage_family_shape_columns`. Localised to `pipeline._train_market_core` plus a golden.
Gate-aware refitting turns the same corrector into the fix (NFL tds CITL 0.762 → 1.056). This
changes a serving distribution and is research-gated — **the brief discharges that gate** — but it
is not on the critical path and must not be built ahead of E1's verdict.

Shape: `roe_mean` (affine) is primary at NFL count means per the §6.1 Rung A house rule; NBA PF's
quintile CITL is flat (0.90–0.99) so affine is well specified there. NFL tds is range-dependent and
non-monotone (worst at Q4), so sweep `isotonic_mean` as the second corner.

## 9. Experiments (pre-registered)

The strategy flags below are **`hidden=True` dev options — they do not appear in
`meditate --help`.** They are real; verify against `training/cli.py` if you doubt it.
`--artifact-output` requires `--frozen-matrix-dir`, `--dependency-namespace` *and*
`--bypass-withholding` together, and it short-circuits the production-artifact lock so the run
writes no `stat_calibration.json`.

### E1 — NFL tds. Run this first. One arm, ~35 min.

```bash
SP=/tmp/scratch/exp-e1
SPORTSTRADAMUS_ARCHIVE_READ_ONLY=1 poetry run python -u -m sportstradamus meditate \
  --league NFL --market tds --force \
  --frozen-matrix-dir research/logs/confirm/frozen_matrix/NFL_tds \
  --artifact-output "$SP/artifacts" --dependency-namespace exp-e1-negbin-roe \
  --bypass-withholding \
  --dist NegBin --posthoc roe_mean --target-normalization none \
  --dist-training-loss nll --blending-loss-fn nll --count-dispersion-objective crps

poetry run python -m sportstradamus ship scorecard --league NFL --market tds \
  --test-sets-dir "$SP/artifacts/test_sets" --scorecard-out "$SP/scorecard.csv" \
  --no-log --no-scatter
```

Baseline is the cell's recorded ZINB incumbent row — g4 0.0544, g1 `ci_hi` −0.0208, BSS 0.159, on
matrix `7918c1b8…`. No second run needed.

*Why first:* only cell that is pass-all-but-g4 on the **current** matrix hash with a real Gate-1
win; smallest gap in the cohort (0.0044); offline temporal-transfer estimate 0.0265, a 5× margin;
no new code; and NegBin is adequate because the cell measures equidispersed.

*Record:* all six gates, `g4_pit_ks`, `g4_tail_pit_ks`, `g6_citl_ci_hi`, `g1_brier_diff_ci_hi`,
`roc_auc`, **`model_weight`**, and CITL **both** ways — `Σ EV/ΣResult` (model-only) and
`Σ Blended_EV·(1−π)/ΣResult` (served). The two CITLs are what make a failure attributable.

| outcome | reading | next |
|---|---|---|
| all six pass | **SHIP.** `dist: NegBin`, `posthoc: roe_mean`, leave `shipped` to the human | run E2 |
| g4 < 0.05, served CITL ∈ [0.97, 1.03], another gate fails | mechanism confirmed, cell not convertible | run E2 |
| **served CITL ∈ [0.97, 1.03] and g4 ≥ 0.05** | **KILL the direction.** The zero-mass gap is not mean-driven and the offline probe was an artifact | stop; no E2/E3 |
| served CITL < 0.95 while model-only CITL ≈ 1.0 | **blend dilution**, not mechanism failure | lever is §6.5 blend structure (closed, research-gated); do not spend E2/E3 |
| g1 `ci_hi` ≥ 0.005 or g6 fires | the mean lift is buying calibration with edge | abandon the mean lever on this cohort; route to the family axis |

### E2 — NBA PF. Only if E1's mechanism confirms. Two arms, ~70 min.

Arm A `--dist DPO --posthoc roe_mean`; Arm B `--dist DPO --posthoc none`. Frozen matrix
`a24c4072…`.

*Why PF:* largest Gate-1 cushion in the cohort (`ci_hi` −0.108…−0.188, BSS 0.29–0.63), 3/3
pass-all-but-g4, largest g4 gap (0.027), flat correction shape, and **the only cell with a measured
structural family mismatch** (φ̂ 0.78 sub-Poisson on a NegBin body).

*Reads:* A ships, B doesn't ⇒ mean-driven. B ships ⇒ family-driven. Neither ⇒ PF's residual is
neither mean nor family; stop escalating it under §8.1 matrix exhaustion. Note PF's mean-only
offline margin (0.0393 vs 0.050) does **not** clearly survive the observed val→test discount of
+0.008–0.010 KS — which is precisely why the family arm is carried.

### E3 — NFL interceptions. Only if E1 ships. One arm, ~35 min.

`--dist NegBin --posthoc roe_mean`, frozen matrix `74dbd8a0…`. Must clear **g4 and g6 together.**
Two preconditions: (i) settle the Gate-1 premise with one `ship scorecard` read; (ii) the cell's
mean has a **negative** calibration slope (−0.166), so a mean corrector partly flattens it toward
the marginal — read `roc_auc` and `g1_brier_diff_ci_hi`, not just g4. n = 378, so treat any pass as
provisional pending a second matrix.

### Do not run, and controls

**NBA TOV** — 2/7 g1 pass, and its Gate 4 is *already solved* by the family axis (DPO 0.0204,
NegBin 0.0239). A Gate-4 fix ships nothing. **NHL goalsAgainst** — 1/10 g1 pass. **MLB runs
allowed** — fails the S2 stationarity leg; the temporal probe harms it 0.0303 → 0.0659.

Controls **NBA OREB** and **NBA BLK** must retain `ship`. Neither passes S1∧S2, so neither should
ever be enrolled.

## 10. Traps that will bite you

- **The cross-fit board is expired.** 14 of 15 board cells with a frozen manifest sit on a stale
  matrix — interceptions board `2a71b000` vs current `74dbd8a0`, PF `469cc6fe` vs `a24c4072`, TOV
  `f9b184f6` vs `6dd89d4c`. **NFL tds has zero board rows at all.** Do not spend a confirm on a
  board nominee before re-sweeping. Board pass costs: interceptions ≤49m, PF ≤37m, tds ≤46m, TOV
  ≤1h10m (`ship sweep --dist-class count --dry-run`).
- **A ledger `ship=True` is evidence only against its own `strategy_matrix_hash`.** See
  [count_dispersion_flip.md](count_dispersion_flip.md) — NFL tds' shelf nominee passed 6/6 on
  `285d0daa` and KILLs on the current matrix.
- **The board's "untried ZINB-hurdle winner" on interceptions is a mirage.** It shows a shipping
  hurdle corner at slack +0.105; 9 of the 10 already-failed confirms are `zinb_mode: hurdle`. What
  is genuinely untried at full HPO there is DPO, NegBin and SkewNormal — all 10 confirms are ZINB.
- **Dumps and ledger rows disagree** on `g4_pit_ks` for NBA PF (0.0680 vs 0.0775) and interceptions
  (0.1199 vs 0.1849) because they come from different runs. **Use dumps for mechanism, the ledger
  for pass/fail.**
- **Blend dilution is the most likely reason E1 under-performs.** `roe_mean` corrects the model
  mean *before* fusion, so the realised move is scaled by `model_weight`. NFL tds' model-only gated
  CITL is 0.903 against a fused 0.737 — the book pulls it down by 0.17. That is why E1 records both
  CITLs.
- **Do not read magnitudes from a small-HP A/B** — it oversells g4. E1–E3 are full-HPO confirms.
- **Do not re-propose Rung C** (isotonic-PIT / IDR) for count cells. Built, and a confirmed dead
  end on the low-mean lattice; the monotone map degrades the lattice and its g4↔g5 tension is real.
- **Do not build a post-hoc recalibrator for the mixture weight.** All three high-π cells pass
  Gate 4 today; the only high-π failure is a mean story. A free logit shift on a mixture weight is
  an unidentified knob with no calibration target of its own — the canonical thing that Goodharts
  under search pressure, which this repo has already been bitten by. If a high-π cell ever does
  fail g4, the defensible procedure is the **derived-π hurdle** the repo already has (calibrated
  binary `q̂`, then `π = clip((q̂ − NB(0))/(1 − NB(0)), 0, 1)`) — and remember the derived-π gate is
  `π_zi`, not `q`, and that hurdle cells bypass Optuna so a `calibrated` HP-selection pin is inert.

## 11. Residue — real findings, outside this lane's ask

- **A low-mean count cell that over-predicts currently has no gate.** NFL receiving tds is shipped
  `devel` with served CITL **1.64** — the served mean is 64 % above the outcome. Gates 2/3 miss it
  because σ is large at μ = 0.16, Gate 6's CITL leg is one-sided (under-prediction only), and its
  over-leg is guarded by `mean(Result) ≥ 1`. Worth an owner look.
- **NHL sogBS predicts `P(Y=0) = 0.0224` against an empirical zero rate of exactly 0.** A hard
  positive floor — a support problem (shifted / truncated), not a dispersion one. No φ knob fixes
  it.
- **The NFL touchdown markets are systematically sub-Poisson on families that cannot represent
  it.** Passing tds (φ̂ 0.77, NegBin, devel, g4 0.0447 — one retrain from failing), rushing tds
  (0.78, ZINB, devel), receiving tds (0.80, ZINB hurdle, devel). Candidates for a DPO supersession
  lane, one cell per confirm.
- **25 of 46 count cells read φ̂ᵢₛₒ < 1.** The strongest are NHL sogBS 0.65 (t −22.4) and MLB batter
  strikeouts 0.68 (t −15.1), both already DPO. The screen in §6 is the way to work that list.
