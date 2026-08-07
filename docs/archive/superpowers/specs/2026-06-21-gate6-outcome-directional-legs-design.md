# Gate 6 — outcome-directional legs + anchor hysteresis (design)

Design record, pending implementation plan; targets the `model-research` branch. Evidence base:
`/tmp/researcher_gate6_scope_and_drift.md` (which builds on `/tmp/researcher_overshrinkage_gate.md`,
the brief that first designed Gate 6). Findings are cited below by their number in those briefs.

## 1. Context and motivation

Gate 6 ("anti-shrinkage") was just widened from the `ratio_meanyr` SkewNormal cohort to **all
cells** (every normalization and distribution family), scoped only by the
`corr(Mean10, Result) ≥ 0.55` anchor. Its single statistic is the recent-form ratio
`r_star = Σ Blended_EV / Σ Mean10` on stable top-MeanYr stars, failing when the clustered
97.5% upper bound sits below the causal floor `star_ref − 0.03`.

The owner raised two gaps:

1. **The binary anchor is a knife-edge.** A cell at corr 0.54 is fully exempt, 0.56 fully gated;
   a retrain that nudges corr across 0.55 flips the ship verdict.
2. **The anchor fully excuses systematic drift on low-corr cells.** A consistent one-sided bias
   (under- or over-prediction) on a bursty/low-corr stat is never caught, because the recent-form
   leg abstains there.

The research evaluated both. Its verdicts (which this design adopts):

- **Graded corr-tolerance is rejected** (research Q1, Findings 1.1–1.3). The over-shrinkers cluster
  *just above* the anchor (NBA fantasy-points corr 0.56, WNBA FGA 0.57, WNBA PR 0.59), so
  "looser as corr falls" exempts the very targets; the mirror form softens the live case out; and
  any form firing below 0.55 false-fires on legitimate regression (NBA DREB/FGM, whose *outcome*
  confirms the regression). The anchor is a proxy-validity switch, not a tolerance knob.
- **The real boundary problem (retrain instability) is cured by hysteresis** (Finding 1.4): four
  cells' player-clustered corr CIs straddle 0.55 (NBA REB, NBA fantasy-points, WNBA FGA,
  NFL receiving-yards). A two-threshold deadband stabilizes the verdict without re-importing
  regression dilution.
- **Drift on low-corr cells is caught by calibration-in-the-large (CITL) against the realized
  outcome, not the recent form** (Q2, Findings 2.1–2.3). The sign test was tested and **rejected**
  (it reads mean-vs-median skew, not bias — false-flags shipped NBA PTS/PRA/REB, Finding 2.2).
  CITL is variance-robust and catches the failure Gate 2 launders: NBA fantasy-points scores
  Gate-2 `z = 0.073` (passes) but `CITL_out = 0.951` (fails).
- **The over-side (bench inflation) is real only for count/ZINB and needs a guard** (Q3,
  Finding 3.3): the zero-inflation floor over-predicts low-volume players' realized counts, which
  `ratio_meanyr` could not. Centered cells do **not** inflate the bench (Finding 3.2).

### Decisions taken (owner)

| Fork | Decision |
|---|---|
| Graded tolerance vs binary anchor | **Keep the binary anchor** (kill graded tolerance). |
| Boundary stability | **Add the hysteresis deadband** (stateful). |
| Directional-drift instrument | **CITL vs outcome** (sign test rejected). |
| Over-side leg | **Build now, guarded** (count/ZINB, segment-mean `Result ≥ 1`). |
| Recent-form leg | **Keep it** (it uniquely catches `ratio_meanyr` holdout-corruption, e.g. WNBA FGA, that CITL is structurally blind to) — with hysteresis. |

## 2. The gate: three OR-ed legs

Gate 6 fails iff **any** leg fires. All three operate on the test-set CSV columns
`MeanYr`, `Mean10`, `Result`, `Blended_EV`, `Player` (the over-leg additionally needs the
distribution-family columns `_infer_dist_from_columns` reads).

| leg | segment | statistic | fails when | scope |
|---|---|---|---|---|
| **recent-form** (existing) | stable top-MeanYr quartile | `Σ Blended_EV / Σ Mean10`, clustered **97.5% UB** | `UB < star_ref − _GATE6_MARGIN` | anchored cells only (corr gate + hysteresis) |
| **CITL-under** (new) | same stable top-MeanYr quartile | `Σ Blended_EV / Σ Result`, clustered **97.5% UB** | `UB < 1.0 − _GATE6_MARGIN` | **all cells**, including corr < 0.55 |
| **over** (new, guarded) | stable bottom-MeanYr quartile | `Σ Blended_EV / Σ Result`, clustered **2.5% LB** | `LB > 1.0 + _GATE6_MARGIN` | count/ZINB only, and only where bench-segment `mean(Result) ≥ _GATE6_OVER_MIN_MEAN` |

Common to all three:

- **Stable mask:** `|Mean10/MeanYr − 1| ≤ _GATE6_STABLE_BAND` (0.12), on rows with `MeanYr > 0` and
  `Mean10 > 0` — unchanged from the existing leg.
- **Segment power floor:** each leg requires `≥ _GATE6_MIN_STAR_ROWS` (30) stable rows in its
  segment; below that the leg **auto-passes** (insufficient power). This is what makes the gate
  fail-safe on thin NFL cells — their clustered CI spans the floor anyway (Finding 2.4).
- **Clustering:** all three CIs use `_bootstrap_ratio_ci_clustered` (player-clustered resampling,
  `n_boot = _GATE1_N_BOOT`, seed `_GATE1_SEED = 1729`) — the existing recent-form machinery, reused.

**Why keep the recent-form leg alongside CITL.** They diverge on exactly one class — `ratio_meanyr`
cells whose holdout outcome is itself artifact-suppressed (WNBA FGA: `r_form = 0.825` fires, but
`CITL_out = 1.000` clears, because the model matches the corrupted holdout). CITL scores against that
corrupted outcome, so it is blind to that class by construction; the recent-form leg is the only
detector for it (research Finding 2.1, prior brief Findings 1/6). For every other family the two
legs agree (centered: clean pred-vs-Result, Finding 3.1) or the recent-form leg is anchor-exempt
(counts). So the recent-form leg is narrow and on a deprecation path (it becomes redundant once the
MeanYr target-construction artifact is fixed at the source — §8), but it is load-bearing today: it is
what keeps WNBA FGA withheld across retrains.

**Net effect:** Gate 6 becomes strictly harder to pass than any single leg. This is intended for a
strict ship gate; it is recorded in `docs/ship_gate.md` (research reality-check 2).

## 3. The anchor and the hysteresis deadband (the stateful part)

The corr anchor gates **only the recent-form leg** — it is the only leg that uses `Mean10`, the
proxy whose validity corr measures. The CITL legs score against the realized outcome, which is a
valid yardstick regardless of corr; that is precisely why they reach into the low-corr region the
recent-form leg cannot (Finding 2.1).

### Deadband semantics

Let `corr = corr(Mean10, Result)` on the in-cell rows and `prior_fired` = whether the recent-form
leg flagged this cell on the previous scored run.

```
recent_form_active =  (corr >= _GATE6_FIRE_ON)                      # 0.58: a fresh cell starts judging here
                   or (corr >= _GATE6_KEEP_ON and prior_fired)      # 0.52: a flagged cell keeps being judged
otherwise          ->  recent-form leg auto-passes (exempt)
```

A flagged cell cannot escape to shippable on a small corr wobble; a clean cell needs solid corr
before Gate 6 starts judging it. The CITL legs ignore this entirely (no anchor).

### State threading — keep `scorecard.py` a pure function

`compute_gates` / `gate_row` gain an optional parameter `prior_g6_fired: bool | None = None`:

- **`report()`** (the production writer) reads the prior verdict from the existing
  `model_stats.parquet` and threads it in. It is **derivable from columns already persisted** —
  `prior_fired = (prior g6_star_ci_hi is not None) and (prior g6_star_ci_hi < prior g6_star_ref − _GATE6_MARGIN)`.
  No new persisted column is strictly required; if a dedicated `g6_recent_form_fired` boolean is
  cleaner to read back, add it to the `gate_row` dict (it is cheap and unambiguous).
- **The A/B `scorecard` CLI** and any first-ever run pass `prior_g6_fired = None`. With no history a
  cell uses the `fire-on` threshold (0.58) — the conservative initial comparator state.
- `scorecard.py` never reads a file itself; the prior state is an input, preserving the gate's
  testability and the repo's "gates are pure functions of one test CSV" property.

### Cold-start safety

On the first run after this lands, `prior_fired` is seeded from the **existing** `model_stats.parquet`
g6 columns, so a currently-flagged cell in the 0.52–0.58 band (WNBA FGA, corr 0.57, current
`g6_star_ci_hi = 0.890 < ref`) seeds `prior_fired = True` and stays judged. It does **not** escape on
the transition. A genuinely new over-shrinker that appears in the 0.52–0.58 band with no fired history
is exempt on the recent-form leg until it crosses 0.58 — but the **CITL-under leg backstops it** for
any cell whose outcome confirms the shrinkage, so the only uncovered case is a brand-new
holdout-corruption artifact cell, which does not occur outside `ratio_meanyr` (already-known, already
seeded).

## 4. Code structure

Consolidate "clustered ratio CI on a stable segment" into one helper, called three times — the
numerator is always `Blended_EV`; the `(denominator, segment)` pair varies:

```
_gate6_segment_ratio(df, *, segment_mask, denom)  ->  (point, lo, hi)   # wraps _bootstrap_ratio_ci_clustered
```

- recent-form: `segment = stable star quartile`, `denom = Mean10[seg]`, take `hi`.
- CITL-under:  `segment = stable star quartile`, `denom = Result[seg]`, take `hi`.
- over:        `segment = stable bench quartile`, `denom = Result[seg]`, take `lo`; count/ZINB + guard.

This shares the stable-mask + MeanYr-quantile setup rather than three copies of it. `_gate6_star_ratio`
keeps ownership of the stable mask and the star/bench segment construction.

`_gate6_passes` folds in the two new clauses (logical OR of the three one-sided tests).
`gate_row` gains `g6_citl_*` and `g6_over_*` keys mirroring the existing `g6_star_*` block.
`min_gate_slack` takes the **min** over the *active* legs' normalized slacks; an auto-passed leg
contributes `+inf` (unchanged convention). The over-leg uses `_infer_dist_from_columns` (the
dist-inference removed from the recent-form guard during the widening, now used legitimately because
the over-leg genuinely is family-specific).

## 5. Constants

| constant | value | role | status |
|---|---|---|---|
| `_GATE6_MARGIN` | 0.03 | tie band for all three legs (under: `1 − margin`; over: `1 + margin`) | existing, reused |
| `_GATE6_STABLE_BAND` | 0.12 | stable-row filter | existing |
| `_GATE6_MIN_STAR_ROWS` | 30 | per-segment power floor | existing |
| `_GATE6_STAR_REF_BASKETBALL` / `_NFL` | 0.95 / 0.94 | recent-form causal floor | existing |
| `_GATE6_MIN_RECENT_CORR` | 0.55 | retired in favor of the two below | **removed** |
| `_GATE6_FIRE_ON` | 0.58 | hysteresis fire-on anchor | **new** |
| `_GATE6_KEEP_ON` | 0.52 | hysteresis keep-on anchor | **new** |
| `_GATE6_OVER_MIN_MEAN` | 1.0 | over-leg degenerate-count guard (`mean(Result[bench]) ≥ this`) | **new** |

No new tuned threshold for the CITL legs — the floor is the natural `1.0`, with `_GATE6_MARGIN` as the
band. The CITL floor is calibration-in-the-large (O:E ratio); the `1.0` reference is not a fitted
number (research recommendation, Steyerberg & Vergouwe 2014; Van Calster et al. 2019).

## 6. Error handling / fail-safe behavior

- **Thin segment (< 30 stable rows):** the affected leg auto-passes. Never a false KILL on small NFL
  cells (Finding 2.4 — their clustered CITL CI spans 1.0).
- **Over-leg degenerate counts:** the `mean(Result[bench]) ≥ 1.0` guard excludes bursty rare counts
  where `Σ Result → 0` makes the ratio explode (Finding 3.3, e.g. NFL receiving-tds bench CITL 3.51
  is discreteness, not a defect).
- **Thin-holdout CITL flag:** a cell flagged **only** by CITL on a single holdout may be an unlucky
  cold window, not a serving defect (reality-check 1). The clustered 97.5% UB and the 0.03 band absorb
  sampling noise, but the lifecycle rule stands: re-confirm on the next `meditate`, do not demote on a
  single run.

## 7. Blast radius and immediate actions

- **NBA fantasy-points-prizepicks** (`centered_additive_mean10`, `shipped: "devel"`): caught by the
  recent-form leg (already, via the widening — `star_hi 0.890`) **and** corroborated by CITL
  (`CITL_out 0.951`, Gate-2 `z 0.073`). **Demote candidate** to `withheld` per §8.1 — confirm with a
  full `scorecard` sweep (the bootstrap UB, not the point estimate, decides).
- **WNBA PR/PRA** (`ratio_meanyr`, already `withheld`): corroborated by CITL (0.922 / 0.961).
- **The 6 shipped count/ZINB cells** (NBA BLK/FG3M, WNBA BLK/FG3M, NFL tds/rushing-tds) now face the
  over-leg. They auto-skip the recent-form leg (anchor) and the under-leg is one-sided, but the
  over-leg is new exposure. **Re-score all six during implementation** before this ships; if any trips
  the guarded over-leg, surface to the owner (do not silently demote — the research flagged the over
  ratios as unstable, which the `Result ≥ 1` guard is meant to absorb).
- No currently-**shipped** cell is demoted by the *under*-side change on its own (the served stars all
  run `CITL_out ≥ 0.94`).

## 8. Testing (TDD)

Each new behavior gets a failing test first, then the minimal code to pass.

- **CITL-under fires** on a synthetic stable-star frame served below outcome (`citl_hi < 0.97`);
  **silent** on a legit-regression frame (low `r_form`, `CITL_out ≈ 1.08`); **silent** (auto-pass) on
  a thin frame (< 30 stable stars) and where the clustered CI spans 1.0.
- **CITL-under applies below the anchor** — a corr < 0.55 frame with a real outcome under-shrinkage
  fires (proves the under-leg is not anchor-gated).
- **Over-leg fires** on a synthetic count/ZINB bench over-prediction with `mean(Result) ≥ 1`;
  **silent** under the `Result ≥ 1` guard; **silent** on a centered frame (family scope);
  **silent** for a non-count family.
- **Hysteresis:** a cell at corr 0.56 with `prior_g6_fired=True` stays judged (recent-form active); the
  same frame with `prior_g6_fired=None/False` is exempt; corr < 0.52 is exempt regardless of prior;
  corr ≥ 0.58 is active regardless of prior.
- **`prior_g6_fired=None` path** (A/B CLI) reproduces the pure binary `fire-on` behavior.
- Update the existing Gate-6 golden tests and `docs/ship_gate.md` (the OR-of-three semantics, the new
  constants, the hysteresis) and the `model_improvement_track.md` gate table + ledger.
- Run the three gates (ruff, golden, integration `-n0`) and the refactoring-specialist on
  `scorecard.py` (and `report.py` if the prior-state read lands there) before any push.

## 9. Open questions (carry into the plan)

- **CITL segment: stable quartile vs full top-decile.** The CITL leg does not need the `Mean10`
  stable filter (it scores vs outcome), so it *could* run on the full top-MeanYr decile for more power.
  Kept on the stable segment for parity and noise-stripping; widening it is a power/robustness
  follow-up probe (research open-question 1).
- **Over-leg vs Gate 4 overlap.** Count over-prediction may already register in Gate 4's
  `central50/80_coverage` or `g4_tail_pit_ks`. Check during implementation that the over-leg is not
  redundant before finalizing it (research open-question 2). The owner chose to build it regardless;
  this check informs whether it stays.
- **Hysteresis state location.** The prior verdict is derivable from existing `model_stats.parquet`
  columns; confirm during implementation that `report()` reads them cleanly, and decide whether to add
  an explicit `g6_recent_form_fired` boolean for unambiguous read-back.
- **Root cause unchanged.** Gate 6 detects the symptom; the MeanYr-window (and any residual centered
  recency mis-weighting) target-construction artifact is a separate research project (new normalization
  axis, must clear all gates). The recent-form leg is deletable once that lands.
