# Lane record — count dispersion flip (R4 / Exp-3): default stays `crps`

**Status: closed 2026-08-24.** The default flip is **not adopted**; `count_dispersion_objective`
remains an opt-in per-cell pin. Context spine: CLAUDE.md,
[docs/ARCHITECTURE.md](../ARCHITECTURE.md),
[model_improvement_track.md](model_improvement_track.md) §6.1 Rung B′.

## The question

The count branch fits its dispersion scalar `c` by CRPS while Gate 4 scores a randomized-PIT
KS. `pipeline._dispersion_pit_ks_loss` retargets the fit at the gate's own statistic; both
objectives carry the same `0.01·log(c)²` brake. This lane asked whether that becomes the
default.

The cohort behind the question is real: 30 over-wide count rows in
`research/confirm_nominee_gates.csv` fail Gate 4 (23 ZINB), and 26 of them belong to cells whose
current `stat_meta.json` pin is `crps`. The mechanism is structural rather than coincidental —
**a failing cell never ships, so a confirm never persists a pin for it, so it trains on whatever
the default is.** The default's blast radius is therefore exactly the unpinned set: 9 ZINB cells
(NBA OREB · NBA PF · NBA TOV · NFL interceptions · NFL rushing tds · NFL tds · NHL faceOffWins ·
NHL goalsAgainst · WNBA BLK). The other 41 count cells carry explicit pins — 17 `crps`, 24
`pit_ks` — that no default change touches.

## Verdict

Three independent measurements. The two offline ones lean pro-`pit_ks` and neither is large
enough to move an outcome; the live full-HPO pair leans the other way and is inside noise.

**Offline replay (44 count cells, median of 6 seeded half-splits).** Each cell's persisted shape
is divided by its shipped `dispersion_cal` to recover the pre-dispersion base, then `c` is refit
under each objective on one half and scored on the other with
`scorecard._randomized_pit_ks`. A cell enters only if scoring its dump unchanged over every row
reproduces its recorded `g4_pit_ks`, so the un-scaling is verified rather than assumed.

- `pit_ks − crps`: median **−0.0001**, better on 25/44, worst case +0.0038.
- Threshold clearance: `crps` 41/44, `pit_ks` 41/44. **Zero flips in either direction.**
- By family the sign is not even stable: ZINB −0.0002, NegBin −0.00003, DPO **+0.0001**.

**Cross-fit board (98 paired corners, `research/model_strategy_board_crossfit.csv`).** Corners
identical in league, market, family, normalization, training loss, zinb_mode, blending,
hpo_selection, posthoc, matrix hash and split — differing only in the objective.

- `g4_pit_ks` lower under `pit_ks` on 82/98 (Wilcoxon p = 4.7e-12), median −0.0017.
- `g4_pass` 60 → 69, but `ships` only 36 → 37 (2 gained, 1 lost).
- `g6_pass` **identical on all 98 pairs**; `g1_brier_diff_ci_hi` median Δ 0.0000.

The board's larger median comes from cells the replay cannot cover (NBA TOV −0.067, NHL
goalsAgainst −0.022). Both are excluded for honest reasons — see the coverage note below — and
both fail Gate 1 anyway, so neither would ship on a Gate-4 fix.

**Live single-axis arm (NFL tds, NegBin, full HPO, current matrix).** Two 300-trial runs on the
same frozen matrix `7918c1b8`, `StrategyControlsJSON` differing only in the objective. Both
converged to the same CV loss (0.5975 / 0.5973) and both **KILL on g4 + g6** — and `pit_ks`
lands *worse* on the statistic it optimizes: `g4_pit_ks` 0.0743 vs 0.0700, tail 0.0738 vs 0.0691.
It is better on g5 (`ece_debiased` 0.0241 vs 0.0336) and slightly worse on g1/g2/g3. One seeded
pair, so ±0.004 sits inside HPO run-to-run noise — but the objective is one axis inside a
300-trial search selected on something else, and it does not steer the outcome even on the cell
with the least headroom to close.

Against the pre-registered bar (≥2 of 3 pilots improving g4 by ≥0.010, no control losing `ship`,
g6 over-leg flat) the flip **fails the ship trigger and does not fire the kill trigger**: the
effect is an order of magnitude too small to ship, and it degrades nothing. That closes §8.2 open
question #9(b) — a PIT-KS objective with no sharpness brake beyond `0.01·log(c)²` did not move
the g6 over-leg on any of the 98 paired corners.

## Why fixing Gate 4 ships almost nothing here

Splitting the over-wide cohort by which *other* gates also fail is what re-points the lane:

| cell | family | g4-only rows | co-failures | replay Δ | board Δ |
|---|---|---|---|---|---|
| NFL tds | ZINB | 5 / 6 | one row g6 | −0.0002 | no pair |
| NBA PF | NegBin | 3 / 3 | none | not replayable | 0.0000 |
| MLB hits allowed | DPO | 2 / 2 | none | cell is SkewNormal today | no pair |
| MLB runs allowed | ZINB | 2 / 4 | g1 | +0.0007 | +0.0083 |
| NBA TOV | ZINB | 0 | g1, g6 | not replayable | −0.0673 |
| NHL goalsAgainst | ZINB/DPO | 0 | g1 | no test CSV | −0.0224 |
| NFL interceptions | ZINB | 0 | g6 (g1 passes — see successor lane) | −0.0013 | +0.0067 |

NBA TOV and NHL goalsAgainst are Gate-1 walled — their deficit is signal, not width, which routes
to §6.3 rather than here. The cells that are Gate-4-only do not respond to *this* lever: NFL
interceptions replays at 0.1300 / 0.1319 / 0.1306 (as-shipped / crps / pit_ks) against a 0.0698
threshold, and NFL tds at 0.0591 / 0.0590 / 0.0588 against 0.0500. What does move them is the
predictive **mean**, not the dispersion — see the successor lane below.

## Corrections to the kickoff framing

- **The ledger is `research/confirm_nominee_gates.csv`.** `research/logs/confirm/` holds raw
  meditate stdout captures, one per cell, with no row structure.
- **`model_stats.parquet` cannot enumerate the cohort.** `report()` rebuilds it from the pickles
  on disk and the ship gate prunes non-shipping pickles, so every row in it passes every gate.
- **`g4_iqr_ratio` is the retired proxy**, not the gate. It flags 3 of the 30 over-wide rows.
  Direction comes from `central50_coverage` / `central80_coverage` — with the caveat below.
- **Coverage is not a dispersion read on a low-mean lattice.** NFL tds' shipping NegBin nominee
  records `central50_coverage` 0.88 alongside `g4_pit_ks` 0.0203: on a support that is mostly
  {0, 1} the central-50 interval covers ~everything by construction. Read the KS, not the
  coverage, on these cells.
- **MLB hits allowed is a SkewNormal cell** (shipped devel), so the knob is inert there and it
  could not serve as a pilot.
- **NBA PF and NBA TOV are not replayable offline.** No row in the ledger reproduces their
  persisted dumps under any recorded `dispersion_cal` (PF off by 0.014, TOV by 0.008), so dump
  and gate row come from different runs. Their evidence is board-only.

## Residue

- The knob stays a swept axis and an opt-in pin. Nothing to revert; no code changed.
- **NFL tds' shelf nominee is matrix-stale — do not re-earn it.** A NegBin corner
  (`dist_training_loss=nll`, `blending=nll`, `count_dispersion_objective=pit_ks`, `posthoc=none`)
  passed all six gates at full HPO on 2026-08-03 (g4 0.0203, g1 ci_hi −0.0271, BSS +0.195) and was
  never landed. Re-run here at full HPO (300 trials) on the currently pinned matrix, with a
  byte-identical `StrategyControlsJSON`, it **KILLs on g4 and g6**: g4 0.0743 against a 0.05
  threshold, `g6_citl_ci_hi` 0.783 against the 0.97 pass line, while g1/g2/g3/g5 hold
  (g1 ci_hi −0.0208, BSS comparable). Its `crps` twin KILLs the same two gates (g4 0.0700,
  `g6_citl_ci_hi` 0.805), so the loss is the matrix, not the objective. The cause is visible in the ledger: that row carries
  `strategy_matrix_hash` `285d0daa` (n 2484), and every NFL tds row from 2026-08-07 onward carries
  `7918c1b8` (n 2466) — the matrix was rebuilt in between and the corner does not survive it.
  **General rule: a ledger `ship=True` is evidence only against its own `strategy_matrix_hash`.**
  On the current matrix the ZINB incumbent (g4 0.054, g6 pass) is strictly the better corner, so
  the cell stays `ZINB` / `withheld` and its residue is a Gate-4 gap of ~0.004 that no width lever
  in this lane closes.
- Three further never-landed `ship=True` nominees sit in the ledger, all SkewNormal and all
  outside this lane — but check the matrix hash before treating any of them as a free re-earn.
  NBA FGA (nominee `f779a7b4`, latest row `723e5597`) and WNBA REB (`3f884021` vs `0ca8378d`) are
  matrix-stale exactly like NFL tds; only MLB hitter fantasy points underdog (`5d00d1fc`) has no
  later row to contradict it, and that only means nothing has re-run the cell since 2026-07-29.
- **Successor lane: [count_mean_calibration.md](count_mean_calibration.md) — closed.** It traced
  the Gate-4 supremum to a predictive-mean deficit rather than dispersion or zero-inflation, and
  confirmed at full HPO that a mean-stage corrector **does** take Gate 4 across its threshold on
  both convertible cells (NFL tds 0.0700 → 0.0431, NBA PF 0.0627 → 0.0366). Neither shipped there:
  NFL tds read as blend-bound on Gate 6 — actually corrector *stage* order, fixed and shipped in
  [mean_corrector_stage_order.md](mean_corrector_stage_order.md) — and NBA PF is blocked by the
  single-valued `posthoc` slot, which still stands. That lane also
  lands two corrections on this record — **NFL interceptions passes Gate 1** on its recent rows and
  fails Gate 6, and **NBA PF's newest ledger rows are NegBin, not ZINB**, so the dump-based reads
  for that cell above describe a run the ledger does not contain.
