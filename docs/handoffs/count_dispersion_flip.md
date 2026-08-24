# Lane brief — count dispersion flip (R4 / Exp-3): kill the over-wide count g4 cohort

**Read first:** CLAUDE.md, [docs/ARCHITECTURE.md](../ARCHITECTURE.md),
[model_improvement_track.md](model_improvement_track.md) §6.1 Lever 1 + Rung B′ + the newest
ledger entries. This brief is the kickoff for a fresh session; it is self-contained but those
docs are the context spine.

## Mandate

The largest g4-failing cohort is **over-wide counts**: 38 confirm-ledger rows, 23 ZINB, and
26 of the 30 over-wide count rows sit on `count_dispersion_objective: crps`. The served
dispersion `c` on these cells is fit by CRPS, which trades width for sharpness and
under-corrects; fitting `c` on PIT-KS targets the gate's own statistic (both objectives share
the `0.01·log(c)²` brake, so neither can run away). The candidate change is the **default flip
crps → pit_ks** for count cells. The per-cell control already exists and is swept opt-in
(Rung B′) — no new machinery; this lane decides the default with a pre-registered experiment.

Selection is exonerated: Exp-2 ran NBA PF (clean over-wide NegBin) through loss / calibrated-v2
/ calibrated-v3 arms and all three landed g4 ≈ 0.07 (0.0724 / 0.0692 / 0.0723). No HP-selection
policy moves this cohort; the dispersion fit must.

## Phase 0 — offline replay (~1 hr, zero training, safe anytime)

The dispersion `c` is a **post-training fit on decoded frames** (fit on validation, applied to
test), so both objectives can be replayed on existing artifacts without retraining — the same
replay class that settled R5 (λ-shrinkage NULL) and R7 (μ-conditional kill).

1. Enumerate the over-wide count cells from `data/training/model_stats.parquet`
   (`g4_pass == False`, count families, `g4_iqr_ratio > 1` side) plus the ledger's 38-row
   cohort in `research/logs/confirm/`.
2. For each cell with a current test-set CSV in `data/test_sets/`: re-fit `c` under crps and
   under pit_ks on the validation portion, score held-out g4 (`training.scorecard` machinery;
   the sandboxed `sportstradamus ship scorecard --test-sets-dir … --scorecard-out /tmp/…` CLI
   never touches production stats) and an over-leg proxy (g6 over-leg direction + g1
   acceptance).
3. Output: ranked per-cell table of g4 delta and over-leg delta. This is the go/no-go for
   Phase 1 and picks the live cells. Work in the session scratchpad; keep the CSV.

## Phase 1 — Exp-3 live (pre-registered, ~half day of solo runs)

Cells: **NBA PF** (clean over-wide NegBin), **MLB hits allowed ZINB corner**,
**NFL interceptions**, plus **two shipped crps controls** (pick controls from Phase 0's
no-change end).

- **Ship the default flip iff:** ≥2 of 3 target cells improve g4 by ≥0.010, no control loses
  `ship`, and g6's over-leg + g1 acceptance are unchanged.
- **Kill the default flip iff:** the over-leg degrades on ≥2 cells → the objective stays
  opt-in per-cell and gains an explicit sharpness penalty instead.
- Either way, per-cell outcomes persist as `count_dispersion_objective` pins in
  `stat_meta.json` so the weekly cron reproduces them.

Run recipe (per arm, per cell — mirrors Exp-2):

```bash
python -m sportstradamus meditate --league <L> --market <M> --force \
  --frozen-matrix-dir research/logs/confirm/frozen_matrix/<STEM> \
  --artifact-output <run_dir>/artifacts --dependency-namespace exp3-<arm> \
  --bypass-withholding [--count-dispersion-objective pit_ks]
```

with a per-process archive copy via `SPORTSTRADAMUS_ARCHIVE_DB` (flags are hidden in `--help`;
they exist). `--artifact-output` ⇒ isolated run: no stat_calibration writes, bypasses the
training-artifacts lock.

**Hard rules for the runs:**

- Arms run **solo sequential** — wall-clipped trial counts are a function of concurrent load;
  parallel arms corrupt the comparison (Exp-2 lesson, memory `parallel-meditate-ab-contention`).
- Frozen-matrix single-axis: both arms from the SAME frozen matrix, only the objective differs.
- Deterministic-mode runs dump to `data/test_sets/deterministic/`, not model_stats — do not
  read the wrong file (memory `deterministic_screen_withheld_skip`).
- Score with the sandboxed scorecard CLI, never by writing `model_stats.parquet`.

## Complement (separate decision, do not bundle)

The worst ZINB cells have a second lever: **ZINB → DPO family swap** (built; TOV shipped it;
DPO inflation +0.004 vs ZINB +0.037). Route a cell there only after the objective flip verdict,
one cell per confirm, through the normal supersession lane.

## Gates and conventions

- Dispersion-mechanism changes are **research-gated** by convention. The R4 flip is
  pre-registered in the approved Gate-4 plan (Track 4) with the §8.2 discharge noted in
  `model_improvement_track.md`; if the research-gate hook fires on an edit, cite that plan or
  dispatch `research-analyst` for a brief rather than waiving silently.
- Any `.py` you touch: `refactoring-specialist` before push/review; then the single
  authoritative gate run — `poetry run ruff check src/sportstradamus/`,
  `poetry run pytest tests/golden/`, `poetry run pytest -m integration -n0`.
- No pushes to devel from the lane; a human merges. Record the verdict in
  `model_improvement_track.md` (Rung B′ section + ledger), one-line caveman style.
