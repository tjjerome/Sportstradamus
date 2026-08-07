# Automating the supersession test in the strategy-sweep confirm loop

## Problem

`model-strategy-sweep --confirm` (in `training/model_strategy_confirm.py`) turns a ranked strategy
board into shipped cells. Today it only handles **withheld** cells: it persists the best reproducible
corner, retrains at full HPO, and keeps the cell on `devel` if it clears a clean 6/6 scorecard, or
reverts (restore `stat_meta` + `prune_model_pickle`) if it fails.

**Already-shipped cells are skipped.** When the board is swept with `--include-shipped` (to hunt a
better strategy for a live cell), any shipped cell that ranks a winning corner is announced and left
for a manual walkthrough. Two reasons it can't reuse the withheld path:

1. **Wrong gate.** A baselined (already-serving) cell must not re-ship on a fresh 6/6 — that would
   let strategy-churn swap a live cell on noise. The correct bar is the **supersession test**
   (S1 + S2 + S3), which requires the candidate to be *statistically sharper* than the incumbent,
   not merely to pass standalone.
2. **Wrong revert.** The withheld revert calls `prune_model_pickle`, which dark-outs the cell. For a
   live cell that must **not** happen on a losing candidate — the incumbent has to come back
   byte-identical and keep serving.

This spec automates the supersession test inside `--confirm` so a live cell can be re-shipped
**safely**: the loop does the expensive work (train the candidate, run S1/S2/S3), shows the operator
the comparative stats, and swaps the live cell **only** when the test passes *and* the operator
confirms. Any failure — a losing verdict, a `meditate` crash, an interrupt — restores the incumbent
byte-identical and never prunes its pickle.

## Goals

- `--confirm` handles shipped cells via the supersession test, in the **same** run that auto-ships
  withheld cells (one mixed walk of the board).
- A live-cell swap requires **both** an S1/S2/S3 pass **and** an explicit operator `yes` at a
  per-cell prompt that shows the comparison.
- On any loss/failure/crash-in-`finally`-reach, the incumbent is restored byte-identical (pickle,
  test-set CSV, calibration, model-stats row, `stat_meta` entry) and its pickle is never pruned.
- Reuse the existing engine: `scorecard.supersede_verdict` and its `_supersede_headline` renderer,
  `helpers.io` path builders. No new module, no new distribution logic.

## Non-goals

- No sandboxed full-HPO training. Real `meditate` has no output-redirect flag (it always writes the
  pickle, `test_sets/{cell}.csv`, the `model_stats` row, and the calibration entry to canonical
  paths), so the candidate is trained **in place over a snapshot** — the established manual flow.
  Building a sandbox-redirect through the training pipeline is out of scope.
- No crash journal / `--recover`. The supersede body runs under `try/finally`, matching the safety
  model the existing withheld loop already has. A hard `SIGKILL`/power-loss mid-train (no `finally`)
  is the one uncovered gap; recovery is the whole-file `stat_meta` backup plus re-`meditate` of the
  incumbent. Adding a journal only to this path would be inconsistent hardening — deferred.
- No `--refresh-baseline`. The baseline is the incumbent's on-disk test-set CSV (see below).
- No unattended promote. `--yes` skips only the upfront gate; live-cell promotions always prompt.

## Flow — one `--confirm`, mixed behavior

`--confirm` walks the ranked board once. Per cell, the family-aware candidate is selected exactly as
today (`_candidate` — best-by-slack **fully persistable** shipping corner, else `ranks_only`/`None`).
`_split_shippable` partitions the confirmable candidates by the cell's current `shipped`:

- **Withheld candidate (fresh, never served)** — *unchanged*. Persist strategy + `shipped="devel"`,
  `meditate --force`, read the official `ship`. Pass → keep (report `SHIPPED`). Fail → restore
  `stat_meta` entry + `prune_model_pickle` (report `REVERTED`, back to dark). This is `_confirm_one`.

- **Shipped candidate (live cell, only present under `--include-shipped`)** — new supersede path,
  `_supersede_one`, run inside one isolated per-cell iteration:

  1. **Snapshot** the six incumbent artifacts to a per-cell backup dir. The test-set CSV copy is the
     **S2/S3 baseline**.
  2. **Persist** the candidate strategy to the `stat_meta` entry (leave `shipped` as-is — the cell
     stays `devel`/`main`). Full-HPO `meditate --league L --market M --force` retrains the candidate
     *over* the incumbent's canonical paths.
  3. **Verdict**: `supersede_verdict(baseline_df, candidate_df, pred_col, league=…, market=…)` where
     `baseline_df` = the snapshot CSV and `candidate_df` = the just-written on-disk CSV. It returns
     `s1_pass` (6 gates standalone), `s2_*` (paired-Brier CI; passes iff `s2_ci_lo > 0`), `s3_*`
     (Memmel Sharpe-z; passes iff `z > 1.645`), and `ship` = AND of the three.
  4. **SUPERSEDE** (`ship` true) → print the comparison (`_supersede_headline`), then
     `click.confirm("Promote {L} {M} to {strategy}?")`. **Yes** → keep the candidate (strategy stays
     swapped, the new pickle serves), report `SUPERSEDED`. **No** → restore incumbent byte-identical,
     report `HELD (declined)`.
  5. **HOLD** (`ship` false) → print the comparison, auto-restore the incumbent, report
     `HELD:<failed leg>`. **No prompt** — the operator cannot override a failed statistical test; the
     prompt only gates promotions the test already passed.

A live-cell swap therefore requires the S1/S2/S3 test to pass **and** an operator `yes`. `--yes`
skips the single upfront gate (`"persist N withheld + test M supersede candidates?"`) but every
supersede promotion still prompts individually.

## Safety — snapshot / restore

### The restore set (serve-read artifacts)

The restore set is scoped to every artifact serving reads back, so a HOLD leaves the incumbent
serving exactly what it served (verified against `training/pipeline.py` + `training/report.py`):

| Artifact | Path | Scope | Why it matters |
|---|---|---|---|
| Model pickle | `data/models/{slug}.mdl` (`model_pickle_path`) | per-cell file | inference loads this by path; carries every calibrator (PIT/isotonic/temp/dispersion/blend) — the thing that serves |
| Test-set CSV | `data/test_sets/{slug}.csv` (pipeline.py:1565) | per-cell file | overwritten by the candidate; the snapshot is the S2/S3 **baseline** |
| Calibration | `data/config/stat_calibration.json` | shared (gitignored) | per-cell `{cv,std,zi}` feeds serve-time decode |
| Book weights | `data/config/book_weights.json` | shared (gitignored) | per-cell key read at serve time (`helpers/config.py`); strategy-independent, restored for exactness |
| Model stats | `data/training/model_stats.parquet` + `.csv` | shared | gate verdict + dashboard row |
| Importances | `data/training/feature_importances.csv` + `feature_correlations.csv` | shared | drift-monitoring only |

The stat_meta entry is restored separately (the whole-file `_backup_stat_meta` + the in-memory
original). **Deliberately excluded** (not serve-read, strategy-independent, regenerated identically by
the next `meditate`, so a candidate-state leftover after a HOLD changes nothing served): the per-cell
training-matrix cache `data/training_data/{slug}.parquet`, and the per-league caches (gamelog,
`comps.json`, correlation matrices). Snapshotting the large matrix parquet for zero serving benefit
would be waste.

`_snapshot_cell(league, market)` copies all six to a per-cell backup dir under
`research/logs/confirm/…`. `_restore_cell(league, market, backup_dir, meta, original_entry)` copies
them back and restores the `stat_meta` entry from the in-memory original.

### Why whole-file restore of the shared files is correct

The snapshot is taken **right before that cell trains**, and restore happens **before the next cell
trains** — the cell's window is isolated. Between snapshot and restore only this cell's `meditate`
runs, so whole-file restore of the shared files (`model_stats`, `stat_calibration`, importances)
reverts *only* this cell's write. Earlier withheld auto-ships in the same run committed their rows
before this snapshot was taken, so they survive the restore.

### try/finally

The entire `_supersede_one` body runs under `try/finally`. Any exception — `meditate` non-zero exit
or timeout, `KeyboardInterrupt`, a verdict error, a `No` at the prompt routed as a restore — triggers
the same `_restore_cell`. A live cell is never left mutated by a failure and its pickle is never
pruned. The only uncovered gap is a hard `SIGKILL` mid-train (no `finally`); recovery is the
whole-file `stat_meta` backup + re-`meditate` of the incumbent.

## S2/S3 baseline vintage

The baseline is the incumbent's on-disk `test_sets/{cell}.csv`, snapshotted before the candidate
overwrites it; the candidate is the post-train CSV at the same path. `supersede_verdict` pairs them
on the shared-event intersection (`_supersede_paired_brier_ci` / `_supersede_paired_sharpe` align on
`Result`/`Line` events). This matches established manual practice (the
`scorecard --baseline … --candidate …` flow in `docs/handoffs/model_improvement_track.md` §7.1).

**Caveat, recorded not fixed:** that CSV is from the incumbent's *last* real train, so if gamedays
have accrued the two holdouts differ in vintage and the pairing falls to their intersection. An
operator wanting a same-vintage comparison re-`meditate`s the incumbent first to refresh the
baseline before running `--confirm --include-shipped`. No `--refresh-baseline` flag is added.

## Code shape

All new code in `training/model_strategy_confirm.py` (stays well under the monolith ceiling). New
functions, each ≤ CC 10:

- `_snapshot_cell(league, market) -> pathlib.Path` — copy the six artifacts to a per-cell backup dir.
- `_restore_cell(league, market, backup_dir, meta, original_entry) -> None` — copy them back +
  restore the `stat_meta` entry.
- `_supersede_one(meta, cand) -> tuple[str, str, str, str]` — the snapshot → persist → train →
  verdict → prompt → keep/restore orchestration under `try/finally`. Parallel to `_confirm_one`.
- `_render_supersede_comparison(verdict, cand) -> str` — reuse `scorecard._supersede_headline`
  (same training package) over the verdict dict; expand to the per-gate lines if a one-liner reads
  too dense.

Reuse: `scorecard.supersede_verdict`, `scorecard._supersede_headline`, `scorecard.load_test_set`;
`helpers.io.model_pickle_path`, `MODEL_STATS_PATH`, `market_file_slug`; the sweep's served pred
column (`_SHIP_PRED_COL = "Blended_EV"`); the existing `_confirm_meditate` (the `meditate --force`
subprocess wrapper — the supersede path calls it identically, then runs the verdict on top).

`run_confirm` change: `_split_shippable` still partitions fresh vs shipped, but shipped candidates
now route to `_supersede_one` instead of `_announce_already_shipped` + skip. The final
`_print_confirm_report` gains the `SUPERSEDED` / `HELD` outcomes alongside `SHIPPED` / `REVERTED`.

## Testing (all mocked — no model trains, no real `stat_meta` touched)

- `_snapshot_cell` / `_restore_cell` round-trip: write known bytes to the six files, snapshot, mutate
  all six, restore, assert byte-identical + `stat_meta` entry restored.
- **Critical safety test**: `_supersede_one` with `supersede_verdict` → HOLD **and**, separately,
  with `_confirm_meditate` raising — both restore every artifact + the `stat_meta` entry and do
  **not** call `prune_model_pickle` (the live cell keeps serving). Mirrors the withheld
  `_confirm_one` revert test.
- SUPERSEDE + operator `yes` keeps the candidate (strategy swapped, no restore). SUPERSEDE +
  operator `no` restores the incumbent.
- `run_confirm` on a mixed board: one withheld candidate auto-ships (existing path), one shipped
  candidate routes to supersede, single combined report with both outcome vocabularies.
- `_render_supersede_comparison` surfaces the S1/S2/S3 numbers (`s1_pass`, `s2_mean`/CI, `s3_memmel_z`,
  `ship`).

## CLI

No new flags. `--confirm` now covers both cell types; `--include-shipped` (already exists) is what
puts live cells on the board so `--confirm` can supersede them. `--yes` still skips only the upfront
gate.

## Docs

- README §1: extend the `--confirm` paragraph — under `--include-shipped`, shipped cells now run the
  supersession test (train candidate → S1/S2/S3 → show comparison → confirm), swapping a live cell
  only on a passing test **and** an operator `yes`; a loss restores the incumbent.
- Memory `project_model_strategy_driver.md`: add the supersede-in-confirm behavior + the snapshot/
  restore landmine (six artifacts, per-cell isolated window, restore-not-prune for live cells).

## Verification

1. `poetry run ruff check src/sportstradamus/` — clean.
2. `poetry run pytest tests/golden/` — clean (new snapshot/restore + supersede tests).
3. `poetry run pytest -m integration -n0` — clean.
4. `refactoring-specialist` over `model_strategy_confirm.py` before any review/commit; then the
   single authoritative gate pass.
5. Live smoke (operator, optional, real compute): `model-strategy-sweep --league WNBA --market AST
   --include-shipped --confirm` on a shipped cell — verify a HOLD restores the pickle + CSV +
   `stat_meta` byte-identical and leaves the cell serving, and a SUPERSEDE + `yes` leaves the
   candidate in place with the `stat_meta` strategy swapped.
