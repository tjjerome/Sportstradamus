# Fix brief: matrix provenance crash + cross-fit gate optimism

Four defects surfaced by phase 2 of the 30-cell holdout-blind sweep; none is caused by the sweep.
A and C are root-caused and fixed. B is a single-cell lead whose first experiment came back
negative — the HP-selection rule is not the lever — so its cause is still open. D is a silent
process death that owns its own brief.

**Operating constraint.** The driver (`research/overnight/run_all_cells.sh`) spawns fresh
`meditate` subprocesses per nominee, which pick up an edit mid-run. Per
[breadth75_plan.md](breadth75_plan.md), no training-code edit lands while a sweep/confirm chain
runs — coordinate landing through the session that owns the driver. Two corollaries learned
stopping it for this fix: `SIGTERM` to the driver `bash` is deferred until its foreground child
returns, so use `SIGKILL` on the driver alone to stop the queue while the in-flight cell finishes
as an orphan; and never edit `run_all_cells.sh` while that `bash` lives, since it reads the script
by byte offset and an edit corrupts the rest of the run. The archive's DuckDB single-writer lock
is held for the lifetime of any `meditate`, so repair work that needs it either waits or points at
a copy via `SPORTSTRADAMUS_ARCHIVE_DB`; the copy sidesteps the lock but not the `model_stats`
race in §B, so only one `meditate` runs at a time either way.

---

## A. A half-populated odds-provenance block kills the run 40 minutes in

**Symptom.** `ValueError: test quote authenticity is missing, unaligned, or unsupported` from
`_split_quote_authenticity_mask`
([pipeline.py:3063](../../src/sportstradamus/training/pipeline.py#L3063)), reached via
`_fuse_skewnormal` → `_step_fuse_predictions` → `_step_calibrate_and_serve`. It fires *after*
the full HPO completes, so a nominee burns ~40 minutes before failing on a condition that is
knowable the moment the matrix loads.

**Root cause.** `_step_load_matrix` is incremental
([pipeline.py:729-752](../../src/sportstradamus/training/pipeline.py#L729)): read the cached
parquet, set `cutoff_date` to that cache's max `Date`, fetch only newer rows, concatenate. The
cached rows predate the odds-provenance block; freshly built rows carry it. `pd.concat` unions
the two schemas and back-fills all of history with `NaN` — the all-NA `FutureWarning` logged on
line 751 is exactly this event.

**The run creates the damage it then dies on, and the damage persists.**
`_step_persist_matrix_and_comps` writes the merged frame back to
[pipeline.py:888](../../src/sportstradamus/training/pipeline.py#L888) *before* the crash fires far
downstream, so the cell's cache is poisoned on disk. Every later run on that cell raises too —
including a cheap `--deterministic` search that would otherwise have taken the clean legacy path.

**Why the search never saw it.**
[pipeline.py:742](../../src/sportstradamus/training/pipeline.py#L742) —
`new_M = pd.DataFrame() if deterministic or matrix_input is not None else
stat_data.get_training_matrix(market, cutoff_date)`. `--deterministic` skips the fetch, so the
sweep trains on cached rows only, the provenance column is simply absent, and the legacy
`Archived`/`Odds_synthetic` fallback runs. Search and confirm are therefore not evaluating the
same matrix — worth holding onto independently of this bug.

**Blast radius: 2 poisoned today, 6 more queued behind them.** A census of all 98 cached matrices
puts 2 in the partly-populated state, 33 fully populated, and 63 with no provenance column at all
— the clean legacy case. MLB runs allowed carries 6849 `QuoteAuthenticity` NaN of 7179 rows; MLB
pitcher fantasy points underdog, 557 of 2435. In both the NaN band is contiguous and oldest-first,
exactly as an append-with-new-columns predicts. Pitcher fantasy points reached real gate verdicts
because its NaN rows happened to miss the scored split; runs allowed crashed **all three** of its
nominees on this and lost the cell's entire ~2 h 50 m confirm budget without producing a single
gate verdict — while still being marked `.done`, because the sweep exits 0 when every nominee
raises.

The set grows because only an in-season league appends rows. All four confirmed cells rewrote
their cache, but NFL and NHL are between seasons and appended nothing, so only the two MLB cells
went partial. `league_activity.json` has MLB and WNBA `live`, which puts six queued cells in the
same legacy-cache-plus-live-league position: MLB hitter fantasy points underdog, MLB pitches
thrown, MLB pitching outs, WNBA BLST, WNBA STL, WNBA PA.

**The block cannot be inverted, and the cheap re-derivation is not equivalent to a rebuild.**
`QuoteSource`, `QuoteBookCount` and `Odds_synthetic` are NaN on *exactly* the same rows
(6849/6849 and 557/557) — the whole provenance block goes missing together and only `Archived`
survives. Nor would the mapping be invertible if it were present: on pitcher fantasy points,
`Odds_synthetic == True` covers both `derived` (1185 rows) and `synthetic` (693).
[`scripts/inject_backfilled_odds.py`](../../src/sportstradamus/scripts/inject_backfilled_odds.py)
looks like the cheap way out — it re-derives the block and `Line`/`Odds`/`EV` per cached row from
the archive — but its `_resolve_row` passes `fallback_ev` only when the row already reads
`QuoteSource == "combo_ev_inversion"`, so it can never reach `check_combo_markets`, whereas the
training join's `_resolve_player_market_odds`
([base.py:2197-2228](../../src/sportstradamus/stats/base.py#L2197)) resolves twice and consults
combos whenever the first pass falls back. Measured on MLB runs allowed, the tool rewrote 90.8% of
`Odds`; a real rebuild of the same cell over the same 341 gamedays left all 133 shared feature
columns identical and reproduced `Odds`/`EV` exactly. The divergence is the tool's approximation,
not archive drift — **do not use it to repair a training matrix.**

**Fix.** Detect the partly-populated provenance block where the matrix is loaded and raise there,
naming the league, market, NaN count and the remedy; repair by cold rebuild. Raise at the end of
`_step_load_matrix`, before it returns: that sits upstream of `_step_synthesize_odds` and
`_step_persist_matrix_and_comps`, so a cell that would have been poisoned never writes the bad
matrix at all and stays re-runnable on its clean legacy cache. **Do not** drop the incomplete
column or widen `_split_quote_authenticity_mask` to accept partial coverage — both land in the
same place, since the fallback returns all-`True` (every quote authentic) when the legacy columns
are also missing, which lets synthetic book quotes contribute book evidence. Failing closed is the
point of that check.

**State.** `incomplete_provenance_rows` in
[matrix_audit.py](../../src/sportstradamus/training/matrix_audit.py) is the shared predicate;
`_step_load_matrix` calls it before returning. All 8 at-risk cells (the 2 poisoned plus the 6
queued in-season) are cold-rebuilt through `get_training_matrix` with the cache moved aside —
`meditate --matrix-only --bypass-withholding`, 4 h 38 m for all 8, longest cell 1 h 55 m. The
census reads 0 partial / 41 full / 57 legacy, and
`python -m sportstradamus.training.matrix_audit --root …/training_data` reports zero `null
required quote provenance` violations. Two rebuilds are their own controls: MLB hitter fantasy
points underdog reproduced its healthy cache exactly (7533 rows, 100% authentic, same date span),
and MLB pitcher fantasy points underdog reproduced the 557 legacy rows the poisoned file had
buried. The remaining 57 legacy caches are deliberately untouched — the guard makes them fail fast
instead of self-poisoning, and they only reach it when an in-season append fires.

MLB runs allowed closes the loop: the cell that crashed **all three** nominees on this and produced
zero verdicts now **ships to devel** on nominee 2 (g4 pit_ks 0.0332, iqr_ratio 1.10, all six gates).
Its matrix came out of the walk at 7209 rows — the rebuild's 6878 `book_direct` plus the 331
`model_fallback` rows the append restores — with zero null-provenance rows.

**Cold rebuild is not a superset of the incremental append — the first append closes the gap.**
On cells whose quotes come out 100% `book_direct` the rebuilt row set stops exactly at the
archive's line horizon for that market (MLB runs allowed 2026-07-16, pitching outs 2026-07-12),
while the append that poisoned them had carried later rows priced against the model. Cells that
already fall back — MLB pitches thrown, the WNBA trio — rebuild all the way to the gamelog horizon.
So a rebuild trades a few hundred model-priced rows for a fully book-grounded base.

MLB pitcher fantasy points underdog then confirmed this end to end: its 557-row rebuilt matrix came
out of the confirm walk at 2436 rows spanning back to 2026-07-27 — essentially the poisoned file's
own 2435 — split `book_direct` 557 / `combo_ev_inversion` 1185 / `model_fallback` 694, with **zero**
null-provenance rows where the poisoned run had 557. Both sides now carry the block, so the concat
that used to corrupt the cache simply works, and all three nominees reached real gate verdicts.
That `combo_ev_inversion` band is also the direct evidence for the resolver gap above: it is exactly
the population `inject_backfilled_odds` cannot reproduce.

**Adjacent smell, not yet a bug report.** `M.drop_duplicates(subset=["Player", "Date"],
keep="last")` means any `(Player, Date)` the fetch re-emits silently replaces its cached row with
a freshly built one, and nothing checks that the two agree.

---

## B. Calibration gates degrade when a corner moves to full HPO

Full HPO regressing a gate should not happen — more search on the same recipe ought to dominate
a fixed-HP fit — and it is regressing them systematically, not occasionally. **Status: one
measured cell plus a concrete mechanism.** Same corner, board (fixed HP, holdout-blind cross-fit
validation) versus its own confirm (full HPO, true holdout) — NHL goalie fantasy points
underdog, the run's only ship so far:

| gate | board | confirm | |
|---|---|---|---|
| g1 `brier_diff_ci_hi` | +0.0024 | −0.0105 | better |
| g3 `bench_z` | 0.1261 | 0.0639 | better |
| g2 `star_z` | 0.0516 | 0.0901 | worse |
| g4 `pit_ks` | 0.0280 | 0.0443 | worse |
| g5 `ece_debiased` | 0.0026 | 0.0260 | 10× worse |

n is 855 vs 818, so this is not a sample-size artifact. The discrimination gates improve under
full HPO, which is what should happen; the two calibration gates are the ones that fall apart.
Across the wider run the dominant confirm failures are g4 and g6, on cells whose boards passed
all six.

**Leading hypothesis: the HPO objective does not include calibration.** Every base corner selects
on CV loss, so the calibration penalty at
[pipeline.py:4417-4422](../../src/sportstradamus/training/pipeline.py#L4417) —
`_calibration_penalty` plus the `_gate4_pit_ks_threshold` bar, both gated on
`hpo_selection == "calibrated"` — is **off for every swept and confirmed corner**. It gets there by
default, not by a pin: `--hpo-selection` defaults to `auto`, and `cli._resolve_cell_knob` resolves
`auto` through the cell's `stat_meta` entry to `"loss"`. The base specs deliberately leave
`hpo_selection` out of both `axes` and `fixed_controls`, for the reason recorded at
[specs.py:104-109](../../src/sportstradamus/training/model_strategy/specs.py#L104); the pin at
[specs.py:235](../../src/sportstradamus/training/model_strategy/specs.py#L235) belongs to
`_yards_controls` and reaches only the two structural yards specs.

The board/confirm asymmetry is sharper than "fixed HP versus more HP": the board runs
`--deterministic`, which skips Optuna outright, so trial selection has no effect there at all,
while confirm runs a real search that optimizes CV loss while the gates score calibration. That
makes "more HPO, worse g4/g5" the expected outcome rather than a paradox, and it predicts the
run-wide pattern of g4-dominated failures. `cli._retry_calibrated_if_g4_only` already flips a cell
to `calibrated` and re-runs — but only when g4 is the *sole* failure, so the g2+g4 and g6 cells
never receive it.

**Why only one cell.** A reverted confirm leaves no trace to compare. `_confirm_one`'s `finally`
calls `prune_model_pickle`, and `report()` rebuilds `model_stats.parquet` from the pickles present
on disk, so a cell that never shipped ends with no pickle and therefore no row; `_restore_cell`
then puts the incumbent file back wholesale. The per-nominee gate values are not recoverable from
the confirm logs either — those carry failing gate *names* only.

Do **not** "fix" this by making `_restore_cell` restore a single row. The wholesale restore is
deliberate: `model_stats.parquet` is read by `report.get_market_calibration` for Kelly sizing and
by `graduation.read_gate1` for Gate-2 promotion, so leaving a reverted candidate's row behind
would size real stakes and drive promotion off a model that is not served. The fix is additive —
have the confirm walk append each nominee's gate values to a research-side ledger before the
revert, where `_failed_gates_after` already reads that row.

**Ruled out.** g5's debias (`_gate5_ece_debiased` / `_ece_debias_offset` take only `p_model` and
`y` from the scored frame, nothing cross-fit-specific). Matrix degradation on the appended rows
(for the confirmed cells, no numeric column is materially more `NaN` in the newest date decile).

**Secondary suspect, now closed.** `_step_crossfit_calibrate_and_serve`
([pipeline.py:4039](../../src/sportstradamus/training/pipeline.py#L4039)) overrides only
`prob_params`, `y_proba_*`, `decoded.ev`, `fused.weighted_mean`, `fused.gate_blend_test`,
`calibrated.{sn_sigma_blend_test, sn_alpha_blend_test, mix_blend_test, r_test, phi_test}`,
`val_calibrated` and `pit_recal_by_row`. Everything else comes from the unrotated
whole-validation fit — notably `c_opt`, `skew_cal`, `T_opt` and `model_weight`, which are fit
on 100 % of validation including the rows they then score, and which ride the pickle into
`_build_filedict` ([pipeline.py:4538](../../src/sportstradamus/training/pipeline.py#L4538)).
None of them re-enters gate scoring: `scorecard.py` never references any of the four, and
`compute_gates` takes only the scored frame plus identity strings. They stay reported context,
and the arrays the gates actually score are the per-fold-calibrated ones.

Note the direction argues *against* a trivial explanation: production's validation/test split is
a `(Player, Date)` identity hash, so the same player sits on both sides and the production
calibrator enjoys mild within-player leakage. Cross-fit folds are player-disjoint and should
therefore be **harsher**, not kinder. It is nonetheless the optimistic one.

**Two experiments, one axis each.** The board→confirm step moves the HP budget *and* the
evaluation frame together, so neither alone is identified by the numbers above.

1. *HPO axis — done, negative.* See below.
2. *Protocol axis.* Hold the corner and hyperparameters fixed and run
   `meditate --deterministic --bypass-withholding` twice, with and without `--holdout-blind`.

**The confirm walk does not hold its own frame fixed either, and new data is not the reason.**
Because every nominee retrains under `--force`, the matrix moves *between* nominees of the same
cell: MLB pitcher fantasy points underdog scored its three at `n_validation` 373, 363, 358, and NHL
goalsAgainst at 815, 801, 785. The obvious explanation — an in-season append — is wrong: NHL's last
gameday is 2026-06-14 and nothing was appended, yet the parquet's mtime lands mid-walk. The cause is
`_step_persist_matrix_and_comps`, which recomputes comps and rewrites the file on every `--force`
run regardless of whether a single row is new, and comp features are not stable across recomputes
(`_nonmlb_comp_features` fires twice per gameday and the matrix keeps the *last* value, not the
mean). So a cell's nominees are ranked against each other on drifting frames even in the offseason.
Same defect as the discarded experiment below; it deserves its own fix — freeze the cell's matrix
once at the top of the walk and run every nominee against it.

**Run any such experiment from a frozen matrix.** `--force` — which is exactly what confirm
passes — rewrites the cell's parquet, so a candidate arm compared against a historical confirm row
moves the matrix as well as the axis under test. The first HPO-axis attempt did this and had to be
discarded: `strategy_matrix_hash` went `e7342f29…` → `828e0e52…` and `n_validation` 818 → 788, and
the incumbent pickle put the candidate on the 5-minute warm search instead of the confirm's full
one. Freeze instead — copy the parquet to a directory, write the five fields
`validate_matrix_manifest` checks (`builder_version`, `schema_version`, `row_count`,
`feature_schema`, `matrix_sha256`) into `<slug>.manifest.json` beside it, and run **both** arms
back to back under `--frozen-matrix-dir` plus `--artifact-output`. `--artifact-output` also
suppresses `report()` ([pipeline.py:4661](../../src/sportstradamus/training/pipeline.py#L4661)), so
production `model_stats.parquet` is never written and nothing needs restoring; compare the two arms'
isolated test-set CSVs with `python -m sportstradamus.training.scorecard --baseline … --candidate …`.
Point the run at a copied archive via `SPORTSTRADAMUS_ARCHIVE_DB`
([archive.py:301](../../src/sportstradamus/helpers/archive.py#L301)) to dodge the DuckDB writer
lock, and never run alongside the driver regardless — two `meditate` processes that both reach
`report()` race on the single `model_stats.parquet`.

**Some confirm failures are diverged fits, not degraded calibration.** Across the nominee ledger
(26 nominees, 15 cells) a saturated dispersion calibrator separates outcomes perfectly: every row
whose `dispersion_cal` sits on its 0.1 floor fails — 6 of 6 — and no ship has one. Those rows carry
absurd `shape_ratio` values (NFL qb yards 1.7e9 and 2.7e9, qb tds 4.6e3–1.5e6, NBA TOV 48), so the
SkewNormal fit blew up and the calibrator clamped trying to rein it in. That is a different failure
from a corner that merely calibrates badly, and it costs a full-HPO retrain to discover.

Read the *floor*, not the shape ratio: `shape_ratio > 10` does not discriminate on its own, because
NFL attempts ships at 76.6 with a healthy `dispersion_cal` of 1.01. The conjunction is what matters.

**The divergence is SkewNormal-only, and it is costing the board's budget on count cells.** All six
floored rows are SkewNormal, on three cells (NFL qb tds, NFL qb yards, NBA TOV). The count families
never come close: across ten rows, DPO / NegBin / ZINB top out at `shape_ratio` 1.31 and every DPO
row sits at `dispersion_cal` 0.83–0.93. Ships by family are DPO 3/5, NegBin 1/1, SkewNormal 5/16,
ZINB 0/4.

Two of the three diverging cells are *count* stats (qb tds, TOV) whose top-slack board corner is
nonetheless SkewNormal — so ranking nominees by board slack spends the cell's full-HPO budget on the
family that will blow up while a fit-stable count corner sits further down the list. That is the same
board-optimism failure as `crossfit_board_ships_optimistic`, one layer down: the board is not just
optimistic about the *gate*, it is optimistic about the *fit converging at all*. A cheap guard is
available without touching the ranking — a count-stat cell whose nominee is SkewNormal should have
its count-family corner tried first, or at least kept in the walk when the SkewNormal one diverges.

NFL qb yards is still the cell to chase, but **not** for the reason first recorded here. Its
`g1_has_edge` True is vacuous: every g1 statistic on both nominees is NaN, and both
`_g1_within_tie_margin` and `_below_zero_ci_bound` return True on a blank CI bound by design — "no
book to beat". So the flag means *unmeasured*, never *measured and winning*. Do not read
`g1_has_edge` without checking `g1_brier_diff_ci_hi` is non-null first; 11 of the ledger's 31 rows
have a vacuous g1, six of them ships.

What is left after that correction is thinner but still worth the retrain: qb yards has 339 authentic
quotes in its matrix, so its book comparison is recoverable once those rows reach the scored split,
and it is continuous, so the count-family route above does not apply to it.

**Read this walk's ship count as breadth, not staking volume.** Five of its ships are structurally
book-less — zero authentic quotes across the entire matrix, not merely across the scored split:

| cell | authentic / rows |
|---|---|
| NFL targets | 0 / 13499 |
| NBA DREB | 0 / 13915 |
| WNBA BLST | 0 / 14287 |
| WNBA STL | 0 / 15000 |
| WNBA DREB | 0 / 14737 |

Rebounds, steals, blocks and targets are DFS-only markets, so this is the market structure rather
than a collection gap, and the pipeline handles it deliberately: `report()` sets
`betting_active = ship AND kelly_shrinkage > 0` with NaN filled to 0, and `get_market_calibration`
returns NaN so Kelly falls through to its next source. Those cells are deployable and sized to zero
until they prove live edge. Nothing to fix — but do not quote them alongside book-backed ships as if
the two mean the same thing.

**HPO axis: refuted on this corner.** NHL goalie fantasy points underdog, both arms trained from
the frozen `e7342f29…` matrix the original confirm used, `n_validation` 818 on both:

| gate | `loss` | `calibrated` |
|---|---|---|
| g1 brier_diff | −0.0240 [−0.0394, −0.0082] | −0.0241 [−0.0394, −0.0083] |
| g2 star_z | 0.08 | 0.09 |
| g3 bench_z | 0.07 | 0.07 |
| g4 iqr_ratio | 0.818 | 0.788 |
| g5 ece_debiased | 0.0228 | 0.0241 |
| ship | SHIP | SHIP |

Calibrated trial selection moves g4 *away* from 1.0 and leaves g5 marginally worse; supersede reads
`d_mean −0.0005 [−0.0024, +0.0013]` → HOLD. The two arms are statistically indistinguishable and
every point estimate favours `loss`, so the HP-selection rule is not the lever here and the
`auto`→`loss` default needs no change.

Read that as corner-specific, not settled. This cell *ships* under both arms — it is not one of the
degrading corners §B is about, and it was chosen only because it was the one cell whose confirm row
survived. The nominee ledger now preserves gate values for every nominee, so re-run the axis on a
corner that actually fails g4 — NFL qb tds carries `pit_ks 0.368` against a threshold two orders of
magnitude tighter, which is far outside anything trial selection could close.

---

## C. A warm-start hurdle retrain dies on a stale monotone vector

**Symptom.** `lightgbm.basic.LightGBMError: Check failed:
(static_cast<size_t>(train_data_->num_total_features())) == (config->monotone_constraints.size())`
from `HurdleZINB.fit` ([hurdle.py:215](../../src/sportstradamus/hurdle.py#L215)). It killed the
top-slack nominee of NFL interceptions outright — the confirm walk logged
`REVERTED … failed retrain error` and moved on, spending the cell's budget on the two weaker
nominees, both of which then failed gates.

**Root cause.** The hurdle branch of `_step_select_hyperparams`
([pipeline.py:1332](../../src/sportstradamus/training/pipeline.py#L1332)) skips Optuna and hands
the pickle's warm-start params straight through to LightGBM. `monotone_constraints` is sized to the
feature set the pickle was trained on, and LightGBM requires exactly one entry per training column.
Every hurdle pickle on disk stores a vector matching its own `expected_columns` (NHL 136, MLB 118,
WNBA 287, NFL 419), so any feature-set change since that fit turns the next warm retrain into an
abort. The two Optuna paths are immune: they pass the vector through
`hp_search_space["monotone_constraints"]`, which is recomputed from the current columns.

That also explains why only nominee 1 crashed. `_confirm_one`'s `finally` prunes the pickle, so
nominees 2 and 3 started cold and sized the vector correctly.

**Fix (landed).** Recompute `monotone_constraints` on the warm hurdle path instead of carrying the
pickle's copy forward — it is a pure function of the current column list, already computed a few
lines above. Covered by `tests/golden/test_hurdle_warm_start_monotone.py`. NFL receiving tds is the
one shipped hurdle cell whose stored width (419) is below the current NFL feature count, so it was
next in line for the same abort. NFL interceptions then re-confirmed all three nominees to real
verdicts (it still fails g4+g6 — recovering the nominee was never going to rescue the cell).

---

## D. A forced-DPO confirm dies without a traceback

The same `REVERTED … failed retrain error` symptom as §C, different cause and **unresolved**: NFL
carries lost both nominees ~6 minutes into each fit with no exception in a log that does capture
stderr. Cause not identified; five queued cells carry DPO in their top-2 corners and are expected
to hit it. Diagnosis, exclusions, and the reproduction recipe: [dpo-confirm-crash.md](dpo-confirm-crash.md).
