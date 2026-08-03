# Fix brief: matrix provenance crash + cross-fit gate optimism

Four defects surfaced by phase 2 of the 30-cell holdout-blind sweep; none is caused by the sweep.
All four are now root-caused and fixed: A and C earlier, D in its own brief, and B below —
attributed across four mechanisms by the 37-nominee ledger plus two single-axis experiments, with
the walk/board fixes landed.

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

**The block cannot be inverted, so it has to be re-resolved.** `QuoteSource`, `QuoteBookCount` and
`Odds_synthetic` are NaN on *exactly* the same rows (6849/6849 and 557/557) — the whole provenance
block goes missing together and only `Archived` survives. Nor would the mapping be invertible if it
were present: on pitcher fantasy points, `Odds_synthetic == True` covers both `derived` (1185 rows)
and `synthetic` (693).

[`scripts/inject_backfilled_odds.py`](../../src/sportstradamus/scripts/inject_backfilled_odds.py)
does that re-resolution, and is **rebuild-equivalent**. It was not always: its `_resolve_row` was a
columns-only function taking no `Stats` instance, so it *structurally could not* reach
`check_combo_markets` and faked the combo branch by preserving `EV` where the row already read
`QuoteSource == "combo_ev_inversion"` — which a legacy matrix never does. Measured on MLB runs
allowed it rewrote 90.8% of `Odds` where a rebuild changed none. It now calls the training join's
own resolver, `Stats.resolve_player_market_odds`
([base.py:2204-2242](../../src/sportstradamus/stats/base.py#L2204)). That reads only `stats.index`
and `stats["Avg10"]` — both cached — and its combo fallback needs only the 300-day log window
(`Stats.window_short_logs`, the cheap prefix of `base_profile`, not its expensive body), so one
call per gameday reproduces what a rebuild resolves at one batched archive query per gameday.

**The control.** `MLB walks-allowed`, cold-rebuilt and migrated against the same archive, agrees on
**100%** of all three provenance columns (4615/4615) and on `Line` (4615/4615); `Odds`/`EV` differ
on 21 rows (0.46%, max delta 0.021), from `fit_book_weights` running during the rebuild. Cost:
**42 m 13 s rebuild vs ~1 min migration**, 345 gamedays at 7.34 s/gameday. The generalizable
lesson is that the tool diverged because it re-implemented the resolver instead of calling it.

One caveat carries over: the combo path is not point-in-time (`_submarket_ev` calls
`archive.get_ev` with no `at=`), so re-running the migration after new archive rows land can yield
different combo quotes. It reproduces *a rebuild now*, which is the right contract.

**Fix.** Detect the partly-populated provenance block where the matrix is loaded and raise there,
naming the league, market, NaN count and a remedy that is runnable exactly as printed — the repair
is a module, not a console script. Raise at the end of
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
`meditate --matrix-only --bypass-withholding`, 4 h 38 m for all 8, longest cell 1 h 55 m. Two
rebuilds are their own controls: MLB hitter fantasy points underdog reproduced its healthy cache
exactly (7533 rows, 100% authentic, same date span), and MLB pitcher fantasy points underdog
reproduced the 557 legacy rows the poisoned file had buried.

The 52 legacy caches that remained after those rebuilds are now migrated, in one
`--all-cached --legacy-only` pass of 2 h 55 m. `matrix_audit --root …/training_data` goes from
**57 failing to 5**, and every survivor is a `*_corr` feature matrix — no book columns at all, so
it violates the same check it violated before. All 93 book matrices carry a complete block, none
is partial, and row counts are **identical** to the pre-migration backup on all 98 files: the
migration adds columns, never rows. No tripwire is left armed for NBA/NFL/NHL to hit when their
seasons open.

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
the population the old `_resolve_row` could not reproduce, and the reason the repair had to call the
real resolver rather than approximate it.

**A DFS pick'em line is a real quote, and the repair now reads it as one.** Re-resolving the legacy
caches surfaced NFL `fantasy points underdog` and `fantasy points prizepicks` losing *all* their
book evidence — `authentic 3378 → 0` and `4045 → 0`. The cause is an era boundary, not a policy
error: `Underdog`/`Sleeper` odds rows begin only 2026-03-16, so every 2022–2024 pick'em entry
archived a `lines` row and no `odds` row at all, and `resolve_training_quote` correctly saw no
price. But an unboosted pick'em entry prices at exactly 50/50 — the operator's rake sits in the
payout multiplier, not the line — so that bare line *is* the platform's own two-sided quote.
`training_quotes.pickem_quote` stands it back up, and `get_training_quote_inputs` consults it only
via `grouped[entity] or pickem_quote(...)`, so any real quote (boosted or not) outranks it; both
cells now resolve `book_direct` at their full historical counts, with `qb tds` and `receiving
yards` unmoved as controls.

Two boundaries are load-bearing. Scope is the `fantasy points <platform>` family alone, because
that is the only place the archive can prove a bare line is a DFS line — `lines` has no book
column and both `add_dfs` and `merge_player_books` write to it, so an unpriced sportsbook line
(receiving yards' 1561 rows) is genuinely unattributable and stays synthetic. And the stand-in must
never join a live cohort: WNBA `fantasy points prizepicks` holds 10,467 real priced Underdog rows,
and averaging a fabricated 0.5 into a real boosted price would corrupt it. Note the assumption this
rests on — "no priced row" means "old era", not "no boost". Historical boosts are unrecoverable
(`lines` has no boost column; `_dedup_offers_by_boost` selects nearest-neutral then discards the
factor), so 0.5 is the best available reading, not an inference from absence. Live capture needs
nothing: `Archive.add_dfs` already runs a boost through `no_vig_odds` into a real skewed
`under_prob`. **Watch the ship gates on these two cells** — `p_book ≡ 0.5` makes `brier_book`
exactly 0.25, so Gate 1 clears easily, the degenerate-book shape from the NFL passing cells. The
difference is that there 0.5 was a measurement artifact hiding a real price; here it is the price.

**Adjacent smell, not yet a bug report.** `M.drop_duplicates(subset=["Player", "Date"],
keep="last")` means any `(Player, Date)` the fetch re-emits silently replaces its cached row with
a freshly built one, and nothing checks that the two agree.

---

## B. Calibration gates degrade when a corner moves to full HPO — root-caused

Full HPO regressing a gate reads backwards — more search on the same recipe ought to dominate a
fixed-HP fit. **Status: root-caused; walk and board fixes landed.** Evidence: the nominee ledger
(`research/confirm_nominee_gates.csv`, 37 full-HPO confirms, every one board-6/6; 14 shipped =
38%) joins 37/37 to the crossfit board on `(league, market, strategy_slug, controls_json)`, plus
the two single-axis experiments below. Confirm failure gates: g4 ×19, g1 ×8, g6 ×8, g2/g3 ×3,
g5 ×2. Four mechanisms stack:

- **M1 — SkewNormal fit divergence under full HPO.** Six ledger rows sit on the
  `dispersion_cal` 0.1 floor with `shape_ratio` up to 2.7e9 (NFL qb tds ×3, qb yards ×2,
  NBA TOV); 6/6 fail, no ship has one. The 30-round deterministic fit cannot diverge, so the
  board is structurally blind to convergence risk. Family ship rates: DPO+NegBin 6/8,
  SkewNormal 8/23, ZINB 0/6.
- **M2 — objective/gate misalignment.** The search minimizes in-train CV CRPS/NLL with no
  calibration term while g2/g4/g5 score calibration on a disjoint frame; the calibration
  search-gate is opt-in (`hpo_selection: "calibrated"`, 11 cells). Excluding diverged rows,
  confirm−board g4 averages +0.012; failures' confirm/board g4 ratio median 1.84 vs ships'
  0.92. The g4-only calibrated retry rescued 4 of its 5 firings this run.
- **M3 — protocol optimism, the dominant near-miss driver (measured in 0a below).** The board
  scores the cross-fit validation frame with per-fold out-of-fold calibrators; confirm scores
  the true holdout with val-fit calibrators. Pre-fix the walk added its own noise: matrix drift
  between nominees from `--force` rewrites, the warm/cold budget lottery, g6 anchor seeding,
  skipped book-weight refits.
- **M4 — winner's curse.** Nominees are the top-3-of-≤48 corners by raw point-estimate slack;
  slack ≥ +0.19 shipped 1/9 while +0.05–0.14 shipped 10/17. Knife-edges flip on scoring noise
  (NFL completions failed at g4 0.0695 against a 0.0695 bar).

**Protocol axis measured (experiment 0a).** Same corner, same fixed HP, same matrix — only the
eval protocol flips (`--holdout-blind` on/off):

| cell | crossfit g4 | true-holdout g4 | holdout verdict |
|---|---|---|---|
| NHL goalsAgainst (SN) | 0.0450 (ship, slack +0.10) | 0.0704, and g1 flips to fail | no-ship, slack −0.41 |
| NFL receiving yards (hurdle) | 0.0302 | 0.0402 | still ships |
| NBA FGA (SN) | 0.0302 | 0.0470 | still ships |

The frame+calibrator protocol alone moves g4 by +0.010–0.025 — the size of a typical pass
margin — and on the near-miss cell reproduces the real confirm verdict at fixed HP. The
crossfit arms bit-reproduced their board rows except goalsAgainst, whose earlier confirm had
`--force`-rewritten the matrix — live proof of the drift defect fixed below.

**HPO axis measured (experiment 0b): null on the near-miss.** NHL goalsAgainst, frozen matrix,
true holdout on both arms: full 300-trial cold HPO lands g4 0.0695 / g1 ci_hi 0.0043 vs the
fixed-HP 0.0704 / 0.0056. Optuna neither causes nor cures the near-miss failure; on
non-diverged cells the board→confirm gap is protocol, not search. (Caveat: the two arms'
scored splits share 768 rows but differ by 17/1 from a filter quirk; direction unaffected.)

**Answers to the operator's questions.** (1) Do not freeze the deterministic HPs — they are an
underfit; full HPO wins the discrimination gates and, once the frame is held fixed, ties the
calibration gates. (2) The board was unrepresentative in the enumerated M3 ways; the fixable
ones are fixed, and the inherent frame gap is now priced into ranking as an empirical discount
instead of trusted. (3) Divergence (M1) is the one genuinely HPO-specific failure class and
routes to the research brief (σ-head guard inside the search).

**Landed fixes.** Confirm walk (`model_strategy/confirm.py`): the cell's matrix is pinned once
per walk (`_pin_cell_matrix` → `--frozen-matrix-dir` with manifest), which also buys cold-start
parity and book-weight parity per nominee; `dispersion_cal` on the 0.1 floor reports as
`diverged`, distinct from gate names; integer-target cells whose top-3 is all continuous get
the best count-family corner interleaved at slot 2; every ledger row echoes its board-side
gates as `board_*` columns. Board (`model_strategy/sweep.py`): nominees rank on
`discounted_slack` — ledger-derived per-gate confirm−board medians, echo columns preferred with
a join fallback, inert until ≥8 usable rows (note: once the first echo-format row lands, the
join path stops and discounts stay inert until 8 echo rows accumulate); a `confirm_risk`
column flags continuous-family-on-integer-target corners and g4 margins inside the measured
inflation; the rollup prints P(ship | slack band × family) from the ledger.

M2 detail — the measured same-corner example (board fixed-HP cross-fit vs its own full-HPO
confirm), NHL goalie fantasy points underdog:

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

**M2 mechanism: the HPO objective does not include calibration.** Every base corner selects
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

**Why evidence was thin at first.** A reverted confirm leaves no production trace:
`_confirm_one`'s `finally` prunes the pickle, `report()` rebuilds `model_stats.parquet` from
pickles on disk, and `_restore_cell` puts the incumbent back wholesale — deliberately, since
that parquet sizes Kelly stakes and drives Gate-2 promotion. The additive fix landed: the walk
appends every nominee's full gate row to `research/confirm_nominee_gates.csv` before the
revert, which is the ledger all the §B numbers come from.

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

The board→confirm step moves the HP budget *and* the evaluation frame together; the 0a/0b
tables at the top of this section separate the two axes, and both single-axis experiments are
done (protocol large, HPO null on the near-miss, HP-selection null on the shipping corner
below).

**The confirm walk did not hold its own frame fixed either, and new data is not the reason.**
Because every nominee retrains under `--force`, the matrix moves *between* nominees of the same
cell: MLB pitcher fantasy points underdog scored its three at `n_validation` 373, 363, 358, and NHL
goalsAgainst at 815, 801, 785. The obvious explanation — an in-season append — is wrong: NHL's last
gameday is 2026-06-14 and nothing was appended, yet the parquet's mtime lands mid-walk. The cause is
`_step_persist_matrix_and_comps`, which recomputes comps and rewrites the file on every `--force`
run regardless of whether a single row is new, and comp features are not stable across recomputes
(`_nonmlb_comp_features` fires twice per gameday and the matrix keeps the *last* value, not the
mean). So a cell's nominees were ranked against each other on drifting frames even in the
offseason. Fixed: `_pin_cell_matrix` freezes the cell's parquet once at the top of the walk and
every nominee trains against the pin via `--frozen-matrix-dir`.

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
(37 nominees) a saturated dispersion calibrator separates outcomes perfectly: every row
whose `dispersion_cal` sits on its 0.1 floor fails — 6 of 6 — and no ship has one. Those rows carry
absurd `shape_ratio` values (NFL qb yards 1.7e9 and 2.7e9, qb tds 4.6e3–1.5e6, NBA TOV 48), so the
SkewNormal fit blew up and the calibrator clamped trying to rein it in. That is a different failure
from a corner that merely calibrates badly, and it costs a full-HPO retrain to discover.

Read the *floor*, not the shape ratio: `shape_ratio > 10` does not discriminate on its own, because
NFL attempts ships at 76.6 with a healthy `dispersion_cal` of 1.01. The conjunction is what matters.

**The divergence is SkewNormal-only, and it is costing the board's budget on count cells.** All six
floored rows are SkewNormal, on three cells (NFL qb tds, NFL qb yards, NBA TOV). The count families
never come close: DPO / NegBin / ZINB top out at `shape_ratio` 1.31 and every DPO
row sits at `dispersion_cal` 0.83–0.93. Ships by family are DPO+NegBin 6/8, SkewNormal 8/23,
ZINB 0/6.

Two of the three diverging cells are *count* stats (qb tds, TOV) whose top-slack board corner is
nonetheless SkewNormal — so ranking nominees by board slack spends the cell's full-HPO budget on the
family that will blow up while a fit-stable count corner sits further down the list. That is the same
board-optimism failure as `crossfit_board_ships_optimistic`, one layer down: the board is not just
optimistic about the *gate*, it is optimistic about the *fit converging at all*. The guard landed:
`_count_class_backup` interleaves the cell's best count-family corner at slot 2 whenever an
integer-target cell's top nominees are all continuous, and diverged nominees are named as such.

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

That refutation was corner-specific (a cell that ships under both arms), but experiment 0b then
closed the general question on a corner that actually fails: on NHL goalsAgainst the full-HPO and
fixed-HP arms land the same g4 on the same holdout, so neither HP budget nor trial selection is
the near-miss lever. The remaining HPO-specific class — SkewNormal divergence — is now guarded:
per the research brief (`research/briefs/researcher_hpo_objective_alignment.md`, R1),
`_skewnormal_dist_obj` clamps the σ-like head to `[0.02, 10] × IQR(y)/1.349` and the direct
alpha head to ±30, riding the pickle into refit, warm cron, and inference. The brief's
Experiment A (5 cells × 2 arms, frozen matrices, full HPO): unclamped NFL qb tds / qb yards
diverge (corrected σ-ratio 216 / 2082, g4 0.37 / 0.41); clamped they score honestly (σ-ratio
1.00 / 1.72) — qb yards **ships** (g4 0.0637, book-less caveat) and qb tds lands 0.0009 past its
bar. Controls (NBA FGA, NFL attempts) show no systematic degradation (σ-ratio ~1.0 both arms;
gate deltas inside the ±0.01 run-to-run HPO spread — the brief's <0.002 inertness bar is
unmeasurable without seed-matched trials; FGA's ship flip at 0.0537 vs bar 0.05 sits inside its
own historical 0.0416–0.0537 spread). The corrected `shape_ratio` (σ/σ after the `_diag_shape`
unit fix) separates diverged from converged with an empty band and is the board-side pre-screen.

The brief's other two levers resolved against build-out: Experiment B (headroom probe, 3 cells,
`hpo_selection=calibrated`, frozen matrices) **killed the default flip** — the calibration
constraint is non-binding on healthy cells (goalie fantasy 146/147 trials feasible, so calibrated
≡ loss, which explains the earlier null refutation), near-vacuous on the FGA near-miss (1/54
feasible at 0.0499 vs bar 0.05), and non-transferring on goalsAgainst (5/210 feasible, 0.038
CRPS price, holdout g4 0.0736 vs the loss arm's 0.0695 on identical rows). `hpo_selection:
"calibrated"` stays opt-in per-cell. The g4-only retry was widened by margin instead (g4 excess
≤ 0.010, sole or with g1 within 0.005+0.002; g6 excluded pending an Experiment-C measurement) —
on the current ledger that adds zero firings (the brief's "5→6" used off-snapshot numbers) and
exists as forward armor with a bounded cost.

**Is full HPO overfitting instead of the deterministic arm underfitting?** No — measured, not
argued. Three paired arms (deterministic 30-round/31-leaf vs full HPO) on **sha-verified identical
matrices**, same true holdout, same code: full HPO never loses a calibration gate, winning g4 once
and tying twice. It wins the disagreement outright — NFL qb yards' deterministic arm fails at
g4 0.0847 against its 0.0789 bar while full HPO returns 0.0620 and ships. Deterministic's only
wins are g2/g3 by 0.02–0.07 on cells that ship either way. Two supporting checks: within a cell,
the recipes that shipped carried *more* capacity than the ones that failed (the aggregate
capacity–failure correlation is a Simpson's reversal), and the confirm−board g4 gap sits inside
the band the frame change alone explains. The pre-registered trigger for a selection-guard lever
(HPO losing a calibration gate beyond noise on ≥3 pairs) is not met at zero. The other five
det-pair cells are live, so their HPO arms need a supersession pass to take this to n=8.

**Nominee policy: only board-confident corners walk.** `_nominees` now requires a positive rank
value (`discounted_slack`, else `slack`) per corner and skips a cell whose best admissible corner
is non-positive; `_candidates` orders cells by that rank so a `--confirm-hours` deadline cuts the
weakest tail. This trades the rare full-HPO rescue of a board-negative recipe (NBA FTM shipped one
at board −1.86) for wall clock. The first batch under it walked 12 withheld cells and **shipped 7**:

| board slack band | cells | shipped |
|---|---|---|
| ≥ +0.07 | 7 | 7 |
| ≤ +0.05 | 5 | 0 |

Board rank now predicts the confirm verdict cleanly, which is the parity claim above holding at
batch scale. Per-gate confirm−board deltas over the batch's 18 board-sourced nominee rows: g1
median **+0.0008** (IQR ±0.002), g5 +0.008, g2/g3 near zero with spread — and g4 **+0.0158**
(IQR +0.008…+0.037), the same frame offset §B measures at +0.010–0.025. That residual is the
live limit: `discounted_slack` prices it into *ranking* but cannot move a *gate threshold*, so a
cell whose board g4 sits within ~0.016 of its bar is still a coin flip at confirm — which is where
every remaining miss landed (NFL interceptions, receiving yards ×3, qb yards' first nominee). The
untried lever is a per-gate board-side handicap at nomination (require board g4 + 0.016 to clear
the bar), which would have skipped those walks.

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

## D. A confirm's full-HPO cross-validation aborts inside LightGBM

The same `REVERTED … failed retrain error` symptom as §C, different cause and **fixed**: NFL carries
lost both nominees to a glibc heap-corruption abort in `lgb.cv`, which builds each fold's Booster
over a `Dataset.subset`. Upstream defect, present in LightGBM 4.6 and 4.7, nothing to do with the
DPO family. The walk now rebuilds the folds through cv's own `fpreproc` hook and reports a signal
death as `native abort (SIGABRT)`. Evidence, exclusions, and the reproduction recipe:
[dpo-confirm-crash.md](dpo-confirm-crash.md).
