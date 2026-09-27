# Lane brief — NFL back to 15+ shipped cells, yards family mandatory

**Read first:** CLAUDE.md, [CONTRIBUTING.md](../../CONTRIBUTING.md),
[docs/ARCHITECTURE.md](../ARCHITECTURE.md), [model_improvement_track.md](model_improvement_track.md)
§6–§8 and the §10 ledger, [docs/ship_gate.md](../ship_gate.md), and
[docs/fantasypoints_expansion.md](../fantasypoints_expansion.md). This brief is self-contained but
those documents are the contract it works under. Status: OPEN, written 2026-09-26 on the honest board of that evening.

## 1. Mandate

Bring the NFL board back to **at least 15 of 20 cells with `ship == True`** on
`data/training/model_stats.parquet`, scored honestly on the rebuilt training population with the
Tier 1 FantasyPoints feature set kept. Five cells are **mandatory members** of the shipped set:

| cell | what it is | why mandatory |
|---|---|---|
| `passing yards` | QB single stat | owner's named market |
| `rushing yards` | RB/QB single stat | owner's named market |
| `receiving yards` | WR/RB/TE single stat | owner's named market |
| `yards` | rush + rec combo (`combo_props`) | owner's named combo |
| `qb yards` | pass + rush combo (`combo_props`) | owner's named combo |

Constraints that define "honestly":

- The six offline gates and their thresholds are inviolable (`ship_gate.md`). No threshold
  reinterpretation, no positional splits, no selection against the retained holdout.
- The 47 new FantasyPoints recipe outputs stay in every cell. They are not the problem: 0 of 47
  were inert by |SHAP| on the 2026-09-25 retrain and 6 of the top 20 new-feature SHAP scores
  landed on passing yards or qb yards.
- Every verdict comes from the production pipeline (`meditate` → `model_stats.parquet`), never
  from a deterministic dump. Deterministic scorecards rank; full-HPO confirms (memory
  `deterministic_ab_g4_oversell`).
- Nothing is pushed from the lane, nothing is synced to prod, and no `stat_meta.json` change is
  committed without the owner reading the diff. The confirm walk writes `stat_meta.json` locally
  as its mechanism; that is allowed, the commit is not.

## 2. What happened before this brief (so you do not redo it)

Three boards exist for NFL in the last ten days. Only the last one is evidence.

1. **2026-09-19 baseline, 11/20.** Pre-bump models on the incremental caches. Those caches
   were a patchwork: `get_depth` inner-merged every historical gameday with the *build-day*
   roster, so a player who had left the league vanished from every past gameday
   (memory `nfl_rebuild_player_membership_shift`; fixed in devel `6eac660c`).
2. **2026-09-25 retrain, 8/20 — not honest.** All 20 matrices rebuilt cold on the fixed
   population with the 47 new features. Passing first downs, passing yards and sacks taken lost
   `ship`. Root cause found afterwards: `parse_pbp` resolved gamelog names through the
   roster-filtered id map, so every departed player carried all-zero play-by-play stats on all
   past games — 30.6% of gamelog rows, 13–28% of matrix rows, and the *targets* of passing
   first downs and sacks taken (memory `nfl_pbp_stats_roster_gated_ids`; fixed in `4c8b4a91`).
   Its artifacts are kept under `~/backups/sportstradamus/2026-09-19-nfl-fp-bump/` with the
   `_v5_zeroedpbp` tag (`report_v5_zeroedpbp.md`).
3. **2026-09-26 honest retrain.** Gamelog recomputed (10,209 zeroed rows → 75 residual, all
   nickname mismatches against `import_ids`), matrices rebuilt again, production retrain. Its
   report is `~/backups/sportstradamus/2026-09-19-nfl-fp-bump/report.md`. Section 3 below is
   read from it.

Two serve-path fixes rode along and matter for anything you measure live: `NEP` → `NE` roster
code (`abb82025`; an unmapped code made `_rescale_team_volume` raise and dropped every NFL market
for the run) and the within-team `Player depth` rank at serve time. Verify the serve path with
`~/backups/sportstradamus/2026-09-19-nfl-fp-bump/serve_parity_probe.py` after any retrain — the
fake-mode integration suite no-ops `Stats` I/O and cannot see these breaks (memory
`nfl_serve_path_parity_probe`).

## 3. Where the numbers stand (honest board, 2026-09-26)

**11 of 20 ship** (baseline 11, the zeroed run 8). Mean BSS over the scored cells rose
0.0515 → 0.0577; 0 of the 47 new columns are inert by |SHAP|; every production scorecard reads
HOLD (S2/S3 cannot separate two models this close, in either direction). Two cells came in
(fantasy points prizepicks, receptions), two went out (passing yards, sacks taken), passing
first downs is back after training on real targets. Two more are knife-edges: sacks taken misses
Gate 4 by 0.0002 and completions misses Gate 1 by 0.0001. Full report with the deterministic
A/B tables and the SHAP list: `~/backups/sportstradamus/2026-09-19-nfl-fp-bump/report.md`;
the board itself is `data/training/model_stats.parquet` (NFL rows, 2026-09-26 20:33).

| cell | 09-19 baseline | 09-25 zeroed | **09-26 honest** | BSS | w | n (authentic) | g4 / cap | binding detail |
|---|---|---|---|---|---|---|---|---|
| attempts | SHIP | SHIP | **SHIP** | +0.016 | 0.13 | 430 (384) | 0.038 / 0.066 |  |
| carries | SHIP | SHIP | **SHIP** | +0.037 | 0.85 | 1280 (1017) | 0.033 / 0.050 |  |
| completions | KILL g4 | KILL g1 | **KILL g1** | +0.007 | 0.28 | 433 (387) | 0.056 / 0.065 | g1 ci_hi 0.0051 vs 0.005; already hpo_selection=calibrated |
| fantasy points prizepicks | KILL g2 | KILL g2 | **SHIP** | — | 1.00 | 2144 (0) | 0.036 / 0.050 | g2 star z 0.50 (was failing); book-less |
| fantasy points underdog | KILL g1 | KILL g1 | **KILL g1** | +0.039 | 0.74 | 2132 (131) | 0.018 / 0.050 | g1 ci_hi 0.056 on 131 authentic rows |
| interceptions | SHIP | SHIP | **SHIP** | +0.267 | 0.90 | 446 (383) | 0.035 / 0.064 |  |
| passing first downs | SHIP | KILL g2 g4 g6 | **SHIP** | — | 1.00 | 401 (0) | 0.055 / 0.068 | target was zeroed in the 09-25 run; ships again |
| passing tds | SHIP | SHIP | **SHIP** | +0.215 | 0.90 | 433 (372) | 0.054 / 0.065 |  |
| passing yards | SHIP | KILL g4 | **KILL g4** | -0.002 | 0.05 | 438 (388) | 0.070 / 0.065 | g4 only, 0.0056 over; g1 ci_hi 0.0010; cov50/80 0.50/0.81 |
| qb tds | KILL g4 | KILL g5 | **KILL g5** | — | 1.00 | 448 (0) | 0.044 / 0.064 | g5 ECE 0.083 vs cap; book-less |
| qb yards | KILL g1 | KILL g1 g4 | **KILL g1 g4** | +0.085 | 0.90 | 391 (29) | 0.081 / 0.069 | g1 mean +0.043, ci_hi 0.077 on 29 rows / 23 clusters; g4 0.0124 over |
| receiving tds | SHIP | SHIP | **SHIP** | — | 1.00 | 3039 (0) | 0.035 / 0.050 |  |
| receiving yards | KILL g4 | KILL g4 | **KILL g4** | +0.022 | 0.38 | 2379 (2304) | 0.058 / 0.050 | g4 only, 0.0076 over; tail KS = whole KS (over-tail); g1 ci_hi 0.0012 |
| receptions | KILL g1 | KILL g1 | **SHIP** | +0.003 | 0.62 | 2576 (2451) | 0.024 / 0.050 | g1 ci_hi 0.0027 (was 0.0063) |
| rushing tds | SHIP | SHIP | **SHIP** | — | 1.00 | 1278 (0) | 0.036 / 0.050 |  |
| rushing yards | SHIP | SHIP | **SHIP** | +0.005 | 0.48 | 1326 (1104) | 0.043 / 0.050 | g1 ci_hi 0.0011; g4 tail 0.043 |
| sacks taken | SHIP | KILL g4 g5 g6 | **KILL g4** | — | 1.00 | 434 (0) | 0.065 / 0.065 | g4 0.0002 over the cap; book-less (g1 blank) |
| targets | KILL g1 | KILL g1 | **KILL g1** | -0.012 | 0.51 | 2176 (131) | 0.020 / 0.050 | g1 ci_hi 0.032 on 131 authentic rows / 101 clusters |
| tds | SHIP | SHIP | **SHIP** | +0.167 | 0.90 | 3298 (3207) | 0.021 / 0.050 |  |
| yards | KILL g1 g4 | KILL g1 | **KILL g1** | -0.042 | 0.50 | 2162 (73) | 0.047 / 0.050 | g1 mean −0.009 but ci_hi 0.027 on 73 rows / 59 clusters; g4 passes |

Two cohort facts to keep in view. Book-less cells (`n authentic = 0`: the fantasy cells, the
first-down/sack/td cells) auto-pass Gate 1 and carry `w = 1.0`; their ship is Gates 2–6. Four
cells grade Gate 1 on a **sliver of authentic quotes** — `yards` 73 rows, `qb yards` 29, `targets`
131, `fantasy points underdog` 131 — so their Gate 1 verdicts move with every retrain and say
little either way (section 5, the combos).

## 4. How a cell ships — the machinery you will drive

**Gates.** `ship = g1 ∧ … ∧ g6` on the cell's test-set CSV. Gate 4 is the whole-CDF PIT-KS
against `max(0.05, 1.358/√n)`: with n ≈ 436 (passing yards) the cap is 0.065, n ≈ 391 (qb
yards) 0.069, and any n above about 740 sits at the 0.05 floor (receiving yards, rushing
yards, yards). Gate 1 is
the paired Brier difference against the book on **authentic** quotes only
(`QuoteAuthenticity == "authentic"`); a cell with no authentic rows, or fewer than the cluster
floor, has a blank Gate 1 that auto-passes (no book to beat) and carries `model_weight = 1.0`.

**Weekly `meditate`** trains each active cell exactly once from its `stat_meta.json` recipe,
computes the gates, WARNS on served cells failing them (`SHIP-GATE WARNINGS`), and never writes
`stat_meta` (memory `meditate_sweep_role_separation`). Serving is pickle-exists; `ship=False`
cells keep serving until a human culls them (memory `project_serve_iff_ship`). So the 8/20 or
whatever the honest count is describes the *gate* state, not what prod serves.

**`sportstradamus ship sweep`** is the search. One conditional-TPE study per cell over the
strategy catalog (`training/model_strategy/specs.py`): for SkewNormal cells the axes are
`target_normalization` ∈ {ratio_meanyr, centered_additive_mean10,
centered_additive_eb_meanyr_k10, ratio_projvol}, `dist_training_loss` ∈ {crps, nll},
`sn_param` ∈ {direct, centered}, `blending` ∈ {crps, crps_1se, nll} and `posthoc` ∈ {none,
prob_recal_isotonic, prob_recal_platt, prob_recal_book_citl, roe_mean, isotonic_mean,
cdf_recal_isotonic}; count families carry their own axes. Every corner trains holdout-blind and
is scored on a cross-fit gate row, so the board's `ships` flag is optimistic (memory
`crossfit_board_ships_optimistic`) and its `slack`/`discounted_slack` is the ranking signal.
Board rows are matrix-scoped (`matrix_hash` column): the 634 NFL rows already in
`data/research/strategy_research_board.csv` were earned on the old caches and are inadmissible
for the rebuilt matrices, so budget every NFL cell as a cold sweep (`--max-trials` default 48
trained corners; `-j` parallelizes cells, never corners).

**`--confirm`** turns a board into shipped cells. Two lanes, decided by the cell's current
`shipped` value:

- *Fresh lane* (`shipped: "withheld"`): nominees walk in slack order; each is persisted to
  `stat_meta.json`, retrained with **full HPO**, and read back from `model_stats.parquet`.
  A clean 6/6 ships the cell (`shipped: "devel"`); a failure auto-reverts stat_meta and pickle.
  Nominee policy: only positive-slack corners nominate; 7/7 shipped at ≥ +0.07 slack, 0/5 at
  ≤ +0.05 on the first campaign (memory `confirm_board_confident_nominee_policy`).
- *Live lane* (`shipped: "devel"`, needs `--include-shipped`): supersession test against the
  incumbent — S1 (candidate passes all six) **and** S2 (paired Brier CI excludes 0 in the
  candidate's favour) **and** S3 (paired Sharpe z above the bar). Against an incumbent that
  itself fails the gates this is nearly unpassable on purpose: S2/S3 measure "better than", not
  "good enough". The one waiver is `--min-model-weight T`: a candidate at or above `T` may
  replace a book-riding incumbent below `T` on S1 alone (`ship_gate.md`, "supersede an
  incumbent"). Passing yards' incumbent rides the book (`model_weight` 0.05), so this waiver is
  available there and nowhere else in the yards family.

Consequence for this lane: **a served cell that fails the gates is best recovered through the
fresh lane**, which means flipping it to `shipped: "withheld"` first. That is a `stat_meta`
edit and therefore the owner's (section 6, decision 1). It also darkens the cell on the next
`meditate` until it ships again — which is what serve-iff-ship says should happen anyway.

**Automatic g4 rescue.** When a nominee fails ship on Gate 4 alone and sits within 0.010 of its
cap (`_G4_RETRY_MAX_EXCESS`), the walk pins `hpo_selection: "calibrated"` and reruns the
identical `meditate` once (memory `calibration_hp_selection_lever`: CV-loss objective with an
OOF PIT-KS feasibility constraint). Passing yards and receiving yards do not carry the pin yet;
yards, qb yards, receptions and completions already do, so the retry cannot help them further.

**Costs measured on this box (2026-09-25/26).** Full-HPO `meditate` over all 20 NFL cells:
1 h 58 (~6 min per cell). Cold matrix rebuild: c2 34 min, c3 2 h 35, c5 6 h 34 (~24 min per
dependent cell). Confirm timeout per nominee: 4 h.

**Single-axis experiments without touching production:** `meditate --league NFL --market M
--frozen-matrix-dir DIR --artifact-output OUT [axis flags]` trains one cell on a pinned matrix
and writes the pickle, test set and gate row under `OUT` (memory
`frozen_matrix_single_axis_experiments`); `--deterministic` dumps fixed-HP test sets for
`ship scorecard --baseline A --candidate B` comparisons (direction only). The lane dir has the
harness that ran the last two A/Bs: `run_det_arm.sh`, `compare_det.sh`, `compare_det_common.sh`,
`parse_det_ab.py`.

**If a lever changes the feature set** (a new recipe output, a dropped column, a normalization
that needs a new matrix column) every NFL matrix must be rebuilt cold again: the caches are
incremental, and a column appended to a cache is NaN on every historical row (§7.2 of the track).
The recipe is `run_phase_c.sh c2 → c3 → c5` in the lane dir (volume matrices → `volume-v1`
registry bootstrap → 17 dependents with `--dependency-namespace volume-v1 --dependency-root
src/sportstradamus/data/model_dependencies`), then `quarantine_sanity_v2.py` (identity coverage,
trim-floor carve-outs, `pbp_dead_share` hard check) and a sha-verified copy into
`training_data/`. `run_phase_c_chain_v5.sh` runs the whole thing unattended with STOP/DONE
markers; expect ~10 h to the sanity gate. The gate's pbp check compares all-zero-probe shares on the
(Player, Date) keys both matrices share; the whole-population share is a report column because the
fixed population legitimately carries more zero rows (2021 charting data does not exist). Memory `nfl_cold_matrix_rebuild_blocked` holds the
history of why the plain cold start crashes.

## 5. Per-cell diagnosis and lever queue — the mandatory five

Read each cell's honest gate row before choosing; the queue below is ordered by cost and by
what the gate signature says, not by hope. Levers marked **research-gated** need a
`research-analyst` brief first (§8.2 of the track; the hook enforces distribution-family
edits, this brief covers the judgement calls).

### passing yards

Recipe: SkewNormal, `centered_additive_eb_meanyr_k10`, nll, direct, blending `crps_1se`,
posthoc `prob_recal_book_citl` (the 2026-09-01 ship: the served probability leaned over the
book and the book-anchored intercept fixed Gate 1 — memory `g1_citl_lean_book_anchored_posthoc`).
`model_weight` 0.05: the fused leg rides the book, so Gate 1 is a tie by construction and the
cell's fate is Gate 4. History: baseline g4 0.0625 under a 0.070 cap; zeroed run 0.0752 under
0.065. Honest board: **KILL g4** — 0.0705 against 0.0649 (0.0056 over, inside the
calibrated-retry window), over-tail KS 0.035, Gate 1 passes (ci_hi 0.0010), Gate 5 0.020,
coverage 0.50/0.81 (nominal: shape-bound, not scale-bound), BSS −0.002, w 0.05.

1. If g4 is the only failure and within 0.010 of the cap, the confirm walk's calibrated retry
   fires by itself — let the fresh lane run before touching anything.
2. `sn_param: centered` (the shape-bound rung; NBA PTS and WNBA MIN ship with it).
3. `cdf_recal_isotonic` — the one corrector Gate 4 rewards (memory
   `slack_blind_to_g4_mechanism`). It occupies the same `posthoc` slot as
   `prob_recal_book_citl` (memory `posthoc_slot_two_jobs_bind`); at `model_weight` 0.05 the
   book carries the level, so losing the CITL intercept may cost nothing — the sweep will say.
4. Normalization: `ratio_meanyr` was the pre-2026-09 recipe; the sweep tries all four.
5. Do **not** chase Gate 1 here: the track's passing-yards brief
   (`nfl-passing-yards-gate1-recovery.md`) documents why every outcome-fitted corrector learns
   the fold's over-rate.

### rushing yards

Recipe: SkewNormal, `ratio_meanyr`, crps, centered, blending crps, posthoc
`cdf_recal_isotonic`. Shipped on both prior boards (g4 0.047 / 0.042 under 0.05,
BSS −0.008 → +0.002, weight 0.46 → 0.39). Honest board: **SHIP** — g4 0.0429 / 0.050, Gate 1 ci_hi 0.0011, BSS +0.005, w 0.48, n 1,326
(1,104 authentic). The risk is a g4 coin-flip
near the cap on future retrains; if it ever fails g4 alone the calibrated retry covers it.
Otherwise leave it alone and do not sweep it under `--include-shipped` — S2/S3 against a
passing incumbent costs confirms and rarely wins.

### receiving yards

Recipe: SkewNormal, `ratio_meanyr`, crps, centered, blending crps, posthoc
`prob_recal_isotonic`. Gate 4 has failed on every board (0.057 / 0.059 under 0.05) with
central coverage slightly under nominal (cov50 0.48–0.49, cov80 0.77–0.79: mildly
under-dispersed, i.e. scale-bound rather than shape-bound), while BSS improved with the bump
(.016 → .021) and the weight doubled (0.17 → 0.36). Honest board: **KILL g4** — 0.0576 against 0.050 (0.0076 over,
inside the retry window); the over-tail KS equals the whole KS, so the entire deviation is
alt-over mispricing; Gate 1 passes (ci_hi 0.0012), Gate 5 0.005, coverage 0.48/0.79, BSS +0.022,
w 0.38, n 2,379 (2,304 authentic).

1. Calibrated retry (within 0.010 of the cap on both prior boards) — automatic in the walk.
2. `cdf_recal_isotonic` in the posthoc slot (replacing `prob_recal_isotonic`; the cell's Gate 1
   passes without help, so the slot is free for the g4 job).
3. `centered_additive_mean10` normalization — the transform that flipped `carries` from
   below-book to beats-the-book at real HPO (memory `nfl_volume_cells_feature_mature`: the
   volume-family deficit is target shape and capacity, not features).
4. Training population: this cell hits the trim floor (`trim_matrix(M, 15000)` in
   `pipeline._step_persist_matrix`); the rebuild ended near 15,900 rows with about half its
   unquoted rows balanced away while all 9,792 quoted keys survived. Raising the floor is a
   global owner knob (section 6, decision 2) and needs a rebuild of the affected matrices.

### yards and qb yards (the combos)

Recipes: both SkewNormal `centered_additive_eb_meanyr_k10`, crps, centered, blending nll,
`hpo_selection: calibrated`; posthoc none (`yards`) / `isotonic_mean` (`qb yards`).

The binding fact is the **quote history**, not the model. The archive stopped receiving the
combo lines after the 2023 season while the components stayed healthy (memory
`nfl_combo_archive_gaps`); Underdog's `rush_rec_yds` and `passing_and_rushing_yds` quotes
are archived again from September 2026 (the test sets' first authentic combo rows). The test sets therefore hold **0 authentic quotes in 2024
and 2025** and only the 2026 weeks: on the zeroed board `yards` graded Gate 1 on 73 authentic
rows (59 clusters) with `model_weight` 0.52 and BSS −0.05, `qb yards` on 29 rows (23
clusters) at weight 0.90 and BSS −0.02. Both sit just above the 10-cluster floor, so Gate 1 is
live, fails, and will keep moving with every retrain as a few dozen authentic rows a week
arrive for `yards` and about ten for `qb yards`. Honest board: `yards` **KILL g1** on 73 authentic rows (59 clusters) — the paired
Brier mean is −0.009 *in the model's favour* but the CI high is 0.027 against a 0.005 bar: the
sample fails it, not the sign; Gate 4 now passes (0.047 / 0.050); BSS −0.042; w 0.50. `qb yards`
**KILL g1 g4** on 29 rows (23 clusters): mean +0.043, CI high 0.077, and g4 0.081 against 0.069;
BSS +0.085; w 0.90.

1. **Find authentic 2024–2025 combo quotes first.** The dev archive never sees DFS polls
   (memory `dev_archive_no_dfs_ladder_polls`); prod's `archive/archive.duckdb` runs `confer`
   and may hold `yards` / `qb yards` odds rows for those seasons. Check it read-only on prod
   (respect `run_job.sh`'s archive flock; the memory `archive_flock_self_deadlock` says never
   wrap an archive CLI in your own flock), then `sportstradamus admin merge-archives --source
   <prod copy> --target archive/archive.duckdb --dry-run` on the dev box before a real merge.
   The Odds API is the other candidate source if it offers combined-yards player markets
   (`player_rush_reception_yds`-style keys — check the market list; `stat_map.json` maps none
   of them today). The designed path is `python -m sportstradamus.scripts.backfill_historical_odds
   --league NFL --markets 'yards,qb yards' --start … --end … --max-dates 1 --dry-run` once the
   keys are mapped: the dry run makes the paid calls and prints the cross-book `ev` spread
   (zero spread = degenerate source, stop), then the real run writes game-day `01:00` rows the
   point-in-time reads prefer. Credits come out of the governor
   (memory `odds_api_budget_governor`); probe one date first (memory
   `capture_stub_drift_live_probe`).
2. With new quotes in the archive, repair the two matrices' odds block with
   `python -m sportstradamus.scripts.inject_backfilled_odds` (calls the real resolver — memory
   `inject_backfilled_odds_not_rebuild_equivalent`) or rebuild just those two cells through
   `run_phase_c.sh`-style `--full-rebuild --matrix-only --market 'yards,qb yards'` (~25 min
   each), then confirm.
3. Read DFS quotes with the known caveat: the line is a real quote, the price is pinned near
   0.5 (memory `dfs_pickem_lines_are_real_quotes`), so a Gate 1 tie against them is cheaper
   than against sportsbooks and says less.
4. If no history exists anywhere, say so and put the decision to the owner: wait for the 2026
   cohort to grow (each retrain re-rolls the verdict) or accept these two as the last to cross.
   The component sum is already unbiased for a `combo_props` spec (memory
   `combo_book_mean_blend`), so do not add a book-mean blend to these cells.

### The other fifteen

Eleven ship. Of the nine that do not, the cheapest recoveries are the two knife-edges: sacks
taken (Gate 4 by 0.0002 — book-less, so the calibrated retry is the whole lever) and
completions (Gate 1 by 0.0001 — already `calibrated`, so a sweep corner rather than a retry).
qb tds fails Gate 5 alone (ECE 0.083; a `prob_recal_*` posthoc is the matching lever, memory
`posthoc_slot_two_jobs_bind`). Targets and fantasy points underdog fail Gate 1 on 131 authentic
rows each — the same sliver problem as the combos, from DFS-only quoting. Count cells
(interceptions, passing tds, receiving tds, rushing tds, tds, carries) shipped on every board
and need nothing. Interceptions is the owner's lowest-priority cell.

## 6. Owner decisions this lane needs up front

1. **Lane choice for failing served cells.** Recommended: flip every NFL cell with
   `ship == False` on the honest board to `shipped: "withheld"` in one reviewed
   `stat_meta.json` commit, so `ship sweep --league NFL --confirm` can auto-ship them on a clean
   6/6. Alternative: keep them live and run `--include-shipped --confirm-auto-promote`, accepting
   that S2/S3 against a failing incumbent will HOLD almost every candidate (`--min-model-weight`
   helps only passing yards). Either way the agent presents the resulting `stat_meta` diff and
   the owner commits it.
2. **Trim floor.** `trim_matrix(M, 15000)` binds on the WR/RB/TE cells after the population
   fix. Raising it grows every league's training set and training time; if authorized, run it
   as a rebuild of the affected NFL cells with a frozen-matrix A/B before promotion.
3. **Combo quote history.** Whether to pull prod's archive (or Odds API history) for
   2024–2025 combo lines — the only lever that changes the combos' Gate 1 evidence base.
4. **Push and sync.** `abb82025` (`NEP` fix — prod is likely dropping every NFL market on
   Patriots slates until it lands) and `4c8b4a91` are unpushed; `scripts/sync_to_prod.sh`
   carries the models, `model_stats`, the two calibration JSONs, the corr parquets and
   `player_data/NFL` + `team_data/NFL` — **not** `leagues/nfl/gamelog.parquet`. Prod's gamelog
   therefore keeps its zeroed pbp rows (30% of history, feeding the team/defense aggregates the
   honest models were trained without) until the recompute runs there: deploy `4c8b4a91`, then
   run `~/backups/sportstradamus/2026-09-19-nfl-fp-bump/recompute_pbp_zeroed.py` on prod at a
   quiet hour (about 4 minutes; it calls `update()`, so keep it clear of the cron jobs' archive
   flock) before the honest models are synced.

## 7. Rules the lane runs under

- No `git push`, no PR creation, no `sync_to_prod.sh`, no `stat_meta.json` commit without the
  owner's review. Commit by pathspec only (`git commit -- <paths>`); never `git add -A`,
  never `git stash`; never a HEAD-moving git command inside a subagent prompt (the checkout is
  shared).
- One `meditate` at a time on the box (`pgrep -af "sportstradamus medi[t]ate|model.strategy"`
  must print nothing first — the bracket keeps the pattern from matching its own shell), and
  never the integration suite while one runs. The sweep's `-j` parallelism is the exception the
  tool manages itself.
- Never fetch a partial FantasyPoints week; 2026 week 3 becomes fetchable after its Monday
  game, `fetch fp run --season 2026 --week 3 --refetch` (memory
  `fp_column_map_applies_at_fetch_time`). Never sync a partial week.
- Any `.py` touched: `refactoring-specialist` on every touched file, then the single gate run
  `poetry run ruff check src/sportstradamus/`, `poetry run pytest tests/golden/`,
  `poetry run pytest -m integration -n0` (one pre-existing WNBA red,
  `test_centered_sn_live_path_reemits_direct_frame`, is known — report it, do not touch the
  marker).
- Distribution-family or dispersion-mechanism changes and §8.2 levers: `research-analyst`
  brief first. Genuinely new families (skew-t, SHASH revisit, comp-PMF ladder) are dead prior
  art until a brief says otherwise (memory `research_verdicts_ledger`).
- Ledger discipline: one caveman line per cell verdict in the track's §10 (cap 15, newest
  first) and a one-line CHANGELOG entry; detail stays in the lane dir and git.

## 8. Traps that have already cost a day each

| trap | memory |
|---|---|
| fake-mode integration proves nothing about the serve path; run the parity probe | `nfl_serve_path_parity_probe` |
| roster-gated id map zeroed 30% of pbp stats; the sanity gate now checks `pbp_dead_share` | `nfl_pbp_stats_roster_gated_ids` |
| build-day roster gated the training population | `nfl_rebuild_player_membership_shift` |
| deterministic HPs overstate g4 by ~0.016; read direction only | `deterministic_ab_g4_oversell` |
| board `ships` = cross-fit pass, not the production verdict | `crossfit_board_ships_optimistic` |
| slack ranks on the binding mean/skill gate; g4-only correctors are invisible to it | `slack_blind_to_g4_mechanism` |
| one `posthoc` slot, two jobs (prob stage for g1, cdf stage for g4) | `posthoc_slot_two_jobs_bind` |
| `ship=True` is stale once `strategy_matrix_hash` moves | `ledger_ship_is_matrix_scoped` |
| a `shipped: devel` flag is not landed code — check the code | `ship75_devel_carve_verify_code` |
| `--target-normalization` must be set before the confirm `meditate` reads stat_meta | `ship75_confirm_set_norm_first` |
| silent `meditate` death = native abort, no traceback; grep the log | `silent_meditate_death_faulthandler` |
| the integration suite restores runtime snapshots byte-for-byte but refreshes mtimes | `integration_clobbers_runtime_snapshots` |
| history.parquet is keyed by game date; prophecize.log is cumulative and `\r`-joined | `prophecize_prod_artifact_read_traps` |
| a walk and a second session racing on `stat_meta.json` | `git_add_races_running_walk` |

## 9. Prior art — dead, do not retry without new evidence

Already tried and refuted for the NFL SkewNormal cells or their mechanisms: SHASH / StudentT
families, the StudentT conditional-scale head, book-skew shape borrow, μ-conditional post-hoc
defaults, the 2-component Gaussian mixture as a ship route, `prob_recal_platt_cv` in the axis
pool, the calibrated-everywhere default, feature-filter rewires (production trains on the full
candidate set by design), and "anchor on the line" (the line is excluded from X on purpose).
The full list with verdict dates is memory `research_verdicts_ledger` and
`shape_bound_triage.md`'s prior-art table.

## 10. Definition of done and the report

Done means: `poetry run python -c` reading `model_stats.parquet` shows ≥ 15 NFL rows with
`ship == True` **including all five mandatory cells**, every one produced by a full-HPO
`meditate` on the promoted matrices, `serve_parity_probe.py` builds every `expected_columns`
entry for every pickle, the three gates are green (WNBA red reported), the `stat_meta.json`
diff is in front of the owner, nothing pushed. If the count stalls below 15, the report says
which cells, which gate binds each, what was tried with its ledger rows, and what the owner
could decide to unblock it — scaling the goal down is the owner's call, not the lane's.

The report (normal prose, in the lane dir and summarized in §10) carries per cell: recipe
before → after, the six gate values with caps, BSS, `model_weight`, n and authentic n, the
confirm ledger rows consulted, and the `matrix_hash` the verdict is bound to.
