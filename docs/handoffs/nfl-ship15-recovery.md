# Lane brief — NFL back to 15+ shipped cells, yards family and targets mandatory

**Read first:** CLAUDE.md, [CONTRIBUTING.md](../../CONTRIBUTING.md),
[docs/ARCHITECTURE.md](../ARCHITECTURE.md), [model_improvement_track.md](model_improvement_track.md)
§6–§8 and the §10 ledger, [docs/ship_gate.md](../ship_gate.md), and
[docs/fantasypoints_expansion.md](../fantasypoints_expansion.md). This brief is self-contained but
those documents are the contract it works under.

Status: OPEN. The confirm campaign is done (19 of 20 cells at ship, section 3) and the count goal is
met, but `passing yards` is withheld, so the definition of done (section 10) is not. What sits with
the owner is section 6, decisions 3 and 6–10: the uncommitted `stat_meta.json` flips for seven cells,
the retrain that re-rolls verdicts the 2026-10-08 game-line re-read made stale, the passing-yards
feature lane, the relabel of the other four leagues' caches, the bulk combo pull, and one serve-side
finding handed to the honest-receipts lane.

## 1. Mandate

Bring the NFL board back to **at least 15 of 20 cells with `ship == True`** on
`data/training/model_stats.parquet`, scored honestly on the rebuilt training population with the
Tier 1 FantasyPoints feature set kept. Six cells are **mandatory members** of the shipped set:

| cell | what it is | why mandatory |
|---|---|---|
| `passing yards` | QB single stat | owner's named market |
| `rushing yards` | RB/QB single stat | owner's named market |
| `receiving yards` | WR/RB/TE single stat | owner's named market |
| `yards` | rush + rec combo (`combo_props`) | owner's named combo |
| `qb yards` | pass + rush combo (`combo_props`) | owner's named combo |
| `targets` | WR/TE volume stat, the league's serve-time volume denominator (`VOLUME_STATS`) | owner's named market (added 2026-09-28) |

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
   Its artifacts are kept in the lane dir, `~/backups/sportstradamus/2026-09-19-nfl-fp-bump/`
   (`$B` below), with the `_v5_zeroedpbp` tag (`report_v5_zeroedpbp.md`).
3. **2026-09-26 honest retrain, 11/20.** Gamelog recomputed (10,209 zeroed rows → 75 residual,
   all nickname mismatches against `import_ids`), matrices rebuilt again, production retrain. Its
   report, with the three-board comparison table, is
   `~/backups/sportstradamus/2026-09-19-nfl-fp-bump/report.md`. The trim-floor retrain the next
   day took it to 12/20 (section 6, decision 2); that board is what the campaign in section 3
   started from.

Two serve-path fixes rode along and matter for anything you measure live: `NEP` → `NE` roster
code (`abb82025`; an unmapped code made `_rescale_team_volume` raise and dropped every NFL market
for the run) and the within-team `Player depth` rank at serve time. Verify the serve path with
`~/backups/sportstradamus/2026-09-19-nfl-fp-bump/serve_parity_probe.py` after any retrain — the
fake-mode integration suite no-ops `Stats` I/O and cannot see these breaks (memory
`nfl_serve_path_parity_probe`).

## 3. Where the numbers stand

**19 of 20 cells ship** on `data/training/model_stats.parquet`; `passing yards` is the one withheld
cell and has no pickle. Twelve shipped going in, and the confirm campaign (`ship sweep --confirm
--confirm-fresh-only --min-model-weight 0.2`, one cell at a time) added the seven marked †. Their
`shipped: "devel"` flips and recipes are in the working-tree `stat_meta.json` only — uncommitted, for
the owner's review. Recipes before → after, all six gate values, the confirm-ledger rows and the walk
logs are in `~/backups/sportstradamus/2026-09-19-nfl-fp-bump/report_ship15.md`.

| cell | dist | BSS | w | n (sportsbook) | g1 ci_hi / 0.005 | g4 / cap | g5 / 0.075 |
|---|---|---|---|---|---|---|---|
| attempts | SkewNormal | +0.012 | 0.34 | 441 (380) | 0.0024 | 0.0550 / 0.0647 | 0.007 |
| carries | DPO | +0.040 | 0.82 | 1,303 (979) | −0.0004 | 0.0280 / 0.050 | 0.004 |
| completions † | SkewNormal | +0.004 | 0.68 | 442 (385) | 0.0041 | 0.0449 / 0.0646 | −0.002 |
| fantasy points prizepicks | SkewNormal | — | 1.00 | 2,851 (0) | — | 0.0349 / 0.050 | 0.007 |
| fantasy points underdog † | SkewNormal | — | 1.00 | 2,983 (0) | — | 0.0213 / 0.050 | −0.003 |
| interceptions | DPO | +0.248 | 0.90 | 455 (382) | −0.0656 | 0.0380 / 0.0637 | 0.064 |
| passing first downs | SkewNormal | — | 1.00 | 407 (0) | — | 0.0521 / 0.0673 | 0.003 |
| passing tds | NegBin | +0.211 | 0.90 | 447 (367) | −0.0260 | 0.0495 / 0.0642 | 0.032 |
| passing yards | SkewNormal | — | — | withheld | — | — | — |
| qb tds † | ZINB | — | 1.00 | 457 (0) | — | 0.0587 / 0.0635 | 0.015 |
| qb yards † | SkewNormal | — | 1.00 | 401 (0) | — | 0.0655 / 0.0678 | 0.016 |
| receiving tds | ZINB | — | 1.00 | 2,989 (0) | — | 0.0359 / 0.050 | 0.007 |
| receiving yards | SkewNormal | +0.021 | 0.36 | 2,993 (2,452) | 0.0014 | 0.0427 / 0.050 | 0.004 |
| receptions | SkewNormal | +0.012 | 0.80 | 2,936 (2,450) | 0.0030 | 0.0293 / 0.050 | 0.043 |
| rushing tds | ZINB | — | 1.00 | 1,294 (0) | — | 0.0351 / 0.050 | 0.003 |
| rushing yards | SkewNormal | +0.003 | 0.60 | 1,352 (1,092) | 0.0028 | 0.0426 / 0.050 | 0.030 |
| sacks taken † | SkewNormal | — | 1.00 | 446 (0) | — | 0.0462 / 0.0643 | 0.070 |
| targets † | SkewNormal | — | 1.00 | 2,783 (0) | — | 0.0316 / 0.050 | 0.000 |
| tds | NegBin | +0.160 | 0.90 | 3,343 (3,222) | −0.0227 | 0.0206 / 0.050 | 0.006 |
| yards † | SkewNormal | — | 1.00 | 2,900 (0) | — | 0.0374 / 0.050 | 0.015 |

Read the board with three facts in view.

- **The verdicts are stale against today's matrices.** The game-line re-read of
  [honest-receipts.md](honest-receipts.md) §6 rewrote all 20 NFL matrices after the campaign closed,
  so no board hash matches its file. The pickles load and serve (`serve_parity_probe.py` passes on
  all 19); the gate values are evidence for the matrices the campaign trained on until the next
  retrain, which also brings that lane's push-as-half-an-Over label and Brier-only temperature fit
  to every cell.
- **Ten cells are book-less** (blank g1 and BSS, `w = 1.0`): no sportsbook quote on their validation
  rows, so Gate 1 auto-passes, no blend weight is fitted — 1.0 is the fallback, not an optimum — and
  their ship is Gates 2–6. A pick'em-only cohort is the `pickem` provenance class and never counts as
  book evidence ([docs/ship_gate.md](../ship_gate.md)). `yards` and `qb yards` will not stay
  book-less (section 5).
- **Four ships are thin**: completions (g1 ci_hi 0.0009 under its margin), qb yards (g4 0.0023 under
  its cap, on the board's largest dispersion correction, 2.58), qb tds (g4 0.0048 under) and sacks
  taken (g5 0.0054 under). Expect a retrain to flip some of them.

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
cells keep serving until a human culls them (memory `project_serve_iff_ship`). So the ship count
describes the *gate* state, not what prod serves.

**`sportstradamus ship sweep`** is the search. One conditional-TPE study per cell over the
strategy catalog (`training/model_strategy/specs.py`): for SkewNormal cells the axes are
`target_normalization` ∈ {ratio_meanyr, centered_additive_mean10,
centered_additive_eb_meanyr_k10, ratio_projvol}, `dist_training_loss` ∈ {crps, nll},
`sn_param` ∈ {direct, centered}, `blending` ∈ {crps, crps_1se, nll} and `posthoc` ∈ {none,
prob_recal_isotonic, prob_recal_platt, prob_recal_book_citl, roe_mean, isotonic_mean,
cdf_recal_isotonic}; count families carry their own axes. Every corner trains holdout-blind and
is scored on a cross-fit gate row, so the board's `ships` flag is optimistic (memory
`crossfit_board_ships_optimistic`) and its `slack`/`discounted_slack` is the ranking signal.
Board rows are matrix-scoped (`matrix_hash` column): every NFL row in
`data/research/strategy_research_board.csv` was earned on a matrix that has since been rewritten
and is inadmissible, so budget every NFL cell as a cold sweep (`--max-trials` default 48 trained
corners; `-j` parallelizes cells, never corners). A corner that outruns the per-corner timeout
(300 s at most) goes unscored — on the 19,000-row cells that drops part of the DPO family.

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
  incumbent").

`--min-model-weight T` also binds the fresh lane, and this lane runs every walk at 0.2 (section 6,
decision 5): a board corner whose fitted weight is below `T` never nominates (when none is left
the seeds and the incumbent still walk), and a confirmed retrain that lands below it reverts even
at 6/6. A book-less cell, at the 1.0 fallback, is unaffected.

Consequence for this lane: **a served cell that fails the gates is best recovered through the
fresh lane**, which means flipping it to `shipped: "withheld"` first. That is a `stat_meta`
edit and therefore the owner's (section 6, decision 1). It also darkens the cell on the next
`meditate` until it ships again — which is what serve-iff-ship says should happen anyway.

**Automatic g4 rescue.** When a nominee fails ship on Gate 4 alone and sits within 0.010 of its
cap (`_G4_RETRY_MAX_EXCESS`), the walk pins `hpo_selection: "calibrated"` and reruns the
identical `meditate` once (memory `calibration_hp_selection_lever`: CV-loss objective with an
OOF PIT-KS feasibility constraint). Yards, qb yards, receptions and completions already carry the
pin, so the retry cannot help them further; no other NFL cell does. Hurdle-ZINB, Mixture and
structural corners get no retry: such a nominee that misses Gate 4 just reverts.

**Costs measured on this box.** Full-HPO `meditate` over all 20 NFL cells: 1 h 58 (~6 min per
cell). Cold matrix rebuild: c2 34 min, c3 2 h 35, c5 6 h 34 (~24 min per dependent cell). One
cell's sweep: 14–29 min (1 h 49 on `yards`, where DPO corners ran to the timeout). One confirm
nominee: 13 min on a 440-row cell with a short booster, about an hour otherwise, 85 s for a count
family; timeout per nominee 4 h. A killed walk leaves its cell `shipped: "devel"` with the
nominee's recipe and no pickle — copy the cell back from the walk's own
`stat_meta.<timestamp>.bak.json` (memory `confirm_walk_killed_leaves_devel_flag`).

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

## 5. Per-cell state and lever queue — the mandatory six

Five of the six ship; passing yards does not. Each subsection gives the recipe in `stat_meta.json`,
the gate row it earned on the board of section 3, and what to reach for when a retrain flips it.
Levers marked **research-gated** need a `research-analyst` brief first (§8.2 of the track; the hook
enforces distribution-family edits, this brief covers the judgement calls).

### passing yards — withheld, the open cell

Recipe (the campaign left it unchanged): SkewNormal, `centered_additive_eb_meanyr_k10`, nll, direct,
blending `crps_1se`, posthoc `prob_recal_book_citl`.

At full HPO this recipe passes all six gates — g1 ci_hi 0.0008, g4 0.0543 / 0.0644, g5 −0.005 — but
only at `model_weight` 0.05 with BSS −0.0002: the blend rides the book and the model adds nothing,
so the confirm walk reverted it on the weight floor (section 6, decision 5). The sweep says the same
of the model on its own. Of its 13 scored corners nine sit at exactly w 0.05; the two that stood
above the floor, both at w 0.90, failed Gate 1 (ci_hi 0.0059 and 0.0217 against 0.005). The gap to
close is about 0.001 of paired Brier against the sportsbook consensus, on 390 validation rows across
63 players. Gate 4, the binding gate on the two boards before, passes now.

1. **Feature design on the FantasyPoints suite — the owner's top priority, and the next lane.** The
   cell is not starved: 13 FantasyPoints-derived columns sit in its top 50 by |SHAP|. What is
   unbuilt, per [docs/fantasypoints_expansion.md](../fantasypoints_expansion.md): the matchup
   interaction between a quarterback's per-context splits (`ps_*`) and the opposing defense's
   coverage, blitz and pressure rates; the Tier 2 fan-outs (`qbPressured`, `playAction`,
   `screenPass`, `down=3`, personnel); the defense-mirror asymmetry (12 FantasyPoints columns on the
   defense profile against 33 on offense); and three Tier 1 endpoints never collected
   (`recsep-roles`, `injury-reports`, `coach-records`). A feature bump costs the cold rebuild of
   section 4 plus a retrain, and re-rolls every NFL verdict.
2. Not a corrector. Every outcome-fitted probability corrector learns the fold's over-rate
   ([nfl-passing-yards-gate1-recovery.md](nfl-passing-yards-gate1-recovery.md)); the book-anchored
   intercept already in the recipe is the one that does not, and it is what makes Gate 1 a tie at
   w 0.05.
3. Not the weight floor. Below 0.20 the served probability is the book's, and the cell would ship
   with nothing of its own to offer.

### rushing yards — ships

Recipe: SkewNormal, `ratio_meanyr`, crps, centered, blending crps, posthoc `cdf_recal_isotonic`.
It has shipped on every board: g4 0.0426 / 0.050, Gate 1 ci_hi 0.0028, BSS +0.003, w 0.60, n 1,352
(1,092 sportsbook). The risk is a g4 coin-flip near the cap on a retrain; if it ever fails g4 alone
the calibrated retry covers it. Otherwise leave it alone and do not sweep it under
`--include-shipped` — S2/S3 against a passing incumbent costs confirms and rarely wins.

### receiving yards — ships

Recipe: SkewNormal, `ratio_meanyr`, crps, centered, blending crps, posthoc `prob_recal_isotonic`.
It ships since the trim floor went to 20,000 (section 6, decision 2), after failing Gate 4 on every
board before that: g4 0.0427 / 0.050, Gate 1 ci_hi 0.0014, BSS +0.021, w 0.36, n 2,993 (2,452
sportsbook). Its g4 misses were scale-bound (central coverage slightly under nominal), so if a
retrain fails it on Gate 4 alone:

1. Calibrated retry — automatic in the walk when the miss is within 0.010 of the cap.
2. `cdf_recal_isotonic` in the posthoc slot (replacing `prob_recal_isotonic`; Gate 1 passes without
   help, so the slot is free for the g4 job).
3. `centered_additive_mean10` normalization — the transform that flipped `carries` from below-book
   to beats-the-book at real HPO (memory `nfl_volume_cells_feature_mature`).

### yards and qb yards (the combos) — ship book-less, Gate 1 about to go live

Recipes: `yards` SkewNormal, `ratio_meanyr`, crps, direct, blending nll, posthoc `roe_mean`;
`qb yards` SkewNormal, `ratio_meanyr`, crps, centered, blending crps, posthoc none; both
`hpo_selection: calibrated`.

Both shipped with a blank Gate 1 because their validation rows held no sportsbook quote. `yards`:
g4 0.0374 / 0.050, g5 0.015, n 2,900, on its third nominee — a DPO corner failed Gate 2 and a
ZINB-hurdle corner missed Gate 4 by 0.0011. `qb yards`: g4 0.0655 / 0.0678 with `dispersion_cal`
2.58, the thinnest margin on the board, n 401.

The binding fact is still the **quote history**. No sportsbook combo line reached the archive after
the 2023 season, on dev or on prod, because `stat_map.json` did not map the two Odds API keys
(memory `nfl_combo_archive_gaps`); the 2026 rows before October are Underdog pick'em quotes, which
are not book evidence. With the keys mapped (`player_rush_reception_yds` → `yards`,
`player_pass_rush_yds` → `qb yards`; live `confer` polls them) sportsbook quotes now arrive: 109
for `yards` and 92 for `qb yards`, 21 players each from four or five books, over the first four
game dates (from 2026-10-05). Once a validation frame holds ten distinct players, Gate 1 scores
and a blend weight is fitted for the first time — on a sample thin enough to move with every
retrain.

1. The deep evidence is two seasons of Odds API history, and it exists: a one-date probe of the 2024
   opener returned both markets with real cross-book spread. The bulk pull is the owner's call
   (section 6, decision 3). It is honest evidence, not a pass ticket: it can hold either cell
   withheld.
2. After a pull, repair the two matrices' odds block with
   `python -m sportstradamus.scripts.inject_backfilled_odds --league NFL --markets 'yards,qb yards'`
   (it calls the real resolver — memory `inject_backfilled_odds_not_rebuild_equivalent`), then sweep
   and confirm each cell.
3. Do not add a book-mean blend to these cells: the component sum is already unbiased for a
   `combo_props` spec (memory `combo_book_mean_blend`).

### targets — ships book-less

Recipe: SkewNormal, `centered_additive_mean10`, crps, centered, blending crps, posthoc
`prob_recal_isotonic`. No sportsbook prices this market, so Gate 1 is blank by construction and the
cell ships on Gates 2–6: g4 0.0316 / 0.050, g5 0.000, n 2,783. It is also one of the league's
serve-time volume denominators (`VOLUME_STATS`), so its pickle is kept even while the cell is
withheld. The number to watch at the retrain is Gate 5:
[honest-receipts.md](honest-receipts.md)'s replay of this cell's test set under the
push-as-half-an-Over label put its debiased ECE at 0.060, inside the 0.075 cap.

### The other fourteen

All fourteen ship. Four came in through the campaign and are the ones to watch:

- **sacks taken** — its incumbent recipe, unchanged. Three alternatives failed Gate 5 and the
  incumbent passed at 0.0696 against 0.075, on a recipe that had missed Gate 4 by 0.0002 one board
  earlier, so this verdict is the most draw-dependent on the board.
- **qb tds** — the sweep changed the family, SkewNormal → ZINB hurdle. That cured Gate 5 (0.083 →
  0.015) and left Gate 4 at 0.0587 against 0.0635.
- **completions** — the one campaign ship with a live Gate 1: ci_hi 0.0041 against 0.005 at w 0.68,
  BSS +0.004. A tie with the book that the gate accepts.
- **fantasy points underdog** — book-less, wide margins.

The count cells (interceptions, passing tds, receiving tds, rushing tds, tds, carries) have shipped
on every board and need nothing. Interceptions is the owner's lowest-priority cell.

## 6. Owner decisions this lane needs up front

1. **Lane choice for failing served cells — decided 2026-09-27.** The nine cells with
   `ship == False` on the 2026-09-26 board are `shipped: "withheld"` (`7d986d3c`, on prod since
   2026-09-27 06:09 CDT; receiving yards back to `devel` in `34fbdcb4` after the floor-20k
   retrain), which put the remaining eight on the fresh lane: `ship sweep --league NFL --confirm`
   auto-ships a withheld cell on a clean 6/6. Serving fails closed on a withheld cell: while its pickle is
   still on disk the market yields no offers at all (not even the book fallback); a dev
   `meditate` prunes a withheld cell's pickle and `sync_to_prod.sh` mirrors `models/` with
   `--delete`.
   The 2026-09-28 sync carried all 20 dev pickles, the withheld eight included (they were left
   in place by the `--bypass-withholding` retrain; serve-neutral, since the withheld check in
   `model_prob` precedes the pickle check, but each logs a `withheld but pickle on disk`
   warning per prophecize run). **Landmine — fixed 2026-09-29 (`05e6e18c`):** `targets` is
   withheld *and* one of the three NFL volume dependencies (`VOLUME_STATS` in `helpers/io.py`,
   read by `StatsNFL.volume_stats`); `prune_model_pickle` now keeps every denominator, so a
   plain `meditate` refreshes its matrix, prunes nothing, and the WR/TE cells keep their
   targets projection while `model_prob` still serves nothing for the withheld cell. The kept
   pickle is not retrained either: a withheld denominator ages until the cell ships or a
   `--bypass-withholding` run refreshes it.
2. **Trim floor — decided 2026-09-27.** `pipeline._MATRIX_TRIM_FLOOR` is 20,000 (`0a22c0a9`;
   was a literal 15,000). The floor is the row count below which `trim_matrix` never balances,
   so at 20,000 the six WR/RB/TE cells the old floor was cutting gained 2.4–5.8k rows: targets
   14,548 → 18,915 (now below the floor, unbalanced), yards 14,546 → 19,863, receiving yards
   15,861 → 19,941, receptions 17,299 → 19,727, fantasy prizepicks 14,262 → 19,953, fantasy
   underdog 14,142 → 19,929 — those five landed just under 20,000, so their pre-trim populations
   exceed 20k and the floor still binds them, more loosely. tds and receiving tds (which the
   balancing barely touched) are unchanged. Global: every league's
   next persist keeps more unquoted rows too (NBA/WNBA caches all sat at the 15k floor). The six
   matrices were rebuilt cold on 2026-09-27 (`$B/run_floor20k.sh`, quarantine
   `$B/quarantine_floor20k`, pre-change artifacts `$B/pre_floor20k`) and the cells retrained:
   12 of 20 at ship, receiving yards passing 6/6 and back to `devel` (`34fbdcb4`; compare script
   `$B/compare_floor20k.py`). Separate finding, fixed 2026-09-28 (`ae52fe12`): every
   `meditate` re-trimmed the whole cache, and the Result band, the line clip and the push drop
   ignored the floor, so a cache lost 0.5–6% of its unquoted rows on each persist even with no
   new games (targets 15,000 → 14,548 between the 09-26 promotion and the 09-26 retrain;
   fantasy prizepicks 19,953 → 18,699 on the floor-20k re-persist). `trim_matrix` now takes
   the `new_rows` mask `_step_load_matrix` returns and applies those three steps to appended
   rows only; re-trimming the current caches with nothing new returns them unchanged (fantasy
   prizepicks was −859 a pass, receptions −273, NBA PTS −17; tds still gives back 60 rows to the
   floor-bounded balancing above 20,000, by design). Golden pin
   `tests/golden/test_trim_matrix_retrim.py`. The rows already lost return only with a rebuild;
   none was run.
3. **Combo quote history — decided 2026-09-29: probe now, bulk pull pending.** Prod's archive
   has the same hole as dev's, so there is nothing to merge; the Odds API has the history. The
   two combo keys are mapped for good in `stat_map.json` (`league_seed_costs.NFL` 165 → 198 in
   `odds_api_budget.json`), and a 21-credit dry run on the 2024 opener returned both markets
   with real cross-book spread. The bulk pull stays the owner's key-and-timing call — about
   11,000 credits for 2024-09-05 to 2026-02-08:
   `python -m sportstradamus.scripts.backfill_historical_odds --league NFL --markets 'yards,qb yards'
   --start 2024-09-05 --end 2026-02-08 --key-name odds_api_plus --dates-from Moneyline --props-only`
   (the default `odds_api_max` key answers HTTP 401; the game-dates seed from `Moneyline` because
   the combos have no archived rows for those seasons; `--props-only` skips the ~30-credit game-line
   call per date, whose rows already exist). Put `--max-dates 1 --dry-run` on it first: the dry run
   makes the paid calls and prints the cross-book spread, and zero spread means a degenerate
   source (memory `capture_stub_drift_live_probe`). The key is the one live `confer` polls with,
   and the governor paces on the live remaining-credits header, so a pull is paid for by
   throttled live polling for the rest of the cycle: on 2026-10-08 the cycle held 16,381 of
   20,000 credits with live use near 470 a day, which leaves about 5,600 spare by the reset on
   the 1st — one season, not two. Size each run to the leftover with `--max-dates N` in a
   cycle's last days; the script resumes from its progress file.
4. **Push and sync — done 2026-09-27.** `abb82025`, `4c8b4a91`, the docs and the withheld flip
   are on `origin/devel` (06:06 CDT) and on prod (`run_job.sh`'s self-deploy pull replicated by
   hand at 06:09 CDT under its pull lock, dashboard restarted). Prod's gamelog was recomputed
   06:09–06:14 CDT with `recompute_pbp_zeroed.py` under the archive flock (5 min, 2.2 GB peak;
   10,209 rows revisited, 75 residual all-zero rows with snap > 0.3 — the same six nickname
   mismatches as dev; log `~/recompute_20260927.log`, pre-run copy
   `~/gamelog.parquet.bak_20260927`, both on prod). `scripts/sync_to_prod.sh` ran on
   2026-09-28 16:14 CDT: the 20 NFL pickles (floor-20k retrain), `model_stats`, the two
   calibration JSONs, the corr files, and the FantasyPoints snapshots — the re-pulled 2021–2025
   seasons (1,320 files each; prod's history had been legacy-era until then) plus 2026 week 1
   and one week-2 file, 6,661 files. Dev's week 3 was 61 empty stubs from a 09-25 pre-game
   fetch; they were moved to `$B/nfl_2026_week_03_empty_stubs_20260925/` first so no partial
   week reached prod (and so the dev-side fetch does not skip week 3 later). The script does
   not carry `leagues/nfl/gamelog.parquet`, which is why the recompute ran on prod directly.
   Prod's own `team_data/NFL/2026/week_03` holds 20 empty 636-byte stubs from a 2026-09-16
   fetch (pre-port); harmless, the collector skips only non-empty files, so Wednesday's
   `fp-fetch` refetches them. The 16:50 CDT prophecize run was the first on the new pickles:
   OK in 896 s, log window identical to the 15:50 run's (no NFL error), receiving yards served
   (130 offers in the 17:04 `current_offers` snapshot), the eight withheld cells absent, one
   `withheld but pickle on disk` warning per withheld cell per platform.
5. **Gate 1 evidence and the weight floor — decided 2026-09-29.** Gate 1 and the blend-weight
   fit count sportsbook quotes only, and a pick'em-only cohort is the `pickem` provenance class
   ([docs/ship_gate.md](../ship_gate.md) holds the rule). Every confirm walk runs with
   `--min-model-weight 0.20`, so a recipe that rides the book cannot ship. That floor is what
   holds passing yards withheld.
6. **The seven flips — pending.** The campaign's `shipped: "devel"` flips and recipes for
   targets, yards, qb yards, sacks taken, qb tds, completions and fantasy points underdog are
   uncommitted in the working-tree `stat_meta.json` (diff: `$B/stat_meta_ship15.diff`); accept or
   reject per cell. Two cautions. Sync the pickles before pushing the flag: prod still holds
   these cells' older pickles, which failed their gates, and serves whatever pickle is on disk
   once a cell stops being withheld. And do not check the diff away before a retrain unless the
   models are meant to go: a plain `meditate` prunes a withheld cell's pickle.
7. **Which retrain re-rolls the stale verdicts — pending.** Recommended: let the
   [honest-receipts.md](honest-receipts.md) lane's instrumented NFL retrain re-verdict all 19
   cells on the patched matrices, then commit the flips of the cells that still ship, sync and
   push once. The alternative is to ship the seven now on the campaign's evidence and let that
   retrain's ship-gate warnings name any cell that falls.
8. **Passing-yards feature lane — directed 2026-10-03, not started.** Section 5.
9. **Relabel the other four leagues' caches — pending; recommended before their next retrain.**
   Decision 5 reached the NFL caches only. All 73 non-NFL matrices carry the old label: no
   `pickem` row, and 14,705 `authentic` rows on seven cells that no sportsbook prices — MLB
   hitter fantasy points underdog 7,701, NHL hits 2,856, NHL skater fantasy points underdog
   1,211, NHL goalie fantasy points underdog 987, WNBA fantasy points prizepicks 767, NBA fantasy
   points prizepicks 626, MLB pitcher fantasy points underdog 557 — plus an uncounted number on
   mixed cells (a matrix stores no book name). `meditate` resolves appended rows only, so the
   first retrain of such a cell still scores Gate 1 and fits its blend weight on the old rows
   while new rows arrive `pickem`: the evidence shrinks week by week and the verdict drifts with
   no model change, and neither `meditate` nor the matrix audit detects it. The repair is
   `python -m sportstradamus.scripts.inject_backfilled_odds --all-cached` (or one league's cells
   with `--league L --markets '…'`), `--dry-run` first: it re-resolves the whole quote block, so
   prices move with the labels and every hash moves. A full matrix rebuild relabels too, so the NHL and MLB rebuilds
   [honest-receipts.md](honest-receipts.md) §8 asks about would cover those two leagues.
10. **A pick'em composite hides its component sum at serve — recorded, no change proposed here.**
    `prediction/book_quotes.py` anchors a combo's component sum on the composite's own direct
    quote whenever one exists, a pick'em-only one included, and the anchored quote then fails
    `_has_serving_support`. So once a platform's row for the composite is archived, the player
    has no book leg although the sum was admitted. It predates this lane, costs nothing while a
    combo cell blends at weight 1.0 (`yards`, `qb yards`, `qb tds`), and matches how training
    scores the same rows (a `pickem` row is model-only). It starts to matter when a combo cell
    gets a fitted weight, which the arriving sportsbook quotes make likely. Changing it alters
    served output, so it belongs with the honest-receipts lane's book-leg rules.

## 7. Rules the lane runs under

- No `git push`, no PR creation, no `sync_to_prod.sh`, no `stat_meta.json` commit without the
  owner's review. Commit by pathspec only (`git commit -- <paths>`); never `git add -A`,
  never `git stash`; never a HEAD-moving git command inside a subagent prompt (the checkout is
  shared).
- One `meditate` at a time on the box (`pgrep -af "sportstradamus medi[t]ate|model.strateg[y]"`
  must print nothing first — the brackets keep the pattern from matching its own shell), and
  never the integration suite while one runs. The sweep's `-j` parallelism is the exception the
  tool manages itself. A confirm walk's retrain opens the DuckDB archive, so it also waits for
  any archive job (`merge-archives`, `confer`).
- Never fetch a partial FantasyPoints week. A week is complete only when a team-grain tool holds
  all 32 teams — the morning after the Monday game it can still show 30. Probe one tool
  (`fetch fp run --season YYYY --week N --only team_proe_report`), then
  `fetch fp run --season YYYY --week N --refetch` (memory `fp_column_map_applies_at_fetch_time`).
  Never sync a partial week.
- Any `.py` touched: `refactoring-specialist` on every touched file, then the single gate run
  `poetry run ruff check src/sportstradamus/`, `poetry run pytest tests/golden/`,
  `poetry run pytest -m integration -n0`.
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
| a plain `meditate` prunes every withheld cell's pickle except the volume denominators (`VOLUME_STATS`); reverting a flip before a retrain deletes the model | section 6, decisions 1 and 6 |
| a killed confirm walk leaves its cell `shipped: "devel"` with no pickle; restore the cell from the walk's `.bak.json` | `confirm_walk_killed_leaves_devel_flag` |
| a `pgrep -af` wait loop matches its own wrapper and never exits; bracket the pattern or gate on an `EXIT=` marker | `pgrep_pattern_self_match` |
| a detached `setsid nohup` driver does not survive a WSL restart; the campaign needs a relaunch script | section 4, costs |
| every persist re-trimmed the whole cache (fixed 2026-09-28; rows already lost need a rebuild) | `matrix_persist_retrim_erosion` |

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
`ship == True` **including all six mandatory cells**, every one produced by a full-HPO
`meditate` on the promoted matrices, `serve_parity_probe.py` builds every `expected_columns`
entry for every pickle, the three gates are green, the `stat_meta.json` diff is in front of the
owner, nothing pushed. If the count stalls below 15 or a mandatory cell stays out, the report
says which cells, which gate binds each, what was tried with its ledger rows, and what the owner
could decide to unblock it — scaling the goal down is the owner's call, not the lane's.

Where that stands: the count is met (19), every ship came from a full-HPO confirm retrain, the
parity probe passes on all 19 pickles, the gates are green and the diff is with the owner.
`passing yards` is out, so the lane is not done; section 5 says what binds it and section 6 lists
what the owner holds.

The report is `~/backups/sportstradamus/2026-09-19-nfl-fp-bump/report_ship15.md` (normal prose).
Per cell it carries: recipe before → after, the six gate values with caps, BSS, `model_weight`, n
and sportsbook-priced n, the confirm ledger rows consulted, and the `matrix_hash` the verdict is
bound to.
