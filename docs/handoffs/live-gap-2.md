# Lane brief — live-gap-2: why the models pass the offline gates and lose live

**Read first:** CLAUDE.md, [CONTRIBUTING.md](../../CONTRIBUTING.md),
[docs/ARCHITECTURE.md](../ARCHITECTURE.md), [docs/ship_gate.md](../ship_gate.md),
[honest-receipts.md](honest-receipts.md) §5–§6 (the three research briefs and the I6 instruments),
[docs/archive/researcher_train_serve_skew.md](../archive/researcher_train_serve_skew.md) (R3, the
decomposition this lane builds on), [docs/archive/researcher_live_gap_2.md](../archive/researcher_live_gap_2.md)
(this lane's pre-registration) and [nfl-ship15-recovery.md](nfl-ship15-recovery.md) §11 (lever T).
This brief is self-contained but those documents are the contract it works under.

Status: OPEN (opened 2026-10-21 on the owner's ask; week 1 runs 10-22 → 10-28). The owner's
standing rules apply unchanged: a live Gate-2 verdict never removes a cell from `devel`
([ship_gate.md](../ship_gate.md)); no model is pulled on realized money; `stat_meta.json` is the owner's
file; every send lands between board runs; work on side branches; the refactoring-specialist runs on every
touched `.py`; the three gates and CI's `ruff format --check` before any push.

## 1. Mandate

Find why cells that pass the six offline gates lose to the book live, fix what can be fixed on
the training and serving side, and set up the live data collection that grades the fixes. The
questions the owner asked on 2026-10-21, answered in §5–§8: how to kick the investigation off, the
soonest fixes can start, what to collect meanwhile, and which repairs of lever T's class go first.

Definition of done: each hypothesis in §3 carries a verdict with its pre-registered statistic; the
training fixes that pass their experiments are on `devel` and on production in one retrain per
league; the live instruments (§6) read the fixed versions over a stable window; the verdicts are in
[model_improvement_track.md](model_improvement_track.md) §3.4 and the lane is closed.

## 2. What is known (verified 2026-10-21; every number has a script)

1. **The gap is mostly selection with little information at the market** (R3, 2026-10-04): serve-path
   skews explain ≤ 0.7 pp of the 12.5 pp recommended-leg overstatement; 89 % is selection on
   model–market disagreement where the served mean knows little beyond the market. Lever T removed the
   star tilt on 2026-10-18 (nfl-ship15-recovery §11) and nothing else: post-send the recommended cohort
   still loses 6–9 % per pick on both sides. Unders are the better half everywhere (NFL since 09-01:
   Over hit .478 / −14 %, Under .488 / −8 %; MLB −14 % / −6 %; NHL −7 % / +0.5 %; WNBA −17 % / −2 %),
   and the edge-captured ratio λ = (hit − book)/(read − book) is positive only on Unders (NFL +.08,
   NHL +.44) and negative on Overs in NFL, MLB and WNBA. The Under share of the board is the market's
   shape plus the argmax side, not a model tilt. Record: `~/backups/sportstradamus/2026-10-21-under-read/`.
2. **The serving boosters are stale.** `training/pipeline.py` splits each matrix by date, puts the
   first 70 % in training (`_TRAIN_FRACTION`), fits the GBDT once on that slice and never refits. Where
   the cut falls today: NFL receptions 2024-12-15, yards 2024-12-08, carries 2024-12-08, tds 2025-01-05;
   NHL shots 2025-12-11, points 2025-12-13; NBA PTS/REB 2026-03-10 (the matrix starts 2025-11-12, one
   season); MLB hits 2026-05-05. No served NFL booster has trained on a 2025 or 2026 game. R3 measured no
   decay of the booster's information across the held-out window and called the cutoff "not a lever"; that
   test measured decay, not the gain from training on the most recent 30 %, which is untested.
3. **The offline gates are optimistic by construction.** The held-out 30 % is split into validation and
   test by a hash of (Player, Date): the two halves cover the same period and the same games, and every
   calibrator (temperature, blend weight, post-hoc correctors, dispersion) is fit on the validation half
   and scored on the interleaved test half. The confirm walk (`model_strategy/confirm.py::_walk_nominees`)
   retrains a cell's nominees in order and ships the first one that passes the gates on those test rows
   (NFL yards shipped its third nominee; the ledger holds 81 full-HPO attempts over 15 NFL cells, 27
   shipped); the Gate-4 near-miss retry, the supersede comparisons and the manual ship flips read the
   same rows. No walk-forward or forward-in-time holdout exists anywhere. `--holdout-blind` (sweep only)
   requires `--deterministic`, so it trains the fixed deterministic booster and cross-fits only its
   calibrators over the validation half; it cannot re-gate a served model. Gate 1 is a non-inferiority
   test (`scorecard._GATE1_NONINF_MARGIN` lets the ensemble be 0.005 Brier worse than the book) with a
   95 % iid bootstrap CI. 31 of the 70 served cells have no computable Gate 1 and pass it blank; NFL
   yards' rests on 19 rows from 18 players over 4 dates (iid upper bound −.001, date-clustered +.022).
   So "pass offline" means tie-or-better on a multiply-inspected random half of the same period, while
   "fail live" means a strictly negative 30-day Brier skill, forward in time, on the props the platforms
   post.
4. **Where the live loss sits.** On NFL sportsbook-quoted main lines over the 30 days to 10-20 the count
   cells lose to the decoded consensus (carries −.083, interceptions −.073, completions −.046, receptions
   −.043, tds −.042, attempts −.034) and the yardage cells tie (receiving yards −.005, passing yards +.001,
   yards +.005, rushing yards +.006). The whole loss sits in the band where the model disagrees with the
   book by more than 10 points (BSS −.115 on 1,735 rows; the three narrower bands read −.002 to +.000).
   The count cells' archived book quotes are mutually inconsistent and both miss the realized outcome
   (memory `nfl_count_quote_columns_disagree`; the 2025 table: tds claimed Under .953 vs realized .802),
   their blend weights are .82–.90, so live the model stands nearly alone against the market exactly
   where it disagrees most. `ship graduation` reads every served NFL cell `demoted` (0 graduated, 10
   demoted) on the 10-15, 10-18 and 10-21 live-metrics files: the verdict predates the campaign and
   scores the 09-27 / 10-03 / 10-15 versions; the 10-18 versions' own 30-day read lands about 11-17.
5. **What serving persists today.** `prediction/feature_log.py` writes one row per scored player, game
   and platform with the frame fed to the model, the decision-time book leg (`Quote Over Prob`, `Quote
   EV`), the raw outputs and the pre-blend projection; partitions older than 30 days are deleted at the
   end of `prophecize`; `scripts/sync_from_prod.sh` pulls `data/runtime/` without deleting, so the dev box
   keeps what prod prunes as long as it syncs inside 30 days (dev holds 2026-10-05 → 10-26 today).
   `history.parquet` holds the blended `Win Prob`, `Market Prob` at the DFS line, `Consensus Line`,
   `Model Weight`, `Temperature`, the blended `Gate`, both payouts and the decision-time columns
   (`history_schema.DECISION_COLS`, since 2026-10-15); it lacks the pre-blend model probability, the
   blended skew (so a SkewNormal `Win Prob` cannot be rebuilt from the row), the book probability at the
   book's own line, the per-book quotes and the first-seen time. `meditate` keeps a replaced model file
   14 days under `data/models/prior/` (dev only; production receives models by rsync and has no prior
   directory). `data/test_sets/*.csv` is overwritten at each retrain.
6. **Offline skill against the book is mostly the book's level error on 11 cells** (exploratory, post
   hoc; [research brief](../archive/researcher_live_gap_2.md) Finding 1). Across the 38 served cells with
   at least 100 priced held-out rows, offline BSS tracks the |CITL| of the training book leg (Spearman
   ρ .75). On 10 of the 11 cells with |CITL| above .05 (NFL interceptions, passing tds, tds, receptions,
   carries; NBA BLK, STL, FG3M, BLST; WNBA FG3M), re-levelling the book by its own mean leaves BSS at
   .052 or less; MLB home runs keeps .154, so its problem is more than level. On the NFL count cells the
   pre-2026 rows carry the error and the 2026 rows are levelled, and live `Market Prob` is within 1.2
   points on receptions, carries, interceptions and tds: Gate 1 and the blend weight were fit against an
   opponent the live board does not field.

## 3. Hypotheses and decision rules (pre-registered before the first experiment)

| # | Hypothesis | Experiment (§4) | Pre-registered rule |
|---|---|---|---|
| H1 | The stale booster costs live skill: refitting the booster on the most recent rows raises forward Brier skill | E1 | At one forward origin (W = rows dated on or after the date of row ⌊0.9n⌋; every arm trains only on earlier rows; served θ pinned), arm C (A's file with the booster refit on all pre-W rows) against arm A (0.7 on the pre-W rows): d̄ = mean[(P_A − y)² − (P_C − y)²] on priced W rows (all W rows for yards), 95 % game-date block bootstrap CI (4,000 draws; NFL week blocks as a sensitivity check, and a cell whose verdict differs between the two counts as neither). At least 3 of 5 cells with the CI above 0 and none below → adopt the refit protocol in the week-3 batch; else H1 is not a lever. Arm B (0.85) is descriptive: a pass only through B is a lead for a new pre-registration |
| H2 | The gate verdicts are optimistic: the first-to-pass walk and the same-period split inflate `ship` | E2 | Descriptive: the served booster re-gated with calibrators cross-fit over the validation half (and over the test half, for the symmetric read): `ship` flips by gate and league, the slack shift, the Gate-1 CI method, E1-A's forward gates. The blind confirm walk is adopted regardless (protocol); whether re-gated cells leave `devel` is the owner's call |
| H3 | The count cells' training book leg is the defect: Gate 1 ties a broken opponent and the blend weight follows | E3 | On the NFL count cells (attempts, carries, completions, interceptions, passing tds, receptions, tds), L = Σ n_c \|CITL_c\| / Σ n_c with CITL = mean(P_book,over − y): L_off on the served split's priced held-out rows (both halves; matrix `Odds`), L_live on settled quoted main lines from 2026-09-21 (`Market Prob` by `Bet`; pushes void; each prop counted once). L_off > .05, L_live ≤ .02 and T1 rejects → the producer is the defect → I6d opens, scoped to book-leg levelling in every league; else the count cells route to H4 |
| H4 | The gate population hides the disagreement band: offline Gate 1 scores rows where model and market agree | E4 | BSS of the served test-set P against the book on priced rows with \|P − P_book\| > .10 whose opponent is levelled (NFL priced cells' rows from 2026-09-01, plus receiving and rushing yards from every season; 356 band rows, counted blind). T2 rejects (BSS < 0) → propose the band read as a monitor column (never a gate; owner packet); else the live loss is a population shift and E3 carries the verdict. The literal Gate-1 band (90 % of it faces a mis-levelled book) is secondary |
| H5 | Version age: a fresh version is more honest than an aged one | I6g | Already pre-registered in honest-receipts §6 (one-sided α = 0.05; ≥ 5 fresh and 10 aged game days; ≥ 300 recommended legs per arm): NHL about 10-29, NBA about 11-05 |

**Multiplicity and counting** (fixed before any confirmatory read). T1 (H3: D = L_off − L_live > 0, with
offline and live dates resampled independently) and T2 (H4) form one Holm family: one-sided, family-wise
error .05, p from 4,000 date-block draws (seed 1729); the smaller p is compared with .025, the larger
with .05. T1 was previewed unblinded (D ≈ .085); T2 is blind (only counts were read). H1 is decided by its
own count rule; E2, E3 (b)–(e) and E5 are descriptive. Every live read counts a prop once:
`realized.settled_offers` does so for games from 2026-10-09, and earlier rows are weighted 1 / (the number
of lines their prop-side was posted at), the expectation of a random one-line draw; the per-row number and
the games-from-10-09 window are reported beside it. Windows, arms, power and exclusions:
[the research brief](../archive/researcher_live_gap_2.md) §3.

Killed levers are not retried without new evidence (memory `research_verdicts_ledger`): the trust
layer, selection shrink toward the book, a higher edge floor, the argmax-EV side rule, price-blind tail
recalibration, the post-fusion mean corrector as built.

## 4. Week 1 (2026-10-22 → 10-28): no retrain, no clock reset

**4.1 Research brief.** Done, with R3 as the prior:
[docs/archive/researcher_live_gap_2.md](../archive/researcher_live_gap_2.md) pre-registers the statistics,
populations and Holm family that §3 pins.

**4.2 Instrumentation push (subagent-driven, one module per subagent; ships to `devel` in a job gap):**

| item | where | change |
|---|---|---|
| Feature-log retention | `prediction/feature_log.py` (the 30-day constant; the prune runs last in `prophecize`) | keep the season (400 days); ~5 MB a day, production has 13 GB free |
| History columns | `history_schema.py` + `prediction/offer_records.py` + `prediction/persist.py` | add the pre-blend model probability of the chosen side at the offer line, the blended skew and `skew_cal`, `Hist Gate`, a quoted-vs-payout-implied flag on `Market Prob`, and `First Scored At` (the minimum on upsert); `DECISION_COLS` rules: offer-level, never backfilled, NaN on older rows |
| Prior model files | `training/prior_models.py` | keep the season on the dev box (14 → 400 days; 109 files are 173 MB today) |
| Test-set dumps per version | `training/pipeline.py` (the dump path) | copy each cell's CSV to `data/test_sets/prior/<model version>/` at retrain, so offline reads can be re-run per version |
| Dev-side sync | dev crontab | `scripts/sync_from_prod.sh` weekly, so no partition is ever pruned before the dev copy exists |
| Recommender stage counts | `prediction/` (the recommendation rule's call site) | optional: one line per board run with the legs surviving each filter per cell and side |

**4.3 Experiments on the data in hand (main session; outputs beside their scripts in the lane folder
`~/backups/sportstradamus/2026-10-21-live-gap-2/`):**

- **E1 train fraction / refit** (decides H1): NFL receptions, yards, carries and NHL shots, points, in a
  throwaway worktree through `--artifact-output` and `--frozen-matrix-dir` with environment-guarded hooks,
  never on the served pickles. Matrices and `stat_calibration.json` are frozen, and each matrix's sha256
  must equal its pickle's `matrix_hash`. Every arm keeps the served corner and θ (no HPO): A = 0.7 on the
  pre-W rows, B = 0.85 on them (descriptive), C = A's file with the booster refit on all of them. A
  blocking reproduction check runs first: the harness at 0.7 on the full matrix must reproduce the served
  test-set `P` and `P_standalone` within 1e-6. Secondaries: B against A and C, the raw booster's Brier and
  NLL, Gate 4 and the tail scorecard per arm on W, and W ∩ 2026. MDE80 is .0006–.0014 Brier on the priced
  cells and .0055 on yards; R3 predicts an NFL null, so NHL is the cleanest test.
- **E2 gate optimism** (H2, descriptive): `--holdout-blind` cannot re-gate a served model (§2.3), so the
  harness loads each of the 70 served cells' booster and cross-fits only the calibrators (player-disjoint
  folds) over the validation half, and over the test half for the symmetric read. Reads: `ship` flips by
  gate and league against the stored verdicts, the symmetric slack shift, Gate 1's upper bound under iid,
  player, date and two-way CIs on both halves, and E1-A's forward gates. The NFL count cells are flagged
  (a mis-levelled opponent on both halves); MLB and WNBA are read as a trimmed population until their 2027
  rebuilds.
- **E3 same-rows decomposition**: (a) H3's statistic (§3); (b) model-probability parity on the live rows
  (feature log joined to history on League, Market = model cell, Platform, Player and Date, then to the
  matrix on Player and Date), NFL through the current pickle on both frames; (c) book-leg parity: matrix
  `Odds`, logged `Quote Over Prob` and `Market Prob`, each against the outcome at its own line; (d) the
  population split (sportsbook-quoted main lines, alt rungs, unquoted) with the board → recommended funnel
  per cell and side, which stands in for recommender stage counts; (e) the push share per population.
  R3's `c16_parity.py` and `c13_bookonly.py` under `~/backups/sportstradamus/2026-10-04-honest-receipts/r3/`
  are the starting scripts.
- **E4 disagreement-band gates** (decides H4): BSS of the served test-set P against the book by
  \|P − P_book\| band (edges −1e-9, .02, .05, .10, 1, as the live table in
  `~/backups/sportstradamus/2026-10-21-under-read/`) on the levelled-opponent population of §3. Secondary:
  the literal priced test rows (expected positive because of H3), the count and yardage strata, NHL.
- **E5 I6e parity read**: the training matrices hold the logged games from 10-05 (NFL to 10-15, NHL to
  10-17). `admin feature-parity` flags only the sd of the raw ΔP, which cannot see a level shift, and
  scores each row through its own version's file, which leaves 4–36 NFL rows per cell. This read flags a
  cell when the CI lower bound of RMS(ΔP) exceeds 1 pp; it scores NFL receptions, receiving yards, rushing
  yards, carries, interceptions, tds and yards through the current pickle on both frames (completions lacks
  42 logged columns, qb yards has no pickle) and NHL through each row's own version, and ranks repairs by
  served RMS (≈ (w/T) × raw) × live volume. A builder mismatch gets a serve-side repair feature by feature;
  a timing gap (as-of game lines or quotes) routes to I6f. No retrain.

## 5. Weeks 2 and 3

- **Week 2 (10-29 → 11-05):** build the training fixes that passed their experiments in a worktree:
  the walk-forward evaluation with a refit on all rows for serving (H1), the blind confirm walk
  (nominees ranked on the cross-fit half, the test verdict read once for the chosen nominee), the
  count-quote producer if the I6d brief says so (H3). I6g reads: NHL about 10-29, NBA about 11-05.
- **Week 3 (about 11-06):** one retrain per league on the fixed code (the lever-T campaign's recipe
  without the rebuild; `scripts/sync_to_prod.sh --league L` between board runs), then four weeks of
  stable versions (to about 12-04) for the live verdict. The 30-day Gate-2 read of the 10-18 versions
  lands about 11-17 and is the control.
- **Sequencing (owner's call, open):** batch every model fix into the week-3 retrain (the default: the
  10-18 versions get a clean 30-day read first and every retrain resets the live clock), or ship each fix
  the day it is proven, accepting mixed versions in the live reads (the owner's preference on 2026-10-18
  was send-now).

## 6. Live reads: what each verdict needs

| read | needs | when |
|---|---|---|
| Pooled recommended gap per league, ±2.5 pt | about 4 weeks (NFL ~370 legs a week post-send, NBA ~250, NHL ~100) | 4 weeks after the versions stabilise |
| Per-cell recommended gap (NFL receptions ~80 legs a week) | ±5.5 pt at 4 weeks, ±2.5 pt over a season | season |
| Gate-2 30-day read per version | 200 settled offers in the window | 30 days after each send (10-18 versions: 11-17) |
| I6g version age | ≥ 5 fresh and 10 aged game days, ≥ 300 recommended legs per arm | NHL 10-29, NBA 11-05 |
| Weekly tail scorecard (`admin tail-scorecard --parquet`) | Mondays into the lane folder; Spearman ρ offline vs live per-cell gaps | from 10-26; promotion after four agreeing weeks |

Mechanisms are tested now on the data in hand (§4); live data grades the fixes.

## 7. Quick fixes, graded

| candidate | verdict | why |
|---|---|---|
| Instrumentation (§4.2) | do now | no model change, no clock reset |
| Serve-side parity repairs (E5) | do after the read, this week | serve-side, no retrain |
| Train fraction / refit on all (H1) | test first (E1), ship in the batch | lever-T class if the gain is real; one constant plus a refit step |
| Blind confirm walk (H2) | ship in the batch | protocol; changes no served probability by itself |
| Count-quote producer (H3, I6d) | needs the brief | both archive columns are wrong; the design is not obvious |
| Gate-1 margin | not a fix | `devel` serves ties by the owner's policy; `main` requires the live pass |
| Side rule, edge floor, selection shrink, trust layer | not retried | killed verdicts |

## 8. Owner asks (one each)

1. Go on week 1: the brief, the instrumentation push and E1–E5 (answered 2026-10-21: go; the owner
   kicks off the week-1 fixes).
2. Sequencing of the model fixes: batch into one retrain about 11-06 (default) or ship each as proven.
3. On E2's result: whether cells whose blind re-gating fails leave `devel` (the owner's file; Gate 2 alone
   never removes a cell).
4. Gate 1's ship CI: today a 95 % iid bootstrap whose upper bound must sit under .005. Of the 39 served
   cells with a computable Gate 1, all 39 pass under it, 38 under a date-clustered CI and 34–35 under a
   two-way (date × player) CI: NBA RA, NFL attempts, NFL yards and WNBA PR flip, and MLB runs allowed sits
   on the bar. Keep iid, or move the ship CI to date-clustered or two-way? Not part of H2's rule; E2 reports
   the counts on both halves.

## 9. Records

- Week-1 lane folder: `~/backups/sportstradamus/2026-10-21-live-gap-2/` (README, research brief, one
  folder per experiment, weekly tail scorecards under `live/`).
- 2026-10-21 Under read: `~/backups/sportstradamus/2026-10-21-under-read/REPORT.md` (scripts and outputs).
- 2026-10-21 code surveys (split and selection; serve-time persistence) are summarised in §2; the facts were
  re-verified against `training/pipeline.py`, `model_strategy/confirm.py`, `training/scorecard.py` and the
  cells' matrix dates before this brief was written.
- Lever T campaign: `~/backups/sportstradamus/2026-10-17-train-live-gap/REPORT.md`.
