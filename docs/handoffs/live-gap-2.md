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
   95 % bootstrap CI (iid when this was verified; game-date resampling since §8 ask 4). 31 of the 70 served cells have no computable Gate 1 and pass it blank; NFL
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
5. **What serving persists.** `prediction/feature_log.py` writes one row per scored player, game
   and platform with the frame fed to the model, the decision-time book leg (`Quote Over Prob`, `Quote
   EV`), the raw outputs and the pre-blend projection; partitions older than `FEATURE_LOG_RETENTION_DAYS`
   (400 since week 1, 30 before) are deleted at the end of `prophecize`. `scripts/sync_from_prod.sh` pulls
   `data/runtime/` without deleting and the dev crontab runs it weekly ([OPERATIONS.md](../OPERATIONS.md)),
   so the dev box keeps every partition (dev holds 2026-10-05 on). `history.parquet` holds the blended
   `Win Prob`, `Market Prob` at the DFS line, `Consensus Line`, `Model Weight`, `Temperature`, the
   blended `Gate`, both payouts and the decision-time columns (`history_schema.DECISION_COLS`, since
   2026-10-15). Week 1 added `Standalone Prob` (the model's own probability of the chosen side at the
   offer line, before the blend and calibration), `Served Skew` and `Skew Cal` (a SkewNormal `Win Prob`
   can now be rebuilt from the row), `Hist Gate`, `Market Prob Implied` and `First Scored At`; rows
   written before hold NaN. History still lacks the book probability at the book's own line and the
   per-book quotes, and keeps 365 days (`prediction/cli.py`) against the log's 400. `meditate` keeps a
   replaced model file and its test set 400 days under `data/models/prior/` and `data/test_sets/prior/`
   (dev only; production receives models by rsync and has no prior directory).
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
| H5 | Version age: a fresh version is more honest than an aged one | I6g | Already pre-registered in honest-receipts §6 (one-sided α = 0.05; ≥ 5 fresh and 10 aged game days; ≥ 300 recommended legs per arm); read when the floors clear, dates in honest-receipts §6 item 2 |

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

**Week-1 verdicts** (every number has its script in the lane folder's `REPORT.md`):

| # | Statistic | Value | Verdict |
|---|---|---|---|
| H1 | E1: d̄ of arm C against arm A on W | 0 wins, 0 losses of 5 cells (NFL yards +.0022 [−.0001, +.0044]; the other four within ±.0002) | Not a lever: the refit protocol is not adopted |
| H2 | E2: the served booster with cross-fit calibrators | 60 of 64 cells ship re-gated on the validation half, 63 on the test half, 64 stored; the test half reads easier on Gates 6 and 5 (34 and 29 of 44 cells) | Descriptive: §8 asks 3 and 4 |
| H3 | T1: D = L_off − L_live | L_off .102 [.084, .118], L_live .019 [.013, .046], D .083 [.050, .099], p .001 | Holds (Holm .025): I6d opens in week 2, scoped to levelling the historical book leg in every league |
| H4 | T2: BSS of the > .10 band on levelled rows | +.001 [−.106, +.106], p .526 | Retained (Holm .05): no monitor column; the live loss is a population shift, read in E3 (d) |

## 4. Week 1 (2026-10-22 → 10-28): no retrain, no clock reset

**4.1 Research brief.** Done, with R3 as the prior:
[docs/archive/researcher_live_gap_2.md](../archive/researcher_live_gap_2.md) pre-registers the statistics,
populations and Holm family that §3 pins.

**4.2 Instrumentation push (subagent-driven, one module per subagent): on `devel` (37b5c449):**

| item | where | status |
|---|---|---|
| Feature-log retention | `prediction/feature_log.py` | done: `FEATURE_LOG_RETENTION_DAYS` 30 → 400 (~5 MB a day; production has 13 GB free) |
| History columns | `history_schema.py`, `prediction/model_prob.py`, `prediction/offer_records.py`, `prediction/cli.py` | done: the six columns in §2 fact 5, offer-level and never backfilled. `First Scored At` is the stamp of the key's first stored row: a re-score keeps it and a key stored before the column stays NaN. `persist.py` needed no change; its allow-list keeps the six out of `current_offers` |
| Prior model files | `training/prior_models.py` | done: 14 → 400 days on the dev box |
| Test-set dumps per version | `training/prior_models.py` (`keep_prior_model`, called from the dump path) | done as flat files, `data/test_sets/prior/{LEAGUE}_{market}__{version}.csv.gz`, not `prior/<version>/` folders: one glob prunes both prior folders, the name joins a test set to its pickle, and no `*.csv` glob over `test_sets/` matches it |
| Dev-side sync | dev crontab | done: weekly, Monday 06:30 CDT; the line and its keep-the-newest-3-backups rule live in [OPERATIONS.md](../OPERATIONS.md) |
| Recommender stage counts | — | skipped (§8 ask 1): E3 (d) reads the board → recommended funnel from history |

**4.3 Experiments on the data in hand (main session; outputs beside their scripts in the lane folder
`~/backups/sportstradamus/2026-10-21-live-gap-2/`). All five ran; E1–E4's results are §3's week-1
verdicts, E5's is below:**

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
  10-17). `admin feature-parity` flagged only the sd of the raw ΔP, which cannot see a level shift, and
  scored each row through its own version's file, which leaves 4–36 NFL rows per cell; it now carries this
  read's rule (RMS flag, `--fixed-file`, rank). This read flags a
  cell when the CI lower bound of RMS(ΔP) exceeds 1 pp; it scores NFL receptions, receiving yards, rushing
  yards, carries, interceptions, tds and yards through the current pickle on both frames (completions lacks
  42 logged columns, qb yards has no pickle) and NHL through each row's own version, and ranks repairs by
  served RMS (≈ (w/T) × raw) × live volume. A builder mismatch gets a serve-side repair feature by feature;
  a timing gap (as-of game lines or quotes) routes to I6f. No retrain.
  Result: 11 of 15 cells flag (NHL points, assists, goals, shots, hits, skater and goalie fantasy; NFL
  receiving yards, tds, rushing yards, yards). The two builder mismatches are repaired in
  `prediction/scoring.py`: serving fed ±inf where training holds 0 (a 0-target game's yards per target),
  and built every row as of the run's date where training builds it as of the game's (NFL tds: 64 % of
  its gap); each game date is now its own pass. The data gaps are production's NFL FTN/PFR-fed columns,
  0 on every 2026 row until the ingest fix of §8 ask 5, and dev's NHL game lines, defaulted on 72
  team-games (re-read before the NHL retrain). Game lines and NHL comps read at scoring time are timing (I6f). With the builder and
  data repairs, NFL's RMS falls under the alarm (tds .0057, receiving yards .0054, rushing yards .0065,
  yards .0030).

## 5. Weeks 2 and 3

Each model fix ships when it is proven (§8 ask 2): its league retrains on the lever-T campaign's recipe
without the rebuild and goes out with `scripts/sync_to_prod.sh --league L` between board runs. Every send
restarts the cell's live clock, so live reads split by `Model Version`; the 30-day Gate-2 read of the
10-18 versions (about 11-17) covers the cells no fix has replaced by then.

- **Week 2 (10-27 → 11-05; status in the lane `REPORT.md`, section Week 2):** done on dev: the NHL
  game-line re-read and matrix patch (72 team-games moved off the moneyline and total defaults), then the
  NHL retrain; the NFL past-season FTN/PFR heal (860 team-weeks, 535 of them 2024's: the reopen of §8 ask 5
  covers the current season only, and production keeps its past-season zeros, which reach no model since
  production never trains) and the Bengals' running backs, which PFR labels `HB` and the snap-count filter
  dropped (`stats/nfl.py`; production gains their 2026 rows at its next update). Running: the NFL rebuild
  and retrain on the healed logs. Building: the blind confirm walk (every nominee ranked on the cross-fit
  half at full HPO, the test verdict read once for the chosen nominee, whose booster `meditate
  --params-from` pins); I6d's research brief, then the book-leg levelling it designs (H3; opening note in
  the lane folder's `research/`). Held for a serve-path parity read: the 2026 PBP season aggregates of the
  NFL quarterbacks' comp profiles (`stats/nfl_pbp_agg.py` retries the current season without the
  participation file nflverse does not serve, never a past season). The H1 refit is out. I6g reads wait
  for their floors (honest-receipts §6 item 2).
- **Week 3 on:** each fix retrains and ships as it is proven; a version then holds four weeks for its live
  verdict.

## 6. Live reads: what each verdict needs

| read | needs | when |
|---|---|---|
| Pooled recommended gap per league, ±2.5 pt | about 4 weeks (NFL ~370 legs a week post-send, NBA ~250, NHL ~100) | 4 weeks after the versions stabilise |
| Per-cell recommended gap (NFL receptions ~80 legs a week) | ±5.5 pt at 4 weeks, ±2.5 pt over a season | season |
| Gate-2 30-day read per version | 200 settled offers in the window | 30 days after each send (10-18 versions: 11-17) |
| I6g version age | ≥ 5 fresh and 10 aged game days, ≥ 300 recommended legs per arm | when the floors clear (honest-receipts §6 item 2) |
| Weekly tail scorecard (`admin tail-scorecard --parquet`) | Mondays into the lane folder; Spearman ρ offline vs live per-cell gaps | from 10-26; promotion after four agreeing weeks |

Mechanisms are tested now on the data in hand (§4); live data grades the fixes.

## 7. Quick fixes, graded

| candidate | verdict | why |
|---|---|---|
| Instrumentation (§4.2) | do now | no model change, no clock reset |
| Serve-side parity repairs (E5) | done in week 1: ±inf zeroed, features built as of each game date (`prediction/scoring.py`) | serve-side, no retrain; E5's FTN/PFR data gap is fixed (§8 ask 5), its timing gaps are I6f |
| Train fraction / refit on all (H1) | not a lever | E1: no cell's forward Brier moved (0 of 5) |
| Blind confirm walk (H2) | ship when proven | protocol; changes no served probability by itself |
| Book-leg levelling (H3, I6d) | brief first, week 2 | H3 holds: offline skill on the count cells is the training book leg's level error, and NHL's priced rows show the same; the design is not obvious |
| Gate-1 margin | not a fix | `devel` serves ties by the owner's policy; `main` requires the live pass |
| Side rule, edge floor, selection shrink, trust layer | not retried | killed verdicts |

## 8. Owner asks (one each)

1. Go on week 1: the brief, the instrumentation push and E1–E5 (answered 2026-10-21: go. Claude pushes
   week 1 to `devel` in a job gap after the refactoring-specialist and the gates; the recommender
   stage-count sink is skipped; the weekly dev sync keeps the newest 3 archive backups; the brief's four
   amendments are adopted).
2. Sequencing of the model fixes: batch into one retrain about 11-06 (default) or ship each as proven
   (answered 2026-10-23: ship each as proven; §5).
3. On E2's result: whether cells whose blind re-gating fails leave `devel` (the owner's file; Gate 2 alone
   never removes a cell). Re-gated on the blind half, 60 of 64 served cells still ship; MLB runs allowed,
   NFL yards, NFL fantasy points (PrizePicks) and NBA FTM fail one gate each. Recommendation: withhold MLB
   runs allowed (fails on both halves) and NFL yards (fails Gate 1 on the blind half, on E1's forward
   window and under a date-clustered CI) now; keep the other two until the week-3 blind walk re-gates
   them. Packet: the lane folder's `e2/PACKET_E2.md` (answered 2026-10-23: yes, all four leave `devel`
   serving, so the board prices their quoted props at the book consensus. Their pickles left
   production's models folder at 16:14 CDT, record and undo line in the lane folder's `pull/`; a withheld
   cell now serves the book consensus instead of leaving the board (`prediction/scoring.py::match_offers`);
   the four `withheld` flips in `stat_meta.json` are the owner's. No sportsbook prices NFL fantasy points,
   so its props, about 200 Sleeper rows a board, left the board with the pickle (`pull/after_pull_board.txt`):
   keep it off, or restore that one pickle and leave it `devel`, is the owner's call).
4. Gate 1's ship CI: today a 95 % iid bootstrap whose upper bound must sit under .005. On E2's 33 cells
   with a computable Gate 1 (validation / test half), iid passes 32 / 33, date-clustered 32 / 31 and the
   two-way estimate (player + date − iid variance) 32 / 31; beyond iid only NFL yards and MLB runs allowed
   fail, the two ask 3 withholds. The brief's two-way bootstrap (28 / 26) multiplies a player draw by a
   date draw, counts row noise about three times, and is not a target. Recommendation: date-clustered,
   from the week-3 walk on (answered 2026-10-23: date-clustered. Gate 1 resamples game dates, and the
   row bootstrap's interval decides when its upper bound is higher: over three dates the date
   bootstrap collapsed and passed withheld NFL qb-yards (`training/scorecard.py`,
   [ship_gate.md](../ship_gate.md)). Of the 39 served cells with a computable Gate 1 on dev's dumps,
   only NFL yards flips, to a fail; `model_stats` takes it at each league's next `meditate`).
5. Production's NFL FTN/PFR-fed columns (first read, breakaway yards, broken tackles, drops, blitz,
   first-read EPA, pressure) are 0 on every 2026 row: ingest writes a source that is not yet published as
   0 and never revisits it. Dev holds real values for most of them, and the models trained on those. The
   sources are published now: re-parse production's 2026 gamelog and teamlog rows for these columns (or
   copy dev's values) ahead of the week-2 ingest fix? It writes production data. Evidence: the lane
   folder's `e5/probes/nfl_classification.md`, group 3 (answered 2026-10-23: find the cause and fix it
   now. Cause: the backfill parses a team-week once, and production ingests within hours of a game,
   before nflverse publishes its FTN charting and weekly PFR rows; PFR also spells the Rams `LA`, which
   zeroed their pressure columns in every season. Fix: each NFL update re-parses this season's
   team-weeks whose FTN- or PFR-fed columns are all zero once that source holds the game
   (`stats/nfl.py::_reopen_late_source_weeks`), and PFR's `LA` reads as `LAR`. Production heals at its
   first NFL update after the deploy: on a copy of its logs the pass re-parsed 186 team-weeks (2,057
   player rows) in about a minute and left zeros only in week 7, which neither source holds yet.
   Past seasons are week 2, §5).

## 9. Records

- Week-1 lane folder: `~/backups/sportstradamus/2026-10-21-live-gap-2/` (README, research brief, one
  folder per experiment, weekly tail scorecards under `live/`).
- 2026-10-21 Under read: `~/backups/sportstradamus/2026-10-21-under-read/REPORT.md` (scripts and outputs).
- 2026-10-21 code surveys (split and selection; serve-time persistence) are summarised in §2; the facts were
  re-verified against `training/pipeline.py`, `model_strategy/confirm.py`, `training/scorecard.py` and the
  cells' matrix dates before this brief was written.
- Lever T campaign: `~/backups/sportstradamus/2026-10-17-train-live-gap/REPORT.md`.
