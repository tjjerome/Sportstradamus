# Lane brief — live-gap-2: why the models pass the offline gates and lose live

**Read first:** CLAUDE.md, [CONTRIBUTING.md](../../CONTRIBUTING.md),
[docs/ARCHITECTURE.md](../ARCHITECTURE.md), [docs/ship_gate.md](../ship_gate.md),
[honest-receipts.md](honest-receipts.md) §5–§6 (the three research briefs and the I6 instruments),
[docs/archive/researcher_train_serve_skew.md](../archive/researcher_train_serve_skew.md) (R3, the
decomposition this lane builds on) and [nfl-ship15-recovery.md](nfl-ship15-recovery.md) §11 (lever T).
This brief is self-contained but those documents are the contract it works under.

Status: OPEN (opened 2026-10-21 on the owner's ask; week 1 starts on the owner's go). The owner's
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
   same rows. No walk-forward or forward-in-time holdout exists anywhere; `--holdout-blind` is a
   player-disjoint cross-fit over the validation half, used by the sweep only. Gate 1 is a
   non-inferiority test (`scorecard._GATE1_NONINF_MARGIN` lets the ensemble be 0.005 Brier worse than
   the book) and a blank Gate 1 passes (NFL yards: 19 authentic rows). So "pass offline" means tie-or-better
   on a multiply-inspected random half of the same period, while "fail live" means a strictly negative
   30-day Brier skill, forward in time, on the props the platforms post.
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

## 3. Hypotheses and decision rules (pre-registered before the first experiment)

| # | Hypothesis | Experiment (§4) | Pre-registered rule |
|---|---|---|---|
| H1 | The stale booster costs live skill: training on the most recent 30 % raises forward Brier skill | E1 | On the same last-10 % rows, the 0.85-fraction (or refit-on-all) model beats the 0.7 model on Brier-vs-book with a day-clustered 95 % CI above 0 on at least 3 of 5 cells and is worse beyond its CI on none → adopt the refit protocol in the week-3 batch; else H1 is not a lever |
| H2 | The gate verdicts are optimistic: the first-to-pass walk and the same-period split inflate `ship` | E2 | For every shipped cell, gates on the blind cross-fit half vs the test verdict; the count of `ship` flips sizes the optimism. The blind confirm walk is adopted regardless (protocol); whether re-gated cells leave `devel` is the owner's call |
| H3 | The count cells' training book leg is the defect: Gate 1 ties a broken opponent and the blend weight follows | E3 | On NFL count cells, the offline book probability's level error against outcomes exceeds 5 points while the live `Market Prob` is within 2 → the producer is the defect → the I6d brief (count-quote levelling) opens; else the count cells route to H4 |
| H4 | The gate population hides the disagreement band: offline Gate 1 scores rows where model and market agree | E4 | Offline Gate-1 BSS on test rows with \|P − P_book\| > 0.10 is negative like live (−.115) → propose the band read as a monitor column (never a gate; owner packet); else the live loss is a population shift and E3's population split carries the verdict |
| H5 | Version age: a fresh version is more honest than an aged one | I6g | Already pre-registered in honest-receipts §6 (one-sided α = 0.05; ≥ 5 fresh and 10 aged game days; ≥ 300 recommended legs per arm): NHL about 10-29, NBA about 11-05 |

Killed levers are not retried without new evidence (memory `research_verdicts_ledger`): the trust
layer, selection shrink toward the book, a higher edge floor, the argmax-EV side rule, price-blind tail
recalibration, the post-fusion mean corrector as built.

## 4. Week 1 (2026-10-22 → 10-28): no retrain, no clock reset

**4.1 Research brief.** Dispatch `research-analyst` with §2 and §3 verbatim and R3 as the prior; it
pre-registers the statistics, the candidate sets and the Holm family for E1–E4 and writes
`/tmp/researcher_live_gap_2.md` (copy into the lane folder; `/tmp` does not survive a restart).

**4.2 Instrumentation push (subagent-driven, one module per subagent; ships to `devel` in a job gap):**

| item | where | change |
|---|---|---|
| Feature-log retention | `prediction/feature_log.py` (the 30-day constant; the prune runs last in `prophecize`) | keep the season (400 days); ~5 MB a day, production has 13 GB free |
| History columns | `history_schema.py` + `prediction/offer_records.py` + `prediction/persist.py` | add the pre-blend model probability of the chosen side at the offer line, the blended skew and `skew_cal`, `Hist Gate`, a quoted-vs-payout-implied flag on `Market Prob`, and `First Scored At` (the minimum on upsert); `DECISION_COLS` rules: offer-level, never backfilled, NaN on older rows |
| Prior model files | `training/prior_models.py` | keep the season on the dev box (14 → 400 days; 109 files are 173 MB today) |
| Test-set dumps per version | `training/pipeline.py` (the dump path) | copy each cell's CSV to `data/test_sets/prior/<model version>/` at retrain, so offline reads can be re-run per version |
| Dev-side sync | dev crontab | `scripts/sync_from_prod.sh` weekly, so no partition is ever pruned before the dev copy exists |
| Recommender stage counts | `prediction/` (the recommendation rule's call site) | optional: one line per board run with the legs surviving each filter per cell and side |

**4.3 Experiments on the data in hand (main session; outputs beside their scripts in the lane folder):**

- **E1 train fraction / refit** (decides H1): a sandbox retrain of NFL receptions, yards, carries and NHL
  shots, points at `_TRAIN_FRACTION` 0.85 and as a refit on all rows with the 0.7 model's hyperparameters,
  through `--artifact-output` and `--frozen-matrix-dir` in a throwaway worktree (the constant needs a flag
  or an environment override; build it in the worktree, never on the served pickles). Score both models on
  the same last-10 % rows: Brier vs book, the tail scorecard's selected gap, Gate 4. About ten minutes a
  cell for the NFL.
- **E2 gate optimism** (decides H2): for the served cells, gates on `--holdout-blind` cross-fit dumps
  against the test verdicts; count the `ship` flips and the Gate-1 CI shifts.
- **E3 same-rows decomposition**: the live window's settled rows (feature log joined to history on
  League, Market, Platform, Player, Date) re-scored through the offline path: model probability parity
  (`admin feature-parity` self-check and `--versions`), the book-leg parity (offline `Odds` / `Book_EV`
  against live `Market Prob` and `Quote Over Prob` at the same line, both against outcomes), the population
  split (sportsbook-quoted main lines vs alt rungs vs unquoted) and the label (push). R3's
  `c16_parity.py` and `c13_bookonly.py` under `~/backups/sportstradamus/2026-10-04-honest-receipts/r3/`
  are the starting scripts.
- **E4 disagreement-band gates** (decides H4): Gate-1 BSS on the test rows by \|P − P_book\| band, beside the
  live table in `~/backups/sportstradamus/2026-10-21-under-read/`.
- **E5 I6e parity read**: the training matrices now hold the logged games of 2026-10-05 → 10-12, so the
  first real serve-vs-training parity read is possible: `admin feature-parity --league NFL` (and NHL)
  over those dates; cells above 1 pp sd(ΔP) get a serve-side repair feature by feature (R3's worst: NFL
  completions, receptions, interceptions, qb yards). Serve-side only, no retrain.

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

## 9. Records

- 2026-10-21 Under read: `~/backups/sportstradamus/2026-10-21-under-read/REPORT.md` (scripts and outputs).
- 2026-10-21 code surveys (split and selection; serve-time persistence) are summarised in §2; the facts were
  re-verified against `training/pipeline.py`, `model_strategy/confirm.py`, `training/scorecard.py` and the
  cells' matrix dates before this brief was written.
- Lever T campaign: `~/backups/sportstradamus/2026-10-17-train-live-gap/REPORT.md`.
