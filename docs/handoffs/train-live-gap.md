# Lane brief — train/live gap: offline metrics pass, live picks lose

**Read first:** CLAUDE.md, [honest-receipts.md](honest-receipts.md) §1, §5, §6,
[model_improvement_track.md](model_improvement_track.md) §8.2 (holes #0c, #0e, #10, #13),
[nfl-ship15-recovery.md](nfl-ship15-recovery.md) §11, memory note `research_verdicts_ledger`.
Opened 2026-10-17: why are training metrics and realized profits so far apart?

## 1. Measured

- Live: recommended legs (edge at or above the story floor at the platform payout, payout at most
  2.5×) lose 6.6 to 9 % per pick in every edge band over 30 days, Overs about −15 %, Unders −7 %;
  the cohort hits .50 against a read of .62 to .65; the platforms' cut is 6.5 to 12 % per pick
  (`realized.settled_offers`;
  `~/backups/sportstradamus/2026-10-16-under-lean/part1b/floor_cost.py`).
- Offline: every served cell passes gates 1 to 6 on its test set and reads calibrated. The
  like-for-like metric is `precision_over`, never `predicted_over_rate`.
- Decomposition (R3, 2026-10-04): 89 % of the 12.5 pp read-minus-hit gap is a selection residual
  (selection harvests the most positive errors; the served mean adds little to the line,
  encompassing +0.43), 11 % bulk miscalibration; train/serve skews at most 0.7 pp.
  `sportstradamus admin tail-scorecard` replays the live rule on test rows and reproduces the live
  gap; the ship gates grade the whole test set, not the selected tail.
- New (2026-10-17, hole #13): `training/data.py:trim_matrix` drops unquoted rows whose result falls
  outside the 5th to 95th percentile, so training and test rows of book-less cells are
  outcome-truncated and the gates pass models that run 4 to 5 points high on stars' Unders live;
  43 served cells in five leagues are capped.

## 2. Killed (do not retry without new evidence)

Shrinking toward the book (λ_sel ≈ −0.10), a trust layer, a higher edge floor, the
post-fusion mean corrector as built, recency weighting, offered-population training, the consensus
as an engine. Owner rules: never pull a model on realized money; a DFS main line is never a book
leg; no model-free engine.

## 3. Plan (in order; one yes per step)

1. Lever T, outcome-blind trimming (nfl-ship15 §11), NFL first: about 20 lines in one shared
   function, the NFL cold rebuild (6 to 10 h), `meditate --league NFL --force`,
   `scripts/sync_to_prod.sh --league NFL`; acceptance `tests/diagnostics/test_star_lean.py` from
   xfail to XPASS plus A1 to A5 and K1 to K3. Then the other leagues. T removes a mechanical
   tilt, not the cohort's losses.
2. Make the offline gate population the live one: a per-cell selected-tail column (read minus hit,
   day-block interval) in `model_stats.parquet`, read by the sweep's selection and by
   `ship graduation`. Pre-register its bar.
3. The at-market information gap (hole #0e; the I6g version-age test is due late October):
   measure encompassing per cell first.
4. Metric hygiene: the push label, strict `>` on integer lines, count-once windows.

Rules: CLAUDE.md and honest-receipts.md §2 apply (Opus subagents, a `research-analyst` brief before
any §8.2 lever, one module per subagent, `git commit -F <msg> -- <paths>`, no stash, checkout,
restore or reset, detached-worktree verification, production read-only, no training run while a
retrain holds the box, the owner's yes before anything that changes predictions reaches `devel`).
