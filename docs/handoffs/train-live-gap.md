# Lane brief — train/live gap: offline metrics pass, live picks lose

**Read first:** CLAUDE.md, [honest-receipts.md](honest-receipts.md) §1, §5, §6,
[model_improvement_track.md](model_improvement_track.md) §8.2 (holes #0c, #0e, #10, #13),
[nfl-ship15-recovery.md](nfl-ship15-recovery.md) §11 (lever T's acceptance and kill triggers),
[MODEL_LIFECYCLE.md](../MODEL_LIFECYCLE.md) § Rebuilding a league's matrices from scratch, memory
notes `research_verdicts_ledger`, `nfl_star_lean_bookless_cells`, `nfl_cold_matrix_rebuild_blocked`.
Opened 2026-10-17: why are training metrics and realized profits so far apart? Lever T is built
on branch `lever-t` (2026-10-17); §4 carries the program state.

## 1. Measured

- Live: recommended legs (edge at or above the story floor, payout at most 2.5×) lose 6.6 to 9 %
  per pick in every edge band over 30 days, Overs about −15 %, Unders −7 %; the cohort hits .50
  against a read of .62 to .65; the platforms' cut is 6.5 to 12 % per pick
  (`realized.settled_offers`;
  `~/backups/sportstradamus/2026-10-16-under-lean/part1b/floor_cost.py`).
- Offline: every served cell passes gates 1 to 6 on its test set and reads calibrated. The
  like-for-like metric is `precision_over`, never `predicted_over_rate`.
- Decomposition (R3, 2026-10-04): 89 % of the 12.5 pp read-minus-hit gap is a selection residual
  (selection harvests the most positive errors; encompassing +0.43), 11 % bulk miscalibration;
  train/serve skews at most 0.7 pp. `sportstradamus admin tail-scorecard` replays the live rule on
  test rows and reproduces the gap; the ship gates grade the whole test set, not the selected tail.
- The star lean (NFL served P(Under) 4 to 5 points high on top-quartile lines, +4.6 pts pooled)
  is the outcome trim of §2 (hole #13), not recency and not the offered population.

## 2. Lever T: the outcome trim

`training/data.py:trim_matrix` keeps a row only if its `Result` lies inside the cell's 5th to
95th percentile (`_RESULT_OUTLIER_LOW` / `_HIGH`), unless the row carries a sportsbook quote
(`Archived == 1`). In a book-less cell (DFS-only `pickem` rows) the top 5 % of outcomes leave
training and the test set. Stars produce most of them, so the model learns a capped
distribution: its mean on a high line runs low, P(Under) reads high, and the gates grade the same
truncated rows and pass. Quoted rows never trim, so the lean survives only where no quote blends
it out. 43 served cells are capped (NFL 14, NBA 13, NHL 9, WNBA 5, MLB 2; NBA FTM at 6 against
15.18, NHL hits at 4 against 8); NHL hits and WNBA fit the mechanism (inferred, untested).

The change, about 20 lines in one shared function: drop the `Result` band from `trim_matrix`;
make `_balance_over_under` outcome-blind (it drops pushes and deletes rows to move the over rate
toward the quoted rows', so with the trimmed rows back it would delete the restored star Over
games); keep the line clip and the line balancing, which read the line, not the outcome. The code
change alone changes nothing: the trimmed rows are gone from the cached matrices, so each league
needs its cold rebuild from the existing game logs and odds archive (no re-scrape; NFL 6 to 10 h
by the MODEL_LIFECYCLE bootstrap, volume cells first), then `meditate --league NFL --force`
(about 1.75 h) and `scripts/sync_to_prod.sh --league NFL`. T removes a mechanical tilt against
stars' Overs; the selection residual of §1 stays for step 2.

## 3. Killed (do not retry without new evidence)

Shrinking toward the book (λ_sel ≈ −0.10), a trust layer, a higher edge floor, the
post-fusion mean corrector as built, recency weighting, offered-population training, the consensus
as an engine. Owner rules: never pull a model on realized money; a DFS main line is never a book
leg; no model-free engine.

## 4. Plan (in order; one yes per step)

1. Lever T, NFL first. Before any code: copy today's NFL test sets and matrices aside and run
   `admin tail-scorecard --league NFL`. A failing test on `trim_matrix` first (an unquoted row
   above the 95th percentile survives), then the change, the rebuild, the retrain. Acceptance A1
   to A5 and kill triggers K1 to K3 are in nfl-ship15 §11; A1 is
   `tests/diagnostics/test_star_lean.py` from xfail to XPASS. The retrained models wait for the
   owner's yes before `devel` and production. Then the other leagues, each at its own rebuild.
   State 2026-10-17: built on `lever-t` (`tests/golden/test_trim_matrix.py` pins it); one cold
   rebuild and one retrain each for NFL, NBA and NHL this weekend; WNBA and MLB at their 2027
   preseason.
2. Make the offline gate population the live one: a per-cell selected-tail column (read minus hit,
   day-block interval) in `model_stats.parquet`, read by the sweep's selection and by
   `ship graduation`. Pre-register its bar.
3. The at-market information gap (hole #0e; the I6g version-age test is due late October):
   measure encompassing per cell first.
4. Metric hygiene: the push label, strict `>` on integer lines, count-once windows.

Rules: CLAUDE.md and honest-receipts.md §2 apply, and the owner's yes comes before anything that
changes predictions reaches `devel`.
