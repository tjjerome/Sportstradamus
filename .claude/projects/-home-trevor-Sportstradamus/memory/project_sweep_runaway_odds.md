---
name: project_sweep_runaway_odds
description: Global archive blown-row sweep + the latest-line/negative-line false-positive fix (5×MAX(line>0) clamp invariant)
metadata:
  type: project
---

`scripts/sweep_runaway_odds.py` (entry point `sweep-runaway-odds`, prod form
`python -m sportstradamus.scripts.sweep_runaway_odds --apply`) is the one-shot,
all-cells post-deploy cleanup of magnitude-blown `ev` rows. Dry-run by default;
`--apply` backs up then deletes; holds `run_job.sh`'s `/tmp/sportstradamus-archive.lock`
flock. Reuses `delete_corrupt_seed.BLOWN_PREDICATE` (promoted `_BLOWN_PREDICATE`→public for this).

**The predicate's line arm MUST key on `5 × MAX(l.line) WHERE l.line>0`, not the
latest line.** Lifting `delete_corrupt_seed`'s original latest-line predicate to a
global sweep produced two false-positive classes that `--apply` would have deleted
as good consensus rows:

- **Multi-book line disagreement.** The `lines` table has no book column, so "latest
  line" is whichever book wrote last. When 4 sportsbooks price FG3M at 2.5 (ev correctly
  clamped to 12.5) and Sleeper later writes a 1.5 line, every book's correct 12.5 trips
  `12.5 > 5×1.5`. On the dev archive this was 152 of 155 flagged rows — all legit.
- **Negative-line team markets** (run line −1.5, spreads): `ev > 5×(negative)` flags anything.

**Why MAX(line>0) is principled, not a magic cut:** the fixed `get_ev` caps every write at
`BLOWN_LINE_FACTOR(5) × its own line` (= `SN_MAX_MEAN_FACTOR`) and records that line, so
`max_line ≥ every legit ev's line` ⇒ no post-fix row can exceed `5×max_line`. Anything above
is pre-fix residue. The `line>0` filter (MAX returns NULL → line-arm off) drops negative-line
markets; they fall back to the absolute `ev>2000` arm. Verified on `archive.duckdb.bak-1780701767`
(Jun-3, corruption present): MAX-arm still catches 87,816 genuine NBA STL runaway, spares 3,163
multi-book false positives. Absolute-only would MISS 65,011 sub-2000 STL runaway — so the line
arm is required, just robustified.

Prod repair order is load-bearing: deploy the `get_ev` clamp (devel `854884e`) FIRST, then
`--apply` the sweep once — an old-encode `confer` repopulates runaway faster than you delete it.
`sync_from_prod.sh` stays unsafe until both done. See [[project_archive_repair_outcome]],
[[project_passing_book_degenerate]], [[add_dfs_pins_calibration_coupled]].
