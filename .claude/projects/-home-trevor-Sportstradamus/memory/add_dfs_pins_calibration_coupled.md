---
name: add_dfs_pins_calibration_coupled
description: add_dfs golden EV pins flap on concurrent meditate; pin clamp invariants for calibrated cells, exact only for config-missing cells
metadata:
  type: project
---

`tests/golden/test_archive_closing_line_add_dfs_characterization.py::test_add_dfs_dedup_market_resolution_and_ev`
exercises the REAL `get_ev`/`book_gate` against live config, so any EV it point-pins for a
**calibrated** cell drifts whenever `meditate` runs. `stat_calibration.json` is gitignored and
recomputed every `meditate`; a concurrent training agent rewriting it silently breaks exact pins.

**Rule:** pin the clamp invariant (`0 < ev <= SN_MAX_MEAN_FACTOR * line`) for cells that carry a
calibration entry; pin exact values only for config-missing cells (`stat_dist`/`stat_cv`/calib all
None → deterministic default SkewNormal/cv=1).

Worked example (2026-06-06): NBA BLK (ZINB, gate `zi=0.626`) encoded to the clamp `7.5 = 5×1.5`,
NHL points (SkewNormal, cv=0.02) to `1.394` — both moved with the 10:55 meditate rewrite → invariant
pins. WNBA `prizepicks_pts` is config-missing → deterministic; it only drifted ~0.08%
(`15.148→15.136`) as binning fallout from the `get_ev` rewrite, so it stays an exact pin (just
re-pinned). Don't chase these one assert at a time — dump the full `add_dfs` output once, classify
each cell calibrated-vs-config-missing, fix all pins together.

Structural asserts (dedup, market resolution BLK→BLK / PTS→points / underdog→prizepicks, accent
stripping, staged `lines`) are calibration-independent — keep them exact. See
[[project_sweep_runaway_odds]], [[integration_test_mutates_stat_meta]].
