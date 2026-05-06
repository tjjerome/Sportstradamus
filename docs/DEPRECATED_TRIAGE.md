# Deprecated Code Triage

Decision-only audit of every file in `src/deprecated/` and every TODO at the
bottom of `README.md`. No code is moved or removed in this pass — deletions
happen in a follow-up session after review.

Decision keys:
- **REVIVE** — feature is genuinely missing from the live tree; reintroduce per
  the target phase from `docs/sportstradamus_roadmap_v2.md`.
- **DELETE** — superseded by a live module, or never wired in and not part of
  any roadmap phase. Git history preserves the implementation.
- **ARCHIVE PERMANENTLY** — keep under `src/deprecated/` indefinitely (none in
  this pass).

Owner: tjjerome (default for every entry).

---

## opt_kelley_bet.py
Decision: REVIVE (Phase 3.1)
Replacement exists in production: NO
Rationale: Kelly stake sizing is genuinely missing from the live codebase.
`docs/sportstradamus_roadmap_v2.md` §3.1 ("Kelly sizing module — genuinely
new") explicitly schedules a rewrite at `src/sportstradamus/strategies/kelly.py`
that modernizes this archive into `fractional_kelly_stake` /
`joint_kelly_portfolio` plus a `kelly` CLI. Keep until that work lands.
Owner: tjjerome

## opt_parlay.py
Decision: DELETE
Replacement exists in production: YES — `prediction/correlation.py:beam_search_parlays`
Rationale: Live module supersedes this with a beam-search parlay enumerator
that consumes the offer DataFrame produced by `find_correlation`. The archived
brute-force `combinations`-based optimizer was never wired into any entry
point; the audited live path covers the same problem with a calibrated EV
computation (see `scripts/audit_parlay_calibration.py`). Keep git history only.
Owner: tjjerome

## unused_funcs.py
Decision: DELETE
Replacement exists in production: YES — `prediction/correlation.py:beam_search_parlays`
Rationale: Contains a single function, `find_bets`, an older recursive +EV
parlay search. Same problem space as `opt_parlay.py`; superseded by the same
beam-search pipeline. No call sites anywhere in the package, scripts, or tests.
Keep git history only.
Owner: tjjerome

## correlation.py
Decision: DELETE
Replacement exists in production: YES — `training/correlate.py`
Rationale: Live module supersedes this — `training/correlate.py:correlate`
builds `{LEAGUE}_corr.csv` from player stat history and is invoked by
`training/cli.py` (the `meditate` command) per league setup. Older standalone
script with no caller; keep git history only.
Owner: tjjerome

## get_lines.py
Decision: DELETE
Replacement exists in production: YES — `moneylines.py:get_props`
Rationale: BettingPros NFL prop scraper, fully superseded by the consolidated
Odds API path in `moneylines.get_props` (called by the `confer` CLI). Per the
README, "redundant with `books.py` scrapers, but a useful fallback" — there is
no plan to wire BettingPros back in, and the Odds API already aggregates the
same books. Keep git history only.
Owner: tjjerome

## see_features.py
Decision: DELETE
Replacement exists in production: YES — `training/shap.py:see_features`
Rationale: Live module supersedes this. The archived script printed raw
LightGBM `feature_importance()` for each pickled model; the live `see_features`
is SHAP-based and integrated with `feature_filter.json` management via
`meditate --rebuild-filter`. Older alternative with no caller; keep git history
only.
Owner: tjjerome

## test.py
Decision: DELETE
Replacement exists in production: N/A — was never a real test harness
Rationale: Ad-hoc experimentation scratchpad (commented-out scriptlets and one
dataframe inspection), not pytest tests. The actual test suite lives at
`tests/golden/` and is wired into CI. Nothing to revive; keep git history only.
Owner: tjjerome

## books_deprecated.py
Decision: DELETE
Replacement exists in production: YES — `moneylines.py:get_props`
Rationale: Bundle of seven direct-book scrapers (`get_dk`, `get_fd`,
`get_pinnacle`, `get_caesars`, `get_thrive`, `get_pp`, `get_parp`) all
superseded by the consolidated Odds API in `moneylines.get_props`. The two
scrapers that were kept live (`get_ud`, `get_sleeper`) remain in `books.py`.
There is no plan in the roadmap to revert to direct-book scraping. Keep git
history only.
Owner: tjjerome

## stats_deprecated.py
Decision: DELETE
Replacement exists in production: YES — `stats/base.py` + per-league subclasses
Rationale: 21 orphan methods across the league subclasses. The `obs_*` family
(`obs_get_stats`, `obs_get_training_matrix`, `obs_profile_market`, `dvpoa`,
`bucket_stats`) was a pre-vectorization per-observation API fully superseded by
the current offer-based vectorized API (`get_stats`, `get_training_matrix`,
`profile_market`). `_load_comps` is superseded by the comps loader on
`Stats._build_comps` / `update_player_comps`. `get_fantasy` (StatsNFL) is unused
NFL fantasy scoring with no caller and no roadmap entry. The README floated
"reintroduce if the obs_* API is revived for analysis tooling," but no phase
plans that. Keep git history only.
Owner: tjjerome

## helpers_orphans.py
Decision: DELETE
Replacement exists in production: PARTIAL — `Archive.add_dfs` covers the live
write path for `confer`; the rest have no equivalent and no plan
Rationale: Two groups, both deletable.

(1) Module-level math/util orphans (`get_active_sports`, `prob_diff`,
`prob_sum`, `accel_asc`): never called and not on any roadmap. `prob_diff` /
`prob_sum` are 2D PDF integrals that the project's predictive math takes a
different route around. `accel_asc` is a partition generator for combinatoric
parlay enumeration — same problem space as `opt_parlay.py`, already covered by
`beam_search_parlays`. `get_active_sports` was a sports-availability probe
against the Odds API, never wired in.

(2) De-methodized `Scrape` / `Archive` orphans (`scrape_get_proxy`,
`scrape_post`, `archive_add`, `archive_clip`, `archive_merge`,
`archive_rename_market`): all six have zero callers in the package, scripts, or
tests. `archive_add` was flagged as "the intended write path for `confer`," but
the live write path is `Archive.add_dfs` (called by `prediction/scoring.py`),
and `Archive.write` handles persistence. `archive_clip` / `archive_merge` /
`archive_rename_market` are one-off migration helpers. `scrape_get_proxy`
duplicates ScrapingFish behavior already inside `Scrape`; `scrape_post` mirrors
`Scrape.get` and was never used. Keep git history only.
Owner: tjjerome

---

## Summary

| Decision | Count | Files |
|---|---|---|
| REVIVE | 1 | `opt_kelley_bet.py` (Phase 3.1) |
| DELETE | 9 | `opt_parlay.py`, `unused_funcs.py`, `correlation.py`, `get_lines.py`, `see_features.py`, `test.py`, `books_deprecated.py`, `stats_deprecated.py`, `helpers_orphans.py` |
| ARCHIVE PERMANENTLY | 0 | — |

Total files reviewed: 10. The single survivor is the Kelly sizing archive,
held until `docs/sportstradamus_roadmap_v2.md` §3.1 lands. Every other archive
either has a live successor or is a stale scratchpad with no roadmap entry.
