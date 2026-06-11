# bestball-2027 — Underdog draft products (Best Ball, Battle Royale)

> Status: BLOCKED (on: D4 — roadmap v3 §7; foundations may start late 2026 at owner's call)

## 1. Mission & money logic

Build the decision stack for Underdog's draft products — Best Ball (season-long
18-round tournaments) and Battle Royale (single-week snake drafts) — in time for
the 2027 draft season. Nothing draft-related is live today: the legacy `drafts/`
package was archived whole to `src/deprecated/drafts/`, so this lane starts
from 0% live code with archived reference implementations.

Draft products are a separate prize ecology from pick'em. The edge is
ADP/advance-equity arbitrage: draft players whose advance equity exceeds their
ADP cost — Week-17-weighted value, mandatory stacking — rather than per-leg line
arbitrage ([underdog_edge_suite.md §4.3–§4.4](../underdog_edge_suite.md)). It
deploys capital in the NFL offseason when pick'em volume dips (Best Ball drafts
run ~Mar–Aug; Battle Royale weekly in season — archived v2 §Phase 5). Honest
framing from the vision doc: roughly +5–15% Best Ball ROI for sharp recreational
play, with high variance and settlement deferred to season end
([underdog_edge_suite.md §10](../underdog_edge_suite.md)).

## 2. Read first (in order)

1. [CONTRIBUTING.md §Package Map](../../CONTRIBUTING.md) — where the new
   `drafts` package must register; helpers to reuse (`Scrape`, config loaders).
2. [roadmap v3 §5 + §7](../sportstradamus_roadmap_v3.md) — seasonality window
   and the D4 gate this lane is blocked on.
3. [archive/sportstradamus_roadmap_v2.md §Phase 5](../archive/sportstradamus_roadmap_v2.md)
   — §5.1–§5.6 hold this lane's full acceptance criteria (spec of record).
4. [underdog_edge_suite.md §4.3–§4.4, §5.1](../underdog_edge_suite.md) —
   strategy framing for draft products. Non-normative vision; orientation only.
5. `src/deprecated/drafts/` — archived legacy package (`update_ez_adp.py`,
   `forecast.py`, `data_process.py`, `data_merge.py`, `train.py`); stage-0
   retire-vs-evolve input, not a base to build on by default.
6. `docs/DEPRECATED_TRIAGE.md` — hygiene-closeout Phase-1.6 verdicts feed
   stage 0; may not exist yet (§3 checks).
7. `src/sportstradamus/strategies/kelly.py` — the money conventions this lane
   extends (`Decimal` stakes, per-entry caps).

## 3. Verify before you trust

Rule: if command output contradicts brief prose, the output wins — fix the
brief in place (minor) or stop and ask the owner (material).

```bash
git fetch origin && git log --oneline origin/devel -3

# D4 status (owner-only gate; resolves by owner commit) + calendar
grep -n "D4" docs/sportstradamus_roadmap_v3.md
date +%F   # unblocks ≥ ~Nov 2026; math must land before 2027 drafts open (~Mar 2027)

# Model-track health (D4 input) — shipped counts per league
python3 -c "import json,collections; m=json.load(open('src/sportstradamus/data/config/stat_meta.json')); [print(l, dict(collections.Counter(c['shipped'] for c in v.values()))) for l,v in m.items()]"
ls -la data/training/model_stats.csv      # per-cell gate numbers fresh?

# Hygiene-closeout triage verdicts (feed stage 0); absent until Phase 1.6 runs
ls docs/DEPRECATED_TRIAGE.md

# Nothing draft-related is live; legacy package fully archived
ls src/sportstradamus/drafts    # expect: No such file or directory
ls src/deprecated/drafts/       # expect five legacy .py + __init__.py + png
```

The archived v2 status table calls Phase 5 "~15% (legacy shape)" — stale: it
predates the archival sweep. Live draft code is 0%.

### Volatile product assumptions

External facts this lane's math depends on. On drift: stop, re-run stage-0
re-verification, revise this brief in place, then resume.

- **Best Ball contest structure** — 18-round rosters with position min/max,
  12-team pods, advance rules (top-2-of-12 after the 14-week regular season,
  then single-week eliminations weeks 15/16/17), payout curve, rake. These
  shift yearly (edge-suite §4.3 flags this explicitly). Re-verify against the
  live Underdog lobby/rules pages each season **before building math on them**;
  capture verified rules in `src/sportstradamus/data/contests/{contest}.json`
  (stage-0/2 artifact).
- **ADP source endpoints** — Underdog publishes pick distributions
  (edge-suite §4.3); the legacy fetcher hit
  `stats.underdogfantasy.com/v1/slates/...` (see
  `src/deprecated/drafts/update_ez_adp.py`). Verify availability and response
  shape before stage 1; manual-CSV fallback exists (§6 stage 1).
- **Battle Royale rules** — single-week 6-player snake draft against a large
  field; top-heavy payouts (~70% of pool to top ~1.5%, edge-suite §4.4).
  Re-verify slate cadence, roster slots, and entry caps each season.
- **ToS surface** — Underdog ToS prohibits automated scraping and bots
  (edge-suite §Reality Check). Data collection is observation-only; never
  programmatic entry. Scraping decisions are owner-only (§8).

## 4. Locked decisions

All locked 2026-06-10 by the owner. Sessions may not relitigate; changes are
owner-only.

- **Never schedules ahead of the model track.** Model correctness leads
  (roadmap v3 §1); this lane yields whenever it competes with `model-track`
  for session time or review bandwidth.
- **2027 season is the target; 2026 explicitly skipped.** No partial-2026
  entries even if foundations land early.
- **Flat-stake sizing for tournament entries, NOT Kelly.** Kelly is unstable
  for steep payout curves with uncertain win probabilities (edge-suite §5.1);
  cap per-entry exposure instead. Do not wire `strategies/kelly.py` sizing
  into draft entries.
- **All money is `Decimal`**, matching the existing `strategies/` convention
  (`kelly.py`, `underdog_pickem.py`).

## 5. Module footprint & canonical paths

- **NEW package `sportstradamus.drafts`** (`src/sportstradamus/drafts/`) — all
  lane code lives here. Modules track stages: `adp.py` (stage 1),
  `projections.py` (stage 2), then Battle Royale / advance-equity / companion /
  exposure modules (archived spec names where given). Adding the package
  requires a CONTRIBUTING §Package Map row at stage 1 — flag for owner review.
- **Read-only model access** — existing LightGBMLSS pickles via
  `sportstradamus.training` interfaces. The projection layer is a TRANSLATION
  layer; no new model (archived v2 §5.2). Never edit serving-path modules from
  this lane; compat notes:
  [operation_ship_references.md](../operation_ship_references.md).
- `sportstradamus.helpers` — `Scrape` for ADP HTTP, config loaders. Reuse,
  don't fork.
- `src/sportstradamus/pages/` — companion/exposure views (stages 5–6). Parquet
  snapshots only, never DuckDB (CLAUDE.md §Hard rules).
- `pyproject.toml` — CLI registration (`drafts-adp-update`,
  `battle-royale-build`, `draft-recommend`, `drafts-exposure`).
- `tests/golden/` — fixture-only tests (captured HTML, no live HTTP).
- Data roots (all under `src/sportstradamus/data/`):
  `adp/{contest_slug}/{YYYY-MM-DD}.parquet`, `contests/{contest}.json`,
  `projections/{contest}_{date}.parquet`.

Editing outside this footprint is a stop condition (§8).

## 6. Stage plan

Acceptance criteria live in the archived v2 spec; each stage cites its §5.x and
restates only what changed (package name `sportstradamus.drafts`; data paths
re-rooted at `src/sportstradamus/data/`). **Lane-level kill criterion:** if 2027
contest rules diverge enough to invalidate the simulation shape (pod size,
advance structure, roster slots), stage-0 re-verification rewrites the affected
stage before any code.

- **Stage 0 — retire-vs-evolve + product-rules re-verification.**
  Goal: decide the fate of each `src/deprecated/drafts/` file and pin current
  contest rules. Entry: owner call — may run pre-D4. Scope: docs +
  `data/contests/` JSON only; no package code. Acceptance: one-paragraph
  REVIVE/DELETE/ARCHIVE verdict per legacy file in `docs/DEPRECATED_TRIAGE.md`
  (extend the hygiene-lane Phase-1.6 doc; create the drafts section if absent —
  archived §5.1 presupposes replacement of `update_ez_adp.py`); §3 volatile
  assumptions re-verified and revised in place; `contests/{slug}.json` skeleton
  for the current Best Ball structure. Est: 1 session. Kill: none — decision
  stage; lane-level criterion above applies.
- **Stage 1 — ADP ingestion** (archived §5.1). Goal: `drafts/adp.py` +
  idempotent `drafts-adp-update` CLI writing stochastic ADP (mean **and**
  stdev) parquet to `src/sportstradamus/data/adp/{contest_slug}/{YYYY-MM-DD}.parquet`.
  Entry: D4 resolved (or owner early-start) + stage 0 done. Scope: `drafts/adp.py`,
  `pyproject.toml`, `tests/golden/`; CONTRIBUTING Package-Map row (owner
  review). Acceptance: archived §5.1 (fixture tests, no live HTTP). Est: 1–2.
  If-it-fails: scraping ToS-blocked or endpoint gone → manual-CSV becomes the
  primary source — that branch is in-spec, not a blocker; record in ledger.
- **Stage 2 — season projection distributions** (archived §5.2). Goal:
  `project_season(...)` translation layer over existing pickles — contest
  scoring from `contests/{contest}.json`, AR(1) on residuals for week-to-week
  correlation, deterministic with seed, cached parquet. Entry: stage 1 parquet
  exists. Scope: `drafts/projections.py` + tests. Acceptance: archived §5.2.
  Est: 2–3. Kill: star-RB consensus fixture persistently >5% off after
  debugging → method-failure verdict in ledger; dispatch research-analyst
  before any redesign.
- **Stage 3 — Battle Royale optimizer** (archived §5.3). Smaller scope than
  Best Ball — ship first. Goal: `optimize_battle_royale(...)` +
  `battle-royale-build` CLI ranking candidate rosters by expected prize equity
  with mandatory stacking. Entry: stage 2 weekly projections. Scope: one
  `drafts/` module + tests. Acceptance: archived §5.3 incl. the <60s perf
  budget. Est: 2. Kill: perf budget unreachable after one vectorization pass →
  cut `n_field_pods`, record; equity ranking unstable across seeds →
  method-failure verdict.
- **Stage 4 — Best Ball advance equity** (archived §5.4). Goal:
  `expected_payout(roster, structure, ...)` — pure simulation (no HTTP, no disk
  writes); Numba-JIT the inner score loop; perf budgets in the archived
  acceptance. Entry: stage 2. Scope: one `drafts/` module + tests. Acceptance:
  archived §5.4. Est: 2–3. Kill: sanity invariants (league-average roster ≈ 1×
  entry minus rake; 50th-pct round-1 advance ≈ 16.7%) still failing after
  debugging → verdict + stop.
- **Stage 5 — live-draft companion** (archived §5.5). Goal:
  `recommend(state, ...)` in <10s + `draft-recommend` CLI; module-level
  `lru_cache` shares the non-candidate sample matrix and opponent sims across
  candidate evaluations. Entry: stage 4. Scope: one `drafts/` module (+
  optional `pages/` view) + tests. Acceptance: archived §5.5. Est: 2. Kill:
  <10s unreachable → shrink `candidate_pool_size`/`n_simulations` per spec
  knobs; still over → park the stage (stage 6 does not depend on it).
- **Stage 6 — portfolio exposure tracker** (archived §5.6). Goal:
  `compute_exposure(contest, entry_pool)` + `drafts-exposure` CLI surfacing
  player/stack concentration, archetype mix, leverage. Entry: stage 3 (entry
  pools exist). Scope: one `drafts/` module (+ optional `pages/` view) +
  tests. Acceptance: archived §5.6. Est: 1. Kill: n/a — pure reporting.

## 7. Working rules

Conflict order: command output > CLAUDE.md/CONTRIBUTING.md > home-of-record doc
> this brief > roadmap v3.

- Archived v2 §5.1–§5.6 is the spec of record; edge-suite §4.3–§4.4 is
  orientation only — never treat its numbers as acceptance criteria.
- Every simulation is deterministic with an explicit seed; tests run on
  fixtures, never live HTTP (archived §5.1–§5.4).
- Dashboard pages read parquet snapshots only, never DuckDB (CLAUDE.md §Hard
  rules); no module-level `Archive()` imports in `pages/`.
- New modules stay under ~300 lines; one module per subagent for multi-module
  work (CLAUDE.md).
- `click` for CLIs, `tqdm` on long loops (CLAUDE.md §General Rules).
- Perf budgets come from the archived acceptance criteria — meet them, don't
  optimize past them.

## 8. Escalation & stop conditions

**STOP and ask the owner when:** entry criteria unmet (D4 unresolved and no
explicit early-start call); gates red at session start through no fault of
yours; smoke regression; any change to gate constants or test tolerances;
anything touching credentials, paid APIs, cron, or ToS surface — all
ToS-adjacent scraping decisions are owner-only; anything resembling
programmatic contest entry; two consecutive sessions with no acceptance
criterion moving (the grind detector).

**PARK AND PIVOT when blocked externally:** append a ledger line with the
blocking reason, set the status line to `BLOCKED (on: …)`, and point the owner
at the roadmap v3 swimlane index for the next lane.

**DISPATCH a subagent when:**

- `research-analyst` (Opus-backed) — optional, for advance-equity simulation
  design questions (pod-sim structure, AR(1) validity, variance reduction).
- `devel-ship-curator` — every devel-bound PR.
- `prompt-engineer` — major re-briefs of this document.
- `refactoring-specialist` — per the five CLAUDE.md triggers.

## 9. Session definition of done

- refactoring-specialist ran on every `.py` touched this session
  (CLAUDE.md five-trigger rule).
- `poetry run ruff check src/sportstradamus/` clean.
- `poetry run pytest tests/golden/` clean.
- `poetry run pytest -m integration -n0` clean, then
  `touch .claude/.state/integration_green`.
- One ledger line appended to §10; status line updated if a stage boundary
  was crossed.
- Never push `devel` directly — devel-ship-curator carves ship PRs.
- Durable non-obvious lesson? Offer a memory capture (CLAUDE.md §Agentic
  workflow conventions).

## 10. Ledger (append-only, newest first, cap ~15 — older lines live in git)

- 2026-06-10 · created · brief drafted from roadmap-v3 migration · next: idle until D4 (stage 0 may run early at owner call)
