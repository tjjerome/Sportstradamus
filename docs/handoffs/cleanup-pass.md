# Cleanup Pass

> Status: DONE — all five stages landed 2026-09-12 (briefed the same day off the
> dashboard-ux close-out). Unpushed on `devel`; §10 carries the measured deltas.

## 1. Mission & money logic

Retired the five debts the dashboard-ux close-out routed instead of fixing
([`../archive/dashboard-ux.md`](../archive/dashboard-ux.md) §11). Two touched money.
`add_dfs` ranked a multi-rung DFS offer by `|Boost_Over - 1|`, but `Boost_Over` is a
full decimal payout — 1.0 implies certainty, not a neutral price — so the rule kept the
lowest rung on 93% of multi-tier Sleeper keys. It now keeps the rung priced nearest even
money, the same main-rung rule `prediction.line_movement` applies. `merge_archives` never
touched the `ladder` table, so every dev↔prod sync silently dropped the ladder history
`dfs-products` stage 3 will price from; the union now covers it. The other three were
hygiene the owner reads every night: Rivals residue in a retired product's code path, a
dead MLB depth recompute, and voice-bank / display-name gaps that rendered raw codes or
bare letters in prose.

## 2. Read first (in order)

1. [`../archive/dashboard-ux.md`](../archive/dashboard-ux.md) §11 — the routed-debts
   list this lane closes (and the constellation residuals it does not).
2. `src/sportstradamus/helpers/archive.py` — the three-table schema (`odds`,
   `lines`, `ladder`), `add_dfs`, `_dedup_offers_by_boost`, `_stage_book_ev`,
   `_dfs_offer_probs`, `_weighted_book_ev`; the docstring near `get_ev` already
   says Sleeper's surviving tier "is the lowest alt rung rather than the main line".
3. `src/sportstradamus/scripts/merge_archives.py` — `_require_tables`,
   `_merge_table`, the report dict; `tests/test_merge_archives.py` (13 pins, all
   odds/lines).
4. `src/sportstradamus/stats/base.py` — `_game_context` (the MLB batting-order
   rebuild), `_join_profiles`, `_join_defense_and_parks`; the readers that
   re-resolve depth themselves: `prediction/correlation.py` (`get_depth` call
   before the profile read) and `stats/mlb.py` `_project_plate_appearances`.
5. `src/sportstradamus/prediction/stories/legs.py` (`_STAT_CATEGORY`,
   `_stat_category`, `enrich_legs`), `engine.py` (`_UNIT_GROUP_DISPLAY`),
   `data/config/voice_bank.json`; `prediction/cli.py` — the `story_sink` capture is
   flagged "pre Market-remap".
6. `src/sportstradamus/helpers/market_display.py` + `data/config/market_display.json`;
   `tests/golden/test_grid_options.py` pins the slug-preserving contract.
7. [`dfs-products.md`](dfs-products.md) §4/§5 — Ladders stage 3 owns the ladder
   table's future and the product names; this lane only stops losing rows.

## 3. Verify before you trust

Each line below returns the closed state; a different answer means something regressed.

```bash
git fetch origin && git log --oneline origin/devel -3
grep -c "_dedup_offers_by_boost" src/sportstradamus/helpers/archive.py       # 0 — survivor ranks on |p_over - 0.5|
grep -n '_LADDER_COLS' src/sportstradamus/scripts/merge_archives.py          # present — ladder is in the union
grep -rn -i "rival" src/sportstradamus tests --include=*.py                  # nothing
grep -n "\"Player depth\"" src/sportstradamus/stats/base.py                  # 2 hits, both joins; no depth write in _game_context
grep -c "k's" src/sportstradamus/data/config/voice_bank.json                 # hockey + shared only, no basketball
poetry run python -c "from sportstradamus.helpers.market_display import market_display_name as m; print(m('NBA','PRA'), m('WNBA','RA'))"   # prose, not the slug
```

The `H2H` question §4 left open is settled: the archive holds **zero** `H2H %` rows in
`odds`, `lines` or `ladder` and zero `'% vs. %'` entities, so no read path depends on
that history. Both `H2H ` strips survive anyway — `parlay.resolve_leg_stat` is pinned by
`tests/golden/test_parlay_search.py` and reads persisted parlay descriptions, not the
archive.

### Volatile product assumptions

- Sleeper's `Boost_Over` is a full decimal payout (`books/sleeper.py`), which is
  why "closest to 1.0" selects the easiest rung. If Sleeper ever quotes balanced
  main lines near 1.0, the stage-1 rule must still pick the balanced tier.
- Underdog retired Rivals 2026-09-10 ([`dfs-products.md`](dfs-products.md) §4). If
  a symmetric single-`Boost` product ever returns, the `_dfs_offer_probs` branch
  returns with it — from git, not from a flag.
- `tests/golden/test_archive_shapefree_storage.py` derives its `_CELLS` from the
  live calibration at import, so its pins move on any `meditate`
  ([`add_dfs` pins drift](../../CLAUDE.md) memory) — re-derive, never hand-edit numbers.

## 4. Locked decisions

- 2026-09-12 — **Scope is the five routed debts, nothing wider** (owner). No
  file-length splits, no tidy-ups of the files touched; the 14 dashboard files
  still over ~300 lines stay parked in the archived brief §11.
- 2026-09-12 — **Ladders stay an archive table.** `dfs-products` stage 3 reads
  `ladder` for its pricer; this lane makes the table survive a merge, it does not
  redesign it.
- 2026-09-10 — **Rivals is retired for good** (dfs-products §4). Residue is
  deleted, not flagged off. The `"H2H "` market-name strip stays only if the
  production archive still holds `H2H` rows the read path must parse (§3 check).
- 2026-09-12 — **Product names are the DFS apps' own** (dfs-products §4).
  Voice-bank edits never introduce prophecy-voice renames.

## 5. Module footprint & canonical paths

| Module | Debt | Notes |
|---|---|---|
| `helpers/archive.py` | 1, 2 | shared seam (roadmap §5): one subagent, one commit; pins in `tests/golden/test_archive_shapefree_storage.py`, `test_get_ev_robustness.py`, `test_line_movement.py` |
| `scripts/merge_archives.py` + `tests/test_merge_archives.py` | 1 | ladder round trip |
| `stats/base.py` | 3 | delete only; MLB feature vector must not move |
| `prediction/stories/{legs,engine,menu}.py`, `prediction/cli.py`, `data/config/voice_bank.json` | 4 | pins in `tests/golden/test_bank_coverage.py`, `test_engine.py`, `test_story_menu.py` |
| `data/config/market_display.json` (+ `helpers/market_display.py` if the contract changes) | 5 | pins in `tests/golden/test_grid_options.py` |

No dashboard modules, no `Archive()` at import anywhere new (the archive-lock golden
auto-discovers dashboard imports).

## 6. What landed

All five stages are on `devel` (unpushed). Each was one commit-sized change; the
measured numbers behind stage 1 are in §10.

1. **Sleeper survivor rule + ladder-aware merge.** `_dedup_offers_by_boost` is gone;
   `add_dfs` computes each offer's `p_over` once and ranks tiers by `|p_over - 0.5|`
   (higher line on a tie) — the rule `line_movement._main_rungs` already used, so the
   archived tier and the tracked line now agree by construction. Every tier still
   ladders. The dedupe key moved from the raw market to the resolved one.
   `merge_archives` unions `ladder` too; a source that predates the ladder DDL
   soft-skips (`report["ladder"] is None`, CLI prints `ladder: skipped`) rather than
   being rejected.
2. **Rivals residue.** The symmetric single-`Boost` branch in `_dfs_offer_probs` is
   deleted — `dfs_boost_probs(0, 0)` already returned the same `[0.5, 0.5]`, proven by
   watching the old pin stay green before deleting it. Prose trued in three places.
   Both `H2H ` strips stay (§3).
3. **Dead MLB depth recompute.** The `battingOrder` rebuild in `_game_context` is gone.
   `tests/golden/test_mlb_batting_order_not_a_feature.py` drives the real `get_stats`
   with a posted lineup and without one and asserts the frames are equal — it passed
   *before* the deletion, which is the proof. `correlation.py`'s comment now names the
   real reason the `get_depth` re-resolve is needed: `base_profile` rebuilds
   `playerProfile` and zeroes depth for the date.
4. **Voice bank, unit words, categorization.** Six unreachable basketball `k's` cells
   dropped (hockey and `shared` keep theirs); the depth floor re-counted 44 → 40 from
   the bank. `_UNIT_GROUP_DISPLAY` gained `("NHL","D"): "defender"` and
   `("NHL","G"): "goalie"` — `defenseman` was rejected because the templates inflect
   `{grp}s`. `lower_leg` now carries the resolved slug and `enrich_legs` falls back to
   it, so a leg that misses the offers join no longer categorizes off the display name;
   that miss used to invert the valence flag `narrative_side` reads, not just the
   category.
5. **`market_display.json`.** Ten identity mappings (five slugs × NBA/WNBA) became
   prose. The pin already existed: deleting `_SLUG_COMBOS` from
   `tests/golden/test_market_display.py` turned its coverage test into the gate.

## 7. Working rules

- Conflict order: command output > CLAUDE.md/CONTRIBUTING.md > home-of-record
  doc > this brief > roadmap v3.
- Multi-module stages run one subagent per module (CLAUDE.md); subagent prompts
  forbid HEAD-moving git in the shared checkout; `git add` by explicit path only
  (the local `:50` prophecize cron churns `data/runtime/*`).
- `helpers/archive.py` is on `dfs-products`' and `sleeper-parity`'s footprints —
  land stage 1 as one small commit and rebase nothing.
- Calibration-coupled pins are re-derived from the code path, never edited to
  the new number.
- refactoring-specialist per touched `.py` before any review or "done".

## 8. Escalation & stop conditions

**Stop and ask the owner:** stage 1(b) if the chosen rule moves any live
consensus line on a dry-run day (report the delta first); stage 2 if `H2H`
rows exist in the production archive (the read path must keep parsing them);
stage 3 if the before/after vector differs at all.

**Park and pivot:** any stage, freely — they are independent.

**Dispatch:** `refactoring-specialist` on every touched `.py`; `devel-ship-curator`
only if the owner wants a carved PR (stages are small enough for direct devel commits).

## 9. Session definition of done

- refactoring-specialist ran on every `.py` touched this session.
- `poetry run ruff check src/sportstradamus/` clean; CI also runs `ruff format`.
- `poetry run pytest tests/golden/` clean.
- `poetry run pytest -m integration -n0` clean, then
  `touch .claude/.state/integration_green`.
- One ledger line appended below; status line updated on stage boundaries; the
  matching bullet in the archived dashboard-ux brief §11 struck when a debt closes.
- Never push `devel`.
- Durable non-obvious lesson? Offer a memory capture.

## 10. Ledger (append-only, newest first, cap ~15)

- 2026-09-12 · stages 1-5 · all five landed, gates green (golden 4720, integration 34); `ladder` is 22.5M rows, 2nd-biggest table, so the merge debt was bigger than briefed · next: push
- 2026-09-12 · stage 1(a) measure · Sleeper sole book on 44.3% of keys (kills the "drop the odds row" option); where a sportsbook overlaps, mean line delta −0.233, 10.8% below, concentrated (MLB hits allowed −1.69 at 99% below, WNBA PRA −4.95); survivor was the lowest rung on 93% of multi-tier keys
- 2026-09-12 · stage 1(b) delta · new rule cuts mean \|p_over−0.5\| of the archived tier 0.157 → 0.100 and lifts the archived line +0.43; real leak was `lines` (no `book` column, so neither `sportsbook_cohort` nor `_drop_divergent_lines` reaches it — 8.0% of keys moved consensus, mean 0.753)
- 2026-09-12 · stage 2 · zero `H2H` rows and zero `'% vs. %'` entities in the archive; branch deletion proven behavior-preserving before the pin was removed
- 2026-09-12 · stage 4(c) · brief's premise was half wrong — `enrich_legs` prefers the offers-frame market, so the divergence only bit on a join miss; the miss also inverted valence, not just the category
- 2026-09-12 · stage 0 · brief written off the archived dashboard-ux brief §11; five debts located (§2/§6); `market_display` debt re-read as five identity mappings, not missing coverage
