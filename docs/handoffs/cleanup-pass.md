# Cleanup Pass

> Status: ACTIVE — stage 1 not started (briefed 2026-09-12 off the dashboard-ux close-out)

## 1. Mission & money logic

Retire the five debts the dashboard-ux close-out routed instead of fixing
([`../archive/dashboard-ux.md`](../archive/dashboard-ux.md) §11). Two touch money.
`add_dfs` writes Sleeper's lowest alt rung as that platform's `odds` row, and DFS
rows carry a real line with a pinned price, so every consensus that includes a
Sleeper quote leans toward the easiest rung. `merge_archives` never touches the
`ladder` table, so every dev↔prod archive sync silently drops the ladder history
that `dfs-products` stage 3 will price from. The other three are hygiene the
owner reads every night: Rivals residue in a retired product's code path, a dead
MLB depth recompute the model trains straight through, and voice-bank /
display-name gaps that render raw codes or bare letters in prose.

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

```bash
git fetch origin && git log --oneline origin/devel -3
grep -n "lowest alt rung\|def _dedup_offers_by_boost" src/sportstradamus/helpers/archive.py
grep -n '"odds", "lines"' src/sportstradamus/scripts/merge_archives.py      # ladder absent = stage 1 open
grep -rn -i "rival" src/sportstradamus tests --include=*.py                  # 5 hits = stage 2 open
grep -n "playerProfile.depth\|\"Player depth\"" src/sportstradamus/stats/base.py
grep -c "k's" src/sportstradamus/data/config/voice_bank.json                 # basketball cells still there?
poetry run python -c "from sportstradamus.helpers.market_display import market_display_name as m; print(m('NBA','PRA'), m('WNBA','RA'))"   # identity = stage 5 open
poetry run python -c "from sportstradamus.helpers import Archive; print(Archive().to_pandas('odds').query('market.str.startswith(\"H2H\")', engine='python').shape)"   # H2H rows in history?
```

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

## 6. Stage plan

Stages are independent; one session each; any can be parked.

1. **Sleeper `odds` row + ladder-aware merge.** (a) Measure first: over the
   last 30 days of the local archive, for each Sleeper `(Player, Market)` how far
   does the surviving `odds` line sit from the multi-book consensus line, and how
   often is Sleeper the only book? Put both numbers in the ledger. (b) Then pick
   the survivor rule in `_dedup_offers_by_boost` / `add_dfs`: the tier whose
   implied over-probability is nearest 0.5 when the price is real; the ladder's
   median line when the price is pinned; or no Sleeper `odds` row at all if (a)
   shows it never moves a consensus — smallest change that fixes the lean wins.
   Ladder rows keep every tier. Re-pin
   `test_multi_tier_ingest_keeps_one_odds_row_and_ladders_every_tier`.
   (c) `merge_archives`: add `ladder` to `_require_tables` and `_merge_table`
   with the same union + dedupe on the table's natural key; the report dict gains
   `ladder`; one round-trip pin in `tests/test_merge_archives.py`.
   Acceptance: measured delta in the ledger; a two-archive merge keeps both
   sides' ladder rows.
2. **Rivals residue.** Remove the symmetric single-`Boost` branch in
   `_dfs_offer_probs` and its pin
   (`test_add_dfs_boost_only_offer_prices_symmetric_never_blown_or_clamped`);
   true the `_strong_legs` docstring in `stories/menu.py` and the two comments in
   `tests/golden/test_correlation_helpers.py`. Run the §3 `H2H` query before
   touching the strip in `_resolve_market` / `resolve_leg_stat`: rows present →
   keep it with a one-line why; none → delete it too. Acceptance: the §3 grep
   returns nothing outside `src/deprecated/`.
3. **Dead depth recompute.** Delete the MLB batting-order rebuild in
   `_game_context` (`self.playerProfile.depth = battingOrder`): the in-frame
   consumer is overwritten by `_join_defense_and_parks` (`Player depth` =
   position for MLB) before `get_stats` returns, and both instance readers call
   `get_depth` first. Prove it: dump an MLB `get_stats` vector for a fixture
   offer before and after — byte-equal — and grep for any other
   `playerProfile["depth"]` reader on the MLB path. No matrix regen (the feature
   value does not change). Acceptance: byte-equal vector; fake-mode integration green.
4. **Voice bank + unit words.** (a) Drop the six basketball `k's` cells
   (`basketball/player/{shootout,grind,blowout,coinflip,even}/Over` +
   `even/Under`) — no NBA/WNBA slug resolves to `k's`; recount the
   `test_basketball_player_bank_depth` floors rather than lowering them by hand.
   (b) Add `("NHL", "D")` and `("NHL", "G")` to `_UNIT_GROUP_DISPLAY` so prose
   stops reading "the D room" / "the Gs"; extend the `test_engine.py` unit-word
   pins from NFL-only to every group in `correlation._LEAGUE_POSITIONS`.
   (c) Categorization divergence: the prophecize story path categorizes on the
   pre-remap display name (`"INTs Thrown"` → `production`) while the dashboard
   path sees the slug (`interceptions` → `mistakes`). Fix on the side that
   changes fewer persisted artifacts — remap `Market` in the `story_sink`
   capture, or resolve display names through `stat_map` inside `enrich_legs`
   before `_stat_category` — and say which in the ledger (`leg_schema.build_leg`
   persists the pre-remap market; `stories/thesis.py` re-enriches). Pin: one
   golden runs `enrich_legs` on an NFL interceptions leg from both paths and
   asserts `mistakes` for both. Acceptance: `test_bank_coverage.py` green,
   `test_no_prose_literals_in_stories_source` green.
5. **`market_display.json`.** The recorded debt was stale: the fallback is the
   slug itself (not `stat_map` names) and every `stat_meta` cell, shipped or
   withheld, already has a label. The real gap is five NBA/WNBA identity
   mappings — `BLST`, `PA`, `PR`, `PRA`, `RA` — that render combo slugs as raw
   codes on Board chips, the constellation card and Receipts. Replace them with
   prose labels; check every `combo_props.json` slug the same way; leave the 14
   orphan entries alone (harmless). Pin: no NBA/WNBA value equals its key.
   Acceptance: the §3 check prints prose for both.

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

- 2026-09-12 · stage 0 · brief written off the archived dashboard-ux brief §11; five debts located (§2/§6); `market_display` debt re-read as five identity mappings, not missing coverage · next: stage 1 (a) measure
