# Dashboard UX — "the Oracle" (lane brief, closed)

> Status: DONE — closed 2026-09-11. Everything below lives on `devel`; the merged lane branch
> `feature/dashboard-ux` was deleted at close. The live design home is
> [docs/dashboard_ux_redesign.md](../dashboard_ux_redesign.md), the tokens are
> [DESIGN.md](../../DESIGN.md), and the roadmap row is DONE
> ([roadmap §4](../sportstradamus_roadmap_v3.md)). Reopen by adding a roadmap row that points
> back here; §11 lists what was left open on purpose.

## 1. Mission & money logic

Turn the dashboard from read-only spreadsheet tabs into the product's primary interface: users
build correlation-aware parlays from model prophecies, see why every pick is made, and can verify
profitability unaided. Money logic: the models only earn if their edge gets *used* — a builder
that makes recommendations editable, explainable, and provable is the conversion layer between
calibration work (model-track) and actual entries on Underdog/Sleeper.

## 2. Read first (for anyone reopening)

1. [docs/dashboard_ux_redesign.md](../dashboard_ux_redesign.md) — what the dashboard is; naming
   map, platform taxonomy, the §8 scar register.
2. [DESIGN.md](../../DESIGN.md) — FIXED tokens, the constellation grammar (§4a), the Obsidian
   Tablet (§4b), mobile (§8a), the NEVER list.
3. [CLAUDE.md](../../CLAUDE.md) §Hard rules — the dashboard never touches DuckDB; §MANDATORY
   refactoring-specialist.
4. `src/sportstradamus/prediction/persist.py` — `_OFFER_KEEP_COLS` gates every snapshot column.
5. `tests/golden/test_design_tokens.py` + `test_dashboard_no_archive_lock.py` — the two hard
   gates every dashboard change keeps green.
6. `src/sportstradamus/prediction/joint.py` (copula pricing) + `prediction/payouts.py`
   (`payout_curve_for`) — the seams `dashboard/slip_engine.py` reuses.
7. [docs/mockups/](../mockups/) — the eight `p8-*.html` files are the locked pixel truth.
8. The archived P8 record: [the spec](superpowers/specs/2026-07-03-p8-oracle-assets-celestial-polish-design.md)
   and the phase plans `superpowers/plans/2026-07-03-p8-*.md`, `2026-07-05-p8-phaseR-*.md`,
   `2026-07-16-dashboard-mobile-*.md` (each carries a status header).
9. [docs/story_voice.md](../story_voice.md) — the voice-bank authoring contract (all prose is
   JSON data, never code).
10. [docs/art_assets.md](../art_assets.md) — every art slot, its placeholder, and what the owner
    does next.

## 3. Verify before you trust

If command output contradicts this brief, the output wins.

    git log --oneline -5 devel
    ls src/sportstradamus/dashboard/                      # the package
    python3 -c "import pandas as pd; print(pd.read_parquet('src/sportstradamus/data/runtime/current_offers.parquet').columns.tolist())"
                                                          # Kelly / Why / Game / Position / Consensus Line present
    ls src/sportstradamus/data/runtime/current_game_{corr,context,stories}.parquet current_offer_details.parquet
    python3 -c "import pandas as pd; m=pd.read_parquet('src/sportstradamus/data/runtime/current_line_movement.parquet'); print(len(m), 'rows,', (m['n_moves']>0).sum(), 'line moved,', ((m['n_moves']==0)&(m['n_price_moves']>0)).sum(), 'price only')"
                                                          # 0 rows ⇒ the last write matched no offers (see §7 on the integration guard)
    ls src/sportstradamus/data/runtime/user_slips.parquet   # locked slips
    poetry run python -c "import streamlit; print(streamlit.__version__)"  # ≥ 1.45 for st.navigation icons
    python3 -c "import pandas as pd; h=pd.read_parquet('src/sportstradamus/data/runtime/history.parquet'); print((h['Close Market Prob']>1.5).sum())"
                                                          # > 0 ⇒ the CLV closing slot is a mean again (Phase 0.9 regressed)

### Volatile product assumptions

- **Underdog payout multipliers** (Power/Flex incl. flex partials) — re-verify against the live
  app before trusting `prediction/payouts.py`; play-type rule (2–3 legs Power, 4+ Flex) is
  owner-stated product behavior.
- **Sleeper schedule** — real Max/Flex curve in `payouts.py:_sleeper_curve` (sourced to Sleeper's
  docs) plus the ≤ 2-leg full-refund rule; re-verify on any Sleeper product change.
- **Headshot / logo CDN URL patterns** — unverified and an owner decision (§11); the initials
  disc and team colours are the shipped fallback.
- **Underdog game lines** — product surface owned by dfs-products; the dashboard shows them only
  behind that lane's stages.

## 4. Locked decisions

Owner-locked 2026-06-11 and amended by owner review during the build; changes are owner-only.

- **IA as shipped:** Tonight / Board / Games / Receipts + Model Lab (Diagnostics · Correlations ·
  Training · Modifiers). The six-surface plan evolved: Game and Slips folded into **Games**
  (the game-first slip editor, 2026-06-14) and the Pick'em tab retired (the `pickem-build` /
  `kelly` CLIs stay). Games-first spine, board inside.
- **Celestial-B skin:** gold `#C9A227`, Cinzel/Cormorant display-only, Spectral serif body +
  IBM Plex Mono numerals; bans (purple gradients, default red, emoji icons) stand.
- **Prose is precomputed JSON banks** at prophecize time (`voice_bank.json`, `why_bank.json`,
  `stat_words.json`), md5-rotated, deterministic; an AST golden bans prose literals in
  `stories/*.py`. Free-LLM rewriter is an optional later seam, never a dependency. No paid APIs.
- **Platform taxonomy:** platforms = Underdog, Sleeper. Power/Flex = auto play types, chip only.
  Rivals retired 2026-09-10. Internal `contest_variant` names unchanged.
- **Games = story menu + constellation editor:** ≤ 5 data-driven stories per game, each with a
  Bankroll Builder (max Kelly log-growth) and a Shoot the Moon (max EV) preset; presets are valid
  parlays (distinct players, both teams unless the game's whole edge set is one team); the
  sidebar shelf is the real allocator.
- **Constellation:** universe and star size are Kelly > 0 (owner weighed Model EV − 1 and kept
  Kelly — identical set, don't re-propose); layout static per game; loose sports shapes dealt
  per night with no repeats, leagues walled to the general library plus their own gear;
  decoration never gold, never interactive, uncaptioned (DESIGN §4a).
- **Underdog game lines get no modeling engine** — correlation-engine citizens only.
- **Art is stock or commissioned, never AI-generated;** licensing is the owner's check before
  a file lands (`dashboard/assets.py` renders any slot that names a file on disk; the code
  gate was dropped 2026-09-12 at the owner's request).
- **Precompute-first:** the dashboard reads snapshots; the only live calc is `slip_engine.py`.

## 5. Module footprint (as shipped)

- `src/sportstradamus/dashboard/` — `app.py`, `data.py`, `theme.py`, `assets.py`, `columns.py`,
  `legs.py`, `lenses.py`, `narrative.py`, `viewport.py`, `slip_engine.py`, `surfaces/`,
  `components/` (the `constellation*` family, `slip_builder`/`slip_state`/`slip_dock`, `grid`,
  `deep_dive*`, `spark_svg`/`form_spark`, `satellite_picker`, `hero`, `glyphs`, `tickets`,
  `profit_sim`, `gate_matrix`, `lab_filters`, `offer_cards`, `locked_shelf`), the two
  hand-authored JS components under `components/*_component/build/` (no build toolchain —
  they speak Streamlit's postMessage protocol directly), `static/` (favicon, logo slot).
- `src/sportstradamus/prediction/stories/` + `data/config/{voice_bank,why_bank,stat_words,
  market_display,constellation_shapes,team_assets,stat_tooltips}.json`;
  `prediction/{persist,correlation,cli,line_movement}.py`; `nightly.py`; `helpers/io.py`;
  `leg_schema.py`; `scripts/{migrate_leg_schema,export_line_movement,build_team_assets}.py`;
  `data/assets/ambient/ambient_manifest.json`.
- Recorded exceptions: the additive `corr_market_summary.parquet` hook in
  `training/correlate.py:_write_corr_outputs`; the read-only
  `helpers/archive.py:get_book_line_histories` behind the line-movement export.
- Out of footprint (still true): `training/`, `stats/`, `strategies/` internals, `stat_meta.json`,
  archive schema, crontab/creds.

## 6. What shipped

| Stage | Landed | Pointer |
|---|---|---|
| P0 celestial tokens + spec + lane | 2026-06-11 | DESIGN.md amendment |
| P1 `dashboard/` package, `st.navigation`, legacy `pages/` deleted | 2026-06-11 | PR #79 (`d94bf3c`) |
| P2 precompute + thesis engine v2 (`current_game_context`, JSON voice banks) | 2026-06-12 | `9b6ff2a`; p3b narrative rework `bc061c8` (2026-07-02) |
| P3 story menu (`current_game_stories`), slip engine, locked shelf, nightly grading | 2026-06-12 | `be0e9bb`, PR #79 |
| P4 Tonight + Games, constellation editor on a hand-authored JS component, satellite legs | 2026-06-13/14 | PR #79 |
| P5 Obsidian AG Grid, lenses, pipeline-wide scoring-column rename | 2026-06-14 | `038bb30`, `3eed7c3` |
| P6 deep-dive v2 + `current_offer_details` (comps, volume, other stats); P6b swap struck (YAGNI) | 2026-06-15 | PR #79 |
| P7 Receipts skeptic surface; #6c nightly strategy-sim grid | 2026-06-15/16 | `f32a3c6` |
| P8 0/A/B/C — leg schema + flat history, shared infra, sober surfaces, celestial surfaces | 2026-07-05 | `84c3342` |
| P8 R remediation (starfield, heroes, board renderer, nebula cards, lab perf, manuscript fonts) | 2026-07-16 | `2fb7bb9..312e5f4`, `906a177` |
| P8 M mobile money loop (viewport, slip dock, board cards, touch constellation) | 2026-07-16; owner phone pass 2026-09-11 | `4ab13ff..0923f67` |
| P8 D loose constellation shapes (100-template bank, supernodes, slate dealer, tuning cockpit) | 2026-08-06 | `7171224b..e0c7219a` |
| Sep polish — Clouded shape, 20 stars, spacing, deep/wider lenses, calm-analyst voice | 2026-09-04/05 | `40205770`, `ea737e63` |
| L1 line movement — fair line from the `ladder` table, Board `Move`, Movement tab, card row | 2026-09-07/10 | `99b0aded`, `75ec341e` |
| L2 comps persistence | 2026-06-15 (P6a.1) | `current_offer_details.parquet` |
| P8 E1–E4 art catalog, ambient manifest + loader, favicon, logo slot | 2026-09-11 | `486a50c7` |
| Close-out — `constellation.py` / `constellation_wider.py` split under the cap (`119b2d2c`), NFL unit words + dead-cell prune (`15bba76e`), integration snapshot guard (`8229fed1`) | 2026-09-11 | `3ddcc4a5` docs |
| P8 E5 ambient files (owner-sourced night sky + nebula, `f82cd816`); Tonight cards slice the sky consecutively, Games hero shares the nebula, loader downscales originals at import | 2026-09-12 | `23946a6e` |

Not built, by decision: **L3–L9** UI scars flip on behind dfs-products (spec §8); **L6**
free-LLM rewriter (documented only); empirical-vs-model ρ overlay (roadmap §8).

## 7. Working rules (still binding for any dashboard edit)

- Parquet snapshots only, never DuckDB — the archive-lock golden is the gate. Never load
  `corr_same_team.parquet` / `corr_opposing.parquet` (NBA = 2.85M rows); per-game slices only.
- Snapshot schema changes are append-only (server tracks `devel`; old process + new snapshot
  must coexist).
- Slip state lives under plain non-widget `st.session_state` keys; the shelf renders from
  `app.py` so its widgets exist on every page. Money is `Decimal`.
- Display copy follows the spec §2 taxonomy.
- Streamlit caches *imported* modules: a component edit needs a server restart, not a rerun.
- Goldens are necessary, not sufficient, for anything AG Grid or JS renders — every grid or
  component change ends with a recorded live-browser verdict (playwright, desktop 1600×1000 and
  phone 390×844 with `?m=1`).
- The fake-mode integration suite writes `current_line_movement.parquet`,
  `current_offer_details.parquet` and `current_pickem.parquet` for real; the conftest snapshot
  guard restores them — keep new runtime writers on that list.

## 8. Reopening

Stop and ask the owner before: any change to payout tables or gate constants; anything touching
crontab, creds, paid APIs, or scraping ToS; editing outside §5. Gates + refactoring-specialist
per CLAUDE.md; a devel-bound PR goes through devel-ship-curator.

## 9. Session definition of done

refactoring-specialist on every touched `.py` · `ruff check` clean · `pytest tests/golden/` clean ·
`pytest -m integration -n0` clean then `touch .claude/.state/integration_green` · one ledger line ·
a live-browser verdict for anything rendered.

## 10. Ledger (newest first; trimmed at close — the full entries are in this file's git history under `docs/handoffs/dashboard-ux.md`)

- 2026-09-12 · ambient files landed (`f82cd816`); Tonight cards show consecutive slices of one sky (per-card offsets from a parent-window script), Games hero shares the nebula, loader downsizes >1600 px originals to WebP at import (`23946a6e`) · live: offsets exactly cumulative on 5 desktop / 17 phone cards, both heroes nebula-backed, 0 page errors.
- 2026-09-12 · licence gate dropped from `assets.py` at owner request (owner checks licences before a file lands); team marks skipped, player headshots wanted → next lane.
- 2026-09-11 · lane closed · Phase E1–E4 built, constellation split (142 figure JSONs byte-equal), story debts, integration guard; docs trued, brief archived, `feature/dashboard-ux` deleted · gates: ruff clean, golden 4682 passed + 1 xpassed, integration 34 passed 2 skipped · live (playwright 1600×1000 + 390×844): favicon inline gold mark, no logo, Tonight wash + Receipts hero unchanged, constellation main/deeper/wider renders with the iframe height posted on both viewports, hover + tap cards render, 0 page errors from our code (Streamlit's telemetry webhook and deep-link `_stcore` probes only) · `export-line-movement` wrote 0 rows: the dev archive holds no MLB ladder polls, so the card's movement row rests on its golden.
- 2026-09-10 · line movement reads the ladder's balanced rung + a fair line; Movement tab; Games card row (`75ec341e`).
- 2026-09-07 · Board `Move` spark (L1 via `get_book_line_histories`); card last five as deviation bars (`99b0aded`, `59464e6e`).
- 2026-09-05 · 20 stars, market display names, deep/wider lens split, calm-analyst voice bank (`ea737e63`).
- 2026-09-04 · MLB ρ fix (positions), star cap + spacing, lenses, Clouded shape (`40205770`).
- 2026-08-06 · Phase D owner pass: ball hubs, nameplate cut, leagues walled, bank 100, `constellation_slate.py` carve (`a7e4a897`, `e0c7219a`).
- 2026-08-06 · Phase D D1–D7: shape bank, supernodes, classifier, slate dealer, decoration, tuning cockpit (`7171224b..f305eb2`).
- 2026-08-06 · step-0 prerequisites: NaN-returning `get_team_market` for game context, PDX→POR, playwright pinned (`fb622f0`).
- 2026-07-16 · Phase M built + live-verified (`4ab13ff..0923f67`); owner phone pass passed 2026-09-11.
- 2026-07-16 · Phase R shipped, every task owner-live-checked; manuscript fonts, Obsidian Tablet, lens animations (`2fb7bb9..312e5f4`, `906a177`).
- 2026-07-11 · dfs-products heads-ups: Model Lab Modifiers page, "Payout incorrect?" chip on Games.
- 2026-07-05 · P8 0/A/B/C merged (`84c3342`); owner review → Phase R planned.
- 2026-07-03 · P8 planned: spec + plans 0/A/B/C, then D + E; Sheets-era data retirement = Phase 0.
- 2026-07-02 · story engine p3b: narrative directions, conviction anchors, one headline per story, prose banks (`bc061c8`).
- 2026-06-16 · #6c strategy-sim rework on the fixed engine; remote-access doc de-drifted (`f32a3c6`).
- 2026-06-15 · P0–P7 merged to devel (`d94bf3c`, PR #79).

## 11. Open at close

Owner decisions (nothing blocks on them; each has an honest fallback in place):

- Prophecy-voice names for Ladder / Combo Entry / game-line leg — RESOLVED 2026-09-12: the
  official names stay, renames would not match the apps
  ([dfs-products](../handoffs/dfs-products.md) §4).
- Game-total star fill — RESOLVED 2026-09-12: gradient blend of the two teams' colours
  (same §4).
- Team marks: skipped; player headshots → the
  [`player-headshots`](../handoffs/player-headshots.md) lane (briefed 2026-09-12).
- The commissioned logo (`docs/art_briefs/logo_guru.md`) — the `st.logo` slot needs zero
  code when the files land.
- The optional `export-line-movement` cron + healthcheck ([OPERATIONS.md](../OPERATIONS.md));
  `prophecize` already writes the snapshot hourly.

Routed debts (recorded here so no session rediscovers them; the archive, Rivals, depth,
`market_display` and voice-bank items are the [`cleanup-pass`](../handoffs/cleanup-pass.md)
lane's scope since 2026-09-12, the silhouettes the
[`constellation-art`](../handoffs/constellation-art.md) lane's):

- `add_dfs` archives Sleeper's lowest tier as its `odds` row (a consensus input) and
  `merge_archives` skips `ladder` — archive/dfs-products territory.
- Rivals residue outside the dashboard (`prediction/stories/menu.py`, `helpers/archive.py`) —
  the Underdog-scraper lane retired the product 2026-09-10.
- Dead depth recompute in `stats/base.py` — stats footprint.
- `market_display.json` covers the fantasy slugs only; other opaque codes fall back to
  `stat_map` names.
- Constellation residuals: 5–6 games in one vertical sky band can still graze by ≤ 12 px on
  desktop; one phone band in four lands near an even pitch by its draw.
- Dashboard files still over the ~300-line guidance after the close-out split (`data.py` 666,
  `grid.py` 532, `constellation_layout.py` 371, `constellation_spacing.py` 307 among them);
  `constellation_slate.py` is owner-exempted.
- Voice-bank reachability, settled at close: the `player` archetype is reachable only from the
  live `slip_headline` path (two legs of one player before Lock it in), so it stays; the seven
  football `stops` cells were pruned (no NFL market maps there). Same pattern, not pruned:
  six basketball `k's` cells (no NBA/WNBA market), and NHL `D`/`G` unit groups still render as
  letters. The prophecize story path categorizes on pre-remap display names while the dashboard
  path uses slugs ("INTs Thrown" → production, `interceptions` → mistakes).
- Phone tap card under Chromium *touch* emulation: one tap fires two `plotly_click`s (the
  `dragmode=False` figure never `preventDefault`s the touch, so the browser synthesizes mouse
  events) and `main.js` reads the second as the confirm tap. The owner's real-phone pass did
  not hit it; if a device does, the fix is a per-tap debounce in `main.js`, not the figure.
