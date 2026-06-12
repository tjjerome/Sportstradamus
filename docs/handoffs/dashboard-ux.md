# Dashboard UX — "the Oracle"

> Status: ACTIVE — P3b complete (slip builders + locked shelf + grading landed); next P4 constellation visual + theming

## 1. Mission & money logic

Turn the dashboard from read-only spreadsheet tabs into the product's primary interface: users
build correlation-aware parlays from model prophecies, see why every pick is made, and can verify
profitability unaided. Money logic: the models only earn if their edge gets *used* — a builder
that makes recommendations editable, explainable, and provable is the conversion layer between
calibration work (model-track) and actual entries on Underdog/Sleeper.

## 2. Read first (in order)

1. [docs/dashboard_ux_redesign.md](../dashboard_ux_redesign.md) — the approved design spec (what
   to build); naming map + platform taxonomy live there.
2. [DESIGN.md](../../DESIGN.md) — visual tokens incl. celestial layer; FIXED vs FLEXIBLE.
3. CLAUDE.md §Hard rules — dashboard never touches DuckDB; §MANDATORY refactoring-specialist.
4. `src/sportstradamus/prediction/persist.py` — `_OFFER_KEEP_COLS` gates every new snapshot
   column.
5. `src/sportstradamus/dashboard_data.py` + `dashboard_detail.py` — the legacy surface being
   replaced (loaders, dialog, phrase bank).
6. `tests/golden/test_design_tokens.py` + `test_dashboard_no_archive_lock.py` — the two hard
   gates this lane must keep green and extend.
7. `src/sportstradamus/prediction/parlay.py` — copula pricing + `_payout_curve_for` the slip
   engine reuses.
8. [docs/mockups/](../mockups/) — approved wireframes (site-map, facelift).

## 3. Verify before you trust

If command output contradicts brief prose, the output wins — fix the brief in place (minor) or
stop and ask the owner (material).

    git fetch origin && git log --oneline origin/devel -3
    git log --oneline -5                          # lane branch state
    ls src/sportstradamus/dashboard/ 2>/dev/null  # package exists ⇒ P1 landed
    ls src/sportstradamus/pages/ 2>/dev/null      # legacy pages gone ⇒ P1 landed
    python3 -c "import pandas as pd; print(pd.read_parquet('src/sportstradamus/data/runtime/current_offers.parquet').columns.tolist())"
                                                  # K/Why/Game present ⇒ P2 landed
    ls src/sportstradamus/data/runtime/current_game_corr.parquet 2>/dev/null
    ls src/sportstradamus/data/runtime/user_slips.parquet 2>/dev/null  # ⇒ P3 produced saves
    poetry run python -c "import streamlit; print(streamlit.__version__)"  # needs ≥1.45 for st.navigation icons

### Volatile product assumptions

- **Underdog payout multipliers** (Power/Flex tiers incl. flex partial payouts) — re-verify
  against the live app before trusting `prediction/parlay.py` tables; play-type rule (2–3 legs
  Power, 4+ Flex) is owner-stated product behavior, re-verify on UD product changes.
- **Sleeper payout table** — `parlay.py` carries placeholder `[1.0, 1.0]`; slip engine must show
  "payout table unverified" until a real table is committed. Re-verify: open Sleeper, record
  multipliers per leg count, commit alongside the UD tables.
- **Headshot/logo CDN URL patterns** — unofficial; re-verify with a curl per league before P8
  relies on them; initials fallback must always work.
- **Underdog game lines** (Total/Spread/Moneyline) — product surface may change; re-verify
  what's offered before building the correlation-engine stage.

## 4. Locked decisions

All owner-locked 2026-06-11 (the mockup-review session); changes are owner-only:

- IA: six surfaces (Tonight/Game/Board/Slips/Receipts/Model Lab), games-first spine, board
  inside. No relitigating page structure.
- Celestial-B skin: gold `#C9A227` + Cinzel/Cormorant display-only faces; DESIGN.md amended
  deliberately by the owner; bans (purple gradients, default red, emoji icons) stand.
- Prose is precomputed templates/phrase banks at prophecize time. Free-LLM rewriter is an
  optional later seam, never a dependency. No paid APIs anywhere in the dashboard path.
- Platform taxonomy: platforms = Underdog, Sleeper. Power/Flex = auto play types (2–3 / 4+
  legs), informational chip only. Rivals = leg type (sits inside a Power/Flex slip beside player
  props and game lines, not a separate contest). Internal `contest_variant` field names unchanged.
- Slips is a **story menu**, not the old `Family` cluster: up to 5 data-driven stories per game,
  each offering a **Bankroll Builder** (max Kelly log-growth, sun motif) and a **Shoot the Moon**
  (max EV, moon motif) starting parlay, edited from there in the constellation. Menu stakes are a
  standalone preview; the slip rail is the real bankroll allocator. Mechanics live in P3.
- Underdog game lines get **no modeling engine** (sharp lines); correlation-engine citizens
  only, book-implied probabilities.
- Ambient/visual art is stock or commissioned only — **never AI-generated**;
  license/attribution tracked in the ambient manifest.
- Precompute-first: dashboard reads snapshots; only live calc is slip joint-prob (copula over
  per-game corr slices).

## 5. Module footprint & canonical paths

May touch:

- `src/sportstradamus/dashboard/` (new package: `app.py`, `data.py`, `theme.py`, `assets.py`,
  `prose.py`, `slip/`, `components/`, `surfaces/`) — all files <300 lines (CLAUDE.md).
- Legacy surface being replaced (delete at end of P1, no shims): `dashboard.py`,
  `dashboard_app.py`, `dashboard_data.py`, `dashboard_detail.py`, `pages/`.
- `src/sportstradamus/prediction/`: `stories.py` (new), `persist.py`, `correlation.py`, `cli.py`.
- `src/sportstradamus/nightly.py` (user-slip grading step only).
- `src/sportstradamus/helpers/io.py` (path constants only).
- `src/sportstradamus/scripts/`: `export_line_movement.py`, `build_team_assets.py` (new).
- `src/sportstradamus/data/config/team_assets.json`, `src/sportstradamus/data/assets/ambient/`.
- `tests/golden/` + `tests/integration/` files for the above; `.claude/hooks/design-lint.py`;
  `DESIGN.md`; `.streamlit/config.toml`; this brief; the spec; roadmap §4/§9.

Out of footprint (stop condition): `training/`, `stats/`, `strategies/` internals (import them,
don't edit), `stat_meta.json`, archive schema, crontab/creds (owner-only; propose in ledger).

## 6. Stage plan

Stages P0–P8 are the redesign; L-stages are the registered later work behind the scars. Each
stage ends with the §9 checklist green and the dashboard runnable.

- **P0 — tokens + docs + lane** ✔ (this commit). Goal: celestial amendment + spec + lane + scars
  registered. Acceptance: design-token golden test green with celestial asserts; spec + brief +
  roadmap rows exist.
- **P1 — package migration + nav skeleton.** Goal: `dashboard/` package, six `st.Page` surfaces,
  behavior-preserving ports (Board←page 1, Slips←2a+2b, Receipts←3+6, Lab←4/5/7), Tonight/Game
  thin, global sport switch, single `set_page_config`, legacy files deleted. Entry: P0. Scope:
  dashboard package + test repoints (`test_dashboard_no_archive_lock.py` → pkgutil
  auto-discovery; two source-slice render pins → new paths). Acceptance: all gates green;
  manual: six nav entries render real snapshots, sport switch filters, deep-dive opens from
  Board. Est: 1–2 sessions. If it fails: revert to last green, halve the port batch.
- **P2 — pipeline precompute + thesis engine v2.** ✔ Two increments landed. First the v1
  generator (`Why`/`K`/`Game` offer keep-cols, `Thesis` on parlays, `current_game_corr.parquet`
  per-game leg-pair ρ via opt-in `find_correlation(corr_sink=…)`, `stories_version` in meta,
  render-time phrase bank retired, `parse_leg` moved to the prediction layer); then the v2 thesis
  engine (the four-archetype router + `current_game_context.parquet` + JSON voice banks detailed
  below). The naming bar (owner, post-P1) is met: headlines unique within a slate, built from
  concrete game context, bank large enough that repeat users don't see the same headline two days
  running; deterministic templates only (locked §4), diversity from bank size + context keying +
  the md5 date seed, never randomness. See §10 for both landings. Design rationale + verified code
  citations live in `docs/superpowers/specs/2026-06-11-thesis-engine-v2-design.md` (background).

  *Why it was reopened:* v1 told exactly one story — a featured player (`_family_thesis` always
  elected a star, breaking ties alphabetically by name) — and its game-shape classifier was dead in
  production. The snapshot `O/U` column is the **team-implied half-total** (~108 NBA / ~82 WNBA;
  `cli.py` overwrites `O/U` with `archive.get_total`, which stores `(total ± spread)/2` per
  team — `moneylines.py:295-302`), but v1's `_TOTAL_BANDS` expected raw game totals (NBA
  235/215). So `total.median() ≈ 108 ≤ 215` always ⇒ every non-lopsided game classified "grind"
  and the shootout/coinflip/even cells were dead inventory. v2 replaces that with the
  league-relative `total_ratio` band classifier.

  1. **Game-context precompute + classifier fix.** New `current_game_context.parquet`, one row per
     `(League, Game, Date)`, path constant in `helpers/io.py` (`CURRENT_GAME_CONTEXT_PATH`), written
     by `persist.py` beside the corr slice from a pure `build_game_context(offers, default_totals)`
     over the finished `snapshot_offers` (no archive reads — cli passes `archive.default_totals` in).
     Columns: `League`/`Game`/`Date`; `game_total` (sum of the two teams' median `O/U`); `spread`
     (abs diff of the two team-implied totals — exact, needs no `get_spread`); `fav_team` (max
     `Moneyline`); `ml_fav_prob`, `ml_margin` (`max|p − 0.5|`); `total_ratio` (`game_total /
     baseline`); `baseline_total`; `shape`; `pos_edges` (JSON `{team: {pos_group: {dvpoa, n}}}`);
     `n_offers`. Replace `_TOTAL_BANDS`/`_DEFAULT_TOTAL_BAND` with one league-agnostic ratio-band
     classifier over `total_ratio`: `_SHOOTOUT_RATIO = 1.05`, `_GRIND_RATIO = 0.95`, keep
     `_ML_LOPSIDED_MARGIN = 0.18` (already probability-space). Baseline = slate-median game total
     when the slate has ≥ 4 games, else `2 × default_totals[league]`. Keep the position labels:
     `_resolve_player_positions` (`correlation.py`) already resolves `G1`/`QB1`/`B3` on the league
     slice but never writes them back — write them onto the offers frame as a new `Position` string
     column and append it to `_OFFER_KEEP_COLS` (append-only); combo/`vs.` legs resolve empty and
     are excluded from `pos_edges`.
  2. **Pure archetype engine.** `thesis(legs, ctxs) -> headline`: pure, deterministic, no I/O, no
     archive, no `Family` input. `Leg`/`GameCtx` are frozen dataclasses built by `enrich_legs(parsed,
     offers)` + the context codec. Variant seed keeps the md5 date scheme keyed on the leg-set:
     `md5(game | sorted-leg-keys | date | shape | archetype)` — identical between prophecize and the
     P3 rail because `date` is the snapshot date, not wall-clock. Four archetypes with named-constant
     firing gates, precedence **player → stack → unit → game-script**:
     - **player** — one player holds ≥ 0.5 of the legs **and** ≥ 2 legs (strict gate — kills the
       alphabetical-star pick on no-standout slips).
     - **stack** — ≥ 3 legs in the primary game, ≥ 2 distinct players, mean bet-signed ρ ≥ 0.10 over
       the slip's leg pairs (read from the corr slice).
     - **unit** — ≥ 2 legs share `(team, position-group, direction)` and that group's aggregated
       DVPOA edge ≥ 0.05 (pos_group = `Position` label minus its rank digit, `G1`→`G`).
     - **game-script** — shape ∈ {shootout, grind, blowout, coinflip}; the no-standout / mixed-slip
       answer ("shootout lifts every stat line"), always available.

     Fallback when nothing fires and shape is "even" → game-script "even" cells, never a forced star.
     Multi-game slip → pick the primary game (most legs, ties by sorted game key) and tell its story.
     `attach_parlay_theses` slims to a pipeline adapter: parse each parlay row's legs → enrich →
     build `GameCtx`s once per game → call `thesis()` per **distinct leg-set** → run the existing
     slate-uniqueness pass over distinct leg-sets within `(League, Date)`. `Family` leaves the thesis
     path entirely but stays on `current_parlays.parquet` for Slips grouping/ordering only.
  3. **Per-sport voice banks (all five leagues now — do not defer for seasonality).** Bank STRINGS
     live in an external committed data file `data/config/voice_bank.json` (same convention as
     `stat_map.json`) — a phrase bank is data, not code, so it is **not** bounded by the 300-line
     code limit and should be authored generously (many variants per reachable cell so repeat users
     rotate; that is what a data file is for). Nested
     `{voice: {archetype: {shape: {direction: {category: [variants]}}}}}`; voices `basketball` (NBA
     **and** WNBA share it), `football`, `hockey`, `baseball`, and `shared` (the league-neutral
     fallback net). `stories/bank.py` is a small pure loader: a cached JSON read plus
     `bank_cell(voice, archetype, shape, direction, category)` walking the fallback chain `(voice,
     arch, shape, dir, cat) → (shared, …) → (shared, …, "even", dir, cat) → (shared, …, "even", dir,
     "production")` (guaranteed hit). v1's 107 player variants are preserved verbatim as
     basketball's player cells. Template slots: player `{p}`/`{g}`; unit `{team}`/`{grp}`/`{opp}`;
     game-script `{g}`; stack `{n}`/`{g}`/`{p}` — voices use only their archetype's slots. Author
     football/hockey/baseball from game knowledge with sport-correct vocabulary and pin via a
     synthetic-fixture coverage golden (no live legs needed; classifier/normalization/categories are
     league-blind, so adding a league is one JSON voice block). Decisions: **no pace source**
     (`total_ratio` is the tempo proxy); **WNBA shares the basketball voice**.

  File layout (300-line cap on CODE only — bank strings are external JSON): `stories/legs.py`
  (parse + `enrich_legs` + `_stat_category`), `why.py` (unchanged), `context.py` (`GameCtx`/`Leg`,
  `build_game_context`, parquet↔ctx codec, classifier + ratio constants), `engine.py` (`thesis`,
  archetype router, variant seed, slate-uniqueness machinery), `thesis.py` (slims to the
  `attach_parlay_theses` adapter), `bank.py` (JSON loader + fallback) + `data/config/voice_bank.json`.
  The stories package stays import-safe (no `Archive()` at module load — pinned by
  `test_dashboard_no_archive_lock.py`) so the P3 rail can import and recompute the engine live. Bump
  `stories_version`.

  Acceptance: context-builder golden (synthetic two-team offers → exact row; game_total / spread /
  ratio hand-computed); a synthetic slate hits all five shapes (the all-grind repro fails);
  archetype routing unit tests per gate (the no-standout fixture routes to game-script, not an
  alphabetical star; stack / unit / player gates); `thesis()` determinism (same inputs + date →
  byte-equal); per-league bank-coverage golden (every league × archetype × shape × direction
  resolves with no KeyError; football/hockey/baseball render sport-correct vocabulary from synthetic
  fixtures); headline-uniqueness-within-slate still holds; persist characterization updated;
  integration -n0 shows the new file + `Position` flowing; archive-lock + design goldens green;
  refactoring-specialist on every touched `.py`. Est: 3–4 sessions.
- **P3 — slip engine + story menu.** Goal: the story-menu workflow that replaces `Family`, the rail
  on all surfaces, both joint probs, payout/EV/kelly, play-type chip rule (single constant), save +
  reflect grading, Sleeper unverified state.

  **Story menu (pipeline precompute — replaces `Family` as the Slips grouping key).** `prophecize`
  writes `current_game_stories.parquet` keyed `(platform, League, Game, story_id, objective)` →
  `{legs, headline, joint_p, model_ev, kelly_stake}`. Per `(platform, game)`, enumerate up to **5**
  stories straight from the game's real signal — strong individual legs (edge ≥ the menu floor;
  start at the 0.05 the unit gate and per-offer "why" already use) and the correlation clusters
  among them. **No "always" archetype** — a thin game yields few or zero stories; game-script
  surfaces only when the shape is itself the signal (the engine keeps its game-script *fallback* for
  labeling a user-built slip, but the generator has none). Eligible leg types are player props,
  **Rivals** matchup legs, and — behind **L3** — game lines, which double as correlation anchors that
  pull player legs into a story (a DFS line diverging from our weighted consensus is the edge). For
  each story emit two parlays over its eligible legs, **free to share legs**:
  - **Bankroll Builder** (sun motif) = max expected **log-growth** G — the full-Kelly geometric
    growth rate, i.e. the bet that compounds bankroll fastest, *not* the one with the biggest stake.
  - **Shoot the Moon** (moon motif) = max **Model EV** — the widest high-edge set inside the
    play-type cap; usually the 5–6-leg extension of the 2–3-leg Builder.

  Score every candidate leg-subset through the existing copula joint-prob → Kelly/EV path (reuse
  `find_correlation`'s scorer, don't rebuild); brute-force over subsets is fine at ~8–12 offers a
  game. Platforms compute independently and **the same story repeating across platforms is expected
  and good** — same story across books = a stronger data-backed call.

  **Slips flow.** Platform select (Underdog/Sleeper) → up to 5 **story cards** (engine headline +
  one-line why) → two **variant chips** (Bankroll Builder / Shoot the Moon) → the chosen parlay
  loads into the constellation editor (P4). Menu stakes are a **standalone preview** (the parlay's
  own full-Kelly fraction of bankroll — "if this were your only bet"); the live rail re-runs the
  joint slate allocation over whatever is actually in the slip — the real allocator, the same
  bankroll slider the pickem page carries.

  **Live thesis regen:** the rail recomputes `thesis(legs, ctxs)` (the P2 engine) on every slip edit
  (add/remove/swap) via a thin `dashboard/slip/story.py`, so a user-edited slip never shows a stale
  headline naming a removed player. Legs come from slip state enriched against the loaded offers
  frame; `GameCtx`s from a new `load_current_game_context` + the existing `load_current_game_corr`.
  An unedited loaded slip shows the precomputed `Thesis` verbatim (it may carry a slate-uniqueness
  bump the pure function can't reproduce); the first edit switches to the live recompute (the
  single-slip recompute skips the slate-uniqueness pass — faithfulness beats slate uniqueness for
  one slip). A slip whose game lacks a context row degrades to shape "even" routing, never a crash.

  Acceptance: story-enumerator unit tests (cap 5; a no-signal game yields zero; edge floor honored;
  correlation-cluster stories form); two-objective construction (Builder's G ≥ any sibling's growth,
  Moon's EV ≥ Builder's EV, shared legs allowed); per-platform menus independent; standalone-preview
  vs slip-allocated sizing diverge as designed; rail unit tests (2-leg hand-computed joint; 3→4 leg
  play-type boundary; Decimal quantization); removing the named player changes the headline and the
  old name never renders; missing-context slip degrades to "even"; manual: slip survives full nav
  cycle; backdated slip grades on reflect; archive-lock golden green. Est: 4–5 sessions (the menu
  generator is the pipeline half). Kill branch: if copula import drags weight, inline the 30-line
  MVN-cdf math instead.
- **P4 — Tonight + Game.** Goal: narrative surfaces (cards, prophecies, why-strings, context
  strip, constellation v1, `?game=` links, scars mounted). MUST FIX (P1 known bug): Tonight's
  View-game button lands on the Game page's default game — `st.switch_page` drops query params
  set in the same run (tonight.py sets `st.query_params["game"]` then switches). Hand off via a
  plain `st.session_state` key; game.py reads session-state first, `?game=` second (deep links).
  **Context strip + headlines read `current_game_context.parquet`** (the P2 artifact): the Game
  context strip shows total / derived spread / favorite (replacing the per-row `O/U`/`Moneyline`
  peek at `surfaces/game.py:51-56`); Tonight cards show the top per-leg-set thesis headline per
  game. **Constellation v1 doubles as the slip editor the P3 story menu loads into** — nodes are the
  game's candidate legs (player props, Rivals, L3 game lines), edges the `current_game_corr` ρ;
  add/remove/swap drives the live rail + thesis regen (graph viz here, the seeded parlay + rail
  mechanics are P3). Acceptance: render pins; archive-lock gate green; View-game lands on the clicked game
  (doubleheaders included); context strip shows the derived spread for a fixture game. Est: 1–2
  sessions.
- **P5 — Board + Slips upgrade.** Goal: themed AG Grid (right-aligned numerals, edge heatmap,
  sparkline, prophecy lenses), load-into-rail. Entry: pin `streamlit-aggrid` version. Acceptance:
  grid-options golden (token colors, no centered numerics). Est: 1–2 sessions. Kill branch: any
  grid the theme fights falls back to `st.dataframe` + `column_config`.
- **P6 — deep-dive v2 + swap.** Goal: SHAP-ranked chips flip charts, case panel, market-trust →
  Lab deep link, swap dialog (story fit, EV deltas, anti-corr flags). Acceptance: chip-ranking +
  swap-ordering unit tests. Est: 1–2 sessions.
- **P7 — Receipts.** Goal: skeptic page (hero units, EV>5% record, CLV beat, worst month,
  grids, Profit Sim fold-in, your-slips). Acceptance: aggregation unit tests on fixture history;
  hero parity vs old Overview totals. Est: 1–2 sessions.
- **P8 — assets + celestial polish.** Goal: player/team assets, ambient slot scaffolding +
  manifest, display-font injection, nebula hero cards, prophecy naming sweep. Acceptance: asset
  unit tests (no network); design gate green; manual fallback check (break one URL). Est: 1–2
  sessions.
- **L1 — line-movement export.** `export-line-movement` cron (Archive→parquet) + run_job.sh case
  + crontab + healthcheck + CLI snapshot; dialog tab flips on. Owner wires cron.
- **L2 — comps persistence.** Persist comp outputs at prophecize time; comps panel + chip flip
  on.
- **L3 — game lines into the correlation engine.** Book-implied probs only; Game-board rows +
  constellation team nodes + slip legs flip on, **and game lines become eligible P3 story-menu legs**
  — a game line whose book-implied prob diverges from our weighted consensus is a candidate strong
  leg and a correlation anchor that pulls player legs into a story. Re-verify UD product first (§3).
- **L4 — correlation-block risk.** UD/Sleeper pairing-rule model; rail chip flips on.
- **L5 — ambient art acquisition.** Stock/commissioned per spec §6 brief; manifest fill-in.
- **L6 — optional free-LLM prose rewriter.** Only if a free API exists; templates remain the
  fallback contract.

## 7. Working rules

- Dashboard reads parquet snapshots only, never DuckDB — CLAUDE.md §Hard rules; the archive-lock
  golden test is the gate.
- Never load `corr_same_team.parquet`/`corr_opposing.parquet` from the dashboard (NBA = 2.85M
  rows); per-game slices only.
- Snapshot schema changes are append-only: add columns/files, never rename/remove (server tracks
  `devel`; old process + new snapshot must coexist).
- Slip state lives under plain non-widget `st.session_state` keys; rail renders from `app.py` so
  its widgets exist on every page.
- Money is `Decimal` (CLAUDE.md); play-type rule is one named constant in `slip/math.py`.
- All display copy follows the spec §2 taxonomy (platforms = UD/Sleeper; Power/Flex = chip;
  Rivals = leg type).
- Conflict order: command output > CLAUDE.md/CONTRIBUTING.md > DESIGN.md/spec > this brief >
  roadmap v3.

## 8. Escalation & stop conditions

STOP and ask the owner when: gates red at session start through no fault of yours; any change to
payout tables or gate constants; anything touching crontab, creds, paid APIs, or scraping ToS;
editing outside §5 footprint; two consecutive sessions with no acceptance criterion moving.
PARK AND PIVOT when blocked externally: ledger line + status `BLOCKED (on: …)` + point owner at
roadmap v3 §4.
DISPATCH: refactoring-specialist per the five CLAUDE.md triggers; devel-ship-curator for every
devel-bound PR; research-analyst only if a stage turns into a modeling question (none planned).

## 9. Session definition of done

- refactoring-specialist ran on every `.py` touched this session (CLAUDE.md five-trigger rule).
- `poetry run ruff check src/sportstradamus/` clean.
- `poetry run pytest tests/golden/` clean.
- `poetry run pytest -m integration -n0` clean, then `touch .claude/.state/integration_green`.
- One ledger line appended to §10; status line updated if a stage boundary was crossed.
- Never push `devel` directly — devel-ship-curator carves ship PRs.
- Durable non-obvious lesson? Offer a memory capture (CLAUDE.md §Agentic workflow conventions).

## 10. Ledger (append-only, newest first, cap ~15)

- 2026-06-12 · P3b fix ✔ · slip mispriced at **+2530% EV** (owner). `dashboard/legs.py:find_offer_idx` masked on Player/Bet/Market/Line only — the `platform` arg just translated the market label, never filtered rows. `current_offers` holds BOTH books' rows for the same leg at different boost scales (UD raw promo ~1.0 after the `cli.py:163` ÷1.78; Sleeper raw API mult ~1.7–3.6), so `matches[0]` returned the lower-index Sleeper row (boost 1.78) for an Underdog-seeded slip → `score_slip` ran ∏1.78 × Power `base[3]`=6 ≈ 36x. Fix = `mask &= offers["Platform"] == platform` when a platform is given; the parlay reprices to +350% (UD) / +339% (Sleeper, ∏boosts) — the residual ~+340% is model calibration, separate. Regression test pins the same-leg-on-both-books case. See [[dashboard-offer-resolution-platform]]. gates ruff ✓ / golden ✓ / integration ✓.
- 2026-06-12 · P2 hotfix ✔ · `prophecize` crash in `write_current_game_context`. `context.py:_game_row` did `grp.loc[mls.idxmax(), "Team"]`; `offers.groupby` keeps the offers frame's index, and a real slate had a label repeated across two teams (`23 SAS` / `23 NYK`) → `.loc` returned a 2-row Series → `fav_team` non-scalar → pyarrow `ArrowInvalid`. Fix = `grp.reset_index(drop=True)` at the top of `_game_row`; regression test pins a duplicate-index two-team game (scalar `fav_team` + parquet round-trip). Pre-existing since the P2 v2 thesis-engine landing (`9b6ff2a`), surfaced now. gates ruff ✓ / golden ✓ (the 2 known reds only) / integration ✓.
- 2026-06-12 · P3b ✔ · slip builders + locked shelf + grading landed (dashboard half of P3). **Two builder types, one sidebar shelf** (the copula is only valid within a game): a **constellation** builder (same-game, correlation + live thesis; hosted on Slips) and a **simple** builder (any-game, grade-only; hosted on Board) — both lock to `data/runtime/user_slips.parquet` (`USER_SLIPS_PATH`) and the global sidebar shelf, both edit-reopen by `builder_type`. New pure `dashboard/slip_engine.py` is the one sanctioned live calc: `score_slip(legs, corr, *, platform, bankroll, shrinkage)` builds a **block-diagonal** SIG (within-game ρ from the `current_game_corr` slice keyed `Player|Market|Bet`, cross-game pairs ρ=0 → joint collapses to the independent product), repairs via `_psd_or_none`, prices through `_parlay_payout_prob` (gate-free copula core, **never `_evaluate_parlay`** — its both-teams gate drops single-team stacks); `slip_headline` reuses the P2 `thesis_variants` and **sorts legs canonically** so the headline is a pure function of the leg-set (path- + order-independent). **Platform pricing:** Underdog = pooled Power(≤3)/Flex(4+) curve × ∏boosts; **Sleeper = ∏ per-leg boosts** (base 1.0; correlation-discount factor deferred to the Sleeper-parity track → `payout_approximate=True` scar caption, EV shown not suppressed). Money is `Decimal` (`fractional_kelly_stake`). Three constellation seeds (story cascade `platform→game→story→Bankroll-Builder/Shoot-the-Moon`; Game-tab same-game multiselect; sidebar edit) + one simple seed (Board cross-game multiselect). **Grading:** `nightly._resolve_user_slips` mirrors `_resolve_parlays` but over per-leg `analysis._resolve_leg` (a slip may span games) — Power all-or-nothing, Flex/Sleeper partial-cash floor (≥2 hits, ≤2 misses), unplayed games keep the slip `pending`; counts surface in `resolve_meta`. Receipts gained a "Your slips" record panel. Legs are snapshotted from `current_offers` at seed/add (Desc carries the `- {pct}%, {boost}x` tail so `parse_leg` *and* the grading `LEG_PATTERN` both parse it). · gates ruff ✓ / golden ✓ (1751 pass; **1 pre-existing unrelated red** — stats `get_training_matrix` data-drift digest; pickem mvn.cdf xdist flake intermittent, passes -n0) / integration ✓ (14 pass) · refactoring-specialist ✓ (inlined `_bankroll` wrapper, dropped 2 section banners + 1 what-docstring) · deleted stale `test_predictions_parlays_render_characterization.py` (pinned the rewritten-away `slips.py::_render_parlay`) · 4 new golden files (slip_engine / headline-determinism / user_slips_io / nightly_user_slips); AppTest smokes deferred (no AppTest pattern in-tree, per golden-cull) · manual dashboard walkthrough not run unprompted · next: P4 constellation *visual* (starfield/edge nodes) + swap-dialog polish + context strip + celestial theming + Sleeper correlation-discount factor
- 2026-06-12 · P3a ✔ · story-menu generator landed (pipeline half of P3; dashboard half = P3b next). New `current_game_stories.parquet` (`CURRENT_GAME_STORIES_PATH`), one menu per `(platform, Game)`: ≤5 correlation-cluster stories × 2 objectives (`builder`/`moon`), keyed `(platform, League, Game, story_id, objective)` → `legs`(JSON)/`headline`/`joint_p`/`model_ev`/`kelly_stake`(full-Kelly fraction)/`bet_size`/`Date`. Pure `prediction/stories/menu.py:build_game_stories`: greedy ρ-graph clustering over strong legs (per-$1 edge ≥ 0.05), then a two-phase subset search — cheap independent-joint proxy ranks all subsets, exact copula (`parlay._parlay_payout_prob`) scores only a shortlist (top-K/objective ∪ all Power subsets) — Builder = argmax single-bet full-Kelly log-growth `_log_growth`, Moon = argmax model EV within the Power/Flex cap. **Scores raw via the gate-free copula core, NOT `_evaluate_parlay`** (its `_parlay_admissible` requires both teams → would silently drop single-team stacks). Per-game scoring bundle exposed via a new opt-in `story_sink` on `find_correlation`/`process_offers` (`GameScoringContext` dataclass in parlay.py; append-only, mirrors `corr_sink`; `_append_story_context` helper keeps `_process_league_games` ≤ CC10). cli builds the menu from the already-built `game_context` + `corr_sink`. `STORIES_VERSION` → `p3a`. **Perf:** menu pass ~5.5s realistic busy slate / ~13s worst case (10×40 fully-correlated strong legs, 1.3s/game) — negligible vs the 15-min budget; the shortlist bounds the 50k-sample flex MC (`_PUSH_MC_SAMPLES` untouched). Note: ≥3-leg `model_ev` carries ~1e-4 noise (scipy `mvn.cdf` is randomized QMC for dim ≥ 3 — same as the existing parlay snapshot). · gates ruff ✓ / golden ✓ (1740 pass; **1 pre-existing unrelated red** — stats `get_training_matrix` data-drift digest; pickem mvn.cdf xdist flake intermittent) / integration ✓ (14 pass, story writer fires column-stable) · refactoring-specialist ✓ (formatting + 1 type annotation only) · live-budget smoke deferred (not run unprompted — scrapes + writes archive + collides with :50 cron) · next: P3b slip rail + Slips story-card UI + live thesis regen + save/reflect grading
- 2026-06-12 · P2 ✔ · thesis engine v2 landed. Four-archetype router `route(legs,ctxs)` → player/stack/unit/game-script (`engine.py`), pure + bank-free, no `Family` input; player gate requires a *unique* leg-majority (kills v1's alphabetical star — a 2-2 tie routes to game-script). New `current_game_context.parquet` (one row per League/Game/Date: `game_total`/`spread`/`fav_team`/`ml_margin`/`total_ratio`/`shape`/`pos_edges`) from pure `build_game_context`; league-relative `total_ratio` band classifier replaces the dead `_TOTAL_BANDS` (the all-grind repro now spans all five shapes). `Position` depth labels written back in `correlation.py` + kept in `_OFFER_KEEP_COLS`. Voice banks externalised to `data/config/voice_bank.json` (5 voices basketball/football/hockey/baseball/shared, 625 variants, v1's 107 preserved verbatim) + `bank.py` loader w/ guaranteed-hit fallback chain. cli builds context once, feeds the same frame to the thesis pass *and* its writer (headline ↔ artifact agree). stories package stays `Archive()`-free for the P3 live rail. `STORIES_VERSION` → `p2-2`. Live-regen → P3, context strip → P4. · gates ruff ✓ / golden ✓ (1728 pass; **1 pre-existing unrelated red left untouched** — `stats` `get_training_matrix` data-drift digest, fails on lane HEAD with P2 stashed) / integration ✓ (14 pass, new `current_game_context` + `Position` flow asserted) · refactoring-specialist ✓ (inlined 3 single-expr helpers, deleted orphan `thesis()`) · next: P3 slip engine
- 2026-06-12 · P2 reopened (status P3→P2) · thesis engine v2 folded into the P2 stage entry. Shipped v1 generator tells one story (featured player, alphabetical tiebreak) and its game-shape classifier is dead in prod — `O/U` is the team half-total (~108 NBA / ~82 WNBA), not the 235 `_TOTAL_BANDS` expect, so every non-lopsided game classifies "grind". P2 now also owns: `current_game_context.parquet` precompute + ratio-band classifier fix + kept `Position` labels; the pure `thesis(legs,ctx)` archetype engine (player/stack/unit/game-script, `Family` demoted to Slips grouping); per-sport voice banks for all five leagues authored from game knowledge + synthetic-fixture tests. Live-regen folds into P3, context strip into P4. Spec: `docs/superpowers/specs/2026-06-11-thesis-engine-v2-design.md`. · next: game-context precompute + classifier fix
- 2026-06-11 · P2 · pipeline precompute: `prediction/stories/` generator (per-family `Thesis` + per-offer `Why`; 107-variant date-keyed prophecy bank; slate-uniqueness pass) replaces & retires the render-time phrase bank; `Game`/`K`/`Why` offer keep-cols; `current_game_corr.parquet` via opt-in `find_correlation(corr_sink=…)` (return-arity avoided per kill-branch); `stories_version` in meta; Slips reads precomputed `Thesis`; `parse_leg` moved to prediction layer (dashboard re-exports). Generator subagent-built then carved to a <300-line package; refactoring-specialist pass (reverted its 2 out-of-footprint pre-existing-comment edits in io.py/cli.py). · gates ruff ✓ / golden ✓ (494 pass; **2 pre-existing unrelated reds left untouched** — `stats` `get_training_matrix` data-drift + pickem SciPy-cdf xdist flake, both fail on lane HEAD with P2 stashed) / integration ✓ · next: P3 slip engine
- 2026-06-11 · P1 · six-surface package landed, legacy pages deleted, 2 review rounds + quality fixes (2eed419..66ede1c); host crash mid-commit recovered via reflog repoint + working-tree recommit; 2 latent profit-sim crashes fixed at source (strategies/profit_sim.py — outside-footprint touch, disclosed) · gates ✓/✓/✓ · next: P2 pipeline precompute (stories.py, keep-cols, corr slices)
- 2026-06-11 · P0 · celestial tokens + spec + lane + mockups landed (6a33b81) · gates ✓/✓/✓ · next: P1 package migration
