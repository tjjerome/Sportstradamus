# Dashboard UX — "the Oracle"

> Status: ACTIVE — stage P2

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
  legs), informational chip only. Rivals = leg type. Internal `contest_variant` field names
  unchanged.
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
- **P2 — pipeline precompute.** Goal: `stories.py` why-strings/theses, `K`/`Why`/`Game` keep-cols,
  `Thesis` on parlays, `current_game_corr.parquet`, `stories_version` in meta. Acceptance:
  persist characterization updated; stories golden test (deterministic, hash-stable); corr-slice
  golden (symmetric, joins to offers); integration -n0 shows columns flowing. Est: 1–2 sessions.
  Kill branch: if corr-slice via `find_correlation` return-arity change breaks pins, fall back to
  a collector param.
- **P3 — slip engine.** Goal: rail on all surfaces, both joint probs, payout/EV/kelly, play-type
  chip rule (single constant), save + reflect grading, Sleeper unverified state. Acceptance:
  unit tests (2-leg hand-computed joint; 3→4 leg play-type boundary; Decimal quantization);
  manual: slip survives full nav cycle; backdated slip grades on reflect. Est: 2–3 sessions.
  Kill branch: if copula import drags weight, inline the 30-line MVN-cdf math instead.
- **P4 — Tonight + Game.** Goal: narrative surfaces (cards, prophecies, why-strings, context
  strip, constellation v1, `?game=` links, scars mounted). Acceptance: render pins; archive-lock
  gate green. Est: 1–2 sessions.
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
  constellation team nodes + slip legs flip on. Re-verify UD product first (§3).
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

- 2026-06-11 · P1 · six-surface package landed, legacy pages deleted, 2 review rounds + quality fixes (2eed419..66ede1c); host crash mid-commit recovered via reflog repoint + working-tree recommit; 2 latent profit-sim crashes fixed at source (strategies/profit_sim.py — outside-footprint touch, disclosed) · gates ✓/✓/✓ · next: P2 pipeline precompute (stories.py, keep-cols, corr slices)
- 2026-06-11 · P0 · celestial tokens + spec + lane + mockups landed (6a33b81) · gates ✓/✓/✓ · next: P1 package migration
