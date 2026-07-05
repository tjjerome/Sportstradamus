# P8 Phase R — Remediation & Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking — **tick them as you go and write the §10 ledger entry
> at the end; Phase 0/A/B/C skipped both and it cost a full review cycle.** Sized for an
> **Opus-class implementer**.

**Goal:** Close the gap between the shipped P8 phases 0/A/B/C and the locked mockups: finish the
half-run data migration, fix six runtime-only bugs the goldens could not see, and add the
celestial skin layer (page heroes, tonight cards, whitespace starfield) that no earlier plan
task ever assigned.

**Architecture:** No new subsystems. One data-ops task (R0), one app-shell task (R1), then four
surface task groups (R2–R5) over existing files. Two owner decisions are already made and are
binding: **(1) the app keeps `layout="wide"`; the starfield must show through all page
whitespace — only genuinely data-dense blocks (grids, dataframes, cards, dialogs) occlude it;
(2) the animated twinkle accents are exempt from the DESIGN §3 ≤0.20 static-ambient ceiling**
(cap them at 0.7 peak; static layers stay ≤0.20) — amend DESIGN.md accordingly (R1).

**The prime directive — browser verification.** Every root cause below shipped "green": the
goldens pin Python-side structures (gridOptions dicts, CSS strings, figure traces) while AG Grid
v34 / Streamlit 1.58 render something else at runtime. Therefore **every task in this plan ends
with a live check**: `poetry run dashboard`, look at the page, and record a one-line verdict
(what you saw, not what you expect) in the commit body. A task without a recorded live verdict
is not done. If you cannot see it fixed, it is not fixed.

**Context — how we got here (read once):** Phases 0/A/B/C were implemented by earlier agents and
merged (devel `84c3342`). The owner's live review found the app flat and partly broken. A
four-agent audit traced everything to three layers: (a) the leg-schema migration
(`scripts/migrate_leg_schema.py`) ran Jul 4 08:02, migrated `parlay_hist.parquet`, then **died
inside the history step** — `data/runtime/history.parquet` is still the old nested schema, so
Receipts/Diagnostics/Calibration render empty; (b) no plan task ever assigned the mockups'
page-hero treatment, tonight-card rebuild, or a wide-layout starfield, so surfaces are stock
`st.title` over a flat background; (c) six runtime bugs (detailed per task below). Mockups in
`docs/mockups/p8-*.html` are pixel truth. DESIGN.md holds FIXED tokens — read it before R1.

**Branch:** work on `feature/dashboard-ux` (currently 4 commits behind devel — start with
`git merge devel`). Gates before every commit: `poetry run ruff check src/sportstradamus/`,
`poetry run pytest tests/golden/`, `poetry run pytest -m integration -n0`, plus the
refactoring-specialist subagent on every touched `.py` before any push/review (CLAUDE.md).

---

### Task R0: Finish the migration + real reflect + kill the fake resolve banner

**Files:**
- Run (not edit, unless it crashes again): `src/sportstradamus/scripts/migrate_leg_schema.py`
- Modify: `src/sportstradamus/nightly.py` (~line 550, resolve-meta path), `src/sportstradamus/helpers/io.py`
- Modify: `tests/test_nightly_run_characterization.py`

**State on this box:** `parlay_hist.parquet` migrated (has `legs` structs + `Legs Resolved`);
`history.parquet` NOT migrated (old nested `Offers` schema, no `Line/Bet/Platform/Alt Line/Close
Market Prob` columns); `current_offers.parquet` still carries old corr-string columns;
`resolve_meta.json` contains **leaked test-fixture values** ("Jul 4 22:48, 1 resolved" — written
by `tests/test_nightly_run_characterization.py` through the real CLI path; an xdist
save/restore race left them on disk).

- [ ] **Step 1: verify idempotency of the completed step.** Read `_migrate_parlay_hist` in the
  script and confirm it detects the already-migrated frame and skips (it should — the plan spec
  required idempotency). Only then rerun.
- [ ] **Step 2: run the migration to completion** (babysit it; the history archive-backfill
  loops are the long part — tqdm shows progress):

```bash
poetry run python -m sportstradamus.scripts.migrate_leg_schema --backfill-close --backfill-alt-lines
```

  If it dies again, diagnose (the death point last time was inside `_migrate_history`'s
  `--backfill-close` / `--backfill-alt-lines` archive loops), fix the cause, and add one
  `print(f"OK {step}", flush=True)` line after each top-level step so a future mid-run death is
  visible. Do not add resume machinery beyond that.
- [ ] **Step 3: verify the data** (paste output into the commit body):

```bash
poetry run python -c "
import pandas as pd
h = pd.read_parquet('src/sportstradamus/data/runtime/history.parquet')
assert 'Line' in h and 'Platform' in h and 'Alt Line' in h and 'Close Market Prob' in h, h.columns.tolist()
print(len(h), 'rows', h['Date'].min(), '→', h['Date'].max())
print('platforms:', h['Platform'].dropna().unique()[:6])
cp = h['Close Market Prob'].dropna()
print('close probs in (0,1):', ((cp>0)&(cp<1)).mean())
"
```

- [ ] **Step 4: real reflect.** `poetry run reflect` — writes a truthful `resolve_meta.json`,
  `calibration_summary.parquet` (B6 panel self-heals), CLV segments on the fixed probabilities.
- [ ] **Step 5: fix the test leak.** Hoist the resolve-meta path into `helpers/io.py` as
  `RESOLVE_META_PATH` (nightly.py currently builds it inline via pkg_resources), point
  `nightly.py` at it, and monkeypatch it to `tmp_path` in
  `tests/test_nightly_run_characterization.py` so the suite can never write the real file again.
- [ ] **Step 6: live check** — Receipts shows resolved rows + real platforms + a truthful
  resolved-at line; Diagnostics body renders; Receipts calibration/reliability panel renders.
  Record the verdict. Gates; commit `fix(p8-r): complete leg-schema migration + truthful resolve meta`.

---

### Task R1: The celestial shell — whitespace starfield, page heroes, banner retirement, gold accents

**Files:**
- Modify: `src/sportstradamus/dashboard/theme.py`, `src/sportstradamus/dashboard/app.py`,
  `src/sportstradamus/dashboard/data.py` (banner removal)
- Create: `src/sportstradamus/dashboard/components/hero.py`
- Modify: all seven surfaces (`tonight.py`, `board.py`, `games.py`, `receipts.py`,
  `lab_correlations.py`, `lab_diagnostics.py`, `lab_training.py`) — hero adoption only
- Modify: `DESIGN.md` §3, `tests/golden/test_app_injections.py`, `tests/golden/test_design_tokens.py`

**R1.a — starfield in the whitespace (owner decision 1).** Current state: the layer exists and
injects (`theme.APP_CSS` + `STARFIELD_HTML`, `app.py:25`), `.stApp` is already transparent, but
it was tuned for centered-page gutters that `layout="wide"` doesn't have — 30 dust dots at
≤0.20 alpha vanish under full-width content. Rework:

1. Keep `layout="wide"`.
2. Make the field legible in whitespace: raise dust count (~30 → ~80, mixed 1.5–2.5px), keep
   static dust/wash alpha ≤ 0.20 (DESIGN static ceiling), distribute via the existing
   deterministic percent positions (no randomness at import).
3. Twinkles: keep the >0.20 animation peak — **cap the keyframe peak at 0.70**, ≤ 12 instances,
   brief flash cadence (current stagger fine).
4. Occlusion becomes *selective*: transparent everywhere except data-dense blocks. Add to
   `APP_CSS`: solid `#12151C`-family backgrounds on (a) the AG Grid iframe wrapper
   (`[data-testid="stCustomComponentV1"]` hosting aggrid — scope by a wrapping
   `st.container(key=...)` class if the testid is too broad), (b) `[data-testid="stDataFrame"]`,
   (c) `[data-testid="stDialog"]` content, (d) the sidebar. Cards/heroes keep their own washes
   (mockups use `rgba(26,29,36,.84)`-class translucency — starfield glows through faintly, text
   stays AA).
5. **DESIGN §3 amendment (owner decision 2):** add one sentence — animated twinkle accents are
   exempt from the 0.20 static-ambient ceiling, capped at 0.70 peak alpha, ≤ 12 instances;
   static ambient layers remain ≤ 0.20. Resolve the parked open question in
   `test_app_injections.py:19-27`: pin static-layer alphas ≤ 0.20 **and** twinkle keyframe peak
   ≤ 0.70 (delete the "open question" comment).

- [ ] Adjust CSS/HTML in theme.py; amend DESIGN §3; update the two goldens.
- [ ] **The browser spike that never happened:** `poetry run dashboard`, all seven pages.
  Verify: field visible in gutters/between blocks on every page, invisible behind Board's grid
  and the Lab tables, no text contrast loss, `prefers-reduced-motion` still kills animation.
  Record a three-line verdict in theme.py's module docstring (replacing the DOMPurify-only
  note) AND the commit body.
- [ ] Gates; commit `feat(p8-r): starfield lives in the whitespace (wide layout)`.

**R1.b — `page_hero` + adoption on all seven surfaces.** No P8 task ever assigned the mockups'
header treatment; every page is `st.title` (`tonight.py:41`, `board.py:21`, `games.py:245`,
`receipts.py:94`, `lab_*.py`). Build once, adopt everywhere:

```python
# components/hero.py
import streamlit as st

def page_hero(kicker: str, title: str, updated: str | None = None) -> None:
    """Mockup .head block: gold Cinzel kicker, display headline, quiet updated line."""
    updated_html = f'<div class="hero-updated">{updated}</div>' if updated else ""
    st.html(
        f'<div class="page-hero">'
        f'<div class="celestial-kicker">◈ {kicker}</div>'
        f'<h1 class="celestial-headline">{title}</h1>'
        f"{updated_html}</div>"
    )
```

Add `.page-hero` / `.hero-updated` rules to `APP_CSS`, lifting exact sizes/colors from
`docs/mockups/p8-board.html` (`.head` block, ~lines 26-31 and 74-78). Kickers per mockups:
Tonight "TONIGHT'S SLATE", Board "THE BOARD", Games "THE CONSTELLATION", Receipts "THE
RECEIPTS", Lab pages their mockup kickers (read each `p8-lab-*.html`). The `updated` line is
the timestamp ONLY (CLAUDE.md rule — no feature announcements).

**R1.c — retire the Sheets-era banners.** `data.py:369-379` `render_banner` (+ the
`#1f4e79`/`#2d6a4f` banner classes in theme.py:124-127) predate the redesign, appear in no
mockup, and clash with the skin. Replace every call site with the hero's `updated` argument;
delete `render_banner` and the banner CSS; grep for stragglers.

**R1.d — gold active states.** DESIGN's interactive-highlight role is wired to exactly one
widget pair (the Games lens buttons). Extend the same key-scoped CSS treatment to: the global
sport switch (`app.py:34`), Board's lens/side segmented controls, Receipts' window control.
Streamlit's segmented-control DOM varies — find the stable selector once, verify in-browser,
and leave a one-line comment naming the Streamlit version it was verified against.

- [ ] hero.py + CSS + seven adoptions + banner deletion + gold accents; extend
  `test_app_injections.py` (hero classes present; banner classes gone).
- [ ] Live check every page (kicker + headline + timestamp render, no double titles); record
  verdict. Gates; commit `feat(p8-r): page heroes on every surface; sheets banners retired`.

---

### Task R2: Board runtime fixes

**Files:**
- Modify: `src/sportstradamus/dashboard/components/grid.py`,
  `src/sportstradamus/dashboard/surfaces/board.py`
- Test: `tests/golden/test_grid_options.py`

**R2.a — the escaped-SVG Line column (CRITICAL).** Root cause: streamlit-aggrid 1.2.1 + AG Grid
34 React classifies a *plain-function* `JsCode` cellRenderer as a React functional component, so
its returned string renders as an escaped text node (`grid.py:51-65`). Fix: a **class-based**
renderer — objects whose prototype has `getGui` route down the JS-component path and
`innerHTML` works:

```python
_ARROW_RENDERER_TMPL = """
class ArrowCellRenderer {
    init(params) {
        this.eGui = document.createElement('span');
        const svg = (params.data && params.data.Bet === 'Over') ? %s : %s;
        this.eGui.innerHTML = svg + '<span class="line-num">' + (params.value ?? '') + '</span>';
    }
    getGui() { return this.eGui; }
}
"""
```

(built with `json.dumps(bet_arrow("Over"))` / `json.dumps(bet_arrow("Under"))` — keep the SVGs
sourced from `narrative.bet_arrow` so the tokens stay single-home). Update the golden to pin
`"getGui"` in the emitted gridOptions string — that pins the *class shape*, the live check pins
the render. If the class route fails in-browser, the sanctioned fallback is Unicode ▲/▼ via
`valueFormatter` + `cellClassRules` for green/red — pick one, delete the other.

**R2.b — edge tint paints outliers only.** `_heatmap_cellstyle` buckets at 20%/60% of the
column max (`grid.py:38-39`); with the board sorted by edge, the whole first screen paints
solid. Mockup (`p8-board.html`): most rows unpainted, mild = pale `DIVERGING_COLORS[6]`
(#7FAAE8-family), strong only for real outliers. Replace fraction-of-max with absolute
thresholds (named constants, module level):

```python
_EDGE_TINT_MILD = 4.0    # |edge| % below this: no paint (the mockup's plain majority)
_EDGE_TINT_STRONG = 10.0 # above this: the saturated bucket
```

mild bucket → `DIVERGING_COLORS[6]`/`[2]`, strong → `[7]`/`[1]`, padded/rounded like the
mockup's `td.heat` (border-radius on the cell style), not edge-to-edge paint.

**R2.c — gold row hover.** AG Grid v34 paints hover via an absolutely-positioned `::before`
overlay that sits above the row background the current CSS targets. Replace `_HOVER_CSS` with:

```python
_HOVER_CSS = {
    ".ag-root-wrapper": {"--ag-row-hover-color": "rgba(201, 162, 39, 0.09)"},
    ".ag-row-hover .ag-cell:first-child": {"box-shadow": "inset 3px 0 0 #C9A227"},
}
```

**R2.d — formatting.** (1) Win % stays numeric (0–100 float) with the grid's percent
`valueFormatter` at **1 dp** (mockup "61.1%") — the current pre-baked `f"{x:.2f}%"` string
(`board.py:132-135`) breaks click-sort; (2) edge formatter gains the explicit `+` sign
(`(v>0?'+':'') + v.toFixed(1) + '%'`); (3) Market **filter** options get display names —
`format_func=` over a league-aware slug→`market_display_name` map (column already uses them;
chips show raw slugs, `board.py:73-74`); (4) delete the dead `"Kelly"` branch in the rounding
loop (`board.py:142-144`).

- [ ] Goldens first (getGui pin, hover-var pin, threshold constants pin, Win % numeric dtype
  pin) → implement → **live check**: arrows render as arrows, first screen mostly unpainted
  with a few tinted outliers, hover shows gold wash + rail, Win % sorts numerically, market
  chips readable. Record verdict. Gates; commit
  `fix(p8-r): board runtime — real arrows, outlier-only tint, gold hover, formats`.

---

### Task R3: Tonight — prophecy cards + story-engine headlines

**Files:**
- Modify: `src/sportstradamus/dashboard/surfaces/tonight.py`,
  `src/sportstradamus/dashboard/narrative.py`, `src/sportstradamus/dashboard/theme.py` (card CSS)
- Test: `tests/golden/test_narrative.py` (or the existing narrative test home)

**R3.a — headlines from the story engine.** Every prophecy line app-wide is blank because
`tonight.py:88` / `games.py:105` read `current_parlays.parquet` (0 rows this run) and were never
rewired to the p3b story engine per spec §4.2 — `current_game_stories.parquet` HAS headlines.
Add to `narrative.py`:

```python
def game_headline(stories: pd.DataFrame, parlays: pd.DataFrame, game: str, date) -> str:
    """The game's oracle line: the story-engine headline for (Game, Date) —
    highest model_ev story wins; falls back to top_thesis(parlays) when the
    stories frame has no row for the game; '' when both are empty."""
```

Tonight cards and the Games hero subline (R4) both call it. Unit-test the precedence with tiny
fixtures (story wins over parlay; parlay fallback fires; double-empty → "").

**R3.b — the nebula card.** `tonight.py:92-104` is a plain `st.container(border=True)`. Rebuild
as an HTML card (same `st.markdown(unsafe_allow_html=True)` pattern the Games hero uses),
porting the mockup block `docs/mockups/p8-tonight.html:43-64` exactly: `#3a3450` border,
two-stop nebula wash over `rgba(26,29,36,.84)`, kicker line (`WNBA · {tip time}`), 22px matchup
line, gold Cormorant prophecy line (R3.a's headline), green `+X.X% top edge` badge, right rail =
58px glyph + Cinzel shape name. Card CSS goes in `APP_CSS` once (class `tonight-card`), not
inline per card. Keep the `View game` `st.button` under the card (Streamlit needs a real widget
for nav; style it quiet).

**R3.c — delete the stray footer.** `tonight.py:118` renders "Prophecies arrive with the next
data wave." unconditionally under populated cards; the true empty-state at line 53 already
covers the no-games case.

- [ ] Fixture test for `game_headline` → implement → rebuild cards → delete footer → **live
  check** (cards show wash/kicker/headline/badge/glyph; headline text present for games with
  stories; no stray footer). Record verdict. Gates; commit
  `feat(p8-r): tonight prophecy cards + story-engine headlines`.

---

### Task R4: Games — hero voice, chrome, constellation presence

**Files:**
- Modify: `src/sportstradamus/dashboard/surfaces/games.py`,
  `src/sportstradamus/dashboard/components/constellation.py`,
  `src/sportstradamus/data/config/team_assets.json` + `src/sportstradamus/scripts/build_team_assets.py`,
  `src/sportstradamus/dashboard/theme.py` (team_name loader beside team_colors)
- Test: `tests/golden/test_constellation.py`, team-assets golden

**R4.a — hero completion.** The wash/typography scaffolding matches the mockup already
(`games.py:87-129`). Add: (1) the gold prophecy subline — `game_headline(...)` from R3.a (the
`.celestial-headline` div at `games.py:107-111` finally gets text); (2) **full team names** —
extend `build_team_assets.py` to emit a `"name"` field per team (each league's stats class
already knows full names; regenerate the JSON), add `theme.team_name(league, abbrev)` with
abbrev fallback, hero renders "Golden State @ Atlanta" instead of "GSV @ ATL"; (3) page header
via `page_hero("THE CONSTELLATION", ...)` (R1.b) — the legacy blue banner call
(`games.py:245-247`) dies in R1.c; (4) Platform + Game pickers into one compact
`st.columns([1, 2])` row (`games.py:176, 210` are stacked full-width).

**R4.b — lens chips.** Figure modes are done and pinned; the chrome isn't: `games.py:236-242`
lays two buttons in half-width columns → stray buttons with a gulf between. Render a tight
inline row — `st.columns([1.2, 1.2, 5.6])`, Cinzel micro-label "LENSES" above (one caption),
buttons keep the existing key-flip gold-active CSS. Compare against
`docs/mockups/p8-games-lenses.html` chip row.

**R4.c — constellation presence (the "5 gray stars in a corner" fix).** Four small changes in
`constellation.py`, each behind a named constant, grammar untouched:

1. **Re-center `_rescale`** (`constellation.py:499-505`) — the latent bug: it scales by
   max-|coord| but never re-centers, so a one-sided game (all legs one team) piles at x≈+0.6
   against fixed [-1.6, 1.6] axes. Subtract the bbox centroid before scaling:

```python
xs = [x for x, _ in pos.values()]; ys = [y for _, y in pos.values()]
cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
# then scale (x - cx, y - cy) by the existing max-|coord| rule
```

   Extend the layout golden: a hand-built single-team pool's bbox center lands within 0.05 of
   the origin.
2. **Candidate stars keep their hue.** `_desaturate(c, 0.55)` + opacity 0.45
   (`constellation.py:72-74, 553, 564`) crushes dark franchise colors to uniform gray. Change to
   desat 0.35 / opacity 0.60 (`_CANDIDATE_DESAT`, `_CANDIDATE_ALPHA`) — dim vs in-slip, still
   team-colored.
3. **The web shows cold.** `_add_edge` (`constellation.py:508-532`) emits non-slip edges at
   opacity 0 — an empty slip renders zero lines. Give them a faint base
   (`_EDGE_BASE_ALPHA = 0.08`), full gold stays slip-only; hover preview unchanged.
4. **Sky dressing** — port from `docs/mockups/p8-games.html:61-66, 164-181`: `paper_bgcolor`
   `#12151C` + two nebula-wash `layout.shapes` in `_blank_figure`, Cinzel team-tag annotations
   at the left/right edges (use `team_name`), brighter `textfont` on in-slip labels vs
   candidates. Widen the size spread while there: `_SIZE_MIN/MAX` 18–30 → 14–38.

- [ ] Goldens first (re-center pin; candidate-alpha/base-edge constants pinned in figure
  traces; team-tag annotations present) → implement → **live check** on a real slate: stars
  team-colored and spread across the canvas, faint web visible pre-pick, team tags framing the
  map, hero speaks a prophecy over full team names, lens chips tight. Record verdict + a
  screenshot note. Gates; commit `feat(p8-r): games hero voice + constellation presence`.

---

### Task R5: Lab & Receipts — perf death, double filters, glyphs, formats

**Files:**
- Modify: `src/sportstradamus/dashboard/surfaces/lab_correlations.py`, `lab_diagnostics.py`,
  `lab_training.py`, `src/sportstradamus/dashboard/data.py`,
  `src/sportstradamus/dashboard/components/grid.py`, `gate_matrix.py`
- Test: extend the relevant lab goldens

**R5.a — Lab Correlations perf death (page renders nothing).** `lab_correlations.py:39-40`
eager-loads the full 1.68M-row `parlay_hist.parquet` — 7.2 GB in RAM, 22 s read, then three
full-frame copies and a per-rerun `to_csv` **including the `legs` struct column**. The heatmap
code below it is fine and its data exists. Fix:

1. Column-project the read: `pd.read_parquet(PARLAY_HIST_PATH, columns=[...])` with only the
   scalar columns this page uses (no `legs`, no pair/thesis columns) — enumerate them from the
   page's actual usage.
2. Downsample the scatter: `df.sample(n=min(len(df), 20_000), random_state=17)` with a caption
   stating the sample size when truncated.
3. The CSV export drops struct columns (export the same projected frame).
4. Measure before/after wall-clock in the live check; the page must render in seconds.

**R5.b — one Filters sidebar, coverage tile rehomed.** `lab_diagnostics.py:66-72` renders its
own sidebar "Filters" header + Time window, then calls `sidebar_filters` (`data.py:532`) which
renders a second one. Fold the Time window into the shared block (parameter on
`sidebar_filters`, rendered under the same header) and delete the page-local section. Move the
"Distribution Data Coverage" metric (`data.py:562-564`) out of `sidebar_filters` — it renders
as a small captioned metric in the Diagnostics page body (it is a diagnostics fact, not a
filter).

**R5.c — Training's double filter UI.** `lab_training.py:273-284` seeds `lab_markets` with
**all** markets (a wall of chips) above the shared panel's "No filters active" caption — the B5
plan said the legacy widgets *move into* the shared panel; they were kept as body widgets
instead. Delete the body league/market multiselects; default the shared panel to no market
filter (= all, zero chips); keep Lifecycle wherever it currently lives (panel if present, else
add it to the panel).

**R5.d — gate matrix glyphs, not blocks.** `grid.py:180-188` `_glyph_cellstyle` reuses the
heatmap expression and sets `backgroundColor` → solid red/green cells. The mockup locks colored
**text** (`p8-lab-training.html:60,221`: green ● / red ○ only). Emit
`{'color': <token>, 'fontWeight': 600}` and no background. Tokens stay `theme.GREEN/RED`.

**R5.e — number formats.** `brier_skill_score` renders full float repr in the gate matrix and
every Training tab. Gate matrix: `toFixed(3)` valueFormatter for its numeric columns
(`gate_matrix.py:74` / `grid.py:117-142`). Tab tables: `st.column_config.NumberColumn(format="%.3f")`
for float metric columns (`lab_training.py:223-225`).

- [ ] Implement all five → **live check**: Correlations renders (heatmap + charts) in seconds;
  Diagnostics has one Filters section + coverage in the body; Training shows zero default
  chips + one filter home; matrix shows colored glyphs on dark cells; three-decimal metrics.
  Record verdict + the Correlations timing. Gates; commit
  `fix(p8-r): lab pages — corr perf, single filters, glyph colors, formats`.

---

### Task R6: Exit — full walkthrough, ledger, checkboxes

- [ ] Full seven-surface walkthrough against the mockups (`p8-tonight/board/games/
  games-lenses/receipts/lab-*.html`), with the migrated data live. For each surface, one line:
  matches / acceptable delta (named) / defect (file a follow-up in the handoff, don't silently
  fix out-of-scope things).
- [ ] Tick every completed checkbox in THIS plan; append the §10 ledger entry to
  `docs/handoffs/dashboard-ux.md` (date, what shipped, live-check verdicts, gate results,
  what's still open for D/E).
- [ ] Final gates + refactoring-specialist across every touched `.py` this phase; do not push —
  the owner reviews first.

## Exit criteria

- Receipts/Diagnostics/Calibration render real resolved data; resolve banner tells the truth.
- Board: real arrows, outlier-only tint, gold hover, sortable Win %, readable market chips.
- Every page opens with kicker + headline + timestamp over a starfield that reads in the
  whitespace; twinkles ≤ 0.70 peak; static layers ≤ 0.20; banners gone.
- Tonight cards and the Games hero speak story-engine prophecy lines; constellation shows
  team-colored, centered stars over a faint web with team tags.
- Lab Correlations renders in seconds; single filter homes; glyph-colored gate matrix;
  3-decimal metrics.
- Every task's commit body carries a live-check verdict; plan checkboxes ticked; ledger entry
  written; three gates + refactoring-specialist green.
