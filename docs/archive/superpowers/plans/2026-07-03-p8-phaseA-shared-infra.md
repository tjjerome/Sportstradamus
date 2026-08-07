# P8 Phase A — Shared Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land every cross-surface primitive P8's surfaces depend on: the app-level CSS
injections (display fonts + global starfield), the Plotly token template + off-token chart purge,
the DESIGN.md gold-highlight amendment, `market_display.json`, `team_assets.json`, the shared Lab
filter panel, the ▲/▼ + Match helpers, and the first real AppTest smoke harness.

**Architecture:** One sanctioned injection block in `dashboard/app.py` (DESIGN §3 names app.py as
the only injection site); tokens stay mirrored `config.toml` ↔ `theme.py` ↔ DESIGN.md; new config
JSONs live under `src/sportstradamus/data/config/` beside `stat_map.json`; display translation is
render-only (slugs stay the logic key).

**Tech stack:** Streamlit 1.58, plotly.io templates, streamlit AppTest.

**Branch:** `feature/dashboard-ux`. Prereq: Phase 0 merged (leg_label upgrade in A4 touches
`leg_schema.py`).

**Spec:** `docs/archive/superpowers/specs/2026-07-03-p8-oracle-assets-celestial-polish-design.md` §3.
Mockups are pixel truth (`docs/mockups/p8-*.html`). Gates + refactoring-specialist after every
task (see Phase 0 plan's Context section; same rules).

---

### Task A1 (FIRST): Starfield DOM spike + the app.py injection block

The spec assumes `.celestial-kicker`/`.celestial-headline` and an ambient layer already exist in
`app.py` — **they do not** (`app.py` is 101 lines, zero CSS). This task creates the entire
sanctioned injection block and validates the starfield against Streamlit's real DOM before
anything is built on top.

**Files:**
- Modify: `src/sportstradamus/dashboard/app.py` (insert after `st.set_page_config`, line 22)
- Modify: `src/sportstradamus/dashboard/theme.py` (starfield SVG + CSS constants)
- Modify: `src/sportstradamus/dashboard/data.py` (fold the off-pattern `render_banner`
  inline-style at ~343-348 onto a class defined in the injection block)
- Test: `tests/golden/test_app_injections.py`

- [ ] **Step 1: Failing golden** — pins the injection contract without a browser:

```python
"""The app-level CSS injection block: fonts, celestial classes, starfield."""

from pathlib import Path

APP = Path("src/sportstradamus/dashboard/app.py").read_text()
import sportstradamus.dashboard.theme as theme


def test_celestial_classes_defined():
    assert ".celestial-kicker" in theme.APP_CSS
    assert ".celestial-headline" in theme.APP_CSS
    assert "Cinzel" in theme.APP_CSS and "Cormorant Garamond" in theme.APP_CSS


def test_starfield_respects_ambient_rules():
    assert "prefers-reduced-motion" in theme.APP_CSS
    # every starfield opacity stays under the DESIGN §3 ambient ceiling
    import re
    for m in re.finditer(r"opacity:\s*\.?([\d.]+)", theme.STARFIELD_SVG):
        assert float(m.group(1)) <= 0.20 or float(m.group(1)) >= 1  # 1 = container reset


def test_app_injects_once():
    assert APP.count("st.html") + APP.count("unsafe_allow_html") == 1
```

- [ ] **Step 2: Run — FAIL** (`theme.APP_CSS` missing).

- [ ] **Step 3: Implement.** `theme.py` gains two module constants; `app.py` injects them once.
The primary structure (from the locked mockups, e.g. `p8-tonight.html:29-38` — note the mockups
use `z-index:-1`, which is also the Streamlit-safe choice):

```python
# theme.py
APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;600&family=Cormorant+Garamond:ital,wght@1,600&display=swap');
.celestial-kicker{font-family:'Cinzel',serif;font-weight:600;font-size:10.5px;
  letter-spacing:.26em;text-transform:uppercase;color:#C9A227}
.celestial-headline{font-family:'Cormorant Garamond',serif;font-style:italic;
  font-weight:600;color:#C9A227}
.starfield{position:fixed;inset:0;z-index:-1;pointer-events:none;
  background:
    radial-gradient(ellipse at 18% 4%, rgba(46,107,230,.10), transparent 46%),
    radial-gradient(ellipse at 86% 22%, rgba(201,162,39,.05), transparent 42%),
    radial-gradient(ellipse at 50% 96%, rgba(46,107,230,.06), transparent 55%)}
.starfield svg{position:absolute;inset:0}
@keyframes tw{0%,100%{opacity:.2}50%{opacity:.64}}
@media (prefers-reduced-motion: no-preference){.tw{animation:tw 4s ease-in-out infinite}}
@media (prefers-reduced-motion: reduce){.tw{animation:none;opacity:.15}}
</style>
"""

STARFIELD_SVG = """<div class="starfield">...generated <svg> with <use> star symbols,
each star opacity <= .20, twinkle via class="tw" (copy the symbol block from
docs/mockups/p8-tonight.html verbatim)...</div>"""
```

`app.py` after `set_page_config`:

```python
from sportstradamus.dashboard import theme

st.html(theme.APP_CSS + theme.STARFIELD_SVG)
```

- [ ] **Step 4: The spike (manual, half a session budget).** `poetry run dashboard`, then verify
in the browser: (a) stars visible in the empty side gutters on a wide window; (b) **zero** stars
visible through tables/cards — Streamlit's `.stApp` and block containers must paint opaque
`backgroundColor`/`secondaryBackgroundColor` over the fixed layer; (c) DevTools → toggle
`prefers-reduced-motion` → twinkle stops. If the fixed layer is invisible (a Streamlit stacking
context above it) or bleeds through content, switch to the **pre-committed fallback**: drop the
`<div>`, set the starfield as a static `background-image: url("data:image/svg+xml,…")` on
`.stApp` in `APP_CSS` (backgrounds render beneath content by definition; twinkle is sacrificed).
Record the verdict as a one-line comment above `STARFIELD_SVG`.

- [ ] **Step 5: Fold `data.py:render_banner`** inline `style=` HTML onto a class in `APP_CSS`
(keeps the "one injection" invariant honest).

- [ ] **Step 6: Gates + commit** `feat(p8-a): app injection block — celestial fonts + global starfield`

---

### Task A2: Plotly token template + off-token purge + hex-ban golden

**Files:**
- Modify: `src/sportstradamus/dashboard/theme.py` (semantic constants + template)
- Modify: `src/sportstradamus/dashboard/app.py` (set default template once)
- Modify: `src/sportstradamus/dashboard/surfaces/lab_diagnostics_charts.py` (hexes at ~22, 109,
  127, 136 + RdYlGn)
- Modify: `src/sportstradamus/dashboard/surfaces/lab_correlations.py` (~102, 211-213)
- Modify: `src/sportstradamus/dashboard/components/profit_sim.py` (~234, 249 — spec §3.6 missed
  this file; the ban golden below would stay red without it)
- Test: extend `tests/golden/test_design_tokens.py`

- [ ] **Step 1: Failing golden:**

```python
_BANNED = ("#2ecc71", "#e74c3c", "#f39c12", "RdYlGn")

def test_no_off_token_chart_colors():
    root = Path("src/sportstradamus/dashboard")
    offenders = [
        f"{p}:{hex_}" for p in root.rglob("*.py")
        for hex_ in _BANNED if hex_.lower() in p.read_text().lower()
    ]
    assert offenders == []

def test_semantic_colors_mirror_config():
    cfg = Path(".streamlit/config.toml").read_text()
    assert f'greenColor = "{theme.GREEN}"' in cfg
    assert f'redColor = "{theme.RED}"' in cfg
    assert f'orangeColor = "{theme.ORANGE}"' in cfg
```

- [ ] **Step 2: Implement** `theme.py`:

```python
GREEN = "#1F9D55"   # good/positive (config.toml greenColor)
RED = "#E5484D"     # bad/negative (config.toml redColor)
ORANGE = "#F5A524"  # warning (config.toml orangeColor)

SEQUENTIAL_COLORS = [...]  # copy the 10-step blue ramp verbatim from config.toml chartSequentialColors

def register_plotly_template() -> None:
    """Register + default the token template so ad-hoc figures inherit DESIGN."""
    import plotly.graph_objects as go
    import plotly.io as pio

    pio.templates["sportstradamus"] = go.layout.Template(
        layout={
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "font": {"family": "IBM Plex Sans", "color": "#E6E9EF", "size": 12},
            "colorway": [...],  # config.toml chartCategoricalColors verbatim
            "xaxis": {"gridcolor": "#2A2E37", "zerolinecolor": "#2A2E37"},
            "yaxis": {"gridcolor": "#2A2E37", "zerolinecolor": "#2A2E37"},
        }
    )
    pio.templates.default = "sportstradamus"
```

`app.py` calls `theme.register_plotly_template()` beside the injection. Then sweep the three
chart modules: greens → `theme.GREEN`, reds → `theme.RED`, oranges → `theme.ORANGE`, any
`RdYlGn`/diverging → `theme.DIVERGING_COLORS`, sequential heatmaps → `theme.SEQUENTIAL_COLORS`.
Charts must keep real axis titles/tick labels (spec §3.6: "style is not an excuse for a bad
chart") — while touching each figure, add missing `xaxis_title`/`yaxis_title`.

- [ ] **Step 3: Gates + commit** `feat(p8-a): plotly token template; off-token chart hexes purged`

---

### Task A3: DESIGN.md gold-highlight amendment

**Files:** `DESIGN.md` (§2 gold paragraph, ~lines 46-49), `.streamlit/config.toml` (comment only).

Rewrite the gold sentence to add the interactive-highlight role (spec §3.1, owner-resolved):
gold marks active/selected/hovered states (row hover, active lens/filter/tab/segment); primary
blue `#2E6BE6` is reserved for primary buttons and links. Keep, verbatim, the still-binding
clauses — never a data mark (constellation correlation edges the sole sanctioned exception,
DESIGN §4a), never body text, never a primary-button fill, never green/red substitute — and the
pinned needles `test_design_tokens.py` greps (run the golden before committing; it asserts
presence phrases like "Never for data, numerals").

- [ ] Edit both files in one commit: `docs(p8-a): gold = interactive-highlight role (DESIGN §2)`
- [ ] `poetry run pytest tests/golden/test_design_tokens.py -v` green.

---

### Task A4: `market_display.json` + `helpers.market_display_name`

**Files:**
- Create: `src/sportstradamus/data/config/market_display.json`
- Modify: `src/sportstradamus/helpers/__init__.py` (the helper; ~15 lines, cached JSON read)
- Modify: `src/sportstradamus/leg_schema.py` (`leg_label` routes market through it)
- Test: `tests/golden/test_market_display.py`

Shape `{ "LEAGUE": { "slug": "Display Name" } }`. Author every opaque code across the five
leagues; combos keep their shorthand (spec §3.3). Representative entries (complete the table for
every `stat_meta.json` market — 99 cells — plus any market present in `stat_map.json` values but
not yet in stat_meta):

```json
{
  "NBA": {
    "PTS": "Points", "REB": "Rebounds", "AST": "Assists", "FG3M": "3-Pt Made",
    "FG3A": "3-Pt Attempts", "FGM": "FG Made", "FGA": "FG Attempts",
    "OREB": "Off. Rebounds", "DREB": "Def. Rebounds", "STL": "Steals",
    "BLK": "Blocks", "TOV": "Turnovers", "FTM": "FT Made", "MIN": "Minutes",
    "PRA": "PRA", "PR": "PR", "RA": "RA", "PA": "PA", "BLST": "BLST"
  },
  "NFL": { "passing yards": "Passing Yards", "...": "..." },
  "WNBA": { "...": "..." }, "MLB": { "...": "..." }, "NHL": { "...": "..." }
}
```

Helper (fallback = the slug, so an unmapped market renders honestly):

```python
@cache
def _market_display() -> dict:
    with open(pkg_resources.files(data) / "config/market_display.json") as f:
        return json.load(f)

def market_display_name(league: str, slug: str) -> str:
    """Display label for a market slug; the slug stays the logic key everywhere."""
    return _market_display().get(league, {}).get(slug, slug)
```

Coverage golden: every `(league, market)` in `stat_meta.json` resolves to a value ≠ slug OR the
market is in the combo allowlist `{PRA, PR, RA, PA, BLST}` OR the slug already reads as English
(NFL's `"passing yards"`-style slugs title-case cleanly — assert the mapping still exists so
casing is explicit, don't exempt them).

- [ ] Test first → author JSON → helper → `leg_label` upgrade → gates → commit
  `feat(p8-a): market display names (render-only translation layer)`

---

### Task A5: `team_assets.json` + loader with safe fallback

**Files:**
- Create: `src/sportstradamus/data/config/team_assets.json` (handoff §5 pre-reserves this path)
- Create: `src/sportstradamus/scripts/build_team_assets.py` (authoring aid: prints league team
  codes found in `data/leagues/{lg}/teamlog.parquet` + gamelogs so authoring covers reality)
- Modify: `src/sportstradamus/dashboard/theme.py` (loader — the dashboard's asset-color seam)
- Test: `tests/golden/test_team_assets.py`

Shape `{ "LEAGUE": { "CODE": {"primary": "#RRGGBB", "secondary": "#RRGGBB"} } }` — colors only
(marks/logos are the §6.2 licensing lane). Author real franchise colors for: NBA 30, NFL 32,
NHL 32, MLB 30, WNBA all current teams (use the codes the pipeline uses — e.g. WNBA `POR` not
`PDX`, `NO` not `NOP` if that's what the logs carry; the build script prints the truth).

Loader contract (the PDX/POR incident is the cautionary tale — **never KeyError**):

```python
def team_colors(league: str, code: str) -> tuple[str, str]:
    """(primary, secondary) for a team; unknown codes get the neutral gray pair."""
    entry = _team_assets().get(league, {}).get(code)
    if not entry:
        return (GRAY, "#5A6070")
    return (entry["primary"], entry["secondary"])
```

Golden: per-league exact counts (30/32/32/30/WNBA-current), every code in each league's recent
teamlog window resolves to a non-fallback color, all values match `#RRGGBB`, fallback path pinned.
Team fills are **never gold** — assert `"#C9A227"` appears nowhere in the JSON.

- [ ] Test first → build script → author JSON → loader → gates → commit
  `feat(p8-a): team_assets.json — real team colors, safe fallback`

---

### Task A6: Shared Lab filter panel + stat_meta loader

**Files:**
- Create: `src/sportstradamus/dashboard/components/lab_filters.py`
- Modify: `src/sportstradamus/dashboard/data.py` (`load_stat_meta`)
- Test: `tests/golden/test_lab_filters.py`

`load_stat_meta` (cached like `load_stat_map`, `data.py:352`): normalize `stat_meta.json` to one
frame `league, market, dist, target_normalization, posthoc, blending, hpo_selection,
count_dispersion_objective, zinb_mode, shipped` — keys are sparse per cell, missing → `"none"`.
No pipeline change; it's a committed-config read (no archive).

`lab_filters.py` API (pure mask-builder + thin render; state in shared session keys so the three
Lab pages keep one filter state):

```python
FILTER_AXES = ("dist", "target_normalization", "posthoc", "blending",
               "hpo_selection", "count_dispersion_objective", "zinb_mode", "shipped")

def render_lab_filters(meta: pd.DataFrame, *, collapsed: bool) -> dict[str, list[str]]:
    """Render the shared panel (full or one-line chip strip + expander) and
    return {axis: selected values}. Widget keys are lab_filter_{axis} so the
    selection follows the user across Lab pages."""

def apply_lab_filters(df: pd.DataFrame, meta: pd.DataFrame, sel: dict) -> pd.DataFrame:
    """Join df to meta on (league, market) [df columns league/market or League/Market]
    and mask by every non-empty selection. Pure; unit-tested."""
```

Diagnostics renders `collapsed=False`; Correlations/Training `collapsed=True` (`.fbar` chip strip
per the two Lab mockups). Plus league/market/min-n scoping the surfaces already have.

- [ ] Unit tests on synthetic meta first (mask logic, sparse-key default, join casing) → build →
  gates → commit `feat(p8-a): shared Lab filter panel over stat_meta axes`

---

### Task A7: Direction/match display helpers + AppTest harness bootstrap

**Files:**
- Modify: `src/sportstradamus/dashboard/narrative.py` (pure helpers beside `home_away`)
- Create: `tests/golden/test_dashboard_render_smoke.py`
- Test: `tests/golden/test_dashboard_narrative.py` (helper pins)

Helpers (spec §3.2 / §3.4; the Board/Receipts/Details tasks consume them):

```python
_ARROW_UP = ('<svg class="ar" viewBox="0 0 9 10" width="9" height="10">'
             '<path d="M4.5 0 9 10 0 10Z" fill="#1F9D55"/></svg>')
_ARROW_DOWN = ('<svg class="ar" viewBox="0 0 9 10" width="9" height="10">'
               '<path d="M4.5 10 9 0 0 0Z" fill="#E5484D"/></svg>')

def bet_arrow(bet: str) -> str:
    """Colorblind-safe side cue: shape + color both carry Over/Under."""
    return _ARROW_UP if bet == "Over" else _ARROW_DOWN

def match_label(team: str, opp: str, home: bool) -> str:
    """Player-team-first matchup: 'LVA @ IND' away, 'LVA v IND' home (spec §3.4)."""
    return f"{team} {'v' if home else '@'} {opp}"
```

(For aggrid the arrow renders via a JsCode cell renderer in Phase B — `bet_arrow` is the one
source of the SVG strings; the grid imports them.)

AppTest smoke (the harness `test_deep_dive.py`'s docstring mentions does not exist — zero
`AppTest` imports in `tests/`). Known gotcha (repo memory): AppTest copies the script to /tmp, so
`__file__`-relative `st.Page` paths in `app.py` break — run through a **runpy wrapper** that
preserves the real path:

```python
from streamlit.testing.v1 import AppTest

_APP = Path("src/sportstradamus/dashboard/app.py").resolve()
_WRAPPER = f"import runpy; runpy.run_path(r'{_APP}', run_name='__main__')"

def test_app_boots_and_tonight_renders():
    at = AppTest.from_string(_WRAPPER, default_timeout=30)
    at.run()
    assert not at.exception
```

If `st.navigation` still fights AppTest after the wrapper, fall back to page-script-level tests
(run one surface file directly with seeded `st.session_state`) and record the limitation in the
test docstring — don't burn more than half a session on the harness itself.

- [ ] Pins for both helpers → implement → smoke → gates → commit
  `feat(p8-a): bet-arrow + match helpers; first AppTest smoke`

---

## Exit criteria (whole plan)

- Starfield verdict recorded (primary or fallback) after a live run; no bleed through tables.
- `test_design_tokens.py` extended suite green; zero banned hexes under `dashboard/`.
- `market_display.json` + `team_assets.json` committed with coverage goldens green.
- Three gates green; refactoring-specialist on every touched `.py`.
