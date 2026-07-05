# P8 Phase B — Sober Surfaces Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the four dense workbench surfaces to their locked mockups: Board (14→10
columns, arrows, gold hover), the three Lab pages (find-the-weak-spots framing, shared filter
panel, g6, stat-pair heatmap), and Receipts' sober parts (window filter, tickets, calibration
panel with the standard-vs-alt split).

**Architecture:** Pure aggregation additions live in `analysis.py` beside the existing Receipts
aggs; nightly precomputes anything the dashboard would otherwise recompute per load (profit-sim
precedent); the Lab reads a new tiny committed corr-summary artifact, never the 2.85M-row corr
parquets. Everything renders through Phase A primitives (template, filter panel, arrows,
display names).

**Prereqs:** Phase 0 + Phase A merged. Spec §4.3, §4.5–4.8. Mockups: `p8-board.html`,
`p8-lab-*.html`, `p8-receipts.html`. Gates + refactoring-specialist per task (Phase 0 plan,
Context section).

---

### Task B1: Board 14 → 10 columns

**Files:**
- Modify: `src/sportstradamus/dashboard/surfaces/board.py` (`MAIN_COLS` ~36-51, lens row ~60,
  sliders ~86-104)
- Modify: `src/sportstradamus/dashboard/columns.py` (Match derivation, display labels)
- Modify: `src/sportstradamus/dashboard/lenses.py` (Tonight lens)
- Modify: `src/sportstradamus/dashboard/components/grid.py` (arrow cell renderer, gold hover CSS)
- Test: `tests/golden/test_grid_options.py`, lens units, AppTest smoke

Target columns (mockup rev 4 is pixel truth — **Date drops**, Tonight lens covers recency;
`show_detail` reads the full `filtered` frame at `board.py:~170` so the detail dialog loses
nothing):

`League · Match · Player · Market · Line · Boost · Win % · Model Edge · Cons Edge · Platform`

- [ ] **Step 1: failing pins** — grid-options golden asserting: `Match` present +
  `Team`/`Opponent`/`Bet`/`Kelly`/`Date` absent from displayed columns; `Line` column carries the
  arrow cellRenderer; `Market` shows display names; hover CSS contains the gold rail.
- [ ] **Step 2: implement.**
  - `columns.py`: `df["Match"] = [match_label(t, o, h) for t, o, h in zip(df["Team"], df["Opponent"], df["Home"])]`
    (helper from Phase A7); `df["Market Display"] = [market_display_name(lg, m) for ...]` rendered
    under the header "Market" (slug column stays in the frame for logic).
  - `grid.py`: new `arrow_col` param on `build_themed_grid_options` → JsCode cellRenderer for
    `Line` that prefixes the ▲/▼ SVG (import the SVG strings from `narrative.bet_arrow`'s
    constants — one source) based on the row's `Bet` field; `Bet` stays in the row data, not
    displayed. Add `custom_css` hover block to `render_themed_grid`:

    ```python
    _HOVER_CSS = {
        ".ag-row-hover": {
            "box-shadow": "inset 3px 0 0 #C9A227 !important",
            "background-color": "rgba(201,162,39,.06) !important",
        }
    }
    ```

    (client-side only — hovering must not rerun Streamlit; spec §4.3.)
  - `lenses.py`: add `Tonight` lens (rows whose `Date` == the slate date — the same recency the
    dropped column carried); `board.py` adds an Over/Under `st.segmented_control` side filter.
    Active lens/segment styling is gold via the widgets' `key=`-scoped CSS already sanctioned for
    structural gaps — if Streamlit's segmented control can't take the gold active state without
    fighting the theme, leave the widget default and record it (theme primary still marks
    active; the mockup's gold chips are the target, not a hill to die on).
- [ ] **Step 3:** gates + AppTest smoke + commit `feat(p8-b): board condensed to 10 cols, arrows, gold hover`

---

### Task B2: Corr-summary artifact (the Lab heatmap's data source)

**Files:**
- Modify: `src/sportstradamus/training/correlate.py` (`_write_corr_outputs`, line ~854 — additive)
- Modify: `src/sportstradamus/dashboard/data.py` (`load_corr_market_summary`)
- Test: `tests/golden/test_corr_summary.py`

**Footprint note:** `training/` is import-don't-edit for this lane (handoff §5); the owner
approved this single additive hook by approving this plan. If review overturns it, the fallback
is a standalone `scripts/build_corr_summary.py` producing the identical artifact.

The committed corr matrices (`data/leagues/{lg}/corr_same_team.parquet` + `corr_opposing.parquet`,
MultiIndex `(team, market_a, market_b) → R`) are dashboard-forbidden (NBA = 2.85M rows; handoff
§7). Emit a compact market×market aggregate beside them in `_write_corr_outputs`:

```python
def _market_summary(blocks: pd.Series, scope: str) -> pd.DataFrame:
    """Aggregate (team, market_a, market_b) → R across teams to market-pair means.

    Position prefixes strip (B1.AST → AST) so the pair grid is market-level;
    n_teams/n_obs let the dashboard filter thin pairs honestly.
    """
    df = blocks.rename("R").reset_index()
    df.columns = ["team", "market_a", "market_b", "R"]
    for c in ("market_a", "market_b"):
        df[c] = df[c].str.split(".").str[-1]
    g = df.groupby(["market_a", "market_b"])["R"]
    out = g.mean().rename("rho_mean").reset_index()
    out["n_teams"] = g.size().values
    out["scope"] = scope
    return out
```

written as `data/leagues/{league}/corr_market_summary.parquet` (both scopes concatenated),
regenerated automatically whenever the corr parquets are. Regenerate + commit for NBA and NFL
(the only leagues with corr parquets on disk); the loader returns an empty frame for absent
leagues and the surface captions it honestly ("no correlation matrices generated yet for
{league} — run correlate").

- [ ] Golden first (synthetic blocks → hand-computed means, prefix stripping, empty-league
  loader) → hook + loader → regenerate NBA/NFL artifacts → gates → commit
  `feat(p8-b): market-pair corr summary artifact (Lab heatmap source)`

---

### Task B3: Lab Correlations surface

**Files:**
- Modify: `src/sportstradamus/dashboard/surfaces/lab_correlations.py`
- Test: surface unit pins + AppTest smoke

Per `p8-lab-correlations.html`: sober workbench + collapsed shared filter bar (`render_lab_filters
(collapsed=True)`) + starfield gutters (free from Phase A).

- [ ] **Lead visual:** stat-pair heatmap from `load_corr_market_summary(league)` — diverging
  red↔blue ramp (`theme.DIVERGING_COLORS`), never gold; market display names on axes; scope
  toggle (same-team / opposing); thin pairs (`n_teams < 3`) masked. Note in a code comment: the
  empirical-vs-model ρ overlay is the §6.4 follow-up, not built here.
- [ ] **Diagnostic callout** up top when correlation isn't paying: boosted-parlay hit rate <
  unboosted → `st.warning` naming the worst driver pair (compute from the existing value-add
  frame already built in this module).
- [ ] Keep + retheme the existing panels (value-add scatter, boost buckets, size table,
  calibration curve) — colors are already token from A2; ensure real ticks/legends everywhere.
- [ ] Path-ban pin: extend the surface test to assert the module never references
  `corr_same_team.parquet`/`corr_opposing.parquet` (string grep on the module source).
- [ ] Gates + commit `feat(p8-b): lab correlations — stat-pair heatmap + diagnostic framing`

---

### Task B4: Lab Diagnostics surface

**Files:**
- Modify: `src/sportstradamus/dashboard/surfaces/lab_diagnostics.py`
- Modify: `src/sportstradamus/dashboard/surfaces/lab_diagnostics_charts.py`
- Test: chart-spec pins

Per `p8-lab-diagnostics.html` (find-the-weak-spots — the opposite intent from Receipts):

- [ ] **Worst-BSS-first** default sort on the market table; weak cells get the amber rail
  (aggrid rowStyle `box-shadow: inset 3px 0 0 {theme.ORANGE}` when `brier_skill_score < 0`).
- [ ] **"Start here" strip**: top-3 worst cells as `st.metric`-style tiles above the table.
- [ ] **Full shared filter panel** (`collapsed=False`) + `Family`/`Norm` as table columns via the
  `load_stat_meta` join (A6).
- [ ] Charts: real mean-CRPS numbers on the CRPS axis (the frame already carries them), axis
  titles + legends on every figure (A2 left them tokened; this task makes each one a *good*
  chart per spec §3.6).
- [ ] Gates + commit `feat(p8-b): lab diagnostics — weak-spots framing, meta columns`

---

### Task B5: Lab Training surface — g6 + gate matrix

**Files:**
- Modify: `src/sportstradamus/dashboard/surfaces/lab_training.py` (TAB_COLUMNS ~141-155, sidebar
  filters ~258-281)
- Create: `src/sportstradamus/dashboard/components/gate_matrix.py` (`lab_training.py` is at ~299
  lines — the matrix goes in its own module, not over the 300 cap)
- Test: `tests/golden/test_gate_matrix.py`

- [ ] **TAB_COLUMNS "Ship gates"** extends to g6 (parquet already carries them):
  `g6_star_ci_hi`, `g6_star_ref`, `g6_recent_corr`, `g6_pass` inserted before the pass flags;
  DIRECTIONS entries added (g6 ↓-flavored like g2/g3). Gates recap in header help: G1
  paired-Brier CI · G2/G3 tail bias-z · G4 IQR ratio · G5 debiased ECE · G6 anti-shrinkage;
  Ship = AND of all six.
- [ ] **Sidebar filters move** into the shared panel (`render_lab_filters(collapsed=True)`) —
  delete the `st.sidebar` block at ~258-281; keep the `nav_cell` deep-link consumption.
- [ ] **Run-at-a-glance strip**: tiles Cells trained / Shipping (all gates) / One gate short /
  Withheld + a lifecycle funnel (trained → G1 pass → all-gates → graduated live) from
  `lifecycle_table()` — plain `st.metric` row + one horizontal funnel bar chart (template-themed).
- [ ] **Gate matrix** (`gate_matrix.py`): pure builder + thin render.

```python
GATE_COLS = ("g1_pass", "g2_pass", "g3_pass", "g4_pass", "g5_pass", "g6_pass")

def gate_matrix_frame(stats: pd.DataFrame) -> pd.DataFrame:
    """Cells × gates, worst-first: one-gate-short cells pinned to the top
    (amber rail), then by fails desc (multi-fail = red rail), then BSS asc.
    Adds n_fails + rail ∈ {amber, red, none}. Pure."""
```

    Render: ● pass (green) / ○ fail (red) — text glyphs colored via Streamlit semantic colors,
    **no gold anywhere on data**; rails via the same aggrid rowStyle mechanism as B4.
- [ ] Matrix ordering golden (one-gate-short pinned above two-fail; ship row order stable) →
  implement → gates → commit `feat(p8-b): lab training — g6, gate matrix, glance strip`

---

### Task B6: Calibration precompute + Receipts calibration panel

**Files:**
- Modify: `src/sportstradamus/analysis.py` (pure agg beside `record_grid`)
- Modify: `src/sportstradamus/nightly.py` (`_precompute_calibration`, mirroring
  `_precompute_profit_sim`)
- Modify: `src/sportstradamus/helpers/io.py` (`CALIBRATION_SUMMARY_PATH` beside
  `PROFIT_SIM_SUMMARY_PATH`)
- Modify: `src/sportstradamus/dashboard/surfaces/receipts.py` (panel render)
- Modify: `src/sportstradamus/dashboard/data.py` (loader)
- Test: `tests/golden/test_calibration_agg.py`

The owner's why (spec §4.5): "alt lines and ladders will be popular, so prove profitability off
more than just picking over/under on the basic line." Phase 0 gave history a per-offer `Alt Line`
bool; this task aggregates it.

```python
# analysis.py
_CAL_BINS = np.arange(0.40, 1.01, 0.05)

def calibration_summary(exploded: pd.DataFrame) -> pd.DataFrame:
    """Reliability frame: one row per (prob bin × alt split) with predicted mean,
    realized hit rate, n; plus per-split ECE and flat-juice ROI. Resolved rows only.
    Pure — nightly persists it, Receipts renders it."""
```

- [ ] Agg golden first: hand-built fixture history (both splits, bin edges incl. the 0.55
  boundary row, empty-split), ECE hand-computed, ROI at `JUICE_PAYOUT`.
- [ ] `nightly._precompute_calibration` writes the parquet after resolve;
  `receipts.py` renders: reliability diagram (diagonal reference; **blue = standard bins, gold =
  alt/ladder bins** — the one sanctioned gold-on-chart use here is the *series identity* the spec
  and mockup lock, gold reaching into the tails), the ECE figure, and the ROI split (standard vs
  alt) as two metrics.
- [ ] Gates + commit `feat(p8-b): calibration precompute + receipts reliability panel`

---

### Task B7: Receipts sober pass

**Files:**
- Modify: `src/sportstradamus/dashboard/surfaces/receipts.py`
- Create: `src/sportstradamus/dashboard/components/tickets.py` (`receipts.py` is ~287 lines)
- Test: ticket unit pins; hero-parity pin stays green

Per `p8-receipts.html` (rev 2 + calibration), everything except the nebula hero (that's C7):

- [ ] **Window segmented filter** (Last week / month / 3mo / year / All) re-scoping the
  by-dimension grid + hero (reuse `TIMEFRAMES` from `analysis.py`).
- [ ] **"Your slips" as receipt tickets** (`tickets.py`): status rail + stamp (won green / lost
  red / pending gold — gold marks *pending*, a state, not a value), thesis headline, legs with
  ✓/✗ from the graded `result` list, EV / stake / return, `leg_label` for leg text. Pure
  builder + thin render; the slip-constellation watermark on wins is a nice-to-have — skip
  unless trivial with the existing constellation figure at low opacity.
- [ ] **Strategy simulator** expander re-labeled "Strategy simulator — retrospective
  (hindsight)" + a scar caption reserving the forward Simulated-Bettor Ledger slot
  (`docs/handoffs/sim-bettor-ledger.md` D6): "Forward paper-trading ledger lands with the
  sim-bettor-ledger lane."
- [ ] **CLV tiles**: coverage stays a rate; the numbers are now real post-Phase-0.9 — update the
  caption to say "closing probability" and drop any hedging.
- [ ] Gates + commit `feat(p8-b): receipts window filter, tickets, honest sim label`

---

## Exit criteria (whole plan)

- Board + three Lab surfaces + Receipts match their mockups structurally (side-by-side eyeball
  in a live run; pixel-perfection is C-phase polish territory only where mockups demand it).
- `calibration_summary` parquet written by a fake-mode reflect (integration suite).
- New goldens green: grid options, corr summary, gate matrix, calibration agg, path-ban.
- Three gates green; refactoring-specialist on every touched `.py`.
