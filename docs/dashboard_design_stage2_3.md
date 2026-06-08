# Dashboard redesign — Stage 2/3 work (parked)

Stage 1 (committed visual identity + governance) shipped: `.streamlit/config.toml` tokens,
[`DESIGN.md`](../DESIGN.md), the design-lint hook, the `test_design_tokens.py` gate, the vendored
frontend-design skill, and the emoji→Material/native swap. This file parks the data-presentation and
polish work for later. The token vocabulary it refers to (colors, fonts, `chart*Colors`) lives in
`DESIGN.md`; don't restate it here.

The credibility pattern every respected stats site shares (FBref, Understat, FanGraphs): **normalization
+ transparency + documented definitions + consistent layout.** Decoration is not what makes a
data-heavy app look professional; table craft is.

## Stage 2 — upgrade data presentation

1. **Replace the main stat tables with themed AG Grid.** `streamlit-aggrid` is already a dependency.
   Theme it to the design tokens (Quartz/Balham, with `backgroundColor` / `headerBackgroundColor` /
   `accentColor` / font / row height set from `DESIGN.md`). Required behaviors: sticky/frozen headers,
   sortable + filterable columns, **right-aligned tabular numerals**, 1px light separators (not heavy
   gridlines), and **conditional heatmap coloring on key stat cells** (the FBref/Understat "readable at
   a glance" move). Use `chartSequentialColors` for the heatmap ramp. Caveat: AG Grid's theming API
   changed at v32/v33 (Quartz is now default) and enterprise features (advanced pivoting) need a
   license — verify before using them. Keep native `st.dataframe` + `column_config` / `ProgressColumn`
   for the simpler tables.

2. **Add normalization + glossary.** Where it aids comparison, surface **per-90 / percentile / rank**
   alongside raw stats (the FanGraphs/FBref credibility move) and put **glossary definitions on stat
   header hover/tooltip**. Keep the layout consistent across leagues/positions so users learn it once.

3. **Replace default metric rows with hierarchy.** Don't drop a uniform row of default `st.metric`
   cards as the hero. Establish one **hero number** (real size, top-left per the F-pattern) + smaller
   supporting metrics; add context (deltas, sparklines, vs-projection). Use `streamlit-extras` or
   `streamlit-shadcn-ui` styled cards (or a tokenized custom card). **Reserve red strictly for
   negative/bad values.**

## Stage 3 — polish and verify

4. **Re-theme the charts.** Plotly (custom `go.layout.Template`), Altair (registered colorblind-safe
   theme), and any ECharts all read from the design palette via `chartCategoricalColors` /
   `chartSequentialColors` / `chartDivergingColors`. Always set explicit axis titles, drop redundant
   legends/colorbars, avoid pie charts with >4 slices and 3D charts entirely (prefer length + 2D
   position — bars, lines, scatter — which are preattentive).

5. **Navigation: sport as the top-level switch, common actions on the surface** (the Sleeper lesson —
   don't bury actions in nested menus). This is also where the **deferred Material-icon nav migration**
   lands: move `pages/*.py` → a non-magic dir, define nav with `st.navigation` + `st.Page(...,
   icon=":material/...:")`, and **remove the per-page `st.set_page_config` calls** (only the entry may
   call it once — secondary calls error under `st.navigation`). This is the structural change Stage 1
   deferred because it can't be runtime-verified without launching the live app.

6. **Run a screenshot → compare-against-`DESIGN.md` → refine loop** each time you ask an agent to
   experiment in the FLEXIBLE zone.

## Failure modes to watch (from the brief)

- **"Generic"** ⇒ tokens aren't distinctive enough — push the typeface/palette further from defaults,
  add a signature element (custom data-viz style, team-color heatmaps).
- **"Cluttered" / bad on mobile** (the ESPN/NFL failure mode) ⇒ cut columns on small screens, add
  progressive disclosure (expandable rows, pagination, column hiding), reduce to ~5–7 KPIs per view.
  Information overload is the most common dashboard failure.
- **Agent reverts to generic choices** ⇒ `DESIGN.md` is being ignored or is too vague — make FIXED
  tokens more explicit, add more NEVER rules, re-paste the system at session start.

## Component libraries (verify versions before pinning)

`streamlit-aggrid` (have it) · `streamlit-extras` (metric cards, actively maintained) ·
`streamlit-shadcn-ui` (modern cards/tabs; weaker dark mode) · `streamlit-echarts` (richer charts, but
maintainer-flagged "best-effort") · `streamlit-option-menu` (clean nav). Material Symbols ship natively
via the `:material/icon_name:` shortcode — no library needed.
