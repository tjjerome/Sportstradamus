# DESIGN.md — Sportstradamus dashboard design system

The rules that keep the Streamlit dashboard looking *designed*, not AI-generated. Read this before
any dashboard/UI work. The FIXED tokens below are inviolable; `.streamlit/config.toml` is their
machine-enforced mirror and the single source of truth — edit the two together. **Do not supplement
these constraints with your own defaults.** When a case isn't covered, pick from the defined scale or
ask — do not invent new values.

Enforcement: the `design-lint` hook nudges live on banned patterns; `tests/golden/test_design_tokens.py`
is the hard gate. Stage 2/3 (table craft, normalization, chart re-theme, nav migration) is parked in
[docs/dashboard_design_stage2_3.md](docs/dashboard_design_stage2_3.md). The vendored
[frontend-design skill](.claude/skills/frontend-design/SKILL.md) carries the same bans at the agent
level.

## 1. Brand & tone

A multi-sport fantasy/props tool used daily, often on mobile, under time pressure (lineups lock).
The feel: **credible sports-broadcast** — fast, data-dense but scannable, trustworthy with numbers.
The differentiation is *table craft and transparency*, not decoration: normalized stats, documented
definitions, consistent layout. Dark by default (late-night lineup setting, reduced eye strain — the
Sleeper precedent).

## 2. Design tokens (FIXED — mirror of `.streamlit/config.toml`)

**Color**

| Role | Token | Hex |
|---|---|---|
| Primary (buttons, links, active) | `primaryColor` | `#2E6BE6` electric blue |
| Background (main) | `backgroundColor` | `#0E1117` |
| Surface (cards, widgets, code) | `secondaryBackgroundColor` | `#1A1D24` |
| Text | `textColor` | `#E6E9EF` |
| Border | `borderColor` | `#2A2E37` |
| Link | `linkColor` | `#5B8DEF` (no underline) |
| Good / positive | `greenColor` | `#1F9D55` |
| Bad / negative | `redColor` | `#E5484D` |
| Warning | `orangeColor` | `#F5A524` |
| Neutral | `grayColor` | `#8A91A0` |

Semantic colors are used **only by intent** in tables/badges/sparklines. Red is reserved for
negative/bad values — never a brand accent.

**Type** — `IBM Plex Sans` (body + headings), `IBM Plex Mono` (code + tabular numerals so stat
columns align by place value). Base 14px. Hierarchy comes from size + weight + color, never size
alone.

**Spacing** — Streamlit's native scale; when adding custom gaps, stay on a 4 / 8 / 12 / 16 / 24 step.

**Radius** — `small` (4px) base and button. Nothing rounder.

**Charts** — `chartCategoricalColors` (brand-led, colorblind-safe), `chartSequentialColors` (10-step
blue ramp, heatmaps), `chartDivergingColors` (red ↔ neutral ↔ blue, above/below average). Plotly /
Altair / Vega-Lite inherit these automatically.

## 3. Component guidelines

- **Theming**: always via `config.toml`. Reserve `st.html` / `unsafe_allow_html` CSS for targeted
  *structural* gaps only, scoped through a widget's `key=`-generated `.st-key-…` class — never for
  colors/fonts/backgrounds that a token already covers.
- **Tables**: `streamlit-aggrid` (already a dependency), themed to these tokens, for the main stat
  grids; native `st.dataframe` + `column_config` / `ProgressColumn` for simpler ones. See §4.
- **Metrics**: establish hierarchy — one hero number + smaller supporting metrics with context
  (deltas, sparklines, vs-projection). Not a uniform row of bare `st.metric` cards.
- **Charts**: Plotly/Altair themed to `chart*Colors`; explicit axis titles; drop redundant
  legends/colorbars; no pie >4 slices, no 3D.
- **Icons**: Google Material Symbols via `:material/icon_name:` (see §5).

## 4. Table & data rules

- **Right-align numbers, left-align text. Never center numeric columns** (centering causes visual
  wobble and slows digit comparison).
- Sticky/frozen headers on tall tables; on small screens **reduce columns** rather than freezing
  everything.
- 1px light separators or zebra striping — not heavy gridlines. Consistent number formatting
  (thousands separators, fixed decimals, % where due), tabular numerals.
- Conditional formatting / heatmaps (use `chartSequentialColors`) and inline bars to surface
  outliers; sortable + filterable columns.
- Group row actions under a single kebab/dropdown — not a billboard of per-row icons.
- Progressive disclosure: expandable rows, pagination, column hiding. Lead with the primary KPI
  (top-left, F-pattern), push detail into drill-downs.

## 5. Iconography

Material Symbols only, via the `:material/icon_name:` shortcode (works in Markdown, labels, and many
widget `icon=` params). **Emoji are banned as functional iconography** — section headers, nav, button
glyphs, status badges. For status that needs color, use Streamlit semantic colored text
(`:red[Strong]` / `:orange[Moderate]` / `:gray[Mild]`) — color **and** text, never color alone. Real
team/sport marks: inline SVG recolored via `currentColor`.

## 6. NEVER (these are the "AI slop" tells)

- **NEVER** use Inter, Roboto, Arial, Open Sans, Lato, or bare system fonts. (IBM Plex only.)
- **NEVER** use purple/violet gradients (on light or dark).
- **NEVER** use the default Streamlit red `#FF4B4B` as primary.
- **NEVER** use emoji as icons or section headers.
- **NEVER** use `st.balloons()` / `st.snow()`.
- **NEVER** render a uniform row of default `st.metric` cards as the hero.
- **NEVER** center-align numeric columns.
- **NEVER** invent off-scale spacing, radii, or colors — use the tokens.
- **NEVER** theme via inline CSS when `config.toml` can do it.

## 7. FIXED vs FLEXIBLE

**FIXED** (never alter): the color palette, fonts, spacing scale, radius, chart palettes, and every
NEVER rule in §6.

**FLEXIBLE** (experiment freely): layout composition, which chart type fits the data, micro-
interactions, information density *within* the spacing scale, and whether a view leans on cards vs
tables. Experiment in the FLEXIBLE zone; never touch a FIXED token. If a case isn't covered, pick
from the defined scale or ask — do not invent.

## 8. Accessibility

- Colorblind-safe palette (≈1 in 12 men have a color-vision deficiency). Never rely on color alone —
  pair with text, icon, or position.
- Contrast: WCAG AA (4.5:1 for text). The dark tokens above clear it; keep it that way for any new
  surface.
- Keep to the 2–3 semantic colors + neutral gray scale. Preserve visible focus states.

## 9. References

- [docs/dashboard_design_stage2_3.md](docs/dashboard_design_stage2_3.md) — parked table/chart/nav work.
- [.claude/skills/frontend-design/SKILL.md](.claude/skills/frontend-design/SKILL.md) — Anthropic's
  frontend-design skill (the named bans, at the agent level).
- Streamlit theming reference (bundled): `streamlit/.agents/skills/developing-with-streamlit/references/theme.md`.
- Refactoring UI (design in grayscale first, hierarchy via size+weight+color); USWDS design tokens.
