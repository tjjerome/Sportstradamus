# DESIGN.md — Sportstradamus dashboard design system

The rules that keep the Streamlit dashboard looking *designed*, not AI-generated. Read this before
any dashboard/UI work. The FIXED tokens below are inviolable; `.streamlit/config.toml` is their
machine-enforced mirror and the single source of truth — edit the two together. **Do not supplement
these constraints with your own defaults.** When a case isn't covered, pick from the defined scale or
ask — do not invent new values.

Enforcement: the `design-lint` hook nudges live on banned patterns; `tests/golden/test_design_tokens.py`
is the hard gate. The UX redesign (six surfaces, slip rail, celestial layer) is specified in
[docs/dashboard_ux_redesign.md](docs/dashboard_ux_redesign.md); it absorbed the old Stage 2/3 parking
doc (now in `docs/archive/`). The vendored
[frontend-design skill](.claude/skills/frontend-design/SKILL.md) carries the same bans at the agent
level.

## 1. Brand & tone

A multi-sport fantasy/props tool used daily, often on mobile, under time pressure (lineups lock).
The feel: **credible sports-broadcast with a celestial-oracle layer** — fast, data-dense but
scannable, trustworthy with numbers; the name promises Nostradamus, so the chrome leans
mystic-meets-sports (prophecy voice, gold accents, constellation motifs) while every number stays
broadcast-sober. The differentiation is *table craft and transparency*, not decoration: normalized
stats, documented definitions, consistent layout. Dark by default (late-night lineup setting,
reduced eye strain — the Sleeper precedent; the night sky is also the oracle's canvas). The
prophecy-voice naming map (stories→prophecies, etc.) lives in
[docs/dashboard_ux_redesign.md](docs/dashboard_ux_redesign.md) — one home, don't restate it here.

## 2. Design tokens (FIXED — mirrored in `.streamlit/config.toml` + `dashboard/theme.py`)

**Color**

| Role | Token | Hex |
|---|---|---|
| Primary (buttons, links) | `primaryColor` | `#2E6BE6` electric blue |
| Background (main) | `backgroundColor` | `#0E1117` |
| Surface (cards, widgets, code) | `secondaryBackgroundColor` | `#1A1D24` |
| Text | `textColor` | `#E6E9EF` |
| Border | `borderColor` | `#2A2E37` |
| Link | `linkColor` | `#5B8DEF` (no underline) |
| Good / positive | `greenColor` | `#1F9D55` |
| Bad / negative | `redColor` | `#E5484D` |
| Warning | `orangeColor` | `#F5A524` |
| Neutral | `grayColor` | `#8A91A0` |
| Prophecy / celestial accent | `goldColor` | `#C9A227` |

Semantic colors are used **only by intent** in tables/badges/sparklines. Red is reserved for
negative/bad values — never a brand accent. Gold is the *oracle's* color and marks
**active/selected/hovered** states: table row hover, active lens, active filter chip, active tab
underline, active segmented-control segment. Gold also highlights kickers, prophecy headlines,
constellation highlights, and correlation strength. Primary blue `#2E6BE6` is reserved for
primary buttons and links only — it is no longer used as a generic "active/selected" accent.
Gold is **never a data mark** — never a bar, heatmap cell, plotted line/point encoding a value,
or team-color star — never body text, never primary button fill, never a substitute for
green/red semantics. Two sanctioned exceptions: the gold correlation edge in the constellation
(DESIGN §4a); and the Receipts reliability diagram's alt-line/ladder series marker, where gold
identifies *which population* a point belongs to (standard line vs. alt/ladder), not the point's
own value — the same role as a legend swatch. Both mark identity/strength, never a plotted
value on their own axis; don't generalize either into license to use gold on other charts.

Streamlit's `[theme]` block has no free-form keys, so `config.toml` mirrors only the tokens
Streamlit understands; `goldColor`, the display fonts, and the ambient-image rules are mirrored in
`sportstradamus/dashboard/theme.py` instead. The two mirrors together are the machine-readable
token set — edit DESIGN.md and the relevant mirror in the same commit.

**Type** — `IBM Plex Sans` (body + headings), `IBM Plex Mono` (code + tabular numerals so stat
columns align by place value). Base 14px. Hierarchy comes from size + weight + color, never size
alone. **Display-only celestial faces**: `Cinzel` (kickers, small-caps labels) and
`Cormorant Garamond` (prophecy/story headlines) — applied solely through the `.celestial-kicker` /
`.celestial-headline` classes injected once in `dashboard/app.py`. Never for data, numerals,
tables, body copy, or anything a user scans for a number.

**Spacing** — Streamlit's native scale; when adding custom gaps, stay on a 4 / 8 / 12 / 16 / 24 step.

**Radius** — `small` (4px) base and button. Nothing rounder.

**Charts** — `chartCategoricalColors` (brand-led, colorblind-safe), `chartSequentialColors` (10-step
blue ramp, heatmaps), `chartDivergingColors` (red ↔ neutral ↔ blue, above/below average). Plotly /
Altair / Vega-Lite inherit these automatically.

## 3. Component guidelines

- **Theming**: always via `config.toml`. Reserve `st.html` / `unsafe_allow_html` CSS for targeted
  *structural* gaps only, scoped through a widget's `key=`-generated `.st-key-…` class — never for
  colors/fonts/backgrounds that a token already covers. Named exceptions, all injected once in
  `dashboard/app.py`'s `APP_CSS`: the display-font CSS (`.celestial-kicker` / `.celestial-headline`,
  §2), the ambient-image layer (below), and the Games surface lens toggles (`.st-key-lens_deep_on` /
  `.st-key-lens_wider_on`, §4a) — `config.toml`'s `[theme]` has no gold slot, so a widget that must
  read celestial-gold when active reaches the token the same narrow way the other two do.
- **Nebula wash**: a faint radial gradient inside the surface palette (blue-family stops drawn from
  `chartSequentialColors`, gold highlights ≤ 12% opacity) is permitted on **hero/prophecy cards
  only**. The purple/violet gradient ban (§6) stands untouched.
- **Ambient imagery**: semi-transparent background art blending mystic + sports (night-sky fields,
  hourglass/sand motifs, celestial sports equipment). Slots and files are declared in
  `data/assets/ambient/ambient_manifest.json` (slot → file, opacity, placement, license/attribution);
  slots without art render token-palette gradients. Rules: opacity ≤ 20% over `backgroundColor`,
  body text on top must keep WCAG AA contrast, **never behind dense tables or stat grids**, and art
  is stock or commissioned only — no AI-generated images, license recorded in the manifest.
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

## 4a. Signature element — the constellation

Correlation rendered as a star map, and the slip editor's primary control. Each leg is a star
**filled with its team's color** (the two sides of the matchup read at a glance) and **sized by
its model edge** (the strongest legs read biggest). The map is **static per game** — its stars are
the game's model-liked legs, fixed in place; selecting a leg never moves a star, it only lights it
up. A leg in the slip burns at **full color and opacity**; a candidate is the same color
**desaturated toward gray and dimmed** — selection is alpha + saturation, never an outline (a gold
ring read as a team color). Pairwise correlation |ρ| is the edge weight (gold, opacity/width ∝ |ρ|,
dashed when ρ < 0 — "fights the thesis"); an edge stays hidden until one of its stars is in the
slip and faintly previews on hover, so the clutter scales with the slip, not the game.
Layout is **team-anchored force-directed**: each team's most-connected leg is pinned to its side,
the correlation edges place the rest, so a cross-matchup leg floats toward the centre and an
unrepresented side leaves its half empty (the both-teams parlay rule, made visual). Legs from
*other* games never become stars here — the map stays one game; a single-sided game reaches its
second team through a separate satellite section beside the map, never by crowding the constellation.
In the editor the map is interactive — click a star to add or remove its leg, and hover a star for a card (its
read plus a **Full detail** link into the offer dialog, slip preserved) — with the modebar and
zoom/pan off (it's a map, not a chart). This is the one piece of decoration that *is* data: use it on the slip
editor, Game pages, and parlay detail, keep it on `backgroundColor`, never let it crowd a table. It
is the brand's signature; treat its grammar — star = leg, **fill = team**, **size = edge**,
**brightness = in the slip**, edge = correlation — as FIXED. Team fills come from
`team_assets.json` via `theme.team_colors(league, code)`; an unmapped code gets the neutral
gray fallback. Team fills are never gold — gold is the correlation-edge color.

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
- **NEVER** set numeric/data content in the display faces (Cinzel/Cormorant) — numerals are always
  Plex.
- **NEVER** ship an AI-generated image; ambient art is stock/commissioned with a manifest license
  line.

## 7. FIXED vs FLEXIBLE

**FIXED** (never alter): the color palette (gold included), the fonts (Plex for everything that
isn't a `.celestial-*` class; Cinzel/Cormorant only through those classes), spacing scale, radius,
chart palettes, the constellation grammar (§4a), the ambient-imagery rules (§3), and every NEVER
rule in §6.

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

- [docs/dashboard_ux_redesign.md](docs/dashboard_ux_redesign.md) — the UX redesign spec (six
  surfaces, slip rail, deep-dive v2, naming map, ambient-imagery slot map + artist brief).
- [.claude/skills/frontend-design/SKILL.md](.claude/skills/frontend-design/SKILL.md) — Anthropic's
  frontend-design skill (the named bans, at the agent level).
- Streamlit theming reference (bundled): `streamlit/.agents/skills/developing-with-streamlit/references/theme.md`.
- Refactoring UI (design in grayscale first, hierarchy via size+weight+color); USWDS design tokens.
