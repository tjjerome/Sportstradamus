# P8 — The Oracle: assets & celestial polish (design spec)

**Stage:** P8 of the dashboard UX redesign ("the Oracle")
**Branch:** `feature/dashboard-ux`
**Status:** Design locked. Ready for implementation planning.

## How to read this spec (handoff chain)

This spec is written to survive a context reset. It is the sole durable artifact
of the P8 brainstorming session: the plan author (fable) and the implementer
(sonnet) will each start from a **clean context** and see only what is on disk.
Everything needed to plan and build P8 is either in this file or in the files it
points at.

- **The mockups are the pixel-level source of truth.** Eight locked HTML mockups
  live in `docs/mockups/p8-*.html`. They are self-contained (CDN fonts, inline
  SVG, no build step) and can be opened directly in a browser. This spec explains
  *intent, data wiring, and rules*; the mockups show *exact layout, spacing, and
  color*. When this spec and a mockup disagree on a pixel, the mockup wins; when
  they disagree on a rule or a data source, this spec wins.
- **Prerequisite reading** for anyone implementing: [DESIGN.md](../../../DESIGN.md)
  (the committed visual identity and the FIXED token set), and
  [docs/dashboard_ux_redesign.md](../../../docs/dashboard_ux_redesign.md) (the six-surface
  redesign spec P8 polishes).
- **The predecessor lane brief** is [docs/handoffs/dashboard-ux.md](../../../docs/handoffs/dashboard-ux.md).
- P8 is **large** (8 surfaces + shared infrastructure + follow-up lanes). §10
  proposes a phasing so the plan can be decomposed into buildable units. Fable
  should feel free to split the implementation plan into parts (shared infra
  first, then surfaces) rather than one monolith.

## Mockup index

| Surface | Mockup file | Celestial level |
|---|---|---|
| Details dialog | `p8-details.html` (header choice), `p8-details-tabs.html` (locked, rev 3) | sober header + gold accents, **no** nebula |
| Tonight | `p8-tonight.html` | full nebula prophecy cards |
| Board | `p8-board.html` | sober workbench, gold hover only |
| Games | `p8-games.html`, `p8-constellation-lab.html` (astrolabe), `p8-games-lenses.html`, `p8-glyphs.html` | celestial centerpiece |
| Receipts | `p8-receipts.html` | sober + **one** nebula hero |
| Diagnostics (Lab) | `p8-lab-diagnostics.html` | sober workbench + starfield gutters, no nebula |
| Correlations (Lab) | `p8-lab-correlations.html` | sober workbench + starfield gutters, no nebula |
| Training (Lab) | `p8-lab-training.html` | sober workbench + starfield gutters, no nebula |

`p8-celestial-refresh.html` is an early global-skin exploration and is **not**
part of the locked set; ignore it.

---

## 1. Goal & governing principle

P8 is the *polish* pass on the Oracle redesign: it takes the already-restructured
surfaces and gives them a coherent, intentional celestial identity ("mystic meets
sports") without ever compromising data legibility. The two jobs are:

1. **Assets** — introduce the real art/theming primitives the earlier stages left
   as placeholders (team colors, glyphs, starfield, astrolabe, constellation
   shapes), and flag the ones that need licensed assets.
2. **Celestial polish** — apply a *disciplined* amount of theme per surface, and
   fix the token/style debt that crept in (off-palette chart colors, missing
   market-name translation, O/U text instead of directional cues).

### The governing principle: a celestial-vs-sober hierarchy

Theme intensity is **not uniform**. Each surface sits at a fixed point on a
spectrum, and this ordering is the single most important rule in P8:

- **Full celestial** — Tonight (nebula prophecy cards). The oracle speaks loudest
  on the landing surface.
- **Celestial centerpiece** — Games (the constellation star map + the astrolabe
  are the whole point of the page).
- **Sober + one hero** — Receipts (a sober performance workbench with exactly one
  nebula "verdict" card at the top).
- **Sober workbench** — Board and the three Lab surfaces (dense data; theme lives
  only in the kicker, section rules, gold highlight, and the starfield showing
  through the empty side gutters). **No nebula on these.**
- **Sober dialog** — the Details popup (themed header with a gold hairline, but no
  nebula wash).

If in doubt, a surface gets *less* theme, not more. Numbers always win.

---

## 2. Non-negotiable guardrails (do not break these)

These are pre-existing invariants. P8 amends the *usage* rules for gold (§3.1) but
must not violate any of the following:

- **DESIGN.md FIXED tokens are inviolable.** The palette (including `goldColor
  #C9A227`), the fonts (Plex for all data; Cinzel/Cormorant display-only, never on
  numerals), spacing scale, and radius are fixed and mirrored in
  `.streamlit/config.toml` + `dashboard/theme.py`. Edit DESIGN.md and the mirror in
  the same commit. `tests/golden/test_design_tokens.py` is the hard gate;
  `design-lint` nudges live.
- **The dashboard never touches the DuckDB archive.** It reads pre-computed parquet
  snapshots only. Any archive-derived data a surface needs must be exported to
  parquet by a cron job first. New dashboard modules must not import anything whose
  top-level code constructs `Archive()`; use `LazyArchive` from
  `sportstradamus.helpers` if a shared `archive` binding is needed. Pinned by
  `tests/golden/test_dashboard_no_archive_lock.py`.
- **Theming goes through `config.toml`/`theme.py`, not hand-rolled CSS.** The only
  sanctioned `unsafe_allow_html` injections are the ones already established in
  `dashboard/app.py`: the display-font classes (`.celestial-kicker` /
  `.celestial-headline`) and the ambient layer. P8 adds the global starfield to
  this short list of app-level injections (§3.5). Everything else is structural CSS
  scoped through a widget `key=`, never colors/fonts a token already covers.
- **Money is `Decimal`, never `float`.**
- **Icons are Google Material Symbols** (`:material/...:`) — never emoji. Mockups
  use inline SVG glyphs as a stand-in; in Streamlit these become Material symbols
  where a symbol exists, and remain inline SVG only for the bespoke celestial
  glyphs (game-shape glyphs, constellation, astrolabe) that have no Material
  equivalent.
- **Ambient imagery** obeys DESIGN §3: ≤20% opacity, never behind dense tables, no
  AI-generated images, license recorded in `ambient_manifest.json`. The starfield
  is *generated SVG* (not a raster image), so it is exempt from the licensing rule
  but still obeys the ≤20% opacity ceiling and the never-behind-tables rule (it is
  fully occluded by opaque blocks — it shows only in empty gutters).
- **Run the three gates before "done":** `poetry run pytest tests/golden/`,
  `poetry run pytest -m integration -n0`, `poetry run ruff check src/sportstradamus/`.
- **Run the `refactoring-specialist` subagent** on every touched Python file before
  any push/PR/review (CLAUDE.md mandatory step).

---

## 3. Cross-cutting decisions (apply on every surface)

These are shared infrastructure and rule changes. They should land **first** (§10),
because most surfaces depend on them.

### 3.1 DESIGN.md amendment — gold becomes the highlight color

**Owner override, resolved this session.** Today DESIGN.md (lines ~46–49) says gold
is "the oracle's color: kickers, prophecy headlines, constellation highlights,
correlation strength — never body text, never primary buttons ... never a
substitute for green/red semantics." P8 **extends** the sanctioned uses of gold to
include the **interactive-highlight / active-state** role:

- Gold marks the **active/selected/hovered** state: table row hover, the active
  lens, the active filter chip, the active tab underline, an active segmented-control
  segment.
- **Primary blue `#2E6BE6` is reserved for primary buttons and links only.** It is
  no longer used as a generic "active/selected" accent.

Everything else about gold is unchanged and still binding:

- Gold is **never a data mark** — never a bar, a heatmap cell, a plotted line/point
  that encodes a value, or a team-color star. (The gold *correlation edge* in the
  constellation is the sole exception, and it is pre-existing DESIGN §4a: correlation
  strength *is* the oracle's domain.)
- Gold is **never** body text, never a primary button fill, never a substitute for
  green/red semantics.

Update DESIGN.md prose to state the highlight role, and keep the `config.toml` /
`theme.py` mirror consistent (the token value does not change; only the usage rule
does). Extend `test_design_tokens.py` only if it currently asserts that gold is
absent from interactive states — otherwise no test change is needed.

### 3.2 Over / under directional arrows

Replace the textual "Over"/"Under" bet-side indicator with a **shape-redundant,
colorblind-safe directional cue**: a green ▲ (over) or red ▼ (under), rendered as a
small inline SVG triangle immediately preceding the line number. The color carries
the semantic (green good-direction/red), the *shape* carries it again so it survives
grayscale. Applies wherever a bet side is shown for a leg — primarily the Board
`Line` column (which folds the old `Bet` column into it), but also anywhere else a
pick's side appears. See `p8-board.html` for the exact glyph.

### 3.3 Verbose market names (`market_display.json` + helper)

Users want readable market titles ("3-Pt Made", not "FG3M"). Introduce a
**display-only** translation layer:

- New config: `src/sportstradamus/data/config/market_display.json`, shape
  `{ "LEAGUE": { "slug": "Display Name", ... }, ... }`.
- New helper: `sportstradamus.helpers.market_display_name(league, slug)` returning
  the display string, falling back to the slug when no mapping exists.
- **The slug stays the logic key everywhere** — feature columns, model cells,
  archive keys, `stat_meta.json`, correlation matrices. Only the *rendered* label
  changes. Never key logic off the display name.
- **Combos keep their established shorthand** (PRA, PR, RA, BLST) — these are
  already reader-friendly.
- **Opaque API codes get spelled out** (FG3M → "3-Pt Made", DREB → "Def. Rebounds",
  etc.).
- Applies to **all user-facing surfaces and all leagues/markets**. The Lab surfaces
  may show the slug alongside the display name (advanced users think in slugs), but
  the consumer surfaces (Tonight, Board, Games, Receipts, Details) show the display
  name.

The mapping table itself is a content deliverable — populate it for every
`(league, market)` cell that currently renders an opaque code. Cross-reference
`stat_map.json` (existing name mappings across APIs/sportsbooks) as a starting point;
do not duplicate its logic, just author the human labels.

### 3.4 Match column format

Wherever a player's game/matchup is shown, format it **player-team-first**:
`{team} {@|v} {opp}` where `@` = the player's team is **away**, `v` = the player's
team is **home**. Example: `LVA @ IND` (Las Vegas visiting Indiana) vs `LVA v IND`
(Las Vegas hosting). This replaces the separate Team + Opp columns on the Board
(§4.3). Home/away comes from the game context; the player's own team is always the
left token.

### 3.5 Global starfield

A single, fixed, faint field of twinkling stars sits behind **every** surface,
filling the empty side gutters that appear on wide screens where there is no data.

- Structure (from the mockups): `.starfield { position:fixed; inset:0; z-index:0;
  pointer-events:none }` sitting behind the main content layer (`z-index:1`).
- It is **generated inline SVG** (star `<use>` symbols + twinkle keyframes), not an
  image — so it is not subject to the ambient-image licensing rule.
- **Masking:** opaque `backgroundColor`/`surface` blocks occlude it completely, so it
  never shows through a table or a card — only through the empty gutters and the gaps
  between blocks. This is what reconciles it with DESIGN's "never behind dense
  tables" rule. The implementer must ensure the app's content blocks have solid
  backgrounds so nothing bleeds through behind data.
- Opacity stays ≤20% (DESIGN ambient ceiling). Twinkle animation must be wrapped in
  `@media (prefers-reduced-motion: no-preference)`.
- **Injected once** at the app level (`dashboard/app.py`), alongside the existing
  display-font and ambient injections — **not** retrofitted into each page module.
  The locked per-page mockups embed their own copy for preview convenience; in the
  real app it is one inject. (Owner explicitly approved not retrofitting the locked
  mocks.)
- **Streamlit integration risk to verify:** Streamlit's DOM nests content in
  `.block-container`. The implementer must confirm the fixed layer renders behind the
  block container and that block backgrounds are opaque. This is the one piece of
  P8 whose feasibility depends on Streamlit's DOM; validate early.

### 3.6 Chart retheme — kill the off-token colors

Several chart modules hardcode **off-palette** colors that violate DESIGN: green
`#2ecc71`, red `#e74c3c`, orange `#f39c12`, and a RdYlGn diverging ramp. These must
all move to the DESIGN tokens. This is real, pre-existing code debt that P8 fixes.

- **Confirmed affected:** `src/sportstradamus/dashboard/surfaces/lab_diagnostics_charts.py`
  (uses `plotly.express` + `plotly.graph_objects` with the off-token hexes and
  RdYlGn) and the inline charts in `lab_correlations.py`. Audit `receipts.py` charts
  for the same.
- **Replacements:** green → `#1F9D55`, red → `#E5484D`, orange → `#F5A524`; any
  diverging heatmap → the DESIGN **red ↔ neutral ↔ blue** diverging ramp
  (`chartDivergingColors`), **never RdYlGn, never gold**. Sequential heatmaps → the
  10-step blue ramp (`chartSequentialColors`).
- Charts are **Plotly**; retheme via the color arguments / `theme.py` template, not by
  hand-picking hexes at each call site. Prefer inheriting from the Plotly template so
  future charts get the tokens for free.
- **Every chart carries real axis tick labels + legends.** "Style is not an excuse
  for a bad chart" (owner). The CRPS axis in Diagnostics shows real mean-CRPS numbers
  (≈3.0–6.0), not hi/lo placeholders.

### 3.7 Shared Lab filter panel

The three Lab surfaces (Diagnostics, Correlations, Training) share one **granular
diagnostic-filter panel** built from the real dimensions in `stat_meta.json`. The Lab
is for advanced users, so filtering is deliberately fine-grained — "basically
everything in `stat_meta.json`."

Filter dimensions (all present in `stat_meta.json`; counts are the current corpus of
99 cells across MLB/NBA/NFL/NHL/WNBA):

- **Distribution family** — `dist`: SkewNormal / ZINB.
- **Target normalization** — `target_normalization`: none / centered_additive_mean10 /
  centered_additive_eb_meanyr_k10 / ratio_meanyr.
- **Post-hoc** — `posthoc`: none / roe_mean / cdf_recal_isotonic / prob_recal_isotonic /
  isotonic_mean.
- **Blending** — `blending`: None / crps.
- **HPO selection** — `hpo_selection`: None / calibrated.
- **Count dispersion objective** — `count_dispersion_objective`: None / pit_ks.
- **ZINB mode** — `zinb_mode`: None / hurdle.
- **Release surface** — `shipped`: withheld / devel / main.
- Plus **league / market / market-type / window / min-n** scoping.

On **Diagnostics**, `Family` and `Norm` are also *table columns* (not just filters).
The panel renders **fully on Diagnostics** and appears **collapsed** on Correlations
and Training (a one-line summary chip strip with an "expand" affordance) since it is
shared chrome the user has already seen — see the `.fbar` element in the two Lab
mockups. Training's current **sidebar** filters (`lab_training.py` lines ~258–281)
move up into this shared top panel.

### 3.8 Canonical scoring-column names

The Board and any surface showing per-leg economics must use the **post-rename**
canonical column names (a pipeline-wide rename already landed on this branch; do not
reintroduce the old overloaded `Model EV`). The distinct quantities are: **Model EV**
= Win% × Boost (betting EV); **Model Edge** = EV − 1; **Consensus Edge** = market
EV − 1; **Kelly** = (EV − 1)/(Boost − 1); **Projection** = the stat mean; **Bet** =
"Over"/"Under" (now shown as the ▲/▼ arrow). **Win %** = the model's win probability
for the pick. Reference the existing dashboard data loaders for the exact frame column
names; do not blind-rename.

---

## 4. Per-surface specifications

Each subsection: the mockup(s), the intent, the layout deltas, the data wiring, and
the surface-specific gotchas. Source modules live in
`src/sportstradamus/dashboard/surfaces/` and `.../components/`.

### 4.1 Details dialog — `p8-details.html`, `p8-details-tabs.html` (locked rev 3)

The per-offer drill-down dialog. Header treatment: the **Themed workbench** option
was chosen (sober header — Cinzel kicker "◈ {LEAGUE} · {Platform}", Plex player
name, gold hairline rule, **no nebula wash**). The "Prophecy dossier" alternative
(nebula header + Cormorant name) was **rejected**: there is no honest per-*leg*
prophecy sentence (the p3b story engine writes those per *parlay*, not per leg), and
DESIGN forbids numerals in Cormorant, so a poetic header here would be decoration
without substance.

**Fixed top block (all tabs):** kicker, player + market (display name) + line/side,
edge badge (green/red), gold hairline, and a one-line **context strip** (implied team
total + Δ vs avg, moneyline-derived **win prob**, game shape, DVPOA/defensive
context).

**Five tabs** (Cinzel tab labels, gold active underline — §3.1):

1. **History** — last-10 result bars vs the line: green bar = over, red = under; the
   line drawn *in front* at full opacity (dashed white) with a "line 23.5" gold tag;
   x-ticks show game date + opponent. A segmented **All games / vs {opp}** filter
   (gold active segment).
2. **Model** — the projected distribution vs the line, in **two variants by cell
   type**: *continuous* (density area split at the line — red mass below, green mass
   above — with book-consensus solid-gray line, the line dashed white, and the model
   projection as a single gold mean dot) and *count* (a discrete histogram, bars ≥line
   green / <line red, same overlays). Bare numbers only (no chart-junk). P(over) shown
   in green.
3. **Comps** — the KNN player-comparable table (strongest comp first): Comp / Games /
   Avg vs opp / vs-their-avg, the last column a **diverging red↔blue heatmap** on the
   tails, zebra striping, Plex Mono right-aligned.
4. **Other** — supporting per-stat rows: a vertical percentile gauge (colored on the
   gold ramp by percentile *among same-position slate players*), the value, and a
   sparkline. A scale key explains the percentile framing.
5. **Correlated** — legs correlated with this pick, grouped **Same team / Opponent**,
   each row = leg description + **EV lift** (green) + View/slip buttons. **EV lift =
   the copula-scored parlay EV of {this pick + candidate} minus the pick alone;
   positive-lift legs only.**

**Data wiring / code notes:**
- Source module: the offer dialog (the summary notes `deep_dive.py` as the current
  Details surface). Verify the current module name and wire the tab bodies there.
- The **Correlated tab and the astrolabe (§4.4) share a dependency**: a **pairwise /
  joint correlation EV+probability compute**. This needs either a precompute at
  `prophecize` time or a live compute in the prediction path (`find_correlation` /
  the copula scorer in the parlay code). Treat it as one shared capability, not two.
- Market names use `market_display_name` (§3.3); win prob is moneyline-derived (§5).

### 4.2 Tonight — `p8-tonight.html`

The landing surface, **full celestial**: nebula prophecy cards. The P8 delta here is
small and mostly consistency:

- The **comet glyph was unified** to the Games style — a radial-gradient glowing core
  with line tails — replacing the older filled-streak-polygon comet. The card glyph
  and the legend glyph both use this form (see the `cometA`/`cometB` gradients in the
  mockup).
- Game-shape glyphs come from the unified glyph set (§4.4, `p8-glyphs.html`).
- Prophecy headlines/sublines come from the p3b story engine (`narrative.py`), already
  merged. Cormorant for the headline (no numerals), Plex for all data.
- Source module: `tonight.py`.

### 4.3 Board — `p8-board.html` (locked rev 4)

The dense all-offers table, **sober workbench**. Columns condensed **14 → 10**:

`League · Match · Player · Market · Line · Boost · Win % · Model Edge · Cons Edge · Platform`

- **Match** = `{team} {@|v} {opp}` (§3.4), merging the old Team + Opp columns.
- **Market** = display name (§3.3).
- **Line** = the ▲/▼ side arrow (§3.2) + the number, folding the old **Bet** column in.
- **Model Edge** = a **diverging red↔blue heatmap** cell (§3.6 ramp).
- **Cut:** the **Kelly** column. **Deferred:** the **Trend** column (until an "L1"
  follow-up); do not add it in P8.
- **Interactions:** a **Tonight** lens + an **Over/Under side** segmented filter; the
  active lens/filter is **gold** (§3.1).
- **Row hover** is a **gold trim applied client-side** (an aggrid `.ag-row-hover` CSS
  rule) so hovering does **not** trigger a Streamlit rerun. Tables are
  `streamlit-aggrid` per DESIGN §3.
- Source module: `board.py`. Uses the canonical scoring-column names (§3.8) and reads
  the `current_offers` parquet snapshot.

### 4.4 Games — `p8-games.html` + `p8-constellation-lab.html` + `p8-games-lenses.html` + `p8-glyphs.html`

The **celestial centerpiece**. Four locked pieces:

**(a) The game hero** (`p8-games.html`) — a nebula card, matchup-first: the two teams,
a **comet/shape glyph** for the game shape, a Cormorant-italic **gold prophecy
subline**, and Total / Spread / Shape. Game shapes come from `current_game_context`
(shootout, grind, blowout, coinflip, even).

**(b) The constellation star map** — the signature element (DESIGN §4a). Its grammar
is fixed and must not drift:
- **star = leg**, **fill = team color**, **size ∝ model edge**, **brightness ∝
  in-slip** (a leg in the slip burns full color/opacity; a candidate is the same
  color desaturated toward gray and dimmed — selection is alpha + saturation, never a
  ring, because a gold ring reads as a team color).
- **Team fills come from the new `team_assets.json` (a P8 deliverable — see below).
  Team fills are NEVER gold.**
- **gold correlation edges** between two active/picked stars only, width/opacity ∝
  |ρ|, dashed when ρ<0 ("fights the thesis"). This is the one sanctioned gold data
  mark (§3.1, DESIGN §4a).
- Team-anchored force-directed layout; a single-sided game reaches its second team
  through the **satellite** section / the "Look wider" lens, never by crowding the
  map.

**(c) The astrolabe** (`p8-constellation-lab.html`, locked rev 5) — a bespoke,
animated readout of the parlay's aggregate stats, replacing a flat metric row. This is
the most intricate new element; build it to the mockup exactly. Structure:
- **Legs + Payout** pinned fixed on the left.
- **Win orbital** (radius ≈112): **two separate dots** — **blue = correlated** win
  probability (this dot carries the thread down to the Kelly core), **gray =
  independent** win probability. The **lift is an arc drawn between the two dots**,
  which grows/shrinks as they separate and is **green when correlated > independent,
  red when correlated < independent**, with a magical gold-shimmer gradient.
- **EV orbital** (radius ≈74) runs in the **opposite angular direction** to the win
  orbital (EV climbs one way, win the other — an owner-specified aesthetic).
- **Kelly = the core gem**, fed by threads from **both** the EV dot and the
  correlated-win dot (Kelly depends on both, made visual).
- **Crown = fixed shared reference maxima**: Win 30% / EV +12% / Kelly 3%. A value
  maps [0 → crown] onto [orbital bottom → orbital top]; **past the crown**, the bead
  pins at 12 o'clock and the orbital glows (an overflow signal).
- **Animates only on select/deselect** of a star (not continuously) — CSS transitions
  triggered by a class toggle. Respect `prefers-reduced-motion`.
- **Readouts: four rows only** — Win (correlated big, independent small beneath),
  Lift, EV, Kelly — **numbers only**, JS-updated in lockstep with the dials. No
  descriptive sublabels.
- The bezel / degree ticks / cardinal ticks / crown reticle / inner twinkle are
  **decorative engraving only** (not data).

**(d) The two lenses** (`p8-games-lenses.html`) — these **replace** the old "Add a leg
from another game" / "Add a leg the model doesn't like" expanders; both become lenses
on the star map, independent toggles, **active lens = gold**:
- **Look deeper** — reveal *this game's* model-passed legs (the ones the model
  doesn't like, K≤0) as **dim, cool, unconnected background stars**; the lit
  constellation stays intact in front. Tapping one rides it along in the slip.
- **Look wider** — **zoom out**: the current constellation shrinks to the center and
  *other games'* best legs orbit the edges, grouped by matchup. Tapping one pulls it
  in as a **satellite** (it rounds out a one-sided game into a valid both-teams
  parlay).

**(e) The glyph set** (`p8-glyphs.html`) — all five game-shape glyphs remade in one
consistent style: **line-art strokes build the form, a glowing radial core marks the
focus.** Comet (shootout), Supernova (blowout), Scales (coinflip), Lone star (even),
Hourglass (grind). Authored at viewBox `0 0 60 60` to drop into Tonight and Games.

**Data wiring / code notes:**
- **`team_assets.json` is a P8 deliverable.** `constellation.py` currently uses a
  2-color placeholder (`_TEAM_PALETTE = ("#2E6BE6", "#E69F00")`) with the comment
  "placeholder until team_assets.json lands (P8)." Author
  `src/sportstradamus/data/assets/team_assets.json` (shape `{ "LEAGUE": { "TEAM":
  {"primary": "#...", "secondary": "#..."}, ... } }`) with real per-team colors for
  every team across the five leagues, and wire both the Plotly `constellation.py` and
  the hand-authored slip constellation to read from it. (Team *logos/marks* are a
  later art-licensing item, §9 — this deliverable is colors only.)
- **Rendering paths differ — verify each.** `constellation.py` renders the star map as
  a **Plotly** figure (transparent paper/plot bg, modebar off). The **game-first slip
  constellation** on the Games surface is a **hand-authored static SVG/JS** widget
  (no npm — the established pattern). The **astrolabe and the lenses** follow the
  hand-authored SVG/JS pattern via `st.components.html`, fed a JSON payload from
  Python and animating on select/deselect client-side. Confirm which path each piece
  uses and keep team colors sourced from `team_assets.json` in all of them.
- Slip state and satellites: `slip_builder.py`, `satellite_picker.py`.
- The astrolabe's correlated-vs-independent win prob + EV lift come from the **shared
  pairwise/joint correlation compute** (§4.1) applied across the whole slip.
- Prophecy sublines: `narrative.py`.

### 4.5 Receipts — `p8-receipts.html` (locked rev 2 + calibration)

Historical betting performance, **sober workbench + one nebula hero**. The intent is
to **show off** (contrast with the Lab surfaces, which are for finding weak spots).

- **One celestial hero** — the **verdict card**: nebula wash, gold ROI, a
  cumulative-units chart with a **real** month/unit axis (and the worst drawdown
  marked, e.g. a −41u February marker).
- **"The scars"** — skeptic tiles (worst month in red) so the page is honest, not just
  a highlight reel.
- **By league / market / platform grid** — a **diverging red↔blue heatmap** on Units,
  with a **Group-by** control and a **Window** segmented filter
  (Last week / month / 3mo / year / All).
- **"Your slips" = receipt tickets** — each a physical-ticket metaphor: a status rail
  + stamp (**won green / lost red / pending gold**), a thesis headline, legs with
  ✓/✗, EV / stake / return, and a faint slip-constellation watermark on wins.
- **Rolling-accuracy chart** — with ticks + legend.
- **Calibration panel** (this **replaced** an earlier "recs/day" panel — owner:
  "alt lines and ladders will be popular, so prove profitability off more than just
  picking over/under on the basic line"): a **reliability diagram** (predicted vs
  realized hit rate, diagonal reference), **blue = standard-line bins + gold =
  alt/ladder bins** (gold reaching into the tails), an **ECE** figure, and an **ROI
  split** (standard vs alt, e.g. +7.2% / +9.4%) demonstrating that ladders pay.
- **CLV tiles** — close-line-value, with **close coverage shown as a RATE** (e.g.
  88.6%), never a raw count (owner: "one big number is meaningless").
- **Strategy simulator** — explicitly **labeled retrospective (hindsight)**, and the
  forward **Simulated-Bettor Ledger** is **scarred in** for the future addition: a
  bankroll curve with the circuit breaker (drawdown >20% halves stakes, >30% halts),
  platform-aware (Underdog + Sleeper). See §9 / `docs/handoffs/sim-bettor-ledger.md`
  (entry D6) — the forward ledger is a *future* lane; P8 only reserves its visual slot
  and labels the existing simulator honestly.

**Data wiring / code notes:**
- Source module: `receipts.py`; reads `parlay_hist.parquet` (+ `history.parquet`).
- **CLV and any archive-derived series must come from a parquet export**, not a live
  archive query (§2 DuckDB rule). If the close-line data isn't already in a snapshot,
  that export is a prerequisite (a cron job writes it; the dashboard reads it).
- The calibration reliability data (predicted vs realized by bin, split
  standard/alt) needs a source frame — confirm it exists in `parlay_hist` or add the
  precompute.

### 4.6 Diagnostics (Lab) — `p8-lab-diagnostics.html` (locked rev 2)

**Sober workbench + starfield gutters, no nebula.** Reframed as
**find-the-weak-spots** — the opposite intent from Receipts. Owner: "the point of the
Lab screens is to help me figure out areas for improvement ... fundamentally
different from Receipts."

- **Worst-BSS-first** default sort; weak cells flagged with an **amber rail**
  (`tbody tr.flag { box-shadow: inset 3px 0 0 var(--orange) }`); a **"start here"**
  summary strip up top.
- The **shared granular filter panel** (§3.7), rendered in full here. `Family` and
  `Norm` are also table columns.
- **Charts rethemed** (§3.6): the off-token `#2ecc71/#e74c3c/#f39c12` + RdYlGn are
  gone; bias bar (green/orange/red tokens), BSS diverging bar, Murphy decomposition
  (blue-resolution / red-reliability), reliability diagram (gold curve), CRPS line
  with **real** mean-CRPS axis numbers.
- Source modules: `lab_diagnostics.py` + `lab_diagnostics_charts.py`; reads
  `model_stats.parquet` + the live-metrics parquet.

### 4.7 Correlations (Lab) — `p8-lab-correlations.html`

**Sober workbench + shared filter bar (collapsed) + starfield.** Retheme + diagnostic
framing + one new visual.

- **New lead visual — the stat-pair correlation heatmap.** A "Correlations" lab was
  missing the canonical correlation artifact. Render the stat-pair correlation matrix
  as a **diverging red↔blue heatmap** (never gold), sourced from the committed
  `{LEAGUE}_corr.csv` files. **Follow-up:** later overlay **empirical vs model ρ** to
  surface pairs the copula mis-prices (that overlay is a follow-up, not P8 core; note
  it in the section but don't build it now).
- **Correlation value-add** — an Independent-P vs Correlated-P scatter (Hit green /
  Miss red, no-adjustment diagonal) + three metrics (boosted-parlay count, hit rate
  boosted vs not).
- **Boost vs hit rate** — bar by boost bucket.
- **Hit rate by parlay size** — a per-platform table + a stacked miss-breakdown bar
  (**Hit-all green / Missed-1 orange / Missed-2+ red**, all tokens).
- **Parlay calibration curve** — predicted correlated-P vs actual hit rate, diagonal
  reference.
- A **diagnostic callout** up top when correlation isn't paying (boosted hit rate <
  unboosted), pointing at the driver pair.
- **All charts rethemed** from the same off-token colors (§3.6) with real ticks.
- Source module: `lab_correlations.py`; reads `parlay_hist.parquet` (value-add,
  calibration) + `{LEAGUE}_corr.csv` (heatmap).

### 4.8 Training (Lab) — `p8-lab-training.html`

**Sober workbench + starfield gutters.** This is a **reference ledger**, not a chart
surface — so the existing **nine metric-family tabs stay** (Overview, Scoring rules,
Discrimination, Rates, EV & lines, Kelly & blending, Dispersion, Ship gates,
Hyperparameters — the `TAB_COLUMNS` in `lab_training.py`). P8 adds:

- **Sober lab chrome** — kicker, starfield, and the **shared filter bar** (§3.7). The
  current **sidebar** filters (`lab_training.py` ~258–281) move up into the shared
  panel.
- **Run-at-a-glance strip** — tiles: Cells trained / Shipping (all gates) / One gate
  short / Withheld — plus a **lifecycle funnel** (trained → Gate-1 pass → all-gates →
  graduated live).
- **The ship-gate matrix — the find-weak-spots hero.** Cells × **G1–G6**, ● pass /
  ○ fail, **worst-first**, with **one-gate-short cells pinned to the top** (amber
  rail) so the cheapest wins are obvious, and multi-fail cells on a red rail.
  **Green ●/red ○ only — no gold on data.** **G6 (anti-shrinkage) joins the matrix**
  per the current `model_stats.parquet` schema (`g1_pass`…`g6_pass`); note that
  `lab_training.py` currently lists only g1–g5 in `TAB_COLUMNS` — extend it to g6.
- Gates recap (for column headers/help): G1 paired-Brier CI vs book · G2/G3 tail
  bias-z · G4 IQR ratio (dispersion) · G5 debiased ECE · G6 anti-shrinkage. **Ship =
  AND of all six.**
- Source module: `lab_training.py`; reads `model_stats.parquet` + `lifecycle_table()`
  from `training/graduation.py` (which joins offline Gate-1 with live Gate-2).

---

## 5. Data wiring summary (new capabilities P8 needs)

The dashboard reads parquet snapshots only. P8 introduces or depends on:

1. **`market_display.json` + `helpers.market_display_name`** (§3.3) — new config +
   helper; display-only.
2. **`team_assets.json`** (§4.4) — new asset config; per-team colors for all five
   leagues; replaces the 2-color placeholder in `constellation.py`.
3. **Pairwise / joint correlation EV+probability compute** (§4.1, §4.4) — one shared
   capability feeding both the Details "Correlated" tab (2-leg EV lift) and the
   astrolabe (whole-slip correlated-vs-independent win prob + EV lift). Precompute at
   `prophecize` or compute live in the parlay/`find_correlation` path. **Do not build
   two.**
4. **Moneyline → win-probability conversion** (§4.1 context strip, Board Win %,
   astrolabe) — a de-vig conversion helper if one is not already present.
5. **Calibration frame** (§4.5) — predicted-vs-realized hit rate by bin, split
   standard vs alt/ladder, for the Receipts calibration panel. Confirm in
   `parlay_hist` or add the precompute.
6. **CLV / close-line export** (§4.5) — if not already in a snapshot, a cron job must
   export it to parquet (never a live archive query).
7. **Global starfield inject** (§3.5) — one app-level injection in `app.py`.
8. **Plotly token template** (§3.6) — retheme so charts inherit DESIGN tokens.

---

## 6. Follow-up lanes (out of P8 scope — captured for direction)

These are explicitly **not** built in P8, but the direction is locked so the plan
knows where things are going. Each is its own future lane.

### 6.1 Skyrim-style constellation shapes (loose)

The constellation should eventually form **recognizable, sports-themed shapes** with a
glowing outline (the Skyrim-skill-tree feeling), matching the league (a catalog of
many shapes). **Loose, not rigid** (owner): like real constellations, a given shape
(e.g. a basketball hoop) need not be the exact same graph every time — a few nodes may
be missing, a few filler stars added at the edges; it can be abstract. What matters is
that the **general outline is present** and a **faint background silhouette (~13%
opacity)** signals the intent to the user. Star→vertex assignment must reconcile the
existing grammar (size ∝ edge, team-seed anchoring, correlation-adjacency) with the
shape template. This is a substantial lane on its own.

### 6.2 Art-slot licensing audit

Walk every place licensed art could enhance the UX and produce a flagged list for the
owner to go acquire assets. Known slots so far: the **astrolabe**, the **starfield
background**, the **game-shape glyphs**, the **constellation outlines / silhouettes**,
and **team marks/logos** (distinct from the team *colors* in `team_assets.json`, which
are P8 core). Expect to find more during implementation. Deliverable: an audit list,
not the assets themselves.

### 6.3 Forward Simulated-Bettor Ledger (D6)

Per `docs/handoffs/sim-bettor-ledger.md`: a pre-registered, append-only,
decision-time-committed paper-trading ledger that settles nightly and reports
walk-forward ROI/CLV, with the circuit breaker and platform-awareness. Distinct from
the retrospective `strategies/profit_sim.py` (which is load-bearing for the S3
supersede gate — **do not modify it in this lane**). P8 only reserves the Receipts
visual slot and labels the existing simulator "retrospective (hindsight)."

### 6.4 Empirical-vs-model ρ overlay (Correlations Lab)

The correlation heatmap (§4.7) ships in P8 showing empirical ρ. A follow-up overlays
**model ρ vs empirical ρ** to flag pairs the copula mis-prices.

---

## 7. Suggested implementation phasing (for the plan author)

P8 is large; the surfaces share infrastructure. A dependency-respecting order:

- **Phase A — shared infrastructure (unblocks everything):** DESIGN.md gold-highlight
  amendment (§3.1) + token mirror; `market_display.json` + helper (§3.3);
  `team_assets.json` (§4.4); the global starfield inject + Streamlit masking
  validation (§3.5); the Plotly chart retheme / token template (§3.6); the shared Lab
  filter panel component (§3.7). Land the over/under arrow + Match-format helpers
  (§3.2, §3.4) here too.
- **Phase B — sober surfaces:** Board (§4.3), the three Lab surfaces (§4.6–4.8),
  Receipts data grids + calibration + tickets (§4.5). These are mostly table/chart
  work on top of Phase A.
- **Phase C — celestial surfaces:** the shared pairwise/joint correlation compute
  (§5 item 3) first, then the Details tabs (§4.1), Tonight glyph unification (§4.2), the
  Games centerpiece — constellation `team_assets` wiring, the astrolabe, the two
  lenses, the glyph set (§4.4) — and the Receipts nebula hero (§4.5). Highest craft;
  build to the mockups exactly.
- **Deferred (not P8):** the follow-up lanes in §6.

Each phase should end with the three gates green and the `refactoring-specialist`
run on touched Python.

---

## 8. Testing

- **Must stay green:** `tests/golden/test_design_tokens.py` (extend only if it
  currently forbids gold in interactive states — §3.1),
  `tests/golden/test_dashboard_no_archive_lock.py` (no new archive-importing module),
  the full `tests/golden/` suite, `pytest -m integration -n0`, and
  `ruff check src/sportstradamus/`.
- **Add:** render-level AppTest smoke pins for the new/changed surfaces (note the
  AppTest `__file__`-relative `st.Page` path gotcha — a runpy wrapper preserves the
  real path; see the existing dashboard AppTest harness). A golden assertion that no
  chart module contains the off-token hexes `#2ecc71`/`#e74c3c`/`#f39c12` or `RdYlGn`
  would lock §3.6 against regression.
- The starfield masking (§3.5) needs a manual visual check in a running Streamlit app;
  it is the one item unit tests can't fully cover.

---

## 9. Open dependencies / risks to surface early

- **Starfield in Streamlit's DOM** (§3.5) — validate the fixed-layer masking before
  building much on top of it.
- **The pairwise/joint correlation compute** (§5 item 3) gates both Details-Correlated and
  the astrolabe — sequence it first in Phase C.
- **CLV / calibration snapshots** (§5 items 5–6) — if the source frames don't exist, a
  cron parquet export is a prerequisite and touches the ops side, not just the
  dashboard.
- **`market_display.json` coverage** — must cover every opaque code across five
  leagues or some labels fall back to slugs; treat authoring the table as real work.
