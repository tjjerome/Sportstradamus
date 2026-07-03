# P8 Phase D — Loose Constellation Shapes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking. This phase is sized for an **Opus-class implementer**:
> the star→vertex assignment and the aesthetic tuning are judgment-heavy; the contracts below are
> binding, the interior math is yours to make beautiful.

**Goal:** The star map forms recognizable, sports-themed constellation shapes with a glowing
outline and a faint background silhouette (the Skyrim-skill-tree feeling) — *loosely*, the way
real constellations only gesture at their namesake — without breaking one atom of the FIXED
star-map grammar.

**Architecture:** A committed per-league shape catalog (`constellation_shapes.json`: normalized
vertex sets + outline edges + silhouette path) + a pure assignment module
(`constellation_shapes.py`) that maps a game's legs onto a deterministically-chosen template
while honoring team sides and correlation adjacency. `constellation.py`'s `_layout` gains a
template-guided mode; the silhouette + outline render as a **decoration layer** (non-interactive,
never gold) beneath the existing data layer. The JS component is untouched except for ignoring
decoration traces on hover.

**Prereqs:** Phase C merged (team-colored stars, lenses — Phase D layers under them). Spec
§6.1 of `docs/superpowers/specs/2026-07-03-p8-oracle-assets-celestial-polish-design.md`; the
"loose shapes" demo section at the bottom of `docs/mockups/p8-constellation-lab.html` is the
visual reference (abstract hoop: "the faint silhouette carries the 'hoop.' Same rules
underneath: size ∝ edge, each team seeds its side").

**Branch:** `feature/dashboard-ux`. Gates + refactoring-specialist per task (see the Phase 0
plan's Context section; same rules).

---

## The non-negotiables (read twice)

DESIGN §4a grammar is FIXED and this phase must leave every clause intact:

- star = leg · **fill = team color** · **size ∝ edge (Kelly)** · **brightness = in-slip**
  (alpha + saturation, never a ring) · **gold edges = correlation only** (width/opacity ∝ |ρ|,
  dashed when ρ < 0).
- The map stays **static per game**: selecting a leg restyles a star, never moves one
  (`test_layout_is_static_under_selection` keeps passing — template positions are a function of
  the game's pool, not the slip).
- Each team still seeds its own side; an unrepresented team still leaves its half visibly empty.
- **The decoration layer is never gold and never interactive.** Template outline strokes and the
  silhouette are engraving (starfield-family blues/grays), visually subordinate to the gold
  correlation edges. If a viewer could mistake an outline stroke for a correlation edge, the
  styling is wrong.
- Silhouette opacity ≈ 13% (`_SILHOUETTE_ALPHA = 0.13`, spec §6.1) — under the DESIGN ambient
  20% ceiling and consistent with its never-behind-dense-tables spirit (the map sits on
  `backgroundColor`, no tables involved).
- **DESIGN §4a's layout clause changes and must be amended in the same commit** (Task D5): the
  owner approves the amendment by approving this plan. Only the layout sentence changes
  ("team-anchored force-directed" → template-guided with team-side seeding and
  correlation-adjacency); every other clause is untouchable.

---

### Task D1: The shape catalog — `constellation_shapes.json` + loader

**Files:**
- Create: `src/sportstradamus/data/config/constellation_shapes.json`
- Create: `src/sportstradamus/dashboard/components/constellation_shapes.py` (loader half)
- Test: `tests/golden/test_constellation_shapes.py`

Template schema (all coordinates normalized to [-1, 1]² so they drop onto the existing anchor
axis; `side` splits vertices for team seeding; `prominence` orders which vertices matter most
when legs are scarce):

```json
{
  "version": 1,
  "leagues": {
    "NBA": ["hoop", "basketball", "sneaker", "hourglass"],
    "WNBA": ["hoop", "basketball", "sneaker", "hourglass"],
    "NFL": ["goalposts", "football", "helmet", "hourglass"],
    "NHL": ["stick_and_puck", "goal_mask", "hourglass"],
    "MLB": ["diamond", "bat_and_ball", "glove", "hourglass"]
  },
  "templates": {
    "hoop": {
      "vertices": [
        {"id": 0, "x": -0.95, "y": 0.55, "side": "L", "prominence": 1},
        {"id": 1, "x": -0.60, "y": 0.80, "side": "L", "prominence": 3},
        {"id": 2, "x": -0.25, "y": 0.90, "side": "L", "prominence": 2},
        {"id": 3, "x": 0.25, "y": 0.90, "side": "R", "prominence": 2},
        {"id": 4, "x": 0.60, "y": 0.80, "side": "R", "prominence": 3},
        {"id": 5, "x": 0.95, "y": 0.55, "side": "R", "prominence": 1},
        {"id": 6, "x": -0.45, "y": 0.10, "side": "L", "prominence": 4},
        {"id": 7, "x": 0.45, "y": 0.10, "side": "R", "prominence": 4},
        {"id": 8, "x": -0.30, "y": -0.55, "side": "L", "prominence": 5},
        {"id": 9, "x": 0.30, "y": -0.55, "side": "R", "prominence": 5},
        {"id": 10, "x": 0.0, "y": -0.85, "side": "C", "prominence": 6}
      ],
      "outline": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [6, 8], [8, 10], [10, 9], [9, 7], [6, 7]],
      "silhouette": "M -0.95 0.55 A 0.95 0.42 0 0 1 0.95 0.55 L 0.45 0.10 L 0.30 -0.85 L -0.30 -0.85 L -0.45 0.10 Z",
      "min_legs": 5
    },
    "goalposts": {
      "vertices": [
        {"id": 0, "x": 0.0, "y": -0.9, "side": "C", "prominence": 4},
        {"id": 1, "x": 0.0, "y": -0.35, "side": "C", "prominence": 5},
        {"id": 2, "x": -0.75, "y": -0.2, "side": "L", "prominence": 2},
        {"id": 3, "x": 0.75, "y": -0.2, "side": "R", "prominence": 2},
        {"id": 4, "x": -0.75, "y": 0.85, "side": "L", "prominence": 1},
        {"id": 5, "x": 0.75, "y": 0.85, "side": "R", "prominence": 1},
        {"id": 6, "x": -0.75, "y": 0.35, "side": "L", "prominence": 3},
        {"id": 7, "x": 0.75, "y": 0.35, "side": "R", "prominence": 3}
      ],
      "outline": [[0, 1], [1, 2], [1, 3], [2, 6], [6, 4], [3, 7], [7, 5]],
      "silhouette": "M -0.06 -0.9 L 0.06 -0.9 L 0.06 -0.4 L 0.78 -0.26 L 0.78 0.85 L 0.66 0.85 L 0.66 -0.14 L 0.06 -0.28 L -0.06 -0.28 L -0.66 -0.14 L -0.66 0.85 L -0.78 0.85 L -0.78 -0.26 L -0.06 -0.4 Z",
      "min_legs": 4
    },
    "hourglass": {
      "vertices": [
        {"id": 0, "x": -0.7, "y": 0.85, "side": "L", "prominence": 1},
        {"id": 1, "x": 0.7, "y": 0.85, "side": "R", "prominence": 1},
        {"id": 2, "x": 0.0, "y": 0.05, "side": "C", "prominence": 3},
        {"id": 3, "x": -0.7, "y": -0.85, "side": "L", "prominence": 2},
        {"id": 4, "x": 0.7, "y": -0.85, "side": "R", "prominence": 2}
      ],
      "outline": [[0, 1], [1, 2], [2, 4], [4, 3], [3, 2], [2, 0]],
      "silhouette": "M -0.7 0.85 L 0.7 0.85 L 0.05 0.05 L 0.7 -0.85 L -0.7 -0.85 L -0.05 0.05 Z",
      "min_legs": 4
    }
  }
}
```

Author the remaining league templates to the same schema (basketball ball-with-seams, sneaker,
football ellipse-with-laces, helmet, hockey stick+puck, goal mask, baseball diamond,
bat_and_ball, glove) — **hand-place every vertex**; this is craft work, not generation. Rules of
thumb: 5–12 vertices; sides balanced enough that a typical 2-team split (3+3 legs) can read;
`hourglass` is the league-agnostic fallback in every league list. It is fine (encouraged) to
iterate coordinates while eyeballing Task D3's live render — the JSON is the tuning surface.

Loader half of `constellation_shapes.py`:

```python
@cache
def shape_catalog() -> dict:
    """Parsed constellation_shapes.json; validated once (ids contiguous, outline
    references valid, sides ∈ {L,R,C}, every league list non-empty)."""

def pick_template(league: str, game: str, date: str, n_legs: int) -> dict | None:
    """Deterministic per (game, date): md5 over 'game|date' indexes the league's
    template list filtered to min_legs <= n_legs. None when the pool is too thin
    (n_legs < every template's min_legs) — caller falls back to spring layout."""
```

- [ ] Golden first: schema validation (every template: contiguous ids, valid outline refs,
  coords within [-1,1], min_legs ≤ vertex count, both sides represented); `pick_template`
  determinism (same inputs → same template; different dates rotate); thin-pool → None.
- [ ] Author the full catalog; loader; gates; commit
  `feat(p8-d): constellation shape catalog + deterministic template picker`

---

### Task D2: Star→vertex assignment (the heart of the phase)

**Files:**
- Modify: `src/sportstradamus/dashboard/components/constellation_shapes.py` (assignment half)
- Test: `tests/golden/test_constellation_shapes.py` (extend)

Contract:

```python
def assign_stars(
    nodes: list[str],                      # leg keys, the _layout node list
    node_team: dict[str, str | None],
    teams: list[str],                      # [left_team, right_team] — anchor order
    edges: list[tuple[str, str, float]],   # (a, b, |rho|) — the layout web
    template: dict,
) -> tuple[dict[str, tuple[float, float]], list[int]]:
    """Map legs onto template vertices. Returns (positions, filler_vertex_ids).

    positions: node -> (x, y), every node placed. filler_vertex_ids: outline
    vertices no leg filled — rendered as faint decorative stars so the shape
    still reads (spec §6.1 'a few filler stars added at the edges').
    Deterministic: no randomness, ties broken by node key.
    """
```

Binding behaviors (the interior algorithm is yours; these are pinned by tests):

1. **Team sides hold.** A leg of `teams[0]` never lands on a `side == "R"` vertex and vice
   versa (`C` vertices take either, preferring the scarcer team). If one team has more legs
   than its side has vertices, the overflow **jitters around its own side's centroid**
   (deterministic golden-angle spiral, radius ~0.18) — never crosses the axis.
2. **Correlation adjacency is respected greedily.** Strongly-tied pairs (top-|ρ| edges) should
   land on template-adjacent or near vertices. Recommended shape (not mandated): seed each side
   with its highest weighted-degree leg on that side's most prominent vertex (mirroring
   `_anchors`' `max(strength)` rule, constellation.py:271-281), then place remaining legs in
   weighted-degree order, each at the free same-side vertex minimizing
   `Σ_assigned |ρ(leg, other)| · dist(vertex, pos(other))`. scipy's `linear_sum_assignment` per
   side is an acceptable alternative if you build a full cost matrix — pick one, delete the
   other, no dual paths.
3. **Loose by design.** `n_legs < vertices`: unfilled outline vertices become filler ids
   (decorative). `n_legs > vertices`: overflow jitters (rule 1). Never stretch/regenerate the
   template per game — the same template must be visibly *the same shape* across games.
4. **Empty-half preserved.** A single-team game fills only its side's vertices; the other
   side's vertices all return as fillers — the empty half still reads at a glance (fillers are
   faint enough not to fake a populated team; verify against D3's styling).

- [ ] Goldens first: side-purity pin; determinism pin (byte-equal positions across calls);
  overflow-jitter stays side-of-axis; single-team → other side all fillers; a hand-built
  4-leg/2-per-team case lands the two strongest-ρ legs on adjacent vertices.
- [ ] Implement; gates; commit `feat(p8-d): star→vertex assignment — loose, team-true, ρ-adjacent`

---

### Task D3: Rendering — silhouette, outline glow, filler stars

**Files:**
- Modify: `src/sportstradamus/dashboard/components/constellation.py`
- Modify: `src/sportstradamus/dashboard/components/constellation_component/build/main.js`
  (decoration traces excluded from hover/click — one guard)
- Test: `tests/golden/test_constellation.py` (extend)

Wiring into `constellation_figure`:

```python
_SILHOUETTE_ALPHA = 0.13   # spec §6.1 "~13%" — faint intent signal, under the 20% ambient ceiling
_OUTLINE_COLOR = "rgba(230,233,239,0.22)"  # engraving family — NEVER gold (grammar §4a)
_FILLER_COLOR = "rgba(138,145,160,0.30)"   # theme GRAY at low alpha; smaller than any leg star
```

1. `_layout` gains the template path: `pick_template(...)` → `assign_stars(...)`; positions
   replace the spring result; `None` template → existing spring layout unchanged (the fallback
   IS the current behavior — zero-risk degradation, pinned).
2. **Silhouette**: the template's `silhouette` path as a Plotly `layout.shapes` entry
   (`type: "path"`, line width 0, `fillcolor` from the *losing* team-neutral blue family at
   `_SILHOUETTE_ALPHA`) — drawn beneath everything.
3. **Outline glow**: template `outline` edges *between filled-or-filler vertices* as one
   scatter trace (`mode="lines"`, `_OUTLINE_COLOR`, width 1, `hoverinfo="skip"`,
   `name="decoration"`), plus a second identical trace at width 4 and ~35% of the alpha
   underneath it — the cheap two-stroke glow. Drawn **below** the gold correlation edges so
   gold always wins visually.
4. **Filler stars**: one scatter trace (`name="decoration"`), size below `_SIZE_MIN`,
   `_FILLER_COLOR`, `hoverinfo="skip"` — present so the outline reads, obviously not legs.
5. **main.js guard**: hover/click handlers skip traces named `decoration` (one `if` on
   `data.name` in the existing `plotly_hover`/`plotly_click` paths).
6. Lens interplay (Phase C): Look-deeper background stars and Look-wider satellites render
   *outside* the template (perimeter) — decoration never repositions them; verify visually.

- [ ] Figure pins first: silhouette shape present at `_SILHOUETTE_ALPHA`; decoration traces
  carry `hoverinfo="skip"` + no gold anywhere in the decoration layer; gold edge trace count
  unchanged vs a no-template render; spring fallback byte-stable when catalog returns None;
  static-under-selection pin still green.
- [ ] Implement; **live tuning session** (`poetry run dashboard`, real slate): iterate catalog
  coordinates + alphas until the shape reads at a glance without shouting; record a
  one-paragraph verdict + screenshot note in the commit body.
- [ ] Gates; commit `feat(p8-d): silhouette + outline glow + filler stars (decoration layer)`

---

### Task D4: Games-surface polish + template provenance caption

**Files:**
- Modify: `src/sportstradamus/dashboard/surfaces/games.py`
- Test: AppTest smoke extension

- [ ] A quiet Cinzel caption under the map names the constellation ("The Hoop", template
  display name from the catalog — add a `"label"` field to each template in D1's schema) —
  the Skyrim moment, one line, `.celestial-kicker` class, no other chrome.
- [ ] Gates; commit `feat(p8-d): constellation nameplate`

---

### Task D5: DESIGN §4a amendment (same-commit rule)

**Files:** `DESIGN.md` §4a.

Replace only the layout sentence: "Layout is **team-anchored force-directed**…" becomes a
clause stating layout is **template-guided** (a per-league constellation-shape catalog; each
team's legs seed its own side; correlation strength still pulls tied legs adjacent; games with
too few legs fall back to the force layout), and add one sentence defining the decoration layer
(silhouette ≈13%, outline/filler engraving; never gold, never interactive, never a data mark).
Every other §4a clause stays byte-identical. Run `tests/golden/test_design_tokens.py` before
committing (presence-needle pins).

- [ ] Commit `docs(p8-d): DESIGN §4a — template-guided constellation layout`

---

## Exit criteria

- Live walkthrough: a real slate shows a nameable shape per game, team sides honest, empty
  halves empty, gold correlation edges unmistakably distinct from the engraving; screenshots
  noted in the ledger entry.
- Goldens: catalog validation, picker determinism, assignment side/adjacency/overflow pins,
  decoration-layer figure pins, spring-fallback stability, static-under-selection.
- Three gates green; refactoring-specialist on every touched `.py`.
