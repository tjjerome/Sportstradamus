# P8 Phase D — Loose Constellation Shapes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [x]`) syntax for tracking. This phase is sized for an **Opus-class implementer**:
> the shape bank is craft work and the star→vertex assignment is judgment-heavy; the contracts
> below are binding, the interior math is yours to make beautiful.

**Goal:** Every game's star map forms a recognizable, sports-themed constellation — *loosely*,
the way real constellations only gesture at their namesake — chosen to fit that game's actual
correlation structure, with **no shape repeating anywhere on a league's slate for the day**, and
without breaking one atom of the FIXED star-map grammar.

**Architecture:** Five pure pieces compose the engine, all deterministic, all testable in
isolation: (1) a committed **shape bank** (`constellation_shapes.json`, 49 hand-authored
templates tagged by topology class and league — story-engine fidelity: authored data under
documented rules, coverage-enforced by goldens); (2) a **player-supernode clusterer** that
collapses a player's tightly-correlated legs into one node for planning; (3) a **topology
classifier** that reads the collapsed graph's structure (hub / chain / twin / mesh / generic);
(4) a **slate assigner** that deals distinct, topology-matched templates across the league's
games for the date; (5) a **star→vertex assignment** that maps supernodes onto template
vertices honoring team sides and ρ-adjacency, then explodes each supernode back into its member
stars. The silhouette + outline render as a **decoration layer** (non-interactive, never gold)
beneath the existing data layer. **Every classifier/assigner threshold is owner-tunable at
runtime**: they live in the catalog's top-level `tuning` block, the loader hot-reloads on file
mtime (edit JSON → browser rerun, no dashboard restart), and a diagnostics expander on the
Games page shows each game's raw topology readings so tuning is done with eyes open — including
a dedicated `variety_lambda` knob that spreads template classes when a whole slate classifies
the same way.

**Prereqs:** Phase C merged (team-colored stars, lenses — Phase D layers under them). Spec §6.1
of `docs/archive/superpowers/specs/2026-07-03-p8-oracle-assets-celestial-polish-design.md`; the "loose
shapes" demo at the bottom of `docs/mockups/p8-constellation-lab.html` is the visual reference
("the faint silhouette carries the 'hoop.' Same rules underneath: size ∝ edge, each team seeds
its side").

**Branch:** `feature/dashboard-ux`. Gates + refactoring-specialist per task (see the Phase 0
plan's Context section; same rules).

---

## The non-negotiables (read twice)

DESIGN §4a grammar is FIXED and this phase must leave every clause intact:

- star = leg · **fill = team color** · **size ∝ edge (Kelly)** · **brightness = in-slip**
  (alpha + saturation, never a ring) · **gold edges = correlation only** (width/opacity ∝ |ρ|,
  dashed when ρ < 0).
- **Supernodes are a planning device, not a grammar change.** Every leg still renders as its
  own star with its own size/brightness/fill; clustering only decides *where* the member stars
  land (a tight knot around one template vertex). After explosion, star = leg, exactly as
  before.
- The map stays **static per game**: selecting a leg restyles a star, never moves one
  (`test_layout_is_static_under_selection` keeps passing — template choice and positions are a
  function of the game's pool and the slate, never the slip).
- Each team still seeds its own side; an unrepresented team still leaves its half visibly empty.
- **The decoration layer is never gold and never interactive.** Template outline strokes and the
  silhouette are engraving (starfield-family blues/grays), visually subordinate to the gold
  correlation edges. If a viewer could mistake an outline stroke for a correlation edge, the
  styling is wrong.
- Silhouette opacity ≈ 13% (`_SILHOUETTE_ALPHA = 0.13`, spec §6.1) — under the DESIGN ambient
  20% ceiling.
- **No trademarked marks.** Shapes are generic sporty objects and mascot *archetypes* (a maple
  leaf, a crown, an octopus — never a team's actual logo geometry). Hand-authored vectors are
  ours; anything traced from a logo is an IP problem Phase E's licensing rules exist to prevent.
- **DESIGN §4a's layout clause changes and must be amended in the same commit** (Task D7): the
  owner approves the amendment by approving this plan. Only the layout sentence changes; every
  other clause is untouchable.

---

## Sizing & topology groundwork (why the engine is shaped this way)

**Slate sizes set the bank floor.** Worst-case games per league per calendar day:

| League | Teams (2026) | Max game instances/day |
|---|---|---|
| NBA | 30 | 15 |
| WNBA | 15 | 7 |
| NFL | 32 | 16 |
| NHL | 32 | 16 |
| MLB | 30 | 17 (doubleheaders — same matchup twice counts twice) |

No-repeat-per-(league, date) therefore needs ≥ 17 eligible templates per league as a hard
floor; the bank ships 37 eligible per league (33 universal + 4 league-flavored) so
topology matching still has room to be choosy on a full slate. Golden-enforced floors:
`eligible(league) >= 20` and `eligible(league, class) >= 6` counting primary+secondary tags,
every class covered in every league.

**Topology classes come from how betting graphs actually look.** The correlation web (post
supernode collapse — see below) lands in a few recurring shapes:

- **hub** — one node carries a dominant share of edge weight: the QB stack (QB passing volume
  at the center, each WR/TE receiving leg a spoke), an NBA usage king with correlated
  teammates. Wants radial templates (dartboard, wheel, sun).
- **chain** — low branching, long path: game-script sequences (workhorse RB carries → rush yds
  → team total → opposing pace) where each node ties mainly to its neighbors. Wants linear
  templates (bolt, snake, arrow).
- **twin** — two dense team clusters, thin cross-game bridge: the classic two-team stack.
  Wants two-lobe templates (goalposts, hourglass, barbell, wings).
- **mesh** — dense all-pairs correlation: same-team NBA star clusters where PTS/REB/AST legs of
  multiple players inter-correlate. Wants compact filled templates (crown, shield, trophy).
- **generic** — no strong signature (sparse book-fallback games). Any template welcome.

**Player supernodes are what make classification honest.** A player's own markets (PTS, PRA,
FGM…) are near-duplicates at ρ ≈ 0.5–0.9; left raw they flood the graph with intra-player
edges and everything classifies mesh. Collapsing each player's tightly-correlated legs into one
planning node exposes the *inter-player* structure — which is the shape worth drawing — and
shrinks n to the 5–15 range the templates are authored for. The members then explode back out
as a tight knot around their vertex (a bright multi-star point, which is also just prettier).

---

### Task D1: The shape bank — `constellation_shapes.json` + loader + validation

**Files:**
- Create: `src/sportstradamus/data/config/constellation_shapes.json`
- Create: `src/sportstradamus/dashboard/components/constellation_shapes.py` (loader half)
- Test: `tests/golden/test_constellation_shapes.py`

**Schema** (per-template tags replace v1's league→list map; coordinates normalized to [-1, 1]²
so they drop onto the existing anchor axis; `side` splits vertices for team seeding;
`prominence` orders which vertices matter most when nodes are scarce):

```json
{
  "version": 2,
  "tuning": {
    "cluster_rho": 0.35,
    "hub_top_share": 0.30,
    "twin_cross_share": 0.25,
    "chain_mean_degree": 2.3,
    "chain_diameter_frac": 0.5,
    "mesh_density": 0.45,
    "min_shape_nodes": 3,
    "variety_lambda": 0.5
  },
  "templates": {
    "the-hoop": {
      "label": "The Hoop",
      "leagues": ["NBA", "WNBA"],
      "topology": {"primary": "mesh", "secondary": ["hub"]},
      "min_nodes": 4,
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
      "silhouette": "M -0.95 0.55 A 0.95 0.42 0 0 1 0.95 0.55 L 0.45 0.10 L 0.30 -0.85 L -0.30 -0.85 L -0.45 0.10 Z"
    },
    "the-goalposts": {
      "label": "The Goalposts",
      "leagues": ["NFL"],
      "topology": {"primary": "twin", "secondary": []},
      "min_nodes": 3,
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
      "silhouette": "M -0.06 -0.9 L 0.06 -0.9 L 0.06 -0.4 L 0.78 -0.26 L 0.78 0.85 L 0.66 0.85 L 0.66 -0.14 L 0.06 -0.28 L -0.06 -0.28 L -0.66 -0.14 L -0.66 0.85 L -0.78 0.85 L -0.78 -0.26 L -0.06 -0.4 Z"
    },
    "the-hourglass": {
      "label": "The Hourglass",
      "leagues": "all",
      "topology": {"primary": "twin", "secondary": ["chain"]},
      "min_nodes": 2,
      "vertices": [
        {"id": 0, "x": -0.7, "y": 0.85, "side": "L", "prominence": 1},
        {"id": 1, "x": 0.7, "y": 0.85, "side": "R", "prominence": 1},
        {"id": 2, "x": 0.0, "y": 0.05, "side": "C", "prominence": 3},
        {"id": 3, "x": -0.7, "y": -0.85, "side": "L", "prominence": 2},
        {"id": 4, "x": 0.7, "y": -0.85, "side": "R", "prominence": 2}
      ],
      "outline": [[0, 1], [1, 2], [2, 4], [4, 3], [3, 2], [2, 0]],
      "silhouette": "M -0.7 0.85 L 0.7 0.85 L 0.05 0.05 L 0.7 -0.85 L -0.7 -0.85 L -0.05 0.05 Z"
    },
    "the-dartboard": {
      "label": "The Dartboard",
      "leagues": "all",
      "topology": {"primary": "hub", "secondary": ["mesh"]},
      "min_nodes": 4,
      "vertices": [
        {"id": 0, "x": 0.0, "y": 0.0, "side": "C", "prominence": 1},
        {"id": 1, "x": 0.45, "y": 0.0, "side": "R", "prominence": 2},
        {"id": 2, "x": 0.0, "y": 0.45, "side": "C", "prominence": 3},
        {"id": 3, "x": -0.45, "y": 0.0, "side": "L", "prominence": 2},
        {"id": 4, "x": 0.0, "y": -0.45, "side": "C", "prominence": 3},
        {"id": 5, "x": 0.95, "y": 0.0, "side": "R", "prominence": 4},
        {"id": 6, "x": 0.48, "y": 0.82, "side": "R", "prominence": 5},
        {"id": 7, "x": -0.48, "y": 0.82, "side": "L", "prominence": 5},
        {"id": 8, "x": -0.95, "y": 0.0, "side": "L", "prominence": 4},
        {"id": 9, "x": -0.48, "y": -0.82, "side": "L", "prominence": 6},
        {"id": 10, "x": 0.48, "y": -0.82, "side": "R", "prominence": 6}
      ],
      "outline": [[0, 1], [0, 2], [0, 3], [0, 4], [1, 5], [3, 8], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 5]],
      "silhouette": "M -0.95 0 A 0.95 0.95 0 1 0 0.95 0 A 0.95 0.95 0 1 0 -0.95 0 Z M -0.2 0 A 0.2 0.2 0 1 0 0.2 0 A 0.2 0.2 0 1 0 -0.2 0 Z"
    },
    "the-bolt": {
      "label": "The Bolt",
      "leagues": "all",
      "topology": {"primary": "chain", "secondary": []},
      "min_nodes": 3,
      "vertices": [
        {"id": 0, "x": -0.8, "y": 0.9, "side": "L", "prominence": 1},
        {"id": 1, "x": -0.3, "y": 0.45, "side": "L", "prominence": 3},
        {"id": 2, "x": -0.45, "y": 0.2, "side": "L", "prominence": 5},
        {"id": 3, "x": 0.15, "y": -0.15, "side": "R", "prominence": 4},
        {"id": 4, "x": 0.0, "y": -0.4, "side": "C", "prominence": 6},
        {"id": 5, "x": 0.75, "y": -0.9, "side": "R", "prominence": 2}
      ],
      "outline": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]],
      "silhouette": "M -0.8 0.9 L -0.22 0.42 L -0.38 0.18 L 0.75 -0.9 L 0.08 -0.12 L 0.24 0.12 Z"
    },
    "the-diamond": {
      "label": "The Diamond",
      "leagues": ["MLB"],
      "topology": {"primary": "mesh", "secondary": ["hub"]},
      "min_nodes": 4,
      "vertices": [
        {"id": 0, "x": 0.0, "y": -0.85, "side": "C", "prominence": 1},
        {"id": 1, "x": 0.62, "y": -0.15, "side": "R", "prominence": 2},
        {"id": 2, "x": 0.0, "y": 0.5, "side": "C", "prominence": 3},
        {"id": 3, "x": -0.62, "y": -0.15, "side": "L", "prominence": 2},
        {"id": 4, "x": 0.0, "y": -0.2, "side": "C", "prominence": 5},
        {"id": 5, "x": -0.9, "y": 0.45, "side": "L", "prominence": 4},
        {"id": 6, "x": 0.9, "y": 0.45, "side": "R", "prominence": 4},
        {"id": 7, "x": 0.0, "y": 0.95, "side": "C", "prominence": 6},
        {"id": 8, "x": -0.85, "y": -0.55, "side": "L", "prominence": 7},
        {"id": 9, "x": 0.85, "y": -0.55, "side": "R", "prominence": 7}
      ],
      "outline": [[0, 1], [1, 2], [2, 3], [3, 0], [0, 5], [0, 6], [5, 7], [7, 6]],
      "silhouette": "M 0 -0.85 L 0.62 -0.15 L 0 0.5 L -0.62 -0.15 Z M -0.9 0.45 A 1.1 0.75 0 0 1 0.9 0.45 L 0 -0.85 Z"
    }
  }
}
```

(The six above are **fully authored exemplars** — one per topology class plus extras — and ship
as-is; they are the format ground truth for the rest.)

**Authoring rules** (documented as `S1–S9` in `constellation_shapes.py`'s module docstring, the
way `bank.py` documents R1–R6; the golden enforces what's mechanical):

- **S1** Coordinates normalized to [-1, 1]², shape centered, use the box — no postage stamps.
- **S2** 5–13 vertices. `prominence` is the importance order (1 = the star the shape can't live
  without). Player-outline shapes simplify to ≤13 vertices; the silhouette carries the rest.
- **S3** Sides must let a typical 3+3 two-team split read: mirrored objects get mirrored L/R;
  asymmetric objects (bat, bolt) still tag both halves along their long axis; `C` for
  spine/center vertices only.
- **S4** `outline` is the minimal line-work that names the object. Every pair references real
  ids; interior vertices (a mound, a bullseye) may be outline-free.
- **S5** `silhouette` is one SVG path — the filled gesture, ≤ ~8 commands. It carries what the
  line-work can't (the bolt's body, the leaf's lobes).
- **S6** `min_nodes` = fewest real supernodes that still read as the object (2–5).
- **S7** `label` is the nameplate text ("The Bolt"). `topology.primary` = the class the vertex
  graph *is*; at most two secondaries.
- **S8** Generic archetypes only, never trademarked geometry (non-negotiables above).

**The required bank** — author all 49 (6 exemplars ✓ above + 43 below). Verts/min are targets,
±2 fine; iterate coordinates while eyeballing Task D5's live render — the JSON is the tuning
surface. Batch by topology class, one commit per batch, golden green after each.

Universal, hub-primary (7 more): `the-wheel` (bicycle wheel: hub, rim ring, spokes; 10/4) ·
`the-sun` (core + 8 rays; 9/3) · `the-whistle` (body + pea + radiating blast lines; 8/3) ·
`the-rocket` (nose-body spine + fin spread + exhaust burst; 10/4, secondary chain) ·
`the-torch` (handle spine + flame burst; 8/3, secondary chain) · `the-stopwatch` (dial + crown
+ hands + ticks; 8/3, secondary mesh) · `the-racket` (string-bed hub + handle; 9/3, secondary
chain).

Universal, chain-primary (6 more): `the-snake` (S-winding serpent, head prominent; 9/3) ·
`the-arrow` (nock→tip with fletching pair; 7/3) · `the-sword` (blade spine + crossguard; 7/3,
secondary twin) · `the-range` (mountain ridge zigzag; 8/3, secondary twin) · `the-pennant`
(pole + triangular flag; 6/2, secondary twin) · `the-oar` (shaft + blade; 6/2).

Universal, twin-primary (8 more): `the-barbell` (bar + plate clusters both ends; 8/3) ·
`the-butterfly` (two wing lobes + body spine — also the goalie-stance pun; 8/3) · `the-gloves`
(two boxing gloves squared up; 10/4) · `the-longhorns` (skull + two horn sweeps; 9/3) ·
`the-antlers` (two beams, tines up — Bucks/moose archetype; 10/4) · `the-scales` (beam, two
pans — the bettor's shape; 9/3) · `the-wings` (spread wings + body center; 11/4, secondary
mesh) · `the-bridge` (two towers + deck span; 10/4, secondary chain).

Universal, mesh-primary (9 more): `the-trophy` (cup + handles + base; 10/4, secondary twin) ·
`the-crown` (5-point crown; 9/3, secondary hub) · `the-shield` (crest outline + boss; 8/3) ·
`the-bell` (bell body + yoke + clapper; 8/3) · `the-shamrock` (three lobes + stem; 9/3,
secondary hub) · `the-horseshoe` (arc + nail-hole studs; 8/3, secondary twin) · `the-trident`
(three prongs + shaft; 8/3, secondary hub) · `the-anchor` (shank + arms + ring; 8/3, secondary
twin) · `the-foam-finger` (mitt + index up; 8/3, secondary chain).

NBA/WNBA-only (3 more): `the-shot-clock` (clock face + digits block; 9/3, hub/mesh) ·
`the-basketball` (ball circle + radiating seams; 9/3, hub/mesh) · `the-jumpshot` (player
mid-jumper outline; 12/5, mesh/hub).

NFL-only (3 more): `the-helmet` (shell + facemask grid; 10/4, mesh) · `the-football` (ellipse +
lace ticks; 8/3, mesh/twin) · `the-heisman` (stiff-arm pose outline; 12/5, mesh/chain).

MLB-only (3 more): `the-mitt` (catcher's mitt + pocket; 9/3, mesh/hub) · `the-slugger` (batter
mid-swing outline; 11/4, mesh/chain) · `the-marlin` (leaping marlin, bill to tail; 8/3,
chain/twin).

NHL-only (4 more): `the-hockey-stick` (shaft + blade; 6/2, chain) · `the-maple-leaf` (leaf
lobes + stem; 9/3, mesh/hub) · `the-zamboni` (box body + wheels + auger; 10/4, mesh/chain) ·
`the-octopus` (head + eight arms — Detroit tradition archetype; 11/4, hub/mesh).

Loader half of `constellation_shapes.py` — **hot-reloading**, because the JSON is the owner's
live tuning surface (thresholds *and* coordinates), and Streamlit only caches imported modules,
not files read at render time:

```python
def shape_catalog() -> dict:
    """Parsed constellation_shapes.json; validated on load (ids contiguous,
    outline refs valid, sides in {L,R,C}, coords within [-1,1], min_nodes <= n
    vertices, topology classes known, labels non-empty; tuning block complete
    with in-range values). Cached on the file's mtime (internal
    lru_cache-keyed-on-mtime_ns helper), so an owner edit + browser rerun
    picks up new values immediately — no dashboard restart."""

def tuning() -> dict:
    """The catalog's top-level 'tuning' block — every classifier/assigner
    threshold, owner-editable (schema above documents semantics; the JSON has
    no comments, this docstring is the reference)."""

def eligible_templates(league: str) -> list[str]:
    """Slugs allowed for a league ('all' or listed), catalog order."""
```

- [x] Golden first: schema validation per template (as docstring above, plus both sides
  represented); tuning-block validation (exactly the eight keys, each within its sane range —
  shares/densities/rho in (0,1), `chain_mean_degree` in (1,4), `min_shape_nodes` ≥ 2,
  `variety_lambda` ≥ 0); **hot-reload pin** (write a temp catalog, load, bump a tuning value +
  mtime, load again → new value visible); **floor pins** — `len(eligible_templates(lg)) >= 20`
  for all five leagues, ≥ 6 per (league, class) counting primary+secondary, all four
  non-generic classes covered per league; a slate-maxima table constant in the test ties the 20
  floor to the sizing table above.
- [x] Author the bank in class batches (exemplars → hub → chain → twin → mesh → league packs);
  loader; gates; commit per batch, final
  `feat(p8-d): constellation shape bank — 49 templates, topology-tagged`

---

### Task D2: Player supernodes + topology classifier

**Files:**
- Modify: `src/sportstradamus/dashboard/components/constellation_shapes.py`
- Test: `tests/golden/test_constellation_shapes.py` (extend)

All thresholds arrive as arguments (pure functions — tests pass explicit values, production
callers pass `tuning()`), so a JSON edit retunes everything downstream with no code involved:

```python
def cluster_players(
    nodes: list[str],                      # leg keys "Player|Market|Bet"
    edges: list[tuple[str, str, float]],   # (a, b, |rho|) — the layout web
    rho_min: float,                        # tuning()['cluster_rho']
) -> dict[str, list[str]]:
    """Supernode key -> member leg keys (sorted; key = '+'.join(sorted(members))).
    Same-player legs merge along pairwise |rho| >= rho_min connected
    components; cross-player never merges; singletons pass through. Collapsed
    edge weight between supernodes = max member-pair |rho| (preserves the
    adjacency signal clustering exists to expose)."""

def topology_class(
    supernodes: list[str],
    node_team: dict[str, str | None],
    edges: list[tuple[str, str, float]],   # collapsed
    cfg: dict,                             # the tuning() block
) -> tuple[str, dict]:
    """(label, readings). label: 'hub' | 'chain' | 'twin' | 'mesh' | 'generic'.
    Deterministic rule cascade, first match wins: n < 4 or no edges ->
    'generic'; cross-team weight share < cfg['twin_cross_share'] (both teams
    populated) -> 'twin'; top node's incident-weight share >=
    cfg['hub_top_share'] -> 'hub'; mean degree <= cfg['chain_mean_degree'] and
    diameter >= cfg['chain_diameter_frac']*n -> 'chain'; density >=
    cfg['mesh_density'] -> 'mesh'; else 'generic'.

    readings: {'n', 'cross_share', 'top_share', 'mean_degree',
    'diameter_frac', 'density'} — the raw metrics behind the verdict, fed to
    the Games diagnostics expander (D6) so threshold tuning is done against
    observed values, not guesses."""
```

The cascade order is deliberate: twin before hub (a two-cluster graph often *contains* a local
hub per side — the two-lobe read wins), hub before chain (a star graph trivially satisfies low
mean degree), mesh last (densest signature, hardest to fake). No thresholds in the code —
everything reads from the caller-supplied tuning block, so the owner retunes against the D6
readings without touching Python.

- [x] Goldens first, synthetic fixtures per class (run at the shipped tuning values): a star
  graph (1 center, 6 spokes) → hub; a 6-path → chain; two 4-cliques + one weak bridge across
  teams → twin; a 6-clique → mesh; 2 nodes → generic. Threshold sensitivity: the 6-clique
  reclassifies away from mesh when `mesh_density` is passed above its reading (proves the knob
  is live end-to-end). Readings dict: keys complete, values match hand-computed metrics for the
  star-graph fixture. Clustering: same player PTS/PRA/AST at ρ .7/.6/.65 → one supernode; same
  player at ρ .1 → stays split; cross-player ρ .9 → never merges; collapsed edge = max member
  pair; determinism (byte-equal outputs).
- [x] Implement; gates; commit
  `feat(p8-d): player supernodes + graph topology classifier`

---

### Task D3: Slate assigner — no repeats, topology-matched

**Files:**
- Modify: `src/sportstradamus/dashboard/components/constellation_shapes.py`
- Test: `tests/golden/test_constellation_shapes.py` (extend)

```python
def assign_templates(
    league: str,
    date: str,
    games: list[tuple[str, str, int]],   # (game_key, topology_class, n_supernodes)
    cfg: dict,                           # the tuning() block
) -> dict[str, str | None]:
    """game_key -> template slug, distinct across the slate. Deterministic:
    md5(f'{league}|{date}|{catalog version}') seeds a Fisher-Yates shuffle of
    eligible_templates(league); games processed in sorted(game_key) order; each
    game takes the highest-scoring unused template, ties broken by shuffle
    position. Score = class fit − variety pressure:

        fit(t) = 2 if t.primary == game class, 1 if in t.secondary, 0 else
                 (generic games: fit = 1 for every template)
        score(t) = fit(t) − cfg['variety_lambda'] * dealt_count[t.primary]

    where dealt_count tracks how many already-dealt templates share t's
    primary class. variety_lambda = 0 -> pure topology fidelity; ~0.5 nudges a
    homogeneous slate toward class spread; >= 2 forces it — the owner's
    'everything is coming up mesh' knob, live via the tuning block. A game
    with n_supernodes < cfg['min_shape_nodes'] or below every remaining
    template's min_nodes gets None (spring fallback). Same inputs -> same
    dealing, forever; a new date reshuffles the deck."""
```

The no-repeat guarantee is structural: eligible pools (≥ 20, actually ~35) exceed every league's
worst-case slate (table above), so the unused set never runs dry; the D1 floor golden is what
keeps that true as the bank evolves. MLB doubleheaders are two game instances with distinct
keys — they get two different shapes by construction.

Caller wiring (Task D5): `games.py` computes the slate list once per (league, date) from the
games frame it already renders (each game's supernode graph + class via D2 — small graphs,
microseconds) and passes each game its slug. Every game on the slate resolves the same
assignment no matter which one the user is looking at.

- [x] Goldens first: 16-game synthetic slate → 16 distinct slugs; determinism across calls;
  changing the date changes the dealing; a hub-classified game receives a hub-primary template
  when one is free; **variety knob** — on an 8-game all-mesh synthetic slate, the count of
  distinct primary classes dealt is strictly higher at `variety_lambda=2` than at `0` (the
  all-mesh escape hatch, proven); pool-exhaustion impossible pin (slate maxima table vs
  floors); thin game → None.
- [x] Implement; gates; commit `feat(p8-d): slate template assigner — distinct, topology-matched`

---

### Task D4: Star→vertex assignment + supernode explosion (the heart of the phase)

**Files:**
- Modify: `src/sportstradamus/dashboard/components/constellation_shapes.py`
- Test: `tests/golden/test_constellation_shapes.py` (extend)

```python
def assign_stars(
    supernodes: list[str],
    node_team: dict[str, str | None],      # per supernode
    teams: list[str],                      # [left_team, right_team] — anchor order
    edges: list[tuple[str, str, float]],   # collapsed web
    template: dict,
) -> tuple[dict[str, tuple[float, float]], list[int]]:
    """Map supernodes onto template vertices. Returns (positions, filler_vertex_ids).
    positions: supernode -> (x, y), every supernode placed. filler_vertex_ids:
    outline vertices no supernode filled — rendered as faint decorative stars so
    the shape still reads. Deterministic: no randomness, ties broken by key."""

_EXPLODE_R0, _EXPLODE_DR, _EXPLODE_RMAX = 0.05, 0.015, 0.11

def explode_clusters(
    positions: dict[str, tuple[float, float]],
    clusters: dict[str, list[str]],
) -> dict[str, tuple[float, float]]:
    """Supernode positions -> per-leg positions. Singleton clusters sit exactly
    on their vertex. k>=2 members ring their vertex (deterministic: members in
    sorted order, angles from -90° equally spaced, radius
    min(_EXPLODE_R0 + _EXPLODE_DR*(k-2), _EXPLODE_RMAX)) — a tight knot that
    reads as one bright point, each star still its own leg."""
```

Binding behaviors (the interior algorithm is yours; these are pinned by tests):

1. **Team sides hold.** A supernode of `teams[0]` never lands on a `side == "R"` vertex and
   vice versa (`C` vertices take either, preferring the scarcer team). If one team has more
   supernodes than its side has vertices, the overflow **jitters around its own side's
   centroid** (deterministic golden-angle spiral, radius ~0.18) — never crosses the axis.
   Explosion never crosses it either (a knot near the axis clamps to its side).
2. **Correlation adjacency is respected greedily.** Strongly-tied supernode pairs (top-|ρ|
   collapsed edges) land on template-adjacent or near vertices. Recommended shape (not
   mandated): seed each side with its highest weighted-degree supernode on that side's most
   prominent vertex (mirroring `_anchors`' `max(strength)` rule, constellation.py:271-281),
   then place the rest in weighted-degree order, each at the free same-side vertex minimizing
   `Σ_assigned |ρ(s, other)| · dist(vertex, pos(other))`. scipy's `linear_sum_assignment` per
   side is an acceptable alternative — pick one, delete the other, no dual paths.
3. **Loose by design.** Fewer supernodes than vertices: unfilled outline vertices become filler
   ids (decorative). More: overflow jitters (rule 1). Never stretch or regenerate a template
   per game — the same template must be visibly *the same shape* across games and dates.
4. **Empty-half preserved.** A single-team game fills only its side's vertices; the other
   side's all return as fillers — the empty half still reads (fillers are faint enough not to
   fake a populated team; verify against D5's styling).
5. **Prominence earns brightness.** Higher-leg-count / higher-degree supernodes take
   lower-prominence-number (more important) vertices — the shape's key stars are the game's key
   players.

- [x] Goldens first: side-purity pin; determinism pin (byte-equal positions across calls);
  overflow-jitter stays side-of-axis; single-team → other side all fillers; a hand-built
  4-supernode/2-per-team case lands the two strongest-ρ nodes on adjacent vertices; a 3-member
  cluster explodes to a ring of radius `_EXPLODE_R0 + _EXPLODE_DR` centered on its vertex;
  explosion clamps at the axis.
- [x] Implement; gates; commit
  `feat(p8-d): star→vertex assignment + supernode explosion — loose, team-true, ρ-adjacent`

---

### Task D5: Rendering — silhouette, outline glow, filler stars, wiring

**Files:**
- Modify: `src/sportstradamus/dashboard/components/constellation.py`
- Modify: `src/sportstradamus/dashboard/surfaces/games.py` (slate wiring)
- Modify: `src/sportstradamus/dashboard/components/constellation_component/build/main.js`
  (decoration traces excluded from hover/click — one guard)
- Test: `tests/golden/test_constellation.py` (extend)

```python
_SILHOUETTE_ALPHA = 0.13   # spec §6.1 "~13%" — faint intent signal, under the 20% ambient ceiling
_OUTLINE_COLOR = "rgba(230,233,239,0.22)"  # engraving family — NEVER gold (grammar §4a)
_FILLER_COLOR = "rgba(138,145,160,0.30)"   # theme GRAY at low alpha; smaller than any leg star
```

1. `_layout` gains the template path: cluster → (template from the slate assignment, passed in
   by games.py) → `assign_stars` → `explode_clusters`; positions replace the spring result;
   `None` template → existing spring layout unchanged (the fallback IS the current behavior —
   zero-risk degradation, pinned).
2. **Silhouette**: the template's `silhouette` path as a Plotly `layout.shapes` entry
   (`type: "path"`, line width 0, fill from the team-neutral blue family at
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
6. **games.py slate wiring**: build the (game_key, class, n_supernodes) list from the frame it
   already holds, call `assign_templates` once per render, hand each game's slug down to the
   figure. **No `st.cache_data` on this path** — the graphs are tiny (microseconds) and a cache
   would eat the tuning loop: an edited tuning block must reclassify + re-deal on the very next
   rerun.
7. Lens interplay (Phase C): Look-deeper background stars and Look-wider satellites render
   *outside* the template (perimeter) — decoration never repositions them; verify visually.

- [x] Figure pins first: silhouette shape present at `_SILHOUETTE_ALPHA`; decoration traces
  carry `hoverinfo="skip"` + no gold anywhere in the decoration layer; gold edge trace count
  unchanged vs a no-template render; spring fallback byte-stable when the assigner returns
  None; static-under-selection pin still green.
- [x] Implement; **live tuning session** (`poetry run dashboard`, real slate): iterate bank
  coordinates + alphas until each shape reads at a glance without shouting, and a full slate
  shows all-distinct shapes; record a one-paragraph verdict + screenshot note in the commit
  body.
- [x] Gates; commit `feat(p8-d): decoration layer + slate wiring (silhouette, glow, fillers)`

---

### Task D6: Nameplate + tuning diagnostics

**Files:**
- Modify: `src/sportstradamus/dashboard/surfaces/games.py`
- Test: AppTest smoke extension

- [x] ~~A quiet Cinzel caption under the map names the constellation.~~ **Built, then cut on the
  owner's call after the live pass** — "just let the shape speak for itself." A shape that has to
  be told to you isn't reading, so the map is uncaptioned and the template's `label` survives only
  in the tuning cockpit. DESIGN §4a records the rule.
- [x] **"Constellation tuning" expander** (collapsed by default, sober styling, bottom of the
  page): one row per slate game — game · class · n_supernodes · the D2 readings
  (cross_share / top_share / mean_degree / diameter_frac / density) · dealt template ·
  spring-fallback flag — plus a caption naming the tuning surface
  (`data/config/constellation_shapes.json` → `tuning`) and the loop: edit → save → rerun
  (`R`), no restart. This is the owner's cockpit: when a slate reads all-mesh, the density
  column says whether to raise `mesh_density` (reclassify) or raise `variety_lambda` (keep the
  verdicts, spread the shapes).
- [x] Gates; commit `feat(p8-d): constellation nameplate + tuning cockpit` (nameplate
  later removed — see above)

---

### Task D7: DESIGN §4a amendment (same-commit rule)

**Files:** `DESIGN.md` §4a.

Replace only the layout sentence: "Layout is **team-anchored force-directed**…" becomes a
clause stating layout is **template-guided** (a topology-matched, slate-distinct shape bank;
each team's legs seed its own side; a player's tightly-correlated legs knot around one vertex;
correlation strength still pulls tied legs adjacent; thin games fall back to the force layout),
and add one sentence defining the decoration layer (silhouette ≈13%, outline/filler engraving;
never gold, never interactive, never a data mark). Every other §4a clause stays byte-identical.
Run `tests/golden/test_design_tokens.py` before committing (presence-needle pins).

- [x] Commit `docs(p8-d): DESIGN §4a — template-guided constellation layout`

---

## Exit criteria

- Live walkthrough on a real multi-game slate: every game a nameable, *distinct* shape;
  topology sanity spot-check (a QB-stack game radial, a two-team stack two-lobed); team sides
  honest; empty halves empty; same-player legs knotted; gold correlation edges unmistakably
  distinct from the engraving; screenshots noted in the ledger entry.
- **Tuning loop verified live**: edit a tuning value (e.g. `mesh_density`) in
  `constellation_shapes.json`, browser rerun → games reclassify and the slate re-deals, no
  dashboard restart; on an all-one-class night, raising `variety_lambda` visibly spreads the
  shape classes; the diagnostics expander's readings match the shapes on screen.
- Goldens: bank validation + tuning-block validation + hot-reload pin + eligibility floors
  (≥20/league, ≥6/league-class), clustering + classifier fixture matrix + threshold-sensitivity
  pin, slate no-repeat + determinism + variety-knob pin, assignment
  side/adjacency/overflow/explosion pins, decoration-layer figure pins, spring-fallback
  stability, static-under-selection.
- Three gates green; refactoring-specialist on every touched `.py`.
