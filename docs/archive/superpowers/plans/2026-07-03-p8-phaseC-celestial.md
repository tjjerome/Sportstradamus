# P8 Phase C — Celestial Surfaces Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the high-craft celestial pieces to their mockups exactly: the shared pairwise
EV-lift capability, the Details dialog's five tabs, the unified glyph set, the Games centerpiece
(team-colored constellation, astrolabe, the two lenses), and the Receipts nebula hero.

**Architecture:** One shared compute (thin `slip_engine` additions) feeds both the Details
Correlated tab and the astrolabe. The astrolabe is the only new hand-authored JS component
(constellation_component postMessage pattern — committed static files, no npm); the lenses are
modes of the existing Plotly constellation; glyphs are inline-SVG string constants.

**Prereqs:** Phase 0 (structured legs + `Corr Same`/`Corr Opp`) and Phase A merged. Phase B only
gates C7 (same file as B7). Spec §4.1, §4.2, §4.4, §5. Mockups: `p8-details.html`,
`p8-details-tabs.html` (rev 3), `p8-tonight.html`, `p8-games.html`, `p8-constellation-lab.html`
(rev 5), `p8-games-lenses.html`, `p8-glyphs.html`. Gates + refactoring-specialist per task.

---

### Task C1: Pairwise EV-lift + astrolabe payload (the shared compute)

**Files:**
- Modify: `src/sportstradamus/dashboard/slip_engine.py`
- Test: `tests/golden/test_slip_engine.py` (extend)

No new engine — `SlipScore` already carries `indep_p` **and** `joint_p` (slip_engine.py:53-54),
so correlated-vs-independent is a read. Two thin additions:

```python
def ev_lift(focus: Mapping, candidate: Mapping, corr: pd.DataFrame, *, platform: str,
            bankroll: float = 0.0, shrinkage: float = 1.0) -> float:
    """EV of {focus + candidate} minus focus alone (spec §4.1 Correlated tab).

    Both scored through the same copula path as the slip rail; ~milliseconds
    per candidate, so the Details tab computes live (no precompute).
    """
    pair = score_slip([focus, candidate], corr, platform=platform,
                      bankroll=bankroll, shrinkage=shrinkage)
    solo = score_slip([focus], corr, platform=platform,
                      bankroll=bankroll, shrinkage=shrinkage)
    return pair.model_ev - solo.model_ev


def astrolabe_payload(score: SlipScore, *, nonce: int) -> dict:
    """JSON contract for the astrolabe component. Crowns are the fixed shared
    reference maxima (spec §4.4c): Win 30% / EV +12% / Kelly 3%."""
    kelly = (score.model_ev - 1) / (score.payout - 1) if score.payout > 1 else 0.0
    return {
        "legs": score.bet_size,
        "play_type": score.play_type,
        "payout": score.payout,
        "payout_approximate": score.payout_approximate,
        "win_corr": score.joint_p,
        "win_indep": score.indep_p,
        "ev": score.model_ev - 1,
        "kelly": max(kelly, 0.0),
        "crowns": {"win": 0.30, "ev": 0.12, "kelly": 0.03},
        "nonce": nonce,
    }
```

(Money display stays out of the payload — the stake is already `Decimal` on the Python side and
renders in the existing rail; the astrolabe shows rates + multiplier only.)

- [ ] Goldens first: 2-leg hand-computed lift on a fixture corr slice (positive-ρ pair lifts,
  zero-ρ pair ≈ 0); payload shape + crown constants pinned; Sleeper `payout_approximate=True`
  passes through.
- [ ] Gates + commit `feat(p8-c): ev_lift + astrolabe payload on the existing copula path`

---

### Task C2: Details dialog rebuild (two sessions)

**Files:**
- Modify: `src/sportstradamus/dashboard/components/deep_dive.py`
- Modify: `src/sportstradamus/dashboard/components/deep_dive_charts.py`
- Create: `src/sportstradamus/dashboard/components/deep_dive_tabs.py` (deep_dive.py is ~382
  lines; the five tab renderers move here to respect the 300-line cap)
- Test: `tests/golden/test_deep_dive.py`, `test_deep_dive_charts.py` (extend)

Build to `p8-details-tabs.html` rev 3. The **Themed workbench** header won (sober; no nebula):
Cinzel kicker `◈ {LEAGUE} · {Platform}` (`.celestial-kicker`), Plex player name, market display
name + line/side with the ▲/▼ arrow, edge badge (existing `_edge_badge`), **gold hairline rule**,
one-line context strip (implied team total + Δ vs avg, moneyline win prob — already available as
`ml_fav_prob` via `load_current_game_context`, no new devig — game shape glyph, DVPOA).

Five tabs (Cinzel labels, gold active underline — Streamlit `st.tabs` styling via the sanctioned
`key=`-scoped CSS; if the underline color can't be reached, theme primary stands and the gap is
noted):

1. **History** — keep the existing last-10 bars; recolor: green over / red under vs the line;
   the app line dashed white at full opacity drawn in front + gold "line {x}" tag; x-ticks =
   date + opponent; add a segmented **All games / vs {opp}** filter (filters the frame the chart
   already receives).
2. **Model** — two variants by cell type off the offer's `Dist`: continuous (density area split
   at the line, red mass below / green above; consensus solid gray; app line dashed white; model
   projection = single gold dot — the existing `_projection_overlay` keeps its two-chart flat
   contract) and count (discrete histogram, bars ≥ line green / < line red, same overlays).
   P(over) in green. Bare numbers, no chart junk.
3. **Comps** — existing sidecar decode (`comps_vs_opp`), restyled: strongest first, zebra, Plex
   Mono right-aligned, last column (vs-their-avg %) heatmapped on `theme.DIVERGING_COLORS` tails.
4. **Other** — existing `other_stats` sidecar rows re-rendered: vertical percentile gauge
   (percentile among same-position slate players — already computed in the sidecar), value,
   sparkline; a one-line scale key.
5. **Correlated** — Phase 0's structured `Corr Same`/`Corr Opp` rows grouped Same team /
   Opponent; each row: `leg_label(...)` + **EV lift** from `ev_lift(...)` (green when positive;
   positive-lift rows only, per spec) + the existing View / +slip buttons.

- [ ] Session 1: header + context strip + History/Model variants (chart pins first: layer spec,
  app-line-unchanged, count-vs-continuous routing on `Dist`).
- [ ] Session 2: Comps/Other restyle + Correlated EV lift + the `deep_dive_tabs.py` split
  (refactoring-specialist will police the split's cohesion — tab renderers move whole, no
  forwarders left behind).
- [ ] Gates + commit per session: `feat(p8-c): details header + history/model tabs`,
  `feat(p8-c): details comps/other/correlated + ev lift`

---

### Task C3: The glyph set + Tonight unification

**Files:**
- Create: `src/sportstradamus/dashboard/components/glyphs.py`
- Modify: `src/sportstradamus/dashboard/surfaces/tonight.py`
- Test: `tests/golden/test_glyphs.py`

Five 60×60 inline-SVG glyphs, one construction rule (mockup: "strokes build the form, a glowing
radial core marks the focus"): Comet (shootout), Supernova (blowout), Scales (coinflip), Lone
star (even), Hourglass (grind). **Copy each `<svg viewBox="0 0 60 60">…</svg>` block verbatim
from `docs/mockups/p8-glyphs.html`** into string constants; namespace the gradient `id`s per
glyph (they collide if two glyphs mount on one page — `id="gc"` → `id="glyph-comet-core"` etc.).

```python
_GLYPHS = {"shootout": _COMET, "blowout": _SUPERNOVA, "coinflip": _SCALES,
           "even": _LONE_STAR, "grind": _HOURGLASS}

def game_shape_glyph(shape: str, *, size: int = 40) -> str:
    """Inline SVG for a game shape; unknown shapes fall back to the lone star."""
    svg = _GLYPHS.get(shape, _LONE_STAR)
    return svg.replace('viewBox="0 0 60 60"', f'width="{size}" height="{size}" viewBox="0 0 60 60"', 1)
```

Tonight: card glyph + legend swap to `game_shape_glyph` (the comet unification — the old
filled-streak polygon dies); shapes come off `current_game_context.shape` via the existing
context loader. These are bespoke celestial glyphs — inline SVG is the sanctioned exception to
Material-icons-only (spec §2).

- [ ] Golden first (five shapes resolve, unknown → lone star, gradient ids unique per glyph) →
  build → Tonight swap → gates → commit `feat(p8-c): unified game-shape glyph set; tonight comet`

---

### Task C4: Games hero + constellation team colors

**Files:**
- Modify: `src/sportstradamus/dashboard/surfaces/games.py`
- Modify: `src/sportstradamus/dashboard/components/constellation.py` (line ~42)
- Test: `tests/golden/test_constellation.py` (repoint)

- [ ] **Hero** (`p8-games.html`): the game banner becomes a nebula card (DESIGN §3 nebula rules:
  blue-family radial stops + gold ≤12%, hero cards only) — matchup first (`{away} @ {home}` from
  `narrative.home_away`), shape glyph (C3), Cormorant-italic gold prophecy subline
  (`.celestial-headline`; text from the story headline the surface already selects), Total /
  Spread / Shape row in Plex.
- [ ] **Team colors**: `_TEAM_PALETTE = ("#2E6BE6", "#E69F00")` dies; `_layout`/node fill reads
  `theme.team_colors(league, team)` (A5). The grammar is FIXED (DESIGN §4a): fill = team, size ∝
  edge, in-slip = full color/alpha, candidate = desaturated+dimmed (`_desaturate` now operates on
  the team primary), edges gold. Team fills are never gold — A5's golden already pins the JSON;
  add a figure-level pin that no node marker color equals `theme.GOLD`.
- [ ] Update `test_constellation` color-source pins (two-team fixture → two distinct team
  primaries; unknown team → gray fallback).
- [ ] Gates + commit `feat(p8-c): games nebula hero; constellation wears real team colors`

---

### Task C5: The astrolabe component

**Files:**
- Create: `src/sportstradamus/dashboard/components/astrolabe_component/__init__.py`
- Create: `src/sportstradamus/dashboard/components/astrolabe_component/build/index.html`
- Create: `src/sportstradamus/dashboard/components/astrolabe_component/build/main.js`
- Modify: `src/sportstradamus/dashboard/components/slip_builder.py` (mount where the flat metric
  row renders today)
- Test: payload golden (C1 already pins it); JS is manual-verify (lane precedent)

Build **to the mockup exactly** (`p8-constellation-lab.html` rev 5 — its CSS/JS is the
prototype; port, don't reinvent):

- Structure: Legs + Payout pinned left; **win orbital** r≈112 with two dots (blue = correlated →
  thread to the core; gray = independent) and the **lift arc** between them (green when
  corr > indep, red otherwise, gold-shimmer gradient — `bandGreen`/`bandRed` gradients in the
  mockup); **EV orbital** r≈74 running the opposite angular direction; **Kelly core gem** fed by
  threads from the EV dot and the correlated-win dot; bezel/degree ticks/crown reticle/twinkle
  are decorative engraving only.
- Mapping: value ∈ [0 → crown] → [orbital bottom → orbital top]; past the crown the bead pins at
  12 o'clock and the orbital glows (`.ovf` in the mockup).
- **Animates only on select/deselect**: the component diffs `payload.nonce`; a changed nonce
  toggles the `.sel`-style class so the CSS transitions fire (all transition CSS exists in the
  mockup at lines ~36-54); `@media (prefers-reduced-motion: reduce)` kills transitions.
- Readouts: exactly four rows — Win (correlated big, independent small beneath), Lift, EV,
  Kelly — numbers only, JS-set in lockstep (the mockup's `set()` function is the model).
- Component protocol: copy `constellation_component/build/index.html`'s postMessage bootstrap
  (setComponentReady / setFrameHeight / RENDER event carrying the payload) — same committed-
  static-files pattern, no npm, no CDN dependency beyond what constellation already uses.
- Python side: `render_astrolabe(payload, *, key)` mirroring `render_constellation`'s
  `components.declare_component` wiring; `slip_builder` replaces the metric row with it and keeps
  a `payout table unverified` caption when `payload["payout_approximate"]` (Sleeper scar).

- [ ] Port HTML/CSS/SVG from the mockup → wire protocol → mount → manual verify (add/remove a
  star: dials sweep, lift arc flips color, crown overflow pins + glows, reduced-motion static)
  → gates → commit `feat(p8-c): the astrolabe — animated slip readout component`

---

### Task C6: The two lenses

**Files:**
- Modify: `src/sportstradamus/dashboard/components/constellation.py`
- Modify: `src/sportstradamus/dashboard/surfaces/games.py` (lens toggles)
- Modify: `src/sportstradamus/dashboard/components/slip_builder.py`
- Modify: `src/sportstradamus/dashboard/components/satellite_picker.py` (pure `satellite_groups`
  survives as the wider-lens data source; the expander/chip renders retire)
- Test: `tests/golden/test_constellation.py` (+ lens figure pins)

These replace the "Add a leg from another game" / "Add a leg the model doesn't like" expanders
(`p8-games-lenses.html`). Two independent toggles above the map, active = gold (§3.1); both are
**modes of the existing Plotly figure**, not new components — taps ride the established click
protocol (`_apply_constellation_action` / `_apply_satellite_action`).

- [ ] **Look deeper**: `constellation_figure(..., deep_pool=...)` adds this game's Kelly ≤ 0
  legs as dim, cool, **unconnected** background stars (no edges, no layout influence — append
  after `_layout` on jittered perimeter positions; desaturated toward `theme.GRAY`, alpha ~0.35,
  below the lit constellation's z-order). Clicking one adds the leg (it "rides along");
  `render_disliked_legs` chips retire.
- [ ] **Look wider**: zoom-out mode — the focus constellation scales toward the center (multiply
  node coords by ~0.55) and other games' best legs (`satellite_groups`, already platform-aware
  and capped per game) orbit the edge grouped by matchup, labeled with their game key. Tapping
  one pulls it in as a satellite leg (existing `_apply_satellite_action` add path); the
  satellite expander retires.
- [ ] Lens state: two `st.session_state` bools rendered as gold-active toggle buttons in
  `games.py` above the map (mockup `.lens.on` styling via `key=`-scoped CSS).
- [ ] Figure pins: deeper stars unconnected + dimmed; wider mode scales focus + groups
  satellites; lenses off = today's figure byte-stable (regression).
- [ ] Gates + commit `feat(p8-c): look-deeper / look-wider lenses on the star map`

---

### Task C7: Receipts nebula verdict hero (after B7 — same file)

**Files:**
- Modify: `src/sportstradamus/dashboard/surfaces/receipts.py`
- Test: hero-parity pin update

The page's **one** nebula element (`p8-receipts.html`): verdict card — nebula wash (DESIGN §3
rules), gold ROI hero number (gold as the oracle's verdict accent on a hero card, not a data
mark in a chart), the cumulative-units area chart upgraded with a **real month/unit axis** and
the **worst-drawdown marker** (annotate the `worst_month` result on the curve, red label). "The
scars" skeptic tiles stay directly beneath (they're B7/P7 furniture — the hero must not hide
them).

- [ ] Pin first (hero parity vs `tailed_record` totals stays green; drawdown annotation present)
  → build → gates → commit `feat(p8-c): receipts nebula verdict hero`

---

## Exit criteria (whole plan)

- Astrolabe, lenses, glyphs, Details tabs match their mockups in a live walkthrough (record a
  manual-verify note per component — no Streamlit runtime in CI; lane precedent).
- `ev_lift`/`astrolabe_payload` goldens green; constellation figure pins green (team colors,
  lens modes, no-gold-node invariant).
- Three gates green; refactoring-specialist on every touched `.py`.
- Follow-ups explicitly NOT built (spec §6): constellation shapes, art licensing audit, forward
  ledger, ρ overlay — confirm none crept in.
