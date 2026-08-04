# Dashboard Mobile — Phase M design

> Status: APPROVED (owner, 2026-07-16). Lane: dashboard-ux, Phase M — follows Phase R,
> precedes the parked Phases D ∥ E. Branch: `feature/dashboard-ux`.

## 1. Problem and scope

DESIGN.md names the primary use context — "used daily, often on mobile, under time
pressure (lineups lock)" — but the dashboard is desktop-first throughout: hardcoded
`layout="wide"`, a sidebar slip rail that mobile hides behind a hamburger, a 10-column
AG-Grid Board, and a hover-driven constellation editor. The only media queries in the
codebase are `prefers-reduced-motion`. Nothing else adapts.

**Scope (owner-locked):** the core money loop must work well on a phone — Tonight →
Games/constellation → slip build → stakes — and the Board must be readable. Receipts and
the Model Lab stay desktop-first: reachable and stacked, no dedicated mobile treatment.

**Non-goals:** no pipeline, `persist.py`, or snapshot-schema changes; no PWA/manifest
work (LAN/tailscale access, browser is fine); no tablet-specific layout (iPad reports a
desktop UA and takes the desktop path); no desktop visual changes of any kind.

## 2. Architecture — UA branch plus a CSS layer

Chosen over pure-CSS (cannot swap the AG-Grid for cards, cannot change constellation
hover semantics inside its iframe, bottom-sheet interaction needs state) and over a
separate mobile app (a parallel nav/state/CSS structure to keep in sync forever —
against CLAUDE.md consolidation rules).

### 2.1 Detection — `dashboard/viewport.py` (new)

`is_mobile() -> bool` resolves once per session, cached in `st.session_state`:

1. `st.session_state["_force_mobile"]` — test hook (AppTest cannot fake headers).
2. `?m=1` / `?m=0` query param — manual override, persisted into session state.
3. `st.context.headers` User-Agent — `"Mobi"` substring, the standard heuristic every
   phone browser satisfies. Chrome devtools device emulation sends a mobile UA, so
   emulation exercises the real branch.

No resize handling: a phone never becomes a desktop mid-session, and a desktop window
dragged narrow stays on the desktop path by design. The UA is available at script
start, so `app.py` can branch navigation before any page renders.

`theme.py` gains one constant, `MOBILE_MAX_PX = 767`, used only by the CSS media
queries; the Python branch never measures pixels.

### 2.2 Global chrome (`app.py`)

- `layout="wide"` stays for both paths (wide on a narrow viewport is full-width).
- Mobile gets `st.navigation(..., position="top")` — pages visible in a top bar rather
  than behind the hamburger. Desktop keeps the sidebar nav untouched.
- Lab pages stay in the nav with a "best at a desk" caption on mobile.
- The 6-option sport segmented control is kept; live-check wrap at 390 px. Fallback if
  it wraps badly: a compact `st.selectbox` on the mobile path only.
- The sidebar (locked shelf: saved slips + bankroll) remains available via hamburger on
  mobile — saved slips are not the time-critical path, and bankroll is also editable in
  the expanded slip dock.

### 2.3 CSS layer — one `@media (max-width: 767px)` block in `theme.APP_CSS`

Style-only reshaping; FIXED tokens untouched and no new colors/fonts/radii:

- `page_hero` kicker/headline sizes step down; custom gaps tighten within the
  4/8/12/16/24 scale.
- Tonight card grid → single column; the shape legend wraps to two columns.
- Games legs panel (2-column gold tablet) → single column.
- Starfield stays as-is: body-layer, already occluded behind content; the 12 twinkle
  animations are negligible on battery.
- `st.dialog` (deep-dive) is already near-fullscreen on narrow viewports natively; its
  five tabs scroll horizontally natively. CSS polish only if the live check demands it.
- Body `padding-bottom` on mobile so the fixed slip dock never covers the last row.

## 3. Slip dock — `dashboard/components/slip_dock.py` (new)

The mobile replacement for the sidebar rail's ambient visibility. Rendered from
`app.py` after `pg.run()` when `is_mobile()` and the active slip is non-empty;
`position: fixed` detaches it from document flow, so render order does not affect
layout.

- **Collapsed** — a fixed bottom bar (~56 px): `N legs · 6.00x · EV +12% · [expand]`.
  Implemented as an `st.container(key="slip_dock")` with key-scoped CSS
  (`position: fixed; bottom: 0`) — the sanctioned structural-CSS lane (DESIGN §3).
  Surface tokens, gold hairline top border, 4 px radius, Plex Mono numerals.
- **Expanded** — a session-state boolean grows the bar into a sheet (max ~70 vh,
  scrollable): leg rows (`leg_label` + remove buttons), stake / payout / EV / kelly,
  and the lock-in button. Reuses `slip_state` and the existing payout/kelly helpers —
  no duplicated math, same slip state the desktop rail reads. Money stays `Decimal`.

## 4. Surfaces

### 4.1 Tonight — CSS only

Cards already exist; the §2.3 layer makes them single-column. Tap card → View game
uses the existing session-state handoff. No render branch.

### 4.2 Games — constellation touch mode

Pickers, hero, story preloader, and lens toggles stack natively. The constellation
changes in two halves:

- **Figure** (`components/constellation.py`): a `mobile=True` parameter raises the
  star-size floor (14 → 22 px tap targets) and label font. The §4a grammar is
  untouched — star=leg, fill=team, size=edge, brightness=in-slip; the floor is a scale
  shift and edge still orders sizes.
- **Component JS** (`constellation_component/build/main.js`): a `mobile` prop, with
  `pointer: coarse` detection as belt-and-braces. Touch flow: tap a star → a docked
  leg card (full-width at the bottom of the canvas, thumb zone) with Add/Remove and
  Full-detail buttons; tap the same star again → toggle into the slip; tap empty sky →
  dismiss. Hover handlers are inert on touch. The desktop path stays byte-identical.
  The component has no build toolchain (hand-authored ES6), so a live-browser check is
  mandatory — the §4b lesson: passing goldens do not prove the iframe renders.

### 4.3 Board — the one real render branch

- `is_mobile()` swaps the 10-column AG-Grid for a card list — new
  `components/offer_cards.py`. Card: Player · market display name · line + bet arrow ·
  Win % · Model Edge · platform/league chip; Plex Mono numerals; obsidian-family
  styling on tokens. Shows 30, a "Show more" button pages. Two actions per card:
  Detail (opens the deep-dive dialog) and Add to slip.
- Filters (three multiselects + player search + range sliders) collapse into an
  `st.expander("Filters")` on mobile, collapsed by default; the lens and side
  segmented controls stay visible on top.
- The desktop grid path is untouched.

### 4.4 Receipts / Lab — no branch

Native stacking plus the §2.2 caption. Out of money-loop scope.

## 5. Phase D compatibility

Phase D (loose constellation shapes) is parked but must land cleanly on top of this
work. The seam:

- D changes **star positions** (template vertex assignment, topology classifier, slate
  assigner, decoration traces). Mobile touches **marker sizes, label fonts, and JS
  event flow** — never positions. Orthogonal by construction; both keep §4a FIXED.
- The JS tap flow keys off `customdata` leg keys, which D preserves (star=leg after
  supernode explosion). D's decoration layer arrives as extra figure traces from
  Python; tap/hover handlers are unaffected.
- **Recorded interaction:** D's player-supernode knots pack a player's stars tightly
  around a vertex — tap ambiguity on a 390 px canvas. Two guards: (1) D's explode
  radius must read the same `mobile` flag and respect a minimum tap-separation
  constant; (2) the select-then-confirm tap flow means a mis-tap costs one tap, never
  a slip mutation. D's plan must carry guard (1) explicitly.
- Ordering: M before D — D implements against the `mobile` parameter rather than
  retrofitting it.

## 6. Testing and acceptance

- **Unit/golden:** `is_mobile()` precedence tests (override → query param → UA); slip
  dock AppTest smoke via `_force_mobile` (renders, expands, remove-leg works); Board
  card-list AppTest (cards render, Add fires, paging works); the design-token golden
  extends to pin the media block and dock CSS on tokens; all existing goldens stay
  green (tests default to the desktop path).
- **Live verdicts** (the Phase R process fix, kept): every task ends with a recorded
  browser check — Chrome devtools device emulation for the mobile branch during
  development, plus at least one real-phone pass over tailscale before the lane
  closes.
- **Gates:** ruff + `tests/golden/` + `pytest -m integration -n0` +
  refactoring-specialist on every touched `.py` before any push. The design-lint hook
  covers CSS edits.
- **Acceptance:** on a 390 px viewport — Tonight cards single-column and tappable
  through to Games; constellation stars tappable with the docked-card confirm flow;
  slip dock visible on every surface with a non-empty slip, expandable, lock-in works;
  Board cards readable with working Detail and Add; no horizontal page scroll on any
  money-loop surface; desktop rendering pixel-unchanged.

## 7. Rollout

Phase M on the dashboard-ux lane: implementation plan under `docs/superpowers/plans/`,
a §6 stage-plan entry and ledger line in `docs/handoffs/dashboard-ux.md`, and a short
DESIGN.md §Mobile note (breakpoint, dock pattern, touch-constellation grammar
addendum). Estimated 3–4 sessions: dock + viewport; constellation touch; Board cards +
CSS layer; polish + phone pass.
