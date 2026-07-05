# astrolabe_component

A read-only Streamlit component for the slip builders' astrolabe — three orbiting dials
(win/EV/Kelly) plus the lift arc between the two win dots, animated only when the slip's
leg-set actually changes. It exists because CSS transitions need a real DOM write to fire,
and Streamlit's own `st.metric` re-renders flat on every rerun with no way to distinguish
"the user added a leg" from "an unrelated rerun recomputed the same slip."

## No build step

`build/` is **hand-authored static files**, committed as-is — same convention as
`constellation_component`:

- `build/index.html` — the astrolabe SVG (bezel, orbitals, win/EV dots, lift arc, Kelly
  gem) ported from `docs/mockups/p8-constellation-lab.html` rev 5, token-mirrored CSS, and
  the four readout rows (Win/Lift/EV/Kelly).
- `build/main.js` — vanilla ES6. Speaks the Streamlit `postMessage` protocol
  (`streamlit:componentReady` / `streamlit:render` / `streamlit:setFrameHeight`), computes
  each dial's angle/scale/opacity off the payload, and writes the four readout numbers.
  No `streamlit:setComponentValue` — this component has no click/detail callback; selection
  happens on the constellation's own stars.

To change behavior, edit those files and reload the dashboard — there is nothing to compile.
`__init__.py` declares the component against `build/` and exposes
`render_astrolabe(payload, *, key)`.

## Contract

Python passes `slip_engine.astrolabe_payload(...)`'s dict straight through as `payload` — it
is already plain JSON-primitive values, so (unlike `render_constellation`'s `figure_json`)
there is nothing to pre-serialize; the component-argument marshalling handles the encoding.

## Design math

- **Win dots** (`win_corr` blue, `win_indep` gray), crown `crowns.win`: `angle = 180 * (1 -
  clamp(value, 0, crown) / crown)` — 0 at the crown (12 o'clock, the dot's un-rotated rest
  position), 180° at value=0 (rotated to the bottom). Reverse-engineered against both mockup
  demo snapshots (weak/strong presets) — matches to rounding.
- **Lift arc**: drawn as a `pathLength="360"` circle (`stroke-dasharray` in degrees).
  `arc_length = |angle_corr - angle_indep|`; `arc_rotate = min(angle_corr, angle_indep) - 90`
  — the `-90` reconciles the circle path's own dash-start reference (3 o'clock) against the
  dot angle's convention (0° = 12 o'clock). Green (`bandGreen`) when `win_corr > win_indep`,
  red (`bandRed`) otherwise. Matches both mockup snapshots exactly.
- **EV dot**, crown `crowns.ev`: `angle = -180 * (1 - clamp(value, 0, crown) / crown)` — same
  shape as the win dots, negated per the spec's "opposite angular direction" wording. EV's
  domain is stated as `[0, crown]`; a negative EV clamps to the value=0 pose (bottom) rather
  than extending the domain below zero. The mockup's own demo EV number doesn't fit any
  formula tried against it exactly (its weak-state pose is illustrative, not a literal plot —
  it uses a negative value outside the stated domain); the crown-boundary behavior (0° at or
  past crown) does match regardless of sign convention.
- **Kelly gem** (not a dot — the fixed centre gem, scale + opacity): linear interpolation
  between the mockup's weak (`scale(.66)`/`opacity:.55`) and strong (`scale(1.08)`/`opacity:1`)
  poses over `t = clamp(value, 0, crown) / crown`. The mockup's own demo Kelly number
  (1.2% of a 3% crown, t≈0.4) renders at the fully-lit pose in the demo, which doesn't fit
  this line at t=0.4 — read as the same illustrative-pose-not-formula situation as EV above,
  since the demo only shows two canned states, not a continuum.
- **Crown overflow** (`.ovf-win` / `.ovf-ev`): each orbital glows independently off its own
  value(s) reaching `t >= 1`, not a single shared flag — the mockup's demo only has one
  `.ovf` circle (on the EV orbital) because its two-state weak/strong toggle only needed one
  glow at a time; a live payload can have the win dial pinned while EV isn't (or vice versa).
- **Threads**: static quadratic-bezier paths living inside each dot's own rotating `<g
  class="grp ...">`, so they rotate with their dot for free — no separate thread-angle math.
  Their opacity is fixed (not JS-driven): the astrolabe only ever mounts over a real,
  already-in-progress slip, so it always reads as the mockup's "selected" pose for the
  thread/halo brightness the demo otherwise toggled on select — the task's four named
  JS-set readouts are Win/Lift/EV/Kelly, not thread/halo brightness.

## Animation

CSS `transition` is unconditional on the animated elements (`.grp`/`.band`/`.bandglow`/
`.gem`/etc.) — any inline style change fires it. `main.js` suppresses the sweep only on the
very first mount (a `.no-anim` class removed on the next animation frame, so the initial pose
paints instantly) and tracks `payload.nonce` in a closure purely to know when a mount is
"first" versus a later, real render — the CSS transition itself doesn't need the nonce
comparison to work correctly on subsequent renders, since an incidental rerun that recomputes
an identical payload just re-applies the same values (a no-op regardless of nonce).
`@media (prefers-reduced-motion: reduce)` kills all transitions.
