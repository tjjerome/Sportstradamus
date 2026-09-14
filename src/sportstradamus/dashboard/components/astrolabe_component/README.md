# astrolabe_component

A read-only Streamlit component for the slip builders' astrolabe — three orbiting dials
(win/EV/Kelly) plus the lift arc between the two win dots, which sweep toward each new price as
legs come and go and hold still when a rerun recomputes the same slip. It exists because
Streamlit's own `st.metric` re-renders flat on every rerun, with no way to ease from one price to
the next.

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

Python passes one of two plain-JSON dicts straight through as `payload` — unlike
`render_constellation`'s `figure_json` there is nothing to pre-serialize; the
component-argument marshalling handles the encoding.

- **Priced** (2+ legs): `slip_engine.astrolabe_payload(score)`'s dict — legs, play type,
  payout, `win_corr`, `win_indep`, `ev`, `kelly` and the `crowns` scale. `payout_approximate`
  rides along undrawn.
- **Rest** (0–1 legs): `{"legs": n}`. Every dial number eases to 0 — both win dots and the EV
  dot at their value-0 poses, a zero-length lift arc, the gem at its minimum, both crown glows
  off. Legs shows the count; the other readouts show "—" with no sign colour.

`main.js` tells the two apart by `crowns`, which only a priced payload carries.

## Design math

- **Win dots** (`win_corr` blue, `win_indep` gray), crown `crowns.win`: `angle = 180 * (1 -
  clamp(value, 0, crown) / crown)` — 0 at the crown (12 o'clock, the dot's un-rotated rest
  position), 180° at value=0 (rotated to the bottom). Reverse-engineered against both mockup
  demo snapshots (weak/strong presets) — matches to rounding.
- **Lift arc**: drawn as a `pathLength="360"` circle (`stroke-dasharray` in degrees).
  `arc_length = |angle_corr - angle_indep|`; `arc_rotate = min(angle_corr, angle_indep) - 90`
  — the `-90` reconciles the circle path's own dash-start reference (3 o'clock) against the
  dot angle's convention (0° = 12 o'clock). Green (`bandGreen`) when `win_corr >= win_indep`,
  red (`bandRed`) below. Matches both mockup snapshots exactly.
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
  Their opacity is fixed (not JS-driven) at the mockup's "selected" pose for every slip, the
  rest pose included: no payload number drives thread/halo brightness.

## Animation

One `requestAnimationFrame` tween drives every continuous part. `main.js` keeps the numbers on
screen (`win_corr`, `win_indep`, `ev`, `kelly`, `payout`) and, on each render, eases them from
wherever they are toward the payload's values (all zero at rest) over `SWEEP_MS`, on the power
ease-out `EASE_POWER` names. Every frame redraws the dot angles, the lift arc, the gem and the
readouts from the same in-between numbers, so the arc stays on the dots it spans and the
readouts tick with them.

- The first render snaps: there is no earlier pose to sweep from.
- A render landing mid-sweep restarts the ease from the numbers on screen, so the dials change
  course without a jump. A render with the same targets (an incidental rerun) starts nothing
  and leaves a running sweep alone.
- The lift arc's gradient and the EV bead's colour follow the sign of the drawn lift and EV.
  The lift changes sign only where the arc has shrunk to nothing, so the gradient swap needs no
  crossfade.
- `main.js` writes numbers only (`--angle`, `--scale`, `data-sign`); `index.html` turns them
  into `rotate()`, `scale()`, the gradients and the sign colours. That keeps CSS function
  notation out of `main.js`'s strings, where the golden call scan in
  `tests/golden/test_constellation_component.py` would read `rotate(` as an undefined call.
- Two discrete flips keep CSS transitions: a crown glow's opacity and the EV bead's fill.
  `main.js` sets `.no-anim` for the first frame so the first pose paints without easing them
  in.
- Under `prefers-reduced-motion: reduce` the tween lands on its targets in one frame, and CSS
  transitions are off.
