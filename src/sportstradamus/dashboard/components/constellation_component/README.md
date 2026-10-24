# constellation_component

A bidirectional Streamlit component for the slip-editor constellation. It exists because
`st.plotly_chart` exposes no hover event and `st.components.v1.html` can't return a value, but
the constellation answers the pointer **client-side** — faint-on-hover edges, a rich hover card,
a star that lights the moment it's clicked, animated lenses — *and* reports clicks and
`Full detail` to Python.

## No build step

`build/` is **hand-authored static files**, committed as-is — there is no npm / node / bundler
in this repo or on the production box:

- `build/index.html` — loads plotly.js + IBM Plex from CDN, holds the stacked map layers and
  the hover-card markup with token-mirrored CSS, and pulls in `main.js`.
- `build/main.js` — vanilla ES6. Speaks the Streamlit `postMessage` protocol directly
  (`streamlit:componentReady` / `streamlit:render` / `streamlit:setComponentValue` /
  `streamlit:setFrameHeight`), draws the layers, and handles hover, clicks and lens motion.

To change behavior, edit those files and reload the dashboard — there is nothing to compile,
and no JS test runner either: `tests/golden/test_constellation_component.py` only catches calls
to functions that don't exist, so check everything else live in a browser. `__init__.py`
declares the component against `build/` and exposes
`render_constellation(fig, *, key, ack, sparks, moves, shots, bans, on_change, mobile=False)`.

## Contract

### Figure

- Python passes the plotly figure as `figure_json` (`fig.to_json()`). The star traces —
  `candidate`, `active`, `deep` and `wider` — carry `customdata`
  `[key, player, market, bet, line, win, boost, kelly, in_slip]`. All but `wider` also carry a
  per-point `meta` list: the colour each star wears while a click flips it (the team colour for
  `candidate` and `deep`, the desaturated one for `active`). A star the platform won't pair with
  the slip wears an × from an overlay trace named `<host>_banned`, which has no `customdata`.
- `layout.meta` (absent from an empty figure) is `{edges, edge_base, dim_alpha, focus_scale}`.
  Each edge is `{a, b, x, y, width, dash, opacity, lit, lens}`: its stars' keys, its data
  coordinates, its stroke width, `"dot"` or `"solid"`, the server's opacity, the opacity when
  both stars are in the slip, and whether the "look deeper" lens owns it. List order is draw
  order.

### Layers

`main.js` draws the figure as three stacked layers:

1. **Engraving** — a `staticPlot` graph of the `decoration` traces, `layout.images` and
   `layout.shapes`. It re-plots only when those, the height or the axis ranges change.
2. **Edges** — an SVG `<line>` per edge record, the lens ties in their own group beneath.
3. **Stars** — the interactive graph of every other trace, with `layout.annotations`. Its
   trace names are unique, and `main.js` sets `uid = name` on each (plotly.py strips `uid`
   from `to_json`) so a trace keeps its `<g class="trace<uid>">` across reacts.

Both graphs share the height, margins and axis ranges. Edges, the click resolver, the desktop
card and the wider glide read plotly's private axis maths (`_fullLayout.xaxis._offset +
xaxis.l2p(x)`), and the glide also reads plotly's SVG (each marker's `transform`, each
caption's bound datum). Both hold only because `index.html` pins plotly.js at 2.27.0: re-check
edge alignment, clicks and the wider glide live before bumping it.

### Intents and ack

- A click paints its star and ties at once and records an intent. The component then sends
  `{seq, lit, detail}`. `seq` rises on every send (`max(seq + 1, Date.now())`, so it keeps
  rising across an iframe remount); `lit` maps every unacknowledged intent's star key to lit or
  not; `detail` is the star whose **Full detail** was pressed, else null.
- Python applies the value idempotently and passes the last `seq` it applied back as `ack` (0
  before any click). Each render drops the intents at or below `ack` and repaints the rest, so a
  render that predates a click never flickers its star, and a value Streamlit drops lands with
  the next one.
- Paint rule: a star is lit by its pending intent, else by `in_slip`. A star whose intent
  contradicts `in_slip` wears its `meta` colour at opacity 1 (`candidate`, `deep`), at
  `dim_alpha` (`active`), or hides until Python redraws it (`wider`). An edge touching an intent
  shows `lit` when both its stars are lit, else `edge_base`; every other edge keeps its
  `opacity`. Hover lifts a star's edges to at least 0.18.
- A click finds its star from its own coordinates, by plotly's closest-point rule (within
  20 px of a marker's edge), not from `plotly_click`, whose hover data plotly refreshes at most
  once per 50 ms, so a quick second click named the first star again. Presses never reach
  plotly's drag handler (the map never drags), which adds a click of its own to every press.

### Card carrier and the other args

- `sparks`, `moves` and `shots` are not component args. `render_constellation` writes them into
  a hidden `st.html` carrier as `window.__cstCards` (its docstring says why), and `main.js`
  reads `window.parent.__cstCards` when a card opens. `sparks` and `moves` map a star's key to
  its card's last-five and line-movement SVG; `shots` maps a player's name to a headshot data
  URI. A key with no `sparks` entry keeps the scar, one with no `moves` entry draws no row, and
  a player with no `shots` entry gets the initials disc.
- `bans` maps a star's key to its "won't pair" row text; no entry, no row.
- `mobile` switches to the touch flow: the first tap docks the card and previews the star's
  edges, a second tap on that star or the card's button toggles it, a tap on empty sky
  dismisses the card, and the frame grows so the card clears the map. On desktop the card sits
  beside its star, never over it, and a click on a star hides it. A fresh card lets presses
  through to the stars beneath it; it takes the pointer, for hover-to-stay and **Full detail**,
  only once the pointer has rested on its star for 350 ms.

### Motion and height

- A lens change animates with WAAPI opacity and CSS transforms, never `Plotly.restyle`:
  "look deeper" fades its stars and ties in, and out before the redraw; "look wider" fades the
  other games' sky and glides the map between its full and receded poses (`focus_scale`).
  Python scales the engraving by `focus_scale` but lays the stars out afresh at their own
  sizes, so the engraving recedes as one piece while each star, caption, × and tie end travels
  on its own, translate only, from where the old figure drew it. A render that arrives while
  lens motion plays waits for it to settle, and only the latest such render is drawn. A click
  never animates, and `prefers-reduced-motion` turns all motion off.
- A render whose `figure_json`, `ack`, `mobile` and `bans` all equal the last render's does
  nothing: Streamlit re-sends unchanged args after a frame-height change, and drawing them again
  would redraw the map for nothing and close a docked card.
- Every other render first posts a frame height of `layout.height` + 8, plus the phone's card
  pads, and only ever grows it there; the exact height follows once the last render's motion
  settles.
