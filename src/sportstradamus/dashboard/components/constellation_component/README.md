# constellation_component

A bidirectional Streamlit component for the slip-editor constellation. It exists because
`st.plotly_chart` exposes no hover event and `st.components.v1.html` can't return a value, but
the constellation needs **client-side** faint-on-hover edges + a rich hover card (no server
rerun) *and* a click / `Full detail` callback to Python.

## No build step

`build/` is **hand-authored static files**, committed as-is — there is no npm / node / bundler
in this repo or on the production box:

- `build/index.html` — loads plotly.js + IBM Plex from CDN, holds the hover-card markup +
  token-mirrored CSS, and pulls in `main.js`.
- `build/main.js` — vanilla ES6. Speaks the Streamlit `postMessage` protocol directly
  (`streamlit:componentReady` / `streamlit:render` / `streamlit:setComponentValue` /
  `streamlit:setFrameHeight`), renders the figure JSON with `Plotly.react`, and wires hover
  (edge restyle + card) and click / detail callbacks.

To change behavior, edit those files and reload the dashboard — there is nothing to compile.
`__init__.py` declares the component against `build/` and exposes `render_constellation(fig, *, key)`.

## Contract

- Python passes the plotly figure as `figure_json` (`fig.to_json()`); each node's `customdata`
  is `[key, player, market, bet, line, win, boost, kelly]` and each edge's `meta` is
  `[endpoint_a, endpoint_b]`. Edges come under two names — `edge` for the permanent web
  and `deep_edge` for a tie the "look deeper" lens brings in — and both hover-preview;
  only `deep_edge` fades in and out with the lens.
- The component returns `{action, key, nonce}` — `action` is `"click"` (toggle the leg) or
  `"detail"` (open the offer dialog); `nonce` increments per emit so a repeat click is a fresh
  value. The caller dedups by `nonce`.
