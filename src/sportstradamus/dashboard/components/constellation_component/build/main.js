/* Constellation component frontend — hand-authored, no build step.
 *
 * Speaks the Streamlit <-> iframe postMessage protocol directly (the same one
 * streamlit-component-lib emits), renders the figure JSON Python passes in with
 * plotly.js (loaded from CDN), and adds the two things st.plotly_chart can't:
 *   - faint-on-hover edge preview + a rich hover card, handled client-side (no rerun);
 *   - a click / "Full detail" callback to Python via setComponentValue.
 * The figure already bakes every color/edge/node; this only adds interaction.
 */
(function () {
  "use strict";

  // --- Minimal Streamlit bridge ------------------------------------------------
  const RENDER_EVENT = "streamlit:render";
  let renderCallback = null;

  function post(msg) {
    window.parent.postMessage(Object.assign({ isStreamlitMessage: true }, msg), "*");
  }
  function setFrameHeight(height) {
    post({ type: "streamlit:setFrameHeight", height: height });
  }
  function setComponentValue(value) {
    post({ type: "streamlit:setComponentValue", value: value, dataType: "json" });
  }

  window.addEventListener("message", function (event) {
    if (event.data && event.data.type === RENDER_EVENT && renderCallback) {
      renderCallback(event.data.args || {});
    }
  });

  // --- DOM + state -------------------------------------------------------------
  const chartDiv = document.getElementById("chart");
  const card = document.getElementById("card");
  const FRAME_PAD = 8; // headroom so top-row star labels aren't clipped
  const HOVER_FAINT = 0.18; // faint opacity for a hidden edge previewed on hover
  const HIDE_DELAY_MS = 220; // hover-intent: keep the card while the cursor travels to it
  // Lens toggles animate two different ways, because they change the map two different ways:
  //   - "Look deeper" only ADDS a "deep" trace of background stars — the lit map doesn't move, so
  //     fade in just that new trace and leave every existing star exactly where it is.
  //   - "Look wider" RESTRUCTURES the map: the focus constellation shrinks toward centre while the
  //     outer "wider"/"wider_labels" clusters appear. Fading only the new clusters would leave the
  //     shrink as an ugly instant snap, so settle the whole map in with one opacity fade instead.
  // A slip click adds no lens trace and never moves the focus, so it never animates. Respect
  // prefers-reduced-motion.
  const LENS_FADE_MS = 900; // "look deeper": new background stars materialize
  const WIDER_FADE_MS = 1000; // "look wider": whole-map settle — a touch slower, hides the reshape
  const LENS_TRACE_NAMES = { deep: true, wider: true, wider_labels: true };
  const REDUCED_MOTION =
    window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let plotted = false;
  let baseOpacity = []; // per-trace baseline opacity (server-set: active-incident vs hidden)
  let edgeEndpoints = []; // per-trace [a, b] for edge traces, else null
  let previewed = []; // edge trace indices currently faint-previewed
  let hideTimer = null;
  let nonce = 0;
  let activeKey = null; // key whose card is showing
  let prevLensNames = null; // {name: true} lens traces shown last render — a new one marks a reveal
  let renderSeq = 0; // bumped each render so an in-flight lens fade from a stale render bails out

  function render(args) {
    hideCard();
    renderSeq += 1;
    const fig = JSON.parse(args.figure_json);
    const config = { displayModeBar: false, scrollZoom: false, responsive: true };
    const curLensNames = lensNamesOf(fig.data);
    const freshLens = plotted && prevLensNames ? newLensTraces(fig.data, prevLensNames) : [];
    const widerReshaped =
      plotted && prevLensNames && !!prevLensNames.wider !== !!curLensNames.wider;
    Plotly.react(chartDiv, fig.data, fig.layout, config);
    prevLensNames = curLensNames;
    baseOpacity = fig.data.map(function (t) {
      return t.opacity == null ? 1 : t.opacity;
    });
    edgeEndpoints = fig.data.map(function (t) {
      return t.name === "edge" ? t.meta : null;
    });
    previewed = [];
    if (!plotted) {
      attachHandlers();
      plotted = true;
    } else if (widerReshaped) {
      fadeInWholeMap();
    } else if (freshLens.length) {
      fadeInLensTraces(freshLens);
    }
    const height = (fig.layout && fig.layout.height ? fig.layout.height : 380) + FRAME_PAD;
    setFrameHeight(height);
  }

  // The lens traces present in a figure, as a {name: true} set (deep / wider / wider_labels).
  function lensNamesOf(data) {
    const names = {};
    data.forEach(function (t) {
      if (LENS_TRACE_NAMES[t.name]) names[t.name] = true;
    });
    return names;
  }

  // Indices of lens traces present now but absent last render — the genuinely new stars to fade in.
  function newLensTraces(data, prev) {
    const idx = [];
    data.forEach(function (t, i) {
      if (LENS_TRACE_NAMES[t.name] && !prev[t.name]) idx.push(i);
    });
    return idx;
  }

  // Fade ONLY the freshly-revealed lens traces up from transparent to their designed opacity, so a
  // "look deeper" / "look wider" toggle reads as new stars materializing while every existing star
  // holds still. Ramps trace-level opacity (which multiplies each marker's own alpha) over
  // LENS_FADE_MS on requestAnimationFrame; the start-at-0 restyle runs in the same tick as
  // Plotly.react, so the browser never paints the full-opacity first frame. A newer render bumps
  // renderSeq and this fade bails, so a stale ramp can't fight the current figure's trace indices.
  function fadeInLensTraces(indices) {
    if (REDUCED_MOTION) return;
    const targets = indices.map(function (i) {
      return baseOpacity[i];
    });
    const seq = renderSeq;
    let start = null;
    Plotly.restyle(chartDiv, { opacity: indices.map(function () { return 0; }) }, indices);
    function step(now) {
      if (seq !== renderSeq) return; // a newer render superseded this fade
      if (start === null) start = now;
      const t = Math.min(1, (now - start) / LENS_FADE_MS);
      const eased = 1 - Math.pow(1 - t, 3); // easeOutCubic — quick lift, gentle settle
      Plotly.restyle(
        chartDiv,
        {
          opacity: targets.map(function (v) {
            return v * eased;
          }),
        },
        indices
      );
      if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  function attachHandlers() {
    chartDiv.on("plotly_click", function (data) {
      const pt = pointFrom(data);
      if (pt) emit("click", pt.customdata[0]);
    });
    chartDiv.on("plotly_hover", function (data) {
      const pt = pointFrom(data);
      if (!pt) return;
      previewEdges(pt.customdata[0]);
      showCard(pt, data.event);
    });
    chartDiv.on("plotly_unhover", function () {
      restoreEdges();
      scheduleHide();
    });
    card.addEventListener("mouseenter", function () {
      clearTimeout(hideTimer);
    });
    card.addEventListener("mouseleave", hideCard);
  }

  // Only node traces carry customdata; edges are hoverinfo="skip" so never reach here.
  function pointFrom(data) {
    if (!data || !data.points || !data.points.length) return null;
    const pt = data.points[0];
    return pt && pt.customdata ? pt : null;
  }

  function emit(action, key) {
    nonce += 1;
    setComponentValue({ action: action, key: key, nonce: nonce });
  }

  // --- Edge faint-on-hover -----------------------------------------------------
  function previewEdges(key) {
    restoreEdges();
    const idx = [];
    const op = [];
    edgeEndpoints.forEach(function (ends, i) {
      if (ends && (ends[0] === key || ends[1] === key)) {
        idx.push(i);
        op.push(Math.max(baseOpacity[i], HOVER_FAINT));
      }
    });
    if (idx.length) {
      Plotly.restyle(chartDiv, { opacity: op }, idx);
      previewed = idx;
    }
  }

  function restoreEdges() {
    if (!previewed.length) return;
    const op = previewed.map(function (i) {
      return baseOpacity[i];
    });
    Plotly.restyle(chartDiv, { opacity: op }, previewed);
    previewed = [];
  }

  // --- Hover card --------------------------------------------------------------
  function showCard(pt, mouseEvent) {
    clearTimeout(hideTimer);
    const cd = pt.customdata; // [key, player, market, bet, line, win, boost, kelly]
    activeKey = cd[0];
    card.innerHTML = cardHtml(cd);
    card.querySelector(".cst-btn").addEventListener("click", function () {
      emit("detail", activeKey);
    });
    card.classList.remove("cst-hidden");
    card.setAttribute("aria-hidden", "false");
    positionCard(mouseEvent);
  }

  function scheduleHide() {
    clearTimeout(hideTimer);
    hideTimer = setTimeout(hideCard, HIDE_DELAY_MS);
  }

  function hideCard() {
    card.classList.add("cst-hidden");
    card.setAttribute("aria-hidden", "true");
    activeKey = null;
  }

  function positionCard(mouseEvent) {
    let x = (mouseEvent ? mouseEvent.clientX : window.innerWidth / 2) + 14;
    let y = (mouseEvent ? mouseEvent.clientY : 40) + 14;
    const cw = card.offsetWidth;
    const ch = card.offsetHeight;
    x = Math.min(x, window.innerWidth - cw - 6);
    y = Math.min(y, window.innerHeight - ch - 6);
    card.style.left = Math.max(6, x) + "px";
    card.style.top = Math.max(6, y) + "px";
  }

  // --- Card content ------------------------------------------------------------
  function pct(v) {
    return Math.round((Number(v) || 0) * 100) + "%";
  }

  function initials(name) {
    return String(name)
      .split(/\s+/)
      .map(function (w) {
        return w[0] || "";
      })
      .join("")
      .slice(0, 2)
      .toUpperCase();
  }

  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  function cardHtml(cd) {
    const player = cd[1];
    const market = cd[2];
    const bet = cd[3];
    const line = cd[4];
    const win = cd[5];
    const boost = cd[6];
    const kelly = cd[7];
    return [
      '<div class="cst-head">',
      '<div class="cst-shot" title="Player headshot — coming soon">',
      esc(initials(player)),
      "</div>",
      '<div class="cst-id"><div class="cst-name">',
      esc(player),
      '</div><div class="cst-leg">',
      esc(market) + " · " + esc(bet) + " " + esc(line),
      "</div></div></div>",
      '<div class="cst-stats">Win ',
      pct(win),
      " · ",
      (Number(boost) || 1).toFixed(2),
      'x · <span class="cst-kelly">Kelly ',
      pct(kelly),
      "</span></div>",
      '<div class="cst-scar">Last 5 — coming soon</div>',
      '<button class="cst-btn" type="button">Full detail →</button>',
    ].join("");
  }

  renderCallback = render;
  post({ type: "streamlit:componentReady", apiVersion: 1 });
})();
