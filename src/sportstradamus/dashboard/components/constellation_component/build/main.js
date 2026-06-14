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
  let plotted = false;
  let baseOpacity = []; // per-trace baseline opacity (server-set: active-incident vs hidden)
  let edgeEndpoints = []; // per-trace [a, b] for edge traces, else null
  let previewed = []; // edge trace indices currently faint-previewed
  let hideTimer = null;
  let nonce = 0;
  let activeKey = null; // key whose card is showing

  function render(args) {
    hideCard();
    const fig = JSON.parse(args.figure_json);
    const config = { displayModeBar: false, scrollZoom: false, responsive: true };
    Plotly.react(chartDiv, fig.data, fig.layout, config);
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
    }
    const height = (fig.layout && fig.layout.height ? fig.layout.height : 380) + FRAME_PAD;
    setFrameHeight(height);
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
