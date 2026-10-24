/* Astrolabe component frontend — hand-authored, no build step.
 *
 * Speaks the Streamlit <-> iframe postMessage protocol directly (the same one
 * streamlit-component-lib emits) and drives the three orbiting dials (win/EV/Kelly)
 * plus the lift arc off the plain `payload` dict Python passes in — no plotly, no
 * server callback (selection happens on the constellation's own stars; this is a
 * pure readout). Ported from docs/mockups/p8-constellation-lab.html rev 5.
 *
 * Motion is one requestAnimationFrame tween of the payload's numbers, not CSS
 * transitions: those restarted on every render, eased the lift arc apart from the
 * dots it spans, and could not tick the readouts. Every frame redraws the dials, arc,
 * gem and readouts from the same in-between numbers, and a render that lands
 * mid-sweep eases on from wherever the dials are. This file writes only numbers
 * (--angle, --scale, data-sign); index.html owns the rotate()/scale()/url()/var()
 * notation, which the golden call scan would misread as undefined calls in a string.
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

  window.addEventListener("message", function (event) {
    if (event.data && event.data.type === RENDER_EVENT && renderCallback) {
      renderCallback(event.data.args || {});
    }
  });

  // --- DOM ----------------------------------------------------------------------
  const astro = document.getElementById("astro");
  const FRAME_PAD = 8; // headroom to match the constellation component's own convention
  const FRAME_HEIGHT = 272 + FRAME_PAD; // the SVG's own fixed height (no dynamic layout here)
  const PLACEHOLDER = "—"; // index.html's own empty readout, held while the slip is unpriced

  const el = {
    legs: document.getElementById("ro-legs"),
    pay: document.getElementById("ro-pay"),
    win: document.getElementById("ro-win"),
    indep: document.getElementById("ro-indep"),
    lift: document.getElementById("ro-lift"),
    ev: document.getElementById("ro-ev"),
    kelly: document.getElementById("ro-kelly"),
    wincorr: astro.querySelector(".wincorr"),
    winindep: astro.querySelector(".winindep"),
    ev_g: astro.querySelector(".ev"),
    gem: astro.querySelector(".gem"),
    band: astro.querySelector(".band"),
    bandglow: astro.querySelector(".bandglow"),
    ovfWin: astro.querySelector(".ovf-win"),
    ovfEv: astro.querySelector(".ovf-ev"),
    evbead: astro.querySelectorAll(".evbead"),
  };

  // The mockup's CSS transitions took 0.9 s; the tween keeps that pace.
  const SWEEP_MS = 900;
  // Ease-out 1 - (1 - t)^4. It starts at slope 4, near the mockup's
  // cubic-bezier(0.2, 0.72, 0.15, 1) at 3.6. The closer-fitting power 5 starts at 5,
  // which jumps ~16deg in the first frame of a full 180deg sweep.
  const EASE_POWER = 4;
  const TWEENED = ["win_corr", "win_indep", "ev", "kelly", "payout"];
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  let priced = false; // false at rest (0-1 legs), where the readouts hold PLACEHOLDER
  let crowns = {}; // the last priced payload's, so a sweep home to rest keeps its scale
  let cur = null; // the numbers on screen; null until the first render
  let from = null;
  let to = null;
  let t0 = 0;
  let frame = 0; // the running sweep's requestAnimationFrame id; 0 when none runs

  function render(args) {
    setFrameHeight(FRAME_HEIGHT);
    const p = args.payload;
    priced = "crowns" in p; // below two legs the builder sends a bare {legs}
    if (priced) crowns = p.crowns;
    const target = {};
    TWEENED.forEach(function (name) {
      target[name] = priced ? p[name] : 0;
    });

    el.legs.textContent = priced ? p.legs + " · " + p.play_type : p.legs;
    if (!priced) {
      [el.pay, el.win, el.indep, el.lift, el.ev, el.kelly].forEach(function (node) {
        node.textContent = PLACEHOLDER;
      });
      el.lift.dataset.sign = "";
      el.ev.dataset.sign = "";
    }

    if (cur === null) {
      // Nothing to sweep in from: paint the first pose with the CSS transitions off,
      // forcing a style flush so the values land untransitioned, then restore them.
      astro.classList.add("no-anim");
      cur = to = target;
      draw();
      astro.offsetHeight;
      requestAnimationFrame(function () {
        astro.classList.remove("no-anim");
      });
      return;
    }
    const moved = TWEENED.some(function (name) {
      return target[name] !== to[name];
    });
    if (!moved) return;
    from = cur;
    to = target;
    t0 = performance.now();
    if (!frame) frame = requestAnimationFrame(step);
  }

  function step() {
    const t = reducedMotion.matches ? 1 : Math.min(1, (performance.now() - t0) / SWEEP_MS);
    // "Target minus the eased share still to go", so t = 1 lands exactly on the target.
    const left = (1 - t) ** EASE_POWER;
    cur = {};
    TWEENED.forEach(function (name) {
      cur[name] = to[name] - (to[name] - from[name]) * left;
    });
    draw();
    frame = t < 1 ? requestAnimationFrame(step) : 0;
  }

  // --- Dial math -----------------------------------------------------------------
  // value ∈ [0, crown] -> orbital angle ∈ [180deg (bottom), 0deg (top)]; the dot's
  // un-rotated rest position in the SVG is already 12 o'clock, so rotate(180deg)
  // is what puts it at the bottom for value=0.
  function winAngle(value, crown) {
    const t = clamp01(value, crown);
    return 180 * (1 - t);
  }

  // EV runs the opposite angular direction per spec (negated sign, same shape).
  // EV's domain is stated as [0 -> crown]; a negative EV clamps to the value=0 pose
  // (bottom) rather than extending the domain below zero — the simplest reading of
  // the spec's literal "[0 -> crown]" statement.
  function evAngle(value, crown) {
    const t = clamp01(value, crown);
    return -180 * (1 - t);
  }

  // Kelly is the fixed centre gem (scale/opacity), not an orbital dot -- different
  // visual encoding entirely. Linear interpolation between the mockup's weak (.66
  // scale/.55 opacity) and strong (1.08 scale/1 opacity) poses; see the README for
  // why this doesn't reproduce the mockup's demo numbers bit-exactly (the demo's
  // "strong" preset is an illustrative fully-lit pose, not a literal plot of its
  // kelly=1.2%-of-3% value).
  const GEM_SCALE_MIN = 0.66;
  const GEM_SCALE_MAX = 1.08;
  const GEM_OPACITY_MIN = 0.55;
  const GEM_OPACITY_MAX = 1.0;

  function clamp01(value, crown) {
    if (!crown) return 0;
    const v = Math.max(0, Math.min(Number(value) || 0, crown));
    return v / crown;
  }

  function draw() {
    const winCorrT = clamp01(cur.win_corr, crowns.win);
    const winIndepT = clamp01(cur.win_indep, crowns.win);
    const angleCorr = winAngle(cur.win_corr, crowns.win);
    const angleIndep = winAngle(cur.win_indep, crowns.win);
    el.wincorr.style.setProperty("--angle", angleCorr + "deg");
    el.winindep.style.setProperty("--angle", angleIndep + "deg");

    // Lift arc: a pathLength=360 circle drawn via stroke-dasharray (1 unit = 1deg),
    // rotated so the dash-start (the circle path's own 3-o'clock reference) lines up
    // with the smaller of the two dot angles; see README.md "Design math" for the
    // -90 reconciliation between that reference and the dot angles' own convention.
    const lift = cur.win_corr - cur.win_indep;
    const arcLen = Math.abs(angleCorr - angleIndep);
    const arcRotate = Math.min(angleCorr, angleIndep) - 90;
    const dash = arcLen.toFixed(1) + " " + (360 - arcLen).toFixed(1);
    [el.band, el.bandglow].forEach(function (node) {
      node.style.setProperty("--angle", arcRotate + "deg");
      node.style.strokeDasharray = dash;
      node.dataset.sign = sign(lift);
    });

    const evT = clamp01(cur.ev, crowns.ev);
    el.ev_g.style.setProperty("--angle", evAngle(cur.ev, crowns.ev) + "deg");
    el.evbead.forEach(function (node) {
      node.dataset.sign = sign(cur.ev);
    });

    const kellyT = clamp01(cur.kelly, crowns.kelly);
    const gemScale = GEM_SCALE_MIN + (GEM_SCALE_MAX - GEM_SCALE_MIN) * kellyT;
    const gemOpacity = GEM_OPACITY_MIN + (GEM_OPACITY_MAX - GEM_OPACITY_MIN) * kellyT;
    el.gem.style.setProperty("--scale", gemScale);
    el.gem.style.opacity = String(gemOpacity);

    // Crown overflow: at-or-past-crown pins the bead at 12 o'clock (handled by
    // winAngle/evAngle's own clamp already) and the orbital glows — each orbital
    // glows off its own value's crown state independently, so a slip can have (say)
    // its win dial pinned+glowing while EV is not.
    el.ovfWin.style.opacity = winCorrT >= 1 || winIndepT >= 1 ? "0.55" : "0";
    el.ovfEv.style.opacity = evT >= 1 ? "0.55" : "0";

    if (!priced) return;
    el.pay.textContent = cur.payout.toFixed(2) + "x";
    el.win.textContent = pct(cur.win_corr);
    el.indep.textContent = pct(cur.win_indep);
    el.lift.textContent = signedPct(lift);
    el.lift.dataset.sign = sign(lift);
    el.ev.textContent = signedPct(cur.ev);
    el.ev.dataset.sign = sign(cur.ev);
    el.kelly.textContent = pct(cur.kelly);
  }

  // --- Formatting ------------------------------------------------------------
  function pct(v) {
    return ((Number(v) || 0) * 100).toFixed(1) + "%";
  }
  function signedPct(v) {
    const n = Number(v) || 0;
    return (n >= 0 ? "+" : "") + (n * 100).toFixed(1) + "%";
  }
  // The data-sign value index.html colours by: "up" (green, zero included) or "down" (red).
  function sign(v) {
    return v < 0 ? "down" : "up";
  }

  renderCallback = render;
  post({ type: "streamlit:componentReady", apiVersion: 1 });
})();
