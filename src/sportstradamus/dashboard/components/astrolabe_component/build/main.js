/* Astrolabe component frontend — hand-authored, no build step.
 *
 * Speaks the Streamlit <-> iframe postMessage protocol directly (the same one
 * streamlit-component-lib emits) and drives the three orbiting dials (win/EV/Kelly)
 * plus the lift arc off the plain `payload` dict Python passes in — no plotly, no
 * server callback (selection happens on the constellation's own stars; this is a
 * pure readout). Ported from docs/mockups/p8-constellation-lab.html rev 5: the
 * mockup's CSS already declares the transitions unconditionally on .grp/.band/
 * .gem/etc, so any inline style change on those elements animates on its own —
 * the only extra state needed here is suppressing that animation on the very
 * first mount (nothing to sweep in from). After the first render, every render
 * just re-applies the payload's current values: a real add/remove changes them
 * (the transition eases), an incidental rerun with the same leg-set recomputes
 * identical values (the transition fires but has nothing to visibly move).
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
  let lastNonce = null; // null until the first render; then the last handled payload.nonce

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

  function render(args) {
    const p = args.payload || {};
    const crowns = p.crowns || {};

    el.legs.textContent = (p.legs || 0) + " · " + (p.play_type || "");
    el.pay.textContent = (Number(p.payout) || 0).toFixed(2) + "x";
    el.win.textContent = pct(p.win_corr);
    el.indep.textContent = pct(p.win_indep);
    el.ev.textContent = signedPct(p.ev);
    el.ev.style.color = Number(p.ev) >= 0 ? "var(--green)" : "var(--red)";
    el.kelly.textContent = pct(p.kelly);

    const lift = (Number(p.win_corr) || 0) - (Number(p.win_indep) || 0);
    el.lift.textContent = signedPct(lift);
    el.lift.style.color = lift >= 0 ? "var(--green)" : "var(--red)";

    applyDials(p, crowns, lift);

    // Diff payload.nonce against the last-seen value purely to detect "is this the
    // very first render" (lastNonce still null): that's the one case needing the
    // sweep suppressed, since there is no prior pose to animate in from. Any later
    // render — nonce changed (a real add/remove) or unchanged (an incidental
    // rerun re-deriving the same slip) — can transition normally; the unchanged
    // case just re-applies identical values, which is a harmless no-op visually.
    if (lastNonce === null) {
      astro.classList.add("no-anim");
      // Force a style flush so the "no transition" rule applies to the values just
      // set above, then drop the override on the next frame — every render after
      // this one (a real nonce change) transitions normally.
      // eslint-disable-next-line no-unused-expressions
      astro.offsetHeight;
      requestAnimationFrame(function () {
        astro.classList.remove("no-anim");
      });
    }
    lastNonce = p.nonce;

    setFrameHeight(FRAME_HEIGHT);
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

  function applyDials(p, crowns, lift) {
    const winCorrT = clamp01(p.win_corr, crowns.win);
    const winIndepT = clamp01(p.win_indep, crowns.win);
    const angleCorr = winAngle(p.win_corr, crowns.win);
    const angleIndep = winAngle(p.win_indep, crowns.win);
    el.wincorr.style.transform = "rotate(" + angleCorr + "deg)";
    el.winindep.style.transform = "rotate(" + angleIndep + "deg)";

    // Lift arc: a pathLength=360 circle drawn via stroke-dasharray (1 unit = 1deg),
    // rotated so the dash-start (the circle path's own 3-o'clock reference) lines up
    // with the smaller of the two dot angles; see README.md "Design math" for the
    // -90 reconciliation between that reference and the dot angles' own convention.
    const arcLen = Math.abs(angleCorr - angleIndep);
    const arcRotate = Math.min(angleCorr, angleIndep) - 90;
    const dash = arcLen.toFixed(1) + " " + (360 - arcLen).toFixed(1);
    const bandUrl = lift >= 0 ? "url(#bandGreen)" : "url(#bandRed)";
    [el.band, el.bandglow].forEach(function (node) {
      node.style.transform = "rotate(" + arcRotate + "deg)";
      node.style.strokeDasharray = dash;
      node.style.stroke = bandUrl;
    });

    const evT = clamp01(p.ev, crowns.ev);
    el.ev_g.style.transform = "rotate(" + evAngle(p.ev, crowns.ev) + "deg)";
    el.evbead.forEach(function (node) {
      node.style.fill = Number(p.ev) >= 0 ? "var(--green)" : "var(--red)";
    });

    const kellyT = clamp01(p.kelly, crowns.kelly);
    const gemScale = GEM_SCALE_MIN + (GEM_SCALE_MAX - GEM_SCALE_MIN) * kellyT;
    const gemOpacity = GEM_OPACITY_MIN + (GEM_OPACITY_MAX - GEM_OPACITY_MIN) * kellyT;
    el.gem.style.transform = "scale(" + gemScale + ")";
    el.gem.style.opacity = String(gemOpacity);

    // Crown overflow: at-or-past-crown pins the bead at 12 o'clock (handled by
    // winAngle/evAngle's own clamp already) and the orbital glows — each orbital
    // glows off its own value's crown state independently, so a slip can have (say)
    // its win dial pinned+glowing while EV is not.
    el.ovfWin.style.opacity = winCorrT >= 1 || winIndepT >= 1 ? "0.55" : "0";
    el.ovfEv.style.opacity = evT >= 1 ? "0.55" : "0";
  }

  // --- Formatting ------------------------------------------------------------
  function pct(v) {
    return ((Number(v) || 0) * 100).toFixed(1) + "%";
  }
  function signedPct(v) {
    const n = Number(v) || 0;
    return (n >= 0 ? "+" : "") + (n * 100).toFixed(1) + "%";
  }

  renderCallback = render;
  post({ type: "streamlit:componentReady", apiVersion: 1 });
})();
