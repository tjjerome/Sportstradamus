/* Constellation component frontend — hand-authored, no build step.
 *
 * Speaks the Streamlit <-> iframe postMessage protocol directly (the same one
 * streamlit-component-lib emits) and draws the figure JSON Python passes in as three
 * stacked layers: a static plotly graph of the engraving, an SVG of the correlation edges
 * carried in layout.meta, and the interactive plotly graph of stars on top. Everything a
 * pointer does is answered here without a rerun: hover previews a star's ties and shows its
 * card, a click paints the star lit or dim at once, and lens changes animate. Clicks reach
 * Python as intents it acknowledges (README "Intents and ack").
 */
(function () {
  "use strict";

  // --- Minimal Streamlit bridge ------------------------------------------------
  const RENDER_EVENT = "streamlit:render";
  let renderCallback = null;
  let postedHeight = 0;

  function post(msg) {
    window.parent.postMessage(Object.assign({ isStreamlitMessage: true }, msg), "*");
  }
  function setFrameHeight(height) {
    postedHeight = height;
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
  const stage = document.getElementById("stage");
  const lowerDiv = document.getElementById("lower");
  const edgeSvg = document.getElementById("edges");
  const lensEdges = edgeSvg.querySelector("g.lens");
  const baseEdges = edgeSvg.querySelector("g.base");
  const chartDiv = document.getElementById("chart");
  const card = document.getElementById("card");
  const SVG_NS = "http://www.w3.org/2000/svg";
  const CONFIG = { displayModeBar: false, scrollZoom: false, responsive: true };
  const LOWER_CONFIG = { staticPlot: true, responsive: true, displayModeBar: false };
  const FRAME_PAD = 8; // headroom so top-row star labels aren't clipped
  const HOVER_FAINT = 0.18; // faint opacity for a hidden edge previewed on hover
  const HIDE_DELAY_MS = 220; // hover-intent: keep the card while the cursor travels to it
  // A fresh desktop card lets presses through to the stars under it, and takes the pointer only
  // once the pointer has rested on its star this long: longer than a pointer takes to pass over
  // a star, shorter than it takes to start reading a card.
  const HOLD_MS = 350;
  const DOT_MIN_PX = 3; // plotly's own "dot" dash: as long as the line is wide, never under 3px
  const HOVER_DISTANCE_PX = 20; // plotly's default hoverdistance, so a click reaches as far as hover
  const PICK_RADIUS_MIN_PX = 3; // plotly picks a marker as if it were at least this wide
  // plotly's "star" symbol draws its points at 1.4x the marker radius, so a card seated off
  // the radius alone would still cover the star's tips.
  const STAR_REACH = 1.4;
  const CARD_GAP_PX = 8; // clear air between a hovered star's tips and its card
  const CARD_EDGE_PX = 6; // the card keeps clear of the frame's edges
  // Extra frame height so the docked card clears the map. It tracks the card's height —
  // last-five and movement rows included — so a row the card gains has to grow it too.
  const MOBILE_CARD_PAD = 200;
  // The ban row, added only while some star has one. On a phone its sentence wraps to two
  // 12px lines (16px each at the card's 1.35 line-height), plus the row's 8px margin.
  const MOBILE_BAN_ROW_PAD = 40;
  // Lens motion runs on WAAPI opacity and CSS transforms, never Plotly.restyle: a restyle is
  // a full style pass (12-148 ms) every frame, which is what made the old fades stutter.
  // "Look deeper" only adds stars inside the map, so they fade; "look wider" recedes the map to
  // make room for the other games' sky, so the map glides while the sky fades.
  const DEEP_FADE_MS = 600; // deeper on: the new stars and their ties materialize
  const DEEP_OUT_MS = 300; // deeper off: shorter, because the redraw waits for it
  const SKY_OUT_MS = 250; // wider off: the sky clears before the map grows back
  const RECEDE_MS = 500; // wider on or off: the map glides between its poses
  const DEEP_GROUPS = ".tracedeep, .tracedeep_banned, #edges g.lens";
  const SKY_GROUPS = ".tracewider, .tracewider_labels, .tracewider_banned";
  // The opacity a star wears while a click contradicts its in_slip: a candidate or deep star
  // lights fully, a slip star dims to layout.meta's dim_alpha, and a wider star hides until
  // Python redraws it as a slip leg.
  const FLIP_OPACITY = { candidate: 1, deep: 1, wider: 0 };
  const NO_EDGES = { edges: [], focus_scale: 1 }; // an empty figure carries no layout.meta
  const REDUCED_MOTION =
    window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const COARSE_POINTER =
    window.matchMedia && window.matchMedia("(pointer: coarse)").matches;
  let MOBILE = false; // set per render from Python's mobile prop (viewport.is_mobile)
  let BANS = {}; // star key -> its card's "won't pair" text, worded server-side
  let meta = NO_EDGES; // the plotted figure's layout.meta
  let shown = null; // {deep, wider}: the lens stars on screen; null until the first render
  let stars = []; // per plotted star trace: its points, original look and flip look
  let inSlip = {}; // star key -> the server's in_slip
  let intents = {}; // star key -> {lit, seq}: clicks Python hasn't acknowledged yet
  let seq = 0;
  let lines = []; // {rec, el} per drawn edge
  let linesByKey = {}; // star key -> its lines
  let drawnMeta = null; // the edges and plot geometry the lines were drawn for
  let drawnGeometry = "";
  let lowerKey = ""; // what the static lower graph last plotted
  let previewKey = null; // star whose ties the hover preview lifts
  let pending = null; // the latest render, {args, fig}
  // Lens motion is playing. A render meanwhile only replaces `pending`, which the motion's end
  // draws, so a rerun mid-motion (a click's reconcile, a lens turned back) never snaps it.
  let moving = false;
  let hideTimer = null;
  let holdTimer = null;
  let pointer = [0, 0]; // the pointer's last position, which a pending hide checks
  let activeKey = null; // key whose card is showing

  // --- Figure layers -----------------------------------------------------------
  function render(args) {
    // Streamlit re-sends unchanged args after a frame-height change, and drawing them again would
    // redraw the map for nothing and close a docked card.
    const last = pending && pending.args;
    if (
      last &&
      args.figure_json === last.figure_json &&
      args.ack === last.ack &&
      args.mobile === last.mobile &&
      JSON.stringify(args.bans) === JSON.stringify(last.bans)
    ) {
      return;
    }
    const fig = JSON.parse(args.figure_json);
    MOBILE = !!args.mobile || COARSE_POINTER;
    BANS = args.bans;
    // The phone sizes its iframe from this post, so it goes out before anything that can
    // throw. It only grows here: a figure that shrinks keeps its old height until its
    // transition settles, so a sky still fading out is never clipped.
    setFrameHeight(Math.max(postedHeight, frameHeight(fig)));
    hideCard();
    pending = { args: args, fig: fig };
    if (!moving) advance();
  }

  // Moves the screen on to the latest render. A lens it turns off fades out first, and the
  // fade's end draws whichever render is latest by then.
  function advance() {
    const lens = lensesOf(pending.fig);
    let out = [];
    if (shown && !REDUCED_MOTION && shown.deep && !lens.deep) {
      out = fade(DEEP_GROUPS, { opacity: 0 }, { duration: DEEP_OUT_MS, fill: "forwards" });
    }
    if (shown && !REDUCED_MOTION && shown.wider && !lens.wider) {
      out = out.concat(fade(SKY_GROUPS, { opacity: 0 }, { duration: SKY_OUT_MS, fill: "forwards" }));
    }
    if (!out.length) {
      draw();
      return;
    }
    moving = true;
    Promise.all(
      out.map(function (animation) {
        return animation.finished;
      })
    ).then(function () {
      moving = false; // before draw(), so a throw there can't hold back every later render
      // The faded lens is off screen now, so a render that turned it straight back on fades in.
      shown = { deep: shown.deep && lens.deep, wider: shown.wider && lens.wider };
      draw();
      out.forEach(function (animation) {
        animation.cancel();
      });
    });
  }

  // Draws the latest render — both plotly graphs, the edges and the painted intents — then
  // runs the motion from what's on screen to its lens state.
  function draw() {
    const current = pending;
    const args = current.args;
    const fig = current.fig;
    const layout = fig.layout;
    const lens = lensesOf(fig);
    const nextMeta = layout.meta || NO_EDGES;
    const receding = shown && !REDUCED_MOTION && nextMeta.focus_scale !== meta.focus_scale;
    const from = receding && poses();
    // Hold the stage at its old height and lift plotly's clips while the map glides, so a figure
    // that shrinks (the phone's sky closing) doesn't clip the map before it arrives.
    stage.style.minHeight = receding ? stage.offsetHeight + "px" : "";
    stage.classList.toggle("cst-gliding", !!receding);
    meta = nextMeta;
    Object.keys(intents).forEach(function (key) {
      if (intents[key].seq <= args.ack) delete intents[key];
    });
    const upper = fig.data.filter(function (trace) {
      return trace.name !== "decoration";
    });
    const lower = fig.data.filter(function (trace) {
      return trace.name === "decoration";
    });
    indexStars(upper);
    // The engraving never changes on a click and its art is the heaviest thing on the map, so
    // the static lower graph re-plots only when something it draws changed.
    const key = JSON.stringify([
      lower,
      layout.images,
      layout.shapes,
      layout.height,
      layout.xaxis.range,
      layout.yaxis.range,
    ]);
    if (key !== lowerKey) {
      lowerKey = key;
      Plotly.react(lowerDiv, lower, without(layout, ["annotations", "meta"]), LOWER_CONFIG);
    }
    // plotly 2.27 empties a caption's group when its text goes blank and never refills that group
    // on a later react (uid = name keeps the layer): a star that gains a caption would show none.
    chartDiv.querySelectorAll(".textpoint:empty").forEach(function (group) {
      group.remove();
    });
    Plotly.react(chartDiv, upper, without(layout, ["images", "shapes", "meta"]), CONFIG);
    if (!shown) attachHandlers();
    drawEdges();
    let motion = [];
    if (shown && !REDUCED_MOTION) {
      if (lens.deep && !shown.deep) {
        const timing = { duration: DEEP_FADE_MS, easing: "ease-out" };
        motion = fade(DEEP_GROUPS, { opacity: 0, offset: 0 }, timing);
      }
      if (receding) motion = motion.concat(recede(from));
      if (lens.wider && !shown.wider) {
        const timing = { duration: RECEDE_MS / 2, delay: RECEDE_MS / 2, fill: "backwards" };
        motion = motion.concat(fade(SKY_GROUPS, { opacity: 0, offset: 0 }, timing));
      }
    }
    shown = lens;
    moving = motion.length > 0;
    Promise.all(
      motion.map(function (animation) {
        return animation.finished;
      })
    ).then(function () {
      moving = false;
      stage.style.minHeight = "";
      stage.classList.remove("cst-gliding");
      // A render that came in meanwhile goes next. The height it posted fits both figures, so the
      // exact height waits for the last settle rather than clipping the next motion.
      if (pending !== current) advance();
      else setFrameHeight(frameHeight(fig));
    });
  }

  // Indexes the star traces (the ones carrying customdata) and writes each one's painted look
  // into the trace before Plotly.react, so a render never flashes a pending star's old look.
  function indexStars(upper) {
    stars = [];
    inSlip = {};
    upper.forEach(function (trace, index) {
      // plotly.py 6.9 strips uid from to_json. A uid equal to the trace's (unique) name keeps
      // its <g class="trace<uid>"> across reacts, so a lens fade holds on to the right group.
      trace.uid = trace.name;
      const points = trace.customdata;
      if (!points) return;
      points.forEach(function (cd) {
        inSlip[cd[0]] = !!cd[8];
      });
      const opacity = trace.marker.opacity === undefined ? 1 : trace.marker.opacity;
      const star = {
        index: index,
        name: trace.name,
        points: points,
        keys: points.map(function (cd) {
          return cd[0];
        }),
        x: trace.x,
        y: trace.y,
        sizes: trace.marker.size,
        colors: perPoint(trace.marker.color, points),
        opacities: perPoint(opacity, points),
        flipColors: trace.meta,
        flipOpacity: trace.name === "active" ? meta.dim_alpha : FLIP_OPACITY[trace.name],
      };
      const next = look(star);
      star.flipped = next.flipped;
      if (next.flipped) {
        trace.marker.color = next.colors;
        trace.marker.opacity = next.opacities;
      }
      stars.push(star);
    });
  }

  function perPoint(value, points) {
    return Array.isArray(value)
      ? value
      : points.map(function () {
          return value;
        });
  }

  // Data coordinates -> px in the star graph, which are px in #layers too. This reads plotly's
  // private axis maths, which holds because index.html pins plotly.js at 2.27.0 (README).
  function toPx(x, y) {
    const full = chartDiv._fullLayout;
    return [full.xaxis._offset + full.xaxis.l2p(x), full.yaxis._offset + full.yaxis.l2p(y)];
  }

  // Strokes layout.meta's edges between the two plotly graphs. It runs on the star graph's
  // plotly_afterplot, which follows every react, restyle and responsive resize, so it rebuilds
  // only when the edges or the plot geometry changed, and it never restyles (that would loop).
  function drawEdges() {
    const full = chartDiv._fullLayout;
    const geometry = [
      full.width,
      full.height,
      full.xaxis._offset,
      full.xaxis._length,
      full.yaxis._offset,
      full.yaxis._length,
    ].join();
    if (meta === drawnMeta && geometry === drawnGeometry) return;
    drawnMeta = meta;
    drawnGeometry = geometry;
    edgeSvg.setAttribute("width", full.width);
    edgeSvg.setAttribute("height", full.height);
    const lensLines = [];
    const baseLines = [];
    lines = [];
    linesByKey = {};
    meta.edges.forEach(function (rec) {
      const el = document.createElementNS(SVG_NS, "line");
      const a = toPx(rec.x[0], rec.y[0]);
      const b = toPx(rec.x[1], rec.y[1]);
      el.setAttribute("x1", a[0]);
      el.setAttribute("y1", a[1]);
      el.setAttribute("x2", b[0]);
      el.setAttribute("y2", b[1]);
      el.setAttribute("stroke-width", rec.width);
      if (rec.dash === "dot") {
        const dash = Math.max(rec.width, DOT_MIN_PX);
        el.setAttribute("stroke-dasharray", dash + "," + dash);
      }
      (rec.lens ? lensLines : baseLines).push(el);
      const line = { rec: rec, el: el };
      lines.push(line);
      [rec.a, rec.b].forEach(function (end) {
        (linesByKey[end] = linesByKey[end] || []).push(line);
      });
    });
    lensEdges.replaceChildren(...lensLines);
    baseEdges.replaceChildren(...baseLines);
    paintEdges(lines);
  }

  // Lens state as the plotted trace names give it: a "deep" trace means deeper is on, a
  // "wider" trace means wider is.
  function lensesOf(fig) {
    const names = fig.data.map(function (trace) {
      return trace.name;
    });
    return { deep: names.includes("deep"), wider: names.includes("wider") };
  }

  function frameHeight(fig) {
    return (
      fig.layout.height +
      FRAME_PAD +
      (MOBILE ? MOBILE_CARD_PAD + (Object.keys(BANS).length > 0 ? MOBILE_BAN_ROW_PAD : 0) : 0)
    );
  }

  function without(layout, names) {
    const copy = Object.assign({}, layout);
    names.forEach(function (name) {
      delete copy[name];
    });
    return copy;
  }

  // --- Lens motion -------------------------------------------------------------
  // Fades every element `selector` matches. The keyframe left out is the element's own
  // opacity, so a fade in lands exactly on the value plotly set inline, whatever it is.
  function fade(selector, keyframe, timing) {
    return Array.from(document.querySelectorAll(selector), function (el) {
      return el.animate([keyframe], timing);
    });
  }

  // What a wider glide starts from, read before the redraw replaces it: the engraving's data
  // origin and focus, and each star's px and caption box by star key.
  function poses() {
    const from = { origin: toPx(0, 0), focus: meta.focus_scale, stars: {}, captions: {} };
    stars.forEach(function (star) {
      const captions = captionsOf(star);
      star.keys.forEach(function (key, i) {
        from.stars[key] = toPx(star.x[i], star.y[i]);
        if (captions[i]) from.captions[key] = captions[i].getBoundingClientRect();
      });
    });
    return from;
  }

  // A star trace's caption <text> by point index. plotly keeps no text group per point (a redraw
  // drops the groups of captions that went empty), so a caption's index is its bound datum's.
  function captionsOf(star) {
    const captions = [];
    chartDiv.querySelectorAll(".trace" + star.name + " .textpoint text").forEach(function (text) {
      captions[text.__data__.i] = text;
    });
    return captions;
  }

  // Glides the map between its "look wider" poses. Python scales the engraving by focus_scale,
  // so it recedes as one piece; it lays the stars out afresh and never scales a marker, a
  // caption or a tie, so those travel one by one, translate only, from where the old figure drew
  // them. The engraving's animation is the clock the ties walk by.
  function recede(from) {
    const timing = { duration: RECEDE_MS, easing: "ease-in-out" };
    const to = toPx(0, 0);
    const shift = from.origin[0] - to[0] + "px, " + (from.origin[1] - to[1]) + "px";
    const start = "translate(" + shift + ") scale(" + from.focus / meta.focus_scale + ")";
    lowerDiv.style.transformOrigin = to[0] + "px " + to[1] + "px";
    const clock = lowerDiv.animate([{ transform: start }, { transform: "none" }], timing);
    glideEdges(clock, from.stars);
    return [clock].concat(
      moves(from).map(function (move) {
        return move.el.animate(
          [
            { transform: "translate(" + move.dx + "px, " + move.dy + "px)" + move.pose },
            { transform: "translate(0px, 0px)" + move.pose },
          ],
          timing
        );
      })
    );
  }

  // What a wider glide carries, as {el, dx, dy, pose}: each star, its caption and its ban mark,
  // with the offset back to where the old figure drew it and the pose it lands on. A marker's
  // pose is its transform attribute, which a CSS transform replaces, so its keyframes repeat it;
  // a caption sits on x and y and needs none. Stars with no old px (the sky, a newly promoted
  // star) stay out, and their groups fade as before.
  function moves(from) {
    const out = [];
    // Returns the move, whose offset a caption new to this figure borrows.
    function point(el, old, now) {
      const matrix = el.transform.baseVal.consolidate().matrix;
      const pose = " translate(" + matrix.e + "px, " + matrix.f + "px)";
      const move = { el: el, dx: old[0] - now[0], dy: old[1] - now[1], pose: pose };
      out.push(move);
      return move;
    }
    stars.forEach(function (star) {
      const points = chartDiv.querySelectorAll(".trace" + star.name + " .points path.point");
      const captions = captionsOf(star);
      star.keys.forEach(function (key, i) {
        if (!(key in from.stars)) return;
        const move = point(points[i], from.stars[key], toPx(star.x[i], star.y[i]));
        const caption = captions[i];
        if (!caption) return;
        // A caption starts on its old box rather than its anchor, since one that switches sides
        // re-anchors its text; a caption new to this figure rides in with its star.
        const old = from.captions[key];
        const now = caption.getBoundingClientRect();
        const offset = old ? { dx: old.left - now.left, dy: old.top - now.top } : move;
        out.push({ el: caption, dx: offset.dx, dy: offset.dy, pose: "" });
      });
    });
    // A ban mark sits on its host star's data coordinates (constellation_bans.py).
    chartDiv.data.forEach(function (trace) {
      const host = stars.find(function (star) {
        return trace.name === star.name + "_banned";
      });
      if (!host) return;
      const marks = chartDiv.querySelectorAll(".trace" + trace.name + " .points path.point");
      trace.x.forEach(function (x, j) {
        const i = host.x.findIndex(function (hostX, k) {
          return hostX === x && host.y[k] === trace.y[j];
        });
        const old = i >= 0 && from.stars[host.keys[i]];
        if (old) point(marks[j], old, toPx(x, trace.y[j]));
      });
    });
    return out;
  }

  // Walks each tie's ends from its stars' old px to their new px on the clock's eased progress,
  // since a line's ends are attributes no CSS animation reaches. The progress turns null once
  // the clock finishes, and the ends land where drawEdges put them. An end whose star had no old
  // px stays there throughout.
  function glideEdges(clock, from) {
    const ends = [];
    lines.forEach(function (line) {
      [line.rec.a, line.rec.b].forEach(function (key, end) {
        const now = toPx(line.rec.x[end], line.rec.y[end]);
        ends.push({ el: line.el, n: end + 1, now: now, old: from[key] || now });
      });
    });
    requestAnimationFrame(function frame() {
      const t = clock.effect.getComputedTiming().progress;
      ends.forEach(function (end) {
        const x = t === null ? end.now[0] : end.old[0] + (end.now[0] - end.old[0]) * t;
        const y = t === null ? end.now[1] : end.old[1] + (end.now[1] - end.old[1]) * t;
        end.el.setAttribute("x" + end.n, x);
        end.el.setAttribute("y" + end.n, y);
      });
      if (t !== null) requestAnimationFrame(frame);
    });
  }

  // --- Intents and paint -------------------------------------------------------
  // A click paints at once and rides in every value until a render's ack reaches its seq, so
  // a value Streamlit drops or coalesces loses nothing and a stale render never flickers it.
  function toggle(key) {
    intents[key] = { lit: !isLit(key), seq: nextSeq() };
    paint();
    send(null);
  }

  // Taken from the clock rather than counted: an iframe Streamlit remounts would restart a
  // counter at zero, and Python would ignore every click after that as already applied.
  function nextSeq() {
    seq = Math.max(seq + 1, Date.now());
    return seq;
  }

  function send(detail) {
    const lit = {};
    Object.keys(intents).forEach(function (key) {
      lit[key] = intents[key].lit;
    });
    setComponentValue({ seq: seq, lit: lit, detail: detail });
  }

  function isLit(key) {
    return key in intents ? intents[key].lit : !!inSlip[key];
  }

  // A star trace's colours and opacities as the paint rule has them, and which points flipped.
  function look(star) {
    const colors = star.colors.slice();
    const opacities = star.opacities.slice();
    const flipped = [];
    star.keys.forEach(function (key, i) {
      if (isLit(key) === inSlip[key]) return;
      flipped.push(i);
      if (star.flipColors) colors[i] = star.flipColors[i];
      opacities[i] = star.flipOpacity;
    });
    return { flipped: flipped.join(), colors: colors, opacities: opacities };
  }

  // A wider star a click has hidden stays out of reach of hover and clicks until Python
  // redraws it.
  function hidden(star, i) {
    const key = star.keys[i];
    return (isLit(key) === inSlip[key] ? star.opacities[i] : star.flipOpacity) === 0;
  }

  // Repaints stars and ties after a click. Only star traces whose flipped points changed are
  // restyled, so a click restyles one trace.
  function paint() {
    const index = [];
    const colors = [];
    const opacities = [];
    stars.forEach(function (star) {
      const next = look(star);
      if (next.flipped === star.flipped) return;
      star.flipped = next.flipped;
      index.push(star.index);
      colors.push(next.colors);
      opacities.push(next.opacities);
    });
    if (index.length) {
      Plotly.restyle(chartDiv, { "marker.color": colors, "marker.opacity": opacities }, index);
    }
    paintEdges(lines);
  }

  // A tie touching a pending intent shows the slip to come — lit only when both ends are —
  // and every other tie keeps the server's opacity. Hover lifts a star's ties to HOVER_FAINT.
  function paintEdges(list) {
    list.forEach(function (line) {
      const rec = line.rec;
      let alpha = rec.opacity;
      if (rec.a in intents || rec.b in intents) {
        alpha = isLit(rec.a) && isLit(rec.b) ? rec.lit : meta.edge_base;
      }
      if (previewKey === rec.a || previewKey === rec.b) alpha = Math.max(alpha, HOVER_FAINT);
      line.el.style.opacity = alpha;
    });
  }

  // Lifts `key`'s ties and lets the previous star's fall back; null ends the preview.
  function preview(key) {
    const touched = (linesByKey[previewKey] || []).concat(linesByKey[key] || []);
    previewKey = key;
    paintEdges(touched);
  }

  // --- Pointer -----------------------------------------------------------------
  function attachHandlers() {
    chartDiv.on("plotly_afterplot", drawEdges);
    // Ban marks and labels skip hover, so a hover always lands on a star trace.
    chartDiv.on("plotly_hover", function (data) {
      if (MOBILE) return;
      const pt = data.points[0];
      const hit = {
        star: stars.find(function (star) {
          return star.index === pt.curveNumber;
        }),
        i: pt.pointNumber,
      };
      if (hidden(hit.star, hit.i)) return;
      preview(hit.star.keys[hit.i]);
      showCard(hit);
    });
    chartDiv.on("plotly_unhover", function () {
      if (MOBILE) return;
      preview(null);
      scheduleHide();
    });
    // The map never drags (dragmode is false), so presses stop before plotly's drag handler,
    // which answers every lift with a click of its own beside the browser's: a phone tap
    // reached the map as three clicks.
    ["mousedown", "touchstart"].forEach(function (type) {
      chartDiv.addEventListener(
        type,
        function (event) {
          event.stopPropagation();
        },
        true
      );
    });
    // A click resolves its own coordinates: plotly_click reports what plotly's last hover saw,
    // and plotly throttles hover to one pass per 50 ms, so a quick second click named the first
    // star again.
    chartDiv.addEventListener("click", function (event) {
      const hit = starAt(event);
      const key = hit && hit.star.keys[hit.i];
      if (!MOBILE) {
        if (hit) {
          toggle(key);
          hideCard(); // now, not at the rerun's render, so it can't take the next press
        }
      } else if (!hit) {
        hideCard(); // an empty-sky tap dismisses the docked card
      } else if (key === activeKey) {
        toggle(key); // a second tap on the focused star
        hideCard();
      } else {
        preview(key);
        showCard(hit);
      }
    });
    // A pointer leaving the frame sends no move, only an out event, which carries where it went.
    ["pointermove", "pointerout"].forEach(function (type) {
      window.addEventListener(type, function (event) {
        pointer = [event.clientX, event.clientY];
      });
    });
    card.addEventListener("mouseleave", function () {
      if (!MOBILE) hideCard();
    });
  }

  // The star under a click, picked by plotly's own closest-point rule so the star whose card
  // shows is the star a click toggles: least distance past a marker's edge, within
  // HOVER_DISTANCE_PX, with a press inside overlapping markers going to the smaller one.
  function starAt(event) {
    const box = chartDiv.getBoundingClientRect();
    let hit = null;
    let best = HOVER_DISTANCE_PX;
    stars.forEach(function (star) {
      star.keys.forEach(function (key, i) {
        if (hidden(star, i)) return;
        const at = toPx(star.x[i], star.y[i]);
        const radius = Math.max(PICK_RADIUS_MIN_PX, star.sizes[i] / 2);
        const off = Math.hypot(event.clientX - box.left - at[0], event.clientY - box.top - at[1]);
        const distance = Math.max(off - radius, 1 - PICK_RADIUS_MIN_PX / radius);
        if (distance <= best) {
          best = distance;
          hit = { star: star, i: i };
        }
      });
    });
    return hit;
  }

  // --- Hover card --------------------------------------------------------------
  function showCard(hit) {
    clearTimeout(hideTimer);
    const cd = hit.star.points[hit.i]; // [key, player, market, bet, line, win, boost, kelly, inSlip]
    activeKey = cd[0];
    card.innerHTML = cardHtml(cd);
    card.querySelector(".cst-btn.cst-detail").addEventListener("click", function () {
      nextSeq();
      send(activeKey);
    });
    const toggleButton = card.querySelector(".cst-btn.cst-toggle");
    if (toggleButton) {
      toggleButton.addEventListener("click", function () {
        toggle(activeKey);
        hideCard();
      });
    }
    card.classList.toggle("cst-docked", MOBILE);
    card.classList.remove("cst-hidden");
    card.setAttribute("aria-hidden", "false");
    if (!MOBILE) {
      placeCard(hit);
      card.classList.remove("cst-held");
      clearTimeout(holdTimer);
      holdTimer = setTimeout(function () {
        card.classList.add("cst-held");
      }, HOLD_MS);
    } else {
      card.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }

  // Seats the desktop card beside its star — on the right, or the left when the right would
  // spill off the frame — and clamps it only vertically. A card over its star would take the
  // press meant for the star, and the click would never reach the map.
  function placeCard(hit) {
    const star = hit.star;
    const at = toPx(star.x[hit.i], star.y[hit.i]);
    const reach = (star.sizes[hit.i] / 2) * STAR_REACH + CARD_GAP_PX;
    const width = card.offsetWidth;
    const height = card.offsetHeight;
    const fitsRight = at[0] + reach + width <= window.innerWidth - CARD_EDGE_PX;
    const top = Math.min(at[1] - height / 2, window.innerHeight - height - CARD_EDGE_PX);
    card.style.left = (fitsRight ? at[0] + reach : at[0] - reach - width) + "px";
    card.style.top = Math.max(CARD_EDGE_PX, top) + "px";
  }

  // The hide keeps waiting while the pointer is on the card. It asks where the pointer is, since
  // Chromium sends no mouseenter to a card that turns solid under a pointer at rest.
  function scheduleHide() {
    clearTimeout(hideTimer);
    hideTimer = setTimeout(function () {
      if (card.contains(document.elementFromPoint(pointer[0], pointer[1]))) scheduleHide();
      else hideCard();
    }, HIDE_DELAY_MS);
  }

  function hideCard() {
    clearTimeout(holdTimer);
    card.classList.remove("cst-held");
    card.classList.add("cst-hidden");
    card.setAttribute("aria-hidden", "true");
    activeKey = null;
    preview(null);
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

  // The card's heavy assets (sparks, moves, headshots) ride in a hidden st.html carrier beside
  // the component, which Streamlit re-sends by hash rather than with every click's render.
  // Read when a card opens: the carrier's script can land after this frame's first render.
  function asset(kind, id) {
    const assets = window.parent.__cstCards;
    return assets && assets[kind][id];
  }

  // The star's last-five row, already drawn as SVG by Python — nothing here computes
  // geometry. A star the gamelog can't answer for (a player it doesn't carry, a market
  // that names no column of it) says so, rather than showing an empty box.
  function sparkHtml(key) {
    const spark = asset("sparks", key);
    return spark
      ? '<div class="cst-spark">' + spark + "</div>"
      : '<div class="cst-scar">No last 5 for this leg</div>';
  }

  // The offer's line-movement row, drawn by Python the same way. Unlike the last five it
  // has no scar: an offer the ladder never saw has no movement to be missing.
  function moveHtml(key) {
    const move = asset("moves", key);
    return move ? '<div class="cst-spark">' + move + "</div>" : "";
  }

  // The player's face, embedded server-side from this box's headshot cache. A box that has
  // never run `fetch headshots`, or a player it holds no file for, draws the initials disc
  // this replaced — so a miss is indistinguishable from the card before faces existed.
  function shotHtml(player) {
    const uri = asset("shots", player);
    return uri
      ? '<img class="cst-shot" alt="" src="' + esc(uri) + '">'
      : '<div class="cst-shot">' + esc(initials(player)) + "</div>";
  }

  // The platform's refusal to pair this leg with one already in the slip, worded
  // server-side. Most stars carry none, and those draw no row.
  function banHtml(key) {
    const ban = BANS[key];
    return ban ? '<div class="cst-ban">' + esc(ban) + "</div>" : "";
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
      shotHtml(player),
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
      banHtml(cd[0]),
      sparkHtml(cd[0]),
      moveHtml(cd[0]),
      '<div class="cst-actions">',
      MOBILE
        ? '<button class="cst-btn cst-toggle" type="button">' +
          (isLit(cd[0]) ? "Remove from slip" : "Add to slip") +
          "</button>"
        : "",
      '<button class="cst-btn cst-detail" type="button">Full detail →</button>',
      "</div>",
    ].join("");
  }

  renderCallback = render;
  post({ type: "streamlit:componentReady", apiVersion: 1 });
})();
