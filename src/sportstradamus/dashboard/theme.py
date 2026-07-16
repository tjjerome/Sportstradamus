"""Non-Streamlit mirror of the dashboard design tokens (DESIGN.md §2)."""

import importlib.resources as pkg_resources
import json
import random
from functools import cache

import plotly.graph_objects as go
import plotly.io as pio

from sportstradamus import data

# Celestial gold accent — oracle decoration (kickers, prophecy, constellation) and
# interactive-highlight (active/selected/hovered UI states); never a data mark except
# the constellation correlation edge (DESIGN §4a) and the Receipts reliability
# diagram's alt-line/ladder series marker (series identity, not a value encoding).
GOLD = "#C9A227"

# Neutral gray (DESIGN.md §2 grayColor) — context/secondary marks, e.g. the
# constellation's non-slip legs against the gold thesis stars.
GRAY = "#8A91A0"

# Fallback secondary for an unmapped team_colors() code — darker neutral than GRAY
# so the (primary, secondary) fallback pair still reads as two distinct tones.
GRAY_SECONDARY = "#5A6070"

# Diverging heatmap ramp (red ↔ neutral ↔ blue) for above/below-centre table cells
# — mirrors config.toml chartDivergingColors so runtime grid code reaches it without a
# TOML read; tests/golden/test_design_tokens.py pins this equal to the config ramp.
DIVERGING_COLORS = [
    "#9A2B2B",
    "#C0453E",
    "#D97A6C",
    "#E8B4A8",
    "#EEE3DC",
    "#CBDDF7",
    "#7FAAE8",
    "#3D72D6",
    "#2351B4",
    "#142C66",
]

# Semantic chart colors — mirror config.toml greenColor/redColor/orangeColor so ad-hoc
# Plotly figures reach the same good/bad/warn hues Streamlit widgets use, instead of the
# off-token placeholder hexes tests/golden/test_design_tokens.py bans.
GREEN = "#1F9D55"
RED = "#E5484D"
ORANGE = "#F5A524"

# Sequential heatmap ramp — mirrors config.toml chartSequentialColors verbatim.
SEQUENTIAL_COLORS = [
    "#EAF1FC",
    "#CBDDF7",
    "#A6C5F0",
    "#7FAAE8",
    "#5B8DE0",
    "#3D72D6",
    "#2E6BE6",
    "#2351B4",
    "#183A82",
    "#0E2350",
]


@cache
def _team_assets() -> dict:
    with (pkg_resources.files(data) / "config/team_assets.json").open() as f:
        return json.load(f)


def team_colors(league: str, code: str) -> tuple[str, str]:
    """(primary, secondary) for a team; unknown codes get the neutral gray pair.

    A past incident had the same WNBA Portland franchise recorded as ``PDX`` on
    one book and ``POR`` on another, producing duplicate slate cards — this must
    never KeyError on an unrecognized code.
    """
    entry = _team_assets().get(league, {}).get(code)
    if not entry:
        return (GRAY, GRAY_SECONDARY)
    return (entry["primary"], entry["secondary"])


def team_name(league: str, code: str) -> str:
    """Full team name for a code (``("NBA", "GSW") -> "Golden State Warriors"``).

    Falls back to the code itself when the team or its ``name`` is unmapped, so a new
    or unknown team degrades to its abbreviation rather than KeyError-ing (same
    unknown-code contract as :func:`team_colors`).
    """
    entry = _team_assets().get(league, {}).get(code)
    if not entry or not entry.get("name"):
        return code
    return entry["name"]


def register_plotly_template() -> None:
    """Register + default the token template so ad-hoc Plotly figures inherit DESIGN.md.

    Mirrors config.toml chartCategoricalColors as the colorway; Streamlit's own theming
    covers st.plotly_chart when a figure's own template/colors aren't set, but ad-hoc
    go.Figure() traces (e.g. profit_sim's per-strategy lines) need an explicit default.
    """
    pio.templates["sportstradamus"] = go.layout.Template(
        layout={
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "font": {"family": "IBM Plex Sans", "color": "#E6E9EF", "size": 12},
            "colorway": [
                "#2E6BE6",
                "#1F9D55",
                "#E69F00",
                "#E5484D",
                "#56B4E9",
                "#CC79A7",
                "#0072B2",
                "#F0E442",
            ],
            "xaxis": {"gridcolor": "#2A2E37", "zerolinecolor": "#2A2E37"},
            "yaxis": {"gridcolor": "#2A2E37", "zerolinecolor": "#2A2E37"},
        }
    )
    pio.templates.default = "sportstradamus"


# Starfield dust is built from a seeded PRNG (below) rather than hand-listed so ~80
# dots stay readable under layout="wide" without an 80-line literal.
_STARFIELD_SEED = 20260705  # fixed so dust positions are identical across reruns/deploys
_STARFIELD_DUST_COUNT = 80  # ~30 was too sparse to read in a wide layout's whitespace
_STARFIELD_MAX_ALPHA = 0.20  # DESIGN §3 static-ambient ceiling (twinkles exempt, see @keyframes)


def _starfield_background() -> str:
    """CSS ``background`` layers for ``.starfield``: dust dots + nebula washes + base.

    Positions come from a seeded PRNG so the field is deterministic (no import-time
    randomness); every static dust/wash alpha stays <= the DESIGN §3 0.20 ceiling.
    """
    rng = random.Random(_STARFIELD_SEED)
    dot_template = (
        "radial-gradient(circle at {x}% {y}%, rgba(230,233,239,{a}) 0 {r}px, transparent {r}px)"
    )
    dots = [
        dot_template.format(
            x=round(rng.uniform(1, 99), 2),
            y=round(rng.uniform(2, 97), 2),
            a=round(rng.uniform(0.06, _STARFIELD_MAX_ALPHA), 2),
            r=round(rng.uniform(1.5, 2.5), 1),
        )
        for _ in range(_STARFIELD_DUST_COUNT)
    ]
    washes = [
        "radial-gradient(ellipse at 18% 4%, rgba(46,107,230,.10), transparent 46%)",
        "radial-gradient(ellipse at 86% 22%, rgba(201,162,39,.05), transparent 42%)",
        "radial-gradient(ellipse at 50% 96%, rgba(46,107,230,.06), transparent 55%)",
    ]
    return ",\n    ".join([*dots, *washes]) + ",\n    #0E1117"


# The one sanctioned CSS injection site for the whole dashboard (DESIGN.md §3):
# display-only celestial fonts (.celestial-kicker / .celestial-headline), the
# page-hero header block, the gold active-state accents (Games lens toggles + the
# segmented controls), and the ambient starfield layer.
# app.py injects this once, right after st.set_page_config; the __STARFIELD_DUST__
# sentinel below is filled by _starfield_background() when APP_CSS is built.
_APP_CSS_TEMPLATE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;600&family=Cormorant+Garamond:ital,wght@1,600&family=Spectral:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');
.celestial-kicker{font-family:'Cinzel',serif;font-weight:600;font-size:10.5px;
  letter-spacing:.26em;text-transform:uppercase;color:#C9A227}
.celestial-headline{font-family:'Cormorant Garamond',serif;font-style:italic;
  font-weight:600;color:#C9A227}
/* Manuscript type (DESIGN §2): Spectral is the config.toml base for body + headings; these two
   rules hold the line that a DATA numeral is never set in the serif body -- metric values/deltas
   stay Plex Mono, metric labels take the Cinzel small-caps. !important because Streamlit's own theme
   CSS mounts after this injected <style> and would otherwise win the source-order tie. Grid/table
   numeric cells set their own inline mono, which already beats any class rule. */
[data-testid="stMetricValue"],[data-testid="stMetricDelta"]{font-family:'IBM Plex Mono',monospace !important}
[data-testid="stMetricLabel"]{font-family:'Cinzel',serif !important;letter-spacing:.06em}
/* Page-hero header (mockup .head block): gold Cinzel kicker over a plain sans display
   title and a quiet updated line, on a faint two-stop nebula wash (blue + gold <= 12%,
   DESIGN §3). Replaces the retired render_banner() Sheets-era banners. */
.page-hero{position:relative;overflow:hidden;border-radius:4px;padding:14px 16px 4px;
  margin-bottom:6px;background:radial-gradient(ellipse at 12% -40%, rgba(46,107,230,.10), transparent 52%),
    radial-gradient(ellipse at 90% 0%, rgba(201,162,39,.05), transparent 44%)}
.page-hero .hero-title{font-size:23px;margin:1px 0;font-weight:700;color:#E6E9EF}
.hero-updated{color:#8A91A0;font-size:12px}
/* Tonight prophecy cards (mockup p8-tonight.html .card): matchup-first in sober Plex,
   the oracle voice demoted to a gold Cormorant subline (DESIGN §1 broadcast-fact +
   mystic-chrome split). Two-stop nebula wash over the surface so stars glow through
   faintly; the game-shape glyph + Cinzel name ride the right rail. .muted is the
   no-edge state. #3a3450 is the mockup's celestial border (solid, not a gradient). */
.tonight-card{position:relative;overflow:hidden;display:flex;gap:14px;border:1px solid #3a3450;
  border-radius:4px;padding:15px 17px;margin:12px 0;text-decoration:none;color:inherit;
  cursor:pointer;transition:border-color .15s ease;
  background:radial-gradient(ellipse at 16% -45%, rgba(46,107,230,.16), transparent 55%),
    radial-gradient(ellipse at 93% 8%, rgba(201,162,39,.10), transparent 46%), rgba(26,29,36,.84)}
/* The card shell is a <div> (a block tag st.markdown passes verbatim; an outer <a> is not a
   CommonMark block tag, so the parser splits it into one box per child). A stretched, transparent
   <a> overlay (.tc-cardlink) turns the whole card into the View-game link; gold border on hover
   signals it clicks through. */
.tonight-card:hover{border-color:#C9A227}
.tc-cardlink{position:absolute;inset:0;z-index:1;border-radius:inherit}
.tonight-card.muted{border-color:#2A2E37;background:rgba(26,29,36,.82)}
.tonight-card .tc-main{flex:1;min-width:0}
.tonight-card .tc-side{flex:0 0 72px;display:flex;flex-direction:column;align-items:center;
  justify-content:center;gap:6px}
.tc-kicker{font-family:'Cinzel',serif;font-weight:600;font-size:10px;letter-spacing:.2em;
  text-transform:uppercase;color:#8A91A0}
/* A game tipping off soon turns its kicker red as a "time running out" cue. */
.tc-kicker.tc-urgent{color:#E5484D}
.tc-matchup{font-size:22px;font-weight:700;line-height:1.12;margin:3px 0 1px;color:#E6E9EF}
.tc-matchup .tc-vs{color:#8A91A0;font-weight:500}
.tc-prophecy{font-family:'Cormorant Garamond',serif;font-style:italic;font-weight:600;
  font-size:16px;color:#C9A227;margin:2px 0 0}
.tc-prophecy.tc-dim{color:#8A91A0}
.tc-foot{display:flex;align-items:center;gap:14px;margin-top:12px;flex-wrap:wrap}
.tc-badge{font-family:'IBM Plex Mono',monospace;padding:1px 8px;border-radius:3px;font-size:13px;
  border:1px solid;white-space:nowrap}
.tc-badge.tc-g{color:#1F9D55;border-color:#1F9D55}
.tc-badge.tc-gray{color:#8A91A0;border-color:#2A2E37}
.tc-meta{font-family:'IBM Plex Mono',monospace;font-size:12px;color:#8A91A0}
.tc-meta b{color:#E6E9EF;font-weight:500}
/* "View game" cue pushed to the right edge of the foot (the card itself is the link). */
.tc-viewcue{margin-left:auto;color:#C9A227;font-size:12px;font-weight:500;white-space:nowrap}
.tonight-card:hover .tc-viewcue{text-decoration:underline}
.tc-shapename{font-family:'Cinzel',serif;font-size:8.5px;letter-spacing:.14em;
  text-transform:uppercase;color:#8A91A0}
/* A little transparency + a soft gold outer glow so the glyphs read as embedded in the
   night sky rather than pasted on. Applied to every game-shape glyph (legend, cards, hero). */
.glyph{opacity:.9;filter:drop-shadow(0 0 5px rgba(201,162,39,.35))}
/* Games slip rail: the fixed-height selected-legs panel (slip_builder._LEG_PANEL_HEIGHT) framed as
   a gold celestial tablet -- a hairline gold border, a faint outer ring + inset glow, and a top-lit
   nebula wash (gold well under the §3 12% ceiling) so it reads as an inscribed plate, not a plain
   scroll box. The two-column read-only leg list flows inside it. */
.st-key-constellation_legpanel{border:1px solid rgba(201,162,39,.42);border-radius:4px;padding:8px 14px;
  background:radial-gradient(ellipse at 50% -30%, rgba(201,162,39,.07), transparent 62%), rgba(26,29,36,.5);
  box-shadow:0 0 0 3px rgba(201,162,39,.05), inset 0 0 12px rgba(201,162,39,.05), 0 2px 14px rgba(0,0,0,.35)}
/* DESIGN §2: gold marks the selected segment of a segmented_control (the global sport
   switch, Board lens/side, Receipts window, and the Games constellation lenses). Streamlit
   has no token slot for the active segment, so reach the gold token on its kind-scoped
   testid. The segmented_controlActive kind is Streamlit-1.58 internal -- reconfirm
   in-browser after any Streamlit bump. */
button[data-testid="stBaseButton-segmented_controlActive"]{
  color:#C9A227 !important;border-color:#C9A227 !important;background:rgba(201,162,39,.13) !important}
.starfield{position:fixed;inset:0;z-index:-1;pointer-events:none;
  background:
    __STARFIELD_DUST__}
/* Streamlit paints backgroundColor (#0E1117) as an opaque fill on its own root
   containers, which sits above this fixed z-index:-1 layer and hides it outright.
   Dropping just these two containers to transparent reveals the starfield (which
   now carries that same #0E1117 base itself) in the gutters; every card/table/
   metric widget keeps its own secondaryBackgroundColor fill untouched, so stars
   still don't bleed into content. !important because Streamlit's own theme CSS
   mounts via its frontend bundle after this injected <style> tag and otherwise
   wins the same-specificity tiebreak on source order. */
.stApp, [data-testid="stAppViewContainer"]{background:transparent !important}
/* Selective occlusion: the field shines through page whitespace and cards everywhere,
   occluded only behind a dense data table so its numbers stay readable (owner: "shine
   through everything except the data tables"). The AG Grid host is already an opaque
   iframe; st.dataframe is the one that needs a solid fill (DESIGN §3 "never behind dense
   tables or stat grids"). */
[data-testid="stDataFrame"]{background:#12151C}
/* app.py injects this via st.markdown(unsafe_allow_html=True) -- its DOMPurify allows
   <style>/<script> (ADD_TAGS) but still strips <svg> outright, so the original
   SVG-based starfield rendered an empty .starfield with zero children. (st.html is
   worse: USE_PROFILES:{html:true} drops <style> too, which is why this can't inject
   through st.html.) The dust above is plain radial-gradient layers for exactly that
   reason; these 11 twinkle accents are plain divs shaped with clip-path instead of
   <svg><use>, since both survive that sanitizer. */
.tw{position:absolute;width:1.4vmin;height:1.4vmin;background:#E6E9EF;
  clip-path:polygon(50% 3%,60.5% 39.5%,97% 50%,60.5% 60.5%,50% 97%,39.5% 60.5%,3% 50%,39.5% 39.5%)}
.tw.gold{width:1.7vmin;height:1.7vmin;background:#C9A227}
.sp1{left:7.35%;top:37.35%;animation-duration:3.8s;animation-delay:.1s}
.sp2{left:42.35%;top:43.35%;animation-duration:5.1s;animation-delay:1.3s}
.sp3{left:48.35%;top:23.35%;animation-duration:4.4s;animation-delay:.7s}
.sp4{left:62.35%;top:64.35%;animation-duration:5.6s;animation-delay:2.1s}
.sp5{left:93.35%;top:26.35%;animation-duration:3.3s;animation-delay:1.8s}
.sp6{left:34.35%;top:85.35%;animation-duration:4.7s;animation-delay:.4s}
.sp7{left:24.2%;top:51.2%;animation-duration:4.9s;animation-delay:.9s}
.sp8{left:51.2%;top:54.2%;animation-duration:3.6s;animation-delay:2.4s}
.sp9{left:67.2%;top:46.2%;animation-duration:5.3s;animation-delay:.2s}
.sp10{left:92.2%;top:51.2%;animation-duration:4.1s;animation-delay:1.5s}
.sp11{left:87.2%;top:6.2%;animation-duration:4.5s;animation-delay:3s}
@keyframes tw{0%,100%{opacity:.2}50%{opacity:.64}}
/* Longhands only, no animation-duration -- the .spN rules above set that per
   star. The old SVG version stashed per-star timing in an inline style=
   attribute (highest specificity, order-proof); as a class rule it collides
   with any later same-specificity animation shorthand, which resets every
   longhand it doesn't restate. Splitting this one keeps the two rule sets on
   disjoint properties so declaration order stops mattering. */
@media (prefers-reduced-motion: no-preference){.tw{animation-name:tw;animation-timing-function:ease-in-out;animation-iteration-count:infinite}}
@media (prefers-reduced-motion: reduce){.tw{animation:none;opacity:.15}}
</style>
"""

APP_CSS = _APP_CSS_TEMPLATE.replace("__STARFIELD_DUST__", _starfield_background())

# Dust + wash render via .starfield's own CSS background above; these 11 divs are
# just the twinkling accents (see the .tw comment in APP_CSS for why plain divs
# instead of <svg>). Position/timing per star lives in the .sp1..sp11 rules above.
STARFIELD_HTML = """
<div class="starfield" aria-hidden="true">
  <div class="tw sp1"></div>
  <div class="tw sp2"></div>
  <div class="tw sp3"></div>
  <div class="tw sp4"></div>
  <div class="tw sp5"></div>
  <div class="tw sp6"></div>
  <div class="tw gold sp7"></div>
  <div class="tw gold sp8"></div>
  <div class="tw gold sp9"></div>
  <div class="tw gold sp10"></div>
  <div class="tw gold sp11"></div>
</div>
"""
