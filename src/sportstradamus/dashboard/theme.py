"""Non-Streamlit mirror of the dashboard design tokens (DESIGN.md §2)."""

import importlib.resources as pkg_resources
import json
from functools import cache

import plotly.graph_objects as go
import plotly.io as pio

from sportstradamus import data

# Celestial gold accent — oracle decoration (kickers, prophecy, constellation) and
# interactive-highlight (active/selected/hovered UI states); never a data mark except
# the constellation correlation edge (DESIGN §4a).
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


# The one sanctioned CSS injection site for the whole dashboard (DESIGN.md §3):
# display-only celestial fonts (.celestial-kicker / .celestial-headline), the
# banner classes render_banner() uses, and the ambient starfield layer.
# app.py injects this once, right after st.set_page_config.
APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;600&family=Cormorant+Garamond:ital,wght@1,600&display=swap');
.celestial-kicker{font-family:'Cinzel',serif;font-weight:600;font-size:10.5px;
  letter-spacing:.26em;text-transform:uppercase;color:#C9A227}
.celestial-headline{font-family:'Cormorant Garamond',serif;font-style:italic;
  font-weight:600;color:#C9A227}
.banner-predictions{background:#1f4e79;padding:10px 14px;border-radius:6px;
  color:white;margin-bottom:14px;font-size:14px}
.banner-stats{background:#2d6a4f;padding:10px 14px;border-radius:6px;
  color:white;margin-bottom:14px;font-size:14px}
.starfield{position:fixed;inset:0;z-index:-1;pointer-events:none;
  background:
    radial-gradient(circle at 4% 7%, rgba(230,233,239,.18) 0 2px, transparent 2px),
    radial-gradient(circle at 11% 19%, rgba(230,233,239,.12) 0 2px, transparent 2px),
    radial-gradient(circle at 14% 60%, rgba(230,233,239,.16) 0 2px, transparent 2px),
    radial-gradient(circle at 6% 76%, rgba(230,233,239,.20) 0 2px, transparent 2px),
    radial-gradient(circle at 19% 88%, rgba(230,233,239,.10) 0 2px, transparent 2px),
    radial-gradient(circle at 23% 11%, rgba(230,233,239,.18) 0 2px, transparent 2px),
    radial-gradient(circle at 27% 34%, rgba(230,233,239,.12) 0 2px, transparent 2px),
    radial-gradient(circle at 31% 68%, rgba(230,233,239,.14) 0 2px, transparent 2px),
    radial-gradient(circle at 39% 14%, rgba(230,233,239,.12) 0 2px, transparent 2px),
    radial-gradient(circle at 37% 58%, rgba(230,233,239,.10) 0 2px, transparent 2px),
    radial-gradient(circle at 45% 77%, rgba(230,233,239,.16) 0 2px, transparent 2px),
    radial-gradient(circle at 47% 92%, rgba(230,233,239,.10) 0 2px, transparent 2px),
    radial-gradient(circle at 55% 9%, rgba(230,233,239,.18) 0 2px, transparent 2px),
    radial-gradient(circle at 59% 37%, rgba(230,233,239,.14) 0 2px, transparent 2px),
    radial-gradient(circle at 53% 81%, rgba(230,233,239,.10) 0 2px, transparent 2px),
    radial-gradient(circle at 67% 19%, rgba(230,233,239,.16) 0 2px, transparent 2px),
    radial-gradient(circle at 73% 71%, rgba(230,233,239,.12) 0 2px, transparent 2px),
    radial-gradient(circle at 66% 89%, rgba(230,233,239,.08) 0 2px, transparent 2px),
    radial-gradient(circle at 77% 29%, rgba(230,233,239,.18) 0 2px, transparent 2px),
    radial-gradient(circle at 81% 57%, rgba(230,233,239,.14) 0 2px, transparent 2px),
    radial-gradient(circle at 84% 14%, rgba(230,233,239,.20) 0 2px, transparent 2px),
    radial-gradient(circle at 88% 43%, rgba(230,233,239,.12) 0 2px, transparent 2px),
    radial-gradient(circle at 80% 80%, rgba(230,233,239,.16) 0 2px, transparent 2px),
    radial-gradient(circle at 90% 67%, rgba(230,233,239,.10) 0 2px, transparent 2px),
    radial-gradient(circle at 96% 86%, rgba(230,233,239,.10) 0 2px, transparent 2px),
    radial-gradient(circle at 3% 52%, rgba(230,233,239,.14) 0 2px, transparent 2px),
    radial-gradient(circle at 2% 28%, rgba(230,233,239,.08) 0 2px, transparent 2px),
    radial-gradient(circle at 58% 95%, rgba(230,233,239,.12) 0 2px, transparent 2px),
    radial-gradient(circle at 33% 5%, rgba(230,233,239,.10) 0 2px, transparent 2px),
    radial-gradient(circle at 70% 10%, rgba(230,233,239,.12) 0 2px, transparent 2px),
    radial-gradient(ellipse at 18% 4%, rgba(46,107,230,.10), transparent 46%),
    radial-gradient(ellipse at 86% 22%, rgba(201,162,39,.05), transparent 42%),
    radial-gradient(ellipse at 50% 96%, rgba(46,107,230,.06), transparent 55%),
    #0E1117}
/* Streamlit paints backgroundColor (#0E1117) as an opaque fill on its own root
   containers, which sits above this fixed z-index:-1 layer and hides it outright.
   Dropping just these two containers to transparent reveals the starfield (which
   now carries that same #0E1117 base itself) in the gutters; every card/table/
   metric widget keeps its own secondaryBackgroundColor fill untouched, so stars
   still don't bleed into content. !important because Streamlit's own theme CSS
   mounts via its frontend bundle after this injected <style> tag and otherwise
   wins the same-specificity tiebreak on source order. */
.stApp, [data-testid="stAppViewContainer"]{background:transparent !important}
/* app.py's st.html() runs every fragment through DOMPurify with
   USE_PROFILES:{html:true} and no svg addition (confirmed in the bundled
   frontend's Html.*.js) -- it silently strips <svg> outright, so the original
   SVG-based starfield rendered an empty .starfield with zero children. The dust
   above is plain radial-gradient layers for exactly that reason; these 11
   twinkle accents are plain divs shaped with clip-path instead of <svg><use>,
   since both survive that sanitizer. */
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
