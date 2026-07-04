"""Non-Streamlit mirror of the dashboard design tokens (DESIGN.md §2)."""

# Celestial gold accent — display faces and ambient art only, never data ink
# (DESIGN.md celestial layer; pinned by tests/golden/test_design_tokens.py).
GOLD = "#C9A227"

# Neutral gray (DESIGN.md §2 grayColor) — context/secondary marks, e.g. the
# constellation's non-slip legs against the gold thesis stars.
GRAY = "#8A91A0"

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


def register_plotly_template() -> None:
    """Register + default the token template so ad-hoc Plotly figures inherit DESIGN.md.

    Mirrors config.toml chartCategoricalColors as the colorway; Streamlit's own theming
    covers st.plotly_chart when a figure's own template/colors aren't set, but ad-hoc
    go.Figure() traces (e.g. profit_sim's per-strategy lines) need an explicit default.
    """
    import plotly.graph_objects as go
    import plotly.io as pio

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
    radial-gradient(ellipse at 18% 4%, rgba(46,107,230,.10), transparent 46%),
    radial-gradient(ellipse at 86% 22%, rgba(201,162,39,.05), transparent 42%),
    radial-gradient(ellipse at 50% 96%, rgba(46,107,230,.06), transparent 55%)}
.starfield svg{position:absolute;inset:0}
@keyframes tw{0%,100%{opacity:.2}50%{opacity:.64}}
@media (prefers-reduced-motion: no-preference){.tw{animation:tw 4s ease-in-out infinite}}
@media (prefers-reduced-motion: reduce){.tw{animation:none;opacity:.15}}
</style>
"""

# primary (layered z-index:-1 SVG) — injection wiring curl-verified; live visual/stacking-context
# confirmation pending owner's browser check
STARFIELD_SVG = """
<div class="starfield" aria-hidden="true">
  <svg viewBox="0 0 100 100" preserveAspectRatio="xMidYMid slice" width="100%" height="100%">
    <defs><symbol id="spk" viewBox="0 0 10 10"><path d="M5 0.3 L6.05 3.95 L9.7 5 L6.05 6.05 L5 9.7 L3.95 6.05 L0.3 5 L3.95 3.95 Z"/></symbol></defs>
    <g fill="#E6E9EF">
      <circle cx="4" cy="7" r="0.3" fill-opacity=".18"/><circle cx="11" cy="19" r="0.22" fill-opacity=".12"/>
      <circle cx="14" cy="60" r="0.24" fill-opacity=".16"/><circle cx="6" cy="76" r="0.3" fill-opacity=".20"/>
      <circle cx="19" cy="88" r="0.2" fill-opacity=".10"/><circle cx="23" cy="11" r="0.28" fill-opacity=".18"/>
      <circle cx="27" cy="34" r="0.2" fill-opacity=".12"/><circle cx="31" cy="68" r="0.24" fill-opacity=".14"/>
      <circle cx="39" cy="14" r="0.22" fill-opacity=".12"/><circle cx="37" cy="58" r="0.2" fill-opacity=".10"/>
      <circle cx="45" cy="77" r="0.26" fill-opacity=".16"/><circle cx="47" cy="92" r="0.2" fill-opacity=".10"/>
      <circle cx="55" cy="9" r="0.28" fill-opacity=".18"/><circle cx="59" cy="37" r="0.24" fill-opacity=".14"/>
      <circle cx="53" cy="81" r="0.2" fill-opacity=".10"/><circle cx="67" cy="19" r="0.26" fill-opacity=".16"/>
      <circle cx="73" cy="71" r="0.22" fill-opacity=".12"/><circle cx="66" cy="89" r="0.2" fill-opacity=".08"/>
      <circle cx="77" cy="29" r="0.28" fill-opacity=".18"/><circle cx="81" cy="57" r="0.24" fill-opacity=".14"/>
      <circle cx="84" cy="14" r="0.3" fill-opacity=".20"/><circle cx="88" cy="43" r="0.22" fill-opacity=".12"/>
      <circle cx="80" cy="80" r="0.26" fill-opacity=".16"/><circle cx="90" cy="67" r="0.2" fill-opacity=".10"/>
      <circle cx="96" cy="86" r="0.2" fill-opacity=".10"/><circle cx="3" cy="52" r="0.24" fill-opacity=".14"/>
      <circle cx="2" cy="28" r="0.2" fill-opacity=".08"/><circle cx="58" cy="95" r="0.22" fill-opacity=".12"/>
      <circle cx="33" cy="5" r="0.2" fill-opacity=".10"/><circle cx="70" cy="10" r="0.22" fill-opacity=".12"/>
    </g>
    <use href="#spk" x="7.35" y="37.35" width="1.3" height="1.3" fill="#E6E9EF" class="tw" style="animation-duration:3.8s;animation-delay:.1s"/>
    <use href="#spk" x="42.35" y="43.35" width="1.3" height="1.3" fill="#E6E9EF" class="tw" style="animation-duration:5.1s;animation-delay:1.3s"/>
    <use href="#spk" x="48.35" y="23.35" width="1.3" height="1.3" fill="#E6E9EF" class="tw" style="animation-duration:4.4s;animation-delay:.7s"/>
    <use href="#spk" x="62.35" y="64.35" width="1.3" height="1.3" fill="#E6E9EF" class="tw" style="animation-duration:5.6s;animation-delay:2.1s"/>
    <use href="#spk" x="93.35" y="26.35" width="1.3" height="1.3" fill="#E6E9EF" class="tw" style="animation-duration:3.3s;animation-delay:1.8s"/>
    <use href="#spk" x="34.35" y="85.35" width="1.3" height="1.3" fill="#E6E9EF" class="tw" style="animation-duration:4.7s;animation-delay:.4s"/>
    <use href="#spk" x="24.2" y="51.2" width="1.6" height="1.6" fill="#C9A227" class="tw" style="animation-duration:4.9s;animation-delay:.9s"/>
    <use href="#spk" x="51.2" y="54.2" width="1.6" height="1.6" fill="#C9A227" class="tw" style="animation-duration:3.6s;animation-delay:2.4s"/>
    <use href="#spk" x="67.2" y="46.2" width="1.6" height="1.6" fill="#C9A227" class="tw" style="animation-duration:5.3s;animation-delay:.2s"/>
    <use href="#spk" x="92.2" y="51.2" width="1.6" height="1.6" fill="#C9A227" class="tw" style="animation-duration:4.1s;animation-delay:1.5s"/>
    <use href="#spk" x="87.2" y="6.2" width="1.6" height="1.6" fill="#C9A227" class="tw" style="animation-duration:4.5s;animation-delay:3s"/>
  </svg>
</div>
"""
