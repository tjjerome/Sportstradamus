"""The app-level CSS injection block: fonts, celestial classes, starfield."""

import re
from pathlib import Path

from sportstradamus.dashboard import theme

APP = Path("src/sportstradamus/dashboard/app.py").read_text()


def test_celestial_classes_defined():
    assert ".celestial-kicker" in theme.APP_CSS
    assert ".celestial-headline" in theme.APP_CSS
    assert "Cinzel" in theme.APP_CSS and "Cormorant Garamond" in theme.APP_CSS


def test_starfield_respects_ambient_rules():
    assert "prefers-reduced-motion" in theme.APP_CSS
    # DESIGN §3: every static dust/wash alpha in the .starfield rule stays at or under
    # the 0.20 ambient ceiling. The animated twinkle accents are the sanctioned
    # exception (owner decision) and are bounded separately below.
    starfield_rule = re.search(r"\.starfield\{(.*?)\}", theme.APP_CSS, re.DOTALL)
    assert starfield_rule is not None
    for m in re.finditer(
        r"rgba\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*(\.?[\d.]+)\s*\)", starfield_rule.group(1)
    ):
        assert float(m.group(1)) <= 0.20
    # Twinkle @keyframes peak stays <= 0.70 (exempt from the static ceiling, still bounded).
    keyframes = re.search(r"@keyframes tw\{.*?\}\}", theme.APP_CSS)
    assert keyframes is not None
    peaks = [float(x) for x in re.findall(r"opacity:\s*(\.?[\d.]+)", keyframes.group(0))]
    assert peaks and max(peaks) <= 0.70


def test_page_hero_defined_and_banners_retired():
    # The page-hero header (components/hero.py) replaced the Sheets-era render_banner banners.
    assert ".page-hero" in theme.APP_CSS
    assert ".hero-updated" in theme.APP_CSS
    assert "banner-predictions" not in theme.APP_CSS
    assert "banner-stats" not in theme.APP_CSS
    # Gold marks the selected segmented-control segment (DESIGN §2 active-state role).
    assert "segmented_controlActive" in theme.APP_CSS


def test_app_injects_css_once_via_markdown():
    # The global <style> block must be injected via st.markdown(unsafe_allow_html=True),
    # NOT st.html: st.html's DOMPurify (USE_PROFILES {html:true}) strips <style> outright,
    # so APP_CSS silently no-ops and the whole celestial layer renders flat. Pin the
    # working mechanism so a revert to st.html is caught here, not in the browser.
    assert APP.count("unsafe_allow_html=True") == 1
    assert "st.html(" not in APP
