"""The app-level CSS injection block: fonts, celestial classes, starfield."""

from pathlib import Path

APP = Path("src/sportstradamus/dashboard/app.py").read_text()
import sportstradamus.dashboard.theme as theme


def test_celestial_classes_defined():
    assert ".celestial-kicker" in theme.APP_CSS
    assert ".celestial-headline" in theme.APP_CSS
    assert "Cinzel" in theme.APP_CSS and "Cormorant Garamond" in theme.APP_CSS


def test_starfield_respects_ambient_rules():
    assert "prefers-reduced-motion" in theme.APP_CSS
    # every starfield opacity stays under the DESIGN §3 ambient ceiling
    import re
    for m in re.finditer(r"opacity:\s*\.?([\d.]+)", theme.STARFIELD_SVG):
        assert float(m.group(1)) <= 0.20 or float(m.group(1)) >= 1  # 1 = container reset


def test_app_injects_once():
    assert APP.count("st.html") + APP.count("unsafe_allow_html") == 1
