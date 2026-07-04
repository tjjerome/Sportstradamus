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
    # every dust/wash alpha in the .starfield rule stays under the DESIGN §3
    # ambient ceiling (the animated twinkle-accent peak is a separate, already
    # flagged question -- not this gate's scope, see the P8 Phase A code review)
    starfield_rule = re.search(r"\.starfield\{(.*?)\}", theme.APP_CSS, re.DOTALL)
    assert starfield_rule is not None
    for m in re.finditer(r"rgba\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*(\.?[\d.]+)\s*\)", starfield_rule.group(1)):
        assert float(m.group(1)) <= 0.20


def test_app_injects_once():
    assert APP.count("st.html") + APP.count("unsafe_allow_html") == 1
