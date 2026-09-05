"""Bespoke celestial glyphs for game shape — the comet unification (P8 Phase C, Task C3).

One construction rule across all five: strokes build the form, a glowing radial core
marks the focus. Ported from ``docs/mockups/p8-glyphs.html`` (gradient/symbol ``id``s
renamed, namespaced per glyph type so two instances of different glyphs never collide
on one page); the comet was later reworked (curved dust tail, larger coma) per owner
so it reads as a meteor rather than a plain slash, and the ``even`` lone star became a
nebula (display name "clouded") so a game nobody quoted reads as uncertain rather
than as a game script of its own — its form is gas, blurred gradient lobes with
tendrils and pinprick stars around the core, so it reads as space, not as a
weather-app cloud. These are bespoke inline SVG — the
sanctioned exception to Material-icons-only (spec §2); render via
``st.markdown(..., unsafe_allow_html=True)``, never ``st.html()`` (see
``dashboard/theme.py``'s DOMPurify comment — ``st.html()`` strips ``<svg>`` outright).
"""

from __future__ import annotations

_COMET = (
    '<svg viewBox="0 0 60 60"><defs><radialGradient id="glyph-comet-core" cx="64%" '
    'cy="34%" r="62%"><stop offset="0" stop-color="#fff"/><stop offset=".42" '
    'stop-color="#C9A227"/><stop offset="1" stop-color="#7a5f18"/></radialGradient></defs>'
    '<path d="M43 17 Q27 27 9 50" stroke="#C9A227" stroke-width="2.8" stroke-linecap="round" '
    'fill="none" opacity=".85"/>'
    '<path d="M45 13 Q31 21 14 45" stroke="#C9A227" stroke-width="1.2" stroke-linecap="round" '
    'fill="none" opacity=".5"/>'
    '<path d="M42 22 Q30 33 16 51" stroke="#C9A227" stroke-width="1" stroke-linecap="round" '
    'fill="none" opacity=".42"/>'
    '<circle cx="44" cy="16" r="10" fill="#C9A227" opacity=".16"/>'
    '<circle cx="44" cy="16" r="8" fill="url(#glyph-comet-core)"/></svg>'
)

_SUPERNOVA = (
    '<svg viewBox="0 0 60 60"><defs><radialGradient id="glyph-supernova-core" cx="50%" '
    'cy="50%" r="50%"><stop offset="0" stop-color="#fff"/><stop offset=".4" '
    'stop-color="#C9A227"/><stop offset="1" stop-color="#6f560f"/></radialGradient></defs>'
    '<circle cx="30" cy="30" r="21" fill="none" stroke="#C9A227" stroke-width="1" '
    'stroke-opacity=".14"/>'
    '<circle cx="30" cy="30" r="13.5" fill="none" stroke="#C9A227" stroke-width="1" '
    'stroke-opacity=".1"/>'
    '<g stroke="#C9A227" stroke-linecap="round">'
    '<line x1="30" y1="30" x2="52" y2="30" stroke-width="1.4" stroke-opacity=".8"/>'
    '<line x1="30" y1="30" x2="42.99" y2="37.5" stroke-width="1.1" stroke-opacity=".55"/>'
    '<line x1="30" y1="30" x2="42" y2="50.78" stroke-width="2.2" stroke-opacity=".85"/>'
    '<line x1="30" y1="30" x2="30" y2="47" stroke-width="1.1" stroke-opacity=".55"/>'
    '<line x1="30" y1="30" x2="23" y2="42.12" stroke-width="1.1" stroke-opacity=".5"/>'
    '<line x1="30" y1="30" x2="10.08" y2="41.5" stroke-width="2.2" stroke-opacity=".85"/>'
    '<line x1="30" y1="30" x2="14" y2="30" stroke-width="1.4" stroke-opacity=".8"/>'
    '<line x1="30" y1="30" x2="8.35" y2="17.5" stroke-width="2.4" stroke-opacity=".9"/>'
    '<line x1="30" y1="30" x2="22.5" y2="17.01" stroke-width="1.1" stroke-opacity=".5"/>'
    '<line x1="30" y1="30" x2="30" y2="10" stroke-width="1.6" stroke-opacity=".8"/>'
    '<line x1="30" y1="30" x2="36.5" y2="18.74" stroke-width="1.1" stroke-opacity=".5"/>'
    '<line x1="30" y1="30" x2="48.19" y2="19.5" stroke-width="2.2" stroke-opacity=".85"/></g>'
    '<circle cx="30" cy="30" r="6.5" fill="url(#glyph-supernova-core)"/></svg>'
)

_SCALES = (
    '<svg viewBox="0 0 60 60"><defs><radialGradient id="glyph-scales-core" cx="50%" '
    'cy="45%" r="55%"><stop offset="0" stop-color="#fff"/><stop offset=".4" '
    'stop-color="#C9A227"/><stop offset="1" stop-color="#6f560f"/></radialGradient></defs>'
    '<g stroke="#C9A227" stroke-width="1.6" stroke-linecap="round" fill="none">'
    '<line x1="12" y1="23" x2="48" y2="23"/><line x1="30" y1="23" x2="30" y2="42"/>'
    '<line x1="24" y1="46" x2="36" y2="46"/>'
    '<line x1="12" y1="23" x2="12" y2="28"/><line x1="48" y1="23" x2="48" y2="28"/></g>'
    '<circle cx="12" cy="33" r="5.5" fill="none" stroke="#C9A227" stroke-width="1.5"/>'
    '<circle cx="48" cy="33" r="5.5" fill="none" stroke="#C9A227" stroke-width="1.5"/>'
    '<circle cx="30" cy="23" r="3.6" fill="url(#glyph-scales-core)"/></svg>'
)

_NEBULA = (
    '<svg viewBox="0 0 60 60"><defs><radialGradient id="glyph-nebula-core" cx="50%" '
    'cy="50%" r="50%"><stop offset="0" stop-color="#fff"/><stop offset=".45" '
    'stop-color="#C9A227"/><stop offset="1" stop-color="#6f560f"/></radialGradient>'
    '<radialGradient id="glyph-nebula-gas" cx="50%" cy="50%" r="50%">'
    '<stop offset="0" stop-color="#8A91A0" stop-opacity=".4"/>'
    '<stop offset=".5" stop-color="#8A91A0" stop-opacity=".16"/>'
    '<stop offset="1" stop-color="#8A91A0" stop-opacity="0"/></radialGradient>'
    '<radialGradient id="glyph-nebula-glow" cx="50%" cy="50%" r="50%">'
    '<stop offset="0" stop-color="#C9A227" stop-opacity=".36"/>'
    '<stop offset=".5" stop-color="#C9A227" stop-opacity=".12"/>'
    '<stop offset="1" stop-color="#C9A227" stop-opacity="0"/></radialGradient>'
    '<filter id="glyph-nebula-soft" x="-20%" y="-20%" width="140%" height="140%">'
    '<feGaussianBlur stdDeviation="1.4"/></filter></defs>'
    '<g filter="url(#glyph-nebula-soft)" fill="url(#glyph-nebula-gas)">'
    '<ellipse cx="28" cy="31" rx="21" ry="12" transform="rotate(-22 28 31)"/>'
    '<ellipse cx="38" cy="25" rx="13" ry="8" transform="rotate(28 38 25)"/>'
    '<ellipse cx="19" cy="37" rx="11" ry="7" transform="rotate(40 19 37)"/>'
    '<ellipse cx="26" cy="21" rx="8" ry="5" transform="rotate(-10 26 21)"/>'
    '<ellipse cx="42" cy="38" rx="9" ry="5" transform="rotate(15 42 38)"/>'
    '<ellipse cx="13" cy="24" rx="10" ry="2.6" transform="rotate(-35 13 24)"/>'
    '<ellipse cx="46" cy="38" rx="9" ry="2.4" transform="rotate(12 46 38)"/>'
    '<ellipse cx="44" cy="19" rx="6" ry="3.4" transform="rotate(-30 44 19)"/>'
    '<ellipse cx="31" cy="30" rx="12" ry="8" transform="rotate(-14 31 30)" '
    'fill="url(#glyph-nebula-glow)"/></g>'
    '<g fill="none" stroke="#8A91A0" stroke-linecap="round">'
    '<path d="M10 30 C15 21 24 19 31 24" stroke-width="1.1" stroke-opacity=".28"/>'
    '<path d="M37 27 C43 21 50 20 54 24" stroke-width=".9" stroke-opacity=".22"/>'
    '<path d="M22 41 C27 46 35 46 43 43" stroke-width=".8" stroke-opacity=".16"/></g>'
    '<g fill="#8A91A0"><circle cx="12" cy="16" r=".9" opacity=".7"/>'
    '<circle cx="50" cy="17" r=".7" opacity=".5"/><circle cx="52" cy="44" r=".8" opacity=".6"/>'
    '<circle cx="15" cy="48" r=".7" opacity=".45"/>'
    '<circle cx="44" cy="48" r=".6" opacity=".4"/></g>'
    '<circle cx="41" cy="22" r=".8" fill="#C9A227" opacity=".55"/>'
    '<circle cx="30" cy="30" r="8" fill="#C9A227" opacity=".14"/>'
    '<circle cx="30" cy="30" r="3.6" fill="url(#glyph-nebula-core)" opacity=".85"/></svg>'
)

_HOURGLASS = (
    '<svg viewBox="0 0 60 60"><defs><radialGradient id="glyph-hourglass-core" cx="50%" '
    'cy="50%" r="55%"><stop offset="0" stop-color="#fff"/><stop offset=".4" '
    'stop-color="#C9A227"/><stop offset="1" stop-color="#6f560f"/></radialGradient></defs>'
    '<g stroke="#C9A227" stroke-width="1.6" fill="none" stroke-linecap="round" '
    'stroke-linejoin="round">'
    '<line x1="17" y1="13" x2="43" y2="13"/><line x1="17" y1="47" x2="43" y2="47"/>'
    '<path d="M19 13 L30 30 L19 47"/><path d="M41 13 L30 30 L41 47"/></g>'
    '<g fill="#C9A227"><circle cx="30" cy="35" r="1.1"/><circle cx="30" cy="39.5" r="1"/>'
    '<circle cx="30" cy="43.5" r=".9"/></g>'
    '<circle cx="30" cy="30" r="3.4" fill="url(#glyph-hourglass-core)"/></svg>'
)

_GLYPHS = {
    "shootout": _COMET,
    "blowout": _SUPERNOVA,
    "coinflip": _SCALES,
    "even": _NEBULA,
    "grind": _HOURGLASS,
}


def game_shape_glyph(shape: str, *, size: int = 40) -> str:
    """Inline SVG for a game shape; unknown shapes fall back to the clouded glyph."""
    svg = _GLYPHS.get(shape, _NEBULA)
    return svg.replace(
        'viewBox="0 0 60 60"',
        f'class="glyph" width="{size}" height="{size}" viewBox="0 0 60 60"',
        1,
    )
