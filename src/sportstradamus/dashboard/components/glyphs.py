"""Bespoke celestial glyphs for game shape — the comet unification (P8 Phase C, Task C3).

One construction rule across all five: strokes build the form, a glowing radial core
marks the focus. Ported from ``docs/mockups/p8-glyphs.html`` (gradient/symbol ``id``s
renamed, namespaced per glyph type so two instances of different glyphs never collide
on one page); the comet was later reworked (curved dust tail, larger coma) per owner
so it reads as a meteor rather than a plain slash. These are bespoke inline SVG — the
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

_LONE_STAR = (
    '<svg viewBox="0 0 60 60"><defs><radialGradient id="glyph-lonestar-core" cx="50%" '
    'cy="50%" r="50%"><stop offset="0" stop-color="#fff"/><stop offset=".42" '
    'stop-color="#C9A227"/><stop offset="1" stop-color="#6f560f"/></radialGradient>'
    '<symbol id="glyph-lonestar-star" viewBox="0 0 24 24"><path d="M12 1.6l3.09 6.9 '
    '7.51.7-5.66 5.02 1.68 7.38L12 17.9l-6.62 3.7 1.68-7.38L1.4 9.2l7.51-.7z"/></symbol>'
    "</defs>"
    '<circle cx="30" cy="30" r="20" fill="none" stroke="#8A91A0" stroke-width="1" '
    'stroke-opacity=".26"/>'
    '<use href="#glyph-lonestar-star" x="11" y="11" width="38" height="38" fill="none" '
    'stroke="#C9A227" stroke-width="1.4" stroke-linejoin="round"/>'
    '<circle cx="30" cy="30" r="4.6" fill="url(#glyph-lonestar-core)"/></svg>'
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
    "even": _LONE_STAR,
    "grind": _HOURGLASS,
}


def game_shape_glyph(shape: str, *, size: int = 40) -> str:
    """Inline SVG for a game shape; unknown shapes fall back to the lone star."""
    svg = _GLYPHS.get(shape, _LONE_STAR)
    return svg.replace(
        'viewBox="0 0 60 60"',
        f'class="glyph" width="{size}" height="{size}" viewBox="0 0 60 60"',
        1,
    )
