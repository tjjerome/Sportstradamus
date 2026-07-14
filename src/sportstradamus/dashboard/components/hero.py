"""The page-hero header: the mockups' `.head` block (celestial kicker + title + timestamp)."""

import streamlit as st


def page_hero(kicker: str, title: str, updated: str | None = None) -> None:
    """Render a surface header: gold Cinzel kicker, plain display title, quiet updated line.

    The kicker is the only celestial (gold) element; the title stays broadcast-sober sans
    (the Cormorant `.celestial-headline` is reserved for prophecy lines). ``updated`` is a
    bare timestamp with no label text (the dashboard timestamp-line rule).
    """
    updated_html = f'<div class="hero-updated">{updated}</div>' if updated else ""
    st.html(
        '<div class="page-hero">'
        f'<div class="celestial-kicker">◈ {kicker}</div>'
        f'<h1 class="hero-title">{title}</h1>'
        f"{updated_html}</div>"
    )
