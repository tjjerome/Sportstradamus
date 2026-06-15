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
