"""Sportstradamus Dashboard — main Streamlit app.

Single entry point for st.navigation. All surfaces register here;
pages are script-style files under surfaces/ loaded via st.Page.
"""

import sys
from pathlib import Path

# Ensure `src/` is on sys.path so `import sportstradamus` resolves to the
# installed package when Streamlit runs this file directly.
_src_dir = str(Path(__file__).parent.parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import streamlit as st

from sportstradamus.dashboard import theme
from sportstradamus.dashboard.components.deep_dive import drop_detail_on_page_change
from sportstradamus.dashboard.components.locked_shelf import render_locked_shelf
from sportstradamus.dashboard.components.slip_state import init_slip_state

st.set_page_config(page_title="Sportstradamus Dashboard", layout="wide")

st.html(theme.APP_CSS + theme.STARFIELD_HTML)
theme.register_plotly_template()

# Slip state is shared across surfaces; seed it before any surface renders.
init_slip_state()

# Global sport switch: stored under plain key so surfaces read it without the
# widget-key indirection. Copy widget value into the plain key each run.
_SPORT_OPTIONS = ["All", "WNBA", "MLB", "NBA", "NFL", "NHL"]
_sport_widget = st.segmented_control(
    "Sport",
    _SPORT_OPTIONS,
    default="All",
    key="_sport_widget",
)
st.session_state["sport"] = _sport_widget if _sport_widget is not None else "All"

_surfaces = Path(__file__).parent / "surfaces"

# Four surfaces live top-level (unnamed section renders headerless);
# only Model Lab gets a group header.
pg = st.navigation(
    {
        "": [
            st.Page(
                str(_surfaces / "tonight.py"),
                title="Tonight",
                icon=":material/dark_mode:",
                default=True,
                url_path="tonight",
            ),
            st.Page(
                str(_surfaces / "board.py"),
                title="Board",
                icon=":material/storefront:",
                url_path="board",
            ),
            st.Page(
                str(_surfaces / "games.py"),
                title="Games",
                icon=":material/sports_basketball:",
                url_path="games",
            ),
            st.Page(
                str(_surfaces / "receipts.py"),
                title="Receipts",
                icon=":material/verified:",
                url_path="receipts",
            ),
        ],
        "Model Lab": [
            st.Page(
                str(_surfaces / "lab_diagnostics.py"),
                title="Diagnostics",
                icon=":material/science:",
                url_path="lab-diagnostics",
            ),
            st.Page(
                str(_surfaces / "lab_correlations.py"),
                title="Correlations",
                icon=":material/hub:",
                url_path="lab-correlations",
            ),
            st.Page(
                str(_surfaces / "lab_training.py"),
                title="Training",
                icon=":material/model_training:",
                url_path="lab-training",
            ),
        ],
    }
)

with st.sidebar:
    render_locked_shelf()

# Close any offer-detail dialog left open on a previous surface (detail_stack is
# global session state and would otherwise re-open on the next dialog-owning page).
drop_detail_on_page_change(st.session_state, pg.url_path)

pg.run()
