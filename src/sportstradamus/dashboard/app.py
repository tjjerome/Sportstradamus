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

st.set_page_config(page_title="Sportstradamus Dashboard", layout="wide")

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

# st.Page paths resolve relative to app.py's directory.
_surfaces = Path(__file__).parent / "surfaces"

# Six surfaces live top-level (unnamed section renders headerless);
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
                str(_surfaces / "game.py"),
                title="Game",
                icon=":material/sports_basketball:",
                url_path="game",
            ),
            st.Page(
                str(_surfaces / "board.py"),
                title="Board",
                icon=":material/storefront:",
                url_path="board",
            ),
            st.Page(
                str(_surfaces / "slips.py"),
                title="Slips",
                icon=":material/receipt_long:",
                url_path="slips",
            ),
            st.Page(
                str(_surfaces / "receipts.py"),
                title="Receipts",
                icon=":material/verified:",
                url_path="receipts",
            ),
            st.Page(
                str(_surfaces / "receipts_sim.py"),
                title="Profit Sim",
                icon=":material/bar_chart:",
                url_path="receipts-sim",
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

pg.run()
