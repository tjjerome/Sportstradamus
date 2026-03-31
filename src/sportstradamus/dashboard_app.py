"""Sportstradamus Dashboard — main Streamlit app.

This file serves as the entry point for Streamlit's multi-page app.
Pages in the pages/ directory are auto-discovered and shown in the sidebar.
"""
import sys
from pathlib import Path

# Ensure `src/` is before `src/sportstradamus/` in sys.path so that
# `import sportstradamus` resolves to the package, not sportstradamus.py.
_src_dir = str(Path(__file__).parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import streamlit as st

st.set_page_config(page_title="Sportstradamus Dashboard", layout="wide")
st.title("Sportstradamus Dashboard")
st.markdown("""
Welcome to the Sportstradamus analysis dashboard. Use the sidebar to navigate between pages:

- **Overview** — KPIs, accuracy trends, profit trends, prediction volume
- **Market Diagnostics** — Per-market accuracy, bias detection, calibration, CRPS, coverage
- **Correlations & Parlays** — Correlation value-add analysis, hit rates by size, parlay calibration
- **Profit Simulation** — Monte Carlo strategy backtesting with preset and custom strategies
""")
