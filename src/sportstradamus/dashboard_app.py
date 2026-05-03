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

from sportstradamus.dashboard_data import PRED_BANNER_COLOR, STATS_BANNER_COLOR

st.set_page_config(page_title="Sportstradamus Dashboard", layout="wide")
st.title("Sportstradamus Dashboard")

left, right = st.columns(2)

with left:
    st.markdown(
        f"""
<div style="border:2px solid {PRED_BANNER_COLOR};border-radius:8px;padding:18px">
  <h3 style="color:{PRED_BANNER_COLOR};margin-top:0">🎯 Predictions</h3>
  <p>Today's scored offers and parlay candidates from the latest
  <code>prophecize</code> run.</p>
  <ul>
    <li><b>Today</b> — every offer with model EV, books prob, boost</li>
    <li><b>Parlays</b> — game-grouped expandables, sorted by EV</li>
  </ul>
</div>
""",
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        f"""
<div style="border:2px solid {STATS_BANNER_COLOR};border-radius:8px;padding:18px">
  <h3 style="color:{STATS_BANNER_COLOR};margin-top:0">📊 Stats</h3>
  <p>Historical accuracy, profit, calibration, and training-time
  diagnostics.</p>
  <ul>
    <li><b>Overview</b> — KPIs, accuracy & profit trends, volume</li>
    <li><b>Diagnostics</b> — per-market accuracy, calibration, CRPS</li>
    <li><b>Correlations</b> — parlay lift, hit rates, calibration</li>
    <li><b>Profit Sim</b> — Monte Carlo strategy backtests</li>
    <li><b>Model Training</b> — meditate diagnostics, HP, dispersion</li>
  </ul>
</div>
""",
        unsafe_allow_html=True,
    )
