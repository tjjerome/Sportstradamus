"""Sportstradamus Dashboard — main Streamlit app.

This file serves as the entry point for Streamlit's multi-page app.
Pages in the pages/ directory are auto-discovered and shown in the sidebar.
"""
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
