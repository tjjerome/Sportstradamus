# ARCHIVED 2026-10-15 from src/sportstradamus/dashboard/surfaces/lab_diagnostics_charts.py
# Reason: Nothing fills a CRPS column since the dashboard-side resolve was archived in
#         June 2026 (production's history has none), so the Lab Diagnostics CRPS panel
#         showed nothing. The surface's CRPS column and "CRPS Over Time" section were
#         removed with it, and the chart's two tests deleted from
#         tests/golden/test_lab_diagnostics_charts.py (git keeps them).
# Last live SHA: 142f73e9
# Original imports (now unresolved here):
#   import pandas as pd
#   import plotly.express as px
#   import plotly.graph_objects as go

# From src/sportstradamus/dashboard/surfaces/lab_diagnostics_charts.py.
#
def crps_line(crps_daily: pd.DataFrame) -> go.Figure:
    """Line chart of mean CRPS over time by league."""
    fig = px.line(
        crps_daily,
        x="_date",
        y="CRPS",
        color="League",
        labels={"_date": "Date", "CRPS": "Mean CRPS (lower=better)"},
    )
    fig.update_layout(height=400)
    return fig
