"""Chart builder for the Lab Training run-at-a-glance strip."""

import plotly.graph_objects as go

_FUNNEL_STAGES = ("Trained", "G1 pass", "All gates (ship)", "Graduated live")


def lifecycle_funnel(*, trained: int, g1_pass: int, all_gates: int, graduated: int) -> go.Figure:
    """Horizontal funnel of the four lifecycle stage counts, rendered as given.

    The four counts come from different formulas (row count; the six-gate system's
    ``g1_pass``/``ship`` columns; ``classify_lifecycle``'s older proxy for
    ``lifecycle_state``), so a real slate is not guaranteed to shrink monotonically
    stage over stage — this builder does not clamp, sort, or otherwise enforce it.
    """
    fig = go.Figure(go.Funnel(y=list(_FUNNEL_STAGES), x=[trained, g1_pass, all_gates, graduated]))
    fig.update_layout(height=320)
    return fig
