"""Pin the Lab Training glance-strip funnel chart builder.

``lifecycle_funnel`` takes the four raw stage counts (not a DataFrame) since the
stages come from three genuinely different sources on the page (row count, the
six-gate system's ``g1_pass``/``ship`` columns, and the older ``classify_lifecycle``
proxy's ``lifecycle_state``) — see ``lab_training.py``'s own module docstring for why
those aren't guaranteed monotonic on real data. The builder must not assume or enforce
monotonicity; it renders whatever counts it's given.
"""

from __future__ import annotations

import plotly.graph_objects as go

from sportstradamus.dashboard.surfaces.lab_training_charts import lifecycle_funnel


def test_lifecycle_funnel_returns_a_funnel_figure():
    fig = lifecycle_funnel(trained=40, g1_pass=30, all_gates=10, graduated=5)
    assert isinstance(fig, go.Figure)
    assert any(isinstance(t, go.Funnel) for t in fig.data)


def test_lifecycle_funnel_stage_values_in_order_monotonic_case():
    # The realistic/intended case: strictly decreasing counts.
    fig = lifecycle_funnel(trained=40, g1_pass=30, all_gates=10, graduated=5)
    funnel_trace = next(t for t in fig.data if isinstance(t, go.Funnel))
    assert list(funnel_trace.x) == [40, 30, 10, 5]
    assert list(funnel_trace.y) == [
        "Trained",
        "G1 pass",
        "All gates (ship)",
        "Graduated live",
    ]


def test_lifecycle_funnel_does_not_enforce_monotonicity():
    # graduated (older classify_lifecycle proxy) is NOT guaranteed to be a subset of
    # ship==True cells (different formulas) — a non-monotonic input must render as-is,
    # not get clamped or reordered.
    fig = lifecycle_funnel(trained=40, g1_pass=30, all_gates=10, graduated=12)
    funnel_trace = next(t for t in fig.data if isinstance(t, go.Funnel))
    assert list(funnel_trace.x) == [40, 30, 10, 12]


def test_lifecycle_funnel_handles_all_zero_counts():
    fig = lifecycle_funnel(trained=0, g1_pass=0, all_gates=0, graduated=0)
    funnel_trace = next(t for t in fig.data if isinstance(t, go.Funnel))
    assert list(funnel_trace.x) == [0, 0, 0, 0]
