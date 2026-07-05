# ARCHIVED 2026-07-05 from src/sportstradamus/dashboard/components/deep_dive_charts.py
# Reason: P8 Phase C Task C2 replaced the Correlated tab's Strong/Moderate/Mild
#         correlation-strength badge with the copula-scored EV-lift figure
#         (slip_engine.ev_lift), per the details-tabs mockup and spec §4.1 — removing
#         this formatter's only call site (deep_dive.py's old _render_corr_cards).
# Last live SHA: 8d7bfee
# Original imports (now unresolved here):
#   (none — pure function, no imports)

# Correlation-strength badge thresholds: a leg whose multiplier on the base edge
# clears these earns the Strong / Moderate badge in the offer-detail view.
_CORR_STRONG_MULT = 1.25
_CORR_MODERATE_MULT = 1.1


def strength_badge(mult: float) -> str:
    """Return a semantic colored badge for correlation strength (color paired with text)."""
    if mult >= _CORR_STRONG_MULT:
        return ":red[Strong]"
    if mult >= _CORR_MODERATE_MULT:
        return ":orange[Moderate]"
    return ":gray[Mild]"
