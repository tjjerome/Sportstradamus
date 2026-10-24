"""The Receipts verdict hero: the model's recommended legs at the platform payout.

The hero is ``realized.cohort_summary`` of the page window's ``Recommended`` legs, the rule
the story menu and the nightly ledger share; a context row beneath sets every posted side in
the same window against it, so the tail the model picks reads against the board it picked
from. ``window_offers`` cuts the page window and ``cohort_figures`` formats one summary, both
pure; ``render_receipts_hero`` draws the card.
"""

from datetime import datetime

import pandas as pd
import streamlit as st

from sportstradamus import realized
from sportstradamus.analysis import TIMEFRAMES
from sportstradamus.dashboard.assets import ambient_css
from sportstradamus.dashboard.theme import BORDER, GOLD, GRAY
from sportstradamus.helpers import UNDERDOG_BOOST_BASELINE

# Nebula wash (DESIGN.md §3): blue radial stop + gold held at 7% opacity, both well under
# the hero-card 12% gold ceiling — literal values ported from the mockup's own .hero
# background (docs/mockups/p8-receipts.html:29-30).
_HERO_BG_FALLBACK = (
    "radial-gradient(ellipse at 88% -20%, rgba(46,107,230,.15), transparent 48%),"
    "radial-gradient(ellipse at 8% 130%, rgba(201,162,39,.07), transparent 46%),#1A1D24"
)
# ambient_css swaps in the ambient_receipts_hero manifest slot's art once a file lands;
# until then this resolves to _HERO_BG_FALLBACK unchanged.
_HERO_BG = ambient_css("ambient_receipts_hero", _HERO_BG_FALLBACK)

# The page window: four of TIMEFRAMES' labels (6m is not offered) plus "All", which cuts
# nothing and is the default view.
WINDOW_LABELS = {
    "7d": "Last week",
    "30d": "Last month",
    "3m": "Last 3 mo",
    "1y": "Last year",
}
_WINDOW_DAYS = {label: days for label, days in TIMEFRAMES if label in WINDOW_LABELS}
WINDOW_OPTIONS = [*WINDOW_LABELS, "All"]

# Numerals inside the hero's prose lines still set in Plex Mono (DESIGN.md §2).
_MONO = "<span style=\"font-family:'IBM Plex Mono',monospace\">{}</span>"


def window_offers(offers: pd.DataFrame, label: str, now: datetime) -> pd.DataFrame:
    """The offers inside page window ``label`` (one of ``WINDOW_OPTIONS``) ending at ``now``.

    "All" keeps every offer; a dated window cuts on ``realized.window``, the cut the nightly
    ledger makes, so a 30-day page window holds the ledger's 30-day offers.
    """
    if label == "All":
        return offers
    return realized.window(offers, _WINDOW_DAYS[label], now)


def cohort_figures(summary: dict[str, float]) -> dict[str, str]:
    """Display strings for one ``realized.cohort_summary``.

    Keys ``n``, ``roi`` (signed), ``hit_rate``, ``pred_rate`` and ``breakeven_rate`` as
    percents, ``payout`` as the mean decimal payout (``1.83x``), ``record`` as wins–losses
    and ``units`` as signed whole units. An empty cohort has no rates, so they read "—".
    """
    n = summary["n"]
    if not n:
        return {"n": "0", "record": "0–0", "units": "+0"} | dict.fromkeys(
            ("roi", "hit_rate", "pred_rate", "breakeven_rate", "payout"), "—"
        )
    wins = round(n * summary["hit_rate"])
    return {
        "n": f"{n:,}",
        "roi": f"{summary['roi']:+.1%}",
        "hit_rate": f"{summary['hit_rate']:.1%}",
        "pred_rate": f"{summary['pred_rate']:.1%}",
        "breakeven_rate": f"{summary['breakeven_rate']:.1%}",
        "payout": f"{summary['payout']:.2f}x",
        "record": f"{wins:,}–{n - wins:,}",
        "units": f"{summary['units']:+,.0f}",
    }


def _hero_stat(label: str, value: str, *, size: str, color: str = "") -> str:
    """One Cinzel-kicker / Plex-mono-value stat span for the hero's stats row.

    ``size`` is ``"xl"`` (38px, gold) for the single hero number or ``"lg"`` (26px,
    default text color) for the supporting stats beside it — the mockup's own
    ``.v.xl``/``.v.lg`` weight split (only ROI gets the hero treatment; the rest read as
    normal-weight context).
    """
    font_size = 38 if size == "xl" else 26
    color_style = f"color:{color};" if color else ""
    return (
        f"<span><div style=\"font-family:'Cinzel',serif;font-size:9px;"
        f'letter-spacing:.13em;text-transform:uppercase;color:{GRAY}">{label}</div>'
        f"<div style=\"font-family:'IBM Plex Mono',monospace;font-weight:600;"
        f'line-height:1;font-size:{font_size}px;{color_style}">{value}</div></span>'
    )


def render_receipts_hero(recommended: dict[str, float], posted: dict[str, float]) -> None:
    """The verdict card over two ``realized.cohort_summary`` results, then its grading caption.

    ``recommended`` (the page window's ``Recommended`` legs) carries the hero figures;
    ``posted`` (every posted side in the same window) fills the context row beneath them.
    """
    rec = cohort_figures(recommended)
    ctx = cohort_figures(posted)
    stats = "".join(
        (
            _hero_stat("ROI", rec["roi"], size="xl", color=GOLD),
            _hero_stat("Hit rate", rec["hit_rate"], size="lg"),
            _hero_stat("Model read", rec["pred_rate"], size="lg"),
            _hero_stat("Breakeven", rec["breakeven_rate"], size="lg"),
            _hero_stat("Avg payout", rec["payout"], size="lg"),
            _hero_stat("Record", rec["record"], size="lg"),
        )
    )
    st.markdown(
        f'<div style="position:relative;overflow:hidden;border:1px solid {BORDER};'
        f'border-radius:4px;padding:18px 20px;background:{_HERO_BG}">'
        f"<div style=\"font-family:'Cinzel',serif;font-size:9.5px;letter-spacing:.18em;"
        f"text-transform:uppercase;color:{GOLD}\">The model's recommendations</div>"
        f'<div style="display:flex;gap:34px;align-items:flex-end;flex-wrap:wrap;'
        f'margin-top:6px">{stats}</div>'
        f'<div style="color:{GRAY};font-size:12px;margin-top:12px">'
        f"{_MONO.format(rec['units'])} units across {_MONO.format(rec['n'])} recommended "
        f"legs — model edge ≥ {realized.RECOMMENDED_EDGE_MIN:.0%} at the platform payout, "
        f"payout ≤ {realized.RECOMMENDED_PAYOUT_MAX:g}x, the rule the story menu and the "
        "nightly ledger share.</div>"
        f'<div style="color:{GRAY};font-size:12px;margin-top:8px;padding-top:8px;'
        f'border-top:1px solid {BORDER}">All posted sides in the window: '
        f"{_MONO.format(ctx['n'])} legs · hit {_MONO.format(ctx['hit_rate'])} · breakeven "
        f"{_MONO.format(ctx['breakeven_rate'])} · ROI {_MONO.format(ctx['roi'])}</div></div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Each leg is graded at the platform's posted payout (Underdog boost × "
        f"{UNDERDOG_BOOST_BASELINE} per pick, Sleeper's posted multiplier): posted sides "
        "only, one bet per platform, pushes excluded."
    )
