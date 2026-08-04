"""Golden pins for Lab Correlations' pure chart/aggregation builders (P8 Task B3).

``corr_heatmap`` pivots a ``load_corr_market_summary`` frame to a market x
market diverging heatmap, masking thin pairs; ``worst_driver_pair``
reconstructs market-pair identity from a resolved-parlay frame's ``legs`` /
``Boost Pairs`` columns to name the strongest driver behind a boosted-miss.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sportstradamus.dashboard.surfaces.lab_correlations_charts import (
    corr_heatmap,
    worst_driver_pair,
)
from sportstradamus.helpers import market_display_name

_AST, _PTS, _TOV = (market_display_name("NBA", m) for m in ("AST", "PTS", "TOV"))

_LAB_CORRELATIONS_SRC = Path(
    "src/sportstradamus/dashboard/surfaces/lab_correlations.py"
).read_text()


def test_value_add_scatter_plots_correlation_adjusted_prob_not_the_independence_prescreen():
    """The Correlation Value-Add scatter must plot the copula-adjusted joint probability
    (``Model EV / Boost``) against the independent one (``Indep P / Boost``), never the ``P``
    column: ``P`` is the independence prescreen and equals ``Indep P`` exactly (verified
    in-data: corr 1.0), so plotting it drew a perfect y=x line (owner: "suspiciously linear").
    """
    src = _LAB_CORRELATIONS_SRC
    assert "p_corr" in src and "p_indep" in src
    assert 'resolved["Model EV"] / resolved["Boost"]' in src
    assert 'y="p_corr"' in src, "value-add scatter no longer plots the correlation-adjusted prob"
    assert 'y="P"' not in src, "scatter regressed to the P independence prescreen (perfect y=x)"


def test_lab_correlations_never_reads_the_dashboard_forbidden_corr_parquets():
    """corr_same_team.parquet/corr_opposing.parquet are 2.85M-row NBA artifacts.

    The Lab heatmap reads only the tiny corr_market_summary.parquet via
    load_corr_market_summary; the module must never reference the raw
    per-team matrices directly.
    """
    assert "corr_same_team.parquet" not in _LAB_CORRELATIONS_SRC
    assert "corr_opposing.parquet" not in _LAB_CORRELATIONS_SRC


_SUMMARY = pd.DataFrame(
    [
        {
            "market_a": "AST",
            "market_b": "AST",
            "rho_mean": 1.0,
            "n_teams": 30,
            "scope": "same_team",
        },
        {
            "market_a": "AST",
            "market_b": "PTS",
            "rho_mean": 0.42,
            "n_teams": 28,
            "scope": "same_team",
        },
        {
            "market_a": "PTS",
            "market_b": "AST",
            "rho_mean": 0.42,
            "n_teams": 28,
            "scope": "same_team",
        },
        {
            "market_a": "PTS",
            "market_b": "PTS",
            "rho_mean": 1.0,
            "n_teams": 30,
            "scope": "same_team",
        },
        {"market_a": "AST", "market_b": "TOV", "rho_mean": 0.9, "n_teams": 2, "scope": "same_team"},
        {"market_a": "TOV", "market_b": "AST", "rho_mean": 0.9, "n_teams": 2, "scope": "same_team"},
        {"market_a": "TOV", "market_b": "TOV", "rho_mean": 1.0, "n_teams": 2, "scope": "same_team"},
        {
            "market_a": "AST",
            "market_b": "PTS",
            "rho_mean": -0.1,
            "n_teams": 15,
            "scope": "opposing",
        },
        {
            "market_a": "PTS",
            "market_b": "AST",
            "rho_mean": -0.1,
            "n_teams": 15,
            "scope": "opposing",
        },
        {"market_a": "AST", "market_b": "AST", "rho_mean": 1.0, "n_teams": 30, "scope": "opposing"},
        {"market_a": "PTS", "market_b": "PTS", "rho_mean": 1.0, "n_teams": 30, "scope": "opposing"},
    ]
)


def test_corr_heatmap_scopes_to_requested_scope():
    fig = corr_heatmap(_SUMMARY, "NBA", "same_team")
    heat = fig.data[0]
    assert set(heat.x) == {_AST, _PTS, _TOV}
    assert set(heat.y) == {_AST, _PTS, _TOV}


def test_corr_heatmap_masks_thin_pairs_below_n_teams_floor():
    """AST-TOV/TOV-AST have n_teams=2 (< 3) — must be masked (NaN), not plotted."""
    fig = corr_heatmap(_SUMMARY, "NBA", "same_team")
    heat = fig.data[0]
    z = pd.DataFrame(heat.z, index=list(heat.y), columns=list(heat.x))
    assert pd.isna(z.loc[_AST, _TOV])
    assert pd.isna(z.loc[_TOV, _AST])
    # TOV-TOV (self pair, n_teams=2) is also thin and must be masked too.
    assert pd.isna(z.loc[_TOV, _TOV])


def test_corr_heatmap_keeps_well_populated_pairs():
    fig = corr_heatmap(_SUMMARY, "NBA", "same_team")
    heat = fig.data[0]
    z = pd.DataFrame(heat.z, index=list(heat.y), columns=list(heat.x))
    assert z.loc[_AST, _PTS] == pytest.approx(0.42)
    assert z.loc[_AST, _AST] == pytest.approx(1.0)


def test_corr_heatmap_is_dark_centered_diverging_never_gold():
    from sportstradamus.dashboard import theme
    from sportstradamus.dashboard.surfaces.lab_correlations_charts import _HEATMAP_SURFACE

    fig = corr_heatmap(_SUMMARY, "NBA", "same_team")
    heat = fig.data[0]
    colorscale_hexes = {stop[1] for stop in heat.colorscale}
    # Saturated ends stay on the committed diverging tokens; the neutral centre is re-anchored
    # to the app background so rho ~ 0 blends into the dark theme instead of rendering as a
    # jarring light-cream block. Never gold — this is a data heatmap, not decoration.
    assert _HEATMAP_SURFACE in colorscale_hexes
    assert (colorscale_hexes - {_HEATMAP_SURFACE}) <= set(theme.DIVERGING_COLORS)
    assert theme.GOLD not in colorscale_hexes


def test_corr_heatmap_opposing_scope_differs_from_same_team():
    fig = corr_heatmap(_SUMMARY, "NBA", "opposing")
    heat = fig.data[0]
    z = pd.DataFrame(heat.z, index=list(heat.y), columns=list(heat.x))
    assert z.loc[_AST, _PTS] == pytest.approx(-0.1)


def test_corr_heatmap_empty_summary_returns_empty_figure():
    empty = pd.DataFrame(columns=["market_a", "market_b", "rho_mean", "n_teams", "scope"])
    fig = corr_heatmap(empty, "MLB", "same_team")
    assert len(fig.data) == 0 or len(fig.data[0].x or []) == 0


def _leg(market: str, league: str = "NBA") -> dict:
    return {"player": "P", "market": market, "league": league}


def test_worst_driver_pair_picks_most_frequent_pair_among_boosted_misses():
    resolved = pd.DataFrame(
        [
            {
                "Hit": 0,
                "P": 0.5,
                "Indep P": 0.4,
                "legs": [_leg("AST"), _leg("TOV")],
                "Boost Pairs": (1.3,),
            },
            {
                "Hit": 0,
                "P": 0.5,
                "Indep P": 0.4,
                "legs": [_leg("AST"), _leg("TOV")],
                "Boost Pairs": (1.4,),
            },
            {
                "Hit": 0,
                "P": 0.6,
                "Indep P": 0.5,
                "legs": [_leg("PTS"), _leg("REB")],
                "Boost Pairs": (1.1,),
            },
            # Boosted but a HIT — must not count toward the worst-pair tally.
            {
                "Hit": 1,
                "P": 0.7,
                "Indep P": 0.5,
                "legs": [_leg("STL"), _leg("BLK")],
                "Boost Pairs": (1.9,),
            },
            # Missed but NOT boosted (P <= Indep P) — must not count either.
            {
                "Hit": 0,
                "P": 0.3,
                "Indep P": 0.35,
                "legs": [_leg("FG3M"), _leg("FTM")],
                "Boost Pairs": (0.9,),
            },
        ]
    )
    result = worst_driver_pair(resolved)
    assert result is not None
    market_a, market_b, count = result
    assert {market_a, market_b} == {"AST", "TOV"}
    assert count == 2


def test_worst_driver_pair_none_when_no_boosted_misses():
    resolved = pd.DataFrame(
        [
            {
                "Hit": 1,
                "P": 0.6,
                "Indep P": 0.4,
                "legs": [_leg("AST"), _leg("TOV")],
                "Boost Pairs": (1.2,),
            }
        ]
    )
    assert worst_driver_pair(resolved) is None


def test_worst_driver_pair_none_when_legs_column_missing():
    resolved = pd.DataFrame([{"Hit": 0, "P": 0.6, "Indep P": 0.4}])
    assert worst_driver_pair(resolved) is None


def test_worst_driver_pair_handles_three_leg_parlay_pair_reconstruction():
    """3 legs -> 3 pairs via itertools.combinations; each pair counted once per row.

    Two rows share the AST-TOV pair (plus a distinct third leg each), so it
    should out-count every other pair, which appears only once each.
    """
    resolved = pd.DataFrame(
        [
            {
                "Hit": 0,
                "P": 0.5,
                "Indep P": 0.3,
                "legs": [_leg("AST"), _leg("PTS"), _leg("TOV")],
                "Boost Pairs": (1.1, 1.2, 1.5),
            },
            {
                "Hit": 0,
                "P": 0.6,
                "Indep P": 0.4,
                "legs": [_leg("AST"), _leg("REB"), _leg("TOV")],
                "Boost Pairs": (1.2, 1.3, 1.4),
            },
        ]
    )
    result = worst_driver_pair(resolved)
    assert result is not None
    market_a, market_b, count = result
    assert {market_a, market_b} == {"AST", "TOV"}
    assert count == 2
