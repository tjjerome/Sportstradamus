"""Chart/aggregation builders for the Lab Correlations surface (P8 Task B3)."""

import itertools

import pandas as pd
import plotly.graph_objects as go

from sportstradamus.dashboard import theme
from sportstradamus.helpers import market_display_name

# Market pairs aggregated from fewer than this many (team x rank-prefix) rows
# are too thin to trust — masked out of the heatmap rather than shown.
_MIN_PAIR_OBSERVATIONS = 3

# theme.DIVERGING_COLORS is light-centred (cream at rho ~ 0) for white backgrounds; on the
# app's dark surface a full grid of near-zero cells reads as a jarring bright block. Re-anchor
# the ramp's neutral at the page background so rho ~ 0 blends into the dark theme and only real
# +/- correlations pick up red/blue — the "near-neutral stays dark" the board grid gets by
# leaving those cells unpainted. Ends reuse the committed diverging tokens.
_HEATMAP_SURFACE = "#0E1117"  # config.toml backgroundColor — the app page background
_DARK_CENTERED_DIVERGING = [
    [0.0, theme.DIVERGING_COLORS[0]],
    [0.25, theme.DIVERGING_COLORS[2]],
    [0.5, _HEATMAP_SURFACE],
    [0.75, theme.DIVERGING_COLORS[7]],
    [1.0, theme.DIVERGING_COLORS[9]],
]


def corr_heatmap(summary: pd.DataFrame, league: str, scope: str) -> go.Figure:
    """Diverging market x market correlation heatmap for one league/scope.

    ``summary`` is a :func:`~sportstradamus.dashboard.data.load_corr_market_summary`
    frame (``market_a, market_b, rho_mean, n_teams, scope``). Pairs with
    ``n_teams < 3`` are masked (rendered as gaps, not a value) rather than
    shown — too few observations to trust. Market slugs are routed through
    :func:`~sportstradamus.helpers.market_display_name` for the axis labels.

    Empirical-vs-model rho overlay is the §6.4 follow-up — not built here.
    """
    scoped = summary.loc[summary["scope"] == scope]
    if scoped.empty:
        return go.Figure(data=[go.Heatmap(x=[], y=[], z=[])])

    thin = scoped["n_teams"] < _MIN_PAIR_OBSERVATIONS
    pivot = scoped.assign(rho_mean=scoped["rho_mean"].mask(thin)).pivot(
        index="market_b", columns="market_a", values="rho_mean"
    )
    labels = [market_display_name(league, slug) for slug in pivot.columns]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.to_numpy(),
            x=labels,
            y=[market_display_name(league, slug) for slug in pivot.index],
            colorscale=_DARK_CENTERED_DIVERGING,
            zmid=0,
            zmin=-1,
            zmax=1,
            hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>",
            colorbar={"title": "ρ"},
        )
    )
    fig.update_layout(
        xaxis_title="Market",
        yaxis_title="Market",
        height=max(400, len(labels) * 40),
    )
    return fig


def worst_driver_pair(resolved: pd.DataFrame) -> tuple[str, str, int] | None:
    """Most frequent market pair among boosted-but-missed parlays.

    "Boosted" = correlated probability exceeded the independent probability
    (``P > Indep P``, the same split the value-add scatter uses); "missed" =
    ``Hit == 0``. Reconstructs pair identity from ``legs`` (list[dict] with a
    ``market`` field, in ``bet_id`` order) since ``Boost Pairs``/``Corr Pairs``
    are bare tuples over ``itertools.combinations(legs, 2)`` order — the same
    order ``np.triu_indices`` produced them in at parlay-search time (P8 B3
    context brief; verified against ``prediction/parlay.py``'s row builder).
    This is a frequency count, not a boost-weighted score — a judgment call,
    not a spec-dictated formula.

    Returns ``(market_a, market_b, count)`` for the most frequent pair, or
    ``None`` if there are no boosted misses or no ``legs`` data to pair from.
    """
    if "legs" not in resolved.columns:
        return None

    boosted_miss = resolved.loc[(resolved["Hit"] == 0) & (resolved["P"] > resolved["Indep P"])]
    if boosted_miss.empty:
        return None

    pair_counts: dict[tuple[str, str], int] = {}
    for legs in boosted_miss["legs"]:
        for leg_a, leg_b in itertools.combinations(legs, 2):
            pair = tuple(sorted((leg_a["market"], leg_b["market"])))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    if not pair_counts:
        return None

    (market_a, market_b), count = max(pair_counts.items(), key=lambda kv: kv[1])
    return market_a, market_b, count
