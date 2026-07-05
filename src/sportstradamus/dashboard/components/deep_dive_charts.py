"""Chart builders for the offer-detail dialog."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from sportstradamus.dashboard.data import GAMELOG_SCHEMA, load_gamelog
from sportstradamus.dashboard.theme import GOLD, GRAY, GREEN, RED

# Distribution families, mirrored from the prediction pipeline.
_CONTINUOUS = ("Gamma", "ZAGamma", "SkewNormal")
_ZERO_INFLATED = ("ZAGamma", "ZINB")

# Offer columns -> distribution-parameter names the curve builders read.
DIST_PARAM_COLS = {
    "Model R": "r",
    "Model Alpha": "alpha",
    "Model Sigma": "sigma",
    "Model Skew": "skew_alpha",
    "Gate": "gate",
}


def history_chart(df: pd.DataFrame, line: float) -> alt.Chart:
    """Bar chart of recent games with a dashed betting-line rule and gold line tag.

    ``df`` must have columns ``Label`` (ordered x-axis string),
    ``StatValue`` (y), and ``Hit`` (bool).  The most-recent game should be
    last so the bars read left-to-right chronologically.  Bars over the line
    read green, under it red (``Hit`` is ``StatValue >= line``); the white
    dashed rule sits in front, gold-tagged with its value.
    """
    df = df.copy()
    df["color"] = np.where(df["Hit"], "Over", "Under")
    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Label:N", sort=None, title="", axis=alt.Axis(labelAngle=-40)),
            y=alt.Y("StatValue:Q", title=""),
            color=alt.Color(
                "color:N",
                scale=alt.Scale(domain=["Over", "Under"], range=[GREEN, RED]),
                legend=None,
            ),
            tooltip=["Label:N", "StatValue:Q"],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"Line": [line]}))
        .mark_rule(strokeDash=[6, 3], color="#FFFFFF", strokeWidth=1.5)
        .encode(y="Line:Q")
    )
    # Gold "line {x}" callout anchored to the leftmost bar at the line's height.
    tag_df = pd.DataFrame(
        {"Label": [df["Label"].iloc[0]], "Line": [line], "tag": [f"line {line:.10g}"]}
    )
    tag = (
        alt.Chart(tag_df)
        .mark_text(align="left", baseline="bottom", dx=-14, dy=-3, color=GOLD, fontWeight="bold")
        .encode(x=alt.X("Label:N", sort=None), y="Line:Q", text="tag:N")
    )
    return bars + rule + tag


def _continuous_curve(dist: str, ev: float, std: float, params: dict):
    xs = np.linspace(max(0.0, ev - 4 * std), ev + 4 * std, 300)
    if dist == "SkewNormal":
        sigma = params.get("sigma")
        if not sigma or sigma <= 0:
            sigma = ev * 0.3
        skew = params.get("skew_alpha") or 0
        return xs, stats.skewnorm.pdf(xs, skew, loc=ev, scale=sigma)

    alpha = params.get("alpha")
    if not alpha or alpha <= 0:
        raise ValueError(f"{dist} requires Model Alpha > 0")
    pdf_vals = stats.gamma.pdf(xs, alpha, scale=ev / alpha)
    if dist == "ZAGamma":
        pdf_vals = (1 - (params.get("gate") or 0)) * pdf_vals
    return xs, pdf_vals


def _discrete_curve(dist: str, ev: float, std: float, params: dict):
    hi = int(np.ceil(ev + 4 * std)) + 1
    xs = np.arange(0, hi + 1, dtype=int)
    if dist == "Poisson":
        return xs, stats.poisson.pmf(xs, ev)
    if dist not in ("NegBin", "ZINB"):
        raise ValueError(f"Unknown discrete dist: {dist}")

    r = params.get("r")
    if not r or r <= 0:
        raise ValueError(f"{dist} requires Model R > 0")
    pmf = stats.nbinom.pmf(xs, r, r / (r + ev))
    if dist == "ZINB":
        gate = params.get("gate") or 0
        pmf = np.where(xs == 0, gate + (1 - gate) * pmf, (1 - gate) * pmf)
    return xs, pmf


def distribution_frame(dist: str, ev: float, std: float, params: dict, line: float):
    is_continuous = dist in _CONTINUOUS
    if is_continuous:
        xs, vals = _continuous_curve(dist, ev, std, params)
        y_title = "Density"
    else:
        xs, vals = _discrete_curve(dist, ev, std, params)
        y_title = "Probability"
    df_pdf = pd.DataFrame(
        {
            "x": xs,
            "P": vals,
            "Side": ["Over" if x >= line else "Under" for x in xs],
        }
    )
    return df_pdf, y_title, is_continuous


def resolve_std(projection_std, ev: float, cv: float) -> float:
    """A positive, finite spread for the distribution x-range.

    Prefers the model's ``Projection STD``; book-fallback offers carry none, so it
    falls back to the CV-implied ``ev * cv`` (``cv`` is always present), then ``ev * 0.3``.
    The NaN guard matters: ``NaN or default`` keeps the NaN (NaN is truthy), which blanks
    the x-range and renders the curve invisible.
    """
    if pd.notna(projection_std) and projection_std > 0:
        return float(projection_std)
    if pd.notna(cv) and cv > 0:
        return float(ev) * float(cv)
    return float(ev) * 0.3


def _projection_overlay(df_pdf: pd.DataFrame, projection: float) -> tuple[alt.Chart, alt.Chart]:
    """A dot at the model mean (sitting on its own density) plus a numeric label.

    Not a rule — the projection is a point estimate, distinct from the two vertical
    book lines. Its height interpolates the curve so the dot rides the distribution.
    Returned as two charts (not pre-layered) so the caller's ``+`` keeps a flat layer
    list rather than nesting a sub-LayerChart.
    """
    y_at = float(np.interp(projection, df_pdf["x"], df_pdf["P"]))
    pt = pd.DataFrame({"x": [projection], "P": [y_at], "label": [f"{projection:.1f}"]})
    dot = (
        alt.Chart(pt)
        .mark_point(filled=True, size=90, color=GOLD, shape="diamond")
        .encode(x="x:Q", y="P:Q")
    )
    label = (
        alt.Chart(pt)
        .mark_text(dy=-12, color=GOLD, fontWeight="bold")
        .encode(x="x:Q", y="P:Q", text="label:N")
    )
    return dot, label


def distribution_chart(
    df_pdf: pd.DataFrame,
    is_continuous: bool,
    line: float,
    market: str,
    y_title: str,
    projection: float | None = None,
    consensus_line: float | None = None,
) -> alt.Chart:
    # Over green / under red keys off position relative to the line, so the color
    # legend is redundant chart junk (mockup rev 3) — drop it, keep the mass split.
    color_enc = alt.Color(
        "Side:N",
        scale=alt.Scale(domain=["Over", "Under"], range=[GREEN, RED]),
        legend=None,
    )
    if is_continuous:
        x_enc = alt.X("x:Q", title=market)
        y_enc = alt.Y("P:Q", title=y_title)
        chart = alt.layer(
            alt.Chart(df_pdf)
            .mark_area(opacity=0.3)
            .encode(x=x_enc, y=y_enc, color=color_enc, tooltip=["x:Q", "P:Q", "Side:N"]),
            alt.Chart(df_pdf).mark_line(strokeWidth=2).encode(x=x_enc, y=y_enc, color=color_enc),
        )
    else:
        # Discrete: touching bars via explicit half-integer boundaries.
        df_pdf["x_start"] = df_pdf["x"] - 0.5
        df_pdf["x_end"] = df_pdf["x"] + 0.5
        chart = (
            alt.Chart(df_pdf)
            .mark_rect(stroke="#444", strokeWidth=1)
            .encode(
                x=alt.X("x_start:Q", title=market, axis=alt.Axis(tickMinStep=1)),
                x2="x_end:Q",
                y=alt.Y("P:Q", title=y_title),
                color=color_enc,
                tooltip=["x:Q", "P:Q", "Side:N"],
            )
        )
    betting_line = (
        alt.Chart(pd.DataFrame({"Line": [line]}))
        .mark_rule(strokeDash=[6, 3], color="#FFFFFF", strokeWidth=1.5)
        .encode(x="Line:Q")
    )
    chart = chart + betting_line
    # Consensus market line: solid GRAY rule, distinct from the white dashed app line,
    # and carrying no color encoding so it never re-splits the over/under shading.
    if consensus_line is not None and not pd.isna(consensus_line):
        chart = chart + (
            alt.Chart(pd.DataFrame({"Consensus": [consensus_line]}))
            .mark_rule(color=GRAY, strokeWidth=2)
            .encode(x="Consensus:Q")
        )
    if projection is not None and not pd.isna(projection):
        dot, label = _projection_overlay(df_pdf, projection)
        chart = chart + dot + label
    return chart


def _has_cols(df: pd.DataFrame, *cols: str | None) -> bool:
    # schema.get() yields None for absent fields, so guard truthiness too.
    return all(c and c in df.columns for c in cols)


def _append_date_suffix(labels: pd.Series, games: pd.DataFrame, dcol: str | None) -> pd.Series:
    if not _has_cols(games, dcol):
        return labels
    dates = pd.to_datetime(games[dcol]).dt.strftime("%m/%d")
    return labels + ", " + dates


def _history_frame(label_vals, stat_vals, line, opp_vals) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Label": label_vals,
            "StatValue": stat_vals,
            "Hit": stat_vals >= line,
            "Opponent": opp_vals,
        }
    )


def build_recent_history(
    row: pd.Series, league: str, stat_key: str, line: float, schema: dict
) -> pd.DataFrame:
    gl = load_gamelog(league)
    pcol, dcol = schema["player"], schema.get("date")
    ocol, hcol = schema.get("opp"), schema.get("home")
    if gl.empty or not _has_cols(gl, pcol, stat_key):
        return pd.DataFrame()

    pg = gl[gl[pcol] == row["Player"]].copy()
    if _has_cols(pg, dcol):
        pg = pg.sort_values(dcol)
    pg = pg.tail(10)

    if _has_cols(pg, ocol, hcol):
        labels = np.where(pg[hcol].astype(bool), "", "@") + pg[ocol].astype(str)
    elif _has_cols(pg, ocol):
        labels = pg[ocol].astype(str)
    elif _has_cols(pg, "week"):
        labels = pd.Series([f"Wk {w}" for w in pg["week"]], index=pg.index)
    else:
        labels = pd.Series([str(i + 1) for i in range(len(pg))], index=pg.index)
    labels = _append_date_suffix(labels, pg, dcol)

    opp_vals = pg[ocol].values if _has_cols(pg, ocol) else ""
    return _history_frame(labels.values, pg[stat_key].values, line, opp_vals)


def build_h2h_history(
    row: pd.Series, league: str, stat_key: str, line: float, opponent: str, schema: dict
) -> pd.DataFrame:
    gl = load_gamelog(league)
    pcol, dcol = schema["player"], schema.get("date")
    ocol, hcol = schema.get("opp"), schema.get("home")
    if gl.empty or not _has_cols(gl, pcol, ocol, stat_key):
        return pd.DataFrame()

    games = gl[(gl[pcol] == row["Player"]) & (gl[ocol] == opponent)].copy()
    if _has_cols(games, dcol):
        games = games.sort_values(dcol)
    games = games.tail(10)
    if games.empty:
        return pd.DataFrame()

    if _has_cols(games, hcol):
        prefix = np.where(games[hcol].astype(bool), "", "@")
        labels = pd.Series([f"{p}{opponent}" for p in prefix], index=games.index)
    else:
        labels = pd.Series([opponent] * len(games), index=games.index)
    labels = _append_date_suffix(labels, games, dcol)

    opp_vals = games[ocol].values if _has_cols(games, ocol) else opponent
    return _history_frame(labels.values, games[stat_key].values, line, opp_vals)


def sparkline(series: list[float], *, height: int = 44) -> alt.Chart:
    """A minimal axis-less gold line for a short value series (Other-stats tab)."""
    df = pd.DataFrame({"i": range(len(series)), "v": series})
    return (
        alt.Chart(df)
        .mark_line(color=GOLD, strokeWidth=2, point=alt.OverlayMarkDef(color=GOLD, size=18))
        .encode(x=alt.X("i:Q", axis=None), y=alt.Y("v:Q", axis=None), tooltip=["v:Q"])
        .properties(height=height)
    )
