"""Distributional probability scoring for a single league/market/platform batch.

:func:`model_prob` is the hot path of the prediction pipeline: it loads
the trained LightGBMLSS model pickle, runs vectorized inference on the
``playerStats`` feature matrix, blends model and bookmaker predictions
via :func:`fused_loc`, applies temperature-scaling calibration, and
returns a list of scored offer dicts ready for :func:`find_correlation`.
"""

from __future__ import annotations

import os.path
import pickle

import numpy as np
import pandas as pd

from sportstradamus.helpers import (
    GATE_PUBLISH_THRESHOLD,
    NONZERO_DENOM_GATE,
    UNDERDOG_BOOST_BASELINE,
    LazyArchive,
    apply_temperature,
    decode_predictive_mean,
    fused_loc,
    get_ev,
    get_odds,
    get_push_prob,
    set_model_start_values,
    stat_cv,
    stat_dist,
    stat_map,
    stat_zi,
)
from sportstradamus.helpers.io import market_file_slug, model_pickle_path
from sportstradamus.spiderLogger import logger
from sportstradamus.training.baselines import get_target_normalization
from sportstradamus.training.posthoc import PROB_STAGE, apply_posthoc

# LazyArchive defers DuckDB lock acquisition until the first attribute
# access. See LazyArchive docstring in helpers/archive.py.
archive = LazyArchive()

# Maximum allowed model confidence before applying a boost.
_MAX_CONFIDENCE = 0.90

# Minimum bookmaker line for a "yards" market to be scored. Combo legs ("vs.")
# are exempt; single-player yards lines at or below this are too low to model
# reliably and are dropped.
_MIN_YARDS_LINE = 8

# Maximum Underdog boost multiplier kept when de-duplicating a player's offers.
# Above this the promo is an outlier (e.g. a discounted special) that distorts
# the per-player distance ranking, so it is filtered out.
_MAX_UNDERDOG_BOOST = 3.65

# Coin-flip prior used when no bookmaker price is available for an offer.
_BOOK_PRIOR_PROB: float = 0.5

# Maximum scored offers retained per player after boost-distance deduplication.
_MAX_OFFERS_PER_PLAYER: int = 3


def normalize_market(league: str, market: str, platform: str) -> str:
    """Canonicalize a platform's market label to the league gamelog/model key.

    Applies the per-platform ``stat_map`` alias plus the NHL and NBA/WNBA
    fixups shared by ``match_offers``, ``model_prob``, and ``book_fallback_prob``.
    """
    market = stat_map[platform].get(market, market)
    if league == "NHL":
        market = {"AST": "assists", "PTS": "points", "BLK": "blocked"}.get(market, market)
    if league in ("NBA", "WNBA"):
        market = market.replace("underdog", "prizepicks")
    return market


def _decode_skewnormal(
    prob_params: pd.DataFrame,
    playerStats: pd.DataFrame,
    hist_gate: float,
    offset_meta: dict | None,
    target_normalization: str,
) -> pd.DataFrame:
    """Decode raw SkewNormal model outputs into absolute EV / sigma / skew.

    Dispatches the ``loc`` and ``scale`` inverse transforms through the
    :mod:`sportstradamus.training.baselines` registry so the prediction-side
    decode mirrors the training-side forward transform exactly. Legacy
    pickles without ``offset_meta`` / ``target_normalization`` keys decode
    through the ``ratio_meanyr`` strategy, which is bit-identical to the
    pre-Task-5 hand-rolled ``loc * MeanYr_clipped`` formula.

    Args:
        prob_params: LightGBMLSS ``predict(pred_type="parameters")`` frame
            with ``loc``, ``scale``, and ``alpha`` columns.
        playerStats: Feature DataFrame; supplies ``MeanYr`` /
            ``MeanYr_nonzero`` (and ``GamesPlayed`` for the EB-centered
            strategy).
        hist_gate: Empirical zero-rate from ``stat_zi`` for the market.
        offset_meta: Pickle-persisted baseline metadata, or ``None`` for
            legacy / ratio-strategy models. The ``global_mean`` snapshot
            inside drives the EB prior at decode time.
        target_normalization: Slug of the baseline strategy the model was
            trained against; defaults to ``"ratio_meanyr"``.

    Returns:
        The same ``prob_params`` frame, mutated in place with the absolute
        ``Model EV``, ``Model Sigma``, ``Model Skew``, and optional
        ``Model Gate`` columns set.
    """
    strategy = get_target_normalization(target_normalization)
    denom_col = (
        "MeanYr_nonzero"
        if (hist_gate > NONZERO_DENOM_GATE and "MeanYr_nonzero" in playerStats.columns)
        else "MeanYr"
    )
    # global_mean snapshot lives in offset_meta for centered strategies; the
    # ratio strategy ignores it (uses MeanYr from features directly).
    global_mean = float((offset_meta or {}).get("global_mean", 0.0))

    ev_loc = strategy.decode_loc(prob_params["loc"].values, playerStats, global_mean, denom_col)
    ev_scale = strategy.decode_scale(prob_params["scale"].values, playerStats, denom_col)

    decoded = decode_predictive_mean(
        prob_params, "SkewNormal", sn_loc=ev_loc, sn_scale=ev_scale, hist_gate=hist_gate
    )
    prob_params["Model EV"] = decoded.ev
    prob_params["Model Sigma"] = decoded.sigma
    prob_params["Model Skew"] = decoded.skew
    if decoded.gate is not None:
        prob_params["Model Gate"] = decoded.gate
    return prob_params


def _apply_prob_posthoc(
    cal_over: np.ndarray, posthoc_slug: str, posthoc_blob: dict | None
) -> np.ndarray:
    """Apply a probability-stage post-hoc corrector to the calibrated over-prob.

    Mirrors the training-side seam in ``pipeline.train_market`` so the live
    probability matches the offline-gated one. No-op for mean-stage slugs and
    legacy pickles (slug ``"none"`` / blob ``None``).
    """
    if posthoc_slug in PROB_STAGE:
        return apply_posthoc(posthoc_slug, posthoc_blob, cal_over)
    return cal_over


def _book_cell_params(
    league: str, market: str
) -> tuple[str | None, float | None, float | None, float | None]:
    """Resolve ``(dist, cv, gate, step)`` for a cell from config, no pickle.

    Mirrors the pickle metadata ``model_prob`` reads, sourced instead from the
    committed ``stat_meta`` / runtime calibration. ``dist`` is ``None`` when the
    cell is unknown. ``step`` is approximated by family (the trained value is
    empirical and not recoverable without the pickle).
    """
    dist = stat_dist.get(league, {}).get(market)
    if dist is None:
        return None, None, None, None
    cv = stat_cv[league].get(market, 1)
    gate = (
        stat_zi.get(league, {}).get(market, 0) if dist in ("ZINB", "ZAGamma", "SkewNormal") else 0
    )
    step = 1.0 if dist in ("NegBin", "ZINB", "Poisson") else 0.5
    return dist, cv, gate, step


def _book_over_prob(
    offer_df: pd.DataFrame, dist: str, cv: float, step: float, gate: float | None
) -> pd.Series:
    """Devigged book probability of the over per row, inverted from ``Books EV``.

    Inverts the composite book EV through the cell distribution at each row's
    line. Shared by :func:`model_prob` and :func:`book_fallback_prob`.
    """
    if dist == "SkewNormal":
        return offer_df.apply(
            lambda x: (
                1
                - get_odds(
                    x["Line"],
                    x["Books EV"],
                    dist,
                    cv=cv,
                    step=step,
                    sigma=x["Books EV"] * cv,
                    skew_alpha=0,
                    gate=gate,
                )
            ),
            axis=1,
        )
    return offer_df.apply(
        lambda x: 1 - get_odds(x["Line"], x["Books EV"], dist, cv, step=step, gate=gate),
        axis=1,
    )


def _composite_book_evs(players, league: str, market: str, date_map: dict, stat_data) -> dict:
    """Composite (book-weighted, vig-free) EV per player, with combo fallback.

    Reads ``archive.get_ev`` for each player; when the archive has no direct
    price, falls back to the convolved consensus from ``check_combo_markets``
    (qb-yards / qb-tds). Players with no book price carry NaN.
    """
    evs = {}
    for player in players:
        date = date_map.get(player, "")
        ev = archive.get_ev(league, market, date, player)
        if np.isnan(ev):
            ev = stat_data.check_combo_markets(market, player, date)
        evs[player] = ev
    return evs


def _annotate_display_shape(offer_df: pd.DataFrame, dist: str) -> None:
    """Set the display-only ``Model Param`` and ``Model STD`` columns in place.

    Reads the distribution-shape parameters the model emitted (``Model R`` /
    ``Model Sigma`` / ``Model Alpha``); falls back to NaN when they are absent,
    as on a book-fallback record. These columns feed the dashboard detail popup
    only — no scoring path reads them.
    """
    # style: allow-complexity  flat per-distribution-family dispatch of closed-form
    # display variances; splitting duplicates the family/param-present guards.
    if dist in ("NegBin", "ZINB") and "Model R" in offer_df.columns:
        offer_df["Model Param"] = offer_df["Model R"]
    elif dist == "SkewNormal" and "Model Sigma" in offer_df.columns:
        offer_df["Model Param"] = offer_df["Model Sigma"]
    elif "Model Alpha" in offer_df.columns:
        offer_df["Model Param"] = offer_df["Model Alpha"]
    else:
        offer_df["Model Param"] = np.nan

    _m = offer_df["Model EV"].to_numpy(dtype=float)
    if dist in ("NegBin", "ZINB") and "Model R" in offer_df.columns:
        _r = offer_df["Model R"].to_numpy(dtype=float)
        offer_df["Model STD"] = np.sqrt(np.clip(_m + _m**2 / np.clip(_r, 1e-6, None), 0, None))
    elif dist == "SkewNormal" and "Model Sigma" in offer_df.columns:
        _sg = offer_df["Model Sigma"].to_numpy(dtype=float)
        _sk = (
            offer_df["Model Skew"].to_numpy(dtype=float)
            if "Model Skew" in offer_df.columns
            else np.zeros_like(_sg)
        )
        _delta = _sk / np.sqrt(1 + _sk**2)
        offer_df["Model STD"] = _sg * np.sqrt(np.clip(1 - 2 * _delta**2 / np.pi, 0, None))
    elif "Model Alpha" in offer_df.columns:
        _a = offer_df["Model Alpha"].to_numpy(dtype=float)
        offer_df["Model STD"] = _m / np.sqrt(np.clip(_a, 1e-6, None))
    else:
        offer_df["Model STD"] = np.nan


def _finalize_records(
    offer_df: pd.DataFrame,
    league: str,
    platform: str,
    dist: str,
    cv: float,
    step: float,
    temperature: float | None,
    dispersion_cal: float,
) -> list[dict]:
    """Resolve the bet side, apply boosts, and project the export schema.

    Shared tail of :func:`model_prob` and :func:`book_fallback_prob`. Expects
    ``offer_df`` to already carry ``Model Over`` / ``Model Under`` (win
    probabilities for each side), the raw ``Books`` over-probability, ``Push P``,
    ``Model EV`` and ``Books EV``. Feature-derived passenger columns absent on a
    book-fallback record (built without a feature matrix) are filled neutral so
    correlation and the dashboard read sane values — ``Player position`` in
    particular must stay an int, never NaN, or correlation's position map breaks.
    """
    totals_map = archive.default_totals
    for _col in ("Avg5", "AvgH2H", "H2HPlayed", "Total", "Defense position", "Moneyline"):
        if _col not in offer_df.columns:
            offer_df[_col] = np.nan

    offer_df["Model Over"] = offer_df["Model Over"].clip(upper=_MAX_CONFIDENCE)
    offer_df["Model Under"] = offer_df["Model Under"].clip(upper=_MAX_CONFIDENCE)

    offer_df["Model P"] = offer_df[["Model Over", "Model Under"]].max(axis=1)
    offer_df["Bet"] = offer_df[["Model Over", "Model Under"]].idxmax(axis=1).str[6:]

    if "Boost" in offer_df.columns:
        offer_df.loc[offer_df["Boost"] == 1, ["Boost_Under", "Boost_Over"]] = 1
    offer_df[["Boost_Under", "Boost_Over"]] = offer_df[["Boost_Under", "Boost_Over"]].fillna(
        0
    ).infer_objects(copy=False) * (UNDERDOG_BOOST_BASELINE if platform == "Underdog" else 1)
    offer_df["Boost"] = offer_df.apply(
        lambda x: (
            (x["Boost_Over"] if x["Bet"] == "Over" else x["Boost_Under"])
            if not np.isnan(x["Boost_Over"])
            else x["Boost"]
        ),
        axis=1,
    )

    offer_df["Model"] = offer_df["Model P"] * offer_df["Boost"]
    offer_df.loc[(offer_df["Bet"] == "Under"), "Books"] = (
        1 - offer_df.loc[(offer_df["Bet"] == "Under"), "Books"]
    )
    offer_df["Books P"] = offer_df["Books"].fillna(_BOOK_PRIOR_PROB)
    offer_df["Books"] = offer_df["Books P"] * offer_df["Boost"]
    offer_df["K"] = (offer_df["Model"] - 1) / (offer_df["Boost"] - 1)
    offer_df["Distance"] = offer_df["Boost"] / UNDERDOG_BOOST_BASELINE
    offer_df.loc[offer_df["Distance"] < 1, "Distance"] = (
        1 / offer_df.loc[offer_df["Distance"] < 1, "Distance"]
    )
    offer_df = (
        offer_df.loc[offer_df["Boost"] <= _MAX_UNDERDOG_BOOST]
        .sort_values("Distance", ascending=True)
        .groupby("Player")
        .head(_MAX_OFFERS_PER_PLAYER)
    )

    offer_df["Avg 5"] = offer_df["Avg5"] - offer_df["Line"]
    offer_df["Avg H2H"] = offer_df["AvgH2H"] - offer_df["Line"]
    offer_df.loc[offer_df["H2HPlayed"] == 0, "Avg H2H"] = 0
    offer_df["O/U"] = offer_df["Total"] / totals_map.get(league, 1)
    offer_df["DVPOA"] = offer_df["Defense position"]
    if "Player position" not in offer_df:
        offer_df["Player position"] = -1

    offer_df["Player position"] = offer_df["Player position"].astype("category")
    offer_df["Player position"] = (
        offer_df["Player position"].cat.set_categories(range(-1, 5)).fillna(-1).astype(int)
    )
    _annotate_display_shape(offer_df, dist)

    offer_df["Dist"] = dist
    offer_df["CV"] = cv
    offer_df["Gate"] = offer_df.get("Model Gate", np.nan)
    offer_df["Temperature"] = temperature
    offer_df["Disp Cal"] = dispersion_cal
    offer_df["Step"] = step

    return offer_df[
        [
            "League",
            "Date",
            "Team",
            "Opponent",
            "Player",
            "Market",
            "Line",
            "Boost",
            "Bet",
            "Books",
            "Model",
            "Avg 5",
            "Avg H2H",
            "Moneyline",
            "O/U",
            "DVPOA",
            "Player position",
            "Model EV",
            "Model Param",
            "Model STD",
            "Model P",
            "Push P",
            "Books EV",
            "Books P",
            "K",
            "Dist",
            "CV",
            "Gate",
            "Temperature",
            "Disp Cal",
            "Step",
        ]
    ].to_dict("records")


def _odds_from_boost(o: dict) -> np.ndarray:
    p = [
        _BOOK_PRIOR_PROB / o.get("Boost_Under", 1)
        if o.get("Boost_Under", 1) > 0
        else 1 - _BOOK_PRIOR_PROB / o.get("Boost_Over", 1),
        _BOOK_PRIOR_PROB / o.get("Boost_Over", 1)
        if o.get("Boost_Over", 1) > 0
        else 1 - _BOOK_PRIOR_PROB / o.get("Boost_Under", 1),
    ]
    return p / np.sum(p)


def _col_or_none(df: pd.DataFrame, col: str, active: bool = True) -> np.ndarray | None:
    return df[col].to_numpy() if active and col in df.columns else None


def _build_prob_params(
    filedict: dict,
    market: str,
    stat_data,
    playerStats: pd.DataFrame,
    dist: str,
    offset_meta,
    normalized: bool,
) -> pd.DataFrame:
    if market in stat_data.volume_stats:
        prob_params = pd.DataFrame(index=playerStats.index)
        prob_params = prob_params.join(stat_data.playerProfile[f"proj {market} mean"])
        if f"proj {market} std" in stat_data.playerProfile.columns:
            prob_params = prob_params.join(stat_data.playerProfile[f"proj {market} std"])
        prob_params.rename(
            columns={f"proj {market} mean": "Model EV", f"proj {market} std": "Model Param"},
            inplace=True,
        )
        return prob_params

    model = filedict["model"]
    categories = ["Home", "Player position"]
    if "Player position" not in playerStats.columns:
        categories.remove("Player position")
    for c in categories:
        playerStats[c] = playerStats[c].astype("category")

    if getattr(model, "is_hurdle", False):
        # HurdleZINB seeds its internal NegBin from its own X; the external
        # set_model_start_values would treat it as a plain LSS and fail.
        model.set_model_start_values(playerStats)
    else:
        set_model_start_values(
            model,
            dist,
            playerStats,
            normalized=normalized,
            offset_mode=bool(offset_meta and offset_meta.get("method") == "eb_additive"),
        )
    prob_params = model.predict(playerStats, pred_type="parameters")
    prob_params.index = playerStats.index
    return prob_params


def _book_evs_for_players(
    playerStats: pd.DataFrame,
    offer_df: pd.DataFrame,
    market: str,
    dist: str,
    cv: float,
    hist_gate: float,
    dateMap: dict,
    stat_data,
) -> list:
    evs = []
    for player in playerStats.index:
        date = dateMap.get(player, "")
        ev = archive.get_ev(stat_data.league, market, date, player)
        line = archive.get_line(stat_data.league, market, date, player)
        if np.isnan(ev):
            ev = stat_data.check_combo_markets(market, player, date)
        if line <= 0:
            line = np.max([playerStats.loc[player, "Avg10"], 0.5])
        if (ev <= 0 or np.isnan(ev)) and player in offer_df.index:
            o = offer_df.loc[player]
            if isinstance(o, pd.DataFrame):
                o = o.iloc[0]
            ev = get_ev(
                line,
                _odds_from_boost(o.to_dict())[0],
                stat_cv[stat_data.league].get(market, 1),
                dist=dist,
                gate=hist_gate or None,
            )
        evs.append(ev)
    return evs


def _set_decoded(prob_params: pd.DataFrame, decoded, shape_col: str, shape_val) -> None:
    prob_params["Model EV"] = decoded.ev
    prob_params[shape_col] = shape_val
    if decoded.gate is not None:
        prob_params["Model Gate"] = decoded.gate


def _decode_model_params(
    prob_params: pd.DataFrame,
    dist: str,
    playerStats: pd.DataFrame,
    hist_gate: float,
    offset_meta,
    target_normalization: str,
) -> None:
    # The column guards skip the volume_stats branch, which already set Model EV
    # from the player profile rather than a fitted distribution.
    if dist in ("NegBin", "ZINB") and "total_count" in prob_params.columns:
        decoded = decode_predictive_mean(prob_params, dist)
        _set_decoded(prob_params, decoded, "Model R", decoded.r)
    elif dist in ("Gamma", "ZAGamma") and "concentration" in prob_params.columns:
        decoded = decode_predictive_mean(prob_params, dist)
        _set_decoded(prob_params, decoded, "Model Alpha", decoded.alpha)
    elif dist == "SkewNormal" and "loc" in prob_params.columns:
        # Dispatch SkewNormal loc/scale through the baselines registry so the
        # train-side forward transform and predict-side inverse cannot drift.
        _decode_skewnormal(prob_params, playerStats, hist_gate, offset_meta, target_normalization)


def _clamp_shape_ceiling(offer_df: pd.DataFrame, dist: str, shape_ceiling) -> None:
    if shape_ceiling is None:
        return
    if dist in ("NegBin", "ZINB") and "Model R" in offer_df.columns:
        offer_df["Model R"] = np.minimum(offer_df["Model R"], shape_ceiling)
    elif dist in ("Gamma", "ZAGamma") and "Model Alpha" in offer_df.columns:
        offer_df["Model Alpha"] = np.minimum(offer_df["Model Alpha"], shape_ceiling)


def _zi_kwargs(offer_df: pd.DataFrame, dist: str, hist_gate: float) -> dict:
    if dist == "SkewNormal":
        return {"gate_book": hist_gate} if hist_gate > GATE_PUBLISH_THRESHOLD else {}
    if dist in ("ZINB", "ZAGamma") and "Model Gate" in offer_df.columns:
        return {"gate_model": offer_df["Model Gate"].to_numpy(), "gate_book": hist_gate}
    return {}


def _blend_with_book(
    offer_df: pd.DataFrame, dist: str, model_weight: float, cv: float, hist_gate: float
):
    model_ev = offer_df["Model EV"].to_numpy()
    books_ev = offer_df["Books EV"].fillna(offer_df["Model EV"]).to_numpy()
    zi = _zi_kwargs(offer_df, dist, hist_gate)
    if dist == "SkewNormal":
        base_mean, sigma_blend, skew_blend, gate_blend = fused_loc(
            model_weight,
            model_ev,
            books_ev,
            cv,
            "SkewNormal",
            sigma=_col_or_none(offer_df, "Model Sigma"),
            skew_alpha=_col_or_none(offer_df, "Model Skew"),
            **zi,
        )
        offer_df["Model Sigma"] = sigma_blend
        offer_df["Model Skew"] = skew_blend
    elif dist in ("NegBin", "ZINB"):
        r_blend, p_blend, gate_blend = fused_loc(
            model_weight,
            model_ev,
            books_ev,
            cv,
            "NegBin",
            r=_col_or_none(offer_df, "Model R"),
            **zi,
        )
        base_mean = r_blend * (1 - p_blend) / p_blend
        offer_df["Model R"] = r_blend
    else:
        alpha_blend, beta_blend, gate_blend = fused_loc(
            model_weight,
            model_ev,
            books_ev,
            cv,
            "Gamma",
            alpha=_col_or_none(offer_df, "Model Alpha"),
            **zi,
        )
        base_mean = alpha_blend / beta_blend
        offer_df["Model Alpha"] = alpha_blend
    if gate_blend is not None:
        offer_df["Model EV"] = (1 - gate_blend) * base_mean
        offer_df["Model Gate"] = gate_blend
    else:
        offer_df["Model EV"] = base_mean
    return base_mean


def _dispersion_calibrate(offer_df: pd.DataFrame, dist: str, dispersion_cal: float) -> None:
    # SkewNormal calibrates via CRPS, not a shape multiplier — skip it.
    if dispersion_cal == 1.0 or dist == "SkewNormal":
        return
    if dist in ("NegBin", "ZINB") and "Model R" in offer_df.columns:
        offer_df["Model R"] = offer_df["Model R"] * dispersion_cal
    elif dist in ("Gamma", "ZAGamma") and "Model Alpha" in offer_df.columns:
        offer_df["Model Alpha"] = offer_df["Model Alpha"] * dispersion_cal


def _model_over_and_push(offer_df: pd.DataFrame, dist: str, cv: float, step, base_mean):
    r = _col_or_none(offer_df, "Model R", dist in ("NegBin", "ZINB"))
    alpha = _col_or_none(offer_df, "Model Alpha", dist in ("Gamma", "ZAGamma"))
    sigma = _col_or_none(offer_df, "Model Sigma", dist == "SkewNormal")
    skew = _col_or_none(offer_df, "Model Skew", dist == "SkewNormal")
    gate = _col_or_none(offer_df, "Model Gate", dist in ("ZINB", "ZAGamma", "SkewNormal"))
    line = offer_df["Line"].to_numpy()
    if dist == "SkewNormal":
        raw_under = get_odds(
            line, base_mean, dist, cv=cv, step=step, sigma=sigma, skew_alpha=skew, gate=gate
        )
    else:
        raw_under = get_odds(line, base_mean, dist, cv, alpha=alpha, step=step, r=r, gate=gate)
    # Push prob for integer-line discrete markets (continuous dists return zero);
    # correlation.py uses it for the Underdog "push drops one leg" rule.
    push = get_push_prob(line, base_mean, dist, cv=cv, r=r, sigma=sigma, skew_alpha=skew, gate=gate)
    return 1 - raw_under, push


def model_prob(
    offers: list[dict],
    league: str,
    market: str,
    platform: str,
    stat_data,
    playerStats: pd.DataFrame,
) -> list[dict]:
    """Score a batch of offers with the trained distributional model.

    Loads the model pickle for ``(league, market)``, runs LightGBMLSS
    prediction on ``playerStats``, blends the model distribution with the
    bookmaker-implied distribution via ``fused_loc``, applies temperature
    scaling, and returns a list of offer dicts augmented with scoring
    columns (``Model``, ``Books``, ``Model EV``, ``Bet``, ``K``, etc.).

    Returns an empty list when no model file exists or when the joined
    DataFrame is empty after filtering.
    """
    dateMap = {x["Player"]: x["Date"] for x in offers}
    market = normalize_market(league, market, platform)
    filename = market_file_slug(league, market)
    filepath = model_pickle_path(league, market)
    if not os.path.isfile(filepath):
        logger.warning(f"{filename} missing")
        return []

    offer_df = pd.DataFrame(offers)
    offer_df.index = offer_df.Player
    if "yards" in market:
        offer_df = offer_df.loc[
            (offer_df.Player.str.contains("vs.")) | (offer_df.Line > _MIN_YARDS_LINE)
        ]

    with open(filepath, "rb") as infile:
        filedict = pickle.load(infile)
    cv = filedict["cv"]
    model_weight = filedict["weight"]
    temperature = filedict.get("temperature", None)
    dispersion_cal = filedict.get("dispersion_cal", 1.0)
    shape_ceiling = filedict.get("shape_ceiling")
    dist = filedict["distribution"]
    step = filedict["step"]
    normalized = filedict.get("normalized", False)
    # Defaults keep legacy pickles (pre-P1, no offset_meta / target_normalization)
    # byte-identical; "target_strategy" is the pre-rename key.
    offset_meta = filedict.get("offset_meta")
    target_normalization = filedict.get(
        "target_normalization", filedict.get("target_strategy", "ratio_meanyr")
    )
    posthoc_slug = filedict.get("posthoc", "none")
    posthoc_blob = filedict.get("posthoc_blob", None)
    hist_gate = (
        stat_zi.get(league, {}).get(market, 0) if dist in ("ZINB", "ZAGamma", "SkewNormal") else 0
    )

    prob_params = _build_prob_params(
        filedict, market, stat_data, playerStats, dist, offset_meta, normalized
    )
    prob_params.sort_index(inplace=True)
    playerStats.sort_index(inplace=True)
    if "Defense position" not in playerStats:
        playerStats["Defense position"] = playerStats["Defense avg"]

    evs = _book_evs_for_players(
        playerStats, offer_df, market, dist, cv, hist_gate, dateMap, stat_data
    )
    playerStats["Books EV"] = evs
    playerStats["Books STD"] = cv * np.array(evs)

    _decode_model_params(
        prob_params, dist, playerStats, hist_gate, offset_meta, target_normalization
    )

    offer_df = offer_df.join(playerStats).join(prob_params).reset_index(drop=True)
    offer_df = offer_df.loc[~offer_df[["Books EV", "Model EV"]].isna().all(axis=1)]
    if offer_df.empty:
        return []

    offer_df["Books"] = _book_over_prob(offer_df, dist, cv, step, hist_gate or None)
    _clamp_shape_ceiling(offer_df, dist, shape_ceiling)

    base_mean = _blend_with_book(offer_df, dist, model_weight, cv, hist_gate)

    # ZI dists: book reports the non-zero component EV; scale to marginal EV.
    if hist_gate and dist in ("ZINB", "ZAGamma", "SkewNormal"):
        offer_df["Books EV"] = (1 - hist_gate) * offer_df["Books EV"]
    _dispersion_calibrate(offer_df, dist, dispersion_cal)

    raw_over, push = _model_over_and_push(offer_df, dist, cv, step, base_mean)
    offer_df["Push P"] = np.asarray(push, dtype=float)

    cal_over = apply_temperature(raw_over, temperature)
    cal_over = _apply_prob_posthoc(cal_over, posthoc_slug, posthoc_blob)
    offer_df["Model Under"] = 1 - cal_over
    offer_df["Model Over"] = 1 - offer_df["Model Under"]

    return _finalize_records(
        offer_df, league, platform, dist, cv, step, temperature, dispersion_cal
    )


def book_fallback_prob(
    offers: list[dict],
    league: str,
    market: str,
    platform: str,
    stat_data,
) -> list[dict]:
    """Score offers from book odds when no trained model exists for the cell.

    Routed to by :func:`process_offers` whenever model scoring would otherwise
    be empty (missing model pickle, or a model that matched no players). Treats
    the composite, vig-free book probability — ``archive.get_ev`` inverted
    through the cell's configured distribution — as the model prediction, so the
    leg still flows through correlation, parlay search, and the export with no
    claimed edge (``Model`` mirrors ``Books``). Returns an empty list when the
    market is unknown to ``stat_meta`` (no distribution to devig with) or no leg
    has book odds.
    """
    market = normalize_market(league, market, platform)
    dist, cv, hist_gate, step = _book_cell_params(league, market)
    if dist is None:
        return []
    gate_arg = hist_gate or None

    date_map = {x["Player"]: x["Date"] for x in offers}
    offer_df = pd.DataFrame(offers)
    offer_df.index = offer_df.Player
    if "yards" in market:
        offer_df = offer_df.loc[
            (offer_df.Player.str.contains("vs.")) | (offer_df.Line > _MIN_YARDS_LINE)
        ]
    if offer_df.empty:
        return []

    evs = _composite_book_evs(offer_df.index.unique(), league, market, date_map, stat_data)
    offer_df["Books EV"] = offer_df.index.map(evs)
    offer_df = offer_df.loc[offer_df["Books EV"].notna() & (offer_df["Books EV"] > 0)]
    if offer_df.empty:
        return []

    playerStats = stat_data.get_stats(market, offers)
    if not playerStats.empty:
        playerStats = (
            playerStats[~playerStats.index.duplicated(keep="first")]
            .fillna(0)
            .infer_objects(copy=False)
        )
        offer_df = offer_df.join(playerStats)
    offer_df = offer_df.reset_index(drop=True)

    offer_df["Books"] = _book_over_prob(offer_df, dist, cv, step, gate_arg)
    _base_ev = offer_df["Books EV"].to_numpy()
    offer_df["Push P"] = np.asarray(
        get_push_prob(offer_df["Line"].to_numpy(), _base_ev, dist, cv=cv, gate=gate_arg),
        dtype=float,
    )
    if hist_gate:
        offer_df["Books EV"] = (1 - hist_gate) * offer_df["Books EV"]

    _over = offer_df["Books"].to_numpy()
    offer_df["Model Over"] = _over
    offer_df["Model Under"] = 1 - _over
    offer_df["Model EV"] = offer_df["Books EV"]

    return _finalize_records(offer_df, league, platform, dist, cv, step, None, 1.0)
