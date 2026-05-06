"""Distributional probability scoring for a single league/market/platform batch.

:func:`model_prob` is the hot path of the prediction pipeline: it loads
the trained LightGBMLSS model pickle, runs vectorized inference on the
``playerStats`` feature matrix, blends model and bookmaker predictions
via :func:`fused_loc`, applies temperature-scaling calibration, and
returns a list of scored offer dicts ready for :func:`find_correlation`.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import os.path
import pickle

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import gamma, norm

from sportstradamus import data
from sportstradamus.helpers import (
    Archive,
    fused_loc,
    get_ev,
    get_odds,
    get_push_prob,
    set_model_start_values,
    stat_cv,
    stat_map,
    stat_zi,
)
from sportstradamus.spiderLogger import logger

archive = Archive()

# Maximum allowed model confidence before applying a boost.
_MAX_CONFIDENCE = 0.90


def model_prob(offers, league, market, platform, stat_data, playerStats):
    """Score a batch of offers with the trained distributional model.

    Loads the model pickle for ``(league, market)``, runs LightGBMLSS
    prediction on ``playerStats``, blends the model distribution with the
    bookmaker-implied distribution via ``fused_loc``, applies temperature
    scaling, and returns a list of offer dicts augmented with scoring
    columns (``Model``, ``Books``, ``Model EV``, ``Bet``, ``K``, etc.).

    Returns an empty list when no model file exists or when the joined
    DataFrame is empty after filtering.

    Args:
        offers: Raw offer dicts from the scraper.
        league: League key (e.g. ``"NBA"``).
        market: Canonical market name.
        platform: DFS platform name (e.g. ``"Underdog"``).
        stat_data: Loaded ``Stats`` instance for ``league``.
        playerStats: Feature DataFrame from ``match_offers``.

    Returns:
        list[dict]: Scored offer records, or ``[]`` on failure.
    """

    def odds_from_boost(o):
        p = [
            0.5 / o.get("Boost_Under", 1)
            if o.get("Boost_Under", 1) > 0
            else 1 - 0.5 / o.get("Boost_Over", 1),
            0.5 / o.get("Boost_Over", 1)
            if o.get("Boost_Over", 1) > 0
            else 1 - 0.5 / o.get("Boost_Under", 1),
        ]
        return p / np.sum(p)

    totals_map = archive.default_totals
    dateMap = {x["Player"]: x["Date"] for x in offers}

    market = stat_map[platform].get(market, market)
    if league == "NHL":
        market = {"AST": "assists", "PTS": "points", "BLK": "blocked"}.get(market, market)
    if league in ("NBA", "WNBA"):
        market = market.replace("underdog", "prizepicks")
    filename = "_".join([league, market]).replace(" ", "-")
    filepath = pkg_resources.files(data) / f"models/{filename}.mdl"
    offer_df = pd.DataFrame(offers)
    offer_df.index = offer_df.Player
    if "yards" in market:
        offer_df = offer_df.loc[(offer_df.Player.str.contains("vs.")) | (offer_df.Line > 8)]
    if os.path.isfile(filepath):
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
        hist_gate = (
            stat_zi.get(league, {}).get(market, 0)
            if dist in ("ZINB", "ZAGamma", "SkewNormal")
            else 0
        )

        if market in stat_data.volume_stats:
            prob_params = pd.DataFrame(index=playerStats.index)
            prob_params = prob_params.join(stat_data.playerProfile[f"proj {market} mean"])
            if f"proj {market} std" in stat_data.playerProfile.columns:
                prob_params = prob_params.join(stat_data.playerProfile[f"proj {market} std"])

            prob_params.rename(
                columns={f"proj {market} mean": "Model EV", f"proj {market} std": "Model Param"},
                inplace=True,
            )

        else:
            model = filedict["model"]

            categories = ["Home", "Player position"]
            if "Player position" not in playerStats.columns:
                categories.remove("Player position")
            for c in categories:
                playerStats[c] = playerStats[c].astype("category")

            set_model_start_values(model, dist, playerStats, normalized=normalized)

            prob_params = model.predict(playerStats, pred_type="parameters")
            prob_params.index = playerStats.index

        prob_params.sort_index(inplace=True)
        playerStats.sort_index(inplace=True)

        if "Defense position" not in playerStats:
            playerStats["Defense position"] = playerStats["Defense avg"]

        evs = []
        for player in playerStats.index:
            ev = archive.get_ev(stat_data.league, market, dateMap.get(player, ""), player)
            line = archive.get_line(stat_data.league, market, dateMap.get(player, ""), player)
            if np.isnan(ev):
                ev = stat_data.check_combo_markets(market, player, dateMap.get(player, ""))
            if line <= 0:
                line = np.max([playerStats.loc[player, "Avg10"], 0.5])
            if (ev <= 0 or np.isnan(ev)) and player in offer_df.index:
                o = offer_df.loc[player]
                if isinstance(o, pd.DataFrame):
                    o = o.iloc[0]
                ev = get_ev(
                    line,
                    odds_from_boost(o.to_dict())[0],
                    stat_cv[stat_data.league].get(market, 1),
                    dist=dist,
                    gate=hist_gate or None,
                )
            elif hist_gate and ev > 0:
                pass  # archive EVs are already base means

            evs.append(ev)

        playerStats["Books EV"] = evs
        playerStats["Books STD"] = cv * np.array(evs)

        # NegBin/ZINB: mean = r * probs / (1 - probs) (PyTorch convention)
        if dist in ("NegBin", "ZINB") and "total_count" in prob_params.columns:
            base_ev = prob_params["total_count"] * prob_params["probs"] / (1 - prob_params["probs"])
            prob_params["Model EV"] = base_ev
            if dist == "ZINB":
                prob_params["Model Gate"] = prob_params["gate"]
            prob_params["Model R"] = prob_params["total_count"]

        # Gamma/ZAGamma: mean = concentration / rate
        if dist in ("Gamma", "ZAGamma") and "concentration" in prob_params.columns:
            base_ev = prob_params["concentration"] / prob_params["rate"]
            prob_params["Model EV"] = base_ev
            if dist == "ZAGamma":
                prob_params["Model Gate"] = prob_params["gate"]
            prob_params["Model Alpha"] = prob_params["concentration"]

        # SkewNormal: denormalize loc/scale then compute EV = loc + scale * delta * sqrt(2/pi)
        if dist == "SkewNormal" and "loc" in prob_params.columns:
            denom_col = (
                "MeanYr_nonzero"
                if (hist_gate > 0.05 and "MeanYr_nonzero" in playerStats.columns)
                else "MeanYr"
            )
            meanyr_vals = playerStats[denom_col].clip(lower=0.5).values
            loc_abs = prob_params["loc"].values * meanyr_vals
            scale_abs = prob_params["scale"].values * meanyr_vals
            alpha_sn = prob_params["alpha"].values

            delta = alpha_sn / np.sqrt(1 + alpha_sn**2)
            base_ev = loc_abs + scale_abs * delta * np.sqrt(2 / np.pi)

            prob_params["Model EV"] = base_ev
            prob_params["Model Sigma"] = scale_abs
            prob_params["Model Skew"] = alpha_sn
            if hist_gate > 0.02:
                prob_params["Model Gate"] = hist_gate

        offer_df = offer_df.join(playerStats).join(prob_params).reset_index(drop=True)
        offer_df = offer_df.loc[~offer_df[["Books EV", "Model EV"]].isna().all(axis=1)]
        if offer_df.empty:
            return []

        # Book-side probability
        if dist == "SkewNormal":
            offer_df["Books"] = offer_df.apply(
                lambda x: 1
                - get_odds(
                    x["Line"],
                    x["Books EV"],
                    dist,
                    cv=cv,
                    step=step,
                    sigma=x["Books EV"] * cv,
                    skew_alpha=0,
                    gate=hist_gate or None,
                ),
                axis=1,
            )
        else:
            offer_df["Books"] = offer_df.apply(
                lambda x: 1
                - get_odds(x["Line"], x["Books EV"], dist, cv, step=step, gate=hist_gate or None),
                axis=1,
            )

        # Clamp model shape to training-time ceiling
        if shape_ceiling is not None:
            if dist in ("NegBin", "ZINB") and "Model R" in offer_df.columns:
                offer_df["Model R"] = np.minimum(offer_df["Model R"], shape_ceiling)
            elif dist in ("Gamma", "ZAGamma") and "Model Alpha" in offer_df.columns:
                offer_df["Model Alpha"] = np.minimum(offer_df["Model Alpha"], shape_ceiling)

        # Blend model and book distributions via fused_loc
        if dist == "SkewNormal":
            _zi_kw = dict(gate_book=hist_gate) if hist_gate > 0.02 else {}
            blended_base_mean, sigma_blend, skew_blend, gate_blend = fused_loc(
                model_weight,
                offer_df["Model EV"].to_numpy(),
                offer_df["Books EV"].fillna(offer_df["Model EV"]).to_numpy(),
                cv,
                "SkewNormal",
                sigma=offer_df["Model Sigma"].to_numpy()
                if "Model Sigma" in offer_df.columns
                else None,
                skew_alpha=offer_df["Model Skew"].to_numpy()
                if "Model Skew" in offer_df.columns
                else None,
                **_zi_kw,
            )
            if gate_blend is not None:
                offer_df["Model EV"] = (1 - gate_blend) * blended_base_mean
                offer_df["Model Gate"] = gate_blend
            else:
                offer_df["Model EV"] = blended_base_mean
            offer_df["Model Sigma"] = sigma_blend
            offer_df["Model Skew"] = skew_blend

        elif dist in ("NegBin", "ZINB"):
            _zi_kw = (
                dict(gate_model=offer_df["Model Gate"].to_numpy(), gate_book=hist_gate)
                if dist == "ZINB" and "Model Gate" in offer_df.columns
                else {}
            )
            r_blend, p_blend, gate_blend = fused_loc(
                model_weight,
                offer_df["Model EV"].to_numpy(),
                offer_df["Books EV"].fillna(offer_df["Model EV"]).to_numpy(),
                cv,
                "NegBin",
                r=offer_df["Model R"].to_numpy() if "Model R" in offer_df.columns else None,
                **_zi_kw,
            )
            blended_base_mean = r_blend * (1 - p_blend) / p_blend
            if gate_blend is not None:
                offer_df["Model EV"] = (1 - gate_blend) * blended_base_mean
                offer_df["Model Gate"] = gate_blend
            else:
                offer_df["Model EV"] = blended_base_mean
            offer_df["Model R"] = r_blend
        else:
            _zi_kw = (
                dict(gate_model=offer_df["Model Gate"].to_numpy(), gate_book=hist_gate)
                if dist == "ZAGamma" and "Model Gate" in offer_df.columns
                else {}
            )
            alpha_blend, beta_blend, gate_blend = fused_loc(
                model_weight,
                offer_df["Model EV"].to_numpy(),
                offer_df["Books EV"].fillna(offer_df["Model EV"]).to_numpy(),
                cv,
                "Gamma",
                alpha=offer_df["Model Alpha"].to_numpy()
                if "Model Alpha" in offer_df.columns
                else None,
                **_zi_kw,
            )
            blended_base_mean = alpha_blend / beta_blend
            if gate_blend is not None:
                offer_df["Model EV"] = (1 - gate_blend) * blended_base_mean
                offer_df["Model Gate"] = gate_blend
            else:
                offer_df["Model EV"] = blended_base_mean
            offer_df["Model Alpha"] = alpha_blend

        # Convert Books EV from base mean to overall mean for ZI/hurdle dists
        if hist_gate and dist in ("ZINB", "ZAGamma", "SkewNormal"):
            offer_df["Books EV"] = (1 - hist_gate) * offer_df["Books EV"]

        # Dispersion calibration (SkewNormal uses CRPS — skip)
        if dispersion_cal != 1.0 and dist != "SkewNormal":
            if dist in ("NegBin", "ZINB") and "Model R" in offer_df.columns:
                offer_df["Model R"] = offer_df["Model R"] * dispersion_cal
            elif dist in ("Gamma", "ZAGamma") and "Model Alpha" in offer_df.columns:
                offer_df["Model Alpha"] = offer_df["Model Alpha"] * dispersion_cal

        # Raw distributional probability
        _r = (
            offer_df["Model R"].to_numpy()
            if (dist in ("NegBin", "ZINB") and "Model R" in offer_df.columns)
            else None
        )
        _alpha = (
            offer_df["Model Alpha"].to_numpy()
            if (dist in ("Gamma", "ZAGamma") and "Model Alpha" in offer_df.columns)
            else None
        )
        _sigma = (
            offer_df["Model Sigma"].to_numpy()
            if (dist == "SkewNormal" and "Model Sigma" in offer_df.columns)
            else None
        )
        _skew = (
            offer_df["Model Skew"].to_numpy()
            if (dist == "SkewNormal" and "Model Skew" in offer_df.columns)
            else None
        )
        _gate = (
            offer_df["Model Gate"].to_numpy()
            if (dist in ("ZINB", "ZAGamma", "SkewNormal") and "Model Gate" in offer_df.columns)
            else None
        )
        _model_ev = blended_base_mean  # base mean (gate handled separately inside get_odds)

        if dist == "SkewNormal":
            _raw_under = get_odds(
                offer_df["Line"].to_numpy(),
                _model_ev,
                dist,
                cv=cv,
                step=step,
                sigma=_sigma,
                skew_alpha=_skew,
                gate=_gate,
            )
        else:
            _raw_under = get_odds(
                offer_df["Line"].to_numpy(),
                _model_ev,
                dist,
                cv,
                alpha=_alpha,
                step=step,
                r=_r,
                gate=_gate,
            )
        _raw_over = 1 - _raw_under

        # Push probability for integer-line discrete-distribution markets. Used
        # by :mod:`sportstradamus.prediction.correlation` to apply the Underdog
        # "push drops one leg" rule. Continuous distributions return zero.
        _push = get_push_prob(
            offer_df["Line"].to_numpy(),
            _model_ev,
            dist,
            cv=cv,
            r=_r,
            sigma=_sigma,
            skew_alpha=_skew,
            gate=_gate,
        )
        offer_df["Push P"] = np.asarray(_push, dtype=float)

        if temperature is not None:
            _raw_over_clipped = np.clip(_raw_over, 1e-6, 1 - 1e-6)
            _cal_over = expit(logit(_raw_over_clipped) / temperature)
            offer_df["Model Under"] = 1 - _cal_over
        else:
            offer_df["Model Under"] = _raw_under

        offer_df["Model Over"] = 1 - offer_df["Model Under"]

        offer_df["Model Over"] = offer_df["Model Over"].clip(upper=_MAX_CONFIDENCE)
        offer_df["Model Under"] = offer_df["Model Under"].clip(upper=_MAX_CONFIDENCE)

        offer_df["Model P"] = offer_df[["Model Over", "Model Under"]].max(axis=1)
        offer_df["Bet"] = offer_df[["Model Over", "Model Under"]].idxmax(axis=1).str[6:]

        if "Boost" in offer_df.columns:
            offer_df.loc[offer_df["Boost"] == 1, ["Boost_Under", "Boost_Over"]] = 1
        offer_df[["Boost_Under", "Boost_Over"]] = offer_df[["Boost_Under", "Boost_Over"]].fillna(
            0
        ).infer_objects(copy=False) * (1.78 if platform == "Underdog" else 1)
        offer_df["Boost"] = offer_df.apply(
            lambda x: (x["Boost_Over"] if x["Bet"] == "Over" else x["Boost_Under"])
            if not np.isnan(x["Boost_Over"])
            else x["Boost"],
            axis=1,
        )

        offer_df["Model"] = offer_df["Model P"] * offer_df["Boost"]
        offer_df.loc[(offer_df["Bet"] == "Under"), "Books"] = (
            1 - offer_df.loc[(offer_df["Bet"] == "Under"), "Books"]
        )
        offer_df["Books P"] = offer_df["Books"].fillna(0.5)
        offer_df["Books"] = offer_df["Books P"] * offer_df["Boost"]
        offer_df["K"] = (offer_df["Model"] - 1) / (offer_df["Boost"] - 1)
        offer_df["Distance"] = offer_df["Boost"] / 1.78
        offer_df.loc[offer_df["Distance"] < 1, "Distance"] = (
            1 / offer_df.loc[offer_df["Distance"] < 1, "Distance"]
        )
        offer_df = (
            offer_df.loc[offer_df["Boost"] <= 3.65]
            .sort_values("Distance", ascending=True)
            .groupby("Player")
            .head(3)
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
        if dist in ("NegBin", "ZINB"):
            offer_df["Model Param"] = offer_df["Model R"]
        elif dist == "SkewNormal":
            offer_df["Model Param"] = offer_df["Model Sigma"]
        else:
            offer_df["Model Param"] = offer_df["Model Alpha"]

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

    else:
        logger.warning(f"{filename} missing")
        return []
