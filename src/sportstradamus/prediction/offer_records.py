"""Scored offers to export records: the book leg's devigged price, boosts, and payouts.

:func:`book_over_prob` devigs a row's ``Market Projection`` into the ``Market EV``
over-probability both scoring paths compare against. :func:`finalize_records` is the
last stage both paths share — it takes a frame that already carries ``Model Over``/
``Market EV``/``Projection`` — whether those came from a trained model or from a
devigged book quote — and turns it into the record dicts ``find_correlation``, the
parlay search, and the dashboard export consume: boost-aware payouts, the chosen side,
and the per-player deduplication.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sportstradamus.helpers import (
    UNDERDOG_BOOST_BASELINE,
    DecodedParams,
    LazyArchive,
    book_skewnormal_shape,
    get_odds,
    predictive_std,
)
from sportstradamus.helpers.distributions import dfs_boost_probs
from sportstradamus.spiderLogger import logger

# LazyArchive defers DuckDB lock acquisition until the first attribute
# access. See LazyArchive docstring in helpers/archive.py.
archive = LazyArchive()

# Maximum allowed model confidence before applying a boost.
_MAX_CONFIDENCE = 0.90

# Maximum Underdog boost multiplier kept when de-duplicating a player's offers.
# Above this the promo is an outlier (e.g. a discounted special) that distorts
# the per-player distance ranking, so it is filtered out.
_MAX_UNDERDOG_BOOST = 3.65

# Drop an unquoted single-player leg when the model disagrees with the payout-implied
# probability by more than this. Underdog prices its own boosts near-fair, so a model
# >15 pts from the only price available has no independent support — these were the
# +51%-edge phantom picks the flat 0.5 prior manufactured.
UNQUOTED_BOOK_DISAGREEMENT_MAX = 0.15

# Maximum scored offers retained per player after boost-distance deduplication.
_MAX_OFFERS_PER_PLAYER: int = 3

# "Player position" category domain: -1 (unknown/missing) through the highest in-league
# position code (e.g. NHL_POSITION_CODES tops out at 4). Fixing the categorical domain
# here — rather than letting pandas infer it from the batch — keeps a position code
# that happens to be absent from one board from silently dropping out of the category.
_POSITION_CATEGORY_RANGE = range(-1, 5)


def book_over_prob(
    offer_df: pd.DataFrame,
    dist: str,
    cv: float,
    step: float,
    gate: float | None,
    league: str,
    market: str,
) -> pd.Series:
    """Devigged book probability of the over per row, inverted from ``Market Projection``.

    Inverts the composite book EV through the cell distribution at each row's line. Shared by
    :func:`model_prob` and :func:`book_fallback_prob`. For SkewNormal the book ``(sigma, skew)``
    comes from the per-cell fitted shape evaluated at the book mean (:func:`book_skewnormal_shape`);
    an unfitted cell returns ``(mean*cv, 0)``, the legacy symmetric constant-CV read. A NaN
    ``Market Projection`` (no independent quote) stays NaN so ``_finalize_records`` prices the
    row payout-implied.
    """
    quoted = offer_df.loc[offer_df["Market Projection"].notna()]
    if quoted.empty:
        return pd.Series(np.nan, index=offer_df.index)
    offer_df = quoted
    if dist == "SkewNormal":
        # The archive encodes SkewNormal sportsbook EVs without the model's
        # external hurdle. Keep the book endpoint gate-free on decode too.
        gate = None

        def _sn_over(x):
            sigma, skew = book_skewnormal_shape(league, market, x["Market Projection"], cv)
            return 1 - get_odds(
                x["Line"],
                x["Market Projection"],
                dist,
                cv=cv,
                step=step,
                sigma=float(sigma),
                skew_alpha=float(skew),
                gate=gate,
            )

        return offer_df.apply(_sn_over, axis=1)
    return offer_df.apply(
        lambda x: 1 - get_odds(x["Line"], x["Market Projection"], dist, cv, step=step, gate=gate),
        axis=1,
    )


def col_or_none(df: pd.DataFrame, col: str, active: bool = True) -> np.ndarray | None:
    """``df[col]`` as an array, or ``None`` when it is absent or ``active`` is false."""
    return df[col].to_numpy() if active and col in df.columns else None


def _annotate_display_shape(offer_df: pd.DataFrame, dist: str) -> None:
    """Set the display-only ``Model Param`` and ``Projection STD`` columns in place.

    Reads the distribution-shape parameters the model emitted (``Model R`` /
    ``Model Phi`` / ``Model Sigma`` / ``Model Alpha``); falls back to NaN when they are absent,
    as on a book-fallback record. These columns feed the dashboard detail popup
    only — no scoring path reads them.
    """
    if dist in ("NegBin", "ZINB") and "Model R" in offer_df.columns:
        offer_df["Model Param"] = offer_df["Model R"]
    elif dist == "DPO" and "Model Phi" in offer_df.columns:
        offer_df["Model Param"] = offer_df["Model Phi"]
    elif dist == "SkewNormal" and "Model Sigma" in offer_df.columns:
        offer_df["Model Param"] = offer_df["Model Sigma"]
    elif "Model Alpha" in offer_df.columns:
        offer_df["Model Param"] = offer_df["Model Alpha"]
    else:
        offer_df["Model Param"] = np.nan

    offer_df["Projection STD"] = predictive_std(
        # "Model Alpha" is Gamma's concentration; the family names its own shape,
        # so an absent column decodes to NaN rather than borrowing another's.
        dist,
        DecodedParams(
            ev=offer_df["Projection"].to_numpy(dtype=float),
            r=col_or_none(offer_df, "Model R"),
            alpha=col_or_none(offer_df, "Model Alpha"),
            sigma=col_or_none(offer_df, "Model Sigma"),
            skew=col_or_none(offer_df, "Model Skew"),
            phi=col_or_none(offer_df, "Model Phi"),
        ),
    )


def finalize_records(
    offer_df: pd.DataFrame,
    league: str,
    platform: str,
    dist: str,
    cv: float,
    step: float,
    temperature: float | None,
    dispersion_cal: float,
    model_version: str,
    pit_recal_blob: dict | None = None,
) -> list[dict]:
    """Resolve the bet side, apply boosts, and project the export schema.

    Shared tail of :func:`model_prob` and :func:`book_fallback_prob`. Expects
    ``offer_df`` to already carry ``Model Over`` / ``Model Under`` (win
    probabilities for each side), the raw ``Market EV`` over-probability (NaN when no
    book quoted the leg — filled with the payout-implied probability of the chosen
    side, and the unquoted-disagreement gate then drops single-player rows the model
    disagrees with beyond ``UNQUOTED_BOOK_DISAGREEMENT_MAX``), ``Push Prob``,
    ``Projection`` and ``Market Projection``. Feature-derived passenger columns absent on a
    book-fallback record (built without a feature matrix) are filled neutral so
    correlation and the dashboard read sane values — ``Player position`` in
    particular must stay an int, never NaN, or correlation's position map breaks.
    """
    totals_map = archive.default_totals
    for _col in ("Avg5", "AvgH2H", "H2HPlayed", "Total", "Defense position", "Moneyline"):
        if _col not in offer_df.columns:
            offer_df[_col] = np.nan
    if "Home" not in offer_df.columns:
        # Bool, not NaN: dashboard/narrative.py's match_label does `'v' if home else '@'`,
        # and NaN is truthy in Python — it would render every no-Home row as the host.
        offer_df["Home"] = False
    if "Commence" not in offer_df.columns:
        # Only Underdog threads a tip-off timestamp (books.py::get_ud); Sleeper and
        # book-fallback rows default to empty so the export projection stays uniform
        # (the dashboard coerces "" → NaT → non-urgent countdown).
        offer_df["Commence"] = ""

    offer_df["Model Over"] = offer_df["Model Over"].clip(upper=_MAX_CONFIDENCE)
    offer_df["Model Under"] = offer_df["Model Under"].clip(upper=_MAX_CONFIDENCE)

    offer_df["Win Prob"] = offer_df[["Model Over", "Model Under"]].max(axis=1)
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

    offer_df["Model EV"] = offer_df["Win Prob"] * offer_df["Boost"]
    offer_df.loc[(offer_df["Bet"] == "Under"), "Market EV"] = (
        1 - offer_df.loc[(offer_df["Bet"] == "Under"), "Market EV"]
    )
    unquoted = offer_df["Market EV"].isna()
    # Post-conversion Boost_Over/Boost_Under are full decimal odds on every platform, so
    # the payout-implied probability of the chosen side is the platform's own price for
    # the leg — the only price available when no book quoted it.
    implied = offer_df.apply(
        lambda x: dfs_boost_probs(x["Boost_Over"], x["Boost_Under"])[
            1 if x["Bet"] == "Under" else 0
        ],
        axis=1,
    )
    offer_df["Market Prob"] = offer_df["Market EV"].fillna(implied)
    phantom = (
        unquoted
        & ~offer_df["Player"].str.contains(" vs. ", regex=False)
        & ((offer_df["Win Prob"] - offer_df["Market Prob"]).abs() > UNQUOTED_BOOK_DISAGREEMENT_MAX)
    )
    if phantom.any():
        logger.warning(
            f"{league} {offer_df['Market'].iat[0]}: dropped {int(phantom.sum())} unquoted "
            "offer(s) disagreeing with the payout-implied price"
        )
        offer_df = offer_df.loc[~phantom]
    offer_df["Market EV"] = offer_df["Market Prob"] * offer_df["Boost"]
    offer_df["Kelly"] = (offer_df["Model EV"] - 1) / (offer_df["Boost"] - 1)
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
        offer_df["Player position"]
        .cat.set_categories(_POSITION_CATEGORY_RANGE)
        .fillna(-1)
        .astype(int)
    )
    _annotate_display_shape(offer_df, dist)

    offer_df["Dist"] = dist
    offer_df["CV"] = cv
    offer_df["Gate"] = offer_df.get("Model Gate", np.nan)
    offer_df["Temperature"] = temperature
    offer_df["Disp Cal"] = dispersion_cal
    offer_df["Step"] = step
    offer_df["Model Version"] = model_version
    # §6.1 Rung C: the whole-CDF recal map g (a cdf_recal_isotonic cell's served
    # warp) so the deep-dive can draw g∘F, the distribution the model actually
    # serves. Null for every other cell — the raw parametric curve is exact there.
    offer_df["Model PIT Recal"] = json.dumps(pit_recal_blob) if pit_recal_blob else None

    return offer_df[
        [
            "League",
            "Date",
            "Commence",
            "Team",
            "Opponent",
            "Home",
            "Player",
            "Market",
            "Line",
            "Boost",
            "Boost_Over",
            "Boost_Under",
            "Bet",
            "Market EV",
            "Model EV",
            "Avg 5",
            "Avg H2H",
            "Moneyline",
            "O/U",
            "DVPOA",
            "Player position",
            "Projection",
            "Model Param",
            "Projection STD",
            "Win Prob",
            "Push Prob",
            "Market Projection",
            "Market Prob",
            "Kelly",
            "Dist",
            "CV",
            "Gate",
            "Temperature",
            "Disp Cal",
            "Step",
            "Model Version",
            "Model PIT Recal",
        ]
    ].to_dict("records")
