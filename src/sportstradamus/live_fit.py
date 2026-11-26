"""EV's win chance: the app's price, the books' and the model's, fitted on settled posted picks.

One penalized logistic fit per league, pick kind and class of sportsbook coverage gives the
chance an offer's Over hits (docs/handoffs/profit-engine.md 5.1; the estimator and its
evidence are docs/archive/researcher_ev_fit.md). ``fit`` estimates the table, ``score``
prices offers with it on the side the model took, and ``walk_forward`` prices each game date
from earlier dates only. ``fit_gain`` and ``ev_bands`` read priced offers against their
outcomes, for the fit's acceptance tests and kill lines. ``picks`` ranks priced offers by
``EV`` to flag the board's picks (docs/handoffs/profit-engine.md 5.3). Pure.

Offers are read under history's column names: ``League``, ``Platform``, ``Payout Over`` and
``Payout Under`` (full decimal payouts, 0 on a side the app did not post), ``Bet``,
``Win Prob`` and ``Book Prob`` (both of the side ``Bet`` names), ``Model Version`` and
``Book Class`` (the last two as ``book_stamp`` writes them).
"""

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from sportstradamus.history_schema import BOOK_FALLBACK_VERSION

# The fit's coefficients, each zero at the no-evidence point, where the fitted chance is the
# app's own fair price: an intercept, the excess slope on the app's fair logit, and the
# weights on the books' and the model's disagreement with the app. A withheld cell serves the
# book consensus as its model, so its rows carry their own weight (w_fb).
TERMS: tuple[str, ...] = ("c", "a1", "w_book", "w_model", "w_fb")
# Settled legs the prior on each coefficient is worth. It pulls the fit to the no-evidence
# point; walk-forward, 4,000 scored best of 250 to 16,000 and unpenalized lost to the app's price.
PRIOR_LEGS = 4000
# Settled legs a league, kind and class needs before its fit is used.
MIN_FIT_LEGS = 400
# The trailing game dates a fit reads.
FIT_WINDOW_DAYS = 90
# One-sided 95 %: the bound the "+EV" mark and the stake read.
LOWER_BOUND_Z = 1.645
# Earlier game dates a walk-forward needs before it scores one.
MIN_TRAIN_DATES = 14
# The EV bands of the research brief's honesty test (docs/archive/researcher_ev_fit.md, AT3).
EV_BAND_EDGES: tuple[float, ...] = (0.0, 0.9, 0.94, 0.97, 1.0, np.inf)
# The most players a slate's board calls picks, per pick kind; a slate of fewer teams has one
# place a team. Walk-forward, those picks returned -3.6 % a leg where the board's 5 %-edge rule
# returned -8.4 %, and three places a game -7.5 % (docs/handoffs/profit-engine.md 5.3).
PICKS_PER_SLATE = 10

FIT_COLS: list[str] = [
    "league",
    "kind",
    "book_class",
    "n_legs",
    "n_dates",
    "first_date",
    "last_date",
    "prior_legs",
    *TERMS,
    *(f"se_{term}" for term in TERMS),
    "cov",
]
# What ``score`` prices an offer with, each for the side ``Bet`` names.
SCORE_COLS: list[str] = ["Fit Prob", "Fit Prob Low", "EV"]
GAIN_COLS: list[str] = ["league", "kind", "n_legs", "n_dates", "gain", "gain_lo", "gain_hi"]
BAND_COLS: list[str] = ["band", "n_legs", "says", "paid", "paid_lo", "paid_hi"]

_STRATUM = ["League", "Kind", "Book Class"]
_SLATE = ["League", "Date", "Platform"]
# Probabilities are clipped this far inside (0, 1) before the logit.
_PROB_EPS = 1e-3
# Floor on a regressor's mean square, so a near-constant column keeps a finite prior.
_MIN_MEAN_SQUARE = 1e-6
_NEWTON_MAX_ITER = 60
_NEWTON_TOL = 1e-8


def _logit(prob) -> np.ndarray:
    return logit(np.clip(np.asarray(prob, dtype=float), _PROB_EPS, 1 - _PROB_EPS))


def pick_kind(offers: pd.DataFrame) -> np.ndarray:
    """Each offer's pick kind, the fit's stratum of payout structure.

    Off Underdog it is the platform's name in lower case; on Underdog ``underdog_even`` when
    both sides pay the same and ``underdog_modified`` when they do not.
    """
    platform = offers["Platform"].to_numpy(str)
    # Even when both sides pay the same: the fair price is one half on every such offer.
    return np.select(
        [platform != "Underdog", np.isclose(offers["Payout Over"], offers["Payout Under"])],
        [np.char.lower(platform), "underdog_even"],
        "underdog_modified",
    )


def _terms(offers: pd.DataFrame) -> pd.DataFrame:
    """Each offer's stratum, the app's fair logit of its Over (``x0``) and the fit's regressors.

    An offer with one side posted has no fair price, so its ``x0`` and slope are NaN. A
    regressor an offer does not carry is zero: the book term without a book at the line, the
    withheld weight on a served row and the model weight on a withheld one.
    """
    payout_over = offers["Payout Over"].to_numpy(float)
    payout_under = offers["Payout Under"].to_numpy(float)
    posted = (payout_over > 0) & (payout_under > 0)
    implied_over = 1 / np.where(posted, payout_over, np.nan)
    implied_under = 1 / np.where(posted, payout_under, np.nan)
    x0 = _logit(implied_over / (implied_over + implied_under))
    over = offers["Bet"].eq("Over").to_numpy()
    win_prob = offers["Win Prob"].to_numpy(float)
    book_prob = offers["Book Prob"].to_numpy(float)
    d_model = _logit(np.where(over, win_prob, 1 - win_prob)) - x0
    d_book = _logit(np.where(over, book_prob, 1 - book_prob)) - x0
    served = offers["Model Version"].ne(BOOK_FALLBACK_VERSION).to_numpy()
    return pd.DataFrame(
        {
            "League": offers["League"].to_numpy(),
            "Kind": pick_kind(offers),
            "Book Class": offers["Book Class"].to_numpy(),
            "x0": x0,
            "c": 1.0,
            "a1": x0,
            "w_book": np.where(np.isnan(book_prob), 0.0, d_book),
            "w_model": np.where(served, d_model, 0.0),
            "w_fb": np.where(served, 0.0, d_model),
        },
        index=offers.index,
    )


def _newton(
    design: np.ndarray, hit: np.ndarray, offset: np.ndarray, precision: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Penalized logistic fit: the coefficients, the inverse Hessian and the fitted chances."""
    beta = np.zeros(design.shape[1])
    penalty = np.diag(precision)
    for _ in range(_NEWTON_MAX_ITER):
        chance = expit(offset + design @ beta)
        hessian = design.T @ (design * (chance * (1 - chance))[:, None]) + penalty
        step = np.linalg.solve(hessian, design.T @ (hit - chance) - precision * beta)
        beta += step
        if np.abs(step).max() < _NEWTON_TOL:
            break
    chance = expit(offset + design @ beta)
    hessian = design.T @ (design * (chance * (1 - chance))[:, None]) + penalty
    return beta, np.linalg.inv(hessian), chance


def fit(offers: pd.DataFrame, prior_legs: float = PRIOR_LEGS) -> pd.DataFrame:
    """The fit table: one row per league, pick kind and book class holding ``MIN_FIT_LEGS``.

    ``offers`` is settled (``Result`` Over or Under, ``Date`` its game date); the caller cuts
    the window. Offers with one side posted, or no ``Win Prob``, are left out. A Gaussian
    prior centres every coefficient on zero and is worth ``prior_legs`` of the legs that carry
    its regressor. A term whose regressor is zero on every row of a stratum is not fitted
    there and reads NaN: an even pick has no slope, a class without a book at the line no
    book weight.

    Returns:
        ``FIT_COLS``: the coefficients; ``se_*``, their standard errors clustered by game date,
        for reading the weights; and ``cov``, the fit's own covariance over the fitted terms
        in ``TERMS`` order, row by row, which ``score`` turns into each offer's lower bound.
    """
    terms = _terms(offers)
    usable = terms[["x0", "w_model", "w_fb"]].notna().all(axis=1)
    terms = terms[usable].assign(
        hit=offers.loc[usable, "Result"].eq("Over").to_numpy(float),
        Date=offers.loc[usable, "Date"].to_numpy(),
    )
    rows = []
    for (league, kind, book_class), legs in terms.groupby(_STRATUM):
        if len(legs) < MIN_FIT_LEGS:
            continue
        fitted = [term for term in TERMS if legs[term].ne(0).any()]
        design = legs[fitted].to_numpy(float)
        # Each term's mean square over the legs that carry it. Over every leg, a term few legs
        # carry (a league with few withheld cells) is held too loosely and overfits them.
        mean_square = (design**2).sum(axis=0) / (design != 0).sum(axis=0)
        # A leg at chance one half adds a quarter of its regressor's square to the Hessian: this
        # precision is what ``prior_legs`` such legs, each at the mean square, would add.
        precision = prior_legs * 0.25 * np.maximum(mean_square, _MIN_MEAN_SQUARE)
        hit = legs["hit"].to_numpy()
        beta, cov, chance = _newton(design, hit, legs["x0"].to_numpy(), precision)
        date_scores = (
            pd.DataFrame(design * (hit - chance)[:, None]).groupby(legs["Date"].to_numpy()).sum()
        ).to_numpy()
        n_dates = len(date_scores)
        clustered = cov @ (date_scores.T @ date_scores) @ cov * n_dates / max(n_dates - 1, 1)
        rows.append(
            {
                "league": league,
                "kind": kind,
                "book_class": book_class,
                "n_legs": len(legs),
                "n_dates": n_dates,
                "first_date": legs["Date"].min(),
                "last_date": legs["Date"].max(),
                "prior_legs": prior_legs,
                **dict(zip(fitted, beta, strict=True)),
                **dict(
                    zip((f"se_{term}" for term in fitted), np.sqrt(np.diag(clustered)), strict=True)
                ),
                "cov": cov.ravel().tolist(),
            }
        )
    return pd.DataFrame(rows, columns=FIT_COLS)


def score(offers: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Price each offer on the side ``Bet`` names, from a ``fit`` table.

    An offer whose league, kind and class has no row in ``table`` takes the no-evidence
    value, the app's fair price, with no interval. One with a side unposted has no fair
    price and gets NaN.

    Returns:
        On ``offers``' index: ``Fit Prob`` (the fitted chance the side wins), ``Fit Prob Low``
        (its one-sided ``LOWER_BOUND_Z`` lower bound) and ``EV`` (``Fit Prob`` times the
        side's payout).
    """
    terms = _terms(offers)
    eta = terms["x0"].to_numpy(copy=True)
    variance = np.zeros(len(terms))
    for fitted in table.itertuples(index=False):
        in_stratum = (
            terms["League"].eq(fitted.league)
            & terms["Kind"].eq(fitted.kind)
            & terms["Book Class"].eq(fitted.book_class)
        ).to_numpy()
        if not in_stratum.any():
            continue
        names = [term for term in TERMS if not np.isnan(getattr(fitted, term))]
        design = terms.loc[in_stratum, names].to_numpy(float)
        eta[in_stratum] += design @ np.array([getattr(fitted, term) for term in names])
        cov = np.reshape(fitted.cov, (len(names), len(names)))
        variance[in_stratum] = np.einsum("ij,jk,ik->i", design, cov, design)
    over = offers["Bet"].eq("Over").to_numpy()
    side_eta = np.where(over, eta, -eta)
    prob = expit(side_eta)
    return pd.DataFrame(
        {
            "Fit Prob": prob,
            "Fit Prob Low": expit(side_eta - LOWER_BOUND_Z * np.sqrt(variance)),
            "EV": prob * np.where(over, offers["Payout Over"], offers["Payout Under"]),
        },
        index=offers.index,
    )


def picks(offers: pd.DataFrame) -> np.ndarray:
    """Which offers a board calls its picks, by the ``EV`` it priced them at.

    A slate is one league's game date on one platform, with a place for each of its teams up
    to ``PICKS_PER_SLATE``. Per pick kind (:func:`pick_kind`) each player's highest-``EV``
    offer competes for the places, wherever on the slate it sits, and a tie goes to the
    earlier offer. An offer with no ``EV`` is never a pick.

    Args:
        offers: Offers as the module reads them, with ``Date``, ``Team``, ``Player`` and
            ``EV``.

    Returns:
        One flag per offer, in ``offers``' order.
    """
    # A combo's Team joins its players' teams with a slash.
    teams = offers.assign(Team=offers["Team"].str.split("/")).explode("Team")
    places = teams.groupby(_SLATE)["Team"].nunique().clip(upper=PICKS_PER_SLATE)
    best_of_player = (
        offers.assign(Kind=pick_kind(offers), Row=np.arange(len(offers)))
        .dropna(subset="EV")
        .sort_values("EV", ascending=False, kind="stable")
        .drop_duplicates([*_SLATE, "Kind", "Player"])
    )
    rank = best_of_player.groupby([*_SLATE, "Kind"]).cumcount()
    placed = rank < best_of_player.join(places.rename("Places"), on=_SLATE)["Places"]
    return np.isin(np.arange(len(offers)), best_of_player.loc[placed, "Row"])


def walk_forward(offers: pd.DataFrame) -> pd.DataFrame:
    """``score`` each game date with fits that read only the ``FIT_WINDOW_DAYS`` before it.

    ``offers`` is settled, as ``fit`` reads it. The first ``MIN_TRAIN_DATES`` game dates only
    train; every later date's offers come back with the ``score`` columns joined.
    """
    dates = pd.to_datetime(offers["Date"])
    window = pd.Timedelta(days=FIT_WINDOW_DAYS)
    scored = []
    for day in np.sort(dates.unique())[MIN_TRAIN_DATES:]:
        earlier = offers[(dates < day) & (dates >= day - window)]
        today = offers[dates == day]
        scored.append(today.join(score(today, fit(earlier))))
    return pd.concat(scored)


def fit_gain(offers: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """How much better than the app's fair price the fit calls the Over, per league and kind.

    ``offers`` is settled and priced: ``walk_forward`` output, or history rows a scoring run
    stamped with ``SCORE_COLS``. An offer the fit gave no price is left out.

    Returns:
        ``GAIN_COLS``: one row per league and fit kind, then every leg pooled as ``all``.
        ``gain`` is the mean over ``n_legs`` of the app's log loss less the fit's, in nats a
        leg and positive when the fit is the better price; ``gain_lo`` / ``gain_hi`` are 95 %
        bounds that resample the ``n_dates`` game dates whole.
    """
    # Imported here: the training package loads torch, which pricing an offer never needs.
    from sportstradamus.training.scorecard import _bootstrap_mean_ci_clustered

    terms = _terms(offers)
    over_hit = offers["Result"].eq("Over").to_numpy()
    fit_over = np.where(offers["Bet"].eq("Over"), offers["Fit Prob"], 1 - offers["Fit Prob"])
    app_over = expit(terms["x0"].to_numpy())
    # The app's log loss less the fit's: log of the fit's chance of what happened over the app's.
    gain = np.log(np.where(over_hit, fit_over / app_over, (1 - fit_over) / (1 - app_over)))
    legs = terms.assign(Date=offers["Date"].to_numpy(), gain=gain).dropna(subset="gain")
    pooled = pd.concat([legs, legs.assign(League="all", Kind="all")])
    rows = [
        (
            league,
            kind,
            len(part),
            part["Date"].nunique(),
            *_bootstrap_mean_ci_clustered(part["gain"].to_numpy(), part["Date"].to_numpy(), rng),
        )
        for (league, kind), part in pooled.groupby(["League", "Kind"])
    ]
    return pd.DataFrame(rows, columns=GAIN_COLS)


def ev_bands(offers: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """What the model's side paid in each band of ``EV``, beside what the fit said it would.

    ``offers`` as :func:`fit_gain` reads them.

    Returns:
        ``BAND_COLS``: one row per band of ``EV_BAND_EDGES``, each holding its lower edge.
        ``says`` is the mean ``EV`` of the band's ``n_legs``, and ``paid`` the mean paid back
        on a unit staked (the side's payout when it won, else 0), with 95 % bounds that
        resample whole game dates.
    """
    # Imported here: the training package loads torch, which pricing an offer never needs.
    from sportstradamus.training.scorecard import _bootstrap_mean_ci_clustered

    payout = np.where(offers["Bet"].eq("Over"), offers["Payout Over"], offers["Payout Under"])
    legs = offers[["Date", "EV"]].assign(paid=payout * offers["Bet"].eq(offers["Result"]))
    bands = pd.cut(legs["EV"], EV_BAND_EDGES, right=False)
    rows = [
        (
            str(band),
            len(part),
            part["EV"].mean(),
            *_bootstrap_mean_ci_clustered(part["paid"].to_numpy(), part["Date"].to_numpy(), rng),
        )
        for band, part in legs.groupby(bands, observed=False)
    ]
    return pd.DataFrame(rows, columns=BAND_COLS)
