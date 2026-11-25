"""EV's win chance: the app's price, the books' and the model's, fitted on settled posted picks.

One penalized logistic fit per league, pick kind and class of sportsbook coverage gives the
chance an offer's Over hits (docs/handoffs/profit-engine.md 5.1; the estimator and its
evidence are docs/archive/researcher_ev_fit.md). ``book_inputs`` reads what the sportsbooks
said at an offer's line and names its class, ``fit`` estimates the table, ``score`` prices
offers with it on the side the model took, and ``walk_forward`` prices each game date from
earlier dates only. Pure.

Every function but ``book_inputs`` reads offers under history's column names: ``League``,
``Platform``, ``Payout Over`` and ``Payout Under`` (full decimal payouts, 0 on a side the app
did not post), ``Bet``, ``Win Prob`` and ``Book Prob`` (both of the side ``Bet`` names),
``Model Version`` and ``Book Class``.
"""

from collections.abc import Sequence

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
# A sportsbook quote older than this at scoring counts for nothing: the weights are earned on
# fresh quotes.
BOOK_MAX_AGE = pd.Timedelta(minutes=90)
# Earlier game dates a walk-forward needs before it scores one.
MIN_TRAIN_DATES = 14

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

_STRATUM = ["League", "Kind", "Book Class"]
# Probabilities are clipped this far inside (0, 1) before the logit.
_PROB_EPS = 1e-3
# Floor on a regressor's mean square, so a near-constant column keeps a finite prior.
_MIN_MEAN_SQUARE = 1e-6
_NEWTON_MAX_ITER = 60
_NEWTON_TOL = 1e-8


def book_inputs(
    rungs: pd.DataFrame, players: Sequence[str], lines: np.ndarray, scored_at: pd.Timestamp
) -> pd.DataFrame:
    """What the sportsbooks said about each offer, as its scoring run saw them.

    ``rungs`` is ``Archive.get_ladder_rungs`` output for the offers' game date, sportsbooks
    only. A quote older than ``BOOK_MAX_AGE`` at ``scored_at`` counts for nothing.

    Returns:
        Positional over ``players`` and ``lines``: ``Book Over`` (the plain mean of the books'
        over-probabilities at the offer's own line, NaN when none prices it), ``Book Count``,
        ``Book Observed At`` (the newest of those quotes) and ``Book Class``: ``A`` when a
        book prices the line, ``B`` when the books price only other lines, ``C`` when none
        quotes the player.
    """
    offers = pd.DataFrame({"entity": players, "offered": lines})
    fresh = rungs[scored_at - rungs["observed_at"] <= BOOK_MAX_AGE]
    quoted = offers.reset_index().merge(fresh, on="entity")
    at_line = (
        quoted[np.isclose(quoted["line"], quoted["offered"])]
        .groupby("index")
        .agg(over=("p_over", "mean"), count=("book", "size"), newest=("observed_at", "max"))
        .reindex(offers.index)
    )
    count = at_line["count"].fillna(0).astype(int)
    return pd.DataFrame(
        {
            "Book Over": at_line["over"],
            "Book Count": count,
            "Book Observed At": pd.to_datetime(at_line["newest"]),
            "Book Class": np.select(
                [count > 0, offers.index.isin(quoted["index"])], ["A", "B"], "C"
            ),
        }
    )


def _logit(prob) -> np.ndarray:
    return logit(np.clip(np.asarray(prob, dtype=float), _PROB_EPS, 1 - _PROB_EPS))


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
    platform = offers["Platform"].to_numpy(str)
    return pd.DataFrame(
        {
            "League": offers["League"].to_numpy(),
            # Even when both sides pay the same: the fair price is one half on every such offer.
            "Kind": np.select(
                [platform != "Underdog", np.isclose(payout_over, payout_under)],
                [np.char.lower(platform), "underdog_even"],
                "underdog_modified",
            ),
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
        legs = (
            terms["League"].eq(fitted.league)
            & terms["Kind"].eq(fitted.kind)
            & terms["Book Class"].eq(fitted.book_class)
        ).to_numpy()
        if not legs.any():
            continue
        names = [term for term in TERMS if not np.isnan(getattr(fitted, term))]
        design = terms.loc[legs, names].to_numpy(float)
        eta[legs] += design @ np.array([getattr(fitted, term) for term in names])
        cov = np.reshape(fitted.cov, (len(names), len(names)))
        variance[legs] = np.einsum("ij,jk,ik->i", design, cov, design)
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
