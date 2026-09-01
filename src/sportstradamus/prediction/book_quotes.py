"""Per-player book quotes for serving: admission, pricing, and the book leg.

Both scoring paths read the same quotes. The model path takes their means as its book
leg (:func:`book_evs_for_players`) and pools them with the model; the book-fallback
path prices the offers off them outright. Admission is shared and deliberately strict —
a DFS platform reposting its own line is not independent support — so neither path can
serve a price the market never made.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sportstradamus.helpers import LazyArchive, book_weights, combo_props, get_odds
from sportstradamus.helpers.training_quotes import (
    COMBO_SUM_SOURCE,
    DFS_PLATFORM_BOOKS,
    TrainingQuote,
    quote_pricing_params,
    resolve_training_quote,
    with_component_sum_shape,
)

# LazyArchive defers DuckDB lock acquisition until the first attribute
# access. See LazyArchive docstring in helpers/archive.py.
archive = LazyArchive()

# A book quote serves only when its same-line cohort holds at least this many real
# sportsbooks. DFS platforms repost (or discount) consensus rather than price
# independently, so a cohort of platforms alone is the platform quoting itself — the
# Sleeper fake-50/50 poisoning vector.
_MIN_FALLBACK_REAL_BOOKS = 1

# Component sums whose components are not fit to quote, so the sum inherits their defect.
# NBA BLK and STL archive a probability that disagrees with their own stored ev at every
# line (+0.17 to +0.27, with 29%/43% of ev NULL); the sum graded claimed 0.141 against a
# realized 0.439. Repairing the components is what lifts this, not a kernel change.
COMBO_SERVE_BLOCKED = frozenset({("NBA", "BLST")})


def book_evs_for_players(
    offer_df: pd.DataFrame,
    league: str,
    market: str,
    dist: str,
    cv: float,
    hist_gate: float,
    date_map: dict,
    stat_data,
    players: pd.Index,
) -> tuple[list, list]:
    """Modal-cohort book means and spreads for the model path's book leg.

    Resolves the same admission-gated quotes as the fallback path and inverts each
    at its own cohort line, replacing the old cross-line average of stored per-row
    means (a DFS platform's boundary-line self-quote inverted under a too-narrow
    fitted shape once implied a 2.5-K mean for a 1.1-K batter). A player whose only
    support is the platform's own quote gets NaN: the blend then runs model-only
    and ``offer_records.finalize_records`` prices ``Market Prob`` payout-implied, applying the
    unquoted-disagreement drop.

    The spread is ``cv * mean`` for a single-market quote, which is all that family
    asserts, but a quote carrying a component sum's shape has its own — the kernel
    added the component variances and their correlation, so ``cv * mean`` would
    substitute the composite cell's generic dispersion for a computed quantity.
    """
    quotes = servable_fallback_quotes(offer_df, league, market, date_map, stat_data, dist, cv)
    gate = None if dist == "SkewNormal" else hist_gate or None
    means, sds = [], []
    for player in players:
        quote = quotes.get(player)
        if quote is None:
            means.append(np.nan)
            sds.append(np.nan)
            continue
        mean = quote_pricing_params(quote, league, market, dist, cv, gate=gate).mean
        means.append(mean)
        sds.append(quote.sum_sd if quote.sum_sd is not None else cv * mean)
    return means, sds


def _has_serving_support(quote: TrainingQuote) -> bool:
    """Whether a quote rests on evidence independent of the platform offering the leg.

    A direct cohort quote needs a real sportsbook in it; a component sum already
    demanded one behind every component to exist at all. Everything else — synthetic
    anchors, ev-inversions, DFS-only cohorts — is the platform quoting itself back.
    """
    if quote.source == COMBO_SUM_SOURCE:
        return True
    real_books = sum(book not in DFS_PLATFORM_BOOKS for book in quote.books)
    return quote.source == "book_direct" and real_books >= _MIN_FALLBACK_REAL_BOOKS


def servable_fallback_quotes(
    offer_df: pd.DataFrame,
    league: str,
    market: str,
    date_map: dict,
    stat_data,
    dist: str,
    cv: float,
) -> dict[str, TrainingQuote]:
    """Resolve per-player book quotes; keep only the ones with independent support.

    Mirrors the training-side consumer (``Stats.resolve_player_market_odds``): one
    modal-line cohort quote per player from :func:`resolve_training_quote`, then a
    second pass on the ``combo_props`` sums that prices the market as a weighted
    component sum through :meth:`Stats.combo_quote`. A quote serves when a real
    sportsbook (non-DFS) sits in its same-line cohort, or when it is a component sum —
    whose own admission already demands a sportsbook behind every component. Pure
    ev-inversions and synthetic quotes never serve: a DFS platform reposting (or
    discounting) a line is not independent support.

    The second pass covers every player, not just the unquoted ones, because the two
    quotes carry different things. Where the book priced the composite it keeps its
    own line, probability and mean, and takes only the sum's shape
    (:func:`with_component_sum_shape`) — a single quoted pair says nothing about any
    other offered line, and on this board a tenth of served legs are priced off the
    quote line. Where it did not, the sum is the whole quote.

    Only the simple ``combo_props`` sums take the second pass. Fantasy-score markets
    build a mean from up to 8 weighted components and stay no-served pending their own
    graded verdict; ``COMBO_SERVE_BLOCKED`` carries the sums whose components are
    themselves unfit to quote.
    """
    weights = book_weights.get(league, {}).get(market, {})
    first_lines = offer_df["Line"].groupby(level=0).first()
    players_by_date: dict[str, list[str]] = {}
    for player in offer_df.index.unique():
        players_by_date.setdefault(date_map.get(player, ""), []).append(player)
    combo_servable = market in combo_props and (league, market) not in COMBO_SERVE_BLOCKED

    quotes: dict[str, TrainingQuote] = {}
    for date, players in players_by_date.items():
        quote_inputs = archive.get_training_quote_inputs(league, market, date, players)
        for player in players:
            rows, legacy_line = quote_inputs.get(player, ([], None))
            quotes[player] = resolve_training_quote(
                rows,
                fallback_ev=None,
                legacy_line=legacy_line,
                # Serving has no Avg10 to mirror base.py's fallback anchor; the
                # player's own offered line anchors the inversion instead.
                fallback_line=max(float(first_lines[player]), 0.5),
                dist=dist,
                cv=cv,
                weights=weights,
            )
        if not combo_servable:
            continue
        # Anchored on the book's own line where it quoted one, so the sum's shape is
        # fitted around the real price rather than replacing it; on the offered line
        # otherwise, where there is no price to preserve.
        lines = {
            player: quotes[player].line
            if quotes[player].source == "book_direct"
            else float(first_lines[player])
            for player in players
        }
        for player, combo in stat_data.combo_quote(
            market, players, date, None, lines=lines
        ).items():
            # Only a direct cohort quote is worth anchoring to. Every other rung's
            # probability is an inversion of a stored ``ev``, which is that quote's
            # gated derivative rather than a native price — untrustworthy enough that
            # `_has_serving_support` refuses to serve it, so anchoring the sum on it
            # would be worse than the sum alone.
            book = quotes[player]
            quotes[player] = (
                with_component_sum_shape(book, combo) if book.source == "book_direct" else combo
            )

    return {player: quote for player, quote in quotes.items() if _has_serving_support(quote)}


def price_offers_at_quotes(
    offer_df: pd.DataFrame,
    quotes: dict[str, TrainingQuote],
    league: str,
    market: str,
    dist: str,
    cv: float,
    step: float,
) -> None:
    """Set ``Market Projection`` and the ``Market EV`` over-probability in place.

    Every offered line for a player is decoded from the one quote-implied mean with
    the same fitted shape (``sigma``/``skew`` or ``phi``) used to invert it, so an
    offer at the quote line reproduces the cohort probability exactly and alternate
    lines price off the same distribution rather than a cross-line average.

    A quote carrying a component sum's shape instead prices every line off that CDF,
    which is the whole reason the kernel exists: the composite cell's marginal family
    has one generic ``cv`` and no component dispersions, and grading the two across
    ±1.5 sd offset lines put the sum ahead on 9 of 10 markets. Its own quoted line is
    unaffected either way — the shape was anchored there.
    """
    combo_cdfs = {p: q.under_prob_at for p, q in quotes.items() if q.under_prob_at is not None}
    priced = {p: quote_pricing_params(q, league, market, dist, cv) for p, q in quotes.items()}
    offer_df["Market Projection"] = offer_df.index.map({p: v.mean for p, v in priced.items()})
    shape_kwargs = {}
    if dist == "SkewNormal":
        shape_kwargs = {
            "sigma": offer_df.index.map({p: v.sigma for p, v in priced.items()}).to_numpy(float),
            "skew_alpha": offer_df.index.map({p: v.skew for p, v in priced.items()}).to_numpy(
                float
            ),
        }
    elif dist == "DPO" and any(v.phi is not None for v in priced.values()):
        shape_kwargs = {
            "phi": offer_df.index.map({p: v.phi for p, v in priced.items()}).to_numpy(float)
        }
    elif dist in ("NegBin", "ZINB") and any(v.r is not None for v in priced.values()):
        shape_kwargs = {
            "r": offer_df.index.map({p: v.r for p, v in priced.items()}).to_numpy(float)
        }
    lines = offer_df["Line"].to_numpy(dtype=float)
    over = 1 - get_odds(
        lines,
        offer_df["Market Projection"].to_numpy(dtype=float),
        dist,
        cv=cv,
        step=step,
        **shape_kwargs,
    )
    if combo_cdfs:
        over = [
            1.0 - combo_cdfs[player](line) if player in combo_cdfs else marginal
            for player, line, marginal in zip(offer_df.index, lines, over, strict=True)
        ]
    offer_df["Market EV"] = over
