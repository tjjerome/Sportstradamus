"""Pure training-quote resolution with explicit provenance.

The archive owns I/O; this module owns the deterministic, outcome-independent
policy that turns its latest-per-book rows into one coherent training quote.
"""

from __future__ import annotations

import dataclasses
import datetime
import operator
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple, TypeVar

import numpy as np

from sportstradamus.helpers.config import book_count_dispersion, book_skewnormal_shape
from sportstradamus.helpers.distributions import get_ev, get_odds

CONSENSUS_LINE_POLICY = "modal-nearest-median-v1"
AUTHENTIC = "authentic"
DERIVED = "derived"
SYNTHETIC = "synthetic"
# Component-sum kernel quotes carry this source with ``authenticity=DERIVED``:
# the probability is derived from real component-book quotes, not fabricated.
COMBO_SUM_SOURCE = "combo_sum"
BLOWN_ABS_CEILING = 2000.0
BLOWN_LINE_FACTOR = 5.0

PROVENANCE_COLUMNS = (
    "QuoteSource",
    "QuoteAuthenticity",
    "QuoteSyntheticReason",
    "QuoteObservedAt",
    "QuoteBookCount",
)

# A fantasy-points market is one DFS platform's scoring formula, so the market
# name names the platform that posts it; no sportsbook offers these.
_PICKEM_PLATFORMS = {
    "underdog": "Underdog",
    "prizepicks": "PrizePicks",
    "parlayplay": "ParlayPlay",
    "parlay": "ParlayPlay",
    "sleeper": "Sleeper",
    "thrive": "Thrive",
}
# Book names under which the DFS pick'em platforms write archive odds rows.
# Their prices repost (and sometimes move) sportsbook consensus, so consensus
# readers cap their weight whenever a real sportsbook also quotes the entry.
DFS_PLATFORM_BOOKS: frozenset[str] = frozenset(_PICKEM_PLATFORMS.values())
# Pick'em pays the same either way at the posted line — the operator's rake sits
# in the payout multiplier, not the line — so an unboosted entry prices at 50/50.
_PICKEM_UNDER_PROBABILITY = 0.5


@dataclasses.dataclass(frozen=True)
class ArchivedBookQuote:
    """Latest archive row for one book at the requested observation cutoff."""

    book: str
    ev: float | None
    under_probability: float | None
    line: float | None
    observed_at: datetime.datetime | None


@dataclasses.dataclass(frozen=True)
class TrainingQuote:
    """One coherent line/probability/EV tuple plus its explicit provenance."""

    line: float
    over_probability: float
    ev: float
    source: str
    authenticity: str
    synthetic_reason: str | None
    observed_at: datetime.datetime | None
    book_count: int
    books: tuple[str, ...] = ()
    # Combo-sum extras (``COMBO_SUM_SOURCE`` quotes): the component-sum sd and a
    # per-line under-probability evaluator. Both are excluded from ``as_record()``
    # and every serialization — records stay byte-identical to legacy quotes.
    sum_sd: float | None = None
    under_prob_at: Callable[[float], float] | None = None

    @property
    def archived(self) -> bool:
        """Legacy compatibility projection: only direct book probability is authentic."""
        return self.authenticity == AUTHENTIC

    @property
    def odds_synthetic(self) -> bool:
        """Every non-authentic probability is derived or synthetic."""
        return not self.archived

    def as_record(self) -> dict[str, object]:
        """Return the canonical matrix book/provenance columns."""
        return {
            "Line": self.line,
            "Odds": self.over_probability,
            "EV": self.ev,
            "Archived": self.archived,
            "Odds_synthetic": self.odds_synthetic,
            "QuoteSource": self.source,
            "QuoteAuthenticity": self.authenticity,
            "QuoteSyntheticReason": self.synthetic_reason,
            "QuoteObservedAt": self.observed_at,
            "QuoteBookCount": self.book_count,
        }


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _positive(value) -> bool:
    return _finite(value) and float(value) > 0.0


def _probability(value) -> bool:
    return _finite(value) and 0.0 < float(value) < 1.0


def archive_ev_is_runaway(value, line) -> bool:
    """Return whether a finite EV crosses the historical runaway thresholds."""
    if not _finite(value):
        return False
    ev = float(value)
    if ev > BLOWN_ABS_CEILING:
        return True
    if not _positive(line):
        return False
    line_ceiling = BLOWN_LINE_FACTOR * float(line)
    # ``get_ev`` may converge one ULP above its intentional 5x line ceiling.
    # Equality is valid under the historical ``> 5x`` contract, so do not let
    # floating-point noise make a derived coherent quote fail the same guard.
    return bool(ev > line_ceiling and not np.isclose(ev, line_ceiling, rtol=1e-12, atol=0.0))


def archive_ev_is_usable(value, line) -> bool:
    """Reject nonpositive/nonfinite and historically runaway archive EV values."""
    return _positive(value) and not archive_ev_is_runaway(value, line)


def _weight(book: str, weights: Mapping[str, float]) -> float:
    value = weights.get(book, 1.0)
    return float(value) if _positive(value) else 1.0


_R = TypeVar("_R")
_quote_book: Callable[[ArchivedBookQuote], str] = operator.attrgetter("book")


def sportsbook_cohort(
    rows: Sequence[_R], book_of: Callable[[_R], str] = _quote_book
) -> Sequence[_R]:
    """The sportsbooks in a cohort, or the whole cohort when none of them quoted it.

    A pick'em platform posts a line its product pays evenly at, so its implied
    probability is anchored near 0.5 however far the truth sits from it. On MLB
    stolen bases the platforms quote 0.50 under where the sportsbooks on the same
    cohort quote 0.86 and the market settles under 0.90; the same holds for home
    runs and doubles, every market whose 0.5-line granularity forces the platform
    off the median. Graded over the archive's mixed cohorts, dropping them beat
    keeping them on 14 of 16 MLB markets and cost 0.0001 Brier on the worst.

    With no sportsbook present the platform's line is the only evidence there is,
    so it stays — the pick'em boards carry entries no book prices at all.
    """
    sharp = [row for row in rows if book_of(row) not in DFS_PLATFORM_BOOKS]
    return sharp or rows


def _weighted_value(
    rows: Sequence[ArchivedBookQuote],
    attribute: str,
    weights: Mapping[str, float],
) -> float | None:
    values, row_weights = [], []
    for row in rows:
        value = getattr(row, attribute)
        if not _finite(value):
            continue
        values.append(float(value))
        row_weights.append(_weight(row.book, weights))
    if not values:
        return None
    return float(np.average(values, weights=row_weights))


def _latest_observation(rows: Sequence[ArchivedBookQuote]) -> datetime.datetime | None:
    observed = [row.observed_at for row in rows if row.observed_at is not None]
    return max(observed) if observed else None


def pickem_quote(
    market: str, line: float, observed_at: datetime.datetime | None
) -> list[ArchivedBookQuote]:
    """The platform's own symmetric quote for a pick'em line no book ever priced.

    DFS priced rows only begin in 2026; before that a pick'em entry archived its
    line and nothing else. That bare line is still the platform's quote, so stand
    the symmetric price back up rather than read it as an absent book. Boosted
    entries do not price at 50/50 — ``Archive.add_dfs`` runs a live boost through
    ``dfs_boost_probs`` into a real skewed quote — so callers must reach here only
    when nothing priced the entry, letting any real quote outrank this.
    """
    if "fantasy points " not in market:
        return []
    platform = _PICKEM_PLATFORMS.get(market.rsplit(" ", 1)[-1])
    if platform is None or not _positive(line) or observed_at is None:
        return []
    return [ArchivedBookQuote(platform, None, _PICKEM_UNDER_PROBABILITY, line, observed_at)]


def _direct_line_cohort(
    rows: Sequence[ArchivedBookQuote],
) -> tuple[float, list[ArchivedBookQuote]] | None:
    direct = [row for row in rows if _positive(row.line) and _probability(row.under_probability)]
    if not direct:
        return None
    lines = [float(row.line) for row in direct]
    counts = Counter(lines)
    median = float(np.median(lines))
    # Most books first; then the observed line nearest the median; lower line
    # wins the final tie. Outcomes never enter this policy.
    selected = min(counts, key=lambda line: (-counts[line], abs(line - median), line))
    return selected, [row for row in direct if float(row.line) == selected]


def _authentic_quote(
    cohort: Sequence[ArchivedBookQuote],
    line: float,
    *,
    dist: str,
    cv: float,
    weights: Mapping[str, float],
) -> TrainingQuote:
    """Build the quote for a same-line cohort that priced the market directly.

    The mean is inverted from the cohort's consensus under-probability under the cell's
    CURRENT config, not read from the archive's stored ``ev``. A stored ``ev`` was encoded
    at scrape time and goes stale the next time ``cv`` is recomputed (every ``meditate``
    run) or the distribution family flips — 71% of a live sample of ungated quotes had
    drifted, by 26% of the mean.

    Inverting the consensus rather than averaging per-book inversions is what keeps the
    three book columns describing *one* distribution: ``get_odds(line, ev, cv)`` returns
    exactly ``1 - over_probability``, so the probability Gate 1 scores and the mean the
    ``fused_loc`` book leg pools are the same quote. ``get_ev`` is convex, so the two
    orders differ by far more than rounding — on MLB total bases at a 0.5 line, 0.64
    against 1.23 — and averaging means yields a number no single distribution produces.

    The inversion stays ungated even for ``ZINB``/``ZAGamma`` cells, matching the fallback
    this replaces: ``book_gate``'s population zero rate (0.78 for NFL tds, 0.89 for MLB
    home runs) exceeds the under-probabilities books actually quote — they only price
    players likely to score — so gating here would clamp every quote to ``get_ev``'s
    ceiling. Neither inversion reproduces such a book, so the book leg of ``fused_loc`` is
    unsound for gated cells either way, a distribution-family question rather than a clamp
    choice.

    The cohort narrows to its sportsbooks first (:func:`sportsbook_cohort`), so the
    quote's ``books``/``book_count`` name exactly the rows that set its price.
    """
    cohort = sportsbook_cohort(cohort)
    under = _weighted_value(cohort, "under_probability", weights)
    if under is None:  # Defensive; the cohort predicate guarantees this.
        raise ValueError("direct quote cohort has no probability")
    return TrainingQuote(
        line=line,
        over_probability=float(np.clip(1.0 - under, 0.0, 1.0)),
        ev=float(get_ev(line, under, cv=cv, dist=dist)),
        source="book_direct",
        authenticity=AUTHENTIC,
        synthetic_reason=None,
        observed_at=_latest_observation(cohort),
        book_count=len(cohort),
        books=tuple(row.book for row in cohort),
    )


def _ev_inversion_quote(
    line: float,
    ev: float | None,
    *,
    dist: str,
    cv: float,
    source: str,
    observed_at: datetime.datetime | None,
    book_count: int,
) -> TrainingQuote | None:
    """Stand a probability back up from an ev-only quote, or ``None`` if it will not invert.

    The fallback for rows predating the shape-free ``(line, under_prob)`` columns, where the
    stored ``ev`` is the only surviving evidence and so cannot be re-derived the way
    :func:`_authentic_quote` does. ``None`` when there is no ev or the inversion lands
    outside ``(0, 1)``, letting the caller try the next rung of the ladder.
    """
    if ev is None or ev <= 0:
        return None
    under = float(get_odds(line, ev, dist, cv=cv))
    if not _probability(under):
        return None
    return TrainingQuote(
        line=line,
        over_probability=float(1.0 - under),
        ev=ev,
        source=source,
        authenticity=DERIVED,
        synthetic_reason="ev_inversion",
        observed_at=observed_at,
        book_count=book_count,
    )


def resolve_training_quote(
    rows: Sequence[ArchivedBookQuote],
    *,
    legacy_line: float | None,
    fallback_line: float,
    fallback_ev: float | None,
    dist: str,
    cv: float,
    weights: Mapping[str, float] | None = None,
    fallback_quote: TrainingQuote | None = None,
) -> TrainingQuote:
    """Resolve one training quote without consulting outcomes or mutable state.

    Direct shape-free probabilities are authentic only when they belong to the
    selected same-line book cohort. Legacy EV inversion and neutral/model
    fallbacks remain explicitly derived or synthetic. A caller-built
    ``fallback_quote`` slots between book EV inversion and the legacy
    ``fallback_ev`` combo inversion: it prices a real component-sum tail, so it
    outranks a scalar EV inverted under the cell's fabricated generic-cv tail.
    """
    # style: allow-complexity — the resolution ladder itself: one early return per
    # rung in provenance order; extracting rungs would scatter one policy.
    weights = weights or {}
    ordered_rows = sorted(rows, key=lambda row: row.book)
    direct = _direct_line_cohort(ordered_rows)
    if direct is not None:
        return _authentic_quote(direct[1], direct[0], dist=dist, cv=cv, weights=weights)

    line_is_observed = _positive(legacy_line)
    line = (
        float(legacy_line)
        if line_is_observed
        else float(fallback_line)
        if _positive(fallback_line)
        else 0.5
    )
    ev_rows = sportsbook_cohort([row for row in ordered_rows if archive_ev_is_usable(row.ev, line)])
    cohort_ev = _weighted_value(ev_rows, "ev", weights)
    derived = _ev_inversion_quote(
        line,
        cohort_ev,
        dist=dist,
        cv=cv,
        source="book_ev_inversion",
        observed_at=_latest_observation(ev_rows),
        book_count=len(ev_rows),
    )
    if derived is not None:
        return derived

    if fallback_quote is not None:
        return fallback_quote

    combo_ev = float(fallback_ev) if archive_ev_is_usable(fallback_ev, line) else None
    derived = _ev_inversion_quote(
        line,
        combo_ev,
        dist=dist,
        cv=cv,
        source="combo_ev_inversion",
        observed_at=None,
        book_count=0,
    )
    if derived is not None:
        return derived

    ev = float(get_ev(line, 0.5, cv=cv, dist=dist))
    return TrainingQuote(
        line=line,
        over_probability=0.5,
        ev=ev,
        source="neutral_fallback" if line_is_observed else "model_fallback",
        authenticity=SYNTHETIC,
        synthetic_reason="no_usable_book_probability",
        observed_at=None,
        book_count=0,
    )


class QuotePricingParams(NamedTuple):
    """The inverted quote plus whichever fitted shape the family carries.

    ``sigma``/``skew`` populate for SkewNormal, ``phi`` for Double Poisson and ``r``
    for NegBin/ZINB; every other family leaves all four ``None`` and prices off
    ``cv`` alone.
    """

    mean: float
    sigma: float | None = None
    skew: float | None = None
    phi: float | None = None
    r: float | None = None


def quote_pricing_params(
    quote: TrainingQuote, league: str, market: str, dist: str, cv: float, gate: float | None = None
) -> QuotePricingParams:
    """Invert the quote at its own line under the exact shape the decode will use.

    Returns ``(mean, sigma, skew, phi, r)``. Fixing the shape across the invert/decode pair
    makes ``decode(invert(p)) == p`` at the quote line — the shape-consistent round trip
    that kills the symmetric-encode/skewed-decode asymmetry (an honest 50/50 no longer
    decodes to 0.57 under). ``(sigma, skew)`` carry it for SkewNormal cells, ``phi``
    for Double Poisson ones and ``r`` for NegBin/ZINB; the remaining families have no
    fitted shape and price off ``cv`` alone. On the fallback path the pair is ungated on both sides: books only price
    players likely to record the stat, so the population zero rate can exceed the quoted
    under-prob and a gated inversion would clamp at ``get_ev``'s ceiling (see
    ``_authentic_quote``). The model path passes the cell gate on count families instead —
    its decode is gated and its blend expects the non-zero component mean.

    The two shapes are evaluated at different points on purpose. ``sigma`` reads at
    ``quote.ev``, the cv-inverted mean, because a SkewNormal cell's line tracks its mean
    closely. The count dispersion reads at ``quote.line``: a count cell quotes 0.5 against
    means from 0.09 to 2.5, and the cv-inverted mean is exactly the quantity the fitted
    dispersion exists to correct, so seeding the correction with it would feed the bias
    back in.
    """
    under = 1.0 - quote.over_probability
    sigma = skew = phi = r = None
    if dist == "SkewNormal":
        sigma, skew = book_skewnormal_shape(league, market, quote.ev, cv)
        sigma, skew = float(sigma), float(skew)
    else:
        phi, r = book_count_dispersion(league, market, quote.line, cv)
        phi = None if phi is None else float(phi)
        r = None if r is None else float(r)
    mean = float(
        get_ev(
            quote.line, under, cv, dist=dist, sigma=sigma, skew_alpha=skew, gate=gate, phi=phi, r=r
        )
    )
    return QuotePricingParams(mean, sigma, skew, phi, r)
