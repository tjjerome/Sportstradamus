"""Pure training-quote resolution with explicit provenance.

The archive owns I/O; this module owns the deterministic, outcome-independent
policy that turns its latest-per-book rows into one coherent training quote.
"""

from __future__ import annotations

import dataclasses
import datetime
from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np

from sportstradamus.helpers.distributions import get_ev, get_odds

CONSENSUS_LINE_POLICY = "modal-nearest-median-v1"
AUTHENTIC = "authentic"
DERIVED = "derived"
SYNTHETIC = "synthetic"
BLOWN_ABS_CEILING = 2000.0
BLOWN_LINE_FACTOR = 5.0

PROVENANCE_COLUMNS = (
    "QuoteSource",
    "QuoteAuthenticity",
    "QuoteSyntheticReason",
    "QuoteObservedAt",
    "QuoteBookCount",
)


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


def _direct_line_cohort(
    rows: Sequence[ArchivedBookQuote],
) -> tuple[float, list[ArchivedBookQuote]] | None:
    direct = [
        row
        for row in rows
        if _positive(row.line) and _probability(row.under_probability)
    ]
    if not direct:
        return None
    lines = [float(row.line) for row in direct]
    counts = Counter(lines)
    median = float(np.median(lines))
    # Most books first; then the observed line nearest the median; lower line
    # wins the final tie. Outcomes never enter this policy.
    selected = min(counts, key=lambda line: (-counts[line], abs(line - median), line))
    return selected, [row for row in direct if float(row.line) == selected]


def resolve_training_quote(
    rows: Sequence[ArchivedBookQuote],
    *,
    legacy_line: float | None,
    fallback_line: float,
    fallback_ev: float | None,
    dist: str,
    cv: float,
    weights: Mapping[str, float] | None = None,
) -> TrainingQuote:
    """Resolve one training quote without consulting outcomes or mutable state.

    Direct shape-free probabilities are authentic only when they belong to the
    selected same-line book cohort. Legacy EV inversion and neutral/model
    fallbacks remain explicitly derived or synthetic.
    """
    weights = weights or {}
    ordered_rows = sorted(rows, key=lambda row: row.book)
    direct = _direct_line_cohort(ordered_rows)
    if direct is not None:
        line, cohort = direct
        under = _weighted_value(cohort, "under_probability", weights)
        if under is None:  # Defensive; the cohort predicate guarantees this.
            raise ValueError("direct quote cohort has no probability")
        ev = _weighted_value(
            [row for row in cohort if archive_ev_is_usable(row.ev, line)],
            "ev",
            weights,
        )
        if ev is None or ev <= 0:
            ev = float(get_ev(line, under, cv=cv, dist=dist))
        return TrainingQuote(
            line=line,
            over_probability=float(np.clip(1.0 - under, 0.0, 1.0)),
            ev=ev,
            source="book_direct",
            authenticity=AUTHENTIC,
            synthetic_reason=None,
            observed_at=_latest_observation(cohort),
            book_count=len(cohort),
        )

    line_is_observed = _positive(legacy_line)
    line = (
        float(legacy_line)
        if line_is_observed
        else float(fallback_line)
        if _positive(fallback_line)
        else 0.5
    )
    ev_rows = [row for row in ordered_rows if archive_ev_is_usable(row.ev, line)]
    ev = _weighted_value(ev_rows, "ev", weights)
    if ev is not None and ev > 0:
        under = float(get_odds(line, ev, dist, cv=cv))
        if _probability(under):
            return TrainingQuote(
                line=line,
                over_probability=float(1.0 - under),
                ev=ev,
                source="book_ev_inversion",
                authenticity=DERIVED,
                synthetic_reason="ev_inversion",
                observed_at=_latest_observation(ev_rows),
                book_count=len(ev_rows),
            )

    if _positive(fallback_ev):
        ev = float(fallback_ev)
        under = float(get_odds(line, ev, dist, cv=cv))
        if _probability(under):
            return TrainingQuote(
                line=line,
                over_probability=float(1.0 - under),
                ev=ev,
                source="combo_ev_inversion",
                authenticity=DERIVED,
                synthetic_reason="ev_inversion",
                observed_at=None,
                book_count=0,
            )

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
