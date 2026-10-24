"""A cohort only pick'em platforms priced resolves as a real line, not book evidence."""

import datetime

from sportstradamus.helpers.training_quotes import (
    AUTHENTIC,
    PICKEM,
    ArchivedBookQuote,
    pickem_quote,
    resolve_training_quote,
)

_SEEN = datetime.datetime(2026, 9, 13, 1)


def _resolve(rows, line=4.5):
    return resolve_training_quote(
        rows, legacy_line=line, fallback_line=line, fallback_ev=None, dist="SkewNormal", cv=0.4
    )


def test_pickem_only_cohort_is_a_real_line_but_not_book_evidence():
    quote = _resolve(
        [
            ArchivedBookQuote("Underdog", None, 0.5, 4.5, _SEEN),
            ArchivedBookQuote("Sleeper", None, 0.5, 4.5, _SEEN),
        ]
    )
    assert quote.authenticity == PICKEM
    assert quote.source == "book_direct"
    assert quote.archived and not quote.odds_synthetic
    assert quote.as_record()["QuoteAuthenticity"] == "pickem"


def test_sportsbook_in_cohort_keeps_the_quote_authentic():
    quote = _resolve(
        [
            ArchivedBookQuote("draftkings", None, 0.55, 4.5, _SEEN),
            ArchivedBookQuote("Underdog", None, 0.5, 4.5, _SEEN),
        ]
    )
    assert quote.authenticity == AUTHENTIC
    assert quote.books == ("draftkings",)


def test_pickem_stand_in_resolves_as_pickem():
    quote = _resolve(pickem_quote("fantasy points underdog", 12.5, _SEEN), line=12.5)
    assert quote.authenticity == PICKEM
