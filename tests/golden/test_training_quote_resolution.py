"""Stage 2 canonical training quote and provenance contracts."""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from sportstradamus.helpers import config
from sportstradamus.helpers.training_quotes import (
    AUTHENTIC,
    DERIVED,
    SYNTHETIC,
    ArchivedBookQuote,
    archive_ev_is_runaway,
    resolve_training_quote,
)
from sportstradamus.scripts.inject_backfilled_odds import resolve_cached_quotes
from sportstradamus.stats import base as base_mod
from sportstradamus.training.data import _clip_lines
from sportstradamus.training.group_conditional_cdf._pipeline_steps_two_part import (
    _explicit_quote_authenticity_mask,
)
from sportstradamus.training.matrix_audit import audit_matrix
from sportstradamus.training.pipeline import (
    _PRETRIM_LINE_COLUMN,
    _reconcile_clipped_neutral_quotes,
    _step_synthesize_odds,
)

_AT_1 = datetime.datetime(2026, 5, 8, 10)
_AT_2 = datetime.datetime(2026, 5, 8, 11)


def _row(book, ev, under, line, observed_at=_AT_1):
    return ArchivedBookQuote(book, ev, under, line, observed_at)


def test_direct_probability_uses_one_deterministic_same_line_cohort():
    quote = resolve_training_quote(
        [
            _row("z", 3.1, 0.60, 2.5, _AT_2),
            _row("a", 2.9, 0.50, 2.5, _AT_1),
            _row("m", 4.0, 0.80, 3.5, _AT_2),
        ],
        legacy_line=9.5,
        fallback_line=1.5,
        fallback_ev=None,
        dist="SkewNormal",
        cv=0.5,
    )

    assert quote.line == 2.5
    assert quote.over_probability == pytest.approx(0.45)
    # Each cohort quote re-inverted at the cell's current cv/dist, then averaged — not the
    # stored 3.1/2.9, which were encoded under whatever config was live when they scraped.
    assert quote.ev == pytest.approx(2.3460, abs=1e-4)
    assert quote.authenticity == AUTHENTIC
    assert quote.source == "book_direct"
    assert quote.synthetic_reason is None
    assert quote.observed_at == _AT_2
    assert quote.book_count == 2
    assert quote.archived and not quote.odds_synthetic


def test_legacy_ev_inversion_is_explicitly_derived_and_synthetic():
    quote = resolve_training_quote(
        [_row("b", 3.0, None, None)],
        legacy_line=2.5,
        fallback_line=1.5,
        fallback_ev=None,
        dist="SkewNormal",
        cv=0.5,
    )

    assert quote.authenticity == DERIVED
    assert quote.source == "book_ev_inversion"
    assert quote.synthetic_reason == "ev_inversion"
    assert quote.book_count == 1
    assert not quote.archived and quote.odds_synthetic


@pytest.mark.parametrize("stored_ev", [2.4, None, 50_000.0])
def test_direct_quote_ev_is_derived_from_the_quote_not_the_stored_ev(stored_ev):
    """A shape-free cohort resolves identically whatever ``ev`` sits beside it.

    The stored ``ev`` is a scrape-time encoding, so a stale one (a later ``meditate``
    moved ``cv``), an absent one (the writer refused a clamped inversion), and a runaway
    one must all yield the same mean: the one today's config implies.
    """
    quote = resolve_training_quote(
        [_row("b", stored_ev, 0.55, 2.5)],
        legacy_line=2.5,
        fallback_line=2.0,
        fallback_ev=None,
        dist="SkewNormal",
        cv=0.5,
    )

    assert quote.authenticity == AUTHENTIC
    assert quote.source == "book_direct"
    assert quote.ev == pytest.approx(2.3390, abs=1e-4)


def test_runaway_legacy_ev_falls_through_to_honest_synthetic_quote():
    quote = resolve_training_quote(
        [_row("b", 50_000.0, None, None)],
        legacy_line=2.5,
        fallback_line=2.0,
        fallback_ev=None,
        dist="SkewNormal",
        cv=0.5,
    )

    assert quote.authenticity == SYNTHETIC
    assert quote.source == "neutral_fallback"
    assert quote.over_probability == 0.5
    assert quote.book_count == 0


def test_runaway_combo_fallback_ev_falls_through_to_honest_synthetic_quote():
    quote = resolve_training_quote(
        [],
        legacy_line=0.5,
        fallback_line=0.5,
        fallback_ev=3.0,
        dist="SkewNormal",
        cv=0.5,
    )

    assert quote.authenticity == SYNTHETIC
    assert quote.source == "neutral_fallback"
    assert quote.over_probability == 0.5
    assert quote.book_count == 0
    assert not archive_ev_is_runaway(quote.ev, quote.line)


def test_runaway_threshold_allows_one_ulp_above_exact_five_x_boundary():
    ceiling = 5.0 * 0.5

    assert not archive_ev_is_runaway(np.nextafter(ceiling, np.inf), 0.5)
    assert archive_ev_is_runaway(ceiling * (1.0 + 1e-9), 0.5)


def test_no_book_quote_is_model_fallback_with_honest_null_observation():
    quote = resolve_training_quote(
        [],
        legacy_line=0,
        fallback_line=2.0,
        fallback_ev=np.nan,
        dist="SkewNormal",
        cv=0.5,
    )

    assert quote.line == 2.0
    assert quote.over_probability == 0.5
    assert quote.authenticity == SYNTHETIC
    assert quote.source == "model_fallback"
    assert quote.synthetic_reason == "no_usable_book_probability"
    assert quote.observed_at is None
    assert quote.book_count == 0


class _FakeArchive:
    def __init__(self, rows, line):
        self.rows = rows
        self.line = line

    def get_training_book_quotes(self, *args, **kwargs):
        return self.rows

    def get_line(self, *args, **kwargs):
        return self.line

    def get_training_quote_inputs(self, _league, _market, _date, entities, **kwargs):
        return dict.fromkeys(entities, (self.rows, self.line))


class _Stub:
    league = "WNBA"
    resolve_player_market_odds = base_mod.Stats.resolve_player_market_odds

    def check_combo_markets(self, *args, **kwargs):
        return np.nan

    def window_short_logs(self, date):
        """The repair windows the logs per gameday; the stub has none to window."""


@pytest.mark.parametrize(
    "rows,legacy_line",
    [
        ([_row("fanduel", 2.4, 0.55, 1.5)], 1.5),
        ([_row("fanduel", 2.4, None, None)], 1.5),
        ([], 0.0),
    ],
)
def test_repair_and_scratch_append_emit_identical_book_fields(
    monkeypatch, rows, legacy_line
):
    archive = _FakeArchive(rows, legacy_line)
    monkeypatch.setattr(base_mod, "archive", archive)
    stats = pd.DataFrame({"Avg10": [2.0]}, index=["P"])
    append_fields = _Stub().resolve_player_market_odds(
        stats, "AST", "2026-05-08", _AT_2
    ).iloc[0]
    cached = pd.DataFrame(
        {
            "Player": ["P"],
            "Date": ["2026-05-08"],
            "Avg10": [2.0],
            "Line": [2.0],
            "EV": [2.0],
        }
    )
    repair = resolve_cached_quotes(_Stub(), "AST", cached).iloc[0]

    assert append_fields.to_dict() == repair.to_dict()


def test_pipeline_ev_inversion_remains_explicitly_synthetic():
    matrix = pd.DataFrame(
        {
            "Result": [1.0, 2.0],
            "Line": [1.5, 2.5],
            "Odds": [0.0, 0.4],
            "EV": [2.0, 2.5],
            "Archived": [True, True],
            "QuoteSource": ["legacy", "book_direct"],
            "QuoteAuthenticity": ["authentic", "authentic"],
            "QuoteSyntheticReason": [None, None],
        }
    )

    resolved, _ = _step_synthesize_odds(matrix, "WNBA", "AST", "SkewNormal", 0.5)

    assert bool(resolved.loc[0, "Odds_synthetic"])
    assert not bool(resolved.loc[0, "Archived"])
    assert resolved.loc[0, "QuoteSource"] == "pipeline_ev_inversion"
    assert resolved.loc[0, "QuoteAuthenticity"] == DERIVED
    assert resolved.loc[0, "QuoteSyntheticReason"] == "ev_inversion"


def test_pipeline_preserves_already_resolved_synthetic_quote_provenance():
    matrix = pd.DataFrame(
        {
            "Result": [1.0, 2.0],
            "Line": [1.5, 2.5],
            "Odds": [0.5, 0.35],
            "EV": [1.5, 2.2],
            "Archived": [False, False],
            "Odds_synthetic": [True, True],
            "QuoteSource": ["model_fallback", "book_ev_inversion"],
            "QuoteAuthenticity": [SYNTHETIC, DERIVED],
            "QuoteSyntheticReason": ["no_usable_book_probability", "ev_inversion"],
            "QuoteObservedAt": [None, _AT_1],
            "QuoteBookCount": [0, 1],
        }
    )
    expected = matrix.copy(deep=True)

    resolved, _ = _step_synthesize_odds(matrix, "WNBA", "AST", "SkewNormal", 0.5)

    pd.testing.assert_frame_equal(resolved, expected)


def test_clipped_neutral_quote_regenerates_ev_from_the_clipped_line():
    matrix = pd.DataFrame(
        {
            "Line": [8.5, 7.5],
            _PRETRIM_LINE_COLUMN: [47.0, 7.5],
            "Odds": [0.5, 0.5],
            "EV": [47.0, 7.5],
            "QuoteSource": ["neutral_fallback", "neutral_fallback"],
            "QuoteAuthenticity": [SYNTHETIC, SYNTHETIC],
        }
    )

    resolved = _reconcile_clipped_neutral_quotes(
        matrix,
        league="NFL",
        market="targets",
        dist="SkewNormal",
        cv=config.stat_cv["NFL"]["targets"],
    )

    assert _PRETRIM_LINE_COLUMN not in resolved
    assert resolved.loc[0, "Odds"] == 0.5
    assert resolved.loc[0, "EV"] == pytest.approx(8.5)
    assert resolved.loc[1, "EV"] == 7.5


def test_line_clip_preserves_evidence_quotes_and_keeps_auditor_coherent(tmp_path):
    rows = 100
    matrix = pd.DataFrame(
        {
            "Player": [f"P{i}" for i in range(rows)] + ["Direct", "Derived", "Synthetic"],
            "Date": pd.date_range("2026-01-01", periods=rows + 3),
            "Result": [2.0] * (rows + 3),
            "Line": [0.5 * (i % 10 + 1) for i in range(rows)] + [8.5, 6.5, 12.0],
            "Odds": [0.5] * rows + [0.46718, 0.900301, 0.5],
            "EV": [0.5 * (i % 10 + 1) for i in range(rows)]
            + [31.043097, 32.5, 12.0],
            "Archived": [False] * rows + [True, False, False],
            "Odds_synthetic": [True] * rows + [False, True, True],
            "QuoteSource": ["model_fallback"] * rows
            + ["book_direct", "book_ev_inversion", "model_fallback"],
            "QuoteAuthenticity": [SYNTHETIC] * rows + [AUTHENTIC, DERIVED, SYNTHETIC],
            "QuoteSyntheticReason": ["no_usable_book_probability"] * rows
            + [None, "ev_inversion", "no_usable_book_probability"],
            "QuoteObservedAt": [pd.NaT] * rows + [_AT_1, _AT_1, pd.NaT],
            "QuoteBookCount": [0] * rows + [1, 1, 0],
        }
    )
    matrix[_PRETRIM_LINE_COLUMN] = matrix["Line"]
    archived = matrix["Archived"]

    clipped = _clip_lines(matrix, archived, int(archived.sum()))
    assert clipped.loc[rows, "Line"] == 8.5
    assert clipped.loc[rows + 1, "Line"] == 6.5
    assert clipped.loc[rows + 2, "Line"] == 5.0

    reconciled = _reconcile_clipped_neutral_quotes(
        clipped,
        league="NBA",
        market="FTM",
        dist="SkewNormal",
        cv=config.stat_cv["NBA"]["FTM"],
    )
    path = tmp_path / "NBA_FTM.parquet"
    reconciled.to_parquet(path)

    assert audit_matrix(path)["violations"] == []


def test_structural_pool_mask_uses_only_explicit_authenticity():
    index = pd.Index([10, 11, 12])
    values = pd.Series(["authentic", "derived", "synthetic"], index=index)

    assert _explicit_quote_authenticity_mask(values, index).tolist() == [True, False, False]
