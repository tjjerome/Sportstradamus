"""Unit tests for ``Stats.combo_quote``: the honest combo-sum training-quote builder.

``combo_quote`` resolves each component's own modal-line book quote, admits a
player only when EVERY component is book-priced (``source == "book_direct"``,
Bernoulli p a finite open-interval probability), and prices the weighted sum
through the NORTA kernel in ``helpers.combined_markets``. Provenance is the
weakest link: min book count, oldest component observation, and the book cohort
of the component with the fewest real (non-DFS) books.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from sportstradamus.helpers.distributions import get_ev
from sportstradamus.helpers.training_quotes import (
    COMBO_SUM_SOURCE,
    DERIVED,
    ArchivedBookQuote,
)
from sportstradamus.stats import base
from sportstradamus.stats.base import ComboSpec, Stats

# Synthetic league: no stat_meta cells and no corr_market_summary.parquet, so the
# kernel's same-player rho lookup reads 0 (independence) deterministically.
_LEAGUE = "TL"
_DATE = "2026-06-03"
_AT = datetime.datetime(2026, 6, 3, 18, 0, 0)
_TS_A = datetime.datetime(2026, 6, 3, 12, 0, 0)
_TS_B = datetime.datetime(2026, 6, 3, 10, 0, 0)
_FANTASY = "myfantasy"

_CV = {"subA": 0.3, "subB": 0.5}
_DIST = {"subA": "Gamma", "subB": "NegBin"}
_UNDER = {"subA": 0.45, "subB": 0.60}
_SUBLINE = {"subA": 20.5, "subB": 4.5}

_MEAN_A = get_ev(_SUBLINE["subA"], _UNDER["subA"], cv=_CV["subA"], dist="Gamma")
_MEAN_B = get_ev(_SUBLINE["subB"], _UNDER["subB"], cv=_CV["subB"], dist="NegBin")

_RECORD_COLUMNS = {
    "Line",
    "Odds",
    "EV",
    "Archived",
    "Odds_synthetic",
    "QuoteSource",
    "QuoteAuthenticity",
    "QuoteSyntheticReason",
    "QuoteObservedAt",
    "QuoteBookCount",
}


class _StubArchive:
    """Minimal ``get_training_quote_inputs`` surface (test_book_fallback_prob style)."""

    def __init__(self, inputs):
        self._inputs = inputs  # {market: {player: (rows, legacy_line)}}

    def get_training_quote_inputs(self, league, market, date, entities, at=None):
        per_market = self._inputs.get(market, {})
        return {e: per_market.get(e, ([], None)) for e in entities}


class _SpecStats(Stats):
    """Bare Stats subclass exposing a fixed fantasy spec and Bernoulli p table."""

    league = _LEAGUE
    log_strings = {"player": "playerName"}

    def __init__(self, spec, bernoulli_p=None, trailing=()):
        self.spec = spec
        self.bernoulli_p = bernoulli_p or {}
        self.short_gamelog = pd.DataFrame(
            {"playerName": ["P"] * len(trailing), "subB": list(trailing)}
        )

    def _fantasy_combo_spec(self, market, player):
        return self.spec if market == _FANTASY else None

    def _combo_bernoulli_p(self, name, player, date):
        return self.bernoulli_p[name]


def _rows(sub, books, ts):
    return [ArchivedBookQuote(book, None, _UNDER[sub], _SUBLINE[sub], ts) for book in books]


def _direct_inputs(player="P"):
    return {
        "subA": {player: (_rows("subA", ("fanduel", "draftkings"), _TS_A), _SUBLINE["subA"])},
        "subB": {player: (_rows("subB", ("Underdog", "betmgm"), _TS_B), _SUBLINE["subB"])},
    }


@pytest.fixture
def combo_env(monkeypatch):
    monkeypatch.setattr(base, "stat_dist", {_LEAGUE: _DIST})
    monkeypatch.setattr(base, "stat_cv", {_LEAGUE: _CV})
    monkeypatch.setattr(base, "book_weights", {})

    def set_archive(inputs):
        monkeypatch.setattr(base, "archive", _StubArchive(inputs))

    return set_archive


def test_weighted_spec_prices_component_sum(combo_env):
    combo_env(_direct_inputs())
    stats = _SpecStats(ComboSpec(marginals=(("subA", 2.0), ("subB", 1.5))))
    expected_ev = 2.0 * _MEAN_A + 1.5 * _MEAN_B
    line = float(np.floor(expected_ev)) + 0.5

    quotes = stats.combo_quote(_FANTASY, ["P"], _DATE, _AT, lines={"P": line})

    assert set(quotes) == {"P"}
    q = quotes["P"]
    assert 0.0 < q.over_probability < 1.0
    # The kernel mean is the exact weighted sum of component marginal means.
    assert q.ev == pytest.approx(expected_ev)
    assert q.line == line
    assert q.sum_sd > 0.0
    grid = [line - 8, line - 3, line, line + 3, line + 8]
    probs = [q.under_prob_at(x) for x in grid]
    assert probs == sorted(probs)
    assert probs[0] < probs[-1]
    assert q.under_prob_at(line) == pytest.approx(1.0 - q.over_probability)


def test_missing_component_omits_player(combo_env):
    inputs = _direct_inputs()
    del inputs["subB"]
    combo_env(inputs)
    stats = _SpecStats(ComboSpec(marginals=(("subA", 1.0), ("subB", 1.0))))

    assert stats.combo_quote(_FANTASY, ["P"], _DATE, _AT, lines={"P": 25.5}) == {}


def test_non_direct_component_omits_player(combo_env):
    inputs = _direct_inputs()
    # Legacy ev-only row: resolves via book_ev_inversion, not book_direct.
    inputs["subB"] = {"P": ([ArchivedBookQuote("fanduel", 4.2, None, None, _TS_B)], 4.5)}
    combo_env(inputs)
    stats = _SpecStats(ComboSpec(marginals=(("subA", 1.0), ("subB", 1.0))))

    assert stats.combo_quote(_FANTASY, ["P"], _DATE, _AT, lines={"P": 25.5}) == {}


def test_assumable_component_falls_back_to_the_trailing_rate(combo_env):
    inputs = _direct_inputs()
    del inputs["subB"]
    combo_env(inputs)
    spec = ComboSpec(marginals=(("subA", 1.0), ("subB", 3.0)), assumable=frozenset({"subB"}))
    # 2 of 5 trailing games recorded the stat, mean 0.6 -> p = 0.4, weight 3 * 0.6 / 0.4.
    stats = _SpecStats(spec, trailing=[0, 0, 1, 0, 2])

    q = stats.combo_quote(_FANTASY, ["P"], _DATE, _AT, lines={"P": 25.5})["P"]

    assert q.ev == pytest.approx(_MEAN_A + 3.0 * 0.6)
    assert "assumed:subB" in q.synthetic_reason
    assert q.sum_sd > 0.0


def test_assumable_component_with_no_trailing_occurrences_contributes_nothing(combo_env):
    inputs = _direct_inputs()
    del inputs["subB"]
    combo_env(inputs)
    spec = ComboSpec(marginals=(("subA", 1.0), ("subB", 3.0)), assumable=frozenset({"subB"}))
    stats = _SpecStats(spec, trailing=[0, 0, 0])

    q = stats.combo_quote(_FANTASY, ["P"], _DATE, _AT, lines={"P": 25.5})["P"]

    assert q.ev == pytest.approx(_MEAN_A)
    assert "assumed:subB" in q.synthetic_reason


def test_assumable_component_prefers_its_real_quote(combo_env):
    combo_env(_direct_inputs())
    spec = ComboSpec(marginals=(("subA", 1.0), ("subB", 3.0)), assumable=frozenset({"subB"}))
    stats = _SpecStats(spec, trailing=[0, 0, 1, 0, 2])

    q = stats.combo_quote(_FANTASY, ["P"], _DATE, _AT, lines={"P": 25.5})["P"]

    assert q.ev == pytest.approx(_MEAN_A + 3.0 * _MEAN_B)
    assert "assumed" not in q.synthetic_reason


def test_no_positive_line_omits_player(combo_env):
    combo_env(_direct_inputs())
    stats = _SpecStats(ComboSpec(marginals=(("subA", 1.0), ("subB", 1.0))))

    assert stats.combo_quote(_FANTASY, ["P"], _DATE, _AT, lines={}) == {}
    assert stats.combo_quote(_FANTASY, ["P"], _DATE, _AT, lines={"P": 0.0}) == {}


def test_combo_props_market_sums_unit_weights(combo_env, monkeypatch):
    monkeypatch.setitem(base.combo_props, "TESTCOMBO", ["subA", "subB"])
    inputs = _direct_inputs()
    inputs["TESTCOMBO"] = {"P": ([], 24.5)}  # the combo market's own consensus line
    combo_env(inputs)
    stats = object.__new__(Stats)
    stats.league = _LEAGUE

    quotes = stats.combo_quote("TESTCOMBO", ["P"], _DATE, _AT)

    q = quotes["P"]
    assert q.line == 24.5
    assert q.ev == pytest.approx(_MEAN_A + _MEAN_B)
    assert q.source == COMBO_SUM_SOURCE


# The fantasy market itself has no stat_meta cell, so its own quote inverts Gamma/cv=1.
_OWN_UNDER, _OWN_LINE = 0.55, 30.5
_OWN_MEAN = get_ev(_OWN_LINE, _OWN_UNDER, cv=1, dist="Gamma")


def _with_own_quote(inputs, market, book="Underdog"):
    return inputs | {
        market: {"P": ([ArchivedBookQuote(book, None, _OWN_UNDER, _OWN_LINE, _TS_A)], _OWN_LINE)}
    }


def test_fantasy_market_quote_pulls_the_sum_halfway_to_it(combo_env):
    spec = ComboSpec(marginals=(("subA", 1.0), ("subB", 1.0)))
    combo_env(_direct_inputs())
    unblended = _SpecStats(spec).combo_quote(_FANTASY, ["P"], _DATE, _AT, lines={"P": 25.5})["P"]
    combo_env(_with_own_quote(_direct_inputs(), _FANTASY))

    q = _SpecStats(spec).combo_quote(_FANTASY, ["P"], _DATE, _AT, lines={"P": 25.5})["P"]

    assert unblended.ev == pytest.approx(_MEAN_A + _MEAN_B)
    assert q.ev == pytest.approx(0.5 * (unblended.ev + _OWN_MEAN))
    assert q.synthetic_reason == "component_sum+book_mean"
    # Pure location shift: same shape, same sd, evaluated `shift` further along.
    shift = q.ev - unblended.ev
    assert q.sum_sd == pytest.approx(unblended.sum_sd)
    for x in (18.0, 25.5, 33.0):
        assert q.under_prob_at(x + shift) == pytest.approx(unblended.under_prob_at(x))
    assert q.under_prob_at(25.5) == pytest.approx(1.0 - q.over_probability)


def test_simple_combo_ignores_its_own_market_quote(combo_env, monkeypatch):
    # A combo_props spec settles exactly as the sum of its components, so the sum is
    # already unbiased and blending only pulls it off centre — measured on all five
    # graded simple combos. Only fantasy specs, which approximate their formula, blend.
    monkeypatch.setitem(base.combo_props, "TESTCOMBO", ["subA", "subB"])
    combo_env(_with_own_quote(_direct_inputs(), "TESTCOMBO"))
    stats = object.__new__(Stats)
    stats.league = _LEAGUE

    q = stats.combo_quote("TESTCOMBO", ["P"], _DATE, _AT)["P"]

    assert q.ev == pytest.approx(_MEAN_A + _MEAN_B)
    assert q.synthetic_reason == "component_sum"


def test_fantasy_market_without_its_own_quote_keeps_the_component_sum(combo_env):
    # NFL fantasy: no book has quoted the composite since 2023, so there is nothing
    # to blend with and the component sum stands alone.
    combo_env(_direct_inputs())
    stats = _SpecStats(ComboSpec(marginals=(("subA", 1.0), ("subB", 1.0))))

    q = stats.combo_quote(_FANTASY, ["P"], _DATE, _AT, lines={"P": 25.5})["P"]

    assert q.ev == pytest.approx(_MEAN_A + _MEAN_B)
    assert q.synthetic_reason == "component_sum"


def test_unknown_market_returns_empty(combo_env):
    combo_env({})
    stats = object.__new__(Stats)
    stats.league = _LEAGUE

    assert stats.combo_quote("no such market", ["P"], _DATE, _AT) == {}


def test_unconfigured_submarket_defaults_gamma_cv1(combo_env):
    # MLB triples has no stat_meta cell; unconfigured components price Gamma/cv=1.
    combo_env({"subC": {"P": ([ArchivedBookQuote("fanduel", None, 0.5, 2.5, _TS_A)], 2.5)}})
    stats = _SpecStats(ComboSpec(marginals=(("subC", 1.0),)))

    quotes = stats.combo_quote(_FANTASY, ["P"], _DATE, _AT, lines={"P": 3.5})

    assert quotes["P"].ev == pytest.approx(get_ev(2.5, 0.5, cv=1, dist="Gamma"))


def test_bernoulli_component_and_analytics(combo_env):
    combo_env(_direct_inputs())
    spec = ComboSpec(
        marginals=(("subA", 1.0), ("subB", 1.0)),
        bernoulli=(("win", 6.0),),
        analytics=("win_map",),
    )
    stats = _SpecStats(spec, bernoulli_p={"win": 0.62})

    quotes = stats.combo_quote(_FANTASY, ["P"], _DATE, _AT, lines={"P": 30.5})

    q = quotes["P"]
    assert q.ev == pytest.approx(_MEAN_A + _MEAN_B + 6.0 * 0.62)
    assert q.synthetic_reason == "component_sum+win_map"


@pytest.mark.parametrize("p", [float("nan"), 0.0, 1.0])
def test_invalid_bernoulli_p_omits_player(combo_env, p):
    combo_env(_direct_inputs())
    spec = ComboSpec(marginals=(("subA", 1.0), ("subB", 1.0)), bernoulli=(("win", 6.0),))
    stats = _SpecStats(spec, bernoulli_p={"win": p})

    assert stats.combo_quote(_FANTASY, ["P"], _DATE, _AT, lines={"P": 30.5}) == {}


def test_provenance_weakest_link(combo_env):
    combo_env(_direct_inputs())
    stats = _SpecStats(ComboSpec(marginals=(("subA", 1.0), ("subB", 1.0))))

    q = stats.combo_quote(_FANTASY, ["P"], _DATE, _AT, lines={"P": 25.5})["P"]

    assert q.source == COMBO_SUM_SOURCE == "combo_sum"
    assert q.authenticity == DERIVED
    assert q.synthetic_reason == "component_sum"
    # subB's Underdog row drops out of its cohort beside betmgm, leaving one book
    # against subA's two, so subB is the weakest link and sets the reported count.
    assert q.book_count == 1
    assert q.books == ("betmgm",)
    assert q.observed_at == _TS_B  # oldest component observation
    record = q.as_record()
    assert set(record) == _RECORD_COLUMNS
    assert record["Archived"] is False
    assert record["Odds_synthetic"] is True
    assert record["QuoteSource"] == COMBO_SUM_SOURCE
