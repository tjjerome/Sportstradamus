"""Consensus-read guards: line-divergence exclusion + DFS platform weight cap.

The 2026-08 Sleeper WNBA incident: the platform posted discounted/moved lines
(PRA 24.5 against a 33.5 sportsbook median) whose fabricated 50/50 pricing
entered the archive as fair quotes and then dominated the weighted consensus
(~92% effective Sleeper weight after zero-row books renormalized away).
``Archive._weighted_book_ev`` now (1) drops rows whose line diverges from the
cohort median, and (2) caps each DFS platform's normalized weight while a real
sportsbook remains; ``dfs_boost_probs`` prices one-sided offers at their
payout-implied breakeven instead of a fabricated symmetric pair.

The fake ``XLG``/``XMKT`` league/market keep ``book_weights`` lookups empty so
every book weighs 1 unless a test plants weights itself.
"""

from __future__ import annotations

import contextlib
import datetime

import pytest

from sportstradamus.helpers import config
from sportstradamus.helpers.archive import Archive
from sportstradamus.helpers.distributions import dfs_boost_probs, no_vig_odds

_TS = datetime.datetime(2026, 8, 28, 12, 0, 0)
_DATE = "2026-08-28"
_LG, _MKT = "XLG", "XMKT"


@pytest.fixture
def archive(tmp_path, monkeypatch):
    db_path = tmp_path / "archive.duckdb"
    monkeypatch.setenv("SPORTSTRADAMUS_ARCHIVE_DB", str(db_path))
    if Archive._instance is not None:
        with contextlib.suppress(Exception):
            Archive._instance._connection.close()
        Archive._instance._initialized = False
    a = Archive()
    yield a
    with contextlib.suppress(Exception):
        a._connection.close()
    Archive._instance._initialized = False


def _insert(archive, book, ev, line):
    archive._connection.execute(
        "INSERT INTO odds (league, market, game_date, entity, book, ev, observed_at, "
        "under_prob, line) VALUES (?, ?, DATE '2026-08-28', 'P', ?, ?, ?, NULL, ?)",
        [_LG, _MKT, book, ev, _TS, line],
    )


def test_divergent_line_row_is_dropped_from_consensus(archive):
    """A 9-point-off line is a different offer and drops; half-point jitter survives."""
    _insert(archive, "fanduel", 30.0, 33.5)
    _insert(archive, "draftkings", 31.0, 33.5)
    _insert(archive, "betmgm", 30.5, 34.0)
    _insert(archive, "Sleeper", 24.0, 24.5)

    # median 33.5, tolerance max(2.0, 0.25 * 33.5) = 8.375: |24.5 - 33.5| = 9 drops,
    # |34.0 - 33.5| = 0.5 survives.
    assert archive.get_ev(_LG, _MKT, _DATE, "P") == pytest.approx((30.0 + 31.0 + 30.5) / 3)


def test_fewer_than_three_lined_rows_leaves_filter_inert(archive):
    """Two lined rows have no meaningful median; both survive however far apart."""
    _insert(archive, "fanduel", 31.0, 33.5)
    _insert(archive, "draftkings", 24.0, 24.5)

    assert archive.get_ev(_LG, _MKT, _DATE, "P") == pytest.approx((31.0 + 24.0) / 2)


def test_null_line_rows_are_immune_to_divergence(archive):
    """Team markets and pre-migration rows carry no line; they must never drop."""
    _insert(archive, "fanduel", 30.0, 33.5)
    _insert(archive, "draftkings", 31.0, 33.5)
    _insert(archive, "betmgm", 30.5, 33.5)
    _insert(archive, "legacy", 29.0, None)

    expected = (30.0 + 31.0 + 30.5 + 29.0) / 4
    assert archive.get_ev(_LG, _MKT, _DATE, "P") == pytest.approx(expected)


def test_dfs_platform_drops_out_when_a_real_book_remains(archive, monkeypatch):
    """A sportsbook on the entry prices it alone, whatever weight the platform carries."""
    monkeypatch.setitem(config.book_weights, _LG, {_MKT: {"Sleeper": 12.0, "fanduel": 1.0}})
    _insert(archive, "fanduel", 20.0, 24.5)
    _insert(archive, "Sleeper", 40.0, 24.5)

    # Sleeper's normalized 12/13 ~ 0.92 is the live WNBA PRA failure mode, but even an
    # even split misprices: a pick'em platform's implied probability sits near 0.5
    # however far the truth is, so it contributes no signal beside a real book.
    assert archive.get_ev(_LG, _MKT, _DATE, "P") == pytest.approx(20.0)


def test_dfs_only_cell_keeps_the_platforms(archive, monkeypatch):
    """With no sportsbook to defer to, platform weights stand as configured."""
    monkeypatch.setitem(config.book_weights, _LG, {_MKT: {"Sleeper": 3.0, "Underdog": 1.0}})
    _insert(archive, "Sleeper", 40.0, 24.5)
    _insert(archive, "Underdog", 20.0, 24.5)

    assert archive.get_ev(_LG, _MKT, _DATE, "P") == pytest.approx(0.75 * 40.0 + 0.25 * 20.0)


def test_dfs_boost_probs_two_sided_matches_proportional_devig():
    assert dfs_boost_probs(1.9, 2.05) == pytest.approx(no_vig_odds(1.9, 2.05))


def test_dfs_boost_probs_symmetric_pair_prices_even():
    assert dfs_boost_probs(1.53, 1.53) == [0.5, 0.5]
    assert dfs_boost_probs(1.78, 1.78) == [0.5, 0.5]


def test_dfs_boost_probs_one_sided_is_raw_breakeven():
    """The offered side stores the platform's own claim, 1/odds, with no hold shave."""
    assert dfs_boost_probs(1.35, 0) == pytest.approx([1 / 1.35, 1 - 1 / 1.35])
    assert dfs_boost_probs(0, 1.35) == pytest.approx([1 - 1 / 1.35, 1 / 1.35])
    assert dfs_boost_probs(1.35, float("nan")) == pytest.approx([1 / 1.35, 1 - 1 / 1.35])


def test_dfs_boost_probs_no_sides_prices_even():
    assert dfs_boost_probs(0, 0) == [0.5, 0.5]
