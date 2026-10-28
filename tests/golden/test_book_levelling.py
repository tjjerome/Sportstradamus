"""The I6d levelled training read, against docs/handoffs/i6d-decode-preregistration.md.

Pins the protocol's read-time rules on hand-built rows: the stamp layer (§2), the producer
class (§3), a migrated row's decode block (§4), the pre-registered decodes (§7) and every §11
levelling rule; then the source a levelled cohort resolves to, the archive read that joins each
row's same-observation ladder rung, and the training join asking for that read. The decode
table is a fixture throughout; ``scripts/fit_book_decode.py`` writes the committed one.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import math

import numpy as np
import pandas as pd
import pytest

from sportstradamus.helpers import book_levelling
from sportstradamus.helpers.archive import Archive
from sportstradamus.helpers.book_levelling import (
    WS1_NATIVE_FROM,
    BlockVerdict,
    block_key,
    decode_under,
    is_native,
    level_book_rows,
    stamp_layer,
)
from sportstradamus.helpers.training_quotes import (
    AUTHENTIC,
    LADDER_REPAIRED,
    REDECODED,
    ArchivedBookQuote,
    resolve_training_quote,
)
from sportstradamus.stats import base as base_mod

_LEAGUE, _MARKET = "NFL", "receptions"
_LINE, _EV = 4.5, 4.1
# Before the WS1 writer: a midnight stamp and a live poll, both migrated, in one live epoch.
_OLD_GAME = datetime.date(2025, 11, 2)
_MIDNIGHT = datetime.datetime(2025, 11, 2)
_OLD_POLL = datetime.datetime(2025, 11, 2, 9, 41)
_EPOCH = (_LEAGUE, _MARKET, "live", "2025-11")
# After it: a native live poll and a backfill stamp.
_NEW_GAME = datetime.date(2026, 9, 13)
_NEW_POLL = datetime.datetime(2026, 9, 13, 11, 37)
_NEW_STAMP = datetime.datetime(2026, 9, 13, 1)


@pytest.fixture
def table(monkeypatch):
    """Install a fixture decode table; ``decode_table`` is cached, so patch the name."""

    def install(verdicts):
        monkeypatch.setattr(book_levelling, "decode_table", lambda: verdicts)

    return install


@pytest.mark.parametrize(
    ("observed_at", "layer"),
    [
        (datetime.datetime(2025, 1, 5, 1), "01:00"),
        (datetime.datetime(2025, 1, 5, 1, 0, 0, 500), "live"),
        (datetime.datetime(2025, 1, 5, 3), "live"),
        (datetime.datetime(2025, 1, 5), "00:00"),
        (datetime.datetime(2025, 1, 5, 23), "23:00"),
    ],
)
def test_only_an_exact_stamp_hour_is_a_stamp_layer(observed_at, layer):
    assert stamp_layer(observed_at) == layer


def test_the_producer_class_reads_layer_time_and_rung():
    assert not is_native("00:00", _NEW_POLL, 0.4)
    assert not is_native("live", WS1_NATIVE_FROM - datetime.timedelta(seconds=1), None)
    assert is_native("live", WS1_NATIVE_FROM, None)
    assert is_native("01:00", _NEW_STAMP, 0.55)
    assert not is_native("01:00", _NEW_STAMP, None)


@pytest.mark.parametrize(
    ("league", "game_date", "layer", "key"),
    [
        ("NFL", datetime.date(2024, 1, 7), "01:00", ("01:00", "2023")),
        ("NBA", datetime.date(2025, 10, 21), "01:00", ("01:00", "2025")),
        ("MLB", datetime.date(2025, 4, 2), "02:00", ("02:00", "2025")),
        ("NHL", datetime.date(2025, 11, 3), "00:00", ("live", "2025-11")),
        ("NHL", datetime.date(2025, 11, 3), "live", ("live", "2025-11")),
    ],
)
def test_a_backfill_blocks_by_season_and_the_rest_by_game_month(league, game_date, layer, key):
    assert block_key(league, game_date, layer) == key


def test_the_negbin_decode_keeps_its_dispersion_at_cv_one():
    """At cv 1 with no ``r`` get_odds reads NegBin as Poisson; the decode passes r = 1/cv."""
    negbin = decode_under("NegBin", 0.5, 0.4, 1.0)

    assert negbin == pytest.approx(1 / 1.4)  # geometric P(X = 0) at mean 0.4
    assert negbin != pytest.approx(decode_under("Poisson", 0.5, 0.4, None))
    assert decode_under("Poisson", 0.5, 0.4, None) == pytest.approx(math.exp(-0.4))
    with pytest.raises(ValueError, match="DPO"):
        decode_under("DPO", 0.5, 0.4, 1.0)


def test_a_midnight_row_in_an_accepted_block_is_redecoded(table):
    table({_EPOCH: BlockVerdict("NegBin", 0.6)})
    row = ArchivedBookQuote("draftkings", _EV, 0.47, _LINE, _MIDNIGHT)

    [levelled] = level_book_rows(_LEAGUE, _MARKET, _OLD_GAME, [row])

    assert levelled == dataclasses.replace(
        row, under_probability=decode_under("NegBin", _LINE, _EV, 0.6), levelling=REDECODED
    )


def test_a_midnight_row_in_a_demoted_block_is_dropped(table):
    table({_EPOCH: BlockVerdict(None, None)})
    row = ArchivedBookQuote("draftkings", _EV, 0.47, _LINE, _MIDNIGHT)

    assert level_book_rows(_LEAGUE, _MARKET, _OLD_GAME, [row]) == []


def test_a_stored_accepted_row_keeps_its_price_unless_it_has_none(table):
    table({_EPOCH: BlockVerdict("stored", None)})
    priced = ArchivedBookQuote("draftkings", _EV, 0.47, _LINE, _MIDNIGHT)
    unpriced = ArchivedBookQuote("fanduel", _EV, None, _LINE, _MIDNIGHT)

    levelled = level_book_rows(_LEAGUE, _MARKET, _OLD_GAME, [priced, unpriced])

    assert levelled == [dataclasses.replace(priced, levelling=REDECODED)]


def test_a_migrated_poll_without_an_ev_in_a_parametric_block_is_dropped(table):
    table({_EPOCH: BlockVerdict("NegBin", 0.6)})
    row = ArchivedBookQuote("draftkings", None, 0.47, _LINE, _OLD_POLL)

    assert level_book_rows(_LEAGUE, _MARKET, _OLD_GAME, [row]) == []


def test_a_native_live_row_is_returned_unchanged(table):
    # A demoted verdict on its month shows the table is never consulted for a native row.
    table({(_LEAGUE, _MARKET, "live", "2026-09"): BlockVerdict(None, None)})
    row = ArchivedBookQuote("draftkings", _EV, 0.47, _LINE, _NEW_POLL)

    assert level_book_rows(_LEAGUE, _MARKET, _NEW_GAME, [row]) == [row]


@pytest.mark.parametrize(
    ("rung", "under", "levelling"),
    [(0.61, 1 - 0.61, LADDER_REPAIRED), (0.59, 0.45, None)],
)
def test_a_native_row_takes_its_rung_only_past_the_disagreement_bar(table, rung, under, levelling):
    table({})
    row = ArchivedBookQuote(
        "draftkings", _EV, 0.45, _LINE, _NEW_STAMP, ladder_over_probability=rung
    )

    [levelled] = level_book_rows(_LEAGUE, _MARKET, _NEW_GAME, [row])

    assert levelled.under_probability == pytest.approx(under)
    assert levelled.levelling == levelling


@pytest.mark.parametrize(
    ("others", "others_line", "kept"),
    [
        ((0.82, 0.85), _LINE, False),
        ((0.82, 0.85), _LINE + 1, True),
        ((0.60, 0.70), _LINE, True),
        # §11 asks the median to exceed .70; at .70 the even price stands.
        ((0.70, 0.70), _LINE, True),
    ],
)
def test_an_even_native_row_without_a_rung_drops_only_under_sure_books(
    table, others, others_line, kept
):
    table({})
    even = ArchivedBookQuote("draftkings", _EV, 0.5, _LINE, _NEW_POLL)
    rows = [
        even,
        *(
            ArchivedBookQuote(book, _EV, under, others_line, _NEW_POLL)
            for book, under in zip(("fanduel", "betmgm"), others, strict=True)
        ),
        # A pick'em platform is never book evidence, however sure it reads.
        ArchivedBookQuote("Underdog", _EV, 0.95, _LINE, _NEW_POLL),
    ]

    assert level_book_rows(_LEAGUE, _MARKET, _NEW_GAME, rows) == (rows if kept else rows[1:])


def test_an_even_native_row_with_an_agreeing_rung_is_not_the_defect(table):
    table({})
    even = ArchivedBookQuote("draftkings", _EV, 0.5, _LINE, _NEW_POLL, ladder_over_probability=0.52)
    rows = [
        even,
        *(ArchivedBookQuote(book, _EV, 0.83, _LINE, _NEW_POLL) for book in ("fanduel", "betmgm")),
    ]

    assert level_book_rows(_LEAGUE, _MARKET, _NEW_GAME, rows) == rows


def test_a_dfs_row_passes_through_a_demoted_block(table):
    table({_EPOCH: BlockVerdict(None, None)})
    row = ArchivedBookQuote("Underdog", _EV, 0.5, _LINE, _MIDNIGHT)

    assert level_book_rows(_LEAGUE, _MARKET, _OLD_GAME, [row]) == [row]


def test_a_migrated_row_with_no_verdict_stays_as_stored(table):
    table({(_LEAGUE, "carries", "live", "2025-11"): BlockVerdict(None, None)})
    row = ArchivedBookQuote("draftkings", _EV, 0.47, _LINE, _MIDNIGHT)

    assert level_book_rows(_LEAGUE, _MARKET, _OLD_GAME, [row]) == [row]


@pytest.mark.parametrize(
    ("levellings", "source"),
    [
        ((REDECODED, LADDER_REPAIRED), "book_redecoded"),
        ((REDECODED, None), "book_redecoded"),
        ((LADDER_REPAIRED, None), "book_ladder_repaired"),
        ((None, None), "book_direct"),
    ],
)
def test_a_levelled_cohort_names_its_source_and_stays_authentic(levellings, source):
    rows = [
        ArchivedBookQuote(book, _EV, 0.45, _LINE, _NEW_POLL, levelling=levelling)
        for book, levelling in zip(("draftkings", "fanduel"), levellings, strict=True)
    ]

    quote = resolve_training_quote(
        rows, legacy_line=_LINE, fallback_line=_LINE, fallback_ev=None, dist="NegBin", cv=0.6
    )

    assert (quote.source, quote.authenticity) == (source, AUTHENTIC)


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


def test_the_archive_read_levels_against_the_same_observation_rung(archive, table):
    table({_EPOCH: BlockVerdict("NegBin", 0.6)})
    stamp = datetime.datetime(2025, 11, 2, 1)
    cell = [_LEAGUE, _MARKET, _OLD_GAME, "P"]
    archive._connection.executemany(
        "INSERT INTO odds (league, market, game_date, entity, book, ev, observed_at, "
        "under_prob, line) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            [*cell, "draftkings", _EV, stamp, 0.45, _LINE],
            [*cell, "fanduel", _EV, _MIDNIGHT, 0.47, _LINE],
        ],
    )
    archive._connection.executemany(
        "INSERT INTO ladder (league, market, game_date, entity, book, line, p_over, "
        "observed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            # Two rungs on the row's own line and observation average to its rung, .61.
            [*cell, "draftkings", _LINE, 0.60, stamp],
            [*cell, "draftkings", _LINE, 0.62, stamp],
            # Neither another line nor another observation is the row's rung.
            [*cell, "draftkings", _LINE + 1, 0.30, stamp],
            [*cell, "draftkings", _LINE, 0.90, stamp - datetime.timedelta(minutes=30)],
        ],
    )
    at = datetime.datetime(2025, 11, 2, 12)

    stored, stored_line = archive.get_training_quote_inputs(
        _LEAGUE, _MARKET, _OLD_GAME, ["P"], at=at
    )["P"]
    (repaired, redecoded), line = archive.get_training_quote_inputs(
        _LEAGUE, _MARKET, _OLD_GAME, ["P"], at=at, levelled=True
    )["P"]

    assert stored == [
        ArchivedBookQuote("draftkings", _EV, 0.45, _LINE, stamp),
        ArchivedBookQuote("fanduel", _EV, 0.47, _LINE, _MIDNIGHT),
    ]
    assert (repaired.book, repaired.levelling) == ("draftkings", LADDER_REPAIRED)
    assert repaired.ladder_over_probability == pytest.approx(0.61)
    assert repaired.under_probability == pytest.approx(0.39)
    assert redecoded == dataclasses.replace(
        stored[1], under_probability=decode_under("NegBin", _LINE, _EV, 0.6), levelling=REDECODED
    )
    assert line == stored_line == _LINE


class _RecordingArchive:
    def __init__(self):
        self.calls = []

    def get_training_quote_inputs(self, _league, market, _date, entities, **kwargs):
        self.calls.append((market, kwargs))
        return dict.fromkeys(entities, ([], 0.0))


class _Stub:
    league = "WNBA"
    resolve_player_market_odds = base_mod.Stats.resolve_player_market_odds
    combo_quote = base_mod.Stats.combo_quote

    def check_combo_markets(self, *args, **kwargs):
        return np.nan


def test_the_training_join_reads_every_quote_levelled(monkeypatch):
    archive = _RecordingArchive()
    monkeypatch.setattr(base_mod, "archive", archive)
    at = datetime.datetime(2026, 5, 8, 16)
    stats = pd.DataFrame({"Avg10": [20.0]}, index=["P"])

    _Stub().resolve_player_market_odds(stats, "PRA", "2026-05-08", at)

    # The unpriced player takes the combo pass, whose component and combo reads level too.
    assert {market for market, _ in archive.calls} == {"PRA", "PTS", "REB", "AST"}
    assert all(kwargs == {"at": at, "levelled": True} for _, kwargs in archive.calls)
