"""``Archive.get_book_line_histories`` returns every posted rung from ``ladder``.

The line-movement snapshot picks each poll's main rung from these rows, so the reader
must hand back every rung with its de-vigged ``p_over`` — not the single tier ``add_dfs``
archives to ``odds``, which is one poll's main rung and nothing else. Rungs only become
visible once :meth:`Archive.write` flushes them, which is why prophecize snapshots line
movement after its flush.
"""

from __future__ import annotations

import contextlib
import datetime

import pandas as pd
import pytest

from sportstradamus.helpers.archive import Archive

_DAY = datetime.date(2026, 9, 6)
_T0 = datetime.datetime(2026, 9, 6, 15, 0, 0)
_LATER = _T0 + datetime.timedelta(minutes=30)


@pytest.fixture
def archive(tmp_path, monkeypatch):
    monkeypatch.setenv("SPORTSTRADAMUS_ARCHIVE_DB", str(tmp_path / "archive.duckdb"))
    if Archive._instance is not None:
        with contextlib.suppress(Exception):
            Archive._instance._connection.close()
        Archive._instance._initialized = False
    a = Archive()
    yield a
    with contextlib.suppress(Exception):
        a._connection.close()
    Archive._instance._initialized = False


def _keys(*triples: tuple[str, str, str]) -> pd.DataFrame:
    return pd.DataFrame(triples, columns=["league", "entity", "market"])


def test_every_rung_comes_back_with_its_price_in_poll_order(archive):
    later_rungs = [(0.5, 0.70), (1.5, 0.44), (2.5, 0.21)]
    first_rungs = [(0.5, 0.72), (1.5, 0.41), (2.5, 0.18)]
    # Staged out of order: the reader, not the insert order, owns the chronology.
    archive.add_ladder("NFL", "tds", _DAY, "Player A", "Sleeper", later_rungs, _LATER)
    archive.add_ladder("NFL", "tds", _DAY, "Player A", "Sleeper", first_rungs, _T0)
    archive.write()

    got = archive.get_book_line_histories(
        _keys(("NFL", "Player A", "tds")), books=["Sleeper"], since=_DAY
    )

    assert list(got.columns) == [
        "league",
        "market",
        "game_date",
        "entity",
        "book",
        "observed_at",
        "line",
        "p_over",
    ]
    assert got["observed_at"].is_monotonic_increasing
    polls = {
        stamp: sorted(zip(rows["line"], rows["p_over"], strict=True))
        for stamp, rows in got.groupby("observed_at")
    }
    assert polls == {pd.Timestamp(_T0): first_rungs, pd.Timestamp(_LATER): later_rungs}


def test_only_requested_books_keys_and_dates_come_back(archive):
    archive.add_ladder("NFL", "tds", _DAY, "Player A", "Sleeper", [(0.5, 0.60)], _T0)
    archive.add_ladder("NFL", "tds", _DAY, "Player A", "Underdog", [(0.5, 0.55)], _T0)
    archive.add_ladder("NFL", "tds", _DAY, "Player A", "ParlayPlay", [(0.5, 0.50)], _T0)
    archive.add_ladder("NFL", "rec", _DAY, "Player A", "Underdog", [(4.5, 0.50)], _T0)
    archive.add_ladder("NFL", "tds", _DAY, "Player B", "Underdog", [(0.5, 0.50)], _T0)
    yesterday = _DAY - datetime.timedelta(days=1)
    archive.add_ladder("NFL", "tds", yesterday, "Player A", "Underdog", [(0.5, 0.50)], _T0)
    archive.write()

    got = archive.get_book_line_histories(
        _keys(("NFL", "Player A", "tds")), books=["Sleeper", "Underdog"], since=_DAY
    )

    assert sorted(zip(got["book"], got["p_over"], strict=True)) == [
        ("Sleeper", 0.60),
        ("Underdog", 0.55),
    ]


def test_staged_rungs_stay_invisible_until_flushed(archive):
    archive.add_ladder("NFL", "tds", _DAY, "Player A", "Underdog", [(0.5, 0.55)], _T0)
    keys = _keys(("NFL", "Player A", "tds"))

    assert archive.get_book_line_histories(keys, books=["Underdog"], since=_DAY).empty
    archive.write()
    assert len(archive.get_book_line_histories(keys, books=["Underdog"], since=_DAY)) == 1
