"""The sportsbooks' price at an offer's line: read from the ladder, stamped per run, filled nightly."""

import numpy as np
import pandas as pd
import pytest

from sportstradamus import book_stamp
from sportstradamus.helpers.training_quotes import DFS_PLATFORM_BOOKS

SCORED_AT = pd.Timestamp("2026-11-20 23:50:00")


def _rungs(rows):
    frame = pd.DataFrame(rows, columns=["entity", "book", "line", "p_over", "minutes_old"])
    frame["observed_at"] = SCORED_AT - pd.to_timedelta(frame.pop("minutes_old"), unit="min")
    return frame


def _book_inputs(rows, players, lines):
    return book_stamp.book_inputs(_rungs(rows), players, np.array(lines, dtype=float), SCORED_AT)


class _Ladder:
    """Stands in for ``Archive.get_ladder_rungs``: each slate's rungs, and the reads made."""

    def __init__(self, slates):
        self.slates, self.reads = slates, []

    def get_ladder_rungs(self, league, market, date, entities, *, exclude_books=(), at=None):
        self.reads.append(
            {"slate": (league, market, date), "exclude_books": set(exclude_books), "at": at}
        )
        rungs = self.slates.get((league, market, date), _rungs([]))
        return rungs[rungs["entity"].isin(entities)]


def _offers(rows, index=None):
    return pd.DataFrame(
        rows, columns=["League", "Market", "Date", "Player", "Line", "Bet"], index=index
    )


def test_book_inputs_average_the_books_at_the_offered_line():
    inputs = _book_inputs(
        [
            ("Ann", "dk", 24.5, 0.50, 10),
            ("Ann", "fd", 24.5, 0.56, 30),
            ("Ann", "fd", 25.5, 0.40, 30),
        ],
        ["Ann"],
        [24.5],
    )

    assert inputs["Book Over"].iat[0] == pytest.approx(0.53)
    assert inputs["Book Count"].iat[0] == 2
    assert inputs["Book Observed At"].iat[0] == SCORED_AT - pd.Timedelta(minutes=10)
    assert inputs["Book Class"].iat[0] == "A"


def test_book_inputs_class_b_when_the_books_price_only_another_line():
    inputs = _book_inputs([("Ann", "dk", 25.5, 0.40, 10)], ["Ann"], [24.5])

    assert np.isnan(inputs["Book Over"].iat[0])
    assert inputs["Book Count"].iat[0] == 0
    assert pd.isna(inputs["Book Observed At"].iat[0])
    assert inputs["Book Class"].iat[0] == "B"


def test_book_inputs_class_c_when_no_book_quotes_the_player():
    inputs = _book_inputs([("Ann", "dk", 24.5, 0.50, 10)], ["Ann", "Bob"], [24.5, 24.5])

    assert inputs["Book Class"].tolist() == ["A", "C"]
    assert inputs.index.tolist() == [0, 1]


def test_book_inputs_ignore_a_quote_older_than_the_freshness_cut():
    stale = book_stamp.BOOK_MAX_AGE / pd.Timedelta(minutes=1) + 1
    inputs = _book_inputs(
        [
            ("Ann", "dk", 24.5, 0.50, stale),
            ("Ann", "fd", 25.5, 0.40, 10),
            ("Bob", "dk", 24.5, 0.50, stale),
        ],
        ["Ann", "Bob"],
        [24.5, 24.5],
    )

    assert inputs["Book Class"].tolist() == ["B", "C"]


def test_book_inputs_class_c_from_the_archives_empty_read():
    # Archive.get_ladder_rungs answers a date it cannot parse with an untyped empty frame.
    empty = pd.DataFrame(columns=["entity", "book", "line", "p_over", "observed_at"])

    inputs = book_stamp.book_inputs(empty, ["Ann"], np.array([24.5]), SCORED_AT)

    assert inputs["Book Class"].tolist() == ["C"]
    assert inputs["Book Count"].tolist() == [0]
    assert inputs["Book Observed At"].isna().all()


def test_stamp_prices_the_side_each_offer_took():
    ladder = _Ladder(
        {
            ("NHL", "shots", "2026-11-20"): _rungs(
                [("Ann", "dk", 2.5, 0.60, 10), ("Bob", "dk", 3.5, 0.45, 20)]
            )
        }
    )
    offers = _offers(
        [
            ("NHL", "shots", "2026-11-20", "Ann", 2.5, "Over"),
            ("NHL", "shots", "2026-11-20", "Ann", 2.5, "Under"),
            ("NHL", "shots", "2026-11-20", "Bob", 2.5, "Over"),
            ("NHL", "shots", "2026-11-20", "Cy", 2.5, "Under"),
        ],
        index=[7, 3, 9, 1],
    )

    stamp = book_stamp.stamp(offers, ladder, SCORED_AT)

    assert stamp.columns.tolist() == book_stamp.COLS
    assert stamp.index.tolist() == [7, 3, 9, 1]
    assert stamp["Book Prob"].tolist() == pytest.approx([0.60, 0.40, np.nan, np.nan], nan_ok=True)
    assert stamp["Book Count"].tolist() == [1, 1, 0, 0]
    assert stamp["Book Class"].tolist() == ["A", "A", "B", "C"]
    assert stamp["Book Observed At"].tolist()[:2] == [SCORED_AT - pd.Timedelta(minutes=10)] * 2
    assert stamp["Book Observed At"].isna().tolist() == [False, False, True, True]


def test_stamp_reads_each_slate_once_from_the_sportsbooks_alone():
    # The same player and line on two slates: a read that pooled them would price both alike.
    ladder = _Ladder(
        {
            ("NHL", "shots", "2026-11-20"): _rungs([("Ann", "dk", 2.5, 0.60, 10)]),
            ("NHL", "shots", "2026-11-21"): _rungs([("Ann", "dk", 2.5, 0.30, 10)]),
            ("NHL", "saves", "2026-11-20"): _rungs([("Ann", "dk", 2.5, 0.70, 10)]),
        }
    )
    offers = _offers(
        [
            ("NHL", "shots", "2026-11-21", "Ann", 2.5, "Over"),
            ("NHL", "saves", "2026-11-20", "Ann", 2.5, "Over"),
            ("NHL", "shots", "2026-11-20", "Ann", 2.5, "Over"),
            ("NHL", "shots", "2026-11-20", "Bob", 2.5, "Over"),
        ]
    )
    cut = SCORED_AT - pd.Timedelta(seconds=60)

    stamp = book_stamp.stamp(offers, ladder, SCORED_AT, at=cut)

    assert stamp["Book Prob"].tolist() == pytest.approx([0.30, 0.70, 0.60, np.nan], nan_ok=True)
    assert sorted(read["slate"] for read in ladder.reads) == sorted(ladder.slates)
    assert all(read["exclude_books"] == set(DFS_PLATFORM_BOOKS) for read in ladder.reads)
    assert all(read["at"] == cut for read in ladder.reads)


def _history(rows):
    """History rows of NHL shots at 2.5: ``(Date, Player, Scored At, served quote's age in s)``."""
    frame = pd.DataFrame(rows, columns=["Date", "Player", "Scored At", "quote_age_s"])
    age = pd.to_timedelta(frame.pop("quote_age_s"), unit="s")
    return frame.assign(
        League="NHL",
        Market="shots",
        Line=2.5,
        Bet="Over",
        **{"Quote Observed At": frame["Scored At"] - age},
    )


def test_fill_history_stamps_the_rows_that_lack_a_class():
    ladder = _Ladder({("NHL", "shots", "2026-11-20"): _rungs([("Ann", "dk", 2.5, 0.60, 10)])})
    history = _history(
        [("2026-11-20", "Ann", SCORED_AT, 900), ("2026-11-20", "Bob", SCORED_AT, 900)]
    )

    filled = book_stamp.fill_history(history, ladder)

    assert filled["Book Class"].tolist() == ["A", "C"]
    assert filled["Book Prob"].tolist() == pytest.approx([0.60, np.nan], nan_ok=True)
    assert filled["Book Count"].tolist() == [1, 0]
    assert filled["Book Observed At"].iat[0] == SCORED_AT - pd.Timedelta(minutes=10)
    assert filled["Player"].tolist() == ["Ann", "Bob"]


def test_fill_history_leaves_a_stamped_row_as_its_run_wrote_it():
    ladder = _Ladder(
        {
            ("NHL", "shots", "2026-11-19"): _rungs([("Ann", "dk", 2.5, 0.70, 10)]),
            ("NHL", "shots", "2026-11-20"): _rungs([("Ann", "dk", 2.5, 0.60, 10)]),
        }
    )
    history = _history(
        [("2026-11-19", "Ann", SCORED_AT, 900), ("2026-11-20", "Ann", SCORED_AT, 900)]
    ).assign(
        **{
            "Book Prob": [0.52, np.nan],
            "Book Count": [3, np.nan],
            "Book Class": ["A", None],
            "Book Observed At": [SCORED_AT - pd.Timedelta(minutes=2), pd.NaT],
        }
    )

    filled = book_stamp.fill_history(history, ladder)

    assert filled["Book Prob"].tolist() == pytest.approx([0.52, 0.60])
    assert filled["Book Count"].tolist() == [3, 1]
    assert filled["Book Observed At"].tolist() == [
        SCORED_AT - pd.Timedelta(minutes=2),
        SCORED_AT - pd.Timedelta(minutes=10),
    ]
    assert [read["slate"][2] for read in ladder.reads] == ["2026-11-20"]


def test_fill_history_reads_nothing_for_rows_it_cannot_place():
    ladder = _Ladder({})
    history = _history(
        [
            ("2026-10-08", "Ann", SCORED_AT, 900),
            ("2026-11-20", "Bob", pd.NaT, None),
        ]
    ).assign(**{"Book Class": [None, None]})

    filled = book_stamp.fill_history(history, ladder)

    assert ladder.reads == []
    assert filled["Book Class"].isna().all()


def test_fill_history_reads_the_archive_under_its_own_market_name():
    ladder = _Ladder({("NHL", "assists", "2026-11-20"): _rungs([("Ann", "dk", 2.5, 0.60, 10)])})
    history = _history([("2026-11-20", "Ann", SCORED_AT, 900)]).assign(Market="AST")

    filled = book_stamp.fill_history(history, ladder)

    assert filled["Book Class"].tolist() == ["A"]
    assert filled["Market"].tolist() == ["AST"]


def test_fill_history_cuts_each_run_at_what_it_could_read():
    early, late = SCORED_AT - pd.Timedelta(hours=1), SCORED_AT
    history = _history(
        [
            # A run whose own served quote is seconds old read the batch stamped in its minute.
            ("2026-11-20", "Ann", early, 900),
            ("2026-11-20", "Bob", early, 20),
            # No quote served within the minute: nothing shows the run read that batch.
            ("2026-11-21", "Ann", late, 900),
            ("2026-11-21", "Bob", late, None),
        ]
    )
    ladder = _Ladder({})

    book_stamp.fill_history(history, ladder)

    cuts = {read["slate"][2]: read["at"] for read in ladder.reads}
    assert cuts == {"2026-11-20": early, "2026-11-21": late - pd.Timedelta(seconds=60)}
