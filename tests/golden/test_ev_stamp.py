"""The EV stamp on scored records: the books' price at each offer's own line, then the fit's EV."""

from itertools import permutations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from sportstradamus import book_stamp, live_fit
from sportstradamus.helpers import UNDERDOG_BOOST_BASELINE
from sportstradamus.prediction import ev_stamp, persist, scoring
from sportstradamus.prediction.correlation import find_correlation
from sportstradamus.prediction.ev_stamp import EV_STAMP_COLS, stamp_records
from sportstradamus.prediction.offer_records import SCORED_RECORD_COLS
from tests.golden.test_correlation_helpers import (
    _WNBA_TOKENS,
    _stats_for,
    _stub_corr_sources,
    _wnba_offers,
)

SCORED_AT = pd.Timestamp("2026-11-20 23:50:00")
DAY = "2026-11-20"


@pytest.fixture(autouse=True)
def fit_path(tmp_path, monkeypatch):
    """Where the stamp looks for the nightly fit table: absent until a test writes one."""
    path = tmp_path / "live_fit.parquet"
    monkeypatch.setattr(ev_stamp, "LIVE_FIT_PATH", path)
    return path


def _rungs(rows):
    """Ladder rungs from ``(entity, book, line, p_over, minutes old at SCORED_AT)``."""
    frame = pd.DataFrame(rows, columns=["entity", "book", "line", "p_over", "minutes_old"])
    frame["observed_at"] = SCORED_AT - pd.to_timedelta(frame.pop("minutes_old"), unit="min")
    return frame


class _Ladder:
    """Stands in for ``Archive.get_ladder_rungs``: each slate's rungs, and the slates read."""

    def __init__(self, slates):
        self.slates, self.reads = slates, []

    def get_ladder_rungs(self, league, market, date, entities, *, exclude_books=(), at=None):
        self.reads.append((league, market, date))
        rungs = self.slates.get((league, market, date), _rungs([]))
        return rungs[rungs["entity"].isin(entities)]


def _record(
    player,
    bet,
    *,
    market="Shots",
    line=2.5,
    boost_over=UNDERDOG_BOOST_BASELINE,
    boost_under=UNDERDOG_BOOST_BASELINE,
    win_prob=0.6,
) -> dict:
    """A scored NHL record under Underdog's market label, an even pick unless a payout is given."""
    payout = boost_over if bet == "Over" else boost_under
    return dict.fromkeys(SCORED_RECORD_COLS) | {
        "League": "NHL",
        "Date": DAY,
        "Player": player,
        "Market": market,
        "Line": line,
        "Bet": bet,
        "Boost": payout,
        "Boost_Over": boost_over,
        "Boost_Under": boost_under,
        "Win Prob": win_prob,
        "Model EV": win_prob * payout,
        "Kelly": (win_prob * payout - 1) / (payout - 1),
        "Model Version": "v1",
    }


def test_each_record_takes_the_books_price_of_its_own_offer_and_side():
    ladder = _Ladder(
        {
            ("NHL", "shots", DAY): _rungs(
                [
                    ("Ann", "dk", 2.5, 0.60, 10),
                    ("Ann", "fd", 2.5, 0.56, 30),
                    ("Bob", "dk", 3.5, 0.45, 20),
                ]
            ),
            ("NHL", "saves", DAY): _rungs([("Cy", "dk", 27.5, 0.30, 5)]),
        }
    )
    # Listed apart from slate order: a stamp laid on slate by slate would land on other records.
    records = [
        _record("Cy", "Under", market="Saves", line=27.5),
        _record("Ann", "Over"),
        _record("Dee", "Over"),
        _record("Ann", "Under"),
        _record("Bob", "Over"),
    ]

    stamp_records(records, "Underdog", SCORED_AT, ladder)

    stamped = pd.DataFrame(records)
    assert stamped["Book Prob"].tolist() == pytest.approx(
        [0.70, 0.58, np.nan, 0.42, np.nan], nan_ok=True
    )
    assert stamped["Book Count"].tolist() == [1, 2, 0, 2, 0]
    assert stamped["Book Class"].tolist() == ["A", "A", "C", "A", "B"]
    newest = SCORED_AT - pd.to_timedelta([5, 10, 10], unit="min")
    assert stamped["Book Observed At"].dropna().tolist() == newest.tolist()
    assert stamped["Book Observed At"].isna().tolist() == [False, False, True, False, True]
    # The records hold plain values, so the frame history is cut from types them as the stamp did.
    assert stamped[["Book Count", "Book Observed At"]].dtypes.astype(str).tolist() == [
        "int64",
        "datetime64[ns]",
    ]


# A label on each platform that the NHL's archive files under another name: Underdog's Assists
# is AST on the board and assists in the archive. Neither is a label on the other platform.
_NHL_LABELS = pytest.mark.parametrize(
    ("platform", "label", "archive_name"),
    [("Underdog", "Assists", "assists"), ("Sleeper", "blocked_shots", "blocked")],
)


@_NHL_LABELS
def test_the_archive_is_read_under_its_own_name_for_the_market(platform, label, archive_name):
    ladder = _Ladder({("NHL", archive_name, DAY): _rungs([("Ann", "dk", 2.5, 0.60, 10)])})
    records = [_record("Ann", "Over", market=label)]

    stamp_records(records, platform, SCORED_AT, ladder)

    assert ladder.reads == [("NHL", archive_name, DAY)]
    assert records[0]["Book Class"] == "A"


def test_a_quotes_age_is_measured_from_the_runs_stamp():
    # SCORED_AT is long past: aged against the clock, the fresh quote would be stale too.
    cut = book_stamp.BOOK_MAX_AGE / pd.Timedelta(minutes=1)
    ladder = _Ladder(
        {
            ("NHL", "shots", DAY): _rungs(
                [("Ann", "dk", 2.5, 0.60, cut - 1), ("Bob", "dk", 2.5, 0.60, cut + 1)]
            )
        }
    )
    records = [_record("Ann", "Over"), _record("Bob", "Over")]

    stamp_records(records, "Underdog", SCORED_AT, ladder)

    assert [record["Book Class"] for record in records] == ["A", "C"]


def test_before_the_first_fit_table_an_offer_prices_at_the_apps_fair_price():
    records = [_record("Ann", "Over"), _record("Bob", "Under", boost_over=1.5, boost_under=2.4)]

    stamp_records(records, "Underdog", SCORED_AT, _Ladder({}))

    even, modified = records
    assert even["Fit Prob"] == pytest.approx(0.5)
    assert even["Fit Prob Low"] == pytest.approx(0.5)
    assert even["EV"] == pytest.approx(0.5 * UNDERDOG_BOOST_BASELINE)
    # The cut removed from the posted pair, on the Under the record took.
    fair_under = (1 / 2.4) / (1 / 1.5 + 1 / 2.4)
    assert modified["Fit Prob"] == pytest.approx(fair_under)
    assert modified["EV"] == pytest.approx(fair_under * 2.4)
    assert even["EV"] < 1 and modified["EV"] < 1


def test_the_fit_values_are_the_nightly_tables_price_of_each_stamped_offer(fit_path):
    # One fitted stratum, NHL even picks a book prices at the line: c, w_book and w_model.
    fitted = dict.fromkeys(live_fit.FIT_COLS, np.nan) | {
        "league": "NHL",
        "kind": "underdog_even",
        "book_class": "A",
        "c": 0.05,
        "w_book": 0.9,
        "w_model": 0.2,
        "cov": (0.01 * np.eye(3)).ravel().tolist(),
    }
    pd.DataFrame([fitted]).to_parquet(fit_path)
    ladder = _Ladder({("NHL", "shots", DAY): _rungs([("Ann", "dk", 2.5, 0.60, 10)])})
    records = [
        _record("Ann", "Over", win_prob=0.62),
        _record("Ann", "Under", win_prob=0.57),
        _record("Dee", "Over", win_prob=0.55),
    ]

    stamp_records(records, "Underdog", SCORED_AT, ladder)

    offers = pd.DataFrame(
        {
            "League": "NHL",
            "Platform": "Underdog",
            "Payout Over": UNDERDOG_BOOST_BASELINE,
            "Payout Under": UNDERDOG_BOOST_BASELINE,
            "Bet": ["Over", "Under", "Over"],
            "Win Prob": [0.62, 0.57, 0.55],
            "Book Prob": [0.60, 0.40, np.nan],
            "Model Version": "v1",
            "Book Class": ["A", "A", "C"],
        }
    )
    priced = live_fit.score(offers, pd.read_parquet(fit_path))
    pd.testing.assert_frame_equal(pd.DataFrame(records)[priced.columns], priced)
    # The table holds a stratum for the two legs a book prices, and moves only those.
    assert np.isclose(priced["Fit Prob"], 0.5).tolist() == [False, False, True]


def test_an_offer_is_priced_in_its_own_platforms_stratum_of_the_table(fit_path):
    # One fitted stratum, Sleeper picks no book quotes: an intercept and nothing else.
    intercept = 0.05
    fitted = dict.fromkeys(live_fit.FIT_COLS, np.nan) | {
        "league": "NHL",
        "kind": "sleeper",
        "book_class": "C",
        "c": intercept,
        "cov": [0.01],
    }
    pd.DataFrame([fitted]).to_parquet(fit_path)
    on_sleeper = [_record("Ann", "Over", market="blocked_shots")]
    on_underdog = [_record("Ann", "Over")]

    stamp_records(on_sleeper, "Sleeper", SCORED_AT, _Ladder({}))
    stamp_records(on_underdog, "Underdog", SCORED_AT, _Ladder({}))

    # An even pick's fair logit is zero, so the fitted chance is the intercept's alone.
    assert on_sleeper[0]["Fit Prob"] == pytest.approx(expit(intercept))
    assert on_underdog[0]["Fit Prob"] == pytest.approx(0.5)


def test_a_record_with_one_side_posted_gets_its_book_stamp_and_no_fit():
    ladder = _Ladder({("NHL", "shots", DAY): _rungs([("Ann", "dk", 2.5, 0.60, 10)])})
    records = [_record("Ann", "Over", boost_under=0.0)]

    stamp_records(records, "Underdog", SCORED_AT, ladder)

    (record,) = records
    assert (record["Book Class"], record["Book Count"]) == ("A", 1)
    assert record["Book Prob"] == pytest.approx(0.60)
    assert np.isnan([record["Fit Prob"], record["Fit Prob Low"], record["EV"]]).all()


def test_the_stamp_adds_its_columns_and_changes_no_key_a_record_already_had():
    ladder = _Ladder({("NHL", "shots", DAY): _rungs([("Ann", "dk", 2.5, 0.60, 10)])})
    records = [_record("Ann", "Over"), _record("Bob", "Under", boost_over=1.5, boost_under=2.4)]
    before = [dict(record) for record in records]

    stamp_records(records, "Underdog", SCORED_AT, ladder)

    for record, was in zip(records, before, strict=True):
        assert list(record) == [*was, *EV_STAMP_COLS]
        assert {key: record[key] for key in was} == was


def test_the_board_snapshot_keeps_the_stamp():
    ladder = _Ladder({("NHL", "shots", DAY): _rungs([("Ann", "dk", 2.5, 0.60, 10)])})
    records = [_record("Ann", "Over"), _record("Dee", "Under", win_prob=0.55)]
    stamp_records(records, "Underdog", SCORED_AT, ladder)
    scored = pd.DataFrame(records)

    snapshot = persist._normalize_offers(scored)

    assert snapshot["Player"].tolist() == ["Ann", "Dee"]
    pd.testing.assert_frame_equal(snapshot[EV_STAMP_COLS], scored[EV_STAMP_COLS])


@_NHL_LABELS
def test_process_offers_stamps_the_records_before_find_correlation_sees_them(
    monkeypatch, platform, label, archive_name
):
    ladder = _Ladder({("NHL", archive_name, DAY): _rungs([("Ann", "dk", 2.5, 0.60, 10)])})
    monkeypatch.setattr(scoring, "archive", ladder)
    scored = [_record("Ann", "Under", market=label)]
    monkeypatch.setattr(scoring, "_match_league_offers", lambda *league_pass: scored)
    seen = []

    def find_correlation(offers, stats, book, **sinks):
        seen.extend(dict(offer) for offer in offers)
        return pd.DataFrame(offers), pd.DataFrame()

    monkeypatch.setattr(scoring, "find_correlation", find_correlation)

    scoring.process_offers({"NHL": {label: [{}]}}, platform, {}, scored_at=SCORED_AT)

    (record,) = seen
    assert list(record)[-len(EV_STAMP_COLS) :] == EV_STAMP_COLS
    # The run's own stamp aged the quote, and the platform named the market.
    assert (record["Book Class"], record["Book Prob"]) == ("A", pytest.approx(0.40))


def test_process_offers_refuses_a_call_without_the_runs_stamp():
    # No default: a caller that left it out would age every quote against some other clock.
    with pytest.raises(TypeError, match="scored_at"):
        scoring.process_offers({}, "Underdog", {})


def test_process_offers_leaves_a_platform_with_no_records_alone(monkeypatch):
    ladder = _Ladder({})
    monkeypatch.setattr(scoring, "archive", ladder)

    offers, parlays = scoring.process_offers({}, "Underdog", {}, scored_at=SCORED_AT)

    assert ladder.reads == []
    assert offers.empty and parlays.empty
    # The empty frame still names the stamp's columns, as it does every scored record's.
    assert set(EV_STAMP_COLS) <= set(offers.columns)


def test_find_correlation_carries_the_stamp_and_reads_none_of_it(monkeypatch):
    _stub_corr_sources(monkeypatch, dict.fromkeys(permutations(_WNBA_TOKENS, 2), 0.2))
    even_pick = {
        "Boost_Over": UNDERDOG_BOOST_BASELINE,
        "Boost_Under": UNDERDOG_BOOST_BASELINE,
        "Model Version": "v1",
    }
    records = [leg | even_pick for leg in _wnba_offers()]
    unstamped = [dict(record) for record in records]
    ladder = _Ladder(
        {("WNBA", "PTS", "2026-05-08"): _rungs([("Aja Wilson", "dk", 22.5, 0.60, 10)])}
    )
    stamp_records(records, "Underdog", SCORED_AT, ladder)
    stats = {"WNBA": _stats_for([leg["Player"] for leg in records], "MIN short", "USG_PCT short")}

    offers, parlays = find_correlation(records, stats, "Underdog", contest_variant="power")

    assert not parlays.empty
    by_player = offers.set_index("Player")
    assert by_player["Book Class"].to_dict() == {
        "Aja Wilson": "A",
        **dict.fromkeys(set(by_player.index) - {"Aja Wilson"}, "C"),
    }
    assert by_player.at["Aja Wilson", "Book Observed At"] == SCORED_AT - pd.Timedelta(minutes=10)
    assert offers[["Fit Prob", "Fit Prob Low", "EV"]].notna().all(axis=None)
    assert len(offers) == len(records)
    # Nothing reads the stamp yet: the slate ranks and builds its parlays as it does unstamped.
    bare_offers, bare_parlays = find_correlation(
        unstamped, stats, "Underdog", contest_variant="power"
    )
    pd.testing.assert_frame_equal(offers.drop(columns=EV_STAMP_COLS), bare_offers)
    # scipy integrates the joint chance of three or more legs by sampling, so such a parlay's
    # price repeats to about four digits from one search to the next.
    pd.testing.assert_frame_equal(parlays, bare_parlays, rtol=1e-3)
