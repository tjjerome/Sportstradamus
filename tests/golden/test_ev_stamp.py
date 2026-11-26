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


def test_the_board_snapshot_lists_priced_legs_by_ev_then_the_rest_by_model_ev():
    """The fitted number orders the snapshot; the model's claim orders only the legs the
    fit did not price, after them."""
    scored = pd.DataFrame(
        {
            "Player": ["Ann", "Bob", "Cy", "Dee"],
            "EV": [0.90, np.nan, 0.95, np.nan],
            "Model EV": [1.40, 1.10, 0.80, 1.30],
        }
    )

    snapshot = persist._normalize_offers(scored)

    assert snapshot["Player"].tolist() == ["Cy", "Ann", "Dee", "Bob"]


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


def test_find_correlation_carries_the_stamp_and_builds_entries_from_it(monkeypatch):
    _stub_corr_sources(monkeypatch, dict.fromkeys(permutations(_WNBA_TOKENS, 2), 0.2))
    even_pick = {
        "Boost_Over": UNDERDOG_BOOST_BASELINE,
        "Boost_Under": UNDERDOG_BOOST_BASELINE,
        "Model Version": "v1",
    }
    records = [leg | even_pick for leg in _wnba_offers()]
    ladder = _Ladder(
        {("WNBA", "PTS", "2026-05-08"): _rungs([("Aja Wilson", "dk", 22.5, 0.60, 10)])}
    )
    stamp_records(records, "Underdog", SCORED_AT, ladder)
    stats = {"WNBA": _stats_for([leg["Player"] for leg in records], "MIN short", "USG_PCT short")}

    def search(slate):
        return find_correlation(slate, stats, "Underdog", contest_variant="power")

    offers, parlays = search(records)

    by_player = offers.set_index("Player")
    assert by_player["Book Class"].to_dict() == {
        "Aja Wilson": "A",
        **dict.fromkeys(set(by_player.index) - {"Aja Wilson"}, "C"),
    }
    assert by_player.at["Aja Wilson", "Book Observed At"] == SCORED_AT - pd.Timedelta(minutes=10)
    assert offers[["Fit Prob", "Fit Prob Low", "EV"]].notna().all(axis=None)
    assert len(offers) == len(records)
    # No fit table yet, so each even pick prices at the app's fair half: every leg sits under
    # break-even, and the game still builds its entries.
    assert offers["EV"].max() < 1
    assert not parlays.empty
    for leg in parlays["legs"].iloc[0]:
        stamped = by_player.loc[leg["player"]]
        assert (leg["fit_prob"], leg["ev"], leg["pick"]) == (
            stamped["Fit Prob"],
            stamped["EV"],
            stamped["Pick"],
        )

    # The search reads the fitted number. The legs pay and correlate alike, so the best
    # two-leg entry is each team's leg the fit likes most: turn the fitted chances around
    # and it is two other players. By the model's chances it would be Wilson and Jones both times.
    def best_pair(fit_probs):
        refit = [
            record | {"Fit Prob": fit_prob, "EV": fit_prob * record["Boost"]}
            for record, fit_prob in zip(records, fit_probs, strict=True)
        ]
        pairs = search(refit)[1].query("`Bet Size` == 2")
        return {leg["player"] for leg in pairs.loc[pairs["EV"].idxmax(), "legs"]}

    rising = np.linspace(0.45, 0.55, len(records))
    assert best_pair(rising) == {"Kelsey Plum", "Jonquel Jones"}
    assert best_pair(rising[::-1]) == {"Aja Wilson", "Sabrina Ionescu"}

    # A slate the fit did not price keeps its place on the board and builds no entry.
    unpriced = [record | dict.fromkeys(live_fit.SCORE_COLS, np.nan) for record in records]
    bare_offers, bare_parlays = search(unpriced)
    assert len(bare_offers) == len(records)
    assert bare_parlays.empty


def _model_led_fit(fit_path) -> None:
    """A fit table whose one stratum prices an NHL even pick no book quotes at the model's chance."""
    fitted = dict.fromkeys(live_fit.FIT_COLS, np.nan) | {
        "league": "NHL",
        "kind": "underdog_even",
        "book_class": "C",
        "c": 0.0,
        "w_model": 1.0,
        "cov": np.zeros(4).tolist(),
    }
    pd.DataFrame([fitted]).to_parquet(fit_path)


def _ranked(count: int, teams: int = 12) -> list[dict]:
    """``count`` even picks on one slate of ``teams`` teams, each its own player, highest EV first."""
    return [
        _record(f"P{rank}", "Over", win_prob=0.61 - rank / 100) | {"Team": f"T{rank % teams}"}
        for rank in range(count)
    ]


def _picks(records: list[dict]) -> list[bool]:
    stamp_records(records, "Underdog", SCORED_AT, _Ladder({}))
    return [record["Pick"] for record in records]


def test_a_slates_picks_are_its_players_with_the_highest_ev_ten_at_most(fit_path):
    _model_led_fit(fit_path)
    # Lowest first: a rule that took the first ten records would keep the wrong players.
    records = _ranked(12)[::-1]

    assert live_fit.PICKS_PER_SLATE == 10
    assert _picks(records) == [False, False] + [True] * 10
    # Plain flags, so the frame history is cut from types the column as a flag.
    assert pd.DataFrame(records)["Pick"].dtype == bool


def test_a_slate_of_fewer_than_ten_teams_has_one_place_a_team(fit_path):
    _model_led_fit(fit_path)
    # One game: two places, taken by the highest EV wherever it sits, here both on one team.
    one_game = _ranked(6, teams=2)
    one_game[1]["Team"] = "T0"

    assert _picks(one_game) == [True, True, False, False, False, False]


def test_a_combo_adds_its_players_teams_to_the_slate_and_no_team_of_its_own(fit_path):
    _model_led_fit(fit_path)
    records = [*_ranked(4, teams=2), _record("P0 + P1", "Over", win_prob=0.4) | {"Team": "T0/T1"}]

    assert _picks(records) == [True, True, False, False, False]


def test_a_player_holds_one_pick_at_the_offer_with_the_highest_ev(fit_path):
    _model_led_fit(fit_path)
    ann = {"Team": "T0"}
    records = [
        _record("Ann", "Over", line=2.5, win_prob=0.70) | ann,
        _record("Ann", "Over", line=3.5, win_prob=0.72) | ann,
        *_ranked(10),
    ]

    picks = _picks(records)

    assert picks[:2] == [False, True]
    # Ann holds one of the ten places, not two: nine of the others follow her.
    assert picks[2:] == [True] * 9 + [False]


def test_each_game_date_league_and_pick_kind_ranks_its_own_picks(fit_path):
    _model_led_fit(fit_path)
    # Each stands alone on its slate at a lower EV than the eleventh even pick, which is cut.
    alone = [
        _record("modified", "Under", boost_over=1.4, boost_under=2.2) | {"Team": "T0"},
        _record("next day", "Over", win_prob=0.45) | {"Team": "T0", "Date": "2026-11-21"},
        _record("other league", "Over") | {"Team": "T0", "League": "NBA"},
    ]
    records = [*_ranked(11), *alone]

    assert _picks(records) == [True] * 10 + [False] + [True] * 3
    assert max(record["EV"] for record in records[11:]) < records[10]["EV"]


def test_an_offer_the_fit_does_not_price_is_no_pick_on_a_slate_with_places_to_spare(fit_path):
    _model_led_fit(fit_path)
    unpriced = _record("one side posted", "Over", win_prob=0.9, boost_under=0.0) | {"Team": "T0"}

    assert _picks([unpriced, *_ranked(3)]) == [False, True, True, True]
