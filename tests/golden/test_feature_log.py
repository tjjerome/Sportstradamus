"""Serve-time feature log: upsert, retention, exact round trip, failure isolation.

``prediction/feature_log.py`` records what each model was fed, so train/serve parity can
be checked and a leg re-scored offline under another model file. The log sits on the
serving path, so the last tests drive the real ``model_prob``: every scored player is
logged, and a log that cannot be written costs no prediction.
"""

from __future__ import annotations

import datetime
import importlib
import logging
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sportstradamus.helpers.training_quotes import ArchivedBookQuote, TrainingQuote
from sportstradamus.prediction import book_quotes, feature_log, offer_records
from sportstradamus.prediction.feature_log import (
    FEATURE_LOG_KEY,
    FEATURE_LOG_RETENTION_DAYS,
    prune_feature_log,
    upsert_feature_log,
)

# The package __init__ re-exports the model_prob *function*, shadowing the submodule.
mp = importlib.import_module("sportstradamus.prediction.model_prob")

_DATE = "2026-10-04"
_OBSERVED_AT = datetime.datetime(2026, 10, 4, 12, 0, 0)


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    path = tmp_path / "feature_log"
    monkeypatch.setattr(feature_log, "FEATURE_LOG_DIR", path)
    return path


def _features(avg5: dict[str, float]) -> pd.DataFrame:
    return pd.Series(avg5, name="Avg5").to_frame()


def _offers(players, date=_DATE, team="LAL") -> list[dict]:
    return [{"Player": player, "Date": date, "Team": team} for player in players]


def _upsert(
    stats,
    *,
    league="NBA",
    market="PTS",
    platform="Underdog",
    offers=None,
    stat_data=None,
    quotes=None,
) -> None:
    upsert_feature_log(
        league,
        market,
        platform,
        _offers(stats.index) if offers is None else offers,
        stat_data,
        stats,
        pd.DataFrame({"Projection": 1.0}, index=stats.index),
        [None] * len(stats) if quotes is None else quotes,
        model_version="20260901.none.abc123",
        step=1.0,
        model_weight=0.6,
        hist_gate=0.0,
    )


def _read(log_dir, slug, date=_DATE) -> pd.DataFrame:
    return pd.read_parquet(log_dir / f"date={date}" / f"{slug}.parquet")


def test_rescoring_replaces_the_row_and_keeps_every_other_row(log_dir):
    _upsert(_features({"A": 1.0, "B": 2.0}))
    _upsert(_features({"A": 9.0}))
    _upsert(_features({"A": 5.0}), platform="Sleeper")

    logged = _read(log_dir, "NBA_PTS")

    assert not logged.duplicated(FEATURE_LOG_KEY).any()
    assert logged.set_index(["Platform", "Player"])["Avg5"].to_dict() == {
        ("Underdog", "A"): 9.0,
        ("Underdog", "B"): 2.0,
        ("Sleeper", "A"): 5.0,
    }


def test_each_market_and_game_date_gets_its_own_file(log_dir):
    offers = [*_offers(["A"]), *_offers(["B"], date="2026-10-05")]
    _upsert(_features({"A": 1.0, "B": 2.0}), offers=offers)
    _upsert(_features({"A": 3.0}), league="NFL", market="rushing yards")

    assert sorted(str(path.relative_to(log_dir)) for path in log_dir.rglob("*.parquet")) == [
        "date=2026-10-04/NBA_PTS.parquet",
        "date=2026-10-04/NFL_rushing-yards.parquet",
        "date=2026-10-05/NBA_PTS.parquet",
    ]
    assert _read(log_dir, "NBA_PTS")["Player"].tolist() == ["A"]
    assert _read(log_dir, "NBA_PTS", date="2026-10-05")["Player"].tolist() == ["B"]
    assert _read(log_dir, "NFL_rushing-yards")["Market"].tolist() == ["rushing yards"]


def test_player_without_a_dated_offer_is_not_logged(log_dir):
    # "B" is a combo leg's component, fed to the model with no offer of its own; "C" is
    # an offer whose game the platform has not dated.
    offers = [*_offers(["A"]), *_offers(["C"], date=None)]
    _upsert(_features({"A": 1.0, "B": 2.0, "C": 3.0}), offers=offers)

    assert _read(log_dir, "NBA_PTS")["Player"].tolist() == ["A"]
    assert [path.name for path in log_dir.iterdir()] == [f"date={_DATE}"]


def test_fed_values_round_trip_exactly(log_dir):
    stats = pd.DataFrame(
        {
            "Float": [np.nextafter(1.0, 2.0), 1 / 3, 5e-324, -0.0],
            # Past 2**53 a float64 detour would round it.
            "Int": np.array([2**53 + 1, -7, 0, 3], dtype="int64"),
            "Home": pd.Categorical([True, False, False, True]),
            "Player position": pd.Categorical([1.0, 3.0, 3.0, 2.0]),
        },
        index=list("ABCD"),
    )
    fed = stats.copy()
    path = log_dir / f"date={_DATE}" / "NBA_PTS.parquet"

    _upsert(stats)
    first_write = pq.read_schema(path)
    # A rescoring of one player sends the other rows through read, concat and rewrite.
    _upsert(stats.loc[["A"]])

    # Categoricals land as plain typed columns, so a first write and an upsert agree and
    # a reader can open many partitions as one table.
    assert first_write.field("Home").type == pa.bool_()
    assert first_write.field("Player position").type == pa.float64()
    assert pq.read_schema(path).equals(first_write)
    logged = _read(log_dir, "NBA_PTS").set_index("Player").loc[fed.index]
    assert logged["Float"].to_numpy().tobytes() == fed["Float"].to_numpy().tobytes()
    assert logged["Int"].dtype == fed["Int"].dtype
    assert logged["Int"].tolist() == fed["Int"].tolist()
    for column in ("Home", "Player position"):
        # The values are what LightGBM matches on; recast, they are the fed categorical.
        pd.testing.assert_series_equal(
            logged[column].astype("category"), fed[column], check_names=False
        )
    pd.testing.assert_frame_equal(stats, fed)


def test_book_leg_columns_follow_each_rows_quote(log_dir):
    quote = TrainingQuote(
        line=1.5,
        over_probability=0.55,
        ev=1.6,
        source="book_direct",
        authenticity="authentic",
        synthetic_reason=None,
        observed_at=_OBSERVED_AT,
        book_count=2,
    )
    _upsert(_features({"A": 1.0, "B": 2.0}), quotes=[quote, None])

    logged = _read(log_dir, "NBA_PTS").set_index("Player")[["Quote Over Prob", "Quote EV"]]
    assert logged.loc["A"].tolist() == [0.55, 1.6]
    assert logged.loc["B"].isna().all()


def test_mlb_rows_carry_the_probable_pitcher_of_the_legs_team(log_dir):
    offers = [*_offers(["A"], team="NYY"), *_offers(["B"], team="SEA")]
    upcoming = SimpleNamespace(upcoming_games={"NYY": {"Opponent Pitcher": "Logan Gilbert"}})
    _upsert(
        _features({"A": 1.0, "B": 2.0}),
        league="MLB",
        market="hits",
        offers=offers,
        stat_data=upcoming,
    )
    _upsert(_features({"A": 1.0}))

    pitchers = _read(log_dir, "MLB_hits").set_index("Player")["Opponent Pitcher"]
    assert pitchers.to_dict() == {"A": "Logan Gilbert", "B": None}
    assert "Opponent Pitcher" not in _read(log_dir, "NBA_PTS")


def test_prune_removes_only_partitions_past_retention(log_dir):
    # A run that scored no model market has no log directory to prune yet.
    prune_feature_log()

    today = pd.Timestamp.today().date()
    partitions = {
        age: log_dir / f"date={today - pd.Timedelta(days=age)}"
        for age in (0, FEATURE_LOG_RETENTION_DAYS, FEATURE_LOG_RETENTION_DAYS + 1)
    }
    for partition in partitions.values():
        partition.mkdir(parents=True)
        (partition / "NBA_PTS.parquet").touch()

    prune_feature_log()

    assert sorted(path.name for path in log_dir.iterdir()) == sorted(
        partitions[age].name for age in (0, FEATURE_LOG_RETENTION_DAYS)
    )


_LEAGUE, _MARKET, _PLATFORM = "NBA", "BLK", "Underdog"
_PLAYERS = ["Player A", "Player B", "Player C"]
_LINE = 1.5
_BOOK_UNDER = 0.55
_MODEL_R = 3.0
_MODEL_MEAN = np.array([0.8, 1.6, 2.4])
# Only the keys model_prob reads: the booster is stood in for by _predict.
_FILEDICT = {
    "cv": 1.0,
    "weight": 0.6,
    "temperature": 1.0,
    "dispersion_cal": 1.0,
    "skew_cal": 0.0,
    "shape_ceiling": None,
    "distribution": "NegBin",
    "step": 1,
    "normalized": False,
    "offset_meta": None,
    "target_normalization": "ratio_meanyr",
    "posthoc": "none",
    "posthoc_blob": None,
    "pit_recal_blob": None,
    "model_version": "20260901.none.abc123",
}


class _StubArchive:
    """The two archive surfaces model_prob reads: the book cohort and the league totals."""

    default_totals = {_LEAGUE: 220.0}

    def get_training_quote_inputs(self, league, market, date, entities, at=None):
        rows = [
            ArchivedBookQuote(book, None, _BOOK_UNDER, _LINE, _OBSERVED_AT)
            for book in ("fanduel", "draftkings")
        ]
        return dict.fromkeys(entities, (rows, _LINE))


def _predict(_filedict, _market, _stat_data, player_stats, *_args) -> pd.DataFrame:
    """Stand in for the booster, keeping ``_build_prob_params``' in-place category cast."""
    for column in ("Home", "Player position"):
        player_stats[column] = player_stats[column].astype("category")
    return pd.DataFrame(
        {"total_count": _MODEL_R, "probs": _MODEL_MEAN / (_MODEL_R + _MODEL_MEAN)},
        index=player_stats.index,
    )


def _score(monkeypatch, tmp_path) -> pd.DataFrame:
    """Serve three NBA BLK offers through the real ``model_prob``; return its records."""
    pickle_path = tmp_path / "NBA_BLK.mdl"
    with open(pickle_path, "wb") as outfile:
        pickle.dump(_FILEDICT, outfile)
    stub_archive = _StubArchive()
    monkeypatch.setattr(mp, "model_pickle_path", lambda _league, _market: str(pickle_path))
    monkeypatch.setattr(mp, "stat_meta", {})
    monkeypatch.setattr(mp, "stat_zi", {})
    monkeypatch.setattr(mp, "_build_prob_params", _predict)
    monkeypatch.setattr(book_quotes, "archive", stub_archive)
    monkeypatch.setattr(offer_records, "archive", stub_archive)
    offers = [
        {
            "Player": player,
            "League": _LEAGUE,
            "Team": "LAL",
            "Opponent": "BOS",
            "Date": _DATE,
            "Market": _MARKET,
            "Line": _LINE,
            "Boost": 1.0,
            "Boost_Over": 1.0,
            "Boost_Under": 1.0,
        }
        for player in _PLAYERS
    ]
    player_stats = pd.DataFrame(
        {
            "Avg5": [1.0, 2.0, 3.0],
            "MeanYr": [0.9, 1.7, 2.2],
            "Home": [True, False, True],
            "Player position": [5.0, 4.0, 5.0],
            "Defense avg": [0.1, -0.2, 0.3],
        },
        index=_PLAYERS,
    )
    records = mp.model_prob(
        offers, _LEAGUE, _MARKET, _PLATFORM, SimpleNamespace(league=_LEAGUE), player_stats
    )
    return pd.DataFrame(records).sort_values("Player", ignore_index=True)


def test_model_prob_logs_each_scored_player(log_dir, monkeypatch, tmp_path):
    served = _score(monkeypatch, tmp_path)

    logged = _read(log_dir, "NBA_BLK")
    assert logged[FEATURE_LOG_KEY].to_dict("list") == {
        "League": [_LEAGUE] * 3,
        "Market": [_MARKET] * 3,
        "Platform": [_PLATFORM] * 3,
        "Player": _PLAYERS,
        "Date": [_DATE] * 3,
    }
    assert logged["Scored At"].notna().all()
    assert logged[["Model Version", "Step", "Model Weight", "Hist Gate"]].drop_duplicates().to_dict(
        "records"
    ) == [
        {
            "Model Version": _FILEDICT["model_version"],
            "Step": _FILEDICT["step"],
            "Model Weight": _FILEDICT["weight"],
            "Hist Gate": 0,
        }
    ]
    # What the model was fed, categoricals as their values.
    assert logged["Avg5"].tolist() == [1.0, 2.0, 3.0]
    assert logged["Home"].tolist() == [True, False, True]
    assert logged["Player position"].tolist() == [5.0, 4.0, 5.0]
    # Its own outputs, ahead of the book blend that moves the served Projection.
    np.testing.assert_allclose(logged["total_count"], _MODEL_R)
    np.testing.assert_allclose(logged["Projection"], _MODEL_MEAN)
    assert not np.allclose(served["Projection"], _MODEL_MEAN)
    # The book leg it was blended with, and the quote that leg was inverted from.
    assert logged["Market Projection"].notna().all()
    assert logged["Quote Line"].tolist() == [_LINE] * 3
    assert logged["Quote Books"].tolist() == [2.0] * 3
    assert logged["Quote Observed At"].tolist() == [pd.Timestamp(_OBSERVED_AT)] * 3
    np.testing.assert_allclose(logged["Quote Over Prob"], 1 - _BOOK_UNDER)
    assert served["Player"].tolist() == _PLAYERS


@pytest.mark.parametrize(
    "error", [OSError("disk full"), pa.ArrowInvalid("mixed column"), KeyError("Date")]
)
def test_failed_log_write_costs_no_prediction(error, log_dir, monkeypatch, tmp_path, caplog):
    served = _score(monkeypatch, tmp_path)

    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(feature_log, "_atomic_write_parquet", fail)
    with caplog.at_level(logging.ERROR, logger="log"):
        served_without_log = _score(monkeypatch, tmp_path)

    assert "NBA_BLK feature log not written" in caplog.text
    pd.testing.assert_frame_equal(served_without_log, served)
