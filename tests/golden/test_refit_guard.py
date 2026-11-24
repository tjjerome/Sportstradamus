"""The in-sample guard: an offline scorer never scores a row a refit pickle's booster trained on.

A refit pickle's booster trained on every row through its ``refit_through``, so each scorer that
re-serves rows through a pickle drops the rows dated on or before it (``out_of_sample``), says how
many per cell, and refuses a cell left with none. The paths: ``price_dumps`` (``ship priced``'s P2,
the realized replay's posted scope, the tail scorecard), ``ship priced``'s P1 (``_served_cells``)
and the feature-parity re-scores, through a file's own rows and through another version's file. The
rows a refit pickle keeps read exactly what a pickle without the key reads on them, and a pickle
without the key keeps every row, silently.

Each fixture puts an in-sample row last and dates one on ``refit_through`` itself, so a guard that
reads only the first rows, or keeps the day of the refit, fails.
"""

from __future__ import annotations

import importlib
import pickle
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from sportstradamus.helpers.training_quotes import AUTHENTIC
from sportstradamus.prediction import feature_log
from sportstradamus.prediction.feature_log import upsert_feature_log
from sportstradamus.scripts import feature_parity
from sportstradamus.scripts import priced_score as ps
from sportstradamus.scripts.tail_pricing import (
    CellExcludedError,
    _training_weight,
    out_of_sample,
    price_dumps,
    serve,
)

# The package __init__ re-exports the model_prob *function*, shadowing the submodule.
mp = importlib.import_module("sportstradamus.prediction.model_prob")

_CELL = "NBA_PTS"
_START = "2026-11-01"
_REFIT = "2026-11-02"
_DROPPED = f"{_CELL}: dropped 2 of 4 rows on or before refit_through {_REFIT}"
_ALL_IN_SAMPLE = (
    f"every {_CELL} row (2026-11-01 to 2026-11-04) is on or before refit_through 2026-11-04, "
    "in the refit's training rows"
)
# A NegBin cell's serving knobs.
_KNOBS = {
    "distribution": "NegBin",
    "weight": 0.6,
    "cv": 0.8,
    "dispersion_cal": 1.1,
    "skew_cal": 0.0,
    "hist_gate": 0.0,
    "step": 1.0,
    "temperature": 1.3,
    "posthoc": "none",
    "posthoc_blob": None,
    "pit_recal_blob": None,
}
# Four test rows; d, the last, is dated on _REFIT and b before it.
_ROWS = pd.DataFrame(
    {
        "Player": ["a", "b", "c", "d"],
        "Date": ["2026-11-03", "2026-11-01", "2026-11-04", _REFIT],
        "Result": [3.0, 1.0, 5.0, 2.0],
        "Line": [2.5, 1.5, 3.5, 2.5],
        "EV": [2.8, 1.4, 3.9, 2.2],
        "Book_EV": [2.6, 1.5, 3.4, 2.4],
        "Odds": [0.45, 0.5, 0.55, 0.5],
        "QuoteAuthenticity": [AUTHENTIC, "pickem", "pickem", AUTHENTIC],
        "StructuralStrategy": "none",
        "R_model": [3.0, 2.0, 4.0, 2.5],
        "P_standalone": 0.5,
    }
)
_LADDER_SCHEMA = (
    "league VARCHAR, market VARCHAR, game_date DATE, entity VARCHAR, book VARCHAR, line DOUBLE, "
    "p_over DOUBLE, observed_at TIMESTAMP"
)
_ODDS_SCHEMA = (
    "league VARCHAR, market VARCHAR, game_date DATE, entity VARCHAR, book VARCHAR, ev DOUBLE, "
    "observed_at TIMESTAMP, under_prob DOUBLE, line DOUBLE"
)


def _cell(root: Path, **keys) -> Path:
    """The cell's dump and pickle in ``root``, the dump's ``P`` the pickle's own (load_cell)."""
    root.mkdir()
    knobs = _KNOBS | keys
    with (root / f"{_CELL}.mdl").open("wb") as outfile:
        pickle.dump(knobs | {"model_version": "fixture"}, outfile)
    p_over = serve(_ROWS, knobs, _training_weight(_ROWS, knobs), _ROWS["Book_EV"].to_numpy(float))
    _ROWS.assign(P=p_over[0]).to_csv(root / f"{_CELL}.csv", index=False)
    return root


@pytest.fixture
def archive(tmp_path) -> duckdb.DuckDBPyConnection:
    """A scratch archive, read-only, whose ladder holds two Underdog rungs for every test row."""
    rungs = pd.concat([_ROWS.assign(rung=_ROWS["Line"] + step) for step in (0.0, 1.0)])
    ladder = pd.DataFrame(
        {
            "league": "NBA",
            "market": "PTS",
            "game_date": pd.to_datetime(rungs["Date"]),
            "entity": rungs["Player"],
            "book": "Underdog",
            "line": rungs["rung"],
            "p_over": 0.5,
            "observed_at": pd.to_datetime(rungs["Date"]) + pd.Timedelta(hours=18),
        }
    )
    path = tmp_path / "archive.duckdb"
    with duckdb.connect(str(path)) as writer:
        writer.execute(f"CREATE TABLE ladder ({_LADDER_SCHEMA})")
        writer.execute(f"CREATE TABLE odds ({_ODDS_SCHEMA})")
        writer.register("rungs", ladder)
        writer.execute(
            "INSERT INTO ladder SELECT league, market, game_date::DATE, entity, book, line, "
            "p_over, observed_at FROM rungs"
        )
    with duckdb.connect(str(path), read_only=True) as con:
        yield con


def _priced(con: duckdb.DuckDBPyConnection, root: Path) -> tuple:
    return price_dumps([root / f"{_CELL}.csv"], con, _START, root)


def test_rows_on_or_before_refit_through_are_dropped_and_counted(capsys):
    kept = out_of_sample(_ROWS, _REFIT, _CELL)

    pd.testing.assert_frame_equal(kept, _ROWS.loc[[0, 2]])
    assert _DROPPED in capsys.readouterr().out


def test_a_pickle_without_refit_through_keeps_every_row_silently(capsys):
    assert out_of_sample(_ROWS, None, _CELL) is _ROWS
    # An empty window is left to the caller's own exclusion, as without the key.
    assert out_of_sample(_ROWS.head(0), _REFIT, _CELL).empty
    assert capsys.readouterr().out == ""


def test_a_cell_with_no_row_left_raises_naming_the_cell_refit_and_window():
    with pytest.raises(CellExcludedError) as raised:
        out_of_sample(_ROWS, "2026-11-04", _CELL)

    assert str(raised.value) == _ALL_IN_SAMPLE


def test_price_dumps_prices_only_the_rows_after_refit_through(tmp_path, archive, capsys):
    refit_rungs, refit_knobs, refit_cells, _, excluded = _priced(
        archive, _cell(tmp_path / "refit", refit_through=_REFIT)
    )
    rungs, knobs, cells, _, _ = _priced(archive, _cell(tmp_path / "plain"))

    assert _DROPPED in capsys.readouterr().out
    assert not excluded
    assert (refit_cells[0]["test_rows"], cells[0]["test_rows"]) == (2, 4)
    assert refit_knobs[_CELL] == knobs[_CELL] | {"refit_through": _REFIT}
    # The rows after it price as a pickle without the key prices them, rung for rung; only the
    # row ids the window renumbers differ.
    after = rungs[rungs["Date"] > _REFIT].reset_index(drop=True)
    pd.testing.assert_frame_equal(refit_rungs.drop(columns="rid"), after.drop(columns="rid"))


def test_a_refit_before_the_window_prices_as_a_pickle_without_the_key(tmp_path, archive, capsys):
    early = _priced(archive, _cell(tmp_path / "early", refit_through="2026-10-31"))
    plain = _priced(archive, _cell(tmp_path / "plain"))

    pd.testing.assert_frame_equal(early[0], plain[0])
    assert early[2] == plain[2]
    assert capsys.readouterr().out == ""


def test_price_dumps_excludes_a_cell_its_refit_trained_on_throughout(tmp_path, archive):
    rungs, _, cells, _, excluded = _priced(
        archive, _cell(tmp_path / "refit", refit_through="2026-11-04")
    )

    assert rungs.empty and not cells
    assert excluded == {_CELL: _ALL_IN_SAMPLE}


def test_ship_priced_serves_only_the_rows_after_refit_through(tmp_path, capsys):
    refit = _cell(tmp_path / "refit", refit_through=_REFIT)
    plain = _cell(tmp_path / "plain")

    served = ps._served_cells([_CELL], refit, refit, _START, "candidate")[_CELL]
    unguarded = ps._served_cells([_CELL], plain, plain, _START, "candidate")[_CELL]

    assert _DROPPED in capsys.readouterr().out
    pd.testing.assert_frame_equal(served, unguarded[unguarded["Date"] > _REFIT])


def test_ship_priced_leaves_out_a_cell_its_refit_trained_on_throughout(tmp_path, capsys):
    refit = _cell(tmp_path / "refit", refit_through="2026-11-04")

    assert ps._served_cells([_CELL], refit, refit, _START, "candidate") == {}
    assert f"excluded {_CELL} (candidate): {_ALL_IN_SAMPLE}" in capsys.readouterr().out


# The feature-parity world: a linear stand-in for the booster, the real decode and log writer.
_FEATURES = ["MeanYr", "STDYr", "Avg5"]
_OLD, _NEW = "20261001.none.aaaaaaaa", "20261006.none.bbbbbbbb"
# The old version's logged game dates, the one on the refit's own date last.
_LOG_DATES = ("2026-10-05", "2026-10-03", "2026-10-04")
_LOG_REFIT = "2026-10-04"


def _booster(filedict, market, stat_data, frame, *_):
    """Stand-in for the LightGBMLSS booster: a DPO mean linear in two features."""
    signal = (frame["MeanYr"] + 0.5 * frame["Avg5"]).to_numpy(float)
    return pd.DataFrame({"mu": filedict["model"] * signal, "phi": 1.0}, index=frame.index)


def _filedict(coefficient: float, version: str, **keys) -> dict:
    return {
        "model": coefficient,
        "expected_columns": _FEATURES,
        "distribution": "DPO",
        "cv": 1.0,
        "step": 1.0,
        "hist_gate": 0.0,
        "target_normalization": "none",
        "normalized": False,
        "offset_meta": None,
        "model_version": version,
        "weight": 0.5,
        **keys,
    }


def _save(path: Path, filedict: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as outfile:
        pickle.dump(filedict, outfile)


def _features(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "MeanYr": rng.uniform(1, 4, 3),
            "STDYr": rng.uniform(0.5, 2, 3),
            "Avg5": rng.uniform(1, 4, 3),
        },
        index=["Player 0", "Player 1", "Player 2"],
    )


@pytest.fixture
def logged(tmp_path, monkeypatch) -> pd.DataFrame:
    """The old version's rows on each of ``_LOG_DATES``, served on Avg5 0.3 above the matrix's.

    Returned in ``_LOG_DATES`` order, so the rows dated on the refit come last. Scratch models
    and matrix sit beside the log, and the booster is stood in.
    """
    monkeypatch.setattr(feature_log, "FEATURE_LOG_DIR", tmp_path / "feature_log")
    monkeypatch.setattr(feature_parity, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(feature_parity, "TRAINING_DATA_DIR", tmp_path / "training_data")
    monkeypatch.setattr(feature_parity, "_build_prob_params", _booster)
    old = _filedict(1.0, _OLD)
    _save(tmp_path / "models" / "prior" / f"{_CELL}__{_OLD}.mdl", old)
    matrix = []
    for seed, date in enumerate(_LOG_DATES):
        features = _features(seed)
        served = features.assign(Avg5=features["Avg5"] + 0.3)
        outputs = _booster(old, "PTS", None, served)
        mp._decode_model_params(outputs, "DPO", served, 0.0, None, "none", None)
        upsert_feature_log(
            "NBA",
            "PTS",
            "Underdog",
            [{"Player": player, "Date": date} for player in served.index],
            None,
            served.assign(**{"Quote Line": 2.5}),
            outputs,
            [None] * len(served),
            model_version=_OLD,
            step=1.0,
            model_weight=0.5,
            hist_gate=0.0,
        )
        rows = features.rename_axis("Player").reset_index()
        matrix.append(rows.assign(Date=pd.Timestamp(date), Line=2.5))
    (tmp_path / "training_data").mkdir()
    pd.concat(matrix).to_parquet(tmp_path / "training_data" / f"{_CELL}.parquet")
    return pd.concat(
        [
            pd.read_parquet(tmp_path / "feature_log" / f"date={date}" / f"{_CELL}.parquet")
            for date in _LOG_DATES
        ],
        ignore_index=True,
    )


def _served_file(**keys) -> None:
    _save(Path(str(feature_parity.MODELS_DIR)) / f"{_CELL}.mdl", _filedict(1.2, _NEW, **keys))


_PARITY = ["log_rows", "matched", "dp_mean", "dp_sd", "dp_rms", "dp_rms_lo", "served_rms", "flag"]


def test_the_fixed_file_scores_only_the_rows_after_its_refit(logged, capsys):
    _served_file(refit_through=_LOG_REFIT)
    records, _, _ = feature_parity.check_cell(logged, False, True)
    dropped = f"{_CELL}: dropped 6 of 9 rows on or before refit_through {_LOG_REFIT}"
    assert dropped in capsys.readouterr().out

    _served_file()
    unguarded, _, _ = feature_parity.check_cell(logged[logged["Date"] > _LOG_REFIT], False, True)

    assert records[0]["log_rows"] == 3
    assert records[0]["dp_mean"] != 0  # the served Avg5 moves every probability
    assert {key: records[0][key] for key in _PARITY} == {key: unguarded[0][key] for key in _PARITY}


def test_a_fixed_file_refit_on_every_logged_row_is_excluded_with_a_note(logged):
    _served_file(refit_through="2026-10-05")

    records, _, _ = feature_parity.check_cell(logged, False, True)

    assert records == [
        {
            "cell": _CELL,
            "League": "NBA",
            "version": _NEW,
            "log_rows": 9,
            "note": f"every {_CELL} row (2026-10-03 to 2026-10-05) is on or before refit_through "
            "2026-10-05, in the refit's training rows",
        }
    ]


def test_another_version_re_scores_only_the_rows_after_its_refit(logged, capsys):
    _served_file(refit_through=_LOG_REFIT)
    _, _, pairs = feature_parity.check_cell(logged, True, False)
    dropped = f"{_CELL}: dropped 6 of 9 rows on or before refit_through {_LOG_REFIT}"
    assert dropped in capsys.readouterr().out

    _served_file()
    _, _, unguarded = feature_parity.check_cell(logged[logged["Date"] > _LOG_REFIT], True, False)

    assert pairs[0]["n"] == 3
    assert pairs[0]["dp_mean"] > 0  # the served file's larger coefficient lifts every mean
    assert pairs == unguarded


def test_another_version_refit_on_every_row_is_noted_not_re_scored(logged):
    _served_file(refit_through="2026-10-05")

    records, _, pairs = feature_parity.check_cell(logged, True, False)

    assert records[0]["reproduced"] == 1.0  # the old version's own file still scores its rows
    assert pairs == [
        {
            "cell": _CELL,
            "League": "NBA",
            "version": _OLD,
            "other_version": _NEW,
            "note": f"every {_CELL} row (2026-10-03 to 2026-10-05) is on or before refit_through "
            "2026-10-05, in the refit's training rows",
        }
    ]
