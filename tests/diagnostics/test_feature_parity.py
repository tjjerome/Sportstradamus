"""Diagnostics tests for the feature-parity monitor (``sportstradamus admin feature-parity``).

(a) Identical serve and training features give a zero probability difference and no flag.
(b) A perturbed feature is the top differing feature and moves the probability; a
    difference inside the tolerance is not reported.
(c) The flag trips once the standard deviation exceeds the alarm, not at it.
(d) A row whose version is only under ``prior/`` is scored from there, and re-scored under
    the served version with the days between the two stamps, each version at its own gate.
(e) Rows the log cannot replay are excluded with a reason: a version whose logged outputs
    do not reproduce, a version with no file, a file serving would refuse, a volume cell. A
    row seeded from an incomplete batch is left out alone.
(f) The coverage listing names a cell, and a platform of a logged cell, with settled history
    legs and no log rows.

Fixture-based under ``tmp_path``: the log is written by the real ``upsert_feature_log``, the
model files and the training matrix sit beside it, and a linear stand-in replaces the
LightGBMLSS booster. The decode, the over-probability and every comparison are the real ones.
"""

from __future__ import annotations

import importlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from sportstradamus.prediction import feature_log
from sportstradamus.prediction.feature_log import upsert_feature_log
from sportstradamus.scripts import feature_parity

# The package __init__ re-exports the model_prob *function*, shadowing the submodule.
mp = importlib.import_module("sportstradamus.prediction.model_prob")

pytestmark = pytest.mark.diagnostics

_DATE = "2026-10-04"
_FEATURES = ["MeanYr", "STDYr", "ZeroYr", "Avg5", "Home"]
_OLD, _NEW = "20260927.none.aaaaaaaa", "20261004.none.bbbbbbbb"
_HISTORY_COLUMNS = ["League", "Market", "Platform", "Date", "Actual", "Model Version"]


def _booster(filedict, market, stat_data, frame, *_):
    """Stand-in for the LightGBMLSS booster: a DPO mean linear in two features.

    A row without dispersion history borrows its batch's median ``STDYr``, the way
    ``set_model_start_values`` seeds it.
    """
    std = frame["STDYr"].to_numpy(float)
    seeded = np.where(std > 0, std, np.median(std[std > 0]))
    signal = (frame["MeanYr"] + 0.5 * frame["Avg5"]).to_numpy(float)
    return pd.DataFrame(
        {"mu": filedict["model"] * signal + 0.1 * seeded, "phi": 1.0}, index=frame.index
    )


def _filedict(coefficient: float, version: str, expected=_FEATURES) -> dict:
    return {
        "model": coefficient,
        "expected_columns": list(expected),
        "distribution": "DPO",
        "cv": 1.0,
        "step": 1.0,
        "hist_gate": 0.0,
        "target_normalization": "none",
        "normalized": False,
        "offset_meta": None,
        "model_version": version,
    }


def _stale(version: str) -> dict:
    """A model file whose strategy identity the serving check no longer accepts."""
    return _filedict(1.0, version) | {"model_strategy": "retired"}


def _save(path: Path, filedict: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as outfile:
        pickle.dump(filedict, outfile)
    return filedict


def _features(n: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "MeanYr": rng.uniform(1, 4, n),
            "STDYr": rng.uniform(0.5, 2, n),
            "ZeroYr": 0.1,
            "Avg5": rng.uniform(1, 4, n),
            "Home": rng.random(n) < 0.5,
        },
        index=[f"Player {i}" for i in range(n)],
    )


def _outputs(filedict: dict, features: pd.DataFrame) -> pd.DataFrame:
    outputs = _booster(filedict, "PTS", None, features)
    mp._decode_model_params(outputs, "DPO", features, 0.0, None, "none")
    return outputs


def _over(filedict: dict, features: pd.DataFrame) -> np.ndarray:
    """The over-probability at the fixtures' 2.5 line, wired here and not through the module."""
    outputs = _outputs(filedict, features)
    return mp._model_over_and_push(
        outputs.assign(Line=2.5), "DPO", 1.0, 1.0, outputs["Projection"].to_numpy()
    )[0]


def _serve(
    filedict: dict,
    features: pd.DataFrame,
    *,
    league: str = "NBA",
    market: str = "PTS",
    date: str = _DATE,
    quote_line=2.5,
    hist_gate: float = 0.0,
) -> None:
    """Log one ``model_prob`` call: stand-in booster, real decode, real log writer."""
    upsert_feature_log(
        league,
        market,
        "Underdog",
        [{"Player": player, "Date": date} for player in features.index],
        None,
        features.assign(**{"Quote Line": quote_line}),
        _outputs(filedict, features),
        [None] * len(features),
        model_version=filedict["model_version"],
        step=1.0,
        model_weight=0.5,
        hist_gate=hist_gate,
    )


def _matrix(world: Path, features: pd.DataFrame, *, cell: str = "NBA_PTS", date: str = _DATE):
    rows = features.rename_axis("Player").reset_index().assign(Date=pd.Timestamp(date), Line=2.5)
    rows.to_parquet(world / "training_data" / f"{cell}.parquet")


def _run(world: Path, *args: str) -> tuple[str, pd.DataFrame]:
    out = world / "parity.csv"
    result = CliRunner().invoke(
        feature_parity.main,
        ["--log-dir", str(world / "feature_log"), "--out", str(out), *args],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    return result.output, pd.read_csv(out)


def _table(csv: pd.DataFrame, name: str) -> pd.DataFrame:
    return csv[csv["table"] == name].dropna(axis=1, how="all").reset_index(drop=True)


@pytest.fixture
def world(tmp_path, monkeypatch) -> Path:
    """Scratch models, matrices, log and an empty history, with the booster stood in."""
    monkeypatch.setattr(feature_log, "FEATURE_LOG_DIR", tmp_path / "feature_log")
    monkeypatch.setattr(feature_parity, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(feature_parity, "TRAINING_DATA_DIR", tmp_path / "training_data")
    monkeypatch.setattr(feature_parity, "_build_prob_params", _booster)
    history = pd.DataFrame(columns=_HISTORY_COLUMNS)
    monkeypatch.setattr(feature_parity, "read_history", lambda: history)
    (tmp_path / "training_data").mkdir()
    return tmp_path


def test_identical_features_give_zero_difference_and_no_flag(world):
    features = _features()
    _serve(_save(world / "models" / "NBA_PTS.mdl", _filedict(1.0, _NEW)), features)
    _matrix(world, features)

    _, csv = _run(world)

    cell = _table(csv, "summary").iloc[0]
    assert (cell["cell"], cell["version"]) == ("NBA_PTS", _NEW)
    assert cell["reproduced"] == 1.0
    assert cell["matched"] == len(features)
    assert cell["dp_mean"] == 0.0
    assert cell["dp_sd"] == 0.0
    assert not cell["flag"]
    assert _table(csv, "feature").empty


def test_perturbed_feature_surfaces_first_and_moves_the_probability(world):
    features = _features()
    served = features.copy()
    served.loc[served.index[:4], "Avg5"] += 1.0
    served.loc[served.index[0], "ZeroYr"] = 0.3
    served["MeanYr"] *= 1 + 1e-9  # float noise, inside FEATURE_TOL
    model = _save(world / "models" / "NBA_PTS.mdl", _filedict(1.0, _NEW))
    _serve(model, served)
    _matrix(world, features)

    output, csv = _run(world)

    differing = _table(csv, "feature")
    assert differing["feature"].tolist() == ["Avg5", "ZeroYr"]
    assert differing["differ_share"].tolist() == [0.5, 0.125]
    assert differing["mean_abs_diff"].iloc[0] == pytest.approx(0.5)
    assert "Avg5" in output
    cell = _table(csv, "summary").iloc[0]
    # A higher Avg5 at serve lifts the mean: serve features minus training features.
    expected = _over(model, served) - _over(model, features)
    assert expected.mean() > 0.01
    assert cell["dp_mean"] == pytest.approx(expected.mean())
    assert cell["dp_sd"] == pytest.approx(expected.std(ddof=1))
    assert cell["dp_sd"] > feature_parity.PARITY_SD_ALARM
    assert cell["flag"]


def test_flag_trips_once_the_sd_exceeds_the_alarm(world, monkeypatch):
    assert feature_parity.PARITY_SD_ALARM == 0.01  # one percentage point

    features = _features()
    served = features.copy()
    served.loc[served.index[:4], "Avg5"] += 0.2
    _serve(_save(world / "models" / "NBA_PTS.mdl", _filedict(1.0, _NEW)), served)
    _matrix(world, features)
    logged = pd.read_parquet(world / "feature_log" / f"date={_DATE}" / "NBA_PTS.parquet")

    def record() -> dict:
        return feature_parity.check_cell(logged, False)[0][0]

    sd = record()["dp_sd"]
    assert sd > 0
    monkeypatch.setattr(feature_parity, "PARITY_SD_ALARM", sd)
    assert not record()["flag"]
    monkeypatch.setattr(feature_parity, "PARITY_SD_ALARM", np.nextafter(sd, 0))
    assert record()["flag"]


def test_version_only_in_prior_is_scored_from_there(world):
    features = _features()
    old = _save(world / "models" / "prior" / f"NBA_PTS__{_OLD}.mdl", _filedict(1.0, _OLD))
    new = _save(world / "models" / "NBA_PTS.mdl", _filedict(1.2, _NEW))
    quote_line = pd.Series(2.5, index=features.index)
    quote_line.iloc[0] = np.nan  # no book quoted this player: the log holds no line for the row
    _serve(old, features, quote_line=quote_line)
    _matrix(world, features, date="2026-09-20")  # the matrix has not reached the logged game

    _, csv = _run(world, "--versions")

    cell = _table(csv, "summary").iloc[0]
    # Only the prior file's coefficient rebuilds what the rows logged.
    assert (cell["version"], cell["reproduced"], cell["matched"]) == (_OLD, 1.0, 0)
    pair = _table(csv, "version").iloc[0]
    assert (pair["version"], pair["other_version"]) == (_OLD, _NEW)
    assert pair["days_between"] == 7
    assert pair["n"] == len(features) - 1
    # The served file's larger coefficient lifts every mean: other version minus the logged one.
    expected = (_over(new, features) - _over(old, features))[1:]
    assert expected.mean() > 0
    assert pair["dp_mean"] == pytest.approx(expected.mean())
    assert pair["dp_sd"] == pytest.approx(expected.std(ddof=1))

    output, csv = _run(world)  # without the option no other version is scored
    assert "version re-score" not in output
    assert _table(csv, "version").empty


def test_each_version_is_decoded_at_its_own_gate(world, monkeypatch):
    features = _features()
    old = _save(world / "models" / "prior" / f"NBA_PTS__{_OLD}.mdl", _filedict(1.0, _OLD))
    _save(world / "models" / "NBA_PTS.mdl", _filedict(1.2, _NEW) | {"hist_gate": 0.4})
    _serve(old, features, hist_gate=0.25)
    _matrix(world, features, date="2026-09-20")
    gates = []
    decode = feature_parity._decode_model_params

    def recording(outputs, dist, frame, hist_gate, *rest):
        gates.append(hist_gate)
        decode(outputs, dist, frame, hist_gate, *rest)

    monkeypatch.setattr(feature_parity, "_decode_model_params", recording)

    _run(world, "--versions")

    # The logged version at the runtime gate its rows logged, not the 0.0 in its file; the
    # other version at the gate its own file keeps.
    assert gates == [0.25, 0.4]


def test_version_needing_unlogged_features_is_not_rescored(world):
    features = _features()
    served = _save(world / "models" / "NBA_PTS.mdl", _filedict(1.0, _NEW))
    wider = _filedict(1.0, _OLD, expected=[*_FEATURES, "Minutes"])
    _save(world / "models" / "prior" / f"NBA_PTS__{_OLD}.mdl", wider)
    _serve(served, features)
    _matrix(world, features)

    output, csv = _run(world, "--versions")

    assert f"not re-scored NBA_PTS {_NEW} under {_OLD}: needs features" in output
    assert "n" not in _table(csv, "version").columns  # the pair is recorded with its note only


def test_unreproducible_version_is_excluded_and_reported(world):
    features = _features()
    served = features.assign(Avg5=features["Avg5"] + 1.0)
    logged_by = _save(world / "models" / "prior" / f"NBA_PTS__{_NEW}.mdl", _filedict(1.0, _NEW))
    _serve(logged_by, served)
    # A same-day retrain: the served file carries the same version stamp with another
    # booster, and the served file answers for a version it still carries.
    _save(world / "models" / "NBA_PTS.mdl", _filedict(1.3, _NEW))
    _save(world / "models" / "prior" / f"NBA_PTS__{_OLD}.mdl", _stale(_OLD))
    _matrix(world, features)

    output, csv = _run(world, "--versions")

    cell = _table(csv, "summary").iloc[0]
    assert cell["reproduced"] == 0.0
    assert f"excluded NBA_PTS {_NEW}: 8 of 8 rows' logged outputs not reproduced" in output
    # Left out of parity although every row is in the matrix and Avg5 differs on all of them,
    # and out of the version re-score, the note on the file it could not use included.
    assert "matched" not in cell.index
    assert _table(csv, "feature").empty
    assert _table(csv, "version").empty


def test_model_file_serving_refuses_is_reported_and_not_scored(world):
    features = _features()
    served = _save(world / "models" / "NBA_PTS.mdl", _filedict(1.0, _NEW))
    stale = _save(world / "models" / "prior" / f"NBA_PTS__{_OLD}.mdl", _stale(_OLD))
    _serve(served, features)
    _serve(stale, features, date="2026-09-27")
    _matrix(world, features)

    output, csv = _run(world)

    refusal = "serving refuses the model file (malformed model_strategy identity)"
    assert f"excluded NBA_PTS {_OLD}: {refusal}" in output
    assert "not re-scored" not in output
    summary = _table(csv, "summary").set_index("version")
    assert summary.loc[_NEW, "reproduced"] == 1.0
    assert summary.loc[_NEW, "matched"] == len(features)
    assert pd.isna(summary.loc[_OLD, "reproduced"])

    output, csv = _run(world, "--versions")

    assert f"not re-scored NBA_PTS {_NEW} under {_OLD}: {refusal}" in output
    assert "n" not in _table(csv, "version").columns  # the pair is recorded with its note only


def test_row_seeded_from_an_incomplete_batch_is_left_out_alone(world):
    features = _features()
    features.loc["Player 0", "STDYr"] = 0.0
    model = _save(world / "models" / "NBA_PTS.mdl", _filedict(1.0, _NEW))
    _serve(model, features)
    # A later call rescored one player, so the first call's batch is no longer whole and its
    # median STDYr, which seeded Player 0, cannot be rebuilt.
    _serve(model, features.loc[["Player 7"]])
    _matrix(world, features)

    output, csv = _run(world)

    cell = _table(csv, "summary").iloc[0]
    assert cell["reproduced"] == 7 / 8
    assert cell["matched"] == 7
    assert cell["dp_sd"] == 0.0
    assert "excluded" not in output


def test_rows_the_log_cannot_replay_are_excluded_with_a_reason(world):
    features = _features()
    _save(world / "models" / "NBA_PTS.mdl", _filedict(1.0, _NEW))
    _serve(_filedict(1.0, _OLD), features)
    _serve(_filedict(1.0, _NEW), features, market="MIN")
    _serve(_save(world / "models" / "NBA_REB.mdl", _filedict(1.0, _NEW)), features, market="REB")
    _matrix(world, features)
    _matrix(world, features, cell="NBA_REB")
    # Rows written before the decode gave this output: nothing logged to compare it with.
    rebounds = world / "feature_log" / f"date={_DATE}" / "NBA_REB.parquet"
    pd.read_parquet(rebounds).drop(columns="Model Phi").to_parquet(rebounds)

    output, csv = _run(world)

    assert f"excluded NBA_PTS {_OLD}: no model file on disk carries this version" in output
    assert f"excluded NBA_MIN {_NEW}: volume cell" in output
    assert f"excluded NBA_REB {_NEW}: 8 of 8 rows' logged outputs not reproduced" in output
    assert _table(csv, "summary")["note"].notna().all()


def test_coverage_lists_a_cell_with_history_and_no_log(world, monkeypatch):
    features = _features()
    for cell, market in (
        ("NBA_PTS", "PTS"),
        ("NBA_fantasy-points-prizepicks", "fantasy points prizepicks"),
    ):
        _serve(
            _save(world / "models" / f"{cell}.mdl", _filedict(1.0, _NEW)), features, market=market
        )
        _matrix(world, features, cell=cell)
    history = pd.DataFrame(
        [
            ("NBA", "PTS", "Underdog", _DATE, 21.0, _NEW),
            # History's name for the leg the "fantasy points prizepicks" cell served and logged.
            ("NBA", "fantasy points underdog", "Underdog", _DATE, 30.5, _NEW),
            ("NBA", "PTS", "Sleeper", _DATE, 19.0, _NEW),  # the cell logged, this platform not
            ("NBA", "REB", "Underdog", _DATE, 7.0, _NEW),
            ("NBA", "REB", "Underdog", _DATE, 9.0, _NEW),
            # Priced off the book, so there was nothing to log.
            ("NBA", "AST", "Underdog", _DATE, 4.0, "book_fallback"),
            ("NBA", "BLK", "Underdog", _DATE, np.nan, _NEW),  # not settled
            ("NBA", "STL", "Underdog", "2026-10-03", 1.0, _NEW),  # before the log's first day
        ],
        columns=_HISTORY_COLUMNS,
    )
    monkeypatch.setattr(feature_parity, "read_history", lambda: history)

    output, csv = _run(world)

    gaps = _table(csv, "coverage")[["Market", "Platform", "Date", "legs"]]
    assert gaps.to_dict("records") == [
        {"Market": "PTS", "Platform": "Sleeper", "Date": _DATE, "legs": 1},
        {"Market": "REB", "Platform": "Underdog", "Date": _DATE, "legs": 2},
    ]
    assert "REB" in output


def test_league_and_date_window_cut_the_log(world):
    features = _features()
    _save(world / "models" / "NBA_PTS.mdl", _filedict(1.0, _NEW))
    _save(world / "models" / "NHL_points.mdl", _filedict(1.0, _NEW))
    _serve(_filedict(1.0, _NEW), features)
    _serve(_filedict(1.0, _NEW), features.iloc[:3], date="2026-10-06")
    _serve(_filedict(1.0, _NEW), features, league="NHL", market="points", date="2026-10-06")
    _matrix(world, features)
    _matrix(world, features, cell="NHL_points")

    _, csv = _run(world, "--league", "NBA", "--start", "2026-10-05", "--end", "2026-10-07")

    summary = _table(csv, "summary")
    assert summary[["cell", "log_rows"]].to_dict("records") == [{"cell": "NBA_PTS", "log_rows": 3}]

    empty = CliRunner().invoke(
        feature_parity.main,
        ["--log-dir", str(world / "feature_log"), "--start", "2026-11-01"],
    )
    assert empty.exit_code == 1
    assert "no feature log rows" in empty.output
