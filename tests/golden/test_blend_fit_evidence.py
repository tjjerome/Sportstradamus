"""Golden pins for the evidence floor on the model/book blend-weight fit."""

import logging

import numpy as np
import pandas as pd
import pytest

from sportstradamus.training import pipeline as pipe

CV = 0.35
TEST_INDEX = pd.Index([100, 101, 102])
TEST_AUTHENTICITY = ["authentic", "synthetic", "authentic"]
MODEL_EV = np.array([4.0, 4.0, 4.0])
BOOK_EV = np.array([6.0, 6.0, 6.0])
SN_SIGMA = np.array([2.0, 2.0, 2.0])
SN_ALPHA = np.array([0.5, 0.5, 0.5])
FITTED_WEIGHT = 0.42
FLOOR_WARNING = "blend weight not fit: 2 authentic validation rows over 2 clusters (floor 10)"


def _splits(n_validation: int, *, players=None, dates=None, authenticity=None) -> dict:
    index = pd.RangeIndex(n_validation)
    return {
        "B_test": pd.DataFrame(
            {"Line": [5.0, 6.0, 7.0], "Odds": [0.5, 0.5, 0.5], "EV": BOOK_EV},
            index=TEST_INDEX,
        ),
        "B_validation": pd.DataFrame(
            {
                "Line": np.full(n_validation, 5.0),
                "Odds": np.full(n_validation, 0.5),
                "EV": np.full(n_validation, 6.0),
            },
            index=index,
        ),
        "y_validation": pd.DataFrame({"Result": np.linspace(1.0, 9.0, n_validation)}, index=index),
        "quote_authenticity_test": pd.Series(TEST_AUTHENTICITY, index=TEST_INDEX),
        "quote_authenticity_validation": pd.Series(
            authenticity or ["authentic"] * n_validation, index=index
        ),
        "players_validation": None if players is None else pd.Series(players, index=index),
        "dates_validation": None if dates is None else pd.Series(dates, index=index),
    }


def _fuse_skewnormal(splits: dict) -> dict:
    n = len(splits["B_validation"])
    decoded = {
        "ev": MODEL_EV,
        "ev_validation": np.full(n, 4.0),
        "sn_sigma_test": SN_SIGMA,
        "sn_sigma_val": np.full(n, 2.0),
        "sn_alpha_test": SN_ALPHA,
        "sn_alpha_val": np.full(n, 0.5),
        "gate_test": None,
        "gate_validation": None,
    }
    return pipe._fuse_skewnormal({}, decoded, splits, CV, 0.0, "crps_1se")


def _fuse_dpo(splits: dict) -> dict:
    n = len(splits["B_validation"])
    decoded = {
        "ev": MODEL_EV,
        "ev_validation": np.full(n, 4.0),
        "phi": np.ones(3),
        "phi_validation": np.ones(n),
    }
    return pipe._fuse_dpo({}, decoded, splits, CV, "crps")


@pytest.mark.parametrize(
    ("fuse", "fitter"),
    [(_fuse_skewnormal, "fit_blend_weight"), (_fuse_dpo, "fit_dpo_weight")],
    ids=["skewnormal", "dpo"],
)
def test_two_authentic_clusters_serve_model_only(monkeypatch, caplog, fuse, fitter):
    """The DPO branch bypasses ``fit_blend_weight``, so the floor lives in the pipeline and every
    fuse branch inherits it."""
    monkeypatch.setattr(
        pipe.calibration,
        fitter,
        lambda *args, **kwargs: pytest.fail("blend weight fit on two clusters"),
    )

    with caplog.at_level(logging.WARNING, logger=pipe.logger.name):
        fused = fuse(_splits(2, players=["ana", "bo"]))

    assert fused["model_weight"] == 1.0
    np.testing.assert_allclose(fused["weighted_mean"], MODEL_EV)
    assert FLOOR_WARNING in caplog.text


def test_ten_authentic_clusters_fit_and_reach_the_served_rows(monkeypatch):
    monkeypatch.setattr(pipe.calibration, "fit_blend_weight", lambda *args, **kwargs: FITTED_WEIGHT)

    fused = _fuse_skewnormal(_splits(10, players=[f"player-{i}" for i in range(10)]))

    assert fused["model_weight"] == FITTED_WEIGHT
    expected, *_ = pipe.fused_loc(
        np.array([FITTED_WEIGHT, 1.0, FITTED_WEIGHT]),
        MODEL_EV,
        BOOK_EV,
        CV,
        "SkewNormal",
        sigma=SN_SIGMA,
        skew_alpha=SN_ALPHA,
    )
    np.testing.assert_allclose(fused["weighted_mean"], expected)
    assert fused["weighted_mean"][1] == MODEL_EV[1]
    assert fused["weighted_mean"][0] != MODEL_EV[0]


@pytest.mark.parametrize(("n_dates", "fits"), [(10, True), (2, False)])
def test_team_markets_cluster_on_dates(monkeypatch, n_dates, fits):
    monkeypatch.setattr(pipe.calibration, "fit_blend_weight", lambda *args, **kwargs: FITTED_WEIGHT)
    dates = pd.to_datetime([f"2026-01-{1 + i % n_dates:02d}" for i in range(10)])

    fused = _fuse_skewnormal(_splits(10, dates=dates))

    assert fused["model_weight"] == (FITTED_WEIGHT if fits else 1.0)


def test_clusters_align_by_label_when_metadata_order_differs(monkeypatch):
    """``players_validation`` keeps the Date-sorted split order while ``B_validation`` (and the
    authenticity mask built over its index) were index-sorted in ``_step_predict_splits``. A
    positional pick would credit the ten synthetic rows' distinct players to the authentic rows."""
    monkeypatch.setattr(
        pipe.calibration,
        "fit_blend_weight",
        lambda *args, **kwargs: pytest.fail("blend weight fit on two clusters"),
    )
    splits = _splits(20, authenticity=["authentic"] * 10 + ["synthetic"] * 10)
    players = ["ana", "bo"] * 5 + [f"player-{i}" for i in range(10)]
    splits["players_validation"] = pd.Series(players, index=pd.RangeIndex(20))[::-1]
    mask = pipe._split_quote_authenticity_mask(splits, "validation")

    assert pipe._blend_fit_cluster_count(splits, mask) == 2
    assert _fuse_skewnormal(splits)["model_weight"] == 1.0
