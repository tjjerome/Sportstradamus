"""Golden pins for probability orientation and served-shape artifact parity."""

import numpy as np
import pandas as pd
import pytest

from sportstradamus.helpers import get_odds
from sportstradamus.helpers.distributions import _dp_mu_from_mean
from sportstradamus.training import baselines
from sportstradamus.training import pipeline as pipe
from sportstradamus.training.model_strategy import (
    MODEL_STRATEGY_MODEL_KEY,
    build_artifact_identity,
    get_strategy,
    validate_strategy_artifacts,
)
from sportstradamus.training.scorecard import _brier_inputs, _pred_cdf_pmf

SERVED_MEAN = np.array([2.4, 7.8])
SERVED_R = np.array([3.2, 11.0])
SERVED_GATE = np.array([0.12, 0.37])
SERVED_PHI = np.array([0.7, 1.6])
INTEGER_LINES = np.array([1.0, 8.0])


def _raw_params() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "total_count": [101.0, 102.0],
            "probs": [0.95, 0.96],
            "gate": [0.91, 0.92],
            "mu": [51.0, 61.0],
            "phi": [0.21, 0.31],
        }
    )


def test_rung_c_validation_pit_randomizes_served_skewnormal_gate():
    n = 7
    fused = {
        "weighted_mean_val": np.full(n, 12.0),
        "sn_sigma_blend_val": np.full(n, 6.0),
        "sn_alpha_blend_val": np.full(n, 1.5),
        "gate_blend_val": np.full(n, 0.2),
    }
    y = np.array([0.0, 0.0, 5.0, 8.0, 12.0, 16.0, 25.0])

    gated = pipe._blended_val_pit(fused, y)
    assert gated.shape == (25, n)

    fused["gate_blend_val"] = None
    ungated = pipe._blended_val_pit(fused, y)
    assert ungated.shape == (n,)


def test_staged_skewnormal_gate_is_fused_served_gate():
    frame = pd.DataFrame({"MeanYr": [3.0, 8.0]})
    pipe._stage_family_shape_columns(
        frame,
        dist="SkewNormal",
        prob_params=_raw_params(),
        weighted_mean=SERVED_MEAN,
        sn_scale_test=np.array([1.2, 2.3]),
        sn_skew_test=np.array([0.4, 0.7]),
        mix_test=None,
        r_test=None,
        gate_blend_test=SERVED_GATE,
        phi_test=None,
        target_normalization="ratio_meanyr",
        global_mean=5.0,
        denom_col="MeanYr",
        hist_gate=0.91,
    )
    np.testing.assert_array_equal(frame["Gate"], SERVED_GATE)


def _stage_count_shape(
    dist: str,
    *,
    r_test: np.ndarray | None,
    gate_blend_test: np.ndarray | None,
    phi_test: np.ndarray | None,
    target_normalization: str = "ratio_meanyr",
) -> pd.DataFrame:
    """Stage one count family's scorecard columns from explicit served parameters."""
    frame = pd.DataFrame({"MeanYr": [3.0, 8.0]})
    pipe._stage_family_shape_columns(
        frame,
        dist=dist,
        prob_params=_raw_params(),
        weighted_mean=SERVED_MEAN,
        sn_scale_test=None,
        sn_skew_test=None,
        mix_test=None,
        r_test=r_test,
        gate_blend_test=gate_blend_test,
        phi_test=phi_test,
        target_normalization=target_normalization,
        global_mean=5.0,
        denom_col="MeanYr",
        hist_gate=0.0,
    )
    return frame


def test_persist_converts_internal_book_over_to_scorecard_under(monkeypatch, tmp_path):
    index = pd.Index([11, 29])
    book_over = np.array([0.2, 0.65])
    splits = {
        "X_test": pd.DataFrame({"MeanYr": [3.0, 8.0]}, index=index),
        "y_test": pd.DataFrame({"Result": [1.0, 10.0]}, index=index),
        "B_test": pd.DataFrame(
            {"Line": INTEGER_LINES, "Odds": book_over, "EV": SERVED_MEAN},
            index=index,
        ),
        "players_test": None,
        "dates_test": np.array(["2026-01-04", "2026-01-11"]),
        "quote_authenticity_test": pd.Series(
            ["authentic", "synthetic"],
            index=index,
        ),
    }
    model_over = np.array([0.4, 0.7])
    matrix_hash = "b" * 64
    selected_controls = {
        "dist": "NegBin",
        "count_dispersion_objective": "crps",
        "blending_loss_fn": "crps",
        "posthoc": "none",
    }
    filedict = {
        "distribution": "NegBin",
        "target_normalization": "none",
        "normalized": False,
        "posthoc": "none",
        MODEL_STRATEGY_MODEL_KEY: build_artifact_identity(
            "NegBin",
            "NFL",
            "artifact parity",
            selected_controls,
            matrix_hash=matrix_hash,
        ).as_model_blob(),
    }
    monkeypatch.setattr(pipe.pkg_resources, "files", lambda _package: tmp_path)

    pipe._step_persist_artifacts(
        filedict=filedict,
        splits=splits,
        prob_params=_raw_params().set_axis(index),
        decoded={"ev": np.array([2.0, 7.0])},
        weighted_mean=SERVED_MEAN,
        y_proba_filt=np.column_stack([1.0 - model_over, model_over]),
        y_proba_raw=np.column_stack([1.0 - model_over, model_over]),
        dist="NegBin",
        hist_gate=0.0,
        filename="NFL_artifact_parity",
        deterministic=False,
        target_normalization="ratio_meanyr",
        zinb_mode="joint",
        sn_scale_test=None,
        sn_skew_test=None,
        mix_test=None,
        r_test=SERVED_R,
        gate_blend_test=None,
        phi_test=None,
        global_mean=5.0,
        denom_col="MeanYr",
    )

    persisted = pd.read_csv(tmp_path / "test_sets" / "NFL_artifact_parity.csv")
    np.testing.assert_allclose(persisted["Odds"], 1.0 - book_over)
    assert persisted["QuoteAuthenticity"].tolist() == ["authentic", "synthetic"]
    brier_inputs = _brier_inputs(persisted)
    assert brier_inputs is not None
    _, recovered_book_over, _, _ = brier_inputs
    np.testing.assert_allclose(recovered_book_over, book_over[:1])
    saved_model = pd.read_pickle(tmp_path / "models" / "NFL_artifact_parity.mdl")
    identity = validate_strategy_artifacts(
        get_strategy("NegBin"),
        selected_controls,
        persisted,
        saved_model,
        league="NFL",
        market="artifact parity",
        matrix_hash=matrix_hash,
    )
    assert identity.strategy_slug == "NegBin"
    assert persisted["ModelStrategy"].eq("NegBin").all()
    assert persisted["StructuralStrategy"].eq("none").all()
    assert persisted["StrategyControlsJSON"].eq(identity.controls_json).all()
    assert persisted["StrategyCornerFingerprint"].eq(identity.corner_fingerprint).all()
    assert persisted["StrategyMatrixHash"].eq(matrix_hash).all()


def test_split_metadata_stays_aligned_when_prediction_sorts_test_rows(monkeypatch):
    class _StatData:
        @staticmethod
        def get_stat_columns(_market):
            return ["Home", "Feature"]

    # Descending labels make the held-out metadata order differ from the sorted
    # order enforced by ``_step_predict_splits``.
    index = pd.Index(range(300, 200, -1))
    matrix = pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=len(index)),
            "Player": [f"player-{i}" for i in index],
            "Result": np.arange(len(index), dtype=float),
            "Home": [i % 2 for i in range(len(index))],
            "Feature": np.arange(len(index), dtype=float),
            "Line": np.arange(len(index), dtype=float) + 0.5,
            "Odds": np.full(len(index), 0.55),
            "EV": np.arange(len(index), dtype=float) + 1.0,
        },
        index=index,
    )
    splits = pipe._step_build_splits(matrix, _StatData(), "receiving yards")
    unsorted_test_index = splits["X_test"].index.copy()
    assert not unsorted_test_index.is_monotonic_increasing

    def _predict(_model, _dist, frame, **_kwargs):
        return pd.DataFrame({"location": np.arange(len(frame), dtype=float)}, index=frame.index)

    monkeypatch.setattr(pipe, "predict_lss_params", _predict)
    pipe._step_predict_splits(
        object(),
        "SkewNormal",
        splits,
        normalize=False,
        offset_mode=False,
        sn_param="direct",
    )

    persisted = splits["X_test"].copy()
    pipe._persist_player_metadata(persisted, splits)
    expected = matrix.loc[persisted.index, ["Player", "Date"]]
    pd.testing.assert_series_equal(persisted["Player"], expected["Player"])
    pd.testing.assert_series_equal(persisted["Date"], expected["Date"])


def test_validation_assignment_is_stable_when_held_out_rows_are_added():
    class _StatData:
        @staticmethod
        def get_stat_columns(_market):
            return ["Home", "Feature"]

    def matrix(rows):
        return pd.DataFrame(
            {
                "Date": pd.date_range("2025-01-01", periods=rows),
                "Player": [f"player-{i}" for i in range(rows)],
                "Result": np.arange(rows, dtype=float),
                "Home": [i % 2 for i in range(rows)],
                "Feature": np.arange(rows, dtype=float),
                "Line": np.arange(rows, dtype=float) + 0.5,
                "Odds": np.full(rows, 0.55),
                "EV": np.arange(rows, dtype=float) + 1.0,
            }
        )

    original = pipe._step_build_splits(matrix(100), _StatData(), "receiving yards")
    extended = pipe._step_build_splits(matrix(101), _StatData(), "receiving yards")

    original_roles = {
        **dict.fromkeys(original["X_test"].index, "test"),
        **dict.fromkeys(original["X_validation"].index, "validation"),
    }
    extended_roles = {
        **dict.fromkeys(extended["X_test"].index, "test"),
        **dict.fromkeys(extended["X_validation"].index, "validation"),
    }
    shared = original_roles.keys() & extended_roles.keys()
    assert shared
    assert all(original_roles[index] == extended_roles[index] for index in shared)


@pytest.mark.parametrize("dist", ["NegBin", "ZINB", "DPO"])
def test_staged_count_shape_uses_served_parameters_not_raw_heads(dist):
    gate = SERVED_GATE if dist == "ZINB" else None
    frame = _stage_count_shape(
        dist,
        r_test=SERVED_R if dist in ("NegBin", "ZINB") else None,
        gate_blend_test=gate,
        phi_test=SERVED_PHI if dist == "DPO" else None,
    )

    if dist in ("NegBin", "ZINB"):
        np.testing.assert_array_equal(frame["R"], SERVED_R)
        np.testing.assert_allclose(frame["NB_P"], SERVED_MEAN / (SERVED_R + SERVED_MEAN))
        assert not np.array_equal(frame["R"], _raw_params()["total_count"])
        assert not np.array_equal(frame["NB_P"], _raw_params()["probs"])
        if dist == "ZINB":
            np.testing.assert_array_equal(frame["Gate"], SERVED_GATE)
            assert not np.array_equal(frame["Gate"], _raw_params()["gate"])
        return

    np.testing.assert_array_equal(frame["DP_PHI"], SERVED_PHI)
    np.testing.assert_allclose(frame["DP_MU"], _dp_mu_from_mean(SERVED_MEAN, SERVED_PHI))
    assert not np.array_equal(frame["DP_PHI"], _raw_params()["phi"])
    assert not np.array_equal(frame["DP_MU"], _raw_params()["mu"])


@pytest.mark.parametrize("dist", ["NegBin", "ZINB", "DPO"])
def test_scorecard_mid_cdf_matches_priced_served_distribution(dist):
    gate = SERVED_GATE if dist == "ZINB" else None
    frame = _stage_count_shape(
        dist,
        r_test=SERVED_R if dist in ("NegBin", "ZINB") else None,
        gate_blend_test=gate,
        phi_test=SERVED_PHI if dist == "DPO" else None,
    )
    cdf, pmf = _pred_cdf_pmf(frame, dist, INTEGER_LINES, strategy="ratio_meanyr")
    scorecard_under = cdf - 0.5 * pmf

    if dist in ("NegBin", "ZINB"):
        priced_under = get_odds(
            INTEGER_LINES,
            SERVED_MEAN,
            dist,
            r=SERVED_R,
            gate=gate,
        )
    else:
        priced_under = get_odds(
            INTEGER_LINES,
            SERVED_MEAN,
            dist,
            phi=SERVED_PHI,
        )
    np.testing.assert_allclose(scorecard_under, priced_under, rtol=1e-12, atol=1e-12)


def test_dpo_count_path_target_normalization_is_probability_inert():
    params = _raw_params().loc[:, ["mu", "phi"]]
    features = pd.DataFrame({"MeanYr": [3.0, 8.0]})

    decoded = {}
    for slug in ("none", "ratio_meanyr"):
        strategy = None if slug == "none" else baselines.get_target_normalization(slug)
        decoded[slug] = pipe._step_decode_predictions(
            params,
            params.copy(),
            features,
            features.copy(),
            "DPO",
            strategy,
            5.0,
            "MeanYr",
            0.0,
        )
    for field in ("ev", "ev_validation", "phi", "phi_validation"):
        np.testing.assert_array_equal(decoded["none"][field], decoded["ratio_meanyr"][field])

    staged = {
        slug: _stage_count_shape(
            "DPO",
            r_test=None,
            gate_blend_test=None,
            phi_test=SERVED_PHI,
            target_normalization=slug,
        )
        for slug in ("none", "ratio_meanyr")
    }
    pd.testing.assert_frame_equal(staged["none"], staged["ratio_meanyr"])
    none_cdf, none_pmf = _pred_cdf_pmf(staged["none"], "DPO", INTEGER_LINES, strategy="none")
    ratio_cdf, ratio_pmf = _pred_cdf_pmf(
        staged["ratio_meanyr"], "DPO", INTEGER_LINES, strategy="ratio_meanyr"
    )
    np.testing.assert_array_equal(none_cdf, ratio_cdf)
    np.testing.assert_array_equal(none_pmf, ratio_pmf)


def test_dpo_distribution_selection_accepts_canonical_none_without_transform(monkeypatch):
    market = "passing tds"
    players = np.repeat(["qb-a", "qb-b"], 12)

    class _StatData:
        log_strings = {"player": "Player"}
        gamelog = pd.DataFrame(
            {
                "Player": players,
                market: np.tile([1.0, 2.0, 1.0], 8),
            }
        )

    monkeypatch.setattr(pipe, "save_cv_std_config", lambda *_args, **_kwargs: None)
    selected = {}
    for slug in ("none", "ratio_meanyr"):
        splits = {
            "X_train": pd.DataFrame({"MeanYr": [1.0, 2.0, 1.5, 2.5]}),
            "y_train_labels": np.array([0.0, 1.0, 2.0, 1.0]),
        }
        selected[slug] = pipe._step_select_distribution(
            splits,
            _StatData(),
            market,
            "NFL",
            slug,
            "joint",
            deterministic=True,
            dist_override="DPO",
        )
        assert selected[slug]["target_normalization"] is None
        np.testing.assert_array_equal(splits["y_train_labels"], [0.0, 1.0, 2.0, 1.0])

    assert selected["none"]["cv"] == selected["ratio_meanyr"]["cv"]
    assert selected["none"]["shape_ceiling"] == selected["ratio_meanyr"]["shape_ceiling"]


@pytest.mark.parametrize(
    ("dist", "r_test", "gate_blend_test", "phi_test", "message"),
    [
        ("NegBin", None, None, None, "served r_test"),
        ("ZINB", None, SERVED_GATE, None, "served r_test"),
        ("ZINB", SERVED_R, None, None, "served gate_blend_test"),
        ("DPO", None, None, None, "served phi_test"),
    ],
)
def test_missing_served_count_shape_fails_loudly(
    dist,
    r_test,
    gate_blend_test,
    phi_test,
    message,
):
    with pytest.raises(ValueError, match=message):
        _stage_count_shape(
            dist,
            r_test=r_test,
            gate_blend_test=gate_blend_test,
            phi_test=phi_test,
        )
