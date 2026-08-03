"""Holdout-blind search mode: the evaluation frame and the cross-fit fold contract."""

import types

import numpy as np
import pandas as pd
import pytest

from sportstradamus.training import pipeline as pipe
from sportstradamus.training.structural_strategies import AFFINE_STRATEGY, TWO_PART_STRATEGY


class _StatData:
    @staticmethod
    def get_stat_columns(_market):
        return ["Home", "Feature"]


def _matrix(rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.date_range("2025-01-01", periods=rows),
            # Repeat players so a player-disjoint fold assignment is observable.
            "Player": [f"player-{i % 25}" for i in range(rows)],
            "Result": np.arange(rows, dtype=float),
            "Home": [i % 2 for i in range(rows)],
            "Feature": np.arange(rows, dtype=float),
            "Line": np.arange(rows, dtype=float) + 0.5,
            "Odds": np.full(rows, 0.55),
            "EV": np.arange(rows, dtype=float) + 1.0,
        }
    )


def test_holdout_blind_evaluates_the_validation_half_and_drops_the_holdout():
    matrix = _matrix(400)
    production = pipe._step_build_splits(matrix, _StatData(), "receiving yards")
    blind = pipe._step_build_splits(matrix, _StatData(), "receiving yards", holdout_blind=True)

    # The search frame IS the validation frame ...
    assert list(blind["X_test"].index) == list(blind["X_validation"].index)
    assert list(blind["X_test"].index) == list(production["X_validation"].index)
    # ... and the rows the ship gate scores are absent from the run entirely.
    assert not set(blind["X_test"].index) & set(production["X_test"].index)
    assert set(production["X_validation"].index) | set(production["X_test"].index) == set(
        blind["X_test"].index
    ) | set(production["X_test"].index)
    # Training data is untouched by the mode.
    assert list(blind["X_train"].index) == list(production["X_train"].index)


def test_calibration_folds_are_player_disjoint_and_deterministic():
    splits = pipe._step_build_splits(
        _matrix(400), _StatData(), "receiving yards", holdout_blind=True
    )
    folds = pipe._calibration_folds(splits)

    assert list(folds.index) == list(splits["X_validation"].index)
    assert folds.between(0, pipe._CALIBRATION_FOLDS - 1).all()
    assert len(folds.unique()) > 1
    players = splits["players_validation"]
    per_player = pd.DataFrame({"player": players, "fold": folds}).groupby("player")["fold"].nunique()
    assert (per_player == 1).all(), "a player's rows must never straddle two folds"
    pd.testing.assert_series_equal(folds, pipe._calibration_folds(splits))


def test_fold_view_rotates_validation_frames_and_leaves_training_alone():
    splits = pipe._step_build_splits(
        _matrix(400), _StatData(), "receiving yards", holdout_blind=True
    )
    folds = pipe._calibration_folds(splits)
    apply_index = folds.index[folds.eq(folds.iloc[0])]
    fit_index = folds.index[folds.ne(folds.iloc[0])]

    view = pipe._fold_splits_view(splits, fit_index, apply_index)

    for stem in pipe._ROTATED_SPLIT_STEMS:
        source = splits[f"{stem}_validation"]
        if source is None:
            assert view[f"{stem}_validation"] is None and view[f"{stem}_test"] is None
            continue
        assert list(view[f"{stem}_validation"].index) == list(fit_index)
        assert list(view[f"{stem}_test"].index) == list(apply_index)
    assert not set(view["X_validation"].index) & set(view["X_test"].index)
    assert view["X_train"] is splits["X_train"]
    assert view["y_train_labels"] is splits["y_train_labels"]


def test_concat_folds_restores_the_callers_row_order():
    target = pd.Index([7, 3, 9, 1])
    first, second = pd.Index([3, 1]), pd.Index([7, 9])
    order = first.append(second)
    parts = [np.array([30.0, 10.0]), np.array([70.0, 90.0])]

    merged = pipe._concat_folds(parts, order, target)

    assert list(merged) == [70.0, 30.0, 90.0, 10.0]
    assert pipe._concat_folds([None, None], order, target) is None


def test_concat_folds_merges_every_shape_the_calibrated_payload_carries():
    """Each concatenated key's real type, not just the 1-D array the first draft assumed.

    ``mix_blend_test`` rides as a dict of per-row arrays; passing it to ``np.asarray`` yields a
    zero-dimensional object array, which is what crashed every Mixture corner under --holdout-blind.
    """
    target = pd.Index([7, 3, 9, 1])
    first, second = pd.Index([3, 1]), pd.Index([7, 9])
    order = first.append(second)

    mixture = pipe._concat_folds(
        [
            {"w1": np.array([0.5, 0.6]), "loc1": np.array([1.0, 2.0])},
            {"w1": np.array([0.7, 0.8]), "loc1": np.array([9.0, 10.0])},
        ],
        order,
        target,
    )
    assert set(mixture) == {"w1", "loc1"}
    assert list(mixture["loc1"]) == [9.0, 1.0, 10.0, 2.0]

    # Served probabilities arrive as an (n, 2) matrix and must merge by row, not flatten.
    proba = pipe._concat_folds(
        [np.array([[0.7, 0.3], [0.9, 0.1]]), np.array([[0.4, 0.6], [0.2, 0.8]])], order, target
    )
    assert proba.shape == (4, 2)
    assert list(proba[:, 0]) == [0.4, 0.7, 0.2, 0.9]

    series = pipe._concat_folds(
        [pd.Series([30.0, 10.0], index=first), pd.Series([70.0, 90.0], index=second)], order, target
    )
    assert list(series) == [70.0, 30.0, 90.0, 10.0]


# --- structural rows -----------------------------------------------------------------------


def _two_part_calibrated(index: pd.Index, payload: str) -> dict:
    routes = pd.Series("high", index=index, dtype="object")
    return {
        "structural_routes": {"test": routes},
        "receiving_candidate_test": types.SimpleNamespace(
            candidate_over=np.linspace(0.4, 0.6, len(index))
        ),
        "receiving_candidate_calibration_payload": payload,
        "receiving_candidate_test_f0": np.full(len(index), 0.2),
        "receiving_candidate_test_role": routes,
        "receiving_candidate_test_position": pd.Series(3, index=index),
    }


def test_structural_rows_normalize_each_method_onto_one_row_shape():
    """All three paths return the same field set; a field the method never fits stays ``None``."""
    index = pd.Index([4, 9])
    two_part = pipe._structural_rows(
        TWO_PART_STRATEGY, _two_part_calibrated(index, "{payload}"), index
    )
    affine = pipe._structural_rows(
        AFFINE_STRATEGY,
        {
            "structural_routes": {"test": pd.Series("QB", index=index, dtype="object")},
            "rushing_candidate_test": types.SimpleNamespace(candidate_over=np.array([0.3, 0.7])),
            "structural_pit_maps": pd.Series(["{map}", "{map}"], index=index),
        },
        index,
    )
    base = pipe._structural_rows("none", {"pit_recal_blob": {"kind": "isotonic_pit"}}, index)
    plain = pipe._structural_rows("none", {}, index)

    for rows in (two_part, affine, base, plain):
        assert set(rows) == set(pipe.STRUCTURAL_ROW_FIELDS)
    assert list(two_part["two_part_calibration_by_row"]) == ["{payload}"] * 2
    assert two_part["pit_recal_by_row"] is None
    assert affine["two_part_f0"] is None and list(affine["prepool_over"]) == [0.3, 0.7]
    # A base cell's whole-CDF map rides the same field as the affine per-position maps.
    assert base["pit_recal_by_row"].iloc[0] == '{"kind": "isotonic_pit"}'
    assert all(value is None for value in plain.values())


def test_crossfit_structural_rows_restore_row_order_and_fail_on_a_half_fitted_field():
    """Each fold's payload lands on its own rows; a field only some folds fit is a hard error."""
    target = pd.Index([7, 3, 9, 1])
    first, second = pd.Index([3, 1]), pd.Index([7, 9])
    order = first.append(second)
    per_fold = [
        {"structural_rows": pipe._structural_rows(TWO_PART_STRATEGY, cal, index)}
        for index, cal in (
            (first, _two_part_calibrated(first, "fold-a")),
            (second, _two_part_calibrated(second, "fold-b")),
        )
    ]

    stitched = pipe._crossfit_structural_rows(per_fold, order, target)

    assert list(stitched["two_part_calibration_by_row"].index) == list(target)
    assert list(stitched["two_part_calibration_by_row"]) == ["fold-b", "fold-a", "fold-b", "fold-a"]
    assert stitched["pit_recal_by_row"] is None

    per_fold[1]["structural_rows"]["prepool_over"] = None
    with pytest.raises(ValueError, match="prepool_over collapsed on at least one fold"):
        pipe._crossfit_structural_rows(per_fold, order, target)


def test_calibration_fit_partitions_are_the_whole_frame_then_each_folds_complement():
    splits = pipe._step_build_splits(
        _matrix(400), _StatData(), "receiving yards", holdout_blind=True
    )
    folds = pipe._calibration_folds(splits)

    partitions = pipe._calibration_fit_partitions(splits)

    assert len(partitions) == 1 + len(folds.unique())
    assert list(partitions[0]) == list(folds.index)
    for fold, partition in zip(sorted(folds.unique()), partitions[1:], strict=True):
        assert set(partition) == set(folds.index[folds.ne(fold)])
