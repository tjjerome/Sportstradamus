"""Deterministic full-rebuild, dependency, and manifest contracts."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import pytest

from sportstradamus.stats.mlb import StatsMLB
from sportstradamus.stats.model_dependencies import (
    DEPENDENCY_NAMESPACE,
    load_model_dependency,
)
from sportstradamus.training.data import trim_matrix
from sportstradamus.training.lineage import validate_matrix_manifest, write_matrix_manifest
from sportstradamus.training.pipeline import _step_load_matrix, _step_persist_matrix_and_comps


class _Stats:
    def __init__(self, matrix: pd.DataFrame):
        self.matrix = matrix
        self.gamelog = matrix[["Player", "Date", "Result"]].copy()
        self.model_dependency_root = None
        self.model_dependency_inventory = {}
        self.cutoffs = []

    def get_training_matrix(self, _market, cutoff):
        self.cutoffs.append(cutoff)
        return self.matrix.copy()


def _matrix() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Player": ["B", "A", "C", "D"],
            "Date": ["2025-01-02", "2025-01-01", "2025-01-03", "2025-01-04"],
            "Result": [2.0, 1.0, 3.0, 4.0],
            "Line": [1.5, 1.5, 2.5, 3.5],
            "Odds": [0.5] * 4,
            "EV": [2.0, 1.0, 3.0, 4.0],
            "Archived": [True] * 4,
            "DaysIntoSeason": [2, 1, 3, 4],
            "MeanYr": [2.0, 1.0, 3.0, 4.0],
        }
    )


def test_full_rebuild_ignores_cached_matrix_and_targets_quarantine(tmp_path):
    stats = _Stats(_matrix())
    output = tmp_path / "quarantine"
    dependency_root = tmp_path / "dependencies"

    rebuilt, path = _step_load_matrix(
        "NFL_passing-yards",
        pd.Timestamp("2024-01-01").date(),
        stats,
        "NFL",
        "passing yards",
        deterministic=False,
        force=False,
        need_model=False,
        full_rebuild=True,
        matrix_output=output,
        dependency_root=dependency_root,
        dependency_namespace="stage5-recovery-20260725-v1",
    )

    expected = _matrix()
    expected["Date"] = pd.to_datetime(expected["Date"])
    pd.testing.assert_frame_equal(rebuilt, expected)
    assert path == output / "NFL_passing-yards.parquet"
    assert stats.cutoffs == [pd.Timestamp("2024-01-01").date()]
    assert stats.model_dependency_root == dependency_root
    assert stats.model_dependency_namespace == "stage5-recovery-20260725-v1"


def test_full_rebuild_rejects_canonical_matrix_root():
    stats = _Stats(_matrix())
    canonical_root = Path("src/sportstradamus/data/training_data").resolve()

    with pytest.raises(ValueError, match="outside canonical training_data"):
        _step_load_matrix(
            "NFL_passing-yards",
            pd.Timestamp("2024-01-01").date(),
            stats,
            "NFL",
            "passing yards",
            deterministic=False,
            force=False,
            need_model=False,
            full_rebuild=True,
            matrix_output=canonical_root,
            dependency_root=Path("/tmp/dependencies"),
        )


def test_frozen_matrix_input_is_read_only_and_skips_rebuild(tmp_path):
    path = tmp_path / "NFL_attempts.parquet"
    expected = _matrix()
    expected.to_parquet(path)
    stats = _Stats(_matrix())
    write_matrix_manifest(
        path,
        expected,
        stats,
        league="NFL",
        market="attempts",
        cutoff_date="2024-01-01",
        repo_root=Path.cwd(),
    )
    before = path.read_bytes()

    loaded, loaded_path = _step_load_matrix(
        "NFL_attempts",
        pd.Timestamp("2024-01-01").date(),
        stats,
        "NFL",
        "attempts",
        deterministic=False,
        force=False,
        need_model=True,
        matrix_input=path,
    )

    expected["Date"] = pd.to_datetime(expected["Date"])
    pd.testing.assert_frame_equal(loaded, expected)
    assert loaded_path == path
    assert stats.cutoffs == []
    assert stats.snapshot_only_rebuild is True
    assert path.read_bytes() == before


def test_frozen_matrix_manifest_rejects_schema_drift(tmp_path):
    path = tmp_path / "NFL_attempts.parquet"
    matrix = _matrix()
    matrix.to_parquet(path)
    stats = _Stats(matrix)
    manifest_path = write_matrix_manifest(
        path,
        matrix,
        stats,
        league="NFL",
        market="attempts",
        cutoff_date="2024-01-01",
        repo_root=Path.cwd(),
    )
    manifest = manifest_path.read_text().replace('"row_count": 4', '"row_count": 5')
    manifest_path.write_text(manifest)

    with pytest.raises(ValueError, match="row_count"):
        validate_matrix_manifest(path, matrix)


def test_dependency_resolution_never_falls_back_to_serving_or_old_models(tmp_path):
    serving = tmp_path / "models"
    old = tmp_path / "old_models"
    serving.mkdir()
    old.mkdir()
    (serving / "NFL_attempts.mdl").write_bytes(b"serving")
    (old / "NFL_attempts.mdl").write_bytes(b"old")

    with pytest.raises(FileNotFoundError, match="not valid dependency registries"):
        load_model_dependency(tmp_path, "NFL", "attempts")

    namespace = tmp_path / DEPENDENCY_NAMESPACE
    namespace.mkdir()
    payload = {
        "dependency_identity": {
            "namespace": DEPENDENCY_NAMESPACE,
            "schema_version": 1,
            "league": "NFL",
            "market": "attempts",
            "training_cutoff": "2025-01-01",
            "matrix_sha256": "matrix-sha",
        },
        "distribution": "SkewNormal",
        "expected_columns": ["MeanYr"],
        "model_version": "test-v1",
        "trained_at": "2025-01-02T00:00:00+00:00",
    }
    (namespace / "NFL_attempts.mdl").write_bytes(pickle.dumps(payload))

    dependency = load_model_dependency(tmp_path, "NFL", "attempts")

    assert dependency.payload == payload
    assert dependency.path.parent == namespace


def test_dependency_resolution_supports_fresh_named_namespace(tmp_path):
    namespace_name = "stage5-recovery-20260725-v1"
    namespace = tmp_path / namespace_name
    namespace.mkdir()
    payload = {
        "dependency_identity": {
            "namespace": namespace_name,
            "schema_version": 1,
            "league": "NFL",
            "market": "attempts",
            "training_cutoff": "2026-01-01",
            "matrix_sha256": "fresh-matrix-sha",
        },
        "distribution": "SkewNormal",
        "expected_columns": ["MeanYr"],
        "model_version": "test-v2",
        "trained_at": "2026-07-25T00:00:00+00:00",
    }
    (namespace / "NFL_attempts.mdl").write_bytes(pickle.dumps(payload))

    dependency = load_model_dependency(
        tmp_path,
        "NFL",
        "attempts",
        namespace=namespace_name,
    )

    assert dependency.payload == payload
    assert dependency.path.parent == namespace


@pytest.mark.parametrize("namespace", ["", ".", "..", "parent/child", "../escape"])
def test_dependency_resolution_rejects_unsafe_namespace(tmp_path, namespace):
    with pytest.raises(ValueError, match="invalid dependency namespace"):
        load_model_dependency(tmp_path, "NFL", "attempts", namespace=namespace)


def test_mlb_hitter_rebuild_does_not_require_pitch_count_model(tmp_path):
    stats = StatsMLB.__new__(StatsMLB)
    stats.league = "MLB"
    stats.volume_stats = ["pitches thrown"]
    stats.model_dependency_root = tmp_path
    stats.model_dependency_inventory = {}

    stats.preflight_training_dependencies("total bases")

    with pytest.raises(RuntimeError, match=r"MLB_pitches-thrown\.mdl"):
        stats.preflight_training_dependencies("pitcher strikeouts")


def test_mlb_frozen_constructor_does_not_fetch_live_pitchers(monkeypatch):
    def fail_live_fetch():
        raise AssertionError("live probable-pitcher lookup must stay off the frozen path")

    monkeypatch.setattr("sportstradamus.stats.mlb.get_mlb_pitchers", fail_live_fetch)

    stats = StatsMLB(load_live_pitchers=False)

    assert stats.pitchers == {}


def test_trim_and_parquet_sha_are_reproducible(tmp_path):
    first = trim_matrix(_matrix().sample(frac=1, random_state=1), min_rows=0, seed=17)
    second = trim_matrix(_matrix().sample(frac=1, random_state=2), min_rows=0, seed=17)
    pd.testing.assert_frame_equal(first, second)

    stats = _Stats(_matrix())
    paths = [tmp_path / "one.parquet", tmp_path / "two.parquet"]
    manifests = []
    for path in paths:
        first.to_parquet(path, compression="zstd", index=True)
        manifests.append(
            write_matrix_manifest(
                path,
                first,
                stats,
                league="NFL",
                market="passing yards",
                cutoff_date="2024-01-01",
                repo_root=Path.cwd(),
            )
        )

    assert paths[0].read_bytes() == paths[1].read_bytes()
    assert '"matrix_sha256"' in manifests[0].read_text()


def test_full_rebuild_write_canonicalizes_feature_order(tmp_path):
    stats = _Stats(_matrix())
    outputs = [tmp_path / "one.parquet", tmp_path / "two.parquet"]
    matrices = [_matrix(), _matrix().reindex(columns=list(reversed(_matrix().columns)))]

    for matrix, output in zip(matrices, outputs, strict=True):
        persisted = _step_persist_matrix_and_comps(
            matrix,
            output,
            stats,
            deterministic=False,
            full_rebuild=True,
        )
        assert list(persisted.columns) == sorted(persisted.columns)

    assert outputs[0].read_bytes() == outputs[1].read_bytes()
