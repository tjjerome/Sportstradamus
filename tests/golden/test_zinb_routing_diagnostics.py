"""Unit + CLI tests for the ZINB routing diagnostics script (Task B1.2).

The CLI test never touches the real cached training-data parquets. It builds a
synthetic 3-league x 3-market input with ``np.random.default_rng(42)``, writes
each cell to a tmp dir under the loader's filename pattern, points the loader at
that dir, runs the click CLI via ``CliRunner``, and asserts the output schema.

The focused unit tests pin the per-cell pure functions: a strongly
zero-inflated sample must route to ``mzinb`` and an underdispersed sample to
``cmp``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from sportstradamus.scripts.zinb_routing_diagnostics import (
    REQUIRED_COLUMNS,
    ROUTING_CHOICES,
    _compute_cell_stats,
    _route,
    _vuong_schwarz,
    _zero_inflation_indices,
    main,
)

# Markets per synthetic league. The NFL stems carry spaces so the loader's
# space -> hyphen filename mapping is exercised.
_SYNTH_MARKETS = {
    "NBA": ["aa", "bb", "cc"],
    "WNBA": ["dd", "ee", "ff"],
    "NFL": ["gg hh", "ii", "jj kk"],
}
_SYNTH_ROWS = 200
# Small bootstrap count keeps the CLI test fast; the production default is larger.
_TEST_BOOTSTRAP = 120


def _zero_heavy_counts(rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw zero-inflated NegBin counts: a structural-zero mask over NB draws."""
    base = rng.negative_binomial(2, 0.5, n)
    structural_zero = rng.random(n) < 0.35
    return np.where(structural_zero, 0, base).astype(float)


def _write_synthetic_cells(root) -> None:
    """Write one parquet per synthetic (league, market) under the loader pattern."""
    training_dir = root / "training_data"
    training_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for league, markets in _SYNTH_MARKETS.items():
        for market in markets:
            counts = _zero_heavy_counts(rng, _SYNTH_ROWS)
            stem = market.replace(" ", "-")
            frame = pd.DataFrame({"Result": counts})
            frame.to_parquet(training_dir / f"{league}_{stem}.parquet", engine="pyarrow")


def _patch_loader_and_cells(monkeypatch, data_root) -> None:
    """Point the loader at the synthetic tree and the cell set at synthetic markets."""
    monkeypatch.setattr(
        "sportstradamus.scripts.zinb_routing_diagnostics._training_data_root",
        lambda: data_root / "training_data",
    )
    monkeypatch.setattr(
        "sportstradamus.scripts.zinb_routing_diagnostics.zinb_cells",
        lambda: [(lg, m) for lg, ms in _SYNTH_MARKETS.items() for m in ms],
    )


def test_cli_writes_one_row_per_cell_with_required_schema(tmp_path, monkeypatch):
    """End-to-end CLI run on synthetic data: schema, finiteness, enum, row count."""
    data_root = tmp_path / "pkgdata"
    _write_synthetic_cells(data_root)
    out_dir = tmp_path / "out"
    _patch_loader_and_cells(monkeypatch, data_root)

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["--out-dir", str(out_dir), "--bootstrap", str(_TEST_BOOTSTRAP), "--seed", "7"],
    )
    assert result.exit_code == 0, result.output

    for league in _SYNTH_MARKETS:
        out_path = out_dir / f"{league}_diagnostics.parquet"
        assert out_path.is_file(), f"missing {out_path}"
        df = pd.read_parquet(out_path, engine="pyarrow")
        assert set(df.columns) == set(REQUIRED_COLUMNS)
        assert len(df) == len(_SYNTH_MARKETS[league])
        assert df["ziNB_index"].notna().all() and np.isfinite(df["ziNB_index"]).all()
        assert df["ziP_index"].notna().all() and np.isfinite(df["ziP_index"]).all()
        assert df["routing_recommendation"].isin(ROUTING_CHOICES).all()
        assert not df.duplicated(subset=["league", "market"]).any()


def test_cli_league_filter_writes_only_requested_league(tmp_path, monkeypatch):
    data_root = tmp_path / "pkgdata"
    _write_synthetic_cells(data_root)
    out_dir = tmp_path / "out"
    _patch_loader_and_cells(monkeypatch, data_root)

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["--league", "NBA", "--out-dir", str(out_dir), "--bootstrap", str(_TEST_BOOTSTRAP)],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "NBA_diagnostics.parquet").is_file()
    assert not (out_dir / "WNBA_diagnostics.parquet").is_file()
    assert not (out_dir / "NFL_diagnostics.parquet").is_file()


def test_zero_inflation_index_positive_for_zero_heavy_sample():
    """A zero-inflated NB sample has a clearly positive excess-zero ziNB index."""
    rng = np.random.default_rng(1)
    y = _zero_heavy_counts(rng, 4000)
    indices = _zero_inflation_indices(y, rng=np.random.default_rng(2), n_boot=200)
    assert indices.zinb_index > 0
    # The bootstrap CI for a strongly inflated sample should exclude zero.
    assert indices.zinb_ci_lo > 0


def test_route_strong_zero_inflation_goes_to_mzinb():
    """Strongly zero-inflated + overdispersed sample routes to mzinb."""
    rng = np.random.default_rng(3)
    y = _zero_heavy_counts(rng, 4000)
    row = _compute_cell_stats(
        league="NBA", market="x", y=y, rng=np.random.default_rng(4), n_boot=300
    )
    assert row["routing_recommendation"] == "mzinb"


def test_route_underdispersed_goes_to_cmp():
    """A near-constant (underdispersed) sample routes to cmp."""
    rng = np.random.default_rng(5)
    # Counts tightly clustered around 5 -> variance << mean -> var/mean < 1.3.
    y = rng.integers(4, 7, size=2000).astype(float)
    row = _compute_cell_stats(
        league="NBA", market="x", y=y, rng=np.random.default_rng(6), n_boot=200
    )
    assert row["var_mean_ratio"] < 1.3
    assert row["routing_recommendation"] == "cmp"


def test_route_is_pure_function_of_stats_row():
    """_route maps a stats dict to the enum without re-reading the sample."""
    underdispersed = {"var_mean_ratio": 1.1, "ziNB_ci_lo": -0.1, "ziNB_ci_hi": 0.1}
    assert _route(underdispersed) == "cmp"
    inflated = {"var_mean_ratio": 2.0, "ziNB_ci_lo": 0.05, "ziNB_ci_hi": 0.3}
    assert _route(inflated) == "mzinb"


def test_vuong_schwarz_is_finite_for_count_sample():
    rng = np.random.default_rng(7)
    y = _zero_heavy_counts(rng, 1500)
    stat = _vuong_schwarz(y)
    assert np.isfinite(stat)


def test_missing_parquet_is_skipped_not_fatal(tmp_path, monkeypatch):
    """A cell whose parquet is absent is warned + skipped, not crashed."""
    data_root = tmp_path / "pkgdata"
    _write_synthetic_cells(data_root)
    # Remove one cell's parquet so the loader must skip it.
    (data_root / "training_data" / "NBA_aa.parquet").unlink()
    out_dir = tmp_path / "out"
    _patch_loader_and_cells(monkeypatch, data_root)

    runner = CliRunner()
    result = runner.invoke(
        main, ["--league", "NBA", "--out-dir", str(out_dir), "--bootstrap", str(_TEST_BOOTSTRAP)]
    )
    assert result.exit_code == 0, result.output
    df = pd.read_parquet(out_dir / "NBA_diagnostics.parquet", engine="pyarrow")
    # Two markets remain (bb, cc); aa was skipped.
    assert len(df) == 2
    assert "aa" not in set(df["market"])


@pytest.mark.parametrize("choice", ROUTING_CHOICES)
def test_routing_choices_enum_is_the_four_documented_values(choice):
    assert choice in {"plain_nb", "hurdle_nb_ztnb", "mzinb", "cmp"}
