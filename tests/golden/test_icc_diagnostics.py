"""Unit + CLI tests for the SkewNormal ICC diagnostics script (Stage A1).

The CLI tests never touch the real cached training-data parquets. They build
synthetic ``(Player, Date, Result)`` frames with ``np.random.default_rng(42)``,
write each cell under the loader's filename pattern, point the loader + cell set
at the synthetic tree, run the click CLI via ``CliRunner``, and assert the output
schema and the pre-registered routing cutoffs end-to-end.

The focused unit tests pin the pure helpers: the transform escalation
(raw -> log1p -> rank), the participation filter's exclusion of structural-zero
player-seasons (the NFL position-confound fix), and the ICC -> routing map.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from sportstradamus.scripts.icc_diagnostics import (
    REQUIRED_COLUMNS,
    ROUTING_CHOICES,
    TRANSFORM_CHOICES,
    _choose_transform,
    _compute_cell_stats,
    _route,
    main,
)

# Two markets per synthetic league: one engineered high-ICC, one low-ICC.
_SYNTH_MARKETS = {"NBA": ["hi", "lo"], "WNBA": ["hi", "lo"], "NFL": ["hi", "lo"]}
# All synthetic dates fall in one month so every league maps to a single season
# (the (Player, season) group-by reduces to per-player), keeping the engineered
# ICC analytic and independent of the per-league season-boundary rule.
_SEASON_MONTH_BASE = pd.Timestamp("2025-11-01")
_GAMES_PER_PLAYER = 30
_N_PLAYERS = 30


def _player_season_frame(
    rng: np.random.Generator,
    league: str,
    n_players: int,
    games: int,
    base_loc: float,
    base_scale: float,
    within_scale: float,
) -> pd.DataFrame:
    """Build a ``(Player, Date, Result)`` frame: per-player base mean + within noise.

    ``base_scale`` is the between-player spread (variance of player base means);
    ``within_scale`` is the within-player game-to-game spread. Large base_scale /
    small within_scale gives high ICC; the reverse gives low ICC.
    """
    players: list[str] = []
    dates: list[pd.Timestamp] = []
    results: list[float] = []
    game_dates = list(_SEASON_MONTH_BASE + pd.to_timedelta(np.arange(games), unit="D"))
    for i in range(n_players):
        base = rng.normal(base_loc, base_scale)
        vals = np.clip(rng.normal(base, within_scale, games), 0.0, None)
        players.extend([f"{league}_p{i}"] * games)
        dates.extend(game_dates)
        results.extend(vals.tolist())
    return pd.DataFrame({"Player": players, "Date": dates, "Result": results})


def _write_frame(root: Path, league: str, market: str, frame: pd.DataFrame) -> None:
    """Write one synthetic cell under the loader's ``{league}_{stem}.parquet`` pattern."""
    training_dir = root / "training_data"
    training_dir.mkdir(parents=True, exist_ok=True)
    stem = market.replace(" ", "-")
    frame.to_parquet(training_dir / f"{league}_{stem}.parquet", engine="pyarrow")


def _write_synthetic_cells(root: Path) -> None:
    """Write a high-ICC ``hi`` and low-ICC ``lo`` cell for each synthetic league."""
    rng = np.random.default_rng(42)
    for league in _SYNTH_MARKETS:
        hi = _player_season_frame(rng, league, _N_PLAYERS, _GAMES_PER_PLAYER, 40.0, 12.0, 2.0)
        lo = _player_season_frame(rng, league, _N_PLAYERS, _GAMES_PER_PLAYER, 40.0, 0.5, 10.0)
        _write_frame(root, league, "hi", hi)
        _write_frame(root, league, "lo", lo)


def _patch_loader_and_cells(monkeypatch, data_root: Path, cells=None) -> None:
    """Point the loader at the synthetic tree and the cell set at synthetic markets."""
    monkeypatch.setattr(
        "sportstradamus.scripts.icc_diagnostics._training_data_root",
        lambda: data_root / "training_data",
    )
    if cells is None:
        cells = [(lg, m) for lg, ms in _SYNTH_MARKETS.items() for m in ms]
    monkeypatch.setattr(
        "sportstradamus.scripts.icc_diagnostics.skewnormal_cells",
        lambda: cells,
    )


def test_cli_writes_one_row_per_cell_with_required_schema(tmp_path, monkeypatch):
    """End-to-end CLI run on synthetic data: schema, finiteness, cutoffs, row count."""
    data_root = tmp_path / "pkgdata"
    _write_synthetic_cells(data_root)
    out_dir = tmp_path / "out"
    _patch_loader_and_cells(monkeypatch, data_root)

    result = CliRunner().invoke(main, ["--out-dir", str(out_dir)])
    assert result.exit_code == 0, result.output

    for league in _SYNTH_MARKETS:
        out_path = out_dir / f"{league}_icc.parquet"
        assert out_path.is_file(), f"missing {out_path}"
        df = pd.read_parquet(out_path, engine="pyarrow")
        assert set(df.columns) == set(REQUIRED_COLUMNS)
        assert len(df) == len(_SYNTH_MARKETS[league])
        assert df["icc"].notna().all()
        assert ((df["icc"] >= 0.0) & (df["icc"] <= 1.0)).all()
        assert df["routing_recommendation"].isin(ROUTING_CHOICES).all()
        assert not df.duplicated(subset=["league", "market"]).any()
        by_market = df.set_index("market")
        # The engineered cutoffs must round-trip through the CLI.
        assert by_market.loc["hi", "routing_recommendation"] == "eb_centering"
        assert by_market.loc["lo", "routing_recommendation"] == "tail_extension"


def test_cli_league_filter_writes_only_requested_league(tmp_path, monkeypatch):
    data_root = tmp_path / "pkgdata"
    _write_synthetic_cells(data_root)
    out_dir = tmp_path / "out"
    _patch_loader_and_cells(monkeypatch, data_root)

    result = CliRunner().invoke(main, ["--league", "NBA", "--out-dir", str(out_dir)])
    assert result.exit_code == 0, result.output
    assert (out_dir / "NBA_icc.parquet").is_file()
    assert not (out_dir / "WNBA_icc.parquet").is_file()
    assert not (out_dir / "NFL_icc.parquet").is_file()


def test_choose_transform_raw_for_symmetric_sample():
    rng = np.random.default_rng(1)
    y = rng.normal(100.0, 15.0, 8000)
    assert _choose_transform(y) == "raw"


def test_choose_transform_log1p_for_right_skewed_sample():
    rng = np.random.default_rng(2)
    y = rng.lognormal(0.0, 1.2, 8000)
    assert _choose_transform(y) == "log1p"


def test_choose_transform_rank_for_extreme_sample():
    """A bulk + far-extreme cluster stays skewed even under log1p -> rank fallback."""
    rng = np.random.default_rng(3)
    bulk = np.abs(rng.normal(1.0, 0.3, 7920))
    extreme = rng.normal(5000.0, 100.0, 80)
    y = np.concatenate([bulk, extreme])
    assert _choose_transform(y) == "rank"


@pytest.mark.parametrize("label", TRANSFORM_CHOICES)
def test_transform_choices_enum(label):
    assert label in {"raw", "log1p", "rank"}


def test_participation_filter_excludes_structural_zero_player_seasons():
    """All-zero player-seasons (e.g. non-QBs in passing-yards) are dropped, not ICC-inflated."""
    rng = np.random.default_rng(7)
    participating = _player_season_frame(rng, "NFL", 10, 20, 200.0, 60.0, 30.0)
    game_dates = list(_SEASON_MONTH_BASE + pd.to_timedelta(np.arange(20), unit="D"))
    zeros = pd.DataFrame(
        {
            "Player": [f"NFL_z{i}" for i in range(10) for _ in range(20)],
            "Date": game_dates * 10,
            "Result": [0.0] * 200,
        }
    )
    frame = pd.concat([participating, zeros], ignore_index=True)

    row = _compute_cell_stats(
        league="NFL", market="passing", frame=frame, min_games=2, min_nonzero_frac=0.5
    )
    assert row is not None
    assert row["n_player_seasons"] == 10
    assert row["n_player_seasons_excluded"] == 10
    assert row["n_players"] == 10
    assert 0.0 <= row["icc"] <= 1.0


def test_singleton_player_seasons_skipped_not_fatal(tmp_path, monkeypatch):
    """A market where every player has one game (within-variance undefined) is skipped."""
    data_root = tmp_path / "pkgdata"
    rng = np.random.default_rng(9)
    hi = _player_season_frame(rng, "NBA", _N_PLAYERS, _GAMES_PER_PLAYER, 40.0, 12.0, 2.0)
    solo = pd.DataFrame(
        {
            "Player": [f"NBA_s{i}" for i in range(40)],
            "Date": list(_SEASON_MONTH_BASE + pd.to_timedelta(np.arange(40), unit="D")),
            "Result": rng.normal(15.0, 5.0, 40),
        }
    )
    _write_frame(data_root, "NBA", "hi", hi)
    _write_frame(data_root, "NBA", "solo", solo)
    out_dir = tmp_path / "out"
    _patch_loader_and_cells(monkeypatch, data_root, cells=[("NBA", "hi"), ("NBA", "solo")])

    result = CliRunner().invoke(main, ["--league", "NBA", "--out-dir", str(out_dir)])
    assert result.exit_code == 0, result.output
    df = pd.read_parquet(out_dir / "NBA_icc.parquet", engine="pyarrow")
    assert set(df["market"]) == {"hi"}


def test_route_is_pure_function_of_icc():
    assert _route({"icc": 0.7}) == "eb_centering"
    assert _route({"icc": 0.5}) == "eb_centering"
    assert _route({"icc": 0.1}) == "tail_extension"
    assert _route({"icc": 0.3}) == "tail_extension"
    assert _route({"icc": 0.4}) == "ambiguous"


@pytest.mark.parametrize("choice", ROUTING_CHOICES)
def test_routing_choices_enum_is_the_three_documented_values(choice):
    assert choice in {"eb_centering", "tail_extension", "ambiguous"}


def test_missing_parquet_is_skipped_not_fatal(tmp_path, monkeypatch):
    data_root = tmp_path / "pkgdata"
    _write_synthetic_cells(data_root)
    (data_root / "training_data" / "NBA_hi.parquet").unlink()
    out_dir = tmp_path / "out"
    _patch_loader_and_cells(monkeypatch, data_root)

    result = CliRunner().invoke(main, ["--league", "NBA", "--out-dir", str(out_dir)])
    assert result.exit_code == 0, result.output
    df = pd.read_parquet(out_dir / "NBA_icc.parquet", engine="pyarrow")
    assert set(df["market"]) == {"lo"}
