"""Behavioral tests for ``training/correlate.py``.

These exercise the residualization, sample-size shrinkage, metadata
side-car, and the ``--rebuild-correlations`` CLI flag added in the
methodology audit follow-up. They lean on synthetic gamelogs and a stub
Stats class so they can run without league-API credentials.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from sportstradamus import data
from sportstradamus.training.cli import meditate
from sportstradamus.training.correlate import (
    MIN_OVERLAP_FOR_FULL_WEIGHT,
    ROLLING_WINDOW_GAMES,
    _residualize_gamelog,
    _shrink_correlations,
    correlate,
)


def test_residualization_breaks_shared_trends() -> None:
    """A trend shared by two players is removed; only their independent noise remains."""
    rng = np.random.default_rng(0)
    n_games = 60
    dates = pd.date_range("2026-01-01", periods=n_games, freq="D").astype(str).tolist()
    trend = np.linspace(0.0, 50.0, n_games)
    a_noise = rng.standard_normal(n_games)
    b_noise = rng.standard_normal(n_games)
    a_values = trend + a_noise
    b_values = trend + b_noise

    raw_corr = float(np.corrcoef(a_values, b_values)[0, 1])
    assert raw_corr > 0.9, f"setup invalid: raw correlation should be near 1, got {raw_corr}"

    gamelog = pd.DataFrame(
        {
            "player": ["A"] * n_games + ["B"] * n_games,
            "date": dates + dates,
            "stat": np.concatenate([a_values, b_values]),
        }
    )

    out = _residualize_gamelog(gamelog, "player", "date", ["stat"])
    a_resid = out.loc[out.player == "A", "stat"].to_numpy()
    b_resid = out.loc[out.player == "B", "stat"].to_numpy()
    valid = ~np.isnan(a_resid) & ~np.isnan(b_resid)
    assert valid.sum() >= n_games - ROLLING_WINDOW_GAMES, "too many residuals are NaN"

    resid_corr = float(np.corrcoef(a_resid[valid], b_resid[valid])[0, 1])
    assert abs(resid_corr) < 0.4, (
        f"residual correlation should be much smaller than raw ({raw_corr:.2f}), got {resid_corr:.2f}"
    )


def test_low_overlap_pairs_shrink_toward_zero() -> None:
    """Pairs with few shared games get shrunk; pairs with full overlap pass through."""
    raw = 0.6
    corr = pd.DataFrame(
        [[1.0, raw, raw], [raw, 1.0, raw], [raw, raw, 1.0]],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
    )
    overlap = pd.DataFrame(
        [
            [100, 5, 100],
            [5, 100, 100],
            [100, 100, 100],
        ],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
    )
    shrunk = _shrink_correlations(corr, overlap)

    full_overlap = shrunk.loc["a", "c"]
    low_overlap = shrunk.loc["a", "b"]

    assert full_overlap == pytest.approx(raw, abs=1e-9)
    assert abs(low_overlap) < abs(full_overlap), "low-overlap pair should be closer to zero"
    expected_low = raw * (5 / MIN_OVERLAP_FOR_FULL_WEIGHT)
    assert low_overlap == pytest.approx(expected_low, abs=1e-9)


class _StubStats:
    """Minimal Stats-shaped object that ``correlate`` can iterate over.

    The gamelog is empty — enough to walk every code path and emit the
    metadata sidecar without needing league-API access.
    """

    def __init__(self) -> None:
        self.season_start = date(2026, 1, 1)
        self.gamelog = pd.DataFrame(
            columns=[
                "GAME_ID",
                "GAME_DATE",
                "PLAYER_NAME",
                "TEAM_ABBREVIATION",
                "OPP",
                "HOME",
                "POS",
                "MIN",
                "USG_PCT",
            ]
        )
        self.log_strings = {
            "game": "GAME_ID",
            "date": "GAME_DATE",
            "player": "PLAYER_NAME",
            "team": "TEAM_ABBREVIATION",
            "opponent": "OPP",
            "home": "HOME",
            "position": "POS",
            "usage": "MIN",
            "usage_sec": "USG_PCT",
        }
        self.playerProfile = pd.DataFrame()

    def load(self) -> None:
        pass

    def update(self) -> None:
        pass

    def update_player_comps(self) -> None:
        pass

    def profile_market(self, *_args, **_kwargs) -> None:
        pass


def _models_snapshot() -> dict[str, float]:
    models_dir = pkg_resources.files(data) / "models"
    if not Path(str(models_dir)).is_dir():
        return {}
    return {f.name: f.stat().st_mtime for f in Path(str(models_dir)).iterdir() if f.is_file()}


@pytest.fixture
def _preserve_nba_correlate_outputs():
    """Snapshot + restore the per-league correlate outputs around each test.

    These tests call ``correlate("NBA", stub, ...)``, which writes to the
    real package data directory at ``data/leagues/nba/``. Without this
    fixture the empty-gamelog stub would leave a ``cache_key`` of
    ``row_count=0`` / ``max_date=null`` behind, making the next real
    ``meditate`` run see a cache mismatch against its populated gamelog
    and rebuild from scratch — the test pollution the user hit on
    2026-05-29.
    """
    league_dir = Path(str(pkg_resources.files(data) / "leagues" / "nba"))
    targets = ["corr_metadata.json", "corr_same_team.parquet", "corr_opposing.parquet"]
    saved: dict[str, bytes | None] = {}
    for name in targets:
        path = league_dir / name
        saved[name] = path.read_bytes() if path.is_file() else None
    try:
        yield
    finally:
        for name, blob in saved.items():
            path = league_dir / name
            if blob is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(blob)


def test_metadata_written_with_required_keys(_preserve_nba_correlate_outputs) -> None:
    """correlate() emits the metadata side-car with the documented keys."""
    stub = _StubStats()
    metadata_path = pkg_resources.files(data) / "leagues" / "nba" / "corr_metadata.json"

    correlate("NBA", stub, force=True)

    assert Path(str(metadata_path)).is_file(), "metadata side-car missing"
    metadata = json.loads(Path(str(metadata_path)).read_text())

    required = {
        "league",
        "generated_at",
        "git_sha",
        "lookback_days",
        "rolling_window_games",
        "min_overlap_for_full_weight",
        "corr_magnitude_floor",
        "date_range",
        "total_team_game_observations",
        "per_team_observations",
    }
    missing = required - set(metadata)
    assert not missing, f"metadata missing keys: {missing}"
    assert metadata["league"] == "NBA"
    assert isinstance(metadata["per_team_observations"], dict)


def test_correlate_skips_when_cache_valid(_preserve_nba_correlate_outputs) -> None:
    """A second non-force call with unchanged inputs leaves the prior outputs untouched."""
    stub = _StubStats()
    metadata_path = Path(str(pkg_resources.files(data) / "leagues" / "nba" / "corr_metadata.json"))
    same_path = Path(str(pkg_resources.files(data) / "leagues" / "nba" / "corr_same_team.parquet"))

    correlate("NBA", stub, force=True)
    first_meta = json.loads(metadata_path.read_text())
    first_same_mtime = same_path.stat().st_mtime if same_path.is_file() else None

    # Second call with the same stub (same empty gamelog) and force=False must
    # take the skip path — the metadata timestamp and parquet mtime stay put.
    correlate("NBA", stub, force=False)
    second_meta = json.loads(metadata_path.read_text())
    second_same_mtime = same_path.stat().st_mtime if same_path.is_file() else None

    assert first_meta["generated_at"] == second_meta["generated_at"], (
        "skip path must not rewrite corr_metadata.json"
    )
    assert first_meta.get("cache_key") == second_meta.get("cache_key"), (
        "cache_key should round-trip unchanged"
    )
    assert first_same_mtime == second_same_mtime, "skip path must not rewrite the parquet"


def test_correlate_force_bypasses_skip(_preserve_nba_correlate_outputs, caplog) -> None:
    """``force=True`` always rebuilds, even when the cache_key matches."""
    import logging

    stub = _StubStats()

    correlate("NBA", stub, force=True)

    # correlate's logger sets propagate=False (helpers.get_logger), so attach
    # caplog's handler directly to it by its real name to capture INFO records.
    target = logging.getLogger("sportstradamus.cli.sportstradamus.training.correlate")
    target.addHandler(caplog.handler)
    caplog.clear()
    # Inputs are identical and the metadata is freshly valid — but force=True
    # must still take the full rebuild path. The skip path logs
    # "Correlating NBA... cache valid, skipped"; the rebuild path logs
    # only "Correlating NBA...".
    try:
        correlate("NBA", stub, force=True)
    finally:
        target.removeHandler(caplog.handler)
    messages = [record.getMessage() for record in caplog.records]

    assert any(msg == "Correlating NBA..." for msg in messages)
    assert not any("cache valid, skipped" in msg for msg in messages), (
        f"force=True took the skip path: {messages!r}"
    )


def test_rebuild_correlations_does_not_touch_models(monkeypatch) -> None:
    """``--rebuild-correlations`` runs the correlation step only — no model files written."""
    before = _models_snapshot()

    calls: list[tuple[str, bool]] = []

    def fake_correlate(league: str, stat_data, *, force: bool = False) -> None:
        calls.append((league, force))

    monkeypatch.setattr("sportstradamus.training.cli.StatsNBA", _StubStats)
    monkeypatch.setattr("sportstradamus.training.cli.StatsNFL", _StubStats)
    monkeypatch.setattr("sportstradamus.training.cli.StatsWNBA", _StubStats)
    monkeypatch.setattr("sportstradamus.training.cli.correlate", fake_correlate)

    sentinel = "training pipeline must not run when --rebuild-correlations is set"

    def fail_train_market(*_args, **_kwargs):
        raise AssertionError(sentinel)

    monkeypatch.setattr("sportstradamus.training.cli.train_market", fail_train_market)

    runner = CliRunner()
    result = runner.invoke(meditate, ["--rebuild-correlations", "--league", "NBA"])

    assert result.exit_code == 0, f"meditate exited {result.exit_code}: {result.output}"
    assert calls == [("NBA", False)], f"expected single NBA correlate call, got {calls}"
    assert _models_snapshot() == before, "model files were modified during --rebuild-correlations"
