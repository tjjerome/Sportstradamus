"""Golden pins for the corr-summary artifact — the Lab heatmap's data source.

``_market_summary`` aggregates the per-team ``(team, market_a, market_b) -> R``
blocks the correlation pipeline already produces down to market-pair means;
``_write_corr_outputs`` writes the result beside the existing two parquets;
``load_corr_market_summary`` reads it back for the dashboard, returning an
empty-but-shaped frame for a league with no corr data yet.
"""

from __future__ import annotations

import importlib
import importlib.resources as pkg_resources

import pandas as pd
import pytest

from sportstradamus import data
from sportstradamus.dashboard.data import load_corr_market_summary
from sportstradamus.training.correlate import _market_summary, _write_corr_outputs

_SUMMARY_COLUMNS = ["market_a", "market_b", "rho_mean", "n_teams", "scope"]


def _synthetic_blocks() -> dict[str, pd.Series]:
    """Two teams, position-prefixed markets, one repeated pair across teams."""
    return {
        "CHI": pd.Series(
            {
                ("B1.AST", "B1.AST"): 1.0,
                ("B1.AST", "W1.PTS"): 0.4,
                ("W1.PTS", "W1.PTS"): 1.0,
            }
        ),
        "DAL": pd.Series({("B1.AST", "W1.PTS"): 0.6}),
    }


def test_market_summary_hand_computed_means_and_prefix_stripping() -> None:
    blocks = pd.concat(_synthetic_blocks())
    out = _market_summary(blocks, "same_team")

    assert list(out.columns) == _SUMMARY_COLUMNS
    out = out.set_index(["market_a", "market_b"])

    assert out.loc[("AST", "AST"), "rho_mean"] == pytest.approx(1.0)
    assert out.loc[("AST", "AST"), "n_teams"] == 1
    assert out.loc[("AST", "PTS"), "rho_mean"] == pytest.approx((0.4 + 0.6) / 2)
    assert out.loc[("AST", "PTS"), "n_teams"] == 2
    assert out.loc[("PTS", "PTS"), "rho_mean"] == pytest.approx(1.0)
    assert out.loc[("PTS", "PTS"), "n_teams"] == 1
    assert (out["scope"] == "same_team").all()


def test_market_summary_empty_series_returns_empty_shaped_frame() -> None:
    """An empty blocks Series (the real ``pd.concat({})``-free empty path) must not raise."""
    out = _market_summary(pd.Series(dtype="float64"), "opposing")
    assert list(out.columns) == _SUMMARY_COLUMNS
    assert out.empty


def test_market_summary_strips_multiple_position_prefix_formats() -> None:
    """Prefixes vary by rank/side (B1., W4., B2.) — split-on-dot strips all of them."""
    blocks = pd.concat(
        {"LAL": pd.Series({("W4.FG3A", "B2.PF"): 0.5})},
    )
    out = _market_summary(blocks, "opposing")
    assert set(out["market_a"]) == {"FG3A"}
    assert set(out["market_b"]) == {"PF"}


@pytest.fixture
def _isolated_correlate_data_dir(monkeypatch, tmp_path):
    """Redirect ``correlate``'s ``pkg_resources.files(data)`` root to ``tmp_path``.

    Mirrors ``tests/golden/test_correlate.py``'s ``_preserve_nba_correlate_outputs``:
    ``sportstradamus.training``'s package ``__init__`` re-exports the ``correlate``
    *function*, shadowing the submodule, so resolve the real module through
    import machinery and patch its ``pkg_resources`` object directly.
    """
    correlate_module = importlib.import_module("sportstradamus.training.correlate")
    real_files = correlate_module.pkg_resources.files

    def _fake_files(pkg):
        return tmp_path if pkg is data else real_files(pkg)

    monkeypatch.setattr(correlate_module.pkg_resources, "files", _fake_files)
    return tmp_path


def test_write_corr_outputs_writes_market_summary_alongside_existing_two(
    _isolated_correlate_data_dir,
) -> None:
    same_blocks = _synthetic_blocks()
    opposing_blocks = {"CHI": pd.Series({("B1.AST", "_OPP_W1.PTS"): 0.3})}

    _write_corr_outputs("NBA", same_blocks, opposing_blocks)

    league_dir = _isolated_correlate_data_dir / "leagues" / "nba"
    assert (league_dir / "corr_same_team.parquet").is_file()
    assert (league_dir / "corr_opposing.parquet").is_file()
    summary_path = league_dir / "corr_market_summary.parquet"
    assert summary_path.is_file()

    summary = pd.read_parquet(summary_path)
    assert list(summary.columns) == _SUMMARY_COLUMNS
    assert set(summary["scope"]) == {"same_team", "opposing"}
    assert not summary["market_a"].str.contains(".", regex=False).any()


def test_write_corr_outputs_empty_blocks_writes_empty_market_summary(
    _isolated_correlate_data_dir,
) -> None:
    """The already-handled empty-``same_team_blocks``/``opposing_blocks`` dict path."""
    _write_corr_outputs("NBA", {}, {})

    league_dir = _isolated_correlate_data_dir / "leagues" / "nba"
    summary = pd.read_parquet(league_dir / "corr_market_summary.parquet")
    assert list(summary.columns) == _SUMMARY_COLUMNS
    assert summary.empty


def test_write_corr_outputs_one_sided_blocks_only_summarizes_populated_scope(
    _isolated_correlate_data_dir,
) -> None:
    """Same-team populated, opposing empty — summary carries only the same_team scope."""
    _write_corr_outputs("NBA", _synthetic_blocks(), {})

    league_dir = _isolated_correlate_data_dir / "leagues" / "nba"
    summary = pd.read_parquet(league_dir / "corr_market_summary.parquet")
    assert set(summary["scope"]) == {"same_team"}


def _has_corr_market_summary(league: str) -> bool:
    league_dir = pkg_resources.files(data) / "leagues" / league
    return (league_dir / "corr_market_summary.parquet").is_file()


def test_load_corr_market_summary_absent_league_returns_empty_shaped_frame() -> None:
    """A league with no corr data on disk must not raise — empty shaped frame."""
    out = load_corr_market_summary("ABSENT")
    assert list(out.columns) == _SUMMARY_COLUMNS
    assert out.empty


@pytest.mark.skipif(
    not _has_corr_market_summary("nba"),
    reason="NBA corr_market_summary.parquet not present on disk",
)
def test_load_corr_market_summary_nba_has_expected_columns() -> None:
    out = load_corr_market_summary("NBA")
    assert list(out.columns) == _SUMMARY_COLUMNS
    assert not out.empty
    assert set(out["scope"]) <= {"same_team", "opposing"}
    assert not out["market_a"].str.contains(".", regex=False).any()
