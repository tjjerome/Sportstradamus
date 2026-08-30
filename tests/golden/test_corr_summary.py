"""Golden pins for the corr-summary artifact — the Lab heatmap's data source.

``_market_summary`` aggregates the per-team ``(team, market_a, market_b) -> R``
blocks the correlation pipeline already produces down to market-pair means;
``_write_corr_outputs`` writes the result beside the existing two parquets —
including the ``same_player`` scope (same-team pairs whose two sides share a
slot prefix, self-pairs dropped) the combo-pricing kernel pools rho from;
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

_SUMMARY_COLUMNS = ["market_a", "market_b", "rho_mean", "n_teams", "scope", "n_teams_distinct"]

# dashboard/data.py's empty-league fallback frame predates n_teams_distinct;
# only files _write_corr_outputs writes carry the full _SUMMARY_COLUMNS.
_LOADER_FALLBACK_COLUMNS = ["market_a", "market_b", "rho_mean", "n_teams", "scope"]


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
    assert out.loc[("AST", "AST"), "n_teams_distinct"] == 1
    assert out.loc[("AST", "PTS"), "rho_mean"] == pytest.approx((0.4 + 0.6) / 2)
    assert out.loc[("AST", "PTS"), "n_teams"] == 2
    assert out.loc[("AST", "PTS"), "n_teams_distinct"] == 2
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
    # _synthetic_blocks holds no equal-slot cross-market pair, so no
    # same_player rows may appear.
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


def _slot_paired_blocks() -> dict[str, pd.Series]:
    """Three teams, slots B1/B2, markets singles/walks/runs.

    Carries one R=1.0 self-pair, one cross-player (B1 x B2) pair, and one
    reversed-ordering pair so the same-player slot filter and the ordered
    pooling it inherits from ``_market_summary`` are all exercised.
    """
    return {
        "ATL": pd.Series(
            {
                ("B1.singles", "B1.singles"): 1.0,
                ("B1.singles", "B1.walks"): 0.3,
                ("B1.singles", "B2.walks"): 0.9,
            }
        ),
        "BOS": pd.Series(
            {
                ("B1.singles", "B1.walks"): 0.5,
                ("B2.runs", "B2.walks"): 0.2,
            }
        ),
        "CHI": pd.Series(
            {
                ("B1.singles", "B1.walks"): 0.3,
                ("B2.singles", "B2.walks"): 0.1,
                ("B2.walks", "B2.singles"): 0.1,
            }
        ),
    }


def test_write_corr_outputs_same_player_scope_filters_and_pools(
    _isolated_correlate_data_dir,
) -> None:
    _write_corr_outputs("MLB", _slot_paired_blocks(), {})

    league_dir = _isolated_correlate_data_dir / "leagues" / "mlb"
    summary = pd.read_parquet(league_dir / "corr_market_summary.parquet")
    assert list(summary.columns) == _SUMMARY_COLUMNS
    assert set(summary["scope"]) == {"same_team", "same_player"}

    sp = summary.loc[summary["scope"] == "same_player"].set_index(["market_a", "market_b"])
    assert ("singles", "singles") not in sp.index  # self-pairs dropped
    # Cross-player B1.singles x B2.walks (0.9) stays out of the pool: with it
    # the mean would be 0.42, not 0.30.
    assert sp.loc[("singles", "walks"), "rho_mean"] == pytest.approx(0.3)
    assert sp.loc[("singles", "walks"), "n_teams"] == 4
    # CHI contributes twice (B1 and B2) — distinct teams stay 3.
    assert sp.loc[("singles", "walks"), "n_teams_distinct"] == 3
    # Pooling stays ordered exactly like _market_summary: the reversed CHI row
    # forms its own (walks, singles) cell.
    assert sp.loc[("walks", "singles"), "rho_mean"] == pytest.approx(0.1)
    assert sp.loc[("runs", "walks"), "rho_mean"] == pytest.approx(0.2)
    assert sp.loc[("runs", "walks"), "n_teams"] == 1
    assert len(sp) == 3


def test_write_corr_outputs_same_player_scope_leaves_same_team_output_unchanged(
    _isolated_correlate_data_dir,
) -> None:
    blocks = _slot_paired_blocks()
    _write_corr_outputs("MLB", blocks, {})

    league_dir = _isolated_correlate_data_dir / "leagues" / "mlb"
    stored = pd.read_parquet(league_dir / "corr_same_team.parquet")["R"]
    expected = pd.concat(blocks)
    assert list(stored.index) == list(expected.index)
    assert list(stored.to_numpy()) == list(expected.to_numpy())

    summary = pd.read_parquet(league_dir / "corr_market_summary.parquet")
    st = summary.loc[summary["scope"] == "same_team"].set_index(["market_a", "market_b"])
    # Same-team pooling still includes cross-player and self-pair rows.
    assert st.loc[("singles", "walks"), "rho_mean"] == pytest.approx(0.42)
    assert st.loc[("singles", "walks"), "n_teams"] == 5
    assert st.loc[("singles", "singles"), "rho_mean"] == pytest.approx(1.0)


def _has_corr_market_summary(league: str) -> bool:
    league_dir = pkg_resources.files(data) / "leagues" / league
    return (league_dir / "corr_market_summary.parquet").is_file()


def test_load_corr_market_summary_absent_league_returns_empty_shaped_frame() -> None:
    """A league with no corr data on disk must not raise — empty shaped frame."""
    out = load_corr_market_summary("ABSENT")
    assert list(out.columns) == _LOADER_FALLBACK_COLUMNS
    assert out.empty


@pytest.mark.skipif(
    not _has_corr_market_summary("nba"),
    reason="NBA corr_market_summary.parquet not present on disk",
)
def test_load_corr_market_summary_nba_has_expected_columns() -> None:
    out = load_corr_market_summary("NBA")
    # Tolerates both the pre-n_teams_distinct parquet already on disk and the
    # regenerated one the next correlate run writes.
    assert set(_LOADER_FALLBACK_COLUMNS) <= set(out.columns)
    assert not out.empty
    assert set(out["scope"]) <= {"same_team", "opposing", "same_player"}
    assert not out["market_a"].str.contains(".", regex=False).any()
