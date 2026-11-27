"""Golden pins for the corr-summary artifact — the Lab heatmap's data source.

``_league_pair_summary`` pools every team's games into one market-pair table for
the ``same_team`` and ``opposing`` scopes; ``_correlate_teams`` captures same-slot
(one player's own) cross-market pairs pre-shrink, pre-floor, and
``_same_player_summary`` pools them n-weighted in Fisher z with a single
credibility shrink at the pooled n — the ``same_player`` scope the combo-pricing
kernel pools rho from; ``_write_corr_outputs`` writes it all beside the two
per-team parquets; ``load_corr_market_summary`` reads it back for the dashboard,
returning an empty-but-shaped frame for a league with no corr data yet.
"""

from __future__ import annotations

import importlib
import importlib.resources as pkg_resources

import numpy as np
import pandas as pd
import pytest

from sportstradamus import data
from sportstradamus.dashboard.data import load_corr_market_summary
from sportstradamus.training.correlate import (
    _correlate_teams,
    _league_pair_summary,
    _same_player_summary,
    _write_corr_outputs,
)

_SUMMARY_COLUMNS = ["market_a", "market_b", "rho_mean", "n_teams", "scope", "n_teams_distinct"]

# dashboard/data.py's empty-league fallback frame predates n_teams_distinct;
# only files _write_corr_outputs writes carry the full _SUMMARY_COLUMNS.
_LOADER_FALLBACK_COLUMNS = ["market_a", "market_b", "rho_mean", "n_teams", "scope"]


_UP = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
_DOWN = _UP[::-1]


def _league_matrix() -> pd.DataFrame:
    """Sixty team-games of two teams: six games ten times over, every column rising or falling.

    A pair of full columns correlates at +1 or -1 on sixty shared games. ``W2.AST`` is
    missing from the last two games of the six, so its pairs rest on forty. Every pair
    holds the thirty games a pooled value needs to count in full.
    """
    six_games = pd.DataFrame(
        {
            "TEAM": ["ATL"] * 3 + ["BOS"] * 3,
            "B1.PTS": _UP,
            "B1.AST": _DOWN,
            "W1.AST": _UP,
            "W2.AST": [6.0, 5.0, 4.0, 3.0, np.nan, np.nan],
            "C1.REB": _UP,
            "C2.REB": _DOWN,
            "_OPP_B1.PTS": _DOWN,
        }
    )
    return pd.concat([six_games] * 10, ignore_index=True)


def test_league_pair_summary_pools_slot_pairs_weighted_by_shared_games() -> None:
    out = _league_pair_summary(_league_matrix())

    assert list(out.columns) == _SUMMARY_COLUMNS
    assert (out["n_teams_distinct"] == 2).all()
    keyed = out.set_index(["scope", "market_a", "market_b"])
    # A teammate pair pools over the slots that carry the two markets, each slot pair
    # weighted by its shared games: (PTS, AST) is B1-W1 at +1 on sixty games and B1-W2 at
    # -1 on forty, which an unweighted mean would cancel. B1's own PTS and AST are one
    # player's pair and stay out. A pair pooling under the floor is dropped: (PTS, REB),
    # (AST, REB) and opposing (REB, PTS) all cancel exactly.
    assert keyed["rho_mean"].to_dict() == pytest.approx(
        {
            ("same_team", "PTS", "AST"): 0.2,
            ("same_team", "AST", "PTS"): 0.2,
            ("same_team", "AST", "AST"): -3 / 7,
            ("same_team", "REB", "REB"): -1.0,
            # market_a is the team's own market, market_b the opponent's.
            ("opposing", "PTS", "PTS"): -1.0,
            ("opposing", "AST", "PTS"): 0.25,
        }
    )
    # n_teams counts the slot pairs pooled, each order of a same-market pair once.
    assert keyed["n_teams"].to_dict() == {
        ("same_team", "PTS", "AST"): 2,
        ("same_team", "AST", "PTS"): 2,
        ("same_team", "AST", "AST"): 6,
        ("same_team", "REB", "REB"): 2,
        ("opposing", "PTS", "PTS"): 1,
        ("opposing", "AST", "PTS"): 3,
    }


def test_league_pair_summary_ranks_every_teams_games_together() -> None:
    """A pair that falls within each team and rises across them reads as rising."""
    six_games = pd.DataFrame(
        {
            "TEAM": ["ATL"] * 3 + ["BOS"] * 3,
            "B1.PTS": [1.0, 2.0, 3.0, 11.0, 12.0, 13.0],
            "W1.AST": [3.0, 2.0, 1.0, 13.0, 12.0, 11.0],
        }
    )
    # Five times over, for the thirty shared games a pooled value needs to count in full.
    matrix = pd.concat([six_games] * 5, ignore_index=True)
    out = _league_pair_summary(matrix).set_index(["scope", "market_a", "market_b"])
    # Ranks 1..6 against 3,2,1,6,5,4: Spearman 1 - 6 * 16 / (6 * 35), then the remap.
    stacked = 2 * np.sin(np.pi / 6 * (1 - 96 / 210))
    assert out.loc[("same_team", "PTS", "AST"), "rho_mean"] == pytest.approx(stacked)


def test_league_pair_summary_trusts_a_pair_by_the_games_behind_it() -> None:
    """A pooled value shrinks by its shared games over thirty; the floor reads it unshrunk."""
    matrix = pd.DataFrame(
        {
            "TEAM": ["ATL"] * 3 + ["BOS"] * 3,
            "B1.PTS": _UP,
            "W1.AST": [2.0, 5.0, 1.0, 6.0, 4.0, 3.0],
            "C1.REB": _UP,
            "C2.REB": _DOWN,
        }
    )
    out = _league_pair_summary(matrix).set_index(["scope", "market_a", "market_b"])["rho_mean"]
    # Ranks 1..6 against 2,5,1,6,4,3: Spearman 1 - 6 * 28 / (6 * 35) = 0.2, then the remap.
    pooled = 2 * np.sin(np.pi / 6 * 0.2)
    # Six games of the thirty: a fifth of the pooled value. That fifth sits under the
    # floor, which the pooled value itself clears, so the pair stays.
    assert out[("same_team", "PTS", "AST")] == pytest.approx(pooled / 5)
    # A same-market pair meets its slot pair in both orders: C1-C2 share six games, not twelve.
    assert out[("same_team", "REB", "REB")] == pytest.approx(-1.0 / 5)


def test_league_pair_summary_reads_the_same_in_either_market_order() -> None:
    """With both teams' rows of every game present, (a, b) and (b, a) agree in each scope."""
    rng = np.random.default_rng(0)
    pace = rng.normal(size=40)
    home = pd.DataFrame(
        {
            "B1.PTS": pace + 0.5 * rng.normal(size=40),
            "W1.AST": pace + 0.5 * rng.normal(size=40),
            "_OPP_B1.PTS": 0.5 * rng.normal(size=40) - pace,
            "_OPP_W1.AST": 0.5 * rng.normal(size=40) - pace,
        }
    )
    away = home.rename(
        columns={
            "B1.PTS": "_OPP_B1.PTS",
            "W1.AST": "_OPP_W1.AST",
            "_OPP_B1.PTS": "B1.PTS",
            "_OPP_W1.AST": "W1.AST",
        }
    )
    matrix = pd.concat([home.assign(TEAM="ATL"), away.assign(TEAM="BOS")], ignore_index=True)

    out = _league_pair_summary(matrix).set_index(["scope", "market_a", "market_b"])["rho_mean"]

    assert set(out.index) == {
        ("same_team", "PTS", "AST"),
        ("same_team", "AST", "PTS"),
        ("opposing", "PTS", "PTS"),
        ("opposing", "PTS", "AST"),
        ("opposing", "AST", "PTS"),
        ("opposing", "AST", "AST"),
    }
    mirrored = out.copy()
    mirrored.index = mirrored.index.swaplevel("market_a", "market_b")
    assert out.to_dict() == pytest.approx(mirrored.to_dict())


def test_league_pair_summary_of_an_empty_matrix_is_empty_and_shaped() -> None:
    """A league with no team-game records yet (``correlate`` on an empty log) must not raise."""
    out = _league_pair_summary(pd.DataFrame())
    assert list(out.columns) == _SUMMARY_COLUMNS
    assert out.empty


def _synthetic_blocks() -> dict[str, pd.Series]:
    """Two teams' stored per-team blocks, one pair repeated across teams."""
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


def test_write_corr_outputs_summarizes_the_matrix_beside_the_two_per_team_files(
    _isolated_correlate_data_dir,
) -> None:
    same_blocks = _synthetic_blocks()
    opposing_blocks = {"CHI": pd.Series({("B1.AST", "_OPP_W1.PTS"): 0.3})}
    matrix = _league_matrix()

    _write_corr_outputs("NBA", same_blocks, opposing_blocks, pd.DataFrame(), matrix)

    league_dir = _isolated_correlate_data_dir / "leagues" / "nba"
    assert (league_dir / "corr_same_team.parquet").is_file()
    assert (league_dir / "corr_opposing.parquet").is_file()
    summary = pd.read_parquet(league_dir / "corr_market_summary.parquet")
    # The team scopes are the league-wide pool of the matrix, whatever the per-team
    # blocks hold; no captured same-slot pairs were passed, so no same_player rows.
    pd.testing.assert_frame_equal(summary, _league_pair_summary(matrix))
    assert set(summary["scope"]) == {"same_team", "opposing"}


def test_write_corr_outputs_empty_inputs_write_an_empty_market_summary(
    _isolated_correlate_data_dir,
) -> None:
    """``correlate`` on a league with no records: empty blocks, pairs and matrix."""
    _write_corr_outputs("NBA", {}, {}, pd.DataFrame(), pd.DataFrame())

    league_dir = _isolated_correlate_data_dir / "leagues" / "nba"
    summary = pd.read_parquet(league_dir / "corr_market_summary.parquet")
    assert list(summary.columns) == _SUMMARY_COLUMNS
    assert summary.empty


def _slot_paired_blocks() -> dict[str, pd.Series]:
    """Three teams, slots B1/B2, markets singles/walks/runs.

    Carries one R=1.0 self-pair, one cross-player (B1 x B2) pair, and one
    reversed-ordering pair. Feeds the stored per-team file's pin only — the
    summary pools from the matrix and from the pre-shrink pairs
    ``_correlate_teams`` captures, not from these stored blocks.
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


def _same_player_pairs() -> pd.DataFrame:
    """Captured-shape (team, market_a, market_b, rho, n) rows for the pool pins."""
    return pd.DataFrame(
        [
            {"team": "ATL", "market_a": "singles", "market_b": "walks", "rho": 0.3, "n": 20},
            {"team": "BOS", "market_a": "singles", "market_b": "walks", "rho": 0.5, "n": 40},
            {"team": "BOS", "market_a": "runs", "market_b": "walks", "rho": 0.02, "n": 50},
        ]
    )


def test_write_corr_outputs_same_player_scope_pools_fisher_z(
    _isolated_correlate_data_dir,
) -> None:
    _write_corr_outputs("MLB", _slot_paired_blocks(), {}, _same_player_pairs(), _league_matrix())

    league_dir = _isolated_correlate_data_dir / "leagues" / "mlb"
    summary = pd.read_parquet(league_dir / "corr_market_summary.parquet")
    assert list(summary.columns) == _SUMMARY_COLUMNS
    assert set(summary["scope"]) == {"same_team", "opposing", "same_player"}

    sp = summary.loc[summary["scope"] == "same_player"].set_index(["market_a", "market_b"])
    # n-weighted Fisher-z pool of the unshrunk pair correlations; the pooled
    # n = 60 >= 30, so the single credibility shrink is 1.
    expected = np.tanh((20 * np.arctanh(0.3) + 40 * np.arctanh(0.5)) / 60)
    assert sp.loc[("singles", "walks"), "rho_mean"] == pytest.approx(expected)
    assert sp.loc[("singles", "walks"), "n_teams"] == 2
    assert sp.loc[("singles", "walks"), "n_teams_distinct"] == 2
    # Unshrunk pooled |rho| = 0.02 <= CORR_MAGNITUDE_FLOOR — the pooled floor
    # still drops genuinely weak market pairs.
    assert ("runs", "walks") not in sp.index
    assert len(sp) == 1


def test_same_player_summary_weak_overlap_pair_survives_pooled_floor() -> None:
    """A pair the old post-shrink floor deleted now survives.

    Old artifact path: stored R = 0.3 * min(1, 4/30) = 0.04 <= 0.05, so the
    (team, slot) row was floored away before the summary ever saw it. Pooled
    honestly, the floor tests the unshrunk pooled rho (0.3 > 0.05) and the
    reported rho_mean carries the single credibility shrink at the pooled n:
    0.3 * (4/30) = 0.04.
    """
    weak = pd.DataFrame(
        [{"team": "ATL", "market_a": "singles", "market_b": "runs", "rho": 0.3, "n": 4}]
    )
    out = _same_player_summary(weak).set_index(["market_a", "market_b"])
    assert out.loc[("singles", "runs"), "rho_mean"] == pytest.approx(0.3 * 4 / 30)
    assert out.loc[("singles", "runs"), "n_teams"] == 1
    assert out.loc[("singles", "runs"), "n_teams_distinct"] == 1


def test_correlate_teams_captures_same_slot_pairs_unshrunk_with_overlap() -> None:
    """Capture is pre-shrink, pre-floor: equal slot prefix, different market,
    own side only — no cross-slot (B1 x B2), no _OPP_, no self-pairs."""
    matrix = pd.DataFrame(
        {
            "TEAM": ["ATL"] * 4,
            "B1.hits": [1.0, 2.0, 3.0, 4.0],
            "B1.walks": [2.0, 1.0, 3.0, 4.0],
            "B2.hits": [4.0, 3.0, 2.0, 1.0],
            "_OPP_B1.hits": [1.0, 2.0, 3.0, 4.0],
            "_OPP_B1.walks": [1.5, 2.5, 3.5, 4.5],
        }
    )
    _, _, pairs, _ = _correlate_teams(matrix)

    keyed = pairs.set_index(["market_a", "market_b"])
    assert set(keyed.index) == {("hits", "walks"), ("walks", "hits")}
    # Ranks 1,2,3,4 vs 2,1,3,4 -> Spearman 0.8; the captured rho is the plain
    # 2*sin(pi*rho_s/6) remap with no min(1, n/30) shrink factor applied.
    assert keyed.loc[("hits", "walks"), "rho"] == pytest.approx(2 * np.sin(np.pi * 0.8 / 6))
    assert keyed.loc[("hits", "walks"), "n"] == 4
    assert (pairs["team"] == "ATL").all()


def test_write_corr_outputs_stores_the_per_team_blocks_as_given(
    _isolated_correlate_data_dir,
) -> None:
    blocks = _slot_paired_blocks()
    _write_corr_outputs("MLB", blocks, {}, _same_player_pairs(), pd.DataFrame())

    league_dir = _isolated_correlate_data_dir / "leagues" / "mlb"
    stored = pd.read_parquet(league_dir / "corr_same_team.parquet")["R"]
    expected = pd.concat(blocks)
    assert list(stored.index) == list(expected.index)
    assert list(stored.to_numpy()) == list(expected.to_numpy())

    # The blocks' self-pairs and cross-player rows no longer reach the summary: with no
    # matrix rows its only scope is the captured same-slot pool.
    summary = pd.read_parquet(league_dir / "corr_market_summary.parquet")
    assert set(summary["scope"]) == {"same_player"}


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
