"""Characterization pin for ``nfl_fp_loader.load_one_year`` orchestration.

The real loader reads FantasyPoints CSVs from disk and joins ``nfl.import_ids()``
over the network, so the leaf loaders + the two downstream transforms
(``derive_comp_metrics``, ``_join_nfl_players``) are monkeypatched to controlled
synthetic frames / identities. What remains — and what this pins — is
``load_one_year``'s own orchestration: the per-file ``combine_first`` chain
(player files -> qb coverage -> team coverage broadcast -> PBP), the HB/FB -> RB
alias collapse *before* the comp-position filter, the POS -> position / G ->
player_game_count rename, and the two empty short-circuits. All patched names are
module globals, so the same patches drive both the pre- and post-refactor code.
"""

from __future__ import annotations

import pandas as pd

from sportstradamus.stats import nfl_fp_loader as m


def _fake_player_file(fp_dir, filename, prefix):
    if filename == "passingBasicExport.csv":
        return pd.DataFrame(
            {
                "POS": ["QB", "HB", "FB", "K"],
                "G": [17, 16, 10, 17],
                "Team": ["KC", "SF", "DAL", "BLT"],
                "pass_basic_YDS": [4000.0, 200.0, 5.0, 0.0],
            },
            index=["Patrick Mahomes", "Christian McCaffrey", "Some Fullback", "Justin Tucker"],
        )
    return None


def _fake_qb_cov(fp_dir):
    return pd.DataFrame({"qb_cov_MAN_pct": [0.3]}, index=["Patrick Mahomes"])


def _fake_team_cov(fp_dir):
    return pd.DataFrame({"_dummy": [1]}, index=["x"])


def _fake_broadcast(per_year, team_cov):
    out = per_year.copy()
    out["tm_cov_off_RATE"] = 0.5
    return out


def _fake_pbp(year):
    return pd.DataFrame({"pbp_EPA": [0.1]}, index=["Christian McCaffrey"])


def _install_fakes(monkeypatch, **overrides):
    fakes = {
        "_fp_dir": lambda y: "DUMMY",
        "_load_player_file": _fake_player_file,
        "_load_qb_coverage_file": _fake_qb_cov,
        "_load_team_coverage_file": _fake_team_cov,
        "_broadcast_team_columns": _fake_broadcast,
        "_load_pbp_named_by_player": _fake_pbp,
        "derive_comp_metrics": lambda df: df,
        "_join_nfl_players": lambda df: df,
    }
    fakes.update(overrides)
    for name, fn in fakes.items():
        monkeypatch.setattr(m, name, fn)


def test_load_one_year_orchestration(monkeypatch):
    _install_fakes(monkeypatch)

    out = m.load_one_year(2024)

    expected = pd.DataFrame(
        {
            "player_game_count": [16, 17, 10],
            "position": ["RB", "QB", "RB"],
            "Team": ["SF", "KC", "DAL"],
            "pass_basic_YDS": [200.0, 4000.0, 5.0],
            "pbp_EPA": [0.1, float("nan"), float("nan")],
            "qb_cov_MAN_pct": [float("nan"), 0.3, float("nan")],
            "tm_cov_off_RATE": [0.5, 0.5, 0.5],
        },
        index=["Christian McCaffrey", "Patrick Mahomes", "Some Fullback"],
    )
    pd.testing.assert_frame_equal(out, expected, check_dtype=False)


def test_load_one_year_empty_no_directory(monkeypatch):
    _install_fakes(monkeypatch, _fp_dir=lambda y: None)
    assert m.load_one_year(2024).empty


def test_load_one_year_empty_no_player_files(monkeypatch):
    _install_fakes(
        monkeypatch,
        _load_player_file=lambda fp_dir, filename, prefix: None,
        _load_qb_coverage_file=lambda fp_dir: None,
        _load_team_coverage_file=lambda fp_dir: None,
        _load_pbp_named_by_player=lambda year: None,
    )
    assert m.load_one_year(2024).empty
