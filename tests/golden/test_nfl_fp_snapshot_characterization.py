"""Characterization pins for the shared snapshot-I/O surface of the two FP
weekly modules: ``nfl_fp_weekly`` (player-grain) and ``nfl_fp_team_weekly``
(team-grain).

The two modules carry seven byte-for-byte-equivalent loader functions that
differ only in three per-module values: the base directory (``player_data``
vs ``team_data``), the ``FILE_KINDS`` map, and the error-message label.
Extracting those loaders into a single ``SnapshotStore`` must preserve every
observable pinned here — in particular the three differentiators, since a
mis-wired store would silently read the wrong directory or report the wrong
kinds. Pinned against the committed 2022 fixtures.
"""

from __future__ import annotations

import pandas as pd
import pytest

from sportstradamus.stats import nfl_fp_team_weekly as team
from sportstradamus.stats import nfl_fp_weekly as player

_SEASON = 2022
_WEEKS_2022 = list(range(1, 37))


def test_available_snapshots_player() -> None:
    assert player.available_snapshots(_SEASON) == _WEEKS_2022


def test_available_snapshots_team() -> None:
    assert team.available_snapshots(_SEASON) == _WEEKS_2022


def test_snapshot_dir_present_and_absent() -> None:
    assert player.snapshot_dir(_SEASON, 1) is not None
    assert player.snapshot_dir(2099, 1) is None
    assert team.snapshot_dir(_SEASON, 1) is not None
    assert team.snapshot_dir(2099, 1) is None


def test_load_snapshot_player_shape() -> None:
    # Exact shape guards the player base-dir + FILE_KINDS wiring: a store
    # pointed at team_data would not return this frame.
    df = player.load_snapshot(_SEASON, 1, "efficiency")
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1141, 73)


def test_load_snapshot_team_shape() -> None:
    df = team.load_snapshot(_SEASON, 1, "coverage_matrix")
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (32, 41)


def test_load_snapshot_unknown_kind_label() -> None:
    with pytest.raises(ValueError, match="unknown FP weekly file_kind"):
        player.load_snapshot(_SEASON, 1, "__nope__")
    with pytest.raises(ValueError, match="unknown FP team weekly file_kind"):
        team.load_snapshot(_SEASON, 1, "__nope__")


def test_load_window_concatenates() -> None:
    one = player.load_snapshot(_SEASON, 1, "efficiency")
    win = player.load_window(_SEASON, 1, 2, "efficiency")
    assert isinstance(win, pd.DataFrame)
    assert len(win) >= len(one)


def test_load_through_player() -> None:
    assert isinstance(player.load_through(_SEASON, 2, "efficiency"), pd.DataFrame)


def test_load_all_snapshots_player() -> None:
    allsnap = player.load_all_snapshots(_SEASON, "efficiency")
    assert isinstance(allsnap, dict)
    assert all(isinstance(k, int) for k in allsnap)
    assert all(isinstance(v, pd.DataFrame) for v in allsnap.values())


def test_snapshot_inventory_columns_and_rows() -> None:
    inv_p = player.snapshot_inventory([_SEASON])
    assert list(inv_p.columns) == ["season", "snapshot_week", *player.FILE_KINDS]
    assert len(inv_p) == 36

    inv_t = team.snapshot_inventory([_SEASON])
    assert list(inv_t.columns) == ["season", "snapshot_week", *team.FILE_KINDS]
    assert len(inv_t) == 36


def test_load_window_or_empty_matches_load_window() -> None:
    # On the happy path the swallow-and-normalize wrapper returns load_window's
    # frame unchanged; the aggregate recipes consumed this contract via the old
    # per-module ``_load_kind`` helpers.
    direct = player.load_window(_SEASON, 1, 2, "efficiency")
    safe = player.load_window_or_empty(_SEASON, 1, 2, "efficiency")
    assert isinstance(safe, pd.DataFrame)
    pd.testing.assert_frame_equal(safe, direct)


def test_load_window_or_empty_missing_season_is_empty() -> None:
    assert player.load_window(2099, 1, 2, "efficiency") is None
    safe = player.load_window_or_empty(2099, 1, 2, "efficiency")
    assert isinstance(safe, pd.DataFrame)
    assert safe.empty


def test_load_window_or_empty_inverted_range_is_empty() -> None:
    with pytest.raises(ValueError):
        player.load_window(_SEASON, 5, 1, "efficiency")
    assert player.load_window_or_empty(_SEASON, 5, 1, "efficiency").empty


def test_load_window_or_empty_unknown_kind_swallowed() -> None:
    # The aggregate path never raises on a bad kind: load_window's ValueError is
    # logged (tagged with the store label) and normalized to an empty frame.
    safe = team.load_window_or_empty(_SEASON, 1, 2, "__nope__")
    assert isinstance(safe, pd.DataFrame)
    assert safe.empty
