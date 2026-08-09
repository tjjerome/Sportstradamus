"""Fake-mode tests for the date-stamped collector sources (CTG / savant).

Covers the shared tabular parser, the player/team dated routing, and a full
``run`` through the date CLI with a stubbed fetch (no network) — asserting a
parquet lands at the ``{season}/{date}/`` path.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from sportstradamus.collectors import tabular
from sportstradamus.collectors.baseballsavant.cli import savant_fetch
from sportstradamus.collectors.catalog import EndpointSpec, save_catalog
from sportstradamus.collectors.cleaningtheglass.cli import ctg_fetch
from sportstradamus.collectors.cli import Source, build_source_cli
from sportstradamus.collectors.tabular import dated_path_for, parse_tabular_response


def _spec(name: str) -> EndpointSpec:
    return EndpointSpec(name=name, url="https://x/tool", output_subdir=name)


def test_parse_dataframe_passthrough():
    df = pd.DataFrame({"a": [1, 2]})
    assert parse_tabular_response(df) is df


def test_parse_csv_text():
    out = parse_tabular_response("a,b\n1,2\n3,4\n")
    assert list(out.columns) == ["a", "b"] and len(out) == 2


def test_parse_list_of_dicts():
    out = parse_tabular_response([{"a": 1}, {"a": 2}])
    assert list(out["a"]) == [1, 2]


def test_parse_json_rows_key():
    out = parse_tabular_response({"rows": [{"a": 1}], "meta": "ignored"})
    assert len(out) == 1


def test_parse_empty_shapes():
    assert parse_tabular_response("").empty
    assert parse_tabular_response({"nope": 1}).empty
    assert parse_tabular_response(42).empty


def test_dated_path_routes_player_and_team():
    player = dated_path_for(
        _spec("player_usage"), season=2025, date="2026-01-15", league="NBA", source_dir="ctg"
    )
    team = dated_path_for(
        _spec("team_four_factors"), season=2025, date="2026-01-15", league="NBA", source_dir="ctg"
    )
    assert player.parts[-4:] == ("ctg", "2025", "2026-01-15", "usage.parquet")
    assert "player_data" in str(player) and "team_data" in str(team)


def test_dated_path_rejects_unprefixed_name():
    with pytest.raises(ValueError, match="player_<tool> or team_<tool>"):
        dated_path_for(
            _spec("usage"), season=2025, date="2026-01-15", league="NBA", source_dir="ctg"
        )


def test_dated_run_writes_parquet(tmp_path, monkeypatch):
    monkeypatch.setattr(tabular, "PLAYER_DATA_BASE", tmp_path / "player_data")
    monkeypatch.setattr(tabular, "TEAM_DATA_BASE", tmp_path / "team_data")
    catalog = tmp_path / "cat.json"
    save_catalog([_spec("player_usage")], catalog)

    source = Source(
        name="test-fetch",
        help="test",
        catalog_path=catalog,
        make_client=lambda _resolved, _sleep: object(),
        default_context=lambda season, _week: {"season": season or 2025, "week": 0},
        path_for=partial(dated_path_for, league="NBA", source_dir="testsrc"),
        dispatch=lambda _client, _spec, **_kw: [{"player": "x", "usg": 0.3}],
        transform=parse_tabular_response,
        report_prefix="test",
        period_kind="date",
    )
    result = CliRunner().invoke(
        build_source_cli(source),
        ["run", "--season", "2025", "--date", "2026-01-15", "--catalog", str(catalog)],
    )
    assert result.exit_code == 0, result.output
    landed = tmp_path / "player_data/NBA/testsrc/2025/2026-01-15/usage.parquet"
    assert landed.is_file()
    assert list(pd.read_parquet(landed)["usg"]) == [0.3]


def test_ctg_has_refresh_auth_savant_does_not():
    ctg_cmds = set(ctg_fetch.commands)
    savant_cmds = set(savant_fetch.commands)
    assert "refresh-auth" in ctg_cmds  # cookie-authed
    assert "refresh-auth" not in savant_cmds  # public
    # Neither cumulative source backfills (deferred) or exposes week/mode.
    assert "backfill" not in ctg_cmds and "backfill" not in savant_cmds
