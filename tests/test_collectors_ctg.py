"""Fake-mode tests for the Cleaning the Glass page parser, dispatch guard and shipped catalog."""

from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest
from click.testing import CliRunner

from sportstradamus.collectors import tabular
from sportstradamus.collectors.catalog import load_catalog
from sportstradamus.collectors.cleaningtheglass.source import CATALOG_PATH, CTG_SOURCE
from sportstradamus.collectors.cli import build_source_cli
from sportstradamus.collectors.transport import CollectorAuthError

SPECS = {spec.name: spec for spec in load_catalog(CATALOG_PATH)}

PLAYERS = [
    {"name": "A One", "usage": 0.25, "ptile_usage": 0.9},
    {"name": "B Two", "usage": 0.18, "ptile_usage": 0.4},
]
# CTG serves several rows per player here: position 0 is on court, 6 the on-off difference.
ONOFF = [
    {"first_name": "A", "last_name": "One", "position": 0, "ptile_diff_oer": 0.5},
    {"first_name": "A", "last_name": "One", "position": 6, "ptile_diff_oer": 0.5},
    {"first_name": "B", "last_name": "Two", "position": 0, "ptile_diff_oer": 0.2},
]


def _page(*tables: list[dict]) -> str:
    args = ",\n".join(json.dumps(table) for table in tables)
    return f"<html><script>window.filter = vuePlayers(\n{args}\n);</script></html>"


# The on/off table comes first so a parser that takes tables by position fails.
PAGE = _page(ONOFF, PLAYERS)


class _Client:
    def __init__(self, body: str) -> None:
        self.body = body
        self.calls: list[dict | None] = []

    def get(self, url, params=None, headers=None, accept="json"):
        self.calls.append(params)
        return self.body


def test_each_tool_gets_its_own_table_whatever_the_order():
    stats = CTG_SOURCE.transform(PAGE, spec=SPECS["player_stats"], season=2026, week=0)
    onoff = CTG_SOURCE.transform(PAGE, spec=SPECS["player_onoff"], season=2026, week=0)
    assert list(stats["name"]) == ["A One", "B Two"]
    assert list(onoff["position"]) == [0, 6, 0]


def test_every_catalog_tool_parses_and_routes_to_player_data():
    assert SPECS
    for spec in SPECS.values():
        assert not CTG_SOURCE.transform(PAGE, spec=spec, season=2026, week=0).empty
        path = CTG_SOURCE.path_for(spec, season=2026, date="2026-01-15")
        assert path.parts[-5:-2] == ("NBA", "cleaningtheglass", "2026")
        assert "player_data" in path.parts


def test_page_without_a_matching_table_fails_loud():
    with pytest.raises(ValueError, match="ptile_usage"):
        CTG_SOURCE.transform(_page([{"unrelated": 1}]), spec=SPECS["player_stats"])


def test_page_without_the_data_call_raises_collector_auth_error():
    login_page = "<html>Please log in to see Cleaning the Glass stats.</html>"
    with pytest.raises(CollectorAuthError):
        CTG_SOURCE.dispatch(_Client(login_page), SPECS["player_stats"], season=2026)


def test_dispatch_pins_the_season_and_season_type_in_the_query():
    client = _Client(PAGE)
    assert CTG_SOURCE.dispatch(client, SPECS["player_stats"], season=2026) == PAGE
    assert client.calls == [{"season": "2026", "seasontype": "regseason"}]


def test_run_writes_both_tools_at_the_dated_path(tmp_path, monkeypatch):
    monkeypatch.setattr(tabular, "PLAYER_DATA_BASE", tmp_path / "player_data")
    monkeypatch.setattr(tabular, "TEAM_DATA_BASE", tmp_path / "team_data")
    source = replace(CTG_SOURCE, make_client=lambda _auth, _sleep: _Client(PAGE), auth_fields=None)
    result = CliRunner().invoke(
        build_source_cli(source), ["run", "--season", "2026", "--date", "2026-01-15"]
    )
    assert result.exit_code == 0, result.output
    snapshot = tmp_path / "player_data/NBA/cleaningtheglass/2026/2026-01-15"
    assert len(pd.read_parquet(snapshot / "stats.parquet")) == len(PLAYERS)
    assert len(pd.read_parquet(snapshot / "onoff.parquet")) == len(ONOFF)
