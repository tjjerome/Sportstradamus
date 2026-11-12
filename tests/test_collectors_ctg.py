"""Fake-mode tests for the Cleaning the Glass parser, dispatch guard, catalog and roster seed."""

from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest
from click.testing import CliRunner

from sportstradamus.collectors import tabular
from sportstradamus.collectors.catalog import load_catalog
from sportstradamus.collectors.cleaningtheglass.source import (
    CATALOG_PATH,
    CTG_SOURCE,
    _roster_seed,
)
from sportstradamus.collectors.cli import build_source_cli
from sportstradamus.collectors.transport import CollectorAuthError
from sportstradamus.stats import nba

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


# Three rows of the 2026-27 player table with the lines CTG's own CSV export gave for them:
# Adebayo's 364.67 minutes read 364 (floored), Rollins' AST% of 26.25 reads 26.3 (half up).
EXPORT_PLAYERS = [
    {
        "name": "Precious Achiuwa",
        "age": 27.11841204654346,
        "team_name": "SAC",
        "pos_category": "big",
        "seconds_played": 16985.0,
        "mpg": 25.7,
        "usage": 0.15298507462686567,
        "psa": 111.11111111111111,
        "ast_perc": 0.06382978723404255,
        "ast_usage_ratio": 0.4172288531395952,
        "tov_perc": 0.12682926829268293,
        "ptile_usage": 0.3670886075949367,
        "ptile_psa": 0.26582278481012656,
        "ptile_ast_perc": 0.16455696202531644,
        "ptile_ast_usage_ratio": 0.11392405063291139,
        "ptile_tov_perc": 0.5822784810126582,
    },
    {
        "name": "Bam Adebayo",
        "age": 29.28952772073922,
        "team_name": "MIA",
        "pos_category": "big",
        "seconds_played": 21880.0,
        "mpg": 33.2,
        "usage": 0.20809248554913296,
        "psa": 117.36526946107784,
        "ast_perc": 0.14634146341463414,
        "ast_usage_ratio": 0.7032520325203251,
        "tov_perc": 0.1,
        "ptile_usage": 0.8227848101265823,
        "ptile_psa": 0.43037974683544306,
        "ptile_ast_perc": 0.7088607594936709,
        "ptile_ast_usage_ratio": 0.5822784810126582,
        "ptile_tov_perc": 0.810126582278481,
    },
    {
        "name": "Ryan Rollins",
        "age": 24.331279945242983,
        "team_name": "MIL",
        "pos_category": "point",
        "seconds_played": 14938.0,
        "mpg": 22.6,
        "usage": 0.25677966101694916,
        "psa": 100.84745762711864,
        "ast_perc": 0.2625,
        "ast_usage_ratio": 1.0222772277227723,
        "tov_perc": 0.15181518151815182,
        "ptile_usage": 0.5428571428571428,
        "ptile_psa": 0.17142857142857143,
        "ptile_ast_perc": 0.5428571428571428,
        "ptile_ast_usage_ratio": 0.45714285714285713,
        "ptile_tov_perc": 0.34285714285714286,
    },
]
EXPORT_HEADER = (
    "Player,Age,Team,Pos,MIN,MPG,Usage Rank,Usage,PSA Rank,PSA,AST% Rank,AST%,"
    "AST:Usg Rank,AST:Usg,TOV% Rank,TOV%"
)
EXPORT_LINES = [
    "Precious Achiuwa,27.1,SAC,Big,283,25.7,37,15.3%,27,111.1,16,6.4%,11,0.42,58,12.7%",
    "Bam Adebayo,29.3,MIA,Big,364,33.2,82,20.8%,43,117.4,71,14.6%,58,0.70,81,10.0%",
    "Ryan Rollins,24.3,MIL,Point,248,22.6,54,25.7%,17,100.8,54,26.3%,46,1.02,34,15.2%",
]
STALE_SEED = "Player,Team\nStale Export,XXX\n"


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
    source = replace(
        CTG_SOURCE,
        make_client=lambda _auth, _sleep: _Client(PAGE),
        auth_fields=None,
        after_write=None,
    )
    result = CliRunner().invoke(
        build_source_cli(source), ["run", "--season", "2026", "--date", "2026-01-15"]
    )
    assert result.exit_code == 0, result.output
    snapshot = tmp_path / "player_data/NBA/cleaningtheglass/2026/2026-01-15"
    assert len(pd.read_parquet(snapshot / "stats.parquet")) == len(PLAYERS)
    assert len(pd.read_parquet(snapshot / "onoff.parquet")) == len(ONOFF)


def test_roster_seed_reproduces_ctgs_own_export_lines():
    seed = _roster_seed(pd.DataFrame(EXPORT_PLAYERS))
    assert ",".join(seed.columns) == EXPORT_HEADER
    assert seed.to_csv(index=False, header=False).splitlines() == EXPORT_LINES


@pytest.fixture
def seed_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tabular, "PLAYER_DATA_BASE", tmp_path / "player_data")
    monkeypatch.setattr(tabular, "TEAM_DATA_BASE", tmp_path / "team_data")
    nba_dir = tmp_path / "player_data/NBA"
    nba_dir.mkdir(parents=True)
    monkeypatch.setattr(nba, "_ROSTER_SEED_DIR", nba_dir)
    return nba_dir


def _run(args: list[str]):
    source = replace(
        CTG_SOURCE,
        make_client=lambda _auth, _sleep: _Client(_page(ONOFF, EXPORT_PLAYERS)),
        auth_fields=None,
    )
    return CliRunner().invoke(
        build_source_cli(source), ["run", "--season", "2026", "--date", "2026-11-11", *args]
    )


def test_player_stats_run_rewrites_the_roster_seed_the_next_update_reads(seed_dir):
    seed = seed_dir / "nba_players_2026-27.csv"
    seed.write_text(STALE_SEED)

    result = _run([])

    assert result.exit_code == 0, result.output
    assert nba._roster_seed_path("2026-27") == seed
    assert seed.read_text().splitlines() == [EXPORT_HEADER, *EXPORT_LINES]
    assert not list(seed_dir.glob("*.tmp"))


def test_onoff_only_run_leaves_the_roster_seed_alone(seed_dir):
    seed = seed_dir / "nba_players_2026-27.csv"
    seed.write_text(STALE_SEED)

    result = _run(["--only", "player_onoff"])

    assert result.exit_code == 0, result.output
    assert seed.read_text() == STALE_SEED
