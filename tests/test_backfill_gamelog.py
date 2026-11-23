"""Unit tests for ``scripts.backfill_gamelog`` — the gamelog backfill CLI.

The command is a shell: for the NHL around ``StatsNHL.backfill_season_sums``, or with
``--reports`` ``fill_report_columns``; for MLB around ``fill_pitch_columns``, a season at a time.
Load, fill the rows that lack the columns, write the gamelog back, report. These pin the call
order, the ``--season`` set-or-None contract, the ``write_gamelog`` arguments and the summary
line, with the stats classes, the fills and ``write_gamelog`` faked so nothing is fetched or
written.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from sportstradamus.scripts import backfill_gamelog
from sportstradamus.stats.mlb_pitch import PITCH_COLUMNS
from sportstradamus.stats.nhl import SEASON_SUM_COLUMNS
from sportstradamus.stats.nhl_reports import REPORT_COLUMNS


class _FakeStats:
    def __init__(self, calls: list, gamelog: pd.DataFrame):
        self.calls = calls
        self.gamelog = gamelog
        self.teamlog = pd.DataFrame({"team": ["BOS"]})
        self.players = {"Filled": {"position": "C"}}

    def load(self):
        self.calls.append("load")

    def backfill_season_sums(self, seasons=None):
        self.calls.append(("backfill", seasons))
        return 3


def _nhl(monkeypatch, calls: list) -> _FakeStats:
    fake = _FakeStats(
        calls,
        pd.DataFrame(
            {"playerName": ["Filled", "Lacking"]}
            | {col: [1.0, np.nan] for col in SEASON_SUM_COLUMNS}
            | {col: [np.nan, np.nan] for col in REPORT_COLUMNS}
        ),
    )
    monkeypatch.setattr(backfill_gamelog, "StatsNHL", lambda: fake)
    monkeypatch.setattr(backfill_gamelog, "write_gamelog", lambda *a: calls.append(("write", *a)))
    return fake


@pytest.mark.parametrize(
    ("args", "seasons"),
    [((), None), (("--season", "2024", "--season", "2025"), {2024, 2025})],
)
def test_nhl_loads_backfills_writes_then_reports(monkeypatch, args, seasons):
    calls = []
    fake = _nhl(monkeypatch, calls)

    result = CliRunner().invoke(backfill_gamelog.main, ["--league", "NHL", *args])

    assert result.exit_code == 0, result.output
    assert calls == [
        "load",
        ("backfill", seasons),
        ("write", "nhl", fake.gamelog, fake.teamlog, fake.players),
    ]
    assert result.output == "games re-fetched: 3; rows still lacking a season-sum column: 1\n"


@pytest.mark.parametrize(
    ("args", "seasons"),
    [(("--reports",), None), (("--reports", "--season", "2025"), {2025})],
)
def test_nhl_reports_fills_the_report_columns_and_re_parses_no_game(monkeypatch, args, seasons):
    calls = []
    fake = _nhl(monkeypatch, calls)

    def fill(gamelog, seasons=None):
        calls.append(("fill", seasons))
        return gamelog.assign(**{col: [1.0, np.nan] for col in REPORT_COLUMNS})

    monkeypatch.setattr(backfill_gamelog, "fill_report_columns", fill)

    result = CliRunner().invoke(backfill_gamelog.main, ["--league", "NHL", *args])

    assert result.exit_code == 0, result.output
    assert calls == [
        "load",
        ("fill", seasons),
        ("write", "nhl", fake.gamelog, fake.teamlog, fake.players),
    ]
    assert result.output == "rows still lacking a report column: 1\n"


@pytest.mark.parametrize(
    ("args", "seasons"),
    [((), [2024, 2025]), (("--season", "2025"), [2025])],
)
def test_mlb_fills_and_writes_a_season_at_a_time(monkeypatch, args, seasons):
    calls = []
    fake = _FakeStats(calls, pd.DataFrame({"gameDate": ["2025-05-01", "2024-05-01"]}))
    monkeypatch.setattr(backfill_gamelog, "StatsMLB", lambda: fake)
    monkeypatch.setattr(
        backfill_gamelog, "write_gamelog", lambda league, *a: calls.append(("write", league))
    )

    def fill(gamelog, seasons):
        calls.append(("fill", seasons))
        # The 2024 game stays lacking, as one the feed fails would.
        return gamelog.assign(**{col: [1.0, np.nan] for col in PITCH_COLUMNS})

    monkeypatch.setattr(backfill_gamelog, "fill_pitch_columns", fill)

    result = CliRunner().invoke(backfill_gamelog.main, ["--league", "MLB", *args])

    assert result.exit_code == 0, result.output
    assert calls == [
        "load",
        *(call for year in seasons for call in (("fill", {year}), ("write", "mlb"))),
    ]
    assert result.output == "rows still lacking a pitch column: 1\n"


def test_reports_is_refused_for_mlb():
    result = CliRunner().invoke(backfill_gamelog.main, ["--league", "MLB", "--reports"])

    assert result.exit_code == 2
    assert "--reports fills NHL columns" in result.output
