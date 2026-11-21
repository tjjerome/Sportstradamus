"""Unit tests for ``scripts.backfill_nhl_gamelog`` — the NHL gamelog backfill CLI.

The command is a shell around ``StatsNHL.backfill_season_sums``, or with ``--reports``
``fill_report_columns``: load, fill the rows that lack the columns, write the gamelog back,
report. These pin the call order, the ``--season`` set-or-None contract, the ``write_gamelog``
arguments and the summary line, with ``StatsNHL``, the report fill and ``write_gamelog`` faked
so nothing is fetched or written.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from sportstradamus.scripts import backfill_nhl_gamelog
from sportstradamus.stats.nhl import SEASON_SUM_COLUMNS
from sportstradamus.stats.nhl_reports import REPORT_COLUMNS


class _FakeStatsNHL:
    def __init__(self, calls: list):
        self.calls = calls
        self.gamelog = pd.DataFrame(
            {"playerName": ["Filled", "Lacking"]}
            | {col: [1.0, np.nan] for col in SEASON_SUM_COLUMNS}
            | {col: [np.nan, np.nan] for col in REPORT_COLUMNS}
        )
        self.teamlog = pd.DataFrame({"team": ["BOS"]})
        self.players = {"Filled": {"position": "C"}}

    def load(self):
        self.calls.append("load")

    def backfill_season_sums(self, seasons=None):
        self.calls.append(("backfill", seasons))
        return 3


@pytest.mark.parametrize(
    ("args", "seasons"),
    [((), None), (("--season", "2024", "--season", "2025"), {2024, 2025})],
)
def test_loads_backfills_writes_then_reports(monkeypatch, args, seasons):
    calls = []
    fake = _FakeStatsNHL(calls)
    monkeypatch.setattr(backfill_nhl_gamelog, "StatsNHL", lambda: fake)
    monkeypatch.setattr(
        backfill_nhl_gamelog,
        "write_gamelog",
        lambda *a: calls.append(("write", *a)),
    )

    result = CliRunner().invoke(backfill_nhl_gamelog.main, list(args))

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
def test_reports_fills_the_report_columns_and_re_parses_no_game(monkeypatch, args, seasons):
    calls = []
    fake = _FakeStatsNHL(calls)
    monkeypatch.setattr(backfill_nhl_gamelog, "StatsNHL", lambda: fake)
    monkeypatch.setattr(
        backfill_nhl_gamelog,
        "write_gamelog",
        lambda *a: calls.append(("write", *a)),
    )

    def fill(gamelog, seasons=None):
        calls.append(("fill", seasons))
        return gamelog.assign(**{col: [1.0, np.nan] for col in REPORT_COLUMNS})

    monkeypatch.setattr(backfill_nhl_gamelog, "fill_report_columns", fill)

    result = CliRunner().invoke(backfill_nhl_gamelog.main, list(args))

    assert result.exit_code == 0, result.output
    assert calls == [
        "load",
        ("fill", seasons),
        ("write", "nhl", fake.gamelog, fake.teamlog, fake.players),
    ]
    assert result.output == "rows still lacking a report column: 1\n"
