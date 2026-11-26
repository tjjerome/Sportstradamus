"""One league's failed gamelog update must not cost every league its board.

``prophecize`` and the ledger build load every league through
``stats.load_live_leagues`` before they score anything. On 2026-10-10 an NBA update
raised on a missing seed file and three production runs died in five seconds each,
leaving no board for any league. The loader holds a failed update to its own league,
which sits that run out, and ``prophecize`` still exits non-zero once its snapshots
are written, so the job's health check fires. A constructor or a first load that
raises is held the same way.
"""

from __future__ import annotations

import logging
from datetime import date
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
from click.testing import CliRunner

from sportstradamus.prediction import cli as prediction_cli
from sportstradamus.prediction.ev_stamp import EV_STAMP_COLS
from sportstradamus.prediction.offer_records import SCORED_RECORD_COLS
from sportstradamus.stats import live

_LEAGUES = ["NBA", "NFL", "WNBA", "MLB", "NHL"]
# One date per league, so the liveness gate is seen reading each league's own.
_SEASON_STARTS = {league: date(2026, month, 1) for month, league in enumerate(_LEAGUES, start=1)}


class _FakeStats:
    """What the loader touches on a ``Stats``; ``load`` and ``update`` land in ``calls``."""

    def __init__(self, league, calls, update_error):
        self.league = league
        self.season_start = _SEASON_STARTS[league]
        self._calls = calls
        self._update_error = update_error

    def load(self):
        self._calls.append((self.league, "load"))

    def update(self):
        self._calls.append((self.league, "update"))
        if self._update_error is not None:
            raise self._update_error


def _install_leagues(monkeypatch, live_leagues, update_errors):
    """Swap the five ``Stats`` classes for ``_FakeStats``; return their shared call log.

    The real classes read gamelog files and call the league APIs.
    """
    calls = []
    for league in _LEAGUES:
        fake = partial(_FakeStats, league, calls, update_errors.get(league))
        monkeypatch.setattr(live, f"Stats{league}", fake)

    def league_is_live(league, season_start):
        assert season_start == _SEASON_STARTS[league]
        return league in live_leagues

    monkeypatch.setattr(live.odds_budget, "league_is_live", league_is_live)
    return calls


def test_every_live_league_is_loaded_updated_and_returned_in_order(monkeypatch):
    calls = _install_leagues(monkeypatch, live_leagues=_LEAGUES, update_errors={})

    stats, failed = live.load_live_leagues()

    assert list(stats) == _LEAGUES
    assert [struct.league for struct in stats.values()] == _LEAGUES
    assert failed == []
    assert calls == [(league, step) for league in _LEAGUES for step in ("load", "update")]


def test_a_league_that_is_not_live_is_neither_updated_nor_returned(monkeypatch):
    calls = _install_leagues(monkeypatch, live_leagues=["NFL", "NHL"], update_errors={})

    stats, failed = live.load_live_leagues()

    assert list(stats) == ["NFL", "NHL"]
    assert failed == []
    assert [league for league, step in calls if step == "update"] == ["NFL", "NHL"]


def test_a_failed_update_is_held_to_its_own_league(monkeypatch, caplog):
    seed_missing = FileNotFoundError("seed file")
    calls = _install_leagues(
        monkeypatch, live_leagues=_LEAGUES, update_errors={"NBA": seed_missing}
    )

    with caplog.at_level(logging.ERROR, logger="log"):
        stats, failed = live.load_live_leagues()

    assert list(stats) == ["NFL", "WNBA", "MLB", "NHL"]
    assert failed == ["NBA"]
    assert [league for league, step in calls if step == "update"] == _LEAGUES
    (record,) = caplog.records
    assert "NBA update failed" in record.getMessage()
    assert record.exc_info[1] is seed_missing


def test_a_failed_constructor_or_first_load_is_held_to_its_own_league(monkeypatch, caplog):
    _install_leagues(monkeypatch, live_leagues=_LEAGUES, update_errors={})
    constructor_error = RuntimeError("MLB constructor")
    load_error = OSError("NHL gamelog bundle")
    unreadable = Mock()
    unreadable.load.side_effect = load_error
    monkeypatch.setattr(live, "StatsMLB", Mock(side_effect=constructor_error))
    monkeypatch.setattr(live, "StatsNHL", Mock(return_value=unreadable))

    with caplog.at_level(logging.ERROR, logger="log"):
        stats, failed = live.load_live_leagues()

    assert list(stats) == ["NBA", "NFL", "WNBA"]
    assert failed == ["MLB", "NHL"]
    assert [record.getMessage().split()[0] for record in caplog.records] == failed
    assert [record.exc_info[1] for record in caplog.records] == [constructor_error, load_error]


# The names ``main`` writes or deletes through, matched by prefix rather than listed:
# this suite runs in the developer's checkout, where a writer added to ``main`` later
# and missed here would replace a real snapshot with this slate's empty frame.
_WRITER_PREFIXES = ("write_", "_write_", "upsert_", "trim_", "prune_", "snapshot_", "capture_")


# No offers, under the columns ``main`` reads before it checks a platform's frame for rows.
_EMPTY_SLATE = pd.DataFrame(columns=["League", "Market", "Boost", "Boost_Over", "Boost_Under"])


def _run_board(monkeypatch, tmp_path, updates_failed, offers=_EMPTY_SLATE, writers=None):
    """Run ``prophecize`` with both platforms scoring ``offers``; return its result and
    one mock holding its writers and its ``process_offers`` calls.

    The one loaded league is a bare name: no slate here has a row of that league, so
    nothing reads its ``Stats``. A test that needs a writer to do something passes its
    own ``writers`` mock.
    """
    monkeypatch.chdir(tmp_path)  # get_logger opens its JSONL under the cwd
    # main rebinds tqdm.__init__ for the whole process; this hands it back afterwards.
    monkeypatch.setattr(prediction_cli.tqdm, "__init__", prediction_cli.tqdm.__init__)
    if writers is None:
        writers = Mock()
    for name in list(vars(prediction_cli)):
        if name.startswith(_WRITER_PREFIXES):
            monkeypatch.setattr(prediction_cli, name, getattr(writers, name))
    stats = dict.fromkeys(["NFL"])
    monkeypatch.setattr(prediction_cli, "load_live_leagues", lambda: (stats, updates_failed))
    writers.process_offers.side_effect = lambda *args, **kwargs: (offers.copy(), pd.DataFrame())
    monkeypatch.setattr(prediction_cli, "process_offers", writers.process_offers)
    monkeypatch.setattr(prediction_cli, "get_ud", lambda capture_dir=None, scored_at=None: {})
    monkeypatch.setattr(prediction_cli, "get_sleeper", dict)
    monkeypatch.setattr(prediction_cli, "read_history", pd.DataFrame)
    monkeypatch.setattr(
        prediction_cli,
        "archive",
        SimpleNamespace(
            default_totals={},
            write=lambda: None,
            get_line=lambda *key: None,
            get_reference_line=lambda *key: None,
            get_team_market=lambda *key: np.nan,
        ),
    )
    return CliRunner().invoke(prediction_cli.main, [], catch_exceptions=False), writers


def test_board_run_exits_nonzero_after_its_writes_when_a_league_update_failed(
    monkeypatch, tmp_path, caplog
):
    with caplog.at_level(logging.ERROR, logger="log"):
        result, writers = _run_board(monkeypatch, tmp_path, updates_failed=["NBA"])

    assert result.exit_code == 1
    writers.write_current_offers.assert_called_once()
    writers.write_history.assert_called_once()
    writers.prune_feature_log.assert_called_once()
    assert "gamelog updates failed: NBA" in caplog.text


def test_board_run_exits_zero_when_every_league_updated(monkeypatch, tmp_path):
    result, writers = _run_board(monkeypatch, tmp_path, updates_failed=[])

    assert result.exit_code == 0
    writers.write_history.assert_called_once()


def test_board_run_builds_no_pickem_snapshot(monkeypatch, tmp_path):
    """``current_pickem.parquet`` had a writer and no reader: a parlay search per contest
    variant per platform, every run, for a file nothing opened."""
    _, writers = _run_board(monkeypatch, tmp_path, updates_failed=[])

    assert not [name for name, _args, _kwargs in writers.mock_calls if "pickem" in name]


def _nhl_assists_slate() -> pd.DataFrame:
    """One NHL leg as ``process_offers`` hands it back, its market under Underdog's label."""
    leg = dict.fromkeys(
        [*SCORED_RECORD_COLS, *EV_STAMP_COLS, "Corr Same", "Corr Opp", "Position", "Game"], np.nan
    )
    leg.update(
        {
            "League": "NHL",
            "Date": "2026-10-12",
            "Team": "EDM",
            "Opponent": "CGY",
            "Game": "CGY/EDM",
            "Player": "Connor McDavid",
            "Market": "Assists",
            "Line": 1.5,
            "Bet": "Over",
            "Boost": 1.0,
            "Model EV": 1.1,
            "Position": "C1",
        }
    )
    return pd.DataFrame([leg])


def test_board_stat_is_the_leagues_own_key_and_market_the_boards_name(monkeypatch, tmp_path):
    """The board renames a market through its platform's map and ``Stat`` names it the
    way the league does, the gamelog column and archive key its readers look up: an NHL
    ``AST`` row is the ``assists`` stat. ``Market`` stays the board's name, which the
    correlation slice, pair modifiers and line-movement rows all join on."""
    result, writers = _run_board(
        monkeypatch, tmp_path, updates_failed=[], offers=_nhl_assists_slate()
    )

    assert result.exit_code == 0
    board = writers.write_current_offers.call_args.args[0].set_index("Platform")
    assert board.loc["Underdog", ["Market", "Stat"]].tolist() == ["AST", "assists"]
    # Sleeper's map holds no "Assists" label: a market with no board name has no key.
    assert board.loc["Sleeper", ["Market", "Stat"]].isna().all()


def test_both_platforms_are_scored_under_the_stamp_their_history_rows_carry(monkeypatch, tmp_path):
    """One stamp per run: a sportsbook quote is aged against the ``Scored At`` its leg is
    stored with, not against a clock read again for each platform."""
    _, writers = _run_board(monkeypatch, tmp_path, updates_failed=[], offers=_nhl_assists_slate())

    (scored_at,) = writers.write_history.call_args.args[0]["Scored At"]
    scoring_stamps = [call.kwargs["scored_at"] for call in writers.process_offers.call_args_list]
    assert scoring_stamps == [scored_at, scored_at]


def test_history_rows_keep_the_ev_stamp_a_scored_leg_carries(monkeypatch, tmp_path):
    stamp = {
        "Book Prob": 0.56,
        "Book Count": 2,
        "Book Class": "A",
        "Book Observed At": pd.Timestamp("2026-10-12T17:40:00"),
        "Fit Prob": 0.53,
        "Fit Prob Low": 0.5,
        "EV": 0.97,
        "Pick": True,
    }
    slate = _nhl_assists_slate().assign(**stamp)

    _, writers = _run_board(monkeypatch, tmp_path, updates_failed=[], offers=slate)

    (row,) = writers.write_history.call_args.args[0][EV_STAMP_COLS].to_dict("records")
    assert row == stamp


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now("UTC").tz_localize(None)


def _slate_starting_in(lead: pd.Timedelta) -> pd.DataFrame:
    """The NHL leg, its game ``lead`` from now under the start time Underdog posts."""
    kickoff = (_utc_now() + lead).strftime("%Y-%m-%dT%H:%M:%SZ")
    return _nhl_assists_slate().assign(Commence=kickoff)


def test_a_board_that_lands_before_the_game_is_recorded(monkeypatch, tmp_path):
    slate = _slate_starting_in(pd.Timedelta(hours=1))

    _, writers = _run_board(monkeypatch, tmp_path, updates_failed=[], offers=slate)

    assert len(writers.write_history.call_args.args[0]) == 1


def test_a_board_that_lands_after_the_game_started_records_none_of_it(monkeypatch, tmp_path):
    """History is the record of what could be bet: the board still shows the leg, and
    no bettor could have acted on it."""
    slate = _slate_starting_in(-pd.Timedelta(hours=1))

    _, writers = _run_board(monkeypatch, tmp_path, updates_failed=[], offers=slate)

    assert len(writers.write_current_offers.call_args.args[0]) == 2
    assert writers.write_history.call_args.args[0].empty


def test_what_could_be_bet_is_judged_from_when_the_board_went_out(monkeypatch, tmp_path):
    """A run publishes once, at its end, some twenty minutes after its stamp. A leg is
    bettable only from then, so the clock is read after the board's last write."""
    clock = {}
    writers = Mock()
    writers.write_current_offer_details.side_effect = lambda *args: clock.setdefault(
        "board_out", _utc_now()
    )
    recordable = prediction_cli._recordable

    def spy(history, new_df, published_at):
        clock["published_at"] = published_at
        return recordable(history, new_df, published_at)

    monkeypatch.setattr(prediction_cli, "_recordable", spy)

    _run_board(
        monkeypatch, tmp_path, updates_failed=[], offers=_nhl_assists_slate(), writers=writers
    )

    assert clock["published_at"] >= clock["board_out"]
