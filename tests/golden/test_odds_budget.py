"""Golden pins for ``helpers.odds_budget`` — the Odds API credit-budget governor.

Covers the monthly cycle arithmetic (``cycle_bounds``), the activity tiers
(``classify_tier`` and the ``league_is_live`` update gate), the broad-run
props/game-line admission decision (``broad_run_allowance``, including the
once-a-day starvation escape), the JSONL usage-ledger writer
(``record_response``), and the ledger-derived cost estimator
(``estimate_costs``). Allowance/estimator configs are built inline so the
pins track the code, not the committed tunables; the committed file's schema
and its August starvation behavior are pinned against ``BUDGET_CFG``.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime, timedelta

from sportstradamus import moneylines
from sportstradamus.helpers import io, odds_budget
from sportstradamus.helpers.odds_budget import BudgetDecision

# Cycle end 2026-08-01 makes days_left exactly 10.0, so every reserve /
# per_slot float asserted below is exact.
_TEN_DAYS_LEFT = datetime(2026, 7, 22, tzinfo=UTC)
_NOW = datetime(2026, 7, 10, 12, 0, tzinfo=UTC)

_QUOTA_HEADERS = {
    "X-Requests-Remaining": "17500",
    "X-Requests-Used": "2500",
    "X-Requests-Last": "5",
}

_ESTIMATE_CFG = {"estimate_window_days": 7, "floor_min_per_day": 10}


def _allowance_cfg(**overrides) -> dict:
    cfg = {
        "cycle_reset_day": 1,
        "broad_slots_per_day": 2,
        "floor_safety_factor": 1.0,
        "default_league_run_cost": 500,
        "league_priority": ["NBA", "MLB", "NFL", "NHL", "WNBA"],
    }
    cfg.update(overrides)
    return cfg


class _FakeResponse:
    def __init__(self, headers, status_code=200, payload=None):
        self.headers = headers
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def _governed(monkeypatch, tmp_path, kind="broad"):
    """Sandbox the ledger path + key names and stamp a fresh run context."""
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(odds_budget, "ODDS_USAGE_LEDGER_PATH", ledger)
    monkeypatch.setattr(
        odds_budget, "_KEY_NAMES", {"fake-plus": "odds_api_plus", "fake-margin": "odds_api"}
    )
    odds_budget.set_run_context(kind)
    return ledger


def _ledger_line(
    days_ago, kind, cost, key="odds_api_plus", run="r1", league=None, endpoint="event_odds"
) -> str:
    record = {
        "ts": (_NOW - timedelta(days=days_ago)).isoformat(),
        "run": run,
        "kind": kind,
        "key": key,
        "endpoint": endpoint,
        "sport_key": None,
        "league": league,
        "cost": cost,
        "used": 0,
        "remaining": 0,
        "status": 200,
    }
    return json.dumps(record) + "\n"


def test_budget_cfg_schema() -> None:
    assert set(odds_budget.BUDGET_CFG) == {
        "cycle_credits",
        "cycle_reset_day",
        "enforce",
        "broad_slots_per_day",
        "floor_safety_factor",
        "floor_min_per_day",
        "default_league_run_cost",
        "estimate_window_days",
        "poll_horizon_days",
        "promote_within_days",
        "league_priority",
        "league_seed_costs",
    }
    assert set(odds_budget.BUDGET_CFG["league_seed_costs"]) == set(
        odds_budget.BUDGET_CFG["league_priority"]
    )


def test_cycle_bounds_mid_cycle() -> None:
    assert odds_budget.cycle_bounds(datetime(2026, 7, 10, tzinfo=UTC), 1) == (
        datetime(2026, 7, 1, tzinfo=UTC),
        datetime(2026, 8, 1, tzinfo=UTC),
    )


def test_cycle_bounds_reset_boundaries() -> None:
    assert odds_budget.cycle_bounds(datetime(2026, 7, 14, tzinfo=UTC), 15) == (
        datetime(2026, 6, 15, tzinfo=UTC),
        datetime(2026, 7, 15, tzinfo=UTC),
    )
    assert odds_budget.cycle_bounds(datetime(2026, 7, 15, tzinfo=UTC), 15) == (
        datetime(2026, 7, 15, tzinfo=UTC),
        datetime(2026, 8, 15, tzinfo=UTC),
    )


def test_cycle_bounds_clamps_reset_day_to_month_length() -> None:
    assert odds_budget.cycle_bounds(datetime(2026, 2, 15, tzinfo=UTC), 31) == (
        datetime(2026, 1, 31, tzinfo=UTC),
        datetime(2026, 2, 28, tzinfo=UTC),
    )
    assert odds_budget.cycle_bounds(datetime(2026, 2, 28, tzinfo=UTC), 31) == (
        datetime(2026, 2, 28, tzinfo=UTC),
        datetime(2026, 3, 31, tzinfo=UTC),
    )


def test_cycle_bounds_january_rolls_back_to_december() -> None:
    assert odds_budget.cycle_bounds(datetime(2026, 1, 5, tzinfo=UTC), 15) == (
        datetime(2025, 12, 15, tzinfo=UTC),
        datetime(2026, 1, 15, tzinfo=UTC),
    )


_TIER_CFG = {"poll_horizon_days": 14, "promote_within_days": 1}


def test_classify_tier_by_days_to_next_game() -> None:
    assert odds_budget.classify_tier(0.5, None, _TIER_CFG) == "live"
    assert odds_budget.classify_tier(-0.1, None, _TIER_CFG) == "live"  # in-progress game
    assert odds_budget.classify_tier(5.0, None, _TIER_CFG) == "preseason"
    assert odds_budget.classify_tier(14.0, None, _TIER_CFG) == "preseason"
    assert odds_budget.classify_tier(14.1, None, _TIER_CFG) is None
    assert odds_budget.classify_tier(None, None, _TIER_CFG) is None


def test_classify_tier_live_is_sticky_within_horizon() -> None:
    # NFL weekdays / all-star breaks: a live league with its next game days
    # out stays live; a preseason league only promotes at the 1-day mark.
    assert odds_budget.classify_tier(5.0, "live", _TIER_CFG) == "live"
    assert odds_budget.classify_tier(5.0, "preseason", _TIER_CFG) == "preseason"
    assert odds_budget.classify_tier(20.0, "live", _TIER_CFG) is None


def _write_activity(monkeypatch, tmp_path, snapshot) -> None:
    path = tmp_path / "league_activity.json"
    monkeypatch.setattr(io, "LEAGUE_ACTIVITY_PATH", path)
    if snapshot is not None:
        path.write_text(json.dumps(snapshot))


def test_league_is_live_reads_fresh_snapshot(monkeypatch, tmp_path) -> None:
    snapshot = {
        "updated": datetime.now(UTC).isoformat(),
        "leagues": {
            "MLB": {"tier": "live"},
            "NFL": {"tier": "preseason"},
            "NHL": {"tier": "idle", "last_game": "2026-06-20T00:00:00Z"},
        },
    }
    _write_activity(monkeypatch, tmp_path, snapshot)
    # Both active tiers poll; idle or absent leagues don't, even if
    # season_start says live.
    assert odds_budget.league_is_live("MLB", date(2099, 1, 1))
    assert odds_budget.league_is_live("NFL", date(2099, 1, 1))
    assert not odds_budget.league_is_live("NHL", date(2000, 1, 1))
    assert not odds_budget.league_is_live("NBA", date(2000, 1, 1))


def test_league_is_live_stale_snapshot_falls_back_to_season_start(monkeypatch, tmp_path) -> None:
    snapshot = {
        "updated": (datetime.now(UTC) - timedelta(days=4)).isoformat(),
        "leagues": {"NBA": {"tier": "live"}},
    }
    _write_activity(monkeypatch, tmp_path, snapshot)
    today = datetime.now(UTC).date()
    assert not odds_budget.league_is_live("NBA", today + timedelta(days=30))
    assert odds_budget.league_is_live("MLB", today + timedelta(days=6))


def test_league_is_live_missing_snapshot_falls_back_to_season_start(monkeypatch, tmp_path) -> None:
    _write_activity(monkeypatch, tmp_path, None)
    today = datetime.now(UTC).date()
    assert odds_budget.league_is_live("NBA", today - timedelta(days=100))
    assert not odds_budget.league_is_live("NFL", today + timedelta(days=30))


def _game_ts(days: float) -> str:
    return (datetime.now(UTC) + timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_update_window_open_around_scheduled_games(monkeypatch, tmp_path) -> None:
    snapshot = {
        "updated": datetime.now(UTC).isoformat(),
        "leagues": {
            "MLB": {"tier": "live", "next_game": _game_ts(0.3)},
            "NFL": {"tier": "preseason", "next_game": _game_ts(12)},
            "WNBA": {"tier": "preseason", "next_game": _game_ts(9)},
            "NHL": {"tier": "idle", "last_game": _game_ts(-5)},
            "NBA": {"tier": "idle", "last_game": _game_ts(-11)},
        },
    }
    _write_activity(monkeypatch, tmp_path, snapshot)
    far_start = datetime.now(UTC).date() + timedelta(days=60)
    assert odds_budget.update_window_open("MLB", far_start)
    assert not odds_budget.update_window_open("NFL", far_start)  # opener 12d out, window is 10
    assert odds_budget.update_window_open("WNBA", far_start)
    assert odds_budget.update_window_open("NHL", far_start)  # post-season ingest tail
    assert not odds_budget.update_window_open("NBA", far_start)  # tail expired
    # Never-seen league is closed even when season_start would allow it.
    assert not odds_budget.update_window_open("XFL", datetime.now(UTC).date())


def test_update_window_open_stale_snapshot_falls_back_to_season_start(
    monkeypatch, tmp_path
) -> None:
    _write_activity(monkeypatch, tmp_path, None)
    today = datetime.now(UTC).date()
    assert odds_budget.update_window_open("NBA", today + timedelta(days=9))
    assert not odds_budget.update_window_open("NBA", today + timedelta(days=11))


def test_update_window_open_force_env_bypasses(monkeypatch, tmp_path) -> None:
    snapshot = {"updated": datetime.now(UTC).isoformat(), "leagues": {}}
    _write_activity(monkeypatch, tmp_path, snapshot)
    monkeypatch.setenv("SPORTSTRADAMUS_FORCE_UPDATE", "1")
    assert odds_budget.update_window_open("NBA", date(2099, 1, 1))


def test_season_opener_converts_to_chicago_date(monkeypatch, tmp_path) -> None:
    snapshot = {
        "updated": datetime.now(UTC).isoformat(),
        "leagues": {
            # 00:15Z is the prior evening in America/Chicago.
            "NFL": {"tier": "preseason", "opener": "2026-09-10T00:15:00Z"},
            "MLB": {"tier": "live", "next_game": "2026-07-10T22:41:00Z"},
        },
    }
    _write_activity(monkeypatch, tmp_path, snapshot)
    assert odds_budget.season_opener("NFL") == date(2026, 9, 9)
    assert odds_budget.season_opener("MLB") is None  # mid-season, opener never observed
    assert odds_budget.season_opener("NBA") is None


def test_season_opener_stale_snapshot_returns_none(monkeypatch, tmp_path) -> None:
    snapshot = {
        "updated": (datetime.now(UTC) - timedelta(days=4)).isoformat(),
        "leagues": {"NFL": {"tier": "preseason", "opener": "2026-09-10T00:15:00Z"}},
    }
    _write_activity(monkeypatch, tmp_path, snapshot)
    assert odds_budget.season_opener("NFL") is None


def test_broad_run_allowance_admits_all() -> None:
    decision = odds_budget.broad_run_allowance(
        _TEN_DAYS_LEFT,
        5000,
        {"NBA": "live", "WNBA": "live"},
        100.0,
        {"NBA": 120.0, "WNBA": 60.0},
        {},
        _allowance_cfg(),
    )
    # reserve 1120 = close 1000 + game-line 3 * 2 leagues * 2 slots * 10 days.
    assert decision == BudgetDecision(("NBA", "WNBA"), ("NBA", "WNBA"), "ok", 1120.0, 194.0, 3880.0)


def test_broad_run_allowance_floor_reserve() -> None:
    decision = odds_budget.broad_run_allowance(
        _TEN_DAYS_LEFT, 1000, {"NBA": "live"}, 100.0, {"NBA": 1.0}, {}, _allowance_cfg()
    )
    # Close floor breached: no props AND no game lines.
    assert decision == BudgetDecision((), (), "floor_reserve", 1060.0, 0.0, -60.0)


def test_broad_run_allowance_all_idle() -> None:
    decision = odds_budget.broad_run_allowance(
        _TEN_DAYS_LEFT, 5000, {}, 100.0, {"NBA": 1.0}, {}, _allowance_cfg()
    )
    assert decision == BudgetDecision((), (), "idle", 0.0, 0.0, 0.0)


def test_broad_run_allowance_partial_respects_priority() -> None:
    decision = odds_budget.broad_run_allowance(
        _TEN_DAYS_LEFT,
        5000,
        {"NFL": "live", "MLB": "live", "NBA": "live"},
        100.0,
        {"NBA": 150.0, "MLB": 100.0, "NFL": 40.0},
        {"MLB": _TEN_DAYS_LEFT - timedelta(hours=1)},  # fresh: keeps the escape out of the way
        _allowance_cfg(),
    )
    # per_slot 191: NBA (150) admits leaving 41, MLB (100) no longer fits, NFL
    # (40) still does — admission order comes from cfg, not the tiers dict.
    assert decision == BudgetDecision(
        ("NBA", "NFL"), ("NBA", "MLB", "NFL"), "partial", 1180.0, 191.0, 3820.0
    )


def test_broad_run_allowance_preseason_rides_the_bottom() -> None:
    decision = odds_budget.broad_run_allowance(
        _TEN_DAYS_LEFT,
        5000,
        {"NBA": "preseason", "MLB": "live", "WNBA": "live"},
        100.0,
        {"NBA": 40.0, "MLB": 100.0, "WNBA": 90.0},
        {},
        _allowance_cfg(),
    )
    # per_slot 191: live MLB (100) and WNBA (90) admit first despite NBA
    # leading the priority list; preseason NBA (40) no longer fits.
    assert decision == BudgetDecision(
        ("MLB", "WNBA"), ("MLB", "WNBA", "NBA"), "partial", 1180.0, 191.0, 3820.0
    )


def test_broad_run_allowance_no_fit() -> None:
    decision = odds_budget.broad_run_allowance(
        _TEN_DAYS_LEFT,
        5000,
        {"NBA": "live"},
        100.0,
        {"NBA": 300.0},
        {"NBA": _TEN_DAYS_LEFT},  # fresh: keeps the escape out of the way
        _allowance_cfg(),
    )
    # Game lines survive a props no_fit — that split is the whole point.
    assert decision == BudgetDecision((), ("NBA",), "no_fit", 1060.0, 197.0, 3940.0)


def test_broad_run_allowance_unknown_league_uses_default_cost() -> None:
    decision = odds_budget.broad_run_allowance(
        _TEN_DAYS_LEFT,
        5000,
        {"NBA": "live", "MLB": "live"},
        100.0,
        {},
        {"MLB": _TEN_DAYS_LEFT},  # fresh: keeps the escape out of the way
        _allowance_cfg(default_league_run_cost=150),
    )
    assert decision.allowed_leagues == ("NBA",)
    assert decision.reason == "partial"


def test_broad_run_allowance_rolls_skipped_league_forward() -> None:
    cfg = _allowance_cfg()
    early = odds_budget.broad_run_allowance(
        _TEN_DAYS_LEFT, 4000, {"NBA": "live"}, 0.0, {"NBA": 500.0}, {}, cfg
    )
    # 500 also outprices the 394 day share, so the starvation escape stays shut.
    assert early == BudgetDecision((), ("NBA",), "no_fit", 60.0, 197.0, 3940.0)
    # Nothing spent, but fewer slots left in the cycle -> a bigger per-slot
    # share admits the league skipped eight days earlier.
    late = odds_budget.broad_run_allowance(
        datetime(2026, 7, 30, tzinfo=UTC), 4000, {"NBA": "live"}, 0.0, {"NBA": 500.0}, {}, cfg
    )
    assert late == BudgetDecision(("NBA",), ("NBA",), "ok", 12.0, 997.0, 3988.0)


def test_broad_run_allowance_end_of_cycle_slot_floor() -> None:
    now = datetime(2026, 7, 31, 18, 0, tzinfo=UTC)  # 0.25 days -> 0.5 slots, floored to 1.0
    decision = odds_budget.broad_run_allowance(
        now, 4000, {"NBA": "live"}, 0.0, {"NBA": 3000.0}, {}, _allowance_cfg()
    )
    assert decision.per_slot == decision.spendable == 3998.5
    assert decision.allowed_leagues == ("NBA",)


def test_broad_run_allowance_starvation_escape_admits_one_league_per_run() -> None:
    decision = odds_budget.broad_run_allowance(
        _TEN_DAYS_LEFT,
        5000,
        {"NBA": "live", "MLB": "live"},
        100.0,
        {"NBA": 300.0, "MLB": 350.0},
        {},
        _allowance_cfg(),
    )
    # per_slot 194 fits neither; both are starved (no last_paid) and both fit
    # the 388 day share, but only the first in priority escapes this run.
    assert decision.allowed_leagues == ("NBA",)
    assert decision.reason == "partial"


def test_broad_run_allowance_escape_never_admits_preseason() -> None:
    decision = odds_budget.broad_run_allowance(
        _TEN_DAYS_LEFT, 5000, {"NBA": "preseason"}, 100.0, {"NBA": 300.0}, {}, _allowance_cfg()
    )
    # Starved and under the 394 day share, but preseason: game lines only.
    assert decision == BudgetDecision((), ("NBA",), "no_fit", 1060.0, 197.0, 3940.0)


def test_broad_run_allowance_fresh_last_paid_blocks_escape() -> None:
    def decide(last_paid):
        return odds_budget.broad_run_allowance(
            _TEN_DAYS_LEFT,
            5000,
            {"NBA": "live"},
            100.0,
            {"NBA": 300.0},
            last_paid,
            _allowance_cfg(),
        )

    # Bought 12h ago (< STARVED_PROPS_AFTER_DAYS): the escape stays shut. The
    # same cell 36h unbought escapes against the 394 day share.
    assert decide({"NBA": _TEN_DAYS_LEFT - timedelta(hours=12)}).allowed_leagues == ()
    assert decide({"NBA": _TEN_DAYS_LEFT - timedelta(hours=36)}).allowed_leagues == ("NBA",)


def test_broad_run_allowance_escape_skips_league_over_day_share() -> None:
    decision = odds_budget.broad_run_allowance(
        _TEN_DAYS_LEFT,
        5000,
        {"NBA": "live", "WNBA": "live"},
        100.0,
        {"NBA": 500.0, "WNBA": 300.0},
        {},
        _allowance_cfg(),
    )
    # Both starved; NBA (500) outprices the 388 day share, so the escape falls
    # through to the cheaper WNBA instead of burning the run on nothing.
    assert decision.allowed_leagues == ("WNBA",)


def test_broad_run_allowance_shipped_config_escape_reaches_mlb() -> None:
    """The committed August config must not starve MLB props forever.

    Under the shipped tunables (20k credits, floor 300, MLB seeded at 220 —
    the priciest live league) the per-slot share sits under every seed for
    most of the cycle, so greedy admission alone bought nothing and strict
    priority starved MLB for weeks (the 2026-07/08 incident). The escape
    admits exactly ONE starved live league per run in priority order, so a
    single run with everything starved reaches WNBA only; MLB is bought once
    the cheaper leagues are fresh — successive runs rotate through the slate.
    Game lines stay on for the whole live slate throughout.
    """
    cfg = odds_budget.BUDGET_CFG
    # Guard: a CI config stub swapping the committed file would silently
    # retune every number below.
    assert cfg["cycle_credits"] == 20000
    assert cfg["league_seed_costs"]["MLB"] == 220
    tiers = {"WNBA": "live", "MLB": "live", "NFL": "live"}
    seeds = cfg["league_seed_costs"]
    floor = cfg["floor_min_per_day"]
    days = {
        1: (datetime(2026, 8, 1, tzinfo=UTC), 20000),
        15: (datetime(2026, 8, 15, tzinfo=UTC), 11000),
        28: (datetime(2026, 8, 28, tzinfo=UTC), 4000),
    }
    starved = {
        day: odds_budget.broad_run_allowance(now, remaining, tiers, floor, seeds, {}, cfg)
        for day, (now, remaining) in days.items()
    }
    for decision in starved.values():
        assert decision.gameline_leagues == ("WNBA", "NFL", "MLB")
    # Everything starved: the one escape goes to WNBA (first live in
    # priority). By day 28 the fatter per-slot share (116) admits WNBA
    # outright and the escape moves down to NFL — never two leagues at once.
    assert starved[1].allowed_leagues == ("WNBA",)  # per_slot 45.03, day share 225.2
    assert starved[15].allowed_leagues == ("WNBA",)  # per_slot 45.41, day share 227.1
    assert starved[28].allowed_leagues == ("WNBA", "NFL")  # WNBA per-slot, NFL escape
    for day, (now, remaining) in days.items():
        fresh = {"WNBA": now - timedelta(hours=2), "NFL": now - timedelta(hours=2)}
        decision = odds_budget.broad_run_allowance(now, remaining, tiers, floor, seeds, fresh, cfg)
        # MLB's 220 fits every day share above (min 225.2 at day 1).
        assert "MLB" in decision.allowed_leagues, f"day {day}"
    for day, (now, remaining) in days.items():
        fresh = {"MLB": now - timedelta(hours=2), "NFL": now - timedelta(hours=2)}
        decision = odds_budget.broad_run_allowance(now, remaining, tiers, floor, seeds, fresh, cfg)
        # A fresh MLB blocks re-admission: the escape doubles as the cadence limiter.
        assert decision.allowed_leagues == ("WNBA",), f"day {day}"


def test_record_response_schema_and_endpoint_kinds(monkeypatch, tmp_path) -> None:
    ledger = _governed(monkeypatch, tmp_path)
    urls = [
        moneylines.ODDS_API_SPORTS_URL,
        moneylines.ODDS_API_EVENT_ODDS_URL.format(sport="basketball_nba", eventId="ev1"),
        moneylines.ODDS_API_ODDS_URL.format(sport="basketball_nba"),
        moneylines.ODDS_API_HISTORICAL_ODDS_URL.format(sport="basketball_nba"),
    ]
    for url in urls:
        odds_budget.record_response(url, {"apiKey": "fake-plus"}, _FakeResponse(_QUOTA_HEADERS))

    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(records) == len(urls)
    assert [r["endpoint"] for r in records] == ["sports_index", "event_odds", "odds", "hist_odds"]
    assert [r["sport_key"] for r in records] == [None] + ["basketball_nba"] * 3
    assert [r["league"] for r in records] == [None] + ["NBA"] * 3
    first = records[0]
    assert set(first) == {
        "ts",
        "run",
        "kind",
        "key",
        "endpoint",
        "sport_key",
        "league",
        "cost",
        "used",
        "remaining",
        "status",
    }
    assert first["kind"] == "broad"
    assert first["key"] == "odds_api_plus"
    assert (first["cost"], first["used"], first["remaining"]) == (5, 2500, 17500)
    assert first["status"] == 200
    assert first["run"] == odds_budget.run_summary()["run_id"]


def test_record_response_skips_response_without_quota_headers(monkeypatch, tmp_path) -> None:
    ledger = _governed(monkeypatch, tmp_path)
    odds_budget.record_response(
        moneylines.ODDS_API_SPORTS_URL, {"apiKey": "fake-plus"}, _FakeResponse({})
    )
    assert not ledger.exists()
    summary = odds_budget.run_summary()
    assert (summary["calls"], summary["cost"]) == (0, 0)


def test_record_response_accumulates_run_totals(monkeypatch, tmp_path) -> None:
    ledger = _governed(monkeypatch, tmp_path, kind="close_lines")
    url = moneylines.ODDS_API_EVENT_ODDS_URL.format(sport="basketball_nba", eventId="ev1")
    for cost in ("3", "5"):
        headers = {"X-Requests-Remaining": "100", "X-Requests-Used": "0", "X-Requests-Last": cost}
        odds_budget.record_response(url, {"apiKey": "fake-plus"}, _FakeResponse(headers))

    summary = odds_budget.run_summary()
    assert (summary["kind"], summary["calls"], summary["cost"]) == ("close_lines", 2, 8)
    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [r["cost"] for r in records] == [3, 5]
    assert {r["run"] for r in records} == {summary["run_id"]}


def test_record_response_never_writes_the_secret(monkeypatch, tmp_path) -> None:
    ledger = _governed(monkeypatch, tmp_path)
    odds_budget.record_response(
        moneylines.ODDS_API_SPORTS_URL, {"apiKey": "fake-plus"}, _FakeResponse(_QUOTA_HEADERS)
    )
    text = ledger.read_text()
    assert "fake-plus" not in text
    assert '"odds_api_plus"' in text


def test_estimate_costs_floor_and_league_means(monkeypatch, tmp_path) -> None:
    ledger = _governed(monkeypatch, tmp_path)
    ledger.write_text(
        _ledger_line(1, "close_lines", 100)
        + _ledger_line(2, "close_lines", 40)
        + _ledger_line(1, "broad", 30, run="r1", league="NBA")
        + _ledger_line(1, "broad", 20, run="r1", league="NBA")
        + _ledger_line(2, "broad", 100, run="r2", league="NBA")
        + _ledger_line(2, "broad", 60, run="r2", league="WNBA")
    )
    floor_per_day, league_costs, last_paid = odds_budget.estimate_costs(_NOW, _ESTIMATE_CFG)
    assert floor_per_day == 20.0  # (100 + 40) / 7
    assert league_costs == {"NBA": 75.0, "WNBA": 60.0}  # NBA: mean of run totals 50 and 100
    # Newest paid-props row per league.
    assert last_paid == {"NBA": _NOW - timedelta(days=1), "WNBA": _NOW - timedelta(days=2)}


def test_estimate_costs_floor_min_ratchet(monkeypatch, tmp_path) -> None:
    ledger = _governed(monkeypatch, tmp_path)
    ledger.write_text(_ledger_line(1, "close_lines", 14))  # 2/day observed, under the floor
    floor_per_day, _, _ = odds_budget.estimate_costs(_NOW, _ESTIMATE_CFG)
    assert floor_per_day == 10.0


def test_estimate_costs_excludes_stale_offkey_and_unknown(monkeypatch, tmp_path) -> None:
    ledger = _governed(monkeypatch, tmp_path)
    ledger.write_text(
        _ledger_line(1, "close_lines", 140)
        + _ledger_line(8, "close_lines", 7000)  # outside the 7-day window
        + _ledger_line(1, "close_lines", 7000, key="odds_api_max")  # backfill key
        + _ledger_line(1, "unknown", 7000, league="NBA")  # untagged run kind
        + _ledger_line(8, "broad", 7000, run="r0", league="NBA")  # outside the window
        + _ledger_line(1, "broad", 7000, run="r1", league="NBA", key="odds_api")  # margin key
    )
    floor_per_day, league_costs, last_paid = odds_budget.estimate_costs(_NOW, _ESTIMATE_CFG)
    assert floor_per_day == 20.0  # only the first line counts
    assert league_costs == {}
    assert last_paid == {}


def test_estimate_costs_seeds_bootstrap_unseen_leagues(monkeypatch, tmp_path) -> None:
    ledger = _governed(monkeypatch, tmp_path)
    ledger.write_text(_ledger_line(1, "broad", 80, run="r1", league="NBA"))
    cfg = {**_ESTIMATE_CFG, "league_seed_costs": {"NBA": 100, "MLB": 300}}
    _, league_costs, _ = odds_budget.estimate_costs(_NOW, cfg)
    # Ledger mean beats the seed; a league the ledger hasn't seen keeps its seed.
    assert league_costs == {"NBA": 80.0, "MLB": 300}


def test_estimate_costs_ignores_free_probe_runs_when_pricing_a_league(
    monkeypatch, tmp_path
) -> None:
    """Cost-0 runs must not price a league — a skipped league is not a cheap one.

    Every broad run logs a free ``events`` tiering probe per league, so a league the
    governor keeps excluding accumulates zero-cost runs. Averaging those in measures
    how often it was skipped, not what it costs: MLB decayed to 37.5 against a true
    218.5, and out-of-season leagues priced at 0.0 — which would have waved NFL
    through as free the moment its season opened. A league with only free probes
    reports no mean at all and falls back to its seed.
    """
    ledger = _governed(monkeypatch, tmp_path)
    ledger.write_text(
        _ledger_line(1, "broad", 200, run="r1", league="MLB")
        + _ledger_line(1, "broad", 0, run="r2", league="MLB")  # probe-only run
        + _ledger_line(1, "broad", 0, run="r3", league="MLB")
        + _ledger_line(1, "broad", 0, run="r1", league="NFL")  # never bought at all
    )
    cfg = {**_ESTIMATE_CFG, "league_seed_costs": {"NFL": 165}}
    _, league_costs, last_paid = odds_budget.estimate_costs(_NOW, cfg)
    assert league_costs == {"MLB": 200.0, "NFL": 165}
    assert last_paid == {"MLB": _NOW - timedelta(days=1)}  # free probes never stamp last_paid


def test_estimate_costs_gameline_rows_never_price_props(monkeypatch, tmp_path) -> None:
    """Game-line spend (endpoint ``odds``) is not props spend.

    Game lines ride every broad run at ~3 credits even for leagues without
    props admission, so folding them in would price a starved league's props
    near 3 and stamp ``last_paid`` — un-starving MLB with a purchase that
    never happened. Only ``event_odds`` rows count for both outputs.
    """
    ledger = _governed(monkeypatch, tmp_path)
    ledger.write_text(
        _ledger_line(1, "broad", 3, run="r1", league="MLB", endpoint="odds")
        + _ledger_line(2, "broad", 60, run="r2", league="WNBA")
    )
    cfg = {**_ESTIMATE_CFG, "league_seed_costs": {"MLB": 220}}
    _, league_costs, last_paid = odds_budget.estimate_costs(_NOW, cfg)
    assert league_costs == {"MLB": 220, "WNBA": 60.0}  # seed retained, not 3.0
    assert last_paid == {"WNBA": _NOW - timedelta(days=2)}


def test_estimate_costs_skips_torn_final_line(monkeypatch, tmp_path) -> None:
    ledger = _governed(monkeypatch, tmp_path)
    ledger.write_text(_ledger_line(1, "close_lines", 140) + '{"ts": "2026-07-10T')
    floor_per_day, league_costs, last_paid = odds_budget.estimate_costs(_NOW, _ESTIMATE_CFG)
    assert floor_per_day == 20.0
    assert league_costs == {}
    assert last_paid == {}


def test_estimate_costs_prunes_stale_records(monkeypatch, tmp_path) -> None:
    ledger = _governed(monkeypatch, tmp_path)
    recent = _ledger_line(1, "close_lines", 140)
    ledger.write_text(_ledger_line(50, "close_lines", 999) + recent)
    odds_budget.estimate_costs(_NOW, _ESTIMATE_CFG)
    assert ledger.read_text() == recent


def _fake_governed_probe(monkeypatch, remaining, floor_per_day, league_costs) -> dict:
    """Wire ``_resolve_broad_run_leagues`` to canned probe/tier/estimate inputs.

    The synthetic enforcing config plus a patched ``estimate_costs`` keep the
    decision deterministic for any wall-clock ``now`` inside a cycle.
    """
    monkeypatch.setattr(odds_budget, "BUDGET_CFG", _allowance_cfg(enforce=True))
    probe = _FakeResponse(
        {"X-Requests-Remaining": str(remaining)},
        payload=[{"title": "NBA", "key": "basketball_nba", "active": True}],
    )
    monkeypatch.setattr(moneylines, "_get_with_retry", lambda url, params=None: probe)
    monkeypatch.setattr(
        moneylines, "_classify_league_activity", lambda key, active: {"NBA": "live"}
    )
    monkeypatch.setattr(
        odds_budget, "estimate_costs", lambda now, cfg: (floor_per_day, league_costs, {})
    )
    return {"odds_api_plus": "fake-plus"}


def test_resolve_broad_run_leagues_gameline_only_run_is_not_a_skip(monkeypatch) -> None:
    # Props priced out of any share while the close floor holds: the run
    # downgrades to a cheap game-line-only pass instead of skipping outright.
    keys = _fake_governed_probe(monkeypatch, 1000, 1.0, {"NBA": 10_000_000.0})
    props, gameline, skip = moneylines._resolve_broad_run_leagues(
        keys, logging.getLogger("confer-test")
    )
    assert props == ()
    assert gameline == ("NBA",)
    assert skip is False


def test_resolve_broad_run_leagues_skips_only_when_both_admissions_empty(monkeypatch) -> None:
    # Remaining under the close reserve: no props AND no game lines -> full skip.
    keys = _fake_governed_probe(monkeypatch, 10, 1e9, {"NBA": 1.0})
    assert moneylines._resolve_broad_run_leagues(keys, logging.getLogger("confer-test")) == (
        None,
        None,
        True,
    )
