"""Golden pins for ``helpers.odds_budget`` — the Odds API credit-budget governor.

Covers the monthly cycle arithmetic (``cycle_bounds``), the broad-run
league-admission decision (``broad_run_allowance``), the JSONL usage-ledger
writer (``record_response``), and the ledger-derived cost estimator
(``estimate_costs``). Allowance/estimator configs are built inline so the
pins track the code, not the committed tunables; the committed file's schema
is pinned once against ``BUDGET_CFG``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from sportstradamus import moneylines
from sportstradamus.helpers import odds_budget
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
    def __init__(self, headers, status_code=200):
        self.headers = headers
        self.status_code = status_code


def _governed(monkeypatch, tmp_path, kind="broad"):
    """Sandbox the ledger path + key names and stamp a fresh run context."""
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(odds_budget, "ODDS_USAGE_LEDGER_PATH", ledger)
    monkeypatch.setattr(
        odds_budget, "_KEY_NAMES", {"fake-plus": "odds_api_plus", "fake-margin": "odds_api"}
    )
    odds_budget.set_run_context(kind)
    return ledger


def _ledger_line(days_ago, kind, cost, key="odds_api_plus", run="r1", league=None) -> str:
    record = {
        "ts": (_NOW - timedelta(days=days_ago)).isoformat(),
        "run": run,
        "kind": kind,
        "key": key,
        "endpoint": "odds",
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


def test_broad_run_allowance_admits_all() -> None:
    decision = odds_budget.broad_run_allowance(
        _TEN_DAYS_LEFT,
        5000,
        ("NBA", "WNBA"),
        100.0,
        {"NBA": 120.0, "WNBA": 60.0},
        _allowance_cfg(),
    )
    assert decision == BudgetDecision(("NBA", "WNBA"), "ok", 1000.0, 200.0, 4000.0)


def test_broad_run_allowance_floor_reserve() -> None:
    decision = odds_budget.broad_run_allowance(
        _TEN_DAYS_LEFT, 1000, ("NBA",), 100.0, {"NBA": 1.0}, _allowance_cfg()
    )
    assert decision == BudgetDecision((), "floor_reserve", 1000.0, 0.0, 0.0)


def test_broad_run_allowance_partial_respects_priority() -> None:
    decision = odds_budget.broad_run_allowance(
        _TEN_DAYS_LEFT,
        5000,
        ("NFL", "MLB", "NBA"),
        100.0,
        {"NBA": 150.0, "MLB": 100.0, "NFL": 40.0},
        _allowance_cfg(),
    )
    # per_slot 200: NBA (150) admits leaving 50, MLB (100) no longer fits, NFL
    # (40) still does — admission order comes from cfg, not the active tuple.
    assert decision == BudgetDecision(("NBA", "NFL"), "partial", 1000.0, 200.0, 4000.0)


def test_broad_run_allowance_no_fit() -> None:
    decision = odds_budget.broad_run_allowance(
        _TEN_DAYS_LEFT, 5000, ("NBA",), 100.0, {"NBA": 300.0}, _allowance_cfg()
    )
    assert decision == BudgetDecision((), "no_fit", 1000.0, 200.0, 4000.0)


def test_broad_run_allowance_unknown_league_uses_default_cost() -> None:
    decision = odds_budget.broad_run_allowance(
        _TEN_DAYS_LEFT,
        5000,
        ("NBA", "MLB"),
        100.0,
        {},
        _allowance_cfg(default_league_run_cost=150),
    )
    assert decision.allowed_leagues == ("NBA",)
    assert decision.reason == "partial"


def test_broad_run_allowance_rolls_skipped_league_forward() -> None:
    cfg = _allowance_cfg()
    early = odds_budget.broad_run_allowance(
        _TEN_DAYS_LEFT, 4000, ("NBA",), 0.0, {"NBA": 500.0}, cfg
    )
    assert early == BudgetDecision((), "no_fit", 0.0, 200.0, 4000.0)
    # Nothing spent, but fewer slots left in the cycle -> a bigger per-slot
    # share admits the league skipped eight days earlier.
    late = odds_budget.broad_run_allowance(
        datetime(2026, 7, 30, tzinfo=UTC), 4000, ("NBA",), 0.0, {"NBA": 500.0}, cfg
    )
    assert late == BudgetDecision(("NBA",), "ok", 0.0, 1000.0, 4000.0)


def test_broad_run_allowance_end_of_cycle_slot_floor() -> None:
    now = datetime(2026, 7, 31, 18, 0, tzinfo=UTC)  # 0.25 days -> 0.5 slots, floored to 1.0
    decision = odds_budget.broad_run_allowance(
        now, 4000, ("NBA",), 0.0, {"NBA": 3000.0}, _allowance_cfg()
    )
    assert decision.per_slot == decision.spendable == 4000.0
    assert decision.allowed_leagues == ("NBA",)


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
    floor_per_day, league_costs = odds_budget.estimate_costs(_NOW, _ESTIMATE_CFG)
    assert floor_per_day == 20.0  # (100 + 40) / 7
    assert league_costs == {"NBA": 75.0, "WNBA": 60.0}  # NBA: mean of run totals 50 and 100


def test_estimate_costs_floor_min_ratchet(monkeypatch, tmp_path) -> None:
    ledger = _governed(monkeypatch, tmp_path)
    ledger.write_text(_ledger_line(1, "close_lines", 14))  # 2/day observed, under the floor
    floor_per_day, _ = odds_budget.estimate_costs(_NOW, _ESTIMATE_CFG)
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
    floor_per_day, league_costs = odds_budget.estimate_costs(_NOW, _ESTIMATE_CFG)
    assert floor_per_day == 20.0  # only the first line counts
    assert league_costs == {}


def test_estimate_costs_seeds_bootstrap_unseen_leagues(monkeypatch, tmp_path) -> None:
    ledger = _governed(monkeypatch, tmp_path)
    ledger.write_text(_ledger_line(1, "broad", 80, run="r1", league="NBA"))
    cfg = {**_ESTIMATE_CFG, "league_seed_costs": {"NBA": 100, "MLB": 300}}
    _, league_costs = odds_budget.estimate_costs(_NOW, cfg)
    # Ledger mean beats the seed; a league the ledger hasn't seen keeps its seed.
    assert league_costs == {"NBA": 80.0, "MLB": 300}


def test_estimate_costs_skips_torn_final_line(monkeypatch, tmp_path) -> None:
    ledger = _governed(monkeypatch, tmp_path)
    ledger.write_text(_ledger_line(1, "close_lines", 140) + '{"ts": "2026-07-10T')
    floor_per_day, league_costs = odds_budget.estimate_costs(_NOW, _ESTIMATE_CFG)
    assert floor_per_day == 20.0
    assert league_costs == {}


def test_estimate_costs_prunes_stale_records(monkeypatch, tmp_path) -> None:
    ledger = _governed(monkeypatch, tmp_path)
    recent = _ledger_line(1, "close_lines", 140)
    ledger.write_text(_ledger_line(50, "close_lines", 999) + recent)
    odds_budget.estimate_costs(_NOW, _ESTIMATE_CFG)
    assert ledger.read_text() == recent
