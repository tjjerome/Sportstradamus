"""Pin settlement of the simulated-bettor ledger (``_ledger_settlement``):
hand-checked P&L against the real Underdog payout config, CLV-join coverage,
O(distinct legs) resolve cost, and entries-JSONL immutability.

Every test monkeypatches ``_ledger_store.entries_path`` into ``tmp_path`` and
``_ledger_bankroll.already_settled_ids`` to an empty set (the sibling module
this file's own settlement code imports isn't built yet in a parallel-track
session -- these tests stub the one call it makes). No network, xdist-safe.
"""

from __future__ import annotations

import datetime
from decimal import Decimal

import pandas as pd
import pytest

from sportstradamus import clv
from sportstradamus.helpers import io as helpers_io
from sportstradamus.helpers import underdog_payouts
from sportstradamus.prediction.payouts import SLEEPER_FULL_REFUND_MAX_SIZE, payout_curve_for
from sportstradamus.strategies import _ledger_bankroll, _ledger_settlement, _ledger_store

DATE = datetime.date(2026, 7, 12)

_GAMELOG = pd.DataFrame(
    [
        {"DATE": "2026-07-12", "TEAM": "BOS", "PLAYER": "Player A", "PTS": 25, "AST": 6},
        {"DATE": "2026-07-12", "TEAM": "BOS", "PLAYER": "Player B", "PTS": 10, "AST": 4},
        {"DATE": "2026-07-12", "TEAM": "LAL", "PLAYER": "Player C", "PTS": 18, "AST": 8},
        {"DATE": "2026-07-12", "TEAM": "LAL", "PLAYER": "Player D", "PTS": 30, "AST": 2},
    ]
)


class _StubStats:
    def __init__(self, gamelog):
        self.gamelog = gamelog
        self.log_strings = {"date": "DATE", "team": "TEAM", "player": "PLAYER"}


def _leg(
    player: str,
    stat: str,
    line: float,
    bet: str,
    game: str,
    *,
    league: str = "NBA",
    date: str = "2026-07-12",
    win_prob: float = 0.6,
    platform: str = "Underdog",
) -> dict:
    return {
        "player": player,
        "team": game.split("/")[0],
        "market": stat,
        "stat": stat,
        "bet": bet,
        "line": line,
        "league": league,
        "game": game,
        "date": date,
        "platform": platform,
        "win_prob": win_prob,
        "boost": 1.0,
        "push_prob": 0.0,
        "kelly": 0.05,
    }


def _record(
    rec_id: str,
    legs: list[dict],
    *,
    contest_variant: str = "power",
    entry_size: int | None = None,
    stake: str = "10",
    persona: str = "safe",
    run_slot: str = "morning",
    replicate_id: int = 0,
    platform: str = "Underdog",
) -> dict:
    return {
        "id": rec_id,
        "legs": [f"{leg['player']} {leg['bet']} {leg['line']}" for leg in legs],
        "canonical_legs": legs,
        "legs_players": sorted({leg["player"] for leg in legs}),
        "lines": [leg["line"] for leg in legs],
        "model_probs": [leg["win_prob"] for leg in legs],
        "book_devig": [0.55 for _ in legs],
        "stake": stake,
        "policy_version": "policy_v1",
        "git_sha": "deadbeef",
        "committed_at": "2026-07-12T12:00:00+00:00",
        "persona": persona,
        "run_slot": run_slot,
        "replicate_id": replicate_id,
        "game_span": len({leg["game"] for leg in legs}),
        "contest_variant": contest_variant,
        "entry_size": entry_size if entry_size is not None else len(legs),
        "joint_prob": 0.3,
        "payout_multiplier": 3.0,
        "ev": 0.1,
        "date": DATE.isoformat(),
        "platform": platform,
    }


def _redirect(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        _ledger_store, "entries_path", lambda date: tmp_path / f"{date.isoformat()}.jsonl"
    )
    monkeypatch.setattr(_ledger_bankroll, "already_settled_ids", lambda date=None: set())
    monkeypatch.setattr(helpers_io, "read_history", pd.DataFrame)
    monkeypatch.setattr(_ledger_settlement, "read_history", pd.DataFrame)


class _NoopArchive:
    """CLV-join stand-in: every lookup misses (returns NaN), exercising
    fill_from_archive's composite-fallback branch, not an error path."""

    def get_composite_under_prob(self, league, market, date, player, *, at=None):
        return float("nan")

    def get_ev(self, league, market, date, player, *, at=None):
        return float("nan")


# --- 1. Hand-checked settlement against real underdog_payouts.json -------------


def test_two_leg_power_both_hit_matches_real_power_2_value(monkeypatch, tmp_path) -> None:
    _redirect(monkeypatch, tmp_path)
    stats = {"NBA": _StubStats(_GAMELOG)}
    legs = [
        _leg("Player A", "PTS", 20.5, "Over", "BOS/LAL"),
        _leg("Player B", "AST", 2.5, "Over", "BOS/LAL"),
    ]
    record = _record("power-hit", legs, contest_variant="power", stake="10")
    _ledger_store.append_entries(DATE, [record])

    settled = _ledger_settlement.settle_day(DATE, stats, _NoopArchive())

    assert len(settled) == 1
    row = settled[0]
    assert row["misses"] == 0
    assert row["effective_size"] == 2
    expected_mult = float(underdog_payouts["power"][2])
    assert row["realized_multiplier"] == pytest.approx(expected_mult)
    assert row["payout"] == Decimal(str(expected_mult)) * Decimal("10")
    assert row["pnl"] == row["payout"] - Decimal("10")


def test_four_leg_flex_one_miss_matches_real_flex_4_value(monkeypatch, tmp_path) -> None:
    _redirect(monkeypatch, tmp_path)
    stats = {"NBA": _StubStats(_GAMELOG)}
    legs = [
        _leg("Player A", "PTS", 20.5, "Over", "BOS/LAL"),  # 25 -> hit
        _leg("Player B", "AST", 2.5, "Over", "BOS/LAL"),  # 4 -> hit
        _leg("Player C", "PTS", 25.5, "Over", "BOS/LAL"),  # 18 -> miss
        _leg("Player D", "AST", 1.5, "Over", "BOS/LAL"),  # 2 -> hit
    ]
    record = _record("flex-miss1", legs, contest_variant="flex", stake="20")
    _ledger_store.append_entries(DATE, [record])

    settled = _ledger_settlement.settle_day(DATE, stats, _NoopArchive())

    assert len(settled) == 1
    row = settled[0]
    assert row["misses"] == 1
    assert row["effective_size"] == 4
    expected_mult = float(underdog_payouts["flex"][4][1])
    assert row["realized_multiplier"] == pytest.approx(expected_mult)
    assert row["payout"] == Decimal(str(expected_mult)) * Decimal("20")


def test_push_reduces_effective_size_and_uses_reduced_curve(monkeypatch, tmp_path) -> None:
    _redirect(monkeypatch, tmp_path)
    stats = {"NBA": _StubStats(_GAMELOG)}
    legs = [
        _leg("Player A", "PTS", 25.0, "Over", "BOS/LAL"),  # 25 == line -> push
        _leg("Player B", "AST", 2.5, "Over", "BOS/LAL"),  # 4 -> hit
        _leg("Player C", "PTS", 10.5, "Over", "BOS/LAL"),  # 18 -> hit
    ]
    record = _record("power-push", legs, contest_variant="power", entry_size=3, stake="15")
    _ledger_store.append_entries(DATE, [record])

    settled = _ledger_settlement.settle_day(DATE, stats, _NoopArchive())

    assert len(settled) == 1
    row = settled[0]
    assert row["pushes"] == 1
    assert row["misses"] == 0
    assert row["effective_size"] == 2  # 3 - 1 push
    expected_mult = float(underdog_payouts["power"][2])
    assert row["realized_multiplier"] == pytest.approx(expected_mult)


def test_settle_entry_reads_platform_from_record_and_passes_to_realized_multiplier(
    monkeypatch, tmp_path
) -> None:
    _redirect(monkeypatch, tmp_path)
    stats = {"NBA": _StubStats(_GAMELOG)}
    legs = [
        _leg("Player A", "PTS", 25.0, "Over", "BOS/LAL", platform="Sleeper"),  # 25==line -> push
        _leg("Player B", "AST", 5.5, "Over", "BOS/LAL", platform="Sleeper"),  # 4 -> miss
    ]
    record = _record(
        "sleeper-push-refund",
        legs,
        contest_variant="power",
        entry_size=2,
        stake="10",
        platform="Sleeper",
    )
    _ledger_store.append_entries(DATE, [record])

    settled = _ledger_settlement.settle_day(DATE, stats, _NoopArchive())

    assert len(settled) == 1
    row = settled[0]
    assert row["pushes"] == 1
    assert row["misses"] == 1
    assert row["realized_multiplier"] == pytest.approx(1.0)
    assert row["payout"] == row["stake"]
    assert row["pnl"] == Decimal("0")


def test_settle_entry_missing_platform_key_defaults_to_underdog(monkeypatch, tmp_path) -> None:
    _redirect(monkeypatch, tmp_path)
    stats = {"NBA": _StubStats(_GAMELOG)}
    legs = [
        _leg("Player A", "PTS", 25.0, "Over", "BOS/LAL"),  # 25 == line -> push
        _leg("Player B", "AST", 5.5, "Over", "BOS/LAL"),  # 4 -> miss
    ]
    record = _record("no-platform-key", legs, contest_variant="power", entry_size=2, stake="10")
    del record["platform"]
    _ledger_store.append_entries(DATE, [record])

    settled = _ledger_settlement.settle_day(DATE, stats, _NoopArchive())

    assert len(settled) == 1
    row = settled[0]
    assert row["pushes"] == 1
    assert row["misses"] == 1
    assert row["realized_multiplier"] == pytest.approx(0.0)
    assert row["payout"] == Decimal("0")


# --- 2. CLV coverage: >=90% of distinct legs resolve a non-NaN Model CLV -------


def test_join_clv_populates_at_least_90pct_of_legs(monkeypatch) -> None:
    n = 20
    legs = {
        (f"Player {i}", "PTS", 20.5, "Over", "NBA", "2026-07-12"): _leg(
            f"Player {i}", "PTS", 20.5, "Over", "BOS/LAL"
        )
        for i in range(n)
    }

    def _fake_fill_from_archive(history, archive):
        history = history.copy()
        for i in range(n):
            mask = history["Player"] == f"Player {i}"
            if i < 18:
                history.loc[mask, "Close Market Prob"] = 0.58
                history.loc[mask, "Market CLV"] = 0.03
                history.loc[mask, "Model CLV"] = 0.02
        return history

    monkeypatch.setattr(clv, "fill_from_archive", _fake_fill_from_archive)
    monkeypatch.setattr(_ledger_settlement, "read_history", pd.DataFrame)

    result = _ledger_settlement.join_clv(legs, _NoopArchive())

    resolved = sum(1 for v in result.values() if pd.notna(v["model_clv"]))
    assert resolved / n >= 0.90


def test_join_clv_calls_fill_from_archive_exactly_once(monkeypatch) -> None:
    legs = {
        (f"Player {i}", "PTS", 20.5, "Over", "NBA", "2026-07-12"): _leg(
            f"Player {i}", "PTS", 20.5, "Over", "BOS/LAL"
        )
        for i in range(5)
    }
    calls = []

    def _spy(history, archive):
        calls.append(len(history))
        return history

    monkeypatch.setattr(clv, "fill_from_archive", _spy)
    monkeypatch.setattr(_ledger_settlement, "read_history", pd.DataFrame)

    _ledger_settlement.join_clv(legs, _NoopArchive())

    assert len(calls) == 1
    assert calls[0] == 5


# --- 3. Distinct-leg dedup: shared leg across entries resolves once ------------


def test_shared_leg_across_two_entries_resolves_once(monkeypatch, tmp_path) -> None:
    _redirect(monkeypatch, tmp_path)
    stats = {"NBA": _StubStats(_GAMELOG)}
    shared_leg = _leg("Player A", "PTS", 20.5, "Over", "BOS/LAL")
    record_1 = _record(
        "safe-rep0",
        [shared_leg],
        persona="safe",
        replicate_id=0,
        entry_size=2,
        contest_variant="power",
    )
    record_2 = _record(
        "highev-rep7",
        [dict(shared_leg)],
        persona="high_ev",
        replicate_id=7,
        entry_size=2,
        contest_variant="power",
    )
    # both entries need entry_size=2 to be a valid power size but only cite one
    # leg each here -- fine, settle_entry only cares about outcomes per leg cited.
    _ledger_store.append_entries(DATE, [record_1, record_2])

    calls = []
    import sportstradamus.strategies._ledger_settlement as mod

    real_resolve_leg = mod._resolve_leg

    def _spy_resolve_leg(game, ls, leg):
        calls.append(leg["player"])
        return real_resolve_leg(game, ls, leg)

    monkeypatch.setattr(mod, "_resolve_leg", _spy_resolve_leg)

    settled = mod.settle_day(DATE, stats, _NoopArchive())

    assert len(settled) == 2
    assert calls == ["Player A"]  # resolved once, not twice


# --- 4. realized_multiplier against every real curve value ---------------------


def test_realized_multiplier_matches_every_power_size() -> None:
    for size_str, mult in underdog_payouts["power"].items():
        size = int(size_str)
        assert _ledger_settlement.realized_multiplier("power", size, size, 0) == pytest.approx(
            float(mult)
        )
        assert _ledger_settlement.realized_multiplier("power", size, size, 1) == pytest.approx(0.0)


def test_realized_multiplier_matches_every_flex_miss_tier() -> None:
    for size_str, row in underdog_payouts["flex"].items():
        size = int(size_str)
        for misses, mult in enumerate(row):
            assert _ledger_settlement.realized_multiplier(
                "flex", size, size, misses
            ) == pytest.approx(float(mult))


def test_realized_multiplier_sub_minimum_size_no_loss_is_refund() -> None:
    assert _ledger_settlement.realized_multiplier("power", 3, 1, 0) == pytest.approx(1.0)


def test_realized_multiplier_sub_minimum_size_with_loss_is_bust() -> None:
    assert _ledger_settlement.realized_multiplier("power", 3, 1, 1) == pytest.approx(0.0)


def test_realized_multiplier_out_of_range_misses_is_bust() -> None:
    assert _ledger_settlement.realized_multiplier("flex", 6, 6, 6) == pytest.approx(0.0)


# --- 4.5 realized_multiplier: Sleeper 2-pick full-refund divergence ------------


def test_realized_multiplier_sleeper_two_leg_push_with_surviving_miss_refunds_in_full() -> None:
    assert _ledger_settlement.realized_multiplier(
        "power", 2, 1, 1, platform="Sleeper"
    ) == pytest.approx(1.0)


def test_realized_multiplier_sleeper_two_leg_push_no_surviving_miss_still_refunds() -> None:
    assert _ledger_settlement.realized_multiplier(
        "power", 2, 1, 0, platform="Sleeper"
    ) == pytest.approx(1.0)


def test_realized_multiplier_underdog_two_leg_push_with_surviving_miss_still_busts() -> None:
    assert _ledger_settlement.realized_multiplier(
        "power", 2, 1, 1, platform="Underdog"
    ) == pytest.approx(0.0)


def test_realized_multiplier_sleeper_three_leg_push_uses_generic_rule_not_full_refund() -> None:
    assert SLEEPER_FULL_REFUND_MAX_SIZE == 2
    _, curve = payout_curve_for("Sleeper", "power")
    expected = curve[2][0]
    assert _ledger_settlement.realized_multiplier(
        "power", 3, 2, 0, platform="Sleeper"
    ) == pytest.approx(expected)


# --- 5. Entries JSONL immutability ----------------------------------------------


def test_settle_day_never_mutates_entries_jsonl_bytes(monkeypatch, tmp_path) -> None:
    _redirect(monkeypatch, tmp_path)
    stats = {"NBA": _StubStats(_GAMELOG)}
    legs = [
        _leg("Player A", "PTS", 20.5, "Over", "BOS/LAL"),
        _leg("Player B", "AST", 2.5, "Over", "BOS/LAL"),
    ]
    record = _record("immut-1", legs, contest_variant="power", stake="10")
    _ledger_store.append_entries(DATE, [record])
    path = tmp_path / f"{DATE.isoformat()}.jsonl"
    before = path.read_bytes()

    _ledger_settlement.settle_day(DATE, stats, _NoopArchive())

    after = path.read_bytes()
    assert before == after


# --- settleable_entries: incomplete slate is excluded ---------------------------


def test_settleable_entries_excludes_entry_with_unplayed_leg(monkeypatch, tmp_path) -> None:
    _redirect(monkeypatch, tmp_path)
    stats = {"NBA": _StubStats(_GAMELOG)}
    legs = [
        _leg("Player A", "PTS", 20.5, "Over", "BOS/LAL"),
        _leg("Nobody Yet", "PTS", 10.5, "Over", "MIA/NYK"),  # no gamelog row -> unplayed
    ]
    record = _record("incomplete", legs, contest_variant="power", entry_size=2, stake="10")
    _ledger_store.append_entries(DATE, [record])

    result = _ledger_settlement.settleable_entries(DATE, stats)

    assert result == []


def test_settleable_entries_excludes_already_settled(monkeypatch, tmp_path) -> None:
    _redirect(monkeypatch, tmp_path)
    stats = {"NBA": _StubStats(_GAMELOG)}
    legs = [
        _leg("Player A", "PTS", 20.5, "Over", "BOS/LAL"),
        _leg("Player B", "AST", 2.5, "Over", "BOS/LAL"),
    ]
    record = _record("already-settled", legs, contest_variant="power", stake="10")
    _ledger_store.append_entries(DATE, [record])
    monkeypatch.setattr(
        _ledger_bankroll, "already_settled_ids", lambda date=None: {"already-settled"}
    )

    result = _ledger_settlement.settleable_entries(DATE, stats)

    assert result == []


def test_settle_day_empty_slate_returns_empty_list(monkeypatch, tmp_path) -> None:
    _redirect(monkeypatch, tmp_path)
    stats = {"NBA": _StubStats(_GAMELOG)}

    assert _ledger_settlement.settle_day(DATE, stats, _NoopArchive()) == []
