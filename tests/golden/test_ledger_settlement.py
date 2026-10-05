"""Pin settlement of the simulated-bettor ledger (``_ledger_settlement``):
hand-checked P&L against the real Underdog payout config, the payout rule it
shares with the pricers, how each policy version's records settle, CLV-join
coverage, O(distinct legs) resolve cost, and entries-JSONL immutability.

Every test monkeypatches ``_ledger_store.entries_path`` into ``tmp_path`` and
``_ledger_bankroll.already_settled_ids`` to an empty set (the sibling module
this file's own settlement code imports isn't built yet in a parallel-track
session -- these tests stub the one call it makes). No network, xdist-safe.
"""

from __future__ import annotations

import datetime
import itertools
import json
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from sportstradamus import clv
from sportstradamus.helpers import io as helpers_io
from sportstradamus.helpers import underdog_payouts
from sportstradamus.prediction.payouts import (
    LEG_LOSS,
    LEG_PUSH,
    LEG_WIN,
    SLEEPER_FULL_REFUND_MAX_SIZE,
    expected_payout_with_pushes,
    outcome_payouts,
    payout_curve_for,
)
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
    boost: float = 1.0,
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
        "boost": boost,
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
    pair_modifier: float | None = None,
) -> dict:
    """A committed record. With no ``pair_modifier`` it is a ``policy_v1`` record, the
    shape on disk before the field existed; with one it is a ``policy_v2`` record."""
    versioned = (
        {"policy_version": "policy_v1"}
        if pair_modifier is None
        else {"policy_version": "policy_v2", "pair_modifier": pair_modifier}
    )
    return versioned | {
        "id": rec_id,
        "legs": [f"{leg['player']} {leg['bet']} {leg['line']}" for leg in legs],
        "canonical_legs": legs,
        "legs_players": sorted({leg["player"] for leg in legs}),
        "lines": [leg["line"] for leg in legs],
        "model_probs": [leg["win_prob"] for leg in legs],
        "book_devig": [0.55 for _ in legs],
        "stake": stake,
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


def test_settle_entry_reads_platform_from_record_and_applies_its_push_rule(
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
    shared_legs = [
        _leg("Player A", "PTS", 20.5, "Over", "BOS/LAL"),
        _leg("Player B", "AST", 2.5, "Over", "BOS/LAL"),
    ]
    record_1 = _record("safe-rep0", shared_legs, persona="safe", replicate_id=0)
    record_2 = _record(
        "highev-rep7", [dict(leg) for leg in shared_legs], persona="high_ev", replicate_id=7
    )
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
    assert calls == ["Player A", "Player B"]  # each resolved once, not once per entry


# --- 4. The payout rule settlement shares with the pricers ----------------------
# Hand-derived from docs/underdog_api.md §6.8: an entry pays its table tier, times the
# pick multipliers Underdog quotes that tier on, times the pair modifier. The picks at
# 0.87 / 1.16 / 0.74 are the app quote captured there: Power 4.85, Flex 2.42 / 1.10.

_PICKS = np.array([0.87, 1.16, 0.74])
_UNDERDOG_POWER = payout_curve_for("Underdog", "power")[1]
_UNDERDOG_FLEX = payout_curve_for("Underdog", "flex")[1]


def _pays(legs: list[int], boost, curve: dict, **kwargs) -> float:
    return float(outcome_payouts(np.array([legs]), boost, curve, **kwargs)[0])


def test_power_all_hit_pays_the_table_times_every_legs_multiplier() -> None:
    paid = _pays([LEG_WIN] * 3, _PICKS, _UNDERDOG_POWER)

    assert paid == pytest.approx(6.5 * 0.87 * 1.16 * 0.74)
    assert paid == pytest.approx(4.85, abs=0.005)


def test_flex_loss_tier_pays_on_the_largest_multipliers_whichever_leg_lost() -> None:
    lost_the_largest = [LEG_WIN, LEG_LOSS, LEG_WIN]

    assert _pays(lost_the_largest, _PICKS, _UNDERDOG_FLEX) == pytest.approx(1.09 * 0.87 * 1.16)
    assert _pays([LEG_WIN] * 3, _PICKS, _UNDERDOG_FLEX) == pytest.approx(3.25 * 0.87 * 1.16 * 0.74)


def test_push_drops_the_entry_one_size_and_takes_its_multiplier_along() -> None:
    pushed_the_074 = [LEG_WIN, LEG_WIN, LEG_PUSH]

    assert _pays(pushed_the_074, _PICKS, _UNDERDOG_POWER) == pytest.approx(3.5 * 0.87 * 1.16)


def test_pair_modifier_multiplies_what_the_legs_pay() -> None:
    paid = _pays([LEG_WIN] * 3, _PICKS, _UNDERDOG_POWER, pair_modifier=0.86)

    assert paid == pytest.approx(6.5 * 0.87 * 1.16 * 0.74 * 0.86)


def test_refund_pays_exactly_the_stake_back() -> None:
    priced = np.array([1.5, 1.2])

    assert _pays([LEG_PUSH, LEG_WIN], priced, _UNDERDOG_POWER, pair_modifier=0.86) == 1.0
    assert _pays([LEG_PUSH, LEG_LOSS], priced, _UNDERDOG_POWER, pair_modifier=0.86) == 0.0


def test_sleeper_two_pick_entry_refunds_in_full_on_any_push() -> None:
    curve = payout_curve_for("Sleeper", "power")[1]
    posted = np.array([1.78, 1.62])

    def paid(legs: list[int]) -> float:
        return _pays(legs, posted, curve, full_refund_below_size=SLEEPER_FULL_REFUND_MAX_SIZE)

    assert paid([LEG_PUSH, LEG_LOSS]) == 1.0
    assert paid([LEG_PUSH, LEG_WIN]) == 1.0
    assert paid([LEG_WIN, LEG_WIN]) == pytest.approx(1.0 * 1.78 * 1.62)


def test_pricer_averages_the_rule_settlement_applies() -> None:
    """The pricer's expectation is the probability-weighted sum of what the rule pays
    every outcome, so a priced entry and a settled one cannot follow different rules."""
    p_win = np.array([0.55, 0.60, 0.50, 0.58])
    p_push = np.array([0.0, 0.05, 0.0, 0.0])
    boosts = np.array([0.74, 0.87, 1.16, 1.30])
    leg_probs = [
        {LEG_WIN: win, LEG_PUSH: push, LEG_LOSS: 1.0 - win - push}
        for win, push in zip(p_win, p_push, strict=True)
    ]
    exact = sum(
        np.prod([probs[leg] for probs, leg in zip(leg_probs, legs, strict=True)])
        * _pays(list(legs), boosts, _UNDERDOG_FLEX, pair_modifier=0.9)
        for legs in itertools.product((LEG_LOSS, LEG_PUSH, LEG_WIN), repeat=4)
    )

    priced = expected_payout_with_pushes(
        p_win,
        p_push,
        np.eye(4),
        4,
        boosts,
        _UNDERDOG_FLEX,
        np.random.default_rng(7),
        pair_modifier=0.9,
    )

    assert priced == pytest.approx(exact, abs=0.03)


# --- 4.5 Each policy version's records settle by their own rule ------------------

_HIT, _MISS, _PUSH = 0, 1, None  # analysis._resolve_leg's verdicts


def _settled_multiplier(
    outcomes: list[int | None],
    *,
    contest_variant: str,
    platform: str = "Underdog",
    boosts: list[float] | None = None,
    pair_modifier: float | None = None,
) -> float:
    """Settle one entry whose legs resolved to ``outcomes`` and return what it paid per $1.
    ``pair_modifier=None`` makes it a ``policy_v1`` record (see :func:`_record`)."""
    legs = [
        _leg(f"Player {i}", "PTS", 10.5, "Over", "BOS/LAL", platform=platform, boost=boost)
        for i, boost in enumerate(boosts or [1.0] * len(outcomes))
    ]
    record = _record(
        "entry",
        legs,
        contest_variant=contest_variant,
        platform=platform,
        pair_modifier=pair_modifier,
    )
    leg_outcomes = {
        _ledger_settlement.distinct_leg_key(leg): outcome
        for leg, outcome in zip(legs, outcomes, strict=True)
    }
    return _ledger_settlement.settle_entry(record, leg_outcomes, {})["realized_multiplier"]


def test_policy_v1_record_pays_every_power_size_at_the_bare_table() -> None:
    for size, mult in underdog_payouts["power"].items():
        priced = [1.3] * size  # multipliers a policy_v1 record never settles on
        all_hit = _settled_multiplier([_HIT] * size, contest_variant="power", boosts=priced)
        one_miss = _settled_multiplier(
            [_MISS] + [_HIT] * (size - 1), contest_variant="power", boosts=priced
        )

        assert all_hit == pytest.approx(float(mult))
        assert one_miss == 0.0


def test_policy_v1_record_pays_every_flex_tier_at_the_bare_table() -> None:
    for size, row in underdog_payouts["flex"].items():
        for misses, mult in enumerate(row):
            outcomes = [_MISS] * misses + [_HIT] * (size - misses)
            paid = _settled_multiplier(outcomes, contest_variant="flex", boosts=[1.3] * size)

            assert paid == pytest.approx(float(mult))


@pytest.mark.parametrize(
    ("outcomes", "contest_variant", "platform", "expected"),
    [
        pytest.param([_PUSH, _PUSH, _HIT], "power", "Underdog", 1.0, id="sub-minimum-refund"),
        pytest.param([_PUSH, _PUSH, _MISS], "power", "Underdog", 0.0, id="sub-minimum-bust"),
        pytest.param([_MISS] * 6, "flex", "Underdog", 0.0, id="misses-past-the-curve"),
        pytest.param([_PUSH, _MISS], "power", "Underdog", 0.0, id="underdog-2-pick-push-miss"),
        pytest.param([_PUSH, _MISS], "power", "Sleeper", 1.0, id="sleeper-2-pick-push-miss"),
        pytest.param([_PUSH, _HIT], "power", "Sleeper", 1.0, id="sleeper-2-pick-push-hit"),
        pytest.param(
            [_PUSH, _HIT, _HIT],
            "power",
            "Sleeper",
            payout_curve_for("Sleeper", "power")[1][2][0],
            id="sleeper-3-pick-push-reprices",
        ),
    ],
)
def test_policy_v1_record_push_and_bust_rules(
    outcomes: list[int | None], contest_variant: str, platform: str, expected: float
) -> None:
    paid = _settled_multiplier(outcomes, contest_variant=contest_variant, platform=platform)

    assert paid == pytest.approx(expected)


def test_policy_v2_record_settles_on_its_leg_multipliers_and_pair_modifier(
    monkeypatch, tmp_path
) -> None:
    _redirect(monkeypatch, tmp_path)
    stats = {"NBA": _StubStats(_GAMELOG)}
    legs = [
        _leg("Player A", "PTS", 20.5, "Over", "BOS/LAL", boost=0.87),  # 25 -> hit
        _leg("Player B", "AST", 2.5, "Over", "BOS/LAL", boost=1.16),  # 4 -> hit
        _leg("Player C", "PTS", 10.5, "Over", "BOS/LAL", boost=0.74),  # 18 -> hit
    ]
    _ledger_store.append_entries(
        DATE,
        [
            _record("v2-priced", legs, stake="10", pair_modifier=0.9),
            _record("v1-priced", [dict(leg) for leg in legs], stake="10"),
        ],
    )

    by_id = {row["id"]: row for row in _ledger_settlement.settle_day(DATE, stats, _NoopArchive())}

    v2 = by_id["v2-priced"]
    assert v2["policy_version"] == "policy_v2"
    assert v2["realized_multiplier"] == pytest.approx(6.5 * 0.87 * 1.16 * 0.74 * 0.9)
    assert v2["payout"] == Decimal(str(v2["realized_multiplier"])) * Decimal("10")
    assert v2["pnl"] == v2["payout"] - Decimal("10")
    v1 = by_id["v1-priced"]
    assert v1["policy_version"] == "policy_v1"
    assert v1["realized_multiplier"] == pytest.approx(float(underdog_payouts["power"][3]))


def test_policy_v2_flex_record_one_miss_pays_on_the_largest_multipliers() -> None:
    lost_the_largest = [_HIT, _HIT, _MISS, _HIT]

    paid = _settled_multiplier(
        lost_the_largest, contest_variant="flex", boosts=[0.74, 0.87, 1.30, 1.16], pair_modifier=1.0
    )

    assert paid == pytest.approx(float(underdog_payouts["flex"][4][1]) * 0.87 * 1.30 * 1.16)


def test_policy_v2_sleeper_record_pays_its_posted_multipliers_under_its_own_curve() -> None:
    sleeper_power = payout_curve_for("Sleeper", "power")[1]
    posted = [1.78, 1.62, 1.45]

    three_hit = _settled_multiplier(
        [_HIT] * 3, contest_variant="power", platform="Sleeper", boosts=posted, pair_modifier=1.0
    )
    two_pick_push = _settled_multiplier(
        [_PUSH, _MISS],
        contest_variant="power",
        platform="Sleeper",
        boosts=posted[:2],
        pair_modifier=1.0,
    )

    assert three_hit == pytest.approx(sleeper_power[3][0] * 1.78 * 1.62 * 1.45)
    assert two_pick_push == 1.0


def test_record_of_a_later_policy_version_settles_by_the_current_rule() -> None:
    """Only ``policy_v1`` is held to the bare table. A version this code has never seen
    carries the current record shape and settles on it."""
    legs = [
        _leg(f"Player {i}", "PTS", 10.5, "Over", "BOS/LAL", boost=boost)
        for i, boost in enumerate((0.87, 1.16))
    ]
    record = _record("later", legs, pair_modifier=0.9) | {"policy_version": "policy_v9"}
    leg_outcomes = {_ledger_settlement.distinct_leg_key(leg): _HIT for leg in legs}

    row = _ledger_settlement.settle_entry(record, leg_outcomes, {})

    assert row["policy_version"] == "policy_v9"
    assert row["realized_multiplier"] == pytest.approx(
        float(underdog_payouts["power"][2]) * 0.87 * 1.16 * 0.9
    )


def test_record_stored_before_the_new_fields_loads_and_settles_on_the_bare_table(
    monkeypatch, tmp_path
) -> None:
    """A line as ``policy_v1`` wrote it (no ``pair_modifier``, no ``platform``), read
    straight off disk rather than through ``append_entries``."""
    _redirect(monkeypatch, tmp_path)
    stats = {"NBA": _StubStats(_GAMELOG)}
    record = _record(
        "stored-v1",
        [
            _leg("Player A", "PTS", 20.5, "Over", "BOS/LAL", boost=1.2),
            _leg("Player B", "AST", 2.5, "Over", "BOS/LAL", boost=1.2),
        ],
    )
    del record["platform"]
    assert "pair_modifier" not in record
    (tmp_path / f"{DATE.isoformat()}.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    assert _ledger_store.read_records(DATE) == [record]
    settled = _ledger_settlement.settle_day(DATE, stats, _NoopArchive())

    assert settled[0]["policy_version"] == "policy_v1"
    assert settled[0]["realized_multiplier"] == pytest.approx(float(underdog_payouts["power"][2]))


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
