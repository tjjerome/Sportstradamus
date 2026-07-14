"""End-to-end wiring pins for the simulated-bettor ledger settlement chain.

``test_ledger_settlement.py`` and ``test_ledger_bankroll.py`` already pin each
module's contract in isolation (payout math, CLV coverage, dedup cost,
compounding arithmetic). This file proves the three modules actually compose:
``nightly._resolve_ledger`` reading a real entries JSONL off disk, walking
through the real (unmocked) ``_ledger_settlement.settle_day`` --
``_ledger_bankroll.already_settled_ids`` idempotency check, and landing correct
rows in both ``settled_entries.parquet`` and ``bankroll.parquet``. No test here
re-derives payout curves, CLV percentages, or bankroll compounding -- see the
two files above for that coverage.

Fixture helpers (``_GAMELOG``, ``_StubStats``, ``_leg``, ``_record``,
``_NoopArchive``) mirror ``test_ledger_settlement.py``'s exactly, copied rather
than imported -- this codebase's convention for reusing a peer test file's
fixture shape (see ``test_corr_summary.py``'s ``_isolated_correlate_data_dir``
docstring for the same pattern) rather than importing across test modules.

Every test monkeypatches ``_ledger_store.entries_path``, ``nightly._LEDGER_ENTRIES_DIR``,
``_ledger_bankroll.SETTLED_ENTRIES_PATH``/``BANKROLL_PATH``, and ``nightly.Archive``
into ``tmp_path`` -- this suite runs under pytest-xdist with ``-n auto``, so a
test touching the real ``data/ledger/`` path would race across workers. No
network, no real DuckDB connection.
"""

from __future__ import annotations

import datetime
import json
from decimal import Decimal

import pandas as pd
import pytest

from sportstradamus import nightly
from sportstradamus.helpers import io as helpers_io
from sportstradamus.helpers import underdog_payouts
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
        "platform": "Underdog",
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
    }


class _NoopArchive:
    """CLV-join stand-in: every lookup misses (returns NaN), exercising
    fill_from_archive's composite-fallback branch, not an error path."""

    def get_composite_under_prob(self, league, market, date, player, *, at=None):
        return float("nan")

    def get_ev(self, league, market, date, player, *, at=None):
        return float("nan")


def _redirect(monkeypatch, tmp_path) -> None:
    """Point every path the full ``_resolve_ledger`` chain touches at ``tmp_path``.

    Distinct from ``test_ledger_settlement.py``'s ``_redirect``: this one does
    NOT stub ``_ledger_bankroll.already_settled_ids`` -- the whole point of
    these tests is exercising the real dedup call, not a mock of it. Also
    redirects ``nightly._LEDGER_ENTRIES_DIR`` (a separate module-level constant
    from ``_ledger_store.entries_path``, but must resolve to the same
    directory) and ``nightly.Archive`` (avoids opening the real DuckDB file).
    """
    entries_dir = tmp_path / "entries"
    monkeypatch.setattr(
        _ledger_store, "entries_path", lambda date: entries_dir / f"{date.isoformat()}.jsonl"
    )
    monkeypatch.setattr(nightly, "_LEDGER_ENTRIES_DIR", entries_dir)
    monkeypatch.setattr(
        _ledger_bankroll, "SETTLED_ENTRIES_PATH", tmp_path / "settled_entries.parquet"
    )
    monkeypatch.setattr(_ledger_bankroll, "BANKROLL_PATH", tmp_path / "bankroll.parquet")
    monkeypatch.setattr(nightly, "Archive", _NoopArchive)
    monkeypatch.setattr(helpers_io, "read_history", pd.DataFrame)
    monkeypatch.setattr(_ledger_settlement, "read_history", pd.DataFrame)


# --- 1. Full chain, hand-checked ------------------------------------------------


def test_resolve_ledger_settles_writes_both_parquets_end_to_end(monkeypatch, tmp_path) -> None:
    _redirect(monkeypatch, tmp_path)
    stats = {"NBA": _StubStats(_GAMELOG)}

    power_legs = [
        _leg("Player A", "PTS", 20.5, "Over", "BOS/LAL"),  # 25 -> hit
        _leg("Player B", "AST", 2.5, "Over", "BOS/LAL"),  # 4 -> hit
    ]
    flex_legs = [
        _leg("Player A", "PTS", 20.5, "Over", "BOS/LAL"),  # 25 -> hit
        _leg("Player B", "AST", 2.5, "Over", "BOS/LAL"),  # 4 -> hit
        _leg("Player C", "PTS", 25.5, "Over", "BOS/LAL"),  # 18 -> miss
        _leg("Player D", "AST", 1.5, "Over", "BOS/LAL"),  # 2 -> hit
    ]
    power_record = _record(
        "power-hit", power_legs, contest_variant="power", stake="10", persona="safe", replicate_id=0
    )
    flex_record = _record(
        "flex-miss1",
        flex_legs,
        contest_variant="flex",
        stake="20",
        persona="high_ev",
        replicate_id=1,
    )
    _ledger_store.append_entries(DATE, [power_record, flex_record])

    n_settled = nightly._resolve_ledger(stats, history_only=False)

    assert n_settled == 2

    settled_path = tmp_path / "settled_entries.parquet"
    bankroll_path = tmp_path / "bankroll.parquet"
    assert settled_path.exists()
    assert bankroll_path.exists()

    settled = pd.read_parquet(settled_path).set_index("id")
    expected_power_mult = float(underdog_payouts["power"][2])
    expected_flex_mult = float(underdog_payouts["flex"][4][1])
    assert settled.loc["power-hit", "misses"] == 0
    assert settled.loc["power-hit", "realized_multiplier"] == pytest.approx(expected_power_mult)
    assert settled.loc["power-hit", "payout"] == pytest.approx(expected_power_mult * 10)
    assert settled.loc["flex-miss1", "misses"] == 1
    assert settled.loc["flex-miss1", "realized_multiplier"] == pytest.approx(expected_flex_mult)
    assert settled.loc["flex-miss1", "payout"] == pytest.approx(expected_flex_mult * 20)

    bankroll = pd.read_parquet(bankroll_path)
    assert len(bankroll) == 2  # (safe, 0) and (high_ev, 1)
    safe_row = bankroll.loc[(bankroll["persona"] == "safe") & (bankroll["replicate_id"] == 0)].iloc[
        0
    ]
    high_ev_row = bankroll.loc[
        (bankroll["persona"] == "high_ev") & (bankroll["replicate_id"] == 1)
    ].iloc[0]

    expected_power_payout = Decimal(str(expected_power_mult)) * Decimal("10")
    expected_power_pnl = expected_power_payout - Decimal("10")
    expected_flex_payout = Decimal(str(expected_flex_mult)) * Decimal("20")
    expected_flex_pnl = expected_flex_payout - Decimal("20")
    assert Decimal(str(safe_row["ending_bankroll"])) == Decimal("5000") + expected_power_pnl
    assert Decimal(str(high_ev_row["ending_bankroll"])) == Decimal("5000") + expected_flex_pnl


# --- 2. history_only short-circuits ----------------------------------------------


def test_resolve_ledger_history_only_writes_neither_parquet(monkeypatch, tmp_path) -> None:
    _redirect(monkeypatch, tmp_path)
    stats = {"NBA": _StubStats(_GAMELOG)}
    legs = [
        _leg("Player A", "PTS", 20.5, "Over", "BOS/LAL"),
        _leg("Player B", "AST", 2.5, "Over", "BOS/LAL"),
    ]
    _ledger_store.append_entries(DATE, [_record("power-hit", legs, stake="10")])

    n_settled = nightly._resolve_ledger(stats, history_only=True)

    assert n_settled == 0
    assert not (tmp_path / "settled_entries.parquet").exists()
    assert not (tmp_path / "bankroll.parquet").exists()


# --- 3. Idempotent re-run at the full-chain level --------------------------------


def test_resolve_ledger_repeat_call_settles_nothing_new_and_parquets_unchanged(
    monkeypatch, tmp_path
) -> None:
    _redirect(monkeypatch, tmp_path)
    stats = {"NBA": _StubStats(_GAMELOG)}
    legs = [
        _leg("Player A", "PTS", 20.5, "Over", "BOS/LAL"),
        _leg("Player B", "AST", 2.5, "Over", "BOS/LAL"),
    ]
    _ledger_store.append_entries(DATE, [_record("power-hit", legs, stake="10")])

    first = nightly._resolve_ledger(stats, history_only=False)
    settled_before = pd.read_parquet(tmp_path / "settled_entries.parquet")
    bankroll_before = pd.read_parquet(tmp_path / "bankroll.parquet")

    second = nightly._resolve_ledger(stats, history_only=False)
    settled_after = pd.read_parquet(tmp_path / "settled_entries.parquet")
    bankroll_after = pd.read_parquet(tmp_path / "bankroll.parquet")

    assert first == 1
    assert second == 0
    pd.testing.assert_frame_equal(settled_before, settled_after)
    pd.testing.assert_frame_equal(bankroll_before, bankroll_after)


# --- 4. _ledger_entry_dates ------------------------------------------------------


def test_ledger_entry_dates_empty_dir_returns_empty_list(monkeypatch, tmp_path) -> None:
    entries_dir = tmp_path / "entries"
    monkeypatch.setattr(nightly, "_LEDGER_ENTRIES_DIR", entries_dir)

    assert nightly._ledger_entry_dates() == []


def test_ledger_entry_dates_nonexistent_dir_returns_empty_list(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(nightly, "_LEDGER_ENTRIES_DIR", tmp_path / "does_not_exist")

    assert nightly._ledger_entry_dates() == []


def test_ledger_entry_dates_returns_sorted_dates_from_multiple_files(monkeypatch, tmp_path) -> None:
    entries_dir = tmp_path / "entries"
    entries_dir.mkdir()
    for stem in ("2026-07-14", "2026-07-12", "2026-07-13"):
        (entries_dir / f"{stem}.jsonl").write_text("")
    monkeypatch.setattr(nightly, "_LEDGER_ENTRIES_DIR", entries_dir)

    assert nightly._ledger_entry_dates() == [
        datetime.date(2026, 7, 12),
        datetime.date(2026, 7, 13),
        datetime.date(2026, 7, 14),
    ]


# --- 5. Entries JSONL immutability, full-chain level ------------------------------


def test_resolve_ledger_never_mutates_entries_jsonl_bytes(monkeypatch, tmp_path) -> None:
    _redirect(monkeypatch, tmp_path)
    stats = {"NBA": _StubStats(_GAMELOG)}
    legs = [
        _leg("Player A", "PTS", 20.5, "Over", "BOS/LAL"),
        _leg("Player B", "AST", 2.5, "Over", "BOS/LAL"),
    ]
    _ledger_store.append_entries(DATE, [_record("power-hit", legs, stake="10")])
    path = tmp_path / "entries" / f"{DATE.isoformat()}.jsonl"
    before = path.read_bytes()

    nightly._resolve_ledger(stats, history_only=False)

    after = path.read_bytes()
    assert before == after


# --- 6. _write_resolve_meta gained the ledger field -------------------------------


def test_write_resolve_meta_includes_ledger_resolved_field(monkeypatch, tmp_path) -> None:
    real_files = nightly.pkg_resources.files

    def _fake_files(pkg):
        return tmp_path if pkg is nightly.data else real_files(pkg)

    monkeypatch.setattr(nightly.pkg_resources, "files", _fake_files)
    (tmp_path / "runtime").mkdir()
    history = pd.DataFrame({"Actual": [1.0, float("nan")]})

    nightly._write_resolve_meta(
        history,
        n_resolved_hist=3,
        n_resolved_parl=2,
        n_resolved_slips=1,
        n_resolved_ledger=5,
    )

    meta = json.loads((tmp_path / "runtime" / "resolve_meta.json").read_text())
    assert meta["ledger_resolved"] == 5
