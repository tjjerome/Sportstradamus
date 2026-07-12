"""Pin parquet persistence + compounding bankroll math of ``_ledger_bankroll``.

Every test monkeypatches ``SETTLED_ENTRIES_PATH``/``BANKROLL_PATH`` to resolve
inside pytest's ``tmp_path`` fixture -- this suite runs under pytest-xdist
with ``-n auto`` (parallel workers), so a test that touched the real
``data/ledger/`` path would race across workers. Rows are hand-constructed
directly against the documented ``settle_day`` contract shape rather than
depending on ``_ledger_settlement`` existing.
"""

from __future__ import annotations

import datetime
from decimal import Decimal

from sportstradamus.strategies import _ledger_bankroll

DATE = datetime.date(2026, 7, 12)
NEXT_DATE = datetime.date(2026, 7, 13)
THIRD_DATE = datetime.date(2026, 7, 14)


def _redirect(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        _ledger_bankroll, "SETTLED_ENTRIES_PATH", tmp_path / "settled_entries.parquet"
    )
    monkeypatch.setattr(_ledger_bankroll, "BANKROLL_PATH", tmp_path / "bankroll.parquet")


def _settled_row(
    rec_id: str,
    *,
    date: datetime.date = DATE,
    persona: str = "safe",
    replicate_id: int = 0,
    stake: str = "10.00",
    payout: str = "0.00",
    pnl: str = "-10.00",
) -> dict:
    return {
        "id": rec_id,
        "date": date.isoformat(),
        "persona": persona,
        "run_slot": "morning",
        "replicate_id": replicate_id,
        "contest_variant": "power",
        "entry_size": 2,
        "effective_size": 2,
        "misses": 1,
        "pushes": 0,
        "realized_multiplier": 0.0,
        "stake": Decimal(stake),
        "payout": Decimal(payout),
        "pnl": Decimal(pnl),
        "game_span": 1,
        "committed_at": "2026-07-12T12:00:00Z",
        "policy_version": "policy_v1",
        "git_sha": "deadbeef",
        "clv_leg_count": 2,
        "model_clv_mean": 0.03,
        "market_clv_mean": 0.01,
        "settled_at": "2026-07-13T04:00:00Z",
    }


def test_write_settled_entries_then_already_settled_ids_round_trip(monkeypatch, tmp_path) -> None:
    _redirect(monkeypatch, tmp_path)
    assert _ledger_bankroll.already_settled_ids() == set()

    rows = [_settled_row("id-1"), _settled_row("id-2")]
    _ledger_bankroll.write_settled_entries(rows)

    assert _ledger_bankroll.already_settled_ids() == {"id-1", "id-2"}
    assert _ledger_bankroll.already_settled_ids(DATE) == {"id-1", "id-2"}
    assert _ledger_bankroll.already_settled_ids(NEXT_DATE) == set()


def test_write_settled_entries_noop_on_empty_input(monkeypatch, tmp_path) -> None:
    _redirect(monkeypatch, tmp_path)
    _ledger_bankroll.write_settled_entries([])
    assert not (tmp_path / "settled_entries.parquet").exists()
    assert _ledger_bankroll.already_settled_ids() == set()


def test_write_settled_entries_appends_across_calls(monkeypatch, tmp_path) -> None:
    _redirect(monkeypatch, tmp_path)
    _ledger_bankroll.write_settled_entries([_settled_row("id-1")])
    _ledger_bankroll.write_settled_entries([_settled_row("id-2")])

    ids = _ledger_bankroll.already_settled_ids()
    assert ids == {"id-1", "id-2"}


def test_parquet_round_trip_preserves_money_float_precision(monkeypatch, tmp_path) -> None:
    _redirect(monkeypatch, tmp_path)
    row = _settled_row("id-1", stake="12.34", payout="37.02", pnl="24.68")
    _ledger_bankroll.write_settled_entries([row])

    df = _ledger_bankroll.read_parquet_safe(_ledger_bankroll.SETTLED_ENTRIES_PATH)
    stored = df.loc[df["id"] == "id-1"].iloc[0]
    assert float(stored["stake"]) == 12.34
    assert float(stored["payout"]) == 37.02
    assert float(stored["pnl"]) == 24.68


def test_update_bankroll_seeds_starting_bankroll_on_first_row(monkeypatch, tmp_path) -> None:
    _redirect(monkeypatch, tmp_path)
    settled = [_settled_row("id-1", pnl="-10.00")]
    _ledger_bankroll.update_bankroll(DATE, settled)

    df = _ledger_bankroll.read_parquet_safe(_ledger_bankroll.BANKROLL_PATH)
    row = df.iloc[0]
    assert Decimal(str(row["starting_bankroll"])) == Decimal("5000")
    assert Decimal(str(row["daily_pnl"])) == Decimal("-10.00")
    assert Decimal(str(row["ending_bankroll"])) == Decimal("4990.00")
    assert row["n_entries_settled"] == 1
    assert row["n_entries_pending"] == 0


def test_update_bankroll_multi_day_compounding_carries_forward(monkeypatch, tmp_path) -> None:
    _redirect(monkeypatch, tmp_path)
    day1 = [_settled_row("id-1", date=DATE, pnl="50.00")]
    _ledger_bankroll.update_bankroll(DATE, day1)

    day2 = [_settled_row("id-2", date=NEXT_DATE, pnl="-20.00")]
    _ledger_bankroll.update_bankroll(NEXT_DATE, day2)

    df = _ledger_bankroll.read_parquet_safe(_ledger_bankroll.BANKROLL_PATH)
    day1_row = df.loc[df["date"] == DATE.isoformat()].iloc[0]
    day2_row = df.loc[df["date"] == NEXT_DATE.isoformat()].iloc[0]

    assert Decimal(str(day1_row["ending_bankroll"])) == Decimal("5050.00")
    assert Decimal(str(day2_row["starting_bankroll"])) == Decimal(str(day1_row["ending_bankroll"]))
    assert Decimal(str(day2_row["ending_bankroll"])) == Decimal("5030.00")


def test_update_bankroll_empty_input_is_pure_noop(monkeypatch, tmp_path) -> None:
    _redirect(monkeypatch, tmp_path)
    _ledger_bankroll.update_bankroll(DATE, [_settled_row("id-1", date=DATE, pnl="50.00")])
    before = _ledger_bankroll.read_parquet_safe(_ledger_bankroll.BANKROLL_PATH).copy()

    _ledger_bankroll.update_bankroll(NEXT_DATE, [])

    after = _ledger_bankroll.read_parquet_safe(_ledger_bankroll.BANKROLL_PATH)
    assert len(after) == len(before)
    assert NEXT_DATE.isoformat() not in set(after["date"])


def test_update_bankroll_gap_day_carries_forward_from_most_recent_row(
    monkeypatch, tmp_path
) -> None:
    _redirect(monkeypatch, tmp_path)
    _ledger_bankroll.update_bankroll(DATE, [_settled_row("id-1", date=DATE, pnl="100.00")])
    _ledger_bankroll.update_bankroll(NEXT_DATE, [])  # gap day: nothing settled
    _ledger_bankroll.update_bankroll(
        THIRD_DATE, [_settled_row("id-3", date=THIRD_DATE, pnl="25.00")]
    )

    df = _ledger_bankroll.read_parquet_safe(_ledger_bankroll.BANKROLL_PATH)
    assert set(df["date"]) == {DATE.isoformat(), THIRD_DATE.isoformat()}

    day1_row = df.loc[df["date"] == DATE.isoformat()].iloc[0]
    day3_row = df.loc[df["date"] == THIRD_DATE.isoformat()].iloc[0]
    assert Decimal(str(day1_row["ending_bankroll"])) == Decimal("5100.00")
    assert Decimal(str(day3_row["starting_bankroll"])) == Decimal("5100.00")
    assert Decimal(str(day3_row["ending_bankroll"])) == Decimal("5125.00")


def test_update_bankroll_separates_persona_replicate_pairs(monkeypatch, tmp_path) -> None:
    _redirect(monkeypatch, tmp_path)
    settled = [
        _settled_row("id-1", persona="safe", replicate_id=0, pnl="10.00"),
        _settled_row("id-2", persona="high_ev", replicate_id=0, pnl="-30.00"),
        _settled_row("id-3", persona="safe", replicate_id=1, pnl="5.00"),
    ]
    _ledger_bankroll.update_bankroll(DATE, settled)

    df = _ledger_bankroll.read_parquet_safe(_ledger_bankroll.BANKROLL_PATH)
    assert len(df) == 3
    safe_0 = df.loc[(df["persona"] == "safe") & (df["replicate_id"] == 0)].iloc[0]
    high_ev_0 = df.loc[(df["persona"] == "high_ev") & (df["replicate_id"] == 0)].iloc[0]
    safe_1 = df.loc[(df["persona"] == "safe") & (df["replicate_id"] == 1)].iloc[0]
    assert Decimal(str(safe_0["ending_bankroll"])) == Decimal("5010.00")
    assert Decimal(str(high_ev_0["ending_bankroll"])) == Decimal("4970.00")
    assert Decimal(str(safe_1["ending_bankroll"])) == Decimal("5005.00")
