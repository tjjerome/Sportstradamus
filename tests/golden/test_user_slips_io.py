"""Round-trip pins for the user-slips parquet store (``helpers/io.py``).

The dashboard "Lock it in" path upserts by ``slip_id`` (a re-lock replaces in
place); nightly grading rewrites the whole table. Both builder types share the
one file. ``USER_SLIPS_PATH`` is monkeypatched to a tmp file so these never
touch the real runtime artifact.
"""

from __future__ import annotations

import json

from sportstradamus.helpers import io


def _row(slip_id, builder_type="constellation", stake=5.0):
    return {
        "slip_id": slip_id,
        "saved_at": "2026-06-12T10:00:00",
        "builder_type": builder_type,
        "platform": "Underdog",
        "legs": json.dumps(
            [
                {
                    "desc": "A. Wilson Over 23.5 PTS - 80.0%, 1.00x",
                    "league": "WNBA",
                    "game": "LV",
                    "date": "2026-06-12",
                }
            ]
        ),
        "stake": stake,
        "status": "pending",
    }


def _patch_path(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "USER_SLIPS_PATH", tmp_path / "user_slips.parquet")


def test_empty_read_returns_empty(tmp_path, monkeypatch):
    _patch_path(tmp_path, monkeypatch)
    assert io.read_user_slips().empty


def test_upsert_appends_then_replaces_in_place(tmp_path, monkeypatch):
    _patch_path(tmp_path, monkeypatch)
    io.upsert_user_slip(_row("s1", stake=5.0))
    io.upsert_user_slip(_row("s2", stake=7.0))
    assert len(io.read_user_slips()) == 2

    io.upsert_user_slip(_row("s1", stake=9.0))  # same slip_id → replace, not append
    df = io.read_user_slips()
    assert len(df) == 2
    assert float(df.loc[df["slip_id"] == "s1", "stake"].iloc[0]) == 9.0


def test_write_user_slips_overwrites_whole_table(tmp_path, monkeypatch):
    _patch_path(tmp_path, monkeypatch)
    io.upsert_user_slip(_row("s1"))
    io.upsert_user_slip(_row("s2"))
    io.write_user_slips(io.read_user_slips().assign(status="won"))
    assert (io.read_user_slips()["status"] == "won").all()


def test_both_builder_types_round_trip(tmp_path, monkeypatch):
    _patch_path(tmp_path, monkeypatch)
    io.upsert_user_slip(_row("c1", builder_type="constellation"))
    io.upsert_user_slip(_row("s1", builder_type="simple"))
    df = io.read_user_slips()
    assert set(df["builder_type"]) == {"constellation", "simple"}
    legs = json.loads(df.loc[df["slip_id"] == "c1", "legs"].iloc[0])
    assert legs[0]["league"] == "WNBA"


def test_delete_removes_one_slip_and_no_ops_when_absent(tmp_path, monkeypatch):
    _patch_path(tmp_path, monkeypatch)
    io.upsert_user_slip(_row("s1"))
    io.upsert_user_slip(_row("s2"))
    io.delete_user_slip("s1")
    remaining = io.read_user_slips()
    assert list(remaining["slip_id"]) == ["s2"]
    io.delete_user_slip("ghost")  # absent id is a no-op, not an error
    assert list(io.read_user_slips()["slip_id"]) == ["s2"]
