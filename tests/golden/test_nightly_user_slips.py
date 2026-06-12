"""Grading pins for dashboard-built slips (``nightly._resolve_user_slips``).

Mirrors the ``_resolve_parlays`` characterization but over per-leg
``analysis._resolve_leg`` (a slip may span games): Power is all-or-nothing,
Flex/Sleeper cash a partial-hit floor, and a slip whose game has not been played
stays ``pending``. ``USER_SLIPS_PATH`` is monkeypatched to a tmp file.
"""

from __future__ import annotations

import json

import pandas as pd

from sportstradamus import nightly
from sportstradamus.helpers import io

_GAMELOG = pd.DataFrame(
    [
        {"DATE": "2026-06-10", "TEAM": "LV", "PLAYER": "A. Wilson", "PTS": 30, "AST": 3, "FG3M": 1},
        {"DATE": "2026-06-10", "TEAM": "LV", "PLAYER": "J. Young", "PTS": 12, "AST": 2, "FG3M": 4},
        {"DATE": "2026-06-10", "TEAM": "IND", "PLAYER": "C. Clark", "PTS": 20, "AST": 6, "FG3M": 3},
    ]
)


class _StubStats:
    """The minimal ``Stats`` surface grading reads: a gamelog + its column map."""

    def __init__(self, gamelog):
        self.gamelog = gamelog
        self.log_strings = {"date": "DATE", "team": "TEAM", "player": "PLAYER"}


def _slip(slip_id, play_type, legs, date="2026-06-10"):
    return {
        "slip_id": slip_id,
        "platform": "Underdog",
        "play_type": play_type,
        "payout_multiplier": 3.0,
        "status": "pending",
        "legs": json.dumps(
            [
                {"desc": f"{p} {b} {ln:.10g} {m} - 75.0%, 1.00x", "league": "WNBA", "game": g, "date": date}
                for p, b, ln, m, g in legs
            ]
        ),
    }


def test_resolve_grades_power_flex_and_skips_unplayed(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "USER_SLIPS_PATH", tmp_path / "user_slips.parquet")
    stats = {"WNBA": _StubStats(_GAMELOG)}

    slips = pd.DataFrame(
        [
            _slip("won", "Power", [("A. Wilson", "Over", 23.5, "PTS", "LV"),
                                   ("C. Clark", "Over", 4.5, "AST", "IND")]),
            _slip("lost", "Power", [("A. Wilson", "Over", 23.5, "PTS", "LV"),
                                    ("C. Clark", "Over", 7.5, "AST", "IND")]),
            _slip("partial", "Flex", [("A. Wilson", "Over", 23.5, "PTS", "LV"),
                                      ("A. Wilson", "Over", 2.5, "AST", "LV"),
                                      ("J. Young", "Over", 3.5, "FG3M", "LV"),
                                      ("C. Clark", "Over", 7.5, "AST", "IND")]),
            _slip("unplayed", "Power", [("A. Wilson", "Over", 23.5, "PTS", "MIN")], date="2026-06-11"),
        ]
    )
    io.write_user_slips(slips)

    graded = nightly._resolve_user_slips(stats, history_only=False)
    assert graded == 3  # the unplayed slip is skipped, not graded

    out = io.read_user_slips().set_index("slip_id")
    assert out.loc["won", "status"] == "won" and out.loc["won", "legs_hit"] == 2
    assert out.loc["lost", "status"] == "lost"
    assert out.loc["partial", "status"] == "partial" and out.loc["partial", "legs_hit"] == 3
    assert out.loc["unplayed", "status"] == "pending"


def test_history_only_skips_slip_grading(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "USER_SLIPS_PATH", tmp_path / "user_slips.parquet")
    io.write_user_slips(
        pd.DataFrame([_slip("x", "Power", [("A. Wilson", "Over", 23.5, "PTS", "LV")])])
    )
    assert nightly._resolve_user_slips({"WNBA": _StubStats(_GAMELOG)}, history_only=True) == 0
