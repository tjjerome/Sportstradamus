"""Characterization test pinning ``StatsMLB.parse_game`` output.

``parse_game`` has no other automated coverage and is being decomposed for
complexity, so this locks its gamelog/teamlog row-building against a real
(trimmed) baseballsavant payload. The input fixture and both expected snapshots
were captured from the pre-refactor code on game_pk 716896 (CHC @ DET).

``_refresh_handedness`` is covered here too: it exists to correct the ``bats`` /
``throws`` values ``parse_game``'s resolvers cache first-write-wins, so the two
behaviors are pinned together.

Park factors are set to 1.0 (so park-adjusted == raw) and the archive-backed
``_enrich_team_markets`` is stubbed, isolating parse_game's own arithmetic from
external state. The instance is built via ``__new__`` to skip ``__init__``'s
file I/O — parse_game only touches ``park_factors``, ``players``, ``gamelog``,
``teamlog``, and ``league``.
"""

import datetime
import json
import pathlib

import pandas as pd
import pytest

from sportstradamus.stats import mlb

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
_GAME_PK = 716896
_PARK_FACTOR_KEYS = ("R", "H", "1B", "2B", "3B", "HR", "BB", "K", "OBP")


@pytest.fixture
def parsed(monkeypatch):
    payload = json.loads((FIXTURES / "mlb_parse_game_716896.json").read_text())
    stats = mlb.StatsMLB.__new__(mlb.StatsMLB)
    stats.gamelog = pd.DataFrame()
    stats.teamlog = pd.DataFrame()
    stats.players = {}
    stats.league = "MLB"
    unit = dict.fromkeys(_PARK_FACTOR_KEYS, 1.0)
    stats.park_factors = {"CHC": dict(unit), "DET": dict(unit)}
    monkeypatch.setattr(mlb.scraper, "get", lambda url: payload)
    monkeypatch.setattr(stats, "_enrich_team_markets", lambda *a, **k: None)
    stats.parse_game(_GAME_PK)
    return stats


def _expected(name: str) -> pd.DataFrame:
    return pd.read_json(
        FIXTURES / f"mlb_parse_game_716896_{name}.json",
        orient="records",
        convert_dates=False,
        convert_axes=False,
    )


def test_parse_game_gamelog_matches_snapshot(parsed):
    pd.testing.assert_frame_equal(
        parsed.gamelog.reset_index(drop=True), _expected("gamelog"), check_dtype=False
    )


def test_parse_game_teamlog_matches_snapshot(parsed):
    pd.testing.assert_frame_equal(
        parsed.teamlog.reset_index(drop=True), _expected("teamlog"), check_dtype=False
    )


def test_refresh_handedness_corrects_but_never_adds(monkeypatch):
    """Switch hitters get their true "S" back without the registry gaining a key."""
    stats = mlb.StatsMLB.__new__(mlb.StatsMLB)
    stats.season_start = datetime.date(2026, 3, 25)
    stats.players = {
        608070: {"name": "Jose Ramirez", "bats": "R"},
        663776: {"name": "Patrick Sandoval", "throws": "L"},
        1: {"name": "Nobody", "bats": "L"},
    }
    calls = []

    def fake_get(endpoint, params):
        calls.append((endpoint, params))
        return {
            "people": [
                {"id": 608070, "batSide": {"code": "S"}, "pitchHand": {"code": "R"}},
                {"id": 663776, "pitchHand": {"code": "L"}},
                {"id": 592450, "batSide": {"code": "L"}, "pitchHand": {"code": "R"}},
            ]
        }

    monkeypatch.setattr(mlb.mlb, "get", fake_get)
    stats._refresh_handedness()

    assert len(calls) == 1
    endpoint, params = calls[0]
    assert endpoint == "sports_players"
    assert params["sportId"] == 1
    assert params["season"] == 2026
    assert stats.players == {
        # Ramirez gains no ``throws`` even though the feed carries his: an absent
        # key is the resolvers' cache-miss signal, so values are corrected, never
        # added.
        608070: {"name": "Jose Ramirez", "bats": "S"},
        663776: {"name": "Patrick Sandoval", "throws": "L"},
        1: {"name": "Nobody", "bats": "L"},
    }
