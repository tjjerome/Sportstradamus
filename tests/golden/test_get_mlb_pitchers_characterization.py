"""Characterization pin for ``helpers.text.get_mlb_pitchers``.

``get_mlb_pitchers`` is ``@functools.cache``-wrapped and hits mlb-statsapi, so
the test monkeypatches ``mlb.schedule`` / ``mlb.get`` with a fixture slate and
clears the cache around the call. The single slate exercises every branch: the
status gate, the team-abbreviation resolution (and its unmatched-id skip), the
``game_num == 1`` gate, the per-side ``probable_pitcher`` presence + already-seen
dedupe, accent stripping, and the franchise-alias backfill.
"""

import pytest

from sportstradamus.helpers import text

_TEAMS = {
    "teams": [
        {"id": 1, "abbreviation": "NYY"},
        {"id": 2, "abbreviation": "BOS"},
        {"id": 3, "abbreviation": "LAD"},
        {"id": 4, "abbreviation": "AZ"},
    ]
}

_GAMES = [
    # gn1, both sides resolve + assign (home name carries accents to strip)
    {
        "status": "Scheduled",
        "away_id": 1,
        "home_id": 2,
        "game_num": 1,
        "away_probable_pitcher": "Gerrit Cole",
        "home_probable_pitcher": "Bréndan Béllo",
    },
    # status not in {Pre-Game, Scheduled} -> skipped entirely
    {
        "status": "Final",
        "away_id": 1,
        "home_id": 2,
        "game_num": 1,
        "away_probable_pitcher": "X",
        "home_probable_pitcher": "Y",
    },
    # away id matches no team -> resolution fails, game skipped
    {
        "status": "Pre-Game",
        "away_id": 999,
        "home_id": 2,
        "game_num": 1,
        "away_probable_pitcher": "A",
        "home_probable_pitcher": "B",
    },
    # game_num != 1 -> resolved but no assignment
    {
        "status": "Scheduled",
        "away_id": 3,
        "home_id": 4,
        "game_num": 2,
        "away_probable_pitcher": "NoAssign1",
        "home_probable_pitcher": "NoAssign2",
    },
    # gn1, away_probable_pitcher key absent -> only home (AZ) assigned
    {
        "status": "Scheduled",
        "away_id": 3,
        "home_id": 4,
        "game_num": 1,
        "home_probable_pitcher": "Zac Gallen",
    },
    # gn1, both sides already seen -> dedupe keeps first values
    {
        "status": "Scheduled",
        "away_id": 1,
        "home_id": 2,
        "game_num": 1,
        "away_probable_pitcher": "OtherGuy",
        "home_probable_pitcher": "OtherBello",
    },
]


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(text.mlb, "schedule", lambda **kw: _GAMES)
    monkeypatch.setattr(text.mlb, "get", lambda *a, **k: _TEAMS)
    text.get_mlb_pitchers.cache_clear()
    yield
    text.get_mlb_pitchers.cache_clear()


def test_get_mlb_pitchers_full_slate(patched):
    assert text.get_mlb_pitchers() == {
        "NYY": "Gerrit Cole",
        "BOS": "Brendan Bello",
        "AZ": "Zac Gallen",
        "LA": "",
        "ANA": "",
        "ARI": "Zac Gallen",
        "WAS": "",
    }
