"""Coverage, slot-safety, and sport-voice pins for the JSON-backed thesis phrase bank.

Every league voice must resolve through ``bank_cell``'s fallback chain for the
full archetype × shape × direction × category vocabulary; every variant in
``voice_bank.json`` may only reference the slots its archetype fills (the
engine ``str.format``s exactly those, so a stray token would crash a render);
the shared voice must carry the eight guaranteed
``(archetype, "even", direction, "production")`` endpoints; and v1's 107
basketball player variants survive verbatim.
"""

from __future__ import annotations

import re

import pytest

from sportstradamus.prediction.stories.bank import _bank, bank_cell

_LEAGUE_VOICE = {
    "NBA": "basketball",
    "WNBA": "basketball",
    "NFL": "football",
    "NHL": "hockey",
    "MLB": "baseball",
}
_VOICES = {*_LEAGUE_VOICE.values(), "shared"}
_ARCHETYPES = ("player", "stack", "unit", "game-script")
_SHAPES = ("shootout", "grind", "blowout", "coinflip", "even")
_DIRECTIONS = ("Over", "Under")
_CATEGORIES = ("scoring", "boards", "playmaking", "stops", "k's", "production")

_ALLOWED_SLOTS = {
    "player": {"p", "g"},
    "game-script": {"g"},
    "unit": {"team", "grp", "opp"},
    "stack": {"n", "g", "p"},
}

_SLOT_RE = re.compile(r"\{([^{}]*)\}")


def _walk_variants():
    for voice, archetypes in _bank().items():
        for archetype, shapes in archetypes.items():
            for shape, directions in shapes.items():
                for direction, categories in directions.items():
                    for category, variants in categories.items():
                        yield voice, archetype, shape, direction, category, variants


@pytest.mark.parametrize("category", _CATEGORIES)
@pytest.mark.parametrize("direction", _DIRECTIONS)
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("archetype", _ARCHETYPES)
@pytest.mark.parametrize("league", sorted(_LEAGUE_VOICE))
def test_every_cell_resolves(league, archetype, shape, direction, category):
    cell = bank_cell(_LEAGUE_VOICE[league], archetype, shape, direction, category)
    assert isinstance(cell, list)
    assert cell
    assert all(isinstance(v, str) and v for v in cell)


def test_bank_keys_use_known_vocabulary():
    assert set(_bank()) == _VOICES
    for voice, archetype, shape, direction, category, variants in _walk_variants():
        key = (voice, archetype, shape, direction, category)
        assert archetype in _ARCHETYPES, key
        assert shape in _SHAPES, key
        assert direction in _DIRECTIONS, key
        assert category in _CATEGORIES, key
        assert isinstance(variants, list) and variants, key


def test_variants_reference_only_their_archetype_slots():
    for voice, archetype, shape, direction, category, variants in _walk_variants():
        allowed = _ALLOWED_SLOTS[archetype]
        for variant in variants:
            tokens = set(_SLOT_RE.findall(variant))
            assert tokens <= allowed, (voice, archetype, shape, direction, category, variant)


@pytest.mark.parametrize("direction", _DIRECTIONS)
@pytest.mark.parametrize("archetype", _ARCHETYPES)
def test_shared_guaranteed_endpoints(archetype, direction):
    cell = _bank()["shared"][archetype]["even"][direction]["production"]
    assert len(cell) >= 3


def test_v1_player_bank_preserved_in_basketball():
    cells = [
        variants
        for voice, archetype, _, _, _, variants in _walk_variants()
        if voice == "basketball" and archetype == "player"
    ]
    assert len(cells) == 44
    assert sum(len(variants) for variants in cells) == 107
    assert "In {g}'s track meet, {p} pours in the points" in bank_cell(
        "basketball", "player", "shootout", "Over", "scoring"
    )


def test_unknown_voice_reads_shared():
    expected = _bank()["shared"]["player"]["even"]["Over"]["production"]
    assert bank_cell("cricket", "player", "even", "Over", "production") == expected


def test_fallback_chain_steps_down_to_even_then_endpoint():
    # Football authors no player/blowout boards cell and shared holds boards
    # only under "even" — the chain must land on shared player/even boards.
    expected = _bank()["shared"]["player"]["even"]["Over"]["boards"]
    assert bank_cell("football", "player", "blowout", "Over", "boards") == expected
    # No unit cell anywhere speaks k's — must land on the guaranteed endpoint.
    endpoint = _bank()["shared"]["unit"]["even"]["Under"]["production"]
    assert bank_cell("hockey", "unit", "grind", "Under", "k's") == endpoint


def _cell_text(voice, archetype, shape, direction, category):
    return " ".join(bank_cell(voice, archetype, shape, direction, category)).lower()


def test_football_voice_speaks_football():
    scoring = _cell_text("football", "player", "even", "Over", "scoring")
    assert any(w in scoring for w in ("yard", "chain", "end zone", "drive", "snap"))
    script = _cell_text("football", "game-script", "grind", "Under", "production")
    assert any(w in script for w in ("punt", "three-and-out", "slugfest", "trench"))


def test_hockey_voice_speaks_hockey():
    scoring = _cell_text("hockey", "player", "even", "Over", "scoring")
    assert any(w in scoring for w in ("net", "lamp", "puck", "twine", "ice"))
    saves = _cell_text("hockey", "player", "even", "Over", "k's")
    assert any(w in saves for w in ("crease", "door", "puck", "save"))


def test_baseball_voice_speaks_baseball():
    ks = _cell_text("baseball", "player", "even", "Over", "k's")
    assert any(w in ks for w in ("strikeout", "punchout", "whiff", "swing", "carve"))
    scoring = _cell_text("baseball", "player", "even", "Over", "scoring")
    assert any(w in scoring for w in ("bases", "line drive", "bat", "plate", "square"))
