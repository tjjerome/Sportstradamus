"""Coverage, slot-safety, depth, and sport-voice pins for the JSON-backed phrase banks.

The engine's ``_DIRECTIONS_BY_ARCHETYPE`` is the single source of truth for
which (archetype, direction) pairs are reachable; every league voice must
resolve through ``bank_cell``'s fallback chain for that full vocabulary and
the shared voice must terminate every reachable pair itself. Direction keys
are narrative (thrive/fade/split), so literal bet-side words are banned
outside ``mistakes`` cells — where the literal word must be the *flipped*
side, because a negative market's thriving bet is the Under. Every cell in
both banks carries at least ``_MIN_VARIANTS_PER_CELL`` variants so the md5
rotation has room to keep a slate's headlines distinct.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from sportstradamus.prediction.stories import bank as bank_module
from sportstradamus.prediction.stories import engine, legs
from sportstradamus.prediction.stories.bank import _bank, bank_cell, why_bank
from sportstradamus.prediction.stories.engine import _DIRECTIONS_BY_ARCHETYPE
from sportstradamus.training.markets import ALL_MARKETS

_LEAGUE_VOICE = engine._VOICE_BY_LEAGUE
_SPORT_VOICES = tuple(sorted(set(_LEAGUE_VOICE.values())))
_VOICES = {*_SPORT_VOICES, "shared"}
_SHAPES = ("shootout", "grind", "blowout", "coinflip", "even")
_CATEGORIES = (*legs._STAT_CATEGORY, "production", "mistakes")

# Every reachable (archetype, direction) pair, straight from the engine.
_ARCH_DIRS = [
    (archetype, direction)
    for archetype, directions in _DIRECTIONS_BY_ARCHETYPE.items()
    for direction in directions
]

# Depth floor for every cell in both banks: enough variants that the md5
# rotation and the slate-uniqueness bump have real room to move.
_MIN_VARIANTS_PER_CELL = 6

_ALLOWED_SLOTS = {
    "player": {"p", "g"},
    "game-script": {"g"},
    "unit": {"team", "grp", "opp"},
    "stack": {"n", "g", "p"},
}

_SLOT_RE = re.compile(r"\{([^{}]*)\}")
# R6: literal bet-side words, word-boundary, any case ("all over" and "turns
# over" count; "overtime" does not).
_BET_WORD_RE = re.compile(r"\b(over|overs|under|unders)\b", re.I)
_OVER_WORD_RE = re.compile(r"\b(over|overs)\b", re.I)
_UNDER_WORD_RE = re.compile(r"\b(under|unders)\b", re.I)


def _walk_variants():
    for voice, archetypes in _bank().items():
        for archetype, shapes in archetypes.items():
            for shape, directions in shapes.items():
                for direction, categories in directions.items():
                    for category, variants in categories.items():
                        yield voice, archetype, shape, direction, category, variants


def _walk_why_branches():
    """Yield ``((section, clause, branch), variants)`` over why_bank's prose cells."""
    bank = why_bank()
    for section in ("why", "dek"):
        for clause, node in bank[section].items():
            if isinstance(node, list):
                yield (section, clause, None), node
            else:
                for branch, variants in node.items():
                    yield (section, clause, branch), variants


@pytest.mark.parametrize("category", _CATEGORIES)
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize(("archetype", "direction"), _ARCH_DIRS)
@pytest.mark.parametrize("league", sorted(_LEAGUE_VOICE))
def test_every_cell_resolves(league, archetype, direction, shape, category):
    cell = bank_cell(_LEAGUE_VOICE[league], archetype, shape, direction, category)
    assert isinstance(cell, list)
    assert cell
    assert all(isinstance(v, str) and v for v in cell)


@pytest.mark.parametrize(("archetype", "direction"), _ARCH_DIRS)
def test_shared_terminates_every_reachable_pair(archetype, direction):
    """The fallback chain's endpoint exists for every pair the engine can emit."""
    cell = _bank()["shared"][archetype]["even"][direction]["production"]
    assert len(cell) >= _MIN_VARIANTS_PER_CELL


def test_bank_keys_use_known_vocabulary():
    assert set(_bank()) == _VOICES
    for voice, archetype, shape, direction, category, variants in _walk_variants():
        key = (voice, archetype, shape, direction, category)
        assert archetype in _DIRECTIONS_BY_ARCHETYPE, key
        assert shape in _SHAPES, key
        assert direction in _DIRECTIONS_BY_ARCHETYPE[archetype], key
        assert category in _CATEGORIES, key
        assert isinstance(variants, list) and variants, key


def test_every_voice_cell_carries_min_variants():
    for voice, archetype, shape, direction, category, variants in _walk_variants():
        key = (voice, archetype, shape, direction, category)
        assert len(variants) >= _MIN_VARIANTS_PER_CELL, key
        assert len(set(variants)) == len(variants), key


def test_variants_reference_only_their_archetype_slots():
    for voice, archetype, shape, direction, category, variants in _walk_variants():
        allowed = _ALLOWED_SLOTS[archetype]
        for variant in variants:
            tokens = set(_SLOT_RE.findall(variant))
            assert tokens <= allowed, (voice, archetype, shape, direction, category, variant)


class _AnySlot(dict):
    def __missing__(self, key):
        return "x"


def test_every_variant_renders():
    """Every template in both banks survives str.format — an unbalanced brace
    or positional slot would raise at story time, not authoring time."""
    for walker in (_walk_variants, _walk_why_branches):
        for *key, variants in walker():
            for variant in variants:
                variant.format_map(_AnySlot())


@pytest.mark.parametrize("voice", _SPORT_VOICES)
@pytest.mark.parametrize(
    ("archetype", "direction"),
    [pair for pair in _ARCH_DIRS if pair[0] in ("player", "stack", "game-script")],
)
def test_sport_voice_parity_on_even_production(voice, archetype, direction):
    """Every sport voice authors its own prose for each reachable headline pair."""
    cell = _bank()[voice][archetype]["even"][direction]["production"]
    assert len(cell) >= _MIN_VARIANTS_PER_CELL


@pytest.mark.parametrize("direction", ("Over", "Under"))
@pytest.mark.parametrize("voice", _SPORT_VOICES)
def test_sport_voice_authors_mistakes(voice, direction):
    cell = _bank()[voice]["player"]["even"][direction]["mistakes"]
    assert len(cell) >= _MIN_VARIANTS_PER_CELL


def test_no_variant_treats_g_as_opponent():
    """{g} is a game slug ("NYK/SAS"), never a team a player performs against."""
    for key_head in (_walk_variants, _walk_why_branches):
        for *key, variants in key_head():
            for variant in variants:
                assert "against {g}" not in variant, (*key, variant)


def test_literal_bet_words_only_in_mistakes_cells():
    """Direction keys are narrative, so prose carries mood, not bet sides —
    except mistakes cells, where the literal word must be the flipped side
    (thriving on a negative market *is* the Under). why_bank is exempt: its
    clauses are bet-factual by design."""
    for voice, archetype, shape, direction, category, variants in _walk_variants():
        key = (voice, archetype, shape, direction, category)
        for variant in variants:
            if category != "mistakes":
                assert not _BET_WORD_RE.search(variant), (key, variant)
            else:
                wrong_side = _OVER_WORD_RE if direction == "Over" else _UNDER_WORD_RE
                assert not wrong_side.search(variant), (key, variant)


def test_v1_player_bank_preserved_in_basketball():
    """The v1 basketball player bank survives in spirit: at least its 44
    Over/Under cells and 107 variants, and the flagship line verbatim. The
    letter-for-letter pin was retired because 12 v1 variants were broken
    "against {g}" strings (treating the game slug as an opponent) that had to
    be rewritten 1:1, and every cell has since been deepened."""
    cells = [
        variants
        for voice, archetype, _, direction, _, variants in _walk_variants()
        if voice == "basketball" and archetype == "player" and direction in ("Over", "Under")
    ]
    assert len(cells) >= 44
    assert sum(len(variants) for variants in cells) >= 107
    assert "In {g}'s track meet, {p} pours in the points" in bank_cell(
        "basketball", "player", "shootout", "Over", "scoring"
    )


def test_unknown_voice_reads_shared():
    expected = _bank()["shared"]["player"]["even"]["Over"]["production"]
    assert bank_cell("cricket", "player", "even", "Over", "production") == expected


def test_fallback_chain_steps_down_to_even_then_endpoint():
    # Football authors no player/blowout boards cell but does author the shape,
    # so the category degrades to production before the shape gives way.
    assert (
        bank_cell("football", "player", "blowout", "Over", "boards")
        is _bank()["football"]["player"]["blowout"]["Over"]["production"]
    )
    # No voice authors a shaped unit cell, and shared's shaped scoring copy
    # outranks its own shaped production copy.
    assert (
        bank_cell("basketball", "unit", "shootout", "Over", "scoring")
        is _bank()["shared"]["unit"]["shootout"]["Over"]["scoring"]
    )
    # Same node, a category shared doesn't speak: shared's shaped production
    # copy still outranks the voice's own even-keyed boards cell.
    assert (
        bank_cell("basketball", "unit", "shootout", "Over", "boards")
        is _bank()["shared"]["unit"]["shootout"]["Over"]["production"]
    )
    # An even-shaped lookup for a vocabulary no unit cell speaks: with the shape
    # already "even", the shared-production step and the endpoint are one key.
    assert (
        bank_cell("hockey", "unit", "even", "Under", "k's")
        is _bank()["shared"]["unit"]["even"]["Under"]["production"]
    )


def test_endpoint_answers_a_bank_that_authors_nothing_else(monkeypatch):
    """The terminal return is unreachable for every mapped league today, so it
    takes a bank stripped to the endpoint cell to exercise it."""
    endpoint = ["the only copy there is"]
    monkeypatch.setattr(
        bank_module,
        "_bank",
        lambda *_a, **_k: {"shared": {"player": {"even": {"Over": {"production": endpoint}}}}},
    )
    assert bank_cell("basketball", "player", "shootout", "Over", "boards") is endpoint


@pytest.mark.parametrize("shape", [s for s in _SHAPES if s != "even"])
@pytest.mark.parametrize(("archetype", "direction"), _ARCH_DIRS)
@pytest.mark.parametrize("voice", _SPORT_VOICES)
def test_shape_never_falls_through_to_even(voice, archetype, direction, shape):
    """Shape outranks both voice and category, for every reachable lookup.

    Every (voice, archetype, non-even shape, direction) node must be authored
    somewhere, and every category under it must resolve to that node's copy —
    a shape-blind fall-through is how a shootout ended up telling readers the
    game had a low ceiling.
    """
    shaped = [
        cell
        for bank_voice in (voice, "shared")
        for cell in _bank()[bank_voice]
        .get(archetype, {})
        .get(shape, {})
        .get(direction, {})
        .values()
    ]
    assert shaped, "no cell authored for this shape and direction"
    for category in _CATEGORIES:
        returned = bank_cell(voice, archetype, shape, direction, category)
        assert any(returned is cell for cell in shaped), category


# Where the board's shape and the model's direction disagree, every voice owes
# the tension its own copy rather than shape-blind even-keyed filler.
_CONTRARIAN_CELLS = (
    ("game-script", "shootout", "Under"),
    ("game-script", "grind", "Over"),
    ("stack", "shootout", "Under"),
    ("stack", "grind", "Over"),
)


@pytest.mark.parametrize(("archetype", "shape", "direction"), _CONTRARIAN_CELLS)
@pytest.mark.parametrize("voice", sorted(_VOICES))
def test_contrarian_cells_exist(voice, archetype, shape, direction):
    cell = _bank()[voice][archetype][shape][direction]["production"]
    assert len(cell) >= _MIN_VARIANTS_PER_CELL


def test_every_shaped_node_carries_production():
    """The chain probes only ``category`` then ``production`` under a shape node,
    so a node authored without a production cell would drop its shape for every
    other category."""
    for voice, archetypes in _bank().items():
        for archetype, shapes in archetypes.items():
            for shape, directions in shapes.items():
                for direction, categories in directions.items():
                    assert "production" in categories, (voice, archetype, shape, direction)


def test_every_pipeline_league_has_a_voice():
    """A league the pipeline trains but the engine doesn't map reads shared copy
    silently — ``ALL_MARKETS`` is the canonical league catalogue."""
    assert set(ALL_MARKETS) <= set(_LEAGUE_VOICE)


# Lead case is a bank-wide convention: Over/Under cells read as headlines and open
# on a capital or a slot, while Mixed and Contrast* cells read as fragments and
# never capitalise. These cells predate the convention and keep their lowercase
# leads; no new cell may join them.
_LOWERCASE_LEAD_CELLS = frozenset(
    {
        ("shared", "player", "even", "Over", "mistakes"),
        ("shared", "player", "even", "Under", "mistakes"),
        ("basketball", "player", "even", "Over", "mistakes"),
        ("basketball", "player", "even", "Under", "mistakes"),
        ("football", "player", "even", "Over", "mistakes"),
        ("football", "player", "even", "Under", "mistakes"),
        ("football", "stack", "even", "Under", "production"),
        ("hockey", "player", "even", "Over", "mistakes"),
        ("hockey", "player", "even", "Under", "mistakes"),
        ("hockey", "stack", "even", "Under", "production"),
        ("baseball", "player", "even", "Over", "mistakes"),
        ("baseball", "player", "even", "Under", "mistakes"),
        ("baseball", "stack", "even", "Under", "production"),
    }
)


def test_lead_case_matches_direction():
    """Mixed within one cell would render the same game capitalised one day and
    lowercase the next, since the md5 rotation moves through the variants."""
    for voice, archetype, shape, direction, category, variants in _walk_variants():
        key = (voice, archetype, shape, direction, category)
        for variant in variants:
            if variant[0] == "{":
                continue
            if direction in ("Mixed", "ContrastOver", "ContrastUnder"):
                assert variant[0].islower(), (key, variant)
            elif key not in _LOWERCASE_LEAD_CELLS:
                assert variant[0].isupper(), (key, variant)


def _cell_text(voice, archetype, shape, direction, category):
    return " ".join(bank_cell(voice, archetype, shape, direction, category)).lower()


def test_basketball_voice_speaks_basketball():
    mixed = _cell_text("basketball", "player", "even", "Mixed", "production")
    assert any(w in mixed for w in ("floor", "glass", "bucket", "rim", "possession"))
    contrast = _cell_text("basketball", "stack", "even", "ContrastOver", "production")
    assert any(w in contrast for w in ("gym", "floor", "bucket", "shot", "rim"))
    mistakes = _cell_text("basketball", "player", "even", "Under", "mistakes")
    assert any(w in mistakes for w in ("turnover", "handle", "giveaway", "pocket"))


def test_football_voice_speaks_football():
    scoring = _cell_text("football", "player", "even", "Over", "scoring")
    assert any(w in scoring for w in ("yard", "chain", "end zone", "drive", "snap"))
    script = _cell_text("football", "game-script", "grind", "Under", "production")
    assert any(w in script for w in ("punt", "three-and-out", "slugfest", "trench"))
    mixed = _cell_text("football", "player", "even", "Mixed", "production")
    assert any(w in mixed for w in ("snap", "drive", "yard", "down", "huddle"))
    contrast = _cell_text("football", "stack", "even", "ContrastUnder", "production")
    assert any(w in contrast for w in ("field", "drive", "chain", "boundary", "clock"))
    mistakes = _cell_text("football", "player", "even", "Under", "mistakes")
    assert any(w in mistakes for w in ("sack", "pick", "fumble", "pocket"))


def test_hockey_voice_speaks_hockey():
    scoring = _cell_text("hockey", "player", "even", "Over", "scoring")
    assert any(w in scoring for w in ("net", "lamp", "puck", "twine", "ice"))
    saves = _cell_text("hockey", "player", "even", "Over", "k's")
    assert any(w in saves for w in ("crease", "door", "puck", "save"))
    mixed = _cell_text("hockey", "player", "even", "Mixed", "production")
    assert any(w in mixed for w in ("shift", "ice", "puck", "period", "bench"))
    contrast = _cell_text("hockey", "stack", "even", "ContrastOver", "production")
    assert any(w in contrast for w in ("lamp", "ice", "skate", "line", "net"))
    mistakes = _cell_text("hockey", "player", "even", "Under", "mistakes")
    assert any(w in mistakes for w in ("crease", "net", "lamp", "goal", "rubber"))


def test_baseball_voice_speaks_baseball():
    ks = _cell_text("baseball", "player", "even", "Over", "k's")
    assert any(w in ks for w in ("strikeout", "punchout", "whiff", "swing", "carve"))
    scoring = _cell_text("baseball", "player", "even", "Over", "scoring")
    assert any(w in scoring for w in ("bases", "line drive", "bat", "plate", "square"))
    mixed = _cell_text("baseball", "player", "even", "Mixed", "production")
    assert any(w in mixed for w in ("plate", "inning", "bat", "swing", "zone"))
    contrast = _cell_text("baseball", "stack", "even", "ContrastOver", "production")
    assert any(w in contrast for w in ("lineup", "order", "bat", "barrel", "plate"))
    mistakes = _cell_text("baseball", "player", "even", "Under", "mistakes")
    assert any(w in mistakes for w in ("walk", "free pass", "zone", "wild"))


# Exact slot vocabulary per why_bank branch, keyed (section, clause, branch);
# a list-valued clause (dek.cluster) carries branch None. Every variant must
# use exactly these slots — the facts are the clause.
_WHY_SLOTS = {
    ("why", "form", "above"): {"dev", "line", "pronoun"},
    ("why", "form", "below"): {"dev", "line", "pronoun"},
    ("why", "form", "at_line"): {"line", "pronoun"},
    ("why", "form_h2h", "above"): {"dev"},
    ("why", "form_h2h", "below"): {"dev"},
    ("why", "form_h2h", "at_line"): set(),
    ("why", "matchup", "favorable"): set(),
    ("why", "matchup", "tough"): set(),
    ("why", "edge", "model_ahead"): {"model_pct", "book_pct", "side", "gap"},
    ("why", "edge", "book_back"): {"model_pct", "book_pct", "side", "gap"},
    ("why", "ev", "solo"): {"ev"},
    ("why", "ev", "vs_book"): {"ev", "book_ev"},
    ("dek", "cluster", None): {"n", "rho"},
    ("dek", "form", "above"): {"p", "dev", "line"},
    ("dek", "form", "below"): {"p", "dev", "line"},
    ("dek", "form", "at_line"): {"p", "line"},
    ("dek", "matchup", "favorable"): {"p"},
    ("dek", "matchup", "tough"): {"p"},
}


def test_why_bank_branch_inventory_and_slots():
    seen = dict(_walk_why_branches())
    assert set(seen) == set(_WHY_SLOTS)
    for key, variants in seen.items():
        for variant in variants:
            assert set(_SLOT_RE.findall(variant)) == _WHY_SLOTS[key], (key, variant)


def test_why_bank_depth_and_mid_sentence_style():
    """Every branch is deep enough to rotate, and every variant renders
    mid-sentence: no trailing period, no leading capital (slot-leading
    variants aside — {p} supplies its own proper noun)."""
    for key, variants in _walk_why_branches():
        assert len(variants) >= _MIN_VARIANTS_PER_CELL, key
        assert len(set(variants)) == len(variants), key
        for variant in variants:
            assert not variant.endswith("."), (key, variant)
            assert variant[0] == "{" or variant[0].islower(), (key, variant)


def test_why_bank_pronouns():
    pronouns = why_bank()["pronouns"]
    assert {"WNBA", "combo", "default"} <= set(pronouns)
    assert all(isinstance(v, str) and v for v in pronouns.values())


def test_no_prose_literals_in_stories_source():
    """Prose lives in the JSON banks, never in code (owner directive).

    Cheap guard: no string literal in stories/*.py may carry a bank slot
    token — a template hiding in code shows up as a literal "{p}"-style
    slot. Docstrings are exempt (they may cite slot names when documenting
    the bank contract); the eyeball pass remains the real gate.
    """
    slot_re = re.compile(
        r"\{(?:p|g|n|team|grp|opp|dev|line|pronoun|model_pct|book_pct|gap|side|ev|book_ev|rho)\}"
    )
    package_dir = Path(engine.__file__).parent
    for source_path in sorted(package_dir.glob("*.py")):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
            ):
                assert not slot_re.search(node.value), (source_path.name, node.value)
