"""Golden pins for constellation_shapes.json + its loader.

The catalog is the owner's live tuning surface for P8 Phase D: hand-edited
coordinates and classifier thresholds, reloaded on the next browser rerun with
no dashboard restart. So these pins split in two — raw-JSON structural checks
that every authored template obeys the S1–S9 authoring rules (the loader's
module docstring is their reference), and loader-contract checks through the
public accessors, including the hot-reload behavior the tuning loop depends on.

The bank floor pins at the bottom are what keep the assigner from running out of
deck *and* from putting a league on a shape that says nothing about its sport:
5 general plus 3 per league in every topology class, so every class can answer
any league with either its own equipment or an unaffiliated object. NBA and WNBA
share a tag list — same equipment, same shapes. A deck that falls under the
floors starts handing games ``None`` and the map reverts to the spring layout.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import pytest

from sportstradamus.dashboard.components import constellation_shapes as cs

_REPO = Path(__file__).resolve().parents[2]
_CATALOG_PATH = _REPO / "src" / "sportstradamus" / "data" / "config" / "constellation_shapes.json"

_LEAGUES = ("NBA", "WNBA", "NFL", "MLB", "NHL")
_CLASSES = ("hub", "chain", "twin", "mesh")

# S2's vertex band. Below 5 a template can't name an object; above 13 a real
# game runs out of legs to fill it and the shape reads as noise.
_MIN_VERTS, _MAX_VERTS = 5, 13

_SANE_TUNING = {
    "cluster_rho": 0.35,
    "hub_top_share": 0.30,
    "twin_cross_share": 0.25,
    "chain_mean_degree": 2.3,
    "chain_diameter_frac": 0.5,
    "mesh_density": 0.45,
    "min_shape_nodes": 3,
    "variety_lambda": 0.5,
    "league_affinity": 1.0,
}

_SANE_TEMPLATES = {
    "the-wedge": {
        "label": "The Wedge",
        "leagues": "all",
        "topology": {"primary": "chain", "secondary": []},
        "min_nodes": 2,
        "vertices": [
            {"id": 0, "x": -0.9, "y": -0.6, "side": "L", "prominence": 1},
            {"id": 1, "x": -0.3, "y": 0.5, "side": "L", "prominence": 2},
            {"id": 2, "x": 0.0, "y": 0.9, "side": "C", "prominence": 3},
            {"id": 3, "x": 0.3, "y": 0.5, "side": "R", "prominence": 4},
            {"id": 4, "x": 0.9, "y": -0.6, "side": "R", "prominence": 5},
        ],
        "outline": [[0, 1], [1, 2], [2, 3], [3, 4]],
        "silhouette": "M -0.9 -0.6 L 0 0.9 L 0.9 -0.6 Z",
    }
}


@pytest.fixture(scope="module")
def raw():
    return json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))


def _write_catalog(path: Path, tuning: dict | None = None, templates: dict | None = None) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "tuning": _SANE_TUNING if tuning is None else tuning,
                "templates": _SANE_TEMPLATES if templates is None else templates,
            }
        ),
        encoding="utf-8",
    )


def test_catalog_version_and_top_level_keys(raw):
    assert raw["version"] == 2
    assert set(raw) == {"version", "tuning", "templates"}
    assert raw["templates"], "catalog ships at least the authored exemplars"


def test_tuning_block_is_exactly_the_shipped_knobs_in_range(raw):
    """Every knob the classifier and assigner read, each inside the band a hand
    edit has to stay in for the cascade to keep meaning anything."""
    assert set(raw["tuning"]) == set(cs.TUNING_BOUNDS)
    for key, value in raw["tuning"].items():
        low, high = cs.TUNING_BOUNDS[key]
        assert cs.in_bounds(value, low, high), f"{key}={value} outside ({low}, {high})"


@pytest.mark.parametrize("slug", json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))["templates"])
def test_template_obeys_authoring_rules(raw, slug):
    tpl = raw["templates"][slug]
    assert slug == slug.lower() and " " not in slug
    assert tpl["label"].strip(), "S7: the cockpit's name for the shape is non-empty"

    leagues = tpl["leagues"]
    assert leagues == "all" or (leagues and set(leagues) <= set(_LEAGUES))

    topo = tpl["topology"]
    assert topo["primary"] in _CLASSES, "S7: primary is the class the vertex graph is"
    assert len(topo["secondary"]) <= 2
    assert set(topo["secondary"]) <= set(_CLASSES)
    assert topo["primary"] not in topo["secondary"]

    verts = tpl["vertices"]
    assert _MIN_VERTS <= len(verts) <= _MAX_VERTS, "S2"
    assert [v["id"] for v in verts] == list(range(len(verts))), "ids contiguous from 0"
    # Mirror pairs deliberately tie, so prominence is a ranking, not a permutation.
    ranks = {v["prominence"] for v in verts}
    assert ranks == set(range(1, len(ranks) + 1)), "S2: prominence ranks run 1..k with no gaps"
    for v in verts:
        assert -1.0 <= v["x"] <= 1.0 and -1.0 <= v["y"] <= 1.0, "S1"
        assert v["side"] in ("L", "R", "C")
    assert {"L", "R"} <= {v["side"] for v in verts}, "S3: a two-team split needs both halves"

    assert 2 <= tpl["min_nodes"] <= min(5, len(verts)), "S6"

    ids = {v["id"] for v in verts}
    for a, b in tpl["outline"]:
        assert a in ids and b in ids and a != b, "S4: outline refs real, distinct ids"

    assert tpl["silhouette"].startswith("M"), "S5: one SVG path"
    assert tpl["silhouette"].rstrip().endswith("Z"), "S5: the gesture is filled, so closed"


def test_templates_use_the_box(raw):
    """S1: a shape hugging one corner of [-1,1]² renders as a postage stamp."""
    for slug, tpl in raw["templates"].items():
        xs = [v["x"] for v in tpl["vertices"]]
        ys = [v["y"] for v in tpl["vertices"]]
        assert max(xs) - min(xs) >= 1.0, f"{slug} too narrow"
        assert max(ys) - min(ys) >= 1.0, f"{slug} too short"


def test_shape_catalog_and_tuning_read_the_shipped_file(raw):
    assert cs.shape_catalog() == raw
    assert cs.tuning() == raw["tuning"]


def test_eligible_templates_are_the_general_library_plus_the_leagues_own(raw):
    """S8 is a gate: another league's equipment is never on a game's table."""
    order = list(raw["templates"])
    for league in _LEAGUES:
        want = [
            slug
            for slug in order
            if raw["templates"][slug]["leagues"] == "all"
            or league in raw["templates"][slug]["leagues"]
        ]
        assert cs.eligible_templates(league) == want


def test_a_league_off_the_map_still_gets_the_general_library(raw):
    assert cs.eligible_templates("CFL") == [
        slug for slug, tpl in raw["templates"].items() if tpl["leagues"] == "all"
    ]


def test_catalog_hot_reloads_on_an_owner_edit(tmp_path, monkeypatch):
    """The whole point of the tuning cockpit: edit the JSON, hit rerun, see it.

    Cached on ``(path, mtime_ns)`` rather than a TTL so the edit lands on the
    very next rerun instead of up to an hour later.
    """
    path = tmp_path / "constellation_shapes.json"
    _write_catalog(path)
    monkeypatch.setattr(cs, "CATALOG_PATH", path)
    assert cs.tuning()["mesh_density"] == 0.45

    _write_catalog(path, {**_SANE_TUNING, "mesh_density": 0.62})
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert cs.tuning()["mesh_density"] == 0.62


@pytest.mark.parametrize(
    ("break_", "needle"),
    [
        ("tuning_out_of_range", "mesh_density"),
        ("tuning_missing_key", "cluster_rho"),
        ("bad_side", "side"),
        ("dangling_outline", "outline"),
        ("noncontiguous_ids", "contiguous"),
        ("unknown_topology", "topology"),
        ("coord_out_of_box", r"\[-1, 1\]"),
    ],
)
def test_a_malformed_catalog_fails_loud_naming_the_field(tmp_path, monkeypatch, break_, needle):
    """A hand-edit typo has to name itself, not surface as a blank star map."""
    tuning = dict(_SANE_TUNING)
    templates = json.loads(json.dumps(_SANE_TEMPLATES))
    verts = templates["the-wedge"]["vertices"]
    if break_ == "tuning_out_of_range":
        tuning["mesh_density"] = 1.4
    elif break_ == "tuning_missing_key":
        del tuning["cluster_rho"]
    elif break_ == "bad_side":
        verts[0]["side"] = "X"
    elif break_ == "dangling_outline":
        templates["the-wedge"]["outline"] = [[0, 9]]
    elif break_ == "noncontiguous_ids":
        verts[2]["id"] = 7
    elif break_ == "unknown_topology":
        templates["the-wedge"]["topology"]["primary"] = "spiral"
    elif break_ == "coord_out_of_box":
        verts[1]["x"] = 4.2

    path = tmp_path / "constellation_shapes.json"
    _write_catalog(path, tuning, templates)
    monkeypatch.setattr(cs, "CATALOG_PATH", path)
    with pytest.raises(ValueError, match=needle):
        cs.shape_catalog()


_ROTATION = ("hub", "chain", "twin", "mesh")


def _stub_deck(tmp_path, monkeypatch, per_class=8, min_nodes=2):
    """Point the loader at a synthetic bank wide enough to deal a whole slate.

    Dealing pins run against this rather than the shipped catalog so they stay
    honest about the algorithm and don't move every time D1b adds templates.
    Each class carries the previous class as its secondary, so a homogeneous
    slate always has a partial-fit alternative for the variety knob to reach for.
    """
    count = max(3, min_nodes)  # a template can't ask for more nodes than it has vertices
    vertices = [
        {
            "id": i,
            "x": round(-0.9 + 1.8 * i / (count - 1), 3),
            "y": 0.9 if i % 2 else -0.5,
            "side": "L" if i < count / 2 else "R",
            "prominence": i + 1,
        }
        for i in range(count)
    ]
    templates = {}
    for index, primary in enumerate(_ROTATION):
        for n in range(per_class):
            templates[f"{primary}-{n}"] = {
                "label": f"The {primary.title()} {n}",
                "leagues": "all",
                "topology": {"primary": primary, "secondary": [_ROTATION[index - 1]]},
                "min_nodes": min_nodes,
                "vertices": vertices,
                "outline": [[i, i + 1] for i in range(count - 1)],
                "silhouette": "M -0.9 -0.5 L 0.9 -0.5 L 0 0.9 Z",
            }
    path = tmp_path / "constellation_shapes.json"
    _write_catalog(path, _SANE_TUNING, templates)
    monkeypatch.setattr(cs, "CATALOG_PATH", path)


def _slate(count, topology="mesh", n_supernodes=6, league="NBA"):
    return [(f"G{i:02d}-{league}", league, topology, n_supernodes) for i in range(count)]


def test_a_full_slate_is_dealt_distinct_shapes(tmp_path, monkeypatch):
    """No two games showing at once may wear the same constellation, whatever
    leagues they belong to — the deal is one deck over the whole night."""
    _stub_deck(tmp_path, monkeypatch)
    dealt = cs.assign_templates("2026-08-06", _slate(8) + _slate(8, league="MLB"), _SANE_TUNING)
    assert len(dealt) == 16
    slugs = list(dealt.values())
    assert None not in slugs
    assert len(set(slugs)) == 16


def test_dealing_is_deterministic_and_reshuffles_on_a_new_date(tmp_path, monkeypatch):
    _stub_deck(tmp_path, monkeypatch)
    games = _slate(10)
    first = cs.assign_templates("2026-08-06", games, _SANE_TUNING)
    assert first == cs.assign_templates("2026-08-06", list(reversed(games)), _SANE_TUNING)
    assert first != cs.assign_templates("2026-08-07", games, _SANE_TUNING)


def test_a_game_is_dealt_a_template_matching_its_topology(tmp_path, monkeypatch):
    _stub_deck(tmp_path, monkeypatch)
    for topology in _ROTATION:
        dealt = cs.assign_templates("2026-08-06", _slate(1, topology), _SANE_TUNING)
        slug = next(iter(dealt.values()))
        assert cs.shape_catalog()["templates"][slug]["topology"]["primary"] == topology


def test_variety_lambda_spreads_a_slate_that_is_all_one_class(tmp_path, monkeypatch):
    """The owner's "everything is coming up mesh" escape hatch, proven live."""
    _stub_deck(tmp_path, monkeypatch)
    games = _slate(8)
    templates = cs.shape_catalog()["templates"]

    def classes_dealt(variety_lambda):
        dealt = cs.assign_templates(
            "2026-08-06", games, {**_SANE_TUNING, "variety_lambda": variety_lambda}
        )
        return {templates[slug]["topology"]["primary"] for slug in dealt.values()}

    assert classes_dealt(0) == {"mesh"}
    assert len(classes_dealt(2)) > 1


def _mixed_league_deck(tmp_path, monkeypatch):
    """Three interchangeable mesh templates — one MLB's, one universal, one NBA's."""
    vertices = [
        {
            "id": i,
            "x": round(-0.9 + 0.6 * i, 3),
            "y": 0.9 if i % 2 else -0.5,
            "side": side,
            "prominence": i + 1,
        }
        for i, side in enumerate("LLRR")
    ]
    templates = {
        slug: {
            "label": f"The {slug.title()}",
            "leagues": leagues,
            "topology": {"primary": "mesh", "secondary": []},
            "min_nodes": 2,
            "vertices": vertices,
            "outline": [[i, i + 1] for i in range(3)],
            "silhouette": "M -0.9 -0.5 L 0.9 -0.5 L 0 0.9 Z",
        }
        for slug, leagues in (("mitt", ["MLB"]), ("net", "all"), ("backboard", ["NBA"]))
    }
    _write_catalog(tmp_path / "constellation_shapes.json", _SANE_TUNING, templates)
    monkeypatch.setattr(cs, "CATALOG_PATH", tmp_path / "constellation_shapes.json")


def test_a_league_is_pulled_toward_its_own_equipment_first(tmp_path, monkeypatch):
    """Own gear outranks the general library when both fit the class equally —
    the whole of what ``league_affinity`` buys once eligibility has had its say."""
    _mixed_league_deck(tmp_path, monkeypatch)
    dealt = cs.assign_templates("2026-08-06", _slate(2, league="MLB"), _SANE_TUNING)
    assert [dealt[game] for game, *_ in _slate(2, league="MLB")] == ["mitt", "net"]


def test_another_leagues_equipment_is_never_dealt_even_with_nothing_left(tmp_path, monkeypatch):
    """S8 is a gate, not a preference: an NBA night that runs out of shapes goes
    shapeless rather than wearing the mitt."""
    _mixed_league_deck(tmp_path, monkeypatch)
    dealt = cs.assign_templates("2026-08-06", _slate(3), _SANE_TUNING)
    assert [dealt[game] for game, *_ in _slate(3)] == ["backboard", "net", None]


def test_league_affinity_at_zero_stops_a_league_reaching_for_its_own(tmp_path, monkeypatch):
    """The knob's off position: eligibility still holds, but own gear stops
    outranking the general library, so the tie falls to shuffle position."""
    _mixed_league_deck(tmp_path, monkeypatch)
    dates = [f"2026-08-{day:02d}" for day in range(1, 11)]

    def dealt(cfg):
        return [cs.assign_templates(date, _slate(1, league="MLB"), cfg)["G00-MLB"] for date in dates]

    assert dealt(_SANE_TUNING) == ["mitt"] * len(dates)
    assert "net" in dealt({**_SANE_TUNING, "league_affinity": 0.0})


def test_a_game_with_too_few_supernodes_keeps_the_spring_layout(tmp_path, monkeypatch):
    _stub_deck(tmp_path, monkeypatch)
    games = [
        ("thin", "NBA", "mesh", _SANE_TUNING["min_shape_nodes"] - 1),
        ("full", "NBA", "mesh", 6),
    ]
    dealt = cs.assign_templates("2026-08-06", games, _SANE_TUNING)
    assert dealt["thin"] is None
    assert dealt["full"] is not None


def test_a_game_below_every_templates_min_nodes_keeps_the_spring_layout(tmp_path, monkeypatch):
    """Templates are never stretched to fit — a shape that can't be filled isn't dealt."""
    _stub_deck(tmp_path, monkeypatch, min_nodes=9)
    dealt = cs.assign_templates("2026-08-06", _slate(1, n_supernodes=5), _SANE_TUNING)
    assert dealt["G00-NBA"] is None


def _inventory(raw) -> dict[tuple[str, str], int]:
    """``(class, "general"|league) -> count`` over primary classes only.

    Primaries, not secondaries: the floor is about a league having a *best*
    answer in every class, and a secondary is by definition a compromise.
    """
    counts: Counter = Counter()
    for tpl in raw["templates"].values():
        cls = tpl["topology"]["primary"]
        if tpl["leagues"] == "all":
            counts[(cls, "general")] += 1
        else:
            for league in tpl["leagues"]:
                counts[(cls, league)] += 1
    return counts


def test_every_class_carries_five_general_shapes(raw):
    """The unaffiliated library every league draws from, deep enough that a class
    never depends on league gear to answer a night."""
    counts = _inventory(raw)
    for cls in _CLASSES:
        assert counts[(cls, "general")] >= 5, f"{cls} has only {counts[(cls, 'general')]} general"


def test_every_league_carries_three_of_its_own_in_every_class(raw):
    """Another league's equipment is barred, so a league with a thin class gets
    pushed onto general shapes for games its own gear should be naming."""
    counts = _inventory(raw)
    for league in _LEAGUES:
        for cls in _CLASSES:
            assert counts[(cls, league)] >= 3, f"{league}/{cls} has only {counts[(cls, league)]}"


def test_nba_and_wnba_share_every_shape_they_are_tagged_with(raw):
    """Same equipment, same shapes — a tag list naming one and not the other is a typo."""
    for slug, tpl in raw["templates"].items():
        if tpl["leagues"] != "all":
            assert ("NBA" in tpl["leagues"]) == ("WNBA" in tpl["leagues"]), slug


def test_the_deck_is_deep_enough_for_a_stacked_multi_league_night(raw):
    """One deck covers every league showing at once, so the floor is the worst
    combined night — MLB's 17-game doubleheader slate beside an NFL Sunday — of
    which only the games with enough supernodes to carry a shape are dealt."""
    assert len(raw["templates"]) >= 100
    for league in _LEAGUES:
        assert len(cs.eligible_templates(league)) >= 60, league


def test_every_class_has_real_depth_behind_it(raw):
    """Counting secondaries, because a secondary match still beats an unrelated
    template — but a class with nothing behind it means those games get shapes that
    say nothing about how they actually correlate."""
    answers: Counter = Counter()
    primaries = set()
    for tpl in raw["templates"].values():
        topo = tpl["topology"]
        primaries.add(topo["primary"])
        for cls in (topo["primary"], *topo["secondary"]):
            answers[cls] += 1
    assert primaries == set(_CLASSES), f"no template is primarily {set(_CLASSES) - primaries}"
    for cls in _CLASSES:
        assert answers[cls] >= 25, f"{cls} has only {answers[cls]}"
