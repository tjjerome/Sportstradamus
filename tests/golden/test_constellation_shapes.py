"""Golden pins for constellation_shapes.json + its loader.

The catalog is the owner's live tuning surface for P8 Phase D: hand-edited
coordinates and classifier thresholds, reloaded on the next browser rerun with
no dashboard restart. So these pins split in two — raw-JSON structural checks
that every authored template obeys the S1–S8 authoring rules (the loader's
module docstring is their reference), and loader-contract checks through the
public accessors, including the hot-reload behavior the tuning loop depends on.

Bank floor pins (≥ 20 eligible per league, ≥ 6 per (league, class), all four
non-generic classes covered) land with the final D1b batch — asserting them
against the six exemplars would hold the suite red for the whole middle of the
phase.
"""

from __future__ import annotations

import json
import os
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
    """The eight knobs the classifier and assigner read, each inside the band a
    hand edit has to stay in for the cascade to keep meaning anything."""
    assert set(raw["tuning"]) == set(cs.TUNING_BOUNDS)
    for key, value in raw["tuning"].items():
        low, high = cs.TUNING_BOUNDS[key]
        assert cs.in_bounds(value, low, high), f"{key}={value} outside ({low}, {high})"


@pytest.mark.parametrize("slug", json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))["templates"])
def test_template_obeys_authoring_rules(raw, slug):
    tpl = raw["templates"][slug]
    assert slug == slug.lower() and " " not in slug
    assert tpl["label"].strip(), "S7: nameplate text is non-empty"

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


def test_eligible_templates_are_league_scoped_in_catalog_order(raw):
    order = list(raw["templates"])
    for league in _LEAGUES:
        want = [
            slug
            for slug in order
            if raw["templates"][slug]["leagues"] == "all"
            or league in raw["templates"][slug]["leagues"]
        ]
        assert cs.eligible_templates(league) == want


def test_eligible_templates_for_an_unlisted_league_is_the_universal_set(raw):
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
