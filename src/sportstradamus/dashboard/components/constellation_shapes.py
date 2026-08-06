"""The constellation shape bank — ``constellation_shapes.json`` and its loader.

Each template is one recognizable sports object drawn as a star chart: a set of
vertices a game's legs get assigned to, the minimal line-work that names the
object, and one filled SVG gesture behind it. A game is classified by the shape
of its own correlation graph and dealt a template that matches, so a star map
gestures at its namesake the way a real constellation does — loosely.

The JSON is the tuning surface, not a build artifact: the owner edits
coordinates and thresholds by hand and sees them on the next browser rerun.
That is why the catalog is cached on ``(path, mtime_ns)`` instead of a TTL, and
why every accessor here stays pure stdlib — no Streamlit import, so the bank is
testable without AppTest (the same contract ``prediction/stories/bank.py`` keeps
for ``voice_bank.json``).

Authoring rules. JSON carries no comments, so this docstring is the schema
reference; ``tests/golden/test_constellation_shapes.py`` enforces the
mechanical half.

* **S1** Coordinates normalized to [-1, 1]², shape centered, use the box — no
  postage stamps.
* **S2** 5–13 vertices. ``prominence`` is the importance order (1 = the star the
  shape can't live without), and mirror pairs tie, so it ranks rather than
  permutes. Player-outline shapes simplify to ≤ 13 vertices; the silhouette
  carries the rest.
* **S3** Sides must let a typical 3+3 two-team split read: mirrored objects get
  mirrored L/R; asymmetric objects (bat, bolt) still tag both halves along their
  long axis; ``C`` for spine/center vertices only.
* **S4** ``outline`` is the minimal line-work that names the object. Every pair
  references real ids; interior vertices (a mound, a bullseye) may be
  outline-free.
* **S5** ``silhouette`` is one closed SVG path — the filled gesture, ≤ ~8
  commands. It carries what the line-work can't (the bolt's body, the leaf's
  lobes). Only ``M L Q C T S Z``, space-separated, every coordinate set
  re-issuing its command letter: Plotly's shape parser silently drops anything
  else (an ``A`` arc costs you the curve with no error), and ``H``/``V`` would
  break the x/y alternation the renderer rescales on. Draw curves as cubics —
  a circular quadrant of radius ``r`` wants control points at ``0.5523 r``.
* **S6** ``min_nodes`` = fewest real supernodes that still read as the object
  (2–5).
* **S7** ``label`` is the nameplate text ("The Bolt"). ``topology.primary`` is
  the class the vertex graph *is*; at most two secondaries, never repeating the
  primary.
* **S8** Generic archetypes only, never trademarked geometry.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.resources as pkg_resources
import json
import math
import random
from collections import Counter
from pathlib import Path

from sportstradamus import data

CATALOG_PATH = Path(str(pkg_resources.files(data) / "config" / "constellation_shapes.json"))

# The four shapes a game's correlation graph can be. "generic" is the classifier's
# no-match answer and is deliberately not a template tag — nothing is authored for it.
TOPOLOGY_CLASSES = frozenset({"hub", "chain", "twin", "mesh"})

_SIDES = frozenset({"L", "R", "C"})

# Plotly's shape paths accept only a subset of SVG, and an unsupported command is
# dropped in silence — an "A" arc costs you the curve with no error anywhere. H and V
# are supported there but banned here anyway: they take a single coordinate, which
# would break the strict x/y alternation the renderer's rescale walks on. Write the
# curve as a cubic and the straight line as an L.
_SVG_COMMANDS = frozenset("MLQCTSZ")

# Owner-editable knob -> the band a hand edit has to stay inside. Bounded knobs
# are strict at both ends: a 0 or 1 share makes the classifier answer the same
# way for every graph, which fails silently rather than loudly. The last two have
# a floor and no natural ceiling.
TUNING_BOUNDS = {
    "cluster_rho": (0.0, 1.0),
    "hub_top_share": (0.0, 1.0),
    "twin_cross_share": (0.0, 1.0),
    "chain_mean_degree": (1.0, 4.0),
    "chain_diameter_frac": (0.0, 1.0),
    "mesh_density": (0.0, 1.0),
    "min_shape_nodes": (2, math.inf),
    "variety_lambda": (0.0, math.inf),
}


def in_bounds(value: float, low: float, high: float) -> bool:
    """Whether a tuning value sits inside its knob's band (see ``TUNING_BOUNDS``)."""
    return low < value < high if math.isfinite(high) else value >= low


def _validate_tuning(block: dict) -> None:
    """Every knob present and inside its band, named in the message if not."""
    missing = sorted(set(TUNING_BOUNDS) - set(block))
    if missing:
        raise ValueError(f"constellation tuning is missing {missing}")
    for key, (low, high) in TUNING_BOUNDS.items():
        if not in_bounds(block[key], low, high):
            raise ValueError(f"constellation tuning {key}={block[key]} outside ({low}, {high})")


def _validate_vertices(slug: str, vertices: list[dict]) -> None:
    """S1 + S3: every vertex sits in the box and picks a side."""
    for vertex in vertices:
        if vertex["side"] not in _SIDES:
            raise ValueError(
                f"{slug}: vertex {vertex['id']} side {vertex['side']!r} not one of L/R/C"
            )
        if not (-1.0 <= vertex["x"] <= 1.0 and -1.0 <= vertex["y"] <= 1.0):
            raise ValueError(f"{slug}: vertex {vertex['id']} falls outside [-1, 1] on an axis")


def _validate_template(slug: str, tpl: dict) -> None:
    """Raise naming the offending field, so a typo can't render as an empty sky."""
    if not tpl["label"].strip():
        raise ValueError(f"{slug}: label is the nameplate text and cannot be blank")
    topo = tpl["topology"]
    if topo["primary"] not in TOPOLOGY_CLASSES or not set(topo["secondary"]) <= TOPOLOGY_CLASSES:
        raise ValueError(f"{slug}: unknown class in topology {topo}")

    ids = [vertex["id"] for vertex in tpl["vertices"]]
    if ids != list(range(len(ids))):
        raise ValueError(f"{slug}: vertex ids must be contiguous from 0, got {ids}")
    _validate_vertices(slug, tpl["vertices"])
    if tpl["min_nodes"] > len(ids):
        raise ValueError(f"{slug}: min_nodes {tpl['min_nodes']} exceeds {len(ids)} vertices")
    for pair in tpl["outline"]:
        if not set(pair) <= set(ids):
            raise ValueError(f"{slug}: outline pair {pair} references a missing vertex id")

    used = {token for token in tpl["silhouette"].split() if token.isalpha()}
    if not used <= _SVG_COMMANDS:
        raise ValueError(
            f"{slug}: silhouette uses {sorted(used - _SVG_COMMANDS)}, "
            f"which Plotly drops without a word; allowed: {sorted(_SVG_COMMANDS)}"
        )


@functools.lru_cache(maxsize=8)
def _load(path: Path, mtime_ns: int) -> dict:
    """Parse + validate the catalog at ``path``.

    ``mtime_ns`` is not read in the body — it is half the cache key, so an owner
    edit invalidates the entry. ``path`` is the other half because tests point the
    module constant at different tmp files and must not collide on a stale entry
    (the hazard ``dashboard/data.py:_mtime`` documents for the parquet loaders).
    """
    catalog = json.loads(path.read_text(encoding="utf-8"))
    _validate_tuning(catalog["tuning"])
    for slug, tpl in catalog["templates"].items():
        _validate_template(slug, tpl)
    return catalog


def shape_catalog() -> dict:
    """The validated catalog, re-read whenever the file changes on disk."""
    return _load(CATALOG_PATH, CATALOG_PATH.stat().st_mtime_ns)


def tuning() -> dict:
    """Every classifier and assigner threshold, owner-editable (see ``TUNING_BOUNDS``)."""
    return shape_catalog()["tuning"]


def eligible_templates(league: str) -> list[str]:
    """Template slugs a league may be dealt — universal plus its own — in catalog order."""
    return [
        slug
        for slug, tpl in shape_catalog()["templates"].items()
        if tpl["leagues"] == "all" or league in tpl["leagues"]
    ]


def _score(tpl: dict, topology: str, dealt: Counter, cfg: dict) -> float:
    """Class fit minus variety pressure.

    A ``generic`` game has no class to match, so every template fits it equally
    and the variety term alone decides. ``variety_lambda`` is the owner's
    "everything is coming up mesh" knob: 0 is pure topology fidelity, ~0.5 nudges
    a homogeneous slate toward spread, ≥ 2 forces it.
    """
    topo = tpl["topology"]
    if topology == "generic":
        fit = 1
    elif topology == topo["primary"]:
        fit = 2
    elif topology in topo["secondary"]:
        fit = 1
    else:
        fit = 0
    return fit - cfg["variety_lambda"] * dealt[topo["primary"]]


def assign_templates(
    league: str,
    date: str,
    games: list[tuple[str, str, int]],
    cfg: dict,
) -> dict[str, str | None]:
    """Deal each ``(game_key, topology_class, n_supernodes)`` a distinct template.

    Returns ``game_key -> slug``, or ``None`` for a game too thin to carry a shape
    (fewer supernodes than ``cfg['min_shape_nodes']``, or below the ``min_nodes`` of
    every template still on the table) — those keep the spring layout.

    Deterministic: an md5 of league, date and catalog version seeds the shuffle,
    and games are dealt in sorted-key order, so every viewer sees the same slate
    dealt the same way and a new date reshuffles the deck. Ties fall to shuffle
    position. ``md5`` rather than ``hash()`` on purpose — ``PYTHONHASHSEED`` would
    make the dealing differ between runs of the same day.
    """
    catalog = shape_catalog()
    templates = catalog["templates"]
    deck = eligible_templates(league)
    seed = f"{league}|{date}|{catalog['version']}".encode()
    random.Random(int(hashlib.md5(seed).hexdigest(), 16)).shuffle(deck)

    dealt: Counter = Counter()
    assignment: dict[str, str | None] = {}
    for game_key, topology, n_supernodes in sorted(games):
        free = [
            slug
            for slug in deck
            if slug not in assignment.values() and templates[slug]["min_nodes"] <= n_supernodes
        ]
        if n_supernodes < cfg["min_shape_nodes"] or not free:
            assignment[game_key] = None
            continue
        best = max(free, key=lambda slug: _score(templates[slug], topology, dealt, cfg))
        assignment[game_key] = best
        dealt[templates[best]["topology"]["primary"]] += 1
    return assignment
