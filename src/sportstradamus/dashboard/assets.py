"""Image loaders for the dashboard's optional art — the scar mechanism.

Streamlit serves no arbitrary static files, so every image here travels to the browser as
a base64 data URI. All of it is dressing: ambient art falls back to a token gradient, a
headshot falls back to the card's initials disc, and a constellation template with no art
draws its outline alone, so a box holding none of it renders exactly as it did before any
of it landed.

Ambient art is optional dressing over a token gradient: a slot with no file renders its
caller's fallback gradient byte-identical to today. A slot that names a real file on
disk renders the image, layered under a solid overlay of the surface color so it never
shows above the slot's manifest opacity. The slot's ``placement`` picks the geometry:
heroes crop to cover their box; a card deck scales the image to the column width and
tiles it downward so consecutive cards can show consecutive slices (``surfaces/tonight.py``
sets the per-card offsets in the browser). Files wider than the embed ceiling are
downscaled at import, so the owner can drop originals in and the page never ships them.
The manifest (``data/assets/ambient/ambient_manifest.json``) is the owner's tuning
surface, not a build artifact — same mtime-cached, validate-on-load contract as
``constellation_shapes.py``. Licensing is the owner's call, made before a file lands;
``attribution`` / ``source_url`` are free notes the loader never reads.

Headshots are the gitignored per-box cache ``fetch headshots`` fills, read through the
index parquet beside it. They are already cropped and sized for the disc, so ``_data_uri``
embeds them whole.

Constellation art is the faint drawing a dealt template shows beneath its stars: one
committed layer per template under ``data/assets/constellations/``, named by the shape
catalog's ``image`` key, with its provenance in the ``manifest.json`` beside it — which is
also where the Games page's credit line for attribution-licensed art comes from.
"""

from __future__ import annotations

import base64
import functools
import importlib.resources as pkg_resources
import io
import json
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
from PIL import Image, ImageOps

from sportstradamus import data
from sportstradamus.dashboard.components.constellation_shapes import ART_DIR
from sportstradamus.helpers.io import read_parquet_safe
from sportstradamus.helpers.text import remove_accents

MANIFEST_PATH = Path(
    str(pkg_resources.files(data) / "assets" / "ambient" / "ambient_manifest.json")
)

HEADSHOT_DIR = Path(str(pkg_resources.files(data) / "assets" / "headshots"))
HEADSHOT_INDEX_PATH = HEADSHOT_DIR / "index.parquet"

# DESIGN.md §2 secondaryBackgroundColor #1A1D24, as r,g,b for the overlay below — the
# same surface tone every wired hero/card background already ends its gradient on.
_SURFACE_RGB = "26,29,36"

# DESIGN.md §3 ambient opacity ceiling — static ambient art never exceeds this.
_OPACITY_CEILING = 0.20

# Widest copy the page ever needs: the Streamlit content column stays under 1600 CSS px
# even at 2× pixel density, while the owner's originals run 6000–7300 px (the nebula
# source is 18 MB). Wider files are downscaled at import so the CSS never embeds one.
_EMBED_MAX_WIDTH = 1600

# WebP quality for the downscaled copy — clean under a ≥ 0.80 surface overlay at about a
# third of the JPEG bytes.
_EMBED_QUALITY = 80

# CSS position/size/repeat per manifest placement (see the module docstring).
_PLACEMENT_GEOMETRY = {
    "hero-background": "center/cover no-repeat",
    "card-background": "0 0/100% auto repeat-y",
}

_REQUIRED_SLOT_KEYS = frozenset({"file", "opacity", "placement"})

_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

# Where the Games credit links each attribution licence the constellation art carries; a
# licence missing here fails loud at render rather than shipping a credit with no link.
_LICENCE_URLS = {"CC BY 3.0": "https://creativecommons.org/licenses/by/3.0/"}


def _validate(manifest: dict) -> dict:
    """Raise on a manifest whose shape or values a hand-edit could break."""
    if manifest.get("version") != 1:
        raise ValueError(f"ambient manifest: unsupported version {manifest.get('version')!r}")
    for slot, entry in manifest["slots"].items():
        missing = _REQUIRED_SLOT_KEYS - entry.keys()
        if missing:
            raise ValueError(f"ambient manifest slot {slot!r}: missing keys {sorted(missing)}")
        if not 0 <= entry["opacity"] <= _OPACITY_CEILING:
            raise ValueError(
                f"ambient manifest slot {slot!r}: opacity {entry['opacity']} exceeds the "
                f"DESIGN.md §3 ceiling of {_OPACITY_CEILING}"
            )
        if entry["placement"] not in _PLACEMENT_GEOMETRY:
            raise ValueError(
                f"ambient manifest slot {slot!r}: unknown placement {entry['placement']!r} "
                f"(one of {sorted(_PLACEMENT_GEOMETRY)})"
            )
    return manifest


@functools.lru_cache(maxsize=8)
def _load(path: Path, mtime_ns: int) -> dict:
    """Parse + validate the manifest at ``path``.

    ``mtime_ns`` is not read in the body — it is half the cache key, so an owner edit
    invalidates the entry (same contract as ``constellation_shapes.py:_load``).
    """
    return _validate(json.loads(path.read_text(encoding="utf-8")))


def _manifest() -> dict:
    return _load(MANIFEST_PATH, MANIFEST_PATH.stat().st_mtime_ns)


# One entry per distinct image. Ambient needs three; a constellation figure can show
# sixty faces at once and re-embeds them on every lens toggle, so the ceiling tracks that.
@functools.lru_cache(maxsize=512)
def _data_uri(path: Path, mtime_ns: int) -> str:
    """Base64 data URI for ``path``; a file wider than the ceiling is downscaled to WebP.

    ``mtime_ns`` is cache-key only, as in ``_load`` — the decode + resize runs once per
    process per file, and every slot that names the same file shares the result.
    """
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        if image.width <= _EMBED_MAX_WIDTH:
            payload, mime = path.read_bytes(), _MIME_BY_SUFFIX[path.suffix.lower()]
        else:
            image.thumbnail((_EMBED_MAX_WIDTH, image.height))
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, "WEBP", quality=_EMBED_QUALITY)
            payload, mime = buffer.getvalue(), "image/webp"
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


def ambient_css(slot: str, fallback_gradient: str) -> str:
    """CSS ``background`` value for ``slot``.

    Returns ``fallback_gradient`` unchanged unless the manifest slot names a ``file``
    that exists under ``data/assets/ambient/``. A present file is embedded as a base64
    data URI (Streamlit serves no arbitrary static files) under a solid surface-color
    overlay, so the image never shows above its manifest ``opacity``, positioned per
    the slot's ``placement``.

    Args:
        slot: Key into ``ambient_manifest.json``'s ``"slots"`` object.
        fallback_gradient: The token-gradient CSS this slot renders when empty.

    Returns:
        A CSS ``background`` property value.
    """
    entry = _manifest()["slots"][slot]
    if not entry["file"]:
        return fallback_gradient
    image_path = MANIFEST_PATH.parent / entry["file"]
    if not image_path.exists():
        return fallback_gradient
    overlay_alpha = round(1 - entry["opacity"], 2)
    overlay = f"rgba({_SURFACE_RGB},{overlay_alpha})"
    uri = _data_uri(image_path, image_path.stat().st_mtime_ns)
    return (
        f"linear-gradient({overlay},{overlay}),url({uri}) {_PLACEMENT_GEOMETRY[entry['placement']]}"
    )


@functools.lru_cache(maxsize=1)
def _headshot_rows(path: Path, mtime_ns: int) -> dict[tuple[str, str], list[dict]]:
    """Cache index grouped by ``(league, name)``; ``mtime_ns`` is cache-key only, as in ``_load``.

    Every row is grouped, misses included, so a name two players share stays visibly
    ambiguous instead of collapsing onto whichever of them has a photo.
    """
    rows: dict[tuple[str, str], list[dict]] = {}
    for row in read_parquet_safe(path).to_dict("records"):
        rows.setdefault((row["league"], row["name"]), []).append(row)
    return rows


def _headshot_index() -> dict[tuple[str, str], list[dict]]:
    mtime_ns = HEADSHOT_INDEX_PATH.stat().st_mtime_ns if HEADSHOT_INDEX_PATH.is_file() else 0
    return _headshot_rows(HEADSHOT_INDEX_PATH, mtime_ns)


def headshot_uris(pool: pd.DataFrame) -> dict[str, str]:
    """Player name → a data URI for their cached headshot, one entry per distinct face.

    Keyed by the display name the constellation card already carries in its customdata, so
    the frontend reads it with no new field and a player's many legs share one embed. A
    player this box has no file for is simply absent and the card draws its initials disc —
    which is also every player on a box that has never run ``fetch headshots``.

    Reads each row's own ``League``: the "look wider" lens puts other games' stars on the
    map and they need not share the focus game's league. ``Team`` breaks a tie between two
    players the index knows under one name, and a name still ambiguous after that resolves
    to nothing at all — initials on both beats the wrong face on one.
    """
    index = _headshot_index()
    files: dict[str, set[str]] = {}
    for offer in pool.to_dict("records"):
        rows = index.get((offer["League"], remove_accents(offer["Player"])), [])
        if len(rows) > 1:
            rows = [row for row in rows if row["team"] == offer["Team"]]
        if len(rows) == 1 and rows[0]["file"]:
            files.setdefault(offer["Player"], set()).add(rows[0]["file"])

    uris: dict[str, str] = {}
    for player, candidates in files.items():
        if len(candidates) != 1:
            continue
        path = HEADSHOT_DIR / candidates.pop()
        uris[player] = _data_uri(path, path.stat().st_mtime_ns)
    return uris


def constellation_layer(file: str) -> tuple[str, float, float]:
    """A constellation art layer as ``(data URI, width, height)``, each side a share of the longer.

    The shares place the layer in its template's [-1, 1]² box with its own aspect kept: the
    longer side spans the box and the shorter one is centred on it.
    """
    path = ART_DIR / file
    with Image.open(path) as layer:
        width, height = layer.size
    longer = max(width, height)
    return _data_uri(path, path.stat().st_mtime_ns), width / longer, height / longer


def constellation_credit() -> str:
    """Markdown credit for the constellation art whose licence asks for one; ``""`` if none does.

    The CC BY family asks for a credit and CC0 / public-domain art does not. Every layer is a
    recoloured, blurred adaptation of its source, so the line says so, then names each
    artist with the site the art came from and a link to the licence.
    """
    manifest = json.loads((ART_DIR / "manifest.json").read_text(encoding="utf-8"))
    artists: dict[tuple[str, str], set[str]] = {}
    for row in manifest["templates"].values():
        if row["licence"].startswith("CC BY"):
            site = urlsplit(row["source_url"]).netloc
            artists.setdefault((site, row["licence"]), set()).add(row["artist"])
    credits = [
        f"works by {', '.join(sorted(names))} ([{site}](https://{site}), "
        f"[{licence}]({_LICENCE_URLS[licence]}))"
        for (site, licence), names in sorted(artists.items())
    ]
    return f"Constellation art adapted from {'; '.join(credits)}." if credits else ""
